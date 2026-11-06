"""Workbook-backed structured timetable extraction and retrieval.

The raw XLSX is the sole data source. This module uses Python's standard
library so it does not require an Excel package or alter the workbook.
"""
from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKBOOK_PATH = PROJECT_ROOT / "data/raw/aiml/timetable/Master TT-26-27-ODD Sem.xlsx"
SUPPLEMENTARY_MARKDOWN_PATH = (
    PROJECT_ROOT
    / "data/processed/aiml/timetable/CampusGPT_FINAL_2026-27_ODD_Sem_Timetable_KB.md"
)
NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
DAY_RE = re.compile(r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)$", re.I)
TIME_RE = re.compile(r"^\s*\d{1,2}:\d{2}\s*[-–—]\s*\d{1,2}:\d{2}\s*$")
GROUP_RE = re.compile(r"\b(?:A[1-9]|B[1-9]|C[1-9])\b", re.I)
CODE_RE = re.compile(r"\(([A-Z][A-Z0-9]{1,7})\)")
ROOM_RE = re.compile(r"\b(?:\d{3}(?:\.0)?|[A-Z]-?\d{2,3}|[A-Z]\d{1,3})\b", re.I)
MARKDOWN_SLOT_RE = re.compile(r"^\s*-\s*\[([^#\]]+)#(\d+)\].*?\|\s*Record type:\s*timetable slot\s*\|", re.I)
MARKDOWN_CELL_RE = re.compile(
    r'(?<![A-Z0-9])(?P<cell>\$?[A-Z]{1,3}\$?\d+)\s*=\s*'
    r'(?:(?:"(?P<quoted>(?:[^\"]|\"\")*)")|(?P<plain>[^;]+))'
)
CLOCK_RE = re.compile(r"\b\d{1,2}:\d{2}\b")


def normalize(value: str) -> str:
    value = value.casefold().replace("&", " and ")
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text, normalized_phrase = normalize(text), normalize(phrase)
    return bool(normalized_phrase and re.search(rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)", normalized_text))


def _column_number(ref: str) -> int:
    number = 0
    for char in re.match(r"[A-Z]+", ref.upper()).group():
        number = number * 26 + ord(char) - 64
    return number


def _column_name(number: int) -> str:
    name = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        name = chr(65 + remainder) + name
    return name


class _WorksheetRows(list):
    """Existing row dictionaries plus their original XLSX row/cell references."""

    def __init__(self) -> None:
        super().__init__()
        self.row_numbers: list[Optional[int]] = []
        self.cell_addresses: list[dict[int, str]] = []


@lru_cache(maxsize=4)
def load_supplementary_markdown(path: Optional[Path] = None) -> str:
    """Read the timetable companion once per configured path.

    This loader deliberately does not convert Markdown into schedule records;
    the structured workbook index remains a separate data source.
    """
    markdown_path = Path(path) if path is not None else SUPPLEMENTARY_MARKDOWN_PATH
    try:
        return markdown_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Supplementary timetable Markdown is missing: {markdown_path}"
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(
            f"Unable to read supplementary timetable Markdown at "
            f"{markdown_path}: {exc}"
        ) from exc


def _evidence_value(value: str) -> str:
    """Normalize Markdown's serialized cell values for exact fact matching."""
    value = value.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    value = value.replace("\\u00a0", " ").replace("\xa0", " ")
    value = value.replace('""', '"')
    return " ".join(value.casefold().split())


def _markdown_field(line: str, name: str) -> str:
    match = re.search(r"\|\s*" + re.escape(name) + r":\s*([^|]+)", line, re.I)
    return match.group(1).strip() if match else ""


@lru_cache(maxsize=4)
def _supplementary_slot_annotations(path: Optional[Path] = None) -> tuple[dict, ...]:
    """Parse source references from the cached companion without making sessions."""
    markdown = load_supplementary_markdown(path)
    annotations = []
    for line in markdown.splitlines():
        header = MARKDOWN_SLOT_RE.match(line)
        if not header:
            continue
        source_marker = re.search(r"\|\s*Source cells:\s*(.*)$", line, re.I)
        refs = []
        seen_refs = set()
        source_text = source_marker.group(1) if source_marker else ""
        for cell_match in MARKDOWN_CELL_RE.finditer(source_text):
            cell = cell_match.group("cell").replace("$", "").upper()
            value = cell_match.group("quoted")
            if value is None:
                value = cell_match.group("plain").strip()
            ref_key = (cell, _evidence_value(value))
            if ref_key in seen_refs:
                continue
            seen_refs.add(ref_key)
            refs.append({"cell": cell, "value": value})
        annotations.append({
            "record_id": f"{header.group(1).strip()}#{header.group(2)}",
            "sheet": header.group(1).strip(),
            "day": _markdown_field(line, "Day"),
            "time": _markdown_field(line, "Time"),
            "line": line.strip(),
            "references": tuple(refs),
        })
    return tuple(annotations)


def _times_overlap_slot(markdown_time: str, workbook_time: str) -> bool:
    markdown_times = CLOCK_RE.findall(markdown_time)
    workbook_times = CLOCK_RE.findall(workbook_time)
    if len(markdown_times) < 2 or len(workbook_times) < 2:
        return True
    start, end = workbook_times[:2]
    return any(
        markdown_times[index] == start and markdown_times[index + 1] == end
        for index in range(0, len(markdown_times) - 1, 2)
    )


def _attach_supplementary_evidence(
    records: list[dict], annotations: tuple[dict, ...]
) -> tuple[tuple[dict, ...], dict]:
    """Attach a unique exact-cell annotation; keep all schedule rows workbook-backed."""
    enriched = []
    matched_ids = set()
    ambiguous = []
    duplicate_annotations = []
    for record in records:
        sheet, source_cell = record.get("sheet"), record.get("source_cell")
        if not sheet or not source_cell:
            enriched.append(record)
            continue
        cell = source_cell.replace("$", "").upper()
        raw_value = _evidence_value(str(record.get("raw", "")))
        candidates = []
        for annotation in annotations:
            if annotation["sheet"] != sheet:
                continue
            if annotation["day"] and normalize(annotation["day"]) != normalize(record.get("day", "")):
                continue
            if annotation["time"] and not _times_overlap_slot(annotation["time"], record.get("time", "")):
                continue
            if any(ref["cell"] == cell and _evidence_value(ref["value"]) == raw_value
                   for ref in annotation["references"]):
                candidates.append(annotation)
        # Repeated identical lines (apart from their sequence number) are a
        # duplicate serialization of one note; report and coalesce them.
        grouped = {}
        for annotation in candidates:
            content = re.sub(r"^\s*-\s*\[[^#\]]+#\d+\]", "- [SHEET#ID]", annotation["line"])
            grouped.setdefault(content, []).append(annotation)
        if len(grouped) > 1:
            ambiguous.append({
                "sheet": sheet, "cell": cell, "workbook_record": record.get("raw"),
                "markdown_record_ids": [item["record_id"] for item in candidates],
            })
            enriched.append(record)
            continue
        if not grouped:
            enriched.append(record)
            continue
        matches = next(iter(grouped.values()))
        if len(matches) > 1:
            duplicate_annotations.extend(item["record_id"] for item in matches[1:])
        chosen = matches[0]
        matched_ids.update(item["record_id"] for item in matches)
        item = dict(record)
        item["supplementary_evidence"] = ({
            "record_id": chosen["record_id"],
            "source_cell": cell,
            "day": chosen["day"],
            "time": chosen["time"],
            "text": chosen["line"],
        },)
        enriched.append(item)
    unmatched = [annotation for annotation in annotations if annotation["record_id"] not in matched_ids]
    diagnostics = {
        "unmatched_annotations": tuple(unmatched),
        "ambiguous_matches": tuple(ambiguous),
        "duplicate_annotations": tuple(duplicate_annotations),
        "parsed_slot_annotations": len(annotations),
        "matched_annotation_count": len(matched_ids),
    }
    return tuple(enriched), diagnostics


def _workbook_sheets(path: Path) -> dict[str, _WorksheetRows]:
    """Read cell text from XLSX XML, including shared and inline strings."""
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in item.findall(".//m:t", NS))
                      for item in root.findall("m:si", NS)]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relroot = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rels = {item.attrib["Id"]: item.attrib["Target"] for item in relroot}
        result: dict[str, _WorksheetRows] = {}
        for sheet in workbook.findall("m:sheets/m:sheet", NS):
            # Only explicitly visible sheets are eligible for student-facing
            # timetable records. XLSX omits state for visible sheets.
            if sheet.attrib.get("state", "visible") != "visible":
                continue
            target = rels[sheet.attrib[f"{{{NS['r']}}}id"]].lstrip("/")
            target = target if target.startswith("xl/") else f"xl/{target}"
            sheet_root = ET.fromstring(archive.read(target))
            rows = _WorksheetRows()
            for row in sheet_root.findall(".//m:sheetData/m:row", NS):
                cells: dict[int, str] = {}
                cell_addresses: dict[int, str] = {}
                for cell in row.findall("m:c", NS):
                    cell_ref = cell.attrib.get("r", "")
                    column = _column_number(cell_ref)
                    kind = cell.attrib.get("t")
                    value = cell.find("m:v", NS)
                    if kind == "inlineStr":
                        text = "".join(t.text or "" for t in cell.findall(".//m:t", NS))
                    elif value is None:
                        text = ""
                    elif kind == "s":
                        text = shared[int(value.text)]
                    else:
                        text = value.text or ""
                    text = text.replace("\xa0", " ").strip()
                    if text:
                        cells[column] = text
                        if re.fullmatch(r"[A-Z]{1,3}\d+", cell_ref):
                            cell_addresses[column] = cell_ref
                rows.append(cells)
                row_number = row.attrib.get("r", "")
                rows.row_numbers.append(
                    int(row_number) if row_number.isdigit() else None
                )
                rows.cell_addresses.append(cell_addresses)
            result[sheet.attrib["name"]] = rows
        return result


@dataclass(frozen=True)
class TimetableIndex:
    records: tuple[dict, ...]
    faculty_aliases: dict[str, dict]
    subject_aliases: dict[str, dict]
    subject_faculty: dict[str, tuple[dict, ...]]
    lab_aliases: dict[str, dict]
    source: str
    supplementary_diagnostics: dict = field(default_factory=dict)
    faculty_alias_conflicts: dict[str, tuple[dict, ...]] = field(default_factory=dict)

    def resolve(self, value: str, mapping: dict[str, dict]) -> Optional[dict]:
        return mapping.get(normalize(value))

    def search(self, query: str, filters: Optional[dict] = None) -> list[dict]:
        filters = filters or {}
        found_faculty = self.resolve(filters.get("faculty", ""), self.faculty_aliases) if filters.get("faculty") else None
        found_subject = self.resolve(filters.get("subject", ""), self.subject_aliases) if filters.get("subject") else None
        found_lab = self.resolve(filters.get("lab", ""), self.lab_aliases) if filters.get("lab") else None
        # An explicit but unknown faculty filter must never degrade to an
        # unfiltered query (which would return the entire timetable).
        if filters.get("faculty") and found_faculty is None:
            return []
        output = []
        for record in self.records:
            if filters.get("day") and normalize(filters["day"]) not in normalize(record["day"]):
                continue
            if filters.get("time_range"):
                requested = filters["time_range"]
                record_times = re.findall(r"\d{1,2}:\d{2}", record["time"])
                if len(record_times) < 2:
                    continue
                request_start, request_end = (_minute_value(value) for value in requested)
                record_start, record_end = (_minute_value(value) for value in record_times[:2])
                if record_start < request_start or record_end > request_end:
                    continue
            elif filters.get("time") and normalize(filters["time"]) not in normalize(record["time"]):
                continue
            if filters.get("sheet") and normalize(filters["sheet"]) != normalize(record["sheet"]):
                continue
            if filters.get("batch") and normalize(filters["batch"]) not in normalize(record["batch"]):
                continue
            if filters.get("academic_year") and normalize(filters["academic_year"]) != normalize(record.get("academic_year", "")):
                continue
            if filters.get("academic_term") and normalize(filters["academic_term"]) != normalize(record.get("academic_term", "")):
                continue
            if filters.get("semester") and normalize(filters["semester"]) != normalize(record["semester"]):
                continue
            if filters.get("section") and normalize(filters["section"]) not in normalize(record["section"]):
                continue
            if filters.get("group") and normalize(filters["group"]) not in normalize(record["group"]):
                continue
            if filters.get("room") and not _contains_phrase(record["room"], filters["room"]):
                continue
            if found_faculty:
                canonical_name = normalize(re.sub(r"^(?:dr|mr|mrs|ms|prof)\.?\s+", "", found_faculty["name"], flags=re.I))
                names_match = any(
                    normalize(re.sub(r"^(?:dr|mr|mrs|ms|prof)\.?\s+", "", name, flags=re.I)) == canonical_name
                    for name in record["faculty"]
                )
                if found_faculty["code"].upper() not in {code.upper() for code in record["faculty_codes"]} and not names_match:
                    continue
            if found_subject and not _contains_phrase(record["subject"] + " " + record["raw"], found_subject["code"]):
                if not _contains_phrase(record["subject"] + " " + record["raw"], found_subject["name"]):
                    continue
            if found_lab and not _contains_phrase(record["raw"] + " " + record["lab"], found_lab["name"]):
                if not _contains_phrase(record["raw"] + " " + record["lab"], found_lab["code"]):
                    continue
            if found_lab and not re.search(r"\blabs?\b", record["raw"] + " " + record["lab"], re.I):
                continue
            if filters.get("query_terms"):
                haystack = normalize(" ".join(str(record[k]) for k in ("raw", "subject", "lab", "room", "group")))
                if normalize(filters["query_terms"]) not in haystack:
                    continue
            output.append(record)
        return output


def _add_alias(
    mapping: dict[str, dict], code: str, name: str, sheet: str,
    conflicts: Optional[dict[str, list[dict]]] = None,
) -> None:
    entry = {"code": code.strip(), "name": name.strip(), "sheet": sheet}
    short_name = re.sub(r"^(?:dr|mr|mrs|ms|prof)\.?\s+", "", name.strip(), flags=re.I)
    for alias in (code, name, short_name):
        key = normalize(alias)
        if not key:
            continue
        previous = mapping.get(key)
        if previous and (
            normalize(previous["code"]) != normalize(entry["code"])
            or normalize(previous["name"]) != normalize(entry["name"])
        ) and conflicts is not None:
            candidates = conflicts.setdefault(key, [])
            for candidate in (previous, entry):
                if not any(
                    normalize(item["code"]) == normalize(candidate["code"])
                    and normalize(item["name"]) == normalize(candidate["name"])
                    for item in candidates
                ):
                    candidates.append(candidate)
        mapping[key] = entry


def _minute_value(value: str) -> int:
    hour, minute = (int(part) for part in value.split(":"))
    return hour * 60 + minute


def _make_mappings(sheets: dict[str, list[dict[int, str]]]) -> tuple[dict, dict, dict, dict, dict]:
    faculty: dict[str, dict] = {}
    faculty_conflicts: dict[str, list[dict]] = {}
    subjects: dict[str, dict] = {}
    labs: dict[str, dict] = {}
    subject_faculty: dict[str, list[dict]] = {}
    for sheet, rows in sheets.items():
        if sheet == "Classroom Lab Nomenclature":
            for row in rows:
                if 1 in row and 2 in row and 3 in row and normalize(row[1]) != "lab nomenclature":
                    _add_alias(labs, row[3], row[2], sheet)
            continue
        header_index = next((i for i, row in enumerate(rows)
                             if any(normalize(value) == "subjects" for value in row.values())), None)
        if header_index is None:
            continue
        header = rows[header_index]
        subject_col = next((col for col, value in header.items() if normalize(value) == "subjects"), None)
        faculty_cols = [col for col, value in header.items() if "faculty" in normalize(value)]
        subject_code_cols = [col for col, value in header.items() if "subject code" in normalize(value)]
        for row in rows[header_index + 1:]:
            name = row.get(subject_col, "") if subject_col else ""
            match = re.search(r"^(.+?)\s*\(([A-Z][A-Z0-9]{1,7})\)\s*$", name)
            subject_name = match.group(1).strip() if match else name
            subject_code = match.group(2) if match else ""
            if match and len(match.group(1).split()) >= 2:
                _add_alias(subjects, match.group(2), match.group(1), sheet)
            for code_col in subject_code_cols:
                code = row.get(code_col, "")
                if name and code and re.fullmatch(r"[A-Za-z][A-Za-z0-9-]{1,10}", code):
                    _add_alias(subjects, code, name, sheet)
                    subject_code = code
                    subject_name = name
            mapped_faculty = []
            for faculty_col in faculty_cols:
                value = row.get(faculty_col, "")
                for match in CODE_RE.finditer(value):
                    prefix = value[:match.start()]
                    # Each name belongs to the nearest preceding delimiter,
                    # allowing multiple mapped faculty names in one cell.
                    name_part = re.split(r"[,;\n]|\s+[/&]\s+|\)\s*&\s*", prefix)[-1].strip(" ,;/&")
                    if name_part and len(name_part.split()) >= 2:
                        _add_alias(faculty, match.group(1), name_part, sheet, faculty_conflicts)
                        mapped_faculty.append({"code": match.group(1), "name": name_part, "sheet": sheet})
            if subject_name and mapped_faculty:
                keys = [normalize(subject_name)]
                if subject_code:
                    keys.append(normalize(subject_code))
                for key in keys:
                    instructors = subject_faculty.setdefault(key, [])
                    for instructor in mapped_faculty:
                        if not any(normalize(item["name"]) == normalize(instructor["name"])
                                   and item["sheet"] == sheet for item in instructors):
                            instructors.append(instructor)
    return (
        faculty, subjects, labs,
        {key: tuple(value) for key, value in subject_faculty.items()},
        {key: tuple(value) for key, value in faculty_conflicts.items()},
    )


def _cell_kind(value: str, faculty_codes: set[str]) -> str:
    clean = value.strip()
    if (re.fullmatch(r"\d{3}(?:\.0)?", clean) or re.fullmatch(r"\d{3}/\d{3}", clean)
            or re.match(r"^\d{3}\s*/\s*(?:[A-Z][\w -]*Lab|\d{3})$", clean, re.I)):
        return "room"
    tokens = re.findall(r"(?<![A-Za-z])([A-Z]{2,6})(?![A-Za-z])", clean)
    if clean.upper() in faculty_codes or (tokens and all(t in faculty_codes for t in tokens) and len(clean) < 24):
        return "faculty"
    return "activity"


def load_timetable(
    path: Path = WORKBOOK_PATH,
    supplementary_markdown_path: Optional[Path] = None,
) -> TimetableIndex:
    sheets = _workbook_sheets(path)
    session_match = re.search(r"(?<!\d)(20\d{2}|\d{2})\s*[-–—/]\s*(\d{2,4})(?!\d)", path.stem)
    if session_match:
        start, end = session_match.groups()
        start = start if len(start) == 4 else f"20{start}"
        academic_year = f"{start}-{end[-2:]}"
    else:
        academic_year = ""
    term_match = re.search(r"\b(odd|even)\b", path.stem, re.I)
    academic_term = term_match.group(1).lower() if term_match else ""
    faculty, subjects, labs, subject_faculty, faculty_conflicts = _make_mappings(sheets)
    faculty_codes = {entry["code"].upper() for entry in faculty.values()}
    records: list[dict] = []
    for sheet, rows in sheets.items():
        if sheet == "Classroom Lab Nomenclature":
            continue
        grid_specs = []
        for header_index, header in enumerate(rows):
            day_columns = [col for col, value in header.items() if normalize(value) == "day time"]
            for day_column in day_columns:
                time_columns = []
                for column in range(day_column + 1, max(header, default=day_column) + 1):
                    value = header.get(column, "")
                    if TIME_RE.match(value):
                        time_columns.append((column, value.strip()))
                    elif time_columns:
                        break
                if time_columns:
                    grid_specs.append((header_index, day_column, time_columns))
        if not grid_specs:
            continue
        cohort = rows[2] if len(rows) > 2 else {}
        title = next((v for v in cohort.values() if v), sheet)
        batch_match = re.search(r"\b20\d{2}\s*[-–—]\s*\d{2,4}\b", title)
        batch = batch_match.group(0) if batch_match else ""
        semester_match = re.search(r"\bSem(?:ester)?[- ]*(I{1,3}|IV|V|VI|VII|VIII|[1-8])\b", title, re.I)
        semester = semester_match.group(0) if semester_match else ""
        section_match = re.search(r"\b([ABC])\b", title)
        section = section_match.group(1) if section_match else ""
        # Treat each Day/Time block independently. Some workbook sheets put
        # a second individualized timetable to the right of the main grid.
        for grid_header, day_column, time_columns in grid_specs:
            day = ""
            for row_index, row in enumerate(rows):
                day_value = row.get(day_column, "").strip()
                # Workbook schedule grids may use a merged Saturday label;
                # without an explicit boundary, that day leaks into the
                # subsequent course legend and signoff rows. The footer marker
                # is outside the schedule grid and ends this grid's scan.
                if re.match(r"^class\s+teacher\s*:", day_value, re.I):
                    break
                if DAY_RE.fullmatch(day_value):
                    day = day_value.title()
                elif re.search(r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", day_value, re.I):
                    day = day_value
                if not day or row_index <= grid_header:
                    continue
                for column, slot in time_columns:
                    raw = row.get(column, "").strip()
                    if not raw or normalize(raw) in {"lunch break", "day time"}:
                        continue
                    kind = _cell_kind(raw, faculty_codes)
                    # A standalone faculty/room row enriches the slot's class
                    # rows and is not emitted as an invented independent class.
                    if kind != "activity":
                        continue
                    activity = raw
                    group_match = GROUP_RE.search(activity)
                    group = group_match.group(0).upper() if group_match else ""
                    room_values, faculty_values = [], []
                    for nearby in rows[max(grid_header + 1, row_index - 1):min(len(rows), row_index + 3)]:
                        candidate = nearby.get(column, "").strip()
                        if not candidate or candidate == raw:
                            continue
                        candidate_kind = _cell_kind(candidate, faculty_codes)
                        if candidate_kind == "room" and candidate not in room_values:
                            room_values.append(candidate)
                        elif candidate_kind == "faculty" and candidate not in faculty_values:
                            faculty_values.append(candidate)
                    codes = re.findall(r"(?<![A-Za-z])([A-Z]{2,6})(?![A-Za-z])", activity)
                    codes.extend(re.findall(r"[A-Z]{2,6}", " ".join(faculty_values)))
                    mapped_faculty_codes = list(dict.fromkeys(
                        faculty[normalize(code)]["code"] for code in codes if normalize(code) in faculty
                    ))
                    resolved_faculty = []
                    for code in mapped_faculty_codes:
                        if normalize(code) in faculty_conflicts:
                            continue
                        found = faculty.get(normalize(code))
                        if found and found["name"] not in resolved_faculty:
                            resolved_faculty.append(found["name"])
                    subject_code = next((code for code in re.findall(r"[A-Z][A-Z0-9]{1,7}", activity)
                                         if normalize(code) in subjects), "")
                    subject = subjects.get(normalize(subject_code), {}).get("name", activity)
                    parenthetical_labs = re.findall(r"\(([^()]*(?:Lab|Laboratory)[^()]*)\)", activity, re.I)
                    if parenthetical_labs:
                        lab = parenthetical_labs[-1].strip()
                    else:
                        lab_match = re.search(r"\b([A-Za-z0-9&-]+(?:\s+[A-Za-z0-9&-]+){0,4}\s+Labs?)\b", activity, re.I)
                        lab = lab_match.group(1).strip() if lab_match else ""
                    # This is the cell containing the activity text itself.
                    # Day labels, time headers, nearby faculty/room cells and
                    # merged-range anchors are not substituted for it.
                    source_cell = rows.cell_addresses[row_index].get(column)
                    source_match = re.fullmatch(r"[A-Z]{1,3}(\d+)", source_cell or "")
                    actual_row = (
                        int(source_match.group(1))
                        if source_match
                        else rows.row_numbers[row_index]
                    )
                    records.append({
                        "sheet": sheet, "cohort": title, "batch": batch, "semester": semester, "section": section,
                        "day": day, "time": slot, "subject": subject, "subject_code": subject_code,
                        "faculty": resolved_faculty, "faculty_codes": mapped_faculty_codes,
                        "room": "; ".join(room_values), "lab": lab, "group": group,
                        "raw": raw, "source": path.relative_to(PROJECT_ROOT).as_posix(),
                        "cell_row": actual_row, "cell_column": column,
                        "cell_row_index": row_index + 1,
                        "source_cell": source_cell,
                        "source_reference_status": "resolved" if source_cell else "unresolved",
                        "academic_year": academic_year,
                        "academic_term": academic_term,
                    })
    annotations = _supplementary_slot_annotations(supplementary_markdown_path)
    records, diagnostics = _attach_supplementary_evidence(records, annotations)
    return TimetableIndex(records, faculty, subjects, subject_faculty, labs,
                          path.relative_to(PROJECT_ROOT).as_posix(), diagnostics,
                          {key: tuple(value) for key, value in faculty_conflicts.items()})


def detect_timetable_intent(query: str, index: Optional[TimetableIndex] = None) -> bool:
    """Route schedule-specific questions, leaving generic RAG unchanged."""
    explicit = bool(re.search(r"\b(timetable|time\s*table|schedule|teaching|teaches)\b", query, re.I))
    day = bool(re.search(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", query, re.I))
    cohort_or_group = bool(re.search(r"\b(?:SY|TY|B\.Tech|M\.Tech)\s*[- ]?[ABC]|\b[ABC][1-9]\b", query, re.I))
    if not explicit and day and cohort_or_group:
        explicit = True
    if not explicit and re.search(r"\bclass(?:es)?\b", query, re.I) and (day or cohort_or_group or re.search(r"\b\d{1,2}:\d{2}\b", query)):
        explicit = True
    if explicit:
        return True
    if index and resolve_mapping_query(query, index):
        return True
    if re.search(r"\b(when|where|who)\b", query, re.I) and index:
        normalized_query = normalize(query)
        return any(alias in normalized_query for alias in index.subject_aliases if len(alias) >= 3) or any(
            alias in normalized_query for alias in index.lab_aliases if len(alias) >= 3
        )
    return False


def resolve_mapping_query(query: str, index: TimetableIndex) -> Optional[dict]:
    """Return a workbook-authored code/name answer for explicit mapping asks."""
    mapping = None
    kind = ""
    normalized_query = normalize(query)
    abbreviation_intent = bool(re.search(r"\b(stand for|abbreviation|acronym|full name)\b", query, re.I))
    direct_question = bool(re.search(r"\b(?:what|who)\s+is\b", query, re.I))
    schedule_context = bool(re.search(
        r"\b(?:timetable|time\s*table|schedule|class(?:es)?|teaching|teaches|slot|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|day|time)\b|"
        r"\b\d{1,2}:\d{2}\b",
        query,
        re.I,
    ))
    mapping_only_question = direct_question and not schedule_context
    for alias, entry in sorted(index.faculty_aliases.items(), key=lambda item: -len(item[0])):
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized_query):
            if abbreviation_intent or (mapping_only_question and alias == normalize(entry["code"])):
                candidates = index.faculty_alias_conflicts.get(alias, ())
                if candidates:
                    names = "; ".join(dict.fromkeys(item["name"] for item in candidates))
                    return {
                        "answer": f"The workbook lists multiple faculty names for {entry['code']}: {names}. The abbreviation is ambiguous, so I cannot identify one name.",
                        "mapping": entry, "kind": "faculty", "ambiguous": True,
                    }
                mapping, kind = entry, "faculty"
                break
    if mapping is None:
        for alias, entry in sorted(index.subject_aliases.items(), key=lambda item: -len(item[0])):
            if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized_query):
                if abbreviation_intent or (mapping_only_question and alias == normalize(entry["code"])):
                    mapping, kind = entry, "subject"
                    break
    if mapping is None or kind == "":
        return None
    short_name = re.sub(r"^(?:dr|mr|mrs|ms|prof)\.?\s+", "", mapping["name"], flags=re.I)
    if abbreviation_intent and normalize(short_name) in normalized_query:
        answer = f"{mapping['name']}'s abbreviation is {mapping['code']}."
    else:
        noun = "faculty member" if kind == "faculty" else "subject"
        answer = f"{mapping['code']} is the abbreviation for {mapping['name']} ({noun})."
    return {"answer": answer, "mapping": mapping, "kind": kind}


def resolve_subject_teacher_query(query: str, index: TimetableIndex) -> Optional[dict]:
    """Find instructors explicitly listed on the workbook's subject rows."""
    if not re.search(r"\bwho\s+(?:is\s+)?teach(?:es|ing)\b", query, re.I):
        return None
    normalized_query = normalize(query)
    subject = None
    for alias, entry in sorted(index.subject_aliases.items(), key=lambda item: -len(item[0])):
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized_query):
            subject = entry
            break
    if subject is None:
        return None
    instructors = list(index.subject_faculty.get(normalize(subject["code"]), ()))
    if not instructors:
        instructors = list(index.subject_faculty.get(normalize(subject["name"]), ()))
    names, seen = [], set()
    for instructor in instructors:
        # Accept only a code/full-name pair present in workbook faculty tables.
        mapped = index.faculty_aliases.get(normalize(instructor["code"]))
        if not mapped or normalize(instructor["code"]) in index.faculty_alias_conflicts:
            continue
        key = normalize(mapped["name"])
        if key not in seen:
            seen.add(key)
            names.append(mapped["name"])
    return {"subject": subject, "instructors": names}


def extract_filters(query: str, index: TimetableIndex) -> dict:
    filters: dict[str, str] = {}
    for alias, entry in sorted(index.faculty_aliases.items(), key=lambda item: -len(item[0])):
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalize(query)):
            filters["faculty"] = entry["code"]
            break
    for alias, entry in sorted(index.subject_aliases.items(), key=lambda item: -len(item[0])):
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalize(query)):
            filters["subject"] = entry["code"]
            break
    for day in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"):
        if re.search(rf"\b{day}\b", query, re.I):
            filters["day"] = day
            break
    time_range_match = re.search(
        r"\b(?:between\s+)?(\d{1,2}:\d{2})\s*(?:-|to|and)\s*(\d{1,2}:\d{2})\b",
        query,
        re.I,
    )
    if time_range_match:
        filters["time"] = time_range_match.group(1)
        filters["time_range"] = (time_range_match.group(1), time_range_match.group(2))
    else:
        time_match = re.search(r"\b(\d{1,2}:\d{2})\b", query, re.I)
        if time_match:
            filters["time"] = time_match.group(1)
    batch_match = re.search(r"\b(?:SY|TY|B\.Tech|M\.Tech)?\s*[- ]?([A-Z])\b", query, re.I)
    if batch_match and re.search(r"\b(?:SY|TY|section|division)\b", query, re.I):
        filters["section"] = batch_match.group(1).upper()
    for sheet in sorted({record["sheet"] for record in index.records}, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(normalize(sheet))}(?!\w)", normalize(query)):
            filters["sheet"] = sheet
            break
    academic_batch = re.search(r"\b(20\d{2})\s*[-\-–—/]\s*(20\d{2}|\d{2})\b", query)
    session_term = re.search(r"\b(odd|even)\s+(?:semester|sem)\b", query, re.I)
    academic_session = bool(session_term or re.search(r"\bacademic\s+(?:session|year)\b", query, re.I))
    if academic_batch:
        start, end = academic_batch.groups()
        normalized_year = f"{start}-{end[-2:]}"
        if academic_session:
            filters["academic_year"] = normalized_year
        else:
            filters["batch"] = normalized_year
    if session_term:
        filters["academic_term"] = session_term.group(1).lower()
    semester = re.search(r"\b(?:semester|sem)\s*[- ]*(I{1,3}|IV|V|VI|VII|VIII|[1-8])\b", query, re.I)
    if semester:
        filters["semester"] = f"Sem-{semester.group(1).upper()}"
    group = GROUP_RE.search(query)
    if group:
        filters["group"] = group.group(0).upper()
    room = re.search(r"\b(?:room\s*)?(\d{3})\b", query, re.I)
    if room:
        filters["room"] = room.group(1)
    for alias, entry in sorted(index.lab_aliases.items(), key=lambda item: -len(item[0])):
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalize(query)):
            filters["lab"] = entry["code"]
            break
    return filters


_INDEX: Optional[TimetableIndex] = None


def get_index() -> TimetableIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = load_timetable()
    return _INDEX
