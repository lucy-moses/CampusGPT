"""CampusGPT's local, metadata-first RAG answer pipeline."""
from __future__ import annotations

import os
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ollama

# `backend/main.py` imports this module with scripts/ on sys.path, while the
# project-root package form is used by other callers. Support both forms.
try:
    from scripts import reranker as rr
    from scripts import timetable as timetable_data
except ImportError:
    import reranker as rr
    import timetable as timetable_data

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("CAMPUSGPT_OLLAMA_MODEL", "qwen2.5:3b")


def _positive_int_env(name: str, default: int) -> int:
    """Read a positive integer setting without letting a bad env value fail requests."""
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


MAX_CONTEXT_CHARS = _positive_int_env("CAMPUSGPT_MAX_CONTEXT_CHARS", 7500)
BROAD_CURRICULUM_CONTEXT_CHARS = _positive_int_env("CAMPUSGPT_BROAD_CURRICULUM_CONTEXT_CHARS", 40000)
# A complete faculty directory often spans more chunks than a fact question,
# but 26k characters fits alongside the prompt in the 8k-token context below.
BROAD_FACULTY_CONTEXT_CHARS = _positive_int_env("CAMPUSGPT_BROAD_FACULTY_CONTEXT_CHARS", 26000)
NARROW_FACULTY_CONTEXT_CHARS = _positive_int_env("CAMPUSGPT_NARROW_FACULTY_CONTEXT_CHARS", 11000)
SIMPLE_GENERATION_TOKENS = _positive_int_env("CAMPUSGPT_SIMPLE_GENERATION_TOKENS", 450)
NORMAL_GENERATION_TOKENS = _positive_int_env("CAMPUSGPT_MAX_GENERATION_TOKENS", 750)
BROAD_CURRICULUM_GENERATION_TOKENS = _positive_int_env("CAMPUSGPT_BROAD_CURRICULUM_GENERATION_TOKENS", 1600)
FACULTY_GENERATION_TOKENS = _positive_int_env("CAMPUSGPT_FACULTY_GENERATION_TOKENS", 700)
STRUCTURED_GENERATION_TOKENS = _positive_int_env("CAMPUSGPT_STRUCTURED_GENERATION_TOKENS", 600)
NUM_THREAD = _positive_int_env("CAMPUSGPT_OLLAMA_NUM_THREAD", os.cpu_count() or 4)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
client = ollama.Client(host=OLLAMA_HOST)
logger = logging.getLogger(__name__)


def _generation_diagnostics(response: dict) -> dict:
    """Normalize Ollama's nanosecond counters to milliseconds when present."""
    def milliseconds(key: str) -> Optional[float]:
        value = response.get(key)
        return round(float(value) / 1_000_000, 3) if isinstance(value, (int, float)) else None

    eval_count = response.get("eval_count")
    eval_duration = response.get("eval_duration")
    tokens_per_second = None
    if isinstance(eval_count, (int, float)) and isinstance(eval_duration, (int, float)) and eval_duration > 0:
        tokens_per_second = round(float(eval_count) / (float(eval_duration) / 1_000_000_000), 3)
    return {
        "model": response.get("model") or MODEL_NAME,
        "prompt_token_count": response.get("prompt_eval_count") if isinstance(response.get("prompt_eval_count"), int) else None,
        "generated_token_count": eval_count if isinstance(eval_count, int) else None,
        "prompt_evaluation_ms": milliseconds("prompt_eval_duration"),
        "generation_duration_ms": milliseconds("eval_duration"),
        "ollama_total_duration_ms": milliseconds("total_duration"),
        "generated_tokens_per_second": tokens_per_second,
    }


def _latency_payload(total_ms: float, **values: float) -> dict:
    """All fields use ms. Planning includes routing; embedding and Chroma are disjoint;
    reranking and context fields time only their named work; llm_generation_ms is
    the synchronous Ollama client request wall time. Total covers answer_question
    end to end, including other Python overhead. Stages are not additive.
    """
    fields = {
        "query_planning_ms": 0.0,
        "query_embedding_ms": 0.0,
        "chromadb_retrieval_ms": 0.0,
        "bge_reranking_ms": 0.0,
        "context_construction_ms": 0.0,
        "llm_generation_ms": 0.0,
        "total_ms": round(total_ms, 2),
    }
    fields.update({key: round(value, 2) for key, value in values.items()})
    return fields


def _finalize_short_route(result: dict, started: float, query_planning_ms: float) -> dict:
    """Add measured route timing to clarification/no-source responses."""
    result["latency"] = _latency_payload(
        (time.perf_counter() - started) * 1000,
        query_planning_ms=query_planning_ms,
    )
    result.setdefault("generation_metrics", None)
    return result

DOCUMENT_KEYWORDS = {
    "curriculum": ("curriculum", "syllabus", "subjects", "courses", "course structure", "semester"),
    # HOD is department information, not a faculty-list request.  Keeping
    # these vocabularies separate is important when a question asks for both.
    "faculty": ("faculty", "professor", "professors", "teacher", "teachers", "staff", "teaching staff", "faculty members"),
    "infrastructure": ("lab", "labs", "laboratory", "infrastructure", "facility", "facilities"),
    "department_information": ("hod", "head of department", "department head", "who heads the department", "who is the head", "head of ", "current head", "professor and head", "professor & head", "department information", "department overview", "about the department", "department details"),
    "timetable": ("timetable", "time table", "schedule", "class timing", "monday", "tuesday", "wednesday", "thursday", "friday"),
    "handbook": ("student handbook", "handbook", "attendance", "leave allowance", "leave policy"),
    "academic_calendar": ("academic calendar", "end semester examination", "examination", "exam", "vacation", "holiday", "odd semester", "even semester"),
    # Placement must be recognized before laboratory/infrastructure intent:
    # placement posters can mention facilities or training, but those words do
    # not make a package, recruitment or internship question an infra query.
    "placement": ("placement", "placements", "recruiter", "recruitment", "package", "packages", "internship", "internships", "stipend", "ctc", "salary", "hiring", "hired", "job role", "job roles"),
    "faq": ("faq", "frequently asked questions"),
    "achievement": ("achievement", "achievements", "award", "awards", "accomplishment", "accomplishments"),
    "notice": ("notice", "circular", "announcement"),
    "admissions": ("admission", "eligibility", "apply", "application"),
}

DEPARTMENT_PATTERNS = {
    "AIML": (r"\baiml\b", r"\bai\s*[-&]?\s*ml\b", r"artificial intelligence\s*(?:and|&)\s*machine learning"),
    "RA": (r"\bra\b", r"robotics\s*(?:and|&)\s*automation"),
    "ENTC": (r"\bentc\b", r"\be\s*&\s*tc\b", r"electronics\s*(?:and|&)\s*telecommunication(?:\s+engineering)?"),
    "CS": (r"\bcs\b", r"\bcse\b", r"computer science(?:\s+and\s+engineering)?"),
    "CIVIL": (r"\bcivil\b", r"civil engineering"),
    "MECH": (r"\bmech\b", r"mechanical engineering"),
}

BATCH_EXPRESSION_RE = re.compile(
    r"(?<!\d)(?:batch\s*)?(20\d{2}|\d{2})\s*(?:-|/|\bto\b)\s*(20\d{2}|\d{2})(?!\d)",
    re.IGNORECASE,
)

# These are intentionally scoped to departments that have no corresponding
# processed source. Returning before retrieval prevents an unrelated
# department's placement or timetable material from leaking into the answer.
NO_SOURCE_DOCUMENT_TYPES = {
    "CS": {"placement", "timetable"},
    "CIVIL": {"placement", "timetable"},
    "MECH": {"placement", "timetable"},
}


@dataclass(frozen=True)
class QueryPlan:
    classification: str
    document_type: Optional[str]
    batches: tuple[str, ...]
    semester: Optional[int]
    comparison: bool
    default_batch_used: bool
    batch_selection_required: bool
    department: Optional[str]
    entc_placement_unavailable: bool


def detect_department(query: str) -> Optional[str]:
    """Identify a department from generic abbreviations and full names."""
    for department, patterns in DEPARTMENT_PATTERNS.items():
        if any(re.search(pattern, query, re.I) for pattern in patterns):
            return department
    return None


def detect_departments(query: str) -> tuple[str, ...]:
    """Return every named department for safe timetable scope checks."""
    return tuple(
        department
        for department, patterns in DEPARTMENT_PATTERNS.items()
        if any(re.search(pattern, query, re.I) for pattern in patterns)
    )


def normalize_batch(value: str) -> Optional[str]:
    value = value.replace("\u2013", "-").replace("\u2014", "-").strip()
    match = re.fullmatch(
        r"(?:batch\s*)?(20\d{2}|\d{2})\s*(?:-|/|\bto\b)\s*(20\d{2}|\d{2})",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    start, end = match.groups()
    start_year = int(start) if len(start) == 4 else 2000 + int(start)
    end_year = int(end) if len(end) == 4 else 2000 + int(end)
    if end_year != start_year + 4:
        return None
    return f"{start_year}-{end_year % 100:02d}"


def detect_batches(query: str) -> tuple[str, ...]:
    query = query.replace("\u2013", "-").replace("\u2014", "-")
    found: list[str] = []
    # Normalize the complete expression matched in the user query.  Keeping
    # this at the extraction boundary means shorthand such as "CS 23-27" is
    # carried into QueryPlan.batches exactly like a fully written batch.
    for match in BATCH_EXPRESSION_RE.finditer(query):
        batch = normalize_batch(match.group(0))
        if batch and batch not in found:
            found.append(batch)
    for start in re.findall(r"\b(20\d{2})\s+batch\b", query, re.I):
        batch = f"{start}-{(int(start) + 4) % 100:02d}"
        if batch not in found:
            found.append(batch)
    return tuple(found)


def detect_semester(query: str) -> Optional[int]:
    match = re.search(r"\b(?:semester|sem)\s*:?[ ]*([1-8])\b", query, flags=re.I)
    return int(match.group(1)) if match else None


def is_person_lookup(query: str) -> bool:
    """Recognize a department-scoped person question without naming people."""
    return bool(
        detect_department(query)
        and re.search(r"\bwho\s+(?:is|are)\s+[a-z]", query, re.I)
    )


FACULTY_TEACHING_SOURCE = "data/processed/aiml/faculty/faculty_information.md"


def is_faculty_teaching_query(query: str) -> bool:
    """Recognize questions about a person's teaching assignments, not schedules."""
    if re.search(r"\b(timetable|time\s*table|schedule|\bSY\s*[- ]?[ABC]\b|\bTY\s*[- ]?[ABC]\b|\b\d{1,2}:\d{2}\b|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", query, re.I):
        return False
    return bool(re.search(
        r"\b(?:what|which)\b.*\b(?:does|do|is|are)\b.+?\b(?:teach(?:es|ing)?|handle|instruct|associated\s+with)\b",
        query,
        re.I,
    ))


def _faculty_name_requests(query: str) -> list[str]:
    match = re.search(
        r"\b(?:what|which)\b.*?\b(?:does|do|is|are)\s+(.+?)\s+"
        r"(?:teach(?:es|ing)?|handle|instruct|associated\s+with)\b",
        query,
        re.I,
    )
    if not match:
        return []
    return [
        re.sub(r"^(?:the\s+)?(?:prof(?:essor)?|dr|mr|mrs|ms)\.?\s+", "", part.strip(), flags=re.I)
        for part in re.split(r",|\band\b|\bor\b", match.group(1), flags=re.I)
        if part.strip()
    ]


def _faculty_profile_names(source_text: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(r"(?m)^##\s+\d+\.\s+(.+?)\s*$", source_text)
    ]


def _faculty_teaching_answer(query: str, source_path: Optional[Path] = None) -> dict:
    """Answer from the faculty profile only; it does not establish courses taught."""
    path = source_path or (PROJECT_ROOT / FACULTY_TEACHING_SOURCE)
    try:
        source_text = path.read_text(encoding="utf-8")
    except OSError:
        return {
            "answer": "The faculty information source is unavailable, so I cannot verify these teaching assignments.",
            "sources": [], "query_classification": "simple", "document_type": "faculty",
            "detected_department": "AIML", "detected_batch": [], "detected_semester": None,
            "default_curriculum_batch_used": False, "semester_metadata_fallback": False,
            "candidate_k": 0, "top_k": 0,
            "context": {"context_chunks": 0, "context_characters": 0, "context_estimated_tokens": 0},
        }
    names = _faculty_profile_names(source_text)
    requested = _faculty_name_requests(query)
    normalized_names = [(name, set(re.findall(r"[a-z0-9]+", name.casefold())) - {"dr", "prof", "professor", "mr", "mrs", "ms"}) for name in names]
    lines: list[str] = []
    for person in requested:
        tokens = set(re.findall(r"[a-z0-9]+", person.casefold())) - {"dr", "prof", "professor", "mr", "mrs", "ms"}
        candidates = [name for name, name_tokens in normalized_names if tokens and tokens.issubset(name_tokens)]
        if len(candidates) == 1:
            lines.append(f"{candidates[0]}: the faculty profile does not list the subjects or courses this person teaches.")
        elif len(candidates) > 1:
            lines.append(f"'{person}' matches multiple faculty profiles ({'; '.join(candidates)}). Please specify which person you mean.")
        else:
            near = [name for name in names if person.casefold() in name.casefold() or name.casefold() in person.casefold()]
            if near:
                lines.append(f"I cannot resolve '{person}' to an exact profile name. The source includes {', '.join(near)}; please confirm whether you mean one of them.")
            else:
                lines.append(f"I could not find an exact faculty profile for '{person}' in the AIML faculty information source.")
    if not lines:
        lines.append("The AIML faculty information source does not provide teaching assignments for the requested person.")
    relative = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix() if path.resolve().is_relative_to(PROJECT_ROOT.resolve()) else path.name
    return {
        "answer": " ".join(lines),
        "sources": [{"source": relative, "document_type": "faculty"}],
        "query_classification": "simple", "document_type": "faculty",
        "detected_department": "AIML", "detected_batch": [], "detected_semester": None,
        "default_curriculum_batch_used": False, "semester_metadata_fallback": False,
        "candidate_k": 0, "top_k": 0,
        "context": {"context_chunks": 1, "context_characters": len(source_text), "context_estimated_tokens": len(source_text) // 4},
    }


def is_hod_query(query: str) -> bool:
    """Recognize requests that require an explicitly supported department head."""
    return bool(re.search(
        # Do not require the user to spell out "Engineering": common forms
        # include "head of AIML department", "head of CS", and "head of the
        # department".  The caller additionally requires a detected
        # department before it uses the HOD-only retrieval path.
        r"\b(?:hod|head\s+of(?:\s+the)?(?:\s+[a-z&]+){0,5}|department\s+head|"
        r"who\s+heads\s+(?:the\s+)?department|current\s+head|"
        r"professor\s*(?:and|&)\s*head)\b",
        query,
        re.I,
    ))


def detect_document_type(
    query: str,
    requested: Optional[str],
    detected_batches: Optional[list[str]] = None,
) -> Optional[str]:
    if requested:
        return {"academic": "academic_calendar", "student_handbook": "handbook"}.get(requested.strip().lower(), requested.strip().lower())
    lowered = query.lower()
    # Session commencement questions may say "academic session" without the
    # phrase "academic calendar". Route that narrow intent to calendar
    # evidence so the source-grounded session-start formatter can handle it.
    if re.search(r"\bacademic\s+session\b", lowered) and re.search(
        r"\b(start(?:s|ed|ing)?|begin(?:s|ning)?|commenc(?:e|es|ed|ing|ement))\b",
        lowered,
    ):
        return "academic_calendar"
    # Specific operational intents outrank generic terms. Placement is ahead
    # of infrastructure so that a query mentioning an internship lab/training
    # context cannot route to laboratory chunks.
    # Handbook must outrank generic examination terms in academic-calendar
    # matching (for example, "handbook examination rules").
    for document_type in ("handbook", "placement", "academic_calendar", "timetable", "infrastructure", "faculty", "department_information", "faq", "achievement", "notice", "admissions"):
        if any(term in lowered for term in DOCUMENT_KEYWORDS[document_type]):
            return document_type
    if detect_semester(query) is not None or (
        detected_batches if detected_batches is not None else detect_batches(query)
    ):
        return "curriculum"
    for document_type in ("curriculum",):
        terms = DOCUMENT_KEYWORDS[document_type]
        if any(term in lowered for term in terms):
            return document_type
    return None


def plan_query(query: str, document_type: Optional[str] = None, batch: Optional[str] = None) -> QueryPlan:
    # Extract once at the entry point, then use this normalized tuple for both
    # intent classification and Chroma filtering. This prevents shorthand
    # batch expressions from being classified in one path but lost in another.
    normalized_batches = detect_batches(query)
    batches = list(normalized_batches)
    supplied_batch = normalize_batch(batch) if batch else None
    if supplied_batch and supplied_batch not in batches:
        batches.insert(0, supplied_batch)
    resolved_type = (
        "faculty"
        if document_type is None and is_faculty_teaching_query(query)
        else detect_document_type(query, document_type, batches)
    )
    department = detect_department(query)
    # "Who is <person> in AIML?" has no generic faculty keyword, but the
    # requested evidence is a faculty/profile record. HOD has already been
    # resolved as department information above, so it is not affected.
    if resolved_type is None and is_person_lookup(query):
        resolved_type = "faculty"
    entc_placement_unavailable = department == "ENTC" and resolved_type == "placement"
    # ENTC has no placement category. Leave document_type unset so callers
    # cannot accidentally retrieve unrelated placement documents.
    if entc_placement_unavailable:
        resolved_type = None
    comparison = bool(re.search(r"\b(compare|comparison|difference|versus|vs\.?)\b", query, re.I))
    # Curriculum sources exist for multiple admission batches. Never silently
    # select one when the question does not state a batch: subjects, credits,
    # and elective structures can differ across batches.
    default_batch_used = False
    batch_selection_required = resolved_type == "curriculum" and not batches and not comparison
    broad = resolved_type == "curriculum" or bool(re.search(r"\b(all|every|complete|full|list)\b", query, re.I))
    return QueryPlan("broad" if broad else "simple", resolved_type, tuple(batches), detect_semester(query), comparison, default_batch_used, batch_selection_required, department, entc_placement_unavailable)


def no_source_answer(plan: QueryPlan, document_type: str) -> dict:
    """Return the no-retrieval response for an absent department source."""
    labels = {"placement": "Placement information", "timetable": "A timetable"}
    return {
        "answer": f"{labels[document_type]} is not currently available in the CampusGPT knowledge base for {plan.department}.",
        "sources": [],
        "query_classification": plan.classification,
        "document_type": None,
        "detected_department": plan.department,
        "detected_batch": list(plan.batches),
        "detected_semester": plan.semester,
        "default_curriculum_batch_used": plan.default_batch_used,
        "semester_metadata_fallback": False,
        "candidate_k": 0,
        "top_k": 0,
        "context": {"context_chunks": 0, "context_characters": 0, "context_estimated_tokens": 0},
        "latency": {"query_embedding_ms": 0.0, "chromadb_retrieval_ms": 0.0, "bge_reranking_ms": 0.0, "llm_generation_ms": 0.0, "total_ms": 0.0},
    }


def no_timetable_answer(plan: QueryPlan, departments: tuple[str, ...]) -> dict:
    """Do not let the AIML workbook stand in for another department's data."""
    unavailable = tuple(department for department in departments if department != "AIML")
    result = no_source_answer(plan, "timetable")
    if "AIML" in departments:
        missing = " and ".join(unavailable)
        result["answer"] = (
            "The knowledge base contains AIML timetable data only; "
            f"no timetable data is currently available for {missing}. "
            "Specify an AIML cohort to ask about its timetable."
        )
    else:
        result["answer"] = (
            "A timetable is not currently available in the CampusGPT "
            f"knowledge base for {' and '.join(unavailable)}."
        )
    result["detected_department"] = unavailable[0] if unavailable else None
    return result


def batch_selection_answer(plan: QueryPlan) -> dict:
    """Avoid mixing batch-specific curriculum facts when batch is omitted."""
    department_label = plan.department or "the requested department"
    return {
        "answer": (
            f"Multiple curriculum batches are available for {department_label}. "
            "Please specify the batch (for example, 2023-27, 2024-28, "
            "2025-29, or 2026-30) so I can provide the correct curriculum."
        ),
        "sources": [],
        "query_classification": plan.classification,
        "document_type": plan.document_type,
        "detected_department": plan.department,
        "detected_batch": [],
        "detected_semester": plan.semester,
        "default_curriculum_batch_used": False,
        "semester_metadata_fallback": False,
        "candidate_k": 0,
        "top_k": 0,
        "retrieved_candidate_count": 0,
        "reranked_chunk_count": 0,
        "context": {"context_chunks": 0, "context_characters": 0, "context_estimated_tokens": 0},
        "latency": {"query_embedding_ms": 0.0, "chromadb_retrieval_ms": 0.0, "bge_reranking_ms": 0.0, "llm_generation_ms": 0.0, "total_ms": 0.0},
    }


def retrieval_settings(plan: QueryPlan) -> tuple[int, int]:
    # Curriculum documents are split across many sections. Broad programme or
    # subject questions therefore need a much larger candidate window and
    # reranked context than a single-fact question.
    if plan.document_type == "curriculum":
        return 30, 12
    if plan.document_type == "placement":
        return 15, 6
    if plan.document_type == "infrastructure":
        return 10, 4
    if plan.document_type == "faculty":
        # Narrow faculty questions are semantic lookups, not a directory dump.
        # Broad list requests use _retrieve_complete_faculty instead.
        return 18, 6
    if plan.document_type == "handbook":
        return 12, 5
    if plan.document_type == "academic_calendar":
        return 10, 5
    return 8, 3


def is_broad_curriculum_query(plan: QueryPlan, query: str) -> bool:
    """Identify requests for a programme-wide subject listing.

    A semester-qualified question remains a normal targeted retrieval.  A
    batch-qualified ``What subjects ...`` request must cover every indexed
    section of the selected programme rather than a semantic top-k slice.
    """
    if plan.document_type != "curriculum" or plan.semester is not None:
        return False
    narrow_fact = bool(re.search(
        r"\b(?:duration|intake|eligibility|eligible|fees?|credits?|hours?)\b",
        query,
        re.I,
    ))
    explicit_list = bool(re.search(
        r"\b(?:all|every|complete|full|entire|list)\b",
        query,
        re.I,
    ))
    return (
        bool(re.search(r"\b(subjects?|courses?|curriculum|syllabus)\b", query, re.I))
        and (not narrow_fact or explicit_list)
    )


def _deduplicate(results: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for result in results:
        key = re.sub(r"\s+", " ", result["document"].strip().lower())
        if key and key not in seen:
            seen.add(key)
            unique.append(result)
    return unique


def filter_named_placement_evidence(
    query: str, plan: QueryPlan, results: list[dict]
) -> list[dict]:
    """Keep only chunks that mention the named student for person-specific asks."""
    if plan.document_type != "placement" or plan.department:
        return results
    names = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", query)
    for name in names:
        matching = [
            item for item in results
            if name.casefold() in str(item.get("document", "")).casefold()
        ]
        if matching:
            return matching
    return results


def format_named_placement_answer(
    query: str, plan: QueryPlan, results: list[dict]
) -> Optional[str]:
    """Read all requested fields from one matching student placement row."""
    matching_results = filter_named_placement_evidence(query, plan, results)
    names = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", query)
    for name in names:
        for item in matching_results:
            for line in str(item.get("document", "")).splitlines():
                if name.casefold() not in line.casefold() or "|" not in line:
                    continue
                cells = [cell.strip().strip("*").strip() for cell in line.strip().strip("|").split("|")]
                if len(cells) < 5:
                    continue
                # Placement table columns: Page, Student, Company, Role, Stipend/Package.
                if cells[0].isdigit() and name.casefold() in cells[1].casefold():
                    company, role, compensation = cells[2:5]
                    compensation = re.sub(r"\*", "", compensation).strip()
                    article = "an" if role[:1].casefold() in {"a", "e", "i", "o", "u"} else "a"
                    return (
                        f"{cells[1]} is listed with {company} for {article} {role} role "
                        f"with {compensation}."
                    )
    return None


def is_broad_faculty_query(plan: QueryPlan, query: str) -> bool:
    """Return true only when the user explicitly asks for a faculty directory."""
    if plan.document_type != "faculty":
        return False
    lowered = re.sub(r"\s+", " ", query.lower())
    return bool(re.search(
        r"\b(?:list|show|give|provide)\s+(?:all\s+)?(?:the\s+)?faculty(?:\s+members)?\b|"
        r"\bwho\s+are\s+(?:all\s+)?(?:the\s+)?faculty(?:\s+members)?\b|"
        r"\b(?:complete|full)\s+faculty(?:\s+list)?\b",
        lowered,
    ))


def _retrieve_complete_faculty(plan: QueryPlan) -> tuple[list[dict], int, float]:
    """Retrieve every faculty chunk for the requested department."""
    started = time.perf_counter()
    candidates = rr.retrieve_all_candidates(
        document_type="faculty",
        batches=plan.batches,
        department=plan.department,
    )
    retrieval_ms = (time.perf_counter() - started) * 1000
    results = [
        {"document": document, "metadata": metadata}
        for document, metadata in zip(
            candidates["documents"], candidates["metadatas"]
        )
    ]
    results.sort(
        key=lambda item: (
            str(item["metadata"].get("source", "")),
            int(item["metadata"].get("chunk_index", 0)),
        )
    )
    return _deduplicate(results), len(results), retrieval_ms


HOD_EVIDENCE_RE = re.compile(
    r"\b(?:hod|head\s*(?:,|of)\s*(?:the\s+)?(?:department|[a-z& ]+)|"
    r"department\s+head|professor\s*(?:and|&)\s*head)\b",
    re.I,
)


def _retrieve_hod_evidence(plan: QueryPlan) -> tuple[list[dict], int, float]:
    """Use only explicit HOD evidence from the department's two source types.

    HOD designations are inconsistently stored: AIML/RA/ENTC place them in
    faculty profiles, while MECH places one in a department overview.  This
    structured metadata retrieval avoids relying on a semantic top-k choice
    and never treats an ordinary faculty designation as HOD evidence.
    """
    started = time.perf_counter()
    results: list[dict] = []
    retrieved_count = 0
    for document_type in ("department_information", "faculty"):
        candidates = rr.retrieve_all_candidates(
            document_type=document_type,
            department=plan.department,
        )
        retrieved_count += len(candidates["documents"])
        results.extend(
            {"document": document, "metadata": metadata}
            for document, metadata in zip(candidates["documents"], candidates["metadatas"])
            if HOD_EVIDENCE_RE.search(document)
        )
    results.sort(key=lambda item: (
        str(item["metadata"].get("source", "")),
        int(item["metadata"].get("chunk_index", 0)),
    ))
    return _deduplicate(results), retrieved_count, (time.perf_counter() - started) * 1000


def _retrieve(
    plan: QueryPlan,
    query: str,
    candidate_k: int,
    top_k: int,
    disable_reranker: bool = False,
) -> tuple[list[dict], int, float, float, float, bool]:
    # Keep the existing Chroma metadata value while exposing the clearer
    # public intent name `handbook` in query results.
    metadata_type = "student_handbook" if plan.document_type == "handbook" else plan.document_type
    kwargs = {"query": query, "top_k": candidate_k, "document_type": metadata_type, "batches": plan.batches, "semester": plan.semester if plan.document_type == "curriculum" else None, "department": plan.department}
    started = time.perf_counter()
    if is_broad_curriculum_query(plan, query):
        candidates = rr.retrieve_all_candidates(
            document_type=metadata_type,
            batches=plan.batches,
            department=plan.department,
        )
        retrieval_ms = (time.perf_counter() - started) * 1000
        results = [
            {"document": document, "metadata": metadata}
            for document, metadata in zip(candidates["documents"], candidates["metadatas"])
        ]
        results.sort(key=lambda item: (str(item["metadata"].get("source", "")), int(item["metadata"].get("chunk_index", 0))))
        # Every chunk in this exact department/type/batch scope is required
        # for a programme-wide listing; do not discard an apparently repeated
        # table fragment and accidentally remove a semester boundary.
        return results, len(candidates["documents"]), 0.0, retrieval_ms, 0.0, False
    candidates = rr.retrieve_candidates(**kwargs)
    # Embedding and Chroma query are timed inside retrieve_candidates so their
    # durations are disjoint and neither is counted twice here.
    query_embedding_ms = candidates.get("query_embedding_ms", 0.0)
    retrieval_ms = candidates.get("chromadb_retrieval_ms", 0.0)
    candidate_count = len(candidates["documents"])
    if disable_reranker:
        results = [
            {
                "document": document,
                "metadata": metadata,
                "chroma_distance": float(distance),
                "rerank_score": None,
                "final_score": -float(distance),
            }
            for document, metadata, distance in zip(
                candidates["documents"],
                candidates["metadatas"],
                candidates["distances"],
            )
        ]
        results.sort(key=lambda item: item["chroma_distance"])
        results = results[:top_k]
        reranking_ms = 0.0
    else:
        started = time.perf_counter()
        results = rr.rerank_candidates(query, candidates, top_k=top_k)
        reranking_ms = (time.perf_counter() - started) * 1000
    return _deduplicate(results), candidate_count, query_embedding_ms, retrieval_ms, reranking_ms, False


def _clean_curriculum_cell(value: str) -> str:
    """Preserve source wording while making converted table cells readable."""
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.I)
    value = value.replace("**", "").replace("__", "")
    return re.sub(r"\s+", " ", value).strip()


def extract_curriculum_subjects(results: list[dict]) -> list[tuple[int, str, str, Optional[str]]]:
    """Extract line-oriented or table-form course rows from curriculum Markdown.

    Converted PDFs are not consistently Markdown tables.  In particular, RA
    stores a course code, title, category, and numeric columns on separate
    lines.  Parse the stable code/title sequence directly and retain only
    Annexure A when annexures explicitly distinguish the optional Honours and
    Minor curricula.
    """
    source_paths = sorted({
        str(item["metadata"].get("source", ""))
        for item in results
    })
    subjects = []
    seen = set()
    # Converted programme structures use two identifiers for most courses:
    # a short catalogue code (``TE7168``) followed by a long course code
    # (``0701220101 Mathematics-I``).  Some PDFs put both on one physical
    # line while others put each on a separate line, so they must not be
    # treated as two unrelated courses.
    catalogue_code_re = re.compile(r"^[A-Z]{1,8}[-/]?\d{2,}[A-Z0-9-]*$", re.I)
    catalogue_prefix_re = re.compile(
        r"^(?P<code>[A-Z]{1,8}[-/]?\d{2,}[A-Z0-9-]*)(?:\s+|$)", re.I
    )
    long_course_re = re.compile(r"^(?P<code>\d{8,}[A-Z0-9-]*)(?:\s+(?P<title>.*))?$", re.I)
    nature_re = re.compile(r"\s+(?:BS|ES|PC|HS|OE|PE|PIS|PD|MC|GE)\b", re.I)
    semester_re = re.compile(r"\bsemester\s*:?\s*([1-8])\b", re.I)
    category_re = re.compile(r"^(?:generic\s+)?(?:core|elective|open elective|specialization|honours?|honors?|minor).*courses?$", re.I)
    stop_title_re = re.compile(r"^(?:total|page\s*:|siu\b|non\s*-?letter|mandatory|course\s*(?:code|title|name)|credits?)", re.I)

    for relative_source in source_paths:
        source_path = PROJECT_ROOT / relative_source
        if not relative_source or not source_path.is_file():
            continue

        semester = None
        category = "Courses"
        in_main_curriculum = True
        lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
        index = 0
        while index < len(lines):
            raw_line = lines[index]
            line = _clean_curriculum_cell(raw_line)
            if not line:
                index += 1
                continue

            annexure = re.search(r"\bannexure\s+([abc])\b", line, re.I)
            if annexure:
                in_main_curriculum = annexure.group(1).upper() == "A"

            m = semester_re.search(line)
            if m:
                semester = int(m.group(1))

            lowered = line.lower()
            compact = re.sub(r"[^a-z]", "", lowered)

            if "genericelectivecoursesgroup" in compact:
                category = "Generic Elective Courses Group"
            elif "genericcorecourses" in compact:
                category = "Generic Core Courses"
            elif "corecourses" in compact:
                category = "Core Courses"
            elif "openelective" in compact:
                category = "Open Elective Courses"
            elif "electivecourses" in compact:
                category = "Elective Courses"
            elif "specialization" in lowered:
                category = "Specialization Courses"
            elif "honours" in lowered or "honors" in lowered:
                category = "Honours Courses"
            elif "minor" in lowered:
                category = "Minor Courses"

            if semester is None or not in_main_curriculum:
                index += 1
                continue

            # Support conventional Markdown tables as well as bare PDF text.
            cells = [_clean_curriculum_cell(c) for c in raw_line.split("|")]
            is_table = raw_line.lstrip().startswith("|")
            code_cell = next(
                (cell for cell in cells if catalogue_code_re.fullmatch(re.sub(r"\s+", "", cell))),
                None,
            ) if is_table else None
            catalogue_match = catalogue_prefix_re.match(line) if not is_table else None
            if code_cell is None and catalogue_match is None:
                index += 1
                continue

            title_parts: list[str] = []
            numeric_values: list[str] = []
            explicit_nature = False

            def add_title_fragment(value: str) -> bool:
                """Add text before the source's nature-code/numeric columns.

                Returns whether the nature code was encountered, after which
                only numeric scheme values are considered for course credits.
                """
                value = _clean_curriculum_cell(value)
                long_match = long_course_re.match(value)
                if long_match:
                    value = long_match.group("title") or ""
                nature_match = nature_re.search(value)
                before_nature = value[:nature_match.start()] if nature_match else value
                if before_nature.strip():
                    title_parts.append(before_nature.strip())
                if nature_match:
                    numeric_values.extend(re.findall(r"\b\d+(?:\.\d+)?\b", value[nature_match.end():]))
                return nature_match is not None

            if is_table:
                code_index = cells.index(code_cell)
                for candidate in cells[code_index + 1:]:
                    if candidate:
                        if add_title_fragment(candidate):
                            break
            else:
                # A flattened row may be ``TE7697 0701260101 Linear Algebra
                # BS ...``.  A line-oriented row starts with only ``TE7697``
                # and continues on the next physical line.
                after_catalogue = line[catalogue_match.end():].strip()
                reached_nature = add_title_fragment(after_catalogue) if after_catalogue else False
                explicit_nature = reached_nature
                lookahead = index + 1
                while lookahead < len(lines):
                    candidate = _clean_curriculum_cell(lines[lookahead])
                    compact_candidate = re.sub(r"\s+", "", candidate)
                    if not candidate:
                        lookahead += 1
                        continue
                    if catalogue_code_re.fullmatch(compact_candidate) or semester_re.search(candidate):
                        break
                    if category_re.fullmatch(candidate) or stop_title_re.match(candidate):
                        break
                    if reached_nature:
                        if re.fullmatch(r"\d+(?:\.\d+)?", candidate):
                            numeric_values.append(candidate)
                            lookahead += 1
                            continue
                        if candidate.upper() == "GE" and numeric_values:
                            # In these converted rows GE follows the numeric
                            # scheme fields; its trailing values include the
                            # explicitly labelled credit columns.
                            explicit_nature = True
                            lookahead += 1
                            continue
                        break
                    if re.fullmatch(r"[A-Z]{1,4}", candidate):
                        # The nature field can be isolated on its own line.
                        reached_nature = candidate.upper() in {
                            "BS", "ES", "PC", "HS", "OE", "PE", "PIS", "PD", "MC", "GE"
                        }
                        # Generic-elective rows place their GE nature value
                        # after the numeric scheme/credit sequence.
                        if candidate.upper() == "GE" and numeric_values:
                            explicit_nature = True
                            lookahead += 1
                            continue
                        explicit_nature = explicit_nature or reached_nature
                        lookahead += 1
                        continue
                    if title_parts and re.fullmatch(r"\d+(?:\.\d+)?", candidate):
                        # Generic-elective rows in several converted PDFs
                        # omit the isolated nature field entirely. Once a
                        # title has been read, a numeric-only line begins its
                        # explicit scheme/credit sequence rather than a new
                        # title fragment.
                        numeric_values.append(candidate)
                        reached_nature = True
                        lookahead += 1
                        continue
                    reached_nature = add_title_fragment(candidate)
                    explicit_nature = explicit_nature or reached_nature
                    lookahead += 1

            title = re.sub(r"\s+", " ", " ".join(title_parts)).strip()
            if not title or title.lower() in {"course name", "course title", "subject", "subject name", "name of course", "course"}:
                index += 1
                continue

            key = (semester, category, title)
            if key not in seen:
                seen.add(key)
                # In the line-oriented source layout, credits are the third
                # value from the end of the published examination/credit
                # sequence. Keep them only when that explicit sequence is
                # present; otherwise do not infer a value.
                credits = numeric_values[-3] if explicit_nature and len(numeric_values) >= 3 else None
                subjects.append((semester, category, title, credits))
            index += 1

    return subjects


def extract_semester_curriculum_rows(
    results: list[dict], semester: int
) -> tuple[list[dict], set[str]]:
    """Return source-parsed course rows for one semester and other-semester IDs.

    Curriculum Markdown often contains both a short catalogue identifier and
    the official long course code. The table parser supplies semester, title,
    category, and explicitly labelled credits; this helper joins those rows to
    the long code in the same authoritative source.
    """
    source_paths = sorted({
        str(item["metadata"].get("source", ""))
        for item in results
        if item.get("metadata", {}).get("source")
    })
    code_by_source_title: dict[tuple[str, str], str] = {}
    for relative_source in source_paths:
        path = PROJECT_ROOT / relative_source
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(r"^\s*(\d{8,}[A-Z0-9-]*)\s+(.+?)\s*$", _clean_curriculum_cell(line), re.I)
            if match:
                title = re.sub(r"\s+", " ", match.group(2)).strip().casefold()
                code_by_source_title[(relative_source, title)] = match.group(1)

    rows: list[dict] = []
    other_semester_codes: set[str] = set()
    for relative_source in source_paths:
        source_results = [
            item for item in results
            if str(item.get("metadata", {}).get("source", "")) == relative_source
        ]
        source_path = PROJECT_ROOT / relative_source
        table_rows = []
        if source_path.is_file():
            in_section = False
            heading = re.compile(rf"^##\s+Semester\s*:?\s*{semester}\b", re.I)
            for source_line in source_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if re.match(r"^##\s+Semester\b", source_line, re.I):
                    in_section = bool(heading.match(source_line))
                    continue
                if not in_section:
                    continue
                cells = [cell.strip() for cell in source_line.strip().strip("|").split("|")]
                if len(cells) < 4 or not re.fullmatch(r"\d{8,}", cells[0]):
                    continue
                if not re.fullmatch(r"\d+(?:\.\d+)?", cells[3]):
                    continue
                table_rows.append({
                    "semester": semester, "category": "Generic Core Courses",
                    "code": cells[0], "title": cells[1], "course_type": cells[2],
                    "credits": cells[3], "source": relative_source,
                })
        if table_rows:
            rows.extend(table_rows)
        for row_semester, category, title, credits in extract_curriculum_subjects(source_results):
            if table_rows:
                continue
            normalized_title = re.sub(r"\s+", " ", title).strip().casefold()
            code = code_by_source_title.get((relative_source, normalized_title))
            if not code:
                continue
            if row_semester == semester:
                rows.append({
                    "semester": row_semester,
                    "category": category,
                    "code": code,
                    "title": title,
                    "credits": credits,
                    "course_type": None,
                    "source": relative_source,
                })
            else:
                other_semester_codes.add(code)
    return rows, other_semester_codes


def format_semester_curriculum_evidence(rows: list[dict]) -> str:
    """Serialize parsed course facts with unambiguous code and credit labels."""
    lines = ["Source-parsed curriculum entries (one row per course):"]
    previous_scope = None
    for row in rows:
        scope = (row["source"], row["semester"], row["category"])
        if scope != previous_scope:
            lines.extend([
                f"Source: {row['source']}",
                f"Semester {row['semester']} - {row['category']}",
            ])
            previous_scope = scope
        fact = f"Course code: {row['code']} | Subject: {row['title']}"
        if row.get("course_type"):
            fact += f" | Course type: {row['course_type']}"
        if row["credits"] is not None:
            fact += f" - {row['credits']} credit" + ("s" if row["credits"] != "1" else "")
        elif "health and wellness module i" in row["title"].casefold():
            fact += " | Mandatory non-credit course"
        lines.append(fact)
    return "\n".join(lines)


def format_semester_curriculum_answer(rows: list[dict]) -> str:
    """Produce a compact source-grounded fallback without calculating totals."""
    lines = [f"Semester {rows[0]['semester']} subjects:"]
    previous_scope = None
    for row in rows:
        scope = (row["source"], row["category"])
        if scope != previous_scope:
            lines.append(f"\n{row['category']}:")
            previous_scope = scope
        fact = f"- {row['code']} - {row['title']}"
        if row["credits"] is not None:
            fact += f" - {row['credits']} credit" + ("s" if row["credits"] != "1" else "")
        elif "health and wellness module i" in row["title"].casefold():
            fact += " - mandatory non-credit course"
        if row.get("course_type"):
            fact += f" | Course type: {row['course_type']}"
        lines.append(fact)
    return "\n".join(lines)


def semester_curriculum_answer_validation_failure(
    answer: str, rows: list[dict], other_semester_codes: set[str]
) -> Optional[str]:
    """Return a concise failure rule for a non-grounded semester answer."""
    normalized_answer = re.sub(r"\s+", " ", answer).casefold()
    if not rows:
        return "no_expected_curriculum_rows"
    course_code_matches = list(re.finditer(r"\b\d{8,}[a-z0-9-]*\b", normalized_answer))
    if any(
        not re.fullmatch(r"\d{8,}[a-z0-9-]*", match.group(0))
        for match in re.finditer(r"\b[a-z]{1,8}[-/]?\d{2,}[a-z0-9-]*\b", normalized_answer)
    ):
        return "unexpected_catalog_code"
    if [match.group(0) for match in course_code_matches] != [
        str(row["code"]).casefold() for row in rows
    ]:
        return "course_code_sequence_mismatch"
    for index, row in enumerate(rows):
        code_match = course_code_matches[index]
        next_code = course_code_matches[index + 1].start() if index + 1 < len(rows) else len(normalized_answer)
        course_text = normalized_answer[code_match.start():next_code]
        if row["title"].casefold() not in course_text:
            return "course_subject_pairing_mismatch"
        if row["credits"] is not None:
            if not re.search(rf"\b{re.escape(str(row['credits']))}\s+credits?\b", course_text):
                return "course_credit_pairing_mismatch"
        if row.get("course_type") and not re.search(
            rf"\b{re.escape(row['course_type'])}\b", course_text, re.I
        ):
            return "course_type_pairing_mismatch"
        elif "health and wellness module i" in row["title"].casefold():
            if "mandatory non-credit" not in course_text or re.search(r"\b\d+\s+credits?\b", course_text):
                return "mandatory_noncredit_designation_mismatch"
    if any(code.casefold() in normalized_answer for code in other_semester_codes):
        return "cross_semester_course_code"
    if re.search(
        r"\b(?:total|overall|sum)\b.{0,80}\bcredits?\b|"
        r"\bcredits?\b.{0,80}\b(?:total|overall|sum)\b",
        normalized_answer,
    ):
        return "unsupported_aggregate_credit_total"
    return None


def semester_curriculum_answer_is_grounded(
    answer: str, rows: list[dict], other_semester_codes: set[str]
) -> bool:
    """Reject incomplete lists, wrong stated credits, cross-semester codes,
    and unsupported aggregate credit totals.
    """
    return semester_curriculum_answer_validation_failure(
        answer, rows, other_semester_codes
    ) is None


def format_broad_curriculum_answer(results: list[dict]) -> Optional[str]:
    """Create a complete semester-wise subject listing without LLM summary."""
    subjects = extract_curriculum_subjects(results)
    if not subjects:
        return None
    lines = ["The selected curriculum lists the following Annexure A subjects:"]
    current_semester: Optional[int] = None
    current_category: Optional[str] = None
    for semester, category, title, credits in subjects:
        if semester != current_semester:
            lines.extend(["", f"Semester {semester}"])
            current_semester, current_category = semester, None
        if category != current_category:
            lines.append(f"{category}:")
            current_category = category
        credit_label = f" - {credits} credits" if credits is not None else ""
        lines.append(f"- {title}{credit_label}")
    return "\n".join(lines)


def build_context(results: list[dict], max_context_chars: int = MAX_CONTEXT_CHARS) -> tuple[str, dict]:
    """Keep ranked evidence whole; deduplicate only repeats from the same source.

    Source headers are emitted once per metadata scope. Later passages from
    that scope refer back to the same source number without repeating its file
    and filter labels. Passage order remains the incoming relevance order.
    """
    parts = []
    seen_passages = set()
    source_numbers = {}
    source_scopes = set()
    context_chars = 0
    separator = "\n\n---\n\n"
    for item in results:
        metadata = item["metadata"]
        source_key = (
            metadata.get("source", "unknown"),
            metadata.get("document_type", "unknown"),
            metadata.get("batch"),
        )
        semester = metadata.get("semester")
        document = item["document"]
        duplicate_key = (source_key, semester, re.sub(r"\s+", " ", document).strip())
        if duplicate_key in seen_passages:
            continue
        seen_passages.add(duplicate_key)

        is_new_source = source_key not in source_numbers
        source_number = source_numbers.get(source_key, len(source_numbers) + 1)
        if is_new_source:
            labels = [
                f"[SOURCE {source_number}]",
                f"Type: {source_key[1]}",
                f"File: {source_key[0]}",
            ]
            if source_key[2]:
                labels.append(f"Batch: {source_key[2]}")
            if semester is not None:
                labels.append(f"Semester: {semester}")
            part = "\n".join(labels) + "\n" + document
            if semester is not None:
                source_scopes.add((source_key, semester))
        else:
            labels = [f"[SOURCE {source_number}, continued]"]
            if semester is not None and (source_key, semester) not in source_scopes:
                labels.append(f"Semester: {semester}")
                source_scopes.add((source_key, semester))
            part = "\n".join(labels) + "\n" + document
        # Always retain the highest-ranked chunk. Later chunks are included
        # whole so a table/list is never silently cut in half.
        added_chars = len(part) + (len(separator) if parts else 0)
        if parts and context_chars + added_chars > max_context_chars:
            continue
        if is_new_source:
            source_numbers[source_key] = source_number
        parts.append(part)
        context_chars += added_chars
    context = separator.join(parts)
    return context, {"context_chunks": len(parts), "context_characters": len(context), "context_estimated_tokens": round(len(context) / 4)}


SYSTEM_PROMPT = """You are CampusGPT for the university. Answer only with evidence in the supplied sources, directly and concisely. Never infer or calculate credits, marks, hours, course codes, dates, names, or semesters. Use exact values only when clearly labelled; preserve names, dates, numbers, course names, and codes. Keep batches, semesters, cohorts, specializations, and departments separate, and treat a specialization option as an option rather than a universal requirement. For broad placement questions, summarize supported companies, roles, explicitly stated packages/stipends, and placement/internship records; do not list individual students unless asked. For broad timetable questions, state the available cohort/sheet scope and supported schedule details, then ask which cohort, division, and day are needed for a specific timetable. For list and faculty questions, provide the complete supported list in a concise structured format. If evidence conflicts or is insufficient, say so plainly. Do not mention internal systems or source labels."""


def generate_with_ollama(query: str, context: str, plan: QueryPlan) -> tuple[str, dict]:
    scope = []
    if plan.batches:
        scope.append("Allowed batch(es): " + ", ".join(plan.batches))
    if plan.semester is not None:
        scope.append(f"Allowed semester: {plan.semester}")
    curriculum_guidance = ""
    num_predict = NORMAL_GENERATION_TOKENS
    num_ctx = 4096
    if plan.document_type == "curriculum":
        num_predict = BROAD_CURRICULUM_GENERATION_TOKENS
        num_ctx = 16384
        curriculum_guidance = """
CURRICULUM ANSWERING RULES:
- Answer the user's curriculum or subjects question directly, using all relevant supplied curriculum context before concluding that information is absent.
- Give a structured answer. When supported, organize subjects by semester and distinguish core, elective, and specialization subjects exactly as the source does.
- For a broad curriculum question, extract every actual supplied subject name. Never summarize a list as "other subjects", "etc.", a subject count, or similar. Combine chunks from the same curriculum source, and report only names explicitly present in them.
- Include credits, batches, and semesters only where they are explicitly labelled in the supplied context.
- Do not stop after a few examples when later supplied context contains additional relevant subjects. Finish every sentence and list.
- Do not combine different batches without naming their batch. Prefer the requested or detected batch when one is supplied.
- If the supplied context covers only part of a curriculum, say exactly which supported portion is shown; never infer missing subjects from general knowledge.
"""
    elif plan.document_type == "faculty":
        # Faculty directories need enough room to list every supported name,
        # whereas a course/person lookup is intentionally concise.
        num_predict = FACULTY_GENERATION_TOKENS if is_broad_faculty_query(plan, query) else SIMPLE_GENERATION_TOKENS
        num_ctx = 8192 if is_broad_faculty_query(plan, query) else 4096
    elif plan.document_type in {"infrastructure", "department_information"}:
        num_predict = STRUCTURED_GENERATION_TOKENS
    elif plan.document_type in {"academic_calendar", "handbook", "placement", "timetable"}:
        num_predict = NORMAL_GENERATION_TOKENS
    elif plan.classification == "simple":
        num_predict = SIMPLE_GENERATION_TOKENS
    prompt = f"{SYSTEM_PROMPT}\n{curriculum_guidance}\n\n{' '.join(scope)}\n\nSOURCES:\n{context}\n\nQUESTION: {query}\nANSWER:"
    try:
        request_started = time.perf_counter()
        response = client.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={
                "temperature": 0,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
                "num_thread": NUM_THREAD,
            },
            keep_alive="15m",
        )
        ollama_request_ms = (time.perf_counter() - request_started) * 1000
    except Exception as exc:
        raise RuntimeError(f"Ollama generation failed for {MODEL_NAME}: {exc}") from exc
    answer = response.get("response", "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty response.")
    metrics = _generation_diagnostics(response)
    metrics["ollama_request_ms"] = round(ollama_request_ms, 2)
    return answer, metrics


def extract_sources(results: list[dict]) -> list[dict]:
    seen, sources = set(), []
    for item in results:
        metadata = item["metadata"]
        key = (metadata.get("source"), metadata.get("document_type"))
        if key[0] and key not in seen:
            seen.add(key)
            sources.append({"source": str(key[0]), "document_type": str(key[1] or "unknown")})
    return sources


def format_calendar_comparison_answer(query: str, results: list[dict]) -> Optional[str]:
    """Format the sourced 2026–27 ODD/EVEN comparison without swapping terms."""
    if not (
        re.search(r"\bodd\b", query, re.I)
        and re.search(r"\beven\b", query, re.I)
        and re.search(r"\b(end[- ]semester|end semester|exam(?:ination)?)\b", query, re.I)
    ):
        return None
    source_paths = {
        str(item.get("metadata", {}).get("source", ""))
        for item in results
        if item.get("metadata", {}).get("document_type") == "academic_calendar"
    }
    for relative in source_paths:
        source_path = PROJECT_ROOT / relative
        if not source_path.is_file():
            continue
        text = source_path.read_text(encoding="utf-8")
        odd_start = re.search(
            r"\|\s*B\.Tech registration/session commencement\s*\|\s*(20 July 2026)\s*\|",
            text, re.I,
        )
        even_start = re.search(
            r"\|\s*B\.Tech registration/session commencement\s*\|\s*(4 January 2027)\s*\|",
            text, re.I,
        )
        odd_exam = re.search(
            r"\|\s*December\s+7\s*[–-]\s*12\s*\|\s*End Semester Examination\s*\(ESE ODD 2026\)",
            text, re.I,
        )
        if odd_start and even_start and odd_exam:
            return (
                f"The ODD session starts {odd_start.group(1)}; the EVEN session starts "
                f"{even_start.group(1)}. ODD end-semester exams are scheduled for "
                "7–12 December 2026."
            )
    return None


def format_academic_session_start_answer(query: str, results: list[dict]) -> Optional[str]:
    """Answer the B.Tech session-start lookup from its labelled calendar row."""
    normalized_query = re.sub(r"[^a-z0-9]+", " ", query.casefold())
    if not (
        re.search(r"\b2026\s+27\b", normalized_query)
        and re.search(r"\bsession\b", normalized_query)
        and re.search(r"\bstart", normalized_query)
    ):
        return None
    source_paths = {
        str(item.get("metadata", {}).get("source", ""))
        for item in results
        if item.get("metadata", {}).get("document_type") == "academic_calendar"
    }
    for relative in source_paths:
        source_path = PROJECT_ROOT / relative
        if not source_path.is_file():
            continue
        text = source_path.read_text(encoding="utf-8")
        if re.search(
            r"\|\s*B\.Tech registration/session commencement\s*\|\s*20 July 2026\s*\|",
            text,
            re.I,
        ):
            return "For B.Tech, registration and session commencement for 2026-27 is 20 July 2026."
    return None


def format_attendance_policy_answer(query: str, results: list[dict]) -> Optional[str]:
    """Preserve the handbook's attendance and leave qualifications."""
    below_threshold_question = bool(
        re.search(r"\b(below|under|less than)\s+(?:the\s+)?75\s*%", query, re.I)
    )
    if not (
        re.search(r"\battendance\b", query, re.I)
        and (re.search(r"\bleave\b", query, re.I) or below_threshold_question)
    ):
        return None
    source_paths = {
        str(item.get("metadata", {}).get("source", ""))
        for item in results
        if item.get("metadata", {}).get("document_type") == "student_handbook"
    }
    for relative in source_paths:
        source_path = PROJECT_ROOT / relative
        if not source_path.is_file():
            continue
        lines = source_path.read_text(encoding="utf-8").splitlines()
        minimum = next((line.strip(" -*") for line in lines if "75% attendance in each course" in line), None)
        maximum_leave = next((line.strip(" -*") for line in lines if "Maximum 25% leave is allowed" in line), None)
        prior_approval = next((line.strip(" -*") for line in lines if "Leave normally requires prior approval" in line), None)
        medical_rule = next((line.strip(" -*") for line in lines if "Medical leave is granted only if" in line), None)
        if minimum and maximum_leave and prior_approval:
            parts = [
                "At least 75% attendance in each course is required to appear for the End Semester Examination (ESE).",
                "If attendance in any course is below 75%, the student is not eligible to appear for the ESE.",
                f"Separately, the handbook states: {maximum_leave} {prior_approval}",
            ]
            if medical_rule:
                parts.append(medical_rule)
            return " ".join(parts)
    return None


def format_ra_placement_summary(query: str, results: list[dict]) -> Optional[str]:
    """Report recruiter names and aggregate outcomes without inventing mappings."""
    if not (
        re.search(r"\b(package|packages|ctc)\b", query, re.I)
        and re.search(r"\b(company|companies|recruiter|recruiters)\b", query, re.I)
    ):
        return None
    source_paths = {
        str(item.get("metadata", {}).get("source", ""))
        for item in results
        if item.get("metadata", {}).get("document_type") == "placement"
        and item.get("metadata", {}).get("department") == "RA"
    }
    for relative in source_paths:
        source_path = PROJECT_ROOT / relative
        if not source_path.is_file():
            continue
        text = source_path.read_text(encoding="utf-8")
        sections = {}
        for block in re.split(r"(?m)^##\s+", text):
            heading, separator, body = block.partition("\n")
            if separator:
                sections[heading.strip()] = body.strip()
        recruiters = next(
            (body for heading, body in sections.items() if heading.startswith("Major recruiters shown")),
            None,
        )
        batch_2022 = next(
            (body for heading, body in sections.items() if "batch 2022" in heading.lower()),
            None,
        )
        batch_2021 = next(
            (body for heading, body in sections.items() if "batch 2021" in heading.lower()),
            None,
        )
        if recruiters and batch_2022 and batch_2021:
            return (
                "Major recruiters shown include " + recruiters + "\n\n"
                "Batch 2022–26 — " + batch_2022 + "\n\n"
                "Batch 2021–25 — " + batch_2021 + "\n\n"
                "These are cohort-level figures. The source does not link a package "
                "or stipend to an individual company or student."
            )
    return None


def format_entc_lab_list(query: str, results: list[dict]) -> Optional[str]:
    """Return the extracted ENTC lab names without adding an acronym expansion."""
    if not re.search(r"\b(labs?|laborator(?:y|ies))\b", query, re.I):
        return None
    source_paths = {
        str(item.get("metadata", {}).get("source", ""))
        for item in results
        if item.get("metadata", {}).get("department") == "ENTC"
        and item.get("metadata", {}).get("document_type") == "infrastructure"
    }
    for relative in source_paths:
        source_path = PROJECT_ROOT / relative
        if not source_path.is_file():
            continue
        text = source_path.read_text(encoding="utf-8")
        names = [
            line.strip()
            for line in text.splitlines()
            if re.search(r"\bLAB\b", line, re.I)
            and line.strip().casefold() != "lab infrastructure"
            and not line.lstrip().startswith("#")
            and not line.lstrip().startswith(">")
            and not line.lstrip().startswith("-")
        ]
        if names:
            return "The ENTC infrastructure source lists:\n" + "\n".join(
                f"- {name}" for name in names
            )
    return None


def answer_question(
    query: str,
    document_type: Optional[str] = None,
    batch: Optional[str] = None,
    top_k: Optional[int] = None,
    candidate_k: Optional[int] = None,
    disable_reranker: bool = False,
) -> dict:
    started = time.perf_counter()
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
    query = query.strip()
    planning_started = time.perf_counter()
    if document_type is None and is_faculty_teaching_query(query):
        return _finalize_short_route(
            _faculty_teaching_answer(query),
            started,
            (time.perf_counter() - planning_started) * 1000,
        )
    # Clear schedule requests use the raw-workbook structured index. This
    # branch runs before Chroma retrieval and leaves ordinary RAG untouched.
    explicit_timetable = timetable_data.detect_timetable_intent(query)
    # A generic person lookup may need timetable alias mapping, but a stronger
    # intent such as an HOD/profile request must not be reclassified because a
    # timetable alias happens to occur inside another word (for example email).
    possible_lookup = (
        bool(re.search(r"\b(when|where|who|stand\s+for)\b", query, re.I))
        and plan_query(query, document_type, batch).document_type is None
    )
    requested_departments = detect_departments(query)
    if (
        (document_type == "timetable" or explicit_timetable)
        and any(department != "AIML" for department in requested_departments)
    ):
        query_planning_ms = (time.perf_counter() - planning_started) * 1000
        plan = plan_query(query, "timetable", batch)
        return _finalize_short_route(
            no_timetable_answer(plan, requested_departments),
            started,
            query_planning_ms,
        )
    if document_type == "timetable" or (document_type is None and (explicit_timetable or possible_lookup)):
        timetable_index = timetable_data.get_index()
        if document_type == "timetable" or timetable_data.detect_timetable_intent(query, timetable_index):
            query_planning_ms = (time.perf_counter() - planning_started) * 1000
            plan = plan_query(query, "timetable", batch)
            if any(department != "AIML" for department in requested_departments):
                return _finalize_short_route(
                    no_timetable_answer(plan, requested_departments),
                    started,
                    query_planning_ms,
                )
            return _answer_timetable(query, timetable_index, started, query_planning_ms)
    plan = plan_query(query, document_type, batch)
    query_planning_ms = (time.perf_counter() - planning_started) * 1000
    if plan.batch_selection_required:
        return _finalize_short_route(batch_selection_answer(plan), started, query_planning_ms)
    if plan.entc_placement_unavailable:
        return _finalize_short_route({"answer": "ENTC placement information is not currently available in the CampusGPT knowledge base.", "sources": [], "query_classification": plan.classification, "document_type": None, "detected_department": plan.department, "detected_batch": list(plan.batches), "detected_semester": plan.semester, "default_curriculum_batch_used": plan.default_batch_used, "semester_metadata_fallback": False, "candidate_k": 0, "top_k": 0, "context": {"context_chunks": 0, "context_characters": 0, "context_estimated_tokens": 0}}, started, query_planning_ms)
    if plan.department in NO_SOURCE_DOCUMENT_TYPES and plan.document_type in NO_SOURCE_DOCUMENT_TYPES[plan.department]:
        return _finalize_short_route(no_source_answer(plan, plan.document_type), started, query_planning_ms)
    # These departments currently have no timetable source. Answer directly so
    # a semantically similar AIML timetable can never be presented as theirs.
    if plan.document_type == "timetable" and plan.department in {"RA", "ENTC"}:
        return _finalize_short_route({"answer": f"No timetable is currently available in the knowledge base for {plan.department}.", "sources": [], "query_classification": plan.classification, "document_type": plan.document_type, "detected_department": plan.department, "detected_batch": list(plan.batches), "detected_semester": plan.semester, "default_curriculum_batch_used": plan.default_batch_used, "semester_metadata_fallback": False, "candidate_k": 0, "top_k": 0, "context": {"context_chunks": 0, "context_characters": 0, "context_estimated_tokens": 0}}, started, query_planning_ms)
    default_candidate_k, default_top_k = retrieval_settings(plan)
    candidate_k, top_k = candidate_k or default_candidate_k, top_k or default_top_k
    broad_faculty = is_broad_faculty_query(plan, query)
    query_embedding_ms = 0.0
    if is_hod_query(query) and plan.department:
        results, retrieved_candidate_count, retrieval_ms = _retrieve_hod_evidence(plan)
        reranking_ms = 0.0
        fallback = False
    elif broad_faculty and plan.department:
        results, retrieved_candidate_count, retrieval_ms = _retrieve_complete_faculty(plan)
        reranking_ms = 0.0
        fallback = False
    else:
        if disable_reranker:
            results, retrieved_candidate_count, query_embedding_ms, retrieval_ms, reranking_ms, fallback = _retrieve(
                plan, query, candidate_k, top_k, disable_reranker=True
            )
        else:
            results, retrieved_candidate_count, query_embedding_ms, retrieval_ms, reranking_ms, fallback = _retrieve(
                plan, query, candidate_k, top_k
            )
    broad_curriculum = is_broad_curriculum_query(plan, query)
    # Complete directories retain all deduplicated faculty evidence. Every
    # other request, including narrow faculty lookups, keeps only reranked
    # top-k evidence before prompt construction.
    if not broad_curriculum and not broad_faculty:
        results = results[:top_k]
    results = filter_named_placement_evidence(query, plan, results)
    exact_named_placement_answer = (
        format_named_placement_answer(query, plan, results)
        if plan.document_type == "placement" else None
    )
    generation_ms = 0.0
    context_construction_ms = 0.0
    generation_metrics = None
    answer = "Information is not currently available in the knowledge base."
    if results:
        exact_session_start_answer = (
            format_academic_session_start_answer(query, results)
            if plan.document_type == "academic_calendar" else None
        )
        exact_calendar_answer = (
            format_calendar_comparison_answer(query, results)
            if plan.document_type == "academic_calendar" else None
        )
        exact_attendance_answer = (
            format_attendance_policy_answer(query, results)
            if plan.document_type == "handbook" else None
        )
        exact_ra_placement_answer = (
            format_ra_placement_summary(query, results)
            if plan.document_type == "placement" and plan.department == "RA" else None
        )
        exact_entc_labs_answer = (
            format_entc_lab_list(query, results)
            if plan.document_type == "infrastructure" and plan.department == "ENTC" else None
        )
        if exact_named_placement_answer:
            answer = exact_named_placement_answer
            context_stats = {
                "context_chunks": len(results),
                "context_characters": len(answer),
                "context_estimated_tokens": round(len(answer) / 4),
            }
        elif exact_session_start_answer:
            answer = exact_session_start_answer
            context_stats = {
                "context_chunks": len(results),
                "context_characters": len(answer),
                "context_estimated_tokens": round(len(answer) / 4),
            }
        elif exact_calendar_answer:
            answer = exact_calendar_answer
            context_stats = {
                "context_chunks": len(results),
                "context_characters": len(answer),
                "context_estimated_tokens": round(len(answer) / 4),
            }
        elif exact_attendance_answer:
            answer = exact_attendance_answer
            context_stats = {
                "context_chunks": len(results),
                "context_characters": len(answer),
                "context_estimated_tokens": round(len(answer) / 4),
            }
        elif exact_ra_placement_answer:
            answer = exact_ra_placement_answer
            context_stats = {
                "context_chunks": len(results),
                "context_characters": len(answer),
                "context_estimated_tokens": round(len(answer) / 4),
            }
        elif exact_entc_labs_answer:
            answer = exact_entc_labs_answer
            context_stats = {
                "context_chunks": len(results),
                "context_characters": len(answer),
                "context_estimated_tokens": round(len(answer) / 4),
            }
        elif broad_curriculum:
            answer = format_broad_curriculum_answer(results)
            context_stats = {
                "context_chunks": len(results),
                "context_characters": sum(len(item["document"]) for item in results),
                "context_estimated_tokens": round(sum(len(item["document"]) for item in results) / 4),
            }
            if answer is None:
                answer = "The selected curriculum did not contain readable course rows."
        else:
            if plan.document_type == "curriculum":
                context_limit = BROAD_CURRICULUM_CONTEXT_CHARS
            elif plan.document_type == "faculty":
                context_limit = BROAD_FACULTY_CONTEXT_CHARS if broad_faculty else NARROW_FACULTY_CONTEXT_CHARS
            else:
                context_limit = MAX_CONTEXT_CHARS
            context_started = time.perf_counter()
            semester_subject_query = (
                plan.document_type == "curriculum"
                and plan.semester is not None
                and bool(re.search(r"\b(subjects?|courses?)\b", query, re.I))
            )
            semester_rows: list[dict] = []
            other_semester_codes: set[str] = set()
            if semester_subject_query:
                semester_rows, other_semester_codes = extract_semester_curriculum_rows(
                    results, plan.semester
                )
            if semester_rows:
                context = format_semester_curriculum_evidence(semester_rows)
                context_stats = {
                    "context_chunks": len(results),
                    "context_characters": len(context),
                    "context_estimated_tokens": round(len(context) / 4),
                }
            else:
                context, context_stats = build_context(results, max_context_chars=context_limit)
            context_construction_ms = (time.perf_counter() - context_started) * 1000
            answer, generation_metrics = generate_with_ollama(query, context, plan)
            if semester_rows:
                validation_failure = semester_curriculum_answer_validation_failure(
                    answer, semester_rows, other_semester_codes
                )
                fallback_activated = validation_failure is not None
                if fallback_activated:
                    answer = format_semester_curriculum_answer(semester_rows)
                logger.info(
                    "semester_curriculum_answer_diagnostics "
                    "validation_passed=%s fallback_activated=%s failed_rule=%s "
                    "final_source=%s llm_generation_calls=1",
                    not fallback_activated,
                    fallback_activated,
                    validation_failure or "none",
                    "source_grounded_fallback" if fallback_activated else "llm",
                )
            generation_ms = generation_metrics.get("ollama_request_ms", 0.0)
    else:
        context_stats = {"context_chunks": 0, "context_characters": 0, "context_estimated_tokens": 0}
    total_ms = (time.perf_counter() - started) * 1000
    return {"answer": answer, "sources": extract_sources(results), "query_classification": plan.classification, "document_type": plan.document_type, "detected_department": plan.department, "detected_batch": list(plan.batches), "detected_semester": plan.semester, "default_curriculum_batch_used": plan.default_batch_used, "semester_metadata_fallback": fallback, "candidate_k": candidate_k, "top_k": top_k, "retrieved_candidate_count": retrieved_candidate_count, "reranked_chunk_count": len(results), "context": context_stats, "generation_metrics": generation_metrics, "latency": _latency_payload(total_ms, query_planning_ms=query_planning_ms, query_embedding_ms=query_embedding_ms, chromadb_retrieval_ms=retrieval_ms, bge_reranking_ms=reranking_ms, context_construction_ms=context_construction_ms, llm_generation_ms=generation_ms)}


def _timetable_supplementary_context(records: list[dict], index) -> tuple[str, bool]:
    """Return only explicit alias/status notes; workbook fields stay authoritative."""
    lines = []
    seen = set()
    diagnostics = getattr(index, "supplementary_diagnostics", {}) or {}
    note_pattern = re.compile(
        r"\[([^\[\]]*(?:source-confirmed|inferred|requires verification)[^\[\]]*)\]",
        re.I,
    )
    for record in records:
        for evidence in record.get("supplementary_evidence", ()):
            text = str(evidence.get("text", ""))
            details = text.split("| Source cells:", 1)[0]
            raw = str(record.get("raw", "")).strip()
            if "Lab / tutorial group entry:" in details:
                activity_start = details.find(raw, details.find("Lab / tutorial group entry:"))
                if activity_start < 0:
                    continue
                next_group = re.search(r';\s*"', details[activity_start + len(raw):])
                activity_end = (
                    activity_start + len(raw) + next_group.start()
                    if next_group else len(details)
                )
                details = details[activity_start:activity_end]
            else:
                subject_start = details.find("Subject/activity:")
                if subject_start >= 0:
                    details = details[subject_start:]
            for note in note_pattern.findall(details):
                note = " ".join(note.split())
                key = (record.get("sheet"), evidence.get("source_cell"), note.casefold())
                if key not in seen:
                    seen.add(key)
                    lines.append(
                        f"- {record.get('sheet')}!{evidence.get('source_cell')}: {note}"
                    )
        source_cell = record.get("source_cell")
        if not source_cell:
            continue
        for warning in diagnostics.get("ambiguous_matches", ()):
            if warning.get("sheet") != record.get("sheet") or warning.get("cell") != source_cell:
                continue
            key = (record.get("sheet"), source_cell, "ambiguous-markdown")
            if key not in seen:
                seen.add(key)
                record_ids = ", ".join(warning.get("markdown_record_ids", ()))
                lines.append(
                    f"- WARNING: Supplementary annotations {record_ids} for "
                    f"{record.get('sheet')}!{source_cell} were ambiguous and were excluded."
                )
    if not lines:
        return "", False
    return (
        "Supplementary Markdown alias/status notes (secondary evidence; do not use "
        "them to change any workbook schedule field):\n" + "\n".join(lines),
        True,
    )


def _answer_timetable(query: str, index, started: float, query_planning_ms: float = 0.0) -> dict:
    subject_lookup = timetable_data.resolve_subject_teacher_query(query, index)
    if subject_lookup:
        names = subject_lookup["instructors"]
        if names:
            answer = f"{subject_lookup['subject']['name']} is taught by {', '.join(names)}."
        else:
            answer = f"Instructor information for {subject_lookup['subject']['name']} is not available in the raw workbook mapping tables."
        total_ms = (time.perf_counter() - started) * 1000
        return {
            "answer": answer,
            "sources": [{"source": index.source, "document_type": "timetable"}],
            "query_classification": "simple", "document_type": "timetable",
            "detected_department": "AIML", "detected_batch": [], "detected_semester": None,
            "default_curriculum_batch_used": False, "semester_metadata_fallback": False,
            "candidate_k": len(names), "top_k": len(names), "retrieved_candidate_count": len(names),
            "reranked_chunk_count": len(names),
            "context": {"context_chunks": len(names), "context_characters": len(answer), "context_estimated_tokens": round(len(answer) / 4)},
            "latency": _latency_payload(total_ms, query_planning_ms=query_planning_ms),
            "generation_metrics": None,
        }
    mapping_lookup = timetable_data.resolve_mapping_query(query, index)
    if mapping_lookup:
        total_ms = (time.perf_counter() - started) * 1000
        return {
            "answer": mapping_lookup["answer"],
            "sources": [{"source": index.source, "document_type": "timetable"}],
            "query_classification": "simple", "document_type": "timetable",
            "detected_department": "AIML", "detected_batch": [], "detected_semester": None,
            "default_curriculum_batch_used": False, "semester_metadata_fallback": False,
            "candidate_k": 1, "top_k": 1, "retrieved_candidate_count": 1,
            "reranked_chunk_count": 1,
            "context": {"context_chunks": 1, "context_characters": len(mapping_lookup["answer"]), "context_estimated_tokens": round(len(mapping_lookup["answer"]) / 4)},
            "latency": _latency_payload(total_ms, query_planning_ms=query_planning_ms),
            "generation_metrics": None,
        }
    filters = timetable_data.extract_filters(query, index)
    records = index.search(query, filters)
    context_construction_ms = 0.0
    context_started = time.perf_counter()
    context = "\n".join(
        " | ".join((
            f"Sheet: {item['sheet']}", f"Cohort: {item['cohort']}",
            f"Day: {item['day']}", f"Time: {item['time']}",
            f"Entry (verbatim): {item['raw']}", f"Subject: {item['subject']}",
            f"Faculty: {', '.join(item['faculty']) or ', '.join(item['faculty_codes']) or 'not listed'}",
            f"Room: {item['room'] or 'not listed'}", f"Lab: {item['lab'] or 'not listed'}",
            f"Group: {item['group'] or 'not listed'}",
            f"Workbook source cell: {item['source']}!{item.get('source_cell') or 'unresolved'}",
        )) for item in records
    )
    context_construction_ms = (time.perf_counter() - context_started) * 1000
    generation_ms = 0.0
    generation_metrics = None
    supplementary_contributed = False
    verbatim_schedule_request = bool(
        re.search(r"\bparallel\b|\bseparat(?:e|ely)\b", query, re.I)
        or filters.get("time_range")
    )
    complete_schedule_request = bool(
        re.search(r"\b(?:complete|full|entire|all)\b", query, re.I)
        and re.search(r"\b(?:timetable|time\s*table|schedule)\b", query, re.I)
    )
    if records and complete_schedule_request:
        session_label = " ".join(
            value for value in (
                filters.get("academic_year"),
                f"{filters['academic_term'].title()} Semester" if filters.get("academic_term") else None,
            ) if value
        )
        heading = f"Complete structured timetable{f' for {session_label}' if session_label else ''} ({len(records)} workbook entries):"
        lines = [heading]
        current_day = None
        for item in records:
            if item["day"] != current_day:
                current_day = item["day"]
                lines.append(f"\n{current_day}:")
            faculty = ", ".join(item["faculty"]) or ", ".join(item["faculty_codes"]) or "not listed"
            lines.append(
                f"- {item['time']}: {item['raw']} | Subject: {item['subject']} | "
                f"Faculty: {faculty} | Room: {item['room'] or 'not listed'} | "
                f"Cohort/batch: {item['cohort']} | Group: {item['group'] or 'not listed'} | "
                f"Source: {item['sheet']}!{item.get('source_cell') or 'unresolved'}"
            )
        lines.append("Blank or unavailable workbook fields are marked 'not listed'; ambiguous faculty codes are not resolved by this answer route.")
        answer = "\n".join(lines)
    elif records and verbatim_schedule_request:
        lines = ["Workbook entries:"]
        for item in records:
            lines.append(
                f"- {item['day']} {item['time']}: {item['raw']} | "
                f"Instructor code row: {', '.join(item['faculty_codes']) or 'not listed'} | "
                f"Room cell: {item['room'] or 'not specified'}"
            )
        answer = "\n".join(lines)
    elif records:
        plan_started = time.perf_counter()
        plan = plan_query(query, "timetable")
        query_planning_ms += (time.perf_counter() - plan_started) * 1000
        supplementary_started = time.perf_counter()
        supplementary_context, supplementary_contributed = _timetable_supplementary_context(
            records, index
        )
        if supplementary_context:
            context = f"{context}\n\n{supplementary_context}"
        context_construction_ms += (time.perf_counter() - supplementary_started) * 1000
        answer, generation_metrics = generate_with_ollama(
            query,
            "The records below are the complete set selected from the structured workbook "
            "before generation. Include every selected session and preserve parallel entries. "
            "Use the workbook as authoritative for day, time, cohort, class, subject, faculty "
            "assignment, room, group, and lab. Supplementary Markdown is secondary and may "
            "support only explicit aliases or notes; it must never replace or override a workbook "
            "schedule field. Use names only when explicitly marked Source-confirmed. Preserve "
            "Inferred/requires-verification labels as uncertain, and do not guess faculty identities. "
            "If an ambiguity warning is present, retain the warning. Do not infer missing names, "
            "rooms, groups, or times; state when a workbook field is not listed.\n\n" + context,
            plan,
        )
        generation_ms = generation_metrics.get("ollama_request_ms", 0.0)
    else:
        answer = "That timetable information is not available in the raw workbook."
    total_ms = (time.perf_counter() - started) * 1000
    sources = ([{"source": index.source, "document_type": "timetable"}] if records else [])
    if records and supplementary_contributed:
        markdown_source = Path(timetable_data.SUPPLEMENTARY_MARKDOWN_PATH).relative_to(
            timetable_data.PROJECT_ROOT
        ).as_posix()
        sources.append({"source": markdown_source, "document_type": "timetable"})
    return {
        "answer": answer,
        "sources": sources,
        "query_classification": "simple",
        "document_type": "timetable",
        "detected_department": "AIML" if records else None,
        "detected_batch": [filters["batch"]] if filters.get("batch") else [],
        "detected_semester": None,
        "default_curriculum_batch_used": False,
        "semester_metadata_fallback": False,
        "candidate_k": len(records),
        "top_k": len(records),
        "retrieved_candidate_count": len(records),
        "reranked_chunk_count": len(records),
        "context": {"context_chunks": len(records), "context_characters": len(context), "context_estimated_tokens": round(len(context) / 4)},
        "generation_metrics": generation_metrics,
        "latency": _latency_payload(total_ms, query_planning_ms=query_planning_ms, context_construction_ms=context_construction_ms, llm_generation_ms=generation_ms),
    }


if __name__ == "__main__":
    print("CampusGPT terminal test. Type exit to stop.")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        try:
            result = answer_question(question)
            print(f"Classification: {result['query_classification']} | Type: {result['document_type']}")
            if result.get("intent_plans"):
                print("MULTI-INTENT QUERY")
                print(f"Original: {question}")
                for index, intent in enumerate(result["intent_plans"], 1):
                    print(f"Intent {index}: Query: {intent['query']} | Department: {intent['department'] or 'UNIVERSITY'} | Type: {intent['document_type']} | Batch: {', '.join(intent['batches']) or 'none'}")
            print(f"Batch: {', '.join(result['detected_batch']) or 'none'} | Semester: {result['detected_semester'] or 'none'}")
            print(f"Candidate K: {result['candidate_k']} | Top K: {result['top_k']}")
            if result.get("retrieved_candidate_count") is not None:
                print(
                    f"Retrieved chunks: {result['retrieved_candidate_count']} | "
                    f"Reranked/selected chunks: {result.get('reranked_chunk_count', 0)} | "
                    f"Chunks sent to Ollama: {result.get('context', {}).get('context_chunks', 0)}"
                )
            print(f"Sources: {result['sources']}\nLatency: {result['latency']}\n\n{result['answer']}\n")
        except Exception as exc:
            print(f"Error: {exc}")
