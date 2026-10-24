"""First-stage BGE-M3 retrieval and CPU-safe BGE reranking for CampusGPT."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Iterable, Optional

import chromadb
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "database" / "chroma"
COLLECTION_NAME = "aiml_knowledge"
RERANK_BATCH_SIZE = int(os.getenv("CAMPUSGPT_RERANK_BATCH_SIZE", "8"))
# Cross-encoder relevance is determined primarily by the section heading and
# opening evidence. Capping enormous chunks prevents CPU reranking from doing
# expensive work on table tails that would otherwise be tokenizer-truncated.
RERANK_MAX_CHARS = int(os.getenv("CAMPUSGPT_RERANK_MAX_CHARS", "1800"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower().replace("&", " and ")).strip()


def expand_retrieval_query(query: str) -> str:
    """Expand campus abbreviations without injecting answer content."""
    expanded = query
    replacements = {
        r"\bhod\b": "HOD Head of Department",
        r"\baiml\b": "AIML AI&ML Artificial Intelligence and Machine Learning",
        r"\bra\b": "RA Robotics and Automation",
        r"\bentc\b": "ENTC E&TC Electronics and Telecommunication",
        r"\be\s*&\s*tc\b": "ENTC E&TC Electronics and Telecommunication",
        r"\bcs\b": "CS Computer Science Computer Science and Engineering",
        r"\bcivil\b": "CIVIL Civil Engineering",
        r"\bmech\b": "MECH Mechanical Engineering",
        r"\bodd sem\b": "odd semester",
        r"\beven sem\b": "even semester",
        r"\bsy\b": "SY second year",
        r"\bty\b": "TY third year",
        # Intent-preserving retrieval expansion. These are generic query
        # synonyms, not answer facts or source-name routing rules.
        r"\bplacements?\b": "placement recruitment internship job package stipend",
        r"\binternships?\b": "internship placement recruitment job stipend",
        r"\bpackages?\b": "package placement salary stipend recruitment",
        r"\btimetable\b": "timetable class schedule cohort day time slots",
        r"\btime table\b": "timetable class schedule cohort day time slots",
    }
    for pattern, replacement in replacements.items():
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
    return expanded


# Module singletons are shared by every FastAPI request.
embedding_model = SentenceTransformer("BAAI/bge-m3", device=DEVICE)
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=DEVICE)
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_collection(name=COLLECTION_NAME)


def _where(document_type: Optional[str] = None, batches: Optional[Iterable[str]] = None, semester: Optional[int] = None, department: Optional[str] = None) -> Optional[dict]:
    conditions = []
    if document_type:
        conditions.append({"document_type": document_type})
    normalized_batches = list(dict.fromkeys(batches or []))
    if len(normalized_batches) == 1:
        conditions.append({"batch": normalized_batches[0]})
    elif normalized_batches:
        conditions.append({"batch": {"$in": normalized_batches}})
    if semester is not None:
        conditions.append({"semester": int(semester)})
    if department:
        # University-scoped documents are shared only where that is explicit
        # in their metadata contract.  They must not become a general escape
        # hatch for a department-specific faculty, placement, or curriculum
        # query.
        allowed_departments = [department]
        if document_type in {"academic_calendar", "student_handbook"}:
            allowed_departments.append("UNIVERSITY")
        conditions.append({"department": {"$in": allowed_departments}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def retrieve_candidates(query: str, top_k: int = 12, document_type: Optional[str] = None, batch: Optional[str] = None, batches: Optional[Iterable[str]] = None, semester: Optional[int] = None, department: Optional[str] = None) -> dict:
    """Metadata filtering occurs in Chroma before cross-encoder inference."""
    requested_batches = list(batches or [])
    if batch:
        requested_batches.append(batch)
    embedding_started = time.perf_counter()
    embedding = embedding_model.encode(expand_retrieval_query(query), normalize_embeddings=True, show_progress_bar=False).tolist()
    query_embedding_ms = (time.perf_counter() - embedding_started) * 1000
    args = {"query_embeddings": [embedding], "n_results": top_k, "include": ["documents", "metadatas", "distances"]}
    where = _where(document_type, requested_batches, semester, department)
    if where is not None:
        args["where"] = where
    retrieval_started = time.perf_counter()
    result = collection.query(**args)
    chromadb_retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
    return {
        "documents": result["documents"][0],
        "metadatas": result["metadatas"][0],
        "distances": result["distances"][0],
        "query_embedding_ms": query_embedding_ms,
        "chromadb_retrieval_ms": chromadb_retrieval_ms,
    }


def retrieve_all_candidates(document_type: Optional[str] = None, batches: Optional[Iterable[str]] = None, semester: Optional[int] = None, department: Optional[str] = None) -> dict:
    """Return every already-indexed chunk matching an exact metadata scope.

    This deliberately has no embedding or semantic-ranking stage.  It is for
    broad, batch-scoped curriculum listings, where omitting a later semester
    is worse than including the entire selected programme structure.
    """
    where = _where(document_type, batches, semester, department)
    args = {"include": ["documents", "metadatas"]}
    if where is not None:
        args["where"] = where
    result = collection.get(**args)
    return {"documents": result["documents"], "metadatas": result["metadatas"]}


def _bonus(query: str, document: str) -> float:
    query_l, document_l = normalize_text(query), normalize_text(document)
    bonus = 0.0
    if ("hod" in query_l or "head of department" in query_l) and ("professor and head" in document_l or "head, ai" in document_l):
        bonus += 8.0
    semester_match = re.search(r"\bsemester\s*:?[ ]*([1-8])\b", query_l)
    if semester_match and re.search(rf"\bsemester\s*:?[ ]*{semester_match.group(1)}\b", document_l):
        bonus += 4.0
    if re.search(r"\b(subjects?|courses?|curriculum)\b", query_l) and re.search(r"\bsemester\s*:?[ ]*[1-8]\b", document_l):
        bonus += 3.0
    # These only order chunks after Chroma's metadata document-type filter;
    # they do not choose a source or inject a factual answer.
    if re.search(r"\b(placement|recruitment|internship|package|stipend)\b", query_l) and re.search(r"\b(placement|internship|package|stipend|role)\b", document_l):
        bonus += 2.0
    if re.search(r"\b(timetable|time table|schedule)\b", query_l) and re.search(r"\b(timetable|monday|tuesday|wednesday|thursday|friday|slot)\b", document_l):
        bonus += 2.0
    return bonus


def rerank_candidates(query: str, candidates: dict, top_k: int = 3) -> list[dict]:
    documents = candidates["documents"]
    if not documents:
        return []
    pairs = [(query, document[:RERANK_MAX_CHARS]) for document in documents]
    # CrossEncoder.predict uses inference/no-grad internally. Batching bounds
    # CPU memory and avoids the old implicit all-at-once behaviour.
    scores = reranker.predict(pairs, batch_size=RERANK_BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True)
    ranked = []
    for document, metadata, distance, score in zip(documents, candidates["metadatas"], candidates["distances"], scores):
        ranked.append({"document": document, "metadata": metadata, "chroma_distance": float(distance), "rerank_score": float(score), "final_score": float(score) + _bonus(query, document)})
    ranked.sort(key=lambda item: item["final_score"], reverse=True)
    return ranked[:top_k]


def search(**kwargs) -> list[dict]:
    candidate_k = kwargs.pop("candidate_k", 12)
    top_k = kwargs.pop("top_k", 3)
    candidates = retrieve_candidates(top_k=candidate_k, **kwargs)
    return rerank_candidates(kwargs["query"], candidates, top_k=top_k)
