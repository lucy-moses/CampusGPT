"""
CampusGPT FastAPI backend.

Lives in: Campus.../backend/main.py
Pipeline lives in: Campus.../scripts/generate_answer.py
    answer_question(query, document_type=None, batch=None, top_k=None, candidate_k=None)
    -> {
         "answer": str,
         "sources": [{"source": ..., "document_type": ...}, ...],
         "query_classification": "simple" | "broad",
         "candidate_k": int,
         "top_k": int,
         "latency": {
             "query_embedding_ms": float,
             "chromadb_retrieval_ms": float,
             "bge_reranking_ms": float,
             "llm_generation_ms": float,
             "total_ms": float,
         },
       }

Wraps the local RAG pipeline (BGE-M3 -> ChromaDB -> BGE reranker -> Qwen via
Ollama) in an HTTP API so the React frontend (or curl/Postman) can call it.

top_k / candidate_k are chosen ADAPTIVELY inside answer_question() based on
the question text (simple factual vs. broad subjects/curriculum/"all"/"list"
questions), unless explicitly overridden in the request body.

IMPORTANT: Ollama must be running locally and the configured Qwen model must
be installed before launching uvicorn.

Test with:
    curl http://127.0.0.1:8000/health
    curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"Who is the HOD of the AIML department?\"}"
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# scripts/ is a sibling folder of backend/, not a package on the path by
# default, so add it manually before importing generate_answer.
# ---------------------------------------------------------------------------
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))

from generate_answer import answer_question  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("campusgpt")

_startup_state = {"ready": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: warming up RAG pipeline (loading BGE-M3 + reranker)...")
    start = time.time()
    try:
        # Forces model loading (BGE-M3, bge-reranker-v2-m3) to happen now, at
        # server startup, instead of on the first real user's request. This
        # import of generate_answer (which imports reranker) already loaded
        # both models and connected to ChromaDB exactly once when this
        # module was first imported above -- this warmup call just also
        # exercises the full request path once so any lazy initialization
        # inside the libraries themselves (e.g. first-call JIT warmup) is
        # paid for here, not on a real user's first request.
        warmup_result = answer_question(query="warmup")
        _startup_state["ready"] = True
        logger.info(
            "Warmup complete in %.1fs (pipeline latency: %s)",
            time.time() - start,
            warmup_result.get("latency"),
        )
    except Exception:
        logger.exception("Warmup failed — pipeline may still work per-request, continuing startup")
    yield
    logger.info("Shutting down CampusGPT API")


app = FastAPI(title="CampusGPT API", version="1.1.0", lifespan=lifespan)

# Allow your React dev server. Adjust/add origins once you deploy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",  # Vite default
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The student's question")
    document_type: Optional[str] = Field(
        None, description="Optional metadata filter, e.g. 'faculty', 'timetable', 'curriculum'"
    )
    batch: Optional[str] = Field(None, description="Optional batch filter, e.g. '2024-28'")
    top_k: Optional[int] = Field(
        None, description="Override the number of chunks sent to local Qwen. Leave unset to size adaptively."
    )
    candidate_k: Optional[int] = Field(
        None, description="Override the number of raw candidates retrieved before reranking. Leave unset to size adaptively."
    )


class Source(BaseModel):
    source: str
    document_type: str


class Latency(BaseModel):
    # All values are milliseconds. Stage intervals are not additive: the total
    # also contains uninstrumented Python/route overhead.
    query_planning_ms: float = 0.0
    query_embedding_ms: float = 0.0
    chromadb_retrieval_ms: float = 0.0
    bge_reranking_ms: float = 0.0
    context_construction_ms: float = 0.0
    llm_generation_ms: float = 0.0
    total_ms: float = 0.0


class GenerationMetrics(BaseModel):
    model: Optional[str] = None
    ollama_request_ms: Optional[float] = None
    prompt_token_count: Optional[int] = None
    generated_token_count: Optional[int] = None
    prompt_evaluation_ms: Optional[float] = None
    generation_duration_ms: Optional[float] = None
    ollama_total_duration_ms: Optional[float] = None
    generated_tokens_per_second: Optional[float] = None


class IntentDebug(BaseModel):
    query: str
    department: Optional[str] = None
    document_type: str
    batches: List[str] = []
    semester: Optional[int] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    query_classification: str
    document_type: Optional[str] = None
    detected_department: Optional[str] = None
    detected_batch: List[str] = []
    detected_semester: Optional[int] = None
    default_curriculum_batch_used: bool = False
    semester_metadata_fallback: bool = False
    candidate_k: int
    top_k: int
    latency: Latency
    generation_metrics: Optional[GenerationMetrics] = None
    intent_plans: List[IntentDebug] = []

@app.get("/")
def root():
    return {
        "name": "CampusGPT API",
        "status": "running",
        "version": "1.1.0",
        "endpoints": {
            "health": "/health",
            "ask": "/ask",
            "docs": "/docs",
        },
    }

@app.get("/health")
def health():
    return {"status": "ok" if _startup_state["ready"] else "starting"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = answer_question(
            query=question,
            document_type=payload.document_type,
            batch=payload.batch,
            top_k=payload.top_k,
            candidate_k=payload.candidate_k,
        )
    except Exception as e:
        logger.exception("Pipeline error while answering: %s", question)
        raise HTTPException(status_code=500, detail=f"Failed to generate an answer: {e}")

    logger.info(
        "question=%r classification=%s candidate_k=%s top_k=%s latency=%s",
        question,
        result.get("query_classification"),
        result.get("candidate_k"),
        result.get("top_k"),
        result.get("latency"),
    )

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        query_classification=result["query_classification"],
        document_type=result.get("document_type"),
        detected_department=result.get("detected_department"),
        detected_batch=result.get("detected_batch", []),
        detected_semester=result.get("detected_semester"),
        default_curriculum_batch_used=result.get("default_curriculum_batch_used", False),
        semester_metadata_fallback=result.get("semester_metadata_fallback", False),
        candidate_k=result["candidate_k"],
        top_k=result["top_k"],
        latency=result["latency"],
        generation_metrics=result.get("generation_metrics"),
    )
