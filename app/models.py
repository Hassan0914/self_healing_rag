"""
Pydantic schemas for API requests/responses and internal data structures.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


# ---------- Ingestion ----------

class IngestResponse(BaseModel):
    corpus_id: str
    filename: str
    num_chunks: int
    message: str


# ---------- Retrieval ----------

class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    distance: float  # lower = more similar (Chroma uses distance, not similarity)


# ---------- Critique ----------

class Critique(BaseModel):
    verdict: str = Field(..., description="'grounded' or 'hallucinated'")
    llm_faithfulness_score: float = Field(..., ge=0, le=1)
    embedding_grounding_score: float = Field(..., ge=0, le=1)
    combined_score: float = Field(..., ge=0, le=1)
    unsupported_claims: List[str] = []
    reasoning: str = ""


# ---------- Per-attempt trace (this is what makes the self-healing loop demoable) ----------

class Attempt(BaseModel):
    attempt_number: int
    query_used: str
    retrieved_chunks: List[RetrievedChunk]
    answer: str
    critique: Critique
    accepted: bool


# ---------- Ask ----------

class AskRequest(BaseModel):
    question: str
    corpus_id: str = "default_corpus"
    top_k: Optional[int] = None
    max_retries: Optional[int] = None


class AskResponse(BaseModel):
    question: str
    final_answer: str
    was_fallback: bool  # True if the system gave up and returned "insufficient information"
    total_attempts: int
    attempts: List[Attempt]
