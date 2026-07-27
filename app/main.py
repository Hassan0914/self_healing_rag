"""
FastAPI entrypoint for the Self-Healing RAG service.

Endpoints:
  POST /ingest   - upload a document (.txt/.md/.pdf) into a corpus
  POST /ask      - ask a question against a corpus; runs the self-healing loop
  GET  /health   - liveness check
  GET  /corpora/{corpus_id}/exists - check whether a corpus has any data
"""
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.ingestion import chunk_text, extract_text
from app.models import AskRequest, AskResponse, IngestResponse
from app.pipeline import run_pipeline
from app.vector_store import get_vector_store

app = FastAPI(
    title="Self-Healing RAG API",
    description=(
        "A Retrieval-Augmented Generation service that critiques its own "
        "answers for hallucination and retries with a reformulated query "
        "before falling back to an honest 'I don't know' response."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    corpus_id: str = Form(default=settings.default_collection),
):
    try:
        file_bytes = await file.read()
        text = extract_text(file.filename, file_bytes)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No extractable text found in file.")

        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Document produced zero chunks.")

        vector_store = get_vector_store()
        num_added = vector_store.add_chunks(corpus_id, chunks, source=file.filename)

        return IngestResponse(
            corpus_id=corpus_id,
            filename=file.filename,
            num_chunks=num_added,
            message=f"Ingested {num_added} chunks into corpus '{corpus_id}'.",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    vector_store = get_vector_store()
    if not vector_store.corpus_exists(request.corpus_id):
        raise HTTPException(
            status_code=404,
            detail=f"Corpus '{request.corpus_id}' is empty or does not exist. Ingest documents first.",
        )

    try:
        return run_pipeline(
            question=request.question,
            corpus_id=request.corpus_id,
            top_k=request.top_k,
            max_retries=request.max_retries,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")


@app.get("/corpora/{corpus_id}/exists")
def corpus_exists(corpus_id: str):
    vector_store = get_vector_store()
    return {"corpus_id": corpus_id, "exists": vector_store.corpus_exists(corpus_id)}
