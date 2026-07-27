"""
Document ingestion: extract raw text from uploaded files and split into
overlapping chunks suitable for embedding + retrieval.
"""
import io
from typing import List

from pypdf import PdfReader

from app.config import settings


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Extract raw text from a .txt, .md, or .pdf file's bytes."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    elif lower.endswith((".txt", ".md")):
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type for '{filename}'. Use .txt, .md, or .pdf")


def chunk_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[str]:
    """
    Sliding-window character-based chunking with overlap, but snapped to
    sentence/paragraph boundaries where possible so chunks stay coherent
    (important for the grounding/critique step later — half-sentence
    chunks make faithfulness checking noisier).
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    # Normalize whitespace, keep paragraph breaks
    text = "\n".join(line.strip() for line in text.splitlines())
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)

        if end < n:
            # Try to snap to the nearest sentence boundary (., !, ?, newline)
            # within a small look-ahead window, so we don't cut mid-sentence.
            window = text[end:min(end + 150, n)]
            boundary = None
            for punct in [". ", "! ", "? ", "\n"]:
                idx = window.find(punct)
                if idx != -1:
                    boundary = idx + len(punct)
                    break
            if boundary:
                end = end + boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        start = max(end - chunk_overlap, start + 1)  # ensure forward progress

    return chunks
