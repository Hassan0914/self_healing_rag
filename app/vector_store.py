"""
Thin wrapper around ChromaDB. We manage embeddings ourselves (via
app.embeddings) rather than letting Chroma auto-embed, so that the exact
same embedding model/vectors are reused for the grounding-score check
during critique.
"""
import uuid
from functools import lru_cache
from typing import List

import chromadb

from app.config import settings
from app.embeddings import get_embedding_model
from app.models import RetrievedChunk


class VectorStore:
    def __init__(self, persist_dir: str):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedder = get_embedding_model()

    def _get_collection(self, corpus_id: str):
        return self.client.get_or_create_collection(
            name=corpus_id,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, corpus_id: str, chunks: List[str], source: str) -> int:
        if not chunks:
            return 0
        collection = self._get_collection(corpus_id)
        embeddings = self.embedder.encode(chunks)
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": source} for _ in chunks]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
        return len(chunks)

    def query(self, corpus_id: str, query_text: str, top_k: int) -> List[RetrievedChunk]:
        collection = self._get_collection(corpus_id)
        if collection.count() == 0:
            return []

        query_embedding = self.embedder.encode_one(query_text)
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, collection.count()),
        )

        chunks = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for i in range(len(ids)):
            chunks.append(
                RetrievedChunk(
                    chunk_id=ids[i],
                    text=docs[i],
                    source=metas[i].get("source", "unknown"),
                    distance=float(dists[i]),
                )
            )
        return chunks

    def corpus_exists(self, corpus_id: str) -> bool:
        try:
            collection = self._get_collection(corpus_id)
            return collection.count() > 0
        except Exception:
            return False


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore(settings.chroma_persist_dir)
