"""
Embedding model wrapper. Loaded once and reused for:
  1. Embedding chunks at ingestion time
  2. Embedding queries at retrieval time
  3. Computing the embedding-based grounding score during critique
    (i.e. "does this answer sentence semantically match any retrieved chunk?")
"""
from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()))
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        # Vectors are already normalized at encode time, so dot product == cosine similarity.
        return float(np.dot(a, b))


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel(settings.embedding_model)
