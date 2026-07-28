"""
Centralized configuration for the Self-Healing RAG pipeline.
All tunables live here so the rest of the codebase never hardcodes values.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Groq ---
    groq_api_key: str = ""
    groq_generation_model: str = "llama-3.3-70b-versatile"
    groq_critic_model: str = "llama-3.3-70b-versatile"
    groq_reformulate_model: str = "llama-3.1-8b-instant"  # cheap/fast model for query rewriting

    # --- Embeddings ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Chunking ---
    chunk_size: int = 800       # characters per chunk
    chunk_overlap: int = 120    # characters of overlap between consecutive chunks

    # --- Retrieval ---
    top_k: int = 4

    # --- Self-healing loop ---
    max_retries: int = 2                     # number of *extra* attempts after the first
    faithfulness_threshold: float = 0.55     # min combined faithfulness score to accept an answer

    # --- Storage ---
    chroma_persist_dir: str = "./chroma_db"
    default_collection: str = "default_corpus"


settings = Settings()
