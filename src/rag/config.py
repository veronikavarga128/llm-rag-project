"""Konfiguráció: környezeti változók és a .env fájl beolvasása."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "data" / "raw" / "corpus.jsonl"

load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    qdrant_url: str
    collection: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    llm_base_url: str
    llm_api_key: str
    llm_model: str


def get_settings() -> Settings:
    env = os.environ.get
    return Settings(
        qdrant_url=env("QDRANT_URL", "http://localhost:6333"),
        collection=env("QDRANT_COLLECTION", "corpus"),
        embedding_model=env("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
        chunk_size=int(env("CHUNK_SIZE", "500")),
        chunk_overlap=int(env("CHUNK_OVERLAP", "100")),
        top_k=int(env("TOP_K", "5")),
        llm_base_url=env("LLM_BASE_URL", "http://localhost:11434/v1"),
        llm_api_key=env("LLM_API_KEY", "ollama"),
        llm_model=env("LLM_MODEL", "llama3.1:8b"),
    )