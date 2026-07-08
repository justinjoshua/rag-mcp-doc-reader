"""Central configuration, read once from the environment.

Everything tunable lives here so ingest / retrieve / mcp_server / ask all agree.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional; env vars still work without it
    pass


@dataclass(frozen=True)
class Config:
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    db_path: str = os.getenv("DB_PATH", "./chroma_db")
    docs_dir: str = os.getenv("DOCS_DIR", "./docs")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "6"))
    collection_name: str = "documents"

    # Retrieval: "hybrid" fuses vector similarity with keyword matching (better
    # for names/IDs/exact terms); "vector" is pure embedding similarity.
    retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "hybrid")

    # Re-ranking: a cross-encoder re-scores the top candidates so the most
    # relevant passage is used first. Big accuracy win; adds a small model
    # (~80MB, downloaded once) and a little latency per query.
    rerank: bool = os.getenv("RERANK", "true").lower() in ("1", "true", "yes", "on")
    rerank_model: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Answer backend: how the reader turns retrieved passages into an answer.
    #   "gemini"     -> Google Gemini API (needs GEMINI_API_KEY)
    #   "extractive" -> no LLM, no key: returns the ranked passages (fallback)
    #   "ollama"     -> local LLM via Ollama (free, offline, no key)
    answer_backend: str = os.getenv("ANSWER_BACKEND", "extractive")

    # Gemini (used when answer_backend == "gemini")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # OCR: when a PDF has no text layer (scanned/image slides), render its pages
    # and have Gemini transcribe them. Needs a Gemini key. Capped to keep the
    # cost/latency of very large PDFs bounded.
    ocr: bool = os.getenv("OCR", "true").lower() in ("1", "true", "yes", "on")
    ocr_max_pages: int = int(os.getenv("OCR_MAX_PAGES", "60"))

    # Ollama (used when answer_backend == "ollama")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")

    @property
    def db_path_abs(self) -> str:
        return str(Path(self.db_path).expanduser().resolve())


CONFIG = Config()
