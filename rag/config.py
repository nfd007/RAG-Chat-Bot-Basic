"""Configuration - used by every step.

Teaching point: every knob that changes RAG behaviour (models, chunk size, overlap,
top-k, similarity threshold) lives in ONE place, read from .env. During the session we
change a value in .env and re-run a step instead of editing code, so attendees can see
that the pipeline is a set of tunable parameters, not magic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

from dotenv import load_dotenv

# rag/config.py -> rag/ -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # --- Ollama (embeddings + generation, both local) ---
    ollama_host: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    chat_model: str = "llama3.2:3b"
    temperature: float = 0.1

    # --- Vector store ---
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    collection_name: str = "knowledge_base"

    # --- Ingestion ---
    data_dir: Path = PROJECT_ROOT / "data"
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Retrieval ---
    top_k: int = 4
    min_score: float = 0.25

    def describe(self) -> str:
        """Human-readable dump of every setting, used at the top of each step."""
        width = max(len(f.name) for f in fields(self))
        lines = ["Settings (from .env, falling back to defaults):"]
        for f in fields(self):
            lines.append(f"  {f.name:<{width}} = {getattr(self, f.name)}")
        return "\n".join(lines)


def _resolve_path(raw: str) -> Path:
    """Relative paths in .env are relative to the project root, not the current dir."""
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_settings() -> Settings:
    """Read .env (if present) and build a Settings object.

    Values already set in the real environment win over .env, which is python-dotenv's
    default behaviour and lets you override a single value from the shell.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    defaults = Settings()

    return Settings(
        ollama_host=os.getenv("OLLAMA_HOST", defaults.ollama_host),
        embedding_model=os.getenv("EMBEDDING_MODEL", defaults.embedding_model),
        chat_model=os.getenv("CHAT_MODEL", defaults.chat_model),
        temperature=float(os.getenv("TEMPERATURE", defaults.temperature)),
        chroma_dir=_resolve_path(os.getenv("CHROMA_DIR", str(defaults.chroma_dir))),
        collection_name=os.getenv("COLLECTION_NAME", defaults.collection_name),
        data_dir=_resolve_path(os.getenv("DATA_DIR", str(defaults.data_dir))),
        chunk_size=int(os.getenv("CHUNK_SIZE", defaults.chunk_size)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", defaults.chunk_overlap)),
        top_k=int(os.getenv("TOP_K", defaults.top_k)),
        min_score=float(os.getenv("MIN_SCORE", defaults.min_score)),
    )
