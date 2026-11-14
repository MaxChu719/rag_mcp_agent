"""Settings and process setup shared by the ingester, the MCP servers and the agent."""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent

# Loaded by absolute path: stdio MCP servers start with a minimal environment.
load_dotenv(PROJECT_DIR / ".env")

_TRUE_VALUES = {"true", "1", "yes", "on"}
_FALSE_VALUES = {"false", "0", "no", "off"}


def configure_logging(level: int = logging.INFO) -> None:
    """Log to stderr so stdout only carries the program's own output."""
    logging.basicConfig(
        stream=sys.stderr,
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for chatty_library in ("httpx", "httpx2", "mcp"):
        logging.getLogger(chatty_library).setLevel(logging.WARNING)


def _env_text(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _env_path(name: str, default: str) -> Path:
    path = Path(_env_text(name) or default).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def _env_number(
    name: str, default: float | None, cast: type[int] | type[float]
) -> int | float | None:
    raw = _env_text(name)
    if raw is None:
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a {cast.__name__}, got {raw!r}") from exc


def _env_bool_or_text(name: str) -> bool | str | None:
    raw = _env_text(name)
    if raw is None:
        return None
    if raw.lower() in _TRUE_VALUES:
        return True
    if raw.lower() in _FALSE_VALUES:
        return False
    return raw


@dataclass(frozen=True)
class Settings:
    chat_model: str
    chat_base_url: str | None
    chat_api_key: str | None
    chat_temperature: float | None
    chat_context_window: int | None
    chat_reasoning: bool | str | None
    embedding_model: str
    docs_dir: Path
    chroma_dir: Path
    collection_name: str
    tickets_file: Path
    chunk_size: int
    chunk_overlap: int
    min_relevance: float
    rag_server_url: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            chat_model=_env_text("CHAT_MODEL") or "anthropic:claude-sonnet-5-5",
            chat_base_url=_env_text("CHAT_BASE_URL"),
            chat_api_key=_env_text("CHAT_API_KEY"),
            chat_temperature=_env_number("CHAT_TEMPERATURE", None, float),
            chat_context_window=_env_number("CHAT_CONTEXT_WINDOW", None, int),
            chat_reasoning=_env_bool_or_text("CHAT_REASONING"),
            embedding_model=_env_text("EMBEDDING_MODEL")
            or "huggingface:sentence-transformers/all-MiniLM-L6-v2",
            docs_dir=_env_path("DOCS_DIR", "sample_docs"),
            chroma_dir=_env_path("CHROMA_DIR", "data/chroma"),
            collection_name=_env_text("COLLECTION_NAME") or "knowledge_base",
            tickets_file=_env_path("TICKETS_FILE", "data/tickets.json"),
            chunk_size=_env_number("CHUNK_SIZE", 800, int),
            chunk_overlap=_env_number("CHUNK_OVERLAP", 120, int),
            min_relevance=_env_number("MIN_RELEVANCE", 0.3, float),
            rag_server_url=_env_text("RAG_SERVER_URL"),
        )
        settings._validate()
        return settings

    def _validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be between 0 and CHUNK_SIZE")
        if not 0 <= self.min_relevance <= 1:
            raise ValueError("MIN_RELEVANCE must be between 0 and 1")
        if self.chat_context_window is not None and self.chat_context_window <= 0:
            raise ValueError("CHAT_CONTEXT_WINDOW must be positive")
        if self.chat_temperature is not None and not 0 <= self.chat_temperature <= 2:
            raise ValueError("CHAT_TEMPERATURE must be between 0 and 2")
