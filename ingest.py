"""Rebuild the knowledge-base index from a folder of Markdown and text files."""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings, configure_logging
from knowledge_base import open_vector_store

SUPPORTED_SUFFIXES = {".md", ".txt"}
UPSERT_BATCH_SIZE = 500

logger = logging.getLogger("ingest")


def load_documents(docs_dir: Path) -> list[Document]:
    documents = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Skipping unreadable file %s: %s", path, exc)
            continue
        if not text:
            logger.warning("Skipping empty file %s", path)
            continue
        source = path.relative_to(docs_dir).as_posix()
        documents.append(Document(page_content=text, metadata={"source": source}))
    return documents


def split_into_chunks(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split documents and number chunks per source, giving each a stable citation id."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    next_index: defaultdict[str, int] = defaultdict(int)
    for chunk in chunks:
        source = chunk.metadata["source"]
        chunk.metadata["chunk"] = next_index[source]
        next_index[source] += 1
    return chunks


def citation_id(chunk: Document) -> str:
    return f"{chunk.metadata['source']}#{chunk.metadata['chunk']}"


def rebuild_index(settings: Settings, docs_dir: Path) -> int:
    documents = load_documents(docs_dir)
    if not documents:
        raise ValueError(f"No {sorted(SUPPORTED_SUFFIXES)} files found in {docs_dir}")

    chunks = split_into_chunks(documents, settings.chunk_size, settings.chunk_overlap)
    store = open_vector_store(settings)
    store.reset_collection()
    for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch = chunks[start : start + UPSERT_BATCH_SIZE]
        store.add_documents(batch, ids=[citation_id(chunk) for chunk in batch])
    return len(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", type=Path, help="defaults to DOCS_DIR from .env")
    args = parser.parse_args()
    configure_logging()

    try:
        settings = Settings.from_env()
        docs_dir = (args.docs_dir or settings.docs_dir).resolve()
        if not docs_dir.is_dir():
            raise ValueError(f"Docs folder not found: {docs_dir}")
        chunk_count = rebuild_index(settings, docs_dir)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Indexed %d chunks from %s", chunk_count, docs_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
