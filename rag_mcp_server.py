"""MCP server that exposes the document knowledge base as retrieval tools."""

import argparse
import logging
from functools import cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from config import Settings
from knowledge_base import knowledge_base_exists, open_vector_store

MAX_RESULTS = 10
NO_MATCH_MESSAGE = (
    "No relevant passages found. Tell the user the knowledge base does not cover this."
)

logger = logging.getLogger("rag-server")
settings = Settings.from_env()
mcp = FastMCP(
    "knowledge-base",
    instructions="Search Acme Co's internal documents. Cite results by their [source#chunk] id.",
    log_level="WARNING",
)


@cache
def get_vector_store() -> Chroma:
    if not knowledge_base_exists(settings):
        raise RuntimeError("Knowledge base is empty. Run `python ingest.py` first.")
    logger.info("Loading vector store from %s", settings.chroma_dir)
    return open_vector_store(settings)


def format_passage(document: Document, relevance: float) -> str:
    source = document.metadata.get("source", "unknown")
    chunk = document.metadata.get("chunk", "?")
    return f"[{source}#{chunk}] (relevance {relevance:.2f})\n{document.page_content}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def search_knowledge_base(query: str, max_results: int = 4) -> str:
    """Search Acme Co's internal documents (policies, shipping, product FAQs).

    Use this before answering any question about company rules or procedures.
    Each passage starts with a [source#chunk] id that must be cited in the answer.
    """
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    max_results = max(1, min(max_results, MAX_RESULTS))

    hits = get_vector_store().similarity_search_with_relevance_scores(query, k=max_results)
    relevant = [(doc, score) for doc, score in hits if score >= settings.min_relevance]
    logger.info("query=%r hits=%d relevant=%d", query, len(hits), len(relevant))

    if not relevant:
        return NO_MATCH_MESSAGE
    return "\n\n".join(format_passage(doc, score) for doc, score in relevant)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def list_knowledge_sources() -> str:
    """List the documents in the knowledge base, to judge what it can answer."""
    metadatas = get_vector_store().get(include=["metadatas"])["metadatas"]
    sources = sorted({meta["source"] for meta in metadatas if meta and "source" in meta})
    return "\n".join(sources) if sources else "The knowledge base is empty."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http", action="store_true", help="serve over streamable HTTP")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.http:
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
