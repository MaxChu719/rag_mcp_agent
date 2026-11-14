"""Vector store access shared by ingestion and retrieval, so both use identical settings."""

from langchain.embeddings import init_embeddings
from langchain_chroma import Chroma

from config import Settings


def open_vector_store(settings: Settings) -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=init_embeddings(settings.embedding_model),
        persist_directory=str(settings.chroma_dir),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def knowledge_base_exists(settings: Settings) -> bool:
    return settings.chroma_dir.is_dir() and any(settings.chroma_dir.iterdir())
