"""Vector store - training step 04.

Teaching point: the vector store is just a table of (id, vector, text, metadata) rows plus
an index that finds nearest vectors fast. We compute the embeddings OURSELVES and hand
them to Chroma, so there is no hidden embedding model - the same Ollama model embeds
documents here and questions in the retriever. Mixing models would silently break search.
"""

from __future__ import annotations

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from rag.chunker import Chunk
from rag.config import Settings
from rag.embeddings import embed_texts


def get_client(settings: Settings) -> chromadb.ClientAPI:
    """PersistentClient stores everything in a folder on disk (CHROMA_DIR).

    Telemetry is switched off so nothing leaves the machine.
    """
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection(settings: Settings) -> Collection:
    """Open (or create) the collection.

    hnsw:space=cosine -> rank by angle between vectors, matching how embedding models are
    trained. Chroma's default is squared L2 distance, which gives different scores.

    embedding_function=None -> Chroma must NOT embed anything itself. Its default would
    download a different model (all-MiniLM, 384 dims) and mix it with our 768-dim vectors.

    We also record which embedding model built the collection, so we can detect the
    "changed EMBEDDING_MODEL but did not re-index" mistake with a clear error message.
    """
    client = get_client(settings)
    collection = client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine", "embedding_model": settings.embedding_model},
        embedding_function=None,
    )

    built_with = (collection.metadata or {}).get("embedding_model")
    if built_with and built_with != settings.embedding_model:
        raise RuntimeError(
            f"Collection '{settings.collection_name}' was built with embedding model "
            f"'{built_with}' but EMBEDDING_MODEL is '{settings.embedding_model}'. "
            "Vectors from different models are not comparable. "
            "Re-index with: python steps/step_04_index.py"
        )
    return collection


def reset(settings: Settings) -> None:
    """Delete the collection so the next index run starts from an empty store."""
    client = get_client(settings)
    existing = [c.name if hasattr(c, "name") else str(c) for c in client.list_collections()]
    if settings.collection_name in existing:
        client.delete_collection(settings.collection_name)


def count(settings: Settings) -> int:
    return get_collection(settings).count()


def index_chunks(chunks: list[Chunk], settings: Settings, batch_size: int = 64) -> int:
    """Embed chunks with Ollama and upsert them into Chroma, one batch at a time.

    upsert (not add): re-running indexing replaces chunks with the same id instead of
    failing or creating duplicates.
    """
    collection = get_collection(settings)
    total = len(chunks)

    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embed_texts([chunk.text for chunk in batch], settings)
        collection.upsert(
            ids=[chunk.id for chunk in batch],
            embeddings=vectors,
            documents=[chunk.text for chunk in batch],
            metadatas=[chunk.metadata for chunk in batch],
        )
        done = min(start + batch_size, total)
        print(f"  indexed {done}/{total} chunks")

    return collection.count()
