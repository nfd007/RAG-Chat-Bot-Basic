"""Retrieval - training step 05.

Teaching point: debug retrieval BEFORE you write a single prompt. If the right chunk is
not in the top-k, no prompt engineering can fix the answer - the LLM never sees the fact.
So we look at raw scores first, with no LLM involved.

The similarity threshold is our first line of defence against hallucination: when even
the best match is weak, it is better to give the LLM NO context (and refuse) than to give
it loosely related text it will happily stitch into a confident wrong answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.config import Settings
from rag.embeddings import embed_query
from rag.store import get_collection


@dataclass
class Hit:
    """One retrieved chunk and how well it matched the question."""

    text: str
    source: str
    score: float  # cosine similarity, higher is better (roughly 0..1 for text)
    chunk_id: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


def distance_to_similarity(distance: float) -> float:
    """Chroma returns cosine DISTANCE (0 = identical, 2 = opposite).

    distance = 1 - cosine_similarity, so similarity = 1 - distance. We convert because
    "higher is better" is far easier to reason about when setting a threshold.
    """
    return 1.0 - distance


def search(question: str, settings: Settings, top_k: int | None = None) -> list[Hit]:
    """Return the top_k most similar chunks, best first. No filtering."""
    k = top_k if top_k is not None else settings.top_k
    collection = get_collection(settings)
    if collection.count() == 0:
        raise RuntimeError("The vector store is empty. Run: python steps/step_04_index.py")

    query_vector = embed_query(question, settings)
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma returns one list per query; we sent one query, so take index [0].
    ids = result["ids"][0]
    documents = result["documents"][0] if result["documents"] else []
    metadatas = result["metadatas"][0] if result["metadatas"] else []
    distances = result["distances"][0] if result["distances"] else []

    hits: list[Hit] = []
    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        metadata = dict(metadata or {})
        hits.append(
            Hit(
                text=text or "",
                source=str(metadata.get("source", "unknown")),
                score=distance_to_similarity(distance),
                chunk_id=chunk_id,
                metadata=metadata,
            )
        )
    return hits


def search_with_threshold(
    question: str,
    settings: Settings,
    top_k: int | None = None,
    min_score: float = 0.25,
) -> list[Hit]:
    """Like search(), but drop hits whose similarity is below min_score.

    An empty result is a meaningful answer: "the knowledge base does not cover this".
    """
    hits = search(question, settings, top_k)
    return [hit for hit in hits if hit.score >= min_score]
