"""Embeddings - training step 03.

Teaching point: an embedding model turns text into a list of numbers (a vector) such that
texts with similar MEANING end up pointing in similar directions, even when they share no
words. "Production is down" and "the server crashed" score high; "annual leave" and
"server crashed" score low. Retrieval is nothing more than "find the stored vectors that
point most nearly in the same direction as the question's vector".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import ollama

from rag.config import Settings

# nomic-embed-text was trained with task prefixes. Documents and queries are embedded
# with DIFFERENT prefixes, which measurably improves retrieval for this model.
# Other embedding models do not expect them, so we only add them for nomic models.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


@lru_cache(maxsize=4)
def get_client(host: str) -> ollama.Client:
    return ollama.Client(host=host)


def _uses_nomic_prefixes(model: str) -> bool:
    return "nomic" in model.lower()


def embed_texts(texts: list[str], settings: Settings, batch_size: int = 32) -> list[list[float]]:
    """Embed many texts (document chunks). Sends them in batches.

    Why batch: one HTTP call per chunk is slow; one call with 10,000 chunks can time out
    or exhaust memory. A batch of a few dozen is a good middle ground.
    """
    client = get_client(settings.ollama_host)
    prefix = DOCUMENT_PREFIX if _uses_nomic_prefixes(settings.embedding_model) else ""

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = [prefix + text for text in texts[start : start + batch_size]]
        response = client.embed(model=settings.embedding_model, input=batch)
        vectors.extend(list(vector) for vector in response.embeddings)
    return vectors


def embed_query(question: str, settings: Settings) -> list[float]:
    """Embed a single user question. Must use the SAME model as the documents."""
    client = get_client(settings.ollama_host)
    prefix = QUERY_PREFIX if _uses_nomic_prefixes(settings.embedding_model) else ""
    response = client.embed(model=settings.embedding_model, input=prefix + question)
    return list(response.embeddings[0])


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """cos(angle between a and b) = (a . b) / (|a| * |b|).

    Written by hand for teaching - ChromaDB does exactly this internally.
    1.0 = same direction (same meaning), 0 = unrelated, negative = opposite.
    We compare DIRECTION, not length, so a long chunk and a short question can still match.
    """
    if len(a) != len(b):
        raise ValueError(f"Vectors have different dimensions: {len(a)} vs {len(b)}")

    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot_product += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b))


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@dataclass
class HealthReport:
    ok: bool
    messages: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)


def _model_is_installed(wanted: str, installed: list[str]) -> bool:
    """Ollama lists models with a tag ("nomic-embed-text:latest"); .env may omit it."""
    wanted_full = wanted if ":" in wanted else wanted + ":latest"
    return wanted in installed or wanted_full in installed


def _installed_model_names(client: ollama.Client) -> list[str]:
    response = client.list()
    names: list[str] = []
    for model in response.models:
        # The field is called "model" in current clients and "name" in older ones.
        name = getattr(model, "model", None) or getattr(model, "name", None)
        if name:
            names.append(str(name))
    return names


def health_check(settings: Settings) -> HealthReport:
    """Verify the Ollama server is reachable AND both models are pulled.

    Checking both up front turns a confusing stack trace in step 04 into a one-line fix.
    """
    report = HealthReport(ok=True)
    client = get_client(settings.ollama_host)

    try:
        installed = _installed_model_names(client)
    except Exception as error:  # connection refused, DNS, timeout...
        report.ok = False
        report.messages.append(f"Cannot reach Ollama at {settings.ollama_host} ({type(error).__name__}: {error})")
        report.fixes.append("Start Ollama: open the Ollama app, or run `ollama serve` in a separate terminal")
        report.fixes.append(f"Check OLLAMA_HOST in .env (currently {settings.ollama_host})")
        return report

    report.messages.append(f"Ollama is reachable at {settings.ollama_host} ({len(installed)} models installed)")

    for role, model in (("embedding", settings.embedding_model), ("chat", settings.chat_model)):
        if _model_is_installed(model, installed):
            report.messages.append(f"{role} model '{model}' is installed")
        else:
            report.ok = False
            report.messages.append(f"{role} model '{model}' is NOT installed")
            report.fixes.append(f"ollama pull {model}")

    return report
