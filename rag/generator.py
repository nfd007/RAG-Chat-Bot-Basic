"""Generation - training steps 06 and 07.

Teaching point: the "G" in RAG is a prompt. We paste the retrieved chunks into the prompt
as NUMBERED sources and instruct the model to (a) use only those sources, (b) cite them,
and (c) say a fixed refusal sentence when they are not enough. The fixed sentence matters:
it lets code (and our evaluation) detect a refusal reliably, instead of guessing whether
"I'm not sure, but..." counts as one.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from rag.config import Settings
from rag.embeddings import get_client
from rag.retriever import Hit, search_with_threshold

REFUSAL_MESSAGE = "I don't have that in the knowledge base."

SYSTEM_PROMPT = f"""You are a helpful assistant that answers questions about the company's internal documents.

Rules:
1. Answer ONLY using the numbered context passages provided in the user message. Do not use any outside knowledge.
2. After every fact you state, cite the passage it came from using its number in square brackets, e.g. [1] or [2][3].
3. If the context does not contain the information needed to answer, reply with exactly this sentence and nothing else:
{REFUSAL_MESSAGE}
4. Be concise. Quote numbers, names and thresholds exactly as they appear in the context."""

REWRITE_PROMPT = """You rewrite follow-up questions so they can be understood without the conversation.

Given the conversation so far and a follow-up question, rewrite the follow-up into a single standalone question that contains every detail needed to search for the answer.
If the follow-up is already standalone, return it unchanged.
Output ONLY the rewritten question - no explanation, no quotes, no answer."""


@dataclass
class RagAnswer:
    question: str
    text: str
    hits: list[Hit]
    messages: list[dict[str, str]] = field(default_factory=list)  # exactly what was sent

    @property
    def refused(self) -> bool:
        return is_refusal(self.text)


@dataclass
class ChatTurn:
    question: str
    answer: str


def is_refusal(text: str) -> bool:
    return REFUSAL_MESSAGE.lower().rstrip(".") in text.lower()


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_context(hits: list[Hit]) -> str:
    """Number each chunk so the model can cite it and we can map [n] back to a source."""
    blocks: list[str] = []
    for number, hit in enumerate(hits, start=1):
        blocks.append(f"[{number}] (source: {hit.source})\n{hit.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[Hit]) -> list[dict[str, str]]:
    user_message = (
        "Context passages:\n\n"
        f"{build_context(hits)}\n\n"
        "---\n"
        f"Question: {question}\n\n"
        "Answer using only the context passages above, with [n] citations."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def _retrieve(question: str, settings: Settings, hits: list[Hit] | None) -> list[Hit]:
    if hits is not None:
        return hits
    return search_with_threshold(question, settings, settings.top_k, settings.min_score)


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------

def answer(question: str, settings: Settings, hits: list[Hit] | None = None) -> RagAnswer:
    """One complete RAG turn: retrieve (unless hits are given), prompt, generate."""
    hits = _retrieve(question, settings, hits)

    # Nothing passed the similarity threshold: refuse WITHOUT calling the LLM.
    # Zero tokens spent, and zero chance the model invents an answer from nothing.
    if not hits:
        return RagAnswer(question=question, text=REFUSAL_MESSAGE, hits=[], messages=[])

    messages = build_messages(question, hits)
    client = get_client(settings.ollama_host)
    response = client.chat(
        model=settings.chat_model,
        messages=messages,
        # Low temperature: we want the most likely wording of facts already in the
        # context, not creative variation (which is where invented details come from).
        options={"temperature": settings.temperature},
    )
    return RagAnswer(question=question, text=response.message.content or "", hits=hits, messages=messages)


def answer_stream(question: str, settings: Settings, hits: list[Hit] | None = None) -> Iterator[str]:
    """Same as answer(), but yields the reply token by token as the model produces it.

    Streaming does not make generation faster, it makes it FEEL faster: the user sees the
    first words after a fraction of a second instead of waiting for the whole answer.
    """
    hits = _retrieve(question, settings, hits)
    if not hits:
        yield REFUSAL_MESSAGE
        return

    client = get_client(settings.ollama_host)
    stream = client.chat(
        model=settings.chat_model,
        messages=build_messages(question, hits),
        options={"temperature": settings.temperature},
        stream=True,
    )
    for part in stream:
        content = part.message.content
        if content:
            yield content


def rewrite_question(question: str, history: list[ChatTurn], settings: Settings) -> str:
    """Turn a follow-up like "and for P2?" into "What is the first response time for P2?".

    Why: the retriever only sees the text we embed. "and for P2?" has almost no meaning
    on its own, so its vector lands nowhere near the SLA table. Rewriting it with the
    conversation history gives the retriever a real question to search with.
    """
    if not history:
        return question

    # Only the last few turns matter for resolving "it", "that", "and for X".
    transcript_lines: list[str] = []
    for turn in history[-3:]:
        transcript_lines.append(f"User: {turn.question}")
        transcript_lines.append(f"Assistant: {turn.answer}")
    transcript = "\n".join(transcript_lines)

    client = get_client(settings.ollama_host)
    response = client.chat(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": REWRITE_PROMPT},
            {
                "role": "user",
                "content": f"Conversation:\n{transcript}\n\nFollow-up question: {question}\n\nStandalone question:",
            },
        ],
        options={"temperature": 0.0},
    )
    rewritten = (response.message.content or "").strip().strip('"').strip()
    return rewritten or question
