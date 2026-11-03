"""Streamlit chat UI - the frontend for the RAG backend in rag/.

Teaching point: the UI contains NO RAG logic of its own. Every question goes through the
same functions the step scripts use (search_with_threshold, answer_stream), so what you see
here is exactly what steps 05-06 printed in the terminal - just easier to watch. The
sidebar exposes the retrieval knobs live, so attendees can change top-k or the threshold
and see the very next answer change.

Run:  streamlit run app.py      (from the project root, with .venv activated)
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

import streamlit as st

# Make `import rag` work no matter which folder streamlit was launched from.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag import store  # noqa: E402
from rag.chunker import chunk_documents  # noqa: E402
from rag.config import Settings, load_settings  # noqa: E402
from rag.embeddings import get_client, health_check  # noqa: E402
from rag.generator import REFUSAL_MESSAGE, answer_stream  # noqa: E402
from rag.loader import load_directory  # noqa: E402
from rag.retriever import Hit, search_with_threshold  # noqa: E402

PASSAGE_PREVIEW_CHARS = 700

st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="📚", layout="wide")


@dataclass
class ChatMessage:
    """One chat bubble. Assistant messages keep their retrieval results so that
    scrolling back through history still shows which passages each answer used."""

    role: str  # "user" or "assistant"
    content: str
    hits: list[Hit] = field(default_factory=list)
    show_passages: bool = True
    min_score: float = 0.0
    elapsed: float | None = None
    no_context: bool = False  # nothing cleared the threshold -> deliberate refusal
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers that talk to the backend. Each one catches errors so a dead Ollama
# server shows a message in the UI instead of a red traceback page.
# ---------------------------------------------------------------------------

def friendly_error(error: Exception, settings: Settings) -> str:
    text = str(error)
    if "connect" in text.lower() or isinstance(error, ConnectionError):
        return (
            f"Could not reach Ollama at {settings.ollama_host}. Is it running? "
            "Start the Ollama app (or `ollama serve`) and ask again."
        )
    if "not found" in text.lower():
        return f"Ollama reported a missing model: {text}. Pull it with `ollama pull <model>`."
    return f"{type(error).__name__}: {text}"


def list_chat_models(settings: Settings) -> list[str]:
    """Models Ollama has pulled, minus embedding models (they cannot chat)."""
    try:
        response = get_client(settings.ollama_host).list()
    except Exception:
        return [settings.chat_model]

    names: list[str] = []
    for model in response.models:
        name = getattr(model, "model", None) or getattr(model, "name", None)
        if name and "embed" not in str(name).lower():
            names.append(str(name))
    return sorted(names) or [settings.chat_model]


def default_model_index(models: list[str], wanted: str) -> int:
    """Pre-select CHAT_MODEL from .env, matching "llama3.2" to "llama3.2:latest" too."""
    for index, name in enumerate(models):
        if name == wanted or name == f"{wanted}:latest":
            return index
    return 0


def safe_count(settings: Settings) -> tuple[int | None, str | None]:
    try:
        return store.count(settings), None
    except Exception as error:  # e.g. collection built with a different embedding model
        return None, str(error)


def rebuild_index(settings: Settings) -> int:
    """Same pipeline as steps/step_04_index.py: load -> chunk -> reset -> embed + store."""
    documents = load_directory(settings.data_dir)
    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    store.reset(settings)
    return store.index_chunks(chunks, settings)


def stream_answer(question: str, settings: Settings, hits: list[Hit], errors: list[str]) -> Iterator[str]:
    """Wrap answer_stream so an Ollama failure MID-answer ends the stream cleanly.

    st.write_stream would otherwise crash the whole page and lose the partial answer.
    The error is passed back through the `errors` list and shown under the answer.
    """
    try:
        for token in answer_stream(question, settings, hits=hits):
            yield token
    except Exception as error:
        errors.append(friendly_error(error, settings))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_passages(hits: list[Hit]) -> None:
    label = f"Retrieved {len(hits)} passage(s)"
    with st.expander(label, expanded=False):
        for number, hit in enumerate(hits, start=1):
            st.markdown(f"**[{number}] {hit.source}** · similarity `{hit.score:.3f}`")
            preview = hit.text[:PASSAGE_PREVIEW_CHARS]
            if len(hit.text) > PASSAGE_PREVIEW_CHARS:
                preview += " ..."
            st.text(preview)
            if number < len(hits):
                st.divider()


def render_sources(hits: list[Hit]) -> None:
    lines = [f"[{number}] {hit.source} — {hit.score:.2f}" for number, hit in enumerate(hits, start=1)]
    st.caption("  \n".join(lines))  # two trailing spaces + newline = markdown line break


def render_no_context_info(min_score: float) -> None:
    st.info(
        f"No passage scored above the similarity threshold ({min_score:.2f}), so the LLM "
        "was not called. This is the bot correctly declining to guess - not an error. "
        "Lower the threshold in the sidebar to see what it would have retrieved."
    )


def render_footer(message: ChatMessage) -> None:
    if message.elapsed is None:
        return
    if message.hits:
        best = f"best similarity {message.hits[0].score:.2f}"
    else:
        best = "no passage above threshold"
    st.caption(f"{message.elapsed:.1f}s · {best}")


def render_message(message: ChatMessage) -> None:
    with st.chat_message(message.role):
        if message.role == "user":
            st.markdown(message.content)
            return
        # Passages go ABOVE the answer: you should see the evidence before the conclusion.
        if message.show_passages and message.hits:
            render_passages(message.hits)
        if message.content:
            st.markdown(message.content)
        if message.error:
            st.error(message.error)
        if message.no_context:
            render_no_context_info(message.min_score)
        if message.hits:
            render_sources(message.hits)
        render_footer(message)


def handle_question(question: str, settings: Settings, top_k: int, min_score: float, show_passages: bool) -> None:
    user_message = ChatMessage(role="user", content=question)
    st.session_state.messages.append(user_message)
    render_message(user_message)

    reply = ChatMessage(role="assistant", content="", show_passages=show_passages, min_score=min_score)

    with st.chat_message("assistant"):
        started = time.perf_counter()

        # 1) Retrieve first - with the sidebar's knobs, not the .env defaults.
        try:
            reply.hits = search_with_threshold(question, settings, top_k, min_score)
        except Exception as error:
            reply.error = friendly_error(error, settings)
            st.error(reply.error)
            st.session_state.messages.append(reply)
            return

        if show_passages and reply.hits:
            render_passages(reply.hits)

        if not reply.hits:
            # 2a) Nothing relevant: refuse without calling the LLM (same as rag.generator).
            reply.no_context = True
            reply.content = REFUSAL_MESSAGE
            st.markdown(reply.content)
            render_no_context_info(min_score)
        else:
            # 2b) Stream the answer token by token.
            errors: list[str] = []
            streamed = st.write_stream(stream_answer(question, settings, reply.hits, errors))
            reply.content = streamed if isinstance(streamed, str) else "".join(str(part) for part in streamed)
            if errors:
                reply.error = errors[0]
                st.error(reply.error)
            render_sources(reply.hits)

        reply.elapsed = time.perf_counter() - started
        render_footer(reply)

    st.session_state.messages.append(reply)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def main() -> None:
    base_settings = load_settings()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ----- Sidebar -----
    with st.sidebar:
        st.header("Status")
        report = health_check(base_settings)
        if report.ok:
            st.success("Ollama is running and both models are pulled.")
        else:
            st.error("\n\n".join(report.messages))
            for fix in report.fixes:
                st.code(fix, language="bash")

        chunk_count, count_error = safe_count(base_settings)
        st.metric("Chunks indexed", chunk_count if chunk_count is not None else "?")
        if count_error:
            st.error(count_error)
        elif chunk_count == 0:
            st.warning("The vector store is empty. Run `python steps/step_04_index.py` or use Rebuild below.")

        st.header("Retrieval")
        st.caption("Takes effect on the next question.")
        top_k = st.slider("Top-k passages", min_value=1, max_value=10, value=base_settings.top_k)
        min_score = st.slider(
            "Similarity threshold",
            min_value=0.0,
            max_value=0.6,
            value=min(base_settings.min_score, 0.6),
            step=0.05,
            help="Passages scoring below this are dropped. If none remain, the bot refuses without calling the LLM.",
        )
        show_passages = st.checkbox("Show retrieved passages", value=True)

        st.header("Model")
        models = list_chat_models(base_settings)
        chat_model = st.selectbox(
            "Chat model",
            models,
            index=default_model_index(models, base_settings.chat_model),
            help="Every chat model Ollama has pulled. `ollama pull llama3.1:8b` adds another.",
        )

        st.header("Session")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button("Rebuild index from data/", use_container_width=True):
            if not report.ok:
                st.error("Fix the Ollama status above first - indexing needs the embedding model.")
            else:
                with st.spinner("Loading, chunking and embedding data/ ..."):
                    try:
                        total = rebuild_index(base_settings)
                        st.session_state.flash = f"Rebuilt the index: {total} chunks."
                    except Exception as error:
                        st.session_state.flash_error = friendly_error(error, base_settings)
                st.rerun()  # refresh the chunk metric above

        if "flash" in st.session_state:
            st.success(st.session_state.pop("flash"))
        if "flash_error" in st.session_state:
            st.error(st.session_state.pop("flash_error"))

    # The settings for THIS question: .env values overridden by the sidebar model choice.
    settings = replace(base_settings, chat_model=chat_model)

    # ----- Chat area -----
    st.title("📚 RAG Knowledge Assistant")
    st.caption(f"Chat model: **{chat_model}** · Embeddings: **{settings.embedding_model}** · ChromaDB · runs fully local")

    for message in st.session_state.messages:
        render_message(message)

    store_ready = bool(chunk_count)
    if not store_ready:
        st.warning("Nothing is indexed yet, so there is nothing to search. Run `python steps/step_04_index.py`.")

    question = st.chat_input(
        "Ask about the documents in data/ ..." if store_ready else "Index the documents first (step_04_index.py)",
        disabled=not store_ready,
    )
    if question:
        handle_question(question, settings, top_k, min_score, show_passages)


main()
