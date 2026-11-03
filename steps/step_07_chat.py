"""Step 07 - Multi-turn chat.

Teaching point: a follow-up like "and for P2?" is meaningless to a retriever on its own.
Before retrieving, we ask the LLM to rewrite it into a standalone question using the
conversation history, then run the normal single-turn RAG pipeline on that.

Try this conversation:
    > What is the first response time for a P1 ticket?
    > and for P2?
    > who approves refunds above that limit?   (hmm - "that limit" needs history too)

Commands:  /sources  show the chunks used for the last answer
           /reset    forget the conversation
           /exit     quit

Run:  python steps/step_07_chat.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `import rag` work when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.config import load_settings  # noqa: E402
from rag.embeddings import health_check  # noqa: E402
from rag.generator import ChatTurn, answer_stream, rewrite_question  # noqa: E402
from rag.retriever import Hit, search_with_threshold  # noqa: E402


def print_sources(hits: list[Hit]) -> None:
    if not hits:
        print("  (no sources - the last answer used no context)")
    for number, hit in enumerate(hits, start=1):
        print(f"  [{number}] {hit.chunk_id}  (score {hit.score:.3f})")


def main() -> None:
    settings = load_settings()
    print("=== Step 07: multi-turn chat ===")
    print(f"model={settings.chat_model}  top_k={settings.top_k}  min_score={settings.min_score}")
    print("Commands: /sources  /reset  /exit\n")

    report = health_check(settings)
    if not report.ok:
        for fix in report.fixes:
            print(f"fix: {fix}")
        return

    history: list[ChatTurn] = []
    last_hits: list[Hit] = []

    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question in ("/exit", "/quit"):
            break
        if question == "/reset":
            history.clear()
            last_hits = []
            print("(conversation cleared)\n")
            continue
        if question == "/sources":
            print_sources(last_hits)
            print()
            continue

        # 1) Make the question standalone. Shown on screen so attendees can see it happen.
        standalone = rewrite_question(question, history, settings)
        if standalone != question:
            print(f"  (searching for: {standalone})")

        # 2) Retrieve with the standalone question, 3) answer it.
        last_hits = search_with_threshold(standalone, settings, settings.top_k, settings.min_score)
        print("bot> ", end="", flush=True)
        parts: list[str] = []
        for token in answer_stream(standalone, settings, hits=last_hits):
            print(token, end="", flush=True)
            parts.append(token)
        print("\n")

        # Store the ORIGINAL question so the next rewrite sees what the user actually said.
        history.append(ChatTurn(question=question, answer="".join(parts)))

    print("Bye.")
    print("\nNext: python evaluation/evaluate.py")


if __name__ == "__main__":
    main()
