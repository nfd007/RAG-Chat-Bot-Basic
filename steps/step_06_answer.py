"""Step 06 - One full RAG turn.

Teaching point: there is no magic between retrieval and the answer - it is a prompt.
--show-prompt prints the exact system and user messages sent to the model, so you can
see the numbered context the [1][2] citations refer to.

Run:  python steps/step_06_answer.py "How long is paternity leave?"
      python steps/step_06_answer.py "Who approves a USD 500 refund?" --show-prompt
      python steps/step_06_answer.py "Who won the 2022 World Cup?"   (should refuse)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make `import rag` work when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.config import load_settings  # noqa: E402
from rag.generator import answer_stream, build_messages  # noqa: E402
from rag.retriever import search_with_threshold  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer one question with RAG.")
    parser.add_argument("question", nargs="*", help="the question (default: a demo question)")
    parser.add_argument("--show-prompt", action="store_true", help="print the exact messages sent to the model")
    args = parser.parse_args()

    settings = load_settings()
    question = " ".join(args.question) or "What is the first response time for a P1 ticket?"

    print("=== Step 06: retrieve + generate ===\n")
    print(f"Q: {question}\n")

    # 1) Retrieve. Done here (not inside answer_stream) so we can show sources first.
    hits = search_with_threshold(question, settings, settings.top_k, settings.min_score)
    print(f"Retrieved {len(hits)} chunks above min_score={settings.min_score}:")
    for number, hit in enumerate(hits, start=1):
        print(f"  [{number}] {hit.chunk_id}  (score {hit.score:.3f})")
    print()

    # 2) Optionally show the prompt - the single most useful debugging view in RAG.
    if args.show_prompt:
        if hits:
            for message in build_messages(question, hits):
                print(f"----- {message['role'].upper()} MESSAGE " + "-" * 50)
                print(message["content"])
            print("-" * 78 + "\n")
        else:
            print("(No prompt: nothing passed the threshold, so the LLM is not called.)\n")

    # 3) Generate, streaming tokens as they arrive.
    print(f"A ({settings.chat_model}, temperature={settings.temperature}):")
    started = time.perf_counter()
    for token in answer_stream(question, settings, hits=hits):
        print(token, end="", flush=True)
    print(f"\n\n({time.perf_counter() - started:.1f}s)")

    print("\nNext: python steps/step_07_chat.py")


if __name__ == "__main__":
    main()
