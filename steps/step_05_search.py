"""Step 05 - Retrieval only (no LLM).

Teaching point: debug retrieval before prompting. If the right chunk is not in this list,
the answer will be wrong no matter how good the prompt or the model is. Scores below the
threshold are flagged: those chunks would be thrown away in step 06.

Run:  python steps/step_05_search.py "How long is paternity leave?"
      python steps/step_05_search.py "What is the Wi-Fi password?" --top-k 6
      python steps/step_05_search.py          (runs a few built-in demo questions)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `import rag` work when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.config import Settings, load_settings  # noqa: E402
from rag.retriever import search  # noqa: E402

DEMO_QUESTIONS: list[str] = [
    "How fast must we respond to a production outage?",  # no shared keywords with "P1 ... 15 minutes"
    "Do I need a doctor's note when I'm sick?",  # document says "medical certificate"
    "What is the office Wi-Fi password?",  # not in the knowledge base at all
]


def show_results(question: str, settings: Settings, top_k: int, min_score: float, preview: int) -> None:
    print("=" * 78)
    print(f"Q: {question}\n")
    hits = search(question, settings, top_k)

    for rank, hit in enumerate(hits, start=1):
        flag = "" if hit.score >= min_score else "   <-- BELOW THRESHOLD, would be dropped"
        print(f"  #{rank}  score {hit.score:.3f}  {hit.chunk_id}{flag}")
        snippet = hit.text.replace("\n", " ")
        if len(snippet) > preview:
            snippet = snippet[:preview] + " ..."
        print(f"       {snippet}\n")

    kept = [hit for hit in hits if hit.score >= min_score]
    print(f"  {len(kept)} of {len(hits)} hits pass min_score={min_score}")
    if not kept:
        print("  -> step 06 would refuse WITHOUT calling the LLM.")
    print()


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Search the vector store without an LLM.")
    parser.add_argument("question", nargs="*", help="question to search for (default: demo questions)")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--min-score", type=float, default=settings.min_score)
    parser.add_argument("--preview", type=int, default=220, help="characters of each chunk to show")
    args = parser.parse_args()

    print("=== Step 05: retrieval only ===\n")
    print(f"top_k={args.top_k}  min_score={args.min_score}  (similarity = 1 - cosine distance)\n")

    questions = [" ".join(args.question)] if args.question else DEMO_QUESTIONS
    for question in questions:
        show_results(question, settings, args.top_k, args.min_score, args.preview)

    print("Compare the scores of answerable vs unanswerable questions: where would YOU set the threshold?")
    print('\nNext: python steps/step_06_answer.py "How long is paternity leave?" --show-prompt')


if __name__ == "__main__":
    main()
