"""Evaluation - run after step 07, and after every tuning change.

Teaching point: "it looked right in the demo" is not a measurement. We score three things
SEPARATELY, because they fail for different reasons and are fixed in different places:

  retrieval hit rate  - was the expected source in the top-k?      fix: chunking, embeddings, top_k
  answer match rate   - did the answer contain the key facts?      fix: prompt, model, context size
  refusal accuracy    - did unanswerable questions get refused?    fix: threshold, system prompt

Run:  python evaluation/evaluate.py
      python evaluation/evaluate.py --retrieval-only   (fast, no LLM calls)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Make `import rag` work when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.config import Settings, load_settings  # noqa: E402
from rag.embeddings import health_check  # noqa: E402
from rag.generator import answer, is_refusal  # noqa: E402
from rag.retriever import search  # noqa: E402

QUESTIONS_FILE = Path(__file__).resolve().parent / "questions.json"


@dataclass
class EvalCase:
    id: str
    question: str
    answerable: bool
    expected_source: str | None
    must_contain: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    case: EvalCase
    retrieved_sources: list[str]
    top_score: float
    retrieval_hit: bool | None = None  # None = not applicable (unanswerable case)
    answer_text: str = ""
    answer_match: bool | None = None
    refused: bool | None = None
    missing_keywords: list[str] = field(default_factory=list)


def load_cases(path: Path) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalCase(
            id=item["id"],
            question=item["question"],
            answerable=item["answerable"],
            expected_source=item.get("expected_source"),
            must_contain=item.get("must_contain", []),
        )
        for item in raw
    ]


def evaluate_case(case: EvalCase, settings: Settings, retrieval_only: bool) -> EvalResult:
    # Retrieval is scored on the raw top-k (no threshold) so we measure the retriever alone.
    hits = search(case.question, settings, settings.top_k)
    sources = [hit.source for hit in hits]
    result = EvalResult(
        case=case,
        retrieved_sources=sources,
        top_score=hits[0].score if hits else 0.0,
    )
    if case.answerable:
        result.retrieval_hit = case.expected_source in sources

    if retrieval_only:
        return result

    rag_answer = answer(case.question, settings)  # full pipeline, threshold included
    result.answer_text = rag_answer.text
    result.refused = is_refusal(rag_answer.text)

    if case.answerable:
        # Case-insensitive substring match: crude, but transparent and deterministic.
        lowered = rag_answer.text.lower()
        result.missing_keywords = [keyword for keyword in case.must_contain if keyword.lower() not in lowered]
        result.answer_match = not result.missing_keywords and not result.refused
    return result


def mark(value: bool | None) -> str:
    if value is None:
        return "  - "
    return "PASS" if value else "FAIL"


def rate(values: list[bool]) -> str:
    if not values:
        return "n/a"
    passed = sum(1 for value in values if value)
    return f"{passed}/{len(values)} = {passed / len(values):.0%}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answers against questions.json.")
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM; score retrieval only")
    parser.add_argument("--verbose", action="store_true", help="print every answer in full")
    args = parser.parse_args()

    settings = load_settings()
    print("=== Evaluation ===\n")
    print(
        f"embedding={settings.embedding_model}  chat={settings.chat_model}  "
        f"chunk={settings.chunk_size}/{settings.chunk_overlap}  top_k={settings.top_k}  "
        f"min_score={settings.min_score}\n"
    )

    report = health_check(settings)
    if not report.ok:
        for fix in report.fixes:
            print(f"fix: {fix}")
        return

    cases = load_cases(QUESTIONS_FILE)
    results: list[EvalResult] = []
    for case in cases:
        result = evaluate_case(case, settings, args.retrieval_only)
        results.append(result)

        line = f"[{case.id:<24}] retrieval {mark(result.retrieval_hit)}  top {result.top_score:.3f}"
        if not args.retrieval_only:
            if case.answerable:
                line += f"  answer {mark(result.answer_match)}"
                if result.missing_keywords:
                    line += f"  (missing: {', '.join(result.missing_keywords)})"
                if result.refused:
                    line += "  (refused)"
            else:
                line += f"  refusal {mark(result.refused)}"
        print(line)

        if args.verbose and result.answer_text:
            print(f"    Q: {case.question}")
            print(f"    A: {result.answer_text.strip()}\n")

    retrieval_values = [r.retrieval_hit for r in results if r.retrieval_hit is not None]
    answer_values = [r.answer_match for r in results if r.answer_match is not None]
    refusal_values = [bool(r.refused) for r in results if not r.case.answerable and r.refused is not None]

    print("\nSummary")
    print(f"  Retrieval hit rate : {rate(retrieval_values)}   (expected source in top {settings.top_k})")
    if args.retrieval_only:
        print("  Answer match rate  : skipped (--retrieval-only)")
        print("  Refusal accuracy   : skipped (--retrieval-only)")
    else:
        print(f"  Answer match rate  : {rate(answer_values)}   (all must_contain keywords present)")
        print(f"  Refusal accuracy   : {rate(refusal_values)}   (unanswerable questions refused)")

    # Show the score gap between answerable and unanswerable questions: it tells you
    # where a sensible min_score threshold lies for YOUR documents and embedding model.
    answerable_scores = [r.top_score for r in results if r.case.answerable]
    unanswerable_scores = [r.top_score for r in results if not r.case.answerable]
    if answerable_scores and unanswerable_scores:
        print(
            f"\n  Lowest top score, answerable   : {min(answerable_scores):.3f}\n"
            f"  Highest top score, unanswerable: {max(unanswerable_scores):.3f}\n"
            "  A min_score between these two would refuse unanswerable questions before the LLM runs."
        )

    print("\nChange one knob in .env, re-run step_04_index.py, and evaluate again.")


if __name__ == "__main__":
    main()
