"""Step 03 - Embeddings.

Teaching point: an embedding is just a list of floats, and similar MEANING gives similar
direction - even with zero shared keywords. That is why RAG beats keyword search for
questions phrased differently from the documents.

Run:  python steps/step_03_embed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `import rag` work when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.config import load_settings  # noqa: E402
from rag.embeddings import cosine_similarity, embed_texts, health_check  # noqa: E402

# Pairs (A,B) and (C,D) mean the same thing but share no meaningful keywords.
# E is unrelated to everything. Watch the matrix: A-B and C-D score high, E scores low.
SENTENCES: list[tuple[str, str]] = [
    ("A", "The production server crashed and nobody can log in."),
    ("B", "Our live system is down; customers are locked out of the app."),
    ("C", "How many vacation days do I get every year?"),
    ("D", "What is my yearly paid time off allowance?"),
    ("E", "The cat is sleeping on a warm windowsill."),
]


def main() -> None:
    settings = load_settings()
    print("=== Step 03: embeddings ===\n")

    report = health_check(settings)
    if not report.ok:
        print("Ollama is not ready:")
        for fix in report.fixes:
            print(f"  fix: {fix}")
        print("\nRun python steps/step_00_check_setup.py for details.")
        return

    # 1) What does a vector actually look like?
    sample = "Paternity leave is 10 working days."
    vector = embed_texts([sample], settings)[0]
    print(f'Text:        "{sample}"')
    print(f"Model:       {settings.embedding_model}")
    print(f"Dimensions:  {len(vector)}")
    print(f"First 8:     [{', '.join(f'{value:+.4f}' for value in vector[:8])}, ...]")
    print("Every chunk in the knowledge base becomes one of these.\n")

    # 2) Similarity matrix with our hand-written cosine_similarity().
    labels = [label for label, _ in SENTENCES]
    vectors = embed_texts([text for _, text in SENTENCES], settings)

    for label, text in SENTENCES:
        print(f"  {label}: {text}")
    print("\nCosine similarity matrix (1.0 = same direction):\n")
    print("      " + "".join(f"{label:>7}" for label in labels))
    for row_label, row_vector in zip(labels, vectors):
        cells = "".join(f"{cosine_similarity(row_vector, column_vector):>7.3f}" for column_vector in vectors)
        print(f"  {row_label:>2}  {cells}")

    print("\nA-B and C-D share almost no words but score high: that is semantic search.")
    print("Note: scores are relative. Compare pairs against each other, not against 1.0.")

    print("\nNext: python steps/step_04_index.py")


if __name__ == "__main__":
    main()
