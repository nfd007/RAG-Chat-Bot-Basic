"""Step 04 - Build the index.

Teaching point: indexing is a batch pipeline you run ahead of time:
load -> chunk -> embed -> store. Question time only embeds the question.
By default we wipe the collection first, because after changing CHUNK_SIZE or the
embedding model, old chunks would otherwise linger next to the new ones.

Run:  python steps/step_04_index.py          (rebuild from scratch)
      python steps/step_04_index.py --keep   (upsert into the existing collection)
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

from rag import store  # noqa: E402
from rag.chunker import chunk_documents  # noqa: E402
from rag.config import load_settings  # noqa: E402
from rag.embeddings import health_check  # noqa: E402
from rag.loader import load_directory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Load, chunk, embed and store the knowledge base.")
    parser.add_argument("--keep", action="store_true", help="keep the existing collection instead of resetting it")
    args = parser.parse_args()

    settings = load_settings()
    print("=== Step 04: build the index ===\n")
    print(settings.describe())
    print()

    report = health_check(settings)
    if not report.ok:
        for message in report.messages:
            print(f"  {message}")
        for fix in report.fixes:
            print(f"  fix: {fix}")
        return

    started = time.perf_counter()

    print("[1/4] Loading documents")
    documents = load_directory(settings.data_dir)
    print(f"  {len(documents)} documents\n")

    print(f"[2/4] Chunking (size={settings.chunk_size}, overlap={settings.chunk_overlap})")
    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    print(f"  {len(chunks)} chunks\n")

    if args.keep:
        print("[3/4] Keeping existing collection (--keep)")
    else:
        print(f"[3/4] Resetting collection '{settings.collection_name}'")
        store.reset(settings)
    print()

    print(f"[4/4] Embedding with {settings.embedding_model} and storing in {settings.chroma_dir}")
    total = store.index_chunks(chunks, settings)

    elapsed = time.perf_counter() - started
    print(f"\nDone in {elapsed:.1f}s. Collection '{settings.collection_name}' now holds {total} chunks.")

    print('\nNext: python steps/step_05_search.py "How long is paternity leave?"')


if __name__ == "__main__":
    main()
