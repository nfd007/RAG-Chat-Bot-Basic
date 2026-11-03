"""Step 02 - Chunk documents.

Teaching point: chunk size and overlap are the first tuning knobs of any RAG system.
Run this three times live and compare how the SAME document gets cut:

    python steps/step_02_chunk.py 300 0       tiny chunks, no overlap: facts lose context
    python steps/step_02_chunk.py 800 120     the default: roughly one section per chunk
    python steps/step_02_chunk.py 2000 200    huge chunks: whole documents, blurry vectors

Without arguments, CHUNK_SIZE and CHUNK_OVERLAP come from .env.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `import rag` work when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.chunker import Chunk, chunk_documents  # noqa: E402
from rag.config import load_settings  # noqa: E402
from rag.loader import load_directory  # noqa: E402


def print_chunk(chunk: Chunk, max_chars: int) -> None:
    print(f"--- {chunk.id}  ({len(chunk.text)} chars)")
    text = chunk.text if len(chunk.text) <= max_chars else chunk.text[:max_chars] + " ..."
    for line in text.splitlines():
        print(f"    | {line}")


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Split documents into chunks and inspect them.")
    parser.add_argument("size", type=int, nargs="?", default=settings.chunk_size, help="chunk size in characters")
    parser.add_argument("overlap", type=int, nargs="?", default=settings.chunk_overlap, help="overlap in characters")
    parser.add_argument("--file", default=None, help="only show chunks of this file (default: first file)")
    parser.add_argument("--show", type=int, default=3, help="how many chunks to print in full")
    parser.add_argument("--max-chars", type=int, default=900, help="truncate printed chunks to this length")
    args = parser.parse_args()

    print("=== Step 02: chunk documents ===\n")
    print(f"chunk size = {args.size} chars, overlap = {args.overlap} chars\n")

    documents = load_directory(settings.data_dir)
    chunks = chunk_documents(documents, args.size, args.overlap)

    # Statistics per file: how many pieces did each document become?
    print("Chunks per file")
    name_width = max(len(document.source) for document in documents)
    for document in documents:
        own = [chunk for chunk in chunks if chunk.source == document.source]
        lengths = [len(chunk.text) for chunk in own]
        print(
            f"  {document.source:<{name_width}}  {len(own):>3} chunks"
            f"   min {min(lengths):>4}  avg {sum(lengths) // len(lengths):>4}  max {max(lengths):>4} chars"
        )
    print(f"  {'TOTAL':<{name_width}}  {len(chunks):>3} chunks\n")

    # Show a few chunks from one file so we can SEE where the boundaries fell
    # and what the overlap repeats.
    target = args.file or documents[0].source
    sample = [chunk for chunk in chunks if chunk.source == target][: args.show]
    print(f"First {len(sample)} chunks of {target}:\n")
    for chunk in sample:
        print_chunk(chunk, args.max_chars)
        print()

    if args.overlap > 0 and len(sample) >= 2:
        print("Look at the start of chunk 0001: it repeats the end of chunk 0000. That is the overlap.")
    print("Try: python steps/step_02_chunk.py 300 0   and   python steps/step_02_chunk.py 2000 200")

    print("\nNext: python steps/step_03_embed.py   (needs Ollama running)")


if __name__ == "__main__":
    main()
