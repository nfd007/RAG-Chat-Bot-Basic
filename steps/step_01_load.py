"""Step 01 - Load documents.

Teaching point: always look at the extracted text before indexing it. Garbage in,
garbage out: if a fact is mangled here, it can never be retrieved or answered later.

Run:  python steps/step_01_load.py               (preview 300 chars of each file)
      python steps/step_01_load.py --preview 0   (counts only)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `import rag` work when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.config import load_settings  # noqa: E402
from rag.loader import load_directory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Load documents and show the extracted text.")
    parser.add_argument("--preview", type=int, default=300, help="characters of text to show per file (0 = none)")
    args = parser.parse_args()

    settings = load_settings()
    print("=== Step 01: load documents ===\n")
    print(f"Data directory: {settings.data_dir}\n")

    documents = load_directory(settings.data_dir)
    if not documents:
        print("No .md/.txt/.pdf/.docx files found. Add some files to the data directory.")
        return

    for document in documents:
        print("-" * 78)
        print(f"{document.source}  ({document.metadata['file_type']}, {document.char_count} chars)")
        print(f"title: {document.metadata['title']}")
        if args.preview > 0:
            print()
            print(document.text[: args.preview])
            if document.char_count > args.preview:
                print(f"... [{document.char_count - args.preview} more chars]")
    print("-" * 78)

    # Summary table: a quick sanity check that nothing came out empty or suspiciously short.
    print("\nPer-file character counts")
    name_width = max(len(document.source) for document in documents)
    total = 0
    for document in documents:
        print(f"  {document.source:<{name_width}}  {document.char_count:>7,}")
        total += document.char_count
    print(f"  {'TOTAL':<{name_width}}  {total:>7,}   ({len(documents)} files)")

    print("\nNext: python steps/step_02_chunk.py")


if __name__ == "__main__":
    main()
