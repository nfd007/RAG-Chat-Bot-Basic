"""Document loading - training step 01.

Teaching point: RAG quality is capped by text extraction quality. If the text that comes
out of a PDF is garbled, no embedding model or LLM can recover the facts. So the first
thing we do is LOOK at the extracted text before building anything on top of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".md", ".txt", ".pdf", ".docx")


@dataclass
class Document:
    """One source file, as clean text plus metadata that travels with every chunk."""

    text: str
    source: str  # file name, e.g. "hr_leave_policy.md" - shown in citations
    metadata: dict[str, str | int] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)


# ---------------------------------------------------------------------------
# Whitespace normalisation
# ---------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    """Make text from different file formats look the same.

    Why bother: PDFs and Word files produce stray spaces, tabs, non-breaking spaces and
    long runs of blank lines. Those waste chunk budget and, worse, break the chunker's
    "split on blank line = paragraph" rule. We keep single newlines (line structure) and
    double newlines (paragraph structure) because the chunker relies on them.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", " ")  # non-breaking space -> normal space
    text = re.sub(r"[ \t]+", " ", text)  # collapse runs of spaces/tabs
    text = re.sub(r" *\n *", "\n", text)  # trim spaces around line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line in a row
    return text.strip()


# ---------------------------------------------------------------------------
# Per-format readers. Each returns (raw_text, extra_metadata).
# ---------------------------------------------------------------------------

def _read_text_file(path: Path) -> tuple[str, dict[str, str | int]]:
    return path.read_text(encoding="utf-8", errors="replace"), {}


def _read_pdf(path: Path) -> tuple[str, dict[str, str | int]]:
    from pypdf import PdfReader  # imported lazily: only needed if you have PDFs

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    # A blank line between pages lets the chunker treat page breaks as paragraph breaks.
    return "\n\n".join(pages), {"page_count": len(reader.pages)}


def _read_docx(path: Path) -> tuple[str, dict[str, str | int]]:
    from docx import Document as DocxDocument  # imported lazily

    doc = DocxDocument(str(path))
    parts: list[str] = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        # Turn Word headings into Markdown headings so the chunker can split on them
        # exactly like it does for .md files.
        style_name = paragraph.style.name if paragraph.style is not None else ""
        if style_name.startswith("Heading"):
            level_text = style_name.replace("Heading", "").strip()
            level = int(level_text) if level_text.isdigit() else 2
            text = "#" * min(level + 1, 6) + " " + text
        parts.append(text)
    return "\n\n".join(parts), {"paragraph_count": len(parts)}


def _guess_title(text: str, path: Path) -> str:
    """First Markdown heading if there is one, otherwise the file name."""
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return path.stem.replace("_", " ")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_file(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        raw, extra = _read_text_file(path)
    elif suffix == ".pdf":
        raw, extra = _read_pdf(path)
    elif suffix == ".docx":
        raw, extra = _read_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {path.name}")

    text = normalize_whitespace(raw)
    metadata: dict[str, str | int] = {
        "source": path.name,
        "path": str(path),
        "file_type": suffix.lstrip("."),
        "title": _guess_title(text, path),
        "char_count": len(text),
    }
    metadata.update(extra)
    return Document(text=text, source=path.name, metadata=metadata)


def load_directory(directory: Path) -> list[Document]:
    """Load every supported file in `directory` (non-recursive), sorted by name."""
    if not directory.exists():
        raise FileNotFoundError(f"Data directory not found: {directory}")

    documents: list[Document] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            document = load_file(path)
            if document.text:  # skip empty files (e.g. scanned PDFs with no text layer)
                documents.append(document)
            else:
                print(f"  [warn] {path.name}: no extractable text, skipped")
    return documents
