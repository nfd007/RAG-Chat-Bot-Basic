"""Chunking - training step 02.

Teaching point: we embed CHUNKS, not documents. One vector per whole document would blur
every topic in it into a single average meaning, and the LLM context window could not
hold many documents anyway. Chunk size is a trade-off:

  * too small (e.g. 300) -> a fact gets separated from the heading/sentence that explains it
  * too large (e.g. 2000) -> one chunk covers several topics, its vector gets "blurry",
                            and fewer distinct facts fit in the prompt

The splitter is "recursive": it tries the most meaningful boundary first (a heading),
and only falls back to weaker boundaries (paragraph, line, sentence, word, character)
for pieces that are still too big. That keeps related text together whenever possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.loader import Document

# Ordered from strongest to weakest boundary. The splitter uses the first separator that
# actually occurs in the text, and recurses with the rest for pieces that are too long.
SEPARATORS: list[str] = [
    "\n## ",  # Markdown level-2 heading: a new section
    "\n### ",  # Markdown level-3 heading: a sub-section
    "\n\n",  # blank line: a new paragraph
    "\n",  # single line break: list items, table rows
    ". ",  # sentence end
    " ",  # word
    "",  # last resort: hard cut by characters
]


@dataclass
class Chunk:
    """A piece of a Document small enough to embed and to paste into a prompt."""

    id: str  # stable id, e.g. "hr_leave_policy.md::chunk-0003" - re-indexing overwrites
    text: str
    source: str
    chunk_index: int
    metadata: dict[str, str | int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def _split_keeping_separator(text: str, separator: str) -> list[str]:
    """Split on `separator` without losing it, so joining the parts gives back `text`.

    A heading marker belongs to the text AFTER it ("## Sick leave" starts a section).
    Every other separator belongs to the text BEFORE it (a sentence keeps its full stop).
    """
    parts = text.split(separator)
    is_heading = separator.lstrip("\n").startswith("#")
    if is_heading:
        result = [parts[0]] + [separator + part for part in parts[1:]]
    else:
        result = [part + separator for part in parts[:-1]] + [parts[-1]]
    return [part for part in result if part]


def _recursive_split(text: str, separators: list[str], max_len: int) -> list[str]:
    """Split `text` into pieces of at most `max_len` characters."""
    if len(text) <= max_len:
        return [text]

    # Pick the strongest separator present in this text.
    separator = ""
    remaining: list[str] = []
    for index, candidate in enumerate(separators):
        if candidate == "" or candidate in text:
            separator = candidate
            remaining = separators[index + 1 :]
            break

    if separator == "":
        # Nothing left to split on: cut every max_len characters.
        return [text[start : start + max_len] for start in range(0, len(text), max_len)]

    # Greedily pack consecutive parts into pieces that stay under max_len.
    # A part that is too big on its own is split again with the weaker separators.
    pieces: list[str] = []
    current = ""
    for part in _split_keeping_separator(text, separator):
        if len(part) > max_len:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_recursive_split(part, remaining, max_len))
        elif len(current) + len(part) <= max_len:
            current += part
        else:
            pieces.append(current)
            current = part
    if current:
        pieces.append(current)
    return pieces


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------

def _overlap_tail(text: str, overlap: int) -> str:
    """The last ~`overlap` characters of `text`, starting at a word boundary.

    Cutting at a word boundary matters: a chunk starting with "ance is 16 weeks" embeds
    worse and reads worse in a citation than one starting with "leave is 16 weeks".
    """
    window = text[-overlap:]
    first_space = window.find(" ")
    first_newline = window.find("\n")
    candidates = [position for position in (first_space, first_newline) if position != -1]
    if not candidates:
        return ""  # one giant "word": better no overlap than half a word
    return window[min(candidates) + 1 :].strip()


def _add_overlap(pieces: list[str], overlap: int) -> list[str]:
    """Prefix every piece with the tail of the previous one.

    Why overlap exists: a boundary can fall in the middle of an idea, e.g. the sentence
    naming "P1" ends one chunk and the "15 minutes" response time starts the next. With
    overlap, the second chunk still carries the context it needs to be found and
    understood on its own. The cost is some duplicated text in the index.
    """
    if overlap <= 0 or len(pieces) < 2:
        return pieces

    result = [pieces[0]]
    for previous, current in zip(pieces, pieces[1:]):
        tail = _overlap_tail(previous, overlap)
        if tail:
            # A newline (not a space) keeps a heading at the start of `current` on its own line.
            result.append(tail + "\n" + current)
        else:
            result.append(current)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks of at most `chunk_size` characters, including overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    # Reserve room for the overlap up front so the final chunks never exceed chunk_size.
    budget = chunk_size - chunk_overlap
    pieces = _recursive_split(text, SEPARATORS, budget)
    pieces = [piece.strip() for piece in pieces if piece.strip()]
    return _add_overlap(pieces, chunk_overlap)


def chunk_document(document: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    texts = split_text(document.text, chunk_size, chunk_overlap)
    chunks: list[Chunk] = []
    for index, text in enumerate(texts):
        # Chroma metadata values must be str/int/float/bool, so keep this flat.
        metadata: dict[str, str | int] = {
            "source": document.source,
            "title": str(document.metadata.get("title", document.source)),
            "file_type": str(document.metadata.get("file_type", "")),
            "chunk_index": index,
            "chunk_count": len(texts),
            "char_count": len(text),
        }
        chunks.append(
            Chunk(
                id=f"{document.source}::chunk-{index:04d}",
                text=text,
                source=document.source,
                chunk_index=index,
                metadata=metadata,
            )
        )
    return chunks


def chunk_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size, chunk_overlap))
    return chunks
