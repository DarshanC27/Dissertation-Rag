"""Document scanning and chunking.

The scanner reads supported files from disk, extracts plain text, and splits the
text into overlapping chunks suitable for retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}


@dataclass
class Document:
    """A single scanned document."""

    source: str
    text: str
    metadata: dict = field(default_factory=dict)


def extract_text(file_path: str | Path) -> str:
    """Extract plain text from a supported file.

    PDF parsing is delegated to :mod:`pypdf`; text-like files are read directly.
    """

    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf_text(path)

    if suffix in SUPPORTED_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")

    raise ValueError(f"Unsupported file type: {suffix!r} ({path})")


def _extract_pdf_text(path: Path) -> str:
    # Imported lazily so the rest of the package works even if pypdf is absent.
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of spaces/tabs but preserve paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[str]:
    """Split ``text`` into overlapping character windows on word boundaries.

    ``chunk_size`` and ``overlap`` are measured in characters. Splitting on word
    boundaries keeps chunks human-readable, which matters because the extractive
    answer returns chunk text verbatim.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    normalized = _normalize_whitespace(text)
    if not normalized:
        return []

    words = normalized.split(" ")
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for word in words:
        # +1 accounts for the joining space.
        addition = len(word) + (1 if current else 0)
        if current_len + addition > chunk_size and current:
            chunks.append(" ".join(current))
            current, current_len = _apply_overlap(current, overlap)
        current.append(word)
        current_len += addition

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]


def _apply_overlap(words: List[str], overlap: int) -> tuple[List[str], int]:
    """Return the trailing words of ``words`` that fit within ``overlap`` chars."""

    if overlap == 0:
        return [], 0

    kept: List[str] = []
    length = 0
    for word in reversed(words):
        addition = len(word) + (1 if kept else 0)
        if length + addition > overlap:
            break
        kept.insert(0, word)
        length += addition
    return kept, length


def scan_path(
    path: str | Path,
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[Document]:
    """Scan a file or directory and return chunked :class:`Document` objects.

    Each returned :class:`Document` corresponds to one chunk so that retrieval
    can operate on fine-grained passages while still tracking the source file.
    """

    root = Path(path)
    files: Iterable[Path]

    if root.is_dir():
        files = sorted(
            p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )
    elif root.is_file():
        files = [root]
    else:
        raise FileNotFoundError(f"Path does not exist: {root}")

    documents: List[Document] = []
    for file_path in files:
        try:
            raw = extract_text(file_path)
        except ValueError:
            continue
        for index, chunk in enumerate(chunk_text(raw, chunk_size, overlap)):
            documents.append(
                Document(
                    source=str(file_path),
                    text=chunk,
                    metadata={"chunk_index": index, "file_name": file_path.name},
                )
            )
    return documents
