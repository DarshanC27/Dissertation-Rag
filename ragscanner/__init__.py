"""Dissertation-Rag: a document-scanning Retrieval-Augmented-Generation engine.

The package exposes three building blocks:

- :mod:`ragscanner.scanner` turns raw files (``.txt``, ``.md``, ``.pdf``) into
  normalized, chunked text.
- :mod:`ragscanner.engine` indexes those chunks and answers questions with
  TF-IDF retrieval (no external API keys required).
- :mod:`ragscanner.app` serves a small web UI and JSON API on top of the engine.
"""

from .engine import RagEngine, Chunk, RetrievedChunk, Answer
from .scanner import Document, scan_path, extract_text, chunk_text

__all__ = [
    "RagEngine",
    "Chunk",
    "RetrievedChunk",
    "Answer",
    "Document",
    "scan_path",
    "extract_text",
    "chunk_text",
]

__version__ = "0.1.0"
