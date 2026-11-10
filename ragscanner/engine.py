"""TF-IDF retrieval engine.

The engine indexes text chunks and answers natural-language questions by
retrieving the most relevant chunks (cosine similarity over TF-IDF vectors) and
assembling an extractive answer. It runs fully offline with no external API
keys, which keeps the development environment reproducible.

If ``OPENAI_API_KEY`` is set the engine will additionally synthesize an answer
with the OpenAI API, but this is strictly optional; the extractive path is the
default and always available.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .scanner import Document, scan_path


@dataclass
class Chunk:
    """An indexed passage of text."""

    text: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A chunk returned from a query, with its similarity score."""

    text: str
    source: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class Answer:
    """The result of a query."""

    question: str
    answer: str
    sources: List[RetrievedChunk]
    generator: str = "extractive"


class RagEngine:
    """An in-memory TF-IDF retrieval engine with optional disk persistence."""

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def source_count(self) -> int:
        return len({c.source for c in self._chunks})

    def sources(self) -> List[str]:
        return sorted({c.source for c in self._chunks})

    def add_documents(self, documents: List[Document]) -> int:
        """Add scanned documents to the index and rebuild TF-IDF vectors."""

        added = 0
        for doc in documents:
            if not doc.text.strip():
                continue
            self._chunks.append(
                Chunk(text=doc.text, source=doc.source, metadata=dict(doc.metadata))
            )
            added += 1
        if added:
            self._rebuild()
        return added

    def ingest_path(
        self,
        path: str | Path,
        chunk_size: int = 800,
        overlap: int = 150,
    ) -> int:
        """Scan ``path`` from disk and add the resulting chunks to the index."""

        documents = scan_path(path, chunk_size=chunk_size, overlap=overlap)
        return self.add_documents(documents)

    def _rebuild(self) -> None:
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        corpus = [c.text for c in self._chunks]
        self._matrix = self._vectorizer.fit_transform(corpus)

    def retrieve(self, question: str, k: int = 4) -> List[RetrievedChunk]:
        """Return the top-``k`` chunks most similar to ``question``."""

        if not question or not question.strip():
            raise ValueError("question must not be empty")
        if self._vectorizer is None or self._matrix is None or not self._chunks:
            return []

        query_vec = self._vectorizer.transform([question])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        top_indices = np.argsort(scores)[::-1][:k]

        results: List[RetrievedChunk] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            chunk = self._chunks[idx]
            results.append(
                RetrievedChunk(
                    text=chunk.text,
                    source=chunk.source,
                    score=score,
                    metadata=dict(chunk.metadata),
                )
            )
        return results

    def answer(self, question: str, k: int = 4) -> Answer:
        """Answer ``question`` using retrieved context."""

        retrieved = self.retrieve(question, k=k)
        if not retrieved:
            return Answer(
                question=question,
                answer=(
                    "I could not find anything relevant in the indexed documents. "
                    "Try ingesting more material or rephrasing the question."
                ),
                sources=[],
                generator="extractive",
            )

        llm_answer = self._maybe_generate_with_llm(question, retrieved)
        if llm_answer is not None:
            return Answer(
                question=question,
                answer=llm_answer,
                sources=retrieved,
                generator="openai",
            )

        return Answer(
            question=question,
            answer=self._extractive_answer(question, retrieved),
            sources=retrieved,
            generator="extractive",
        )

    @staticmethod
    def _extractive_answer(question: str, retrieved: List[RetrievedChunk]) -> str:
        best = retrieved[0]
        summary = best.text.strip()
        if len(summary) > 600:
            summary = summary[:600].rsplit(" ", 1)[0] + "…"
        origin = Path(best.source).name
        return (
            f"Based on the most relevant passage (from {origin}, "
            f"similarity {best.score:.2f}):\n\n{summary}"
        )

    def _maybe_generate_with_llm(
        self, question: str, retrieved: List[RetrievedChunk]
    ) -> Optional[str]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        try:  # pragma: no cover - only exercised when a key is configured
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            context = "\n\n".join(
                f"[{i + 1}] {c.text}" for i, c in enumerate(retrieved)
            )
            completion = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer the question using only the provided context. "
                            "Cite passage numbers like [1]. If the context is "
                            "insufficient, say so."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}",
                    },
                ],
                temperature=0.1,
            )
            return completion.choices[0].message.content
        except Exception:
            # Never let an optional integration break the core retrieval path.
            return None

    def save(self, path: str | Path) -> None:
        """Persist the index to disk so the CLI and web server can share it."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"chunks": self._chunks}, fh)

    @classmethod
    def load(cls, path: str | Path) -> "RagEngine":
        """Load a previously saved index, or return an empty engine if missing."""

        engine = cls()
        path = Path(path)
        if not path.exists():
            return engine
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        engine._chunks = payload.get("chunks", [])
        if engine._chunks:
            engine._rebuild()
        return engine
