from pathlib import Path

import pytest

from ragscanner.engine import RagEngine
from ragscanner.scanner import Document


def _sample_docs():
    return [
        Document(source="rag.md", text="RAG retrieves passages then generates an answer."),
        Document(source="chunking.md", text="Chunking splits documents into overlapping windows."),
        Document(source="eval.md", text="Precision at k measures retrieval quality for RAG."),
    ]


def test_add_documents_and_counts():
    engine = RagEngine()
    added = engine.add_documents(_sample_docs())
    assert added == 3
    assert engine.chunk_count == 3
    assert engine.source_count == 3


def test_retrieve_returns_relevant_chunk_first():
    engine = RagEngine()
    engine.add_documents(_sample_docs())
    results = engine.retrieve("How does chunking split documents?", k=3)
    assert results
    assert Path(results[0].source).name == "chunking.md"
    assert results[0].score > 0


def test_answer_extractive_grounded():
    engine = RagEngine()
    engine.add_documents(_sample_docs())
    answer = engine.answer("What does precision at k measure?", k=2)
    assert answer.generator == "extractive"
    assert answer.sources
    assert Path(answer.sources[0].source).name == "eval.md"
    assert "precision" in answer.answer.lower()


def test_answer_with_no_index():
    engine = RagEngine()
    answer = engine.answer("anything?")
    assert answer.sources == []
    assert "could not find" in answer.answer.lower()


def test_retrieve_empty_question_raises():
    engine = RagEngine()
    engine.add_documents(_sample_docs())
    with pytest.raises(ValueError):
        engine.retrieve("   ")


def test_save_and_load_roundtrip(tmp_path: Path):
    engine = RagEngine()
    engine.add_documents(_sample_docs())
    index_path = tmp_path / "index.pkl"
    engine.save(index_path)

    loaded = RagEngine.load(index_path)
    assert loaded.chunk_count == 3
    results = loaded.retrieve("retrieval quality", k=1)
    assert results


def test_load_missing_returns_empty(tmp_path: Path):
    loaded = RagEngine.load(tmp_path / "missing.pkl")
    assert loaded.chunk_count == 0
