from pathlib import Path

import pytest

from ragscanner.scanner import chunk_text, extract_text, scan_path


def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 220 for c in chunks)  # small slack for word boundaries
    # Consecutive chunks should share some overlapping words.
    first_tail = chunks[0].split(" ")[-1]
    assert first_tail in chunks[1]


def test_chunk_text_empty_returns_empty():
    assert chunk_text("   \n\n  ") == []


def test_chunk_text_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=10, overlap=10)


def test_extract_text_txt(tmp_path: Path):
    f = tmp_path / "note.txt"
    f.write_text("Hello dissertation world", encoding="utf-8")
    assert "dissertation" in extract_text(f)


def test_extract_text_unsupported(tmp_path: Path):
    f = tmp_path / "note.docx"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        extract_text(f)


def test_scan_path_directory(tmp_path: Path):
    (tmp_path / "a.md").write_text("# Title\n" + "alpha " * 300, encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta " * 300, encoding="utf-8")
    (tmp_path / "ignore.docx").write_text("skip me", encoding="utf-8")

    docs = scan_path(tmp_path, chunk_size=200, overlap=40)
    sources = {Path(d.source).name for d in docs}
    assert "a.md" in sources
    assert "b.txt" in sources
    assert "ignore.docx" not in sources
    assert all(d.text.strip() for d in docs)
