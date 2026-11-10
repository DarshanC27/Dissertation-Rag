import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Point the app at an isolated corpus and index for the test.
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "seed.md").write_text(
        "TF-IDF retrieval uses cosine similarity over term frequency vectors. " * 20,
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_DOCS_DIR", str(docs_dir))
    monkeypatch.setenv("RAG_INDEX_PATH", str(tmp_path / "index.pkl"))

    import ragscanner.app as app_module

    importlib.reload(app_module)
    return TestClient(app_module.create_app())


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_stats_seeded(client):
    res = client.get("/stats")
    assert res.status_code == 200
    body = res.json()
    assert body["chunk_count"] > 0
    assert "seed.md" in body["sources"]


def test_index_page_renders(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Dissertation-Rag Scanner" in res.text


def test_ask_returns_answer_and_sources(client):
    res = client.post("/ask", json={"question": "What does TF-IDF use?"})
    assert res.status_code == 200
    body = res.json()
    assert body["sources"]
    assert "cosine" in body["answer"].lower() or "tf-idf" in body["answer"].lower()


def test_ask_requires_question(client):
    res = client.post("/ask", json={"question": "   "})
    assert res.status_code == 400


def test_ingest_uploaded_file(client):
    content = b"Spectral clustering groups nodes using graph Laplacian eigenvectors. " * 20
    res = client.post(
        "/ingest",
        files={"file": ("upload.txt", content, "text/plain")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ingested_chunks"] > 0

    # The newly ingested content should now be retrievable.
    ask = client.post("/ask", json={"question": "What does spectral clustering use?"})
    assert ask.status_code == 200
    assert "spectral" in ask.json()["answer"].lower() or "laplacian" in ask.json()["answer"].lower()


def test_ingest_rejects_unsupported(client):
    res = client.post(
        "/ingest",
        files={"file": ("bad.docx", b"nope", "application/octet-stream")},
    )
    assert res.status_code == 400
