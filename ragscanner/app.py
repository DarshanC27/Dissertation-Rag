"""FastAPI web application exposing the RAG scanner.

Endpoints:

- ``GET  /``        HTML UI for asking questions and ingesting documents.
- ``GET  /health``  Liveness probe.
- ``GET  /stats``   Index statistics (chunk/source counts).
- ``POST /ask``     JSON question answering.
- ``POST /ingest``  Upload a document (``.txt``/``.md``/``.pdf``) into the index.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from .engine import RagEngine
from .scanner import scan_path

DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", "data"))
INDEX_PATH = Path(os.environ.get("RAG_INDEX_PATH", DATA_DIR / "index.pkl"))
DOCS_DIR = Path(os.environ.get("RAG_DOCS_DIR", DATA_DIR / "documents"))
TEMPLATES_DIR = Path(__file__).parent / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def build_engine() -> RagEngine:
    """Load a persisted index if present, otherwise scan the sample corpus."""

    engine = RagEngine.load(INDEX_PATH)
    if engine.chunk_count == 0 and DOCS_DIR.exists():
        engine.ingest_path(DOCS_DIR)
        if engine.chunk_count:
            engine.save(INDEX_PATH)
    return engine


def create_app() -> FastAPI:
    app = FastAPI(title="Dissertation-Rag Scanner", version="0.1.0")
    app.state.engine = build_engine()

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        engine: RagEngine = app.state.engine
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "chunk_count": engine.chunk_count,
                "source_count": engine.source_count,
                "sources": [Path(s).name for s in engine.sources()],
            },
        )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/stats")
    async def stats() -> dict:
        engine: RagEngine = app.state.engine
        return {
            "chunk_count": engine.chunk_count,
            "source_count": engine.source_count,
            "sources": [Path(s).name for s in engine.sources()],
        }

    @app.post("/ask")
    async def ask(payload: dict) -> JSONResponse:
        question = (payload or {}).get("question", "").strip()
        top_k = int((payload or {}).get("k", 4))
        if not question:
            raise HTTPException(status_code=400, detail="question is required")

        engine: RagEngine = app.state.engine
        result = engine.answer(question, k=top_k)
        return JSONResponse(
            {
                "question": result.question,
                "answer": result.answer,
                "generator": result.generator,
                "sources": [
                    {
                        "source": Path(s.source).name,
                        "score": round(s.score, 4),
                        "text": s.text,
                    }
                    for s in result.sources
                ],
            }
        )

    @app.post("/ingest")
    async def ingest(file: UploadFile = File(...), persist: bool = Form(True)) -> JSONResponse:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".txt", ".md", ".markdown", ".pdf"}:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {suffix or 'unknown'}",
            )

        engine: RagEngine = app.state.engine
        original_name = Path(file.filename or f"upload{suffix}").name
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Preserve the original filename so retrieved sources are readable.
            tmp_path = Path(tmp_dir) / original_name
            tmp_path.write_bytes(await file.read())
            documents = scan_path(tmp_path)
            for doc in documents:
                doc.source = original_name
            added = engine.add_documents(documents)

        if persist and added:
            engine.save(INDEX_PATH)

        return JSONResponse(
            {
                "ingested_chunks": added,
                "file_name": file.filename,
                "chunk_count": engine.chunk_count,
                "source_count": engine.source_count,
            }
        )

    return app


app = create_app()
