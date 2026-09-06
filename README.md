# Dissertation-Rag

A document-scanning **Retrieval-Augmented-Generation (RAG)** engine with a web
UI and JSON API. Point it at your notes, papers, and PDFs; it scans and chunks
them, indexes the chunks with TF-IDF, and answers questions grounded in the
retrieved passages.

The default retrieval + extractive-answer pipeline runs **fully offline with no
API keys**. If `OPENAI_API_KEY` is set, answers are additionally synthesized by
an LLM, but this is optional.

## Features

- **Scanner** — ingests `.txt`, `.md`, and `.pdf` files, normalizes whitespace,
  and splits them into overlapping chunks.
- **Retriever** — TF-IDF (unigram + bigram) vectors with cosine similarity.
- **Answerer** — extractive by default; optional OpenAI synthesis.
- **Web UI + API** — FastAPI app to ask questions and upload new documents.
- **CLI** — `scripts/ingest.py` to scan documents or query the index.

## Quick start

```bash
# 1. Install dependencies (creates .venv and builds the sample index)
bash .cursor/install.sh
source .venv/bin/activate

# 2. Run the server
uvicorn ragscanner.app:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

### Command line

```bash
# Ingest a file or directory
python scripts/ingest.py data/documents --index data/index.pkl

# Query the index
python scripts/ingest.py --query "What retrieval method does the scanner use?"
```

### API

| Method | Path      | Description                              |
| ------ | --------- | ---------------------------------------- |
| GET    | `/`       | Web UI                                   |
| GET    | `/health` | Liveness probe                           |
| GET    | `/stats`  | Index statistics                         |
| POST   | `/ask`    | `{"question": "..."}` → answer + sources |
| POST   | `/ingest` | Upload a document into the index         |

```bash
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "How does chunking work?"}'
```

## Development

```bash
source .venv/bin/activate
pytest            # run the test suite
```

## Project layout

```
ragscanner/         # package: scanner, engine, FastAPI app, templates
scripts/ingest.py   # CLI to scan/query the index
data/documents/     # sample dissertation corpus
tests/              # pytest suite (scanner, engine, API)
.cursor/            # Cloud Agent environment (install script + environment.json)
```

## Cloud Agent environment

`.cursor/environment.json` provisions the environment: `bash .cursor/install.sh`
creates the virtualenv, installs pinned dependencies, and builds the starter
index; the `rag-server` terminal then serves the app on port 8000.
