<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0a0e27,100:6e40c9&height=200&section=header&text=Dissertation-Rag&fontSize=56&fontColor=ffffff&fontAlignY=36&desc=Document-scanning%20RAG%20engine&descSize=17&descAlignY=58&animation=fadeIn" alt="Dissertation-Rag"/>

**Point it at notes, papers and PDFs: it scans, chunks and indexes them, then answers questions grounded in the retrieved passages.**<br/>Runs fully offline by default — no API keys required.

<img src="https://img.shields.io/badge/Python-3.12-3776ab?style=for-the-badge&labelColor=0a0e27&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/FastAPI-web%20UI%20%2B%20API-009688?style=for-the-badge&labelColor=0a0e27&logo=fastapi&logoColor=white" alt="FastAPI"/>
<img src="https://img.shields.io/badge/Retrieval-TF--IDF-6e40c9?style=for-the-badge&labelColor=0a0e27" alt="Retrieval"/>
<img src="https://img.shields.io/badge/Tests-pytest-0a9edc?style=for-the-badge&labelColor=0a0e27&logo=pytest&logoColor=white" alt="Tests"/>
<img src="https://img.shields.io/badge/Licence-MIT-16213e?style=for-the-badge&labelColor=0a0e27" alt="Licence"/>

[Quick start](#quick-start) · [Layout](#project-layout)

</div>

---

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
bash scripts/setup.sh
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
scripts/setup.sh    # one-shot environment setup (venv, pinned deps, sample index)
```

## Environment setup

`bash scripts/setup.sh` creates the virtualenv, installs pinned dependencies and builds
the starter index. It is idempotent, so run it again after pulling changes. Then start
the server with `uvicorn ragscanner.app:app --port 8000`.
