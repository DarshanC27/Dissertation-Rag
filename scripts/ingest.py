#!/usr/bin/env python3
"""Command-line document scanner/ingester.

Usage:
    python scripts/ingest.py <path> [--index data/index.pkl] [--reset]
    python scripts/ingest.py --query "your question" [--index data/index.pkl]

Scans a file or directory into the TF-IDF index, or queries an existing index.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragscanner.engine import RagEngine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan documents into a RAG index.")
    parser.add_argument("path", nargs="?", help="File or directory to scan.")
    parser.add_argument("--index", default="data/index.pkl", help="Index file path.")
    parser.add_argument("--query", help="Query the existing index instead of ingesting.")
    parser.add_argument("--k", type=int, default=4, help="Number of passages to retrieve.")
    parser.add_argument("--reset", action="store_true", help="Start a fresh index.")
    args = parser.parse_args()

    index_path = Path(args.index)

    if args.query:
        engine = RagEngine.load(index_path)
        if engine.chunk_count == 0:
            print(f"Index at {index_path} is empty. Ingest documents first.", file=sys.stderr)
            return 1
        result = engine.answer(args.query, k=args.k)
        print(f"Q: {result.question}\n")
        print(f"A ({result.generator}):\n{result.answer}\n")
        print("Sources:")
        for src in result.sources:
            print(f"  - {Path(src.source).name} (score {src.score:.3f})")
        return 0

    if not args.path:
        parser.error("provide a path to ingest, or --query to search")

    engine = RagEngine() if args.reset else RagEngine.load(index_path)
    before = engine.chunk_count
    added = engine.ingest_path(args.path)
    engine.save(index_path)
    print(
        f"Ingested {added} chunks from {args.path} "
        f"(index now has {engine.chunk_count} chunks from {engine.source_count} sources; "
        f"was {before})."
    )
    print(f"Saved index to {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
