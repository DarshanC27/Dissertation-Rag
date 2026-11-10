#!/usr/bin/env bash
# Idempotent dependency refresh for the Dissertation-Rag environment.
# Safe to run repeatedly: it reuses the venv, reinstalls pinned deps, and
# rebuilds the sample index from data/documents.
set -euo pipefail

# Resolve the repository root (parent of the .cursor directory).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Ensure the venv module can bootstrap pip. The default base image ships
# python3.12 but not the python3.12-venv package (ensurepip). Install it once
# if needed; this is idempotent and a no-op on images that already have it.
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  echo "Installing python3-venv (ensurepip missing)..."
  sudo apt-get update -qq
  sudo apt-get install -y -qq "python3.12-venv" || sudo apt-get install -y -qq python3-venv
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

# Rebuild the starter index from the bundled sample corpus so the server has
# something to answer immediately. Regenerated each run, so it stays in sync.
if [ -d data/documents ]; then
  python scripts/ingest.py data/documents --index data/index.pkl --reset
fi

echo "Dissertation-Rag environment ready."
