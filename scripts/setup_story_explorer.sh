#!/usr/bin/env bash
# Add the source-only React Flow explorer to the existing premium RiskLens app.
# Does NOT retrain models, access the holdout evaluator, call Groq, or push Git.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo 'ERROR: Node.js and npm are required. Install Node.js 24 LTS and rerun.' >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo 'ERROR: Activate your existing RiskLens Python 3.12 virtual environment first.' >&2
  exit 1
fi

echo '[1/4] Install frontend dependencies from npm (network needed for first build)...'
cd src/risklens_story/frontend
npm install --no-audit --no-fund

echo '[2/4] Typecheck and compile real React Flow component...'
npm run build

echo '[3/4] Register frontend assets with the existing RiskLens Python installation...'
cd "$ROOT"
python -m pip install -e '.[ui,dev]'

echo '[4/4] Read-only source check and focused tests...'
python scripts/verify_story_explorer.py
PYTHONPATH=.:src python -m pytest -q tests/test_story_explorer.py tests/test_phase8_*.py

echo 'All local setup checks passed. Launch using: python -m streamlit run app/Home.py'
