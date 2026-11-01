#!/usr/bin/env bash
# Read-only evaluation. Run from the RiskLens project root.
set -euo pipefail
ARCHIVE="${1:-$HOME/Downloads/Archive(1).zip}"
QUESTIONS='data/processed/research/real_acceptance/phase7_completion/v71_unseen_34/live_human_audit.csv'
DB='data/processed/research/real_acceptance/real_index.sqlite3'
OUT='data/processed/research/xbrl_34_evaluation_offline'
if [[ ! -f "$ARCHIVE" ]]; then
  echo "ERROR: original eight-file NSE XML archive missing: $ARCHIVE" >&2
  exit 1
fi
if [[ ! -f "$QUESTIONS" ]]; then
  echo "ERROR: original 34-question audit file missing: $QUESTIONS" >&2
  exit 1
fi
export RISKLENS_XBRL_TEST_ARCHIVE="$ARCHIVE"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"
echo '[1/2] Checking XBRL integration against the eight original XML files'
python -m pytest -q tests/test_phase7_xbrl_rag_v1.py
echo '[2/2] Evaluating existing 34 questions, offline, without Groq'
if [[ -f "$DB" ]]; then
  python scripts/phase7_xbrl_offline_evaluate.py \
    --questions "$QUESTIONS" --xbrl-archive "$ARCHIVE" --db "$DB" --out "$OUT"
else
  echo 'Original PDF RAG SQLite index not found; evaluating XBRL route only.'
  python scripts/phase7_xbrl_offline_evaluate.py \
    --questions "$QUESTIONS" --xbrl-archive "$ARCHIVE" --out "$OUT"
fi
echo "Results saved in $OUT (results.csv and summary.json)."
