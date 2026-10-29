#!/usr/bin/env bash
# Run this from RiskLens repository root after: pip install -e '.[dev,deep]'
set -euo pipefail
NIFTY=data/processed/multiyear/nifty50_ohlc.csv
INTEGRATED=data/processed/external/nifty50_external_development.csv
REFERENCE=data/processed/multifactor_comparison/multifactor_comparison_predictions.csv
OUT=data/processed/citi_validation
if [[ ! -f "$NIFTY" || ! -f "$INTEGRATED" ]]; then
 echo 'Missing NIFTY or integrated inputs. Run Step 3.9 first.' >&2; exit 2
fi
if [[ ! -f "$REFERENCE" ]]; then
 echo 'Generating previous 9-vs-11 XGBoost reference on development years...'
 python -m risklens.models.multifactor_comparison_cli "$NIFTY" "$INTEGRATED" --output-dir data/processed/multifactor_comparison
fi
mkdir -p "$OUT"
python -m risklens.models.market_risk_monitor_cli "$NIFTY" "$REFERENCE" --output-dir "$OUT" --overwrite
python -m risklens.models.sequence_lstm_cli "$NIFTY" "$INTEGRATED" "$REFERENCE" --output-dir "$OUT" --var-report "$OUT/historical_var_report.json" --overwrite
printf '\nDONE. Open: %s\n' "$OUT/risk_model_validation_dashboard.html"
