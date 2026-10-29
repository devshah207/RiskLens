# Repository maintenance

RiskLens preserves research evidence in Git history and local study databases, while keeping the active source tree focused.

## Maintained workflows

- `risklens compare` — time-aligned XGBoost baseline comparison
- `risklens demo --overwrite` — VaR, LSTM, and HTML dashboard
- `risklens tune ... --device mps` — resumable Optuna search on Apple Silicon
- `python -m pytest -q` — automated validation

## Data safety

`data/raw/` and `data/processed/` are ignored by Git. Do not remove source NIFTY, S&P 500, or India VIX CSVs; the integrated dataset; the multifactor comparison reference; or Optuna SQLite study/trial evidence. Generated outputs are local and can be regenerated, but check dependent scripts before archiving.

Prior intermediate cleanup instructions are preserved in Git history, not repeated in the active documentation.

## Verification

After any cleanup run `python -m pytest -q` and `risklens demo --overwrite`, and inspect the generated dashboard. Keep `main` untouched until local verification succeeds.
