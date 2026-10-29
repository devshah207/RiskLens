# Cleanup Pass 5 — Retire old comparison CLIs

Deleted five experiment-only command wrappers: calibration, diagnostics, OHLC comparison, walk-forward, and XGBoost walk-forward CLIs. Their computation modules remain intact; the unified `risklens` CLI and Citi demo remain. These deletions do not modify model weights, data, feature engineering, or evaluation rules.

Safety gate: run `python -m pytest -q`, `risklens --help`, and `bash scripts/run_citi_demo.sh` on `refactor/codebase-cleanup` before merging. The baseline comparison and LSTM / VaR entry points remain available. Do not commit raw data or Optuna SQLite studies.
