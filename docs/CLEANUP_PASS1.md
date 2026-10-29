# Cleanup Pass 1 — Shared invariants and metrics

This is a **behavior-preserving refactor**, not model retraining or model selection. No experiment scripts were removed.

- `risklens.evaluation.metrics`: common classification and pooled probability scores.
- `risklens.evaluation.splits`: immutable 2020–2024 walk-forward years and 2025 holdout boundary.
- `risklens.evaluation.features`: external feature names and 11-feature grouping.
- Legacy modules re-export public/private helpers used by older tests for backwards compatibility.
- XGBoost and LSTM parameters, optimization search spaces, labels, and chronology were not changed.

## Check

```bash
python -m pytest -q
bash scripts/run_citi_demo.sh
```

Use the existing local datasets; preserve your SQLite Optuna studies separately. Do not merge the cleanup branch until these checks pass and the previously recorded benchmark metrics are reproduced.
