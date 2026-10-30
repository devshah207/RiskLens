# Phase 4 — Frozen Champion/Challenger Evaluation

> **Dated historical-protocol notice (30 October 2026):** Statements below that the 2025+ holdout was *sealed* describe the state **at the time of Phase 4**. The predeclared one-time holdout was subsequently evaluated, audited and frozen; see `PHASE5_HOLDOUT_FINDINGS.md` and the Phase 6 governance memo. This older protocol is intentionally retained rather than rewritten as though it had always described later evidence. **Do not rerun the holdout or tune models.**

**Primary candidate:** fixed 11-feature XGBoost (original conservative parameters). **Challenger:** Optuna LSTM trial **#86**, using its *actual tuned architecture* and 45-session lookback.

This implementation runs **zero additional Optuna trials**. It checks your saved SQLite trial #86 parameters and objective value before fitting either model. The previous manually configured LSTM benchmark remains separate and must not be called trial #86.

## Run on Apple Silicon

From the repository root:

```bash
python -m pip install -e '.[dev,deep,tuning]'
python -m pytest -q
ruff check src tests scripts --select F,E4,E7,E9
ls -lh data/processed/optuna_search/optuna_studies.sqlite3
risklens evaluate-frozen --device mps
```

If the database has a different location, specify `--study-db /absolute/path/optuna_studies.sqlite3`. For a clean re-run, pass `--overwrite`. CPU is also supported with `--device cpu`.

The command compares both models on identical prediction dates and targets in the 2020–2024 development period. It reports **2020–2022 tuning years separately from the repeatedly inspected 2023–2024 diagnostic years**. Neither constitutes untouched independent validation.

## Outputs

`data/processed/phase4_validation/`:

- `frozen_model_predictions.csv` — aligned dates, actual labels, two probability columns
- `frozen_model_report.json` — per-year/period metrics, source hashes, exact parameters, study SHA256, dependency versions
- `frozen_model_report.md` — compact comparison table and limitations

The **2025+ final holdout remains sealed**, and no retraining should depend on those outcomes. Trial #86 uses the original Optuna training implementation, not the default parameters in `sequence_lstm.py`.

Run the command locally on the actual datasets to generate numerical evidence; this commit does **not** assert that the new evaluator has been run on your Mac or its final scores reproduced.
