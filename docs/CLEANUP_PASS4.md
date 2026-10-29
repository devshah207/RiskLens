# Cleanup Pass 4 — Unified CLI

This is a **non-destructive migration** from many module-specific commands to one maintained `risklens` CLI. Older entry points are preserved for reproducing historical experiments and Citi demo scripts. No model, threshold, feature, or evaluation logic is changed.

After `python -m pip install -e ".[dev,deep,tuning]"`:

```bash
risklens --help
risklens compare data/processed/multiyear/nifty50_ohlc.csv data/processed/external/nifty50_external_development.csv --output-dir data/processed/multifactor_comparison
risklens demo --overwrite
risklens tune data/processed/multiyear/nifty50_ohlc.csv data/processed/external/nifty50_external_development.csv --output-dir data/processed/optuna_search --xgb-trials 0 --lstm-trials 200 --device mps
```

`demo` requires the existing local data. It generates an XGBoost reference if missing, then 95%/99% VaR analysis and the LSTM dashboard. It intentionally does not rerun earlier historical experiments if the reference already exists.

The `tune` command calls the existing Optuna optimizer. It uses **the same SQLite directory** to resume prior trials; never replace or delete that directory. This command will run the requested additional trials.

## Before deleting legacy CLIs

1. Run the entire pytest suite locally.
2. Run `bash scripts/run_citi_demo.sh` and `risklens demo --overwrite`, compare outputs and numeric results.
3. Inspect all `tests/`, `scripts/`, and documentation references to module CLIs.
4. Remove wrappers only after confirming nothing needs them; preserve older commands until then.
5. Keep 2025+ holdout outcomes sealed.

This pass does not delete any file. It establishes the replacement interface required to safely delete redundant wrappers in the next pass.
