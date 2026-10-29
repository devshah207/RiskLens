# Cleanup Pass 3 — Decouple integrated-data validation from model comparison

This is a **behavior-preserving** change. The pre-2025 OHLC/source-date verification function is now in `risklens.evaluation.integrated_validation`. Both XGBoost comparison and the LSTM sequence pipeline use this shared implementation. The prior symbols remain re-exported through `models.multifactor_comparison` so historical code and tests keep working.

No training hyperparameters, feature calculations, chronology, labels, Optuna study paths, or data outputs were changed. No files were deleted in this pass because additional module consolidation requires more dependency work.

On branch `refactor/codebase-cleanup`, run:

```bash
python -m pytest -q
bash scripts/run_citi_demo.sh
```

Compare the demo scores with the pre-refactor evidence before committing. Keep all raw files and Optuna studies locally.
