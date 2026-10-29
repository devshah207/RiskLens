# RiskLens | Reproducible Optuna search

## Question
Can more systematic parameter search improve a probability forecast for NIFTY 50 next-session downside >=1%, without looking at the 2025+ final holdout?

### Time protocol
- **Fit years:** all earlier development history before each evaluated year, with the cross-year target label purged by the existing feature builder.
- **Optimization-only folds:** 2020, 2021, 2022. Optuna minimizes **pooled Brier score** across these years.
- **Post-selection diagnostic folds:** 2023 and 2024, not part of the Optuna objective. Important: these years were seen during *earlier* RiskLens experiments, so they are **not independent new validation**.
- **Final holdout:** 2025+, never constructed as model features or labels by this module.

### How to run/resume

```bash
python -m pip install -e '.[dev,tuning]'
python -m risklens.models.optuna_research \
  data/processed/multiyear/nifty50_ohlc.csv \
  data/processed/external/nifty50_external_development.csv \
  --output-dir data/processed/optuna_search --xgb-trials 200 --lstm-trials 60
```

Run the **same command again** to append additional trials. The persisted SQLite database is `optuna_studies.sqlite3` in the output folder. `--xgb-trials` and `--lstm-trials` are **new trials to attempt per invocation**, not lifetime caps. Increase them to any desired positive integer, subject to computer time/storage. Training is CPU-based and LSTM trials can be slow. Interrupted `RUNNING` trials can remain in SQLite and are excluded from completed trial counts.

### Model choices
- XGBoost: trees, max depth, learning rate, minimum child weight, gamma, row and column subsampling, L1 and L2 penalties. No validation set passed to XGBoost `.fit`, no tuning-year early stopping.
- LSTM: sequence length, hidden size, layers, dropout, learning rate, weight decay, batch size, epochs. StandardScaler is fitted on training windows only, per fold. The network is initialized deterministically per year/trial with fixed seeds. Class weighting stays off so probabilities retain an interpretable training prevalence.
- No training on evaluation years' outcomes. SMOTE is not used. Metrics are computed at original event prevalence.

### Important caveats
- Optimizing enough trials can eventually fit peculiarities of 2020–2022 rather than true predictive signal. A lower Optuna objective does **not** establish better future performance.
- Each run reports the 2023–2024 diagnostic result once per selected configuration. Do not pick a candidate by repeatedly optimizing against that diagnostic.
- Overlapping LSTM windows are dependent observations, and using the same underlying train windows for StandardScaler makes row-wise weights nonuniform. That is not future leakage, but affects scaling.
- Running this pipeline does not itself produce a calibrated operational model or a deployable live prediction API.

### Existing results
Previous fixed-parameter XGBoost-11 achieved pooled 2020–2022 Brier `0.12334460`, and 2023–2024 Brier `0.07176196` on the same validated dates. Both are **development-period** scores. Do not compare scores across different date sets without re-alignment.
