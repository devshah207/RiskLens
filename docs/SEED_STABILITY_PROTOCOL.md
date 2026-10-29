# RiskLens — predeclared seed-stability robustness experiment

**Status:** Proposal for user review. No GitHub push or merge, no real-data result in this package.

## Decision this experiment addresses

On already-studied 2020–2024 development folds, does adding four causal regime-context variables to **fixed Optuna LSTM Trial #86** improve ranking *consistently across different neural-network training seeds*, rather than from one lucky training initialization?

**Not answered:** whether any variation improves genuine future 2027+ out-of-sample performance. The original 2025–2026 holdout is already open and MUST NOT be rescored or used for selection. This is hypothesis screening, not independent verification.

## Locked specifications

- Exactly **two** variants: `full_11` and `full_plus_regime_context` from the already-approved drift-hypothesis experiment. **No new candidate variations.**
- Exactly **seven** random-seed offsets: `[0, 17, 43, 79, 131, 191, 263]`. Per fold, `torch_seed = 2026 + validation_year + offset`; seed offset zero matches the former default.
- Exactly **five purged expanding folds** with validation years 2020–2024. Report previously tuned years 2020–2022 separately from already-revisited diagnostics 2023–2024, plus each year and full pooled development.
- The same labeled train/validation rows and model seeds for both variants in each fold. Check sequence-dates and outcomes for both.
- **No Optuna runs, no retuning:** same Trial #86 lookback 45, 24 epochs, layers 2, hidden 24, dropout, weight decay, learning rate and batch size. Per-fold scalers use prior training history only. Training runs: **7 seeds × 5 years × 2 variants = 70**.
- Comparisons: paired ΔROC-AUC, Δaverage precision, and Brier/log-loss gains, grouped per seed+year and per seed+development period. Every seed is reported; never pick the best seed. 
- Report median, mean, minimum, maximum and fraction of seeds favoring regime context; **these are not confidence intervals**, because every run shares the same development data. Don't confuse training instability with generalization uncertainty.
- Source SHA-256 and saved Optuna trial #86 provenance recorded; no changes to original model or frozen 2025–2026 holdout.

## How to review the result

Strong exploratory signs would include *positive median ΔAUC across seeds* in **both** development periods, at least 5/7 positive seeds in both periods, reasonable Brier/AP behavior, and no dominant annual collapse. The 2024 performance issue must be surfaced even if pooled metrics improve. These are *screening criteria*, not a model-promotion rule or a statistical significance test.

A mixed or negative result means regime-context benefit is not seed-stable; don't manufacture a positive conclusion by searching extra seeds or new variants. Even a strong positive result cannot alter `main`'s champion or the sealed 2025–2026 claim without legitimately new future evaluation data.

## Installation / usage after local patch is approved

From the existing experimental branch of the RiskLens repo, copy these **additive** files into the matching paths. Do not overwrite prior experiment files. Then:

```bash
python -m pytest -q tests/test_seed_stability.py
python -m risklens.experiments.seed_stability --device mps
```

Input defaults (local Mac, same original development datasets): `data/processed/multiyear/nifty50_ohlc.csv`, `data/processed/external/nifty50_external_development.csv`, `data/processed/optuna_search/optuna_studies.sqlite3`.

Outputs saved under `data/processed/lstm_seed_stability_development/`:

1. `seed_stability_manifest.json`: inputs, hashes, seeds, version, lock and warnings.
2. `seed_stability_predictions.csv`: one row per seed × variant × historical validation date.
3. `seed_stability_yearly.csv`: paired year-specific performance and differences.
4. `seed_stability_periods.csv`: paired period-specific performance and differences.
5. `seed_stability_summary.csv`: across-seed descriptive stability statistics.
6. `seed_stability_report.md`: readable result tables and interpretation limits.

No default overwrite is allowed; `--overwrite` must be explicit. No GitHub push, PR, or merge is performed by the Python workflow.
