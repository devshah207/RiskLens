# Phase 4 — Frozen Model Selection: Optuna LSTM Trial #86 vs XGBoost

**Decision date:** 2026-10-29. **Status:** development-only selection, before sealed final holdout. **No new Optuna trials.**

## Corrected head-to-head comparison

The LSTM challenger is **Optuna trial #86**, **not** the original manually selected 30-session LSTM. The original LSTM benchmark must not be substituted as evidence for trial #86.

| Candidate | 2020–2022 Optuna tuning Brier ↓ | 2023–2024 post-selection diagnostic Brier ↓ |
|---|---:|---:|
| Fixed 11-feature XGBoost | ~0.12334 | **0.07176** |
| Optuna-selected XGBoost trial #53 | 0.122535 | 0.07186 |
| **Optuna-selected LSTM trial #86** | **0.120225** | 0.073162 |

Source: previously audited Optuna study summaries and previously recorded later diagnostic metrics; values have **not** been independently recomputed in this documentation update. Tuning scores and later diagnostic scores use different years, so **never compare scores across columns**. The 2023–2024 diagnostic years were already repeatedly inspected; they are **not independent test data**.

**Selection:** fixed 11-feature XGBoost as champion *candidate*. Trial #86 is retained as the designated LSTM challenger. Reason: trial #86 has lower tuning Brier, but fixed XGBoost has lower Brier on the later diagnostic years (difference 0.001402). This is a modest descriptive difference, **not** a statistically established superiority or proof of future generalization. If a stricter selection criterion is required, treat the decision as provisional pending a predeclared, one-time holdout evaluation.

## Frozen trial #86 LSTM hyperparameters

```json
{
  "lookback": 45,
  "hidden": 24,
  "layers": 2,
  "batch": 128,
  "epochs": 24,
  "dropout": 0.3413771488170494,
  "lr": 0.0027467789382135,
  "weight_decay": 0.0006684358163984
}
```

Optuna study trial number **86** is not a count of runs. Preserve the original local SQLite Optuna study database and trial exports. A 2020–2024 benchmark of an earlier manually configured LSTM (Brier 0.103196, ROC-AUC 0.583) is *not* trial #86 performance and should remain labeled **legacy benchmark only**.

## Freeze and next deliverables

1. **Champion:** fixed XGBoost, 11 predefined features, parameters from `src/risklens/models/xgboost_walk_forward.py`.
2. **Challenger:** selected Optuna LSTM #86, as above. Do not silently substitute `sequence_lstm.py`'s legacy default LSTM architecture.
3. **Training boundaries:** features must be point-in-time and strictly earlier for external sources; purge boundary-crossing target labels; select only on development years.
4. **Evaluation:** report aligned observations, per-year Brier/log loss/ROC-AUC/AP, calibration and uncertainty. Do not falsely treat repeated diagnostic inspection as an independent test.
5. **2025+ holdout:** remains sealed. Do not read it to tune, select, debug model performance, or change model choices.
6. **No more trials:** freeze current research evidence. Existing Optuna trials (historically audited as 330 completed LSTM and 137 completed XGBoost) are preserved, not rerun.
7. **Reproducibility caution:** this document records previous results; a fresh, pinned-environment run of the selected trial configurations, matched at the observation level, has not been executed here. Results may vary by device and seed (especially Apple MPS).

**Model risk conclusion:** modest advantage for fixed XGBoost on the already-reviewed later diagnostic period; neither model qualifies as independently validated, reliably profitable, or production-ready.
