# Phase 6 — Explainability, Monitoring and Model-Risk Reporting

**Original RiskLens 10-phase roadmap — implementation-first, no further Optuna experiments.**

## Two maintained commands

The existing 2025–2026 holdout has already been evaluated once on the user's machine. The commands below **never rerun that evaluation, unlock data, or select models**. Both use the locked fixed XGBoost + actual Optuna LSTM trial #86 decision.

```bash
git switch main
git pull --ff-only origin main
python -m pip install -e ".[dev,deep,tuning]"
python -m pytest -q
ruff check src tests scripts --select F,E4,E7,E9

risklens monitor-models
risklens explain-xgb
risklens monitor-features \
  --vix data/processed/holdout_sources/india_vix_2024_2026.csv \
  --sp500 data/raw/sp500/sp500_daily.csv
```

### Monitoring: `risklens monitor-models`

Inputs:
- `data/processed/phase4_validation/frozen_model_predictions.csv`
- `data/processed/phase5_final_holdout/final_holdout_predictions.csv`

Requires exact frozen-prediction columns, strict ISO dates, complete development 2020–2024 and holdout 2025–2026, same model identities and valid binary outcomes.

Writes under `data/processed/phase6_monitoring/`:
- `monitoring_summary.json`: source SHA-256, annual and pooled descriptive scores, risk register, predeclared research alert thresholds
- `monitoring_periods.csv`: annual Brier, log loss, ROC-AUC, AP, event frequency, mean risk and calibration gap
- `monitoring_score_bins.csv`: fixed-bin PSI components comparing **model prediction-score distributions**, not feature drift
- `monitoring_report.md`: short monitoring results
- `model_risk_register.md`: MR-01 to MR-05, controls and limitations

Fixed bins `[0, .05, .10, .15, .20, .30, .50, 1]`, half-count smoothing. PSI 0.20, absolute mean-probability event gap 0.05, and Brier deterioration 0.02 are **research heuristics** that trigger investigation only. They are not bank-approved thresholds and do not cause auto retraining.

### Explainability: `risklens explain-xgb`

Retrains the **frozen fixed 11-feature XGBoost** once using eligible pre-2025 history and labels, reconstructed with pre-2025 causal features and a boundary-purged last label. It never uses holdout outcomes, changes parameters, or performs feature selection.

Uses XGBoost's **native TreeSHAP `pred_contribs=True`**, validated against raw logistic margins and predicted probabilities. Explanation sample is the last 500 eligible *pre-2025 training rows*, chosen by date independent of outcomes.

Writes under `data/processed/phase6_explainability/`:
- `xgboost_shap_summary.csv`: feature mean absolute/signed SHAP contributions in **log-odds units**, and normalized gain comparison
- `xgboost_shap_report.json`: source hashes, frozen params, sample choice, additivity audit
- `xgboost_model_card.md`: readable feature-attribution table and limitations

**Do not interpret SHAP as causality or evidence of future predictive success.** This is an in-sample descriptive explanation of a pre-2025 refit. It is not an exact archived model object from the user's one-time holdout run and cannot claim model-instance identity without a persisted model artifact.

## Remaining Phase 6 completion gates

1. Real-data runs on the user's Mac and verification of the generated reports.
2. Full **LSTM trial #86** sequence-level attribution/sensitivity audit, avoiding training on the opened holdout.
3. **Implemented:** `risklens monitor-features` reconstructs the same 11 causal inputs on pre-2025 history and exactly the already-scored 2025-2026 dates. It uses ONLY development-derived quantile bin edges and reports per-feature PSI, mean/std, out-of-development 1st/99th percentile share, provenance, and source-alignment audits. Real-data report remains to be verified.
4. Integrate model risk decisions, monitoring findings, annual holdout regime shift and VaR warning into a final governance memo.
5. Keep original 10 phases; after this, Phase 7 financial-report RAG and cited AI research assistant remains required.

No separate cleanup stage should be necessary; code lives in focused monitoring, feature-drift and explainability modules, the existing CLI and targeted tests.
