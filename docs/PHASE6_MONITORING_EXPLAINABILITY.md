# Phase 6 — Explainability, Monitoring and Model-Risk Reporting

> **Status update (30 October 2026):** All original real-data monitoring and explainability workflows have been run; three-feature investigation, development-only feature variants, seven-seed stability and saved-prediction PSI audit have been reviewed. Freeze verifier passed 9/9 and local tests passed (303). The numerical LSTM monitoring PSI alert is verified but the development-refit baseline confounder is unresolved causally. Refer to `RiskLens_Phase6_Review_Draft.md` and `RiskLens_Phase6_Saved_Prediction_PSI_Audit_2026-10-30.md`. Formal governance approval and PR #6 merge remain pending; no new research reruns are needed for sign-off.

**Original RiskLens 10-phase roadmap — implementation-first, no further Optuna experiments.**

## Historical implemented commands (already executed; do not rerun for sign-off)

The existing 2025–2026 holdout has already been evaluated once on the user's machine. The commands below **never rerun that evaluation, unlock data, or select models**. Both use the locked fixed XGBoost + actual Optuna LSTM trial #86 decision.

The original implemented workflows were `risklens monitor-models`, `risklens explain-xgb`, `risklens explain-lstm --device mps` and `risklens monitor-features`. They were already executed and inspected on the user's Mac. **Do not switch branches or rerun model fitting just to close Phase 6.** The read-only local sign-off checks are:

```bash
python scripts/verify_phase6_freeze.py
python -m pytest -q
ruff check src tests scripts --select F,E4,E7,E9
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

## Original implementation checklist (completed; governance approval pending)

1. **Completed:** Real-data runs on the user's Mac and verification of the generated reports.
2. **Implemented:** `risklens explain-lstm` trains one frozen Trial #86 sequence network with pre-2024 training data and performs three deterministic whole-sequence feature permutations on previously inspected 2024 development data. Report per-feature Brier change; this is noncausal sensitivity, not SHAP, and does not open 2025+ or run Optuna.
3. **Implemented:** `risklens monitor-features` reconstructs the same 11 causal inputs on pre-2025 history and exactly the already-scored 2025-2026 dates. It uses ONLY development-derived quantile bin edges and reports per-feature PSI, mean/std, out-of-development 1st/99th percentile share, provenance, and source-alignment audits. Real-data report subsequently verified; follow-up findings are in the scientific review.
4. **Completed for owner review:** Integrate model risk decisions, monitoring findings, annual holdout regime shift and VaR warnings into scientific governance memo and freeze record; merge still needs owner authorization.
5. Keep original 10 phases; after this, Phase 7 financial-report RAG and cited AI research assistant remains required.

No separate cleanup stage should be necessary; code lives in focused monitoring, feature-drift and explainability modules, the existing CLI and targeted tests.
