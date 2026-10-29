# Phase 5 — Independent Risk-Model Validation: locked protocol and development diagnostics

**Original ten-phase RiskLens roadmap — Phase 5.**

This phase includes: discrimination, calibration, out-of-time performance, historical VaR backtesting, exception independence, failure patterns and limitations. **The implemented command below is the development-data validation component, not the one-time independent 2025+ holdout.**

## Run the development-only validation

After Phase 4 has generated the **actual frozen Optuna LSTM #86** predictions, and the existing Citi demo has generated historical VaR:

```bash
risklens evaluate-frozen --device mps --output-dir data/processed/phase4_validation
risklens demo --overwrite
risklens validate-risk
```

Inputs are the exact aligned development outputs:
- `data/processed/phase4_validation/frozen_model_predictions.csv`
- `data/processed/citi_validation/historical_var_backtest.csv`

The validation command does **not** fit or tune a model or generate new risk forecasts. It requires dates to match exactly; it cross-checks NIFTY returns against downside-event labels and recomputes historical VaR exception flags rather than trusting a CSV flag blindly. It refuses dates in or after 2025.

Outputs in `data/processed/phase5_validation/`:
- `risk_validation_summary.json`: reproducible source SHA-256s, model Brier/log loss/ROC-AUC/AP, fixed-bin calibration, per-level VaR exception counts and coverage tests, model difference diagnostics
- `risk_validation_yearly.csv`: year-specific discrimination, calibration-sensitive scores and event prevalence
- `risk_validation_reliability.csv`: predeclared probability bins, observed event frequencies, sample-size warnings
- `risk_validation_cases.csv`: descriptive highest-surprise actual events/non-events, by candidate
- `risk_validation_report.md`: compact human-readable validation memo

Use `--overwrite` explicitly to replace generated reports. The block bootstrap defaults to 1,000 samples, 10-session circular blocks, seed 2026, sampled **within** each calendar year; it describes paired Brier uncertainty, not a universally valid confidence claim. Changing bootstrap settings for sensitivity investigation must not be used to select the champion.

## Independent 2025+ evaluation protocol — freeze before unlocking

1. **Target:** `1[close_(t+1)/close_t - 1 <= -0.01]` from the NIFTY 50 *price index*.
2. **Information time:** after Indian market close t; NIFTY indicators may include session t, while India VIX and S&P 500 must originate from strictly earlier dated observations (conservative existing join rule, maximum seven calendar days).
3. **Train/refit:** train the frozen fixed 11-feature XGBoost model on all eligible, boundary-purged data through 2024-12-31; use *no* 2025+ target outcomes for training, calibration, thresholds, selection or early stopping. Freeze exact preprocessing and package versions. The locked Optuna LSTM #86 remains challenger; refit its 45-session network using the pre-2025 training data only and its original seed policy.
4. **Out-of-time validation:** one pass on eligible 2025-01-01 through 2026-09-30 NIFTY prediction dates, stopping at the last date with a **known next-session close**. The external feature CSV currently covers development dates only. Before evaluating, gather and independently validate the 2025–2026 India VIX and completed-prior-session S&P 500 series with identical point-in-time semantics; the existing development integrator intentionally excludes post-2024 observations. Do not impute external gaps or silently omit a candidate unevenly.
5. **Primary assessment:** paired Brier score, with log loss, annual/per-regime ROC-AUC, AP, calibration-in-the-large and fixed-bin reliability as secondary; compare only identical eligible dates and outcomes. Report evaluation sample size and event counts; no performance threshold/feature set changes after seeing holdout scores.
6. **VaR:** 250-session trailing historical returns inclusive of t, compared against next-session losses at 95% and 99%, with exact coverage and conditional exception independence tests. This is an *index* VaR monitor, separate from the downside probability classifier.
7. **Promotion decision:** retain the preselected fixed XGBoost unless a *predeclared* governance process approves a changed model **based on evidence outside the opened holdout**. A better score for LSTM on the one-time holdout is a finding, not permission to tune again on that holdout. Clearly report any deterioration and uncertainty.
8. **Audit trail:** retain source checksums, trial #86 SQLite checksum, frozen model parameters, dependency versions, training cutoff, prediction dates, all forecasts, validation scores, error samples and data exclusion reasons.
9. **Final verdict:** report successes and failures, limitations, bias from earlier inspected development years and absence of production deployment; do not equate empirical risk modeling with approved regulatory capital models.
10. **Holdout opening:** requires a separate, deliberate one-time run once the full external data inputs exist and the implementation of this locked protocol is independently checked. This development-only CLI **does not open the holdout**.

## Scientific limitations

- Tuning years 2020–2022 and repeatedly viewed diagnostics 2023–2024 are *not independent test sets*.
- Pooled ROC-AUC/AP across yearly refits should be interpreted cautiously due to changing probability scales and prevalence; yearly metrics are central.
- Fixed-bin expected calibration error (ECE) is sensitive to bin boundaries and sparse events; it is a diagnostic, never a learned recalibration.
- Christoffersen-style likelihood-ratio exception independence / conditional coverage tests rely on asymptotic approximations that can be poor in sparse 99% tail events; undefined transitions are explicitly reported as null.
- Bootstrap intervals use a fixed within-year circular block scheme; financial nonstationarity and overlapping windows limit inferential interpretation.
- No evidence here supports tradability, stable future predictive power, or regulator-approved market-risk methods.

**Status:** validation instrumentation implemented; external final-holdout data acquisition, one-time scoring, and independent verdict remain open Phase 5 deliverables. Do not declare Phase 5 complete until those are satisfied.
