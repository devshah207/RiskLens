# Citi RMS — RiskLens project evidence and interview defense

**Reconciled on 13 November 2026** against the archived project prediction files and the existing frozen validation memo. This is the *current interview guide*, not a replacement for historical experiment records. Never represent development statistics as the completed final holdout results.

## Project and deliverable

**RiskLens** is an implemented, local-first financial risk analytics and model-validation workbench. It estimates the probability that the NIFTY 50 price index will decline by at least **1% at the next trading-session close**, combines chronology-safe data engineering with XGBoost/LSTM model comparison, and presents separate historical Value-at-Risk (VaR), validation, calibration and monitoring evidence. A separate XBRL/PDF research assistant supports evidence-grounded financial filing questions; its documents are **not** model predictors.

## Resume-ready evidence (use for future submissions)

- Built a leakage-audited market-risk pipeline integrating NIFTY 50 OHLC, India VIX and S&P 500 data, with **11 predictors** and strictly earlier-dated external-market joins.
- Benchmarked XGBoost and PyTorch LSTM classifiers for ≥1% next-session downside classification. On **1,238 archived 2020–2024 development-period walk-forward predictions**, fixed XGBoost achieved **0.609487 ROC-AUC** and **0.102824 Brier score**.
- Completed a separate, one-time **433-session 2025–September 2026 out-of-time research evaluation**; implemented historical **95%/99% VaR** exception analysis, monitoring, explainability and a read-only Streamlit risk dashboard.

**Reconciliation of an older resume:** The older 0.612 development ROC-AUC is not reproduced by the two archived aligned prediction CSVs; the result calculated from them is **0.609487**. Do not retroactively claim that a previously submitted resume showed the corrected number. If asked, explain that a later evidence reconciliation identified a small reporting inconsistency and provide the archived metric. The Brier score of approximately **0.1028** is consistent.

**Sequence warning:** The legacy sequence benchmark uses **30 sessions**. The separate tuned Optuna **LSTM Trial #86** uses **45 sessions × 11 features**. Do not attribute the earlier model's architecture or metrics to Trial #86.

## Distinguish the experiments

| Evaluation | Fixed XGBoost | LSTM | How to describe it |
|---|---|---|---|
| Archived 2020–2024 development predictions, 1,238 dates | **AUC 0.609487; Brier 0.102824** | Legacy 30-session LSTM: AUC approximately **0.583** | Earlier benchmark; repeatedly examined development data, not an untouched holdout |
| Frozen development comparison | Fixed 11-feature research champion | Tuned 45-session Trial #86 retained as challenger | Selection was documented before final holdout |
| One-time 2025–Sep 2026 final research holdout, 433 dates, 42 downside events | **AUC 0.672756; Brier 0.084773** | **AUC 0.684874; Brier 0.084299** | Independent *out-of-time research evaluation* of the fixed candidates, not another round of model selection |

The LSTM slightly exceeded XGBoost on pooled final-holdout metrics, but the previously selected XGBoost remained the named champion to avoid using final-test results as model-selection feedback. This is **not** proof that XGBoost is inherently superior.

Both candidates had **0/42 true-positive alerts at the default probability cutoff of 0.50**. AUC measures score ranking, not operational detection at that cutoff. Do not describe either as a validated live alarm or trading system.

## Interview introduction (about 35–45 seconds)

> RiskLens is a financial risk analytics and model-validation platform I built to estimate the probability of significant next-session declines in the NIFTY 50. I developed a market-data pipeline combining NIFTY, India VIX and S&P 500 information, then compared XGBoost and LSTM approaches through time-ordered evaluation. The platform also includes a completed out-of-time model test, historical Value-at-Risk backtesting, monitoring and a dashboard for examining saved risk evidence. It is an end-to-end research implementation focused on auditable risk estimation, not just model training.

Give the opening answer, **stop**, and expand only when an interviewer asks.

## High-priority technical follow-ups

**Why define this problem?** It estimates a specific downside-event probability rather than claiming an exact market level, trading profit, or portfolio loss amount. The **1% threshold** is a research target definition, not an industry-standard optimum.

**Why use NIFTY, India VIX and S&P 500?** NIFTY-based features summarize market returns and volatility; earlier-dated India VIX supplies volatility context, and the latest previously completed US session supplies international market context. Strict source-date alignment reduces look-ahead.

**Why XGBoost and LSTM?** XGBoost handles nonlinear relationships in tabular risk features and is comparatively straightforward to validate. The LSTM processes feature sequences and can represent path dependence. Its additional complexity does not guarantee generalization.

**Why keep XGBoost champion when Trial #86 scored higher on final holdout?** The champion/challenger roles were predeclared based on development diagnostics. Switching after opening the final holdout would contaminate the independence of that evaluation.

**How did you reduce leakage?** Earlier-only external market joins, purging labels at chronological boundaries and training-only scaling in the LSTM. These controls do not certify the vendors' original data publication timestamps.

**What does 0.609487 AUC mean?** It measures the ability to rank positive versus negative cases across the archived development predictions, not 60.95% classification accuracy or investment success. Brier is a probability-error score; lower values are better.

**What is VaR here?** A **separate historical-quantile risk calculation** using past returns, tested through 95%/99% exceptions. It is not generated by the XGBoost/LSTM downside-probability outputs.

**What is distinctive?** The contribution is an integrated, reproducible *workflow*: point-in-time data alignment, fixed champion/challenger comparison, untouched out-of-time evaluation, distinct VaR and monitoring evidence, and a local analysis interface. Do not claim invention of XGBoost, LSTM or VaR.

**What are the limitations?** The 0.50 cutoff produced no positive alerts on the final holdout; the independent period is finite; original final trained weights were not archived; financial-report answers are not bank- or human-certified. The document research path has recorded source-reverification issues in the inspected snapshot. These are boundaries of a completed research implementation, not evidence of bank production approval.

## Source evidence and safe checks

- [Frozen model validation memo](CITI_MODEL_VALIDATION_MEMO.md)
- [Completed holdout findings](PHASE5_HOLDOUT_FINDINGS.md)
- [Historical pre-opening protocol](PHASE5_FINAL_HOLDOUT.md)
- Saved prediction artifacts: `data/processed/citi_validation/`, `data/processed/phase4_validation/`, `data/processed/phase5_final_holdout/`

Safe verification: `python scripts/verify_phase6_freeze.py` and the existing project tests. Do **not** rerun the already opened final holdout, retune against its outcomes, or claim every research test necessarily passes.
