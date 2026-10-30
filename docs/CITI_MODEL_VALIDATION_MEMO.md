# RiskLens — Market-Risk Model Development, Independent Research Validation & Monitoring Memo

**Audience:** Citi Risk Model Development / Validation / Monitoring (academic internship interview)
**Updated:** 30 October 2026
**Status:** Retrospectively audited **research prototype**; Phase 6 owner governance sign-off pending. This is **not** a Citi-approved model or an independent bank validation opinion.

## Executive assessment

RiskLens estimates the probability that the **NIFTY 50 price index** declines by **at least 1% from close `t` to next observed close `t+1`**, using information available at the close of session `t`. It is **market-index downside event classification**, not loan default prediction, a credit model, portfolio risk-capital VaR or a trading strategy.

The preselected **fixed 11-feature XGBoost** remains the research champion. The actual **Optuna LSTM trial #86**, using a 45-session lookback, remains the challenger. Original 2025–September 2026 one-time holdout of **433 aligned dates, 42 realized events** has been opened once and is frozen. LSTM's numerical advantage does not override pre-holdout selection or prove statistical superiority.

| Frozen candidate | Holdout ROC-AUC ↑ | Average precision ↑ | Brier ↓ | Log loss ↓ |
|---|---:|---:|---:|---:|
| XGBoost — preselected champion | 0.672756 | 0.186103 | 0.084773 | 0.303295 |
| Optuna trial #86 LSTM — challenger | 0.684874 | 0.218973 | 0.084299 | 0.300362 |

**Priority model-risk findings:** both models generate **0/42 true-positive alerts at a 0.5 cutoff**; actual downside prevalence changes from **6.02% in 2025 to 14.67% in Jan–Sep 2026**, reversing the direction of annual calibration-in-the-large bias. Historical 95% VaR exception clustering fails approximate exception-independence and conditional-coverage research tests. Monitoring PSI has non-identical annual model-instance baseline limitations. Data-source publication timestamps, usage rights and exact original trained model weights have not been independently authenticated/archived. These limitations must accompany any portfolio or interview claims.

## 1. Point-in-time research contract and data integrity

- **Timing and label:** `y_t = 1[(close[t+1] / close[t] - 1) <= -0.01]`. The `t+1` close is an *outcome only*, not a predictor. Training purges observations whose label would require a validation/holdout-session close.
- **11 frozen predictors:** NIFTY 1-/5-day returns, 20-/10-day annualized realized volatility, intraday range, intraday return, opening gap, RSI14, EMA50 distance; prior-dated India VIX; and previous completed US S&P 500 return. Preserve canonical feature order in `evaluation/features.py`.
- **Data:** NIFTY OHLC 2016–30 Sep 2026 and supporting VIX/S&P observations. Earlier external sources must be strictly *prior calendar dates* within seven days; no same-day US close, future imputation or random train/test split.
- **QA:** Source provenance hashes, duplicate/invalid OHLC checks, strict temporal joins, feature warmups and missing windows. Independent RSI/EMA reimplementations and 2024 VIX overlap inspection found no demonstrated formula/source mismatch, but exchange dissemination timestamps and licensing are not fully independently verified.
- **Chronology:** Expanding yearly 2020–2024 development validation, with 2020–2022 used in Optuna selection and 2023–2024 inspected repeatedly. The **2025–Sep 2026 holdout** was the original one-time independent *research* test. No 2025+ outcomes informed frozen hyperparameter choices.

## 2. Frozen champion and challenger

### XGBoost: interpretable conservative benchmark

Fixed 11-feature XGBoost, with 80 estimators, max depth 2, learning rate 0.04, gamma 1, min child weight 10, subsample 0.8, column sampling 0.9, `reg_lambda=10`, `reg_alpha=0`, binary-logistic objective, `hist` trees and seed 42. It was **preselected** as research champion on governance/generalization grounds; there is no post-hoc promotion after the opened holdout.

### LSTM: verified genuine trial #86, not the legacy model

The genuine completed `risklens_lstm` Optuna **trial 86** uses 45 market sessions × 11 features; two-layer LSTM, hidden 24, dropout `0.3413771488170494`, 24 fixed epochs, batch 128, AdamW learning rate `0.0027467789382135`, weight decay `0.0006684358163984`. Uses train-only `StandardScaler`, bounded standardized inputs, gradient clipping and seed-specific chronological minibatches. The original final fit seed was 4051 on Apple MPS. The saved SQLite Optuna fingerprint and the trial's eight hyperparameters were audited. An older separate LSTM using 30 sessions, dropout 0.15, LayerNorm/MLP and internal early stopping exists and **must not** be described as trial #86.

**No serialized fitted weights:** Original locked predictions, metrics, parameters and evidence fingerprints were stored, not the exact original fitted XGBoost object or LSTM PyTorch state dictionary. Re-fitting does not recover an identical model instance; MPS numerics may vary.

## 3. Development-only context, not independent performance

The original **frozen** 11-feature aligned development comparison on 1,238 scored dates (2020–2024; 145 downside events) recorded:

| Candidate | Development ROC-AUC ↑ | Average precision ↑ | Brier ↓ | Log loss ↓ |
|---|---:|---:|---:|---:|
| Fixed XGBoost | 0.609487 | 0.169863 | 0.102824 | 0.358132 |
| Actual trial #86 LSTM | 0.619365 | 0.167665 | 0.102600 | 0.360885 |

Optuna tuned on **2020–2022**, and **2023–2024 was repeatedly reviewed**: neither part represents a fresh untouched validation sample. The distinct *legacy* 30-session LSTM had approximately Brier **0.10320**, AUC **0.583** versus XGBoost ~Brier **0.10280**, AUC **0.612** in its earlier benchmark. The archived historical code and results are retained for methodological transparency, **not** combined as though they were trial #86.

## 4. Original independent holdout and probability limitations

The **433 forecasts** share dates/labels for both models: 249 forecasts and 15 events in 2025; 184 forecasts and 27 events through Sep 2026. No new model tuning, re-thresholding or reselection is permitted after viewing these scores.

| Holdout year | XGBoost AUC | LSTM AUC | XGBoost Brier | LSTM Brier | Event rate |
|---|---:|---:|---:|---:|---:|
| 2025 | 0.693447 | 0.715385 | 0.055743 | 0.057976 | 6.02% |
| Jan–Sep 2026 | 0.619486 | 0.636943 | 0.124057 | 0.119922 | 14.67% |

In **2025**, each model **overestimated** mean event risk; in **2026**, each **underestimated** it. Their pooled calibration statistics partially mask these opposite annual errors. At fixed threshold 0.50, each model predicted **391 true negatives, 42 false negatives, zero true positives and zero false positives** on the pooled holdout. **AUC/AP are ranking measures, not evidence of a viable operational alarm cutoff.** Choosing a favorable cutoff on this already-opened holdout would be prohibited post-hoc optimization.

The LSTM's slightly better pooled ranking and Brier do not justify a champion switch or formal confidence interval claims about superiority. The index series, nonstationary prevalence and relatively small event counts limit statistical inference.

## 5. VaR — separate historical 250-return risk model

Historical-simulation VaR quantiles use the latest **250 daily NIFTY simple returns known at `t`**, taking the empirical 5th and 1st percentiles for 95% and 99% one-session downside VaR. Next-session realized returns are compared with those thresholds. VaR is **not** calculated from the XGBoost/LSTM probability outputs.

| Holdout diagnostic | 95% historical VaR | 99% historical VaR |
|---|---:|---:|
| Exceptions / 433 | 30 (6.93%) | 7 (1.62%) |
| Target exception rate | 5% | 1% |
| Exact unconditional binomial p | 0.07631 | 0.21602 |
| Approx. Markov independence p | 0.01403 | 0.63068 |
| Approx. conditional-coverage p | 0.01069 | 0.44206 |

The **95% unconditional** test does **not** reject at 5%, but independence and joint conditional-coverage tests flag clustering under the stated finite-sample/asymptotic assumptions. 2025 had eight 95% breaches, 2026 had 22. **March 2026 alone had 9 of 19 evaluated dates breaching 95% VaR and 4 breaching 99% VaR**. Independent NIFTY return/quantile reconstruction recovered all 30/7 exceptions and the 95% within-year transition counts. These findings identify quantile lag/exception clustering, not an authenticated macroeconomic cause; 99% inference remains weak with seven events.

**Earlier development VaR** was a different interval (2020–2024), with 67/1,238 95% and 22/1,238 99% exceptions. Do not present those historical study counts as final-holdout results.

## 6. Explainability, drift, negative experiments and model-risk management

- **XGBoost SHAP:** fixed 11-feature model fitted on eligible pre-2025 data, native TreeSHAP on 500 recent *training* observations; margin/log-odds units, max additivity difference about `1.65e-6`. Strongest mean absolute associations included India VIX (0.2451), RSI14 (0.1928) and opening gap (0.0932). Not causality, and not an exact archived final-fitted instance.
- **LSTM permutation:** trial #86 network evaluated on 248 already-inspected 2024 development sequences, three temporal-feature trajectory shuffles; small Brier perturbation effects and correlation-breaking limitations. Not SHAP, causal sensitivity or future holdout.
- **Feature PSI:** EMA50 distance ~0.479, India VIX ~0.426 and RSI14 ~0.393 triggered *research* investigation. Independent EMA/RSI math matched; VIX overlap matched on 248 dates. Comparison of VIX against 2023–2024 rather than 2020–2024 sharply reduced its PSI, highlighting baseline sensitivity. Drift is not synonymous with a code bug.
- **Ablations and robustness:** eight predeclared development-only feature variants and seven fixed-seed repeated LSTM comparisons showed **no robust improvement** from dropping VIX/EMA/RSI or adding regime-context fields. Keep original frozen features; no best-seed cherry-picking.
- **Score PSI:** saved predictions reproduced original pooled PSI: **XGBoost 0.06829280**, **LSTM 0.47019283**. Within each single unchanged final fitted model, comparing 2025 versus 2026 produces PSI about **0.42518 for XGBoost** and **0.42334 for LSTM**. Original 2020–2024 development scores, however, pool five annually refitted model instances, whereas 2025–2026 uses one final fitted instance, so pooled PSI differences cannot be uniquely ascribed to market shifts. Preserve the monitoring alert and root-cause limitation; do not automatically retrain.
- **Data/model governance:** No recovered original fitted-weight archive; vendor publication and licensing not independently certified; no calibrated action policy. Retrospective research and risk documentation are complete but do **not** establish operational/regulatory sign-off.

The source model-risk register, exact PSI bins and period counts, model specifications and holdout constraints are available at [`RiskLens_Phase6_Review_Draft.md`](RiskLens_Phase6_Review_Draft.md), [`RiskLens_Phase6_Saved_Prediction_PSI_Audit_2026-10-30.md`](RiskLens_Phase6_Saved_Prediction_PSI_Audit_2026-10-30.md), [`PHASE6_MODEL_FREEZE_RECORD.md`](PHASE6_MODEL_FREEZE_RECORD.md) and [`PHASE5_HOLDOUT_FINDINGS.md`](PHASE5_HOLDOUT_FINDINGS.md).

## 7. Research decision, next work and interview integrity

**Research decision:** Retain the preselected frozen XGBoost champion and trial #86 LSTM challenger, without any model/feature/threshold change. Original holdout is a **one-time evaluation**, not another tuning set. Retain scientific failure cases, competing explanations and provenance limitations in reports. Model risk items are documented for *prospective monitoring*, not claimed resolved causally.

**Publication status:** The Phase 6 scientific review and evidence freeze are on [draft PR #6](https://github.com/devshah207/RiskLens/pull/6). Phase 6 formal repository/governance closure requires owner approval and **separately authorized merge**. After merge, proceed with Phase 7 cited financial-report RAG, followed by Phase 8 Streamlit integration, Phase 9 portfolio evidence and Phase 10 actual code-based Citi/StockGro interview defense.

**Interview claims must be accurate:** RiskLens demonstrates a financial risk-data pipeline, temporal modeling, validation, interpretability, exception chronology and monitoring discipline. It is not proof of Citi deployment, regulatory compliance, trading profitability, a validated alerting model or portfolio-wide bank capital risk estimation.
