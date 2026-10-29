# RiskLens — Phase 6 Scientific Review and Model-Risk Decision

**Status:** DRAFT FOR OWNER REVIEW — published to a separate review branch; no model promotion  
**Review date:** 30 October 2026  
**Study:** NIFTY 50 next-session loss of at least 1%; predictions made after the current session closes  
**Evaluation scope:** Development 2020–2024; original single locked holdout 2025 through September 2026

## Executive decision

**Preserve the preselected 11-feature fixed XGBoost as the research champion and the frozen Optuna LSTM Trial #86 as challenger.** No feature removal, regime-context replacement, seed selection, threshold tuning, or holdout-based retraining is warranted by the studies completed to date. The retrospective Phase 6 analysis is substantively complete, but a clean scientific sign-off should explicitly address three limitations: (1) the LSTM's reported score PSI was measured across differently refitted development and holdout models; (2) historical 95% VaR exceptions cluster and its conditional coverage test fails at 5% under the stated approximation; and (3) a 0.5 classifier cutoff produces no positive signals. **This project is a research prototype, not an operational risk or trading system.**

## Source evidence and audit boundary

Reviewed the uploaded `risklens_phase6_final_review.zip` (the Phase 5 frozen-holdout JSON plus 11 Phase 6 monitoring, TreeSHAP and LSTM sensitivity files), and cross-referenced the previously supplied feature-drift investigation, eight-variant ablation research, and seven-seed stability research. Checked the repository's `monitoring.py`, `final_holdout.py`, and `risk_validation.py` read-only on the `main` branch.

**Integrity checks:** 71 report-level numerical, structural and cross-file checks passed. Verified the annual observation/event totals, pooled weighted Brier and log-loss values, model/holdout metric agreement, score histogram counts and smoothed PSI, SHAP units and additivity tolerance, sensitivity Brier-change arithmetic, cross-report source and trial identifiers, and VaR exception/conditional-test arithmetic. The provenance fingerprint of the monitoring holdout predictions also matches the previously delivered three-feature investigation's stored prediction-file fingerprint.

**Boundary:** This is a verification of submitted outputs and relevant source-code contracts. Raw 2025–2026 model-prediction CSVs were not included in the final-review ZIP; however, the original NIFTY OHLC source was available from a previous upload and its SHA-256 matched the holdout provenance, allowing independent reconstruction of the historical VaR series and exception dates. No full raw-data rerun of ML model fits or vendor timestamp audit was performed. Reported source hashes do not independently prove vendor release-time availability.

## Frozen model results — no re-scoring

| Measure | Fixed XGBoost champion | LSTM Trial #86 challenger |
|---|---:|---:|
| Scored holdout rows | 433 | 433 |
| Actual downside events | 42 | 42 |
| Holdout ROC-AUC | 0.672756 | 0.684874 |
| Holdout average precision | 0.186103 | 0.218973 |
| Holdout Brier (lower better) | 0.084773 | 0.084299 |
| Holdout log loss (lower better) | 0.303295 | 0.300362 |
| Mean probability | 0.094258 | 0.113031 |
| Observed event rate | 0.096998 | 0.096998 |

Both score sets have *some* discriminative ranking ability. The LSTM appears numerically stronger on these metrics, but no claim of a statistically established superiority or independent generalization improvement follows from this one opened holdout. The preselected champion designation does not change.

**Annual instability:** The observed event rate rose from 15/249 = **6.02%** in 2025 to 27/184 = **14.67%** in 2026. The XGBoost holdout AUC changed from 0.6934 to 0.6195; LSTM from 0.7154 to 0.6369. In 2025, mean predicted risks were above event frequency (XGBoost 8.39%, LSTM 9.81% vs 6.02%). In 2026 both were below it (XGBoost 10.83%, LSTM 13.32% vs 14.67%). Pooled calibration-in-the-large values therefore conceal opposing annual biases.

**Operating-point limitation:** At an illustrative 0.50 probability cutoff, both models predicted 0 positives out of 433 rows, producing 42 false negatives and zero true positives. This means the fixed cutoff has **0% sensitivity** for the downside event on this holdout. AUC does not imply useful binary event alerts at 0.5. Do **not** tune a new cutoff against these now-viewed outcomes; treat alert-policy definition as prospective work requiring a separate validation framework.

## Explainability audit

**Fixed XGBoost (in-sample development explanations):** The original frozen 11-feature configuration was refitted on 2,179 pre-2025 labeled observations and explained on 500 recent pre-2025 rows. Native XGBoost SHAP contributions were measured in raw margin/log-odds, *not* probability units. The recorded maximum additivity discrepancy was **1.65 × 10^-6**. Top features by mean absolute contribution were India VIX **0.2451**, RSI-14 **0.1928**, and opening gap **0.0932**; EMA-50 distance was **0.0229**. This ordering is model- and sample-specific and is neither causal attribution nor proof that EMA should be deleted.

**LSTM (permutation sensitivity on 2024 development only):** The validated original Brier was **0.090561**, evaluated over 248 rows with 24 events. Three trajectory-shuffle repetitions per feature yielded approximate Brier increases of **+0.000621 for RSI-14** and **+0.000580 for India VIX**; the change for EMA distance was **−0.000109**. These perturbations break correlations among features. Their size and ranking must not be compared directly with SHAP or interpreted as a causal feature-removal test. Three repetitions and 24 events are limited evidence.

## Three-feature drift investigation

- **EMA-50 distance:** Independent calculation integrity passed. Mean changed from **+0.630% in 2025** to **−1.144% in 2026**. Only 38.6% of 2026 evaluated sessions had a positive distance. This is a material shift toward a below-EMA trend state, not a demonstrated calculation error.
- **RSI-14:** Independent Wilder formula integrity passed. Mean changed from **52.39 in 2025** to **46.03 in 2026**, with more weak-momentum sessions. A genuine distributional change, not a demonstrated formula defect.
- **India VIX:** 248 matched 2024 source-overlap sessions had zero value or source-date discrepancies (one overlap session lacked a matched observation). All holdout VIX source dates were strictly prior to the prediction date, up to four calendar days old. Overall held-out VIX PSI falls from **0.613 vs a 2020–2024 reference** to **0.134 vs 2023–2024**; the 2020–2022 high-VIX reference contributes materially to the apparent drift. Exchange publication timestamps for all historical observations have not been independently authenticated.

PSI is an investigation heuristic, not proof of model damage or financial causation.

## Feature hypothesis and seed robustness verdict

The eight-variant development-only experiment compared the original 11 features with individual and combined ablations, regime flags, and relative-VIX representations using fixed original model settings. Dropping India VIX generally degraded ranking across both models and historical periods. The tempting **single-seed** improvement from regime context on LSTM did not survive a seven-seed targeted experiment (70 training runs over five development years): median pooled 2020–2024 ΔAUC **+0.00064**, only **4/7** seeds positive; median 2023–2024 ΔAUC **−0.01591**, only **2/7** positive; in 2023 **0/7** seeds improved AUC or Brier. The context representation was much more initialization-sensitive. **No evidence warrants promoting it or removing all three features.** These experiments were motivated after looking at the opened holdout and are expressly exploratory, not new independent validation.

## Monitoring risk exceptions requiring explicit treatment

| Risk | Verified finding | Interpretation and control |
|---|---|---|
| **MR-02a: LSTM score PSI** | LSTM **0.47019** vs XGBoost **0.06829**; LSTM development had 0 predictions in the 0.30–0.50 bin while holdout had 12. | **Investigate**, but do not equate this PSI with pure input-regime drift: the 2020–2024 development scores came from different annual refits while 2025–2026 came from one fixed pre-2025 refit. The baseline is not from a single identical fitted scoring function. Review annual bins/score distribution and the model-version comparator before assigning a root cause. |
| **MR-01: Annual event-rate/calibration changes** | Observed event rate **6.02% in 2025**, **14.67% in 2026**. Both models overpredicted average risk in 2025 and underpredicted in 2026. | Annual controls should be emphasized over pooled flags; retain locked predictions, do not recalibrate on holdout. |
| **MR-05: 95% historical VaR exceptions** | **30/433 = 6.93%** vs 5% expected. Exact unconditional binomial p=**0.0763**; unconditional LR p=**0.0811**; independence p=**0.0140**; conditional coverage p=**0.0107**. **8 exceptions in 2025, 22 in 2026.** | Do not say unconditional coverage conclusively failed at 5%. Exception clustering and the approximate joint test **do** flag model-risk weakness under their assumptions. This historical return-quantile VaR is **separate** from the two ML classifiers. Study exception chronology, overlapping dates, and reference-window sensitivity as documentation/research, not retuning against the holdout. |
| **99% historical VaR** | **7/433 = 1.62%** vs 1% expected; exact p=**0.216**, conditional coverage p=**0.442**. | No statistically persuasive rejection from these tests; rare-event sample remains small. |
| **MR-06: 0.5 action cutoff** | 0/42 detected downside events for either classifier at threshold 0.5. | Risk scores are research outputs, not validated trigger-based alerts; any eventual alert threshold must be predeclared and independently evaluated. |
| **MR-07: Data governance** | Prior-date matching and formula/source-overlap checks passed, but vendor publication timestamps and data licensing not authenticated. | No live or bank-approved use; make availability and license checks separate prerequisites. |

## Independent historical-VaR exception reconstruction

Using the **byte-matching original NIFTY OHLC file** (`SHA-256 e60289f9cf45eb099a62a9a0ba4dbaeb9927cf365666ede722b24a1fcec48a58`), independently computed `return[t] = close[t]/close[t−1]−1`, a 250-observation rolling **5th / 1st percentile** of observed returns through prediction date *t*, and compared the **next-session return** at *t+1* with the frozen loss thresholds. All dates from 2025-01-01 through 2026-09-29 yield exactly **433** predictor rows with known next-session outcome. The independent reconstruction reproduces the report's **30 95% exceptions (8 in 2025; 22 in 2026)**, **7 99% exceptions (1 in 2025; 6 in 2026)**, and **95% exception transition counts `[[377,24],[24,6]]`** when annual boundaries are separated.

| Prediction-date period | Sessions | 95% breaches | 99% breaches |
|---|---:|---:|---:|
| 2025 full year | 249 | 8 | 1 |
| 2026 January | 20 | 2 | 1 |
| 2026 February | 21 | 5 | 0 |
| **2026 March** | **19** | **9** | **4** |
| 2026 April–September | 124 | 6 | 1 |

**Interpretation:** The exceptional clustering is localized, especially to March 2026; 9 of that month's 19 forecast-date observations exceeded the 95% historical loss quantile and 4 exceeded the 99% quantile. The 250-session trailing quantiles were relatively low entering the adverse-return sequence. The return series and exception chronology explain the mechanical VaR failures, **not the external economic cause** of the adverse returns. All CSV dates are *prediction dates*; the realized return is the return of the next observed session, not the same-day close. Supporting file: `RiskLens_VaR_Independent_Reconstruction.csv`.

## Owner-review decisions

1. **Approve retaining the original frozen XGBoost champion and Trial #86 challenger**; do not rewrite the original holdout verdict.
2. **Approve classifying the three-feature drift and ablation/seed-stability questions as investigated for the research prototype**, with no verified reparable input defect and no reproducible feature-engineering win.
3. **Require a focused explanation of LSTM score PSI**, especially the annual-refit versus final-model comparator issue, before presenting the monitoring alert as a verified regime effect. Existing prediction files suffice for retrospective analysis; no model re-scoring is authorized.
4. **Record the historical 95% VaR clustering, annual calibration shifts, and 0.5 threshold's zero sensitivity as explicit limitations**. No threshold optimization, VaR-window tuning, or holdout-based model promotion.
5. **Classify Phase 6 as research-reviewed, pending risk-register/documentation updates; not production approved**. Phase 7 document-retrieval work may proceed separately once these findings have been accepted and filed.

## Publication control

This report is **review-only**. The owner approved publication to a separate review branch only. A merge into `main` or any later GitHub push requires a separate explicit approval.