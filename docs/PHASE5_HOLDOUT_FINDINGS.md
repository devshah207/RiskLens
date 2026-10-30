# RiskLens — Phase 5 Observed Holdout Findings

**Source/evidence status (updated 30 October 2026):** The original one-time Mac holdout verdict, full `final_holdout_report.json`, saved 433-row holdout probabilities and 1,238-row development probabilities were subsequently supplied. Source fingerprints, prediction-score metrics and original monitoring PSI were cross-checked; the NIFTY source also permitted independent 95%/99% historical VaR exception reconstruction. This is retrospective audit of an **already-opened** holdout, not new model fitting or bank-grade source authentication.

**Frozen 2025–2026 assessment:** 433 aligned observations, zero exclusions. Models not retuned.

| Candidate | Brier ↓ | Log loss ↓ | ROC-AUC ↑ | AP ↑ |
|---|---:|---:|---:|---:|
| Fixed XGBoost (11 features) | 0.084773 | 0.303295 | 0.67275606 | 0.18610282 |
| Optuna LSTM trial #86 (45-session) | **0.084299** | **0.300362** | **0.68487395** | **0.21897271** |

LSTM leads on the four pooled metrics; Brier margin is 0.000474 (about 0.56% of XGBoost Brier), small enough to avoid grand performance claims. No significance test is established by this report.

| Year | N | Events | Event rate | XGBoost Brier | LSTM #86 Brier | XGBoost ROC-AUC | LSTM ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025 | 249 | 15 | 6.02% | **0.055743** | 0.057976 | 0.69344729 | 0.71538462 |
| 2026 through Sep | 184 | 27 | 14.67% | 0.124057 | **0.119922** | 0.61948573 | 0.63694268 |

2025 XGBoost had better Brier; 2026 LSTM had better Brier. Annual event prevalence and probabilistic losses deteriorated substantially into 2026; investigate drift, calibration and changing regimes, not post-hoc model-selection.

**Historical simulation VaR:** 95% = 30 breaches / 433 (6.9284%; expected 5%, exact coverage p=0.07631016); 99% = 7 / 433 (1.6166%; expected 1%, exact coverage p=0.21602421). Both exceed their target exception rates, but these exact tests do not reject at the conventional 5% level. The full JSON was subsequently inspected: at 95%, approximate exception-independence p=0.01403 and conditional-coverage p=0.01069 flag clustering; at 99%, independence p=0.63068 and conditional p=0.44206 do not reject. In March 2026, 9/19 scored prediction dates breached 95% VaR and four breached 99% VaR. Interpretation remains subject to small-tail-sample and Markov/asymptotic assumptions.

**Model governance:** Fixed XGBoost remains the *preselected champion candidate* and LSTM trial #86 remains a challenger. No model switching, further Optuna tuning, feature selection, threshold optimization or recalibration based on holdout.

**Phase 5 status (30 October 2026):** One-time research holdout completed and its report, saved predictions and original source-provenance contracts audited. The original one-time test remains closed to new selection decisions; vendor release times, data rights and production/regulatory validation remain unverified. The scientific evidence and risk register are tracked in the Phase 6 governance review.

## Source limitations

- Structural validation of VIX/S&P dates and point-in-time lag rules does not verify vendor publication timestamps or rights to redistribute the raw data.
- The index-level 250-day historical VaR monitor is separate from ML downside probabilities.
- Rare events, financial nonstationarity and possible tail dependence limit inference; no production or regulatory approval is implied.
