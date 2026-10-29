# RiskLens — Phase 5 Observed Holdout Findings

**Origin:** User-provided `final_holdout_verdict.md` from the one-time local holdout run, shared 2026-10-30. **Provenance limitation:** The full `final_holdout_report.json`, raw source hashes and predictions were not provided in this conversation for independent re-audit. These metrics are transcribed from the provided verdict; they have not been recomputed here.

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

**Historical simulation VaR:** 95% = 30 breaches / 433 (6.9284%; expected 5%, exact coverage p=0.07631016); 99% = 7 / 433 (1.6166%; expected 1%, exact coverage p=0.21602421). Both exceed their target exception rates, but these exact tests do not reject at the conventional 5% level. Exception independence and conditional coverage are stored in the full JSON report, which has not yet been inspected here.

**Model governance:** Fixed XGBoost remains the *preselected champion candidate* and LSTM trial #86 remains a challenger. No model switching, further Optuna tuning, feature selection, threshold optimization or recalibration based on holdout.

**Phase 5 status:** One-time holdout result received and recorded; complete source/JSON/prediction provenance audit remains open. These findings inform Phase 6 descriptive monitoring but must not be used as another model-selection set.

## Source limitations

- Structural validation of VIX/S&P dates and point-in-time lag rules does not verify vendor publication timestamps or rights to redistribute the raw data.
- The index-level 250-day historical VaR monitor is separate from ML downside probabilities.
- Rare events, financial nonstationarity and possible tail dependence limit inference; no production or regulatory approval is implied.
