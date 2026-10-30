# RiskLens — Phase 6 saved-prediction score PSI independent audit
**Date:** 30 October 2026

**Status:** Retrospective scientific audit for owner review; not production validation or a model-selection change.

**Source:** Original frozen development (2020–2024) and one-time holdout (2025–September 2026) saved prediction CSVs uploaded by the project owner. No model retraining or prediction regeneration.

## File integrity, provenance and computation
- `frozen_model_predictions.csv`: SHA-256 `6d1fdc455ed30b35c594955dc22e7af1a1fd450a3246ee3dc367ebff6577874a`
- `final_holdout_predictions.csv`: SHA-256 `47f84bb4e0242cd03de5a46cd4cb88120a9ad36c5841064fa6c21484d08f162e`
- Both SHA-256 values match `phase6_monitoring/monitoring_summary.json` source-fingerprint fields.
- Development: **1,238 rows (145 events), five validation years 2020–2024**; holdout: **433 rows (42 events), 249 in 2025 and 184 in 2026 (through 29 September)**. No duplicate scored dates, null model scores, or probabilities outside [0,1].
- Independent recalculation of both models' development and holdout ROC-AUC, AP, Brier and log loss agrees with the original JSON reports (to displayed precision).
- Fixed histogram edges `[0, .05, .10, .15, .20, .30, .50, 1]`; exactly **0.5 pseudocount per bin** before each group's normalization; PSI = sum over bins of `(holdout_proportion - development_proportion) * ln(holdout_proportion / development_proportion)`.

## Reproduction of original score-PSI alert
| Model | PSI independently calculated | Saved PSI | Conclusion |
|---|---:|---:|---|
| `xgboost_fixed` | 0.06829280 | 0.06829280 | Exact to report precision |
| `lstm_optuna_trial86` | 0.47019283 | 0.47019283 | Exact to report precision |
The research investigation threshold 0.20 is a **heuristic**, not a formal test or regulatory limit. Distinguish input-feature PSI from output-score PSI.

### Exact fixed-bin observed counts
| Score range | XGB dev | XGB holdout | LSTM dev | LSTM holdout |
|---|---:|---:|---:|---:|
| [0.00, 0.05) | 181 | 56 | 365 | 93 |
| [0.05, 0.10) | 577 | 204 | 335 | 144 |
| [0.10, 0.15) | 310 | 126 | 436 | 77 |
| [0.15, 0.20) | 111 | 43 | 76 | 66 |
| [0.20, 0.30) | 59 | 4 | 26 | 41 |
| [0.30, 0.50) | 0 | 0 | 0 | 12 |
| [0.50, 1.00] | 0 | 0 | 0 | 0 |
For LSTM, scores at least 0.20: **26/1,238 = 2.10%** development versus **53/433 = 12.24%** holdout. Scores [0.30,0.50): **0** development, **12** holdout. No probability reached the fixed 0.50 illustration threshold in either population.

## Annual scores and prevalence
| Year | n | Events | Event prevalence | XGB mean score | LSTM mean score | LSTM >=0.20 | LSTM [0.30,0.50) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 251 | 41 | 16.33% | 9.31% | 9.63% | 0 | 0 |
| 2021 | 247 | 29 | 11.74% | 9.87% | 8.29% | 2 | 0 |
| 2022 | 247 | 37 | 14.98% | 13.34% | 11.48% | 22 | 0 |
| 2023 | 245 | 14 | 5.71% | 7.24% | 6.83% | 2 | 0 |
| 2024 | 248 | 24 | 9.68% | 8.34% | 7.87% | 0 | 0 |
| 2025 | 249 | 15 | 6.02% | 8.39% | 9.81% | 20 | 2 |
| 2026 | 184 | 27 | 14.67% | 10.83% | 13.32% | 33 | 10 |
The **same fixed final fitted model** generated both 2025 and 2026 holdout scores, whereas each development validation year used a different trained model instance. In 2025, both models overpredicted average event frequency; in 2026 both underpredicted it.

## Alternative descriptive score PSI comparisons — fixed bins and same smoothing
| Comparison (first = reference; second = comparison) | XGB PSI | LSTM PSI |
|---|---:|---:|
| All 2020–2024 annual-refit predictions → all 2025–2026 | 0.068293 | 0.470193 |
| All 2020–2024 → 2025 | 0.209678 | 0.374510 |
| All 2020–2024 → 2026 | 0.176979 | 0.871749 |
| 2023–2024 annual-refit predictions → 2025–2026 | 0.142137 | 0.719101 |
| 2025 → 2026, **same final fitted model** | 0.425183 | 0.423342 |
**Critical nuance:** The within-holdout comparison shows a substantive 2025-to-2026 change in output-score distribution for *both* fixed model instances. It cannot be attributed to annual retraining between 2025 and 2026. However, it does **not** identify the financial causes or establish whether individual forecasts were useful. The 2020–2024 versus final-model PSI remains confounded by differences in training/model instances, as well as changes in input/market regimes. These descriptive PSI values are sensitive to reference window, score binning, smoothing and sample size.

## Concentration of high LSTM scores
- **12** holdout rows with LSTM score in [0.30,0.50): **2 in 2025**, **10 in 2026**.
- Of 2026 high-score observations, **9 in March 2026**, **4 labeled downside events**; the other high-score observation is in April 2026.
- The Phase 5 separate historical 95% VaR investigation also documented clustered March 2026 exceptions. Co-occurrence is a descriptive observation, **not** proof that high LSTM probabilities predict VaR threshold breaches or of a particular market mechanism.

### High-score dates (for reproducibility)
| Date | LSTM probability | Downside event observed next session? |
|---|---:|---|
| 2025-03-07 | 0.3081 | No |
| 2025-03-11 | 0.3170 | No |
| 2026-03-09 | 0.3358 | No |
| 2026-03-10 | 0.3936 | Yes |
| 2026-03-11 | 0.3581 | No |
| 2026-03-12 | 0.3437 | Yes |
| 2026-03-16 | 0.3506 | No |
| 2026-03-17 | 0.4004 | No |
| 2026-03-18 | 0.4355 | Yes |
| 2026-03-19 | 0.3712 | No |
| 2026-03-20 | 0.3403 | Yes |
| 2026-04-08 | 0.3130 | No |

## Scientific verdict and governance recommendations
1. **PSI computation verified.** The LSTM score PSI of 0.47019 is not an arithmetic defect. Its probability outputs shifted toward higher-risk bins; a fixed 2025–2026 model also shows within-model year-to-year score shifts.
2. **Root cause only partially resolved.** The entire pooled development-to-holdout PSI cannot be attributed uniquely to market regime because the development comparator pools five different annual refits. Reference-window sensitivity is demonstrable; a fixed-model in-sample development score series does not exist in these original saved data. No weight archive means exact retracing would require new fitting, which is not warranted for this frozen analysis.
3. **XGBoost is not exempt from annual score changes.** Although pooled PSI=0.06829 is below the heuristic 0.20 flag, its same-model 2025-vs-2026 PSI=0.42518 is an annual distribution comparison worth recording, not evidence of production-level stability.
4. **No authorized model change.** Preserve frozen 11-feature XGBoost champion, true Optuna trial #86 LSTM challenger, existing train cutoff and original 433 holdout results; no rescore, thresholds, calibration, feature changes, new Optuna or promotion from the opened period.
5. **Disposition:** Mark the *saved-predictions comparator audit* complete, but keep MR-02 as **verified monitoring concern, cause not uniquely identified, research monitoring follow-up**. Phase 6 itself still requires owner review, documentation alignment, tests/CI and separately authorized PR merge. Research sign-off cannot be equated to production validation.

## Frozen governance and Git state (owner-confirmed 30 October 2026)
- Owner ran `python scripts/verify_phase6_freeze.py` **after local branch synchronization**: **PASS: all 9 frozen evidence hashes and champion/challenger / original holdout contracts match**.
- Local and tracked remote review branch are now at **`4aa006e`**, reached by safe fast-forward from `9158a6d`; the earlier local manifest modification was backed up and stashed, **not** reapplied.
- The recorded Git status describes the checkout immediately after branch synchronization, before the governance documentation edits were staged. Git status is a historical snapshot, not a permanent project invariant.
- GitHub PR #6 was last independently checked as **open and DRAFT** with `main` unmerged. Every subsequent push needs owner approval and each merge requires separate approval.
- `weights_archived=false`: byte fingerprints and specifications are not serialized fitted model instances. This audit does not open the 2025–2026 holdout for any new model decisions, certify source rights, or authorize deployment.
