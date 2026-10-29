# RiskLens — Phase 3, Step 3.10
## Does external market information improve next-session NIFTY downside probabilities?

**Read this guide before changing parameters or opening final-holdout data.**

### The question

We want to compare the same fixed-parameter XGBoost classifier using:

- **Model A (9 features):** three closing-price indicators (1-day and 5-day return, 20-day historical volatility), three OHLC indicators (intraday range, intraday return, opening gap), and three technical indicators (RSI-14, relative position versus EMA-50, 10-day volatility).
- **Model B (11 features):** Model A plus `india_vix_level` and `sp500_previous_return_1d`, taken from the already-validated **Phase 3.9** merged data.

The target remains `downside_next_session`: **1 if the NIFTY close-to-close return at the next observed session is at most -1%; otherwise 0**. We predict after the current Indian trading session closes. We are *not* predicting the magnitude of a crash, whether NIFTY will rise, or an investment decision.

### What do the external features mean?

**India VIX** is a market index built from prices of NIFTY options. It measures **implied volatility**, a market-derived measure of expected variation in future prices. A VIX of 20 represents roughly 20% **annualized** implied volatility under the index convention. It does **not** mean NIFTY will fall 20% tomorrow, nor does it tell us whether prices will rise or fall. Compared to trailing NIFTY volatility, VIX adds information from a *different market mechanism* (options pricing).

**S&P 500** is a broad US stock-market index. We use its last fully available *close-to-close simple return*. If the previous US close was 5,000 and the latest completed US close was 4,950, that return is `4950/5000 - 1 = -0.01`, or **-1%**. This is a **US movement**, not NIFTY's return.

The experiment tests whether these two observable values contribute predictive information beyond our nine original NIFTY-derived features. Neither is guaranteed to help.

### Why use earlier source dates?

The model's decision moment is **after NIFTY session t closes**. The US exchange closes much later in Indian time, so today's US close would not be known at NIFTY's closing time.

Phase 3.9 deliberately uses an observation whose *source calendar date is strictly earlier* than the Indian date. For example, for a Monday Indian prediction, the completed Friday US close may be usable; Monday's US close is not. The same prior-date-only rule is conservatively applied to India VIX until same-day release timing is independently established. Observations older than **7 calendar days** are unavailable, not carried forward forever.

The comparison independently rechecks each source date, actual calendar-day age, missing-value flag and the NIFTY OHLC source. It refuses any 2025+ integrated row. This is **point-in-time input discipline**: use only information plausibly known at the historical prediction time. A verified time-of-day publication record would permit a more precise as-of join in a later project phase.

### Walk-forward validation, simply

Imagine you are standing at the end of 2019. You can study 2016–2019 returns and outcomes, but you do not know what will happen in 2020. You train your model, then score 2020 without modifying your fit using 2020 outcomes.

We repeat the process:

| Fold | Training period | Evaluation year |
|---|---|---|
| 1 | 2016–2019 | 2020 |
| 2 | 2016–2020 | 2021 |
| 3 | 2016–2021 | 2022 |
| 4 | 2016–2022 | 2023 |
| 5 | 2016–2023 | 2024 |

In every fold, **the last training prediction row whose label requires the first validation close is removed**, and the last validation row requiring a close outside that fold is also removed. This is called a **boundary purge**. It protects our labels, not just our input features.

The data loader builds all indicators using prices *up to each prediction date*. The target alone is allowed to use the **next observed** NIFTY close for historical supervised learning. That next close is never an input predictor.

### The critical fairness test: identical dates and labels

A tempting but invalid comparison is to fit the nine-feature model on all its usable rows, then fit the eleven-feature model on fewer available rows, and compare their performance. The outcome difference might merely be due to different days.

Our code instead:

1. Builds NIFTY features and split-boundary labels **once per year** using `build_development_features(..., include_ohlc_features=True, include_technical_features=True)`.
2. Merges earlier-dated external inputs by the exact NIFTY date.
3. Removes any row with missing VIX *or* S&P 500 return **from both model versions together**. No value is invented and no future interpolation is used.
4. Fits two **new** XGBoost estimators with the **same** fixed parameters on the exact same training rows and target labels; only feature lists differ.
5. Scores both models on the exact same validation dates and event outcomes.

The earlier technical-model experiment already removed the EMA's first 49 price rows for a fair comparison, and this step retains that same warm-up. In our real downloaded development files, the few missing external rows are all in those initial warm-up dates, so no 2020–2024 validation observations are lost.

### What exactly is XGBoost doing?

**XGBoost** (eXtreme Gradient Boosting) builds many small decision trees and combines them. Each tree asks questions like “Is the historical volatility greater than this value?” The next tree learns to correct errors from the preceding trees. This is a flexible, nonlinear classifier, but flexibility can also fit noise. We reuse the original predeclared parameters from Step 3.5: `n_estimators=80`, `max_depth=2`, `learning_rate=0.04`, `min_child_weight=10`, `gamma=1`, `subsample=0.8`, `colsample_bytree=0.9`, `reg_lambda=10`, and all other settings unchanged. No scaler, no hyperparameter search, no early stopping on validation outcomes.

Model A receives 9 predictors. Model B receives 11. This is a controlled *feature ablation*: comparing models where we change which input information they can see.

### Read the three evaluation metrics

**Brier score** is the mean squared error of a probability prediction. If a model predicts risk `p = 0.20` and the event occurs (`y=1`), the contribution is `(0.20-1)^2=0.64`; if it does not occur (`y=0`), the contribution is `0.20^2=0.04`. **Lower is better**.

**Log loss** penalizes confident wrong forecasts especially strongly. A model that assigns 1% probability to an event that actually happens is heavily penalized. **Lower is better**.

**ROC-AUC** asks whether event days tend to get higher risk scores than non-event days. AUC around 0.5 indicates near-random ranking. Yearly ROC-AUC can be noisy because our target is uncommon; **higher is better**, but good ranking alone does not prove calibrated probabilities.

A **constant baseline** ignores features and forecasts the *training* event frequency for every validation date; it is an essential reference. This baseline uses no validation labels for estimating its rate.

### Actual results (NIFTY downloaded data, 2020–2024 development folds)

The comparison produces **1,238** validation predictions and **145** observed downside events. Across all these dates:

| Model | Pooled Brier ↓ | Pooled log loss ↓ |
|---|---:|---:|
| Constant baseline | 0.10462 | 0.36840 |
| Fixed XGBoost, 9 features | 0.10291 | 0.35940 |
| Fixed XGBoost, 11 features | **0.10280** | **0.35815** |

Pooled Brier improvement: about `0.10291 - 0.10280 = 0.00011` in absolute score, about 0.11% relative to the nine-feature score. These figures are rounded in the table; see JSON for eight-decimal values.

Year-by-year (Brier, lower is better):

| Evaluation year | 9 features | 11 features | Events |
|---|---:|---:|---:|
| 2020 | 0.14118 | **0.14053** | 41 |
| 2021 | 0.10053 | **0.10036** | 29 |
| 2022 | **0.12710** | 0.12887 | 37 |
| 2023 | 0.05440 | **0.05406** | 14 |
| 2024 | 0.09037 | **0.08925** | 24 |

This is a *slight, mixed development improvement*, not robust evidence of forecasting ability. In 2024 both XGBoost variants still lose to the year's constant baseline on Brier score (baseline about 0.08742), and their ROC-AUC values are below 0.5. Adding data did not resolve the instability of high-risk day ranking.

**Important methodological limitation:** The team already explored this same 2020–2024 period with multiple feature sets and algorithms. We are not obtaining a fresh unbiased estimate by trying another feature combination on it. Do **not** optimize the model after observing these development scores or declare the 11-feature version to be the production winner. The 2025+ untouched holdout is reserved for one prespecified final evaluation **after** all decisions are frozen.

### Files and functions

New files:

- `src/risklens/models/multifactor_comparison.py`: joins the approved data, crosschecks original NIFTY OHLC values, enforces strict external source dates and equal samples, runs five yearly XGBoost comparisons, and creates the predictions/report.
- `src/risklens/models/multifactor_comparison_cli.py`: friendly terminal command.
- `tests/test_multifactor_comparison.py`: synthetic adversarial and chronology tests.
- This guide.

One existing source file is minimally updated:

- `src/risklens/models/xgboost_walk_forward.py`: adds the two external column names as an *approved fourth feature tuple* for the already-existing `_fit_predict_fixed_xgb` function. Its estimator settings and older supported feature sets remain unchanged.

Key code:

```python
EXTERNAL_FEATURE_COLUMNS = ("india_vix_level", "sp500_previous_return_1d")
MULTIFACTOR_FEATURE_COLUMNS = FULL_FEATURE_COLUMNS + EXTERNAL_FEATURE_COLUMNS

# Source OHLC and timing checked first.
train = train.merge(external, on="date", how="left", validate="one_to_one", sort=False)
validation = validation.merge(external, on="date", how="left", validate="one_to_one", sort=False)

# Same eligible rows go to both models.
train = train.loc[train[list(EXTERNAL_FEATURE_COLUMNS)].notna().all(axis=1)]
validation = validation.loc[validation[list(EXTERNAL_FEATURE_COLUMNS)].notna().all(axis=1)]

nine, _ = _fit_predict_fixed_xgb(train, validation, FULL_FEATURE_COLUMNS)
eleven, _ = _fit_predict_fixed_xgb(train, validation, MULTIFACTOR_FEATURE_COLUMNS)
```

- `train.merge(...)` combines historical technical and external columns by exact dates. It is **not** the risky as-of join: Step 3.9 already performed the *strictly earlier source-date join*, and Step 3.10 revalidates its provenance.
- `validate="one_to_one"` stops accidental duplicate-date expansion.
- `.notna().all(axis=1)` excludes a row missing either external feature. Both variants see the filtered `train` and `validation`, so they remain comparable.
- `_fit_predict_fixed_xgb` trains on `train` labels only, then returns probabilities for the distinct `validation` dates. Training and validation outcomes never mix.
- `report['source_audit']` holds SHA-256 fingerprints for provenance. Hashes make a file reproducible; **they do not independently certify authenticity**.

### Running on your computer

You already ran Step 3.9 and have the merged CSV. From the RiskLens repo root:

```bash
unzip -o ~/Downloads/RiskLens_Phase3_Step3_10_Changes.zip -d .
source .venv/bin/activate
python -m pytest -q
python -m risklens.models.multifactor_comparison_cli \
  data/processed/multiyear/nifty50_ohlc.csv \
  data/processed/external/nifty50_external_development.csv \
  --output-dir data/processed/multifactor_comparison
```

Outputs (kept outside Git in `data/processed/`):

- `multifactor_comparison_predictions.csv` — date, observed outcome, baseline risk, nine-feature risk, eleven-feature risk for each of 1,238 historical validation observations.
- `multifactor_comparison_report.json` — pooled probability scores, yearly metrics, training sample sizes, source SHA-256 hashes, fixed parameter declaration, audit warnings.

Reruns refuse overwriting existing outputs unless `--overwrite` is provided.

### Next responsible research step

**Freeze and document model-selection and final-evaluation rules.** Choose the model using a declared rationale that acknowledges the repeated development exploration. Specify in writing the final model, training data cutoff, handling of missing external data, target definition, scoring metrics, and conditions under which the final holdout will be opened. Only then perform **one** untouched evaluation on 2025–2026 outcomes. Inability to authenticate the S&P data source or use it under appropriate terms should be resolved before any production-like claims.
