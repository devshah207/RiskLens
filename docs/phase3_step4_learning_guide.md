# RiskLens Phase 3, Step 3.4 — Does intraday OHLC information help?

**Goal:** Compare the original three closing-price predictors against the same three plus three OHLC-derived predictors, under identical walk-forward year splits and fixed Logistic Regression settings. We do **not** add XGBoost, India VIX, or inspect 2025–2026 holdout outcomes.

## 1. The finance vocabulary (start here)

**NIFTY 50**: an index, or numerical benchmark representing a defined group of large companies listed on India's National Stock Exchange. An **index level** is not the price of one company share.

**Trading session**: a day when the market trades. Next *trading session* is not necessarily next calendar day.

**OHLC**: **Open** = index level when the session starts; **High** = highest session level; **Low** = lowest session level; **Close** = official closing index level.

**Feature**: a numerical measurement known when we make a prediction, after session `t` closes. **Target/label**: the next session's outcome, calculated later for supervised training.

Example market observations:

| Value | Level |
| --- | ---: |
| Previous session close | 25,000 |
| Today's open | 25,100 |
| Today's high | 25,400 |
| Today's low | 24,900 |
| Today's close | 25,200 |

The three **new** features use only today's OHLC and the previous known close:

1. `intraday_range = (high[t] - low[t]) / open[t]` → `500/25,100 ≈ 0.01992`, or **1.99%**. This describes the width of today's trading range, *not* its direction.
2. `intraday_return = close[t] / open[t] - 1` → `25,200/25,100 - 1 ≈ 0.003984`, or **+0.40%**. This describes how the index moved between opening and closing.
3. `opening_gap = open[t] / close[t-1] - 1` → `25,100/25,000 - 1 = 0.004`, or **+0.40%**. This describes the change between yesterday's close and today's opening.

**All three values are known after today's close**. A `shift(1)` in pandas means *yesterday*, which is allowed; `shift(-1)` means *tomorrow* and is used only when constructing the historical target.

### Important: Some features are related, not independent

The day's **close-to-close** return is mathematically composed of the opening gap and intraday return:

```
1 + return_1d = (1 + opening_gap) * (1 + intraday_return)
```

In other words, splitting a day into an overnight/opening move and a within-day move adds *a different description* of the same total movement; it doesn't create independent data. Highly related predictors can make linear coefficients less stable. Our Logistic Regression uses L2 regularization, but redundancy is still worth noting. The intraday range contains a different type of information: within-session spread.

## 2. Experiment design — change ONE factor

| Unchanged | Changed |
| --- | --- |
| Same NIFTY 50 OHLC source CSV | **Only** add three features |
| Same next-session downside event (return <= -1%) |  |
| Same five validation years: 2020–2024 |  |
| Same expanding training windows: all dates before that validation year |  |
| Same boundary purge and initial 20-return warm-up |  |
| Same training-only StandardScaler and fixed LogisticRegression (`C=1.0`, L2, `lbfgs`, `max_iter=2000`, no class weights) |  |
| Same baseline (constant historical training event rate) |  |
| No 2025–2026 outcomes, no parameter tuning |  |

- **Original predictors (3):** `return_1d`, `return_5d`, `volatility_20d`.
- **Expanded predictors (6):** those three plus `intraday_range`, `intraday_return`, `opening_gap`.

This is a **development-period feature ablation**: compare otherwise identical models that differ by a feature group. It is *not* final independent evaluation. We have already viewed validation results, so many follow-up trials on these same years could gradually overfit our research decisions to them.

## 3. File-by-file explanation and important actual code

### `src/risklens/features/dataset.py` (existing file extended)

A new flag defaults to the original behavior:

```python
include_ohlc_features: bool = False
```

Only when `True` are the three intraday features added:

```python
dev["intraday_range"] = (dev["high"] - dev["low"]) / dev["open"]
dev["intraday_return"] = dev["close"] / dev["open"] - 1.0
dev["opening_gap"] = dev["open"] / dev["close"].shift(1) - 1.0
```

The function first restricts observations to **before the fold's cutoff**. Only then does it calculate returns, features, and target labels. It still removes the final training row when its next-session label would fall in validation. The first 20 price rows still lack sufficient history for 20-day historical volatility and are not model-ready.

### `src/risklens/models/baselines.py` (existing file extended)

`compare_models` now allows only the two predefined feature lists. The default stays the original three, so old workflows still work unchanged.

For either configuration, the code selects the exact allowed features, verifies dates and labels, fits the scaler and model on `X_train, y_train` **only**, and calls `predict_proba(X_valid)` to generate validation probabilities. It will reject unexpected columns (e.g., a leaked `tomorrow_close`). It **never** trains on the labels of the same year's validation set.

### `src/risklens/models/ohlc_comparison.py` (new)

1. Run the original Step 3.3 walk-forward procedure unchanged.
2. For each year `Y = 2020, ..., 2024`, call `build_development_features(..., validation_start=Y, holdout_start=Y+1, include_ohlc_features=True)`.
3. Fit the same fixed baseline and Logistic Regression using the six predictors on training dates **before year Y**.
4. Verify that original and expanded runs have **identical prediction dates, target labels and baseline probabilities**. Stop with an error if not.
5. Store both logistic probabilities on the same validation rows. Calculate annual ROC-AUC and other scores, plus pooled Brier and log loss across the five validation years.

### `src/risklens/models/ohlc_comparison_cli.py` (new)

`argparse` reads your input filename and optional output folder, calls the comparison function, and prints scores. A successful terminal command returns exit status `0`; a handled failure returns `1`.

### `tests/test_ohlc_comparison.py` (new)

Tests verify the exact formulas; that existing three-feature outputs remain unchanged; matching dates, labels, and baselines; that future price edits cannot alter earlier features or 2025+ holdout edits alter evaluation; training-only scaling; validation labels not entering model fitting; rejection of unexpected feature columns; and refusal to overwrite outputs without explicit permission.

## 4. Real-data results (2016–2024 development only)

Using the NIFTY 50 source from this project, each fold trains only on prior years and scores the shown year:

| Validation year | 3-feature ROC-AUC | 6-feature ROC-AUC |
| --- | ---: | ---: |
| 2020 | 0.621 | 0.616 |
| 2021 | 0.689 | 0.705 |
| 2022 | 0.597 | 0.584 |
| 2023 | 0.538 | 0.525 |
| 2024 | 0.446 | 0.441 |

Pooled probability scores (five validation years combined):

| Model | Brier score (lower better) | Log loss (lower better) |
| --- | ---: | ---: |
| Constant training-frequency baseline | **0.10433** | **0.36644** |
| Logistic Regression, 3 features | 0.11268 | 0.39735 |
| Logistic Regression, 6 features | 0.11278 | 0.39897 |

These are **descriptive validation findings**. The extra OHLC features did **not** improve pooled probability performance for this fixed Logistic Regression experiment. The changes are small, and annual event samples are limited. Do not conclude the features will never be useful or that the model has stable future predictive skill.

### Score vocabulary

- **ROC-AUC:** how often an event receives a higher risk score than a non-event in a random event/non-event pair. About `0.5` means little ranking ability; `1` is perfect.
- **Brier score:** average of `(predicted probability - actual binary outcome)²`; smaller is better.
- **Log loss:** penalizes forecasts that are confidently wrong; smaller is better.
- **Baseline:** the simplest fixed comparison model, predicting the **training** event fraction for each validation observation.
- **Walk-forward validation:** train on past dates, evaluate on later dates, repeat with later years; **never** use a validation year's labels for its own fitting.
- **Holdout:** later data deliberately kept away from development experiments. Here, 2025–September 2026 is reserved.

## 5. Run on your existing Mac / VS Code folder

1. Unzip the **changes-only ZIP** into the root of your existing RiskLens folder. Do not replace your `.git` or `.venv` folders.
2. Activate `.venv` and run tests:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

3. Run the new experiment:

```bash
python -m risklens.models.ohlc_comparison_cli \
  data/processed/multiyear/nifty50_ohlc.csv \
  --output-dir data/processed/ohlc_comparison
```

Expected on the previously provided source CSV: **5 years**, **1,238 validation observations**, pooled Brier scores about **0.10433** (baseline), **0.11268** (3-feature logistic), **0.11278** (6-feature logistic). If running again, choose a new output directory or explicitly pass `--overwrite` after checking existing outputs.

Outputs (local and Git-ignored):
- `data/processed/ohlc_comparison/ohlc_feature_comparison_predictions.csv`
- `data/processed/ohlc_comparison/ohlc_feature_comparison_report.json`

After verifying:

```bash
git status
git add src/risklens/features/dataset.py src/risklens/models/baselines.py \
  src/risklens/models/ohlc_comparison.py src/risklens/models/ohlc_comparison_cli.py \
  tests/test_ohlc_comparison.py docs/phase3_step4_learning_guide.md
git commit -m "Compare OHLC features under fixed walk-forward validation"
git push origin main
```

## 6. How to explain it in an interview

> I tested whether Open, High and Low provide incremental information beyond my closing-price features. I kept the model, training periods, labels and evaluation protocol fixed, adding only intraday range, intraday return and opening gap. I checked that original and expanded models scored exactly the same dates and events. In development-period walk-forward validation, the expanded model did not improve aggregate probability scores, while both models trailed a constant training-frequency baseline. This shows why I assess added complexity instead of assuming more indicators help. I did not inspect the final holdout.

**Decision now:** Record this outcome. Don't tune the feature set repeatedly against the same validation years merely to obtain a better number. Next research decisions should be explicitly justified and assessed using a controlled protocol; 2025–2026 remains unopened.
