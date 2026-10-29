# RiskLens — Phase 3, Step 3.8: Technical Indicators (Simple, Causal and Testable)

## 1. What we're doing—and not doing

We use **only the NIFTY 50 OHLC CSV you already imported**. OHLC means Open, High, Low, Close—the opening, highest, lowest, and closing index levels for a trading session. No external CSVs, downloads, India VIX, S&P 500, new model class or hyperparameter optimization in this step.

Our prediction is made **after today's close** and asks whether **the next observed session's closing level** will be **at least 1% below today's closing level**. All technical features are calculated using information at or before today's close. Tomorrow's close is used only to label *past* examples.

We preserve the 2025+ holdout: the feature builder validates the overall file structure but filters away 2025+ price rows before calculating **any** features or target outcomes.

## 2. The three new technical indicators

### RSI-14 (Relative Strength Index)

RSI is a number between **0 and 100** that compares recent *upward* closing-price movements with *downward* movements. "14" denotes **14 trading-session price changes**, not 14 calendar days. High RSI means recent price gains have been relatively strong; it is **not** a reliable guarantee of a fall tomorrow. The financial-market word **overbought** is a conventional description for high RSI, not a causal law.

Our implementation uses **Wilder-style smoothing**:

1. `delta[t] = close[t] - close[t-1]` — price movement in index points.
2. `gain[t] = max(delta[t], 0)`, `loss[t] = max(-delta[t], 0)`.
3. Seed average gain and average loss with the **simple mean of the first 14 changes** (so the first RSI appears at the 15th close).
4. After that, update causally: `avg_gain[t] = (13*avg_gain[t-1] + gain[t])/14` and similarly for average loss.
5. `RS = avg_gain / avg_loss`, `RSI = 100 - 100/(1+RS)`.

We explicitly define both-average-zero as RSI **50** (no directional movement), positive gain and zero loss as **100**, and zero gain with positive loss as **0**. Small implementation differences in seed conventions can lead to slightly different RSI values across charting libraries. Our convention is documented and tested.

**Code:** `_wilder_rsi_14()` in `src/risklens/features/dataset.py`. Its `for` loop moves forward through dates and does **not** look ahead. It does not fit statistical parameters using later data.

### EMA-50 distance (Exponential Moving Average)

A **moving average** smooths noisy price observations; an EMA gives more weight to more recent prices. The 50-session EMA is calculated as:

`EMA[t] = alpha*close[t] + (1-alpha)*EMA[t-1]`, where `alpha = 2/(50+1)`.

We use the explicit pandas convention `ewm(span=50, adjust=False, min_periods=50)`: the **first close initializes the recursion**, but the first **published** EMA is after 50 price observations. Some trading platforms use a different seed convention; that would produce different early values.

The feature is **not the raw EMA number**. It is normalized as:

`ema_50_distance[t] = close[t] / EMA50[t] - 1`.

If NIFTY closes at 25,500 and EMA50 is 25,000, the distance is `+0.02 = +2%`. This tells us NIFTY is 2% above that smoothed price reference, **not** that it must decline soon.

**Code:**

```python
ema = dev["close"].ewm(span=50, adjust=False, min_periods=50).mean()
dev["ema_50_distance"] = dev["close"] / ema - 1.0
```

### Volatility-10d (10-trading-session historical volatility)

**Volatility** measures the amount of variation in recent percentage returns. This indicator calculates the sample standard deviation of the **latest 10 daily close-to-close returns**, including today's realized return; it then multiplies by `sqrt(252)` to express an annualized-equivalent value.

`volatility_10d[t] = std(return[t-9], ..., return[t], ddof=1) * sqrt(252)`

Why both 10-day and 20-day volatility? A short window can react more rapidly to large recent movements, while a longer window smooths more of them. Whether the extra measure helps predictions is an **empirical question**. This feature is **not GARCH** and does not assert it can predict market direction.

**Code:**

```python
dev["volatility_10d"] = (
    dev["return_1d"].rolling(window=10, min_periods=10).std(ddof=1)
    * np.sqrt(sessions_per_year)
)
```

## 3. Important detail: a FAIR comparison

We already had six candidate features:

- `return_1d`: today's close vs prior close, expressed as a fractional return.
- `return_5d`: today's close vs five sessions earlier.
- `volatility_20d`: previous 20 daily returns' standard deviation, annualized.
- `intraday_range`: (high-low)/open.
- `intraday_return`: close/open-1.
- `opening_gap`: open/prior_close-1.

Now the nine-feature model adds `rsi_14`, `ema_50_distance`, `volatility_10d`. But EMA-50 requires 50 observed closes. This means the first **49 price rows** have incomplete feature history.

**To compare fairly, we exclude those same warm-up rows from BOTH models' training sets.** Otherwise one variant would have more historical training examples, mixing feature differences with sample-size differences.

The comparison code builds the full nine-feature table and then uses either the first six columns or all nine, on **identical training dates, validation dates, and target labels**. Both use the same fixed XGBoost settings from Step 3.5: 80 small depth-two trees and the same regularization. There is no parameter search or early stopping.

**Five expanding walk-forward folds:**

| Training period | Year being evaluated |
|---|---|
| 2016–2019 | 2020 |
| 2016–2020 | 2021 |
| 2016–2021 | 2022 |
| 2016–2022 | 2023 |
| 2016–2023 | 2024 |

The final prediction row of each training period is excluded when its next-session target would fall in the validation period. The same is done at the end of each validation year. A validation feature can use *earlier* training prices, because they were known by then.

This is development research: we have inspected 2020–2024 many times in earlier steps. **Do not present a tiny improvement here as independent confirmation.** The 2025–2026 final holdout remains sealed.

## 4. The actual experiment results from our NIFTY 50 CSV

All five folds comprise **1,238 validation predictions and 145 realized downside events**. Pooled results (lower is better):

| Model | Brier score | Log loss |
|---|---:|---:|
| Constant training-rate baseline | 0.10462 | 0.36840 |
| XGBoost (6 original features; fair aligned sample) | **0.10244** | **0.35650** |
| XGBoost (9 features) | 0.10291 | 0.35940 |

Brier score is the average squared error between predicted probability and binary outcome. **Log loss** is another probability error metric that penalizes especially confident wrong predictions. Their values here show that the new indicators did **not** improve overall predictive probability quality.

These six-feature results differ slightly from the earlier Step 3.5 six-feature results because the current comparison excludes 29 additional early training dates, keeping sample sizes equal between feature variants. This difference is intentional.

| Year | Six-feature ROC-AUC | Nine-feature ROC-AUC |
|---|---:|---:|
| 2020 | 0.458 | 0.530 |
| 2021 | 0.703 | 0.682 |
| 2022 | 0.599 | 0.584 |
| 2023 | 0.645 | 0.638 |
| 2024 | 0.434 | 0.450 |

**ROC-AUC** checks whether higher forecast probabilities tended to be assigned to actual downside events. Roughly 0.5 is random ranking. Results vary across years, and 2024 is still weak. Indicators may help one year and harm another.

**Decision:** Preserve the comparison as an honest negative result; do not tune thresholds or XGBoost to improve the same development period. We can move to external India VIX and previous-completed S&P 500 information in Step 3.9 after verifying sources and timestamp alignment.

## 5. Read the code in this order

1. `src/risklens/features/dataset.py` — constants `TECHNICAL_FEATURE_COLUMNS`, `FULL_FEATURE_COLUMNS`; `_wilder_rsi_14`; `include_technical_features` branch of `build_development_features()`. Existing default options and outputs are preserved.
2. `src/risklens/models/xgboost_walk_forward.py` — `_fit_predict_fixed_xgb()` now explicitly accepts only the existing 3, 6, or defined 9 features; **model settings are unchanged**.
3. `src/risklens/models/technical_comparison.py` — `compare_technical_features()` rebuilds each year's datasets, trains both variants on the same rows, evaluates later-year labels, records fold metrics and pooled Brier/log loss; `save_technical_comparison()` saves outputs.
4. `src/risklens/models/technical_comparison_cli.py` — `argparse` command-line interface prints result summary.
5. `tests/test_technical_comparison.py` — mathematical examples, warm-up tests, chronological boundaries, rejection of invalid mode, mutation of later prices and final holdout, identical-fold checks, and output safety.

Important Python syntax: `shift(1)` means the *previous* observed row; `rolling(window=10)` uses a *backward-looking* window; `ewm(..., adjust=False)` uses a recursive, earlier-dependent value; `np.diff` calculates consecutive changes; `np.maximum(delta,0)` separates upward changes; `for i in range(15,n)` loops only from earlier dates to later ones.

## 6. Run it in your EXISTING RiskLens folder

Extract the **changes-only** ZIP, without replacing `.git`, `.venv` or raw CSVs. This step updates the existing `dataset.py` and `xgboost_walk_forward.py`, so back up any uncommitted edits to those two files first.

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
python -m risklens.models.technical_comparison_cli \
  data/processed/multiyear/nifty50_ohlc.csv \
  --output-dir data/processed/technical_comparison
```

Outputs (not to commit):

- `data/processed/technical_comparison/technical_feature_comparison_predictions.csv`: year, prediction date, realized outcome, fixed baseline probability, six-feature and nine-feature XGBoost probabilities.
- `data/processed/technical_comparison/technical_feature_comparison_report.json`: fold boundaries, sample counts, fixed settings, scores, provenance and limitations.

Existing outputs require `--overwrite` to be replaced. Do not upload market-data outputs to public GitHub without checking data licenses.

### Interview-ready summary

> I tested RSI-14, distance from a 50-session EMA and 10-session realized volatility as additional signals for next-session NIFTY 50 downside risk. I used causal indicator definitions, separated training/validation chronologically and matched the exact observations between six- and nine-feature fixed-parameter XGBoost models. Across 2020–2024 development folds, the larger model did not improve pooled probability scores, so I retained the simpler reference and documented the negative result rather than tuning on already-observed validation periods.
