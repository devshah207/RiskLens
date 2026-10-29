# Phase 3, Step 3.1 — Feature/Label Alignment and Chronological Splits

**Goal:** Construct a *development-only*, leakage-aware tabular dataset for a first supervised financial classification model. **No model is trained in this step.**

## 1. What happens at the prediction timestamp?

Our intended prediction occurs **after session `t` closes** (assuming the close has been published and finalized). At that moment, we know prices at `t`, `t-1`, `t-5`, etc., but **not the next session's close**.

| Item | Definition | Known at end of t? | Usage |
|---|---|---|---|
| Daily simple return | `close[t]/close[t-1] - 1` | Yes | `return_1d` feature |
| Five-session simple return | `close[t]/close[t-5] - 1` | Yes | `return_5d` feature |
| 20-return sample volatility | `std(return[t-19], …, return[t]) * sqrt(252)` | Yes | `volatility_20d` feature |
| Next-session simple return | `close[t+1]/close[t] - 1` | **No** | Construct historical target **after the event occurs** |
| Binary label | `1` if next return `<= -0.01`, else `0` | **No** | Training/validation outcome, **never a feature** |

**Why these three features?** They are interpretable, use only one index's history, and represent daily movement, medium-short-term movement, and recent variability. Their predictive value is **unknown**; later baseline comparisons will test that. We do not add a long list of technical indicators for appearances.

### Tiny numerical example

Assume closes over six sessions: `100, 102, 101, 104, 103, 105`.

- On session 6, `return_1d = 105/103 - 1 ≈ +1.9417%`.
- On session 6, `return_5d = 105/100 - 1 = +5%`.
- We cannot calculate a 20-return volatility until we have **20 daily returns (21 closes)**.
- If session 7 later closes at `103`, label on session 6 is `1`, because `103/105 - 1 ≈ -1.9048%`.

The five-session return is **not** automatically the arithmetic sum of five daily percentage returns because returns compound.

## 2. Why does the code use a future shift for the target but NOT the features?

Look at `src/risklens/features/dataset.py`:

```python
# Only historical information; available at t close.
dev["return_1d"] = dev["close"].pct_change(fill_method=None)
dev["return_5d"] = dev["close"].pct_change(periods=5, fill_method=None)
dev["volatility_20d"] = (
    dev["return_1d"].rolling(window=20, min_periods=20).std(ddof=1)
    * np.sqrt(sessions_per_year)
)

# Future used ONLY for constructing past labels.
dev["next_session_date"] = dev["date"].shift(-1)
next_close = dev["close"].shift(-1)
```

- `pct_change()` compares each row to an **earlier** row; it never needs a later close.
- `rolling(window=20)` looks backward over the *current and previous 19* daily returns; it does not center a window around t.
- `shift(-1)` places **tomorrow's close on today's row**. That is **only** acceptable for constructing the historical supervised outcome. Putting `next_close`, `next_session_date`, or `target` in a model's feature matrix would be leakage.
- A missing target at the end of the development period stays missing and is not treated as `0`.

The exported tables deliberately contain **only** `date`, the three features, and `downside_next_session`. The temporary helper `next_session_date` is never exported as a model feature.

## 3. Why do we discard the earliest 20 rows?

To calculate a 20-return volatility value, we need 20 observed daily returns. The first price row has no preceding close, so its 1-day return is missing; the **first complete volatility value occurs on the 21st price row**. All earlier rows are omitted from the *feature dataset* rather than replaced with fabricated numbers. The original price history remains preserved in your input CSV.

`ready_features = dev[list(FEATURE_COLUMNS)].notna().all(axis=1)` marks rows on which all three features are available.

## 4. Chronological train/validation and purging labels at boundaries

Provisional partitions:

- **Training:** prediction dates before `2023-01-01` AND their target-observation dates before `2023-01-01`.
- **Validation:** prediction dates starting `2023-01-01` AND their target-observation dates before `2025-01-01`.
- **Final holdout:** `2025-01-01` onward; **no features/labels analyzed or exported now**.

Suppose the last 2022 trading session is `2022-12-30`. Its next close is observed on a 2023 date. Because the label would require a validation-period outcome, this last 2022 **prediction row is excluded from training**. On the other hand, the first 2023 prediction row may compute `return_5d` using 2022 prices, since those are past observations already available in 2023.

Actual code:

```python
training = (dev["date"] < vstart) & (dev["next_session_date"] < vstart)
validation = (dev["date"] >= vstart) & (dev["next_session_date"] < hstart)
```

This is a **one-session label-boundary purge**, justified by our one-session prediction horizon. It is not a general multi-horizon purged cross-validation solution.

## 5. Why do we cut off holdout BEFORE calculating anything?

```python
prices = load_ohlc_csv(source_csv)      # Validates the full CSV structure.
dev = prices.loc[prices["date"] < hstart, ["date", "close"]].copy()
# Only now do we calculate features and labels.
```

The 2025–2026 observations are present in your original source, so we check their **structural validity**, but we do not calculate their returns, labels, or distribution summaries. The tests deliberately change holdout prices while keeping earlier prices identical; training and validation features, outcomes and summaries remain unchanged.

**Future modeling rule:** All fitted preprocessing (standardizers, imputation, feature selection) must be fit on training data (or within each training fold), then applied to validation. We are **not** fitting any transformer at this step.

## 6. Exact meaning of the Python expressions

| Code | Explanation |
|---|---|
| `pd.DataFrame` | A table with named columns |
| `pd.Series` | A single column of values |
| `Path(source_csv)` | A platform-independent filesystem path |
| `.loc[mask, columns]` | Filter rows by a Boolean condition and select columns |
| `.shift(-1)` | Align the next row's value to the current row (used only for target construction here) |
| `.pct_change(periods=5)` | Percentage change from five observations earlier |
| `.rolling(...).std(ddof=1)` | Sample standard deviation in a trailing window |
| `.notna().all(axis=1)` | Keep only rows where *all selected features* are present |
| `.astype("int64")` | Store target labels as integer 0 or 1 |
| `.reset_index(drop=True)` | Restore sequential row indices after filtering |
| `TemporaryDirectory()` | Stage output files until calculations finish |
| `json.dumps(..., indent=2)` | Serialize the methodology/count audit in human-readable JSON |
| `pytest.approx()` | Check floating-point computations with numerical tolerance |

### Function-by-function map

1. `_strict_date()` rejects ambiguous or impossible split dates.
2. `build_development_features()` loads and validates prices, removes the holdout, calculates backward-looking features, constructs historical targets, purges split-boundary labels, and returns training/validation tables and a report.
3. `_split_summary()` records per-partition row counts, time coverage, and class frequencies **for development data only**.
4. `save_development_features()` writes two CSVs plus one JSON report. Refuses to overwrite without explicit consent.
5. `features/cli.py:main()` parses terminal arguments, invokes the save function, prints counts, and returns exit code 0 or 1.
6. `tests/test_feature_dataset.py` covers formulas, target timing, chronology, holdout mutations, boundary purging, output format, and failure conditions.

## 7. Running locally (Mac and existing VS Code project)

Copy *only the files in the changes ZIP* over your existing RiskLens project; do not replace `.git` or `.venv`.

```bash
source .venv/bin/activate
python -m pytest -q
python -m risklens.features.cli \
    data/processed/multiyear/nifty50_ohlc.csv \
    --output-dir data/processed/features
```

Output files under `data/processed/features/`:

- `train_features.csv` — 2016–2022 labeled feature rows, after warm-up and boundary purge.
- `validation_features.csv` — 2023–2024 labeled feature rows, with only historical features.
- `feature_dataset_report.json` — feature definitions, source SHA-256 checksum, exact split boundaries, sample/event counts, and warnings.

Outputs are ignored by Git. To deliberately rerun over existing output files, pass `--overwrite`.

## 8. Interpreting your real-data counts

With your uploaded `nifty50_ohlc.csv` and default splits, the prepared data contains:

| Split | Rows | Historical events | Event frequency |
|---|---:|---:|---:|
| Training | 1,713 | 187 | 10.92% |
| Validation | 494 | 38 | 7.69% |

Compared with the **1,734** raw eligible training-period prediction dates before 2023, we lose 20 initial rows due to unavailable 20-return history and 1 boundary row whose label would need a 2023 close, producing **1,713** fully labeled training rows.

The earlier full development event report counted **230/2,228** historical events across 2016–2024. This step produces fewer *training* events after the feature warm-up and the internal training/validation split; this is deliberate, not silent missing-data deletion.

These are historical *class frequencies*, not model accuracy, calibration, significance, or evidence of predictability.

## 9. Questions to be able to answer in an interview

- Why does `return_5d` use `close[t-5]` instead of summing five daily percentage returns?
- Why may the first validation feature use some late-2022 prices without leaking information?
- Why cannot the last 2022 row be a training example with a next-session target in 2023?
- Why are early rolling-volatility values missing, and why is replacing them with zero inappropriate?
- What would go wrong if we included `next_close` or `next_session_date` among model features?
- Does validation event frequency 7.69% prove a classifier can predict losses? **No.**

## 10. What we deliberately did not do

- Fit Logistic Regression, XGBoost, or any other model.
- Compute any 2025–2026 target frequencies, predictive scores, or holdout return statistics.
- Fit scalers/normalizers or choose features using validation performance.
- Use live market data, claim a trading strategy, validate source authenticity, or perform VaR backtesting.

Next, *after understanding and verifying this step*, we can establish a constant-probability baseline and then introduce Logistic Regression.
