# Phase 2, Step 2.3 — Development-only financial EDA

**RiskLens: Explainable Financial Risk Modeling, Validation & AI Research Platform**

This task operates on `data/processed/multiyear/nifty50_ohlc.csv` and **does not analyze the final holdout's returns or labels**. It teaches core descriptive statistics while creating an auditable development-only CSV and JSON summary.

## 0. Exact command (Mac / VS Code)

From your **existing** RiskLens project root, activate `.venv` (Python 3.12):

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
python -m risklens.analysis.cli data/processed/multiyear/nifty50_ohlc.csv --output-dir data/processed/eda
```

If you need to rerun without changing folder names, use `--overwrite` deliberately. Outputs in `data/processed/` are excluded from Git. Nothing in this task needs an API key or internet connection.

## 1. Why analyze returns rather than index levels?

An **index level** (e.g., 8,000 or 20,000) is the numeric level of the price index. Its magnitude changes over time, so a 100-point movement means something different when the index is 8,000 versus 20,000. A **return** is a proportional change. It is more comparable across dates:

- Close yesterday: `100`; close today: `102`.
- Simple close-to-close return: `102/100 - 1 = 0.02 = +2%`.
- Close tomorrow: `99`; its return is `99/102 - 1 ≈ -2.94%`.

**Code:** `dev["daily_return"] = dev["close"].pct_change(fill_method=None)`.

This is the return *realized at the end of day t*, calculated from day t and the previous trading day's closing level. It is available after day t closes. It is **not** next day's return, which remains unknown at prediction time.

Why a missing value on the first date? We did not import the previous trading day's close, so the first percentage change cannot be computed without inventing history. Leaving it missing is correct.

## 2. Simple vs log return

- **Simple return:** `P_t/P_(t-1) - 1`.
- **Log return:** `ln(P_t/P_(t-1)) = ln(1 + simple_return)`.

Simple returns are easy to interpret as percentage gains/losses. Log returns add over time, making them useful in some statistical models. They are not numerically identical for large moves. Our downside target remains based on **simple** next-session return, as stated in the project charter.

**Code:** `np.log1p(dev["daily_return"])`; `log1p(x)` computes `ln(1+x)` accurately for small x.

## 3. Mean and median

- The **mean** is the sum of observed daily returns divided by their count: an average of **historical observations**, not a future forecast.
- The **median** divides observed returns into two halves; it is less sensitive to extreme values.

A small positive historical mean does not establish a profitable trading strategy. Our NIFTY 50 index is a **price index**, and simple index-level returns do not include dividend reinvestment (unlike a total-return index).

## 4. Standard deviation and volatility

**Standard deviation** measures how spread out daily returns are around their average. We use sample standard deviation (`ddof=1`):

```text
s = sqrt( sum((r_i - mean(r))^2) / (n - 1) )
```

Our **conventional annualized volatility estimate** uses `s_daily * sqrt(252)`, with 252 representing an approximate number of trading sessions per year. This is a scale conversion under familiar assumptions; it does **not** prove prices are normal, returns independent, volatility constant, or that risk can be summarized by one number.

**Rolling volatility:** On day t, calculate the sample standard deviation of the last 20 **observed daily returns**, including day t, then multiply by `sqrt(252)`:

```python
dev["daily_return"].rolling(window=20, min_periods=20).std(ddof=1) * np.sqrt(252)
```

`rolling` moves a historical window across the rows. The first 20 valid returns are required; the earliest rows stay blank. The calculation is legal **after day t closes**, when predicting the outcome for day t+1. We do not use `shift(-1)` in any feature calculation.

This historical realized-volatility series is **not** a prediction or proof of predictive power; later we must evaluate whether it improves future-outcome predictions outside training.

## 5. Distribution, outliers, skewness and kurtosis

A **quantile** cuts the observed return distribution at a chosen fraction. The 1st-percentile return is a value below which about 1% of observed returns lie. It is descriptive, and **not automatically VaR**; VaR needs a clearly specified horizon, confidence level, estimation method, and validation.

- **Skewness:** measures asymmetry of a distribution; negative indicates a longer/heavier left tail in this sample.
- **Excess kurtosis:** a measure of tail/peak departure from normal-distribution kurtosis (normal excess kurtosis is 0). Extreme values and sampling variability can strongly influence it.
- **Outliers:** unusually large movements. These should be inspected in context, not automatically deleted. Removing genuine crashes would make a downside-risk model look unrealistically safe.

## 6. Our 1% downside target vs today's return

The target defined for prediction **after today's close** is:

```text
Y_t = 1 if close_(t+1)/close_t - 1 <= -0.01, else 0
```

The EDA CSV contains only `daily_return_t = close_t/close_(t-1)-1` (and other past-known measurements). It intentionally does **not** contain `Y_t` or tomorrow's return. The EDA summary counts return days with a realized `<= -1%` move, *dated at the close when it occurred*. The earlier batch-import event report groups by **prediction day t** instead. The **total** counts match in a continuous development sample, but calendar-year breakdowns can differ at year boundaries. This distinction is important in interviews and time-series coding.

## 7. Holdout leakage: the key design decision

The code executes the holdout filter **before** any return, log return, volatility, quantile, skewness, or event calculation:

```python
all_prices = load_ohlc_csv(source_csv)  # structural validation only
dev = all_prices.loc[all_prices["date"] < cutoff, ["date", "close"]]
# ONLY NOW are returns and statistics calculated.
```

The original CSV includes 2025–2026 dates. Those prices are parsed only to validate structure; future **returns, labels, and descriptive statistics** are excluded. Tests deliberately change all post-cutoff prices and verify that all non-provenance development results are unchanged.

Do not tune features using information from the final holdout. Later we will similarly separate the 2016–2022 training and 2023–2024 validation periods with leakage-safe target boundaries.

## 8. Files and provenance

Outputs (private working files; do not push):

- `data/processed/eda/nifty50_eda_development.csv` — development-only date, close, daily simple return, daily log return, backward-looking 20-session annualized volatility.
- `data/processed/eda/nifty50_eda_development.json` — row counts, time span, return summary, annual volatility, worst-return dates, missing-window count, source SHA-256 fingerprint, methods and caveats.

SHA-256 helps check whether analyses used the **exact same input file**; it is not proof the source is genuine NSE data. The raw files should remain unchanged and stored locally in `data/raw/nifty50/`.

## 9. Reading the functions in order

1. `load_ohlc_csv()` in `validation.py` validates columns, dates and OHLC consistency.
2. `_date_cutoff()` in `eda.py` rejects ambiguous dates.
3. `analyze_development()` filters all post-cutoff observations, computes historical returns, rolling volatility and summary statistics.
4. `_rounded()` replaces undefined statistics with JSON `null` instead of invalid `NaN` values.
5. `save_development_eda()` writes the private development CSV and report, requiring `--overwrite` for existing outputs.
6. `main()` in `analysis/cli.py` parses terminal options and prints success or a clear error.
7. `tests/test_eda.py` checks return arithmetic, window causality, strict cutoff, no leakage, invalid input, and overwrite behavior.

## 10. Interview-ready explanation, without exaggeration

> I combined and structurally validated roughly 11 years of NIFTY 50 price-index observations. Before modeling, I computed simple and log returns, sample volatility, rolling backward-looking volatility and distribution diagnostics using only a predeclared development period. The historical 2020 volatility was much higher than 2017, suggesting that market risk conditions vary over time. I kept the final holdout sealed, so these observations describe the past rather than establish a predictive result. Next, I will build time-aware baselines and determine whether these features improve performance out of sample.

Don't claim regulatory compliance, verified source authenticity, statistically significant predictability, market-beating returns, or actual model performance at this stage.
