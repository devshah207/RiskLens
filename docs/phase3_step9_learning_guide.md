# RiskLens — Phase 3, Step 3.9: External Market Data Integration

**Scope:** NIFTY 50 OHLC + historical India VIX + S&P 500 into a development-only, time-aligned dataset. **No model training, feature selection, supervised targets, tuning, or 2025+ evaluation.**

## 1. Why we add other market information

The NIFTY 50 is a **price index**, a calculated benchmark reflecting a selected basket of large Indian equities. Our earlier three, six and nine features mostly described the same NIFTY series. External data ask genuinely different questions:

- **India VIX:** an **implied-volatility index** calculated from NIFTY options prices. An option is a financial contract linked to the price of an underlying asset. A VIX level of 20 conventionally represents roughly **20% annualized expected volatility** over the index's horizon, not a 20% probability of a market fall. **Volatility** means variability of returns, not direction.
- **S&P 500:** a broad US stock-market price index. The **daily return** of the S&P 500 is `US_close[s] / US_close[s-1] - 1`. An example +0.02 means that index increased 2% between two consecutive observed US market sessions.
- **Our target remains unchanged:** whether the next observed NIFTY closing level falls by at least 1%. The external variables are candidates for future testing, not guaranteed predictive signals.

## 2. When can we know a market value?

RiskLens forecasts after Indian session `t` closes. At that moment, **the US market's closing value for the SAME calendar date is not yet known**. Joining the US series to NIFTY on equal date would therefore leak future information.

We deliberately apply a stricter rule to **both** external datasets: select the most recent source observation with `source_date < nifty_date`. This is conservative for India VIX: although some same-date VIX observations could be published by Indian market close, we have not verified their precise publication timestamps. We do **not** assume they were available.

Example for NIFTY's Monday 2024-12-30 prediction:

| Market data | Used or excluded | Reason |
| --- | --- | --- |
| NIFTY Monday, December 30 close | Used | Already known after NIFTY session close |
| India VIX Friday, December 27 close | Used | Earlier date, known by prediction time |
| S&P 500 Friday, December 27 close-to-close return | Used | Last earlier-date completed US session |
| S&P 500 Monday, December 30 close | **Excluded** | US market session has not yet finished at the Indian close |
| India VIX Monday, December 30 | **Excluded conservatively** | Same-date availability timestamp not independently verified |

Note: we use an *exchange-local session date* convention. This rule is deliberately stricter than a fully timestamp-aware data feed; we must verify official publication timestamps before considering a less conservative rule.

## 3. What files and data you already have

- `data/processed/multiyear/nifty50_ohlc.csv`: cleaned NIFTY series; 2,663 observations through 2026-09-30. This phase works only with **2,229 pre-2025 price rows**.
- `data/raw/india_vix/india_vix_2016.csv` ... `india_vix_2024.csv`: 9 source CSVs, 2,228 rows total.
- `data/raw/sp500/sp500_daily.csv`: a single CSV with Date, Open, High, Low, Close, Adj Close and Volume; 2,701 rows through 2026-09-30, including 2,264 source observations in 2016–2024. The source URL is unconfirmed; do not call this independently authenticated S&P 500 history.

**India VIX quality finding:** source dates **2021-02-12** and **2021-03-30** have `Open=High=Low=0` and a positive `Close`. We do not use their closing levels as valid daily observations. We **quarantine** those two dates, record them in the report, and leave the raw files untouched. Other malformed OHLC rows cause a clear error instead of being silently repaired.

**Missing-market-date example:** the unusual Saturday NIFTY session on 2024-03-02 has no matching VIX record for that same date. The conservative prior-date rule may use the last known earlier valid VIX date (2024-03-01) if it meets freshness requirements. We do not create fake Saturday VIX records.

## 4. How the code works, function by function

Open `src/risklens/data/external_integration.py`.

### `_read_as_text(path)`

Loads CSV fields as strings with `pd.read_csv(dtype="string")`. This avoids accidentally coercing unknown text identifiers into numbers. It rejects empty files and duplicate normalized column names.

### `_load_vix(vix_dir)`

1. Find `*.csv` files directly inside the VIX source directory.
2. Standardize headers by stripping whitespace and ignoring capitalization. Actual source column names such as `Date ` and `Close ` then become `date`, `close`.
3. Parse the source's `DD-MON-YYYY` dates, convert Open/High/Low/Close to numerical values, reject duplicate dates and invalid closing levels.
4. **Quarantine** only the identifiable pattern `Open=High=Low=0` with positive Close. Record every such date in the JSON audit.
5. Validate remaining VIX OHLC relationships with our earlier `validate_ohlc()` function.
6. Retain only source observations strictly before 2025-01-01.

A SHA-256 checksum identifies the exact contents of each source file; it does NOT prove authenticity or redistribution rights.

### `_load_sp500(sp500_csv)`

1. Read and validate the source OHLC; ignore but report extra fields such as adjusted close and volume.
2. Filter out all dates on or after 2025-01-01 **before** calculating returns.
3. Compute `sp500_previous_return_1d` with `.pct_change(fill_method=None)` using consecutive historical US closing values.
4. Keep the date (`sp500_source_date`), closing index level and daily return; preserve a source checksum in the audit.

The first US daily return is missing, because its preceding close is not in the CSV. Do not replace that missing return with zero.

### `integrate_external_development(nifty_csv, vix_dir, sp500_csv, max_age_days=7)`

Load and validate NIFTY, then filter pre-2025 before doing any integration. It joins VIX and S&P 500 using pandas `merge_asof()`:

```python
pd.merge_asof(
    nifty_rows,
    external_rows,
    left_on="date",
    right_on="external_source_date",
    direction="backward",
    allow_exact_matches=False,
    tolerance=pd.Timedelta(days=7),
)
```

- `direction="backward"`: choose an earlier record, not a future record.
- `allow_exact_matches=False`: exclude the same calendar date; **strictly earlier only**.
- `tolerance=7 calendar days`: a match older than one week becomes missing, rather than being used indefinitely.
- If a matching source observation is absent, pandas returns `NaN`/`NaT` for that field. No prices or returns are fabricated.

The code stores `vix_source_date` and `sp500_source_date` with each result for independent time checks, and age measured in calendar days. A Boolean flag `external_features_ready` says whether both the VIX level and previous completed US return are available.

**Important:** This is an external market data table, not the supervised modeling table. It intentionally preserves all 2,229 development NIFTY observations, including rows with missing external values. A later feature-builder must make train/validation boundary decisions, create labels correctly, and fit all transformations on training folds only.

### `save_external_development(...)` and `external_cli.py`

Write the merged CSV and audit JSON only after all checks succeed, initially staging in a temporary directory. Existing outputs require an explicit `--overwrite`. The terminal interface handles arguments and prints the real row counts.

## 5. Why do the real-data counts differ?

| Check | Real-data result | Explanation |
| --- | ---: | --- |
| NIFTY sessions before 2025 | 2,229 | Every pre-holdout NIFTY price row is preserved |
| India VIX matches | 2,228 | Earliest NIFTY date has no earlier India VIX date in the files |
| Previous completed S&P return matches | 2,226 | The US CSV starts 2016-01-04. A first-session return requires a prior US close, so the earliest NIFTY sessions lack a usable return |
| Rows with **both** external features | 2,226 | These are candidates for future supervised feature engineering |
| VIX zero-OHL rows quarantined | 2 | 2021-02-12 and 2021-03-30 |

**These are coverage checks, not model results.** We do not yet know whether external data improve downside prediction.

## 6. How to place your downloaded files locally

From your existing RiskLens project root (the directory with `pyproject.toml`):

```bash
mkdir -p data/raw/india_vix data/raw/sp500
unzip -j ~/Downloads/india_vix.zip 'india_vix/*.csv' -d data/raw/india_vix
unzip -j ~/Downloads/Archive.zip 'SP500*.csv' -d data/raw/sp500
mv data/raw/sp500/SP500*.csv data/raw/sp500/sp500_daily.csv
```

The `-j` flag ignores internal ZIP folders; the quoted ZIP patterns select only data CSVs, avoiding Mac metadata and unrelated NIFTY files. If you have already copied or renamed the S&P file, don't run the `mv` again.

Install/verify and run:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
python -m risklens.data.external_cli \
  data/processed/multiyear/nifty50_ohlc.csv \
  data/raw/india_vix \
  data/raw/sp500/sp500_daily.csv \
  --output-dir data/processed/external
```

The generated outputs are:

- `data/processed/external/nifty50_external_development.csv`
- `data/processed/external/external_integration_report.json`

Your `.gitignore` excludes downloaded and processed market data. Do **not** upload those raw CSVs to a public repo without verifying source rights.

## 7. What we deliberately did NOT do

- No trading model was fitted, compared, tuned or calibrated.
- No target labels were calculated on this merged external dataset.
- No 2025+ price returns, external features, event counts or predictive scores were calculated.
- No same-date S&P close was used for an Indian post-close prediction.
- No invalid India VIX source record was silently repaired.
- No external source was authenticated. The S&P provider remains to be established.
- We did not add India VIX as though it were the probability that NIFTY falls. It measures implied volatility, not a directional event probability.

**Step 3.10 will be a separate controlled experiment**: build leakage-safe supervised features using the joined external information, compare against the frozen existing model, and preserve the final holdout.

## 8. Interview-ready explanation

> I integrated NIFTY price-index observations with India VIX and prior completed S&P 500 session returns. Rather than joining records simply by equal calendar date, I used a strictly earlier-date as-of join to avoid inadvertently using US market closes that happened after the Indian prediction timestamp. I quarantined two questionable VIX observations, enforced a seven-day maximum observation age, preserved missingness flags and source dates, and kept the final holdout out of derived data. This is an auditable external-data preparation step, not evidence that the added variables improve model performance.
