# RiskLens — Phase 2, Step 2.2: multi-year market-data integration

> **Learning + engineering guide.** This implementation was tested against **synthetic** examples. It has not yet been run on your Mac's actual NSE historical downloads. The code checks internal consistency, **not the authenticity or completeness** of the downloaded market data.

## 1. What we are trying to accomplish

NSE's web interface limits a single date-range export to one year. You downloaded several files for 2016–2026. We want to keep those original exports unchanged while creating **one** sorted NIFTY 50 daily OHLC dataset.

Data flow:

`data/raw/nifty50/*.csv` → read files → identify index → validate each file → check overlaps → sort and combine → flag suspicious gaps → save combined CSV + audit report + **development-only** downside-event frequency.

We do **not** train a model here, use external APIs, impute missing dates, or make investment recommendations.

## 2. Beginner-friendly finance definitions

- **Trading session:** A day when the exchange trades. Weekends and exchange holidays do not automatically mean missing data.
- **Index level:** A numerical value representing an index, such as the NIFTY 50 *price index*. It is not a stock price for one company.
- **OHLC:** Open, High, Low, Close index levels for a session. `low <= open <= high`, and `low <= close <= high` must hold.
- **Next-session return:** `close[t+1] / close[t] - 1`, where `t+1` is the *next observed trading session*, not necessarily the next calendar day.
- **Downside event:** Our pre-declared label is 1 if the next-session return is at most -1%; otherwise 0.
- **Event frequency:** The proportion of eligible historical prediction days for which that label is 1. **This is not predictive accuracy.**
- **Holdout:** A later, reserved time period for final evaluation. We intentionally do not compute event outcomes there at this step.

### Worked example

| Session | Close | Next observed close | Return | Downside event |
|---|---:|---:|---:|---:|
| Monday | 100 | 98 | (98/100)-1 = **-2%** | 1 |
| Tuesday | 98 | 100 | (100/98)-1 = **+2.04%** | 0 |
| Wednesday | 100 | unknown | unknown | no label yet |

For the two eligible days, the event frequency is `1 / 2 = 50%`. This tiny *synthetic* illustration is not a real market estimate.

## 3. The exact code you'll study

### `src/risklens/data/batch_ingestion.py`

Read these helpers in this order:

1. `_csv_paths(folder)`: searches that directory for `.csv` files, sorts their names, and rejects an empty folder. **Python:** `Path`, list comprehensions, file suffixes.
2. `_parse_cutoff(date_text)`: checks the fixed YYYY-MM-DD date for the reserved holdout. **Python:** `datetime.strptime` and exceptions.
3. `_read_one_file(path, assume_nifty50)`: reuses the existing CSV reader, header standardization, index selection, and `validate_ohlc`. **Python:** importing functions from another module, tuples, dictionaries.
4. `_combine_and_check(frames)`: stacks tables with `pd.concat`; compares dates and OHLC values when two annual files overlap. Identical overlaps are recorded and removed; **conflicting values stop the import**. **Python/pandas:** `groupby`, Boolean conditions, `drop_duplicates`.
5. `_date_gap_report(dates)`: looks for gaps longer than 7 **calendar** days. These are alerts for manual inspection, **not proof of missing sessions**. **Python/pandas:** `.diff()` and `.dt.days`.
6. `_development_event_frequency(data, holdout_start, threshold)`: estimates the historical frequency of a pre-specified event using **only the development portion** of the dataset. **Mathematics:** ratios, classification labels, class imbalance. **Python:** `.shift()` concepts, aligned arrays, Boolean masks, `.sum()` and `.mean()`.
7. `import_nifty50_batch(...)`: orchestrates the preceding functions and writes CSV/JSON outputs only if all checks succeed. **Engineering:** reproducibility, reports, staged file writes.

### `src/risklens/data/batch_cli.py`

This file provides a terminal entry point. `argparse` reads terminal options; `try`/`except` handles errors; exit code `0` means the import completed and `1` means something was rejected. This is the same pattern as our single-file CLI.

### `tests/test_batch_ingestion.py`

Tests use **invented** numbers so expected outcomes are known exactly. Examples include overlapping dates, conflicting prices, a date gap, missing index names, an invalid cutoff and an exit code check. They do not test the authenticity of downloaded NSE prices.

## 4. Key engineering decisions (and alternatives)

| Decision | Why | Alternative and its downside |
|---|---|---|
| Keep each source CSV unchanged | Audit trail and reproducibility | Hand-editing exports makes experiments hard to reproduce |
| Reuse existing single-file validation | Avoid duplicate data-quality logic | Write a second, inconsistent validation function |
| Require NIFTY 50 identity evidence | Don't mix indexes | Blindly accept every file in a folder |
| Remove only **identical** overlaps and count them | Annual exports can overlap at boundaries | Silently deduplicating could conceal conflicting data |
| Reject conflicting overlaps | We cannot know which price is correct | Arbitrarily pick one and corrupt the series |
| Flag long calendar gaps, but don't fill them | Weekends/holidays are legitimate | Forward-fill invented trading prices |
| Exclude final holdout outcomes from event-frequency report | Avoid using the reserved period to choose model design | Peek at holdout before training |
| Use a single **price-index** series | Less complex first risk experiment | Immediately mix stocks, total return indexes and corporate actions |

## 5. Why we exclude the last development day from label counts

Our default holdout begins **2025-01-01**. Suppose the last observed 2024 trading session is 2024-12-31. Its **next** observed session could be 2025-01-01 or later. Looking up that next close would reveal an outcome from the reserved period. Thus `_development_event_frequency()` takes two consecutive closes **both earlier than the holdout cutoff**. The final development day is excluded from the label count.

We may later create an explicit training/validation split within the development period. **This report is descriptive only.** Its event rate is not a validation metric, a probability forecast, or a trading strategy.

## 6. How to run it on your Mac

From the RiskLens root in VS Code, after updating the code:

```bash
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

Then inspect file names and first lines:

```bash
ls data/raw/nifty50
head -n 2 data/raw/nifty50/*.csv
```

Default import **when the CSVs contain `Index Name`**:

```bash
python -m risklens.data.batch_cli data/raw/nifty50 \
  --output-dir data/processed/multiyear \
  --source-page https://www.niftyindices.com/reports/historical-data
```

If one or more files lack an index-name column, first confirm **every such file** really contains the *NIFTY 50 price index*, then add `--assume-nifty50`. This switch is a **user attestation**, not automatic verification:

```bash
python -m risklens.data.batch_cli data/raw/nifty50 \
  --output-dir data/processed/multiyear \
  --source-page https://www.niftyindices.com/reports/historical-data \
  --assume-nifty50
```

If you run it a second time against the same output directory, it will refuse to overwrite existing reports. After reviewing the outputs, you may explicitly add `--overwrite`.

### Files created

- `data/processed/multiyear/nifty50_ohlc.csv` — combined, date-sorted OHLC data from **all** input files, including reserved later dates. **No target labels** are written here.
- `data/processed/multiyear/nifty50_batch_import_report.json` — source file names and SHA-256 hashes, input and final row counts, duplicate handling, per-year sample counts and potential calendar gaps.
- `data/processed/multiyear/nifty50_event_frequency_development.json` — downside event counts and rates for **development prediction dates only** (before 2025-01-01 by default).

These outputs remain outside Git because the repository excludes `data/raw/` and `data/processed/`.

## 7. When an error occurs

- **"No CSV files"** — files may be in a different folder or have the wrong extension. This command scans only the exact folder, not subfolders.
- **"No 'Index Name' column"** — confirm the index identity, then use `--assume-nifty50` only if appropriate.
- **"OHLC validation failed"** — inspect the named source file; do not silently correct prices.
- **"Conflicting OHLC values"** — two files disagree on the same date; investigate/download again.
- **"Batch output already exists"** — use a new output folder or `--overwrite` after review.
- **"gap longer than seven days"** — a warning, not a failure; compare the dates against exchange sessions/holidays.

## 8. What to share for the next learning step

Send the **terminal summary**, the import report's **per-year counts and warnings**, and the **development event-frequency report**. You do not need to upload all the raw CSVs, but if import errors occur share a small redacted/sample extract of the relevant file and the exact error. Do not claim data coverage or event frequencies until you have run the import on the real files.

### Completion criteria

- Every annual CSV is discovered and validated.
- No conflicting duplicate trading dates remain.
- The resulting dataset is date-sorted and unique by session.
- Any long gaps are reviewed, not automatically filled.
- The event frequency is reported only for development dates.
- You can explain both the calculation and why the final holdout is reserved.
