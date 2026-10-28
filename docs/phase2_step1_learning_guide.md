# Phase 2 — Step 2.1: NIFTY 50 CSV ingestion and provenance

## What we are building

The original `validation.py` answers: **Is this table structurally valid OHLC data?**

The new `ingestion.py` answers: **Can this CSV be consistently interpreted as the NIFTY 50 series, validated, and saved with a traceable source record?**

We are not yet predicting risk, computing returns, downloading data automatically, or proving financial-source authenticity.

## 1. Finance foundations

- **NIFTY 50** is an index level, not a stock price and not a bank loan.
- **Open / High / Low / Close (OHLC)** describe one index trading session.
- **Price index vs total returns index:** price index reflects index price-level movement; a total-return index also incorporates reinvested dividends. RiskLens initially studies the NIFTY 50 price index.
- **Trading session:** a market-open date. Missing calendar weekends do not imply missing data.
- **Data origin:** a downloaded CSV can be internally consistent without being an authenticated NSE export.
- **Provenance:** source page declared by the user + original file SHA-256 fingerprint + date coverage + validation output.

**Source download:** https://www.niftyindices.com/reports/historical-data

Use Historical Index Data, select **NIFTY 50 (price index)**, choose a date range, and download CSV. Another official source is https://www.nseindia.com/reports-indices-historical-index-data. The exact website interface and export format may change; inspect the header before importing.

We have not bundled NSE prices, as this project avoids distributing unverified or potentially licensed market data.

## 2. Programming concepts used

| Code | Meaning | Why it exists |
| --- | --- | --- |
| `Path(input_csv)` | Represents a path on your computer | Keeps file-handling consistent |
| `pd.read_csv(..., dtype="string")` | Reads columns as text | Avoids silent conversions during initial import |
| `.str.strip().str.casefold()` | Removes edge spaces and compares without case differences | Matches `Nifty 50`, `NIFTY 50` |
| `frame.loc[keep]` | Selects rows where a Boolean condition is True | Excludes rows for other indexes |
| `validate_ohlc(selected)` | Reuses our existing validator | One definition of valid OHLC everywhere |
| `sha256(source_path.read_bytes())` | Computes a digest of the original file | Helps identify exactly which file was used |
| `.to_csv(..., index=False)` | Saves a data table | Creates a standardized file for later features |
| `json.dumps(report, indent=2)` | Converts a dictionary into readable JSON | Makes the import decisions inspectable |
| `try/except` in `import_cli.py` | Handles a known import problem | Prints FAIL and returns a nonzero process code |

### Why preserve original data?

If you discover an issue in the transformation later, you can rerun the importer on the *same original input*. Never silently edit `data/raw/` and pretend it was the source you downloaded.

### What is SHA-256?

It is a cryptographic **file fingerprint**. If any byte of a file changes, its SHA-256 digest is almost certain to change. A matching fingerprint shows that the same bytes were used, **not** that the original source is authentic.

## 3. Files added

- `src/risklens/data/ingestion.py`: source parsing, index identification, validation, outputs and provenance.
- `src/risklens/data/import_cli.py`: terminal command for importing a CSV.
- `tests/test_ingestion.py`: synthetic positive/negative examples.
- `data/fixtures/sample_nse_export.csv`: deliberately synthetic multi-index data to test filtering.

## 4. Running the code on a synthetic example

From the project root, with Python 3.11+ and `.venv` activated:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m risklens.data.import_cli data/fixtures/sample_nse_export.csv --output-dir data/processed
```

The last command generates `data/processed/nifty50_ohlc.csv` and `data/processed/nifty50_import_report.json`. Both outputs are ignored by Git, so they won't be committed by default. Synthetic data is for testing only.

To repeat that import into the same directory, **after confirming it is safe to replace the outputs**, add `--overwrite`.

## 5. Running on the real data after you download it

1. Download NIFTY 50 historical index CSV from the official page and note the actual page URL.
2. Put the original, unchanged CSV in `data/raw/`. Example local filename: `nifty50_history.csv`.
3. Open the file in VS Code and examine the header. We recognize `Date` or `Index Date`, `Close` or `Closing`, and `Open`, `High`, `Low`.
4. Run:

```bash
python -m risklens.data.import_cli data/raw/nifty50_history.csv \
  --source-page https://www.niftyindices.com/reports/historical-data \
  --output-dir data/processed
```

If the CSV has NO `Index Name` column, the importer refuses to guess the instrument. Only after you personally confirm that all rows describe NIFTY 50, repeat the command with `--assume-nifty50`.

5. Inspect the JSON report: coverage dates, row counts, excluded rows, source hash, warnings.
6. Inspect the processed CSV. Confirm there are no unexpected gaps, price jumps or duplicate dates. Such investigations are **not fully automated by this step**.
7. Never commit source data until you have checked licensing/redistribution permissions.

**Important:** The output can be structurally valid while still containing wrong, manipulated, or stale financial values. We will inspect data distributions in the next step before training.

## 6. Important implementation decisions

- Manual download first: controlled provenance, no fragile scraping.
- No automatic filling of calendar gaps: non-trading days are legitimate, and unverified market data must not be invented.
- No changing or overwriting raw files.
- No output unless all source-row validation succeeds.
- Existing processed output cannot be overwritten accidentally.
- Filter to `NIFTY 50` when the CSV contains an `Index Name` column. If the column is absent, require the `--assume-nifty50` flag.
- Report source identity as **user-declared**, never as authenticated by code.
- If a multi-index CSV is filtered, validation error row positions refer to the filtered selection, **not necessarily original CSV line numbers**. Use the source to investigate and avoid blindly trusting line numbers after filtering.

## 7. Interview questions this step prepares you for

- What is the difference between raw and processed data?
- Why do you record the SHA-256 fingerprint of the raw file?
- How did you ensure only the intended financial instrument was imported?
- What does a successful OHLC validator prove, and what does it NOT prove?
- Why is reusing one validator better than duplicating validation code?
- Why did you avoid forward-filling missing trading dates?
- Why do you need an import report before beginning ML training?

**Next step (2.2):** Inspect a real CSV, generate basic data-quality and time-coverage reports, and understand changes in index levels. Still no training.
