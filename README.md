# RiskLens

**Explainable Financial Risk Modeling, Validation & AI Research Platform**

RiskLens is an **educational research prototype** motivated by financial machine learning, data quality, and independent model-validation workflows. It is **not investment advice, a trading strategy, or a regulatory-compliant risk engine**.

## Current status

**Phase 1 complete:** project specification, synthetic OHLC fixtures, data validator, command-line tool, and automated tests.

**Phase 2, Step 2.1 implemented:** an importer for manually downloaded NIFTY 50 historical CSV files, with index-selection checks, reused OHLC validation, standardized output, SHA-256 fingerprint, and a source-provenance report. The importer has been tested against **synthetic data only** so far; real NSE data has not been imported or verified in this repository.

**Not implemented yet:** financial feature engineering, machine-learning models, VaR, RAG, dashboards, or measured predictive results.

## Problem definition

After NIFTY 50 price-index trading session `t` closes, estimate the probability that the following trading session closes **at least 1% below** that close.

For index closing levels `P_t` and `P_(t+1)`:

```text
r_(t+1) = P_(t+1) / P_t - 1
label_t = 1 if r_(t+1) <= -0.01 else 0
```

Features will use only information available **at or before** the prediction time (after session `t` closes). The next-session closing level will be used **only for constructing historical labels**, never as a feature. The `-1%` threshold is provisional pending evaluation of event frequency using development data.

## Install and run tests

Requires Python 3.11+; Python 3.12 in a local `.venv` works well.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

### Validate existing synthetic OHLC data

```bash
python -m risklens.data.cli data/fixtures/sample_ohlc.csv
```

### Import a synthetic NSE-style CSV (for code testing only)

```bash
python -m risklens.data.import_cli data/fixtures/sample_nse_export.csv \
    --output-dir data/processed
```

This writes a standardized CSV and a JSON import report under `data/processed/`. These files are excluded from Git.

### Import actual NIFTY 50 historical data later

Download a historical NIFTY 50 **price index** CSV manually from [NSE Indices](https://www.niftyindices.com/reports/historical-data). Keep the original file unchanged at `data/raw/nifty50_history.csv` and record the actual source page.

```bash
python -m risklens.data.import_cli data/raw/nifty50_history.csv \
    --source-page https://www.niftyindices.com/reports/historical-data \
    --output-dir data/processed
```

If the export does not identify the index in an `Index Name` column, **first verify it manually** and then pass `--assume-nifty50`. Import success does **not** prove financial-source authenticity or correctness of all recorded prices. For detailed instructions and risks, see [`docs/phase2_step1_learning_guide.md`](docs/phase2_step1_learning_guide.md).

## Repository map

- `docs/project_charter.md` — goals, scope and boundaries.
- `docs/finance_glossary.md` — essential financial vocabulary.
- `docs/data_contract.md` — expected columns, consistency and leakage rules.
- `docs/decisions.md` — design choices and alternatives.
- `docs/task2_learning_guide.md` — validator walkthrough.
- `docs/phase2_step1_learning_guide.md` — importing and source provenance.
- `src/risklens/data/validation.py` — reusable OHLC validator.
- `src/risklens/data/cli.py` — validate a CSV from the terminal.
- `src/risklens/data/ingestion.py` — import an identified NIFTY 50 series.
- `src/risklens/data/import_cli.py` — importer command-line entry.
- `tests/` — synthetic valid and invalid cases.
- `data/fixtures/` — synthetic examples (never real market prices).
- `data/raw/`, `data/processed/` — local market data and generated outputs, excluded from Git.

## Roadmap

1. Financial foundations and input validation **(done)**
2. Data ingestion and quality checks **(in progress)**
3. Exploratory data analysis and leakage-safe features
4. Baselines, model training and tuning
5. Independent model validation and backtesting
6. Explainability and monitoring
7. Grounded financial-report retrieval and RAG
8. Interface, testing and integration
9. Evidence-backed resume and demonstration
10. Technical interview defense

## Responsible use

- Don't use the system for investment or trading decisions.
- Don't claim model performance before running real experiments.
- Don't publish or redistribute market data without checking provider permissions.
- Don't use final holdout data for iterative tuning.
- Preserve all original data and document its declared source and transformations.
