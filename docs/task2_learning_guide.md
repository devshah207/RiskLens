# Task 2: Beginner's Code Walkthrough

## What we've built

A CSV validator answers a **data engineering** question: *Does this table obey our documented OHLC structure and consistency rules?* It does **not** answer the finance/ML question: *Can we predict next-day downside risk?*

### Vocabulary you need

- **CSV:** A plain-text file where each line represents a row. Commas separate fields (unless a field is quoted).
- **DataFrame:** A table-like object provided by pandas, with columns and indexed rows.
- **Function:** A named block of code with input parameters and a return value.
- **Type annotation:** A hint such as `path: str | Path` describing acceptable types.
- **Exception:** A deliberate, descriptive error, raised when an operation cannot be safely completed.
- **Unit test:** A repeatable check of specific program behavior; here run with pytest.
- **CLI:** Command-line interface, so a user can invoke the validator from a terminal.

## Read these Python files in order

1. `validation.py`: `REQUIRED_COLUMNS` and `ACCEPTED_DATE_FORMATS` specify the input contract.
2. `_standardize_columns(...)`: trims and lowercases headers, then rejects missing/duplicated normalized names. Why? `Date` and ` date ` should mean the same thing; two `close` headers are ambiguous.
3. `_parse_dates(...)`: tries only **three explicit formats**, avoiding ambiguous day/month interpretation. A missing or invalid date raises a clear error.
4. `_parse_prices(...)`: converts numeric text into floats and checks for valid positive finite values. Only properly grouped thousands separators are accepted.
5. `validate_ohlc(...)`: combines checks, detects duplicate dates, verifies low/high relationships and sorts valid rows by date.
6. `load_ohlc_csv(...)`: reads a local CSV and passes it to the validator.
7. `cli.py`: prints PASS or FAIL and returns 0 for success or 1 for invalid data.
8. `tests/test_validation.py`: tests normal cases, missing columns, duplicate dates, invalid OHLC and command-line behavior.

## Examples to understand

```python
import pandas as pd

# A DataFrame is a table in memory:
prices = pd.DataFrame({"date": ["2026-09-01"], "close": [25080]})
print(prices["close"])  # Select the 'close' column
```

```python
from risklens.data.validation import load_ohlc_csv

validated = load_ohlc_csv("data/fixtures/sample_ohlc.csv")
print(validated.head())
print(validated.attrs["reordered"])  # Whether row sorting was necessary
```

### Why validate OHLC mathematically?

The session maximum must be at least as large as both the opening and closing value:

`high >= max(open, close)`

The session minimum must be at most as large as both:

`low <= min(open, close)`

A file violating these simple rules is internally inconsistent and must be investigated.

### What does a green test mean?

A passing test means the program behaved as the test expected for that test case. It is **not** proof that the financial data comes from the official provider, that all errors have been found, or that a risk model is accurate.

## Run from the repository root

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m risklens.data.cli data/fixtures/sample_ohlc.csv
```

## Questions to check your understanding (no need to answer yet)

1. Why would we reject a `high` value below `close`?
2. What is the difference between `pd.read_csv()` and `validate_ohlc()`?
3. What does a DataFrame represent?
4. Why keep synthetic test data separate from actual market observations?
5. Why is silently dropping rows dangerous in financial ML?
6. How does `raise OHLCValidationError(...)` help locate a problem?
7. Why do we allow `Date` and `date`, but not two columns named `Date` and `date`?
8. Why do we sort rows *after* checking duplicate dates?
9. What do the command-line exit codes 0 and 1 signify?
10. Why does PASS not mean the model makes accurate predictions?

### Demonstrate an intentionally failing input

```bash
python -m risklens.data.cli data/fixtures/invalid_ohlc.csv
```

This command **should report FAIL and exit with code 1** because the invented `high` is below the invented `close`. That is the correct behavior of a validator.
