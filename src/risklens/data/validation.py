"""Read and validate historical daily OHLC data for the RiskLens project.

Only checks data structure and internal consistency. It does NOT verify that
an observation came from NSE or that the prices are economically plausible.
"""

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("date", "open", "high", "low", "close")
ACCEPTED_DATE_FORMATS = ("%Y-%m-%d", "%d-%b-%Y", "%d %b %Y")


class OHLCValidationError(ValueError):
    """Raised when a source table violates the expected OHLC data contract."""


def _standardize_columns(source: pd.DataFrame) -> pd.DataFrame:
    """Make required header names lowercase; reject ambiguous duplicate names."""
    aliases = {str(name): str(name).strip().lower() for name in source.columns}
    normalized = list(aliases.values())
    duplicates = sorted({name for name in normalized if normalized.count(name) > 1})
    if duplicates:
        raise OHLCValidationError(
            f"Duplicate column names after normalization: {duplicates}. "
            "Rename or remove ambiguous columns in the source."
        )

    table = source.rename(columns=aliases)
    missing = [name for name in REQUIRED_COLUMNS if name not in table.columns]
    if missing:
        raise OHLCValidationError(
            f"Missing required columns: {missing}. Expected: {list(REQUIRED_COLUMNS)}."
        )
    return table


def _row_locations(mask: pd.Series) -> str:
    """Show up to five CSV line numbers; first data row is line 2."""
    positions = np.flatnonzero(mask.to_numpy())
    lines = [int(i) + 2 for i in positions[:5]]
    suffix = ", ..." if len(positions) > 5 else ""
    return ", ".join(map(str, lines)) + suffix


def _parse_dates(values: pd.Series) -> pd.Series:
    """Parse only documented unambiguous date formats; reject malformed dates."""
    strings = values.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    for date_format in ACCEPTED_DATE_FORMATS:
        candidate = pd.to_datetime(strings, format=date_format, errors="coerce")
        parsed = parsed.fillna(candidate)

    bad = parsed.isna()
    if bad.any():
        raise OHLCValidationError(
            "Invalid or missing dates at CSV lines "
            f"{_row_locations(bad)}. Allowed formats: YYYY-MM-DD, "
            "DD-Mon-YYYY, DD Mon YYYY."
        )
    return parsed


def _parse_prices(values: pd.Series, column: str) -> pd.Series:
    """Convert numeric price text to floats; reject missing/nonpositive/nonfinite."""
    strings = values.astype("string").str.strip()
    # Commas are common thousands separators in financial exports.
    contains_comma = strings.str.contains(",", regex=False, na=False)
    valid_grouping = strings.str.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", na=False)
    malformed_grouping = contains_comma & ~valid_grouping
    if malformed_grouping.any():
        raise OHLCValidationError(
            f"Invalid thousands separators in '{column}' at CSV lines "
            f"{_row_locations(malformed_grouping)}. Example valid value: 25,123.45."
        )

    without_separators = strings.str.replace(",", "", regex=False)
    numbers = pd.to_numeric(without_separators, errors="coerce")
    bad = (
        numbers.isna()
        | ~np.isfinite(numbers.to_numpy(dtype=float, na_value=np.nan))
        | (numbers <= 0)
    )
    if bad.any():
        raise OHLCValidationError(
            f"Invalid values in '{column}' at CSV lines {_row_locations(bad)}. "
            "Prices must be present, numeric, finite, and greater than zero."
        )
    return numbers.astype(float)


def validate_ohlc(source: pd.DataFrame) -> pd.DataFrame:
    """Return a checked, date-sorted table with only date/open/high/low/close.

    The returned DataFrame has two informational attrs:
    ``ignored_columns`` (extra source columns) and ``reordered`` (whether
    the original dates were out of chronological order).

    Errors raise OHLCValidationError, never silently drop/fill invalid rows.
    """
    if source.empty:
        raise OHLCValidationError("The CSV contains no data rows.")

    table = _standardize_columns(source)
    ignored = [name for name in table.columns if name not in REQUIRED_COLUMNS]
    clean = table.loc[:, list(REQUIRED_COLUMNS)].copy()

    clean["date"] = _parse_dates(clean["date"])
    for column in REQUIRED_COLUMNS[1:]:
        clean[column] = _parse_prices(clean[column], column)

    duplicates = clean["date"].duplicated(keep=False)
    if duplicates.any():
        raise OHLCValidationError(
            f"Duplicate trading dates at CSV lines {_row_locations(duplicates)}. "
            "Resolve duplicates in the source; do not silently deduplicate."
        )

    invalid_range = (
        (clean["low"] > clean["high"])
        | (clean["low"] > clean["open"])
        | (clean["low"] > clean["close"])
        | (clean["high"] < clean["open"])
        | (clean["high"] < clean["close"])
    )
    if invalid_range.any():
        raise OHLCValidationError(
            "Inconsistent OHLC relationships at CSV lines "
            f"{_row_locations(invalid_range)}. Require "
            "low <= min(open, close) and high >= max(open, close)."
        )

    reordered = not clean["date"].is_monotonic_increasing
    clean = clean.sort_values("date").reset_index(drop=True)
    clean.attrs["ignored_columns"] = ignored
    clean.attrs["reordered"] = reordered
    return clean


def load_ohlc_csv(path: str | Path) -> pd.DataFrame:
    """Read a UTF-8 CSV and validate its daily OHLC observations."""
    filepath = Path(path)
    if not filepath.is_file():
        raise OHLCValidationError(f"CSV file not found: {filepath}")

    try:
        raw = pd.read_csv(
            filepath,
            dtype="string",
            keep_default_na=False,
            skip_blank_lines=False,
            encoding="utf-8-sig",
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise OHLCValidationError(f"Could not read CSV '{filepath}': {exc}") from exc

    return validate_ohlc(raw)
