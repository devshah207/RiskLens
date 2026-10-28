"""Behavioral tests for daily OHLC data validation."""

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from risklens.data.validation import OHLCValidationError, load_ohlc_csv, validate_ohlc


FIXTURE = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "sample_ohlc.csv"


def example_table() -> pd.DataFrame:
    """Simple, deliberately synthetic data used by individual tests."""
    return pd.DataFrame(
        {
            "date": ["2026-09-01", "2026-09-02"],
            "open": [100, 102],
            "high": [104, 105],
            "low": [99, 100],
            "close": [102, 103],
        }
    )


def test_sample_fixture_loads():
    result = load_ohlc_csv(FIXTURE)
    assert len(result) == 5
    assert result["date"].is_monotonic_increasing
    assert list(result.columns) == ["date", "open", "high", "low", "close"]


def test_headers_are_case_and_space_insensitive():
    table = example_table().rename(columns={"date": " Date ", "close": "CLOSE"})
    result = validate_ohlc(table)
    assert "date" in result.columns
    assert "close" in result.columns


def test_additional_columns_not_used_as_features():
    table = example_table()
    table["Volume"] = [120, 140]
    result = validate_ohlc(table)
    assert "volume" not in result.columns
    assert result.attrs["ignored_columns"] == ["volume"]


def test_unsorted_valid_dates_are_sorted_and_reported():
    table = example_table().iloc[::-1].reset_index(drop=True)
    result = validate_ohlc(table)
    assert result["date"].is_monotonic_increasing
    assert result.attrs["reordered"] is True


def test_extra_date_formats_are_explicit_and_unambiguous():
    table = example_table()
    table["date"] = ["01-Sep-2026", "02 Sep 2026"]
    assert validate_ohlc(table)["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-09-01", "2026-09-02"
    ]


@pytest.mark.parametrize(
    "edit,error_fragment",
    [
        (lambda t: t.drop(columns=["close"]), "Missing required columns"),
        (lambda t: t.assign(date=["2026-09-01"] * 2), "Duplicate trading dates"),
        (lambda t: t.assign(date=["2026-13-40", "2026-09-02"]), "Invalid or missing dates"),
        (lambda t: t.assign(date=["01/09/2026", "02/09/2026"]), "Invalid or missing dates"),
        (lambda t: t.assign(open=[0, 102]), "Invalid values in 'open'"),
        (lambda t: t.assign(close=[-2, 103]), "Invalid values in 'close'"),
        (lambda t: t.assign(low=[float("inf"), 100]), "Invalid values in 'low'"),
        (lambda t: t.assign(high=["none", 105]), "Invalid values in 'high'"),
        (lambda t: t.assign(high=[float("nan"), 105]), "Invalid values in 'high'"),
        (lambda t: t.assign(high=[100, 105]), "Inconsistent OHLC relationships"),
        (lambda t: t.assign(low=[103, 100]), "Inconsistent OHLC relationships"),
        (lambda t: t.assign(high=[98, 105], low=[99, 100]), "Inconsistent OHLC relationships"),
    ],
)
def test_invalid_data_fails_loudly(edit, error_fragment):
    with pytest.raises(OHLCValidationError, match=error_fragment):
        validate_ohlc(edit(example_table()))


def test_duplicate_headers_after_normalization_are_rejected():
    table = example_table()
    table["CLOSE"] = [102, 103]
    with pytest.raises(OHLCValidationError, match="Duplicate column names"):
        validate_ohlc(table)


def test_empty_table_fails():
    with pytest.raises(OHLCValidationError, match="no data rows"):
        validate_ohlc(example_table().iloc[:0])


def test_missing_file_fails(tmp_path):
    with pytest.raises(OHLCValidationError, match="CSV file not found"):
        load_ohlc_csv(tmp_path / "absent.csv")


def test_blank_csv_row_not_silently_skipped(tmp_path):
    csv = tmp_path / "blank_row.csv"
    csv.write_text("date,open,high,low,close\n2026-09-01,100,110,90,105\n\n2026-09-02,102,110,95,108\n")
    with pytest.raises(OHLCValidationError, match="Invalid or missing dates"):
        load_ohlc_csv(csv)


def test_thousands_separators_are_accepted():
    table = example_table().astype({"close": "string", "high": "string"})
    table.loc[0, "close"] = "1,002.50"
    table.loc[0, "high"] = "1,100.00"
    assert validate_ohlc(table).loc[0, "close"] == 1002.50


def test_cli_returns_zero_on_valid_fixture():
    completed = subprocess.run(
        [sys.executable, "-m", "risklens.data.cli", str(FIXTURE)],
        capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0
    assert "PASS:" in completed.stdout


def test_cli_returns_nonzero_on_invalid_file(tmp_path):
    csv = tmp_path / "invalid.csv"
    csv.write_text("date,open,high,low,close\n2026-09-01,100,95,90,103\n")
    completed = subprocess.run(
        [sys.executable, "-m", "risklens.data.cli", str(csv)],
        capture_output=True, text=True, check=False
    )
    assert completed.returncode == 1
    assert "FAIL:" in completed.stdout


def test_malformed_thousands_separator_fails():
    table = example_table().astype({"close": "string"})
    table.loc[0, "close"] = "12,3"
    with pytest.raises(OHLCValidationError, match="Invalid thousands separators"):
        validate_ohlc(table)
