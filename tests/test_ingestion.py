"""Tests of financial CSV importing; ALL example values are synthetic."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from risklens.data.ingestion import ImportErrorData, import_nifty50_csv


FIXTURE = (
    Path(__file__).resolve().parents[1] / "data" / "fixtures" / "sample_nse_export.csv"
)


def create_source(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "export.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_import_filters_index_and_writes_provenance(tmp_path):
    before = FIXTURE.read_bytes()
    output = tmp_path / "processed"
    report = import_nifty50_csv(
        FIXTURE, output, source_page="https://example.org/source"
    )
    assert report["source_rows"] == 4
    assert report["excluded_non_nifty50_rows"] == 1
    assert report["rows_imported"] == 3
    assert report["input_sha256"] == sha256(before).hexdigest()
    assert report["first_session"] == "2026-09-01"
    assert report["last_session"] == "2026-09-03"
    assert report["claimed_source_page"] == "https://example.org/source"
    assert FIXTURE.read_bytes() == before  # importer never alters raw file
    result = pd.read_csv(output / "nifty50_ohlc.csv")
    assert list(result.columns) == ["date", "open", "high", "low", "close"]
    assert result["close"].tolist() == [25080.0, 24750.0, 24850.0]
    from_disk = json.loads((output / "nifty50_import_report.json").read_text())
    assert from_disk == report


def test_source_without_index_name_requires_confirmation(tmp_path):
    source = create_source(
        tmp_path, "Date,Open,High,Low,Close\n01-Sep-2026,100,110,90,105\n"
    )
    with pytest.raises(ImportErrorData, match="assume-nifty50"):
        import_nifty50_csv(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_explicit_confirmation_for_single_series_without_name(tmp_path):
    source = create_source(
        tmp_path, "Date,Open,High,Low,Close\n01-Sep-2026,100,110,90,105\n"
    )
    report = import_nifty50_csv(source, tmp_path / "out", assume_nifty50=True)
    assert report["identity_assumed_without_index_column"] is True
    assert "not verified" in " ".join(report["warnings"])


def test_alternative_nse_column_headers(tmp_path):
    source = create_source(
        tmp_path,
        "Index Name,Index Date,Open,High,Low,Closing,Shares Traded\n"
        "Nifty 50,01-Jan-2019,100,110,90,105,100000\n",
    )
    report = import_nifty50_csv(source, tmp_path / "out")
    assert report["rows_imported"] == 1
    assert "shares traded" in report["ignored_extra_columns"]
    assert pd.read_csv(tmp_path / "out/nifty50_ohlc.csv").loc[0, "date"] == "2019-01-01"


def test_ambiguous_aliases_fail(tmp_path):
    source = create_source(
        tmp_path,
        "Index Name,Date,Index Date,Open,High,Low,Close\n"
        "NIFTY 50,2026-09-01,2026-09-01,100,110,90,105\n",
    )
    with pytest.raises(ImportErrorData, match="ambiguous"):
        import_nifty50_csv(source, tmp_path / "out")


def test_no_matching_index_fails(tmp_path):
    source = create_source(
        tmp_path,
        "Index Name,Date,Open,High,Low,Close\nNIFTY BANK,2026-09-01,100,110,90,105\n",
    )
    with pytest.raises(ImportErrorData, match="No 'NIFTY 50'"):
        import_nifty50_csv(source, tmp_path / "out")


def test_blank_index_name_fails(tmp_path):
    source = create_source(
        tmp_path, "Index Name,Date,Open,High,Low,Close\n,2026-09-01,100,110,90,105\n"
    )
    with pytest.raises(ImportErrorData, match="Missing values"):
        import_nifty50_csv(source, tmp_path / "out")


def test_invalid_ohlc_does_not_write_outputs(tmp_path):
    source = create_source(
        tmp_path,
        "Index Name,Date,Open,High,Low,Close\nNIFTY 50,2026-09-01,100,95,90,105\n",
    )
    with pytest.raises(ImportErrorData, match="OHLC validation failed"):
        import_nifty50_csv(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_overwriting_outputs_requires_consent(tmp_path):
    output = tmp_path / "processed"
    import_nifty50_csv(FIXTURE, output)
    with pytest.raises(ImportErrorData, match="already exists"):
        import_nifty50_csv(FIXTURE, output)
    report = import_nifty50_csv(FIXTURE, output, overwrite=True)
    assert report["rows_imported"] == 3


def test_unordered_dates_are_sorted_and_disclosed(tmp_path):
    source = create_source(
        tmp_path,
        "Index Name,Date,Open,High,Low,Close\n"
        "NIFTY 50,2026-09-03,100,110,90,105\n"
        "NIFTY 50,2026-09-01,100,110,90,104\n",
    )
    report = import_nifty50_csv(source, tmp_path / "out")
    assert any("sorted" in item for item in report["warnings"])
    assert pd.read_csv(tmp_path / "out/nifty50_ohlc.csv")["date"].tolist() == [
        "2026-09-01",
        "2026-09-03",
    ]


def test_command_line_exit_codes(tmp_path):
    command = [
        sys.executable,
        "-m",
        "risklens.data.import_cli",
        str(FIXTURE),
        "--output-dir",
        str(tmp_path / "out"),
    ]
    success = subprocess.run(command, capture_output=True, text=True, check=False)
    assert success.returncode == 0
    assert "PASS: Imported 3" in success.stdout
    failure = subprocess.run(command, capture_output=True, text=True, check=False)
    assert failure.returncode == 1
    assert "FAIL:" in failure.stdout
