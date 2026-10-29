"""Synthetic test cases for merging annual NIFTY 50 exports; no real prices."""

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from risklens.data.batch_ingestion import import_nifty50_batch
from risklens.data.ingestion import ImportErrorData

HEADER = "Index Name,Date,Open,High,Low,Close\n"


def make_file(folder: Path, name: str, rows: list[str], header: str = HEADER) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    target.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    return target


def fixture_years(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    make_file(
        root,
        "one.csv",
        [
            "NIFTY 50,2024-12-26,100,101,99,100",
            "NIFTY 50,2024-12-27,100,101,97,98",
        ],
    )
    make_file(
        root,
        "two.csv",
        [
            "NIFTY 50,2024-12-30,98,102,97,100",
            "NIFTY 50,2025-01-02,100,101,98,99",
            "NIFTY 50,2025-01-03,99,100,93,94",
        ],
    )
    return root


def test_combines_files_and_counts_only_development_events(tmp_path):
    source = fixture_years(tmp_path)
    audit, events = import_nifty50_batch(source, tmp_path / "out")
    assert audit["number_of_files"] == 2
    assert audit["final_unique_trading_sessions"] == 5
    assert audit["observations_per_calendar_year"] == {"2024": 3, "2025": 2}
    assert events["eligible_prediction_days"] == 2  # 2024-12-30 is excluded
    assert events["downside_event_days"] == 1  # 100 -> 98
    assert events["event_rate"] == 0.5
    assert events["last_development_session_not_labeled"] == "2024-12-30"
    assert events["by_prediction_year"][0]["eligible_prediction_days"] == 2
    assert pd.read_csv(tmp_path / "out" / "nifty50_ohlc.csv")["date"].tolist() == [
        "2024-12-26",
        "2024-12-27",
        "2024-12-30",
        "2025-01-02",
        "2025-01-03",
    ]
    saved = json.loads(
        (tmp_path / "out" / "nifty50_event_frequency_development.json").read_text()
    )
    assert saved == events
    assert "2025" not in str(events["by_prediction_year"])


def test_overlap_identical_values_reported_and_deduplicated(tmp_path):
    source = fixture_years(tmp_path)
    make_file(source, "overlap.csv", ["NIFTY 50,2024-12-30,98,102,97,100"])
    audit, _ = import_nifty50_batch(source, tmp_path / "out")
    assert audit["identical_cross_file_overlap_rows_removed"] == 1
    assert audit["cross_file_overlap_dates"] == ["2024-12-30"]
    assert audit["final_unique_trading_sessions"] == 5


def test_conflicting_overlap_blocks_output(tmp_path):
    source = fixture_years(tmp_path)
    make_file(source, "overlap.csv", ["NIFTY 50,2024-12-30,98,102,97,101"])
    with pytest.raises(ImportErrorData, match="Conflicting OHLC"):
        import_nifty50_batch(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_bad_csv_does_not_create_partial_output(tmp_path):
    source = fixture_years(tmp_path)
    make_file(source, "bad.csv", ["NIFTY 50,2024-01-01,100,90,80,100"])
    with pytest.raises(ImportErrorData, match="bad.csv"):
        import_nifty50_batch(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_missing_index_name_requires_explicit_assumption(tmp_path):
    source = tmp_path / "raw"
    make_file(
        source,
        "no_index.csv",
        ["2024-10-01,100,105,95,102", "2024-10-02,102,107,98,100"],
        "Date,Open,High,Low,Close\n",
    )
    with pytest.raises(ImportErrorData, match="assume-nifty50"):
        import_nifty50_batch(source, tmp_path / "out")
    audit, _ = import_nifty50_batch(source, tmp_path / "out", assume_nifty50=True)
    assert audit["input_files"][0]["identity_assumed_without_index_name"]


def test_non_nifty_index_is_excluded(tmp_path):
    source = fixture_years(tmp_path)
    make_file(
        source,
        "other.csv",
        [
            "NIFTY BANK,2024-03-01,100,105,99,101",
            "NIFTY 50,2024-10-01,100,105,99,101",
        ],
    )
    audit, _ = import_nifty50_batch(source, tmp_path / "out")
    selected = next(
        item for item in audit["input_files"] if item["source_filename"] == "other.csv"
    )
    assert selected["non_nifty50_rows_excluded"] == 1


def test_warns_on_gap_but_does_not_invent_trading_days(tmp_path):
    source = tmp_path / "raw"
    make_file(
        source,
        "first.csv",
        [
            "NIFTY 50,2024-01-02,100,103,98,101",
            "NIFTY 50,2024-02-02,101,103,98,99",
        ],
    )
    audit, _ = import_nifty50_batch(source, tmp_path / "out")
    assert len(audit["unusually_long_calendar_gaps"]) == 1
    assert audit["unusually_long_calendar_gaps"][0]["calendar_days_between"] == 31
    assert audit["final_unique_trading_sessions"] == 2


def test_output_must_not_be_overwritten_without_consent(tmp_path):
    source = fixture_years(tmp_path)
    import_nifty50_batch(source, tmp_path / "out")
    with pytest.raises(ImportErrorData, match="already exists"):
        import_nifty50_batch(source, tmp_path / "out")
    audit, _ = import_nifty50_batch(source, tmp_path / "out", overwrite=True)
    assert audit["final_unique_trading_sessions"] == 5


def test_bad_cutoff_fails_without_output(tmp_path):
    source = fixture_years(tmp_path)
    with pytest.raises(ImportErrorData, match="YYYY-MM-DD"):
        import_nifty50_batch(source, tmp_path / "out", holdout_start="01-01-2025")
    assert not (tmp_path / "out").exists()


def test_requires_two_development_sessions(tmp_path):
    source = fixture_years(tmp_path)
    with pytest.raises(ImportErrorData, match="at least two"):
        import_nifty50_batch(source, tmp_path / "out", holdout_start="2024-12-27")


def test_missing_directory_or_no_csv(tmp_path):
    with pytest.raises(ImportErrorData, match="does not exist"):
        import_nifty50_batch(tmp_path / "missing", tmp_path / "out")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ImportErrorData, match="No CSV"):
        import_nifty50_batch(tmp_path / "empty", tmp_path / "out")


def test_no_future_data_features_or_labels_in_event_report(tmp_path):
    source = fixture_years(tmp_path)
    _, events = import_nifty50_batch(source, tmp_path / "out")
    assert events["holdout_starts_on"] == "2025-01-01"
    assert events["eligible_prediction_days"] == 2
    assert events["non_event_days"] == 1
    assert events["downside_event_days"] == 1


def test_cli_pass_and_fail_codes(tmp_path):
    source = fixture_years(tmp_path)
    command = [
        sys.executable,
        "-m",
        "risklens.data.batch_cli",
        str(source),
        "--output-dir",
        str(tmp_path / "out"),
    ]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr + done.stdout
    assert "PASS: 2 files, 5 unique trading sessions" in done.stdout
    again = subprocess.run(command, capture_output=True, text=True, check=False)
    assert again.returncode == 1
    assert "FAIL:" in again.stdout
