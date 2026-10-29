"""Synthetic tests for strictly earlier-date external market data joins.

All fixture price levels are made up ONLY for tests, never market-data claims.
"""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from risklens.data.external_integration import (
    ExternalDataError, integrate_external_development, save_external_development,
)


def source_files(tmp_path: Path):
    nifty = tmp_path / "nifty.csv"
    dates = ["2024-12-27", "2024-12-30", "2024-12-31", "2025-01-02"]
    prices = [100., 101., 99., 400.]
    pd.DataFrame({"date": dates, "open": prices, "high": prices,
                  "low": prices, "close": prices}).to_csv(nifty, index=False)
    vdir = tmp_path / "vix"; vdir.mkdir()
    vix = vdir / "vix_2024.csv"
    vix.write_text("Date ,Open ,High ,Low ,Close ,Prev. Close\n"
                   "26-DEC-2024,18,19,17,18,18\n"
                   "27-DEC-2024,19,20,18,19,18\n"
                   "30-DEC-2024,50,51,49,50,19\n"
                   "31-DEC-2024,60,61,59,60,50\n", encoding="utf-8")
    sp = tmp_path / "sp500.csv"
    sp.write_text("Date,Open,High,Low,Close,Adj Close,Volume\n"
                  "2024-12-26,1000,1010,990,1000,1000,500\n"
                  "2024-12-27,1020,1030,1010,1020,1020,500\n"
                  "2024-12-30,1040,1050,1030,1040,1040,500\n"
                  "2024-12-31,9999,10000,9990,9999,9999,500\n"
                  "2025-01-02,12000,12001,11999,12000,12000,500\n", encoding="utf-8")
    return nifty, vdir, sp


def test_strict_prior_date_and_us_daily_return(tmp_path):
    nifty, vix, sp = source_files(tmp_path)
    result, report = integrate_external_development(nifty, vix, sp)
    assert len(result) == 3
    assert result["date"].max() == pd.Timestamp("2024-12-31")
    dec30 = result.loc[result["date"] == "2024-12-30"].iloc[0]
    assert dec30["vix_source_date"] == pd.Timestamp("2024-12-27")
    assert dec30["india_vix_level"] == 19
    assert dec30["vix_age_calendar_days"] == 3
    assert dec30["sp500_source_date"] == pd.Timestamp("2024-12-27")
    assert dec30["sp500_previous_return_1d"] == pytest.approx(.02)
    assert dec30["sp500_age_calendar_days"] == 3
    dec31 = result.loc[result["date"] == "2024-12-31"].iloc[0]
    assert dec31["vix_source_date"] == pd.Timestamp("2024-12-30")
    assert dec31["sp500_source_date"] == pd.Timestamp("2024-12-30")
    assert dec31["sp500_previous_return_1d"] == pytest.approx(1040/1020-1)
    assert report["nifty_rows_with_both_external_features"] == 2


def test_2025_market_prices_cannot_affect_2024_features(tmp_path):
    nifty, vix, sp = source_files(tmp_path)
    original, report1 = integrate_external_development(nifty, vix, sp)
    nf = pd.read_csv(nifty)
    nf.loc[nf["date"] >= "2025-01-01", ["open", "high", "low", "close"]] = 9000
    nf.to_csv(nifty, index=False)
    sf = pd.read_csv(sp)
    sf.loc[sf["Date"] >= "2025-01-01", ["Open", "High", "Low", "Close"]] = 9000
    sf.to_csv(sp, index=False)
    after, report2 = integrate_external_development(nifty, vix, sp)
    pd.testing.assert_frame_equal(original, after)
    assert report1["nifty_rows_with_both_external_features"] == report2["nifty_rows_with_both_external_features"]


def test_no_same_date_information_used_even_if_updated(tmp_path):
    nifty, vix, sp = source_files(tmp_path)
    before, _ = integrate_external_development(nifty, vix, sp)
    vi = pd.read_csv(vix / "vix_2024.csv")
    vi.loc[vi["Date "] == "30-DEC-2024", ["Open ", "High ", "Low ", "Close "]] = [900, 901, 899, 900]
    vi.to_csv(vix / "vix_2024.csv", index=False)
    spf = pd.read_csv(sp)
    spf.loc[spf["Date"] == "2024-12-30", ["Open", "High", "Low", "Close"]] = [3000, 3001, 2999, 3000]
    spf.to_csv(sp, index=False)
    after, _ = integrate_external_development(nifty, vix, sp)
    for col in ("india_vix_level", "sp500_previous_return_1d"):
        assert before.loc[1, col] == pytest.approx(after.loc[1, col])


def test_vix_zero_open_high_low_quarantined(tmp_path):
    nifty, vix, sp = source_files(tmp_path)
    file = vix / "vix_2024.csv"
    contents = file.read_text().replace("27-DEC-2024,19,20,18,19,18", "27-DEC-2024,0,0,0,19,0")
    file.write_text(contents)
    result, report = integrate_external_development(nifty, vix, sp)
    assert report["vix_source_audit"]["zero_ohl_rows_quarantined"] == ["2024-12-27"]
    assert result.loc[1, "vix_source_date"] == pd.Timestamp("2024-12-26")
    assert result.loc[1, "india_vix_level"] == 18


def test_old_external_records_left_missing(tmp_path):
    nifty, vix, sp = source_files(tmp_path)
    joined, audit = integrate_external_development(nifty, vix, sp, max_age_days=1)
    dec30 = joined.loc[1]
    assert pd.isna(dec30["india_vix_level"])
    assert pd.isna(dec30["sp500_previous_return_1d"])
    assert not bool(dec30["external_features_ready"])
    assert audit["nifty_rows_without_both"] > 0


def test_no_future_backfill_for_first_observation(tmp_path):
    nifty, vix, sp = source_files(tmp_path)
    result, _ = integrate_external_development(nifty, vix, sp)
    assert result.loc[0, "vix_source_date"] == pd.Timestamp("2024-12-26")
    assert pd.isna(result.loc[0, "sp500_previous_return_1d"])
    assert not bool(result.loc[0, "external_features_ready"])


def test_duplicate_vix_date_across_files_is_rejected(tmp_path):
    nifty, vix, sp = source_files(tmp_path)
    (vix / "another.csv").write_text("Date,Open,High,Low,Close\n27-DEC-2024,19,20,18,19\n")
    with pytest.raises(ExternalDataError, match="duplicate date"):
        integrate_external_development(nifty, vix, sp)


def test_corrupted_vix_high_fails_not_silently_repaired(tmp_path):
    nifty, vix, sp = source_files(tmp_path)
    file = vix / "vix_2024.csv"
    contents = file.read_text().replace("26-DEC-2024,18,19,17,18,18", "26-DEC-2024,18,15,17,18,18")
    file.write_text(contents)
    with pytest.raises(ExternalDataError, match="OHLC failed validation"):
        integrate_external_development(nifty, vix, sp)


def test_negative_or_missing_vix_closing_level_is_rejected(tmp_path):
    nifty, vix, sp = source_files(tmp_path)
    file = vix / "vix_2024.csv"
    file.write_text(file.read_text().replace("26-DEC-2024,18,19,17,18,18", "26-DEC-2024,18,19,17,0,18"))
    with pytest.raises(ExternalDataError, match="Close must be strictly positive"):
        integrate_external_development(nifty, vix, sp)


def test_sp500_bad_ohlc_detected(tmp_path):
    nifty, vix, sp = source_files(tmp_path)
    sp.write_text(sp.read_text().replace("2024-12-27,1020,1030,1010,1020", "2024-12-27,1020,1010,1010,1020"))
    with pytest.raises(ExternalDataError, match="S&P 500 OHLC failed validation"):
        integrate_external_development(nifty, vix, sp)


def test_sp500_duplicate_dates_detected(tmp_path):
    nifty, vix, sp = source_files(tmp_path)
    sp.write_text(sp.read_text() + "2024-12-27,1020,1030,1010,1020,1020,500\n")
    with pytest.raises(ExternalDataError, match="duplicate"):
        integrate_external_development(nifty, vix, sp)


@pytest.mark.parametrize("age", [-2, 0, 1.5, True, 31])
def test_reject_invalid_staleness_parameters(tmp_path, age):
    nifty, vix, sp = source_files(tmp_path)
    with pytest.raises(ExternalDataError, match="max_age_days"):
        integrate_external_development(nifty, vix, sp, max_age_days=age)


def test_missing_input_directory_or_file(tmp_path):
    nifty, vix, sp = source_files(tmp_path)
    with pytest.raises(ExternalDataError, match="directory not found"):
        integrate_external_development(nifty, tmp_path / "missing", sp)
    with pytest.raises(ExternalDataError, match="CSV file not found"):
        integrate_external_development(nifty, vix, tmp_path / "no.csv")


def test_save_report_and_refuse_overwrite(tmp_path):
    nifty, vix, sp = source_files(tmp_path)
    out = tmp_path / "processed"
    audit = save_external_development(nifty, vix, sp, out)
    saved = pd.read_csv(out / "nifty50_external_development.csv")
    assert list(saved.columns) == [
        "date", "open", "high", "low", "close", "vix_source_date", "india_vix_level",
        "vix_age_calendar_days", "sp500_source_date", "sp500_previous_close",
        "sp500_previous_return_1d", "sp500_age_calendar_days", "external_features_ready",
    ]
    assert saved["date"].max() < "2025-01-01"
    assert json.loads((out / "external_integration_report.json").read_text()) == audit
    assert audit["vix_source_audit"]["files"][0]["sha256"] == sha256((vix / "vix_2024.csv").read_bytes()).hexdigest()
    with pytest.raises(ExternalDataError, match="output exists"):
        save_external_development(nifty, vix, sp, out)
    report2 = save_external_development(nifty, vix, sp, out, overwrite=True)
    assert report2["nifty_rows_with_both_external_features"] == audit["nifty_rows_with_both_external_features"]


def test_cli_success_and_refuses_second_write(tmp_path):
    nifty, vix, sp = source_files(tmp_path)
    command = [sys.executable, "-m", "risklens.data.external_cli", str(nifty), str(vix), str(sp),
               "--output-dir", str(tmp_path / "out")]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "PASS:" in done.stdout
    again = subprocess.run(command, capture_output=True, text=True, check=False)
    assert again.returncode == 1
    assert "FAIL:" in again.stdout


def test_vix_2025_sentinel_excluded(tmp_path):
    nifty, vix, sp = source_files(tmp_path)
    initial, _ = integrate_external_development(nifty, vix, sp)
    with (vix / "vix_2025.csv").open("w") as stream:
        stream.write("Date,Open,High,Low,Close\n02-JAN-2025,100,110,90,105\n")
    changed, audit = integrate_external_development(nifty, vix, sp)
    pd.testing.assert_frame_equal(initial, changed)
    assert audit["vix_source_audit"]["usable_pre_holdout_rows"] == 4
