"""Synthetic-only final-holdout contract tests; NEVER open real 2025+ data."""
import numpy as np
import pandas as pd
import pytest

from risklens.cli import build_parser
from risklens.evaluation import final_holdout as holdout


def test_unlock_required(tmp_path):
    with pytest.raises(holdout.HoldoutError, match="confirm-locked-protocol"):
        holdout.save_holdout("n.csv", "e.csv", "v.csv", "s.csv", "study.db", tmp_path)


def test_one_time_output_guard(tmp_path):
    target = tmp_path / "out"
    target.mkdir()
    (target / holdout.OUTPUT_REPORT).write_text("previously evaluated")
    with pytest.raises(holdout.HoldoutError, match="refuses overwrite"):
        holdout.save_holdout("n.csv", "e.csv", "v.csv", "s.csv", "study.db",
                             target, confirm_locked_protocol=True)


def test_history_date_guard():
    x = pd.DataFrame({
        "date": pd.to_datetime(["2024-12-31", "2025-01-02"]),
        **{f: [1.0, 1.0] for f in holdout.FEATURES},
    })
    labeled = x.loc[1:, ["date"]].copy()
    labeled[holdout.TARGET_COLUMN] = 0
    with pytest.raises(holdout.HoldoutError, match="No holdout dates"):
        holdout._holdout_sequences(x, labeled)


def test_vix_same_day_source_never_used():
    valid = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
        **{f: [0.1, 0.1] for f in holdout.FULL_FEATURE_COLUMNS},
        holdout.TARGET_COLUMN: [0, 1],
    })
    vix = pd.DataFrame({
        "vix_source_date": pd.to_datetime(["2024-12-31", "2025-01-02", "2025-01-03"]),
        "india_vix_level": [10.0, 20.0, 30.0],
    })
    sp = pd.DataFrame({
        "sp500_source_date": pd.to_datetime(["2024-12-31", "2025-01-02", "2025-01-03"]),
        "sp500_previous_return_1d": [0.01, 0.02, 0.03],
    })
    out, audit = holdout._attach_holdout_external(valid, vix, sp)
    assert list(out.india_vix_level) == [10.0, 20.0]
    assert list(out.sp500_previous_return_1d) == [0.01, 0.02]
    assert audit["total_labeled_holdout_dates"] == 2
    assert audit["complete_current_day_external"] == 2


def test_end_of_holdout_label_missing():
    """Only labels whose next close exists within the locked end date are scored."""
    # Full integration below covers this with real feature construction.
    assert holdout.END_EXCLUSIVE == pd.Timestamp("2026-10-01")
    assert holdout.EXPECTED_LAST_SESSION == pd.Timestamp("2026-09-30")


def test_holdout_cli_needs_confirmation():
    parser = build_parser()
    args = parser.parse_args(["evaluate-holdout", "--vix", "v.csv", "--sp500", "s.csv"])
    assert args.command == "evaluate-holdout"
    assert not args.confirm_locked_protocol


def test_synthetic_full_holdout_pipeline(tmp_path, monkeypatch):
    """Exercise original causal features, pre-2025 purge, VIX lag, VaR and reporting.

    All data are SYNTHETIC, with artificial dates and prices. This is not
    a substitute for a real locked holdout run.
    """
    dates = pd.bdate_range("2016-01-01", "2026-09-30")
    n = len(dates)
    # Deterministic rare-ish 2% price drops and positive sessions.
    returns = np.where(np.arange(n) % 11 == 0, -0.02, 0.002)
    close = 1000 * np.cumprod(1 + returns)
    nifty = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": close * 0.998,
        "high": close * 1.012,
        "low": close * 0.986,
        "close": close,
    })
    nifty_csv = tmp_path / "nifty.csv"
    nifty.to_csv(nifty_csv, index=False)
    dt = pd.Series(dates)
    prev = dt.shift(1)
    age = (dt - prev).dt.days
    dev_mask = dates < pd.Timestamp("2025-01-01")
    dev = nifty.loc[dev_mask].copy().reset_index(drop=True)
    dev_dt = dt[dev_mask].reset_index(drop=True)
    dev_prev = prev[dev_mask].reset_index(drop=True)
    dev_age = age[dev_mask].reset_index(drop=True)
    dev["vix_source_date"] = dev_prev.dt.strftime("%Y-%m-%d")
    dev["india_vix_level"] = np.where(dev_prev.notna(), 15.0, np.nan)
    dev["vix_age_calendar_days"] = dev_age
    dev["sp500_source_date"] = dev_prev.dt.strftime("%Y-%m-%d")
    dev["sp500_previous_close"] = np.where(dev_prev.notna(), 4000.0, np.nan)
    dev["sp500_previous_return_1d"] = np.where(dev_prev.notna(), 0.001, np.nan)
    dev["sp500_age_calendar_days"] = dev_age
    dev["external_features_ready"] = dev_prev.notna()
    dev_csv = tmp_path / "development_external.csv"
    dev.to_csv(dev_csv, index=False)
    raw_dates = dates[dates >= pd.Timestamp("2024-12-02")]
    vix = pd.DataFrame({
        "Date": raw_dates.strftime("%Y-%m-%d"),
        "Open": 14.0,
        "High": 16.0,
        "Low": 13.0,
        "Close": 15.0,
    })
    vix_csv = tmp_path / "vix.csv"
    vix.to_csv(vix_csv, index=False)
    sp = pd.DataFrame({
        "date": raw_dates.strftime("%Y-%m-%d"),
        "open": 4000.0,
        "high": 4010.0,
        "low": 3985.0,
        "close": 4000 + np.arange(len(raw_dates), dtype=float) * 1.5,
    })
    # Prevent close from ever exceeding high so synthetic OHLC remains valid.
    sp["high"] = sp.close + 15
    sp["low"] = sp.close - 15
    sp["open"] = sp.close - 2
    sp_csv = tmp_path / "sp500.csv"
    sp.to_csv(sp_csv, index=False)
    monkeypatch.setattr(
        holdout, "verify_trial86",
        lambda path: {"trial_number": 86, "database_sha256": "synthetic-test-only"}
    )
    monkeypatch.setattr(
        holdout, "_lstm_one",
        lambda train_x, train_y, valid_x, params, seed, device: np.full(len(valid_x), 0.11),
    )

    class FakeXGB:
        def __init__(self, **params):
            assert params == holdout.FIXED_XGB_PARAMS

        def fit(self, x, y):
            assert len(x) > 100 and set(y) == {0, 1}
            return self

        def predict_proba(self, x):
            return np.column_stack([np.full(len(x), 0.9), np.full(len(x), 0.1)])

    monkeypatch.setattr(holdout, "XGBClassifier", FakeXGB)
    out = tmp_path / "final_report"
    result = holdout.save_holdout(
        nifty_csv, dev_csv, vix_csv, sp_csv, "synthetic.sqlite3",
        out, device="cpu", confirm_locked_protocol=True
    )
    assert result["training_cutoff_exclusive"] == "2025-01-01"
    assert result["split_purge_count"] == 1
    assert result["pre_2025_labeled_training_rows"] > 100
    assert result["eligible_holdout_rows"] > 200
    assert result["excluded_holdout_rows_incomplete_45_session_history"] == 0
    assert sorted(result["yearly_scores"]) == ["2025", "2026"]
    assert sorted(result["historical_var"]["95"]["annual_exceptions"]) == ["2025", "2026"]
    assert len(pd.read_csv(out / holdout.OUTPUT_PREDICTIONS)) == result["eligible_holdout_rows"]
    assert all((out / filename).is_file() for filename in (
        holdout.OUTPUT_PREDICTIONS, holdout.OUTPUT_REPORT,
        holdout.OUTPUT_RELIABILITY, holdout.OUTPUT_MEMO,
    ))
    preds = pd.read_csv(out / holdout.OUTPUT_PREDICTIONS)
    assert (preds.date < "2026-09-30").all()
    assert (preds.date >= "2025-01-01").all()
