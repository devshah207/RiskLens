"""Root-cause investigation: three flagged features only; synthetic data only."""
import numpy as np
import pandas as pd
import pytest

from risklens.cli import build_parser
from risklens.evaluation import drift_investigation as diag
from risklens.features.dataset import _wilder_rsi_14


def test_independent_ema50_and_wilder_rsi():
    rng = np.random.default_rng(42)
    close = 1000 * np.cumprod(1+rng.normal(0.001, 0.02, 400))
    ema = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean()
    actual = close / ema.to_numpy() - 1.0
    assert np.allclose(diag._independent_ema50(close), actual, equal_nan=True, atol=1e-11)
    rsi = _wilder_rsi_14(pd.Series(close)).to_numpy()
    assert np.allclose(diag._independent_rsi14(close), rsi, equal_nan=True, atol=1e-11)
    assert np.isnan(diag._independent_rsi14(close)[:14]).all()


def test_vix_overlap_verifies_original_and_new_source(tmp_path, monkeypatch):
    dates = pd.bdate_range("2024-01-08", periods=12)
    prior = (dates-pd.Timedelta(days=1))
    # Monday age1 is intentionally synthetic, used only to verify matching rules.
    old = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "vix_source_date": prior.strftime("%Y-%m-%d"),
        "india_vix_level": np.arange(len(dates), dtype=float)+15,
    })
    file = tmp_path/"dev.csv"
    old.to_csv(file, index=False)
    original_new_source_values = old.india_vix_level.to_numpy(copy=True)
    monkeypatch.setattr(diag, "_read_holdout_vix", lambda _:
        (
            pd.DataFrame({"vix_source_date": prior,
                          "india_vix_level": original_new_source_values}),
            {"quarantined_zero_ohl_rows": 0},
        )
    )
    audit = diag._vix_overlap_audit(file, "some-vix-file")
    assert audit["assessment"] == "PASS"
    assert audit["2024_overlap_sessions_matched"] == 12
    assert audit["source_date_mismatches"] == 0

    old.loc[0, "india_vix_level"] = 999
    old.to_csv(file, index=False)
    error = diag._vix_overlap_audit(file, "some-vix-file")
    assert error["assessment"] == "SOURCE_DISAGREEMENT_REQUIRES_REVIEW"
    assert error["value_mismatches"] == 1


def test_drift_report_years_and_distinct_baselines(tmp_path, monkeypatch):
    rng = np.random.default_rng(45)
    dev_dates = pd.bdate_range("2016-03-01", "2024-12-30")
    dates = pd.bdate_range("2025-01-02", "2026-09-29")
    dev = pd.DataFrame({
        "date": dev_dates,
        "ema_50_distance": rng.normal(0, 0.04, len(dev_dates)),
        "india_vix_level": rng.normal(17, 4, len(dev_dates)),
        "rsi_14": rng.normal(50, 9, len(dev_dates)),
    })
    hold = pd.DataFrame({
        "date": dates,
        "ema_50_distance": rng.normal(-0.07, 0.04, len(dates)),
        "india_vix_level": rng.normal(24, 5, len(dates)),
        "rsi_14": rng.normal(40, 9, len(dates)),
        "vix_source_date": dates - pd.Timedelta(days=1),
    })
    predictions = hold[["date"]].copy()
    predictions["actual_downside_event"] = (rng.random(len(dates)) < 0.12).astype(int)
    predictions["xgboost_11_fixed_probability"] = 0.11
    predictions["lstm_optuna_86_probability"] = 0.12
    for path in ("n.csv", "e.csv", "v.csv", "s.csv", "p.csv"):
        (tmp_path/path).write_text("synthetic")

    monkeypatch.setattr(
        diag, "load_feature_frames",
        lambda *args: (dev, hold, predictions, {"data": "synthetic"}),
    )
    monkeypatch.setattr(
        diag, "_indicator_integrity",
        lambda *_: {"formula_consistency": "PASS"},
    )
    monkeypatch.setattr(
        diag, "_vix_overlap_audit",
        lambda *args: {"assessment": "PASS", "2024_overlap_sessions_matched": 200},
    )
    args = (tmp_path/"n.csv", tmp_path/"e.csv",
            tmp_path/"v.csv", tmp_path/"s.csv", tmp_path/"p.csv")
    report, periods, months, extremes, slices = diag.investigate_three_features(*args)
    assert report["already_scored_holdout_rows"] == len(dates)
    assert report["vix_holdout_source_age"]["max_age_days"] == 1
    assert report["feature_findings"]["ema_50_distance"]["holdout_2026"]["n"] > 100
    assert set(periods.feature) == set(diag.FLAGGED)
    assert set(periods.period) == {
        "old_2016_2019", "search_2020_2022", "recent_2023_2024",
        "reference_2020_2024", "all_pre_2025", "holdout_2025",
        "holdout_2026", "holdout_all",
    }
    assert len(months) >= 60
    assert len(extremes) == 3 * 10
    assert len(slices) == 3 * 2 * 3
    assert periods.loc[
        (periods.feature == "india_vix_level") & (periods.period == "holdout_all"),
        "psi_against_2020_2024",
    ].iloc[0] > 0.20
    out = tmp_path/"reports"
    diag.save_investigation(*args, output_dir=out)
    assert all((out/file).exists() for file in diag.OUT)
    with pytest.raises(diag.DriftInvestigationError, match="--overwrite"):
        diag.save_investigation(*args, output_dir=out)


def test_invalid_vix_provenance_fails_loudly(tmp_path, monkeypatch):
    rng = np.random.default_rng(45)
    dev_dates = pd.bdate_range("2016-03-01", "2024-12-30")
    dates = pd.bdate_range("2025-01-02", "2026-09-29")
    dev = pd.DataFrame({
        "date": dev_dates,
        "ema_50_distance": 0.001+rng.normal(0, 0.05, len(dev_dates)),
        "india_vix_level": 15+rng.normal(0, 2, len(dev_dates)),
        "rsi_14": 50+rng.normal(0, 10, len(dev_dates)),
    })
    hold = pd.DataFrame({
        "date": dates,
        "ema_50_distance": 0.01,
        "india_vix_level": 20.0,
        "rsi_14": 55.0,
        # same-day VIX date must always be rejected
        "vix_source_date": dates,
    })
    pred = hold[["date"]].copy()
    pred["actual_downside_event"] = 0
    pred["xgboost_11_fixed_probability"] = 0.1
    pred["lstm_optuna_86_probability"] = 0.1
    monkeypatch.setattr(diag, "load_feature_frames",
                        lambda *args: (dev, hold, pred, {}))
    monkeypatch.setattr(diag, "_indicator_integrity",
                        lambda *args: {"formula_consistency": "PASS"})
    monkeypatch.setattr(diag, "_vix_overlap_audit",
                        lambda *args: {"assessment": "PASS"})
    f = tmp_path/"synthetic.csv"
    f.write_text("synthetic")
    with pytest.raises(diag.DriftInvestigationError, match="same-day"):
        diag.investigate_three_features(f, f, f, f, f)


def test_cli_targeted_investigation_registered():
    opts = build_parser().parse_args([
        "investigate-drift",
        "--vix", "v.csv", "--sp500", "s.csv",
    ])
    assert opts.command == "investigate-drift"
    assert opts.vix == "v.csv"
