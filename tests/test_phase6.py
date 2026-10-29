"""Phase 6 tests: score drift audit, no post-hoc selection, pre-holdout SHAP."""
import numpy as np
import pandas as pd
import pytest

from risklens.cli import build_parser
from risklens.evaluation.monitoring import (
    MonitoringError,
    _fixed_bin_drift,
    evaluate_monitoring,
    save_monitoring,
    OUTPUTS as MONITOR_FILES,
)
from risklens.models import explainability


def _make_predictions(tmp_path):
    records = []
    for year in range(2020, 2027):
        # Distinct class labels and rare-ish risk events; synthetic data only.
        for n in range(60):
            d = pd.Timestamp(f"{year}-02-01") + pd.Timedelta(days=n)
            y = int(n % 9 == 0)
            risk_x = min(0.99, 0.06 + 0.03 * (n % 4) + 0.015 * (year - 2020))
            risk_l = min(0.99, 0.05 + 0.02 * (n % 5) + 0.02 * (year - 2020))
            records.append({
                "validation_year": year,
                "date": d.strftime("%Y-%m-%d"),
                "actual_downside_event": y,
                "xgboost_11_fixed_probability": risk_x,
                "lstm_optuna_86_probability": risk_l,
            })
    frame = pd.DataFrame(records)
    dev = tmp_path / "development.csv"
    hold = tmp_path / "holdout.csv"
    frame.loc[frame.validation_year < 2025].to_csv(dev, index=False)
    # Holdout output order differs from frozen development; both must work.
    cols = ["date", "validation_year", "actual_downside_event",
            "xgboost_11_fixed_probability", "lstm_optuna_86_probability"]
    frame.loc[frame.validation_year >= 2025, cols].to_csv(hold, index=False)
    return dev, hold


def test_monitoring_fixed_bin_psi_zero_for_identical_distributions():
    a = np.linspace(0, 1, 100)
    bins, drift = _fixed_bin_drift(a, a)
    assert len(bins) == 7
    assert drift["score_population_stability_index"] == 0.0
    assert not drift["investigation_flag"]


def test_monitoring_synthetic_report(tmp_path):
    development, holdout = _make_predictions(tmp_path)
    yearly, bins, report = evaluate_monitoring(development, holdout)
    assert len(yearly) == 14
    assert len(bins) == 14
    assert report["models"]["xgboost_fixed"]["development"]["observations"] == 300
    assert report["models"]["xgboost_fixed"]["holdout"]["observations"] == 120
    assert report["holdout_annual_event_frequencies"]["2026"] > 0
    assert "NO promotion" in report["selection_status"]
    assert len(report["risk_register"]) >= 5
    out = tmp_path / "out"
    save_monitoring(development, holdout, out)
    assert all((out / item).exists() for item in MONITOR_FILES)
    with pytest.raises(MonitoringError, match="--overwrite"):
        save_monitoring(development, holdout, out)


def test_monitoring_rejects_wrong_dates(tmp_path):
    dev, hold = _make_predictions(tmp_path)
    df = pd.read_csv(hold)
    df.loc[0, "date"] = "2024-11-01"
    df.to_csv(hold, index=False)
    with pytest.raises(MonitoringError, match="Holdout|Validation year"):
        evaluate_monitoring(dev, hold)


def test_monitoring_rejects_duplicate_predictions(tmp_path):
    dev, hold = _make_predictions(tmp_path)
    df = pd.read_csv(dev)
    df.loc[1, "date"] = df.loc[0, "date"]
    df.to_csv(dev, index=False)
    with pytest.raises(MonitoringError, match="unique"):
        evaluate_monitoring(dev, hold)


def test_explainability_frozen_shap_dev_only(monkeypatch, tmp_path):
    rng = np.random.default_rng(42)
    # Synthetic pre-2025 history and labels only. No real holdout data accessed.
    dates = pd.bdate_range("2022-01-03", periods=350)
    daily_ret = np.where(np.arange(350) % 15 == 0, -0.025, 0.002)
    close = 1500 * np.cumprod(1 + daily_ret)
    prices = pd.DataFrame({"date": dates, "close": close})
    synthetic_features = pd.DataFrame({"date": dates})
    for i, name in enumerate(explainability.FEATURES):
        synthetic_features[name] = rng.normal(size=len(dates)) + (i+1)*0.1
    monkeypatch.setattr(explainability, "load_ohlc_csv", lambda _: prices)
    monkeypatch.setattr(
        explainability, "build_causal_history",
        lambda *args: (synthetic_features, {"verification": "synthetic test only"}),
    )
    nifty, external = tmp_path / "n.csv", tmp_path / "e.csv"
    nifty.write_text("synthetic")
    external.write_text("synthetic")
    features, report = explainability.explain_development_xgboost(
        nifty, external, max_examples=80
    )
    assert len(features) == len(explainability.FEATURES) == 11
    assert report["explanation_rows"] == 80
    assert report["pre_2025_training_events"] > 5
    assert report["latest_training_label_prediction_date"] < "2025-01-01"
    assert report["tree_shap_additivity_max_abs_error"] < 0.0001
    assert (features.mean_absolute_shap_log_odds >= 0).all()


def test_explainability_rejects_holdout_feature_history(monkeypatch, tmp_path):
    dates = pd.to_datetime(["2024-01-01", "2025-01-02"])
    history = pd.DataFrame({"date": dates})
    for name in explainability.FEATURES:
        history[name] = 0.5
    monkeypatch.setattr(
        explainability, "build_causal_history",
        lambda *args: (history, {}),
    )
    nifty, ext = tmp_path / "n.csv", tmp_path / "e.csv"
    nifty.touch()
    ext.touch()
    with pytest.raises(explainability.ExplanationError, match="holdout"):
        explainability.explain_development_xgboost(nifty, ext)


def test_phase6_cli_registered():
    parser = build_parser()
    monitor = parser.parse_args(["monitor-models"])
    explain = parser.parse_args(["explain-xgb", "--max-examples", "250"])
    assert monitor.command == "monitor-models"
    assert explain.command == "explain-xgb"
    assert explain.max_examples == 250
