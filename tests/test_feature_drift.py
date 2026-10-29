"""Phase 6 feature drift: original scored dates and development-only bins."""
import numpy as np
import pandas as pd
import pytest

from risklens.cli import build_parser
from risklens.evaluation import feature_drift


def test_identical_feature_distribution_yields_zero_psi():
    sample = np.linspace(-3, 3, 101)
    drift, bins = feature_drift._feature_psi(sample, sample)
    assert drift == pytest.approx(0.0, abs=1e-12)
    assert 2 <= bins <= 5


def test_constant_development_feature_handles_collapsed_bins():
    baseline = np.ones(100)
    future = np.full(40, 1.5)
    value, bins = feature_drift._feature_psi(baseline, future)
    assert bins == 2
    assert np.isfinite(value)
    assert value > 0


def test_feature_drift_respects_saved_holdout_prediction_dates(tmp_path, monkeypatch):
    rng = np.random.default_rng(2026)
    dev_dates = pd.bdate_range("2023-01-02", periods=180)
    future_dates = pd.to_datetime(["2025-03-04", "2025-03-05", "2026-04-07", "2026-04-08"])
    history = pd.DataFrame({"date": dev_dates})
    candidate = pd.DataFrame({"date": future_dates})
    for n, col in enumerate(feature_drift.FEATURES):
        history[col] = rng.normal(loc=n, scale=1, size=len(history))
        if col not in feature_drift.FEATURES[-2:]:
            candidate[col] = rng.normal(loc=n+0.2, scale=1, size=len(candidate))
    for path in ("nifty", "devext", "vix", "sp500"):
        (tmp_path / f"{path}.csv").write_text("synthetic source")
    prediction = pd.DataFrame({
        "date": future_dates.strftime("%Y-%m-%d"),
        "validation_year": future_dates.year,
        "actual_downside_event": [0, 1, 0, 1],
        "xgboost_11_fixed_probability": [0.05, 0.15, 0.08, 0.20],
        "lstm_optuna_86_probability": [0.04, 0.21, 0.10, 0.26],
    })
    predictions_path = tmp_path / "pred.csv"
    prediction.to_csv(predictions_path, index=False)

    monkeypatch.setattr(
        feature_drift, "build_causal_history",
        lambda *args: (history, {"audit": "synthetic"}),
    )
    monkeypatch.setattr(
        feature_drift, "build_development_features",
        lambda *args, **kwargs: (pd.DataFrame(), candidate, {}),
    )
    monkeypatch.setattr(
        feature_drift, "_read_holdout_vix",
        lambda *args: (pd.DataFrame(), {"audit": "vix synthetic"}),
    )
    monkeypatch.setattr(
        feature_drift, "_read_holdout_sp500",
        lambda *args: (pd.DataFrame(), {"audit": "sp synthetic"}),
    )

    def attach(frame, _vix, _sp):
        updated = frame.copy()
        updated["india_vix_level"] = 15.0
        updated["sp500_previous_return_1d"] = 0.002
        return updated, {"audit": "alignment synthetic"}
    monkeypatch.setattr(feature_drift, "_attach_holdout_external", attach)
    kwargs = {
        "nifty_csv": tmp_path / "nifty.csv",
        "development_external_csv": tmp_path / "devext.csv",
        "vix_csv": tmp_path / "vix.csv",
        "sp500_csv": tmp_path / "sp500.csv",
        "holdout_predictions_csv": predictions_path,
    }
    table, report = feature_drift.evaluate_feature_drift(**kwargs)
    assert len(table) == 11
    assert report["holdout_dates_from_single_existing_prediction_file"] == 4
    assert report["development_pre_2025_feature_rows"] == 180
    assert all(table.num_development_derived_bins >= 2)
    assert "holdout_vix" in report["source_sha256"]
    out = tmp_path / "results"
    feature_drift.save_feature_drift(**kwargs, output_dir=out)
    assert all((out / name).exists() for name in feature_drift.OUTPUTS)


def test_phase6_feature_drift_cli_requires_sources():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["monitor-features"])
    args = parser.parse_args([
        "monitor-features",
        "--vix", "vix.csv", "--sp500", "sp.csv",
    ])
    assert args.command == "monitor-features"
    assert args.vix == "vix.csv"
