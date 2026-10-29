"""Leakage and reproducibility checks for LSTM and VaR research components."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from risklens.models.sequence_lstm import (
    make_sequences,
    _check_reference,
    _fit_predict,
    build_causal_history,
    SequenceModelError,
    render_dashboard,
)
from risklens.models.market_risk_monitor import backtest_historical_var
from risklens.models.xgboost_walk_forward import MULTIFACTOR_FEATURE_COLUMNS


def history_frame(n=45):
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {
            "date": dates,
            "x": np.arange(n, dtype=float),
            "z": np.arange(n, dtype=float) + 100,
        }
    )


def labels_frame(dates):
    return pd.DataFrame(
        {"date": dates, "downside_next_session": [i % 2 for i in range(len(dates))]}
    )


def test_sequence_last_row_is_prediction_date_and_uses_only_history():
    h = history_frame()
    lab = labels_frame(h.date.iloc[10:15])
    x, y, dates = make_sequences(h, lab, ("x", "z"), lookback=5)
    assert x.shape == (5, 5, 2)
    assert x[0, :, 0].tolist() == [6, 7, 8, 9, 10]
    assert x[-1, -1, 0] == 14
    assert dates[-1] == h.date.iloc[14]


def test_future_changes_do_not_change_past_sequence():
    h = history_frame()
    lab = labels_frame(h.date.iloc[10:15])
    a, _, _ = make_sequences(h, lab, ("x",), lookback=5)
    h.loc[15:, "x"] = 1e9
    b, _, _ = make_sequences(h, lab, ("x",), lookback=5)
    np.testing.assert_array_equal(a, b)


def test_missing_window_excluded_not_forward_filled():
    h = history_frame()
    h.loc[10, "x"] = np.nan
    lab = labels_frame(h.date.iloc[10:18])
    x, _, dates = make_sequences(h, lab, ("x",), lookback=3)
    assert dates[0] == h.date.iloc[13]
    assert np.isfinite(x).all()


def test_reject_final_holdout_in_sequence_history():
    h = history_frame()
    h.loc[44, "date"] = pd.Timestamp("2025-01-02")
    with pytest.raises(SequenceModelError, match="holdout"):
        make_sequences(h, labels_frame(h.date.iloc[10:15]), ("x",), lookback=3)


def test_reject_duplicate_dates():
    h = history_frame()
    h.loc[5, "date"] = h.loc[4, "date"]
    with pytest.raises(SequenceModelError, match="unique"):
        make_sequences(h, labels_frame(h.date.iloc[10:15]), ("x",), lookback=3)


def test_labels_never_part_of_input_features():
    h = history_frame()
    with pytest.raises(SequenceModelError, match="Invalid feature"):
        make_sequences(
            h, labels_frame(h.date.iloc[10:15]), ("downside_next_session",), lookback=3
        )


def test_lstm_returns_probabilities_and_deterministic():
    rng = np.random.default_rng(7)
    tx = rng.normal(size=(96, 8, 3)).astype("float32")
    ty = np.array([0, 1] * 48)
    vx = rng.normal(size=(5, 8, 3)).astype("float32")
    p1, a1 = _fit_predict(tx, ty, vx, seed=101, max_epochs=2)
    p2, a2 = _fit_predict(tx, ty, vx, seed=101, max_epochs=2)
    assert len(p1) == 5 and ((p1 >= 0) & (p1 <= 1)).all()
    np.testing.assert_array_equal(p1, p2)
    assert a1 == a2 and a1["early_stop_inner_train_rows"] < 96


def test_lstm_does_not_see_validation_labels():
    rng = np.random.default_rng(5)
    tx = rng.normal(size=(85, 5, 3)).astype("float32")
    ty = np.array([0, 1] * 42 + [0])
    vx = rng.normal(size=(4, 5, 3)).astype("float32")
    p, _ = _fit_predict(tx, ty, vx, seed=12, max_epochs=2)
    assert p.shape == (4,)


def reference_file(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=5)
    df = pd.DataFrame(
        {
            "validation_year": [2024] * 5,
            "date": dates.strftime("%Y-%m-%d"),
            "actual_downside_event": [0, 1, 0, 1, 0],
            "baseline_probability": [0.1] * 5,
            "xgboost_9_features_probability": [0.1] * 5,
            "xgboost_11_features_probability": [0.1] * 5,
        }
    )
    # only a helper for rejection tests; valid reference requires all five years
    return df


def test_reference_rejects_holdout(tmp_path):
    df = reference_file(tmp_path)
    df.loc[4, "date"] = "2025-01-02"
    df.loc[4, "validation_year"] = 2025
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(SequenceModelError, match="holdout"):
        _check_reference(path)


def test_reference_rejects_invalid_probability(tmp_path):
    df = reference_file(tmp_path)
    df.loc[0, "baseline_probability"] = 1.3
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(SequenceModelError):
        _check_reference(path)


def test_dashboard_is_offline_without_scripts():
    report = {
        "n_validation_predictions": 1238,
        "n_events": 145,
        "pooled_scores": {
            k: {
                "brier_score": 0.1,
                "log_loss": 0.2,
                "roc_auc": 0.6,
                "average_precision": 0.2,
            }
            for k in [
                "constant_baseline",
                "xgboost_9",
                "xgboost_11",
                "lstm_9",
                "lstm_11",
            ]
        },
        "folds": [
            {
                "year": 2020,
                "validation_sequences": 250,
                "validation_events": 20,
                "year_scores": {"lstm_11": {"brier_score": 0.1, "roc_auc": 0.6}},
            }
        ],
    }
    out = render_dashboard(report)
    assert "RiskLens" in out and "2025+" in out and "<script" not in out


def test_feature_history_sealed_and_causal_if_future_prices_change():
    n = 300
    dates = pd.bdate_range("2024-01-01", periods=n)
    prices = pd.DataFrame({"date": dates, "close": 100 + np.arange(n) * 0.1})
    prices["open"] = prices.close * 0.999
    prices["high"] = prices.close * 1.002
    prices["low"] = prices.open * 0.998
    external = pd.DataFrame(
        {
            "date": dates[dates < pd.Timestamp("2025-01-01")],
            "india_vix_level": 18.0,
            "sp500_previous_return_1d": 0.002,
        }
    )

    def source():
        return prices.copy()

    def joined(*args):
        return external.copy(), {"original_nifty_ohlc_and_dates_match": True}

    with (
        patch(
            "risklens.models.sequence_lstm.load_ohlc_csv",
            side_effect=lambda *_: source(),
        ),
        patch("risklens.models.sequence_lstm._validate_integrated", side_effect=joined),
    ):
        a, _ = build_causal_history("fake.csv", "fake_ext.csv")
        prices.loc[prices.date >= pd.Timestamp("2025-01-01"), "close"] *= 100
        b, _ = build_causal_history("fake.csv", "fake_ext.csv")
    pd.testing.assert_frame_equal(a, b)
    assert (a.date < pd.Timestamp("2025-01-01")).all()


def test_invalid_lookback_input():
    h = history_frame()
    lab = labels_frame(h.date.iloc[10:15])
    with pytest.raises(SequenceModelError):
        make_sequences(h, lab, ("x",), lookback=1)


def test_var_uses_past_only_and_never_reads_2025():
    from risklens.models.market_risk_monitor import backtest_historical_var

    dates = pd.bdate_range("2018-01-01", "2025-02-28")
    close = 100 * np.exp(np.cumsum(0.001 * np.sin(np.arange(len(dates)) * 0.23)))
    prices = pd.DataFrame({"date": dates, "close": close})
    ref_dates = pd.bdate_range("2020-01-01", "2024-12-25", freq="B")
    rf = pd.DataFrame({"date": ref_dates, "validation_year": ref_dates.year})
    with (
        patch(
            "risklens.models.market_risk_monitor.load_ohlc_csv",
            side_effect=lambda *_: prices.copy(),
        ),
        patch(
            "risklens.models.market_risk_monitor._check_reference",
            side_effect=lambda *_: rf.copy(),
        ),
    ):
        a, ra = backtest_historical_var("fake.csv", "fake_ref.csv")
        prices.loc[prices.date >= pd.Timestamp("2025-01-01"), "close"] *= 2
        b, rb = backtest_historical_var("fake.csv", "fake_ref.csv")
    pd.testing.assert_frame_equal(a, b)
    assert ra["results"] == rb["results"]
    assert not (a.date >= pd.Timestamp("2025-01-01")).any()
    assert all(0 <= v["observed_exception_rate"] <= 1 for v in ra["results"].values())


def test_var_rejects_insufficient_rolling_history():
    from risklens.models.market_risk_monitor import backtest_historical_var

    dates = pd.bdate_range("2020-01-01", periods=100)
    prices = pd.DataFrame({"date": dates, "close": 100 + np.arange(100)})
    ref = pd.DataFrame({"date": dates[50:60], "validation_year": 2020})
    with (
        patch("risklens.models.market_risk_monitor.load_ohlc_csv", return_value=prices),
        patch("risklens.models.market_risk_monitor._check_reference", return_value=ref),
    ):
        with pytest.raises(SequenceModelError, match="Insufficient"):
            backtest_historical_var("fake.csv", "fake_ref.csv")
