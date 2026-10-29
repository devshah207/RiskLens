"""Synthetic tests for RSI/EMA/10d volatility and fair time-aware XGBoost evaluation."""

import json
import numpy as np
import pandas as pd
import pytest

from risklens.features.dataset import (
    FEATURE_COLUMNS,
    EXPANDED_FEATURE_COLUMNS,
    FULL_FEATURE_COLUMNS,
    FeatureDatasetError,
    TARGET_COLUMN,
    _wilder_rsi_14,
    build_development_features,
)
from risklens.models.technical_comparison import (
    PREDICTIONS_FILENAME,
    REPORT_FILENAME,
    TechnicalComparisonError,
    compare_technical_features,
    save_technical_comparison,
)
from risklens.models.xgboost_walk_forward import _fit_predict_fixed_xgb


def make_ohlc(tmp_path, *, name="prices.csv", shock_future=False, shock_2024=False):
    days = pd.bdate_range("2016-01-04", "2025-01-10")
    ix = np.arange(len(days), dtype=float)
    returns = np.where(
        (ix.astype(int) % 13) == 0, -0.022, 0.002 + 0.005 * np.sin(ix / 8)
    )
    close = 100 * np.cumprod(1 + returns)
    if shock_2024:
        # Change only a late 2024 value; 2023 and earlier features stay the same.
        close[days >= "2024-09-02"] *= 1.1
    if shock_future:
        close[days >= "2025-01-01"] *= 20.0
    before = np.r_[100.0, close[:-1]]
    op = before * (1 + 0.002 * np.cos(ix / 2))
    frame = pd.DataFrame(
        {
            "date": days,
            "open": op,
            "high": np.maximum(op, close) * 1.005,
            "low": np.minimum(op, close) * 0.995,
            "close": close,
        }
    )
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def test_wilder_rsi_rising_prices_reach_100():
    rsi = _wilder_rsi_14(pd.Series(np.arange(100.0, 125.0)))
    assert rsi.iloc[:14].isna().all()
    assert rsi.iloc[14:].eq(100).all()


def test_wilder_rsi_falling_prices_reach_zero():
    rsi = _wilder_rsi_14(pd.Series(np.arange(125.0, 100.0, -1)))
    assert rsi.iloc[14:].eq(0).all()


def test_wilder_rsi_flat_prices_neutral_and_short_input():
    assert _wilder_rsi_14(pd.Series([100.0] * 15)).iloc[14] == 50
    assert _wilder_rsi_14(pd.Series([100.0] * 14)).isna().all()


def test_wilder_rsi_recursive_update_matches_manual_formula():
    prices = pd.Series(
        [
            100.0,
            102.0,
            99.0,
            103.0,
            100.0,
            105.0,
            103.0,
            100.0,
            101.0,
            99.0,
            103.0,
            98.0,
            99.0,
            97.0,
            102.0,
            104.0,
        ]
    )
    delta = np.diff(prices.to_numpy())
    g = np.maximum(delta, 0)
    losses = np.maximum(-delta, 0)
    avg_g, avg_l = g[:14].mean(), losses[:14].mean()
    expected_first = 100 - 100 / (1 + avg_g / avg_l)
    avg_g = (13 * avg_g + g[14]) / 14
    avg_l = (13 * avg_l + losses[14]) / 14
    expected_second = 100 - 100 / (1 + avg_g / avg_l)
    result = _wilder_rsi_14(prices)
    assert result.iloc[14] == pytest.approx(expected_first)
    assert result.iloc[15] == pytest.approx(expected_second)


def test_ema_and_short_vol_formulas_at_first_eligible_date(tmp_path):
    source = make_ohlc(tmp_path)
    prices = pd.read_csv(source)
    train, valid, report = build_development_features(
        source, include_ohlc_features=True, include_technical_features=True
    )
    assert train.iloc[0]["date"] == pd.Timestamp(prices.iloc[49]["date"])
    assert len(FULL_FEATURE_COLUMNS) == 9
    p = prices["close"]
    ema = p.ewm(span=50, adjust=False, min_periods=50).mean()
    ret = p.pct_change(fill_method=None)
    expected_vol = ret.iloc[40:50].std(ddof=1) * np.sqrt(252)
    assert train.iloc[0]["ema_50_distance"] == pytest.approx(
        p.iloc[49] / ema.iloc[49] - 1
    )
    assert train.iloc[0]["volatility_10d"] == pytest.approx(expected_vol)
    assert train["rsi_14"].between(0, 100).all()
    assert valid[list(FULL_FEATURE_COLUMNS)].notna().all().all()
    assert report["technical_feature_warmup_price_rows"] == 49


def test_default_old_feature_path_is_unchanged(tmp_path):
    source = make_ohlc(tmp_path)
    a, b, _ = build_development_features(source)
    c, d, _ = build_development_features(source, include_ohlc_features=True)
    assert len(a) == len(c)
    assert a["date"].equals(c["date"]) and b["date"].equals(d["date"])
    assert list(a.columns) == ["date", *FEATURE_COLUMNS, TARGET_COLUMN]
    assert list(c.columns) == ["date", *EXPANDED_FEATURE_COLUMNS, TARGET_COLUMN]
    assert a[TARGET_COLUMN].equals(c[TARGET_COLUMN])


def test_technical_requires_ohlc_mode(tmp_path):
    with pytest.raises(FeatureDatasetError, match="require include_ohlc"):
        build_development_features(make_ohlc(tmp_path), include_technical_features=True)


def test_future_prices_do_not_change_features_labels_or_results(tmp_path):
    orig = make_ohlc(tmp_path, name="first.csv")
    changed = make_ohlc(tmp_path, name="second.csv", shock_future=True)
    a, b, _ = build_development_features(
        orig, include_ohlc_features=True, include_technical_features=True
    )
    x, y, _ = build_development_features(
        changed, include_ohlc_features=True, include_technical_features=True
    )
    pd.testing.assert_frame_equal(a, x)
    pd.testing.assert_frame_equal(b, y)


def test_later_prices_cannot_affect_earlier_technical_features(tmp_path):
    orig = make_ohlc(tmp_path, name="first.csv")
    changed = make_ohlc(tmp_path, name="second.csv", shock_2024=True)
    _, a, _ = build_development_features(
        orig, include_ohlc_features=True, include_technical_features=True
    )
    _, b, _ = build_development_features(
        changed, include_ohlc_features=True, include_technical_features=True
    )
    mask = a["date"] < pd.Timestamp("2024-09-02")
    pd.testing.assert_frame_equal(a.loc[mask], b.loc[mask])


def test_fixed_xgb_supports_predeclared_nine_features(tmp_path):
    tr, va, _ = build_development_features(
        make_ohlc(tmp_path), include_ohlc_features=True, include_technical_features=True
    )
    p, audit = _fit_predict_fixed_xgb(tr, va, FULL_FEATURE_COLUMNS)
    assert len(p) == len(va)
    assert np.isfinite(p).all()
    assert set(audit["feature_importance_gain"]) == set(FULL_FEATURE_COLUMNS)
    assert audit["trees_fitted"] == 80


def test_comparison_dates_labels_and_year_boundaries(tmp_path):
    pred, report = compare_technical_features(make_ohlc(tmp_path))
    assert len(report["yearly_folds"]) == 5
    assert set(pred["validation_year"]) == {2020, 2021, 2022, 2023, 2024}
    assert pred["date"].max() < pd.Timestamp("2025-01-01")
    assert pred["date"].is_unique
    assert pred["actual_downside_event"].isin([0, 1]).all()
    for fold in report["yearly_folds"]:
        assert fold["training_last_date"] < fold["validation_first_date"]
        assert fold["boundary_labels_purged"] == 1
        assert fold["feature_warmup_price_rows"] == 49
        assert (
            fold["fit_audit"]["xgboost_6_features"]["train_rows"]
            == fold["fit_audit"]["xgboost_9_features"]["train_rows"]
        )
        assert (
            fold["fit_audit"]["xgboost_6_features"]["validation_rows"]
            == fold["fit_audit"]["xgboost_9_features"]["validation_rows"]
        )
    assert report["same_training_and_validation_rows_both_variants"] is True


def test_prediction_values_unchanged_if_holdout_is_modified(tmp_path):
    a, _ = compare_technical_features(make_ohlc(tmp_path, name="a.csv"))
    b, _ = compare_technical_features(
        make_ohlc(tmp_path, name="b.csv", shock_future=True)
    )
    pd.testing.assert_frame_equal(a, b)


def test_saves_reports_and_refuses_overwrite(tmp_path):
    source = make_ohlc(tmp_path)
    out = tmp_path / "outputs"
    report = save_technical_comparison(source, out)
    disk = json.loads((out / REPORT_FILENAME).read_text())
    assert disk == report
    pred = pd.read_csv(out / PREDICTIONS_FILENAME)
    assert len(pred) == report["total_validation_rows"]
    assert pred["date"].max() < "2025-01-01"
    with pytest.raises(TechnicalComparisonError, match="exist"):
        save_technical_comparison(source, out)


def test_invalid_missing_source_rejected(tmp_path):
    with pytest.raises(TechnicalComparisonError, match="not found"):
        compare_technical_features(tmp_path / "missing.csv")
