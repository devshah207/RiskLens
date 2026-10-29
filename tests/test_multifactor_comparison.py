"""Phase 3.10: chronology, source integrity, fair samples and reproducibility."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from risklens.models.multifactor_comparison import (
    MultiFactorComparisonError,
    _validate_integrated,
    compare_multifactor_features,
    save_multifactor_comparison,
    PREDICTIONS_FILENAME,
    REPORT_FILENAME,
)
from risklens.models.xgboost_walk_forward import (
    FIXED_XGB_PARAMS,
    FULL_FEATURE_COLUMNS,
    EXTERNAL_FEATURE_COLUMNS,
    MULTIFACTOR_FEATURE_COLUMNS,
    _fit_predict_fixed_xgb,
)


def make_source(tmp_path: Path, *, suffix: str = "", modify_holdout=False):
    """Synthetic ONLY, deterministic, complete training folds 2016-2024."""
    dates = pd.date_range("2016-01-01", "2025-03-31", freq="B")
    i = np.arange(len(dates))
    returns = np.where(i % 23 < 3, -0.015, 0.0025 + 0.0013 * np.sin(i * 0.27))
    close = 10000 * np.cumprod(1 + returns)
    open_ = close * (1 + 0.001 * np.sin(i * 0.4))
    high = np.maximum(close, open_) * 1.005
    low = np.minimum(close, open_) * 0.995
    prices = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )
    if modify_holdout:
        mask = dates >= pd.Timestamp("2025-01-01")
        prices.loc[mask, ["open", "high", "low", "close"]] *= 1.3
    source = tmp_path / f"nifty{suffix}.csv"
    prices.to_csv(source, index=False)
    dev = prices[dates < pd.Timestamp("2025-01-01")].copy().reset_index(drop=True)
    dev_date = pd.to_datetime(dev["date"])
    n = len(dev)
    dev["vix_source_date"] = dev_date.shift(1).dt.strftime("%Y-%m-%d")
    dev["india_vix_level"] = 13 + 4 * np.sin(np.arange(n) * 0.04) ** 2
    dev["vix_age_calendar_days"] = (dev_date - dev_date.shift(1)).dt.days
    dev["sp500_source_date"] = dev_date.shift(1).dt.strftime("%Y-%m-%d")
    dev["sp500_previous_close"] = 2000.0 + np.arange(n) * 0.2
    dev["sp500_previous_return_1d"] = np.sin(np.arange(n) * 0.19) * 0.008
    dev["sp500_age_calendar_days"] = (dev_date - dev_date.shift(1)).dt.days
    dev.loc[
        0,
        [
            "vix_source_date",
            "india_vix_level",
            "vix_age_calendar_days",
            "sp500_source_date",
            "sp500_previous_close",
            "sp500_previous_return_1d",
            "sp500_age_calendar_days",
        ],
    ] = np.nan
    dev["external_features_ready"] = (
        dev[["india_vix_level", "sp500_previous_return_1d"]].notna().all(axis=1)
    )
    external = tmp_path / f"external{suffix}.csv"
    dev.to_csv(external, index=False)
    return source, external


def modify_merged(tmp_path, update, *, error):
    src, merged = make_source(tmp_path)
    df = pd.read_csv(merged)
    update(df)
    df.to_csv(merged, index=False)
    with pytest.raises(MultiFactorComparisonError, match=error):
        _validate_integrated(src, merged)


def test_frozen_feature_contract():
    assert len(FULL_FEATURE_COLUMNS) == 9
    assert EXTERNAL_FEATURE_COLUMNS == ("india_vix_level", "sp500_previous_return_1d")
    assert (
        MULTIFACTOR_FEATURE_COLUMNS == FULL_FEATURE_COLUMNS + EXTERNAL_FEATURE_COLUMNS
    )
    assert FIXED_XGB_PARAMS["n_estimators"] == 80
    assert FIXED_XGB_PARAMS["max_depth"] == 2
    assert FIXED_XGB_PARAMS["n_jobs"] == 1


def test_valid_join_and_early_missing_row(tmp_path):
    src, ext = make_source(tmp_path)
    merged, audit = _validate_integrated(src, ext)
    assert len(merged) == audit["integrated_development_rows"]
    assert audit["rows_with_missing_external_feature"] == 1
    assert not merged.iloc[0]["external_features_ready"]
    assert merged.iloc[1]["external_features_ready"]
    assert merged["date"].max() < pd.Timestamp("2025-01-01")


def test_missing_nifty_source_rejected(tmp_path):
    with pytest.raises(MultiFactorComparisonError, match="source not found"):
        _validate_integrated(tmp_path / "no.csv", tmp_path / "also.csv")


def test_missing_external_rejected(tmp_path):
    src, _ = make_source(tmp_path)
    with pytest.raises(MultiFactorComparisonError, match="integrated CSV not found"):
        _validate_integrated(src, tmp_path / "missing.csv")


def test_disallow_final_holdout_rows(tmp_path):
    def change(df):
        df.loc[0, "date"] = "2025-01-02"

    modify_merged(tmp_path, change, error="2025\\+ final-holdout")


def test_disallow_same_day_sp500_close(tmp_path):
    def change(df):
        df.loc[60, "sp500_source_date"] = df.loc[60, "date"]
        df.loc[60, "sp500_age_calendar_days"] = 0

    modify_merged(tmp_path, change, error="S&P 500: future/same-date")


def test_disallow_future_vix_source(tmp_path):
    def change(df):
        date = pd.Timestamp(df.loc[60, "date"])
        df.loc[60, "vix_source_date"] = (date + pd.Timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )
        df.loc[60, "vix_age_calendar_days"] = -1

    modify_merged(tmp_path, change, error="India VIX: future/same-date")


def test_disallow_stale_vix_observation(tmp_path):
    def change(df):
        date = pd.Timestamp(df.loc[60, "date"])
        df.loc[60, "vix_source_date"] = (date - pd.Timedelta(days=9)).strftime(
            "%Y-%m-%d"
        )
        df.loc[60, "vix_age_calendar_days"] = 9

    modify_merged(tmp_path, change, error="stale")


def test_disallow_vix_age_misstatement(tmp_path):
    modify_merged(
        tmp_path,
        lambda d: d.__setitem__(
            "vix_age_calendar_days", d["vix_age_calendar_days"].fillna(1) + 1
        ),
        error="inconsistent source observation age",
    )


def test_disallow_modified_nifty_close(tmp_path):
    def change(df):
        df.loc[80, "close"] *= 1.2

    modify_merged(tmp_path, change, error="Integrated close differs")


def test_disallow_drop_nifty_session(tmp_path):
    src, ext = make_source(tmp_path)
    df = pd.read_csv(ext)
    df.drop(index=55).to_csv(ext, index=False)
    with pytest.raises(MultiFactorComparisonError, match="exactly all"):
        _validate_integrated(src, ext)


def test_disallow_duplicate_nifty_date(tmp_path):
    def change(df):
        df.loc[81, "date"] = df.loc[80, "date"]

    modify_merged(tmp_path, change, error="duplicate or out of order")


def test_disallow_bad_ready_flag(tmp_path):
    def change(df):
        df["external_features_ready"] = df["external_features_ready"].astype(str)
        df.loc[10, "external_features_ready"] = "False"

    modify_merged(tmp_path, change, error="flags inconsistent")


def test_disallow_unprovenanced_external_value(tmp_path):
    def change(df):
        df.loc[10, "vix_source_date"] = np.nan
        df.loc[10, "vix_age_calendar_days"] = np.nan

    modify_merged(tmp_path, change, error="no provenance date")


def test_disallow_nonfinite_vix(tmp_path):
    def change(df):
        df["india_vix_level"] = df["india_vix_level"].astype(object)
        df.loc[70, "india_vix_level"] = "inf"

    modify_merged(tmp_path, change, error="Nonfinite India VIX")


def test_disallow_non_numeric_sp500(tmp_path):
    def change(df):
        df["sp500_previous_return_1d"] = df["sp500_previous_return_1d"].astype(object)
        df.loc[70, "sp500_previous_return_1d"] = "incorrect"

    modify_merged(tmp_path, change, error="nonnumeric S&P 500")


def test_disallow_nonsensical_sp500_return(tmp_path):
    def change(df):
        df.loc[70, "sp500_previous_return_1d"] = -1.5

    modify_merged(tmp_path, change, error="exceed -100%")


def test_disallow_sp500_close_with_no_date(tmp_path):
    def change(df):
        df.loc[70, "sp500_source_date"] = np.nan
        df.loc[70, "sp500_age_calendar_days"] = np.nan

    modify_merged(tmp_path, change, error="no provenance date")


def test_full_comparison_chronology_and_parity(tmp_path):
    src, ext = make_source(tmp_path)
    pred, report = compare_multifactor_features(src, ext)
    assert pred["date"].is_unique
    assert set(pred["validation_year"]) == {2020, 2021, 2022, 2023, 2024}
    assert pred["date"].max() < pd.Timestamp("2025-01-01")
    assert pred["actual_downside_event"].isin((0, 1)).all()
    assert report["total_validation_rows"] == len(pred)
    assert report["total_validation_events"] == int(pred["actual_downside_event"].sum())
    assert set(report["pooled_probability_scores"]) == {
        "constant_baseline",
        "xgboost_9_features",
        "xgboost_11_features",
    }
    for fold in report["yearly_folds"]:
        assert fold["purged_cross_boundary_training_labels"] == 1
        assert (
            fold["last_training_prediction_date"]
            < fold["first_validation_prediction_date"]
        )
        assert (
            fold["fit_audit"]["xgboost_9_features"]["train_rows"]
            == fold["fit_audit"]["xgboost_11_features"]["train_rows"]
        )
        assert (
            fold["fit_audit"]["xgboost_9_features"]["validation_rows"]
            == fold["fit_audit"]["xgboost_11_features"]["validation_rows"]
        )
        assert fold["fit_audit"]["xgboost_9_features"]["trees_fitted"] == 80
        assert fold["fit_audit"]["xgboost_11_features"]["trees_fitted"] == 80
    for col in (
        "baseline_probability",
        "xgboost_9_features_probability",
        "xgboost_11_features_probability",
    ):
        assert pred[col].between(0, 1).all()


def test_holdout_changes_have_zero_effect(tmp_path):
    src, ext = make_source(tmp_path, suffix="a")
    changed, _ = make_source(tmp_path, suffix="b", modify_holdout=True)
    left, _ = compare_multifactor_features(src, ext)
    right, _ = compare_multifactor_features(changed, ext)
    pd.testing.assert_frame_equal(left, right)


def test_later_external_prices_do_not_affect_earlier_predictions(tmp_path):
    src, ext = make_source(tmp_path, suffix="a")
    df = pd.read_csv(ext)
    df.loc[df["date"] >= "2024-01-01", "india_vix_level"] *= 2
    changed = tmp_path / "changed.csv"
    df.to_csv(changed, index=False)
    left, _ = compare_multifactor_features(src, ext)
    right, _ = compare_multifactor_features(src, changed)
    pd.testing.assert_frame_equal(
        left.loc[left["validation_year"] < 2024],
        right.loc[right["validation_year"] < 2024],
    )


def test_file_export_and_overwrite_protection(tmp_path):
    src, ext = make_source(tmp_path)
    out = tmp_path / "out"
    report = save_multifactor_comparison(src, ext, out)
    assert json.loads((out / REPORT_FILENAME).read_text()) == report
    preds = pd.read_csv(out / PREDICTIONS_FILENAME)
    assert len(preds) == report["total_validation_rows"]
    assert preds["date"].max() < "2025-01-01"
    with pytest.raises(MultiFactorComparisonError, match="already exist"):
        save_multifactor_comparison(src, ext, out)
