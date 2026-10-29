"""Phase 3.5 tests. All input market prices are deliberately synthetic."""
import json

import numpy as np
import pandas as pd
import pytest

from risklens.features.dataset import EXPANDED_FEATURE_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN
from risklens.models.xgboost_walk_forward import (
    FIXED_XGB_PARAMS, XGBComparisonError, _fit_predict_fixed_xgb,
    compare_xgboost_walk_forward, save_xgboost_walk_forward,
)


def simple_frames():
    x = np.arange(120, dtype=float)
    y = (np.arange(120) % 9 == 0).astype(int)
    data = pd.DataFrame({
        "return_1d": 0.004*np.sin(x),
        "return_5d": 0.016*np.sin(x/3),
        "volatility_20d": 0.10 + .04*(x % 7)/7,
        "intraday_range": .012 + .004*(x % 5)/5,
        "intraday_return": .003*np.cos(x),
        "opening_gap": .001*np.sin(x/2),
        TARGET_COLUMN: y,
    })
    return data.iloc[:90].copy(), data.iloc[90:].copy()


def make_synthetic_ohlc(tmp_path, name="synthetic.csv", alter_holdout=False):
    days = pd.bdate_range("2016-01-04", "2025-01-10")
    indexes = np.arange(len(days))
    returns = np.where(indexes % 11 == 0, -.022, .003 + .002*np.sin(indexes/4))
    closes = 100 * np.cumprod(1+returns)
    if alter_holdout:
        closes[days >= "2025-01-01"] *= 4
    previous = np.r_[100., closes[:-1]]
    opens = previous*(1+.002*np.sin(indexes/2))
    highs = np.maximum(opens, closes) * 1.003
    lows = np.minimum(opens, closes) * .997
    path = tmp_path / name
    pd.DataFrame({"date":days,"open":opens,"high":highs,"low":lows,"close":closes}).to_csv(path,index=False)
    return path


def test_fixed_params_are_conservative_and_have_no_search():
    assert FIXED_XGB_PARAMS["max_depth"] == 2
    assert FIXED_XGB_PARAMS["n_estimators"] == 80
    assert FIXED_XGB_PARAMS["random_state"] == 42
    assert FIXED_XGB_PARAMS["scale_pos_weight"] == 1.0
    assert FIXED_XGB_PARAMS["tree_method"] == "hist"


@pytest.mark.parametrize("features", [FEATURE_COLUMNS, EXPANDED_FEATURE_COLUMNS])
def test_xgb_produces_finite_probabilities_and_exact_tree_count(features):
    train, valid = simple_frames()
    probs, audit = _fit_predict_fixed_xgb(train, valid, features)
    assert probs.shape == (30,)
    assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()
    assert audit["trees_fitted"] == 80
    assert audit["train_rows"] == 90
    assert audit["train_events"] == int(train[TARGET_COLUMN].sum())
    assert set(audit["feature_importance_gain"]) == set(features)


def test_validation_outcomes_never_used_for_training():
    train, valid = simple_frames()
    original, _ = _fit_predict_fixed_xgb(train, valid, FEATURE_COLUMNS)
    valid[TARGET_COLUMN] = 1 - valid[TARGET_COLUMN]
    modified, _ = _fit_predict_fixed_xgb(train, valid, FEATURE_COLUMNS)
    np.testing.assert_array_equal(original, modified)


def test_invalid_input_columns_are_rejected():
    train, val = simple_frames()
    with pytest.raises(XGBComparisonError, match="Only the original"):
        _fit_predict_fixed_xgb(train, val, ("tomorrow_close",))


def test_one_class_training_rejected():
    train, val = simple_frames()
    train[TARGET_COLUMN] = 0
    with pytest.raises(XGBComparisonError, match="both label classes"):
        _fit_predict_fixed_xgb(train, val, FEATURE_COLUMNS)


@pytest.mark.parametrize("which", ["train", "validation"])
def test_nonfinite_features_rejected(which):
    train, val = simple_frames()
    table = train if which == "train" else val
    table.iloc[0, table.columns.get_loc("return_1d")] = np.inf
    with pytest.raises(XGBComparisonError, match="finite"):
        _fit_predict_fixed_xgb(train, val, FEATURE_COLUMNS)


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("xgb_realistic_synthetic")
    source = make_synthetic_ohlc(directory)
    predictions, report = compare_xgboost_walk_forward(source)
    return source, predictions, report, directory


def test_full_walk_forward_aligned_and_holdout_excluded(synthetic_run):
    _, predictions, report, _ = synthetic_run
    assert report["validation_years"] == [2020,2021,2022,2023,2024]
    assert predictions["date"].max() < pd.Timestamp("2025-01-01")
    assert predictions["date"].is_monotonic_increasing
    assert not predictions["date"].duplicated().any()
    assert report["total_validation_rows"] == len(predictions)
    assert report["total_validation_events"] == int(predictions["actual_downside_event"].sum())
    assert len(report["per_fold_scores"]) == 5
    for fold in report["per_fold_scores"]:
        assert fold["training_before"] == f"{fold['year']}-01-01"
        assert fold["xgboost_fit_audit"]["xgboost_3_features"]["trees_fitted"] == 80
        assert fold["xgboost_fit_audit"]["xgboost_6_features"]["trees_fitted"] == 80
    assert set(report["pooled_probability_scores"]) == {
        "constant_baseline", "logistic_3_features", "logistic_6_features",
        "xgboost_3_features", "xgboost_6_features",
    }


def test_altered_2025_prices_cannot_change_development_predictions(synthetic_run):
    _, before, report_before, directory = synthetic_run
    alternate = make_synthetic_ohlc(directory, "modified_holdout.csv", alter_holdout=True)
    after, report_after = compare_xgboost_walk_forward(alternate)
    pd.testing.assert_frame_equal(before, after)
    assert report_before["pooled_probability_scores"] == report_after["pooled_probability_scores"]
    assert report_before["per_fold_scores"] == report_after["per_fold_scores"]


def test_save_and_overwrite_protection(synthetic_run):
    source, _, report, directory = synthetic_run
    out = directory / "outputs"
    saved = save_xgboost_walk_forward(source, out)
    assert saved["pooled_probability_scores"] == report["pooled_probability_scores"]
    loaded = json.loads((out / "xgboost_walk_forward_report.json").read_text())
    assert loaded == saved
    preds = pd.read_csv(out / "xgboost_walk_forward_predictions.csv")
    assert preds["date"].max() < "2025-01-01"
    with pytest.raises(XGBComparisonError, match="already exists"):
        save_xgboost_walk_forward(source, out)


def test_missing_csv_rejected(tmp_path):
    with pytest.raises(XGBComparisonError, match="not found"):
        compare_xgboost_walk_forward(tmp_path / "not_here.csv")

