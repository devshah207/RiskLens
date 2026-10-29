"""Synthetic tests for the controlled OHLC-feature experiment; never use final holdout outcomes."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from risklens.features.dataset import (
    FEATURE_COLUMNS, EXPANDED_FEATURE_COLUMNS, TARGET_COLUMN,
    build_development_features, FeatureDatasetError,
)
from risklens.models.baselines import ModelInputError, compare_models
from risklens.models.ohlc_comparison import (
    OHLCComparisonError, compare_ohlc_features, save_ohlc_feature_comparison,
)
from risklens.models.walk_forward import walk_forward_validation


def synthetic_prices(tmp_path, *, holdout_multiplier=1.0, filename="prices.csv") -> Path:
    days = pd.bdate_range("2018-01-02", "2025-01-15")
    n = len(days)
    idx = np.arange(n)
    returns = 0.0003 + 0.006 * np.sin(idx / 8.0)
    returns[idx % 19 == 0] = -0.021
    closes = 100 * np.cumprod(1 + returns)
    opens = closes * (1 + 0.002 * np.sin(idx / 3.0))
    highs = np.maximum(opens, closes) * (1 + 0.002 + 0.002 * (idx % 5))
    lows = np.minimum(opens, closes) * (1 - 0.002 - 0.001 * (idx % 4))
    future = days >= pd.Timestamp("2025-01-01")
    for series in (closes, opens, highs, lows):
        series[future] *= holdout_multiplier
    path = tmp_path / filename
    pd.DataFrame({"date": days, "open": opens, "high": highs, "low": lows, "close": closes}).to_csv(path, index=False)
    return path


def test_original_output_stays_three_features(tmp_path):
    data = synthetic_prices(tmp_path)
    train, valid, report = build_development_features(data)
    assert list(train.columns) == ["date", *FEATURE_COLUMNS, TARGET_COLUMN]
    assert list(valid.columns) == list(train.columns)
    assert report["feature_columns"] == list(FEATURE_COLUMNS)


def test_expanded_columns_and_exact_ohlc_formulas(tmp_path):
    path = synthetic_prices(tmp_path)
    train, valid, audit = build_development_features(path, include_ohlc_features=True)
    assert list(train.columns) == ["date", *EXPANDED_FEATURE_COLUMNS, TARGET_COLUMN]
    assert list(valid.columns) == list(train.columns)
    first = train.iloc[0]
    original = pd.read_csv(path).set_index("date")
    current = original.loc[first["date"].strftime("%Y-%m-%d")]
    prior = original.loc[:first["date"].strftime("%Y-%m-%d")].iloc[-2]
    assert first["intraday_range"] == pytest.approx((current.high - current.low) / current.open)
    assert first["intraday_return"] == pytest.approx(current.close / current.open - 1)
    assert first["opening_gap"] == pytest.approx(current.open / prior.close - 1)
    assert set(audit["feature_definitions"]) == set(EXPANDED_FEATURE_COLUMNS)


def test_features_are_backward_looking(tmp_path):
    source = synthetic_prices(tmp_path)
    changed = tmp_path / "later_change.csv"
    data = pd.read_csv(source)
    row = data.index[data["date"] == "2024-06-20"][0]
    data.loc[row, ["open", "high", "low", "close"]] *= 1.5
    data.to_csv(changed, index=False)
    _, before, _ = build_development_features(source, include_ohlc_features=True)
    _, after, _ = build_development_features(changed, include_ohlc_features=True)
    pd.testing.assert_frame_equal(
        before.loc[before["date"] < "2024-06-20", ["date", *EXPANDED_FEATURE_COLUMNS]],
        after.loc[after["date"] < "2024-06-20", ["date", *EXPANDED_FEATURE_COLUMNS]],
    )


def test_expanded_does_not_change_labels_or_row_dates(tmp_path):
    path = synthetic_prices(tmp_path)
    old_train, old_valid, _ = build_development_features(path)
    new_train, new_valid, _ = build_development_features(path, include_ohlc_features=True)
    for old, new in ((old_train, new_train), (old_valid, new_valid)):
        assert old["date"].equals(new["date"])
        assert old[TARGET_COLUMN].equals(new[TARGET_COLUMN])
        pd.testing.assert_frame_equal(old[list(FEATURE_COLUMNS)], new[list(FEATURE_COLUMNS)])


def test_model_reuses_same_settings_and_training_only_scaler(tmp_path):
    path = synthetic_prices(tmp_path)
    train, valid, _ = build_development_features(path, include_ohlc_features=True)
    train_file, valid_file = tmp_path / "train.csv", tmp_path / "valid.csv"
    train.to_csv(train_file, index=False)
    valid.to_csv(valid_file, index=False)
    predictions, report = compare_models(train_file, valid_file, feature_columns=EXPANDED_FEATURE_COLUMNS)
    assert predictions["logistic_probability"].between(0, 1).all()
    means = report["models"]["logistic_regression"]["scaler_mean"]
    for col in EXPANDED_FEATURE_COLUMNS:
        assert means[col] == pytest.approx(train[col].mean())
    assert report["features"] == list(EXPANDED_FEATURE_COLUMNS)
    assert report["models"]["logistic_regression"]["hyperparameters"]["C"] == 1.0


def test_unknown_feature_configuration_and_extra_leaked_column_rejected(tmp_path):
    path = synthetic_prices(tmp_path)
    train, valid, _ = build_development_features(path, include_ohlc_features=True)
    train_file, valid_file = tmp_path / "train.csv", tmp_path / "valid.csv"
    train.to_csv(train_file, index=False)
    valid.to_csv(valid_file, index=False)
    with pytest.raises(ModelInputError, match="predefined"):
        compare_models(train_file, valid_file, feature_columns=("tomorrow_close",))
    train["tomorrow_close"] = 1.0
    train.to_csv(train_file, index=False)
    with pytest.raises(ModelInputError, match="unexpected columns"):
        compare_models(train_file, valid_file, feature_columns=EXPANDED_FEATURE_COLUMNS)


def test_validation_labels_do_not_affect_fitted_probabilities(tmp_path):
    path = synthetic_prices(tmp_path)
    train, valid, _ = build_development_features(path, include_ohlc_features=True)
    train_file, valid_file = tmp_path / "train.csv", tmp_path / "valid.csv"
    train.to_csv(train_file, index=False)
    valid.to_csv(valid_file, index=False)
    old, _ = compare_models(train_file, valid_file, feature_columns=EXPANDED_FEATURE_COLUMNS)
    valid[TARGET_COLUMN] = 1 - valid[TARGET_COLUMN]
    valid.to_csv(valid_file, index=False)
    new, _ = compare_models(train_file, valid_file, feature_columns=EXPANDED_FEATURE_COLUMNS)
    np.testing.assert_allclose(old["logistic_probability"], new["logistic_probability"])
    assert not old["actual_downside_event"].equals(new["actual_downside_event"])


def test_walk_forward_matches_old_dates_labels_and_baseline(tmp_path):
    path = synthetic_prices(tmp_path)
    old, old_report = walk_forward_validation(path)
    combined, report = compare_ohlc_features(path)
    assert len(combined) == len(old)
    assert combined["date"].equals(old["date"])
    assert combined["actual_downside_event"].equals(old["actual_downside_event"])
    np.testing.assert_allclose(combined["baseline_probability"], old["baseline_probability"])
    np.testing.assert_allclose(combined["original_3feature_probability"], old["logistic_probability"])
    assert report["validation_years"] == [2020, 2021, 2022, 2023, 2024]
    assert report["pooled_probability_scores"]["constant_baseline"] == old_report["pooled_probability_scores"]["constant_baseline"]


def test_no_holdout_labels_or_predictions(tmp_path):
    path = synthetic_prices(tmp_path)
    predictions, report = compare_ohlc_features(path)
    assert predictions["date"].max() < pd.Timestamp("2025-01-01")
    assert max(report["validation_years"]) == 2024
    assert report["final_holdout_start"] == "2025-01-01"


def test_holdout_price_changes_do_not_change_results(tmp_path):
    original = synthetic_prices(tmp_path, filename="before.csv")
    changed = synthetic_prices(tmp_path, holdout_multiplier=50, filename="after.csv")
    a, audit_a = compare_ohlc_features(original)
    b, audit_b = compare_ohlc_features(changed)
    pd.testing.assert_frame_equal(a, b)
    assert audit_a["pooled_probability_scores"] == audit_b["pooled_probability_scores"]
    for key in ("training_rows", "validation_rows", "validation_events", "original_3feature_scores", "expanded_6feature_scores"):
        assert [x[key] for x in audit_a["folds"]] == [x[key] for x in audit_b["folds"]]


def test_save_report_refuses_overwrite_and_is_valid_json(tmp_path):
    path = synthetic_prices(tmp_path)
    out = tmp_path / "out"
    audit = save_ohlc_feature_comparison(path, out)
    assert (out / "ohlc_feature_comparison_predictions.csv").is_file()
    stored = json.loads((out / "ohlc_feature_comparison_report.json").read_text())
    assert stored == audit
    with pytest.raises(OHLCComparisonError, match="already exist"):
        save_ohlc_feature_comparison(path, out)
    save_ohlc_feature_comparison(path, out, overwrite=True)


def test_cli_success_and_failure(tmp_path):
    path = synthetic_prices(tmp_path)
    command = [sys.executable, "-m", "risklens.models.ohlc_comparison_cli", str(path),
               "--output-dir", str(tmp_path / "out")]
    good = subprocess.run(command, text=True, capture_output=True, check=False)
    assert good.returncode == 0, good.stdout + good.stderr
    assert "PASS:" in good.stdout
    bad = subprocess.run(command, text=True, capture_output=True, check=False)
    assert bad.returncode == 1
    assert "FAIL:" in bad.stdout


def test_missing_file_and_wrong_parameter_type(tmp_path):
    with pytest.raises(OHLCComparisonError, match="not found"):
        compare_ohlc_features(tmp_path / "missing.csv")
    path = synthetic_prices(tmp_path)
    with pytest.raises(FeatureDatasetError, match="boolean"):
        build_development_features(path, include_ohlc_features="yes")
