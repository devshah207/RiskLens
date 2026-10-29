"""Synthetic tests: the actual 2025+ NIFTY 50 holdout is never used."""

import json

import numpy as np
import pandas as pd
import pytest

from risklens.models.baselines import (
    FIXED_THRESHOLD, ModelInputError, compare_models, save_comparison,
)
from risklens.features.dataset import FEATURE_COLUMNS, TARGET_COLUMN


def split_files(tmp_path, *, n_train=80, n_valid=30):
    """Artificial chronological feature tables with two labels in each set."""
    def frame(start, n):
        k = np.arange(n, dtype=float)
        y = (np.arange(n) % 5 == 0).astype(int)
        return pd.DataFrame({
            "date": pd.bdate_range(start, periods=n).strftime("%Y-%m-%d"),
            "return_1d": np.where(y == 1, -0.02, 0.003) + k * 1e-6,
            "return_5d": np.where(y == 1, -0.04, 0.008) + k * 2e-6,
            "volatility_20d": 0.11 + (k % 13) / 1000,
            TARGET_COLUMN: y,
        })
    train = tmp_path / "train_features.csv"
    valid = tmp_path / "validation_features.csv"
    frame("2022-01-03", n_train).to_csv(train, index=False)
    frame("2023-01-02", n_valid).to_csv(valid, index=False)
    return train, valid


def test_baseline_is_training_rate_on_all_validation_rows(tmp_path):
    train, valid = split_files(tmp_path)
    predictions, report = compare_models(train, valid)
    assert len(predictions) == 30
    assert set(predictions.columns) == {
        "date", "actual_downside_event", "baseline_probability", "logistic_probability"
    }
    assert np.allclose(predictions["baseline_probability"], 0.2)
    assert report["training"]["events"] == 16
    assert report["validation"]["events"] == 6
    assert report["validation_scores"]["constant_baseline"]["roc_auc"] == pytest.approx(.5)
    assert report["validation_scores"]["constant_baseline"]["average_precision"] == pytest.approx(.2)


def test_logistic_returns_valid_probabilities_and_keeps_feature_order(tmp_path):
    train, valid = split_files(tmp_path)
    predictions, report = compare_models(train, valid)
    assert predictions["logistic_probability"].between(0, 1).all()
    assert report["features"] == list(FEATURE_COLUMNS)
    assert set(report["models"]["logistic_regression"]["coefficients_per_standardized_feature"]) == set(FEATURE_COLUMNS)
    assert report["models"]["logistic_regression"]["hyperparameters"]["class_weight"] is None
    assert FIXED_THRESHOLD == .5


def test_scaler_fits_training_data_only(tmp_path):
    train, valid = split_files(tmp_path)
    predictions, report = compare_models(train, valid)
    x = pd.read_csv(train)
    train_means = x[list(FEATURE_COLUMNS)].mean()
    means = report["models"]["logistic_regression"]["scaler_mean"]
    for col in FEATURE_COLUMNS:
        assert means[col] == pytest.approx(train_means[col])
    # Shift validation distribution massively. Training scaler remains unchanged.
    v = pd.read_csv(valid)
    v["volatility_20d"] = v["volatility_20d"] + 100
    v.to_csv(valid, index=False)
    _, changed = compare_models(train, valid)
    assert changed["models"]["logistic_regression"]["scaler_mean"] == means
    assert changed["models"]["logistic_regression"]["coefficients_per_standardized_feature"] == report["models"]["logistic_regression"]["coefficients_per_standardized_feature"]
    assert changed["training"] == report["training"]


def test_changing_validation_labels_does_not_change_predictions(tmp_path):
    train, valid = split_files(tmp_path)
    before, _ = compare_models(train, valid)
    v = pd.read_csv(valid)
    v[TARGET_COLUMN] = 1-v[TARGET_COLUMN]
    v.to_csv(valid, index=False)
    after, _ = compare_models(train, valid)
    assert np.allclose(before["baseline_probability"], after["baseline_probability"])
    assert np.allclose(before["logistic_probability"], after["logistic_probability"])
    assert not before["actual_downside_event"].equals(after["actual_downside_event"])


def test_no_2025_data_accepted_even_if_file_only_contains_validation_rows(tmp_path):
    train, valid = split_files(tmp_path)
    v = pd.read_csv(valid)
    v.loc[0, "date"] = "2025-01-02"
    v = v.sort_values("date")
    v.to_csv(valid, index=False)
    with pytest.raises(ModelInputError, match="holdout stays sealed"):
        compare_models(train, valid)


def test_training_date_cannot_cross_validation_boundary(tmp_path):
    train, valid = split_files(tmp_path)
    t = pd.read_csv(train)
    t.loc[len(t)-1, "date"] = "2023-01-01"
    t.to_csv(train, index=False)
    with pytest.raises(ModelInputError, match="2023-01-01"):
        compare_models(train, valid)


def test_extra_leaked_feature_rejected(tmp_path):
    train, valid = split_files(tmp_path)
    t = pd.read_csv(train)
    t["tomorrow_close"] = 5
    t.to_csv(train, index=False)
    with pytest.raises(ModelInputError, match="unexpected columns"):
        compare_models(train, valid)


@pytest.mark.parametrize("problem", ["missing", "infinite", "nan", "nonnumeric"])
def test_bad_features_rejected(tmp_path, problem):
    train, valid = split_files(tmp_path)
    v = pd.read_csv(valid)
    v["return_1d"] = v["return_1d"].astype("object")
    v.loc[0, "return_1d"] = {"missing": "", "infinite": np.inf,
                              "nan": np.nan, "nonnumeric": "oops"}[problem]
    v.to_csv(valid, index=False)
    with pytest.raises(ModelInputError, match="(nonfinite|nonnumeric)"):
        compare_models(train, valid)


def test_bad_target_rejected(tmp_path):
    train, valid = split_files(tmp_path)
    v = pd.read_csv(valid)
    v.loc[0, TARGET_COLUMN] = 2
    v.to_csv(valid, index=False)
    with pytest.raises(ModelInputError, match="only integer 0 and 1"):
        compare_models(train, valid)


def test_training_with_one_class_refused(tmp_path):
    train, valid = split_files(tmp_path)
    t = pd.read_csv(train)
    t[TARGET_COLUMN] = 0
    t.to_csv(train, index=False)
    with pytest.raises(ModelInputError, match="both non-events"):
        compare_models(train, valid)


def test_one_class_validation_scores_that_are_undefined_are_null(tmp_path):
    train, valid = split_files(tmp_path)
    v = pd.read_csv(valid)
    v[TARGET_COLUMN] = 0
    v.to_csv(valid, index=False)
    _, report = compare_models(train, valid)
    for scores in report["validation_scores"].values():
        assert scores["roc_auc"] is None
        assert scores["average_precision"] is None
        assert scores["brier_score"] >= 0
    json.dumps(report, allow_nan=False)


def test_save_and_refuse_overwrite(tmp_path):
    train, valid = split_files(tmp_path)
    out = tmp_path / "out"
    report = save_comparison(train, valid, out)
    assert json.loads((out / "baseline_logistic_validation_report.json").read_text()) == report
    predictions = pd.read_csv(out / "validation_model_predictions.csv")
    assert len(predictions) == 30
    assert predictions["date"].max() < "2025-01-01"
    with pytest.raises(ModelInputError, match="already exists"):
        save_comparison(train, valid, out)
    save_comparison(train, valid, out, overwrite=True)


def test_unsorted_dates_refused(tmp_path):
    train, valid = split_files(tmp_path)
    v = pd.read_csv(valid)
    v = v.iloc[::-1]
    v.to_csv(valid, index=False)
    with pytest.raises(ModelInputError, match="chronologically increasing"):
        compare_models(train, valid)
