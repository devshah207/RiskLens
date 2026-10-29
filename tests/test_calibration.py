"""Synthetic calibration tests: never use real price files or 2025+ outcomes."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import risklens.models.calibration as module
from risklens.features.dataset import TARGET_COLUMN, build_development_features
from risklens.models.calibration import (
    CalibrationError, _fit_sigmoid_calibrator, _logit, _reliability_bins,
    compare_calibration, save_calibration_comparison,
)


@pytest.fixture(scope="module")
def example_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("calibration_synthetic")
    dates = pd.bdate_range("2016-01-01", "2025-01-10")
    rng = np.random.default_rng(82)
    returns = rng.normal(0.0003, 0.012, len(dates))
    closes = 1000 * np.exp(np.cumsum(returns))
    frame = pd.DataFrame({"date": dates, "open": closes, "high": closes,
                          "low": closes, "close": closes})
    original = root / "source.csv"
    modified = root / "holdout_changed.csv"
    frame.to_csv(original, index=False)
    frame.loc[frame["date"] >= pd.Timestamp("2025-01-01"), ["open", "high", "low", "close"]] *= 4
    frame.to_csv(modified, index=False)
    return original, modified


@pytest.fixture(scope="module")
def example_comparison(example_source):
    return compare_calibration(example_source[0])


def test_all_five_years_and_holdout_sealed(example_comparison):
    predictions, report = example_comparison
    assert report["validation_years"] == [2020, 2021, 2022, 2023, 2024]
    assert sorted(predictions["validation_year"].unique()) == [2020, 2021, 2022, 2023, 2024]
    assert predictions["date"].max() < pd.Timestamp("2025-01-01")
    assert predictions["date"].is_unique
    assert report["total_validation_rows"] == len(predictions)
    assert report["total_validation_events"] == int(predictions["actual_downside_event"].sum())


def test_each_training_calibration_validation_boundary(example_comparison):
    _, report = example_comparison
    for fold in report["folds"]:
        year = fold["validation_year"]
        assert fold["base_fit_period"][1] < f"{year-1}-01-01"
        assert f"{year-1}-01-01" <= fold["calibration_period"][0] <= fold["calibration_period"][1] < f"{year}-01-01"
        assert f"{year}-01-01" <= fold["validation_period"][0] <= fold["validation_period"][1] < f"{year+1}-01-01"
        assert fold["boundary_purges_checked"] == {"base_to_calibration": 1, "calibration_to_validation": 1}
        assert fold["calibrator_fit_audit"]["fit_rows"] == fold["calibration_rows"]
        assert fold["calibrator_fit_audit"]["fit_events"] == fold["calibration_events"]
        assert fold["base_model_fit_audit"]["train_rows"] == fold["base_fit_rows"]


def test_scores_and_valid_probability_vectors(example_comparison):
    prediction, report = example_comparison
    assert prediction.columns.tolist() == [
        "validation_year", "date", "actual_downside_event", "baseline_probability",
        "xgboost_raw_probability", "xgboost_calibrated_probability",
    ]
    for col in ["baseline_probability", "xgboost_raw_probability", "xgboost_calibrated_probability"]:
        assert prediction[col].between(0, 1).all()
    assert not np.allclose(prediction["xgboost_raw_probability"], prediction["xgboost_calibrated_probability"])
    for score in report["pooled_probability_scores"].values():
        assert score["brier_score"] >= 0
        assert score["log_loss"] >= 0


def test_baseline_prevalidation_history_only(example_source, example_comparison):
    prediction, _ = example_comparison
    for year in range(2020, 2025):
        train, _, _ = build_development_features(
            example_source[0], validation_start=f"{year}-01-01",
            holdout_start=f"{year+1}-01-01",
        )
        selected = prediction.loc[prediction["validation_year"] == year]
        assert np.allclose(selected["baseline_probability"], train[TARGET_COLUMN].mean())


def test_logit_and_sigmoid_transform_manual():
    raw = np.array([0.02, 0.12, 0.3, 0.7])
    assert np.allclose(_logit(raw).ravel(), np.log(raw/(1-raw)))
    p, audit = _fit_sigmoid_calibrator(
        raw, np.array([0, 0, 1, 1]), np.array([0.1, 0.5, 0.9])
    )
    manual_z = audit["intercept"] + audit["slope"] * _logit(np.array([0.1, 0.5, 0.9])).ravel()
    manual_p = 1 / (1 + np.exp(-manual_z))
    assert np.allclose(p, manual_p)
    assert audit["fit_rows"] == 4
    assert audit["fit_events"] == 2


@pytest.mark.parametrize("bad", [np.array([-0.2]), np.array([1.1]), np.array([np.nan]), np.array([[0.1]])])
def test_logit_invalid_values_fail(bad):
    with pytest.raises(CalibrationError, match="Probabilities must"):
        _logit(bad)


@pytest.mark.parametrize("labels", [np.array([0, 0]), np.array([1, 1]), np.array([0, 2])])
def test_invalid_calibration_classes_fail(labels):
    with pytest.raises(CalibrationError):
        _fit_sigmoid_calibrator(np.array([0.1, 0.2]), labels, np.array([0.3]))


def test_mismatched_calibration_lengths_fail():
    with pytest.raises(CalibrationError):
        _fit_sigmoid_calibrator(np.array([0.1, 0.2, 0.3]), np.array([0, 1]), np.array([0.3]))


def test_reliability_bins_account_for_all_rows(example_comparison):
    prediction, report = example_comparison
    for name, bins in report["pooled_reliability_bins"].items():
        assert sum(b["observations"] for b in bins) == len(prediction)
        assert len(bins) == 5
        for b in bins:
            if b["observations"] == 0:
                assert b["mean_predicted_probability"] is None
                assert b["observed_event_frequency"] is None
            else:
                assert 0 <= b["observed_event_frequency"] <= 1


def test_reliability_known_small_example():
    frame = pd.DataFrame({"actual_downside_event": [0, 1, 1], "p": [0.01, 0.02, 0.25]})
    bins = _reliability_bins(frame, "p")
    assert bins[0]["observations"] == 2
    assert bins[0]["observed_event_frequency"] == .5
    assert bins[-1]["observations"] == 1


def test_holdout_changes_do_not_affect_results(example_source, example_comparison):
    a, ar = example_comparison
    b, br = compare_calibration(example_source[1])
    pd.testing.assert_frame_equal(a, b)
    exclude = {"source_filename", "source_sha256", "generated_at_utc"}
    assert {k: v for k, v in ar.items() if k not in exclude} == {k: v for k, v in br.items() if k not in exclude}


def test_validation_outcomes_not_used_in_calibrator_fit(example_source, monkeypatch):
    monkeypatch.setattr(module, "FOLD_YEARS", (2024,))
    reference, _ = module.compare_calibration(example_source[0])
    original = module.build_development_features

    def changed_validation_labels(*args, **kwargs):
        train, valid, audit = original(*args, **kwargs)
        if kwargs["validation_start"] == "2024-01-01":
            valid[TARGET_COLUMN] = 1 - valid[TARGET_COLUMN]
        return train, valid, audit

    monkeypatch.setattr(module, "build_development_features", changed_validation_labels)
    changed, _ = module.compare_calibration(example_source[0])
    for col in ("baseline_probability", "xgboost_raw_probability", "xgboost_calibrated_probability"):
        assert np.array_equal(reference[col].to_numpy(), changed[col].to_numpy())
    assert not reference["actual_downside_event"].equals(changed["actual_downside_event"])


def test_xgb_base_fit_never_sees_calibration_year(example_source, monkeypatch):
    monkeypatch.setattr(module, "FOLD_YEARS", (2024,))
    original_fit = module._fit_predict_fixed_xgb
    calls = []

    def spy(train, to_score, features):
        calls.append((train["date"].max(), to_score["date"].min(), tuple(features)))
        return original_fit(train, to_score, features)

    monkeypatch.setattr(module, "_fit_predict_fixed_xgb", spy)
    module.compare_calibration(example_source[0])
    assert len(calls) == 1
    assert calls[0][0] < pd.Timestamp("2023-01-01") <= calls[0][1]
    assert calls[0][2] == module.FEATURE_COLUMNS


def test_missing_file_fails(tmp_path):
    with pytest.raises(CalibrationError, match="not found"):
        compare_calibration(tmp_path / "missing.csv")


def test_write_and_refuse_overwrite(example_source, tmp_path):
    out = tmp_path / "processed"
    report = save_calibration_comparison(example_source[0], out)
    assert json.loads((out / module.REPORT_FILENAME).read_text()) == report
    frame = pd.read_csv(out / module.PREDICTIONS_FILENAME)
    assert len(frame) == report["total_validation_rows"]
    assert (frame["date"] < "2025-01-01").all()
    with pytest.raises(CalibrationError, match="already exists"):
        save_calibration_comparison(example_source[0], out)
    assert save_calibration_comparison(example_source[0], out, overwrite=True)["total_validation_rows"] == len(frame)


def test_cli_reports_pass_then_fail(example_source, tmp_path):
    target = tmp_path / "output"
    cmd = [sys.executable, "-m", "risklens.models.calibration_cli", str(example_source[0]),
           "--output-dir", str(target)]
    success = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert success.returncode == 0, success.stderr + success.stdout
    assert "PASS:" in success.stdout
    failure = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert failure.returncode == 1
    assert "FAIL:" in failure.stdout
