"""Synthetic checks for year-by-year expanding-window model evaluation."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from risklens.features.dataset import build_development_features
from risklens.models.baselines import compare_models
from risklens.models.walk_forward import (
    FOLD_YEARS, FINAL_HOLDOUT_START, WalkForwardError,
    walk_forward_validation, save_walk_forward_validation,
)


@pytest.fixture(scope="module")
def synthetic_history(tmp_path_factory):
    """Synthetic daily levels for 2016-2025; not actual NSE observations."""
    root = tmp_path_factory.mktemp("walk_forward_synthetic")
    days = pd.bdate_range("2016-01-04", "2025-03-31")
    n = len(days)
    # Smooth, fully synthetic return variation plus occasional negative shocks.
    t = np.arange(n, dtype=float)
    one_day = 0.00025 + 0.010 * np.sin(t / 7.0) + 0.004 * np.cos(t / 13.0)
    one_day[(np.arange(n) % 83) == 0] -= 0.025
    close = 100.0 * np.cumprod(1.0 + one_day)
    frame = pd.DataFrame({
        "date": days, "open": close, "high": close,
        "low": close, "close": close,
    })
    path = root / "prices.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def evaluation(synthetic_history):
    return walk_forward_validation(synthetic_history)


def test_five_yearly_folds_are_independent_chronological_evaluations(evaluation):
    predictions, report = evaluation
    assert report["validation_years"] == list(FOLD_YEARS)
    assert [x["validation_year"] for x in report["folds"]] == list(FOLD_YEARS)
    assert len(predictions) == sum(f["validation_range"]["rows"] for f in report["folds"])
    assert predictions["date"].is_monotonic_increasing
    assert predictions["date"].is_unique
    assert predictions["date"].max() < pd.Timestamp(FINAL_HOLDOUT_START)
    for fold in report["folds"]:
        year = fold["validation_year"]
        assert fold["training_range"]["last_date"] < f"{year}-01-01"
        assert f"{year}-01-01" <= fold["validation_range"]["first_date"]
        assert fold["validation_range"]["last_date"] < f"{year+1}-01-01"
        assert fold["training_boundary_labels_purged"] == 1
        assert len(predictions.loc[predictions["validation_year"] == year]) == fold["validation_range"]["rows"]


def test_training_expands_instead_of_reusing_one_fixed_model(evaluation):
    _, report = evaluation
    folds = report["folds"]
    counts = [f["training_range"]["rows"] for f in folds]
    assert counts == sorted(counts)
    assert len(set(counts)) == 5
    assert folds[-1]["training_range"]["last_date"] >= "2023-01-01"
    assert folds[-2]["training_range"]["last_date"] < "2023-01-01"
    for f in folds:
        means = f["training_only_scaler_means"]
        assert set(means) == {"return_1d", "return_5d", "volatility_20d"}


def test_each_fold_predicts_probabilities_and_has_auditable_scores(evaluation):
    preds, report = evaluation
    for col in ("baseline_probability", "logistic_probability"):
        assert preds[col].between(0, 1).all()
    for fold in report["folds"]:
        scores = fold["validation_scores"]
        assert set(scores) == {"constant_baseline", "logistic_regression"}
        for m in scores.values():
            assert m["brier_score"] >= 0
            assert m["log_loss"] >= 0
        year_preds = preds.loc[preds["validation_year"] == fold["validation_year"]]
        assert year_preds["actual_downside_event"].sum() == fold["validation_range"]["events"]
        assert np.allclose(year_preds["baseline_probability"], fold["training_range"]["event_rate"], atol=1e-8)


def test_2024_fold_matches_unchanged_existing_step32_model(synthetic_history, evaluation, tmp_path):
    all_preds, report = evaluation
    train, validation, _ = build_development_features(
        synthetic_history, validation_start="2024-01-01", holdout_start="2025-01-01"
    )
    p_train, p_val = tmp_path / "train.csv", tmp_path / "validation.csv"
    for dataframe, out in ((train, p_train), (validation, p_val)):
        copy = dataframe.copy()
        copy["date"] = copy["date"].dt.strftime("%Y-%m-%d")
        copy.to_csv(out, index=False)
    expected, audit = compare_models(p_train, p_val,
                                     validation_start="2024-01-01", holdout_start="2025-01-01")
    got = all_preds.loc[all_preds["validation_year"] == 2024].reset_index(drop=True)
    assert got["date"].tolist() == expected["date"].tolist()
    for col in ("baseline_probability", "logistic_probability"):
        assert np.allclose(got[col], expected[col])
    assert report["folds"][-1]["validation_scores"] == audit["validation_scores"]


def test_2025_prices_do_not_change_folds_or_predictions(synthetic_history, evaluation, tmp_path):
    prices = pd.read_csv(synthetic_history)
    mask = prices["date"] >= "2025-01-01"
    for col in ("open", "high", "low", "close"):
        prices.loc[mask, col] *= 4.0
    mutated = tmp_path / "future_changed.csv"
    prices.to_csv(mutated, index=False)
    new_predictions, new_report = walk_forward_validation(mutated)
    original_predictions, original_report = evaluation
    pd.testing.assert_frame_equal(new_predictions, original_predictions)
    assert new_report["folds"] == original_report["folds"]
    assert new_report["pooled_probability_scores"] == original_report["pooled_probability_scores"]


def test_changing_2024_does_not_change_earlier_yearly_models(synthetic_history, evaluation, tmp_path):
    prices = pd.read_csv(synthetic_history)
    mask = prices["date"] >= "2024-01-01"
    # 2024 is a valid evaluation year, but is future relative to 2020-2023 folds.
    for col in ("open", "high", "low", "close"):
        prices.loc[mask, col] *= 1.3
    changed = tmp_path / "changed_from_2024.csv"
    prices.to_csv(changed, index=False)
    new_predictions, _ = walk_forward_validation(changed)
    original, _ = evaluation
    pd.testing.assert_frame_equal(
        new_predictions.loc[new_predictions["validation_year"] < 2024].reset_index(drop=True),
        original.loc[original["validation_year"] < 2024].reset_index(drop=True),
    )


def test_pooled_brier_calculation_uses_validation_predictions_only(evaluation):
    predictions, report = evaluation
    y = predictions["actual_downside_event"].to_numpy()
    for model, col in (("constant_baseline", "baseline_probability"),
                       ("logistic_regression", "logistic_probability")):
        p = predictions[col].to_numpy()
        assert report["pooled_probability_scores"][model]["brier_score"] == pytest.approx(
            np.mean((p - y) ** 2), abs=1e-8
        )
    assert "roc_auc" not in report["pooled_probability_scores"]["logistic_regression"]


def test_missing_or_invalid_csv_fails(tmp_path, synthetic_history):
    with pytest.raises(WalkForwardError, match="not found"):
        walk_forward_validation(tmp_path / "absent.csv")
    x = pd.read_csv(synthetic_history)
    x.loc[0, "high"] = x.loc[0, "close"] - 10
    bad = tmp_path / "bad.csv"
    x.to_csv(bad, index=False)
    with pytest.raises(WalkForwardError, match="Fold 2020 feature preparation failed"):
        walk_forward_validation(bad)


def test_outputs_write_and_refuse_overwrite(synthetic_history, tmp_path):
    report = save_walk_forward_validation(synthetic_history, tmp_path / "out")
    saved = json.loads((tmp_path / "out" / "walk_forward_validation_report.json").read_text())
    p = pd.read_csv(tmp_path / "out" / "walk_forward_validation_predictions.csv")
    assert saved == report
    assert len(p) == report["total_validation_rows"]
    assert p["date"].max() < "2025-01-01"
    with pytest.raises(WalkForwardError, match="already exists"):
        save_walk_forward_validation(synthetic_history, tmp_path / "out")
    assert save_walk_forward_validation(synthetic_history, tmp_path / "out", overwrite=True)["folds"] == report["folds"]


def test_cli_output_and_explicit_overwrite(synthetic_history, tmp_path):
    command = [sys.executable, "-m", "risklens.models.walk_forward_cli", str(synthetic_history),
               "--output-dir", str(tmp_path / "output")]
    success = subprocess.run(command, capture_output=True, text=True, check=False)
    assert success.returncode == 0, success.stderr + success.stdout
    assert "PASS:" in success.stdout
    assert "2024:" in success.stdout
    failure = subprocess.run(command, capture_output=True, text=True, check=False)
    assert failure.returncode == 1
    assert "FAIL:" in failure.stdout
