"""Synthetic, development-only tests of retrospective probability diagnostics."""

import json

import numpy as np
import pandas as pd
import pytest

from risklens.models.diagnostics import (
    BIN_LABELS, DiagnosticsError, MODEL_COLUMNS, analyze_predictions,
    save_diagnostics,
)


def synthetic_predictions(tmp_path, *, name="predictions.csv"):
    dates = list(pd.bdate_range("2023-01-02", periods=22)) + list(pd.bdate_range("2024-01-02", periods=22))
    i = np.arange(len(dates))
    labels = (i % 5 == 0).astype(int)
    # Explicit all valid probabilities, including a few large ones.
    risk = np.where(labels == 1, 0.19, 0.075) + 0.004 * (i % 7)
    source = pd.DataFrame({
        "validation_year": [d.year for d in dates],
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "actual_downside_event": labels,
        "original_3feature_probability": np.clip(risk+.015, 0, 1),
        "expanded_6feature_probability": np.clip(risk+.025, 0, 1),
        "baseline_probability": np.full(len(dates), .1),
        "xgboost_3_features_probability": risk,
        "xgboost_6_features_probability": np.clip(risk-.015, 0, 1),
    })
    path = tmp_path / name
    source.to_csv(path, index=False)
    return path


def test_yearly_results_and_pooled_probabilities(tmp_path):
    yearly, reliability, cases, report = analyze_predictions(synthetic_predictions(tmp_path))
    assert report["rows"] == 44
    assert report["years"] == [2023, 2024]
    assert report["observed_events"] == 9
    assert len(yearly) == len(MODEL_COLUMNS) * 2
    assert len(reliability) == len(MODEL_COLUMNS) * len(BIN_LABELS)
    assert report["pooled_scores"]["constant_baseline"]["brier_score"] == pytest.approx(
        (9*.9**2 + 35*.1**2)/44, abs=1e-7)
    assert "roc_auc" not in report["pooled_scores"]["xgboost_3_features"]
    assert not cases.empty


def test_per_year_mean_gap_and_no_fitting(tmp_path):
    p = synthetic_predictions(tmp_path)
    year, _, _, _ = analyze_predictions(p)
    base = year[(year.model == "constant_baseline") & (year.validation_year == 2023)].iloc[0]
    assert base.mean_predicted_probability == pytest.approx(.1)
    assert base.probability_gap_predicted_minus_observed == pytest.approx(.1 - 5/22)
    assert base.roc_auc == pytest.approx(.5)


def test_reliability_bins_counts_add_up(tmp_path):
    _, reliability, _, report = analyze_predictions(synthetic_predictions(tmp_path))
    for name in MODEL_COLUMNS:
        sub = reliability[reliability.model == name]
        assert int(sub.rows.sum()) == report["rows"]
        assert int(sub.events.sum()) == report["observed_events"]
    assert report["fixed_reliability_bin_edges"] == [0, .05, .10, .20, .40, 1]


def test_exact_bin_cutoffs_and_probability_one(tmp_path):
    p = synthetic_predictions(tmp_path)
    df = pd.read_csv(p)
    df.loc[:4, "xgboost_3_features_probability"] = [0, .05, .10, .20, 1.0]
    df.to_csv(p, index=False)
    _, bins, _, _ = analyze_predictions(p)
    b = bins[bins.model == "xgboost_3_features"].set_index("probability_bin")
    assert b.loc["0–5%", "rows"] == 1
    assert b.loc["20–40%", "rows"] >= 1
    assert b.loc["40–100%", "rows"] == 1


def test_error_examples_are_observed_labels_not_model_alerts(tmp_path):
    _, _, cases, _ = analyze_predictions(synthetic_predictions(tmp_path), example_count=3)
    assert len(cases) == 6
    assert (cases.iloc[:3].actual_downside_event == 1).all()
    assert (cases.iloc[3:].actual_downside_event == 0).all()
    assert cases.iloc[:3].xgboost_3_features_probability.is_monotonic_increasing
    assert cases.iloc[3:].xgboost_3_features_probability.is_monotonic_decreasing


def test_reject_any_2025_holdout_row(tmp_path):
    p = synthetic_predictions(tmp_path)
    df = pd.read_csv(p)
    df.loc[43, "date"] = "2025-01-02"
    df.loc[43, "validation_year"] = 2025
    df.to_csv(p, index=False)
    with pytest.raises(DiagnosticsError, match="holdout|2025"):
        analyze_predictions(p)


def test_reject_future_column(tmp_path):
    p = synthetic_predictions(tmp_path)
    df = pd.read_csv(p)
    df["tomorrow_close"] = 999
    df.to_csv(p, index=False)
    with pytest.raises(DiagnosticsError, match="schema"):
        analyze_predictions(p)


def test_reject_year_not_matching_date(tmp_path):
    p = synthetic_predictions(tmp_path)
    df = pd.read_csv(p)
    df.loc[0, "validation_year"] = 2024
    df.to_csv(p, index=False)
    with pytest.raises(DiagnosticsError, match="must match"):
        analyze_predictions(p)


@pytest.mark.parametrize("problem", ["missing", "nan", "inf", "negative", "over_one", "string"])
def test_reject_invalid_probabilities(tmp_path, problem):
    p = synthetic_predictions(tmp_path)
    df = pd.read_csv(p)
    value = {"missing": None, "nan": np.nan, "inf": np.inf,
             "negative": -.01, "over_one": 1.01, "string": "not a probability"}[problem]
    df["xgboost_6_features_probability"] = df["xgboost_6_features_probability"].astype(object)
    df.loc[0, "xgboost_6_features_probability"] = value
    df.to_csv(p, index=False)
    with pytest.raises(DiagnosticsError, match="probabilities|numeric"):
        analyze_predictions(p)


def test_reject_unordered_dates_and_duplicates(tmp_path):
    p = synthetic_predictions(tmp_path)
    df = pd.read_csv(p)
    df.iloc[::-1].to_csv(p, index=False)
    with pytest.raises(DiagnosticsError, match="sorted"):
        analyze_predictions(p)
    df.loc[1, "date"] = df.loc[0, "date"]
    df.to_csv(p, index=False)
    with pytest.raises(DiagnosticsError, match="unique"):
        analyze_predictions(p)


def test_reject_invalid_event_label_and_date(tmp_path):
    p = synthetic_predictions(tmp_path)
    df = pd.read_csv(p)
    df.loc[0, "actual_downside_event"] = 2
    df.to_csv(p, index=False)
    with pytest.raises(DiagnosticsError, match="only 0 or 1"):
        analyze_predictions(p)
    df.loc[0, "actual_downside_event"] = 1
    df.loc[0, "date"] = "2023-02-30"
    df.to_csv(p, index=False)
    with pytest.raises(DiagnosticsError, match="ISO"):
        analyze_predictions(p)


def test_single_class_year_has_null_auc_ap(tmp_path):
    p = synthetic_predictions(tmp_path)
    df = pd.read_csv(p)
    df.loc[df.validation_year == 2024, "actual_downside_event"] = 0
    df.to_csv(p, index=False)
    yearly, _, _, _ = analyze_predictions(p)
    group = yearly[(yearly.model == "xgboost_3_features") & (yearly.validation_year == 2024)]
    assert pd.isna(group.iloc[0].roc_auc)
    assert pd.isna(group.iloc[0].average_precision)


def test_missing_file_and_bad_examples(tmp_path):
    with pytest.raises(DiagnosticsError, match="not found"):
        analyze_predictions(tmp_path / "missing.csv")
    p = synthetic_predictions(tmp_path)
    for bad in (0, -1, 101, 1.2, True):
        with pytest.raises(DiagnosticsError, match="example_count"):
            analyze_predictions(p, example_count=bad)


def test_save_and_prevent_overwrite(tmp_path):
    p = synthetic_predictions(tmp_path)
    out = tmp_path / "diagnostics"
    report = save_diagnostics(p, out)
    assert json.loads((out / "model_diagnostics_report.json").read_text()) == report
    assert len(pd.read_csv(out / "model_yearly_diagnostics.csv")) == 10
    assert len(pd.read_csv(out / "model_probability_reliability.csv")) == 25
    assert pd.read_csv(out / "xgboost_3feature_error_examples.csv").date.max() < "2025-01-01"
    with pytest.raises(DiagnosticsError, match="already exists"):
        save_diagnostics(p, out)
    save_diagnostics(p, out, overwrite=True)

