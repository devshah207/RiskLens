"""Phase 8 app contracts: all fixtures are synthetic and never touch the holdout."""
from __future__ import annotations

import hashlib
import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from app.services.saved_risk_evidence import (
    EvidenceError, PREDICTION_COLUMNS, RECORDED_METRICS, load_evidence,
    model_comparison, read_saved_predictions, recorded_comparison,
)
from app.services.demo_data import EXAMPLES, recorded_question, synthetic_timeline


def sample_predictions(path: Path):
    dates = pd.bdate_range("2025-01-01", periods=433)
    # Synthetic inputs test validation only; not real risk probabilities.
    data = pd.DataFrame({
        "validation_year": dates.year,
        "date": dates.strftime("%Y-%m-%d"),
        "actual_downside_event": [1] * 42 + [0] * 391,
        "xgboost_11_fixed_probability": [.15] * 433,
        "lstm_optuna_86_probability": [.12] * 433,
    })
    data.loc[:, list(PREDICTION_COLUMNS)].to_csv(path, index=False)
    return data


def test_saved_predictions_read_only(tmp_path):
    csv = tmp_path / "predictions.csv"
    sample_predictions(csv)
    digest = hashlib.sha256(csv.read_bytes()).hexdigest()
    result = read_saved_predictions(csv)
    assert len(result) == 433 and result.actual_downside_event.sum() == 42
    assert result.date.is_monotonic_increasing
    assert hashlib.sha256(csv.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("mutation", ["duplicate", "probability", "outcome", "schema", "date"])
def test_bad_prediction_fails_closed(tmp_path, mutation):
    csv = tmp_path / "holdout.csv"
    original = sample_predictions(csv)
    if mutation == "duplicate":
        original.loc[1, "date"] = original.loc[0, "date"]
    if mutation == "probability":
        original.loc[0, "lstm_optuna_86_probability"] = 1.6
    if mutation == "outcome":
        original.loc[3, "actual_downside_event"] = 2
    if mutation == "schema":
        original.rename(columns={"validation_year": "year"}, inplace=True)
    if mutation == "date":
        original.loc[0, "date"] = "2024-12-30"
    original.to_csv(csv, index=False)
    with pytest.raises(EvidenceError):
        read_saved_predictions(csv)


def test_comparison_reads_report_without_recomputing_metrics():
    report = {"eligible_holdout_rows": 433,
              "selected_before_opening": "fixed_xgboost_11",
              "model_scores": copy.deepcopy(RECORDED_METRICS)}
    result = model_comparison(report)
    assert result.loc[0, "ROC-AUC ↑"] == RECORDED_METRICS["xgboost_fixed"]["roc_auc"]
    report["model_scores"]["xgboost_fixed"] = dict(RECORDED_METRICS["xgboost_fixed"], roc_auc=.99)
    with pytest.raises(EvidenceError, match="differs"):
        model_comparison(report)


def test_missing_local_files_not_faked(tmp_path):
    evidence = load_evidence(tmp_path)
    assert evidence.predictions is None
    assert evidence.report is None
    assert "predictions" in evidence.missing
    assert recorded_comparison().shape == (2, 5)


def test_synthetic_timeline_has_no_fake_dates_or_outcome_labels():
    fake = synthetic_timeline(40)
    assert list(fake.columns) == [
        "fictional_session", "fictional_xgboost_probability", "fictional_lstm_probability"]
    assert "date" not in fake.columns and "actual_downside_event" not in fake.columns


def test_demo_has_honest_provenance_and_statuses():
    for ex in EXAMPLES:
        recorded = recorded_question(ex["id"])
        assert recorded["provider"] == "recorded_demo_only"
        assert recorded["status"] in {"answered", "abstained", "needs_review"}
        assert recorded["recorded_source"]
    with pytest.raises(KeyError):
        recorded_question("unknown")


def test_load_original_layout_with_synthetic_saved_outputs(tmp_path):
    """Representative real path contract; values remain made up for unit testing."""
    pred_dir = tmp_path / "data/processed/phase5_final_holdout"
    pred_dir.mkdir(parents=True)
    sample_predictions(pred_dir / "final_holdout_predictions.csv")
    (pred_dir / "final_holdout_report.json").write_text(json.dumps({
        "eligible_holdout_rows": 433,
        "selected_before_opening": "fixed_xgboost_11",
        "model_scores": RECORDED_METRICS,
        "historical_var": {"95": {"exceptions": 30, "observed_rate": 30/433}},
    }))
    pd.DataFrame([{"model": "xgboost_fixed", "observations": 5,
                   "mean_probability": .1, "event_frequency": .2}]).to_csv(
                       pred_dir / "final_holdout_reliability.csv", index=False)
    shap = tmp_path / "data/processed/phase6_explainability"
    shap.mkdir(parents=True)
    pd.DataFrame([{"feature": "return_1d", "mean_absolute_shap_log_odds": .01}]).to_csv(
        shap / "xgboost_shap_summary.csv", index=False)
    orig = (pred_dir / "final_holdout_predictions.csv").read_bytes()
    loaded = load_evidence(tmp_path)
    assert len(loaded.predictions) == 433
    assert loaded.shap is not None
    assert loaded.calibration is not None
    assert loaded.report["historical_var"]["95"]["exceptions"] == 30
    assert (pred_dir / "final_holdout_predictions.csv").read_bytes() == orig


def test_final_holdout_column_order_matches_original_writer():
    """The actual Phase 5 CSV saves DATE first (unlike Phase 4 development CSV)."""
    assert PREDICTION_COLUMNS == (
        "date", "validation_year", "actual_downside_event",
        "xgboost_11_fixed_probability", "lstm_optuna_86_probability",
    )


def test_owner_saved_sources_when_present_are_read_only_and_match_freeze():
    """Additional integration check on the owner's Mac; skips on public CI.

    Reads ONLY the existing files. Does not fit any model, calculate new metrics,
    or update the protected Phase 6 evidence. Original local outputs are not
    bundled in this test module or copied into a repository fixture.
    """
    root = Path(__file__).resolve().parents[1]
    original_predictions = root / "data/processed/phase5_final_holdout/final_holdout_predictions.csv"
    original_report = root / "data/processed/phase5_final_holdout/final_holdout_report.json"
    manifest_path = root / "docs/PHASE6_MODEL_FREEZE_MANIFEST.json"
    if not all(p.is_file() for p in (original_predictions, original_report, manifest_path)):
        pytest.skip("The original saved Phase 5 evidence is local-only and not in public CI")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protected = manifest["protected_file_sha256"]
    protected_available = [root / name for name in protected if (root / name).is_file()]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected_available}
    for p, digest in before.items():
        assert digest == protected[str(p.relative_to(root))], f"Original evidence hash mismatch: {p.name}"

    evidence = load_evidence(root)
    assert evidence.predictions is not None
    assert len(evidence.predictions) == 433
    assert int(evidence.predictions.actual_downside_event.sum()) == 42
    assert evidence.report["selected_before_opening"] == "fixed_xgboost_11"
    assert evidence.report["historical_var"]["95"]["exceptions"] == 30
    assert evidence.report["historical_var"]["99"]["exceptions"] == 7
    if evidence.feature_drift is not None:
        assert len(evidence.feature_drift) == 11
    expected = model_comparison(evidence.report)
    assert expected.loc[0, "ROC-AUC ↑"] == RECORDED_METRICS["xgboost_fixed"]["roc_auc"]
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected_available} == before
