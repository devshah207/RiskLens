"""Read-only adapters for RiskLens Phase 4–6 saved research outputs.

No imports from model training, holdout evaluation, or feature rebuilding modules.
No CSV/JSON is created, rewritten or silently substituted with synthetic scores.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


class EvidenceError(ValueError):
    """A saved result is missing a required field or contradicts its own contract."""


PREDICTION_COLUMNS = (
    "date", "validation_year", "actual_downside_event",
    "xgboost_11_fixed_probability", "lstm_optuna_86_probability",
)
MODEL_NAMES = {
    "xgboost_fixed": "Fixed XGBoost — preselected champion",
    "lstm_optuna_trial86": "Optuna LSTM #86 — frozen challenger",
}
PROBABILITY_COLUMNS = {
    "xgboost_fixed": "xgboost_11_fixed_probability",
    "lstm_optuna_trial86": "lstm_optuna_86_probability",
}
METRIC_NAMES = {
    "roc_auc": "ROC-AUC ↑", "average_precision": "Average precision ↑",
    "brier_score": "Brier score ↓", "log_loss": "Log loss ↓",
}
# Documentation-only values from the authoritative 2026-11-02 handover.
# These are NEVER passed off as fresh calculations on locally mounted data.
RECORDED_METRICS = {
    "xgboost_fixed": {"roc_auc": .67275606, "average_precision": .18610282,
                      "brier_score": .08477265, "log_loss": .30329503},
    "lstm_optuna_trial86": {"roc_auc": .68487395, "average_precision": .21897271,
                           "brier_score": .08429920, "log_loss": .30036197},
}
SAVED_FILES = {
    "predictions": "data/processed/phase5_final_holdout/final_holdout_predictions.csv",
    "report": "data/processed/phase5_final_holdout/final_holdout_report.json",
    "calibration": "data/processed/phase5_final_holdout/final_holdout_reliability.csv",
    "monitoring": "data/processed/phase6_monitoring/monitoring_summary.json",
    "monitoring_periods": "data/processed/phase6_monitoring/monitoring_periods.csv",
    "monitoring_bins": "data/processed/phase6_monitoring/monitoring_score_bins.csv",
    "shap": "data/processed/phase6_explainability/xgboost_shap_summary.csv",
    "lstm_sensitivity": "data/processed/phase6_lstm_sensitivity/lstm_sequence_sensitivity.csv",
    "feature_drift": "data/processed/phase6_feature_drift/feature_drift_table.csv",
    "drift_investigation": "data/processed/phase6_drift_investigation/three_feature_investigation.json",
    "manifest": "docs/PHASE6_MODEL_FREEZE_MANIFEST.json",
}


@dataclass
class SavedEvidence:
    predictions: pd.DataFrame | None
    report: dict | None
    calibration: pd.DataFrame | None
    monitoring: dict | None
    monitoring_periods: pd.DataFrame | None
    monitoring_bins: pd.DataFrame | None
    shap: pd.DataFrame | None
    lstm_sensitivity: pd.DataFrame | None
    feature_drift: pd.DataFrame | None
    drift_investigation: dict | None
    manifest: dict | None
    missing: dict[str, str]


def read_json(path: Path) -> dict | None:
    """Reading JSON never computes or modifies research evidence."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise EvidenceError(f"Unreadable saved JSON {path.name}") from exc
    if not isinstance(data, dict):
        raise EvidenceError(f"{path.name} must contain a JSON object")
    return data


def read_table(path: Path, required: tuple[str, ...] = ()) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        result = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise EvidenceError(f"Cannot read saved table {path.name}") from exc
    if result.empty or not set(required).issubset(result.columns):
        raise EvidenceError(f"{path.name}: empty or missing columns: {sorted(set(required)-set(result.columns))}")
    return result


def read_saved_predictions(path: Path) -> pd.DataFrame | None:
    """Accept only frozen 433-row 2025–Sep 2026 saved outcome CSV, never rescore."""
    if not path.is_file():
        return None
    try:
        raw = pd.read_csv(path, dtype="string", encoding="utf-8-sig")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise EvidenceError("Could not read frozen saved prediction CSV") from exc
    if tuple(raw.columns) != PREDICTION_COLUMNS:
        raise EvidenceError("Frozen prediction CSV has an unexpected schema; refusing to guess columns")
    dates = pd.to_datetime(raw["date"], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any() or not dates.dt.strftime("%Y-%m-%d").eq(raw["date"]).all():
        raise EvidenceError("Saved prediction dates are invalid or not ISO YYYY-MM-DD")
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise EvidenceError("Saved prediction dates must be sorted and unique")
    if (dates < pd.Timestamp("2025-01-01")).any() or (dates >= pd.Timestamp("2026-10-01")).any():
        raise EvidenceError("Saved prediction dates exceed the original sealed holdout")
    yr = pd.to_numeric(raw["validation_year"], errors="coerce")
    if yr.isna().any() or not np.array_equal(dates.dt.year.to_numpy(), yr.to_numpy(dtype=int)):
        raise EvidenceError("Saved validation year does not match the calendar date")
    y = pd.to_numeric(raw["actual_downside_event"], errors="coerce")
    if y.isna().any() or not np.isin(y.to_numpy(dtype=float), [0, 1]).all():
        raise EvidenceError("Saved outcomes are not binary 0/1 values")
    if len(raw) != 433 or int(y.sum()) != 42 or set(dates.dt.year) != {2025, 2026}:
        raise EvidenceError("Saved holdout does not match frozen 433-row / 42-event study")
    for col in PROBABILITY_COLUMNS.values():
        values = pd.to_numeric(raw[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise EvidenceError(f"Saved probability column {col} is invalid")
        raw[col] = values
    raw["date"] = dates
    raw["validation_year"] = yr.astype(int)
    raw["actual_downside_event"] = y.astype(int)
    return raw


def model_comparison(report: dict | None) -> pd.DataFrame | None:
    """Read scores FROM the saved report; do not calculate new holdout scores."""
    if report is None:
        return None
    if report.get("eligible_holdout_rows") != 433:
        raise EvidenceError("Saved report does not identify the frozen 433-row holdout")
    if report.get("selected_before_opening") != "fixed_xgboost_11":
        raise EvidenceError("Saved report disagrees with the original model champion")
    result = []
    models = report.get("model_scores", {})
    for model in MODEL_NAMES:
        item = models.get(model)
        if not isinstance(item, dict):
            raise EvidenceError(f"Model score missing: {model}")
        row = {"Model": MODEL_NAMES[model]}
        for k, label in METRIC_NAMES.items():
            try:
                v = float(item[k])
            except (KeyError, TypeError, ValueError) as exc:
                raise EvidenceError(f"Bad saved metric: {model}/{k}") from exc
            if not np.isfinite(v) or abs(v - RECORDED_METRICS[model][k]) > 0.000001:
                raise EvidenceError(f"Saved {model} {k} differs from authoritative freeze")
            row[label] = v
        result.append(row)
    return pd.DataFrame(result)


def recorded_comparison() -> pd.DataFrame:
    """Exposed documented results, shown ONLY with recorded-not-local warning."""
    return pd.DataFrame([{"Model": MODEL_NAMES[m], **{
        label: RECORDED_METRICS[m][key] for key, label in METRIC_NAMES.items()
    }} for m in MODEL_NAMES])


def load_evidence(project_root: Path) -> SavedEvidence:
    """Load only documented saved paths under the chosen project root."""
    root = Path(project_root).resolve()
    paths = {key: root / rel for key, rel in SAVED_FILES.items()}
    missing = {k: str(v.relative_to(root)) for k, v in paths.items() if not v.is_file()}
    values = dict(
        predictions=read_saved_predictions(paths["predictions"]),
        report=read_json(paths["report"]),
        calibration=read_table(paths["calibration"],
                               ("model", "mean_probability", "event_frequency")),
        monitoring=read_json(paths["monitoring"]),
        monitoring_periods=read_table(paths["monitoring_periods"]),
        monitoring_bins=read_table(paths["monitoring_bins"]),
        shap=read_table(paths["shap"], ("feature", "mean_absolute_shap_log_odds")),
        lstm_sensitivity=read_table(paths["lstm_sensitivity"],
                                    ("feature", "mean_brier_increase")),
        feature_drift=read_table(paths["feature_drift"],
                                 ("feature", "population_stability_index")),
        drift_investigation=read_json(paths["drift_investigation"]),
        manifest=read_json(paths["manifest"]),
        missing=missing,
    )
    if values["report"] is not None:
        model_comparison(values["report"])
    return SavedEvidence(**values)
