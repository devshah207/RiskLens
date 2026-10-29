"""Time-separated sigmoid probability calibration for the FIXED 3-feature XGBoost.

For validation year Y (2020..2024):
  - Fit base XGBoost on dates strictly before Jan 1 of Y-1.
  - Fit a fixed L2 sigmoid calibrator on predictions/outcomes within Y-1.
  - Evaluate raw and calibrated predictions on year Y.
The calibration year is NEVER used to fit the base XGBoost, and the validation
outcomes are NEVER used for fitting or selecting hyperparameters. 2025+ sealed.

This is development research: the years and model family were previously
examined. No untouched generalization or tradability is claimed.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from risklens.features.dataset import (
    FEATURE_COLUMNS, TARGET_COLUMN, FeatureDatasetError, build_development_features,
)
from risklens.models.baselines import _metrics
from risklens.models.ohlc_comparison import _pooled_probability_scores
from risklens.models.walk_forward import FOLD_YEARS, FINAL_HOLDOUT_START
from risklens.models.xgboost_walk_forward import (
    FIXED_XGB_PARAMS, XGBComparisonError, _fit_predict_fixed_xgb,
)

PREDICTIONS_FILENAME = "calibration_walk_forward_predictions.csv"
REPORT_FILENAME = "calibration_walk_forward_report.json"
# Fixed before any calibration results. Numerical guard for exact 0/1 estimates.
PROBABILITY_CLIP = 1e-6
CALIBRATOR_C = 1.0
# Fixed descriptive bins; never optimized based on event frequencies.
RELIABILITY_EDGES = (0.0, 0.05, 0.10, 0.15, 0.20, 1.0)


class CalibrationError(ValueError):
    """Time boundary, dataset, calibration, or output validation failed."""


def _logit(probs: np.ndarray) -> np.ndarray:
    """Transform probabilities p into ln(p/(1-p)), with numerical clipping."""
    p = np.asarray(probs, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise CalibrationError("Probabilities must be a 1D finite vector between 0 and 1")
    p = np.clip(p, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    return np.log(p / (1.0 - p)).reshape(-1, 1)


def _fit_sigmoid_calibrator(
    calibration_probabilities: np.ndarray,
    calibration_targets: np.ndarray,
    validation_probabilities: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Platt-style sigmoid trained ONLY on preceding calibration-year outcomes.

    LogisticRegression(C=1.0) provides L2-regularized slope and intercept.
    Slope can be negative on small/nonstationary samples; this is recorded.
    """
    y = np.asarray(calibration_targets)
    if len(y) != len(calibration_probabilities) or len(y) == 0 or not np.isin(y, [0, 1]).all():
        raise CalibrationError("Calibration probabilities/targets must align and labels must be 0 or 1")
    if len(np.unique(y)) != 2:
        raise CalibrationError("Calibration year must contain both event classes")
    X_cal = _logit(calibration_probabilities)
    X_val = _logit(validation_probabilities)
    cal = LogisticRegression(C=CALIBRATOR_C, solver="lbfgs", max_iter=2000)
    cal.fit(X_cal, y.astype(int))
    result = cal.predict_proba(X_val)[:, 1]
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise CalibrationError("Sigmoid produced invalid probability values")
    slope = float(cal.coef_[0, 0])
    return result, {
        "method": "L2-regularized sigmoid (Platt-style) on clipped XGBoost logits",
        "C_fixed": CALIBRATOR_C,
        "clip_probability_fixed": PROBABILITY_CLIP,
        "slope": slope,
        "intercept": float(cal.intercept_[0]),
        "slope_is_negative": slope < 0,
        "fit_rows": int(len(y)),
        "fit_events": int(y.sum()),
    }


def _reliability_bins(frame: pd.DataFrame, probability_col: str) -> list[dict]:
    """Descriptive pooled risk bins; small-bin frequencies should not be trusted."""
    probabilities = frame[probability_col].to_numpy(dtype=float)
    outcomes = frame["actual_downside_event"].to_numpy(dtype=int)
    result = []
    for i, (lo, hi) in enumerate(zip(RELIABILITY_EDGES[:-1], RELIABILITY_EDGES[1:])):
        mask = (probabilities >= lo) & (probabilities <= hi if i == len(RELIABILITY_EDGES)-2 else probabilities < hi)
        n = int(mask.sum())
        result.append({
            "range": f"[{lo:.2f}, {hi:.2f}{']' if i == len(RELIABILITY_EDGES)-2 else ')'}",
            "observations": n,
            "mean_predicted_probability": round(float(probabilities[mask].mean()), 8) if n else None,
            "observed_event_frequency": round(float(outcomes[mask].mean()), 8) if n else None,
            "small_bin_warning": n < 30,
        })
    return result


def compare_calibration(source_csv: str | Path) -> tuple[pd.DataFrame, dict]:
    """Five ordered (base-train, calibrate, validate) XGB comparisons.

    The constant baseline uses every eligible earlier-than-Y labeled row,
    as in Step 3.3; raw and calibrated XGB share the SAME older base model.
    This paired comparison isolates the effect of the calibration method.
    """
    source = Path(source_csv)
    if not source.is_file():
        raise CalibrationError(f"Combined price CSV not found: {source}")

    fold_predictions = []
    fold_audits = []
    for year in FOLD_YEARS:
        calibration_start = f"{year-1}-01-01"
        validation_start = f"{year}-01-01"
        validation_end = f"{year+1}-01-01"
        try:
            base_train, calibration, base_audit = build_development_features(
                source, validation_start=calibration_start, holdout_start=validation_start
            )
            full_past, validation, validation_audit = build_development_features(
                source, validation_start=validation_start, holdout_start=validation_end
            )
        except FeatureDatasetError as exc:
            raise CalibrationError(f"Year {year}: feature preparation failed: {exc}") from exc

        # Explicitly verify each chronological boundary and label purge.
        if not (base_train["date"] < pd.Timestamp(calibration_start)).all():
            raise CalibrationError(f"Year {year}: base-training date entered calibration year")
        if not ((calibration["date"] >= pd.Timestamp(calibration_start)) &
                (calibration["date"] < pd.Timestamp(validation_start))).all():
            raise CalibrationError(f"Year {year}: calibration year misaligned")
        if not ((validation["date"] >= pd.Timestamp(validation_start)) &
                (validation["date"] < pd.Timestamp(validation_end))).all():
            raise CalibrationError(f"Year {year}: evaluation year misaligned")
        if (base_train["date"].max() >= calibration["date"].min()
                or calibration["date"].max() >= validation["date"].min()
                or base_audit["training_boundary_labels_purged"] != 1
                or validation_audit["training_boundary_labels_purged"] != 1):
            raise CalibrationError(f"Year {year}: missing chronological purge or overlap")
        if (not full_past["date"].max() < validation["date"].min()
                or len(np.unique(base_train[TARGET_COLUMN])) != 2):
            raise CalibrationError(f"Year {year}: invalid baseline or base-training sample")

        to_score = pd.concat([calibration, validation], ignore_index=True)
        try:
            raw, base_fit_audit = _fit_predict_fixed_xgb(base_train, to_score, FEATURE_COLUMNS)
        except XGBComparisonError as exc:
            raise CalibrationError(f"Year {year}: XGBoost failed: {exc}") from exc
        raw_cal = raw[:len(calibration)]
        raw_valid = raw[len(calibration):]
        calibrated, cal_fit_audit = _fit_sigmoid_calibrator(
            raw_cal,
            calibration[TARGET_COLUMN].to_numpy(dtype=int),
            raw_valid,
        )
        y = validation[TARGET_COLUMN].to_numpy(dtype=int)
        # Baseline is available as-of validation year start; no evaluation labels.
        baseline_rate = float(full_past[TARGET_COLUMN].mean())
        baseline_probs = np.full(len(validation), baseline_rate, dtype=float)

        yearly = pd.DataFrame({
            "validation_year": year,
            "date": validation["date"],
            "actual_downside_event": y,
            "baseline_probability": baseline_probs,
            "xgboost_raw_probability": raw_valid,
            "xgboost_calibrated_probability": calibrated,
        })
        fold_predictions.append(yearly)
        fold_audits.append({
            "validation_year": year,
            "base_fit_period": [base_train["date"].min().strftime("%Y-%m-%d"), base_train["date"].max().strftime("%Y-%m-%d")],
            "base_fit_rows": int(len(base_train)),
            "base_fit_events": int(base_train[TARGET_COLUMN].sum()),
            "calibration_period": [calibration["date"].min().strftime("%Y-%m-%d"), calibration["date"].max().strftime("%Y-%m-%d")],
            "calibration_rows": int(len(calibration)),
            "calibration_events": int(calibration[TARGET_COLUMN].sum()),
            "validation_period": [validation["date"].min().strftime("%Y-%m-%d"), validation["date"].max().strftime("%Y-%m-%d")],
            "validation_rows": int(len(validation)),
            "validation_events": int(y.sum()),
            "boundary_purges_checked": {"base_to_calibration": 1, "calibration_to_validation": 1},
            "baseline_train_rows": int(len(full_past)),
            "baseline_train_event_rate": round(baseline_rate, 8),
            "base_model_fit_audit": base_fit_audit,
            "calibrator_fit_audit": cal_fit_audit,
            "validation_scores": {
                "constant_baseline": _metrics(y, baseline_probs),
                "xgboost_raw": _metrics(y, raw_valid),
                "xgboost_calibrated": _metrics(y, calibrated),
            },
        })

    combined = pd.concat(fold_predictions, ignore_index=True).sort_values("date").reset_index(drop=True)
    if combined["date"].duplicated().any() or (combined["date"] >= pd.Timestamp(FINAL_HOLDOUT_START)).any():
        raise CalibrationError("Overlapping yearly validations or final holdout contamination")
    cols = {
        "constant_baseline": "baseline_probability",
        "xgboost_raw": "xgboost_raw_probability",
        "xgboost_calibrated": "xgboost_calibrated_probability",
    }
    pooled = {key: _pooled_probability_scores(combined, column) for key, column in cols.items()}
    report = {
        "scope": "Development only; years 2020-2024 evaluated; 2025+ holdout sealed",
        "source_filename": source.name,
        "source_sha256": sha256(source.read_bytes()).hexdigest(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "validation_years": list(FOLD_YEARS),
        "final_holdout_start": FINAL_HOLDOUT_START,
        "features": list(FEATURE_COLUMNS),
        "xgboost_fixed_parameters": FIXED_XGB_PARAMS,
        "calibrator_fixed_method": "L2 sigmoid on base-model logit; C=1.0; clip=1e-6",
        "procedure": "For Y: XGB fit on dates before Y-1; sigmoid fit on Y-1 only; validation on Y; baseline fits all pre-Y labels",
        "calibration_years": [year-1 for year in FOLD_YEARS],
        "total_validation_rows": int(len(combined)),
        "total_validation_events": int(combined["actual_downside_event"].sum()),
        "folds": fold_audits,
        "pooled_probability_scores": pooled,
        "pooled_reliability_bins": {key: _reliability_bins(combined, col) for key, col in cols.items()},
        "warnings": [
            "Calibration is fitted only on the previous year, not the evaluation year.",
            "Raw and calibrated XGB use the same base fit; this paired comparison isolates calibration.",
            "The historical baseline is trained on all pre-validation-year labels, so it uses more recent data than the XGB base fit.",
            "Raw scores differ from Step 3.5 because XGBoost is intentionally fitted through Y-2, leaving Y-1 for calibration.",
            "Sigmoid slope can be negative if calibration data show an inverse association; this may reverse ranking.",
            "Calibration-year event counts are small and market conditions change; sigmoid fitting can worsen predictions.",
            "Reliability bins are descriptive; bins with fewer than 30 observations are unstable.",
            "2023-2024 development outcomes have been previously inspected; these are NOT untouched generalization results.",
            "No XGBoost or calibrator hyperparameter tuning, model selection, or 2025+ holdout analysis.",
        ],
    }
    return combined, report


def save_calibration_comparison(source_csv: str | Path, output_dir: str | Path, *, overwrite: bool = False) -> dict:
    """Write development-only predictions/report, refusing overwrite by default."""
    destination = Path(output_dir)
    paths = (destination / PREDICTIONS_FILENAME, destination / REPORT_FILENAME)
    if not overwrite and any(path.exists() for path in paths):
        raise CalibrationError("Calibration output already exists; choose another directory or use --overwrite")
    predictions, report = compare_calibration(source_csv)
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination) as tmp:
        stage = Path(tmp)
        exported = predictions.copy()
        exported["date"] = exported["date"].dt.strftime("%Y-%m-%d")
        exported.to_csv(stage / PREDICTIONS_FILENAME, index=False, float_format="%.12g")
        (stage / REPORT_FILENAME).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for path in paths:
            (stage / path.name).replace(path)
    return report
