"""Leakage-audited 9 vs 11-feature XGBoost development comparison (2020-2024).

Inputs: validated NIFTY source OHLC CSV + the PRE-2025 output from Step 3.9.
The two fixed-param XGBoost models see identical train/validation rows and labels.
The 2025+ final holdout is never analyzed or predicted in this module.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from risklens.data.validation import OHLCValidationError, load_ohlc_csv
from risklens.features.dataset import (
    FULL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    FeatureDatasetError,
    build_development_features,
)
from risklens.evaluation.metrics import _metrics
from risklens.evaluation.metrics import _pooled_probability_scores
from risklens.evaluation.splits import FOLD_YEARS, FINAL_HOLDOUT_START
from risklens.models.xgboost_walk_forward import (
    EXTERNAL_FEATURE_COLUMNS,
    MULTIFACTOR_FEATURE_COLUMNS,
    FIXED_XGB_PARAMS,
    XGBComparisonError,
    _fit_predict_fixed_xgb,
)

PREDICTIONS_FILENAME = "multifactor_comparison_predictions.csv"
REPORT_FILENAME = "multifactor_comparison_report.json"
# Backwards-compatible names retained for existing research scripts and tests.
from risklens.evaluation.integrated_validation import (
    MAX_EXTERNAL_AGE_DAYS,
    MultiFactorComparisonError,
    _strict_iso_dates,
    _validate_integrated,
)


def compare_multifactor_features(
    nifty_csv: str | Path,
    external_csv: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """Identical fixed XGBoost models on identical training/validation samples.

    Only the 9-vs-11 feature input differs. No hyperparameter/feature selection
    based on development results; 2025+ remains unopened.
    """
    external, external_audit = _validate_integrated(nifty_csv, external_csv)
    frames = []
    fold_audits = []
    for year in FOLD_YEARS:
        start, end = f"{year}-01-01", f"{year + 1}-01-01"
        try:
            train, validation, dataset_audit = build_development_features(
                nifty_csv,
                validation_start=start,
                holdout_start=end,
                include_ohlc_features=True,
                include_technical_features=True,
            )
        except FeatureDatasetError as exc:
            raise MultiFactorComparisonError(f"Fold {year}: {exc}") from exc
        train = train.merge(
            external, on="date", how="left", validate="one_to_one", sort=False
        )
        validation = validation.merge(
            external, on="date", how="left", validate="one_to_one", sort=False
        )
        train_before, valid_before = len(train), len(validation)
        train = train.loc[
            train[list(EXTERNAL_FEATURE_COLUMNS)].notna().all(axis=1)
        ].reset_index(drop=True)
        validation = validation.loc[
            validation[list(EXTERNAL_FEATURE_COLUMNS)].notna().all(axis=1)
        ].reset_index(drop=True)
        if train.empty or validation.empty:
            raise MultiFactorComparisonError(
                f"Fold {year}: no usable rows after external-data alignment"
            )
        if (
            (train["date"] >= pd.Timestamp(start)).any()
            or (validation["date"] < pd.Timestamp(start)).any()
            or (validation["date"] >= pd.Timestamp(end)).any()
        ):
            raise MultiFactorComparisonError(
                f"Fold {year}: partition boundary violated"
            )
        if train["date"].max() >= validation["date"].min():
            raise MultiFactorComparisonError(
                f"Fold {year}: overlapping train/validation dates"
            )
        if dataset_audit["training_boundary_labels_purged"] != 1:
            raise MultiFactorComparisonError(
                f"Fold {year}: expected a single purged training boundary label"
            )
        y_train = train[TARGET_COLUMN].to_numpy(dtype=int)
        y_valid = validation[TARGET_COLUMN].to_numpy(dtype=int)
        if len(np.unique(y_train)) != 2:
            raise MultiFactorComparisonError(
                f"Fold {year}: training data missing event/non-event class"
            )
        baseline = np.full(len(validation), float(y_train.mean()), dtype=float)
        try:
            nine, nine_fit = _fit_predict_fixed_xgb(
                train, validation, FULL_FEATURE_COLUMNS
            )
            eleven, eleven_fit = _fit_predict_fixed_xgb(
                train, validation, MULTIFACTOR_FEATURE_COLUMNS
            )
        except XGBComparisonError as exc:
            raise MultiFactorComparisonError(
                f"Fold {year}: XGBoost fitting failed: {exc}"
            ) from exc
        frames.append(
            pd.DataFrame(
                {
                    "validation_year": year,
                    "date": validation["date"],
                    "actual_downside_event": y_valid,
                    "baseline_probability": baseline,
                    "xgboost_9_features_probability": nine,
                    "xgboost_11_features_probability": eleven,
                }
            )
        )
        fold_audits.append(
            {
                "year": year,
                "training_before": start,
                "validation_before": end,
                "train_rows_before_external_filter": train_before,
                "train_rows_evaluated": len(train),
                "train_rows_dropped_missing_external": train_before - len(train),
                "train_events": int(y_train.sum()),
                "validation_rows_before_external_filter": valid_before,
                "validation_rows_evaluated": len(validation),
                "validation_rows_dropped_missing_external": valid_before
                - len(validation),
                "validation_events": int(y_valid.sum()),
                "last_training_prediction_date": train["date"]
                .iloc[-1]
                .date()
                .isoformat(),
                "first_validation_prediction_date": validation["date"]
                .iloc[0]
                .date()
                .isoformat(),
                "purged_cross_boundary_training_labels": dataset_audit[
                    "training_boundary_labels_purged"
                ],
                "models": {
                    "baseline": _metrics(y_valid, baseline),
                    "xgboost_9_features": _metrics(y_valid, nine),
                    "xgboost_11_features": _metrics(y_valid, eleven),
                },
                "fit_audit": {
                    "xgboost_9_features": nine_fit,
                    "xgboost_11_features": eleven_fit,
                },
            }
        )
    combined = (
        pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    )
    if (
        combined["date"].duplicated().any()
        or (combined["date"] >= pd.Timestamp(FINAL_HOLDOUT_START)).any()
    ):
        raise MultiFactorComparisonError("Fold dates overlap or holdout contamination")
    pooled = {
        "constant_baseline": _pooled_probability_scores(
            combined, "baseline_probability"
        ),
        "xgboost_9_features": _pooled_probability_scores(
            combined, "xgboost_9_features_probability"
        ),
        "xgboost_11_features": _pooled_probability_scores(
            combined, "xgboost_11_features_probability"
        ),
    }
    report = {
        "scope": "Repeated development-only 2020-2024 walk-forward; FINAL 2025+ HOLDOUT SEALED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "holdout_start": FINAL_HOLDOUT_START,
        "prediction_time": "After each NIFTY trading session closes",
        "prediction_target": "Next observed trading-session close-to-close decline of at least 1%",
        "fold_years": list(FOLD_YEARS),
        "nine_features": list(FULL_FEATURE_COLUMNS),
        "eleven_features": list(MULTIFACTOR_FEATURE_COLUMNS),
        "external_feature_meaning": {
            "india_vix_level": "India VIX index level at most recent available earlier source date",
            "sp500_previous_return_1d": "Close-to-close return for latest completed US session with strictly earlier source date",
        },
        "fixed_xgboost_parameters": FIXED_XGB_PARAMS,
        "training_preprocessing": "No scaling, no imputation; drop unavailable external rows symmetrically in both models",
        "prior_date_only_external_matches": True,
        "validation_used_for_training_or_early_stopping": False,
        "hyperparameter_search": False,
        "source_audit": external_audit,
        "total_validation_rows": len(combined),
        "total_validation_events": int(combined["actual_downside_event"].sum()),
        "pooled_probability_scores": pooled,
        "yearly_folds": fold_audits,
        "warnings": [
            "This is exploratory repeated development validation; prior outcomes have already been inspected.",
            "An observed development improvement is NOT proof of out-of-sample performance.",
            "External inputs use a conservative strict prior-date rule; actual publication timestamps unverified.",
            "Existing XGBoost tree settings and train/validation years are unchanged; no tuning performed.",
            "All models share the exact same training and validation rows after missing external data exclusion.",
            "2025+ NIFTY outcomes and external features were not accessed, computed or evaluated.",
            "Original source market prices were structurally validated but not independently authenticated.",
        ],
    }
    return combined, report


def save_multifactor_comparison(
    nifty_csv: str | Path,
    external_csv: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict:
    """Write audit and predictions, preventing accidental replacement by default."""
    out = Path(output_dir)
    targets = (out / PREDICTIONS_FILENAME, out / REPORT_FILENAME)
    if not overwrite and any(x.exists() for x in targets):
        raise MultiFactorComparisonError(
            "Outputs already exist; choose another output directory or --overwrite"
        )
    predictions, report = compare_multifactor_features(nifty_csv, external_csv)
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as tmp:
        temp = Path(tmp)
        export = predictions.copy()
        export["date"] = export["date"].dt.strftime("%Y-%m-%d")
        export.to_csv(temp / PREDICTIONS_FILENAME, index=False, float_format="%.12g")
        (temp / REPORT_FILENAME).write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        for item in targets:
            (temp / item.name).replace(item)
    return report
