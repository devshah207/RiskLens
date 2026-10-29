"""Controlled walk-forward comparison: original 3 closing-price features vs 6 OHLC features.

The two models use identical train/validation dates, labels, Logistic Regression
settings, and train-only scaling. Only the available predictors differ.
Only years 2020-2024 are evaluated; 2025+ outcomes remain sealed.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from risklens.features.dataset import (
    FeatureDatasetError,
    EXPANDED_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    build_development_features,
)
from risklens.evaluation.metrics import _pooled_probability_scores
from risklens.models.baselines import ModelInputError, compare_models
from risklens.models.walk_forward import (
    FOLD_YEARS,
    FINAL_HOLDOUT_START,
    WalkForwardError,
    walk_forward_validation,
)

PREDICTIONS_FILENAME = "ohlc_feature_comparison_predictions.csv"
REPORT_FILENAME = "ohlc_feature_comparison_report.json"


class OHLCComparisonError(ValueError):
    """Raised when the controlled feature-set comparison cannot be completed."""


def compare_ohlc_features(source_csv: str | Path) -> tuple[pd.DataFrame, dict]:
    """Evaluate the original and expanded feature sets on the SAME five folds.

    A fresh StandardScaler and L2 Logistic Regression are fitted on earlier
    years only for each fold. No hyperparameter search or threshold tuning.
    """
    source_path = Path(source_csv)
    if not source_path.is_file():
        raise OHLCComparisonError(f"Combined price CSV not found: {source_path}")

    # The original pipeline is the reference; its behavior is left unchanged.
    try:
        original, original_report = walk_forward_validation(source_path)
    except (WalkForwardError, OSError) as exc:
        raise OHLCComparisonError(
            f"Original walk-forward validation failed: {exc}"
        ) from exc

    expanded_predictions = []
    fold_reports = []
    with TemporaryDirectory() as scratch:
        scratch_dir = Path(scratch)
        for year in FOLD_YEARS:
            start = f"{year}-01-01"
            end = f"{year + 1}-01-01"
            try:
                train, valid, dataset_audit = build_development_features(
                    source_path,
                    validation_start=start,
                    holdout_start=end,
                    include_ohlc_features=True,
                )
            except FeatureDatasetError as exc:
                raise OHLCComparisonError(
                    f"OHLC feature preparation failed in {year}: {exc}"
                ) from exc

            for label, frame in (("train", train), ("validation", valid)):
                export = frame.copy()
                export["date"] = export["date"].dt.strftime("%Y-%m-%d")
                export.to_csv(scratch_dir / f"{year}_{label}.csv", index=False)
            try:
                predictions, model_audit = compare_models(
                    scratch_dir / f"{year}_train.csv",
                    scratch_dir / f"{year}_validation.csv",
                    validation_start=start,
                    holdout_start=end,
                    feature_columns=EXPANDED_FEATURE_COLUMNS,
                )
            except ModelInputError as exc:
                raise OHLCComparisonError(
                    f"Expanded model fitting failed in {year}: {exc}"
                ) from exc

            expected = original.loc[original["validation_year"] == year].reset_index(
                drop=True
            )
            actual = predictions.reset_index(drop=True)
            if (
                not expected["date"].equals(actual["date"])
                or not expected["actual_downside_event"].equals(
                    actual["actual_downside_event"]
                )
                or not np.allclose(
                    expected["baseline_probability"],
                    actual["baseline_probability"],
                    rtol=1e-12,
                    atol=1e-12,
                )
            ):
                raise OHLCComparisonError(
                    f"Year {year} does not have identical dates, labels, and baseline"
                )

            expanded_predictions.append(
                pd.DataFrame(
                    {
                        "validation_year": year,
                        "date": actual["date"],
                        "actual_downside_event": actual["actual_downside_event"],
                        "original_3feature_probability": expected[
                            "logistic_probability"
                        ],
                        "expanded_6feature_probability": actual["logistic_probability"],
                        "baseline_probability": actual["baseline_probability"],
                    }
                )
            )
            fold_reports.append(
                {
                    "validation_year": year,
                    "training_rows": model_audit["training"]["rows"],
                    "validation_rows": model_audit["validation"]["rows"],
                    "validation_events": model_audit["validation"]["events"],
                    "training_boundary_labels_purged": dataset_audit[
                        "training_boundary_labels_purged"
                    ],
                    "original_3feature_scores": original_report["folds"][
                        len(fold_reports)
                    ]["validation_scores"]["logistic_regression"],
                    "expanded_6feature_scores": model_audit["validation_scores"][
                        "logistic_regression"
                    ],
                    "training_only_scaler_means_expanded": model_audit["models"][
                        "logistic_regression"
                    ]["scaler_mean"],
                }
            )

    combined = (
        pd.concat(expanded_predictions, ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
    if (
        combined["date"].duplicated().any()
        or (combined["date"] >= pd.Timestamp(FINAL_HOLDOUT_START)).any()
    ):
        raise OHLCComparisonError("Internal validation overlap or 2025+ contamination")

    pooled = {
        "constant_baseline": _pooled_probability_scores(
            combined, "baseline_probability"
        ),
        "logistic_3_features": _pooled_probability_scores(
            combined, "original_3feature_probability"
        ),
        "logistic_6_features": _pooled_probability_scores(
            combined, "expanded_6feature_probability"
        ),
    }
    report = {
        "scope": "Development-only controlled 3-vs-6-feature comparison; 2025+ holdout not evaluated",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_filename": source_path.name,
        "source_sha256": sha256(source_path.read_bytes()).hexdigest(),
        "validation_years": list(FOLD_YEARS),
        "final_holdout_start": FINAL_HOLDOUT_START,
        "original_features": list(FEATURE_COLUMNS),
        "expanded_features": list(EXPANDED_FEATURE_COLUMNS),
        "new_feature_definitions": {
            "intraday_range": "(high[t] - low[t]) / open[t]",
            "intraday_return": "close[t] / open[t] - 1",
            "opening_gap": "open[t] / close[t-1] - 1",
        },
        "fixed_model": "StandardScaler + L2 LogisticRegression (C=1.0; lbfgs; max_iter=2000; class_weight=None)",
        "procedure": "Same expanding training windows, same yearly validation samples, scaler fit on training only; no tuning",
        "total_validation_rows": int(len(combined)),
        "total_validation_events": int(combined["actual_downside_event"].sum()),
        "folds": fold_reports,
        "pooled_probability_scores": pooled,
        "warnings": [
            "These results are exploratory development comparisons, not final generalization estimates.",
            "The 2023-2024 development outcomes were examined before this feature experiment; repeated iteration may overfit development validation.",
            "OHLC features are not independent: daily closing return relates to opening gap and intraday return.",
            "No hyperparameter, probability threshold, or model family was changed.",
            "The 2025+ final holdout remains sealed; no predictions or labels were computed for it.",
            "No improvement is guaranteed; market behavior changes and event samples are limited.",
        ],
    }
    return combined, report


def save_ohlc_feature_comparison(
    source_csv: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict:
    """Save predictions and audit report after all validation checks pass."""
    destination = Path(output_dir)
    targets = [destination / PREDICTIONS_FILENAME, destination / REPORT_FILENAME]
    if not overwrite and any(target.exists() for target in targets):
        raise OHLCComparisonError(
            "Output files already exist; choose a different directory or use --overwrite"
        )
    predictions, report = compare_ohlc_features(source_csv)
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination) as tmp:
        stage = Path(tmp)
        export = predictions.copy()
        export["date"] = export["date"].dt.strftime("%Y-%m-%d")
        export.to_csv(stage / PREDICTIONS_FILENAME, index=False, float_format="%.12g")
        (stage / REPORT_FILENAME).write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        for target in targets:
            (stage / target.name).replace(target)
    return report
