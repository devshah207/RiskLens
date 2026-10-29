"""Expanding-window, development-only validation with fixed existing models.

Five folds: train all data strictly before year Y, predict on year Y, for
Y = 2020, 2021, 2022, 2023, 2024. 2025+ prices are not used for any
feature, target, fit or evaluation. Each fold is fitted independently.

This is *development research*, not a final untouched assessment. Results
from the 2023-2024 development period have already been inspected before.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from risklens.evaluation.splits import FOLD_YEARS, FINAL_HOLDOUT_START
from risklens.features.dataset import FeatureDatasetError, build_development_features
from risklens.models.baselines import ModelInputError, compare_models

PREDICTIONS_FILENAME = "walk_forward_validation_predictions.csv"
REPORT_FILENAME = "walk_forward_validation_report.json"


class WalkForwardError(ValueError):
    """Invalid source, fold data, or requested output operation."""


def walk_forward_validation(source_csv: str | Path) -> tuple[pd.DataFrame, dict]:
    """Rebuild features and refit the *same fixed models* for every yearly fold.

    Uses the existing supervised feature builder to purge labels across each
    train/validation boundary. Uses the existing baseline and Logistic
    Regression implementation (including per-fold training-only StandardScaler).

    No features/labels or model scores are created for 2025+ observations.
    """
    source_path = Path(source_csv)
    if not source_path.is_file():
        raise WalkForwardError(f"Combined price CSV not found: {source_path}")

    fold_results: list[dict] = []
    all_predictions: list[pd.DataFrame] = []

    # This temporary directory holds only derived, development-only fold CSVs;
    # it is removed automatically after the comparison is complete.
    with TemporaryDirectory() as scratch:
        root = Path(scratch)
        for year in FOLD_YEARS:
            validation_start = f"{year}-01-01"
            validation_end = f"{year + 1}-01-01"
            # Rebuild for each fold, sealing everything on or after its end.
            # The final fold ends at 2025-01-01; no fold examines 2025+.
            try:
                training, validation, dataset_audit = build_development_features(
                    source_path,
                    validation_start=validation_start,
                    holdout_start=validation_end,
                )
            except FeatureDatasetError as exc:
                raise WalkForwardError(
                    f"Fold {year} feature preparation failed: {exc}"
                ) from exc

            # The existing compare_models() safely reads these temporary tables,
            # fits on training only, and scores validation only. No new model
            # logic or experimental hyperparameters are introduced here.
            train_path = root / f"training_before_{year}.csv"
            valid_path = root / f"validation_{year}.csv"
            training_export = training.copy()
            validation_export = validation.copy()
            for frame in (training_export, validation_export):
                frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
            training_export.to_csv(train_path, index=False)
            validation_export.to_csv(valid_path, index=False)
            try:
                predictions, model_audit = compare_models(
                    train_path,
                    valid_path,
                    validation_start=validation_start,
                    holdout_start=validation_end,
                )
            except ModelInputError as exc:
                raise WalkForwardError(f"Fold {year} modeling failed: {exc}") from exc

            # Only fixed model scores, prediction probabilities and explicit
            # training-only fitted parameters are recorded in our new report.
            predictions.insert(0, "validation_year", year)
            all_predictions.append(predictions)
            fold_results.append(
                {
                    "validation_year": year,
                    "training_range": {
                        "first_date": model_audit["training"]["first_date"],
                        "last_date": model_audit["training"]["last_date"],
                        "rows": model_audit["training"]["rows"],
                        "events": model_audit["training"]["events"],
                        "event_rate": model_audit["training"]["event_rate"],
                    },
                    "validation_range": {
                        "first_date": model_audit["validation"]["first_date"],
                        "last_date": model_audit["validation"]["last_date"],
                        "rows": model_audit["validation"]["rows"],
                        "events": model_audit["validation"]["events"],
                        "event_rate": model_audit["validation"]["event_rate"],
                    },
                    "training_boundary_labels_purged": dataset_audit[
                        "training_boundary_labels_purged"
                    ],
                    "validation_last_date_unlabeled": dataset_audit[
                        "last_development_price_date_has_no_label"
                    ],
                    "training_only_scaler_means": model_audit["models"][
                        "logistic_regression"
                    ]["scaler_mean"],
                    "validation_scores": model_audit["validation_scores"],
                }
            )

    full = pd.concat(all_predictions, ignore_index=True)
    full = full.sort_values("date").reset_index(drop=True)
    if (
        full["date"].duplicated().any()
        or (full["date"] >= pd.Timestamp(FINAL_HOLDOUT_START)).any()
    ):
        raise WalkForwardError(
            "Internal fold date overlap or final holdout contamination"
        )

    # Pooled probability scores summarize performance over the annual folds.
    # ROC-AUC / AP across pooled folds are intentionally omitted because
    # the probabilities come from different independently fitted models,
    # potentially with different baseline class prevalences.
    y = full["actual_downside_event"].to_numpy(dtype=int)
    pooled = {}
    for model, col in (
        ("constant_baseline", "baseline_probability"),
        ("logistic_regression", "logistic_probability"),
    ):
        p = full[col].to_numpy(dtype=float)
        eps = np.finfo(float).eps
        clipped = np.clip(p, eps, 1 - eps)
        pooled[model] = {
            "brier_score": round(float(np.mean((p - y) ** 2)), 8),
            "log_loss": round(
                float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))), 8
            ),
        }

    report = {
        "scope": "Development-only expanding-window experiments; final 2025+ holdout sealed",
        "source_filename": source_path.name,
        "source_sha256": sha256(source_path.read_bytes()).hexdigest(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "validation_years": list(FOLD_YEARS),
        "final_holdout_start": FINAL_HOLDOUT_START,
        "feature_columns": ["return_1d", "return_5d", "volatility_20d"],
        "target": "next observed session close-to-close decline <= -1%",
        "model_configuration": "Unchanged Phase 3.2 training-only constant baseline and StandardScaler + L2 LogisticRegression (C=1, lbfgs, max_iter=2000, class_weight=None)",
        "procedure": "At each year Y: train strictly before Y (excluding cross-boundary labels), evaluate eligible dates in Y only, refit models and scaler from scratch",
        "folds": fold_results,
        "total_validation_rows": int(len(full)),
        "total_validation_events": int(y.sum()),
        "pooled_probability_scores": pooled,
        "warnings": [
            "These are repeated *development* validation results; earlier 2023-2024 outcomes were already inspected in Phase 3.2.",
            "Validation outcomes are used ONLY to evaluate each fold; each fold trains on earlier observations.",
            "Once a year is past, its observations may become training data in the NEXT fold.",
            "No per-fold tuning, feature selection, or threshold optimization was performed.",
            "Yearly ROC-AUC and average precision may be noisy when few downside events occur.",
            "The final 2025+ holdout remains unexamined and is not a component of these scores.",
            "The input series is structurally checked; official trading calendar and NSE authenticity are not independently verified.",
        ],
    }
    return full, report


def save_walk_forward_validation(
    source_csv: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict:
    """Save fold-by-fold validation probabilities and an audit JSON."""
    destination = Path(output_dir)
    targets = [destination / PREDICTIONS_FILENAME, destination / REPORT_FILENAME]
    if not overwrite and any(p.exists() for p in targets):
        raise WalkForwardError(
            "Walk-forward output already exists; choose another directory or use --overwrite"
        )
    predictions, report = walk_forward_validation(source_csv)
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
