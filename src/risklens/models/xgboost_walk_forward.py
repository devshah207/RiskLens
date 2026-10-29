"""Fixed-configuration XGBoost comparison on the EXISTING development folds.

For each year 2020..2024, use only earlier dated observations for fitting.
The 2025+ holdout is never used in features, fitting, predictions or scores.
This is *repeated development analysis*, not an untouched performance claim.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import xgboost

from risklens.features.dataset import (
    EXPANDED_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    FULL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    FeatureDatasetError,
    build_development_features,
)
from risklens.evaluation.metrics import _metrics, _pooled_probability_scores
from risklens.evaluation.features import (
    EXTERNAL_FEATURE_COLUMNS,
    MULTIFACTOR_FEATURE_COLUMNS,
)
from risklens.models.ohlc_comparison import (
    OHLCComparisonError,
    compare_ohlc_features,
)
from risklens.evaluation.splits import FOLD_YEARS, FINAL_HOLDOUT_START


PREDICTIONS_FILENAME = "xgboost_walk_forward_predictions.csv"
REPORT_FILENAME = "xgboost_walk_forward_report.json"

# CHOSEN BEFORE LOOKING AT ANY XGBOOST VALIDATION RESULTS. NO PARAMETER SEARCH.
# Intentionally modest capacity for ~1k-2k training rows and rare events.
FIXED_XGB_PARAMS = {
    "n_estimators": 80,
    "max_depth": 2,
    "learning_rate": 0.04,
    "min_child_weight": 10,
    "gamma": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "reg_lambda": 10.0,
    "reg_alpha": 0.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": 1,
    "scale_pos_weight": 1.0,
    "verbosity": 0,
}


class XGBComparisonError(ValueError):
    """Invalid input, unexpected fold alignment or output destination."""


def _fit_predict_fixed_xgb(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: tuple[str, ...],
) -> tuple[np.ndarray, dict]:
    """Fit a fresh XGBoost classifier using train ONLY; never pass validation labels.

    Tree splits depend on feature ordering, not linear feature scale; no scaler.
    There is deliberately no early stopping or eval_set, as that would use the
    reported validation fold for a training-duration decision.
    """
    if features not in (
        FEATURE_COLUMNS,
        EXPANDED_FEATURE_COLUMNS,
        FULL_FEATURE_COLUMNS,
        MULTIFACTOR_FEATURE_COLUMNS,
    ):
        raise XGBComparisonError(
            "Only the original 3 or predefined 6, 9 or 11 feature sets are supported"
        )
    x_train = train.loc[:, list(features)].to_numpy(dtype=float)
    y_train = train[TARGET_COLUMN].to_numpy(dtype=int)
    x_valid = validation.loc[:, list(features)].to_numpy(dtype=float)
    if (not np.isfinite(x_train).all()) or (not np.isfinite(x_valid).all()):
        raise XGBComparisonError("All model features must be finite")
    if len(np.unique(y_train)) != 2:
        raise XGBComparisonError("Each training fold must contain both label classes")
    model = XGBClassifier(**FIXED_XGB_PARAMS)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_valid)[:, 1]
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise XGBComparisonError("XGBoost returned invalid probabilities")
    # Feature importance here is normalized *gain*, a descriptive measure of
    # fitted splits, NOT causality or guaranteed future predictive relevance.
    gains = model.get_booster().get_score(importance_type="gain")
    gain_named = {
        name: float(gains.get(f"f{i}", 0.0)) for i, name in enumerate(features)
    }
    return probabilities, {
        "train_rows": int(len(train)),
        "train_events": int(y_train.sum()),
        "validation_rows": int(len(validation)),
        "trees_fitted": int(model.get_booster().num_boosted_rounds()),
        "feature_importance_gain": gain_named,
    }


def compare_xgboost_walk_forward(source_csv: str | Path) -> tuple[pd.DataFrame, dict]:
    """Compare fixed XGB (3 and 6 features) with frozen existing references.

    Comparisons use identical yearly validation dates and target labels across
    ALL models; only 2016-2024 development outcomes are examined.
    """
    source = Path(source_csv)
    if not source.is_file():
        raise XGBComparisonError(f"Combined price CSV not found: {source}")
    try:
        reference, reference_report = compare_ohlc_features(source)
    except (OHLCComparisonError, OSError) as exc:
        raise XGBComparisonError(f"Prior-model reference failed: {exc}") from exc

    all_folds = []
    fold_reports = []
    for year in FOLD_YEARS:
        validation_start = f"{year}-01-01"
        validation_end = f"{year + 1}-01-01"
        expected = reference.loc[reference["validation_year"] == year].reset_index(
            drop=True
        )
        yearly = expected.copy()
        fold_summary = {
            "year": year,
            "training_before": validation_start,
            "validation_before": validation_end,
            "validation_rows": len(expected),
            "validation_events": int(expected["actual_downside_event"].sum()),
            "scores": {
                "constant_baseline": _metrics(
                    expected["actual_downside_event"].to_numpy(dtype=int),
                    expected["baseline_probability"].to_numpy(dtype=float),
                ),
                "logistic_3_features": _metrics(
                    expected["actual_downside_event"].to_numpy(dtype=int),
                    expected["original_3feature_probability"].to_numpy(dtype=float),
                ),
                "logistic_6_features": _metrics(
                    expected["actual_downside_event"].to_numpy(dtype=int),
                    expected["expanded_6feature_probability"].to_numpy(dtype=float),
                ),
            },
            "xgboost_fit_audit": {},
        }
        for features, label, expanded in (
            (FEATURE_COLUMNS, "xgboost_3_features", False),
            (EXPANDED_FEATURE_COLUMNS, "xgboost_6_features", True),
        ):
            # Reuse the boundary-purging feature builder; no new target logic.
            try:
                train, valid, dataset_audit = build_development_features(
                    source,
                    validation_start=validation_start,
                    holdout_start=validation_end,
                    include_ohlc_features=expanded,
                )
            except FeatureDatasetError as exc:
                raise XGBComparisonError(
                    f"{year}: feature preparation failed: {exc}"
                ) from exc

            # A strict controlled comparison: same dates, same true labels.
            if not valid["date"].reset_index(drop=True).equals(expected["date"]):
                raise XGBComparisonError(
                    f"{year}: different validation dates in {label}"
                )
            if not np.array_equal(
                valid[TARGET_COLUMN].to_numpy(dtype=int),
                expected["actual_downside_event"].to_numpy(dtype=int),
            ):
                raise XGBComparisonError(
                    f"{year}: different validation labels in {label}"
                )
            if not (train["date"] < pd.Timestamp(validation_start)).all():
                raise XGBComparisonError(
                    f"{year}: train date crossed validation boundary"
                )
            if (
                not valid.empty
                and not train.empty
                and train["date"].max() >= valid["date"].min()
            ):
                raise XGBComparisonError(
                    f"{year}: training and validation dates overlap"
                )
            if dataset_audit["training_boundary_labels_purged"] != 1:
                raise XGBComparisonError(f"{year}: expected one cross-year label purge")
            probabilities, fit_audit = _fit_predict_fixed_xgb(train, valid, features)
            yearly[f"{label}_probability"] = probabilities
            fold_summary["scores"][label] = _metrics(
                valid[TARGET_COLUMN].to_numpy(dtype=int), probabilities
            )
            fold_summary["xgboost_fit_audit"][label] = fit_audit
        all_folds.append(yearly)
        fold_reports.append(fold_summary)

    combined = (
        pd.concat(all_folds, ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )
    if (
        combined["date"].duplicated().any()
        or (combined["date"] >= pd.Timestamp(FINAL_HOLDOUT_START)).any()
    ):
        raise XGBComparisonError(
            "Overlapping validation folds or final-holdout contamination"
        )
    columns = {
        "constant_baseline": "baseline_probability",
        "logistic_3_features": "original_3feature_probability",
        "logistic_6_features": "expanded_6feature_probability",
        "xgboost_3_features": "xgboost_3_features_probability",
        "xgboost_6_features": "xgboost_6_features_probability",
    }
    pooled = {
        name: _pooled_probability_scores(combined, col) for name, col in columns.items()
    }
    report = {
        "scope": "Development-only 2020-2024 walk-forward comparisons; final 2025+ holdout sealed",
        "source_filename": source.name,
        "source_sha256": sha256(source.read_bytes()).hexdigest(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "xgboost_version": xgboost.__version__,
        "validation_years": list(FOLD_YEARS),
        "holdout_starts": FINAL_HOLDOUT_START,
        "target": "next trading session close-to-close return <= -1%",
        "features_original": list(FEATURE_COLUMNS),
        "features_expanded": list(EXPANDED_FEATURE_COLUMNS),
        "xgboost_parameters_fixed_before_evaluation": FIXED_XGB_PARAMS,
        "xgboost_no_scaler": True,
        "early_stopping": False,
        "hyperparameter_search": False,
        "total_validation_rows": int(len(combined)),
        "total_validation_events": int(combined["actual_downside_event"].sum()),
        "per_fold_scores": fold_reports,
        "pooled_probability_scores": pooled,
        "previous_reference_scores": reference_report["pooled_probability_scores"],
        "warnings": [
            "Repeated development experiments, not a fresh untouched test; earlier scores informed choice to try a tree model.",
            "These are a fixed illustrative XGBoost configuration, not optimized hyperparameters.",
            "Validation outcomes are NEVER supplied to fitting; each fold is fitted on earlier years only.",
            "No final-holdout rows/features/returns/labels or predictions are computed or evaluated.",
            "Importance measured by fitted split gain is not a causal explanation or proof of predictability.",
            "Scores for individual years are noisy when downside-event counts are small.",
            "Even an apparent improvement requires confirmation; don't choose hyperparameters on these years repeatedly.",
        ],
    }
    return combined, report


def save_xgboost_walk_forward(
    source_csv: str | Path, output_dir: str | Path, *, overwrite: bool = False
) -> dict:
    """Write private development predictions and JSON audit, refusing overwrite by default."""
    output = Path(output_dir)
    paths = [output / PREDICTIONS_FILENAME, output / REPORT_FILENAME]
    if not overwrite and any(p.exists() for p in paths):
        raise XGBComparisonError(
            "Output already exists; choose another folder or use --overwrite"
        )
    predictions, report = compare_xgboost_walk_forward(source_csv)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as tmp:
        stage = Path(tmp)
        exported = predictions.copy()
        exported["date"] = exported["date"].dt.strftime("%Y-%m-%d")
        exported.to_csv(stage / PREDICTIONS_FILENAME, index=False, float_format="%.12g")
        (stage / REPORT_FILENAME).write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        for p in paths:
            (stage / p.name).replace(p)
    return report
