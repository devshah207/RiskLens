"""Compare fixed XGBoost with 6 vs 9 historical NIFTY features, fairly.

The 2020-2024 development folds, 2025+ holdout cutoff and XGBoost settings
are predeclared. Both model variants are trained and tested on IDENTICAL rows.
We never use the validation labels for fitting and never examine holdout returns.
"""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from risklens.features.dataset import (
    EXPANDED_FEATURE_COLUMNS, FULL_FEATURE_COLUMNS, TARGET_COLUMN,
    FeatureDatasetError, build_development_features,
)
from risklens.models.baselines import _metrics
from risklens.models.ohlc_comparison import _pooled_probability_scores
from risklens.models.walk_forward import FOLD_YEARS, FINAL_HOLDOUT_START
from risklens.models.xgboost_walk_forward import (
    FIXED_XGB_PARAMS, _fit_predict_fixed_xgb, XGBComparisonError,
)

PREDICTIONS_FILENAME = "technical_feature_comparison_predictions.csv"
REPORT_FILENAME = "technical_feature_comparison_report.json"


class TechnicalComparisonError(ValueError):
    """A market-data input, comparison rule or output check failed."""


def compare_technical_features(source_csv: str | Path) -> tuple[pd.DataFrame, dict]:
    """Compare the prior six features with those six plus RSI/EMA/10d vol.

    A fair 6-vs-9 comparison: the 50-day EMA warm-up is applied to BOTH
    versions, and all folds share the same labels, dates and fixed XGB settings.
    This is repeated development analysis, not an independent test.
    """
    source = Path(source_csv)
    if not source.is_file():
        raise TechnicalComparisonError(f"NIFTY OHLC CSV not found: {source}")
    folds = []
    results = []
    for year in FOLD_YEARS:
        start = f"{year}-01-01"
        end = f"{year+1}-01-01"
        try:
            train, valid, audit = build_development_features(
                source, validation_start=start, holdout_start=end,
                include_ohlc_features=True, include_technical_features=True,
            )
        except FeatureDatasetError as exc:
            raise TechnicalComparisonError(f"Fold {year}: {exc}") from exc
        if (train["date"].ge(start).any() or valid["date"].lt(start).any()
                or valid["date"].ge(end).any()):
            raise TechnicalComparisonError(f"Fold {year}: unexpected date boundary")
        if audit["training_boundary_labels_purged"] != 1:
            raise TechnicalComparisonError(f"Fold {year}: expected 1-session label purge")
        if train["date"].max() >= valid["date"].min():
            raise TechnicalComparisonError(f"Fold {year}: overlapping train/validation")

        y_train = train[TARGET_COLUMN].to_numpy(dtype=int)
        y_val = valid[TARGET_COLUMN].to_numpy(dtype=int)
        baseline = np.full(len(valid), float(y_train.mean()), dtype=float)
        probs = {}
        fit = {}
        scores = {"constant_baseline": _metrics(y_val, baseline)}
        for label, columns in (
            ("xgboost_6_features", EXPANDED_FEATURE_COLUMNS),
            ("xgboost_9_features", FULL_FEATURE_COLUMNS),
        ):
            try:
                probs[label], fit[label] = _fit_predict_fixed_xgb(train, valid, columns)
            except XGBComparisonError as exc:
                raise TechnicalComparisonError(f"Fold {year} {label}: {exc}") from exc
            scores[label] = _metrics(y_val, probs[label])
        results.append(pd.DataFrame({
            "validation_year": year,
            "date": valid["date"],
            "actual_downside_event": y_val,
            "baseline_probability": baseline,
            "xgboost_6_features_probability": probs["xgboost_6_features"],
            "xgboost_9_features_probability": probs["xgboost_9_features"],
        }))
        folds.append({
            "validation_year": year,
            "training_first_date": train["date"].min().date().isoformat(),
            "training_last_date": train["date"].max().date().isoformat(),
            "training_rows": int(len(train)),
            "training_events": int(y_train.sum()),
            "validation_first_date": valid["date"].min().date().isoformat(),
            "validation_last_date": valid["date"].max().date().isoformat(),
            "validation_rows": int(len(valid)),
            "validation_events": int(y_val.sum()),
            "boundary_labels_purged": audit["training_boundary_labels_purged"],
            "feature_warmup_price_rows": audit["technical_feature_warmup_price_rows"],
            "scores": scores,
            "fit_audit": fit,
        })

    combined = pd.concat(results, ignore_index=True).sort_values("date").reset_index(drop=True)
    if combined["date"].duplicated().any() or (combined["date"] >= pd.Timestamp(FINAL_HOLDOUT_START)).any():
        raise TechnicalComparisonError("Date overlap or 2025+ final holdout contamination")
    col_map = {
        "constant_baseline": "baseline_probability",
        "xgboost_6_features": "xgboost_6_features_probability",
        "xgboost_9_features": "xgboost_9_features_probability",
    }
    pooled = {name: _pooled_probability_scores(combined, col) for name, col in col_map.items()}
    report = {
        "scope": "Development-only, 2020-2024 walk-forward; 2025+ outcomes not used",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_filename": source.name,
        "source_sha256": sha256(source.read_bytes()).hexdigest(),
        "holdout_start": FINAL_HOLDOUT_START,
        "fold_years": list(FOLD_YEARS),
        "six_features": list(EXPANDED_FEATURE_COLUMNS),
        "nine_features": list(FULL_FEATURE_COLUMNS),
        "new_features": {
            "rsi_14": "Wilder RSI using 14 returns; first mean gains/losses seed subsequent smoothed means",
            "ema_50_distance": "close / pandas EWM(span=50, adjust=False, min_periods=50) - 1",
            "volatility_10d": "trailing 10-return sample std * sqrt(252)",
        },
        "fixed_xgb_parameters": FIXED_XGB_PARAMS,
        "same_training_and_validation_rows_both_variants": True,
        "total_validation_rows": int(len(combined)),
        "total_validation_events": int(combined["actual_downside_event"].sum()),
        "pooled_probability_scores": pooled,
        "yearly_folds": folds,
        "warnings": [
            "These are exploratory development results; the 2020-2024 outcomes have already been inspected repeatedly.",
            "Both versions use the same dates and label outcomes, after the 50-price EMA warm-up.",
            "The 6-feature scores in THIS report are not identical to previous 6-feature scores because early training rows have been removed for parity.",
            "Indicator values depend on the documented RSI/EMA initialization conventions; alternative implementations may differ.",
            "The feature set is a research hypothesis, not evidence that overbought conditions predict a next-day decline.",
            "No fitting, selection, or evaluation uses 2025+ outcomes. Training is always strictly earlier than each validation year.",
        ],
    }
    return combined, report


def save_technical_comparison(source_csv: str | Path, output_dir: str | Path, *, overwrite: bool = False) -> dict:
    """Write private predictions and audit report, refusing overwrite by default."""
    out = Path(output_dir)
    targets = [out / PREDICTIONS_FILENAME, out / REPORT_FILENAME]
    if not overwrite and any(p.exists() for p in targets):
        raise TechnicalComparisonError("Outputs exist; choose another directory or use --overwrite")
    predictions, report = compare_technical_features(source_csv)
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as temporary:
        stage = Path(temporary)
        export = predictions.copy()
        export["date"] = export["date"].dt.strftime("%Y-%m-%d")
        export.to_csv(stage / PREDICTIONS_FILENAME, index=False, float_format="%.12g")
        (stage / REPORT_FILENAME).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for path in targets:
            (stage / path.name).replace(path)
    return report
