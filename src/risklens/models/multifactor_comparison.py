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
    FULL_FEATURE_COLUMNS, TARGET_COLUMN, FeatureDatasetError, build_development_features,
)
from risklens.evaluation.metrics import _metrics
from risklens.evaluation.metrics import _pooled_probability_scores
from risklens.evaluation.splits import FOLD_YEARS, FINAL_HOLDOUT_START
from risklens.models.xgboost_walk_forward import (
    EXTERNAL_FEATURE_COLUMNS, MULTIFACTOR_FEATURE_COLUMNS, FIXED_XGB_PARAMS,
    XGBComparisonError, _fit_predict_fixed_xgb,
)

PREDICTIONS_FILENAME = "multifactor_comparison_predictions.csv"
REPORT_FILENAME = "multifactor_comparison_report.json"
MAX_EXTERNAL_AGE_DAYS = 7  # matches Step 3.9 default; no alternate ages tested


class MultiFactorComparisonError(ValueError):
    """Joined data unsafe, predictions non-comparable or output invalid."""


def _strict_iso_dates(series: pd.Series, name: str, *, allow_missing: bool = False) -> pd.Series:
    """Parse ISO dates only, including optional blank source-date cells."""
    strings = series.astype("string")
    missing = strings.isna() | strings.eq("")
    parsed = pd.to_datetime(strings, format="%Y-%m-%d", errors="coerce")
    if ((~missing) & parsed.isna()).any() or (not allow_missing and missing.any()):
        raise MultiFactorComparisonError(f"Invalid or missing {name}; expected YYYY-MM-DD")
    return parsed


def _validate_integrated(nifty_csv: str | Path, external_csv: str | Path) -> tuple[pd.DataFrame, dict]:
    """Check original OHLC equality, external dates, numeric values and readiness.

    DO NOT trust an arbitrary joined CSV simply because it has expected columns.
    All checks refer exclusively to <2025 NIFTY dates. The original NIFTY file
    is only structurally validated on any later dates, with no later features.
    """
    source, merged_path = Path(nifty_csv), Path(external_csv)
    if not source.is_file():
        raise MultiFactorComparisonError(f"NIFTY source not found: {source}")
    if not merged_path.is_file():
        raise MultiFactorComparisonError(f"Step 3.9 integrated CSV not found: {merged_path}")
    try:
        prices = load_ohlc_csv(source)
    except OHLCValidationError as exc:
        raise MultiFactorComparisonError(f"Invalid original NIFTY OHLC: {exc}") from exc
    original = prices.loc[prices["date"] < pd.Timestamp(FINAL_HOLDOUT_START),
                          ["date", "open", "high", "low", "close"]].reset_index(drop=True)
    try:
        merged = pd.read_csv(merged_path, dtype={"external_features_ready": "string"})
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise MultiFactorComparisonError(f"Could not read merged CSV: {exc}") from exc
    required = ["date", "open", "high", "low", "close",
                "vix_source_date", "india_vix_level", "vix_age_calendar_days",
                "sp500_source_date", "sp500_previous_close",
                "sp500_previous_return_1d", "sp500_age_calendar_days",
                "external_features_ready"]
    if any(c not in merged.columns for c in required):
        raise MultiFactorComparisonError("Integrated CSV missing Step 3.9 required columns")
    if len(merged) != len(original):
        raise MultiFactorComparisonError("Integrated CSV does not have exactly all pre-2025 NIFTY rows")
    merged["date"] = _strict_iso_dates(merged["date"], "NIFTY date")
    if (merged["date"] >= pd.Timestamp(FINAL_HOLDOUT_START)).any():
        raise MultiFactorComparisonError("Integrated CSV contains 2025+ final-holdout dates")
    if merged["date"].duplicated().any() or not merged["date"].is_monotonic_increasing:
        raise MultiFactorComparisonError("Integrated NIFTY dates are duplicate or out of order")
    if not merged["date"].equals(original["date"]):
        raise MultiFactorComparisonError("Integrated NIFTY dates do not match original development history")
    for col in ("open", "high", "low", "close"):
        values = pd.to_numeric(merged[col], errors="coerce").to_numpy(dtype=float)
        if (not np.isfinite(values).all()
                or not np.allclose(values, original[col].to_numpy(dtype=float), rtol=1e-11, atol=1e-6)):
            raise MultiFactorComparisonError(f"Integrated {col} differs from original NIFTY OHLC")

    for prefix, datecol, feature, agecol in (
        ("India VIX", "vix_source_date", EXTERNAL_FEATURE_COLUMNS[0], "vix_age_calendar_days"),
        ("S&P 500", "sp500_source_date", EXTERNAL_FEATURE_COLUMNS[1], "sp500_age_calendar_days"),
    ):
        merged[datecol] = _strict_iso_dates(merged[datecol], f"{prefix} source date", allow_missing=True)
        raw_vals = merged[feature]
        vals = pd.to_numeric(raw_vals, errors="coerce")
        if (raw_vals.notna() & vals.isna()).any():
            raise MultiFactorComparisonError(f"Invalid nonnumeric {prefix} value")
        if not np.isfinite(vals.dropna().to_numpy(dtype=float)).all():
            raise MultiFactorComparisonError(f"Nonfinite {prefix} feature")
        if prefix == "India VIX" and (vals.dropna() <= 0).any():
            raise MultiFactorComparisonError("India VIX levels must be positive")
        if prefix == "S&P 500" and (vals.dropna() <= -1).any():
            raise MultiFactorComparisonError("S&P 500 simple returns must exceed -100%")
        merged[feature] = vals.astype(float)
        matched = merged[datecol].notna()
        if matched.any():
            age = (merged.loc[matched, "date"] - merged.loc[matched, datecol]).dt.days
            reported_age = pd.to_numeric(merged.loc[matched, agecol], errors="coerce")
            if ((age < 1) | (age > MAX_EXTERNAL_AGE_DAYS)).any():
                raise MultiFactorComparisonError(f"{prefix}: future/same-date or stale source matches")
            if reported_age.isna().any() or not np.array_equal(age.to_numpy(), reported_age.to_numpy()):
                raise MultiFactorComparisonError(f"{prefix}: inconsistent source observation age")
        if (~matched & merged[feature].notna()).any():
            raise MultiFactorComparisonError(f"{prefix}: observation with no provenance date")
        if (~matched & merged[agecol].notna()).any():
            raise MultiFactorComparisonError(f"{prefix}: observation age with no source date")
        if prefix == "India VIX" and (matched & merged[feature].isna()).any():
            raise MultiFactorComparisonError("India VIX source date given without a usable value")
    sp_close = pd.to_numeric(merged["sp500_previous_close"], errors="coerce")
    if (merged["sp500_previous_close"].notna() & sp_close.isna()).any():
        raise MultiFactorComparisonError("Invalid S&P 500 previous closing level")
    if (sp_close.dropna() <= 0).any() or not np.isfinite(sp_close.dropna()).all():
        raise MultiFactorComparisonError("S&P 500 previous closing level must be positive and finite")
    if (sp_close.notna() != merged["sp500_source_date"].notna()).any():
        raise MultiFactorComparisonError("S&P 500 previous close/date do not match")
    flags = merged["external_features_ready"].str.strip().str.lower()
    if flags.isna().any() or not flags.isin(("true", "false")).all():
        raise MultiFactorComparisonError("external_features_ready must contain true/false")
    claimed = flags.eq("true").to_numpy(dtype=bool)
    actual = merged[list(EXTERNAL_FEATURE_COLUMNS)].notna().all(axis=1).to_numpy(dtype=bool)
    if not np.array_equal(claimed, actual):
        raise MultiFactorComparisonError("external_features_ready flags inconsistent with missing inputs")
    merged["external_features_ready"] = actual
    audit = {
        "integrated_csv_sha256": sha256(merged_path.read_bytes()).hexdigest(),
        "nifty_original_sha256": sha256(source.read_bytes()).hexdigest(),
        "integrated_development_rows": int(len(merged)),
        "rows_with_both_external_features": int(actual.sum()),
        "rows_with_missing_external_feature": int((~actual).sum()),
        "external_matching_rule": "Strictly earlier source DATE, maximum 7 calendar days, verified on every row",
        "original_nifty_ohlc_and_dates_match": True,
    }
    return merged[["date", *EXTERNAL_FEATURE_COLUMNS, "external_features_ready"]].copy(), audit


def compare_multifactor_features(
    nifty_csv: str | Path, external_csv: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """Identical fixed XGBoost models on identical training/validation samples.

    Only the 9-vs-11 feature input differs. No hyperparameter/feature selection
    based on development results; 2025+ remains unopened.
    """
    external, external_audit = _validate_integrated(nifty_csv, external_csv)
    frames = []
    fold_audits = []
    for year in FOLD_YEARS:
        start, end = f"{year}-01-01", f"{year+1}-01-01"
        try:
            train, validation, dataset_audit = build_development_features(
                nifty_csv, validation_start=start, holdout_start=end,
                include_ohlc_features=True, include_technical_features=True,
            )
        except FeatureDatasetError as exc:
            raise MultiFactorComparisonError(f"Fold {year}: {exc}") from exc
        train = train.merge(external, on="date", how="left", validate="one_to_one", sort=False)
        validation = validation.merge(external, on="date", how="left", validate="one_to_one", sort=False)
        train_before, valid_before = len(train), len(validation)
        train = train.loc[train[list(EXTERNAL_FEATURE_COLUMNS)].notna().all(axis=1)].reset_index(drop=True)
        validation = validation.loc[validation[list(EXTERNAL_FEATURE_COLUMNS)].notna().all(axis=1)].reset_index(drop=True)
        if train.empty or validation.empty:
            raise MultiFactorComparisonError(f"Fold {year}: no usable rows after external-data alignment")
        if (train["date"] >= pd.Timestamp(start)).any() or (validation["date"] < pd.Timestamp(start)).any() or (validation["date"] >= pd.Timestamp(end)).any():
            raise MultiFactorComparisonError(f"Fold {year}: partition boundary violated")
        if train["date"].max() >= validation["date"].min():
            raise MultiFactorComparisonError(f"Fold {year}: overlapping train/validation dates")
        if dataset_audit["training_boundary_labels_purged"] != 1:
            raise MultiFactorComparisonError(f"Fold {year}: expected a single purged training boundary label")
        y_train = train[TARGET_COLUMN].to_numpy(dtype=int)
        y_valid = validation[TARGET_COLUMN].to_numpy(dtype=int)
        if len(np.unique(y_train)) != 2:
            raise MultiFactorComparisonError(f"Fold {year}: training data missing event/non-event class")
        baseline = np.full(len(validation), float(y_train.mean()), dtype=float)
        try:
            nine, nine_fit = _fit_predict_fixed_xgb(train, validation, FULL_FEATURE_COLUMNS)
            eleven, eleven_fit = _fit_predict_fixed_xgb(train, validation, MULTIFACTOR_FEATURE_COLUMNS)
        except XGBComparisonError as exc:
            raise MultiFactorComparisonError(f"Fold {year}: XGBoost fitting failed: {exc}") from exc
        frames.append(pd.DataFrame({
            "validation_year": year,
            "date": validation["date"],
            "actual_downside_event": y_valid,
            "baseline_probability": baseline,
            "xgboost_9_features_probability": nine,
            "xgboost_11_features_probability": eleven,
        }))
        fold_audits.append({
            "year": year,
            "training_before": start,
            "validation_before": end,
            "train_rows_before_external_filter": train_before,
            "train_rows_evaluated": len(train),
            "train_rows_dropped_missing_external": train_before - len(train),
            "train_events": int(y_train.sum()),
            "validation_rows_before_external_filter": valid_before,
            "validation_rows_evaluated": len(validation),
            "validation_rows_dropped_missing_external": valid_before - len(validation),
            "validation_events": int(y_valid.sum()),
            "last_training_prediction_date": train["date"].iloc[-1].date().isoformat(),
            "first_validation_prediction_date": validation["date"].iloc[0].date().isoformat(),
            "purged_cross_boundary_training_labels": dataset_audit["training_boundary_labels_purged"],
            "models": {
                "baseline": _metrics(y_valid, baseline),
                "xgboost_9_features": _metrics(y_valid, nine),
                "xgboost_11_features": _metrics(y_valid, eleven),
            },
            "fit_audit": {
                "xgboost_9_features": nine_fit,
                "xgboost_11_features": eleven_fit,
            },
        })
    combined = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    if combined["date"].duplicated().any() or (combined["date"] >= pd.Timestamp(FINAL_HOLDOUT_START)).any():
        raise MultiFactorComparisonError("Fold dates overlap or holdout contamination")
    pooled = {
        "constant_baseline": _pooled_probability_scores(combined, "baseline_probability"),
        "xgboost_9_features": _pooled_probability_scores(combined, "xgboost_9_features_probability"),
        "xgboost_11_features": _pooled_probability_scores(combined, "xgboost_11_features_probability"),
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
    nifty_csv: str | Path, external_csv: str | Path, output_dir: str | Path, *, overwrite: bool = False,
) -> dict:
    """Write audit and predictions, preventing accidental replacement by default."""
    out = Path(output_dir)
    targets = (out / PREDICTIONS_FILENAME, out / REPORT_FILENAME)
    if not overwrite and any(x.exists() for x in targets):
        raise MultiFactorComparisonError("Outputs already exist; choose another output directory or --overwrite")
    predictions, report = compare_multifactor_features(nifty_csv, external_csv)
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as tmp:
        temp = Path(tmp)
        export = predictions.copy()
        export["date"] = export["date"].dt.strftime("%Y-%m-%d")
        export.to_csv(temp / PREDICTIONS_FILENAME, index=False, float_format="%.12g")
        (temp / REPORT_FILENAME).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for item in targets:
            (temp / item.name).replace(item)
    return report
