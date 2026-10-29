"""Validate pre-2025 integrated market data, independently of any model family.

Shared by XGBoost and LSTM: do not weaken point-in-time source checks.
"""
from pathlib import Path
from hashlib import sha256
import numpy as np
import pandas as pd
from risklens.data.validation import OHLCValidationError, load_ohlc_csv
from risklens.evaluation.splits import FINAL_HOLDOUT_START
from risklens.evaluation.features import EXTERNAL_FEATURE_COLUMNS

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


