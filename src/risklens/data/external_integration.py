"""Integrate historical India VIX and S&P 500 data without look-ahead leakage.

RiskLens predicts strictly AFTER a NIFTY 50 session closes. For BOTH external
series we conservatively use only observations with source dates STRICTLY
EARLIER than the NIFTY prediction date. In particular, same-date US closing
prices are not yet available at the Indian close. Same-date VIX observations
are conservatively excluded until their publication time is verified.

The final 2025+ NIFTY holdout is sealed before any derived values are made.
This module DOES NOT train models, create targets, or evaluate outcomes.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from risklens.data.validation import OHLCValidationError, load_ohlc_csv, validate_ohlc

HOLDOUT_START = pd.Timestamp("2025-01-01")
DEFAULT_MAX_AGE_DAYS = 7
OUTPUT_CSV = "nifty50_external_development.csv"
REPORT_JSON = "external_integration_report.json"


class ExternalDataError(ValueError):
    """Input source, alignment rule, or requested output is not safe."""


def _read_as_text(path: Path) -> pd.DataFrame:
    """Read everything as text so malformed/missing values cannot be masked."""
    if not path.is_file():
        raise ExternalDataError(f"CSV file not found: {path}")
    try:
        result = pd.read_csv(path, dtype="string", keep_default_na=False,
                             skip_blank_lines=False, encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ExternalDataError(f"Could not read {path}: {exc}") from exc
    if result.empty:
        raise ExternalDataError(f"CSV is empty: {path}")
    result.columns = [str(c).strip().casefold() for c in result.columns]
    if len(set(result.columns)) != len(result.columns):
        raise ExternalDataError(f"Duplicate column names in {path}")
    return result


def _load_vix(vix_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    """Validate nine-ish annual CSV exports; quarantine only known zero-OHL pattern."""
    root = Path(vix_dir)
    if not root.is_dir():
        raise ExternalDataError(f"India VIX directory not found: {root}")
    paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.casefold() == ".csv")
    if not paths:
        raise ExternalDataError(f"No India VIX CSVs found directly in {root}")

    frames = []
    provenance = []
    for path in paths:
        raw = _read_as_text(path)
        required = ("date", "open", "high", "low", "close")
        if any(c not in raw.columns for c in required):
            raise ExternalDataError(f"{path.name}: need Date, Open, High, Low, Close columns")
        dates = pd.to_datetime(raw["date"].str.strip(), format="%d-%b-%Y", errors="coerce")
        if dates.isna().any():
            raise ExternalDataError(f"{path.name}: unparseable Date; expected DD-MON-YYYY")
        table = pd.DataFrame({"date": dates})
        for col in required[1:]:
            # VIX exported levels may contain decimals; never interpolate.
            values = pd.to_numeric(raw[col].str.strip(), errors="coerce")
            if not np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).all():
                raise ExternalDataError(f"{path.name}: missing/nonnumeric {col}")
            table[col] = values.astype(float)
        frames.append(table)
        provenance.append({
            "filename": path.name,
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "rows": int(len(table)),
        })

    combined = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    if combined["date"].duplicated().any():
        repeats = combined.loc[combined["date"].duplicated(keep=False), "date"]
        raise ExternalDataError(f"India VIX has duplicate date(s): {repeats.dt.strftime('%Y-%m-%d').unique().tolist()[:5]}")
    if (combined["close"] <= 0).any():
        raise ExternalDataError("India VIX Close must be strictly positive")

    # Known NSE-export pattern: Open=High=Low=0, Close present. Not a valid
    # complete OHLC observation. Flag and quarantine; do not use stale close.
    suspect = ((combined["open"] == 0) & (combined["high"] == 0)
               & (combined["low"] == 0) & (combined["close"] > 0))
    quarantined = combined.loc[suspect, "date"].dt.strftime("%Y-%m-%d").tolist()
    usable = combined.loc[~suspect].copy()
    try:
        checked = validate_ohlc(usable)
    except OHLCValidationError as exc:
        raise ExternalDataError(f"India VIX OHLC failed validation: {exc}") from exc

    # Seal final holdout before ANY feature alignment; VIX may grow later.
    checked = checked.loc[checked["date"] < HOLDOUT_START, ["date", "close"]].copy()
    if checked.empty:
        raise ExternalDataError("No usable India VIX observations before final holdout")
    checked = checked.rename(columns={"date": "vix_source_date", "close": "india_vix_level"})
    report = {
        "files": provenance,
        "total_source_rows": int(len(combined)),
        "zero_ohl_rows_quarantined": quarantined,
        "zero_ohl_count": len(quarantined),
        "usable_pre_holdout_rows": int(len(checked)),
        "first_usable_date": checked["vix_source_date"].min().date().isoformat(),
        "last_usable_date": checked["vix_source_date"].max().date().isoformat(),
    }
    return checked.sort_values("vix_source_date").reset_index(drop=True), report


def _load_sp500(sp500_csv: str | Path) -> tuple[pd.DataFrame, dict]:
    """Validate OHLC, then calculate *historical* US returns before holdout."""
    path = Path(sp500_csv)
    raw = _read_as_text(path)
    try:
        checked = validate_ohlc(raw)
    except OHLCValidationError as exc:
        raise ExternalDataError(f"S&P 500 OHLC failed validation: {exc}") from exc

    # US source can include 2025-26, but those observations do not enter work.
    development = checked.loc[checked["date"] < HOLDOUT_START, ["date", "close"]].copy()
    if len(development) < 2:
        raise ExternalDataError("S&P 500 needs at least two pre-holdout observations")
    development["sp500_previous_return_1d"] = development["close"].pct_change(fill_method=None)
    development = development.rename(columns={"date": "sp500_source_date", "close": "sp500_previous_close"})
    report = {
        "filename": path.name,
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "all_source_rows_structurally_checked": int(len(checked)),
        "pre_holdout_rows": int(len(development)),
        "first_date": development["sp500_source_date"].min().date().isoformat(),
        "last_date": development["sp500_source_date"].max().date().isoformat(),
        "extra_columns_ignored": checked.attrs.get("ignored_columns", []),
    }
    return development.sort_values("sp500_source_date").reset_index(drop=True), report


def integrate_external_development(
    nifty_csv: str | Path, vix_dir: str | Path, sp500_csv: str | Path, *,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
) -> tuple[pd.DataFrame, dict]:
    """Attach conservative prior-date external observations to NIFTY sessions.

    Missing/stale matches remain NaN; NO filling with future or synthetic data.
    Every pre-2025 NIFTY row is retained. `external_features_ready` identifies
    rows eligible for a later model requiring both external variables.
    """
    if isinstance(max_age_days, bool) or not isinstance(max_age_days, int) or not 1 <= max_age_days <= 30:
        raise ExternalDataError("max_age_days must be an integer between 1 and 30")
    try:
        nifty = load_ohlc_csv(nifty_csv)
    except OHLCValidationError as exc:
        raise ExternalDataError(f"NIFTY file invalid: {exc}") from exc
    dev = nifty.loc[nifty["date"] < HOLDOUT_START].copy().reset_index(drop=True)
    if dev.empty:
        raise ExternalDataError("NIFTY data has no pre-2025 observations")
    vix, vix_audit = _load_vix(vix_dir)
    sp, sp_audit = _load_sp500(sp500_csv)

    age_limit = pd.Timedelta(days=max_age_days)
    result = pd.merge_asof(
        dev.sort_values("date"), vix, left_on="date", right_on="vix_source_date",
        direction="backward", tolerance=age_limit, allow_exact_matches=False,
    )
    result = pd.merge_asof(
        result.sort_values("date"), sp, left_on="date", right_on="sp500_source_date",
        direction="backward", tolerance=age_limit, allow_exact_matches=False,
    )
    result["vix_age_calendar_days"] = (result["date"] - result["vix_source_date"]).dt.days.astype("Int64")
    result["sp500_age_calendar_days"] = (result["date"] - result["sp500_source_date"]).dt.days.astype("Int64")
    result["external_features_ready"] = (
        result["india_vix_level"].notna()
        & result["sp500_previous_return_1d"].notna()
    )

    for label in ("vix", "sp500"):
        source_dates = result[f"{label}_source_date"].dropna()
        prediction_dates = result.loc[source_dates.index, "date"]
        if not (source_dates < prediction_dates).all():
            raise ExternalDataError(f"Internal error: {label} source was not strictly earlier than prediction date")
    if (result["date"] >= HOLDOUT_START).any():
        raise ExternalDataError("Internal error: final holdout contamination")
    if len(result) != len(dev):
        raise ExternalDataError("Unexpected change to NIFTY row count during external join")

    # Exact data-availability counts, not guessed trading-calendar coverage.
    report = {
        "scope": "PRE-2025 DEVELOPMENT ONLY; no modeling or holdout outcome analysis",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "holdout_starts": HOLDOUT_START.date().isoformat(),
        "nifty_source_filename": Path(nifty_csv).name,
        "nifty_sha256": sha256(Path(nifty_csv).read_bytes()).hexdigest(),
        "nifty_development_sessions": int(len(dev)),
        "nifty_development_first": dev["date"].min().date().isoformat(),
        "nifty_development_last": dev["date"].max().date().isoformat(),
        "vix_source_audit": vix_audit,
        "sp500_source_audit": sp_audit,
        "date_alignment_rule": (
            "For BOTH external series, use the most recent source DATE STRICTLY before each "
            "NIFTY session date; never same-date; exclude matches older than max_age_calendar_days"
        ),
        "max_age_calendar_days": max_age_days,
        "source_date_lag": "Dates are exchange-local dates; strict prior-date rule is conservative; actual publication timestamps unverified",
        "india_vix_units": "VIX index points (e.g., 20 conventionally denotes approximately 20% annualized implied volatility)",
        "sp500_return_definition": "Close[s] / Close[previous US trading session] - 1; s strictly earlier than NIFTY date t",
        "nifty_rows_with_vix": int(result["india_vix_level"].notna().sum()),
        "nifty_rows_with_sp500_return": int(result["sp500_previous_return_1d"].notna().sum()),
        "nifty_rows_with_both_external_features": int(result["external_features_ready"].sum()),
        "nifty_rows_without_both": int((~result["external_features_ready"]).sum()),
        "vix_missing_prediction_dates_sample": result.loc[result["india_vix_level"].isna(), "date"].dt.strftime("%Y-%m-%d").head(10).tolist(),
        "sp500_return_missing_prediction_dates_sample": result.loc[result["sp500_previous_return_1d"].isna(), "date"].dt.strftime("%Y-%m-%d").head(10).tolist(),
        "warnings": [
            "Original market data were structurally validated, not independently authenticated or licensed.",
            "The S&P 500 data source URL and publication metadata have not been independently established.",
            "Same-date India VIX values intentionally excluded because their availability at NIFTY close was not verified.",
            "Invalid zero-OHL VIX source records were quarantined, not repaired; raw files are unchanged.",
            "Missing/stale external observations are left missing and explicitly counted; no forward/backward price imputation.",
            "This file is NOT yet a leakage-safe supervised ML dataset; split-boundary targets and training-only preprocessing remain separate tasks.",
            "2025+ outcomes, features and external returns were not calculated or inspected.",
        ],
    }
    columns = [
        "date", "open", "high", "low", "close",
        "vix_source_date", "india_vix_level", "vix_age_calendar_days",
        "sp500_source_date", "sp500_previous_close", "sp500_previous_return_1d", "sp500_age_calendar_days",
        "external_features_ready",
    ]
    return result.loc[:, columns].reset_index(drop=True), report


def save_external_development(
    nifty_csv: str | Path, vix_dir: str | Path, sp500_csv: str | Path,
    output_dir: str | Path, *, max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    overwrite: bool = False,
) -> dict:
    """Validate/merge first, then stage final CSV/JSON; refuse overwrite by default."""
    destination = Path(output_dir)
    targets = [destination / OUTPUT_CSV, destination / REPORT_JSON]
    if not overwrite and any(p.exists() for p in targets):
        raise ExternalDataError("External integration output exists; choose another folder or pass --overwrite")
    merged, report = integrate_external_development(
        nifty_csv, vix_dir, sp500_csv, max_age_days=max_age_days,
    )
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination) as temp:
        staged = Path(temp)
        export = merged.copy()
        for col in ("date", "vix_source_date", "sp500_source_date"):
            export[col] = export[col].dt.strftime("%Y-%m-%d")
        export.to_csv(staged / OUTPUT_CSV, index=False, na_rep="", float_format="%.12g")
        (staged / REPORT_JSON).write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        for target in targets:
            (staged / target.name).replace(target)
    return report
