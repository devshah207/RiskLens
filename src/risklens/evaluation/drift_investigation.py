"""Investigate the *three* flagged Phase 6 input features, not just repeat PSI.

No model fitting, no Optuna, no final-holdout rescore or post-hoc model selection.
Reuses the one-time holdout's exact scored dates and original causal feature join.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from risklens.data.validation import load_ohlc_csv
from risklens.evaluation.feature_drift import (
    _feature_psi,
    load_feature_frames,
)
from risklens.evaluation.final_holdout import (
    _read_holdout_vix,
)
from risklens.features.dataset import _wilder_rsi_14

FLAGGED = ("ema_50_distance", "india_vix_level", "rsi_14")
REFERENCE_START = pd.Timestamp("2020-01-01")
REFERENCE_END = pd.Timestamp("2025-01-01")
OUT = (
    "three_feature_investigation.json",
    "three_feature_periods.csv",
    "three_feature_months.csv",
    "three_feature_extremes.csv",
    "three_feature_risk_slices.csv",
    "three_feature_investigation.md",
)


class DriftInvestigationError(ValueError):
    """Feature investigation cannot substantiate the data or chronology."""


def _stats(values: np.ndarray) -> dict:
    a = np.asarray(values, dtype=float)
    if not len(a) or not np.isfinite(a).all():
        raise DriftInvestigationError("Feature group empty or contains nonfinite values")
    percentiles = np.percentile(a, [1, 10, 25, 50, 75, 90, 99])
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std(ddof=1)) if len(a) > 1 else None,
        "min": float(a.min()),
        "max": float(a.max()),
        **{f"p{q:02d}": float(v) for q, v in zip(
            (1, 10, 25, 50, 75, 90, 99), percentiles
        )},
    }


def _reference_psi(reference: np.ndarray, sample: np.ndarray) -> float:
    """Reference quantiles derive entirely from pre-2025 observations."""
    if len(reference) < 50 or len(sample) < 10:
        return float("nan")
    psi, _ = _feature_psi(reference, sample)
    return float(psi)


def _independent_ema50(prices: np.ndarray) -> np.ndarray:
    # Recursion matches ewm(span=50, adjust=False), but is independent code.
    a = np.asarray(prices, dtype=float)
    ema = np.empty(len(a), dtype=float)
    if not len(a):
        return ema
    ema[0] = a[0]
    alpha = 2.0 / 51.0
    for i in range(1, len(a)):
        ema[i] = alpha * a[i] + (1.0-alpha) * ema[i-1]
    result = a / ema - 1.0
    result[:49] = np.nan  # min_periods=50
    return result


def _independent_rsi14(prices: np.ndarray) -> np.ndarray:
    """Direct Wilder recursion; not using the production RSI helper."""
    a = np.asarray(prices, dtype=float)
    result = np.full(len(a), np.nan)
    if len(a) <= 14:
        return result
    d = np.diff(a)
    gains = np.maximum(d, 0)
    losses = np.maximum(-d, 0)
    gain = float(np.mean(gains[:14]))
    loss = float(np.mean(losses[:14]))

    def value(g, l):
        if l == 0:
            return 50.0 if g == 0 else 100.0
        if g == 0:
            return 0.0
        return 100.0 - 100.0 / (1 + g/l)

    result[14] = value(gain, loss)
    for i in range(15, len(a)):
        gain = (gain*13 + float(gains[i-1]))/14
        loss = (loss*13 + float(losses[i-1]))/14
        result[i] = value(gain, loss)
    return result


def _indicator_integrity(
    nifty_csv: str | Path, development: pd.DataFrame, holdout: pd.DataFrame
) -> dict:
    prices = load_ohlc_csv(nifty_csv)[["date", "close"]].copy()
    if prices.date.max() != pd.Timestamp("2026-09-30"):
        raise DriftInvestigationError("NIFTY source should end on 2026-09-30")
    arr = prices.close.to_numpy(dtype=float)
    prices["ema50_recomputed"] = _independent_ema50(arr)
    prices["rsi14_recomputed"] = _independent_rsi14(arr)
    # A comparison to the production formula is helpful, but never substitutes
    # for checking its output against the actual scored feature observations.
    independent_rsi_check = _wilder_rsi_14(prices.close).to_numpy(dtype=float)
    if not np.allclose(
        prices.rsi14_recomputed.to_numpy(dtype=float),
        independent_rsi_check,
        atol=1e-9, rtol=1e-10, equal_nan=True,
    ):
        raise DriftInvestigationError("Independent RSI/Wilder formula mismatch")
    records = []
    for period, frame in (("development", development), ("holdout", holdout)):
        merged = frame[["date", "ema_50_distance", "rsi_14"]].merge(
            prices[["date", "ema50_recomputed", "rsi14_recomputed"]],
            on="date", validate="one_to_one", how="left",
        )
        if len(merged) != len(frame) or merged[["ema50_recomputed", "rsi14_recomputed"]].isna().any().any():
            raise DriftInvestigationError(f"{period}: independent feature reconstruction incomplete")
        for col, expected in (
            ("ema_50_distance", "ema50_recomputed"),
            ("rsi_14", "rsi14_recomputed"),
        ):
            actual = merged[col].to_numpy(dtype=float)
            predicted = merged[expected].to_numpy(dtype=float)
            maximum = float(np.max(np.abs(actual-predicted)))
            records.append({
                "period": period, "feature": col,
                "rows": len(merged),
                "max_absolute_reconstruction_difference": maximum,
                "passed": bool(np.allclose(actual, predicted, rtol=1e-10, atol=1e-8)),
            })
        if (merged.rsi_14 < 0).any() or (merged.rsi_14 > 100).any():
            raise DriftInvestigationError(f"{period}: RSI outside theoretical 0–100 interval")
        if (merged.ema_50_distance <= -1).any():
            raise DriftInvestigationError(f"{period}: EMA distance <= -1")
    if not all(r["passed"] for r in records):
        raise DriftInvestigationError(
            "EMA/RSI production values differ from independent causal reconstruction"
        )
    return {"checks": records, "formula_consistency": "PASS"}


def _vix_overlap_audit(
    development_external_csv: str | Path,
    holdout_vix_csv: str | Path,
) -> dict:
    """Check the 2024 overlap of old integrated VIX vs newly combined VIX source."""
    dev = pd.read_csv(development_external_csv, dtype="string", encoding="utf-8-sig")
    expected = {"date", "vix_source_date", "india_vix_level"}
    if not expected.issubset(dev.columns):
        raise DriftInvestigationError("Development external CSV lacks VIX provenance fields")
    for col in ("date", "vix_source_date"):
        dev[col] = pd.to_datetime(dev[col], format="%Y-%m-%d", errors="coerce")
    dev["india_vix_level"] = pd.to_numeric(dev["india_vix_level"], errors="coerce")
    if dev["date"].isna().any():
        raise DriftInvestigationError("Malformed development NIFTY dates")
    overlap = dev.loc[
        (dev.date >= pd.Timestamp("2024-01-01")) &
        (dev.date < pd.Timestamp("2025-01-01")),
        ["date", "vix_source_date", "india_vix_level"],
    ].sort_values("date").copy()
    combined, raw_audit = _read_holdout_vix(holdout_vix_csv)
    overlap = pd.merge_asof(
        overlap,
        combined.sort_values("vix_source_date").rename(columns={
            "vix_source_date": "new_vix_source_date",
            "india_vix_level": "new_india_vix_level",
        }),
        left_on="date", right_on="new_vix_source_date",
        direction="backward", allow_exact_matches=False,
        tolerance=pd.Timedelta(days=7),
    )
    matched = overlap.dropna(subset=[
        "vix_source_date", "india_vix_level",
        "new_vix_source_date", "new_india_vix_level",
    ])
    if len(matched) == 0:
        raise DriftInvestigationError(
            "No matching 2024 VIX overlap between original development and new holdout sources"
        )
    old = matched.india_vix_level.to_numpy(dtype=float)
    new = matched.new_india_vix_level.to_numpy(dtype=float)
    deltas = np.abs(old - new)
    dates_match = matched.vix_source_date.eq(matched.new_vix_source_date)
    problematic = (~dates_match) | (~np.isclose(old, new, rtol=1e-8, atol=1e-5))
    output = {
        "2024_overlap_sessions_matched": len(matched),
        "2024_overlap_sessions_missing_either": len(overlap)-len(matched),
        "source_date_mismatches": int((~dates_match).sum()),
        "value_mismatches": int((~np.isclose(old, new, rtol=1e-8, atol=1e-5)).sum()),
        "max_absolute_vix_level_difference": float(deltas.max()),
        "first_problem_dates": matched.loc[problematic, "date"].dt.strftime("%Y-%m-%d").head(12).tolist(),
        "combined_vix_quarantined_rows": raw_audit["quarantined_zero_ohl_rows"],
        "assessment": "PASS" if not problematic.any() else "SOURCE_DISAGREEMENT_REQUIRES_REVIEW",
        "limitation": "A 2024 overlap check cannot prove data-source continuity or publication-time validity in 2025–2026.",
    }
    return output


def _period_mask(frame: pd.DataFrame, start: str, end: str) -> np.ndarray:
    return ((frame.date >= pd.Timestamp(start)) & (
        frame.date < pd.Timestamp(end)
    )).to_numpy(dtype=bool)


def _quantile_slices(
    feature: str, reference: np.ndarray, hold: pd.DataFrame
) -> list[dict]:
    low, high = np.quantile(reference, [0.10, 0.90])
    out = []
    for year in (2025, 2026):
        frame = hold.loc[hold.date.dt.year == year]
        for label, mask in (
            ("below_development_p10", frame[feature] < low),
            ("middle_p10_p90", (frame[feature] >= low) & (frame[feature] <= high)),
            ("above_development_p90", frame[feature] > high),
        ):
            selected = frame.loc[mask]
            y = selected.actual_downside_event.to_numpy(dtype=int)
            item = {
                "feature": feature, "year": year, "risk_bucket": label,
                "rows": len(selected),
                "events": int(y.sum()),
                "event_rate": float(y.mean()) if len(y) else None,
                "small_sample_warning": bool(len(selected) < 30 or y.sum() < 5),
            }
            for model, col in (
                ("xgboost", "xgboost_11_fixed_probability"),
                ("lstm_86", "lstm_optuna_86_probability"),
            ):
                pred = selected[col].to_numpy(dtype=float)
                item[f"{model}_mean_predicted_risk"] = (
                    float(pred.mean()) if len(pred) else None
                )
                item[f"{model}_brier"] = (
                    float(np.mean((pred-y)**2)) if len(pred) else None
                )
            out.append(item)
    return out


def investigate_three_features(
    nifty_csv: str | Path,
    development_external_csv: str | Path,
    vix_csv: str | Path,
    sp500_csv: str | Path,
    holdout_predictions_csv: str | Path,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate evidence for three flags; no model modification or holdout re-score."""
    dev, feature_rows, predictions, audit = load_feature_frames(
        nifty_csv, development_external_csv, vix_csv, sp500_csv,
        holdout_predictions_csv,
    )
    if any(name not in dev or name not in feature_rows for name in FLAGGED):
        raise DriftInvestigationError("One of three flagged features absent")
    if not feature_rows.date.equals(predictions.date):
        raise DriftInvestigationError("Holdout feature rows and saved prediction dates differ")
    hold = feature_rows.merge(
        predictions[[
            "date", "actual_downside_event", "xgboost_11_fixed_probability",
            "lstm_optuna_86_probability",
        ]], on="date", how="inner", validate="one_to_one"
    )
    if len(hold) != len(predictions):
        raise DriftInvestigationError("Risk slices cannot silently discard scored dates")
    indicator = _indicator_integrity(nifty_csv, dev, hold)
    overlap = _vix_overlap_audit(development_external_csv, vix_csv)

    groups = [
        ("old_2016_2019", dev, "2016-01-01", "2020-01-01"),
        ("search_2020_2022", dev, "2020-01-01", "2023-01-01"),
        ("recent_2023_2024", dev, "2023-01-01", "2025-01-01"),
        ("reference_2020_2024", dev, "2020-01-01", "2025-01-01"),
        ("all_pre_2025", dev, "2016-01-01", "2025-01-01"),
        ("holdout_2025", hold, "2025-01-01", "2026-01-01"),
        ("holdout_2026", hold, "2026-01-01", "2026-10-01"),
        ("holdout_all", hold, "2025-01-01", "2026-10-01"),
    ]

    rows, month_rows, extreme_rows, slices = [], [], [], []
    feature_findings = {}
    for feature in FLAGGED:
        ref_mask = _period_mask(dev, "2020-01-01", "2025-01-01")
        recent_mask = _period_mask(dev, "2023-01-01", "2025-01-01")
        ref = dev.loc[ref_mask, feature].to_numpy(dtype=float)
        recent = dev.loc[recent_mask, feature].to_numpy(dtype=float)
        if len(ref) < 100 or len(recent) < 50:
            raise DriftInvestigationError("Insufficient recent development reference")
        q10, q25, q75, q90 = np.percentile(ref, [10, 25, 75, 90])
        per_feature = {}
        for label, frame, start, end in groups:
            selected = frame.loc[_period_mask(frame, start, end)]
            if selected.empty:
                raise DriftInvestigationError(f"{feature}: empty group {label}")
            val = selected[feature].to_numpy(dtype=float)
            stat = _stats(val)
            stat["feature"] = feature
            stat["period"] = label
            stat["psi_against_2020_2024"] = _reference_psi(ref, val)
            stat["psi_against_2023_2024"] = _reference_psi(recent, val)
            stat["below_2020_2024_p10_fraction"] = float((val < q10).mean())
            stat["above_2020_2024_p90_fraction"] = float((val > q90).mean())
            stat["above_zero_fraction"] = float((val > 0).mean()) if feature == "ema_50_distance" else None
            stat["rsi_below_30_fraction"] = float((val < 30).mean()) if feature == "rsi_14" else None
            stat["rsi_above_70_fraction"] = float((val > 70).mean()) if feature == "rsi_14" else None
            stat["vix_above_30_fraction"] = float((val > 30).mean()) if feature == "india_vix_level" else None
            rows.append(stat)
            per_feature[label] = stat
        all_m = []
        # Monthly medians highlight sustained shifts versus isolated spikes.
        grouped = hold.assign(month=hold.date.dt.strftime("%Y-%m")).groupby("month", sort=True)
        for month, group in grouped:
            vals = group[feature].to_numpy(dtype=float)
            if len(vals) < 5:
                continue
            median = float(np.median(vals))
            outside = bool(median < q25 or median > q75)
            all_m.append(outside)
            month_rows.append({
                "feature": feature, "month": month, "sessions": len(vals),
                "mean": float(vals.mean()),
                "median": median,
                "min": float(vals.min()),
                "max": float(vals.max()),
                "median_outside_2020_2024_iqr": outside,
                "fraction_below_reference_p10": float((vals < q10).mean()),
                "fraction_above_reference_p90": float((vals > q90).mean()),
            })
        for side, top in (
            ("lowest", hold.nsmallest(5, feature)),
            ("highest", hold.nlargest(5, feature)),
        ):
            for r in top.itertuples():
                age = None
                if feature == "india_vix_level" and "vix_source_date" in hold.columns:
                    if pd.notna(getattr(r, "vix_source_date")):
                        age = (r.date-r.vix_source_date).days
                extreme_rows.append({
                    "feature": feature, "extreme_side": side,
                    "date": r.date.strftime("%Y-%m-%d"),
                    "feature_value": float(getattr(r, feature)),
                    "vix_observation_age_days": age,
                })
        slices.extend(_quantile_slices(feature, ref, hold))
        per_feature["monthly_median_outside_reference_iqr"] = {
            "months_with_5plus_sessions": len(all_m),
            "months_outside_iqr": int(sum(all_m)),
            "fraction_outside_iqr": float(np.mean(all_m)) if all_m else None,
        }
        per_feature["provisional_interpretation"] = (
            "PSI/level shifts are descriptive. Verify source/formula checks first; "
            "assess monthly persistence and regime-by-year; no causal conclusion or model change."
        )
        feature_findings[feature] = per_feature

    if "vix_source_date" not in hold:
        raise DriftInvestigationError("Holdout VIX provenance date unavailable")
    vix_age = (hold.date-hold.vix_source_date).dt.days
    if vix_age.isna().any() or (vix_age < 1).any() or (vix_age > 7).any():
        raise DriftInvestigationError("Holdout India VIX source dates are unavailable, same-day, or stale")
    source_age = {
        "n": len(vix_age),
        "by_age_calendar_days": {
            str(int(age)): int(n)
            for age, n in sorted(vix_age.value_counts().items())
        },
        "median_age_days": float(vix_age.median()),
        "max_age_days": int(vix_age.max()),
        "point_in_time_date_audit": "PASS",
        "note": "Strict prior date verified; precise exchange publication timestamp not independently checked.",
    }

    summary = {
        "title": "RiskLens Phase 6 | Root-cause investigation of 3 flagged inputs",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": list(FLAGGED),
        "scope": "Descriptive retrospective feature/quality investigation only, after one-time holdout",
        "already_scored_holdout_rows": len(hold),
        "holdout_periods": ["2025", "2026 through Sep"],
        "model_lock": "NO retraining, model switch, hyperparameter tuning, threshold selection or second holdout scoring.",
        "feature_formula_integrity": indicator,
        "vix_2024_source_overlap": overlap,
        "vix_holdout_source_age": source_age,
        "feature_findings": feature_findings,
        "data_audits": audit,
        "source_sha256": {
            name: sha256(Path(path).read_bytes()).hexdigest()
            for name, path in (
                ("nifty", nifty_csv),
                ("development_external", development_external_csv),
                ("holdout_vix", vix_csv),
                ("holdout_sp500", sp500_csv),
                ("holdout_predictions", holdout_predictions_csv),
            )
        },
        "limitations": [
            "Actual shifts cannot be attributed to macroeconomic causes without contemporaneous sourced event evidence.",
            "PSI thresholds are descriptive research flags and sensitive to reference period and binning.",
            "2020-2024 reference includes years previously consulted in model development, not an independent baseline.",
            "Data from calendar 2026 cover only through September; monthly comparisons are not seasonally adjusted.",
            "Risk slices use already viewed holdout outcomes for retrospective diagnosis ONLY, never model selection.",
            "Holdout was never rerun. These calculations are retrospective and must not affect trained parameters.",
        ],
    }
    return (
        summary, pd.DataFrame(rows),
        pd.DataFrame(month_rows),
        pd.DataFrame(extreme_rows),
        pd.DataFrame(slices),
    )


def _markdown(report: dict, periods: pd.DataFrame) -> str:
    lines = [
        "# RiskLens — Investigation of EMA-50 Distance, India VIX, RSI-14",
        "",
        f"Previously scored holdout dates: **{report['already_scored_holdout_rows']}**.",
        "**No retuning or repeat holdout evaluation.**",
        "",
        "## Source/formula integrity",
        f"- EMA/RSI independent formula: **{report['feature_formula_integrity']['formula_consistency']}**",
        f"- 2024 old/new India VIX agreement: **{report['vix_2024_source_overlap']['assessment']}**",
        f"- India VIX strict prior-date audit: **{report['vix_holdout_source_age']['point_in_time_date_audit']}**",
        "",
        "## Separating the shifted distributions",
        "| Feature | Period | N | Median | P10 | P90 | PSI vs 2020–24 | PSI vs 2023–24 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in periods.itertuples():
        if r.period not in ("reference_2020_2024", "recent_2023_2024", "holdout_2025", "holdout_2026", "holdout_all"):
            continue
        lines.append(
            f"| {r.feature} | {r.period} | {r.n} | {r.median:.4f} | "
            f"{r.p10:.4f} | {r.p90:.4f} | "
            f"{r.psi_against_2020_2024:.4f} | {r.psi_against_2023_2024:.4f} |"
        )
    lines += [
        "",
        "## Feature-by-feature diagnostic interpretation",
        "",
        "### ema_50_distance",
        "Check fraction of positive EMA distance, 2025 vs 2026, monthly median persistence, "
        "independent EMA reconstruction, and whether the shift extends beyond extremes.",
        "",
        "### india_vix_level",
        "Check 2024 source-overlap agreement, strict source-age distribution, level "
        "quantiles, persistent volatility regimes, and proportion above 30 index points.",
        "",
        "### rsi_14",
        "Check independent Wilder formula, fraction below 30/above 70, monthly "
        "momentum regimes, and extreme-date concentration.",
        "",
        "## Additional evidence",
        "Inspect three_feature_months.csv, three_feature_extremes.csv and "
        "three_feature_risk_slices.csv for persistent shifts, extreme dates, "
        "and small-sample-labelled retrospective outcome associations.",
        "",
        "## Limits",
        *[f"- {x}" for x in report["limitations"]],
        "",
        "Do not interpret the existence of a flag alone as proof of a data bug, "
        "financial cause, or need to retrain.",
        "",
    ]
    return "\n".join(lines)


def save_investigation(
    nifty_csv: str | Path,
    development_external_csv: str | Path,
    vix_csv: str | Path,
    sp500_csv: str | Path,
    holdout_predictions_csv: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict:
    out = Path(output_dir)
    targets = [out / item for item in OUT]
    if not overwrite and any(path.exists() for path in targets):
        raise DriftInvestigationError("Investigation output exists; use --overwrite explicitly")
    report, periods, months, extremes, risk = investigate_three_features(
        nifty_csv, development_external_csv, vix_csv, sp500_csv, holdout_predictions_csv
    )
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as td:
        scratch = Path(td)
        (scratch / OUT[0]).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        for dataframe, file in (
            (periods, OUT[1]), (months, OUT[2]),
            (extremes, OUT[3]), (risk, OUT[4]),
        ):
            dataframe.to_csv(scratch / file, index=False, float_format="%.12g")
        (scratch / OUT[5]).write_text(_markdown(report, periods), encoding="utf-8")
        for target in targets:
            (scratch / target.name).replace(target)
    return report
