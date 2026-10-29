"""Phase 6: descriptive, post-selection monitoring of existing model predictions.

No models are trained or tuned; final holdout is read ONLY as already-produced
probability/label CSV for model-risk reporting. Never use monitoring to reselect
the champion or run a second holdout evaluation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from risklens.evaluation.metrics import _metrics

COLUMNS = (
    "validation_year",
    "date",
    "actual_downside_event",
    "xgboost_11_fixed_probability",
    "lstm_optuna_86_probability",
)
MODELS = {
    "xgboost_fixed": "xgboost_11_fixed_probability",
    "lstm_optuna_trial86": "lstm_optuna_86_probability",
}
# Frozen a priori, not chosen after observing holdout distributions.
BINS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0)
PSI_INVESTIGATE_THRESHOLD = 0.20
PROBABILITY_GAP_THRESHOLD = 0.05
BRIER_DEGRADATION_THRESHOLD = 0.02
OUTPUTS = (
    "monitoring_summary.json",
    "monitoring_periods.csv",
    "monitoring_score_bins.csv",
    "monitoring_report.md",
    "model_risk_register.md",
)


class MonitoringError(ValueError):
    """Invalid model monitoring inputs or unsafe date partition."""


def _read_predictions(path: str | Path, *, holdout: bool) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise MonitoringError(f"Prediction CSV not found: {path}")
    try:
        data = pd.read_csv(path, dtype="string", encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise MonitoringError(f"Cannot read prediction CSV {path}: {exc}") from exc
    if data.empty or set(data.columns) != set(COLUMNS):
        raise MonitoringError(f"{path.name}: unexpected prediction schema")
    data = data.loc[:, list(COLUMNS)].copy()
    dates = pd.to_datetime(data["date"], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any() or not dates.dt.strftime("%Y-%m-%d").eq(data["date"]).all():
        raise MonitoringError("Prediction dates must be strict ISO YYYY-MM-DD")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise MonitoringError("Prediction dates must be unique and sorted")
    if not data["validation_year"].str.fullmatch(r"\d{4}", na=False).all():
        raise MonitoringError("Invalid validation year")
    years = data["validation_year"].astype(int)
    if not np.array_equal(dates.dt.year.to_numpy(), years.to_numpy()):
        raise MonitoringError("Validation year/date mismatch")
    if holdout:
        if dates.min() < pd.Timestamp("2025-01-01") or dates.max() >= pd.Timestamp("2026-10-01"):
            raise MonitoringError("Holdout predictions must be strictly within 2025-09/2026-09")
        if set(years) != {2025, 2026}:
            raise MonitoringError("Holdout predictions must include 2025 and 2026")
    else:
        if dates.min() < pd.Timestamp("2020-01-01") or dates.max() >= pd.Timestamp("2025-01-01"):
            raise MonitoringError("Development data must be dated 2020-2024 only")
        if set(years) != {2020, 2021, 2022, 2023, 2024}:
            raise MonitoringError("Incomplete 2020-2024 development years")
    if not data["actual_downside_event"].isin(("0", "1")).all():
        raise MonitoringError("Downside events must be binary")
    data["actual_downside_event"] = data["actual_downside_event"].astype(int)
    for name, col in MODELS.items():
        arr = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(arr).all() or ((arr < 0) | (arr > 1)).any():
            raise MonitoringError(f"{name}: invalid probabilities")
        data[col] = arr
    data["date"] = dates
    data["validation_year"] = years
    return data


def _fixed_bin_drift(dev: np.ndarray, hold: np.ndarray) -> tuple[list[dict], dict]:
    # Jeffreys-like 0.5 pseudo-count avoids log(0) without moving fixed bins.
    edges = np.asarray(BINS, dtype=float)
    a = np.histogram(dev, bins=edges)[0].astype(float)
    b = np.histogram(hold, bins=edges)[0].astype(float)
    k = len(a)
    af = (a + 0.5) / (a.sum() + 0.5*k)
    bf = (b + 0.5) / (b.sum() + 0.5*k)
    terms = (bf-af)*np.log(bf/af)
    rows = [
        {
            "bin_lower": float(edges[i]),
            "bin_upper": float(edges[i+1]),
            "development_count": int(a[i]),
            "holdout_count": int(b[i]),
            "development_frequency_smoothed": float(af[i]),
            "holdout_frequency_smoothed": float(bf[i]),
            "psi_contribution": float(terms[i]),
        } for i in range(k)
    ]
    psi = float(terms.sum())
    return rows, {
        "score_population_stability_index": round(psi, 8),
        "investigate_if_psi_at_least": PSI_INVESTIGATE_THRESHOLD,
        "investigation_flag": bool(psi >= PSI_INVESTIGATE_THRESHOLD),
        "smoothing": "add 0.5 to every fixed-bin count before normalization",
        "interpretation": "PSI compares SCORE distributions, not feature drift or predictive skill.",
    }


def evaluate_monitoring(
    development_predictions: str | Path,
    holdout_predictions: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Descriptive monitoring only: no fitting, hyperparameter choice or optimization."""
    dev = _read_predictions(development_predictions, holdout=False)
    hold = _read_predictions(holdout_predictions, holdout=True)
    if dev.date.max() >= hold.date.min():
        raise MonitoringError("Development and holdout time periods overlap")

    monitoring_rows = []
    for label, frame in (("development_2020_2024", dev), ("holdout_2025_2026", hold)):
        for year in sorted(frame.validation_year.unique().tolist()):
            group = frame.loc[frame.validation_year == year]
            y = group.actual_downside_event.to_numpy(dtype=int)
            for model, col in MODELS.items():
                p = group[col].to_numpy(dtype=float)
                metrics = _metrics(y, p)
                monitoring_rows.append({
                    "period": label,
                    "year": int(year),
                    "model": model,
                    "observations": int(len(group)),
                    "events": int(y.sum()),
                    "event_frequency": float(y.mean()),
                    "mean_predicted_probability": float(p.mean()),
                    "probability_gap": float(p.mean()-y.mean()),
                    "brier_score": metrics["brier_score"],
                    "log_loss": metrics["log_loss"],
                    "roc_auc": metrics["roc_auc"],
                    "average_precision": metrics["average_precision"],
                })
    drift_rows, model_summaries = [], {}
    for name, col in MODELS.items():
        dev_y = dev.actual_downside_event.to_numpy(dtype=int)
        hold_y = hold.actual_downside_event.to_numpy(dtype=int)
        dev_p = dev[col].to_numpy(dtype=float)
        hold_p = hold[col].to_numpy(dtype=float)
        dev_metrics = _metrics(dev_y, dev_p)
        hold_metrics = _metrics(hold_y, hold_p)
        bins, drift = _fixed_bin_drift(dev_p, hold_p)
        drift_rows.extend([{"model": name, **r} for r in bins])
        brier_change = hold_metrics["brier_score"] - dev_metrics["brier_score"]
        gap = float(hold_p.mean()-hold_y.mean())
        model_summaries[name] = {
            "development": {
                "observations": len(dev),
                "events": int(dev_y.sum()),
                "event_frequency": float(dev_y.mean()),
                "mean_risk": float(dev_p.mean()),
                "metrics": dev_metrics,
            },
            "holdout": {
                "observations": len(hold),
                "events": int(hold_y.sum()),
                "event_frequency": float(hold_y.mean()),
                "mean_risk": float(hold_p.mean()),
                "metrics": hold_metrics,
            },
            "brier_delta_holdout_minus_development": round(float(brier_change), 8),
            "brier_deterioration_flag": bool(brier_change > BRIER_DEGRADATION_THRESHOLD),
            "holdout_calibration_in_the_large_gap": round(gap, 8),
            "absolute_calibration_gap_flag": bool(abs(gap) >= PROBABILITY_GAP_THRESHOLD),
            "score_drift": drift,
        }
    annual_freq = {
        str(int(y)): float(group.actual_downside_event.mean())
        for y, group in hold.groupby("validation_year")
    }
    report = {
        "title": "RiskLens Phase 6 | Frozen risk-model score & performance monitoring",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_scope": "Already-produced development (2020-2024) and one-time holdout (2025-2026) prediction CSVs",
        "selection_status": "XGBoost was preselected; LSTM trial #86 is challenger. NO promotion or refit.",
        "development_source_sha256": sha256(Path(development_predictions).read_bytes()).hexdigest(),
        "holdout_source_sha256": sha256(Path(holdout_predictions).read_bytes()).hexdigest(),
        "models": model_summaries,
        "holdout_annual_event_frequencies": annual_freq,
        "risk_register": [
            {
                "id": "MR-01",
                "risk": "Time-varying downside-event prevalence and calibration drift",
                "evidence": "Year-specific event rates and mean predicted risk; compare 2025 with 2026.",
                "control": "Report annual Brier, log loss, calibration-in-the-large; do not recalibrate on final holdout.",
            },
            {
                "id": "MR-02",
                "risk": "Model score distribution drift, including shifted operating regimes",
                "evidence": "Fixed-bin, smoothed PSI for each frozen candidate.",
                "control": "PSI >=0.20 triggers investigation, not automatic promotion or retraining.",
            },
            {
                "id": "MR-03",
                "risk": "Feature availability and time-varying relationships",
                "evidence": "External data are strictly prior-date; this report has no per-feature holdout input matrix.",
                "control": "Preserve source-lag audits and add feature-level drift when audited feature snapshots exist.",
            },
            {
                "id": "MR-04",
                "risk": "Repeated development consultation and post-hoc model choice",
                "evidence": "Optuna selected trial #86 on 2020–2022, diagnostics reviewed on 2023–2024.",
                "control": "Keep original XGBoost designation; do not choose new params or thresholds from holdout.",
            },
            {
                "id": "MR-05",
                "risk": "Historical VaR tail undercoverage and exception clustering",
                "evidence": "Separate Phase 5 index-level VaR backtest; not produced by the ML classifiers.",
                "control": "Monitor unconditional/conditional coverage and exceptions by regime.",
            },
        ],
        "limitations": [
            "Post-selection monitoring is descriptive only and does not establish stable future performance.",
            "Development years use yearly refits; holdout uses one final pre-2025 refit. Comparing pooled metrics conflates changing economic regimes and fitting schemes.",
            "PSI bins and thresholds are research heuristics, not Citi-approved control limits.",
            "The report does not claim input-feature drift without actual feature-level holdout observations.",
            "No SHAP interpretation is inferred from probability drift, and neither model's feature relevance is causal.",
            "Raw data licensing, vendor publication timestamps, and production monitoring alerts remain unverified.",
        ],
    }
    return pd.DataFrame(monitoring_rows), pd.DataFrame(drift_rows), report


def _risk_register(report: dict) -> str:
    lines = [
        "# RiskLens — Model Risk Register (Research Prototype)",
        "",
        "| ID | Risk | Evidence | Mitigation |",
        "|---|---|---|---|",
    ]
    for item in report["risk_register"]:
        lines.append(
            f"| {item['id']} | {item['risk']} | {item['evidence']} | {item['control']} |"
        )
    lines += [
        "",
        "Controls are demonstrations only; not bank/model-owner approvals.",
        "",
    ]
    return "\n".join(lines)


def _markdown(report: dict) -> str:
    lines = [
        "# RiskLens Phase 6 — Score and Performance Monitoring",
        "",
        "**Preselected champion:** fixed 11-feature XGBoost. "
        "**Challenger:** Optuna LSTM trial #86.",
        "",
        "| Model | Development Brier | Holdout Brier | Holdout ROC-AUC | Score PSI |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, data in report["models"].items():
        lines.append(
            f"| {name} | {data['development']['metrics']['brier_score']:.6f} | "
            f"{data['holdout']['metrics']['brier_score']:.6f} | "
            f"{data['holdout']['metrics']['roc_auc']} | "
            f"{data['score_drift']['score_population_stability_index']:.5f} |"
        )
    lines += [
        "", "## Event-rate drift",
        *[f"- {year}: {value:.2%} observed downside events" for year, value in report["holdout_annual_event_frequencies"].items()],
        "", "## Limitations",
        *[f"- {msg}" for msg in report["limitations"]],
        "",
        "This is a retrospective audit. No model retraining, threshold selection or champion promotion.",
        "",
    ]
    return "\n".join(lines)


def save_monitoring(
    development_predictions: str | Path,
    holdout_predictions: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict:
    out = Path(output_dir)
    targets = [out / x for x in OUTPUTS]
    if not overwrite and any(x.exists() for x in targets):
        raise MonitoringError("Phase 6 monitoring outputs exist; specify --overwrite explicitly")
    by_year, bins, report = evaluate_monitoring(development_predictions, holdout_predictions)
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as td:
        tmp = Path(td)
        (tmp / OUTPUTS[0]).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        by_year.to_csv(tmp / OUTPUTS[1], index=False)
        bins.to_csv(tmp / OUTPUTS[2], index=False)
        (tmp / OUTPUTS[3]).write_text(_markdown(report), encoding="utf-8")
        (tmp / OUTPUTS[4]).write_text(_risk_register(report), encoding="utf-8")
        for dest in targets:
            (tmp / dest.name).replace(dest)
    return report
