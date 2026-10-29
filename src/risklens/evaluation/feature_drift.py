"""Phase 6: audited 11-feature distribution drift after the one-time holdout.

The final holdout has already been evaluated and is NOT scored again here.
Reconstructs causal features solely to compare distributions on exactly the
previously saved holdout prediction dates; no label-based feature selection.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from risklens.evaluation.features import MULTIFACTOR_FEATURE_COLUMNS
from risklens.evaluation.final_holdout import (
    _attach_holdout_external,
    _read_holdout_sp500,
    _read_holdout_vix,
)
from risklens.evaluation.monitoring import _read_predictions
from risklens.features.dataset import build_development_features
from risklens.models.sequence_lstm import build_causal_history

FEATURES = MULTIFACTOR_FEATURE_COLUMNS
OUTPUTS = ("feature_drift_summary.json", "feature_drift_table.csv", "feature_drift_report.md")
PSI_INVESTIGATE = 0.20


class FeatureDriftError(ValueError):
    """Missing original sources or invalid point-in-time feature alignment."""


def _development_only_edges(values: np.ndarray) -> np.ndarray:
    """Bin boundaries depend on development data alone, never final holdout data."""
    boundaries = np.unique(np.quantile(values, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    # -/+ infinity makes out-of-development-range holdout values auditable,
    # rather than excluded by histogram's finite bin bounds.
    if len(boundaries) == 1:
        return np.asarray([-np.inf, boundaries[0], np.inf], dtype=float)
    interior = boundaries[1:-1]
    return np.concatenate(([-np.inf], interior, [np.inf]))


def _feature_psi(dev: np.ndarray, hold: np.ndarray) -> tuple[float, int]:
    edges = _development_only_edges(dev)
    dev_counts = np.histogram(dev, bins=edges)[0].astype(float)
    hold_counts = np.histogram(hold, bins=edges)[0].astype(float)
    k = len(dev_counts)
    p = (dev_counts + 0.5) / (len(dev) + 0.5*k)
    q = (hold_counts + 0.5) / (len(hold) + 0.5*k)
    return float(np.sum((q-p) * np.log(q/p))), k


def evaluate_feature_drift(
    nifty_csv: str | Path,
    development_external_csv: str | Path,
    vix_csv: str | Path,
    sp500_csv: str | Path,
    holdout_predictions_csv: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """Report descriptive feature drift; do not refit/re-evaluate either ML model."""
    holdout = _read_predictions(holdout_predictions_csv, holdout=True)
    history, source_audit = build_causal_history(nifty_csv, development_external_csv)
    if (history.date >= pd.Timestamp("2025-01-01")).any():
        raise FeatureDriftError("Development baseline contains holdout dates")
    dev = history.dropna(subset=list(FEATURES)).reset_index(drop=True)
    if len(dev) < 100:
        raise FeatureDriftError("Need at least 100 complete pre-2025 baseline rows")
    # The pre-existing feature builder is the same as in the locked holdout
    # evaluation; it is used here ONLY for causal feature values, not labels.
    _, candidate, _ = build_development_features(
        nifty_csv,
        validation_start="2025-01-01",
        holdout_start="2026-10-01",
        include_ohlc_features=True,
        include_technical_features=True,
    )
    vix, vix_audit = _read_holdout_vix(vix_csv)
    sp500, sp_audit = _read_holdout_sp500(sp500_csv)
    attached, alignment = _attach_holdout_external(candidate, vix, sp500)
    # Evaluate only dates that the original one-time model scoring accepted;
    # never explore alternative subsets based on labels or performance.
    feature_rows = holdout[["date"]].merge(
        attached[["date", *FEATURES]], on="date", how="left", validate="one_to_one"
    )
    if not feature_rows.date.equals(holdout.date):
        raise FeatureDriftError("Holdout feature reconstruction order mismatch")
    if feature_rows[list(FEATURES)].isna().any().any():
        raise FeatureDriftError("Saved model prediction lacks complete causal source features")
    for column in FEATURES:
        if not np.isfinite(feature_rows[column].to_numpy(dtype=float)).all():
            raise FeatureDriftError(f"Nonfinite holdout feature {column}")

    rows = []
    for name in FEATURES:
        before = dev[name].to_numpy(dtype=float)
        after = feature_rows[name].to_numpy(dtype=float)
        psi, bins = _feature_psi(before, after)
        low, high = np.quantile(before, [0.01, 0.99])
        rows.append({
            "feature": name,
            "development_rows": len(before),
            "holdout_rows": len(after),
            "development_mean": float(before.mean()),
            "holdout_mean": float(after.mean()),
            "development_std": float(before.std(ddof=1)),
            "holdout_std": float(after.std(ddof=1)),
            "development_p01": float(low),
            "development_p99": float(high),
            "holdout_outside_development_p01_p99_fraction": float(
                ((after < low) | (after > high)).mean()
            ),
            "population_stability_index": round(psi, 8),
            "num_development_derived_bins": bins,
            "investigate_drift": bool(psi >= PSI_INVESTIGATE),
        })
    result = pd.DataFrame(rows).sort_values(
        "population_stability_index", ascending=False
    ).reset_index(drop=True)
    report = {
        "title": "RiskLens Phase 6 | Feature-level distribution drift",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "Retrospective, unlabeled feature summaries on original frozen holdout prediction dates",
        "development_pre_2025_feature_rows": len(dev),
        "holdout_dates_from_single_existing_prediction_file": len(holdout),
        "feature_list": list(FEATURES),
        "features_with_psi_at_least_0_20": result.loc[
            result.investigate_drift, "feature"
        ].tolist(),
        "source_sha256": {
            "nifty": sha256(Path(nifty_csv).read_bytes()).hexdigest(),
            "development_external": sha256(Path(development_external_csv).read_bytes()).hexdigest(),
            "vix_holdout": sha256(Path(vix_csv).read_bytes()).hexdigest(),
            "sp500_holdout": sha256(Path(sp500_csv).read_bytes()).hexdigest(),
            "already_scored_holdout_predictions": sha256(Path(holdout_predictions_csv).read_bytes()).hexdigest(),
        },
        "development_source_audit": source_audit,
        "holdout_alignment_audit": alignment,
        "holdout_vix_audit": vix_audit,
        "holdout_sp500_audit": sp_audit,
        "warnings": [
            "Uses holdout only after the original final evaluation; this must NOT influence model, features, thresholds or training.",
            "PSI is a descriptive histogram statistic, not proof of causality or a formal hypothesis test.",
            "Quantile boundaries use DEVELOPMENT values only; duplicate boundaries collapse to fewer bins.",
            "Baseline feature rows are all available pre-2025 feature-complete dates, not a refit of the model.",
            "Only previously scored holdout dates are analyzed, so availability exclusions are not silently redesigned.",
            "PSI >=0.2 is a research investigation trigger, never a bank-approved limit.",
        ],
    }
    return result, report


def _markdown(summary: pd.DataFrame, report: dict) -> str:
    lines = [
        "# RiskLens — Feature Distribution Drift (After Locked Holdout)",
        "",
        f"Development rows: {report['development_pre_2025_feature_rows']}; "
        f"scored holdout dates: {report['holdout_dates_from_single_existing_prediction_file']}.",
        "",
        "| Feature | Development mean | Holdout mean | PSI | Investigate |",
        "|---|---:|---:|---:|---|",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"| {r.feature} | {r.development_mean:.5f} | {r.holdout_mean:.5f} | "
            f"{r.population_stability_index:.5f} | {r.investigate_drift} |"
        )
    lines += ["", "## Limits",
              *["- " + x for x in report["warnings"]], ""]
    return "\n".join(lines)


def save_feature_drift(
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
    targets = [out / x for x in OUTPUTS]
    if not overwrite and any(t.exists() for t in targets):
        raise FeatureDriftError("Feature-drift reports already exist; pass --overwrite")
    summary, report = evaluate_feature_drift(
        nifty_csv, development_external_csv, vix_csv, sp500_csv,
        holdout_predictions_csv,
    )
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as td:
        temp = Path(td)
        (temp / OUTPUTS[0]).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        summary.to_csv(temp / OUTPUTS[1], index=False)
        (temp / OUTPUTS[2]).write_text(_markdown(summary, report), encoding="utf-8")
        for dest in targets:
            (temp / dest.name).replace(dest)
    return report
