"""Predeclared EMA50 / India VIX / RSI ablations, with causal regime alternatives.

RESEARCH-ONLY. NO Optuna trials. NO 2025+ features, predictions or outcomes.
Evaluation years 2020-2022 were previously tuned; 2023-2024 have already
been inspected. All results are development diagnostics, NOT new holdout scores.
The original fixed XGBoost and trial-86 hyperparameters are never changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

from risklens.evaluation.features import MULTIFACTOR_FEATURE_COLUMNS
from risklens.evaluation.splits import FINAL_HOLDOUT_START, FOLD_YEARS
from risklens.features.dataset import TARGET_COLUMN
from risklens.models.frozen_comparison import TRIAL86, verify_trial86
from risklens.models.optuna_research import _lstm_one, make_folds
from risklens.models.sequence_lstm import make_sequences
from risklens.models.xgboost_walk_forward import FIXED_XGB_PARAMS

FULL = tuple(MULTIFACTOR_FEATURE_COLUMNS)
FLAGGED = ("ema_50_distance", "india_vix_level", "rsi_14")
EXTRA = ("ema_below_50", "rsi_below_40", "vix_above_63session_median")
RELATIVE = "vix_relative_63session_median"
TUNE_YEARS = (2020, 2021, 2022)
CHECK_YEARS = (2023, 2024)
MODEL_FAMILIES = ("xgboost_fixed", "lstm_trial86_hyperparameters")
OUTPUTS = (
    "experiment_manifest.json",
    "experiment_yearly.csv",
    "experiment_period_comparison.csv",
    "experiment_uncertainty.csv",
    "experiment_predictions.csv",
    "experiment_report.md",
)

# Predeclared BEFORE reading experiment metrics; no Optuna and no adaptive
# additions. We only compare a very small set of focused hypotheses.
VARIANTS: dict[str, tuple[str, ...]] = {
    "full_11": FULL,
    "remove_ema50": tuple(f for f in FULL if f != "ema_50_distance"),
    "remove_india_vix": tuple(f for f in FULL if f != "india_vix_level"),
    "remove_rsi14": tuple(f for f in FULL if f != "rsi_14"),
    "remove_all_three": tuple(f for f in FULL if f not in FLAGGED),
    "full_plus_regime_flags": FULL + EXTRA,
    "vix_relative_replaces_raw": tuple(
        f for f in FULL if f != "india_vix_level"
    ) + (RELATIVE,),
    "full_plus_regime_context": FULL + EXTRA + (RELATIVE,),
}


class DriftExperimentError(ValueError):
    """Unsafe evaluation boundary or noncomparable experimental data."""


def _check_protocol() -> None:
    if tuple(FOLD_YEARS) != TUNE_YEARS + CHECK_YEARS:
        raise DriftExperimentError("Changed development fold years; refuse experiment")
    if str(FINAL_HOLDOUT_START) != "2025-01-01":
        raise DriftExperimentError("Changed final-holdout cutoff; refuse experiment")
    if len(FULL) != 11 or set(FLAGGED) - set(FULL):
        raise DriftExperimentError("Unexpected frozen 11-feature specification")
    if len(VARIANTS) != len(set(VARIANTS)):
        raise DriftExperimentError("Duplicated model-variant label")
    if any(len(cols) != len(set(cols)) for cols in VARIANTS.values()):
        raise DriftExperimentError("Feature names duplicated in variant")
    if TRIAL86["lookback"] != 45:
        raise DriftExperimentError("Unexpected trial86 sequence length")


def _lagged_regime_history(history: pd.DataFrame) -> pd.DataFrame:
    """Only already-known current/past features, with no target or future shifts.

    India VIX at NIFTY date t already came from a STRICTLY EARLIER VIX
    source date. Its rolling median is based on that lagged, known series,
    including t; this is *not* a look-ahead join.
    """
    needed = {"date", *FULL}
    if not needed.issubset(history.columns):
        raise DriftExperimentError("History lacks original frozen features")
    h = history.copy()
    if h.date.duplicated().any() or not h.date.is_monotonic_increasing:
        raise DriftExperimentError("History chronology invalid")
    if (h.date >= pd.Timestamp(FINAL_HOLDOUT_START)).any():
        raise DriftExperimentError("Attempt to use sealed 2025+ feature history")
    vix = h["india_vix_level"].astype(float)
    ema = h["ema_50_distance"].astype(float)
    rsi = h["rsi_14"].astype(float)
    median = vix.rolling(window=63, min_periods=1).median()
    h[RELATIVE] = (vix / median) - 1.0
    h["ema_below_50"] = (ema < 0).astype(float).where(ema.notna())
    h["rsi_below_40"] = (rsi < 40).astype(float).where(rsi.notna())
    h["vix_above_63session_median"] = (vix > median).astype(float).where(
        vix.notna() & median.notna()
    )
    return h


def _align_fold_views(history: pd.DataFrame, train: pd.DataFrame,
                      valid: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All 8 candidates use EXACTLY the same eligible training/test dates."""
    view_features = tuple(dict.fromkeys(
        feature for cols in VARIANTS.values() for feature in cols
    ))
    align = history.loc[:, ["date", *view_features]]
    label_train = train[["date", TARGET_COLUMN]].merge(
        align, on="date", how="left", validate="one_to_one"
    )
    label_valid = valid[["date", TARGET_COLUMN]].merge(
        align, on="date", how="left", validate="one_to_one"
    )
    if label_train[list(view_features)].isna().any().any() or label_valid[list(view_features)].isna().any().any():
        raise DriftExperimentError(f"{year}: scenario features absent on baseline dates")
    for df in (label_train, label_valid):
        if not np.isfinite(df[list(view_features)].to_numpy(dtype=float)).all():
            raise DriftExperimentError(f"{year}: nonfinite candidate features")
        if df.date.duplicated().any() or not df.date.is_monotonic_increasing:
            raise DriftExperimentError(f"{year}: unsorted or duplicate feature dates")
    if (
        label_train.empty or label_valid.empty
        or label_train.date.max() >= pd.Timestamp(f"{year}-01-01")
        or label_valid.date.min() < pd.Timestamp(f"{year}-01-01")
        or label_valid.date.max() >= pd.Timestamp(f"{year+1}-01-01")
        or label_valid.date.max() >= pd.Timestamp(FINAL_HOLDOUT_START)
    ):
        raise DriftExperimentError(f"{year}: unsafe chronological fold")
    if not label_train[["date", TARGET_COLUMN]].equals(
        train[["date", TARGET_COLUMN]].reset_index(drop=True)
    ) or not label_valid[["date", TARGET_COLUMN]].equals(
        valid[["date", TARGET_COLUMN]].reset_index(drop=True)
    ):
        raise DriftExperimentError(f"{year}: variant changed original dates/labels")
    return label_train, label_valid


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    if not len(y) or len(p) != len(y) or not np.isfinite(p).all() or (
        ((p < 0) | (p > 1)).any()
    ) or not np.isin(y, (0, 1)).all():
        raise DriftExperimentError("Invalid binary labels or predictions")
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "brier": float(np.mean((p-y)**2)),
        "logloss": float(log_loss(y, np.clip(p, 1e-15, 1-1e-15), labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, p)) if y.sum() else None,
    }


def _period_mask(p: pd.DataFrame, period: str) -> pd.Series:
    return p.validation_year.isin(
        TUNE_YEARS if period == "previous_tuning_2020_2022"
        else CHECK_YEARS if period == "previous_diagnostics_2023_2024"
        else TUNE_YEARS + CHECK_YEARS
    )


def _block_bootstrap_differences(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    repetitions: int = 400,
    block: int = 10,
    seed: int = 2026,
) -> dict:
    """Descriptive paired/within-year circular block CIs, NOT a new test set."""
    if repetitions < 0 or block < 1:
        raise DriftExperimentError("Invalid bootstrap parameters")
    if not baseline[["date", "actual"]].equals(candidate[["date", "actual"]]):
        raise DriftExperimentError("Bootstrap requires identical dates and outcomes")
    if repetitions == 0:
        return {"replicates": 0, "auc_delta_ci95": None, "brier_gain_ci95": None}
    rng = np.random.default_rng(seed)
    yrs = baseline.validation_year.to_numpy(dtype=int)
    y = baseline.actual.to_numpy(dtype=int)
    p0 = baseline.probability.to_numpy(dtype=float)
    p1 = candidate.probability.to_numpy(dtype=float)
    blocks = [np.flatnonzero(yrs == yr) for yr in np.unique(yrs)]
    auc_deltas, brier_gains = [], []
    for _ in range(repetitions):
        samples = []
        for indices in blocks:
            n = len(indices)
            starts = rng.integers(0, n, size=(n+block-1)//block)
            chunks = [indices[(np.arange(block)+start) % n] for start in starts]
            samples.append(np.concatenate(chunks)[:n])
        choice = np.concatenate(samples)
        yy, base, alt = y[choice], p0[choice], p1[choice]
        brier_gains.append(float(np.mean((yy-base)**2 - (yy-alt)**2)))
        if len(np.unique(yy)) == 2:
            auc_deltas.append(float(roc_auc_score(yy, alt)-roc_auc_score(yy, base)))
    def bounds(v):
        return [float(x) for x in np.quantile(v, [0.025, 0.975])] if v else None
    return {
        "replicates": repetitions,
        "valid_auc_replicates": len(auc_deltas),
        "auc_delta_ci95": bounds(auc_deltas),
        "brier_gain_ci95": bounds(brier_gains),
        "method": f"paired circular {block}-session blocks within each year",
        "warning": "Exploratory repeated-development uncertainty, NOT unbiased out-of-sample validation",
    }


def _fit_one_xgb(train: pd.DataFrame, valid: pd.DataFrame,
                 columns: tuple[str, ...]) -> np.ndarray:
    model = XGBClassifier(**FIXED_XGB_PARAMS)
    model.fit(train[list(columns)].to_numpy(dtype=float),
              train[TARGET_COLUMN].to_numpy(dtype=int))
    return np.asarray(model.predict_proba(valid[list(columns)].to_numpy(dtype=float))[:, 1], dtype=float)


def _fit_one_lstm(history: pd.DataFrame, train: pd.DataFrame, valid: pd.DataFrame,
                  columns: tuple[str, ...], year: int, device: str) -> np.ndarray:
    tx, ty, td = make_sequences(history, train, columns, TRIAL86["lookback"])
    vx, vy, vd = make_sequences(history, valid, columns, TRIAL86["lookback"])
    if not td.max() < vd.min() or not vd.equals(pd.DatetimeIndex(valid.date)):
        raise DriftExperimentError(f"{year}: LSTM windows dropped or rearranged validation dates")
    if not np.array_equal(vy, valid[TARGET_COLUMN].to_numpy(dtype=int)):
        raise DriftExperimentError(f"{year}: LSTM labels mismatch")
    return np.asarray(_lstm_one(tx, ty, vx, dict(TRIAL86),
                      seed=2026+year, device=device), dtype=float)


def evaluate_dev_hypotheses(
    nifty_csv: str | Path,
    external_csv: str | Path,
    study_db: str | Path,
    *,
    models: str = "both",
    device: str = "auto",
    bootstrap_samples: int = 400,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Fit frozen hyperparams on 2020–24 folds; NEVER touch final holdout."""
    _check_protocol()
    if models not in ("both", "xgb", "lstm"):
        raise DriftExperimentError("models must be both, xgb or lstm")
    if device not in ("auto", "cpu", "mps"):
        raise DriftExperimentError("device must be auto, cpu or mps")
    if not isinstance(bootstrap_samples, int) or not 0 <= bootstrap_samples <= 5000:
        raise DriftExperimentError("bootstrap_samples must be 0..5000")
    for path in (nifty_csv, external_csv, study_db):
        if not Path(path).is_file():
            raise DriftExperimentError(f"Required development input missing: {path}")
    trial_provenance = verify_trial86(study_db)
    causal_history, folds = make_folds(nifty_csv, external_csv, years=tuple(FOLD_YEARS))
    if [yr for yr, _, _ in folds] != list(FOLD_YEARS):
        raise DriftExperimentError("Incorrect historical development folds")
    h = _lagged_regime_history(causal_history)
    predictions = []
    fold_report = []
    active = (
        MODEL_FAMILIES if models == "both"
        else (MODEL_FAMILIES[0],) if models == "xgb" else (MODEL_FAMILIES[1],)
    )
    for year, old_train, old_valid in folds:
        train, valid = _align_fold_views(h, old_train, old_valid, year)
        y = valid[TARGET_COLUMN].to_numpy(dtype=int)
        for family in active:
            for label, features in VARIANTS.items():
                if family == "xgboost_fixed":
                    prob = _fit_one_xgb(train, valid, features)
                else:
                    prob = _fit_one_lstm(h, train, valid, features, year, device)
                score = _metrics(y, prob)
                fold_report.append({
                    "model": family, "variant": label,
                    "year": year, "period": "previous_tuning" if year in TUNE_YEARS
                    else "previous_diagnostic",
                    "n_features": len(features), **score,
                })
                predictions.extend({
                    "model": family, "variant": label,
                    "validation_year": year,
                    "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                    "actual": int(target), "probability": float(p),
                } for d, target, p in zip(valid.date, y, prob))

    by_year = pd.DataFrame(fold_report)
    baseline_year = by_year.loc[by_year.variant == "full_11", [
        "model", "year", "brier", "roc_auc", "average_precision"
    ]].rename(columns={
        "brier": "full_11_brier",
        "roc_auc": "full_11_roc_auc",
        "average_precision": "full_11_average_precision",
    })
    by_year = by_year.merge(
        baseline_year, on=["model", "year"], how="left", validate="many_to_one"
    )
    by_year["brier_gain_vs_full_11"] = by_year.full_11_brier - by_year.brier
    by_year["auc_delta_vs_full_11"] = by_year.roc_auc - by_year.full_11_roc_auc
    by_year["ap_delta_vs_full_11"] = (
        by_year.average_precision - by_year.full_11_average_precision
    )
    preds = pd.DataFrame(predictions)
    if preds.empty or preds.date.ge("2025-01-01").any():
        raise DriftExperimentError("No development predictions or 2025+ contamination")
    summaries, interval_rows = [], []
    for family in active:
        family_pred = preds.loc[preds.model == family]
        for era in ("previous_tuning_2020_2022", "previous_diagnostics_2023_2024", "all_development_descriptive"):
            era_df = family_pred.loc[_period_mask(family_pred, era)]
            baseline = era_df.loc[era_df.variant == "full_11"].sort_values("date").reset_index(drop=True)
            bmetrics = _metrics(baseline.actual.to_numpy(int), baseline.probability.to_numpy(float))
            for variant in VARIANTS:
                candidate = era_df.loc[era_df.variant == variant].sort_values("date").reset_index(drop=True)
                if not baseline[["date", "actual"]].equals(candidate[["date", "actual"]]):
                    raise DriftExperimentError(f"{family}/{era}/{variant}: unequal prediction dates/labels")
                metrics = _metrics(candidate.actual.to_numpy(int), candidate.probability.to_numpy(float))
                row = {
                    "model": family, "period": era, "variant": variant,
                    "n_features": len(VARIANTS[variant]), **metrics,
                    "auc_delta_vs_full_11": (
                        metrics["roc_auc"]-bmetrics["roc_auc"] if (
                            metrics["roc_auc"] is not None and bmetrics["roc_auc"] is not None
                        ) else None
                    ),
                    "brier_gain_vs_full_11": bmetrics["brier"]-metrics["brier"],
                    "logloss_gain_vs_full_11": bmetrics["logloss"]-metrics["logloss"],
                    "ap_delta_vs_full_11": (
                        metrics["average_precision"]-bmetrics["average_precision"] if (
                            metrics["average_precision"] is not None and bmetrics["average_precision"] is not None
                        ) else None
                    ),
                }
                summaries.append(row)
                if variant != "full_11":
                    unc = _block_bootstrap_differences(
                        baseline, candidate, repetitions=bootstrap_samples,
                        block=10, seed=2026 + len(variant)*137 + len(era),
                    )
                    interval_rows.append({
                        "model": family, "period": era, "variant": variant,
                        "bootstrap_valid_auc_replicates": unc.get("valid_auc_replicates"),
                        "auc_delta_ci95_lower": (unc["auc_delta_ci95"][0] if unc["auc_delta_ci95"] else None),
                        "auc_delta_ci95_upper": (unc["auc_delta_ci95"][1] if unc["auc_delta_ci95"] else None),
                        "brier_gain_ci95_lower": (unc["brier_gain_ci95"][0] if unc["brier_gain_ci95"] else None),
                        "brier_gain_ci95_upper": (unc["brier_gain_ci95"][1] if unc["brier_gain_ci95"] else None),
                        "bootstrap_replicates": unc["replicates"],
                    })

    report = {
        "title": "RiskLens: EMA50/VIX/RSI DEVELOPMENT-ONLY hypothesis experiments",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "holdout_status": "UNTOUCHED by this experiment; pre-2025 data ONLY",
        "already_opened_2025_2026": "Historical final holdout results cannot be reused for candidate selection",
        "versions": {"protocol": "frozen-ablation-regime-v1"},
        "selection_warning": (
            "Hypotheses were motivated by an already-opened 2025-2026 holdout. "
            "Even 2020-2024 comparisons are exploratory, not independent proof "
            "of generalization. Future 2027+ independent evaluation is required "
            "to claim a novel model has improved forward performance."
        ),
        "training": "purged expanding chronological pre-validation development history",
        "evaluation_folds": list(FOLD_YEARS),
        "previous_tuning_years": list(TUNE_YEARS),
        "repeated_diagnostic_years": list(CHECK_YEARS),
        "source_sha256": {
            "nifty": sha256(Path(nifty_csv).read_bytes()).hexdigest(),
            "development_external": sha256(Path(external_csv).read_bytes()).hexdigest(),
        },
        "trial_86_provenance": trial_provenance,
        "xgb_fixed_params": dict(FIXED_XGB_PARAMS),
        "lstm_trial_86_hyperparams": dict(TRIAL86),
        "methods": list(active),
        "variants": {key: list(value) for key, value in VARIANTS.items()},
        "causal_regime_variables": {
            "ema_below_50": "1 when index close[t] < EMA50[t] using close[t]",
            "rsi_below_40": "1 when Wilder RSI[t] < 40, threshold predeclared for experiment",
            "vix_above_63session_median": "1 when point-in-time lagged VIX[t] > median(last 63 lagged-VIX NIFTY-session values, inclusive of t)",
            "vix_relative_63session_median": "lagged VIX[t]/median(last 63 point-in-time lagged VIX values) - 1",
        },
        "bootstrap": "400 (default) paired circular 10-session within-year resamples; descriptive CI only",
        "safety": [
            "No Optuna or hyperparameter search",
            "No final-holdout re-scoring, refitting, recalibration or champion change",
            "Frozen parameters and feature construction for original 11 predictors",
            "All candidates on exact same development prediction dates",
            "VIX input comes solely from already-verified previous-date external join",
            "New regime flags use no future observations or outcomes",
            "Do not select a variant using formerly unopened 2025-2026 results",
        ],
    }
    return preds, by_year, pd.DataFrame(summaries), pd.DataFrame(interval_rows), report


def _markdown(summary: pd.DataFrame, report: dict) -> str:
    lines = [
        "# RiskLens — Three-Feature Hypothesis Experiment",
        "",
        "**Research-only results, previously examined 2020–2024 development.**",
        "**2025–2026 final holdout was NOT opened by this experiment.**",
        "",
        "Direction: +AUC delta and +Brier gain favor the candidate versus full_11.",
        "",
    ]
    for method in report["methods"]:
        lines += [f"## {method}", ""]
        for period in ("previous_tuning_2020_2022", "previous_diagnostics_2023_2024"):
            lines += [f"### {period}", "",
                      "| Variant | ROC-AUC | ΔAUC | Brier | Brier gain | AP |",
                      "|---|---:|---:|---:|---:|---:|"]
            for row in summary.loc[(summary.model == method) & (summary.period == period)].itertuples(index=False):
                lines.append(
                    f"| {row.variant} | {row.roc_auc:.5f} | {row.auc_delta_vs_full_11:+.5f} | "
                    f"{row.brier:.5f} | {row.brier_gain_vs_full_11:+.5f} | {row.average_precision:.5f} |"
                )
            lines.append("")
    lines += [
        "## Interpretation rules", "",
        "- Do not treat a greater 2020–2024 AUC as an improvement on the already-opened 2025–2026 final holdout.",
        "- Compare year-by-year consistency, AP, Brier and confidence intervals—not only pooled ROC-AUC.",
        "- A single positive result can easily be chance or multiple-comparison selection bias across eight variants.",
        "- Dropping a drifting feature is not justified solely by its PSI; feature value is a conditional question.",
        "- Do not change RiskLens main-branch champion or frozen claim without fresh future validation.",
        "", "## Study limitations", "",
        report["selection_warning"], "",
    ]
    return "\n".join(lines)


def save_experiment(nifty_csv: str | Path, external_csv: str | Path,
                    study_db: str | Path, out_dir: str | Path, *,
                    models: str = "both", device: str = "auto",
                    bootstrap_samples: int = 400, overwrite: bool = False) -> dict:
    out = Path(out_dir)
    targets = [out / filename for filename in OUTPUTS]
    if not overwrite and any(path.exists() for path in targets):
        raise DriftExperimentError("Experiment output already exists; use --overwrite deliberately")
    predictions, yearly, summary, uncertainty, report = evaluate_dev_hypotheses(
        nifty_csv, external_csv, study_db,
        models=models, device=device, bootstrap_samples=bootstrap_samples,
    )
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as td:
        temp = Path(td)
        (temp / OUTPUTS[0]).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        yearly.to_csv(temp / OUTPUTS[1], index=False)
        summary.to_csv(temp / OUTPUTS[2], index=False)
        uncertainty.to_csv(temp / OUTPUTS[3], index=False)
        predictions.to_csv(temp / OUTPUTS[4], index=False, float_format="%.12g")
        (temp / OUTPUTS[5]).write_text(_markdown(summary, report))
        for target in targets:
            (temp / target.name).replace(target)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Research-only pre-2025 model feature ablation and causal regimes"
    )
    parser.add_argument("--nifty", default="data/processed/multiyear/nifty50_ohlc.csv")
    parser.add_argument("--external", default="data/processed/external/nifty50_external_development.csv")
    parser.add_argument("--study-db", default="data/processed/optuna_search/optuna_studies.sqlite3")
    parser.add_argument("--models", choices=("both", "xgb", "lstm"), default="both")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=400)
    parser.add_argument("--output-dir", default="data/processed/drift_hypotheses_development")
    parser.add_argument("--overwrite", action="store_true")
    options = parser.parse_args(argv)
    report = save_experiment(
        options.nifty, options.external, options.study_db, options.output_dir,
        models=options.models, device=options.device,
        bootstrap_samples=options.bootstrap_samples, overwrite=options.overwrite,
    )
    print("PASS: completed DEVELOPMENT ONLY feature-hypothesis experiment")
    print("Models:", ", ".join(report["methods"]))
    print("Variants:", len(report["variants"]))
    print("Output:", Path(options.output_dir) / "experiment_report.md")
    print("NO 2025+ holdout metrics were evaluated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
