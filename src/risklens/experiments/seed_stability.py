"""Fixed-seed stability audit for two *existing* LSTM feature configurations.

Research only: validation 2020–2024, no 2025+ data, no Optuna, no new feature
selection, and NO alteration to RiskLens's locked 2025–2026 holdout.

Predetermined seeds pair full_11 against full_plus_regime_context with exactly
one frozen Trial #86 training run per (year, seed, variant). No best-seed picks.
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

from risklens.evaluation.splits import FINAL_HOLDOUT_START, FOLD_YEARS
from risklens.features.dataset import TARGET_COLUMN
from risklens.models.frozen_comparison import TRIAL86, verify_trial86
from risklens.models.optuna_research import _lstm_one, make_folds
from risklens.models.sequence_lstm import make_sequences
from risklens.experiments.drift_hypotheses import (
    FULL, VARIANTS, _align_fold_views, _check_protocol, _lagged_regime_history,
)

VARIANT_NAMES = ("full_11", "full_plus_regime_context")
YEARS = (2020, 2021, 2022, 2023, 2024)
PERIODS = {
    "prior_tuning_2020_2022": (2020, 2021, 2022),
    "previous_diagnostics_2023_2024": (2023, 2024),
    "all_2020_2024_descriptive": YEARS,
}
# Prespecified BEFORE these runs. At offset zero we reproduce the original
# deterministic call in the existing frozen research code: 2026 + year.
SEED_OFFSETS = (0, 17, 43, 79, 131, 191, 263)
OUTPUTS = (
    "seed_stability_manifest.json",
    "seed_stability_predictions.csv",
    "seed_stability_yearly.csv",
    "seed_stability_periods.csv",
    "seed_stability_summary.csv",
    "seed_stability_report.md",
)


class SeedStabilityError(ValueError):
    """Data leak, altered research protocol, or incomparable fold."""


def _check_locked_design() -> None:
    _check_protocol()
    if tuple(FOLD_YEARS) != YEARS or str(FINAL_HOLDOUT_START) != "2025-01-01":
        raise SeedStabilityError("Historical folds or sealed holdout boundary changed")
    if VARIANT_NAMES != ("full_11", "full_plus_regime_context"):
        raise SeedStabilityError("Variant roster changed")
    if tuple(VARIANTS["full_11"]) != tuple(FULL):
        raise SeedStabilityError("Original feature set differs from frozen set")
    if tuple(VARIANTS["full_plus_regime_context"][:len(FULL)]) != tuple(FULL):
        raise SeedStabilityError("Regime context no longer augments original features")
    if len(VARIANTS["full_plus_regime_context"]) != len(FULL) + 4:
        raise SeedStabilityError("Unexpected regime context feature count")
    if len(SEED_OFFSETS) != 7 or len(set(SEED_OFFSETS)) != 7 or SEED_OFFSETS[0] != 0:
        raise SeedStabilityError("Seed schedule was modified")
    if TRIAL86["lookback"] != 45 or TRIAL86["epochs"] != 24:
        raise SeedStabilityError("Frozen Optuna Trial #86 specification changed")


def _score(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    if not len(y) or len(y) != len(p) or not np.isin(y, (0, 1)).all():
        raise SeedStabilityError("Labels or prediction length invalid")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise SeedStabilityError("Predicted probabilities outside [0,1]")
    return {
        "n": len(y),
        "events": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "brier": float(np.mean((y-p)**2)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1-1e-15), labels=[0,1])),
        "average_precision": float(average_precision_score(y, p)) if y.sum() else None,
    }


def _paired_scores(rows: pd.DataFrame, *, year: int | None = None,
                   period: str | None = None) -> dict:
    """Score both variants on identical dates/labels for ONE seed and scope."""
    a = rows.loc[rows.variant == VARIANT_NAMES[0]].sort_values("date").reset_index(drop=True)
    b = rows.loc[rows.variant == VARIANT_NAMES[1]].sort_values("date").reset_index(drop=True)
    if a.empty or not a[["date", "actual", "validation_year"]].equals(
        b[["date", "actual", "validation_year"]]
    ):
        raise SeedStabilityError("Full versus context use different dates/outcomes")
    if a.date.duplicated().any() or not a.date.is_monotonic_increasing:
        raise SeedStabilityError("Duplicate or unsorted validation dates")
    if a.date.ge("2025-01-01").any() or not set(a.validation_year).issubset(YEARS):
        raise SeedStabilityError("2025+ or invalid validation-year access")
    full = _score(a.actual.to_numpy(dtype=int), a.probability.to_numpy(dtype=float))
    context = _score(b.actual.to_numpy(dtype=int), b.probability.to_numpy(dtype=float))
    return {
        "year": year, "period": period,
        "n": full["n"], "events": full["events"],
        **{f"original_{name}": value for name, value in full.items() if name not in ("n", "events")},
        **{f"context_{name}": value for name, value in context.items() if name not in ("n", "events")},
        "delta_auc": (
            context["roc_auc"] - full["roc_auc"]
            if context["roc_auc"] is not None and full["roc_auc"] is not None else None
        ),
        "brier_gain": full["brier"] - context["brier"],
        "log_loss_gain": full["log_loss"] - context["log_loss"],
        "delta_ap": (
            context["average_precision"] - full["average_precision"]
            if context["average_precision"] is not None and full["average_precision"] is not None else None
        ),
    }


def _stability_summary(periods: pd.DataFrame, yearly: pd.DataFrame) -> pd.DataFrame:
    """Variation across paired training seeds, NOT a sampling-theory CI."""
    items = []
    for scope_name, scope in (("pooled", periods), ("annual", yearly)):
        key = "period" if scope_name == "pooled" else "year"
        for group, df in scope.groupby(key, sort=True):
            if len(df) != len(SEED_OFFSETS) or df.seed_index.nunique() != len(SEED_OFFSETS):
                raise SeedStabilityError("Missing or duplicated seed in summary")
            record = {"scope": scope_name, "group": str(group), "seeds": len(df),
                      "observations_per_seed": int(df.n.iloc[0]),
                      "events_per_seed": int(df.events.iloc[0])}
            if df.n.nunique() != 1 or df.events.nunique() != 1:
                raise SeedStabilityError("Seed experiments changed evaluation population")
            for field in ("delta_auc", "brier_gain", "delta_ap", "log_loss_gain"):
                vals = df[field].dropna().to_numpy(dtype=float)
                record[f"{field}_mean"] = float(vals.mean()) if len(vals) else None
                record[f"{field}_median"] = float(np.median(vals)) if len(vals) else None
                record[f"{field}_min"] = float(vals.min()) if len(vals) else None
                record[f"{field}_max"] = float(vals.max()) if len(vals) else None
                record[f"{field}_positive_seeds"] = int((vals > 0).sum())
                record[f"{field}_valid_seeds"] = len(vals)
            items.append(record)
    return pd.DataFrame(items)


def run_seed_stability(nifty_csv: str | Path, external_csv: str | Path,
                       study_db: str | Path, *, device: str = "auto") -> tuple[
                       pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Same fold/train data across seeds and variants; no feature search."""
    _check_locked_design()
    if device not in ("auto", "cpu", "mps"):
        raise SeedStabilityError("device must be auto, cpu or mps")
    for name, path in (("NIFTY", nifty_csv), ("External development", external_csv),
                       ("Optuna study", study_db)):
        if not Path(path).is_file():
            raise SeedStabilityError(f"{name} input missing: {path}")
    provenance = verify_trial86(study_db)  # inspect existing record; do NOT launch Optuna
    history, folds = make_folds(nifty_csv, external_csv, years=YEARS)
    if [year for year, _, _ in folds] != list(YEARS):
        raise SeedStabilityError("Unexpected or missing chronological development folds")
    h = _lagged_regime_history(history)
    predictions = []
    for year, previous_train, previous_valid in folds:
        train, valid = _align_fold_views(h, previous_train, previous_valid, year)
        if valid.date.max() >= pd.Timestamp(FINAL_HOLDOUT_START):
            raise SeedStabilityError("Final holdout data requested")
        sequences = {}
        for variant in VARIANT_NAMES:
            columns = VARIANTS[variant]
            tx, ty, td = make_sequences(h, train, columns, TRIAL86["lookback"])
            vx, vy, vd = make_sequences(h, valid, columns, TRIAL86["lookback"])
            if (not vd.equals(pd.DatetimeIndex(valid.date)) or
                not np.array_equal(vy, valid[TARGET_COLUMN].to_numpy(dtype=int)) or
                td.max() >= vd.min() or td.max() >= pd.Timestamp(f"{year}-01-01") or
                len(np.unique(ty)) != 2):
                raise SeedStabilityError(f"{year}/{variant}: invalid date/label alignment")
            sequences[variant] = (tx, ty, vx)
        if (sequences[VARIANT_NAMES[0]][0].shape[0] !=
                sequences[VARIANT_NAMES[1]][0].shape[0]):
            raise SeedStabilityError(f"{year}: unequal training windows")
        for seed_index, offset in enumerate(SEED_OFFSETS):
            seed = 2026 + year + offset
            for variant in VARIANT_NAMES:
                tx, ty, vx = sequences[variant]
                p = np.asarray(
                    _lstm_one(tx, ty, vx, dict(TRIAL86), seed=seed, device=device),
                    dtype=float,
                )
                _score(valid[TARGET_COLUMN].to_numpy(dtype=int), p)
                for date, actual, prob in zip(valid.date,
                                              valid[TARGET_COLUMN].to_numpy(dtype=int), p):
                    predictions.append({
                        "variant": variant, "seed_index": seed_index,
                        "seed_offset": offset, "training_seed": seed,
                        "validation_year": year,
                        "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                        "actual": int(actual), "probability": float(prob),
                    })

    preds = pd.DataFrame(predictions)
    if preds.empty or preds.date.ge("2025-01-01").any():
        raise SeedStabilityError("No data or holdout access")
    yearly, periods = [], []
    for seed_index in range(len(SEED_OFFSETS)):
        subset = preds[preds.seed_index == seed_index]
        for year in YEARS:
            result = _paired_scores(subset[subset.validation_year == year], year=year)
            yearly.append({"seed_index": seed_index, "seed_offset": SEED_OFFSETS[seed_index],
                           **result})
        for name, years in PERIODS.items():
            result = _paired_scores(subset[subset.validation_year.isin(years)], period=name)
            periods.append({"seed_index": seed_index, "seed_offset": SEED_OFFSETS[seed_index],
                            **result})
    by_year = pd.DataFrame(yearly)
    by_period = pd.DataFrame(periods)
    summary = _stability_summary(by_period, by_year)
    report = {
        "title": "RiskLens — fixed Trial #86 LSTM seed stability / regime-context comparison",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "RESEARCH ONLY: previously inspected 2020-2024 development; NO 2025+ holdout",
        "original_variant": VARIANT_NAMES[0],
        "challenger_variant": VARIANT_NAMES[1],
        "feature_columns": {v: list(VARIANTS[v]) for v in VARIANT_NAMES},
        "trial86_frozen_hyperparameters": dict(TRIAL86),
        "trial86_study_provenance": provenance,
        "seed_offsets": list(SEED_OFFSETS),
        "training_seed_formula": "2026 + validation_year + seed_offset",
        "fold_years": list(YEARS),
        "periods": {name: list(yrs) for name, yrs in PERIODS.items()},
        "training_protocol": "Same purged expanding training histories, same validation dates, same trial #86 parameters and epochs; only feature set and RNG seed vary.",
        "expected_training_runs": len(YEARS)*len(SEED_OFFSETS)*len(VARIANT_NAMES),
        "device": device,
        "source_sha256": {
            "nifty": sha256(Path(nifty_csv).read_bytes()).hexdigest(),
            "development_external": sha256(Path(external_csv).read_bytes()).hexdigest(),
            "study_db": sha256(Path(study_db).read_bytes()).hexdigest(),
        },
        "interpretation_rules": [
            "Compare two PREDECLARED variants only; do not add variants after inspecting results.",
            "Positive delta_auc/delta_ap and positive brier_gain/log_loss_gain favor context.",
            "Inspect both pooled historical periods separately and every individual validation year.",
            "A positive median across seeds or 5/7 positive seeds is exploratory, not statistical proof.",
            "Seeds share the same historical observations; ranges across seeds are NOT 95% confidence intervals.",
            "Do not choose a best random seed, retrain/promote on the opened 2025-26 holdout, or assert future ROC-AUC improvement.",
            "Training and validation on previously consulted 2020-24 data cannot constitute a new independent test.",
            "Future previously unseen data are needed to validate any improved variant.",
            "Apple MPS may be nondeterministic; exact hardware-dependent reruns are not guaranteed.",
        ],
    }
    return preds, by_year, by_period, summary, report


def _markdown(summary: pd.DataFrame, report: dict) -> str:
    lines = [
        "# LSTM Trial #86 — Seed-stability study (research only)", "",
        "**No Optuna runs; no 2025–2026 holdout access; no code or model promotion.**", "",
        f"Compared {report['original_variant']} against {report['challenger_variant']} ",
        f"using {len(SEED_OFFSETS)} predeclared random seeds per validation year, 2020–2024.", "",
        "**Interpretation:** Positive ΔROC-AUC / ΔAP and positive Brier gain favor regime context.", "",
        "| Scope | Group | N | Events | Median ΔAUC | Seeds with ΔAUC > 0 | Median Brier gain | Seeds with Brier gain > 0 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"| {r.scope} | {r.group} | {r.observations_per_seed} | {r.events_per_seed} | "
            f"{r.delta_auc_median:+.5f} | {r.delta_auc_positive_seeds}/{r.delta_auc_valid_seeds} | "
            f"{r.brier_gain_median:+.6f} | {r.brier_gain_positive_seeds}/{r.brier_gain_valid_seeds} |"
        )
    lines += ["", "## Research limitations", ""]
    lines += [f"- {line}" for line in report["interpretation_rules"]]
    return "\n".join(lines) + "\n"


def save_stability(nifty_csv: str | Path, external_csv: str | Path,
                   study_db: str | Path, output_dir: str | Path, *,
                   device: str = "auto", overwrite: bool = False) -> dict:
    out = Path(output_dir)
    if any((out / item).exists() for item in OUTPUTS) and not overwrite:
        raise SeedStabilityError("Outputs already exist; specify --overwrite explicitly")
    preds, by_year, by_period, summary, report = run_seed_stability(
        nifty_csv, external_csv, study_db, device=device
    )
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as td:
        p = Path(td)
        preds.to_csv(p / OUTPUTS[1], index=False, float_format="%.12g")
        by_year.to_csv(p / OUTPUTS[2], index=False)
        by_period.to_csv(p / OUTPUTS[3], index=False)
        summary.to_csv(p / OUTPUTS[4], index=False)
        (p / OUTPUTS[0]).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        (p / OUTPUTS[5]).write_text(_markdown(summary, report), encoding="utf-8")
        for item in OUTPUTS:
            (p / item).replace(out / item)
    return report


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description="Frozen LSTM #86: seven-seed stability research on pre-2025 folds")
    cli.add_argument("--nifty", default="data/processed/multiyear/nifty50_ohlc.csv")
    cli.add_argument("--external", default="data/processed/external/nifty50_external_development.csv")
    cli.add_argument("--study-db", default="data/processed/optuna_search/optuna_studies.sqlite3")
    cli.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    cli.add_argument("--output-dir", default="data/processed/lstm_seed_stability_development")
    cli.add_argument("--overwrite", action="store_true")
    args = cli.parse_args(argv)
    report = save_stability(args.nifty, args.external, args.study_db,
                            args.output_dir, device=args.device,
                            overwrite=args.overwrite)
    print("PASS: completed 7-seed LSTM robustness comparison on development folds only")
    print("Runs:", report["expected_training_runs"])
    print("Output:", Path(args.output_dir) / OUTPUTS[5])
    print("NO 2025-2026 holdout access and NO model selection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
