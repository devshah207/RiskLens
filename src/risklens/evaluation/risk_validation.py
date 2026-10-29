"""Phase 5: audited development diagnostics for frozen XGBoost vs Optuna LSTM #86.

Strictly reads 2020–2024 model predictions and matching historical VaR rows.
Never trains, calibrates, optimizes, accesses, or evaluates the sealed 2025+ holdout.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import binomtest, chi2

from risklens.evaluation.metrics import _metrics

YEARS = (2020, 2021, 2022, 2023, 2024)
TARGET = "actual_downside_event"
MODELS = {
    "xgboost_fixed": "xgboost_11_fixed_probability",
    "lstm_optuna_trial86": "lstm_optuna_86_probability",
}
PREDICTION_COLUMNS = ("validation_year", "date", TARGET, *MODELS.values())
VAR_COLUMNS = (
    "date", "validation_year", "next_session_return",
    "var_95_loss_fraction", "var_99_loss_fraction",
    "var_95_exception", "var_99_exception",
)
# Predefined fixed bins; never optimize bins using evaluation labels.
BINS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0)
OUTPUT_FILES = (
    "risk_validation_summary.json",
    "risk_validation_yearly.csv",
    "risk_validation_reliability.csv",
    "risk_validation_cases.csv",
    "risk_validation_report.md",
)


class RiskValidationError(ValueError):
    """Invalid evaluation inputs or unsafe validation boundaries."""


def _read_strict_csv(path: str | Path, columns: tuple[str, ...]) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise RiskValidationError(f"Required input missing: {path}")
    try:
        raw = pd.read_csv(path, dtype="string", encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise RiskValidationError(f"Invalid CSV {path}: {exc}") from exc
    if raw.empty or tuple(raw.columns) != columns:
        raise RiskValidationError(f"{path.name}: exact columns required: {columns}")
    text_dates = raw["date"]
    dates = pd.to_datetime(text_dates, format="%Y-%m-%d", errors="coerce")
    if dates.isna().any() or not dates.dt.strftime("%Y-%m-%d").eq(text_dates).all():
        raise RiskValidationError(f"{path.name}: invalid ISO prediction dates")
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise RiskValidationError(f"{path.name}: dates must be unique and increasing")
    years = raw["validation_year"]
    if not years.str.fullmatch(r"\d{4}", na=False).all():
        raise RiskValidationError(f"{path.name}: invalid validation_year")
    parsed_years = years.astype(int)
    if not np.array_equal(parsed_years.to_numpy(), dates.dt.year.to_numpy()):
        raise RiskValidationError(f"{path.name}: validation_year/date mismatch")
    if (dates.dt.year >= 2025).any() or (dates.dt.year < 2020).any():
        raise RiskValidationError("2025+ final holdout is sealed; only 2020–2024 allowed")
    if tuple(sorted(parsed_years.unique())) != YEARS:
        raise RiskValidationError(f"{path.name}: incomplete 2020–2024 validation years")
    raw["date"] = dates
    raw["validation_year"] = parsed_years
    return raw


def load_predictions(path: str | Path) -> pd.DataFrame:
    frame = _read_strict_csv(path, PREDICTION_COLUMNS)
    if not frame[TARGET].isin(("0", "1")).all():
        raise RiskValidationError("Actual downside labels must be binary")
    frame[TARGET] = frame[TARGET].astype(int)
    for name, column in MODELS.items():
        try:
            x = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise RiskValidationError(f"{name}: malformed probabilities") from exc
        if not np.isfinite(x).all() or ((x < 0) | (x > 1)).any():
            raise RiskValidationError(f"{name}: probabilities must be finite in [0,1]")
        frame[column] = x
    return frame


def load_var(path: str | Path, predictions: pd.DataFrame) -> pd.DataFrame:
    frame = _read_strict_csv(path, VAR_COLUMNS)
    if len(frame) != len(predictions) or not frame["date"].equals(predictions["date"]):
        raise RiskValidationError("VaR and model predictions must have identical dates")
    if not frame["validation_year"].equals(predictions["validation_year"]):
        raise RiskValidationError("VaR/model validation years differ")
    for col in VAR_COLUMNS[2:]:
        try:
            values = pd.to_numeric(frame[col], errors="raise").to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            raise RiskValidationError(f"VaR: invalid values in {col}") from exc
        if not np.isfinite(values).all():
            raise RiskValidationError(f"VaR: nonfinite values in {col}")
        frame[col] = values
    returns = frame["next_session_return"].to_numpy(dtype=float)
    if (returns <= -1).any() or not np.array_equal(
        (returns <= -0.01).astype(int), predictions[TARGET].to_numpy(dtype=int)
    ):
        raise RiskValidationError("VaR realized returns disagree with downside event labels")
    for level in ("95", "99"):
        v = frame[f"var_{level}_loss_fraction"].to_numpy(dtype=float)
        exceptions = frame[f"var_{level}_exception"].to_numpy(dtype=float)
        if (v < 0).any() or not np.isin(exceptions, [0, 1]).all():
            raise RiskValidationError(f"Invalid VaR {level} thresholds or exception flags")
        calculated = (-returns > v).astype(int)
        if not np.array_equal(calculated, exceptions.astype(int)):
            raise RiskValidationError(f"VaR {level} exception flags inconsistent with returns")
        frame[f"var_{level}_exception"] = exceptions.astype(int)
    if (
        frame["var_95_loss_fraction"] > frame["var_99_loss_fraction"] + 1e-12
    ).any():
        raise RiskValidationError("VaR 95% loss threshold exceeds 99% threshold")
    return frame


def _reliability(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    labels = frame[TARGET].to_numpy(dtype=int)
    rows = []
    scores = {}
    for name, column in MODELS.items():
        p = frame[column].to_numpy(dtype=float)
        indices = np.minimum(np.searchsorted(BINS[1:], p, side="right"), len(BINS)-2)
        ece = 0.0
        for i in range(len(BINS)-1):
            mask = indices == i
            count = int(mask.sum())
            mp = float(p[mask].mean()) if count else None
            observed = float(labels[mask].mean()) if count else None
            events = int(labels[mask].sum()) if count else 0
            if count:
                ece += count / len(frame) * abs(mp - observed)
            rows.append({
                "model": name,
                "bin_lower": BINS[i],
                "bin_upper": BINS[i+1],
                "observations": count,
                "events": events,
                "mean_probability": mp,
                "event_frequency": observed,
                "small_bin_warning": bool(count < 30 or events < 5),
            })
        scores[name] = {
            "mean_probability": round(float(p.mean()), 8),
            "event_frequency": round(float(labels.mean()), 8),
            "mean_probability_minus_event_frequency": round(float(p.mean()-labels.mean()), 8),
            "expected_calibration_error_fixed_bins": round(float(ece), 8),
            "brier_reliability_and_resolution_not_estimated": True,
        }
    return pd.DataFrame(rows), scores


def _paired_block_bootstrap(
    frame: pd.DataFrame, *, n_resamples: int, block_length: int, seed: int = 2026
) -> dict:
    if not isinstance(n_resamples, int) or not 100 <= n_resamples <= 10000:
        raise RiskValidationError("Bootstrap samples must be an integer between 100 and 10000")
    if not isinstance(block_length, int) or not 2 <= block_length <= 100:
        raise RiskValidationError("Block length must be an integer between 2 and 100")
    # Positive difference favors the fixed XGBoost candidate.
    y = frame[TARGET].to_numpy(dtype=float)
    e_xgb = (frame[MODELS["xgboost_fixed"]].to_numpy(dtype=float) - y) ** 2
    e_lstm = (frame[MODELS["lstm_optuna_trial86"]].to_numpy(dtype=float) - y) ** 2
    diff = e_lstm - e_xgb
    groups = [np.flatnonzero(frame.validation_year.to_numpy() == year) for year in YEARS]
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        chunks = []
        for group in groups:
            length = len(group)
            width = min(length, block_length)
            count = ceil(length / width)
            starts = rng.integers(0, length, size=count)
            offsets = (starts[:, None] + np.arange(width)) % length
            chunks.append(group[offsets.reshape(-1)[:length]])
        sample = np.concatenate(chunks)
        values[i] = float(diff[sample].mean())
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "metric": "Brier(LSTM trial 86) minus Brier(fixed XGBoost)",
        "observed_delta": round(float(diff.mean()), 8),
        "descriptive_block_bootstrap_95_percentile_interval": [round(float(low), 8), round(float(high), 8)],
        "bootstrap_samples": n_resamples,
        "circular_block_length_sessions": block_length,
        "seed": seed,
        "interpretation": "Positive favors XGBoost. Dependence assumptions and repeated development inspection limit inferential validity.",
    }


def _exception_independence(flags: np.ndarray, years: np.ndarray) -> dict:
    """First-order Markov LR independence diagnostic within years only."""
    counts = np.zeros((2, 2), dtype=int)
    for i in range(1, len(flags)):
        if years[i] == years[i-1]:
            counts[int(flags[i-1]), int(flags[i])] += 1
    n00, n01, n10, n11 = counts.ravel()
    n0, n1 = n00+n01, n10+n11
    if n0 == 0 or n1 == 0:
        return {"transition_counts": counts.tolist(), "lr_independence": None,
                "p_value": None, "warning": "Insufficient transitions from both states"}
    p = (n01+n11) / (n0+n1)
    p01, p11 = n01/n0, n11/n1
    log_ind = xlogy(n00+n10, 1-p) + xlogy(n01+n11, p)
    log_markov = (
        xlogy(n00, 1-p01) + xlogy(n01, p01)
        + xlogy(n10, 1-p11) + xlogy(n11, p11)
    )
    lr = max(0.0, float(2*(log_markov-log_ind)))
    return {
        "transition_counts": counts.tolist(),
        "lr_independence": round(lr, 8),
        "p_value": round(float(chi2.sf(lr, df=1)), 8),
        "warning": "Asymptotic first-order Markov approximation; within-year only; skipped/missing dates not individually modeled.",
    }


def _var_monitor(frame: pd.DataFrame) -> dict:
    out = {}
    years = frame["validation_year"].to_numpy(dtype=int)
    for level, tail in ((95, 0.05), (99, 0.01)):
        flags = frame[f"var_{level}_exception"].to_numpy(dtype=int)
        n, k = len(flags), int(flags.sum())
        rate = k / n
        # Likelihood ratio unconditional coverage; do not apply log(0) directly.
        ll_null = xlogy(k, tail) + xlogy(n-k, 1-tail)
        ll_alt = xlogy(k, rate) + xlogy(n-k, 1-rate)
        lr_uc = max(0.0, float(2*(ll_alt-ll_null)))
        independence = _exception_independence(flags, years)
        cc = None
        if independence["lr_independence"] is not None:
            cc = {
                "lr_conditional_coverage": round(lr_uc + independence["lr_independence"], 8),
                "p_value": round(float(chi2.sf(lr_uc + independence["lr_independence"], df=2)), 8),
            }
        out[str(level)] = {
            "observations": n,
            "exceptions": k,
            "expected_rate": tail,
            "observed_rate": round(rate, 8),
            "exact_binomial_coverage_p_value": round(float(binomtest(k, n, tail).pvalue), 8),
            "lr_unconditional_coverage": round(lr_uc, 8),
            "lr_unconditional_coverage_p_value": round(float(chi2.sf(lr_uc, 1)), 8),
            "independence": independence,
            "conditional_coverage": cc,
            "annual_exceptions": {
                str(year): int(frame.loc[frame.validation_year == year, f"var_{level}_exception"].sum())
                for year in YEARS
            },
        }
    return out


def _cases(frame: pd.DataFrame, per_type: int = 10) -> pd.DataFrame:
    rows = []
    for model, col in MODELS.items():
        for event, case in ((1, "event_low_probability"), (0, "non_event_high_probability")):
            sub = frame.loc[frame[TARGET] == event, ["date", "validation_year", TARGET, col]].copy()
            sub = sub.sort_values([col, "date"], ascending=[bool(event), True], kind="stable").head(per_type)
            for row in sub.itertuples(index=False):
                rows.append({
                    "model": model,
                    "case_type": case,
                    "date": row.date,
                    "validation_year": int(row.validation_year),
                    TARGET: int(getattr(row, TARGET)),
                    "probability": float(getattr(row, col)),
                })
    return pd.DataFrame(rows)


def validate_development(
    predictions_csv: str | Path,
    var_csv: str | Path,
    *,
    n_resamples: int = 1000,
    block_length: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Analyze existing frozen predictions and VaR backtest; NEVER refit models."""
    predicted = load_predictions(predictions_csv)
    var = load_var(var_csv, predicted)
    yearly = []
    for year in YEARS:
        sub = predicted.loc[predicted.validation_year == year]
        y = sub[TARGET].to_numpy(dtype=int)
        for model, column in MODELS.items():
            values = _metrics(y, sub[column].to_numpy(dtype=float))
            yearly.append({
                "year": year, "model": model, "observations": len(sub),
                "events": int(y.sum()), "observed_frequency": float(y.mean()),
                **{key: values[key] for key in (
                    "brier_score", "log_loss", "roc_auc", "average_precision",
                    "precision_at_threshold", "recall_at_threshold",
                )},
            })
    reliability, calibration = _reliability(predicted)
    pooled = {}
    for model, column in MODELS.items():
        pooled[model] = _metrics(
            predicted[TARGET].to_numpy(dtype=int),
            predicted[column].to_numpy(dtype=float),
        )
    report = {
        "title": "RiskLens Phase 5 | Development-only frozen-model validation",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "2020–2024 development predictions only; NOT independent holdout validation",
        "primary_candidate": "xgboost_fixed",
        "challenger": "lstm_optuna_trial86",
        "validation_years": list(YEARS),
        "observations": int(len(predicted)),
        "events": int(predicted[TARGET].sum()),
        "source_sha256": {
            "frozen_predictions": sha256(Path(predictions_csv).read_bytes()).hexdigest(),
            "historical_var": sha256(Path(var_csv).read_bytes()).hexdigest(),
        },
        "pooled_model_metrics_descriptive": pooled,
        "probability_calibration": calibration,
        "fixed_reliability_bin_edges": list(BINS),
        "paired_brier_comparison": _paired_block_bootstrap(
            predicted, n_resamples=n_resamples, block_length=block_length
        ),
        "var_exception_monitoring": _var_monitor(var),
        "warnings": [
            "Optuna LSTM trial 86 tuned on 2020–2022; training-search results are not independent validation.",
            "2023–2024 years were inspected repeatedly before this report and cannot be an independent holdout.",
            "Pooled ROC-AUC/AP across distinct yearly refits may reflect changing event base rates; examine per-year scores.",
            "The paired circular block bootstrap is descriptive, not a guarantee of inferential coverage for financial time series.",
            "Fixed-bin ECE is sample and bin dependent; it is not fitted calibration and does not prove accuracy.",
            "Historical VaR is based on trailing index returns; it is distinct from the ML downside classifier, not generated by it.",
            "Binomial and likelihood ratio coverage tests rely on independence/asymptotic approximations; exception clustering can violate them.",
            "No post-hoc threshold selection, model fitting, calibration, or 2025+ holdout access occurs in this command.",
        ],
    }
    return pd.DataFrame(yearly), reliability, _cases(predicted), report


def _markdown(report: dict) -> str:
    lines = [
        "# RiskLens Phase 5 — Development Risk-Model Validation",
        "",
        "**NOT independent final validation. 2025+ holdout sealed.**",
        "",
        f"Aligned observations: {report['observations']}; events: {report['events']}.",
        "",
        "| Candidate | Brier ↓ | Log loss ↓ | Pooled ROC-AUC* | Pooled AP* |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in report["pooled_model_metrics_descriptive"].items():
        lines.append(
            f"| {name} | {values['brier_score']:.6f} | {values['log_loss']:.6f} | "
            f"{values['roc_auc']} | {values['average_precision']} |"
        )
    paired = report["paired_brier_comparison"]
    lines += [
        "",
        "*Pooled discrimination is descriptive across yearly refits; use yearly CSV for stability.*",
        "",
        "## Paired Brier difference (LSTM minus XGBoost)",
        f"Observed: {paired['observed_delta']:.6f}; descriptive 95% block-bootstrap interval: "
        f"{paired['descriptive_block_bootstrap_95_percentile_interval']}.",
        "",
        "## Historical VaR exceptions",
        "| Level | Breaches | Rate | Expected | Exact coverage p | Independence p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for level, value in report["var_exception_monitoring"].items():
        lines.append(
            f"| {level}% | {value['exceptions']} | {value['observed_rate']:.4f} | "
            f"{value['expected_rate']:.4f} | {value['exact_binomial_coverage_p_value']} | "
            f"{value['independence']['p_value']} |"
        )
    lines += [
        "",
        "## Limits",
        *["- " + warning for warning in report["warnings"]],
        "",
        "Raw details and data hashes: risk_validation_summary.json; "
        "annual, reliability and error-case CSV files are included.",
        "",
    ]
    return "\n".join(lines)


def save_risk_validation(
    predictions_csv: str | Path,
    var_csv: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    n_resamples: int = 1000,
    block_length: int = 10,
) -> dict:
    out = Path(output_dir)
    targets = [out / name for name in OUTPUT_FILES]
    if not overwrite and any(p.exists() for p in targets):
        raise RiskValidationError("Phase 5 outputs already exist; pass --overwrite")
    yearly, reliability, cases, report = validate_development(
        predictions_csv, var_csv, n_resamples=n_resamples, block_length=block_length
    )
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as td:
        scratch = Path(td)
        (scratch / OUTPUT_FILES[0]).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        for df, file in (
            (yearly, OUTPUT_FILES[1]),
            (reliability, OUTPUT_FILES[2]),
            (cases, OUTPUT_FILES[3]),
        ):
            export = df.copy()
            if "date" in export:
                export["date"] = export["date"].dt.strftime("%Y-%m-%d")
            export.to_csv(scratch / file, index=False, float_format="%.12g")
        (scratch / OUTPUT_FILES[4]).write_text(_markdown(report), encoding="utf-8")
        for target in targets:
            (scratch / target.name).replace(target)
    return report
