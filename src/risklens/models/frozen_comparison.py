"""Frozen, development-only comparison: fixed 11-feature XGBoost vs Optuna LSTM #86.

Runs ZERO Optuna trials. The LSTM is refitted with trial #86's exact architecture
and hyperparameters on each historical training fold, using the original
optuna_research._lstm_one implementation. Never evaluates 2025+ outcomes.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import platform

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import xgboost
import sklearn

from risklens.evaluation.features import MULTIFACTOR_FEATURE_COLUMNS
from risklens.evaluation.metrics import _metrics
from risklens.evaluation.splits import FINAL_HOLDOUT_START
from risklens.features.dataset import TARGET_COLUMN
from risklens.models.optuna_research import (
    _lstm_one,
    make_folds,
    sequence_folds,
)
from risklens.models.xgboost_walk_forward import FIXED_XGB_PARAMS

TUNING_YEARS = (2020, 2021, 2022)
DIAGNOSTIC_YEARS = (2023, 2024)
YEARS = TUNING_YEARS + DIAGNOSTIC_YEARS
TRIAL_NUMBER = 86
TRIAL_BRIER_RECORDED = 0.120224828144602
TRIAL86 = {
    "lookback": 45,
    "hidden": 24,
    "layers": 2,
    "batch": 128,
    "epochs": 24,
    "dropout": 0.3413771488170494,
    "lr": 0.0027467789382135,
    "weight_decay": 0.0006684358163984,
}
FEATURES = MULTIFACTOR_FEATURE_COLUMNS
PREDICTIONS_FILE = "frozen_model_predictions.csv"
REPORT_FILE = "frozen_model_report.json"
MARKDOWN_FILE = "frozen_model_report.md"


class FrozenComparisonError(ValueError):
    """Invalid study provenance, holdout contamination, or model alignment."""


def verify_trial86(study_db: str | Path) -> dict:
    """Verify actual completed Optuna record #86; never create or optimize a study."""
    path = Path(study_db)
    if not path.is_file():
        raise FrozenComparisonError(
            f"Optuna SQLite database does not exist: {path}. "
            "Specify --study-db pointing to your saved optuna_studies.sqlite3."
        )
    try:
        import optuna
        study = optuna.load_study(
            study_name="risklens_lstm",
            storage="sqlite:///" + str(path.resolve()),
        )
        trials = [t for t in study.get_trials(deepcopy=False) if t.number == TRIAL_NUMBER]
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise FrozenComparisonError(f"Cannot inspect saved LSTM Optuna study: {exc}") from exc
    if len(trials) != 1 or trials[0].state != optuna.trial.TrialState.COMPLETE:
        raise FrozenComparisonError("LSTM trial #86 is absent or not COMPLETE")
    trial = trials[0]
    if set(trial.params) != set(TRIAL86):
        raise FrozenComparisonError("Trial #86 hyperparameter names do not match frozen specification")
    for key, expected in TRIAL86.items():
        actual = trial.params[key]
        if isinstance(expected, int):
            valid = type(actual) is int and actual == expected
        else:
            valid = np.isclose(actual, expected, rtol=0, atol=1e-12)
        if not valid:
            raise FrozenComparisonError(
                f"Trial #86 mismatch for {key}: study={actual}, frozen={expected}"
            )
    if trial.value is None or not np.isclose(
        trial.value, TRIAL_BRIER_RECORDED, rtol=0, atol=1e-8
    ):
        raise FrozenComparisonError("Trial #86 recorded objective differs from prior audit")
    return {
        "study_name": study.study_name,
        "database_sha256": sha256(path.read_bytes()).hexdigest(),
        "trial_number": int(trial.number),
        "trial_objective_brier": float(trial.value),
        "frozen_params": dict(TRIAL86),
    }


def _validate_probabilities(y: np.ndarray, p: np.ndarray, label: str) -> None:
    if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise FrozenComparisonError(f"{label}: invalid probability vector")
    if not np.isin(y, (0, 1)).all():
        raise FrozenComparisonError(f"{label}: invalid target labels")


def _model_scores(frame: pd.DataFrame) -> dict:
    y = frame["actual_downside_event"].to_numpy(dtype=int)
    if not len(y):
        raise FrozenComparisonError("Cannot score an empty period")
    return {
        name: _metrics(y, frame[column].to_numpy(dtype=float))
        for name, column in (
            ("xgboost_11_fixed", "xgboost_11_fixed_probability"),
            ("lstm_optuna_86", "lstm_optuna_86_probability"),
        )
    }


def evaluate_frozen(
    nifty_csv: str | Path,
    external_csv: str | Path,
    *,
    device: str = "auto",
    study_db: str | Path = "data/processed/optuna_search/optuna_studies.sqlite3",
) -> tuple[pd.DataFrame, dict]:
    """Retrain frozen configurations on 2020–2024 folds, with identical evaluation rows."""
    if device not in ("auto", "cpu", "mps"):
        raise FrozenComparisonError("Device must be auto, cpu, or mps")
    if max(YEARS) >= pd.Timestamp(FINAL_HOLDOUT_START).year:
        raise FrozenComparisonError("Refusing any final-holdout validation year")
    for path in (nifty_csv, external_csv):
        if not Path(path).is_file():
            raise FrozenComparisonError(f"Input CSV not found: {path}")
    provenance = verify_trial86(study_db)
    # Reuse precisely the feature construction and purged folds used in Optuna research.
    history, folds = make_folds(nifty_csv, external_csv, years=YEARS)
    sequences = sequence_folds(history, folds, TRIAL86["lookback"])
    if len(folds) != len(YEARS) or len(sequences) != len(YEARS):
        raise FrozenComparisonError("Incomplete chronological folds")

    predictions = []
    yearly = []
    for (year, train, valid), (seq_year, train_x, train_y, valid_x, valid_y) in zip(
        folds, sequences
    ):
        if year != seq_year or year not in YEARS:
            raise FrozenComparisonError("Mismatched XGBoost/LSTM validation year")
        valid_dates = pd.DatetimeIndex(valid["date"])
        if (
            valid_dates.empty
            or not valid_dates.is_monotonic_increasing
            or not valid_dates.is_unique
            or (valid_dates >= pd.Timestamp(FINAL_HOLDOUT_START)).any()
            or valid_dates.min().year != year
            or valid_dates.max().year != year
            or train["date"].max() >= valid_dates.min()
            or train["date"].max() >= pd.Timestamp(f"{year}-01-01")
        ):
            raise FrozenComparisonError(f"Fold {year}: chronological boundary violation")
        y_train = train[TARGET_COLUMN].to_numpy(dtype=int)
        y_valid = valid[TARGET_COLUMN].to_numpy(dtype=int)
        if not np.array_equal(y_valid, valid_y) or len(valid_x) != len(valid):
            raise FrozenComparisonError(f"Fold {year}: LSTM dates/labels misaligned")
        if len(np.unique(y_train)) != 2 or not np.array_equal(
            np.unique(train_y), np.array([0, 1])
        ):
            raise FrozenComparisonError(f"Fold {year}: training labels lack both classes")
        x_train = train.loc[:, list(FEATURES)].to_numpy(dtype=float)
        x_valid = valid.loc[:, list(FEATURES)].to_numpy(dtype=float)
        if not np.isfinite(x_train).all() or not np.isfinite(x_valid).all():
            raise FrozenComparisonError(f"Fold {year}: invalid tabular feature values")

        fixed = XGBClassifier(**FIXED_XGB_PARAMS)
        fixed.fit(x_train, y_train)
        p_xgb = fixed.predict_proba(x_valid)[:, 1].astype(float)
        # EXACT trial-86 architecture and training routine, not the legacy sequence_lstm defaults.
        p_lstm = np.asarray(
            _lstm_one(
                train_x, train_y, valid_x, dict(TRIAL86),
                seed=2026 + year, device=device,
            ),
            dtype=float,
        )
        _validate_probabilities(y_valid, p_xgb, "XGBoost")
        _validate_probabilities(y_valid, p_lstm, "LSTM #86")
        frame = pd.DataFrame({
            "validation_year": year,
            "date": valid_dates,
            "actual_downside_event": y_valid,
            "xgboost_11_fixed_probability": p_xgb,
            "lstm_optuna_86_probability": p_lstm,
        })
        predictions.append(frame)
        yearly.append({
            "year": int(year),
            "period": "tuning" if year in TUNING_YEARS else "repeated_diagnostic",
            "observations": int(len(frame)),
            "events": int(y_valid.sum()),
            "last_training_date": str(pd.Timestamp(train["date"].max()).date()),
            "first_validation_date": str(valid_dates.min().date()),
            "scores": _model_scores(frame),
        })
    all_predictions = pd.concat(predictions, ignore_index=True)
    if (
        all_predictions.date.duplicated().any()
        or not all_predictions.date.is_monotonic_increasing
        or (all_predictions.date >= pd.Timestamp(FINAL_HOLDOUT_START)).any()
    ):
        raise FrozenComparisonError("Combined predictions contain duplicate/unsorted/holdout dates")
    groups = {
        "tuning_2020_2022": all_predictions[
            all_predictions.validation_year.isin(TUNING_YEARS)
        ],
        "repeated_diagnostic_2023_2024": all_predictions[
            all_predictions.validation_year.isin(DIAGNOSTIC_YEARS)
        ],
        "all_development_2020_2024_descriptive_only": all_predictions,
    }
    report = {
        "title": "RiskLens Phase 4 | Frozen Champion-Challenger Development Comparison",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "primary_candidate": "xgboost_11_fixed",
        "challenger": "lstm_optuna_86",
        "status": "DEVELOPMENT-ONLY: 2025+ final holdout NEVER evaluated",
        "selection_note": "Previously inspected 2023-2024 diagnostics are not an independent test. "
                          "Do not reinterpret this rerun as independent model selection evidence.",
        "features": list(FEATURES),
        "training_specification": {
            "xgboost_fixed_params": dict(FIXED_XGB_PARAMS),
            "lstm_trial86_params": dict(TRIAL86),
            "lstm_seed_by_year": "2026 + validation_year",
            "lstm_training_impl": "risklens.models.optuna_research._lstm_one",
            "training_data": "Expanding prior-year development history, purged cross-year labels",
            "evaluation_data": "Same prediction dates and outcomes for both models",
            "device_requested": device,
        },
        "optuna_trial86_provenance": provenance,
        "period_scores": {name: {
            "rows": int(len(frame)),
            "events": int(frame.actual_downside_event.sum()),
            "metrics": _model_scores(frame),
        } for name, frame in groups.items()},
        "yearly_scores": yearly,
        "source_sha256": {
            "nifty": sha256(Path(nifty_csv).read_bytes()).hexdigest(),
            "external": sha256(Path(external_csv).read_bytes()).hexdigest(),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "limitations": [
            "This rerun is not an independent holdout and is not a claim of future predictive skill.",
            "Optuna trial #86 was selected on 2020-2022; those years are training-search evidence.",
            "2023-2024 model diagnostics were consulted repeatedly during earlier research.",
            "Model scores can vary with device/library version, particularly PyTorch MPS.",
            "The final holdout beginning 2025-01-01 remains sealed.",
        ],
    }
    return all_predictions, report


def _markdown(report: dict) -> str:
    sections = [
        "# Frozen XGBoost vs Optuna LSTM Trial #86",
        "",
        "**Research only:** 2025+ final holdout sealed. All listed scores are development results.",
        "",
        "| Period | Model | N | Events | Brier ↓ | Log loss ↓ | ROC-AUC ↑ | AP ↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for period, bundle in report["period_scores"].items():
        for model, scores in bundle["metrics"].items():
            sections.append(
                f"| {period} | {model} | {bundle['rows']} | {bundle['events']} | "
                f"{scores['brier_score']:.6f} | {scores['log_loss']:.6f} | "
                f"{scores['roc_auc'] if scores['roc_auc'] is not None else 'undefined'} | "
                f"{scores['average_precision'] if scores['average_precision'] is not None else 'undefined'} |"
            )
    sections += [
        "",
        "**Interpretation:** The 2020–2022 period was used in Optuna tuning; "
        "2023–2024 was repeatedly examined during development. Neither is an untouched test.",
        "This report is a reproducibility check, not a new optimization or promotion decision.",
        "",
        f"Frozen LSTM study trial: #{report['optuna_trial86_provenance']['trial_number']}.",
        "See frozen_model_report.json for source hashes, full parameters, and per-year metrics.",
        "",
    ]
    return "\n".join(sections)


def save_frozen_comparison(
    nifty_csv: str | Path,
    external_csv: str | Path,
    output_dir: str | Path,
    *,
    study_db: str | Path = "data/processed/optuna_search/optuna_studies.sqlite3",
    device: str = "auto",
    overwrite: bool = False,
) -> dict:
    out = Path(output_dir)
    paths = [out / PREDICTIONS_FILE, out / REPORT_FILE, out / MARKDOWN_FILE]
    if not overwrite and any(path.exists() for path in paths):
        raise FrozenComparisonError("Phase 4 output already exists; use --overwrite explicitly")
    frame, report = evaluate_frozen(
        nifty_csv, external_csv, device=device, study_db=study_db
    )
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as temporary:
        tmp = Path(temporary)
        to_save = frame.copy()
        to_save["date"] = to_save.date.dt.strftime("%Y-%m-%d")
        to_save.to_csv(tmp / PREDICTIONS_FILE, index=False, float_format="%.12g")
        (tmp / REPORT_FILE).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        (tmp / MARKDOWN_FILE).write_text(_markdown(report), encoding="utf-8")
        for destination in paths:
            (tmp / destination.name).replace(destination)
    return report
