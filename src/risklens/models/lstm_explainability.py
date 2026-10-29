"""Phase 6: descriptive Optuna LSTM trial #86 sequence-feature sensitivity.

Refits EXACT frozen trial #86 once for the previously reviewed 2024
development year. Permutes entire feature trajectories across observations
with a fixed seed; NO extra trials, no 2025+ data, no feature selection.
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
from risklens.models.frozen_comparison import TRIAL86, verify_trial86
from risklens.models.optuna_research import _lstm_one, make_folds, sequence_folds

FEATURES = MULTIFACTOR_FEATURE_COLUMNS
DIAGNOSTIC_YEAR = 2024
PERMUTATION_REPEATS = 3
OUTPUTS = (
    "lstm_sequence_sensitivity.csv",
    "lstm_sequence_sensitivity_report.json",
    "lstm_sequence_sensitivity.md",
)


class LSTMSensitivityError(ValueError):
    """Unsafe sequence sensitivity request or incompatible model output."""


def analyze_lstm_sensitivity(
    nifty_csv: str | Path,
    external_csv: str | Path,
    study_db: str | Path,
    *,
    device: str = "auto",
) -> tuple[pd.DataFrame, dict]:
    if device not in ("auto", "cpu", "mps"):
        raise LSTMSensitivityError("device must be auto, cpu or mps")
    trial = verify_trial86(study_db)
    history, folds = make_folds(
        nifty_csv, external_csv, years=(DIAGNOSTIC_YEAR,)
    )
    sequences = sequence_folds(history, folds, TRIAL86["lookback"])
    if len(folds) != 1 or len(sequences) != 1:
        raise LSTMSensitivityError("Expected one audited 2024 development fold")
    year, train_x, train_y, valid_x, valid_y = sequences[0]
    if year != DIAGNOSTIC_YEAR or len(valid_x) < 20:
        raise LSTMSensitivityError("Wrong validation year or too few sequences")
    if len(folds[0][2]) != len(valid_x):
        raise LSTMSensitivityError("Incomplete, unequal 2024 prediction sequences")
    baseline, perturbed = _lstm_one(
        train_x, train_y, valid_x, dict(TRIAL86),
        seed=2026 + DIAGNOSTIC_YEAR,
        device=device,
        permutation_repeats=PERMUTATION_REPEATS,
    )
    baseline = np.asarray(baseline, dtype=float)
    perturbed = np.asarray(perturbed, dtype=float)
    if perturbed.shape != (PERMUTATION_REPEATS, len(FEATURES), len(valid_y)):
        raise LSTMSensitivityError("Unexpected perturbation prediction shape")
    if (
        not np.isfinite(baseline).all()
        or not np.isfinite(perturbed).all()
        or ((baseline < 0) | (baseline > 1)).any()
        or ((perturbed < 0) | (perturbed > 1)).any()
    ):
        raise LSTMSensitivityError("Nonfinite or invalid sensitivity probabilities")
    y = np.asarray(valid_y, dtype=int)
    if not np.isin(y, [0, 1]).all() or len(y) != len(baseline):
        raise LSTMSensitivityError("Invalid 2024 labels")
    reference_brier = float(np.mean((y-baseline)**2))
    perturbed_brier = np.mean((perturbed-y[None, None, :])**2, axis=2)
    change = perturbed_brier-reference_brier
    results = pd.DataFrame({
        "feature": list(FEATURES),
        "mean_permuted_brier": perturbed_brier.mean(axis=0),
        "mean_brier_increase": change.mean(axis=0),
        "std_brier_change_over_3_shuffles": change.std(axis=0, ddof=1),
        "permutation_repeats": PERMUTATION_REPEATS,
    }).sort_values("mean_brier_increase", ascending=False).reset_index(drop=True)
    report = {
        "title": "RiskLens Phase 6 | Optuna trial #86 LSTM temporal feature sensitivity",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "Previously consulted 2024 DEVELOPMENT validation only; NO final holdout data",
        "trial_provenance": trial,
        "frozen_lstm_params": dict(TRIAL86),
        "validation_year": DIAGNOSTIC_YEAR,
        "training_sequences": len(train_x),
        "validation_sequences": len(valid_x),
        "validation_events": int(y.sum()),
        "baseline_brier": reference_brier,
        "permutation_repeats": PERMUTATION_REPEATS,
        "permutation_seed_rule": "2026 + 2024 + 10000",
        "method": (
            "Freeze model after ONE fixed-parameter training run; for each feature, "
            "shuffle full 45-session trajectories across 2024 validation sequences. "
            "Evaluate Brier deterioration without retraining."
        ),
        "source_sha256": {
            "nifty": sha256(Path(nifty_csv).read_bytes()).hexdigest(),
            "development_external": sha256(Path(external_csv).read_bytes()).hexdigest(),
        },
        "cautions": [
            "Permuting correlated sequence features disrupts the joint data distribution and is NOT a causal intervention.",
            "Feature sensitivity is descriptive for 2024 only, a previously inspected development period.",
            "Negative Brier change means shuffling improved Brier in these samples, not that a feature is globally harmful.",
            "Trial #86 is NOT retuned; ranking cannot be used for feature selection after holdout.",
            "Native TreeSHAP and sequence permutation sensitivity measure different concepts and have incompatible units.",
            "Apple MPS may yield small floating-point differences compared with CPU.",
        ],
    }
    return results, report


def _markdown(result: pd.DataFrame, report: dict) -> str:
    lines = [
        "# Optuna LSTM Trial #86 — Temporal Feature Sensitivity",
        "",
        "**Development diagnostic (2024 only), not a new independent test.**",
        f"Validation observations: {report['validation_sequences']}; "
        f"baseline Brier: {report['baseline_brier']:.6f}.",
        "",
        "| Feature | Brier after whole-window shuffle | Change from original Brier |",
        "|---|---:|---:|",
    ]
    for row in result.itertuples(index=False):
        lines.append(
            f"| {row.feature} | {row.mean_permuted_brier:.6f} | "
            f"{row.mean_brier_increase:+.6f} |"
        )
    lines += ["", "## Interpretation limitations",
              *["- " + msg for msg in report["cautions"]],
              ""]
    return "\n".join(lines)


def save_lstm_sensitivity(
    nifty_csv: str | Path,
    external_csv: str | Path,
    study_db: str | Path,
    output_dir: str | Path,
    *,
    device: str = "auto",
    overwrite: bool = False,
) -> dict:
    out = Path(output_dir)
    targets = [out / name for name in OUTPUTS]
    if not overwrite and any(target.exists() for target in targets):
        raise LSTMSensitivityError("LSTM explanation exists; pass --overwrite")
    table, report = analyze_lstm_sensitivity(
        nifty_csv, external_csv, study_db, device=device
    )
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as td:
        tmp = Path(td)
        table.to_csv(tmp / OUTPUTS[0], index=False)
        (tmp / OUTPUTS[1]).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        (tmp / OUTPUTS[2]).write_text(_markdown(table, report), encoding="utf-8")
        for target in targets:
            (tmp / target.name).replace(target)
    return report
