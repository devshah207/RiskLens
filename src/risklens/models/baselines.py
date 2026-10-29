"""Fixed, development-only baseline and Logistic Regression comparison.

Design constraints:
- Inputs are the TWO dated, labeled feature tables built by Step 3.1.
- Fit any preprocessing and coefficients on training rows ONLY.
- Validation rows are scored, never used to fit or select hyperparameters.
- 2025+ final holdout is neither accepted nor evaluated.
- No optimization on validation, backtesting, VaR or claims of tradability.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from risklens.evaluation.metrics import _metrics
from risklens.features.dataset import (
    FEATURE_COLUMNS,
    EXPANDED_FEATURE_COLUMNS,
    TARGET_COLUMN,
)

EXPECTED_COLUMNS = ("date", *FEATURE_COLUMNS, TARGET_COLUMN)
PREDICTION_OUTPUT = "validation_model_predictions.csv"
REPORT_OUTPUT = "baseline_logistic_validation_report.json"
FIXED_THRESHOLD = 0.50


class ModelInputError(ValueError):
    """Model input violates the development-only dataset contract."""


def _read_split(
    path: str | Path,
    name: str,
    *,
    start: str | None,
    end: str,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Validate an exported feature table without silently accepting new columns."""
    path = Path(path)
    if not path.is_file():
        raise ModelInputError(f"{name} CSV not found: {path}")
    try:
        frame = pd.read_csv(path, dtype="string", encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ModelInputError(f"Cannot read {name} CSV '{path}': {exc}") from exc
    expected = ("date", *feature_columns, TARGET_COLUMN)
    if list(frame.columns) != list(expected):
        raise ModelInputError(
            f"{name} columns must be exactly {list(expected)} in that order; "
            "unexpected columns could contain leaked future information"
        )
    if frame.empty:
        raise ModelInputError(f"{name} CSV has no observations")

    dates_text = frame["date"].astype("string")
    dates = pd.to_datetime(dates_text, format="%Y-%m-%d", errors="coerce")
    if dates.isna().any() or not dates.dt.strftime("%Y-%m-%d").eq(dates_text).all():
        raise ModelInputError(f"{name} contains invalid or non-ISO prediction dates")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ModelInputError(
            f"{name} dates must be unique and chronologically increasing"
        )
    if start is not None and (dates < pd.Timestamp(start)).any():
        raise ModelInputError(f"{name} includes prediction dates before {start}")
    if (dates >= pd.Timestamp(end)).any():
        raise ModelInputError(
            f"{name} includes prediction dates on or after {end}; holdout stays sealed"
        )

    for feature in feature_columns:
        try:
            numeric = pd.to_numeric(frame[feature], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ModelInputError(f"{name} has nonnumeric values in {feature}") from exc
        if not np.isfinite(numeric.to_numpy()).all():
            raise ModelInputError(
                f"{name} contains nonfinite feature values in {feature}"
            )
        frame[feature] = numeric
    raw_target = frame[TARGET_COLUMN].astype("string")
    if not raw_target.isin(["0", "1"]).all():
        raise ModelInputError(f"{name} target must contain only integer 0 and 1")
    frame[TARGET_COLUMN] = raw_target.astype(int)
    frame["date"] = dates
    return frame


def compare_models(
    train_csv: str | Path,
    validation_csv: str | Path,
    *,
    validation_start: str = "2023-01-01",
    holdout_start: str = "2025-01-01",
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, dict]:
    """Fit on training only, return *validation* probabilities and descriptive scores.

    Hyperparameters are fixed in advance. There is no validation-driven tuning.
    """
    if feature_columns not in (FEATURE_COLUMNS, EXPANDED_FEATURE_COLUMNS):
        raise ModelInputError(
            "Only the predefined 3-feature or 6-feature configuration is allowed"
        )
    try:
        vstart = datetime.strptime(validation_start, "%Y-%m-%d")
        hstart = datetime.strptime(holdout_start, "%Y-%m-%d")
    except ValueError as exc:
        raise ModelInputError("Cutoffs must use valid YYYY-MM-DD dates") from exc
    if (
        vstart.strftime("%Y-%m-%d") != validation_start
        or hstart.strftime("%Y-%m-%d") != holdout_start
        or vstart >= hstart
    ):
        raise ModelInputError(
            "Require canonical YYYY-MM-DD cutoffs, validation before holdout"
        )

    train = _read_split(
        train_csv,
        "training",
        start=None,
        end=validation_start,
        feature_columns=feature_columns,
    )
    valid = _read_split(
        validation_csv,
        "validation",
        start=validation_start,
        end=holdout_start,
        feature_columns=feature_columns,
    )
    if train["date"].max() >= valid["date"].min():
        raise ModelInputError("Training and validation date ranges overlap")

    X_train = train.loc[:, list(feature_columns)].to_numpy(dtype=float)
    y_train = train[TARGET_COLUMN].to_numpy(dtype=int)
    X_valid = valid.loc[:, list(feature_columns)].to_numpy(dtype=float)
    y_valid = valid[TARGET_COLUMN].to_numpy(dtype=int)
    if len(np.unique(y_train)) != 2:
        raise ModelInputError(
            "Training must contain both non-events (0) and downside events (1)"
        )

    training_rate = float(y_train.mean())
    p_baseline = np.full(len(valid), training_rate, dtype=float)

    # StandardScaler.fit is called ONLY on X_train by Pipeline.fit.
    model = Pipeline(
        [
            ("standardize", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight=None,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    p_logistic = model.predict_proba(X_valid)[:, 1]

    predictions = pd.DataFrame(
        {
            "date": valid["date"],
            "actual_downside_event": y_valid,
            "baseline_probability": p_baseline,
            "logistic_probability": p_logistic,
        }
    )
    scaler = model.named_steps["standardize"]
    logistic = model.named_steps["classifier"]
    coef = logistic.coef_[0]
    scores = {
        "constant_baseline": _metrics(y_valid, p_baseline),
        "logistic_regression": _metrics(y_valid, p_logistic),
    }
    report = {
        "scope": f"DEVELOPMENT ONLY: trained before {validation_start}, validated before {holdout_start}; final holdout outcomes unexamined",
        "training_file": Path(train_csv).name,
        "validation_file": Path(validation_csv).name,
        "training_sha256": sha256(Path(train_csv).read_bytes()).hexdigest(),
        "validation_sha256": sha256(Path(validation_csv).read_bytes()).hexdigest(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prediction_timestamp": "after prediction-date close; label is NEXT trading-session downside event",
        "validation_start": validation_start,
        "holdout_start": holdout_start,
        "features": list(feature_columns),
        "target": TARGET_COLUMN,
        "training": {
            "rows": int(len(train)),
            "events": int(y_train.sum()),
            "event_rate": round(training_rate, 8),
            "first_date": train["date"].iloc[0].strftime("%Y-%m-%d"),
            "last_date": train["date"].iloc[-1].strftime("%Y-%m-%d"),
        },
        "validation": {
            "rows": int(len(valid)),
            "events": int(y_valid.sum()),
            "event_rate": round(float(y_valid.mean()), 8),
            "first_date": valid["date"].iloc[0].strftime("%Y-%m-%d"),
            "last_date": valid["date"].iloc[-1].strftime("%Y-%m-%d"),
        },
        "models": {
            "constant_baseline": "Fixed training event fraction, predicted for every validation date",
            "logistic_regression": {
                "hyperparameters": {
                    "C": 1.0,
                    "penalty": "l2 (solver default)",
                    "solver": "lbfgs",
                    "max_iter": 2000,
                    "class_weight": None,
                },
                "scaler_fitted_on": "training only",
                "scaler_mean": {
                    k: float(v) for k, v in zip(feature_columns, scaler.mean_)
                },
                "scaler_scale": {
                    k: float(v) for k, v in zip(feature_columns, scaler.scale_)
                },
                "coefficients_per_standardized_feature": {
                    k: float(v) for k, v in zip(feature_columns, coef)
                },
                "intercept": float(logistic.intercept_[0]),
                "iterations": int(logistic.n_iter_[0]),
            },
        },
        "validation_scores": scores,
        "warnings": [
            "Validation results are exploratory; no hyperparameters or thresholds optimized in this step.",
            "Final holdout prices, outcomes, and features are not used.",
            "Training class rate is not a probability forecast of unconditional future risk.",
            "A 0.50 classification threshold can yield no alerts for rare events; rank/probability metrics matter.",
            "Model coefficients describe association, not causal financial mechanisms.",
            "This evaluation is one chronological split, not a walk-forward model validation or VaR backtest.",
        ],
    }
    return predictions, report


def save_comparison(
    train_csv: str | Path,
    validation_csv: str | Path,
    output_dir: str | Path,
    *,
    validation_start: str = "2023-01-01",
    holdout_start: str = "2025-01-01",
    overwrite: bool = False,
) -> dict:
    """Write validation predictions and model audit report after checks succeed."""
    predictions, report = compare_models(
        train_csv,
        validation_csv,
        validation_start=validation_start,
        holdout_start=holdout_start,
    )
    out = Path(output_dir)
    targets = [out / PREDICTION_OUTPUT, out / REPORT_OUTPUT]
    if not overwrite and any(path.exists() for path in targets):
        raise ModelInputError(
            "Output already exists; choose another folder or use --overwrite"
        )
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as tmp:
        stage = Path(tmp)
        predictions.to_csv(stage / PREDICTION_OUTPUT, index=False, float_format="%.12g")
        (stage / REPORT_OUTPUT).write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        for target in targets:
            (stage / target.name).replace(target)
    return report
