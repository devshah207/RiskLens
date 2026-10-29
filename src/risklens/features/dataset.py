"""Build a time-aligned, development-only NIFTY 50 classification dataset.

Prediction is made after session t closes. Features use dates <= t; the
historical label uses the next observed session's close. Training labels are
never allowed to cross into validation, nor validation labels into holdout.
No model is fitted here; the final holdout is not converted into features or labels.
"""

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from risklens.data.validation import OHLCValidationError, load_ohlc_csv


FEATURE_COLUMNS = ("return_1d", "return_5d", "volatility_20d")
TARGET_COLUMN = "downside_next_session"
OUTPUT_NAMES = ("train_features.csv", "validation_features.csv", "feature_dataset_report.json")


class FeatureDatasetError(ValueError):
    """Raised if safe features, label boundaries, or outputs cannot be produced."""


def _strict_date(raw: str, name: str) -> pd.Timestamp:
    try:
        result = datetime.strptime(raw, "%Y-%m-%d")
        if result.strftime("%Y-%m-%d") != raw:
            raise ValueError("non-canonical date")
    except (TypeError, ValueError) as exc:
        raise FeatureDatasetError(f"{name} must be a valid YYYY-MM-DD date") from exc
    return pd.Timestamp(result)


def _split_summary(data: pd.DataFrame) -> dict:
    """Summarize only dates and labels in an allowed development partition."""
    events = int(data[TARGET_COLUMN].sum())
    return {
        "rows": int(len(data)),
        "first_prediction_date": data["date"].iloc[0].date().isoformat(),
        "last_prediction_date": data["date"].iloc[-1].date().isoformat(),
        "downside_events": events,
        "non_events": int(len(data) - events),
        "event_frequency": round(events / len(data), 8),
    }


def build_development_features(
    source_csv: str | Path,
    *,
    validation_start: str = "2023-01-01",
    holdout_start: str = "2025-01-01",
    downside_threshold: float = 0.01,
    sessions_per_year: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return chronological train and validation tables plus an audit report.

    Output columns: prediction date; 1-session/5-session trailing simple returns;
    trailing 20-return annualized sample volatility; historical binary target.
    A date t is labeled only when the following observed trading session is
    within the SAME training/validation period. Warm-up feature rows are removed.
    """
    vstart = _strict_date(validation_start, "validation_start")
    hstart = _strict_date(holdout_start, "holdout_start")
    if vstart >= hstart:
        raise FeatureDatasetError("validation_start must precede holdout_start")
    if not isinstance(sessions_per_year, int) or isinstance(sessions_per_year, bool) or sessions_per_year <= 0:
        raise FeatureDatasetError("sessions_per_year must be a positive integer")
    if not isinstance(downside_threshold, (int, float)) or not np.isfinite(downside_threshold) or not (0 < downside_threshold < 1):
        raise FeatureDatasetError("downside_threshold must be strictly between 0 and 1")

    try:
        prices = load_ohlc_csv(source_csv)
    except OHLCValidationError as exc:
        raise FeatureDatasetError(str(exc)) from exc

    # Seal the final holdout BEFORE calculating any returns, features or labels.
    dev = prices.loc[prices["date"] < hstart, ["date", "close"]].copy().reset_index(drop=True)
    if len(dev) < 22:  # 21 closes for 20 returns, plus one later close for a label.
        raise FeatureDatasetError("Need at least 22 development price rows for 20-return volatility and a next-session label")

    # All three features are available AFTER session t closes; no negative shift.
    dev["return_1d"] = dev["close"].pct_change(fill_method=None)
    dev["return_5d"] = dev["close"].pct_change(periods=5, fill_method=None)
    dev["volatility_20d"] = (
        dev["return_1d"].rolling(window=20, min_periods=20).std(ddof=1)
        * np.sqrt(sessions_per_year)
    )

    # Shifting into the future is permitted ONLY for constructing the label.
    dev["next_session_date"] = dev["date"].shift(-1)
    next_close = dev["close"].shift(-1)
    has_label = next_close.notna()
    target = pd.Series(pd.NA, index=dev.index, dtype="Int8")
    next_return = (next_close.loc[has_label] / dev.loc[has_label, "close"]) - 1.0
    target.loc[has_label] = (next_return <= -downside_threshold).astype("int8")
    dev[TARGET_COLUMN] = target

    ready_features = dev[list(FEATURE_COLUMNS)].notna().all(axis=1)
    # Boundary purge: a prediction from 2022 cannot learn a label observed in 2023.
    training = (dev["date"] < vstart) & (dev["next_session_date"] < vstart)
    validation = (dev["date"] >= vstart) & (dev["next_session_date"] < hstart)
    columns = ["date", *FEATURE_COLUMNS, TARGET_COLUMN]
    train = dev.loc[training & ready_features, columns].copy().reset_index(drop=True)
    valid = dev.loc[validation & ready_features, columns].copy().reset_index(drop=True)

    if train.empty or valid.empty:
        raise FeatureDatasetError(
            "Both train and validation need eligible labeled sessions after 20-return feature warm-up; "
            "check date cutoffs and available history"
        )
    train[TARGET_COLUMN] = train[TARGET_COLUMN].astype("int64")
    valid[TARGET_COLUMN] = valid[TARGET_COLUMN].astype("int64")

    report = {
        "scope": "DEVELOPMENT ONLY; 2025+ holdout outcomes and features not calculated",
        "source_filename": Path(source_csv).name,
        "source_sha256": sha256(Path(source_csv).read_bytes()).hexdigest(),
        "prediction_time": "after the prediction-date trading session closes",
        "validation_start": vstart.date().isoformat(),
        "holdout_start": hstart.date().isoformat(),
        "target_column": TARGET_COLUMN,
        "target_definition": f"1 when next observed session simple return <= -{downside_threshold:.4f}; else 0",
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_definitions": {
            "return_1d": "close[t]/close[t-1] - 1",
            "return_5d": "close[t]/close[t-5] - 1",
            "volatility_20d": f"sample standard deviation of returns[t-19:t] times sqrt({sessions_per_year})",
        },
        "first_20_price_rows_have_incomplete_feature_history": True,
        "training_boundary_labels_purged": int(((dev["date"] < vstart) & (dev["next_session_date"] >= vstart)).sum()),
        "last_development_price_date_has_no_label": dev["date"].iloc[-1].date().isoformat(),
        "train": _split_summary(train),
        "validation": _split_summary(valid),
        "output_columns": columns,
        "warnings": [
            "No model trained; class rates are historical descriptive summaries, not accuracy.",
            "Validation features may use earlier training-period prices; that history was already known at prediction time.",
            "The final holdout is structurally validated but its prices are excluded before all feature and label calculations.",
            "20-return volatility annualization assumes a conventional sqrt(sessions-per-year) scaling.",
            "Only observed sessions are checked; official NSE trading-calendar completeness is not verified.",
        ],
    }
    return train, valid, report


def save_development_features(
    source_csv: str | Path,
    output_dir: str | Path,
    *,
    validation_start: str = "2023-01-01",
    holdout_start: str = "2025-01-01",
    overwrite: bool = False,
) -> dict:
    """Write two development-only model-ready tables and a transparent report."""
    train, valid, report = build_development_features(
        source_csv, validation_start=validation_start, holdout_start=holdout_start
    )
    destination = Path(output_dir)
    paths = [destination / name for name in OUTPUT_NAMES]
    if not overwrite and any(path.exists() for path in paths):
        raise FeatureDatasetError("Output already exists; specify another directory or pass --overwrite")

    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination) as tmp:
        stage = Path(tmp)
        for frame, name in ((train, OUTPUT_NAMES[0]), (valid, OUTPUT_NAMES[1])):
            exported = frame.copy()
            exported["date"] = exported["date"].dt.strftime("%Y-%m-%d")
            exported.to_csv(stage / name, index=False)
        (stage / OUTPUT_NAMES[2]).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for target_path in paths:
            (stage / target_path.name).replace(target_path)
    return report
