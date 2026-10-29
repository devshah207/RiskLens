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
OHLC_FEATURE_COLUMNS = ("intraday_range", "intraday_return", "opening_gap")
EXPANDED_FEATURE_COLUMNS = FEATURE_COLUMNS + OHLC_FEATURE_COLUMNS
TECHNICAL_FEATURE_COLUMNS = ("rsi_14", "ema_50_distance", "volatility_10d")
FULL_FEATURE_COLUMNS = EXPANDED_FEATURE_COLUMNS + TECHNICAL_FEATURE_COLUMNS
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



def _wilder_rsi_14(closes: pd.Series) -> pd.Series:
    """Causal 14-return Wilder RSI, seeded with the first 14 mean gains/losses.

    First value occurs at the 15th close (index 14). With no gains and no
    losses, RSI is neutral 50; zero losses with gains yields RSI 100.
    No values from later dates are ever consulted to calculate an earlier RSI.
    """
    prices = closes.to_numpy(dtype=float)
    n = len(prices)
    values = np.full(n, np.nan, dtype=float)
    if n < 15:
        return pd.Series(values, index=closes.index, name="rsi_14")
    delta = np.diff(prices)
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    mean_gain = float(gains[:14].mean())
    mean_loss = float(losses[:14].mean())

    def score(gain: float, loss: float) -> float:
        if loss == 0:
            return 50.0 if gain == 0 else 100.0
        if gain == 0:
            return 0.0
        rs = gain / loss
        return 100.0 - 100.0 / (1.0 + rs)

    values[14] = score(mean_gain, mean_loss)
    for i in range(15, n):
        mean_gain = (mean_gain * 13 + gains[i - 1]) / 14
        mean_loss = (mean_loss * 13 + losses[i - 1]) / 14
        values[i] = score(mean_gain, mean_loss)
    return pd.Series(values, index=closes.index, name="rsi_14")


def build_development_features(
    source_csv: str | Path,
    *,
    validation_start: str = "2023-01-01",
    holdout_start: str = "2025-01-01",
    downside_threshold: float = 0.01,
    sessions_per_year: int = 252,
    include_ohlc_features: bool = False,
    include_technical_features: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return chronological train and validation tables plus an audit report.

    The optional OHLC predictors use only current-session or earlier prices.
    Default behavior is unchanged: the original three closing-price features.
    Output columns: prediction date; 1-session/5-session trailing simple returns;
    trailing 20-return annualized sample volatility; historical binary target.
    A date t is labeled only when the following observed trading session is
    within the SAME training/validation period. Warm-up feature rows are removed.
    """
    vstart = _strict_date(validation_start, "validation_start")
    hstart = _strict_date(holdout_start, "holdout_start")
    if not isinstance(include_ohlc_features, bool):
        raise FeatureDatasetError("include_ohlc_features must be a boolean")
    if not isinstance(include_technical_features, bool):
        raise FeatureDatasetError("include_technical_features must be a boolean")
    if include_technical_features and not include_ohlc_features:
        raise FeatureDatasetError("technical features require include_ohlc_features=True for the controlled 6-vs-9 comparison")
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
    dev = prices.loc[prices["date"] < hstart, ["date", "open", "high", "low", "close"]].copy().reset_index(drop=True)
    if len(dev) < 22:  # 21 closes for 20 returns, plus one later close for a label.
        raise FeatureDatasetError("Need at least 22 development price rows for 20-return volatility and a next-session label")

    # All three features are available AFTER session t closes; no negative shift.
    dev["return_1d"] = dev["close"].pct_change(fill_method=None)
    dev["return_5d"] = dev["close"].pct_change(periods=5, fill_method=None)
    dev["volatility_20d"] = (
        dev["return_1d"].rolling(window=20, min_periods=20).std(ddof=1)
        * np.sqrt(sessions_per_year)
    )

    # Intraday features are known after today closes, never use tomorrow's OHLC.
    if include_ohlc_features:
        dev["intraday_range"] = (dev["high"] - dev["low"]) / dev["open"]
        dev["intraday_return"] = dev["close"] / dev["open"] - 1.0
        dev["opening_gap"] = dev["open"] / dev["close"].shift(1) - 1.0

    if include_technical_features:
        # All three measurements use only current/earlier sessions. They are
        # descriptive indicators, not fitted prediction models.
        dev["rsi_14"] = _wilder_rsi_14(dev["close"])
        ema = dev["close"].ewm(span=50, adjust=False, min_periods=50).mean()
        dev["ema_50_distance"] = dev["close"] / ema - 1.0
        dev["volatility_10d"] = (
            dev["return_1d"].rolling(window=10, min_periods=10).std(ddof=1)
            * np.sqrt(sessions_per_year)
        )

    active_features = (FULL_FEATURE_COLUMNS if include_technical_features else
                       EXPANDED_FEATURE_COLUMNS if include_ohlc_features else FEATURE_COLUMNS)

    # Shifting into the future is permitted ONLY for constructing the label.
    dev["next_session_date"] = dev["date"].shift(-1)
    next_close = dev["close"].shift(-1)
    has_label = next_close.notna()
    target = pd.Series(pd.NA, index=dev.index, dtype="Int8")
    next_return = (next_close.loc[has_label] / dev.loc[has_label, "close"]) - 1.0
    target.loc[has_label] = (next_return <= -downside_threshold).astype("int8")
    dev[TARGET_COLUMN] = target

    ready_features = dev[list(active_features)].notna().all(axis=1)
    # Boundary purge: a prediction from 2022 cannot learn a label observed in 2023.
    training = (dev["date"] < vstart) & (dev["next_session_date"] < vstart)
    validation = (dev["date"] >= vstart) & (dev["next_session_date"] < hstart)
    columns = ["date", *active_features, TARGET_COLUMN]
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
        "feature_columns": list(active_features),
        "feature_definitions": {
            "return_1d": "close[t]/close[t-1] - 1",
            "return_5d": "close[t]/close[t-5] - 1",
            "volatility_20d": f"sample standard deviation of returns[t-19:t] times sqrt({sessions_per_year})",
            **({
                "intraday_range": "(high[t]-low[t])/open[t]",
                "intraday_return": "close[t]/open[t]-1",
                "opening_gap": "open[t]/close[t-1]-1",
            } if include_ohlc_features else {}),
            **({
                "rsi_14": "14-return Wilder RSI, initial mean gain/loss over first 14 returns, causal recursive smoothing",
                "ema_50_distance": "close[t]/EMA50[t]-1, pandas ewm(span=50, adjust=False, min_periods=50)",
                "volatility_10d": f"trailing 10-return sample std times sqrt({sessions_per_year})",
            } if include_technical_features else {}),
        },
        "first_20_price_rows_have_incomplete_feature_history": True,
        "technical_feature_warmup_price_rows": 49 if include_technical_features else None,
        "technical_features_requested": include_technical_features,
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
    include_ohlc_features: bool = False,
    include_technical_features: bool = False,
) -> dict:
    """Write two development-only model-ready tables and a transparent report."""
    train, valid, report = build_development_features(
        source_csv, validation_start=validation_start, holdout_start=holdout_start,
        include_ohlc_features=include_ohlc_features,
        include_technical_features=include_technical_features,
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
