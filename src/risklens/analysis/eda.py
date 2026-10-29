"""Development-only exploratory analysis of NIFTY 50 closing prices.

No statistics, returns, rolling features, or outcomes are computed for the
predeclared final holdout (default: 2025-01-01 onward). All reported figures
are descriptive, NOT evidence of predictive model performance.
"""

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from risklens.data.validation import OHLCValidationError, load_ohlc_csv


class EDAError(ValueError):
    """Invalid EDA parameters, inputs, or output destination."""


def _date_cutoff(raw: str) -> pd.Timestamp:
    """Parse a strict, unambiguous ISO date."""
    try:
        parsed = datetime.strptime(raw, "%Y-%m-%d")
        if parsed.strftime("%Y-%m-%d") != raw:
            raise ValueError("not canonical")
    except (TypeError, ValueError) as exc:
        raise EDAError("--holdout-start must be a valid YYYY-MM-DD date") from exc
    return pd.Timestamp(parsed)


def _rounded(value: float, places: int = 8) -> float | None:
    """Serialize finite statistics as ordinary JSON numbers (not NaN)."""
    return (
        round(float(value), places) if pd.notna(value) and np.isfinite(value) else None
    )


def analyze_development(
    source_csv: str | Path,
    *,
    holdout_start: str = "2025-01-01",
    rolling_window: int = 20,
    sessions_per_year: int = 252,
    downside_threshold: float = 0.01,
) -> tuple[pd.DataFrame, dict]:
    """Return development-only time series and interpretable summary statistics.

    Daily return on session t uses only close[t] and close[t-1], both known
    after t closes. Rolling sample SD includes return[t] and the previous
    (window - 1) returns. No future returns, labels, or holdout outcomes are
    calculated. NaNs at the start indicate insufficient history, NOT bad data.

    This method's volatility scaling uses sqrt(sessions_per_year). It does NOT
    assert independence or constant volatility, and it does not estimate VaR.
    """
    if (
        not isinstance(rolling_window, int)
        or isinstance(rolling_window, bool)
        or rolling_window < 2
    ):
        raise EDAError("rolling_window must be an integer >= 2")
    if (
        not isinstance(sessions_per_year, int)
        or isinstance(sessions_per_year, bool)
        or sessions_per_year < 1
    ):
        raise EDAError("sessions_per_year must be a positive integer")
    if not np.isfinite(downside_threshold) or not (0 < downside_threshold < 1):
        raise EDAError("downside_threshold must be strictly between 0 and 1")

    cutoff = _date_cutoff(holdout_start)
    try:
        all_prices = load_ohlc_csv(source_csv)
    except OHLCValidationError as exc:
        raise EDAError(str(exc)) from exc

    # First seal the final holdout; only then calculate ANY returns/statistics.
    dev = (
        all_prices.loc[all_prices["date"] < cutoff, ["date", "close"]]
        .copy()
        .reset_index(drop=True)
    )
    if len(dev) < 2:
        raise EDAError("Need at least two sessions strictly before holdout start")

    dev["daily_return"] = dev["close"].pct_change(fill_method=None)
    dev["log_return"] = np.log1p(dev["daily_return"])
    volatility_name = f"rolling_vol_{rolling_window}d_annualized"
    dev[volatility_name] = dev["daily_return"].rolling(
        window=rolling_window, min_periods=rolling_window
    ).std(ddof=1) * np.sqrt(sessions_per_year)

    valid = dev["daily_return"].dropna()
    # Each return is attributed to the day on which the price change is observed.
    # This differs from the prior event report's PREDICTION-DATE yearly grouping.
    annual = []
    for year, group in dev.groupby(dev["date"].dt.year, sort=True):
        rr = group["daily_return"].dropna()
        annual.append(
            {
                "return_year": int(year),
                "return_observations": int(len(rr)),
                "mean_daily_return": _rounded(rr.mean()),
                "annualized_volatility": _rounded(
                    rr.std(ddof=1) * np.sqrt(sessions_per_year)
                ),
            }
        )

    downside_count = int((valid <= -downside_threshold).sum())
    quantile_levels = {"p01": 0.01, "p05": 0.05, "p50": 0.50, "p95": 0.95, "p99": 0.99}
    worst = dev.nsmallest(min(5, len(valid)), "daily_return")
    worst_days = [
        {"date": row.date.strftime("%Y-%m-%d"), "return": _rounded(row.daily_return)}
        for row in worst.itertuples(index=False)
        if pd.notna(row.daily_return)
    ]

    summary = {
        "scope": "Development observations ONLY. Final holdout outcomes not analyzed.",
        "source_file": Path(source_csv).name,
        "source_sha256": sha256(Path(source_csv).read_bytes()).hexdigest(),
        "holdout_start": cutoff.date().isoformat(),
        "development_first_session": dev["date"].min().date().isoformat(),
        "development_last_session": dev["date"].max().date().isoformat(),
        "development_price_rows": len(dev),
        "valid_daily_returns": len(valid),
        "return_definition": "close[t] / close[t-1] - 1 (return dated t)",
        "log_return_definition": "ln(close[t] / close[t-1])",
        "mean_daily_return": _rounded(valid.mean()),
        "median_daily_return": _rounded(valid.median()),
        "sample_daily_std": _rounded(valid.std(ddof=1)),
        "annualized_volatility": _rounded(
            valid.std(ddof=1) * np.sqrt(sessions_per_year)
        ),
        "sample_skewness": _rounded(valid.skew()),
        "excess_kurtosis": _rounded(valid.kurt()),
        "min_daily_return": _rounded(valid.min()),
        "max_daily_return": _rounded(valid.max()),
        "return_quantiles": {
            k: _rounded(valid.quantile(q)) for k, q in quantile_levels.items()
        },
        "positive_daily_returns": int((valid > 0).sum()),
        "negative_daily_returns": int((valid < 0).sum()),
        "zero_daily_returns": int((valid == 0).sum()),
        "downside_threshold": downside_threshold,
        "downside_event_close_dates": downside_count,
        "downside_event_fraction": _rounded(downside_count / len(valid)),
        "downside_count_attribution": "Return is grouped by the realized close date, not prediction date",
        "rolling_window_sessions": rolling_window,
        "rolling_annualization_sessions": sessions_per_year,
        "rolling_volatility_column": volatility_name,
        "rolling_volatility_nonnull_rows": int(dev[volatility_name].notna().sum()),
        "rolling_volatility_max": _rounded(dev[volatility_name].max()),
        "worst_daily_return_dates": worst_days,
        "annual_statistics": annual,
        "warnings": [
            "Descriptive statistics only; no predictive model has been trained.",
            "The sqrt(252) annualization is a convention; it is not a claim of constant volatility.",
            "Outlying returns are observations to investigate, not automatically bad data.",
            "Index source authenticity and exchange-calendar completeness require independent checks.",
            "Do not use any 2025+ outcomes to choose features or hyperparameters.",
        ],
    }
    return dev, summary


def save_development_eda(
    source_csv: str | Path,
    output_dir: str | Path,
    *,
    holdout_start: str = "2025-01-01",
    rolling_window: int = 20,
    overwrite: bool = False,
) -> dict:
    """Validate and write development-only CSV/JSON, refusing overwrite by default."""
    result, report = analyze_development(
        source_csv, holdout_start=holdout_start, rolling_window=rolling_window
    )
    destination = Path(output_dir)
    targets = [
        destination / "nifty50_eda_development.csv",
        destination / "nifty50_eda_development.json",
    ]
    if not overwrite and any(target.exists() for target in targets):
        raise EDAError(
            "EDA output already exists. Use a different directory or --overwrite."
        )
    destination.mkdir(parents=True, exist_ok=True)
    export = result.copy()
    export["date"] = export["date"].dt.strftime("%Y-%m-%d")
    with TemporaryDirectory(dir=destination) as tmp:
        stage = Path(tmp)
        export.to_csv(stage / targets[0].name, index=False, na_rep="")
        (stage / targets[1].name).write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        for target in targets:
            (stage / target.name).replace(target)
    return report
