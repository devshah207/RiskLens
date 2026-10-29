"""One-time locked out-of-time holdout evaluation. Explicit opt-in is REQUIRED.

The previous development-only functions continue to refuse 2025+ data.
This module alone handles 2025-01-01..2026-09-30, and never optimizes or
selects parameters using holdout outcomes. User-supplied sources only.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from risklens.data.validation import load_ohlc_csv, validate_ohlc
from risklens.evaluation.features import MULTIFACTOR_FEATURE_COLUMNS
from risklens.evaluation.metrics import _metrics
from risklens.evaluation.risk_validation import _cases, _reliability, _var_monitor
from risklens.features.dataset import (
    FULL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_development_features,
)
from risklens.models.frozen_comparison import (
    TRIAL86,
    _validate_probabilities,
    verify_trial86,
)
from risklens.models.optuna_research import _lstm_one
from risklens.models.sequence_lstm import build_causal_history, make_sequences
from risklens.models.xgboost_walk_forward import FIXED_XGB_PARAMS

HOLDOUT_START = pd.Timestamp("2025-01-01")
END_EXCLUSIVE = pd.Timestamp("2026-10-01")
EXPECTED_LAST_SESSION = pd.Timestamp("2026-09-30")
MAX_EXTERNAL_AGE_DAYS = 7
LOOKBACK = TRIAL86["lookback"]
FEATURES = MULTIFACTOR_FEATURE_COLUMNS
OUTPUT_PREDICTIONS = "final_holdout_predictions.csv"
OUTPUT_REPORT = "final_holdout_report.json"
OUTPUT_MEMO = "final_holdout_verdict.md"
OUTPUT_RELIABILITY = "final_holdout_reliability.csv"
OUTPUT_CASES = "final_holdout_error_cases.csv"


class HoldoutError(ValueError):
    """Locked holdout inputs or evidence do not match the preregistered contract."""


def _source_hash(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _read_holdout_vix(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Use the predeclared zero-OHL VIX quarantine, no other repairs or imputation."""
    path = Path(path)
    if not path.is_file():
        raise HoldoutError(f"VIX OHLC CSV not found: {path}")
    raw = pd.read_csv(path, dtype="string", encoding="utf-8-sig")
    raw.columns = [str(x).strip().lower() for x in raw.columns]
    if len(set(raw.columns)) != len(raw.columns):
        raise HoldoutError("Duplicate VIX columns")
    required = ("date", "open", "high", "low", "close")
    if any(c not in raw for c in required):
        raise HoldoutError("VIX file must have Date/Open/High/Low/Close columns")
    def numeric(col):
        return pd.to_numeric(
            raw[col].str.strip().str.replace(",", "", regex=False), errors="coerce"
        )
    bad_ohl = (numeric("open") == 0) & (numeric("high") == 0) & (numeric("low") == 0)
    quarantine = bad_ohl & (numeric("close") > 0)
    checked = validate_ohlc(raw.loc[~quarantine, list(required)].reset_index(drop=True))
    if checked.empty or checked.date.min() >= HOLDOUT_START or (
        checked.date.max() < HOLDOUT_START
    ):
        raise HoldoutError("VIX source needs 2024 year-end context and 2025+ observations")
    if (checked.date >= END_EXCLUSIVE).any():
        raise HoldoutError("VIX contains dates after locked 2026-09-30 cutoff")
    return checked[["date", "close"]].rename(
        columns={"date": "vix_source_date", "close": "india_vix_level"}
    ), {
        "sha256": _source_hash(path),
        "quarantined_zero_ohl_rows": int(quarantine.sum()),
        "usable_rows": len(checked),
        "source_first": str(checked.date.min().date()),
        "source_last": str(checked.date.max().date()),
    }


def _read_holdout_sp500(path: str | Path) -> tuple[pd.DataFrame, dict]:
    prices = load_ohlc_csv(path)
    if prices.date.min() >= HOLDOUT_START or prices.date.max() < HOLDOUT_START:
        raise HoldoutError("S&P 500 source must span 2024 year-end and 2025+")
    if (prices.date >= END_EXCLUSIVE).any():
        raise HoldoutError("S&P 500 file contains dates beyond locked 2026-09-30 cutoff")
    prices["sp500_previous_return_1d"] = prices.close.pct_change(fill_method=None)
    table = prices[["date", "sp500_previous_return_1d"]].rename(
        columns={"date": "sp500_source_date"}
    )
    return table, {"sha256": _source_hash(path), "rows": len(prices)}


def _attach_holdout_external(
    valid: pd.DataFrame, vix: pd.DataFrame, sp500: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Strictly prior source DATES, never same-day, and no stale >7-calendar-day matches."""
    if any(column not in valid.columns for column in FULL_FEATURE_COLUMNS):
        raise HoldoutError("Missing expected causal NIFTY indicator features")
    tolerance = pd.Timedelta(days=MAX_EXTERNAL_AGE_DAYS)
    out = pd.merge_asof(
        valid.sort_values("date"),
        vix.sort_values("vix_source_date"),
        left_on="date", right_on="vix_source_date",
        direction="backward", tolerance=tolerance, allow_exact_matches=False,
    )
    out = pd.merge_asof(
        out.sort_values("date"),
        sp500.sort_values("sp500_source_date"),
        left_on="date", right_on="sp500_source_date",
        direction="backward", tolerance=tolerance, allow_exact_matches=False,
    )
    for source_col in ("vix_source_date", "sp500_source_date"):
        observed = out[source_col].notna()
        ages = out.loc[observed, "date"] - out.loc[observed, source_col]
        if ((ages < pd.Timedelta(days=1)) | (ages > tolerance)).any():
            raise HoldoutError(f"Future, same-day, or stale external source: {source_col}")
    if out.date.duplicated().any() or not out.date.is_monotonic_increasing:
        raise HoldoutError("Invalid aligned holdout dates")
    ready = out[list(FEATURES)].notna().all(axis=1)
    return out, {
        "total_labeled_holdout_dates": int(len(out)),
        "complete_current_day_external": int(ready.sum()),
        "missing_current_day_external": int((~ready).sum()),
        "missing_date_sample": out.loc[~ready, "date"].dt.strftime("%Y-%m-%d").head(12).tolist(),
        "rule": "strictly preceding calendar date for VIX and previous completed US session, max 7 calendar days",
    }


def _holdout_sequences(history: pd.DataFrame, labeled: pd.DataFrame):
    """Specialized locked-holdout path; does not weaken development make_sequences."""
    if history.date.duplicated().any() or not history.date.is_monotonic_increasing:
        raise HoldoutError("History must have one row per session in chronological order")
    if (history.date >= END_EXCLUSIVE).any():
        raise HoldoutError("History contains dates beyond locked evaluation cutoff")
    if labeled.empty or (
        (labeled.date < HOLDOUT_START) | (labeled.date >= END_EXCLUSIVE)
    ).any():
        raise HoldoutError("Unsafe or empty labeled holdout")
    values = history.loc[:, list(FEATURES)].to_numpy(dtype=float)
    positions = pd.Series(np.arange(len(history)), index=history.date)
    xs, keep = [], []
    for row in labeled.itertuples():
        date = row.date
        if date not in positions:
            raise HoldoutError(f"Holdout date absent from causal feature history: {date}")
        i = int(positions.at[date])
        if i + 1 < LOOKBACK:
            continue
        window = values[i-LOOKBACK+1:i+1]
        if not np.isfinite(window).all():
            continue
        xs.append(window)
        keep.append(row.Index)
    if not xs:
        raise HoldoutError("No holdout dates with full 45-session external history")
    return np.asarray(xs, dtype=np.float32), labeled.loc[keep].reset_index(drop=True)


def _historical_var(prices: pd.DataFrame, dates: pd.DatetimeIndex, labels: np.ndarray):
    prices = prices.loc[prices.date < END_EXCLUSIVE, ["date", "close"]].copy()
    prices["r"] = prices.close.pct_change(fill_method=None)
    prices["next_r"] = prices.close.shift(-1)/prices.close - 1
    prices["var_95_loss_fraction"] = -prices.r.rolling(250, min_periods=250).quantile(0.05)
    prices["var_99_loss_fraction"] = -prices.r.rolling(250, min_periods=250).quantile(0.01)
    matched = pd.DataFrame({"date": dates}).merge(
        prices[["date", "next_r", "var_95_loss_fraction", "var_99_loss_fraction"]],
        on="date", validate="one_to_one", how="left",
    )
    if matched.isna().any().any():
        raise HoldoutError("Missing next-session returns or 250-session VaR history")
    if not np.array_equal(
        (matched.next_r.to_numpy(dtype=float) <= -0.01).astype(int), labels
    ):
        raise HoldoutError("Realized NIFTY returns disagree with frozen event labels")
    matched["validation_year"] = matched.date.dt.year
    for level in ("95", "99"):
        matched[f"var_{level}_exception"] = (
            -matched.next_r > matched[f"var_{level}_loss_fraction"]
        ).astype(int)
    return matched


def evaluate_holdout(
    nifty_csv: str | Path,
    development_external_csv: str | Path,
    vix_csv: str | Path,
    sp500_csv: str | Path,
    study_db: str | Path,
    *,
    device: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Exactly one frozen 2025–2026 scoring pass; no score-based retuning."""
    if device not in ("auto", "cpu", "mps"):
        raise HoldoutError("Device must be auto/cpu/mps")
    trial = verify_trial86(study_db)
    prices = load_ohlc_csv(nifty_csv)
    if prices.date.max() != EXPECTED_LAST_SESSION:
        raise HoldoutError("NIFTY file must end exactly 2026-09-30 for locked holdout")
    if prices.date.min() >= pd.Timestamp("2017-01-01"):
        raise HoldoutError("NIFTY needs its original 2016+ training and volatility history")
    # Explicit opt-in: the existing feature builder uses a validation partition of 2025+
    # here ONLY. Its train partition purges any 2024 label crossing into 2025.
    train, valid, split = build_development_features(
        nifty_csv,
        validation_start="2025-01-01",
        holdout_start="2026-10-01",
        include_ohlc_features=True,
        include_technical_features=True,
    )
    if split["training_boundary_labels_purged"] != 1:
        raise HoldoutError("Expected one purged December 2024 cross-boundary label")
    if train.date.max() >= HOLDOUT_START or valid.date.min() < HOLDOUT_START:
        raise HoldoutError("Train and holdout dates crossed the 2025 split")
    # Verified original pre-2025 external data; no change to development path.
    dev_history, development_audit = build_causal_history(
        nifty_csv, development_external_csv
    )
    if (dev_history.date >= HOLDOUT_START).any():
        raise HoldoutError("Pre-2025 history contains holdout values")
    train = train.merge(
        dev_history[["date", *FEATURES[-2:]]], on="date", how="left", validate="one_to_one"
    ).dropna(subset=list(FEATURES)).reset_index(drop=True)
    if train.empty or len(np.unique(train[TARGET_COLUMN])) != 2:
        raise HoldoutError("Insufficient pre-2025 training rows or label diversity")
    vix, vix_audit = _read_holdout_vix(vix_csv)
    sp500, sp_audit = _read_holdout_sp500(sp500_csv)
    full_valid, holdout_audit = _attach_holdout_external(valid, vix, sp500)
    # Full history includes missing external rows so 45-session windows cannot silently
    # bridge stale/missing dates; those windows are excluded with an explicit audit.
    history = pd.concat(
        [
            dev_history[["date", *FEATURES]],
            full_valid[["date", *FEATURES]],
        ], ignore_index=True
    )
    train_x, train_y, train_dates = make_sequences(
        dev_history, train, FEATURES, LOOKBACK
    )
    if train_dates.max() >= HOLDOUT_START:
        raise HoldoutError("LSTM train sequence crossed holdout boundary")
    valid_x, eligible = _holdout_sequences(history, full_valid)
    excluded = len(full_valid) - len(eligible)
    y_train = train[TARGET_COLUMN].to_numpy(dtype=int)
    y = eligible[TARGET_COLUMN].to_numpy(dtype=int)
    x_train = train[list(FEATURES)].to_numpy(dtype=float)
    x_test = eligible[list(FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise HoldoutError("Nonfinite tabular model features")
    model = XGBClassifier(**FIXED_XGB_PARAMS)
    model.fit(x_train, y_train)
    px = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
    pl = np.asarray(_lstm_one(
        train_x, train_y, valid_x, dict(TRIAL86), seed=2026+2025, device=device
    ), dtype=float)
    _validate_probabilities(y, px, "Fixed XGBoost")
    _validate_probabilities(y, pl, "Optuna LSTM #86")
    scored = pd.DataFrame({
        "date": eligible.date,
        "validation_year": eligible.date.dt.year,
        "actual_downside_event": y,
        "xgboost_11_fixed_probability": px,
        "lstm_optuna_86_probability": pl,
    })
    var = _historical_var(prices, pd.DatetimeIndex(scored.date), y)
    # Reuse generic statistical routines; do not touch old development-only readers.
    reliability, calibration = _reliability(scored)
    scores = {
        name: _metrics(y, scored[column].to_numpy(dtype=float))
        for name, column in (
            ("xgboost_fixed", "xgboost_11_fixed_probability"),
            ("lstm_optuna_trial86", "lstm_optuna_86_probability"),
        )
    }
    years = {}
    for year in (2025, 2026):
        subset = scored[scored.validation_year == year]
        if subset.empty:
            raise HoldoutError(f"Missing holdout year {year}; source coverage incomplete")
        yy = subset.actual_downside_event.to_numpy(dtype=int)
        years[str(year)] = {
            "observations": int(len(subset)), "events": int(yy.sum()),
            "models": {
                name: _metrics(yy, subset[column].to_numpy(dtype=float))
                for name, column in (
                    ("xgboost_fixed", "xgboost_11_fixed_probability"),
                    ("lstm_optuna_trial86", "lstm_optuna_86_probability"),
                )
            },
        }
    summary = {
        "title": "RiskLens: ONE-TIME 2025–2026 Locked Out-of-Time Holdout Verdict",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_start_inclusive": "2025-01-01",
        "test_last_possible_predictor": "2026-09-29 (subject to trading calendar)",
        "source_last_session": "2026-09-30",
        "selected_before_opening": "fixed_xgboost_11",
        "challenger": "lstm_optuna_trial_86",
        "selection_remains_frozen": True,
        "study_record": trial,
        "frozen_xgb_params": dict(FIXED_XGB_PARAMS),
        "frozen_lstm_params": dict(TRIAL86),
        "features": list(FEATURES),
        "training_cutoff_exclusive": "2025-01-01",
        "lstm_seed": 4051,
        "device_requested": device,
        "split_purge_count": split["training_boundary_labels_purged"],
        "pre_2025_labeled_training_rows": int(len(train)),
        "pre_2025_lstm_training_sequences": int(len(train_x)),
        "eligible_holdout_rows": int(len(scored)),
        "excluded_holdout_rows_incomplete_45_session_history": int(excluded),
        "excluded_holdout_date_sample": [
            str(x.date()) for x in full_valid.loc[
                ~full_valid.date.isin(scored.date), "date"
            ].head(20)
        ],
        "source_alignment": holdout_audit,
        "vix_source": vix_audit,
        "sp500_source": sp_audit,
        "source_hashes": {
            "nifty": _source_hash(nifty_csv),
            "development_external": _source_hash(development_external_csv),
            "vix": _source_hash(vix_csv),
            "sp500": _source_hash(sp500_csv),
            "optuna_study": trial["database_sha256"],
        },
        "development_history_provenance": development_audit,
        "model_scores": scores,
        "yearly_scores": years,
        "probability_calibration": calibration,
        "historical_var": _var_monitor(var),
        "limitations": [
            "The fixed model selection remains the original decision; better holdout LSTM performance does not trigger refitting or retuning.",
            "This is a single index-level probability classification study, not a bank risk capital model or investment strategy.",
            "Historical simulation VaR is independent of the ML probability models; no claim that XGBoost improves VaR.",
            "External source dates strictly precede forecast date; publication timestamps and licensing are not independently authenticated.",
            "Early holdout observations or missing external windows may be excluded by the predeclared 45-session completeness rule.",
            "Rare-event metrics and tail exception tests have substantial uncertainty; no statistical significance claim without separate analysis.",
            "Post-2024 index returns are used only as evaluation outcomes and for rolling VaR forecasts known by time t.",
        ],
    }
    return scored, reliability, summary


def _memo(report: dict) -> str:
    lines = [
        "# RiskLens — Locked 2025–2026 Final Holdout Verdict", "",
        "**No tuning, recalibration, threshold selection or model promotion after opening.**",
        f"Holdout observations: {report['eligible_holdout_rows']}; exclusions: {report['excluded_holdout_rows_incomplete_45_session_history']}.",
        "", "| Frozen candidate | Brier ↓ | Log loss ↓ | ROC-AUC ↑ | AP ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in report["model_scores"].items():
        lines.append(
            f"| {name} | {m['brier_score']:.6f} | {m['log_loss']:.6f} | "
            f"{m['roc_auc']} | {m['average_precision']} |"
        )
    lines += [
        "", "## Annual results", "",
        "| Year | Candidate | Observations | Events | Brier ↓ | ROC-AUC ↑ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for year, data in report["yearly_scores"].items():
        for name, m in data["models"].items():
            lines.append(
                f"| {year} | {name} | {data['observations']} | {data['events']} | "
                f"{m['brier_score']:.6f} | {m['roc_auc']} |"
            )
    lines += ["", "## Historical VaR backtest", ""]
    for level, info in report["historical_var"].items():
        lines.append(
            f"- {level}% VaR: {info['exceptions']}/{info['observations']} breaches "
            f"({info['observed_rate']:.4%}), expected {info['expected_rate']:.2%}; "
            f"exact coverage p={info['exact_binomial_coverage_p_value']}"
        )
    lines += [
        "", "## Model-selection verdict",
        "Fixed XGBoost remains the **preselected champion candidate**. "
        "Holdout findings are reported without post-hoc reoptimization.",
        "", "## Limitations", "",
        *["- "+x for x in report["limitations"]], "",
    ]
    return "\n".join(lines)


def save_holdout(
    nifty_csv: str | Path,
    development_external_csv: str | Path,
    vix_csv: str | Path,
    sp500_csv: str | Path,
    study_db: str | Path,
    out_dir: str | Path,
    *,
    device: str = "auto",
    confirm_locked_protocol: bool = False,
) -> dict:
    if not confirm_locked_protocol:
        raise HoldoutError(
            "Final holdout remains sealed. Explicit --confirm-locked-protocol is required."
        )
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise HoldoutError(
            "Final holdout output directory is not empty. One-time evaluation refuses "
            "overwrite. Do not use a fresh output directory to tune on viewed outcomes."
        )
    scored, reliability, report = evaluate_holdout(
        nifty_csv, development_external_csv, vix_csv, sp500_csv, study_db, device=device
    )
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as temporary:
        tmp = Path(temporary)
        export = scored.copy()
        export["date"] = export.date.dt.strftime("%Y-%m-%d")
        export.to_csv(tmp/OUTPUT_PREDICTIONS, index=False, float_format="%.12g")
        reliability.to_csv(tmp/OUTPUT_RELIABILITY, index=False)
        cases = _cases(scored)
        cases["date"] = cases.date.dt.strftime("%Y-%m-%d")
        cases.to_csv(tmp/OUTPUT_CASES, index=False)
        (tmp/OUTPUT_REPORT).write_text(
            json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8"
        )
        (tmp/OUTPUT_MEMO).write_text(_memo(report), encoding="utf-8")
        for name in (
            OUTPUT_PREDICTIONS, OUTPUT_RELIABILITY, OUTPUT_CASES,
            OUTPUT_REPORT, OUTPUT_MEMO,
        ):
            (tmp/name).replace(out/name)
    return report
