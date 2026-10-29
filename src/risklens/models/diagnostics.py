"""Descriptive, development-only diagnostics of ALREADY PRODUCED model predictions.

No models are fitted, calibrated, tuned or selected here. The only accepted
validation prediction dates are from 2020 through 2024 (inclusive). The final
2025+ holdout is never read. Historical labels are used for evaluation only.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

YEAR_START, YEAR_END = 2020, 2024
TARGET = "actual_downside_event"
MODEL_COLUMNS = {
    "constant_baseline": "baseline_probability",
    "logistic_3_features": "original_3feature_probability",
    "logistic_6_features": "expanded_6feature_probability",
    "xgboost_3_features": "xgboost_3_features_probability",
    "xgboost_6_features": "xgboost_6_features_probability",
}
EXPECTED_COLUMNS = (
    "validation_year",
    "date",
    TARGET,
    "original_3feature_probability",
    "expanded_6feature_probability",
    "baseline_probability",
    "xgboost_3_features_probability",
    "xgboost_6_features_probability",
)
EDGES = (0.0, 0.05, 0.10, 0.20, 0.40, 1.0)
BIN_LABELS = ("0–5%", "5–10%", "10–20%", "20–40%", "40–100%")
YEARLY_FILE = "model_yearly_diagnostics.csv"
RELIABILITY_FILE = "model_probability_reliability.csv"
CASES_FILE = "xgboost_3feature_error_examples.csv"
REPORT_FILE = "model_diagnostics_report.json"


class DiagnosticsError(ValueError):
    """Invalid prediction data or requested output operation."""


def _load_development_predictions(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise DiagnosticsError(f"Prediction CSV not found: {path}")
    try:
        df = pd.read_csv(path, dtype="string", encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise DiagnosticsError(f"Cannot read prediction CSV: {exc}") from exc
    if list(df.columns) != list(EXPECTED_COLUMNS):
        raise DiagnosticsError(
            "Prediction CSV columns differ from the exact Step 3.5 output schema"
        )
    if df.empty:
        raise DiagnosticsError("Prediction CSV contains no rows")
    date_text = df["date"]
    dates = pd.to_datetime(date_text, format="%Y-%m-%d", errors="coerce")
    if dates.isna().any() or not dates.dt.strftime("%Y-%m-%d").eq(date_text).all():
        raise DiagnosticsError("Prediction dates must be valid ISO YYYY-MM-DD dates")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise DiagnosticsError(
            "Prediction dates must be unique and sorted chronologically"
        )
    if (dates.dt.year < YEAR_START).any() or (dates.dt.year > YEAR_END).any():
        raise DiagnosticsError(
            "Only 2020–2024 development validation dates are accepted; 2025+ holdout is prohibited"
        )
    years_text = df["validation_year"]
    if not years_text.str.fullmatch(r"\d{4}", na=False).all():
        raise DiagnosticsError("validation_year must be an integer calendar year")
    years = years_text.astype(int)
    if not np.array_equal(years.to_numpy(), dates.dt.year.to_numpy()):
        raise DiagnosticsError("validation_year must match the prediction date's year")
    if not df[TARGET].isin(["0", "1"]).all():
        raise DiagnosticsError("actual_downside_event must contain only 0 or 1")
    df[TARGET] = df[TARGET].astype(int)
    for name, column in MODEL_COLUMNS.items():
        try:
            p = pd.to_numeric(df[column], errors="raise").astype(float)
        except (ValueError, TypeError) as exc:
            raise DiagnosticsError(f"Invalid numeric probabilities in {name}") from exc
        if not np.isfinite(p.to_numpy()).all() or not p.between(0, 1).all():
            raise DiagnosticsError(
                f"{name} probabilities must be finite values between 0 and 1"
            )
        df[column] = p
    df["date"] = dates
    df["validation_year"] = years
    return df


def _probability_scores(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    """Scores are descriptive, never used to change models or fit calibrators."""
    eps = np.finfo(float).eps
    p = np.clip(probabilities, eps, 1.0 - eps)
    both_classes = len(np.unique(labels)) == 2
    return {
        "brier_score": round(float(np.mean((probabilities - labels) ** 2)), 8),
        "log_loss": round(
            float(-np.mean(labels * np.log(p) + (1 - labels) * np.log1p(-p))), 8
        ),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 8)
        if both_classes
        else None,
        "average_precision": round(
            float(average_precision_score(labels, probabilities)), 8
        )
        if both_classes
        else None,
    }


def analyze_predictions(
    source_csv: str | Path, *, example_count: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Read 2020–24 validation predictions and return yearly, reliability and error diagnostics.

    Reliability bins were predeclared, not learned or adjusted based on labels.
    The ranked examples show notable misses / non-events, not trade signals.
    """
    if (
        not isinstance(example_count, int)
        or isinstance(example_count, bool)
        or not 1 <= example_count <= 100
    ):
        raise DiagnosticsError("example_count must be an integer from 1 to 100")
    frame = _load_development_predictions(source_csv)
    labels = frame[TARGET].to_numpy(dtype=int)
    years = sorted(frame["validation_year"].unique().tolist())
    yearly_rows, reliability_rows, pooled = [], [], {}
    for model_name, column in MODEL_COLUMNS.items():
        p_all = frame[column].to_numpy(dtype=float)
        pooled_probabilities = _probability_scores(labels, p_all)
        # Do not pool ROC-AUC/AP across independently fitted yearly models;
        # per-fold score rankings are meaningful, but cross-fold comparisons
        # may reflect different intercepts/event rates rather than skill.
        pooled[model_name] = {
            "brier_score": pooled_probabilities["brier_score"],
            "log_loss": pooled_probabilities["log_loss"],
            "mean_predicted_event_probability": round(float(p_all.mean()), 8),
            "observed_event_frequency": round(float(labels.mean()), 8),
        }
        for year in years:
            subset = frame[frame["validation_year"] == year]
            y = subset[TARGET].to_numpy(dtype=int)
            p = subset[column].to_numpy(dtype=float)
            yearly_rows.append(
                {
                    "model": model_name,
                    "validation_year": int(year),
                    "rows": len(subset),
                    "downside_events": int(y.sum()),
                    "observed_event_frequency": round(float(y.mean()), 8),
                    "mean_predicted_probability": round(float(p.mean()), 8),
                    "probability_gap_predicted_minus_observed": round(
                        float(p.mean() - y.mean()), 8
                    ),
                    **_probability_scores(y, p),
                }
            )
        # SAME fixed bins for every model; no optimizing edges after seeing the results.
        bin_index = np.searchsorted(EDGES[1:], p_all, side="right")
        bin_index = np.minimum(
            bin_index, len(BIN_LABELS) - 1
        )  # include p=1 in final bin
        for index, bin_label in enumerate(BIN_LABELS):
            matching = bin_index == index
            count = int(matching.sum())
            n_events = int(labels[matching].sum()) if count else 0
            reliability_rows.append(
                {
                    "model": model_name,
                    "probability_bin": bin_label,
                    "bin_lower_inclusive": EDGES[index],
                    "bin_upper": EDGES[index + 1],
                    "rows": count,
                    "events": n_events,
                    "mean_predicted_probability": round(
                        float(p_all[matching].mean()), 8
                    )
                    if count
                    else None,
                    "observed_event_frequency": round(float(labels[matching].mean()), 8)
                    if count
                    else None,
                    "small_sample_warning": bool(count < 30 or n_events < 5),
                }
            )

    # Descriptive examples: positive event assigned smallest risk, and no event
    # assigned largest risk. These are ranked mistakes, not threshold-based alerts.
    selected = "xgboost_3_features_probability"
    realized_events = (
        frame[frame[TARGET] == 1]
        .sort_values([selected, "date"], ascending=[True, True], kind="stable")
        .head(example_count)
    )
    non_events = (
        frame[frame[TARGET] == 0]
        .sort_values([selected, "date"], ascending=[False, True], kind="stable")
        .head(example_count)
    )
    case_columns = ["date", "validation_year", TARGET, selected]
    misses = realized_events[case_columns].copy()
    misses.insert(0, "case_type", "event_assigned_low_probability")
    alarms = non_events[case_columns].copy()
    alarms.insert(0, "case_type", "non_event_assigned_high_probability")
    cases = pd.concat([misses, alarms], ignore_index=True)
    yearly = pd.DataFrame(yearly_rows)
    reliability = pd.DataFrame(reliability_rows)
    report = {
        "scope": "Existing 2020–2024 development predictions ONLY; no model training or 2025+ holdout use",
        "source_filename": Path(source_csv).name,
        "source_sha256": sha256(Path(source_csv).read_bytes()).hexdigest(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "years": [int(year) for year in years],
        "rows": len(frame),
        "observed_events": int(labels.sum()),
        "observed_event_frequency": round(float(labels.mean()), 8),
        "models": MODEL_COLUMNS,
        "pooled_scores": pooled,
        "fixed_reliability_bin_edges": list(EDGES),
        "error_case_model": "xgboost_3_features",
        "examples_per_case_type_requested": example_count,
        "warnings": [
            "All scores use existing out-of-sample DEVELOPMENT predictions. No new model is fitted, calibrated, or tuned.",
            "The 2023–2024 development results have already influenced earlier model investigations; these diagnostics are not independent evidence of generalization.",
            "Reliability bins show historical frequencies, NOT a newly fitted probability calibrator.",
            "Pooled ROC-AUC and Average Precision are not reported: yearly probabilities were produced by independently fitted models with changing base rates; view ranking metrics per year.",
            "Small bins or few positive events give noisy frequencies; a 2020–2024 pooled view can hide differences across years.",
            "Predicted versus observed average gaps indicate aggregate under/over-estimation, not causal reasons for the error.",
            "The example case rankings depend on realized labels and are for retrospective investigation only, never live signals.",
            "A low probability on an actual event is not necessarily an invalid prediction: even well-calibrated models sometimes assign low probabilities to events.",
            "2025 onward cannot enter any input or result through this interface. No conclusions about the final holdout.",
        ],
    }
    return yearly, reliability, cases, report


def save_diagnostics(
    source_csv: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    example_count: int = 10,
) -> dict:
    """Save CSV diagnostics and JSON report; refuse overwrite unless explicit."""
    out = Path(output_dir)
    targets = [
        out / name for name in (YEARLY_FILE, RELIABILITY_FILE, CASES_FILE, REPORT_FILE)
    ]
    if not overwrite and any(path.exists() for path in targets):
        raise DiagnosticsError(
            "Diagnostics output already exists; choose another folder or use --overwrite"
        )
    yearly, reliability, cases, report = analyze_predictions(
        source_csv, example_count=example_count
    )
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as scratch:
        stage = Path(scratch)
        for data, filename in (
            (yearly, YEARLY_FILE),
            (reliability, RELIABILITY_FILE),
            (cases, CASES_FILE),
        ):
            export = data.copy()
            if "date" in export.columns:
                export["date"] = export["date"].dt.strftime("%Y-%m-%d")
            export.to_csv(stage / filename, index=False, float_format="%.12g")
        (stage / REPORT_FILE).write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        for target in targets:
            (stage / target.name).replace(target)
    return report
