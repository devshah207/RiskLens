"""Phase 6 XGBoost TreeSHAP research explanation on pre-2025 training rows only.

Uses XGBoost's built-in exact TreeSHAP contributions (pred_contribs) rather
than fitting a surrogate explainer. Explanations are in raw log-odds units,
NOT causal impact and NOT a holdout performance or feature-selection study.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import xgboost
from scipy.special import expit
from xgboost import XGBClassifier

from risklens.data.validation import load_ohlc_csv
from risklens.evaluation.features import MULTIFACTOR_FEATURE_COLUMNS
from risklens.evaluation.splits import FINAL_HOLDOUT_START
from risklens.features.dataset import TARGET_COLUMN
from risklens.models.sequence_lstm import build_causal_history
from risklens.models.xgboost_walk_forward import FIXED_XGB_PARAMS

FEATURES = MULTIFACTOR_FEATURE_COLUMNS
CUTOFF = pd.Timestamp(FINAL_HOLDOUT_START)
OUTPUT_FILES = ("xgboost_shap_summary.csv", "xgboost_shap_report.json", "xgboost_model_card.md")


class ExplanationError(ValueError):
    """Unsafe explanation input, model integrity, or output overwrite."""


def explain_development_xgboost(
    nifty_csv: str | Path,
    external_csv: str | Path,
    *,
    max_examples: int = 500,
) -> tuple[pd.DataFrame, dict]:
    """Fit frozen XGBoost using PRE-2025 labels only, then explain training subset.

    The last pre-2025 session is NEVER labeled from a 2025 next close.
    Refit is explanatory and not a new independently validated model.
    """
    if isinstance(max_examples, bool) or not isinstance(max_examples, int) or not 20 <= max_examples <= 5000:
        raise ExplanationError("max_examples must be between 20 and 5000")
    if not Path(nifty_csv).is_file() or not Path(external_csv).is_file():
        raise ExplanationError("NIFTY source and development external CSV are required")
    history, history_audit = build_causal_history(nifty_csv, external_csv)
    if (history["date"] >= CUTOFF).any():
        raise ExplanationError("Development history contains final holdout dates")
    prices = load_ohlc_csv(nifty_csv)
    dev = prices.loc[prices.date < CUTOFF, ["date", "close"]].copy().reset_index(drop=True)
    dev["next_close"] = dev.close.shift(-1)
    # Never use the 2025 next-close to label any pre-2025 observation.
    labeled = dev.loc[dev.next_close.notna(), ["date", "close", "next_close"]].copy()
    labeled[TARGET_COLUMN] = (labeled.next_close / labeled.close - 1 <= -0.01).astype(int)
    dataset = labeled[["date", TARGET_COLUMN]].merge(
        history[["date", *FEATURES]], on="date", how="left", validate="one_to_one"
    ).dropna(subset=list(FEATURES)).reset_index(drop=True)
    if dataset.empty or len(dataset) < 100 or dataset.date.max() >= CUTOFF:
        raise ExplanationError("Insufficient pre-2025 feature-complete training rows")
    if len(np.unique(dataset[TARGET_COLUMN])) != 2:
        raise ExplanationError("Training data need both event classes")
    x_all = dataset.loc[:, list(FEATURES)].to_numpy(dtype=float)
    y_all = dataset[TARGET_COLUMN].to_numpy(dtype=int)
    if not np.isfinite(x_all).all():
        raise ExplanationError("Nonfinite or missing pre-2025 input features")

    model = XGBClassifier(**FIXED_XGB_PARAMS)
    model.fit(x_all, y_all)
    # Fixed deterministic subset by chronological recency; do not select based on labels.
    chosen = dataset.iloc[-min(len(dataset), max_examples):]
    x = chosen.loc[:, list(FEATURES)].to_numpy(dtype=float)
    booster = model.get_booster()
    dmatrix = xgboost.DMatrix(x)
    contributions = np.asarray(booster.predict(dmatrix, pred_contribs=True), dtype=float)
    raw_margin = np.asarray(booster.predict(dmatrix, output_margin=True), dtype=float).reshape(-1)
    if contributions.shape != (len(x), len(FEATURES)+1):
        raise ExplanationError("TreeSHAP returned unexpected feature dimensions")
    if not np.isfinite(contributions).all() or not np.allclose(
        contributions.sum(axis=1), raw_margin, rtol=1e-5, atol=1e-4
    ):
        raise ExplanationError("SHAP additivity audit failed")
    predicted = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    if not np.allclose(expit(raw_margin), predicted, rtol=1e-5, atol=1e-5):
        raise ExplanationError("TreeSHAP raw margins disagree with logistic probabilities")
    gains = booster.get_score(importance_type="gain")
    features = pd.DataFrame({
        "feature": list(FEATURES),
        "mean_absolute_shap_log_odds": np.mean(np.abs(contributions[:, :-1]), axis=0),
        "mean_signed_shap_log_odds": np.mean(contributions[:, :-1], axis=0),
        "xgboost_split_gain": [float(gains.get(f"f{i}", 0.0)) for i in range(len(FEATURES))],
    }).sort_values("mean_absolute_shap_log_odds", ascending=False).reset_index(drop=True)
    report = {
        "title": "RiskLens Phase 6 | Fixed 11-feature XGBoost TreeSHAP model card",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "IN-SAMPLE PRE-2025 training explanations only; final holdout never read as labeled data",
        "frozen_params": dict(FIXED_XGB_PARAMS),
        "input_features": list(FEATURES),
        "pre_2025_training_rows": len(dataset),
        "pre_2025_training_events": int(y_all.sum()),
        "latest_training_label_prediction_date": str(dataset.date.max().date()),
        "explanation_rows": len(chosen),
        "earliest_explanation_date": str(chosen.date.min().date()),
        "latest_explanation_date": str(chosen.date.max().date()),
        "explanation_sample_rule": "last N eligible pre-2025 training dates, independent of labels",
        "feature_contribution_units": "raw margin/log-odds (NOT probability units)",
        "tree_shap_additivity_max_abs_error": float(np.max(np.abs(contributions.sum(axis=1)-raw_margin))),
        "external_provenance": history_audit,
        "source_sha256": {
            "nifty": sha256(Path(nifty_csv).read_bytes()).hexdigest(),
            "development_external": sha256(Path(external_csv).read_bytes()).hexdigest(),
        },
        "xgboost_version": xgboost.__version__,
        "limitations": [
            "Feature attributions are descriptive; they do not establish financial causation.",
            "Mean absolute SHAP can reflect collinearity and feature interactions; it is not stable model selection evidence.",
            "These explanations are in-sample, generated from a pre-2025 refit with the frozen configuration, not from holdout data.",
            "LSTM trial #86 is not explained by this XGBoost-specific method; sequence-model attribution requires separate audit.",
            "No changes to the model, prediction threshold, training data cutoff or post-2024 scoring are made.",
        ],
    }
    return features, report


def _model_card(report: dict, summary: pd.DataFrame) -> str:
    lines = [
        "# RiskLens — Fixed XGBoost Model Card and TreeSHAP",
        "",
        "**Development-data explanation. Not an independent validation or causal analysis.**",
        "",
        f"Training rows: {report['pre_2025_training_rows']}; events: {report['pre_2025_training_events']}. "
        f"Explanation rows: {report['explanation_rows']}.",
        "",
        "| Feature | Mean absolute SHAP (log-odds) | Mean signed SHAP | Split gain |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.feature} | {row.mean_absolute_shap_log_odds:.6f} | "
            f"{row.mean_signed_shap_log_odds:.6f} | {row.xgboost_split_gain:.6f} |"
        )
    lines += ["", "## Interpretation and limitations",
              *["- " + x for x in report["limitations"]],
              "", "Native XGBoost pred_contribs is an additive TreeSHAP breakdown of raw margins, "
              "not a direct contribution to predicted event probability.", ""]
    return "\n".join(lines)


def save_explanations(
    nifty_csv: str | Path,
    external_csv: str | Path,
    output_dir: str | Path,
    *,
    max_examples: int = 500,
    overwrite: bool = False,
) -> dict:
    out = Path(output_dir)
    targets = [out / filename for filename in OUTPUT_FILES]
    if not overwrite and any(path.exists() for path in targets):
        raise ExplanationError("Explanation outputs already exist; use --overwrite")
    features, report = explain_development_xgboost(
        nifty_csv, external_csv, max_examples=max_examples
    )
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out) as td:
        temp = Path(td)
        features.to_csv(temp / OUTPUT_FILES[0], index=False)
        (temp / OUTPUT_FILES[1]).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        (temp / OUTPUT_FILES[2]).write_text(_model_card(report, features), encoding="utf-8")
        for path in targets:
            (temp / path.name).replace(path)
    return report
