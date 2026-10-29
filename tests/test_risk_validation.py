"""Phase 5: strict input boundaries, matching evidence, honest validation."""
import numpy as np
import pandas as pd
import pytest

from risklens.cli import build_parser
from risklens.evaluation.risk_validation import (
    RiskValidationError, validate_development, load_predictions, load_var,
    _exception_independence, save_risk_validation, OUTPUT_FILES,
)


def _fixture(tmp_path, *, include_holdout=False):
    pred_rows, var_rows = [], []
    years = range(2020, 2026) if include_holdout else range(2020, 2025)
    for year in years:
        for i in range(8):
            day = pd.Timestamp(f"{year}-02-03") + pd.Timedelta(days=i)
            event = int(i in (1, 5))
            ret = -0.02 if event else 0.002
            p1 = 0.17 + (0.03 if event else 0.0)
            p2 = 0.14 + (0.02 if event else 0.0)
            pred_rows.append([year, day.strftime("%Y-%m-%d"), event, p1, p2])
            var_rows.append([
                day.strftime("%Y-%m-%d"), year, ret, 0.015, 0.025,
                int(-ret > 0.015), int(-ret > 0.025),
            ])
    preds = tmp_path / "frozen_model_predictions.csv"
    var = tmp_path / "historical_var_backtest.csv"
    pd.DataFrame(pred_rows, columns=[
        "validation_year", "date", "actual_downside_event",
        "xgboost_11_fixed_probability", "lstm_optuna_86_probability",
    ]).to_csv(preds, index=False)
    pd.DataFrame(var_rows, columns=[
        "date", "validation_year", "next_session_return",
        "var_95_loss_fraction", "var_99_loss_fraction",
        "var_95_exception", "var_99_exception",
    ]).to_csv(var, index=False)
    return preds, var


def test_phase5_rejects_final_holdout(tmp_path):
    preds, _ = _fixture(tmp_path, include_holdout=True)
    with pytest.raises(RiskValidationError, match="holdout"):
        load_predictions(preds)


def test_phase5_rejects_fabricated_var_flags(tmp_path):
    preds, var = _fixture(tmp_path)
    df = pd.read_csv(var)
    df.loc[1, "var_95_exception"] = 0
    df.to_csv(var, index=False)
    with pytest.raises(RiskValidationError, match="inconsistent"):
        load_var(var, load_predictions(preds))


def test_phase5_rejects_wrong_returns_and_labels(tmp_path):
    preds, var = _fixture(tmp_path)
    df = pd.read_csv(var)
    df.loc[1, "next_session_return"] = 0.001
    df.loc[1, "var_95_exception"] = 0
    df.to_csv(var, index=False)
    with pytest.raises(RiskValidationError, match="disagree"):
        load_var(var, load_predictions(preds))


def test_phase5_rejects_misaligned_dates(tmp_path):
    preds, var = _fixture(tmp_path)
    df = pd.read_csv(var)
    df.loc[1, "date"] = "2020-02-15"
    df.to_csv(var, index=False)
    with pytest.raises(RiskValidationError, match="identical dates|increasing"):
        load_var(var, load_predictions(preds))


def test_phase5_rejects_invalid_probability(tmp_path):
    preds, _ = _fixture(tmp_path)
    df = pd.read_csv(preds)
    df.loc[0, "lstm_optuna_86_probability"] = 1.5
    df.to_csv(preds, index=False)
    with pytest.raises(RiskValidationError, match="probabilities"):
        load_predictions(preds)


def test_var_exception_independence_undefined_if_zero_breaches():
    flags = np.zeros(10, dtype=int)
    years = np.full(10, 2020, dtype=int)
    result = _exception_independence(flags, years)
    assert result["p_value"] is None
    assert result["lr_independence"] is None


def test_phase5_report_and_metrics(tmp_path):
    preds, var = _fixture(tmp_path)
    yearly, reliability, cases, report = validate_development(
        preds, var, n_resamples=100, block_length=3
    )
    assert len(yearly) == 10
    assert len(reliability) == 12
    assert report["observations"] == 40
    assert report["events"] == 10
    assert report["primary_candidate"] == "xgboost_fixed"
    assert report["challenger"] == "lstm_optuna_trial86"
    assert report["var_exception_monitoring"]["95"]["exceptions"] == 10
    assert report["var_exception_monitoring"]["99"]["exceptions"] == 0
    assert report["var_exception_monitoring"]["99"]["independence"]["p_value"] is None
    assert report["paired_brier_comparison"]["bootstrap_samples"] == 100
    assert (cases.actual_downside_event.isin([0, 1])).all()


def test_phase5_saves_and_blocks_unintended_overwrite(tmp_path):
    preds, var = _fixture(tmp_path)
    out = tmp_path / "out"
    save_risk_validation(preds, var, out, n_resamples=100, block_length=3)
    assert all((out / name).is_file() for name in OUTPUT_FILES)
    with pytest.raises(RiskValidationError, match="--overwrite"):
        save_risk_validation(preds, var, out, n_resamples=100, block_length=3)


def test_phase5_cli_registered():
    args = build_parser().parse_args(["validate-risk", "--block-length", "20"])
    assert args.command == "validate-risk"
    assert args.block_length == 20
