"""Smoke and routing tests for the unified CLI. No network or market data needed."""

from unittest.mock import Mock

import pytest
from risklens.cli import build_parser, main


def test_help_exposes_three_workflows():
    text = build_parser().format_help()
    assert all(name in text for name in ("demo", "compare", "tune"))


def test_invalid_tune_count_is_rejected():
    with pytest.raises(SystemExit) as exc:
        main(["tune", "a.csv", "b.csv", "--output-dir", "tmp", "--lstm-trials", "-4"])
    assert exc.value.code == 2


def test_comparison_calls_existing_library(monkeypatch):
    import risklens.models.multifactor_comparison as module

    mock = Mock(return_value={"total_validation_rows": 1238})
    monkeypatch.setattr(module, "save_multifactor_comparison", mock)
    assert main(["compare", "nifty.csv", "merged.csv", "--output-dir", "results"]) == 0
    mock.assert_called_once_with("nifty.csv", "merged.csv", "results", overwrite=False)


def test_tune_preserves_resume_and_device(monkeypatch):
    import risklens.models.optuna_research as module

    mock = Mock()
    monkeypatch.setattr(module, "optimize", mock)
    assert (
        main(
            [
                "tune",
                "n.csv",
                "e.csv",
                "--output-dir",
                "study",
                "--lstm-trials",
                "200",
                "--device",
                "mps",
            ]
        )
        == 0
    )
    mock.assert_called_once_with(
        "n.csv", "e.csv", "study", xgb_trials=0, lstm_trials=200, device="mps"
    )


def test_demo_refuses_missing_files(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "demo",
                "--nifty",
                str(tmp_path / "missing.csv"),
                "--external",
                str(tmp_path / "missing2.csv"),
            ]
        )
    assert exc.value.code == 2
