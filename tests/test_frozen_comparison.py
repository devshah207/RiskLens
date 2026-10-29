"""Frozen model selection: specification, study provenance, and alignment."""
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

from risklens.cli import build_parser
from risklens.models import frozen_comparison as frozen


def test_frozen_trial86_specification():
    assert frozen.TRIAL_NUMBER == 86
    assert frozen.TRIAL86["lookback"] == 45
    assert frozen.TRIAL86["hidden"] == 24
    assert frozen.TRIAL86["epochs"] == 24
    assert frozen.YEARS == (2020, 2021, 2022, 2023, 2024)
    assert max(frozen.YEARS) < 2025


def test_missing_study_database_rejected(tmp_path):
    with pytest.raises(frozen.FrozenComparisonError, match="does not exist"):
        frozen.verify_trial86(tmp_path / "missing.sqlite3")


def test_mismatched_trial86_record_rejected(tmp_path, monkeypatch):
    study_db = tmp_path / "study.sqlite3"
    study_db.touch()
    params = dict(frozen.TRIAL86, lookback=30)
    trial = SimpleNamespace(
        number=86,
        params=params,
        value=frozen.TRIAL_BRIER_RECORDED,
        state="complete",
    )
    study = SimpleNamespace(
        study_name="risklens_lstm", get_trials=lambda deepcopy: [trial]
    )
    fake_optuna = SimpleNamespace(
        trial=SimpleNamespace(TrialState=SimpleNamespace(COMPLETE="complete")),
        load_study=lambda **kwargs: study,
    )
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)
    with pytest.raises(frozen.FrozenComparisonError, match="lookback"):
        frozen.verify_trial86(study_db)


def _mock_folds(monkeypatch, *, mismatch=False):
    folds, sequences = [], []
    for year in frozen.YEARS:
        train = pd.DataFrame({
            "date": pd.to_datetime([f"{year-1}-01-03", f"{year-1}-01-04",
                                    f"{year-1}-01-07", f"{year-1}-01-08"]),
            frozen.TARGET_COLUMN: [0, 1, 0, 1],
        })
        valid = pd.DataFrame({
            "date": pd.to_datetime([f"{year}-02-03", f"{year}-02-04"]),
            frozen.TARGET_COLUMN: [0, 1],
        })
        for name in frozen.FEATURES:
            train[name] = [0.02, 0.01, 0.03, 0.04]
            valid[name] = [0.02, 0.04]
        folds.append((year, train, valid))
        actual = np.array([1, 0]) if mismatch and year == 2024 else np.array([0, 1])
        sequences.append((
            year,
            np.ones((4, 45, len(frozen.FEATURES))),
            np.array([0, 1, 0, 1]),
            np.ones((2, 45, len(frozen.FEATURES))),
            actual,
        ))
    def fake_make_folds(_nifty, _external, *, years):
        assert years == frozen.YEARS
        return None, folds

    monkeypatch.setattr(frozen, "make_folds", fake_make_folds)
    monkeypatch.setattr(frozen, "sequence_folds", lambda *_: sequences)
    monkeypatch.setattr(
        frozen, "verify_trial86", lambda _: {"trial_number": 86}
    )
    monkeypatch.setattr(
        frozen, "_lstm_one", lambda *args, **kwargs: np.array([0.2, 0.8])
    )

    class FakeXGB:
        def __init__(self, **parameters):
            assert parameters == frozen.FIXED_XGB_PARAMS

        def fit(self, x, y):
            assert x.shape[1] == 11 and set(y) == {0, 1}
            return self

        def predict_proba(self, x):
            return np.array([[0.7, 0.3], [0.1, 0.9]])

    monkeypatch.setattr(frozen, "XGBClassifier", FakeXGB)


def test_frozen_models_score_identical_dates(monkeypatch, tmp_path):
    _mock_folds(monkeypatch)
    nifty, external = tmp_path / "n.csv", tmp_path / "e.csv"
    nifty.touch()
    external.touch()
    frame, report = frozen.evaluate_frozen(nifty, external, device="cpu")
    assert len(frame) == 10
    assert frame.validation_year.tolist() == [
        year for year in frozen.YEARS for _ in range(2)
    ]
    assert list(frame.columns) == [
        "validation_year", "date", "actual_downside_event",
        "xgboost_11_fixed_probability", "lstm_optuna_86_probability",
    ]
    assert report["period_scores"]["tuning_2020_2022"]["rows"] == 6
    assert report["period_scores"]["repeated_diagnostic_2023_2024"]["rows"] == 4
    assert report["optuna_trial86_provenance"]["trial_number"] == 86
    assert len(report["yearly_scores"]) == 5


def test_rejects_mismatched_labels(monkeypatch, tmp_path):
    _mock_folds(monkeypatch, mismatch=True)
    nifty, external = tmp_path / "n.csv", tmp_path / "e.csv"
    nifty.touch()
    external.touch()
    with pytest.raises(frozen.FrozenComparisonError, match="dates/labels"):
        frozen.evaluate_frozen(nifty, external)


def test_existing_output_requires_overwrite(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    (output / frozen.PREDICTIONS_FILE).write_text("existing")
    with pytest.raises(frozen.FrozenComparisonError, match="already exists"):
        frozen.save_frozen_comparison("n.csv", "e.csv", output)


def test_cli_exposes_frozen_model_comparison():
    args = build_parser().parse_args(["evaluate-frozen", "--device", "mps"])
    assert args.command == "evaluate-frozen"
    assert args.device == "mps"
