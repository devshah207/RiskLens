"""No tuning: fixed Optuna LSTM #86 sequence sensitivity checks."""
import numpy as np
import pandas as pd

from risklens.cli import build_parser
from risklens.models import lstm_explainability as sensitivity
from risklens.models.optuna_research import _lstm_one


def test_original_lstm_return_contract_preserved_with_optional_permutations():
    rng = np.random.default_rng(42)
    training = rng.normal(size=(24, 5, 3)).astype("float32")
    labels = np.array([0, 1] * 12)
    validation = rng.normal(size=(8, 5, 3)).astype("float32")
    params = {
        "hidden": 6,
        "layers": 1,
        "dropout": 0.0,
        "lr": 0.002,
        "weight_decay": 0.0001,
        "batch": 8,
        "epochs": 1,
    }
    baseline = _lstm_one(training, labels, validation, params, seed=505, device="cpu")
    baseline2, perturbed = _lstm_one(
        training, labels, validation, params, seed=505, device="cpu",
        permutation_repeats=2,
    )
    assert baseline.shape == baseline2.shape == (8,)
    assert np.allclose(baseline, baseline2)
    assert perturbed.shape == (2, 3, 8)
    assert np.isfinite(perturbed).all()


def test_lstm_sensitivity_one_fixed_training_call(tmp_path, monkeypatch):
    n = 30
    rng = np.random.default_rng(42)
    valid_y = np.array([0, 1] * 15)
    tr_x = rng.normal(size=(80, 45, 11)).astype("float32")
    va_x = rng.normal(size=(n, 45, 11)).astype("float32")
    folds = [(2024, pd.DataFrame(), pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=n)}))]
    sequences = [(2024, tr_x, np.array([0, 1] * 40), va_x, valid_y)]
    monkeypatch.setattr(sensitivity, "make_folds", lambda *args, **kwargs: (None, folds))
    monkeypatch.setattr(sensitivity, "sequence_folds", lambda *args: sequences)
    monkeypatch.setattr(sensitivity, "verify_trial86", lambda path: {"trial_number": 86})
    called = []
    def fake_lstm(train_x, train_y, val_x, params, seed, device, permutation_repeats):
        called.append((params, seed, device, permutation_repeats))
        base = np.full(n, 0.12)
        shifted = np.full((permutation_repeats, len(sensitivity.FEATURES), n), 0.14)
        return base, shifted
    monkeypatch.setattr(sensitivity, "_lstm_one", fake_lstm)
    nifty, external = tmp_path / "nifty.csv", tmp_path / "external.csv"
    nifty.write_text("synthetic")
    external.write_text("synthetic")
    output = tmp_path / "out"
    report = sensitivity.save_lstm_sensitivity(
        nifty, external, "local-optuna.sqlite3", output, device="cpu"
    )
    assert report["validation_year"] == 2024
    assert report["validation_sequences"] == n
    assert len(called) == 1
    assert called[0][0] == sensitivity.TRIAL86
    assert called[0][1:] == (4050, "cpu", 3)
    assert all((output / name).exists() for name in sensitivity.OUTPUTS)
    table = pd.read_csv(output / sensitivity.OUTPUTS[0])
    assert len(table) == 11
    assert table["mean_brier_increase"].notna().all()


def test_lstm_sensitivity_cli_registered():
    args = build_parser().parse_args(["explain-lstm", "--device", "mps"])
    assert args.command == "explain-lstm"
    assert args.device == "mps"
