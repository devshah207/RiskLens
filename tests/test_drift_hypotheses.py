"""Synthetic only: no real 2025+ data are opened in these tests."""
import numpy as np
import pandas as pd
import pytest

from risklens.experiments import drift_hypotheses as ex


def _features(dates):
    rng = np.random.default_rng(100)
    h = pd.DataFrame({"date": dates})
    for col in ex.FULL:
        if col == "india_vix_level":
            h[col] = 15 + rng.random(len(dates)) * 8
        elif col == "rsi_14":
            h[col] = 30 + rng.random(len(dates)) * 45
        else:
            h[col] = rng.normal(0, .03, len(dates))
    return h


def test_variants_no_duplicate_and_feature_counts():
    ex._check_protocol()
    assert len(ex.VARIANTS) == 8
    assert [len(ex.VARIANTS[key]) for key in ex.VARIANTS] == [
        11, 10, 10, 10, 8, 14, 11, 15
    ]
    assert set(ex.VARIANTS["remove_all_three"]).isdisjoint(ex.FLAGGED)
    assert set(ex.FLAGGED).issubset(ex.VARIANTS["full_plus_regime_flags"])


def test_regime_history_strictly_causal():
    days = pd.bdate_range("2019-01-01", periods=130)
    old = _features(days)
    a = ex._lagged_regime_history(old)
    tampered = old.copy()
    tampered.loc[129, "india_vix_level"] = 1000
    b = ex._lagged_regime_history(tampered)
    # Changing a future date cannot affect previously calculated regime features.
    assert np.allclose(a.loc[:128, ex.RELATIVE], b.loc[:128, ex.RELATIVE])
    median = old["india_vix_level"].iloc[:63].median()
    assert a[ex.RELATIVE].iloc[62] == pytest.approx(
        old["india_vix_level"].iloc[62] / median - 1
    )
    contaminated = old.copy()
    contaminated.loc[129, "date"] = pd.Timestamp("2025-01-01")
    with pytest.raises(ex.DriftExperimentError, match="2025"):
        ex._lagged_regime_history(contaminated)


def test_identical_train_validation_dates_and_boundary_purge():
    dates = pd.bdate_range("2018-01-01", "2020-04-30")
    features = ex._lagged_regime_history(_features(dates))
    train = features.loc[features.date < "2020-01-01", ["date"]].copy().iloc[:-1]
    valid = features.loc[features.date >= "2020-01-01", ["date"]].copy()
    train[ex.TARGET_COLUMN] = np.arange(len(train)) % 2
    valid[ex.TARGET_COLUMN] = np.arange(len(valid)) % 2
    tr, va = ex._align_fold_views(features, train, valid, 2020)
    assert tr[["date", ex.TARGET_COLUMN]].equals(train.reset_index(drop=True))
    assert va[["date", ex.TARGET_COLUMN]].equals(valid.reset_index(drop=True))
    tr.loc[0, "date"] = pd.Timestamp("2025-01-01")
    with pytest.raises(ex.DriftExperimentError):
        ex._align_fold_views(features, tr, valid, 2020)


def test_paired_bootstrap_identical_predictions_zero_delta():
    n = 120
    p = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n).strftime("%Y-%m-%d"),
        "validation_year": 2020,
        "actual": [0, 0, 1] * 40,
        "probability": np.linspace(.02, .23, n),
    })
    result = ex._block_bootstrap_differences(p, p, repetitions=30)
    assert result["valid_auc_replicates"] == 30
    assert result["auc_delta_ci95"] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert result["brier_gain_ci95"] == pytest.approx([0.0, 0.0], abs=1e-12)
    tampered = p.copy()
    tampered.loc[0, "date"] = "2025-01-01"
    with pytest.raises(ex.DriftExperimentError, match="identical"):
        ex._block_bootstrap_differences(p, tampered, repetitions=2)


def test_synthetic_eight_variants_both_frozen_families(tmp_path, monkeypatch):
    days = pd.bdate_range("2016-01-04", "2024-12-30")
    history = _features(days)
    rng = np.random.default_rng(12)
    folds = []
    for year in ex.FOLD_YEARS:
        train = history.loc[history.date < pd.Timestamp(f"{year}-01-01"), ["date"]].copy()
        valid = history.loc[
            (history.date >= pd.Timestamp(f"{year}-01-01")) &
            (history.date < pd.Timestamp(f"{year+1}-01-01")), ["date"]
        ].copy()
        train = train.iloc[:-1].reset_index(drop=True)  # purged boundary label
        valid = valid.iloc[:-1].reset_index(drop=True)
        train[ex.TARGET_COLUMN] = rng.integers(0, 2, len(train))
        valid[ex.TARGET_COLUMN] = rng.integers(0, 2, len(valid))
        folds.append((year, train, valid))
    monkeypatch.setattr(ex, "make_folds", lambda *args, **kwargs: (history, folds))
    monkeypatch.setattr(ex, "verify_trial86", lambda path: {
        "trial_number": 86, "database_sha256": "synthetic",
    })

    class FakeXgb:
        def __init__(self, **kwargs):
            assert kwargs == ex.FIXED_XGB_PARAMS
        def fit(self, x, y):
            assert x.ndim == 2 and len(x) == len(y)
            return self
        def predict_proba(self, x):
            probabilities = np.clip(.11 + x[:, 0] * 0.2, .01, .95)
            return np.stack((1-probabilities, probabilities), axis=1)

    def fake_sequences(h, labeled, features, lookback):
        assert lookback == 45
        assert (h.date < "2025-01-01").all()
        arr = labeled[list(features)].to_numpy(dtype=np.float32)
        # Synthetic mock: repeat the per-date feature into the 45-session input;
        # the independent causal history check is tested above.
        return (
            np.repeat(arr[:, None, :], 45, axis=1),
            labeled[ex.TARGET_COLUMN].to_numpy(int),
            pd.DatetimeIndex(labeled.date),
        )

    def fake_lstm(train_x, train_y, valid_x, params, seed, device):
        assert params == ex.TRIAL86
        assert seed in {2026+y for y in ex.FOLD_YEARS}
        assert train_x.shape[-1] == valid_x.shape[-1]
        return np.clip(.13+valid_x[:, -1, 0] * .2, .01, .95)

    monkeypatch.setattr(ex, "XGBClassifier", FakeXgb)
    monkeypatch.setattr(ex, "make_sequences", fake_sequences)
    monkeypatch.setattr(ex, "_lstm_one", fake_lstm)
    dummy = [tmp_path / name for name in ("n.csv", "external.csv", "trial.sqlite")]
    for path in dummy:
        path.write_text("SYNTHETIC TEST DATA ONLY")
    preds, yearly, summary, uncertainty, manifest = ex.evaluate_dev_hypotheses(
        *dummy, models="both", device="cpu", bootstrap_samples=0,
    )
    assert len(yearly) == 2 * 8 * 5
    assert len(summary) == 2 * 8 * 3
    assert len(uncertainty) == 2 * 7 * 3
    assert preds.date.lt("2025-01-01").all()
    assert (yearly.n_features >= 8).all()
    assert manifest["holdout_status"].startswith("UNTOUCHED")
    assert all((preds.groupby(["model", "variant", "validation_year"]).size() > 0))
    out = tmp_path / "results"
    ex.save_experiment(*dummy, out, models="both", device="cpu", bootstrap_samples=0)
    assert all((out / name).is_file() for name in ex.OUTPUTS)
    with pytest.raises(ex.DriftExperimentError, match="already exists"):
        ex.save_experiment(*dummy, out, models="xgb", bootstrap_samples=0)
