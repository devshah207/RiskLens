"""Focused synthetic-only tests. No 2025+ real data, no Optuna trials."""
import numpy as np
import pandas as pd
import pytest

from risklens.experiments import seed_stability as ex


def _data():
    n = 24
    dates = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(22)
    pred = []
    for seed_index in range(7):
        for variant in ex.VARIANT_NAMES:
            for year in ex.YEARS:
                for k, date in enumerate(dates):
                    y = int(k % 5 == 0)
                    # one candidate has uniformly better event separation
                    p = 0.07 + 0.23*y + (seed_index-3)*0.001
                    if variant == ex.VARIANT_NAMES[1]:
                        p += 0.06 if y else -0.02
                    pred.append({
                        "variant": variant, "seed_index": seed_index,
                        "seed_offset": ex.SEED_OFFSETS[seed_index],
                        "validation_year": year,
                        "date": date.strftime("%Y-%m-%d") if year == 2022 else
                                date.replace(year=year).strftime("%Y-%m-%d"),
                        "actual": y, "probability": p,
                    })
    return pd.DataFrame(pred)


def test_frozen_7_seed_design_and_only_2_variants():
    ex._check_locked_design()
    assert len(ex.SEED_OFFSETS) == 7
    assert ex.SEED_OFFSETS[0] == 0
    assert ex.VARIANT_NAMES == ("full_11", "full_plus_regime_context")
    assert ex.TRIAL86["lookback"] == 45


def test_paired_results_are_not_independent_confidence_intervals():
    rng = np.random.default_rng(5)
    periods = []
    yearly = []
    for s in range(7):
        for period in ex.PERIODS:
            periods.append(dict(seed_index=s, period=period, n=100, events=8,
                                delta_auc=0.01 + rng.normal(0, 0.003),
                                brier_gain=0.0001, delta_ap=0.005, log_loss_gain=0.002))
        for year in ex.YEARS:
            yearly.append(dict(seed_index=s, year=year, n=100, events=8,
                               delta_auc=0.01, brier_gain=0.0001,
                               delta_ap=0.005, log_loss_gain=0.002))
    summary = ex._stability_summary(pd.DataFrame(periods), pd.DataFrame(yearly))
    assert len(summary) == 8
    assert (summary.delta_auc_positive_seeds == 7).all()
    assert (summary.delta_auc_median > 0).all()
    assert not any("ci95" in col.lower() for col in summary.columns)


def test_detects_different_dates_or_holdout_dates():
    d = pd.DataFrame({
        "variant": [ex.VARIANT_NAMES[0], ex.VARIANT_NAMES[1]],
        "validation_year": [2024, 2024],
        "date": ["2024-01-02", "2024-01-03"],
        "actual": [1, 1], "probability": [0.3, 0.4],
    })
    with pytest.raises(ex.SeedStabilityError, match="different dates"):
        ex._paired_scores(d)
    d.loc[1, "date"] = "2025-01-02"
    with pytest.raises(ex.SeedStabilityError):
        ex._paired_scores(d)


def test_synthetic_end_to_end_seven_seed_pairing_and_reports(monkeypatch, tmp_path):
    from risklens.experiments import drift_hypotheses as base
    dates = pd.bdate_range("2016-01-04", "2024-12-30")
    rng = np.random.default_rng(71)
    h = pd.DataFrame({"date": dates})
    for col in base.FULL:
        h[col] = (15.0 + rng.random(len(dates)) * 2
                  if col == "india_vix_level" else
                  50.0 + rng.random(len(dates))*10
                  if col == "rsi_14" else
                  rng.normal(0.01, 0.05, len(dates)))
    folds = []
    for year in ex.YEARS:
        train = h.loc[h.date < f"{year}-01-01", ["date"]].iloc[:-1].reset_index(drop=True)
        valid = h.loc[(h.date >= f"{year}-01-01") & (h.date < f"{year+1}-01-01"), ["date"]].iloc[:-1].reset_index(drop=True)
        train[ex.TARGET_COLUMN] = (np.arange(len(train)) % 13 == 0).astype(int)
        valid[ex.TARGET_COLUMN] = (np.arange(len(valid)) % 11 == 0).astype(int)
        folds.append((year, train, valid))
    monkeypatch.setattr(ex, "make_folds", lambda *args, **kwargs: (h, folds))
    monkeypatch.setattr(ex, "verify_trial86", lambda path: {"trial_number": 86})
    called = []

    def fake_sequences(hist, labeled, columns, lookback):
        assert lookback == 45
        assert (hist.date < "2025-01-01").all()
        assert tuple(columns) in [base.VARIANTS[v] for v in ex.VARIANT_NAMES]
        # Mock creates aligned windows to test orchestration, not chronology logic.
        X = np.repeat(np.ones((len(labeled), 1, len(columns))), 45, axis=1)
        return X, labeled[ex.TARGET_COLUMN].to_numpy(dtype=int), pd.DatetimeIndex(labeled.date)

    def fake_train(tx, ty, vx, params, seed, device):
        assert params == ex.TRIAL86
        assert seed in {2026+y+o for y in ex.YEARS for o in ex.SEED_OFFSETS}
        assert device == "cpu"
        called.append((len(ty), len(vx), vx.shape[-1], seed))
        # Frozen numerical predictions independent of outcome, just mocks.
        return np.clip(0.09 + 0.0001*seed + 0.001*vx.shape[-1] +
                       np.linspace(0, 0.02, len(vx)), 0.001, 0.999)

    monkeypatch.setattr(ex, "make_sequences", fake_sequences)
    monkeypatch.setattr(ex, "_lstm_one", fake_train)
    files = [tmp_path/name for name in ("nifty.csv", "external.csv", "db.sqlite")]
    for path in files:
        path.write_text("synthetic ONLY")
    preds, annual, periods, summary, report = ex.run_seed_stability(*files, device="cpu")
    assert len(called) == 70  # 5 years * 7 seeds * 2 candidates
    assert len(annual) == 5*7
    assert len(periods) == 3*7
    assert len(summary) == 8
    assert preds.date.lt("2025-01-01").all()
    assert report["expected_training_runs"] == 70
    assert set(preds.variant) == set(ex.VARIANT_NAMES)
    assert periods.n.nunique() >= 1
    assert (annual.delta_auc.notna()).all()
    output = tmp_path/"results"
    ex.save_stability(*files, output, device="cpu")
    assert all((output/name).is_file() for name in ex.OUTPUTS)
    with pytest.raises(ex.SeedStabilityError, match="--overwrite"):
        ex.save_stability(*files, output, device="cpu")


def test_enforced_pre_2025_data(monkeypatch, tmp_path):
    for path in ("n.csv", "e.csv", "db.sqlite"):
        (tmp_path/path).write_text("synthetic")
    monkeypatch.setattr(ex, "verify_trial86", lambda path: {"trial_number": 86})
    monkeypatch.setattr(ex, "make_folds", lambda *args, **kwargs: (pd.DataFrame(), [
        (2025, pd.DataFrame(), pd.DataFrame()) for _ in range(5)
    ]))
    with pytest.raises(ex.SeedStabilityError, match="chronological development folds"):
        ex.run_seed_stability(tmp_path/"n.csv", tmp_path/"e.csv", tmp_path/"db.sqlite", device="cpu")
