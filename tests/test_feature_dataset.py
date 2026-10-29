"""Small synthetic examples proving Phase 3 feature timing and split boundaries."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from risklens.features.dataset import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    FeatureDatasetError,
    build_development_features,
    save_development_features,
)


def sample_prices(
    tmp_path: Path, name: str = "prices.csv", change_holdout: bool = False
) -> Path:
    # Business weekdays in synthetic tests are NOT an official NSE trading calendar.
    days = pd.bdate_range("2022-10-03", "2025-01-10")
    levels = (
        100.0 + np.arange(len(days)) * 0.025 + 0.8 * np.sin(np.arange(len(days)) / 3.0)
    )
    levels = pd.Series(levels, index=days)
    # A known large decline fully inside the training range.
    levels.loc["2022-11-18"] = levels.loc["2022-11-17"] * 0.95
    # A huge boundary event (2022-12-30 -> 2023-01-02) must NOT become a training label.
    levels.loc["2023-01-02"] = levels.loc["2022-12-30"] * 0.75
    if change_holdout:
        levels.loc[levels.index >= "2025-01-01"] *= 7.0
    frame = pd.DataFrame(
        {
            "date": days,
            "open": levels.to_numpy(),
            "high": levels.to_numpy(),
            "low": levels.to_numpy(),
            "close": levels.to_numpy(),
        }
    )
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def test_output_shape_and_chronological_partitions(tmp_path):
    train, valid, report = build_development_features(sample_prices(tmp_path))
    assert list(train.columns) == ["date", *FEATURE_COLUMNS, TARGET_COLUMN]
    assert list(valid.columns) == list(train.columns)
    assert not train.empty and not valid.empty
    assert train["date"].is_monotonic_increasing
    assert valid["date"].is_monotonic_increasing
    assert train["date"].max() == pd.Timestamp("2022-12-29")
    assert valid["date"].min() == pd.Timestamp("2023-01-02")
    assert valid["date"].max() == pd.Timestamp("2024-12-31") - pd.offsets.BDay(1)
    assert (train["date"] < "2023-01-01").all()
    assert ((valid["date"] >= "2023-01-01") & (valid["date"] < "2025-01-01")).all()
    assert report["training_boundary_labels_purged"] == 1
    assert report["last_development_price_date_has_no_label"] == "2024-12-31"
    assert set(train[TARGET_COLUMN].unique()) <= {0, 1}


def test_three_feature_formulas_use_current_and_past_returns(tmp_path):
    file = sample_prices(tmp_path)
    prices = pd.read_csv(file)
    train, _, _ = build_development_features(file)
    # First 20 index levels lack 20 returns; first eligible predictor is index 20.
    assert train["date"].iloc[0] == pd.to_datetime(prices["date"].iloc[20])
    first = train.iloc[0]
    close = prices["close"].to_numpy()
    expected_r1 = close[20] / close[19] - 1
    expected_r5 = close[20] / close[15] - 1
    returns = close[1:21] / close[:20] - 1
    expected_vol = np.std(returns, ddof=1) * np.sqrt(252)
    assert first["return_1d"] == pytest.approx(expected_r1)
    assert first["return_5d"] == pytest.approx(expected_r5)
    assert first["volatility_20d"] == pytest.approx(expected_vol)
    assert train[list(FEATURE_COLUMNS)].notna().all().all()


def test_target_is_next_session_not_todays_return(tmp_path):
    file = sample_prices(tmp_path)
    train, _, _ = build_development_features(file)
    target_day = train.loc[train["date"] == pd.Timestamp("2022-11-17")].iloc[0]
    assert target_day[TARGET_COLUMN] == 1  # 5% fall observed on Nov 18.
    source = pd.read_csv(file)
    idx = source.index[source["date"] == "2022-11-17"][0]
    assert source.loc[idx + 1, "close"] / source.loc[idx, "close"] - 1 < -0.01


def test_last_train_target_is_purged_even_if_next_day_is_down(tmp_path):
    train, valid, report = build_development_features(sample_prices(tmp_path))
    assert pd.Timestamp("2022-12-30") not in set(train["date"])
    assert report["training_boundary_labels_purged"] == 1
    assert pd.Timestamp("2023-01-02") in set(valid["date"])


def test_validation_features_reuse_known_training_history(tmp_path):
    file = sample_prices(tmp_path)
    _, valid, _ = build_development_features(file)
    first_valid = valid.iloc[0]
    closes = pd.read_csv(file).set_index("date")["close"]
    days = pd.to_datetime(closes.index)
    i = list(days).index(pd.Timestamp("2023-01-02"))
    assert first_valid["return_5d"] == pytest.approx(
        closes.iloc[i] / closes.iloc[i - 5] - 1
    )
    assert pd.Timestamp(closes.index[i - 5]) < pd.Timestamp("2023-01-01")


def test_holdout_changes_cannot_change_any_development_result(tmp_path):
    original = sample_prices(tmp_path, "original.csv")
    mutated = sample_prices(tmp_path, "mutated.csv", change_holdout=True)
    a, b, report_a = build_development_features(original)
    x, y, report_b = build_development_features(mutated)
    pd.testing.assert_frame_equal(a, x)
    pd.testing.assert_frame_equal(b, y)
    for k in ("source_filename", "source_sha256"):
        report_a.pop(k)
        report_b.pop(k)
    assert report_a == report_b


def test_later_development_prices_cannot_change_earlier_features(tmp_path):
    original = sample_prices(tmp_path, "original.csv")
    changed = tmp_path / "changed.csv"
    frame = pd.read_csv(original)
    frame.loc[frame["date"] == "2024-06-20", ["open", "high", "low", "close"]] *= 1.5
    frame.to_csv(changed, index=False)
    _, a, _ = build_development_features(original)
    _, b, _ = build_development_features(changed)
    a = a.loc[a["date"] <= "2024-06-18", list(FEATURE_COLUMNS)]
    b = b.loc[b["date"] <= "2024-06-18", list(FEATURE_COLUMNS)]
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize(
    "validation_start,holdout_start",
    [
        ("2023/01/01", "2025-01-01"),
        ("2023-02-30", "2025-01-01"),
        ("2025-01-01", "2025-01-01"),
        ("2025-02-01", "2025-01-01"),
        ("2023-01-01", "2025-1-1"),
    ],
)
def test_invalid_cutoffs_fail(tmp_path, validation_start, holdout_start):
    with pytest.raises(FeatureDatasetError):
        build_development_features(
            sample_prices(tmp_path),
            validation_start=validation_start,
            holdout_start=holdout_start,
        )


def test_missing_price_file_fails(tmp_path):
    with pytest.raises(FeatureDatasetError, match="not found"):
        build_development_features(tmp_path / "missing.csv")


def test_short_development_history_fails(tmp_path):
    file = sample_prices(tmp_path)
    with pytest.raises(FeatureDatasetError, match="22 development"):
        build_development_features(
            file, validation_start="2022-10-10", holdout_start="2022-10-20"
        )


def test_no_training_or_validation_fails(tmp_path):
    file = sample_prices(tmp_path)
    with pytest.raises(FeatureDatasetError, match="Both train and validation"):
        build_development_features(file, validation_start="2022-10-12")
    with pytest.raises(FeatureDatasetError, match="Both train and validation"):
        build_development_features(
            file, validation_start="2025-01-08", holdout_start="2025-01-09"
        )


@pytest.mark.parametrize(
    "threshold,annual_sessions",
    [
        (0, 252),
        (1, 252),
        (float("nan"), 252),
        (0.01, 0),
        (0.01, True),
        (0.01, -5),
    ],
)
def test_invalid_parameters_fail(tmp_path, threshold, annual_sessions):
    with pytest.raises(FeatureDatasetError):
        build_development_features(
            sample_prices(tmp_path),
            downside_threshold=threshold,
            sessions_per_year=annual_sessions,
        )


def test_bad_ohlc_fails_before_features(tmp_path):
    file = sample_prices(tmp_path)
    frame = pd.read_csv(file)
    frame.loc[10, "high"] = frame.loc[10, "close"] - 1
    frame.to_csv(file, index=False)
    with pytest.raises(FeatureDatasetError, match="Inconsistent OHLC"):
        build_development_features(file)


def test_saves_only_model_features_label_and_report(tmp_path):
    file = sample_prices(tmp_path)
    output = tmp_path / "out"
    report = save_development_features(file, output)
    train = pd.read_csv(output / "train_features.csv")
    val = pd.read_csv(output / "validation_features.csv")
    saved = json.loads((output / "feature_dataset_report.json").read_text())
    assert saved == report
    assert list(train.columns) == ["date", *FEATURE_COLUMNS, TARGET_COLUMN]
    assert train["date"].max() < "2023-01-01"
    assert val["date"].min() >= "2023-01-01"
    assert val["date"].max() < "2025-01-01"
    assert not train.isna().any().any()
    assert not val.isna().any().any()
    with pytest.raises(FeatureDatasetError, match="already exists"):
        save_development_features(file, output)
    assert (
        save_development_features(file, output, overwrite=True)["train"]
        == report["train"]
    )


def test_command_line_pass_then_refuse_overwrite(tmp_path):
    file = sample_prices(tmp_path)
    command = [
        sys.executable,
        "-m",
        "risklens.features.cli",
        str(file),
        "--output-dir",
        str(tmp_path / "out"),
    ]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "PASS:" in first.stdout
    second = subprocess.run(command, capture_output=True, text=True, check=False)
    assert second.returncode == 1
    assert "FAIL:" in second.stdout
