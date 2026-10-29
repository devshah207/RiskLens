"""Resumable, chronological Optuna research. NEVER accesses 2025+ features/outcomes.

Selection: validation years 2020-2022. Diagnostic-only 2023-2024 (already
consulted in earlier work, thus NOT an independent holdout). Fixed trial
search spaces declared below; never optimize on the diagnostic years.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from risklens.features.dataset import build_development_features, TARGET_COLUMN
from risklens.evaluation.features import MULTIFACTOR_FEATURE_COLUMNS
from risklens.models.sequence_lstm import build_causal_history, make_sequences
from risklens.evaluation.splits import FINAL_HOLDOUT_START

TUNING_YEARS = (2020, 2021, 2022)
DIAGNOSTIC_YEARS = (2023, 2024)
FEATURES = MULTIFACTOR_FEATURE_COLUMNS


def make_folds(nifty_csv, external_csv, *, years=TUNING_YEARS):
    """Create identical features, dates and labels per fold; strict year boundary purge."""
    history, _ = build_causal_history(
        nifty_csv, external_csv
    )  # validates source integrity
    out = []
    for year in years:
        train, valid, audit = build_development_features(
            nifty_csv,
            validation_start=f"{year}-01-01",
            holdout_start=f"{year + 1}-01-01",
            include_ohlc_features=True,
            include_technical_features=True,
        )
        ext = history[["date", *FEATURES[-2:]]]
        train = train.merge(ext, on="date", validate="one_to_one")
        valid = valid.merge(ext, on="date", validate="one_to_one")
        train = train.dropna(subset=list(FEATURES)).reset_index(drop=True)
        valid = valid.dropna(subset=list(FEATURES)).reset_index(drop=True)
        assert audit["training_boundary_labels_purged"] == 1
        assert train.date.max() < pd.Timestamp(f"{year}-01-01") <= valid.date.min()
        assert (
            valid.date.max()
            < pd.Timestamp(f"{year + 1}-01-01")
            <= pd.Timestamp(FINAL_HOLDOUT_START)
        )
        out.append((year, train, valid))
    return history, out


def xgb_space(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 60, 650),
        max_depth=trial.suggest_int("max_depth", 1, 5),
        learning_rate=trial.suggest_float("learning_rate", 0.008, 0.20, log=True),
        min_child_weight=trial.suggest_float("min_child_weight", 2, 40, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 8.0),
        subsample=trial.suggest_float("subsample", 0.60, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.60, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.5, 80, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-5, 10, log=True),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        verbosity=0,
        scale_pos_weight=1.0,
    )


def brier(y, p):
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def fit_xgb(params, folds):
    scores = []
    per = []
    for year, tr, va in folds:
        xtr = tr[list(FEATURES)].to_numpy(float)
        xva = va[list(FEATURES)].to_numpy(float)
        ytr = tr[TARGET_COLUMN].to_numpy(int)
        yva = va[TARGET_COLUMN].to_numpy(int)
        model = XGBClassifier(**params)
        model.fit(xtr, ytr)
        p = model.predict_proba(xva)[:, 1]
        v = brier(yva, p)
        scores.extend((yva - p) ** 2)
        per.append(
            {
                "year": year,
                "brier": v,
                "rows": len(yva),
                "events": int(yva.sum()),
                "auc": float(roc_auc_score(yva, p)) if len(set(yva)) == 2 else None,
            }
        )
    return float(np.mean(scores)), per


def sequence_folds(history, labeled_folds, lookback):
    out = []
    for year, tr, va in labeled_folds:
        ax, ay, ad = make_sequences(history, tr, FEATURES, lookback)
        bx, by, bd = make_sequences(history, va, FEATURES, lookback)
        if len(bd) != len(va) or not bd.equals(pd.DatetimeIndex(va.date)):
            raise ValueError(
                "Sequence dates do not match validation; refuse uneven comparison"
            )
        assert ad.max() < bd.min()
        out.append((year, ax, ay, bx, by))
    return out


def lstm_space(trial):
    return dict(
        lookback=trial.suggest_categorical("lookback", [10, 20, 30, 45, 60]),
        hidden=trial.suggest_categorical("hidden", [12, 24, 40, 64]),
        layers=trial.suggest_int("layers", 1, 3),
        dropout=trial.suggest_float("dropout", 0.0, 0.40),
        lr=trial.suggest_float("lr", 0.0002, 0.005, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 0.03, log=True),
        batch=trial.suggest_categorical("batch", [32, 64, 128]),
        epochs=trial.suggest_int("epochs", 8, 40),
    )


def resolve_torch_device(requested="auto"):
    """Resolve Apple Metal (MPS) or CPU; never silently ignore explicit MPS."""
    import torch

    if requested not in ("auto", "mps", "cpu"):
        raise ValueError("device must be auto, mps, or cpu")
    available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "mps" and not available:
        raise RuntimeError(
            "MPS unavailable. Install Apple Silicon PyTorch, use macOS 12.3+ and verify torch.backends.mps.is_available()."
        )
    return torch.device(
        "mps" if (requested == "mps" or (requested == "auto" and available)) else "cpu"
    )


def _lstm_one(tr_x, tr_y, va_x, params, seed, device="auto"):
    import torch
    from torch import nn

    selected_device = resolve_torch_device(device)
    # CPU thread control matters only for the CPU fallback / input preparation.
    if selected_device.type == "cpu":
        torch.set_num_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    scaler = StandardScaler().fit(tr_x.reshape(-1, tr_x.shape[-1]))

    def transform(z):
        scaled = scaler.transform(z.reshape(-1, z.shape[-1])).reshape(z.shape)
        return torch.tensor(
            np.clip(scaled, -8, 8), dtype=torch.float32, device=selected_device
        )

    x = transform(tr_x)
    v = transform(va_x)
    y = torch.tensor(tr_y, dtype=torch.float32, device=selected_device)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.LSTM(
                x.shape[-1],
                params["hidden"],
                num_layers=params["layers"],
                dropout=params["dropout"] if params["layers"] > 1 else 0,
                batch_first=True,
            )
            self.head = nn.Linear(params["hidden"], 1)

        def forward(self, z):
            h, _ = self.net(z)
            return self.head(h[:, -1]).squeeze(-1)

    model = Net().to(selected_device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    loss = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    for _ in range(params["epochs"]):
        model.train()
        for idx in torch.randperm(len(x), generator=gen).split(params["batch"]):
            idx = idx.to(selected_device)
            optim.zero_grad()
            cost = loss(model(x[idx]), y[idx])
            cost.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(v)).cpu().numpy().astype(float)


def fit_lstm(params, history, folds, device="auto"):
    ss = sequence_folds(history, folds, params["lookback"])
    errors = []
    per = []
    for year, x, y, v, vy in ss:
        p = _lstm_one(x, y, v, params, seed=2026 + year, device=device)
        errors.extend((vy - p) ** 2)
        per.append(
            {
                "year": year,
                "brier": brier(vy, p),
                "rows": len(vy),
                "events": int(vy.sum()),
                "auc": float(roc_auc_score(vy, p)) if len(set(vy)) == 2 else None,
            }
        )
    return float(np.mean(errors)), per


def optimize(nifty, external, output, *, xgb_trials=200, lstm_trials=60, device="auto"):
    import optuna

    if device not in ("auto", "mps", "cpu"):
        raise ValueError("device must be auto, mps, or cpu")
    if lstm_trials > 0:
        chosen = resolve_torch_device(device)
        print(
            f"LSTM training device: {chosen}; XGBoost device: CPU (CUDA not available on Apple Silicon)",
            flush=True,
        )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history, tuning = make_folds(nifty, external, years=TUNING_YEARS)
    _, diagnostic = make_folds(nifty, external, years=DIAGNOSTIC_YEARS)
    storage = "sqlite:///" + str((output / "optuna_studies.sqlite3").resolve())
    reports = {}
    for name, num in [("xgboost", xgb_trials), ("lstm", lstm_trials)]:
        if num <= 0:
            continue
        study = optuna.create_study(
            study_name="risklens_" + name,
            storage=storage,
            load_if_exists=True,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=2026, n_startup_trials=20),
        )

        def obj(trial):
            p = xgb_space(trial) if name == "xgboost" else lstm_space(trial)
            score, per = (
                fit_xgb(p, tuning)
                if name == "xgboost"
                else fit_lstm(p, history, tuning, device=device)
            )
            trial.set_user_attr("tuning_years", list(TUNING_YEARS))
            trial.set_user_attr("tuning_per_year", per)
            return score

        study.optimize(
            obj, n_trials=num, show_progress_bar=False, catch=(RuntimeError,)
        )
        best = study.best_trial
        params = {**best.params}
        if name == "xgboost":
            params.update(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=42,
                n_jobs=1,
                verbosity=0,
                scale_pos_weight=1.0,
            )
            diag, diag_years = fit_xgb(params, diagnostic)
        else:
            diag, diag_years = fit_lstm(params, history, diagnostic, device=device)
        reports[name] = {
            "completed_trials": sum(
                t.state == optuna.trial.TrialState.COMPLETE for t in study.trials
            ),
            "new_trials_requested": num,
            "best_trial": best.number,
            "best_tuning_brier": best.value,
            "best_params": best.params,
            "diagnostic_2023_2024_brier": diag,
            "diagnostic_per_year": diag_years,
            "note": "2023-2024 diagnostic scores are NOT independent; prior project experiments examined these years.",
        }
        study.trials_dataframe().to_csv(output / f"{name}_trials.csv", index=False)
        print(
            name,
            "trials",
            reports[name]["completed_trials"],
            "tuning",
            round(best.value, 6),
            "diagnostic",
            round(diag, 6),
            flush=True,
        )
    result = {
        "selection_years": list(TUNING_YEARS),
        "diagnostic_years": list(DIAGNOSTIC_YEARS),
        "final_holdout": "2025+ NEVER ACCESSED",
        "objective": "pooled Brier score",
        "models": reports,
        "lstm_device_requested": device,
        "limitation": "Already examined development years; selecting the best of many trials creates optimization bias. Final results require untouched holdout.",
    }
    (output / "optuna_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def main():
    a = argparse.ArgumentParser()
    a.add_argument("nifty_csv")
    a.add_argument("external_csv")
    a.add_argument("--output-dir", required=True)
    a.add_argument("--xgb-trials", type=int, default=200)
    a.add_argument("--lstm-trials", type=int, default=60)
    a.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
        default="auto",
        help="PyTorch LSTM device; XGBoost stays on CPU",
    )
    args = a.parse_args()
    optimize(
        args.nifty_csv,
        args.external_csv,
        args.output_dir,
        xgb_trials=args.xgb_trials,
        lstm_trials=args.lstm_trials,
        device=args.device,
    )


if __name__ == "__main__":
    main()
