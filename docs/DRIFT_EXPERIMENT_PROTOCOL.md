# Research-only: possible value of EMA50, India VIX, and RSI14

**Published to `experiment/ema-vix-rsi-hypotheses` after the owner's explicit approval. NOT merged into `main`.** These Python files are an isolated development-only branch experiment.

## Why run it?

Your 2025–Sep 2026 audit found changes in NIFTY momentum and India VIX distributions. Neither PSI nor a changing mean proves a feature is unhelpful. We will experimentally test **removal** and **different causal representations**, without changing the final holdout/champion.

## Predeclared hypotheses (8 per model, NOT Optuna trials)

| ID | Features | Question |
|---|---|---|
| `full_11` | Frozen 11 | Matched reference baseline |
| `remove_ema50` | Drop EMA distance | Did EMA add useful information? |
| `remove_india_vix` | Drop absolute VIX | Did absolute implied volatility help? |
| `remove_rsi14` | Drop RSI | Did momentum indicator help? |
| `remove_all_three` | Keep other 8 | Are the three features jointly helpful? |
| `full_plus_regime_flags` | 11 plus 3 indicators | Can regime labels complement original values? |
| `vix_relative_replaces_raw` | 10 originals plus trailing-relative VIX | Could a time-relative VIX reduce level nonstationarity? |
| `full_plus_regime_context` | All 11 plus flags and relative VIX | Can richer causal context improve risk ranking? |

Indicators, known by day t close: `close[t] < EMA50[t]`; `RSI14[t] < 40`; `lagged India VIX[t] > median(last 63 observed, lagged VIX values)`; and `lagged VIX[t] / median(last 63 observed lagged VIX values) - 1`. Original India VIX is sourced strictly BEFORE day t. Indicators are causal (no labels or future market observations). These hypotheses were inspired by the already-opened holdout; results are therefore **exploratory**.

**Models:** frozen XGBoost `FIXED_XGB_PARAMS` and frozen *hyperparameters* from actual Optuna LSTM trial #86 (changing input dimension means a refitted architecture with the same hyperparams, not the exact historical trained model). No hyperparameter tuning, no new Optuna trials or changes to main-branch models.

## How do we avoid misleading improvements?

- The original **expanding chronological 2020–2024** walks, purged train/validation crossing labels, and exactly the same 11-feature eligible **prediction dates and labels** for every variant.
- LSTM uses exactly 45 sessions, trial #86 parameters and original deterministic seed scheme. External data remain strictly earlier source-date observations.
- Main benchmark is **2020–2022** (already used for original tuning), with **2023–2024 previously inspected diagnostics** shown separately. Neither is a pristine new holdout. 2025–2026 is not read or scored by these scripts.
- Compare ROC-AUC, Brier, log loss and average precision, by year and pooled, plus small-sample paired circular-block bootstrap uncertainty. Multiple comparisons and changing regimes limit interpretation.
- **No champion replacement, claim of improved final-holdout AUC, or repeated 2025–2026 testing.** Freeze a new contender for evaluation on future *newly unseen* data, e.g. 2027+, only if justified and pre-registered before seeing those outcomes.

## Run locally on the experimental branch ONLY

```bash
git fetch origin
git switch experiment/ema-vix-rsi-hypotheses
python -m pip install -e ".[dev,deep,tuning]"
python -m pytest -q tests/test_drift_hypotheses.py
python -m risklens.experiments.drift_hypotheses \
  --models both --device mps \
  --output-dir data/processed/drift_hypotheses_development
```

The report and CSVs contain model/variant/year metrics, fixed feature definitions, provenance hashes, every prediction and paired uncertainty. Real local data are required for any actual AUC claims.

## Separate approval gate

**Pushing to this experimental branch was approved. Merging into `main` was not approved. Obtain another explicit approval before any merge.** An experiment can be useful without ever merging.
