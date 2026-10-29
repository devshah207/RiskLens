# RiskLens — Phase 3, Step 3.5: First fixed XGBoost walk-forward experiment

**Goal:** Understand what XGBoost does, implement an intentionally small gradient-boosted tree classifier, and compare it fairly with the existing constant baseline and Logistic Regression in 2020–2024. **We do not tune models on validation or evaluate 2025–2026.**

## 1. Where we are and why this step exists

RiskLens estimates the probability of an event defined as: *After today's NIFTY 50 trading session closes, will the next observed trading session's close be at least 1% lower?*

- **NIFTY 50**: a benchmark value representing a collection of large listed Indian companies; not one company's share price.
- **OHLC**: Open (beginning), High (highest), Low (lowest), Close (end) index levels for a trading session.
- **Trading session**: a day in which the exchange operates; consecutive sessions need not be consecutive calendar days.
- **Return**: fractional change, `close_today / close_previous_session - 1`.
- **Downside event**: next-session return `<= -0.01`, which means a decrease of 1% or greater.
- **Volatility**: historical variability of returns. It is not direction and does not guarantee a negative move.
- **Feature**: a number already known when we make a forecast.
- **Label / target**: what subsequently happened (0 or 1), used to train/test a historical classifier.
- **Validation**: checking a trained model on later historical examples that were not included in its fit.
- **Holdout**: reserved final period (2025 onward), not touched during the iterative development experiments.

Earlier comparisons found Logistic Regression and a six-OHLC-feature variant worse than the constant-probability baseline overall. A *fixed* XGBoost experiment tests whether modest nonlinear behavior is better captured by trees.

## 2. Decision tree: simple intuition

A tree learns questions about numerical features:

```text
Is last-20-session historical volatility greater than a learned threshold?
  yes -> What was today's return?
  no  -> What was today's opening gap?
```

The thresholds are *learned from past training observations*. They are not rules we manually picked. The tree's output is a numerical contribution to the model's score. A shallow tree can learn interactions and differences between feature ranges that a single linear combination may miss.

**Logistic Regression vs tree:** Logistic Regression learns a weighted sum of inputs and applies a sigmoid. A tree learns threshold rules and can model nonlinear relationships. Both can overfit or underfit; neither implies future market predictability.

## 3. What does gradient boosting mean?

**Boosting** means training a sequence of small trees. Tree 2 tries to improve the combined predictions of Tree 1; Tree 3 tries to improve the combination of Trees 1 and 2, and so on. The `learning_rate` ensures each tree changes the combined score gradually.

For binary classification, XGBoost uses a logistic objective, with predictions:

```text
score = initial_log_odds + tree_1_contribution + ... + tree_80_contribution
probability = 1 / (1 + exp(-score))
```

Mathematical intuition: If the model predicts probability `p` for an actual outcome `y` (0 or 1), the gradient of binary logistic loss with respect to the raw log-odds score is `g = p - y`. The curvature is `h = p*(1-p)`.

- If an actual event is `y=1` and the model estimates `p=0.10`, then `g=-0.90`. The model needs to push the score upward for similar examples.
- If an actual non-event is `y=0` and `p=0.10`, then `g=+0.10`. The model should push its score down a little.

XGBoost finds useful tree splits and regularized leaf scores from these gradients/curvatures. Simplified leaf contribution (ignoring some other penalties):

`leaf_weight = -sum(gradients) / (sum(curvatures) + reg_lambda)`.

You **do not need to implement these derivatives manually** to use `XGBClassifier`, but understanding their meaning helps explain what boosting is doing beyond “trees correct previous errors.”

## 4. Why the fixed hyperparameters?

The initial settings were chosen **before reading XGBoost validation results**, with a preference for fewer degrees of freedom rather than a large grid search:

| Code parameter | Value | Plain-English meaning |
|---|---:|---|
| `n_estimators` | `80` | Train 80 trees, not hundreds or thousands of adaptive search trials |
| `max_depth` | `2` | Each tree has at most two levels of yes/no splitting; limits complexity |
| `learning_rate` | `0.04` | Each tree makes a small change to the running model |
| `min_child_weight` | `10` | A child node must have sufficient aggregate second-derivative (Hessian) weight; **not** literally at least ten rows |
| `gamma` | `1.0` | Require enough loss reduction before making an additional split |
| `subsample` | `0.8` | Each tree randomly uses roughly 80% of training rows |
| `colsample_bytree` | `0.9` | Each tree samples roughly 90% of available feature columns |
| `reg_lambda` | `10.0` | Penalize large leaf scores (L2 regularization) |
| `reg_alpha` | `0.0` | No additional L1 weight penalty for this baseline |
| `scale_pos_weight` | `1.0` | No class reweighting; preserving estimated probability quality is important |
| `random_state` | `42` | Reproducibility of random subsampling |
| `n_jobs` | `1` | More predictable and reproducible CPU execution |
| `tree_method` | `hist` | Efficient histogram approximation for finding split thresholds |
| `objective` | `binary:logistic` | Return probabilities for 0/1 labels |
| `eval_metric` | `logloss` | Track probabilistic classification loss within XGBoost (no external validation passed) |

**Overfitting:** fitting historical noise so well that performance deteriorates on later dates. Limited depth, minimum node weight, split penalty, random sampling and L2 penalty can reduce overfitting, but cannot eliminate it. Fixed tree count also avoids choosing a stopping point based on these validation-year labels.

**Class imbalance:** about one in ten training days are downside events, making positive examples less common. We deliberately do not force the two classes to have equal weight, oversample events, or use SMOTE. Such choices can affect probability calibration; they need an independent, controlled study.

Official references:
- https://xgboost.readthedocs.io/en/latest/python/python_api.html
- https://xgboost.readthedocs.io/en/latest/parameter.html
- https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html

## 5. Why run both 3- and 6-feature XGBoost?

Both feature configurations were already defined *before* this step:

**Original three closing-price features:**
1. `return_1d`: close-to-close change over the last one observed session.
2. `return_5d`: close-to-close change over five observed sessions.
3. `volatility_20d`: annualized sample standard deviation of the most recent 20 daily returns.

**Expanded six:** the same three, plus:
4. `intraday_range = (high - low)/open`, how widely today's index level moved.
5. `intraday_return = close/open - 1`, movement from today's open to close.
6. `opening_gap = open/previous_close - 1`, movement from yesterday's close to today's open.

All are available *after session t closes*, our chosen forecast time. We compare the two previously established feature sets under **the same** fixed tree parameters and fold dates. We do not secretly search for extra indicators.

`opening_gap` and `intraday_return` are not fully independent of one-session return: `(1+gap)*(1+intraday_return)-1 = return_1d`. Extra columns do not automatically mean extra predictive information.

## 6. What is walk-forward validation, exactly?

We simulate five points in historical time:

| Train (only earlier examples) | Evaluate (a later year) |
|---|---|
| 2016–2019 | 2020 |
| 2016–2020 | 2021 |
| 2016–2021 | 2022 |
| 2016–2022 | 2023 |
| 2016–2023 | 2024 |

Each year fits a **new** model from scratch, using all eligible earlier years. Once 2023 has passed, its labeled observations may join the *2024* model's training history. This does not mean 2023 is used to fit the model that is *being evaluated on 2023*.

**Boundary purge:** If the last 2022 prediction would require a 2023 price to know whether the downside event happened, we exclude that prediction from the 2023 fold's training data. The `build_development_features()` function already implements this and returns the number of removed boundary labels.

**Holdout isolation:** 2025–2026 is structurally checked as part of the original input file read, but its prices are excluded before feature and target construction. No 2025+ outcome is fed into fitting or scoring.

**Important caveat:** The 2020–2024 years are repeated *development* validation, and we have already looked at many of these scores. Even if an experiment improves the score, that improvement is exploratory and subject to model-selection bias. We cannot call it an untouched test result.

## 7. Read the new file, function by function

File: `src/risklens/models/xgboost_walk_forward.py`.

### `FIXED_XGB_PARAMS`

One dictionary containing **all parameters**, frozen before this step's scoring. This avoids hidden search loops and makes it easy to reproduce.

### `_fit_predict_fixed_xgb(train, validation, features)`

The core model code:

```python
x_train = train.loc[:, list(features)].to_numpy(dtype=float)
y_train = train[TARGET_COLUMN].to_numpy(dtype=int)
x_valid = validation.loc[:, list(features)].to_numpy(dtype=float)

model = XGBClassifier(**FIXED_XGB_PARAMS)
model.fit(x_train, y_train)
probabilities = model.predict_proba(x_valid)[:, 1]
```

- `.loc[:, features]`: select only authorized feature columns, *never* target or next-day closing price.
- `.to_numpy(dtype=float)`: convert numerical features to a rectangular numeric array.
- `**FIXED_XGB_PARAMS`: unpack the named parameter dictionary into `XGBClassifier()`.
- `.fit(x_train, y_train)`: learn tree splits and leaf scores from training **only**.
- `.predict_proba(x_valid)[:, 1]`: return each validation observation's estimated probability of event `1`.
- **No validation labels are passed to `.fit()`**. Validation targets are used only *after prediction* to compute scores.
- **No `StandardScaler` needed**: tree comparisons use thresholds on each feature; linear scaling doesn't change their ordering. Our Logistic Regression pipeline continues to use a scaler trained on training only.
- **No `eval_set` or early stopping**: those can become another tuning choice if supplied with the same validation period whose results we report.

`get_booster().get_score(importance_type="gain")` returns a measure of how much historical training loss is reduced by splits involving each feature. We map `f0`, `f1`, etc. back to feature names. This is **not SHAP**, not proof of causality and not guaranteed future predictive usefulness.

### `compare_xgboost_walk_forward(source_csv)`

1. Calls existing `compare_ohlc_features()` to reproduce previous *constant baseline* and *3/6-feature Logistic Regression* comparisons.
2. For each year 2020–2024, rebuilds feature/target DataFrames using existing `build_development_features` and the correct yearly training/validation boundaries.
3. Checks that the new XGBoost fold's validation dates and labels **exactly match** the earlier models.
4. Checks training rows precede validation rows and that the one-row boundary target is purged.
5. Fits one XGBoost with three features and one with six, both with the same frozen settings.
6. Calculates each fold's ROC-AUC, average precision, Brier score, log loss and fixed-50%-threshold confusion counts using our existing `_metrics()` implementation.
7. Aggregates *pooled* Brier and log loss across all evaluated dates, and exports probabilities plus a JSON report.

A pooled ROC-AUC is deliberately **not** reported: validation probabilities are produced by different yearly fitted models with different base event rates, so cross-fold rankings are not directly comparable. Examine ROC-AUC *per year* instead.

### `save_xgboost_walk_forward(source_csv, output_dir, overwrite=False)`

Writes the two private output files to a temporary directory first, and refuses to replace existing outputs unless you deliberately pass `--overwrite`.

### `xgboost_walk_forward_cli.py:main()`

Uses `argparse` to accept terminal paths, prints concise scores, returns operating-system exit code `0` on success and `1` on handled errors.

## 8. Actual NIFTY 50 development results from this step

*These are descriptive results of our fixed experiment, not claims of future predictability.*

| Model | Pooled Brier score (lower better) | Pooled log loss (lower better) |
|---|---:|---:|
| Constant training-frequency baseline | 0.10433 | 0.36644 |
| Logistic Regression, original 3 features | 0.11268 | 0.39735 |
| Logistic Regression, expanded 6 features | 0.11278 | 0.39897 |
| XGBoost, original 3 features | **0.10179** | 0.35390 |
| XGBoost, expanded 6 features | 0.10186 | **0.35363** |

**Brier score:** mean squared difference between predicted probability and the realized outcome `0` or `1`. Lower is better.

**Log loss:** penalizes incorrect and overconfident probabilities. Lower is better.

**ROC-AUC:** probability that a randomly chosen actual event receives a higher score than a randomly chosen non-event; ~0.5 indicates poor ranking.

| Validation year | XGBoost 3-feature ROC-AUC | XGBoost 6-feature ROC-AUC |
|---|---:|---:|
| 2020 | 0.534 | 0.488 |
| 2021 | 0.703 | 0.711 |
| 2022 | 0.603 | 0.619 |
| 2023 | 0.597 | 0.651 |
| 2024 | **0.438** | **0.427** |

**Interpretation:** XGBoost's pooled probability scores were modestly better than our constant baseline, but its ranking performance was worse than 0.5 in 2024. The 3-versus-6 feature difference is negligible for Brier score. Do not declare the 3-feature variant a winner on the basis of a tiny post-hoc difference, and do not optimize parameters to squeeze more from these same validation years.

## 9. Run locally with existing VS Code setup (Mac)

From the project root, after extracting the **changes-only ZIP** into the existing project:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
python -m risklens.models.xgboost_walk_forward_cli \
  data/processed/multiyear/nifty50_ohlc.csv \
  --output-dir data/processed/xgboost_comparison
```

`pyproject.toml` now includes `xgboost>=2.1,<4`. The install command downloads the compatible package for your environment. Different supported package versions may cause small numeric differences.

Output files under `data/processed/xgboost_comparison/` (excluded from Git):
- `xgboost_walk_forward_predictions.csv` — date, actual historical event and each model's probability.
- `xgboost_walk_forward_report.json` — fixed hyperparameters, fold scores, training-only audit, fitted feature importance gain, pooled metrics and limitations.

To rerun deliberately over existing outputs, add `--overwrite`. Your raw files, `.git` directory and `.venv` must not be replaced.

Commit **only the new source/test/docs and changed dependency manifest**, not generated outputs or raw market data:

```bash
git add pyproject.toml \
  src/risklens/models/xgboost_walk_forward.py \
  src/risklens/models/xgboost_walk_forward_cli.py \
  tests/test_xgboost_walk_forward.py \
  docs/phase3_step5_learning_guide.md
git commit -m "Add fixed XGBoost walk-forward comparison"
git push origin main
```

## 10. Interview-ready explanation

> I designed a fixed, conservative XGBoost comparison to test whether shallow boosted trees could model nonlinear relationships missing from Logistic Regression. I used the same development-only expanding-window folds from 2020–2024, the same 3- and 6-feature inputs previously defined, and the same outcomes. Every fold fits on strictly earlier years with no label leakage across its boundary. I avoided selecting tree counts or hyperparameters on the validation folds and kept 2025+ reserved. XGBoost modestly improved pooled probability scores over the training-frequency baseline, but its 2024 ranking was worse than random, so I do not claim reliable next-day downside prediction. The experiment remains exploratory until final protocol selection and independent assessment.

**Next planned concept, not implemented here:** whether predicted probabilities are *calibrated* (for example, among dates assigned a 10% risk, did approximately 10% actually experience the defined event?) and whether probability improvements are stable enough to justify model complexity. Do not rush to use the final holdout to settle exploratory choices.
