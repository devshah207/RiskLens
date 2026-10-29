# RiskLens — Phase 3.2: Constant baseline and first Logistic Regression

**Read this before modifying models or thresholds.** The purpose of the step is to establish evidence-based reference performance, not to claim a successful forecasting model.

## 1. Exact question and chronology

After session t closes, estimate the probability of a NIFTY 50 closing-level decline of **at least 1% at the next observed session**:

```
y_t = 1 if close[t+1] / close[t] - 1 <= -0.01; else 0
```

Inputs are exactly three features created by Phase 3.1: `return_1d`, `return_5d`, `volatility_20d`. Only features available through t are allowed. Training prediction dates end in 2022; validation prediction dates cover 2023–2024. No 2025+ data are fed to this program.

The program expects **the generated `train_features.csv` and `validation_features.csv`**, not the original OHLC CSV. Phase 3.1 handles label timing and one-session boundary purging. This model script checks input columns, dates, ordering, and split constraints, but cannot re-derive labels from CSVs that do not contain original close values.

## 2. Baseline: predict the training event rate everywhere

A constant-probability baseline learns a single historical number:

```
p = number of training event labels / number of training examples
```

Your training sample: 187 / 1713 = 0.1091652 (10.92%). For all 494 validation dates, predict approximately 10.92%, **even though the validation event rate was 38/494 = 7.69%**.

**Why not set the baseline to 7.69%?** That would be using validation outcomes to construct a model prediction. Using the observed validation fraction to predict that same validation set leaks future information.

Code in `models/baselines.py`: `training_rate = float(y_train.mean())`, followed by `np.full(len(valid), training_rate)`.

## 3. Logistic Regression: weighted score and sigmoid

First standardize inputs using **training-fitted** means and standard deviations:

```
z_j = (x_j - train_mean_j) / train_standard_deviation_j
```

Then build a weighted score:

```
s = intercept + w1*z1 + w2*z2 + w3*z3
p = 1 / (1 + exp(-s))
```

The weights are learned by minimizing binary cross-entropy (log loss), plus L2 regularization:

```
-log-likelihood = -(1/n) * SUM[y*ln(p) + (1-y)*ln(1-p)]
```

L2 penalizes large weights and can reduce overfitting. `C=1.0` is a **fixed, predeclared** inverse regularization strength; smaller C generally implies stronger regularization. The algorithm uses `lbfgs`, `max_iter=2000`, `class_weight=None` and the default L2 penalty. These are **not tuned against the validation results**.

**Why not use `class_weight='balanced'` automatically?** It changes the learning objective and can impair unadjusted probability calibration. We first want an honest reference for probability prediction. We may investigate alternatives later using proper time-series cross-validation.

Python code:

```python
model = Pipeline([
    ('standardize', StandardScaler()),
    ('classifier', LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000,
                                      class_weight=None, random_state=42)),
])
model.fit(X_train, y_train)                  # Fits scaler and model on TRAIN ONLY
p_logistic = model.predict_proba(X_valid)[:, 1]  # Scores validation, never fits it
```

The `[:, 1]` selects class-1 (downside-event) probability. Scikit-learn `LogisticRegression.classes_` is sorted `[0,1]` for our binary target.

## 4. Why these metrics?

All metrics in this step are calculated on **the 2023–2024 validation rows**. They are exploratory measurements on a single chronological split.

- **ROC-AUC (higher better):** probability that a randomly selected positive observation tends to receive a higher score than a negative observation. 0.5 indicates random ranking, though performance varies with sample size. Defined only if validation has both classes.
- **Average Precision (AP, higher better):** summarizes precision over recall thresholds, using scikit-learn's `average_precision_score`; **not** the same as trapezoidal PR-AUC. The constant-score model has AP approximately equal to validation prevalence.
- **Brier score (lower better):** mean squared probability error, `mean((p - y)^2)`. For example y=1, p=.2 contributes .64, whereas y=0, p=.2 contributes .04.
- **Log loss (lower better):** penalizes confidently incorrect probabilities.
- **Precision at fixed 0.50:** TP/(TP+FP) if any alerts; 0 when there are no alerts.
- **Recall at fixed 0.50:** TP/(TP+FN), fraction of actual downside events flagged.

A **50% probability threshold** is only a transparent initial diagnostic. It is usually unsuitable to regard no-alert behavior as success for rare-event risk warnings. Later threshold setting must account for alert costs and be determined without inspecting final holdout labels.

## 5. Actual development validation evidence (one fixed setup)

Inputs: user's combined NIFTY 50 history, feature-builder default cutoffs, 1713 train rows / 187 positive labels, 494 validation rows / 38 positive labels.

| Validation metric | Constant baseline | Logistic Regression |
|---|---:|---:|
| ROC-AUC (higher better) | 0.50000 | 0.50756 |
| Average Precision (higher better) | 0.07692 | 0.07831 |
| Brier score (lower better) | 0.07205 | 0.07194 |
| Log loss (lower better) | 0.27708 | 0.27584 |
| True positives at threshold 0.50 | 0 | 0 |
| False negatives at threshold 0.50 | 38 | 38 |

**Conclusion:** Only tiny improvements over the constant baseline; no demonstrated useful discrimination in this period. Do not claim strong accuracy or predictive skill. All 38 actual events are missed as alerts at threshold 0.50. This is **not** a market-beating trading system or an approved bank risk model. It has not been evaluated in any 2025+ period.

Standardized logistic coefficients from this run (associations only):

- `return_1d`: about +0.02525
- `return_5d`: about −0.27761
- `volatility_20d`: about +0.29109

Because inputs are standardized, each coefficient is associated with a one-**training-standard-deviation** change in its feature while others are held fixed. It is a *conditional log-odds* association, not the direct percentage-point change in probability, and not a causal statement. Correlated financial features can yield unstable weights.

## 6. Read the actual Python functions in order

**`_read_split(path, name, start, end)`**

1. Reads CSV data as text, checks exactly the four permitted columns plus target (date + 3 features + 1 target).
2. Parses dates in strict `YYYY-MM-DD`, rejecting duplicates and unsorted rows.
3. Rejects training dates at or after 2023-01-01 and validation dates outside 2023–2024. This is a safeguard against accidentally scoring holdout data.
4. Converts the three features to finite floating-point numbers; requires target labels to be exactly integer 0 or 1.

**`compare_models(train_csv, validation_csv)`**

1. Loads the two validated development tables.
2. Extracts features with `frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)` to guarantee the same feature order in training and validation.
3. Uses the training target mean as the constant baseline probability.
4. Calls `Pipeline.fit(X_train, y_train)` only; this fits the scaler and Logistic Regression without looking at validation targets.
5. Calls `predict_proba(X_valid)[:, 1]` to calculate future-facing probability estimates for validation dates.
6. Scores predictions against validation labels, records train-only scaler statistics and standardized coefficients, and returns prediction rows + JSON report.

**`_metrics(y_true, probabilities)`** computes the metrics above; if validation has one class, ROC-AUC and AP are reported as JSON `null` rather than fabricated values.

**`save_comparison(...)`** writes two processed artifacts and refuses to overwrite existing outputs without `--overwrite`.

**`models/cli.py`** wraps the functions in terminal commands and provides a nonzero exit code for invalid inputs.

**`tests/test_baselines.py`** uses **synthetic data only** to check training-only scaling, validation-label nonleakage, baseline definition, fixed threshold, invalid target/feature values, chronology, holdout refusal and overwrite safety.

## 7. Run it yourself on Mac / VS Code

From the existing RiskLens directory, using Python 3.12:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
# If you have not yet generated Phase 3.1 feature CSVs:
python -m risklens.features.cli data/processed/multiyear/nifty50_ohlc.csv --output-dir data/processed/features
# Then run this step:
python -m risklens.models.cli
```

If an output directory exists already, **do not delete it reflexively**. Review its files and then pass `--overwrite` intentionally if you want to regenerate them.

Outputs (private, excluded from Git):

- `data/processed/models/validation_model_predictions.csv`: validation dates, observed labels, both model probabilities.
- `data/processed/models/baseline_logistic_validation_report.json`: provenance SHA-256s, split coverage, hyperparameters, scaler details, coefficients, scores, limitations.

The model is **not** persisted as an executable artifact yet. We store its fitted parameters, but do not claim the JSON is a fully reproducible model binary or serialized pipeline.

## 8. Interview defense (no exaggeration)

> I defined a next-session downside classification target using NIFTY 50 price-index history, engineered three backward-looking price features, and purged labels that crossed chronological boundaries. I first built a training-frequency baseline, then a train-only standardized, L2-regularized Logistic Regression classifier. In 2023–2024 validation it only marginally improved ROC-AUC from 0.50 to about 0.508 and missed all 38 events at a default 50% alert threshold. I documented that negative result, kept 2025 onward untouched, and would next investigate richer financial signals and walk-forward validation without chasing the held-out outcomes.

## 9. What is NOT established and what comes later

- No calibrated probabilities demonstrated; we have only basic proper scoring rules, not a calibration plot or calibration test.
- No causal relationship or reliable next-day risk forecast demonstrated.
- No model tuning or multiple walk-forward folds yet. Avoid repeatedly changing models solely in response to the same 2023–2024 validation numbers; that is adaptive overfitting to validation.
- No 2025–2026 results examined. The final holdout remains isolated.
- No VaR backtest, profit strategy, transaction-cost accounting, or regulatory-compliance claim.

**Do not advance to more complex models until you can explain each of these points.**
