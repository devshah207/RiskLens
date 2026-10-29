# Phase 3, Step 3.3 — Walk-forward validation of our existing models

**What this step does:** Run our *unchanged* three-feature models across five different validation years. **No new feature, new algorithm, parameter search, or 2025+ final evaluation is introduced.**

## 1. The exact confusion to resolve: Are we training on validation data?

**Never within the same fold.** Each fold imitates a model built at the start of a year, learning only from earlier years:

| Fold | Model trained on (past only) | Model evaluated on (unseen in this fold) |
|---|---|---|
| 2020 | 2016–2019 | 2020 |
| 2021 | 2016–2020 | 2021 |
| 2022 | 2016–2021 | 2022 |
| 2023 | 2016–2022 | 2023 |
| 2024 | 2016–2023 | 2024 |

**Why does 2023 move into training for the 2024 fold?** Because at the start of 2024, the 2023 outcomes have already happened. We train a **fresh model**, not reuse the model trained for 2023. This is called **expanding-window walk-forward validation**.

**Vocabulary:**
- *Fold*: One train-on-past, evaluate-on-future experiment.
- *Training*: Historical dates the algorithm learns from.
- *Validation*: Later dates used to measure how well this trained algorithm would have done. No fitting is allowed on the same fold's validation labels.
- *Final holdout*: Our reserved 2025–2026 data. We do not use it for model choices or scores.
- *Refitting*: Training the model again using an updated set of earlier dates.
- *Nonstationarity*: The behavior of market returns changes across time.

**One nuance:** We previously inspected 2023–2024 validation results in Step 3.2. Therefore, these five folds are *development research*, not a totally untouched final test. That is why 2025+ remains sealed.

## 2. The logic behind the code

We already have good feature-building and modeling functions. Instead of writing another implementation of Logistic Regression, `walk_forward.py` calls the existing ones **five times**.

The loop is:

```python
for year in (2020, 2021, 2022, 2023, 2024):
    validation_start = f"{year}-01-01"
    validation_end = f"{year + 1}-01-01"
    train, validation, audit = build_development_features(
        input_csv,
        validation_start=validation_start,
        holdout_start=validation_end,
    )
    # Then the existing compare_models() function reads these two
    # separately saved temporary feature tables and fits on train only.
```

**Important:** Each call to `build_development_features()` excludes prices from the validation end onward **before feature/target calculations**. A 2020 fold cannot use information from 2021, 2022, etc.; the last 2024 fold cannot use 2025 onward.

A one-session label is also removed at the year boundary. For example, the last 2023 prediction row is excluded from the 2024 fold's **training set** because its label would require the first 2024 close. That prevents 2024 results from entering 2024 training. Similarly, the last 2024 row is not evaluated using its next-session close in 2025.

**Past features are allowed:** An early January 2024 feature can use December 2023 prices because those prices were already known. This is not leakage.

## 3. What gets fitted and what does not?

Within **each fold**, `compare_models()` performs:

1. Load training and validation tables with strict date boundaries.
2. Compute the constant baseline from **training events / training rows**.
3. Run `Pipeline([StandardScaler(), LogisticRegression(...)])`.
4. `Pipeline.fit(X_train, y_train)` learns the **scaler's mean and standard deviation** and the Logistic Regression coefficients from training only.
5. `Pipeline.predict_proba(X_valid)` estimates probabilities without refitting anything on that validation year's labels.
6. Calculate metric values from predictions and actual outcomes **for that validation year only**.

We retain precisely the Phase 3.2 settings: `C=1`, `solver=lbfgs`, `max_iter=2000`, `class_weight=None`, L2 penalty and the original three features:
- `return_1d`: close today / close previous session − 1
- `return_5d`: close today / close five sessions ago − 1
- `volatility_20d`: variability of the latest 20 daily returns, expressed on an annualized scale

No advanced model, volatility index, or additional OHLC signal is added at this step.

## 4. What do the metrics mean?

- **ROC-AUC**: How well the model ranks actual downside-event cases above non-events; ~0.5 means little useful ranking.
- **Average Precision (AP)**: A ranking measure focused on the less frequent positive event. Its reference level depends strongly on that year's downside-event frequency.
- **Brier score**: Mean squared difference between predicted probability and the actual 0/1 outcome. **Smaller is better**.
- **Log loss**: Penalizes confident incorrect probabilities especially strongly. **Smaller is better**.
- **Positive event / downside event**: A next-session NIFTY 50 close-to-close decline of at least 1%.

We show ROC-AUC and AP **per year**. We report overall Brier and log loss across all yearly validation predictions. We intentionally do **not** calculate one pooled ROC-AUC/AP, because the fold predictions come from separately fitted models with potentially different probability scales.

## 5. Actual result with your 2016–2026 NIFTY 50 CSV

Running the code on the uploaded file produced these development-year results:

| Evaluation year | Training rows | Validation rows | Downside events | Logistic ROC-AUC | Logistic Brier | Baseline Brier |
|---|---:|---:|---:|---:|---:|---:|
| 2020 | 965 | 251 | 41 | 0.621 | 0.18765 | 0.14314 |
| 2021 | 1,217 | 247 | 29 | 0.689 | 0.10176 | 0.10395 |
| 2022 | 1,465 | 247 | 37 | 0.597 | 0.12843 | 0.12961 |
| 2023 | 1,713 | 245 | 14 | 0.538 | 0.05504 | 0.05658 |
| 2024 | 1,959 | 248 | 24 | 0.446 | 0.08892 | 0.08744 |

**Overall scores across the 1,238 yearly validation observations:**

| Metric (lower is better) | Constant baseline | Logistic Regression |
|---|---:|---:|
| Brier score | 0.10433 | 0.11268 |
| Log loss | 0.36644 | 0.39735 |

**Interpretation:** Logistic Regression ranked outcomes better in some years (especially 2021), but poorer in 2024. Its overall probability scores were worse than those of the simple baseline. So the current model is **not reliably better than a simple constant-risk estimate**. That is a legitimate negative result, not permission to tune repeatedly on one weak year.

This supports investigating changing market conditions and carefully evaluated additional features, **after** understanding the baseline. It does not imply any model will necessarily predict the next day successfully.

## 6. How to run this in your VS Code terminal

From the root of your existing RiskLens folder:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
python -m risklens.models.walk_forward_cli \
  data/processed/multiyear/nifty50_ohlc.csv \
  --output-dir data/processed/walk_forward
```

Two local output files (ignored by Git):
- `walk_forward_validation_predictions.csv`: each evaluated date, its year, actual downside event, and both model probabilities.
- `walk_forward_validation_report.json`: dates, sample sizes, event frequencies, fold scores, training-only scaler means, cross-boundary label purges and aggregated probability scores.

If the output folder already contains these files, the CLI refuses to overwrite until you deliberately pass `--overwrite`.

## 7. Testing: what we explicitly verify

Automated tests use **synthetic** rather than privately downloaded market data.

- All five evaluation years are correctly isolated and appear in chronological order.
- Training data grows over time, and no model's training range overlaps its validation year.
- Validation probabilities come from the existing, unchanged Phase 3.2 model code.
- Modifying **2025+ prices** cannot affect any fold's predictions or scores.
- Modifying **2024 prices** cannot affect the results for 2020–2023 folds.
- Cross-boundary training labels are removed; final holdout scores are never produced.
- Existing outputs are not silently overwritten.

## 8. Honest interview description

> I built an expanding-window validation procedure for financial downside-event classification. I repeatedly trained a fresh model on earlier years and evaluated it on the following year, with training-only scaling and target purges at calendar boundaries. Logistic Regression's ranking performance varied substantially across evaluation years, and its pooled probability accuracy was worse than a training-frequency baseline. That led me to investigate regime sensitivity and more informative features rather than claim strong predictive skill. The 2025+ data remained reserved for later independent evaluation.

**No live trades, causal claims, guaranteed accuracy, or final holdout assessment have been performed.**
