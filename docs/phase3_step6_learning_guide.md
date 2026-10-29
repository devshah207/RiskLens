# RiskLens — Phase 3 Step 3.6: Time-separated probability calibration

**Teaching-first goal:** Learn what calibration is, why it differs from classification and ROC-AUC, and implement one fixed sigmoid calibration experiment without training on the year being evaluated. The 2025–2026 holdout remains sealed.

## 1. The plain-language financial problem

At the end of trading session **t**, RiskLens estimates the probability that the NIFTY 50 **price index** will close at least **1% lower** at the end of the *next observed trading session*. For example, if today's official close is 25,000, our target event occurs if the next close is 24,750 or lower.

- **NIFTY 50 index:** A benchmark computed from a basket of Indian listed companies; the index level is *not* a share price.
- **Trading session:** A day/period in which market transactions occur; the next observed session may be after a weekend or holiday.
- **Return:** Proportional price movement, `(close_next/close_today)-1`.
- **Downside event:** Our specifically defined fall of at least 1% next session, with binary historical label 1 or 0.
- **Market volatility:** Variability in returns. It is not the same as the probability of this downside event.
- **Feature:** Information known at prediction time. Here: 1-day close return, 5-day close return, trailing 20-day annualized standard deviation.
- **Validation:** Evaluating predictions on later data unseen during fitting. A final untouched *holdout* is reserved for a single later assessment.

## 2. What is calibration?

Suppose a model predicts approximately 10% risk on 100 different historical sessions. If the defined event happens about 10 times, this group of probabilities is approximately calibrated. If it happens 25 times, the model tends to understate risk for that group. If it happens 2 times, it tends to overstate risk.

**Calibration ≠ ranking:** ROC-AUC tests whether event days tend to receive higher scores than non-event days. Calibration asks whether quoted probabilities correspond to event frequencies. A model can be well ranked but badly calibrated, or the reverse.

- **Brier score:** `mean((predicted_probability - outcome)^2)`; a squared probability error; **lower is better**.
- **Log loss:** `-mean(y*ln(p) + (1-y)*ln(1-p))`; heavily penalizes confident wrong predictions; **lower is better**.
- **Reliability bin:** A group of sessions with predictions in a fixed probability range, e.g. 10–15%. Compare its average predicted probability with the observed event fraction. Small bins are noisy and depend on bin choices.

## 3. Our strict training sequence

| Validation year | Base XGBoost fits on | Calibrator fits on | Evaluated on |
|---|---|---|---|
| 2020 | 2016–2018 | 2019 | 2020 |
| 2021 | 2016–2019 | 2020 | 2021 |
| 2022 | 2016–2020 | 2021 | 2022 |
| 2023 | 2016–2021 | 2022 | 2023 |
| 2024 | 2016–2022 | 2023 | 2024 |

All labels that would cross the train/calibration or calibration/validation boundary are excluded. A validation year's labels are **never** used to fit either the XGBoost model or the calibrator for that year.

The simple historical baseline uses *all* eligible earlier-than-validation-year labels. This is fair as a chronological reference, but its training data is more recent than that of the raw base XGBoost model. **The paired experiment for the effect of calibration is raw XGBoost versus calibrated XGBoost**, which share the identical base model and evaluation rows.

**Important distinction from Step 3.5:** Earlier XGBoost models trained through year Y-1 and tested Y. In Step 3.6, base XGBoost trains through year Y-2, calibrates on Y-1, and tests Y. We cannot directly attribute differences between Step 3.5 and Step 3.6 to calibration alone because their base-model training windows differ.

## 4. How sigmoid (Platt-style) calibration works

A previously trained XGBoost model supplies historical out-of-training predictions for the calibration year.

1. First get its probability `p_raw` (e.g. `0.20`).
2. Convert it into *log-odds* using `logit(p) = ln(p/(1-p))`.
3. Fit a second, small Logistic Regression model on the **calibration year only**, using its raw log-odds as the one input and the following-session outcomes as labels.
4. Apply the learned intercept `a` and slope `b` to raw predictions for the later validation year:

   `p_calibrated = sigmoid(a + b * logit(p_raw))`, with `sigmoid(z)=1/(1+exp(-z))`.

For numerical stability, the raw probabilities are clipped to `[0.000001, 0.999999]` **only before the logit transform**. We choose `C=1` (L2 regularization) and `max_iter=2000` in advance and do not search the validation years for tuning parameters.

The sigmoid's slope is fitted freely and might be negative in an unusual calibration period. A negative slope could reverse ranking; therefore the implementation records its sign and does not claim calibration always preserves ROC-AUC. All slopes happened to be positive on the supplied real development data.

**Why not fit the calibrator on the original XGBoost training predictions?** Those are *in-sample* predictions, where XGBoost has already seen the outcomes. Their confidence patterns can be misleading. We need a distinct, later calibration year that was not used to train the base model.

## 5. Read the actual code in execution order

### `src/risklens/models/calibration.py`

- `FOLD_YEARS` is reused from the fixed 2020–2024 walk-forward design.
- `_logit()` validates probabilities, clips only for numerical safety, and converts to a column of `ln(p/(1-p))` values.
- `_fit_sigmoid_calibrator(raw_calibration_predictions, calibration_year_labels, raw_validation_predictions)` **never accepts validation labels**. It fits a regularized LogisticRegression on the preceding year's predictions and outcomes, then transforms the next year's raw scores. This is a different model from the original three-feature Logistic Regression studied in Step 3.2.
- `compare_calibration(source_csv)` iterates over five years. For each year it calls the **existing** `build_development_features()` twice: once to obtain pre-calibration base training + calibration year, and once to obtain all-pre-validation history (for the baseline) + validation year. The existing feature builder ensures all features are past-known and labels do not cross boundaries.
- `_fit_predict_fixed_xgb()` is reused without changed XGBoost settings. **One** fitted base XGBoost model scores both the calibration and validation dates. Only the calibration predictions/labels reach the sigmoid fit.
- `_metrics()` calculates per-fold Brier, log loss, ROC-AUC, AP, and fixed 50%-threshold diagnostics.
- `_reliability_bins()` produces fixed-width descriptive probability groups. Empty bins are `null`, bins with <30 observations carry warnings. Reliability bins are not themselves an optimization target.
- `_pooled_probability_scores()` pools Brier and log loss over all five held-out development years. We don't pool AUC values across independently fitted yearly models.
- `save_calibration_comparison()` refuses overwriting existing outputs unless `--overwrite` and stages the new private CSV/JSON before replacing outputs.

### `src/risklens/models/calibration_cli.py`

`argparse` reads the input CSV and optional output folder. `main()` prints PASS/FAIL and summary scores. Run with `python -m risklens.models.calibration_cli ...`. No external data or API keys are required.

### `tests/test_calibration.py`

The synthetic tests verify exact chronology, boundary purges, independent calibration, valid probability arithmetic, both-class requirements, holdout mutation invariance, no use of 2024 evaluation labels to fit the 2024 sigmoid, file export/overwrite behavior, and command-line errors.

## 6. Results on the supplied *real* dataset

The code was executed against the supplied processed NIFTY 50 file (development dates only). Exactly 1,238 evaluation observations and 145 event labels across 2020–2024.

| Pooled validation score | Brier (lower) | Log loss (lower) |
|---|---:|---:|
| Constant baseline (all historical eligible pre-Y observations) | 0.10433 | 0.36644 |
| Raw 3-feature XGBoost (trained through Y-2) | **0.10172** | **0.35358** |
| Calibrated 3-feature XGBoost (calibration year Y-1) | 0.10322 | 0.36177 |

**Interpretation:** On the existing development years, sigmoid calibration **made the pooled probability scores worse** than uncalibrated XGBoost. It was not consistently beneficial. With approximately 14–41 calibration events per yearly fit, statistical uncertainty and changing market conditions are real concerns. We should not search new calibrator settings on these same years merely to make the numbers look better.

Do not claim that calibration *can never* help, or that this experiment verifies future predictions. The 2025–2026 final holdout is still untouched.

### Example: reading reliability bins

If a group's average forecast is 12% but its realized downside rate is 14%, the model **underestimated risk** in that group by approximately 2 percentage points. That does not necessarily mean the difference is statistically meaningful; sample size and time dependence matter. The report contains counts for each fixed bin so you can spot sparse ones.

## 7. Install and run in your existing Mac / VS Code RiskLens project

Unzip changes-only ZIP into the RiskLens project root; do not replace `.git`, `.venv`, old source, raw CSVs or processed outputs.

```bash
unzip -o ~/Downloads/RiskLens_Phase3_Step3_6_Changes.zip -d .
source .venv/bin/activate
python -m pytest -q
python -m risklens.models.calibration_cli \
  data/processed/multiyear/nifty50_ohlc.csv \
  --output-dir data/processed/calibration
```

If re-running against an existing output directory, **only after reviewing existing results**, append `--overwrite`.

Expected private outputs:

- `data/processed/calibration/calibration_walk_forward_predictions.csv`
- `data/processed/calibration/calibration_walk_forward_report.json`

These outputs stay under `data/processed/`, which `.gitignore` excludes.

After successful tests:

```bash
git add src/risklens/models/calibration.py \
  src/risklens/models/calibration_cli.py \
  tests/test_calibration.py \
  docs/phase3_step6_learning_guide.md
git commit -m "Assess time-separated XGBoost probability calibration"
git push origin main
```

## 8. How to explain this in a technical interview

> I trained a fixed-complexity XGBoost classifier on historical NIFTY 50 market features and evaluated its probability calibration with strictly ordered training, calibration, and validation years. For each validation year, the previous year was reserved exclusively for learning a sigmoid correction, while the base XGBoost never saw that year's calibration labels. I evaluated both uncalibrated and calibrated predictions on the same subsequent dates. Calibration worsened pooled Brier and log-loss scores in this experiment, so I documented the negative result rather than optimizing repeatedly on already-inspected development years. I retained a sealed 2025+ final holdout for independent assessment later.

**Out of scope:** Feature engineering, threshold optimization, XGBoost tuning, model selection, VaR, backtesting and any final-holdout result. Those are separate activities.
