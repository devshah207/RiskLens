# RiskLens — Phase 3, Step 3.7: Error Analysis and Model Reliability

**Scope:** Investigate the predictions already produced in Step 3.5. **No model is trained, calibrated, tuned or changed here.** We read only `data/processed/xgboost_comparison/xgboost_walk_forward_predictions.csv`, which was generated using the development-only 2020–2024 validation years. The final 2025+ holdout remains sealed.

## 1. What are we trying to learn?

RiskLens predicts the chance that the **NIFTY 50 price index** (a market benchmark representing 50 major listed companies) will decline at least **1% at the next observed session's close**. A **trading session** is an open market day, not simply a calendar day. A **prediction probability** of 0.12 is the model's estimated 12% chance of that event. It is **not** a prediction that the index will decline by exactly 12%.

At Phase 3.5, fixed-parameter XGBoost had somewhat better overall **Brier score** and **log loss** than the historical baseline, but ranking performance differed across years. These measures alone do not answer whether the model consistently estimates the *right amount of risk*, whether individual years are problematic, or which historical days warrant case reviews. This diagnostic step answers those questions without changing the models.

Vocabulary:

- **Observed event frequency:** fraction of dates where the following session actually declined at least 1%. If 20 of 100 historical predictions were followed by such a decline, it is 20%.
- **Predicted event probability:** model's numerical estimate **before** that following session's outcome occurred.
- **Calibration / reliability:** whether groups receiving roughly a given probability see events about that often. This step **measures** reliability; it does not fit a calibrator.
- **Risk underestimation:** for a group of historical predictions, average forecast probability is below the observed fraction. The reverse is aggregate overestimation. This does **not** imply every prediction in that group was wrong.
- **ROC-AUC:** how often an event date is ranked above a non-event date, in an equivalent probability interpretation. 0.5 is near random ranking; 1 is perfect.
- **Average Precision:** a ranking metric suited to examining how positive events are concentrated among high-scored observations. It is sensitive to event prevalence.
- **Brier score:** average squared difference between predicted probability and binary outcome (0 or 1); smaller is better.
- **Log loss:** penalizes assigning very low probability to events that occur and very high probability to events that do not; smaller is better.
- **Nonstationarity:** market behavior changes with time. For example, the historical frequency of declines differed substantially between 2020 and 2023.

## 2. What data does the new code use?

Input file:

`data/processed/xgboost_comparison/xgboost_walk_forward_predictions.csv`

Columns are `validation_year`, `date`, `actual_downside_event`, and the existing predictions from the baseline, 3-feature Logistic Regression, 6-feature Logistic Regression, 3-feature XGBoost, and 6-feature XGBoost.

Those probabilities are **out-of-sample within each yearly development fold**: each fold's model trained only on earlier years, then predicted that year's dates. A 2024 result came from a model trained before 2024. We do not refit anything in the diagnostics step.

The code rejects any row dated 2025 or later. It does not read raw index prices, the 2025+ holdout, or the Step 3.6 calibrated-model predictions. Do not mix Step 3.6's *different training/calibration protocol* with Step 3.5 forecasts in a direct apples-to-apples model score comparison.

## 3. Exactly what the new Python code does

Open `src/risklens/models/diagnostics.py`.

### `_load_development_predictions()` — trustworthy inputs

1. Reads the CSV as strings initially, to avoid silent changes to dates and binary labels.
2. Requires the **exact column names** from Step 3.5. It refuses unexpected columns, including any accidentally added future-price fields.
3. Parses `YYYY-MM-DD` dates, requires chronological sorting and uniqueness, and verifies the year column matches the date.
4. Refuses all prediction dates outside 2020–2024. This is an explicit holdout guard.
5. Checks that true labels are exactly 0 or 1 and each forecast probability is finite and between 0 and 1.

### `_probability_scores()` — metrics, not training

Given the true labels (`y`) and predicted probabilities (`p`), Brier score is:

`mean((p - y) ** 2)`

Log loss is:

`-mean(y * ln(p) + (1-y) * ln(1-p))`

The code clips probabilities to a tiny positive number **only when calculating log loss**, because `ln(0)` is undefined. It does not alter the original forecasts.

ROC-AUC and Average Precision are reported separately **for each year**, provided that year contains both event and non-event examples. They are not pooled across the five years because different models were fitted in each fold, potentially using different historical event rates and probability scales.

### `analyze_predictions()` — yearly stability

For each already-trained model and year, the function compares:

- Number of validation days and actual downside events.
- **Observed frequency** = number of events / days.
- **Mean prediction** = average of forecast probabilities across those days.
- **Gap** = average predicted probability − observed frequency.
- Year-specific Brier, log loss, ROC-AUC and Average Precision.

Example from the real development results, using the raw **three-feature XGBoost**:

| Validation year | Average predicted risk | Actual event frequency | ROC-AUC |
|---|---:|---:|---:|
| 2020 | 12.03% | 16.33% | 0.534 |
| 2021 | 10.97% | 11.74% | 0.703 |
| 2022 | 13.10% | 14.98% | 0.603 |
| 2023 | 7.33% | 5.71% | 0.597 |
| 2024 | 9.66% | 9.68% | 0.438 |

Interpretation: the 2024 **average** probability was close to the event fraction, but ranking performance was poor. Matching an average does not mean the model correctly identifies the riskier individual dates. The 2020 model understated the group-average event rate by about 4.31 percentage points, yet that alone does not say *why* it did so.

### Reliability bins — grouped probability checks

We choose the bins **before examining results**:

`[0%, 5%), [5%, 10%), [10%, 20%), [20%, 40%), [40%, 100%]`

The final bin includes exactly 100%. For each model, the code counts how many predictions land in each bin, how many events followed, their average predicted probability and the actual event fraction. Empty bins are kept visible and have missing means/frequencies (JSON null / blank CSV), not invented zero event rates.

**Example**: in our real data, the three-feature XGBoost model placed 589 development predictions in the 10–20% bin. Their *mean predicted risk* was 14.31%, while the *observed event fraction* was 15.28%. This is a useful check, but it cannot by itself certify calibration for future years.

Bins with fewer than 30 observations **or** fewer than 5 realized events are explicitly marked `small_sample_warning=True`. These cutoffs are descriptive caution flags, not a statistical significance test. Pooled bins can hide differences between years.

`np.searchsorted(EDGES[1:], p_all, side="right")` assigns each prediction to the predetermined percentage range. The result is clipped so probability exactly 1 belongs to the final bin.

### Error examples — reviewing unusual predictions

The file `xgboost_3feature_error_examples.csv` shows two kinds of retrospective cases (default 10 of each):

1. **Events assigned low probability:** Days where the next-session decline actually happened but the model had given it one of its *lowest* probabilities among all event observations.
2. **Non-events assigned high probability:** Days where the decline did not happen but the model gave the date one of its *highest* probabilities among non-events.

These are **not necessarily mistakes in the binary-classification sense**, because an event can correctly occur despite an estimated 5% risk, and a non-event can correctly occur despite an estimated 20% risk. They are cases worth investigating for patterns. They are not trading recommendations and are deliberately **not** used to change parameters in this step.

## 4. Why this is not model improvement or a new calibration method

No `model.fit()`, `predict_proba()`, `CalibratedClassifierCV`, feature engineering, hyperparameter search, or threshold tuning is executed. The new code only processes existing prediction tables and realized development labels. The Step 3.6 experiment already showed that an additional sigmoid calibrator did not improve scores overall; we are not secretly retrying it or selecting a different one here.

The project has iterated on development validation years several times. The final holdout must remain isolated for an eventual one-time evaluation after the modeling protocol and model choice are recorded. Even then, strong confidence about real future performance may require fresh later observations.

## 5. Run on your Mac in VS Code

Unzip the changes-only archive in the existing RiskLens project directory. The new code has no additional package dependencies beyond those from Step 3.5.

```bash
source .venv/bin/activate
python -m pytest -q
python -m risklens.models.diagnostics_cli \
  data/processed/xgboost_comparison/xgboost_walk_forward_predictions.csv \
  --output-dir data/processed/model_diagnostics
```

If the predictions CSV does not exist, first rerun the Step 3.5 CLI. Do not use the raw 2025–2026 prices as a diagnostics input.

Outputs (excluded from Git):

- `model_yearly_diagnostics.csv` — 5 models × 5 development years, with sample sizes, event rates and metrics.
- `model_probability_reliability.csv` — fixed-bin probability comparisons, including empty/small-bin caution flags.
- `xgboost_3feature_error_examples.csv` — date-level cases for retrospective investigation.
- `model_diagnostics_report.json` — source SHA-256, aggregated Brier/log loss, methods and limitations.

The command will refuse to overwrite any existing output unless you explicitly supply `--overwrite`.

## 6. What the tests prove (and do not prove)

`tests/test_diagnostics.py` uses **synthetic** predictions to verify the metrics, date guards, label checks, group counts, fixed bin boundaries including 100% probability, historical case selection, output files, CLI success/failure codes, and refusal to overwrite outputs by default.

The tests validate **our diagnostic implementation**. They do not guarantee the actual historical NSE source is authentic, that XGBoost is causally learning market behavior, that the observed improvements are statistically significant, or that its future forecasting performance will be positive.

## 7. Interview-ready summary

> After using expanding-window validation to estimate next-session downside risk from NIFTY 50 historical observations, I audited forecast reliability without retraining models. I compared the average predicted probability with observed event frequencies, grouped forecasts into fixed probability bins, and inspected year-by-year Brier/log-loss and ranking metrics. In 2024 the three-feature XGBoost model closely matched the overall downside-event rate but had ROC-AUC below 0.5, illustrating why average probability accuracy is different from correctly ranking individual days. I kept the post-2024 final holdout sealed and did not tune the model based on these diagnostics.
