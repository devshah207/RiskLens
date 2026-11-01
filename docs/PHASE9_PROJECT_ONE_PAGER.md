# RiskLens — one-page project profile

**RiskLens** | Python, pandas, XGBoost, PyTorch LSTM, Optuna, historical Value at Risk, SHAP, NSE XBRL, BM25, MiniLM, Groq, Streamlit

**Source:** [devshah207/RiskLens](https://github.com/devshah207/RiskLens)  
**Premium dashboard source:** [merged Phase 8 PR #8](https://github.com/devshah207/RiskLens/pull/8) and the `app/` folder on `main`. The app runs locally; public hosting is not claimed.

## The problem

A financial analyst must both quantify market risk and investigate what companies disclose. RiskLens combines these as **two separate workflows** in one local application. The company filings are **not features** used to train the NIFTY 50 prediction model.

## Market Risk Analytics

- **Question:** After session *t* closes, how likely is the NIFTY 50 index to close **at least 1% lower in the next observed session**?
- **Method:** Carefully ordered time-series data, 11 predeclared indicators, fixed XGBoost champion and Optuna LSTM trial #86 challenger. The final model selection and parameters were frozen before the independent 2025–September 2026 holdout.
- **Evidence:** 433 saved historical holdout predictions and 42 downside events. XGBoost ROC-AUC **0.672756** and Brier **0.084773**; LSTM ROC-AUC **0.684874** and Brier **0.084299**.
- **Critical limitation:** At the fixed 0.50 classification threshold, **both models had zero recall** (0 out of 42 positive events detected). The project does not claim a deployable alarm system or profitable trading strategy.
- **Additional research:** Historical 95% and 99% VaR, score calibration, TreeSHAP, LSTM permutation sensitivity and data-distribution monitoring. The 95% VaR backtest had 30 breaches in 433 sessions, with clustering concerns.

## Financial Research Assistant

- **Method:** Search the issuer's original FY2025 NSE **XBRL** filing for an exact match on company, reported concept, fiscal year, standalone/consolidated scope and unit.
- When the exact XML fact is unavailable, use the **existing annual-report PDF** index (BM25 plus MiniLM) and optionally authorize Groq to interpret evidence.
- The UI displays the **actual answer immediately**, then source citations, raw units, reporting dates and review status.
- **Published evaluation limits:** 18 source-supported outcomes, six correct refusals, four incorrect refusals and six unfinished cases across an exposed 34-question development benchmark. Not independently blind-validated or human-certified.

## What makes the project interview-worthy

**Engineering:** Read-only saved-evidence adapters, source-level citations, disciplined chronological splits, source freshness checks, conservative refusal behavior and explicit Groq-consent controls.  
**Research governance:** Published model-freeze evidence, honest zero-recall and VaR findings, no after-the-fact promotion of LSTM based on holdout performance.  
**Demonstration:** A premium two-section Streamlit dashboard that reuses original code and local research artifacts without requiring model retraining or paid API calls merely to view it.

## Correct scope statement

A **local, source-grounded educational research prototype**, not a production trading service, regulatory model, credit-risk system or investment recommendation. The original final fitted model weights were not saved, so the application shows previously recorded results rather than recovering the exact original fitted estimators.
