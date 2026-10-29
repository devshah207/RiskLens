# Citi RMS Internship — Resume-ready Project and Interview Defense

Use only claims matching code you have actually installed, run, and can explain. Results below were measured on existing 2020–2024 development data; **not an independent 2025–2026 test**. Keep wording accurate.

## Recommended project title

**RiskLens | Financial Risk Analytics, LSTM Forecasting & Model Validation**  
*Python, PyTorch, XGBoost, scikit-learn, Pandas, risk modeling, time-series analysis, model monitoring*

## Resume version (strong and factual: 3 bullets)

- **Engineered a leakage-audited multi-factor market-risk pipeline** using 2016–2024 NIFTY 50 OHLC, India VIX and S&P 500 data; implemented 11 causal predictors, source-integrity validation, strictly earlier cross-market joins and rolling 30-session input sequences.
- **Developed and benchmarked PyTorch LSTM and XGBoost classifiers** for next-session >=1% NIFTY downside risk, using expanding-window evaluation across **1,238 development observations (2020–2024)**; best tested XGBoost reached **0.612 ROC-AUC / 0.1028 Brier**, while LSTM reached **0.583 ROC-AUC / 0.1032 Brier**.
- **Implemented risk-model validation and monitoring controls**, including train-only feature scaling, chronological early stopping, probability-score audits, reproducible test suites, offline reporting dashboard and 95%/99% historical VaR exception backtests (**22 vs. 12.38 expected 99% exceptions**).

## Short resume version (space-constrained: 2 bullets)

- Built **RiskLens**, a multi-factor financial risk modeling platform (PyTorch LSTM, XGBoost, 11 signals, 30-session sequences) with leakage-safe market-data joins and 2020–2024 expanding-window backtesting.
- Benchmarked **1,238 development predictions** (best model **0.612 ROC-AUC**) and implemented probability-risk monitoring, 95%/99% historical VaR exception testing and an offline model-validation dashboard.

## 45-second interview introduction

"I built RiskLens to understand how financial institutions develop and independently validate risk models. It combines NIFTY 50 price history with historical India VIX and completed US-session returns. My binary target is whether NIFTY falls at least 1% on the next trading session. I compared a conventional tabular XGBoost benchmark against a two-layer PyTorch LSTM trained on 30-day sequences, using five expanding chronological validation folds and train-only preprocessing. The LSTM did not beat XGBoost, and that was an important finding: advanced architectures can overfit small financial datasets. Beyond prediction, I developed historical VaR exception tests and an auditable risk-reporting dashboard. I focused on point-in-time data integrity, realistic validation and documenting model limitations, because those are essential for risk model monitoring."

## Likely Citi RMS questions and answers

**Why is LSTM appropriate?** A sequence model sees a 30-session path of returns, volatility, momentum and exogenous risk indicators. It may learn path-dependent dynamics. Suitability does not guarantee better accuracy; our actual experiment found it underperformed XGBoost.

**Why is XGBoost a good baseline for time series?** The inputs contain causal temporal features and the evaluation uses chronological folds. XGBoost is trained on tabular snapshots that summarize past movements; it is not blind to time when those features are supplied.

**How did you prevent leakage?** Training labels with outcomes in the subsequent validation year are purged, the scaler fits only the earlier training portion, early stopping uses an internal earlier period, and the S&P 500 return is always from a completed *earlier* US trading date. The 2025+ final holdout was not opened.

**Why Brier and log loss?** A risk model needs meaningful numerical probabilities, not just correct high/low rankings. Both are proper scoring rules, so they reward truthful probabilities; log loss penalizes unwarranted confidence heavily.

**What does 0.612 ROC-AUC mean?** In the pooled development predictions, the model ranked a randomly chosen true downside day above a randomly chosen non-downside day roughly 61.2% of the time. It does not imply 61.2% accuracy, profitability or a pass in all years. Pooled cross-fold comparisons can also be affected by changing score distributions.

**What is VaR, and what do the exceptions show?** 99% one-day VaR is a loss threshold intended to be exceeded roughly 1% of days under its model assumptions. Our 250-return historical quantile had 22 breaches on 1,238 development dates versus 12.38 expected. We would investigate tail underestimation and exception clustering, not claim the test alone establishes risk-model invalidity.

**What would you improve with more time?** More high-quality market history and variables (calendar-verified), model-selection discipline, volatility regime analysis, stress testing, validation of tail clustering, confidence intervals, governance around risk thresholds, and a single precommitted untouched 2025–2026 evaluation. I would not blindly tune the LSTM until it beats XGBoost on previously examined data.

## Submission checklist

- Confirm tests and run the LSTM / VaR benchmark locally.
- Commit source, documentation and a screenshot of the dashboard if permitted; **do not commit raw third-party market data**.
- Do not call the project a credit-risk model, bank regulatory capital engine or profitable trading system.
- If your GitHub repository is private, a recruiter cannot view it without access. Make it public only after reviewing source/data licensing, privacy and any restrictions; you can still list a project title and demonstrate the local dashboard.
- If you include performance metrics, label them **2020–2024 development walk-forward results**, not 2025+ out-of-sample test results.
