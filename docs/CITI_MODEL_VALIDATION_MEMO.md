# RiskLens — Model Development, Independent Validation and Monitoring Memo

**Audience:** RMS / Risk Model Development, Validation & Monitoring (Citi campus interview)  
**Status:** research prototype using development data; not an independent bank validation opinion.  
**Question:** At the close of NIFTY day t, estimate P(next-session index decline >=1%).

## Executive findings

- Five chronological development validations (2020–2024), **1,238 aligned observations and 145 downside events**; no final 2025–2026 test was opened.
- Feature groups: NIFTY return, range, momentum and realized-volatility indicators; earlier India VIX and completed prior US S&P 500 return.
- Two-layer LSTM with 30-session sequences generated a development **Brier 0.10320, ROC-AUC 0.583** (11 inputs) vs fixed XGBoost **Brier 0.10280, ROC-AUC 0.612** (11 inputs); constant within-year baseline pooled Brier **0.10462**. Lower Brier is better.
- LSTM without external inputs had Brier **0.10903**. This *association* supports investigating external features; it is NOT proof of causal influence or robust out-of-sample predictive lift.
- Historical simulation VaR with 250 prior daily returns had **67/1,238 exceptions for the 95% level** (expected 61.9) and **22/1,238 at 99%** (expected 12.38). The simple two-sided exact binomial coverage test for 99% gives **p=0.0098** under its independence assumptions.
- **Decision: do not designate LSTM as champion.** Fixed XGBoost remains a candidate reference model; freeze the plan before an untouched final evaluation. There is no evidence here of a deployable or regulator-validated risk engine.

## 1. Time horizon, label and data integrity

Forecast is created *after NIFTY t closes*. Label is 1 only if NIFTY t+1 close-to-close return <= -0.01. The label cannot be known until the next observed session. For a training fold ending in December Y-1, the last potential training label whose outcome occurs in January Y is purged. For early stopping, the last inner-training label crossing its boundary is also purged.

Data pipeline checks that source NIFTY OHLC values and dates exactly match the integrated data. India VIX and S&P 500 observation dates are strictly earlier than the NIFTY prediction date, aged <=7 calendar days. Two questionable India VIX OHLC rows were quarantined in the prior integration stage. Any sequence with absent external input is rejected rather than forward-filled from the future. Any holdout date >= 2025-01-01 is rejected from model fitting or evaluation.

## 2. Sequence-model implementation

- Tensor shape: `(samples, 30 consecutive market sessions, 11 features)` (or 9 in the ablation).
- Architecture: two-layer `nn.LSTM(input_size=11, hidden_size=24, dropout=0.15)` → last hidden output → LayerNorm → Linear/ReLU/Linear → logit.
- Loss: binary cross-entropy with logits, unweighted to preserve probabilistic interpretation; optimizer AdamW (learning rate 0.001, weight decay 0.01), batch 64, clipped gradient norm 1.
- An internal chronological 85%/15% partition of the preceding training years chooses the best epoch (up to 28, patience 5) using internal loss. A new model then **refits on the full preceding-years dataset** for that many epochs; preprocessing scaler is fit again on the preceding-years data only. Validation-year outcomes are not used for stopping, tuning or fitting.
- All experiments use CPU and fixed random seeds to support reproducibility. Exact floating results can vary across versions/platforms; see JSON audit and lock locally to reproduce.

## 3. Benchmark protocol and interpretation

Models are evaluated on identical 2020–2024 dates. XGBoost probabilities come from the audited prior fixed-configuration benchmark; LSTM fits are new. We report: **Brier** for squared probability error; **log loss** for probability confidence penalty; **ROC-AUC** for ranking; and **average precision** to account for an infrequent event. A static 0.5 threshold is not useful for practical rare-event decisioning and should not be optimized on validation data.

| Model | Brier ↓ | Log loss ↓ | ROC-AUC ↑ | AP ↑ |
|---|---:|---:|---:|---:|
| XGBoost 11 | 0.102803 | 0.358147 | 0.612 | 0.172 |
| LSTM 11 | 0.103196 | 0.365019 | 0.583 | 0.171 |
| LSTM 9 | 0.109034 | 0.388117 | 0.476 | 0.128 |

**Root-cause interpretation / model risk:** An LSTM has more learnable parameters and receives strongly overlapping 30-session windows, so 1,000–2,000 independent-ish training outcomes may be insufficient to identify reliable complex temporal relationships. Higher nominal input dimensionality does not create more independent training examples. In 2024 the LSTM-11 ROC-AUC was about 0.432, showing substantial instability across years. Additional hyperparameter trials using already viewed development periods would compound selection bias. These are plausible diagnoses, not experimentally established causes.

**Historical baseline ROC-AUC note:** constant within a given year, but baseline event frequency varies by fold. Pooling these fold constants to compute ROC-AUC can misleadingly produce non-0.5 rankings; therefore do not compare a pooled baseline ROC-AUC to the classifiers.

## 4. Risk measure validation: rolling historical VaR

Historical VaR uses the empirical 5th and 1st percentiles of the **last 250 daily NIFTY returns available on day t**. The 95% and 99% risk thresholds are compared against the actual next-session loss. Results: 95% exceptions 5.41% (p=0.514), 99% exceptions 1.78% (p=0.0098) on 1,238 validation dates. Coverage assessment is descriptive: financial returns are dependent/heteroskedastic and a binomial model does not capture clustered exceptions. The 2020 pandemic and 2024 outcomes can materially influence the results. A formal risk program would also test breach clustering, stressed periods, parameter stability and independent reporting controls.

## 5. Model governance / future independent holdout

1. Freeze data provenance, target, information sets, features, folds, architectures, model weights policy, scaling, and evaluation metrics.
2. Document a *single* primary candidate, baseline and predetermined promotion decision before opening 2025+.
3. Do not tune 30-day lookback, hidden size, feature selection or epochs after seeing holdout results. Treat final holdout performance as one-time independent evaluation, not a new tuning loop.
4. Monitor realized event rate vs mean probability, Brier and log loss by time period, outliers and VaR exception counts; investigate root causes when drift or calibration deterioration occurs.
5. Add independent trading-calendar verification, full source authenticity, cost/impact analysis, portfolio exposures and banking-specific risk controls before any application claims.

## 6. Scope boundary / interview integrity

This prototype demonstrates Python development, testing, point-in-time market-data engineering, temporal deep learning, benchmark validation, model-monitoring design and risk-report generation. It is **not** a Citi-approved model, credit score, FRTB model, production deployment or investment strategy. The market-risk task was selected as a concrete technical demonstration for RMS interviews, not as a substitute for bank-credit domain experience.
