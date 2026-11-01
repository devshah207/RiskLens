# RiskLens — Citi / StockGro resume material and interview framing

**Prepared:** 2 November 2026. Use only statements supported by the original saved study or merged Phase 7 evidence. A resume should fit the role; do not combine every tool into one inflated sentence.

## A. Citi — Risk Management internship

### Suggested resume entry (2 bullets)

**RiskLens | Financial Risk Research Workbench** — Python, XGBoost, LSTM, historical VaR, model monitoring, Streamlit

- Built and evaluated a **frozen 11-feature NIFTY 50 downside-risk classifier**, comparing a preselected XGBoost model with an Optuna LSTM challenger on **433 chronological holdout sessions (42 decline events)**; documented AUC, average precision, Brier and log loss without post-test retuning.
- Developed a **read-only risk analytics dashboard** for saved predictions, 95%/99% historical VaR exceptions, calibration, TreeSHAP, LSTM sensitivity and feature PSI; disclosed that **both models had 0% recall at the original 0.50 cutoff** and preserved an auditable model-freeze record.

**Optional shorter version (tight space):**

> Developed RiskLens, a frozen NIFTY 50 downside-risk study (11-feature XGBoost vs Optuna LSTM; 433-session out-of-time test) with historical VaR backtesting, drift monitoring, model explanations and a read-only Streamlit dashboard.

### 30-second Citi pitch

> I built RiskLens to learn how a financial risk model should be evaluated responsibly. It predicts the probability of a next-session NIFTY 50 drop of at least 1% from information available after the previous close. I froze an XGBoost model and an Optuna LSTM before looking at the 433-session final holdout. I also evaluated independent 95% and 99% historical VaR, calibration, explanations and drift. A key finding was that both models missed all 42 declines at a fixed 0.50 threshold, so I reported that limitation instead of presenting the system as an approved risk alert.

### Likely Citi defense

**Q: What is the distinction between classifier probability and VaR?** The classifier estimates the chance of a binary next-session decline of 1% or more. Historical VaR estimates a loss threshold associated with a chosen tail probability, using trailing returns. They are different financial quantities and are calculated separately.

**Q: Why not lower the threshold after seeing 0% recall?** A new threshold chosen using holdout outcomes would make the holdout part of training/model selection. A new threshold would require a separately defined development or future prospective validation policy.

**Q: Was there evidence of VaR clustering?** Yes. The saved 95% VaR had 30 exceptions in 433 sessions (6.93% versus 5% nominal). Original independence and conditional-coverage tests raised concerns. This does not by itself establish a market cause or regulatory failure.

## B. StockGro — Data Science / Machine Learning internship

### Suggested resume entry (2 bullets)

**RiskLens | ML and Source-Grounded Financial Research** — Python, pandas, XGBoost, PyTorch, Optuna, NSE XBRL, BM25, MiniLM, Groq, Streamlit

- Implemented a leakage-aware, chronological classification study for NIFTY 50 next-session downside events using **11 market/technical/external features**, fixed XGBoost and Optuna LSTM trial #86; tracked held-out probability ranking, calibration and rare-event classification limitations.
- Integrated a **source-cited financial-document research assistant** that prefers exact NSE XBRL facts before existing PDF retrieval and optional Groq responses, exposing original financial units, reporting scope, citations and `needs_review`/refusal statuses in a local Streamlit application.

**Optional concise version:**

> Built RiskLens, combining leakage-aware time-series ML (XGBoost vs Optuna LSTM) with an original-source financial RAG assistant (NSE XBRL-first, BM25/MiniLM PDF retrieval, optional Groq) and a Streamlit evidence dashboard.

### 30-second StockGro pitch

> RiskLens combines two complementary research workflows. I developed a time-series classifier for next-session NIFTY downside events with careful chronological validation, then built an XBRL-first company research assistant. The RAG pipeline uses exact tagged financial facts when available and falls back to annual-report retrieval and Groq for harder disclosures. I created a Streamlit UI that displays the answer and its financial context, while preserving uncertainty. The important technical lessons were leakage prevention, rare-event probability metrics, financial-unit correctness and showing users when an answer needs review.

### Likely StockGro defense

**Q: How were features protected from leakage?** Features use information available at or before prediction close; external VIX and S&P references use previously available observations. The next-session close is used to create historical labels only. Chronological splits purge labels that would cross period boundaries.

**Q: What is the value of RAG?** RAG retrieves relevant original document passages before an LLM answers. Here, exact XBRL tagging handles common financial facts more reliably than PDF number matching, while PDF retrieval handles untagged descriptions. The answer retains source references and may abstain or request review.

**Q: What results can you honestly claim?** The Phase 7 published set had 18 source-supported answers, six correct future-year refusals, four incorrect refusals, and six not run, across 34 repeatedly exposed development questions. It is **not** a blind test or 100% accuracy claim.

## Do NOT claim

- Production deployment, live trading profits, business lift, regulatory approval, independently audited numerical correctness, or achieved blind RAG accuracy.
- That XGBoost and LSTM fitted weights were archived. The specification and evidence hashes were frozen; the original holdout-fitted weight objects were not saved.
- That the PDF annual reports were features of the NIFTY classifier.
- That Camelot, Groq Vision or external hosted OCR was the production-tested RAG route.
- That an Optuna search was equally applied to XGBoost. The preselected XGBoost settings were fixed; Optuna selected the LSTM trial.
- That lower recall can be fixed safely by selecting a new threshold from the opened 433-session holdout.

## Source map for interview claims

| Claim | Actual supporting artifact |
|---|---|
| One-time final scores, 433 sessions, 42 events | `data/processed/phase5_final_holdout/final_holdout_report.json` (owner-local) |
| 0.50 cutoff zero recall | Original final report and `docs/PHASE5_HOLDOUT_FINDINGS.md` |
| Frozen original model choice and 11 inputs | `docs/PHASE6_MODEL_FREEZE_MANIFEST.json` and record |
| SHAP, PSI, sensitivity, VaR | Original Phase 6 saved artifacts and governance documents |
| Phase 7 18 / 6 / 4 / 6 benchmark | `docs/phase7/README.md` on merged `main` |
| Premium Streamlit UI and tests | [Phase 8 feature branch](https://github.com/devshah207/RiskLens/tree/feature/phase8-premium-streamlit-20261102), with original backend tests; not yet merged to `main` |
