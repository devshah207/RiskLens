# RiskLens — Canonical 10-Phase Roadmap

**Project:** Explainable Financial Risk Modeling, Validation & AI Research Platform
**Status reviewed:** 30 October 2026
**Scope authority:** This **ten-phase** plan supersedes all abbreviated project outlines. No phase may be silently omitted, renamed into an unrelated deliverable, or marked complete from superficial outputs.

**Execution priority:** Build, test and verify all engineering deliverables, then complete deep line-by-line finance/ML/code education and interview defense in Phase 10. Provide concise design rationale while implementing; do not cancel the teaching requirement.

## Phase 1 — Financial foundations, project definition and environment
Define risk question, NIFTY 50 instrument, post-close forecast time, 1% next-session downside label, environment, reproducibility and project charter.

**Status: COMPLETED.** Charter, working Python package, setup, data contract and finance background documentation created.

## Phase 2 — Financial data collection and quality assurance
Multi-year NIFTY OHLC, earlier-dated India VIX and completed S&P 500 inputs; ingestion, provenance, date and numerical validation, source fingerprinting, missingness and anomaly rules.

**Status: COMPLETED for agreed research scope.** Validated 2016–September 2026 local market records and point-in-time integration. Remaining vendor-release-time authenticity / licensing caveats are documented; they are not silently assumed solved.

## Phase 3 — Exploratory analysis and financial feature engineering
Reproducible distributional EDA, returns/volatility/RSI/EMA/market features, chronology, no look-ahead, exact target and training boundary purge.

**Status: COMPLETED.** Eleven frozen causal features and labeled research dataset.

## Phase 4 — Model training and hyperparameter optimization
Historical constant/logistic baselines, XGBoost, earlier LSTM, real Optuna LSTM trials, strict expanding 2020–2024 development validation and model/parameter freeze.

**Status: COMPLETED.** Fixed 11-feature XGBoost remains preselected research champion. Real Optuna **LSTM trial #86**, 45-session lookback, remains challenger. Original Optuna SQLite provenance and development reports verified. 2020–2022 Optuna selection and repeatedly inspected 2023–2024 diagnostics are not untouched tests. **Do not launch more trials.**

## Phase 5 — Independent risk-model validation
Brier and log loss, ROC-AUC/AP, annual calibration/discrimination, historical 95%/99% VaR exception coverage and independence, paired development uncertainty, independently controlled once-only out-of-time holdout.

**Status: COMPLETED as research validation, with limitations.** Original **433-row**, **42-event** 2025–September 2026 holdout executed **once** on user Mac; existing results, saved-prediction fingerprints and VaR exception chronology reviewed. Holdout XGBoost Brier 0.084773; LSTM #86 Brier 0.084299. Both have zero recall at 0.50; 95% VaR exception independence/conditional coverage flags are disclosed. This is not an independent bank or deployment sign-off. See `docs/PHASE5_HOLDOUT_FINDINGS.md` and Phase 6 scientific review. **Do not rerun the holdout.**

## Phase 6 — Explainability, monitoring and model-risk reporting
Fixed XGBoost TreeSHAP, LSTM feature sensitivity, model-score/input-feature PSI, annual performance/calibration, three-feature root-cause audit, exploratory feature hypotheses, seed robustness, freeze evidence, model-risk register and governance decision.

**Status: SCIENTIFIC WORK COMPLETE; FINAL GOVERNANCE MERGE PENDING.** Real outputs inspected; 8-variant ablation + seven-seed testing support no model alteration. Saved-prediction PSI audit reproduced XGBoost 0.06829/LSTM 0.47019 and documented non-identical development/holdout model-instance comparator. Local freeze verifier **PASS 9/9** and **303 local tests passed**; the relevant review-branch GitHub Actions test/lint run passed. Review documents and verifier are on [draft PR #6](https://github.com/devshah207/RiskLens/pull/6); owner approval and **separate authorization to merge** are required before formal Phase 6 closure. No new scientific experiment is a prerequisite, and documented risks are not evidence of operational validation.

## Phase 7 — Financial-report RAG and AI research assistant
Legally usable genuine financial filings; versioned source provenance; page/section-aware extraction and chunking; lexical and embedding retrieval comparison; cited evidence-grounded LLM answers and abstention; injection/privacy safeguards; manually verified gold-set evaluation including numeric, multi-year, no-answer and adversarial questions.

**Status: NOT STARTED.** Must deliver a real tested, evaluated end-to-end research assistant, not a single demonstration prompt or an unevaluated vector index.

## Phase 8 — Application, testing and integration
**Streamlit** app with separate Risk Analytics and Financial Document Research pages, reusable typed back-end services, loaded frozen saved evidence, integration/smoke tests, clear limitation display and controlled access to documents.

**Status: NOT STARTED.** The existing offline HTML Citi development dashboard does **not** satisfy this phase.

## Phase 9 — GitHub, documentation and resume preparation
Reproducible setup, architecture and data-flow diagrams, CI/test/demo evidence, truthful current metrics, privacy/license and secrets audits, honest Citi and StockGro project narratives/resume bullets and usable walkthrough.

**Status: PARTIALLY STARTED, NOT COMPLETE.** Core repository, tests and governance docs exist. Further portfolio/demo/RAG-integrated packaging awaits Phases 7–8. Historical records must remain distinguishable from current frozen-study findings.

## Phase 10 — Complete technical-interview defense
After implementing Phases 1–9: explain actual modules and engineering decisions line by line, finance/ML math and alternatives, failure studies, temporal leakage, validation, sensitivity, MPS/Optuna, RAG/citations/security, limitations, and mock Citi RMS + StockGro DS/ML interviews.

**Status: NOT STARTED as the comprehensive final learning/defense.** Teaching deliberately deferred for implementation speed, **never cancelled**.

## Project-wide governance and completion rules

1. Distinguish tested code, Mac execution, source/provenance evidence, model-risk interpretation, CI and owner governance approval; no claim of production readiness from a passing test.
2. Do not alter or retune frozen XGBoost, genuine LSTM #86, 11 features, 2025–Sep 2026 holdout evidence, or Optuna SQLite source. New feature ideas/seed trials must not be promoted based on the opened holdout.
3. Preserve ignored local raw/processed data and research CSVs; do not push vendor data, credentials or fitted model objects claimed to exist when no archive was created.
4. Prefer focused reusable modules/CLIs with automated tests; avoid duplicated cleanup stages and endless small documentation loops.
5. Each **new GitHub push** requires explicit owner permission. Each **merge** requires **separate explicit permission**, even if CI passes.
6. Once the Phase 6 draft PR is approved/merged, begin Phase 7 without reopening completed experiments; a later prospective monitoring program remains a risk-register item, not a Phase 7 prerequisite.
