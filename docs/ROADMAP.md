# RiskLens — Canonical 10-Phase Roadmap

**Project:** Explainable Financial Risk Modeling, Validation & AI Research Platform.

**Scope authority:** This ten-phase plan supersedes any abbreviated 5- or 6-phase completion plan. Do **not** renumber, collapse, or silently omit phases. A phase is complete only when its promised code, tests, data contracts and deliverables are verified.

**Execution priority (user-directed, October 2026):** Finish and verify all ten engineering phases first. Defer detailed line-by-line ML, finance, math and coding lessons until after implementation. Keep brief engineering explanations and decision records during construction. Phase 10 includes the thorough project defense.

## Phase 1 — Financial foundations, project definition and environment
Essential market vocabulary, exact target, financial instruments, data sources, reproducible Python project.
**Deliverable:** Project charter, environment and data contract. **Status:** completed.

## Phase 2 — Financial data collection and quality assurance
Historical OHLC and external inputs, timestamps, duplicates, invalid records, missingness, source provenance.
**Deliverable:** Validated source datasets and ingestion tests. **Status:** completed.

## Phase 3 — Exploratory analysis and financial feature engineering
Returns, volatility, momentum, technical features, mathematical interpretations, point-in-time availability, chronology.
**Deliverable:** Reproducible EDA and leakage-audited feature engineering. **Status:** completed.

## Phase 4 — Model training and hyperparameter optimization
Logistic Regression and historical baselines, tree ensemble, LSTM challenger, strictly chronological validation, limited historical Optuna search, hyperparameter freeze. **Do not run more trials.**
**Frozen primary candidate:** fixed 11-feature XGBoost. **Challenger:** *actual* Optuna LSTM trial #86, 45-session sequence (not older manually selected LSTM).
**Deliverable:** Frozen configurations, working aligned champion/challenger comparison, tests and reproducibility evidence. **Status:** code merged, local real-data report still to be verified.

## Phase 5 — Independent risk-model validation
Discrimination (ROC-AUC, PR-AUC), calibration, period stability, paired score differences, historical VaR coverage/independence, exceptions, limitations and failures; independent one-time **2025+** holdout after locked protocol and extension of point-in-time external data.
**Deliverable:** Audited validation report with genuinely untouched holdout results and limitations. **Status:** development diagnostics in implementation; independent holdout not opened.

## Phase 6 — Explainability, monitoring and model-risk reporting
SHAP or justified alternatives, feature contributions, stability, changes/drift, monitoring thresholds, model-risk register, audit artifacts and ongoing validation.
**Deliverable:** Trustworthy interpretability, drift monitoring and governance report. **Status:** not complete.

## Phase 7 — Financial-report RAG and AI research assistant
Annual report ingestion, parsing and chunking, evidence retrieval, embeddings, cited LLM answers, retrieval and answer quality tests, hallucination controls and security guardrails.
**Deliverable:** Working evidence-grounded AI financial research assistant, evaluated on manually verified questions. **Status:** not started.

## Phase 8 — Application, testing and integration
Streamlit app with financial risk and research pages, shared data contracts/services, end-to-end workflows, smoke tests and automated integration checks.
**Deliverable:** Usable tested integrated application. **Status:** not started.

## Phase 9 — GitHub, documentation and resume preparation
Reproducible setup, verified metrics, architecture diagrams, demos, privacy/license audit, CI and honest StockGro/Citi-specific resume bullets.
**Deliverable:** Interview-ready repository and portfolio evidence. **Status:** repository quality work already started; phase not complete.

## Phase 10 — Complete technical-interview defense
After engineering delivery: review actual code, every key ML and finance concept, math, design choices, alternatives, limitations and likely StockGro/Citi technical interview questions.
**Deliverable:** Demonstrated detailed comprehension and interview rehearsal. **Status:** postponed until implementation completion, **not removed**.

## Project-wide completion gate

- Keep single-purpose reusable modules and one maintained CLI; do not duplicate implementations or accumulate throwaway cleanup scripts.
- Write appropriate unit/integration tests with each feature; lint, tests and CI must pass before merging.
- Maintain temporal leak prevention, point-in-time source rules, immutable raw data, study provenance and audit trails.
- Treat local commands, GitHub CI, actual market data performance and independent final-holdout results as **distinct** evidence levels.
- Keep the post-2024 holdout sealed until a documented one-time evaluation protocol and complete external feature sources exist.
- Record deliverables and blockers truthfully; never claim a phase completed based solely on a commit, a preliminary result, or an HTML artifact.
