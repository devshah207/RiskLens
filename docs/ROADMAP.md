# RiskLens — Canonical 10-Phase Roadmap

**Status reviewed: 2 November 2026.** This is the authoritative current project-phase summary; historical phase-specific reports preserve the evidence and dates available when written.

| Phase | Name | Status | Evidence / boundary |
|---|---|---|---|
| 1 | Financial foundations, scope, environment | **Complete** | Target, data contract, reproducible package |
| 2 | Historical finance data and quality | **Complete for agreed research scope** | NIFTY/VIX/S&P inputs validated; provenance/licensing timing caveats remain |
| 3 | EDA and 11-feature engineering | **Complete** | Chronological feature construction and leakage controls |
| 4 | Training, chronological development and model freeze | **Complete** | Fixed XGBoost champion; Optuna LSTM trial #86 challenger; no retuning |
| 5 | One-time out-of-time risk validation | **Complete as research** | 433 held-out sessions, 42 events; 0% recall at 0.50; VaR concerns disclosed |
| 6 | Explainability, drift and governance | **Complete and merged** | [PR #6](https://github.com/devshah207/RiskLens/pull/6); original evidence freeze preserved |
| 7 | XBRL-first PDF/Groq financial research | **Implemented and merged; research limitations documented** | [PR #7](https://github.com/devshah207/RiskLens/pull/7); 34 exposed questions: 18 supported, 6 correct refusals, 4 incorrect refusals, 6 unrun |
| 8 | Premium Streamlit app and source integration | **Implemented and merged** | [PR #8](https://github.com/devshah207/RiskLens/pull/8); local-first UI; final Phase 8 CI passed |
| 9 | README, portfolio, truthful resume narratives and release audit | **Documentation release** | README + `docs/PHASE9_*.md`, source-only changes, CI checks; see release notes |
| 10 | Beginner-first technical study and Citi/StockGro interview defense | **Next phase** | Explain real code, algorithms, risk mathematics, RAG verification, limitations and mocks |

## Rules that remain in force

1. The independent 433-row 2025–September 2026 holdout was already opened once. **Do not rerun it, tune against it, change the 0.50 threshold based on it, or switch the selected champion.**
2. The preselected XGBoost is still champion; a higher holdout AUC for the frozen LSTM challenger does not authorize switching models.
3. AUC is a ranking measure; at the **original 0.50 cutoff, recall was zero** for both classifiers. Historical VaR is separate, with coverage/exception-clustering concerns.
4. The XBRL/PDF/Groq research assistant uses original company filings for separate analyst questions. Company PDFs **were never predictors** in the NIFTY classifier. Exact XBRL facts precede PDF retrieval. PDF candidate pages are not automatically certified answers.
5. The exposed Phase 7 benchmark is not an independently blind, human-audited evaluation. No 100% correctness, live production, or public deployment is claimed.
6. Original fitted estimator binaries were not archived. The Streamlit market dashboard displays **saved research evidence**, not current trading predictions.
7. Never commit API keys, original vendor data, raw report PDFs, original XBRL archives, large private SQLite research indexes, or training results not intended for publication. Live Groq requires explicit consent.
8. Keep model and Phase 7 source code fixed while preparing interview documentation and demonstrations. Phase 10 teaching is **not cancelled**.

## Next: Phase 10

The work now shifts from changing code to mastering it: beginner-first concept definitions, full mathematics and derivations, module-by-module explanations, concise and technical interview answers, failure-case defense, and structured Citi Risk Management / StockGro Data Science mock interviews.

## Technical scope and lessons retained from the original roadmap

### Phase 1 — Foundations

Defined an index-level downside-event question after the daily NIFTY close, its chronology contract, reproducibility standards, Python environment, risk vocabulary and project charter.

### Phase 2 — Market evidence

Used multi-year NIFTY OHLC history with earlier-dated India VIX and previously completed S&P 500 information. Validated numerical values, missing sessions, data coverage, financial data timing and source files. Vendor release-time and redistribution guarantees must not be inferred merely from market-date alignment.

### Phase 3 — Exploratory study

Studied returns, volatility and directional balance, RSI and EMA features, earlier-dated external information, label construction and chronology-purged development splits. The eleven selected features were frozen before the independent final period was accessed.

### Phase 4 — Model development

Compared simple baselines, XGBoost and PyTorch LSTM models on historical expanding validation windows. Optuna was used for the LSTM. The fixed 11-feature XGBoost research champion and LSTM trial #86 (45-session lookback) challenger were predeclared; repeatedly inspected development-year results are not untouched tests.

### Phase 5 — Independent financial validation

Scored one sealed 2025–September 2026 holdout **once**, saving all 433 probability pairs and labels and the evaluation report. Reported ROC-AUC, average precision, Brier, log loss, annual discrimination and calibration, and 95%/99% independent historical VaR coverage/independence evidence. Both classifiers missed all 42 decline events when thresholded at 0.50. A separate prospective test would be required to approve any new alert policy.

### Phase 6 — Explainability and monitoring

Saved fixed XGBoost TreeSHAP, LSTM temporal permutation sensitivity, score and input-feature population stability, model risk register, seed stability and investigation of EMA50 distance, India VIX and RSI14 drift. The source audit and freeze-manifest checks defend the recorded findings; drift and explanation are not causal proofs. Evidence governance was merged in [PR #6](https://github.com/devshah207/RiskLens/pull/6).

### Phase 7 — Company disclosures

Integrated genuine FY2025 annual-report PDFs and eight NSE XBRL filings for four companies. Chose exact XBRL accounting-tag/period/scope/unit matching before lexical BM25 plus MiniLM PDF retrieval, optional Groq answer generation, and conservative refusals/review labels. The published exposed 34-case result and errors belong to an educational prototype, not a reliable production financial advisor. See [Phase 7 evidence](phase7/README.md).

### Phase 8 — Application

Implemented the premium local Streamlit home, separate Market Risk and Financial Research views, interactive Plotly displays, an answer-first financial UI, read-only saved-evidence adapters and explicit Groq consent. Its existing tests, original-source offline preflight, privacy limitations and handover are documented in [the Phase 8 guide](PHASE8_PREMIUM_UI_GUIDE_2026-11-02.md) and [PR #8](https://github.com/devshah207/RiskLens/pull/8).

### Phase 9 — Portfolio release

Publish precise onboarding instructions, architecture, CI status, reproducible demo, realistic outcomes, data-license/privacy boundaries, and separate Citi and StockGro resume narratives. Actual premium screenshots are optional future enhancements and must be captured from the real running app without personal data; no pre-premium or fabricated images are used.

### Phase 10 — Interview mastery

Start from the basics of index values and returns before advancing to the real project implementation: every feature and timing assumption, objective and gradient, imbalance and calibration, walk-forward and one-time holdout, VaR backtesting, SHAP, PSI, Optuna/LSTM, XBRL periods/units, PDF retrieval, Groq verification, security, code tests and realistic interview questions. Treat the existing source code as the authority and explain each limitation instead of memorizing inflated claims.
