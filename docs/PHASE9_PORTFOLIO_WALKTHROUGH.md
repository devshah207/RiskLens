# RiskLens — recruiter demo and screenshot plan

**Status:** Portfolio walkthrough for the premium Phase 8 application, merged to `main` as [PR #8](https://github.com/devshah207/RiskLens/pull/8). This is a **local application**, not a publicly hosted service. Do not claim a successful end-to-end Groq call from the offline-only XBRL/PDF preflight.

## A professional 2–4 minute walkthrough

### 0:00–0:25 — What problem does it solve?

> RiskLens is a financial risk research workbench with two complementary tools. The first analyzes the probability of a NIFTY 50 downside event using frozen historical machine-learning results. The second investigates company disclosures from actual XBRL filings and annual reports. They share a user interface, not a single predictive model.

Open `Home` and show the two workflows. Explain that the project is **research-oriented**, not a live trading or bank-production product.

### 0:25–1:25 — Market Risk Analytics

Open the **Overview** tab. Explain the next-session decline event (at least 1%), the **433** held-out observations and **42** actual events, and the two frozen models. The fixed XGBoost is the **preselected champion**; the Optuna LSTM trial #86 is a challenger. Point out the exact comparison table and explain ROC-AUC in one sentence: it measures how often higher-risk cases are ranked above lower-risk cases.

Then state the limitation yourself **before being challenged**: at the original 0.50 cutoff both models flagged zero events, missing all 42 declines. This is why the app displays probabilities as research, not as automatic alerts.

### 1:25–2:05 — Monitoring and explanations

Show the **Historical VaR** and **Explanations & drift** tabs. Say VaR is a separate trailing-loss-threshold method, with 30/433 breaches at 95% and 7/433 at 99%. Highlight the 2026 clustering caveat. Explain TreeSHAP as feature contribution in model log-odds and LSTM permutation sensitivity as a performance change after shuffling past sequences; neither proves cause.

### 2:05–3:20 — Financial Research Assistant

Switch to the original-documents, Groq-disabled mode. Select **Infosys** and ask:

`What was Infosys consolidated revenue from operations for FY2025, in crore?`

On the owner's Mac, this question has previously returned the source-backed XBRL answer **₹162,990 crore**, with the original NSE XML citation, concept, reporting basis, FY2025 period and INR unit metadata. The result is **not independently human-certified**. Point to the **answer card**, then expand the original citation to show why the source context matters. Exact XBRL answers use **no Groq calls**.

Explain the fallback: when exact XBRL is unavailable, existing PDF retrieval uses BM25 (word matching) and MiniLM (semantic embeddings); optional Groq composes explanations. The UI preserves `needs_review`, `abstained` and `provider_unavailable` rather than pretending every answer is certain.

**Avoid a live Groq call merely for a demo screenshot.** If it is necessary, use only the owner's account, make one explicitly approved request, and record the original source and review status without exposing the API key.

### 3:20–4:00 — Scientific honesty and closing

Show the published Phase 7 outcome table: **18** source-supported, **6** correct refusals, **4** incorrect refusals, **6** not run, from a **repeatedly exposed 34-question development benchmark**. Explain that this is not independent blind validation. Close by saying the main value is a traceable research workflow: original evidence, clear model risk and honest uncertainty.

## Screenshot capture plan (owner's Mac)

Capture **actual premium UI** screenshots, not synthetic mockups. Suggested image names for a future portfolio commit:

| File to capture | Page/area | Why it matters |
|---|---|---|
| `docs/media/01_risklens_home.png` | Premium Home, navigation visible | Establishes the two workflows |
| `docs/media/02_market_overview.png` | Market Overview comparison and zero-recall warning | Shows honest saved-study performance |
| `docs/media/03_market_probability.png` | Probability explorer timeline with original event markers | Shows read-only interactive analysis |
| `docs/media/04_market_var_drift.png` | VaR or feature drift chart | Shows risk monitoring rather than pure accuracy |
| `docs/media/05_research_xbrl_answer.png` | Research XBRL answer + source | Shows the actual answer prominently |

Capture only when pages are running from the **owner's original local files**. Crop browser tabs, username, local directory paths and API/log output. Before publishing, inspect each screenshot for secrets, sensitive filenames, licensed source material, and unintended personal information. Images are optional until the owner provides screenshots of the premium UI; this package intentionally includes none.

## Short interview answers

**Why two modules?** A financial-risk analyst measures broad-market downside risk and investigates issuer disclosures. They are complementary workflows, **not** two parts of the classifier's input pipeline.

**Why XGBoost champion if LSTM held-out AUC is higher?** Because champion designation and parameters were fixed before the one-time test. Changing it afterward would make the test a selection dataset and compromise the scientific evaluation.

**Why XBRL before PDF?** Original XBRL facts carry declared accounting tags, reporting period, company, scope and unit. PDF table retrieval can copy a correct number while attaching the wrong year, unit or concept. Exact matching is safer; it also does not cover all disclosures.

**How do you defend zero recall?** AUC and thresholded recall measure different things. An AUC around 0.67–0.68 shows moderate ranking; 0.50-threshold recall was zero. The project does not claim the classifiers are deployable alerts.

**What needs validation before production?** New independent chronological tests, external data licensing and publication timestamps, cost/latency/availability checks, calibration/threshold policy approved *before* testing, validated source auditing and a much larger blind financial-document benchmark.
