# RiskLens — Interactive Project Explorer (React Flow inside premium Streamlit)

**Delivery:** Source-only add-on for the existing RiskLens premium application. Not a replacement app or an ML/RAG engine.

**Research status:** Phases 1–9 remain unchanged and merged on `main`. This is an optional portfolio enhancement. Do not silently mark Phase 10 interview mastery complete.

## What the user sees

The new sidebar page **Interactive System Explorer** includes:

- Two selectable journeys: **Market Risk Analytics** and **Financial Research Assistant**.
- A node-and-edge architecture rendered with React Flow (`@xyflow/react`), powered by local React 19 and TypeScript.
- Guided autoplay (6.5 seconds per stage), pause, previous/next, replay, stage-selection rail, pan, zoom, fit-to-map and MiniMap.
- Clickable nodes showing a **Simple** or **Technical** explanation, evidence warnings, and links to the corresponding original Python files.
- Motion-respecting behavior: when the OS requests reduced motion, autoplay does not run; navigation remains available.
- A safe static, source-linked Streamlit preview **if the TypeScript/React bundle has not yet been built**, so a missing `npm install` never breaks the existing app.
- Original premium **Market Risk** and **Financial Research** pages stay unchanged and continue to be available by Streamlit navigation.

## Architecture accuracy

### Journey A — Frozen Market Risk

Historical NIFTY OHLC → chronological information guard → 11 causal features → fixed XGBoost champion **and** Optuna LSTM #86 challenger → saved holdout diagnostics → model governance/monitoring.

- Frozen final holdout: **433 sessions and 42 actual downside events**; XGBoost remains preselected champion.
- Both frozen classifiers had **0% recall at threshold 0.50**. This finding is not hidden by the animation.
- Distinct evaluations: classifier probabilities versus historical Value at Risk, SHAP, temporal sensitivity and Population Stability Index.
- All graphical content is **architecture narrative**; this add-on performs **no inference, no rescoring, no re-tuning and no holdout re-opening**.
- When the local `final_holdout_report.json` exists and matches the frozen contract, the small evidence note is constructed from its **saved values**. Otherwise the page explicitly states that the local saved report was unavailable or unvalidated; it does **not** invent a fresh score.

### Journey B — Financial Research

Question → issuer/year/basis guard → exact XBRL source lookup → **either** an exact XBRL answer with original citation **or** original PDF retrieval (BM25/MiniLM) → optional Groq after explicit consent → answer/review/refusal and provenance.

- Exact-XBRL and PDF/Groq are **alternative branches**, not sequential mandatory LLM calls.
- The explorer **does not send a question**, read a PDF, or call Groq. It only describes the code flow. The existing Financial Research Assistant remains the working interface.
- The source-exposed historical benchmark was **18 source-supported / 6 correct refusals / 4 incorrect refusals / 6 untested**; it was **not** an independently blind-validated or human-certified evaluation.
- Citation UI in the original Financial Research page remains the place to inspect real answers and source files. The architecture page contains **no fake live financial answers**.

## How the integration is packaged

```text
app/
  Home.py                                (adds one navigation link)
  pages/3_Explore_RiskLens.py            (new Streamlit page)
  services/story_blueprints.py           (read-only story/evidence data)
src/risklens_story/
  __init__.py                            (Streamlit Components v2 registration)
  pyproject.toml                         (asset_dir metadata)
  frontend/
    package.json                         (React, React Flow, Vite, TypeScript)
    vite.config.ts                       (single ES module, local CSS)
    tsconfig.json
    src/index.tsx                         (React root / v2 render lifecycle)
    src/Explorer.tsx                      (animations, graph, controls, details)
    src/types.ts                          (typed story contract)
    src/styles.css                        (premium dark navy + mint theme)
    build/                                (created on YOUR Mac by `npm run build`)
pyproject.toml                            (Streamlit minimum 1.51; package-data)
scripts/verify_story_explorer.py          (read-only diagnostic)
scripts/setup_story_explorer.sh           (optional one-command Mac build+test)
tests/test_story_explorer.py              (contracts + no side effects)
docs/INTERACTIVE_PROJECT_EXPLORER_GUIDE_2026-11-02.md
```

Only the files listed above need to be installed. The ZIP does **not** contain source market data, PDF filings, research SQLite, Groq API keys, the freeze manifest, or any copied model artifacts.

## Installation on the original Mac repository

**Before the changes:** Stop Streamlit with Control+C, keep a clean checkout, and take a backup if your `app/Home.py` or `pyproject.toml` has additional uncommitted changes. The ZIP adds/replaces exactly those two existing files; it does not touch the underlying research modules.

```bash
cd ~/Downloads/RiskLens
source .venv/bin/activate

# Only if you want a local backup first:
cp app/Home.py ~/Downloads/RiskLens_Home_Before_Explorer.py
cp pyproject.toml ~/Downloads/RiskLens_pyproject_Before_Explorer.toml

unzip -o -q ~/Downloads/RiskLens_Interactive_ReactFlow_Explorer_2026-11-02.zip -d .

# Build React + TypeScript + React Flow locally (requires internet ONCE for npm).
cd src/risklens_story/frontend
npm install
npm run build

# Return to original RiskLens project, install the Streamlit v2 UI package.
cd ~/Downloads/RiskLens
python -m pip install -e '.[ui,dev]'

# Verify the local source-data explanation and frontend build readiness.
python scripts/verify_story_explorer.py
PYTHONPATH=.:src python -m pytest -q tests/test_story_explorer.py tests/test_phase8_*.py

# Launch the SAME premium Streamlit app.
python -m streamlit run app/Home.py
```

Open **Interactive System Explorer** in the app navigation. Try both stories. Click XBRL and see the direct branch; click PDF fallback and see the optional Groq branch. Toggle Simple ↔ Technical and open the linked source file. Try Play, Pause, Next, Replay, zoom, MiniMap and View whole map.

**Shorter installation option:** After unzipping into the original repo and activating its `.venv`, simply run `bash scripts/setup_story_explorer.sh`. This builds the same React Flow component, reinstalls the editable package and runs the focused tests. It does not push anything to GitHub or use paid RAG services.

**Node version:** A modern Node.js LTS release is needed to build the frontend. The current Streamlit template recommends Node 24 LTS; this manually configured Vite 6 project also supports Node 20/22. If npm is missing, install Node 24 LTS first. `npm install` writes your local `package-lock.json`, which should be reviewed and committed in a later approved GitHub release for deterministic npm installations.

**Why a build step is needed:** This workspace has no network access to `registry.npmjs.org`; therefore I could not fetch React Flow or generate the actual built JavaScript here. The ZIP supplies the **real React/TypeScript source code**, not a simulated or mislabeled React Flow bundle. Build and browser rendering remain to be verified on your Mac.

**If the new page says “frontend not built”:** Run `npm install && npm run build` in the frontend directory, reinstall the editable Python project, then restart Streamlit. The two existing premium pages still work regardless.

**If the page says it cannot mount:** Confirm `python -m streamlit --version` is at least 1.51, `npm run build` created exactly one `index-*.js` and one `index.css`, and `python scripts/verify_story_explorer.py` prints `READY FOR LOCAL BROWSER REVIEW`.

## Quality checks / limitations

This release has been checked locally in an isolated copy of the user's original RiskLens Python sources. Tests cover story topology, exact-XBRL vs PDF branching, frozen model limitations, source link allowlisting, absence of Groq/model invocation, safe fallback and packaging metadata. Global TypeScript transpilation checked TS/TSX **syntax**, but not npm module type resolution.

**Not verified in this environment:** Full `npm install`, `npm run build`, the compiled JavaScript size, Streamlit v2 browser lifecycle, interactive animation behavior on macOS, or production deployment. These require an actual npm build and browser test on the user's Mac. Do not mark these checks passed before seeing Mac results.

**Security:** Streamlit Components v2 JS is trusted first-party code running in the app page, not an isolated security sandbox. The React node contents are provided as text properties, not HTML, and no raw PDF passages, credentials, private file paths or external content are injected into the component's HTML/JS or bundled asset directory. Code links are constructed only from internally curated repository-relative paths. There is no network call in the frontend.

**GitHub permissions:** No push, PR or merge is part of this local delivery; separate authorization will be requested after Mac browser verification.
