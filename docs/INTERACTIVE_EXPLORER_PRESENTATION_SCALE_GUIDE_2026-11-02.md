# RiskLens — Presentation-Scale Interactive Explorer

**Date:** 2 November 2026  
**Purpose:** Enlarge the EXISTING React Flow architecture explorer without changing RiskLens's frozen market models, published RAG backend, premium Market Risk page or premium Financial Research page.

## What changed

This release updates only the `Explore RiskLens` page and the React Flow presentation layer:

1. **Much larger map:** The page no longer imposes the old main-column maximum width. The graph uses a taller canvas (approximately 82% of the viewport height, up to 890 CSS pixels) and more of the available width. The outer header is compact, so the map receives more attention.
2. **Readable nodes:** Each node is approximately **298 CSS pixels wide** and at least **154 CSS pixels tall** at 100% graph zoom, with substantially larger headings, labels and iconography. Horizontal and branch spacing is expanded *only in the visualization*. The original Python story coordinates, data and calculations are unchanged.
3. **Guided camera:** The initial stage and stage changes center the chosen node at **1.13× zoom**, after React Flow has initialized its nodes. The previous whole-graph overview remains available via **View whole map**. Use **Focus stage** at any time to zoom back to the active node.
4. **Collapsible details:** **Hide details** grants the architecture canvas the entire row width; **Show details** restores the explanation panel without losing the current stage.
5. **Presentation mode:** **Presentation mode** requests real browser fullscreen from a user click, hides the details panel and expands the graph. **Exit presentation** or browser **Escape** returns to normal. If fullscreen is blocked, an in-page immersive fallback is used.
6. **Responsive + accessible:** Small screens place the panel below the graph; motion-reduction preferences continue to be respected. Buttons remain keyboard accessible, and Arrow Left/Right navigates stages.

The Market Risk and Financial RAG tours still use the original seven-stage architecture and link to original Python source. No API calls are made from this walkthrough.

## Files changed (only these)

```text
app/pages/3_Explore_RiskLens.py
src/risklens_story/frontend/src/Explorer.tsx
src/risklens_story/frontend/src/styles.css
tests/test_story_presentation_scale.py
this documentation
```

**No files under `src/risklens/`, no original data, no saved predictions, no original RAG adapters or Groq configuration, and no premium dashboard source are changed.**

## Installation on the owner's Mac

Stop Streamlit (Ctrl+C) before installing the ZIP in `~/Downloads`.

```bash
cd ~/Downloads/RiskLens
source .venv/bin/activate

unzip -o -q \
  ~/Downloads/RiskLens_ReactFlow_Massive_Presentation_Explorer_2026-11-02.zip -d .

# Recompile the UI: JSX and CSS changed (existing npm packages can be reused).
cd src/risklens_story/frontend
npm run build
cd ../../..

# Verify only the app-facing tests; no model tuning, holdout reruns or Groq.
PYTHONPATH=.:src python -m pytest -q tests/test_story_*.py tests/test_phase8_*.py

python -m streamlit run app/Home.py
```

Then visit `http://localhost:8501/Explore_RiskLens`. If the old display persists, refresh Chrome with **Command+Shift+R** after rebuilding. No `npm install` is needed unless `node_modules` has been removed.

## Browser verification checklist

- [ ] The map now fills most of the page and the first node is shown LARGE, not as a tiny seven-node diagram.
- [ ] **Focus stage** enlarges the selected node; **View whole map** deliberately zooms out to show all seven.
- [ ] **Hide details** expands the graph to nearly full page width; **Show details** restores the panel.
- [ ] **Presentation mode** fills the browser viewport; **Escape** or **Exit presentation** exits it.
- [ ] Click a node: the selected explanation, evidence and source-link panel match the node.
- [ ] **Play guided tour** moves between stages; the Market Risk and Financial RAG stories both work.
- [ ] Original Market Risk and Financial Research pages look and behave exactly as before.

## Testing performed here

- `PYTHONPATH=.:src python -m pytest -q tests/test_story_*.py tests/test_phase8_*.py` → **73 passed, 0 failed**.
- A TypeScript/TSX syntax transpilation using the locally installed TypeScript compiler → **0 syntax diagnostics**.
- Both seven-node stories passed a geometry check against the larger node size and expanded layout.
- `npm install --offline` was attempted, but the container has no cached `@streamlit/component-v2-lib`; therefore the **complete React dependency typecheck/build and browser visual test could not run here**. These are required on the Mac. The earlier repair for Streamlit's inline asset registration and browser `process` binding remains in the existing Python component wrapper and was not altered by this ZIP.

## Scientific and security boundary

This is an animated explanation of already-published architecture metadata. It is **not** an interactive execution of XGBoost, LSTM, financial-report extraction, Groq or a sealed holdout evaluation. It does not read the private source documents into the frontend. It is appropriate for education and presentation, not a live trading or audited-accounting workflow.
