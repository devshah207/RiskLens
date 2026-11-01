# RiskLens — Phase 8 premium dashboard update

This update is **UI-only** for the owner's existing RiskLens checkout. It reuses the original Phase 7 XBRL/PDF/Groq backend and frozen Phase 4–6 saved study; nothing is trained, tuned, rescored, or pushed.

## Changes

**Home:** branded dark navy/mint landing page, clearer navigation and concise historical evidence limits.

**Market Risk Analytics:** five focused tabs (Overview; Probability explorer; Historical VaR; Explanations & drift; Governance), with interactive Plotly charts made only from the original saved CSV/JSON: actual XGBoost/LSTM probability trends, highlighted actual downside-event dates, saved reliability bins, historical VaR breach rates, original SHAP/sensitivity and feature PSI. Original 0.50-threshold recall remains explicitly 0% for both models. XGBoost remains preselected champion. Research comparisons do not trigger re-evaluation.

**Financial Research Assistant:** source question form near the top. The actual **answer** is displayed prominently in a highlighted card immediately after submission — for example, the original XBRL backend's `₹162,990 crore` is large and the full returned sentence appears beneath it. Status and review labels appear separately. Citation cards show original filing filename, concept, reporting scope, period, unit, physical PDF page when available, and expandable raw technical metadata. The recorded development benchmark is moved to a closed expander. Local file settings and readiness checks are in the sidebar rather than between the question and answer.

**Safety:** Every dynamic answer/status inserted into HTML is escaped, and PDF excerpts are rendered as plain text. The default local research mode does not call Groq. Groq mode requires an explicit consent checkbox and a separate submit action; the checkbox is outside the form so Streamlit can update its disabled submit button. An XBRL answer bypasses Groq entirely. The UI uses the existing `filing_research` service without changing Phase 7 financial rules.

## Files included

```text
app/Home.py
app/components/ui.py
app/components/charts.py              (new)
app/pages/1_Market_Risk.py
app/pages/2_Financial_Research.py
tests/test_phase8_render_smoke.py     (UI-stub support for sidebar, tabs and form)
tests/test_phase8_premium_ui.py       (new answer/security/chart regression tests)
patches/phase8_premium_plotly.patch    (adds Plotly to optional ui extra)
docs/PHASE8_PREMIUM_UI_GUIDE_2026-11-02.md
```

No original `src/risklens/`, Phase 6 manifest, holdout files or original Phase 7 code is included or changed. No real research data or credentials are bundled.

## Install on your Mac

Stop the running app with Control+C first. With the ZIP downloaded to `~/Downloads`:

```bash
cd ~/Downloads/RiskLens
source .venv/bin/activate

# Optional backup of the existing UI outside the repository
cp -R app ~/Downloads/RiskLens_App_Before_Premium

unzip -o -q \
  ~/Downloads/RiskLens_Phase8_Premium_Portfolio_UI_2026-11-02.zip -d .

# Add Plotly to the existing optional UI dependencies; safe on repeated use
if ! grep -q 'plotly>=6,<7' pyproject.toml; then
  git apply --check patches/phase8_premium_plotly.patch && \
  git apply patches/phase8_premium_plotly.patch
fi

python -m pip install -e '.[ui,dev]'
PYTHONPATH=.:src python -m pytest -q tests/test_phase8_*.py
python -m streamlit run app/Home.py
```

**Plotly** is an interactive graph library. It powers hover details, zooming and more legible chart styling. It is added only to the optional `ui` installation group; it is not a new market prediction model or part of RAG.

## What to verify visually

1. Open **Financial Research**. In default *Local original documents · no Groq* mode, choose Infosys / FY2025 / consolidated, and submit `What was Infosys consolidated revenue from operations for FY2025, in crore?`. The XBRL result should display the amount prominently, the full original answer, review caveat and source citation immediately below the question form.
2. For a PDF-only question in offline mode, the UI should explain retrieval-only status and candidate pages rather than invent an answer. An explicitly authorized live Groq request is optional, not part of this release check.
3. Open **Market Risk**. Verify the **Overview** and **Probability explorer** use actual saved results, then inspect VaR and explanations tabs. Confirm the displayed 0.50-threshold recall caveat.

## Tests and limitations

- **45 passed, 0 failed** in the Phase 8 tests within the assistant's isolated working copy.
- **69 passed, 40 skipped** in a selected Phase 7 + Phase 8 suite. Skipped tests need original private research sources and are not passes.
- All Plotly chart generators were also run against the uploaded original saved 433-row predictions, reliability bins, VaR results, original SHAP/LSTM sensitivity and PSI files. They constructed valid chart objects without modifying the saved data.
- Streamlit was not installed in the assistant's container; real browser layout and installed Streamlit compatibility must still be confirmed on the user's Mac. This work is tested offline with the existing Streamlit page stub and real Plotly, not claimed to be verified in the user's browser.
- No Groq quota consumed, no new scientific metrics produced, and nothing pushed to GitHub.
