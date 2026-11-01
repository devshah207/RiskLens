"""RiskLens Phase 8 — professional local-first research workbench landing page."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
from app.components.ui import hero, inject_theme, mini_stats, note, section, sidebar_brand  # noqa: E402

st.set_page_config(page_title='RiskLens | Research Workbench', page_icon='💠', layout='wide',
                   initial_sidebar_state='expanded')
inject_theme(st)
sidebar_brand(st, 'Overview')
hero(
    st, 'RiskLens · research edition',
    'See the risk. Trace the evidence.',
    'A local-first financial risk workbench combining frozen NIFTY 50 downside-event research with a source-anchored financial document assistant.',
)

mini_stats(st, [
    ('Market observations', '433', 'Original final holdout sessions'),
    ('Recorded downside events', '42', '1% next-session declines'),
    ('Research models', '2', 'Preselected champion + challenger'),
    ('Document corpus', '4', 'Companies in the original FY2025 study'),
])

section(st, 'Explore the workbench')
st.markdown('''<div class="rl-feature-grid">
<div class="rl-feature">
  <div class="hint">01 / Market intelligence</div>
  <h3>Market Risk Analytics</h3>
  <p>Explore saved probability traces, frozen model performance, calibration, historical Value at Risk, SHAP, LSTM sensitivity and feature drift.</p>
</div>
<div class="rl-feature">
  <div class="hint">02 / Document intelligence</div>
  <h3>Financial Research Assistant</h3>
  <p>Ask financial questions against original NSE XBRL filings, followed by indexed PDF retrieval and explicitly authorized Groq when needed. Inspect exact citations.</p>
</div>
</div>''', unsafe_allow_html=True)
left, right = st.columns(2, gap='large')
with left:
    st.page_link('pages/1_Market_Risk.py', label='Open Market Risk Analytics', icon='📈')
with right:
    st.page_link('pages/2_Financial_Research.py', label='Open Financial Research Assistant', icon='📑')

section(st, 'Scientific boundaries')
col1, col2 = st.columns(2, gap='large')
with col1:
    st.subheader('Frozen predictions, not a trading signal')
    st.write('The 2025–September 2026 holdout was opened once. Neither XGBoost nor LSTM detected any of the 42 real downturns at the fixed 0.50 cutoff. The interface only reads saved evidence.')
with col2:
    st.subheader('Evidence-backed, not accounting-certified')
    st.write('The published 34-question Phase 7 development set had 18 source-supported answers, 6 correct refusals, 4 incorrect refusals and 6 untested cases. These are not blind-validation accuracy claims.')
note(st, 'The two systems are complementary research tools. FY2025 annual reports were not historical predictors for the NIFTY 50 classification study.')
st.caption('RiskLens is an educational research prototype. Not investment advice, live forecasting or a bank-approved risk engine.')
