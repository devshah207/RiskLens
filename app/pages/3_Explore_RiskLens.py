"""Interactive React Flow architecture tour; never imports or executes ML or RAG."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
from app.components.ui import inject_theme, sidebar_brand  # noqa: E402
from app.services.story_blueprints import build_stories, source_link  # noqa: E402
from risklens_story import frontend_is_built, render_explorer  # noqa: E402

st.set_page_config(page_title='RiskLens | Interactive System Explorer', page_icon='🧭',
                   layout='wide', initial_sidebar_state='expanded')
inject_theme(st)
sidebar_brand(st, 'Interactive System Explorer')

# Scope full-width layout to the architecture explorer page only. Original
# premium Market Risk and Financial Research pages are intentionally untouched.
st.markdown("""<style>
[data-testid="stMainBlockContainer"], .block-container {
  max-width: 100% !important;
  padding-left: clamp(12px, 1.8vw, 30px) !important;
  padding-right: clamp(12px, 1.8vw, 30px) !important;
  padding-top: 1.25rem !important;
}
.rl-explorer-intro {padding:17px 22px !important; margin-bottom:11px !important;}
.rl-explorer-intro h1 {font-size:clamp(1.3rem,2.6vw,2rem) !important; margin:3px 0 !important;}
.rl-explorer-intro p {font-size:.91rem !important; line-height:1.4 !important;}
</style>""", unsafe_allow_html=True)
st.markdown('''<div class="rl-hero rl-explorer-intro">
  <div class="rl-kicker">03 / Architecture Lab · Interactive Portfolio Walkthrough</div>
  <h1>Explore the real RiskLens architecture.</h1>
  <p>Follow the animated flow in a large, zoomable graph. Use <strong>Presentation mode</strong> to fill your screen, or hide the details panel to give the diagram maximum room.</p>
</div>''', unsafe_allow_html=True)

stories = build_stories(ROOT)
if frontend_is_built():
    try:
        render_explorer(st, stories)
    except (RuntimeError, FileNotFoundError) as exc:
        st.error(f'Interactive component cannot mount: {exc}')
else:
    st.warning('React Flow frontend has not been built on this machine. The existing RiskLens pages remain fully usable.')
    st.code('''cd ~/Downloads/RiskLens/src/risklens_story/frontend
npm install
npm run build
cd ~/Downloads/RiskLens
python -m pip install -e '.[ui,dev]'
python -m streamlit run app/Home.py''', language='bash')
    st.caption('Below is a static explanation preview; it is NOT an animated React Flow diagram.')
    track = st.radio('Preview architecture', ['market', 'research'], format_func=lambda s: stories[s]['title'],
                     horizontal=True)
    for i, code in enumerate(stories[track]['journey'], 1):
        node = next(n for n in stories[track]['nodes'] if n['id'] == code)
        with st.expander(f"{i:02d} · {node['title']} — {node['subtitle']}", expanded=i == 1):
            st.write(node['plain'])
            st.caption(node['evidence'])
            st.markdown(f"[Inspect original code]({source_link(node['sourcePath'])})")

st.markdown('### Open the working research tools')
left, right = st.columns(2, gap='large')
with left:
    st.page_link('pages/1_Market_Risk.py', label='Go to original Market Risk Analytics', icon='📈')
with right:
    st.page_link('pages/2_Financial_Research.py', label='Go to original Financial Research Assistant', icon='📑')

st.caption('The explorer displays truthful design and saved-research context, not a live financial forecast, new model evaluation, or live Groq answer.')
