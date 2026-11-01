"""Original Phase 7 XBRL-first/PDF/Groq engine, displayed as an answer-first UI."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
from app.components.ui import (  # noqa: E402
    hero, inject_theme, mini_stats, note, render_answer_card,
    section, show_citations, show_status, sidebar_brand,
)
from app.services.demo_data import EXAMPLES, recorded_question  # noqa: E402
from app.services.filing_research import (  # noqa: E402
    COMPANIES, ResearchPaths, ask_saved_sources, normalize_response, source_readiness,
)

st.set_page_config(page_title='RiskLens | Financial Research', page_icon='📑', layout='wide',
                   initial_sidebar_state='expanded')
inject_theme(st)
sidebar_brand(st, 'Financial Research Assistant')
hero(
    st, '02 / Financial Research Assistant', 'Find the fact. Show the source.',
    'Ask a question about a company filing. The original Phase 7 XBRL engine looks for an exact tagged fact first; only unanswered questions continue to PDF retrieval and optional Groq.',
)
st.caption('Research prototype · Source-backed is not independently human-certified · Numbers marked needs_review must be checked against the original filing.')

with st.sidebar:
    st.markdown('### Local source settings')
    st.caption('Paths below refer to files on your Mac. The dashboard never uploads or edits them.')
    default = ResearchPaths.for_root(ROOT)
    with st.expander('Change local research file paths'):
        archive_text = st.text_input('Eight original XBRL files (ZIP)',
                                     value=str(default.archive))
        index_text = st.text_input('Existing SQLite PDF index', value=str(default.index))
        banks_text = st.text_input('Three original financial fact banks', value=str(default.banks))
        pdfs_text = st.text_input('Five original annual-report PDFs', value=str(default.pdfs))
    # These variables default to project paths even if the expander is closed.
    paths = ResearchPaths(
        archive=Path(archive_text).expanduser(),
        index=Path(index_text).expanduser(),
        banks=Path(banks_text).expanduser(),
        pdfs=Path(pdfs_text).expanduser(),
    )
    readiness = source_readiness(paths)
    st.caption('Original local files (presence only, not a content audit):')
    for ok, label in [
        (readiness['xbrl_archive_present'], 'Eight-file XBRL ZIP'),
        (readiness['pdf_index_present'], 'Indexed PDF research store'),
        (readiness['bank_files_present'], 'Three structured fact banks'),
        (readiness['pdf_directory_present'], 'PDF document folder'),
    ]:
        st.markdown(f'{"🟢" if ok else "🟠"} {label}')
    if readiness['missing_bank_filenames']:
        st.caption('Missing banks: ' + ', '.join(readiness['missing_bank_filenames']))

st.markdown('#### Select how to research')
mode = st.radio(
    'Source mode',
    ['Local original documents · no Groq',
     'Local original documents · Groq if needed',
     'Recorded Phase 7 demonstration'],
    index=0, horizontal=True, label_visibility='collapsed',
)

if mode.startswith('Recorded'):
    note(st, 'Recorded demonstration only: these are previously documented outcomes from an exposed development benchmark. This mode makes NO original backend request and does not provide new validation.')
    examples = {e['id']: f"{e['company']} · {e['question']}" for e in EXAMPLES}
    chosen = st.selectbox('Select a recorded development example', list(examples), format_func=examples.get)
    if st.button('Show recorded research outcome', type='primary'):
        st.session_state['phase8_recorded_result'] = normalize_response(recorded_question(chosen))
        st.session_state['phase8_recorded_selection'] = chosen
    if st.session_state.get('phase8_recorded_selection') == chosen:
        result = st.session_state.get('phase8_recorded_result')
        if result:
            section(st, 'Research result · recorded example')
            show_status(st, result)
            render_answer_card(st, result)
            show_citations(st, result['citations'])
            st.caption('Record: docs/phase7/README.md and November 2 project handover; no local filing verification was performed in this display mode.')
else:
    live = 'Groq if needed' in mode
    # Consent is deliberately OUTSIDE the form. Form-contained checkboxes do
    # not trigger a rerun until submission, leaving a disabled submit button
    # unable to react to the consent selection.
    if live:
        st.warning('Live PDF fallback may send retrieved financial excerpts to your own Groq account and can use multiple API requests. Exact XBRL answers do not call Groq.')
        authorized = st.checkbox(
            'I authorize the original Phase 7 Groq requests for this single question if exact XBRL is unavailable.',
            value=False,
        )
    else:
        authorized = False
    with st.form('phase8_research_question', clear_on_submit=False):
        c1, c2, c3 = st.columns([1.3, .85, 1.05], gap='medium')
        companies = list(COMPANIES)
        with c1:
            company = st.selectbox('Company', companies, index=2)
            ticker = COMPANIES[company]
        with c2:
            year = st.selectbox('Fiscal year', [2025, 2024], index=0)
        with c3:
            scope = st.selectbox('Reporting basis', ['consolidated', 'standalone', 'Not specified'], index=0)
        first = f'What was {company} consolidated net profit for FY{year}, in crore?'
        if ticker in {'INFY', 'RIL'}:
            first = f'What was {company} consolidated revenue from operations for FY{year}, in crore?'
        question = st.text_area(
            'Ask a question', value=first, height=95, max_chars=1200,
            help='Include company, financial year, standalone/consolidated basis and requested unit when applicable.',
        )
        if not live:
            st.caption('Groq is off. Exact XBRL answers remain available; PDF retrieval can find candidate pages but may not generate an answer.')
        submitted = st.form_submit_button('Find answer in original sources',
                                          type='primary', disabled=live and not authorized)

    request_identity = (
        question, ticker, year, scope, mode,
        str(paths.archive.resolve()), str(paths.index.resolve()),
        str(paths.banks.resolve()), str(paths.pdfs.resolve()),
    )
    if submitted:
        try:
            with st.spinner('Checking the original XBRL and PDF research sources…'):
                result = ask_saved_sources(
                    question, ticker=ticker, paths=paths,
                    scope=None if scope == 'Not specified' else scope,
                    year=year, provider='groq' if live else 'offline',
                    groq_allowed=authorized,
                )
        except ValueError as exc:
            result = normalize_response({
                'question': question, 'answer': str(exc), 'status': 'not_evaluated',
                'route': 'invalid_input', 'citations': [],
            })
        st.session_state['phase8_source_result'] = result
        st.session_state['phase8_source_request'] = request_identity

    if st.session_state.get('phase8_source_request') == request_identity:
        result = st.session_state.get('phase8_source_result')
        if result:
            section(st, 'Research result')
            show_status(st, result)
            render_answer_card(st, result)  # BEFORE citations / diagnostics
            source_count = len(result['citations'])
            mini_stats(st, [
                ('Outcome', str(result['status']).replace('_', ' ').title(), 'Original backend status'),
                ('Backend route', str(result.get('route', 'unknown')).replace('_', ' '), 'No route is silently upgraded'),
                ('Source references', str(source_count), 'Original returned citations'),
                ('Groq', 'Allowed' if live else 'Off', 'Only for explicitly authorized live mode'),
            ])
            if result.get('intent'):
                with st.expander('Question interpretation (original Phase 7 output)'):
                    st.json(result['intent'], expanded=False)
            show_citations(st, result['citations'])
            candidates = result.get('retrieved_physical_pages') or []
            if candidates and not result['citations']:
                section(st, 'Retrieved leads, not citations')
                note(st, 'The original offline retrieval found candidate PDF pages but did not generate a supported answer. These are investigation leads, not proof that a number has the right accounting period or unit.', warning=True)
                st.dataframe([{'File fingerprint': sha[:12] + '…', 'Physical PDF page': page}
                              for sha, page in candidates[:10]],
                             hide_index=True, use_container_width=True)
    else:
        note(st, 'Start with an exact XBRL question for a fast, source-backed answer. For a question about an untagged PDF disclosure, choose the Groq mode only when you want to authorize the external API call.')

section(st, 'Published evaluation · context only')
with st.expander('View the published 34-question development benchmark'):
    mini_stats(st, [
        ('Source-supported', '18', 'Assistant-reviewed evidence'),
        ('Correct refusals', '6', 'Unsupported requests'),
        ('Incorrect refusals', '4', 'Answerable questions missed'),
        ('Not tested', '6', 'Not run end-to-end'),
    ])
    st.caption('Exposed and repeatedly seen questions; NOT a blind accuracy rate, a production guarantee or independent human certification.')
    st.markdown('[Original Phase 7 report](https://github.com/devshah207/RiskLens/blob/main/docs/phase7/README.md)')
