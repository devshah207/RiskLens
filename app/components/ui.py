"""Presentation helpers only: never interpret financial evidence or call providers.

All dynamic text inserted in HTML is escaped. Retrieved PDF text and LLM output
are untrusted, even when the underlying source is authentic.
"""
from __future__ import annotations

from html import escape
import re

from app.services.filing_research import safe_source_url

STATUS_TEXT = {
    "answered": "Source-backed answer",
    "needs_review": "Answer needs review",
    "abstained": "Insufficient evidence — refused",
    "not_evaluated": "Not evaluated",
    "provider_unavailable": "Provider unavailable",
    "provider_error": "Provider error",
}
STATUS_COLORS = {
    "answered": ("#4ade80", "rgba(20,83,45,.26)"),
    "needs_review": ("#fbbf24", "rgba(120,53,15,.26)"),
    "abstained": ("#93c5fd", "rgba(30,58,138,.24)"),
    "not_evaluated": ("#cbd5e1", "rgba(71,85,105,.2)"),
    "provider_unavailable": ("#fca5a5", "rgba(127,29,29,.23)"),
    "provider_error": ("#fca5a5", "rgba(127,29,29,.23)"),
}


def _html(value: object) -> str:
    """Never trust source excerpts, answer bodies or provider metadata as HTML."""
    return escape(str(value), quote=True)


def inject_theme(st) -> None:
    st.markdown(
        """<style>
        :root { --ink:#070c16; --midnight:#0c1525; --surface:#101c2d; --line:#27374b;
                --mint:#55e7c2; --sky:#58bdf8; --lilac:#a5a0ff; --muted:#9aafc4; }
        .stApp {background: radial-gradient(ellipse 900px 450px at 72% -100px,rgba(35,111,142,.19), transparent),
                            linear-gradient(180deg,#09121f 0%,#090f1a 65%,#070c14 100%); color:#eaf4ff; }
        .stApp, .stApp p, .stApp label {font-family:'SF Pro Text',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;}
        [data-testid='stHeader'] {background:rgba(9,17,29,.8);}
        [data-testid='stSidebar'] {background:linear-gradient(180deg,#101a2c,#0a1120);
          border-right:1px solid #27374b;}
        [data-testid='stSidebar'] hr {border-color:#243447;}
        [data-testid='stSidebarNavLink'] {border-radius:10px;}
        [data-testid='stSidebarNavLink'][aria-current='page'] {background:rgba(85,231,194,.14);}
        .block-container {max-width:1290px;padding-top:2.2rem;padding-bottom:3rem;}
        h1,h2,h3 {font-family:'SF Pro Display',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif!important;letter-spacing:-.035em!important;}
        h1 {color:#f4fbff!important;} h2,h3 {color:#eaf4ff!important;}
        [data-testid='stMetric'] {background:linear-gradient(165deg,#111f30,#0d1828);
             border:1px solid #27374b;border-radius:15px;padding:15px 18px;}
        [data-testid='stMetricLabel'] {color:#a9bad0!important;}
        [data-testid='stMetricValue'] {color:#effff8!important;}
        [data-testid='stTabs'] button {font-weight:700;letter-spacing:.005em;}
        [data-testid='stExpander'] {border:1px solid #27374b!important;border-radius:14px!important;}
        [data-testid='stDataFrame'] {border:1px solid #263649;border-radius:14px;overflow:hidden;}
        [data-testid='stForm'] {border:1px solid #304359;border-radius:19px;padding:22px;
             background:linear-gradient(135deg,rgba(20,37,55,.8),rgba(12,24,39,.8));}
        [data-testid='stMarkdownContainer'] a {color:#8cebd8;}
        [data-testid='stBaseButton-primary'] {background:linear-gradient(100deg,#51d8b5,#46b8db)!important;
            color:#06151d!important;border:none!important;font-weight:800!important;border-radius:11px!important;}
        [data-testid='stBaseButton-primary']:hover {filter:brightness(1.08);}
        [data-testid='stPlotlyChart'] {border:1px solid #223348;border-radius:16px;background:#0d1827;}
        .rl-kicker {font-size:.74rem;text-transform:uppercase;letter-spacing:.17em;
            font-weight:800;color:#5be5c7;margin-bottom:.55rem;}
        .rl-hero {border:1px solid rgba(95,185,201,.28);border-radius:22px;
            padding:30px 34px 28px;background:radial-gradient(circle at 98% 2%,rgba(87,212,194,.2),transparent 35%),
              linear-gradient(116deg,#14273c 0%,#102137 52%,#151b36 100%);
            margin-bottom:22px;box-shadow:0 14px 35px rgba(0,0,0,.12);}
        .rl-hero h1{font-family:'Space Grotesk',sans-serif;font-size:clamp(2.0rem,4vw,3.1rem);
            line-height:1.08;margin:0 0 .7rem;font-weight:700;}
        .rl-hero p {font-size:1.02rem;line-height:1.65;color:#bcd0e1;max-width:790px;margin:0;}
        .rl-hero .rl-kicker{color:#85f5de;}
        .rl-stat-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0 22px;}
        .rl-stat {border:1px solid #263a4e;border-radius:15px;padding:19px 17px 16px;
                  background:linear-gradient(160deg,#132337,#101c2c);min-height:133px;}
        .rl-stat-label{font-size:.82rem;color:#a6b8c8;font-weight:700;}
        .rl-stat-value{font-family:'Space Grotesk',sans-serif;font-size:1.9rem;color:#edfef9;
            letter-spacing:-.04em;margin-top:7px;font-weight:700;}
        .rl-stat-note{font-size:.78rem;color:#92a9bd;margin-top:5px;line-height:1.4;}
        .rl-note {border-left:3px solid #46d7ba;background:rgba(67,168,161,.075);
            padding:13px 16px;border-radius:0 12px 12px 0;color:#bdd9d8;margin:12px 0 24px;
            line-height:1.65;font-size:.93rem;}
        .rl-warning {border-left-color:#f59e0b;background:rgba(245,158,11,.08);color:#ead4a0;}
        .rl-section {display:flex;align-items:center;gap:11px;margin:27px 0 10px;}
        .rl-section span{font-size:.74rem;letter-spacing:.13em;color:#55e7c2;text-transform:uppercase;font-weight:800;}
        .rl-section:after{content:'';height:1px;background:#243647;flex:1;}
        .rl-status {display:inline-flex;align-items:center;padding:.42rem .8rem;border-radius:999px;
            border:1px solid rgba(255,255,255,.1);font-size:.81rem;font-weight:700;margin:10px 0 7px;}
        .rl-result {border:1px solid #325a5a;border-radius:19px;
            background:linear-gradient(127deg,rgba(16,76,77,.25),rgba(18,38,60,.77));
            padding:25px 28px;margin:6px 0 13px;box-shadow:0 10px 24px rgba(0,0,0,.13);}
        .rl-result-caution{border-color:#765c3d;background:linear-gradient(127deg,rgba(102,70,35,.19),rgba(18,38,60,.74));}
        .rl-result-caution .overline{color:#f4c978;}
        .rl-result .overline{font-size:.76rem;text-transform:uppercase;letter-spacing:.16em;color:#6cf5d0;font-weight:800;}
        .rl-result .amount{font-family:'Space Grotesk',sans-serif;font-size:clamp(2rem,5vw,3.2rem);
           font-weight:700;line-height:1.2;letter-spacing:-.05em;color:#f6fffd;margin:12px 0 8px;overflow-wrap:anywhere;}
        .rl-result .body{font-size:1.05rem;line-height:1.75;color:#e0eef4;overflow-wrap:anywhere;white-space:pre-wrap;}
        .rl-result .foot{color:#a9bdcb;font-size:.78rem;margin-top:14px;overflow-wrap:anywhere;}
        .rl-feature-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px;margin:18px 0 22px;}
        .rl-feature{border:1px solid #293d52;border-radius:17px;padding:22px;
           background:linear-gradient(155deg,#172b3f,#111d31);min-height:175px;}
        .rl-feature h3{margin:6px 0 9px;font-size:1.45rem;}
        .rl-feature p{color:#b0c3d5;line-height:1.65;font-size:.94rem;}
        .rl-feature .hint{font-size:.77rem;font-weight:800;letter-spacing:.12em;color:#5be5c7;text-transform:uppercase;}
        .rl-fine{font-size:.8rem;line-height:1.55;color:#849bb0;}
        @media(max-width:820px){.rl-hero{padding:24px 20px}.rl-stat-grid{grid-template-columns:repeat(2,1fr)}
            .rl-feature-grid{grid-template-columns:1fr}}
        </style>""",
        unsafe_allow_html=True,
    )


def sidebar_brand(st, section: str) -> None:
    """Keep repository-defined multipage navigation; add non-interactive branding."""
    with st.sidebar:
        st.markdown(
            '<div style="padding:10px 2px 8px"><span style="font-size:1.5rem;font-weight:800;'
            'color:#ecfffb;letter-spacing:-.05em">◈ RiskLens</span>'
            '<div style="color:#8facbc;font-size:.8rem;margin-top:4px">Evidence-driven research workbench</div></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Viewing: {section}")
        st.divider()


def hero(st, eyebrow: str, title: str, body: str) -> None:
    st.markdown(
        f'<div class="rl-hero"><div class="rl-kicker">{_html(eyebrow)}</div>'
        f'<h1>{_html(title)}</h1><p>{_html(body)}</p></div>',
        unsafe_allow_html=True,
    )


def mini_stats(st, items: list[tuple[str, str, str]]) -> None:
    blocks = []
    for label, value, note in items:
        blocks.append(
            '<div class="rl-stat"><div class="rl-stat-label">'
            f'{_html(label)}</div><div class="rl-stat-value">{_html(value)}</div>'
            f'<div class="rl-stat-note">{_html(note)}</div></div>'
        )
    st.markdown('<div class="rl-stat-grid">' + ''.join(blocks) + '</div>', unsafe_allow_html=True)


def section(st, label: str) -> None:
    st.markdown(f'<div class="rl-section"><span>{_html(label)}</span></div>', unsafe_allow_html=True)


def note(st, text: str, *, warning: bool = False) -> None:
    cls = 'rl-note rl-warning' if warning else 'rl-note'
    st.markdown(f'<div class="{cls}">{_html(text)}</div>', unsafe_allow_html=True)


def show_status(st, result: dict) -> None:
    status = str(result.get('status') or 'not_evaluated')
    color, bg = STATUS_COLORS.get(status, STATUS_COLORS['not_evaluated'])
    label = STATUS_TEXT.get(status, STATUS_TEXT['not_evaluated'])
    st.markdown(
        f'<div class="rl-status" style="color:{color};background:{bg};">'
        f'● &nbsp; {_html(label)}</div>', unsafe_allow_html=True,
    )


def _xbrl_display(answer: str, route: str) -> tuple[str | None, str]:
    """Format an already-returned answer; do not infer a new financial number."""
    if route == 'xbrl_exact_fact' and ':' in answer:
        context, figure = answer.rsplit(':', 1)
        if re.fullmatch(r'\s*₹?[-+]?\s*[\d,]+(?:\.\d+)?\s*(?:crore|thousand rupees|rupees|per share)?\s*', figure, flags=re.I):
            return figure.strip(), answer
    return None, answer


def render_answer_card(st, result: dict) -> None:
    """Show the actual answer FIRST, not just its validation status."""
    answer = str(result.get('answer') or 'No answer was returned.')
    status = result.get('status', 'not_evaluated')
    route = str(result.get('route') or 'unknown')
    highlight, full_text = _xbrl_display(answer, route)
    title = {
        'answered': 'Answer from source evidence',
        'needs_review': 'Provisional answer — verify before use',
        'abstained': 'No supported answer available',
        'not_evaluated': 'Answer not evaluated',
        'provider_unavailable': 'Provider could not complete the request',
        'provider_error': 'Provider could not complete the request',
    }.get(str(status), 'Research result')
    kind = 'rl-result' if status == 'answered' else 'rl-result rl-result-caution'
    content = (
        f'<div class="amount">{_html(highlight)}</div>'
        f'<div class="body">{_html(full_text)}</div>' if highlight
        else f'<div class="body" style="font-size:1.3rem;font-weight:600">{_html(full_text)}</div>'
    )
    provider = result.get('provider') or 'unknown'
    review = result.get('verification_status') or 'not independently certified'
    st.markdown(
        f'<div class="{kind}"><div class="overline">{_html(title)}</div>{content}'
        f'<div class="foot">Route: {_html(route)} &nbsp;·&nbsp; Provider: {_html(provider)}'
        f' &nbsp;·&nbsp; Review: {_html(review)}</div></div>',
        unsafe_allow_html=True,
    )


def show_citations(st, citations: list[dict]) -> None:
    """Display original source metadata without turning PDF content into commands."""
    section(st, 'Source transparency')
    if not citations:
        st.caption('The backend returned no answer citation for this outcome.')
        return
    st.markdown(f'### Evidence trail · {len(citations)} source{"s" if len(citations) != 1 else ""}')
    for idx, cite in enumerate(citations, start=1):
        kind = str(cite.get('source_kind') or ('PDF' if cite.get('pdf_page') else 'Source'))
        who = str(cite.get('ticker') or 'source')
        page = cite.get('pdf_page')
        suffix = f' · physical PDF p. {page}' if page is not None else ''
        with st.expander(f'{idx:02d}  {who}  ·  {kind}{suffix}', expanded=idx == 1):
            left, right = st.columns(2)
            with left:
                st.caption('SOURCE FILE')
                st.write(cite.get('source_file') or 'Original filename not provided')
                st.caption('ACCOUNTING CONCEPT / METRIC')
                concept = cite.get('concept_qname') or cite.get('metric') or cite.get('row_label')
                st.write(str(concept).split('}')[-1] if concept else 'Not specified')
                st.caption('SOURCE DATE / PERIOD')
                st.write(cite.get('period_end') or cite.get('period') or cite.get('published_on') or 'Not specified')
            with right:
                st.caption('REPORTING BASIS')
                st.write(cite.get('reported_scope') or cite.get('scope') or 'Not specified')
                st.caption('ORIGINAL UNIT')
                st.write(cite.get('unit_measures') or cite.get('unit') or 'Not specified')
                st.caption('SOURCE TYPE')
                st.write(kind)
            excerpt = cite.get('evidence_excerpt') or cite.get('quote')
            if excerpt:
                st.markdown('**Original supporting excerpt**')
                st.text(str(excerpt)[:900])  # untrusted data, rendered as plain text
            url = safe_source_url(cite.get('source_url'))
            if url:
                label = ('Open filing portal' if kind == 'NSE_XBRL_XML' else 'Open original source website')
                st.link_button(label, url)
            with st.expander('Detailed source metadata (technical audit)'):
                for field in ('source_sha256', 'file_sha256', 'raw_reported_value', 'raw_value',
                              'decimals', 'context_id', 'financial_year', 'period_kind',
                              'period_start', 'period_end', 'published_on', 'pdf_page',
                              'fact_id', 'note', 'verification'):
                    if cite.get(field) is not None:
                        st.write(f'{field}: {cite[field]}')
