"""V5: V3 curated facts + V4 native-table JSON + guarded semantic RAG.

The existing frozen Phase 6 forecasting system and five-report index are untouched.
Prioritized numeric answers use deterministic original-PDF evidence checks, not LLM
agreement. Groq review remains available for uncertain generated numeric claims.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import re
from typing import Any

from .financial_facts import read_facts, fact_answer, validate_original_pdfs
from .financial_auto_v4 import load_auto_bank, confirm_original_table_rows
from .financial_scope_v4 import automatic_fact_answer, inferred_intent, rerank_hits, structured_evidence
from .structured_pipeline_v4 import _provider_client, _provider_status, _verify_result, _has_numeric_claim, _source_map
from .retrieval import search
from .answer import respond
from .layout_panels_v6 import (
    default_panel_bank_path, read_panel_bank, answer_from_layout_panels,
)

# Hard constraints, not fuzzy string matching. Never treat parent-company revenue
# as the answer to an explicit business-segment question.
_SEGMENT_ALIASES = {
    'oil and gas': (r'\boil\s*(?:and|&)\s*gas\b', r'\bo&g\b'),
    'oil to chemicals': (r'\boil[- ]to[- ]chemicals\b', r'\bo2c\b'),
    'digital services': (r'\bdigital services\b', r'\bjio\b'),
    'media and entertainment': (r'\bmedia\s*(?:and|&)\s*entertainment\b',),
    'retail': (r'\bretail\b',),
    'instabiz': (r'\binstabiz\b',),
    'imobile': (r'\bimobile\b',),
}


def explicit_segments(question: str) -> set[str]:
    q = question.casefold()
    return {name for name, patterns in _SEGMENT_ALIASES.items()
            if any(re.search(pattern, q) for pattern in patterns)}


def _answering_facts_are_in_scope(question: str, response: dict[str, Any]) -> bool:
    """Fail closed on explicit segment, year and scope mismatches in curated output."""
    demanded = explicit_segments(question)
    if not demanded:
        return True
    cites = [c for c in response.get('citations', []) if c.get('fact_role') != 'definition']
    if not cites:
        return False
    return all(str(c.get('segment', '')).strip().lower() in demanded
               for c in cites)


def _unit_requested(question: str) -> str | None:
    q = question.casefold()
    if re.search(r'\b(?:thousands? of rupees|in thousands? of (?:inr|rupees)|rupees?\s*\'?000s)\b', q):
        return 'thousand_rupees'
    if re.search(r'\b(?:crore|crores)\b', q):
        return 'crore'
    if re.search(r'\b(?:percentage|percent)\b', q) or '%' in q:
        return 'percent'
    if re.search(r'\b(?:rupees|inr)\b', q):
        return 'rupees'
    return None


def _format_decimal(amount: Decimal) -> str:
    if amount == amount.to_integral():
        return f'{amount:,.0f}'
    return f'{amount:,.6f}'.rstrip('0').rstrip('.')


def _display_auto_value(fact: dict, question: str) -> str | None:
    """Honor the unit requested by the user; never silently change to crore."""
    val = Decimal(fact['value'])
    unit, wanted = fact['unit'], _unit_requested(question)
    if unit == 'percent':
        return _format_decimal(val) + '%' if wanted in (None, 'percent') else None
    if unit == 'thousand_rupees':
        if wanted in (None, 'thousand_rupees'):
            return '₹' + _format_decimal(val) + ' thousand (₹ thousands)'
        if wanted == 'crore':
            return '₹' + _format_decimal(val / Decimal(10000)) + ' crore'
        if wanted == 'rupees':
            return '₹' + _format_decimal(val * Decimal(1000))
        return None
    if unit == 'crore':
        if wanted not in (None, 'crore'):
            return None
        return ('₹' if fact.get('currency') == 'INR' else '') + _format_decimal(val) + ' crore'
    if unit == 'rupees':
        if wanted not in (None, 'rupees'):
            return None
        return '₹' + _format_decimal(val)
    if wanted not in (None, unit):
        return None
    return _format_decimal(val) + (' ' + unit if unit not in ('count', '') else '')


def _future_unavailable(question: str) -> bool:
    q = question.lower()
    # Refuse unsupported future requests locally, without burning API quota.
    # Corpus ends FY2025; do not extrapolate historical filings into forecasts.
    if not re.search(r'\b20(?:2[7-9]|[3-9]\d)\b', q):
        return False
    # Refuse requests for realized future market/financial values, but still
    # search documents for *historical management projections or plans*.
    if re.search(r'\b(?:what (?:did|does) .{0,40}(?:project|plan|forecast|expect)|management (?:guidance|forecast))\b', q):
        return False  # past disclosure about future plans might be in the report
    measured = bool(re.search(r'\b(?:revenue|earnings|ebitda|npa|profit|stock price|share price|next.day return|return on assets)\b', q))
    market_outcome = bool(re.search(r'\b(?:nifty|share price|stock price|closing price|intraday|index return|market return)\b', q))
    categorical_exactness = bool(re.search(r'\b(?:exact|precise|guaranteed|probability|what will|how much will)\b', q))
    return (measured and categorical_exactness) or (market_outcome and categorical_exactness)


def choose_fact_answer(question: str, curated: list, automatic: list[dict], *,
                       ticker: str | None = None, eligible_sha256: set[str] | None = None,
                       as_of: str | None = None,
                       pdf_dir: str | Path | None = None,
                       verify_sources: bool = True) -> dict[str, Any] | None:
    """Return a source-checked numeric answer or None for the RAG fallback.

    No gold labels or predefined question-to-answer mappings are used here.
    A source-validation failure is a review result, never a successful answer.
    """
    if _future_unavailable(question):
        return {'question': question, 'answer': 'Not contained in the historical annual reports.',
                'status': 'abstained', 'provider': 'source_period_guard',
                'route': 'deterministic_future_refusal', 'citations': [],
                'verification_status': 'historical_corpus_limit'}
    requested_segments = explicit_segments(question)
    initial = fact_answer(question, curated, ticker=ticker, eligible_sha256=eligible_sha256,
                          as_of=as_of) if curated else None
    if initial and not _answering_facts_are_in_scope(question, initial):
        initial = None
    if initial and initial['status'] == 'answered':
        if verify_sources:
            if not pdf_dir:
                raise ValueError('pdf_dir is required for source verification')
            try:
                selected_ids = {c['fact_id'] for c in initial['citations']}
                validate_original_pdfs((f for f in curated if f.fact_id in selected_ids), pdf_dir)
            except Exception as exc:
                return {'question': question, 'answer': 'Source revalidation failed; no confident numeric answer.',
                        'status': 'needs_review', 'provider': 'source_checked_financial_facts',
                        'route': 'v3_pdf_check_failed', 'citations': initial['citations'],
                        'verification_status': 'source_recheck_failed',
                        'error_type': type(exc).__name__}
        return {**initial, 'question': question, 'route': 'v3_curated_json',
                'verification_status': 'pdf_sha_and_span_checked' if verify_sources else 'not_rechecked'}

    # Do not let V4's incomplete segment parser silently choose a company value
    # when the question explicitly requests a separate business segment.
    parsed_segments = set(inferred_intent(question, ticker)['segments'])
    if requested_segments and not requested_segments.issubset(parsed_segments):
        alternative = None
    else:
        alternative = automatic_fact_answer(question, automatic, ticker=ticker,
                                            source_hashes=eligible_sha256, as_of=as_of)
    if alternative:
        chosen_ids = {c['fact_id'] for c in alternative['citations']}
        selected = [f for f in automatic if f['fact_id'] in chosen_ids]
        # Require explicit scope to match; no company-level substitutions.
        if requested_segments and any(f.get('segment', '').casefold() not in requested_segments
                                      for f in selected):
            selected = []
        if selected:
            conversions = [_display_auto_value(f, question) for f in selected]
            if all(v is not None for v in conversions):
                if verify_sources:
                    if not pdf_dir:
                        raise ValueError('pdf_dir is required for source verification')
                    try:
                        confirm_original_table_rows(selected, pdf_dir=pdf_dir)
                    except Exception as exc:
                        return {'question': question, 'answer': 'Automatic source-table check failed.',
                                'status': 'needs_review', 'provider': 'auto_financial_table_json',
                                'route': 'v4_table_check_failed', 'citations': alternative['citations'],
                                'verification_status': 'source_recheck_failed',
                                'error_type': type(exc).__name__}
                answer = '; '.join(
                    f"{f['company']} {f['reporting_scope']} {f['row_label']} ({f['observed_period']}): {display}"
                    for f, display in zip(selected, conversions))
                return {**alternative, 'question': question, 'answer': answer,
                        'status': 'answered' if verify_sources else 'needs_review',
                        'route': 'v4_native_table_json',
                        'verification_status': ('native_row_column_year_scope_checked_not_human_certified'
                                                if verify_sources else 'source_check_pending'),
                        'note': 'Source table row, column, year, unit and PDF hash rechecked. Automatic semantic attribution is not human-certified.'}

    # Explicitly ambiguous V3 answers may be useful if correctly scoped, but
    # cannot be promoted into a confident number or replaced with arbitrary scope.
    if initial and initial['status'] == 'needs_review' and _answering_facts_are_in_scope(question, initial):
        return {**initial, 'question': question, 'route': 'v3_ambiguous_json',
                'verification_status': 'scope_ambiguous_not_accepted'}
    return None


def run_query(store, embedder, question: str, *, provider: str = 'offline',
              ticker: str | None = None, fiscal_year: str | None = None,
              as_of: str | None = None, top_k: int = 5, mode: str = 'hybrid',
              model: str = 'openai/gpt-oss-120b', curated_path: str | Path | None = None,
              automatic_path: str | Path | None = None,
              pdf_dir: str | Path = 'data/raw/research_reports',
              verify_financial: bool = True, verify_vision: bool = False,
              client=None, cross_encoder=None, pinecone_index: str | None = None,
              allow_curated: bool = True) -> dict[str, Any]:
    """Integrated V5 CLI query with safe V3/V4 handoff and V4 semantic RAG.

    The model/provider is unchanged; all uncertain numeric LLM claims go through
    the original PDF page reviewer when requested and provider is available.
    """
    docs = store.list_documents()
    eligible = {d.sha256 for d in docs if (not ticker or d.spec.ticker == ticker.upper())
                and (not fiscal_year or d.spec.fiscal_year == fiscal_year)
                and (not as_of or d.spec.published_on <= as_of)}
    curated = read_facts(curated_path) if allow_curated and curated_path and Path(curated_path).is_file() else []
    automatic = [f for f in load_auto_bank(automatic_path) if f["file_sha256"] in {d.sha256 for d in docs}] if automatic_path and Path(automatic_path).is_file() else []
    root = Path(pdf_dir)
    direct = choose_fact_answer(question, curated, automatic, ticker=ticker,
                                eligible_sha256=eligible, as_of=as_of,
                                pdf_dir=root, verify_sources=True)
    if direct is not None:
        return {**direct, 'retrieved_physical_pages': [], 'source_page_verification': None}

    # V6: source-rechecked financial-summary cards (no Groq or embeddings).
    # V3 and V4 were already tried above. Only exact company/segment/period
    # matches are allowed; source hash and page geometry are recomputed.
    panel_path = default_panel_bank_path()
    if panel_path.is_file():
        layout_result = answer_from_layout_panels(
            question, read_panel_bank(panel_path), pdf_dir=root,
            eligible_hashes=eligible, ticker=ticker, as_of=as_of,
            source_registry={d.sha256: d.spec.ticker for d in docs},
        )
        if layout_result is not None:
            return {**layout_result, 'retrieved_physical_pages': [],
                    'source_page_verification': None}

    if provider == 'offline':
        return {'question': question, 'answer': 'RAG fallback requires an available provider.',
                'status': 'not_evaluated', 'provider': 'offline',
                'route': 'financial_scope_reranked_document_rag', 'citations': [],
                'verification_status': 'not_evaluated', 'retrieved_physical_pages': []}

    pool = min(30, max(top_k * 5, 20))
    hits = search(store, question, embedder, ticker=ticker, fiscal_year=fiscal_year,
                  as_of=as_of, top_k=pool, mode=mode, pinecone_index=pinecone_index)
    hits = rerank_hits(question, hits, auto_facts=automatic, curated_facts=curated,
                       cross_encoder=cross_encoder, top_k=top_k)
    active = _provider_client(client) if provider == 'groq' else None
    reply = respond(question, hits, provider=provider, model=model,
                    client=active, facts_path=None,
                    structured_facts=structured_evidence(hits, auto_facts=automatic,
                                                         curated_facts=curated))
    result = _provider_status(reply.to_dict())
    out = {**result, 'route': 'financial_scope_reranked_document_rag',
           'verification_status': 'not_requested', 'source_page_verification': None,
           'retrieved_physical_pages': [(h.document.sha256, h.chunk.page_no) for h in hits]}
    if provider != 'groq' or not verify_financial or out['status'] not in {'answered', 'needs_review'} or not _has_numeric_claim(question, out.get('answer', '')):
        return out
    if not out.get('citations'):
        return {**out, 'status': 'needs_review', 'verification_status': 'missing_citation'}
    mapping = _source_map(store, curated, automatic, root)
    review = _verify_result(question, out, client=active, pdf_dir=root, file_map=mapping,
                            vision=verify_vision)
    out['source_page_verification'] = review
    out['verification_status'] = review['status']
    if review['verdict'] == 'contradicted':
        # A single corrective retrieval attempt, never a fabricated replacement.
        refined = search(store, question + ' exact financial line item segment reporting basis fiscal year unit',
                         embedder, ticker=ticker, fiscal_year=fiscal_year, as_of=as_of,
                         top_k=pool, mode=mode, pinecone_index=pinecone_index)
        refined = rerank_hits(question, refined, auto_facts=automatic,
                              curated_facts=curated, cross_encoder=cross_encoder, top_k=top_k)
        replacement = respond(question, refined, provider=provider, model=model, client=active,
                              facts_path=None,
                              structured_facts=structured_evidence(refined, auto_facts=automatic,
                                                                   curated_facts=curated))
        trial = _provider_status(replacement.to_dict())
        if trial['status'] in {'answered', 'needs_review'} and trial.get('citations'):
            checked = _verify_result(question, trial, client=active, pdf_dir=root,
                                     file_map=mapping, vision=verify_vision)
            if checked['verdict'] == 'supported':
                return {**trial, 'status': 'needs_review',
                        'note': 'Correction supported by AI page reviewer, not independently certified',
                        'route': 'rag_corrective_attempt', 'verification_status': checked['status'],
                        'source_page_verification': checked,
                        'retrieved_physical_pages': [(h.document.sha256, h.chunk.page_no) for h in refined]}
        out['status'] = 'needs_review'
        out['note'] = 'Original numeric answer contradicted; corrective retrieval unconfirmed'
    elif review['verdict'] != 'supported':
        out['status'] = 'needs_review'
    return out
