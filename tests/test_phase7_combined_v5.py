"""V5 regression tests: real-PDF source checks and adversarial scope handling."""
from __future__ import annotations
import json
from pathlib import Path
import pytest

from risklens.research.financial_facts import read_facts
from risklens.research.combined_v5 import choose_fact_answer, explicit_segments, _unit_requested

@pytest.fixture(scope='module')
def banks():
    return (read_facts('data/processed/research/real_acceptance/phase7_completion/financial_facts_v3.json'),
            json.loads(Path('data/processed/research/real_acceptance/phase7_completion/financial_facts_auto_v4.json').read_text())['facts'])


def test_segment_parser_catches_ril_oil_gas():
    assert explicit_segments('Reliance Oil and Gas revenue FY2025') == {'oil and gas'}
    assert explicit_segments('Reliance Media & Entertainment revenue FY2025') == {'media and entertainment'}


def test_no_wrong_company_revenue_for_oil_and_gas(banks):
    curated, auto = banks
    result = choose_fact_answer('What was Reliance Oil and Gas revenue in FY2025, in crore?', curated, auto,
                                ticker='RIL', verify_sources=False)
    assert result is None or result['status'] != 'answered'
    if result:
        assert '557,163' not in result['answer'] and '1,071,174' not in result['answer']


def test_v3_trusted_digital_services(banks):
    curated, auto = banks
    result = choose_fact_answer('What was Reliance Digital Services revenue in FY2025 in crore?', curated, auto,
                                ticker='RIL', verify_sources=False)
    assert result and result['route'] == 'v3_curated_json'
    assert '154,119' in result['answer']


def test_v4_auto_matches_units_exactly(banks):
    curated, auto = banks
    q = 'What were ICICI Bank standalone operating expenses for FY2025 in thousands of rupees?'
    result = choose_fact_answer(q, curated, auto, ticker='ICICIBANK', verify_sources=False)
    assert result and result['route'] == 'v4_native_table_json'
    assert '423,723,243 thousand' in result['answer']
    assert 'crore' not in result['answer']
    assert result['status'] == 'needs_review'  # no PDF check => NOT trusted


def test_no_silent_unit_change(banks):
    curated, auto = banks
    q = 'What were ICICI Bank standalone cash and balances with RBI at March 2025, in thousands of rupees?'
    result = choose_fact_answer(q, curated, auto, ticker='ICICIBANK', verify_sources=False)
    assert result and '1,199,281,192 thousand' in result['answer']


def test_prior_year_source_column(banks):
    curated, auto = banks
    q = 'What was ICICI Bank standalone net profit for FY2024 in thousands of rupees?'
    result = choose_fact_answer(q, curated, auto, ticker='ICICIBANK', verify_sources=False)
    assert result and '408,882,694 thousand' in result['answer']
    assert result['citations'][0]['reported_period'] == 'FY2023-24'


def test_future_question_refuses_without_groq(banks):
    curated, auto = banks
    result = choose_fact_answer('What will HDFC Bank gross NPA ratio be on 31 March 2031?', curated, auto,
                                ticker='HDFCBANK', verify_sources=False)
    assert result['status'] == 'abstained' and result['route'] == 'deterministic_future_refusal'


def test_v4_bad_cell_cannot_answer(banks, monkeypatch):
    curated, auto = banks
    import risklens.research.combined_v5 as v5
    modified = [dict(f) for f in auto]
    for f in modified:
        if f['metric'] == 'cash_rbi' and f['status'] == 'auto_anchored' and f['observed_period'] == 'FY2024-25':
            f['value'] = '999999999'
    def fail_confirm(*a,**kw):
        raise ValueError('Original PDF table cell mismatch')
    monkeypatch.setattr(v5, 'confirm_original_table_rows', fail_confirm)
    result = v5.choose_fact_answer('What were ICICI Bank standalone cash and balances with RBI at March 2025 in thousands of rupees?',
                                   curated, modified, ticker='ICICIBANK', pdf_dir='data/raw/research_reports')
    assert result is None or result['status'] != 'answered'


def test_source_verified_v4_answer_promoted(banks, monkeypatch):
    curated, auto = banks
    import risklens.research.combined_v5 as v5
    monkeypatch.setattr(v5, 'confirm_original_table_rows', lambda *args, **kwargs: {'rows_verified_against_original_table': 1})
    result = v5.choose_fact_answer('What was HDFC Bank return on assets in FY2025 as a percentage?',
                                  curated, auto, ticker='HDFCBANK', pdf_dir='data/raw/research_reports')
    assert result and result['route'] == 'v4_native_table_json' and result['status'] == 'answered'
    assert '1.91%' in result['answer']
    assert result['verification_status'] == 'native_row_column_year_scope_checked_not_human_certified'


def test_reported_unit_recognizer():
    assert _unit_requested('in thousands of rupees') == 'thousand_rupees'
    assert _unit_requested('in crore') == 'crore'


def test_curated_wrong_source_refused(banks, monkeypatch):
    curated, auto = banks
    import risklens.research.combined_v5 as v5
    monkeypatch.setattr(v5, 'validate_original_pdfs', lambda *args, **kwargs: (_ for _ in ()).throw(ValueError('SHA mismatch')))
    res = v5.choose_fact_answer('Reliance Digital Services revenue FY2025 in crore', curated, auto,
                                ticker='RIL', pdf_dir='data/raw/research_reports')
    assert res['status'] == 'needs_review' and res['verification_status'] == 'source_recheck_failed'


def test_cli_combined_direct_answer_needs_no_embedding_or_groq(banks):
    """Real PDF-backed direct fact must bypass API client and vector model."""
    from types import SimpleNamespace
    from risklens.research.combined_v5 import run_query
    sha = next(f.file_sha256 for f in banks[0] if f.ticker == 'RIL')
    class FakeStore:
        def list_documents(self):
            return [SimpleNamespace(sha256=sha, spec=SimpleNamespace(ticker='RIL',
                    fiscal_year='2024-25', published_on='2025-08-07'))]
    class NoEmbedder:
        def encode(self, texts):
            raise AssertionError('Embedding should not load for source-verified direct answer')
    result=run_query(FakeStore(), NoEmbedder(), 'What was Reliance Digital Services revenue in FY2025 in crore?',
                     provider='groq', ticker='RIL',
                     curated_path='data/processed/research/real_acceptance/phase7_completion/financial_facts_v3.json',
                     automatic_path='data/processed/research/real_acceptance/phase7_completion/financial_facts_auto_v4.json',
                     pdf_dir='data/raw/research_reports')
    assert result['status'] == 'answered' and result['route'] == 'v3_curated_json'
    assert '154,119' in result['answer']


def test_other_index_does_not_import_financial_facts_from_unrelated_company(banks):
    """The full-bank JSON must not cause a synthetic or limited index to error."""
    from types import SimpleNamespace
    from risklens.research.combined_v5 import run_query
    class FakeStore:
        def list_documents(self):
            return [SimpleNamespace(sha256='ab'*32, spec=SimpleNamespace(ticker='MOCK',
                    fiscal_year='2024-25', published_on='2025-08-07'))]
    result=run_query(FakeStore(), object(), 'What was the illustrative revenue?',
                     provider='offline', ticker='MOCK',
                     curated_path='data/processed/research/real_acceptance/phase7_completion/financial_facts_v3.json',
                     automatic_path='data/processed/research/real_acceptance/phase7_completion/financial_facts_auto_v4.json',
                     pdf_dir='data/raw/research_reports')
    assert result['status'] == 'not_evaluated'
    assert result['route'] == 'financial_scope_reranked_document_rag'


def test_future_plan_question_still_searches_reports(banks):
    curated, auto = banks
    result = choose_fact_answer('What does Reliance plan for 2030?', curated, auto,
                                ticker='RIL', verify_sources=False)
    assert result is None  # route to narrative RAG rather than categorical rejection
