"""Financial safety regression tests; no real Groq or Pinecone calls."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from risklens.research.contracts import Chunk, Document, Hit, SourceSpec
from risklens.research.llm_guided_v7 import (
    _interpret_offline, _metric_match, _matching, _question_year,
    _source_year, _source_label_row, _unsupported_outcome,
    interpret, load_records, run_query,
)

ROOT = Path('data/raw/research_reports')
BANKS = Path('data/processed/research/real_acceptance/phase7_completion')
HDFC = 'hdfcbank_fy2025.pdf'
SHA_HDFC = '2a1879eee544526b7cecacb5380c2664f92e0d031361f5d54ffe55370c9487ea'


class GroqClient:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps(result)))])


class FakeStore:
    def __init__(self, documents):
        self.documents = documents

    def list_documents(self):
        return self.documents


def hdfc_document():
    spec = SourceSpec(company='HDFC Bank Limited', ticker='HDFCBANK',
                      fiscal_year='2024-25', title='HDFC Bank Annual Report FY2025',
                      source_url='https://www.hdfc.bank.in/annual-reports',
                      published_on='2025-07-25',
                      local_path='data/raw/research_reports/' + HDFC,
                      license_note='public annual report')
    return Document('hdfc_test_doc', spec, SHA_HDFC, 590, 'pymupdf')


def hdfc_hit():
    doc = hdfc_document()
    chunk = Chunk('hdfc_399', doc.doc_id, 399, 0, 200,
                  'Cost of Deposits3 5.21% 4.87%. Cost of Deposits is the ratio '
                  'of interest expense on deposits to daily average total deposits.')
    return Hit(chunk, doc, 0.91, ('lexical', 'dense'))


def good_intent(**changes):
    data = {'kind': 'numeric', 'ticker': 'HDFCBANK',
            'metric_label': 'cost of deposits', 'metric_code': 'cost_of_deposits',
            'segment': '', 'reporting_scope': 'unspecified',
            'period_end_year': 2025, 'unit': 'unspecified'}
    return {**data, **changes}


@pytest.mark.parametrize(('question', 'expected'), [
    ('What was net profit in FY2025?', 2025),
    ('What was net profit in FY2024-25?', 2025),
    ('What was net profit in 2023-24?', 2024),
    ('What were deposits at March 31, 2033?', 2033),
    ('What were deposits at March 2033?', 2033),
    ('What was the exact share price on December 18, 2027?', 2027),
    ('How is return on assets defined?', None),
])
def test_explicit_year_parser(question, expected):
    assert _question_year(question) == expected


@pytest.mark.parametrize(('year', 'expected'), [
    ('FY2024-25', 2025), ('FY2023-24', 2024),
    ('2025-03-31', 2025), ('FY2025', 2025),
])
def test_source_year_parser(year, expected):
    assert _source_year(year) == expected


@pytest.mark.parametrize('question', [
    'How is HDFC Bank cost of deposits defined?',
    'Which two income lines are added to form Reliance total income?',
    'Explain the cost of deposits ratio.',
])
def test_explanations_never_short_circuit_numeric(question):
    assert _interpret_offline(question).kind == 'explanation'


def test_cost_of_deposits_cannot_be_replaced_by_total_deposits():
    intent = _interpret_offline('What was HDFC Bank cost of deposits in FY2025?')
    wrong = {'metric': 'deposits', 'aliases': ['deposits', 'total deposits']}
    assert not _metric_match(intent, wrong, 'What was HDFC Bank cost of deposits in FY2025?')


def test_future_year_cannot_use_fy2025_deposits():
    question = 'What were HDFC Bank audited standalone deposits on March 31, 2033?'
    intent = _interpret_offline(question)
    result = _unsupported_outcome(question, intent, 2025)
    assert result['status'] == 'abstained'
    assert result['route'] == 'historical_period_guard'


def test_groq_misreads_requested_fy2033_year_but_literal_question_wins():
    client = GroqClient([good_intent(metric_label='deposits', metric_code='deposits',
                                    period_end_year=2025)])
    question = 'What were HDFC Bank deposits on March 31, 2033?'
    intent = interpret(question, provider='groq', client=client)
    assert intent.year == 2033
    assert intent.metric_code == 'deposits'


def test_groq_cannot_truncate_financial_metric_qualifier():
    question = 'What was HDFC Bank cost of deposits in FY2025?'
    client = GroqClient([good_intent(metric_label='total deposits',
                                    metric_code='deposits')])
    intent = interpret(question, provider='groq', client=client)
    assert intent.metric_label == ''
    assert intent.metric_code == ''


def test_groq_strict_json_format_is_required():
    client = GroqClient([good_intent()])
    interpret('What was HDFC Bank cost of deposits in FY2025?',
              provider='groq', client=client)
    response_format = client.calls[0]['response_format']
    assert response_format['type'] == 'json_schema'
    assert response_format['json_schema']['strict'] is True


def _have_original_hdfc() -> bool:
    return (ROOT / HDFC).is_file() and sha256((ROOT / HDFC).read_bytes()).hexdigest() == SHA_HDFC


@pytest.mark.skipif(not _have_original_hdfc(), reason='Original HDFC PDF unavailable')
def test_original_hdfc_cost_of_deposits_is_2025_value():
    q = 'What was HDFC Bank cost of deposits in FY2025?'
    result = _source_label_row(_interpret_offline(q), q, [hdfc_hit()], ROOT, {SHA_HDFC})
    assert result['status'] == 'answered'
    assert '5.21%' in result['answer']
    assert result['citations'][0]['pdf_page'] == 399
    assert '2025' in result['citations'][0]['source_year_header']


@pytest.mark.skipif(not _have_original_hdfc(), reason='Original HDFC PDF unavailable')
def test_original_hdfc_prior_year_not_confused():
    q = 'What was HDFC Bank cost of deposits in FY2024?'
    result = _source_label_row(_interpret_offline(q), q, [hdfc_hit()], ROOT, {SHA_HDFC})
    assert result['status'] == 'answered'
    assert '4.87%' in result['answer']
    assert '5.21%' not in result['answer']


def test_wrong_scope_segment_never_fallbacks_to_parent():
    q = 'What was Reliance Oil and Gas revenue in FY2025?'
    intent = _interpret_offline(q)
    fake = {'metric': 'revenue', 'aliases': ['revenue'], 'hash': 'h',
            'ticker': 'RIL', 'year': 2025, 'segment': '',
            'scope': 'consolidated', 'unit': 'crore'}
    assert not _matching(intent, fake, q, {'h'})


@pytest.mark.skipif(not (BANKS/'financial_facts_v3.json').is_file(),
                    reason='Phase 7 bank not installed')
def test_structured_banks_load_without_mutation():
    records = load_records(bank_dir=BANKS)
    assert {r['bank'] for r in records} == {'v3', 'v4', 'v6'}
    assert len(records) >= 37


@pytest.mark.skipif(not _have_original_hdfc() or not (BANKS/'financial_facts_v3.json').is_file(),
                    reason='Original source bank or report unavailable')
def test_end_to_end_cost_of_deposits_one_groq_call(monkeypatch):
    from risklens.research import llm_guided_v7 as module
    monkeypatch.setattr(module, 'search', lambda *a, **k: [hdfc_hit()])
    monkeypatch.setattr(module, 'rerank_hits', lambda *a, **k: [hdfc_hit()])
    client = GroqClient([good_intent()])
    answer = run_query(FakeStore([hdfc_document()]), None,
                       'What was HDFC Bank cost of deposits in FY2025?',
                       provider='groq', client=client, bank_dir=BANKS, pdf_dir=ROOT)
    assert answer['status'] == 'answered'
    assert '5.21%' in answer['answer']
    assert answer['route'] == 'llm_metric_exact_pdf_row'
    assert len(client.calls) == 1


@pytest.mark.skipif(not _have_original_hdfc() or not (BANKS/'financial_facts_v3.json').is_file(),
                    reason='Original source bank or report unavailable')
def test_end_to_end_explanation_routes_to_source_rag(monkeypatch):
    from risklens.research import llm_guided_v7 as module
    monkeypatch.setattr(module, 'search', lambda *a, **k: [hdfc_hit()])
    monkeypatch.setattr(module, 'rerank_hits', lambda *a, **k: [hdfc_hit()])
    client = GroqClient([
        good_intent(kind='definition', period_end_year=0),
        {'answer': 'Interest expense on deposits divided by daily average total deposits.',
         'evidence_ids': ['C0'], 'numeric_fact_ids': [], 'insufficient': False},
    ])
    q = 'Explain how HDFC Bank cost of deposits is calculated.'
    out = run_query(FakeStore([hdfc_document()]), None, q,
                    provider='groq', client=client, bank_dir=BANKS, pdf_dir=ROOT)
    assert out['status'] == 'answered'
    assert out['route'] == 'llm_supported_explanation'
    assert 'daily average' in out['answer']
    assert len(client.calls) == 2


@pytest.mark.skipif(not (BANKS/'financial_facts_v3.json').is_file(),
                    reason='Phase 7 bank not installed')
def test_explanation_cannot_answer_from_deposit_number(monkeypatch):
    from risklens.research import llm_guided_v7 as module
    monkeypatch.setattr(module, 'search', lambda *a, **k: [])
    monkeypatch.setattr(module, 'rerank_hits', lambda *a, **k: [])
    answer = run_query(FakeStore([hdfc_document()]), None,
                       'Why did HDFC Bank cost of deposits increase?',
                       provider='offline', bank_dir=BANKS, pdf_dir=ROOT,
                       top_k=1)
    assert answer['status'] != 'answered'


@pytest.mark.skipif(not (BANKS/'financial_facts_v3.json').is_file(),
                    reason='Phase 7 bank not installed')
def test_provider_failure_is_not_reported_as_evidence_absence():
    client = GroqClient([RuntimeError('RateLimitError')])
    answer = run_query(FakeStore([hdfc_document()]), None,
                       'Why did HDFC Bank cost of deposits increase?',
                       provider='groq', client=client, bank_dir=BANKS, pdf_dir=ROOT)
    assert answer['status'] == 'provider_unavailable'
    assert 'unavailable' in answer['route']


@pytest.mark.skipif(not (BANKS/'financial_facts_v3.json').is_file(),
                    reason='Phase 7 bank not installed')
def test_fy2033_abstains_even_if_groq_claims_fy2025():
    client = GroqClient([good_intent(metric_label='deposits', metric_code='deposits',
                                    period_end_year=2025)])
    answer = run_query(FakeStore([hdfc_document()]), None,
                       'What were HDFC Bank deposits on March 31, 2033?',
                       provider='groq', client=client, bank_dir=BANKS, pdf_dir=ROOT)
    assert answer['status'] == 'abstained'
    assert answer['route'] == 'historical_period_guard'
    assert len(client.calls) == 1


@pytest.mark.skipif(not _have_original_hdfc() or not (BANKS/'financial_facts_v3.json').is_file(),
                    reason='Original source bank or report unavailable')
def test_unknown_numeric_llm_claim_is_never_certified(monkeypatch):
    from risklens.research import llm_guided_v7 as module
    monkeypatch.setattr(module, 'search', lambda *a, **k: [hdfc_hit()])
    monkeypatch.setattr(module, 'rerank_hits', lambda *a, **k: [hdfc_hit()])
    client = GroqClient([
        good_intent(),
        {'answer': 'HDFC cost of deposits was 100%.', 'evidence_ids': ['C0'],
         'numeric_fact_ids': [], 'insufficient': False},
    ])
    result = run_query(FakeStore([hdfc_document()]), None,
                       'What was HDFC Bank cost of deposits FY2025?',
                       provider='groq', client=client, bank_dir=BANKS, pdf_dir=ROOT,
                       verify_financial=False)
    # A genuinely source-checked row overrides the bad LLM generated answer.
    assert result['status'] == 'answered'
    assert '5.21%' in result['answer']
    assert len(client.calls) == 1


@pytest.mark.skipif(not _have_original_hdfc(), reason='Original HDFC PDF unavailable')
def test_original_hdfc_consolidated_cannot_take_standalone_ratio():
    q = 'What was HDFC Bank consolidated cost of deposits in FY2025?'
    result = _source_label_row(_interpret_offline(q), q, [hdfc_hit()], ROOT, {SHA_HDFC})
    assert result is None


@pytest.mark.skipif(not _have_original_hdfc(), reason='Original HDFC PDF unavailable')
def test_hdfc_exact_label_row_rejects_different_metric():
    q = 'What was HDFC Bank interest on deposits in FY2025?'
    result = _source_label_row(_interpret_offline(q), q, [hdfc_hit()], ROOT, {SHA_HDFC})
    assert result is None


def test_invoice_year_not_inferred_from_bank_report_date():
    assert _question_year('What is the definition of capital adequacy?') is None


def test_bank_metric_label_prefers_exact_row_not_interest_or_savings():
    from risklens.research.llm_guided_v7 import _source_row_specific_enough
    intent = _interpret_offline('What were ICICI Bank deposits in FY2025?')
    for wrong in ('Interest on deposits', 'Savings bank deposits',
                  'Increase/(decrease) in deposits'):
        assert not _source_row_specific_enough(intent,
            {'bank': 'v4', 'label': wrong, 'metric': 'deposits'})
    assert _source_row_specific_enough(intent,
        {'bank': 'v4', 'label': 'TOTAL DEPOSITS', 'metric': 'deposits'})
