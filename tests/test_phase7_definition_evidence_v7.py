"""Regression checks for original-PDF definition evidence and V7 handoff."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json

import pymupdf
import pytest

from risklens.research.contracts import Document, SourceSpec
from risklens.research.definition_evidence_v7 import (
    DefinitionEvidence, find_definition_evidence, strict_definition_request,
    verified_definition_answer,
)
from risklens.research.llm_guided_v7 import _interpret_offline, run_query

ROOT = Path('data/raw/research_reports')
BANKS = Path('data/processed/research/real_acceptance/phase7_completion')
HDFC = 'hdfcbank_fy2025.pdf'
SHA_HDFC = '2a1879eee544526b7cecacb5380c2664f92e0d031361f5d54ffe55370c9487ea'


class FakeStore:
    def __init__(self, docs):
        self.docs = docs

    def list_documents(self):
        return self.docs


class FakeGroq:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps(value)))])


def _doc(pdf: Path, ticker: str, name: str, year: str = '2024-25') -> Document:
    digest = sha256(pdf.read_bytes()).hexdigest()
    with pymupdf.open(pdf) as pages:
        count = len(pages)
    spec = SourceSpec(company=name, ticker=ticker, fiscal_year=year,
                      title=f'{name} Annual Report',
                      source_url='https://example.org/annual-report.pdf',
                      published_on='2025-07-25', local_path=str(pdf),
                      license_note='original source test')
    return Document('test_' + digest[:12], spec, digest, count, 'pymupdf')


def _intent(question: str):
    return _interpret_offline(question)


def _have_hdfc() -> bool:
    path = ROOT / HDFC
    return path.is_file() and sha256(path.read_bytes()).hexdigest() == SHA_HDFC


@pytest.mark.parametrize(('q', 'want'), [
    ('How is HDFC Bank cost of deposits defined?', True),
    ('Give the formula for HDFC cost of deposits.', True),
    ('How did HDFC Bank cost of deposits change?', False),
    ('Why did HDFC cost of deposits increase?', False),
])
def test_only_literal_definition_requests_short_circuit(q, want):
    assert strict_definition_request(q) is want


@pytest.mark.skipif(not _have_hdfc(), reason='Original HDFC report not available')
def test_real_hdfc_definition_is_found_with_true_pdf_page():
    doc = _doc(ROOT / HDFC, 'HDFCBANK', 'HDFC Bank')
    found = find_definition_evidence(
        _intent('How is HDFC Bank cost of deposits defined?'), [doc], ROOT)
    result = verified_definition_answer('How is HDFC Bank cost of deposits defined?', found)
    assert result['status'] == 'answered'
    assert result['route'] == 'explicit_original_pdf_definition'
    assert result['answer'] == (
        'Cost of Deposits is the ratio of interest expense on deposits '
        'to daily average of total deposits.')
    assert result['citations'][0]['file_sha256'] == SHA_HDFC
    assert result['citations'][0]['pdf_page'] == 399


@pytest.mark.skipif(not _have_hdfc() or not (BANKS / 'financial_facts_v3.json').is_file(),
                    reason='Original HDFC report or existing JSON bank missing')
def test_end_to_end_real_pdf_needs_only_one_groq_intent_call():
    doc = _doc(ROOT / HDFC, 'HDFCBANK', 'HDFC Bank')
    client = FakeGroq([{
        'kind': 'definition', 'ticker': 'HDFCBANK',
        'metric_label': 'cost of deposits', 'metric_code': 'cost_of_deposits',
        'segment': '', 'reporting_scope': 'unspecified',
        'period_end_year': 0, 'unit': 'unspecified',
    }])
    answer = run_query(FakeStore([doc]), None,
                       'How is HDFC Bank cost of deposits defined?',
                       provider='groq', client=client, pdf_dir=ROOT, bank_dir=BANKS)
    assert answer['status'] == 'answered'
    assert answer['citations'][0]['pdf_page'] == 399
    assert 'interest expense on deposits' in answer['answer']
    assert client.calls == 1


@pytest.mark.skipif(not _have_hdfc() or not (BANKS / 'financial_facts_v3.json').is_file(),
                    reason='Original HDFC report or existing JSON bank missing')
def test_real_definition_available_even_if_groq_intent_rate_limited():
    doc = _doc(ROOT / HDFC, 'HDFCBANK', 'HDFC Bank')
    client = FakeGroq([RuntimeError('RateLimitError')])
    result = run_query(FakeStore([doc]), None,
                       'How is HDFC Bank cost of deposits defined?',
                       provider='groq', client=client, pdf_dir=ROOT, bank_dir=BANKS)
    assert result['status'] == 'answered'
    assert result['route'] == 'explicit_original_pdf_definition'
    assert client.calls == 1


def test_generic_new_company_definition_is_not_hardcoded(tmp_path: Path):
    pdf_path = tmp_path / 'new_bank.pdf'
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((60, 70), 'Definitions of Financial Ratios')
    page.insert_text((60, 95),
                     'Liquidity buffer ratio is the ratio of cash reserves to daily average liabilities.')
    doc.save(pdf_path)
    doc.close()
    source = _doc(pdf_path, 'TESTBANK', 'Test Bank')
    intent = SimpleNamespace(kind='definition', ticker='TESTBANK',
                             metric_label='liquidity buffer ratio')
    found = find_definition_evidence(intent, [source], tmp_path)
    result = verified_definition_answer('Define liquidity buffer ratio.', found)
    assert result['status'] == 'answered'
    assert result['citations'][0]['pdf_page'] == 1
    assert 'daily average liabilities' in result['answer']


def test_tampered_source_is_never_used(tmp_path: Path):
    pdf_path = tmp_path / 'test_report.pdf'
    doc = pymupdf.open()
    doc.new_page().insert_text((60, 70), 'Capital ratio is the ratio of capital to assets.')
    doc.save(pdf_path)
    doc.close()
    source = _doc(pdf_path, 'TESTBANK', 'Test Bank')
    with pdf_path.open('ab') as f:
        f.write(b'changed source after indexing')
    intent = SimpleNamespace(kind='definition', ticker='TESTBANK',
                             metric_label='capital ratio')
    found = find_definition_evidence(intent, [source], tmp_path)
    assert found == []
    assert verified_definition_answer('Define capital ratio.', found) is None


def test_no_definition_cannot_be_fabricated(tmp_path: Path):
    pdf_path = tmp_path / 'report.pdf'
    doc = pymupdf.open()
    doc.new_page().insert_text((60, 70),
                               'Capital ratio increased to 15 percent from 12 percent.')
    doc.save(pdf_path)
    doc.close()
    source = _doc(pdf_path, 'TESTBANK', 'Test Bank')
    intent = SimpleNamespace(kind='definition', ticker='TESTBANK',
                             metric_label='capital ratio')
    found = find_definition_evidence(intent, [source], tmp_path)
    assert len(found) == 1
    assert found[0].explicit_definition is None
    assert verified_definition_answer('Define capital ratio.', found) is None


def test_missing_ticker_does_not_guess_a_company():
    intent = SimpleNamespace(kind='definition', ticker='', metric_label='cost of deposits')
    assert find_definition_evidence(intent, [], ROOT) == []


def test_ambiguous_source_definitions_need_review(tmp_path: Path):
    pdf_path = tmp_path / 'two.pdf'
    doc = pymupdf.open()
    doc.new_page().insert_text((60, 70),
                               'Capital ratio is the ratio of capital to assets.')
    doc.new_page().insert_text((60, 70),
                               'Capital ratio is the ratio of reserves to liabilities.')
    doc.save(pdf_path)
    doc.close()
    source = _doc(pdf_path, 'TESTBANK', 'Test Bank')
    intent = SimpleNamespace(kind='definition', ticker='TESTBANK',
                             metric_label='capital ratio')
    found = find_definition_evidence(intent, [source], tmp_path)
    assert len(found) == 2
    assert all(isinstance(f, DefinitionEvidence) for f in found)
    result = verified_definition_answer('Define capital ratio.', found)
    assert result['status'] == 'needs_review'
    assert result['route'] == 'conflicting_original_pdf_definitions'
