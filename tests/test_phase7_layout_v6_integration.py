"""Real installed CLI-route test: V6 direct facts must bypass Groq and embeddings."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from risklens.research.combined_v5 import run_query


BANK = Path('data/processed/research/real_acceptance/phase7_completion/financial_layout_panels_v6.json')
PDF_DIR = Path('data/raw/research_reports')


@pytest.fixture(scope='module')
def sources():
    if not BANK.is_file() or not PDF_DIR.exists():
        pytest.skip('V6 real-PDF bank not yet installed')
    return json.loads(BANK.read_text())['sources']


def _fakestore(ticker, specs):
    sha = next(s['file_sha256'] for s in specs if s['ticker'] == ticker)

    class FakeStore:
        def list_documents(self):
            return [SimpleNamespace(sha256=sha, spec=SimpleNamespace(
                ticker=ticker, fiscal_year='2024-25', published_on='2025-08-07'))]

    class NoEmbedding:
        def encode(self, texts):
            raise AssertionError('PDF-backed direct answer must not call embeddings')

    return FakeStore(), NoEmbedding()


@pytest.mark.parametrize('ticker,question,expected', [
    ('RIL', 'Reliance Oil and Gas EBITDA FY2025 in crore?', '21,188 crore'),
    ('RIL', 'Reliance Media and Entertainment revenue FY2025 in crore?', '20,696 crore'),
    ('RIL', 'Reliance Oil and Gas revenue FY2025 in crore?', '25,211 crore'),
    ('HDFCBANK', 'HDFC Bank FY2025 earnings per share in rupees?', '₹88.3'),
    ('HDFCBANK', 'HDFC Bank cost-to-income ratio FY2025?', '40.5%'),
    ('HDFCBANK', 'HDFC Bank FY2025 gross NPA ratio?', '1.33%'),
    ('HDFCBANK', 'HDFC Bank total net revenue FY2025 in crore?', '168,302.4 crore'),
    ('INFY', 'Infosys FY2025 basic earnings per share in rupees?', '₹64.5'),
    ('INFY', 'Infosys consolidated cash and investments FY2025 in crore?', '47,549 crore'),
    ('INFY', 'Infosys standalone revenue from operations FY2025 in crore?', '136,592 crore'),
    ('INFY', 'Infosys consolidated revenue from operations FY2024 in crore?', '153,670 crore'),
    ('INFY', 'Infosys FY2025 consolidated goodwill and other intangible assets in crore?', '12,872 crore'),
])
def test_v6_direct_cli_path_no_groq_no_embedding(sources, ticker, question, expected):
    store, embedder = _fakestore(ticker, sources)
    answer = run_query(store, embedder, question, provider='groq', ticker=ticker,
                       curated_path=None, automatic_path=None, pdf_dir=PDF_DIR)
    assert answer['status'] == 'answered', answer
    assert answer['route'] == 'spatial_financial_panel_json', answer
    assert expected in answer['answer'], answer
    assert answer['citations'][0]['file_sha256'] in {s['file_sha256'] for s in sources}


def test_v6_no_wrong_company_value(sources):
    store, embedder = _fakestore('RIL', sources)
    result = run_query(store, embedder, 'Reliance Oil and Gas consolidated revenue FY2025?',
                       provider='offline', ticker='RIL', curated_path=None, automatic_path=None,
                       pdf_dir=PDF_DIR)
    assert result['status'] == 'not_evaluated'  # Cannot turn segment into consolidated revenue.
