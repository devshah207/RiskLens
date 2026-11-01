"""Extra source-backed questions NOT selected from the original 24-case benchmark.

These are coverage/regression checks, not a random independent blinded sample.
They run against original report PDFs with no Groq calls or embedding inference.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from risklens.research.layout_panels_v6 import answer_from_layout_panels, read_panel_bank


ROOT = Path('data/processed/research/real_acceptance/phase7_completion')
BANK = ROOT / 'financial_layout_panels_v6.json'
PDF_DIR = Path('data/raw/research_reports')


@pytest.fixture(scope='module')
def bank():
    if not BANK.is_file() or not PDF_DIR.is_dir():
        pytest.skip('Original real-PDF V6 bank is not installed in this repository')
    return read_panel_bank(BANK)


@pytest.mark.parametrize('ticker,question,expected', [
    ('RIL', 'Reliance Oil to Chemicals EBITDA FY2025 in crore?', '54,988 crore'),
    ('RIL', 'Reliance Retail revenue FY2025 in crore?', '330,943 crore'),
    ('RIL', 'Reliance Retail EBITDA FY2025 in crore?', '25,094 crore'),
    ('RIL', 'Reliance Media and Entertainment EBITDA FY2025?', '1,833 crore'),
    ('HDFCBANK', 'HDFC Bank return on equity FY2025 percentage?', '14.6%'),
    ('HDFCBANK', 'HDFC Bank dividend per share FY2025 in rupees?', '₹22'),
    ('HDFCBANK', 'HDFC Bank deposits FY2025 in crore?', '2,714,715 crore'),
    ('HDFCBANK', 'HDFC Bank profit after tax FY2025 in crore?', '67,347 crore'),
    ('INFY', 'Infosys consolidated total income FY2025 in crore?', '166,590 crore'),
    ('INFY', 'Infosys consolidated net current assets FY2025 in crore?', '54,249 crore'),
    ('INFY', 'Infosys standalone profit after tax FY2024 in crore?', '27,234 crore'),
    ('INFY', 'Infosys consolidated total assets FY2025 in crore?', '148,903 crore'),
    ('INFY', 'Infosys consolidated goodwill and other intangible assets FY2024?', '8,700 crore'),
])
def test_extra_source_metrics(bank, ticker, question, expected):
    answer = answer_from_layout_panels(question, bank, pdf_dir=PDF_DIR, ticker=ticker)
    assert answer is not None and answer['status'] == 'answered', (question, answer)
    assert expected in answer['answer'], (question, answer)
    assert answer['verification_status'].startswith('pdf_sha_')


@pytest.mark.parametrize('ticker,question', [
    ('RIL', 'Reliance Digital Services revenue FY2024 in crore?'),
    ('RIL', 'Reliance Oil and Gas consolidated revenue FY2025?'),
    ('RIL', 'Reliance Oil and Gas EBITDA FY2025 in USD?'),
    ('HDFCBANK', 'HDFC Bank profit after tax FY2023 in crore?'),
    ('INFY', 'Infosys standalone profit after tax FY2026 in crore?'),
])
def test_extra_guardrails(bank, ticker, question):
    assert answer_from_layout_panels(question, bank, pdf_dir=PDF_DIR, ticker=ticker) is None
