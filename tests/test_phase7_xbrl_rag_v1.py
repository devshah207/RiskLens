"""Real-instance tests; source record matching uses no benchmark gold in answerer."""
import os
from pathlib import Path
import pytest

from risklens.research.xbrl_rag_v1 import lookup_xbrl, load_filings, run_query_xbrl_first
from risklens.research.llm_guided_v7 import run_query

ARCHIVE = Path(__file__).resolve().parents[3] / 'Archive(1).zip'
# Tests may run on another machine with the archive passed via the env var.
ARCHIVE = Path(os.environ.get('RISKLENS_XBRL_TEST_ARCHIVE', str(ARCHIVE)))

@pytest.fixture(scope='session')
def archive():
    if not ARCHIVE.is_file():
        pytest.skip('Original FY2025 eight-instance XML archive not provided')
    return ARCHIVE


def ask(q, archive, **kw):
    return lookup_xbrl(q, archive=archive, **kw)


def test_all_eight_original_company_xmls(archive):
    assert len(load_filings(str(archive))) == 8


def test_original_icici_reserves_source(archive):
    v = ask('How much were ICICI Bank standalone reserves and surplus on 31 March 2025, in thousands of rupees?', archive)
    assert v['route'] == 'xbrl_exact_fact'
    assert v['citations'][0]['concept_qname'].endswith('}ReservesAndSurplus')
    assert v['citations'][0]['raw_reported_value'] == '2885818600000'
    assert v['citations'][0]['period_kind'] == 'instant'
    assert v['citations'][0]['decimals'] == '-5'


def test_original_reliance_eps(archive):
    v=ask('What was the basic consolidated earnings per equity share of Reliance in FY2025, in rupees?',archive)
    assert '₹51.47 per share' in v['answer']
    assert v['citations'][0]['period_start']=='2024-04-01'


def test_original_reliance_noncontrolling_profit(archive):
    v=ask('What portion of Reliance consolidated FY2025 net profit was attributable to non-controlling interests, in crore?',archive)
    assert '11,661 crore' in v['answer']


def test_original_infosys_deferred_tax_liabilities(archive):
    v=ask('What were Infosys consolidated deferred tax liabilities (net) as of March 2025, in crore?',archive)
    assert '1,722 crore' in v['answer']
    assert v['citations'][0]['period_kind']=='instant'

@pytest.mark.parametrize('question',[
    'How much value of sales, excluding income from services, did Reliance report in its FY2025 consolidated profit and loss statement, in crore?',
    'What was HDFC Bank standalone net interest income as a percentage of working funds at March 2024?',
    'What were Infosys standalone selling and marketing expenses for FY2025, in crore?',
    'What were Infosys consolidated cost of sales for FY2025, in crore?',
    'What was HDFC Bank standalone profit per employee for March 2025, in crore?',
    'What was Reliance consolidated total income for FY2034, in crore?',
    'What was the definition of total income in Reliance consolidated FY2025 reports?',
    'What was Reliance consolidated FY2025 revenue from operations excluding services, in crore?',
    'What were Infosys consolidated deferred tax liabilities (net) as of March 2026, in crore?',
    'What were Infosys consolidated deferred tax liabilities (net) as of March 2025, in percent?',
    'What were Infosys deferred tax liabilities (net) as of March 2025, in crore?',  # scope missing
    'What were Infosys standalone and consolidated deferred tax liabilities (net) as of March 2025, in crore?',
])
def test_unsafe_lookups_fall_back(question, archive):
    assert ask(question,archive) is None


def test_scope_changes_amount(archive):
    a=ask('What was Infosys standalone revenue from operations for FY2025, in crore?',archive)
    b=ask('What was Infosys consolidated revenue from operations for FY2025, in crore?',archive)
    assert '136,592 crore' in a['answer']
    assert '162,990 crore' in b['answer']


def test_annual_not_quarterly(archive):
    v=ask('What was Infosys standalone revenue from operations for FY2025, in crore?',archive)
    assert v['citations'][0]['period_start']=='2024-04-01'
    assert v['citations'][0]['period_end']=='2025-03-31'


def test_as_of_avoids_future_publication(archive):
    q='What were Infosys consolidated deferred tax liabilities (net) as of March 2025, in crore?'
    assert ask(q,archive,as_of='2025-04-01') is None
    assert ask(q,archive,as_of='2025-04-30') is not None


def test_ticker_hint_cannot_override_question(archive):
    assert ask('What was Infosys consolidated revenue from operations for FY2025, in crore?',archive,ticker='RIL') is None


def test_integration_returns_before_old_store_or_groq(archive):
    class NoStore:
        def list_documents(self):
            raise AssertionError('Must not enter old PDF route on an exact XML match')
    v=run_query_xbrl_first(NoStore(),None,'What was Infosys standalone revenue from operations for FY2025, in crore?',provider='offline',xbrl_archive=archive)
    assert v['route']=='xbrl_exact_fact'


def test_integration_fallback_unchanged(archive):
    class EmptyStore:
        def list_documents(self):
            return []
    q='What were Infosys standalone selling and marketing expenses for FY2025, in crore?'
    v=run_query_xbrl_first(EmptyStore(),None,q,provider='offline',xbrl_archive=archive)
    assert v['route']=='empty_index'  # old route reached; not replaced by XBRL guessing


def test_integration_without_xbrl_unchanged():
    class EmptyStore:
        def list_documents(self):
            return []
    q='What were Infosys standalone selling and marketing expenses for FY2025, in crore?'
    assert run_query(EmptyStore(),None,q,provider='offline')['route']=='empty_index'


def test_missing_archive_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError):
        ask('What was Infosys standalone revenue from operations for FY2025, in crore?',tmp_path/'missing.zip')


def test_additive_wrapper_source_first_without_touching_existing_engine(archive):
    from risklens.research.xbrl_rag_v1 import run_query_xbrl_first
    class NoStore:
        def list_documents(self):
            raise AssertionError('Old RAG must not be needed for exact XBRL data')
    q='What were ICICI Bank standalone reserves and surplus on 31 March 2025, in thousands of rupees?'
    assert run_query_xbrl_first(NoStore(),None,q,xbrl_archive=archive)['route']=='xbrl_exact_fact'


def test_additive_wrapper_uses_original_rag_on_miss(archive):
    from risklens.research.xbrl_rag_v1 import run_query_xbrl_first
    class EmptyStore:
        def list_documents(self):return []
    q='What were Infosys standalone selling and marketing expenses for FY2025, in crore?'
    assert run_query_xbrl_first(EmptyStore(),None,q,xbrl_archive=archive,provider='offline')['route']=='empty_index'


def test_xbrl_revenue_cannot_answer_value_of_sales_even_with_parenthetical_revenue(archive):
    assert ask('What was Reliance consolidated value of sales (revenue from operations) in FY2025, in crore?',archive) is None


def test_fiscal_year_range_ends_in_2025(archive):
    v=ask('What was Infosys standalone revenue from operations for FY2024-25, in crore?',archive)
    assert v is not None and '136,592 crore' in v['answer']

def test_nonconsolidated_not_assumed_consolidated(archive):
    assert ask('What was Infosys non-consolidated revenue from operations for FY2025, in crore?',archive) is None

def test_continuing_operations_profit_not_substituted(archive):
    assert ask('What was Reliance standalone net profit from continuing operations in FY2025, in crore?',archive) is None
