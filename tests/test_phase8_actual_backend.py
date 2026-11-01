"""Real Phase 7 imports and routing with synthetic documents, no private sources.

These tests DO execute the unchanged published XBRL parser and PDF RAG.
Amounts and issuer filings in this test are FICTIONAL, never benchmark evidence.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import zipfile

from app.services import filing_research as adapter
from risklens.research.chunking import make_chunks
from risklens.research.contracts import Document, Page, SourceSpec, source_document_id
from risklens.research.embeddings import HashFixtureEmbedder
from risklens.research.index import ResearchStore


def synthetic_xbrl_archive(tmp_path: Path) -> Path:
    """Produce eight deliberately fictional but well-formed XBRL 2.1 instances."""
    archive = tmp_path / "synthetic_eight_filings.zip"
    with zipfile.ZipFile(archive, 'w') as z:
        for company in ('RELIANCE', 'HDFCBANK', 'INFY', 'ICICIBANK'):
            for scope in ('standalone', 'consolidated'):
                xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"
   xmlns:demo="https://synthetic.invalid/xbrl"
   xmlns:iso4217="http://www.xbrl.org/2003/iso4217">
  <xbrli:context id="period">
    <xbrli:entity><xbrli:identifier scheme="https://synthetic.invalid/company">{company}</xbrli:identifier></xbrli:entity>
    <xbrli:period><xbrli:startDate>2024-04-01</xbrli:startDate><xbrli:endDate>2025-03-31</xbrli:endDate></xbrli:period>
  </xbrli:context>
  <xbrli:context id="instant">
    <xbrli:entity><xbrli:identifier scheme="https://synthetic.invalid/company">{company}</xbrli:identifier></xbrli:entity>
    <xbrli:period><xbrli:instant>2025-03-31</xbrli:instant></xbrli:period>
  </xbrli:context>
  <xbrli:unit id="inr"><xbrli:measure>iso4217:INR</xbrli:measure></xbrli:unit>
  <demo:Symbol contextRef="instant">{company}</demo:Symbol>
  <demo:NatureOfReportStandaloneConsolidated contextRef="instant">{scope}</demo:NatureOfReportStandaloneConsolidated>
  <demo:DateOfStartOfFinancialYear contextRef="instant">2024-04-01</demo:DateOfStartOfFinancialYear>
  <demo:DateOfEndOfFinancialYear contextRef="instant">2025-03-31</demo:DateOfEndOfFinancialYear>
  <demo:WhetherResultsAreAuditedOrUnaudited contextRef="instant">Audited</demo:WhetherResultsAreAuditedOrUnaudited>
  <demo:RevenueFromOperations contextRef="period" unitRef="inr" decimals="0">1230000000</demo:RevenueFromOperations>
</xbrli:xbrl>
'''
                z.writestr(f'{company}_{scope}_30042025120000_WEB.xml', xml)
    return archive


def test_real_original_xbrl_parser_and_adapter_without_index(tmp_path):
    archive = synthetic_xbrl_archive(tmp_path)
    paths = adapter.ResearchPaths(archive, tmp_path / 'missing.sqlite3', tmp_path/'banks', tmp_path/'pdfs')
    answer = adapter.ask_saved_sources(
        'What was Infosys consolidated revenue from operations for FY2025, in crore?',
        ticker='INFY', scope='consolidated', year=2025, paths=paths,
    )
    assert answer['status'] == 'answered'
    assert answer['route'] == 'xbrl_exact_fact'
    assert '₹123 crore' in answer['answer']
    assert answer['citations'][0]['period_start'] == '2024-04-01'
    assert answer['citations'][0]['period_end'] == '2025-03-31'
    assert answer['citations'][0]['source_kind'] == 'NSE_XBRL_XML'
    assert not paths.index.exists()  # No PDF store needed or created by direct XBRL.


def synthetic_pdf_index(tmp_path):
    """Build an index with the genuine Phase 7 indexer, synthetic PDFs and vectors."""
    import pymupdf
    pdfs = tmp_path / 'pdfs'
    pdfs.mkdir()
    source_path = pdfs/'fictional_ril.pdf'
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((40, 70), 'FICTIONAL annual-report text for routing tests only.')
        page.insert_text((40, 100), 'Reliance crude oil price volatility is an illustrative risk.', fontsize=10)
        pdf.save(source_path)
    identity = sha256(source_path.read_bytes()).hexdigest()
    spec = SourceSpec(
        company='Reliance Industries (FICTIONAL TEST)', ticker='RIL',
        fiscal_year='2024-25', title='SYNTHETIC REPORT -- NOT NSE FINANCIAL DATA',
        source_url='https://synthetic.invalid/example.pdf',
        published_on='2025-07-01', local_path=str(source_path),
        license_note='Local synthetic fixture, not real audited facts',
    )
    doc_id = source_document_id(spec, identity)
    doc = Document(doc_id, spec, identity, 1, 'synthetic')
    p = Page(doc_id, 1,
             'FICTIONAL annual report: Reliance crude oil price volatility is an illustrative risk.',
             'synthetic')
    db = tmp_path/'index.sqlite3'
    embedder = HashFixtureEmbedder()
    with ResearchStore(db) as store:
        store.insert(doc, [p], make_chunks(p), embedder)
    banks = tmp_path/'banks'
    banks.mkdir()
    (banks/'financial_facts_v3.json').write_text(json.dumps({'schema_version':1,'facts':[]}))
    (banks/'financial_facts_auto_v4.json').write_text(json.dumps({'schema_version':4,'facts':[]}))
    (banks/'financial_layout_panels_v6.json').write_text(json.dumps({
        'schema_version':6,'facts':[], 'sources':[{'file_sha256': identity}],
    }))
    return db, pdfs, banks


def test_original_pdf_fallback_runs_on_existing_index_without_writes(monkeypatch, tmp_path):
    archive = synthetic_xbrl_archive(tmp_path)
    db, pdfs, banks = synthetic_pdf_index(tmp_path)
    monkeypatch.setattr(adapter, '_lazy_embedder', HashFixtureEmbedder)
    before = sha256(db.read_bytes()).hexdigest()
    paths = adapter.ResearchPaths(archive, db, banks, pdfs)
    question = 'Which commodities does Reliance identify as subject to price volatility in FY2025?'
    answer = adapter.ask_saved_sources(question, ticker='RIL', year=2025, paths=paths)
    # Local fallback uses the genuine Phase 7 research store, retrieval and
    # llm_guided_v7.run_query, but without Groq it cannot invent a final answer.
    assert answer['route'] == 'llm_guided_rag_not_run', answer
    assert answer['status'] == 'not_evaluated'
    assert answer['retrieved_physical_pages']
    assert answer['retrieved_physical_pages'][0][1] == 1
    assert sha256(db.read_bytes()).hexdigest() == before


def test_missing_original_fact_banks_reported_before_pdf_backend(tmp_path):
    archive = synthetic_xbrl_archive(tmp_path)
    db, pdfs, banks = synthetic_pdf_index(tmp_path)
    (banks/'financial_layout_panels_v6.json').unlink()
    paths = adapter.ResearchPaths(archive, db, banks, pdfs)
    answer = adapter.ask_saved_sources(
        'Which commodities does Reliance discuss as risks in FY2025?',
        ticker='RIL', year=2025, paths=paths,
    )
    assert answer['status'] == 'not_evaluated'
    assert answer['route'] == 'missing_original_phase7_banks'

