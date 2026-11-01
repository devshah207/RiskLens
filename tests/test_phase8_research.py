"""No paid APIs or originals: original Phase 7 integration interface is mocked."""
from __future__ import annotations

from contextlib import contextmanager
import sqlite3
import sys
from types import ModuleType, SimpleNamespace

import pytest

from app.services.filing_research import (
    ResearchPaths, _read_only_store, ask_saved_sources,
    normalize_response, safe_source_url,
)


def paths(tmp_path, *, archive=True, index=False):
    obj = ResearchPaths(tmp_path / "Archive.zip", tmp_path / "db.sqlite3",
                        tmp_path / "banks", tmp_path / "pdfs")
    if archive:
        obj.archive.touch()
    if index:
        obj.index.touch()
        obj.pdfs.mkdir()
        obj.banks.mkdir()
        for filename in ("financial_facts_v3.json", "financial_facts_auto_v4.json",
                         "financial_layout_panels_v6.json"):
            (obj.banks / filename).touch()
    return obj


def patch_xbrl(monkeypatch, direct, fallback=None):
    root = ModuleType("risklens")
    research = ModuleType("risklens.research")
    xbrl = ModuleType("risklens.research.xbrl_rag_v1")
    calls = []

    def lookup(question, **kwargs):
        calls.append(("lookup", question, kwargs))
        return direct

    def run(*args, **kwargs):
        calls.append(("fallback", args, kwargs))
        return fallback or {"status": "not_evaluated", "citations": []}

    xbrl.lookup_xbrl = lookup
    xbrl.run_query_xbrl_first = run
    for name, mod in (("risklens", root), ("risklens.research", research),
                      ("risklens.research.xbrl_rag_v1", xbrl)):
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def test_direct_xbrl_does_not_need_index_or_provider(monkeypatch, tmp_path):
    calls = patch_xbrl(monkeypatch, {
        "question": "X", "answer": "Example", "route": "xbrl_exact_fact",
        "status": "answered", "citations": [{"source_file": "original.xml"}],
    })
    res = ask_saved_sources("What was Infosys FY2025 consolidated net profit in crore?",
                            ticker="INFY", paths=paths(tmp_path))
    assert res["status"] == "answered"
    assert res["route"] == "xbrl_exact_fact"
    assert len(calls) == 1 and calls[0][0] == "lookup"


def test_missing_archive_fails_gracefully(tmp_path):
    res = ask_saved_sources("Sample", ticker="RIL", paths=paths(tmp_path, archive=False))
    assert res["status"] == "not_evaluated"
    assert res["route"] == "missing_original_xbrl_archive"


def test_groq_requires_consent_even_when_archive_exists(tmp_path):
    res = ask_saved_sources("Sample", ticker="INFY", paths=paths(tmp_path),
                            provider="groq", groq_allowed=False)
    assert res["route"] == "groq_consent_required"


def test_original_status_is_never_promoted():
    for status in ("answered", "needs_review", "abstained",
                   "provider_unavailable", "not_evaluated"):
        result = normalize_response({"status": status, "answer": "a", "citations": []})
        assert result["status"] == status
    assert normalize_response({"status": "unknown"})["status"] == "not_evaluated"


def test_citations_keep_source_units_and_pdf_pages():
    original = {"status": "needs_review", "citations": [
        {"pdf_page": 123, "unit": "crore", "scope": "standalone", "source_file": "a.pdf"},
        {"concept_qname": "facts:Revenue", "unit_measures": ["iso4217:INR"]},
    ]}
    assert normalize_response(original)["citations"] == original["citations"]


@pytest.mark.parametrize("source", ["javascript:alert(1)", "file:///tmp/a.pdf", "/local/a.pdf",
                                    "http://example.org", "https://user:pass@example.org", None])
def test_unsafe_source_url_rejected(source):
    assert safe_source_url(source) is None


def test_https_source_ok():
    assert safe_source_url("https://www.nseindia.com/companies-listing/corporate-integrated-filing")


def test_read_only_sqlite_no_schema_creation(monkeypatch, tmp_path):
    root = ModuleType("risklens")
    research = ModuleType("risklens.research")
    idx = ModuleType("risklens.research.index")

    class ResearchStore:
        def __init__(self, path):
            raise AssertionError("ResearchStore constructor must NOT run")
    idx.ResearchStore = ResearchStore
    for name, mod in (("risklens", root), ("risklens.research", research), ("risklens.research.index", idx)):
        monkeypatch.setitem(sys.modules, name, mod)
    db = tmp_path / "index.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE entries (id INTEGER)")
        c.execute("INSERT INTO entries VALUES (1)")
    with _read_only_store(db) as store:
        assert store.conn.execute("SELECT id FROM entries").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            store.conn.execute("INSERT INTO entries VALUES (2)")
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT count(*) FROM entries").fetchone()[0] == 1


def test_pdf_fallback_calls_existing_rag_with_offline_provider(monkeypatch, tmp_path):
    calls = patch_xbrl(monkeypatch, None, fallback={
        "answer": "uncertain", "status": "needs_review", "route": "llm_numeric_unverified",
        "citations": [{"pdf_page": 9, "source_file": "report.pdf"}],
    })
    @contextmanager
    def fake_store(_):
        yield SimpleNamespace(list_documents=lambda: [1])
    import app.services.filing_research as fr
    monkeypatch.setattr(fr, "_read_only_store", fake_store)
    res = ask_saved_sources("What was RIL FY2025 consolidated net profit in crore?",
                            ticker="RIL", paths=paths(tmp_path, index=True))
    assert res["status"] == "needs_review"
    assert calls[1][0] == "fallback"
    assert calls[1][2]["provider"] == "offline"
    assert calls[1][2]["verify_financial"] is True


def test_no_pdf_index_preserves_nondirect_nonanswer(monkeypatch, tmp_path):
    patch_xbrl(monkeypatch, None)
    res = ask_saved_sources("What was RIL FY2025 consolidated selling expense in crore?",
                            ticker="RIL", paths=paths(tmp_path))
    assert res["status"] == "not_evaluated"
    assert res["route"] == "pdf_index_missing"


def test_filter_conflicts_refuse_without_backend(monkeypatch, tmp_path):
    calls = patch_xbrl(monkeypatch, None)
    res = ask_saved_sources("What was INFY FY2024 consolidated net income?", ticker="INFY",
                            paths=paths(tmp_path), scope="standalone", year=2025)
    assert res["route"] == "scope_filter_conflict"
    assert calls == []


def test_company_conflict_is_not_silently_changed(monkeypatch, tmp_path):
    calls = patch_xbrl(monkeypatch, None)
    res = ask_saved_sources("What was ICICI Bank FY2025 standalone net profit?",
                            ticker="INFY", paths=paths(tmp_path))
    assert res["route"] == "issuer_filter_conflict"
    assert calls == []


def test_year_conflict_is_not_silently_changed(monkeypatch, tmp_path):
    calls = patch_xbrl(monkeypatch, None)
    res = ask_saved_sources("What was Infosys FY2024 standalone net profit?",
                            ticker="INFY", paths=paths(tmp_path), year=2025)
    assert res["route"] == "year_filter_conflict"
    assert calls == []


def test_fy2024_25_pair_is_valid_for_fy2025(monkeypatch, tmp_path):
    calls = patch_xbrl(monkeypatch, {"status": "answered", "answer": "mock", "citations": []})
    res = ask_saved_sources("What was Infosys FY2024-25 standalone net profit?",
                            ticker="INFY", paths=paths(tmp_path), year=2025)
    assert res["status"] == "answered"
    assert len(calls) == 1
