"""V4 regression suite: general financial table extraction, reranking, verification.

No expected answers from the 24-question V4 blind set are used in these tests.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from risklens.research.financial_auto_v4 import (numeric_cell, table_headers,
                                                  canonical_metric, load_auto_bank)
from risklens.research.financial_scope_v4 import (rerank_hits,
                                                   select_automatic, automatic_fact_answer)
from risklens.research.structured_pipeline_v4 import _verify_result
from risklens.research.answer import respond
from risklens.research.contracts import Document, SourceSpec, Chunk, Hit, ResearchError


def sample_doc(sha, company, ticker):
    spec = SourceSpec(company=company, ticker=ticker, fiscal_year="2024-25",
                      title="Test original annual report", published_on="2025-08-01",
                      source_url="https://example.com/annual.pdf", local_path="dummy.pdf",
                      license_note="test fixture")
    return Document(doc_id="f"*24, spec=spec, sha256=sha, pages=100, parser="pymupdf")


def hit(doc, page, text, ident, score=.02):
    return Hit(Chunk(ident*24, doc.doc_id, page, 0, len(text), text), doc, score, ("hybrid",))


def source_auto(val="5000", period="FY2024-25", status="auto_anchored", scope="standalone", page=4):
    return {"fact_id":"a"*24,"ticker":"HDFCBANK", "company":"HDFC Bank Limited",
            "source_file":"hdfc.pdf", "file_sha256":"a"*64, "source_url":"https://example.com/a.pdf",
            "published_on":"2025-08-01", "report_fiscal_year":"2024-25",
            "physical_pdf_page":page, "table_index":0,"row_index":1,"column_index":1,
            "row_label":"Deposits", "column_label":"At March 31, 2025", "metric":"deposits",
            "segment":"","reporting_scope":scope,"observed_period":period,"value":val,
            "source_literal":val,"currency":"INR","unit":"crore","status":status,
            "reason":"native test fixture"}


def test_table_year_headers_map_values_not_adjacent_rows():
    rows = [["Particulars", "At March 31, 2025", "At March 31, 2024"],
            ["Deposits", "5000", "4000"], ["Advances", "3000", "2000"]]
    parsed = table_headers(rows)
    assert parsed[1][0] == "FY2024-25" and parsed[2][0] == "FY2023-24"
    assert numeric_cell(rows[1][1]) == ("5000", "")
    assert numeric_cell("5000\n4000") is None
    assert canonical_metric("Gross NPA to Gross Advances") == "gross_npa_ratio"


def test_scope_and_period_must_match_for_auto_json():
    current = source_auto()
    past = {**current, "fact_id":"b"*24,"value":"4000", "source_literal":"4000", "observed_period":"FY2023-24"}
    selection, meta = select_automatic("What were HDFC Bank standalone deposits in FY2025?", [past,current])
    assert meta["complete"] and selection[0]["value"] == "5000"
    past_selection, _ = select_automatic("What were HDFC Bank standalone deposits in FY2024?", [past,current])
    assert past_selection[0]["value"] == "4000"
    assert select_automatic("What were HDFC Bank consolidated deposits in FY2025?", [past,current])[0] == []


def test_unreviewed_fact_cannot_be_promoted():
    selected, reason = select_automatic("HDFC Bank standalone deposits FY2025?", [source_auto(status="review_required")])
    assert selected == []
    assert automatic_fact_answer("What were HDFC Bank standalone deposits FY2025?", [source_auto()])["status"] == "needs_review"


def test_conflicting_reported_amounts_cannot_be_selected():
    a, b = source_auto(), source_auto(val="9000",page=5)
    assert select_automatic("HDFC Bank standalone deposits FY2025?", [a,b])[0] == []


def test_source_hash_filter_blocks_other_pdf():
    assert select_automatic("HDFC Bank standalone deposits FY2025?", [source_auto()], source_hashes={"0"*64})[0] == []


def test_reranker_rewards_matching_segment_and_scope():
    doc = sample_doc("a"*64, "Reliance Industries Limited", "RIL")
    wrong = hit(doc, 40, "Standalone revenue 5,57,163", "w", .030)
    right = hit(doc, 3, "Digital Services revenue 1,54,119", "r", .010)
    class F:
        def __init__(self, page, seg, scope):
            self.file_sha256 = doc.sha256
            self.pdf_page = page
            self.segment = seg
            self.reporting_scope = scope
            self.period = "FY2024-25"
            self.metric = "revenue"
    reranked = rerank_hits("What was Reliance Digital Services revenue in FY2025?", [wrong,right],
                            curated_facts=[F(40,"","standalone"),F(3,"Digital Services","segment")],top_k=2)
    assert reranked[0].chunk.page_no == 3


def test_missing_real_citation_is_not_marked_verified(monkeypatch):
    def unavailable(*args, **kwargs):
        raise ValueError("missing original page")
    import risklens.research.structured_pipeline_v4 as p
    monkeypatch.setattr(p, "verify_with_groq", unavailable)
    result = _verify_result("deposits?", {"answer":"₹5000 crore", "citations":[{"file_sha256":"a","pdf_page":4}]},
                            client=object(), pdf_dir=Path('.'), file_map={"a":"report.pdf"})
    assert result["status"] == "verifier_unavailable" and result["verdict"] == "unclear"


def test_groq_rate_limit_not_reported_as_document_abstention():
    class RateLimitError(Exception):
        pass
    class Chat:
        def create(self, **kwargs):
            raise RateLimitError("too many requests")
    c = SimpleNamespace(chat=SimpleNamespace(completions=Chat()))
    doc = sample_doc("a"*64,"HDFC Bank Limited","HDFCBANK")
    result = respond("What is the NIM?",[hit(doc, 4,"Net interest margin: 3.5%","x")],provider="groq",client=c)
    assert result.status == "provider_unavailable"
    assert "Insufficient" not in result.answer


def test_reranking_preserves_semantic_candidate_when_not_exact_words():
    doc = sample_doc("f"*64,"HDFC Bank Limited","HDFCBANK")
    h = hit(doc,5,"Sustained efficiency from new banking practices", "x")
    assert rerank_hits("What risks has HDFC Bank identified?",[h])[0] == h


def test_structured_evidence_is_in_generation_prompt():
    class Chat:
        def create(self, **kwargs):
            assert "Structured financial evidence" in kwargs["messages"][1]["content"]
            assert "standalone" in kwargs["messages"][1]["content"]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"answer":"Net interest margin was 3.5%.","supporting_chunk_ids":["x"*24]})))], usage=None)
    c=SimpleNamespace(chat=SimpleNamespace(completions=Chat()))
    doc=sample_doc("a"*64,"HDFC Bank Limited","HDFCBANK")
    r=respond("What was net interest margin?",[hit(doc,4,"Net interest margin 3.5%", "x")],provider="groq",client=c,
              structured_facts=[{"metric":"net_interest_margin","reporting_scope":"standalone","value":"3.5","unit":"percent"}])
    assert r.status in {"answered","needs_review"}


def test_corrupted_json_auto_bank_rejected(tmp_path):
    p = tmp_path / 'test.json'
    p.write_text(json.dumps({"schema_version": 4, "facts": [{"value": "32"}]}))
    with pytest.raises(ResearchError):
        load_auto_bank(p)


def test_correction_not_treated_as_certified(monkeypatch):
    import risklens.research.structured_pipeline_v4 as p
    monkeypatch.setattr(p, "verify_with_groq",lambda *a, **k: {"reviewer_verdict":"supported", "status":"AI_PAGE_VERIFICATION_NOT_CERTIFICATION"})
    r=p._verify_result("revenue?",{"answer":"200 crore", "citations":[{"file_sha256":"a","pdf_page":2}]},
                      client=object(),pdf_dir=Path('.'),file_map={"a":"r.pdf"})
    assert r["status"] == "model_supported_not_certified"


def test_heldout_not_same_as_prior_35():
    p=Path(__file__).parents[1]/'docs'/'phase7_heldout_v4.json'
    data=json.loads(p.read_text())
    assert len(data['questions'])>=20 and len({x['id'] for x in data['questions']})==len(data['questions'])
    assert all(x['id'].startswith('V4_') for x in data['questions'])


def test_default_cli_routes_to_v4_without_legacy_flag():
    import argparse
    from risklens.research.cli import configure_parsers
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command")
    configure_parsers(sub)
    parsed = root.parse_args(["research-ask", "ICICI standalone operating expenses FY2025?"])
    assert not parsed.legacy_v3
    assert parsed.auto_facts.endswith("financial_facts_auto_v4.json")
    assert not parsed.no_page_verification


def test_original_icici_table_cell_mismatch_is_rejected_when_report_present(tmp_path):
    import hashlib
    from risklens.research.financial_auto_v4 import extract_pdf_tables, confirm_original_table_rows
    root = Path("data/raw/research_reports")
    path = root / "icici-bank-2025-financial-statements.pdf"
    if not path.is_file():
        pytest.skip("Original PDF not installed in this test checkout")
    source = dict(pdf_path=str(path), source_file=path.name, ticker="ICICIBANK",
                  company="ICICI Bank Limited", source_url="https://example.com/a.pdf",
                  file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  fiscal_year="2024-25", published_on="2025-08-05")
    facts, _ = extract_pdf_tables(source, max_pages=12)
    matched = [f for f in facts if f["physical_pdf_page"] == 12 and f["metric"] == "operating_expenses"
               and f["observed_period"] == "FY2024-25" and f["status"] == "auto_anchored"]
    assert len(matched) == 1
    assert confirm_original_table_rows(matched, pdf_dir=root)["rows_verified_against_original_table"] == 1
    tampered = {**matched[0], "source_literal": "391,327,336", "value": "391327336"}
    with pytest.raises(ResearchError, match="row/value attribution"):
        confirm_original_table_rows([tampered], pdf_dir=root)


def test_prebuilt_facts_do_not_claim_human_certification(tmp_path):
    from risklens.research.financial_auto_v4 import load_auto_bank
    candidate = source_auto()
    path = tmp_path / "v4_bank.json"
    path.write_text(json.dumps({"schema_version": 4, "facts": [candidate]}))
    parsed = load_auto_bank(path, expected_hashes={candidate["file_sha256"]})
    assert parsed[0]["status"] == "auto_anchored"
    out = automatic_fact_answer("What were HDFC Bank standalone deposits FY2025?", parsed)
    assert out["status"] == "needs_review"
    assert out["citations"][0]["trust_level"].endswith("not_human_certified")
