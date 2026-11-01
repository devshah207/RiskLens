"""Financial attribution: prevention of wrong-segment/wrong-basis/wrong-unit claims."""
import json

import pytest

from risklens.research.contracts import ResearchError
from risklens.research.financial_facts import Fact, read_facts, fact_answer, lookup_facts, validate_original_pdfs


def facts():
    base = dict(
        fact_id="a"*24, ticker="RIL", company="Reliance Industries Limited", fiscal_year="2024-25",
        published_on="2025-08-07", source_url="https://example.com/report.pdf",
        file_sha256="f"*64, source_file="ril.pdf", pdf_page=3, segment="Digital Services",
        reporting_scope="segment", metric="revenue", display_label="segment revenue",
        aliases=["revenue", "value of sales and services"], value="154119", scale_factor="1",
        currency="INR", unit="crore", source_literal="1,54,119",
        source_evidence="Digital Services revenue (₹ crore) 1,54,119", period="FY2024-25",
        review_status="source_checked_ai_reviewed",
        definition="", definition_pdf_page=0, definition_source_quote="",
    )
    a = Fact.parse(base)
    b = Fact.parse({**base, "fact_id":"b"*24, "segment":"", "reporting_scope":"standalone",
                    "metric":"revenue", "value":"557163", "source_literal":"5,57,163",
                    "source_evidence":"Reliance standalone sales 5,57,163"})
    c = Fact.parse({**base, "fact_id":"c"*24, "segment":"Oil to Chemicals", "value":"626921",
                    "source_literal":"6,26,921", "source_evidence":"Oil to Chemicals revenue 6,26,921"})
    d = Fact.parse({**base, "fact_id":"d"*24, "segment":"Oil to Chemicals", "reporting_scope":"standalone",
                    "value":"491622", "source_literal":"4,91,622",
                    "source_evidence":"Oil to Chemicals standalone disaggregated revenue 4,91,622"})
    return [a, b, c, d]


def test_revenue_does_not_bleed_across_segments():
    result = fact_answer("What was Reliance Digital Services revenue FY2025?", facts())
    assert result and "154,119" in result["answer"]
    assert "557,163" not in result["answer"]
    assert len(result["citations"]) == 1
    assert result["citations"][0]["segment"] == "Digital Services"


def test_generic_company_question_does_not_take_segment_value():
    result = fact_answer("What was Reliance standalone revenue FY2025?", facts())
    assert result and "557,163" in result["answer"] and "154,119" not in result["answer"]


def test_o2c_segment_summary_does_not_use_standalone():
    result = fact_answer("What was Reliance Oil-to-Chemicals revenue in its FY2025 segment summary?", facts())
    assert result and "626,921" in result["answer"] and "491,622" not in result["answer"]


def test_unspecified_o2c_scope_is_ambiguous_not_guessed():
    result, info = lookup_facts("What was Reliance Oil-to-Chemicals revenue FY2025?", facts())
    assert len(result) == 2 and info["reason"] == "ambiguous_metric_values"
    rendered = fact_answer("What was Reliance Oil-to-Chemicals revenue FY2025?", facts())
    assert rendered["status"] == "needs_review"
    assert "626,921" in rendered["answer"] and "491,622" in rendered["answer"]


def test_non_financial_question_does_not_trigger_a_fact():
    assert fact_answer("What are Reliance's financial risks?", facts()) is None


def test_wrong_year_and_ticker_never_substitute():
    assert fact_answer("Reliance Digital Services revenue FY2025?", facts(), ticker="INFY") is None
    assert fact_answer("Reliance Digital Services revenue FY2025?", facts(), as_of="2024-12-31") is None
    assert fact_answer("Reliance Digital Services revenue FY2025?", facts(), eligible_sha256={"0"*64}) is None


def test_tampered_conversion_rejected():
    raw = facts()[0].as_dict()
    raw["value"] = "557163"
    with pytest.raises(ResearchError, match="conversion factor"):
        Fact.parse(raw)


def test_source_literal_must_exist_in_excerpt():
    raw = facts()[0].as_dict()
    raw["source_evidence"] = "Digital revenue: 557,163"
    with pytest.raises(ResearchError, match="source evidence"):
        Fact.parse(raw)


def test_invalid_source_record_not_certified():
    raw = facts()[0].as_dict()
    raw["review_status"] = "groq_auto_approved"
    with pytest.raises(ResearchError, match="Unreviewed"):
        Fact.parse(raw)


def test_json_bank_roundtrip(tmp_path):
    path = tmp_path / "facts.json"
    path.write_text(json.dumps({"schema_version":1,"facts":[x.as_dict() for x in facts()]}))
    assert len(read_facts(path)) == 4
    corrupted = json.loads(path.read_text())
    corrupted["facts"].append(corrupted["facts"][0])
    path.write_text(json.dumps(corrupted))
    with pytest.raises(ResearchError, match="Duplicate"):
        read_facts(path)


def test_original_pdf_provenance_fail_closed(tmp_path):
    (tmp_path / "ril.pdf").write_bytes(b"bogus")
    with pytest.raises(ResearchError, match="SHA256"):
        validate_original_pdfs(facts(), tmp_path)


def test_percentage_and_units_are_unambiguous():
    src = facts()[0].as_dict()
    src.update(fact_id="e"*24, ticker="ICICIBANK", company="ICICI Bank Limited", segment="InstaBIZ",
               reporting_scope="product", metric="transaction_value_growth", display_label="transaction growth",
               value="37", unit="percent", currency="none", source_literal="37",
               source_evidence="InstaBIZ transactions up 37%")
    target = Fact.parse(src)
    result = fact_answer("How much did ICICI InstaBIZ transaction value grow FY2025?", [target])
    assert result and "37%" in result["answer"]
    assert "154,119" not in result["answer"]


def test_no_future_prediction_from_historical_facts():
    assert fact_answer("What will Reliance revenue be in November 2029?", facts()) is None


def test_deterministic_numeric_verifier_catches_wrong_but_real_revenue():
    from risklens.research.financial_verifier import deterministic_fact_conflict
    data = facts()
    q = "What was Reliance Digital Services revenue FY2025?"
    wrong = deterministic_fact_conflict(q, "Revenue was ₹557,163 crore.", data)
    assert wrong["status"] == "possible_wrong_metric_or_unit"
    assert "154119" in wrong["expected"]
    correct = deterministic_fact_conflict(q, "Revenue was ₹154,119 crore.", data)
    assert correct["status"] == "source_fact_values_present"


def test_original_page_verifier_rejects_wrong_pdf_hash(tmp_path):
    from risklens.research.financial_verifier import _original_citation_page
    path = tmp_path / "ril.pdf"
    path.write_bytes(b"arbitrary unrelated PDF")
    with pytest.raises(ResearchError, match="SHA256"):
        _original_citation_page({"file_sha256":"f"*64,"pdf_page":3},tmp_path,{"f"*64:"ril.pdf"})
