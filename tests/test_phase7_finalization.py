"""Targeted regressions for final Phase 7 reliability changes."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from risklens.research.answer import respond, _normalized_numbers
from risklens.research.contracts import Chunk, Document, Hit, SourceSpec


def _hit(text="FY2025 consolidated revenue was ₹1,62,990 crore, up 6.1% year-on-year."):
    spec = SourceSpec(company="Infosys Ltd", ticker="INFY", fiscal_year="2024-25", title="Annual report",
                      source_url="https://example.com/infosys.pdf", published_on="2025-07-01",
                      local_path="unused.pdf", license_note="synthetic")
    d = Document("doc1", spec, "hash", 300, "pymupdf")
    c = Chunk("c1", "doc1", 15, 0, len(text), text)
    return Hit(c, d, 1.0, ("lexical", "dense"))


def test_comma_grouping_numeric_equivalence_and_percent_preserved():
    assert _normalized_numbers("₹1,62,990 crores") == _normalized_numbers("162,990 crores")
    assert "6.1%" in _normalized_numbers("6.1%")
    assert "6.1%" not in _normalized_numbers("6.1 (no percent)")
    assert "24." not in _normalized_numbers("FY2024-25")


def test_supported_real_formatting_does_not_trigger_false_abstention():
    quote = "FY2025 consolidated revenue was ₹1,62,990 crore, up 6.1% year-on-year."
    data = {"answer": "Infosys revenue was ₹162,990 crore, with growth of 6.1%.",
            "citations": [{"chunk_id": "c1", "quote": quote}]}
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))],
                            usage=SimpleNamespace(total_tokens=100))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: reply)))
    response = respond("How much did revenues grow?", [_hit(quote)], provider="groq", client=client)
    assert response.status == "answered", response.note
    assert response.citations[0]["pdf_page"] == 15


def test_unsupported_financial_figure_rejected():
    quote = "FY2025 consolidated revenue was ₹1,62,990 crore, up 6.1% year-on-year."
    data = {"answer": "Revenue was ₹900,000 crore, up 6.1%.",
            "citations": [{"chunk_id": "c1", "quote": quote}]}
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))], usage=None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: reply)))
    response = respond("What was revenue?", [_hit(quote)], provider="groq", client=client)
    assert response.status == "abstained"
    assert "900000" in response.note


def test_unverified_source_gold_disclosed_as_provisional():
    seed = Path(__file__).resolve().parents[1] / "docs" / "phase7_independent_source_anchors.json"
    obj = json.loads(seed.read_text())
    questions = obj["questions"]
    assert len(questions) == 20
    assert sum(x["answerable"] for x in questions) == 16
    assert all(x["verification_method"].endswith("not_human_certified") for x in questions)
    fifth = [x for x in questions if x["id"] in {"ICICI01", "ICICI02"}]
    assert all(x["report_kind"] == "financial_statements" for x in fifth)
    assert all(x["source_sha256"] == "774a4a2286824d7ab7b317ba9fe02659577362a121b831736f595773b1033077" for x in fifth)
