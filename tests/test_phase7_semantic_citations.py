"""Offline, deterministic regression checks for relaxed-but-traceable RAG answers."""
import json
from types import SimpleNamespace

from risklens.research.answer import respond
from risklens.research.contracts import SourceSpec, Document, Chunk, Hit


def hit(text, chunk_id="ch1"):
    spec = SourceSpec(company="Example Ltd", ticker="MOCK", fiscal_year="2024-25",
                      title="Test financial statement", source_url="https://example.org/test.pdf",
                      published_on="2025-07-01", local_path="unused.pdf", license_note="test")
    doc = Document("doc1", spec, "hash", 2, "pymupdf")
    chunk = Chunk(chunk_id, "doc1", 1, 0, len(text), text)
    return Hit(chunk, doc, 1.0, ("dense",))


class MockClient:
    def __init__(self, payload):
        self.payload = payload
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        assert "supporting_chunk_ids" in kwargs["messages"][0]["content"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.payload)))],
                               usage=SimpleNamespace(total_tokens=100))


def test_paraphrased_quote_does_not_fail_source_id():
    passage = "FY2025 consolidated revenue was ₹1,62,990 crore, up 6.1% year-on-year."
    r = respond("How much did revenue grow?", [hit(passage)], provider="groq", client=MockClient({
        "answer": "Revenue increased 6.1%, reaching ₹162,990 crore.",
        "citations": [{"chunk_id": "ch1", "quote": "The revenue went up by 6.1 percent"}],
    }))
    assert r.status == "answered", r.note
    assert r.citations[0]["evidence_quote"] in passage
    assert r.citations[0]["evidence_quote"] != "The revenue went up by 6.1 percent"


def test_source_id_only_is_accepted():
    text = "Retail gross value of sales and services 3,30,943 crore. EBITDA 25,094 crore."
    r = respond("What is Retail EBITDA?", [hit(text)], provider="groq", client=MockClient({
        "answer": "Retail EBITDA was 25,094 crore.", "supporting_chunk_ids": ["ch1"]
    }))
    assert r.status == "answered"
    assert "25,094" in r.citations[0]["evidence_quote"]


def test_growth_percent_with_table_header():
    text = "2024-25 2023-24 YoY growth (%)\nRevenue from operations 1,62,990 1,53,670 6.1"
    r = respond("How much did revenue grow?", [hit(text)], provider="groq", client=MockClient({
        "answer": "Revenue increased 6.1%.", "supporting_chunk_ids": ["ch1"]
    }))
    assert r.status == "answered", r.note


def test_unsupported_fabricated_monetary_amount_still_blocked():
    r = respond("What revenue?", [hit("Revenue was ₹1,200 crore.")], provider="groq", client=MockClient({
        "answer": "Revenue was ₹999,999 crore.", "supporting_chunk_ids": ["ch1"]
    }))
    assert r.status == "abstained" and "monetary" in r.note


def test_nonmoney_numeric_uncertainty_returns_cautioned_answer():
    r = respond("What was the risk ratio?", [hit("NPA ratio was 1.73%.")], provider="groq", client=MockClient({
        "answer": "The risk ratio was 1.75%.", "supporting_chunk_ids": ["ch1"]
    }))
    assert r.status == "needs_review"
    assert "1.75%" in r.note


def test_unknown_citation_only_abstains():
    r = respond("What revenue?", [hit("Revenue was ₹1,200 crore.")], provider="groq", client=MockClient({
        "answer": "Purple elephants", "supporting_chunk_ids": ["unknown"]
    }))
    assert r.status == "abstained"


def test_unknown_id_alongside_real_id_yields_review_status():
    r = respond("What revenue?", [hit("Revenue was ₹1,200 crore.")], provider="groq", client=MockClient({
        "answer": "Revenue was ₹1,200 crore.", "supporting_chunk_ids": ["ch1", "unknown"]
    }))
    assert r.status == "needs_review"
    assert len(r.citations) == 1 and r.citations[0]["chunk_id"] == "ch1"


def test_supported_partial_answer_accepted():
    r = respond("What were revenue and margin?", [hit("Revenue was ₹1,200 crore. Margin not disclosed.")],
                provider="groq", client=MockClient({
        "answer": "Revenue was ₹1,200 crore. The supplied source does not establish the margin.",
        "supporting_chunk_ids": ["ch1"]
    }))
    assert r.status == "answered"


def test_unrelated_text_without_ids_abstains():
    r = respond("What revenue?", [hit("Revenue was ₹1,200 crore.")], provider="groq", client=MockClient({
        "answer": "Hello goodbye", "supporting_chunk_ids": []
    }))
    assert r.status == "abstained"
