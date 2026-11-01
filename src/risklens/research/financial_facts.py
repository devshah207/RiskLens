"""Source-traceable JSON financial facts with conservative scope-aware lookup.

A fact is only eligible after its source PDF byte hash, physical page and
literal numeric observation are verified. JSON is the representation, not
an assertion that an LLM's interpretation is correct.

This module intentionally does not depend on Groq, embeddings or cloud APIs.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Iterable

from .contracts import ResearchError

_SCHEMA = 1


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower().replace("&", " and ")))


def _value_number(value: str | int | float) -> Decimal:
    try:
        number = Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ResearchError("Invalid financial-fact numeric value") from exc
    if not number.is_finite():
        raise ResearchError("Nonfinite financial-fact number")
    return number


def _normalized_literal(value: str) -> str:
    try:
        return str(_value_number(value))
    except ResearchError:
        return ""


def _page_has_value(page_text: str, needle: str) -> bool:
    if needle in page_text:
        return True
    norm = _normalized_literal(needle)
    if not norm:
        return False
    return any(_normalized_literal(match.group()) == norm
               for match in re.finditer(r"(?<!\w)\d[\d,]*(?:\.\d+)?(?!\w)", page_text))


@dataclass(frozen=True)
class Fact:
    fact_id: str
    ticker: str
    company: str
    fiscal_year: str
    published_on: str
    source_url: str
    file_sha256: str
    source_file: str
    pdf_page: int
    segment: str
    reporting_scope: str
    metric: str
    display_label: str
    aliases: tuple[str, ...]
    value: str
    scale_factor: str
    currency: str
    unit: str
    source_literal: str
    source_evidence: str
    period: str
    review_status: str
    definition: str
    definition_pdf_page: int
    definition_source_quote: str

    @classmethod
    def parse(cls, raw: dict) -> Fact:
        expected = set(cls.__dataclass_fields__)
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ResearchError("Financial fact JSON contains missing or unrecognized fields")
        data = dict(raw)
        if not isinstance(data["aliases"], list) or not all(isinstance(v, str) and v.strip() for v in data["aliases"]):
            raise ResearchError("Aliases must be nonempty strings")
        data["aliases"] = tuple(data["aliases"])
        f = cls(**data)
        if not re.fullmatch(r"[a-f0-9]{64}", f.file_sha256):
            raise ResearchError("Financial fact missing original PDF SHA256")
        if not isinstance(f.pdf_page, int) or f.pdf_page < 1 or not re.fullmatch(r"[a-f0-9]{24}", f.fact_id):
            raise ResearchError("Invalid financial fact page/id")
        if f.definition and (f.definition_pdf_page < 1 or not f.definition_source_quote):
            raise ResearchError("Financial definition requires a real source page and quote")
        if f.review_status not in {"source_checked_ai_reviewed", "human_approved"}:
            raise ResearchError("Unreviewed fact must not enter trusted fact bank")
        if f.reporting_scope not in {"segment", "standalone", "consolidated", "bank_standalone", "product"}:
            raise ResearchError("Unknown financial scope")
        if f.currency not in {"INR", "USD", "none"} or f.unit not in {"crore", "billion", "million", "percent", "count", "thousand_rupees"}:
            raise ResearchError("Unsupported financial unit")
        if _value_number(f.source_literal) * _value_number(f.scale_factor) != _value_number(f.value):
            raise ResearchError("Financial value is not equal to source literal * declared conversion factor")
        if not _page_has_value(f.source_evidence, f.source_literal):
            raise ResearchError("Financial fact literal absent from its source evidence")
        return f

    def as_dict(self) -> dict:
        out = vars(self).copy()
        out["aliases"] = list(self.aliases)
        return out

    @property
    def printed_value(self) -> str:
        v = _value_number(self.value)
        pretty = f"{v:,.4f}".rstrip("0").rstrip(".") if v % 1 else f"{v:,.0f}"
        if self.unit == "percent":
            return pretty + "%"
        if self.unit == "thousand_rupees":
            return "₹" + pretty + " thousand"
        if self.currency == "INR":
            return "₹" + pretty + " " + self.unit
        if self.currency == "USD":
            return "US$" + pretty + " " + self.unit
        return pretty + (" " + self.unit if self.unit != "count" else "")


def read_facts(path: str | Path) -> list[Fact]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != _SCHEMA or not isinstance(data.get("facts"), list):
        raise ResearchError("Invalid financial fact bank schema")
    facts = [Fact.parse(item) for item in data["facts"]]
    if len({f.fact_id for f in facts}) != len(facts):
        raise ResearchError("Duplicate financial fact IDs")
    return facts


def validate_original_pdfs(facts: Iterable[Fact], pdf_directory: str | Path) -> dict:
    """Full PDF hash and original page-text checks; never trust JSON alone."""
    try:
        import pymupdf
    except ImportError as exc:
        raise ResearchError("pymupdf required for financial-fact provenance verification") from exc
    files: dict[str, list[Fact]] = {}
    for fact in facts:
        files.setdefault(fact.source_file, []).append(fact)
    output = {}
    root = Path(pdf_directory).resolve()
    for name, batch in files.items():
        if Path(name).name != name:
            raise ResearchError("Source filename must not contain a path")
        path = root / name
        if not path.is_file():
            raise ResearchError(f"Missing financial fact source PDF: {path}")
        got = sha256(path.read_bytes()).hexdigest()
        if len({f.file_sha256 for f in batch}) != 1 or got != batch[0].file_sha256:
            raise ResearchError(f"Financial fact PDF SHA256 mismatch: {name}")
        with pymupdf.open(path) as pdf:
            for fact in batch:
                if fact.pdf_page > len(pdf):
                    raise ResearchError(f"Financial fact page out of range: {fact.fact_id}")
                text = pdf[fact.pdf_page - 1].get_text(sort=True)
                if not _page_has_value(text, fact.source_literal):
                    raise ResearchError(f"Financial fact literal missing from original page: {fact.fact_id}")
                if fact.source_evidence not in text:
                    raise ResearchError(f"Source text span not found verbatim on page: {fact.fact_id}")
                if fact.definition:
                    if fact.definition_pdf_page > len(pdf):
                        raise ResearchError("Financial definition page out of range")
                    definition_page = pdf[fact.definition_pdf_page-1].get_text(sort=True)
                    if fact.definition_source_quote not in definition_page:
                        raise ResearchError("Financial definition source quote not found")
        output[name] = {"sha256": got, "facts": len(batch), "status": "original_pdf_span_confirmed"}
    return output


def default_fact_bank_path() -> Path:
    return Path(os.getenv("RISKLENS_FINANCIAL_FACTS", "data/processed/research/real_acceptance/phase7_completion/financial_facts_v3.json"))


def _intent(text: str) -> dict:
    q = text.lower()
    scopes = {
        "standalone": bool(re.search(r"\bstandalone\b", q)),
        "consolidated": bool(re.search(r"\bconsolidated\b", q)),
    }
    segments = {
        "digital services": ("digital services", "jio digital"),
        "retail": ("retail",),
        "oil to chemicals": ("oil-to-chemicals", "oil to chemicals", "o2c"),
        "media and entertainment": ("media and entertainment", "media & entertainment"),
        "instabiz": ("instabiz",),
        "imobile": ("imobile",),
    }
    seg = [name for name, aliases in segments.items() if any(alias in q for alias in aliases)]
    # A question explicitly naming a product/segment cannot be answered from
    # a different (e.g. company-standalone) scope.
    req = set()
    if "ebitda" in q:
        req.add("ebitda")
    if "free cash flow" in q:
        req.add("free_cash_flow")
    if "net interest margin" in q or re.search(r"\bnim\b", q):
        req.add("net_interest_margin")
    if "operating margin" in q:
        req.add("operating_margin")
    if "net current assets" in q:
        req.add("net_current_assets")
    if "cost of sales" in q:
        req.add("cost_of_sales")
    if "capital adequacy" in q or "crar" in q:
        req.add("capital_adequacy_ratio")
    if "gross non-performing" in q or "gross npa" in q:
        req.add("gross_npa_ratio")
    if "net npa" in q or "net non-performing" in q:
        req.add("net_npa_ratio")
    if "balance-sheet size" in q or "balance sheet size" in q:
        req.add("balance_sheet_size")
    if "return on equity" in q or re.search(r"\broe\b", q):
        req.add("return_on_equity")
    if "deposits" in q:
        req.add("deposits_growth" if ("grow" in q or "growth" in q) else "deposits")
    if "advances" in q and "npa" not in q and "non-performing" not in q:
        req.add("advances_growth" if ("grow" in q or "growth" in q) else "advances")
    if "profit" in q and "ebitda" not in q:
        req.add("profit_after_tax")
    if "revenue" in q or "sales and services" in q:
        req.add("revenue")
    if "imobile" in q and "transactions" in q:
        req.add("transaction_count" if "how many" in q else "transaction_value")
    if "instabiz" in q and "transaction value" in q and ("grow" in q or "growth" in q):
        req.add("transaction_value_growth")
    if "grow" in q or "growth" in q or "year-over-year" in q or "yoy" in q:
        if "revenue" in req and "revenue" in q:
            req.add("revenue_growth")
        if "profit_after_tax" in req:
            req.add("profit_growth")
    if "minimum" in q and "cet1" in q:
        req = {"minimum_cet1_ratio"}
    return {"segments": seg, "scope": scopes, "metrics": req, "question": q}


def _financial_question(question: str) -> bool:
    return bool(_intent(question)["metrics"])


def lookup_facts(question: str, facts: Iterable[Fact], *, ticker: str | None = None,
                 eligible_sha256: set[str] | None = None, as_of: str | None = None) -> tuple[list[Fact], dict]:
    """Exact scope/metric alignment; never guess a financial measure from nearby digits.

    Returns NO match on ambiguous entities, periods or scope. This intentionally
    gives qualitative questions to ordinary semantic RAG instead.
    """
    intent = _intent(question)
    metrics = intent["metrics"]
    if not metrics:
        return [], {"reason": "no_structured_financial_intent"}
    q = intent["question"]
    if re.search(r"\b(?:will|predict|forecast|estimate|projection|next trading day|future)\b", q) and re.search(r"\b20(?:2[7-9]|[3-9][0-9])\b", q):
        return [], {"reason": "future_outside_report_period"}
    pool = []
    for f in facts:
        if ticker and f.ticker != ticker.upper():
            continue
        if eligible_sha256 is not None and f.file_sha256 not in eligible_sha256:
            continue
        if as_of and f.published_on > as_of:
            continue
        if f.ticker == "RIL" and not re.search(r"reliance|\bril\b", q) and not ticker:
            continue
        if f.ticker == "HDFCBANK" and "hdfc" not in q and not ticker:
            continue
        if f.ticker == "INFY" and "infosys" not in q and not ticker:
            continue
        if f.ticker == "ICICIBANK" and "icici" not in q and not ticker:
            continue
        if "2023-24" in q or "2024" in q and "2025" not in q and "2024-25" not in q:
            # Comparative FY2024 only requests are supported by separate records,
            # never silently by FY2025 facts.
            if f.period != "FY2023-24":
                continue
        else:
            if f.period not in {"FY2024-25", "2025-03-31"}:
                continue
        if intent["segments"]:
            if f.segment.lower() not in intent["segments"]:
                continue
        elif f.segment:
            # Never answer a generic company question with a segment figure.
            continue
        if intent["scope"]["standalone"] and f.reporting_scope not in {"standalone", "bank_standalone"}:
            continue
        if intent["scope"]["consolidated"] and f.reporting_scope != "consolidated":
            continue
        if f.metric in metrics:
            if "segment" in q and f.segment and f.reporting_scope != "segment" and f.segment.lower() not in {"imobile", "instabiz"}:
                continue
            pool.append(f)
    if not pool:
        return [], {"reason": "no_matching_metric_scope_period"}
    # Do not accidentally combine values from unrelated reporting scopes for
    # questions which don't specify the basis.
    if not any(intent["scope"].values()) and not intent["segments"]:
        by_metric: dict[str, set[str]] = {}
        for f in pool:
            by_metric.setdefault(f.metric, set()).add(f.reporting_scope)
        if any(len(scopes) > 1 for scopes in by_metric.values()):
            # Still provide source-labeled alternatives, never silently pick a
            # wrong but real scope. Caller must render NEEDS_REVIEW clearly.
            return sorted(pool, key=lambda f: (f.reporting_scope, f.segment, f.metric)), {
                "reason": "ambiguous_reporting_scope", "complete": False,
                "missing_metrics": [], "scopes": sorted({f.reporting_scope for f in pool}),
            }
    # Do not call an incomplete fact match a complete answer.
    selected = []
    for metric in metrics:
        matches = [f for f in pool if f.metric == metric]
        if len(matches) == 1:
            selected += matches
        elif len(matches) > 1:
            return sorted(matches, key=lambda f: (f.reporting_scope, f.segment)), {
                "reason": "ambiguous_metric_values", "complete": False,
                "missing_metrics": [], "metric": metric,
            }
    if not selected:
        return [], {"reason": "no_unambiguous_fact"}
    return sorted(selected, key=lambda f: f.metric), {
        "reason": "exact_scope_metric_match",
        "complete": len({f.metric for f in selected}) == len(metrics),
        "missing_metrics": sorted(metrics - {f.metric for f in selected}),
    }


def fact_answer(question: str, facts: Iterable[Fact], *, ticker: str | None = None,
                eligible_sha256: set[str] | None = None, as_of: str | None = None) -> dict | None:
    selected, match = lookup_facts(question, facts, ticker=ticker,
                                   eligible_sha256=eligible_sha256, as_of=as_of)
    if not selected:
        return None
    phrases = []
    for f in selected:
        label = f.display_label
        # Human-readable, nonduplicative labels retain segment/basis as defined
        # by structured provenance; provenance is repeated on the citation.
        scope = f.segment if f.segment and f.segment.lower() not in label.lower() else ""
        basis = f.reporting_scope.replace("_", " ") if not f.segment and f.reporting_scope not in label.lower() else ""
        heading = " ".join(x for x in (f.company, scope, basis, label) if x)
        phrases.append(f"{heading} ({f.period}): {f.printed_value}")
    if ("year-over-year" in question.lower() or "year over year" in question.lower()
            or "comparison" in question.lower()) and any(f.metric == "revenue" for f in selected):
        for current in selected[:]:
            if current.metric != "revenue":
                continue
            past = [f for f in facts if f.ticker == current.ticker and f.segment == current.segment
                    and f.reporting_scope == current.reporting_scope and f.metric == current.metric
                    and f.period == "FY2023-24" and f.file_sha256 == current.file_sha256]
            if len(past) == 1:
                prev = past[0]
                phrases.append(f"Prior-year {prev.period} {prev.display_label}: {prev.printed_value}")
                selected.append(prev)
    for fact in selected:
        if fact.definition and ("define" in question.lower() or "definition" in question.lower()):
            phrases.append(f"Definition: {fact.definition.rstrip(chr(46))}")
    if match["missing_metrics"]:
        phrases.append("Not established from source-checked structured facts: " + ", ".join(match["missing_metrics"]))
    citations = [{
        "fact_id": f.fact_id, "chunk_id": None, "doc_id": None,
        "ticker": f.ticker, "company": f.company, "fiscal_year": f.fiscal_year,
        "published_on": f.published_on, "source_url": f.source_url,
        "file_sha256": f.file_sha256, "pdf_page": f.pdf_page,
        "evidence_quote": f.source_evidence, "verification_status": f.review_status,
        "metric": f.metric, "segment": f.segment, "reporting_scope": f.reporting_scope,
        "period": f.period, "unit": f.unit,
    } for f in selected]
    for fact in selected:
        if fact.definition and ("define" in question.lower() or "definition" in question.lower()):
            citations.append({
                "fact_id": fact.fact_id, "chunk_id": None, "doc_id": None,
                "ticker": fact.ticker, "company": fact.company,
                "source_url": fact.source_url, "file_sha256": fact.file_sha256,
                "pdf_page": fact.definition_pdf_page,
                "evidence_quote": fact.definition_source_quote,
                "verification_status": fact.review_status,
                "fact_role": "definition",
            })
    if match["reason"].startswith("ambiguous"):
        phrases.insert(0, "Reporting basis was not specified; alternative reported values follow (do not interchange them)")
    return {"answer": "; ".join(phrases) + ".", "citations": citations,
            "status": "answered" if match["complete"] else "needs_review",
            "provider": "source_checked_financial_facts",
            "note": "Original PDF hash, physical page and literal source span checked. Attribution is AI-reviewed, NOT independently human-certified."
                    + (" Incomplete metric coverage." if not match["complete"] else ""),
            "api_latency_ms": None, "api_tokens": None,
            "scope_match": match}
