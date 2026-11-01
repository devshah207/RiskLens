"""Financially meaningful retrieval reranking and structured-evidence selection.

A similarity score finds related passages. Attribution checks which entity,
segment, accounting scope, period and table column a figure represents.
No gold-question IDs, memorized answers or exact-quotation gates are used.
"""
from __future__ import annotations

from decimal import Decimal
import re
from typing import Iterable

from .contracts import Hit
from .financial_facts import Fact, _intent

SEGMENTS = {
    "digital services": ("digital services", "jio", "digital"),
    "oil to chemicals": ("oil-to-chemicals", "oil to chemicals", "o2c"),
    "media and entertainment": ("media and entertainment", "media & entertainment"),
    "retail": ("retail",),
    "instabiz": ("instabiz",),
    "imobile": ("imobile",),
}


def inferred_intent(question: str, ticker: str | None = None) -> dict:
    q = question.lower().replace("–", "-")
    old = _intent(q)
    metrics = set(old["metrics"])
    # Generic metrics not covered by V3's curated question parser.
    for phrase, metric in (
        ("net profit", "profit_after_tax"), ("profit for the year", "profit_after_tax"),
        ("net interest income", "net_interest_income"), ("return on assets", "return_on_assets"),
        ("total assets", "balance_sheet_size"), ("operating expenses", "operating_expenses"),
        ("interest expense", "interest_expense"), ("total equity", "equity"),
        ("operating cash flow", "cash_flow_operations"),
        ("earnings per share", "earnings_per_share"),
        ("cost-to-income ratio", "cost_income_ratio"),
        ("cost to income ratio", "cost_income_ratio"),
        ("consolidated cash and investments", "cash_investments"),
        ("cash and investments", "cash_investments"),
        ("goodwill and other intangible", "goodwill_intangibles"),
        ("provision coverage ratio", "provision_coverage_ratio"),
        ("total net revenue", "revenue"),
        ("total income", "total_income"),
        ("cash and balances with rbi", "cash_rbi"),
        ("cash and balances with reserve bank", "cash_rbi"),
    ):
        if phrase in q:
            metrics.add(metric)
    segment = [name for name, aliases in SEGMENTS.items() if any(alias in q for alias in aliases)]
    # Substring "jio" is intentionally less strong than an exact division name.
    if "digital services" in segment and "digital services" not in q and "jio" not in q:
        segment.remove("digital services")
    fyear = re.search(r"\b(?:FY\s*)?(20\d\d)[-–/](?:20)?(\d\d)\b", q, re.I)
    year = f"FY{fyear.group(1)}-{fyear.group(2)}" if fyear else None
    if year is None:
        m = re.search(r"\b(?:fy|fiscal)\s*(20\d\d)\b", q)
        if m:
            yr = int(m.group(1))
            year = f"FY{yr-1}-{str(yr)[-2:]}"
    if year is None:
        march = re.search(r"\b(?:march|mar)\s+(20\d\d)", q)
        if march:
            yr = int(march.group(1))
            year = f"FY{yr-1}-{str(yr)[-2:]}"
    if year is None and ("2024-25" in q or "fy2025" in q or "fy 2025" in q):
        year = "FY2024-25"
    scope = "standalone" if "standalone" in q else "consolidated" if "consolidated" in q else None
    query_tickers = set()
    if re.search(r"\breliance\b|\bril\b", q):
            query_tickers.add("RIL")
    if "hdfc" in q:
            query_tickers.add("HDFCBANK")
    if "infosys" in q:
            query_tickers.add("INFY")
    if "icici" in q:
            query_tickers.add("ICICIBANK")
    if ticker:
            query_tickers.add(ticker.upper())
    return {"metrics": sorted(metrics), "segments": segment, "period": year,
            "scope": scope, "tickers": sorted(query_tickers), "question": q}


def _scope_matches(wanted: str | None, observed: str) -> bool:
    if not wanted:
        return True
    return wanted == observed or wanted == "standalone" and observed == "bank_standalone"


def _unit_phrase(f: dict, value: Decimal) -> str:
    def pretty(n):
        return f"{n:,.4f}".rstrip("0").rstrip(".") if n % 1 else f"{n:,.0f}"
    u = f["unit"]
    if u == "percent":
        return pretty(value) + "%"
    if u == "crore":
        return "₹" + pretty(value) + " crore" if f.get("currency") == "INR" else pretty(value) + " crore"
    if u == "thousand_rupees":
        return "₹" + pretty(value / Decimal(10000)) + " crore (converted from ₹ thousand)"
    if u == "billion":
        return "₹" + pretty(value) + " billion" if f.get("currency") == "INR" else pretty(value) + " billion"
    return pretty(value) + (" " + u if u != "count" else "")


def select_automatic(question: str, auto_facts: Iterable[dict], *, ticker: str | None = None,
                     source_hashes: set[str] | None = None,
                     as_of: str | None = None) -> tuple[list[dict], dict]:
    intent = inferred_intent(question, ticker)
    if not intent["metrics"]:
        return [], {"reason": "non_numeric_question"}
    if len(intent["tickers"]) > 1:
        return [], {"reason": "ambiguous_company"}
    found = []
    for f in auto_facts:
        if f["status"] != "auto_anchored":
            continue
        if source_hashes is not None and f["file_sha256"] not in source_hashes:
            continue
        if as_of and f["published_on"] > as_of:
            continue
        if intent["tickers"] and f["ticker"] not in intent["tickers"]:
            continue
        if f["metric"] not in intent["metrics"]:
            continue
        if intent["period"] and f["observed_period"] != intent["period"]:
            continue
        if not _scope_matches(intent["scope"], f["reporting_scope"]):
            continue
        if intent["segments"] and f.get("segment", "").lower() not in intent["segments"]:
            continue
        if not intent["segments"] and f.get("segment"):
            continue
        found.append(f)
    if not found:
        return [], {"reason": "no_high_confidence_auto_table_fact"}
    # A generic question cannot silently choose one of multiple accounting bases.
    scope_keys = {(f["metric"], f["reporting_scope"], f["observed_period"]) for f in found}
    if not intent["scope"] and len({(m,s) for m,s,_ in scope_keys}) > len({m for m,s,_ in scope_keys}):
        return [], {"reason": "ambiguous_reporting_scope"}
    by_metric = {}
    for f in found:
        by_metric.setdefault(f["metric"], []).append(f)
    selected = []
    for m in intent["metrics"]:
        candidates = by_metric.get(m, [])
        if not candidates:
            return [], {"reason": "incomplete_requested_metrics", "missing": m}
        # Identical financial values can be reiterated on multiple pages.
        unique = {(f["value"], f["currency"], f["unit"], f["reporting_scope"], f["observed_period"]) for f in candidates}
        if len(unique) > 1:
            return [], {"reason": "conflicting_table_values", "metric": m, "values": sorted(map(str,unique))}
        selected.append(sorted(candidates, key=lambda f: (f["physical_pdf_page"], f["table_index"]))[0])
    return selected, {"reason": "table_row_year_scope_match", "complete": True}


def automatic_fact_answer(question: str, auto_facts: Iterable[dict], *, ticker: str | None = None,
                          source_hashes: set[str] | None = None,
                          as_of: str | None = None) -> dict | None:
    selected, reason = select_automatic(question, auto_facts, ticker=ticker,
                                        source_hashes=source_hashes, as_of=as_of)
    if not selected:
        return None
    citations = []
    descriptions = []
    for f in selected:
        value = Decimal(f["value"])
        descriptions.append(f"{f['company']} {f['reporting_scope']} {f['row_label']} ({f['observed_period']}): {_unit_phrase(f, value)}")
        citations.append({"fact_id": f["fact_id"], "ticker": f["ticker"], "company": f["company"],
                          "file_sha256": f["file_sha256"], "source_url": f["source_url"],
                          "pdf_page": f["physical_pdf_page"], "source_table_row": f["row_label"],
                          "source_table_column": f["column_label"], "source_literal": f["source_literal"],
                          "reported_scope": f["reporting_scope"], "reported_period": f["observed_period"],
                          "trust_level": "native_table_auto_anchored_not_human_certified"})
    return {"question": question, "answer": "; ".join(descriptions), "citations": citations,
            "status": "needs_review", "provider": "auto_financial_table_json",
            "note": "Row/column linked to original PDF table; automatic scope attribution is not human-certified"}


def _text_has_segment(text: str, segment: str) -> bool:
    return any(alias in text.lower() for alias in SEGMENTS.get(segment, (segment,)))


def rerank_hits(question: str, hits: list[Hit], *, auto_facts: Iterable[dict] = (),
                curated_facts: Iterable[Fact] = (),
                cross_encoder=None, top_k: int = 5) -> list[Hit]:
    """Rerank a LARGE hybrid candidate pool, not a pretruncated top-five.

    Optional cross_encoder must have a .predict([(question,passage), ...]) method.
    Explicit scope mismatch outweighs a good embedding score. No candidate is
    admitted solely because its score is high: the answer verifier still applies.
    """
    intent = inferred_intent(question)
    if not hits:
        return []
    rows_by_page: dict[tuple[str, int], list[dict]] = {}
    for f in auto_facts:
        if f.get("status") == "auto_anchored":
            rows_by_page.setdefault((f["file_sha256"], f["physical_pdf_page"]), []).append(f)
    curated_by_page: dict[tuple[str, int], list[Fact]] = {}
    for f in curated_facts:
        curated_by_page.setdefault((f.file_sha256, f.pdf_page), []).append(f)
    ce_scores = None
    if cross_encoder is not None:
        prediction = cross_encoder.predict([(question, h.chunk.text[:1800]) for h in hits])
        if len(prediction) != len(hits):
            raise ValueError("Cross-encoder returned wrong candidate count")
        ce_scores = [max(-20.0, min(20.0, float(x))) for x in prediction]
    candidates = []
    for i, hit in enumerate(hits):
        text = (hit.chunk.section + " " + hit.chunk.text[:3500]).lower()
        page = (hit.document.sha256, hit.chunk.page_no)
        same = list(rows_by_page.get(page, []))
        same.extend({"metric": f.metric, "segment": f.segment,
                     "reporting_scope": f.reporting_scope,
                     "observed_period": f.period} for f in curated_by_page.get(page, []))
        bonus = 0.0
        if intent["tickers"]:
            bonus += 1.5 if hit.document.spec.ticker in intent["tickers"] else -7.0
        for m in intent["metrics"]:
            if any(f["metric"] == m for f in same):
                bonus += 2.5
            elif m.replace("_", " ") in text:
                bonus += 0.6
        for s in intent["segments"]:
            if any(f.get("segment", "").lower() == s for f in same):
                bonus += 2.5
            elif _text_has_segment(text, s):
                bonus += 0.75
            elif same and any(f.get("segment") for f in same):
                bonus -= 2.0
        if intent["scope"]:
            scope = intent["scope"]
            aligned = any(_scope_matches(scope, f["reporting_scope"]) for f in same)
            wrong = same and all(not _scope_matches(scope, f["reporting_scope"]) for f in same)
            if aligned:
                bonus += 2.5
            elif wrong:
                bonus -= 4.0
            elif scope in text:
                bonus += 0.5
        if intent["period"]:
            if any(f["observed_period"] == intent["period"] for f in same):
                bonus += 1.0
            elif same and all(f["observed_period"] != intent["period"] for f in same):
                bonus -= 1.2
        if ce_scores is not None:
            bonus += ce_scores[i] * 0.25
        score = hit.score * 50 + bonus
        candidates.append((score, hit))
    candidates.sort(key=lambda v: (-v[0], v[1].chunk.chunk_id))
    per_page = set()
    output = []
    for _, hit in candidates:
        key = (hit.document.doc_id, hit.chunk.page_no)
        if key in per_page:
            continue
        per_page.add(key)
        output.append(hit)
        if len(output) == top_k:
            return output
    return output


def structured_evidence(hits: list[Hit], *, auto_facts: Iterable[dict] = (),
                        curated_facts: Iterable[Fact] = (),
                        max_per_page: int = 10) -> list[dict]:
    """Source-scoped JSON context for Groq; ambiguous facts remain visibly marked."""
    by_page: dict[tuple[str, int], list[dict]] = {}
    for f in curated_facts:
        by_page.setdefault((f.file_sha256, f.pdf_page), []).append({
            "type": "human_review_pending_curated_pdf_fact", "metric": f.metric,
            "segment": f.segment, "reporting_scope": f.reporting_scope,
            "period": f.period, "value": f.value, "unit": f.unit,
            "source_literal": f.source_literal, "fact_id": f.fact_id,
        })
    for f in auto_facts:
        if f["status"] == "auto_anchored":
            by_page.setdefault((f["file_sha256"], f["physical_pdf_page"]), []).append({
                "type": "native_table_auto_anchored", "metric": f["metric"],
                "segment": f["segment"], "reporting_scope": f["reporting_scope"],
                "period": f["observed_period"], "value": f["value"], "unit": f["unit"],
                "row_label": f["row_label"], "year_column": f["column_label"],
                "fact_id": f["fact_id"],
            })
    return [{"chunk_id": h.chunk.chunk_id, "file_sha256": h.document.sha256,
             "physical_pdf_page": h.chunk.page_no,
             "facts": by_page.get((h.document.sha256, h.chunk.page_no), [])[:max_per_page]}
            for h in hits]
