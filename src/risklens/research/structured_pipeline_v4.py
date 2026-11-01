"""Integrated Phase 7: structured facts -> scope-aware RAG -> source-page audit.

This code is used by the maintained research-ask CLI, not an isolated demo.
A reviewer model is advisory: never certify a claim solely from its verdict.
API failures are never recorded as factual abstentions.
"""
from __future__ import annotations

from pathlib import Path
import re

from .answer import respond
from .config import load_project_env
from .contracts import ResearchError
from .financial_auto_v4 import load_auto_bank, confirm_original_table_rows
from .financial_facts import read_facts, fact_answer
from .financial_scope_v4 import (automatic_fact_answer, rerank_hits, structured_evidence,
                                  inferred_intent)
from .financial_verifier import verify_with_groq
from .retrieval import search


def _provider_client(client):
    if client is not None:
        return client
    import os
    load_project_env()
    if not os.getenv("GROQ_API_KEY"):
        raise ResearchError("GROQ_API_KEY required for live independent page verification")
    from groq import Groq
    return Groq(api_key=os.environ["GROQ_API_KEY"], timeout=70.0, max_retries=1)


def _unavailable(error: Exception) -> dict:
    return {"status": "provider_unavailable", "reviewer_verdict": "unclear",
            "error_type": type(error).__name__, "verdict": "unclear", "verified": False}


def _provider_status(result: dict) -> dict:
    if result.get("status") == "abstained" and str(result.get("note", "")).startswith("Provider unavailable/malformed:"):
        return {**result, "status": "provider_unavailable",
                "answer": "Groq provider unavailable; financial evidence could not be assessed in this request.",
                "verification_status": "not_performed"}
    return result


def _source_map(store, curated: list, automatic: list[dict], pdf_dir: str | Path) -> dict[str, str]:
    seen = {}
    for fact in curated:
        seen[fact.file_sha256] = fact.source_file
    for fact in automatic:
        seen[fact["file_sha256"]] = fact["source_file"]
    for doc in store.list_documents():
        if doc.sha256 not in seen and doc.spec.local_path:
            seen[doc.sha256] = Path(doc.spec.local_path).name
    return {key: name for key, name in seen.items() if (Path(pdf_dir) / name).is_file()}


def _verify_result(question: str, result: dict, *, client, pdf_dir: Path, file_map: dict[str, str],
                   vision: bool = False, max_citations: int = 2) -> dict:
    """Review each cited original page, not the LLM-created excerpt string."""
    citations = result.get("citations", [])
    if not citations:
        return {"status": "no_source_to_review", "verdict": "unclear", "checks": []}
    checks = []
    visited = set()
    for citation in citations:
        key = (citation.get("file_sha256"), citation.get("pdf_page"))
        if key in visited:
            continue
        visited.add(key)
        if len(checks) == max_citations:
            break
        if key[0] not in file_map:
            checks.append({"status": "source_not_locally_available", "verdict": "unclear"})
            continue
        try:
            check = verify_with_groq(question, result["answer"], [citation],
                                     pdf_dir=pdf_dir, original_files_by_sha=file_map,
                                     client=client, vision=vision)
            checks.append({**check, "verdict": check["reviewer_verdict"]})
        except Exception as exc:
            checks.append(_unavailable(exc))
    verdicts = {c["verdict"] for c in checks}
    if "contradicted" in verdicts:
        return {"status": "model_detected_contradiction", "verdict": "contradicted", "checks": checks}
    if "partial" in verdicts:
        return {"status": "model_partial", "verdict": "partial", "checks": checks}
    if verdicts == {"supported"} and checks:
        return {"status": "model_supported_not_certified", "verdict": "supported", "checks": checks}
    if any(c["status"] == "provider_unavailable" for c in checks):
        return {"status": "verifier_unavailable", "verdict": "unclear", "checks": checks}
    return {"status": "model_unclear", "verdict": "unclear", "checks": checks}


def _has_numeric_claim(question: str, answer: str) -> bool:
    intent = inferred_intent(question)
    if not intent["metrics"]:
        return False
    # Only financial measured amounts/ratios qualify, not years/dates in text.
    return bool(re.search(r"(?i)(?:₹|\b(?:rs|inr|usd|crore|billion|million|percent|percentage|margin)\b|\d\s*%)", answer))


def run_query(store, embedder, question: str, *, provider: str = "offline",
              ticker: str | None = None, fiscal_year: str | None = None,
              as_of: str | None = None, top_k: int = 5,
              mode: str = "hybrid", model: str = "openai/gpt-oss-120b",
              curated_path: str | Path | None = None,
              automatic_path: str | Path | None = None,
              pdf_dir: str | Path = "data/raw/research_reports",
              verify_financial: bool = True, verify_vision: bool = False,
              client=None, cross_encoder=None, pinecone_index: str | None = None,
              allow_curated: bool = True) -> dict:
    """Returns a complete, explainable answer with verification metadata.

    Supports normal CLI queries and offline fixture tests. No network calls
    without provider='groq'; retries never silently rewrite the source bank.
    """
    docs = store.list_documents()
    eligible = {d.sha256 for d in docs if (not ticker or d.spec.ticker == ticker.upper())
                and (not fiscal_year or d.spec.fiscal_year == fiscal_year)
                and (not as_of or d.spec.published_on <= as_of)}
    curated = read_facts(curated_path) if allow_curated and curated_path and Path(curated_path).is_file() else []
    automatic = [f for f in load_auto_bank(automatic_path) if f["file_sha256"] in {d.sha256 for d in docs}] if automatic_path and Path(automatic_path).is_file() else []
    file_map = _source_map(store, curated, automatic, pdf_dir)
    primary = (fact_answer(question, curated, ticker=ticker, eligible_sha256=eligible, as_of=as_of)
               if curated else None)
    route = "source_checked_curated_json"
    if not primary or primary.get("status") != "answered":
        derived = automatic_fact_answer(question, automatic, ticker=ticker, source_hashes=eligible, as_of=as_of)
        if derived is not None:
            # Auto JSON rows must still match the same physical table cell and
            # year header in the exact source PDF (not just the same page).
            selected = [f for f in automatic if f["fact_id"] in {c["fact_id"] for c in derived["citations"]}]
            try:
                confirm_original_table_rows(selected, pdf_dir=pdf_dir)
            except (ResearchError, OSError, ValueError):
                derived = None
            if derived is not None:
                primary = derived
                route = "automatic_source_table_json"
    hits = []
    if not primary or primary.get("status") not in {"answered", "needs_review"}:
        pool = max(top_k * 5, 20)
        pool = min(30, pool)
        initial = search(store, question, embedder, ticker=ticker, fiscal_year=fiscal_year,
                         as_of=as_of, top_k=pool, mode=mode, pinecone_index=pinecone_index)
        hits = rerank_hits(question, initial, auto_facts=automatic,
                           curated_facts=curated, cross_encoder=cross_encoder, top_k=top_k)
        content = structured_evidence(hits, auto_facts=automatic, curated_facts=curated)
        response = respond(question, hits, provider=provider, model=model,
                           client=_provider_client(client) if provider == "groq" else None,
                           facts_path=None, structured_facts=content)
        primary = _provider_status(response.to_dict())
        route = "financial_scope_reranked_document_rag"
    if primary is None:
        primary = {"question": question, "answer": "Insufficient report evidence",
                   "status": "abstained", "citations": [], "provider": provider}
    out = {**primary, "route": route,
           "verification_status": "not_requested", "source_page_verification": None,
           "retrieved_physical_pages": [(h.document.sha256, h.chunk.page_no) for h in hits]}
    if not verify_financial or provider != "groq" or not _has_numeric_claim(question, primary.get("answer", "")):
        return out
    if not primary.get("citations"):
        out["verification_status"] = "missing_citation"
        out["status"] = "needs_review" if out["status"] == "answered" else out["status"]
        return out
    active_client = _provider_client(client)
    review = _verify_result(question, out, client=active_client, pdf_dir=Path(pdf_dir),
                            file_map=file_map, vision=verify_vision)
    out["source_page_verification"] = review
    out["verification_status"] = review["status"]
    if route == "automatic_source_table_json" and review["verdict"] == "supported":
        # Still only model-approved, NEVER human/source-certified.
        out["status"] = "needs_review"
    if review["verdict"] == "supported":
        return out
    if review["verdict"] != "contradicted":
        if out["status"] == "answered":
            out["status"] = "needs_review"
        return out
    # Contradiction: NEVER display confidently. Use deterministic trusted facts
    # where available, otherwise one evidence-focused retrieval/answer retry.
    if route == "source_checked_curated_json":
        out["status"] = "needs_review"
        out["note"] = "Verifier disagrees with original source-checked fact; manual source audit required"
        return out
    if not hits:
        out["status"] = "needs_review"
        out["note"] = "Verifier rejected automatic table attribution; no unsafe numerical substitution"
        return out
    refined_query = question + " financial reporting segment basis unit period table heading"
    refreshed = search(store, refined_query, embedder, ticker=ticker,
                       fiscal_year=fiscal_year, as_of=as_of,
                       top_k=30, mode=mode, pinecone_index=pinecone_index)
    reranked = rerank_hits(question, refreshed, auto_facts=automatic,
                           curated_facts=curated, cross_encoder=cross_encoder, top_k=top_k)
    alternate = respond(question, reranked, provider="groq", model=model, client=active_client,
                        structured_facts=structured_evidence(reranked, auto_facts=automatic, curated_facts=curated))
    revised = _provider_status(alternate.to_dict())
    if revised.get("status") not in {"answered", "needs_review"}:
        out["status"] = "needs_review"
        out["note"] = "Contradicted; corrected retry did not return source-backed evidence"
        out["corrective_attempt"] = {"status": revised.get("status"), "note": revised.get("note")}
        return out
    review2 = _verify_result(question, revised, client=active_client, pdf_dir=Path(pdf_dir),
                             file_map=file_map, vision=verify_vision)
    if review2["verdict"] == "supported":
        return {**revised, "route": route + "+source_corrective_retry",
                "status": "needs_review",  # a second LLM cannot certify
                "note": "Corrected answer supported by independent AI reviewer; requires confirmation",
                "verification_status": review2["status"],
                "source_page_verification": review2,
                "retrieved_physical_pages": [(h.document.sha256, h.chunk.page_no) for h in reranked]}
    out["status"] = "needs_review"
    out["note"] = "Original cited answer contradicted; corrective retry was not verified"
    out["corrective_attempt"] = {"status": revised.get("status"), "review": review2}
    return out
