"""Verified-gold retrieval evaluation; never invent source-backed answer metrics."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from .answer import respond
from .contracts import ResearchError, valid_iso_date
from .retrieval import search


def _validate_gold(data: dict, store) -> list[dict]:
    if not isinstance(data, dict) or not data.get("questions"):
        raise ResearchError("Gold dataset requires nonempty questions array")
    questions = data["questions"]
    seen = set()
    for q in questions:
        if not isinstance(q, dict) or not isinstance(q.get("id"), str) or not q.get("question"):
            raise ResearchError("Gold item requires id and question")
        if q["id"] in seen:
            raise ResearchError("Duplicate gold question ID")
        seen.add(q["id"])
        if not q.get("verified_by") or not q.get("verified_on"):
            raise ResearchError("Every gold question needs human verified_by and verified_on")
        valid_iso_date(q["verified_on"])
        if not isinstance(q.get("answerable"), bool):
            raise ResearchError("Gold question answerable must be boolean")
        evidence = q.get("evidence", [])
        if not isinstance(evidence, list) or (q["answerable"] and not evidence) or (not q["answerable"] and evidence):
            raise ResearchError("Gold evidence must exist exactly for answerable questions")
        for entry in evidence:
            if not isinstance(entry, dict) or not all(k in entry for k in ("doc_id", "page_no", "quote")):
                raise ResearchError("Gold evidence requires doc_id, page_no, and quote")
            quote = entry["quote"]
            if not isinstance(quote, str) or len(quote.strip()) < 12:
                raise ResearchError("Gold quote must be a manually verified >=12 character exact span")
            if quote not in store.page_text(entry["doc_id"], entry["page_no"]):
                raise ResearchError(f"Gold evidence is absent from indexed PDF page: {q['id']}")
        if q.get("as_of"):
            valid_iso_date(q["as_of"])
    return questions


def evaluate(store, embedder, gold_file: str | Path, *, top_k: int = 5,
             provider: str = "offline", model: str = "openai/gpt-oss-120b",
             client=None) -> dict:
    gold_path = Path(gold_file)
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    questions = _validate_gold(gold, store)
    by_mode = {}
    cases = []
    for mode in ("lexical", "dense", "hybrid"):
        recalls, precisions, reciprocal, latencies = [], [], [], []
        for q in questions:
            start_time = perf_counter()
            hits = search(store, q["question"], embedder, ticker=q.get("ticker"),
                          fiscal_year=q.get("fiscal_year"), as_of=q.get("as_of"),
                          top_k=top_k, mode=mode)
            elapsed = (perf_counter() - start_time) * 1000
            latencies.append(elapsed)
            relevant = {(e["doc_id"], int(e["page_no"])) for e in q.get("evidence", [])}
            ranks = [idx for idx, h in enumerate(hits, 1)
                     if (h.document.doc_id, h.chunk.page_no) in relevant]
            recovered = {(h.document.doc_id, h.chunk.page_no) for h in hits} & relevant
            if relevant:
                recalls.append(len(recovered) / len(relevant))
                precisions.append(sum((h.document.doc_id, h.chunk.page_no) in relevant for h in hits) / top_k)
                reciprocal.append(1 / min(ranks) if ranks else 0.0)
            case = {"mode": mode, "question_id": q["id"], "answerable": q["answerable"],
                    "expected": sorted([list(v) for v in relevant]),
                    "retrieved": [{"doc_id": h.document.doc_id, "page_no": h.chunk.page_no,
                                   "chunk_id": h.chunk.chunk_id} for h in hits],
                    "recall_at_k": len(recovered) / len(relevant) if relevant else None,
                    "first_relevant_rank": min(ranks) if ranks else None,
                    "retrieval_latency_ms": round(elapsed, 2)}
            if mode == "hybrid":
                reply = respond(q["question"], hits, provider=provider, model=model, client=client)
                cited = {(c["doc_id"], c["pdf_page"]) for c in reply.citations}
                case["answer_status"] = reply.status
                case["citation_page_precision"] = (len(cited & relevant) / len(cited)) if cited and relevant else None
                case["abstention_correct"] = (reply.status == "abstained") if not q["answerable"] else None
                case["answer_verification"] = "NOT_AUTOMATED_REQUIRES_HUMAN_REVIEW"
                case["answer_text"] = reply.answer
                case["llm_latency_ms"] = reply.api_latency_ms
                case["api_total_tokens"] = reply.api_tokens
            cases.append(case)
        by_mode[mode] = {"n_answerable": len(recalls), "recall_at_k": mean(recalls) if recalls else None,
                         "precision_at_k": mean(precisions) if precisions else None,
                         "mrr": mean(reciprocal) if reciprocal else None,
                         "mean_retrieval_latency_ms": round(mean(latencies), 2) if latencies else None}
    provenance = gold.get("verification_method", "unspecified")
    return {"gold_source_file": str(gold_path), "gold_question_count": len(questions),
            "gold_verification_method": provenance,
            "gold_human_certified": provenance == "human_verified",
            "top_k": top_k, "retrieval": by_mode, "cases": cases,
            "caveats": ["AI-assisted source-anchored gold is provisional until independent owner verification.",
                        "Citation ID and literal quote validation is not semantic answer correctness.",
                        "Offline evidence-only responses are not LLM answers.",
                        "No automatic score for financial factuality or faithfulness is claimed.",
                        "Model API calls, if enabled, may incur usage/rate limits."]}
