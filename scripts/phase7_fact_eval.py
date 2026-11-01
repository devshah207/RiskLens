"""Real-report financial fact diagnostics without Groq calls or hidden shortcuts.

This evaluates *coverage and source-linked facts*, not independently certified
accuracy or production-wide Recall@5. The old 20/new 15 are development sets.

Pass --live to send ONLY cases not answered by source facts through normal RAG,
with an explicit, bounded rate-limit retry. Rate limits never count as abstention
correctness or success. No fitted ML models or Pinecone entries are changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from risklens.research.answer import respond
from risklens.research.embeddings import create_embedder
from risklens.research.financial_facts import read_facts, fact_answer
from risklens.research.index import ResearchStore
from risklens.research.retrieval import search


def evaluate(source_files: list[Path], facts_path: Path, output: Path, *, db: Path | None = None,
             live: bool = False, retries: int = 2) -> dict:
    facts = read_facts(facts_path)
    cases = []
    seen = set()
    for path in source_files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        for case in raw["questions"]:
            key = case["id"]
            if key in seen:
                raise ValueError(f"Duplicate benchmark case {key}")
            seen.add(key)
            cases.append({"id": key, "question": case["question"],
                          "ticker": case["ticker"], "answerable_label": bool(case["answerable"]),
                          "benchmark_file": path.name})

    result = []
    embedder = create_embedder("sentence") if live else None
    store = ResearchStore(db) if live and db else None
    if live and not store:
        raise ValueError("Live mode requires --db pointing to existing five-report research index")
    try:
        shas = {doc.sha256 for doc in store.list_documents()} if store else {f.file_sha256 for f in facts}
        for c in cases:
            r = fact_answer(c["question"], facts, ticker=c["ticker"], eligible_sha256=shas)
            if r and r["status"] == "answered":
                response = r
                route = "original_pdf_source_checked_json"
            elif not live:
                response = None
                route = "unassessed_normal_rag_fallback"
            else:
                route = "groq_semantic_document_rag"
                hits = search(store, c["question"], embedder, ticker=c["ticker"], top_k=5)
                response = None
                for attempt in range(retries + 1):
                    reply = respond(c["question"], hits, provider="groq")
                    if reply.note.startswith("Provider unavailable/malformed: RateLimitError") and attempt < retries:
                        print(f"RATE LIMITED {c['id']}: retry {attempt + 1}/{retries} after 60s", flush=True)
                        time.sleep(60)
                        continue
                    response = reply.to_dict()
                    break
            result.append({**c, "route": route, "result": response})
            print(f"{c['id']}: {route} -> {(response or {}).get('status', 'not_live_tested')}", flush=True)
    finally:
        if store:
            store.conn.close()
    n_structured = sum(r["route"] == "original_pdf_source_checked_json" for r in result)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Source-checked structured fact coverage and optional live fallback; NOT independently human-certified answer accuracy.",
        "n_cases": len(result), "answerable_labeled": sum(r["answerable_label"] for r in result),
        "structured_fact_answers": n_structured,
        "fallback_cases": len(result) - n_structured,
        "source_checked_fact_records": len(facts),
        "live_groq_enabled": live,
        "fall_back_to_groq_does_not_use_frozen_models": True,
        "unseen_external_benchmark": False,
        "cases": result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {k: summary[k] for k in ("n_cases", "answerable_labeled", "structured_fact_answers", "fallback_cases", "source_checked_fact_records", "live_groq_enabled")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fact-bank", default="data/processed/research/real_acceptance/phase7_completion/financial_facts_v3.json")
    ap.add_argument("--output", default="data/processed/research/real_acceptance/phase7_completion/financial_fact_eval_v3.json")
    ap.add_argument("--old", default="docs/phase7_independent_source_anchors.json")
    ap.add_argument("--new", default="docs/phase7_new_challenge_set_v1.json")
    ap.add_argument("--db", default="data/processed/research/real_acceptance/real_index.sqlite3")
    ap.add_argument("--live", action="store_true", help="Explicitly allow API calls for cases not resolved by structured facts")
    args = ap.parse_args()
    print(json.dumps(evaluate([Path(args.old), Path(args.new)], Path(args.fact_bank),
                              Path(args.output), db=Path(args.db), live=args.live), indent=2))


if __name__ == "__main__":
    main()
