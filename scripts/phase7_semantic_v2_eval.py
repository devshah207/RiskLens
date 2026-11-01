"""Compare original 20-question retrospective diagnostics and a new 15-case challenge.

Scores are source-anchored/provisional, not exhaustive human-certified relevance.
Live Groq calls require --live and use only the existing ignored local SQLite index.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/research/real_acceptance/phase7_completion"
OUT = DATA / "semantic_v2"
INDEX = ROOT / "data/processed/research/real_acceptance/real_index.sqlite3"


def save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def add_evidence(store, documents: list, item: dict, page_num: int, needle: str, source_hash: str) -> dict:
    matching = [doc for doc in documents if doc.sha256 == source_hash]
    if len(matching) != 1:
        raise ValueError(f"Source fingerprint not uniquely indexed for {item['id']}")
    doc = matching[0]
    page = store.page_text(doc.doc_id, page_num)
    where = page.lower().find(needle.lower())
    if where < 0:
        raise ValueError(f"Source quote not in indexed page: {item['id']} page {page_num} {needle!r}")
    quote = page[max(0, where - 70):min(len(page), where + len(needle) + 200)]
    if len(quote.strip()) < 12:
        raise ValueError("Too little verbatim evidence")
    return {"doc_id": doc.doc_id, "page_no": page_num, "quote": quote}


def make_gold(store) -> tuple[dict, dict]:
    documents = store.list_documents()
    old = json.loads((DATA / "source_anchored_gold_20_PROVISIONAL.json").read_text())
    old = json.loads(json.dumps(old))
    by_id = {q["id"]: q for q in old["questions"]}
    if len(by_id) != 20:
        raise ValueError("Original 20-item baseline is not present")
    additions = json.loads((ROOT / "docs/phase7_retrospective_reference_addenda.json").read_text())
    for row in additions["addenda"]:
        q = by_id[row["question_id"]]
        item = {"id": q["id"]}
        extra = add_evidence(store, documents, item, row["physical_pdf_page"],
                             row["exact_text_needle"], row["source_sha256"])
        if not any(e["doc_id"] == extra["doc_id"] and e["page_no"] == extra["page_no"]
                   for e in q["evidence"]):
            q["evidence"].append(extra)
    old["verification_method"] = "AI_assisted_RETROSPECTIVE_post_ranking_reference_expansion_NOT_human_certified"
    old["caveat"] = ("Retrospective alternate reference pages were added after looking at the previous rankings. "
                     "Do not compare this score to the prior score as if it were a model improvement.")

    challenge = json.loads((ROOT / "docs/phase7_new_challenge_set_v1.json").read_text())
    new_questions = []
    for src in challenge["questions"]:
        item = {"id": src["id"], "ticker": src["ticker"], "question": src["question"],
                "answerable": src["answerable"], "fiscal_year": "2024-25",
                "as_of": "2026-10-30", "verified_on": "2026-10-30",
                "verified_by": "AI-assisted source review; NOT human-certified", "evidence": []}
        if src["answerable"]:
            for needle in src["evidence_needles"]:
                item["evidence"].append(add_evidence(
                    store, documents, src, src["physical_pdf_page"], needle, src["source_sha256"]
                ))
        new_questions.append(item)
    new = {"verification_method": "AI_assisted_new_question_set_NOT_human_certified",
           "question_set_frozen_before_first_v2_evaluation": True,
           "caveat": "Known-page source evidence only; labels are not exhaustively complete or human-certified.",
           "questions": new_questions}
    return old, new


def run_live(gold: dict, *, count: int, label: str, embedder) -> dict:
    from risklens.research.answer import respond
    from risklens.research.index import ResearchStore
    from risklens.research.retrieval import search

    results = []
    with ResearchStore(INDEX) as store:
        for i, q in enumerate(gold["questions"][:count], 1):
            hits = search(store, q["question"], embedder, ticker=q["ticker"],
                          fiscal_year=q.get("fiscal_year"), as_of=q.get("as_of"),
                          mode="hybrid", top_k=5)
            response = respond(q["question"], hits, provider="groq")
            for retry in range(2):
                if "RateLimitError" not in response.note:
                    break
                print(f"{label} {q['id']}: rate limited; retry {retry + 1}/2 after 60s", flush=True)
                time.sleep(60)
                response = respond(q["question"], hits, provider="groq")
            results.append({"id": q["id"], "answerable": q["answerable"],
                            "response": response.to_dict(),
                            "retrieved": [{"doc_id": h.document.doc_id,
                                           "page_no": h.chunk.page_no,
                                           "chunk_id": h.chunk.chunk_id} for h in hits],
                            "financial_answer_correct": None,
                            "claim_level_citation_support": None})
            save(OUT / f"groq_{label}_review_required.json",
                 {"status": "LIVE_AI_OUTPUT_NOT_HUMAN_CERTIFIED", "cases": results})
            print(f"GROQ {label} {i}/{count}: {q['id']} -> {response.status}", flush=True)
    statuses = {status: sum(v["response"]["status"] == status for v in results)
                for status in ("answered", "needs_review", "abstained")}
    return {"completed": len(results), **statuses,
            "file": str(OUT / f"groq_{label}_review_required.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Explicitly make Groq API calls")
    parser.add_argument("--live-old-count", type=int, default=20)
    parser.add_argument("--live-challenge-count", type=int, default=15)
    args = parser.parse_args()
    if not INDEX.is_file():
        raise SystemExit("Real 5-report SQLite index not found; run from RiskLens root")
    from risklens.research.index import ResearchStore
    from risklens.research.embeddings import SentenceEmbedder
    from risklens.research.evaluation import evaluate
    with ResearchStore(INDEX) as store:
        old, new = make_gold(store)
        orig = OUT / "retrospective_20_gold_not_human_certified.json"
        fresh = OUT / "new_15_gold_not_human_certified.json"
        save(orig, old)
        save(fresh, new)
        embedder = SentenceEmbedder()
        old_results = evaluate(store, embedder, orig, provider="offline")
        new_results = evaluate(store, embedder, fresh, provider="offline")
    save(OUT / "retrospective_20_retrieval.json", old_results)
    save(OUT / "new_15_retrieval.json", new_results)
    for label, data in (("RETROSPECTIVE 20", old_results), ("NEW 15", new_results)):
        print(f"{label} (provisional, not exhaustive, not human-certified):", flush=True)
        for mode, metrics in data["retrieval"].items():
            print(f" {mode}: curated-page recall@5={metrics['recall_at_k']:.3f} MRR={metrics['mrr']:.3f}", flush=True)
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "original_20_retrospective": old_results["retrieval"],
               "new_15_provisional": new_results["retrieval"],
               "warning": "Not externally certified accuracy and not exhaustive full-corpus recall.",
               "groq": {"status": "not_run"}}
    if args.live:
        if not 0 <= args.live_old_count <= 20 or not 0 <= args.live_challenge_count <= 15:
            raise SystemExit("Live question counts exceed allowed ranges")
        live = {}
        if args.live_old_count:
            live["old"] = run_live(old, count=args.live_old_count, label="old20_v2", embedder=embedder)
        if args.live_challenge_count:
            live["new"] = run_live(new, count=args.live_challenge_count, label="new15_v2", embedder=embedder)
        summary["groq"] = live
    save(OUT / "semantic_v2_summary.json", summary)
    print("SAVED:", OUT, flush=True)
    print("No GitHub push or merge; generated scores remain provisional.")


if __name__ == "__main__":
    main()
