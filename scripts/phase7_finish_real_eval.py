"""Phase 7 real-source acceptance: one additional ICICI document, evidence-anchored
20-question evaluation, optional Pinecone and Groq tests.

No original XGBoost/LSTM data or model code is touched; source PDF not committed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
# The file is deliberately external to Git. These are the bytes previously audited.
FIFTH_SHA = "774a4a2286824d7ab7b317ba9fe02659577362a121b831736f595773b1033077"
FIFTH_URL = "https://www.icici.bank.in/content/dam/icicibank/managed-assets/docs/investor/annual-reports/2025/icici-bank-2025-financial-statements.pdf"
INDEX = ROOT / "data/processed/research/real_acceptance/real_index.sqlite3"
OUT = ROOT / "data/processed/research/real_acceptance/phase7_completion"
TARGET = ROOT / "data/raw/research_reports/icici-bank-2025-financial-statements.pdf"


def say(*values):
    print(*values, flush=True)


def src_checksum(p: Path) -> str:
    return sha256(p.read_bytes()).hexdigest()


def prepare_source(path_arg: str | None) -> Path:
    locations = [TARGET]
    if path_arg:
        locations.append(Path(path_arg).expanduser())
    locations += [Path.home() / "Downloads" / x for x in
                  ["icici-bank-2025-financial-statements.pdf", "icici_bank_2025_financial_statements.pdf",
                   "icici-bank-2025-financial-statements (1).pdf"]]
    found = next((p for p in locations if p.is_file() and src_checksum(p) == FIFTH_SHA), None)
    if found is None:
        raise RuntimeError("Missing byte-matching fifth PDF. Place the existing file at " + str(TARGET) +
                           " or give --fifth /absolute/path/to/icici-bank-2025-financial-statements.pdf")
    if found != TARGET:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found, TARGET)
    if src_checksum(TARGET) != FIFTH_SHA:
        raise RuntimeError("Fifth ICICI PDF fingerprint mismatch; refusing wrong edition")
    import pymupdf
    with pymupdf.open(str(TARGET)) as doc:
        if len(doc) != 202:
            raise RuntimeError(f"Unexpected fifth PDF page count {len(doc)}")
    say("SOURCE VALIDATED: 202 pages, SHA-256", FIFTH_SHA)
    return TARGET


def ingest_fifth() -> None:
    from risklens.research.contracts import SourceSpec
    from risklens.research.index import ResearchStore
    spec = SourceSpec(
        company="ICICI Bank Limited", ticker="ICICIBANK", fiscal_year="2024-25",
        title="Financial Statements FY2024-25 (separate report)",
        source_url=FIFTH_URL,
        # Conservative: verified accessible to project by this review date.
        # NOT a claim about initial exchange publication date.
        published_on="2026-10-30", local_path=str(TARGET),
        license_note="Public issuer PDF, local research use only; original publication date unverified",
        report_kind="financial_statements", document_version="publisher_pdf_fy2025",
        sector="Banking",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "fifth_source_manifest.json"
    from dataclasses import asdict
    manifest.write_text(json.dumps({"sources": [asdict(spec)]}, indent=2) + "\n")
    with ResearchStore(INDEX) as store:
        existing = [d for d in store.list_documents() if d.sha256 == FIFTH_SHA]
        if existing:
            say("FIFTH DOCUMENT ALREADY INDEXED: no re-embedding", existing[0].doc_id)
            return
    say("INGESTING ICICI 202-PAGE AUDITED FINANCIAL STATEMENTS (PyMuPDF + real MiniLM)")
    subprocess.run([sys.executable, "-m", "risklens.cli", "research-ingest",
        "--manifest", str(manifest), "--db", str(INDEX), "--parser", "pymupdf",
        "--embedder", "sentence", "--audit-dir", str(OUT / "page_audits")],
        cwd=ROOT, check=True)


def build_reference_gold() -> Path:
    """Source-anchored provisional labels; never disguise AI review as human attestation."""
    from risklens.research.index import ResearchStore
    anchors = json.loads((ROOT / "docs/phase7_independent_source_anchors.json").read_text())
    locked = json.loads((ROOT / "docs/phase7_gold_question_candidates.json").read_text())
    source_questions = [(v["id"], v["ticker"], v["question"], v["answerable"])
                        for v in anchors["questions"]]
    original_questions = [(v["id"], v["ticker"], v["question"], v["answerable"])
                          for v in locked["questions"]]
    if source_questions != original_questions or len(source_questions) != 20:
        raise RuntimeError("The original frozen 20 research questions have changed")
    with ResearchStore(INDEX) as store:
        docs = store.list_documents()
        questions = []
        for item in anchors["questions"]:
            q = {"id": item["id"], "ticker": item["ticker"],
                 "question": item["question"], "answerable": item["answerable"],
                 "fiscal_year": "2024-25", "as_of": "2026-10-30",
                 "verified_by": "AI-assisted original-PDF source review (NOT human-certified)",
                 "verified_on": "2026-10-30", "evidence": [],
                 "reference_answer_for_human_review": item.get("reference_answer")}
            if item["answerable"]:
                matches = [d for d in docs if d.sha256 == item["source_sha256"]]
                if len(matches) != 1:
                    raise RuntimeError(f"Expected exactly one source edition for {item['id']}")
                doc = matches[0]
                page_no = item["physical_pdf_page"]
                page = store.page_text(doc.doc_id, page_no)
                for needle in item["evidence_needles"]:
                    index = page.lower().find(needle.lower())
                    if index < 0:
                        raise RuntimeError(f"Verified source anchor absent from indexed PDF {item['id']} p{page_no}: {needle!r}")
                    quote = page[max(0, index - 45): min(len(page), index + len(needle) + 150)]
                    if len(quote.strip()) < 12:
                        raise RuntimeError(f"Insufficient exact source quote {item['id']} {needle!r} {quote!r}")
                    q["evidence"].append({"doc_id": doc.doc_id, "page_no": page_no, "quote": quote})
            questions.append(q)
    result = {"verification_method": "AI_assisted_source_page_review_NOT_human_certified",
              "caveat": "Relevant pages are curated and not exhaustively annotated; report observed page-recall, not complete semantic recall.",
              "questions": questions}
    gold = OUT / "source_anchored_gold_20_PROVISIONAL.json"
    gold.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    say("BUILT 20-CASE REAL-PDF REFERENCE (16 answerable, 4 intentionally unsupported):", gold)
    return gold


def eval_offline(gold: Path, embedder) -> dict:
    from risklens.research.index import ResearchStore
    from risklens.research.evaluation import evaluate
    with ResearchStore(INDEX) as store:
        result = evaluate(store, embedder, gold, top_k=5, provider="offline")
    path = OUT / "retrieval_20_real_sources_PROVISIONAL.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    say("REAL-PDF RETRIEVAL @ 5 (provisional curated pages):")
    for name, data in result["retrieval"].items():
        say(" ", name, "recall", data["recall_at_k"], "MRR", data["mrr"],
            "precision", data["precision_at_k"])
    return result


def cloud_fifth_only() -> dict:
    from risklens.research.config import load_project_env
    from risklens.research.index import ResearchStore, _pinecone_index
    load_project_env()
    key = os.getenv("PINECONE_API_KEY")
    if not key:
        return {"status": "not_configured"}
    index_name = os.getenv("PINECONE_INDEX_NAME", "risklens-minilm384")
    index = _pinecone_index(index_name, key)
    with ResearchStore(INDEX) as store:
        match = [d for d in store.list_documents() if d.sha256 == FIFTH_SHA]
        if len(match) != 1:
            raise RuntimeError("Missing fifth document in SQLite")
        doc = match[0]
        chunks = [row for row in store.candidates(ticker="ICICIBANK") if row[1].doc_id == doc.doc_id]
    say("CLOUD: uploading", len(chunks), "FIFTH-REPORT CHUNKS (idempotent IDs)")
    for off in range(0, len(chunks), 50):
        batch = [{"_id": c.chunk_id, "embedding": list(vector), "ticker": doc.spec.ticker,
                  "fiscal_year": doc.spec.fiscal_year, "published_on": doc.spec.published_on,
                  "document_id": doc.doc_id, "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                  "physical_pdf_page": c.page_no} for c, _, vector in chunks[off:off + 50]]
        index.documents.upsert(namespace="risklens-research", documents=batch)
    # Probe actual retrieval using local MiniLM encoder and confirm ID membership
    from risklens.research.embeddings import SentenceEmbedder
    from risklens.research.index import pinecone_ids
    vec = SentenceEmbedder().encode(["ICICI Bank gross NPA ratio 31 March 2025"])[0]
    ids = pinecone_ids(vec, index_name=index_name, top_k=30)
    local_ids = {c.chunk_id for c, _, _ in chunks}
    matches = len(local_ids.intersection(ids))
    return {"status": "uploaded_and_queried", "fifth_chunks_upserted": len(chunks),
            "remote_results": len(ids), "fifth_doc_matches_top30": matches,
            "note": "A matching ID verifies data linkage, not answer relevance."}


def groq_cases(gold: Path, embedder, n: int) -> dict:
    from risklens.research.config import load_project_env
    from risklens.research.index import ResearchStore
    from risklens.research.answer import respond
    from risklens.research.retrieval import search
    load_project_env()
    if not os.getenv("GROQ_API_KEY"):
        return {"status": "no_key", "completed": 0}
    questions = json.loads(gold.read_text())["questions"]
    # Keep same locked questions and order; distribute initial probes across issuers.
    if n <= 5:
        selected = [next(q for q in questions if q["id"] == qid)
                    for qid in ["RIL01", "HDFC01", "INFY01", "ICICI01", "ICICI05"][:n]]
    else:
        selected = questions[:n]
    path = OUT / "groq_live_answers_REVIEW_REQUIRED.json"
    prior = {}
    if path.exists():
        prior = {case["id"]: case for case in json.loads(path.read_text()).get("cases", [])}
    records = []
    for idx, q in enumerate(selected, 1):
        if q["id"] in prior and "RateLimitError" not in prior[q["id"]]["result"].get("note", ""):
            records.append(prior[q["id"]])
            say("GROQ reuse saved result", q["id"])
            continue
        with ResearchStore(INDEX) as store:
            hits = search(store, q["question"], embedder, ticker=q["ticker"],
                          fiscal_year=q["fiscal_year"], as_of=q["as_of"], top_k=5, mode="hybrid")
        say("GROQ", idx, "/", len(selected), q["id"], "retrieved", len(hits), "chunks")
        response = respond(q["question"], hits, provider="groq")
        for retry in range(3):
            if "RateLimitError" not in response.note:
                break
            say("GROQ rate limit: retry", retry + 1, "of 3 after 45-second cooldown")
            time.sleep(45)
            response = respond(q["question"], hits, provider="groq")
        record = {"id": q["id"], "expected_answerable": q["answerable"],
                  "reference_answer_for_review": q.get("reference_answer_for_human_review"),
                  "result": response.to_dict(),
                  "retrieved": [{"doc_id": h.document.doc_id, "physical_pdf_page": h.chunk.page_no,
                                 "chunk_id": h.chunk.chunk_id, "source_text": h.chunk.text[:5500]}
                                for h in hits],
                  "financial_answer_correct": None,
                  "answer_semantically_supported": None}
        records.append(record)
        path.write_text(json.dumps({"status": "LIVE_OUTPUT_NOT_HUMAN_VERIFIED", "cases": records}, indent=2, ensure_ascii=False) + "\n")
    answered = sum(c["result"]["status"] == "answered" for c in records)
    return {"status": "completed", "n_live": len(records), "n_accepted_answers": answered,
            "n_abstained": sum(c["result"]["status"] == "abstained" for c in records),
            "saved_at": str(path), "financial_answer_accuracy": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fifth", help="Local path of downloaded fifth ICICI PDF, if not in Downloads")
    parser.add_argument("--cloud", action="store_true", help="Explicitly authorize Pinecone and Groq API calls")
    parser.add_argument("--live-answers", type=int, default=5, choices=range(0, 21), metavar="0-20",
                        help="Number of live Groq answer probes, 0-20; default 5; may hit free-tier limits")
    args = parser.parse_args()
    if not INDEX.is_file():
        raise SystemExit(f"Missing existing 1910-chunk SQLite index: {INDEX}")
    prepare_source(args.fifth)
    ingest_fifth()
    gold = build_reference_gold()
    from risklens.research.embeddings import SentenceEmbedder
    embedder = SentenceEmbedder()
    result = eval_offline(gold, embedder)
    cloud = {"status": "skipped_by_user"}
    live = {"status": "skipped_by_user"}
    if args.cloud:
        cloud = cloud_fifth_only()
        if args.live_answers:
            live = groq_cases(gold, embedder, args.live_answers)
    summary = {"utc": datetime.now(timezone.utc).isoformat(),
               "fifth_pdf_sha256": FIFTH_SHA, "original_20_questions_unchanged": True,
               "human_gold_certification": False, "full_relevance_annotations_exhaustive": False,
               "retrieval": result["retrieval"], "cloud": cloud, "groq": live,
               "conclusion": "Source-anchored provisional retrieval metrics; Groq output requires human grading."}
    path = OUT / "phase7_real_evaluation_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    say("PHASE7 SOURCE-GROUNDED EVALUATION COMPLETE:", path)
    say("No ML holdout, original source PDFs, other SQL documents or GitHub branches modified.")


if __name__ == "__main__":
    main()
