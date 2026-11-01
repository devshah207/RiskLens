"""Thin Phase 7 command handlers plugged into the maintained `risklens` CLI."""
from __future__ import annotations

from .config import load_project_env

import argparse
import json
from pathlib import Path

from .answer import respond
from .financial_facts import default_fact_bank_path, read_facts, fact_answer
from .contracts import valid_iso_date
from .chunking import make_chunks
from .contracts import ResearchError, read_manifest
from .embeddings import create_embedder
from .evaluation import evaluate
from .index import ResearchStore, mirror_pinecone
from .ingest import load_pdf, page_audit
from .retrieval import search

DEFAULT_DB = "data/processed/research/rag_index.sqlite3"


def configure_parsers(sub) -> None:
    ingest = sub.add_parser("research-ingest", help="Phase 7: index local annual-report PDFs with page provenance")
    ingest.add_argument("--manifest", required=True, help="Verified sources manifest, with local PDFs")
    ingest.add_argument("--db", default=DEFAULT_DB)
    ingest.add_argument("--parser", choices=("docling", "pymupdf"), default="docling")
    ingest.add_argument("--embedder", choices=("sentence", "hash-fixture"), default="sentence")
    ingest.add_argument("--vision-fallback", action="store_true", help="Opt-in external Groq Vision OCR for low-text pages; transcripts always excluded pending review")
    ingest.add_argument("--pinecone-index", help="Opt-in cloud mirror; uses existing index and PINECONE_API_KEY")
    ingest.add_argument("--max-tokens", type=int, default=450)
    ingest.add_argument("--overlap", type=int, default=50)
    ingest.add_argument("--audit-dir", default="data/processed/research/audits")
    ingest.add_argument("--review-dir", help="Optional local JSON overrides named SHA256.json; only manually verified pages and exact source hash")
    ask = sub.add_parser("research-ask", help="Phase 7: retrieve source evidence and optionally ask Groq")
    ask.add_argument("question")
    ask.add_argument("--db", default=DEFAULT_DB)
    ask.add_argument("--embedder", choices=("sentence", "hash-fixture"), default="sentence")
    ask.add_argument("--ticker")
    ask.add_argument("--fiscal-year")
    ask.add_argument("--as-of", help="Only include PDFs published no later than YYYY-MM-DD")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--mode", choices=("hybrid", "dense", "lexical"), default="hybrid")
    ask.add_argument("--provider", choices=("offline", "groq"), default="offline")
    ask.add_argument("--model", default="openai/gpt-oss-120b")
    ask.add_argument("--facts", default=str(default_fact_bank_path()), help="Source-checked financial fact bank")
    ask.add_argument("--no-facts", action="store_true", help="Disable structured financial facts")
    ask.add_argument("--pinecone-index", help="Read dense candidates from existing Pinecone index")
    ask.add_argument("--legacy-v3", action="store_true", help="Use old V3 query execution without V4 attribution")
    ask.add_argument("--combined-v5", action="store_true", help="Opt in to guarded V3+V4 combined financial QA")
    ask.add_argument("--auto-facts", default="data/processed/research/real_acceptance/phase7_completion/financial_facts_auto_v4.json")
    ask.add_argument("--pdf-dir", default="data/raw/research_reports")
    ask.add_argument("--no-page-verification", action="store_true", help="Disable additional original-PDF Groq reviewer for numeric answers")
    ask.add_argument("--verify-vision", action="store_true", help="Review 2x original cited page as images; uses vision quota")
    ask.add_argument("--cross-encoder-model", help="Optional sentence-transformers CrossEncoder local model for second-stage ranking")
    ev = sub.add_parser("research-evaluate", help="Phase 7: validate human-verified gold retrieval/citations")
    ev.add_argument("--gold", required=True)
    ev.add_argument("--db", default=DEFAULT_DB)
    ev.add_argument("--embedder", choices=("sentence", "hash-fixture"), default="sentence")
    ev.add_argument("--top-k", type=int, default=5)
    ev.add_argument("--provider", choices=("offline", "groq"), default="offline")
    ev.add_argument("--model", default="openai/gpt-oss-120b")
    ev.add_argument("--output", default="data/processed/research/evaluation.json")
    info = sub.add_parser("research-inspect", help="Phase 7: inspect indexed sources and provenance")
    info.add_argument("--db", default=DEFAULT_DB)


def run(args: argparse.Namespace) -> int:
    load_project_env()
    if args.command == "research-ask" and not args.legacy_v3:
        if args.combined_v5:
            from .combined_v5 import run_query
        else:
            from .structured_pipeline_v4 import run_query
        from .embeddings import HashFixtureEmbedder, SentenceEmbedder
        # Avoid loading a transformer for answers supplied by trusted JSON.
        class LazyEmbedder:
            name = SentenceEmbedder.name if args.embedder == "sentence" else HashFixtureEmbedder.name
            dimension = SentenceEmbedder.dimension if args.embedder == "sentence" else HashFixtureEmbedder.dimension
            def __init__(self):
                self.actual = None
            def encode(self, texts):
                if self.actual is None:
                    self.actual = create_embedder(args.embedder)
                return self.actual.encode(texts)
        ce = None
        if args.cross_encoder_model:
            from sentence_transformers import CrossEncoder
            ce = CrossEncoder(args.cross_encoder_model)
        lazy = LazyEmbedder()
        with ResearchStore(args.db) as store:
            data = run_query(store, lazy, args.question, provider=args.provider,
                             ticker=args.ticker, fiscal_year=args.fiscal_year, as_of=args.as_of,
                             top_k=args.top_k, mode=args.mode, model=args.model,
                             curated_path=None if args.no_facts else args.facts,
                             automatic_path=None if args.no_facts else args.auto_facts,
                             pdf_dir=args.pdf_dir,
                             verify_financial=not args.no_page_verification,
                             verify_vision=args.verify_vision,
                             cross_encoder=ce, pinecone_index=args.pinecone_index,
                             allow_curated=not args.no_facts)
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return 0
    if args.command == "research-ingest":
        specs = read_manifest(args.manifest)
        embedder = create_embedder(args.embedder)
        audit_dir = Path(args.audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        with ResearchStore(args.db) as store:
            for spec in specs:
                doc, pages = load_pdf(spec, parser=args.parser, vision_fallback=args.vision_fallback,
                                      local_base=Path(args.manifest).resolve().parent,
                                      review_dir=args.review_dir)
                # Review-required pages are audited but excluded from retrieval.
                chunks = [c for page in pages if not page.review_required
                          for c in make_chunks(page, max_tokens=args.max_tokens, overlap=args.overlap)]
                result = store.insert(doc, pages, chunks, embedder)
                audit = page_audit(doc, pages)
                audit["index_result"] = result
                audit["excluded_review_pages"] = [p.page_no for p in pages if p.review_required]
                audit_path = audit_dir / f"{doc.doc_id}.json"
                audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                print(json.dumps({"ticker": spec.ticker, "fiscal_year": spec.fiscal_year,
                                  "source_sha256": doc.sha256, "audit": str(audit_path), **result}))
                if args.pinecone_index and result["status"] != "unchanged":
                    count = mirror_pinecone(doc, chunks, embedder, index_name=args.pinecone_index)
                    print(f"Pinecone mirror: {count} vectors; old cloud versions need namespace lifecycle cleanup")
        return 0
    if args.command == "research-inspect":
        with ResearchStore(args.db) as store:
            print(json.dumps([doc.to_dict() for doc in store.list_documents()], indent=2))
        return 0
    # Structured financial facts: preserve metric, entity, scope, period and unit.
    if args.command == "research-ask" and not args.no_facts and Path(args.facts).is_file():
        if args.as_of:
            valid_iso_date(args.as_of)
        with ResearchStore(args.db) as fact_store:
            docs_by_sha = {d.sha256: d.doc_id for d in fact_store.list_documents()
                           if not args.fiscal_year or d.spec.fiscal_year == args.fiscal_year}
            candidate = fact_answer(args.question, read_facts(args.facts), ticker=args.ticker,
                                    eligible_sha256=set(docs_by_sha), as_of=args.as_of)
            if candidate and candidate["status"] in {"answered", "needs_review"}:
                for citation in candidate["citations"]:
                    citation["doc_id"] = docs_by_sha.get(citation["file_sha256"])
                print(json.dumps(candidate, ensure_ascii=False, indent=2))
                return 0
    embedder = create_embedder(args.embedder)
    with ResearchStore(args.db) as store:
        if args.command == "research-ask":
            hits = search(store, args.question, embedder, ticker=args.ticker,
                          fiscal_year=args.fiscal_year, as_of=args.as_of,
                          top_k=args.top_k, mode=args.mode, pinecone_index=args.pinecone_index)
            response = respond(args.question, hits, provider=args.provider, model=args.model,
                               facts_path=None if args.no_facts else args.facts,
                               allowed_source_hashes={doc.sha256 for doc in store.list_documents()},
                               as_of=args.as_of)
            print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "research-evaluate":
            result = evaluate(store, embedder, args.gold, top_k=args.top_k,
                              provider=args.provider, model=args.model)
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps({"output": str(path), "retrieval": result["retrieval"],
                              "gold_question_count": result["gold_question_count"]}, indent=2))
            return 0
    raise ResearchError(f"Unknown research command: {args.command}")
