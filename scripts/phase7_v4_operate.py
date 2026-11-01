"""RiskLens Phase 7 V4 full-corpus financial extraction and independent benchmark.

Sample use:
 python scripts/phase7_v4_operate.py --build-auto --evaluate --live --vision-budget 8

PDFs, SQLite embeddings and credentials stay on owner's machine. The script
writes source-verifiable JSON and per-case checkpoints, never pushes to GitHub,
changes Pinecone, retrains models or reuses the frozen ML holdout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re

from risklens.research.config import load_project_env
from risklens.research.contracts import ResearchError
from risklens.research.financial_auto_v4 import build_automatic_bank, hydrate_prebuilt_bank
from risklens.research.index import ResearchStore
from risklens.research.retrieval import _QA_QUARANTINE
from risklens.research.structured_pipeline_v4 import run_query

BASE = Path("data/processed/research/real_acceptance/phase7_completion")


def source_manifest(inventory: Path, pdf_dir: Path) -> list[dict]:
    raw = json.loads(inventory.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or not isinstance(raw.get("observations"), list):
        raise ResearchError("Expected original five-report source inventory")
    dedup = {}
    for f in raw["observations"]:
        name = f["source_file"]
        s = dedup.setdefault(name, {k: f[k] for k in ("ticker", "company", "source_file", "file_sha256",
                                                    "source_url", "published_on", "fiscal_year")})
        if s["file_sha256"] != f["file_sha256"]:
            raise ResearchError("Conflicting report hashes in inventory")
    if len(dedup) < 5:
        raise ResearchError("All five official PDFs must be identified before automatic extraction")
    return [{**s, "pdf_path": str(pdf_dir / name)} for name, s in sorted(dedup.items())]


def grade_response(case: dict, response: dict) -> dict:
    """Machine-assess numeric equivalence, refusal and reference page separately.

    This is NOT human-certified semantic accuracy. Incorrect scope on another
    page remains an independent review problem even for a matching number.
    """
    status = response.get("status", "missing")
    if status == "provider_unavailable":
        return {"result": "not_scored_provider_unavailable", "value_matched": None, "source_page_matched": None}
    if not case["answerable"]:
        return {"result": "correct_refusal" if status == "abstained" else "unsupported_question_not_safely_refused",
                "value_matched": None, "source_page_matched": None}
    if status not in {"answered", "needs_review"}:
        return {"result": "answerable_not_answered", "value_matched": False, "source_page_matched": False}
    expected = Decimal(case["metric_value"])
    answer = response.get("answer", "")
    figures = []
    for m in re.finditer(r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9])", answer):
        try:
            figures.append(Decimal(m.group().replace(",", "")))
        except InvalidOperation:
            pass
    # `10000 INR thousand` = 1 INR crore, and 100 crore = 1 INR billion.
    equivalences = {expected}
    if case.get("unit") == "thousand_rupees":
        equivalences.add(expected / Decimal(10000))
        equivalences.add(expected / Decimal(1000000))
    if case.get("unit") == "crore":
        equivalences.add(expected / 100)
        equivalences.add(expected * 10000)
    matched = any(any(abs(f - v) <= max(Decimal("0.000001"), abs(v) * Decimal("0.0000001")) for v in equivalences) for f in figures)
    cites = response.get("citations", [])
    page = case.get("source_physical_pdf_page")
    page_ok = any(c.get("file_sha256") == case.get("source_sha256") and c.get("pdf_page") == page for c in cites)
    if status == "needs_review":
        verdict = "matched_but_needs_review" if matched else "unverified_or_wrong_numeric_claim"
    else:
        verdict = "numeric_value_matched_not_semantically_certified" if matched else "numeric_value_wrong_or_missing"
    return {"result": verdict, "value_matched": matched, "source_page_matched": page_ok}


class LazyEmbedder:
    name = "sentence-transformers/all-MiniLM-L6-v2"
    dimension = 384
    def __init__(self):
        self.instance = None
    def encode(self, texts):
        if self.instance is None:
            from risklens.research.embeddings import create_embedder
            self.instance = create_embedder("sentence")
        return self.instance.encode(texts)


def evaluate(gold: Path, *, db: Path, curated: Path, automatic: Path,
             pdf_dir: Path, output: Path, live: bool,
             vision: bool = False, rerun_unavailable: bool = False,
             cross_encoder_model: str | None = None) -> dict:
    bank = json.loads(gold.read_text(encoding="utf-8"))
    if bank.get("schema_version") != 1 or len(bank.get("questions", [])) < 20:
        raise ResearchError("Incomplete independent financial evaluation benchmark")
    records: dict[str, dict] = {}
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        records = {x["id"]: x for x in previous.get("cases", [])}
    embedder = LazyEmbedder()
    ce = None
    if cross_encoder_model:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(cross_encoder_model)
    with ResearchStore(db) as store:
        available_sha = {d.sha256 for d in store.list_documents()}
        for i, case in enumerate(bank["questions"], 1):
            old = records.get(case["id"])
            if old and (not live or old.get("live", False)) and not (rerun_unavailable and old.get("response", {}).get("status") == "provider_unavailable"):
                continue
            # Unseen expected answers are NEVER supplied to the model pipeline.
            if case.get("source_sha256") and case["source_sha256"] not in available_sha:
                raise ResearchError(f"Missing real indexed source for benchmark {case['id']}")
            response = run_query(store, embedder, case["question"], provider="groq" if live else "offline",
                                 ticker=case["ticker"], curated_path=curated,
                                 automatic_path=automatic, pdf_dir=pdf_dir,
                                 verify_financial=live, verify_vision=vision, cross_encoder=ce)
            graded = grade_response(case, response)
            records[case["id"]] = {"id": case["id"], "question": case["question"],
                                   "answerable": case["answerable"], "reference_page": case.get("source_physical_pdf_page"),
                                   "response": response, "automated_diagnostic": graded, "live": live}
            result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                      "reference_policy": "preexisting independent PDF-source labels; AI-assisted, not externally human-certified",
                      "question_count": len(bank["questions"]), "cases": [records[c["id"]] for c in bank["questions"] if c["id"] in records]}
            output.parent.mkdir(parents=True, exist_ok=True)
            tmp = output.with_suffix(output.suffix + ".tmp")
            tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp.replace(output)
            print(f"CASE {i}/{len(bank['questions'])} {case['id']}: {response['status']} | {graded['result']}", flush=True)
    completed = [records[c["id"]] for c in bank["questions"] if c["id"] in records]
    safe = sum(x["automated_diagnostic"]["result"] == "numeric_value_matched_not_semantically_certified" for x in completed)
    page_match = sum(x["automated_diagnostic"].get("source_page_matched") is True for x in completed)
    unavailable = sum(x["response"]["status"] == "provider_unavailable" for x in completed)
    return {"questions": len(bank["questions"]), "processed": len(completed),
            "numeric_value_matched_with_answered_status": safe,
            "exact_curated_reference_pages_cited": page_match,
            "provider_unavailable": unavailable,
            "note": "Metric values/page IDs are partial automatic diagnostics, not complete claim-level audited accuracy; curated page labels non-exhaustive.",
            "output": str(output)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-auto", action="store_true")
    ap.add_argument("--prebuilt-bank", type=Path, help="Reuse full scan only if all five PDF SHA256s and every native table row recheck")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--live", action="store_true", help="Opt-in real Groq generation and verification")
    ap.add_argument("--vision-budget", type=int, default=0, help="Early ingestion Groq Vision calls across complex pages")
    ap.add_argument("--verify-vision", action="store_true", help="Pass cited original PDF page images to reviewer")
    ap.add_argument("--max-pages-per-pdf", type=int, help="Development-only scan cap; omit for complete corpus")
    ap.add_argument("--rerun-unavailable", action="store_true")
    ap.add_argument("--cross-encoder-model", help="Optional CrossEncoder model for financial reranking")
    ap.add_argument("--pdf-dir", type=Path, default=Path("data/raw/research_reports"))
    ap.add_argument("--source-inventory", type=Path, default=Path("docs/phase7_financial_source_inventory.json"))
    ap.add_argument("--curated", type=Path, default=BASE / "financial_facts_v3.json")
    ap.add_argument("--auto-bank", type=Path, default=BASE / "financial_facts_auto_v4.json")
    ap.add_argument("--gold", type=Path, default=Path("docs/phase7_heldout_v4.json"))
    ap.add_argument("--db", type=Path, default=Path("data/processed/research/real_acceptance/real_index.sqlite3"))
    ap.add_argument("--results", type=Path, default=BASE / "phase7_v4_independent_evaluation.json")
    args = ap.parse_args()
    if not args.build_auto and not args.evaluate:
        ap.error("Select --build-auto, --evaluate or both")
    if args.vision_budget < 0 or (args.vision_budget and not args.build_auto):
        ap.error("Vision budget must be nonnegative and requires --build-auto")
    if args.live and not args.evaluate and not args.vision_budget:
        ap.error("--live without --evaluate has no work")
    load_project_env()
    if args.build_auto:
        client = None
        if args.vision_budget:
            import os
            from groq import Groq
            if not os.getenv("GROQ_API_KEY"):
                raise ResearchError("GROQ_API_KEY is required for early vision extraction")
            client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=70, max_retries=1)
        sources = source_manifest(args.source_inventory, args.pdf_dir)
        quarantine = {}
        for h, pn in _QA_QUARANTINE:
            quarantine.setdefault(h, []).append(pn)
        if args.prebuilt_bank:
            summary = hydrate_prebuilt_bank(args.prebuilt_bank, args.auto_bank,
                                            pdf_dir=args.pdf_dir, vision_client=client,
                                            vision_page_budget=args.vision_budget,
                                            quarantine_by_sha=quarantine)
        else:
            summary = build_automatic_bank(sources, output=args.auto_bank,
                                           max_pages=args.max_pages_per_pdf,
                                           quarantine_by_sha=quarantine,
                                           vision_client=client, vision_page_budget=args.vision_budget)
        print("AUTO FINANCIAL JSON EXTRACTION:\n" + json.dumps(summary, indent=2), flush=True)
    if args.evaluate:
        result = evaluate(args.gold, db=args.db, curated=args.curated,
                          automatic=args.auto_bank, pdf_dir=args.pdf_dir,
                          output=args.results, live=args.live,
                          vision=args.verify_vision,
                          rerun_unavailable=args.rerun_unavailable,
                          cross_encoder_model=args.cross_encoder_model)
        print("INDEPENDENT BENCHMARK:\n" + json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
