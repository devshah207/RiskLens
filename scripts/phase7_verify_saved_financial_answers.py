"""Optional Groq second-opinion verification against ORIGINAL cited PDF pages.

This is not an LLM self-certification: it checks source file SHA first, then
supplies original-page evidence and stores the second model's advisory verdict.
Source-checked JSON assertions override a conflicting model opinion.

External API calls require --live, and this script never edits the fact bank.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from risklens.research.config import load_project_env
from risklens.research.financial_facts import read_facts
from risklens.research.financial_verifier import deterministic_fact_conflict, verify_with_groq


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="JSON from phase7_fact_eval.py --live")
    ap.add_argument("--fact-bank", default="data/processed/research/real_acceptance/phase7_completion/financial_facts_v3.json")
    ap.add_argument("--pdf-dir", default="data/raw/research_reports")
    ap.add_argument("--output", default="data/processed/research/real_acceptance/phase7_completion/financial_page_second_opinion.json")
    ap.add_argument("--vision", action="store_true", help="Groq Qwen original-page 2x images, potentially higher quota use")
    ap.add_argument("--live", action="store_true", help="Explicitly authorize Groq API calls")
    ap.add_argument("--max-calls", type=int, default=6)
    args = ap.parse_args()
    if not args.live:
        raise SystemExit("No API calls made. Explicitly specify --live when ready for a second-opinion audit.")
    if not 1 <= args.max_calls <= 20:
        raise SystemExit("--max-calls must be 1..20")
    load_project_env()
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("Missing GROQ_API_KEY in ignored .env")
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=60.0, max_retries=1)
    facts = read_facts(args.fact_bank)
    source_names = {f.file_sha256: f.source_file for f in facts}
    raw = json.loads(Path(args.results).read_text(encoding="utf-8"))
    if not isinstance(raw.get("cases"), list):
        raise SystemExit("Invalid evaluation file")
    records = []
    calls = 0
    for case in raw["cases"]:
        result = case.get("result")
        if not isinstance(result, dict) or result.get("status") not in {"answered", "needs_review"}:
            continue
        # Verifying structured facts through Groq is unnecessary; their source
        # page, unit and scope were already checked deterministically.
        if result.get("provider") == "source_checked_financial_facts":
            continue
        question = case["question"]
        answer = result["answer"]
        local = deterministic_fact_conflict(question, answer, facts, ticker=case.get("ticker"))
        record = {"id": case["id"], "deterministic_check": local}
        if calls < args.max_calls and result.get("citations"):
            try:
                record["second_opinion"] = verify_with_groq(
                    question, answer, result["citations"],
                    pdf_dir=args.pdf_dir, original_files_by_sha=source_names,
                    client=client, vision=args.vision)
                calls += 1
            except Exception as exc:
                record["second_opinion"] = {"status": "VERIFIER_FAILED", "error_type": type(exc).__name__}
        records.append(record)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "status": "ADVISORY_AI_SECOND_OPINION_NOT_CERTIFICATION",
        "external_calls_completed": calls,
        "vision_mode": args.vision,
        "cases": records,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("GROQ SECOND OPINION COMPLETE:", output, "calls:", calls)


if __name__ == "__main__":
    main()
