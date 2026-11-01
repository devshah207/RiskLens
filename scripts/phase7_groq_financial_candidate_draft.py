"""Explicitly call Qwen Vision to propose JSON facts for selected PDF pages.

Output is REVIEW REQUIRED and cannot be inserted into source-checked facts.
User's original 2x rendered PDF technique is retained in financial_candidates.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import os

from risklens.research.config import load_project_env
from risklens.research.financial_candidates import draft_page_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", required=True, help="One original official PDF")
    ap.add_argument("--pages", nargs="+", type=int, required=True, help="1-indexed physical PDF pages")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="qwen/qwen3.8-27b")
    ap.add_argument("--max-pages", type=int, default=8)
    args = ap.parse_args()
    if not 1 <= len(set(args.pages)) <= args.max_pages or args.max_pages > 20:
        raise SystemExit("Explicit page count exceeds the per-run vision quota cap")
    load_project_env()
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY missing in ignored .env")
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=60.0, max_retries=2)
    results = []
    for page in sorted(set(args.pages)):
        try:
            result = draft_page_json(args.pdf, page, client=client, model=args.model)
        except Exception as exc:
            # Failures are never interpreted as successfully extracted evidence.
            result = {"physical_pdf_page": page, "status": "VISION_FAILED", "error_type": type(exc).__name__}
        print(f"Page {page}: {result.get('candidate_count', result.get('status'))}", flush=True)
        results.append(result)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status": "UNVERIFIED_DRAFT_ONLY", "pages": results}, indent=2) + "\n")
    print("SAVED UNVERIFIED JSON DRAFT:", output)

if __name__ == "__main__":
    main()
