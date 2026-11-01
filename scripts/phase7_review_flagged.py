#!/usr/bin/env python3
"""Research-only retrospective QA for ACTUAL FY2025 PDF pages.

Produces per-page native-vs-Docling transcripts and optional UNVERIFIED Groq Vision
transcripts, with original PDF fingerprints and rendered pages. Never updates the
research database, reviews, or labels. Never uses the opened 2025-26 market holdout.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile

TARGETS = {"INFY": (184,), "ICICIBANK": (29,), "RIL": (73,), "HDFCBANK": (295,)}
TOKEN = re.compile(r"(?<![\w])(?:\(?[-₹$]?\d[\d,]*(?:\.\d+)?%?\)?)(?![\w])")


def numeric_tokens(text: str) -> Counter[str]:
    # Diagnostic only: raw tokens are not reliable monetary row/column relationships.
    return Counter(m.group(0).strip() for m in TOKEN.finditer(text))


def compare_numbers(native: str, docling: str, vision: str | None = None) -> dict:
    a, b = numeric_tokens(native), numeric_tokens(docling)
    dropped, added = a-b, b-a
    result = {
        "native_numeric_tokens": sum(a.values()),
        "docling_numeric_tokens": sum(b.values()),
        "native_not_found_in_docling": sum(dropped.values()),
        "docling_not_found_in_native": sum(added.values()),
        "native_missing_token_samples": list(dropped.elements())[:50],
        "docling_extra_token_samples": list(added.elements())[:50],
        "warning": "Token differences do NOT establish which parser is correct. Verify rows, units, headings against rendered PDF."
    }
    if vision is not None:
        c = numeric_tokens(vision)
        result["vision_numeric_tokens"] = sum(c.values())
        result["vision_not_found_in_native_samples"] = list((c-a).elements())[:50]
        result["native_not_found_in_vision_samples"] = list((a-c).elements())[:50]
    return result


def quality_review(pdf: Path, page_no: int, output: Path, *, groq: bool = False, model: str = "qwen/qwen3.8-27b", parser=None, vision=None) -> dict:
    import pymupdf
    from risklens.research.ingest import _docling_pages, groq_vision_transcript

    if parser is None:
        parser = _docling_pages
    if vision is None:
        vision = groq_vision_transcript
    with pymupdf.open(str(pdf)) as doc:
        if doc.needs_pass or not 1 <= page_no <= len(doc):
            raise ValueError("Invalid, encrypted, or out-of-range PDF page")
        p = doc[page_no - 1]
        native = p.get_text(sort=True)
        output.mkdir(parents=True, exist_ok=True)
        p.get_pixmap(matrix=pymupdf.Matrix(1.6, 1.6), alpha=False).save(str(output / "original_page.png"))
        with tempfile.TemporaryDirectory() as temp:
            single = pymupdf.open()
            single.insert_pdf(doc, from_page=page_no - 1, to_page=page_no - 1)
            single_path = Path(temp) / "one_original_page_only.pdf"
            single.save(str(single_path))
            single.close()
            parsed = parser(single_path, 1)
            if len(parsed) != 1:
                raise ValueError("Docling returned wrong page count for single-page audit")
            docling = parsed[0]
    (output / "pymupdf_original.txt").write_text(native, encoding="utf-8")
    (output / "docling_comparison.txt").write_text(docling, encoding="utf-8")
    vision_text = None
    vision_error = None
    if groq:
        # Network/data disclosure is explicit and page-scoped. An API failure
        # must not prevent us saving the independently extracted page evidence.
        try:
            vision_text = vision(pdf, page_no, model=model)
        except Exception as exc:
            vision_error = type(exc).__name__  # never log request credentials
        if vision_text is not None:
            (output / "groq_vision_UNVERIFIED.txt").write_text(vision_text, encoding="utf-8")
    evidence = {
        "source_pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "source_file": str(pdf),
        "original_physical_pdf_page": page_no,
        "docling_excerpt_pdf_page": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "native_chars": len(native), "docling_chars": len(docling),
        "groq_attempted": groq, "groq_model": model if groq else None,
        "groq_chars": len(vision_text) if vision_text is not None else None,
        "groq_error_type": vision_error,
        "numerical_comparison": compare_numbers(native, docling, vision_text),
        "verified_against_original_render": False,
        "human_review_status": "PENDING",
        "do_not_index_groq_draft": True,
        "context": "Physical PDF page number is not necessarily the printed annual-report footer page number.",
    }
    (output / "audit_UNVERIFIED.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/processed/research/real_acceptance/verified_source_manifest.json")
    parser.add_argument("--out", default="data/processed/research/real_acceptance/flagged_page_quality")
    parser.add_argument("--groq", action="store_true", help="Explicitly submit ONLY four flagged page images to the Groq vision model")
    parser.add_argument("--model", default="qwen/qwen3.8-27b")
    args = parser.parse_args()
    data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    by_ticker = {s["ticker"]: s for s in data["sources"]}
    missing = set(TARGETS) - by_ticker.keys()
    if missing:
        raise SystemExit(f"Source manifest missing required tickers {sorted(missing)}")
    output = Path(args.out)
    report = []
    for ticker, pages in TARGETS.items():
        src = by_ticker[ticker]
        pdf = Path(src["local_path"])
        if not pdf.is_file():
            raise SystemExit(f"Missing genuine report on this Mac: {pdf}")
        for page in pages:
            dest = output / f"{ticker}_physical_pdf_page_{page}"
            print(f"AUDIT {ticker} ORIGINAL PDF PAGE {page} (Groq opt-in={args.groq})", flush=True)
            evidence = quality_review(pdf, page, dest, groq=args.groq, model=args.model)
            report.append({"ticker": ticker, "page": page, "status": evidence["human_review_status"],
                           "differences": evidence["numerical_comparison"]["native_not_found_in_docling"],
                           "file": str(dest / "audit_UNVERIFIED.json")})
    output.mkdir(parents=True, exist_ok=True)
    (output / "SUMMARY_UNVERIFIED.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audited_original_pages": len(report), "groq_used": args.groq,
                      "source_qa_complete": False, "manual_table_review_required": len(report),
                      "results_dir": str(output)}, indent=2))

if __name__ == "__main__":
    main()
