"""Create traceable financial-fact JSON from an original-PDF-reviewed evidence inventory.

This script is intentionally deterministic and offline. The reviewed inventory is
source/period/metric keyed, NOT question/answer keyed. It reads real PDF pages,
checks exact original file hashes and observed numeric strings, and copies
literal source text spans into the JSON bank. No API or model claims certification.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from risklens.research.financial_facts import Fact, validate_original_pdfs
from risklens.research.contracts import stable_id, ResearchError


def build(inventory: str | Path, pdf_dir: str | Path, output: str | Path) -> dict:
    import pymupdf

    spec = json.loads(Path(inventory).read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1 or not isinstance(spec.get("observations"), list):
        raise ResearchError("Unexpected source-reviewed inventory")
    pdf_dir = Path(pdf_dir)
    files: dict[str, tuple] = {}
    facts = []
    for item in spec["observations"]:
        name = item["source_file"]
        if Path(name).name != name:
            raise ResearchError("Inventories must use simple local filenames")
        if name not in files:
            p = pdf_dir / name
            if not p.is_file():
                raise ResearchError(f"Missing official report: {p}")
            hash_ = sha256(p.read_bytes()).hexdigest()
            if hash_ != item["file_sha256"]:
                raise ResearchError(f"Original report hash mismatch: {name}")
            files[name] = (pymupdf.open(p), hash_)
        pdf, hash_ = files[name]
        pn = item["pdf_page"]
        if not isinstance(pn, int) or not 1 <= pn <= len(pdf):
            raise ResearchError("Invalid physical page")
        text = pdf[pn - 1].get_text(sort=True)
        needle = item["source_literal"]
        pos = text.find(needle)
        if pos < 0:
            raise ResearchError(f"Source numerical literal absent on {name} p{pn}: {needle}")
        # The extracted passage is owned by the *original PDF*, not the LLM.
        # Source evidence is derived from the native page, not a guess.
        left, right = item.get("window_before", 150), item.get("window_after", 150)
        excerpt = text[max(0, pos - left):min(len(text), pos + len(needle) + right)]
        identity = {k: v for k, v in item.items() if k not in {"window_before", "window_after"}}
        fact_id = stable_id(hash_, pn, item["metric"], item["segment"], item["reporting_scope"], item["period"])
        row = {**identity, "fact_id": fact_id, "source_evidence": excerpt,
               "review_status": "source_checked_ai_reviewed",
               "display_label": item.get("display_label", item["metric"].replace("_", " ")),
               "definition": item.get("definition", ""),
               "definition_pdf_page": item.get("definition_pdf_page", 0),
               "definition_source_quote": item.get("definition_source_quote", "")}
        facts.append(Fact.parse(row))
    for pdf, _ in files.values():
        pdf.close()
    verification = validate_original_pdfs(facts, pdf_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 1,
        "disclaimer": "Source spans checked against exact original PDFs; AI attribution review is not third-party or independent human certification.",
        "facts": [f.as_dict() for f in facts],
        "pdf_verification": verification,
    }
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"facts": len(facts), "files": list(verification), "path": str(output),
            "original_pdf_spans_verified": True}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", default="docs/phase7_financial_source_inventory.json")
    ap.add_argument("--pdf-dir", default="data/raw/research_reports")
    ap.add_argument("--output", default="data/processed/research/real_acceptance/phase7_completion/financial_facts_v3.json")
    args = ap.parse_args()
    print(json.dumps(build(args.inventory, args.pdf_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
