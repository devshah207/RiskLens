"""Full-corpus, page/table-aware financial JSON extraction.

No question/benchmark-dependent extraction rules. Every row is attached to its
actual table and year-column header, original PDF identity and physical page.
Automatic extraction is *not* human certification; uncertain rows stay in a
review queue and are never promoted merely because an LLM says they are right.

The existing 37 curated source-reviewed V3 facts remain separate and untouched.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
import json
import re
from typing import Iterable

from .contracts import ResearchError, stable_id
from .financial_candidates import draft_page_json

YEAR = re.compile(r"(?i)(?:FY\s*20\d\d\s*[-–/]\s*\d\d|(?:20\d\d)[-–/](?:20)?\d\d|(?:20\d\d)|(?:march|mar|31st\s+march)[^\n]{0,24}20\d\d)")
NUMBER = re.compile(r"^\s*\(?[-+]?\s*(?:[₹$]|Rs\.?|INR\s*|US\$\s*)?\s*(\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(\d+))?\s*\)?\s*(%)?\s*$", re.I)
CURRENCY = re.compile(r"(?i)\b(?:crores?|billion|million|lakhs?|in\s*'000s|₹|rupees|INR|Rs\.)\b")
# Generic accounting metrics. Aliases describe financial concepts, not benchmark questions.
METRICS = (
    ("earnings_per_share", r"\b(?:basic earnings per share|diluted earnings per share|earnings per share|eps)\b"),
    ("cash_investments", r"\b(?:cash and investments|consolidated cash and investments)\b"),
    ("goodwill_intangibles", r"\b(?:goodwill and other intangible assets|goodwill and intangible assets)\b"),
    ("provision_coverage_ratio", r"\bprovision coverage ratio\b"),
    ("cost_income_ratio", r"\b(?:cost.to.income ratio|cost income ratio)\b"),
    ("total_income", r"\btotal income\b"),
    ("cash_rbi", r"\bcash and balances with (?:reserve bank|rbi)\b"),
    ("gross_npa_ratio", r"\bgross\s+(?:npa|non.performing).*\b(?:ratio|advances)\b"),
    ("net_npa_ratio", r"\bnet\s+(?:npa|non.performing).*\b(?:ratio|advances)\b"),
    ("net_interest_margin", r"\b(?:net interest margin|nim)\b"),
    ("capital_adequacy_ratio", r"\b(?:capital adequacy|total capital ratio|crar)\b"),
    ("return_on_equity", r"\b(?:return on equity|roe)\b"),
    ("return_on_assets", r"\b(?:return on assets|roa)\b"),
    ("profit_after_tax", r"\b(?:profit after tax|net profit|profit for the year|profit attributable to)\b"),
    ("ebitda", r"\bebitda\b"),
    ("net_interest_income", r"\bnet interest income\b"),
    ("free_cash_flow", r"\bfree cash flow\b"),
    ("operating_margin", r"\boperating margin\b"),
    ("net_current_assets", r"\bnet current assets\b"),
    ("cost_of_sales", r"\bcost of sales\b"),
    ("revenue", r"\b(?:revenue from operations|total revenue|revenue|value of sales(?: and| &)? services|net sales|turnover)\b"),
    ("deposits", r"\b(?:total deposits|deposits)\b"),
    ("advances", r"\b(?:net advances|loans and advances|total advances|advances)\b"),
    ("balance_sheet_size", r"\b(?:balance.sheet size|total assets)\b"),
    ("operating_expenses", r"\b(?:operating expenses|operating costs)\b"),
    ("interest_expense", r"\binterest expenses?\b"),
    ("equity", r"\b(?:total equity|shareholders.? equity)\b"),
    ("cash_flow_operations", r"\b(?:net cash (?:generated|provided) (?:by|from) operating|cash flow from operating)\b"),
)


def normalized_year(text: str) -> str | None:
    s = re.sub(r"\s+", " ", text).lower()
    found = re.findall(r"20[1-3]\d", s)
    if found:
        year = int(found[-1])
        if "20" in s and re.search(r"20\d\d[-–/]20\d\d", s):
            return f"FY{year-1}-{str(year)[-2:]}"
        if re.search(r"20\d\d[-–/]\d\d", s):
            left = int(re.search(r"20\d\d", s).group())
            return f"FY{left}-{str(left+1)[-2:]}"
        return f"FY{year-1}-{str(year)[-2:]}" if ("march" in s or "mar" in s or "fy" in s) else f"FY{year-1}-{str(year)[-2:]}"
    return None


def canonical_metric(label: str) -> str | None:
    label = re.sub(r"\s+", " ", label).strip().lower()
    for code, pattern in METRICS:
        if re.search(pattern, label, re.I):
            return code
    return None


def numeric_cell(s: str | None) -> tuple[str, str] | None:
    if not s or "\n" in s or len(s) > 38:
        return None
    m = NUMBER.fullmatch(s)
    if not m:
        return None
    value = m.group(1).replace(",", "") + ("." + m.group(2) if m.group(2) else "")
    try:
        if not Decimal(value).is_finite():
            return None
    except InvalidOperation:
        return None
    return value, ("percent" if m.group(3) else "")


def report_scope(top: str, previous: str = "") -> str:
    # This section detector uses report text itself, not known page numbers.
    first = "\n".join(top.splitlines()[:24])
    standalone = bool(re.search(r"(?i)\bstandalone (?:financial|balance|profit|statement|results)", first))
    consolidated = bool(re.search(r"(?i)\bconsolidated (?:financial|balance|profit|statement|results)", first))
    if consolidated and not standalone:
        return "consolidated"
    if standalone and not consolidated:
        return "standalone"
    # Keep propagated scope only when the header contains no contradictory context.
    return previous if previous in {"consolidated", "standalone"} else "unspecified"


def table_headers(rows: list[list[str | None]]) -> dict[int, tuple[str, str, str]]:
    """Column year and reported unit from top 4 table rows, not neighboring cells."""
    out = {}
    for i in range(min(4, len(rows))):
        row = rows[i]
        for j, cell in enumerate(row[1:], 1):
            if not cell or len(cell) > 90:
                continue
            y = normalized_year(cell) if YEAR.search(cell) else None
            if not y:
                continue
            out[j] = (y, cell.strip(), "")
    return out


def page_unit(text: str) -> tuple[str, str, bool]:
    # Rate ratios are always inferred per-cell. Currency requires explicit scale.
    head = "\n".join(text.splitlines()[:35]).lower()
    if re.search(r"(?i)(?:₹|rs\.?|inr|`|\b[in]\b)\s*(?:in\s*)?(?:crore|cr\b)", head) or re.search(r"\bcrores?\b", head):
        return "INR", "crore", True
    if re.search(r"\bin\s+billion\b|\b₹\s*billion\b", head):
        return "INR", "billion", True
    if re.search(r"\b(?:in\s*)?[‘’'`]?(?:000s|thousands)\b", head):
        return "INR", "thousand_rupees", True
    return "none", "unspecified", False


def _table_years(rows: list[list[str | None]], page_text: str) -> dict[int, tuple[str, str, str]]:
    mapped = table_headers(rows)
    if mapped:
        return mapped
    # Never fabricate a year column from a report publication year.
    return {}


def _same_page_literal(text: str, literal: str) -> bool:
    from .financial_facts import _page_has_value
    return _page_has_value(text, literal.strip().replace("%", ""))


def _candidate(source: dict, page: int, table_no: int, row_no: int, col_no: int,
               label: str, metric: str, val: str, literal: str, unit: str,
               currency: str, scope: str, period: str, year_heading: str,
               status: str, reason: str, table_box=None) -> dict:
    identity = stable_id(source["file_sha256"], page, table_no, row_no, col_no, label, period, val)
    return {
        "fact_id": identity, "ticker": source["ticker"], "company": source["company"],
        "source_file": source["source_file"], "file_sha256": source["file_sha256"],
        "source_url": source["source_url"], "published_on": source["published_on"],
        "report_fiscal_year": source["fiscal_year"], "physical_pdf_page": page,
        "table_index": table_no, "row_index": row_no, "column_index": col_no,
        "table_bbox": list(table_box) if table_box else None,
        "row_label": label.strip(), "column_label": year_heading.strip(),
        "metric": metric, "segment": "", "reporting_scope": scope,
        "observed_period": period, "value": val, "source_literal": literal.strip(),
        "currency": currency, "unit": unit,
        "status": status, "reason": reason,
        "provenance": "native_pdf_table_row_and_year_column",
    }


def extract_pdf_tables(source: dict, *, max_pages: int | None = None,
                       exclude_pages: Iterable[int] = ()) -> tuple[list[dict], dict]:
    """Automatically scan *every* page (unless caller intentionally limits pages).

    Flagged numeric-OCR pages stay out of auto-approved facts. Table candidates
    with unclear scope/unit/year are retained in JSON for review, not discarded.
    """
    import pymupdf
    path = Path(source["pdf_path"])
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise ResearchError(f"Missing valid source PDF: {path}")
    got = sha256(path.read_bytes()).hexdigest()
    if got != source["file_sha256"]:
        raise ResearchError(f"PDF SHA256 mismatch for {path.name}")
    excluded = set(exclude_pages)
    output = []
    scanned = 0
    considered = 0
    with pymupdf.open(path) as pdf:
        for idx in range(min(len(pdf), max_pages or len(pdf))):
            scanned += 1
            pg = pdf[idx]
            text = pg.get_text(sort=True)
            # Skip pages without any reporting-year labels or financial terminology.
            if not YEAR.search(text) or not any(re.search(p, text, re.I) for _, p in METRICS):
                continue
            considered += 1
            try:
                tables = pg.find_tables()
            except Exception:
                continue
            scope = report_scope(text)
            money_currency, money_unit, unit_known = page_unit(text)
            for ti, table in enumerate(tables.tables):
                rows = table.extract()
                if len(rows) < 2 or not rows or max(map(len, rows)) > 16:
                    continue
                years = _table_years(rows, text)
                for ri, row in enumerate(rows):
                    if not row or not isinstance(row[0], str):
                        continue
                    # Do not accept combined multiline row labels as a single fact.
                    label = row[0].strip()
                    if not label or "\n" in label or len(label) > 95:
                        continue
                    metric = canonical_metric(label)
                    if not metric:
                        continue
                    for ci in sorted(years):
                        if ci >= len(row):
                            continue
                        literal = (row[ci] or "").strip()
                        parsed = numeric_cell(literal)
                        if not parsed:
                            continue
                        val, percentage = parsed
                        period, yhead, _ = years[ci]
                        currency = "none" if percentage else money_currency
                        unit = "percent" if percentage else money_unit
                        # Ratios that are clearly percentages but omit the symbol need manual QA.
                        ratio_metric = metric.endswith("_ratio") or metric.endswith("_margin") or metric.startswith("return_on_")
                        high = (idx + 1 not in excluded and _same_page_literal(text, literal)
                                and (percentage or unit_known) and scope != "unspecified"
                                and (not ratio_metric or percentage))
                        status = "auto_anchored" if high else "review_required"
                        reasons = []
                        if idx + 1 in excluded:
                            reasons.append("source_page_quarantined")
                        if scope == "unspecified":
                            reasons.append("financial_scope_unresolved")
                        if unit == "unspecified":
                            reasons.append("unit_unresolved")
                        if ratio_metric and not percentage:
                            reasons.append("ratio_unit_unresolved")
                        if not _same_page_literal(text, literal):
                            reasons.append("native_literal_unresolved")
                        output.append(_candidate(source, idx+1, ti, ri, ci, label, metric, val,
                                                 literal, unit, currency, scope, period, yhead,
                                                 status, ",".join(reasons) or "table_row_and_column_native_anchored",
                                                 table.bbox))
    return output, {"physical_pages_scanned": scanned, "pages_considered": considered,
                    "table_candidates": len(output), "auto_anchored": sum(x["status"] == "auto_anchored" for x in output),
                    "review_required": sum(x["status"] == "review_required" for x in output)}


def build_automatic_bank(sources: list[dict], *, output: str | Path,
                         max_pages: int | None = None,
                         quarantine_by_sha: dict[str, list[int]] | None = None,
                         vision_client=None, vision_model="qwen/qwen3.8-27b",
                         vision_page_budget: int = 0) -> dict:
    """One-pass automatic corpus scan with optional *early* Groq table drafting.

    The 2x vision call runs DURING ingestion for selected ambiguous table pages,
    not after answer failure. Its drafts are never silently promoted.
    """
    quarantine_by_sha = quarantine_by_sha or {}
    all_rows = []
    summary = {}
    vision = []
    remaining = int(vision_page_budget)
    fair_share = max(1, (remaining + max(1, len(sources))-1) // max(1, len(sources)))
    for source in sources:
        rows, stats = extract_pdf_tables(source, max_pages=max_pages,
                                         exclude_pages=quarantine_by_sha.get(source["file_sha256"], []))
        all_rows.extend(rows)
        summary[source["source_file"]] = stats
        if vision_client and remaining > 0:
            # Prioritize complex or unresolved table pages, then capture results
            # while building the knowledge base, not at query time.
            quarantined = sorted(quarantine_by_sha.get(source["file_sha256"], []))
            candidates = quarantined + sorted({r["physical_pdf_page"] for r in rows if r["status"] == "review_required"}
                                              - set(quarantined))
            for page_no in candidates[:min(remaining, fair_share)]:
                try:
                    draft = draft_page_json(source["pdf_path"], page_no, client=vision_client, model=vision_model)
                    vision.append({"source_file": source["source_file"], "pdf_page": page_no,
                                   "status": "review_required", "draft": draft})
                except Exception as exc:
                    vision.append({"source_file": source["source_file"], "pdf_page": page_no,
                                   "status": "provider_unavailable", "error_type": type(exc).__name__})
                remaining -= 1
                if remaining <= 0:
                    break
    ids = [r["fact_id"] for r in all_rows]
    if len(set(ids)) != len(ids):
        raise ResearchError("Duplicate automatically extracted financial fact IDs")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {"schema_version": 4, "status": "AUTO_EXTRACTED_REQUIRES_SCOPE_REVIEW",
            "notice": "Auto-anchored means native table row+year+number confirmed; NOT human certified. Review candidates stay untrusted.",
            "sources": [{k: s[k] for k in ("ticker", "company", "source_file", "file_sha256", "source_url", "published_on", "fiscal_year")} for s in sources],
            "facts": all_rows, "vision_drafts": vision, "source_stats": summary}
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(output)
    return {"output": str(output), "facts": len(all_rows), "auto_anchored": sum(r["status"] == "auto_anchored" for r in all_rows),
            "review_required": sum(r["status"] == "review_required" for r in all_rows),
            "vision_drafts": len(vision), "sources": summary}


def load_auto_bank(path: str | Path, *, expected_hashes: set[str] | None = None) -> list[dict]:
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    if blob.get("schema_version") != 4 or not isinstance(blob.get("facts"), list):
        raise ResearchError("Invalid automatic financial fact schema")
    facts = blob["facts"]
    for f in facts:
        for field in ("source_file", "file_sha256", "physical_pdf_page", "metric", "reporting_scope",
                      "observed_period", "value", "status", "row_label", "column_label"):
            if field not in f:
                raise ResearchError("Incomplete source-backed auto fact: " + field)
        if expected_hashes is not None and f["file_sha256"] not in expected_hashes:
            raise ResearchError("Automatic fact source hash not in expected corpus")
    return facts


def confirm_original_table_rows(facts: list[dict], *, pdf_dir: str | Path) -> dict:
    """Recheck automatic rows against exact original PDF table cells at query time.

    Presence of a number elsewhere on the page never passes this validation.
    In particular this prevents a doctored JSON row from borrowing another
    financial metric's value on the same page.
    """
    import pymupdf
    grouped: dict[str, list[dict]] = {}
    for f in facts:
        grouped.setdefault(f["source_file"], []).append(f)
    verified = 0
    for filename, rows in grouped.items():
        if Path(filename).name != filename:
            raise ResearchError("Automatic source filename contains directory")
        path = Path(pdf_dir) / filename
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != rows[0]["file_sha256"]:
            raise ResearchError("Auto fact original report file hash mismatch")
        with pymupdf.open(path) as pdf:
            cached = {}
            for f in rows:
                if f["status"] != "auto_anchored":
                    raise ResearchError("Review-required automatic fact cannot answer a question")
                page = f["physical_pdf_page"]
                if not 1 <= page <= len(pdf):
                    raise ResearchError("Auto fact physical PDF page out of range")
                if page not in cached:
                    cached[page] = pdf[page-1].find_tables().tables
                tables = cached[page]
                ti, ri, ci = f["table_index"], f["row_index"], f["column_index"]
                if not 0 <= ti < len(tables):
                    raise ResearchError("Auto fact source table no longer found")
                table = tables[ti]
                cells = table.extract()
                if not 0 <= ri < len(cells) or not 0 <= ci < len(cells[ri]):
                    raise ResearchError("Auto fact source table cell no longer found")
                row = cells[ri]
                if row[0].strip() != f["row_label"].strip() or (row[ci] or "").strip() != f["source_literal"]:
                    raise ResearchError("Auto fact table row/value attribution mismatch")
                original_page_text = pdf[page-1].get_text(sort=True)
                original_scope = report_scope(original_page_text)
                original_currency, original_unit, original_unit_known = page_unit(original_page_text)
                if original_scope != f["reporting_scope"]:
                    raise ResearchError("Auto fact reporting scope does not match source page")
                if f["unit"] == "percent":
                    if not (row[ci] or "").strip().endswith("%"):
                        raise ResearchError("Auto fact ratio unit does not match source cell")
                elif original_unit_known and (f["currency"], f["unit"]) != (original_currency, original_unit):
                    raise ResearchError("Auto fact currency/scale does not match source page")
                years = _table_years(cells, original_page_text)
                if ci not in years or years[ci][0] != f["observed_period"] or years[ci][1] != f["column_label"]:
                    raise ResearchError("Auto fact year column does not match original PDF")
                if str(Decimal(numeric_cell(row[ci])[0])) != str(Decimal(f["value"])):
                    raise ResearchError("Auto fact numeric value does not match original PDF")
                verified += 1
    return {"rows_verified_against_original_table": verified, "status": "native_row_col_verified_not_human_certified"}


def hydrate_prebuilt_bank(source_bank: str | Path, output: str | Path, *, pdf_dir: str | Path,
                         vision_client=None, vision_page_budget: int = 0,
                         quarantine_by_sha: dict[str, list[int]] | None = None) -> dict:
    """Verify and reuse a full-corpus scan of byte-identical original reports.

    This avoids asking the owner to rescan unchanged PDFs. Before use, verify
    EVERY existing auto-anchored native table row (and file SHA) against the
    local source PDF. Optional Groq Vision drafts run at ingest time afterward.
    """
    data = json.loads(Path(source_bank).read_text(encoding="utf-8"))
    if data.get("schema_version") != 4 or len(data.get("sources", [])) < 5:
        raise ResearchError("Prebuilt source JSON is incomplete or wrong schema")
    sources = data["sources"]
    # Verify all five original PDF hashes even if one has zero extracted tables.
    for s in sources:
        name = s["source_file"]
        if Path(name).name != name:
            raise ResearchError("Invalid prebuilt PDF name")
        path = Path(pdf_dir) / name
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != s["file_sha256"]:
            raise ResearchError(f"Prebuilt JSON has different original PDF: {name}")
    facts = load_auto_bank(source_bank, expected_hashes={s["file_sha256"] for s in sources})
    validated = confirm_original_table_rows([f for f in facts if f["status"] == "auto_anchored"],
                                            pdf_dir=pdf_dir)
    data["prebuilt_source_validation"] = validated
    quarantine_by_sha = quarantine_by_sha or {}
    if vision_page_budget:
        if vision_client is None:
            raise ResearchError("Groq client required when vision budget is nonzero")
        budget = int(vision_page_budget)
        fair = max(1, (budget + len(sources)-1)//len(sources))
        for source in sources:
            if budget <= 0:
                break
            pages = sorted(quarantine_by_sha.get(source["file_sha256"], []))
            pages += sorted({f["physical_pdf_page"] for f in facts
                             if f["file_sha256"] == source["file_sha256"] and f["status"] == "review_required"}
                            - set(pages))
            for pn in pages[:min(budget, fair)]:
                try:
                    result = draft_page_json(Path(pdf_dir) / source["source_file"], pn, client=vision_client)
                    data.setdefault("vision_drafts", []).append({"source_file": source["source_file"],
                                                                  "pdf_page": pn, "status": "review_required",
                                                                  "draft": result})
                except Exception as exc:
                    data.setdefault("vision_drafts", []).append({"source_file": source["source_file"],
                                                                  "pdf_page": pn, "status": "provider_unavailable",
                                                                  "error_type": type(exc).__name__})
                budget -= 1
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    return {"output": str(output), "facts": len(facts),
            "auto_anchored_original_table_reverified": validated["rows_verified_against_original_table"],
            "review_required": sum(f["status"] == "review_required" for f in facts),
            "original_sources_sha_checked": len(sources),
            "vision_drafts": len(data.get("vision_drafts", []))}
