"""Conservative, general-purpose PDF *spatial* financial-KPI panel extractor.

This extractor is independent of questions and gold answers. It indexes paired
metric cards adjacent to a visibly larger segment heading, tied to an explicit
financial period disclosure, source bytes and physical PDF geometry.

It never invents missing values, and re-extracts the original page at answer
 time. Panel facts are separate from source-reviewed V3 facts and V4 native
 table facts. No language-model calls occur in this module.
"""
from __future__ import annotations

from hashlib import sha256
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
import json
import re
import unicodedata


METRIC_LINE = re.compile(r"^\s*(REVENUE|EBITDA)\s*(?:[↑↓+\- ]|\d|\.|%)*\s*$", re.I)
AMOUNT = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(\d+))?\s*(Crore|Crores)\b", re.I)
YEAR_ENDED = re.compile(
    r"(?:all\s+)?revenue\s+and\s+ebitda\s+figures?\s+are\s+for\s+the\s+year\s+ended\s+"
    r"(?:on\s+)?(?:march\s+31|31\s+march|31st\s+march)[, ]+\s*(20\d{2})",
    re.I,
)
FISCAL_NOTE = re.compile(r"(?:all\s+)?revenue\s+and\s+ebitda\s+figures?.{0,85}?\bFY\s*(20\d{2})\b", re.I | re.S)
OVERVIEW_NOTE = re.compile(
    r"(?:the\s+)?results?\s+for\s+the\s+year\s+ended\s+"
    r"(?:march\s+31|31\s+march)[, ]+\s*(20\d{2})", re.I)
CARD_METRICS = (
    ("gross_npa_ratio", r"^gross npa ratio"),
    ("cash_investments", r"^cash and investments"),
    ("earnings_per_share", r"^earnings per share"),
    ("return_on_assets", r"^return on assets"),
    ("return_on_equity", r"^return on equity"),
    ("cost_income_ratio", r"^cost to income ratio"),
    ("profit_after_tax", r"^profit after tax"),
    ("balance_sheet_size", r"^balance sheet size"),
    ("deposits", r"^deposits"),
    ("advances", r"^advances"),
    ("dividend_per_share", r"^dividend per share"),
)
CARD_NUMBER = re.compile(r"^\s*(\d[\d,]*(?:\.\d+)?)(?:\s*\^)?\s*$")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _blocks(page) -> list[dict[str, Any]]:
    blocks = []
    for raw in page.get_text("dict").get("blocks", []):
        if raw.get("type") != 0:
            continue
        lines = []
        for line in raw.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(s.get("text", "") for s in spans).strip()
            if line_text:
                lines.append(line_text)
        text = "\n".join(lines)
        if not text:
            continue
        spans = [span for line in raw.get("lines", []) for span in line.get("spans", [])]
        blocks.append({"text": text, "bbox": tuple(float(x) for x in raw["bbox"]),
                       "font_size": max((float(s.get("size", 0)) for s in spans), default=0.0),
                       "rupee_glyph": any("rupee" in str(s.get("font", "")).casefold() for s in spans) or
                       "₹" in text or bool(re.search(r"\bINR\b|\bRs\.?\s*\d", text))})
    return blocks


def _period(page_text: str) -> tuple[str | None, str | None]:
    normalized = re.sub(r"\s+", " ", page_text)
    m = YEAR_ENDED.search(normalized)
    if m is None:
        m = FISCAL_NOTE.search(normalized)
    if m is None:
        m = OVERVIEW_NOTE.search(normalized)
    if m is None:
        return None, None
    year = int(m.group(1))
    if year < 2000 or year > 2100:
        return None, None
    return f"FY{year-1}-{year % 100:02d}", m.group(0)


def _metrics(block: dict[str, Any]) -> dict[str, str] | None:
    lines = block["text"].splitlines()
    if len(lines) < 2 or len(lines) > 5 or not METRIC_LINE.fullmatch(lines[0]):
        return None
    metric = lines[0].split()[0].casefold()
    # Amount is tied to the specific numeric block, not another nearby text block.
    values = [AMOUNT.search(line) for line in lines[1:]]
    values = [m for m in values if m]
    if len(values) != 1 or not block["rupee_glyph"]:
        return None
    amount = values[0]
    literal = amount.group(0)
    value = amount.group(1).replace(",", "") + ("." + amount.group(2) if amount.group(2) else "")
    try:
        if not Decimal(value).is_finite():
            return None
    except InvalidOperation:
        return None
    return {"metric": metric, "value": value, "literal": literal}


def _heading(block: dict[str, Any], revenue: dict[str, Any]) -> bool:
    title = _norm(block["text"])
    if not title or len(title) > 55 or len(block["text"].splitlines()) > 3:
        return False
    if re.search(r"\d", title) or title.startswith(("revenue", "ebitda", "total", "fy")):
        return False
    bx0, by0, bx1, by1 = block["bbox"]
    rx0, ry0, _, _ = revenue["bbox"]
    if not (by1 <= ry0 + 5 and 0 <= ry0-by0 <= 105):
        return False
    if abs(rx0-bx0) > 25 or block["font_size"] < max(15.0, revenue["font_size"] * 1.6):
        return False
    return True



def _overview_card(block: dict[str, Any]) -> dict[str, str] | None:
    """An explicitly labeled KPI and its single primary number in one PDF block."""
    lines = block["text"].splitlines()
    if len(lines) < 2 or len(lines) > 5:
        return None
    match = CARD_NUMBER.fullmatch(lines[-1])
    if match is None:
        return None
    label = _norm(" ".join(lines[:-1]))
    # Explicit units and KPI labels, never implied financial measurement units.
    code = next((key for key, pattern in CARD_METRICS if re.search(pattern, label)), None)
    if not code:
        return None
    if re.search(r"\b(?:percent|percentage)\b", label) or "%" in block["text"]:
        unit = "percent"
    elif re.search(r"\bcr\b|\bcrore\b", label):
        unit = "crore"
    elif block["rupee_glyph"] and re.search(r"\b(?:share|eps)\b", label):
        unit = "rupees"
    else:
        return None
    if code in {"return_on_equity", "return_on_assets", "cost_income_ratio", "gross_npa_ratio"} and unit != "percent":
        return None
    if code in {"balance_sheet_size", "profit_after_tax", "deposits", "advances"} and unit != "crore":
        return None
    value = match.group(1).replace(",", "")
    return {"metric": code, "value": value, "literal": lines[-1].strip(), "unit": unit,
            "label": " ".join(lines[:-1]).strip()}


def _extract_overview_cards(blocks: list[dict[str, Any]], source: dict[str, str],
                            page_number: int, period: str) -> list[dict[str, Any]]:
    """Only on pages explicitly noting the financial results reporting year."""
    found = []
    for b in blocks:
        parsed = _overview_card(b)
        if not parsed:
            continue
        # Financial display typography: the primary value must visibly dominate
        # or at least match the metric label, and be within the same source block.
        if b["font_size"] < 13:
            continue
        identity = "|".join([source["file_sha256"], str(page_number), "card",
                             parsed["metric"], period, parsed["value"]])
        found.append({
            "fact_id": sha256(identity.encode()).hexdigest()[:24],
            "ticker": source["ticker"], "company": source["company"],
            "source_file": source["source_file"], "file_sha256": source["file_sha256"],
            "source_url": source["source_url"], "published_on": source["published_on"],
            "physical_pdf_page": page_number, "metric": parsed["metric"],
            "segment": "", "segment_source_label": "",
            "reporting_scope": "overview_unspecified", "observed_period": period,
            "value": parsed["value"], "source_literal": parsed["literal"],
            "currency": "INR" if parsed["unit"] in {"crore", "rupees"} else "none",
            "unit": parsed["unit"], "status": "auto_anchored",
            "row_label": parsed["label"],
            "source_heading_bbox": [round(v, 1) for v in b["bbox"]],
            "source_metric_bbox": [round(v, 1) for v in b["bbox"]],
            "source_metric_block_text": b["text"],
            "period_disclosure": "year explicitly disclosed in page overview note",
            "provenance": "single_block_labeled_kpi_with_explicit_year_note",
        })
    return found




def _extract_spatial_overview_cards(page, source: dict[str, str],
                                    page_number: int, period: str) -> list[dict[str, Any]]:
    """Rebuild financial cards from *word positions*, not PDF block grouping.

    PyMuPDF versions may split an otherwise identical KPI card into several
    text blocks.  The large current-year number must lie below an unambiguous
    metric label within the same visual column, with an explicit year note
    elsewhere on the page.  Small prior-year bar labels are not eligible.
    """
    words = page.get_text("words")
    if not words:
        return []

    # A headline value is visibly larger than a small graph/bar annotation.
    # Word bounding boxes are stable even when block boundaries change.
    headline = [w for w in words
                if re.fullmatch(r"\d[\d,]*(?:\.\d+)?\^?", w[4])
                and w[3] - w[1] >= 20]
    if not headline:
        return []

    # Identify currency from the PDF's glyph font, not from the ambiguous
    # extracted letter 'I' used by some Indian-rupee fonts.
    glyph_spans = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                glyph_spans.append(span)

    result = []
    for value_word in headline:
        vx0, vy0, vx1, vy1, literal, *_ = value_word
        # Calculate the actual visual column width from neighboring large
        # values on the SAME row, rather than assuming a specific page size.
        right_neighbors = [w[0] - vx0 for w in headline
                           if 0 < w[0] - vx0 and abs(w[1] - vy0) < 35]
        width = max(85.0, min(170.0, min(right_neighbors) - 10.0
                             if right_neighbors else 165.0))
        near = [w for w in words
                if vy0 - 55 <= w[1] and w[3] <= vy0 + 2
                and vx0 - 8 <= w[0] <= vx0 + width
                and w[3] - w[1] < 19]
        if not near:
            continue
        # Read original visual lines from top to bottom.  Avoid mixing in a
        # neighboring label when values have similar vertical positions.
        rows: list[list] = []
        for w in sorted(near, key=lambda v: (v[1], v[0])):
            matching = next((row for row in rows if abs(row[0][1] - w[1]) <= 2.5), None)
            if matching is None:
                rows.append([w])
            else:
                matching.append(w)
        rows.sort(key=lambda row: min(w[1] for w in row))
        # The last line of the label must sit close to the large value.
        if not rows or vy0 - max(w[3] for w in rows[-1]) > 20:
            continue
        label_words = [w for row in rows for w in sorted(row, key=lambda v: v[0])]
        label = " ".join(" ".join(w[4] for w in sorted(row, key=lambda v: v[0]))
                         for row in rows).strip()
        label_normalized = _norm(label)
        match = [code for code, pattern in CARD_METRICS
                 if re.search(pattern, label_normalized)]
        if len(match) != 1:
            continue
        metric = match[0]
        if len(label) > 100 or len(rows) > 3:
            continue

        if "%" in label or re.search(r"\bpercent(?:age)?\b", label_normalized):
            unit = "percent"
        elif re.search(r"\b(?:cr|crore|crores)\b", label_normalized):
            unit = "crore"
        else:
            # Some PDFs map the rupee character to literal 'I'.  Require
            # the special currency font or an explicit printed ₹/Rs/INR.
            label_box = (min(w[0] for w in label_words),
                         min(w[1] for w in label_words),
                         max(w[2] for w in label_words),
                         max(w[3] for w in label_words))
            glyph_currency = any(
                "rupee" in str(span.get("font", "")).casefold()
                and span["bbox"][0] < label_box[2] + 2
                and span["bbox"][2] > label_box[0] - 2
                and span["bbox"][1] < label_box[3] + 2
                and span["bbox"][3] > label_box[1] - 2
                for span in glyph_spans
            )
            print_currency = bool(re.search(r"₹|\b(?:INR|Rs)\b", label, re.I))
            if metric not in {"earnings_per_share", "dividend_per_share"} or not (
                glyph_currency or print_currency
            ):
                continue
            unit = "rupees"

        if metric in {"return_on_equity", "return_on_assets", "cost_income_ratio",
                      "gross_npa_ratio"} and unit != "percent":
            continue
        if metric in {"balance_sheet_size", "profit_after_tax", "deposits",
                      "advances"} and unit != "crore":
            continue
        value = literal.removesuffix("^").replace(",", "")
        if not re.fullmatch(r"\d+(?:\.\d+)?", value):
            continue
        if len(rows) and max(w[3] for w in rows[-1]) > vy0 + 2:
            continue
        label_bbox = [round(min(w[0] for w in label_words), 1),
                      round(min(w[1] for w in label_words), 1),
                      round(max(w[2] for w in label_words), 1),
                      round(max(w[3] for w in label_words), 1)]
        value_bbox = [round(float(v), 1) for v in value_word[:4]]
        identity = "|".join([source["file_sha256"], str(page_number), "spatial-card",
                             metric, period, value])
        result.append({
            "fact_id": sha256(identity.encode()).hexdigest()[:24],
            "ticker": source["ticker"], "company": source["company"],
            "source_file": source["source_file"], "file_sha256": source["file_sha256"],
            "source_url": source["source_url"], "published_on": source["published_on"],
            "physical_pdf_page": page_number, "metric": metric,
            "segment": "", "segment_source_label": "",
            "reporting_scope": "overview_unspecified", "observed_period": period,
            "value": value, "source_literal": literal, "currency": "INR" if unit in {
                "crore", "rupees"} else "none",
            "unit": unit, "status": "auto_anchored", "row_label": label,
            "source_heading_bbox": label_bbox, "source_metric_bbox": value_bbox,
            "source_metric_block_text": label + "\n" + literal,
            "period_disclosure": "year explicitly disclosed in page overview note",
            "provenance": "spatial_word_labeled_kpi_explicit_year_note",
        })
    # Ambiguous duplicate amounts/labels must never become trusted facts.
    grouped: dict[str, set[str]] = {}
    for row in result:
        grouped.setdefault(row["metric"], set()).add(row["value"])
    return [row for row in result if len(grouped[row["metric"]]) == 1]


_GRID_ROW_METRICS = (
    ("revenue", r"^revenues? from operations$"),
    ("goodwill_intangibles", r"^goodwill and other intangible assets$"),
    ("net_current_assets", r"^net current assets$"),
    ("total_income", r"^total income$"),
    ("profit_after_tax", r"^profit after tax$"),
    ("balance_sheet_size", r"^total assets$"),
    ("profit_after_tax", r"^net profit$"),
)
_GRID_AMOUNT = re.compile(r"^\(?-?\d[\d,]*(?:\.\d+)?\)?$")


def _extract_text_financial_grid(page, blocks: list[dict[str, Any]], source: dict,
                                 page_number: int) -> list[dict[str, Any]]:
    """Recover unlatticed text tables using native word coordinates.

    Requires TWO physically separate scope headings and EXACTLY FOUR adjacent
    year-column headers (two per scope). Values must align with those columns.
    No fiscal year is inferred from the report publication date.
    """
    words = page.get_text("words")
    header = " ".join(b["text"] for b in blocks if b["bbox"][1] < 230)
    if not re.search(r"\bin\s*.{0,12}\bcrore\b", header, re.I):
        return []
    scope_words = [(w[0], w[1], _norm(w[4])) for w in words
                   if _norm(w[4]) in {"standalone", "consolidated"}]
    candidates = []
    for y in sorted({round(w[1]) for w in words if re.fullmatch(r"20[1-3]\d", w[4])}):
        line = [w for w in words if abs(w[1]-y) < 2 and re.fullmatch(r"20[1-3]\d", w[4])]
        if len(line) != 4:
            continue
        line.sort(key=lambda w: w[0])
        if [int(w[4]) for w in line] not in ([2025, 2024, 2025, 2024], [2024, 2023, 2024, 2023]):
            continue
        s = [w for w in scope_words if 0 < y-w[1] < 65 and line[0][0]-25 <= w[0] <= line[1][0] + 20]
        c = [w for w in scope_words if 0 < y-w[1] < 65 and line[2][0]-25 <= w[0] <= line[3][0] + 20]
        if len(s) != 1 or len(c) != 1 or s[0][2] != "standalone" or c[0][2] != "consolidated":
            continue
        candidates.append((y, line))
    if len(candidates) != 1:
        return []
    header_y, columns = candidates[0]
    out = []
    for b in blocks:
        x0, y0, x1, y1 = b["bbox"]
        if y0 <= header_y + 8 or y1-y0 > 19 or x1-x0 < 250:
            continue
        label = _norm(b["text"].splitlines()[0])
        metric = next((code for code, pat in _GRID_ROW_METRICS if re.fullmatch(pat, label)), None)
        if not metric:
            continue
        row_values = [w for w in words if abs(w[1]-y0) < 2 and _GRID_AMOUNT.fullmatch(w[4])
                      and w[0] >= columns[0][0] - 30 and w[0] < x1 + 2]
        if len(row_values) < 4:
            continue
        for ci, col in enumerate(columns):
            # Right alignment is stable when labels and numbers have different widths.
            aligned = sorted(row_values, key=lambda w: abs(w[2] - col[2]))
            if len(aligned) == 0 or abs(aligned[0][2] - col[2]) > 14:
                continue
            cell = aligned[0]
            if len(aligned) > 1 and abs(aligned[1][2]-col[2]) < 14:
                continue  # Ambiguous column attribution.
            value = cell[4].replace(",", "")
            if not value or not re.fullmatch(r"\d+(?:\.\d+)?", value):
                continue
            scope = "standalone" if ci < 2 else "consolidated"
            year = int(col[4])
            period = f"FY{year-1}-{year%100:02d}"
            identity = "|".join([source["file_sha256"], str(page_number), "grid", scope,
                                 label, metric, period, value])
            out.append({
                "fact_id": sha256(identity.encode()).hexdigest()[:24],
                "ticker": source["ticker"], "company": source["company"],
                "source_file": source["source_file"], "file_sha256": source["file_sha256"],
                "source_url": source["source_url"], "published_on": source["published_on"],
                "physical_pdf_page": page_number, "metric": metric,
                "segment": "", "segment_source_label": "",
                "reporting_scope": scope, "observed_period": period,
                "value": value, "source_literal": cell[4],
                "currency": "INR", "unit": "crore", "status": "auto_anchored",
                "source_heading_bbox": [round(v, 1) for v in b["bbox"]],
                "source_metric_bbox": [round(v, 1) for v in cell[:4]],
                "source_metric_block_text": b["text"],
                "period_disclosure": f"{scope} explicit March 31 {year} column",
                "provenance": "native_text_grid_scoped_year_column",
            })
    return out



def _extract_local_year_cards(blocks: list[dict[str, Any]], source: dict,
                              page_number: int) -> list[dict[str, Any]]:
    """A KPI printed directly above a single explicit FYyy label in its card."""
    out = []
    year_blocks = []
    for b in blocks:
        year_match = re.fullmatch(r"\s*FY\s*([2-3]\d)\s*", b["text"], re.I)
        if year_match:
            year_blocks.append((int(year_match.group(1)), b))
    for b in blocks:
        parsed = _overview_card(b)
        if not parsed or b["font_size"] < 13:
            continue
        x0, y0, _, y1 = b["bbox"]
        following = [(gap, y) for y, lb in year_blocks
                     if abs(lb["bbox"][0]-x0) < 20
                     and 0 <= (gap := lb["bbox"][1] - y1) <= 25]
        following.sort()
        if not following or (len(following) > 1 and following[1][0] - following[0][0] < 7):
            continue
        fy = 2000 + following[0][1]
        period = f"FY{fy-1}-{fy%100:02d}"
        identity = "|".join([source["file_sha256"], str(page_number), "local-fy-card",
                             parsed["metric"], period, parsed["value"]])
        out.append({
            "fact_id": sha256(identity.encode()).hexdigest()[:24],
            "ticker": source["ticker"], "company": source["company"],
            "source_file": source["source_file"], "file_sha256": source["file_sha256"],
            "source_url": source["source_url"], "published_on": source["published_on"],
            "physical_pdf_page": page_number, "metric": parsed["metric"],
            "segment": "", "segment_source_label": "",
            "reporting_scope": "overview_unspecified", "observed_period": period,
            "value": parsed["value"], "source_literal": parsed["literal"],
            "currency": "INR" if parsed["unit"] in {"crore", "rupees"} else "none",
            "unit": parsed["unit"], "status": "auto_anchored",
            "row_label": parsed["label"],
            "source_heading_bbox": [round(v, 1) for v in b["bbox"]],
            "source_metric_bbox": [round(v, 1) for v in b["bbox"]],
            "source_metric_block_text": b["text"],
            "period_disclosure": f"FY{following[0][1]} label directly below metric card",
            "provenance": "local_year_labeled_kpi_card",
        })
    return out


_NARRATIVE_REVENUE = re.compile(
    r"\b(total net revenue)\s+for\s+the\s+year\s+ended\s+(?:march\s+31|31\s+march)"
    r"[,]?\s*(20\d{2})[,]?\s+(?:reached|stood at|was|amounted to)"
    r"[^0-9]{0,20}(\d[\d,]*(?:\.\d+)?)\s+crore\b", re.I,
)


_CASH_SCOPE_PAIR = re.compile(
    r"\bcash\s+and\s+investments\s+(?:stand\s+at|were|are)"
    r"[^0-9]{0,15}(\d[\d,]*(?:\.\d+)?)\s+crore\s+on\s+a\s+standalone\s+basis"
    r"\s*(?:,?\s*and)\s*[₹`Rs. ]*(\d[\d,]*(?:\.\d+)?)\s+crore\s+on\s+a\s+consolidated"
    r"\s+basis\s+as\s+on\s+march\s+31[,]?\s*(20\d{2})", re.I,
)


def _extract_explicit_period_narratives(blocks: list[dict[str, Any]], source: dict,
                                        page_number: int) -> list[dict[str, Any]]:
    """Exact amount, line-item name and fiscal period must share one PDF block."""
    result = []
    for b in blocks:
        text = re.sub(r"\s+", " ", b["text"])
        for pair in _CASH_SCOPE_PAIR.finditer(text):
            year = int(pair.group(3))
            period = f"FY{year-1}-{year%100:02d}"
            for scope, raw in (("standalone", pair.group(1)), ("consolidated", pair.group(2))):
                value = raw.replace(",", "")
                identity = "|".join([source["file_sha256"], str(page_number), "narrative-cash",
                                     scope, period, value])
                result.append({
                    "fact_id": sha256(identity.encode()).hexdigest()[:24],
                    "ticker": source["ticker"], "company": source["company"],
                    "source_file": source["source_file"], "file_sha256": source["file_sha256"],
                    "source_url": source["source_url"], "published_on": source["published_on"],
                    "physical_pdf_page": page_number, "metric": "cash_investments",
                    "segment": "", "segment_source_label": "",
                    "reporting_scope": scope, "observed_period": period,
                    "value": value, "source_literal": raw + " crore",
                    "currency": "INR", "unit": "crore", "status": "auto_anchored",
                    "row_label": "Cash and investments",
                    "source_heading_bbox": [round(v, 1) for v in b["bbox"]],
                    "source_metric_bbox": [round(v, 1) for v in b["bbox"]],
                    "source_metric_block_text": b["text"],
                    "period_disclosure": pair.group(0),
                    "provenance": "single_block_explicit_standalone_consolidated_fy_metric",
                })
        for match in _NARRATIVE_REVENUE.finditer(text):
            fy = int(match.group(2))
            period = f"FY{fy-1}-{fy%100:02d}"
            value = match.group(3).replace(",", "")
            identity = "|".join([source["file_sha256"], str(page_number), "narrative",
                                 match.group(1).casefold(), period, value])
            result.append({
                "fact_id": sha256(identity.encode()).hexdigest()[:24],
                "ticker": source["ticker"], "company": source["company"],
                "source_file": source["source_file"], "file_sha256": source["file_sha256"],
                "source_url": source["source_url"], "published_on": source["published_on"],
                "physical_pdf_page": page_number, "metric": "total_net_revenue",
                "segment": "", "segment_source_label": "",
                "reporting_scope": "overview_unspecified", "observed_period": period,
                "value": value, "source_literal": match.group(3) + " crore",
                "currency": "INR", "unit": "crore", "status": "auto_anchored",
                "row_label": "Total net revenue",
                "source_heading_bbox": [round(v, 1) for v in b["bbox"]],
                "source_metric_bbox": [round(v, 1) for v in b["bbox"]],
                "source_metric_block_text": b["text"],
                "period_disclosure": match.group(0),
                "provenance": "single_block_named_financial_metric_with_explicit_year",
            })
    return result



def _extract_cross_checked_highlights(blocks: list[dict[str, Any]], source: dict,
                                      page_number: int) -> list[dict[str, Any]]:
    """A headline KPI independently repeated in a fiscal-year table on that page.

    Current-year columns and the accounting basis must be explicit in the PDF;
    the primary headline amount alone is NOT enough to infer a fiscal period.
    """
    all_text = "\n".join(b["text"] for b in blocks)
    if not re.search(r"based\s+on\s+IFRS\s+consolidated\s+financial\s+statements", all_text, re.I):
        return []
    has_header = re.search(r"FY\s*(20\d{2})\s+FY\s*(20\d{2})", re.sub(r"\s+", " ", all_text), re.I)
    if not has_header:
        return []
    fy = int(has_header.group(1))
    if fy <= int(has_header.group(2)):
        return []
    period = f"FY{fy-1}-{fy%100:02d}"
    result = []
    for b in blocks:
        lines = [x.strip() for x in b["text"].splitlines() if x.strip()]
        if not lines or _norm(lines[0]) != "basic earnings per share":
            continue
        if b["font_size"] < 13:
            continue
        amounts = [x for x in lines[1:] if re.fullmatch(r"\d+(?:\.\d+)?", x)]
        if len(amounts) != 1:
            continue
        amount = amounts[0]
        # Two independent source regions must agree: headline and the FY2025
        # first-column row in the consolidated key-trends table.
        confirming = []
        for row in blocks:
            rowlines = [x.strip() for x in row["text"].splitlines() if x.strip()]
            if len(rowlines) >= 3 and re.match(r"^basic earnings per share\s*\(in\s*₹", rowlines[0], re.I):
                if rowlines[1] == amount and row["bbox"][1] > b["bbox"][3]:
                    confirming.append(row)
        if len(confirming) != 1:
            continue
        identity = "|".join([source["file_sha256"], str(page_number), "cross-checked-headline",
                             "earnings_per_share", period, amount])
        result.append({
            "fact_id": sha256(identity.encode()).hexdigest()[:24],
            "ticker": source["ticker"], "company": source["company"],
            "source_file": source["source_file"], "file_sha256": source["file_sha256"],
            "source_url": source["source_url"], "published_on": source["published_on"],
            "physical_pdf_page": page_number, "metric": "earnings_per_share",
            "segment": "", "segment_source_label": "",
            "reporting_scope": "consolidated", "observed_period": period,
            "component": "basic", "value": amount, "source_literal": amount,
            "currency": "INR", "unit": "rupees", "status": "auto_anchored",
            "source_heading_bbox": [round(v, 1) for v in b["bbox"]],
            "source_metric_bbox": [round(v, 1) for v in confirming[0]["bbox"]],
            "source_metric_block_text": b["text"],
            "period_disclosure": f"FY{fy} first column and basic EPS row; IFRS consolidated",
            "provenance": "headline_fiscal_table_double_source_check",
        })
    return result


def extract_page_panels(page, source: dict[str, str], page_number: int) -> list[dict[str, Any]]:
    """Only pairs of revenue/EBITDA cards under nearby larger headings qualify."""
    blocks = _blocks(page)
    page_text = "\n".join(b["text"] for b in blocks)
    year, year_note = _period(page_text)
    results = _extract_text_financial_grid(page, blocks, source, page_number)
    results.extend(_extract_local_year_cards(blocks, source, page_number))
    results.extend(_extract_explicit_period_narratives(blocks, source, page_number))
    results.extend(_extract_cross_checked_highlights(blocks, source, page_number))
    if year is None:
        return results  # Grid dates come from actual column headings, not the report title.
    metrics = [(b, parsed) for b in blocks if (parsed := _metrics(b))]
    revenues = [(b, parsed) for b, parsed in metrics if parsed["metric"] == "revenue"]
    ebitdas = [(b, parsed) for b, parsed in metrics if parsed["metric"] == "ebitda"]
    consumed = set()
    for revenue, rev_value in revenues:
        rx0, ry0, rx1, ry1 = revenue["bbox"]
        possible = [(abs(b["bbox"][1]-ry0), b, v) for b, v in ebitdas
                    if b["bbox"][0] > rx1 + 3 and b["bbox"][0] - rx1 < 210
                    and abs(b["bbox"][1]-ry0) < 12 and abs(b["bbox"][3]-ry1) < 20]
        if len(possible) != 1:
            continue
        _, ebitda, ebitda_value = possible[0]
        headings = [b for b in blocks if _heading(b, revenue)]
        if not headings:
            continue
        headings.sort(key=lambda b: ry0-b["bbox"][3])
        if len(headings) >= 2 and abs(headings[0]["bbox"][3]-headings[1]["bbox"][3]) < 3:
            continue  # Ambiguous heading assignment.
        heading = headings[0]
        segment = _norm(heading["text"])
        # Cross panel uniqueness: never accept two distinct values for the same
        # company, segment, metric, fiscal period on one page.
        for metric_block, extracted in ((revenue, rev_value), (ebitda, ebitda_value)):
            key = (segment, extracted["metric"], year)
            if key in consumed:
                continue
            consumed.add(key)
            identity = "|".join([source["file_sha256"], str(page_number), segment,
                                 extracted["metric"], year, extracted["value"]])
            results.append({
                "fact_id": sha256(identity.encode()).hexdigest()[:24],
                "ticker": source["ticker"], "company": source["company"],
                "source_file": source["source_file"], "file_sha256": source["file_sha256"],
                "source_url": source["source_url"], "published_on": source["published_on"],
                "physical_pdf_page": page_number, "metric": extracted["metric"],
                "segment": segment, "segment_source_label": heading["text"],
                "reporting_scope": "segment", "observed_period": year,
                "value": extracted["value"], "source_literal": extracted["literal"],
                "currency": "INR", "unit": "crore", "status": "auto_anchored",
                "source_heading_bbox": [round(v, 1) for v in heading["bbox"]],
                "source_metric_bbox": [round(v, 1) for v in metric_block["bbox"]],
                "source_metric_block_text": metric_block["text"],
                "period_disclosure": year_note, "provenance": "spatial_paired_revenue_ebitda_panel",
            })
    # Overview cards need a *different* fiscal-year disclosure: unlike the
    # revenue/EBITDA panels, a generic summary page must explicitly say its
    # RESULTS cover the year. A general report title/year is insufficient.
    normalized_text = re.sub(r"\s+", " ", page_text)
    overview_note = OVERVIEW_NOTE.search(normalized_text)
    if overview_note and re.search(r"\bFY\s*\d{2}\b", page_text, re.I):
        y = int(overview_note.group(1))
        card_period = f"FY{y-1}-{y%100:02d}"
        results.extend(_extract_overview_cards(blocks, source, page_number, card_period))
        # Word geometry is independent of PyMuPDF's block-merging decisions.
        # Prefer the existing block result; recover only genuinely missing KPIs.
        present = {(row["metric"], row["value"], row["observed_period"])
                   for row in results}
        for recovered in _extract_spatial_overview_cards(page, source, page_number,
                                                         card_period):
            key = (recovered["metric"], recovered["value"], recovered["observed_period"])
            if key not in present:
                results.append(recovered)
                present.add(key)
    return results


def build_panel_bank(source_specs: list[dict[str, str]], pdf_dir: str | Path, output: str | Path,
                     *, max_pages: int | None = None) -> dict[str, Any]:
    import pymupdf
    pdf_dir, output = Path(pdf_dir), Path(output)
    all_facts: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    for s in source_specs:
        path = pdf_dir / s["source_file"]
        if path.name != s["source_file"] or not path.is_file():
            raise ValueError("Missing or invalid source PDF: " + s["source_file"])
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != s["file_sha256"]:
            raise ValueError("Source PDF hash mismatch: " + path.name)
        count = 0
        scanned = 0
        with pymupdf.open(path) as pdf:
            for ix in range(min(len(pdf), max_pages or len(pdf))):
                scanned += 1
                facts = extract_page_panels(pdf[ix], s, ix+1)
                count += len(facts)
                all_facts.extend(facts)
        stats[s["source_file"]] = {"pages_scanned": scanned, "facts": count}
    # Ambiguously repeated same financial observation with conflicting values
    # cannot be promoted to a confident answer.
    grouped: dict[tuple[str, str, str, str], set[str]] = {}
    for f in all_facts:
        key = (f["ticker"], f["segment"], f["metric"], f["observed_period"], f["reporting_scope"])
        grouped.setdefault(key, set()).add(f["value"])
    for f in all_facts:
        key = (f["ticker"], f["segment"], f["metric"], f["observed_period"], f["reporting_scope"])
        if len(grouped[key]) != 1:
            f["status"] = "review_required"
            f["reason"] = "conflicting_values_across_report"
    data = {"schema_version": 6, "status": "SPATIAL_PANEL_EXTRACTION_NOT_HUMAN_CERTIFIED",
            "note": "Source geometry + exact PDF hash rechecked at answer time; not full financial-statement understanding.",
            "sources": source_specs, "facts": all_facts, "source_stats": stats}
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(output)
    return {"path": str(output), "facts": len(all_facts), "auto_anchored": sum(x["status"] == "auto_anchored" for x in all_facts),
            "review_required": sum(x["status"] != "auto_anchored" for x in all_facts), "sources": stats}


def default_panel_bank_path() -> Path:
    import os
    return Path(os.getenv(
        "RISKLENS_LAYOUT_BANK_V6",
        "data/processed/research/real_acceptance/phase7_completion/financial_layout_panels_v6.json",
    ))


def read_panel_bank(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 6 or not isinstance(data.get("facts"), list):
        raise ValueError("Unrecognized layout-panel fact bank")
    # Internal source identity is part of the schema, not an arbitrary label
    # that a malformed fact is allowed to redefine.
    identities = {s["file_sha256"]: s for s in data.get("sources", [])}
    if not identities:
        raise ValueError("Layout bank is missing source provenance")
    for fact in data["facts"]:
        spec = identities.get(fact.get("file_sha256"))
        if spec is None or any(fact.get(k) != spec.get(k) for k in (
            "ticker", "company", "source_file", "source_url", "published_on",
        )):
            raise ValueError("Layout fact has inconsistent source identity")
    return data["facts"]


def verify_original_panel(f: dict[str, Any], pdf_dir: str | Path) -> bool:
    """Trust neither stored value nor geometry: re-extract original spatial panel."""
    import pymupdf
    name = f.get("source_file", "")
    if Path(name).name != name or not name:
        return False
    path = Path(pdf_dir) / name
    if not path.is_file() or sha256(path.read_bytes()).hexdigest() != f.get("file_sha256"):
        return False
    with pymupdf.open(path) as pdf:
        page_no = f.get("physical_pdf_page", 0)
        if not isinstance(page_no, int) or not 1 <= page_no <= len(pdf):
            return False
        reference = extract_page_panels(pdf[page_no-1], f, page_no)
        for candidate in reference:
            fields = ("fact_id", "ticker", "company", "source_file", "file_sha256",
                      "segment", "segment_source_label", "metric", "observed_period",
                      "reporting_scope", "value", "currency", "unit", "source_literal",
                      "source_metric_block_text", "row_label", "provenance")
            if any(candidate.get(k) != f.get(k) for k in fields):
                continue
            if candidate["status"] != "auto_anchored" or f.get("status") != "auto_anchored":
                continue
            a, b = candidate["source_heading_bbox"], f.get("source_heading_bbox", [])
            c, d = candidate["source_metric_bbox"], f.get("source_metric_bbox", [])
            if len(b) != 4 or len(d) != 4 or any(abs(x-y) > 10 for x,y in zip(a,b)) or any(abs(x-y) > 10 for x,y in zip(c,d)):
                continue
            return True
    return False


def _requested_period(question: str) -> str | None:
    q = question.casefold()
    m = re.search(r"\bFY\s*(20\d{2})\b", q, re.I)
    if not m:
        m = re.search(r"\b(?:march|mar)\s+(?:31[,]?\s*)?(20\d{2})\b", q, re.I)
    if m:
        y = int(m.group(1))
        return f"FY{y-1}-{y%100:02d}"
    m = re.search(r"\b(20\d{2})\s*[-–/]\s*(?:20)?(\d{2})\b", q)
    if m:
        return f"FY{m.group(1)}-{m.group(2)}"
    return None


def answer_from_layout_panels(question: str, facts: list[dict[str, Any]], *, pdf_dir: str | Path,
                              eligible_hashes: set[str] | None = None, ticker: str | None = None,
                              as_of: str | None = None,
                              source_registry: dict[str, str] | None = None) -> dict[str, Any] | None:
    q = _norm(question)
    requested = set()
    if re.search(r"\brevenue\b", q):
        requested.add("revenue")
    if re.search(r"\bebitda\b", q):
        requested.add("ebitda")
    for metric, pattern in CARD_METRICS + _GRID_ROW_METRICS:
        # Source-table row patterns are exact matches; user questions also
        # contain fiscal years and units *after* the metric words.
        query_pattern = r"\b" + pattern.removeprefix("^").removesuffix("$") + r"\b"
        if re.search(query_pattern, q):
            requested.add(metric)
    if "goodwill and other intangible" in q:
        requested.add("goodwill_intangibles")
    if "net current assets" in q:
        requested.add("net_current_assets")
    if "total net revenue" in q:
        requested = {"total_net_revenue"}
    if len(requested) != 1:
        return None  # Never conflate two metrics or answer a compound question partially.
    requested_period = _requested_period(question)
    if re.search(r"\b20\d{2}\b", question) and requested_period is None:
        return None  # Explicit year not successfully parsed; fail closed.
    requested_scope = "consolidated" if "consolidated" in q else "standalone" if "standalone" in q else None
    wanted = next(iter(requested))
    company_q = {"reliance": "RIL", "ril": "RIL", "hdfc": "HDFCBANK", "infosys": "INFY", "icici": "ICICIBANK"}
    company_tokens = {symbol for label, symbol in company_q.items() if re.search(r"\b"+re.escape(label)+r"\b", q)}
    if ticker:
        company_tokens.add(ticker.upper())
    if len(company_tokens) > 1:
        return None
    chosen = []
    for f in facts:
        if f.get("status") != "auto_anchored" or f.get("metric") != wanted:
            continue
        if eligible_hashes is not None and f["file_sha256"] not in eligible_hashes:
            continue
        if source_registry is not None and source_registry.get(f["file_sha256"]) != f["ticker"]:
            continue
        if as_of and f["published_on"] > as_of:
            continue
        if company_tokens and f["ticker"] not in company_tokens:
            continue
        segment = _norm(f["segment_source_label"])
        if segment:
            if requested_scope is not None:
                continue  # A segment KPI is not a consolidated/standalone statement.
            if not re.search(r"\b" + re.escape(segment) + r"\b", q):
                continue
        else:
            if not company_tokens:  # An unlabeled company overview is not globally attributable.
                continue
            # Overview cards have no audited accounting scope. Word-grid cells
            # have explicit Standalone and Consolidated column headings.
            observed_scope = f["reporting_scope"]
            if requested_scope and observed_scope != requested_scope:
                continue
            if not requested_scope and observed_scope in {"standalone", "consolidated"}:
                # Caller may get more than one alternative; unique-key check
                # below will fail closed instead of picking arbitrarily.
                pass
        if requested_period is not None and f["observed_period"] != requested_period:
            continue
        if re.search(r"\b(?:thousands? of rupees|billion|million|usd)\b", q):
            continue
        if f["unit"] == "percent" and re.search(r"\bcrore\b|\brupees\b", q):
            continue
        if f["unit"] == "crore" and re.search(r"\bpercent\b|\bpercentage\b", q):
            continue
        if f["unit"] == "rupees" and re.search(r"\b(?:crore|percent|percentage)\b", q):
            continue
        component = f.get("component")
        if component and component not in q:
            continue
        if "diluted" in q and component != "diluted":
            continue
        chosen.append(f)
    uniq = {(f["ticker"], f["segment"], f["metric"], f["observed_period"], f["value"], f["unit"], f["reporting_scope"]) for f in chosen}
    if len(uniq) != 1:
        return None
    f = sorted(chosen, key=lambda x: (x["physical_pdf_page"], x["fact_id"]))[0]
    if not verify_original_panel(f, pdf_dir):
        return {"question": question, "answer": "Original financial panel could not be reverified.",
                "status": "needs_review", "route": "layout_panel_source_check_failed", "provider": "local_pdf",
                "citations": [], "verification_status": "source_check_failed"}
    amount = Decimal(f["value"])
    val = format(amount, ",f") if amount != amount.to_integral() else f"{amount:,.0f}"
    suffix = "%" if f["unit"] == "percent" else " crore" if f["unit"] == "crore" else ""
    prefix = "" if f["unit"] == "percent" else "₹"
    citation = {"fact_id": f["fact_id"], "ticker": f["ticker"], "company": f["company"],
                "file_sha256": f["file_sha256"], "source_url": f["source_url"],
                "pdf_page": f["physical_pdf_page"], "source_heading": f["segment_source_label"],
                "segment": f["segment"], "metric": f["metric"], "unit": f["unit"],
                "source_metric": f["metric"], "source_literal": f["source_literal"],
                "reported_period": f["observed_period"], "reported_scope": f["reporting_scope"],
                "trust_level": "native_spatial_panel_rechecked_not_human_certified"}
    scope_name = f["reporting_scope"] if f["reporting_scope"] in {"standalone", "consolidated"} else ""
    if f["segment_source_label"]:
        name = f["segment_source_label"].replace("\n", " ") + " " + f["metric"].upper()
    else:
        label = f.get("row_label", f["metric"].replace("_", " "))
        name = (scope_name + " " + label).strip()
    return {"question": question,
            "answer": f"{f['company']} — {name} ({f['observed_period']}): {prefix}{val}{suffix}",
            "status": "answered", "provider": "source_checked_layout_panel_json",
            "route": "spatial_financial_panel_json", "citations": [citation],
            "verification_status": "pdf_sha_heading_metric_value_geometry_rechecked_not_human_certified"}
