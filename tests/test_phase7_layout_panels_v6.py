"""No benchmark answer mapping enters the extractor. Independent source tests."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import pytest

from risklens.research.layout_panels_v6 import (
    extract_page_panels,
    answer_from_layout_panels,
    verify_original_panel,
    build_panel_bank,
)


def _make_synthetic(tmp_path):
    import pymupdf
    pdf_path = tmp_path / "arbitrary_company.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=750, height=600)
        page.insert_text((42, 76), "Exploration and Production", fontsize=25)
        page.insert_text((42, 104), "REVENUE  6.4%\nRs 17,543 Crore", fontsize=10, lineheight=1.0)
        page.insert_text((225, 104), "EBITDA  3.2%\nRs 9,875 Crore", fontsize=10, lineheight=1.0)
        page.insert_text((42, 530),
                         "Note: All Revenue and EBITDA figures are for the year ended March 31, 2025",
                         fontsize=8)
        pdf.save(pdf_path)
    source = {
        "ticker": "DEMO", "company": "Demonstration Company", "source_file": pdf_path.name,
        "file_sha256": sha256(pdf_path.read_bytes()).hexdigest(),
        "source_url": "https://example.invalid/synthetic_source.pdf",
        "published_on": "2025-08-01",
    }
    return pdf_path, source


def test_independent_synthetic_kpi_panel(tmp_path):
    import pymupdf
    path, spec = _make_synthetic(tmp_path)
    with pymupdf.open(path) as pdf:
        found = extract_page_panels(pdf[0], spec, 1)
    assert {f["metric"]: f["value"] for f in found} == {"revenue": "17543", "ebitda": "9875"}
    assert all(verify_original_panel(f, tmp_path) for f in found)
    result = answer_from_layout_panels(
        "Exploration and Production EBITDA FY2025 in crore?", found,
        pdf_dir=tmp_path, ticker="DEMO", eligible_hashes={spec["file_sha256"]},
    )
    assert result is not None and result["status"] == "answered"
    assert "9,875 crore" in result["answer"]
    assert result["citations"][0]["segment"] == "exploration and production"
    assert answer_from_layout_panels(
        "Exploration and Production EBITDA FY2025", found, pdf_dir=tmp_path,
        source_registry={spec["file_sha256"]: "WRONG"}, ticker="DEMO",
    ) is None


def test_year_cannot_be_fabricated(tmp_path):
    import pymupdf
    path, spec = _make_synthetic(tmp_path)
    with pymupdf.open() as doc:
        page = doc.new_page(width=750, height=600)
        page.insert_text((42, 76), "Exploration and Production", fontsize=25)
        page.insert_text((42, 104), "REVENUE  6.4%\nRs 17,543 Crore", fontsize=10, lineheight=1.0)
        page.insert_text((225, 104), "EBITDA  3.2%\nRs 9,875 Crore", fontsize=10, lineheight=1.0)
        assert extract_page_panels(page, spec, 1) == []


def test_tampering_and_wrong_segment_fail(tmp_path):
    import pymupdf
    path, spec = _make_synthetic(tmp_path)
    with pymupdf.open(path) as pdf:
        facts = extract_page_panels(pdf[0], spec, 1)
    f = next(f for f in facts if f["metric"] == "ebitda")
    assert not verify_original_panel({**f, "value": "9876"}, tmp_path)
    assert not verify_original_panel({**f, "segment": "different company"}, tmp_path)
    assert not verify_original_panel({**f, "reporting_scope": "consolidated"}, tmp_path)
    assert not verify_original_panel({**f, "currency": "USD"}, tmp_path)
    assert not verify_original_panel({**f, "source_metric_bbox": [0, 0, 0, 0]}, tmp_path)
    assert answer_from_layout_panels("Exploration and Production EBITDA FY2024", facts,
                                     pdf_dir=tmp_path, ticker="DEMO") is None
    assert answer_from_layout_panels("Exploration and Production EBITDA FY2025 consolidated", facts,
                                     pdf_dir=tmp_path, ticker="DEMO") is None
    assert answer_from_layout_panels("Different Division EBITDA FY2025", facts,
                                     pdf_dir=tmp_path, ticker="DEMO") is None
    assert answer_from_layout_panels("Exploration and Production EBITDA FY2025 in million", facts,
                                     pdf_dir=tmp_path, ticker="DEMO") is None
    assert answer_from_layout_panels("Exploration and Production revenue and EBITDA FY2025", facts,
                                     pdf_dir=tmp_path, ticker="DEMO") is None


def test_hash_protection_and_index_scoping(tmp_path):
    import pymupdf
    path, spec = _make_synthetic(tmp_path)
    with pymupdf.open(path) as pdf:
        facts = extract_page_panels(pdf[0], spec, 1)
    assert answer_from_layout_panels("Exploration and Production EBITDA FY2025", facts,
                                     pdf_dir=tmp_path, ticker="DEMO", eligible_hashes={"a"*64}) is None
    path.write_bytes(path.read_bytes() + b"\n% silently altered\n")
    assert not verify_original_panel(facts[0], tmp_path)
    out = answer_from_layout_panels("Exploration and Production EBITDA FY2025", facts,
                                    pdf_dir=tmp_path, ticker="DEMO")
    assert out and out["status"] == "needs_review"


def test_original_reliance_page3_and_hdfc_page5():
    import pymupdf
    root = Path("data/raw/research_reports")
    v4 = Path("data/processed/research/real_acceptance/phase7_completion/financial_facts_auto_v4.json")
    if not (root / "ril_fy2025.pdf").exists() or not v4.is_file():
        pytest.skip("Original source PDFs not installed in test checkout")
    srcs = json.loads(v4.read_text())["sources"]
    for ticker, page_number, assertions in (
        ("RIL", 3, {("oil and gas", "ebitda"): "21188",
                      ("oil and gas", "revenue"): "25211",
                      ("media and entertainment", "revenue"): "20696",
                      ("digital services", "ebitda"): "65001"}),
        ("HDFCBANK", 5, {("", "earnings_per_share"): "88.3",
                         ("", "cost_income_ratio"): "40.5"}),
    ):
        spec = next(x for x in srcs if x["ticker"] == ticker)
        with pymupdf.open(root / spec["source_file"]) as pdf:
            facts = extract_page_panels(pdf[page_number-1], spec, page_number)
        by_key = {(f["segment"], f["metric"]): f["value"] for f in facts}
        for key, value in assertions.items():
            assert by_key[key] == value
            selected = next(f for f in facts if (f["segment"], f["metric"]) == key)
            assert verify_original_panel(selected, root)


def test_bank_build_idempotent_on_synthetic(tmp_path):
    path, spec = _make_synthetic(tmp_path)
    output = tmp_path / "facts.json"
    one = build_panel_bank([spec], tmp_path, output)
    before = output.read_bytes()
    two = build_panel_bank([spec], tmp_path, output)
    assert one["facts"] == two["facts"] == 2
    assert output.read_bytes() == before


def test_original_hdfc_page5_with_fragmented_pdf_text_blocks():
    """HDFC KPI words must survive PyMuPDF version-specific block splitting."""
    import copy
    import pymupdf

    root = Path('data/raw/research_reports')
    v4 = Path('data/processed/research/real_acceptance/phase7_completion/financial_facts_auto_v4.json')
    if not (root / 'hdfcbank_fy2025.pdf').exists() or not v4.is_file():
        pytest.skip('Original source PDFs not installed in test checkout')
    spec = next(s for s in json.loads(v4.read_text())['sources']
                if s['ticker'] == 'HDFCBANK')

    class SplitTextBlocks:
        def __init__(self, page):
            self.page = page

        def get_text(self, mode, *args, **kwargs):
            if mode != 'dict':
                return self.page.get_text(mode, *args, **kwargs)
            original = copy.deepcopy(self.page.get_text('dict', *args, **kwargs))
            blocks = []
            for block in original['blocks']:
                if block.get('type') != 0 or 'Note:' in str(block):
                    blocks.append(block)
                    continue
                # Some real PyMuPDF versions split even a visually single
                # KPI card into separate lines/spans. Mimic that difference.
                for line in block.get('lines', []):
                    for span in line.get('spans', []):
                        blocks.append({'type': 0, 'bbox': span['bbox'],
                                       'lines': [{'spans': [span]}]})
            original['blocks'] = blocks
            return original

    with pymupdf.open(root / spec['source_file']) as pdf:
        recovered = extract_page_panels(SplitTextBlocks(pdf[4]), spec, 5)

    expected = {
        'earnings_per_share': ('88.3', 'rupees'),
        'cost_income_ratio': ('40.5', 'percent'),
        'return_on_assets': ('1.91', 'percent'),
        'profit_after_tax': ('67347', 'crore'),
    }
    for metric, (value, unit) in expected.items():
        found = [f for f in recovered if f['metric'] == metric]
        assert len(found) == 1, (metric, [(f['metric'], f['value']) for f in recovered])
        assert (found[0]['value'], found[0]['unit']) == (value, unit)
        assert found[0]['provenance'] == 'spatial_word_labeled_kpi_explicit_year_note'


def test_general_spatial_card_with_separate_label_number_blocks(tmp_path):
    """A synthetic KPI with arbitrary values should not need a joined block."""
    import pymupdf
    path = tmp_path / 'new_report.pdf'
    with pymupdf.open() as doc:
        p = doc.new_page(width=600, height=500)
        p.insert_text((48, 90), 'Cost to Income Ratio (%)', fontsize=12)
        p.insert_text((48, 124), '29.7', fontsize=27)
        p.insert_text((48, 380),
                      'Note: The results for the year ended March 31, 2025 '
                      'include the current company.', fontsize=8)
        p.insert_text((48, 405), 'FY25 FY24', fontsize=8)
        doc.save(path)
    spec = {'ticker': 'NEWCO', 'company': 'New Example Financial Co',
            'source_file': path.name,
            'file_sha256': sha256(path.read_bytes()).hexdigest(),
            'source_url': 'https://example.invalid/new.pdf',
            'published_on': '2025-09-01'}
    with pymupdf.open(path) as doc:
        found = extract_page_panels(doc[0], spec, 1)
    ratio = next(f for f in found if f['metric'] == 'cost_income_ratio')
    assert ratio['value'] == '29.7'
    assert ratio['unit'] == 'percent'
    assert ratio['provenance'] in {
        'spatial_word_labeled_kpi_explicit_year_note',
        'single_block_labeled_kpi_with_explicit_year_note',
    }
    assert verify_original_panel(ratio, tmp_path)
    assert not verify_original_panel({**ratio, 'value': '29.8'}, tmp_path)
