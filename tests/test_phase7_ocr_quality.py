"""No-network tests for targeted financial PDF QA and conservative vision fallback."""

from pathlib import Path
import importlib.util
import pytest


def _module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/phase7_review_flagged.py"
    spec = importlib.util.spec_from_file_location("phase7_review_flagged", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_multiset_counts_and_no_auto_truth():
    diff = _module().compare_numbers(
        "FY2025 revenue 100 100 280", "FY2025 revenue 100 280", "100 999"
    )
    assert diff["native_not_found_in_docling"] == 1
    assert "999" in diff["vision_not_found_in_native_samples"]
    assert "do NOT establish" in diff["warning"]


def test_docling_slice_and_optional_groq(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    pdf = pymupdf.open()
    for page in range(3):
        p = pdf.new_page()
        p.insert_text((60, 80), f"ORIGINAL PAGE {page + 1} CONSOLIDATED REVENUE 125000")
    path = tmp_path / "sample.pdf"
    pdf.save(path)
    pdf.close()
    parsed = []
    called = []

    def faux_docling(p, count):
        with pymupdf.open(str(p)) as chunk:
            parsed.append((len(chunk), chunk[0].get_text()))
        return ["ORIGINAL PAGE 2 CONSOLIDATED REVENUE 125000"]

    def faux_groq(p, page, *, model):
        called.append((p, page, model))
        return "UNVERIFIED REVENUE 125000"

    qa = _module().quality_review(
        path, 2, tmp_path / "qa", groq=True, parser=faux_docling, vision=faux_groq
    )
    assert parsed[0][0] == 1
    assert "PAGE 2" in parsed[0][1]
    assert called[0][1] == 2
    assert qa["original_physical_pdf_page"] == 2
    assert qa["verified_against_original_render"] is False
    assert qa["do_not_index_groq_draft"] is True
    assert (tmp_path / "qa" / "original_page.png").is_file()


def test_no_network_call_when_not_opted_in(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    pdf = pymupdf.open()
    pdf.new_page().insert_text((40, 70), "Revenue FY 2025 999")
    f = tmp_path / "a.pdf"
    pdf.save(f)
    pdf.close()

    def forbidden(*a, **kw):
        raise AssertionError("Must not call Groq without opt-in")

    r = _module().quality_review(
        f,
        1,
        tmp_path / "local",
        parser=lambda *x: ["Revenue FY 2025 999"],
        vision=forbidden,
    )
    assert r["groq_attempted"] is False


def test_existing_ingest_numeric_mismatch_triggers_vision_only_if_requested(
    monkeypatch, tmp_path
):
    pymupdf = pytest.importorskip("pymupdf")
    from risklens.research.contracts import SourceSpec
    from risklens.research import ingest as ing

    pdf = pymupdf.open()
    pdf.new_page().insert_text(
        (40, 70), "Revenue 100000 profit 200000 liabilities 300000"
    )
    f = tmp_path / "audit.pdf"
    pdf.save(f)
    pdf.close()
    spec = SourceSpec(
        "Synthetic",
        "MOCK",
        "2024-25",
        "Synthetic report",
        "https://example.org/audit.pdf",
        "2025-06-02",
        str(f),
        "fixture",
    )
    monkeypatch.setattr(
        ing, "_docling_pages", lambda *a: ["Revenue 100000 profit UNKNOWN"]
    )
    calls = []
    monkeypatch.setattr(
        ing,
        "groq_vision_transcript",
        lambda *a, **kw: (
            calls.append(a)
            or "UNVERIFIED Revenue 100000 profit 200000 liabilities 300000"
        ),
    )
    _, pg = ing.load_pdf(spec, parser="docling", vision_fallback=False)
    assert pg[0].review_required and not calls
    _, pg = ing.load_pdf(spec, parser="docling", vision_fallback=True)
    assert (
        pg[0].review_required
        and pg[0].method == "groq_vision_unverified"
        and len(calls) == 1
    )
