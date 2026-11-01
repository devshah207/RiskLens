"""Thin, opt-in, no-write wrapper over merged Phase 7 XBRL-first/PDF RAG.

The Phase 7 database builder normally initializes CREATE TABLE statements.
This UI deliberately creates a READ-ONLY SQLite connection instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
import zipfile
from urllib.parse import urlparse


COMPANIES = {
    "Reliance Industries": "RIL",
    "HDFC Bank": "HDFCBANK",
    "Infosys": "INFY",
    "ICICI Bank": "ICICIBANK",
}


@dataclass(frozen=True)
class ResearchPaths:
    archive: Path
    index: Path
    banks: Path
    pdfs: Path

    @classmethod
    def for_root(cls, root: Path) -> "ResearchPaths":
        root = Path(root).resolve()
        return cls(
            archive=Path.home() / "Downloads" / "Archive.zip",  # original owner archive
            index=root / "data/processed/research/real_acceptance/real_index.sqlite3",
            banks=root / "data/processed/research/real_acceptance/phase7_completion",
            pdfs=root / "data/raw/research_reports",
        )


REQUIRED_BANKS = (
    "financial_facts_v3.json",
    "financial_facts_auto_v4.json",
    "financial_layout_panels_v6.json",
)


def source_readiness(paths: ResearchPaths) -> dict[str, object]:
    """Report only local resource presence; NOT a full file/provenance audit."""
    missing_banks = [name for name in REQUIRED_BANKS if not (paths.banks / name).is_file()]
    return {
        "xbrl_archive_present": paths.archive.is_file(),
        "pdf_index_present": paths.index.is_file(),
        "pdf_directory_present": paths.pdfs.is_dir(),
        "bank_files_present": not missing_banks,
        "missing_bank_filenames": missing_banks,
    }


def safe_source_url(value: object) -> str | None:
    """Permit an HTTPS citation only, never a local file/JS URL from a PDF."""
    if not isinstance(value, str) or len(value) > 2048:
        return None
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        return None
    return value


def normalize_response(payload: dict) -> dict:
    """Preserve Phase 7 statuses/citations; never upgrade a model answer to verified."""
    if not isinstance(payload, dict):
        raise ValueError("Research backend returned an unexpected response")
    status = str(payload.get("status") or "not_evaluated")
    known = {"answered", "needs_review", "abstained", "not_evaluated",
             "provider_unavailable", "provider_error"}
    if status not in known:
        status = "not_evaluated"
    raw_citations = payload.get("citations")
    citations = [c for c in raw_citations if isinstance(c, dict)] if isinstance(raw_citations, list) else []
    # The original offline RAG returns candidate pages even when no answer is
    # certified. Keep these separately -- they are NOT source citations.
    raw_pages = payload.get("retrieved_physical_pages")
    candidates = []
    if isinstance(raw_pages, list):
        for location in raw_pages[:20]:
            if (isinstance(location, (list, tuple)) and len(location) == 2
                    and isinstance(location[0], str) and isinstance(location[1], int)
                    and 1 <= location[1] <= 100000):
                candidates.append((location[0], location[1]))
    return {
        "question": str(payload.get("question") or ""),
        "answer": str(payload.get("answer") or "No answer returned."),
        "status": status,
        "route": str(payload.get("route") or "unknown"),
        "provider": str(payload.get("provider") or "unknown"),
        "verification_status": str(payload.get("verification_status") or "not_independently_certified"),
        "intent": payload.get("intent") if isinstance(payload.get("intent"), dict) else {},
        "citations": citations,
        "retrieved_physical_pages": candidates,
        "recorded_source": payload.get("recorded_source"),
    }


def _not_evaluated(question: str, reason: str, route: str) -> dict:
    return normalize_response({"question": question, "answer": reason,
                               "status": "not_evaluated", "route": route, "citations": []})


def _lazy_embedder():
    """Return a 384-dimensional adapter; cache MISS never downloads a model."""
    class CachedLocalMiniLM:
        name = "sentence-transformers/all-MiniLM-L6-v2"
        dimension = 384

        def __init__(self):
            self._model = None

        def encode(self, texts):
            if self._model is None:
                from risklens.research.embeddings import SentenceEmbedder
                self._model = SentenceEmbedder(local_files_only=True)
            return self._model.encode(texts)

    return CachedLocalMiniLM()


def _read_only_store(db_path: Path):
    """Yield a Phase 7 ResearchStore instance with SQLite query-only mode.

    The constructor is intentionally NOT called: its normal purpose includes
    creating tables, which is not acceptable from a read-only dashboard.
    """
    from risklens.research.index import ResearchStore

    class StoreContext:
        def __enter__(self):
            self.conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA query_only=ON")
            self.store = ResearchStore.__new__(ResearchStore)
            self.store.path = db_path
            self.store.conn = self.conn
            return self.store

        def __exit__(self, *_):
            self.conn.close()

    return StoreContext()


def ask_saved_sources(
    question: str,
    *,
    ticker: str,
    paths: ResearchPaths,
    scope: str | None = None,
    year: int = 2025,
    groq_allowed: bool = False,
    provider: str = "offline",
) -> dict:
    """Call ORIGINAL XBRL-first route. Network/Groq only after explicit consent.

    A question must itself include the year and reporting scope where needed:
    silently appending them could change the meaning of a question. This UI
    validates the matching company instead of rewriting the user's question.
    """
    question = question.strip()
    if not question or len(question) > 1200:
        raise ValueError("Enter a question between 1 and 1200 characters")
    if ticker not in COMPANIES.values():
        raise ValueError("Select a supported original issuer")
    if year != 2025 and year != 2024:
        return _not_evaluated(question, "The local FY2025 filings cover FY2025 and some FY2024 comparative facts; other years are not offered in this UI.", "unavailable_year")
    if scope not in {None, "standalone", "consolidated"}:
        raise ValueError("Unknown scope filter")
    if provider not in {"groq", "offline"}:
        raise ValueError("Unknown provider")
    # Never silently change a question's issuer/year/scope using a dropdown.
    issuer_phrases = {
        "RIL": r"\b(?:reliance(?:\s+industries)?|ril)\b",
        "HDFCBANK": r"\b(?:hdfc(?:\s+bank)?|hdfcbank)\b",
        "INFY": r"\b(?:infosys|infy)\b",
        "ICICIBANK": r"\b(?:icici(?:\s+bank)?|icicibank)\b",
    }
    mentioned = {symbol for symbol, pattern in issuer_phrases.items()
                 if re.search(pattern, question, flags=re.I)}
    if mentioned and mentioned != {ticker}:
        return _not_evaluated(question, "Question issuer conflicts with selected company. Single-company research questions only.", "issuer_filter_conflict")
    if scope:
        found = set(re.findall(r"\b(?:standalone|consolidated)\b", question.lower()))
        if found and found != {scope}:
            return _not_evaluated(question, "Question reporting basis conflicts with the selected reporting basis. Please make them match.", "scope_filter_conflict")
    period_pair = re.search(r"\bFY\s*(20\d{2})\s*[-–/]\s*(?:20)?(\d{2})\b", question, flags=re.I)
    if period_pair:
        start, end = int(period_pair.group(1)), 2000 + int(period_pair.group(2))
        requested_year = end if end == start + 1 else None
    else:
        found_years = {int(y) for y in re.findall(r"\bFY\s*(20\d{2})\b", question, flags=re.I)}
        requested_year = next(iter(found_years)) if len(found_years) == 1 else None
    if requested_year is not None and requested_year != year:
        return _not_evaluated(question, "Question fiscal year conflicts with selected year. Please make them match.", "year_filter_conflict")
    if provider == "groq" and not groq_allowed:
        return _not_evaluated(question, "Groq requires an explicit confirmation for each live question.", "groq_consent_required")
    if not paths.archive.is_file():
        return _not_evaluated(question,
                              "Original NSE XBRL Archive.zip is unavailable. Set the local archive path before asking a source-backed question.",
                              "missing_original_xbrl_archive")

    # Exact XBRL first; no Groq and no PDF index necessary for a direct match.
    try:
        from risklens.research.xbrl_rag_v1 import lookup_xbrl, run_query_xbrl_first
        direct = lookup_xbrl(question, archive=paths.archive, ticker=ticker,
                             fiscal_year=None, as_of=None)
    except (ImportError, FileNotFoundError, ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        return _not_evaluated(question, f"Cannot read the original XBRL source ({type(exc).__name__}); check the eight-file archive and installed research requirements.",
                              "xbrl_source_unavailable")
    if direct is not None:
        return normalize_response(direct)
    if not paths.index.is_file():
        return _not_evaluated(question, "No exact matching XBRL fact. Local PDF research index is unavailable; restore the original Phase 7 index first.",
                              "pdf_index_missing")
    readiness = source_readiness(paths)
    if not readiness["bank_files_present"]:
        missing = ", ".join(readiness["missing_bank_filenames"])
        return _not_evaluated(question,
                              f"Original Phase 7 PDF fallback requires saved bank files: {missing}. Restore them from the existing Phase 7 research outputs.",
                              "missing_original_phase7_banks")
    if not readiness["pdf_directory_present"]:
        return _not_evaluated(question, "Original Phase 7 annual-report PDF folder is missing. Restore the original report files before PDF research.",
                              "missing_original_report_directory")

    try:
        with _read_only_store(paths.index) as store:
            if not store.list_documents():
                return _not_evaluated(question, "Original Phase 7 PDF index contains no documents.", "pdf_index_empty")
            result = run_query_xbrl_first(
                store, _lazy_embedder(), question, xbrl_archive=paths.archive,
                ticker=ticker, fiscal_year=None, as_of=None,
                provider=provider, top_k=5, bank_dir=paths.banks,
                pdf_dir=paths.pdfs, verify_financial=True, verify_vision=False,
                # Preserve original Phase 7 verification policy. In live Groq
                # mode an uncertain numeric answer can require an extra request.
            )
        return normalize_response(result)
    except (ImportError, OSError, sqlite3.Error, RuntimeError, ValueError) as exc:
        # Type only: do not leak secrets, source paths, or Groq HTTP payloads.
        return normalize_response({
            "question": question,
            "answer": f"Local research could not finish ({type(exc).__name__}). Check the saved index, cached embedding model and original PDFs; no answer is claimed.",
            "status": "not_evaluated",
            "route": "research_adapter_unavailable", "citations": [],
        })
