"""Immutable source identities, strict provenance, and public service contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import urlparse


class ResearchError(ValueError):
    """Fail closed on invalid provenance, unsafe generation, or missing evidence."""


def valid_iso_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise ResearchError(f"Expected an ISO publication/as-of date, got {value!r}") from exc
    if parsed.isoformat() != value:
        raise ResearchError(f"Date must be YYYY-MM-DD: {value!r}")
    return value


def stable_id(*parts: object) -> str:
    encoded = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class SourceSpec:
    company: str
    ticker: str
    fiscal_year: str
    title: str
    source_url: str
    published_on: str
    local_path: str
    license_note: str
    report_kind: str = "annual_report"
    document_version: str = "original"
    sector: str = "unspecified"

    def __post_init__(self) -> None:
        for name in ("company", "ticker", "fiscal_year", "title", "local_path", "license_note", "document_version"):
            if not getattr(self, name).strip():
                raise ResearchError(f"Required source metadata missing: {name}")
        valid_iso_date(self.published_on)
        u = urlparse(self.source_url)
        if u.scheme != "https" or not u.hostname:
            raise ResearchError("Source URL must use a provenance-bearing HTTPS link")
        if not self.ticker.replace("-", "").replace("_", "").isalnum():
            raise ResearchError("Ticker must contain only letters, digits, underscore or hyphen")

    @classmethod
    def from_dict(cls, data: dict) -> SourceSpec:
        if not isinstance(data, dict):
            raise ResearchError("Each source must be an object")
        allowed = set(cls.__dataclass_fields__)
        extra = set(data) - allowed
        if extra:
            raise ResearchError(f"Unexpected source fields: {sorted(extra)}")
        try:
            return cls(**data)
        except (TypeError, AttributeError) as exc:
            raise ResearchError(f"Invalid source metadata: {exc}") from exc


def read_manifest(path: str | Path) -> tuple[SourceSpec, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("sources"), list) or not raw["sources"]:
        raise ResearchError("Manifest must contain a nonempty sources array")
    specs = tuple(SourceSpec.from_dict(v) for v in raw["sources"])
    ids = [(s.ticker, s.fiscal_year, s.report_kind, s.document_version) for s in specs]
    if len(ids) != len(set(ids)):
        raise ResearchError("Duplicate ticker/year/report/version identities in manifest")
    return specs


@dataclass(frozen=True)
class Document:
    doc_id: str
    spec: SourceSpec
    sha256: str
    pages: int
    parser: str

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "spec": asdict(self.spec), "sha256": self.sha256,
                "pages": self.pages, "parser": self.parser}


@dataclass(frozen=True)
class Page:
    doc_id: str
    page_no: int  # 1-indexed physical PDF page, not printed footer number
    text: str
    method: str
    quality_flags: tuple[str, ...] = ()
    review_required: bool = False


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    page_no: int
    start: int
    end: int
    text: str
    section: str = ""
    review_required: bool = False


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    document: Document
    score: float
    retrieval_methods: tuple[str, ...]

    @property
    def citation(self) -> str:
        s = self.document.spec
        return f"{s.ticker} FY{s.fiscal_year}, PDF page {self.chunk.page_no}, chunk {self.chunk.chunk_id}"


def source_document_id(spec: SourceSpec, file_sha: str) -> str:
    return stable_id(spec.ticker, spec.fiscal_year, spec.report_kind, spec.document_version, file_sha)
