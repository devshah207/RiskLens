"""Provenance-first SQLite/FTS5 store, plus optional Pinecone vector mirror."""
from __future__ import annotations

from .config import load_project_env

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import struct

from .contracts import Chunk, Document, Page, ResearchError, SourceSpec, valid_iso_date

_WORD = re.compile(r"[a-zA-Z0-9]+")


def _pack(vector: list[float]) -> bytes:
    if not vector or any(not math.isfinite(float(v)) for v in vector):
        raise ResearchError("Embedding must be finite, nonempty vector")
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack(blob: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(blob)//4}f", blob)


class ResearchStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS config(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS documents(
          doc_id TEXT PRIMARY KEY, identity TEXT NOT NULL UNIQUE,
          sha256 TEXT NOT NULL, pages INTEGER NOT NULL, parser TEXT NOT NULL,
          spec_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS pages(
          doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
          page_no INTEGER NOT NULL, text TEXT NOT NULL, method TEXT NOT NULL,
          flags_json TEXT NOT NULL, review_required INTEGER NOT NULL,
          PRIMARY KEY(doc_id,page_no));
        CREATE TABLE IF NOT EXISTS chunks(
          chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
          page_no INTEGER NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL,
          text TEXT NOT NULL, section TEXT NOT NULL, review_required INTEGER NOT NULL,
          vector BLOB NOT NULL);
        CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, content);
        """)
        self.conn.commit()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.conn.close()

    @staticmethod
    def _identity(doc: Document) -> str:
        s = doc.spec
        return json.dumps([s.ticker, s.fiscal_year, s.report_kind, s.document_version])

    def _embedding_contract(self, name: str, dimension: int) -> None:
        existing = {row["key"]: row["value"] for row in self.conn.execute("SELECT * FROM config")}
        if existing and (existing.get("embedding_name") != name or existing.get("dimension") != str(dimension)):
            raise ResearchError("Embedding model/dimension mismatch: use a NEW index DB instead of mixing vectors")
        self.conn.execute("INSERT OR IGNORE INTO config VALUES ('embedding_name', ?)", (name,))
        self.conn.execute("INSERT OR IGNORE INTO config VALUES ('dimension', ?)", (str(dimension),))

    def validate_embedder(self, embedder) -> None:
        row = self.conn.execute("SELECT value FROM config WHERE key='embedding_name'").fetchone()
        if row and row["value"] != embedder.name:
            raise ResearchError(f"Index was built for {row['value']}, not {embedder.name}")
        row = self.conn.execute("SELECT value FROM config WHERE key='dimension'").fetchone()
        if row and int(row["value"]) != embedder.dimension:
            raise ResearchError("Embedding dimension does not match existing index")

    def insert(self, document: Document, pages: list[Page], chunks: list[Chunk], embedder) -> dict:
        """Idempotent on exact file; version replacement removes stale local vectors atomically."""
        if not pages or any(p.doc_id != document.doc_id or not 1 <= p.page_no <= document.pages for p in pages):
            raise ResearchError("Pages violate document provenance")
        if any(c.doc_id != document.doc_id for c in chunks):
            raise ResearchError("Chunks must belong to this source document")
        if len({c.chunk_id for c in chunks}) != len(chunks):
            raise ResearchError("Duplicate chunk identifiers")
        self.validate_embedder(embedder)
        old = self.conn.execute("SELECT doc_id, sha256, parser, spec_json FROM documents WHERE identity=?",
                                (self._identity(document),)).fetchone()
        if old and old["sha256"] == document.sha256 and old["parser"] == document.parser:
            existing_pages = self.conn.execute(
                "SELECT page_no, text, method, review_required FROM pages WHERE doc_id=? ORDER BY page_no",
                (document.doc_id,),
            ).fetchall()
            unchanged_pages = len(existing_pages) == len(pages) and all(
                (r["page_no"], r["text"], r["method"], bool(r["review_required"])) ==
                (p.page_no, p.text, p.method, p.review_required)
                for r, p in zip(existing_pages, sorted(pages, key=lambda v: v.page_no))
            )
            if unchanged_pages:
                n_chunks = self.conn.execute("SELECT COUNT(*) FROM chunks WHERE doc_id=?", (document.doc_id,)).fetchone()[0]
                new_spec = json.dumps(asdict(document.spec))
                if old["spec_json"] != new_spec:
                    with self.conn:
                        self.conn.execute("UPDATE documents SET spec_json=? WHERE doc_id=?", (new_spec, document.doc_id))
                    return {"status": "metadata_updated", "doc_id": document.doc_id, "chunks": n_chunks}
                return {"status": "unchanged", "doc_id": document.doc_id, "chunks": n_chunks}
        vectors = embedder.encode([c.text for c in chunks])
        if len(vectors) != len(chunks) or any(len(v) != embedder.dimension for v in vectors):
            raise ResearchError("Embedding output dimension/count mismatch")
        with self.conn:
            self._embedding_contract(embedder.name, embedder.dimension)
            if old:
                self.conn.execute("DELETE FROM chunks_fts WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id=?)", (old["doc_id"],))
                self.conn.execute("DELETE FROM documents WHERE doc_id=?", (old["doc_id"],))
            self.conn.execute("INSERT INTO documents VALUES (?,?,?,?,?,?)", (
                document.doc_id, self._identity(document), document.sha256,
                document.pages, document.parser, json.dumps(asdict(document.spec))))
            self.conn.executemany("INSERT INTO pages VALUES (?,?,?,?,?,?)", [
                (p.doc_id, p.page_no, p.text, p.method, json.dumps(list(p.quality_flags)), int(p.review_required))
                for p in pages])
            self.conn.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)", [
                (c.chunk_id, c.doc_id, c.page_no, c.start, c.end, c.text, c.section,
                 int(c.review_required), _pack(v)) for c, v in zip(chunks, vectors)])
            self.conn.executemany("INSERT INTO chunks_fts(chunk_id,content) VALUES (?,?)", [
                (c.chunk_id, c.text) for c in chunks])
        return {"status": "replaced" if old else "indexed", "doc_id": document.doc_id,
                "chunks": len(chunks)}

    def list_documents(self) -> list[Document]:
        return [self._document(r) for r in self.conn.execute("SELECT * FROM documents ORDER BY doc_id")]

    @staticmethod
    def _document(row) -> Document:
        return Document(doc_id=row["doc_id"], spec=SourceSpec.from_dict(json.loads(row["spec_json"])),
                        sha256=row["sha256"], pages=int(row["pages"]), parser=row["parser"])

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        row = self.conn.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        return self._chunk(row) if row else None

    @staticmethod
    def _chunk(row) -> Chunk:
        return Chunk(chunk_id=row["chunk_id"], doc_id=row["doc_id"],
                     page_no=int(row["page_no"]), start=int(row["start"]), end=int(row["end"]),
                     text=row["text"], section=row["section"], review_required=bool(row["review_required"]))

    def page_text(self, doc_id: str, page_no: int) -> str:
        row = self.conn.execute("SELECT text FROM pages WHERE doc_id=? AND page_no=?", (doc_id, page_no)).fetchone()
        if not row:
            raise ResearchError("Source page not present")
        return row["text"]

    def _filter(self, ticker: str | None, fiscal_year: str | None, as_of: str | None) -> tuple[str, list[str]]:
        where = ["c.review_required=0"]
        args: list[str] = []
        if ticker:
            where.append("json_extract(d.spec_json, '$.ticker')=?")
            args.append(ticker.upper())
        if fiscal_year:
            where.append("json_extract(d.spec_json, '$.fiscal_year')=?")
            args.append(fiscal_year)
        if as_of:
            valid_iso_date(as_of)
            where.append("json_extract(d.spec_json, '$.published_on')<=?")
            args.append(as_of)
        return " AND ".join(where), args

    def candidates(self, *, ticker: str | None = None, fiscal_year: str | None = None,
                   as_of: str | None = None) -> list[tuple[Chunk, Document, tuple[float, ...]]]:
        where, args = self._filter(ticker, fiscal_year, as_of)
        rows = self.conn.execute(
            "SELECT c.*, d.identity, d.sha256, d.pages, d.parser, d.spec_json "
            "FROM chunks c JOIN documents d ON c.doc_id=d.doc_id WHERE " + where, args).fetchall()
        return [(self._chunk(r), self._document(r), _unpack(r["vector"])) for r in rows]

    def lexical(self, query: str, *, ticker: str | None = None, fiscal_year: str | None = None,
                as_of: str | None = None, top_k: int = 20) -> list[tuple[str, float]]:
        tokens = list(dict.fromkeys(t.lower() for t in _WORD.findall(query) if len(t) >= 2))[:24]
        if not tokens:
            return []
        expr = " OR ".join('"' + t + '"' for t in tokens)  # alphanumeric-only parser, no FTS injection
        where, args = self._filter(ticker, fiscal_year, as_of)
        rows = self.conn.execute(
            "SELECT c.chunk_id AS chunk_id, bm25(chunks_fts) AS rank FROM chunks_fts "
            "JOIN chunks c ON c.chunk_id=chunks_fts.chunk_id "
            "JOIN documents d ON c.doc_id=d.doc_id "
            "WHERE chunks_fts MATCH ? AND " + where + " ORDER BY rank LIMIT ?",
            [expr, *args, max(1, top_k)],
        ).fetchall()
        return [(r["chunk_id"], float(r["rank"])) for r in rows]


def _pinecone_index(index_name: str, api_key: str):
    """Validate named cloud schema *before* any document upsert or query."""
    from pinecone import Pinecone
    if index_name != "risklens-minilm384":
        raise ResearchError("Only the audited 384D MiniLM cloud index is supported")
    pc = Pinecone(api_key=api_key)
    info = pc.indexes.describe(index_name)
    def field(obj, name, default=None):
        return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)
    fields = field(field(info, "schema"), "fields", {})
    embedding = field(fields, "embedding")
    if (int(field(embedding, "dimension", -1)) != 384 or
            str(field(embedding, "metric", "")).lower().split(".")[-1] != "cosine" or
            field(info, "embed")):
        raise ResearchError("Remote index schema does not match 384D cosine MiniLM")
    if field(field(info, "status"), "ready") is not True:
        raise ResearchError("Remote Pinecone index is not ready")
    index = pc.index(index_name)
    if not hasattr(index, "documents"):
        raise ResearchError("SDK v10 document-index interface is required")
    return index


def mirror_pinecone(document: Document, chunks: list[Chunk], embedder, *, index_name: str,
                    namespace: str = "risklens-research") -> int:
    """Opt-in mirroring into schema-verified Pinecone SDK v10 index.

    Default ingestion must not silently upload uncertain OCR pages. The dedicated
    cloud acceptance step additionally filters the four outstanding QA pages.
    """
    load_project_env()
    if not os.getenv("PINECONE_API_KEY"):
        raise ResearchError("PINECONE_API_KEY not configured in private .env")
    if embedder.name != "sentence-transformers/all-MiniLM-L6-v2" or embedder.dimension != 384:
        raise ResearchError("Cloud mirror accepts MiniLM 384D only")
    if any(c.review_required for c in chunks):
        raise ResearchError("Cannot mirror unverified source chunks")
    index = _pinecone_index(index_name, os.environ["PINECONE_API_KEY"])
    vectors = embedder.encode([c.text for c in chunks])
    if len(vectors) != len(chunks) or any(len(v) != 384 for v in vectors):
        raise ResearchError("Pinecone vector dimension/count mismatch")
    for start in range(0, len(chunks), 50):
        batch = []
        for chunk, vec in zip(chunks[start:start + 50], vectors[start:start + 50]):
            batch.append({"_id": chunk.chunk_id, "embedding": vec,
                          "ticker": document.spec.ticker, "fiscal_year": document.spec.fiscal_year,
                          "published_on": document.spec.published_on,
                          "document_id": document.doc_id, "embedding_model": embedder.name,
                          "physical_pdf_page": chunk.page_no})
        index.documents.upsert(namespace=namespace, documents=batch)
    return len(chunks)


def pinecone_ids(query_vector: list[float], *, index_name: str,
                 namespace: str = "risklens-research", top_k: int = 40) -> list[str]:
    """Remote ID candidate generation; caller always rechecks local provenance."""
    load_project_env()
    if not os.getenv("PINECONE_API_KEY"):
        raise ResearchError("PINECONE_API_KEY not configured in private .env")
    if len(query_vector) != 384:
        raise ResearchError("Pinecone query must contain 384 values")
    from pinecone import DenseVectorQuery
    index = _pinecone_index(index_name, os.environ["PINECONE_API_KEY"])
    result = index.documents.search(namespace=namespace, top_k=top_k,
        score_by=[DenseVectorQuery(field="embedding", values=query_vector)],
        include_fields=["ticker", "published_on", "document_id"])
    return [item.id for item in result.matches]
