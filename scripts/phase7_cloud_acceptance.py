"""Phase 7: real-PDF OCR, Pinecone and Groq probes; no automatic gold certifications.

User-owned account calls. Never prints API keys. Uses existing 384-D MiniLM
vectors and previously downloaded genuine financial reports. Rerunnable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('data/processed/research/real_acceptance')
INDEX = 'risklens-minilm384'
NAMESPACE = 'risklens-research'
QUARANTINED = {'RIL': {73}, 'HDFCBANK': {295}, 'INFY': {184}, 'ICICIBANK': {29}}
MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def _field(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def index_contract(pc):
    """Reject wrong model, dimension, metric or not-ready cloud index."""
    data = pc.indexes.describe(INDEX)
    fields = _field(_field(data, 'schema'), 'fields')
    if not fields:
        raise RuntimeError('Cloud index has no schema fields; refusing upload')
    embed = _field(fields, 'embedding')
    metric = str(_field(embed, 'metric', '')).lower().split('.')[-1]
    dim = _field(embed, 'dimension')
    if int(dim or -1) != 384 or metric != 'cosine' or _field(data, 'embed'):
        raise RuntimeError('Cloud index != local MiniLM 384D cosine contract')
    status = _field(data, 'status')
    if _field(status, 'ready') is not True:
        raise RuntimeError('Cloud index is not ready')
    return pc.index(INDEX)


def _ensure_local(store):
    config = dict(store.conn.execute('SELECT key,value FROM config'))
    if config.get('dimension') != '384' or config.get('embedding_name') != MODEL:
        raise RuntimeError('Local SQLite vector-model provenance mismatch')
    docs = store.list_documents()
    if {d.spec.ticker for d in docs} != set(QUARANTINED):
        raise RuntimeError('Expected exactly 4 verified-source issuer IDs')
    if sum(d.pages for d in docs) != 1165:
        raise RuntimeError('Original 1165-page PDF corpus changed; review first')
    for doc in docs:
        path = Path(doc.spec.local_path)
        if not path.is_file():
            raise RuntimeError(f'Original report missing: {path.name}')
        from hashlib import sha256
        if sha256(path.read_bytes()).hexdigest() != doc.sha256:
            raise RuntimeError(f'Original report fingerprint mismatch: {doc.spec.ticker}')
    return docs


def select_upload_rows(store, docs):
    """Only persist reviewed-by-policy pages; isolate 4 known uncertain pages."""
    from risklens.research.index import _unpack
    batches = []
    excluded = {}
    for doc in docs:
        rows = store.conn.execute(
            'SELECT chunk_id,page_no,vector,review_required FROM chunks WHERE doc_id=? ORDER BY chunk_id',
            (doc.doc_id,)
        )
        skipped = 0
        for row in rows:
            if row['review_required'] or int(row['page_no']) in QUARANTINED[doc.spec.ticker]:
                skipped += 1
                continue
            vector = list(_unpack(row['vector']))
            if len(vector) != 384:
                raise RuntimeError(f'Corrupt vector: {row["chunk_id"]}')
            batches.append({
                '_id': row['chunk_id'], 'embedding': vector,
                'ticker': doc.spec.ticker, 'fiscal_year': doc.spec.fiscal_year,
                'published_on': doc.spec.published_on,
                'document_id': doc.doc_id, 'pdf_sha256': doc.sha256,
                'physical_pdf_page': int(row['page_no']), 'embedding_model': MODEL,
            })
        excluded[doc.spec.ticker] = skipped
    if len(batches) < 1500:
        raise RuntimeError('Unexpectedly few eligible source chunks')
    return batches, excluded


def groq_retry(call, *, attempts=3):
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:
            if type(exc).__name__ not in {'RateLimitError', 'InternalServerError', 'APIConnectionError', 'APITimeoutError'} or attempt == attempts - 1:
                raise
            time.sleep([8, 25, 65][attempt])


def audit_ocr(run_vision):
    cmd = [sys.executable, 'scripts/phase7_review_flagged.py']
    if run_vision:
        cmd += ['--groq']
    print('RUNNING 4-PAGE AUDIT; unverified output is NEVER added to SQLite', flush=True)
    subprocess.run(cmd, check=True)
    return str(BASE / 'flagged_page_quality' / 'SUMMARY_UNVERIFIED.json')


def cloud_probe(do_upload, do_answers, candidates_path):
    from pinecone import Pinecone, DenseVectorQuery
    from risklens.research.config import load_project_env
    from risklens.research.index import ResearchStore
    from risklens.research.embeddings import SentenceEmbedder
    from risklens.research.retrieval import search
    from risklens.research.answer import respond

    load_project_env()
    if not os.getenv('PINECONE_API_KEY') or not os.getenv('GROQ_API_KEY'):
        raise RuntimeError('Keys not found in project .env; do not print or paste them')
    db = BASE / 'real_index.sqlite3'
    if not db.is_file():
        raise RuntimeError('Real SQLite research index missing')

    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    remote = index_contract(pc)
    if not hasattr(remote, 'documents'):
        raise RuntimeError('Pinecone index does not expose SDK document API')

    with ResearchStore(db) as store:
        docs = _ensure_local(store)
        batch, excluded = select_upload_rows(store, docs)
        print('LOCAL VECTORS:', len(batch), 'eligible, excluded for unresolved page QA:', excluded, flush=True)
        if do_upload:
            # Upsert is idempotent, so intermittent network failure can be retried without duplicates.
            for start in range(0, len(batch), 50):
                subset = batch[start:start + 50]
                groq_retry(lambda: remote.documents.upsert(namespace=NAMESPACE, documents=subset))
                if start % 250 == 0:
                    print('PINECONE UPLOADED:', start + len(subset), '/', len(batch), flush=True)
        else:
            print('PINECONE UPLOAD DISABLED')

        embedder = SentenceEmbedder()
        store.validate_embedder(embedder)
        q = 'What financial risk factors did HDFC Bank report for FY2024-25?'
        vector = embedder.encode([q])[0]
        valid_ids = {item['_id'] for item in batch}
        ids = []
        for attempt in range(5):
            result = groq_retry(lambda: remote.documents.search(
                namespace=NAMESPACE, top_k=10,
                score_by=[DenseVectorQuery(field='embedding', values=vector)],
                include_fields=['ticker', 'document_id', 'published_on']))
            ids = [str(_field(m, 'id', _field(m, '_id'))) for m in _field(result, 'matches', [])]
            if ids and all(i in valid_ids and store.get_chunk(i) for i in ids):
                break
            # Serverless indexes can be eventually consistent after a bulk upsert.
            if attempt < 4:
                time.sleep([3, 8, 15, 25][attempt])
        else:
            raise RuntimeError('Pinecone did not return eligible local source IDs after bounded visibility retries')
        print('PINECONE QUERY PASS:', len(ids), 'IDs matched local verified-by-policy chunks')

        questions = json.loads(Path(candidates_path).read_text(encoding='utf-8'))['questions']
        if len(questions) != 20:
            raise RuntimeError('Expected original 20 unverified benchmark candidates')
        dossier = []
        calls = 0
        for question in questions:
            hits = search(store, question['question'], embedder, ticker=question['ticker'],
                          fiscal_year='2024-25', top_k=5, mode='hybrid')  # local hybrid benchmark candidate discovery
            # Suppress the 4 flagged physical pages from candidate evidence until source verification.
            hits = [h for h in hits if h.chunk.page_no not in QUARANTINED[h.document.spec.ticker]]
            record = {
                'id': question['id'], 'ticker': question['ticker'],
                'question': question['question'], 'candidate_answerable_UNVERIFIED': question['answerable'],
                'evidence': [{'doc_id': h.document.doc_id, 'pdf_page': h.chunk.page_no,
                              'chunk_id': h.chunk.chunk_id, 'excerpt': h.chunk.text[:850]}
                             for h in hits],
                'gold_label_status': 'NEEDS_MANUAL_VERIFICATION',
            }
            if do_answers and question['id'] in {'RIL01', 'HDFC01', 'INFY01', 'ICICI01', 'ICICI05'}:
                reply = groq_retry(lambda: respond(question['question'], hits,
                                                    provider='groq', model='openai/gpt-oss-120b'))
                record['groq_probe'] = reply.to_dict()
                record['groq_answer_accuracy'] = 'UNVERIFIED_HUMAN_REVIEW_REQUIRED'
                calls += 1
            dossier.append(record)
            print('RETRIEVED:', question['id'], len(hits), 'candidate citations')
        out = BASE / 'live_cloud_acceptance'
        out.mkdir(parents=True, exist_ok=True)
        report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'index': INDEX,
                  'namespace': NAMESPACE, 'embedding_name': MODEL, 'uploaded': do_upload,
                  'count_upload_eligible': len(batch), 'excluded_flagged_chunks': excluded,
                  'pinecone_query_matched': len(ids), 'groq_probe_count': calls,
                  'questions_retrieved': len(dossier), 'questions_verified': 0,
                  'genuine_retrieval_metrics': None,
                  'financial_answer_accuracy': None,
                  'limitations': ['Four suspicious PDF pages require visual verification',
                    '20 candidate questions are not independent verified gold labels',
                    'No automatic financial factuality is claimed'],
                  'items': dossier}
        path = out / 'cloud_qa_UNVERIFIED.json'
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        md = ['# Phase 7: 20-case financial evidence review', '',
              '**UNVERIFIED candidate retrieval. No gold accuracy inferred.**', '']
        for item in dossier:
            md.extend([f"## {item['id']} — {item['ticker']}", '',
                       f"**Question:** {item['question']}", '',
                       '**Human reviewer:** __________________  **Review date:** __________', '',
                       '**Verified correct evidence pages / quotes / answer:**', '',
                       '_____________________________________________', ''])
            for e in item['evidence']:
                md.extend([f"### PDF page {e['pdf_page']} · chunk `{e['chunk_id']}`", '',
                           '```text', e['excerpt'], '```', ''])
            if 'groq_probe' in item:
                md.extend(['**Groq draft (UNVERIFIED):**', '',
                           '```json', json.dumps(item['groq_probe'], indent=2, ensure_ascii=False), '```', ''])
        (out / 'REVIEW_20_QUESTIONS_UNVERIFIED.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
        print('SAVED CLOUD QA DOSSIER:', path)
        print('REVIEW SHEET:', out / 'REVIEW_20_QUESTIONS_UNVERIFIED.md')
        # Compact shareable QA evidence only: never include complete vendor PDFs, .env,
        # local vector database, model weights, or API credentials.
        from zipfile import ZipFile, ZIP_DEFLATED
        bundle = BASE / 'phase7_QA_review_evidence.zip'
        with ZipFile(bundle, 'w', ZIP_DEFLATED) as z:
            for evidence in sorted((BASE / 'flagged_page_quality').rglob('*')):
                if evidence.is_file() and evidence.suffix.lower() in {'.png', '.txt', '.json'}:
                    z.write(evidence, str(evidence.relative_to(BASE)))
            z.write(path, str(path.relative_to(BASE)))
            sheet = out / 'REVIEW_20_QUESTIONS_UNVERIFIED.md'
            z.write(sheet, str(sheet.relative_to(BASE)))
        print('QA REVIEW BUNDLE (no API secrets or full PDFs):', bundle)
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-groq', action='store_true', help='Send 4 specific original PDF page images to Groq Vision')
    parser.add_argument('--upload', action='store_true', help='Upload nonflagged source vectors to Pinecone')
    parser.add_argument('--answer-groq', action='store_true', help='Send five selected retrieved question contexts to Groq')
    parser.add_argument('--candidates', default='docs/phase7_gold_question_candidates.json')
    args = parser.parse_args()
    audit_ocr(args.audit_groq)
    cloud_probe(args.upload, args.answer_groq, args.candidates)
    print('COMPLETE: cloud functionality probed; human evidence verification remains required')

if __name__ == '__main__':
    main()
