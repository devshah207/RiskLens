#!/usr/bin/env python3
"""No-Groq real-index smoke for an explicitly defined HDFC financial term."""
from __future__ import annotations

from pathlib import Path
import json

from risklens.research.index import ResearchStore
from risklens.research.llm_guided_v7 import run_query

DB = Path('data/processed/research/real_acceptance/real_index.sqlite3')
QUESTION = 'How is HDFC Bank cost of deposits defined?'


def main() -> None:
    if not DB.is_file():
        raise SystemExit('FAIL: Original Phase 7 SQLite index missing')
    with ResearchStore(DB) as store:
        if not store.list_documents():
            raise SystemExit('FAIL: Original research index contains no documents')
        result = run_query(store, None, QUESTION, provider='offline')
    citations = result.get('citations', [])
    if not (result.get('status') == 'answered'
            and result.get('route') == 'explicit_original_pdf_definition'
            and 'interest expense on deposits' in result.get('answer', '').lower()
            and len(citations) == 1
            and citations[0].get('pdf_page') == 399
            and citations[0].get('ticker') == 'HDFCBANK'):
        raise SystemExit('FAIL: Exact original-PDF definition smoke did not pass:\n'
                         + json.dumps(result, indent=2))
    print('PASS: Real index + original HDFC PDF definition verified; '
          'no Groq or embedding calls required.')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
