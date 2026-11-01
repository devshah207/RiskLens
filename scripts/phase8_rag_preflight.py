#!/usr/bin/env python3
"""Read-only, zero-Groq check of original Phase 7 sources from Phase 8 UI.

Run from original repo root: python scripts/phase8_rag_preflight.py
The synthetic tests execute the same code on fictional filings, but only this
script on the owner's computer can validate original private sources.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.filing_research import ResearchPaths, ask_saved_sources, source_readiness  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    base = ResearchPaths.for_root(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=base.archive)
    parser.add_argument('--index', type=Path, default=base.index)
    parser.add_argument('--banks', type=Path, default=base.banks)
    parser.add_argument('--pdfs', type=Path, default=base.pdfs)
    parser.add_argument('--pdf-offline', action='store_true',
                        help='Also use original PDF retrieval, without Groq or network requests')
    a = parser.parse_args(argv)
    paths = ResearchPaths(a.archive.expanduser(), a.index.expanduser(),
                          a.banks.expanduser(), a.pdfs.expanduser())
    state = source_readiness(paths)
    print('RiskLens Phase 8 original-source check. READ ONLY; Groq DISABLED.')
    print('XBRL 8-filing ZIP:', 'present' if state['xbrl_archive_present'] else 'MISSING')
    print('Existing PDF index:', 'present' if state['pdf_index_present'] else 'MISSING')
    print('Three fact banks:', 'present' if state['bank_files_present'] else 'MISSING')
    print('Original PDF folder:', 'present' if state['pdf_directory_present'] else 'MISSING')
    if state['missing_bank_filenames']:
        print('Missing bank filenames:', ', '.join(state['missing_bank_filenames']))
    if not state['xbrl_archive_present']:
        print('NOT TESTED: original XML archive is not at the configured path.')
        return 2

    direct = ask_saved_sources(
        'What was Infosys consolidated revenue from operations for FY2025, in crore?',
        ticker='INFY', scope='consolidated', year=2025, paths=paths,
        provider='offline', groq_allowed=False,
    )
    print('Actual original XBRL route:', direct['route'], '| status:', direct['status'])
    if direct['route'] != 'xbrl_exact_fact':
        print('Original XBRL direct match did not succeed; see the reported status in Streamlit.')
        return 1
    print('Source citations:', len(direct['citations']))

    if a.pdf_offline:
        pdf = ask_saved_sources(
            'Which commodities does Reliance identify as subject to price volatility in FY2025?',
            ticker='RIL', year=2025, paths=paths, provider='offline', groq_allowed=False,
        )
        print('Original PDF fallback:', pdf['route'], '| status:', pdf['status'])
        print('Candidate retrieved pages (NOT certified citations):',
              len(pdf.get('retrieved_physical_pages', [])))
        if pdf['route'] in {'pdf_index_missing', 'missing_original_phase7_banks',
                            'missing_original_report_directory',
                            'research_adapter_unavailable'}:
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
