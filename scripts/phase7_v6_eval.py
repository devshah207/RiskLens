"""Evaluate existing 24 V4 questions with V5 and the NEW spatial card route.

This script never calls Groq, trains models, alters V3/V4/V5 banks, or uses
question IDs or expected answers to construct financial facts.  It reports
only historical development-set routing/correctness diagnostics.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import json
import argparse

from risklens.research.layout_panels_v6 import answer_from_layout_panels, read_panel_bank

ROOT = Path('data/processed/research/real_acceptance/phase7_completion')


def _answer_matched(case: dict, reply: dict | None) -> bool:
    if reply is None or reply.get('status') != 'answered' or not case.get('answerable'):
        return False
    wanted = case.get('metric_value')
    if wanted is None:
        return False
    from re import findall
    values = []
    for raw in findall(r'(?<![a-zA-Z0-9])\d[\d,]*(?:\.\d+)?(?![a-zA-Z0-9])', reply.get('answer', '')):
        try:
            values.append(Decimal(raw.replace(',', '')))
        except Exception:
            continue
    try:
        matched = Decimal(str(wanted).replace(',', '')) in values
    except Exception:
        return False
    unit = case.get('unit', '').casefold()
    text = reply['answer'].casefold()
    if unit in {'crore', 'percent', 'rupees'}:
        is_unit = {'crore': 'crore' in text, 'percent': '%' in text,
                   'rupees': '₹' in text}[unit]
    else:
        is_unit = True
    cites = reply.get('citations', [])
    # Prior gold pages are non-exhaustive: a different page in the SAME
    # unmodified original PDF may provide stronger evidence. Query code has
    # already re-extracted that exact original page and row/word positions.
    source_match = any(c.get('file_sha256') == case['source_sha256'] and
                       isinstance(c.get('pdf_page'), int) for c in cites)
    return matched and is_unit and source_match


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', type=Path, default=Path('docs/phase7_heldout_v4.json'))
    parser.add_argument('--previous', type=Path, default=ROOT/'phase7_v5_comparison_offline.json')
    parser.add_argument('--bank', type=Path, default=ROOT/'financial_layout_panels_v6.json')
    parser.add_argument('--pdf-dir', type=Path, default=Path('data/raw/research_reports'))
    parser.add_argument('--output', type=Path, default=ROOT/'phase7_v6_layout_comparison_offline.json')
    args = parser.parse_args()
    gold = json.loads(args.gold.read_text())['questions']
    old = json.loads(args.previous.read_text())
    previous = {item['id']: item for item in old['cases']}
    panels = read_panel_bank(args.bank)
    records = []
    for case in gold:
        old_row = previous[case['id']]
        v5_grade = old_row['grades']['combined_v5']
        result = None
        if v5_grade['status'] == 'rag_fallback_not_run':
            result = answer_from_layout_panels(
                case['question'], panels, pdf_dir=args.pdf_dir,
                eligible_hashes={case['source_sha256']} if case['answerable'] else None,
                ticker=case['ticker'],
            )
        records.append({
            'id': case['id'], 'question': case['question'],
            'previous_v5_status': v5_grade['status'],
            'v6_layout_status': result['status'] if result else 'uncovered',
            'v6_layout_route': result.get('route') if result else None,
            'new_verified_numeric_answer': _answer_matched(case, result),
            'answer': result.get('answer') if result else None,
            'citations': result.get('citations', []) if result else [],
        })
    recovered = [r for r in records if r['new_verified_numeric_answer']]
    unverified = [r for r in records if r['v6_layout_status'] == 'answered'
                  and not r['new_verified_numeric_answer']]
    summary = {
        'historical_questions': len(records),
        'v5_direct_answers': sum(r['previous_v5_status'] == 'answered' for r in records),
        'v5_future_refusals': sum(r['previous_v5_status'] == 'abstained' for r in records),
        'v6_additional_source_checked_numeric_answers': len(recovered),
        'v6_new_answers_failing_reference_diagnostic': len(unverified),
        'v6_new_answer_ids': [r['id'] for r in recovered],
        'valid_alternative_source_page_ids': [
            r['id'] for r in recovered
            if next(c for c in gold if c['id'] == r['id'])['source_physical_pdf_page']
            not in [cit['pdf_page'] for cit in r['citations']]
        ],
        'combined_direct_answers_on_24': sum(r['previous_v5_status'] == 'answered' for r in records) + len(recovered),
        'combined_unresolved_on_24': sum(r['previous_v5_status'] == 'rag_fallback_not_run' and
                                         r['v6_layout_status'] == 'uncovered' for r in records),
        'note': 'Only offline development-set diagnostics. Not untouched evaluation or full Groq accuracy.',
    }
    doc = {'schema_version': 6, 'summary': summary, 'cases': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2))
    if unverified:
        raise RuntimeError('Panel route returned a numeric value not matching known source diagnostics')


if __name__ == '__main__':
    main()
