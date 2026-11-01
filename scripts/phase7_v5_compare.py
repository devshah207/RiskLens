"""Compare V3, V4 and combined V5 on IDENTICAL real-report questions.

OFFLINE result measures deterministic financial-fact routing only, NOT full
Groq RAG accuracy. Gold answers are read only in this evaluator, never sent to
any query function. Provider failures never count as successful refusals.

Optional --live-v5 probes bounded previously unresolved questions through the
live combined RAG path, writing resumable checkpoints; no automatic retries.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import time

from risklens.research.combined_v5 import choose_fact_answer, explicit_segments, run_query
from risklens.research.financial_facts import read_facts, fact_answer
from risklens.research.financial_scope_v4 import automatic_fact_answer
from risklens.research.financial_auto_v4 import load_auto_bank, confirm_original_table_rows

DEFAULT_BASE = Path('data/processed/research/real_acceptance/phase7_completion')


def load_questions(paths):
    items=[]
    for path in paths:
        doc=json.loads(Path(path).read_text())
        for row in doc['questions']:
            if any(x['id']==row['id'] for x in items):
                raise ValueError('Duplicate benchmark ID')
            items.append({**row, '_suite': 'numeric_24' if row['id'].startswith('V4_') else 'development_35'})
    return items


def baseline_v3(case, curated, shas):
    result = fact_answer(case['question'],curated,ticker=case['ticker'],eligible_sha256=shas)
    return {**result, 'route':'v3_curated_json'} if result else None


def baseline_v4(case, curated, auto, shas, pdf_dir):
    result=baseline_v3(case,curated,shas)
    if result and result['status']=='answered':
        return result
    auto_result=automatic_fact_answer(case['question'],auto,ticker=case['ticker'],source_hashes=shas)
    if not auto_result:
        return result
    ids={x['fact_id'] for x in auto_result['citations']}
    facts=[x for x in auto if x['fact_id'] in ids]
    try:
        confirm_original_table_rows(facts,pdf_dir=pdf_dir)
    except Exception:
        return result
    return {**auto_result,'route':'v4_native_table_json'}


def _values_in_text(text):
    found=[]
    for x in re.findall(r'(?<![a-zA-Z0-9])\d[\d,]*(?:\.\d+)?(?![a-zA-Z0-9])',text):
        try:
            found.append(Decimal(x.replace(',', '')))
        except InvalidOperation:
            continue
    return found


def grade(case,result):
    if result is None:
        return {'status':'rag_fallback_not_run','source':'fallback', 'numeric_match':None,'scope_match':None,'requested_unit_match':None,'citation_hash_match':None,'verified_numeric_answer':False}
    status=result.get('status','missing')
    out={'status':status,'source':result.get('route','unknown'),
         'numeric_match':None,'scope_match':None,'requested_unit_match':None,
         'citation_hash_match':None,'verified_numeric_answer':False}
    if not case['answerable']:
        out['correct_safe_refusal']=status=='abstained'
        return out
    if 'metric_value' not in case:
        # Old development prompts can require multiple metrics or prose.
        # We intentionally do not fabricate a numeric ground-truth score.
        return out
    out['numeric_match']=Decimal(case['metric_value']) in _values_in_text(result.get('answer',''))
    required=case.get('unit')
    answer=result.get('answer','').lower()
    if required=='thousand_rupees':
        out['requested_unit_match']='thousand' in answer and 'crore' not in answer
    elif required=='percent':
        out['requested_unit_match']='%' in answer or 'percent' in answer
    elif required=='crore':
        out['requested_unit_match']='crore' in answer
    elif required=='rupees':
        out['requested_unit_match']='₹' in answer or 'rupee' in answer
    else:
        out['requested_unit_match']=True
    citations=result.get('citations',[])
    out['citation_hash_match']=any(c.get('file_sha256')==case.get('source_sha256') for c in citations)
    wanted_segments=explicit_segments(case['question'])
    cited_segments=[str(c.get('segment', '')).lower() for c in citations if c.get('fact_role')!='definition']
    out['scope_match']=(not wanted_segments or bool(cited_segments) and
                        all(s in wanted_segments for s in cited_segments))
    out['verified_numeric_answer']=(status=='answered' and out['numeric_match'] and
                                    out['requested_unit_match'] and out['citation_hash_match'] and
                                    out['scope_match'])
    return out


def compare(questions, curated, auto, pdf_dir, *, verify_sources=True):
    shas={f.file_sha256 for f in curated} | {f['file_sha256'] for f in auto}
    rows=[]
    for case in questions:
        responses={
            'v3':baseline_v3(case,curated,shas),
            'v4':baseline_v4(case,curated,auto,shas,pdf_dir),
            'combined_v5':choose_fact_answer(case['question'],curated,auto,
                                               ticker=case['ticker'], eligible_sha256=shas,
                                               pdf_dir=pdf_dir,verify_sources=verify_sources),
        }
        grades={version:grade(case,r) for version,r in responses.items()}
        rows.append({'id':case['id'],'suite':case['_suite'],'question':case['question'],
                     'answerable_label':case['answerable'],
                     'grades':grades,
                     'answers':{version:({'answer':r.get('answer'),'status':r.get('status'),
                                           'route':r.get('route'), 'citations':r.get('citations',[]),
                                           'verification_status':r.get('verification_status')}
                                          if r else None) for version,r in responses.items()}})
    summary={}
    for suite in sorted({r['suite'] for r in rows}):
        data=[r for r in rows if r['suite']==suite]
        summary[suite]={}
        for v in ('v3','v4','combined_v5'):
            g=[r['grades'][v] for r in data]
            summary[suite][v]={
                'cases':len(data),
                'structured_answered':sum(r['status']=='answered' for r in g),
                'structured_needs_review':sum(r['status']=='needs_review' for r in g),
                'structured_uncovered':sum(r['status']=='rag_fallback_not_run' for r in g),
                'correct_future_refusals':sum(r.get('correct_safe_refusal',False) for r in g),
                'verified_numeric_answers_on_numeric_suite':sum(r['verified_numeric_answer'] for r in g),
            }
    return {'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'STRUCTURED-ONLY apples-to-apples offline comparison; no Groq calls; original V4 live results not replayed',
            'limitations':['These are reused development questions, not an untouched benchmark.',
                           'Table-cell revalidation does not prove human-certified financial semantics.',
                           'No benchmark measures semantic RAG accuracy or unavailable-provider recovery offline.',
                           'Numeric correctness is a diagnostic including unit/segment/source-hash checks; not a substitute for manual page audit.'],
            'summary':summary,'cases':rows}


def run_live_v5(questions, *, outfile:Path, db:Path, auto_path:Path, curated_path:Path,
                pdf_dir:Path, max_live_fallbacks:int=4, pause_seconds:float=20.0):
    """Bounded resume-safe V5 fallback evaluation, avoiding Groq calls for facts."""
    from risklens.research.index import ResearchStore
    from risklens.research.embeddings import SentenceEmbedder, create_embedder
    class LazyEmbedder:
        name=SentenceEmbedder.name
        dimension=SentenceEmbedder.dimension
        def __init__(self): self.actual=None
        def encode(self,texts):
            if self.actual is None:
                self.actual=create_embedder('sentence')
            return self.actual.encode(texts)
    output={'schema_version':1,'cases':{},'notice':'Live answers: review actual source citations; provider failures are unscored'}
    if outfile.is_file():
        output=json.loads(outfile.read_text())
    count=0
    with ResearchStore(db) as store:
        if not store.list_documents():
            raise RuntimeError('Index has no documents; refuse live evaluation')
        lazy=LazyEmbedder()
        for q in questions:
            if q['id'] in output['cases']:
                continue
            if count>=max_live_fallbacks:
                break
            # Only use live quota when offline route does not already answer.
            facts=read_facts(curated_path)
            auto=load_auto_bank(auto_path)
            direct=choose_fact_answer(q['question'],facts,auto,ticker=q['ticker'],
                                      pdf_dir=pdf_dir,verify_sources=True)
            if direct and direct['status'] in ('answered','abstained','needs_review'):
                continue
            if count:
                time.sleep(pause_seconds)
            result=run_query(store,lazy,q['question'],provider='groq',ticker=q['ticker'],
                             curated_path=curated_path,automatic_path=auto_path,pdf_dir=pdf_dir,
                             verify_financial=True)
            output['cases'][q['id']]={'question':q['question'],'result':result}
            outfile.parent.mkdir(parents=True,exist_ok=True)
            outfile.write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
            count+=1
            print(q['id'],result.get('route'),result['status'],flush=True)
            if result['status']=='provider_unavailable' or result.get('verification_status')=='verifier_unavailable':
                print('API unavailable; stopping instead of burning further quota',flush=True)
                break
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v3-bank',type=Path,default=DEFAULT_BASE/'financial_facts_v3.json')
    p.add_argument('--v4-bank',type=Path,default=DEFAULT_BASE/'financial_facts_auto_v4.json')
    p.add_argument('--pdf-dir',type=Path,default=Path('data/raw/research_reports'))
    p.add_argument('--gold24',type=Path,default=Path('docs/phase7_heldout_v4.json'))
    p.add_argument('--gold20',type=Path,default=Path('docs/phase7_independent_source_anchors.json'))
    p.add_argument('--gold15',type=Path,default=Path('docs/phase7_new_challenge_set_v1.json'))
    p.add_argument('--output',type=Path,default=DEFAULT_BASE/'phase7_v5_comparison_offline.json')
    p.add_argument('--live-v5',action='store_true',help='Run a bounded Groq evaluation ONLY for unresolved V5 fallback questions')
    p.add_argument('--max-live-fallbacks',type=int,default=4)
    p.add_argument('--pause-seconds',type=float,default=20.0)
    p.add_argument('--db',type=Path,default=Path('data/processed/research/real_acceptance/real_index.sqlite3'))
    a=p.parse_args()
    questions=load_questions([a.gold24,a.gold20,a.gold15])
    curated=read_facts(a.v3_bank)
    auto=load_auto_bank(a.v4_bank)
    report=compare(questions,curated,auto,a.pdf_dir)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report['summary'],indent=2),flush=True)
    print('Structured-only comparison saved:',a.output,flush=True)
    if a.live_v5:
        path=a.output.with_name('phase7_v5_live_fallback_checkpoint.json')
        run_live_v5(questions,outfile=path,db=a.db,auto_path=a.v4_bank,curated_path=a.v3_bank,
                    pdf_dir=a.pdf_dir,max_live_fallbacks=a.max_live_fallbacks,
                    pause_seconds=a.pause_seconds)
        print('Live checkpoints:',path,flush=True)

if __name__=='__main__':
    main()
