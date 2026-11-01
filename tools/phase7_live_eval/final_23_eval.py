#!/usr/bin/env python3
"""Run ONLY the 23 pending RiskLens PDF/Groq questions, resumably.

Never edits the RiskLens research code, ML models, PDF index or XBRL source.
Reads the previous offline 34-question result as a frozen routing snapshot.
No benchmark gold/reference fields are supplied to any answer function or model.

Without --live: preflight only; zero API calls.
With --live: explicit live Groq authorization required and rate-limited.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import traceback
from collections import Counter, deque
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
import zipfile
from types import SimpleNamespace

KIT = Path(__file__).resolve().parent
DEFAULT_BASELINE = KIT / 'evidence/offline_34_results.csv'
DEFAULT_QUESTIONS = Path('data/processed/research/real_acceptance/phase7_completion/v71_unseen_34/live_human_audit.csv')
DEFAULT_DB = Path('data/processed/research/real_acceptance/real_index.sqlite3')
DEFAULT_BANK = Path('data/processed/research/real_acceptance/phase7_completion')
DEFAULT_PDF = Path('data/raw/research_reports')
DEFAULT_OUT = Path('data/processed/research/xbrl_34_final_live_20261101')
EXPECTED_ROUTES = Counter({'llm_guided_rag_not_run': 23, 'xbrl_exact_fact': 4,
                           'historical_period_guard': 6, 'llm_guided_structured_fact': 1})
EXPECTED_PENDING_KINDS = Counter({'numeric': 15, 'explanation': 8})
DEFAULT_GROQ_COOLDOWN_SECONDS = 75.0  # Across every API request, including retries and restarts.



def digest(path: Path) -> str:
    h = sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def validate_inputs(questions: Path, baseline: Path) -> tuple[list[dict], list[dict]]:
    source = read_csv(questions)
    previous = read_csv(baseline)
    if len(source) != 34 or len(previous) != 34:
        raise ValueError('Expected exactly 34 original benchmark rows in both inputs')
    if len({r['id'] for r in source}) != 34 or len({r['id'] for r in previous}) != 34:
        raise ValueError('Duplicate benchmark question identifiers')
    prior = {r['id']: r for r in previous}
    for c in source:
        p = prior.get(c['id'])
        if p is None or any(c[k] != p[k] for k in ('id', 'question', 'ticker', 'kind')):
            raise ValueError('Original benchmark differs from saved baseline at ' + c['id'])
    counts = Counter(r['route'] for r in previous)
    if counts != EXPECTED_ROUTES:
        raise ValueError('Saved baseline routes changed; refuse unintended live calls: ' + repr(counts))
    pending = [c for c in source if prior[c['id']]['route'] == 'llm_guided_rag_not_run']
    if Counter(c['kind'] for c in pending) != EXPECTED_PENDING_KINDS:
        raise ValueError('Pending cases no longer contain 15 numeric and 8 explanation questions')
    return pending, previous


def check_sqlite(db: Path) -> int:
    # Read-only preflight, so missing or invalid databases cannot be created accidentally.
    uri = db.resolve().as_uri() + '?mode=ro'
    with sqlite3.connect(uri, uri=True) as conn:
        n = int(conn.execute('SELECT COUNT(*) FROM documents').fetchone()[0])
    if n < 1:
        raise ValueError('The existing PDF index has no documents')
    return n


def json_atomic(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def json_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + '\n')
        f.flush()
        os.fsync(f.fileno())


def _estimate_tokens(kwargs: dict) -> int:
    # Conservative request budget. Upper-bound completion is counted in full.
    data = json.dumps(kwargs.get('messages', []), ensure_ascii=False, default=str)
    return math.ceil(len(data) / 3.5) + int(kwargs.get('max_completion_tokens', 1024))


def _fit_evidence_request(kwargs: dict, token_limit: int) -> tuple[dict, dict]:
    """Fit PDF source excerpts within our own conservative Groq token budget.

    Changes ONLY the user-facing *copy* of an already retrieved evidence payload.
    Original PDF pages, fact banks, retrieved IDs and source citations stay intact.
    Never removes a question, changes a financial value, or fabricates evidence.
    If a request is not a known evidence JSON object, fail clearly rather than
    guessing how to modify the prompt. No API call is made in that case.
    """
    original_estimate = _estimate_tokens(kwargs)
    info = {'original_estimate': original_estimate,
            'bounded_estimate': original_estimate, 'source_text_shortened': False,
            'old_chunk_chars': 0, 'new_chunk_chars': 0}
    if original_estimate <= token_limit:
        return kwargs, info
    messages = kwargs.get('messages', [])
    if not messages or not isinstance(messages[-1], dict) or messages[-1].get('role') != 'user':
        raise RuntimeError('Groq request exceeds local limit; no supported evidence payload to shorten')
    try:
        content = json.loads(messages[-1].get('content', ''))
    except (ValueError, TypeError):
        raise RuntimeError('Groq request exceeds local limit; prompt is not evidence JSON') from None
    if not isinstance(content, dict) or not isinstance(content.get('evidence'), dict):
        raise RuntimeError('Groq request exceeds local limit; no PDF evidence section')
    chunks = content['evidence'].get('chunks', [])
    if not isinstance(chunks, list) or not chunks or any(not isinstance(c, dict) or
                                                       not isinstance(c.get('text'), str)
                                                       for c in chunks):
        raise RuntimeError('Groq request exceeds local limit; no source excerpts can be shortened')
    info['old_chunk_chars'] = sum(len(c['text']) for c in chunks)
    # First keep all source identities and most source text. Reduce only as much
    # as needed. A shortening label makes omission visible to the answering LLM.
    for cap in (4000, 3500, 3000, 2500, 2000, 1600, 1200, 900, 650, 400, 250):
        proposed = deepcopy(content)
        for c in proposed['evidence']['chunks']:
            text = c['text']
            if len(text) > cap:
                c['text'] = text[:cap] + '\n[END OF SELECTED SOURCE EXCERPT; REMAINDER NOT PROVIDED]'
        edited = dict(kwargs)
        edited['messages'] = [dict(m) for m in messages]
        edited['messages'][-1]['content'] = json.dumps(proposed, ensure_ascii=False)
        estimate = _estimate_tokens(edited)
        if estimate <= token_limit:
            info.update(bounded_estimate=estimate, source_text_shortened=True,
                        new_chunk_chars=sum(len(c['text']) for c in proposed['evidence']['chunks']))
            return edited, info
    raise RuntimeError('Groq PDF evidence remains over local token limit after safe excerpt shortening')


def _safe_exception_info(exc: Exception) -> dict:
    """Keep only code locations, never exception payloads, prompts, keys or headers."""
    frames = traceback.extract_tb(exc.__traceback__)
    locations = [{'file': Path(f.filename).name, 'function': f.name, 'line': f.lineno}
                 for f in frames[-5:]]
    message = str(exc)
    reason = 'uncategorized_program_error'
    if isinstance(exc, RuntimeError) and 'Groq request exceeds local limit' in message:
        reason = 'local_request_token_limit'
    elif isinstance(exc, RuntimeError) and 'Groq PDF evidence remains over local token limit' in message:
        reason = 'local_request_token_limit_after_excerpt_shortening'
    elif isinstance(exc, RuntimeError) and 'Single Groq request estimate exceeds' in message:
        reason = 'local_request_token_limit'
    elif isinstance(exc, RuntimeError) and 'daily token budget' in message:
        reason = 'configured_daily_token_limit'
    elif isinstance(exc, RuntimeError) and 'Groq request count' in message:
        reason = 'configured_call_count_limit'
    status = getattr(exc, 'status_code', None)
    if status == 400:
        reason = 'groq_http_400_' + _safe_http_category(exc)
    info = {'error_type': type(exc).__name__, 'error_reason': reason,
            'code_locations': locations}
    if isinstance(status, int):
        info['http_status'] = status
    return info


def _safe_http_category(exc: Exception) -> str:
    """Classify a provider error without logging the message, body or request."""
    body = getattr(exc, 'body', None)
    body = body.get('error', body) if isinstance(body, dict) else None
    message = str(body.get('message', '')).casefold() if isinstance(body, dict) else ''
    if not message:
        # The SDK's string may contain user data; examine in memory only.
        message = str(exc).casefold()
    if 'schema' in message or 'json' in message or 'response_format' in message:
        return 'response_format_or_schema'
    if 'token' in message or 'context' in message or 'length' in message:
        return 'token_or_context'
    if 'unsupported' in message or 'parameter' in message:
        return 'unsupported_parameter'
    if 'content_filter' in message or 'safety' in message:
        return 'content_filter'
    return 'unclassified'


def _check_json_object_fallback(result, schema: dict) -> None:
    """JSON object mode is weaker than strict mode; check the FULL intent shape."""
    try:
        content = result.choices[0].message.content
        decoded = json.loads(content)
    except (AttributeError, ValueError, TypeError, IndexError) as exc:
        raise ValueError('Groq intent JSON fallback was invalid') from exc
    required = schema['required']
    props = schema['properties']
    if not isinstance(decoded, dict) or set(decoded) != set(required):
        raise ValueError('Groq intent JSON fallback keys do not match the schema')
    for name, rule in props.items():
        value = decoded[name]
        allowed_type = rule.get('type')
        if allowed_type == 'string' and not isinstance(value, str):
            raise ValueError('Groq intent JSON fallback type mismatch')
        if allowed_type == 'integer' and (type(value) is not int):
            raise ValueError('Groq intent JSON fallback type mismatch')
        if 'enum' in rule and value not in rule['enum']:
            raise ValueError('Groq intent JSON fallback enum mismatch')


class UsageGovernor:
    """One shared rate limit across all internal Groq calls (intent/answer/review)."""
    def __init__(self, path: Path, *, per_minute_tokens: int = 7000,
                 per_minute_requests: int = 10, daily_tokens: int = 140000,
                 daily_requests: int = 140, max_calls: int = 100,
                 minimum_gap: float = DEFAULT_GROQ_COOLDOWN_SECONDS, sleep=time.sleep, clock=time.time):
        self.path = path
        self.per_minute_tokens = per_minute_tokens
        self.per_minute_requests = per_minute_requests
        self.daily_tokens = daily_tokens
        self.daily_requests = daily_requests
        self.max_calls = max_calls
        self.minimum_gap = minimum_gap
        self.sleep, self.clock = sleep, clock
        self.events = json_lines(path)

    def _recent(self, now, age):
        return [r for r in self.events if now - float(r['unix_time']) < age and
                now >= float(r['unix_time'])]

    @staticmethod
    def charged_tokens(entry: dict) -> int:
        """Count tokens that Groq reported using, not unused estimated output.

        Keep the original budget_tokens entries unchanged for audit/replay.
        When no provider usage was returned (failed or unknown request), charge
        the FULL estimated budget. This fails conservatively, never for free.
        """
        measured = entry.get('reported_tokens')
        reserved = int(entry['budget_tokens'])
        if measured is None:
            return reserved
        measured = int(measured)
        if measured < 0:
            raise ValueError('Invalid negative provider token usage')
        return max(0, measured)

    def reserve(self, estimate: int) -> None:
        if estimate > self.per_minute_tokens:
            raise RuntimeError('Single Groq request estimate exceeds permitted minute token budget')
        while True:
            now = self.clock()
            day = self._recent(now, 86400)
            if len(day) >= self.daily_requests or len(self.events) >= self.max_calls:
                raise RuntimeError('Configured maximum Groq request count reached')
            # Use real usage for *completed* calls; reserve the full estimate
            # for the next call, and for any previous call with missing usage.
            # Keep the original conservative daily ceiling unchanged (140k).
            if sum(self.charged_tokens(e) for e in day) + estimate > self.daily_tokens:
                raise RuntimeError('Configured maximum daily token budget reached')
            minute = self._recent(now, 60)
            remain = max(0.0, self.minimum_gap - (now - float(self.events[-1]['unix_time']))) if self.events else 0.0
            if len(minute) >= self.per_minute_requests or sum(self.charged_tokens(e) for e in minute) + estimate > self.per_minute_tokens:
                remain = max(remain, 61.0 - (now - float(minute[0]['unix_time'])))
            if remain <= 0:
                return
            # One pause governs all Groq calls, including model retries within a question.
            if remain >= 1.0:
                print(f'  Groq cooldown / rate budget: pausing {remain:.1f} seconds before next request', flush=True)
            self.sleep(max(0.1, remain))

    def record(self, *, estimate: int, model: str, success: bool, usage=None, error: str = ''):
        # Never log prompts, API keys, response text or headers.
        total = getattr(usage, 'total_tokens', None) if usage is not None else None
        used = max(estimate, int(total)) if total is not None else estimate
        entry = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                 'unix_time': self.clock(), 'model': model,
                 'budget_tokens': used,
                 'reported_tokens': int(total) if total is not None else None,
                 'success': success, 'error_type': error}
        append_jsonl(self.path, entry)
        self.events.append(entry)


class PacedCompletions:
    def __init__(self, target, governor: UsageGovernor):
        self.target, self.gov = target, governor

    def create(self, **kwargs):
        kwargs, fit = _fit_evidence_request(kwargs, self.gov.per_minute_tokens - 150)
        if fit['source_text_shortened']:
            print('  Source excerpts shortened to respect Groq token budget: '
                  f"{fit['original_estimate']} -> {fit['bounded_estimate']} estimated tokens "
                  '(financial values and evidence IDs unchanged).', flush=True)
        original_format = kwargs.get('response_format', {})
        structured = (original_format.get('json_schema', {})
                      if isinstance(original_format, dict) else {})
        schema_name = structured.get('name')
        original_schema = structured.get('schema')
        # Only this small intent-parsing request gets a one-time alternative.
        # Do not switch document-answering requests to weaker JSON output.
        allowed_fallback = (original_format.get('type') == 'json_schema'
                            and schema_name == 'financial_intent'
                            and structured.get('strict') is True
                            and isinstance(original_schema, dict))
        used_fallback = False
        transient_tries = 0
        current = kwargs
        while True:
            estimate = _estimate_tokens(current)
            self.gov.reserve(estimate)
            try:
                result = self.target.create(**current)
            except Exception as exc:
                self.gov.record(estimate=estimate, model=str(current.get('model', '')),
                                success=False, error=type(exc).__name__)
                status = getattr(exc, 'status_code', None)
                if status == 400 and allowed_fallback and not used_fallback:
                    # Groq documents json_object as supported for this model.
                    # Treat output as untrusted and validate EVERY required field.
                    print('  Groq rejected strict financial-intent JSON (HTTP 400); '
                          'trying JSON object mode once, with full field validation.', flush=True)
                    current = {**kwargs, 'response_format': {'type': 'json_object'}}
                    used_fallback = True
                    transient_tries = 0
                    continue  # Governor enforces cooldown and token caps again.
                retry = status in {429, 500, 502, 503, 504}
                if not retry or transient_tries >= 2:
                    raise
                transient_tries += 1
                headers = getattr(getattr(exc, 'response', None), 'headers', {})
                try:
                    retry_after = float(headers.get('retry-after', ''))
                except (ValueError, TypeError, AttributeError):
                    retry_after = None
                if retry_after is not None and retry_after > 600:
                    raise RuntimeError('Groq asks for a cooldown over 10 minutes; stop and resume later') from exc
                self.gov.sleep(max(15, retry_after or 15))
            else:
                # A successful HTTP response is billed even if its JSON is invalid.
                self.gov.record(estimate=estimate, model=str(current.get('model', '')),
                                success=True, usage=getattr(result, 'usage', None))
                if used_fallback:
                    _check_json_object_fallback(result, original_schema)
                return result


class PacedGroq:
    def __init__(self, client, governor):
        self.chat = SimpleNamespace(completions=PacedCompletions(client.chat.completions, governor))


def _finished(path: Path) -> dict[str, dict]:
    items = json_lines(path)
    ids = [r.get('id') for r in items]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate completed answers in checkpoint')
    return {r['id']: r for r in items}


def build_summary(pending: list[dict], checkpoint: dict[str, dict], governor=None) -> dict:
    statuses = Counter(x['result'].get('status', 'missing') for x in checkpoint.values())
    routes = Counter(x['result'].get('route', 'missing') for x in checkpoint.values())
    return {'benchmark': 'original_34_exposed_development_questions',
            'baseline_preexisting_answers': 5, 'baseline_preexisting_future_refusals': 6,
            'pending_live_questions': len(pending),
            'live_completed': len(checkpoint),
            'live_unfinished': len(pending) - len(checkpoint),
            'live_numeric_completed': sum(x['kind'] == 'numeric' for x in checkpoint.values()),
            'live_explanations_completed': sum(x['kind'] == 'explanation' for x in checkpoint.values()),
            'live_status_counts': dict(statuses),
            'live_route_counts': dict(routes),
            'groq_api_requests_recorded': len(governor.events) if governor else 0,
            'groq_token_budget_reserved': sum(int(e['budget_tokens']) for e in governor.events) if governor else 0,
            'groq_tokens_charged_or_conservatively_estimated': sum(
                governor.charged_tokens(e) for e in governor.events) if governor else 0,
            'groq_daily_token_cap': governor.daily_tokens if governor else 140000,
            'accuracy': 'pending_independent_review; do not infer correctness from answered status'}


def write_outputs(out: Path, pending: list[dict], checkpoint: dict[str, dict], governor=None) -> None:
    json_atomic(out/'summary.json', build_summary(pending, checkpoint, governor))
    fields = ['id', 'ticker', 'kind', 'status', 'route', 'answer', 'citations_count',
              'verification_status', 'review_needed', 'error_type']
    with (out/'results.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fields); w.writeheader()
        for row in pending:
            item = checkpoint.get(row['id'])
            result = item['result'] if item else {}
            w.writerow({'id': row['id'], 'ticker': row['ticker'], 'kind': row['kind'],
                        'status': result.get('status', 'not_run'),
                        'route': result.get('route', ''),
                        'answer': result.get('answer', ''),
                        'citations_count': len(result.get('citations') or []),
                        'verification_status': result.get('verification_status', ''),
                        'review_needed': 'yes' if item else 'not_run',
                        'error_type': result.get('error_type', '')})
    fields = ['id', 'ticker', 'kind', 'question', 'answer', 'status', 'route',
              'source_citations', 'source_page_review',
              'human_grade', 'reason', 'correct_value_if_different']
    with (out/'manual_review.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fields); w.writeheader()
        for row in pending:
            item = checkpoint.get(row['id'])
            result = item['result'] if item else {}
            w.writerow({'id': row['id'], 'ticker': row['ticker'], 'kind': row['kind'],
                        'question': row['question'], 'answer': result.get('answer', ''),
                        'status': result.get('status', 'not_run'), 'route': result.get('route', ''),
                        'source_citations': json.dumps(result.get('citations') or [], ensure_ascii=False),
                        'source_page_review': json.dumps(result.get('source_page_verification') or {}, ensure_ascii=False),
                        'human_grade': '', 'reason': '', 'correct_value_if_different': ''})


def pack_results(out: Path) -> Path:
    target = out/'RiskLens_23_Live_Results_For_Review.zip'
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for filename in ('manifest.json', 'summary.json', 'results.csv', 'manual_review.csv',
                         'live_checkpoint.jsonl', 'api_usage.jsonl', 'run_interruptions.jsonl'):
            p = out/filename
            if p.is_file(): z.write(p, arcname=filename)
    return target


def run_cases(pending: list[dict], out: Path, answer_fn, governor=None,
              max_questions: int = 23) -> dict:
    checkpoint_file = out/'live_checkpoint.jsonl'
    checkpoint = _finished(checkpoint_file)
    known = {c['id'] for c in pending}
    if set(checkpoint) - known:
        raise ValueError('Checkpoint includes an unknown question')
    done_now = 0
    for case in pending:
        if case['id'] in checkpoint or done_now >= max_questions:
            continue
        print(f"Running {case['id']} ({case['kind']}) ...", flush=True)
        try:
            # IMPORTANT: do not pass expected_value, expected_unit or benchmark ID.
            result = answer_fn(case['question'], case['ticker'])
            if not isinstance(result, dict) or 'status' not in result:
                raise ValueError('Answerer failed to return status dictionary')
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            # Keep the detailed code location, not raw exception messages that
            # might contain prompts/headers/secrets. Failed cases remain pending.
            result = {'status': 'run_error', 'route': 'batch_exception', 'answer': '',
                      **_safe_exception_info(exc), 'citations': []}
        if result.get('status') in {'provider_unavailable', 'run_error'}:
            append_jsonl(out/'run_interruptions.jsonl',
                         {'id': case['id'], 'status': result['status'],
                          'error_type': result.get('error_type', ''),
                          'error_reason': result.get('error_reason', ''),
                          'code_locations': result.get('code_locations', []),
                          'http_status': result.get('http_status'),
                          'saved_at': datetime.now(timezone.utc).isoformat()})
            write_outputs(out, pending, checkpoint, governor)
            print(f"  -> {result.get('status')}; paused safely, question NOT marked complete", flush=True)
            break
        entry = {'id': case['id'], 'ticker': case['ticker'], 'kind': case['kind'],
                 'question': case['question'], 'result': result,
                 'saved_at': datetime.now(timezone.utc).isoformat()}
        append_jsonl(checkpoint_file, entry)
        checkpoint[case['id']] = entry
        done_now += 1
        write_outputs(out, pending, checkpoint, governor)
        print(f"  -> {result.get('status')} / {result.get('route')}", flush=True)
    pack_results(out)
    return build_summary(pending, checkpoint, governor)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path('.'), help='Existing RiskLens project root')
    p.add_argument('--archive', type=Path, required=True, help='Original ZIP with eight NSE XML filings')
    p.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE,
                   help='Saved offline 34-question result from previous Mac run')
    p.add_argument('--questions', type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument('--db', type=Path, default=DEFAULT_DB)
    p.add_argument('--bank-dir', type=Path, default=DEFAULT_BANK)
    p.add_argument('--pdf-dir', type=Path, default=DEFAULT_PDF)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--max-questions', type=int, default=23,
                   help='Optional first live batch size; can resume later')
    p.add_argument('--cooldown-seconds', type=float, default=DEFAULT_GROQ_COOLDOWN_SECONDS,
                   help='Minimum gap between ANY Groq calls, even internal/retries/resume; default 75')
    p.add_argument('--live', action='store_true', help='Actually call Groq; never implicit')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 1 <= args.top_k <= 20 or not 1 <= args.max_questions <= 23:
        raise SystemExit('--top-k must be 1..20 and --max-questions must be 1..23')
    if not math.isfinite(args.cooldown_seconds) or args.cooldown_seconds < 60:
        raise SystemExit('--cooldown-seconds must be at least 60 (recommended 75 or 90)')
    repo = args.repo.resolve()
    if not (repo/'pyproject.toml').is_file():
        raise SystemExit('Run from an existing RiskLens repository, using --repo .')
    def local(p: Path) -> Path:
        return p if p.is_absolute() else repo/p
    archive, db, bank, pdf_dir, questions, out = map(local,
        (args.archive, args.db, args.bank_dir, args.pdf_dir, args.questions, args.out))
    baseline = args.baseline.resolve()
    pending, previous = validate_inputs(questions, baseline)
    if not archive.is_file():
        raise FileNotFoundError('Original eight-filing archive missing: ' + str(archive))
    if not db.is_file():
        raise FileNotFoundError('Existing PDF search index missing: ' + str(db))
    if not bank.is_dir() or not all((bank/f).is_file() for f in
            ('financial_facts_v3.json', 'financial_facts_auto_v4.json',
             'financial_layout_panels_v6.json')):
        raise FileNotFoundError('Existing Phase 7 fact-bank files missing; no writes performed')
    if not pdf_dir.is_dir() or not list(pdf_dir.glob('*.pdf')):
        raise FileNotFoundError('Original PDF directory is missing or has no PDFs')
    count = check_sqlite(db)
    manifest = {'mode': '23_pending_groq_only',
                'baseline_sha256': digest(baseline),
                'questions_sha256': digest(questions),
                'xbrl_sha256': digest(archive),
                'questions_count': 34, 'pending_ids': [x['id'] for x in pending],
                'existing_index_documents': count,
                'original_repo': str(repo), 'benchmark_is_blind': False}
    out.mkdir(parents=True, exist_ok=True)
    mf = out/'manifest.json'
    if mf.exists():
        if json.loads(mf.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Sources changed since this live run; choose a new --out directory')
    else:
        json_atomic(mf, manifest)
    from_path = repo/'src'
    sys.path.insert(0, str(from_path))
    os.chdir(repo)
    from risklens.research.xbrl_rag_v1 import load_filings, lookup_xbrl, run_query_xbrl_first
    filings = load_filings(archive)
    if len(filings) != 8:
        raise ValueError('Expected exactly eight FY2025 XBRL filings')
    for case in pending:
        if lookup_xbrl(case['question'], archive=archive, ticker=case['ticker']) is not None:
            raise ValueError('An XBRL direct answer unexpectedly appeared for ' + case['id'])
    print(f'Preflight passed: {len(filings)} XBRL files, {count} indexed PDF reports, '
          f'{len(pending)} pending questions (15 numeric + 8 explanations).')
    if not args.live:
        print('PREVIEW ONLY: zero Groq API calls. Add --live to execute.'); return 0
    from risklens.research.config import load_project_env
    load_project_env()
    key = os.environ.get('GROQ_API_KEY')
    if not key:
        raise RuntimeError('GROQ_API_KEY is not set in your existing environment/.env. No key needs to be shared.')
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError('Your existing venv is missing groq. No installation attempted.') from exc
    governor = UsageGovernor(out/'api_usage.jsonl', minimum_gap=args.cooldown_seconds)
    print(f'Groq minimum cooldown: {args.cooldown_seconds:g} seconds between API requests; '
          'existing usage log respected on resume.', flush=True)
    client = PacedGroq(Groq(api_key=key, timeout=75.0, max_retries=0), governor)
    from risklens.research.index import ResearchStore
    from risklens.research.embeddings import SentenceEmbedder, create_embedder
    class LazySentence:
        name = SentenceEmbedder.name
        dimension = SentenceEmbedder.dimension
        def __init__(self): self._model = None
        def encode(self, txt):
            if self._model is None: self._model = create_embedder('sentence')
            return self._model.encode(txt)
    with ResearchStore(db) as store:
        embedder = LazySentence()
        def answer(question: str, ticker: str) -> dict:
            return run_query_xbrl_first(store, embedder, question,
                 xbrl_archive=archive, provider='groq', ticker=ticker,
                 client=client, bank_dir=bank, pdf_dir=pdf_dir, top_k=args.top_k,
                 verify_financial=True, verify_vision=False)
        summary = run_cases(pending, out, answer, governor, args.max_questions)
    print(json.dumps(summary, indent=2))
    print('Result ZIP:', out/'RiskLens_23_Live_Results_For_Review.zip')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
