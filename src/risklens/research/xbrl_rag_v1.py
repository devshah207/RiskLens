"""Source-constrained, XBRL-first answer route for RiskLens Phase 7.

Loads eight actual NSE XBRL instances locally. A direct answer is permitted
only after exact concept, company, accounting scope, year, period and unit
checks. On any uncertainty returns None so existing PDF RAG is unchanged.

No gold benchmark values or benchmark IDs are used to answer questions.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from hashlib import sha256
from datetime import datetime
from pathlib import Path
import re
import tempfile
import zipfile

from .xbrl_instance_v1 import parse_instance

_FY_START, _FY_END = '2024-04-01', '2025-03-31'
_NAMES = {'RELIANCE':'RIL', 'HDFCBANK':'HDFCBANK', 'INFY':'INFY', 'ICICIBANK':'ICICIBANK'}
_TICKER_PATTERNS = {
    'RIL': r'\b(?:reliance(?:\s+industries)?|ril)\b',
    'HDFCBANK': r'\b(?:hdfc(?:\s+bank)?|hdfcbank)\b',
    'INFY': r'\b(?:infosys|infy)\b',
    'ICICIBANK': r'\b(?:icici(?:\s+bank)?|icicibank)\b',
}
_INR = ['iso4217:INR']
_INR_SHARE = ['iso4217:INR', 'xbrli:shares']


@dataclass(frozen=True)
class Filing:
    ticker: str
    scope: str
    filename: str
    sha256: str
    published_on: str
    facts: tuple[dict, ...]


def _tag(f: dict) -> str:
    return f['concept'].split('}')[-1]


def _one_text(facts: list[dict], name: str) -> str:
    vals = {f['raw_value'] for f in facts if _tag(f) == name and f['raw_value'] is not None}
    if len(vals) != 1:
        raise ValueError(f'XBRL filing missing or conflicting {name}: {vals}')
    return vals.pop().strip()


@lru_cache(maxsize=3)
def load_filings(archive: str | Path) -> dict[tuple[str, str], Filing]:
    """Parse the real XML, never a sample fixture or the NSE HTML preview."""
    archive = Path(archive).resolve()
    if not archive.is_file():
        raise FileNotFoundError(f'XBRL archive missing: {archive}')
    results = {}
    with zipfile.ZipFile(archive) as z:
        xmls = [e for e in z.infolist() if e.filename.lower().endswith('.xml')
                and '/' not in e.filename and not e.filename.startswith('.')]
        if len(xmls) != 8:
            raise ValueError(f'Expected 8 original XBRL XML filings, found {len(xmls)}')
        with tempfile.TemporaryDirectory() as d:
            for info in xmls:
                if info.file_size > 70_000_000:
                    raise ValueError('XML source larger than 70 MB')
                payload = z.read(info)
                path = Path(d) / info.filename
                path.write_bytes(payload)
                facts = parse_instance(path)
                symbol = _one_text(facts, 'Symbol').upper()
                scope = _one_text(facts, 'NatureOfReportStandaloneConsolidated').lower()
                if symbol not in _NAMES or scope not in {'standalone', 'consolidated'}:
                    raise ValueError(f'Unknown company/scope: {symbol}/{scope}')
                if (_one_text(facts, 'DateOfStartOfFinancialYear') != _FY_START or
                    _one_text(facts, 'DateOfEndOfFinancialYear') != _FY_END or
                    _one_text(facts, 'WhetherResultsAreAuditedOrUnaudited').lower() != 'audited'):
                    raise ValueError('Incorrect reporting year or audit status')
                key = (_NAMES[symbol], scope)
                if key in results:
                    raise ValueError(f'Duplicate company-scope filing: {key}')
                m = re.search(r'_(\d{2})(\d{2})(20\d{2})\d{6}_WEB\.xml$', info.filename)
                if m is None:
                    raise ValueError('Cannot establish filing publication date')
                publication = datetime.strptime(''.join(m.groups()), '%d%m%Y').date().isoformat()
                results[key] = Filing(key[0], scope, info.filename,
                                      sha256(payload).hexdigest(), publication, tuple(facts))
    if set(results) != {(ticker, scope) for ticker in _NAMES.values()
                         for scope in ('standalone', 'consolidated')}:
        raise ValueError('Incomplete eight-filing XBRL archive')
    return results


def _explicit_company(question: str, ticker: str | None) -> str | None:
    explicit = {t for t, pat in _TICKER_PATTERNS.items() if re.search(pat, question, re.I)}
    if ticker:
        ticker = ticker.upper()
        if ticker not in _TICKER_PATTERNS or (explicit and explicit != {ticker}):
            return None
        return ticker
    return next(iter(explicit)) if len(explicit) == 1 else None


def _explicit_year(question: str) -> int | None:
    # Require a literal year. FY2024-25 means the year ENDING in 2025.
    pair = re.search(r'\bFY\s*(20\d{2})\s*[-–/]\s*(?:20)?(\d{2})\b', question, re.I)
    if pair:
        start, end = int(pair.group(1)), 2000 + int(pair.group(2))
        return end if end == start + 1 else None
    nums = set(int(x) for x in re.findall(r'(?<![0-9])20\d{2}\b', question))
    if len(nums) != 1:
        return None
    return nums.pop()


def _requested_scope(question: str) -> str | None:
    words = {w.lower() for w in re.findall(r'[a-z]+', question, re.I)}
    if re.search(r'\bnon[- ]consolidated\b', question, re.I):
        return None
    scopes = words.intersection({'standalone', 'consolidated'})
    return next(iter(scopes)) if len(scopes) == 1 else None


def _concept(question: str, ticker: str) -> tuple[str, str, str] | None:
    """Small, declared financial vocabulary; NEVER choose by token similarity.

    Returns (exact XBRL concept, instant/duration, quantity class).
    Unsupported modifiers and non-standard concepts fall through to PDF RAG.
    """
    q = ' '.join(re.findall(r'[a-z0-9]+', question.lower()))
    def has(pattern):
        return re.search(pattern, q) is not None
    if has(r'\b(?:how|what|which|define|definition|explain|describe|why)\b') and has(
            r'\b(?:define|definition|explain|describe|why|which|principles|framework|items|risk types)\b'):
        return None
    # Financial modifiers radically change meaning; do not discard them.
    if has(r'\b(?:excluding|except|net of|without|per employee|working funds|operating profit|value of sales|sales and services|'
           r'non interest income|cost of deposits|segment|ebitda|income from services|'
           r'cost of sales|selling and marketing|general and administration|'
           r'net current assets|transfer to statutory|transferred to statutory|'
           r'profit per employee|business per employee|excise duty|continuing operations|discontinued operations|before extraordinary items|after extraordinary items)\b'):
        return None
    if has(r'\b(?:attributable to non controlling interests|non controlling interest share of (?:net )?profit)\b'):
        return ('ProfitOrLossAttributableToNonControllingInterests', 'duration', 'money') if ticker == 'RIL' else None
    if has(r'\b(?:non controlling interests|minority interest)\b'):
        return None
    if has(r'\b(?:basic (?:consolidated )?earnings per (?:equity )?share|basic eps)\b'):
        if ticker == 'RIL':
            return ('BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations', 'duration', 'per_share')
        return None  # Banking filings have before/after-extraordinary variants.
    if has(r'\breserves and surplus\b'):
        return ('ReservesAndSurplus', 'instant', 'money')
    if has(r'\bdeferred tax liabilities net\b'):
        return ('DeferredTaxLiabilitiesNet', 'instant', 'money')
    if has(r'\b(?:revenue from operations|revenue)\b') and not has(r'\b(?:net revenue|segment revenue|revenue growth|revenue per|sales revenue)\b'):
        return ('RevenueFromOperations', 'duration', 'money') if ticker in {'RIL', 'INFY'} else None
    if has(r'\b(?:net profit|net income|profit after tax|profit for (?:the )?period)\b'):
        return ('ProfitLossForThePeriod' if ticker in {'HDFCBANK', 'ICICIBANK'} else 'ProfitLossForPeriod',
                'duration', 'money')
    if has(r'\btotal income\b'):
        return ('Income', 'duration', 'money')
    if has(r'\bborrowings\b') and not has(r'\b(?:from borrowings|interest on borrowings|new borrowings)\b'):
        return ('Borrowings', 'instant', 'money') if ticker in {'HDFCBANK','ICICIBANK'} else None
    return None


def _output_unit(question: str, quantity: str) -> tuple[str, Decimal, str] | None:
    q = question.lower()
    if quantity == 'per_share':
        if re.search(r'\b(?:crore|crores|thousands|million|percent|%)\b', q):
            return None
        return ('INR/share', Decimal(1), ' per share')
    if '%' in question or re.search(r'\b(?:percent|percentage|million|billion|usd)\b', q):
        return None
    if re.search(r'\b(?:thousand|thousands)\b', q):
        return ('thousand_rupees', Decimal(1000), ' thousand rupees')
    if re.search(r'\b(?:crore|crores)\b', q):
        return ('crore', Decimal(10_000_000), ' crore')
    return ('rupees', Decimal(1), ' rupees')


def _pretty(x: Decimal) -> str:
    if x == x.to_integral():
        return f'{x:,.0f}'
    return f'{x:,.5f}'.rstrip('0').rstrip('.')


def lookup_xbrl(question: str, *, archive: str | Path,
                ticker: str | None = None, fiscal_year: str | None = None,
                as_of: str | None = None) -> dict | None:
    """Exact XBRL match or None. None always means 'try original PDF RAG'."""
    company = _explicit_company(question, ticker)
    scope = _requested_scope(question)
    year = _explicit_year(question)
    if company is None or scope is None or year is None:
        return None
    if fiscal_year and not (str(year) in fiscal_year or str(year-1) in fiscal_year):
        return None
    # Annual 2025 filings cannot answer later years, and missing comparative
    # contexts are never silently replaced with FY2025 amounts.
    if year > 2025 or year < 2015:
        return None
    if as_of and as_of < '2025-04-01':
        return None  # FY2025 filing cannot be used before it existed.
    selected = _concept(question, company)
    if selected is None:
        return None
    concept, period, quantity = selected
    unit = _output_unit(question, quantity)
    if unit is None:
        return None
    kind, multiplier, label = unit
    filing = load_filings(str(Path(archive).resolve()))[(company, scope)]
    if as_of and as_of < filing.published_on:
        return None
    candidates = []
    expected_measures = _INR if quantity == 'money' else _INR_SHARE
    for fact in filing.facts:
        if (_tag(fact) != concept or fact.get('numeric_value') is None or
                fact.get('is_nil') or fact.get('dimensions') or
                fact['period_kind'] != period or
                fact['unit_measures'] != expected_measures or
                fact['end'] != f'{year}-03-31'):
            continue
        if period == 'duration' and fact['start'] != f'{year-1}-04-01':
            continue
        try:
            val = Decimal(fact['numeric_value'])
        except (TypeError, ArithmeticError):
            continue
        if not val.is_finite():
            continue
        candidates.append(fact)
    # The presence of multiple facts is ambiguity, not a reason to pick one.
    if len(candidates) != 1:
        return None
    fact = candidates[0]
    value = Decimal(fact['numeric_value']) / multiplier
    answer = (f'{company} {scope} {concept} (FY{year}): '
              + ('₹' if quantity == 'money' or quantity == 'per_share' else '')
              + _pretty(value) + (label if quantity == 'money' else ' per share'))
    return {
        'question': question,
        'answer': answer,
        'status': 'answered',
        'provider': 'nse_xbrl_xml_local',
        'route': 'xbrl_exact_fact',
        'verification_status': 'xbrl_original_tag_context_and_unit_checked_not_human_certified',
        'citations': [{
            'source_kind': 'NSE_XBRL_XML', 'source_file': filing.filename,
            'source_sha256': filing.sha256, 'published_on': filing.published_on, 'source_url':
                'https://www.nseindia.com/companies-listing/corporate-integrated-filing',
            'ticker': company, 'reported_scope': scope,
            'concept_qname': fact['concept'], 'context_id': fact['context_id'],
            'financial_year': year, 'period_kind': period,
            'period_start': fact['start'], 'period_end': fact['end'],
            'unit_measures': fact['unit_measures'],
            'raw_reported_value': fact['raw_value'],
            'decimals': fact['decimals'], 'dimensions': fact['dimensions'],
            'note': 'XBRL filing citation (not a PDF page citation).'
        }],
        'intent': {'ticker': company, 'reporting_scope': scope,
                   'year': year, 'metric_code': concept,
                   'metric_label': concept, 'unit': kind, 'model_used': False},
    }


def run_query_xbrl_first(store, embedder, question: str, *, xbrl_archive: str | Path,
                         ticker: str | None = None, fiscal_year: str | None = None,
                         as_of: str | None = None, **rag_options) -> dict:
    """Additive adapter that leaves the existing RiskLens RAG code unchanged.

    Exact XBRL match -> answer with original XML provenance.
    Any other question -> original Phase 7 run_query (retrieval/Groq unchanged).
    """
    direct = lookup_xbrl(question, archive=xbrl_archive, ticker=ticker,
                         fiscal_year=fiscal_year, as_of=as_of)
    if direct is not None:
        return direct
    from .llm_guided_v7 import run_query
    return run_query(store, embedder, question, ticker=ticker,
                     fiscal_year=fiscal_year, as_of=as_of, **rag_options)
