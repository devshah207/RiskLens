"""Read a supplied *real XBRL 2.1 XML instance* without guessing financial meanings.

This does NOT parse an NSE webpage, a ZIP, or inline-XBRL HTML. It does not
load an external taxonomy. Original concept identifiers remain intact.
"""
from __future__ import annotations
import argparse
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from lxml import etree

XBRLI = 'http://www.xbrl.org/2003/instance'
XBRLDI = 'http://xbrl.org/2006/xbrldi'
XSI = 'http://www.w3.org/2001/XMLSchema-instance'


def _qname(el):
    return etree.QName(el).text


def _text(el, local):
    candidates = el.xpath(f'.//*[local-name()="{local}"]')
    return (candidates[0].text or '').strip() if candidates else None


def _contexts(root):
    ans = {}
    for context in root.findall(f'{{{XBRLI}}}context'):
        cid = context.get('id')
        if not cid or cid in ans:
            raise ValueError('Missing or repeated context ID')
        entity = context.find(f'{{{XBRLI}}}entity/{{{XBRLI}}}identifier')
        period = context.find(f'{{{XBRLI}}}period')
        if entity is None or period is None:
            raise ValueError(f'Incomplete context: {cid}')
        instant = period.find(f'{{{XBRLI}}}instant')
        start = period.find(f'{{{XBRLI}}}startDate')
        end = period.find(f'{{{XBRLI}}}endDate')
        if instant is not None and start is None and end is None:
            kind = 'instant'; st = None; ed = (instant.text or '').strip()
        elif instant is None and start is not None and end is not None:
            kind = 'duration'; st = (start.text or '').strip(); ed = (end.text or '').strip()
        else:
            raise ValueError(f'Unknown/mixed period in {cid}')
        # Any dimension must stay attached to the fact. It can change its meaning.
        dims=[]
        for d in context.findall(f'.//{{{XBRLDI}}}explicitMember'):
            dims.append({'axis':d.get('dimension'), 'member':(d.text or '').strip()})
        for d in context.findall(f'.//{{{XBRLDI}}}typedMember'):
            dims.append({'axis':d.get('dimension'), 'typed_xml':etree.tostring(d,encoding='unicode')})
        ans[cid] = {'entity':(entity.text or '').strip(),
                    'entity_scheme':entity.get('scheme'),
                    'period_kind':kind,'start':st,'end':ed,'dimensions':dims}
    return ans


def _units(root):
    ans={}
    for u in root.findall(f'{{{XBRLI}}}unit'):
        uid=u.get('id')
        if not uid or uid in ans: raise ValueError('Missing or repeated unit ID')
        measures=[(m.text or '').strip() for m in u.findall(f'.//{{{XBRLI}}}measure')]
        if not measures: raise ValueError(f'Missing measure for unit: {uid}')
        ans[uid]=measures
    return ans


def parse_instance(path: str | Path):
    path=Path(path)
    if path.stat().st_size > 100_000_000:
        raise ValueError('Refusing instance larger than 100 MB')
    xml_parser=etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
    root=etree.parse(str(path),parser=xml_parser).getroot()
    if root.tag != f'{{{XBRLI}}}xbrl':
        raise ValueError('Expected an XBRL 2.1 XML instance. NSE web-view HTML/iXBRL needs Arelle.')
    contexts=_contexts(root); units=_units(root)
    out=[]
    for el in root:
        context_id=el.get('contextRef')
        if not context_id: continue
        if context_id not in contexts:
            raise ValueError(f'Unknown contextRef {context_id}')
        is_nil=el.get(f'{{{XSI}}}nil','false').lower()=='true'
        raw=''.join(el.itertext()).strip()
        unit_id=el.get('unitRef')
        if unit_id is not None and unit_id not in units:
            raise ValueError(f'Unknown unitRef {unit_id}')
        # Preserve exact strings and concept QName, do not "resolve" to similar metrics.
        data={'concept':_qname(el), 'context_id':context_id,
              'unit_id':unit_id,'unit_measures':units.get(unit_id),
              'raw_value':None if is_nil else raw,
              'is_nil':is_nil,'decimals':el.get('decimals'),
              'precision':el.get('precision'), **contexts[context_id]}
        if unit_id and not is_nil:
            try: data['numeric_value']=str(Decimal(raw))
            except InvalidOperation:
                raise ValueError(f'Invalid numeric fact {data["concept"]}: {raw!r}')
        out.append(data)
    return out


def main():
    p=argparse.ArgumentParser(description='Extract original context/units from supplied XBRL XML.')
    p.add_argument('instance_xml')
    p.add_argument('--out',required=True)
    a=p.parse_args()
    facts=parse_instance(a.instance_xml)
    Path(a.out).write_text(json.dumps(facts,indent=2,ensure_ascii=False)+'\n')
    print(f'Extracted {len(facts)} raw facts; no financial metric mapping inferred')

if __name__=='__main__': main()
