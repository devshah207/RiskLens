"""No-network verification of the Pinecone/cloud acceptance guardrails."""
from pathlib import Path
import importlib.util
from types import SimpleNamespace
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'phase7_cloud_acceptance.py'
    spec = importlib.util.spec_from_file_location('phase7_cloud_acceptance', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PC:
    def __init__(self, *, dimension=384, metric='cosine', ready=True, embedded=None):
        self.info = SimpleNamespace(schema=SimpleNamespace(fields={'embedding': SimpleNamespace(
            dimension=dimension, metric=metric)}), status=SimpleNamespace(ready=ready), embed=embedded)
        self.indexes = SimpleNamespace(describe=lambda name: self.info)
        self.index = lambda name: 'audited-index'


def test_dimension_and_metric_verified_before_index_access():
    mod = _module()
    assert mod.index_contract(PC()) == 'audited-index'
    with pytest.raises(RuntimeError, match='contract'):
        mod.index_contract(PC(dimension=1024))
    with pytest.raises(RuntimeError, match='contract'):
        mod.index_contract(PC(metric='dotproduct'))
    with pytest.raises(RuntimeError, match='contract'):
        mod.index_contract(PC(embedded='hosted-model'))


def test_index_must_be_ready():
    with pytest.raises(RuntimeError, match='not ready'):
        _module().index_contract(PC(ready=False))


def test_uncertain_financial_pages_quarantined():
    mod = _module()
    assert mod.QUARANTINED == {'RIL': {73}, 'HDFCBANK': {295}, 'INFY': {184}, 'ICICIBANK': {29}}
    assert mod.NAMESPACE == 'risklens-research'
    assert mod.MODEL == 'sentence-transformers/all-MiniLM-L6-v2'


def test_gold_candidates_20_unverified():
    import json
    path = Path(__file__).resolve().parents[1] / 'docs' / 'phase7_gold_question_candidates.json'
    if not path.is_file():
        pytest.skip('Optional Phase 7 research-gold JSON is not in this checkout')
    questions = json.loads(path.read_text())['questions']
    assert len(questions) == 20
    assert sum(not q['answerable'] for q in questions) == 4
    assert all('verified_by' not in q for q in questions)
