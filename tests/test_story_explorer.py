"""Read-only architecture explorer tests; no npm, Groq or Streamlit required."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tomllib

import pytest

from app.services.story_blueprints import build_stories, source_link
from risklens_story import frontend_is_built

ROOT = Path(__file__).resolve().parents[1]
FRONT = ROOT / 'src/risklens_story/frontend'


def test_both_stories_have_real_nodes_edges_journeys():
    stories = build_stories(ROOT)
    assert set(stories) == {'market', 'research', 'context'}
    for which in ['market', 'research']:
        graph = stories[which]
        ids = {node['id'] for node in graph['nodes']}
        assert len(ids) == 7
        assert len(graph['journey']) == 7
        assert all(edge['source'] in ids and edge['target'] in ids for edge in graph['edges'])
        assert set(graph['journey']) == ids
        for n in graph['nodes']:
            assert n['title'] and n['plain'] and n['technical'] and n['evidence']
            assert source_link(n['sourcePath']).startswith('https://github.com/devshah207/RiskLens/')
            assert (ROOT / n['sourcePath']).is_file()
            assert n['position']['x'] >= 0


def test_exact_xbrl_and_pdf_are_alternative_routes():
    graph = build_stories(ROOT)['research']
    links = {(e['source'], e['target']): e['label'] for e in graph['edges']}
    assert links['research-xbrl', 'research-answer'] == 'Exact fact'
    assert links['research-xbrl', 'research-pdf'] == 'No exact fact'
    assert ('research-answer', 'research-groq') not in links
    assert ('research-pdf', 'research-groq') in links
    assert 'Explicit consent' in next(n['subtitle'] for n in graph['nodes'] if n['id'] == 'research-groq')


def test_frozen_model_selection_and_failure_boundary_truthful():
    nodes = build_stories(ROOT)['market']['nodes']
    xgb = next(n for n in nodes if n['id'] == 'market-xgb')
    lstm = next(n for n in nodes if n['id'] == 'market-lstm')
    validation = next(n for n in nodes if n['id'] == 'market-holdout')
    assert 'champion' in xgb['technical'].lower()
    assert '#86' in lstm['technical']
    assert 'zero recall' in validation['technical'].lower()
    assert '433' in validation['subtitle']
    assert '42' in validation['subtitle']


def test_saved_report_read_only_and_real_evidence_label(tmp_path):
    actual = build_stories(ROOT)
    if (ROOT / 'data/processed/phase5_final_holdout/final_holdout_report.json').is_file():
        assert actual['context']['savedHoldoutVerified'] is True
        assert 'ROC-AUC 0.673' in next(n['evidence'] for n in actual['market']['nodes'] if n['id'] == 'market-holdout')
    else:
        assert actual['context']['savedHoldoutVerified'] is False
    missing = build_stories(tmp_path)
    assert missing['context']['savedHoldoutVerified'] is False
    assert 'not mounted' in next(n['evidence'] for n in missing['market']['nodes'] if n['id'] == 'market-holdout')
    # Malformed local evidence is rejected, not replaced by an invented metric.
    path = tmp_path / 'data/processed/phase5_final_holdout/final_holdout_report.json'
    path.parent.mkdir(parents=True)
    path.write_text('{"model_scores": {}}', encoding='utf-8')
    assert build_stories(tmp_path)['context']['savedHoldoutVerified'] is False


def test_graph_json_is_serializable_and_no_local_paths_leak():
    graph = build_stories(ROOT)
    encoded = json.dumps(graph)
    assert str(ROOT) not in encoded
    assert '/Users/' not in encoded
    assert 'GROQ_API_KEY' not in encoded
    assert 'api_key' not in encoded.lower()
    assert graph == copy.deepcopy(graph)


def test_safe_code_reference_paths():
    with pytest.raises(ValueError):
        source_link('../../secret')
    with pytest.raises(ValueError):
        source_link('https://attacker.test/steal')
    with pytest.raises(ValueError):
        source_link('src/risklens/../../.env')
    assert source_link('src/risklens/research/xbrl_rag_v1.py').endswith('xbrl_rag_v1.py')


def test_component_metadata_and_build_contract():
    metadata = tomllib.loads((ROOT/'src/risklens_story/pyproject.toml').read_text())
    assert metadata['project']['name'] == 'risklens-educational'
    assert metadata['tool']['streamlit']['component']['components'][0]['name'] == 'story_explorer'
    assert metadata['tool']['streamlit']['component']['components'][0]['asset_dir'] == 'frontend/build'
    project = tomllib.loads((ROOT/'pyproject.toml').read_text())
    assert any(dep.startswith('streamlit>=1.51') for dep in project['project']['optional-dependencies']['ui'])
    assert 'risklens_story' in project['tool']['setuptools']['package-data']
    pkg = json.loads((FRONT/'package.json').read_text())
    assert '@xyflow/react' in pkg['dependencies']
    assert '@streamlit/component-v2-lib' in pkg['dependencies']
    assert 'typescript' in pkg['devDependencies']
    if frontend_is_built():
        assert len(list((FRONT/'build').glob('index-*.js'))) == 1
        assert (FRONT/'build/index.css').is_file()


def test_architecture_ui_mentions_interactions_and_accessibility():
    source = (FRONT/'src/Explorer.tsx').read_text()
    for text in ('ReactFlow', 'onNodeClick', 'Play guided tour', 'useReducedMotion',
                 'ArrowRight', 'ArrowLeft', 'MiniMap', 'Controls', 'View whole map',
                 'Technical', 'aria-label'):
        assert text in source
    assert 'fetch(' not in source
    assert 'setStateValue' not in source
    assert 'setTriggerValue' not in source


def test_no_research_model_invocation_in_explorer_python():
    for file in (ROOT/'app/pages/3_Explore_RiskLens.py', ROOT/'app/services/story_blueprints.py'):
        source = file.read_text()
        assert 'run_query_xbrl_first(' not in source
        assert 'lookup_xbrl(' not in source
        assert 'model.fit(' not in source
        assert 'evaluate_holdout(' not in source
        assert 'groq.Client(' not in source


def test_home_links_to_new_explorer_without_removing_original_pages():
    home = (ROOT/'app/Home.py').read_text()
    assert 'pages/1_Market_Risk.py' in home
    assert 'pages/2_Financial_Research.py' in home
    assert 'pages/3_Explore_RiskLens.py' in home


def test_unbuilt_explorer_page_renders_safe_preview(monkeypatch):
    """A missing npm bundle must not break the original Streamlit application."""
    import runpy
    import sys

    from tests.test_phase8_render_smoke import UIStub
    fake = UIStub()
    monkeypatch.setitem(sys.modules, 'streamlit', fake)
    if not frontend_is_built():
        runpy.run_path(str(ROOT/'app/pages/3_Explore_RiskLens.py'), run_name='__main__')
