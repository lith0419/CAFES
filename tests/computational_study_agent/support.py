from __future__ import annotations

import json
import unittest
from pathlib import Path

from computational_study_agent import StudyCase, StudyPlan


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STUDY_WEB_ASSET_DIR = REPOSITORY_ROOT / 'computational_study_agent' / 'web_assets'
SHARED_WEB_ASSET_DIR = REPOSITORY_ROOT / 'pyscf_agent' / 'web_assets'
PLANNER_SOURCE_ASSETS = (
    (STUDY_WEB_ASSET_DIR, 'planner.css'),
    (SHARED_WEB_ASSET_DIR, 'molecular-preview.css'),
    (SHARED_WEB_ASSET_DIR, 'molecular-preview.js'),
    (STUDY_WEB_ASSET_DIR, 'planner-core.js'),
    (STUDY_WEB_ASSET_DIR, 'planner-spec.js'),
    (STUDY_WEB_ASSET_DIR, 'planner-dataset.js'),
    (STUDY_WEB_ASSET_DIR, 'planner-postprocessing.js'),
    (STUDY_WEB_ASSET_DIR, 'planner-adaptive-decisions.js'),
    (STUDY_WEB_ASSET_DIR, 'planner-adaptive-review.js'),
    (STUDY_WEB_ASSET_DIR, 'planner-rendering.js'),
    (STUDY_WEB_ASSET_DIR, 'planner-saved-studies.js'),
    (STUDY_WEB_ASSET_DIR, 'planner-dmet.js'),
    (STUDY_WEB_ASSET_DIR, 'planner-init.js'),
)


def planner_ui_source(page_html: str) -> str:
    sources = [page_html]
    sources.extend(
        (asset_dir / asset_name).read_text(encoding='utf-8')
        for asset_dir, asset_name in PLANNER_SOURCE_ASSETS
    )
    return '\n'.join(sources)


def json_dumps(payload):
    return json.dumps(payload, ensure_ascii=False).encode('utf-8')


def json_dumps_text(payload):
    return json.dumps(payload, ensure_ascii=False).encode('utf-8')


def json_loads(body):
    return json.loads(body.decode('utf-8'))


def hubbard_dimer_spec():
    return {
        'schema': 'pyscf-agent.model-hamiltonian.v1',
        'model': 'hubbard',
        'dimension': 1,
        'preset': 'chain',
        'boundary': 'open',
        'energy_unit': 'a.u.',
        'nelec': [1, 1],
        'sites': [
            {'id': 0, 'x': 0, 'y': 0, 'epsilon': 0, 'U': 4},
            {'id': 1, 'x': 1, 'y': 0, 'epsilon': 0, 'U': 4},
        ],
        'bonds': [
            {'id': 0, 'source': 0, 'target': 1, 't': -1, 'V': 0, 'effective_t': -1, 'effective_V': 0},
        ],
    }


def disconnected_bond_spec():
    spec = hubbard_dimer_spec()
    spec['sites'] = [
        {'id': 0, 'x': 0, 'y': 0, 'epsilon': 0, 'U': 4},
        {'id': 1, 'x': 1, 'y': 0, 'epsilon': 0, 'U': 4},
        {'id': 2, 'x': 2, 'y': 0, 'epsilon': 0, 'U': 4},
        {'id': 4, 'x': 3, 'y': 0, 'epsilon': 0, 'U': 4},
    ]
    spec['bonds'] = [
        {'id': 0, 'source': 0, 'target': 1, 't': -1, 'V': 0, 'effective_t': -1, 'effective_V': 0},
        {'id': 1, 'source': 2, 'target': 4, 't': -1, 'V': 0, 'effective_t': -1, 'effective_V': 0},
    ]
    spec['nelec'] = [2, 2]
    return spec


class StudyAgentTestCase(unittest.TestCase):
    @staticmethod
    def _resumable_molecular_plan(method='mp2'):
        return StudyPlan(
            study_id='resumable-molecular-study',
            name='resumable molecular study',
            objective='verify checkpoint reuse',
            system_type='molecular',
            cases=[
                StudyCase(
                    case_id='case-0001',
                    label='H2 first point',
                    request={
                        'atom': 'H 0 0 0; H 0 0 0.74',
                        'basis': 'sto-3g',
                        'method': method,
                    },
                    variables={'bond_length': 0.74},
                ),
                StudyCase(
                    case_id='case-0002',
                    label='H2 second point',
                    request={
                        'atom': 'H 0 0 0; H 0 0 1.00',
                        'basis': 'sto-3g',
                        'method': method,
                    },
                    variables={'bond_length': 1.00},
                ),
            ],
            observables=['energy'],
        )

    @staticmethod
    def _mock_succeeded_workflow(state):
        return {
            'task_report': {
                'execution_status': 'succeeded',
                'work_dir': state['work_dir'],
                'run_id': state['run_id'],
                'structured_results': {
                    'task_type': 'molecular',
                    'method': 'mp2',
                    'energy': -1.0,
                    'final_energy': -1.0,
                },
            },
        }

    @staticmethod
    def _mock_failed_workflow(state):
        return {
            'task_report': {
                'execution_status': 'failed',
                'work_dir': state['work_dir'],
                'run_id': state['run_id'],
                'structured_results': {'task_type': 'molecular', 'method': 'mp2'},
            },
        }
