from __future__ import annotations

import unittest

from computational_study_agent.adaptive.entanglement_active_space import (
    build_entanglement_active_space_review_plan,
)
from computational_study_agent.gates.evaluators import active_space_approval_items
from computational_study_agent.gates import build_report_workflow
from computational_study_agent.schema import StudyCase, StudyPlan


def _request():
    return {
        'task_type': 'molecular',
        'system': {
            'atom': 'N 0 0 0; N 0 0 1.2',
            'basis': 'sto-3g',
            'spin': 0,
        },
        'method': 'casscf',
        'solver': {
            'name': 'block2_dmrg',
            'options': {'preset': 'screening'},
        },
        'active_space': {
            'enabled': True,
            'ncas': 4,
            'nelecas': [2, 2],
            'orbital_indices': [2, 3, 4, 5],
            'approved': True,
        },
    }


def _plan():
    return StudyPlan(
        study_id='entanglement-study',
        name='entanglement-study',
        objective='review a saturated active space',
        system_type='molecular',
        cases=[StudyCase(
            'case-0001',
            'R=1.2',
            _request(),
            variables={'bond_length': 1.2},
        )],
        observables=['energy'],
    )


def _manifest():
    return {
        'schema': 'pyscf-agent.block2-mps-manifest.v1',
        'scratch_directory': '/shared/entanglement-study/case-0001',
        'n_orbitals': 4,
        'n_electrons': [2, 2],
        'spin': 0,
        'symmetry': 'su2',
        'nroots': 1,
        'mps_tag': 'GS',
        'files': [
            {'relative_path': 'GS-MPS_INFO'},
            {'relative_path': 'optimized-mo-coeff.npy'},
        ],
        'orbital_context': {
            'kind': 'casscf_optimized_orbitals_and_mps',
            'external_restart_supported': True,
            'ncas': 4,
            'nelecas': [2, 2],
        },
        'checkpoint_components': ['mps', 'optimized_orbitals'],
    }


def _recommendation():
    active_space = {
        'enabled': True,
        'selection_method': 'manual',
        'ncas': 6,
        'nelecas': [3, 3],
        'orbital_indices': [1, 2, 3, 4, 5, 6],
        'orbital_index_basis': 'checkpoint_optimized_mo',
        'approved': False,
        'audit': {
            'manual_approval': {'approved': False, 'status': 'requires_user_review'},
        },
    }
    return {
        'schema': 'pyscf-agent.entanglement-active-space-recommendation.v1',
        'status': 'approval_recommended',
        'reason': 'Both occupied and virtual active-space boundaries remain fractional.',
        'current_active_space': {'ncas': 4, 'nelecas': [2, 2], 'ncore': 2},
        'candidate_active_space': active_space,
    }


def _report():
    return {
        'study_id': 'entanglement-study',
        'system_type': 'molecular',
        'status': 'succeeded',
        'cases': [{
            'case_id': 'case-0001',
            'task_report': {
                'execution_status': 'succeeded',
                'structured_results': {
                    'entanglement_active_space_recommendation': _recommendation(),
                    'cas_result': {
                        'solver': 'block2_dmrg',
                        'dmrg_result': {'checkpoint_manifest': _manifest()},
                    },
                },
            },
        }],
    }


class EntanglementActiveSpaceTests(unittest.TestCase):
    def test_builder_creates_reviewable_orbital_only_restart(self):
        payload = build_entanglement_active_space_review_plan(_plan(), _report())

        self.assertIsNotNone(payload)
        plan = payload['entanglement_active_space_plan']
        request = plan['cases'][0]['request']
        options = request['solver']['options']
        self.assertEqual(plan['study_id'], 'entanglement-active-space-review')
        self.assertFalse(request['active_space']['approved'])
        self.assertEqual(request['active_space']['ncas'], 6)
        self.assertIn('orbital_restart_manifest', options)
        self.assertNotIn('restart_manifest', options)
        self.assertEqual(
            options['orbital_restart_provenance']['selection'],
            'entanglement_active_space_expansion',
        )
        self.assertIn('cost_estimate', plan)

    def test_builder_accepts_structured_taskspec_method(self):
        plan = _plan()
        plan.cases[0].request['method'] = {
            'name': 'casscf',
            'restricted': True,
        }

        payload = build_entanglement_active_space_review_plan(plan, _report())

        self.assertIsNotNone(payload)
        request = payload['entanglement_active_space_plan']['cases'][0]['request']
        self.assertEqual(request['method']['name'], 'casscf')
        self.assertFalse(request['active_space']['approved'])

    def test_review_plan_uses_shared_active_space_approval_gate(self):
        payload = build_entanglement_active_space_review_plan(_plan(), _report())
        report = _report()
        report['adaptive'] = {
            'entanglement_active_space_plan': payload['entanglement_active_space_plan'],
            'entanglement_active_space_decisions': payload[
                'entanglement_active_space_decisions'
            ],
        }

        items = active_space_approval_items(report)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]['plan_kind'], 'entanglement_active_space')
        self.assertEqual(items[0]['active_space']['ncas'], 6)

        workflow = build_report_workflow(report, analysis_requested=True)
        self.assertEqual(workflow['stage'], 'active_space_review_required')
        self.assertEqual(workflow['active_plan_kind'], 'entanglement_active_space')
        self.assertIn(
            'approve_active_space',
            [action['id'] for action in workflow['allowed_actions']],
        )

    def test_builder_requires_joint_optimized_orbital_checkpoint(self):
        report = _report()
        manifest = report['cases'][0]['task_report']['structured_results'][
            'cas_result'
        ]['dmrg_result']['checkpoint_manifest']
        manifest.pop('orbital_context')

        self.assertIsNone(
            build_entanglement_active_space_review_plan(_plan(), report)
        )


if __name__ == '__main__':
    unittest.main()
