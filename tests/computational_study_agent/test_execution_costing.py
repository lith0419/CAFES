from __future__ import annotations

import json
import tempfile
import unittest.mock
from pathlib import Path

from computational_study_agent import StudyCase, StudyPlan
from computational_study_agent.executor import run_study
from computational_study_agent.gates.presentation import build_initial_plan_workflow, build_report_workflow
from computational_study_agent.costing import (
    CostApprovalRequired,
    ensure_plan_cost_estimate,
    estimate_plan_cost,
    require_plan_cost_approval,
)
from computational_study_agent.validation import validate_study_plan
from tests.computational_study_agent.support import StudyAgentTestCase


class StudyExecutionAndCostingTests(StudyAgentTestCase):
    def test_block2_active_space_cost_uses_tensor_network_proxy(self):
        case = StudyCase(
            case_id='case-dmrg',
            label='DMRG CAS(18, 18)',
            request={
                'task_type': 'molecular',
                'atom': 'H 0 0 0; H 0 0 1',
                'basis': 'sto-3g',
                'method': 'casci',
                'solver': {
                    'name': 'block2_dmrg',
                    'options': {
                        'bond_dimensions': [100, 250],
                        'sweeps': 8,
                        'nroots': 2,
                    },
                },
                'active_space': {'ncas': 18, 'nelecas': 18},
            },
        )

        estimate = estimate_plan_cost([case], 'molecular')
        item = estimate['cases'][0]
        self.assertEqual(item['solver'], 'block2_dmrg')
        self.assertEqual(item['estimation_model'], 'block2_dmrg_sweep_proxy')
        self.assertEqual(item['active_norb'], 18)
        self.assertEqual(item['bond_dimension'], 250)
        self.assertEqual(item['sweeps'], 8)
        self.assertEqual(item['nroots'], 2)
        self.assertGreater(item['work_units'], 0)
        self.assertGreater(item['memory_mb'], 0)

    def test_dmet_block2_cost_uses_embedded_impurity_dimensions(self):
        model_spec = {
            'model': 'hubbard',
            'nelec': [16, 16],
            'primitive_cell': {'basis_size': 2},
            'sites': [
                {'id': index, 'x': index, 'y': 0}
                for index in range(32)
            ],
            'bonds': [],
        }
        case = StudyCase(
            case_id='case-dmet-dmrg',
            label='DMET with a block2 impurity solver',
            request={
                'task_type': 'model_hamiltonian',
                'solver': {
                    'name': 'dmet',
                    'options': {
                        'impurity_solver': 'block2_dmrg',
                        'impurity_shape': [2, 1],
                        'impurity_solver_options': {'preset': 'screening'},
                    },
                },
                'model_hamiltonian': {'spec': model_spec},
            },
        )

        estimate = estimate_plan_cost([case], 'model_hamiltonian')
        item = estimate['cases'][0]

        self.assertEqual(item['method'], 'dmet')
        self.assertEqual(item['solver'], 'dmet')
        self.assertEqual(item['impurity_solver'], 'block2_dmrg')
        self.assertEqual(item['estimation_model'], 'dmet_block2_impurity_sweep_proxy')
        self.assertEqual(item['fragment_norb'], 4)
        self.assertEqual(item['norb'], 8)
        self.assertEqual(item['dimension_source'], 'dmet_fragment_plus_bath_upper_bound')
        self.assertEqual(item['bond_dimension'], 200)
        self.assertGreater(item['memory_mb'], 0)

    def test_block2_work_proxy_is_informational_unless_policy_enables_it(self):
        case = StudyCase(
            case_id='case-dmrg',
            label='DMRG CAS(18, 18)',
            request={
                'task_type': 'molecular',
                'atom': 'H 0 0 0; H 0 0 1',
                'basis': 'sto-3g',
                'method': 'casci',
                'solver': {'name': 'block2_dmrg', 'options': {'bond_dimensions': [100]}},
                'active_space': {'ncas': 18, 'nelecas': 18},
            },
        )

        default_estimate = estimate_plan_cost([case], 'molecular')
        reviewed_estimate = estimate_plan_cost(
            [case],
            'molecular',
            {'total_work_review_threshold': 1, 'review_work_estimates': True},
        )

        self.assertGreater(default_estimate['totals']['work_units'], 10_000_000)
        self.assertFalse(default_estimate['approval_required'])
        self.assertTrue(default_estimate['can_execute'])
        self.assertTrue(reviewed_estimate['approval_required'])

    def test_slurm_memory_limit_is_a_hard_gate(self):
        case = StudyCase(
            case_id='case-dmrg',
            label='DMRG CAS(18, 18)',
            request={
                'task_type': 'molecular',
                'atom': 'H 0 0 0; H 0 0 1',
                'basis': 'sto-3g',
                'method': 'casci',
                'solver': {'name': 'block2_dmrg', 'options': {'bond_dimensions': [250]}},
                'active_space': {'ncas': 18, 'nelecas': 18},
            },
        )

        blocked = estimate_plan_cost(
            [case],
            'molecular',
            {'memory_limit_mb': 128, 'memory_limit_source': 'slurm_profile'},
        )
        allowed = estimate_plan_cost(
            [case],
            'molecular',
            {'memory_limit_mb': 256, 'memory_limit_source': 'slurm_profile'},
        )

        self.assertTrue(blocked['resource_limit_exceeded'])
        self.assertFalse(blocked['approval_allowed'])
        self.assertFalse(blocked['can_execute'])
        self.assertEqual(blocked['status'], 'blocked')
        self.assertFalse(allowed['approval_required'])
        self.assertTrue(allowed['can_execute'])

    def test_active_space_approval_does_not_invalidate_cost_approval(self):
        plan = StudyPlan(
            study_id='cas-cost-fingerprint',
            name='cas-cost-fingerprint',
            objective='keep scientific and resource approvals independent',
            system_type='molecular',
            cases=[StudyCase(
                case_id='case-0001',
                label='CAS(14, 14)',
                request={
                    'task_type': 'molecular',
                    'atom': 'N 0 0 0; N 0 0 1.1',
                    'basis': 'sto-3g',
                    'method': 'casscf',
                    'active_space': {'ncas': 14, 'nelecas': 14, 'approved': False},
                },
            )],
            observables=['energy'],
        )
        first = ensure_plan_cost_estimate(plan)
        self.assertTrue(first['approval_required'])
        first['approved'] = True
        first['policy']['approved'] = True
        plan.resource_policy['approved'] = True

        plan.cases[0].request['active_space']['approved'] = True
        refreshed = ensure_plan_cost_estimate(plan)

        self.assertTrue(refreshed['approved'])
        self.assertNotIn('approval_invalidated', refreshed)

    def test_study_checkpoint_reuses_succeeded_cases_and_invalidates_changed_case(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = self._resumable_molecular_plan()
            with unittest.mock.patch(
                'pyscf_agent.executors.local.run_workflow_sequential',
                side_effect=self._mock_succeeded_workflow,
            ) as execute:
                first_report = run_study(plan, work_dir=tmpdir)
                resumed_report = run_study(plan, work_dir=tmpdir)
                changed_plan = self._resumable_molecular_plan()
                changed_plan.cases[0].request['method'] = 'ccsd'
                changed_report = run_study(changed_plan, work_dir=tmpdir)

            self.assertEqual(execute.call_count, 3)
            self.assertEqual(first_report.execution['executor']['executor_id'], 'local')
            self.assertEqual(first_report.execution['executor']['execution_mode'], 'synchronous')
            self.assertEqual(first_report.execution['counts']['executed'], 2)
            self.assertEqual(resumed_report.execution['counts']['resumed'], 2)
            changed_sources = {
                row['case_id']: row['execution_source']
                for row in changed_report.comparison_table
            }
            self.assertEqual(changed_sources['case-0001'], 'executed')
            self.assertEqual(changed_sources['case-0002'], 'resumed')
            state_path = Path(tmpdir) / plan.study_id / 'study-state.json'
            state_payload = json.loads(state_path.read_text(encoding='utf-8'))
            self.assertEqual(state_payload['schema'], 'pyscf-agent.study-state.v1')
            self.assertEqual(len(state_payload['cases']), 2)

    def test_study_resolves_adjacent_case_1rdm_before_execution(self):
        plan = self._resumable_molecular_plan(method='ccsd')
        plan.study_id = 'adjacent-1rdm-continuation'
        plan.cases[1].request['initial_state'] = {
            'mode': 'projected_1rdm',
            'source_case_id': 'case-0001',
            'source_artifact': {
                'kind': 'one_particle_state',
                'deferred_case_id': 'case-0001',
            },
        }
        executor = unittest.mock.Mock()
        executor.describe.return_value = {
            'executor_id': 'test',
            'execution_mode': 'synchronous',
        }

        def execute_task(request_text, **kwargs):
            request = json.loads(request_text)
            run_id = kwargs['run_id']
            if run_id == 'case-0001':
                self.assertNotIn('initial_state', request)
                artifact = {
                    'kind': 'one_particle_state',
                    'path': '/tmp/generated-case-0001.npz',
                }
            else:
                initial_state = request['initial_state']
                self.assertEqual(initial_state['source_case_id'], 'case-0001')
                self.assertEqual(
                    initial_state['source_artifact']['path'],
                    '/tmp/generated-case-0001.npz',
                )
                artifact = {
                    'kind': 'one_particle_state',
                    'path': '/tmp/generated-case-0002.npz',
                }
            return {
                'execution_status': 'succeeded',
                'work_dir': kwargs['work_dir'],
                'run_id': run_id,
                'structured_results': {
                    'task_type': 'molecular',
                    'method': 'ccsd',
                    'final_energy': -1.0,
                },
                'artifacts': [artifact],
            }

        executor.execute_task.side_effect = execute_task
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_study(
                plan,
                work_dir=tmpdir,
                task_executor=executor,
            )

        self.assertEqual(executor.execute_task.call_count, 2)
        self.assertEqual(report.status, 'succeeded')
        resolved_state = report.cases[1]['request']['initial_state']
        self.assertEqual(
            resolved_state['source_artifact']['path'],
            '/tmp/generated-case-0001.npz',
        )

    def test_study_checkpoint_limits_automatic_failed_case_retries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = self._resumable_molecular_plan()
            plan.cases = [plan.cases[0]]
            with unittest.mock.patch(
                'pyscf_agent.executors.local.run_workflow_sequential',
                side_effect=self._mock_failed_workflow,
            ) as execute:
                run_study(plan, work_dir=tmpdir, max_case_attempts=2)
                run_study(plan, work_dir=tmpdir, max_case_attempts=2)
                limited_report = run_study(plan, work_dir=tmpdir, max_case_attempts=2)
                forced_report = run_study(
                    plan,
                    work_dir=tmpdir,
                    max_case_attempts=2,
                    rerun_case_ids=['case-0001'],
                )

            self.assertEqual(execute.call_count, 3)
            self.assertEqual(limited_report.execution['counts']['attempt_limit_reached'], 1)
            self.assertEqual(limited_report.cases[0]['attempt_count'], 2)
            self.assertEqual(forced_report.cases[0]['attempt_count'], 3)
            approval_sources = [
                item.get('details', {}).get('source')
                for item in forced_report.lifecycle['history']
                if item.get('event') == 'approval_granted'
            ]
            self.assertIn('automatic_retry', approval_sources)
            self.assertIn('explicit_rerun', approval_sources)

    def test_cost_estimate_gates_large_fci_plan_before_execution(self):
        model_spec = {
            'model': 'hubbard',
            'nelec': [12, 12],
            'sites': [{'id': index, 'x': index, 'y': 0} for index in range(24)],
            'bonds': [],
        }
        case = StudyCase(
            case_id='case-0001',
            label='large fci',
            request={
                'task_type': 'model_hamiltonian',
                'solver': 'fci',
                'model_hamiltonian': {'spec': model_spec},
            },
            model_spec=model_spec,
        )
        plan = StudyPlan(
            study_id='cost-gate',
            name='cost-gate',
            objective='validate cost approval',
            system_type='model_hamiltonian',
            cases=[case],
            observables=['energy'],
        )
        estimate = estimate_plan_cost(plan.cases, plan.system_type)
        self.assertTrue(estimate['approval_required'])
        self.assertFalse(estimate['can_execute'])
        self.assertGreater(estimate['cases'][0]['determinant_count'], 1_000_000)
        with self.assertRaises(CostApprovalRequired):
            run_study(plan)

    def test_explicit_resource_approval_allows_a_cost_gated_plan_to_reach_execution(self):
        model_spec = {
            'model': 'hubbard',
            'nelec': [12, 12],
            'sites': [{'id': index, 'x': index, 'y': 0} for index in range(24)],
            'bonds': [],
        }
        case = StudyCase(
            case_id='case-0001',
            label='large fci',
            request={'task_type': 'model_hamiltonian', 'solver': 'fci', 'model_hamiltonian': {'spec': model_spec}},
            model_spec=model_spec,
        )
        plan = StudyPlan(
            study_id='cost-approved',
            name='cost-approved',
            objective='validate cost approval',
            system_type='model_hamiltonian',
            cases=[case],
            observables=['energy'],
            resource_policy={'approved': True},
        )
        with unittest.mock.patch('pyscf_agent.executors.local.run_workflow_sequential', side_effect=RuntimeError('executor reached')):
            with tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaisesRegex(RuntimeError, 'executor reached'):
                    run_study(plan, work_dir=tmpdir)

    def test_full_fci_diagonalization_uses_dense_cost_and_limit(self):
        def model_spec(site_count, nelec):
            return {
                'model': 'hubbard',
                'nelec': nelec,
                'sites': [{'id': index, 'x': index, 'y': 0} for index in range(site_count)],
                'bonds': [],
            }

        supported_case = StudyCase(
            case_id='case-0001',
            label='8-site full spectrum',
            request={
                'task_type': 'model_hamiltonian',
                'solver': 'fci',
                'model_hamiltonian': {'spec': model_spec(8, [4, 4])},
                'analysis': {'outputs': ['energy', 'strong_correlation_diagnostics']},
            },
        )
        supported_estimate = estimate_plan_cost([supported_case], 'model_hamiltonian')
        supported_cost = supported_estimate['cases'][0]
        self.assertEqual(supported_cost['estimation_model'], 'fci_full_diagonalization')
        self.assertTrue(supported_cost['full_diagonalization_supported'])
        self.assertGreater(supported_cost['memory_mb'], 512)
        self.assertTrue(supported_estimate['approval_required'])

        unsupported_plan = StudyPlan(
            study_id='full-diag-limit',
            name='full-diag-limit',
            objective='reject excessive full diagonalization',
            system_type='model_hamiltonian',
            cases=[StudyCase(
                case_id='case-0001',
                label='10-site full spectrum',
                request={
                    'task_type': 'model_hamiltonian',
                    'solver': 'fci',
                    'model_hamiltonian': {'spec': model_spec(10, [5, 5])},
                    'analysis': {'outputs': ['strong_correlation_diagnostics']},
                },
            )],
            observables=['strong_correlation_diagnostics'],
        )
        issues = validate_study_plan(unsupported_plan)
        self.assertTrue(any(issue.code == 'unsupported_full_fci_diagonalization' for issue in issues))

    def test_cost_approval_is_invalidated_when_executable_plan_changes(self):
        model_spec = {
            'model': 'hubbard',
            'nelec': [4, 4],
            'sites': [{'id': index, 'x': index, 'y': 0, 'U': 4} for index in range(8)],
            'bonds': [],
        }
        plan = StudyPlan(
            study_id='cost-fingerprint',
            name='cost-fingerprint',
            objective='verify cost approval binding',
            system_type='model_hamiltonian',
            cases=[StudyCase(
                case_id='case-0001',
                label='full spectrum',
                request={
                    'task_type': 'model_hamiltonian',
                    'solver': 'fci',
                    'model_hamiltonian': {'spec': model_spec},
                    'analysis': {'outputs': ['strong_correlation_diagnostics']},
                },
            )],
            observables=['strong_correlation_diagnostics'],
        )
        first = ensure_plan_cost_estimate(plan)
        self.assertTrue(first['approval_required'])
        first['approved'] = True
        first['policy']['approved'] = True
        plan.resource_policy['approved'] = True
        self.assertTrue(ensure_plan_cost_estimate(plan)['approved'])

        plan.cases[0].request['model_hamiltonian']['spec']['sites'][0]['U'] = 8

        refreshed = ensure_plan_cost_estimate(plan)
        self.assertTrue(refreshed['approval_invalidated'])
        self.assertFalse(refreshed['approved'])
        with self.assertRaises(CostApprovalRequired):
            require_plan_cost_approval(plan)

    def test_adaptive_workflow_exposes_cost_review_before_execution(self):
        estimate = {
            'approval_required': True,
            'approved': False,
            'review_reason': 'case-0001: determinant space reaches the review threshold',
            'cases': [{
                'case_id': 'case-0001',
                'label': 'large fci',
                'method': 'fci',
                'determinant_count': 2_000_000,
                'memory_mb': 128.0,
            }],
        }
        plan = {'cases': [{'case_id': 'case-0001'}], 'cost_estimate': estimate}
        initial_workflow = build_initial_plan_workflow(plan)
        self.assertEqual(initial_workflow['stage'], 'cost_review_required')
        self.assertEqual(initial_workflow['allowed_actions'][0]['id'], 'approve_cost_estimate')

        report = {
            'system_type': 'molecular',
            'adaptive': {
                'refined_plan': plan,
                'recovery_plan': None,
            },
        }
        self.assertEqual(build_report_workflow(report)['stage'], 'cost_review_required')
