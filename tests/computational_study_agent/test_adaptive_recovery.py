from __future__ import annotations

import tempfile
import unittest.mock

from computational_study_agent import StudyCase, StudyPlan, StudyReport
from computational_study_agent.adaptive.executor import (
    build_recovery_plan_for_unconverged_refined_cases,
    run_adaptive_study,
)
from computational_study_agent.gates.presentation import build_report_workflow
from tests.computational_study_agent.support import StudyAgentTestCase, hubbard_dimer_spec


class AdaptiveRecoveryTests(StudyAgentTestCase):
    def test_adaptive_refined_unconverged_is_rerun_with_recovery_solver(self):
        diagnostics = {
            'level': 'moderate',
            'score': 0.5,
            'confidence': 'medium',
            'summary': 'Moderate finite-size Hubbard strong-correlation tendency.',
            'parameter_summary': {'site_count': 2},
            'diagnostics': [],
        }
        initial_scan_report = StudyReport(
            study_id='initial-scan',
            name='initial-scan',
            objective='initial scan',
            system_type='model_hamiltonian',
            status='succeeded',
            work_dir='initial-scan',
            cases=[
                {
                    'case_id': 'case-0001',
                    'label': 'U=8',
                    'variables': {'U': 8},
                    'request': {'solver': 'mp2', 'model_hamiltonian': {'spec': hubbard_dimer_spec()}},
                    'task_report': {
                        'execution_status': 'succeeded',
                        'structured_results': {'strong_correlation_diagnostics': diagnostics},
                    },
                },
            ],
            comparison_table=[
                {'case_id': 'case-0001', 'label': 'U=8', 'status': 'succeeded', 'U': '8 a.u.', 'solver': 'mp2', 'energy': '-1.0 a.u.'},
            ],
        )
        refined_report = StudyReport(
            study_id='refined',
            name='refined',
            objective='refined',
            system_type='model_hamiltonian',
            status='completed_with_issues',
            work_dir='refined',
            cases=[
                {
                    'case_id': 'case-0001',
                    'label': 'U=8',
                    'variables': {'U': 8},
                    'request': {'solver': 'ccsd'},
                    'task_report': {
                        'execution_status': 'unconverged',
                        'raw_stderr': 'SCF did not converge.',
                        'structured_results': {'energy': -0.5, 'solver': 'ccsd'},
                    },
                },
            ],
            comparison_table=[
                {'case_id': 'case-0001', 'label': 'U=8', 'status': 'unconverged', 'U': '8 a.u.', 'solver': 'ccsd', 'energy': '-0.5 a.u.'},
            ],
        )
        recovery_report = StudyReport(
            study_id='refined-recovery',
            name='refined-recovery',
            objective='recovery',
            system_type='model_hamiltonian',
            status='succeeded',
            work_dir='recovery',
            cases=[
                {
                    'case_id': 'case-0001',
                    'label': 'U=8',
                    'variables': {'U': 8},
                    'request': {'solver': 'fci'},
                    'task_report': {
                        'execution_status': 'succeeded',
                        'structured_results': {'energy': -0.8, 'solver': 'fci'},
                    },
                },
            ],
            comparison_table=[
                {'case_id': 'case-0001', 'label': 'U=8', 'status': 'succeeded', 'U': '8 a.u.', 'solver': 'fci', 'energy': '-0.8 a.u.'},
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir, unittest.mock.patch(
            'computational_study_agent.adaptive.executor.run_study',
            side_effect=[initial_scan_report, refined_report, recovery_report],
        ) as mock_run:
            report = run_adaptive_study({
                'name': 'adaptive-recovery',
                'objective': 'recover unconverged refined solver',
                'system_type': 'model_hamiltonian',
                'base_model_spec': hubbard_dimer_spec(),
                'base_task': {'solver': 'mp2'},
                'sweep': {'U': [8]},
                'observables': ['energy'],
            }, work_dir=tmpdir, locale='en', options={})

        self.assertEqual(mock_run.call_count, 3)
        self.assertTrue(mock_run.call_args_list[0].kwargs['batch_independent'])
        self.assertTrue(mock_run.call_args_list[1].kwargs['batch_independent'])
        self.assertTrue(mock_run.call_args_list[2].kwargs['batch_independent'])
        self.assertEqual(report['status'], 'succeeded')
        self.assertEqual(report['comparison_table'][0]['status'], 'succeeded')
        self.assertEqual(report['comparison_table'][0]['solver'], 'fci')
        self.assertEqual(report['comparison_table'][0]['initial_status'], 'unconverged')
        self.assertTrue(report['comparison_table'][0]['recovery_applied'])
        self.assertEqual(report['adaptive']['recovery_decisions'][0]['recovery_solver'], 'fci')
        self.assertEqual(report['adaptive']['recovery_decisions'][0]['level'], 'strong')
        self.assertEqual(report['adaptive']['recovery_decisions'][0]['solver_stress_level'], 'strong')
        self.assertEqual(report['adaptive']['decision_log'][0]['initial_refined_solver'], 'ccsd')
        self.assertEqual(report['adaptive']['decision_log'][0]['recovery_solver'], 'fci')
        self.assertEqual(report['adaptive']['decision_log'][0]['level'], 'strong')
        self.assertEqual(report['adaptive']['decision_log'][0]['solver_stress_level'], 'strong')
        self.assertIn('solver_failure_strong_correlation_evidence', report['adaptive']['decision_log'][0]['tags'])
        self.assertIn('method_recovery_applied', report['adaptive']['decision_log'][0]['tags'])

    def test_molecular_adaptive_refined_unconverged_without_audit_uses_small_fci_recovery(self):
        diagnostics = {
            'kind': 'molecular_correlation_diagnostics',
            'level': 'moderate',
            'score': 0.55,
            'confidence': 'medium',
            'molecular_correlation_risk': {
                'kind': 'molecular_correlation_risk',
                'level': 'moderate',
                'overall_score': 0.55,
                'physics_score': 0.42,
                'solver_stress_score': 0.60,
                'method_recommendation': {'preferred': ['ccsd']},
            },
        }
        initial_scan_report = StudyReport(
            study_id='initial-scan',
            name='initial-scan',
            objective='initial scan',
            system_type='molecular',
            status='succeeded',
            work_dir='initial-scan',
            cases=[
                {
                    'case_id': 'case-0001',
                    'label': 'stretched H2',
                    'variables': {},
                    'request': {'task_type': 'molecular', 'method': 'mp2'},
                    'task_report': {
                        'execution_status': 'succeeded',
                        'structured_results': {
                            'task_type': 'molecular',
                            'method': 'mp2',
                            'energy': -1.0,
                            'correlation_diagnostics': diagnostics,
                        },
                    },
                },
            ],
            comparison_table=[
                {'case_id': 'case-0001', 'label': 'stretched H2', 'status': 'succeeded', 'method': 'mp2', 'final_energy': '-1.0 Ha'},
            ],
        )
        refined_report = StudyReport(
            study_id='refined',
            name='refined',
            objective='refined',
            system_type='molecular',
            status='completed_with_issues',
            work_dir='refined',
            cases=[
                {
                    'case_id': 'case-0001',
                    'label': 'stretched H2',
                    'variables': {},
                    'request': {'task_type': 'molecular', 'method': 'ccsd'},
                    'task_report': {
                        'execution_status': 'unconverged',
                        'raw_stderr': 'SCF did not converge.',
                        'structured_results': {'task_type': 'molecular', 'method': 'ccsd', 'energy': -1.1},
                    },
                },
            ],
            comparison_table=[
                {'case_id': 'case-0001', 'label': 'stretched H2', 'status': 'unconverged', 'method': 'ccsd', 'final_energy': '-1.1 Ha'},
            ],
        )
        recovery_report = StudyReport(
            study_id='refined-recovery',
            name='refined-recovery',
            objective='recovery',
            system_type='molecular',
            status='succeeded',
            work_dir='recovery',
            cases=[
                {
                    'case_id': 'case-0001',
                    'label': 'stretched H2',
                    'variables': {},
                    'request': {
                        'task_type': 'molecular',
                        'method': 'fci',
                    },
                    'task_report': {
                        'execution_status': 'succeeded',
                        'structured_results': {'task_type': 'molecular', 'method': 'fci', 'energy': -1.12},
                    },
                },
            ],
            comparison_table=[
                {'case_id': 'case-0001', 'label': 'stretched H2', 'status': 'succeeded', 'method': 'fci'},
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir, unittest.mock.patch(
            'computational_study_agent.adaptive.executor.run_study',
            side_effect=[initial_scan_report, refined_report, recovery_report],
        ) as mock_run:
            report = run_adaptive_study({
                'name': 'adaptive-molecular-recovery',
                'objective': 'recover unconverged molecular refined method',
                'system_type': 'molecular',
                'base_task': {
                    'atom': 'H 0 0 0; H 0 0 2.5',
                    'basis': 'sto-3g',
                    'method': 'hf',
                },
                'observables': ['energy'],
            }, work_dir=tmpdir, locale='en')

        self.assertEqual(mock_run.call_count, 3)
        self.assertEqual(report['status'], 'succeeded')
        self.assertEqual(report['comparison_table'][0]['status'], 'succeeded')
        self.assertEqual(report['comparison_table'][0]['method'], 'fci')
        self.assertEqual(report['comparison_table'][0]['initial_method'], 'ccsd')
        self.assertTrue(report['comparison_table'][0]['recovery_applied'])
        self.assertEqual(report['adaptive']['recovery_decisions'][0]['recovery_method'], 'fci')
        self.assertIsNone(report['adaptive']['recovery_decisions'][0]['active_space_contract'])
        self.assertEqual(report['adaptive']['decision_log'][0]['initial_refined_method'], 'ccsd')
        self.assertEqual(report['adaptive']['decision_log'][0]['recovery_method'], 'fci')
        self.assertEqual(report['adaptive']['decision_log'][0]['recommended_method'], 'fci')
        self.assertEqual(report['adaptive']['decision_log'][0]['diagnostic_level'], 'moderate')
        self.assertEqual(report['adaptive']['decision_log'][0]['physics_level'], 'moderate')
        self.assertEqual(report['adaptive']['decision_log'][0]['solver_stress_level'], 'strong')
        self.assertEqual(report['adaptive']['decision_log'][0]['level'], 'strong')
        self.assertIn('method_recovery_applied', report['adaptive']['decision_log'][0]['tags'])
        self.assertNotIn('requires_active_space_approval', report['adaptive']['decision_log'][0]['tags'])

    def test_molecular_recovery_increases_max_cycle_when_no_active_space_can_be_estimated(self):
        refined_plan = StudyPlan(
            study_id='refined',
            name='n2-refined',
            objective='refined N2 stretch',
            system_type='molecular',
            cases=[
                StudyCase(
                    case_id='case-0001',
                    label='stretched N2',
                    variables={'bond_length': 2.4},
                    request={
                        'task_type': 'molecular',
                        'method': 'ccsd',
                        'restricted': False,
                    },
                ),
            ],
            observables=['energy'],
        )
        refined_report_payload = {
            'status': 'completed_with_issues',
            'comparison_table': [
                {
                    'case_id': 'case-0001',
                    'label': 'stretched N2',
                    'status': 'unconverged',
                    'method': 'ccsd',
                    'final_energy': '-108.5 Ha',
                },
            ],
            'cases': [],
        }
        recovery = build_recovery_plan_for_unconverged_refined_cases(
            refined_plan,
            refined_report_payload,
            [
                {
                    'case_id': 'case-0001',
                    'recommended_method': 'ccsd',
                    'tags': ['moderate'],
                },
            ],
            {},
        )

        self.assertIsNotNone(recovery)
        recovery_plan = recovery['recovery_plan']
        request = recovery_plan['cases'][0]['request']
        self.assertEqual(request['method'], 'ccsd')
        self.assertEqual(request['runtime']['max_cycle'], 200)
        self.assertEqual(recovery['recovery_decisions'][0]['recovery_action'], 'increase_max_cycle')
        self.assertIn('max_cycle=200', recovery['recovery_decisions'][0]['reason'])

    def test_molecular_recovery_routes_unconverged_ccsd_t_through_ccsd(self):
        refined_plan = StudyPlan(
            study_id='refined',
            name='n2-refined',
            objective='refined N2 stretch',
            system_type='molecular',
            cases=[
                StudyCase(
                    case_id='case-0001',
                    label='stretched N2',
                    variables={'bond_length': 2.4},
                    request={
                        'task_type': 'molecular',
                        'atom': 'N 0 0 0; N 0 0 2.4',
                        'basis': 'cc-pvdz',
                        'method': 'ccsd_t',
                        'restricted': False,
                    },
                ),
            ],
            observables=['energy'],
        )
        refined_report_payload = {
            'status': 'completed_with_issues',
            'comparison_table': [{
                'case_id': 'case-0001',
                'label': 'stretched N2',
                'status': 'unconverged',
                'method': 'ccsd_t',
            }],
            'cases': [],
        }
        recovery = build_recovery_plan_for_unconverged_refined_cases(
            refined_plan,
            refined_report_payload,
            [{
                'case_id': 'case-0001',
                'diagnostic_level': 'weak',
                'physics_level': 'weak',
                'recommended_method': 'ccsd_t',
                'active_space_contract': {
                    'enabled': True,
                    'selection_method': 'manual',
                    'ncas': 6,
                    'nelecas': 6,
                    'orbital_indices': [4, 5, 6, 7, 8, 9],
                    'approved': False,
                },
            }],
            {},
        )

        self.assertIsNotNone(recovery)
        request = recovery['recovery_plan']['cases'][0]['request']
        decision = recovery['recovery_decisions'][0]
        self.assertEqual(request['method'], 'ccsd')
        self.assertFalse(request['active_space']['enabled'])
        self.assertEqual(decision['recovery_method'], 'ccsd')
        self.assertEqual(decision['diagnostic_level'], 'weak')
        self.assertEqual(decision['physics_level'], 'weak')
        self.assertEqual(decision['solver_stress_level'], 'strong')
        self.assertEqual(decision['level'], 'strong')

    def test_molecular_recovery_routes_failed_refined_cases_to_method_recovery(self):
        refined_plan = StudyPlan(
            study_id='refined',
            name='h2-refined',
            objective='refined H2 stretch',
            system_type='molecular',
            cases=[
                StudyCase(
                    case_id='case-0001',
                    label='stretched H2',
                    variables={'bond_length': 2.2},
                    request={
                        'task_type': 'molecular',
                        'atom': 'H 0 0 0; H 0 0 2.2',
                        'basis': 'sto-3g',
                        'method': 'ccsd',
                        'restricted': False,
                    },
                ),
            ],
            observables=['energy'],
            resource_policy={'total_work_review_threshold': 1, 'review_work_estimates': True},
        )
        refined_report_payload = {
            'status': 'completed_with_issues',
            'comparison_table': [
                {
                    'case_id': 'case-0001',
                    'label': 'stretched H2',
                    'status': 'failed',
                    'method': 'ccsd',
                },
            ],
            'cases': [],
        }
        recovery = build_recovery_plan_for_unconverged_refined_cases(
            refined_plan,
            refined_report_payload,
            [{'case_id': 'case-0001', 'recommended_method': 'ccsd', 'tags': ['moderate']}],
            {},
        )

        self.assertIsNotNone(recovery)
        recovery_plan = recovery['recovery_plan']
        self.assertEqual(recovery_plan['resource_policy']['total_work_review_threshold'], 1)
        self.assertTrue(recovery_plan['cost_estimate'])
        self.assertEqual(recovery['recovery_decisions'][0]['initial_status'], 'failed')
        self.assertEqual(recovery['recovery_decisions'][0]['level'], 'strong')
        self.assertEqual(recovery['recovery_decisions'][0]['solver_stress_level'], 'strong')
        self.assertIn('solver_failure_strong_correlation_evidence', recovery['recovery_decisions'][0]['tags'])
        self.assertEqual(recovery_plan['cases'][0]['request']['method'], 'fci')
        self.assertIn('failed during execution', recovery['recovery_decisions'][0]['reason'])

    def test_molecular_recovery_does_not_fabricate_frontier_active_space(self):
        refined_plan = StudyPlan(
            study_id='refined',
            name='h2-refined',
            objective='refined H2 stretch',
            system_type='molecular',
            cases=[
                StudyCase(
                    case_id='case-0001',
                    label='stretched H2',
                    variables={'bond_length': 2.2},
                    request={
                        'task_type': 'molecular',
                        'atom': 'H 0 0 0; H 0 0 2.2',
                        'basis': 'sto-3g',
                        'method': 'ccsd',
                        'restricted': False,
                    },
                ),
            ],
            observables=['energy'],
        )
        refined_report_payload = {
            'status': 'completed_with_issues',
            'comparison_table': [
                {
                    'case_id': 'case-0001',
                    'label': 'stretched H2',
                    'status': 'unconverged',
                    'method': 'ccsd',
                },
            ],
            'cases': [],
        }
        recovery = build_recovery_plan_for_unconverged_refined_cases(
            refined_plan,
            refined_report_payload,
            [
                {
                    'case_id': 'case-0001',
                    'recommended_method': 'ccsd',
                    'tags': ['moderate'],
                },
            ],
            {},
        )

        self.assertIsNotNone(recovery)
        request = recovery['recovery_plan']['cases'][0]['request']
        self.assertEqual(request['method'], 'fci')
        self.assertEqual(request['runtime']['max_cycle'], 200)
        self.assertFalse(request['active_space']['enabled'])
        self.assertEqual(recovery['recovery_decisions'][0]['recovery_action'], 'switch_method')
        self.assertNotIn('frontier-orbital estimate', recovery['recovery_decisions'][0]['reason'])

    def test_molecular_recovery_uses_casscf_active_space_candidate_for_unconverged_ccsd(self):
        refined_plan = StudyPlan(
            study_id='refined',
            name='n2-refined',
            objective='refined N2 stretch',
            system_type='molecular',
            cases=[
                StudyCase(
                    case_id='case-0001',
                    label='stretched N2',
                    variables={'bond_length': 2.4},
                    request={
                        'task_type': 'molecular',
                        'atom': 'N 0 0 0; N 0 0 2.4',
                        'basis': 'cc-pvdz',
                        'method': 'ccsd',
                        'restricted': False,
                        'active_space_solver': 'block2_dmrg',
                    },
                ),
            ],
            observables=['energy'],
        )
        refined_report_payload = {
            'status': 'completed_with_issues',
            'comparison_table': [
                {
                    'case_id': 'case-0001',
                    'label': 'stretched N2',
                    'status': 'unconverged',
                    'method': 'ccsd',
                    'final_energy': '-108.5 Ha',
                },
            ],
            'cases': [],
        }
        recovery = build_recovery_plan_for_unconverged_refined_cases(
            refined_plan,
            refined_report_payload,
            [
                {
                    'case_id': 'case-0001',
                    'recommended_method': 'ccsd',
                    'tags': ['moderate', 'review_active_space'],
                    'active_space_contract': {
                        'enabled': True,
                        'selection_method': 'manual',
                        'ncas': 6,
                        'nelecas': 8,
                        'orbital_indices': [3, 4, 5, 6, 7, 8],
                        'approved': False,
                    },
                },
            ],
            {},
        )

        self.assertIsNotNone(recovery)
        recovery_plan = recovery['recovery_plan']
        request = recovery_plan['cases'][0]['request']
        self.assertEqual(recovery_plan['study_id'], 'refined-recovery')
        self.assertEqual(request['method'], 'casscf')
        self.assertTrue(request['restricted'])
        self.assertIsNone(request['xc'])
        self.assertEqual(request['active_space']['ncas'], 6)
        self.assertEqual(request['active_space']['nelecas'], 8)
        self.assertEqual(request['active_space']['orbital_indices'], [3, 4, 5, 6, 7, 8])
        self.assertFalse(request['active_space']['approved'])
        self.assertEqual(request['solver']['name'], 'block2_dmrg')
        self.assertEqual(request['solver']['options']['preset'], 'balanced')
        self.assertTrue(request['solver']['options']['save_mps'])
        self.assertEqual(recovery['recovery_decisions'][0]['recovery_method'], 'casscf')
        self.assertEqual(recovery['recovery_decisions'][0]['recovery_solver'], 'block2_dmrg')
        self.assertIn('ActiveSpaceAudit', recovery['recovery_decisions'][0]['reason'])

    def test_adaptive_workflow_reviews_recovery_active_spaces_one_case_at_a_time(self):
        workflow = build_report_workflow({
            'system_type': 'molecular',
            'status': 'completed_with_issues',
            'adaptive': {
                'recovery_plan': {
                    'study_id': 'refined-recovery',
                    'system_type': 'molecular',
                    'cases': [
                        {
                            'case_id': 'case-0001',
                            'label': 'R=2.2',
                            'request': {
                                'task_type': 'molecular',
                                'method': 'casscf',
                                'active_space': {
                                    'enabled': True,
                                    'ncas': 6,
                                    'nelecas': 6,
                                    'orbital_indices': [2, 3, 4, 5, 6, 7],
                                    'approved': False,
                                },
                            },
                        },
                        {
                            'case_id': 'case-0002',
                            'label': 'R=2.4',
                            'request': {
                                'task_type': 'molecular',
                                'method': 'casscf',
                                'active_space': {
                                    'enabled': True,
                                    'ncas': 8,
                                    'nelecas': 8,
                                    'orbital_indices': [1, 2, 3, 4, 5, 6, 7, 8],
                                    'approved': False,
                                },
                            },
                        },
                    ],
                },
                'refined_plan': None,
            },
            'comparison_table': [
                {'case_id': 'case-0001', 'status': 'blocked', 'method': 'casscf'},
                {'case_id': 'case-0002', 'status': 'blocked', 'method': 'casscf'},
            ],
        })

        self.assertEqual(workflow['stage'], 'active_space_review_required')
        self.assertEqual(workflow['review_mode'], 'batch')
        self.assertEqual(workflow['pending_case_ids'], ['case-0001', 'case-0002'])
        self.assertIsNone(workflow['current_case_id'])
        self.assertEqual(len(workflow['items']), 2)
        self.assertEqual(workflow['items'][0]['case_id'], 'case-0001')
        self.assertEqual(workflow['items'][1]['case_id'], 'case-0002')
        actions = {action['id']: action for action in workflow['allowed_actions']}
        self.assertEqual(actions['approve_active_space']['case_ids'], ['case-0001', 'case-0002'])
        self.assertEqual(actions['expand_active_space']['case_ids'], ['case-0001', 'case-0002'])
        self.assertEqual(actions['cancel_active_space_review']['label'], 'Cancel')
        self.assertNotIn('acknowledge', actions)

    def test_active_space_review_embeds_cost_without_a_second_approval_action(self):
        cost_estimate = {
            'approval_required': True,
            'approved': False,
            'review_reason': 'case-0001: determinant space reaches the review threshold',
            'cases': [{
                'case_id': 'case-0001',
                'method': 'casscf',
                'determinant_count': 1_234_567,
                'memory_mb': 96.0,
            }],
            'totals': {'work_units': 24_691_340, 'peak_memory_mb': 96.0},
        }
        workflow = build_report_workflow({
            'system_type': 'molecular',
            'status': 'completed_with_issues',
            'adaptive': {
                'recovery_plan': {
                    'study_id': 'refined-recovery',
                    'system_type': 'molecular',
                    'cost_estimate': cost_estimate,
                    'cases': [{
                        'case_id': 'case-0001',
                        'label': 'R=2.2',
                        'request': {
                            'task_type': 'molecular',
                            'method': 'casscf',
                            'active_space': {
                                'enabled': True,
                                'ncas': 6,
                                'nelecas': 6,
                                'approved': False,
                            },
                        },
                    }],
                },
                'refined_plan': None,
            },
        })

        self.assertEqual(workflow['stage'], 'active_space_review_required')
        self.assertEqual(workflow['cost_estimate'], cost_estimate)
        action_ids = [action['id'] for action in workflow['allowed_actions']]
        self.assertIn('approve_active_space', action_ids)
        self.assertNotIn('approve_cost_estimate', action_ids)

    def test_adaptive_workflow_unresolved_actions_follow_current_case_only(self):
        workflow = build_report_workflow({
            'system_type': 'molecular',
            'status': 'completed_with_issues',
            'adaptive': {
                'refined_plan': {'study_id': 'refined', 'cases': []},
                'decision_log': [
                    {'case_id': 'case-0001', 'recommended_method': 'ccsd'},
                    {'case_id': 'case-0002', 'recommended_method': 'casscf'},
                ],
            },
            'comparison_table': [
                {'case_id': 'case-0001', 'status': 'blocked', 'method': 'ccsd'},
                {'case_id': 'case-0002', 'status': 'unconverged', 'method': 'casscf'},
            ],
        })

        self.assertEqual(workflow['stage'], 'recovery_review_required')
        self.assertEqual(workflow['review_mode'], 'case_queue')
        self.assertEqual(workflow['current_case_id'], 'case-0001')
        self.assertEqual(workflow['items'][0]['case_id'], 'case-0001')
        actions = {action['id']: action for action in workflow['allowed_actions']}
        self.assertIn('show_case_guidance', actions)
        self.assertNotIn('expand_active_space', actions)
        self.assertEqual(actions['show_case_guidance']['case_ids'], ['case-0001'])

    def test_block2_resource_failure_uses_dmrg_recovery_action(self):
        workflow = build_report_workflow({
            'system_type': 'model_hamiltonian',
            'status': 'completed_with_issues',
            'comparison_table': [{
                'case_id': 'case-0001',
                'status': 'failed',
                'solver': 'block2_dmrg',
                'dmrg_failure_class': 'memory_exhausted',
                'dmrg_recovery_action': 'increase_memory_profile',
                'dmrg_recovery_summary': 'Select a larger execution memory profile.',
            }],
        })

        self.assertEqual(workflow['stage'], 'recovery_review_required')
        actions = {action['id']: action for action in workflow['allowed_actions']}
        self.assertEqual(actions['show_case_guidance']['label'], 'Show DMRG Recovery')
        self.assertNotIn('expand_active_space', actions)
        self.assertNotIn('increase_recovery_max_cycle', actions)
