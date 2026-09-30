from __future__ import annotations

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from computational_study_agent import StudyReport
from computational_study_agent.adaptive.executor import run_adaptive_study
from computational_study_agent.adaptive.path_diagnostics import (
    analyze_scan_path,
    annotate_path_diagnostics,
)
from computational_study_agent.adaptive.refinement import (
    build_path_refinement_plan,
    build_path_window_restart_plans,
    evaluate_path_window_restarts,
    merge_path_window_restart_reports,
)
from computational_study_agent.gates.presentation import build_report_workflow
from computational_study_agent.gates.evaluators import active_space_approval_items
from computational_study_agent.schema import StudyCase, StudyPlan


def _report(energies, methods):
    cases = []
    rows = []
    for index, (energy, method) in enumerate(zip(energies, methods), start=1):
        factor = float(index)
        case_id = 'case-{0:04d}'.format(index)
        cases.append({'case_id': case_id, 'variables': {'bond_factor': factor}})
        rows.append({
            'case_id': case_id,
            'label': 'bond_factor={0}'.format(factor),
            'status': 'succeeded',
            'bond_factor': factor,
            'final_energy': '{0:.8f} Ha'.format(energy),
            'method': method,
        })
    return {
        'system_type': 'molecular',
        'cases': cases,
        'comparison_table': rows,
    }


class ScanPathDiagnosticsTests(unittest.TestCase):
    def test_active_space_review_counts_only_current_path_window(self):
        def case(case_id, approved):
            return {
                'case_id': case_id,
                'label': case_id,
                'request': {
                    'method': 'casscf',
                    'active_space': {
                        'enabled': True,
                        'ncas': 6,
                        'nelecas': 6,
                        'orbital_indices': [1, 2, 3, 4, 5, 6],
                        'approved': approved,
                    },
                },
            }

        report = {
            'system_type': 'molecular',
            'status': 'pending_review',
            'comparison_table': [],
            'adaptive': {
                'mode': 'result_path_review',
                'recovery_plan': {
                    'cases': [case('case-0015', True)],
                },
                'path_refinement_plan': {
                    'cases': [
                        case('case-0013', False),
                        case('case-0014', False),
                        case('case-0015', False),
                    ],
                },
            },
        }

        items = active_space_approval_items(report)
        workflow = build_report_workflow(report)

        self.assertEqual([item['case_id'] for item in items], [
            'case-0013',
            'case-0014',
            'case-0015',
        ])
        self.assertEqual(workflow['stage'], 'path_refinement_review_required')
        self.assertEqual(workflow['pending_case_ids'], [
            'case-0013',
            'case-0014',
            'case-0015',
        ])

    def test_detects_large_discontinuity_at_method_transition(self):
        report = _report(
            [-1.00, -1.08, -1.16, -1.00, -1.01, -1.02],
            ['ccsd', 'ccsd', 'ccsd', 'casscf', 'casscf', 'casscf'],
        )

        diagnostics = analyze_scan_path(report)

        self.assertEqual(diagnostics['status'], 'review_required')
        self.assertEqual(len(diagnostics['anomalies']), 1)
        anomaly = diagnostics['anomalies'][0]
        self.assertEqual(anomaly['recommended_method'], 'casscf')
        self.assertEqual(anomaly['target_case_ids'], ['case-0002', 'case-0003'])
        self.assertGreater(anomaly['transition']['energy_residual'], 0.005)

        annotated = annotate_path_diagnostics(report, diagnostics)
        self.assertEqual(annotated['comparison_table'][1]['path_review'], 'required')
        self.assertNotIn('path_review', annotated['comparison_table'][3])

    def test_does_not_flag_a_smooth_method_boundary(self):
        report = _report(
            [-1.00, -1.08, -1.14, -1.18, -1.20, -1.21],
            ['ccsd', 'ccsd', 'ccsd', 'casscf', 'casscf', 'casscf'],
        )

        diagnostics = analyze_scan_path(report)

        self.assertEqual(diagnostics['status'], 'completed')
        self.assertEqual(diagnostics['anomalies'], [])

    def test_flags_one_sided_endpoint_for_bidirectional_validation(self):
        report = _report(
            [-149.10, -149.82, -149.94, -149.96, -149.97],
            ['ccsd'] * 5,
        )

        diagnostics = analyze_scan_path(report)

        self.assertEqual(diagnostics['status'], 'review_required')
        self.assertEqual(len(diagnostics['anomalies']), 1)
        anomaly = diagnostics['anomalies'][0]
        self.assertEqual(anomaly['kind'], 'endpoint_continuity_risk')
        self.assertEqual(anomaly['endpoint_side'], 'left')
        self.assertEqual(anomaly['target_case_ids'], ['case-0001', 'case-0002'])
        self.assertEqual(anomaly['recommended_method'], 'ccsd')
        self.assertIn('both directions', anomaly['reason'])

    def test_does_not_requeue_bidirectionally_validated_endpoint(self):
        report = _report(
            [-149.10, -149.82, -149.94, -149.96, -149.97],
            ['ccsd'] * 5,
        )
        report['adaptive'] = {
            'path_window_restart_decisions': [
                {
                    'case_id': case_id,
                    'path_anomaly_id': 'path-anomaly-001',
                    'path_anomaly_kind': 'endpoint_continuity_risk',
                }
                for case_id in ('case-0001', 'case-0002')
            ],
            'path_window_restart_validation': {
                'status': 'validated',
                'accepted_case_ids': ['case-0001', 'case-0002'],
            },
        }

        diagnostics = analyze_scan_path(report)

        self.assertEqual(diagnostics['status'], 'completed')
        self.assertEqual(diagnostics['anomalies'], [])
        self.assertEqual(len(diagnostics['validated_findings']), 1)
        self.assertEqual(
            diagnostics['validated_findings'][0]['status'],
            'validated_physical_endpoint',
        )
        self.assertIn('passed bidirectional 1RDM validation', diagnostics['summary'])

    def test_is_not_applicable_to_multi_dimensional_scan(self):
        report = _report(
            [-1.00, -1.08, -1.14, -1.18],
            ['ccsd', 'ccsd', 'casscf', 'casscf'],
        )
        for case, row in zip(report['cases'], report['comparison_table']):
            case['variables']['field'] = 0.1 * case['variables']['bond_factor']
            row['field'] = case['variables']['field']

        diagnostics = analyze_scan_path(report)

        self.assertEqual(diagnostics['status'], 'not_applicable')

    def test_path_refinement_does_not_borrow_orbital_indices_between_cases(self):
        cases = []
        rows = []
        decisions = []
        active_space = {
            'enabled': True,
            'selection_method': 'manual',
            'ncas': 6,
            'nelecas': 6,
            'orbital_indices': [4, 5, 6, 7, 8, 9],
            'approved': False,
        }
        for index in range(1, 5):
            case_id = 'case-{0:04d}'.format(index)
            method = 'ccsd' if index < 4 else 'casscf'
            request = {
                'task_type': 'molecular',
                'atom': 'N 0 0 0; N 0 0 {0}'.format(1.0 + index / 10),
                'basis': 'sto-3g',
                'method': method,
                'analysis': {'outputs': ['energy']},
                'active_space': {'enabled': False},
            }
            cases.append(StudyCase(
                case_id=case_id,
                label='R={0}'.format(index),
                request=request,
                variables={'bond_factor': float(index)},
            ))
            rows.append({
                'case_id': case_id,
                'status': 'succeeded',
                'bond_factor': float(index),
                'final_energy': '-107.{0} Ha'.format(index),
                'method': method,
            })
            decisions.append({
                'case_id': case_id,
                # An overfilled CAS(6,2) candidate must not shrink the local
                # overlap window below the valid CAS(6,6) candidates.
                'active_space_contract': (
                    {
                        'enabled': True,
                        'selection_method': 'manual',
                        'ncas': 2,
                        'nelecas': 6,
                        'orbital_indices': [4, 5],
                        'approved': False,
                    }
                    if index == 2 else active_space
                ),
            })

        refined_plan = StudyPlan(
            study_id='refined',
            name='n2-refined',
            objective='N2 scan',
            system_type='molecular',
            cases=cases,
            observables=['energy'],
        )
        path_diagnostics = {
            'coordinate': 'bond_factor',
            'anomalies': [{
                'id': 'path-anomaly-001',
                'target_case_ids': ['case-0002', 'case-0003'],
                'recommended_method': 'casscf',
                'reason': 'Large method-transition discontinuity.',
                'transition': {
                    'left_case_id': 'case-0003',
                    'right_case_id': 'case-0004',
                },
            }],
        }

        payload = build_path_refinement_plan(
            refined_plan,
            {'comparison_table': rows},
            decisions,
            path_diagnostics,
        )

        self.assertIsNotNone(payload)
        plan = payload['path_refinement_plan']
        self.assertEqual(plan['study_id'], 'path-refinement')
        self.assertEqual([item['case_id'] for item in plan['cases']], ['case-0003', 'case-0004'])
        self.assertEqual(payload['active_space_probe_case_ids'], ['case-0002'])
        self.assertTrue(all(item['request']['method'] == 'casscf' for item in plan['cases']))
        self.assertTrue(all(item['request']['active_space']['approved'] is False for item in plan['cases']))
        self.assertEqual({item['request']['active_space']['ncas'] for item in plan['cases']}, {6})
        self.assertEqual({item['request']['active_space']['nelecas'] for item in plan['cases']}, {6})

        workflow = build_report_workflow({
            'status': 'pending_review',
            'comparison_table': rows,
            'adaptive': {
                'path_refinement_plan': plan,
                'path_refinement_decisions': payload['path_refinement_decisions'],
            },
        })
        self.assertEqual(workflow['stage'], 'path_refinement_review_required')
        self.assertEqual(workflow['review_mode'], 'batch')
        self.assertIsNone(workflow['current_case_id'])
        self.assertEqual(workflow['pending_case_ids'], ['case-0003', 'case-0004'])
        self.assertEqual([item['case_id'] for item in workflow['items']], ['case-0003', 'case-0004'])
        self.assertEqual(workflow['allowed_actions'][0]['id'], 'approve_active_space')
        self.assertEqual(workflow['allowed_actions'][0]['case_ids'], ['case-0003', 'case-0004'])

    def test_retries_path_anomaly_from_two_outer_anchors_before_casscf(self):
        cases = []
        rows = []
        case_reports = []
        for index in range(1, 7):
            case_id = 'case-{0:04d}'.format(index)
            method = 'ccsd' if index < 4 else 'casscf'
            request = {
                'task_type': 'molecular',
                'atom': 'H 0 0 0; H 0 0 {0}'.format(0.70 + index / 10),
                'basis': 'sto-3g',
                'charge': 0,
                'spin': 0,
                'method': method,
                'restricted': True,
                'analysis': {'outputs': ['energy']},
            }
            cases.append(StudyCase(
                case_id=case_id,
                label='R={0}'.format(index),
                request=request,
                variables={'bond_factor': float(index)},
            ))
            rows.append({
                'case_id': case_id,
                'status': 'succeeded',
                'bond_factor': float(index),
                'final_energy': '-1.{0} Ha'.format(index),
                'method': method,
            })
            task_report = {'execution_status': 'succeeded'}
            if index in (1, 5):
                task_report['artifacts'] = [{
                    'kind': 'one_particle_state',
                    'path': '/tmp/compatible-neighbor-{0}.npz'.format(index),
                }]
            case_reports.append({'case_id': case_id, 'task_report': task_report})

        refined_plan = StudyPlan(
            study_id='refined',
            name='h2-refined',
            objective='H2 scan',
            system_type='molecular',
            cases=cases,
            observables=['energy'],
        )
        diagnostics = {
            'coordinate': 'bond_factor',
            'anomalies': [{
                'id': 'path-anomaly-001',
                'kind': 'method_transition_discontinuity',
                'target_case_ids': ['case-0002', 'case-0003'],
                'reason': 'Large method-transition discontinuity.',
                'transition': {
                    'left_case_id': 'case-0003',
                    'right_case_id': 'case-0004',
                },
            }],
        }
        payload = build_path_window_restart_plans(
            refined_plan,
            {'comparison_table': rows, 'cases': case_reports},
            [],
            diagnostics,
        )

        self.assertIsNotNone(payload)
        left_plan = payload['path_window_restart_plans']['left']
        right_plan = payload['path_window_restart_plans']['right']
        self.assertEqual(left_plan['study_id'], 'path-window-restart-left')
        self.assertEqual(right_plan['study_id'], 'path-window-restart-right')
        self.assertEqual([item['case_id'] for item in left_plan['cases']], ['case-0002', 'case-0003'])
        self.assertEqual([item['case_id'] for item in right_plan['cases']], ['case-0003', 'case-0002'])
        self.assertTrue(all(item['request']['method'] == 'ccsd' for item in left_plan['cases']))
        self.assertEqual(
            [item['request']['initial_state']['source_case_id'] for item in left_plan['cases']],
            ['case-0001', 'case-0002'],
        )
        self.assertEqual(
            [item['request']['initial_state']['source_case_id'] for item in right_plan['cases']],
            ['case-0005', 'case-0003'],
        )
        self.assertEqual(
            left_plan['cases'][1]['request']['initial_state']['source_artifact']['deferred_case_id'],
            'case-0002',
        )
        self.assertEqual(
            right_plan['cases'][1]['request']['initial_state']['source_artifact']['deferred_case_id'],
            'case-0003',
        )
        self.assertTrue(payload['approval_required'])
        self.assertEqual(payload['path_window_restart_approval']['status'], 'approval_required')
        self.assertTrue(payload['path_window_restart_approval']['approval_token'])
        approval_item = payload['path_window_restart_approval']['items'][0]
        self.assertEqual(
            [branch['direction'] for branch in approval_item['branches']],
            ['left_to_right', 'right_to_left'],
        )
        self.assertEqual(approval_item['branches'][0]['source_case_id'], 'case-0001')
        self.assertEqual(approval_item['branches'][0]['target_case_id'], 'case-0002')
        self.assertEqual(
            approval_item['branches'][0]['source_artifact']['path'],
            '/tmp/compatible-neighbor-1.npz',
        )
        self.assertEqual(approval_item['branches'][1]['source_case_id'], 'case-0003')
        self.assertEqual(
            approval_item['branches'][1]['source_artifact']['deferred_case_id'],
            'case-0003',
        )
        decisions_by_id = {
            item['case_id']: item
            for item in payload['path_window_restart_decisions']
        }
        self.assertFalse(decisions_by_id['case-0002']['cross_method_restart'])
        self.assertTrue(decisions_by_id['case-0002']['approval_required'])
        self.assertTrue(decisions_by_id['case-0003']['cross_method_restart'])
        self.assertEqual(
            decisions_by_id['case-0003']['right_source_artifact']['path'],
            '/tmp/compatible-neighbor-5.npz',
        )
        workflow = build_report_workflow({
            'status': 'pending_review',
            'comparison_table': rows,
            'adaptive': {
                'path_window_restart_approval': payload['path_window_restart_approval'],
                'path_window_restart_validation': {'status': 'approval_required'},
            },
        })
        self.assertEqual(workflow['stage'], 'continuation_review_required')
        self.assertEqual(workflow['allowed_actions'][0]['id'], 'approve_path_restart')

        left_rows = [
            dict(rows[1], final_energy='-1.01000000 Ha'),
            dict(rows[2], final_energy='-1.02000000 Ha'),
        ]
        left_cases = [
            {'case_id': 'case-0002', 'task_report': {'execution_status': 'succeeded'}},
            {'case_id': 'case-0003', 'task_report': {'execution_status': 'succeeded'}},
        ]
        right_rows = [
            dict(rows[1], final_energy='-1.01000001 Ha'),
            dict(rows[2], final_energy='-1.02000001 Ha'),
        ]
        right_cases = [
            {'case_id': 'case-0002', 'task_report': {'execution_status': 'succeeded'}},
            {'case_id': 'case-0003', 'task_report': {'execution_status': 'succeeded'}},
        ]
        decisions = payload['path_window_restart_decisions']
        decision_log = [{'case_id': 'case-0002'}, {'case_id': 'case-0003'}]
        validation = evaluate_path_window_restarts(
            {'comparison_table': left_rows, 'cases': left_cases},
            {'comparison_table': right_rows, 'cases': right_cases},
            decisions,
        )
        merged = merge_path_window_restart_reports(
            {'comparison_table': rows, 'cases': case_reports},
            {'comparison_table': left_rows, 'cases': left_cases},
            {'comparison_table': right_rows, 'cases': right_cases},
            decision_log,
            decisions,
            validation,
        )
        merged_rows = {row['case_id']: row for row in merged['comparison_table']}
        self.assertTrue(merged_rows['case-0002']['path_restart_applied'])
        self.assertEqual(merged_rows['case-0002']['path_restart_validation'], 'bidirectional_validated')
        self.assertEqual(
            merged_rows['case-0002']['path_restart_anomaly_kind'],
            'method_transition_discontinuity',
        )
        self.assertEqual(decision_log[0]['path_restart_action'], 'bidirectional_projected_1rdm')

    def test_endpoint_restart_uses_baseline_seed_and_adjacent_branch_results(self):
        report = _report(
            [-149.10, -149.82, -149.94, -149.96, -149.97],
            ['ccsd'] * 5,
        )
        cases = []
        case_reports = []
        for index, row in enumerate(report['comparison_table'], start=1):
            case_id = row['case_id']
            cases.append(StudyCase(
                case_id=case_id,
                label=row['label'],
                request={
                    'task_type': 'molecular',
                    'atom': 'O 0 0 0; O 0 0 {0}'.format(0.6 + index / 5),
                    'basis': '6-31g*',
                    'charge': 0,
                    'spin': 2,
                    'method': 'ccsd',
                    'restricted': False,
                },
                variables={'bond_factor': float(index)},
            ))
            case_reports.append({
                'case_id': case_id,
                'task_report': {
                    'execution_status': 'succeeded',
                    'artifacts': [{
                        'kind': 'one_particle_state',
                        'path': '/tmp/o2-state-{0}.npz'.format(index),
                    }],
                },
            })
        diagnostics = analyze_scan_path({
            **report,
            'cases': [
                {'case_id': case.case_id, 'variables': case.variables}
                for case in cases
            ],
        })
        payload = build_path_window_restart_plans(
            StudyPlan(
                study_id='o2-refined',
                name='o2-refined',
                objective='O2 dissociation',
                system_type='molecular',
                cases=cases,
                observables=['energy'],
            ),
            {
                'comparison_table': report['comparison_table'],
                'cases': case_reports,
            },
            [],
            diagnostics,
        )

        self.assertIsNotNone(payload)
        left_cases = payload['path_window_restart_plans']['left']['cases']
        right_cases = payload['path_window_restart_plans']['right']['cases']
        self.assertEqual([item['case_id'] for item in left_cases], ['case-0001', 'case-0002'])
        self.assertEqual([item['case_id'] for item in right_cases], ['case-0002', 'case-0001'])
        self.assertEqual(left_cases[0]['request']['initial_state']['source_case_id'], 'case-0001')
        self.assertEqual(
            left_cases[0]['request']['initial_state']['source_artifact']['path'],
            '/tmp/o2-state-1.npz',
        )
        self.assertEqual(
            left_cases[1]['request']['initial_state']['source_artifact']['deferred_case_id'],
            'case-0001',
        )
        self.assertEqual(right_cases[0]['request']['initial_state']['source_case_id'], 'case-0003')
        self.assertEqual(
            right_cases[1]['request']['initial_state']['source_artifact']['deferred_case_id'],
            'case-0002',
        )
        approval = payload['path_window_restart_approval']
        self.assertEqual(approval['status'], 'approval_required')
        self.assertTrue(all(
            item['approval_reason'] == 'endpoint_bidirectional_validation'
            for item in approval['items']
        ))

    def test_inconclusive_window_restart_keeps_original_points_and_promotes_full_window(self):
        cases = []
        rows = []
        decisions = []
        active_space = {
            'enabled': True,
            'selection_method': 'manual',
            'ncas': 6,
            'nelecas': 6,
            'orbital_indices': [4, 5, 6, 7, 8, 9],
            'approved': False,
        }
        for index, energy in enumerate([-1.00, -1.08, -1.16, -1.00, -1.01, -1.02], start=1):
            case_id = 'case-{0:04d}'.format(index)
            method = 'ccsd' if index <= 3 else 'casscf'
            cases.append(StudyCase(
                case_id=case_id,
                label='R={0}'.format(index),
                request={
                    'task_type': 'molecular',
                    'atom': 'N 0 0 0; N 0 0 {0}'.format(1.0 + index / 10),
                    'basis': 'sto-3g',
                    'method': method,
                    'restricted': True,
                },
                variables={'bond_factor': float(index)},
            ))
            rows.append({
                'case_id': case_id,
                'status': 'succeeded',
                'bond_factor': float(index),
                'final_energy': '{0:.8f} Ha'.format(energy),
                'method': method,
            })
            decisions.append({'case_id': case_id, 'active_space_contract': active_space})
        refined_plan = StudyPlan(
            study_id='refined',
            name='n2-refined',
            objective='N2 scan',
            system_type='molecular',
            cases=cases,
            observables=['energy'],
        )
        path_diagnostics = analyze_scan_path({'cases': [case.to_dict() for case in cases], 'comparison_table': rows})
        restart_decisions = [{
            'case_id': 'case-0002',
            'left_source_case_id': 'case-0001',
            'right_source_case_id': 'case-0005',
        }, {
            'case_id': 'case-0003',
            'left_source_case_id': 'case-0001',
            'right_source_case_id': 'case-0005',
        }]
        left_rows = [dict(rows[1], final_energy='-1.01000000 Ha'), dict(rows[2], final_energy='-1.02000000 Ha')]
        right_rows = [dict(rows[1], final_energy='-0.90000000 Ha'), dict(rows[2], final_energy='-0.90000000 Ha')]
        branch_cases = [
            {'case_id': 'case-0002', 'task_report': {'execution_status': 'succeeded'}},
            {'case_id': 'case-0003', 'task_report': {'execution_status': 'succeeded'}},
        ]
        validation = evaluate_path_window_restarts(
            {'comparison_table': left_rows, 'cases': branch_cases},
            {'comparison_table': right_rows, 'cases': branch_cases},
            restart_decisions,
        )
        self.assertEqual(validation['status'], 'review_required')
        merged = merge_path_window_restart_reports(
            {'comparison_table': rows, 'cases': [case.to_dict() for case in cases]},
            {'comparison_table': left_rows, 'cases': branch_cases},
            {'comparison_table': right_rows, 'cases': branch_cases},
            decisions,
            restart_decisions,
            validation,
        )
        merged_rows = {row['case_id']: row for row in merged['comparison_table']}
        self.assertFalse(merged_rows['case-0002']['path_restart_applied'])
        self.assertEqual(merged_rows['case-0002']['final_energy'], rows[1]['final_energy'])
        refinement = build_path_refinement_plan(
            refined_plan,
            merged,
            decisions,
            path_diagnostics,
        )
        self.assertIsNotNone(refinement)
        self.assertEqual(
            [case['case_id'] for case in refinement['path_refinement_plan']['cases']],
            ['case-0002', 'case-0003', 'case-0004'],
        )

    def test_adaptive_execution_defers_cross_case_path_analysis(self):
        diagnostics = {
            'kind': 'molecular_correlation_diagnostics',
            'level': 'moderate',
            'score': 0.5,
            'confidence': 'medium',
            'molecular_correlation_risk': {
                'kind': 'molecular_correlation_risk',
                'level': 'moderate',
                'overall_score': 0.5,
                'physics_score': 0.4,
                'solver_stress_score': 0.1,
                'method_recommendation': {'preferred': ['ccsd']},
            },
        }
        initial_cases = []
        refined_cases = []
        initial_rows = []
        refined_rows = []
        for index, energy in enumerate([-1.00, -1.08, -1.16, -1.00, -1.01, -1.02], start=1):
            case_id = 'case-{0:04d}'.format(index)
            variables = {'bond_factor': float(index)}
            initial_request = {
                'task_type': 'molecular',
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'method': 'mp2',
            }
            refined_method = 'ccsd' if index <= 3 else 'casscf'
            refined_request = dict(initial_request, method=refined_method)
            initial_cases.append({
                'case_id': case_id,
                'label': 'bond_factor={0}'.format(index),
                'variables': variables,
                'request': initial_request,
                'task_report': {
                    'execution_status': 'succeeded',
                    'structured_results': {
                        'task_type': 'molecular',
                        'method': 'mp2',
                        'energy': -1.0,
                        'correlation_diagnostics': diagnostics,
                    },
                },
            })
            refined_task_report = {
                'execution_status': 'succeeded',
                'structured_results': {
                    'task_type': 'molecular',
                    'method': refined_method,
                    'energy': energy,
                },
            }
            if index in (1, 5):
                refined_task_report['artifacts'] = [{
                    'kind': 'one_particle_state',
                    'path': '/tmp/path-window-state-{0}.npz'.format(index),
                }]
            refined_cases.append({
                'case_id': case_id,
                'label': 'bond_factor={0}'.format(index),
                'variables': variables,
                'request': refined_request,
                'task_report': refined_task_report,
            })
            initial_rows.append({
                'case_id': case_id,
                'label': 'bond_factor={0}'.format(index),
                'status': 'succeeded',
                'bond_factor': float(index),
                'final_energy': '-1.0 Ha',
                'method': 'mp2',
            })
            refined_rows.append({
                'case_id': case_id,
                'label': 'bond_factor={0}'.format(index),
                'status': 'succeeded',
                'bond_factor': float(index),
                'final_energy': '{0:.8f} Ha'.format(energy),
                'method': refined_method,
            })
        initial_report = StudyReport(
            study_id='initial-scan',
            name='initial-scan',
            objective='initial scan',
            system_type='molecular',
            status='succeeded',
            work_dir='initial-scan',
            cases=initial_cases,
            comparison_table=initial_rows,
        )
        refined_report = StudyReport(
            study_id='refined',
            name='refined',
            objective='refined scan',
            system_type='molecular',
            status='succeeded',
            work_dir='refined',
            cases=refined_cases,
            comparison_table=refined_rows,
        )
        with tempfile.TemporaryDirectory() as tmpdir, unittest.mock.patch(
            'computational_study_agent.adaptive.executor.run_study',
            side_effect=[initial_report, refined_report],
        ) as mock_run:
            report = run_adaptive_study({
                'name': 'path-refinement',
                'objective': 'inspect a molecular method boundary',
                'system_type': 'molecular',
                'base_task': {
                    'atom': 'H 0 0 0; H 0 0 0.74',
                    'basis': 'sto-3g',
                    'method': 'mp2',
                },
                'sweep': {'bond_factor': [1, 2, 3, 4, 5, 6]},
                'observables': ['energy'],
            }, work_dir=tmpdir, locale='en')

            artifact_by_kind = {item['kind']: item for item in report['artifacts']}
            diagnostic_payload = json.loads(
                Path(artifact_by_kind['scan-path-diagnostics']['path']).read_text(encoding='utf-8')
            )

        self.assertEqual(mock_run.call_count, 2)
        self.assertEqual(report['status'], 'succeeded')
        self.assertIsNone(report['adaptive']['path_window_restart_plans'])
        self.assertIsNone(report['adaptive']['path_window_restart_validation'])
        self.assertIsNone(report['adaptive']['path_window_restart_approval'])
        self.assertIsNone(report['adaptive']['path_refinement_plan'])
        self.assertNotIn('adaptive-path-window-restart-plan', artifact_by_kind)
        self.assertNotIn('adaptive-path-window-restart-validation', artifact_by_kind)
        self.assertNotIn('adaptive-path-refinement-plan', artifact_by_kind)
        restart_rows = {
            row['case_id']: row
            for row in report['comparison_table']
        }
        self.assertNotIn('path_restart_applied', restart_rows['case-0002'])
        self.assertEqual(diagnostic_payload['schema'], 'pyscf-agent.scan-path-diagnostics.v1')
        self.assertEqual(diagnostic_payload['status'], 'deferred')
        self.assertEqual(report['adaptive']['workflow']['stage'], 'completed')
