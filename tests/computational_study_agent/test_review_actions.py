from __future__ import annotations

import copy
import unittest

from pyscf_agent.lifecycle import new_lifecycle, transition_lifecycle

from computational_study_agent.gates.review_actions import (
    StudyReviewActionError,
    apply_study_review_action,
)
from computational_study_agent.gates.presentation import build_report_workflow
from pyscf_agent.backend.parsing import task_spec_from_partial
from tests.pyscf_agent.test_dmet_recovery import failed_dmet_task


def review_lifecycle():
    lifecycle = new_lifecycle('study', 'review-action-study')
    return transition_lifecycle(lifecycle, 'review_requested', details={'source': 'test'})


def molecular_case(*, approved=False):
    return {
        'case_id': 'case-0001',
        'label': 'R=2.0',
        'request': {
            'task_type': 'molecular',
            'atom': 'N 0 0 0; N 0 0 2.0',
            'basis': 'sto-3g',
            'method': 'casscf',
            'active_space': {
                'enabled': True,
                'selection_method': 'manual',
                'ncas': 2,
                'nelecas': 2,
                'orbital_indices': [2, 3],
                'approved': approved,
                'expansion_evidence': {
                    'source': 'test occupations',
                    'orbital_count': 6,
                    'orbitals': [
                        {'index': 0, 'occupation': 2.0},
                        {'index': 1, 'occupation': 2.0},
                        {'index': 2, 'occupation': 1.0},
                        {'index': 3, 'occupation': 1.0},
                        {'index': 4, 'occupation': 0.0},
                        {'index': 5, 'occupation': 0.0},
                    ],
                },
            },
        },
    }


def recovery_plan(case=None):
    return {
        'study_id': 'refined-recovery',
        'name': 'recovery',
        'objective': 'recover unresolved cases',
        'system_type': 'molecular',
        'cases': [case or molecular_case()],
        'observables': ['energy'],
        'lifecycle': review_lifecycle(),
    }


def static_dmet_plan():
    return {
        'study_id': 'static-dmet-study',
        'name': 'static dmet',
        'objective': 'retry one unconverged DMET case',
        'system_type': 'model_hamiltonian',
        'cases': [{
            'case_id': 'case-0002',
            'label': 'U=2, V=4',
            'request': {
                'task_type': 'model_hamiltonian',
                'solver': {
                    'name': 'dmet',
                    'options': {'max_iterations': 50, 'energy_tolerance': 1e-6},
                },
            },
        }],
        'observables': ['energy'],
        'workflow_configuration': {
            'nodes': [{'module_id': 'study.static_execution'}],
        },
        'lifecycle': review_lifecycle(),
    }


class StudyReviewActionTests(unittest.TestCase):
    def test_diagnosed_dmet_failure_offers_and_prepares_only_the_failed_case(self):
        plan = static_dmet_plan()
        failed = copy.deepcopy(plan['cases'][0])
        failed['task_report'] = failed_dmet_task()
        succeeded = copy.deepcopy(plan['cases'][0])
        succeeded['case_id'] = 'case-0001'
        plan['cases'].append(succeeded)
        state = {
            'study_id': plan['study_id'], 'system_type': 'model_hamiltonian',
            'status': 'completed_with_issues', 'lifecycle': review_lifecycle(),
            'cases': [failed, succeeded],
            'comparison_table': [
                {'case_id': 'case-0001', 'status': 'succeeded', 'solver': 'dmet'},
                {'case_id': 'case-0002', 'status': 'failed', 'solver': 'dmet'},
            ],
        }
        before = copy.deepcopy((state, plan))
        workflow = build_report_workflow(copy.deepcopy(state))
        proposal = workflow['allowed_actions'][0]
        self.assertEqual(proposal['id'], 'retry_dmet_without_impurity_diis')
        self.assertEqual(proposal['case_ids'], ['case-0002'])
        self.assertIn('impurity_scf_diis=false', proposal['description'])
        result = apply_study_review_action(proposal['id'], study_state=state, plan=plan,
            case_ids=proposal['case_ids'], plan_kind='static')
        self.assertTrue(result['can_run'])
        self.assertEqual(result['plan']['study_id'], plan['study_id'])
        self.assertEqual([case['case_id'] for case in result['plan']['cases']], ['case-0002'])
        expected = copy.deepcopy(plan['cases'][0]['request'])
        expected['solver']['options']['impurity_scf_diis'] = False
        self.assertEqual(result['plan']['cases'][0]['request'], expected)
        self.assertEqual((state, plan), before)
        with self.assertRaisesRegex(StudyReviewActionError, 'with DIIS still enabled'):
            apply_study_review_action(proposal['id'], study_state=result['study_state'],
                plan=result['plan'], case_ids=['case-0002'], plan_kind='static')

    def test_impurity_retry_needs_specific_failure_evidence(self):
        with self.assertRaisesRegex(StudyReviewActionError, 'reported impurity SCF DIIS failure'):
            apply_study_review_action('retry_dmet_without_impurity_diis',
                plan=static_dmet_plan(), case_ids=['case-0002'], plan_kind='static')

    def test_execution_errors_offer_details_without_iteration_or_method_changes(self):
        for system_type, solver, method in (
            ('model_hamiltonian', 'dmet', None),
            ('model_hamiltonian', 'fci', None),
            ('molecular', None, 'ccsd'),
            ('molecular', 'fci', 'casscf'),
        ):
            with self.subTest(solver=solver, method=method):
                workflow = build_report_workflow({
                    'system_type': system_type, 'status': 'completed_with_issues',
                    'comparison_table': [
                        {'case_id': 'failed', 'status': 'failed', 'solver': solver, 'method': method},
                        {'case_id': 'unconverged', 'status': 'unconverged', 'solver': 'dmet'},
                    ],
                })
                self.assertEqual(workflow['current_case_id'], 'failed')
                self.assertEqual(
                    [action['id'] for action in workflow['allowed_actions']],
                    ['show_case_guidance', 'acknowledge'],
                )

    def test_unconverged_dmet_offers_only_its_outer_iteration_retry(self):
        workflow = build_report_workflow({
            'system_type': 'model_hamiltonian', 'status': 'completed_with_issues',
            'comparison_table': [{'case_id': 'case-0002', 'status': 'unconverged', 'solver': 'dmet'}],
        })
        self.assertEqual(
            [action['id'] for action in workflow['allowed_actions']],
            ['increase_dmet_iterations', 'acknowledge'],
        )

    def test_generic_iteration_action_rejects_dmet_even_without_a_report(self):
        for solver in ('dmet', {'name': 'dmet', 'options': {'max_iterations': 50}}):
            with self.subTest(solver=solver):
                plan = static_dmet_plan()
                plan['cases'][0]['request']['solver'] = solver
                before = copy.deepcopy(plan)
                with self.assertRaisesRegex(StudyReviewActionError, 'runtime.max_cycle does not control DMET'):
                    apply_study_review_action(
                        'increase_recovery_max_cycle', plan=plan,
                        case_ids=['case-0002'], plan_kind='static',
                    )
                self.assertEqual(plan, before)

    def test_stale_iteration_actions_reject_execution_errors_without_mutating_state(self):
        for action in ('increase_recovery_max_cycle', 'increase_dmet_iterations'):
            for evidence in ('cases', 'comparison_table'):
                with self.subTest(action=action, evidence=evidence):
                    plan = static_dmet_plan()
                    state = {'study_id': plan['study_id'], 'lifecycle': review_lifecycle()}
                    if evidence == 'cases':
                        state['cases'] = [{'case_id': 'case-0002', 'task_report': {
                            'execution_status': 'failed',
                            'errors': [{'code': 'execution_exception', 'exception_type': 'LinAlgError'}],
                        }}]
                    else:
                        state['comparison_table'] = [{'case_id': 'case-0002', 'status': 'failed'}]
                    before = copy.deepcopy((state, plan))
                    with self.assertRaisesRegex(StudyReviewActionError, 'execution failed with an error'):
                        apply_study_review_action(action, study_state=state, plan=plan,
                            case_ids=['case-0002'], plan_kind='static')
                    self.assertEqual((state, plan), before)

    def test_dmet_iteration_retry_prepares_static_selected_subset(self):
        plan = static_dmet_plan()
        state = {
            'study_id': 'static-dmet-study',
            'lifecycle': review_lifecycle(),
            'workflow': {
                'stage': 'recovery_review_required',
                'active_plan_kind': 'static',
            },
        }

        result = apply_study_review_action(
            'increase_dmet_iterations',
            study_state=state,
            plan=plan,
            case_ids=['case-0002'],
            plan_kind='static',
        )

        self.assertTrue(result['can_run'])
        self.assertEqual(result['lifecycle']['stage'], 'refining')
        self.assertEqual(result['plan']['_review_kind'], 'static')
        self.assertEqual(len(result['plan']['cases']), 1)
        options = result['plan']['cases'][0]['request']['solver']['options']
        self.assertEqual(options['max_iterations'], 100)
        self.assertEqual(options['energy_tolerance'], 1e-6)
        self.assertEqual(task_spec_from_partial(result['plan']['cases'][0]['request']).solver.options['max_iterations'], 100)
        self.assertEqual(plan['cases'][0]['request']['solver']['options']['max_iterations'], 50)
        self.assertEqual(result['plan']['study_id'], plan['study_id'])
        self.assertNotIn('_adaptive_parent_study_id', result['plan'])

    def test_max_cycle_retry_prepares_selected_subset(self):
        plan = recovery_plan()
        plan['cases'][0]['request']['runtime'] = {'max_cycle': 200}
        state = {
            'study_id': 'review-action-study',
            'lifecycle': review_lifecycle(),
            'adaptive': {'recovery_plan': plan},
        }

        result = apply_study_review_action(
            'increase_recovery_max_cycle',
            study_state=state,
            plan=plan,
            case_ids=['case-0001'],
            plan_kind='recovery',
        )

        self.assertTrue(result['can_run'])
        self.assertEqual(result['lifecycle']['stage'], 'refining')
        self.assertEqual(result['workflow']['stage'], 'refinement_ready')
        self.assertEqual(result['plan']['cases'][0]['request']['runtime']['max_cycle'], 400)

    def test_manual_max_cycle_retry_uses_effective_runtime_without_alias_override(self):
        plan = recovery_plan()
        request = plan['cases'][0]['request']
        request['active_space'].pop('expansion_evidence')
        request.update({'max_cycle': 800, 'runtime': {'max_cycle': 50}})
        result = apply_study_review_action('increase_recovery_max_cycle', plan=plan,
            case_ids=['case-0001'], plan_kind='recovery')
        updated = result['plan']['cases'][0]['request']
        self.assertNotIn('max_cycle', updated)
        self.assertEqual(task_spec_from_partial(updated).runtime.max_cycle, 1600)
        self.assertEqual(request['max_cycle'], 800)

    def test_expand_active_space_keeps_review_required(self):
        plan = recovery_plan()
        state = {
            'study_id': 'review-action-study',
            'lifecycle': review_lifecycle(),
            'adaptive': {'recovery_plan': plan},
        }

        result = apply_study_review_action(
            'expand_active_space',
            study_state=state,
            plan=plan,
            case_ids=['case-0001'],
            plan_kind='recovery',
        )

        active_space = result['plan']['cases'][0]['request']['active_space']
        self.assertFalse(result['can_run'])
        self.assertEqual(result['status'], 'review_required')
        self.assertEqual(active_space['orbital_indices'], [1, 2, 3, 4])
        self.assertEqual(active_space['ncas'], 4)
        self.assertEqual(active_space['nelecas'], 4)
        self.assertFalse(active_space['approved'])

    def test_approve_active_space_prepares_refinement(self):
        plan = recovery_plan()
        plan['cases'][0]['request']['active_space_solver'] = 'block2_dmrg'
        state = {
            'study_id': 'review-action-study',
            'lifecycle': review_lifecycle(),
            'adaptive': {'recovery_plan': plan},
        }

        result = apply_study_review_action(
            'approve_active_space',
            study_state=state,
            plan=plan,
            case_ids=['case-0001'],
            plan_kind='recovery',
        )

        self.assertTrue(result['can_run'])
        self.assertEqual(result['lifecycle']['stage'], 'refining')
        self.assertTrue(result['plan']['cases'][0]['request']['active_space']['approved'])
        self.assertEqual(
            result['plan']['cases'][0]['request']['solver'],
            {'name': 'block2_dmrg', 'options': {'preset': 'balanced', 'save_mps': True}},
        )
        stored = result['study_state']['adaptive']['recovery_plan']['cases'][0]
        self.assertTrue(stored['request']['active_space']['approved'])
        self.assertEqual(stored['request']['active_space']['ncas'], 2)
        self.assertEqual(stored['request']['solver']['name'], 'block2_dmrg')

    def test_path_restart_approval_is_stored_in_study_state(self):
        state = {
            'study_id': 'review-action-study',
            'lifecycle': review_lifecycle(),
            'adaptive': {
                'path_window_restart_approval': {
                    'schema': 'pyscf-agent.path-window-restart-approval.v1',
                    'approval_token': 'token-1',
                    'status': 'approval_required',
                    'case_ids': ['case-0001'],
                },
                'path_window_restart_validation': {'status': 'approval_required'},
            },
        }

        result = apply_study_review_action(
            'approve_path_restart',
            study_state=state,
            plan=recovery_plan(),
            case_ids=['case-0001'],
            approval_token='token-1',
        )

        adaptive = result['study_state']['adaptive']
        self.assertTrue(result['can_run'])
        self.assertEqual(result['lifecycle']['stage'], 'refining')
        self.assertEqual(adaptive['path_window_restart_review']['decision'], 'approve')
        self.assertEqual(adaptive['path_window_restart_validation']['status'], 'approved')
        self.assertEqual(result['analysis_approval']['approval_token'], 'token-1')

    def test_path_restart_rejects_stale_token(self):
        state = {
            'lifecycle': review_lifecycle(),
            'adaptive': {
                'path_window_restart_approval': {
                    'approval_token': 'current-token',
                    'status': 'approval_required',
                },
            },
        }
        with self.assertRaises(StudyReviewActionError):
            apply_study_review_action(
                'approve_path_restart',
                study_state=state,
                plan=recovery_plan(),
                approval_token='stale-token',
            )

    def test_mps_continuation_approval_is_stored_for_result_analysis(self):
        state = {
            'study_id': 'review-action-study',
            'lifecycle': review_lifecycle(),
            'adaptive': {
                'mps_continuation_approval': {
                    'schema': 'pyscf-agent.mps-continuation-approval.v1',
                    'approval_token': 'mps-token-1',
                    'status': 'approval_required',
                    'case_ids': ['case-0001'],
                },
            },
        }

        result = apply_study_review_action(
            'approve_mps_continuation',
            study_state=state,
            plan=recovery_plan(),
            case_ids=['case-0001'],
            approval_token='mps-token-1',
        )

        adaptive = result['study_state']['adaptive']
        self.assertTrue(result['can_run'])
        self.assertEqual(result['lifecycle']['stage'], 'refining')
        self.assertEqual(result['workflow']['stage'], 'mps_continuation_retry_ready')
        self.assertEqual(adaptive['mps_continuation_review']['decision'], 'approve')
        self.assertEqual(result['analysis_approval'], {
            'decision': 'approve',
            'approval_token': 'mps-token-1',
        })

    def test_mps_continuation_skip_advances_without_a_retry_plan(self):
        state = {
            'study_id': 'review-action-study',
            'lifecycle': review_lifecycle(),
            'adaptive': {
                'mps_continuation_approval': {
                    'schema': 'pyscf-agent.mps-continuation-approval.v1',
                    'approval_token': 'mps-token-2',
                    'status': 'approval_required',
                    'case_ids': ['case-0001'],
                },
            },
        }

        result = apply_study_review_action(
            'skip_mps_continuation',
            study_state=state,
            plan=recovery_plan(),
            case_ids=['case-0001'],
            approval_token='mps-token-2',
        )

        self.assertFalse(result['can_run'])
        self.assertEqual(result['status'], 'skipped')
        self.assertEqual(result['workflow']['stage'], 'mps_continuation_skipped')
        self.assertEqual(result['analysis_approval']['decision'], 'skip')

    def test_initial_cost_approval_returns_study_spec_patch(self):
        plan = {
            **recovery_plan(),
            'study_id': 'initial-scan',
            'resource_policy': {'approved': False},
            'cost_estimate': {
                'approval_required': True,
                'approved': False,
                'policy': {'approved': False},
            },
        }
        state = {
            'lifecycle': review_lifecycle(),
            'adaptive': {
                'initial_scan_plan': plan,
                'initial_scan_cost_estimate': {
                    'approval_required': True,
                    'approved': False,
                    'policy': {'approved': False},
                },
            },
        }

        result = apply_study_review_action(
            'approve_cost_estimate',
            study_state=state,
            plan=plan,
            plan_kind='initial',
        )

        self.assertTrue(result['can_run'])
        self.assertEqual(result['study_spec_patch'], {'resource_policy': {'approved': True}})
        self.assertTrue(result['study_state']['adaptive']['initial_scan_cost_estimate']['approved'])

    def test_modified_retry_cost_approval_matches_the_reviewed_subset(self):
        from computational_study_agent.costing import ensure_plan_cost_estimate, require_plan_cost_approval
        from computational_study_agent.schema import StudyPlan

        plan = recovery_plan()
        plan['resource_policy'] = {'total_work_review_threshold': 1, 'approved': True}
        typed = StudyPlan.from_dict(plan)
        ensure_plan_cost_estimate(typed)
        plan = typed.to_dict()
        old_fingerprint = plan['cost_estimate']['plan_fingerprint']
        changed = apply_study_review_action(
            'increase_recovery_max_cycle', plan=plan,
            study_state={'study_id': plan['study_id'], 'lifecycle': review_lifecycle()},
            case_ids=['case-0001'], plan_kind='static',
        )
        estimate = changed['plan']['cost_estimate']
        self.assertNotEqual(estimate['plan_fingerprint'], old_fingerprint)
        self.assertFalse(estimate['approved'])
        approved = apply_study_review_action(
            'approve_cost_estimate', plan=changed['plan'],
            study_state=changed['study_state'], plan_kind='static',
        )
        # A freshly loaded executor must accept the review without another gate.
        fresh = StudyPlan.from_dict(approved['plan'])
        self.assertTrue(require_plan_cost_approval(fresh)['can_execute'])
        self.assertEqual(len(fresh.cases), 1)

        # Previously persisted reviews can contain the old estimate as well.
        stale = changed['plan']
        stale['cost_estimate'] = plan['cost_estimate']
        approved = apply_study_review_action(
            'approve_cost_estimate', plan=stale,
            study_state=changed['study_state'], plan_kind='static',
        )
        self.assertTrue(require_plan_cost_approval(StudyPlan.from_dict(approved['plan']))['can_execute'])


if __name__ == '__main__':
    unittest.main()
