from __future__ import annotations

import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from computational_study_agent.application import StudyApplicationService
from computational_study_agent.application.background import (
    run_invocation,
    INVOCATION_FILENAME,
)
from computational_study_agent.retry import (
    RetryAction,
    compile_retry,
    validate_execution_guard,
)
from computational_study_agent.planner_intent import (
    draft_saved_action,
    resolve_retry_cases,
)
from tests.computational_study_agent.test_retry_collection import (
    Remote,
    plan_for_service,
)
from tests.computational_study_agent.test_study_background import inline_launcher


class RetryActionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.executor = Remote(fail_first=False)
        self.service = StudyApplicationService(
            task_executor=self.executor, execution_config={'execution_target': 'remote'}
        )
        self.service._study_launcher = inline_launcher(self.service)
        self.plan = plan_for_service(3)
        self.first = self.service.run_study(self.plan, work_dir=str(self.root))
        self.directory = self.root / self.plan.study_id

    def action(self, ids=None, **updates):
        return dict(
            self.service.retry_context(self.plan.study_id, work_dir=str(self.root)),
            case_ids=['case-0003'] if ids is None else ids,
            case_overrides={},
            **updates,
        )

    def prepare(self, payload=None):
        return self.service.prepare_retry(
            payload or self.action(), work_dir=str(self.root)
        )

    def start(self, preview):
        result = self.service.start_retry(
            self.plan.study_id, preview['action_id'], work_dir=str(self.root)
        )
        invocation = json.loads((self.directory / INVOCATION_FILENAME).read_text())
        self.assertIsNone(invocation['error'], invocation['error'])
        return result

    def test_single_point_preview_executes_exact_scope_and_duplicate_is_noop(self):
        payload = self.action()
        payload['case_overrides'] = {'case-0003': {'runtime.max_cycle': 100}}
        preview = self.prepare(payload)
        self.assertEqual(preview['execution_case_ids'], ['case-0003'])
        self.assertEqual(preview['total_case_count'], 3)
        self.assertEqual(self.executor.submit_count, 1)
        self.start(preview)
        result = self.service.load_report(self.plan.study_id, work_dir=str(self.root))
        self.assertEqual(
            [t.task_id for t in self.executor.submissions[-1]], ['case-0003']
        )
        self.assertEqual(result['cases'][:2], self.first.to_dict()['cases'][:2])
        self.assertEqual(len(result['comparison_table']), 3)
        self.assertFalse(self.start(preview)['started'])
        self.assertEqual(self.executor.submit_count, 2)

    def test_empty_unknown_and_out_of_scope_changes_rejected_before_writes(self):
        payloads = [self.action([]), self.action(['missing']), self.action()]
        payloads[-1]['case_overrides'] = {'case-0001': {'runtime.max_cycle': 100}}
        for payload in payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.prepare(payload)
        self.assertEqual(self.executor.submit_count, 1)
        self.assertFalse((self.directory / 'retry-actions').exists())

    def test_base_task_changes_are_rejected_with_field_name(self):
        action = self.action()
        action['base_task'] = {'runtime': {'max_cycle': 100}}
        with self.assertRaisesRegex(ValueError, 'base_task'):
            self.prepare(action)
        action.pop('base_task')
        action['case_overrides'] = {
            'case-0003': {'base_task.solver.options.max_iterations': 100}
        }
        with self.assertRaisesRegex(ValueError, 'base_task'):
            self.prepare(action)

    def test_stale_preview_and_execution_guard_reject_before_submission(self):
        preview = self.prepare()
        saved = self.service.load_plan(self.plan.study_id, work_dir=str(self.root))
        saved['cases'][0]['request']['runtime'] = {'max_cycle': 100}
        (self.directory / 'study-plan.json').write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, 'Study changed'):
            self.start(preview)
        self.assertEqual(self.executor.submit_count, 1)

    def test_worker_revalidates_after_launch_before_any_execution(self):
        requests = []
        self.service._study_launcher = lambda directory, request, config, lock=None: (
            requests.append(request) or {'started': True}
        )
        preview = self.prepare()
        self.service.start_retry(
            self.plan.study_id, preview['action_id'], work_dir=str(self.root)
        )
        invocation = {'request': requests[0], 'execution_config': {}, 'error': None}
        (self.directory / INVOCATION_FILENAME).write_text(json.dumps(invocation))
        saved = self.service.load_plan(self.plan.study_id, work_dir=str(self.root))
        saved['cases'][0]['request']['runtime'] = {'max_cycle': 300}
        (self.directory / 'study-plan.json').write_text(json.dumps(saved))
        run_invocation(self.directory, service=self.service)
        invocation = json.loads((self.directory / INVOCATION_FILENAME).read_text())
        self.assertIn('Study changed', invocation['error']['message'])
        self.assertEqual(self.executor.submit_count, 1)

    def test_tampered_execution_plan_cannot_expand_or_change_other_case(self):
        from computational_study_agent.schema import StudyPlan

        preview = self.prepare()
        plan = StudyPlan.from_dict(preview['plan'])
        plan.cases.append(copy.deepcopy(self.plan.cases[0]))
        with self.assertRaisesRegex(ValueError, 'scope'):
            validate_execution_guard(self.directory, plan, preview, ['case-0003'])
        plan.cases.pop()
        plan.cases[0].request['runtime'] = {'max_cycle': 500}
        with self.assertRaisesRegex(ValueError, 'case-0003'):
            validate_execution_guard(self.directory, plan, preview, ['case-0003'])

    def test_dependency_closure_for_every_nonempty_subset_and_fixed_artifacts(self):
        plan = copy.deepcopy(self.plan)
        plan.cases[1].request['initial_state'] = {
            'mode': 'projected_1rdm',
            'source_case_id': 'case-0001',
        }
        plan.cases[2].request['solver'] = {
            'name': 'fci',
            'options': {
                'orbital_restart_manifest': {
                    'source_case_id': 'case-0002',
                    'artifact_kind': 'orbitals',
                }
            },
        }
        ids = [c.case_id for c in plan.cases]
        for size in (1, 2, 3):
            for selected in itertools.combinations(ids, size):
                for include in (False, True):
                    action = RetryAction.from_dict(
                        self.action(list(selected), include_dependents=include)
                    )
                    subset, _, _ = compile_retry(plan, action)
                    expected = (
                        set(ids[ids.index(min(selected)) :])
                        if include
                        else set(selected)
                    )
                    self.assertEqual({c.case_id for c in subset.cases}, expected)
        plan.cases[1].request['initial_state']['source_artifact'] = {
            'path': '/saved/fixed.json'
        }
        subset, _, _ = compile_retry(
            plan,
            RetryAction.from_dict(self.action(['case-0001'], include_dependents=True)),
        )
        self.assertEqual([c.case_id for c in subset.cases], ['case-0001'])

    def test_conversation_exact_id_never_calls_study_spec_planner(self):
        # Use a valid DMET fixture so max_iterations is a real solver setting.
        saved = self.service.load_plan(self.plan.study_id, work_dir=str(self.root))
        saved['cases'][2]['request']['solver'] = {'name': 'dmet', 'options': {}}
        (self.directory / 'study-plan.json').write_text(json.dumps(saved))
        with patch(
            'computational_study_agent.llm_planner.build_study_spec_from_goal_with_evidence',
            side_effect=AssertionError,
        ):
            result = draft_saved_action(
                self.service,
                self.plan.study_id,
                'retry case-0003 with max_iterations 100',
                work_dir=str(self.root),
            )
        self.assertEqual(result['intent'], 'retry_cases')
        self.assertEqual(result['retry_action']['action']['case_ids'], ['case-0003'])
        self.assertEqual(result['retry_action']['execution_case_ids'], ['case-0003'])

    def test_status_selection_uses_saved_results_and_never_defaults_to_every_case(self):
        report = self.first.to_dict()
        report['cases'][1]['task_report']['structured_results']['converged'] = False
        self.assertEqual(
            resolve_retry_cases('retry unconverged', report), ['case-0002']
        )
        with self.assertRaises(ValueError):
            resolve_retry_cases('retry the third point', report)
        with self.assertRaises(ValueError):
            resolve_retry_cases('retry case-0003', report, ['case-0002'])

    def test_failed_retry_preserves_other_results_and_plot_context(self):
        # A failed attempt is a result, not a reason to discard other rows.
        original = self.executor.collect_independent_tasks

        def fail(batches):
            result = original(batches)
            for report in result.reports.values():
                report['execution_status'] = 'failed'
            return result

        self.executor.collect_independent_tasks = fail
        preview = self.prepare()
        self.start(preview)
        result = self.service.load_report(self.plan.study_id, work_dir=str(self.root))
        self.assertEqual(result['cases'][:2], self.first.to_dict()['cases'][:2])
        self.assertEqual(len(result['comparison_table']), 3)
        self.assertEqual(
            result['cases'][2]['task_report']['execution_status'], 'failed'
        )

    def test_selected_parent_failure_blocks_child_without_submitting_it(self):
        saved = self.service.load_plan(self.plan.study_id, work_dir=str(self.root))
        saved['cases'][1]['request']['initial_state'] = {
            'mode': 'projected_1rdm',
            'source_case_id': 'case-0001',
        }
        (self.directory / 'study-plan.json').write_text(json.dumps(saved))
        original = self.executor.collect_independent_tasks

        def fail(batches):
            result = original(batches)
            for report in result.reports.values():
                report['execution_status'] = 'failed'
            return result

        self.executor.collect_independent_tasks = fail
        preview = self.prepare(self.action(['case-0001'], include_dependents=True))
        self.assertEqual(preview['execution_case_ids'], ['case-0001', 'case-0002'])
        self.assertTrue(preview['contexts'][1]['awaits_selected_parent'])
        self.start(preview)
        result = self.service.load_report(self.plan.study_id, work_dir=str(self.root))
        self.assertEqual(
            [t.task_id for t in self.executor.submissions[-1]], ['case-0001']
        )
        self.assertEqual(
            result['cases'][1]['task_report']['execution_status'], 'blocked'
        )
        self.assertIn(
            'case-0001', result['cases'][1]['task_report']['validation_errors'][0]
        )
        self.assertEqual(result['cases'][2], self.first.to_dict()['cases'][2])

    def test_all_independent_subsets_execute_only_their_declared_cases(self):
        ids = [c.case_id for c in self.plan.cases]
        for size in (1, 2, 3):
            for selected in itertools.combinations(ids, size):
                preview = self.prepare(self.action(list(selected)))
                self.start(preview)
                self.assertEqual(
                    {t.task_id for t in self.executor.submissions[-1]}, set(selected)
                )

    def test_cost_approval_applies_to_this_preview_only(self):
        from computational_study_agent.costing import CostApprovalRequired

        saved = self.service.load_plan(self.plan.study_id, work_dir=str(self.root))
        saved['resource_policy']['determinant_review_threshold'] = 1
        (self.directory / 'study-plan.json').write_text(json.dumps(saved))
        preview = self.prepare()
        self.assertTrue(preview['cost_estimate']['approval_required'])
        with self.assertRaises(CostApprovalRequired):
            self.service.start_retry(
                self.plan.study_id, preview['action_id'], work_dir=str(self.root)
            )
        self.service.start_retry(
            self.plan.study_id,
            preview['action_id'],
            work_dir=str(self.root),
            approve_cost=True,
        )
        invocation = json.loads((self.directory / INVOCATION_FILENAME).read_text())
        self.assertIsNone(invocation['error'], invocation['error'])
        self.assertEqual(self.executor.submit_count, 2)

    def test_changed_execution_target_requires_new_preview(self):
        preview = self.prepare()
        self.service._execution_config = {'execution_target': 'local'}
        with self.assertRaisesRegex(ValueError, 'target changed'):
            self.start(preview)
        self.assertEqual(self.executor.submit_count, 1)

    def test_invalid_or_missing_parameters_never_execute(self):
        for patch_value in (-1, True, '100'):
            action = self.action()
            action['case_overrides'] = {
                'case-0003': {'solver.options.max_iterations': patch_value}
            }
            with self.assertRaises(ValueError):
                self.prepare(action)
        with self.assertRaises(ValueError):
            RetryAction.from_dict({'study_id': self.plan.study_id})

    def test_duplicate_worker_cannot_replay_completed_action(self):
        preview = self.prepare()
        self.start(preview)
        run_invocation(self.directory, service=self.service)
        self.assertEqual(self.executor.submit_count, 2)
        record = json.loads(
            (
                self.directory / 'retry-actions' / (preview['action_id'] + '.json')
            ).read_text()
        )
        self.assertEqual(record['status'], 'completed')

    def test_http_preview_and_chat_never_accept_complete_plan_as_retry(self):
        from computational_study_agent.web_api import (
            handle_saved_study_request,
            handle_study_llm_draft_request,
        )

        with patch(
            'computational_study_agent.web_api.get_study_application_service',
            return_value=self.service,
        ):
            payload = {
                'study_id': self.plan.study_id,
                'work_dir': str(self.root),
                'retry_action': dict(
                    self.action(), base_task={'runtime': {'max_cycle': 100}}
                ),
            }
            status, _, body = handle_saved_study_request(
                json.dumps(payload).encode(), action='retry-prepare'
            )
            self.assertEqual(status, 400)
            self.assertIn('base_task', json.loads(body)['error'])
            payload = {
                'study_id': self.plan.study_id,
                'work_dir': str(self.root),
                'goal': 'retry case-0003 with max_cycle 100',
            }
            with patch(
                'computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
                side_effect=AssertionError,
            ):
                status, _, body = handle_study_llm_draft_request(
                    json.dumps(payload).encode()
                )
            self.assertEqual(status, 200, body)
            response = json.loads(body)
            self.assertEqual(response['intent'], 'retry_cases')
            self.assertEqual(
                response['retry_action']['execution_case_ids'], ['case-0003']
            )
            self.assertNotIn('study_spec', response)

    def test_llm_cannot_choose_scope_or_disguise_retry_as_new_study(self):
        from computational_study_agent.planner_intent import classify_goal

        env = {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://invalid.example',
            'PYSCF_AGENT_LLM_MODEL': 'test',
        }
        with patch.dict('os.environ', env):
            for result in (
                {'intent': 'retry_cases', 'case_ids': ['case-0001']},
                {'intent': 'new_study'},
            ):
                with self.subTest(result=result), self.assertRaises(ValueError):
                    classify_goal(
                        'please retry the failed cases with better settings',
                        http_post=lambda *args: {
                            'choices': [{'message': {'content': json.dumps(result)}}]
                        },
                    )
