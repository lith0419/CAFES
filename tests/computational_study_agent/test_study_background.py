from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from computational_study_agent.application import StudyApplicationService
from computational_study_agent.application.background import (
    INVOCATION_FILENAME, inspect_invocation, invocation_lock, launch_study, run_invocation,
)
from computational_study_agent.execution_receipts import StudyExecutionBusy
from tests.computational_study_agent.test_retry_collection import Remote, plan_for_service


H2_STUDY = {'name': 'H2 comparison', 'system_type': 'molecular',
            'base_task': {'atom': 'H 0 0 0; H 0 0 0.74', 'basis': 'sto-3g', 'method': 'hf'},
            'sweep': {'basis': ['sto-3g', '6-31g']}, 'observables': ['energy']}


def inline_launcher(service):
    """Use the real launcher/dispatcher with a synchronous process fixture."""
    def launch(directory, request, config, *, lock=None):
        def process(*args, **kwargs):
            run_invocation(directory, service=service)
            return SimpleNamespace(pid=os.getpid())
        return launch_study(directory, request, config, popen_factory=process, lock=lock)
    return launch


class StudyBackgroundTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.executor = Remote(fail_first=False)
        self.plan = plan_for_service()
        self.service = StudyApplicationService(task_executor=self.executor,
            plan_builder=lambda spec: copy.deepcopy(self.plan), execution_config={'execution_target': 'remote'})
        self.service._study_launcher = inline_launcher(self.service)
        self.service.prepare_study(H2_STUDY, work_dir=str(self.root))
        self.directory = self.root / self.plan.study_id

    def start(self, **kwargs):
        result = self.service.start_study(self.plan.study_id, work_dir=str(self.root), **kwargs)
        invocation = inspect_invocation(self.directory)
        self.assertIsNone(invocation['error'], invocation)
        return result

    def collect(self):
        return self.service.collect_saved_study(self.plan.study_id, work_dir=str(self.root))

    def test_one_start_calls_existing_runner_for_all_tasks_and_collect_never_submits(self):
        runner = self.service._study_runner
        self.service._study_runner = Mock(wraps=runner)
        self.start()
        self.service._study_runner.assert_called_once()
        first = self.collect()
        self.assertEqual(first['status'], 'succeeded')
        self.assertEqual(len(first['cases']), 2)
        self.assertEqual(self.executor.submit_count, 1)
        self.assertEqual(self.collect()['cases'], first['cases'])
        self.assertEqual(self.executor.submit_count, 1)

    def test_reviewed_subset_uses_same_agent_and_keeps_other_result(self):
        self.start()
        first = self.collect()
        review = self.service.review_study(self.plan.study_id, 'increase_recovery_max_cycle',
            work_dir=str(self.root), case_ids=['case-0001'], plan_kind='static')
        self.assertEqual(self.executor.submit_count, 1)
        self.assertIn('pending_review', self.service.load_report(self.plan.study_id, work_dir=str(self.root)))
        self.start()
        result = self.collect()
        self.assertEqual([case['attempt_count'] for case in result['cases']], [2, 1])
        self.assertEqual(result['cases'][1], first['cases'][1])
        self.assertEqual(result['study_id'], self.plan.study_id)
        self.assertNotIn('pending_review', result)

    def test_stale_cycle_retry_for_execution_error_does_not_persist_or_submit(self):
        from computational_study_agent.gates.review_actions import StudyReviewActionError

        self.start()
        report = self.collect()
        report['cases'][0]['task_report']['execution_status'] = 'failed'
        report['cases'][0]['task_report']['errors'] = [{
            'code': 'execution_exception', 'message': 'Internal Error.',
        }]
        report['comparison_table'][0]['status'] = 'failed'
        self.service._save_report(report, work_dir=str(self.root))
        path = self.directory / 'study-report.json'
        before = path.read_bytes()
        with self.assertRaisesRegex(StudyReviewActionError, 'execution failed with an error'):
            self.service.review_study(self.plan.study_id, 'increase_recovery_max_cycle',
                work_dir=str(self.root), case_ids=['case-0001'], plan_kind='static')
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.executor.submit_count, 1)

    def test_dmet_scf_recovery_prepares_saved_subset_before_any_submission(self):
        from computational_study_agent.schema import StudyPlan
        from tests.computational_study_agent.support import hubbard_dimer_spec
        from tests.computational_study_agent.test_review_actions import static_dmet_plan
        from tests.pyscf_agent.test_dmet_recovery import failed_dmet_task

        payload = static_dmet_plan()
        payload['cases'][0]['request']['model_hamiltonian'] = {'spec': hubbard_dimer_spec()}
        payload['cases'][0]['request']['solver']['options'].update({
            'execution_mode': 'finite_graph', 'impurity_size': 1,
        })
        sibling = copy.deepcopy(payload['cases'][0])
        sibling['case_id'] = 'case-0001'
        payload['cases'].append(sibling)
        self.plan = StudyPlan.from_dict(payload)
        self.service.prepare_study(H2_STUDY, work_dir=str(self.root))
        self.directory = self.root / self.plan.study_id
        self.start()
        self.collect()
        self.start(rerun_case_ids=['case-0002'])
        report = self.collect()
        failed = report['cases'][0]
        failed['task_report'].update(failed_dmet_task())
        report['comparison_table'][0]['status'] = 'failed'
        self.service._save_report(report, work_dir=str(self.root))
        other = copy.deepcopy(report['cases'][1])
        submits = self.executor.submit_count

        review = self.service.review_study(self.plan.study_id, 'retry_dmet_without_impurity_diis',
            work_dir=str(self.root), case_ids=[failed['case_id']], plan_kind='static')
        self.assertEqual(self.executor.submit_count, submits)
        prepared = review['plan']['cases']
        self.assertEqual([case['case_id'] for case in prepared], [failed['case_id']])
        self.assertIs(prepared[0]['request']['solver']['options']['impurity_scf_diis'], False)
        self.start()
        collected = self.collect()
        self.assertEqual(collected['cases'][0]['attempt_count'], 3)
        self.assertIs(collected['cases'][0]['request']['solver']['options']['impurity_scf_diis'], False)
        self.assertEqual(collected['cases'][1], other)

    def test_launch_never_calls_task_executor_and_inherited_lock_prevents_duplicate(self):
        handles = []
        def process(*args, **kwargs):
            handles.append(os.dup(kwargs['pass_fds'][0]))
            return SimpleNamespace(pid=12345)
        self.service._study_launcher = lambda directory, request, config, **kwargs: launch_study(
            directory, request, config, popen_factory=process, **kwargs)
        try:
            first = self.service.start_study(self.plan.study_id, work_dir=str(self.root))
            self.assertTrue(first['started'])
            self.assertEqual(self.executor.submit_count, 0)
            self.assertTrue(inspect_invocation(self.directory)['running'])
            second = self.service.start_study(self.plan.study_id, work_dir=str(self.root))
            self.assertFalse(second['started'])
            self.assertEqual(len(handles), 1)
            with self.assertRaises(StudyExecutionBusy):
                self.collect()
            with self.assertRaises(StudyExecutionBusy):
                self.service.review_study(self.plan.study_id, 'approve_cost_estimate', work_dir=str(self.root))
        finally:
            for fd in handles:
                os.close(fd)
        self.assertTrue(inspect_invocation(self.directory)['interrupted'])
        self.service._study_launcher = inline_launcher(self.service)
        self.start()
        self.assertEqual(self.collect()['status'], 'succeeded')

    def test_launcher_failure_is_persisted_and_lock_released(self):
        def fail(*args, **kwargs):
            raise OSError('cannot start interpreter')
        with self.assertRaises(OSError):
            launch_study(self.directory, {'mode': 'static'}, {}, popen_factory=fail)
        status = inspect_invocation(self.directory)
        self.assertFalse(status['running'])
        self.assertEqual(status['error']['code'], 'OSError')
        with invocation_lock(self.directory):
            pass

    def test_interrupted_remote_invocation_is_collected_without_resubmission(self):
        self.executor.wait_count = 0
        self.service.start_study(self.plan.study_id, work_dir=str(self.root))
        invocation = inspect_invocation(self.directory)
        self.assertEqual(invocation['error']['code'], 'StudyExecutionInterrupted')
        self.assertIn('receipt', invocation['error'])
        self.assertEqual(self.collect()['status'], 'succeeded')
        self.assertEqual(self.executor.submit_count, 1)

    def test_saved_review_survives_collect_and_failed_launch(self):
        self.start()
        original = self.collect()
        self.service.review_study(self.plan.study_id, 'increase_recovery_max_cycle',
            work_dir=str(self.root), case_ids=['case-0001'], plan_kind='static')
        collected = self.collect()
        self.assertEqual(collected['cases'], original['cases'])
        self.assertTrue(collected['pending_review']['can_run'])
        self.service._study_launcher = Mock(side_effect=OSError('launcher unavailable'))
        with self.assertRaises(OSError):
            self.start()
        self.assertIn('pending_review', self.service.load_report(self.plan.study_id, work_dir=str(self.root)))
        self.service._study_launcher = inline_launcher(self.service)
        # An uncertain launch is not automatically replayed under the same action ID.
        self.assertFalse(self.service.start_study(self.plan.study_id, work_dir=str(self.root))['started'])
        self.service.review_study(self.plan.study_id, 'increase_recovery_max_cycle',
            work_dir=str(self.root), case_ids=['case-0001'], plan_kind='static')
        self.start()
        self.assertNotIn('pending_review', self.collect())
        self.assertEqual(self.executor.submit_count, 2)

    def test_saved_continuation_approval_dispatches_existing_analysis_use_case(self):
        self.start()
        report = self.collect()
        report['adaptive'] = {'mps_continuation_approval': {
            'schema': 'test-approval', 'approval_token': 'case-evidence', 'case_ids': ['case-0001']}}
        self.service._save_report(report, work_dir=str(self.root))
        with self.assertRaises(ValueError):
            self.service.review_study(self.plan.study_id, 'approve_mps_continuation',
                work_dir=str(self.root), approval_token='wrong')
        reviewed = self.service.review_study(self.plan.study_id, 'approve_mps_continuation',
            work_dir=str(self.root), approval_token='case-evidence', case_ids=['case-0001'])
        before = self.executor.submit_count
        analyze = Mock(return_value={'report': report})
        self.service._analyze_saved_results = analyze
        self.start()
        analyze.assert_called_once()
        self.assertEqual(analyze.call_args.kwargs['review']['analysis_approval'], reviewed['analysis_approval'])
        self.assertNotIn('pending_review', self.collect())
        self.assertEqual(self.executor.submit_count, before)

    def test_wrong_plan_and_missing_remote_config_rejected_before_start(self):
        wrong = self.plan.to_dict()
        wrong['study_id'] = 'another-study'
        with self.assertRaisesRegex(ValueError, 'different Study'):
            self.start(prepared_plan=wrong)
        self.service._execution_config = None
        with self.assertRaisesRegex(ValueError, 'connection options'):
            self.start()
        self.assertFalse((self.directory / INVOCATION_FILENAME).exists())

    def test_explicit_active_space_preparation_saves_review_without_execution(self):
        service = StudyApplicationService(task_executor=self.executor)
        spec = dict(H2_STUDY, base_task=dict(H2_STUDY['base_task'], method='casscf', active_space={
            'enabled': True, 'ncas': 2, 'nelecas': 2, 'approved': False,
            'selection_method': 'manual', 'orbital_indices': [0, 1]}))
        prepared = service.prepare_study(spec, work_dir=str(self.root))
        self.assertEqual(prepared['status'], 'active_space_review')
        report = service.load_report(prepared['study_id'], work_dir=str(self.root))
        self.assertEqual(report['study_id'], prepared['study_id'])
        self.assertEqual(report['lifecycle']['entity_id'], prepared['study_id'])
        self.assertEqual(prepared['plan']['lifecycle']['entity_id'], prepared['study_id'])
        self.assertEqual(report['status'], 'pending_review')
        self.assertEqual(self.executor.submit_count, 0)
        review = service.review_study(prepared['study_id'], 'approve_active_space',
            work_dir=str(self.root), case_ids=[case['case_id'] for case in prepared['plan']['cases']])
        self.assertTrue(all(case['request']['active_space']['approved'] for case in review['plan']['cases']))

    def test_adaptive_initial_cost_approval_keeps_prepared_identity(self):
        from computational_study_agent.costing import CostApprovalRequired
        from computational_study_agent.adaptive.executor import _adaptive_resume_matches_request, _adaptive_request_fingerprint
        from computational_study_agent.adaptive.initial_scan import normalize_adaptive_options
        runner = Mock(return_value={})
        service = StudyApplicationService(task_executor=self.executor, adaptive_runner=runner,
                                          execution_config={'execution_target': 'remote'})
        service._study_launcher = inline_launcher(service)
        spec = dict(H2_STUDY, study_mode='adaptive', resource_policy={
            'review_work_estimates': True, 'total_work_review_threshold': 1})
        prepared = service.prepare_study(spec, work_dir=str(self.root))
        study_id = prepared['study_id']
        with self.assertRaises(CostApprovalRequired):
            service.start_study(study_id, work_dir=str(self.root))
        review = service.review_study(study_id, 'approve_cost_estimate', work_dir=str(self.root))
        runner.assert_not_called()
        service.start_study(study_id, work_dir=str(self.root))
        runner.assert_called_once()
        self.assertEqual(runner.call_args.kwargs['resume_study_id'], study_id)
        spec = runner.call_args.args[0].to_dict()
        self.assertTrue(spec['resource_policy']['approved'])
        fingerprint = _adaptive_request_fingerprint(spec, normalize_adaptive_options(runner.call_args.kwargs['options']))
        self.assertTrue(_adaptive_resume_matches_request(self.root / study_id, fingerprint))
        self.assertIsNone(inspect_invocation(self.root / study_id)['error'])

    def test_adaptive_preparation_and_start_delegate_to_existing_adaptive_runner(self):
        runner = Mock(return_value={})
        service = StudyApplicationService(task_executor=self.executor, adaptive_runner=runner,
                                          execution_config={'execution_target': 'remote'})
        service._study_launcher = inline_launcher(service)
        prepared = service.prepare_study(dict(H2_STUDY, study_mode='adaptive'), work_dir=str(self.root))
        study_id = prepared['study_id']
        self.assertEqual(prepared['mode'], 'adaptive')
        self.assertEqual(self.executor.submit_count, 0)
        service.start_study(study_id, work_dir=str(self.root))
        runner.assert_called_once()
        self.assertEqual(runner.call_args.kwargs['resume_study_id'], study_id)
        self.assertEqual(runner.call_args.kwargs['task_executor'], self.executor)
        self.assertIsNone(inspect_invocation(self.root / study_id)['error'])
        self.assertEqual(self.executor.submit_count, 0)


if __name__ == '__main__':
    unittest.main()
