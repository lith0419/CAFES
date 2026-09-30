from __future__ import annotations

import copy
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from computational_study_agent.application.service import StudyApplicationService
from computational_study_agent.execution_receipts import StudyExecutionInterrupted
from computational_study_agent.schema import StudyReport
from computational_study_agent.web_api import handle_study_execution_collect_request
from tests.computational_study_agent.support import hubbard_dimer_spec
from tests.computational_study_agent.test_batch_execution import _plan, _RecoverableBatchExecutor


def plan_for_service(count=2):
    plan = _plan(count)
    for case in plan.cases:
        case.request['model_hamiltonian']['spec'] = hubbard_dimer_spec()
    return plan


class Remote(_RecoverableBatchExecutor):
    def __init__(self, *, fail_first=True, interrupt=False):
        super().__init__()
        self.fail_first = fail_first
        self.wait_count = 0 if interrupt else 1
        self.submissions = []
        self.omit_report = False

    def submit_independent_tasks(self, tasks, **kwargs):
        self.submissions.append(list(tasks))
        return super().submit_independent_tasks(tasks, **kwargs)

    def collect_independent_tasks(self, batches):
        result = super().collect_independent_tasks(batches)
        for batch in batches:
            for handle in batch.jobs:
                case_id = handle.job_id.split(':', 1)[1]
                report = result.reports[case_id]
                report.update(run_id=handle.run_id, work_dir=handle.work_dir)
                if self.fail_first and handle.run_id == 'case-0001':
                    report['execution_status'] = 'failed'
                    report['structured_results']['converged'] = False
                report['structured_results']['correlation_diagnostics'] = {
                    'level': 'weak', 'score': 0.1, 'confidence': 'high',
                }
                run_dir = Path(handle.work_dir) / handle.run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / 'task-report.json').write_text(json.dumps(report))
        if self.omit_report:
            result.reports.pop(next(iter(result.reports)))
        return result


class RetryCollectionTests(unittest.TestCase):
    def test_completed_scheduler_without_visible_report_cannot_be_collected_yet(self):
        from dataclasses import replace
        from pyscf_agent.executors import JobState
        executor = Remote(fail_first=False, interrupt=True)
        plan = plan_for_service()
        service = StudyApplicationService(task_executor=executor)
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(StudyExecutionInterrupted):
                service.run_study(plan, work_dir=root)
            status = executor.status
            executor.status = lambda handle: replace(status(handle), state=JobState.COMPLETED, report_available=False)
            self.assertFalse(service.inspect_execution(plan.study_id, work_dir=root)['can_collect'])
            with self.assertRaises(StudyExecutionInterrupted):
                service.collect_study(plan, work_dir=root)
            self.assertEqual(executor.submit_count, 1)
            executor.status = status
            self.assertEqual(service.collect_study(plan, work_dir=root).status, 'succeeded')
            self.assertEqual(executor.submit_count, 1)

    def test_concurrent_explicit_run_is_busy_instead_of_queuing_another_attempt(self):
        import threading
        from computational_study_agent.execution_receipts import StudyExecutionBusy
        executor = Remote()
        service = StudyApplicationService(task_executor=executor)
        plan = plan_for_service()
        entered, release = threading.Event(), threading.Event()
        with tempfile.TemporaryDirectory() as root:
            service.run_study(plan, work_dir=root)
            submit = executor.submit_independent_tasks
            def slow_submit(*args, **kwargs):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('Test submission was not released')
                return submit(*args, **kwargs)
            executor.submit_independent_tasks = slow_submit
            with ThreadPoolExecutor(max_workers=1) as pool:
                first = pool.submit(service.run_study, plan, work_dir=root, rerun_case_ids=['case-0001'])
                try:
                    self.assertTrue(entered.wait(5))
                    with self.assertRaises(StudyExecutionBusy):
                        service.run_study(plan, work_dir=root, rerun_case_ids=['case-0001'])
                finally:
                    release.set()
                report = first.result(timeout=10)
            self.assertEqual(executor.submit_count, 2)
            self.assertEqual([c['attempt_count'] for c in report.cases], [2, 1])

    def test_unknown_submission_is_reconciled_without_another_submission(self):
        from pyscf_agent.serialization import json_fingerprint
        executor = Remote(fail_first=False)
        submit = executor.submit_independent_tasks
        evidence = {}
        def lose_ack(tasks, **kwargs):
            batches = submit(tasks, **kwargs)
            evidence.update(batches=[b.to_dict() for b in batches], task_fingerprints={
                t.task_id: json_fingerprint({'request': t.request, 'channel': t.channel,
                                        'locale': t.locale, 'run_id': t.run_id}) for t in tasks})
            raise OSError('SSH acknowledgement lost after real submission')
        executor.submit_independent_tasks = lose_ack
        executor.recover_submission = lambda path: copy.deepcopy(evidence)
        service = StudyApplicationService(task_executor=executor)
        plan = plan_for_service()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(StudyExecutionInterrupted):
                service.run_study(plan, work_dir=root)
            saved = copy.deepcopy(evidence)
            evidence['task_fingerprints']['case-0001'] = 'different request'
            with self.assertRaisesRegex(ValueError, 'requests do not match'):
                service.reconcile_execution(plan, work_dir=root, submission_path='/server/submission.json')
            evidence.update(copy.deepcopy(saved))
            evidence['batches'][0]['jobs'][0]['run_id'] = 'different-attempt'
            with self.assertRaisesRegex(ValueError, 'pending case attempt'):
                service.reconcile_execution(plan, work_dir=root, submission_path='/server/submission.json')
            evidence.update(copy.deepcopy(saved))
            with self.assertRaises(ValueError):
                service.reconcile_execution(plan, work_dir=root, submission_path='/server/submission.json',
                                            receipt_file=str(Path(root) / 'outside.json'))
            for _ in range(2):
                receipt = service.reconcile_execution(plan, work_dir=root, submission_path='/server/submission.json')
                self.assertEqual(receipt['status'], 'submitted')
            report = service.collect_study(plan, work_dir=root)
            self.assertEqual(report.status, 'succeeded')
            self.assertEqual([c['attempt_count'] for c in report.cases], [1, 1])
            self.assertEqual(executor.submit_count, 1)

    def test_failed_retry_submits_only_selected_case_and_keeps_old_artifacts(self):
        executor = Remote()
        service = StudyApplicationService(task_executor=executor)
        plan = plan_for_service()
        with tempfile.TemporaryDirectory() as root:
            first = service.run_study(plan, work_dir=root)
            old_path = Path(first.cases[0]['task_report']['work_dir']) / first.cases[0]['task_report']['run_id'] / 'task-report.json'
            old_bytes = old_path.read_bytes()
            second = service.run_study(plan, work_dir=root, rerun_case_ids=['case-0001'])
            self.assertEqual(executor.submit_count, 2)
            self.assertEqual([t.task_id for t in executor.submissions[1]], ['case-0001'])
            self.assertNotEqual(first.cases[0]['task_report']['run_id'], second.cases[0]['task_report']['run_id'])
            self.assertEqual([c['attempt_count'] for c in second.cases], [2, 1])
            self.assertEqual(second.status, 'succeeded')
            self.assertEqual(old_bytes, old_path.read_bytes())
            self.assertEqual(service.inspect_execution(plan.study_id, work_dir=root)['expected'], 2)
            self.assertEqual(StudyReport.from_dict(second.to_dict()).cases, second.cases)

    def test_collect_endpoint_and_repeated_collection_never_retry_mixed_results(self):
        executor = Remote()
        service = StudyApplicationService(task_executor=executor)
        plan = plan_for_service()
        with tempfile.TemporaryDirectory() as root:
            service.run_study(plan, work_dir=root)
            with patch('computational_study_agent.web_api.get_study_application_service', return_value=service):
                for _ in range(2):
                    status, _, body = handle_study_execution_collect_request(json.dumps({
                        'study_id': plan.study_id, 'plan': plan.to_dict(), 'work_dir': root,
                    }).encode())
                    payload = json.loads(body)
                    self.assertEqual(status.value, 200, payload)
                    self.assertEqual(payload['report']['status'], 'completed_with_issues')
                    self.assertEqual([c['attempt_count'] for c in payload['report']['cases']], [1, 1])
            self.assertEqual(executor.submit_count, 1)

    def test_new_service_collects_interrupted_handles_without_wait_or_retry(self):
        executor = Remote(interrupt=True)
        plan = plan_for_service()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(StudyExecutionInterrupted):
                StudyApplicationService(task_executor=executor).run_study(plan, work_dir=root)
            service = StudyApplicationService(task_executor=executor)
            with ThreadPoolExecutor(max_workers=2) as pool:
                reports = list(pool.map(lambda _: service.collect_study(plan, work_dir=root), range(2)))
            self.assertEqual(executor.submit_count, 1)
            self.assertEqual(executor.wait_count, 1)
            self.assertTrue(all([c['attempt_count'] for c in r.cases] == [1, 1] for r in reports))
            self.assertTrue(all(r.status == 'completed_with_issues' for r in reports))

    def test_collect_sequential_partial_execution_does_not_start_next_case(self):
        executor = Remote(fail_first=False, interrupt=True)
        service = StudyApplicationService(task_executor=executor)
        plan = plan_for_service()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(StudyExecutionInterrupted):
                service.run_study(plan, work_dir=root, batch_independent=False)
            report = service.collect_study(plan, work_dir=root)
            self.assertEqual(executor.submit_count, 1)
            self.assertEqual([c['attempt_count'] for c in report.cases], [1, 0])
            self.assertEqual(report.cases[1]['task_report']['execution_status'], 'not_executed')
            completed = service.run_study(plan, work_dir=root, batch_independent=False)
            self.assertEqual(executor.submit_count, 2)
            self.assertEqual(completed.status, 'succeeded')

    def test_explicit_subset_keeps_other_failed_cases(self):
        executor = Remote()
        service = StudyApplicationService(task_executor=executor)
        plan = plan_for_service()
        with tempfile.TemporaryDirectory() as root:
            first = service.run_study(plan, work_dir=root)
            result = service.run_study(plan, work_dir=root, rerun_case_ids=['case-0002'])
            self.assertEqual([t.task_id for t in executor.submissions[1]], ['case-0002'])
            self.assertEqual(result.cases[0]['task_report'], first.cases[0]['task_report'])
            self.assertEqual([c['attempt_count'] for c in result.cases], [1, 2])

    def test_corrupt_or_missing_checkpoint_never_becomes_a_fresh_execution(self):
        executor = Remote(fail_first=False)
        service = StudyApplicationService(task_executor=executor)
        plan = plan_for_service()
        with tempfile.TemporaryDirectory() as root:
            service.run_study(plan, work_dir=root)
            path = Path(root) / plan.study_id / 'study-state.json'
            original = path.read_text()
            for content in ('{broken', '{}', None):
                with self.subTest(content=content):
                    if content is None:
                        path.unlink()
                    else:
                        path.write_text(content)
                    with self.assertRaises(ValueError):
                        service.run_study(plan, work_dir=root, resume=False)
                    self.assertEqual(executor.submit_count, 1)
                    path.write_text(original)

    def test_corrupt_receipt_and_unknown_submission_block_resubmission(self):
        for unknown in (False, True):
            executor = Remote(interrupt=True)
            service = StudyApplicationService(task_executor=executor)
            plan = plan_for_service()
            with self.subTest(unknown=unknown), tempfile.TemporaryDirectory() as root:
                if unknown:
                    original = executor.submit_independent_tasks
                    def submit_then_disconnect(*args, **kwargs):
                        original(*args, **kwargs)
                        raise OSError('acknowledgement lost')
                    executor.submit_independent_tasks = submit_then_disconnect
                with self.assertRaises((OSError, StudyExecutionInterrupted)):
                    service.run_study(plan, work_dir=root)
                receipt = Path(root) / plan.study_id / 'execution-receipt.json'
                if unknown:
                    self.assertEqual(service.inspect_execution(plan.study_id, work_dir=root)['status'], 'submission_unknown')
                else:
                    receipt.write_text('{corrupt')
                with self.assertRaises(ValueError):
                    service.run_study(plan, work_dir=root)
                self.assertEqual(executor.submit_count, 1)

    def test_legacy_interrupted_v1_receipt_is_adopted_without_new_submission(self):
        executor = Remote(interrupt=True)
        service = StudyApplicationService(task_executor=executor)
        plan = plan_for_service()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(StudyExecutionInterrupted):
                service.run_study(plan, work_dir=root)
            path = Path(root) / plan.study_id / 'study-state.json'
            state = json.loads(path.read_text())
            state['cases'] = {}  # v1 wrote per-case state only after collection.
            state.pop('execution_associations_version', None)
            path.write_text(json.dumps(state))
            report = service.collect_study(plan, work_dir=root)
            self.assertEqual(executor.submit_count, 1)
            self.assertEqual([c['attempt_count'] for c in report.cases], [1, 1])

    def test_incomplete_collection_keeps_receipt_recoverable(self):
        executor = Remote(fail_first=False)
        executor.omit_report = True
        service = StudyApplicationService(task_executor=executor)
        plan = plan_for_service()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(StudyExecutionInterrupted):
                service.run_study(plan, work_dir=root)
            path = Path(root) / plan.study_id / 'execution-receipt.json'
            self.assertNotEqual(json.loads(path.read_text())['status'], 'collected')
            executor.omit_report = False
            report = service.collect_study(plan, work_dir=root)
            self.assertEqual(report.status, 'succeeded')
            self.assertEqual(executor.submit_count, 1)

    def test_adaptive_collect_prepares_refinement_without_submitting_it(self):
        executor = Remote(fail_first=False, interrupt=True)
        service = StudyApplicationService(task_executor=executor)
        spec = {'name': 'adaptive-collection', 'objective': 'collect diagnostics',
                'system_type': 'molecular', 'base_task': {
                    'task_type': 'molecular', 'atom': 'H 0 0 0; H 0 0 0.74',
                    'basis': 'sto-3g', 'method': 'mp2'}, 'observables': ['energy']}
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(StudyExecutionInterrupted):
                service.run_adaptive_study(spec, work_dir=root, requested_study_id='adaptive-collection')
            report = service.collect_adaptive_study(spec, work_dir=root, study_id='adaptive-collection')
            self.assertEqual(executor.submit_count, 1)
            self.assertEqual(report['status'], 'pending_review')
            self.assertIsNone(report['adaptive']['refined_report'])
            self.assertEqual(report['adaptive']['execution_pending_plan_kind'], 'refined')
            self.assertEqual(report['adaptive']['workflow']['stage'], 'plan_ready')
            again = service.collect_adaptive_study(spec, work_dir=root, study_id='adaptive-collection')
            self.assertEqual(again['study_id'], report['study_id'])
            self.assertEqual(executor.submit_count, 1)
            with self.assertRaises(ValueError):
                service.collect_adaptive_study({**spec, 'name': 'changed'}, work_dir=root, study_id='adaptive-collection')
            self.assertEqual(executor.submit_count, 1)

    def test_local_retries_have_distinct_run_directories_and_bounded_attempts(self):
        class Local(Remote):
            supports_recoverable_batch = False
            supports_independent_batch = False

            def execute_task(self, request, *, run_id, work_dir, **kwargs):
                self.single_calls.append(run_id)
                report = {'run_id': run_id, 'work_dir': work_dir,
                          'execution_status': 'failed', 'structured_results': {}}
                path = Path(work_dir) / run_id
                path.mkdir(parents=True)
                (path / 'task-report.json').write_text(json.dumps(report))
                return report

        executor = Local()
        service = StudyApplicationService(task_executor=executor)
        plan = plan_for_service(1)
        with tempfile.TemporaryDirectory() as root:
            first = service.run_study(plan, work_dir=root)
            collected = service.collect_study(plan, work_dir=root)
            self.assertEqual(collected.cases[0]['attempt_count'], 1)
            second = service.run_study(plan, work_dir=root)
            limited = service.run_study(plan, work_dir=root)
            self.assertEqual(len(executor.single_calls), 2)
            self.assertNotEqual(executor.single_calls[0], executor.single_calls[1])
            self.assertEqual([first.cases[0]['attempt_count'], second.cases[0]['attempt_count'],
                              limited.cases[0]['attempt_count']], [1, 2, 2])
            self.assertEqual(len(list((Path(root) / plan.study_id / 'cases').glob('*/task-report.json'))), 2)

    def test_new_checkpoint_missing_association_does_not_migrate_as_legacy(self):
        executor = Remote()
        service = StudyApplicationService(task_executor=executor)
        plan = plan_for_service(1)
        with tempfile.TemporaryDirectory() as root:
            service.run_study(plan, work_dir=root)
            path = Path(root) / plan.study_id / 'study-state.json'
            state = json.loads(path.read_text())
            del state['cases']['case-0001']['execution']
            path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, 'association is missing'):
                service.run_study(plan, work_dir=root)
            self.assertEqual(executor.submit_count, 1)

    def test_incorrect_report_identity_is_not_marked_collected(self):
        executor = Remote()
        service = StudyApplicationService(task_executor=executor)
        original = executor.collect_independent_tasks
        def wrong_identity(batches):
            result = original(batches)
            result.reports['case-0001']['run_id'] = 'another-run'
            return result
        executor.collect_independent_tasks = wrong_identity
        plan = plan_for_service(1)
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(StudyExecutionInterrupted):
                service.run_study(plan, work_dir=root)
            path = Path(root) / plan.study_id / 'execution-receipt.json'
            self.assertNotEqual(json.loads(path.read_text())['status'], 'collected')
            executor.collect_independent_tasks = original
            service.collect_study(plan, work_dir=root)
            self.assertEqual(executor.submit_count, 1)

    def test_invalid_batch_does_not_partially_update_any_checkpoint(self):
        from pyscf_agent.executors import ExecutorContractError
        for recoverable in (True, False):
            for problem in ('missing', 'extra', 'wrong_case', 'wrong_run'):
                with self.subTest(recoverable=recoverable, problem=problem), tempfile.TemporaryDirectory() as root:
                    executor = Remote(fail_first=False)
                    executor.supports_recoverable_batch = recoverable
                    method = 'collect_independent_tasks' if recoverable else 'execute_independent_tasks'
                    original = getattr(executor, method)
                    def malformed(*args, **kwargs):
                        result = original(*args, **kwargs)
                        if problem == 'missing':
                            result.reports.pop('case-0002')
                        elif problem == 'extra':
                            result.reports['unexpected'] = copy.deepcopy(result.reports['case-0001'])
                        else:
                            result.reports['case-0002']['case_id' if problem == 'wrong_case' else 'run_id'] = 'unexpected'
                        return result
                    setattr(executor, method, malformed)
                    plan = plan_for_service()
                    service = StudyApplicationService(task_executor=executor)
                    with self.assertRaises(StudyExecutionInterrupted if recoverable else ExecutorContractError):
                        service.run_study(plan, work_dir=root)
                    state = json.loads((Path(root) / plan.study_id / 'study-state.json').read_text())
                    self.assertTrue(all(case['execution']['pending'] for case in state['cases'].values()))
                    self.assertTrue(all('task_report' not in case for case in state['cases'].values()))

    def test_adaptive_refined_collection_does_not_submit_recovery(self):
        class RefinementFailure(Remote):
            def wait_for_independent_tasks(self, batches):
                if self.submit_count == 2:
                    raise OSError('disconnected during refinement')

            def collect_independent_tasks(self, batches):
                result = super().collect_independent_tasks(batches)
                for report in result.reports.values():
                    if 'initial-scan' not in report['work_dir']:
                        report['execution_status'] = 'unconverged'
                        report['structured_results'].update(converged=False, final_method='ccsd')
                return result

        executor = RefinementFailure(fail_first=False)
        service = StudyApplicationService(task_executor=executor)
        spec = {'name': 'adaptive-recovery-collection', 'objective': 'collect refinement',
                'system_type': 'molecular', 'base_task': {
                    'task_type': 'molecular', 'atom': 'H 0 0 0; H 0 0 0.74',
                    'basis': 'sto-3g', 'method': 'mp2'}, 'observables': ['energy']}
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(StudyExecutionInterrupted):
                service.run_adaptive_study(spec, work_dir=root, requested_study_id='adaptive-recovery-collection')
            report = service.collect_adaptive_study(spec, work_dir=root, study_id='adaptive-recovery-collection')
            self.assertEqual(executor.submit_count, 2)
            self.assertEqual(report['adaptive']['execution_pending_plan_kind'], 'recovery')
            self.assertIsNone(report['adaptive']['recovery_report'])
            self.assertEqual(report['adaptive']['refined_report']['cases'][0]['attempt_count'], 1)
