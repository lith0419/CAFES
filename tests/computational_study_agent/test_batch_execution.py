from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from computational_study_agent.executor import run_study
from computational_study_agent.execution_receipts import StudyExecutionInterrupted
from computational_study_agent.schema import StudyCase, StudyPlan
from pyscf_agent.executors import (
    BatchExecutionResult,
    BatchHandle,
    JobHandle,
    JobState,
    JobStatus,
)


def _model_request(value):
    model_spec = {
        'model': 'hubbard',
        'nelec': [1, 1],
        'sites': [
            {'id': 0, 'x': 0, 'y': 0, 'U': value},
            {'id': 1, 'x': 1, 'y': 0, 'U': value},
        ],
        'bonds': [{'i': 0, 'j': 1, 't': -1}],
    }
    return {
        'task_type': 'model_hamiltonian',
        'solver': 'fci',
        'model_hamiltonian': {'spec': model_spec},
    }


def _plan(case_count=3):
    return StudyPlan(
        study_id='batch-study',
        name='batch-study',
        objective='execute independent cases together',
        system_type='model_hamiltonian',
        cases=[
            StudyCase(
                case_id='case-{0:04d}'.format(index),
                label='U={0}'.format(index),
                request=_model_request(index),
                variables={'U': index},
            )
            for index in range(1, case_count + 1)
        ],
        observables=['energy'],
    )


class _RecordingBatchExecutor:
    executor_id = 'recording-batch'
    execution_mode = 'test'
    supports_independent_batch = True

    def __init__(self):
        self.batch_calls = []
        self.single_calls = []

    @staticmethod
    def _report(task_id):
        index = int(task_id.rsplit('-', 1)[1])
        return {
            'execution_status': 'succeeded',
            'structured_results': {
                'final_energy': -float(index),
                'final_method': 'fci',
                'converged': True,
            },
            'artifacts': [],
        }

    def execute_independent_tasks(self, tasks):
        self.batch_calls.append(list(tasks))
        return BatchExecutionResult(
            reports={
                task.task_id: self._report(task.task_id)
                for task in tasks
            },
            batches=[{
                'batch_id': 'batch-test',
                'case_ids': [task.task_id for task in tasks],
            }],
        )

    def execute_task(self, _request, *, run_id=None, **_kwargs):
        self.single_calls.append(run_id)
        return self._report(run_id)

    def submit_task(self, *_args, **_kwargs):
        raise NotImplementedError

    def status(self, *_args, **_kwargs):
        raise NotImplementedError

    def cancel(self, *_args, **_kwargs):
        raise NotImplementedError

    def fetch(self, *_args, **_kwargs):
        raise NotImplementedError

    def logs(self, *_args, **_kwargs):
        raise NotImplementedError

    def artifacts(self, *_args, **_kwargs):
        raise NotImplementedError

    def describe(self):
        return {
            'executor_id': self.executor_id,
            'supports_independent_batch': True,
        }


class _RecoverableBatchExecutor(_RecordingBatchExecutor):
    executor_id = 'recoverable-batch'
    supports_recoverable_batch = True

    def __init__(self):
        super().__init__()
        self.submit_count = 0
        self.wait_count = 0
        self.status_many_count = 0
        self.submitted_profiles = []

    def submit_independent_tasks(self, tasks, *, resource_profile=None):
        self.submit_count += 1
        self.submitted_profiles.append(resource_profile)
        self.batch_calls.append(list(tasks))
        submitted_at = '2026-08-04T00:00:00+00:00'
        jobs = [
            JobHandle(
                job_id='batch-test:{0}'.format(task.task_id),
                executor_id=self.executor_id,
                run_id=task.run_id,
                work_dir=task.work_dir,
                submitted_at=submitted_at,
                backend_job_id='4815_{0}'.format(index),
            )
            for index, task in enumerate(tasks)
        ]
        return [BatchHandle(
            batch_id='batch-test',
            executor_id=self.executor_id,
            backend_job_id='4815',
            submitted_at=submitted_at,
            profile_id=resource_profile or 'test',
            manifest_path='/remote/batch-manifest.json',
            jobs=jobs,
        )]

    def wait_for_independent_tasks(self, _batches):
        self.wait_count += 1
        if self.wait_count == 1:
            raise OSError('temporary transport failure')

    def collect_independent_tasks(self, batches):
        task_ids = [
            job.job_id.split(':', 1)[1]
            for batch in batches
            for job in batch.jobs
        ]
        return BatchExecutionResult(
            reports={task_id: self._report(task_id) for task_id in task_ids},
            batches=[batch.to_dict() for batch in batches],
        )

    def status(self, handle):
        return JobStatus(
            handle=handle,
            state=JobState.COMPLETED,
            updated_at='2026-08-04T00:01:00+00:00',
            completed_at='2026-08-04T00:01:00+00:00',
            report_available=True,
        )

    def status_many(self, handles):
        self.status_many_count += 1
        return [self.status(handle) for handle in handles]

    def describe(self):
        return {
            'executor_id': self.executor_id,
            'execution_mode': 'remote_scheduler',
            'location': 'test_cluster',
            'supports_independent_batch': True,
            'supports_recoverable_batch': True,
        }


class BatchStudyExecutionTests(unittest.TestCase):
    def test_ready_independent_cases_use_one_batch_call(self):
        executor = _RecordingBatchExecutor()

        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_study(
                _plan(),
                work_dir=tmpdir,
                resume=False,
                task_executor=executor,
            )

        self.assertEqual(len(executor.batch_calls), 1)
        self.assertEqual(len(executor.batch_calls[0]), 3)
        self.assertEqual(executor.single_calls, [])
        self.assertEqual(report.execution['strategy'], 'independent_batch')
        self.assertEqual(report.execution['batching']['case_count'], 3)
        self.assertEqual(
            [row['status'] for row in report.comparison_table],
            ['succeeded', 'succeeded', 'succeeded'],
        )

    def test_more_than_eight_independent_cases_still_use_one_batch_call(self):
        executor = _RecordingBatchExecutor()

        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_study(
                _plan(11),
                work_dir=tmpdir,
                resume=False,
                task_executor=executor,
            )

        self.assertEqual(len(executor.batch_calls), 1)
        self.assertEqual(len(executor.batch_calls[0]), 11)
        self.assertEqual(executor.single_calls, [])
        self.assertEqual(report.execution['strategy'], 'independent_batch')
        self.assertEqual(report.execution['batching']['case_count'], 11)

    def test_refinement_mode_keeps_cases_on_single_task_path(self):
        executor = _RecordingBatchExecutor()

        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_study(
                _plan(2),
                work_dir=tmpdir,
                resume=False,
                task_executor=executor,
                batch_independent=False,
            )

        self.assertEqual(executor.batch_calls, [])
        self.assertEqual(executor.single_calls, ['case-0001', 'case-0002'])
        self.assertEqual(report.execution['strategy'], 'sequential')
        self.assertFalse(report.execution['batching']['requested'])

    def test_projected_1rdm_case_is_not_added_to_independent_batch(self):
        executor = _RecordingBatchExecutor()
        plan = _plan(2)
        plan.cases[1].request['initial_state'] = {
            'mode': 'projected_1rdm',
            'source_case_id': 'anchor',
            'source_artifact': {
                'kind': 'one_particle_state',
                'path': '/shared/anchor.npz',
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_study(
                plan,
                work_dir=tmpdir,
                resume=False,
                task_executor=executor,
            )

        self.assertEqual(
            [task.task_id for task in executor.batch_calls[0]],
            ['case-0001'],
        )
        self.assertEqual(executor.single_calls, ['case-0002'])
        self.assertEqual(report.execution['batching']['case_count'], 1)

    def test_interrupted_batch_resumes_from_receipt_without_resubmission(self):
        executor = _RecoverableBatchExecutor()

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(StudyExecutionInterrupted):
                run_study(
                    _plan(2),
                    work_dir=tmpdir,
                    resume=False,
                    task_executor=executor,
                    resource_profile='memory_intensive',
                )
            receipt = Path(tmpdir) / 'batch-study' / 'execution-receipt.json'
            self.assertTrue(receipt.is_file())
            receipt_payload = json.loads(receipt.read_text(encoding='utf-8'))

            report = run_study(
                _plan(2),
                work_dir=tmpdir,
                resume=False,
                task_executor=executor,
                resource_profile='cpu_intensive',
            )

        self.assertEqual(executor.submit_count, 1)
        self.assertEqual(executor.submitted_profiles, ['memory_intensive'])
        self.assertEqual(executor.wait_count, 2)
        self.assertEqual(executor.status_many_count, 1)
        self.assertEqual(
            receipt_payload['requested_resource_profile'],
            'memory_intensive',
        )
        self.assertEqual(
            report.execution['resource_profiles'],
            ['memory_intensive'],
        )
        self.assertEqual(report.status, 'succeeded')
        self.assertEqual(len(report.cases), 2)
        self.assertIn(
            'execution-receipt',
            [artifact['kind'] for artifact in report.artifacts],
        )

    def test_interrupted_sequential_case_resumes_before_next_case(self):
        executor = _RecoverableBatchExecutor()

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(StudyExecutionInterrupted):
                run_study(
                    _plan(2),
                    work_dir=tmpdir,
                    resume=False,
                    task_executor=executor,
                    batch_independent=False,
                )
            first_receipt = (
                Path(tmpdir)
                / 'batch-study'
                / 'execution-receipts'
                / 'case-0001'
                / 'execution-receipt.json'
            )
            self.assertTrue(first_receipt.is_file())

            report = run_study(
                _plan(2),
                work_dir=tmpdir,
                resume=False,
                task_executor=executor,
                batch_independent=False,
            )
            second_receipt = (
                Path(tmpdir)
                / 'batch-study'
                / 'execution-receipts'
                / 'case-0002'
                / 'execution-receipt.json'
            )
            self.assertTrue(second_receipt.is_file())

        self.assertEqual(executor.submit_count, 2)
        self.assertEqual(executor.wait_count, 3)
        self.assertEqual(report.status, 'succeeded')
        self.assertEqual(len(report.cases), 2)
        self.assertEqual(
            [artifact['kind'] for artifact in report.artifacts].count('execution-receipt'),
            2,
        )


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
