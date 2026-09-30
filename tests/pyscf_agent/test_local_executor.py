from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pyscf_agent.executors import (
    JOB_HANDLE_SCHEMA,
    JOB_STATUS_SCHEMA,
    ExecutorContractError,
    JobNotReadyError,
    JobState,
    LocalExecutor,
    TaskExecutor,
)


class LocalExecutorTests(unittest.TestCase):
    def test_execute_task_forwards_context_and_returns_task_report(self):
        calls = []

        def runner(request, **kwargs):
            calls.append((request, kwargs))
            return {
                'task_report': {
                    'execution_status': 'succeeded',
                    'structured_results': {'final_energy': -1.0},
                },
            }

        executor = LocalExecutor(request_runner=runner)
        report = executor.execute_task(
            '{"task_type":"molecular"}',
            channel='test',
            locale='en',
            work_dir='runs',
            run_id='local-test',
        )

        self.assertIsInstance(executor, TaskExecutor)
        self.assertEqual(report['execution_status'], 'succeeded')
        self.assertEqual(calls[0][1], {
            'channel': 'test',
            'locale': 'en',
            'work_dir': 'runs',
            'run_id': 'local-test',
        })

    def test_accepts_injected_request_runner(self):
        class Runner:
            @staticmethod
            def execute_request(request, **_kwargs):
                return {'task_report': {'execution_status': 'blocked', 'request': request}}

        report = LocalExecutor(request_runner=Runner().execute_request).execute_task('request')

        self.assertEqual(report['execution_status'], 'blocked')
        self.assertEqual(report['request'], 'request')

    def test_invalid_runner_result_raises_contract_error(self):
        with self.assertRaises(ExecutorContractError):
            LocalExecutor(request_runner=lambda *_args, **_kwargs: {}).execute_task('request')

    def test_descriptor_does_not_claim_queue_or_cancel_support(self):
        descriptor = LocalExecutor(
            request_runner=lambda *_args, **_kwargs: {'task_report': {}},
        ).describe()

        self.assertEqual(descriptor['executor_id'], 'local')
        self.assertEqual(descriptor['execution_mode'], 'synchronous')
        self.assertFalse(descriptor['supports_queue'])
        self.assertFalse(descriptor['supports_cancel'])
        self.assertTrue(descriptor['supports_job_handle'])
        self.assertEqual(descriptor['submission_mode'], 'immediate')

    def test_submit_persists_handle_status_and_task_report(self):
        def runner(request, **kwargs):
            return {
                'task_report': {
                    'schema': 'pyscf-agent.task-report.v1',
                    'run_id': kwargs['run_id'],
                    'work_dir': kwargs['work_dir'],
                    'execution_status': 'blocked',
                    'logs': [{'event': 'validation_completed'}],
                    'artifacts': [{'kind': 'generated_input', 'path': 'input.py'}],
                    'request': request,
                },
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            executor = LocalExecutor(request_runner=runner)
            handle = executor.submit_task(
                'request',
                work_dir=tmpdir,
                run_id='submitted-local-test',
            )
            reader = LocalExecutor(
                request_runner=lambda *_args, **_kwargs: {'task_report': {}},
            )
            status = reader.status(handle.to_dict())
            report = reader.fetch(handle)
            logs = reader.logs(handle)
            artifacts = reader.artifacts(handle)

            run_dir = Path(tmpdir) / 'submitted-local-test'
            self.assertTrue((run_dir / 'job-state.json').is_file())
            self.assertTrue((run_dir / 'job-task-report.json').is_file())

        self.assertEqual(handle.to_dict()['schema'], JOB_HANDLE_SCHEMA)
        self.assertEqual(status.to_dict()['schema'], JOB_STATUS_SCHEMA)
        self.assertEqual(status.state, JobState.COMPLETED)
        self.assertEqual(status.task_status, 'blocked')
        self.assertTrue(status.report_available)
        self.assertEqual(report['request'], 'request')
        self.assertEqual(logs, [{'event': 'validation_completed'}])
        artifact_kinds = [item['kind'] for item in artifacts]
        self.assertEqual(artifact_kinds[:2], ['job-state', 'task-report'])
        self.assertIn('generated_input', artifact_kinds)
        self.assertTrue(all(item.get('size_bytes') for item in artifacts[:2]))
        self.assertTrue(all('sha256' not in item for item in artifacts[:2]))

    def test_submit_persists_executor_failure_without_fabricating_task_report(self):
        def runner(_request, **_kwargs):
            raise RuntimeError('calculation process stopped')

        with tempfile.TemporaryDirectory() as tmpdir:
            executor = LocalExecutor(request_runner=runner)
            handle = executor.submit_task(
                'request',
                work_dir=tmpdir,
                run_id='failed-local-test',
            )
            status = executor.status(handle)

            self.assertEqual(status.state, JobState.FAILED)
            self.assertFalse(status.report_available)
            self.assertIn('calculation process stopped', status.message)
            with self.assertRaises(JobNotReadyError):
                executor.fetch(handle)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
