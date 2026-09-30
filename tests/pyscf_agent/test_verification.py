from __future__ import annotations

import unittest

from pyscf_agent.executors import BatchExecutionResult
from pyscf_agent.schema_contracts import PUBLIC_CONTRACT_VERSION, TASK_REPORT_SCHEMA, public_schema_manifest
from pyscf_agent.verification import verify_remote_cluster


class _RemoteExecutor:
    def __init__(self, *, contract_version=PUBLIC_CONTRACT_VERSION, task_status='succeeded'):
        self.contract_version = contract_version
        self.task_status = task_status
        self.requests = []
        self.batch_requests = []

    def remote_capabilities(self):
        manifest = public_schema_manifest()
        manifest['contract_version'] = self.contract_version
        return {
            'executor_id': 'slurm',
            'public_contract': manifest,
        }

    def execute_task(self, request, **options):
        self.requests.append((request, options))
        return self._report()

    def _report(self):
        return {
            'schema': TASK_REPORT_SCHEMA,
            'execution_status': self.task_status,
            'task_spec': {},
            'attempts': [],
            'errors': [],
            'artifacts': [],
        }

    def execute_independent_tasks(self, tasks):
        self.batch_requests.append(list(tasks))
        return BatchExecutionResult(
            reports={task.task_id: self._report() for task in tasks},
            batches=[{'batch_id': 'batch-smoke', 'jobs': [task.task_id for task in tasks]}],
            artifacts=[
                {'kind': 'slurm-batch-manifest'},
                {'kind': 'slurm-batch-handle'},
            ],
        )


class _UnavailableRemoteExecutor:
    def remote_capabilities(self):
        raise OSError('host unavailable')


class VerificationTests(unittest.TestCase):
    def test_remote_capability_contract_passes_without_submission(self):
        report = verify_remote_cluster('cluster', executor=_RemoteExecutor())

        self.assertEqual(report['status'], 'passed')
        self.assertFalse(report['smoke_submitted'])
        self.assertEqual(len(report['checks']), 2)

    def test_remote_smoke_validates_task_report(self):
        executor = _RemoteExecutor()
        report = verify_remote_cluster(
            'cluster',
            executor=executor,
            submit_smoke=True,
        )

        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['smoke']['execution_status'], 'succeeded')
        self.assertIn('"method": "hf"', executor.requests[0][0])

    def test_remote_contract_mismatch_fails_verification(self):
        executor = _RemoteExecutor(contract_version='0.9')
        report = verify_remote_cluster(
            'cluster',
            executor=executor,
            submit_batch_smoke=True,
        )

        self.assertEqual(report['status'], 'failed')
        self.assertFalse(report['checks'][1]['passed'])
        self.assertFalse(executor.batch_requests)
        self.assertEqual(
            report['batch_smoke']['submission_blocked'],
            'remote_public_contract',
        )

    def test_remote_batch_smoke_validates_independent_task_reports_and_evidence(self):
        executor = _RemoteExecutor()
        report = verify_remote_cluster(
            'cluster',
            executor=executor,
            submit_batch_smoke=True,
        )

        self.assertEqual(report['status'], 'passed')
        self.assertTrue(report['batch_smoke_submitted'])
        self.assertEqual(
            report['batch_smoke']['task_ids'],
            ['remote-batch-h2-070', 'remote-batch-h2-090'],
        )
        self.assertEqual(len(executor.batch_requests[0]), 2)

    def test_remote_transport_failure_returns_structured_report(self):
        report = verify_remote_cluster(
            'cluster',
            executor=_UnavailableRemoteExecutor(),
        )

        self.assertEqual(report['status'], 'failed')
        self.assertIn('host unavailable', report['checks'][0]['error'])


if __name__ == '__main__':
    unittest.main()
