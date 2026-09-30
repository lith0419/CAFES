from __future__ import annotations

import io
import base64
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pyscf_agent.executors import BatchHandle, JobHandle, JobState, JobStatus
from pyscf_agent.remote.rpc_cli import dispatch_rpc, main
from computational_study_agent.hamiltonian_dataset_postprocessing import (
    HAMILTONIAN_DATASET_GENERATION_REQUEST_SCHEMA,
)
from computational_study_agent.hamiltonian_dataset_contracts import (
    HAMILTONIAN_MANIFEST_SCHEMA,
)


class _FakeSlurmExecutor:
    def __init__(self):
        self.tasks = []

    def describe(self):
        return {'executor_id': 'slurm'}

    def submit_task(self, request, **kwargs):
        self.single_request = request
        self.single_kwargs = kwargs
        return JobHandle(
            job_id='single-test',
            executor_id='slurm',
            run_id=kwargs['run_id'],
            work_dir=kwargs['work_dir'],
            submitted_at='2026-08-03T00:00:00+00:00',
            backend_job_id='2_0',
        )

    def submit_independent_tasks(self, tasks, **kwargs):
        self.tasks = list(tasks)
        self.batch_kwargs = kwargs
        job = JobHandle(
            job_id='batch-test:{0}'.format(tasks[0].task_id),
            executor_id='slurm',
            run_id=tasks[0].run_id,
            work_dir=tasks[0].work_dir,
            submitted_at='2026-08-03T00:00:00+00:00',
            backend_job_id='1_0',
        )
        return [BatchHandle(
            batch_id='batch-test',
            executor_id='slurm',
            backend_job_id='1',
            submitted_at='2026-08-03T00:00:00+00:00',
            profile_id='molecular.mp2',
            manifest_path=str(Path(tasks[0].work_dir).parent / 'manifest.json'),
            jobs=[job],
        )]

    def status(self, handle):
        return JobStatus(
            handle=handle,
            state=JobState.COMPLETED,
            updated_at='2026-08-03T00:01:00+00:00',
            completed_at='2026-08-03T00:01:00+00:00',
            report_available=True,
        )


class RemoteRpcTests(unittest.TestCase):
    def test_recovery_reads_server_owned_evidence_without_submitting_again(self):
        from pyscf_agent.serialization import json_fingerprint
        executor = _FakeSlurmExecutor()
        task = {'task_id': 'case-1', 'request': {'method': 'hf'},
                'run_id': 'case-1', 'channel': 'study', 'locale': 'en'}
        with tempfile.TemporaryDirectory() as root:
            submitted = dispatch_rpc('submit-independent', {'tasks': [task]},
                                     executor=executor, work_root=root)
            evidence_path = Path(executor.tasks[0].work_dir).parent / 'submission.json'
            executor.submit_independent_tasks = mock.Mock(side_effect=AssertionError('Must not submit'))
            recovered = dispatch_rpc('recover-submission', {'submission_path': str(evidence_path)},
                                     executor=executor, work_root=root)
            self.assertEqual(recovered['batches'], submitted['batches'])
            self.assertEqual(recovered['task_fingerprints'], {
                'case-1': json_fingerprint({key: task[key] for key in ('request', 'channel', 'locale', 'run_id')})})
            with self.assertRaises(ValueError):
                dispatch_rpc('recover-submission', {'submission_path': str(Path(root).parent / 'submission.json')},
                             executor=executor, work_root=root)
            executor.submit_independent_tasks.assert_not_called()

    def test_server_generates_dataset_under_environment_work_root(self):
        import numpy as np

        executor = _FakeSlurmExecutor()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'remote-submissions' / 'run' / 'trajectory.npz'
            source.parent.mkdir(parents=True)
            np.savez(
                source,
                fock_matrices=np.eye(2)[None, :, :],
                overlap_matrices=np.eye(2)[None, :, :],
            )
            matrix = {
                'path': str(source),
                'array_index': 0,
                'shape': [2, 2],
            }
            request = {
                'schema': HAMILTONIAN_DATASET_GENERATION_REQUEST_SCHEMA,
                'study_id': 'study-1',
                'source_root': '/client/path-is-not-trusted',
                'source_manifest': {
                    'schema': HAMILTONIAN_MANIFEST_SCHEMA,
                    'spec': {'dataset_id': 'qh9-small'},
                    'artifacts': {},
                },
                'samples': [{
                    'sample_id': 'mol-1:frame-000000',
                    'molecule_id': 'mol-1',
                    'fock': {**matrix, 'array_key': 'fock_matrices'},
                    'overlap': {**matrix, 'array_key': 'overlap_matrices'},
                }],
                'rejections': [],
            }

            result = dispatch_rpc(
                'generate-hamiltonian-dataset',
                request,
                executor=executor,
                work_root=directory,
            )

            dataset_root = Path(result['dataset_root'])
            self.assertTrue(
                dataset_root.resolve().is_relative_to((root / 'datasets').resolve())
            )
            self.assertEqual(result['location'], 'remote_executor')
            self.assertTrue((dataset_root / 'dataset-manifest.json').is_file())
            self.assertTrue((dataset_root / 'arrays' / 'trajectory-0001-mol-1.npz').is_file())

    def test_server_materializes_checked_input_attachment_and_rewrites_request(self):
        executor = _FakeSlurmExecutor()
        content = b'MODEL_SPEC = {"model": "hubbard"}\n'
        source_path = '/client/run/input-builder-model-hamiltonian.py'
        request = json.dumps({
            'task_type': 'model_hamiltonian',
            'model_hamiltonian_input_file': source_path,
        })
        with tempfile.TemporaryDirectory() as directory:
            dispatch_rpc(
                'submit-task',
                {
                    'request': request,
                    'run_id': 'model-task',
                    'attachments': [{
                        'source_path': source_path,
                        'filename': 'input-builder-model-hamiltonian.py',
                        'size_bytes': len(content),
                        'content_base64': base64.b64encode(content).decode('ascii'),
                    }],
                },
                executor=executor,
                work_root=directory,
            )
            rewritten = json.loads(executor.single_request)
            remote_path = Path(rewritten['model_hamiltonian_input_file'])

            self.assertTrue(remote_path.is_relative_to(Path(directory).resolve()))
            self.assertEqual(remote_path.read_bytes(), content)
            self.assertEqual(remote_path.stat().st_mode & 0o777, 0o600)

    def test_server_rebases_batch_work_directories_under_work_root(self):
        executor = _FakeSlurmExecutor()
        with tempfile.TemporaryDirectory() as directory:
            data = dispatch_rpc(
                'submit-independent',
                {'tasks': [{
                    'schema': 'pyscf-agent.batch-task.v1',
                    'task_id': 'case-0001',
                    'request': {'method': 'mp2'},
                    'channel': 'study',
                    'locale': 'en',
                    'work_dir': '/client/path/that/must/not/be-used',
                    'run_id': '../../escape',
                }]},
                executor=executor,
                work_root=directory,
            )
            task = executor.tasks[0]
            resolved_work_dir = Path(task.work_dir).resolve()

            self.assertTrue(resolved_work_dir.is_relative_to(Path(directory).resolve()))
            self.assertNotIn('/client/path', task.work_dir)
            self.assertNotIn('..', task.run_id)
            self.assertEqual(data['batches'][0]['jobs'][0]['executor_id'], 'slurm')

    def test_server_forwards_selected_resource_profile(self):
        executor = _FakeSlurmExecutor()
        with tempfile.TemporaryDirectory() as directory:
            dispatch_rpc(
                'submit-independent',
                {
                    'resource_profile': 'memory_intensive',
                    'tasks': [{
                        'schema': 'pyscf-agent.batch-task.v1',
                        'task_id': 'case-0001',
                        'request': {'method': 'mp2'},
                        'channel': 'study',
                        'locale': 'en',
                    }],
                },
                executor=executor,
                work_root=directory,
            )

        self.assertEqual(
            executor.batch_kwargs['resource_profile'],
            'memory_intensive',
        )

    def test_batch_status_returns_serialized_statuses(self):
        executor = _FakeSlurmExecutor()
        handle = JobHandle(
            job_id='batch-test:case-0001',
            executor_id='slurm',
            run_id='case-0001',
            work_dir='/cluster/work',
            submitted_at='2026-08-03T00:00:00+00:00',
            backend_job_id='1_0',
        )
        data = dispatch_rpc(
            'batch-status',
            {'jobs': [handle.to_dict()]},
            executor=executor,
            work_root='/cluster/work',
        )

        self.assertEqual(data['statuses'][0]['state'], 'completed')
        self.assertTrue(data['statuses'][0]['report_available'])

    def test_cli_writes_one_versioned_json_response(self):
        executor = _FakeSlurmExecutor()
        config = SimpleNamespace(
            connection=SimpleNamespace(mode='direct'),
            remote=SimpleNamespace(work_root='/cluster/work'),
        )
        output = io.StringIO()
        with mock.patch(
            'pyscf_agent.remote.rpc_cli.load_slurm_executor_config',
            return_value=config,
        ), mock.patch(
            'pyscf_agent.remote.rpc_cli.SlurmExecutor.from_config',
            return_value=executor,
        ), mock.patch('sys.stdin', io.StringIO('{}')), mock.patch('sys.stdout', output):
            return_code = main((
                'capabilities',
                '--slurm-config',
                '/cluster/slurm.ini',
            ))

        lines = output.getvalue().splitlines()
        self.assertEqual(return_code, 0)
        self.assertEqual(len(lines), 1)
        response = json.loads(lines[0])
        self.assertTrue(response['ok'])
        self.assertEqual(response['schema'], 'pyscf-agent.remote-rpc.v1')
        self.assertEqual(response['data']['public_contract']['contract_version'], '1.0')


if __name__ == '__main__':
    unittest.main()
