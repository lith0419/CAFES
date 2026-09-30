from __future__ import annotations

import json
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pyscf_agent.executors import BatchTask, JobHandle, JobState, SshSlurmExecutor
from pyscf_agent.executors.remote_config import SshRemoteProfile
from pyscf_agent.remote.protocol import RemoteProtocolError, success_response
from pyscf_agent.runtime_identity import build_runtime_identity


class _CommandResult:
    def __init__(self, payload, returncode=0, stderr='', prefix=''):
        self.stdout = '{0}{1}'.format(prefix, json.dumps(payload))
        self.stderr = stderr
        self.returncode = returncode


def _profile(
    *,
    report_propagation_timeout_seconds=120,
    expected_cluster_id=None,
    runtime_identity_file=None,
    require_runtime_match=False,
    local_source_root=None,
) -> SshRemoteProfile:
    return SshRemoteProfile(
        source_path='/client/remote.ini',
        profile_id='amarel',
        host='amarel.example.edu',
        port=22,
        username='scientist',
        private_key='/home/scientist/.ssh/id_ed25519',
        known_hosts='/home/scientist/.ssh/known_hosts',
        strict_host_key_checking=True,
        connect_timeout_seconds=10,
        remote_python='/cluster/env/bin/python',
        remote_slurm_config='/cluster/config/slurm.ini',
        poll_interval_seconds=0,
        wait_timeout_seconds=5,
        report_propagation_timeout_seconds=report_propagation_timeout_seconds,
        control_persist_seconds=300,
        remote_slurm_profile='amarel',
        expected_cluster_id=expected_cluster_id,
        runtime_identity_file=runtime_identity_file,
        require_runtime_match=require_runtime_match,
        local_source_root=local_source_root,
    )


def _job(task_id='case-0001') -> JobHandle:
    return JobHandle(
        job_id='batch-test:{0}'.format(task_id),
        executor_id='slurm',
        run_id=task_id,
        work_dir='/cluster/work/cases',
        submitted_at='2026-08-03T00:00:00+00:00',
        backend_job_id='4815_0',
    )


class SshSlurmExecutorTests(unittest.TestCase):
    def test_task_inspection_uses_one_rpc_and_preserves_handle_identity(self):
        job = _job()
        calls = []
        view = {'handle': job.to_dict(), 'status': {'state': 'running'}, 'solver_progress': None}

        def run_command(args, input_text):
            calls.append((list(args), json.loads(input_text)))
            return _CommandResult(success_response('inspect-task', view))

        executor = SshSlurmExecutor(_profile(), command_runner=run_command)
        self.assertEqual(executor.inspect_task(job), view)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], {'handle': job.to_dict()})
        self.assertIn('inspect-task', calls[0][0][-1])
        view['handle'] = _job('different').to_dict()
        with self.assertRaisesRegex(RemoteProtocolError, 'does not match'):
            executor.inspect_task(job)

    def test_generate_dataset_uses_remote_rpc_without_sftp(self):
        calls = []
        response = {
            'schema': 'pyscf-agent.hamiltonian-dataset-generation.v1',
            'status': 'succeeded',
            'location': 'remote_executor',
            'dataset_root': '/scratch/environment/datasets/test',
            'files': [],
        }

        def run_command(args, input_text):
            calls.append((list(args), json.loads(input_text)))
            return _CommandResult(
                success_response('generate-hamiltonian-dataset', response)
            )

        executor = SshSlurmExecutor(_profile(), command_runner=run_command)
        returned = executor.generate_hamiltonian_dataset({'schema': 'request.v1'})

        self.assertEqual(returned, response)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][0], 'ssh')
        self.assertTrue(calls[0][0][-1].endswith('generate-hamiltonian-dataset'))
        self.assertEqual(calls[0][1]['schema'], 'request.v1')

    def test_collect_files_downloads_remote_artifacts_with_one_sftp_batch(self):
        calls = []

        def run_command(args, input_text):
            calls.append((list(args), input_text))
            for line in input_text.splitlines():
                fields = shlex.split(line)
                Path(fields[2]).write_bytes(b'npz-data')
            return _CommandResult({})

        with tempfile.TemporaryDirectory() as directory:
            destination = str(Path(directory) / 'arrays' / 'trajectory-0001.npz')
            executor = SshSlurmExecutor(_profile(), command_runner=run_command)

            transfers = executor.collect_files({
                '/scratch/study/case-0001/result-molecular-md-frames.npz': destination,
            })

            self.assertEqual(Path(destination).read_bytes(), b'npz-data')

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][0], 'sftp')
        self.assertIn('get "/scratch/study/case-0001/', calls[0][1])
        self.assertEqual(transfers[0]['path'], str(Path(destination).resolve()))

    def test_submit_task_uses_fixed_remote_rpc_and_json_stdin(self):
        calls = []
        handle = _job()

        def run_command(args, input_text):
            calls.append((list(args), json.loads(input_text)))
            return _CommandResult(success_response('submit-task', handle.to_dict()))

        executor = SshSlurmExecutor(_profile(), command_runner=run_command)
        returned = executor.submit_task(
            'calculate nitrogen',
            channel='cli',
            locale='en',
            work_dir='/client/must-not-cross-boundary',
            run_id='n2-point',
        )

        args, payload = calls[0]
        self.assertEqual(returned, handle)
        self.assertIn('scientist@amarel.example.edu', args)
        self.assertIn('ConnectTimeout=10', args)
        self.assertNotIn('ConnectTimeout=10.0', args)
        self.assertIn('pyscf_agent.remote.rpc_cli', args[-1])
        self.assertIn('--slurm-profile amarel', args[-1])
        self.assertTrue(args[-1].endswith('submit-task'))
        self.assertNotIn('calculate nitrogen', args[-1])
        self.assertNotIn('work_dir', payload)
        self.assertEqual(payload['request'], 'calculate nitrogen')

    def test_submit_task_sends_selected_resource_profile(self):
        captured = {}
        handle = _job()

        def run_command(_args, input_text):
            captured.update(json.loads(input_text))
            return _CommandResult(success_response('submit-task', handle.to_dict()))

        executor = SshSlurmExecutor(_profile(), command_runner=run_command)
        executor.submit_task('request', resource_profile='cpu_intensive')

        self.assertEqual(captured['resource_profile'], 'cpu_intensive')

    def test_remote_capabilities_accept_matching_cluster_identity(self):
        def run_command(args, _input_text):
            operation = args[-1].split()[-1]
            return _CommandResult(success_response(operation, {
                'cluster_id': 'amarel',
            }))

        executor = SshSlurmExecutor(
            _profile(expected_cluster_id='amarel'),
            command_runner=run_command,
        )

        self.assertEqual(executor.remote_capabilities()['cluster_id'], 'amarel')

    def test_remote_capabilities_reject_mismatched_cluster_identity(self):
        def run_command(args, _input_text):
            operation = args[-1].split()[-1]
            return _CommandResult(success_response(operation, {
                'cluster_id': 'secondary',
            }))

        executor = SshSlurmExecutor(
            _profile(expected_cluster_id='amarel'),
            command_runner=run_command,
        )

        with self.assertRaisesRegex(
            RemoteProtocolError,
            'expected Slurm cluster amarel',
        ):
            executor.remote_capabilities()

    def test_strict_runtime_binding_matches_before_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'pyproject.toml').write_text('[project]\nname="demo"\n', encoding='utf-8')
            (root / 'module.py').write_text('VALUE = 1\n', encoding='utf-8')
            identity = build_runtime_identity(
                root,
                environment_id='worktree-5872',
                release_id='release-001',
            )
            private = root / '.pyscf-agent'
            private.mkdir()
            identity_path = private / 'runtime-identity.json'
            identity_path.write_text(json.dumps(identity.to_dict()), encoding='utf-8')
            operations = []

            def run_command(args, _input_text):
                operation = args[-1].split()[-1]
                operations.append(operation)
                data = (
                    {'cluster_id': 'amarel', 'runtime_identity': identity.to_dict()}
                    if operation == 'capabilities'
                    else _job().to_dict()
                )
                return _CommandResult(success_response(operation, data))

            executor = SshSlurmExecutor(
                _profile(
                    expected_cluster_id='amarel',
                    runtime_identity_file=str(identity_path),
                    require_runtime_match=True,
                    local_source_root=str(root),
                ),
                command_runner=run_command,
            )
            executor.submit_task('request')

        self.assertEqual(operations, ['capabilities', 'submit-task'])

    def test_running_executor_reloads_redeployed_worktree_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'pyproject.toml').write_text(
                '[project]\nname="demo"\n',
                encoding='utf-8',
            )
            identity = build_runtime_identity(
                root,
                environment_id='worktree-5872',
                release_id='release-002',
            )
            private = root / '.pyscf-agent'
            private.mkdir()
            identity_path = private / 'runtime-identity.json'
            identity_path.write_text(
                json.dumps(identity.to_dict()),
                encoding='utf-8',
            )
            config_path = private / 'remote.ini'
            config_path.write_text('''
[remote:amarel]
host = amarel.example.edu
username = scientist
remote_python = /cluster/releases/release-001/bin/python
remote_slurm_config = /cluster/server-slurm.ini
remote_slurm_profile = amarel-worktree-5872
expected_cluster_id = amarel
runtime_identity_file = runtime-identity.json
require_runtime_match = true
local_source_root = ..
''', encoding='utf-8')
            executor = SshSlurmExecutor.from_config('amarel', str(config_path))
            config_path.write_text(
                config_path.read_text(encoding='utf-8').replace(
                    'release-001/bin/python',
                    'release-002/bin/python',
                ),
                encoding='utf-8',
            )
            calls = []

            def run_command(args, _input_text):
                calls.append(list(args))
                operation = args[-1].split()[-1]
                data = (
                    {'cluster_id': 'amarel', 'runtime_identity': identity.to_dict()}
                    if operation == 'capabilities'
                    else _job().to_dict()
                )
                return _CommandResult(success_response(operation, data))

            executor._command_runner = run_command
            executor.submit_task('request')

        self.assertEqual(len(calls), 2)
        self.assertTrue(all(
            command[-1].startswith(
                '/cluster/releases/release-002/bin/python '
            )
            for command in calls
        ))

    def test_strict_runtime_binding_rejects_different_remote_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'pyproject.toml').write_text('[project]\nname="demo"\n', encoding='utf-8')
            identity = build_runtime_identity(
                root,
                environment_id='worktree-5872',
                release_id='release-local',
            )
            private = root / '.pyscf-agent'
            private.mkdir()
            identity_path = private / 'runtime-identity.json'
            identity_path.write_text(json.dumps(identity.to_dict()), encoding='utf-8')
            remote_payload = identity.to_dict()
            remote_payload['release_id'] = 'release-remote'

            def run_command(args, _input_text):
                operation = args[-1].split()[-1]
                return _CommandResult(success_response(operation, {
                    'cluster_id': 'amarel',
                    'runtime_identity': remote_payload,
                }))

            executor = SshSlurmExecutor(
                _profile(
                    expected_cluster_id='amarel',
                    runtime_identity_file=str(identity_path),
                    require_runtime_match=True,
                    local_source_root=str(root),
                ),
                command_runner=run_command,
            )
            with self.assertRaisesRegex(RemoteProtocolError, 'does not match remote release'):
                executor.submit_task('request')

    def test_ignores_login_banner_before_rpc_json(self):
        handle = _job()

        def run_command(_args, _input_text):
            return _CommandResult(
                success_response('submit-task', handle.to_dict()),
                prefix='Authorized users only\nCluster maintenance Friday\n',
            )

        executor = SshSlurmExecutor(_profile(), command_runner=run_command)

        self.assertEqual(executor.submit_task('request'), handle)

    def test_model_input_file_is_sent_as_checked_attachment(self):
        handle = _job()
        captured = {}

        def run_command(_args, input_text):
            captured.update(json.loads(input_text))
            return _CommandResult(success_response('submit-task', handle.to_dict()))

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / 'model.py'
            input_path.write_text('MODEL_SPEC = {"model": "hubbard"}\n', encoding='utf-8')
            request = json.dumps({
                'task_type': 'model_hamiltonian',
                'model_hamiltonian_input_file': str(input_path),
            })
            executor = SshSlurmExecutor(_profile(), command_runner=run_command)
            executor.submit_task(request)

        attachment = captured['attachments'][0]
        self.assertEqual(attachment['source_path'], str(input_path))
        self.assertEqual(attachment['filename'], 'model.py')
        self.assertGreater(attachment['size_bytes'], 0)
        self.assertNotIn('sha256', attachment)

    def test_inline_remote_mps_manifest_crosses_without_file_attachment(self):
        handle = _job()
        captured = {}

        def run_command(_args, input_text):
            captured.update(json.loads(input_text))
            return _CommandResult(success_response('submit-task', handle.to_dict()))

        manifest = {
            'schema': 'pyscf-agent.block2-mps-manifest.v1',
            'scratch_directory': '/cluster/work/source-case/block2-scratch',
            'n_orbitals': 2,
            'n_electrons': [1, 1],
            'spin': 0,
            'symmetry': 'su2',
            'nroots': 1,
            'files': [{'relative_path': 'GS-MPS_INFO'}],
        }
        request = {
            'solver': {
                'name': 'block2_dmrg',
                'options': {'restart_manifest': manifest},
            },
        }
        SshSlurmExecutor(_profile(), command_runner=run_command).submit_task(request)

        self.assertEqual(captured['request']['solver']['options']['restart_manifest'], manifest)
        self.assertEqual(captured['attachments'], [])

    def test_remote_mps_restart_rejects_path_string(self):
        executor = SshSlurmExecutor(_profile(), command_runner=lambda *_args: None)
        request = {
            'solver': {
                'name': 'block2_dmrg',
                'options': {'restart_manifest': '/client/checkpoints/mps.json'},
            },
        }

        with self.assertRaisesRegex(ValueError, 'inline checkpoint manifest'):
            executor.submit_task(request)

    def test_independent_tasks_submit_poll_and_fetch(self):
        job = _job()
        batch = {
            'schema': 'pyscf-agent.batch-handle.v1',
            'batch_id': 'batch-test',
            'executor_id': 'slurm',
            'backend_job_id': '4815',
            'submitted_at': '2026-08-03T00:00:00+00:00',
            'profile_id': 'molecular.mp2',
            'manifest_path': '/cluster/work/batch-manifest.json',
            'jobs': [job.to_dict()],
        }
        operations = []

        def run_command(args, _input_text):
            operation = args[-1].split()[-1]
            operations.append(operation)
            if operation == 'submit-independent':
                data = {'batches': [batch]}
            elif operation == 'batch-status':
                data = {'statuses': [{
                    'schema': 'pyscf-agent.job-status.v1',
                    'handle': job.to_dict(),
                    'state': JobState.COMPLETED.value,
                    'terminal': True,
                    'updated_at': '2026-08-03T00:01:00+00:00',
                    'started_at': None,
                    'completed_at': '2026-08-03T00:01:00+00:00',
                    'task_status': 'succeeded',
                    'report_available': True,
                    'message': 'done',
                }]}
            else:
                data = {
                    'schema': 'pyscf-agent.batch-execution-result.v1',
                    'reports': {'case-0001': {'execution_status': 'succeeded'}},
                    'batches': [batch],
                    'artifacts': [],
                }
            return _CommandResult(success_response(operation, data))

        executor = SshSlurmExecutor(
            _profile(),
            command_runner=run_command,
            sleeper=lambda _seconds: None,
        )
        result = executor.execute_independent_tasks([
            BatchTask(task_id='case-0001', request={'method': 'mp2'}),
        ])

        self.assertEqual(
            operations,
            ['submit-independent', 'batch-status', 'batch-fetch'],
        )
        self.assertEqual(result.reports['case-0001']['execution_status'], 'succeeded')

    def test_status_many_uses_one_batch_status_rpc(self):
        jobs = [_job('case-0001'), _job('case-0002')]
        calls = []

        def run_command(args, input_text):
            operation = args[-1].split()[-1]
            calls.append((operation, json.loads(input_text)))
            statuses = [
                {
                    'schema': 'pyscf-agent.job-status.v1',
                    'handle': job.to_dict(),
                    'state': JobState.COMPLETED.value,
                    'terminal': True,
                    'updated_at': '2026-08-03T00:01:00+00:00',
                    'started_at': None,
                    'completed_at': '2026-08-03T00:01:00+00:00',
                    'task_status': 'succeeded',
                    'report_available': True,
                    'message': 'done',
                }
                for job in jobs
            ]
            return _CommandResult(success_response(operation, {'statuses': statuses}))

        executor = SshSlurmExecutor(_profile(), command_runner=run_command)
        statuses = executor.status_many(jobs)

        self.assertEqual([status.handle for status in statuses], jobs)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], 'batch-status')
        self.assertEqual(len(calls[0][1]['jobs']), 2)

    def test_completed_job_waits_until_task_report_is_visible(self):
        job = _job()
        status_calls = []

        def run_command(args, _input_text):
            operation = args[-1].split()[-1]
            if operation == 'batch-status':
                status_calls.append(operation)
                report_available = len(status_calls) > 1
                data = {'statuses': [{
                    'schema': 'pyscf-agent.job-status.v1',
                    'handle': job.to_dict(),
                    'state': JobState.COMPLETED.value,
                    'terminal': True,
                    'updated_at': '2026-08-03T00:01:00+00:00',
                    'started_at': None,
                    'completed_at': '2026-08-03T00:01:00+00:00',
                    'task_status': 'succeeded' if report_available else None,
                    'report_available': report_available,
                    'message': 'done',
                }]}
                return _CommandResult(success_response(operation, data))
            raise AssertionError('Unexpected RPC operation: {0}'.format(operation))

        executor = SshSlurmExecutor(
            _profile(),
            command_runner=run_command,
            sleeper=lambda _seconds: None,
        )
        executor._wait_for_jobs([job])

        self.assertEqual(len(status_calls), 2)

    def test_report_propagation_timeout_comes_from_remote_profile(self):
        job = _job()

        def run_command(args, _input_text):
            operation = args[-1].split()[-1]
            if operation != 'batch-status':
                raise AssertionError('Unexpected RPC operation: {0}'.format(operation))
            return _CommandResult(success_response(operation, {'statuses': [{
                'schema': 'pyscf-agent.job-status.v1',
                'handle': job.to_dict(),
                'state': JobState.COMPLETED.value,
                'terminal': True,
                'updated_at': '2026-08-03T00:01:00+00:00',
                'started_at': None,
                'completed_at': '2026-08-03T00:01:00+00:00',
                'task_status': None,
                'report_available': False,
                'message': 'report pending',
            }]}))

        executor = SshSlurmExecutor(
            _profile(report_propagation_timeout_seconds=0.1),
            command_runner=run_command,
            sleeper=lambda _seconds: None,
        )
        with mock.patch(
            'pyscf_agent.executors.ssh_slurm.time.monotonic',
            side_effect=[0.0, 0.2, 0.4],
        ):
            with self.assertRaisesRegex(RemoteProtocolError, 'TaskReport did not become visible'):
                executor._wait_for_jobs([job])


if __name__ == '__main__':
    unittest.main()
