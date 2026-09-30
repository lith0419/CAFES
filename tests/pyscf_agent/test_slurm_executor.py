from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pyscf_agent.executors import (
    BATCH_HANDLE_SCHEMA,
    BatchHandle,
    BatchTask,
    BatchTaskExecutor,
    JobHandle,
    JobState,
    JobStatus,
    LocalExecutor,
    SlurmExecutor,
)
from pyscf_agent.executors.slurm_worker import (
    SLURM_BATCH_MANIFEST_SCHEMA,
    run_manifest_task,
)


class _CommandResult:
    def __init__(self, stdout='', stderr='', returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class SlurmExecutorTests(unittest.TestCase):
    def test_completed_job_with_delayed_report_is_never_replaced_by_a_failure(self):
        from dataclasses import replace
        from pyscf_agent.executors.base import JobNotReadyError
        with tempfile.TemporaryDirectory() as root:
            executor = SlurmExecutor(command_runner=lambda args: _CommandResult(stdout='42\n'))
            batch = executor.submit_batch([BatchTask(task_id='h2', request={}, work_dir=root, run_id='h2')])
            handle = batch.jobs[0]
            status = JobStatus(handle=handle, state=JobState.COMPLETED,
                               updated_at='2026-09-06T00:00:00Z', report_available=False)
            with mock.patch.object(executor, 'status', return_value=status):
                with self.assertRaisesRegex(JobNotReadyError, 'not yet available'):
                    executor.fetch_independent_tasks([batch])
            path = executor._report_path(handle)
            self.assertFalse(path.exists())
            self.assertFalse(executor._status_path(handle).exists())
            path.parent.mkdir(parents=True, exist_ok=True)
            expected = {'execution_status': 'succeeded', 'run_id': handle.run_id}
            path.write_text(json.dumps(expected))
            status = replace(status, report_available=True)
            with mock.patch.object(executor, 'status', return_value=status):
                result = executor.fetch_independent_tasks([batch])
            self.assertEqual(result.reports['h2'], expected)

    def test_worker_module_entrypoint_is_not_preloaded_by_package_imports(self):
        result = subprocess.run(
            [
                sys.executable,
                '-W',
                'error',
                '-m',
                'pyscf_agent.executors.slurm_worker',
                '--help',
            ],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('RuntimeWarning', result.stderr)

    def test_submit_batch_writes_manifest_and_array_mapping(self):
        commands = []

        def run_command(args):
            commands.append(tuple(args))
            return _CommandResult(stdout='4815;cluster\n')

        with tempfile.TemporaryDirectory() as tmpdir:
            tasks = [
                BatchTask(
                    task_id='case-0001',
                    request='{"task_type":"molecular","method":"mp2"}',
                    work_dir=str(Path(tmpdir) / 'cases'),
                    run_id='case-0001',
                ),
                BatchTask(
                    task_id='case-0002',
                    request='{"task_type":"molecular","method":"mp2"}',
                    work_dir=str(Path(tmpdir) / 'cases'),
                    run_id='case-0002',
                ),
            ]
            executor = SlurmExecutor(
                command_runner=run_command,
                max_parallel=2,
                task_command_prefix=('time', 'srun'),
                job_name_prefix='TenghuiLi',
                profile_sbatch_args={
                    'molecular.mp2': ('--cpus-per-task=4', '--mem=8G'),
                },
            )

            handle = executor.submit_batch(tasks)

            manifest = json.loads(
                Path(handle.manifest_path).read_text(encoding='utf-8')
            )
            persisted_handle = json.loads(
                (Path(handle.manifest_path).parent / 'batch-handle.json').read_text(
                    encoding='utf-8'
                )
            )
            script = (
                Path(handle.manifest_path).parent / 'run-array.sh'
            ).read_text(encoding='utf-8')

        self.assertIsInstance(executor, BatchTaskExecutor)
        self.assertEqual(handle.backend_job_id, '4815')
        self.assertEqual(
            [item.backend_job_id for item in handle.jobs],
            ['4815_0', '4815_1'],
        )
        self.assertEqual(persisted_handle['schema'], BATCH_HANDLE_SCHEMA)
        self.assertEqual(manifest['schema'], SLURM_BATCH_MANIFEST_SCHEMA)
        self.assertEqual(
            [item['task_id'] for item in manifest['tasks']],
            ['case-0001', 'case-0002'],
        )
        self.assertIn('--array=0-1%2', commands[0])
        self.assertIn('--cpus-per-task=4', commands[0])
        self.assertIn('--mem=8G', commands[0])
        self.assertIn('--job-name=TenghuiLi-molecular.mp2', commands[0])
        self.assertIn('time srun ', script)
        self.assertIn('-m pyscf_agent.executors.slurm_worker', script)

    def test_explicit_profile_overrides_per_task_automatic_profiles(self):
        commands = []

        def run_command(args):
            commands.append(tuple(args))
            return _CommandResult(stdout='1\n')

        with tempfile.TemporaryDirectory() as tmpdir:
            tasks = [
                BatchTask(
                    task_id='mp2',
                    request='{"task_type":"molecular","method":"mp2"}',
                    work_dir=tmpdir,
                ),
                BatchTask(
                    task_id='fci',
                    request='{"task_type":"molecular","method":"fci"}',
                    work_dir=tmpdir,
                ),
            ]
            executor = SlurmExecutor(
                command_runner=run_command,
                profile_sbatch_args={'memory_intensive': ('--mem=64G',)},
            )

            handle = executor.submit_batch(
                tasks,
                profile_id='memory_intensive',
            )

        self.assertEqual(handle.profile_id, 'memory_intensive')
        self.assertIn('--mem=64G', commands[0])

    def test_automatic_method_profile_resolves_to_configured_standard_profile(self):
        task = BatchTask(
            task_id='case-0001',
            request={'task_type': 'molecular', 'method': 'mp2'},
            work_dir='runs',
        )
        executor = SlurmExecutor(
            profile_sbatch_args={
                'standard': ('--mem=8G',),
                'memory_intensive': ('--mem=64G',),
            },
        )

        self.assertEqual(
            executor._requested_or_automatic_profile(task, 'auto'),
            'standard',
        )

    def test_independent_tasks_are_grouped_by_profile_and_chunked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            executor = SlurmExecutor(
                command_runner=lambda _args: _CommandResult(stdout='1\n'),
                max_array_size=2,
            )
            tasks = [
                BatchTask(
                    task_id='mp2-{0}'.format(index),
                    request='{"task_type":"molecular","method":"mp2"}',
                    work_dir=str(root / 'cases'),
                    run_id='mp2-{0}'.format(index),
                )
                for index in range(3)
            ] + [
                BatchTask(
                    task_id='fci-{0}'.format(index),
                    request='{"task_type":"molecular","method":"fci"}',
                    work_dir=str(root / 'cases'),
                    run_id='fci-{0}'.format(index),
                )
                for index in range(2)
            ]
            submitted = []

            def submit_batch(batch_tasks, *, profile_id=None):
                batch_index = len(submitted) + 1
                batch_root = root / 'batch-{0}'.format(batch_index)
                batch_root.mkdir()
                manifest_path = batch_root / 'batch-manifest.json'
                manifest_path.write_text('{}', encoding='utf-8')
                jobs = [
                    JobHandle(
                        job_id='batch-{0}:{1}'.format(batch_index, task.task_id),
                        executor_id='slurm',
                        run_id=task.run_id,
                        work_dir=task.work_dir,
                        submitted_at='2026-07-30T00:00:00+00:00',
                        backend_job_id='{0}_{1}'.format(batch_index, index),
                    )
                    for index, task in enumerate(batch_tasks)
                ]
                handle = BatchHandle(
                    batch_id='batch-{0}'.format(batch_index),
                    executor_id='slurm',
                    backend_job_id=str(batch_index),
                    submitted_at='2026-07-30T00:00:00+00:00',
                    profile_id=profile_id,
                    manifest_path=str(manifest_path),
                    jobs=jobs,
                )
                (batch_root / 'batch-handle.json').write_text(
                    json.dumps(handle.to_dict()),
                    encoding='utf-8',
                )
                submitted.append(
                    (profile_id, [task.task_id for task in batch_tasks])
                )
                return handle

            with mock.patch.object(
                executor,
                'submit_batch',
                side_effect=submit_batch,
            ), mock.patch.object(
                executor,
                'wait_for_jobs',
            ), mock.patch.object(
                executor,
                'fetch',
                side_effect=lambda handle: {
                    'execution_status': 'succeeded',
                    'job_id': handle.job_id,
                },
            ):
                result = executor.execute_independent_tasks(tasks)

        self.assertEqual(submitted, [
            ('molecular.mp2', ['mp2-0', 'mp2-1']),
            ('molecular.mp2', ['mp2-2']),
            ('molecular.fci', ['fci-0', 'fci-1']),
        ])
        self.assertEqual(set(result.reports), {task.task_id for task in tasks})
        self.assertEqual(len(result.batches), 3)

    def test_worker_executes_one_manifest_entry_and_persists_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / 'batch-manifest.json'
            task = BatchTask(
                task_id='case-0002',
                request='request-2',
                channel='study',
                locale='en',
                work_dir=str(root / 'cases'),
                run_id='case-0002',
            )
            manifest_path.write_text(
                json.dumps({
                    'schema': SLURM_BATCH_MANIFEST_SCHEMA,
                    'batch_id': 'batch-test',
                    'executor_id': 'slurm',
                    'profile_id': 'molecular.mp2',
                    'submitted_at': '2026-07-30T00:00:00+00:00',
                    'work_dir': str(root / 'cases'),
                    'tasks': [task.to_dict()],
                }),
                encoding='utf-8',
            )
            local = LocalExecutor(request_runner=lambda request, **kwargs: {
                'task_report': {
                    'execution_status': 'succeeded',
                    'run_id': kwargs['run_id'],
                    'request': request,
                },
            })

            with mock.patch.dict(
                'os.environ',
                {
                    'SLURM_ARRAY_JOB_ID': '4815',
                    'SLURM_ARRAY_TASK_ID': '0',
                },
            ):
                report = run_manifest_task(
                    str(manifest_path),
                    0,
                    executor=local,
                )

            run_dir = root / 'cases' / 'case-0002'
            status_payload = json.loads(
                (run_dir / 'job-state.json').read_text(encoding='utf-8')
            )
            report_payload = json.loads(
                (run_dir / 'job-task-report.json').read_text(encoding='utf-8')
            )

        self.assertEqual(report['request'], 'request-2')
        self.assertEqual(report_payload['execution_status'], 'succeeded')
        self.assertEqual(
            report_payload['execution']['resource_profile'],
            'molecular.mp2',
        )
        self.assertEqual(status_payload['state'], JobState.COMPLETED.value)
        self.assertEqual(
            status_payload['handle']['backend_job_id'],
            '4815_0',
        )

    def test_worker_executes_the_compiled_module_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / 'batch-manifest.json'
            task = BatchTask(
                task_id='case-runtime',
                request=json.dumps({
                    'atom': 'H 0 0 0; H 0 0 0.74',
                    'basis': 'sto-3g',
                    'method': 'hf',
                    'restricted': True,
                    'outputs': ['energy'],
                }),
                channel='study',
                locale='en',
                work_dir=str(root / 'cases'),
                run_id='case-runtime',
            )
            manifest_path.write_text(
                json.dumps({
                    'schema': SLURM_BATCH_MANIFEST_SCHEMA,
                    'batch_id': 'batch-runtime',
                    'executor_id': 'slurm',
                    'profile_id': 'molecular.hf',
                    'submitted_at': '2026-08-07T00:00:00+00:00',
                    'work_dir': str(root / 'cases'),
                    'tasks': [task.to_dict()],
                }),
                encoding='utf-8',
            )

            report = run_manifest_task(str(manifest_path), 0)

        self.assertEqual(report['execution_status'], 'succeeded')
        self.assertEqual(
            [entry['module_id'] for entry in report['module_execution_trace']],
            list(report['workflow_configuration']['execution_order']),
        )
        self.assertTrue(all(
            entry['workflow_id'] == report['workflow_configuration']['workflow_id']
            for entry in report['module_execution_trace']
        ))

    def test_status_falls_back_to_scheduler_when_persisted_state_is_unreadable(self):
        commands = []

        def run_command(args):
            commands.append(tuple(args))
            if args[0] == 'squeue':
                return _CommandResult(stdout='RUNNING\n')
            return _CommandResult()

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir) / 'cases'
            run_dir = work_dir / 'case-0001'
            run_dir.mkdir(parents=True)
            (run_dir / 'job-state.json').write_text('{', encoding='utf-8')
            handle = JobHandle(
                job_id='batch-test:case-0001',
                executor_id='slurm',
                run_id='case-0001',
                work_dir=str(work_dir),
                submitted_at='2026-08-04T00:00:00+00:00',
                backend_job_id='4815_0',
            )
            executor = SlurmExecutor(command_runner=run_command)

            status = executor.status(handle)

        self.assertEqual(status.state, JobState.RUNNING)
        self.assertIn('temporarily unreadable', status.message)
        self.assertEqual(commands[0][0], 'squeue')

    def test_failed_block2_array_task_without_worker_report_becomes_task_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task = BatchTask(
                task_id='case-0001',
                request={
                    'task_type': 'model_hamiltonian',
                    'solver': {'name': 'block2_dmrg'},
                },
                work_dir=str(root / 'cases'),
                run_id='case-0001',
            )
            manifest = root / 'batch-manifest.json'
            manifest.write_text(json.dumps({
                'schema': SLURM_BATCH_MANIFEST_SCHEMA,
                'tasks': [task.to_dict()],
            }), encoding='utf-8')
            handle = JobHandle(
                job_id='batch-test:case-0001',
                executor_id='slurm',
                run_id='case-0001',
                work_dir=str(root / 'cases'),
                submitted_at='2026-08-16T00:00:00+00:00',
                backend_job_id='4815_0',
            )
            batch = BatchHandle(
                batch_id='batch-test',
                executor_id='slurm',
                backend_job_id='4815',
                submitted_at='2026-08-16T00:00:00+00:00',
                profile_id='memory_intensive',
                manifest_path=str(manifest),
                jobs=[handle],
            )
            (root / 'batch-handle.json').write_text(
                json.dumps(batch.to_dict()),
                encoding='utf-8',
            )
            (root / 'slurm-4815_0.out').write_text(
                'worker started\nlast stdout line\n',
                encoding='utf-8',
            )
            (root / 'slurm-4815_0.err').write_text(
                'CANCELLED by scheduler\n',
                encoding='utf-8',
            )
            failed = JobStatus(
                handle=handle,
                state=JobState.FAILED,
                updated_at='2026-08-16T00:05:00+00:00',
                completed_at='2026-08-16T00:05:00+00:00',
                report_available=False,
                message='Slurm state: OUT_OF_MEMORY',
            )
            executor = SlurmExecutor(command_runner=lambda _args: _CommandResult())

            with mock.patch.object(executor, 'status', return_value=failed):
                result = executor.fetch_independent_tasks([batch])

            report = result.reports['case-0001']
            persisted = json.loads(
                (root / 'cases' / 'case-0001' / 'job-task-report.json').read_text(
                    encoding='utf-8'
                )
            )

        self.assertEqual(report['execution_status'], 'failed')
        self.assertEqual(report['errors'][0]['code'], 'resource_memory_exhausted')
        recovery = report['errors'][0]['details']['recovery_recommendation']
        self.assertEqual(recovery['recommended_action'], 'increase_memory_profile')
        self.assertTrue(report['execution']['synthetic_task_report'])
        self.assertIn('last stdout line', report['raw_stderr'])
        self.assertIn('CANCELLED by scheduler', report['raw_stderr'])
        self.assertTrue(report['errors'][0]['details']['scheduler_stdout_available'])
        self.assertTrue(report['errors'][0]['details']['scheduler_stderr_available'])
        self.assertEqual(persisted['schema'], report['schema'])


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
