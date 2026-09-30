from __future__ import annotations

import tempfile
import subprocess
import sys
import time
import unittest
from pathlib import Path

from pyscf_agent.executors import JobState, LocalProcessExecutor
from pyscf_agent.executors.base import ExecutorContractError
from pyscf_agent.executors.local_process_control import PROCESS_FILENAME, process_snapshot


class FakeProcess:
    def __init__(self):
        self.return_code = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.return_code

    def terminate(self):
        self.terminated = True
        self.return_code = -15

    def kill(self):
        self.killed = True
        self.return_code = -9

    def wait(self, timeout=None):
        return self.return_code


class LocalProcessExecutorTests(unittest.TestCase):
    def test_spawn_failure_persists_a_failed_job(self):
        def fail(*args, **kwargs):
            raise OSError('executable missing')
        with tempfile.TemporaryDirectory() as root:
            executor = LocalProcessExecutor(popen_factory=fail)
            with self.assertRaisesRegex(OSError, 'executable missing'):
                executor.submit_task({}, work_dir=root, run_id='spawn-failed')
            import json
            status = json.loads((Path(root) / 'spawn-failed' / 'job-state.json').read_text())
            self.assertEqual(status['state'], 'failed')

    def test_missing_identity_after_restart_does_not_falsely_cancel_legacy_worker(self):
        process = FakeProcess()
        with tempfile.TemporaryDirectory() as tmpdir:
            original = LocalProcessExecutor(popen_factory=lambda *a, **kw: process)
            handle = original.submit_task({}, work_dir=tmpdir, run_id='legacy')
            with self.assertRaisesRegex(ExecutorContractError, 'identity is unavailable'):
                LocalProcessExecutor().cancel(handle)
            self.assertEqual(original.status(handle).state, JobState.QUEUED)
            self.assertFalse(process.terminated)
            original.cancel(handle)

    def test_wall_time_rejects_nonpositive_and_nonfinite_values(self):
        for limit in (0, -1, float('nan'), float('inf')):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                LocalProcessExecutor(wall_time_seconds=limit)

    def test_submit_and_cancel_terminate_worker_and_persist_cancelled_state(self):
        process = FakeProcess()
        popen_calls = []

        def popen_factory(command, **kwargs):
            popen_calls.append((command, kwargs))
            return process

        with tempfile.TemporaryDirectory() as tmpdir:
            executor = LocalProcessExecutor(popen_factory=popen_factory)
            handle = executor.submit_task(
                '{"task_type": "molecular"}',
                channel='web',
                work_dir=tmpdir,
                run_id='cancel-me',
            )

            self.assertEqual(executor.status(handle).state, JobState.QUEUED)
            cancelled = executor.cancel(handle)
            self.assertEqual(cancelled.state, JobState.CANCELLED)
            self.assertTrue(process.terminated)
            self.assertFalse(process.killed)
            self.assertEqual(executor.status(handle).state, JobState.CANCELLED)
            self.assertTrue((Path(tmpdir) / 'cancel-me' / 'job-state.json').is_file())
            self.assertTrue(any('local_process_worker' in item for item in popen_calls[0][0]))

    def test_descriptor_advertises_real_cancellation(self):
        description = LocalProcessExecutor(popen_factory=lambda *args, **kwargs: FakeProcess()).describe()

        self.assertEqual(description['execution_mode'], 'local_process')
        self.assertTrue(description['supports_cancel'])
        self.assertEqual(description['location'], 'child_process')

    def test_same_run_id_in_different_directories_cancels_only_selected_worker(self):
        processes = [FakeProcess(), FakeProcess()]
        pending = iter(processes)
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = LocalProcessExecutor(popen_factory=lambda *_args, **_kwargs: next(pending))
            first = executor.submit_task({}, work_dir=str(Path(tmpdir) / 'first'), run_id='same-case')
            second = executor.submit_task({}, work_dir=str(Path(tmpdir) / 'second'), run_id='same-case')

            self.assertEqual(executor.cancel(first.to_dict()).state, JobState.CANCELLED)
            self.assertTrue(processes[0].terminated)
            self.assertFalse(processes[1].terminated)
            # Re-reading a terminal job must not discard the other job's process.
            self.assertEqual(executor.status(first).state, JobState.CANCELLED)
            self.assertEqual(executor.status(second).state, JobState.QUEUED)
            self.assertEqual(executor.cancel(second).state, JobState.CANCELLED)
            self.assertTrue(processes[1].terminated)

    def test_same_run_id_worker_exit_does_not_change_other_job_status(self):
        processes = [FakeProcess(), FakeProcess()]
        pending = iter(processes)
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = LocalProcessExecutor(popen_factory=lambda *_args, **_kwargs: next(pending))
            first = executor.submit_task({}, work_dir=str(Path(tmpdir) / 'first'), run_id='same-case')
            second = executor.submit_task({}, work_dir=str(Path(tmpdir) / 'second'), run_id='same-case')
            processes[0].return_code = 1

            self.assertEqual(executor.status(first).state, JobState.FAILED)
            self.assertEqual(executor.status(second).state, JobState.QUEUED)
            executor.cancel(second)
            self.assertTrue(processes[1].terminated)


class LocalProcessLifecycleTests(unittest.TestCase):
    def test_study_timeout_finishes_its_attempt_instead_of_leaving_it_pending(self):
        from computational_study_agent.application import StudyApplicationService
        from computational_study_agent.schema import StudyPlan, StudyCase
        with tempfile.TemporaryDirectory() as root:
            executor = LocalProcessExecutor(popen_factory=self.launch_fixture,
                                            terminate_timeout=0.3, wall_time_seconds=0.4)
            service = StudyApplicationService(task_executor=executor)
            plan = StudyPlan(study_id='timeout-study', name='timeout-study', objective='timeout',
                             system_type='molecular', observables=['energy'], cases=[
                StudyCase(case_id='h2', label='h2', request={
                    'task_type': 'molecular', 'atom': 'H 0 0 0; H 0 0 0.74', 'basis': 'sto-3g', 'method': 'hf'})])
            report = service.run_study(plan, work_dir=root)
            self.assertEqual(report.cases[0]['task_report']['execution_status'], 'failed')
            self.assertIn('wall-time limit', report.cases[0]['task_report']['errors'][0]['message'])
            collected = service.collect_study(plan, work_dir=root)
            self.assertEqual(collected.cases[0]['attempt_count'], 1)

    def test_dead_supervisor_is_failed_only_after_calculation_cleanup(self):
        with tempfile.TemporaryDirectory() as root:
            original = LocalProcessExecutor(popen_factory=self.launch_fixture, terminate_timeout=0.3)
            handle = original.submit_task({'fixture': 'tree'}, work_dir=root, run_id='crash')
            process = original._processes[original._job_dir(handle)]
            try:
                self.wait_for(lambda: (Path(root) / 'crash' / 'fixture-ready').exists())
                descendant = int((Path(root) / 'crash' / 'fixture-descendant.pid').read_text())
                process.kill()
                process.wait(timeout=5)
                self.assertEqual(original.status(handle).state, JobState.FAILED)
                self.assertTrue((process_snapshot(descendant) or {'state': 'Z'})['state'].startswith('Z'))
            finally:
                original.cancel(handle)
                process.wait(timeout=5)

    def test_separate_submitter_can_exit_and_a_new_client_can_cancel(self):
        import json
        with tempfile.TemporaryDirectory() as root:
            submitted = subprocess.run([sys.executable, '-c', '''
import json, sys
from pyscf_agent.executors import LocalProcessExecutor
from tests.pyscf_agent.test_local_process_executor import LocalProcessLifecycleTests
executor = LocalProcessExecutor(popen_factory=LocalProcessLifecycleTests.launch_fixture, terminate_timeout=0.3)
handle = executor.submit_task({'fixture': 'tree'}, work_dir=sys.argv[1], run_id='parent-exit')
print(json.dumps(handle.to_dict()))
''', root], capture_output=True, text=True, check=True)
            handle = json.loads(submitted.stdout)
            reloaded = LocalProcessExecutor(terminate_timeout=0.3)
            try:
                self.wait_for(lambda: (Path(root) / 'parent-exit' / 'fixture-ready').exists())
                descendant = int((Path(root) / 'parent-exit' / 'fixture-descendant.pid').read_text())
                self.assertEqual(reloaded.status(handle).state, JobState.RUNNING)
                self.assertEqual(reloaded.cancel(handle).state, JobState.CANCELLED)
                self.assertTrue((process_snapshot(descendant) or {'state': 'Z'})['state'].startswith('Z'))
            finally:
                reloaded.cancel(handle)

    @staticmethod
    def launch_fixture(command, **kwargs):
        command = list(command)
        command[2] = 'tests.pyscf_agent.local_process_fixture'
        return subprocess.Popen(command, **kwargs)

    def wait_for(self, predicate, seconds=10):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        self.fail('Real process fixture did not reach expected state')

    def test_recreated_executor_cancels_running_process_group_and_keeps_terminal_state(self):
        with tempfile.TemporaryDirectory() as root:
            original = LocalProcessExecutor(popen_factory=self.launch_fixture, terminate_timeout=0.3)
            handle = original.submit_task({'fixture': 'tree'}, work_dir=root, run_id='restart')
            process = original._processes[original._job_dir(handle)]
            try:
                self.wait_for(lambda: (Path(root) / 'restart' / 'fixture-ready').exists())
                descendant = int((Path(root) / 'restart' / 'fixture-descendant.pid').read_text())
                reloaded = LocalProcessExecutor(terminate_timeout=0.3)
                self.assertEqual(reloaded.status(handle.to_dict()).state, JobState.RUNNING)
                self.assertEqual(reloaded.cancel(handle.to_dict()).state, JobState.CANCELLED)
                self.wait_for(lambda: (process_snapshot(descendant) or {'state': 'Z'})['state'].startswith('Z'))
                process.wait(timeout=5)
                self.assertEqual(reloaded.status(handle).state, JobState.CANCELLED)
            finally:
                original.cancel(handle)
                process.wait(timeout=5)

    def test_timeout_is_enforced_without_client_polling_after_restart(self):
        with tempfile.TemporaryDirectory() as root:
            original = LocalProcessExecutor(popen_factory=self.launch_fixture,
                                            terminate_timeout=0.3, wall_time_seconds=1.0)
            handle = original.submit_task({'fixture': 'tree'}, work_dir=root, run_id='timeout')
            process = original._processes[original._job_dir(handle)]
            try:
                process.wait(timeout=10)  # No status call or in-client watchdog.
                status = LocalProcessExecutor().status(handle)
                self.assertEqual(status.state, JobState.FAILED)
                self.assertIn('wall-time limit exceeded', status.message)
                self.assertFalse(status.report_available)
            finally:
                original.cancel(handle)
                process.wait(timeout=5)

    def test_completed_report_can_be_collected_by_a_new_executor(self):
        with tempfile.TemporaryDirectory() as root:
            original = LocalProcessExecutor(popen_factory=self.launch_fixture)
            handle = original.submit_task({'duration': 0.05}, work_dir=root, run_id='complete')
            process = original._processes[original._job_dir(handle)]
            process.wait(timeout=10)
            reloaded = LocalProcessExecutor()
            self.assertEqual(reloaded.status(handle).state, JobState.COMPLETED)
            self.assertEqual(reloaded.fetch(handle)['run_id'], handle.run_id)
            self.assertEqual(reloaded.cancel(handle).state, JobState.COMPLETED)

    def test_identity_mismatch_refuses_to_control_process(self):
        import json
        with tempfile.TemporaryDirectory() as root:
            original = LocalProcessExecutor(popen_factory=self.launch_fixture, terminate_timeout=0.3)
            handle = original.submit_task({}, work_dir=root, run_id='mismatch')
            process = original._processes[original._job_dir(handle)]
            identity_path = original._job_dir(handle) / PROCESS_FILENAME
            try:
                self.wait_for(lambda: (Path(root) / 'mismatch' / 'fixture-ready').exists())
                saved = identity_path.read_text()
                identity = json.loads(saved)
                identity['token'] = 'not-this-worker'
                identity_path.write_text(json.dumps(identity))
                with self.assertRaisesRegex(ExecutorContractError, 'identity changed'):
                    LocalProcessExecutor().cancel(handle)
                self.assertIsNone(process.poll())
                identity_path.write_text(saved)
            finally:
                original.cancel(handle)
                process.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
