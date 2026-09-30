import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from computational_study_agent.locking import LOCK_FILENAME, study_lock, study_is_locked
from computational_study_agent.execution_receipts import StudyExecutionBusy
from computational_study_agent.executor import _serialized_execution
from tests.computational_study_agent.test_retry_collection import plan_for_service


class StudyLockTests(unittest.TestCase):
    def test_foreground_lock_excludes_a_second_process_and_is_reentrant(self):
        with tempfile.TemporaryDirectory() as root:
            plan = plan_for_service()
            directory = Path(root) / plan.study_id
            code = '''
from computational_study_agent.locking import study_lock
from computational_study_agent.execution_receipts import StudyExecutionBusy
import sys
try:
    with study_lock(sys.argv[1]):
        sys.exit(2)
except StudyExecutionBusy:
    sys.exit(0)
'''
            @_serialized_execution
            def execute(plan, **kwargs):
                with study_lock(directory):
                    self.assertTrue(study_is_locked(directory))
                    result = subprocess.run([sys.executable, '-c', code, str(directory)], timeout=10)
                    self.assertEqual(result.returncode, 0)
            execute(plan, work_dir=root)
            self.assertFalse(study_is_locked(directory))
            with study_lock(directory):
                pass

    def test_lock_released_when_worker_is_killed(self):
        with tempfile.TemporaryDirectory() as root:
            ready = Path(root) / 'ready'
            code = '''
from computational_study_agent.locking import study_lock
from pathlib import Path
import sys,time
with study_lock(sys.argv[1]):
    Path(sys.argv[2]).touch()
    time.sleep(30)
'''
            child = subprocess.Popen([sys.executable, '-c', code, root, str(ready)])
            try:
                deadline = time.monotonic() + 10
                while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(ready.exists())
                with self.assertRaises(StudyExecutionBusy), study_lock(root):
                    pass
                child.kill()
                child.wait(timeout=5)
                with study_lock(root):
                    pass
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)

    def test_status_probe_cannot_cause_false_busy(self):
        with tempfile.TemporaryDirectory() as root:
            with study_lock(root):
                pass
            reader = open(Path(root) / LOCK_FILENAME, 'rb')
            self.addCleanup(reader.close)
            fcntl.flock(reader, fcntl.LOCK_SH)
            entered, started = threading.Event(), threading.Event()
            errors = []
            def writer():
                started.set()
                try:
                    with study_lock(root):
                        entered.set()
                except Exception as exc:
                    errors.append(exc)
            thread = threading.Thread(target=writer)
            thread.start()
            self.assertTrue(started.wait(2))
            self.assertFalse(study_is_locked(root))
            fcntl.flock(reader, fcntl.LOCK_UN)
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue(entered.is_set())

    def test_background_worker_adopts_the_inherited_lock(self):
        from computational_study_agent.application import background
        with tempfile.TemporaryDirectory() as root, study_lock(root) as stream:
            def run(directory):
                with study_lock(directory):
                    self.assertTrue(study_is_locked(directory))
            fd = os.dup(stream.fileno())
            with patch.object(background, 'run_invocation', side_effect=run):
                self.assertEqual(background.main(['--study-directory', root, '--lock-fd', str(fd)]), 0)
