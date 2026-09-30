from __future__ import annotations

from pyscf_agent.serialization import json_default

from pyscf_agent.timestamps import utc_timestamp as _utc_timestamp

import json
import math
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

from pyscf_agent.identifiers import make_run_id
from pyscf_agent.paths import resolve_work_dir
from ..contracts import TaskReport
from .base import ExecutorContractError, JobConflictError, JobNotFoundError, JobNotReadyError
from .contracts import JobHandle, JobState, JobStatus
from .local import (
    JOB_STATE_FILENAME,
    LocalExecutor,
    _write_json_atomic,
)
from .local_process_control import (
    CANCEL_FILENAME, CHILD_PROCESS_FILENAME, PROCESS_FILENAME,
    group_is_running, identity_is_running, read_process_identity, stop_process_group,
)


LOCAL_PROCESS_REQUEST_FILENAME = 'job-request.json'
LOCAL_PROCESS_STDOUT_FILENAME = 'job-worker-stdout.log'
LOCAL_PROCESS_STDERR_FILENAME = 'job-worker-stderr.log'


class LocalProcessExecutor(LocalExecutor):
    """Run Web calculations in cancellable child processes."""

    executor_id = 'local-process'
    execution_mode = 'local_process'
    submission_mode = 'subprocess'

    def __init__(
        self,
        *,
        python_executable: Optional[str] = None,
        popen_factory: Any = subprocess.Popen,
        terminate_timeout: float = 5.0,
        wall_time_seconds: Optional[float] = None,
    ) -> None:
        super().__init__()
        self._python_executable = str(python_executable or sys.executable)
        self._popen_factory = popen_factory
        self._terminate_timeout = float(terminate_timeout)
        self._wall_time_seconds = None if wall_time_seconds is None else float(wall_time_seconds)
        if not math.isfinite(self._terminate_timeout) or self._terminate_timeout <= 0:
            raise ValueError('terminate_timeout must be finite and positive')
        if self._wall_time_seconds is not None and (
            not math.isfinite(self._wall_time_seconds) or self._wall_time_seconds <= 0
        ):
            raise ValueError('wall_time_seconds must be finite and positive')
        self._processes: Dict[Path, Any] = {}
        self._process_lock = threading.Lock()

    @classmethod
    def _coerce_handle(cls, handle: Any) -> JobHandle:
        if isinstance(handle, JobHandle):
            normalized = handle
        elif isinstance(handle, dict):
            normalized = JobHandle.from_dict(handle)
        else:
            raise TypeError('handle must be a JobHandle or dictionary')
        if normalized.executor_id != cls.executor_id:
            raise JobNotFoundError(
                'Job {0} belongs to executor {1}, not {2}'.format(
                    normalized.job_id,
                    normalized.executor_id,
                    cls.executor_id,
                )
            )
        return normalized

    def submit_task(
        self,
        request: Any,
        *,
        channel: str = 'agent',
        locale: str = 'en',
        work_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        resource_profile: Optional[str] = None,
    ) -> JobHandle:
        if str(resource_profile or '').strip().lower() not in ('', 'auto'):
            raise ValueError('Local execution does not use Slurm resource profiles')
        if os.name != 'posix':
            raise ValueError('Supervised local execution requires POSIX process groups')
        normalized_run_id = str(run_id or '').strip() or make_run_id()
        submitted_at = _utc_timestamp()
        root = resolve_work_dir(work_dir, normalized_run_id)
        handle = JobHandle(
            job_id=normalized_run_id,
            executor_id=self.executor_id,
            run_id=normalized_run_id,
            work_dir=str(root),
            submitted_at=submitted_at,
        )
        run_dir = self._job_dir(handle)
        from ..artifacts.arrays import compact_request
        request = compact_request(request, run_dir / 'arrays')
        status_path = run_dir / JOB_STATE_FILENAME
        run_dir.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        request_path = run_dir / LOCAL_PROCESS_REQUEST_FILENAME
        # Exclusive creation also prevents two Web processes submitting the
        # same run between a status-file check and worker launch.
        if status_path.exists():
            raise JobConflictError('Job already exists: {0}'.format(handle.job_id))
        try:
            request_stream = request_path.open('x', encoding='utf-8')
        except FileExistsError as exc:
            raise JobConflictError('Job already exists: {0}'.format(handle.job_id)) from exc
        with request_stream:
            json.dump({
                'handle': handle.to_dict(),
                'request': request,
                'channel': channel,
                'locale': locale,
                'worker_token': token,
                'wall_time_seconds': self._wall_time_seconds,
                'terminate_timeout': self._terminate_timeout,
            }, request_stream, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)
        self._write_status(JobStatus(
            handle=handle,
            state=JobState.QUEUED,
            updated_at=submitted_at,
            message='Accepted by the local process executor.',
        ))
        stdout_path = run_dir / LOCAL_PROCESS_STDOUT_FILENAME
        stderr_path = run_dir / LOCAL_PROCESS_STDERR_FILENAME
        try:
            with stdout_path.open('ab') as stdout_stream, stderr_path.open('ab') as stderr_stream:
                process = self._popen_factory(
                    (
                        self._python_executable,
                        '-m',
                        'pyscf_agent.executors.local_process_worker',
                        '--request-file',
                        str(request_path),
                        '--worker-token',
                        token,
                    ),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    start_new_session=True,
                )
        except OSError as exc:
            failed_at = _utc_timestamp()
            self._write_status(JobStatus(
                handle=handle, state=JobState.FAILED, updated_at=failed_at,
                completed_at=failed_at, message='Could not start local worker: {0}'.format(exc),
            ))
            raise
        with self._process_lock:
            self._processes[self._job_dir(handle)] = process
        return handle

    def execute_task(self, request: Any, **kwargs: Any) -> Dict[str, Any]:
        """Synchronous callers use the same supervised worker and time limit."""
        handle = self.submit_task(request, **kwargs)
        status = self.status(handle)
        while not status.terminal:
            time.sleep(0.1)
            status = self.status(handle)
        return self.fetch(handle)

    def fetch(self, handle: Any) -> Dict[str, Any]:
        handle = self._coerce_handle(handle)
        status = self.status(handle)
        if status.terminal and not status.report_available:
            # Close a reserved attempt after timeout/cancellation, including
            # collection by a different process after the submitting client exits.
            saved = json.loads((self._job_dir(handle) / LOCAL_PROCESS_REQUEST_FILENAME).read_text(encoding='utf-8'))
            request = saved['request']
            try:
                spec = json.loads(request) if isinstance(request, str) else request
            except (TypeError, ValueError):
                spec = {'raw_request': str(request)}
            if not isinstance(spec, dict):
                spec = {'raw_request': str(request)}
            report = TaskReport(
                run_id=handle.run_id, work_dir=handle.work_dir,
                channel=saved.get('channel', 'agent'), task_spec=spec,
                execution_status='failed', analysis_summary=status.message,
                errors=[{'stage': 'execution', 'code': 'local_process_' + status.state.value,
                         'message': status.message, 'details': {'job_handle': handle.to_dict()}}],
            ).to_dict()
            _write_json_atomic(self._report_path(handle), report, kind='task-report')
            _write_json_atomic(self._job_dir(handle) / 'task-report.json', report, kind='task-report')
            self._write_status(replace(status, report_available=True, task_status='failed'))
        return super().fetch(handle)

    def status(self, handle: Any) -> JobStatus:
        normalized = self._coerce_handle(handle)
        status = super().status(normalized)
        if status.terminal:
            self._forget_process(normalized)
            return status
        with self._process_lock:
            process = self._processes.get(self._job_dir(normalized))
        if process is None:
            identity = read_process_identity(self._job_dir(normalized) / PROCESS_FILENAME, normalized)
            if identity is None or identity_is_running(identity):
                return status
            return_code = 'unknown (worker no longer running)'
        else:
            return_code = process.poll()
            if return_code is None:
                return status
        # The worker normally writes a terminal status before exiting. Re-read
        # once to avoid overwriting that status after process.poll().
        status = super().status(normalized)
        if status.terminal:
            self._forget_process(normalized)
            return status
        # A dead supervisor may have left a calculation behind, including when
        # the original submitting executor still holds its Popen object.
        child = read_process_identity(self._job_dir(normalized) / CHILD_PROCESS_FILENAME, normalized)
        if child is not None:
            if identity_is_running(child):
                stop_process_group(child['pgid'], self._terminate_timeout)
            elif group_is_running(child['pgid']):
                raise ExecutorContractError(
                    'Supervisor and calculation leader exited but descendants remain; termination is unconfirmed'
                )
        failed_at = _utc_timestamp()
        failed = JobStatus(
            handle=normalized,
            state=JobState.FAILED,
            updated_at=failed_at,
            started_at=status.started_at,
            completed_at=failed_at,
            message='Local calculation worker exited with code {0}.'.format(return_code),
        )
        self._write_status(failed)
        self._forget_process(normalized)
        return failed

    def cancel(self, handle: Any) -> JobStatus:
        normalized = self._coerce_handle(handle)
        current = self.status(normalized)
        if current.terminal:
            return current
        with self._process_lock:
            process = self._processes.get(self._job_dir(normalized))
        identity = read_process_identity(self._job_dir(normalized) / PROCESS_FILENAME, normalized)
        if identity is None and (process is None or isinstance(getattr(process, 'pid', None), int)):
            # The supervisor persists its identity before spawning numerical
            # work. Wait for that startup handshake, including after reconnect.
            deadline = time.monotonic() + 2.0
            while identity is None and time.monotonic() < deadline:
                time.sleep(0.05)
                current = self.status(normalized)
                if current.terminal:
                    return current
                identity = read_process_identity(self._job_dir(normalized) / PROCESS_FILENAME, normalized)
        if identity is not None:
            if not identity_is_running(identity):
                return self.status(normalized)
            _write_json_atomic(self._job_dir(normalized) / CANCEL_FILENAME, {
                'handle': normalized.to_dict(), 'worker_token': identity['token'],
                'requested_at': _utc_timestamp(),
            }, kind='job-state')
            # The supervisor acknowledges only after the entire calculation
            # process group has stopped. This also works after a Web restart.
            deadline = time.monotonic() + 2 * self._terminate_timeout + 5.0
            while time.monotonic() < deadline:
                status = self.status(normalized)
                if status.terminal:
                    return status
                time.sleep(0.05)
            raise JobNotReadyError('Cancellation requested; worker has not confirmed termination')
        if process is None or isinstance(getattr(process, 'pid', None), int):
            raise ExecutorContractError(
                'Worker identity is unavailable; cannot confirm cancellation of this legacy or starting job'
            )
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self._terminate_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=self._terminate_timeout)
        cancelled_at = _utc_timestamp()
        cancelled = JobStatus(
            handle=normalized,
            state=JobState.CANCELLED,
            updated_at=cancelled_at,
            started_at=current.started_at,
            completed_at=cancelled_at,
            message='Local calculation process was stopped by the user.',
        )
        self._write_status(cancelled)
        self._forget_process(normalized)
        return cancelled

    def _forget_process(self, handle: JobHandle) -> None:
        with self._process_lock:
            process = self._processes.get(self._job_dir(handle))
            if process is not None:
                try:
                    process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    return
                self._processes.pop(self._job_dir(handle), None)

    def describe(self) -> Dict[str, Any]:
        return {
            'executor_id': self.executor_id,
            'execution_mode': self.execution_mode,
            'submission_mode': self.submission_mode,
            'location': 'child_process',
            'supports_queue': False,
            'supports_cancel': True,
            'supports_job_handle': True,
            'job_state_filename': JOB_STATE_FILENAME,
            'supports_restart_control': True,
            'wall_time_seconds': self._wall_time_seconds,
        }


__all__ = [
    'LOCAL_PROCESS_REQUEST_FILENAME',
    'LOCAL_PROCESS_STDERR_FILENAME',
    'LOCAL_PROCESS_STDOUT_FILENAME',
    'LocalProcessExecutor',
]
