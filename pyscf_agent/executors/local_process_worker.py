from __future__ import annotations

from pyscf_agent.timestamps import utc_timestamp as _utc_timestamp

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .contracts import JobHandle, JobState, JobStatus
from .local import JOB_REPORT_FILENAME, LocalExecutor, _write_json_atomic
from .local_process import LocalProcessExecutor
from .local_process_control import (
    CANCEL_FILENAME, CHILD_PROCESS_FILENAME, PROCESS_FILENAME,
    group_is_running, stop_process_group, write_process_identity,
)


def _load_request(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError('Local process request is unreadable: {0}'.format(path)) from exc
    if not isinstance(payload, dict):
        raise TypeError('Local process request must be a JSON object')
    return payload


def _execute_request_file(request_file: str, worker_token: str) -> Dict[str, Any]:
    """Only this subprocess enters the numerical runtime; it owns no job status."""
    payload = _load_request(Path(request_file))
    if payload.get('worker_token') != worker_token:
        raise ValueError('Worker token does not match the task request')
    handle = JobHandle.from_dict(payload.get('handle'))
    report = LocalExecutor().execute_task(
        payload.get('request'), channel=str(payload.get('channel') or 'web'),
        locale=str(payload.get('locale') or 'en'), work_dir=handle.work_dir, run_id=handle.run_id,
    )
    _write_json_atomic(LocalProcessExecutor._report_path(handle), report, kind='task-report')
    return report


def run_request_file(request_file: str, worker_token: str, *, popen_factory: Any = subprocess.Popen) -> None:
    """Supervise one calculation independently of the submitting Web process."""
    path = Path(request_file).expanduser().resolve()
    payload = _load_request(path)
    if payload.get('worker_token') != worker_token:
        raise ValueError('Worker token does not match the task request')
    handle = JobHandle.from_dict(payload.get('handle'))
    run_dir = LocalProcessExecutor._job_dir(handle)
    started_at = _utc_timestamp()
    started = time.monotonic()
    wall_time = payload.get('wall_time_seconds')
    grace = float(payload.get('terminate_timeout', 5.0))
    interrupted = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, _frame: interrupted.append(signum))

    def cancellation_requested() -> bool:
        if interrupted:
            return True
        cancel_path = run_dir / CANCEL_FILENAME
        if not cancel_path.exists():
            return False
        cancel = _load_request(cancel_path)
        if (cancel.get('worker_token') != worker_token
                or JobHandle.from_dict(cancel.get('handle')) != handle):
            raise ValueError('Cancellation request does not match the running task')
        return True

    def finish(state: JobState, message: str, report: Optional[Dict[str, Any]] = None) -> None:
        LocalProcessExecutor._write_status(JobStatus(
            handle=handle, state=state, updated_at=_utc_timestamp(), started_at=started_at,
            completed_at=_utc_timestamp(), task_status=report.get('execution_status') if report else None,
            report_available=report is not None, message=message,
        ))

    process = None
    try:
        # Startup failure is terminal only before any calculation was launched.
        LocalProcessExecutor(wall_time_seconds=wall_time, terminate_timeout=grace)
        write_process_identity(run_dir / PROCESS_FILENAME, handle, os.getpid(), worker_token)
        if cancellation_requested():
            finish(JobState.CANCELLED, 'Cancelled before numerical execution started.')
            return
        LocalProcessExecutor._write_status(JobStatus(
            handle=handle, state=JobState.RUNNING, updated_at=started_at, started_at=started_at,
            message='Running under an independent local process supervisor.',
        ))
        process = popen_factory([
            sys.executable, '-m', 'pyscf_agent.executors.local_process_worker',
            '--execute-request-file', str(path), '--worker-token', worker_token,
        ], stdin=subprocess.DEVNULL, start_new_session=True)
        write_process_identity(run_dir / CHILD_PROCESS_FILENAME, handle, process.pid, worker_token)
        reason = None
        while process.poll() is None:
            if cancellation_requested():
                reason = 'cancelled'
            elif wall_time is not None and time.monotonic() - started >= float(wall_time):
                reason = 'timeout'
            if reason:
                stop_process_group(process.pid, grace)
                process.wait(timeout=grace)
                break
            time.sleep(0.1)
        # Reap/stop descendants before publishing any terminal job state.
        if group_is_running(process.pid):
            stop_process_group(process.pid, grace)
        process.wait(timeout=grace)
        if reason == 'cancelled':
            finish(JobState.CANCELLED, 'Calculation process group stopped by the user.')
        elif reason == 'timeout':
            finish(JobState.FAILED, 'Local wall-time limit exceeded ({0:g} seconds); calculation process group stopped.'.format(wall_time))
        elif process.returncode != 0:
            finish(JobState.FAILED, 'Local calculation exited with code {0}; see worker logs.'.format(process.returncode))
        else:
            report = _load_request(run_dir / JOB_REPORT_FILENAME)
            if report.get('run_id') != handle.run_id:
                raise ValueError('TaskReport belongs to a different run')
            finish(JobState.COMPLETED, 'TaskReport is available.', report)
    except BaseException as exc:
        # If termination itself fails, leave the job nonterminal. Claiming a
        # completed cancellation while a child is alive would conceal work.
        if process is not None:
            stop_process_group(process.pid, grace)
            process.wait(timeout=grace)
        finish(JobState.FAILED, '{0}: {1}'.format(type(exc).__name__, exc))
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Supervise one cancellable local PySCF Agent task')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--request-file')
    mode.add_argument('--execute-request-file')
    parser.add_argument('--worker-token', required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute_request_file:
        _execute_request_file(args.execute_request_file, args.worker_token)
    else:
        run_request_file(args.request_file, args.worker_token)
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
