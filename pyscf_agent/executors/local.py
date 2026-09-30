from __future__ import annotations

from .job_paths import job_directory

import copy
import json
from pyscf_agent.timestamps import utc_timestamp as _utc_timestamp
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..artifacts import default_artifact_repository
from pyscf_agent.identifiers import make_run_id
from pyscf_agent.paths import resolve_work_dir
from ..backend.state import default_state
from ..backend.workflow import run_workflow_sequential
from .base import (
    ExecutorContractError,
    JobCancellationUnsupportedError,
    JobConflictError,
    JobNotFoundError,
    JobNotReadyError,
)
from .contracts import JobHandle, JobState, JobStatus


JOB_STATE_FILENAME = 'job-state.json'
JOB_REPORT_FILENAME = 'job-task-report.json'




def _write_json_atomic(
    path: Path,
    payload: Dict[str, Any],
    *,
    kind: str,
) -> None:
    default_artifact_repository().write_json(
        path,
        payload,
        kind=kind,
        atomic=True,
    )


def _file_artifact_ref(
    path: Path,
    *,
    kind: str,
    description: str,
) -> Dict[str, Any]:
    reference = default_artifact_repository().register_existing(
        path,
        kind=kind,
        mime_type='application/json; charset=utf-8',
        description=description,
    )
    if reference is None:
        raise FileNotFoundError('Executor artifact is unavailable: {0}'.format(path))
    return reference


def _run_local_workflow(
    request: Any,
    *,
    channel: str,
    locale: str,
    work_dir: Optional[str],
    run_id: Optional[str],
) -> Dict[str, Any]:
    state = default_state(
        request,
        channel=channel,
        locale=locale,
        work_dir=work_dir,
        run_id=run_id,
    )
    return run_workflow_sequential(state)


class LocalExecutor:
    """Execute tasks in-process and persist the common submitted-job contract."""

    executor_id = 'local'
    execution_mode = 'synchronous'
    submission_mode = 'immediate'

    def __init__(
        self,
        *,
        request_runner: Callable[..., Dict[str, Any]] = _run_local_workflow,
    ):
        if not callable(request_runner):
            raise TypeError('request_runner must be callable')
        self._request_runner = request_runner

    def execute_task(
        self,
        request: Any,
        *,
        channel: str = 'agent',
        locale: str = 'en',
        work_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        resource_profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        if str(resource_profile or '').strip().lower() not in ('', 'auto'):
            raise ValueError('Local execution does not use Slurm resource profiles')
        result = self._request_runner(
            request,
            channel=channel,
            locale=locale,
            work_dir=work_dir,
            run_id=run_id,
        )
        if not isinstance(result, dict):
            raise ExecutorContractError('Local execution runner must return a workflow-state dictionary')
        report = result.get('task_report')
        if not isinstance(report, dict):
            raise ExecutorContractError('Local execution result must contain a TaskReport dictionary')
        return report

    @staticmethod
    def _coerce_handle(handle: Any) -> JobHandle:
        if isinstance(handle, JobHandle):
            normalized = handle
        elif isinstance(handle, dict):
            normalized = JobHandle.from_dict(handle)
        else:
            raise TypeError('handle must be a JobHandle or dictionary')
        if normalized.executor_id != LocalExecutor.executor_id:
            raise JobNotFoundError(
                'Job {0} belongs to executor {1}, not local'.format(
                    normalized.job_id,
                    normalized.executor_id,
                )
            )
        return normalized

    @staticmethod
    def _job_dir(handle: JobHandle) -> Path:
        return job_directory(handle)

    @classmethod
    def _status_path(cls, handle: JobHandle) -> Path:
        return cls._job_dir(handle) / JOB_STATE_FILENAME

    @classmethod
    def _report_path(cls, handle: JobHandle) -> Path:
        return cls._job_dir(handle) / JOB_REPORT_FILENAME

    @classmethod
    def _write_status(cls, status: JobStatus) -> None:
        _write_json_atomic(
            cls._status_path(status.handle),
            status.to_dict(),
            kind='job-state',
        )

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
        """Run immediately while exposing the scheduler-neutral job lifecycle."""

        normalized_run_id = str(run_id or '').strip() or make_run_id()
        submitted_at = _utc_timestamp()
        handle = JobHandle(
            job_id=normalized_run_id,
            executor_id=self.executor_id,
            run_id=normalized_run_id,
            work_dir=str(resolve_work_dir(work_dir, normalized_run_id)),
            submitted_at=submitted_at,
        )
        status_path = self._status_path(handle)
        if status_path.exists():
            raise JobConflictError('Job already exists: {0}'.format(handle.job_id))

        self._write_status(JobStatus(
            handle=handle,
            state=JobState.QUEUED,
            updated_at=submitted_at,
            message='Accepted by the local immediate executor.',
        ))
        started_at = _utc_timestamp()
        self._write_status(JobStatus(
            handle=handle,
            state=JobState.RUNNING,
            updated_at=started_at,
            started_at=started_at,
            message='Running in the current Python process.',
        ))
        try:
            report = self.execute_task(
                request,
                channel=channel,
                locale=locale,
            work_dir=handle.work_dir,
            run_id=handle.run_id,
            resource_profile=resource_profile,
            )
        except Exception as exc:
            failed_at = _utc_timestamp()
            self._write_status(JobStatus(
                handle=handle,
                state=JobState.FAILED,
                updated_at=failed_at,
                started_at=started_at,
                completed_at=failed_at,
                message='{0}: {1}'.format(type(exc).__name__, exc),
            ))
            return handle

        _write_json_atomic(self._report_path(handle), report, kind='task-report')
        completed_at = _utc_timestamp()
        self._write_status(JobStatus(
            handle=handle,
            state=JobState.COMPLETED,
            updated_at=completed_at,
            started_at=started_at,
            completed_at=completed_at,
            task_status=str(report.get('execution_status') or 'unknown'),
            report_available=True,
            message='TaskReport is available.',
        ))
        return handle

    def status(self, handle: Any) -> JobStatus:
        normalized = self._coerce_handle(handle)
        path = self._status_path(normalized)
        if not path.is_file():
            raise JobNotFoundError('Job state does not exist: {0}'.format(normalized.job_id))
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            status = JobStatus.from_dict(payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ExecutorContractError(
                'Job state is unreadable for {0}'.format(normalized.job_id)
            ) from exc
        if status.handle != normalized:
            raise ExecutorContractError('Persisted job state does not match the supplied handle')
        return status

    def cancel(self, handle: Any) -> JobStatus:
        status = self.status(handle)
        if status.terminal:
            return status
        raise JobCancellationUnsupportedError(
            'Local immediate jobs cannot be cancelled after execution starts'
        )

    def fetch(self, handle: Any) -> Dict[str, Any]:
        status = self.status(handle)
        if not status.report_available:
            detail = status.message or 'TaskReport is not available'
            raise JobNotReadyError(
                'Job {0} is {1}: {2}'.format(
                    status.handle.job_id,
                    status.state.value,
                    detail,
                )
            )
        path = self._report_path(status.handle)
        if not path.is_file():
            raise ExecutorContractError(
                'Job {0} claims a report that does not exist'.format(status.handle.job_id)
            )
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExecutorContractError(
                'TaskReport is unreadable for {0}'.format(status.handle.job_id)
            ) from exc
        if not isinstance(payload, dict):
            raise ExecutorContractError('Persisted TaskReport must be a dictionary')
        return payload

    def inspect_task(self, handle: Any) -> Dict[str, Any]:
        from .inspection import inspect_saved_task
        return inspect_saved_task(self, handle)

    def logs(self, handle: Any) -> List[Dict[str, Any]]:
        report = self.fetch(handle)
        logs = report.get('logs')
        return copy.deepcopy(logs) if isinstance(logs, list) else []

    def artifacts(self, handle: Any) -> List[Dict[str, Any]]:
        status = self.status(handle)
        report = self.fetch(status.handle)
        artifacts = report.get('artifacts')
        scientific_artifacts = copy.deepcopy(artifacts) if isinstance(artifacts, list) else []
        return [
            _file_artifact_ref(
                self._status_path(status.handle),
                kind='job-state',
                description='Persisted scheduler-level job state',
            ),
            _file_artifact_ref(
                self._report_path(status.handle),
                kind='task-report',
                description='Persisted scientific TaskReport',
            ),
            *scientific_artifacts,
        ]

    def describe(self) -> Dict[str, Any]:
        return {
            'executor_id': self.executor_id,
            'execution_mode': self.execution_mode,
            'submission_mode': self.submission_mode,
            'location': 'current_process',
            'supports_queue': False,
            'supports_cancel': False,
            'supports_job_handle': True,
            'job_state_filename': JOB_STATE_FILENAME,
            'job_report_filename': JOB_REPORT_FILENAME,
        }


_DEFAULT_LOCAL_EXECUTOR = LocalExecutor()


def get_local_executor() -> LocalExecutor:
    """Return the process-wide local executor."""
    return _DEFAULT_LOCAL_EXECUTOR
