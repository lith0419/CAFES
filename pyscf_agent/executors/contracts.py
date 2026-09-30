from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..schema_contracts import (
    BATCH_EXECUTION_RESULT_SCHEMA,
    BATCH_HANDLE_SCHEMA,
    BATCH_TASK_SCHEMA,
    JOB_HANDLE_SCHEMA,
    JOB_STATUS_SCHEMA,
    SLURM_BATCH_MANIFEST_SCHEMA as SLURM_BATCH_MANIFEST_SCHEMA,
)


class JobState(str, Enum):
    """Scheduler-level lifecycle state for one submitted task."""

    QUEUED = 'queued'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


TERMINAL_JOB_STATES = frozenset({
    JobState.COMPLETED,
    JobState.FAILED,
    JobState.CANCELLED,
})


def _required_text(value: Any, field_name: str) -> str:
    normalized = str(value or '').strip()
    if not normalized:
        raise ValueError('{0} must be a non-empty string'.format(field_name))
    return normalized


@dataclass(frozen=True)
class JobHandle:
    """Stable, serializable identity returned when a task is submitted."""

    job_id: str
    executor_id: str
    run_id: str
    work_dir: str
    submitted_at: str
    backend_job_id: Optional[str] = None
    runtime_release_id: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name in ('job_id', 'executor_id', 'run_id', 'work_dir', 'submitted_at'):
            _required_text(getattr(self, field_name), field_name)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            'schema': JOB_HANDLE_SCHEMA,
            'job_id': self.job_id,
            'executor_id': self.executor_id,
            'run_id': self.run_id,
            'work_dir': self.work_dir,
            'submitted_at': self.submitted_at,
        }
        if self.backend_job_id:
            payload['backend_job_id'] = self.backend_job_id
        if self.runtime_release_id:
            payload['runtime_release_id'] = self.runtime_release_id
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'JobHandle':
        if not isinstance(payload, dict):
            raise TypeError('JobHandle payload must be a dictionary')
        schema = payload.get('schema')
        if schema not in (None, JOB_HANDLE_SCHEMA):
            raise ValueError('Unsupported JobHandle schema: {0}'.format(schema))
        return cls(
            job_id=_required_text(payload.get('job_id'), 'job_id'),
            executor_id=_required_text(payload.get('executor_id'), 'executor_id'),
            run_id=_required_text(payload.get('run_id'), 'run_id'),
            work_dir=_required_text(payload.get('work_dir'), 'work_dir'),
            submitted_at=_required_text(payload.get('submitted_at'), 'submitted_at'),
            backend_job_id=(
                str(payload.get('backend_job_id')).strip()
                if payload.get('backend_job_id') is not None
                else None
            ),
            runtime_release_id=(
                str(payload.get('runtime_release_id')).strip()
                if payload.get('runtime_release_id') is not None
                else None
            ),
        )


@dataclass(frozen=True)
class JobStatus:
    """Current scheduler state plus the separate scientific task status."""

    handle: JobHandle
    state: JobState
    updated_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    task_status: Optional[str] = None
    report_available: bool = False
    message: str = ''

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_JOB_STATES

    def to_dict(self) -> Dict[str, Any]:
        return {
            'schema': JOB_STATUS_SCHEMA,
            'handle': self.handle.to_dict(),
            'state': self.state.value,
            'terminal': self.terminal,
            'updated_at': self.updated_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'task_status': self.task_status,
            'report_available': bool(self.report_available),
            'message': self.message,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'JobStatus':
        if not isinstance(payload, dict):
            raise TypeError('JobStatus payload must be a dictionary')
        schema = payload.get('schema')
        if schema not in (None, JOB_STATUS_SCHEMA):
            raise ValueError('Unsupported JobStatus schema: {0}'.format(schema))
        try:
            state = JobState(str(payload.get('state') or '').strip())
        except ValueError as exc:
            raise ValueError('Unsupported job state: {0}'.format(payload.get('state'))) from exc
        return cls(
            handle=JobHandle.from_dict(payload.get('handle') or {}),
            state=state,
            updated_at=_required_text(payload.get('updated_at'), 'updated_at'),
            started_at=(
                str(payload.get('started_at')).strip()
                if payload.get('started_at') is not None
                else None
            ),
            completed_at=(
                str(payload.get('completed_at')).strip()
                if payload.get('completed_at') is not None
                else None
            ),
            task_status=(
                str(payload.get('task_status')).strip()
                if payload.get('task_status') is not None
                else None
            ),
            report_available=bool(payload.get('report_available')),
            message=str(payload.get('message') or ''),
        )


@dataclass(frozen=True)
class BatchTask:
    """One independent task supplied to a batch-capable executor."""

    task_id: str
    request: Any
    channel: str = 'study'
    locale: str = 'en'
    work_dir: Optional[str] = None
    run_id: Optional[str] = None

    def __post_init__(self) -> None:
        _required_text(self.task_id, 'task_id')
        _required_text(self.channel, 'channel')
        _required_text(self.locale, 'locale')
        if self.run_id is not None:
            _required_text(self.run_id, 'run_id')

    def to_dict(self) -> Dict[str, Any]:
        return {
            'schema': BATCH_TASK_SCHEMA,
            'task_id': self.task_id,
            'request': self.request,
            'channel': self.channel,
            'locale': self.locale,
            'work_dir': self.work_dir,
            'run_id': self.run_id,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'BatchTask':
        if not isinstance(payload, dict):
            raise TypeError('BatchTask payload must be a dictionary')
        schema = payload.get('schema')
        if schema not in (None, BATCH_TASK_SCHEMA):
            raise ValueError('Unsupported BatchTask schema: {0}'.format(schema))
        return cls(
            task_id=_required_text(payload.get('task_id'), 'task_id'),
            request=payload.get('request'),
            channel=_required_text(payload.get('channel') or 'study', 'channel'),
            locale=_required_text(payload.get('locale') or 'en', 'locale'),
            work_dir=(
                str(payload.get('work_dir')).strip()
                if payload.get('work_dir') is not None
                else None
            ),
            run_id=(
                str(payload.get('run_id')).strip()
                if payload.get('run_id') is not None
                else None
            ),
        )


@dataclass(frozen=True)
class BatchHandle:
    """Stable identity and task mapping for one scheduler batch."""

    batch_id: str
    executor_id: str
    backend_job_id: str
    submitted_at: str
    profile_id: str
    manifest_path: str
    jobs: List[JobHandle]

    def __post_init__(self) -> None:
        for field_name in (
            'batch_id',
            'executor_id',
            'backend_job_id',
            'submitted_at',
            'profile_id',
            'manifest_path',
        ):
            _required_text(getattr(self, field_name), field_name)
        if not isinstance(self.jobs, list) or not self.jobs:
            raise ValueError('jobs must be a non-empty list')
        if any(not isinstance(item, JobHandle) for item in self.jobs):
            raise TypeError('BatchHandle jobs must contain JobHandle objects')

    def to_dict(self) -> Dict[str, Any]:
        return {
            'schema': BATCH_HANDLE_SCHEMA,
            'batch_id': self.batch_id,
            'executor_id': self.executor_id,
            'backend_job_id': self.backend_job_id,
            'submitted_at': self.submitted_at,
            'profile_id': self.profile_id,
            'manifest_path': self.manifest_path,
            'jobs': [item.to_dict() for item in self.jobs],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'BatchHandle':
        if not isinstance(payload, dict):
            raise TypeError('BatchHandle payload must be a dictionary')
        schema = payload.get('schema')
        if schema not in (None, BATCH_HANDLE_SCHEMA):
            raise ValueError('Unsupported BatchHandle schema: {0}'.format(schema))
        jobs = payload.get('jobs')
        if not isinstance(jobs, list):
            raise TypeError('BatchHandle jobs must be a list')
        return cls(
            batch_id=_required_text(payload.get('batch_id'), 'batch_id'),
            executor_id=_required_text(payload.get('executor_id'), 'executor_id'),
            backend_job_id=_required_text(
                payload.get('backend_job_id'),
                'backend_job_id',
            ),
            submitted_at=_required_text(payload.get('submitted_at'), 'submitted_at'),
            profile_id=_required_text(payload.get('profile_id'), 'profile_id'),
            manifest_path=_required_text(
                payload.get('manifest_path'),
                'manifest_path',
            ),
            jobs=[JobHandle.from_dict(item) for item in jobs],
        )


@dataclass(frozen=True)
class BatchExecutionResult:
    """TaskReport mapping and scheduler evidence for one batch execution call."""

    reports: Dict[str, Dict[str, Any]]
    batches: List[Dict[str, Any]]
    artifacts: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.reports, dict):
            raise TypeError('reports must be a dictionary')
        if not isinstance(self.batches, list):
            raise TypeError('batches must be a list')
        if not isinstance(self.artifacts, list):
            raise TypeError('artifacts must be a list')
        for task_id, report in self.reports.items():
            _required_text(task_id, 'reports task_id')
            if not isinstance(report, dict):
                raise TypeError('Batch task reports must be dictionaries')
        if any(not isinstance(item, dict) for item in self.batches):
            raise TypeError('Batch execution metadata entries must be dictionaries')
        if any(not isinstance(item, dict) for item in self.artifacts):
            raise TypeError('Batch artifact references must be dictionaries')

    def to_dict(self) -> Dict[str, Any]:
        return {
            'schema': BATCH_EXECUTION_RESULT_SCHEMA,
            'reports': self.reports,
            'batches': self.batches,
            'artifacts': self.artifacts,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'BatchExecutionResult':
        if not isinstance(payload, dict):
            raise TypeError('BatchExecutionResult payload must be a dictionary')
        schema = payload.get('schema')
        if schema not in (None, BATCH_EXECUTION_RESULT_SCHEMA):
            raise ValueError(
                'Unsupported BatchExecutionResult schema: {0}'.format(schema)
            )
        reports = payload.get('reports')
        batches = payload.get('batches')
        artifacts = payload.get('artifacts')
        return cls(
            reports=dict(reports) if isinstance(reports, dict) else {},
            batches=list(batches) if isinstance(batches, list) else [],
            artifacts=list(artifacts) if isinstance(artifacts, list) else [],
        )


__all__ = [
    'BATCH_EXECUTION_RESULT_SCHEMA',
    'BATCH_HANDLE_SCHEMA',
    'BATCH_TASK_SCHEMA',
    'BatchExecutionResult',
    'BatchHandle',
    'BatchTask',
    'JOB_HANDLE_SCHEMA',
    'JOB_STATUS_SCHEMA',
    'JobHandle',
    'JobState',
    'JobStatus',
    'TERMINAL_JOB_STATES',
]
