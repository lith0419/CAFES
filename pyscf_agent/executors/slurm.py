from __future__ import annotations

from .job_paths import job_directory

from .job_paths import collectable_status as _collectable_status

from pyscf_agent.executors.commands import run_bounded, command_timeout

from pyscf_agent.timestamps import utc_timestamp as _utc_timestamp

import copy
import json
import re
import shlex
import subprocess
import sys
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from pyscf_agent.identifiers import make_run_id
from pyscf_agent.paths import resolve_work_dir
from ..contracts import TaskReport
from ..runtime_identity import load_runtime_identity
from .base import (
    ExecutorContractError,
    JobCancellationUnsupportedError,
    JobNotFoundError,
    JobNotReadyError,
)
from .contracts import (
    SLURM_BATCH_MANIFEST_SCHEMA,
    BatchExecutionResult,
    BatchHandle,
    BatchTask,
    JobHandle,
    JobState,
    JobStatus,
)
from .local import (
    JOB_REPORT_FILENAME,
    JOB_STATE_FILENAME,
    _file_artifact_ref,
    _write_json_atomic,
)
from .slurm_config import load_slurm_executor_config


SLURM_BATCH_HANDLE_FILENAME = 'batch-handle.json'
REPORT_PROPAGATION_TIMEOUT_SECONDS = 120.0

_SLURM_MEMORY_MULTIPLIERS = {
    '': 1.0,
    'k': 1.0 / 1024.0,
    'kb': 1.0 / 1024.0,
    'kib': 1.0 / 1024.0,
    'm': 1.0,
    'mb': 1.0,
    'mib': 1.0,
    'g': 1024.0,
    'gb': 1024.0,
    'gib': 1024.0,
    't': 1024.0 * 1024.0,
    'tb': 1024.0 * 1024.0,
    'tib': 1024.0 * 1024.0,
}


def _slurm_memory_mb(value: Any) -> Optional[int]:
    """Normalize a Slurm ``--mem`` value to MiB for capability reporting."""

    match = re.fullmatch(
        r'\s*([0-9]+(?:\.[0-9]+)?)\s*([kmgt](?:i?b)?|mb)?\s*',
        str(value or ''),
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    multiplier = _SLURM_MEMORY_MULTIPLIERS.get((match.group(2) or '').lower())
    if multiplier is None:
        return None
    return max(1, int(float(match.group(1)) * multiplier))


def _memory_mb_from_sbatch_args(args: Sequence[str]) -> Optional[int]:
    value = None
    normalized = [str(item) for item in args]
    for index, item in enumerate(normalized):
        if item.startswith('--mem='):
            value = item.split('=', 1)[1]
        elif item == '--mem' and index + 1 < len(normalized):
            value = normalized[index + 1]
    return _slurm_memory_mb(value)


def _read_scheduler_output(path: Path, *, max_chars: int = 65536) -> str:
    """Read a bounded tail of scheduler output for failure evidence."""

    if not path.is_file():
        return ''
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return ''
    return text[-max_chars:].strip()

_QUEUED_STATES = {
    'PENDING',
    'CONFIGURING',
    'REQUEUED',
    'RESIZING',
}
_RUNNING_STATES = {
    'RUNNING',
    'COMPLETING',
    'SIGNALING',
    'STAGE_OUT',
}
_COMPLETED_STATES = {'COMPLETED'}
_CANCELLED_STATES = {'CANCELLED'}
_FAILED_STATES = {
    'BOOT_FAIL',
    'DEADLINE',
    'FAILED',
    'NODE_FAIL',
    'OUT_OF_MEMORY',
    'PREEMPTED',
    'REVOKED',
    'TIMEOUT',
}


def _default_command_runner(args: Sequence[str]) -> subprocess.CompletedProcess:
    return run_bounded(
        list(args),
        capture_output=True,
        check=False,
        text=True,
    )


def _normalized_slurm_state(value: Any) -> str:
    return str(value or '').strip().upper().split('+', 1)[0].split(' ', 1)[0]


def _job_state(value: Any) -> Optional[JobState]:
    state = _normalized_slurm_state(value)
    if state in _QUEUED_STATES:
        return JobState.QUEUED
    if state in _RUNNING_STATES:
        return JobState.RUNNING
    if state in _COMPLETED_STATES:
        return JobState.COMPLETED
    if state in _CANCELLED_STATES:
        return JobState.CANCELLED
    if state in _FAILED_STATES:
        return JobState.FAILED
    return None





def _profile_component(value: Any, fallback: str) -> str:
    if isinstance(value, dict):
        value = value.get('name') or value.get('id') or value.get('method')
    normalized = str(value or fallback).strip().lower().replace('-', '_')
    return normalized or fallback


def _profile_display_name(profile_id: str) -> str:
    return profile_id.replace('.', ' ').replace('_', ' ').replace('-', ' ').title()


def _default_profile(task: BatchTask) -> str:
    payload: Dict[str, Any] = {}
    if isinstance(task.request, dict):
        payload = task.request
    elif isinstance(task.request, str):
        try:
            decoded = json.loads(task.request)
        except json.JSONDecodeError:
            decoded = {}
        if isinstance(decoded, dict):
            payload = decoded
    system_type = _profile_component(
        payload.get('task_type')
        or payload.get('system_type')
        or 'task',
        'task',
    )
    method = _profile_component(
        payload.get('solver')
        or payload.get('method')
        or 'default',
        'default',
    )
    return '{0}.{1}'.format(system_type, method)


def _request_payload(request: Any) -> Dict[str, Any]:
    if isinstance(request, dict):
        return copy.deepcopy(request)
    if isinstance(request, str):
        try:
            payload = json.loads(request)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _request_uses_block2(request: Any) -> bool:
    payload = _request_payload(request)
    values: List[Any] = [payload.get('solver'), payload.get('method')]
    model = payload.get('model_hamiltonian')
    if isinstance(model, dict):
        values.append(model.get('solver'))
        spec = model.get('spec')
        if isinstance(spec, dict):
            values.append(spec.get('solver'))
    for value in values:
        if isinstance(value, dict):
            value = value.get('name') or value.get('id') or value.get('solver')
        normalized = str(value or '').strip().lower().replace('-', '_')
        if normalized in ('block2', 'dmrg', 'block2_dmrg'):
            return True
    return False


def _manifest_tasks(path: Path) -> Dict[str, BatchTask]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}
    tasks = payload.get('tasks') if isinstance(payload, dict) else None
    if not isinstance(tasks, list):
        return {}
    result: Dict[str, BatchTask] = {}
    for item in tasks:
        try:
            task = BatchTask.from_dict(item)
        except (TypeError, ValueError):
            continue
        result[task.task_id] = task
    return result


class SlurmExecutor:
    """Slurm job-array adapter for independent tasks and single-task refinement."""

    executor_id = 'slurm'
    execution_mode = 'scheduler'
    submission_mode = 'sbatch'
    supports_independent_batch = True
    supports_recoverable_batch = True

    def __init__(
        self,
        *,
        command_runner: Callable[[Sequence[str]], Any] = _default_command_runner,
        sleeper: Callable[[float], None] = time.sleep,
        poll_interval: float = 5.0,
        wait_timeout: Optional[float] = None,
        max_array_size: int = 1000,
        max_parallel: Optional[int] = None,
        sbatch_args: Sequence[str] = (),
        profile_sbatch_args: Optional[Mapping[str, Sequence[str]]] = None,
        profile_labels: Optional[Mapping[str, str]] = None,
        profile_selector: Callable[[BatchTask], str] = _default_profile,
        script_preamble: Sequence[str] = (),
        python_executable: Optional[str] = None,
        task_command_prefix: Sequence[str] = (),
        job_name_prefix: str = 'pyscf-agent',
        cluster_id: Optional[str] = None,
        configuration_source: Optional[str] = None,
    ):
        if not callable(command_runner):
            raise TypeError('command_runner must be callable')
        if not callable(sleeper):
            raise TypeError('sleeper must be callable')
        if not callable(profile_selector):
            raise TypeError('profile_selector must be callable')
        if int(max_array_size) < 1:
            raise ValueError('max_array_size must be at least 1')
        if max_parallel is not None and int(max_parallel) < 1:
            raise ValueError('max_parallel must be at least 1 when provided')
        self._command_runner = command_runner
        self._sleeper = sleeper
        self._poll_interval = max(0.0, float(poll_interval))
        self._wait_timeout = (
            None if wait_timeout is None else max(0.0, float(wait_timeout))
        )
        self._max_array_size = int(max_array_size)
        self._max_parallel = int(max_parallel) if max_parallel is not None else None
        self._sbatch_args = tuple(str(item) for item in sbatch_args)
        self._profile_sbatch_args = {
            str(key): tuple(str(item) for item in value)
            for key, value in (profile_sbatch_args or {}).items()
        }
        self._profile_labels = {
            profile_id: str((profile_labels or {}).get(profile_id) or '').strip()
            or _profile_display_name(profile_id)
            for profile_id in self._profile_sbatch_args
        }
        self._profile_selector = profile_selector
        self._script_preamble = tuple(str(item) for item in script_preamble)
        self._python_executable = str(python_executable or sys.executable)
        self._task_command_prefix = tuple(
            str(item) for item in task_command_prefix if str(item).strip()
        )
        self._job_name_prefix = str(job_name_prefix or 'pyscf-agent').strip()
        self._cluster_id = str(cluster_id or '').strip() or 'slurm'
        self._configuration_source = str(configuration_source or '').strip() or None

    @classmethod
    def from_config(
        cls,
        path: Optional[str] = None,
        *,
        profile_id: Optional[str] = None,
        command_runner: Optional[Callable[[Sequence[str]], Any]] = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> 'SlurmExecutor':
        config = load_slurm_executor_config(path, profile_id=profile_id)
        kwargs = config.executor_kwargs()
        kwargs.update({
            'command_runner': command_runner or _default_command_runner,
            'sleeper': sleeper,
            'cluster_id': config.cluster_id,
            'configuration_source': config.source_path,
        })
        return cls(**kwargs)

    @staticmethod
    def _coerce_handle(handle: Any) -> JobHandle:
        if isinstance(handle, JobHandle):
            normalized = handle
        elif isinstance(handle, dict):
            normalized = JobHandle.from_dict(handle)
        else:
            raise TypeError('handle must be a JobHandle or dictionary')
        if normalized.executor_id != SlurmExecutor.executor_id:
            raise JobNotFoundError(
                'Job {0} belongs to executor {1}, not slurm'.format(
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

    def _run_command(self, args: Sequence[str], *, operation: str) -> Any:
        try:
            result = self._command_runner(tuple(str(item) for item in args))
        except subprocess.TimeoutExpired as exc:
            raise command_timeout(operation, exc) from exc
        return_code = int(getattr(result, 'returncode', 0))
        if return_code != 0:
            stderr = str(getattr(result, 'stderr', '') or '').strip()
            raise ExecutorContractError(
                '{0} failed with exit code {1}: {2}'.format(
                    operation,
                    return_code,
                    stderr or 'no error output',
                )
            )
        return result

    @staticmethod
    def _batch_root(tasks: Sequence[BatchTask], batch_id: str) -> Path:
        first_work_dir = tasks[0].work_dir
        root = resolve_work_dir(first_work_dir)
        return root / 'slurm-batches' / batch_id

    def _write_batch_files(
        self,
        tasks: Sequence[BatchTask],
        *,
        batch_id: str,
        profile_id: str,
        submitted_at: str,
    ) -> Tuple[Path, Path]:
        batch_root = self._batch_root(tasks, batch_id)
        manifest_path = batch_root / 'batch-manifest.json'
        script_path = batch_root / 'run-array.sh'
        runtime_identity = load_runtime_identity()
        from ..artifacts.arrays import compact_request
        task_records = []
        for task in tasks:
            record = task.to_dict()
            record['request'] = compact_request(task.request, batch_root / 'arrays')
            task_records.append(record)
        manifest = {
            'schema': SLURM_BATCH_MANIFEST_SCHEMA,
            'batch_id': batch_id,
            'executor_id': self.executor_id,
            'profile_id': profile_id,
            'submitted_at': submitted_at,
            'work_dir': str(resolve_work_dir(tasks[0].work_dir)),
            'tasks': task_records,
            'runtime_identity': (
                runtime_identity.to_dict() if runtime_identity is not None else None
            ),
        }
        _write_json_atomic(
            manifest_path,
            manifest,
            kind='slurm-batch-manifest',
        )
        command_prefix = ' '.join(
            shlex.quote(item) for item in self._task_command_prefix
        )
        worker_command = '{0} -m pyscf_agent.executors.slurm_worker --manifest {1} --index "${{SLURM_ARRAY_TASK_ID}}"'.format(
            shlex.quote(self._python_executable),
            shlex.quote(str(manifest_path)),
        )
        if command_prefix:
            worker_command = '{0} {1}'.format(command_prefix, worker_command)
        script_lines = [
            '#!/usr/bin/env bash',
            'set -euo pipefail',
            *self._script_preamble,
            worker_command,
        ]
        script_path.write_text('\n'.join(script_lines) + '\n', encoding='utf-8')
        script_path.chmod(0o750)
        return manifest_path, script_path

    def submit_batch(
        self,
        tasks: Sequence[BatchTask],
        *,
        profile_id: Optional[str] = None,
    ) -> BatchHandle:
        normalized_tasks = list(tasks)
        if not normalized_tasks:
            raise ValueError('tasks must be a non-empty sequence')
        if any(not isinstance(task, BatchTask) for task in normalized_tasks):
            raise TypeError('tasks must contain BatchTask objects')
        task_ids = [task.task_id for task in normalized_tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError('BatchTask task_id values must be unique within a batch')
        normalized_profile = str(
            profile_id or self._profile_selector(normalized_tasks[0]) or 'default'
        ).strip()
        if not normalized_profile:
            normalized_profile = 'default'
        batch_id = 'batch-{0}'.format(uuid.uuid4().hex[:12])
        submitted_at = _utc_timestamp()
        manifest_path, script_path = self._write_batch_files(
            normalized_tasks,
            batch_id=batch_id,
            profile_id=normalized_profile,
            submitted_at=submitted_at,
        )
        runtime_identity = load_runtime_identity()
        array_value = '0-{0}'.format(len(normalized_tasks) - 1)
        if self._max_parallel is not None:
            array_value = '{0}%{1}'.format(array_value, self._max_parallel)
        batch_root = manifest_path.parent
        command = [
            'sbatch',
            '--parsable',
            '--array={0}'.format(array_value),
            '--job-name={0}-{1}'.format(
                self._job_name_prefix,
                normalized_profile,
            ),
            '--output={0}'.format(batch_root / 'slurm-%A_%a.out'),
            '--error={0}'.format(batch_root / 'slurm-%A_%a.err'),
            *self._sbatch_args,
            *self._profile_sbatch_args.get(normalized_profile, ()),
            str(script_path),
        ]
        result = self._run_command(command, operation='sbatch')
        stdout = str(getattr(result, 'stdout', '') or '').strip()
        backend_job_id = stdout.splitlines()[0].split(';', 1)[0].strip() if stdout else ''
        if not backend_job_id:
            raise ExecutorContractError('sbatch did not return a job id')
        jobs = [
            JobHandle(
                job_id='{0}:{1}'.format(batch_id, task.task_id),
                executor_id=self.executor_id,
                run_id=str(task.run_id or task.task_id),
                work_dir=str(resolve_work_dir(task.work_dir)),
                submitted_at=submitted_at,
                backend_job_id='{0}_{1}'.format(backend_job_id, index),
                runtime_release_id=(
                    runtime_identity.release_id if runtime_identity is not None else None
                ),
            )
            for index, task in enumerate(normalized_tasks)
        ]
        handle = BatchHandle(
            batch_id=batch_id,
            executor_id=self.executor_id,
            backend_job_id=backend_job_id,
            submitted_at=submitted_at,
            profile_id=normalized_profile,
            manifest_path=str(manifest_path),
            jobs=jobs,
        )
        _write_json_atomic(
            batch_root / SLURM_BATCH_HANDLE_FILENAME,
            handle.to_dict(),
            kind='slurm-batch-handle',
        )
        return handle

    def _scheduler_state(self, backend_job_id: str) -> Tuple[Optional[JobState], str]:
        queue = self._run_command(
            (
                'squeue',
                '--noheader',
                '--jobs',
                backend_job_id,
                '--format=%T',
            ),
            operation='squeue',
        )
        queue_output = str(getattr(queue, 'stdout', '') or '').strip()
        if queue_output:
            raw_state = queue_output.splitlines()[0].strip()
            return _job_state(raw_state), raw_state
        accounting = self._run_command(
            (
                'sacct',
                '--noheader',
                '--parsable2',
                '--jobs',
                backend_job_id,
                '--format=State',
            ),
            operation='sacct',
        )
        accounting_output = str(getattr(accounting, 'stdout', '') or '').strip()
        if accounting_output:
            raw_state = accounting_output.splitlines()[0].split('|', 1)[0].strip()
            return _job_state(raw_state), raw_state
        return None, ''

    def status(self, handle: Any) -> JobStatus:
        normalized = self._coerce_handle(handle)
        path = self._status_path(normalized)
        persisted_error: Optional[Exception] = None
        if path.is_file():
            try:
                persisted = JobStatus.from_dict(
                    json.loads(path.read_text(encoding='utf-8'))
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                persisted_error = exc
            else:
                if persisted.handle != normalized:
                    raise ExecutorContractError(
                        'Persisted job state does not match the supplied handle'
                    )
                if persisted.terminal:
                    return persisted
        if not normalized.backend_job_id:
            if persisted_error is not None:
                raise ExecutorContractError(
                    'Job state is unreadable for {0}'.format(normalized.job_id)
                ) from persisted_error
            raise JobNotFoundError(
                'Slurm JobHandle is missing backend_job_id: {0}'.format(
                    normalized.job_id
                )
            )
        state, raw_state = self._scheduler_state(normalized.backend_job_id)
        if state is None:
            if persisted_error is not None:
                raise ExecutorContractError(
                    'Job state is unreadable for {0}'.format(normalized.job_id)
                ) from persisted_error
            raise JobNotFoundError(
                'Slurm job state is unavailable: {0}'.format(
                    normalized.backend_job_id
                )
            )
        now = _utc_timestamp()
        return JobStatus(
            handle=normalized,
            state=state,
            updated_at=now,
            completed_at=now if state in (
                JobState.COMPLETED,
                JobState.FAILED,
                JobState.CANCELLED,
            ) else None,
            report_available=self._report_path(normalized).is_file(),
            message='{0}{1}'.format(
                'Slurm state: {0}'.format(raw_state or state.value),
                (
                    '; persisted job state was temporarily unreadable.'
                    if persisted_error is not None
                    else ''
                ),
            ),
        )

    def cancel(self, handle: Any) -> JobStatus:
        normalized = self._coerce_handle(handle)
        current = self.status(normalized)
        if current.terminal:
            return current
        if not normalized.backend_job_id:
            raise JobCancellationUnsupportedError(
                'Slurm JobHandle is missing backend_job_id'
            )
        self._run_command(
            ('scancel', normalized.backend_job_id),
            operation='scancel',
        )
        now = _utc_timestamp()
        cancelled = JobStatus(
            handle=normalized,
            state=JobState.CANCELLED,
            updated_at=now,
            completed_at=now,
            message='Cancellation requested through scancel.',
        )
        _write_json_atomic(
            self._status_path(normalized),
            cancelled.to_dict(),
            kind='job-state',
        )
        return cancelled

    def fetch(self, handle: Any) -> Dict[str, Any]:
        normalized = self._coerce_handle(handle)
        status = self.status(normalized)
        path = self._report_path(normalized)
        if not status.report_available or not path.is_file():
            raise JobNotReadyError(
                'Job {0} is {1}: TaskReport is not available'.format(
                    normalized.job_id,
                    status.state.value,
                )
            )
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExecutorContractError(
                'TaskReport is unreadable for {0}'.format(normalized.job_id)
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
        normalized = self._coerce_handle(handle)
        report = self.fetch(normalized)
        scientific = report.get('artifacts')
        scientific_artifacts = (
            copy.deepcopy(scientific)
            if isinstance(scientific, list)
            else []
        )
        return [
            _file_artifact_ref(
                self._status_path(normalized),
                kind='job-state',
                description='Persisted Slurm job state',
            ),
            _file_artifact_ref(
                self._report_path(normalized),
                kind='task-report',
                description='Persisted scientific TaskReport',
            ),
            *scientific_artifacts,
        ]

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
        normalized_run_id = str(run_id or '').strip() or make_run_id()
        task = BatchTask(
            task_id=normalized_run_id,
            request=request,
            channel=channel,
            locale=locale,
            work_dir=str(resolve_work_dir(work_dir)),
            run_id=normalized_run_id,
        )
        return self.submit_batch(
            [task],
            profile_id=self._requested_or_automatic_profile(task, resource_profile),
        ).jobs[0]

    def _requested_or_automatic_profile(
        self,
        task: BatchTask,
        resource_profile: Optional[str],
    ) -> str:
        requested = str(resource_profile or '').strip()
        if requested.lower() == 'auto':
            requested = ''
        if requested:
            if requested not in self._profile_sbatch_args:
                raise ValueError(
                    'Unknown Slurm resource profile {0}. Available profiles: {1}'.format(
                        requested,
                        ', '.join(sorted(self._profile_sbatch_args)) or 'none',
                    )
                )
            return requested
        automatic = str(self._profile_selector(task) or 'default').strip() or 'default'
        if automatic in self._profile_sbatch_args or not self._profile_sbatch_args:
            return automatic
        # Scientific method labels are useful grouping hints, but they are not
        # necessarily configured Slurm profiles.  Resolve auto selection to a
        # real, advertised profile so the manifest and resource gate describe
        # the resources that sbatch actually receives.
        if 'standard' in self._profile_sbatch_args:
            return 'standard'
        if 'default' in self._profile_sbatch_args:
            return 'default'
        raise ValueError(
            'Automatic Slurm profile {0} is not configured. Available profiles: {1}'.format(
                automatic,
                ', '.join(sorted(self._profile_sbatch_args)) or 'none',
            )
        )

    def wait_for_jobs(self, jobs: Sequence[JobHandle]) -> None:
        """Wait until every submitted job reaches a scheduler terminal state."""

        pending = {job.job_id: job for job in jobs}
        completed_without_report: Dict[str, float] = {}
        started = time.monotonic()
        while pending:
            for job_id, handle in list(pending.items()):
                status = self.status(handle)
                if _collectable_status(status):
                    pending.pop(job_id)
                    completed_without_report.pop(job_id, None)
                elif status.state == JobState.COMPLETED:
                    now = time.monotonic()
                    first_seen = completed_without_report.setdefault(job_id, now)
                    if now - first_seen >= REPORT_PROPAGATION_TIMEOUT_SECONDS:
                        raise JobNotReadyError(
                            'Slurm job completed but its TaskReport did not become visible: {0}'.format(
                                job_id
                            )
                        )
            if not pending:
                return
            if (
                self._wait_timeout is not None
                and time.monotonic() - started >= self._wait_timeout
            ):
                raise TimeoutError(
                    'Timed out waiting for Slurm jobs: {0}'.format(
                        ', '.join(sorted(pending))
                    )
                )
            self._sleeper(self._poll_interval)

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
        handle = self.submit_task(
            request,
            channel=channel,
            locale=locale,
            work_dir=work_dir,
            run_id=run_id,
            resource_profile=resource_profile,
        )
        self.wait_for_jobs([handle])
        return self.fetch(handle)

    def submit_independent_tasks(
        self,
        tasks: Sequence[BatchTask],
        *,
        resource_profile: Optional[str] = None,
    ) -> List[BatchHandle]:
        """Group independent tasks by resource profile and submit job arrays."""

        normalized_tasks = list(tasks)
        if not normalized_tasks:
            return []
        groups: 'OrderedDict[str, List[BatchTask]]' = OrderedDict()
        for task in normalized_tasks:
            if not isinstance(task, BatchTask):
                raise TypeError('tasks must contain BatchTask objects')
            profile_id = self._requested_or_automatic_profile(task, resource_profile)
            groups.setdefault(profile_id or 'default', []).append(task)

        submitted: List[BatchHandle] = []
        for profile_id, profile_tasks in groups.items():
            for offset in range(0, len(profile_tasks), self._max_array_size):
                submitted.append(self.submit_batch(
                    profile_tasks[offset:offset + self._max_array_size],
                    profile_id=profile_id,
                ))
        return submitted

    def fetch_independent_tasks(
        self,
        batches: Sequence[BatchHandle],
    ) -> BatchExecutionResult:
        """Collect reports and scheduler evidence for submitted job arrays."""

        submitted = list(batches)
        if any(not isinstance(batch, BatchHandle) for batch in submitted):
            raise TypeError('batches must contain BatchHandle objects')
        reports: Dict[str, Dict[str, Any]] = {}
        artifacts: List[Dict[str, Any]] = []
        for batch in submitted:
            batch_root = Path(batch.manifest_path).parent
            tasks_by_id = _manifest_tasks(Path(batch.manifest_path))
            artifacts.extend([
                _file_artifact_ref(
                    Path(batch.manifest_path),
                    kind='slurm-batch-manifest',
                    description='Slurm array index to independent task mapping',
                ),
                _file_artifact_ref(
                    batch_root / SLURM_BATCH_HANDLE_FILENAME,
                    kind='slurm-batch-handle',
                    description='Submitted Slurm array identity and per-task job handles',
                ),
            ])
            for handle in batch.jobs:
                task_id = handle.job_id.split(':', 1)[1]
                try:
                    reports[task_id] = self.fetch(handle)
                    continue
                except JobNotReadyError:
                    status = self.status(handle)
                if status.state not in (JobState.FAILED, JobState.CANCELLED):
                    raise JobNotReadyError(
                        'Job {0} is {1}: TaskReport is not yet available; collect again when visible'.format(
                            handle.job_id,
                            status.state.value,
                        )
                    )
                task = tasks_by_id.get(task_id)
                report = self._scheduler_failure_report(
                    handle,
                    status,
                    task=task,
                    batch=batch,
                )
                _write_json_atomic(
                    self._report_path(handle),
                    report,
                    kind='task-report',
                )
                _write_json_atomic(
                    self._status_path(handle),
                    JobStatus(
                        handle=handle,
                        state=status.state,
                        updated_at=status.updated_at,
                        started_at=status.started_at,
                        completed_at=status.completed_at or status.updated_at,
                        task_status='failed',
                        report_available=True,
                        message=status.message,
                    ).to_dict(),
                    kind='job-state',
                )
                reports[task_id] = report
        return BatchExecutionResult(
            reports=reports,
            batches=[batch.to_dict() for batch in submitted],
            artifacts=artifacts,
        )

    def _scheduler_failure_report(
        self,
        handle: JobHandle,
        status: JobStatus,
        *,
        task: Optional[BatchTask],
        batch: BatchHandle,
    ) -> Dict[str, Any]:
        request = task.request if task is not None else {}
        scheduler_state = status.message.partition('Slurm state: ')[2].split(';', 1)[0]
        scheduler_state = scheduler_state.strip() or status.state.value
        recovery = None
        if _request_uses_block2(request):
            from ..providers.block2 import block2_scheduler_recovery

            recovery = block2_scheduler_recovery(scheduler_state, status.message)
        code = (
            'resource_memory_exhausted'
            if isinstance(recovery, dict)
            and recovery.get('failure_class') == 'memory_exhausted'
            else 'scheduler_job_cancelled'
            if status.state == JobState.CANCELLED
            else 'scheduler_job_failed'
        )
        message = status.message or 'The scheduler ended the task without a TaskReport.'
        batch_root = Path(batch.manifest_path).parent
        scheduler_id = str(handle.backend_job_id or '').strip()
        stdout_text = _read_scheduler_output(batch_root / 'slurm-{0}.out'.format(scheduler_id))
        stderr_text = _read_scheduler_output(batch_root / 'slurm-{0}.err'.format(scheduler_id))
        scheduler_output = '\n'.join(
            section
            for section in (
                'Slurm stdout:\n{0}'.format(stdout_text) if stdout_text else '',
                'Slurm stderr:\n{0}'.format(stderr_text) if stderr_text else '',
            )
            if section
        )
        raw_stderr = '\n'.join(item for item in (message, scheduler_output) if item)
        details = {
            'scheduler_state': scheduler_state,
            'backend_job_id': handle.backend_job_id,
            'batch_id': batch.batch_id,
            'resource_profile': batch.profile_id,
            'scheduler_stdout_available': bool(stdout_text),
            'scheduler_stderr_available': bool(stderr_text),
        }
        if recovery is not None:
            details['recovery_recommendation'] = copy.deepcopy(recovery)
        error = {
            'stage': 'execution',
            'code': code,
            'message': message,
            'details': details,
        }
        summary = message
        if isinstance(recovery, dict) and recovery.get('summary'):
            summary = '{0} {1}'.format(summary.rstrip('.'), recovery['summary'])
        report = TaskReport(
            run_id=handle.run_id,
            work_dir=handle.work_dir,
            channel=task.channel if task is not None else 'study',
            task_spec=_request_payload(request),
            execution_status='failed',
            analysis_summary=summary,
            attempts=[{
                'index': 1,
                'timestamp': status.completed_at or status.updated_at,
                'stage': 'execution',
                'status': 'failed',
                'retry_count': 0,
                'task_spec': _request_payload(request),
                'raw_stderr': raw_stderr,
                'errors': [copy.deepcopy(error)],
                'structured_results': {},
                'compact_results': {},
                'artifacts': [],
            }],
            errors=[error],
            raw_stderr=raw_stderr,
            messages=[{
                'role': 'assistant',
                'kind': 'summary',
                'content': summary,
                'channel': task.channel if task is not None else 'study',
                'metadata': {},
            }],
            logs=[{
                'level': 'error',
                'event': 'scheduler.task_failed_without_report',
                'details': copy.deepcopy(details),
                'timestamp': status.completed_at or status.updated_at,
            }],
        ).to_dict()
        report['execution'] = {
            'executor': 'slurm',
            'resource_profile': batch.profile_id,
            'batch_id': batch.batch_id,
            'backend_job_id': handle.backend_job_id,
            'scheduler_state': scheduler_state,
            'synthetic_task_report': True,
        }
        return report

    def execute_independent_tasks(
        self,
        tasks: Sequence[BatchTask],
        *,
        resource_profile: Optional[str] = None,
    ) -> BatchExecutionResult:
        submitted = self.submit_independent_tasks(
            tasks,
            resource_profile=resource_profile,
        )
        if not submitted:
            return BatchExecutionResult(reports={}, batches=[])
        self.wait_for_independent_tasks(submitted)
        return self.collect_independent_tasks(submitted)

    def wait_for_independent_tasks(
        self,
        batches: Sequence[BatchHandle],
    ) -> None:
        normalized = list(batches)
        if any(not isinstance(batch, BatchHandle) for batch in normalized):
            raise TypeError('batches must contain BatchHandle objects')
        self.wait_for_jobs([
            job
            for batch in normalized
            for job in batch.jobs
        ])

    def collect_independent_tasks(
        self,
        batches: Sequence[BatchHandle],
    ) -> BatchExecutionResult:
        return self.fetch_independent_tasks(batches)

    def describe(self) -> Dict[str, Any]:
        default_memory_mb = _memory_mb_from_sbatch_args(self._sbatch_args)
        default_limits = (
            {'memory_mb': default_memory_mb}
            if default_memory_mb is not None
            else {}
        )
        return {
            'executor_id': self.executor_id,
            'execution_mode': self.execution_mode,
            'submission_mode': self.submission_mode,
            'location': 'slurm_cluster',
            'cluster_id': self._cluster_id,
            'supports_queue': True,
            'supports_cancel': True,
            'supports_job_handle': True,
            'supports_independent_batch': True,
            'supports_recoverable_batch': True,
            'supports_resource_profiles': bool(self._profile_sbatch_args),
            'batch_transport': 'job_array',
            'max_array_size': self._max_array_size,
            'max_parallel': self._max_parallel,
            'resource_profiles': sorted(self._profile_sbatch_args),
            'resource_profile_options': [
                {
                    'id': profile_id,
                    'label': self._profile_labels.get(profile_id, profile_id),
                    **({
                        'memory_mb': profile_memory_mb,
                    } if profile_memory_mb is not None else {}),
                }
                for profile_id in sorted(self._profile_sbatch_args)
                for profile_memory_mb in [
                    _memory_mb_from_sbatch_args(
                        self._sbatch_args + self._profile_sbatch_args[profile_id]
                    )
                ]
            ],
            'default_resource_limits': default_limits,
            'task_command_prefix': list(self._task_command_prefix),
            'job_name_prefix': self._job_name_prefix,
            'configuration_source': self._configuration_source,
            'job_state_filename': JOB_STATE_FILENAME,
            'job_report_filename': JOB_REPORT_FILENAME,
        }


__all__ = [
    'SLURM_BATCH_HANDLE_FILENAME',
    'SlurmExecutor',
]
