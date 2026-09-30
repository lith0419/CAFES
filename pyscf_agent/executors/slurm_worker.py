from __future__ import annotations

from pyscf_agent.timestamps import utc_timestamp as _utc_timestamp

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .contracts import (
    SLURM_BATCH_MANIFEST_SCHEMA,
    BatchTask,
    JobHandle,
    JobState,
    JobStatus,
)
from .local import (
    JOB_REPORT_FILENAME,
    JOB_STATE_FILENAME,
    LocalExecutor,
    _write_json_atomic,
)
from ..runtime_identity import RuntimeIdentity


def _load_manifest(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError('Slurm batch manifest is unreadable: {0}'.format(path)) from exc
    if not isinstance(payload, dict):
        raise TypeError('Slurm batch manifest must be a dictionary')
    if payload.get('schema') != SLURM_BATCH_MANIFEST_SCHEMA:
        raise ValueError(
            'Unsupported Slurm batch manifest schema: {0}'.format(
                payload.get('schema')
            )
        )
    return payload


def _task_handle(
    manifest: Dict[str, Any],
    task: BatchTask,
    *,
    index: int,
) -> JobHandle:
    array_job_id = str(os.getenv('SLURM_ARRAY_JOB_ID') or '').strip()
    array_task_id = str(os.getenv('SLURM_ARRAY_TASK_ID') or index).strip()
    backend_job_id = (
        '{0}_{1}'.format(array_job_id, array_task_id)
        if array_job_id
        else None
    )
    raw_runtime_identity = manifest.get('runtime_identity')
    runtime_identity = (
        RuntimeIdentity.from_dict(raw_runtime_identity)
        if isinstance(raw_runtime_identity, dict)
        else None
    )
    return JobHandle(
        job_id='{0}:{1}'.format(manifest['batch_id'], task.task_id),
        executor_id=str(manifest.get('executor_id') or 'slurm'),
        run_id=str(task.run_id or task.task_id),
        work_dir=str(task.work_dir or manifest['work_dir']),
        submitted_at=str(manifest['submitted_at']),
        backend_job_id=backend_job_id,
        runtime_release_id=(
            runtime_identity.release_id if runtime_identity is not None else None
        ),
    )


def _job_dir(handle: JobHandle) -> Path:
    return Path(handle.work_dir).expanduser().resolve() / handle.run_id


def run_manifest_task(
    manifest_path: str,
    index: int,
    *,
    executor: Optional[LocalExecutor] = None,
) -> Dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    manifest = _load_manifest(path)
    tasks = manifest.get('tasks')
    if not isinstance(tasks, list) or index < 0 or index >= len(tasks):
        raise IndexError('Slurm array index is outside the batch manifest')
    task = BatchTask.from_dict(tasks[index])
    handle = _task_handle(manifest, task, index=index)
    job_dir = _job_dir(handle)
    status_path = job_dir / JOB_STATE_FILENAME
    report_path = job_dir / JOB_REPORT_FILENAME
    started_at = _utc_timestamp()
    _write_json_atomic(
        status_path,
        JobStatus(
            handle=handle,
            state=JobState.RUNNING,
            updated_at=started_at,
            started_at=started_at,
            message='Running as a Slurm array task.',
        ).to_dict(),
        kind='job-state',
    )

    task_executor = executor or LocalExecutor()
    try:
        report = task_executor.execute_task(
            task.request,
            channel=task.channel,
            locale=task.locale,
            work_dir=task.work_dir,
            run_id=task.run_id,
        )
    except Exception as exc:
        failed_at = _utc_timestamp()
        _write_json_atomic(
            status_path,
            JobStatus(
                handle=handle,
                state=JobState.FAILED,
                updated_at=failed_at,
                started_at=started_at,
                completed_at=failed_at,
                message='{0}: {1}'.format(type(exc).__name__, exc),
            ).to_dict(),
            kind='job-state',
        )
        raise

    execution = report.get('execution')
    if not isinstance(execution, dict):
        execution = {}
    execution.update({
        'executor': str(manifest.get('executor_id') or 'slurm'),
        'resource_profile': str(manifest.get('profile_id') or 'default'),
        'batch_id': str(manifest.get('batch_id') or ''),
        'runtime_identity': manifest.get('runtime_identity'),
    })
    report['execution'] = execution
    _write_json_atomic(report_path, report, kind='task-report')
    completed_at = _utc_timestamp()
    _write_json_atomic(
        status_path,
        JobStatus(
            handle=handle,
            state=JobState.COMPLETED,
            updated_at=completed_at,
            started_at=started_at,
            completed_at=completed_at,
            task_status=str(report.get('execution_status') or 'unknown'),
            report_available=True,
            message='TaskReport is available.',
        ).to_dict(),
        kind='job-state',
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Execute one PySCF Agent task from a Slurm batch manifest.'
    )
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--index', required=True, type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_manifest_task(args.manifest, args.index)
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
