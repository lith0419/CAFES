from __future__ import annotations

from pyscf_agent.serialization import json_default

import argparse
import base64
import binascii
import json
import hashlib
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ..executors import BatchHandle, BatchTask, JobHandle, SlurmExecutor
from ..executors.slurm_config import load_slurm_executor_config
from ..executors.local import _write_json_atomic
from ..schema_contracts import public_schema_manifest
from ..runtime_identity import load_runtime_identity
from .protocol import error_response, success_response


RPC_OPERATIONS = (
    'capabilities',
    'submit-task',
    'status',
    'inspect-task',
    'fetch',
    'logs',
    'artifacts',
    'cancel',
    'submit-independent',
    'recover-submission',
    'batch-status',
    'batch-fetch',
    'generate-hamiltonian-dataset',
)
MAX_REMOTE_INPUT_BYTES = 5 * 1024 * 1024
MAX_REMOTE_ARRAY_BYTES = 64 * 1024 * 1024


def _safe_run_id(value: Any, *, fallback: str) -> str:
    normalized = re.sub(r'[^A-Za-z0-9_-]+', '-', str(value or '')).strip('-_')
    return (normalized or fallback)[:120]


def _submission_root(work_root: str) -> Path:
    root = Path(work_root).expanduser().resolve()
    submission_id = 'remote-{0}'.format(uuid.uuid4().hex[:16])
    path = (root / 'remote-submissions' / submission_id).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:  # pragma: no cover - defensive path invariant
        raise ValueError('Remote submission path escapes the configured work root') from exc
    path.mkdir(parents=True, exist_ok=False)
    return path


def _dataset_output_root(work_root: str, payload: Dict[str, Any]) -> Path:
    root = Path(work_root).expanduser().resolve()
    manifest = payload.get('source_manifest')
    spec = manifest.get('spec') if isinstance(manifest, dict) else None
    dataset_id = spec.get('dataset_id') if isinstance(spec, dict) else None
    study_id = _safe_run_id(payload.get('study_id'), fallback='study')
    dataset_name = _safe_run_id(dataset_id, fallback='dataset')
    path = (
        root
        / 'datasets'
        / study_id
        / '{0}-{1}'.format(dataset_name, uuid.uuid4().hex[:12])
    ).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:  # pragma: no cover - defensive path invariant
        raise ValueError('Remote dataset path escapes the configured work root') from exc
    return path


def _job_handle(payload: Dict[str, Any]) -> JobHandle:
    return JobHandle.from_dict(payload.get('handle') or {})


def _replace_attached_paths(value: Any, replacements: Dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_attached_paths(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_attached_paths(item, replacements) for item in value]
    if isinstance(value, str):
        if value in replacements:
            return replacements[value]
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            updated = value
            for source, target in replacements.items():
                updated = updated.replace(source, target)
            return updated
        updated = _replace_attached_paths(decoded, replacements)
        return json.dumps(updated, ensure_ascii=False)
    return value


def _materialize_attachments(
    request: Any,
    raw_attachments: Any,
    target_root: Path,
) -> Any:
    if raw_attachments in (None, []):
        return request
    if not isinstance(raw_attachments, list):
        raise TypeError('attachments must be a list')
    if len(raw_attachments) > 8:
        raise ValueError('At most 8 remote input attachments are allowed per task')
    target_root.mkdir(parents=True, exist_ok=True)
    replacements: Dict[str, str] = {}
    total_size = 0
    from ..artifacts.arrays import array_references
    try:
        decoded_request = json.loads(request) if isinstance(request, str) else request
    except ValueError:
        decoded_request = None
    array_paths = {ref['path'] for ref in array_references(decoded_request)}
    array_size = 0
    for index, item in enumerate(raw_attachments):
        if not isinstance(item, dict):
            raise TypeError('attachment entries must be JSON objects')
        source_path = str(item.get('source_path') or '').strip()
        if not source_path:
            raise ValueError('attachment source_path is required')
        filename = Path(str(item.get('filename') or 'input.dat')).name
        if filename in ('', '.', '..'):
            filename = 'input.dat'
        encoded = str(item.get('content_base64') or '')
        try:
            content = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError('attachment content_base64 is invalid') from exc
        declared_size = int(item.get('size_bytes', -1))
        if declared_size != len(content):
            raise ValueError('attachment size_bytes does not match decoded content')
        if source_path in array_paths:
            array_size += len(content)
            if array_size > MAX_REMOTE_ARRAY_BYTES:
                raise ValueError('Remote numerical attachments exceed the array transfer limit')
        else:
            total_size += len(content)
        if total_size > MAX_REMOTE_INPUT_BYTES:
            raise ValueError(
                'Remote task attachments exceed {0} bytes'.format(
                    MAX_REMOTE_INPUT_BYTES
                )
            )
        target = target_root / '{0:02d}-{1}'.format(index + 1, filename)
        target.write_bytes(content)
        target.chmod(0o600)
        replacements[source_path] = str(target)
    return _replace_attached_paths(request, replacements)


def dispatch_rpc(
    operation: str,
    payload: Dict[str, Any],
    *,
    executor: SlurmExecutor,
    work_root: str,
) -> Any:
    """Execute one trusted RPC operation against a direct Slurm executor."""

    if operation == 'capabilities':
        capabilities = executor.describe()
        capabilities['public_contract'] = public_schema_manifest()
        runtime_identity = load_runtime_identity()
        capabilities['runtime_identity'] = (
            runtime_identity.to_dict() if runtime_identity is not None else None
        )
        return capabilities
    if operation == 'submit-task':
        submission_root = _submission_root(work_root)
        run_id = _safe_run_id(
            payload.get('run_id'),
            fallback='task-{0}'.format(uuid.uuid4().hex[:12]),
        )
        submit_kwargs = {
            'channel': str(payload.get('channel') or 'remote'),
            'locale': str(payload.get('locale') or 'en'),
            'work_dir': str(submission_root / 'tasks'),
            'run_id': run_id,
        }
        if payload.get('resource_profile'):
            submit_kwargs['resource_profile'] = payload.get('resource_profile')
        return executor.submit_task(
            _materialize_attachments(
                payload.get('request'),
                payload.get('attachments'),
                submission_root / 'inputs',
            ),
            **submit_kwargs,
        ).to_dict()
    if operation == 'status':
        return executor.status(_job_handle(payload)).to_dict()
    if operation == 'inspect-task':
        return executor.inspect_task(_job_handle(payload))
    if operation == 'fetch':
        return executor.fetch(_job_handle(payload))
    if operation == 'logs':
        return executor.logs(_job_handle(payload))
    if operation == 'artifacts':
        return executor.artifacts(_job_handle(payload))
    if operation == 'cancel':
        return executor.cancel(_job_handle(payload)).to_dict()
    if operation == 'submit-independent':
        raw_tasks = payload.get('tasks')
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise ValueError('tasks must be a non-empty list')
        submission_root = _submission_root(work_root)
        cases_root = submission_root / 'cases'
        attachment_map = payload.get('attachments')
        if attachment_map is None:
            attachment_map = {}
        if not isinstance(attachment_map, dict):
            raise TypeError('attachments must be an object keyed by task_id')
        tasks = []
        used_run_ids = set()
        for index, item in enumerate(raw_tasks):
            source = BatchTask.from_dict(item)
            run_id = _safe_run_id(
                source.run_id or source.task_id,
                fallback='case-{0:04d}'.format(index + 1),
            )
            if run_id in used_run_ids:
                run_id = '{0}-{1:04d}'.format(run_id, index + 1)
            used_run_ids.add(run_id)
            tasks.append(BatchTask(
                task_id=source.task_id,
                request=_materialize_attachments(
                    source.request,
                    attachment_map.get(source.task_id),
                    submission_root / 'inputs' / run_id,
                ),
                channel=source.channel,
                locale=source.locale,
                work_dir=str(cases_root),
                run_id=run_id,
            ))
        submit_kwargs = {}
        if payload.get('resource_profile'):
            submit_kwargs['resource_profile'] = payload.get('resource_profile')
        result = {
            'batches': [
                batch.to_dict()
                for batch in executor.submit_independent_tasks(tasks, **submit_kwargs)
            ],
        }
        # Persist server-owned evidence before returning the SSH acknowledgement.
        # Hash the original request, before attachment paths are rebased.
        evidence = {**result, 'task_fingerprints': {
            task.task_id: hashlib.sha256(json.dumps({
                'request': task.request, 'channel': task.channel,
                'locale': task.locale, 'run_id': task.run_id,
            }, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=json_default, allow_nan=False).encode('utf-8')).hexdigest()
            for task in (BatchTask.from_dict(item) for item in raw_tasks)
        }}
        _write_json_atomic(submission_root / 'submission.json', evidence, kind='execution-receipt')
        return result
    if operation == 'recover-submission':
        root = (Path(work_root).expanduser().resolve() / 'remote-submissions').resolve()
        path = Path(str(payload.get('submission_path') or '')).expanduser().resolve()
        path.relative_to(root)
        if path.name != 'submission.json':
            raise ValueError('Expected the server submission.json evidence file')
        evidence = json.loads(path.read_text(encoding='utf-8'))
        if not evidence.get('batches') or not evidence.get('task_fingerprints'):
            raise ValueError('Submission evidence is incomplete; do not resubmit automatically')
        for raw in evidence['batches']:
            batch = BatchHandle.from_dict(raw)
            Path(batch.manifest_path).resolve().relative_to(path.parent)
            for handle in batch.jobs:
                Path(handle.work_dir).resolve().relative_to(path.parent)
        return evidence
    if operation == 'batch-status':
        raw_jobs = payload.get('jobs')
        if not isinstance(raw_jobs, list):
            raise ValueError('jobs must be a list')
        return {
            'statuses': [
                executor.status(JobHandle.from_dict(item)).to_dict()
                for item in raw_jobs
            ],
        }
    if operation == 'batch-fetch':
        raw_batches = payload.get('batches')
        if not isinstance(raw_batches, list):
            raise ValueError('batches must be a list')
        batches = [BatchHandle.from_dict(item) for item in raw_batches]
        return executor.fetch_independent_tasks(batches).to_dict()
    if operation == 'generate-hamiltonian-dataset':
        from computational_study_agent.datasets.hamiltonian.postprocessing import (  # pylint: disable=import-outside-toplevel
            materialize_hamiltonian_dataset,
        )

        return materialize_hamiltonian_dataset(
            payload,
            output_dir=_dataset_output_root(work_root, payload),
            allowed_source_root=work_root,
            location='remote_executor',
        )
    raise ValueError('Unsupported RPC operation: {0}'.format(operation))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Internal SSH JSON-RPC endpoint for pyscf-agent Slurm execution.'
    )
    parser.add_argument('operation', choices=RPC_OPERATIONS)
    parser.add_argument(
        '--slurm-config',
        default=None,
        help='Server-side direct Slurm configuration file.',
    )
    parser.add_argument(
        '--slurm-profile',
        default=None,
        help='Named [server:<profile>] section in server-slurm.ini.',
    )
    return parser


def _read_payload() -> Dict[str, Any]:
    if hasattr(sys.stdin, 'isatty') and sys.stdin.isatty():
        return {}
    text = sys.stdin.read()
    if not text.strip():
        return {}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError('RPC request payload must be a JSON object')
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_slurm_executor_config(
            args.slurm_config,
            profile_id=args.slurm_profile,
        )
        payload = _read_payload()
        data = dispatch_rpc(
            args.operation,
            payload,
            executor=SlurmExecutor.from_config(
                args.slurm_config,
                profile_id=args.slurm_profile,
            ),
            work_root=config.remote.work_root,
        )
        response = success_response(args.operation, data)
    except Exception as exc:
        response = error_response(args.operation, exc)
    sys.stdout.write(json.dumps(response, ensure_ascii=False, default=json_default, allow_nan=False))
    sys.stdout.write('\n')
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
