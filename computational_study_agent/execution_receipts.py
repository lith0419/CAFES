from __future__ import annotations



import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pyscf_agent.artifacts import default_artifact_repository
from pyscf_agent.executors import BatchHandle, JobHandle, JobState
from pyscf_agent.schema_contracts import (
    STUDY_EXECUTION_RECEIPT_SCHEMA,
    STUDY_EXECUTION_STATUS_SCHEMA,
)

from computational_study_agent.datasets.hamiltonian.postprocessing import (
    HAMILTONIAN_DATASET_GENERATION_FILENAME,
    HAMILTONIAN_DATASET_GENERATION_SCHEMA,
)


EXECUTION_RECEIPT_SCHEMA = STUDY_EXECUTION_RECEIPT_SCHEMA
EXECUTION_RECEIPT_FILENAME = 'execution-receipt.json'


class StudyExecutionBusy(ValueError):
    """A concurrent mutation must not queue another execution of the same study."""


class StudyExecutionInterrupted(RuntimeError):
    """Raised after submission when execution can be resumed without resubmission."""

    def __init__(self, message: str, receipt: Dict[str, Any]):
        self.receipt = copy.deepcopy(receipt)
        super().__init__(message)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def receipt_path(study_work_dir: Path) -> Path:
    return Path(study_work_dir).expanduser().resolve() / EXECUTION_RECEIPT_FILENAME


def write_receipt(path: Path, receipt: Dict[str, Any]) -> Dict[str, Any]:
    payload = copy.deepcopy(receipt)
    payload['schema'] = EXECUTION_RECEIPT_SCHEMA
    payload['updated_at'] = _timestamp()
    target = Path(path).expanduser().resolve()
    default_artifact_repository().write_json(
        target, payload, kind='execution-receipt', atomic=True,
    )
    return payload


def load_receipt(path: Path) -> Optional[Dict[str, Any]]:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        return None
    try:
        payload = json.loads(target.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError('Unreadable execution receipt: {0}'.format(target)) from exc
    if not isinstance(payload, dict) or payload.get('schema') != EXECUTION_RECEIPT_SCHEMA:
        raise ValueError('Invalid execution receipt schema: {0}'.format(target))
    if not isinstance(payload.get('batches'), list):
        raise ValueError('Invalid execution receipt batches: {0}'.format(target))
    return payload


def executor_identity(executor: Any) -> Dict[str, Any]:
    describe = getattr(executor, 'describe', None)
    description = describe() if callable(describe) else {}
    if not isinstance(description, dict):
        description = {}
    return {
        key: copy.deepcopy(description.get(key))
        for key in (
            'executor_id',
            'execution_mode',
            'location',
            'transport',
            'remote_profile',
        )
        if description.get(key) is not None
    }


def batches_from_receipt(receipt: Dict[str, Any]) -> List[BatchHandle]:
    return [
        BatchHandle.from_dict(item)
        for item in receipt.get('batches') or []
    ]


def jobs_from_receipt(receipt: Dict[str, Any]) -> Dict[str, JobHandle]:
    """Case-to-run associations for batch schedulers and local worker processes."""
    jobs = {key: JobHandle.from_dict(value) for key, value in (receipt.get('jobs') or {}).items()}
    for batch in batches_from_receipt(receipt):
        for handle in batch.jobs:
            case_id = handle.job_id.split(':', 1)[-1]
            if case_id in jobs:
                raise ValueError('Duplicate case in execution receipt: ' + case_id)
            jobs[case_id] = handle
    return jobs


def receipt_matches(
    receipt: Optional[Dict[str, Any]],
    *,
    study_id: str,
    study_fingerprint: str,
    task_fingerprints: Dict[str, str],
    executor: Any,
) -> bool:
    if not isinstance(receipt, dict):
        return False
    return (
        receipt.get('study_id') == study_id
        and receipt.get('study_fingerprint') == study_fingerprint
        and receipt.get('task_fingerprints') == task_fingerprints
        and receipt.get('executor') == executor_identity(executor)
        and bool(receipt.get('batches'))
    )


def new_receipt(
    *,
    study_id: str,
    study_fingerprint: str,
    task_fingerprints: Dict[str, str],
    executor: Any,
    batches: Sequence[BatchHandle],
    resource_profile: Optional[str] = None,
) -> Dict[str, Any]:
    now = _timestamp()
    return {
        'schema': EXECUTION_RECEIPT_SCHEMA,
        'study_id': study_id,
        'study_fingerprint': study_fingerprint,
        'task_fingerprints': copy.deepcopy(task_fingerprints),
        'task_ids': list(task_fingerprints),
        'executor': executor_identity(executor),
        'requested_resource_profile': resource_profile or 'auto',
        'status': 'submitted',
        'submitted_at': now,
        'updated_at': now,
        'batches': [batch.to_dict() for batch in batches],
        'last_error': None,
        'status_summary': None,
    }


def inspect_receipt(
    receipt: Dict[str, Any],
    executor: Any,
) -> Dict[str, Any]:
    if receipt.get('executor') != executor_identity(executor):
        raise ValueError('Execution receipt belongs to a different executor')
    # A durable collection is terminal. Historical jobs may already have been
    # purged by the scheduler; status polling must not reopen those receipts.
    if receipt.get('status') == 'collected':
        updated = copy.deepcopy(receipt)
        count = len(jobs_from_receipt(receipt))
        updated['status_summary'] = {
            'status': 'collected', 'expected': count, 'terminal': count,
            'report_available': count, 'state_counts': {'completed': count},
            'cases': [], 'unreadable': [],
        }
        return updated
    jobs = jobs_from_receipt(receipt)
    if not jobs or set(jobs) != set(receipt.get('task_ids') or jobs):
        updated = copy.deepcopy(receipt)
        updated['status'] = 'submission_unknown'
        return updated
    statuses = []
    unreadable = []
    handles = list(jobs.values())
    case_ids = {handle.job_id: key for key, handle in jobs.items()}
    status_many = getattr(executor, 'status_many', None)
    inspected_pairs = []
    if callable(status_many) and handles:
        try:
            inspected_statuses = list(status_many(handles))
            if len(inspected_statuses) != len(handles):
                raise ValueError('status_many returned an incomplete result')
            inspected_pairs = list(zip(handles, inspected_statuses))
        except Exception as exc:
            unreadable.extend({
                'case_id': case_ids[handle.job_id],
                'job_id': handle.job_id,
                'message': '{0}: {1}'.format(type(exc).__name__, exc),
            } for handle in handles)
    else:
        for handle in handles:
            try:
                status = executor.status(handle)
            except Exception as exc:
                unreadable.append({
                    'case_id': case_ids[handle.job_id],
                    'job_id': handle.job_id,
                    'message': '{0}: {1}'.format(type(exc).__name__, exc),
                })
                continue
            inspected_pairs.append((handle, status))
    for handle, status in inspected_pairs:
        statuses.append({
            'case_id': case_ids[handle.job_id],
            'job_id': handle.job_id,
            'backend_job_id': handle.backend_job_id,
            'state': status.state.value,
            'terminal': status.terminal,
            'task_status': status.task_status,
            'report_available': status.report_available,
            'message': status.message,
        })
    state_counts: Dict[str, int] = {}
    for item in statuses:
        state = str(item.get('state') or 'unknown')
        state_counts[state] = state_counts.get(state, 0) + 1
    expected = len(jobs)
    terminal = sum(1 for item in statuses if item.get('terminal'))
    report_available = sum(1 for item in statuses if item.get('report_available'))
    if unreadable:
        overall = 'connection_interrupted'
    elif expected and terminal == expected and all(
        item['report_available'] or item['state'] in (JobState.FAILED.value, JobState.CANCELLED.value)
        for item in statuses
    ):
        overall = 'results_available'
    elif any(item.get('state') == JobState.RUNNING.value for item in statuses):
        overall = 'running'
    else:
        overall = 'submitted'
    summary = {
        'status': overall,
        'expected': expected,
        'terminal': terminal,
        'report_available': report_available,
        'state_counts': state_counts,
        'cases': statuses,
        'unreadable': unreadable,
    }
    updated = copy.deepcopy(receipt)
    updated['status'] = overall
    updated['status_summary'] = summary
    return updated


def discover_receipts(root: Path) -> List[Path]:
    target = Path(root).expanduser().resolve()
    if not target.is_dir():
        return []
    direct = target / EXECUTION_RECEIPT_FILENAME
    paths = [direct] if direct.is_file() else []
    paths.extend(
        path
        for path in sorted(target.rglob(EXECUTION_RECEIPT_FILENAME))
        if path != direct
    )
    return paths


def _read_json_object(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _dataset_execution_progress(
    root: Path,
    receipt_summaries: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    plan = _read_json_object(root / 'study-plan.json')
    comparison = plan.get('comparison') if isinstance(plan.get('comparison'), dict) else {}
    if comparison.get('mode') != 'hamiltonian_dataset_assembly':
        return None
    spec = comparison.get('dataset_spec') if isinstance(comparison.get('dataset_spec'), dict) else {}
    try:
        expected_trajectories = int(
            spec.get('target_molecule_count') or comparison.get('expected_case_count') or len(plan.get('cases') or [])
        )
        structures_per_trajectory = int(spec.get('geometries_per_molecule') or 0)
        expected_structures = int(
            spec.get('target_structure_count')
            or comparison.get('expected_sample_count')
            or expected_trajectories * structures_per_trajectory
        )
    except (TypeError, ValueError):
        return None
    case_statuses: Dict[str, Dict[str, Any]] = {}
    for receipt in receipt_summaries:
        summary = receipt.get('summary') if isinstance(receipt.get('summary'), dict) else {}
        for item in summary.get('cases') or []:
            if isinstance(item, dict) and item.get('case_id'):
                case_statuses[str(item['case_id'])] = item
    terminal_trajectories = sum(bool(item.get('terminal')) for item in case_statuses.values())
    succeeded_trajectories = sum(
        bool(item.get('terminal')) and str(item.get('task_status') or '').lower() == 'succeeded'
        for item in case_statuses.values()
    )
    failed_trajectories = sum(
        bool(item.get('terminal')) and str(item.get('task_status') or '').lower() != 'succeeded'
        for item in case_statuses.values()
    )
    running_trajectories = sum(
        str(item.get('state') or '').lower() == JobState.RUNNING.value
        for item in case_statuses.values()
    )
    available_structures = min(
        expected_structures,
        succeeded_trajectories * structures_per_trajectory,
    )
    failed_structures = min(
        max(0, expected_structures - available_structures),
        failed_trajectories * structures_per_trajectory,
    )
    progress = {
        'expected_trajectories': expected_trajectories,
        'terminal_trajectories': terminal_trajectories,
        'succeeded_trajectories': succeeded_trajectories,
        'failed_trajectories': failed_trajectories,
        'running_trajectories': running_trajectories,
        'structures_per_trajectory': structures_per_trajectory,
        'expected_structures': expected_structures,
        'available_structures': available_structures,
        'failed_structures': failed_structures,
        'pending_structures': max(
            0,
            expected_structures - available_structures - failed_structures,
        ),
        'finalized': False,
        'generated': False,
        'collected': False,
    }
    report = _read_json_object(root / 'study-report.json')
    manifest = report.get('dataset_manifest') if isinstance(report.get('dataset_manifest'), dict) else {}
    if manifest:
        progress.update({
            'accepted_structures': int(manifest.get('accepted_structure_count') or 0),
            'rejected_structures': int(manifest.get('rejected_structure_count') or 0),
            'missing_structures': int(manifest.get('missing_structure_count') or 0),
            'finalized': True,
        })
    generation = _read_json_object(
        root / 'postprocessing' / HAMILTONIAN_DATASET_GENERATION_FILENAME
    )
    if (
        generation.get('schema') == HAMILTONIAN_DATASET_GENERATION_SCHEMA
        and generation.get('status') == 'succeeded'
    ):
        progress.update({
            'generated': True,
            'generation_location': generation.get('location'),
            'generated_dataset_root': generation.get('dataset_root'),
            'generated_trajectory_files': int(
                generation.get('trajectory_file_count') or 0
            ),
        })
    collected_manifest = _read_json_object(
        root / 'postprocessing' / 'hamiltonian-dataset' / 'dataset-manifest.json'
    )
    collection = collected_manifest.get('collection') if isinstance(collected_manifest.get('collection'), dict) else {}
    if collection.get('status') == 'complete':
        progress['generated'] = True
        progress['collected'] = True
        progress['collected_trajectory_files'] = int(collection.get('trajectory_file_count') or 0)
    return progress


def inspect_execution_tree(root: Path, executor: Any) -> Dict[str, Any]:
    target = Path(root).expanduser().resolve()
    receipt_summaries = []
    totals = {
        'expected': 0,
        'terminal': 0,
        'report_available': 0,
    }
    for path in discover_receipts(target):
        receipt = load_receipt(path)
        if receipt is None:
            continue
        # Show only the current execution for each case. Keep old receipts on
        # disk for provenance without counting previous attempts as new work.
        owner = path.parent
        while owner != target and not (owner / 'study-state.json').is_file():
            owner = owner.parent
        state = _read_json_object(owner / 'study-state.json')
        associations = {
            case_id: checkpoint['execution']
            for case_id, checkpoint in (state.get('cases') or {}).items()
            if isinstance(checkpoint, dict) and isinstance(checkpoint.get('execution'), dict)
        }
        if associations:
            active = {case_id: execution for case_id, execution in associations.items()
                      if execution.get('receipt_path') and Path(execution['receipt_path']).resolve() == path}
            if not active:
                continue
            receipt = copy.deepcopy(receipt)
            for batch in receipt['batches']:
                batch['jobs'] = [handle for handle in batch.get('jobs', [])
                                 if any(handle.get('run_id') == entry['run_id'] for entry in active.values())]
            receipt['batches'] = [batch for batch in receipt['batches'] if batch['jobs']]
            receipt['jobs'] = {key: handle for key, handle in (receipt.get('jobs') or {}).items()
                               if key in active and handle['run_id'] == active[key]['run_id']}
            receipt['task_ids'] = [key for key in receipt.get('task_ids', []) if key in active]
        inspected = inspect_receipt(receipt, executor)
        summary = inspected.get('status_summary') or {}
        for key in totals:
            totals[key] += int(summary.get(key) or 0)
        receipt_summaries.append({
            'path': str(path),
            'study_id': inspected.get('study_id'),
            'status': inspected.get('status'),
            'submitted_at': inspected.get('submitted_at'),
            'summary': summary,
            'batches': copy.deepcopy(inspected.get('batches') or []),
            'jobs': copy.deepcopy(inspected.get('jobs') or {}),
        })
    statuses = {item['status'] for item in receipt_summaries}
    if not receipt_summaries:
        overall = 'not_found'
    elif 'submission_unknown' in statuses:
        overall = 'submission_unknown'
    elif 'connection_interrupted' in statuses:
        overall = 'connection_interrupted'
    elif statuses <= {'results_available', 'collected'}:
        overall = 'results_available'
    elif 'running' in statuses:
        overall = 'running'
    elif 'submitted' in statuses:
        overall = 'submitted'
    else:
        overall = 'partially_available'
    result = {
        'schema': STUDY_EXECUTION_STATUS_SCHEMA,
        'status': overall,
        **totals,
        'receipt_count': len(receipt_summaries),
        'receipts': receipt_summaries,
        'can_collect': bool(receipt_summaries) and all(
            item['status'] in ('results_available', 'collected')
            for item in receipt_summaries
        ),
        'checked_at': _timestamp(),
    }
    dataset = _dataset_execution_progress(target, receipt_summaries)
    if dataset is not None:
        result['dataset'] = dataset
    return result


__all__ = [
    'EXECUTION_RECEIPT_FILENAME',
    'EXECUTION_RECEIPT_SCHEMA',
    'StudyExecutionInterrupted',
    'StudyExecutionBusy',
    'batches_from_receipt',
    'jobs_from_receipt',
    'discover_receipts',
    'executor_identity',
    'inspect_execution_tree',
    'inspect_receipt',
    'load_receipt',
    'new_receipt',
    'receipt_matches',
    'receipt_path',
    'write_receipt',
]
