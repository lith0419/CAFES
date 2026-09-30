from __future__ import annotations

from pyscf_agent.serialization import json_fingerprint


import copy
import json
import time
from functools import wraps
from pathlib import Path
from uuid import uuid4
from typing import Any, Dict, Iterable, List, Optional, Set

from pyscf_agent.artifacts import default_artifact_repository
from pyscf_agent.registry import default_registry as default_platform_capability_registry
from pyscf_agent.paths import resolve_work_dir
from pyscf_agent.executors import (
    BatchExecutionResult,
    BatchTask,
    BatchTaskExecutor,
    ExecutorContractError,
    TaskExecutor,
    get_local_executor,
)
from pyscf_agent.lifecycle import (
    begin_study_execution,
    complete_study_execution,
    synchronize_study_report_lifecycle,
    transition_lifecycle,
)
from pyscf_agent.workflow_gates import gate_artifact_documents

from .locking import study_lock
from .planner import build_study_plan
from .study_state import REPORT_LOCK, case_index, load_checkpoint, read_json, checkpoint_payload
from .reports import merge_case_records, update_study_report
from pyscf_agent.schema_contracts import ADAPTIVE_STUDY_REPORT_SCHEMA
from .costing import require_plan_cost_approval
from .execution_receipts import (
    StudyExecutionInterrupted,
    StudyExecutionBusy as StudyExecutionBusy,
    batches_from_receipt,
    discover_receipts,
    executor_identity,
    inspect_receipt,
    jobs_from_receipt,
    load_receipt,
    new_receipt,
    receipt_path,
    write_receipt,
)
from .gates.runtime import evaluate_study_gates
from .gates.presentation import build_report_workflow_from_gate_decisions
from .postprocessing import run_postprocessing
from .schema import StudyCase, StudyPlan, StudyReport
from computational_study_agent.datasets.hamiltonian.finalize import finalize_hamiltonian_dataset


MODEL_ENERGY_PARAMETER_COLUMNS = {
    'u',
    't',
    'v',
    'epsilon',
    'effective_t',
    'effective_v',
    'omega',
    'phononomega',
    'phonon_omega',
    'electronphonong',
    'electron_phonon_g',
    'g',
}
MODEL_ENERGY_PARAMETER_PREFIXES = (
    'u_',
    't_',
    'v_',
    'epsilon_',
    'omega_',
    'phonon_omega_',
    'phononomega_',
    'g_',
    'electron_phonon_g_',
    'electronphonong_',
)

MODEL_ENERGY_OBSERVABLE_COLUMNS = {
    item.id
    for item in default_platform_capability_registry().observables(
        namespace='model_hamiltonian.observable',
        planner_allowed=True,
    )
    if item.unit == 'energy'
}
MODEL_OBSERVABLE_SUMMARY_COLUMNS = {
    item.id: item.comparison_field
    for item in default_platform_capability_registry().observables(
        namespace='model_hamiltonian.observable',
        planner_allowed=True,
    )
    if item.comparison_field
}
MODEL_POSTPROCESSING_METRICS = tuple(
    default_platform_capability_registry().observables(
        namespace='postprocessing.metric',
        system_type='model_hamiltonian',
        backend_allowed=True,
    )
)

STUDY_STATE_SCHEMA = 'pyscf-agent.study-state.v1'
RETRYABLE_CASE_STATUSES = {'failed', 'unconverged'}
def _serialized_execution(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        plan = args[0]
        if not isinstance(plan, StudyPlan):
            plan = build_study_plan(plan)
            args = (plan, *args[1:])
        root = resolve_work_dir(kwargs.get('work_dir')).resolve()
        directory = (root / plan.study_id).resolve()
        directory.relative_to(root)
        if directory == root:
            raise ValueError('Study must have its own directory')
        with study_lock(directory, wait_seconds=30.0 if kwargs.get('collect_only') else 0.0):
            return function(*args, **kwargs)
    return guarded


def _case_fingerprint(case) -> str:
    """Fingerprint the computational contract, rather than its display label."""
    return json_fingerprint({
        'request': case.request,
        'variables': case.variables,
        'operations': case.operations,
        'model_spec': case.model_spec,
    })


def _study_fingerprint(plan: StudyPlan) -> str:
    return json_fingerprint({
        'system_type': plan.system_type,
        'observables': plan.observables,
        'comparison': plan.comparison,
        'resource_policy': plan.resource_policy,
        'cases': [
            {'case_id': case.case_id, 'fingerprint': _case_fingerprint(case)}
            for case in plan.cases
        ],
    })


def _sanitize_run_token(value: str) -> str:
    token = ''.join(character if character.isalnum() or character in ('-', '_') else '-' for character in value)
    token = token.strip('-_')
    return token or 'case'


def _case_run_id(case_id: str) -> str:
    return _sanitize_run_token(case_id)


def _write_json(
    path: Path,
    payload: Any,
    *,
    kind: Optional[str] = None,
    description: str = '',
) -> Dict[str, Any]:
    artifact_kind = kind or path.stem
    return default_artifact_repository().write_json(
        path,
        payload,
        kind=artifact_kind,
        description=description,
    )


def _write_json_atomic(
    path: Path,
    payload: Any,
    *,
    kind: str,
    description: str = '',
) -> Dict[str, Any]:
    """Persist checkpoints atomically so an interrupted study has a valid prior state."""
    if kind == 'study-state':
        payload = checkpoint_payload(path, payload)
    return default_artifact_repository().write_json(
        path,
        payload,
        kind=kind,
        description=description,
        atomic=True,
    )


def _load_study_state(path: Path, plan: StudyPlan, *, initial_cases=None) -> Dict[str, Any]:
    default = {
        'schema': STUDY_STATE_SCHEMA,
        'study_id': plan.study_id,
        'study_fingerprint': _study_fingerprint(plan),
        'orbital_storage': 'artifacts',
        'lifecycle': copy.deepcopy(plan.lifecycle),
        'cases': initial_cases or {},
    }
    if not path.is_file():
        if initial_cases is None and ((path.parent / 'study-report.json').exists() or discover_receipts(path.parent)):
            raise ValueError('Study checkpoint is missing for an existing execution: {0}'.format(path))
        return default
    payload = load_checkpoint(path, plan.study_id)
    if payload.get('parent_study'):
        raise ValueError('Historical child execution is read-only; resume tasks under the original Study')
    payload['study_fingerprint'] = _study_fingerprint(plan)
    if not isinstance(payload.get('lifecycle'), dict):
        payload['lifecycle'] = copy.deepcopy(plan.lifecycle)
    return payload


def _study_context(study_dir, plan, supplied_report, *, review_kind, collect_only):
    """Load one Study and keep its complete task plan across subset executions."""
    if supplied_report is not None and supplied_report.get('study_id') != plan.study_id:
        raise ValueError('The reviewed plan and report must belong to the same Study')
    adaptive_path = study_dir / 'adaptive-study-report.json'
    report_path = adaptive_path if adaptive_path.exists() or (
        (supplied_report or {}).get('schema') == ADAPTIVE_STUDY_REPORT_SCHEMA
    ) else study_dir / 'study-report.json'
    persisted = report_path.is_file()
    previous = read_json(report_path) if persisted else copy.deepcopy(supplied_report or {})
    if previous and (previous.get('study_id') != plan.study_id or previous.get('system_type') != plan.system_type):
        raise ValueError('Saved report belongs to a different Study or system type')
    previous_cases = case_index(previous.get('cases') or [], 'Study report')
    if not persisted and any(
        (case.get('task_report') or {}).get('execution_status') not in (None, 'blocked', 'not_executed')
        for case in previous_cases.values()
    ):
        raise ValueError('Historical results require saved execution evidence; the supplied report is read-only')
    plan_path = study_dir / 'study-plan.json'
    saved = read_json(plan_path) if plan_path.is_file() else None
    if saved and saved.get('study_id') != plan.study_id:
        raise ValueError('Saved plan belongs to a different Study')
    known = case_index(saved['cases'] if saved else list(previous_cases.values()), 'Stored Study plan')
    selected = case_index([case.to_dict() for case in plan.cases], 'Execution plan')
    if review_kind and known and not set(selected).issubset(known):
        raise ValueError('Review contains tasks outside the Study')
    full = copy.deepcopy(saved or plan.to_dict())
    base_cases = saved['cases'] if saved else [
        StudyCase.from_dict(case).to_dict() for case in previous_cases.values()
    ]
    if collect_only and saved:
        # The persisted plan defines the work already submitted. A subset is a view, not a new plan.
        if any(key not in known or _case_fingerprint(StudyCase.from_dict(case)) !=
               _case_fingerprint(StudyCase.from_dict(known[key])) for key, case in selected.items()):
            raise ValueError('Collect inputs differ from the executed plan')
    else:
        full['cases'] = merge_case_records(base_cases, list(selected.values()))
        full['observables'] = list(dict.fromkeys((full.get('observables') or []) + plan.observables))
        full['resource_policy'] = copy.deepcopy(plan.resource_policy)
        full['cost_estimate'] = copy.deepcopy(plan.cost_estimate)
    full_plan = StudyPlan.from_dict(full)

    # Import completed pre-existing stage/child results once, without submitting or moving runs.
    state_path = study_dir / 'study-state.json'
    initial_cases = {} if persisted and not state_path.exists() and (
        report_path == adaptive_path or (not previous_cases and not discover_receipts(study_dir))
    ) else None
    loaded = {}
    for key, case in previous_cases.items():
        reference = case.pop('task_reference', None)
        source = Path(reference['checkpoint']).resolve() if reference else state_path
        if source != state_path:
            if source not in loaded:
                loaded[source] = load_checkpoint(source, reference['study_id'])
            state = loaded[source]
            owner = state.get('parent_study') or {}
            if owner and owner.get('study_id') != plan.study_id:
                raise ValueError('Historical execution belongs to a different Study')
            case = copy.deepcopy(state['cases'].get(key) or {})
            if not case or (case.get('execution') or {}).get('pending'):
                raise ValueError('Collect the historical execution before migrating its tasks')
            initial_cases = initial_cases or {}
        if initial_cases is not None:
            case = copy.deepcopy(case)
            case.pop('task_reference', None)
            case.setdefault('attempt_count', 1 if case.get('task_report') else 0)
            case.setdefault('task_report', {'execution_status': 'not_executed', 'structured_results': {}})
            case.setdefault('case_fingerprint', _case_fingerprint(StudyCase.from_dict(case)))
            initial_cases[key] = case
    if initial_cases:
        # Historical retries may have changed the request as well as the result.
        known.update({key: {**known.get(key, {}), **case} for key, case in initial_cases.items()})
        full_plan.cases = [
            StudyCase.from_dict(known[case.case_id])
            if case.case_id in initial_cases and (collect_only or case.case_id not in selected)
            else case
            for case in full_plan.cases
        ]
    return previous, report_path, full_plan, initial_cases, known


def _complete_task_view(plan, state, updated_cases, previous, study_dir):
    """Project every task's current run; retain annotations on unchanged results."""
    changed = {case['case_id']: case for case in updated_cases}
    old_cases = {case['case_id']: case for case in previous.get('cases') or []}
    old_rows = {row['case_id']: row for row in previous.get('comparison_table') or []}
    cases, comparison = [], []
    for case in plan.cases:
        key = case.case_id
        record = copy.deepcopy(changed.get(key) or state['cases'].get(key) or case.to_dict())
        record.pop('task_reference', None)
        execution = record.get('execution') or {}
        if execution.get('pending') or not record.get('task_report'):
            record['task_report'] = {
                'execution_status': 'pending' if execution.get('pending') else 'not_executed',
                'structured_results': {},
            }
        unchanged = record.get('task_report') == old_cases.get(key, {}).get('task_report')
        row = copy.deepcopy(old_rows.get(key, {})) if unchanged else {}
        row.update(_case_summary(
            case, record['task_report'], plan.observables, study_dir,
            execution_source=record.get('execution_source') or 'resumed',
            attempt_count=record.get('attempt_count', 0),
        ))
        cases.append(record)
        comparison.append(row)
    return cases, comparison


def _normalize_case_ids(values: Optional[Iterable[Any]]) -> Set[str]:
    if not isinstance(values, (list, tuple, set)):
        return set()
    return {
        str(value).strip()
        for value in values
        if str(value).strip()
    }


def _normalize_statuses(values: Optional[Iterable[Any]]) -> Set[str]:
    if not isinstance(values, (list, tuple, set)):
        return set()
    return {
        str(value).strip().lower()
        for value in values
        if str(value).strip()
    }


def _non_negative_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _checkpoint_case_report(
    case,
    task_report: Dict[str, Any],
    *,
    case_fingerprint: str,
    attempt_count: int,
    execution_source: str,
) -> Dict[str, Any]:
    return {
        'case_id': case.case_id,
        'label': case.label,
        'variables': copy.deepcopy(case.variables),
        'operations': copy.deepcopy(case.operations),
        'request': copy.deepcopy(case.request),
        'task_report': copy.deepcopy(task_report),
        'case_fingerprint': case_fingerprint,
        'attempt_count': int(attempt_count),
        'execution_source': execution_source,
    }


def _checkpoint_artifact(
    checkpoint: Dict[str, Any],
    kind: str,
) -> Optional[Dict[str, Any]]:
    task_report = checkpoint.get('task_report') if isinstance(checkpoint, dict) else None
    artifacts = task_report.get('artifacts') if isinstance(task_report, dict) else None
    if not isinstance(artifacts, list):
        return None
    for artifact in artifacts:
        if (
            isinstance(artifact, dict)
            and artifact.get('kind') == kind
            and artifact.get('path')
        ):
            return copy.deepcopy(artifact)
    return None


def _resolve_case_density_state(
    case,
    study_state: Dict[str, Any],
):
    """Resolve an adjacent-case 1RDM dependency immediately before execution."""
    resolved_case = copy.deepcopy(case)
    request = resolved_case.request if isinstance(resolved_case.request, dict) else {}
    initial_state = request.get('initial_state') if isinstance(request.get('initial_state'), dict) else None
    if not initial_state or initial_state.get('mode') != 'projected_1rdm':
        return resolved_case, None
    artifact = initial_state.get('source_artifact')
    if isinstance(artifact, dict) and artifact.get('path'):
        return resolved_case, None

    source_case_id = str(
        initial_state.get('source_case_id')
        or (artifact.get('deferred_case_id') if isinstance(artifact, dict) else '')
        or ''
    ).strip()
    if not source_case_id:
        return resolved_case, 'Projected-1RDM continuation is missing source_case_id.'
    source_checkpoint = (
        study_state.get('cases', {}).get(source_case_id)
        if isinstance(study_state.get('cases'), dict)
        else None
    )
    source_report = (
        source_checkpoint.get('task_report')
        if isinstance(source_checkpoint, dict)
        and isinstance(source_checkpoint.get('task_report'), dict)
        else {}
    )
    source_status = str(source_report.get('execution_status') or '').strip().lower()
    source_artifact = _checkpoint_artifact(source_checkpoint or {}, 'one_particle_state')
    if source_status != 'succeeded' or not source_artifact:
        return resolved_case, (
            'Projected-1RDM continuation for {0} requires the succeeded preceding branch case {1} '
            'and its one_particle_state artifact; the dependency is currently unavailable.'
        ).format(case.case_id, source_case_id)
    initial_state['source_case_id'] = source_case_id
    initial_state['source_artifact'] = source_artifact
    request['initial_state'] = initial_state
    resolved_case.request = request
    return resolved_case, None


def _resolve_case_initial_state(case, study_state: Dict[str, Any]):
    """Resolve density and orbital artifact dependencies using the saved parent Run.

    An orbital_restart_manifest or restart_manifest may be a deferred reference containing
    source_case_id and artifact_kind. It becomes the parent's persisted artifact
    path; checkpoint compatibility remains the receiving solver's responsibility.
    """
    resolved, error = _resolve_case_density_state(case, study_state)
    if error:
        return resolved, error
    from .dmet_continuation import resolve_dmet_density
    error = resolve_dmet_density(resolved, study_state)
    if error:
        return resolved, error
    request = resolved.request if isinstance(resolved.request, dict) else {}
    solver = request.get('solver') if isinstance(request.get('solver'), dict) else {}
    options = solver.get('options') if isinstance(solver.get('options'), dict) else {}
    if options.get('orbital_restart_manifest') and options.get('restart_manifest'):
        return resolved, 'Orbital-only continuation cannot also request an MPS restart.'
    option_key = 'restart_manifest' if options.get('restart_manifest') else 'orbital_restart_manifest'
    reference = options.get(option_key)
    if not isinstance(reference, dict) or 'source_case_id' not in reference:
        return resolved, None
    source_id = str(reference.get('source_case_id') or '').strip()
    artifact_kind = str(reference.get('artifact_kind') or '').strip()
    if (set(reference) - {'source_case_id', 'artifact_kind'} or not source_id
            or not artifact_kind or source_id == case.case_id):
        return resolved, 'Invalid deferred orbital artifact reference: require a preceding source_case_id and artifact_kind only.'
    cases = study_state.get('cases') or {}
    checkpoint = cases.get(source_id) or {}
    source_report = checkpoint.get('task_report') or {}
    artifact = _checkpoint_artifact(checkpoint, artifact_kind)
    if source_report.get('execution_status') != 'succeeded' or not artifact:
        return resolved, (
            'Orbital continuation for {0} requires the succeeded preceding case {1} '
            'and its {2} artifact; the dependency is currently unavailable.'
        ).format(case.case_id, source_id, artifact_kind)
    results = source_report.get('structured_results') or {}
    if results.get('converged') is False:
        return resolved, 'Orbital continuation requires a converged source case: {0}.'.format(source_id)
    options[option_key] = artifact['path']
    provenance_key = 'restart_provenance' if option_key == 'restart_manifest' else 'orbital_restart_provenance'
    provenance = copy.deepcopy(options.get(provenance_key) or {})
    provenance.update(source_case_id=source_id, source_run_id=source_report.get('run_id'),
                      source_artifact=artifact,
                      transfer='orbitals_and_mps' if option_key == 'restart_manifest' else 'orbitals_only')
    options[provenance_key] = provenance
    return resolved, None


def _blocked_dependency_report(
    case,
    *,
    message: str,
    work_dir: Path,
    run_id: str,
) -> Dict[str, Any]:
    request = case.request if isinstance(case.request, dict) else {}
    return {
        'execution_status': 'blocked',
        'work_dir': str(work_dir),
        'run_id': run_id,
        'validation_errors': [message],
        'analysis_summary': 'Task was not executed: {0}'.format(message),
        'structured_results': {
            'task_type': request.get('task_type') or 'molecular',
            'method': request.get('method'),
        },
        'artifacts': [],
    }


def _uses_projected_1rdm(case: Any) -> bool:
    request = case.request if isinstance(getattr(case, 'request', None), dict) else {}
    initial_state = (
        request.get('initial_state')
        if isinstance(request.get('initial_state'), dict)
        else {}
    )
    return str(initial_state.get('mode') or '').strip().lower() == 'projected_1rdm'


def _uses_solver_continuation(case: Any) -> bool:
    request = case.request if isinstance(getattr(case, 'request', None), dict) else {}
    solver = request.get('solver') if isinstance(request.get('solver'), dict) else {}
    options = solver.get('options') if isinstance(solver.get('options'), dict) else {}
    return bool(options.get('orbital_restart_manifest') or options.get('restart_manifest')
                or options.get('reference_density_source'))


def _supports_independent_batch(executor: TaskExecutor) -> bool:
    return (
        getattr(executor, 'supports_independent_batch', None) is True
        and isinstance(executor, BatchTaskExecutor)
    )


def _case_execution_context(
    case: Any,
    study_state: Dict[str, Any],
    *,
    resume: bool,
    selected_case_ids: Set[str],
    selected_statuses: Set[str],
    max_case_attempts: int,
) -> Dict[str, Any]:
    execution_case, dependency_error = _resolve_case_initial_state(case, study_state)
    fingerprint = _case_fingerprint(execution_case)
    if dependency_error:
        fingerprint = json_fingerprint({
            'case': fingerprint,
            'dependency_error': dependency_error,
        })
    checkpoint = study_state['cases'].get(case.case_id)
    checkpoint_matches = (
        resume
        and isinstance(checkpoint, dict)
        and checkpoint.get('case_fingerprint') == fingerprint
        and isinstance(checkpoint.get('task_report'), dict)
    )
    previous_status = str(
        (checkpoint.get('task_report') or {}).get('execution_status')
        if isinstance(checkpoint, dict)
        else ''
    ).strip().lower()
    previous_attempts = (
        _non_negative_int(checkpoint.get('attempt_count'))
        if isinstance(checkpoint, dict)
        else 0
    )
    explicitly_selected = (
        case.case_id in selected_case_ids
        or previous_status in selected_statuses
    )
    should_execute = not checkpoint_matches or not resume or explicitly_selected
    execution_source = 'executed'
    if checkpoint_matches and not explicitly_selected:
        if previous_status in ('succeeded', 'blocked'):
            should_execute = False
            execution_source = 'resumed'
        elif (
            previous_status in RETRYABLE_CASE_STATUSES
            and previous_attempts >= max_case_attempts
        ):
            should_execute = False
            execution_source = 'attempt_limit_reached'
        elif previous_status in RETRYABLE_CASE_STATUSES:
            should_execute = True
        elif previous_status == 'not_executed':
            should_execute = True
        else:
            should_execute = False
            execution_source = 'resumed'
    if (selected_case_ids or selected_statuses) and not explicitly_selected:
        should_execute = False
        execution_source = 'resumed'
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('task_report'), dict):
            raise ValueError('Unselected case has no result to retain: {0}'.format(case.case_id))
    return {
        'execution_case': execution_case,
        'dependency_error': dependency_error,
        'fingerprint': fingerprint,
        'checkpoint': checkpoint,
        'checkpoint_matches': checkpoint_matches,
        'previous_attempts': previous_attempts,
        'should_execute': should_execute,
        'execution_source': execution_source,
    }


def _supports_recoverable_batch(executor: TaskExecutor) -> bool:
    return (
        getattr(executor, 'supports_recoverable_batch', False) is True
        and all(callable(getattr(executor, name, None)) for name in (
        'submit_independent_tasks',
        'wait_for_independent_tasks',
        'collect_independent_tasks',
        ))
    )


def _supports_process_submission(executor):
    return executor_identity(executor).get('location') == 'child_process'


def _execution_receipt_artifact(path: Path) -> Dict[str, Any]:
    reference = default_artifact_repository().register_existing(
        path,
        kind='execution-receipt',
        mime_type='application/json; charset=utf-8',
        description=(
            'Submitted scheduler handles used to inspect and collect this study '
            'without resubmitting completed tasks'
        ),
    )
    if reference is None:
        raise FileNotFoundError('Execution receipt is unavailable: {0}'.format(path))
    return reference


def _save_state(path: Path, state: Dict[str, Any]) -> None:
    with REPORT_LOCK:
        _write_json_atomic(path, state, kind='study-state',
                           description='Per-case results and current execution association')


def _collect_receipt(executor, path: Path, state, contexts, state_path, *, wait: bool) -> BatchExecutionResult:
    receipt = load_receipt(path)
    jobs = jobs_from_receipt(receipt) if receipt else {}
    if not jobs or set(jobs) != set(receipt.get('task_ids') or jobs):
        raise ValueError('Submission outcome is unknown; reconcile the existing execution before retrying: {0}'.format(path))
    if receipt.get('study_id') != state['study_id']:
        raise ValueError('Execution receipt belongs to a different study')
    # Validate the target before querying or fetching a persisted handle.
    if receipt.get('executor') != executor_identity(executor):
        raise ValueError('Execution receipt belongs to a different executor')
    batches = batches_from_receipt(receipt)
    if not wait and receipt.get('status') != 'collected':
        inspected = inspect_receipt(receipt, executor)
        if inspected.get('status') != 'results_available':
            raise StudyExecutionInterrupted('Execution results are not all available yet; inspect and collect later.', inspected)
    try:
        if batches:
            if wait:
                executor.wait_for_independent_tasks(batches)
            result = executor.collect_independent_tasks(batches)
        else:
            if wait:
                for handle in jobs.values():
                    while not executor.status(handle).terminal:
                        time.sleep(0.1)
            result = BatchExecutionResult(reports={key: executor.fetch(handle) for key, handle in jobs.items()}, batches=[])
        result = _finish_cases(state, contexts, result, state_path,
                               expected_runs={key: handle.run_id for key, handle in jobs.items()})
    except Exception as exc:
        try:
            receipt = inspect_receipt(receipt, executor)
        except Exception:
            receipt = copy.deepcopy(receipt)
        receipt['status'] = 'connection_interrupted'
        receipt['last_error'] = '{0}: {1}'.format(type(exc).__name__, exc)
        write_receipt(path, receipt)
        raise StudyExecutionInterrupted('Collect the persisted execution without resubmitting tasks.', receipt) from exc
    receipt['status'] = 'collected'
    receipt['last_error'] = None
    receipt['collected_task_ids'] = sorted(result.reports)
    write_receipt(path, receipt)
    result.artifacts.append(_execution_receipt_artifact(path))
    return result


def _finish_cases(state, contexts, result, state_path, *, expected_runs=None):
    """Accept a complete result batch once, before changing any case checkpoint."""
    if isinstance(result, dict):
        result = BatchExecutionResult.from_dict(result)
    if not isinstance(result, BatchExecutionResult):
        raise ExecutorContractError('Batch executor must return a BatchExecutionResult')
    current = {
        c['execution_case'].case_id: state['cases'][c['execution_case'].case_id]['execution']['run_id']
        for c in contexts
    }
    expected = expected_runs if expected_runs is not None else current
    if any(expected.get(key) != run_id for key, run_id in current.items()):
        raise ExecutorContractError('Submitted handle does not match the current execution')
    if set(result.reports) != set(expected):
        raise ExecutorContractError('Batch executor omitted expected TaskReports or returned unexpected cases')
    for case_id, report in result.reports.items():
        if not isinstance(report, dict) or not report.get('execution_status'):
            raise ExecutorContractError('Executor returned an invalid TaskReport')
        if report.get('run_id') not in (None, expected[case_id]) or report.get('case_id') not in (None, case_id):
            raise ExecutorContractError('TaskReport identity does not match the submitted execution')
        # Normalize older adapters at this boundary; downstream code sees a run ID.
        report['run_id'] = report.get('run_id') or expected[case_id]
    for context in contexts:
        case = context['execution_case']
        previous = state['cases'][case.case_id]
        execution = copy.deepcopy(previous['execution'])
        task_report = copy.deepcopy(result.reports[case.case_id])
        execution['pending'] = False
        checkpoint = _checkpoint_case_report(
            case, task_report, case_fingerprint=context['fingerprint'],
            attempt_count=execution['attempt_count'], execution_source='executed',
        )
        checkpoint['execution'] = execution
        from .dmet_branches import preserve_candidates
        preserve_candidates(previous, checkpoint)
        state['cases'][case.case_id] = checkpoint
    _save_state(state_path, state)
    return result


def _execute_contexts(plan, executor, contexts, study_work_dir, state, *, locale,
                      resource_profile, independent):
    state_path = study_work_dir / 'study-state.json'
    tasks = []
    for context in contexts:
        case = context['execution_case']
        attempt = context['previous_attempts'] + 1
        run_id = _case_run_id(case.case_id)
        if attempt > 1 or (study_work_dir / 'cases' / run_id).exists():
            run_id += '-attempt-{0}-{1}'.format(attempt, uuid4().hex[:8])
        tasks.append(BatchTask(task_id=case.case_id, request=_request_text(case.request),
                               channel='study', locale=locale,
                               work_dir=str(study_work_dir / 'cases'), run_id=run_id))
    recoverable_batch = _supports_recoverable_batch(executor)
    recoverable = recoverable_batch or _supports_process_submission(executor)
    path = None
    if recoverable:
        path = (receipt_path(study_work_dir) if independent else
                study_work_dir / 'execution-receipts' / tasks[0].run_id / 'execution-receipt.json')
        if path.exists():
            path = study_work_dir / 'execution-receipts' / ('batch-' + uuid4().hex) / 'execution-receipt.json'
    for context, task in zip(contexts, tasks):
        checkpoint = copy.deepcopy(context['checkpoint'] or {})
        checkpoint.update(case_id=task.task_id, case_fingerprint=context['fingerprint'],
                          attempt_count=context['previous_attempts'])
        checkpoint['execution'] = {
            'run_id': task.run_id, 'pending': True,
            'attempt_count': context['previous_attempts'] + 1,
            'receipt_path': str(path) if path else None,
        }
        state['cases'][task.task_id] = checkpoint
    _save_state(state_path, state)
    kwargs = {'resource_profile': resource_profile} if resource_profile else {}
    if recoverable:
        receipt = new_receipt(
            study_id=plan.study_id, study_fingerprint=_study_fingerprint(plan),
            task_fingerprints={t.task_id: json_fingerprint({'request': t.request, 'channel': t.channel,
                                                      'locale': t.locale, 'run_id': t.run_id}) for t in tasks},
            executor=executor, batches=[], resource_profile=resource_profile,
        )
        # Persist intent BEFORE crossing the transport boundary. If submission
        # raises or the process dies before handle persistence, do not resubmit.
        receipt['status'] = 'submission_unknown'
        write_receipt(path, receipt)
        try:
            if recoverable_batch:
                batches = list(executor.submit_independent_tasks(tasks, **kwargs))
                if not batches:
                    raise ExecutorContractError('Submission returned no batch handles')
                receipt['batches'] = [batch.to_dict() for batch in batches]
            else:
                receipt['jobs'] = {}
                for task in tasks:
                    handle = executor.submit_task(task.request, channel=task.channel, locale=task.locale,
                                                  work_dir=task.work_dir, run_id=task.run_id, **kwargs)
                    receipt['jobs'][task.task_id] = handle.to_dict()
                    write_receipt(path, receipt)
        except Exception as exc:
            receipt['last_error'] = '{0}: {1}'.format(type(exc).__name__, exc)
            write_receipt(path, receipt)
            raise StudyExecutionInterrupted(
                'Submission outcome is unknown. Reconcile the existing job before starting another execution.',
                receipt,
            ) from exc
        receipt['status'] = 'submitted'
        write_receipt(path, receipt)
        for task in tasks:
            state['cases'][task.task_id]['attempt_count'] = state['cases'][task.task_id]['execution']['attempt_count']
        _save_state(state_path, state)
        result = _collect_receipt(executor, path, state, contexts, state_path, wait=True)
    else:
        for task in tasks:
            state['cases'][task.task_id]['attempt_count'] = state['cases'][task.task_id]['execution']['attempt_count']
        _save_state(state_path, state)
        if independent:
            result = executor.execute_independent_tasks(tasks, **kwargs)
        else:
            task = tasks[0]
            result = BatchExecutionResult(reports={task.task_id: executor.execute_task(
                task.request, channel=task.channel, locale=task.locale,
                work_dir=task.work_dir, run_id=task.run_id, **kwargs)}, batches=[])
        result = _finish_cases(state, contexts, result, state_path)
    return result


def _adopt_legacy_receipts(plan, executor, state, study_work_dir):
    changed = False
    for path in discover_receipts(study_work_dir):
        receipt = load_receipt(path)
        if receipt.get('study_id') != plan.study_id:
            continue
        handles = jobs_from_receipt(receipt)
        for case in plan.cases:
            checkpoint = state['cases'].get(case.case_id) or {}
            if checkpoint.get('execution') or case.case_id not in handles:
                continue
            if state.get('execution_associations_version') == 1:
                raise ValueError('Execution association is missing for a persisted receipt')
            # A complete v1 checkpoint remains usable without inventing a retry.
            resolved, error = _resolve_case_initial_state(case, state)
            finished = isinstance(checkpoint.get('task_report'), dict)
            if not finished and (error or receipt.get('study_fingerprint') != _study_fingerprint(plan)
                                 or receipt.get('executor') != executor_identity(executor)):
                raise ValueError('Legacy pending receipt does not match the execution request')
            attempt = checkpoint.get('attempt_count', 1)
            checkpoint.update(case_id=case.case_id, attempt_count=attempt)
            checkpoint.setdefault('case_fingerprint', _case_fingerprint(resolved))
            checkpoint['execution'] = {'run_id': handles[case.case_id].run_id,
                                       'receipt_path': str(path), 'pending': not finished,
                                       'attempt_count': attempt}
            state['cases'][case.case_id] = checkpoint
            changed = True
    if changed:
        _save_state(study_work_dir / 'study-state.json', state)
    state['execution_associations_version'] = 1


def _resume_pending_cases(plan, executor, state, study_work_dir, *, collect_only):
    results = BatchExecutionResult(reports={}, batches=[])
    pending = {}
    for case in plan.cases:
        checkpoint = state['cases'].get(case.case_id) or {}
        execution = checkpoint.get('execution') or {}
        if not execution.get('pending'):
            continue
        case_resolved, error = _resolve_case_initial_state(case, state)
        if error or checkpoint.get('case_fingerprint') != _case_fingerprint(case_resolved):
            raise ValueError('A pending execution has different inputs; collect its original plan first')
        path = execution.get('receipt_path')
        if not path:
            raise ValueError('Interrupted local execution has no recoverable handle; reconcile its TaskReport before retrying')
        path = Path(path).resolve()
        path.relative_to(study_work_dir.resolve())
        pending.setdefault(path, []).append({'execution_case': case_resolved,
                                             'fingerprint': checkpoint['case_fingerprint']})
    for path, contexts in pending.items():
        result = _collect_receipt(executor, path, state, contexts,
                                  study_work_dir / 'study-state.json', wait=not collect_only)
        results.reports.update({c['execution_case'].case_id: result.reports[c['execution_case'].case_id] for c in contexts})
        results.batches.extend(result.batches)
        results.artifacts.extend(result.artifacts)
    return results


def _write_text(path: Path, content: str, *, kind: str, description: str = '') -> Dict[str, Any]:
    return default_artifact_repository().write_text(
        path,
        content,
        kind=kind,
        description=description,
    )


def _request_text(request: Dict[str, Any]) -> str:
    return json.dumps(request, ensure_ascii=False)


def _observable_value(results: Dict[str, Any], observable: str) -> Any:
    summary_column = MODEL_OBSERVABLE_SUMMARY_COLUMNS.get(observable)
    if summary_column:
        return results.get(summary_column)
    if observable == 'energy_gap':
        return results.get('gap')
    if observable in results:
        value = results[observable]
        if isinstance(value, dict) and value.get('kind') in ('site_vector', 'pair_matrix', 'q_vector'):
            return None
        return value
    return None


def _relative_run_dir(task_report: Dict[str, Any], study_work_dir: Path) -> str:
    work_dir = task_report.get('work_dir')
    run_id = task_report.get('run_id')
    if not work_dir or not run_id:
        return ''
    run_dir = Path(str(work_dir)) / str(run_id)
    try:
        return str(run_dir.relative_to(study_work_dir))
    except ValueError:
        return str(run_dir)


def _with_energy_unit(value: Any, unit: str) -> Any:
    if value is None or not unit:
        return value
    if isinstance(value, (int, float)):
        return '{0} {1}'.format(value, unit)
    return value


def _is_model_energy_parameter_column(name: str) -> bool:
    normalized = str(name or '').strip()
    if not normalized:
        return False
    lowered = normalized.lower()
    compact = lowered.replace('-', '_')
    if compact in MODEL_ENERGY_PARAMETER_COLUMNS:
        return True
    return any(compact.startswith(prefix) for prefix in MODEL_ENERGY_PARAMETER_PREFIXES)


def _case_variables_with_units(case, energy_unit: str) -> Dict[str, Any]:
    variables = copy.deepcopy(case.variables)
    if not case.model_spec or not energy_unit:
        return variables
    for key, value in list(variables.items()):
        if _is_model_energy_parameter_column(key):
            variables[key] = _with_energy_unit(value, energy_unit)
    return variables


def _observable_artifact_payload(results: Dict[str, Any], observable: str) -> Any:
    value = results.get(observable)
    if isinstance(value, dict) and value.get('kind') in ('site_vector', 'pair_matrix', 'q_vector'):
        return value
    if isinstance(value, list):
        return {'kind': 'array', 'values': value}
    return None


def _diagnostic_value(results: Dict[str, Any], name: str, key: str = None) -> Any:
    diagnostics = results.get('strong_correlation_diagnostics')
    items = diagnostics.get('diagnostics') if isinstance(diagnostics, dict) else None
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict) or item.get('name') != name:
            continue
        value = item.get('value')
        if key and isinstance(value, dict):
            return value.get(key)
        return value
    return None


def _model_filling(case, results: Dict[str, Any]) -> Any:
    diagnostics = results.get('strong_correlation_diagnostics')
    parameter_summary = diagnostics.get('parameter_summary') if isinstance(diagnostics, dict) else None
    if isinstance(parameter_summary, dict) and parameter_summary.get('filling') is not None:
        return parameter_summary.get('filling')
    sites = case.model_spec.get('sites') if isinstance(case.model_spec, dict) else None
    nelec = case.model_spec.get('nelec') if isinstance(case.model_spec, dict) else None
    if not isinstance(sites, list) or not sites or not isinstance(nelec, list) or len(nelec) != 2:
        return None
    try:
        return float((float(nelec[0]) + float(nelec[1])) / len(sites))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _add_model_scalar_summary(row: Dict[str, Any], case, results: Dict[str, Any], energy_unit: str) -> None:
    if results.get('task_type') != 'model_hamiltonian':
        return
    if str(results.get('solver') or '').strip().lower() != 'dmet':
        for column in ('reference_energy',):
            if results.get(column) is not None:
                row[column] = _with_energy_unit(results.get(column), energy_unit)
    filling = _model_filling(case, results)
    if filling is not None:
        row['filling'] = filling
    gap = results.get('gap')
    if gap is None:
        gap = _diagnostic_value(results, 'many_body_gap', 'gap')
    if gap is not None:
        row['gap'] = _with_energy_unit(gap, energy_unit)
    mean_field_gap = results.get('mean_field_gap')
    if mean_field_gap is None:
        mean_field_gap = _diagnostic_value(results, 'mean_field_homo_lumo_gap', 'gap')
    if mean_field_gap is not None:
        row['mean_field_gap'] = _with_energy_unit(mean_field_gap, energy_unit)
    natural_fractionality = results.get('natural_occupation_fractionality')
    if natural_fractionality is None:
        natural_fractionality = _diagnostic_value(results, 'natural_orbital_occupations', 'average_fractionality')
    if natural_fractionality is None:
        natural_fractionality = _diagnostic_value(results, 'natural_orbital_occupations', 'max_fractionality')
    if natural_fractionality is not None:
        row['natural_occupation_fractionality'] = natural_fractionality
    fractional_count = results.get('fractional_natural_orbital_count')
    if fractional_count is None:
        fractional_count = _diagnostic_value(results, 'natural_orbital_occupations', 'fractional_orbital_count')
    if fractional_count is not None:
        row['fractional_natural_orbital_count'] = fractional_count
    max_t2 = results.get('max_double_excitation_amplitude')
    if max_t2 is None:
        max_t2 = _diagnostic_value(results, 'max_double_excitation_amplitude', 'max_abs_t2')
    if max_t2 is not None:
        row['max_double_excitation_amplitude'] = max_t2
    for metric in MODEL_POSTPROCESSING_METRICS:
        field = metric.comparison_field or metric.id
        diagnostic_name = metric.metadata.get('diagnostic_name')
        if not diagnostic_name:
            continue
        value = _diagnostic_value(
            results,
            str(diagnostic_name),
            key=metric.metadata.get('diagnostic_key'),
        )
        if value is not None:
            row[field] = value


def _add_molecular_scalar_summary(row: Dict[str, Any], results: Dict[str, Any]) -> None:
    if results.get('task_type') != 'molecular':
        return
    diagnostics = results.get('correlation_diagnostics')
    if not isinstance(diagnostics, dict):
        return
    t1_d1 = diagnostics.get('ccsd_t1_d1')
    if not isinstance(t1_d1, dict) or t1_d1.get('status') != 'available':
        return
    if t1_d1.get('t1') is not None:
        row['ccsd_t1'] = t1_d1['t1']
    if t1_d1.get('d1') is not None:
        row['ccsd_d1'] = t1_d1['d1']


def _block2_dmrg_result(results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    direct = results.get('dmrg_result')
    if isinstance(direct, dict):
        return direct
    cas_result = results.get('cas_result')
    nested = cas_result.get('dmrg_result') if isinstance(cas_result, dict) else None
    return nested if isinstance(nested, dict) else None


def _add_block2_scalar_summary(row: Dict[str, Any], results: Dict[str, Any], energy_unit: str) -> None:
    dmrg = _block2_dmrg_result(results)
    if not isinstance(dmrg, dict):
        return
    state_energies = dmrg.get('state_energies')
    if isinstance(state_energies, list):
        row['dmrg_state_count'] = len(state_energies)
    excitation_energies = dmrg.get('excitation_energies')
    if isinstance(excitation_energies, list) and len(excitation_energies) > 1:
        row['first_excitation_energy'] = _with_energy_unit(excitation_energies[1], energy_unit)
    entanglement = dmrg.get('entanglement_diagnostics')
    if isinstance(entanglement, dict):
        scalar_fields = {
            'max_single_orbital_entropy': 'max_single_orbital_entropy',
            'mean_single_orbital_entropy': 'mean_single_orbital_entropy',
            'max_mutual_information': 'max_mutual_information',
            'max_bipartite_entanglement': 'max_bipartite_entanglement',
        }
        for source, target in scalar_fields.items():
            if entanglement.get(source) is not None:
                row[target] = entanglement[source]
    symmetry = dmrg.get('symmetry_analysis')
    if isinstance(symmetry, dict):
        for source, target in (
            ('particle_number', 'dmrg_particle_number'),
            ('spin_square', 'dmrg_spin_square'),
            ('inferred_total_spin', 'dmrg_inferred_total_spin'),
            ('spin_square_deviation', 'dmrg_spin_square_deviation'),
        ):
            if symmetry.get(source) is not None:
                row[target] = symmetry[source]
    convergence = dmrg.get('convergence')
    if isinstance(convergence, dict):
        if convergence.get('final_discarded_weight') is not None:
            row['dmrg_final_discarded_weight'] = convergence['final_discarded_weight']
        if convergence.get('final_energy_change') is not None:
            row['dmrg_final_energy_change'] = _with_energy_unit(
                convergence['final_energy_change'],
                energy_unit,
            )
    error_estimate = dmrg.get('energy_error_estimate')
    if isinstance(error_estimate, dict) and error_estimate.get('status') == 'available':
        if error_estimate.get('estimated_absolute_error') is not None:
            row['dmrg_estimated_energy_error'] = _with_energy_unit(
                error_estimate['estimated_absolute_error'],
                energy_unit,
            )
        if error_estimate.get('extrapolated_energy') is not None:
            row['dmrg_extrapolated_energy'] = _with_energy_unit(
                error_estimate['extrapolated_energy'],
                energy_unit,
            )
        row['dmrg_error_estimate_method'] = error_estimate.get('method')
    recovery = dmrg.get('recovery_recommendation')
    if isinstance(recovery, dict):
        row['dmrg_recovery_action'] = recovery.get('recommended_action')
        row['dmrg_failure_class'] = recovery.get('failure_class')
        row['dmrg_recovery_summary'] = recovery.get('summary')
        row['dmrg_automatic_retry_safe'] = recovery.get('automatic_retry_safe')
    adaptive_schedule = dmrg.get('adaptive_schedule')
    if isinstance(adaptive_schedule, dict):
        row['dmrg_adaptive_stages'] = adaptive_schedule.get('adaptive_stages_used')
        row['dmrg_final_bond_dimension'] = adaptive_schedule.get('final_bond_dimension')
        row['dmrg_adaptive_budget_exhausted'] = adaptive_schedule.get('budget_exhausted')
    restart = dmrg.get('restart')
    if isinstance(restart, dict) and restart.get('requested'):
        row['mps_restart_applied'] = bool(restart.get('applied'))
        if restart.get('source_case_id'):
            row['mps_restart_source_case'] = restart['source_case_id']


def _add_common_scalar_summary(row: Dict[str, Any], results: Dict[str, Any], energy_unit: str) -> None:
    final_energy = results.get('final_energy')
    if final_energy is None:
        final_energy = results.get('energy')
    if final_energy is not None:
        row['final_energy'] = _with_energy_unit(final_energy, energy_unit)
    if results.get('energy_per_site') is not None:
        row['energy_per_site'] = _with_energy_unit(results['energy_per_site'], '{0}/site'.format(energy_unit))
    for field in ('energy_over_abs_t', 'energy_per_site_over_abs_t'):
        if results.get(field) is not None:
            row[field] = results[field]
    is_dmet = (
        results.get('task_type') == 'model_hamiltonian'
        and str(results.get('solver') or '').strip().lower() == 'dmet'
    )
    if is_dmet:
        return
    for column in ('reference_energy',):
        if column not in row and results.get(column) is not None:
            row[column] = _with_energy_unit(results.get(column), energy_unit)


def _case_summary(
    case,
    task_report: Dict[str, Any],
    observables: List[str],
    study_work_dir: Path,
    *,
    execution_source: str = 'executed',
    attempt_count: int = 1,
) -> Dict[str, Any]:
    structured = task_report.get('structured_results') or {}
    execution_status = str(task_report.get('execution_status') or '').strip().lower()
    quality_decision = next((
        decision for decision in reversed(task_report.get('gate_decisions') or [])
        if isinstance(decision, dict)
        and decision.get('gate_id') == 'task.execution_quality'
    ), {})
    quality_details = (
        quality_decision.get('details')
        if isinstance(quality_decision.get('details'), dict)
        else {}
    )
    quality_status = str(quality_details.get('quality_status') or 'legacy')
    publication_eligible = quality_details.get('publication_eligible')
    if execution_status != 'succeeded':
        publication_eligible = False
    energy_unit = str(structured.get('energy_unit') or case.model_spec.get('energy_unit') or '').strip()
    if not energy_unit and structured.get('task_type') != 'model_hamiltonian':
        energy_unit = 'Ha'
    row = {
        'case_id': case.case_id,
        'label': case.label,
        'status': task_report.get('execution_status'),
        'quality_status': quality_status,
        'publication_eligible': publication_eligible,
        'run_dir': _relative_run_dir(task_report, study_work_dir),
        'execution_source': execution_source,
        'attempt_count': attempt_count,
    }
    row.update(_case_variables_with_units(case, energy_unit))
    for observable in observables:
        if observable == 'correlation_energy':
            continue
        value = _observable_value(structured, observable)
        row[observable] = _with_energy_unit(value, energy_unit) if observable in MODEL_ENERGY_OBSERVABLE_COLUMNS else value
    _add_common_scalar_summary(row, structured, energy_unit)
    _add_model_scalar_summary(row, case, structured, energy_unit)
    _add_molecular_scalar_summary(row, structured)
    _add_block2_scalar_summary(row, structured, energy_unit)
    if not row.get('dmrg_recovery_action'):
        for error in reversed(task_report.get('errors') or []):
            details = error.get('details') if isinstance(error, dict) else None
            recovery = details.get('recovery_recommendation') if isinstance(details, dict) else None
            if not isinstance(recovery, dict):
                continue
            row['dmrg_recovery_action'] = recovery.get('recommended_action')
            row['dmrg_failure_class'] = recovery.get('failure_class')
            row['dmrg_recovery_summary'] = recovery.get('summary')
            row['dmrg_automatic_retry_safe'] = recovery.get('automatic_retry_safe')
            break
    if 'method' in structured:
        row['method'] = structured['method']
    if 'solver' in structured:
        row['solver'] = structured['solver']
    elif isinstance(case.request.get('solver'), dict) and case.request['solver'].get('name'):
        row['solver'] = case.request['solver']['name']
    if 'method' not in row and case.request.get('method') is not None:
        method = case.request.get('method')
        row['method'] = method.get('name') if isinstance(method, dict) else method
    return row


def _write_case_observable_artifacts(case, task_report: Dict[str, Any], observables: List[str], study_work_dir: Path) -> List[Dict[str, Any]]:
    structured = task_report.get('structured_results') or {}
    artifacts: List[Dict[str, Any]] = []
    for observable in observables:
        if observable == 'correlation_energy':
            continue
        payload = _observable_artifact_payload(structured, observable)
        if payload is None:
            continue
        artifact_path = study_work_dir / 'observables' / case.case_id / '{0}.json'.format(observable)
        artifact = _write_json(
            artifact_path,
            {
                'case_id': case.case_id,
                'observable': observable,
                'payload': payload,
            },
            kind='observable-{0}'.format(observable),
            description='Structured {0} output for study case {1}'.format(
                observable,
                case.case_id,
            ),
        )
        artifacts.append(artifact)
    return artifacts


def _comparison_table_text(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ''
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    lines = ['\t'.join(columns)]
    for row in rows:
        lines.append('\t'.join('' if row.get(column) is None else str(row.get(column)) for column in columns))
    return '\n'.join(lines) + '\n'


def _study_summary(plan: StudyPlan, rows: List[Dict[str, Any]]) -> str:
    succeeded = sum(1 for row in rows if row.get('status') == 'succeeded')
    pieces = [
        'Computational Study completed.',
        'name={0}'.format(plan.name),
        'objective={0}'.format(plan.objective),
        'system_type={0}'.format(plan.system_type),
        'cases={0}'.format(len(rows)),
        'succeeded={0}'.format(succeeded),
    ]
    energies = [
        row.get('energy')
        for row in rows
        if isinstance(row.get('energy'), (int, float))
    ]
    if energies:
        pieces.append('energy_min={0}'.format(min(energies)))
        pieces.append('energy_max={0}'.format(max(energies)))
    return '; '.join(pieces)


def _finalize_hamiltonian_dataset_study(
    plan: StudyPlan,
    case_reports: List[Dict[str, Any]],
    study_work_dir: Path,
) -> Dict[str, Any]:
    comparison = plan.comparison if isinstance(plan.comparison, dict) else {}
    if comparison.get('mode') != 'hamiltonian_dataset_assembly':
        return {}
    dataset_spec = comparison.get('dataset_spec')
    if not isinstance(dataset_spec, dict):
        raise ValueError(
            'Hamiltonian dataset studies require comparison.dataset_spec.'
        )
    manifest = finalize_hamiltonian_dataset(
        {
            'study_id': plan.study_id,
            'work_dir': str(study_work_dir),
            'cases': case_reports,
        },
        dataset_spec,
        study_work_dir / 'dataset',
    )
    return manifest.to_dict()


def _dataset_manifest_artifacts(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    artifact_paths = manifest.get('artifacts') if isinstance(manifest, dict) else None
    if not isinstance(artifact_paths, dict):
        return []
    specifications = (
        (
            'sample_index',
            'hamiltonian_sample_index',
            'application/x-ndjson; charset=utf-8',
            'Accepted QH9-compatible Hamiltonian sample index',
        ),
        (
            'rejection_index',
            'hamiltonian_rejection_index',
            'application/x-ndjson; charset=utf-8',
            'Rejected or missing Hamiltonian sample index',
        ),
        (
            'dataset_manifest',
            'hamiltonian_dataset_manifest',
            'application/json; charset=utf-8',
            'QH9-compatible dataset completion, split, and artifact manifest',
        ),
    )
    references = []
    for key, kind, mime_type, description in specifications:
        path = artifact_paths.get(key)
        if not path:
            continue
        reference = default_artifact_repository().register_existing(
            path,
            kind=kind,
            mime_type=mime_type,
            description=description,
        )
        if reference is not None:
            references.append(reference)
    return references


def run_study(
    plan_or_spec: Any,
    *,
    work_dir: str = None,
    locale: str = 'en',
    postprocess: bool = False,
    postprocess_specs: List[Dict[str, Any]] = None,
    resume: bool = True,
    rerun_case_ids: Optional[List[str]] = None,
    rerun_statuses: Optional[List[str]] = None,
    max_case_attempts: int = 2,
    task_executor: Optional[TaskExecutor] = None,
    batch_independent: bool = True,
    resource_profile: Optional[str] = None,
    study_report: Optional[Dict[str, Any]] = None,
    review_kind: Optional[str] = None,
    retry_guard: Optional[Dict[str, Any]] = None,
    grid_round_target: Optional[int] = None,
) -> StudyReport:
    plan = plan_or_spec if isinstance(plan_or_spec, StudyPlan) else build_study_plan(plan_or_spec)
    from computational_study_agent.grid.refinement import normalize_policy
    grid_policy = normalize_policy(plan.grid_refinement)
    runner = _run_grid_study if grid_policy and not (rerun_case_ids or rerun_statuses or review_kind) else _run_study
    extra = {}
    if grid_round_target is not None:
        if runner is not _run_grid_study:
            raise ValueError('grid_round_target requires a grid refinement Study')
        extra['grid_round_target'] = grid_round_target
    return runner(
        plan, work_dir=work_dir, locale=locale, postprocess=postprocess,
        postprocess_specs=postprocess_specs, resume=resume, rerun_case_ids=rerun_case_ids,
        rerun_statuses=rerun_statuses, max_case_attempts=max_case_attempts,
        task_executor=task_executor, batch_independent=batch_independent,
        resource_profile=resource_profile,
        study_report=study_report, review_kind=review_kind, retry_guard=retry_guard, **extra,
    )


@_serialized_execution
def _run_grid_study(plan, **kwargs):
    from computational_study_agent.grid.execution import run_grid_study
    return run_grid_study(plan, runner=_run_study, **kwargs)


@_serialized_execution
def reconcile_study_execution(plan: StudyPlan, *, submission_path: str, work_dir=None,
                              receipt_file=None, task_executor=None) -> Dict[str, Any]:
    """Associate an unknown submission with matching, persisted server evidence."""
    root = resolve_work_dir(work_dir).resolve()
    study_root = (root / plan.study_id).resolve()
    study_root.relative_to(root)
    path = Path(receipt_file).expanduser().resolve() if receipt_file else receipt_path(study_root)
    path.relative_to(study_root)
    receipt = load_receipt(path)
    if receipt is None or receipt.get('study_id') != plan.study_id:
        raise ValueError('Execution receipt does not belong to this study')
    if (receipt.get('study_fingerprint') != _study_fingerprint(plan)
            or receipt.get('executor') != executor_identity(task_executor)):
        raise ValueError('Execution receipt does not match the plan or executor')
    recover = getattr(task_executor, 'recover_submission', None)
    if not callable(recover):
        raise ValueError('Executor cannot read persisted submission evidence')
    evidence = recover(submission_path)
    if evidence.get('task_fingerprints') != receipt.get('task_fingerprints'):
        raise ValueError('Server submission requests do not match the local execution intent')
    batches = batches_from_receipt(evidence)
    jobs = [job for batch in batches for job in batch.jobs]
    task_ids = [job.job_id.split(':', 1)[-1] for job in jobs]
    if len(task_ids) != len(set(task_ids)) or set(task_ids) != set(receipt['task_fingerprints']):
        raise ValueError('Server submission contains missing, duplicate, or unexpected cases')
    if receipt.get('batches'):
        if receipt['batches'] == evidence['batches']:
            return receipt  # Repeated confirmation is harmless, including after Collect.
        raise ValueError('Execution receipt already refers to a different submission')
    if receipt.get('status') != 'submission_unknown':
        raise ValueError('Only an unknown submission can be reconciled')
    state = _load_study_state(study_root / 'study-state.json', plan)
    for task_id, job in zip(task_ids, jobs):
        execution = (state['cases'].get(task_id) or {}).get('execution') or {}
        if (not execution.get('pending') or execution.get('run_id') != job.run_id
                or execution.get('receipt_path') != str(path)):
            raise ValueError('Server job does not match the pending case attempt')
    receipt.update(batches=evidence['batches'], status='submitted', last_error=None,
                   reconciled_submission_path=submission_path)
    return write_receipt(path, receipt)


def collect_study(plan_or_spec: Any, *, work_dir=None, locale='en', task_executor=None,
                  batch_independent=True, resource_profile=None) -> StudyReport:
    """Collect existing handles and rebuild a report without scheduling any task."""
    return _run_study(plan_or_spec, work_dir=work_dir, locale=locale,
                      task_executor=task_executor, collect_only=True)


@_serialized_execution
def _run_study(
    plan_or_spec: Any,
    *,
    work_dir: str = None,
    locale: str = 'en',
    postprocess: bool = False,
    postprocess_specs: List[Dict[str, Any]] = None,
    resume: bool = True,
    rerun_case_ids: Optional[List[str]] = None,
    rerun_statuses: Optional[List[str]] = None,
    max_case_attempts: int = 2,
    task_executor: Optional[TaskExecutor] = None,
    batch_independent: bool = True,
    resource_profile: Optional[str] = None,
    collect_only: bool = False,
    study_report: Optional[Dict[str, Any]] = None,
    review_kind: Optional[str] = None,
    retry_guard: Optional[Dict[str, Any]] = None,
) -> StudyReport:
    if max_case_attempts < 1:
        raise ValueError('max_case_attempts must be at least 1')
    plan = plan_or_spec if isinstance(plan_or_spec, StudyPlan) else build_study_plan(plan_or_spec)
    executor = task_executor or get_local_executor()
    cost_estimate = copy.deepcopy(plan.cost_estimate) if collect_only else require_plan_cost_approval(plan)
    root = resolve_work_dir(work_dir).resolve()
    study_work_dir = (root / plan.study_id).resolve()
    study_work_dir.relative_to(root)
    from dataclasses import replace
    from pyscf_agent.artifacts.arrays import compact_request

    if retry_guard is not None:
        from .retry import validate_execution_guard
        validate_execution_guard(study_work_dir, plan, retry_guard, rerun_case_ids)
    state_path = study_work_dir / 'study-state.json'
    # Existing inline Studies retain their submitted contract and fingerprint.
    # New Studies use references from the first submission onward.
    file_backed = not state_path.exists() or read_json(state_path).get('orbital_storage') == 'artifacts'
    if file_backed:
        plan = replace(plan, cases=[replace(case, request=compact_request(
            case.request, study_work_dir / 'arrays',
        )) for case in plan.cases])
    if collect_only and not state_path.is_file():
        raise ValueError('No existing study checkpoint to collect')
    previous_report, report_path, full_plan, initial_cases, stored_cases = _study_context(
        study_work_dir, plan, study_report, review_kind=review_kind, collect_only=collect_only,
    )
    if collect_only:
        plan = full_plan
    from computational_study_agent.grid.execution import STATE_FILE, execution_case as grid_execution_case
    grid_state_path = study_work_dir / STATE_FILE
    overrides = (read_json(grid_state_path).get('convergence_retry_overrides', {})
                 if plan.grid_refinement and grid_state_path.exists() else {})
    def resolved_case(case):
        return grid_execution_case(case, plan.grid_refinement, overrides)
    study_state = _load_study_state(state_path, plan, initial_cases=initial_cases)
    if initial_cases is not None:
        study_state['cases'].update(initial_cases)
    _adopt_legacy_receipts(plan, executor, study_state, study_work_dir)
    selected_case_ids = _normalize_case_ids(rerun_case_ids)
    selected_statuses = _normalize_statuses(rerun_statuses)
    unknown = selected_case_ids - {case.case_id for case in plan.cases}
    if unknown:
        raise ValueError('Unknown rerun case ids: {0}'.format(', '.join(sorted(unknown))))
    if selected_case_ids or selected_statuses:
        full_plan.cases = [
            StudyCase.from_dict(stored_cases[case.case_id])
            if case.case_id in stored_cases and case.case_id not in selected_case_ids
            and (study_state['cases'].get(case.case_id, {}).get('task_report') or {}).get('execution_status') not in selected_statuses
            else case
            for case in full_plan.cases
        ]
    had_pending = any((c.get('execution') or {}).get('pending') for c in study_state['cases'].values())
    execution_plan = replace(full_plan, cases=[resolved_case(case) for case in full_plan.cases])
    batch_result = _resume_pending_cases(execution_plan, executor, study_state, study_work_dir,
                                        collect_only=collect_only)
    resumed_case_ids = set(batch_result.reports)
    # An invocation recovering an interrupted run keeps already finished cases,
    # even when its caller repeated resume=False. A later explicit Run can retry.
    effective_resume = resume or had_pending
    def case_context(case):
        context = _case_execution_context(
            resolved_case(case), study_state, resume=effective_resume,
            selected_case_ids=selected_case_ids, selected_statuses=selected_statuses,
            max_case_attempts=max_case_attempts,
        )
        if collect_only or case.case_id in batch_result.reports:
            checkpoint = study_state['cases'].get(case.case_id) or {}
            if (not isinstance(checkpoint.get('task_report'), dict) or
                    checkpoint['task_report'].get('execution_status') == 'not_executed'):
                context.update(should_execute=False, execution_source='not_executed')
                return context
            if checkpoint.get('case_fingerprint') != context['fingerprint']:
                raise ValueError('Collect inputs differ from the executed plan')
            associated = (checkpoint.get('execution') or {}).get('receipt_path')
            if collect_only and associated and load_receipt(Path(associated)).get('executor') != executor_identity(executor):
                raise ValueError('Collection target differs from the submitted execution')
            context.update(should_execute=False, execution_source='resumed')
        return context
    pending_execution = had_pending or (not collect_only and any(
        case_context(case)['should_execute'] for case in plan.cases
    ))
    plan.lifecycle = copy.deepcopy(study_state.get('lifecycle') or plan.lifecycle)
    if pending_execution and not collect_only and plan.lifecycle.get('stage') == 'review_required':
        plan.lifecycle = transition_lifecycle(
            plan.lifecycle,
            'approval_granted',
            details={
                'source': 'explicit_rerun' if selected_case_ids or selected_statuses else 'automatic_retry',
            },
        )
    if pending_execution and not collect_only:
        plan.lifecycle = begin_study_execution(
            plan.lifecycle,
            entity_id=plan.study_id,
            details={'source': 'study_executor'},
        )
    study_state['lifecycle'] = copy.deepcopy(plan.lifecycle)
    if not collect_only and review_kind and not had_pending:
        study_state['review'] = {'kind': review_kind, 'case_ids': [case.case_id for case in plan.cases]}
    elif not collect_only and not had_pending:
        study_state.pop('review', None)
    if pending_execution and previous_report:
        _save_state(state_path, study_state)
        previous_report['status'] = 'completed_with_issues'
        previous_report['summary'] = 'Awaiting updated task results.'
        _write_json_atomic(report_path, previous_report, kind=report_path.stem)
    artifacts: List[Dict[str, Any]] = []
    artifacts.append(_write_json(
        study_work_dir / 'study-plan.json',
        full_plan.to_dict(),
        kind='study-plan',
        description='Expanded executable computational-study plan',
    ))
    artifacts.append(_write_json(
        study_work_dir / 'cost-estimate.json',
        cost_estimate,
        kind='cost-estimate',
        description='Dimension-based resource estimate and approval state',
    ))
    _write_json_atomic(
        state_path,
        study_state,
        kind='study-state',
        description='Atomic per-case checkpoint with study lifecycle state',
    )

    execution_counts = {
        'executed': 0,
        'resumed': 0,
        'attempt_limit_reached': 0,
    }
    if not collect_only and batch_independent and _supports_independent_batch(executor):
        contexts = [case_context(case) for case in plan.cases]
        ready = [c for c in contexts if c['should_execute'] and not c['dependency_error']
                 and not _uses_projected_1rdm(c['execution_case'])
                 and not _uses_solver_continuation(c['execution_case'])]
        if ready:
            executed = _execute_contexts(plan, executor, ready, study_work_dir, study_state,
                                         locale=locale, resource_profile=resource_profile,
                                         independent=True)
            batch_result.reports.update(executed.reports)
            batch_result.batches.extend(executed.batches)
            batch_result.artifacts.extend(executed.artifacts)
    artifacts.extend(copy.deepcopy(batch_result.artifacts))

    case_reports = []
    for case in plan.cases:
        context = case_context(case)
        execution_case = context['execution_case']
        execution_source = context['execution_source']
        if context['should_execute']:
            if context['dependency_error']:
                task_report = _blocked_dependency_report(
                    execution_case, message=context['dependency_error'],
                    work_dir=study_work_dir / 'cases', run_id=_case_run_id(case.case_id),
                )
                checkpoint = _checkpoint_case_report(
                    execution_case, task_report, case_fingerprint=context['fingerprint'],
                    attempt_count=context['previous_attempts'], execution_source='blocked',
                )
                from .dmet_branches import preserve_candidates
                preserve_candidates(study_state['cases'].get(case.case_id) or {}, checkpoint)
                study_state['cases'][case.case_id] = checkpoint
                _save_state(state_path, study_state)
                execution_source = 'blocked'
            else:
                executed = _execute_contexts(plan, executor, [context], study_work_dir, study_state,
                                             locale=locale, resource_profile=resource_profile,
                                             independent=False)
                artifacts.extend(executed.artifacts)
                execution_source = 'executed'
        elif case.case_id in batch_result.reports:
            execution_source = 'resumed' if case.case_id in resumed_case_ids else 'executed'
        if execution_source == 'not_executed':
            checkpoint = _checkpoint_case_report(
                execution_case, {'execution_status': 'not_executed', 'structured_results': {}},
                case_fingerprint=context['fingerprint'], attempt_count=0,
                execution_source=execution_source,
            )
            study_state['cases'][case.case_id] = checkpoint
        else:
            checkpoint = copy.deepcopy(study_state['cases'][case.case_id])
            checkpoint['execution_source'] = execution_source
            study_state['cases'][case.case_id] = checkpoint
        existing_receipt = (checkpoint.get('execution') or {}).get('receipt_path')
        if existing_receipt and not any(item.get('path') == existing_receipt for item in artifacts):
            artifacts.append(_execution_receipt_artifact(Path(existing_receipt)))
        task_report = checkpoint['task_report']

        execution_counts[execution_source] = execution_counts.get(execution_source, 0) + 1
        checkpoint.pop('task_reference', None)
        case_reports.append(copy.deepcopy(checkpoint))
        artifacts.extend(_write_case_observable_artifacts(
            execution_case,
            task_report,
            plan.observables,
            study_work_dir,
        ))

    # A subset execution updates tasks in one Study; the report always covers its full plan.
    case_reports, comparison_rows = _complete_task_view(
        full_plan, study_state, case_reports, previous_report, study_work_dir,
    )
    table_text = _comparison_table_text(comparison_rows)
    artifacts.append(_write_text(
        study_work_dir / 'comparison-table.tsv',
        table_text,
        kind='comparison-table',
        description='Tab-separated merged comparison table',
    ))
    dataset_manifest = _finalize_hamiltonian_dataset_study(
        full_plan,
        case_reports,
        study_work_dir,
    )
    artifacts.extend(_dataset_manifest_artifacts(dataset_manifest))
    actual_resource_profiles = sorted({
        str(batch.get('profile_id') or '').strip()
        for batch in batch_result.batches
        if isinstance(batch, dict) and str(batch.get('profile_id') or '').strip()
    })
    tasks_pending = any((case.get('execution') or {}).get('pending') for case in study_state['cases'].values())
    completed_lifecycle = (
        complete_study_execution(
            plan.lifecycle,
            entity_id=plan.study_id,
            details={'source': 'study_executor'},
        )
        if pending_execution and not tasks_pending
        else copy.deepcopy(plan.lifecycle)
    )
    report = StudyReport(
        study_id=plan.study_id,
        name=full_plan.name,
        objective=full_plan.objective,
        system_type=plan.system_type,
        status='succeeded' if all(row.get('status') == 'succeeded' for row in comparison_rows) else 'completed_with_issues',
        work_dir=str(study_work_dir),
        cases=case_reports,
        comparison_table=comparison_rows,
        artifacts=artifacts,
        summary=_study_summary(full_plan, comparison_rows),
        execution={
            'executor': executor.describe(),
            'strategy': (
                'independent_batch'
                if batch_result.reports
                else 'sequential'
            ),
            'batching': {
                'requested': bool(batch_independent),
                'supported': _supports_independent_batch(executor),
                'case_count': len(batch_result.reports),
                'batches': copy.deepcopy(batch_result.batches),
            },
            'resume_enabled': bool(resume),
            'requested_resource_profile': resource_profile or 'auto',
            'resource_profiles': actual_resource_profiles,
            'max_case_attempts': max_case_attempts,
            'counts': execution_counts,
            'checkpoint': str(state_path),
        },
        gate_configuration=copy.deepcopy(plan.gate_configuration),
        gate_provenance=copy.deepcopy(plan.gate_provenance),
        lifecycle=completed_lifecycle,
        dataset_manifest=dataset_manifest,
    )
    gate_context = evaluate_study_gates(
        {
            'study_plan': plan.to_dict(),
            'study_report': report.to_dict(),
        },
        hooks=(
            'study.after_compile',
            'study.before_execute',
            'study.after_execute',
            'study.before_finalize',
        ),
    )
    report.gate_decisions = copy.deepcopy(gate_context.get('gate_decisions') or [])
    report.gate_execution_trace = copy.deepcopy(gate_context.get('gate_execution_trace') or [])
    if not tasks_pending:
        report.lifecycle = synchronize_study_report_lifecycle(
            report.lifecycle,
            entity_id=report.study_id,
            decisions=report.gate_decisions,
            details={'source': 'study_gates'},
        )
    report.workflow = build_report_workflow_from_gate_decisions(
        report.to_dict(),
        report.gate_decisions,
    )
    review = study_state.get('review') or {}
    review_plan = full_plan.to_dict()
    review_plan['cases'] = [case for case in review_plan['cases'] if case['case_id'] in review.get('case_ids', [])]
    report = StudyReport.from_dict(update_study_report(
        previous_report, report.to_dict(), review_plan=review_plan, review_kind=review.get('kind'),
    ))
    from .dmet_branches import analyze_branches, is_dmet
    if any(is_dmet(case) for case in report.cases):
        analysis = analyze_branches(report.cases, full_plan.comparison.get('dmet_branch_policy')
                                    or (previous_report.get('dmet_phase_analysis') or {}).get('policy'))
        report = StudyReport.from_dict({**report.to_dict(), 'dmet_phase_analysis': analysis})
        report.artifacts.append(_write_json(
            study_work_dir / 'dmet-phase-analysis.json', analysis,
            kind='dmet-phase-analysis', description='DMET candidates, lowest energies, coexistence and crossing brackets',
        ))
        from .dmet_phase_plots import write_phase_plots
        phase_artifacts = write_phase_plots(analysis, study_work_dir / 'dmet-phases')
        phase_paths = {a['path'] for a in phase_artifacts}
        report.artifacts = [a for a in report.artifacts if a.get('path') not in phase_paths] + phase_artifacts
    artifacts = report.artifacts
    if postprocess:
        analysis_lifecycle = report.lifecycle
        if analysis_lifecycle.get('stage') == 'completed':
            analysis_lifecycle = transition_lifecycle(
                analysis_lifecycle,
                'analysis_started',
                details={'source': 'study_postprocessing'},
            )
        postprocess_result = run_postprocessing(
            report,
            specs=postprocess_specs,
            output_dir=str(study_work_dir / 'postprocessing'),
        )
        artifacts.extend(postprocess_result.get('artifacts') or [])
        report.summary = '{0}; postprocessing={1}'.format(
            report.summary,
            postprocess_result.get('status'),
        )
        if analysis_lifecycle.get('stage') == 'analyzing':
            analysis_lifecycle = transition_lifecycle(
                analysis_lifecycle,
                'analysis_completed',
                details={'source': 'study_postprocessing'},
            )
        report.lifecycle = analysis_lifecycle
    study_state['lifecycle'] = copy.deepcopy(report.lifecycle)
    artifacts.append(_write_json_atomic(
        state_path,
        study_state,
        kind='study-state',
        description='Atomic per-case checkpoint with study lifecycle state',
    ))
    from computational_study_agent.grid.execution import attach_grid_report
    attach_grid_report(report, full_plan)
    for document in gate_artifact_documents(report.to_dict()):
        artifacts.append(_write_json(
            study_work_dir / document['filename'],
            document['payload'],
            kind=document['kind'],
            description=document['description'],
        ))
    artifacts.append(_write_json_atomic(
        report_path,
        report.to_dict(),
        kind=report_path.stem,
        description='Merged computational-study report',
    ))
    report.artifacts = artifacts
    return report
