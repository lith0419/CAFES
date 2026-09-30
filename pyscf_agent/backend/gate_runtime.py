from __future__ import annotations

import copy
import functools
from typing import Any, Dict, Mapping

from .artifacts import write_result_json_artifact
from .state import append_log
from ..result_quality import evaluate_quality_checks
from ..workflow_gates import (
    GATE_STATUS_BLOCKED,
    GATE_STATUS_PASSED,
    GATE_STATUS_RETRY,
    GATE_STATUS_REVIEW_REQUIRED,
    GateDecision,
    GateInvocation,
    GateRuntimeAdapter,
    GateRuntimeDispatcher,
    GateRuntimeError,
    GateRuntimeRegistry,
    gate_execution_trace_payload,
    validate_gate_runtime_configuration,
)


def _decision(
    invocation: GateInvocation,
    status: str,
    summary: str,
    *,
    checks: Any = (),
    actions: Any = (),
    evidence: Any = (),
    details: Any = None,
) -> GateDecision:
    return GateDecision(
        gate_id=invocation.gate_id,
        scope=invocation.scope,
        hook=invocation.hook,
        status=status,
        summary=summary,
        checks=tuple(copy.deepcopy(checks or ())),
        recommended_actions=tuple(copy.deepcopy(actions or ())),
        evidence=tuple(copy.deepcopy(evidence or ())),
        details=copy.deepcopy(details or {}),
        provenance={
            'gate_set_id': invocation.gate_set_id,
            'evaluator_id': invocation.evaluator_id,
            'configuration': copy.deepcopy(invocation.configuration),
        },
    )


def _compilation_gate(state: Dict[str, Any], invocation: GateInvocation) -> GateDecision:
    configuration = state.get('workflow_configuration')
    provenance = state.get('workflow_provenance')
    compiled = bool(
        isinstance(configuration, dict)
        and configuration.get('workflow_id')
        and isinstance(provenance, dict)
        and provenance.get('status') == 'compiled'
    )
    return _decision(
        invocation,
        GATE_STATUS_PASSED if compiled else GATE_STATUS_BLOCKED,
        'Task workflow compilation is complete.' if compiled else 'Task workflow compilation is incomplete or rejected.',
        checks=({
            'id': 'module_compilation',
            'status': 'passed' if compiled else 'failed',
            'message': 'The module graph and provenance are available.' if compiled else 'A compiled module graph is required.',
        },),
        evidence=({
            'workflow_id': configuration.get('workflow_id') if isinstance(configuration, dict) else None,
            'provenance_status': provenance.get('status') if isinstance(provenance, dict) else None,
        },),
    )


def _input_readiness_gate(state: Dict[str, Any], invocation: GateInvocation) -> GateDecision:
    validation_errors = [str(item) for item in state.get('validation_errors') or [] if str(item).strip()]
    has_spec = isinstance(state.get('task_spec'), dict) and bool(state.get('task_spec'))
    has_input = bool(state.get('generated_input'))
    blocked = bool(validation_errors or not has_spec or not has_input)
    checks = (
        {
            'id': 'task_spec',
            'status': 'passed' if has_spec else 'failed',
            'message': 'Validated TaskSpec is available.' if has_spec else 'TaskSpec is missing.',
        },
        {
            'id': 'generated_input',
            'status': 'passed' if has_input else 'failed',
            'message': 'Reproducible input is available.' if has_input else 'Generated input is missing.',
        },
        {
            'id': 'validation_errors',
            'status': 'failed' if validation_errors else 'passed',
            'message': '; '.join(validation_errors) if validation_errors else 'No blocking validation errors were reported.',
        },
    )
    return _decision(
        invocation,
        GATE_STATUS_BLOCKED if blocked else GATE_STATUS_PASSED,
        'Task input requires correction before execution.' if blocked else 'Task input is ready for execution.',
        checks=checks,
        actions=({'id': 'fix_task_input', 'label': 'Fix Task Input'},) if blocked else (),
        evidence=({'validation_errors': validation_errors},),
    )


def _block2_availability_gate(state: Dict[str, Any], invocation: GateInvocation) -> GateDecision:
    from ..providers.block2.availability import block2_availability  # pylint: disable=import-outside-toplevel

    provider = state.get('solver_provider')
    availability = provider if isinstance(provider, dict) else block2_availability()
    available = bool(availability.get('available'))
    reason = str(availability.get('reason') or '')
    return _decision(
        invocation,
        GATE_STATUS_PASSED if available else GATE_STATUS_BLOCKED,
        'block2 DMRG provider is available.' if available else 'block2 DMRG provider is unavailable.',
        checks=({
            'id': 'block2_provider',
            'status': 'passed' if available else 'failed',
            'message': 'Exactly one usable block2 distribution is installed.' if available else reason,
        },),
        actions=() if available else ({'id': 'install_block2', 'label': 'Install block2'},),
        evidence=({
            'distribution': availability.get('distribution'),
            'version': availability.get('version'),
            'mpi_enabled': availability.get('mpi_enabled'),
            'distribution_conflict': availability.get('distribution_conflict'),
        },),
    )
def _execution_quality_gate(state: Dict[str, Any], invocation: GateInvocation) -> GateDecision:
    status = str(state.get('execution_status') or 'pending').strip().lower()
    retry_count = int(state.get('retry_count') or 0)
    max_retries = int(state.get('max_retries') or 0)
    structured_results = (
        state.get('structured_results')
        if isinstance(state.get('structured_results'), Mapping)
        else {}
    )
    quality = evaluate_quality_checks(structured_results.get('quality_checks'))
    if status == 'succeeded':
        if quality['quality_status'] == 'blocked':
            gate_status = GATE_STATUS_BLOCKED
            summary = 'Task execution completed, but required result constraints failed or lack valid evidence.'
            actions = ({'id': 'inspect_result_quality', 'label': 'Inspect Result Quality'},)
        elif quality['quality_status'] == 'review_required':
            gate_status = GATE_STATUS_REVIEW_REQUIRED
            summary = 'Task execution completed, but required numerical convergence checks failed.'
            actions = ({'id': 'review_method', 'label': 'Review Method'},)
        else:
            gate_status = GATE_STATUS_PASSED
            summary = (
                'Task execution completed successfully; this legacy result does not expose structured quality checks.'
                if quality['quality_status'] == 'legacy'
                else 'Task execution and required result-quality checks completed successfully.'
            )
            actions = ()
    elif status == 'pending' and retry_count < max_retries:
        gate_status = GATE_STATUS_RETRY
        summary = 'Task execution is ready for another bounded retry.'
        actions = ({'id': 'retry_task', 'label': 'Retry Task'},)
    elif status == 'unconverged':
        gate_status = GATE_STATUS_REVIEW_REQUIRED
        summary = 'Task remains unconverged after automatic recovery.'
        actions = (
            {'id': 'increase_max_cycle', 'label': 'Increase max_cycle'},
            {'id': 'review_method', 'label': 'Review Method'},
        )
    else:
        gate_status = GATE_STATUS_BLOCKED
        summary = 'Task execution did not produce a usable result.'
        actions = ({'id': 'inspect_task_failure', 'label': 'Inspect Failure'},)
    publication_eligible = quality.get('publication_eligible') if status == 'succeeded' else False
    quality_checks = tuple({
        **copy.deepcopy(check),
        'status': (
            'failed' if check.get('status') == 'invalid' else check.get('status')
        ),
        'message': (
            'Invalid quality evidence: {0}'.format(check.get('message'))
            if check.get('status') == 'invalid'
            else check.get('message')
        ),
    } for check in quality.get('checks') or [])
    return _decision(
        invocation,
        gate_status,
        summary,
        checks=(
            {
                'id': 'execution_status',
                'status': 'passed' if status == 'succeeded' else 'failed',
                'message': 'execution_status={0}'.format(status),
            },
            *quality_checks,
        ),
        actions=actions,
        evidence=({
            'execution_status': status,
            'retry_count': retry_count,
            'max_retries': max_retries,
            'errors': copy.deepcopy(state.get('errors') or []),
            'quality': copy.deepcopy(quality),
        },),
        details={
            'quality_status': quality.get('quality_status'),
            'quality_evidence_status': quality.get('evidence_status'),
            'publication_eligible': publication_eligible,
            'legacy_quality_evidence': quality.get('evidence_status') == 'legacy_unavailable',
        },
    )


def _artifact_integrity_gate(state: Dict[str, Any], invocation: GateInvocation) -> GateDecision:
    execution_status = str(state.get('execution_status') or '').strip().lower()
    results = state.get('structured_results')
    artifacts = state.get('artifacts') if isinstance(state.get('artifacts'), list) else []
    malformed = [
        index for index, artifact in enumerate(artifacts)
        if not isinstance(artifact, dict) or not artifact.get('kind') or not artifact.get('path')
    ]
    paths = [str(item.get('path')) for item in artifacts if isinstance(item, dict) and item.get('path')]
    duplicate_paths = sorted({path for path in paths if paths.count(path) > 1})
    missing_results = execution_status == 'succeeded' and not isinstance(results, dict)
    if missing_results:
        status = GATE_STATUS_BLOCKED
        summary = 'A successful execution is missing structured numerical results.'
    elif malformed or duplicate_paths:
        status = GATE_STATUS_REVIEW_REQUIRED
        summary = 'Result artifacts contain incomplete or duplicate references.'
    else:
        status = GATE_STATUS_PASSED
        summary = 'Structured results and artifact references are internally consistent.'
    return _decision(
        invocation,
        status,
        summary,
        checks=(
            {
                'id': 'structured_results',
                'status': 'failed' if missing_results else 'passed',
                'message': 'Structured results are present.' if not missing_results else 'Structured results are missing.',
            },
            {
                'id': 'artifact_references',
                'status': 'failed' if malformed or duplicate_paths else 'passed',
                'message': 'Artifact references are well formed.' if not malformed and not duplicate_paths else 'Artifact references need review.',
            },
        ),
        actions=({'id': 'inspect_artifacts', 'label': 'Inspect Artifacts'},) if status != GATE_STATUS_PASSED else (),
        evidence=({
            'artifact_count': len(artifacts),
            'malformed_indices': malformed,
            'duplicate_paths': duplicate_paths,
        },),
    )


def build_default_gate_runtime_registry() -> GateRuntimeRegistry:
    return GateRuntimeRegistry([
        GateRuntimeAdapter(
            'pyscf_agent.gates.compilation',
            _compilation_gate,
            ('task.after_compile',),
            description='Validate compiled task modules and provenance.',
        ),
        GateRuntimeAdapter(
            'pyscf_agent.gates.input_readiness',
            _input_readiness_gate,
            ('task.before_execute',),
            description='Validate executable TaskSpec and generated input.',
        ),
        GateRuntimeAdapter(
            'pyscf_agent.gates.block2_availability',
            _block2_availability_gate,
            ('task.before_execute',),
            description='Validate the optional block2 DMRG provider before numerical execution.',
        ),
        GateRuntimeAdapter(
            'pyscf_agent.gates.execution_quality',
            _execution_quality_gate,
            ('task.after_recovery',),
            description='Classify final task execution quality.',
        ),
        GateRuntimeAdapter(
            'pyscf_agent.gates.artifact_integrity',
            _artifact_integrity_gate,
            ('task.before_finalize',),
            description='Validate structured results and artifact references.',
        ),
    ])


@functools.lru_cache(maxsize=1)
def get_default_gate_runtime_registry() -> GateRuntimeRegistry:
    return build_default_gate_runtime_registry()


@functools.lru_cache(maxsize=1)
def get_default_gate_runtime_dispatcher() -> GateRuntimeDispatcher:
    return GateRuntimeDispatcher(get_default_gate_runtime_registry())


def validate_compiled_gates(configuration: Mapping[str, Any]) -> None:
    issues = validate_gate_runtime_configuration(configuration, get_default_gate_runtime_registry())
    if issues:
        raise GateRuntimeError(issues)


def run_task_gate_hook(state: Dict[str, Any], hook: str) -> Dict[str, Any]:
    configuration = state.get('gate_configuration')
    if not isinstance(configuration, Mapping) or not configuration:
        return state
    try:
        result = get_default_gate_runtime_dispatcher().run_hook(state, configuration, hook)
    except GateRuntimeError as exc:
        result = copy.deepcopy(state)
        for issue in exc.issues:
            result.setdefault('errors', []).append({
                'stage': 'gate_runtime',
                'code': issue.code,
                'message': issue.message,
                'details': copy.deepcopy(issue.details),
            })
        append_log(result, 'error', 'workflow.gate_runtime_failed', {
            'hook': hook,
            'issues': [issue.to_dict() for issue in exc.issues],
        })
        return result
    decisions = [
        item for item in result.get('gate_decisions') or []
        if isinstance(item, dict) and item.get('hook') == hook
    ]
    blocked = [item for item in decisions if item.get('status') == GATE_STATUS_BLOCKED]
    if blocked and hook in ('task.after_compile', 'task.before_execute'):
        result['execution_status'] = 'blocked'
        for decision in blocked:
            message = str(decision.get('summary') or 'A quality gate blocked execution.')
            if message not in result.setdefault('validation_errors', []):
                result['validation_errors'].append(message)
    append_log(result, 'info', 'workflow.gate_hook_completed', {
        'hook': hook,
        'decisions': [
            {'gate_id': item.get('gate_id'), 'status': item.get('status')}
            for item in decisions
        ],
    })
    return result


def compiled_gate_after_compile(state: Dict[str, Any]) -> Dict[str, Any]:
    return run_task_gate_hook(state, 'task.after_compile')


def compiled_gate_before_execute(state: Dict[str, Any]) -> Dict[str, Any]:
    return run_task_gate_hook(state, 'task.before_execute')


def compiled_gate_after_recovery(state: Dict[str, Any]) -> Dict[str, Any]:
    return run_task_gate_hook(state, 'task.after_recovery')


def compiled_gate_before_finalize(state: Dict[str, Any]) -> Dict[str, Any]:
    result = run_task_gate_hook(state, 'task.before_finalize')
    if isinstance(result.get('gate_configuration'), Mapping):
        write_result_json_artifact(
            result,
            'gate-execution-trace',
            'gate-execution-trace.json',
            gate_execution_trace_payload(result),
            description='Deterministic decisions emitted by compiled task quality gates.',
        )
    return result


__all__ = [
    'build_default_gate_runtime_registry',
    'compiled_gate_after_compile',
    'compiled_gate_after_recovery',
    'compiled_gate_before_execute',
    'compiled_gate_before_finalize',
    'get_default_gate_runtime_dispatcher',
    'get_default_gate_runtime_registry',
    'run_task_gate_hook',
    'validate_compiled_gates',
]
