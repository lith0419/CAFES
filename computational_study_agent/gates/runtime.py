from __future__ import annotations

import copy
import functools
from typing import Any, Dict, Iterable, Mapping, Optional

from pyscf_agent.registry import default_registry
from pyscf_agent.workflow_gates import (
    STUDY_GATE_HOOKS,
    GateRuntimeAdapter,
    GateRuntimeDispatcher,
    GateRuntimeRegistry,
    compile_study_gates,
)

from .evaluators import (
    active_space_approval_gate,
    compilation_gate,
    continuation_approval_gate,
    execution_quality_gate,
    initial_scan_quality_gate,
    path_consistency_gate,
    resource_feasibility_gate,
)


def build_study_gate_runtime_registry() -> GateRuntimeRegistry:
    return GateRuntimeRegistry([
        GateRuntimeAdapter(
            'computational_study_agent.gates.compilation',
            compilation_gate,
            ('study.after_compile',),
            description='Validate compiled study modules and provenance.',
        ),
        GateRuntimeAdapter(
            'computational_study_agent.gates.active_space_approval',
            active_space_approval_gate,
            ('study.before_execute',),
            description='Require approval for CASSCF/CASCI active-space candidates.',
        ),
        GateRuntimeAdapter(
            'computational_study_agent.gates.resource_feasibility',
            resource_feasibility_gate,
            ('study.before_execute',),
            description='Evaluate dimension-based resource review policy.',
        ),
        GateRuntimeAdapter(
            'computational_study_agent.gates.initial_scan_quality',
            initial_scan_quality_gate,
            ('study.after_execute',),
            description='Reject incomplete initial diagnostics before method routing.',
        ),
        GateRuntimeAdapter(
            'computational_study_agent.gates.execution_quality',
            execution_quality_gate,
            ('study.after_execute',),
            description='Expose unresolved cases and case-specific recovery actions.',
        ),
        GateRuntimeAdapter(
            'computational_study_agent.gates.continuation_approval',
            continuation_approval_gate,
            ('study.after_review',),
            description='Require approval for projected one-particle-state reuse.',
        ),
        GateRuntimeAdapter(
            'computational_study_agent.gates.path_consistency',
            path_consistency_gate,
            ('study.after_review',),
            description='Evaluate study-level path continuity after result analysis.',
        ),
    ])


@functools.lru_cache(maxsize=1)
def get_study_gate_runtime_registry() -> GateRuntimeRegistry:
    return build_study_gate_runtime_registry()


@functools.lru_cache(maxsize=1)
def get_study_gate_runtime_dispatcher() -> GateRuntimeDispatcher:
    return GateRuntimeDispatcher(get_study_gate_runtime_registry())


def validate_study_gate_runtime(configuration: Mapping[str, Any]) -> None:
    get_study_gate_runtime_dispatcher().validate(configuration)


def _gate_configuration_from_context(context: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    direct = context.get('gate_configuration')
    if isinstance(direct, dict) and direct:
        return copy.deepcopy(direct)
    plan = context.get('study_plan')
    if isinstance(plan, dict):
        direct = plan.get('gate_configuration')
        if isinstance(direct, dict) and direct:
            return copy.deepcopy(direct)
        workflow = plan.get('workflow_configuration')
        nested = workflow.get('gate_configuration') if isinstance(workflow, dict) else None
        if isinstance(nested, dict) and nested:
            return copy.deepcopy(nested)
    report = context.get('study_report')
    if isinstance(report, dict):
        direct = report.get('gate_configuration')
        if isinstance(direct, dict) and direct:
            return copy.deepcopy(direct)
        adaptive = report.get('adaptive') if isinstance(report.get('adaptive'), dict) else {}
        for plan_kind in (
            'entanglement_active_space',
            'recovery',
            'refined',
            'path_refinement',
            'initial_scan',
        ):
            candidate = adaptive.get('{0}_plan'.format(plan_kind))
            if not isinstance(candidate, dict):
                continue
            direct = candidate.get('gate_configuration')
            if isinstance(direct, dict) and direct:
                return copy.deepcopy(direct)
            workflow = candidate.get('workflow_configuration')
            nested = workflow.get('gate_configuration') if isinstance(workflow, dict) else None
            if isinstance(nested, dict) and nested:
                return copy.deepcopy(nested)
    return None


def _fallback_gate_configuration(context: Mapping[str, Any]) -> Dict[str, Any]:
    report = context.get('study_report') if isinstance(context.get('study_report'), dict) else {}
    plan = context.get('study_plan') if isinstance(context.get('study_plan'), dict) else {}
    payload = {
        'name': report.get('name') or plan.get('name') or 'study-review',
        'objective': report.get('objective') or plan.get('objective') or 'review study',
        'system_type': report.get('system_type') or plan.get('system_type') or 'molecular',
        'study_mode': 'adaptive' if isinstance(report.get('adaptive'), dict) else 'static',
        'quality_gates': copy.deepcopy(context.get('quality_gates') or {}),
    }
    registry = default_registry()
    return compile_study_gates(payload, registry.gates_for(scope='study')).to_dict()


def evaluate_study_gates(
    context: Dict[str, Any],
    *,
    hooks: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    current = copy.deepcopy(context)
    configuration = _gate_configuration_from_context(current) or _fallback_gate_configuration(current)
    current['gate_configuration'] = copy.deepcopy(configuration)
    dispatcher = get_study_gate_runtime_dispatcher()
    selected_hooks = list(hooks or STUDY_GATE_HOOKS)
    for hook in selected_hooks:
        if hook in configuration.get('hooks', {}):
            current = dispatcher.run_hook(current, configuration, hook)
    return current


def decision_by_gate(decisions: Iterable[Mapping[str, Any]], gate_id: str) -> Optional[Dict[str, Any]]:
    for decision in decisions:
        if decision.get('gate_id') == gate_id:
            return copy.deepcopy(dict(decision))
    return None


__all__ = [
    'build_study_gate_runtime_registry',
    'decision_by_gate',
    'evaluate_study_gates',
    'get_study_gate_runtime_dispatcher',
    'get_study_gate_runtime_registry',
    'validate_study_gate_runtime',
]
