from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pyscf_agent.timestamps import utc_timestamp as _utc_timestamp
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..workflow_utils import path_value

from .contracts import (
    GATE_EXECUTION_TRACE_SCHEMA,
    GATE_STATUS_BLOCKED,
    GATE_STATUS_PASSED,
    GATE_STATUS_RETRY,
    GATE_STATUS_REVIEW_REQUIRED,
    GATE_STATUS_SKIPPED,
    GateDecision,
)




@dataclass(frozen=True)
class GateRuntimeIssue:
    code: str
    message: str
    gate_id: str = ''
    evaluator_id: str = ''
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GateRuntimeError(RuntimeError):
    def __init__(self, issues: Sequence[GateRuntimeIssue]):
        self.issues = tuple(issues)
        super().__init__('; '.join(issue.message for issue in self.issues))


@dataclass(frozen=True)
class GateInvocation:
    gate_set_id: str
    gate_id: str
    evaluator_id: str
    scope: str
    hook: str
    configuration: Dict[str, Any]
    required_evidence: Tuple[Dict[str, Any], ...] = ()


GateEvaluator = Callable[[Dict[str, Any], GateInvocation], GateDecision]


@dataclass(frozen=True)
class GateRuntimeAdapter:
    evaluator_id: str
    handler: GateEvaluator
    permitted_hooks: Tuple[str, ...]
    description: str = ''

    def __post_init__(self) -> None:
        if not self.evaluator_id.strip():
            raise ValueError('evaluator_id must be non-empty')
        if not callable(self.handler):
            raise TypeError('handler must be callable')
        if not self.permitted_hooks:
            raise ValueError('permitted_hooks must be non-empty')


class GateRuntimeRegistry:
    def __init__(self, adapters: Iterable[GateRuntimeAdapter] = ()):
        self._adapters: Dict[str, GateRuntimeAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: GateRuntimeAdapter) -> None:
        if adapter.evaluator_id in self._adapters:
            raise ValueError("Duplicate gate evaluator id: '{0}'".format(adapter.evaluator_id))
        self._adapters[adapter.evaluator_id] = adapter

    def resolve(self, evaluator_id: str) -> Optional[GateRuntimeAdapter]:
        return self._adapters.get(str(evaluator_id or '').strip())

    def ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def as_dict(self) -> Dict[str, Any]:
        return {
            evaluator_id: {
                'evaluator_id': adapter.evaluator_id,
                'permitted_hooks': list(adapter.permitted_hooks),
                'description': adapter.description,
            }
            for evaluator_id, adapter in sorted(self._adapters.items())
        }


def validate_gate_runtime_configuration(
    configuration: Mapping[str, Any],
    registry: GateRuntimeRegistry,
) -> Tuple[GateRuntimeIssue, ...]:
    issues: List[GateRuntimeIssue] = []
    nodes_payload = configuration.get('nodes')
    hooks_payload = configuration.get('hooks')
    order_payload = configuration.get('evaluation_order')
    if not isinstance(nodes_payload, (list, tuple)):
        return (GateRuntimeIssue('invalid_gate_configuration', 'Compiled gate nodes must be a list.'),)
    if not isinstance(hooks_payload, Mapping):
        issues.append(GateRuntimeIssue('invalid_gate_configuration', 'Compiled gate hooks must be a mapping.'))
        hooks_payload = {}
    if not isinstance(order_payload, (list, tuple)):
        issues.append(GateRuntimeIssue('invalid_gate_configuration', 'Compiled gate evaluation_order must be a list.'))
        order_payload = ()

    nodes: Dict[str, Mapping[str, Any]] = {}
    for raw_node in nodes_payload:
        if not isinstance(raw_node, Mapping):
            issues.append(GateRuntimeIssue('invalid_gate_node', 'Every compiled gate node must be a mapping.'))
            continue
        gate_id = str(raw_node.get('gate_id') or '').strip()
        evaluator_id = str(raw_node.get('evaluator_id') or '').strip()
        hook = str(raw_node.get('hook') or '').strip()
        if not gate_id:
            issues.append(GateRuntimeIssue('invalid_gate_node', 'Compiled gate node is missing gate_id.'))
            continue
        if gate_id in nodes:
            issues.append(GateRuntimeIssue(
                'duplicate_gate_node',
                "Compiled gate configuration contains duplicate gate '{0}'.".format(gate_id),
                gate_id=gate_id,
            ))
            continue
        nodes[gate_id] = raw_node
        adapter = registry.resolve(evaluator_id)
        if adapter is None:
            issues.append(GateRuntimeIssue(
                'unknown_gate_evaluator',
                "Gate '{0}' references unavailable evaluator '{1}'.".format(gate_id, evaluator_id),
                gate_id=gate_id,
                evaluator_id=evaluator_id,
            ))
        elif hook not in adapter.permitted_hooks:
            issues.append(GateRuntimeIssue(
                'gate_hook_mismatch',
                "Evaluator '{0}' cannot run gate '{1}' at hook '{2}'.".format(
                    evaluator_id, gate_id, hook
                ),
                gate_id=gate_id,
                evaluator_id=evaluator_id,
                details={'hook': hook, 'permitted_hooks': list(adapter.permitted_hooks)},
            ))

    order = [str(item) for item in order_payload]
    if len(order) != len(set(order)):
        issues.append(GateRuntimeIssue('duplicate_gate_order', 'Gate evaluation_order contains duplicates.'))
    if set(order) != set(nodes):
        issues.append(GateRuntimeIssue(
            'gate_order_mismatch',
            'Gate evaluation_order does not match the compiled nodes.',
            details={
                'missing_from_order': sorted(set(nodes).difference(order)),
                'unknown_in_order': sorted(set(order).difference(nodes)),
            },
        ))
    hooked: List[str] = []
    for hook, raw_gate_ids in hooks_payload.items():
        if not isinstance(raw_gate_ids, (list, tuple)):
            issues.append(GateRuntimeIssue(
                'invalid_gate_hook',
                "Compiled gate hook '{0}' must contain a gate list.".format(hook),
            ))
            continue
        gate_ids = [str(item) for item in raw_gate_ids]
        expected = [gate_id for gate_id in order if nodes.get(gate_id, {}).get('hook') == hook]
        if gate_ids != expected:
            issues.append(GateRuntimeIssue(
                'gate_hook_order_mismatch',
                "Compiled gate hook '{0}' does not follow evaluation_order.".format(hook),
                details={'hook_gates': gate_ids, 'expected_gates': expected},
            ))
        hooked.extend(gate_ids)
    if sorted(hooked) != sorted(nodes):
        issues.append(GateRuntimeIssue(
            'gate_hook_coverage_mismatch',
            'Compiled gate hooks must contain every gate exactly once.',
            details={
                'unhooked_gates': sorted(set(nodes).difference(hooked)),
                'duplicate_hook_gates': sorted({item for item in hooked if hooked.count(item) > 1}),
            },
        ))
    return tuple(issues)


def _missing_evidence(context: Mapping[str, Any], invocation: GateInvocation) -> List[Dict[str, Any]]:
    missing = []
    for evidence in invocation.required_evidence:
        if evidence.get('optional'):
            continue
        path = str(evidence.get('path') or '')
        if path_value(context, path) is None:
            missing.append({
                'name': evidence.get('name'),
                'path': path,
                'schema': evidence.get('schema') or '',
            })
    return missing


class GateRuntimeDispatcher:
    def __init__(self, registry: GateRuntimeRegistry):
        self.registry = registry

    def validate(self, configuration: Mapping[str, Any]) -> None:
        issues = validate_gate_runtime_configuration(configuration, self.registry)
        if issues:
            raise GateRuntimeError(issues)

    def run_hook(
        self,
        context: Dict[str, Any],
        configuration: Mapping[str, Any],
        hook: str,
    ) -> Dict[str, Any]:
        self.validate(configuration)
        nodes = {
            str(node['gate_id']): node
            for node in configuration.get('nodes', ())
            if isinstance(node, Mapping) and node.get('gate_id')
        }
        current = copy.deepcopy(context)
        for gate_id in configuration.get('hooks', {}).get(hook, ()):
            node = nodes[str(gate_id)]
            evaluator_id = str(node.get('evaluator_id') or '')
            adapter = self.registry.resolve(evaluator_id)
            if adapter is None:
                raise GateRuntimeError((GateRuntimeIssue(
                    'unknown_gate_evaluator',
                    "Gate evaluator '{0}' is unavailable.".format(evaluator_id),
                    gate_id=str(gate_id),
                    evaluator_id=evaluator_id,
                ),))
            invocation = GateInvocation(
                gate_set_id=str(configuration.get('gate_set_id') or ''),
                gate_id=str(gate_id),
                evaluator_id=evaluator_id,
                scope=str(node.get('scope') or configuration.get('scope') or ''),
                hook=hook,
                configuration=copy.deepcopy(node.get('configuration') or {}),
                required_evidence=tuple(copy.deepcopy(node.get('required_evidence') or ())),
            )
            trace = {
                'index': len(current.get('gate_execution_trace') or []) + 1,
                'gate_set_id': invocation.gate_set_id,
                'gate_id': invocation.gate_id,
                'evaluator_id': invocation.evaluator_id,
                'scope': invocation.scope,
                'hook': invocation.hook,
                'configuration': copy.deepcopy(invocation.configuration),
                'status': 'running',
                'started_at': _utc_timestamp(),
            }
            current.setdefault('gate_execution_trace', []).append(trace)
            missing = _missing_evidence(current, invocation)
            try:
                if missing:
                    decision = GateDecision(
                        gate_id=invocation.gate_id,
                        scope=invocation.scope,
                        hook=invocation.hook,
                        status=GATE_STATUS_BLOCKED,
                        summary='Required gate evidence is unavailable.',
                        checks=tuple({
                            'id': 'required_evidence',
                            'status': 'failed',
                            'message': "Missing evidence '{0}' at '{1}'.".format(
                                item.get('name'), item.get('path')
                            ),
                        } for item in missing),
                        evidence=tuple(missing),
                    )
                else:
                    decision = adapter.handler(current, invocation)
                if not isinstance(decision, GateDecision):
                    raise TypeError('Gate evaluators must return GateDecision')
            except Exception as exc:
                current['gate_execution_trace'][-1].update({
                    'status': 'failed',
                    'completed_at': _utc_timestamp(),
                    'error': '{0}: {1}'.format(type(exc).__name__, exc),
                })
                raise
            decision_payload = decision.to_dict()
            current.setdefault('gate_decisions', []).append(decision_payload)
            current['gate_execution_trace'][-1].update({
                'status': 'completed',
                'decision_status': decision.status,
                'completed_at': _utc_timestamp(),
            })
        return current


def actionable_gate_decisions(decisions: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    actionable = {
        GATE_STATUS_BLOCKED,
        GATE_STATUS_RETRY,
        GATE_STATUS_REVIEW_REQUIRED,
    }
    return [copy.deepcopy(dict(item)) for item in decisions if item.get('status') in actionable]


def aggregate_gate_status(decisions: Iterable[Mapping[str, Any]]) -> str:
    statuses = {str(item.get('status') or '') for item in decisions}
    for status in (
        GATE_STATUS_BLOCKED,
        GATE_STATUS_REVIEW_REQUIRED,
        GATE_STATUS_RETRY,
        GATE_STATUS_PASSED,
        GATE_STATUS_SKIPPED,
    ):
        if status in statuses:
            return status
    return GATE_STATUS_SKIPPED


def gate_execution_trace_payload(context: Mapping[str, Any]) -> Dict[str, Any]:
    configuration = context.get('gate_configuration')
    gate_set_id = configuration.get('gate_set_id') if isinstance(configuration, Mapping) else ''
    return {
        'schema': GATE_EXECUTION_TRACE_SCHEMA,
        'gate_set_id': str(gate_set_id or ''),
        'entries': copy.deepcopy(context.get('gate_execution_trace') or []),
        'decisions': copy.deepcopy(context.get('gate_decisions') or []),
    }


__all__ = [
    'GateInvocation',
    'GateRuntimeAdapter',
    'GateRuntimeDispatcher',
    'GateRuntimeError',
    'GateRuntimeIssue',
    'GateRuntimeRegistry',
    'actionable_gate_decisions',
    'aggregate_gate_status',
    'gate_execution_trace_payload',
    'validate_gate_runtime_configuration',
]
