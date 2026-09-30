from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from ..workflow_utils import activation_rule_matches

from .contracts import (
    GATE_COMPILER_VERSION,
    GATE_PROVENANCE_SCHEMA,
    STUDY_GATE_HOOKS,
    TASK_GATE_HOOKS,
    GateCompilationError,
    GateCompilationIssue,
    GateConfiguration,
    GateContract,
)


def _is_activated(contract: GateContract, payload: Mapping[str, Any]) -> bool:
    return bool(contract.activation_rules) and all(
        activation_rule_matches(rule, payload) for rule in contract.activation_rules
    )


def _quality_gate_request(payload: Mapping[str, Any]) -> Tuple[bool, List[str], Dict[str, Dict[str, Any]]]:
    raw = payload.get('quality_gates')
    request = raw if isinstance(raw, Mapping) else {}
    enabled = bool(request.get('enabled', True))
    raw_ids = request.get('gates') or []
    gate_ids = [
        str(item).strip()
        for item in raw_ids
        if str(item).strip()
    ] if isinstance(raw_ids, (list, tuple)) else []
    raw_config = request.get('gate_config') or {}
    gate_config = {
        str(key): copy.deepcopy(value)
        for key, value in raw_config.items()
        if isinstance(value, dict)
    } if isinstance(raw_config, Mapping) else {}
    return enabled, list(dict.fromkeys(gate_ids)), gate_config


def _task_type(payload: Mapping[str, Any], scope: str) -> str:
    key = 'system_type' if scope == 'study' else 'task_type'
    return str(payload.get(key) or 'molecular').strip().lower()


def _method(payload: Mapping[str, Any], scope: str) -> str:
    task_type = _task_type(payload, scope)
    source: Mapping[str, Any] = payload
    if scope == 'study' and isinstance(payload.get('base_task'), Mapping):
        source = payload['base_task']
    key = 'solver' if task_type == 'model_hamiltonian' else 'method'
    raw = source.get(key)
    if isinstance(raw, Mapping):
        raw = raw.get('name')
    return str(raw or '').strip().lower()


def _compatibility_issue(
    contract: GateContract,
    payload: Mapping[str, Any],
) -> Optional[GateCompilationIssue]:
    compatibility = contract.compatibility
    task_type = _task_type(payload, contract.scope)
    method = _method(payload, contract.scope)
    if compatibility.task_types and task_type not in compatibility.task_types:
        return GateCompilationIssue(
            'incompatible_gate_task_type',
            "Gate '{0}' does not support task type '{1}'.".format(contract.gate_id, task_type),
            contract.gate_id,
            {'task_type': task_type, 'supported_task_types': list(compatibility.task_types)},
        )
    if compatibility.methods and method not in compatibility.methods:
        return GateCompilationIssue(
            'incompatible_gate_method',
            "Gate '{0}' does not support method or solver '{1}'.".format(
                contract.gate_id, method or 'unspecified'
            ),
            contract.gate_id,
            {'method': method, 'supported_methods': list(compatibility.methods)},
        )
    return None


def _validate_config(
    value: Any,
    schema: Mapping[str, Any],
    path: str,
    gate_id: str,
    issues: List[GateCompilationIssue],
) -> None:
    expected = schema.get('type')
    valid = {
        'object': isinstance(value, dict),
        'array': isinstance(value, list),
        'string': isinstance(value, str),
        'integer': isinstance(value, int) and not isinstance(value, bool),
        'number': isinstance(value, (int, float)) and not isinstance(value, bool),
        'boolean': isinstance(value, bool),
    }.get(expected, True)
    if not valid:
        issues.append(GateCompilationIssue(
            'invalid_gate_configuration',
            "Configuration '{0}' for gate '{1}' must be {2}.".format(path, gate_id, expected),
            gate_id,
        ))
        return
    if 'enum' in schema and value not in schema['enum']:
        issues.append(GateCompilationIssue(
            'invalid_gate_configuration',
            "Configuration '{0}' for gate '{1}' is not an allowed value.".format(path, gate_id),
            gate_id,
            {'allowed_values': list(schema['enum'])},
        ))
    if expected == 'integer' and 'minimum' in schema and value < schema['minimum']:
        issues.append(GateCompilationIssue(
            'invalid_gate_configuration',
            "Configuration '{0}' for gate '{1}' is below the minimum.".format(path, gate_id),
            gate_id,
            {'minimum': schema['minimum']},
        ))
    if expected == 'object' and isinstance(value, dict):
        properties = schema.get('properties') or {}
        if schema.get('additionalProperties') is False:
            for key in sorted(set(value).difference(properties)):
                issues.append(GateCompilationIssue(
                    'unknown_gate_configuration',
                    "Configuration key '{0}' is not supported by gate '{1}'.".format(key, gate_id),
                    gate_id,
                ))
        for key, child_schema in properties.items():
            if key in value and isinstance(child_schema, Mapping):
                _validate_config(value[key], child_schema, path + '.' + key, gate_id, issues)


def compile_gate_configuration(
    spec: Mapping[str, Any],
    contracts: Sequence[GateContract],
    *,
    scope: str,
) -> GateConfiguration:
    if not isinstance(spec, Mapping):
        raise TypeError('spec must be a mapping')
    if scope not in ('task', 'study'):
        raise ValueError("scope must be 'task' or 'study'")
    hooks = TASK_GATE_HOOKS if scope == 'task' else STUDY_GATE_HOOKS
    issues: List[GateCompilationIssue] = []
    scoped_contracts = [contract for contract in contracts if contract.scope == scope]
    gate_ids = [contract.gate_id for contract in scoped_contracts]
    for gate_id in sorted(set(gate_ids)):
        if gate_ids.count(gate_id) > 1:
            issues.append(GateCompilationIssue(
                'duplicate_gate_id',
                "Gate registry contains duplicate gate id '{0}'.".format(gate_id),
                gate_id,
            ))
    by_id = {contract.gate_id: contract for contract in scoped_contracts}
    enabled, explicit, requested_config = _quality_gate_request(spec)
    selected: Set[str] = set()
    reasons: Dict[str, Set[str]] = {}

    def select(gate_id: str, reason: str) -> None:
        if gate_id not in by_id:
            issues.append(GateCompilationIssue(
                'unknown_gate',
                "Quality gate '{0}' is not registered for {1} workflows.".format(gate_id, scope),
                gate_id,
            ))
            return
        selected.add(gate_id)
        reasons.setdefault(gate_id, set()).add(reason)

    for contract in scoped_contracts:
        if contract.mandatory:
            select(contract.gate_id, 'mandatory')
        elif enabled and contract.always_select:
            select(contract.gate_id, 'core')
        elif enabled and _is_activated(contract, spec):
            select(contract.gate_id, 'activated')
    if enabled:
        for gate_id in explicit:
            select(gate_id, 'requested')

    effective_config: Dict[str, Dict[str, Any]] = {}
    hook_rank = {hook: index for index, hook in enumerate(hooks)}
    for gate_id in sorted(selected):
        contract = by_id[gate_id]
        if contract.default_hook not in hooks or contract.default_hook not in contract.permitted_hooks:
            issues.append(GateCompilationIssue(
                'invalid_gate_hook',
                "Gate '{0}' cannot be evaluated at hook '{1}'.".format(gate_id, contract.default_hook),
                gate_id,
                {'permitted_hooks': list(contract.permitted_hooks)},
            ))
        if not contract.evaluator_id:
            issues.append(GateCompilationIssue(
                'missing_gate_evaluator',
                "Gate '{0}' does not declare an evaluator.".format(gate_id),
                gate_id,
            ))
        compatibility_issue = _compatibility_issue(contract, spec)
        if compatibility_issue is not None:
            issues.append(compatibility_issue)
        for evidence in contract.required_evidence:
            if not evidence.name.strip() or not evidence.path.strip():
                issues.append(GateCompilationIssue(
                    'invalid_gate_evidence',
                    "Gate '{0}' declares an empty evidence contract.".format(gate_id),
                    gate_id,
                ))
        config = copy.deepcopy(contract.default_configuration)
        config.update(copy.deepcopy(requested_config.get(gate_id) or {}))
        if contract.optional_configuration:
            _validate_config(config, contract.optional_configuration, gate_id, gate_id, issues)
        elif config:
            issues.append(GateCompilationIssue(
                'unsupported_gate_configuration',
                "Gate '{0}' does not accept configuration.".format(gate_id),
                gate_id,
            ))
        effective_config[gate_id] = config

    for gate_id in sorted(set(requested_config).difference(by_id)):
        issues.append(GateCompilationIssue(
            'unknown_gate_configuration',
            "Configuration was provided for unregistered gate '{0}'.".format(gate_id),
            gate_id,
        ))
    for gate_id in sorted(set(requested_config).intersection(by_id).difference(selected)):
        issues.append(GateCompilationIssue(
            'configuration_for_unselected_gate',
            "Configuration was provided for gate '{0}', but it is not selected.".format(gate_id),
            gate_id,
        ))

    edges: Set[Tuple[str, str]] = set()
    for gate_id in sorted(selected):
        contract = by_id[gate_id]
        for conflict in contract.conflicts_with:
            if conflict in selected:
                issues.append(GateCompilationIssue(
                    'gate_conflict',
                    "Gates '{0}' and '{1}' cannot be combined.".format(gate_id, conflict),
                    gate_id,
                    {'conflicting_gate': conflict},
                ))
        for predecessor in contract.after:
            if predecessor in selected:
                edges.add((predecessor, gate_id))
        for successor in contract.before:
            if successor in selected:
                edges.add((gate_id, successor))

    for source, target in sorted(edges):
        if hook_rank[by_id[source].default_hook] > hook_rank[by_id[target].default_hook]:
            issues.append(GateCompilationIssue(
                'backward_gate_dependency',
                'Gate ordering routes a later hook into an earlier hook.',
                target,
                {'source': source, 'target': target},
            ))

    adjacency: Dict[str, Set[str]] = {gate_id: set() for gate_id in selected}
    indegree: Dict[str, int] = {gate_id: 0 for gate_id in selected}
    for source, target in edges:
        if target not in adjacency[source]:
            adjacency[source].add(target)
            indegree[target] += 1

    def sort_key(gate_id: str) -> Tuple[int, int, str]:
        contract = by_id[gate_id]
        return hook_rank[contract.default_hook], int(contract.priority), gate_id

    ready = sorted((gate_id for gate_id, degree in indegree.items() if degree == 0), key=sort_key)
    evaluation_order: List[str] = []
    while ready:
        gate_id = ready.pop(0)
        evaluation_order.append(gate_id)
        for target in sorted(adjacency[gate_id]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort(key=sort_key)
    if len(evaluation_order) != len(selected):
        issues.append(GateCompilationIssue(
            'gate_dependency_cycle',
            'Quality-gate ordering contains a cycle.',
            details={'gates': sorted(gate_id for gate_id, degree in indegree.items() if degree > 0)},
        ))

    provenance: Dict[str, Any] = {
        'schema': GATE_PROVENANCE_SCHEMA,
        'status': 'rejected' if issues else 'compiled',
        'scope': scope,
        'compiler_version': GATE_COMPILER_VERSION,
        'selected_gates': [
            {
                'gate_id': gate_id,
                'version': by_id[gate_id].version,
                'hook': by_id[gate_id].default_hook,
                'evaluator_id': by_id[gate_id].evaluator_id,
                'selection_reasons': sorted(reasons.get(gate_id, ())),
                'configuration': copy.deepcopy(effective_config.get(gate_id, {})),
            }
            for gate_id in sorted(selected)
        ],
        'issues': [issue.to_dict() for issue in issues],
    }
    if issues:
        raise GateCompilationError(tuple(issues), provenance)

    nodes = tuple({
        'gate_id': gate_id,
        'version': by_id[gate_id].version,
        'scope': scope,
        'hook': by_id[gate_id].default_hook,
        'priority': by_id[gate_id].priority,
        'evaluator_id': by_id[gate_id].evaluator_id,
        'configuration': copy.deepcopy(effective_config[gate_id]),
        'required_evidence': [item.to_dict() for item in by_id[gate_id].required_evidence],
    } for gate_id in evaluation_order)
    hook_payload = {
        hook: tuple(
            gate_id for gate_id in evaluation_order
            if by_id[gate_id].default_hook == hook
        )
        for hook in hooks
        if any(by_id[gate_id].default_hook == hook for gate_id in evaluation_order)
    }
    return GateConfiguration(
        gate_set_id='gate-set-' + uuid.uuid4().hex[:16],
        scope=scope,
        compiler_version=GATE_COMPILER_VERSION,
        nodes=nodes,
        hooks=hook_payload,
        evaluation_order=tuple(evaluation_order),
        provenance=provenance,
    )


def compile_task_gates(
    task_spec: Mapping[str, Any],
    contracts: Sequence[GateContract],
) -> GateConfiguration:
    return compile_gate_configuration(task_spec, contracts, scope='task')


def compile_study_gates(
    study_spec: Mapping[str, Any],
    contracts: Sequence[GateContract],
) -> GateConfiguration:
    return compile_gate_configuration(study_spec, contracts, scope='study')


__all__ = ['compile_gate_configuration', 'compile_study_gates', 'compile_task_gates']
