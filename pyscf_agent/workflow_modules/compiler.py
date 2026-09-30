from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from ..workflow_utils import activation_rule_matches, path_value

from .contracts import (
    CompilationIssue,
    ModuleContract,
    STUDY_STAGE_ORDER,
    STUDY_WORKFLOW_PROVENANCE_SCHEMA,
    TASK_STAGE_ORDER,
    WORKFLOW_COMPILER_VERSION,
    WORKFLOW_PROVENANCE_SCHEMA,
    StudyWorkflowConfiguration,
    WorkflowCompilationError,
    WorkflowConfiguration,
)


def _is_activated(contract: ModuleContract, task_spec: Mapping[str, Any]) -> bool:
    return bool(contract.activation_rules) and all(
        activation_rule_matches(rule, task_spec) for rule in contract.activation_rules
    )


def _task_method(task_spec: Mapping[str, Any]) -> str:
    task_type = str(task_spec.get('task_type') or 'molecular').strip().lower()
    section = 'solver' if task_type == 'model_hamiltonian' else 'method'
    value = path_value(task_spec, section + '.name')
    return str(value or '').strip().lower()


def _compatibility_issue(
    contract: ModuleContract,
    task_spec: Mapping[str, Any],
    *,
    task_type_path: str = 'task_type',
) -> Optional[CompilationIssue]:
    compatibility = contract.compatibility
    task_type = str(path_value(task_spec, task_type_path) or 'molecular').strip().lower()
    method = _task_method(task_spec)
    job = str(path_value(task_spec, 'job.name') or '').strip().lower()
    if compatibility.task_types and task_type not in compatibility.task_types:
        return CompilationIssue(
            'incompatible_task_type',
            "Module '{0}' does not support task type '{1}'.".format(contract.module_id, task_type),
            contract.module_id,
            {'supported_task_types': list(compatibility.task_types), 'task_type': task_type},
        )
    if compatibility.methods and method not in compatibility.methods:
        return CompilationIssue(
            'incompatible_method',
            "Module '{0}' does not support method or solver '{1}'.".format(contract.module_id, method or 'unspecified'),
            contract.module_id,
            {'supported_methods': list(compatibility.methods), 'method': method},
        )
    if compatibility.jobs and job not in compatibility.jobs:
        return CompilationIssue(
            'incompatible_job',
            "Module '{0}' does not support job '{1}'.".format(contract.module_id, job or 'unspecified'),
            contract.module_id,
            {'supported_jobs': list(compatibility.jobs), 'job': job},
        )
    return None


def _validate_config_value(value: Any, schema: Mapping[str, Any], path: str, issues: List[CompilationIssue], module_id: str) -> None:
    expected = schema.get('type')
    valid_type = {
        'object': isinstance(value, dict),
        'array': isinstance(value, list),
        'string': isinstance(value, str),
        'integer': isinstance(value, int) and not isinstance(value, bool),
        'number': isinstance(value, (int, float)) and not isinstance(value, bool),
        'boolean': isinstance(value, bool),
    }.get(expected, True)
    if not valid_type:
        issues.append(CompilationIssue(
            'invalid_module_configuration',
            "Configuration '{0}' for module '{1}' must be {2}.".format(path, module_id, expected),
            module_id,
        ))
        return
    if 'enum' in schema and value not in schema['enum']:
        issues.append(CompilationIssue(
            'invalid_module_configuration',
            "Configuration '{0}' for module '{1}' is not an allowed value.".format(path, module_id),
            module_id,
            {'allowed_values': list(schema['enum'])},
        ))
    if expected == 'integer' and 'minimum' in schema and value < schema['minimum']:
        issues.append(CompilationIssue(
            'invalid_module_configuration',
            "Configuration '{0}' for module '{1}' is below the minimum.".format(path, module_id),
            module_id,
            {'minimum': schema['minimum']},
        ))
    if expected == 'object' and isinstance(value, dict):
        properties = schema.get('properties') or {}
        if schema.get('additionalProperties') is False:
            for key in sorted(set(value).difference(properties)):
                issues.append(CompilationIssue(
                    'unknown_module_configuration',
                    "Configuration key '{0}' is not supported by module '{1}'.".format(key, module_id),
                    module_id,
                ))
        for key, child_schema in properties.items():
            if key in value and isinstance(child_schema, Mapping):
                _validate_config_value(value[key], child_schema, path + '.' + key, issues, module_id)


def _workflow_request(task_spec: Mapping[str, Any]) -> Tuple[List[str], Dict[str, Dict[str, Any]], str]:
    workflow = task_spec.get('workflow') if isinstance(task_spec.get('workflow'), Mapping) else {}
    raw_modules = workflow.get('modules') or []
    modules = [str(item).strip() for item in raw_modules if str(item).strip()] if isinstance(raw_modules, (list, tuple)) else []
    raw_config = workflow.get('module_config') or {}
    module_config = {
        str(key): copy.deepcopy(value)
        for key, value in raw_config.items()
        if isinstance(value, dict)
    } if isinstance(raw_config, Mapping) else {}
    dependency_policy = str(workflow.get('dependency_policy') or 'auto').strip().lower()
    return list(dict.fromkeys(modules)), module_config, dependency_policy


def _compile_workflow(
    task_spec: Mapping[str, Any],
    module_contracts: Sequence[ModuleContract],
    *,
    stage_order: Tuple[str, ...],
    root_input: str,
    task_type_path: str,
    provenance_schema: str,
    workflow_prefix: str,
    configuration_type: Any,
) -> Any:
    if not isinstance(task_spec, Mapping):
        raise TypeError('task_spec must be a mapping')
    issues: List[CompilationIssue] = []
    module_ids = [contract.module_id for contract in module_contracts]
    for module_id in sorted(set(module_ids)):
        if module_ids.count(module_id) > 1:
            issues.append(CompilationIssue(
                'duplicate_module_id',
                "Workflow registry contains duplicate module id '{0}'.".format(module_id),
                module_id,
            ))
    contracts = {contract.module_id: contract for contract in module_contracts}
    explicit, requested_config, dependency_policy = _workflow_request(task_spec)
    if dependency_policy not in ('auto', 'strict'):
        issues.append(CompilationIssue(
            'invalid_dependency_policy',
            "Workflow dependency_policy must be 'auto' or 'strict'.",
            details={'dependency_policy': dependency_policy},
        ))

    selected: Set[str] = set()
    reasons: Dict[str, Set[str]] = {}

    def select(module_id: str, reason: str) -> None:
        if module_id not in contracts:
            issues.append(CompilationIssue(
                'unknown_module',
                "Workflow module '{0}' is not registered.".format(module_id),
                module_id,
            ))
            return
        selected.add(module_id)
        reasons.setdefault(module_id, set()).add(reason)

    for contract in module_contracts:
        if contract.always_select:
            select(contract.module_id, 'core')
        elif _is_activated(contract, task_spec):
            select(contract.module_id, 'activated')
    for module_id in explicit:
        select(module_id, 'requested')

    expanded: Set[str] = set()
    changed = True
    while changed:
        changed = False
        for module_id in sorted(selected):
            if module_id in expanded:
                continue
            expanded.add(module_id)
            contract = contracts[module_id]
            for dependency in contract.requires_modules:
                if dependency not in selected:
                    if dependency_policy == 'auto':
                        select(dependency, 'dependency:' + module_id)
                        if dependency in selected:
                            changed = True
                    else:
                        issues.append(CompilationIssue(
                            'missing_module_dependency',
                            "Module '{0}' requires module '{1}'.".format(module_id, dependency),
                            module_id,
                            {'required_module': dependency},
                        ))

        selected_ports = {
            output.name
            for module_id in selected
            for output in contracts[module_id].provided_outputs
        }
        selected_ports.add('task.spec')
        for module_id in sorted(selected):
            for required in contracts[module_id].required_inputs:
                if required.optional or required.name in selected_ports or dependency_policy != 'auto':
                    continue
                candidates = [
                    candidate.module_id
                    for candidate in module_contracts
                    if candidate.module_id not in selected
                    and any(output.name == required.name for output in candidate.provided_outputs)
                    and _compatibility_issue(
                        candidate,
                        task_spec,
                        task_type_path=task_type_path,
                    ) is None
                ]
                if len(candidates) == 1:
                    select(candidates[0], 'data_dependency:' + module_id + ':' + required.name)
                    changed = True

    for module_id in sorted(selected):
        issue = _compatibility_issue(
            contracts[module_id],
            task_spec,
            task_type_path=task_type_path,
        )
        if issue is not None:
            issues.append(issue)
        stage = contracts[module_id].default_stage
        if stage not in contracts[module_id].permitted_stages or stage not in stage_order:
            issues.append(CompilationIssue(
                'invalid_module_stage',
                "Module '{0}' cannot be placed at hook '{1}'.".format(module_id, stage),
                module_id,
                {'permitted_stages': list(contracts[module_id].permitted_stages)},
            ))
        if not contracts[module_id].runtime_id:
            issues.append(CompilationIssue(
                'missing_module_runtime',
                "Module '{0}' does not declare a runtime adapter.".format(module_id),
                module_id,
            ))

    effective_config: Dict[str, Dict[str, Any]] = {}
    for module_id in sorted(selected):
        contract = contracts[module_id]
        config = copy.deepcopy(contract.default_configuration)
        config.update(copy.deepcopy(requested_config.get(module_id) or {}))
        if contract.optional_configuration:
            _validate_config_value(config, contract.optional_configuration, module_id, issues, module_id)
        elif config:
            issues.append(CompilationIssue(
                'unsupported_module_configuration',
                "Module '{0}' does not accept configuration.".format(module_id),
                module_id,
            ))
        effective_config[module_id] = config
    for module_id in sorted(set(requested_config).difference(contracts)):
        issues.append(CompilationIssue(
            'unknown_module_configuration',
            "Configuration was provided for unregistered module '{0}'.".format(module_id),
            module_id,
        ))
    for module_id in sorted(set(requested_config).intersection(contracts).difference(selected)):
        issues.append(CompilationIssue(
            'configuration_for_unselected_module',
            "Configuration was provided for module '{0}', but the module was not selected.".format(module_id),
            module_id,
        ))

    groups: Dict[str, List[str]] = {}
    for module_id in sorted(selected):
        contract = contracts[module_id]
        for conflict in contract.conflicts_with:
            if conflict in selected:
                issues.append(CompilationIssue(
                    'module_conflict',
                    "Modules '{0}' and '{1}' cannot be combined.".format(module_id, conflict),
                    module_id,
                    {'conflicting_module': conflict},
                ))
        if contract.exclusive_group:
            groups.setdefault(contract.exclusive_group, []).append(module_id)
    for group, members in sorted(groups.items()):
        if len(members) > 1:
            issues.append(CompilationIssue(
                'exclusive_module_conflict',
                "Modules in exclusive group '{0}' cannot be combined: {1}.".format(group, ', '.join(members)),
                details={'exclusive_group': group, 'modules': members},
            ))

    providers: Dict[str, List[str]] = {root_input: ['__input__']}
    for module_id in sorted(selected):
        for output in contracts[module_id].provided_outputs:
            providers.setdefault(output.name, []).append(module_id)
    for port, port_providers in sorted(providers.items()):
        real_providers = [item for item in port_providers if item != '__input__']
        policies = {
            output.merge_policy
            for module_id in real_providers
            for output in contracts[module_id].provided_outputs
            if output.name == port
        }
        if len(real_providers) > 1 and policies != {'append'} and policies != {'merge'}:
            issues.append(CompilationIssue(
                'ambiguous_output_provider',
                "Data port '{0}' has multiple single-writer providers.".format(port),
                details={'port': port, 'providers': real_providers},
            ))

    edges: Set[Tuple[str, str, str]] = set()
    for module_id in sorted(selected):
        contract = contracts[module_id]
        for required in contract.required_inputs:
            candidates = providers.get(required.name, [])
            if not candidates:
                if not required.optional:
                    issues.append(CompilationIssue(
                        'missing_input',
                        "Module '{0}' requires unavailable input '{1}'.".format(module_id, required.name),
                        module_id,
                        {'input': required.name},
                    ))
                continue
            for provider in candidates:
                if provider != '__input__':
                    edges.add((provider, module_id, required.name))
        for predecessor in contract.after:
            if predecessor in selected:
                edges.add((predecessor, module_id, 'ordering'))
        for successor in contract.before:
            if successor in selected:
                edges.add((module_id, successor, 'ordering'))

    stage_rank = {stage: index for index, stage in enumerate(stage_order)}
    for source, target, port in sorted(edges):
        if stage_rank[contracts[source].default_stage] > stage_rank[contracts[target].default_stage]:
            issues.append(CompilationIssue(
                'backward_stage_dependency',
                "Dependency '{0}' routes data from a later hook to an earlier hook.".format(port),
                target,
                {'source': source, 'target': target, 'port': port},
            ))

    adjacency: Dict[str, Set[str]] = {module_id: set() for module_id in selected}
    indegree: Dict[str, int] = {module_id: 0 for module_id in selected}
    for source, target, _port_name in edges:
        if target not in adjacency[source]:
            adjacency[source].add(target)
            indegree[target] += 1
    ready = sorted(
        (module_id for module_id, degree in indegree.items() if degree == 0),
        key=lambda item: (stage_rank[contracts[item].default_stage], item),
    )
    execution_order: List[str] = []
    while ready:
        module_id = ready.pop(0)
        execution_order.append(module_id)
        for target in sorted(adjacency[module_id]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort(key=lambda item: (stage_rank[contracts[item].default_stage], item))
    if len(execution_order) != len(selected):
        cycle_members = sorted(module_id for module_id, degree in indegree.items() if degree > 0)
        issues.append(CompilationIssue(
            'module_dependency_cycle',
            'Workflow module dependencies contain a cycle.',
            details={'modules': cycle_members},
        ))

    provenance: Dict[str, Any] = {
        'schema': provenance_schema,
        'status': 'rejected' if issues else 'compiled',
        'compiler_version': WORKFLOW_COMPILER_VERSION,
        'dependency_policy': dependency_policy,
        'selected_modules': [
            {
                'module_id': module_id,
                'version': contracts[module_id].version,
                'stage': contracts[module_id].default_stage,
                'runtime_id': contracts[module_id].runtime_id,
                'selection_reasons': sorted(reasons.get(module_id, ())),
                'configuration': copy.deepcopy(effective_config.get(module_id, {})),
            }
            for module_id in sorted(selected)
        ],
        'issues': [issue.to_dict() for issue in issues],
    }
    if issues:
        raise WorkflowCompilationError(tuple(issues), provenance)

    nodes = tuple(
        {
            'module_id': module_id,
            'version': contracts[module_id].version,
            'stage': contracts[module_id].default_stage,
            'runtime_id': contracts[module_id].runtime_id,
            'configuration': copy.deepcopy(effective_config[module_id]),
            'required_inputs': [item.to_dict() for item in contracts[module_id].required_inputs],
            'provided_outputs': [item.to_dict() for item in contracts[module_id].provided_outputs],
        }
        for module_id in execution_order
    )
    edge_payload = tuple(
        {'source': source, 'target': target, 'port': port}
        for source, target, port in sorted(edges)
    )
    hooks: Dict[str, Tuple[str, ...]] = {
        stage: tuple(module_id for module_id in execution_order if contracts[module_id].default_stage == stage)
        for stage in stage_order
        if any(contracts[module_id].default_stage == stage for module_id in execution_order)
    }
    expected_outputs = tuple(sorted({
        output.name
        for module_id in selected
        for output in contracts[module_id].provided_outputs
    }))
    return configuration_type(
        workflow_id=workflow_prefix + uuid.uuid4().hex[:16],
        compiler_version=WORKFLOW_COMPILER_VERSION,
        dependency_policy=dependency_policy,
        nodes=nodes,
        edges=edge_payload,
        hooks=hooks,
        execution_order=tuple(execution_order),
        expected_outputs=expected_outputs,
        provenance=provenance,
    )


def compile_task_workflow(
    task_spec: Mapping[str, Any],
    module_contracts: Sequence[ModuleContract],
) -> WorkflowConfiguration:
    return _compile_workflow(
        task_spec,
        module_contracts,
        stage_order=TASK_STAGE_ORDER,
        root_input='task.spec',
        task_type_path='task_type',
        provenance_schema=WORKFLOW_PROVENANCE_SCHEMA,
        workflow_prefix='workflow-',
        configuration_type=WorkflowConfiguration,
    )


def compile_study_workflow(
    study_spec: Mapping[str, Any],
    module_contracts: Sequence[ModuleContract],
    *,
    study_mode: Optional[str] = None,
) -> StudyWorkflowConfiguration:
    if not isinstance(study_spec, Mapping):
        raise TypeError('study_spec must be a mapping')
    payload = copy.deepcopy(dict(study_spec))
    mode = str(study_mode or payload.get('study_mode') or 'static').strip().lower()
    if mode == 'adaptive_scan':
        mode = 'adaptive'
    if mode not in ('static', 'adaptive'):
        raise ValueError("study_mode must be 'static' or 'adaptive'")
    payload['study_mode'] = mode
    payload['workflow'] = copy.deepcopy(payload.get('study_workflow') or {})
    return _compile_workflow(
        payload,
        module_contracts,
        stage_order=STUDY_STAGE_ORDER,
        root_input='study.spec',
        task_type_path='system_type',
        provenance_schema=STUDY_WORKFLOW_PROVENANCE_SCHEMA,
        workflow_prefix='study-workflow-',
        configuration_type=StudyWorkflowConfiguration,
    )
