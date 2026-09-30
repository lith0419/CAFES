from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pyscf_agent.timestamps import utc_timestamp as _utc_timestamp
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


WORKFLOW_EXECUTION_TRACE_SCHEMA = 'pyscf-agent.workflow-execution-trace.v1'




@dataclass(frozen=True)
class RuntimeIssue:
    code: str
    message: str
    module_id: str = ''
    runtime_id: str = ''
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WorkflowRuntimeError(RuntimeError):
    def __init__(
        self,
        issues: Sequence[RuntimeIssue],
        *,
        state: Optional[Dict[str, Any]] = None,
    ):
        self.issues = tuple(issues)
        self.state = copy.deepcopy(state) if isinstance(state, dict) else None
        super().__init__('; '.join(issue.message for issue in self.issues))


@dataclass(frozen=True)
class ModuleInvocation:
    workflow_id: str
    module_id: str
    runtime_id: str
    stage: str
    configuration: Dict[str, Any]
    required_inputs: Tuple[Dict[str, Any], ...] = ()
    provided_outputs: Tuple[Dict[str, Any], ...] = ()


ModuleRuntimeHandler = Callable[[Dict[str, Any], ModuleInvocation], Dict[str, Any]]


def record_module_observation(
    state: Dict[str, Any],
    invocation: ModuleInvocation,
    status: str,
    *,
    provided_result_paths: Sequence[str] = (),
    **details: Any,
) -> Dict[str, Any]:
    """Append one uniform module observation to workflow state."""

    observation = {
        'module_id': invocation.module_id,
        'runtime_id': invocation.runtime_id,
        'status': str(status),
        'configuration': copy.deepcopy(invocation.configuration),
        'execution_status': state.get('execution_status'),
    }
    if provided_result_paths:
        observation['provided_result_paths'] = [
            str(path) for path in provided_result_paths
        ]
    observation.update(copy.deepcopy(details))
    state.setdefault('module_runtime_observations', []).append(observation)
    return observation


@dataclass(frozen=True)
class ModuleRuntimeAdapter:
    runtime_id: str
    handler: ModuleRuntimeHandler
    permitted_stages: Tuple[str, ...]
    mode: str = 'handler'
    description: str = ''

    def __post_init__(self) -> None:
        if not self.runtime_id.strip():
            raise ValueError('runtime_id must be non-empty')
        if not callable(self.handler):
            raise TypeError('handler must be callable')
        if not self.permitted_stages:
            raise ValueError('permitted_stages must be non-empty')
        if self.mode not in ('handler', 'integrated'):
            raise ValueError("mode must be 'handler' or 'integrated'")


class ModuleRuntimeRegistry:
    """Explicit runtime implementations available to compiled workflows."""

    def __init__(self, adapters: Iterable[ModuleRuntimeAdapter] = ()):
        self._adapters: Dict[str, ModuleRuntimeAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: ModuleRuntimeAdapter) -> None:
        if adapter.runtime_id in self._adapters:
            raise ValueError("Duplicate module runtime id: '{0}'".format(adapter.runtime_id))
        self._adapters[adapter.runtime_id] = adapter

    def resolve(self, runtime_id: str) -> Optional[ModuleRuntimeAdapter]:
        return self._adapters.get(str(runtime_id or '').strip())

    def ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def as_dict(self) -> Dict[str, Any]:
        return {
            runtime_id: {
                'runtime_id': adapter.runtime_id,
                'permitted_stages': list(adapter.permitted_stages),
                'mode': adapter.mode,
                'description': adapter.description,
            }
            for runtime_id, adapter in sorted(self._adapters.items())
        }


def validate_runtime_configuration(
    configuration: Mapping[str, Any],
    registry: ModuleRuntimeRegistry,
) -> Tuple[RuntimeIssue, ...]:
    issues: List[RuntimeIssue] = []
    raw_nodes = configuration.get('nodes')
    raw_hooks = configuration.get('hooks')
    raw_order = configuration.get('execution_order')
    if not isinstance(raw_nodes, (list, tuple)):
        return (RuntimeIssue('invalid_runtime_configuration', 'Compiled workflow nodes must be a list.'),)
    if not isinstance(raw_hooks, Mapping):
        issues.append(RuntimeIssue('invalid_runtime_configuration', 'Compiled workflow hooks must be a mapping.'))
        raw_hooks = {}
    if not isinstance(raw_order, (list, tuple)):
        issues.append(RuntimeIssue('invalid_runtime_configuration', 'Compiled workflow execution_order must be a list.'))
        raw_order = ()

    nodes: Dict[str, Mapping[str, Any]] = {}
    for raw_node in raw_nodes:
        if not isinstance(raw_node, Mapping):
            issues.append(RuntimeIssue('invalid_runtime_node', 'Every compiled workflow node must be a mapping.'))
            continue
        module_id = str(raw_node.get('module_id') or '').strip()
        runtime_id = str(raw_node.get('runtime_id') or '').strip()
        stage = str(raw_node.get('stage') or '').strip()
        if not module_id:
            issues.append(RuntimeIssue('invalid_runtime_node', 'Compiled workflow node is missing module_id.'))
            continue
        if module_id in nodes:
            issues.append(RuntimeIssue(
                'duplicate_runtime_module',
                "Compiled workflow contains duplicate module '{0}'.".format(module_id),
                module_id=module_id,
            ))
            continue
        nodes[module_id] = raw_node
        adapter = registry.resolve(runtime_id)
        if adapter is None:
            issues.append(RuntimeIssue(
                'unknown_runtime_adapter',
                "Module '{0}' references unavailable runtime adapter '{1}'.".format(module_id, runtime_id),
                module_id=module_id,
                runtime_id=runtime_id,
            ))
        elif stage not in adapter.permitted_stages:
            issues.append(RuntimeIssue(
                'runtime_stage_mismatch',
                "Runtime adapter '{0}' cannot execute module '{1}' at hook '{2}'.".format(
                    runtime_id, module_id, stage
                ),
                module_id=module_id,
                runtime_id=runtime_id,
                details={'stage': stage, 'permitted_stages': list(adapter.permitted_stages)},
            ))

    order = [str(item) for item in raw_order]
    if len(order) != len(set(order)):
        issues.append(RuntimeIssue('duplicate_runtime_order', 'Compiled workflow execution_order contains duplicates.'))
    missing_from_order = sorted(set(nodes).difference(order))
    unknown_in_order = sorted(set(order).difference(nodes))
    if missing_from_order or unknown_in_order:
        issues.append(RuntimeIssue(
            'runtime_order_mismatch',
            'Compiled workflow execution_order does not match its nodes.',
            details={
                'missing_from_order': missing_from_order,
                'unknown_in_order': unknown_in_order,
            },
        ))

    hooked: List[str] = []
    for stage, raw_module_ids in raw_hooks.items():
        if not isinstance(raw_module_ids, (list, tuple)):
            issues.append(RuntimeIssue(
                'invalid_runtime_hook',
                "Compiled workflow hook '{0}' must contain a module list.".format(stage),
            ))
            continue
        module_ids = [str(item) for item in raw_module_ids]
        expected = [module_id for module_id in order if nodes.get(module_id, {}).get('stage') == stage]
        if module_ids != expected:
            issues.append(RuntimeIssue(
                'runtime_hook_order_mismatch',
                "Compiled workflow hook '{0}' does not follow execution_order.".format(stage),
                details={'hook_modules': module_ids, 'expected_modules': expected},
            ))
        for module_id in module_ids:
            hooked.append(module_id)
            node = nodes.get(module_id)
            if node is None:
                issues.append(RuntimeIssue(
                    'unknown_runtime_hook_module',
                    "Compiled workflow hook '{0}' references unknown module '{1}'.".format(stage, module_id),
                    module_id=module_id,
                ))
            elif str(node.get('stage') or '') != str(stage):
                issues.append(RuntimeIssue(
                    'runtime_hook_stage_mismatch',
                    "Module '{0}' is assigned to a different hook than its compiled stage.".format(module_id),
                    module_id=module_id,
                    details={'hook': stage, 'compiled_stage': node.get('stage')},
                ))
    if sorted(hooked) != sorted(nodes):
        issues.append(RuntimeIssue(
            'runtime_hook_coverage_mismatch',
            'Compiled workflow hooks must contain every node exactly once.',
            details={
                'unhooked_modules': sorted(set(nodes).difference(hooked)),
                'duplicate_hook_modules': sorted({item for item in hooked if hooked.count(item) > 1}),
            },
        ))
    return tuple(issues)


class WorkflowRuntimeDispatcher:
    """Execute compiled modules through registered adapters, one hook at a time."""

    def __init__(self, registry: ModuleRuntimeRegistry):
        self.registry = registry

    def validate(self, configuration: Mapping[str, Any]) -> None:
        issues = validate_runtime_configuration(configuration, self.registry)
        if issues:
            raise WorkflowRuntimeError(issues)

    def run_stage(self, state: Dict[str, Any], stage: str) -> Dict[str, Any]:
        configuration = state.get('workflow_configuration')
        if not isinstance(configuration, Mapping) or not configuration:
            raise WorkflowRuntimeError((RuntimeIssue(
                'missing_runtime_configuration',
                'A compiled workflow configuration is required before module execution.',
                details={'stage': stage},
            ),))
        self.validate(configuration)
        nodes = {
            str(node['module_id']): node
            for node in configuration.get('nodes', ())
            if isinstance(node, Mapping) and node.get('module_id')
        }
        module_ids = configuration.get('hooks', {}).get(stage, ())
        current = state
        for module_id in module_ids:
            node = nodes[str(module_id)]
            runtime_id = str(node.get('runtime_id') or '')
            adapter = self.registry.resolve(runtime_id)
            if adapter is None:  # Defensive; validate() reports this first.
                raise WorkflowRuntimeError((RuntimeIssue(
                    'unknown_runtime_adapter',
                    "Runtime adapter '{0}' is unavailable.".format(runtime_id),
                    module_id=str(module_id),
                    runtime_id=runtime_id,
                ),))
            invocation = ModuleInvocation(
                workflow_id=str(configuration.get('workflow_id') or ''),
                module_id=str(module_id),
                runtime_id=runtime_id,
                stage=stage,
                configuration=copy.deepcopy(node.get('configuration') or {}),
                required_inputs=tuple(copy.deepcopy(node.get('required_inputs') or ())),
                provided_outputs=tuple(copy.deepcopy(node.get('provided_outputs') or ())),
            )
            trace_entry = {
                'index': len(current.get('module_execution_trace') or []) + 1,
                'workflow_id': invocation.workflow_id,
                'module_id': invocation.module_id,
                'runtime_id': invocation.runtime_id,
                'stage': invocation.stage,
                'mode': adapter.mode,
                'configuration': copy.deepcopy(invocation.configuration),
                'retry_count': int(current.get('retry_count') or 0),
                'status': 'running',
                'started_at': _utc_timestamp(),
            }
            current = copy.deepcopy(current)
            current.setdefault('module_execution_trace', []).append(trace_entry)
            try:
                next_state = adapter.handler(current, invocation)
                if not isinstance(next_state, dict):
                    raise TypeError('Module runtime handlers must return a workflow-state dictionary')
            except Exception as exc:
                failure_state = (
                    copy.deepcopy(exc.state)
                    if isinstance(exc, WorkflowRuntimeError) and isinstance(exc.state, dict)
                    else current
                )
                failure_state.setdefault('module_execution_trace', [])[-1].update({
                    'status': 'failed',
                    'completed_at': _utc_timestamp(),
                    'error': '{0}: {1}'.format(type(exc).__name__, exc),
                })
                if isinstance(exc, WorkflowRuntimeError):
                    issues = tuple(
                        issue if issue.module_id and issue.runtime_id else RuntimeIssue(
                            issue.code,
                            issue.message,
                            module_id=issue.module_id or invocation.module_id,
                            runtime_id=issue.runtime_id or invocation.runtime_id,
                            details=copy.deepcopy(issue.details),
                        )
                        for issue in exc.issues
                    )
                else:
                    issues = (RuntimeIssue(
                        'runtime_adapter_exception',
                        "Workflow module '{0}' failed: {1}".format(invocation.module_id, exc),
                        module_id=invocation.module_id,
                        runtime_id=invocation.runtime_id,
                        details={
                            'exception_type': type(exc).__name__,
                            'stage': stage,
                            'configuration': copy.deepcopy(invocation.configuration),
                        },
                    ),)
                raise WorkflowRuntimeError(issues, state=failure_state) from exc
            current = next_state
            current.setdefault('module_execution_trace', [])[-1].update({
                'status': 'observed' if adapter.mode == 'integrated' else 'completed',
                'completed_at': _utc_timestamp(),
                'provided_outputs': [
                    str(item.get('name') or '')
                    for item in invocation.provided_outputs
                    if isinstance(item, Mapping) and item.get('name')
                ],
            })
        return current


def execution_trace_payload(state: Mapping[str, Any]) -> Dict[str, Any]:
    configuration = state.get('workflow_configuration')
    workflow_id = configuration.get('workflow_id') if isinstance(configuration, Mapping) else ''
    return {
        'schema': WORKFLOW_EXECUTION_TRACE_SCHEMA,
        'workflow_id': str(workflow_id or ''),
        'run_id': str(state.get('run_id') or ''),
        'entries': copy.deepcopy(state.get('module_execution_trace') or []),
    }


__all__ = [
    'ModuleInvocation',
    'ModuleRuntimeAdapter',
    'ModuleRuntimeRegistry',
    'RuntimeIssue',
    'WORKFLOW_EXECUTION_TRACE_SCHEMA',
    'WorkflowRuntimeDispatcher',
    'WorkflowRuntimeError',
    'execution_trace_payload',
    'record_module_observation',
    'validate_runtime_configuration',
]
