from __future__ import annotations

import copy
from typing import Any, Dict

from .artifacts import write_json_artifact
from .gate_runtime import validate_compiled_gates
from .module_runtime import validate_compiled_runtime
from .state import append_log
from ..registry import default_registry
from ..workflow_modules import WorkflowCompilationError, WorkflowRuntimeError, compile_task_workflow
from ..workflow_gates import GateCompilationError, GateRuntimeError, compile_task_gates


def module_compiler(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compile the validated TaskSpec into a deterministic module graph."""

    state = copy.deepcopy(state)
    registry = default_registry()
    try:
        configuration = compile_task_workflow(
            state.get('task_spec') or {},
            registry.modules_for_template('task.template.default'),
        )
        validate_compiled_runtime(configuration.to_dict())
        gate_configuration = compile_task_gates(
            state.get('task_spec') or {},
            registry.gates_for(scope='task'),
        )
        validate_compiled_gates(gate_configuration.to_dict())
    except WorkflowCompilationError as exc:
        state['workflow_configuration'] = {}
        state['workflow_provenance'] = copy.deepcopy(exc.provenance)
        for issue in exc.issues:
            state.setdefault('errors', []).append({
                'stage': 'workflow_compilation',
                'code': issue.code,
                'message': issue.message,
                'details': copy.deepcopy(issue.details),
            })
            state.setdefault('validation_errors', []).append(issue.message)
        write_json_artifact(
            state,
            'workflow-provenance',
            'workflow-provenance.json',
            state['workflow_provenance'],
            description='Rejected module selection and structured compilation issues.',
        )
        append_log(state, 'warning', 'workflow.module_compilation_rejected', {
            'issue_count': len(exc.issues),
            'issues': [issue.to_dict() for issue in exc.issues],
        })
        return state
    except GateCompilationError as exc:
        state['workflow_configuration'] = {}
        state['gate_configuration'] = {}
        state['gate_provenance'] = copy.deepcopy(exc.provenance)
        for issue in exc.issues:
            state.setdefault('errors', []).append({
                'stage': 'gate_compilation',
                'code': issue.code,
                'message': issue.message,
                'details': copy.deepcopy(issue.details),
            })
            state.setdefault('validation_errors', []).append(issue.message)
        write_json_artifact(
            state,
            'gate-provenance',
            'gate-provenance.json',
            state['gate_provenance'],
            description='Rejected quality-gate selection and structured compilation issues.',
        )
        append_log(state, 'warning', 'workflow.gate_compilation_rejected', {
            'issue_count': len(exc.issues),
            'issues': [issue.to_dict() for issue in exc.issues],
        })
        return state
    except WorkflowRuntimeError as exc:
        state['workflow_configuration'] = {}
        state['workflow_provenance'] = {
            'status': 'runtime_rejected',
            'issues': [issue.to_dict() for issue in exc.issues],
        }
        for issue in exc.issues:
            state.setdefault('errors', []).append({
                'stage': 'workflow_compilation',
                'code': issue.code,
                'message': issue.message,
                'details': copy.deepcopy(issue.details),
            })
            state.setdefault('validation_errors', []).append(issue.message)
        write_json_artifact(
            state,
            'workflow-provenance',
            'workflow-provenance.json',
            state['workflow_provenance'],
            description='Rejected runtime adapter resolution and structured issues.',
        )
        append_log(state, 'warning', 'workflow.module_runtime_rejected', {
            'issue_count': len(exc.issues),
            'issues': [issue.to_dict() for issue in exc.issues],
        })
        return state
    except GateRuntimeError as exc:
        state['workflow_configuration'] = {}
        state['gate_configuration'] = {}
        state['gate_provenance'] = {
            'status': 'runtime_rejected',
            'issues': [issue.to_dict() for issue in exc.issues],
        }
        for issue in exc.issues:
            state.setdefault('errors', []).append({
                'stage': 'gate_compilation',
                'code': issue.code,
                'message': issue.message,
                'details': copy.deepcopy(issue.details),
            })
            state.setdefault('validation_errors', []).append(issue.message)
        append_log(state, 'warning', 'workflow.gate_runtime_rejected', {
            'issue_count': len(exc.issues),
            'issues': [issue.to_dict() for issue in exc.issues],
        })
        return state

    state['workflow_configuration'] = configuration.to_dict()
    state['gate_configuration'] = gate_configuration.to_dict()
    state['workflow_configuration']['gate_configuration'] = copy.deepcopy(state['gate_configuration'])
    state['workflow_provenance'] = copy.deepcopy(configuration.provenance)
    state['gate_provenance'] = copy.deepcopy(gate_configuration.provenance)
    write_json_artifact(
        state,
        'workflow-configuration',
        'workflow-configuration.json',
        state['workflow_configuration'],
        description='Deterministic executable module graph compiled from the TaskSpec.',
    )
    write_json_artifact(
        state,
        'workflow-provenance',
        'workflow-provenance.json',
        state['workflow_provenance'],
        description='Module selection, configuration, compatibility, and contract provenance.',
    )
    write_json_artifact(
        state,
        'gate-configuration',
        'gate-configuration.json',
        state['gate_configuration'],
        description='Deterministic quality gates compiled around the task module graph.',
    )
    write_json_artifact(
        state,
        'gate-provenance',
        'gate-provenance.json',
        state['gate_provenance'],
        description='Quality-gate selection, configuration, and contract provenance.',
    )
    append_log(state, 'info', 'workflow.modules_compiled', {
        'workflow_id': configuration.workflow_id,
        'module_count': len(configuration.nodes),
        'execution_order': list(configuration.execution_order),
        'gate_set_id': gate_configuration.gate_set_id,
        'gate_count': len(gate_configuration.nodes),
    })
    return state
