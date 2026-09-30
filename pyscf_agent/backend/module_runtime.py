from __future__ import annotations

import copy
import functools
from typing import Any, Callable, Dict, Mapping

from .artifacts import write_result_json_artifact
from .execution import (
    input_generator,
    repair_or_retry,
    result_analyst,
    result_extractor,
    runner,
    sanitize_task_report,
    task_reporter,
)
from .molecular_modules import (
    run_molecular_active_space_audit,
    run_molecular_correlation_diagnostics,
    run_molecular_orbital_processing,
)
from .model_hamiltonian.modules import run_model_correlation_diagnostics
from .periodic.modules import run_periodic_band_analysis
from .probe_modules import (
    prepare_molecular_active_space_probe,
    refine_molecular_active_space_probe,
)
from .runtime_context import discard_runtime_context
from .state import append_log
from ..lifecycle import transition_lifecycle
from ..providers.block2.module import (
    configure_bond_dimension_planning,
    configure_entanglement_diagnostics,
    configure_excited_states,
    configure_mps_continuation,
    configure_symmetry_analysis,
    prepare_block2_provider,
)
from ..providers.libdmet.module import configure_reference_density, prepare_dmet_provider
from ..providers.fcdmft.module import (
    execute_gw_dmft,
    execute_hf_dmft,
    execute_local_gw_double_counting,
    execute_periodic_gw,
    prepare_hf_dmft_provider,
    prepare_hf_dmft_subspace,
)
from ..workflow_modules.runtime import (
    ModuleInvocation,
    ModuleRuntimeAdapter,
    ModuleRuntimeRegistry,
    RuntimeIssue,
    WorkflowRuntimeDispatcher,
    WorkflowRuntimeError,
    execution_trace_payload,
    validate_runtime_configuration,
)


def _state_handler(handler: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Callable[[Dict[str, Any], ModuleInvocation], Dict[str, Any]]:
    def invoke(state: Dict[str, Any], _invocation: ModuleInvocation) -> Dict[str, Any]:
        return handler(state)

    return invoke


def build_default_runtime_registry() -> ModuleRuntimeRegistry:
    """Bind declared runtime ids to the current trusted backend implementations."""

    return ModuleRuntimeRegistry([
        ModuleRuntimeAdapter(
            'pyscf_agent.backend.execution.input_generator',
            _state_handler(input_generator),
            ('task.prepare',),
            description='Generate the reproducible PySCF input program.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.backend.active_space.probe',
            prepare_molecular_active_space_probe,
            ('task.prepare',),
            description='Prepare an active-space probe without activating the target CAS solver.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.backend.active_space.probe_refinement',
            refine_molecular_active_space_probe,
            ('task.diagnose',),
            description='Refine an unresolved SCF/AVAS active-space proposal with MP2 evidence.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.backend.execution.runner',
            _state_handler(runner),
            ('task.execute',),
            description='Execute the numerical task in the selected executor process.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.providers.libdmet.reference_density',
            configure_reference_density,
            ('task.prepare',),
            description='Configure a PM, AF, or FM initial reference density with zero auxiliary vcor.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.providers.libdmet.dmet',
            prepare_dmet_provider,
            ('task.prepare',),
            description='Validate and configure the libDMET Hubbard self-consistency workflow.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.providers.fcdmft.prepare_hf_dmft',
            prepare_hf_dmft_provider,
            ('task.prepare',),
            description='Validate approved periodic embedding artifacts and configure fcDMFT.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.providers.fcdmft.execute_hf_dmft',
            execute_hf_dmft,
            ('task.execute',),
            description='Run the terminal fcDMFT HF+DMFT numerical stage.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.providers.fcdmft.execute_periodic_gw',
            execute_periodic_gw,
            ('task.execute',),
            description='Run or reuse the periodic fcDMFT GW provider stage.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.providers.fcdmft.execute_local_gw_double_counting',
            execute_local_gw_double_counting,
            ('task.execute',),
            description='Build or reuse local GW double-counting artifacts.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.providers.fcdmft.execute_gw_dmft',
            execute_gw_dmft,
            ('task.execute',),
            description='Run the terminal fcDMFT GW+DMFT numerical stage.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.providers.fcdmft.prepare_hf_dmft_subspace',
            prepare_hf_dmft_subspace,
            ('task.execute',),
            description=(
                'Generate reviewable IAO-localized periodic Hamiltonian artifacts '
                'before the first fcDMFT execution.'
            ),
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.providers.block2.prepare',
            prepare_block2_provider,
            ('task.prepare',),
            description='Validate the optional block2 installation and normalize DMRG settings.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.providers.block2.bond_dimension_planning',
            configure_bond_dimension_planning,
            ('task.prepare',),
            description='Configure exact-sector bond-dimension planning for block2 DMRG.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.providers.block2.entanglement',
            configure_entanglement_diagnostics,
            ('task.prepare',),
            description='Configure orbital and bipartite entanglement diagnostics for block2 DMRG.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.providers.block2.excited_states',
            configure_excited_states,
            ('task.prepare',),
            description='Configure state-averaged multi-root block2 DMRG.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.providers.block2.mps_continuation',
            configure_mps_continuation,
            ('task.prepare',),
            description='Configure a block2 MPS checkpoint as the initial state for another task.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.providers.block2.symmetry_analysis',
            configure_symmetry_analysis,
            ('task.prepare',),
            description='Configure particle-number, spin, and orbital-symmetry analysis for block2 DMRG.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.backend.execution.repair_or_retry',
            _state_handler(repair_or_retry),
            ('task.recover',),
            description='Apply the bounded single-task repair policy.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.backend.execution.result_extractor',
            _state_handler(result_extractor),
            ('task.extract',),
            description='Normalize raw numerical output into structured results.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.backend.execution.result_analyst',
            _state_handler(result_analyst),
            ('task.diagnose',),
            description='Create deterministic task-level analysis.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.backend.correlation_diagnostics',
            run_molecular_correlation_diagnostics,
            ('task.diagnose',),
            description='Compute molecular diagnostics from the process-local solver context.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.backend.orbital_processing',
            run_molecular_orbital_processing,
            ('task.diagnose',),
            description='Compute localized- or natural-orbital processing output.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.backend.active_space.audit',
            run_molecular_active_space_audit,
            ('task.diagnose',),
            description='Compute a reviewable ActiveSpaceAudit from molecular solver state.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.backend.model_hamiltonian.diagnostics',
            run_model_correlation_diagnostics,
            ('task.diagnose',),
            description='Compute finite-cluster model-Hamiltonian diagnostics.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.backend.periodic.band_analysis',
            run_periodic_band_analysis,
            ('task.diagnose',),
            description='Compute periodic band-path and band-energy results.',
        ),
        ModuleRuntimeAdapter(
            'pyscf_agent.backend.execution.task_reporter',
            _state_handler(task_reporter),
            ('task.finalize',),
            description='Assemble the versioned TaskReport.',
        ),
    ])


@functools.lru_cache(maxsize=1)
def get_default_runtime_registry() -> ModuleRuntimeRegistry:
    return build_default_runtime_registry()


@functools.lru_cache(maxsize=1)
def get_default_runtime_dispatcher() -> WorkflowRuntimeDispatcher:
    return WorkflowRuntimeDispatcher(get_default_runtime_registry())


def validate_compiled_runtime(configuration: Mapping[str, Any]) -> None:
    issues = validate_runtime_configuration(configuration, get_default_runtime_registry())
    if issues:
        raise WorkflowRuntimeError(issues)


def _runtime_failure(state: Dict[str, Any], exc: Exception, stage: str) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    if isinstance(exc, WorkflowRuntimeError):
        issues = exc.issues
    else:
        issues = (RuntimeIssue(
            'runtime_adapter_exception',
            'Workflow module execution failed: {0}'.format(exc),
            details={'exception_type': type(exc).__name__, 'stage': stage},
        ),)
    for issue in issues:
        state.setdefault('errors', []).append({
            'stage': 'workflow_runtime',
            'code': issue.code,
            'message': issue.message,
            'details': copy.deepcopy(issue.details),
        })
    message = '; '.join(issue.message for issue in issues)
    state['raw_stderr'] = message
    if stage in ('task.prepare', 'task.execute', 'task.recover'):
        state['execution_status'] = 'failed'
    lifecycle = state.get('lifecycle')
    if (
        stage == 'task.execute'
        and isinstance(lifecycle, dict)
        and lifecycle.get('stage') == 'running'
    ):
        state['lifecycle'] = transition_lifecycle(
            lifecycle,
            'execution_failed',
            details={'source': 'workflow_runtime', 'stage': stage},
        )
    append_log(state, 'error', 'workflow.module_runtime_failed', {
        'stage': stage,
        'issues': [issue.to_dict() for issue in issues],
    })
    return state


def run_compiled_stage(
    state: Dict[str, Any],
    stage: str,
) -> Dict[str, Any]:
    """Run one compiled hook or an explicit validation-rejection stage."""

    configuration = state.get('workflow_configuration')
    if not isinstance(configuration, Mapping) or not configuration:
        rejection_handlers = {
            'task.execute': runner,
            'task.extract': result_extractor,
            'task.diagnose': result_analyst,
            'task.finalize': task_reporter,
        }
        handler = rejection_handlers.get(stage)
        if state.get('validation_errors') and handler is not None:
            append_log(state, 'info', 'workflow.validation_rejection_stage', {
                'stage': stage,
            })
            return handler(state)
        issue = RuntimeIssue(
            'missing_workflow_configuration',
            'A validated task cannot run without a compiled workflow configuration.',
            details={'stage': stage},
        )
        return _runtime_failure(state, WorkflowRuntimeError((issue,)), stage)
    try:
        module_ids = list(configuration.get('hooks', {}).get(stage, ()))
        append_log(state, 'info', 'workflow.module_stage_started', {
            'workflow_id': configuration.get('workflow_id'),
            'stage': stage,
            'modules': module_ids,
        })
        result = get_default_runtime_dispatcher().run_stage(state, stage)
        append_log(result, 'info', 'workflow.module_stage_completed', {
            'workflow_id': configuration.get('workflow_id'),
            'stage': stage,
            'modules': module_ids,
        })
        return result
    except Exception as exc:  # Runtime failures become structured task failures.
        failed_state = exc.state if isinstance(exc, WorkflowRuntimeError) else None
        return _runtime_failure(failed_state or state, exc, stage)


def compiled_input_generator(state: Dict[str, Any]) -> Dict[str, Any]:
    return run_compiled_stage(state, 'task.prepare')


def compiled_runner(state: Dict[str, Any]) -> Dict[str, Any]:
    return run_compiled_stage(state, 'task.execute')


def compiled_repair_or_retry(state: Dict[str, Any]) -> Dict[str, Any]:
    return run_compiled_stage(state, 'task.recover')


def compiled_result_extractor(state: Dict[str, Any]) -> Dict[str, Any]:
    return run_compiled_stage(state, 'task.extract')


def compiled_result_analyst(state: Dict[str, Any]) -> Dict[str, Any]:
    return run_compiled_stage(state, 'task.diagnose')


def compiled_task_reporter(state: Dict[str, Any]) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    discard_runtime_context(state.pop('module_context_ref', None))
    state.pop('_module_results', None)
    result = run_compiled_stage(state, 'task.finalize')
    configuration = result.get('workflow_configuration')
    if not isinstance(configuration, Mapping) or not configuration:
        return result
    trace_payload = execution_trace_payload(result)
    write_result_json_artifact(
        result,
        'workflow-execution-trace',
        'workflow-execution-trace.json',
        trace_payload,
        description='Ordered runtime record for every compiled workflow module invocation.',
    )
    report = result.get('task_report')
    if isinstance(report, dict):
        report['module_execution_trace'] = copy.deepcopy(result.get('module_execution_trace') or [])
        report['module_runtime_observations'] = copy.deepcopy(result.get('module_runtime_observations') or [])
        report['artifacts'] = copy.deepcopy(result.get('artifacts') or [])
        report['logs'] = copy.deepcopy(result.get('logs') or [])
        for warning in result.get('warnings') or []:
            if warning not in report.setdefault('warnings', []):
                report['warnings'].append(copy.deepcopy(warning))
        result['task_report'] = sanitize_task_report(report)
    return result


__all__ = [
    'build_default_runtime_registry',
    'compiled_input_generator',
    'compiled_repair_or_retry',
    'compiled_result_analyst',
    'compiled_result_extractor',
    'compiled_runner',
    'compiled_task_reporter',
    'get_default_runtime_dispatcher',
    'get_default_runtime_registry',
    'run_compiled_stage',
    'validate_compiled_runtime',
]
