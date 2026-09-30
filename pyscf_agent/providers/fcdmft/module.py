from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

from ...backend.artifacts import (
    get_run_dir,
    register_artifact_reference,
    register_existing_artifact,
    retry_artifact_filename,
    serialize_npz_arrays,
    write_binary_artifact,
    write_json_artifact,
    write_text_artifact,
)
from ...backend.execution import complete_deferred_numerical_execution
from ...contracts import task_spec_from_dict, task_spec_to_dict
from ...backend.result_artifacts import refresh_periodic_module_artifacts
from ...backend.runtime_context import get_runtime_context
from ...backend.state import append_log
from ...embedding.contracts import (
    CORRELATED_SUBSPACE_AUDIT_SCHEMA,
    approve_correlated_subspace_audit,
)
from ...workflow_modules.runtime import (
    ModuleInvocation,
    RuntimeIssue,
    WorkflowRuntimeError,
    record_module_observation,
)
from .adapter import FCDMFTExecutionError, run_gw_dmft, run_hf_dmft
from .availability import fcdmft_availability
from .contracts import (
    GW_DMFT_ARRAYS_SCHEMA,
    GW_DMFT_RESULT_SCHEMA,
    HF_DMFT_ARRAYS_SCHEMA,
    HF_DMFT_RESULT_SCHEMA,
    PERIODIC_GW_ARRAYS_SCHEMA,
    load_registered_gw_artifacts,
    normalize_gw_dmft_options,
    normalize_hf_dmft_options,
    normalize_periodic_gw_options,
    validate_periodic_gw_reference,
    validate_hf_dmft_request,
)
from .gw import run_local_gw_double_counting, run_periodic_gw
from .preparation import (
    prepare_periodic_fcdmft_subspace,
    validate_fcdmft_preparation_request,
)


def _solver_name(task_spec: Any) -> str:
    return str(task_spec.solver.name or '').strip().lower().replace('-', '_')


def _record_correlated_subspace_approval(
    state: Dict[str, Any],
    task_spec: Any,
) -> None:
    reference = task_spec.embedding.audit_artifact
    path = Path(str(reference.get('path') or '')).expanduser() if isinstance(reference, dict) else None
    if path is None or not path.is_file():
        raise ValueError('Approved HF+DMFT requests require an available correlated-subspace audit')
    try:
        audit = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError('The correlated-subspace audit could not be read') from exc
    if str(audit.get('schema') or '') != CORRELATED_SUBSPACE_AUDIT_SCHEMA:
        raise ValueError('The correlated-subspace audit schema is incompatible')
    approval = audit.get('approval') if isinstance(audit.get('approval'), dict) else {}
    if not bool(approval.get('approved')):
        audit = approve_correlated_subspace_audit(
            audit,
            approved_by='user',
            note='Approved through the calculation review workflow.',
        )
        task_spec.embedding.audit_artifact = write_json_artifact(
            state,
            'correlated_subspace_audit',
            'correlated-subspace-audit-approved.json',
            audit,
            description='Approved correlated-orbital subspace for fcDMFT execution.',
        )


def prepare_hf_dmft_provider(
    state: Dict[str, Any],
    _invocation: ModuleInvocation,
) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    task_spec = task_spec_from_dict(state.get('task_spec') or {})
    solver_name = _solver_name(task_spec)
    if solver_name not in ('hf_dmft', 'gw', 'gw_dmft'):
        return state
    availability = fcdmft_availability()
    if not availability['available']:
        raise RuntimeError(
            '{0} requires the optional fcDMFT provider: {1}'.format(
                solver_name.upper().replace('_', '+'),
                availability.get('reason') or 'provider unavailable'
            )
        )
    if solver_name == 'hf_dmft':
        task_spec.solver.options = normalize_hf_dmft_options(task_spec.solver.options)
    elif solver_name == 'gw':
        task_spec.solver.options = normalize_periodic_gw_options(task_spec.solver.options)
    else:
        task_spec.solver.options = normalize_gw_dmft_options(task_spec.solver.options)
    if solver_name == 'gw':
        validate_periodic_gw_reference(task_spec)
        preparation_mode = 'lattice_gw'
    elif task_spec.embedding.approved:
        _record_correlated_subspace_approval(state, task_spec)
        if solver_name == 'hf_dmft':
            validate_hf_dmft_request(task_spec)
        else:
            validate_periodic_gw_reference(task_spec, for_dmft=True)
        preparation_mode = 'approved_execution'
    else:
        validate_fcdmft_preparation_request(task_spec)
        preparation_mode = 'correlated_subspace_proposal'
    state['task_spec'] = task_spec_to_dict(task_spec)
    state['solver_provider'] = {
        'provider': 'fcdmft',
        'method': solver_name,
        'mode': preparation_mode,
        'configuration': copy.deepcopy(task_spec.solver.options),
    }
    append_log(state, 'info', 'provider.fcdmft.prepared', {
        'solver': solver_name,
        'impurity_solver': task_spec.solver.options.get('impurity_solver'),
        'ncore': task_spec.solver.options.get('ncore'),
        'nval': task_spec.solver.options.get('nval'),
        'nbath': task_spec.solver.options.get('nbath'),
        'mode': preparation_mode,
    })
    return state


def prepare_hf_dmft_subspace(
    state: Dict[str, Any],
    invocation: ModuleInvocation,
) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    task_spec = task_spec_from_dict(state.get('task_spec') or {})
    solver_name = _solver_name(task_spec)
    if solver_name not in ('hf_dmft', 'gw_dmft'):
        return state
    if task_spec.embedding.approved:
        return state
    results = state.get('_module_results')
    context = get_runtime_context(state.get('module_context_ref'))
    if not isinstance(results, dict) or not isinstance(context, dict):
        raise RuntimeError(
            'fcDMFT correlated-subspace preparation requires the process-local periodic reference context'
        )
    prepared = prepare_periodic_fcdmft_subspace(state, task_spec, context)
    patch = prepared['task_spec_patch']
    task_payload = task_spec_to_dict(task_spec)
    task_payload['embedding'] = copy.deepcopy(patch['embedding'])
    task_payload['solver'] = copy.deepcopy(patch['solver'])
    task_spec = task_spec_from_dict(task_payload)
    state['task_spec'] = task_spec_to_dict(task_spec)

    updated = copy.deepcopy(results)
    updated['embedding_preparation'] = {
        'status': 'review_required',
        'provider': 'fcdmft',
        'localization_method': task_spec.embedding.localization_method,
        'localized_orbital_count': prepared['localized_orbital_count'],
        'source_orbital_indices': copy.deepcopy(prepared['source_orbital_indices']),
        'estimated_eri_memory_mb': prepared['estimated_eri_memory_mb'],
        'audit': copy.deepcopy(prepared['audit']),
        'task_spec_patch': copy.deepcopy(patch),
    }
    updated['analysis_text'] = (
        'The periodic reference converged and generated a candidate localized correlated subspace. '
        'Approve or edit the correlated-subspace audit before fcDMFT execution.'
    )
    state['_module_results'] = refresh_periodic_module_artifacts(state, updated)
    state = complete_deferred_numerical_execution(
        state,
        task_spec,
        converged=True,
        label='periodic correlated-subspace preparation',
    )
    observation = record_module_observation(
        state,
        invocation,
        'review_required',
        provided_result_paths=('embedding_preparation',),
        localized_orbital_count=prepared['localized_orbital_count'],
    )
    append_log(state, 'info', 'provider.fcdmft.correlated_subspace_prepared', {
        **observation,
        'solver': solver_name,
    })
    return state


def execute_periodic_gw(
    state: Dict[str, Any],
    invocation: ModuleInvocation,
) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    task_spec = task_spec_from_dict(state.get('task_spec') or {})
    solver_name = _solver_name(task_spec)
    if solver_name not in ('gw', 'gw_dmft'):
        return state
    results = state.get('_module_results')
    context = get_runtime_context(state.get('module_context_ref'))
    if not isinstance(results, dict) or not isinstance(context, dict):
        raise RuntimeError('Periodic GW requires the process-local periodic reference context')
    mean_field = context.get('mean_field')
    if not bool(getattr(mean_field, 'converged', False)):
        raise RuntimeError('Periodic GW cannot start from an unconverged periodic reference')

    options = (
        normalize_gw_dmft_options(task_spec.solver.options)
        if solver_name == 'gw_dmft'
        else normalize_periodic_gw_options(task_spec.solver.options)
    )
    reused = False
    try:
        existing_paths = load_registered_gw_artifacts(options)
    except ValueError:
        existing_paths = {}
    if existing_paths:
        try:
            gw_result = json.loads(existing_paths['lattice_result'].read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise ValueError('The registered periodic GW result artifact could not be read') from exc
        provider_log = 'Reused registered periodic GW artifacts without recomputing the lattice self-energy.\n'
        gw_references = copy.deepcopy(options['gw_artifacts'])
        for reference in gw_references.values():
            if isinstance(reference, dict) and reference.get('path'):
                register_artifact_reference(state, reference)
        reused = True
    else:
        scratch = get_run_dir(state) / 'solver-fcdmft-gw'
        try:
            gw_result, arrays, provider_log, provider_files = run_periodic_gw(
                mean_field,
                task_spec,
                scratch_directory=scratch,
            )
        except FCDMFTExecutionError as exc:
            write_text_artifact(
                state,
                'periodic_gw_output_log',
                retry_artifact_filename('log-fcdmft-gw-output', 'log', state.get('retry_count', 0)),
                exc.provider_log,
                description='Complete fcDMFT periodic GW failure log',
            )
            raise WorkflowRuntimeError((RuntimeIssue(
                'fcdmft_gw_execution_failed',
                str(exc),
                module_id=invocation.module_id,
                runtime_id=invocation.runtime_id,
                details={'failure_stage': exc.stage, 'exception_type': exc.exception_type},
            ),), state=state) from exc
        result_ref = write_json_artifact(
            state,
            'periodic_gw_result',
            retry_artifact_filename('result-periodic-gw', 'json', state.get('retry_count', 0)),
            gw_result,
            description='fcDMFT periodic GW quasiparticle and self-energy summary',
        )
        if arrays:
            arrays_ref = write_binary_artifact(
                state,
                'periodic_gw_arrays',
                retry_artifact_filename('result-periodic-gw-arrays', 'npz', state.get('retry_count', 0)),
                serialize_npz_arrays(arrays, schema=PERIODIC_GW_ARRAYS_SCHEMA),
                mime_type='application/x-npz',
                description='Periodic GW quasiparticle energies, occupations, and frequency grid',
            )
        else:
            arrays_ref = None
        log_ref = write_text_artifact(
            state,
            'periodic_gw_output_log',
            retry_artifact_filename('log-fcdmft-gw-output', 'log', state.get('retry_count', 0)),
            provider_log or 'fcDMFT periodic GW produced no stdout or stderr output.\n',
            description='Complete fcDMFT periodic GW stdout and stderr log',
        )
        gw_references = {
            'lattice_result': copy.deepcopy(result_ref),
            'lattice_ac': register_existing_artifact(
                state,
                'periodic_gw_analytic_continuation',
                provider_files['lattice_ac'],
                mime_type='application/x-hdf5',
                description='fcDMFT lattice GW analytic-continuation coefficients',
            ),
            'lattice_vxc': register_existing_artifact(
                state,
                'periodic_gw_mean_field_potential',
                provider_files['lattice_vxc'],
                mime_type='application/x-hdf5',
                description='fcDMFT lattice GW exchange and reference potential matrices',
            ),
            'lattice_sigma_imag': register_existing_artifact(
                state,
                'periodic_gw_imaginary_self_energy',
                provider_files['lattice_sigma_imag'],
                mime_type='application/x-hdf5',
                description='fcDMFT lattice GW self-energy on the imaginary axis',
            ),
        }
        if arrays_ref:
            gw_references['lattice_arrays'] = copy.deepcopy(arrays_ref)
        if log_ref:
            gw_references['lattice_log'] = copy.deepcopy(log_ref)

    options['gw_artifacts'] = gw_references
    task_spec.solver.options = options
    state['task_spec'] = task_spec_to_dict(task_spec)
    updated = copy.deepcopy(results)
    updated['gw_result'] = copy.deepcopy(gw_result)
    updated['gw_artifacts_reused'] = reused
    updated['raw_stdout'] = '\n'.join(
        part for part in (updated.get('raw_stdout', ''), provider_log) if part
    )
    if solver_name == 'gw':
        reference_energy = updated.get('final_energy')
        updated.update({
            'method': 'gw',
            'solver': 'gw',
            'reference_method': gw_result.get('reference_method'),
            'reference_energy': reference_energy,
            'mean_field_energy': reference_energy,
            'reference_converged': bool(updated.get('converged')),
            'energy': None,
            'final_energy': None,
            'final_method': 'gw',
            'energy_kind': 'gw_total_energy_unavailable',
            'converged': True,
            'gap': gw_result.get('quasiparticle_gap'),
            'analysis_text': 'Periodic GW completed and produced registered lattice self-energy artifacts.',
        })
    state['_module_results'] = refresh_periodic_module_artifacts(state, updated)
    if solver_name == 'gw':
        state = complete_deferred_numerical_execution(
            state,
            task_spec,
            converged=True,
            label='periodic GW',
        )
    observation = record_module_observation(
        state,
        invocation,
        'reused' if reused else 'completed',
        provided_result_paths=('gw_result',),
    )
    append_log(state, 'info', 'provider.fcdmft.periodic_gw_completed', observation)
    return state


def execute_local_gw_double_counting(
    state: Dict[str, Any],
    invocation: ModuleInvocation,
) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    task_spec = task_spec_from_dict(state.get('task_spec') or {})
    if _solver_name(task_spec) != 'gw_dmft' or not task_spec.embedding.approved:
        return state
    results = state.get('_module_results')
    context = get_runtime_context(state.get('module_context_ref'))
    if not isinstance(results, dict) or not isinstance(context, dict):
        raise RuntimeError('Local GW double counting requires the periodic DFT reference context')
    options = normalize_gw_dmft_options(task_spec.solver.options)
    try:
        existing_paths = load_registered_gw_artifacts(options, require_local=True)
    except ValueError:
        existing_paths = {}
    reused = bool(existing_paths)
    if reused:
        local_result = {
            'provider': 'fcdmft',
            'method': 'gw_double_counting',
            'status': 'reused',
        }
        provider_log = 'Reused registered local GW double-counting artifacts.\n'
        references = copy.deepcopy(options['gw_artifacts'])
        for reference in references.values():
            if isinstance(reference, dict) and reference.get('path'):
                register_artifact_reference(state, reference)
    else:
        scratch = get_run_dir(state) / 'solver-fcdmft-gw-dc'
        try:
            local_result, provider_log, provider_files = run_local_gw_double_counting(
                context.get('mean_field'),
                task_spec,
                scratch_directory=scratch,
            )
        except FCDMFTExecutionError as exc:
            write_text_artifact(
                state,
                'gw_dmft_local_output_log',
                retry_artifact_filename('log-fcdmft-gw-dc-output', 'log', state.get('retry_count', 0)),
                exc.provider_log,
                description='Complete fcDMFT local GW double-counting failure log',
            )
            raise WorkflowRuntimeError((RuntimeIssue(
                'fcdmft_gw_double_counting_failed',
                str(exc),
                module_id=invocation.module_id,
                runtime_id=invocation.runtime_id,
                details={'failure_stage': exc.stage, 'exception_type': exc.exception_type},
            ),), state=state) from exc
        references = copy.deepcopy(options['gw_artifacts'])
        references.update({
            'local_ac': register_existing_artifact(
                state,
                'gw_dmft_local_analytic_continuation',
                provider_files['local_ac'],
                mime_type='application/x-hdf5',
                description='Local GW double-counting analytic-continuation coefficients',
            ),
            'local_sigma_imag': register_existing_artifact(
                state,
                'gw_dmft_local_imaginary_self_energy',
                provider_files['local_sigma_imag'],
                mime_type='application/x-hdf5',
                description='Local GW double-counting self-energy on the imaginary axis',
            ),
            'local_transform': register_existing_artifact(
                state,
                'gw_dmft_local_orbital_transform',
                provider_files['local_transform'],
                mime_type='application/x-hdf5',
                description='Periodic MO-to-local-orbital transformation consumed by GW+DMFT',
            ),
        })
        write_text_artifact(
            state,
            'gw_dmft_local_output_log',
            retry_artifact_filename('log-fcdmft-gw-dc-output', 'log', state.get('retry_count', 0)),
            provider_log or 'fcDMFT local GW produced no stdout or stderr output.\n',
            description='Complete local GW double-counting stdout and stderr log',
        )
    options['gw_artifacts'] = references
    task_spec.solver.options = options
    state['task_spec'] = task_spec_to_dict(task_spec)
    updated = copy.deepcopy(results)
    updated['gw_double_counting_result'] = copy.deepcopy(local_result)
    updated['raw_stdout'] = '\n'.join(
        part for part in (updated.get('raw_stdout', ''), provider_log) if part
    )
    state['_module_results'] = refresh_periodic_module_artifacts(state, updated)
    observation = record_module_observation(
        state,
        invocation,
        'reused' if reused else 'completed',
        provided_result_paths=('gw_double_counting_result',),
    )
    append_log(state, 'info', 'provider.fcdmft.gw_double_counting_completed', observation)
    return state


def _record_dmft_failure(
    state: Dict[str, Any],
    results: Dict[str, Any],
    exc: FCDMFTExecutionError,
    *,
    method_id: str,
) -> Dict[str, Any]:
    is_gw = method_id == 'gw_dmft'
    label = 'GW+DMFT' if is_gw else 'HF+DMFT'
    schema = GW_DMFT_RESULT_SCHEMA if is_gw else HF_DMFT_RESULT_SCHEMA
    artifact_prefix = 'gw_dmft' if is_gw else 'hf_dmft'
    failure = {
        'schema': schema,
        'provider': 'fcdmft',
        'method': method_id,
        'reference_method': 'dft_gw' if is_gw else 'hf',
        'status': 'failed',
        'converged': False,
        'failure_stage': exc.stage,
        'exception_type': exc.exception_type,
        'message': str(exc),
        'energy_available': False,
    }
    write_json_artifact(
        state,
        '{0}_result'.format(artifact_prefix),
        retry_artifact_filename(
            'result-{0}-failure'.format(artifact_prefix.replace('_', '-')),
            'json',
            state.get('retry_count', 0),
        ),
        failure,
        description='Structured fcDMFT {0} provider failure summary'.format(label),
    )
    write_text_artifact(
        state,
        '{0}_output_log'.format(artifact_prefix),
        retry_artifact_filename(
            'log-fcdmft-output', 'log', state.get('retry_count', 0)
        ),
        exc.provider_log,
        description='Complete fcDMFT provider stdout, stderr, and failure traceback',
    )
    register_existing_artifact(
        state,
        '{0}_checkpoint'.format(artifact_prefix),
        exc.scratch_directory / 'dmft-checkpoint.h5',
        mime_type='application/x-hdf5',
        description='Partial fcDMFT checkpoint retained after provider failure',
    )
    updated = copy.deepcopy(results)
    reference_energy = updated.get('final_energy')
    reference_converged = bool(updated.get('converged'))
    updated.update({
        'method': method_id,
        'solver': method_id,
        'reference_method': 'dft_gw' if is_gw else 'hf',
        'reference_energy': reference_energy,
        'mean_field_energy': reference_energy,
        'reference_converged': reference_converged,
        'energy': None,
        'final_energy': None,
        'final_method': method_id,
        'energy_kind': 'dmft_total_energy_unavailable',
        'converged': False,
        'dmft_result': failure,
        'analysis_text': str(exc),
        'raw_stdout': '\n'.join(
            part for part in (updated.get('raw_stdout', ''), exc.provider_log) if part
        ),
    })
    state['_module_results'] = refresh_periodic_module_artifacts(state, updated)
    append_log(state, 'error', 'provider.fcdmft.{0}_failed'.format(method_id), failure)
    return state


def _execute_dmft(
    state: Dict[str, Any],
    invocation: ModuleInvocation,
    *,
    method_id: str,
) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    task_spec = task_spec_from_dict(state.get('task_spec') or {})
    if _solver_name(task_spec) != method_id:
        return state
    is_gw = method_id == 'gw_dmft'
    label = 'GW+DMFT' if is_gw else 'HF+DMFT'
    reference_label = 'DFT+GW' if is_gw else 'HF'
    if not task_spec.embedding.approved:
        append_log(state, 'info', 'provider.fcdmft.execution_waiting_for_subspace_approval', {})
        return state
    results = state.get('_module_results')
    context = get_runtime_context(state.get('module_context_ref'))
    if not isinstance(results, dict) or not isinstance(context, dict):
        raise RuntimeError('{0} requires the process-local periodic reference context'.format(label))
    if not bool(getattr(context.get('mean_field'), 'converged', False)):
        raise RuntimeError('{0} cannot start from an unconverged periodic reference'.format(label))

    scratch = get_run_dir(state) / 'solver-fcdmft-{0}'.format(method_id.replace('_', '-'))
    try:
        runner = run_gw_dmft if is_gw else run_hf_dmft
        dmft_result, arrays, provider_log = runner(
            task_spec,
            scratch_directory=scratch,
            reference_chemical_potential=context.get('fermi_energy'),
            reference_occupancy=context.get('electron_count'),
        )
    except FCDMFTExecutionError as exc:
        state = _record_dmft_failure(state, results, exc, method_id=method_id)
        raise WorkflowRuntimeError((RuntimeIssue(
            'fcdmft_execution_failed',
            str(exc),
            module_id=invocation.module_id,
            runtime_id=invocation.runtime_id,
            details={
                'failure_stage': exc.stage,
                'exception_type': exc.exception_type,
            },
        ),), state=state) from exc
    updated = copy.deepcopy(results)
    reference_energy = updated.get('final_energy')
    reference_converged = bool(updated.get('converged'))
    updated.update({
        'method': method_id,
        'solver': method_id,
        'reference_method': 'dft_gw' if is_gw else 'hf',
        'reference_energy': reference_energy,
        'mean_field_energy': reference_energy,
        'reference_converged': reference_converged,
        'energy': None,
        'final_energy': None,
        'final_method': method_id,
        'energy_kind': 'dmft_total_energy_unavailable',
        'converged': bool(dmft_result['converged']),
        'dmft_result': copy.deepcopy(dmft_result),
        'analysis_text': (
            '{0} converged.'.format(label) if dmft_result['converged']
            else '{0} did not satisfy the hybridization convergence tolerance.'.format(label)
        ),
        'raw_stdout': '\n'.join(
            part for part in (updated.get('raw_stdout', ''), provider_log) if part
        ),
    })
    write_json_artifact(
        state,
        '{0}_result'.format(method_id),
        retry_artifact_filename(
            'result-{0}'.format(method_id.replace('_', '-')),
            'json',
            state.get('retry_count', 0),
        ),
        dmft_result,
        description='fcDMFT {0} convergence, bath, solver, and resource summary'.format(label),
    )
    if arrays:
        write_binary_artifact(
            state,
            '{0}_arrays'.format(method_id),
            retry_artifact_filename(
                'result-{0}-arrays'.format(method_id.replace('_', '-')),
                'npz',
                state.get('retry_count', 0),
            ),
            serialize_npz_arrays(
                arrays,
                schema=(GW_DMFT_ARRAYS_SCHEMA if is_gw else HF_DMFT_ARRAYS_SCHEMA),
            ),
            mime_type='application/x-npz',
            description='fcDMFT hybridization, self-energy, and frequency arrays',
        )
    write_text_artifact(
        state,
        '{0}_output_log'.format(method_id),
        retry_artifact_filename(
            'log-fcdmft-output', 'log', state.get('retry_count', 0)
        ),
        provider_log or 'fcDMFT produced no stdout or stderr output.\n',
        description='Complete fcDMFT provider stdout and stderr log',
    )
    register_existing_artifact(
        state,
        '{0}_checkpoint'.format(method_id),
        Path(scratch) / 'dmft-checkpoint.h5',
        mime_type='application/x-hdf5',
        description='fcDMFT hybridization and self-energy checkpoint',
    )
    state['_module_results'] = refresh_periodic_module_artifacts(state, updated)
    state = complete_deferred_numerical_execution(
        state,
        task_spec,
        converged=bool(dmft_result['converged']),
        label=label,
    )
    observation = record_module_observation(
        state,
        invocation,
        'completed' if dmft_result['converged'] else 'unconverged',
        provided_result_paths=('dmft_result', 'reference_energy', 'final_method'),
    )
    append_log(state, 'info', 'provider.fcdmft.{0}_completed'.format(method_id), {
        **observation,
        'reference': reference_label,
    })
    return state


def execute_hf_dmft(state: Dict[str, Any], invocation: ModuleInvocation) -> Dict[str, Any]:
    return _execute_dmft(state, invocation, method_id='hf_dmft')


def execute_gw_dmft(state: Dict[str, Any], invocation: ModuleInvocation) -> Dict[str, Any]:
    return _execute_dmft(state, invocation, method_id='gw_dmft')


__all__ = [
    'execute_gw_dmft',
    'execute_hf_dmft',
    'execute_local_gw_double_counting',
    'execute_periodic_gw',
    'prepare_hf_dmft_provider',
    'prepare_hf_dmft_subspace',
]
