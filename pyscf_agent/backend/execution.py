from __future__ import annotations

import copy
import io
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..lifecycle import (
    ensure_lifecycle,
    lifecycle_requires_review,
    prepare_task_for_execution,
    transition_lifecycle,
)

from .artifacts import (
    compact_structured_results,
    ensure_run_id,
    get_run_dir,
    preview_text,
    register_existing_artifact,
    retry_artifact_filename,
    write_result_json_artifact,
    write_text_artifact,
)
from .correlation.strong import (
    build_correlation_diagnostics,
    build_orbital_processing_summary,
    build_scf_stability_summary,
    configure_fci_solver,
    propose_active_space,
    run_cas_method,
    state_target_configuration,
)
from .model_hamiltonian.solver import generate_model_hamiltonian_input_script, run_model_hamiltonian_solver
from ..contracts import DEFAULT_ANALYSIS, InitialStateSpec, TaskReport, TaskSpec, task_spec_from_dict, task_spec_to_dict
from .one_particle_state import capture_reference_1rdm, projected_initial_density
from .periodic.solver import run_periodic_task
from .result_artifacts import _model_energy_levels_table as _model_energy_levels_table, write_result_artifacts
from .runtime_context import discard_runtime_context, store_runtime_context
from .script_helpers import _collect_result_payload, _extract_homo_lumo as _extract_homo_lumo, _format_script_output as _format_script_output, _safe_to_list as _safe_to_list
from .state import (
    _append_attempt,
    _append_errors_to_state,
    _latest_error_message,
    _make_error,
    append_log,
    append_message,
)
from ..pyscf_i18n import join_items, t, yes_no


def _build_method_constructor(task_spec: TaskSpec) -> Tuple[str, List[str]]:
    settings = []
    method = task_spec.method
    if method.name in ('hf', 'mp2', 'ccsd', 'ccsd_t', 'fci', 'casci', 'casscf'):
        if method.name in ('casci', 'casscf'):
            # A spin-adapted CAS may be initialized from broken-symmetry UHF
            # orbitals without turning the orbital optimizer into UCASSCF.
            constructor = 'scf.UHF(mol)'
        else:
            constructor = 'scf.RHF(mol)' if method.restricted else 'scf.UHF(mol)'
    else:
        constructor = 'dft.RKS(mol)' if method.restricted else 'dft.UKS(mol)'
        settings.append("mf.xc = {0}".format(repr(method.xc)))
    settings.append('mf.max_cycle = {0}'.format(task_spec.runtime.max_cycle))
    if task_spec.runtime.conv_tol is not None:
        settings.append('mf.conv_tol = {0}'.format(task_spec.runtime.conv_tol))
    if task_spec.runtime.conv_tol_grad is not None:
        settings.append('mf.conv_tol_grad = {0}'.format(task_spec.runtime.conv_tol_grad))
    if task_spec.runtime.grid_level is not None and method.name == 'dft':
        settings.append('mf.grids.level = {0}'.format(task_spec.runtime.grid_level))
    if task_spec.runtime.diis_space is not None:
        settings.append('mf.diis_space = {0}'.format(task_spec.runtime.diis_space))
    if task_spec.runtime.scf_algorithm == 'newton':
        settings.insert(0, 'mf = mf.newton()')
    return constructor, settings


def _density_fitting_metadata(task_spec: TaskSpec, mf: Any = None, cas_result: Any = None) -> Dict[str, Any]:
    reference_df = getattr(mf, 'with_df', None)
    cas_df = (cas_result or {}).get('density_fitting') or {}
    applied_to = ['scf'] if reference_df else []
    if cas_df.get('enabled'):
        applied_to.append(task_spec.method.name)
    return {
        'enabled': bool(task_spec.density_fitting.enabled),
        'auxbasis': task_spec.density_fitting.auxbasis,
        'apply_to': task_spec.density_fitting.apply_to or 'scf',
        'applied_to': applied_to,
        'resolved_auxbasis': getattr(reference_df, 'auxbasis', None),
        'casscf_implementation': cas_df.get('implementation') if task_spec.method.name == 'casscf' else None,
    }


def _molecular_reference_label(task_spec: TaskSpec) -> str:
    if task_spec.method.name in ('casci', 'casscf'):
        return 'uhf'
    return 'rhf' if task_spec.method.restricted else 'uhf'


def _cas_reference_label(task_spec: TaskSpec) -> str:
    """Return the CAS formulation after the UHF orbital-reference stage."""
    if not task_spec.method.restricted:
        return 'uhf'
    return 'rohf' if task_spec.system.spin != 0 else 'rhf'


def _reference_convergence_controls_task(task_spec: TaskSpec) -> bool:
    """Return whether the initial mean-field convergence is terminal.

    CASSCF performs a variational orbital optimization after the initial SCF
    orbital guess.  Once that optimizer and its active-space solver converge,
    an unconverged initial UHF reference is provenance worth reporting, not a
    failure of the final CASSCF calculation.  CASCI and single-reference
    methods do not have that independent orbital-optimization step.
    """

    return not (
        task_spec.task_type == 'molecular'
        and task_spec.method.name == 'casscf'
    )


def _molecular_task_converged(
    task_spec: TaskSpec,
    *,
    reference_converged: bool,
    solver_converged: Optional[bool] = None,
) -> bool:
    if solver_converged is None:
        return bool(reference_converged)
    return bool(solver_converged) and (
        bool(reference_converged)
        or not _reference_convergence_controls_task(task_spec)
    )


def _apply_single_electron_exact_limit(
    mol: Any,
    post_hf: Any,
    reference_energy: float,
) -> bool:
    """Avoid empty-spin tensor paths when mean field is exact for one electron."""
    if int(mol.nelectron) != 1:
        return False
    post_hf.e_hf = float(reference_energy)
    post_hf.e_corr = 0.0
    post_hf.converged = True
    return True


def generate_input_script(task_spec: TaskSpec) -> str:
    if task_spec.task_type == 'model_hamiltonian':
        return generate_model_hamiltonian_input_script(
            task_spec.model_hamiltonian.spec,
            solver_name=task_spec.solver.name,
            outputs=_normalized_outputs(task_spec),
            solver_options=task_spec.solver.options,
            runtime=task_spec.runtime,
        )
    task_payload = task_spec_to_dict(task_spec)
    active_space_probe = 'molecular.active_space_probe' in set(task_spec.workflow.modules or [])
    if (
        active_space_probe
        or 'ccsd_labels' in task_spec.analysis.outputs
        or (
            task_spec.task_type == 'periodic'
            and str(task_spec.solver.name or '').strip().lower().replace('-', '_')
            in ('gw', 'hf_dmft', 'gw_dmft')
        )
    ):
        return '''from __future__ import annotations

import json

from pyscf_agent.serialization import json_default
from pyscf_agent.backend.state import default_state
from pyscf_agent.backend.workflow import run_workflow_sequential

request = json.dumps(TASK_SPEC_JSON)
state = run_workflow_sequential(default_state(request, channel='generated-script'))
print(json.dumps(state['task_report'], indent=2, sort_keys=True, default=json_default, allow_nan=False))
'''.replace('TASK_SPEC_JSON', repr(task_payload))
    return '''from __future__ import annotations

from pyscf_agent.backend.execution import _run_pyscf_task
from pyscf_agent.contracts import task_spec_from_dict
from pyscf_agent.backend.script_helpers import _format_script_output

task_spec = task_spec_from_dict(TASK_SPEC_JSON)
result = _run_pyscf_task(task_spec)
result.pop('_transient_one_particle_state', None)
result.pop('_transient_module_context', None)
result.pop('_transient_molecular_md_arrays', None)
print(_format_script_output(result))
'''.replace('TASK_SPEC_JSON', repr(task_payload))


def _effective_task_spec(state: Dict[str, Any]) -> TaskSpec:
    task_spec = task_spec_from_dict(state['task_spec'])
    provider = state.get('solver_provider')
    if isinstance(provider, dict) and provider.get('provider') == 'block2_dmrg':
        configuration = provider.get('configuration')
        if isinstance(configuration, dict):
            task_spec.solver.name = 'block2_dmrg'
            task_spec.solver.options = copy.deepcopy(configuration)
    return task_spec


def input_generator(state: Dict[str, Any]) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    ensure_run_id(state)
    task_spec = _effective_task_spec(state)
    generated_input = generate_input_script(task_spec)
    write_text_artifact(
        state,
        'generated_input',
        retry_artifact_filename('input-generated-pyscf', 'py', state.get('retry_count', 0)),
        generated_input,
        mime_type='text/x-python; charset=utf-8',
        description='Generated PySCF input script',
    )
    state['generated_input'] = preview_text(generated_input)
    append_log(state, 'info', 'workflow.input_generated', {
        'line_count': len(generated_input.splitlines()),
    })
    return state


def _normalized_outputs(task_spec: TaskSpec) -> List[str]:
    outputs = []
    for item in task_spec.analysis.outputs:
        if item not in outputs:
            outputs.append(item)
    return outputs or list(DEFAULT_ANALYSIS)


MOLECULAR_NUMERICAL_MODULES = frozenset({
    'molecular.correlation_diagnostics',
    'molecular.active_space_probe_refinement',
    'molecular.orbital_processing',
    'molecular.active_space_audit',
})

MODEL_NUMERICAL_MODULES = frozenset({
    'model.correlation_diagnostics',
})

PERIODIC_NUMERICAL_MODULES = frozenset({
    'periodic.band_analysis',
    'embedding.fcdmft.periodic_gw',
    'embedding.fcdmft.prepare_subspace',
    'embedding.fcdmft.gw_double_counting',
    'embedding.fcdmft.hf_dmft',
    'embedding.fcdmft.gw_dmft',
})

TERMINAL_DEFERRED_NUMERICAL_MODULES = frozenset({
    'embedding.fcdmft.periodic_gw',
    'embedding.fcdmft.prepare_subspace',
    'embedding.fcdmft.hf_dmft',
    'embedding.fcdmft.gw_dmft',
})

NUMERICAL_MODULES_BY_TASK_TYPE = {
    'molecular': MOLECULAR_NUMERICAL_MODULES,
    'model_hamiltonian': MODEL_NUMERICAL_MODULES,
    'periodic': PERIODIC_NUMERICAL_MODULES,
}


def _run_pyscf_task(
    task_spec: TaskSpec,
    *,
    deferred_modules: Tuple[str, ...] = (),
    execution_directory: Any = None,
) -> Dict[str, Any]:
    solver_scratch = None
    if execution_directory is not None:
        solver_id = str(task_spec.solver.name or 'pyscf').strip().lower().replace('_', '-')
        solver_scratch = str(Path(execution_directory) / 'solver-{0}'.format(solver_id))
    if task_spec.task_type == 'model_hamiltonian':
        return run_model_hamiltonian_solver(
            task_spec.model_hamiltonian.spec,
            solver_name=task_spec.solver.name,
            outputs=_normalized_outputs(task_spec),
            deferred_modules=deferred_modules,
            solver_options=task_spec.solver.options,
            runtime=task_spec.runtime,
            scratch_directory=solver_scratch,
        )
    if task_spec.task_type == 'periodic':
        return run_periodic_task(task_spec, deferred_modules=deferred_modules)
    if task_spec.job.name == 'molecular_dynamics':
        from .molecular_dynamics import run_molecular_dynamics_task  # pylint: disable=import-outside-toplevel

        return run_molecular_dynamics_task(task_spec)

    from pyscf import cc, dft, fci, gto, mp, scf  # pylint: disable=import-outside-toplevel

    outputs = _normalized_outputs(task_spec)
    scf_stdout = io.StringIO()

    mol = gto.Mole()
    mol.stdout = scf_stdout
    mol.build(
        dump_input=False,
        parse_arg=False,
        atom=task_spec.system.atom,
        basis=task_spec.system.basis,
        unit=task_spec.system.unit,
        charge=task_spec.system.charge,
        spin=task_spec.system.spin,
        symmetry=task_spec.system.symmetry,
        verbose=task_spec.runtime.verbose,
    )

    if task_spec.method.name in ('hf', 'mp2', 'ccsd', 'ccsd_t', 'fci', 'casci', 'casscf'):
        if task_spec.method.name in ('casci', 'casscf'):
            mf = scf.UHF(mol)
        else:
            mf = scf.RHF(mol) if task_spec.method.restricted else scf.UHF(mol)
    else:
        mf = dft.RKS(mol) if task_spec.method.restricted else dft.UKS(mol)
        mf.xc = task_spec.method.xc

    if task_spec.density_fitting.enabled:
        mf = mf.density_fit(auxbasis=task_spec.density_fitting.auxbasis)
    if task_spec.runtime.scf_algorithm == 'newton':
        mf = mf.newton()
    mf.stdout = scf_stdout
    mf.max_cycle = task_spec.runtime.max_cycle
    if task_spec.runtime.conv_tol is not None:
        mf.conv_tol = task_spec.runtime.conv_tol
    if task_spec.runtime.conv_tol_grad is not None:
        mf.conv_tol_grad = task_spec.runtime.conv_tol_grad
    if task_spec.runtime.grid_level is not None and task_spec.method.name == 'dft':
        mf.grids.level = task_spec.runtime.grid_level
    if task_spec.runtime.diis_space is not None:
        mf.diis_space = task_spec.runtime.diis_space

    initial_density, initial_state_summary = projected_initial_density(task_spec, mol)
    reference_energy = mf.kernel(dm0=initial_density) if initial_density is not None else mf.kernel()
    energy = reference_energy
    correlation_energy = None
    triples_correction = None
    post_hf = None
    cas_result: Dict[str, Any] = {}
    state_results: Dict[str, Any] = {}

    if task_spec.method.name == 'mp2':
        post_hf = mp.MP2(mf) if task_spec.method.restricted else mp.UMP2(mf)
        correlation_energy, _amps = post_hf.kernel()
        energy = post_hf.e_tot
    elif task_spec.method.name in ('ccsd', 'ccsd_t'):
        post_hf = cc.CCSD(mf) if task_spec.method.restricted else cc.UCCSD(mf)
        single_electron_limit = _apply_single_electron_exact_limit(
            mol,
            post_hf,
            reference_energy,
        )
        if single_electron_limit:
            correlation_energy, _t1, _t2 = 0.0, None, None
        else:
            correlation_energy, _t1, _t2 = post_hf.kernel()
        energy = post_hf.e_tot
        if task_spec.method.name == 'ccsd_t':
            triples_correction = (
                0.0
                if single_electron_limit
                else float(post_hf.ccsd_t())
            )
            energy = post_hf.e_tot + triples_correction
            correlation_energy = energy - reference_energy
    elif task_spec.method.name == 'fci':
        post_hf = fci.FCI(mol, mf.mo_coeff)
        default_nroots = 2 if 'excited_states' in outputs else 1
        requested_nroots, _state_weights = state_target_configuration(
            task_spec.solver.options,
            default_nroots=default_nroots,
        )
        configure_fci_solver(post_hf, requested_nroots)
        raw_energy, _civec = post_hf.kernel()
        raw_state_energies = (
            raw_energy.tolist() if hasattr(raw_energy, 'tolist') else raw_energy
        )
        state_energies = (
            [float(value) for value in raw_state_energies]
            if isinstance(raw_state_energies, (list, tuple))
            else [float(raw_state_energies)]
        )
        energy = state_energies[0]
        state_results = {
            'state_energies': state_energies,
            'excitation_energies': [
                float(value - state_energies[0]) for value in state_energies
            ],
            'requested_root_count': requested_nroots,
            'computed_root_count': len(state_energies),
            'targeted_roots_complete': len(state_energies) >= requested_nroots,
            'state_sector': {
                'electron_count': int(mol.nelectron),
                'spin_2s': int(mol.spin),
                'scope': 'fixed_particle_spin_projection_sector',
            },
        }
        correlation_energy = energy - reference_energy
    elif task_spec.method.name in ('casci', 'casscf'):
        initial_reference_label = _molecular_reference_label(task_spec)
        reference_label = _cas_reference_label(task_spec)
        solver_options = dict(task_spec.solver.options or {})
        if 'excited_states' in outputs and 'nroots' not in solver_options:
            solver_options['nroots'] = 2
        if task_spec.solver.name == 'block2_dmrg':
            from ..providers.block2 import block2_options_for_outputs  # pylint: disable=import-outside-toplevel

            solver_options = block2_options_for_outputs(solver_options, outputs)
        cas_result = run_cas_method(
            mf,
            mol,
            task_spec.method.name,
            task_spec.active_space,
            task_spec.runtime.max_cycle,
            unrestricted=not task_spec.method.restricted,
            reference_label=reference_label,
            sc_nevpt2=task_spec.post_cas.sc_nevpt2,
            solver_name=task_spec.solver.name,
            solver_options=solver_options,
            scratch_directory=solver_scratch,
            orbital_processing=task_spec.orbital_processing,
            density_fitting=task_spec.density_fitting,
        )
        cas_result['initial_reference'] = initial_reference_label
        cas_result['cas_reference'] = reference_label
        dmrg_arrays = cas_result.pop('_transient_dmrg_arrays', None)
        post_cas_results = cas_result.pop('post_cas_results', {})
        post_hf = type('CASResultProxy', (), {'converged': cas_result.get('converged', True)})()
        energy = cas_result['energy']
        correlation_energy = energy - reference_energy

    result = _collect_result_payload(mf, mol, energy, outputs, scf_stdout.getvalue())
    result['task_type'] = 'molecular'
    result['method'] = task_spec.method.name
    if task_spec.method.name in ('casci', 'casscf'):
        result['solver'] = str(task_spec.solver.name or 'fci').strip().lower().replace('-', '_')
    result['density_fitting'] = _density_fitting_metadata(task_spec, mf, cas_result)
    result['scf_algorithm'] = task_spec.runtime.scf_algorithm
    result['reference'] = _molecular_reference_label(task_spec)
    if task_spec.method.name in ('casci', 'casscf'):
        result['initial_reference'] = _molecular_reference_label(task_spec)
        result['cas_reference'] = _cas_reference_label(task_spec)
        result['cas_spin_adapted'] = bool(task_spec.method.restricted)
    result['reference_energy'] = float(reference_energy)
    result['reference_converged'] = bool(mf.converged)
    reference_controls_task = _reference_convergence_controls_task(task_spec)
    result['reference_status'] = {
        'role': (
            'initial_orbital_guess'
            if task_spec.method.name == 'casscf'
            else 'mean_field_reference'
        ),
        'converged': bool(mf.converged),
        'affects_task_status': reference_controls_task,
    }
    result['requested_outputs'] = list(outputs)
    if task_spec.method.name in ('casci', 'casscf'):
        for field in (
            'state_energies',
            'excitation_energies',
            'requested_root_count',
            'computed_root_count',
            'targeted_roots_complete',
            'state_average_energy',
            'state_average_weights',
            'orbital_optimization_mode',
            'state_sector',
        ):
            if cas_result.get(field) is not None:
                state_results[field] = copy.deepcopy(cas_result[field])
    result.update(state_results)
    if initial_state_summary:
        result['initial_state'] = initial_state_summary
    if correlation_energy is not None:
        result['correlation_energy'] = float(correlation_energy)
    if triples_correction is not None:
        result['triples_correction'] = float(triples_correction)
    if post_hf is not None:
        result['solver_converged'] = bool(getattr(post_hf, 'converged', True))
        result['converged'] = _molecular_task_converged(
            task_spec,
            reference_converged=bool(mf.converged),
            solver_converged=result['solver_converged'],
        )
    deferred = set(deferred_modules)
    orbital_requested = (
        task_spec.orbital_processing.enabled
        or task_spec.orbital_processing.localization_method not in ('', 'none')
    )
    active_space_requested = task_spec.active_space.enabled or task_spec.method.name in ('casci', 'casscf')
    if orbital_requested and 'molecular.orbital_processing' not in deferred:
        result['orbital_processing'] = build_orbital_processing_summary(mf, mol, task_spec.orbital_processing)
    if active_space_requested and 'molecular.active_space_audit' not in deferred:
        result['active_space'] = propose_active_space(
            mf,
            task_spec.active_space,
            post_hf=post_hf,
            mol=mol,
            localization_method=task_spec.orbital_processing.localization_method,
            localization_scope=task_spec.orbital_processing.localization_scope,
            orbital_ordering=task_spec.orbital_processing.orbital_ordering,
            orbital_order=task_spec.orbital_processing.orbital_order,
        )
        if isinstance(result['active_space'].get('audit'), dict) and isinstance(cas_result, dict):
            result['active_space']['audit']['executed_orbital_processing'] = copy.deepcopy(
                cas_result.get('orbital_provenance') or {}
            )
            dmrg_result = cas_result.get('dmrg_result') if isinstance(cas_result.get('dmrg_result'), dict) else {}
            if dmrg_result.get('orbital_ordering'):
                result['active_space']['audit']['orbital_ordering'] = copy.deepcopy(dmrg_result['orbital_ordering'])
    diagnostics_enabled = task_spec.workflow.module_config.get('molecular.correlation_diagnostics', {}).get('enabled', True)
    if diagnostics_enabled and 'molecular.correlation_diagnostics' not in deferred:
        scf_stability = build_scf_stability_summary(mf, task_spec=task_spec)
        result['scf_stability'] = scf_stability
        result['correlation_diagnostics'] = build_correlation_diagnostics(
            mf,
            mol,
            post_hf=post_hf,
            scf_stability=scf_stability,
            reference_energy=reference_energy,
            correlation_energy=correlation_energy,
            state_energies=result.get('state_energies'),
            state_sector=result.get('state_sector'),
            correlated_natural_occupations=(
                cas_result.get('natural_occupations')
                if isinstance(cas_result, dict)
                else None
            ),
            current_method=task_spec.method.name,
        )
    if task_spec.method.name in ('casci', 'casscf'):
        result['cas_result'] = cas_result
        dmrg_result_payload = cas_result.get('dmrg_result') if isinstance(cas_result, dict) else None
        if isinstance(dmrg_result_payload, dict):
            from ..providers.block2 import block2_compact_result_fields

            result.update(block2_compact_result_fields(dmrg_result_payload))
            for field in (
                'bond_dimension_plan',
                'entanglement_diagnostics',
                'symmetry_analysis',
                'state_energies',
                'excitation_energies',
                'state_average_energy',
                'state_average_weights',
                'recovery_recommendation',
            ):
                if dmrg_result_payload.get(field) is not None:
                    result[field] = copy.deepcopy(dmrg_result_payload[field])
            recommendation = dmrg_result_payload.get('entanglement_active_space_recommendation')
            if isinstance(recommendation, dict):
                result['entanglement_active_space_recommendation'] = copy.deepcopy(recommendation)
                candidate = recommendation.get('candidate_active_space')
                if (
                    recommendation.get('status') == 'approval_recommended'
                    and isinstance(candidate, dict)
                    and isinstance(result.get('active_space'), dict)
                ):
                    audit = result['active_space'].get('audit')
                    if isinstance(audit, dict):
                        audit.setdefault('candidate_active_spaces', []).append(copy.deepcopy(
                            candidate.get('audit', {}).get('candidate_active_spaces', [{}])[0]
                        ))
                        audit['entanglement_expansion_recommendation'] = copy.deepcopy(recommendation)
        if isinstance(dmrg_arrays, dict) and dmrg_arrays:
            result['_transient_dmrg_arrays'] = dmrg_arrays
        result['cas_ncas'] = cas_result.get('ncas')
        result['cas_nelecas'] = cas_result.get('nelecas')
        result['energy_kind'] = 'cas_energy'
        result['final_energy'] = float(energy)
        result['final_method'] = task_spec.method.name
        if isinstance(post_cas_results, dict) and post_cas_results:
            result['post_cas_results'] = post_cas_results
            sc_nevpt2_result = post_cas_results.get('sc_nevpt2')
            if isinstance(sc_nevpt2_result, dict) and sc_nevpt2_result.get('total_energy') is not None:
                result['final_energy'] = float(sc_nevpt2_result['total_energy'])
                result['final_method'] = '{0}+sc_nevpt2'.format(task_spec.method.name)
    if 'ccsd_labels' in outputs:
        from .ccsd_labels import capture_ccsd_labels
        from .script_helpers import _compose_raw_stdout
        result['_transient_ccsd_labels'] = capture_ccsd_labels(task_spec, mf, post_hf)
        result['raw_scf_output'] = scf_stdout.getvalue()
        result['raw_stdout'] = _compose_raw_stdout(result['raw_scf_output'], result.get('analysis_text', ''))
    one_particle_state = capture_reference_1rdm(task_spec, mf)
    if one_particle_state and one_particle_state.get('status') == 'failed':
        result.setdefault('evidence_errors', []).append(one_particle_state)
    elif one_particle_state:
        result['_transient_one_particle_state'] = one_particle_state
    if deferred.intersection(MOLECULAR_NUMERICAL_MODULES):
        result['_transient_module_context'] = {
            'mf': mf,
            'mol': mol,
            'post_hf': post_hf,
            'reference_energy': reference_energy,
            'correlation_energy': correlation_energy,
            'state_energies': result.get('state_energies'),
            'state_sector': result.get('state_sector'),
            'correlated_natural_occupations': (
                copy.deepcopy(cas_result.get('natural_occupations'))
                if isinstance(cas_result, dict)
                else None
            ),
            'current_method': task_spec.method.name,
        }
    return result


def _compiled_numerical_modules(state: Dict[str, Any], task_type: str) -> Tuple[str, ...]:
    configuration = state.get('workflow_configuration')
    if not isinstance(configuration, dict):
        return ()
    selected = {
        str(node.get('module_id') or '')
        for node in configuration.get('nodes', ())
        if isinstance(node, dict)
    }
    supported = NUMERICAL_MODULES_BY_TASK_TYPE.get(task_type, frozenset())
    return tuple(sorted(selected.intersection(supported)))


def _dmrg_convergence_details(results: Dict[str, Any]) -> Dict[str, Any]:
    dmrg_result = results.get('dmrg_result')
    if not isinstance(dmrg_result, dict):
        cas_result = results.get('cas_result')
        dmrg_result = cas_result.get('dmrg_result') if isinstance(cas_result, dict) else None
    if not isinstance(dmrg_result, dict):
        return {}
    convergence = dmrg_result.get('convergence')
    return copy.deepcopy(convergence) if isinstance(convergence, dict) else {}


def _unconverged_execution_error(task_spec: TaskSpec, results: Dict[str, Any]) -> Dict[str, Any]:
    if (
        results.get('reference_converged') is False
        and _reference_convergence_controls_task(task_spec)
    ):
        return _make_error(
            'execution',
            'scf_unconverged',
            'The mean-field reference did not converge.',
            details={'reference_converged': False},
        )

    solver_name = str(task_spec.solver.name or '').strip().lower().replace('-', '_')
    if solver_name in ('block2', 'dmrg'):
        solver_name = 'block2_dmrg'
    cas_result = results.get('cas_result')
    if isinstance(cas_result, dict) and cas_result.get('solver'):
        solver_name = str(cas_result['solver']).strip().lower().replace('-', '_')
    if solver_name == 'block2_dmrg':
        details = {'solver': solver_name, 'converged': False}
        details.update(_dmrg_convergence_details(results))
        return _make_error(
            'execution',
            'solver_unconverged',
            'block2 DMRG did not meet its energy and discarded-weight convergence criteria.',
            details=details,
        )

    method_name = str(task_spec.method.name or '').strip().lower()
    if method_name in ('casci', 'casscf'):
        return _make_error(
            'execution',
            'solver_unconverged',
            '{0} did not converge.'.format(method_name.upper()),
            details={'solver': method_name, 'converged': False},
        )
    if task_spec.task_type == 'model_hamiltonian' or method_name in ('mp2', 'ccsd', 'ccsd_t', 'fci'):
        solver_label = solver_name or method_name or 'correlated solver'
        return _make_error(
            'execution',
            'solver_unconverged',
            '{0} did not converge.'.format(solver_label.upper()),
            details={'solver': solver_label, 'converged': False},
        )
    return _make_error(
        'execution',
        'scf_unconverged',
        'SCF did not converge.',
        details={'converged': False},
    )


def runner(state: Dict[str, Any]) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    if state['validation_errors']:
        lifecycle = ensure_lifecycle(
            state.get('lifecycle'),
            'task',
            entity_id=state.get('run_id'),
        )
        if lifecycle['stage'] != 'blocked':
            lifecycle = transition_lifecycle(
                lifecycle,
                'validation_failed',
                details={'source': 'execution_gate'},
            )
        state['lifecycle'] = lifecycle
        state['execution_status'] = 'blocked'
        state['raw_stderr'] = '\n'.join(state['validation_errors'])
        _append_attempt(
            state,
            status=state['execution_status'],
            stage='validation',
            errors=state.get('errors'),
        )
        append_log(state, 'warning', 'workflow.execution_blocked', {
            'errors': list(state['validation_errors']),
        })
        return state

    task_spec = _effective_task_spec(state)
    state['lifecycle'] = prepare_task_for_execution(
        ensure_lifecycle(
            state.get('lifecycle'),
            'task',
            entity_id=state.get('run_id'),
        ),
        details={
            'attempt': int(state.get('retry_count') or 0) + 1,
            'task_type': task_spec.task_type,
            'method': task_spec.method.name,
            'solver': task_spec.solver.name,
        },
    )
    append_log(state, 'info', 'workflow.execution_started', {
        'task_type': task_spec.task_type,
        'method': task_spec.method.name,
        'job': task_spec.job.name,
        'solver': task_spec.solver.name,
    })
    try:
        deferred_modules = _compiled_numerical_modules(state, task_spec.task_type)
        results = _run_pyscf_task(
            task_spec,
            deferred_modules=deferred_modules,
            execution_directory=get_run_dir(state),
        )
    except Exception as exc:  # pragma: no cover - exercised only with PySCF installed
        from .molecular_dynamics import MolecularDynamicsExecutionError

        md_failure = None
        if isinstance(exc, MolecularDynamicsExecutionError):
            md_failure = copy.deepcopy(exc.diagnostics)
            write_text_artifact(
                state, 'raw_scf_output',
                retry_artifact_filename('log-pyscf-output', 'log', state.get('retry_count', 0)),
                exc.raw_scf_output, description='Full PySCF output from the failed MD trajectory',
            )
            write_result_json_artifact(
                state, 'molecular_md_failure',
                retry_artifact_filename('result-molecular-md-failure', 'json', state.get('retry_count', 0)),
                md_failure, description='Failed MD frame and last observed SCF iteration',
            )
            state['raw_scf_output'] = preview_text(exc.raw_scf_output)
        solver_name = str(task_spec.solver.name or '').strip().lower().replace('-', '_')
        if md_failure is not None:
            solver_name = 'rks'
        if solver_name in ('block2', 'dmrg'):
            solver_name = 'block2_dmrg'
        recovery = None
        if solver_name == 'block2_dmrg':
            from ..providers.block2 import classify_block2_exception
            recovery = classify_block2_exception(exc)
        if solver_name == 'dmet':
            from ..providers.libdmet.recovery import dmet_scf_recovery
            recovery = dmet_scf_recovery(
                task_spec.solver.options, exception_type=type(exc).__name__,
                traceback_text=traceback.format_exc(),
            )
            register_existing_artifact(
                state,
                'dmet_output_log',
                get_run_dir(state) / 'solver-dmet' / 'log-libdmet-output.log',
                mime_type='text/plain; charset=utf-8',
                description='Complete libDMET provider stdout and stderr log from the failed run',
            )
        error = _make_error(
            'execution',
            'resource_memory_exhausted' if recovery and recovery.get('failure_class') == 'memory_exhausted' else 'execution_exception',
            'PySCF execution failed: {0}'.format(exc),
            details={
                'traceback': traceback.format_exc(),
                'solver': solver_name or None,
                'recovery_recommendation': copy.deepcopy(recovery),
                **({'molecular_dynamics': md_failure} if md_failure is not None else {}),
            },
            exception_type=type(exc).__name__,
        )
        state['execution_status'] = 'failed'
        state['lifecycle'] = transition_lifecycle(
            state['lifecycle'],
            'execution_failed',
            details={'exception_type': type(exc).__name__},
        )
        _append_errors_to_state(state, [error])
        state['raw_stderr'] = error['message']
        _append_attempt(
            state,
            status=state['execution_status'],
            stage='execution',
            errors=[error],
        )
        append_log(state, 'error', 'workflow.execution_failed', {
            'error': error['message'],
            'exception_type': error.get('exception_type'),
        })
        return state

    module_context = results.pop('_transient_module_context', None)
    discard_runtime_context(state.pop('module_context_ref', None))
    if isinstance(module_context, dict) and module_context:
        state['module_context_ref'] = store_runtime_context(
            state.get('run_id'),
            state.get('retry_count'),
            module_context,
        )
    full_results = write_result_artifacts(state, task_spec, results)
    if deferred_modules:
        state['_module_results'] = full_results

    terminal_deferred = sorted(
        set(deferred_modules).intersection(TERMINAL_DEFERRED_NUMERICAL_MODULES)
    )
    if results.get('converged') and terminal_deferred:
        state['execution_status'] = 'running'
        append_log(state, 'info', 'workflow.reference_execution_succeeded', {
            'converged': True,
            'pending_numerical_modules': terminal_deferred,
        })
    elif results.get('converged'):
        state['execution_status'] = 'succeeded'
        state['lifecycle'] = transition_lifecycle(
            state['lifecycle'],
            'execution_succeeded',
        )
        _append_attempt(
            state,
            status=state['execution_status'],
            stage='execution',
        )
        append_log(state, 'info', 'workflow.execution_succeeded', {
            'converged': True,
        })
    else:
        unconverged_error = _unconverged_execution_error(task_spec, results)
        state['execution_status'] = 'unconverged'
        state['lifecycle'] = transition_lifecycle(
            state['lifecycle'],
            'execution_unconverged',
        )
        _append_errors_to_state(state, [unconverged_error])
        state['raw_stderr'] = unconverged_error['message']
        _append_attempt(
            state,
            status=state['execution_status'],
            stage='execution',
            errors=[unconverged_error],
        )
        append_log(state, 'warning', 'workflow.execution_unconverged', {
            'converged': False,
        })

    return state


def complete_deferred_numerical_execution(
    state: Dict[str, Any],
    task_spec: TaskSpec,
    *,
    converged: bool,
    label: str,
) -> Dict[str, Any]:
    """Finalize a task whose terminal numerical module runs after core.execution."""

    state = copy.deepcopy(state)
    if converged:
        state['execution_status'] = 'succeeded'
        state['lifecycle'] = transition_lifecycle(
            state['lifecycle'],
            'execution_succeeded',
            details={'terminal_module': str(label)},
        )
        _append_attempt(state, status='succeeded', stage='execution')
        append_log(state, 'info', 'workflow.execution_succeeded', {
            'converged': True,
            'terminal_module': str(label),
        })
        return state

    error = _make_error(
        'execution',
        'solver_unconverged',
        '{0} did not converge.'.format(label),
        details={
            'solver': str(task_spec.solver.name or '').strip().lower(),
            'converged': False,
            'terminal_module': str(label),
        },
    )
    state['execution_status'] = 'unconverged'
    state['lifecycle'] = transition_lifecycle(
        state['lifecycle'],
        'execution_unconverged',
        details={'terminal_module': str(label)},
    )
    _append_errors_to_state(state, [error])
    state['raw_stderr'] = error['message']
    _append_attempt(
        state,
        status='unconverged',
        stage='execution',
        errors=[error],
    )
    append_log(state, 'warning', 'workflow.execution_unconverged', {
        'converged': False,
        'terminal_module': str(label),
    })
    return state


def result_extractor(state: Dict[str, Any]) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    results = copy.deepcopy(state.get('structured_results') or {})
    compact_results = compact_structured_results(results) or results
    state['structured_results'] = compact_results
    state['compact_results'] = compact_results
    append_log(state, 'info', 'workflow.results_extracted', {
        'keys': sorted(compact_results.keys()),
    })
    return state


def result_analyst(state: Dict[str, Any]) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    locale = state.get('locale', 'en')
    status = state.get('execution_status')
    results = state.get('structured_results') or {}

    if status == 'blocked':
        if state.get('validation_errors'):
            state['analysis_summary'] = t(locale, 'summary_blocked_prefix', details=join_items(locale, state['validation_errors']))
        else:
            state['analysis_summary'] = t(locale, 'summary_blocked_invalid')
        append_message(state, role='assistant', kind='summary', content=state['analysis_summary'])
        append_log(state, 'warning', 'workflow.summary_created', {
            'status': status,
        })
        return state

    if status == 'failed':
        error_message = _latest_error_message(state, stage='execution') or state.get('raw_stderr') or t(locale, 'summary_failed_unknown')
        state['analysis_summary'] = t(locale, 'summary_failed', message=error_message)
        latest_error = state.get('errors', [])[-1] if state.get('errors') else {}
        details = latest_error.get('details') if isinstance(latest_error, dict) else None
        recovery = details.get('recovery_recommendation') if isinstance(details, dict) else None
        if isinstance(recovery, dict) and recovery.get('summary'):
            state['analysis_summary'] = join_items(
                locale,
                [state['analysis_summary'], str(recovery['summary'])],
            )
        append_message(state, role='assistant', kind='summary', content=state['analysis_summary'])
        append_log(state, 'warning', 'workflow.summary_created', {
            'status': status,
        })
        return state

    if status == 'unconverged':
        dmft_result = results.get('dmft_result') if isinstance(results.get('dmft_result'), dict) else {}
        dmft_method = str(
            dmft_result.get('method')
            or results.get('solver')
            or results.get('method')
            or ''
        ).strip().lower().replace('-', '_')
        if dmft_method in ('hf_dmft', 'gw_dmft'):
            dmft_label = 'GW+DMFT' if dmft_method == 'gw_dmft' else 'HF+DMFT'
            reference_label = 'DFT+GW' if dmft_method == 'gw_dmft' else 'HF'
            pieces = [
                '{0} did not converge within max_iterations={1}.'.format(
                    dmft_label,
                    dmft_result.get('max_iterations', 'n/a')
                ),
                '{0} reference completed={1}.'.format(
                    reference_label,
                    yes_no(locale, bool(results.get('reference_converged'))),
                ),
            ]
            if results.get('reference_energy') is not None:
                pieces.append(
                    '{0} mean-field reference energy={1} {2}.'.format(
                        reference_label,
                        results['reference_energy'],
                        results.get('energy_unit') or 'Ha/cell',
                    )
                )
            if dmft_result.get('impurity_solver'):
                pieces.append('Impurity solver={0}.'.format(dmft_result['impurity_solver']))
            if dmft_result.get('nval') is not None:
                ncore = int(dmft_result.get('ncore') or 0)
                pieces.append(
                    'Correlated window=[{0}, {1}); bath orbitals={2} ({3}).'.format(
                        ncore,
                        int(dmft_result['nval']),
                        dmft_result.get('nbath', 'n/a'),
                        dmft_result.get('bath_discretization', 'n/a'),
                    )
                )
            if dmft_result.get('convergence_tolerance') is not None:
                pieces.append(
                    'Hybridization tolerance={0}.'.format(
                        dmft_result['convergence_tolerance']
                    )
                )
            pieces.extend((
                'No {0} total energy is available from the current adapter.'.format(dmft_label),
                'Next: increase the DMFT iteration limit or review the approved correlated '
                'subspace and bath discretization before trusting self-energy or spectral results.',
            ))
            state['analysis_summary'] = ' '.join(pieces)
            append_message(state, role='assistant', kind='summary', content=state['analysis_summary'], metadata={
                'structured_results': copy.deepcopy(results),
                'warning': 'unconverged',
            })
            append_log(state, 'warning', 'workflow.summary_created', {
                'status': status,
                'solver': dmft_method,
            })
            return state
        latest_error = state.get('errors', [])[-1] if state.get('errors') else {}
        if latest_error.get('code') == 'solver_unconverged':
            cas_result = results.get('cas_result') if isinstance(results.get('cas_result'), dict) else {}
            solver_label = (
                cas_result.get('solver')
                or results.get('solver')
                or results.get('method')
                or 'solver'
            )
            pieces = [t(locale, 'summary_solver_unconverged_intro', solver=str(solver_label).upper())]
        else:
            pieces = [t(locale, 'summary_unconverged_intro')]
        if results.get('energy') is not None:
            pieces.append(t(locale, 'summary_current_energy', value=results['energy']))
        if results.get('homo') is not None and results.get('lumo') is not None:
            pieces.append(t(locale, 'summary_homo_lumo_gap', homo=results['homo'], lumo=results['lumo'], gap=results['gap']))
        if results.get('dipole') is not None:
            dipole = ', '.join('{0:.6f}'.format(value) for value in results['dipole'])
            pieces.append(t(locale, 'summary_dipole', value=dipole))
        recovery = results.get('recovery_recommendation')
        if not isinstance(recovery, dict):
            dmrg_result = results.get('dmrg_result') if isinstance(results.get('dmrg_result'), dict) else {}
            recovery = dmrg_result.get('recovery_recommendation')
        if isinstance(recovery, dict) and recovery.get('summary'):
            pieces.append(str(recovery['summary']))
        state['analysis_summary'] = join_items(locale, pieces)
        append_message(state, role='assistant', kind='summary', content=state['analysis_summary'], metadata={
            'structured_results': copy.deepcopy(results),
            'warning': 'unconverged',
        })
        append_log(state, 'warning', 'workflow.summary_created', {
            'status': status,
        })
        return state

    if status != 'succeeded':
        state['analysis_summary'] = t(locale, 'summary_unknown_status', status=status)
        append_message(state, role='assistant', kind='summary', content=state['analysis_summary'])
        append_log(state, 'warning', 'workflow.summary_created', {
            'status': status,
        })
        return state

    if results.get('task_type') == 'model_hamiltonian':
        energy_unit = str(results.get('energy_unit') or 'a.u.')
        is_dmet = str(results.get('solver') or '').strip().lower() == 'dmet'
        pieces = [
            'Model Hamiltonian {0} completed.'.format(str(results.get('solver') or 'solver').upper()),
        ]
        if results.get('representation'):
            pieces.append('representation={0}'.format(results['representation']))
        if results.get('model'):
            pieces.append('model={0}'.format(results['model']))
        if results.get('norb') is not None:
            pieces.append('norb={0}'.format(results['norb']))
        if results.get('nelec') is not None:
            pieces.append('nelec={0}'.format(tuple(results['nelec'])))
        if results.get('electrons_per_cell') is not None:
            pieces.append('electrons_per_cell={0}'.format(results['electrons_per_cell']))
        if results.get('kmesh') is not None:
            pieces.append('kmesh={0}'.format(tuple(results['kmesh'])))
        if results.get('energy') is not None:
            energy_suffix = '/cell' if results.get('energy_kind') == 'noninteracting_band_energy_per_cell' else ''
            pieces.append('energy={0} {1}{2}'.format(results['energy'], energy_unit, energy_suffix))
        if results.get('energy_per_site') is not None:
            pieces.append('energy_per_site={0} {1}/site'.format(results['energy_per_site'], energy_unit))
        if not is_dmet and results.get('reference_energy') is not None:
            pieces.append('reference_energy={0} {1}'.format(results['reference_energy'], energy_unit))
        if results.get('site_count') is not None and results.get('bond_count') is not None:
            pieces.append('sites={0}, bonds={1}'.format(results['site_count'], results['bond_count']))
        if results.get('band_gap') is not None:
            pieces.append('band_gap={0} {1}'.format(results['band_gap'], energy_unit))
        if results.get('fermi_energy') is not None:
            pieces.append('fermi_energy={0} {1}'.format(results['fermi_energy'], energy_unit))
        if results.get('is_metal') is not None:
            pieces.append('is_metal={0}'.format(results['is_metal']))
        if results.get('energy_level_count') is not None:
            spectrum_method = str(results.get('energy_spectrum_method') or results.get('solver') or 'solver').upper()
            pieces.append('{0}_energy_levels={1} saved as artifacts'.format(spectrum_method, results['energy_level_count']))
        elif results.get('energy_spectrum_unavailable_reason'):
            pieces.append('energy spectrum not saved: {0}'.format(results['energy_spectrum_unavailable_reason']))
        if results.get('computed_root_count') is not None:
            pieces.append('targeted_roots={0}/{1} completed'.format(
                results['computed_root_count'],
                results.get('requested_root_count', results['computed_root_count']),
            ))
        if results.get('missing_requested_outputs'):
            pieces.append('missing_requested_outputs={0}'.format(','.join(results['missing_requested_outputs'])))
        state['analysis_summary'] = '; '.join(pieces)
        append_message(state, role='assistant', kind='summary', content=state['analysis_summary'], metadata={
            'structured_results': copy.deepcopy(results),
        })
        append_log(state, 'info', 'workflow.summary_created', {
            'status': status,
            'task_type': 'model_hamiltonian',
        })
        return state

    if results.get('task_type') == 'periodic':
        dmft_result = results.get('dmft_result') if isinstance(results.get('dmft_result'), dict) else {}
        dmft_method = str(
            dmft_result.get('method')
            or results.get('solver')
            or results.get('method')
            or ''
        ).strip().lower().replace('-', '_')
        is_dmft = dmft_method in ('hf_dmft', 'gw_dmft')
        is_gw = dmft_method in ('gw', 'gw_dmft') or isinstance(results.get('gw_result'), dict)
        method_label = (
            'GW+DMFT' if dmft_method == 'gw_dmft'
            else 'HF+DMFT' if dmft_method == 'hf_dmft'
            else 'GW' if is_gw
            else str(results.get('method') or 'SCF').upper()
        )
        pieces = [
            'Periodic {0} completed.'.format(method_label),
        ]
        structure = results.get('periodic_structure') if isinstance(results.get('periodic_structure'), dict) else {}
        if structure.get('formula'):
            pieces.append('cell={0}'.format(structure['formula']))
        if structure.get('cell_role') == 'seekpath_standardized_primitive':
            standardization = (
                results.get('periodic_standardization')
                or results.get('seekpath_standardization')
                or {}
            )
            pieces.append('cell_role=seekpath_standardized_primitive')
            if standardization.get('spacegroup_international'):
                pieces.append('spacegroup={0}'.format(standardization['spacegroup_international']))
        if results.get('kmesh'):
            pieces.append('kmesh={0}'.format(tuple(results['kmesh'])))
        if is_dmft:
            reference_label = 'DFT_GW' if dmft_method == 'gw_dmft' else 'HF'
            pieces.append('{0}_reference_completed={1}'.format(reference_label,
                yes_no(locale, bool(results.get('reference_converged')))
            ))
            if results.get('reference_energy') is not None:
                pieces.append('{0}_mean_field_energy={1} {2}'.format(
                    reference_label,
                    results['reference_energy'],
                    results.get('energy_unit') or 'Ha/cell',
                ))
            pieces.append('DMFT_converged={0}'.format(
                yes_no(locale, bool(dmft_result.get('converged')))
            ))
            if dmft_result.get('impurity_solver'):
                pieces.append('impurity_solver={0}'.format(dmft_result['impurity_solver']))
            if dmft_result.get('nval') is not None:
                ncore = int(dmft_result.get('ncore') or 0)
                pieces.append('correlated_window=[{0}, {1})'.format(
                    ncore,
                    int(dmft_result['nval']),
                ))
            if dmft_result.get('nbath') is not None:
                pieces.append('bath={0} ({1})'.format(
                    dmft_result['nbath'],
                    dmft_result.get('bath_discretization') or 'unspecified',
                ))
            pieces.append('DMFT_total_energy=unavailable')
        elif is_gw:
            gw_result = results.get('gw_result') if isinstance(results.get('gw_result'), dict) else {}
            pieces.append('GW_total_energy=unavailable')
            if gw_result.get('quasiparticle_gap') is not None:
                pieces.append('quasiparticle_gap={0} Ha'.format(gw_result['quasiparticle_gap']))
        elif results.get('energy') is not None:
            pieces.append('energy={0} {1}'.format(results['energy'], results.get('energy_unit') or 'Ha/cell'))
        if results.get('band_gap') is not None:
            pieces.append('mean_field_gap={0} Ha'.format(results['band_gap']))
        if results.get('fermi_energy') is not None:
            pieces.append('fermi_estimate={0} Ha'.format(results['fermi_energy']))
        if results.get('band_structure_status') == 'completed':
            pieces.append('band_path_mode={0}; band_path={1}; points={2}'.format(
                results.get('band_path_mode'),
                results.get('band_path'),
                results.get('band_path_point_count'),
            ))
        elif results.get('band_structure_status'):
            pieces.append('band_structure={0}'.format(results['band_structure_status']))
        state['analysis_summary'] = '; '.join(pieces)
        append_message(state, role='assistant', kind='summary', content=state['analysis_summary'], metadata={
            'structured_results': copy.deepcopy(results),
        })
        append_log(state, 'info', 'workflow.summary_created', {
            'status': status,
            'task_type': 'periodic',
        })
        return state

    pieces = []
    if results.get('method'):
        pieces.append('Method={0}'.format(str(results['method']).upper()))
    if results.get('solver'):
        pieces.append('Solver={0}'.format(str(results['solver']).upper()))
    if results.get('method') in ('casci', 'casscf'):
        if results.get('initial_reference'):
            pieces.append('initial_reference={0}'.format(
                str(results['initial_reference']).upper()
            ))
        if results.get('cas_reference'):
            cas_suffix = ' (spin-adapted)' if results.get('cas_spin_adapted') else ''
            pieces.append('CAS_reference={0}{1}'.format(
                str(results['cas_reference']).upper(),
                cas_suffix,
            ))
    density_fitting = results.get('density_fitting') or {}
    if isinstance(density_fitting, dict) and density_fitting.get('enabled'):
        auxbasis = density_fitting.get('auxbasis') or 'auto'
        stages = '+'.join(density_fitting.get('applied_to') or ['scf']).upper()
        pieces.append('density_fitting={0}(auxbasis={1})'.format(stages, auxbasis))
    if results.get('scf_algorithm') not in (None, '', 'standard'):
        pieces.append('SCF_algorithm={0}'.format(str(results['scf_algorithm']).upper()))
    if results.get('method') in ('casci', 'casscf'):
        if results.get('reference_converged') is not None:
            reference_note = ''
            reference_status = results.get('reference_status')
            if (
                results.get('reference_converged') is False
                and isinstance(reference_status, dict)
                and reference_status.get('affects_task_status') is False
            ):
                reference_note = ' (initial-guess warning only)'
            pieces.append('initial_reference_converged={0}{1}'.format(
                yes_no(locale, results['reference_converged']),
                reference_note,
            ))
    elif results.get('converged') is not None:
        pieces.append(t(locale, 'summary_scf_converged', value=yes_no(locale, results['converged'])))
    if results.get('energy') is not None:
        pieces.append(t(locale, 'summary_total_energy', value=results['energy']))
    if results.get('reference_energy') is not None:
        pieces.append('reference_energy={0}'.format(results['reference_energy']))
    if results.get('cas_result'):
        cas_result = results.get('cas_result') or {}
        pieces.append('active_space=CAS({0}, {1})'.format(
            cas_result.get('ncas'),
            cas_result.get('nelecas'),
        ))
        pieces.append('CAS converged={0}'.format(yes_no(locale, bool(cas_result.get('converged', True)))))
        post_cas_results = results.get('post_cas_results') if isinstance(results.get('post_cas_results'), dict) else {}
        sc_nevpt2 = post_cas_results.get('sc_nevpt2') if isinstance(post_cas_results, dict) else None
        if isinstance(sc_nevpt2, dict):
            pieces.append('SC-NEVPT2 correction={0}; total_energy={1}'.format(
                sc_nevpt2.get('correction_energy'),
                sc_nevpt2.get('total_energy'),
            ))
    elif results.get('active_space'):
        active_space = results.get('active_space') or {}
        if active_space.get('ncas') is not None:
            pieces.append('active_space_candidate=CAS({0}, {1})'.format(
                active_space.get('ncas'),
                active_space.get('nelecas'),
            ))
        audit_summary = active_space.get('audit_summary') if isinstance(active_space, dict) else {}
        if isinstance(audit_summary, dict) and audit_summary.get('status'):
            pieces.append('active_space_audit={0}; next={1}'.format(
                audit_summary.get('status'),
                audit_summary.get('recommended_next_step'),
            ))
    orbital_processing = results.get('orbital_processing') or {}
    if isinstance(orbital_processing, dict) and orbital_processing.get('enabled'):
        localization_method = orbital_processing.get('localization_method') or 'none'
        localization_status = orbital_processing.get('localization_status') or 'unknown'
        if localization_method not in ('', 'none'):
            pieces.append('orbital_localization={0}({1})'.format(localization_method, localization_status))
        elif orbital_processing.get('use_natural_orbitals'):
            pieces.append('orbital_processing=natural_orbital_summary')
    diagnostics = results.get('correlation_diagnostics') or {}
    if diagnostics.get('warnings'):
        pieces.append('diagnostics={0}'.format(', '.join(str(item) for item in diagnostics['warnings'])))
    risk = diagnostics.get('molecular_correlation_risk') if isinstance(diagnostics, dict) else {}
    if isinstance(risk, dict) and risk.get('level'):
        pieces.append('molecular_correlation_risk={0}'.format(risk.get('level')))
    scf_stability = results.get('scf_stability')
    if not isinstance(scf_stability, dict) and isinstance(diagnostics, dict):
        scf_stability = diagnostics.get('scf_stability')
    if isinstance(scf_stability, dict) and scf_stability.get('stable') is not None:
        pieces.append('scf_stability={0}'.format('stable' if scf_stability.get('stable') else 'unstable'))
    if results.get('homo') is not None and results.get('lumo') is not None:
        pieces.append(t(locale, 'summary_homo_lumo_gap', homo=results['homo'], lumo=results['lumo'], gap=results['gap']))
    if results.get('dipole') is not None:
        dipole = ', '.join('{0:.6f}'.format(value) for value in results['dipole'])
        pieces.append(t(locale, 'summary_dipole', value=dipole))
    state['analysis_summary'] = join_items(locale, pieces) if pieces else t(locale, 'summary_success_default')
    append_message(state, role='assistant', kind='summary', content=state['analysis_summary'], metadata={
        'structured_results': copy.deepcopy(results),
    })
    append_log(state, 'info', 'workflow.summary_created', {
        'status': status,
    })
    return state


def repair_or_retry(state: Dict[str, Any]) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    locale = state.get('locale', 'en')
    if state['execution_status'] not in ('failed', 'unconverged'):
        return state
    if state['retry_count'] >= state.get('max_retries', 0):
        return state

    task_spec = task_spec_from_dict(state['task_spec'])
    is_convergence_issue = state.get('execution_status') == 'unconverged'

    should_retry = False
    original_spec = copy.deepcopy(state['task_spec'])
    results = state.get('structured_results') if isinstance(state.get('structured_results'), dict) else {}
    execution_error_codes = {
        str(item.get('code') or '')
        for item in state.get('errors', [])
        if isinstance(item, dict) and item.get('stage') == 'execution'
    }
    reference_unconverged = bool(
        task_spec.task_type == 'molecular'
        and _reference_convergence_controls_task(task_spec)
        and (
            results.get('reference_converged') is False
            or 'scf_unconverged' in execution_error_codes
        )
    )

    if reference_unconverged and is_convergence_issue:
        original_max_cycle = task_spec.runtime.max_cycle
        task_spec.runtime.max_cycle *= 2
        should_retry = task_spec.runtime.max_cycle != original_max_cycle

        # Reuse this attempt's last density through the existing restart path.
        reference_state = results.get('one_particle_state') or {}
        density_artifact = reference_state.get('data_artifact') or {}
        if (
            density_artifact.get('kind') == 'one_particle_state'
            and density_artifact.get('path')
            and Path(density_artifact['path']).is_file()
        ):
            task_spec.initial_state = InitialStateSpec(
                mode='projected_1rdm',
                source_artifact=copy.deepcopy(density_artifact),
            )

    solver_name = str(task_spec.solver.name or '').strip().lower().replace('-', '_')
    if solver_name in ('block2', 'dmrg'):
        solver_name = 'block2_dmrg'
    if (
        not reference_unconverged
        and solver_name == 'block2_dmrg'
        and state.get('execution_status') == 'unconverged'
    ):
        dmrg_result = results.get('dmrg_result') if isinstance(results.get('dmrg_result'), dict) else None
        if not isinstance(dmrg_result, dict):
            cas_result = results.get('cas_result') if isinstance(results.get('cas_result'), dict) else {}
            dmrg_result = cas_result.get('dmrg_result') if isinstance(cas_result.get('dmrg_result'), dict) else {}
        recommendation = dmrg_result.get('recovery_recommendation') if isinstance(dmrg_result, dict) else None
        if isinstance(recommendation, dict) and recommendation.get('automatic_retry_safe'):
            from ..providers.block2 import escalated_block2_options
            task_spec.solver.options = escalated_block2_options(
                task_spec.solver.options,
                recommendation,
            )
            checkpoint = dmrg_result.get('checkpoint_manifest')
            if isinstance(checkpoint, dict) and checkpoint.get('files'):
                task_spec.solver.options.update({
                    'restart_manifest': copy.deepcopy(checkpoint),
                    'restart_required': True,
                    'restart_provenance': {
                        'selection': 'same_task_automatic_dmrg_recovery',
                        'source_retry_count': state.get('retry_count', 0),
                    },
                })
            should_retry = True

    if solver_name in ('hf_dmft', 'gw_dmft') and state.get('execution_status') == 'unconverged':
        solver_options = copy.deepcopy(task_spec.solver.options or {})
        from ..providers.fcdmft.contracts import normalize_hf_dmft_options, normalize_gw_dmft_options
        normalize = normalize_gw_dmft_options if solver_name == 'gw_dmft' else normalize_hf_dmft_options
        current_iterations = normalize(solver_options)['max_iterations']
        increased_iterations = max(2, current_iterations * 2)
        if increased_iterations != current_iterations:
            solver_options['max_iterations'] = increased_iterations
            task_spec.solver.options = solver_options
            should_retry = True

    if (
        is_convergence_issue
        and not reference_unconverged
        and solver_name not in ('block2_dmrg', 'dmet', 'hf_dmft', 'gw_dmft')
        and (task_spec.task_type == 'molecular' or task_spec.method.name in ('hf', 'dft'))
    ):
        original_max_cycle = task_spec.runtime.max_cycle
        task_spec.runtime.max_cycle *= 2
        if task_spec.runtime.max_cycle != original_max_cycle:
            should_retry = True


    if not should_retry:
        append_log(state, 'info', 'workflow.retry_skipped', {
            'reason': 'no_applicable_repair_strategy',
            'status': state.get('execution_status'),
        })
        return state

    state['retry_count'] += 1
    state['task_spec'] = task_spec_to_dict(task_spec)
    state['execution_status'] = 'pending'
    state['raw_stderr'] = ''
    state['errors'] = [
        error for error in state.get('errors', [])
        if not isinstance(error, dict) or error.get('stage') != 'execution'
    ]
    state['structured_results'] = None
    state['compact_results'] = {}
    state['raw_stdout'] = ''
    state['raw_scf_output'] = ''
    state['analysis_text'] = ''
    discard_runtime_context(state.pop('module_context_ref', None))
    state.pop('_module_results', None)
    append_log(state, 'info', 'workflow.retry_scheduled', {
        'previous_task_spec': original_spec,
        'retry_task_spec': copy.deepcopy(state['task_spec']),
        'retry_count': state['retry_count'],
        'max_cycle': task_spec.runtime.max_cycle,
        'scf_algorithm': task_spec.runtime.scf_algorithm,
        'initial_state': copy.deepcopy(state['task_spec']['initial_state']),
        'solver': solver_name or None,
        'solver_options': (
            copy.deepcopy(task_spec.solver.options)
            if solver_name in ('block2_dmrg', 'hf_dmft', 'gw_dmft')
            else None
        ),
    })
    append_message(state, role='system', kind='retry', content=t(locale, 'retry_message'), metadata={
        'retry_count': state['retry_count'],
        'runtime': copy.deepcopy(state['task_spec']['runtime']),
    })
    return state


def _active_space_approval_contract(
    state: Dict[str, Any],
    task_spec: TaskSpec,
) -> Optional[Dict[str, Any]]:
    structured_results = state.get('structured_results') or {}
    active_result = structured_results.get('active_space') if isinstance(structured_results, dict) else None
    active_spec = task_spec.active_space
    source = None

    if isinstance(active_result, dict) and active_result.get('audit') and not active_result.get('approved'):
        contract_source = active_result
        source = 'active_space_audit'
    else:
        error_codes = {
            str(item.get('code'))
            for item in state.get('errors', [])
            if isinstance(item, dict) and item.get('code')
        }
        is_blocked_contract = (
            'active_space_not_approved' in error_codes
            and task_spec.method.name in ('casci', 'casscf')
            and active_spec.ncas is not None
            and active_spec.nelecas is not None
            and not active_spec.approved
        )
        if not is_blocked_contract:
            return None
        contract_source = task_spec_to_dict(task_spec).get('active_space') or {}
        source = 'blocked_active_space_contract'

    ncas = contract_source.get('ncas')
    nelecas = contract_source.get('nelecas')
    if ncas is None or nelecas is None:
        return None

    audit = active_result.get('audit') if isinstance(active_result, dict) else {}
    selection_parameters = audit.get('selection_parameters') if isinstance(audit, dict) else {}
    if not isinstance(selection_parameters, dict):
        selection_parameters = {}
    target_method = contract_source.get('target_method') or task_spec.method.name
    if target_method not in ('casci', 'casscf'):
        target_method = 'casscf'
    target_solver = contract_source.get('target_solver')
    if target_solver is None and task_spec.solver.name == 'block2_dmrg':
        target_solver = 'block2_dmrg'

    return {
        'type': 'active_space',
        'status': 'requires_user_review',
        'source': source,
        'reason_code': 'active_space_audit_ready' if source == 'active_space_audit' else 'active_space_not_approved',
        'target_method': target_method,
        'target_solver': target_solver,
        'runtime_contract': {
            'max_cycle': task_spec.runtime.max_cycle,
            'conv_tol': task_spec.runtime.conv_tol,
            'verbose': task_spec.runtime.verbose,
            'scf_algorithm': task_spec.runtime.scf_algorithm,
        },
        'probe_reference_status': (
            {
                'reference_converged': structured_results.get('reference_converged'),
                'retry_count': int(state.get('retry_count') or 0),
            }
            if state.get('calculation_role') == 'active_space_probe'
            else None
        ),
        'active_space_contract': {
            'enabled': True,
            'selection_method': contract_source.get('selection_method') or 'manual',
            'ncas': ncas,
            'nelecas': copy.deepcopy(nelecas),
            'orbital_indices': copy.deepcopy(contract_source.get('orbital_indices') or []),
            'avas_targets': copy.deepcopy(contract_source.get('avas_targets') or []),
            'avas_threshold': (
                contract_source.get('avas_parameters', {}).get('threshold')
                if isinstance(contract_source.get('avas_parameters'), dict)
                else None
            ) or selection_parameters.get('avas_threshold') or active_spec.avas_threshold,
            'initial_mo_coeff': copy.deepcopy(contract_source.get('initial_mo_coeff')),
            'target_method': target_method,
            'target_solver': target_solver,
            'target_solver_options': copy.deepcopy(
                contract_source.get('target_solver_options') or active_spec.target_solver_options or {}
            ),
            'approved': False,
        },
    }


def _correlated_subspace_approval_contract(
    state: Dict[str, Any],
    task_spec: TaskSpec,
) -> Optional[Dict[str, Any]]:
    if task_spec.task_type != 'periodic' or task_spec.embedding.approved:
        return None
    structured_results = state.get('structured_results') or {}
    preparation = (
        structured_results.get('embedding_preparation')
        if isinstance(structured_results, dict)
        else None
    )
    if not isinstance(preparation, dict) or preparation.get('status') != 'review_required':
        return None
    audit = preparation.get('audit')
    patch = preparation.get('task_spec_patch')
    if not isinstance(audit, dict) or not isinstance(patch, dict):
        return None
    embedding_patch = patch.get('embedding')
    if not isinstance(embedding_patch, dict) or not embedding_patch.get('correlated_orbital_indices'):
        return None
    solver_patch = patch.get('solver') if isinstance(patch.get('solver'), dict) else {}
    target_solver = str(solver_patch.get('name') or task_spec.solver.name or '').strip().lower().replace('-', '_')
    if target_solver not in ('hf_dmft', 'gw_dmft'):
        return None
    target_method = 'dft' if target_solver == 'gw_dmft' else 'hf'
    return {
        'type': 'correlated_subspace',
        'status': 'requires_user_review',
        'source': 'correlated_subspace_audit',
        'reason_code': 'correlated_subspace_audit_ready',
        'target_method': target_method,
        'target_solver': target_solver,
        'task_spec_patch': copy.deepcopy(patch),
        'correlated_subspace_review': {
            'provider': preparation.get('provider') or 'fcdmft',
            'localization_method': preparation.get('localization_method'),
            'localized_orbital_count': preparation.get('localized_orbital_count'),
            'estimated_eri_memory_mb': preparation.get('estimated_eri_memory_mb'),
            'audit': copy.deepcopy(audit),
        },
    }


def _task_approval_contract(
    state: Dict[str, Any],
    task_spec: TaskSpec,
) -> Optional[Dict[str, Any]]:
    return (
        _active_space_approval_contract(state, task_spec)
        or _correlated_subspace_approval_contract(state, task_spec)
    )


def _build_task_report(state: Dict[str, Any]) -> Dict[str, Any]:
    task_spec = task_spec_from_dict(state['task_spec'])
    return TaskReport(
        run_id=state.get('run_id'),
        work_dir=str(state.get('work_dir') or ''),
        channel=str(state.get('channel') or ''),
        calculation_role=str(state.get('calculation_role') or 'calculation'),
        task_spec=task_spec_to_dict(task_spec),
        generated_input=state.get('generated_input'),
        execution_status=str(state.get('execution_status') or 'pending'),
        structured_results=copy.deepcopy(state.get('structured_results')),
        compact_results=copy.deepcopy(state.get('compact_results', {})),
        analysis_summary=str(state.get('analysis_summary') or ''),
        attempts=copy.deepcopy(state.get('attempts', [])),
        errors=copy.deepcopy(state.get('errors', [])),
        warnings=copy.deepcopy(state.get('warnings', [])),
        applied_defaults=copy.deepcopy(state.get('applied_defaults', [])),
        validation_errors=list(state.get('validation_errors', [])),
        clarification_questions=list(state.get('clarification_questions', [])),
        approval=_task_approval_contract(state, task_spec),
        raw_stdout=str(state.get('raw_stdout') or ''),
        raw_scf_output=str(state.get('raw_scf_output') or ''),
        analysis_text=str(state.get('analysis_text') or ''),
        raw_stderr=str(state.get('raw_stderr') or ''),
        retry_count=int(state.get('retry_count') or 0),
        max_retries=int(state.get('max_retries') or 0),
        artifacts=copy.deepcopy(state.get('artifacts', [])),
        messages=copy.deepcopy(state.get('messages', [])),
        logs=copy.deepcopy(state.get('logs', [])),
        workflow_configuration=copy.deepcopy(state.get('workflow_configuration', {})),
        workflow_provenance=copy.deepcopy(state.get('workflow_provenance', {})),
        module_execution_trace=copy.deepcopy(state.get('module_execution_trace', [])),
        module_runtime_observations=copy.deepcopy(state.get('module_runtime_observations', [])),
        gate_configuration=copy.deepcopy(state.get('gate_configuration', {})),
        gate_provenance=copy.deepcopy(state.get('gate_provenance', {})),
        gate_decisions=copy.deepcopy(state.get('gate_decisions', [])),
        gate_execution_trace=copy.deepcopy(state.get('gate_execution_trace', [])),
        lifecycle=copy.deepcopy(state.get('lifecycle', {})),
    ).to_dict()


def sanitize_task_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare a report snapshot, including late workflow evidence, for strict JSON."""
    from ..serialization import canonical_json, sanitize_result_json

    # Never repair invalid input contracts as though they were solver output.
    canonical_json(report['task_spec'])
    report, warnings = sanitize_result_json(report)
    recorded = report.setdefault('warnings', [])
    for warning in warnings:
        if warning not in recorded:
            recorded.append(warning)
    return report


def task_reporter(state: Dict[str, Any]) -> Dict[str, Any]:
    from ..artifacts.task_storage import compact_task_payload
    from .artifacts import get_run_dir

    state = compact_task_payload(state, get_run_dir(state))
    task_spec = task_spec_from_dict(state['task_spec'])
    approval = _task_approval_contract(state, task_spec)
    lifecycle = ensure_lifecycle(
        state.get('lifecycle'),
        'task',
        entity_id=state.get('run_id'),
    )
    if (
        approval is not None
        or lifecycle_requires_review(state.get('gate_decisions') or [])
    ) and lifecycle['stage'] in ('prepared', 'blocked', 'failed', 'unconverged', 'succeeded'):
        lifecycle = transition_lifecycle(
            lifecycle,
            'review_requested',
            details={
                'approval_type': approval.get('type') if isinstance(approval, dict) else None,
                'gate_review': lifecycle_requires_review(state.get('gate_decisions') or []),
            },
        )
    state['lifecycle'] = lifecycle
    append_log(state, 'info', 'workflow.finalized', {
        'status': state.get('execution_status'),
        'message_count': len(state.get('messages', [])),
        'log_count': len(state.get('logs', [])),
    })
    state['task_report'] = sanitize_task_report(_build_task_report(state))
    return state
