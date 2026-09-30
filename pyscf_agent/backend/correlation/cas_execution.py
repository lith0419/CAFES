from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from ...contracts import ActiveSpaceSpec, DensityFittingSpec, OrbitalProcessingSpec
from ..script_helpers import _safe_to_list


CAS_METHODS = ('casci', 'casscf')


def configure_fci_solver(solver: Any, nroots: int = 1) -> Any:
    """Configure a PySCF FCI solver for an explicit number of low-energy roots."""
    if solver is None:
        return solver
    for field, value in (
        ('nroots', max(1, int(nroots))),
        ('davidson_only', False),
    ):
        if hasattr(solver, field):
            try:
                setattr(solver, field, value)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(f'FCI solver rejected required setting {field}={value!r}') from exc
    return solver


def configure_single_root_fci_solver(solver: Any) -> Any:
    """Backward-compatible single-root FCI configuration helper."""
    return configure_fci_solver(solver, 1)


def state_target_configuration(
    solver_options: Optional[Dict[str, Any]],
    *,
    default_nroots: int = 1,
) -> Tuple[int, List[float]]:
    options = solver_options if isinstance(solver_options, dict) else {}
    raw_nroots = options.get('nroots', default_nroots)
    requested_nroots = max(1, int(raw_nroots))
    if requested_nroots == 1:
        return requested_nroots, []
    raw_weights = options.get('state_average_weights')
    if raw_weights is None:
        weights = [1.0 / requested_nroots] * requested_nroots
    else:
        weights = [float(value) for value in raw_weights]
    return requested_nroots, weights


def _state_energies_from_cas(mc: Any, kernel_energy: Any, nroots: int) -> List[float]:
    if nroots > 1 and getattr(mc, 'e_states', None) is not None:
        raw_energies = getattr(mc, 'e_states')
    else:
        raw_energies = getattr(mc, 'e_tot', kernel_energy)
    if hasattr(raw_energies, 'tolist'):
        raw_energies = raw_energies.tolist()
    if isinstance(raw_energies, (list, tuple)):
        return [float(value) for value in raw_energies]
    return [float(raw_energies)]


def _root_natural_occupations(
    mc: Any,
    root: int = 0,
) -> Optional[List[float]]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    ci_states = getattr(mc, 'ci', None)
    ci_state = ci_states
    if isinstance(ci_states, (list, tuple)):
        if root < 0 or root >= len(ci_states):
            return None
        states_make_rdm1 = getattr(mc.fcisolver, 'states_make_rdm1', None)
        if callable(states_make_rdm1):
            densities = states_make_rdm1(
                ci_states,
                int(mc.ncas),
                getattr(mc, 'nelecas'),
            )
            if root >= len(densities):
                return None
            density = densities[root]
        else:
            ci_state = ci_states[root]
            density = mc.fcisolver.make_rdm1(
                ci_state,
                int(mc.ncas),
                getattr(mc, 'nelecas'),
            )
    else:
        density = None
    if ci_state is None:
        return None
    if density is None:
        density = mc.fcisolver.make_rdm1(
            ci_state,
            int(mc.ncas),
            getattr(mc, 'nelecas'),
        )
    if isinstance(density, (list, tuple)) and len(density) == 2:
        density = np.asarray(density[0]) + np.asarray(density[1])
    density = np.asarray(density)
    occupations = np.linalg.eigvalsh(0.5 * (density + density.T.conj()))[::-1]
    return occupations.real.tolist()


def _active_orbital_indices_payload(active_space: ActiveSpaceSpec) -> Any:
    if isinstance(active_space.orbital_indices, dict):
        return {
            spin_label: list(active_space.orbital_indices.get(spin_label, []))
            for spin_label in ('alpha', 'beta')
            if active_space.orbital_indices.get(spin_label)
        }
    return list(active_space.orbital_indices)


def _active_orbital_spin_policy(active_space: ActiveSpaceSpec, unrestricted: bool) -> str:
    if unrestricted and isinstance(active_space.orbital_indices, dict):
        return 'spin_resolved_alpha_beta'
    if unrestricted:
        return 'shared_alpha_beta'
    return 'restricted'


def _nelecas_for_pyscf(nelecas: Any) -> Any:
    if isinstance(nelecas, tuple):
        return nelecas
    if isinstance(nelecas, list) and len(nelecas) == 2:
        return (int(nelecas[0]), int(nelecas[1]))
    return int(nelecas)


def _cas_orbital_indices_for_sort(active_space: ActiveSpaceSpec, unrestricted: bool) -> Any:
    if isinstance(active_space.orbital_indices, dict):
        return (
            list(active_space.orbital_indices.get('alpha', [])),
            list(active_space.orbital_indices.get('beta', [])),
        )
    indices = list(active_space.orbital_indices)
    if unrestricted:
        return (indices, indices)
    return indices


def _initial_mo_coeff(active_space: ActiveSpaceSpec, unrestricted: bool) -> Any:
    """Return an audit-provided orbital guess, currently for spin-adapted CAS."""
    payload = active_space.initial_mo_coeff
    if payload is None:
        return None
    if unrestricted:
        raise ValueError('Projected orbital initial guesses require a restricted or ROHF spin-adapted CAS reference.')
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel

        from ...artifacts.arrays import read_array

        matrix = np.asarray(read_array(payload), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('active_space.initial_mo_coeff must be a numeric two-dimensional orbital matrix.') from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError('active_space.initial_mo_coeff must be a nonempty two-dimensional orbital matrix.')
    return matrix


def _active_orbital_provenance(
    mf: Any,
    mol: Any,
    mo_coeff: Any,
    *,
    ncas: int,
    nelecas: Any,
    active_space: ActiveSpaceSpec,
    orbital_processing: Optional[OrbitalProcessingSpec],
) -> tuple[Any, Dict[str, Any]]:
    """Apply an explicitly requested localization to the fixed active block."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    spec = orbital_processing or OrbitalProcessingSpec()
    method = str(spec.localization_method or 'none').strip().lower()
    if method == 'pm':
        method = 'pipek_mezey'
    input_basis = (
        'avas' if active_space.selection_method == 'avas'
        else 'active_space.initial_mo_coeff'
    ) if active_space.initial_mo_coeff is not None else 'canonical'
    provenance: Dict[str, Any] = {
        'input_basis': input_basis,
        'localization_method': method,
        'localization_scope': str(spec.localization_scope or 'analysis'),
        'localization_applied': False,
        'orbital_ordering': str(spec.orbital_ordering or 'canonical'),
        'requested_orbital_order': list(spec.orbital_order or []),
    }
    if str(spec.localization_scope or 'analysis') != 'active_space' or method in ('', 'none'):
        provenance['output_basis'] = input_basis
        return mo_coeff, provenance

    total_active_electrons = sum(nelecas) if isinstance(nelecas, (list, tuple)) else int(nelecas)
    inactive_electrons = int(mol.nelectron) - int(total_active_electrons)
    if inactive_electrons < 0 or inactive_electrons % 2:
        raise ValueError('Active-space localization requires a paired inactive core.')
    ncore = inactive_electrons // 2
    matrix = np.asarray(mo_coeff, dtype=float)
    if matrix.ndim != 2 or ncore + int(ncas) > matrix.shape[1]:
        raise ValueError('Active-space localization received an incompatible molecular-orbital matrix.')
    active_columns = matrix[:, ncore:ncore + int(ncas)]
    from .orbital_localization import localize_orbital_block

    reference_rdm1 = None
    if spec.localization_occupation_thresholds:
        density = np.asarray(mf.make_rdm1(), dtype=float)
        if density.ndim == 3 and density.shape[0] == 2:
            density = density.sum(axis=0)
        overlap = mf.get_ovlp()
        reference_rdm1 = active_columns.T @ overlap @ density @ overlap @ active_columns
        provenance['localization_density_source'] = 'spin_summed_scf_reference_projected_into_active_space'
    localized, localization_details = localize_orbital_block(
        mol, active_columns, method=method,
        occupation_thresholds=spec.localization_occupation_thresholds,
        reference_rdm1=reference_rdm1,
    )
    transformed = np.array(matrix, copy=True)
    transformed[:, ncore:ncore + int(ncas)] = localized
    provenance.update({
        'output_basis': method + '_localized_active_space',
        'localization_applied': True,
        'active_column_range': [int(ncore), int(ncore + int(ncas) - 1)],
        'active_orbital_count': int(ncas),
        **localization_details,
    })
    return transformed, provenance


def run_cas_method(
    mf: Any,
    mol: Any,
    method_name: str,
    active_space: ActiveSpaceSpec,
    max_cycle: int,
    unrestricted: bool = False,
    reference_label: Optional[str] = None,
    sc_nevpt2: Optional[Any] = None,
    solver_name: str = 'fci',
    solver_options: Optional[Dict[str, Any]] = None,
    scratch_directory: Optional[str] = None,
    orbital_processing: Optional[OrbitalProcessingSpec] = None,
    density_fitting: Optional[DensityFittingSpec] = None,
) -> Dict[str, Any]:
    if method_name not in CAS_METHODS:
        raise ValueError('Unsupported CAS method: {0}'.format(method_name))
    if not active_space.ncas or active_space.nelecas is None:
        raise ValueError('CASCI/CASSCF require active_space.ncas and active_space.nelecas')

    normalized_solver = str(solver_name or 'fci').strip().lower().replace('-', '_')
    if normalized_solver in ('dmrg', 'block2'):
        normalized_solver = 'block2_dmrg'
    if normalized_solver not in ('fci', 'block2_dmrg'):
        raise ValueError('Unsupported active-space solver: {0}'.format(solver_name))
    if normalized_solver == 'block2_dmrg':
        if unrestricted:
            raise ValueError('block2 DMRG-CASCI/CASSCF currently requires a restricted or ROHF spin-adapted reference.')
        if sc_nevpt2 is not None and bool(getattr(sc_nevpt2, 'enabled', False)):
            raise ValueError('SC-NEVPT2 is not available for the block2 DMRG active-space provider yet.')

    requested_nroots, state_average_weights = state_target_configuration(solver_options)

    from pyscf import mcscf  # pylint: disable=import-outside-toplevel

    nelecas = _nelecas_for_pyscf(active_space.nelecas)
    if method_name == 'casci':
        mc = (mcscf.UCASCI if unrestricted else mcscf.CASCI)(mf, int(active_space.ncas), nelecas)
    else:
        mc = (mcscf.UCASSCF if unrestricted else mcscf.CASSCF)(mf, int(active_space.ncas), nelecas)
    if (
        method_name == 'casscf' and not unrestricted
        and density_fitting is not None and density_fitting.enabled
    ):
        mc = mc.density_fit(auxbasis=density_fitting.auxbasis)
    # Read the configured optimizer rather than inferring its integral path
    # from the requested flag. The block2 adapter consumes these same ERIs.
    cas_df = getattr(mc, 'with_df', None)
    cas_density_fitting = {
        'enabled': bool(cas_df),
        'auxbasis': getattr(cas_df, 'auxbasis', None),
        'implementation': type(mc).__name__,
    }
    configure_fci_solver(getattr(mc, 'fcisolver', None), requested_nroots)
    if method_name == 'casci' and requested_nroots > 1 and hasattr(mc, 'canonicalization'):
        # PySCF otherwise canonicalizes from root 0 and emits a misleading
        # warning.  This workflow keeps the input orbitals fixed and computes
        # the explicitly weighted natural occupations below.
        mc.canonicalization = False
    if normalized_solver == 'fci' and method_name == 'casscf' and requested_nroots > 1:
        mc = mcscf.state_average_(mc, state_average_weights)
    if hasattr(mc, 'max_cycle_macro'):
        mc.max_cycle_macro = max_cycle
    if hasattr(mc, 'max_cycle'):
        mc.max_cycle = max_cycle

    initial_mo_coeff = _initial_mo_coeff(active_space, unrestricted)
    # PySCF converts a UHF reference to a shared-orbital RHF/ROHF-style
    # CASSCF object here. Use that converted spatial basis rather than the
    # original alpha/beta coefficient pair unless the audit supplied a guess.
    mo_coeff = initial_mo_coeff if initial_mo_coeff is not None else getattr(mc, 'mo_coeff', None)
    if mo_coeff is None:
        mo_coeff = getattr(mf, 'mo_coeff', None)
    checkpoint_orbital_restart = None
    orbital_restart_value = (
        solver_options.get('orbital_restart_manifest')
        if isinstance(solver_options, dict)
        else None
    )
    joint_restart_value = (
        solver_options.get('restart_manifest')
        if isinstance(solver_options, dict)
        else None
    )
    if orbital_restart_value and joint_restart_value:
        raise ValueError('Choose either orbital-only projection or joint orbital/MPS continuation, not both.')
    if (solver_options or {}).get('restart_geometry_policy') == 'transport' and (
            normalized_solver != 'block2_dmrg' or method_name != 'casscf' or unrestricted
            or not joint_restart_value or not solver_options.get('restart_required', True)):
        raise ValueError('MPS transport requires restricted block2 CASSCF and a required joint checkpoint.')
    if (
        normalized_solver == 'block2_dmrg'
        and method_name == 'casscf'
        and isinstance(solver_options, dict)
        and (orbital_restart_value or joint_restart_value)
    ):
        from ...providers.block2 import load_optimized_orbitals  # pylint: disable=import-outside-toplevel

        mo_coeff, checkpoint_orbital_restart = load_optimized_orbitals(
            orbital_restart_value or joint_restart_value,
            mf,
            ncas=int(active_space.ncas),
            nelecas=nelecas,
            require_same_active_space=not bool(orbital_restart_value),
            allow_geometry_projection=bool(orbital_restart_value) and not bool(joint_restart_value),
            restart_geometry_policy=solver_options.get('restart_geometry_policy', 'same_geometry'),
            restart_min_active_overlap=solver_options.get('restart_min_active_overlap', 0.9),
            frozen_orbital_indices=list(getattr(orbital_processing, 'frozen_orbital_indices', []) or []),
            continuation_policy=getattr(orbital_processing, 'continuation_policy', 'project_all'),
            casscf=mc,
        )
    if (
        checkpoint_orbital_restart is None
        and active_space.initial_mo_coeff is None
        and active_space.orbital_indices
    ):
        mo_coeff = mcscf.sort_mo(mc, mo_coeff, _cas_orbital_indices_for_sort(active_space, unrestricted), base=0)

    # Indices refer to full CASSCF MO columns after active-space selection or
    # checkpoint loading; they are not DMRG site indices or AO basis indices.
    frozen_orbitals = list(getattr(orbital_processing, 'frozen_orbital_indices', []) or [])
    if frozen_orbitals:
        if method_name != 'casscf' or unrestricted:
            raise ValueError('Frozen orbital optimization requires restricted/ROHF CASSCF.')
        from ...input_validation import integer_list
        frozen_orbitals = integer_list(frozen_orbitals, 'orbital_processing.frozen_orbital_indices')
        nmo = int(mo_coeff.shape[1])
        if len(set(frozen_orbitals)) != len(frozen_orbitals) or any(i < 0 or i >= nmo for i in frozen_orbitals):
            raise ValueError('Frozen orbital indices must be unique and within the full CASSCF MO matrix.')
        mc.frozen = frozen_orbitals
    orbital_optimization = {
        'frozen_orbital_indices': frozen_orbitals,
        'frozen_orbital_count': len(frozen_orbitals),
        'index_basis': 'zero_based_full_casscf_mo_columns_after_active_space_preparation',
        'implementation': 'pyscf.mcscf.CASSCF.frozen',
        'continuation_policy': getattr(orbital_processing, 'continuation_policy', 'project_all'),
        'frozen_reference': ('target_scf' if getattr(orbital_processing, 'continuation_policy', 'project_all')
                             == 'target_scf_core' else 'initial_casscf_orbitals'),
    }

    if normalized_solver == 'block2_dmrg':
        from ...providers.block2 import (  # pylint: disable=import-outside-toplevel
            Block2FCISolverAdapter,
            attach_optimized_orbitals,
            build_molecular_active_space_hamiltonian,
            entanglement_active_space_recommendation,
            run_block2_dmrg,
        )
        from ...providers.block2.config import block2_options_for_orbital_processing
        from ..orbital_projection import molecular_basis_signature

        mo_coeff, orbital_provenance = _active_orbital_provenance(
            mf,
            mol,
            mo_coeff,
            ncas=int(active_space.ncas),
            nelecas=nelecas,
            active_space=active_space,
            orbital_processing=(
                None if checkpoint_orbital_restart is not None and not orbital_restart_value else orbital_processing
            ),
        )
        if checkpoint_orbital_restart is not None:
            orbital_provenance.update({
                'input_basis': 'checkpoint_optimized_orbitals',
                'output_basis': 'checkpoint_optimized_orbitals',
                'joint_checkpoint_restart': copy.deepcopy(checkpoint_orbital_restart),
            })
        effective_solver_options = block2_options_for_orbital_processing(solver_options, orbital_processing)
        joint_restart_provenance = copy.deepcopy(
            effective_solver_options.get('restart_provenance', {}) or {}
        )
        orbital_restart_provenance = copy.deepcopy(
            effective_solver_options.pop('orbital_restart_provenance', {}) or {}
        )
        effective_solver_options.pop('orbital_restart_manifest', None)
        if method_name in ('casci', 'casscf') and effective_solver_options.get(
            'entanglement_active_space_review',
            True,
        ):
            effective_solver_options.setdefault('compute_entanglement', True)
            effective_solver_options.setdefault('compute_mutual_information', False)
        if method_name == 'casscf':
            adapter = Block2FCISolverAdapter(
                mol,
                effective_solver_options,
                scratch_directory=scratch_directory,
            )
            mc.fcisolver = adapter
            if requested_nroots > 1:
                mc = mcscf.state_average_(mc, state_average_weights)
                adapter = mc.fcisolver
            macro_trace_by_iteration: Dict[int, Dict[str, Any]] = {}

            def _capture_macro_iteration(envs: Dict[str, Any]) -> None:
                raw_iteration = envs.get('imacro')
                iteration_key = (
                    int(raw_iteration)
                    if raw_iteration is not None
                    else len(macro_trace_by_iteration)
                )
                macro_trace_by_iteration[iteration_key] = {
                    'optimizer_iteration': (
                        int(raw_iteration) if raw_iteration is not None else None
                    ),
                    'energy': float(envs['e_tot']) if envs.get('e_tot') is not None else None,
                    'orbital_gradient_norm': (
                        float(envs['norm_gorb']) if envs.get('norm_gorb') is not None else None
                    ),
                    'density_change_norm': (
                        float(envs['norm_ddm']) if envs.get('norm_ddm') is not None else None
                    ),
                }

            mc.callback = _capture_macro_iteration
            from .casscf_space import run_casscf_kernel
            initial_frozen_mo = mo_coeff[:, frozen_orbitals].copy()
            kernel_result, integral_space = run_casscf_kernel(mc, mo_coeff)
            orbital_optimization['integral_space'] = integral_space
            energy = kernel_result[0] if isinstance(kernel_result, tuple) else kernel_result
            final_mo_coeff = getattr(mc, 'mo_coeff', mo_coeff)
            final_h1e, final_core_energy = mc.get_h1eff(final_mo_coeff)
            final_eri = mc.get_h2eff(final_mo_coeff)
            final_energies, final_states = Block2FCISolverAdapter.kernel(
                adapter,
                final_h1e,
                final_eri,
                int(active_space.ncas),
                getattr(mc, 'nelecas', nelecas),
                ecore=final_core_energy,
                nroots=requested_nroots,
                _call_role='final_analysis',
                _final_analysis=True,
            )
            states = final_states if isinstance(final_states, list) else [final_states]
            state = states[0] if states else None
            state_energies = (
                [float(value) for value in final_energies]
                if isinstance(final_energies, (list, tuple))
                else [float(final_energies)]
            )
            if state is None or len(state_energies) != requested_nroots:
                raise RuntimeError('block2 DMRG-CASSCF completed without a final active-space solver state.')
            state_average_energy = (
                sum(
                    weight * value
                    for weight, value in zip(state_average_weights, state_energies)
                )
                if requested_nroots > 1
                else None
            )
            energy = state_energies[0]
            import numpy as np  # pylint: disable=import-outside-toplevel

            dmrg_result = copy.deepcopy(state.dmrg_result)
            dmrg_result['energy'] = float(energy)
            dmrg_result['state_energies'] = list(state_energies)
            dmrg_result['excitation_energies'] = [
                float(value - state_energies[0]) for value in state_energies
            ]
            dmrg_result.pop('state_average_energy', None)
            dmrg_result.pop('state_average_weights', None)
            if requested_nroots > 1:
                dmrg_result['state_average_energy'] = float(state_average_energy)
                dmrg_result['state_average_weights'] = list(state_average_weights)
            final_ordering = dmrg_result.get('orbital_ordering')
            if isinstance(final_ordering, dict):
                internal_method = final_ordering.get('requested_method')
                original_method = final_ordering.get('casscf_initial_request')
                if original_method:
                    final_ordering['requested_method'] = original_method
                    final_ordering['requested_order'] = list(
                        final_ordering.get('casscf_initial_requested_order') or []
                    )
                    final_ordering['internal_continuation_method'] = internal_method
            checkpoint_manifest = dmrg_result.get('checkpoint_manifest')
            if isinstance(checkpoint_manifest, dict):
                checkpoint_ordering = checkpoint_manifest.get('orbital_ordering')
                if isinstance(checkpoint_ordering, dict) and isinstance(final_ordering, dict):
                    checkpoint_ordering.update({
                        'requested_method': final_ordering.get('requested_method'),
                        'requested_order': list(final_ordering.get('requested_order') or []),
                        'casscf_initial_request': final_ordering.get('casscf_initial_request'),
                        'casscf_initial_requested_order': list(
                            final_ordering.get('casscf_initial_requested_order') or []
                        ),
                        'internal_continuation_method': final_ordering.get(
                            'internal_continuation_method'
                        ),
                    })
                total_active_electrons = (
                    sum(int(value) for value in getattr(mc, 'nelecas', nelecas))
                    if isinstance(getattr(mc, 'nelecas', nelecas), (list, tuple))
                    else int(getattr(mc, 'nelecas', nelecas))
                )
                ncore = int(getattr(
                    mc,
                    'ncore',
                    (int(mol.nelectron) - total_active_electrons) // 2,
                ))
                attach_optimized_orbitals(
                    checkpoint_manifest,
                    final_mo_coeff,
                    ncore=ncore,
                    ncas=int(active_space.ncas),
                    nelecas=getattr(mc, 'nelecas', nelecas),
                    reference=reference_label or 'restricted',
                    orbital_provenance=orbital_provenance,
                    molecular_signature=molecular_basis_signature(mol),
                    molecule=mol,
                    frozen_orbital_indices=frozen_orbitals,
                )
            arrays = dict(state.arrays)
            optimized_mo_coeff = np.asarray(final_mo_coeff)
            if frozen_orbitals:
                frozen_error = float(np.max(np.abs(optimized_mo_coeff[:, frozen_orbitals] - initial_frozen_mo)))
                orbital_optimization['frozen_max_abs_change_during_optimization'] = frozen_error
                arrays['initial_frozen_mo_coeff'] = initial_frozen_mo
                if not np.isfinite(frozen_error) or frozen_error > 1e-8:
                    raise ValueError('CASSCF did not preserve the configured frozen orbitals.')
            solver_trace = adapter.trace
            energy_solver_calls = [
                item for item in solver_trace
                if item.get('role') in ('active_space_energy', 'final_analysis')
            ]
            arrays.update({
                'optimized_mo_coeff': optimized_mo_coeff,
                'casscf_solver_energies': np.asarray(
                    [item['energy'] for item in energy_solver_calls],
                    dtype=float,
                ),
            })
            orbital_converged = bool(getattr(mc, 'converged', False))
            dmrg_converged = bool(state.dmrg_result.get('converged'))
            combined_converged = bool(orbital_converged and dmrg_converged)
            macro_trace = []
            for index, key in enumerate(sorted(macro_trace_by_iteration), start=1):
                item = macro_trace_by_iteration[key]
                item['macro_iteration'] = index
                macro_trace.append(item)
            casscf_result = {
                'schema': 'pyscf-agent.block2-dmrg-casscf.v1',
                'orbital_optimization': copy.deepcopy(orbital_optimization),
                'orbital_optimizer': 'pyscf.mcscf.CASSCF',
                'orbital_optimization_mode': (
                    'state_averaged' if requested_nroots > 1 else 'state_specific'
                ),
                'nroots': requested_nroots,
                'state_energies': list(state_energies),
                'orbital_converged': orbital_converged,
                'dmrg_converged': dmrg_converged,
                'converged': combined_converged,
                'macro_iterations': len(macro_trace),
                'active_space_solver_calls': len(solver_trace),
                'energy_solver_calls': len(energy_solver_calls),
                'orbital_response_solver_calls': len(solver_trace) - len(energy_solver_calls),
                'internal_mps_continuation': True,
                'mps_initial_guess_transport_applied': bool(
                    checkpoint_orbital_restart
                    and checkpoint_orbital_restart.get('mps_initial_guess_transport')
                    and solver_trace and solver_trace[0].get('restart_applied')
                ),
                'joint_checkpoint_restart_applied': checkpoint_orbital_restart is not None,
                'joint_checkpoint_restart': copy.deepcopy(checkpoint_orbital_restart),
                'joint_checkpoint_restart_provenance': joint_restart_provenance,
                'orbital_restart_only': bool(orbital_restart_value),
                'orbital_restart_provenance': orbital_restart_provenance,
                'requested_save_mps': bool(effective_solver_options.get('save_mps')),
                'effective_save_mps': True,
                'macro_iteration_trace': macro_trace,
                'solver_call_trace': solver_trace,
                'optimized_mo_coeff_artifact': 'block2_dmrg_arrays',
            }
            if requested_nroots > 1:
                casscf_result.update({
                    'state_average_weights': list(state_average_weights),
                    'state_average_energy': float(state_average_energy),
                })
            dmrg_result.update({
                'energy': float(energy),
                'converged': combined_converged,
                'dmrg_converged': dmrg_converged,
                'casscf': casscf_result,
            })
            dmrg_result['entanglement_active_space_recommendation'] = (
                entanglement_active_space_recommendation(
                    dmrg_result,
                    ncore=ncore,
                    ncas=int(active_space.ncas),
                    nelecas=getattr(mc, 'nelecas', nelecas),
                    nmo=int(optimized_mo_coeff.shape[1]),
                    options=effective_solver_options,
                )
            )
            state_rdm1 = [np.asarray(item.rdm1) for item in states]
            rdm1 = state_rdm1[0]
            natural_occupations = np.linalg.eigvalsh(
                0.5 * (rdm1 + rdm1.T.conj())
            )[::-1].real.tolist()
            dmrg_result['natural_occupations'] = list(natural_occupations)
            dmrg_result['natural_occupations_scope'] = 'root_0'
            dmrg_result['entanglement_diagnostics_scope'] = 'root_0'
            initial_orbital_basis = orbital_provenance.get('output_basis')
            orbital_provenance.update({
                'initial_orbital_basis': initial_orbital_basis,
                'output_basis': 'casscf_optimized_active_space',
                'orbital_optimizer': 'pyscf.mcscf.CASSCF',
                'optimization_applied': True,
                'optimization_converged': orbital_converged,
                'optimized_mo_coeff_artifact': 'block2_dmrg_arrays',
            })
            if isinstance(checkpoint_manifest, dict):
                orbital_context = checkpoint_manifest.get('orbital_context')
                if isinstance(orbital_context, dict):
                    orbital_context['orbital_provenance'] = copy.deepcopy(
                        orbital_provenance
                    )
            effective_reference = reference_label or 'restricted'
            result = {
                'energy': float(energy),
                'orbital_optimization': copy.deepcopy(orbital_optimization),
                'density_fitting': cas_density_fitting,
                'converged': combined_converged,
                'solver': 'block2_dmrg',
                'reference': effective_reference,
                'spin_adapted': True,
                'ncas': int(active_space.ncas),
                'nelecas': _safe_to_list(getattr(mc, 'nelecas', nelecas)),
                'active_orbital_indices': _active_orbital_indices_payload(active_space),
                'active_orbital_index_spin_policy': _active_orbital_spin_policy(active_space, unrestricted),
                'initial_orbital_guess': initial_orbital_basis,
                'orbital_provenance': orbital_provenance,
                'natural_occupations': natural_occupations,
                'state_energies': list(state_energies),
                'excitation_energies': [
                    float(value - state_energies[0]) for value in state_energies
                ],
                'requested_root_count': requested_nroots,
                'computed_root_count': len(state_energies),
                'targeted_roots_complete': len(state_energies) >= requested_nroots,
                'orbital_optimization_mode': (
                    'state_averaged' if requested_nroots > 1 else 'state_specific'
                ),
                'state_sector': {
                    'electron_count': int(mol.nelectron),
                    'spin_2s': int(getattr(mol, 'spin', 0)),
                    'scope': 'fixed_particle_total_spin_and_configured_symmetry_sector',
                },
                'entanglement_active_space_recommendation': copy.deepcopy(
                    dmrg_result.get('entanglement_active_space_recommendation')
                ),
                'dmrg_result': dmrg_result,
                'hamiltonian': dmrg_result.get('hamiltonian'),
                '_transient_dmrg_arrays': arrays,
            }
            if requested_nroots > 1:
                result['state_average_energy'] = float(state_average_energy)
                result['state_average_weights'] = list(state_average_weights)
            return result
        hamiltonian = build_molecular_active_space_hamiltonian(
            mf,
            mo_coeff,
            ncas=int(active_space.ncas),
            nelecas=nelecas,
            active_orbital_indices=_active_orbital_indices_payload(active_space),
            orbital_provenance=orbital_provenance,
        )
        dmrg_result = run_block2_dmrg(
            hamiltonian,
            effective_solver_options,
            scratch_directory=scratch_directory,
            compute_2rdm=False,
        )
        arrays = dmrg_result.pop('_transient_dmrg_arrays', {})
        total_active_electrons = (
            sum(int(value) for value in nelecas)
            if isinstance(nelecas, (list, tuple))
            else int(nelecas)
        )
        ncore = (int(mol.nelectron) - total_active_electrons) // 2
        nmo = int(getattr(mo_coeff, 'shape', (0, 0))[1])
        dmrg_result['entanglement_active_space_recommendation'] = (
            entanglement_active_space_recommendation(
                dmrg_result,
                ncore=ncore,
                ncas=int(active_space.ncas),
                nelecas=nelecas,
                nmo=nmo,
                options=effective_solver_options,
            )
        )
        effective_reference = reference_label or 'restricted'
        state_energies = list(dmrg_result.get('state_energies') or [dmrg_result['energy']])
        return {
            'energy': float(dmrg_result['energy']),
            'converged': bool(dmrg_result.get('converged')),
            'solver': 'block2_dmrg',
            'reference': effective_reference,
            'spin_adapted': True,
            'ncas': int(active_space.ncas),
            'nelecas': _safe_to_list(nelecas),
            'active_orbital_indices': _active_orbital_indices_payload(active_space),
            'active_orbital_index_spin_policy': _active_orbital_spin_policy(active_space, unrestricted),
            'initial_orbital_guess': (
                orbital_provenance.get('output_basis')
                if orbital_provenance.get('localization_applied')
                else 'active_space.initial_mo_coeff'
                if active_space.initial_mo_coeff is not None
                else 'canonical_orbitals'
            ),
            'orbital_provenance': orbital_provenance,
            'natural_occupations': dmrg_result.get('natural_occupations'),
            'state_energies': state_energies,
            'excitation_energies': [
                float(value - state_energies[0]) for value in state_energies
            ],
            'requested_root_count': requested_nroots,
            'computed_root_count': len(state_energies),
            'targeted_roots_complete': len(state_energies) >= requested_nroots,
            'state_sector': {
                'electron_count': int(mol.nelectron),
                'spin_2s': int(getattr(mol, 'spin', 0)),
                'scope': 'fixed_particle_total_spin_and_configured_symmetry_sector',
            },
            'entanglement_active_space_recommendation': copy.deepcopy(
                dmrg_result.get('entanglement_active_space_recommendation')
            ),
            'dmrg_result': dmrg_result,
            'hamiltonian': hamiltonian.summary(),
            '_transient_dmrg_arrays': arrays,
        }

    if method_name == 'casscf' and not unrestricted:
        from .casscf_space import run_casscf_kernel
        kernel_result, integral_space = run_casscf_kernel(mc, mo_coeff)
        orbital_optimization['integral_space'] = integral_space
    else:
        kernel_result = mc.kernel(mo_coeff)
    if isinstance(kernel_result, tuple):
        kernel_energy = kernel_result[0]
    else:
        kernel_energy = kernel_result

    state_energies = _state_energies_from_cas(mc, kernel_energy, requested_nroots)
    energy = state_energies[0]
    excitation_energies = [float(value - energy) for value in state_energies]
    state_average_energy = (
        sum(
            weight * value
            for weight, value in zip(state_average_weights, state_energies)
        )
        if requested_nroots > 1
        else None
    )

    natural_occupations = None
    natural_occupation_evidence = {
        'status': 'available', 'quantity': 'cas_natural_occupations',
    }
    try:
        if requested_nroots > 1:
            natural_occupations = _root_natural_occupations(mc, root=0)
        else:
            natural_occupations = mc.cas_natorb()[2]
        if natural_occupations is None:
            natural_occupation_evidence.update(
                status='unavailable', reason='Solver returned no natural occupations for the requested root',
            )
    except Exception as exc:
        natural_occupation_evidence.update(status='failed', reason=str(exc))

    effective_reference = reference_label or ('uhf' if unrestricted else 'restricted')
    result = {
        'energy': float(energy),
        'density_fitting': cas_density_fitting,
        'converged': bool(getattr(mc, 'converged', True)),
        'reference': effective_reference,
        'spin_adapted': not unrestricted,
        'ncas': int(active_space.ncas),
        'nelecas': _safe_to_list(getattr(mc, 'nelecas', nelecas)),
        'active_orbital_indices': _active_orbital_indices_payload(active_space),
        'active_orbital_index_spin_policy': _active_orbital_spin_policy(active_space, unrestricted),
        'initial_orbital_guess': 'active_space.initial_mo_coeff' if active_space.initial_mo_coeff is not None else 'canonical_orbitals',
        'natural_occupations': _safe_to_list(natural_occupations),
        'natural_occupations_scope': 'root_0',
        'natural_occupation_evidence': natural_occupation_evidence,
        'state_energies': state_energies,
        'excitation_energies': excitation_energies,
        'requested_root_count': requested_nroots,
        'computed_root_count': len(state_energies),
        'targeted_roots_complete': len(state_energies) >= requested_nroots,
        'orbital_optimization_mode': (
            'state_averaged'
            if method_name == 'casscf' and requested_nroots > 1
            else 'state_specific'
        ),
        'state_sector': {
            'electron_count': int(mol.nelectron),
            'spin_2s': int(getattr(mol, 'spin', 0)),
            'scope': 'fixed_particle_spin_projection_sector',
        },
        'solver': 'fci',
    }
    if method_name == 'casscf':
        result['orbital_optimization'] = copy.deepcopy(orbital_optimization)
    if method_name == 'casscf' and requested_nroots > 1:
        result['state_average_energy'] = float(state_average_energy)
        result['state_average_weights'] = list(state_average_weights)
    if sc_nevpt2 is not None and bool(getattr(sc_nevpt2, 'enabled', False)):
        from pyscf import mrpt  # pylint: disable=import-outside-toplevel

        root = int(getattr(sc_nevpt2, 'root', 0))
        if root != 0:
            raise ValueError('SC-NEVPT2 excited-state roots require a multi-state or state-averaged CAS workflow; current backend supports root=0 only.')
        density_fit = bool(getattr(sc_nevpt2, 'density_fit', True))
        nevpt = mrpt.NEVPT(mc, root=root, density_fit=density_fit)
        correction_energy = float(nevpt.kernel())
        result['post_cas_results'] = {
            'sc_nevpt2': {
                'method': 'sc_nevpt2',
                'enabled': True,
                'status': 'succeeded',
                'correction_energy': correction_energy,
                'total_energy': float(energy) + correction_energy,
                'reference_energy': float(energy),
                'reference_method': method_name,
                'root': root,
                'density_fit': density_fit,
                'reference': effective_reference,
            },
        }
    return result
