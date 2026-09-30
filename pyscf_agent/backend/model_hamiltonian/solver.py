from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from ...contracts import RuntimeSpec
from ...input_validation import validate_dataclass_input




from .spec import (
    MODEL_HAMILTONIAN_SCHEMA as MODEL_HAMILTONIAN_SCHEMA,
    SUPPORTED_MODEL_HAMILTONIANS as SUPPORTED_MODEL_HAMILTONIANS,
    SUPPORTED_MODEL_SOLVERS as SUPPORTED_MODEL_SOLVERS,
    _as_float as _as_float,
    _as_int as _as_int,
    ensure_model_spec_graph as ensure_model_spec_graph,
    load_model_spec_from_file as load_model_spec_from_file,
    model_spec_graph_view as model_spec_graph_view,
    normalize_model_spec,
    normalize_solver_name,
    parse_model_spec_from_python_input as parse_model_spec_from_python_input,
    validate_model_hamiltonian_spec,
)
from .observables import (
    _compute_fci_charge_correlation,
    _compute_fci_double_occupancy,
    _compute_fci_site_density,
    _compute_fci_spin_correlation,
    _compute_charge_correlation_from_spin_rdms,
    _compute_double_occupancy_from_spin_rdms,
    _compute_site_density_from_spin_rdms,
    _compute_spin_correlation_from_spin_rdms,
    _compute_model_hamiltonian_energy_levels,
    _compute_strong_correlation_diagnostics,
    _double_excitation_amplitude_summary,
    _fci_natural_occupation_summary,
    _mean_field_homo_lumo_summary,
    _natural_occupation_summary_from_dm1,
    _natural_occupation_summary_from_solver,
    _pair_matrix_payload,
    require_supported_full_diagonalization,
    _site_ids_from_spec,
    _site_vector_payload,
)
from .bloch import run_bloch_tight_binding
from .dmet_observables import assemble_dmet_local_observables
from .fci_observables import fci_determinant_space_dimension


COMPLETE_SPECTRUM_OUTPUTS = frozenset({
    'energy_levels',
    'energy_level_count',
    'many_body_basis_dimension',
})


def build_model_hamiltonian_matrices(spec: Dict[str, Any]):
    import numpy as np  # pylint: disable=import-outside-toplevel

    validation_errors = validate_model_hamiltonian_spec(spec)
    if validation_errors:
        raise ValueError('; '.join(validation_errors))

    ordered_sites = sorted(spec['sites'], key=lambda item: int(item['id']))
    site_to_orb = {int(site['id']): index for index, site in enumerate(ordered_sites)}
    norb = len(ordered_sites)
    h1e = np.zeros((norb, norb), dtype=float)
    eri = np.zeros((norb, norb, norb, norb), dtype=float)

    onsite_u_count = 0
    for site in ordered_sites:
        i = site_to_orb[int(site['id'])]
        h1e[i, i] = float(site.get('epsilon', 0.0))
        u_value = float(site.get('U', 0.0))
        eri[i, i, i, i] += u_value
        if abs(u_value) > 1e-12:
            onsite_u_count += 1

    nonzero_hopping_count = 0
    nonzero_v_count = 0
    for bond in spec.get('bonds', []):
        i = site_to_orb[int(bond['source'])]
        j = site_to_orb[int(bond['target'])]
        tij = float(bond.get('effective_t', bond.get('t', 0.0)))
        vij = float(bond.get('effective_V', bond.get('V', 0.0)))
        h1e[i, j] += tij
        h1e[j, i] += tij
        if abs(tij) > 1e-12:
            nonzero_hopping_count += 1
        if abs(vij) > 1e-12:
            eri[i, i, j, j] += vij
            eri[j, j, i, i] += vij
            nonzero_v_count += 1

    metadata = {
        'norb': norb,
        'nelec': [int(spec['nelec'][0]), int(spec['nelec'][1])],
        'site_count': len(ordered_sites),
        'bond_count': len(spec.get('bonds', [])),
        'nonzero_hopping_count': nonzero_hopping_count,
        'nonzero_v_count': nonzero_v_count,
        'onsite_u_count': onsite_u_count,
    }
    return h1e, eri, metadata


def _fake_molecule_for_model_hamiltonian(norb: int, nelec: Tuple[int, int]):
    from pyscf import gto  # pylint: disable=import-outside-toplevel

    mol = gto.Mole()
    mol.nelectron = int(nelec[0]) + int(nelec[1])
    mol.spin = int(nelec[0]) - int(nelec[1])
    mol.incore_anyway = True
    mol.verbose = 0
    mol.build(False, False)
    mol.nao_nr = lambda *args: norb
    mol.energy_nuc = lambda *args: 0.0
    return mol


def _model_uhf_initial_guesses(norb: int, nelec: Tuple[int, int]):
    import numpy as np  # pylint: disable=import-outside-toplevel

    nalpha, nbeta = int(nelec[0]), int(nelec[1])

    def diagonal_guess(alpha_indices, beta_indices):
        alpha_dm = np.zeros((norb, norb), dtype=float)
        beta_dm = np.zeros((norb, norb), dtype=float)
        for index in list(alpha_indices)[:nalpha]:
            if 0 <= int(index) < norb:
                alpha_dm[int(index), int(index)] = 1.0
        for index in list(beta_indices)[:nbeta]:
            if 0 <= int(index) < norb:
                beta_dm[int(index), int(index)] = 1.0
        return alpha_dm, beta_dm

    even_sites = list(range(0, norb, 2)) + list(range(1, norb, 2))
    odd_sites = list(range(1, norb, 2)) + list(range(0, norb, 2))
    forward_sites = list(range(norb))
    reverse_sites = list(reversed(forward_sites))
    guesses = [
        ('default', None),
        ('antiferromagnetic_even_alpha', diagonal_guess(even_sites, odd_sites)),
        ('antiferromagnetic_odd_alpha', diagonal_guess(odd_sites, even_sites)),
        ('edge_separated', diagonal_guess(forward_sites, reverse_sites)),
    ]
    if norb:
        guesses.append((
            'uniform_density',
            (np.eye(norb) * (nalpha / norb), np.eye(norb) * (nbeta / norb)),
        ))
    return guesses


def _model_hamiltonian_reference(h1e, eri, metadata: Dict[str, Any], *, restricted: bool, runtime: Optional[RuntimeSpec] = None):
    import numpy as np  # pylint: disable=import-outside-toplevel
    from pyscf import ao2mo, scf  # pylint: disable=import-outside-toplevel

    runtime = runtime or RuntimeSpec()
    norb = metadata['norb']
    nelec = tuple(metadata['nelec'])
    mol = _fake_molecule_for_model_hamiltonian(norb, nelec)
    restored_eri = ao2mo.restore(8, eri, norb)

    def build_reference():
        mf = scf.RHF(mol) if restricted else scf.UHF(mol)
        mf.get_hcore = lambda *args: h1e
        mf.get_ovlp = lambda *args: np.eye(norb)
        mf._eri = restored_eri
        if runtime.scf_algorithm == 'newton':
            mf = mf.newton()
        mf.max_cycle = runtime.max_cycle
        for field in ('conv_tol', 'conv_tol_grad', 'diis_space'):
            value = getattr(runtime, field)
            if value is not None:
                setattr(mf, field, value)
        mf.chkfile = None
        mf.verbose = runtime.verbose
        return mf

    guesses = [('default', None)] if restricted else _model_uhf_initial_guesses(norb, nelec)
    candidates = []
    attempts = []
    for guess_name, guess_dm in guesses:
        mf = build_reference()
        try:
            if guess_dm is None:
                mf.kernel()
            else:
                mf.kernel(dm0=guess_dm)
            energy = float(mf.e_tot)
            if not math.isfinite(energy):
                raise ValueError('SCF returned a nonfinite energy')
        except Exception as exc:  # Each bounded initial guess is independent.
            attempts.append({'initial_guess': guess_name, 'status': 'failed', 'error': str(exc)})
            continue
        converged = bool(getattr(mf, 'converged', False))
        attempts.append({'initial_guess': guess_name, 'status': 'succeeded' if converged else 'unconverged',
                         'converged': converged, 'energy': energy})
        candidates.append((converged, energy, guess_name, mf))
    if not candidates:
        raise RuntimeError('All model reference initial guesses failed: {0}'.format(json.dumps(attempts)))

    converged_candidates = [candidate for candidate in candidates if candidate[0]]
    candidate_pool = converged_candidates or candidates
    _converged, _energy, guess_name, best_mf = min(candidate_pool, key=lambda item: item[1])
    best_mf.pyscf_agent_initial_guess = guess_name
    best_mf.pyscf_agent_reference_candidates = attempts
    return best_mf, mol


def _normalize_model_outputs(outputs: Any = None) -> List[str]:
    if outputs is None:
        return ['energy']
    if isinstance(outputs, str):
        raw_outputs = [outputs]
    elif isinstance(outputs, (list, tuple, set)):
        raw_outputs = list(outputs)
    else:
        raw_outputs = ['energy']
    normalized_outputs = []
    for item in raw_outputs:
        output = str(item or '').strip()
        if output and output not in normalized_outputs:
            normalized_outputs.append(output)
    return normalized_outputs or ['energy']


def _model_root_count(solver_name: str, outputs: List[str], solver_options: Any = None) -> int:
    options = solver_options if isinstance(solver_options, dict) else {}
    raw_nroots = options.get('nroots')
    if raw_nroots is None:
        return 2 if 'excited_states' in outputs else 1
    if isinstance(raw_nroots, bool):
        raise ValueError('solver.options.nroots must be a positive integer.')
    try:
        nroots = int(raw_nroots)
    except (TypeError, ValueError) as exc:
        raise ValueError('solver.options.nroots must be a positive integer.') from exc
    if nroots <= 0 or str(raw_nroots).strip() not in (str(nroots), '{0}.0'.format(nroots)):
        raise ValueError('solver.options.nroots must be a positive integer.')
    if solver_name not in ('fci', 'block2_dmrg') and (
        'nroots' in options or 'excited_states' in outputs
    ):
        raise ValueError(
            'Low-lying roots are supported only by FCI and block2 DMRG model solvers.'
        )
    return nroots


def _fci_symmetry_summary(ci_vectors: List[Any], norb: int, nelec: Tuple[int, int]) -> Dict[str, Any]:
    from pyscf import fci  # pylint: disable=import-outside-toplevel

    root_spin_square = []
    root_multiplicity = []
    root_total_spin = []
    for ci_vector in ci_vectors:
        spin_square, multiplicity = fci.spin_op.spin_square0(ci_vector, norb, nelec)
        spin_square = float(spin_square)
        root_spin_square.append(spin_square)
        root_multiplicity.append(float(multiplicity))
        root_total_spin.append(0.5 * (max(0.0, 1.0 + 4.0 * spin_square) ** 0.5 - 1.0))
    spin_projection = 0.5 * float(nelec[0] - nelec[1])
    return {
        'particle_number': int(sum(nelec)),
        'symmetry_backend': 'pyscf_direct_spin1',
        'spin_quantum_number_kind': 'spin_projection',
        'target_spin_projection': spin_projection,
        'spin_square': root_spin_square[0],
        'inferred_total_spin': root_total_spin[0],
        'multiplicity': root_multiplicity[0],
        'root_spin_square': root_spin_square,
        'root_total_spin': root_total_spin,
        'root_multiplicity': root_multiplicity,
    }


def _run_model_post_hf_solver(
    h1e,
    eri,
    metadata: Dict[str, Any],
    solver_name: str,
    outputs: List[str],
    *,
    solver_options: Any = None,
    scratch_directory: Optional[str] = None,
    runtime: Optional[RuntimeSpec] = None,
) -> Dict[str, Any]:
    from pyscf import cc, fci, mp  # pylint: disable=import-outside-toplevel

    nelec = tuple(metadata['nelec'])
    requested_root_count = _model_root_count(solver_name, outputs, solver_options)
    # Model Hamiltonians, especially half-filled Hubbard systems, can have
    # singlet electron counts while restricted references are qualitatively poor.
    # Default to unrestricted references for all mean-field based model solvers.
    restricted = False
    if solver_name == 'dmet':
        from ...providers.libdmet import run_hubbard_dmet  # pylint: disable=import-outside-toplevel

        model_spec = metadata.get('_model_spec')
        if not isinstance(model_spec, dict):
            raise ValueError('DMET execution requires the normalized Model Hamiltonian spec')
        dmet_result = run_hubbard_dmet(
            model_spec,
            solver_options if isinstance(solver_options, dict) else {},
            scratch_directory=scratch_directory,
            collect_local_rdms=any(output in outputs for output in (
                'strong_correlation_diagnostics', 'double_occupancy', 'spin_correlation', 'charge_correlation',
            )),
        )
        arrays = dmet_result.pop('_transient_dmet_arrays', {})
        provider_logs = dmet_result.pop('_transient_provider_logs', [])
        local_observables = assemble_dmet_local_observables(
            model_spec,
            dmet_result,
            arrays,
        )
        dmet_result['local_observables'] = copy.deepcopy(local_observables)
        result = {
            'solver': 'dmet',
            'converged': bool(dmet_result.get('converged')),
            'energy': float(dmet_result['energy']),
            'energy_per_site': float(dmet_result['energy_per_site']),
            'quality_checks': copy.deepcopy(dmet_result.get('quality_checks') or []),
            'reference': 'libdmet_{0}'.format(dmet_result['configuration']['reference']),
            'restricted_reference': dmet_result['configuration']['reference'] == 'restricted',
            # libDMET exposes the outer self-consistency result here, but does
            # not provide a separate reference-SCF convergence flag.
            'reference_converged': None,
            'dmet_result': dmet_result,
            '_transient_dmet_arrays': arrays,
            '_transient_provider_logs': provider_logs,
        }
        if local_observables.get('status') == 'available':
            result['dmet_local_observables'] = local_observables
            result['density'] = list(local_observables['density'])
            result['local_magnetization'] = list(local_observables['local_magnetization'])
            double_occupancy = local_observables.get('double_occupancy')
            if isinstance(double_occupancy, list) and all(
                value is not None for value in double_occupancy
            ):
                result['double_occupancy'] = [float(value) for value in double_occupancy]
            for field in ('spin_correlation', 'charge_correlation'):
                matrix = local_observables.get(field)
                if isinstance(matrix, list) and any(
                    value is not None
                    for row in matrix
                    if isinstance(row, list)
                    for value in row
                ):
                    result[field] = matrix

            global_density = arrays.get('global_embedding_density_matrix_full')
            if global_density is None:
                global_density = arrays.get('global_embedding_density_matrix')
            if global_density is not None:
                import numpy as np  # pylint: disable=import-outside-toplevel

                global_density = np.asarray(global_density)
                if global_density.ndim == 4:
                    global_density = global_density[:, 0]
                if global_density.ndim == 3 and global_density.shape[0] >= 1:
                    if dmet_result['configuration']['reference'] == 'restricted':
                        natural_input = [global_density[0], global_density[0]]
                    elif global_density.shape[0] >= 2:
                        natural_input = [global_density[0], global_density[1]]
                    else:
                        natural_input = None
                    if natural_input is not None:
                        natural_summary = _natural_occupation_summary_from_dm1(
                            natural_input, nelec=nelec,
                            source='dmet_assembled_correlated_1rdm', scope='assembled_dmet_lattice',
                        )
                        if natural_summary is not None:
                            result['natural_occupation_summary'] = natural_summary
        return result
    if solver_name == 'block2_dmrg':
        from ...providers.block2 import (  # pylint: disable=import-outside-toplevel
            block2_compact_result_fields,
            block2_options_for_outputs,
            build_model_electronic_hamiltonian,
            run_block2_dmrg,
        )

        needs_strong_correlation = 'strong_correlation_diagnostics' in outputs
        needs_2rdm = needs_strong_correlation or any(
            output in outputs
            for output in ('double_occupancy', 'spin_correlation', 'charge_correlation')
        )
        options = block2_options_for_outputs(solver_options, outputs)
        options['symmetry'] = 'sz'
        options['compute_1rdm'] = True
        options['compute_2rdm'] = needs_2rdm
        hamiltonian = build_model_electronic_hamiltonian(h1e, eri, metadata)
        dmrg_result = run_block2_dmrg(
            hamiltonian,
            options,
            scratch_directory=scratch_directory,
            compute_2rdm=needs_2rdm,
        )
        arrays = dmrg_result.pop('_transient_dmrg_arrays', {})
        reference_mf, _mol = _model_hamiltonian_reference(h1e, eri, metadata, restricted=False, runtime=runtime)
        reference_energy = float(reference_mf.e_tot)
        result = {
            'solver': 'block2_dmrg',
            'converged': bool(dmrg_result.get('converged')),
            'energy': float(dmrg_result['energy']),
            'reference_energy': reference_energy,
            'correlation_energy': float(dmrg_result['energy'] - reference_energy),
            'reference': 'block2_dmrg',
            'correlation_reference': 'uhf',
            'restricted_reference': False,
            'reference_converged': bool(getattr(reference_mf, 'converged', False)),
            'reference_initial_guess': getattr(reference_mf, 'pyscf_agent_initial_guess', None),
            'reference_candidates': getattr(reference_mf, 'pyscf_agent_reference_candidates', []),
            'dmrg_result': dmrg_result,
            '_transient_dmrg_arrays': arrays,
        }
        result.update(block2_compact_result_fields(dmrg_result))
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
            if dmrg_result.get(field) is not None:
                result[field] = copy.deepcopy(dmrg_result[field])
        rdm1a = arrays.get('rdm1_alpha')
        rdm1b = arrays.get('rdm1_beta')
        rdm1 = [rdm1a, rdm1b] if rdm1a is not None and rdm1b is not None else arrays.get('rdm1')
        if rdm1 is not None:
            natural_summary = _natural_occupation_summary_from_dm1(
                rdm1, nelec=nelec, source='dmrg_ground_state_1rdm',
            )
            if natural_summary is not None:
                result['natural_occupation_summary'] = natural_summary
        dm2aa = arrays.get('rdm2_alpha_alpha')
        dm2ab = arrays.get('rdm2_alpha_beta')
        dm2bb = arrays.get('rdm2_beta_beta')
        if rdm1a is not None and rdm1b is not None:
            result['density'] = _compute_site_density_from_spin_rdms(rdm1a, rdm1b)
        if dm2ab is not None:
            result['double_occupancy'] = _compute_double_occupancy_from_spin_rdms(dm2ab)
        if all(value is not None for value in (rdm1a, rdm1b, dm2aa, dm2ab, dm2bb)):
            result['spin_correlation'] = _compute_spin_correlation_from_spin_rdms(
                rdm1a, rdm1b, dm2aa, dm2ab, dm2bb,
            )
            result['charge_correlation'] = _compute_charge_correlation_from_spin_rdms(
                rdm1a, rdm1b, dm2aa, dm2ab, dm2bb,
            )
        return result

    if solver_name == 'fci':
        needs_strong_correlation = 'strong_correlation_diagnostics' in outputs
        needs_full_diagonalization = needs_strong_correlation or any(
            output in outputs
            for output in ('gap', 'energy_levels', 'energy_level_count', 'many_body_basis_dimension')
        )
        if needs_full_diagonalization:
            # Model-Hamiltonian FCI diagnostics deliberately use full dense
            # diagonalization so every reported many-body level is exact.
            require_supported_full_diagonalization(metadata['norb'], nelec)
        determinant_dimension = fci_determinant_space_dimension(metadata['norb'], nelec)
        if requested_root_count > determinant_dimension:
            raise ValueError(
                'Requested nroots={requested:,}, but the selected FCI particle/spin sector contains '
                'only {available:,} states.'.format(
                    requested=requested_root_count,
                    available=determinant_dimension,
                )
            )
        raw_energies, raw_ci_vectors = fci.direct_spin1.kernel(
            h1e,
            eri,
            metadata['norb'],
            nelec,
            nroots=requested_root_count,
        )
        if requested_root_count == 1:
            state_energies = [float(raw_energies)]
            ci_vectors = [raw_ci_vectors]
        else:
            state_energies = [float(value) for value in raw_energies]
            ci_vectors = list(raw_ci_vectors)
        energy = state_energies[0]
        ci_vector = ci_vectors[0]
        reference_mf, _mol = _model_hamiltonian_reference(h1e, eri, metadata, restricted=restricted, runtime=runtime)
        reference_energy = float(reference_mf.e_tot)
        result = {
            'solver': 'fci',
            'converged': True,
            'energy': float(energy),
            'reference_energy': reference_energy,
            'correlation_energy': float(energy - reference_energy),
            'reference': 'direct_spin1',
            'correlation_reference': 'uhf',
            'restricted_reference': False,
            'reference_converged': bool(getattr(reference_mf, 'converged', False)),
            'reference_initial_guess': getattr(reference_mf, 'pyscf_agent_initial_guess', None),
            'reference_candidates': getattr(reference_mf, 'pyscf_agent_reference_candidates', []),
            'state_energies': state_energies,
            'excitation_energies': [float(value - energy) for value in state_energies],
            'requested_root_count': requested_root_count,
            'computed_root_count': len(state_energies),
            'targeted_roots_complete': len(state_energies) >= requested_root_count,
        }
        if 'symmetry_analysis' in outputs:
            result['symmetry_analysis'] = _fci_symmetry_summary(
                ci_vectors,
                metadata['norb'],
                nelec,
            )
        if needs_strong_correlation:
            natural_summary = _fci_natural_occupation_summary(ci_vector, metadata)
            if natural_summary is not None:
                result['natural_occupation_summary'] = natural_summary
        if needs_strong_correlation or any(output in outputs for output in ('gap', 'energy_levels', 'energy_level_count', 'many_body_basis_dimension')):
            result['spectrum'] = _compute_model_hamiltonian_energy_levels(h1e, eri, metadata)
        if needs_strong_correlation or 'density' in outputs:
            result['density'] = _compute_fci_site_density(ci_vector, metadata)
        if needs_strong_correlation or 'double_occupancy' in outputs:
            result['double_occupancy'] = _compute_fci_double_occupancy(ci_vector, metadata)
        if needs_strong_correlation or 'spin_correlation' in outputs:
            result['spin_correlation'] = _compute_fci_spin_correlation(ci_vector, metadata)
        if needs_strong_correlation or 'charge_correlation' in outputs:
            result['charge_correlation'] = _compute_fci_charge_correlation(ci_vector, metadata)
        return result

    needs_strong_correlation = 'strong_correlation_diagnostics' in outputs
    mf, _mol = _model_hamiltonian_reference(h1e, eri, metadata, restricted=restricted, runtime=runtime)
    if solver_name == 'mp2':
        solver = mp.MP2(mf) if restricted else mp.UMP2(mf)
    elif solver_name in ('ccsd', 'ccsd_t'):
        solver = cc.CCSD(mf) if restricted else cc.UCCSD(mf)
    else:
        raise ValueError('Unsupported Model Hamiltonian solver: {0}'.format(solver_name))
    if runtime is not None:
        if hasattr(solver, 'max_cycle'):
            solver.max_cycle = runtime.max_cycle
        if runtime.conv_tol is not None and hasattr(solver, 'conv_tol'):
            solver.conv_tol = runtime.conv_tol
    kernel_result = solver.kernel()
    correlation_energy = kernel_result[0] if isinstance(kernel_result, tuple) else kernel_result
    total_energy = getattr(solver, 'e_tot', None)
    if total_energy is None:
        total_energy = mf.e_tot + correlation_energy
    triples_correction = None
    if solver_name == 'ccsd_t':
        triples_correction = float(solver.ccsd_t())
        total_energy = total_energy + triples_correction
        correlation_energy = total_energy - mf.e_tot
    converged = bool(getattr(solver, 'converged', True))
    supports_screening_orbital_metrics = solver_name in ('mp2', 'ccsd', 'ccsd_t')
    mean_field_homo_lumo = (
        _mean_field_homo_lumo_summary(mf)
        if needs_strong_correlation and supports_screening_orbital_metrics
        else None
    )
    natural_summary = None
    amplitude_summary = None
    if needs_strong_correlation and supports_screening_orbital_metrics:
        natural_summary = _natural_occupation_summary_from_solver(solver)
        amplitude_summary = _double_excitation_amplitude_summary(solver)
    result = {
        'solver': solver_name,
        'converged': converged,
        'energy': float(total_energy),
        'reference_energy': float(mf.e_tot),
        'correlation_energy': float(correlation_energy),
        'reference': 'rhf' if restricted else 'uhf',
        'restricted_reference': restricted,
        'reference_converged': bool(getattr(mf, 'converged', False)),
        'reference_initial_guess': getattr(mf, 'pyscf_agent_initial_guess', None),
        'reference_candidates': getattr(mf, 'pyscf_agent_reference_candidates', []),
    }
    if triples_correction is not None:
        result['triples_correction'] = float(triples_correction)
    if mean_field_homo_lumo is not None:
        result['mean_field_homo_lumo'] = mean_field_homo_lumo
    if natural_summary is not None:
        result['natural_occupation_summary'] = natural_summary
    if amplitude_summary is not None:
        result['double_excitation_amplitude_summary'] = amplitude_summary
    return result


def run_model_hamiltonian_solver(
    spec: Dict[str, Any],
    solver_name: Any = None,
    outputs: Any = None,
    *,
    deferred_modules: Tuple[str, ...] = (),
    solver_options: Any = None,
    scratch_directory: Optional[str] = None,
    runtime: Optional[RuntimeSpec] = None,
) -> Dict[str, Any]:
    if runtime is None:
        runtime = RuntimeSpec()
    elif isinstance(runtime, dict):
        runtime = RuntimeSpec(**validate_dataclass_input(runtime, RuntimeSpec, 'runtime'))
    normalized_spec = normalize_model_spec(spec)
    normalized_solver = normalize_solver_name(solver_name or normalized_spec.get('solver', 'fci'))
    normalized_spec['solver'] = normalized_solver
    normalized_outputs = _normalize_model_outputs(outputs)
    gap_requested = 'gap' in normalized_outputs
    complete_spectrum_requested = any(
        output in COMPLETE_SPECTRUM_OUTPUTS for output in normalized_outputs
    )
    validation_errors = validate_model_hamiltonian_spec(normalized_spec, solver_name=normalized_solver)
    if validation_errors:
        raise ValueError('; '.join(validation_errors))
    if normalized_spec.get('representation') == 'bloch':
        return run_bloch_tight_binding(normalized_spec, outputs=normalized_outputs)
    energy_unit = str(normalized_spec.get('energy_unit') or 'a.u.').strip() or 'a.u.'
    h1e, eri, metadata = build_model_hamiltonian_matrices(normalized_spec)
    metadata['_model_spec'] = normalized_spec
    solver_results = _run_model_post_hf_solver(
        h1e,
        eri,
        metadata,
        normalized_solver,
        normalized_outputs,
        solver_options=solver_options,
        scratch_directory=scratch_directory,
        runtime=runtime,
    )
    spectrum = solver_results.get('spectrum')
    site_ids = _site_ids_from_spec(normalized_spec)
    result = {
        'task_type': 'model_hamiltonian',
        'model': normalized_spec.get('model', 'hubbard'),
        'solver': solver_results['solver'],
        'converged': solver_results['converged'],
        'energy': solver_results['energy'],
        'reference': solver_results.get('reference'),
        'correlation_reference': solver_results.get('correlation_reference'),
        'restricted_reference': solver_results.get('restricted_reference'),
        'reference_converged': solver_results.get('reference_converged'),
        'reference_initial_guess': solver_results.get('reference_initial_guess'),
        'reference_candidates': solver_results.get('reference_candidates'),
        'runtime': asdict(runtime),
        'energy_unit': energy_unit,
        'norb': metadata['norb'],
        'nelec': metadata['nelec'],
        'site_count': metadata['site_count'],
        'bond_count': metadata['bond_count'],
        'nonzero_hopping_count': metadata['nonzero_hopping_count'],
        'nonzero_v_count': metadata['nonzero_v_count'],
        'onsite_u_count': metadata['onsite_u_count'],
        'h1e': h1e.tolist() if metadata['norb'] <= 24 else None,
        'raw_scf_output': '',
        'analysis_text': (
            'Model Hamiltonian {solver} completed: model={model}, norb={norb}, '
            'nelec={nelec}, energy={energy:.12f} {energy_unit}'
        ).format(
            solver=solver_results['solver'].upper(),
            model=normalized_spec.get('model', 'hubbard'),
            norb=metadata['norb'],
            nelec=tuple(metadata['nelec']),
            energy=solver_results['energy'],
            energy_unit=energy_unit,
        ),
    }
    if solver_results.get('reference_energy') is not None:
        result['reference_energy'] = solver_results['reference_energy']
    if solver_results.get('correlation_energy') is not None:
        result['correlation_energy'] = solver_results['correlation_energy']
    if solver_results.get('energy_per_site') is not None:
        result['energy_per_site'] = solver_results['energy_per_site']
    hopping_scales = [
        abs(float(bond.get('effective_t', bond.get('t', 0.0))))
        for bond in normalized_spec.get('bonds', [])
        if isinstance(bond, dict)
        and abs(float(bond.get('effective_t', bond.get('t', 0.0)))) > 1.0e-12
    ]
    if hopping_scales:
        mean_abs_t = float(sum(hopping_scales) / len(hopping_scales))
        result['model_energy_scale'] = {
            'kind': 'mean_absolute_hopping',
            'value': mean_abs_t,
            'unit': energy_unit,
        }
        result['energy_over_abs_t'] = float(solver_results['energy'] / mean_abs_t)
        if solver_results.get('energy_per_site') is not None:
            result['energy_per_site_over_abs_t'] = float(
                solver_results['energy_per_site'] / mean_abs_t
            )
    if solver_results.get('reference_energy_per_site') is not None:
        result['reference_energy_per_site'] = solver_results['reference_energy_per_site']
    if solver_results.get('correlation_energy_per_site') is not None:
        result['correlation_energy_per_site'] = solver_results['correlation_energy_per_site']
    if isinstance(spectrum, dict):
        if complete_spectrum_requested or gap_requested:
            result['energy_level_count'] = len(spectrum['levels'])
            result['many_body_basis_dimension'] = spectrum['basis_dimension']
            result['energy_spectrum_method'] = spectrum.get('method', normalized_solver)
            result['energy_levels'] = spectrum['levels']
        if gap_requested and len(spectrum['levels']) >= 2:
            result['gap'] = float(spectrum['levels'][1] - spectrum['levels'][0])
    elif complete_spectrum_requested or (gap_requested and normalized_solver != 'block2_dmrg'):
        result['energy_spectrum_unavailable_reason'] = (
            '{0} does not define a complete excited-state spectrum in this workflow'.format(normalized_solver.upper())
        )
    if solver_results.get('triples_correction') is not None:
        result['triples_correction'] = solver_results['triples_correction']
    mean_field_homo_lumo = solver_results.get('mean_field_homo_lumo')
    if isinstance(mean_field_homo_lumo, dict):
        result['mean_field_homo_lumo'] = mean_field_homo_lumo
        result['mean_field_homo'] = mean_field_homo_lumo.get('homo')
        result['mean_field_lumo'] = mean_field_homo_lumo.get('lumo')
        result['mean_field_gap'] = mean_field_homo_lumo.get('gap')
    natural_summary = solver_results.get('natural_occupation_summary')
    if isinstance(natural_summary, dict):
        result['natural_occupation_summary'] = natural_summary
        result['natural_occupations'] = natural_summary.get('occupations')
        result['natural_occupation_fractionality'] = natural_summary.get('average_fractionality')
        result['fractional_natural_orbital_count'] = natural_summary.get('fractional_orbital_count')
    amplitude_summary = solver_results.get('double_excitation_amplitude_summary')
    if isinstance(amplitude_summary, dict):
        result['double_excitation_amplitude_summary'] = amplitude_summary
        result['max_double_excitation_amplitude'] = amplitude_summary.get('max_abs_t2')
    if 'density' in normalized_outputs and isinstance(solver_results.get('density'), list):
        result['density'] = _site_vector_payload(
            solver_results['density'],
            site_ids,
            operator='n_i_up + n_i_down',
        )
        result['density_mean'] = float(sum(solver_results['density']) / len(solver_results['density'])) if solver_results['density'] else None
    if 'double_occupancy' in normalized_outputs and isinstance(solver_results.get('double_occupancy'), list):
        result['double_occupancy'] = _site_vector_payload(
            solver_results['double_occupancy'],
            site_ids,
            operator='n_i_up n_i_down',
        )
        result['double_occupancy_mean'] = (
            float(sum(solver_results['double_occupancy']) / len(solver_results['double_occupancy']))
            if solver_results['double_occupancy']
            else None
        )
    if 'spin_correlation' in normalized_outputs and isinstance(solver_results.get('spin_correlation'), list):
        result['spin_correlation'] = _pair_matrix_payload(
            solver_results['spin_correlation'],
            site_ids,
            operator='S_i dot S_j',
        )
    if 'charge_correlation' in normalized_outputs and isinstance(solver_results.get('charge_correlation'), list):
        result['charge_correlation'] = _pair_matrix_payload(
            solver_results['charge_correlation'],
            site_ids,
            operator='(n_i - <n_i>)(n_j - <n_j>)',
        )
    deferred = set(deferred_modules)
    diagnostics_deferred = (
        'strong_correlation_diagnostics' in normalized_outputs
        and 'model.correlation_diagnostics' in deferred
    )
    if 'strong_correlation_diagnostics' in normalized_outputs and not diagnostics_deferred:
        result['strong_correlation_diagnostics'] = _compute_strong_correlation_diagnostics(normalized_spec, solver_results)
        result['analysis_text'] += '; strong_correlation={0}; {1}'.format(
            result['strong_correlation_diagnostics']['level'],
            result['strong_correlation_diagnostics']['summary'],
        )
    if diagnostics_deferred:
        result['_transient_module_context'] = {
            'model_spec': normalized_spec,
            'solver_results': solver_results,
        }
    for field in (
        'symmetry_analysis',
        'state_energies',
        'excitation_energies',
        'state_average_energy',
        'state_average_weights',
        'requested_root_count',
        'computed_root_count',
        'targeted_roots_complete',
    ):
        if solver_results.get(field) is not None:
            result[field] = copy.deepcopy(solver_results[field])
    if isinstance(solver_results.get('dmrg_result'), dict):
        result['dmrg_result'] = solver_results['dmrg_result']
        from ...providers.block2 import block2_compact_result_fields  # pylint: disable=import-outside-toplevel

        result.update(block2_compact_result_fields(solver_results['dmrg_result']))
        for field in (
            'bond_dimension_plan',
            'entanglement_diagnostics',
            'recovery_recommendation',
        ):
            if solver_results.get(field) is not None:
                result[field] = copy.deepcopy(solver_results[field])
        state_energies = result.get('state_energies')
        configuration = result['dmrg_result'].get('configuration') or {}
        if isinstance(state_energies, list):
            requested_root_count = int(configuration.get('nroots') or len(state_energies) or 1)
            result['requested_root_count'] = requested_root_count
            result['computed_root_count'] = len(state_energies)
            result['targeted_roots_complete'] = len(state_energies) >= requested_root_count
    if isinstance(solver_results.get('dmet_result'), dict):
        result['dmet_result'] = copy.deepcopy(solver_results['dmet_result'])
    if isinstance(solver_results.get('quality_checks'), list):
        result['quality_checks'] = copy.deepcopy(solver_results['quality_checks'])
    if isinstance(solver_results.get('dmet_local_observables'), dict):
        result['dmet_local_observables'] = copy.deepcopy(
            solver_results['dmet_local_observables']
        )
    state_energies = result.get('state_energies')
    if gap_requested and isinstance(state_energies, list) and len(state_energies) >= 2:
        result['gap'] = float(state_energies[1] - state_energies[0])
    if isinstance(solver_results.get('_transient_dmrg_arrays'), dict):
        result['_transient_dmrg_arrays'] = solver_results['_transient_dmrg_arrays']
    if isinstance(solver_results.get('_transient_dmet_arrays'), dict):
        result['_transient_dmet_arrays'] = solver_results['_transient_dmet_arrays']
    if isinstance(solver_results.get('_transient_provider_logs'), list):
        result['_transient_provider_logs'] = solver_results['_transient_provider_logs']
    missing_requested_outputs = []
    for output in normalized_outputs:
        if output in COMPLETE_SPECTRUM_OUTPUTS.union({'gap'}) and output not in result:
            missing_requested_outputs.append(output)
        elif output == 'excited_states' and not result.get('targeted_roots_complete'):
            missing_requested_outputs.append(output)
        elif output == 'symmetry_analysis' and not result.get('symmetry_analysis'):
            missing_requested_outputs.append(output)
        elif output == 'entanglement_diagnostics' and not result.get('entanglement_diagnostics'):
            missing_requested_outputs.append(output)
    result['requested_outputs'] = list(normalized_outputs)
    result['missing_requested_outputs'] = list(dict.fromkeys(missing_requested_outputs))
    return result


def run_model_hamiltonian_fci(spec: Dict[str, Any], outputs: Any = None) -> Dict[str, Any]:
    return run_model_hamiltonian_solver(spec, solver_name='fci', outputs=outputs)


def generate_model_hamiltonian_input_script(
    spec: Dict[str, Any],
    solver_name: Any = None,
    outputs: Any = None,
    solver_options: Any = None,
    runtime: Optional[RuntimeSpec] = None,
) -> str:
    if runtime is None:
        runtime = RuntimeSpec()
    elif isinstance(runtime, dict):
        runtime = RuntimeSpec(**validate_dataclass_input(runtime, RuntimeSpec, 'runtime'))
    normalized_spec = normalize_model_spec(spec)
    normalized_solver = normalize_solver_name(solver_name or normalized_spec.get('solver', 'fci'))
    normalized_spec['solver'] = normalized_solver
    normalized_outputs = _normalize_model_outputs(outputs)
    options = dict(solver_options) if isinstance(solver_options, dict) else {}
    spec_json = json.dumps(normalized_spec, ensure_ascii=False, indent=2)
    return f'''from __future__ import annotations

import json
from pyscf_agent.backend.model_hamiltonian.solver import run_model_hamiltonian_solver

model_spec = json.loads({spec_json!r})
outputs = {normalized_outputs!r}
solver_options = {options!r}
result = run_model_hamiltonian_solver(
    model_spec,
    solver_name={normalized_solver!r},
    outputs=outputs,
    solver_options=solver_options,
    runtime={asdict(runtime)!r},
)
result.pop('_transient_dmrg_arrays', None)
result.pop('_transient_dmet_arrays', None)
result.pop('_transient_provider_logs', None)
print(json.dumps(result, indent=2))
'''
