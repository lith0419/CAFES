from __future__ import annotations

import shutil
import math
import copy
from time import perf_counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ...backend.electronic_hamiltonian import ElectronicHamiltonian
from .availability import block2_availability
from .bond_dimension import apply_bond_dimension_plan, exact_bond_dimension_plan
from .checkpoint import read_checkpoint_manifest
from .config import (
    Block2DMRGConfig,
    MAX_NPDM_ORDER,
    MUTUAL_INFORMATION_REQUIRED_NPDM_ORDER,
    normalize_block2_options,
    resolve_block2_runtime_options,
)
from .recovery import block2_convergence_recovery
from .result_contract import finalize_block2_result_contract


def _mpo_algorithm(config: Block2DMRGConfig, n_orbitals: int) -> Any:
    from pyblock2.driver.core import MPOAlgorithmTypes

    if config.mpo_algorithm == 'conventional':
        # block2's mixed NC/CN construction warns against two-site systems.
        # Include the Conventional flag so get_qc_mpo uses HamiltonianQC.
        return MPOAlgorithmTypes.ConventionalNC if n_orbitals == 2 else MPOAlgorithmTypes.Conventional
    return MPOAlgorithmTypes.FastBipartite


def _integral_cutoff(config: Block2DMRGConfig) -> float:
    if config.integral_cutoff is not None:
        return config.integral_cutoff
    # Match block2main for Conventional; retain the driver's previous default
    # for FastBipartite. Tiny floating-point terms can break Conventional's
    # symbolic simplification in block2 0.5.3 (reproduced on canonical H6).
    return 1e-12 if config.mpo_algorithm == 'conventional' else 1e-20


def _get_qc_mpo(driver: Any, hamiltonian: ElectronicHamiltonian, config: Block2DMRGConfig, *, iprint: int) -> Any:
    # get_qc_mpo screens/symmetrizes integrals in place. PySCF reuses these
    # arrays for orbital optimization and energy/RDM validation.
    return driver.get_qc_mpo(
        h1e=copy.deepcopy(hamiltonian.h1e),
        g2e=copy.deepcopy(hamiltonian.g2e),
        ecore=float(hamiltonian.core_energy),
        algo_type=_mpo_algorithm(config, hamiltonian.n_orbitals),
        integral_cutoff=_integral_cutoff(config),
        reorder=_orbital_ordering_request(config, hamiltonian.n_orbitals),
        iprint=iprint,
    )


def _extended_schedule(values: Any, sweeps: int) -> list:
    items = list(values)
    if len(items) >= sweeps:
        return items[:sweeps]
    return items + [items[-1]] * (sweeps - len(items))


def _convergence_summary(energies: Any, discarded_weights: Any, config: Block2DMRGConfig) -> Dict[str, Any]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    energy_array = np.asarray(energies, dtype=float)
    if energy_array.ndim == 1:
        energy_array = energy_array[:, None]
    elif energy_array.ndim > 2:
        energy_array = energy_array.reshape(energy_array.shape[0], -1)
    energy_series = energy_array[:, 0]
    discarded = np.asarray(discarded_weights, dtype=float).reshape(-1)
    energy_change = None
    root_energy_changes = []
    if len(energy_series) >= 2:
        root_energy_changes = np.abs(energy_array[-1] - energy_array[-2]).astype(float).tolist()
        energy_change = max(root_energy_changes)
    final_discarded_weight = float(discarded[-1]) if len(discarded) else None
    energy_converged = energy_change is not None and energy_change <= config.energy_tolerance
    discarded_weight_converged = (
        final_discarded_weight is not None
        and final_discarded_weight <= config.discarded_weight_tolerance
    )
    return {
        'converged': bool(energy_converged and discarded_weight_converged),
        'energy_converged': bool(energy_converged),
        'discarded_weight_converged': bool(discarded_weight_converged),
        'final_energy_change': energy_change,
        'final_root_energy_changes': root_energy_changes,
        'final_discarded_weight': final_discarded_weight,
        'energy_tolerance': config.energy_tolerance,
        'discarded_weight_tolerance': config.discarded_weight_tolerance,
        'sweeps_completed': int(len(energy_series)),
    }


def _energy_error_estimate(
    energies: Any,
    discarded_weights: Any,
    *,
    enabled: bool,
) -> Dict[str, Any]:
    """Estimate the remaining variational error from sweep convergence data."""

    import numpy as np  # pylint: disable=import-outside-toplevel

    payload: Dict[str, Any] = {
        'schema': 'pyscf-agent.dmrg-energy-error-estimate.v1',
        'enabled': bool(enabled),
        'status': 'disabled' if not enabled else 'unavailable',
        'method': None,
        'estimated_absolute_error': None,
        'extrapolated_energy': None,
        'fit_points': 0,
    }
    if not enabled:
        return payload
    energy_array = np.asarray(energies, dtype=float)
    if energy_array.ndim == 1:
        energy_series = energy_array
    else:
        energy_series = energy_array.reshape(energy_array.shape[0], -1)[:, 0]
    weights = np.asarray(discarded_weights, dtype=float).reshape(-1)
    count = min(len(energy_series), len(weights))
    finite = [
        (float(weights[index]), float(energy_series[index]))
        for index in range(count)
        if np.isfinite(weights[index])
        and np.isfinite(energy_series[index])
        and float(weights[index]) > 0.0
    ][-8:]
    unique_weights = {round(item[0], 18) for item in finite}
    if len(finite) >= 3 and len(unique_weights) >= 3:
        x_values = np.asarray([item[0] for item in finite], dtype=float)
        y_values = np.asarray([item[1] for item in finite], dtype=float)
        slope, intercept = np.polyfit(x_values, y_values, 1)
        predicted = slope * x_values + intercept
        residual = float(np.sum((y_values - predicted) ** 2))
        total = float(np.sum((y_values - np.mean(y_values)) ** 2))
        r_squared = 1.0 if total <= 1.0e-30 else max(0.0, 1.0 - residual / total)
        current_energy = float(energy_series[-1])
        estimated_error = current_energy - float(intercept)
        if slope >= 0.0 and estimated_error >= 0.0:
            payload.update({
                'status': 'available',
                'method': 'linear_energy_vs_discarded_weight_extrapolation',
                'estimated_absolute_error': float(estimated_error),
                'extrapolated_energy': float(intercept),
                'slope': float(slope),
                'r_squared': r_squared,
                'fit_points': len(finite),
                'discarded_weight_range': [
                    float(np.min(x_values)),
                    float(np.max(x_values)),
                ],
                'interpretation': 'Heuristic zero-discarded-weight extrapolation; not a rigorous error bound.',
            })
            return payload
    if len(energy_series) >= 2:
        final_change = abs(float(energy_series[-1]) - float(energy_series[-2]))
        payload.update({
            'status': 'available',
            'method': 'final_sweep_energy_change',
            'estimated_absolute_error': final_change,
            'fit_points': 2,
            'interpretation': 'Fallback convergence proxy; not a variational extrapolation or rigorous error bound.',
        })
    return payload


def _stage_history(driver: Any) -> Tuple[Any, Any, Any]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    dimensions, discarded, energies = driver.get_dmrg_results()
    energy_array = np.asarray(energies, dtype=float)
    if energy_array.ndim == 1:
        energy_array = energy_array[:, None]
    return (
        np.asarray(dimensions).reshape(-1),
        np.asarray(discarded, dtype=float).reshape(-1),
        energy_array,
    )


def _run_adaptive_schedule(
    driver: Any,
    mpo: Any,
    ket: Any,
    config: Block2DMRGConfig,
) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
    """Run a bounded sequence of increasingly expressive DMRG schedules."""

    import numpy as np  # pylint: disable=import-outside-toplevel

    stages = []
    dimensions_history = []
    discarded_history = []
    energy_history = []

    def run_stage(
        stage_index: int,
        sweeps: int,
        bond_dimensions: Any,
        noises: Any,
        thresholds: Any,
        reason: str,
    ) -> Tuple[Any, Dict[str, Any]]:
        stage_energy = driver.dmrg(
            mpo,
            ket,
            n_sweeps=sweeps,
            tol=config.energy_tolerance,
            bond_dims=_extended_schedule(bond_dimensions, sweeps),
            noises=_extended_schedule(noises, sweeps),
            thrds=_extended_schedule(thresholds, sweeps),
            cutoff=config.cutoff,
            iprint=config.iprint,
        )
        dimensions, discarded, energies = _stage_history(driver)
        dimensions_history.append(dimensions)
        discarded_history.append(discarded)
        energy_history.append(energies)
        combined_energies = np.concatenate(energy_history, axis=0)
        combined_discarded = np.concatenate(discarded_history, axis=0)
        convergence = _convergence_summary(combined_energies, combined_discarded, config)
        stages.append({
            'stage': int(stage_index),
            'reason': reason,
            'sweeps_requested': int(sweeps),
            'sweeps_completed': int(len(energies)),
            'maximum_bond_dimension': int(np.max(dimensions)) if dimensions.size else int(max(bond_dimensions)),
            'final_energy_change': convergence.get('final_energy_change'),
            'final_discarded_weight': convergence.get('final_discarded_weight'),
            'converged_after_stage': bool(convergence.get('converged')),
        })
        return stage_energy, convergence

    energy, convergence = run_stage(
        0,
        config.sweeps,
        config.bond_dimensions,
        config.noises,
        config.davidson_thresholds,
        'requested_schedule',
    )
    current_bond_dimension = max(config.bond_dimensions)
    adaptive_stage = 0
    while (
        config.adaptive_schedule
        and not convergence['converged']
        and adaptive_stage < config.max_adaptive_stages
        and current_bond_dimension < config.max_bond_dimension
    ):
        adaptive_stage += 1
        next_bond_dimension = min(
            config.max_bond_dimension,
            max(current_bond_dimension + 1, int(math.ceil(
                current_bond_dimension * config.bond_dimension_growth_factor
            ))),
        )
        stage_noise = config.adaptive_noise / (10.0 ** max(0, adaptive_stage - 1))
        adaptive_noises = [stage_noise]
        if config.adaptive_sweeps > 1:
            adaptive_noises.append(stage_noise * 0.1)
        adaptive_noises.append(0.0)
        energy, convergence = run_stage(
            adaptive_stage,
            config.adaptive_sweeps,
            [next_bond_dimension],
            adaptive_noises,
            [config.davidson_thresholds[-1]],
            'increase_bond_dimension_extend_sweeps_and_reintroduce_noise',
        )
        current_bond_dimension = next_bond_dimension

    dimensions = np.concatenate(dimensions_history) if dimensions_history else np.asarray([])
    discarded = np.concatenate(discarded_history) if discarded_history else np.asarray([])
    energies = np.concatenate(energy_history, axis=0) if energy_history else np.asarray([])
    summary = {
        'schema': 'pyscf-agent.block2-adaptive-schedule.v1',
        'enabled': bool(config.adaptive_schedule),
        'stages': stages,
        'adaptive_stages_used': max(0, len(stages) - 1),
        'maximum_adaptive_stages': int(config.max_adaptive_stages),
        'maximum_bond_dimension': int(config.max_bond_dimension),
        'final_bond_dimension': int(current_bond_dimension),
        'budget_exhausted': bool(
            config.adaptive_schedule
            and not convergence['converged']
            and (
                len(stages) - 1 >= config.max_adaptive_stages
                or current_bond_dimension >= config.max_bond_dimension
            )
        ),
    }
    return energy, dimensions, discarded, energies, summary


def _set_state_average_weights(driver: Any, ket: Any, config: Block2DMRGConfig) -> None:
    """Attach the requested root weights to a block2 multi-root MPS."""

    if int(config.nroots) > 1:
        ket.weights = driver.bw.VectorFP([
            float(value) for value in config.state_average_weights
        ])


def _prepare_restart_files(
    scratch: Path,
    hamiltonian: ElectronicHamiltonian,
    config: Block2DMRGConfig,
    symmetry: str,
) -> Tuple[bool, Dict[str, Any]]:
    manifest = read_checkpoint_manifest(config.restart_manifest)
    if manifest is None:
        return False, {'requested': False, 'applied': False}
    orbital_context = manifest.get('orbital_context')
    orbital_context = orbital_context if isinstance(orbital_context, dict) else {}
    if orbital_context.get('external_restart_supported') is False:
        message = (
            'This block2 checkpoint depends on optimized orbitals that are not '
            'part of the MPS-only restart contract.'
        )
        if config.restart_required:
            raise ValueError(message)
        return False, {
            'requested': True,
            'applied': False,
            'reason': message,
            'orbital_context': orbital_context,
        }
    checks = {
        'n_orbitals': int(manifest.get('n_orbitals', -1)) == int(hamiltonian.n_orbitals),
        'n_electrons': list(manifest.get('n_electrons') or []) == [int(value) for value in hamiltonian.n_electrons],
        'spin': int(manifest.get('spin', -10 ** 9)) == int(hamiltonian.spin),
        'symmetry': str(manifest.get('symmetry') or '') == symmetry,
        'nroots': int(manifest.get('nroots') or 1) == int(config.nroots),
        'state_average_weights': list(manifest.get('state_average_weights') or [])
        == [float(value) for value in config.state_average_weights],
    }
    source_ordering = manifest.get('orbital_ordering') if isinstance(manifest.get('orbital_ordering'), dict) else {}
    source_method = str(source_ordering.get('requested_method') or 'canonical')
    source_manual_order = list(source_ordering.get('requested_order') or [])
    checks['orbital_ordering_request'] = (
        source_method == config.orbital_ordering
        and source_manual_order == list(config.orbital_order or [])
    )
    if not all(checks.values()):
        message = 'block2 restart manifest is incompatible with the requested orbital, electron, spin, symmetry, or root sector.'
        if config.restart_required:
            raise ValueError(message)
        return False, {'requested': True, 'applied': False, 'reason': message, 'checks': checks}

    source_directory = Path(str(manifest.get('scratch_directory') or '')).expanduser().resolve()
    copied_files = []
    missing_files = []
    for item in manifest.get('files') or ():
        if not isinstance(item, dict):
            continue
        relative_path = str(item.get('relative_path') or '').strip()
        if not relative_path:
            continue
        source = source_directory / relative_path
        target = scratch / relative_path
        if not source.is_file():
            missing_files.append(str(source))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied_files.append(relative_path)
    if missing_files or not copied_files:
        message = 'block2 restart checkpoint files are missing or empty.'
        if config.restart_required:
            raise ValueError(message)
        return False, {
            'requested': True,
            'applied': False,
            'reason': message,
            'missing_files': missing_files,
            'checks': checks,
        }
    provenance = dict(config.restart_provenance or {})
    return True, {
        'requested': True,
        'applied': True,
        'source_scratch_directory': str(source_directory),
        'mps_tag': str(manifest.get('mps_tag') or config.restart_tag),
        'copied_files': copied_files,
        'checks': checks,
        'source_case_id': provenance.get('source_case_id'),
        'selection': provenance.get('selection'),
        'normalized_distance': provenance.get('normalized_distance'),
        'source_artifact': provenance.get('source_artifact'),
        'source_orbital_ordering': source_ordering,
    }


def _orbital_ordering_request(config: Block2DMRGConfig, n_orbitals: int) -> Any:
    import numpy as np  # pylint: disable=import-outside-toplevel

    if config.orbital_ordering == 'canonical':
        return None
    if config.orbital_ordering == 'fiedler':
        return 'fiedler'
    order = [int(value) for value in config.orbital_order]
    if sorted(order) != list(range(int(n_orbitals))):
        raise ValueError(
            'block2 manual orbital_order must be a complete zero-based permutation of all solver orbitals.'
        )
    return np.asarray(order, dtype=int)


def _orbital_ordering_summary(driver: Any, config: Block2DMRGConfig, n_orbitals: int) -> Dict[str, Any]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    actual = getattr(driver, 'reorder_idx', None)
    permutation = (
        list(range(int(n_orbitals)))
        if actual is None
        else [int(value) for value in np.asarray(actual, dtype=int).reshape(-1)]
    )
    inverse = [int(value) for value in np.argsort(np.asarray(permutation, dtype=int))]
    return {
        'requested_method': config.orbital_ordering,
        'requested_order': list(config.orbital_order or []),
        'applied': permutation != list(range(int(n_orbitals))),
        'permutation': permutation,
        'inverse_permutation': inverse,
        'index_basis': 'electronic_hamiltonian_orbitals',
    }


def _reject_incompatible_restart_ordering(
    restart: Dict[str, Any],
    ordering: Dict[str, Any],
    config: Block2DMRGConfig,
    scratch: Path,
) -> bool:
    if not restart.get('applied'):
        return False
    source = restart.get('source_orbital_ordering')
    source = source if isinstance(source, dict) else {}
    source_permutation = list(source.get('permutation') or range(len(ordering['permutation'])))
    compatible = source_permutation == list(ordering['permutation'])
    restart.setdefault('checks', {})['orbital_ordering_permutation'] = compatible
    if compatible:
        return True
    message = 'block2 restart MPS uses an orbital permutation incompatible with the target Hamiltonian.'
    if config.restart_required:
        raise ValueError(message)
    for relative_path in restart.get('copied_files') or ():
        target = scratch / str(relative_path)
        if target.is_file():
            target.unlink()
    restart.update({'applied': False, 'reason': message})
    return False


def _single_orbital_entropy_from_spin_free_rdms(
    rdm1: Any,
    rdm2: Any,
    *,
    spin_symmetric: bool = False,
) -> Any:
    """Construct spatial-orbital entropies from spin-free 1/2-RDM diagonals."""

    import numpy as np  # pylint: disable=import-outside-toplevel

    if not spin_symmetric:
        raise ValueError('Spin-free entropy fallback requires a verified spin-symmetric singlet.')
    one_body = np.asarray(rdm1, dtype=float)
    two_body = np.asarray(rdm2, dtype=float)
    if one_body.ndim != 2 or two_body.ndim != 4:
        raise ValueError('Spin-free 1- and 2-RDMs are required for entropy fallback.')
    occupations = np.real(np.diag(one_body))
    double_occupancies = np.asarray([
        0.5 * float(np.real(two_body[index, index, index, index]))
        for index in range(one_body.shape[0])
    ])
    probabilities = np.column_stack((
        1.0 - occupations + double_occupancies,
        0.5 * occupations - double_occupancies,
        0.5 * occupations - double_occupancies,
        double_occupancies,
    ))
    if np.min(probabilities) < -1.0e-6 or np.max(probabilities) > 1.0 + 1.0e-6:
        raise ValueError('Spin-free RDMs produced invalid local orbital probabilities.')
    probabilities = np.clip(probabilities, 0.0, 1.0)
    normalization = probabilities.sum(axis=1)
    if np.any(normalization <= 1.0e-12):
        raise ValueError('Spin-free RDMs produced an empty local orbital probability distribution.')
    probabilities = probabilities / normalization[:, None]
    terms = np.zeros_like(probabilities)
    positive = probabilities > 0.0
    terms[positive] = probabilities[positive] * np.log(probabilities[positive])
    return -np.sum(terms, axis=1)


def _prepare_npdm_entanglement_state(
    driver: Any,
    ket: Any,
    hamiltonian: ElectronicHamiltonian,
    config: Block2DMRGConfig,
    symmetry: str,
) -> Tuple[Any, Dict[str, Any]]:
    """Return an NPDM-compatible state without reoptimizing the DMRG wavefunction."""

    if symmetry != 'su2' or not (config.compute_entanglement or config.compute_mutual_information):
        return ket, {
            'source_symmetry': symmetry,
            'analysis_symmetry': symmetry,
            'wavefunction_conversion': 'not_required',
        }

    import numpy as np  # pylint: disable=import-outside-toplevel
    from pyblock2.driver.core import SymmetryTypes  # pylint: disable=import-outside-toplevel

    driver.align_mps_center(ket, ref=0)
    converted = driver.mps_change_to_sz(
        ket,
        ket.info.tag + '@ENT-SZ',
        sz=int(hamiltonian.spin),
    )
    driver.symm_type = SymmetryTypes.SZ
    driver.initialize_system(
        n_sites=int(hamiltonian.n_orbitals),
        n_elec=int(hamiltonian.total_electrons),
        spin=int(hamiltonian.spin),
        orb_sym=list(hamiltonian.orbital_symmetries or [0] * hamiltonian.n_orbitals),
    )
    _get_qc_mpo(driver, hamiltonian, config, iprint=0)
    identity = driver.get_identity_mpo()
    norm_squared = float(np.real(driver.expectation(converted, identity, converted, iprint=0)))
    if not np.isfinite(norm_squared) or norm_squared <= 1.0e-14:
        raise ValueError('SU(2)-to-SZ MPS conversion produced an invalid norm.')
    converted.iscale(1.0 / np.sqrt(norm_squared))
    return converted, {
        'source_symmetry': 'su2',
        'analysis_symmetry': 'sz',
        'wavefunction_conversion': 'su2_to_sz_component',
        'spin_projection': int(hamiltonian.spin),
        'norm_squared_before_normalization': norm_squared,
        'wavefunction_reoptimized': False,
    }


def _entanglement_summary(
    driver: Any,
    ket: Any,
    config: Block2DMRGConfig,
    *,
    rdm1: Any = None,
    rdm2: Any = None,
    npdm_context: Optional[Dict[str, Any]] = None,
    spin_symmetric: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    if not (config.compute_entanglement or config.compute_mutual_information or config.compute_bipartite_entanglement):
        return None, {}
    arrays: Dict[str, Any] = {}
    payload: Dict[str, Any] = {
        'root': 0,
        'backend': (
            'block2_npdm_sz_from_su2'
            if (npdm_context or {}).get('wavefunction_conversion') == 'su2_to_sz_component'
            else 'block2_npdm'
        ),
        'npdm_context': dict(npdm_context or {}),
        'npdm_order_limit': MAX_NPDM_ORDER,
        'availability': {},
        'errors': [],
        'limitations': [],
    }
    if config.compute_entanglement or config.compute_mutual_information:
        try:
            single_entropy = np.asarray(
                driver.get_orbital_entropies(
                    ket,
                    orb_type=1,
                    use_npdm=True,
                    iprint=config.iprint,
                ),
                dtype=float,
            )
            payload['availability']['single_orbital_entropy'] = 'available'
        except (RuntimeError, TypeError, ValueError) as exc:
            payload['errors'].append({
                'quantity': 'single_orbital_entropy',
                'backend': payload['backend'],
                'message': str(exc),
            })
            try:
                single_entropy = _single_orbital_entropy_from_spin_free_rdms(
                    rdm1,
                    rdm2,
                    spin_symmetric=spin_symmetric,
                )
                payload['backend'] = 'spin_free_rdm_fallback'
                payload['availability']['single_orbital_entropy'] = 'available_fallback'
                payload['single_orbital_entropy_note'] = (
                    'Computed from spin-free singlet 1/2-RDM local occupation probabilities '
                    'because the installed block2 NPDM entropy interface was unavailable.'
                )
            except (RuntimeError, TypeError, ValueError) as fallback_exc:
                single_entropy = None
                payload['availability']['single_orbital_entropy'] = 'unavailable'
                payload['errors'].append({
                    'quantity': 'single_orbital_entropy',
                    'backend': 'spin_free_rdm_fallback',
                    'message': str(fallback_exc),
                })
        if single_entropy is not None:
            arrays['single_orbital_entropy'] = single_entropy
            payload.update({
                'single_orbital_entropy': single_entropy.tolist(),
                'max_single_orbital_entropy': float(np.max(single_entropy)) if single_entropy.size else 0.0,
                'mean_single_orbital_entropy': float(np.mean(single_entropy)) if single_entropy.size else 0.0,
            })
    if config.compute_mutual_information:
        payload['availability']['mutual_information'] = 'not_computed_order_limit'
        payload['limitations'].append({
            'quantity': 'mutual_information',
            'configured_max_npdm_order': MAX_NPDM_ORDER,
            'required_npdm_order': MUTUAL_INFORMATION_REQUIRED_NPDM_ORDER,
            'message': (
                'Orbital mutual information was not computed because its '
                'two-orbital density matrices require NPDM terms through order 4.'
            ),
        })
    if config.compute_bipartite_entanglement:
        bipartite = np.asarray(driver.get_bipartite_entanglement(ket), dtype=float)
        arrays['bipartite_entanglement'] = bipartite
        payload['bipartite_entanglement'] = bipartite.tolist()
        payload['max_bipartite_entanglement'] = float(np.max(bipartite)) if bipartite.size else 0.0
    return payload, arrays


def _symmetry_summary(driver: Any, ket: Any, hamiltonian: ElectronicHamiltonian, symmetry: str) -> Dict[str, Any]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    spin_square = float(np.real(driver.expectation(
        ket,
        driver.get_spin_square_mpo(iprint=0),
        ket,
        iprint=0,
    )))
    spin_quantum_number = float(hamiltonian.spin) / 2.0
    minimum_total_spin = abs(spin_quantum_number)
    reference_spin_square = minimum_total_spin * (minimum_total_spin + 1.0)
    inferred_spin = 0.5 * (max(0.0, 1.0 + 4.0 * spin_square) ** 0.5 - 1.0)
    payload = {
        'particle_number': int(hamiltonian.total_electrons),
        'symmetry_backend': symmetry,
        'spin_quantum_number_kind': 'total_spin' if symmetry == 'su2' else 'spin_projection',
        'twice_spin_quantum_number': int(hamiltonian.spin),
        'spin_square': spin_square,
        'inferred_total_spin': inferred_spin,
        'orbital_symmetries': [int(value) for value in hamiltonian.orbital_symmetries],
        'point_group_metadata': dict(hamiltonian.metadata).get('point_group'),
    }
    if symmetry == 'su2':
        payload.update({
            'target_total_spin': minimum_total_spin,
            'expected_spin_square': reference_spin_square,
            'spin_square_deviation': float(spin_square - reference_spin_square),
        })
    else:
        payload.update({
            'target_spin_projection': spin_quantum_number,
            'minimum_total_spin_for_projection': minimum_total_spin,
            'minimum_spin_square_for_projection': reference_spin_square,
            'spin_square_excess_above_minimum': float(spin_square - reference_spin_square),
        })
    return payload


def _checkpoint_manifest(
    scratch: Path,
    hamiltonian: ElectronicHamiltonian,
    config: Block2DMRGConfig,
    symmetry: str,
    mps_tag: str,
    orbital_ordering: Dict[str, Any],
) -> Dict[str, Any]:
    files = []
    if scratch.exists() and config.save_mps:
        for path in sorted(item for item in scratch.rglob('*') if item.is_file()):
            if any(marker in path.name for marker in ('@ENT-SZ', '@ORB-ENT-TMP')):
                continue
            files.append({
                'path': str(path),
                'relative_path': str(path.relative_to(scratch)),
                'size_bytes': int(path.stat().st_size),
            })
    return {
        'schema': 'pyscf-agent.block2-mps-manifest.v1',
        'scratch_directory': str(scratch),
        'n_orbitals': int(hamiltonian.n_orbitals),
        'n_electrons': [int(value) for value in hamiltonian.n_electrons],
        'spin': int(hamiltonian.spin),
        'symmetry': symmetry,
        'nroots': int(config.nroots),
        'state_average_weights': (
            [float(value) for value in config.state_average_weights]
            if config.nroots > 1
            else None
        ),
        'mps_tag': mps_tag,
        'orbital_ordering': dict(orbital_ordering),
        'files': files,
    }


def run_block2_dmrg(
    hamiltonian: ElectronicHamiltonian,
    options: Any = None,
    *,
    scratch_directory: Optional[str] = None,
    compute_2rdm: bool = False,
) -> Dict[str, Any]:
    availability = block2_availability()
    started_at = perf_counter()
    if not availability['available']:
        raise RuntimeError('block2 DMRG provider is unavailable: {0}'.format(availability['reason']))
    hamiltonian.validate()
    runtime_options, threading = resolve_block2_runtime_options(options)
    requested_config = normalize_block2_options(runtime_options, compute_2rdm=compute_2rdm)
    bond_dimension_plan = exact_bond_dimension_plan(hamiltonian, requested_config)
    root_capacity = int(bond_dimension_plan.get('target_root_sector_dimension') or 0)
    if int(requested_config.nroots) > root_capacity:
        raise ValueError(
            'block2 nroots={0} exceeds the {1}-state capacity of the requested '
            'electron/spin sector.'.format(requested_config.nroots, root_capacity)
        )
    config = apply_bond_dimension_plan(requested_config, bond_dimension_plan)
    symmetry = config.symmetry
    if symmetry == 'auto':
        symmetry = 'su2'
    # Keep the submitted shared mount namespace in persisted checkpoints.
    # A symlink's physical target can differ between compute nodes.
    scratch = Path(scratch_directory or (Path.cwd() / 'block2-scratch')).expanduser().absolute()
    scratch.mkdir(parents=True, exist_ok=True)
    restart_applied, restart = _prepare_restart_files(scratch, hamiltonian, config, symmetry)

    import numpy as np  # pylint: disable=import-outside-toplevel
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes  # pylint: disable=import-outside-toplevel

    symmetry_type = SymmetryTypes.SU2 if symmetry == 'su2' else SymmetryTypes.SZ
    config_payload = config.to_dict()
    restart_manifest = config_payload.get('restart_manifest')
    if isinstance(restart_manifest, dict):
        config_payload['restart_manifest'] = {
            key: restart_manifest.get(key)
            for key in ('schema', 'scratch_directory', 'nroots', 'mps_tag')
            if restart_manifest.get(key) is not None
        }
    config_payload['symmetry'] = symmetry
    config_payload['integral_cutoff'] = _integral_cutoff(config)
    driver = DMRGDriver(
        scratch=str(scratch),
        clean_scratch=not restart_applied,
        symm_type=symmetry_type,
        n_threads=config.n_threads,
        n_mkl_threads=config.n_mkl_threads,
        stack_mem=config.stack_memory_bytes,
    )
    try:
        if hasattr(driver.bw.b, 'Random'):
            driver.bw.b.Random.rand_seed(config.seed)
        driver.initialize_system(
            n_sites=int(hamiltonian.n_orbitals),
            n_elec=int(hamiltonian.total_electrons),
            spin=int(hamiltonian.spin),
            orb_sym=list(hamiltonian.orbital_symmetries or [0] * hamiltonian.n_orbitals),
        )
        mpo_algorithm = _mpo_algorithm(config, hamiltonian.n_orbitals)
        stage_started_at = perf_counter()
        mpo = _get_qc_mpo(driver, hamiltonian, config, iprint=config.iprint)
        timings = {'mpo_construction_seconds': perf_counter() - stage_started_at}
        orbital_ordering = _orbital_ordering_summary(driver, config, hamiltonian.n_orbitals)
        restart_applied = _reject_incompatible_restart_ordering(
            restart,
            orbital_ordering,
            config,
            scratch,
        )
        mps_tag = str(restart.get('mps_tag') or config.restart_tag or 'GS')
        if restart_applied:
            ket = driver.load_mps(tag=mps_tag, nroots=config.nroots)
        else:
            mps_tag = 'GS'
            ket = driver.get_random_mps(
                tag=mps_tag,
                bond_dim=int(config.bond_dimensions[0]),
                nroots=config.nroots,
            )
        _set_state_average_weights(driver, ket, config)
        stage_started_at = perf_counter()
        energy, bond_dimensions, discarded_weights, sweep_energies, adaptive_schedule = _run_adaptive_schedule(
            driver,
            mpo,
            ket,
            config,
        )
        timings['dmrg_sweeps_seconds'] = perf_counter() - stage_started_at
        convergence = _convergence_summary(sweep_energies, discarded_weights, config)
        convergence['adaptive_schedule'] = adaptive_schedule
        energy_error_estimate = _energy_error_estimate(
            sweep_energies,
            discarded_weights,
            enabled=config.estimate_energy_error,
        )
        stage_started_at = perf_counter()
        root_energies = np.asarray(energy, dtype=float).reshape(-1)
        root_kets = (
            [driver.split_mps(ket, root, mps_tag + '-ROOT{0}'.format(root)) for root in range(config.nroots)]
            if config.nroots > 1
            else [ket]
        )
        analysis_ket = root_kets[0]
        rdm1_root_kets = (
            root_kets
            if config.rdm1_root_scope == 'all_states'
            else root_kets[:1]
        )
        rdm2_root_kets = (
            root_kets
            if config.rdm2_root_scope == 'all_states'
            else root_kets[:1]
        )
        root_rdm1_raw = [
            driver.get_1pdm(root_ket) if config.compute_1rdm else None
            for root_ket in rdm1_root_kets
        ]
        root_rdm2_raw = [
            driver.get_2pdm(root_ket) if config.compute_2rdm else None
            for root_ket in rdm2_root_kets
        ]
        rdm1_raw = root_rdm1_raw[0]
        rdm2_raw = root_rdm2_raw[0]
        if symmetry == 'su2':
            analyzed_rdm1 = [None if value is None else np.asarray(value) for value in root_rdm1_raw]
            analyzed_rdm2 = [
                None if value is None else np.asarray(value).transpose(0, 3, 1, 2)
                for value in root_rdm2_raw
            ]
            rdm1 = analyzed_rdm1[0]
            rdm2 = analyzed_rdm2[0]
            root_rdm1 = analyzed_rdm1 if config.rdm1_root_scope == 'all_states' else None
            root_rdm2 = analyzed_rdm2 if config.rdm2_root_scope == 'all_states' else None
            spin_rdms = None
        else:
            root_rdm1 = None
            root_rdm2 = None
            spin_rdm1 = None if rdm1_raw is None else [np.asarray(value) for value in rdm1_raw]
            spin_rdm2 = None if rdm2_raw is None else [np.asarray(value).transpose(0, 3, 1, 2) for value in rdm2_raw]
            rdm1 = None if spin_rdm1 is None else spin_rdm1[0] + spin_rdm1[1]
            rdm2 = None
            spin_rdms = None if spin_rdm1 is None else {
                'rdm1_alpha': spin_rdm1[0],
                'rdm1_beta': spin_rdm1[1],
                'rdm2_alpha_alpha': spin_rdm2[0] if spin_rdm2 is not None else None,
                'rdm2_alpha_beta': spin_rdm2[1] if spin_rdm2 is not None else None,
                'rdm2_beta_beta': spin_rdm2[2] if spin_rdm2 is not None else None,
            }
        timings['rdm_seconds'] = perf_counter() - stage_started_at
        natural_occupations = None
        if rdm1 is not None:
            natural_occupations = np.linalg.eigvalsh(0.5 * (rdm1 + rdm1.T.conj()))[::-1].real
        symmetry_analysis = (
            _symmetry_summary(driver, analysis_ket, hamiltonian, symmetry)
            if config.compute_symmetry_analysis
            else None
        )
        npdm_ket = analysis_ket
        npdm_context: Dict[str, Any] = {
            'source_symmetry': symmetry,
            'analysis_symmetry': symmetry,
            'wavefunction_conversion': 'not_required',
        }
        try:
            npdm_ket, npdm_context = _prepare_npdm_entanglement_state(
                driver,
                analysis_ket,
                hamiltonian,
                config,
                symmetry,
            )
        except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
            npdm_context = {
                'source_symmetry': symmetry,
                'analysis_symmetry': symmetry,
                'wavefunction_conversion': 'failed',
                'conversion_error': str(exc),
            }
        entanglement, entanglement_arrays = _entanglement_summary(
            driver,
            npdm_ket,
            config,
            rdm1=rdm1,
            rdm2=rdm2,
            npdm_context=npdm_context,
            spin_symmetric=(symmetry == 'su2' and hamiltonian.spin == 0),
        )
        arrays = {
            'bond_dimensions': np.asarray(bond_dimensions),
            'discarded_weights': np.asarray(discarded_weights),
            'sweep_energies': np.asarray(sweep_energies),
        }
        if rdm1 is not None:
            arrays['rdm1'] = np.asarray(rdm1)
        if rdm2 is not None:
            arrays['rdm2'] = np.asarray(rdm2)
        if root_rdm1 is not None and all(value is not None for value in root_rdm1):
            arrays['root_rdm1'] = np.asarray(root_rdm1)
        if root_rdm2 is not None and all(value is not None for value in root_rdm2):
            arrays['root_rdm2'] = np.asarray(root_rdm2)
        if spin_rdms is not None:
            arrays.update({key: value for key, value in spin_rdms.items() if value is not None})
        arrays.update(entanglement_arrays)
        result = {
            'schema': 'pyscf-agent.block2-dmrg-result.v1',
            'solver': 'block2_dmrg',
            'provider': availability,
            'energy': float(root_energies[0]),
            'state_energies': root_energies.tolist(),
            'excitation_energies': (root_energies - root_energies[0]).tolist(),
            'excited_state_mode': config.excited_state_mode if config.nroots > 1 else None,
            'state_average_energy': (
                float(np.dot(
                    np.asarray(config.state_average_weights, dtype=float),
                    root_energies,
                ))
                if config.nroots > 1
                else None
            ),
            'state_average_weights': (
                list(config.state_average_weights) if config.nroots > 1 else None
            ),
            'rdm_evaluation_scope': config.rdm_root_scope,
            'rdm1_evaluation_scope': config.rdm1_root_scope,
            'rdm2_evaluation_scope': config.rdm2_root_scope,
            'rdm_evaluation_roots': (
                list(range(config.nroots))
                if config.rdm_root_scope == 'all_states'
                else [0]
            ),
            'rdm1_evaluation_roots': (
                list(range(config.nroots))
                if config.rdm1_root_scope == 'all_states'
                else [0]
            ),
            'rdm2_evaluation_roots': (
                list(range(config.nroots))
                if config.rdm2_root_scope == 'all_states'
                else [0]
            ),
            'reported_diagnostics_scope': 'ground_state',
            'reported_diagnostics_roots': [0],
            'converged': convergence['converged'],
            'convergence': convergence,
            'energy_error_estimate': energy_error_estimate,
            'adaptive_schedule': adaptive_schedule,
            'bond_dimension_plan': bond_dimension_plan,
            'configuration': config_payload,
            'mpo_algorithm': {'requested': config.mpo_algorithm, 'effective': mpo_algorithm.name},
            'timings': timings,
            'threading': threading,
            'hamiltonian': hamiltonian.summary(),
            'orbital_ordering': orbital_ordering,
            'natural_occupations': None if natural_occupations is None else natural_occupations.tolist(),
            'entanglement_diagnostics': entanglement,
            'symmetry_analysis': symmetry_analysis,
            'restart': restart,
            'checkpoint_manifest': _checkpoint_manifest(
                scratch,
                hamiltonian,
                config,
                symmetry,
                mps_tag,
                orbital_ordering,
            ),
            '_transient_dmrg_arrays': arrays,
        }
        recovery = block2_convergence_recovery(result)
        if recovery is not None:
            result['recovery_recommendation'] = recovery
        finalize_block2_result_contract(result, hamiltonian, config, symmetry)
        timings['total_seconds'] = perf_counter() - started_at
        return result
    finally:
        driver.finalize()
