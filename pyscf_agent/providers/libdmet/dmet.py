from __future__ import annotations

import copy
import inspect
import itertools
import math
import sys
import threading
from functools import partial
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ...input_validation import boolean, integer, finite_float, reject_unknown_fields
from ...embedding.reference_density import (
    build_reference_density_seed,
    normalize_reference_density_bias,
    normalize_reference_density_guess,
    validate_reference_density_guess,
)
from .fragment_fitting import (
    fit_fragment_local_correlation_potential,
    prepare_density_fit_lattice,
)
from .density_restart import normalize_density_source, load_density_seed, state_metadata
from .translation import audit_builder_translation
from .translation import resolve_primitive_cell_layout as _primitive_cell_layout
from .block2_impurity import Block2DmetImpuritySolver
from .rdm_bridge import impurity_spin_rdm2
from .impurity_scf import configure_ccsd_impurity_scf
from .hexagonal_fragments import honeycomb_hexagonal_fragments


DMET_RESULT_SCHEMA = 'pyscf-agent.dmet-result.v1'
DMET_ITERATION_HISTORY_SCHEMA = 'pyscf-agent.dmet-iteration-history.v1'


_LIBDMET_NATIVE_FIT_BETA = 1000.0
_LIBDMET_NATIVE_FIT_MAX_ITERATIONS = 300
_FRACTIONAL_OCCUPATION_TOLERANCE = 1.0e-8
_PROJECTED_ELECTRON_COUNT_TOLERANCE = 1.0e-5
_RESULT_IDENTITY_RELATIVE_TOLERANCE = 1.0e-10


def _expanded_global_density_matrix(lattice: Any, global_density: Any, np: Any) -> Any:
    """Expand libDMET's translational stripe 1RDM into the full site basis."""

    density = np.asarray(global_density)
    if density.ndim == 3 and density.shape[-1] == density.shape[-2]:
        return density
    try:
        expanded = np.asarray(lattice.expand(density))
    except (AttributeError, AssertionError, TypeError, ValueError):
        if density.ndim == 4 and density.shape[1] == 1:
            expanded = density[:, 0]
        else:
            return None
    if expanded.ndim != 3 or expanded.shape[-1] != expanded.shape[-2]:
        return None
    return expanded


_LIBDMET_UIHF_COMPATIBILITY_LOCK = threading.RLock()
_LIBDMET_OUTPUT_LOCK = threading.RLock()


def _dmet_option_contract() -> Tuple[Dict[str, Any], Dict[str, str], Tuple[str, ...]]:
    """Read DMET defaults, resolution policies, and allowed names from Registry."""

    from ...registry import default_registry  # pylint: disable=import-outside-toplevel

    capability = default_registry().capability('dmet', namespace='model_hamiltonian.solver')
    option_contracts = (
        capability.metadata.get('solver_options')
        if capability is not None and isinstance(capability.metadata, Mapping)
        else None
    )
    if not isinstance(option_contracts, Mapping):
        raise RuntimeError('Registry does not define the model_hamiltonian.solver.dmet option contract')
    defaults = {
        str(name): copy.deepcopy(contract['default'])
        for name, contract in option_contracts.items()
        if isinstance(contract, Mapping) and 'default' in contract
    }
    resolution_policies = {
        str(name): str(contract.get('resolution_policy') or '').strip()
        for name, contract in option_contracts.items()
        if isinstance(contract, Mapping) and contract.get('resolution_policy')
    }
    required_defaults = {
        'impurity_solver', 'impurity_solver_options', 'impurity_size',
        'impurity_shape', 'impurity_site_ids', 'fragments', 'reference',
        'max_iterations', 'energy_tolerance', 'density_tolerance',
        'density_fit_tolerance',
        'diis_start', 'diis_space', 'diis_enabled', 'correlation_potential_mixing',
        'trace_fix_start', 'interacting_bath',
        'initial_correlation_potential', 'reference_density_guess',
        'reference_density_bias', 'solver_tolerance', 'solver_max_cycle',
        'solver_max_memory_mb', 'impurity_scf_diis', 'bath_spin_dimension_policy',
    }
    missing = sorted(required_defaults.difference(defaults))
    if missing:
        raise RuntimeError('Registry DMET option defaults are incomplete: {0}'.format(', '.join(missing)))
    return defaults, resolution_policies, tuple(option_contracts)


def _positive_int(value: Any, default: int, field_name: str) -> int:
    raw = default if value is None else value
    try:
        normalized = integer(raw, field_name)
    except (TypeError, ValueError) as exc:
        raise ValueError('{0} must be a positive integer'.format(field_name)) from exc
    if isinstance(raw, bool) or normalized < 1:
        raise ValueError('{0} must be a positive integer'.format(field_name))
    return normalized


def _positive_float(value: Any, default: float, field_name: str) -> float:
    raw = default if value is None else value
    try:
        normalized = finite_float(raw, field_name)
    except (TypeError, ValueError) as exc:
        raise ValueError('{0} must be positive'.format(field_name)) from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError('{0} must be positive'.format(field_name))
    return normalized


def _shape(value: Any) -> List[int]:
    if value in (None, ''):
        return []
    if isinstance(value, str):
        values: Sequence[Any] = [item.strip() for item in value.split(',') if item.strip()]
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        values = [value]
    return [_positive_int(item, 1, 'impurity_shape') for item in values]


def _integer_list(value: Any, field_name: str) -> List[int]:
    if value in (None, ''):
        return []
    if isinstance(value, str):
        values: Sequence[Any] = [item.strip() for item in value.split(',') if item.strip()]
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        values = [value]
    normalized: List[int] = []
    for item in values:
        try:
            integer = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError('{0} must contain integer site ids'.format(field_name)) from exc
        if isinstance(item, bool) or integer < 0:
            raise ValueError('{0} must contain non-negative site ids'.format(field_name))
        if integer not in normalized:
            normalized.append(integer)
    return normalized


def _fragment_list(value: Any) -> List[Dict[str, Any]]:
    if value in (None, ''):
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError('fragments must be a list of fragment mappings')
    normalized: List[Dict[str, Any]] = []
    seen_ids = set()
    for position, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValueError('fragments must be a list of fragment mappings')
        fragment_id = str(raw.get('fragment_id') or 'fragment-{0}'.format(position + 1)).strip()
        if not fragment_id or fragment_id in seen_ids:
            raise ValueError('fragment_id values must be non-empty and unique')
        seen_ids.add(fragment_id)
        site_ids = _integer_list(raw.get('site_ids'), 'fragment site_ids')
        orbital_indices = _integer_list(raw.get('orbital_indices'), 'fragment orbital_indices')
        if not site_ids and not orbital_indices:
            raise ValueError('each fragment requires site_ids or orbital_indices')
        normalized.append({
            'fragment_id': fragment_id,
            'label': str(raw.get('label') or fragment_id),
            'site_ids': site_ids,
            'orbital_indices': orbital_indices,
            'metadata': copy.deepcopy(raw.get('metadata') or {}),
        })
    return normalized


def normalize_dmet_options(options: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    defaults, resolution_policies, allowed_names = _dmet_option_contract()
    supplied = copy.deepcopy(dict(options)) if isinstance(options, Mapping) else {}
    normalized = copy.deepcopy(defaults)
    if supplied:
        if 'nroots' in supplied:
            raise ValueError('The current DMET workflow is ground-state only; remove nroots')
        if 'correlation_potential_tolerance' in supplied:
            raise ValueError(
                'DMET convergence follows the native energy and 1RDM criteria; '
                'replace correlation_potential_tolerance with density_tolerance'
            )
        unknown = sorted(str(name) for name in supplied if name not in allowed_names)
        if unknown:
            raise ValueError(
                'Unknown DMET solver option(s): {0}. Use the registered DMET solver.options contract.'.format(
                    ', '.join(unknown)
                )
            )
        normalized.update(supplied)

    if 'execution_mode' not in supplied:
        if resolution_policies.get('execution_mode') != 'translational_if_eligible_else_finite_graph':
            raise RuntimeError('Registry does not define the DMET execution-mode resolution policy')
        normalized['execution_mode'] = 'auto'
    if 'fragment_definition' not in supplied:
        if resolution_policies.get('fragment_definition') != 'explicit_selector_else_primitive_cell':
            raise RuntimeError('Registry does not define the DMET fragment resolution policy')
        normalized['fragment_definition'] = 'auto'

    impurity_solver = str(normalized.get('impurity_solver') or defaults['impurity_solver']).strip().lower().replace('-', '_')
    if impurity_solver in ('full_ci', 'fullci', 'exact_diagonalization', 'ed'):
        impurity_solver = 'fci'
    if impurity_solver in ('coupled_cluster', 'coupled_cluster_singles_doubles'):
        impurity_solver = 'ccsd'
    if impurity_solver in ('block2', 'block2_dmrg', 'dmrg'):
        impurity_solver = 'block2_dmrg'
    if impurity_solver not in ('fci', 'ccsd', 'block2_dmrg'):
        raise ValueError('DMET impurity_solver must be fci, ccsd, or block2_dmrg')
    normalized['impurity_solver'] = impurity_solver

    execution_mode = str(normalized.get('execution_mode') or 'auto').strip().lower().replace('-', '_')
    execution_mode = {
        'translated': 'translational',
        'translated_representative': 'translational',
        'graph': 'finite_graph',
    }.get(execution_mode, execution_mode)
    if execution_mode not in ('auto', 'translational', 'finite_graph'):
        raise ValueError('DMET execution_mode must be translational or finite_graph')
    normalized['execution_mode'] = execution_mode

    reference = str(normalized.get('reference') or defaults['reference']).strip().lower().replace('-', '_')
    aliases = {'rhf': 'restricted', 'uhf': 'unrestricted', 'paramagnetic': 'restricted', 'af': 'unrestricted'}
    reference = aliases.get(reference, reference)
    if reference not in ('restricted', 'unrestricted'):
        raise ValueError('DMET reference must be restricted or unrestricted')
    normalized['reference'] = reference
    impurity_solver_options = normalized.get('impurity_solver_options') or {}
    if not isinstance(impurity_solver_options, Mapping):
        raise ValueError('DMET impurity_solver_options must be a mapping')
    impurity_solver_options = copy.deepcopy(dict(impurity_solver_options))
    if impurity_solver == 'block2_dmrg':
        reserved = {
            'nroots',
            'state_average_weights',
            'restart_manifest',
            'restart_provenance',
            'restart_tag',
            'compute_1rdm',
            'compute_2rdm',
            'compute_entanglement',
            'compute_mutual_information',
            'compute_bipartite_entanglement',
            'compute_symmetry_analysis',
            'rdm_root_scope',
            'rdm1_root_scope',
            'rdm2_root_scope',
        }.intersection(impurity_solver_options)
        if reserved:
            raise ValueError(
                'DMET manages these block2 impurity options internally: {0}'.format(
                    ', '.join(sorted(reserved))
                )
            )
        from ..block2.config import normalize_block2_options  # pylint: disable=import-outside-toplevel

        normalize_block2_options({
            **impurity_solver_options,
            'nroots': 1,
            'symmetry': 'su2' if reference == 'restricted' else 'sz',
            'compute_1rdm': True,
            'compute_2rdm': True,
        }, compute_2rdm=True)
    elif impurity_solver == 'ccsd' or (impurity_solver == 'fci' and impurity_solver_options):
        from ...registry import default_registry

        label = 'DMET {0}'.format(impurity_solver.upper())
        capability = default_registry().capability(impurity_solver, namespace='embedding.impurity_solver')
        reject_unknown_fields(impurity_solver_options, capability.metadata['solver_options'],
                              label + ' impurity_solver_options')
        beta = finite_float(
            impurity_solver_options.get('beta', capability.metadata['solver_options']['beta']['default']),
            label + ' beta',
        )
        if beta <= 0:
            raise ValueError(label + ' beta must be positive')
        impurity_solver_options['beta'] = beta
        if 'smearing' in impurity_solver_options:
            impurity_solver_options['smearing'] = boolean(
                impurity_solver_options['smearing'], label + ' smearing',
            )
    elif impurity_solver_options:
        raise ValueError(
            'DMET impurity_solver_options are currently available only for block2_dmrg, ccsd or fci'
        )
    normalized['reference_density_source'] = normalize_density_source(normalized.get('reference_density_source'))
    normalized['impurity_solver_options'] = impurity_solver_options
    normalized['bath_spin_dimension_policy'] = str(normalized['bath_spin_dimension_policy']).lower()
    if normalized['bath_spin_dimension_policy'] not in ('native', 'max'):
        raise ValueError('bath_spin_dimension_policy must be native or max')
    normalized['impurity_scf_diis'] = boolean(
        normalized.get('impurity_scf_diis'), 'DMET impurity_scf_diis',
    )
    if impurity_solver == 'block2_dmrg' and not normalized['impurity_scf_diis']:
        raise ValueError('DMET block2 does not use a preliminary impurity SCF; impurity_scf_diis does not apply')
    impurity_size = normalized.get('impurity_size')
    normalized['impurity_size'] = (
        None
        if impurity_size in (None, '')
        else _positive_int(impurity_size, 1, 'impurity_size')
    )
    normalized['impurity_shape'] = _shape(normalized.get('impurity_shape'))
    normalized['impurity_site_ids'] = _integer_list(
        normalized.get('impurity_site_ids'),
        'impurity_site_ids',
    )
    normalized['fragments'] = _fragment_list(normalized.get('fragments'))
    fragment_definition = str(
        normalized.get('fragment_definition') or 'auto'
    ).strip().lower().replace('-', '_')
    if fragment_definition not in ('auto', 'primitive_cell', 'cell_shape', 'site_count', 'honeycomb_hexagon'):
        raise ValueError(
            'DMET fragment_definition must be primitive_cell, cell_shape, site_count, or honeycomb_hexagon'
        )
    normalized['fragment_definition'] = fragment_definition
    normalized['max_iterations'] = _positive_int(normalized.get('max_iterations'), defaults['max_iterations'], 'max_iterations')
    normalized['correlation_potential_mixing'] = _positive_float(
        normalized.get('correlation_potential_mixing'), defaults['correlation_potential_mixing'],
        'correlation_potential_mixing',
    )
    if normalized['correlation_potential_mixing'] > 1.0:
        raise ValueError('correlation_potential_mixing must be in (0, 1]')
    normalized['diis_enabled'] = boolean(normalized.get('diis_enabled'), 'DMET diis_enabled')
    normalized['diis_start'] = _positive_int(normalized.get('diis_start'), defaults['diis_start'], 'diis_start')
    normalized['diis_space'] = _positive_int(normalized.get('diis_space'), defaults['diis_space'], 'diis_space')
    normalized['trace_fix_start'] = _positive_int(normalized.get('trace_fix_start'), defaults['trace_fix_start'], 'trace_fix_start')
    normalized['solver_max_cycle'] = _positive_int(normalized.get('solver_max_cycle'), defaults['solver_max_cycle'], 'solver_max_cycle')
    normalized['solver_max_memory_mb'] = _positive_int(
        normalized.get('solver_max_memory_mb'), defaults['solver_max_memory_mb'], 'solver_max_memory_mb'
    )
    normalized['energy_tolerance'] = _positive_float(
        normalized.get('energy_tolerance'), defaults['energy_tolerance'], 'energy_tolerance'
    )
    normalized['density_tolerance'] = _positive_float(
        normalized.get('density_tolerance'),
        defaults['density_tolerance'],
        'density_tolerance',
    )
    normalized['density_fit_tolerance'] = _positive_float(
        normalized.get('density_fit_tolerance'),
        defaults['density_fit_tolerance'],
        'density_fit_tolerance',
    )
    normalized['solver_tolerance'] = _positive_float(
        normalized.get('solver_tolerance'), defaults['solver_tolerance'], 'solver_tolerance'
    )
    normalized['interacting_bath'] = bool(normalized.get('interacting_bath', defaults['interacting_bath']))
    initial_correlation_potential = str(
        normalized.get('initial_correlation_potential') or 'zero'
    ).strip().lower().replace('-', '_')
    if initial_correlation_potential != 'zero':
        raise ValueError(
            'DMET initial_correlation_potential currently supports only zero'
        )
    normalized['initial_correlation_potential'] = 'zero'
    normalized['reference_density_guess'] = normalize_reference_density_guess(
        normalized.get('reference_density_guess')
    )
    normalized['reference_density_bias'] = normalize_reference_density_bias(
        normalized.get('reference_density_bias')
    )
    return normalized


def _mix_correlation_potential(
    current: Any, fitted: Any, configuration: Mapping[str, Any],
    iteration: int, adiis: Any, np: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """Damp the outer update after trace fixing and optional DIIS extrapolation."""
    previous = np.array(current, copy=True).reshape(-1)
    proposed = np.hstack(fitted).copy()
    raw_change = float(np.linalg.norm(proposed - previous) / previous.size)
    use_diis = configuration['diis_enabled'] and iteration >= configuration['diis_start']
    if use_diis:
        proposed = adiis.update(proposed)
    mixing = configuration['correlation_potential_mixing']
    # Preserve the legacy update exactly when damping is disabled.
    updated = proposed if mixing == 1.0 else previous + mixing * (proposed - previous)
    return updated, {
        'correlation_potential_mixing': mixing,
        'correlation_potential_diis_applied': bool(use_diis),
        'correlation_potential_fit_change_per_parameter': raw_change,
        'correlation_potential_proposal_change_per_parameter': float(
            np.linalg.norm(proposed - previous) / previous.size
        ),
        'correlation_potential_change_per_parameter': float(
            np.linalg.norm(updated - previous) / previous.size
        ),
    }


def _mean_field_rdm1_change(current_density: Any, previous_density: Any, np: Any) -> float:
    current = np.asarray(current_density)
    if previous_density is None:
        return float(np.max(np.abs(current)))
    return float(np.max(np.abs(current - np.asarray(previous_density))))


def _embedding_basis_policy_for_mean_field(
    mean_field_result: Mapping[str, Any],
    np: Any,
) -> Tuple[str, Dict[str, Any]]:
    """Choose a bath construction that preserves fractional occupations."""

    raw_occupations = mean_field_result.get('mo_occ')
    if raw_occupations is None:
        return 'svd', {
            'kind': 'svd',
            'reason': 'mean_field_occupations_unavailable',
            'fractional_occupation_count': None,
            'fractional_occupation_tolerance': _FRACTIONAL_OCCUPATION_TOLERANCE,
        }
    occupations = np.asarray(raw_occupations)
    if np.max(np.abs(occupations.imag), initial=0.0) > _FRACTIONAL_OCCUPATION_TOLERANCE:
        raise ValueError('DMET mean-field occupations must be real-valued')
    occupations = occupations.real.astype(float)
    fractional = np.logical_and(
        occupations > _FRACTIONAL_OCCUPATION_TOLERANCE,
        occupations < 1.0 - _FRACTIONAL_OCCUPATION_TOLERANCE,
    )
    fractional_count = int(np.count_nonzero(fractional))
    kind = 'eig' if fractional_count else 'svd'
    return kind, {
        'kind': kind,
        'reason': (
            'fractional_mean_field_occupations'
            if fractional_count
            else 'idempotent_mean_field_occupations'
        ),
        'occupation_count': int(occupations.size),
        'fractional_occupation_count': fractional_count,
        'fractional_occupation_tolerance': _FRACTIONAL_OCCUPATION_TOLERANCE,
        'minimum_occupation': (
            float(np.min(occupations)) if occupations.size else None
        ),
        'maximum_occupation': (
            float(np.max(occupations)) if occupations.size else None
        ),
    }


def _embedding_electron_count_from_projected_density(
    folded_density: Any,
    orbital_count: int,
    reference: str,
    np: Any,
    *,
    fractional_occupation_count: int = 0,
    fractional_sector_electron_count: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Derive a stable impurity sector after the bath projection.

    An idempotent auxiliary state has a well-defined integer particle count in
    its projected embedding space.  A fractionally occupied ensemble does not:
    the eigenvalue bath carries additional partially occupied environment
    modes, while the impurity solver still requires one fixed-N sector.  Keep
    the established valence-sector count for that case and record the choice.
    """

    density = np.asarray(folded_density)
    if density.ndim == 2:
        density = density[np.newaxis, :, :]
    if density.ndim != 3 or density.shape[-2:] != (orbital_count, orbital_count):
        raise ValueError(
            'Projected DMET density has shape {0}; expected spin blocks of {1}x{1}'.format(
                tuple(density.shape),
                int(orbital_count),
            )
        )
    traces = np.trace(density, axis1=-2, axis2=-1)
    imaginary_error = float(np.max(np.abs(traces.imag), initial=0.0))
    if imaginary_error > _PROJECTED_ELECTRON_COUNT_TOLERANCE:
        raise ValueError(
            'Projected DMET electron count has a non-negligible imaginary component '
            '({0:.3e})'.format(imaginary_error)
        )
    real_traces = traces.real.astype(float)
    projected_count = float(np.sum(real_traces))
    fractional_occupation_count = int(fractional_occupation_count)
    if fractional_occupation_count:
        if fractional_sector_electron_count is None:
            raise ValueError(
                'Fractionally occupied DMET references require an explicit fixed-N '
                'embedding sector'
            )
        electron_count = int(fractional_sector_electron_count)
        rounding_error = None
        selection = 'valence_sector_for_fractional_mean_field'
    else:
        electron_count = int(np.rint(projected_count))
        rounding_error = abs(projected_count - electron_count)
        if rounding_error > _PROJECTED_ELECTRON_COUNT_TOLERANCE:
            raise ValueError(
                'Projected DMET density trace {0:.12f} is not an integer electron count '
                '(tolerance={1:.3e})'.format(
                    projected_count,
                    _PROJECTED_ELECTRON_COUNT_TOLERANCE,
                )
            )
        selection = 'projected_mean_field_density_trace'
    capacity = 2 * int(orbital_count)
    if electron_count < 0 or electron_count > capacity:
        raise ValueError(
            'Projected DMET electron count {0} does not fit {1} spatial orbitals'.format(
                electron_count,
                int(orbital_count),
            )
        )
    return electron_count, {
        'selection': selection,
        'reference': str(reference),
        'density_representation': (
            'spin_summed' if str(reference) == 'restricted' else 'spin_resolved'
        ),
        'projected_density_traces': [float(value) for value in real_traces],
        'projected_electron_count': projected_count,
        'selected_electron_count': electron_count,
        'rounded_electron_count': (
            None if fractional_occupation_count else electron_count
        ),
        'fractional_sector_electron_count': (
            electron_count if fractional_occupation_count else None
        ),
        'rounding_error': rounding_error,
        'rounding_tolerance': _PROJECTED_ELECTRON_COUNT_TOLERANCE,
        'embedding_spatial_orbital_count': int(orbital_count),
        'fractional_occupation_count': fractional_occupation_count,
    }


def _native_dmet_converged(
    energy_change: Optional[float],
    density_change: float,
    density_fit_error: float,
    configuration: Mapping[str, Any],
) -> bool:
    """Apply the native DMET stopping rule: stationary energy and mean-field 1RDM.

    The density-fit residual is reported but does not gate convergence: a single
    determinant need not reproduce a correlated 1RDM (for example a strong
    charge-density wave), so the residual can plateau at a stationary point.
    """
    del density_fit_error
    return bool(
        energy_change is not None
        and abs(float(energy_change)) < float(configuration['energy_tolerance'])
        and float(density_change) < float(configuration['density_tolerance'])
    )


def _solve_impurity_hamiltonians_with_fitting(dmet: Any, *args: Any, **kwargs: Any) -> Any:
    """Reject invalid chemical-potential fits across old and new SciPy versions."""

    try:
        result = dmet.SolveImpHam_with_fitting(*args, **kwargs)
    except ValueError as exc:
        if 'all x values are identical' not in str(exc):
            raise
        raise RuntimeError(
            'DMET impurity chemical-potential fitting encountered a flat '
            'electron-number response; verify the projected impurity electron '
            'sector and embedding-orbital capacity'
        ) from None
    # Newer SciPy returns NaN instead of the historical regression ValueError.
    # Check the public libDMET result before updating last_dmu or transforming it.
    if not math.isfinite(float(result[3])):
        raise RuntimeError(
            'DMET impurity chemical-potential fitting returned a non-finite shift; '
            'check for a flat electron-number response and verify the projected '
            'impurity electron sector and embedding-orbital capacity'
        )
    return result


def _native_convergence_contract(
    configuration: Mapping[str, Any],
    *,
    energy_measure: str,
) -> Dict[str, Any]:
    return {
        'source': 'libdmet_native_kernel',
        'criteria': 'energy_and_mean_field_rdm1',
        'energy_measure': energy_measure,
        'energy_tolerance': float(configuration['energy_tolerance']),
        'density_measure': 'max_abs_consecutive_mean_field_rdm1',
        'density_tolerance': float(configuration['density_tolerance']),
        'density_fit_measure': 'correlated_to_auxiliary_density_residual',
        'density_fit_tolerance': float(configuration['density_fit_tolerance']),
        'correlation_potential_change_is_diagnostic_only': True,
        'density_fit_error_is_diagnostic_only': True,
    }


def _fit_native_lattice_correlation_potential(
    global_density: Any,
    lattice: Any,
    correlation_potential: Any,
    filling: Any,
    *,
    beta: float,
    slater: Any,
    np: Any,
) -> Tuple[Any, float, float]:
    spin_dimension = int(np.asarray(global_density).shape[0])
    cell_count = int(getattr(lattice, 'ncells', getattr(lattice, 'nkpts', 1)))
    identity_basis = np.zeros(
        (spin_dimension, cell_count, lattice.nscsites, lattice.nscsites),
        dtype=float,
    )
    identity_basis[:, 0] = np.eye(lattice.nscsites)
    fit_lattice = prepare_density_fit_lattice(lattice)
    candidate = copy.deepcopy(correlation_potential)
    fitted, fit_error_begin, fit_error_end = slater.FitVcorFull(
        np.asarray(global_density)[:, 0],
        fit_lattice,
        identity_basis,
        candidate,
        beta,
        filling,
        MaxIter=_LIBDMET_NATIVE_FIT_MAX_ITERATIONS,
        imp_fit=True,
        method='BFGS',
        ytol=1.0e-7,
        gtol=1.0e-5,
        test_grad=False,
        # Native libDMET has no analytic zero-temperature fitting gradient.
        num_grad=math.isinf(beta),
    )
    return fitted, float(fit_error_begin), float(fit_error_end)


def _uses_shared_smearing(configuration: Mapping[str, Any]) -> bool:
    return configuration['impurity_solver'] == 'ccsd' or (
        configuration['impurity_solver'] == 'fci'
        and 'beta' in configuration.get('impurity_solver_options', {})
    )


def _lattice_scf_beta(configuration: Mapping[str, Any]) -> float:
    # FCI opts in explicitly; an empty FCI options mapping preserves its
    # legacy zero-temperature lattice and finite-beta fitting policies.
    if _uses_shared_smearing(configuration):
        if not configuration['impurity_solver_options'].get('smearing', True):
            return math.inf
        return float(configuration['impurity_solver_options']['beta'])
    return math.inf


def _density_fit_beta(configuration: Mapping[str, Any]) -> float:
    if _uses_shared_smearing(configuration):
        return _lattice_scf_beta(configuration)
    return _LIBDMET_NATIVE_FIT_BETA


def _ccsd_smearing_metadata(configuration: Mapping[str, Any]) -> Dict[str, Any]:
    if not _uses_shared_smearing(configuration):
        return {}
    beta = _lattice_scf_beta(configuration)
    reported_beta = beta if math.isfinite(beta) else None
    return {'smearing': {
        'beta': reported_beta,
        'sigma': 1.0 / beta,
        'method': 'fermi' if math.isfinite(beta) else 'none',
        'energy_unit': 'model_energy_unit',
        'lattice_scf_beta': reported_beta,
        'density_fit_beta': reported_beta,
        'impurity_scf_beta': reported_beta,
        'correlated_solver_temperature': 'ground_state',
    }}


def _native_fitting_contract(beta: float) -> Dict[str, Any]:
    return {
        'space': 'full_lattice',
        'target': 'impurity_block_of_assembled_correlated_rdm1',
        'optimizer': 'BFGS',
        'fit_beta': beta if math.isfinite(beta) else None,
        'gradient': 'analytic' if math.isfinite(beta) else 'numerical',
        'max_iterations': _LIBDMET_NATIVE_FIT_MAX_ITERATIONS,
        'ytol': 1.0e-7,
        'gtol': 1.0e-5,
    }


def _fragment_fitting_contract(beta: float, *, translation_tied: bool) -> Dict[str, Any]:
    return {
        'space': 'finite_graph_fragment_blocks',
        'target': 'fragment_local_blocks_of_assembled_correlated_rdm1',
        'parameterization': (
            'translation_tied_fragment_blocks'
            if translation_tied
            else 'independent_fragment_blocks'
        ),
        'residual_aggregation': 'maximum_fragment_block_residual',
        'optimizer': 'BFGS',
        'fit_beta': beta if math.isfinite(beta) else None,
        'gradient': 'analytic' if math.isfinite(beta) else 'numerical',
        'max_iterations': _LIBDMET_NATIVE_FIT_MAX_ITERATIONS,
        'ytol': 1.0e-7,
        'gtol': 1.0e-5,
    }


def _uniform(values: Sequence[float], field_name: str, tolerance: float = 1.0e-10) -> float:
    if not values:
        raise ValueError('DMET requires at least one {0} value'.format(field_name))
    reference = float(values[0])
    if any(abs(float(value) - reference) > tolerance for value in values[1:]):
        raise ValueError('The current DMET adapter requires uniform {0}'.format(field_name))
    return reference


def _square_shape(spec: Mapping[str, Any]) -> Tuple[int, int]:
    sites = list(spec.get('sites') or [])
    xs = sorted({round(float(site.get('x', 0.0)), 10) for site in sites})
    ys = sorted({round(float(site.get('y', 0.0)), 10) for site in sites})
    if len(xs) * len(ys) != len(sites):
        raise ValueError('The current 2D DMET adapter requires a complete rectangular square lattice')
    coordinates = {
        (round(float(site.get('x', 0.0)), 10), round(float(site.get('y', 0.0)), 10))
        for site in sites
    }
    if coordinates != {(x, y) for x in xs for y in ys}:
        raise ValueError('The current 2D DMET adapter requires one site at every square-lattice coordinate')
    return len(xs), len(ys)


def _default_impurity_shape(lattice_shape: Sequence[int]) -> List[int]:
    # Without Builder primitive-cell metadata there is no physical basis for
    # inferring a larger fragment from the parity of the finite lattice size.
    # Use one lattice cell per axis and require larger clusters explicitly.
    return [1 for _ in lattice_shape]


def _primitive_cell_groups(
    layout: Mapping[str, Any],
    impurity_shape: Sequence[int],
) -> List[List[int]]:
    axes = list(layout['axes'])
    shape = list(impurity_shape) or [1] * int(layout['dimension'])
    groups: List[List[int]] = []
    offsets = [range(0, len(axis), int(length)) for axis, length in zip(axes, shape)]
    for block_offset in itertools.product(*offsets):
        selected_axes = [
            axis[offset:offset + int(length)]
            for axis, offset, length in zip(axes, block_offset, shape)
        ]
        group: List[int] = []
        for cell_index in itertools.product(*selected_axes):
            group.extend(layout['cells'][tuple(cell_index)])
        groups.append(group)
    return groups


def _resolved_fragments(
    sites: Sequence[Mapping[str, Any]],
    options: Mapping[str, Any],
    *,
    execution_mode: str,
    impurity_shape: Sequence[int],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    ordered_site_ids = sorted(int(site.get('id')) for site in sites)
    site_to_orbital = {
        site_id: orbital_index
        for orbital_index, site_id in enumerate(ordered_site_ids)
    }
    fragments = copy.deepcopy(list(options.get('fragments') or []))
    impurity_site_ids = list(options.get('impurity_site_ids') or [])
    impurity_size = options.get('impurity_size')
    primitive_layout, layout_errors = _primitive_cell_layout(sites)
    if layout_errors:
        return [], layout_errors
    selectors = sum(bool(value) for value in (fragments, impurity_site_ids, impurity_size))
    if selectors > 1:
        return [], [
            'Choose only one of fragments, impurity_site_ids, or impurity_size; '
            'impurity_shape is a separate regular-lattice convenience'
        ]

    def automatic_groups(fragment_size: int) -> List[List[int]]:
        if primitive_layout is not None:
            if impurity_size and int(primitive_layout['basis_size']) > 1:
                raise ValueError(
                    'impurity_size counts sites and is ambiguous for a multi-site primitive cell; '
                    'use the primitive-cell definition, impurity_shape in primitive-cell units, '
                    'or provide explicit fragments'
                )
            if not impurity_size:
                return _primitive_cell_groups(primitive_layout, impurity_shape)
        if impurity_shape and len(impurity_shape) == 1:
            spatial_order = [
                int(site['id'])
                for site in sorted(sites, key=lambda item: (float(item.get('x', 0.0)), int(item['id'])))
            ]
            return [
                spatial_order[offset:offset + fragment_size]
                for offset in range(0, len(spatial_order), fragment_size)
            ]
        if impurity_shape and len(impurity_shape) == 2:
            xs = sorted({round(float(site.get('x', 0.0)), 10) for site in sites})
            ys = sorted({round(float(site.get('y', 0.0)), 10) for site in sites})
            coordinate_to_site = {
                (
                    round(float(site.get('x', 0.0)), 10),
                    round(float(site.get('y', 0.0)), 10),
                ): int(site['id'])
                for site in sites
            }
            if len(coordinate_to_site) == len(sites) and len(xs) * len(ys) == len(sites):
                groups = []
                x_size, y_size = (int(value) for value in impurity_shape)
                for y_offset in range(0, len(ys), y_size):
                    for x_offset in range(0, len(xs), x_size):
                        group = [
                            coordinate_to_site[(x, y)]
                            for y in ys[y_offset:y_offset + y_size]
                            for x in xs[x_offset:x_offset + x_size]
                        ]
                        groups.append(group)
                return groups
        return [
            ordered_site_ids[offset:offset + fragment_size]
            for offset in range(0, len(ordered_site_ids), fragment_size)
        ]

    if fragments:
        raw_fragments = fragments
    elif impurity_site_ids:
        raw_fragments = [{
            'fragment_id': 'fragment-1',
            'label': 'Selected impurity',
            'site_ids': impurity_site_ids,
            'orbital_indices': [],
            'metadata': {'selection': 'impurity_site_ids'},
        }]
    else:
        fragment_size = int(impurity_size or math.prod(impurity_shape or [1]))
        try:
            groups = automatic_groups(fragment_size)
        except ValueError as exc:
            return [], [str(exc)]
        if execution_mode == 'translational':
            groups = groups[:1]
        selection = (
            'impurity_size'
            if impurity_size
            else ('primitive_cell_shape' if primitive_layout is not None else 'impurity_shape')
        )
        raw_fragments = [
            {
                'fragment_id': 'fragment-{0}'.format(position + 1),
                'label': 'Fragment {0}'.format(position + 1),
                'site_ids': group,
                'orbital_indices': [],
                'metadata': {
                    'selection': selection,
                    'cell_shape': list(impurity_shape) if impurity_shape else None,
                },
            }
            for position, group in enumerate(groups)
        ]

    resolved: List[Dict[str, Any]] = []
    seen_site_ids = set()
    for position, fragment in enumerate(raw_fragments):
        site_ids = list(fragment.get('site_ids') or [])
        orbital_indices = list(fragment.get('orbital_indices') or [])
        if site_ids and orbital_indices:
            errors.append('Fragment {0} cannot define both site_ids and orbital_indices'.format(position + 1))
            continue
        if orbital_indices:
            if any(index >= len(ordered_site_ids) for index in orbital_indices):
                errors.append('Fragment orbital_indices exceed the number of model sites')
                continue
            site_ids = [ordered_site_ids[index] for index in orbital_indices]
        unknown = [site_id for site_id in site_ids if site_id not in site_to_orbital]
        if unknown:
            errors.append('Fragment site_ids are not present in the ModelSpec: {0}'.format(unknown))
            continue
        overlap = [site_id for site_id in site_ids if site_id in seen_site_ids]
        if overlap:
            errors.append('DMET fragments must not overlap; repeated site_ids: {0}'.format(overlap))
            continue
        seen_site_ids.update(site_ids)
        resolved.append({
            'fragment_id': str(fragment.get('fragment_id') or 'fragment-{0}'.format(position + 1)),
            'label': str(fragment.get('label') or 'Fragment {0}'.format(position + 1)),
            'site_ids': site_ids,
            'orbital_indices': [site_to_orbital[site_id] for site_id in site_ids],
            'size': len(site_ids),
            'metadata': copy.deepcopy(fragment.get('metadata') or {}),
        })

    if any(not fragment['site_ids'] for fragment in resolved):
        errors.append('DMET fragments cannot be empty')
    if any(len(fragment['site_ids']) >= len(ordered_site_ids) for fragment in resolved):
        errors.append('Each DMET fragment must leave at least one environment site for bath construction')
    if execution_mode == 'finite_graph' and seen_site_ids != set(ordered_site_ids):
        missing = sorted(set(ordered_site_ids) - seen_site_ids)
        errors.append(
            'Finite-graph DMET requires non-overlapping fragments that cover every model site; '
            'missing site_ids: {0}'.format(missing)
        )
    return resolved, errors


def _is_uniform(values: Sequence[float], tolerance: float = 1.0e-10) -> bool:
    return bool(values) and all(abs(float(value) - float(values[0])) <= tolerance for value in values[1:])


def _validate_regular_topology(
    sites: Sequence[Mapping[str, Any]],
    bonds: Sequence[Mapping[str, Any]],
    *,
    dimension: int,
    preset: str,
) -> List[str]:
    """Reject graphs that the native Chain/Square constructors would replace."""

    errors: List[str] = []
    supported_preset = (
        dimension == 1 and preset in ('chain', 'ring')
    ) or (
        dimension == 2 and preset == 'square'
    )
    if not supported_preset:
        return [
            'Native translated DMET is registered only for one-dimensional '
            'chain/ring and two-dimensional square presets'
        ]
    site_ids = set()
    for site in sites:
        try:
            site_id = int(site.get('id'))
        except (TypeError, ValueError):
            errors.append('Every DMET site must have an integer id')
            continue
        if site_id in site_ids:
            errors.append('DMET site ids must be unique')
        site_ids.add(site_id)
    if errors:
        return errors
    degrees = {site_id: 0 for site_id in site_ids}
    for bond in bonds:
        try:
            source = int(bond.get('source'))
            target = int(bond.get('target'))
        except (TypeError, ValueError):
            errors.append('Every DMET bond must reference integer site ids')
            continue
        if source not in site_ids or target not in site_ids:
            errors.append('Every DMET bond endpoint must reference an existing site')
            continue
        degrees[source] += 1
        degrees[target] += 1

    expected_degree = 2 if dimension == 1 else 4
    expected_bond_count = len(sites) if dimension == 1 else 2 * len(sites)
    if len(bonds) != expected_bond_count or any(
        degree != expected_degree for degree in degrees.values()
    ):
        lattice_name = 'ring' if dimension == 1 else 'periodic square lattice'
        errors.append(
            'The current DMET adapter requires the complete nearest-neighbor {0} topology '
            '({1} bonds and degree {2} at every site)'.format(
                lattice_name,
                expected_bond_count,
                expected_degree,
            )
        )
        return errors

    use_cell_indices = all(
        isinstance(site.get('cell_index'), (list, tuple))
        and len(site.get('cell_index')) >= dimension
        for site in sites
    )
    coordinate_to_site: Dict[Tuple[Any, ...], int] = {}
    for site in sites:
        if use_cell_indices:
            coordinate = tuple(int(value) for value in site['cell_index'][:dimension])
        else:
            coordinate = tuple(
                round(float(site.get(axis, 0.0)), 10)
                for axis in ('x', 'y')[:dimension]
            )
        if coordinate in coordinate_to_site:
            errors.append(
                'The translated lattice contract requires one site at every lattice coordinate'
            )
            return errors
        coordinate_to_site[coordinate] = int(site['id'])

    axes = [
        sorted({coordinate[index] for coordinate in coordinate_to_site})
        for index in range(dimension)
    ]
    expected_coordinates = set(itertools.product(*axes))
    if set(coordinate_to_site) != expected_coordinates:
        errors.append(
            'The translated lattice coordinates must form a complete rectangular grid'
        )
        return errors

    expected_edges: Counter = Counter()
    for coordinate, source in coordinate_to_site.items():
        for axis_index in range(dimension):
            axis = axes[axis_index]
            axis_position = axis.index(coordinate[axis_index])
            target_coordinate = list(coordinate)
            target_coordinate[axis_index] = axis[(axis_position + 1) % len(axis)]
            target = coordinate_to_site[tuple(target_coordinate)]
            expected_edges[tuple(sorted((source, target)))] += 1
    actual_edges = Counter(
        tuple(sorted((int(bond['source']), int(bond['target']))))
        for bond in bonds
    )
    if actual_edges != expected_edges:
        errors.append(
            'The model bonds do not match the periodic nearest-neighbor '
            '{0} graph implied by the Builder coordinates'.format(
                'ring' if dimension == 1 else 'square lattice'
            )
        )
    return errors


def validate_dmet_model_request(
    spec: Mapping[str, Any],
    options: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    errors: List[str] = []
    try:
        normalized_options = normalize_dmet_options(options)
    except ValueError as exc:
        return [str(exc)], None

    if str(spec.get('model') or 'hubbard').strip().lower() != 'hubbard':
        errors.append('The libDMET adapter currently supports the single-band Hubbard model only')
    if str(spec.get('representation') or 'finite_cluster').strip().lower() != 'finite_cluster':
        errors.append('DMET requires representation=finite_cluster; Bloch one-body inputs are not supported')
    sites = list(spec.get('sites') or [])
    bonds = list(spec.get('bonds') or [])
    if not sites:
        errors.append('DMET requires at least one model site')
        return errors, None
    dimension = int(spec.get('dimension') or 1)
    preset = str(spec.get('preset') or '').strip().lower().replace('-', '_')
    primitive_layout, primitive_layout_errors = _primitive_cell_layout(sites)
    errors.extend(primitive_layout_errors)
    builder_translation_audit = audit_builder_translation(spec)
    if primitive_layout is not None:
        lattice_shape = list(primitive_layout['lattice_shape'])
    elif dimension == 1:
        lattice_shape = [len(sites)]
    elif dimension == 2:
        if preset == 'square':
            try:
                lattice_shape = list(_square_shape(spec))
            except ValueError:
                lattice_shape = []
        else:
            lattice_shape = []
    else:
        errors.append('The current libDMET adapter supports one- or two-dimensional finite graphs')
        lattice_shape = []

    nelec = spec.get('nelec')
    if not isinstance(nelec, (list, tuple)) or len(nelec) != 2:
        errors.append('DMET requires nelec=[nalpha, nbeta]')
        nalpha = nbeta = 0
    else:
        try:
            nalpha, nbeta = (int(value) for value in nelec)
        except (TypeError, ValueError):
            errors.append('DMET requires integer values in nelec=[nalpha, nbeta]')
            nalpha = nbeta = 0
    if nalpha < 0 or nbeta < 0 or nalpha + nbeta > 2 * len(sites):
        errors.append('DMET electron counts must fit the available spin orbitals')
    spin_difference = nalpha - nbeta

    onsite_values = [float(site.get('U', 0.0)) for site in sites]
    epsilon_values = [float(site.get('epsilon', 0.0)) for site in sites]
    hopping_values = [
        float(bond.get('effective_t', bond.get('t', 0.0)))
        for bond in bonds
    ]
    intersite_values = [
        float(bond.get('effective_V', bond.get('V', 0.0)))
        for bond in bonds
    ]
    topology_errors = (
        _validate_regular_topology(
            sites,
            bonds,
            dimension=dimension,
            preset=preset,
        )
        if dimension in (1, 2) and lattice_shape
        else ['not a regular translated lattice']
    )
    native_translated_topology = (
        not topology_errors
        and (
            primitive_layout is None
            or int(primitive_layout['basis_size']) == 1
        )
    )
    builder_translated_topology = bool(builder_translation_audit['eligible'])
    translation_backend_candidate = (
        'native_lattice'
        if native_translated_topology
        else ('builder_primitive_cell' if builder_translated_topology else None)
    )
    builder_translation_checks = builder_translation_audit.get('checks') or {}
    uses_builder_template = translation_backend_candidate == 'builder_primitive_cell'
    onsite_terms_are_translated = bool(
        builder_translation_checks.get('translation_equivalent_onsite_terms')
    )
    bond_terms_are_translated = bool(
        builder_translation_checks.get('translation_equivalent_bonds')
    )
    translational_checks = {
        'periodic_boundary': (
            str(spec.get('boundary') or '').strip().lower() == 'periodic'
        ),
        'translation_equivalent_graph': translation_backend_candidate is not None,
        # For Builder lattices, "uniform" means equal under primitive-cell
        # translations. Different basis sites inside one cell may carry
        # different onsite terms or bond parameters.
        'uniform_onsite_u': (
            onsite_terms_are_translated
            if uses_builder_template
            else _is_uniform(onsite_values)
        ),
        'uniform_onsite_energy': (
            onsite_terms_are_translated
            if uses_builder_template
            else _is_uniform(epsilon_values)
        ),
        'uniform_hopping': (
            bond_terms_are_translated
            if uses_builder_template
            else _is_uniform(hopping_values)
        ),
        'no_intersite_interaction': all(
            abs(value) <= 1.0e-10 for value in intersite_values
        ),
        'automatic_fragment_selector': not any((
            normalized_options['fragments'],
            normalized_options['impurity_site_ids'],
            normalized_options['impurity_size'],
            normalized_options['fragment_definition'] == 'honeycomb_hexagon',
        )),
    }
    translated_eligible = all(translational_checks.values())
    failed_translational_checks = [
        name for name, passed in translational_checks.items() if not passed
    ]
    requested_execution_mode = normalized_options['execution_mode']
    if requested_execution_mode == 'translational':
        execution_mode = 'translational'
        if not translated_eligible:
            errors.append(
                'Translated representative DMET was requested, but the model does not satisfy: {0}. '
                'Choose execution_mode=finite_graph to preserve the complete Builder graph.'.format(
                    ', '.join(failed_translational_checks)
                )
            )
    elif requested_execution_mode == 'finite_graph':
        execution_mode = 'finite_graph'
    else:
        execution_mode = 'translational' if translated_eligible else 'finite_graph'
    translation_backend = (
        translation_backend_candidate
        if execution_mode == 'translational' and translated_eligible
        else None
    )
    execution_mode_reason = (
        (
            'The model satisfies the registered translated chain/ring or '
            'square-lattice contract.'
            if translation_backend == 'native_lattice'
            else (
                'Builder primitive-cell metadata and the Hamiltonian graph pass '
                'the translation-equivalence audit; one representative impurity '
                'is reused across equivalent cells.'
            )
        )
        if execution_mode == 'translational' and translated_eligible
        else (
            'The complete Builder Hamiltonian is preserved as an explicit finite-graph partition.'
            if requested_execution_mode == 'finite_graph'
            else (
                'The original Builder Hamiltonian is preserved because translated '
                'lattice assumptions failed: {0}.'.format(
                    ', '.join(failed_translational_checks)
                )
            )
        )
    )

    fragment_definition = normalized_options['fragment_definition']
    impurity_shape = list(normalized_options['impurity_shape'])
    fragment_options = normalized_options
    if fragment_definition == 'honeycomb_hexagon':
        if execution_mode == 'translational':
            errors.append('honeycomb_hexagon requires execution_mode=finite_graph or automatic mode')
        if any((normalized_options['fragments'], normalized_options['impurity_site_ids'],
                normalized_options['impurity_size'], impurity_shape)):
            errors.append('honeycomb_hexagon cannot be combined with another fragment selector')
        try:
            hexagons = honeycomb_hexagonal_fragments(spec)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            fragment_options = {**normalized_options, 'fragments': hexagons}
        execution_mode_reason = (
            'Elementary honeycomb hexagons form a complete disjoint six-site partition; '
            'finite-graph DMET preserves all original sites, bonds and interactions.'
        )
    elif fragment_definition == 'primitive_cell':
        if any((
            normalized_options['fragments'],
            normalized_options['impurity_site_ids'],
            normalized_options['impurity_size'],
            impurity_shape,
        )):
            errors.append(
                'fragment_definition=primitive_cell cannot be combined with explicit fragment selectors'
            )
        impurity_shape = [1] * dimension
    elif fragment_definition == 'cell_shape' and not impurity_shape:
        errors.append('fragment_definition=cell_shape requires impurity_shape')
    elif fragment_definition == 'site_count':
        if execution_mode == 'translational':
            errors.append(
                'fragment_definition=site_count is available only with execution_mode=finite_graph'
            )
        if normalized_options['impurity_size'] is None:
            errors.append('fragment_definition=site_count requires impurity_size')
    selector_count = sum(bool(value) for value in (
        normalized_options['fragments'],
        normalized_options['impurity_site_ids'],
        normalized_options['impurity_size'],
        impurity_shape,
    ))
    if selector_count > 1:
        errors.append(
            'Choose one fragment selector: fragments, impurity_site_ids, '
            'impurity_size, or impurity_shape'
        )
    if not impurity_shape and execution_mode == 'translational':
        impurity_shape = (
            [1] * dimension
            if translation_backend == 'builder_primitive_cell'
            else _default_impurity_shape(lattice_shape)
        )
    if impurity_shape and lattice_shape and len(impurity_shape) != len(lattice_shape):
        errors.append('impurity_shape must contain one value per lattice dimension')
    elif impurity_shape and lattice_shape:
        for lattice_length, impurity_length in zip(lattice_shape, impurity_shape):
            if impurity_length > lattice_length or (
                execution_mode == 'translational' and lattice_length % impurity_length != 0
            ):
                errors.append(
                    'Each impurity_shape value must fit the corresponding lattice length; '
                    'translated fragments must tile it exactly'
                )

    fragments, fragment_errors = _resolved_fragments(
        sites,
        fragment_options,
        execution_mode=execution_mode,
        impurity_shape=impurity_shape,
    )
    errors.extend(fragment_errors)
    resolved_reference = normalized_options['reference']
    if spin_difference and resolved_reference == 'restricted':
        errors.append(
            'Restricted DMET requires nalpha=nbeta; use reference=unrestricted '
            'for a nonzero spin projection'
        )
    if spin_difference and execution_mode != 'translational':
        errors.append(
            'Nonzero-spin DMET currently requires one translated representative '
            'fragment; fragment-specific spin sectors for finite-graph partitions '
            'are not implemented'
        )
    if spin_difference and fragments:
        impurity_electron_count = 2 * len(fragments[0]['site_ids'])
        if abs(spin_difference) > impurity_electron_count:
            errors.append(
                'The requested libDMET Sz={0} exceeds the translated impurity '
                'electron count {1}; increase the impurity size'.format(
                    spin_difference,
                    impurity_electron_count,
                )
            )
        elif (impurity_electron_count + spin_difference) % 2:
            errors.append(
                'The requested libDMET Sz={0} is parity-incompatible with the '
                'translated impurity electron count {1}; choose a compatible '
                'impurity size'.format(spin_difference, impurity_electron_count)
            )
    if normalized_options['bath_spin_dimension_policy'] == 'max' and (
        execution_mode != 'finite_graph' or normalized_options['interacting_bath']
    ):
        errors.append('Max spin bath requires finite_graph execution and interacting_bath=false')
    if normalized_options['reference_density_source'] and execution_mode != 'finite_graph':
        errors.append('DMET density warm starts require finite_graph execution')
    errors.extend(validate_reference_density_guess(spec, {
        **normalized_options,
        'execution_mode': execution_mode,
        'requested_execution_mode': requested_execution_mode,
        'fragments': fragments,
        'reference': resolved_reference,
        'nalpha': nalpha,
        'nbeta': nbeta,
    }))
    if errors:
        return errors, None
    average_filling_per_spin = float(nalpha + nbeta) / float(2 * len(sites))
    spin_filling = [
        float(nalpha) / float(len(sites)),
        float(nbeta) / float(len(sites)),
    ]
    mean_field_filling: Any = (
        average_filling_per_spin
        if resolved_reference == 'restricted'
        else spin_filling
    )
    configuration = {
        **normalized_options,
        'lattice_shape': lattice_shape,
        'impurity_shape': impurity_shape,
        'fragment_definition': fragment_definition,
        'dimension': dimension,
        'site_count': len(sites),
        'average_filling_per_spin': average_filling_per_spin,
        'spin_filling': spin_filling,
        'mean_field_filling': mean_field_filling,
        'nalpha': nalpha,
        'nbeta': nbeta,
        # libDMET calls Nalpha-Nbeta "Sz". Physical S_z is half this value.
        'libdmet_sz': spin_difference,
        'physical_sz': 0.5 * spin_difference,
        'execution_mode': execution_mode,
        'requested_execution_mode': requested_execution_mode,
        'translation_backend': translation_backend,
        'execution_mode_reason': execution_mode_reason,
        'translation_symmetry': {
            'eligible': translated_eligible,
            'checks': translational_checks,
            'failed_checks': failed_translational_checks,
            'topology_issues': (
                []
                if translation_backend_candidate == 'native_lattice'
                else (
                    list(builder_translation_audit['issues'])
                    if primitive_layout is not None
                    else topology_errors
                )
            ),
            'builder_audit': copy.deepcopy(builder_translation_audit),
        },
        'fragments': fragments,
        'fragment_count': len(fragments),
        'fragment_site_counts': [len(fragment['site_ids']) for fragment in fragments],
        'primitive_cell_basis_size': (
            int(primitive_layout['basis_size']) if primitive_layout is not None else None
        ),
        'onsite_u': float(onsite_values[0]) if _is_uniform(onsite_values) else None,
        'hopping_in_input_convention': (
            float(hopping_values[0]) if _is_uniform(hopping_values) else None
        ),
        'libdmet_hopping_parameter': (
            float(-hopping_values[0]) if _is_uniform(hopping_values) else None
        ),
        'contains_intersite_v': any(abs(value) > 1.0e-10 for value in intersite_values),
        'interacting_bath': normalized_options['interacting_bath'],
        'reference': resolved_reference,
    }
    return [], configuration


def _adapt_unrestricted_solver_for_pyscf(solver: Any, configuration: Mapping[str, Any]) -> Any:
    if configuration['reference'] != 'unrestricted':
        return solver
    original_run = solver.run

    def compatible_run(*args: Any, **kwargs: Any) -> Any:
        from libdmet.solver import scf as libdmet_scf  # pylint: disable=import-outside-toplevel

        # The compatibility shim changes a provider class method. Keep the
        # complete solver call inside the lock so parallel local cases cannot
        # observe or restore another case's temporary method.
        with _LIBDMET_UIHF_COMPATIBILITY_LOCK:
            ui_hf_class = libdmet_scf.UIHF
            original_eig = ui_hf_class.eig
            if 'x' in inspect.signature(original_eig).parameters:
                return original_run(*args, **kwargs)

            def compatible_eig(instance: Any, fock: Any, overlap: Any, x: Any = None) -> Any:
                del x
                return original_eig(instance, fock, overlap)

            ui_hf_class.eig = compatible_eig
            try:
                return original_run(*args, **kwargs)
            finally:
                ui_hf_class.eig = original_eig

    solver.run = compatible_run
    return solver


def _solver(
    dmet: Any,
    configuration: Mapping[str, Any],
    scratch_directory: Optional[str],
    *,
    solver_id: str = 'impurity-1',
) -> Any:
    common = {
        'restricted': configuration['reference'] == 'restricted',
        'tol': configuration['solver_tolerance'],
        'max_cycle': configuration['solver_max_cycle'],
        'max_memory': configuration['solver_max_memory_mb'],
        # libDMET exposes a spin-resolved overlap for UIHF. PySCF's Newton
        # wrapper expects one square overlap matrix, while the standard UIHF
        # kernel accepts the spin-resolved representation directly.
        'scf_newton': configuration['reference'] == 'restricted',
        'Sz': int(configuration['libdmet_sz']),
    }
    if scratch_directory:
        Path(scratch_directory).mkdir(parents=True, exist_ok=True)
        common['TmpDir'] = str(scratch_directory)
    if configuration['impurity_solver'] == 'fci':
        # Newton orbital rotations assume integer occupations in PySCF;
        # use the ordinary SCF kernel for explicitly smeared FCI references.
        if math.isfinite(_lattice_scf_beta(configuration)):
            common['scf_newton'] = False
        solver = dmet.impurity_solver.FCI(**common, beta=_lattice_scf_beta(configuration))
    elif configuration['impurity_solver'] == 'ccsd':
        common['scf_newton'] = False
        solver_options = dict(configuration['impurity_solver_options'])
        solver_options.pop('smearing', None)
        solver_options['beta'] = _lattice_scf_beta(configuration)
        solver = dmet.impurity_solver.CCSD(**common, **solver_options)
        solver = configure_ccsd_impurity_scf(
            solver, use_diis=configuration['impurity_scf_diis'],
        )
    else:
        solver = Block2DmetImpuritySolver(
            restricted=configuration['reference'] == 'restricted',
            spin=int(configuration['libdmet_sz']),
            options=configuration.get('impurity_solver_options') or {},
            scratch_directory=scratch_directory,
            solver_id=solver_id,
        )
        return solver
    if configuration['impurity_solver'] == 'fci' and not configuration['impurity_scf_diis']:
        # Configure this solver instance, including every chemical-potential
        # fitting call. Do not change the lattice or outer-loop DIIS settings.
        solver.scfsolver.HF = partial(solver.scfsolver.HF, do_diis=False)
    return _adapt_unrestricted_solver_for_pyscf(solver, configuration)


def _impurity_solver_details(solvers: Sequence[Any]) -> List[Dict[str, Any]]:
    details = []
    for position, solver in enumerate(solvers):
        summary = getattr(solver, 'summary', None)
        if not callable(summary):
            continue
        payload = summary()
        payload['impurity_index'] = position
        details.append(payload)
    return details


def _dmet_quality_checks(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Expose numerical observations for the shared execution-quality gate."""

    history = (
        result.get('iteration_history')
        if isinstance(result.get('iteration_history'), Mapping)
        else {}
    )
    contract = (
        history.get('convergence_contract')
        if isinstance(history.get('convergence_contract'), Mapping)
        else {}
    )
    records = history.get('records') if isinstance(history.get('records'), list) else []
    final = records[-1] if records and isinstance(records[-1], Mapping) else {}
    energy = result.get('energy')
    energy_per_site = result.get('energy_per_site')
    site_count = result.get('site_count')
    normalization_residual = None
    normalization_limit = _RESULT_IDENTITY_RELATIVE_TOLERANCE
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (
        energy,
        energy_per_site,
        site_count,
    )):
        normalization_residual = float(energy) - float(energy_per_site) * int(site_count)
        normalization_limit *= max(1.0, abs(float(energy)))
    checks: List[Dict[str, Any]] = [
        {
            'id': 'dmet_energy_change',
            'category': 'convergence',
            'observed': final.get('energy_change'),
            'operator': 'absolute_less_than',
            'limit': contract.get('energy_tolerance'),
            'required': True,
            'source': 'iteration_history.records[-1].energy_change',
        },
        {
            'id': 'dmet_mean_field_rdm1_change',
            'category': 'convergence',
            'observed': final.get('mean_field_rdm1_change_max_abs'),
            'operator': 'less_than',
            'limit': contract.get('density_tolerance'),
            'required': True,
            'source': 'iteration_history.records[-1].mean_field_rdm1_change_max_abs',
        },
        {
            'id': 'dmet_density_fit_error',
            'category': 'convergence',
            'observed': final.get('density_fit_error'),
            'operator': 'less_than',
            'limit': contract.get('density_fit_tolerance'),
            # Reported diagnostic; the native stopping rule does not use it.
            'required': not contract.get('density_fit_error_is_diagnostic_only', False),
            'source': 'iteration_history.records[-1].density_fit_error',
        },
        {
            'id': 'dmet_total_energy_finite',
            'category': 'constraint',
            'observed': energy,
            'operator': 'finite',
            'required': True,
            'source': 'energy',
        },
        {
            'id': 'dmet_energy_normalization',
            'category': 'constraint',
            'observed': normalization_residual,
            'operator': 'absolute_less_than',
            'limit': normalization_limit,
            'required': True,
            'source': 'energy-energy_per_site*site_count',
        },
    ]
    fragments = result.get('fragments') if isinstance(result.get('fragments'), list) else []
    capacity_margins: List[int] = []
    sector_consistency: List[bool] = []
    selection_consistency: List[bool] = []
    for fragment in fragments:
        if not isinstance(fragment, Mapping):
            sector_consistency.append(False)
            selection_consistency.append(False)
            continue
        sector = (
            fragment.get('impurity_spin_sector')
            if isinstance(fragment.get('impurity_spin_sector'), Mapping)
            else {}
        )
        selection = (
            sector.get('electron_count_selection')
            if isinstance(sector.get('electron_count_selection'), Mapping)
            else {}
        )
        orbital_count = fragment.get('embedding_orbital_count')
        nelec = sector.get('nelec')
        nalpha = sector.get('nalpha')
        nbeta = sector.get('nbeta')
        capacity_margin = None
        if all(isinstance(value, int) and not isinstance(value, bool) for value in (
            orbital_count,
            nelec,
            nalpha,
            nbeta,
        )):
            capacity_margin = min(
                nelec,
                2 * orbital_count - nelec,
                nalpha,
                nbeta,
                orbital_count - nalpha,
                orbital_count - nbeta,
            )
            capacity_margins.append(capacity_margin)
        sector_consistent = bool(
            isinstance(nelec, int)
            and isinstance(nalpha, int)
            and isinstance(nbeta, int)
            and nelec == nalpha + nbeta
            and isinstance(sector.get('libdmet_sz'), int)
            and nalpha - nbeta == sector.get('libdmet_sz')
        )
        selected_count = selection.get('selected_electron_count')
        selection_kind = str(selection.get('selection') or '')
        selected_consistent = bool(
            isinstance(nelec, int)
            and selected_count == nelec
            and selection.get('embedding_spatial_orbital_count') == orbital_count
        )
        if selection_kind == 'projected_mean_field_density_trace':
            rounding_error = selection.get('rounding_error')
            rounding_tolerance = selection.get('rounding_tolerance')
            selected_consistent = bool(
                selected_consistent
                and isinstance(rounding_error, (int, float))
                and isinstance(rounding_tolerance, (int, float))
                and float(rounding_error) <= float(rounding_tolerance)
            )
        elif selection_kind == 'valence_sector_for_fractional_mean_field':
            selected_consistent = bool(
                selected_consistent
                and selection.get('fractional_sector_electron_count') == nelec
                and int(selection.get('fractional_occupation_count') or 0) > 0
            )
        else:
            selected_consistent = False
        sector_consistency.append(sector_consistent)
        selection_consistency.append(selected_consistent)
    checks.extend((
        {
            'id': 'dmet_impurity_capacity',
            'category': 'constraint',
            'observed': (
                min(capacity_margins)
                if fragments and len(capacity_margins) == len(fragments)
                else None
            ),
            'operator': 'greater_than_or_equal',
            'limit': 0,
            'required': True,
            'source': 'fragments[*].impurity_spin_sector',
        },
        {
            'id': 'dmet_impurity_sector',
            'category': 'constraint',
            'observed': bool(fragments and all(sector_consistency)),
            'operator': 'equal',
            'expected': True,
            'required': True,
            'source': 'fragments[*].impurity_spin_sector',
        },
        {
            'id': 'dmet_projected_sector',
            'category': 'constraint',
            'observed': bool(fragments and all(selection_consistency)),
            'operator': 'equal',
            'expected': True,
            'required': True,
            'source': 'fragments[*].impurity_spin_sector.electron_count_selection',
        },
    ))
    solver_details = (
        result.get('impurity_solver_details')
        if isinstance(result.get('impurity_solver_details'), list)
        else []
    )
    if solver_details and all(
        isinstance(detail, Mapping) and 'converged' in detail for detail in solver_details
    ):
        checks.append({
            'id': 'dmet_impurity_solver_convergence',
            'category': 'convergence',
            'observed': all(bool(detail.get('converged')) for detail in solver_details),
            'operator': 'equal',
            'expected': True,
            'required': True,
            'source': 'impurity_solver_details[*].converged',
        })
    return checks


def _serializable_chemical_potential(value: Any, np: Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return [float(item) for item in array.ravel()]


def _impurity_spin_sector(
    electron_count: int,
    orbital_count: int,
    configuration: Mapping[str, Any],
) -> Dict[str, Any]:
    libdmet_sz = int(configuration['libdmet_sz'])
    if (electron_count + libdmet_sz) % 2:
        raise ValueError(
            'Impurity nelec={0} and libDMET Sz={1} have incompatible parity'.format(
                electron_count,
                libdmet_sz,
            )
        )
    nalpha = (electron_count + libdmet_sz) // 2
    nbeta = (electron_count - libdmet_sz) // 2
    if min(nalpha, nbeta) < 0 or max(nalpha, nbeta) > orbital_count:
        raise ValueError(
            'Impurity sector nelec={0}, Sz={1} gives (Nalpha, Nbeta)=({2}, {3}), '
            'which does not fit {4} spatial orbitals'.format(
                electron_count,
                libdmet_sz,
                nalpha,
                nbeta,
                orbital_count,
            )
        )
    return {
        'nelec': int(electron_count),
        'libdmet_sz': libdmet_sz,
        'physical_sz': 0.5 * libdmet_sz,
        'nalpha': int(nalpha),
        'nbeta': int(nbeta),
    }


def _array_component_shapes(value: Any) -> Dict[str, List[int]]:
    if isinstance(value, Mapping):
        return {
            str(key): list(component.shape)
            for key, component in value.items()
            if hasattr(component, 'shape')
        }
    if hasattr(value, 'shape'):
        return {'default': list(value.shape)}
    return {}


def _add_numerical_components(target: Dict[str, Any], prefix: str, value: Any) -> None:
    if isinstance(value, Mapping):
        for key, component in value.items():
            if hasattr(component, 'shape'):
                target['{0}_{1}'.format(prefix, key)] = component
        return
    if hasattr(value, 'shape'):
        target[prefix] = value


def _spin_population_summary(
    density_matrix: Any,
    configuration: Mapping[str, Any],
    np: Any,
) -> Dict[str, Any]:
    """Summarize fragment spin populations without discarding UHF spin resolution."""

    density = np.asarray(density_matrix)
    if density.ndim == 2:
        density = density[np.newaxis, :, :]
    expected_spin_blocks = 1 if configuration['reference'] == 'restricted' else 2
    if density.ndim != 3 or density.shape[0] != expected_spin_blocks:
        raise ValueError(
            'DMET fragment density has shape {0}; expected {1} spin block(s)'.format(
                tuple(density.shape),
                expected_spin_blocks,
            )
        )
    alpha_population = np.real(np.diagonal(density[0])).astype(float)
    beta_population = (
        alpha_population.copy()
        if expected_spin_blocks == 1
        else np.real(np.diagonal(density[1])).astype(float)
    )
    local_magnetization = alpha_population - beta_population
    alpha_count = float(np.sum(alpha_population))
    beta_count = float(np.sum(beta_population))
    return {
        'spin_resolved': expected_spin_blocks == 2,
        'alpha_electron_count': alpha_count,
        'beta_electron_count': beta_count,
        'total_electron_count': alpha_count + beta_count,
        'spin_population_difference': alpha_count - beta_count,
        # This is n_alpha - n_beta. Local <S_z> is one half of this value.
        'local_magnetization': [float(value) for value in local_magnetization],
        'maximum_absolute_local_magnetization': float(
            np.max(np.abs(local_magnetization), initial=0.0)
        ),
    }


def _adapt_unrestricted_overlap_for_pyscf(
    impurity_hamiltonian: Any,
    configuration: Mapping[str, Any],
    np: Any,
    *,
    tolerance: float = 1.0e-8,
) -> None:
    """Collapse libDMET's equivalent spin overlaps for current PySCF UIHF."""

    if configuration['reference'] != 'unrestricted':
        return
    overlap = np.asarray(impurity_hamiltonian.ovlp)
    if overlap.ndim == 2:
        return
    if overlap.ndim != 3 or overlap.shape[0] != 2:
        raise ValueError(
            'Unrestricted DMET requires one common overlap matrix or two spin overlap blocks'
        )
    spin_difference = float(np.max(np.abs(overlap[0] - overlap[1])))
    if spin_difference > tolerance:
        raise ValueError(
            'Unrestricted DMET produced distinct alpha and beta overlap matrices; '
            'the current PySCF UIHF impurity interface requires a common overlap '
            '(max difference={0:.3e})'.format(spin_difference)
        )
    impurity_hamiltonian.ovlp = 0.5 * (overlap[0] + overlap[1])


def _finite_graph_lattice(spec: Mapping[str, Any], np: Any) -> Tuple[Any, Any]:
    from libdmet.system import hamiltonian as libdmet_hamiltonian  # pylint: disable=import-outside-toplevel
    from libdmet.system import lattice as libdmet_lattice  # pylint: disable=import-outside-toplevel

    from ...backend.model_hamiltonian.solver import (  # pylint: disable=import-outside-toplevel
        build_model_hamiltonian_matrices,
    )

    ordered_sites = sorted(spec.get('sites') or [], key=lambda item: int(item['id']))
    dimension = int(spec.get('dimension') or 1)
    raw_coordinates = np.asarray([
        [
            float(site.get('x', orbital_index)),
            *([float(site.get('y', 0.0))] if dimension >= 2 else []),
        ]
        for orbital_index, site in enumerate(ordered_sites)
    ], dtype=float)
    if len({tuple(row) for row in raw_coordinates}) != len(raw_coordinates):
        raw_coordinates = np.zeros((len(ordered_sites), dimension), dtype=float)
        raw_coordinates[:, 0] = np.arange(len(ordered_sites), dtype=float)
    coordinate_minimum = np.min(raw_coordinates, axis=0)
    coordinates = raw_coordinates - coordinate_minimum + 1.0
    spans = np.ptp(coordinates, axis=0)
    cell_size = np.diag(np.maximum(spans + 2.0, 2.0))
    unit_cell = libdmet_lattice.UnitCell(
        cell_size,
        [(coordinates[index], 'H') for index in range(len(ordered_sites))],
    )
    supercell = libdmet_lattice.SuperCell(unit_cell, [1] * dimension)
    lattice = libdmet_lattice.LatticeModel(supercell, [1] * dimension)
    h1e, eri, _metadata = build_model_hamiltonian_matrices(dict(spec))
    hamiltonian = libdmet_hamiltonian.HamNonInt(
        lattice,
        h1e[np.newaxis, :, :],
        eri,
    )
    lattice.setHam(hamiltonian, use_hcore_as_emb_ham=True)
    return lattice, hamiltonian


def _translated_builder_lattice(
    spec: Mapping[str, Any],
    configuration: Mapping[str, Any],
    np: Any,
) -> Tuple[Any, Any]:
    """Build libDMET translation blocks from an audited Builder primitive cell."""

    from libdmet.system import hamiltonian as libdmet_hamiltonian  # pylint: disable=import-outside-toplevel
    from libdmet.system import lattice as libdmet_lattice  # pylint: disable=import-outside-toplevel

    audit = configuration['translation_symmetry']['builder_audit']
    if not audit.get('eligible'):
        raise ValueError('Builder translated DMET requires a successful translation audit')
    dimension = int(audit['dimension'])
    lattice_shape = [int(value) for value in audit['repetitions']]
    impurity_shape = [int(value) for value in configuration['impurity_shape']]
    outer_shape = [
        lattice_length // impurity_length
        for lattice_length, impurity_length in zip(lattice_shape, impurity_shape)
    ]
    primitive_cell = spec.get('primitive_cell')
    primitive_cell = primitive_cell if isinstance(primitive_cell, Mapping) else {}
    vectors = np.asarray(primitive_cell.get('vectors'), dtype=float)
    cell_matrix = np.asarray(
        [vector[:dimension] for vector in vectors[:dimension]],
        dtype=float,
    )
    if cell_matrix.shape != (dimension, dimension) or abs(float(np.linalg.det(cell_matrix))) < 1.0e-10:
        lengths = [float(np.linalg.norm(vector)) for vector in vectors[:dimension]]
        cell_matrix = np.diag([length if length > 1.0e-8 else 1.0 for length in lengths])
    basis_sites = sorted(audit['basis_sites'], key=lambda item: int(item['basis_index']))
    unit_cell = libdmet_lattice.UnitCell(
        cell_matrix,
        [
            (np.asarray(site['position'], dtype=float), 'H')
            for site in basis_sites
        ],
    )
    supercell = libdmet_lattice.SuperCell(unit_cell, impurity_shape)
    lattice = libdmet_lattice.LatticeModel(supercell, outer_shape)

    basis_size = int(audit['basis_size'])
    internal_cells = list(itertools.product(*[range(length) for length in impurity_shape]))
    internal_cell_position = {
        tuple(cell): position for position, cell in enumerate(internal_cells)
    }
    h1e = np.zeros((lattice.ncells, lattice.nscsites, lattice.nscsites), dtype=float)
    for source_cell in internal_cells:
        source_cell_array = np.asarray(source_cell, dtype=int)
        source_cell_position = internal_cell_position[tuple(source_cell)]
        for term in audit['one_body_terms']:
            source_basis = int(term['source_basis'])
            target_basis = int(term['target_basis'])
            primitive_offset = np.asarray(term['cell_offset'], dtype=int)
            target_primitive_cell = source_cell_array + primitive_offset
            target_internal_cell = tuple(
                int(value % length)
                for value, length in zip(target_primitive_cell, impurity_shape)
            )
            supercell_offset = np.asarray([
                int(value // length)
                for value, length in zip(target_primitive_cell, impurity_shape)
            ], dtype=int)
            source_orbital = source_cell_position * basis_size + source_basis
            target_orbital = (
                internal_cell_position[target_internal_cell] * basis_size + target_basis
            )
            cell_index = lattice.cell_pos2idx(supercell_offset)
            h1e[cell_index, target_orbital, source_orbital] += (
                float(term['t']) * int(term.get('multiplicity', 1))
            )

    zero_cell = lattice.cell_pos2idx(np.zeros(dimension, dtype=int))
    for internal_position, _cell in enumerate(internal_cells):
        for basis_site in basis_sites:
            orbital = internal_position * basis_size + int(basis_site['basis_index'])
            h1e[zero_cell, orbital, orbital] += float(basis_site.get('epsilon', 0.0))

    h1e_k = lattice.R2k(h1e)
    hermiticity_error = max(
        float(np.max(np.abs(matrix - matrix.conj().T)))
        for matrix in h1e_k
    )
    if hermiticity_error > 1.0e-8:
        raise ValueError(
            'Audited Builder translation blocks are not Hermitian '
            '(maximum k-space error={0:.3e})'.format(hermiticity_error)
        )

    h2e = np.zeros((lattice.nscsites,) * 4, dtype=float)
    for internal_position, _cell in enumerate(internal_cells):
        for basis_site in basis_sites:
            orbital = internal_position * basis_size + int(basis_site['basis_index'])
            h2e[orbital, orbital, orbital, orbital] = float(basis_site.get('U', 0.0))
    hamiltonian = libdmet_hamiltonian.HamNonInt(lattice, h1e, h2e)
    lattice.setHam(hamiltonian, use_hcore_as_emb_ham=True)
    return lattice, hamiltonian


def _zero_correction_starting_potential(
    spec: Mapping[str, Any],
    dmet: Any,
    hamiltonian: Any,
    configuration: Mapping[str, Any],
    orbital_count: int,
    *,
    slater: Any,
    np: Any,
) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
    """Build libDMET's physical mean-field baseline with zero auxiliary correction.

    libDMET 0.5 represents the physical Hartree-Fock baseline and fitted DMET
    correction in one Vcor object.  The public ``zero`` policy applies only to
    the auxiliary correction; clearing the combined object would remove the
    Hubbard mean field itself.
    """

    restricted = configuration['reference'] == 'restricted'
    if configuration.get('reference_density_source'):
        initial_density, initialization = load_density_seed(spec, configuration, np=np)
    else:
        initial_density, initialization = build_reference_density_seed(
            spec,
            configuration,
            int(orbital_count),
            np=np,
        )

    def physical_baseline(spin_density: Any) -> Any:
        reference_density = (
            np.asarray(spin_density, dtype=float).sum(axis=0)
            if restricted
            else np.asarray(spin_density, dtype=float)
        )
        value = np.asarray(slater.get_veff(reference_density, hamiltonian.H2), dtype=float)
        if value.ndim == 2:
            value = value[np.newaxis, :, :]
        if restricted and value.shape[0] == 1:
            value = np.repeat(value, 2, axis=0)
        if value.shape != (2, int(orbital_count), int(orbital_count)):
            raise ValueError(
                'DMET mean-field baseline has shape {0}; expected (2, {1}, {1})'.format(
                    tuple(value.shape),
                    int(orbital_count),
                    int(orbital_count),
                )
            )
        return value

    density_seed = np.asarray(initial_density, dtype=float)
    baseline = physical_baseline(density_seed)
    potential = dmet.VcorLocal(restricted, False, int(orbital_count))
    potential.assign(baseline)
    initial_correction = np.zeros_like(baseline, dtype=float)
    initialization.update({
        'status': 'prepared',
        'self_consistent': False,
        'density_array_key': 'reference_density_seed',
        'correlation_potential_strategy': 'zero',
    })
    return (
        potential,
        baseline.copy(),
        initial_correction,
        density_seed.copy(),
        initialization,
    )


def _finite_graph_fragment_h2(
    fragment_lattice: Any,
    basis: Any,
    impurity_orbital_count: int,
    np: Any,
) -> Any:
    """Project only the fragment interaction into a noninteracting bath."""

    source_h2 = np.asarray(
        fragment_lattice.getH2(compact=False, kspace=False)
    )
    if source_h2.ndim != 4:
        raise ValueError(
            'Finite-graph noninteracting-bath DMET requires a four-index model interaction tensor'
        )
    if basis.shape[1] != 1:
        raise ValueError(
            'Finite-graph noninteracting-bath DMET requires a single-cell graph representation'
        )
    spin_count = int(basis.shape[0])
    embedding_orbital_count = int(basis.shape[-1])
    component_count = spin_count * (spin_count + 1) // 2
    embedded_h2 = np.zeros(
        (
            component_count,
            embedding_orbital_count,
            embedding_orbital_count,
            embedding_orbital_count,
            embedding_orbital_count,
        ),
        dtype=source_h2.dtype,
    )

    def transform(coefficients_1: Any, coefficients_2: Any) -> Any:
        return np.einsum(
            'pqrs,pi,qj,rk,sl->ijkl',
            source_h2,
            coefficients_1,
            coefficients_1,
            coefficients_2,
            coefficients_2,
            optimize=True,
        )

    alpha_impurity = basis[0, 0, :, :impurity_orbital_count]
    impurity_slice = (
        slice(None),
        slice(0, impurity_orbital_count),
        slice(0, impurity_orbital_count),
        slice(0, impurity_orbital_count),
        slice(0, impurity_orbital_count),
    )
    if spin_count == 1:
        embedded_h2[impurity_slice] = transform(alpha_impurity, alpha_impurity)[None]
        return embedded_h2

    beta_impurity = basis[1, 0, :, :impurity_orbital_count]
    embedded_h2[0, :impurity_orbital_count, :impurity_orbital_count,
                :impurity_orbital_count, :impurity_orbital_count] = transform(
                    alpha_impurity,
                    alpha_impurity,
                )
    embedded_h2[1, :impurity_orbital_count, :impurity_orbital_count,
                :impurity_orbital_count, :impurity_orbital_count] = transform(
                    beta_impurity,
                    beta_impurity,
                )
    embedded_h2[2, :impurity_orbital_count, :impurity_orbital_count,
                :impurity_orbital_count, :impurity_orbital_count] = transform(
                    alpha_impurity,
                    beta_impurity,
                )
    return embedded_h2


def _construct_finite_graph_impurity_hamiltonian(
    fragment_lattice: Any,
    density: Any,
    correlation_potential: Any,
    configuration: Mapping[str, Any],
    basis_kind: str,
    *,
    dmet: Any,
    slater: Any,
    np: Any,
    mean_field_baseline: Any = None,
    bath_diagnostics: Any = None,
) -> Tuple[Any, Any, Any]:
    if configuration['interacting_bath']:
        return dmet.ConstructImpHam(
            fragment_lattice,
            density,
            correlation_potential,
            matching=False,
            int_bath=True,
            orth=True,
            kind=basis_kind,
        )

    if basis_kind == 'eig' and configuration.get('bath_spin_dimension_policy', 'native') == 'max':
        from .spin_bath import max_spin_eigen_bath
        basis, diagnostics = max_spin_eigen_bath(fragment_lattice, density)
        if bath_diagnostics is not None:
            bath_diagnostics.append(diagnostics)
        print('Bath spin dimension policy: {0}'.format(diagnostics), flush=True)
    else:
        basis = slater.embBasis(
            fragment_lattice, density, local=True, orth=True, kind=basis_kind,
        )
        if bath_diagnostics is not None:
            bath_diagnostics.append({'policy': 'native', 'kind': basis_kind})
    fragment_h2 = _finite_graph_fragment_h2(
        fragment_lattice,
        basis,
        int(fragment_lattice.nimp),
        np,
    )
    if configuration['contains_intersite_v']:
        # The omitted two-body terms contribute through F[D] - JK_imp[D].
        # vcor contains the seed HF potential as well as the fitted correction;
        # adding that baseline again would double count the environment field.
        if mean_field_baseline is None:
            raise ValueError('Noninteracting bath with V requires the HF baseline')
        correction = np.array(correlation_potential.get() - mean_field_baseline, copy=True)
        indices = np.asarray(fragment_lattice.imp_idx, dtype=int)
        correction[:, indices[:, None], indices] = 0.0
        correlation_potential = dmet.VcorLocal(
            configuration['reference'] == 'restricted', False, correction.shape[-1],
        )
        correlation_potential.assign(correction)
        fragment_lattice.use_hcore_as_emb_ham = False
    impurity_hamiltonian, h1e_energy = slater.embHam(
        fragment_lattice,
        basis,
        correlation_potential,
        local=True,
        int_bath=False,
        H2_given=fragment_h2,
        # Native transform_imp removes the whole cell, which is the entire
        # graph here. We have already removed only this fragment's block.
        fitting=bool(configuration['contains_intersite_v']),
    )
    return impurity_hamiltonian, h1e_energy, basis


def _noninteracting_v_fragment_energy(
    fragment_lattice: Any,
    basis: Any,
    correlation_potential: Any,
    impurity_solver: Any,
    solver_args: Mapping[str, Any],
    *,
    slater: Any,
) -> float:
    """Evaluate the physical U+V operator in the NIB correlated state.

    Build a separate full-interaction energy Hamiltonian so the solving
    Hamiltonian and its omitted-interaction mean field remain untouched.
    libDMET applies democratic impurity-index weights and half the frozen-core
    potential. No auxiliary correlation potential enters this observable.
    """
    energy_lattice = copy.copy(fragment_lattice)
    physical_ham, _ = slater.embHam(
        energy_lattice, basis, correlation_potential,
        local=True, int_bath=True,
    )
    return float(slater.get_E_dmet(
        basis, energy_lattice, physical_ham, 0.0,
        solver=impurity_solver, solver_args=dict(solver_args),
    ))


def _prepare_interacting_bath_mean_field(
    lattice: Any,
    hamiltonian: Any,
    mean_field_result: Mapping[str, Any],
    configuration: Mapping[str, Any],
    *,
    slater: Any,
    np: Any,
    mirrors: Sequence[Any] = (),
) -> None:
    """Cache the lattice density and Fock matrices required by an interacting bath."""

    total_density_k = np.asarray(mean_field_result['rho_k'])
    if configuration['reference'] == 'restricted':
        total_density_k = total_density_k * 2.0
    # Cell-local interactions use the R=0 density (the k-point average).
    # Their physical HF potential is the same at every k point.
    mean_field_veff = slater.get_veff(total_density_k.mean(axis=1), hamiltonian.H2)
    if configuration['reference'] == 'restricted':
        mean_field_veff_k = mean_field_veff[0]
    else:
        mean_field_veff_k = mean_field_veff[:, np.newaxis]
    lattice.rdm1_lo_k = total_density_k
    lattice.rdm1_lo_R = lattice.k2R(total_density_k)
    lattice.fock_lo_k = lattice.hcore_lo_k + mean_field_veff_k
    lattice.fock_lo_R = lattice.k2R(lattice.fock_lo_k)
    for target in mirrors:
        target.rdm1_lo_k = lattice.rdm1_lo_k
        target.rdm1_lo_R = lattice.rdm1_lo_R
        target.fock_lo_k = lattice.fock_lo_k
        target.fock_lo_R = lattice.fock_lo_R


def _run_finite_graph_dmet(
    spec: Mapping[str, Any],
    configuration: Mapping[str, Any],
    *,
    dmet: Any,
    np: Any,
    la: Any,
    lib: Any,
    scratch_directory: Optional[str],
    collect_local_rdms: bool = False,
) -> Dict[str, Any]:
    from libdmet.routine import slater  # pylint: disable=import-outside-toplevel

    lattice, hamiltonian = _finite_graph_lattice(spec, np)
    fragments = list(configuration['fragments'])
    fragment_lattices = []
    for fragment in fragments:
        fragment_lattice = copy.copy(lattice)
        fragment_lattice.set_val_virt_core(fragment['orbital_indices'], [], [])
        fragment_lattice.setHam(hamiltonian, use_hcore_as_emb_ham=True)
        fragment_lattices.append(fragment_lattice)

    filling = copy.deepcopy(configuration['mean_field_filling'])
    (
        correlation_potential,
        mean_field_baseline_potential,
        initial_correlation_potential,
        reference_density_seed,
        reference_density_initialization,
    ) = _zero_correction_starting_potential(
        spec,
        dmet,
        hamiltonian,
        configuration,
        lattice.nscsites,
        slater=slater,
        np=np,
    )
    impurity_solvers = [
        _solver(
            dmet,
            configuration,
            scratch_directory,
            solver_id='impurity-{0}'.format(position + 1),
        )
        for position, _fragment in enumerate(fragments)
    ]
    chemical_potential = None
    last_dmu = 0.0
    previous_energy = 0.0
    previous_mean_field_density = None
    converged = False
    iteration_records: List[Dict[str, Any]] = []
    adiis = lib.diis.DIIS()
    adiis.space = configuration['diis_space']
    final_arrays: Dict[str, Any] = {}
    fragment_hamiltonian_shapes: List[Dict[str, Any]] = []
    fragment_results: List[Dict[str, Any]] = []
    total_energy = None
    reference_energy = None
    translation_tied_fit = bool(
        configuration['translation_symmetry']['eligible']
        and configuration['translation_symmetry']['checks']['automatic_fragment_selector']
    )
    final_fragment_fit: Dict[str, Any] = {}

    for iteration in range(configuration['max_iterations']):
        mean_field = (
            dmet.RHartreeFock
            if configuration['reference'] == 'restricted'
            else dmet.HartreeFock
        )
        density, chemical_potential, mean_field_result = mean_field(
            lattice,
            correlation_potential,
            filling,
            chemical_potential,
            beta=_lattice_scf_beta(configuration),
            ires=True,
        )
        density_change = _mean_field_rdm1_change(
            density,
            previous_mean_field_density,
            np,
        )
        previous_mean_field_density = np.array(density, copy=True)
        reference_energy = float(mean_field_result['E'])
        basis_kind, occupation_diagnostics = _embedding_basis_policy_for_mean_field(
            mean_field_result,
            np,
        )
        if configuration['interacting_bath'] or configuration['contains_intersite_v']:
            _prepare_interacting_bath_mean_field(
                lattice,
                hamiltonian,
                mean_field_result,
                configuration,
                slater=slater,
                np=np,
                mirrors=fragment_lattices,
            )
        impurity_hamiltonians = []
        h1e_energies = []
        bases = []
        solver_arguments = []
        impurity_sectors = []
        bath_dimension_diagnostics = []
        for fragment_lattice in fragment_lattices:
            try:
                impurity_hamiltonian, h1e_energy, basis = _construct_finite_graph_impurity_hamiltonian(
                    fragment_lattice,
                    density,
                    correlation_potential,
                    configuration,
                    basis_kind,
                    dmet=dmet,
                    slater=slater,
                    np=np,
                    mean_field_baseline=mean_field_baseline_potential,
                    bath_diagnostics=bath_dimension_diagnostics,
                )
            except ValueError:
                if basis_kind == 'eig' and scratch_directory:
                    from .bath_diagnostics import save_bath_failure
                    try:
                        save_bath_failure(
                            scratch_directory, fragment_lattice, density,
                            mean_field_result, correlation_potential,
                            iteration=iteration + 1,
                        )
                    except Exception as diagnostic_error:
                        print('Unable to save bath diagnostics: {0}'.format(diagnostic_error), file=sys.stderr)
                raise
            _adapt_unrestricted_overlap_for_pyscf(
                impurity_hamiltonian,
                configuration,
                np,
            )
            impurity_hamiltonian = dmet.apply_dmu(
                fragment_lattice,
                impurity_hamiltonian,
                basis,
                last_dmu,
            )
            basis_k = lattice.R2k_basis(basis)
            folded_density = dmet.foldRho_k(mean_field_result['rho_k'], basis_k)
            if configuration['reference'] == 'restricted':
                folded_density = folded_density * 2.0
            impurity_electron_count, electron_count_diagnostics = (
                _embedding_electron_count_from_projected_density(
                    folded_density,
                    int(impurity_hamiltonian.norb),
                    configuration['reference'],
                    np,
                    fractional_occupation_count=(
                        occupation_diagnostics['fractional_occupation_count'] or 0
                    ),
                    fractional_sector_electron_count=min(
                        int(fragment_lattice.nval) * 2,
                        int(configuration['nalpha']) + int(configuration['nbeta']),
                    ),
                )
            )
            impurity_sector = {
                **_impurity_spin_sector(
                    impurity_electron_count,
                    int(impurity_hamiltonian.norb),
                    configuration,
                ),
                'electron_count_selection': electron_count_diagnostics,
            }
            solver_args = {
                'nelec': impurity_electron_count,
                'dm0': folded_density,
            }
            impurity_hamiltonians.append(impurity_hamiltonian)
            h1e_energies.append(h1e_energy)
            bases.append(basis)
            solver_arguments.append(solver_args)
            impurity_sectors.append(impurity_sector)

        rho_embeddings, embedded_energies, impurity_hamiltonians, dmu = (
            _solve_impurity_hamiltonians_with_fitting(
                dmet,
                fragment_lattices,
                filling,
                impurity_hamiltonians,
                bases,
                impurity_solvers,
                solver_args=solver_arguments,
            )
        )
        last_dmu += float(dmu)
        fragment_results = []
        fragment_energies = []
        fragment_density_matrices = []
        fragment_hamiltonian_shapes = []
        for fragment, fragment_lattice, rho_embedding, embedded_energy, basis, (
            impurity_hamiltonian
        ), h1e_energy, impurity_solver, solver_args, impurity_sector in zip(
            fragments,
            fragment_lattices,
            rho_embeddings,
            embedded_energies,
            bases,
            impurity_hamiltonians,
            h1e_energies,
            impurity_solvers,
            solver_arguments,
            impurity_sectors,
        ):
            if not configuration['interacting_bath'] and configuration['contains_intersite_v']:
                rho_imp, _, _ = slater.transformResults(
                    rho_embedding, None, basis, impurity_hamiltonian,
                    lattice=fragment_lattice,
                )
                fragment_energy = _noninteracting_v_fragment_energy(
                    fragment_lattice, basis, correlation_potential,
                    impurity_solver, solver_args, slater=slater,
                )
            else:
                rho_imp, normalized_energy, _normalized_electron_count = dmet.transformResults(
                    rho_embedding,
                    embedded_energy,
                    basis,
                    impurity_hamiltonian,
                    h1e_energy,
                    lattice=fragment_lattice,
                    last_dmu=last_dmu,
                    dmu_idx=fragment_lattice.imp_idx,
                    int_bath=bool(configuration['interacting_bath']),
                    solver=impurity_solver,
                    solver_args=solver_args,
                )
                fragment_energy = float(normalized_energy * lattice.nscsites)
            fragment_energies.append(fragment_energy)
            fragment_density_matrices.append(rho_imp)
            spin_population = _spin_population_summary(rho_imp, configuration, np)
            fragment_electron_count = float(spin_population['total_electron_count'])
            fragment_results.append({
                **copy.deepcopy(fragment),
                'energy': fragment_energy,
                'electron_count': fragment_electron_count,
                'embedding_orbital_count': int(basis.shape[-1]),
                'spin_population': spin_population,
                'impurity_spin_sector': impurity_sector,
            })
            fragment_hamiltonian_shapes.append({
                'fragment_id': fragment['fragment_id'],
                'one_body_component_shapes': _array_component_shapes(impurity_hamiltonian.H1),
                'two_body_component_shapes': _array_component_shapes(impurity_hamiltonian.H2),
            })

        total_energy = float(sum(fragment_energies))
        global_density = slater.get_rho_glob_R(
            bases,
            fragment_lattices,
            rho_embeddings,
        )
        full_global_density = _expanded_global_density_matrix(
            lattice,
            global_density,
            np,
        )
        (
            new_correlation_potential,
            density_fit_error_begin,
            density_fit_error,
            fragment_fit,
        ) = fit_fragment_local_correlation_potential(
            global_density,
            lattice,
            correlation_potential,
            filling,
            fragments,
            translation_tied=translation_tied_fit,
            slater=slater,
            beta=_density_fit_beta(configuration),
            max_iterations=_LIBDMET_NATIVE_FIT_MAX_ITERATIONS,
        )
        final_fragment_fit = fragment_fit
        if iteration >= configuration['trace_fix_start']:
            diagonal_change = np.average(np.diagonal(
                (new_correlation_potential.get() - correlation_potential.get())[:2],
                0,
                1,
                2,
            ))
            new_correlation_potential = dmet.addDiag(new_correlation_potential, -diagonal_change)
        next_parameters, potential_update = _mix_correlation_potential(
            correlation_potential.param, new_correlation_potential.param,
            configuration, iteration, adiis, np,
        )
        correlation_potential.update(next_parameters)
        energy_change = None if previous_energy is None else float(total_energy - previous_energy)
        previous_energy = total_energy
        iteration_records.append({
            'iteration': iteration + 1,
            'energy': total_energy,
            'energy_per_site': total_energy / lattice.nscsites,
            'mean_field_energy': reference_energy,
            'mean_field_energy_per_site': reference_energy / lattice.nscsites,
            'energy_change': energy_change,
            'mean_field_rdm1_change_max_abs': density_change,
            'bath_basis': copy.deepcopy(occupation_diagnostics),
            'bath_spin_dimension_diagnostics': copy.deepcopy(bath_dimension_diagnostics),
            'embedding_orbital_counts': [int(basis.shape[-1]) for basis in bases],
            'impurity_electron_counts': [
                int(item['nelec']) for item in solver_arguments
            ],
            'electron_count_selections': [
                copy.deepcopy(item['electron_count_selection'])
                for item in impurity_sectors
            ],
            'density_fit_error_begin': float(density_fit_error_begin),
            'density_fit_error': float(density_fit_error),
            'density_fit_parameterization': fragment_fit['parameterization'],
            **potential_update,
            'fragment_electron_counts': [item['electron_count'] for item in fragment_results],
            'chemical_potential': _serializable_chemical_potential(chemical_potential, np),
            'impurity_chemical_potential_shift': float(last_dmu),
            'diis_vectors': int(adiis.get_num_vec()),
        })
        final_arrays = {
            # Store the density that produced this iteration's energy/bath,
            # before applying the next fitted-potential update.
            'mean_field_density_matrix': np.repeat(density[:, 0], 2, axis=0) if density.shape[0] == 1 else density[:, 0].copy(),
            'global_embedding_density_matrix': global_density,
            'initial_correlation_potential': initial_correlation_potential,
            'reference_density_seed': reference_density_seed,
            'mean_field_baseline_potential': mean_field_baseline_potential,
            'correlation_potential': (
                correlation_potential.get() - mean_field_baseline_potential
            ),
            'effective_local_potential': correlation_potential.get(),
        }
        if full_global_density is not None:
            final_arrays['global_embedding_density_matrix_full'] = full_global_density
        for position, (basis, rho_embedding, rho_imp, impurity_hamiltonian, impurity_solver) in enumerate(zip(
            bases,
            rho_embeddings,
            fragment_density_matrices,
            impurity_hamiltonians,
            impurity_solvers,
        )):
            prefix = 'fragment_{0}'.format(position + 1)
            final_arrays['{0}_embedding_basis'.format(prefix)] = basis
            final_arrays['{0}_embedding_density_matrix'.format(prefix)] = rho_embedding
            final_arrays['{0}_density_matrix'.format(prefix)] = rho_imp
            local_rdm2 = impurity_spin_rdm2(
                impurity_solver,
                int(rho_imp.shape[-1]),
            )
            if local_rdm2 is not None:
                final_arrays['{0}_spin_rdm2'.format(prefix)] = local_rdm2
            _add_numerical_components(final_arrays, '{0}_impurity_h1'.format(prefix), impurity_hamiltonian.H1)
            _add_numerical_components(final_arrays, '{0}_impurity_h2'.format(prefix), impurity_hamiltonian.H2)
        if _native_dmet_converged(
            energy_change,
            density_change,
            density_fit_error,
            configuration,
        ):
            converged = True
            break

    if collect_local_rdms and configuration['impurity_solver'] == 'ccsd':
        # CCSD energy evaluation need not retain its 2RDM. Materialize it only
        # for requested diagnostics, once per final fragment, after the loop.
        for position, impurity_solver in enumerate(impurity_solvers):
            impurity_solver.make_rdm2()
            prefix = 'fragment_{0}'.format(position + 1)
            count = int(final_arrays['{0}_density_matrix'.format(prefix)].shape[-1])
            local_rdm2 = impurity_spin_rdm2(impurity_solver, count)
            if local_rdm2 is not None:
                final_arrays['{0}_spin_rdm2'.format(prefix)] = local_rdm2

    assert total_energy is not None
    assert reference_energy is not None
    site_count = int(configuration['site_count'])
    iteration_history = {
        'schema': DMET_ITERATION_HISTORY_SCHEMA,
        'converged': converged,
        'iteration_count': len(iteration_records),
        'convergence_contract': _native_convergence_contract(
            configuration,
            energy_measure='finite_graph_total_energy',
        ),
        'correlation_potential_fit': {
            **_fragment_fitting_contract(_density_fit_beta(configuration), translation_tied=translation_tied_fit),
            **final_fragment_fit,
        },
        'records': iteration_records,
    }
    return {
        'schema': DMET_RESULT_SCHEMA,
        'provider': 'libdmet',
        'method': 'dmet',
        'impurity_solver': configuration['impurity_solver'],
        **_ccsd_smearing_metadata(configuration),
        'impurity_solver_details': _impurity_solver_details(impurity_solvers),
        'converged': converged,
        'energy': total_energy,
        'energy_per_site': total_energy / site_count,
        'energy_normalization': 'finite_graph_fragment_partition',
        'site_count': site_count,
        'fragment_count': len(fragment_results),
        'fragment_site_count': sum(len(item['site_ids']) for item in fragment_results),
        'fragment_electron_count': sum(item['electron_count'] for item in fragment_results),
        'fragments': fragment_results,
        'spin_reference': {
            'mode': (
                'restricted'
                if configuration['reference'] == 'restricted'
                else 'unrestricted_fixed_sz'
            ),
            'initial_density_strategy': reference_density_initialization['strategy'],
            'nalpha': int(configuration['nalpha']),
            'nbeta': int(configuration['nbeta']),
            'total_sz': 0.5 * (
                int(configuration['nalpha']) - int(configuration['nbeta'])
            ),
            'libdmet_sz': int(configuration['libdmet_sz']),
        },
        'chemical_potential': _serializable_chemical_potential(chemical_potential, np),
        'impurity_chemical_potential_shift': float(last_dmu),
        'initial_correlation_potential': {
            'strategy': 'zero',
            'array_key': 'initial_correlation_potential',
            'mean_field_baseline': 'reference_density_seed_hartree_fock',
            'mean_field_baseline_array_key': 'mean_field_baseline_potential',
        },
        'reference_density_initialization': reference_density_initialization,
        'mean_field_state': state_metadata(spec, configuration, converged=converged),
        'configuration': copy.deepcopy(dict(configuration)),
        'iteration_history': iteration_history,
        'bath': {
            'construction': 'schmidt_decomposition',
            'interacting_bath': bool(configuration['interacting_bath']),
            'spin_dimension_policy': configuration['bath_spin_dimension_policy'],
            'noninteracting_v_convention': (
                {
                    'environment': 'density_updated_physical_fock_minus_impurity_jk',
                    'auxiliary_potential': 'fitted_correction_without_seed_hf_baseline',
                    'energy': 'full_physical_hamiltonian_democratic_partition',
                }
                if not configuration['interacting_bath'] and configuration['contains_intersite_v']
                else None
            ),
            'embedding_orbital_counts': [item['embedding_orbital_count'] for item in fragment_results],
            'basis_policy': 'eig_for_fractional_mean_field_occupations_otherwise_svd',
            'basis_kinds_used': sorted({
                str(record['bath_basis']['kind']) for record in iteration_records
            }),
            'final_basis_kind': str(iteration_records[-1]['bath_basis']['kind']),
        },
        'impurity_hamiltonians': fragment_hamiltonian_shapes,
        '_transient_dmet_arrays': final_arrays,
    }


def _execute_hubbard_dmet(
    spec: Mapping[str, Any],
    options: Optional[Mapping[str, Any]] = None,
    *,
    scratch_directory: Optional[str] = None,
    collect_local_rdms: bool = False,
) -> Dict[str, Any]:
    errors, configuration = validate_dmet_model_request(spec, options)
    if errors or configuration is None:
        raise ValueError('; '.join(errors))

    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import scipy.linalg as la  # pylint: disable=import-outside-toplevel
        from pyscf import lib  # pylint: disable=import-outside-toplevel
        import libdmet.dmet.Hubbard as dmet  # pylint: disable=import-outside-toplevel
        from libdmet.routine import slater  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - optional provider
        raise RuntimeError('DMET execution requires the optional libDMET provider') from exc

    if configuration['execution_mode'] == 'finite_graph':
        return _run_finite_graph_dmet(
            spec,
            configuration,
            dmet=dmet,
            np=np,
            la=la,
            lib=lib,
            scratch_directory=scratch_directory,
            collect_local_rdms=collect_local_rdms,
        )

    lattice_shape = list(configuration['lattice_shape'])
    impurity_shape = list(configuration['impurity_shape'])
    if configuration['translation_backend'] == 'builder_primitive_cell':
        lattice, hamiltonian = _translated_builder_lattice(spec, configuration, np)
    elif configuration['dimension'] == 1:
        lattice = dmet.ChainLattice(lattice_shape[0], impurity_shape[0])
        hamiltonian = dmet.Ham(
            lattice,
            configuration['onsite_u'],
            tlist=[configuration['libdmet_hopping_parameter']],
        )
        lattice.setHam(hamiltonian, use_hcore_as_emb_ham=True)
    else:
        lattice = dmet.SquareLattice(*(lattice_shape + impurity_shape))
        hamiltonian = dmet.Ham(
            lattice,
            configuration['onsite_u'],
            tlist=[configuration['libdmet_hopping_parameter']],
        )
        lattice.setHam(hamiltonian, use_hcore_as_emb_ham=True)
    filling = copy.deepcopy(configuration['mean_field_filling'])
    restricted = configuration['reference'] == 'restricted'
    (
        correlation_potential,
        mean_field_baseline_potential,
        initial_correlation_potential,
        reference_density_seed,
        reference_density_initialization,
    ) = _zero_correction_starting_potential(
        spec,
        dmet,
        hamiltonian,
        configuration,
        lattice.nscsites,
        slater=slater,
        np=np,
    )
    impurity_solver = _solver(
        dmet,
        configuration,
        scratch_directory,
        solver_id='representative-impurity',
    )
    chemical_potential = None
    last_dmu = 0.0
    previous_energy = 0.0
    previous_mean_field_density = None
    converged = False
    iteration_records: List[Dict[str, Any]] = []
    adiis = lib.diis.DIIS()
    adiis.space = configuration['diis_space']
    final_arrays: Dict[str, Any] = {}
    impurity_h1_shapes: Dict[str, List[int]] = {}
    impurity_h2_shapes: Dict[str, List[int]] = {}
    energy_per_site = None
    mean_field_energy_per_site = None
    fragment_electron_count = None

    for iteration in range(configuration['max_iterations']):
        mean_field = dmet.RHartreeFock if restricted else dmet.HartreeFock
        density, chemical_potential, mean_field_result = mean_field(
            lattice,
            correlation_potential,
            filling,
            chemical_potential,
            beta=_lattice_scf_beta(configuration),
            ires=True,
        )
        density_change = _mean_field_rdm1_change(
            density,
            previous_mean_field_density,
            np,
        )
        previous_mean_field_density = np.array(density, copy=True)
        mean_field_energy_per_site = float(mean_field_result['E'] / lattice.supercell.nsites)
        basis_kind, occupation_diagnostics = _embedding_basis_policy_for_mean_field(
            mean_field_result,
            np,
        )
        if configuration['interacting_bath']:
            _prepare_interacting_bath_mean_field(
                lattice,
                hamiltonian,
                mean_field_result,
                configuration,
                slater=slater,
                np=np,
            )
        impurity_hamiltonian, h1e_energy, basis = dmet.ConstructImpHam(
            lattice,
            density,
            correlation_potential,
            matching=False,
            int_bath=bool(configuration['interacting_bath']),
            kind=basis_kind,
        )
        _adapt_unrestricted_overlap_for_pyscf(
            impurity_hamiltonian,
            configuration,
            np,
        )
        impurity_hamiltonian = dmet.apply_dmu(lattice, impurity_hamiltonian, basis, last_dmu)
        basis_k = lattice.R2k_basis(basis)
        folded_density = dmet.foldRho_k(mean_field_result['rho_k'], basis_k)
        if restricted:
            folded_density = folded_density * 2.0
        impurity_electron_count, electron_count_diagnostics = (
            _embedding_electron_count_from_projected_density(
                folded_density,
                int(impurity_hamiltonian.norb),
                configuration['reference'],
                np,
                fractional_occupation_count=(
                    occupation_diagnostics['fractional_occupation_count'] or 0
                ),
                fractional_sector_electron_count=(
                    int(lattice.ncore) + int(lattice.nval)
                ) * 2,
            )
        )
        impurity_sector = {
            **_impurity_spin_sector(
                impurity_electron_count,
                int(impurity_hamiltonian.norb),
                configuration,
            ),
            'electron_count_selection': electron_count_diagnostics,
        }
        solver_args = {
            'nelec': impurity_electron_count,
            'dm0': folded_density,
        }
        rho_emb, embedded_energy, impurity_hamiltonian, dmu = (
            _solve_impurity_hamiltonians_with_fitting(
                dmet,
                lattice,
                filling,
                impurity_hamiltonian,
                basis,
                impurity_solver,
                solver_args,
            )
        )
        last_dmu += float(dmu)
        rho_imp, energy_per_site, _normalized_electron_count = dmet.transformResults(
            rho_emb,
            embedded_energy,
            basis,
            impurity_hamiltonian,
            h1e_energy,
            lattice=lattice,
            last_dmu=last_dmu,
            int_bath=bool(configuration['interacting_bath']),
            solver=impurity_solver,
            solver_args=solver_args,
        )
        energy_per_site = float(energy_per_site)
        fragment_spin_population = _spin_population_summary(rho_imp, configuration, np)
        fragment_electron_count = float(
            fragment_spin_population['total_electron_count']
        )
        global_density = slater.get_rho_glob_R(
            [basis],
            [lattice],
            [rho_emb],
        )
        full_global_density = _expanded_global_density_matrix(
            lattice,
            global_density,
            np,
        )
        (
            new_correlation_potential,
            density_fit_error_begin,
            density_fit_error,
        ) = _fit_native_lattice_correlation_potential(
            global_density,
            lattice,
            correlation_potential,
            filling,
            beta=_density_fit_beta(configuration),
            slater=slater,
            np=np,
        )
        if iteration >= configuration['trace_fix_start']:
            diagonal_change = np.average(np.diagonal(
                (new_correlation_potential.get() - correlation_potential.get())[:2],
                0,
                1,
                2,
            ))
            new_correlation_potential = dmet.addDiag(new_correlation_potential, -diagonal_change)
        next_parameters, potential_update = _mix_correlation_potential(
            correlation_potential.param, new_correlation_potential.param,
            configuration, iteration, adiis, np,
        )
        correlation_potential.update(next_parameters)
        convergence_energy = float(energy_per_site * lattice.supercell.nsites)
        energy_change = float(convergence_energy - previous_energy)
        previous_energy = convergence_energy
        iteration_records.append({
            'iteration': iteration + 1,
            'energy_per_site': energy_per_site,
            'representative_fragment_energy': convergence_energy,
            'mean_field_energy_per_site': mean_field_energy_per_site,
            'energy_change': energy_change,
            'mean_field_rdm1_change_max_abs': density_change,
            'bath_basis': copy.deepcopy(occupation_diagnostics),
            'embedding_orbital_count': int(basis.shape[-1]),
            'impurity_electron_count': int(solver_args['nelec']),
            'electron_count_selection': copy.deepcopy(
                impurity_sector['electron_count_selection']
            ),
            'density_fit_error_begin': float(density_fit_error_begin),
            'density_fit_error': float(density_fit_error),
            **potential_update,
            'fragment_electron_count': fragment_electron_count,
            'chemical_potential': _serializable_chemical_potential(chemical_potential, np),
            'impurity_chemical_potential_shift': float(last_dmu),
            'diis_vectors': int(adiis.get_num_vec()),
        })
        final_arrays = {
            'embedding_basis': basis,
            'embedding_density_matrix': rho_emb,
            'global_embedding_density_matrix': global_density,
            'fragment_density_matrix': rho_imp,
            'initial_correlation_potential': initial_correlation_potential,
            'reference_density_seed': reference_density_seed,
            'mean_field_baseline_potential': mean_field_baseline_potential,
            'correlation_potential': (
                correlation_potential.get() - mean_field_baseline_potential
            ),
            'effective_local_potential': correlation_potential.get(),
        }
        if full_global_density is not None:
            final_arrays['global_embedding_density_matrix_full'] = full_global_density
        impurity_h1_shapes = _array_component_shapes(impurity_hamiltonian.H1)
        impurity_h2_shapes = _array_component_shapes(impurity_hamiltonian.H2)
        _add_numerical_components(final_arrays, 'impurity_h1', impurity_hamiltonian.H1)
        _add_numerical_components(final_arrays, 'impurity_h2', impurity_hamiltonian.H2)
        local_rdm2 = impurity_spin_rdm2(
            impurity_solver,
            int(rho_imp.shape[-1]),
        )
        if local_rdm2 is not None:
            final_arrays['fragment_spin_rdm2'] = local_rdm2
        if _native_dmet_converged(
            energy_change,
            density_change,
            density_fit_error,
            configuration,
        ):
            converged = True
            break

    if collect_local_rdms and configuration['impurity_solver'] == 'ccsd':
        impurity_solver.make_rdm2()
        local_rdm2 = impurity_spin_rdm2(impurity_solver, int(rho_imp.shape[-1]))
        if local_rdm2 is not None:
            final_arrays['fragment_spin_rdm2'] = local_rdm2

    assert energy_per_site is not None
    assert mean_field_energy_per_site is not None
    site_count = int(configuration['site_count'])
    total_energy = float(energy_per_site * site_count)
    iteration_history = {
        'schema': DMET_ITERATION_HISTORY_SCHEMA,
        'converged': converged,
        'iteration_count': len(iteration_records),
        'convergence_contract': _native_convergence_contract(
            configuration,
            energy_measure='representative_fragment_energy',
        ),
        'correlation_potential_fit': _native_fitting_contract(_density_fit_beta(configuration)),
        'records': iteration_records,
    }
    result = {
        'schema': DMET_RESULT_SCHEMA,
        'provider': 'libdmet',
        'method': 'dmet',
        'impurity_solver': configuration['impurity_solver'],
        **_ccsd_smearing_metadata(configuration),
        'impurity_solver_details': _impurity_solver_details([impurity_solver]),
        'converged': converged,
        'energy': total_energy,
        'energy_per_site': float(energy_per_site),
        'energy_normalization': 'total_lattice_estimate',
        'site_count': site_count,
        'fragment_count': 1,
        'fragment_site_count': int(lattice.supercell.nsites),
        'fragment_electron_count': fragment_electron_count,
        'fragments': [
            {
                **copy.deepcopy(configuration['fragments'][0]),
                'energy': float(
                    energy_per_site * lattice.supercell.nsites
                ),
                'energy_normalization': 'representative_fragment_estimate',
                'electron_count': fragment_electron_count,
                'embedding_orbital_count': int(final_arrays['embedding_basis'].shape[-1]),
                'translation_equivalent': True,
                'spin_population': fragment_spin_population,
                'impurity_spin_sector': impurity_sector,
            }
        ],
        'spin_reference': {
            'mode': 'restricted' if restricted else 'unrestricted_fixed_sz',
            'initial_density_strategy': reference_density_initialization['strategy'],
            'nalpha': int(configuration['nalpha']),
            'nbeta': int(configuration['nbeta']),
            'total_sz': 0.5 * (
                int(configuration['nalpha']) - int(configuration['nbeta'])
            ),
            'libdmet_sz': int(configuration['libdmet_sz']),
        },
        'chemical_potential': _serializable_chemical_potential(chemical_potential, np),
        'impurity_chemical_potential_shift': float(last_dmu),
        'initial_correlation_potential': {
            'strategy': 'zero',
            'array_key': 'initial_correlation_potential',
            'mean_field_baseline': 'reference_density_seed_hartree_fock',
            'mean_field_baseline_array_key': 'mean_field_baseline_potential',
        },
        'reference_density_initialization': reference_density_initialization,
        'configuration': copy.deepcopy(configuration),
        'iteration_history': iteration_history,
        'bath': {
            'construction': 'schmidt_decomposition',
            'interacting_bath': bool(configuration['interacting_bath']),
            'spin_dimension_policy': configuration['bath_spin_dimension_policy'],
            'embedding_orbital_count': int(final_arrays['embedding_basis'].shape[-1]),
            'basis_shape': list(final_arrays['embedding_basis'].shape),
            'basis_policy': 'eig_for_fractional_mean_field_occupations_otherwise_svd',
            'basis_kinds_used': sorted({
                str(record['bath_basis']['kind']) for record in iteration_records
            }),
            'final_basis_kind': str(iteration_records[-1]['bath_basis']['kind']),
        },
        'impurity_hamiltonian': {
            'one_body_component_shapes': impurity_h1_shapes,
            'two_body_component_shapes': impurity_h2_shapes,
        },
        '_transient_dmet_arrays': final_arrays,
    }
    return result


def run_hubbard_dmet(
    spec: Mapping[str, Any],
    options: Optional[Mapping[str, Any]] = None,
    *,
    scratch_directory: Optional[str] = None,
    collect_local_rdms: bool = False,
) -> Dict[str, Any]:
    scratch_path = Path(scratch_directory or 'libdmet-scratch').expanduser().resolve()
    scratch_path.mkdir(parents=True, exist_ok=True)
    log_path = scratch_path / 'log-libdmet-output.log'

    # libDMET caches sys.stdout in a module-global logger stream. Keep the
    # complete provider call under one lock so concurrent tasks cannot route
    # output into another task's log while that global stream is redirected.
    with _LIBDMET_OUTPUT_LOCK, log_path.open('a', encoding='utf-8') as log_stream:
        log_stream.write('\n=== libDMET provider run ===\n')
        log_stream.flush()
        provider_logger = None
        pyscf_misc = None
        pyscf_logger = None
        previous_pyscf_flush = None
        previous_sanity_defaults = None
        previous_stream_stdout = None
        with redirect_stdout(log_stream), redirect_stderr(log_stream):
            try:
                import libdmet.utils.logger as provider_logger  # pylint: disable=import-outside-toplevel
                from pyscf.lib import misc as pyscf_misc  # pylint: disable=import-outside-toplevel
                from pyscf.lib import logger as pyscf_logger  # pylint: disable=import-outside-toplevel

                provider_logger.stdout = log_stream
                previous_pyscf_flush = pyscf_logger.flush
                previous_sanity_defaults = pyscf_misc.check_sanity.__defaults__
                previous_stream_stdout = pyscf_misc.StreamObject.stdout
                pyscf_misc.StreamObject.stdout = log_stream
                if previous_sanity_defaults:
                    pyscf_misc.check_sanity.__defaults__ = (
                        *previous_sanity_defaults[:-1],
                        log_stream,
                    )
                result = _execute_hubbard_dmet(
                    spec,
                    options,
                    scratch_directory=str(scratch_path),
                    collect_local_rdms=collect_local_rdms,
                )
                result['quality_checks'] = _dmet_quality_checks(result)
                if 'mean_field_state' in result:
                    from ...result_quality import evaluate_quality_checks
                    result['mean_field_state']['quality_passed'] = (
                        evaluate_quality_checks(result['quality_checks'])['quality_status'] == 'passed'
                    )
            finally:
                if pyscf_misc is not None and previous_stream_stdout is not None:
                    pyscf_misc.StreamObject.stdout = previous_stream_stdout
                if pyscf_misc is not None and previous_sanity_defaults is not None:
                    pyscf_misc.check_sanity.__defaults__ = previous_sanity_defaults
                if pyscf_logger is not None and previous_pyscf_flush is not None:
                    pyscf_logger.flush = previous_pyscf_flush
                if provider_logger is not None:
                    provider_logger.stdout = sys.__stdout__ or sys.stdout

    result['provider_log'] = {
        'provider': 'libdmet',
        'filename': log_path.name,
        'captured_streams': ['stdout', 'stderr'],
    }
    result['_transient_provider_logs'] = [{
        'provider': 'libdmet',
        'kind': 'dmet_output_log',
        'path': str(log_path),
        'mime_type': 'text/plain; charset=utf-8',
        'description': 'Complete libDMET provider stdout and stderr log',
    }]
    return result


__all__ = [
    'DMET_ITERATION_HISTORY_SCHEMA',
    'DMET_RESULT_SCHEMA',
    'normalize_dmet_options',
    'run_hubbard_dmet',
    'validate_dmet_model_request',
]
