from __future__ import annotations

import math
from typing import Any, Dict, List

from ..correlation.scoring import correlation_strength_level as _diagnostic_level

from ..correlation.low_energy import summarize_low_energy_manifold
from .fci_observables import (
    _compute_fci_charge_correlation as _compute_fci_charge_correlation,
    _compute_fci_double_occupancy as _compute_fci_double_occupancy,
    _compute_fci_site_density as _compute_fci_site_density,
    _compute_fci_spin_correlation as _compute_fci_spin_correlation,
    _compute_charge_correlation_from_spin_rdms as _compute_charge_correlation_from_spin_rdms,
    _compute_double_occupancy_from_spin_rdms as _compute_double_occupancy_from_spin_rdms,
    _compute_site_density_from_spin_rdms as _compute_site_density_from_spin_rdms,
    _compute_spin_correlation_from_spin_rdms as _compute_spin_correlation_from_spin_rdms,
    _compute_model_hamiltonian_energy_levels as _compute_model_hamiltonian_energy_levels,
    FULL_DIAGONALIZATION_MAX_DIMENSION,
    fci_determinant_space_dimension,
    _pair_matrix_payload as _pair_matrix_payload,
    require_supported_full_diagonalization as require_supported_full_diagonalization,
    _site_ids_from_spec as _site_ids_from_spec,
    _site_vector_payload as _site_vector_payload,
)
from .reference_diagnostics import (
    _clip_unit,
    _double_excitation_amplitude_summary as _double_excitation_amplitude_summary,
    _fci_natural_occupation_summary as _fci_natural_occupation_summary,
    _mean_field_homo_lumo_summary as _mean_field_homo_lumo_summary,
    _natural_occupation_summary_from_dm1 as _natural_occupation_summary_from_dm1,
    _natural_occupation_summary_from_solver as _natural_occupation_summary_from_solver,
)


def _mean_abs_nonzero(values: List[float]) -> float:
    filtered = [abs(float(value)) for value in values if abs(float(value)) > 1e-12]
    return float(sum(filtered) / len(filtered)) if filtered else 0.0


def _diagnostic_item(name: str, value: Any, score: Any, interpretation: str, *, severity: str = None) -> Dict[str, Any]:
    item = {
        'name': name,
        'value': value,
        'score': None if score is None else round(float(score), 6),
        'severity': severity or ('inconclusive' if score is None else _diagnostic_level(float(score))),
        'interpretation': interpretation,
    }
    return item


def _low_energy_manifold_summary(
    solver_results: Dict[str, Any],
    *,
    mean_abs_t: float,
    fallback_energy_scale: float,
) -> Dict[str, Any]:
    spectrum = solver_results.get('spectrum')
    if isinstance(spectrum, dict) and isinstance(spectrum.get('levels'), list):
        raw_levels = spectrum.get('levels')
        spectrum_scope = 'complete_fixed_particle_spin_projection_sector'
    elif isinstance(solver_results.get('state_energies'), list):
        raw_levels = solver_results.get('state_energies')
        spectrum_scope = 'targeted_roots_fixed_particle_spin_projection_sector'
    else:
        raw_levels = []
        spectrum_scope = 'unavailable'

    energy_scale = mean_abs_t if mean_abs_t > 1e-12 else max(float(fallback_energy_scale), 1.0)
    numerical_tolerance = max(1e-10, 1e-8 * energy_scale)
    near_degeneracy_tolerance = max(100.0 * numerical_tolerance, 0.05 * energy_scale)
    summary = summarize_low_energy_manifold(
        raw_levels,
        spectrum_scope=spectrum_scope,
        numerical_tolerance=numerical_tolerance,
        near_degeneracy_tolerance=near_degeneracy_tolerance,
    )
    if summary.get('status') == 'available':
        summary['gap_over_t'] = (
            round(float(summary['gap']) / mean_abs_t, 6)
            if mean_abs_t > 1e-12
            else None
        )
    return summary


def _model_parameter_diagnostic_summary(spec: Dict[str, Any]) -> Dict[str, Any]:
    sites = spec.get('sites') if isinstance(spec.get('sites'), list) else []
    bonds = spec.get('bonds') if isinstance(spec.get('bonds'), list) else []
    u_values = [float(site.get('U', 0.0)) for site in sites if isinstance(site, dict)]
    t_values = [float(bond.get('effective_t', bond.get('t', 0.0))) for bond in bonds if isinstance(bond, dict)]
    v_values = [float(bond.get('effective_V', bond.get('V', 0.0))) for bond in bonds if isinstance(bond, dict)]
    mean_abs_u = _mean_abs_nonzero(u_values)
    mean_abs_t = _mean_abs_nonzero(t_values)
    mean_abs_v = _mean_abs_nonzero(v_values)
    site_count = len(sites)
    total_electrons = 0
    if isinstance(spec.get('nelec'), (list, tuple)) and len(spec.get('nelec')) == 2:
        total_electrons = int(spec['nelec'][0]) + int(spec['nelec'][1])
    filling = float(total_electrons / site_count) if site_count else None
    return {
        'mean_abs_u': mean_abs_u,
        'mean_abs_t': mean_abs_t,
        'mean_abs_v': mean_abs_v,
        'u_over_t': None if mean_abs_t <= 1e-12 else mean_abs_u / mean_abs_t,
        'v_over_t': None if mean_abs_t <= 1e-12 else mean_abs_v / mean_abs_t,
        'site_count': site_count,
        'total_electrons': total_electrons,
        'filling': filling,
        'half_filling_deviation': None if filling is None else abs(filling - 1.0),
    }


def _nearest_neighbor_values(matrix: Any, spec: Dict[str, Any]) -> List[float]:
    if not isinstance(matrix, list):
        return []
    sites = spec.get('sites') if isinstance(spec.get('sites'), list) else []
    site_to_index = {int(site['id']): index for index, site in enumerate(sorted(sites, key=lambda item: int(item['id']))) if isinstance(site, dict) and 'id' in site}
    values = []
    for bond in spec.get('bonds', []):
        if not isinstance(bond, dict) or 'source' not in bond or 'target' not in bond:
            continue
        source = site_to_index.get(int(bond['source']))
        target = site_to_index.get(int(bond['target']))
        if source is None or target is None:
            continue
        try:
            values.append(float(matrix[source][target]))
        except (TypeError, ValueError, IndexError):
            continue
    return values


def _strong_correlation_method_recommendation(
    level: str,
    spec: Dict[str, Any],
    solver_results: Dict[str, Any],
) -> Dict[str, List[str]]:
    """Recommend registered model solvers without assuming every cluster is FCI-sized."""

    solver_name = str(solver_results.get('solver') or '').strip().lower()
    site_count = len(spec.get('sites') or [])
    nelec = spec.get('nelec')
    try:
        determinant_dimension = fci_determinant_space_dimension(site_count, nelec)
    except (TypeError, ValueError):
        determinant_dimension = None
    exact_diagonalization_feasible = (
        determinant_dimension is not None
        and determinant_dimension <= FULL_DIAGONALIZATION_MAX_DIMENSION
    )

    if level == 'unknown':
        return {'preferred': [], 'usable_for_trends': [], 'risky': []}

    if solver_name == 'dmet':
        usable = ['block2_dmrg']
        if exact_diagonalization_feasible:
            usable.append('fci')
        risky = [] if level == 'weak' else ['mp2']
        if level == 'strong':
            risky.extend(['ccsd', 'ccsd_t'])
        return {
            'preferred': ['dmet'],
            'usable_for_trends': usable,
            'risky': risky,
        }

    if exact_diagonalization_feasible:
        if level == 'strong':
            return {
                'preferred': ['fci', 'block2_dmrg'],
                'usable_for_trends': ['ccsd'],
                'risky': ['mp2', 'ccsd_t'],
            }
        if level == 'moderate':
            return {
                'preferred': ['fci', 'block2_dmrg'],
                'usable_for_trends': ['ccsd', 'mp2'],
                'risky': ['ccsd_t'],
            }
        return {
            'preferred': ['fci', 'ccsd'],
            'usable_for_trends': ['mp2', 'block2_dmrg'],
            'risky': [],
        }

    if level == 'strong':
        return {
            'preferred': ['block2_dmrg', 'dmet'],
            'usable_for_trends': ['ccsd'],
            'risky': ['mp2', 'ccsd_t', 'fci'],
        }
    if level == 'moderate':
        return {
            'preferred': ['block2_dmrg', 'dmet'],
            'usable_for_trends': ['ccsd', 'mp2'],
            'risky': ['ccsd_t', 'fci'],
        }
    return {
        'preferred': ['ccsd', 'dmet'],
        'usable_for_trends': ['mp2', 'block2_dmrg'],
        'risky': ['fci'],
    }


def _local_double_occupancy_summary(solver_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare matched local RDMs, never a uniform/unpolarized filling guess."""
    local = solver_results.get('dmet_local_observables') or {}
    natural = solver_results.get('natural_occupation_summary') or {}
    spin = natural.get('spin_resolved') or {}
    values = local.get('double_occupancy') if local else solver_results.get('double_occupancy')
    reference = local.get('double_occupancy_reference')
    source = local.get('double_occupancy_reference_source')
    if not local and spin.get('status') == 'available':
        channels = spin['channels']
        alpha = channels['alpha']['site_populations']
        beta = channels['beta']['site_populations']
        reference = [a * b for a, b in zip(alpha, beta)]
        source = natural.get('source')
    payload = {
        'status': 'unavailable', 'suppression': None,
        'baseline': 'site_resolved_n_alpha_times_n_beta',
        'source': source, 'coverage': local.get('coverage'),
        'scope': 'covered_fragment_sites' if local else 'full_system',
    }
    if not isinstance(values, list) or not isinstance(reference, list) or len(values) != len(reference):
        payload['reason'] = 'Matched site-resolved double occupancy and spin-density baseline are required.'
        return payload
    pairs = []
    covered_values = []
    excluded = 0
    for value, baseline in zip(values, reference):
        if value is None or baseline is None:
            continue
        value, baseline = float(value), float(baseline)
        if not all(math.isfinite(v) and -1e-6 <= v <= 1.0 + 1e-6 for v in (value, baseline)):
            payload.update(status='failed', reason='Local double occupancy or its baseline is nonphysical.')
            return payload
        covered_values.append(value)
        if baseline <= 1e-12:
            if value > 1e-6:
                payload.update(status='failed', reason='Nonzero double occupancy conflicts with a zero opposite-spin population.')
                return payload
            excluded += 1
            continue
        pairs.append((value, baseline))
    payload.update(evaluated_site_count=len(pairs), zero_reference_site_count=excluded)
    payload['mean_double_occupancy'] = (
        sum(covered_values) / len(covered_values) if covered_values else None
    )
    if not pairs:
        payload['reason'] = 'No covered sites have a nonzero opposite-spin baseline; suppression is undefined.'
        return payload
    mean_double = sum(value for value, _ in pairs) / len(pairs)
    mean_reference = sum(value for _, value in pairs) / len(pairs)
    payload.update(
        status='available', scored_site_mean_double_occupancy=mean_double,
        reference_uncorrelated_estimate=mean_reference,
        connected_opposite_spin_density=mean_double - mean_reference,
        suppression=_clip_unit(1.0 - mean_double / mean_reference),
    )
    return payload


def _compute_strong_correlation_diagnostics(spec: Dict[str, Any], solver_results: Dict[str, Any]) -> Dict[str, Any]:
    parameter_summary = _model_parameter_diagnostic_summary(spec)
    dmet_local = (
        solver_results.get('dmet_local_observables')
        if isinstance(solver_results.get('dmet_local_observables'), dict)
        else {}
    )
    local_coverage = (
        dmet_local.get('coverage')
        if isinstance(dmet_local.get('coverage'), dict)
        else {}
    )
    internal_quantities = {
        'gap': False,
        'mean_field_gap': isinstance(solver_results.get('mean_field_homo_lumo'), dict),
        'frontier_orbital_degeneracy': (
            isinstance(solver_results.get('mean_field_homo_lumo'), dict)
            and isinstance(solver_results.get('mean_field_homo_lumo', {}).get('frontier_degeneracy'), dict)
        ),
        'low_energy_manifold': False,
        'natural_occupations': (
            isinstance(solver_results.get('natural_occupation_summary'), dict)
            and solver_results['natural_occupation_summary'].get('status', 'available') == 'available'
        ),
        'max_double_excitation_amplitude': isinstance(solver_results.get('double_excitation_amplitude_summary'), dict),
        'density': isinstance(solver_results.get('density'), list),
        'local_magnetization': isinstance(solver_results.get('local_magnetization'), list),
        'double_occupancy': isinstance(solver_results.get('double_occupancy'), list) or any(
            value is not None for value in dmet_local.get('double_occupancy', [])
        ),
        'spin_correlation': isinstance(solver_results.get('spin_correlation'), list),
        'charge_correlation': isinstance(solver_results.get('charge_correlation'), list),
        'sublattice_order_parameters': isinstance(dmet_local.get('sublattice_order'), dict),
        'nearest_neighbor_one_body_coherence': (
            dmet_local.get('mean_nearest_neighbor_one_body_coherence') is not None
        ),
    }
    diagnostics = []
    weighted_scores = []
    stress_scores = []

    u_over_t = parameter_summary['u_over_t']
    mean_abs_u = parameter_summary['mean_abs_u']
    mean_abs_t = parameter_summary['mean_abs_t']
    if u_over_t is None:
        if mean_abs_u > 1e-12 and mean_abs_t <= 1e-12:
            u_value = 'atomic_limit'
            u_text = 'Finite U with vanishing hopping defines the atomic-limit interaction scale; the state still needs density evidence.'
        else:
            u_value = None
            u_text = 'U/t could not be evaluated because no nonzero hopping scale was found.'
    else:
        u_value = round(float(u_over_t), 6)
        u_text = 'The interaction-to-hopping magnitude is parameter context, not a state-correlation measurement; attractive and repulsive interactions differ.'
    diagnostics.append(_diagnostic_item('u_over_t', u_value, None, u_text))

    v_over_t = parameter_summary.get('v_over_t')
    if v_over_t is not None and parameter_summary.get('mean_abs_v', 0.0) > 1e-12:
        diagnostics.append(_diagnostic_item(
            'v_over_t',
            round(float(v_over_t), 6),
            None,
            'Large V/t enhances intersite charge correlations; it is an interaction-scale indicator, not a finite-size phase boundary.',
        ))

    correlation_energy = solver_results.get('correlation_energy')
    reference_energy = solver_results.get('reference_energy')
    solver_name = str(solver_results.get('solver') or '').lower()
    if correlation_energy is not None and solver_name in ('mp2', 'ccsd', 'ccsd_t'):
        try:
            corr_abs = abs(float(correlation_energy))
            ref_abs = abs(float(reference_energy)) if reference_energy is not None else 0.0
            if not math.isfinite(corr_abs) or not math.isfinite(ref_abs):
                raise ValueError('Nonfinite correlation or reference energy')
            electron_scale = max(float(parameter_summary.get('total_electrons') or 0.0), 1.0)
            hopping_scale = mean_abs_t * electron_scale
            energy_scale = max(ref_abs, hopping_scale, 1.0)
            corr_fraction = corr_abs / energy_scale
            corr_score = _clip_unit(corr_fraction / 0.25)
            diagnostics.append(_diagnostic_item(
                'perturbative_correlation_stress',
                {
                    'correlation_energy': float(correlation_energy),
                    'reference_energy': None if reference_energy is None else float(reference_energy),
                    'correlation_fraction': round(float(corr_fraction), 6),
                    'energy_scale': round(float(energy_scale), 6),
                    'solver': solver_results.get('solver'),
                },
                corr_score,
                'Large low-order correlation energy relative to the reference scale is a screening warning for single-reference breakdown.',
            ))
            stress_scores.append(corr_score)
        except (TypeError, ValueError):
            stress_scores.append(1.0)
            diagnostics.append(_diagnostic_item(
                'perturbative_correlation_stress',
                None,
                1.0,
                'Correlation-energy stress could not be evaluated from the solver output.',
                severity='strong',
            ))
    elif correlation_energy is not None:
        diagnostics.append(_diagnostic_item(
            'reference_correlation_energy',
            {
                'correlation_energy': float(correlation_energy),
                'reference_energy': None if reference_energy is None else float(reference_energy),
                'correlation_reference': solver_results.get('correlation_reference') or solver_results.get('reference'),
                'reference_converged': solver_results.get('reference_converged'),
                'reference_initial_guess': solver_results.get('reference_initial_guess'),
                'solver': solver_results.get('solver'),
            },
            None,
            'Correlation energy is reported relative to the mean-field reference; it is not used as a perturbative-stress score for this solver.',
            severity='inconclusive',
        ))
    else:
        diagnostics.append(_diagnostic_item(
            'perturbative_correlation_stress',
            None,
            None,
            'Correlation-energy stress requires a low-cost correlated screening solver such as MP2 or CCSD.',
            severity='inconclusive',
        ))

    solver_convergence_failed = solver_results.get('converged') is False
    if solver_results.get('reference_converged') is False:
        stress_scores.append(1.0)
        diagnostics.append(_diagnostic_item(
            'reference_convergence', {'converged': False}, 1.0,
            'The reference SCF did not converge; this is solver stress, not evidence of strong physical correlation.',
        ))
    if solver_convergence_failed:
        diagnostics.append(_diagnostic_item(
            'solver_convergence',
            {'solver': solver_results.get('solver'), 'converged': False},
            1.0,
            'Failure of the selected solver to converge is a strong warning that a more robust method is needed.',
            severity='strong',
        ))
    elif 'converged' in solver_results:
        diagnostics.append(_diagnostic_item(
            'solver_convergence',
            {'solver': solver_results.get('solver'), 'converged': bool(solver_results.get('converged'))},
            0.0,
            'The selected solver reported convergence; this is useful bookkeeping but not proof of weak correlation.',
            severity='weak',
        ))

    mean_field_homo_lumo = solver_results.get('mean_field_homo_lumo')
    if isinstance(mean_field_homo_lumo, dict) and mean_field_homo_lumo.get('gap') is not None:
        try:
            mf_gap = float(mean_field_homo_lumo['gap'])
            positive_gap = max(0.0, mf_gap)
            if mean_abs_t > 1e-12:
                mf_gap_ratio = positive_gap / mean_abs_t
                mf_gap_over_t = round(float(mf_gap_ratio), 6)
            else:
                mf_gap_over_t = None
            diagnostics.append(_diagnostic_item(
                'mean_field_homo_lumo_gap',
                {
                    'homo': mean_field_homo_lumo.get('homo'),
                    'lumo': mean_field_homo_lumo.get('lumo'),
                    'gap': mf_gap,
                    'gap_over_t': mf_gap_over_t,
                    'min_spin_channel_gap': mean_field_homo_lumo.get('min_spin_channel_gap'),
                },
                None,
                'A small or closed mean-field HOMO-LUMO gap indicates near-degeneracy and weakens single-reference screening.',
            ))
        except (TypeError, ValueError):
            diagnostics.append(_diagnostic_item(
                'mean_field_homo_lumo_gap',
                None,
                None,
                'Mean-field HOMO-LUMO gap could not be evaluated from the reference orbitals.',
                severity='inconclusive',
            ))
    else:
        diagnostics.append(_diagnostic_item(
            'mean_field_homo_lumo_gap',
            None,
            None,
            'Mean-field HOMO-LUMO gap requires an MP2/CCSD-style reference calculation.',
            severity='inconclusive',
        ))

    frontier_degeneracy = (
        mean_field_homo_lumo.get('frontier_degeneracy')
        if isinstance(mean_field_homo_lumo, dict)
        else None
    )
    if isinstance(frontier_degeneracy, dict) and frontier_degeneracy.get('status') == 'available':
        try:
            alpha_beta = frontier_degeneracy.get('alpha_beta') if isinstance(frontier_degeneracy.get('alpha_beta'), dict) else {}
            alpha_beta_pairs = alpha_beta.get('pairs') if isinstance(alpha_beta.get('pairs'), list) else []
            alpha_beta_degenerate_pairs = [
                str(item.get('pair'))
                for item in alpha_beta_pairs
                if isinstance(item, dict) and item.get('degenerate') and item.get('pair')
            ]
            alpha_beta_spacings = [
                float(item.get('energy_difference'))
                for item in alpha_beta_pairs
                if isinstance(item, dict) and item.get('energy_difference') is not None
            ]
            same_spin_clusters = frontier_degeneracy.get('same_spin_degenerate_clusters')
            if not isinstance(same_spin_clusters, list):
                same_spin_clusters = []
            diagnostics.append(_diagnostic_item(
                'frontier_orbital_degeneracy',
                {
                    'tolerance': frontier_degeneracy.get('tolerance'),
                    'max_same_spin_degeneracy': frontier_degeneracy.get('max_same_spin_degeneracy'),
                    'same_spin_cluster_count': len(same_spin_clusters),
                    'alpha_beta_degenerate_pairs': alpha_beta_degenerate_pairs,
                    'minimum_alpha_beta_frontier_spacing': min(alpha_beta_spacings) if alpha_beta_spacings else None,
                },
                None,
                'Near-degenerate HOMO/LUMO frontier clusters or alpha/beta frontier pairs indicate a fragile single-reference frontier manifold.',
            ))
        except (TypeError, ValueError):
            diagnostics.append(_diagnostic_item(
                'frontier_orbital_degeneracy',
                None,
                None,
                'Frontier-orbital degeneracy could not be evaluated from the reference orbitals.',
                severity='inconclusive',
            ))
    else:
        diagnostics.append(_diagnostic_item(
            'frontier_orbital_degeneracy',
            None,
            None,
            'Frontier-orbital degeneracy requires an MP2/CCSD-style reference calculation.',
            severity='inconclusive',
        ))

    natural_summary = solver_results.get('natural_occupation_summary')
    spin_summary = natural_summary.get('spin_resolved', {}) if isinstance(natural_summary, dict) else {}
    invalid_density = (
        isinstance(natural_summary, dict)
        and (natural_summary.get('status') in ('failed', 'out_of_bounds')
             or spin_summary.get('status') in ('failed', 'out_of_bounds'))
    )
    if invalid_density:
        stress_scores.append(1.0)
        diagnostics.append(_diagnostic_item(
            'correlated_density_quality', spin_summary or natural_summary, 1.0,
            'The correlated density fails physical checks; its occupation score is excluded.',
        ))
    if isinstance(natural_summary, dict) and natural_summary.get('status', 'available') != 'available':
        diagnostics.append(_diagnostic_item(
            'natural_orbital_occupations', natural_summary, None,
            natural_summary.get('reason') or 'Natural occupations are unavailable.',
            severity='inconclusive',
        ))
    elif isinstance(natural_summary, dict) and isinstance(natural_summary.get('occupations'), list):
        try:
            fractionality = float(natural_summary.get('average_fractionality') or natural_summary.get('max_fractionality') or 0.0)
            natural_score = (
                _clip_unit(spin_summary['score'])
                if not invalid_density and spin_summary.get('status') == 'available'
                and natural_summary.get('scope') != 'assembled_dmet_lattice' else None
            )
            diagnostics.append(_diagnostic_item(
                'natural_orbital_occupations',
                {
                    'occupations': [round(float(value), 6) for value in natural_summary.get('occupations', [])],
                    'frontier_occupations': [round(float(value), 6) for value in natural_summary.get('frontier_occupations', [])],
                    'average_fractionality': round(float(fractionality), 6),
                    'max_fractionality': round(float(natural_summary.get('max_fractionality') or 0.0), 6),
                    'fractional_orbital_count': int(natural_summary.get('fractional_orbital_count') or 0),
                    'spin_resolved': spin_summary or None,
                    'source': natural_summary.get('source'),
                    'scope': natural_summary.get('scope'),
                    'scoring_scope': (
                        'informational_assembled_density' if natural_summary.get('scope') == 'assembled_dmet_lattice'
                        else 'spin_resolved_correlated_density'
                    ),
                },
                natural_score,
                'Raw spatial occupations are retained. Correlation scoring uses departures from 0/1 in each spin channel; spatial SOMOs and broken-symmetry determinants alone do not contribute. A mixed or assembled density needs its own provenance.',
            ))
            if natural_score is not None:
                weighted_scores.append((0.16, natural_score))
        except (TypeError, ValueError):
            diagnostics.append(_diagnostic_item(
                'natural_orbital_occupations',
                None,
                None,
                'Natural occupations could not be evaluated from the correlated one-particle density matrix.',
                severity='inconclusive',
            ))
    else:
        diagnostics.append(_diagnostic_item(
            'natural_orbital_occupations',
            None,
            None,
            'Natural-occupation diagnostic requires FCI or a correlated solver with an available one-particle density matrix.',
            severity='inconclusive',
        ))

    amplitude_summary = solver_results.get('double_excitation_amplitude_summary')
    if isinstance(amplitude_summary, dict) and amplitude_summary.get('max_abs_t2') is not None:
        try:
            max_abs_t2 = float(amplitude_summary['max_abs_t2'])
            if not math.isfinite(max_abs_t2):
                raise ValueError('Nonfinite double-excitation amplitude')
            t2_score = _clip_unit(max_abs_t2 / 0.5)
            diagnostics.append(_diagnostic_item(
                'max_double_excitation_amplitude',
                {
                    'max_abs_t2': max_abs_t2,
                    'rms_t2': amplitude_summary.get('rms_t2'),
                    'large_amplitude_count_0_10': amplitude_summary.get('large_amplitude_count_0_10'),
                    'large_amplitude_count_0_20': amplitude_summary.get('large_amplitude_count_0_20'),
                },
                t2_score,
                'Large double-excitation amplitudes indicate that the low-order single-reference expansion is under stress.',
            ))
            stress_scores.append(t2_score)
        except (TypeError, ValueError):
            stress_scores.append(1.0)
            diagnostics.append(_diagnostic_item(
                'max_double_excitation_amplitude',
                None,
                1.0,
                'Double-excitation amplitudes could not be evaluated from the solver output.',
                severity='strong',
            ))
    else:
        diagnostics.append(_diagnostic_item(
            'max_double_excitation_amplitude',
            None,
            None,
            'Double-excitation amplitude diagnostic requires MP2, CCSD, or CCSD(T).',
            severity='inconclusive',
        ))

    low_energy = _low_energy_manifold_summary(
        solver_results,
        mean_abs_t=mean_abs_t,
        fallback_energy_scale=max(mean_abs_u, parameter_summary.get('mean_abs_v') or 0.0),
    )
    if low_energy.get('status') == 'available':
        internal_quantities['gap'] = True
        internal_quantities['low_energy_manifold'] = True
        classification = low_energy.get('classification')
        if classification == 'numerically_degenerate':
            interpretation = (
                'The computed spectrum contains multiple numerically degenerate ground-state levels in the selected '
                'particle and spin-projection sector. A single-root expectation value is not a unique description; '
                'inspect root-resolved observables or define a symmetry-pure state or physically weighted ensemble.'
            )
        elif classification == 'near_degenerate':
            interpretation = (
                'Several low-energy states lie within the reported near-degeneracy window. They can become relevant '
                'under small perturbations or finite temperature, so report roots separately and use an ensemble only '
                'when its physical weights are defined.'
            )
        else:
            interpretation = (
                'The ground state is separated from the other computed levels within this particle and spin-projection '
                'sector; this does not exclude degeneracy with states in sectors that were not calculated.'
            )
        if low_energy.get('manifold_may_extend_beyond_computed_roots'):
            interpretation += ' Every requested root lies in the low-energy window, so increase nroots before treating the manifold as complete.'
        low_energy.pop('risk_score', None)
        diagnostics.append(_diagnostic_item('many_body_gap', low_energy, None, interpretation))
    else:
        diagnostics.append(_diagnostic_item(
            'many_body_gap',
            low_energy,
            None,
            'Ground-state degeneracy requires at least two computed roots in the same particle and spin-projection sector.',
            severity='inconclusive',
        ))

    double_payload = _local_double_occupancy_summary(solver_results)
    double_score = double_payload['suppression']
    if double_payload.get('connected_opposite_spin_density', 0.0) > 1e-6:
        # Pairing/enhancement is outside a repulsive-suppression metric; a
        # clipped zero must not dilute independent correlated-density evidence.
        double_score = None
        double_payload['scoring_status'] = 'enhancement_not_scored_as_suppression'
    diagnostics.append(_diagnostic_item(
        'double_occupancy_suppression', double_payload, double_score,
        'Double occupancy is compared with the product of local alpha and beta populations from the same density source and covered sites. Suppression measures repulsive local correlation; zero baselines and enhancement are unscored by this metric.',
    ))
    if double_score is not None:
        weighted_scores.append((0.26, double_score))
    if double_payload['status'] == 'failed':
        stress_scores.append(1.0)
        diagnostics.append(_diagnostic_item('local_density_quality', double_payload, 1.0, double_payload['reason']))

    spin_values = _nearest_neighbor_values(solver_results.get('spin_correlation'), spec)
    if spin_values:
        mean_spin = float(sum(spin_values) / len(spin_values))
        diagnostics.append(_diagnostic_item(
            'nearest_neighbor_spin_correlation',
            {
                'mean': mean_spin,
                'minimum': min(spin_values),
                'maximum': max(spin_values),
                'coverage': local_coverage or None,
            },
            None,
            'Negative nearest-neighbor spin correlation indicates antiferromagnetic local-moment tendency.',
        ))
    else:
        diagnostics.append(_diagnostic_item('nearest_neighbor_spin_correlation', None, None, 'Spin correlation requires bonded sites covered by an available local two-particle density matrix.', severity='inconclusive'))

    charge_values = _nearest_neighbor_values(solver_results.get('charge_correlation'), spec)
    if charge_values:
        mean_abs_charge = float(sum(abs(value) for value in charge_values) / len(charge_values))
        diagnostics.append(_diagnostic_item(
            'nearest_neighbor_charge_correlation',
            {
                'mean_abs': mean_abs_charge,
                'mean': float(sum(charge_values) / len(charge_values)),
                'coverage': local_coverage or None,
            },
            None,
            'Connected charge correlations measure correlated density fluctuations; charge-order interpretation requires a clear spatial pattern, parameter trend, or structure-factor analysis.',
        ))
    else:
        diagnostics.append(_diagnostic_item('nearest_neighbor_charge_correlation', None, None, 'Charge correlation requires bonded sites covered by an available local two-particle density matrix.', severity='inconclusive'))

    sublattice_order = dmet_local.get('sublattice_order')
    if isinstance(sublattice_order, dict):
        charge_imbalance = abs(float(sublattice_order.get('charge_imbalance') or 0.0))
        staggered_magnetization = abs(float(sublattice_order.get('staggered_magnetization') or 0.0))
        diagnostics.append(_diagnostic_item(
            'sublattice_order_parameters',
            {
                'charge_imbalance': charge_imbalance,
                'staggered_magnetization': staggered_magnetization,
                'groups': sublattice_order.get('groups'),
                'coverage': local_coverage or None,
            },
            None,
            'Sublattice charge imbalance and staggered magnetization describe broken-symmetry order in the selected DMET solution; compare seeds, sizes, and competing solutions before assigning a phase.',
        ))

    one_body_coherence = dmet_local.get('mean_nearest_neighbor_one_body_coherence')
    if one_body_coherence is not None:
        diagnostics.append(_diagnostic_item(
            'nearest_neighbor_one_body_coherence',
            {'mean': float(one_body_coherence), 'coverage': local_coverage or None},
            None,
            'Nearest-neighbor one-particle coherence measures delocalization in the assembled DMET 1RDM and should be interpreted through parameter trends rather than as a standalone correlation score.',
            severity='inconclusive',
        ))

    dmet_configuration = (solver_results.get('dmet_result') or {}).get('configuration') or {}
    impurity_beta = (dmet_configuration.get('impurity_solver_options') or {}).get('beta')
    if impurity_beta is not None:
        diagnostics.append(_diagnostic_item(
            'impurity_scf_smearing', {'beta': impurity_beta, 'scope': 'preliminary_impurity_scf'}, None,
            'Beta controls preliminary impurity SCF smearing. Correlation diagnostics use the correlated density, not smeared SCF occupations; this is not finite-temperature CCSD.',
        ))

    physics_names = {'natural_orbital_occupations', 'double_occupancy_suppression'}
    stress_names = {
        'perturbative_correlation_stress', 'max_double_excitation_amplitude',
        'solver_convergence', 'reference_convergence', 'correlated_density_quality', 'local_density_quality',
    }
    evidence_groups = {'physics': [], 'solver_stress': [], 'context': []}
    for item in diagnostics:
        category = 'physics' if item['name'] in physics_names else 'solver_stress' if item['name'] in stress_names else 'context'
        item['category'] = category
        if category == 'context':
            # Preserve the observable, but do not present magnetism, exchange,
            # small gaps, or interaction scales as measured correlation scores.
            item['score'] = None
            item['severity'] = 'informational' if item['value'] is not None else 'inconclusive'
        evidence_groups[category].append(item['name'])

    if weighted_scores:
        weight_total = sum(weight for weight, _score in weighted_scores)
        score = sum(weight * score for weight, score in weighted_scores) / weight_total
        confidence = 'medium' if len(weighted_scores) >= 2 and not dmet_local else 'low'
    else:
        score = None
        confidence = 'none'
    physics_level = _diagnostic_level(score) if score is not None else 'unknown'
    if 'converged' in solver_results:
        stress_scores.append(1.0 if solver_convergence_failed else 0.0)
    solver_stress_score = max(stress_scores) if stress_scores else None
    solver_stress_level = _diagnostic_level(solver_stress_score) if solver_stress_score is not None else 'unknown'
    # Overall level remains a warning for existing clients. A weak/unknown
    # solver warning cannot establish weak physical correlation when unmeasured.
    level = physics_level
    level_reason = 'physics_evidence' if score is not None else 'insufficient_physics_evidence'
    if solver_stress_score is not None and solver_stress_score >= 0.35 and (score is None or solver_stress_score > score):
        level = solver_stress_level
        level_reason = 'solver_nonconvergence' if solver_convergence_failed else 'solver_stress'
    summary = 'Hubbard correlation evidence: {0}; solver stress: {1}; overall warning: {2} (confidence={3}).'.format(
        physics_level, solver_stress_level, level, confidence,
    )
    return {
        'kind': 'strong_correlation_diagnostics',
        'scope': 'finite_size_model_hamiltonian',
        'level': level,
        'level_reason': level_reason,
        'score': None if score is None else round(float(score), 6),
        'physics_score': None if score is None else round(float(score), 6),
        'physics_level': physics_level,
        'solver_stress_score': solver_stress_score,
        'solver_stress_level': solver_stress_level,
        'confidence': confidence,
        'summary': summary,
        'internal_quantities': internal_quantities,
        'parameter_summary': parameter_summary,
        'diagnostics': diagnostics,
        'evidence_groups': evidence_groups,
        'method_recommendation': _strong_correlation_method_recommendation(
            level,
            spec,
            solver_results,
        ),
        'limitations': [
            'Heuristic thresholds are intended for finite-size Hubbard-style clusters, not thermodynamic-limit phase boundaries.',
            'The overall level is a warning combining physical evidence and solver stress, not a physical correlation label. Method lists are suggestions, not automatic solver changes.',
            'Missing spin or local-density evidence remains unknown. Confidence describes evidence availability and scope, not a probability or energy-error bound.',
            'Spatial fractionality, magnetization, exchange correlations and degeneracy alone do not establish interaction-driven strong correlation.',
            'Spin-density nonidempotency can also reflect state mixing or embedding assembly; assembled DMET densities and covered fragment pairs are not full-system exact RDMs.',
            'Low-energy degeneracy is assessed only within the calculated particle and spin-projection sector; compare other sectors when symmetry permits.',
            'For robust phase diagrams, compare trends across size, boundary condition, filling, and solver.',
        ],
    }
