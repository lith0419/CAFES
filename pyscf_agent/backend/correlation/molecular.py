from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .scoring import correlation_strength_level as _risk_level

from .low_energy import summarize_low_energy_manifold
from .orbital_diagnostics import (
    _ao_natural_orbitals,
    natural_orbital_summary as _natural_orbital_summary,
    natural_orbital_summary_from_occupations as _natural_orbital_summary_from_occupations,
    occupation_fractionality as _occupation_fractionality,
    t2_orbital_importance as _t2_orbital_importance,
)
from ...contracts import OrbitalProcessingSpec
from ..script_helpers import _extract_homo_lumo, _safe_to_list


FRONTIER_DEGENERACY_TOLERANCE = 0.02
FRONTIER_OCCUPATION_EPS = 1.0e-8
SPIN_SYMMETRY_BREAKING_TOLERANCE = 0.2
CCSD_T1_WARNING_THRESHOLD_RESTRICTED = 0.02
CCSD_T1_WARNING_THRESHOLD_OPEN_SHELL = 0.04
CCSD_T1_PROMOTION_THRESHOLD = 0.05
CCSD_D1_WARNING_THRESHOLD = 0.05
CCSD_D1_PROMOTION_THRESHOLD = 0.10
CORRELATED_NATURAL_OCCUPATION_MODERATE_THRESHOLD = 0.50
CORRELATED_NATURAL_OCCUPATION_STRONG_THRESHOLD = 0.80


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean_score(weighted_scores: Sequence[Tuple[float, float]]) -> float:
    valid = [
        (float(weight), _clip_unit(float(score)))
        for weight, score in weighted_scores
        if weight is not None and score is not None
    ]
    if not valid:
        return 0.0
    total_weight = sum(weight for weight, _score in valid)
    if total_weight <= 0:
        return 0.0
    return sum(weight * score for weight, score in valid) / total_weight


def _flatten_restricted_values(values: Any) -> List[float]:
    if values is None:
        return []
    if hasattr(values, 'tolist'):
        values = values.tolist()
    if isinstance(values, (list, tuple)) and values and isinstance(values[0], (list, tuple)):
        # UHF occupations are per spin orbital (0 or 1).  Diagnostics use
        # spatial-orbital occupations (0 to 2), so combine alpha and beta
        # before applying the fractional-occupation window.
        alpha = values[0]
        beta = values[1] if len(values) > 1 and isinstance(values[1], (list, tuple)) else []
        values = [
            float(alpha[index]) + (float(beta[index]) if index < len(beta) else 0.0)
            for index in range(len(alpha))
        ]
    if not isinstance(values, (list, tuple)):
        return []
    normalized = []
    for item in values:
        converted = _as_float(item)
        if converted is not None:
            normalized.append(converted)
    return normalized


def _has_spin_resolved_occupations(mf: Any) -> bool:
    occupations = getattr(mf, 'mo_occ', None)
    if hasattr(occupations, 'tolist'):
        occupations = occupations.tolist()
    return bool(
        isinstance(occupations, (list, tuple))
        and len(occupations) >= 2
        and isinstance(occupations[0], (list, tuple))
        and isinstance(occupations[1], (list, tuple))
    )


def _reference_spin_symmetry_summary(
    mf: Any,
    mol: Any,
    *,
    spin_square: Optional[float] = None,
) -> Dict[str, Any]:
    try:
        spin_value = int(getattr(mol, 'spin', 0) or 0)
    except (TypeError, ValueError):
        spin_value = 0
    target_s = abs(float(spin_value)) / 2.0
    target_s2 = target_s * (target_s + 1.0)
    if spin_square is None:
        try:
            spin_square_value = mf.spin_square()
            spin_square = _as_float(spin_square_value[0] if isinstance(spin_square_value, tuple) else spin_square_value)
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            spin_square = None
    deviation = None if spin_square is None else abs(float(spin_square) - target_s2)
    spin_resolved = _has_spin_resolved_occupations(mf)
    broken_symmetry = bool(
        spin_resolved
        and deviation is not None
        and deviation > SPIN_SYMMETRY_BREAKING_TOLERANCE
    )
    return {
        'spin_resolved_reference': spin_resolved,
        'target_spin_square': target_s2,
        'spin_square': spin_square,
        'spin_square_deviation': deviation,
        'spin_contaminated': bool(deviation is not None and deviation > SPIN_SYMMETRY_BREAKING_TOLERANCE),
        'broken_symmetry': broken_symmetry,
        # A singlet UHF determinant may become broken-symmetry at dissociation;
        # total molecular spin alone must not suppress its frontier diagnostic.
        'alpha_beta_relevant': bool(spin_value or broken_symmetry),
    }


def _spin_summed_uno_occupations(mf: Any, mol: Any) -> Dict[str, Any]:
    """Build UNO occupations from a UHF spin-summed AO density when possible."""
    summary: Dict[str, Any] = {'status': 'unavailable', 'occupations': []}
    if not _has_spin_resolved_occupations(mf) or not hasattr(mf, 'make_rdm1'):
        summary['reason'] = 'spin-resolved mean-field density matrix is unavailable'
        return summary
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel

        dm1 = mf.make_rdm1()
        if isinstance(dm1, (tuple, list)) and len(dm1) >= 2:
            density = np.asarray(dm1[0], dtype=float) + np.asarray(dm1[1], dtype=float)
        else:
            values = np.asarray(dm1, dtype=float)
            if values.ndim != 3 or values.shape[0] < 2:
                summary['reason'] = 'mean-field density matrix does not contain alpha and beta channels'
                return summary
            density = values[0] + values[1]
        occupations, _coefficients, _overlap = _ao_natural_orbitals(density, mf)
        summary.update({
            'status': 'available',
            'source': 'uhf_spin_summed_ao_density_uno',
            'occupations': [float(value) for value in sorted(occupations, reverse=True)],
        })
        return summary
    except Exception as exc:  # pragma: no cover - diagnostics must not invalidate a calculation
        summary['reason'] = str(exc)
        return summary


def _mean_field_occupation_summary(mf: Any, mol: Any) -> Dict[str, Any]:
    if not _has_spin_resolved_occupations(mf):
        return {
            'status': 'available',
            'source': 'restricted_mo_occupations',
            'occupations': _flatten_restricted_values(getattr(mf, 'mo_occ', None)),
        }
    uno_summary = _spin_summed_uno_occupations(mf, mol)
    if uno_summary.get('status') == 'available':
        return uno_summary
    return uno_summary


def _orbital_table(mf: Any) -> List[Dict[str, Any]]:
    mo_energy = getattr(mf, 'mo_energy', None)
    mo_occ = getattr(mf, 'mo_occ', None)
    if hasattr(mo_energy, 'tolist'):
        mo_energy = mo_energy.tolist()
    if hasattr(mo_occ, 'tolist'):
        mo_occ = mo_occ.tolist()

    if isinstance(mo_energy, (list, tuple)) and mo_energy and isinstance(mo_energy[0], (list, tuple)):
        rows = []
        labels = ('alpha', 'beta')
        for spin_index, spin_energy in enumerate(mo_energy):
            spin_occ = mo_occ[spin_index] if isinstance(mo_occ, (list, tuple)) and len(mo_occ) > spin_index else []
            for index, energy in enumerate(spin_energy):
                occ = spin_occ[index] if isinstance(spin_occ, (list, tuple)) and len(spin_occ) > index else None
                rows.append({
                    'index': index,
                    'spin': labels[spin_index] if spin_index < len(labels) else str(spin_index),
                    'energy': _as_float(energy),
                    'occupation': _as_float(occ),
                })
        return rows

    rows = []
    energies = mo_energy if isinstance(mo_energy, (list, tuple)) else []
    occupations = mo_occ if isinstance(mo_occ, (list, tuple)) else []
    for index, energy in enumerate(energies):
        occ = occupations[index] if len(occupations) > index else None
        rows.append({
            'index': index,
            'spin': None,
            'energy': _as_float(energy),
            'occupation': _as_float(occ),
        })
    return rows


def _frontier_orbital_degeneracy_summary(
    mf: Any,
    *,
    tolerance: float = FRONTIER_DEGENERACY_TOLERANCE,
    evaluate_alpha_beta: bool = False,
) -> Dict[str, Any]:
    rows = [
        row for row in _orbital_table(mf)
        if row.get('energy') is not None and row.get('occupation') is not None
    ]
    if not rows:
        return {
            'status': 'unavailable',
            'reason': 'mean-field orbital energies or occupations are unavailable',
            'tolerance': tolerance,
        }

    channels: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        spin_label = row.get('spin') or 'restricted'
        channels.setdefault(spin_label, {'rows': []})['rows'].append(row)

    channel_summaries: Dict[str, Dict[str, Any]] = {}
    for label, channel in channels.items():
        channel_rows = channel['rows']
        occupied = sorted(
            (row for row in channel_rows if float(row['occupation']) > FRONTIER_OCCUPATION_EPS),
            key=lambda item: float(item['energy']),
        )
        virtual = sorted(
            (row for row in channel_rows if float(row['occupation']) <= FRONTIER_OCCUPATION_EPS),
            key=lambda item: float(item['energy']),
        )
        homo_row = occupied[-1] if occupied else None
        lumo_row = virtual[0] if virtual else None
        homo = float(homo_row['energy']) if homo_row is not None else None
        lumo = float(lumo_row['energy']) if lumo_row is not None else None
        homo_cluster = [
            row for row in occupied
            if homo is not None and abs(float(row['energy']) - homo) <= tolerance
        ]
        lumo_cluster = [
            row for row in virtual
            if lumo is not None and abs(float(row['energy']) - lumo) <= tolerance
        ]
        channel_summaries[label] = {
            'homo': homo,
            'lumo': lumo,
            'gap': None if homo is None or lumo is None else float(lumo - homo),
            'homo_degeneracy': len(homo_cluster) if homo is not None else 0,
            'lumo_degeneracy': len(lumo_cluster) if lumo is not None else 0,
            'homo_indices': [int(row['index']) for row in homo_cluster],
            'lumo_indices': [int(row['index']) for row in lumo_cluster],
        }

    if not channel_summaries:
        return {
            'status': 'unavailable',
            'reason': 'no complete spin channel frontier could be identified',
            'tolerance': tolerance,
        }

    same_spin_clusters = []
    for label, summary in channel_summaries.items():
        for frontier in ('homo', 'lumo'):
            count = int(summary.get('{0}_degeneracy'.format(frontier)) or 0)
            if count > 1:
                same_spin_clusters.append({
                    'channel': label,
                    'frontier': frontier,
                    'count': count,
                    'indices': summary.get('{0}_indices'.format(frontier), []),
                })
    max_same_spin_degeneracy = max(
        [
            int(summary.get('homo_degeneracy') or 0)
            for summary in channel_summaries.values()
        ] + [
            int(summary.get('lumo_degeneracy') or 0)
            for summary in channel_summaries.values()
        ] + [0]
    )
    same_spin_score = _clip_unit((max_same_spin_degeneracy - 1.0) / 2.0)

    alpha_beta_pairs = []
    alpha_beta_score = 0.0
    if evaluate_alpha_beta and 'alpha' in channel_summaries and 'beta' in channel_summaries:
        frontier_values = {
            'alpha_homo': channel_summaries['alpha'].get('homo'),
            'alpha_lumo': channel_summaries['alpha'].get('lumo'),
            'beta_homo': channel_summaries['beta'].get('homo'),
            'beta_lumo': channel_summaries['beta'].get('lumo'),
        }
        for left, right in (
            ('alpha_homo', 'beta_homo'),
            ('alpha_lumo', 'beta_lumo'),
            ('alpha_homo', 'beta_lumo'),
            ('alpha_lumo', 'beta_homo'),
        ):
            left_value = frontier_values.get(left)
            right_value = frontier_values.get(right)
            if left_value is None or right_value is None:
                continue
            spacing = abs(float(left_value) - float(right_value))
            pair_score = _clip_unit(1.0 - spacing / max(float(tolerance), 1.0e-12))
            alpha_beta_score = max(alpha_beta_score, pair_score)
            alpha_beta_pairs.append({
                'pair': '{0}-{1}'.format(left, right),
                'energy_difference': spacing,
                'degenerate': spacing <= tolerance,
            })

    score = max(same_spin_score, alpha_beta_score)
    return {
        'status': 'available',
        'tolerance': tolerance,
        'score': round(score, 6),
        'max_same_spin_degeneracy': max_same_spin_degeneracy,
        'same_spin_degenerate_clusters': same_spin_clusters,
        'alpha_beta': {
            'evaluated': bool(evaluate_alpha_beta and 'alpha' in channel_summaries and 'beta' in channel_summaries),
            'pairs': alpha_beta_pairs,
            'degenerate_pairs': [
                item for item in alpha_beta_pairs
                if item.get('degenerate')
            ],
            'score': round(alpha_beta_score, 6),
        },
        'spin_channels': channel_summaries,
    }


def _bool_or_none(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return bool(value)
    if value.__class__.__name__ == 'bool_':
        return bool(value)
    return None


def build_scf_stability_summary(mf: Any, *, task_spec: Any = None) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        'status': 'unavailable',
        'stable': None,
        'internal_stable': None,
        'external_stable': None,
        'analysis': 'scf_stability',
    }
    if task_spec is not None:
        configuration = task_spec.workflow.module_config.get('molecular.correlation_diagnostics', {})
        if not configuration.get('scf_stability', False):
            summary.update({
                'status': 'not_requested',
                'reason': 'SCF stability is optional; enable molecular.correlation_diagnostics.scf_stability in workflow.module_config to run it.',
            })
            return summary
    stability = getattr(mf, 'stability', None)
    if not callable(stability):
        summary['reason'] = 'SCF object does not expose a stability() method'
        return summary
    try:
        try:
            result = stability(internal=True, external=True, return_status=True, nroots=1)
        except TypeError:
            try:
                result = stability(return_status=True, nroots=1)
            except TypeError:
                result = stability(return_status=True)
        items = list(result) if isinstance(result, tuple) else [result]
        bools = [_bool_or_none(item) for item in items]
        bools = [item for item in bools if item is not None]
        if len(items) >= 4:
            summary['internal_stable'] = _bool_or_none(items[-2])
            summary['external_stable'] = _bool_or_none(items[-1])
        if summary['internal_stable'] is None and bools:
            summary['internal_stable'] = bools[0]
        if summary['external_stable'] is None and len(bools) >= 2:
            summary['external_stable'] = bools[1]
        known = [
            item for item in (summary['internal_stable'], summary['external_stable'])
            if item is not None
        ]
        summary['stable'] = all(known) if known else None
        summary['status'] = 'completed' if known else 'unknown'
        summary['orbital_update_available'] = any(
            item is not None and _bool_or_none(item) is None
            for item in items
        )
        return summary
    except Exception as exc:  # pragma: no cover - PySCF stability can fail for method-specific reasons
        summary['status'] = 'failed'
        summary['reason'] = str(exc)
        return summary


def _occupation_routing_summary(occupations: Sequence[float], spin: int) -> Dict[str, Any]:
    """Compare an open-shell spectrum with its spin-adapted determinant baseline.

    Raw 2/0 fractionality remains useful for selecting CAS orbitals, including
    SOMOs. Routing instead measures deviations from a sorted 2/1/0 spectrum
    with the same electron count and |Nalpha - Nbeta| singly occupied orbitals.
    No occupations are discarded. Paired inactive orbitals may be omitted,
    so this also applies to CAS-only and fractional mean-field spectra.
    """
    unpaired = abs(spin)
    if not all(math.isfinite(value) for value in occupations):
        return {'status': 'unavailable', 'score': None, 'reason': 'Non-finite occupations.'}
    if not unpaired:
        return {
            'status': 'available',
            'reference': 'closed_shell_2_0',
            'score': max((_occupation_fractionality(value) for value in occupations), default=0.0),
        }
    electron_count = int(round(sum(occupations)))
    paired_electrons = electron_count - unpaired
    doubly_occupied = paired_electrons // 2
    if paired_electrons < 0 or paired_electrons % 2 or doubly_occupied + unpaired > len(occupations):
        return {
            'status': 'unavailable', 'score': None,
            'reason': 'The occupation spectrum does not define a compatible open-shell electron/spin sector.',
        }
    reference = (
        [2.0] * doubly_occupied
        + [1.0] * unpaired
        + [0.0] * (len(occupations) - doubly_occupied - unpaired)
    )
    deviations = [abs(value - expected) for value, expected in zip(sorted(occupations, reverse=True), reference)]
    return {
        'status': 'available',
        'reference': 'spin_adapted_2_1_0',
        'reference_electron_count': electron_count,
        'singly_occupied_count': unpaired,
        'reference_occupations': reference,
        'max_deviation': max(deviations, default=0.0),
        'score': _clip_unit(max(deviations, default=0.0)),
    }


def _natural_occupation_metrics(natural_summary: Dict[str, Any], *, spin: int = 0) -> Dict[str, Any]:
    occupations = []
    out_of_bounds = []
    if isinstance(natural_summary, dict):
        for item in natural_summary.get('orbitals') or []:
            if not isinstance(item, dict):
                continue
            value = _as_float(item.get('occupation'))
            if value is not None:
                occupations.append(value)
                if value < -0.02 or value > 2.02:
                    out_of_bounds.append({
                        'natural_orbital_index': item.get('natural_orbital_index'),
                        'occupation': value,
                    })
    physical_occupations = occupations if not out_of_bounds else []
    fractionalities = [_occupation_fractionality(value) for value in physical_occupations]
    frontier = [value for value in physical_occupations if 0.02 < value < 1.98]
    return {
        'status': (
            'out_of_bounds'
            if out_of_bounds
            else natural_summary.get('status')
            if isinstance(natural_summary, dict)
            else 'unavailable'
        ),
        'source': natural_summary.get('source') if isinstance(natural_summary, dict) else None,
        'occupation_count': len(occupations),
        'occupation_sum': sum(occupations),
        'out_of_bounds': out_of_bounds,
        'frontier_occupations': frontier,
        'fractional_orbital_count': len(frontier),
        'max_fractionality': max(fractionalities, default=0.0),
        'average_frontier_fractionality': (
            sum(_occupation_fractionality(value) for value in frontier) / len(frontier)
            if frontier else 0.0
        ),
        'routing': (
            _occupation_routing_summary(physical_occupations, spin)
            if physical_occupations and not out_of_bounds and natural_summary.get('status') == 'available'
            else {'status': 'unavailable', 'score': None}
        ),
    }


def _mean_field_electron_count(mf: Any, mol: Any) -> Optional[float]:
    value = _as_float(getattr(mol, 'nelectron', None))
    if value is not None and value > 0:
        return value
    occupations = getattr(mf, 'mo_occ', None)
    if hasattr(occupations, 'tolist'):
        occupations = occupations.tolist()
    try:
        if isinstance(occupations, (list, tuple)) and occupations and isinstance(occupations[0], (list, tuple)):
            return sum(float(item) for channel in occupations for item in channel)
        if isinstance(occupations, (list, tuple)):
            return sum(float(item) for item in occupations)
    except (TypeError, ValueError):
        return None
    return None


def _ccsd_t1_d1_diagnostics(post_hf: Any, mf: Any, mol: Any) -> Dict[str, Any]:
    """Return CCSD T1/D1 diagnostics from restricted or unrestricted amplitudes.

    T1 is the norm of the single-excitation amplitudes normalized by the
    electron count. D1 is the largest singular value of a spin-channel T1
    matrix. These are routing diagnostics, not universal error estimates.
    """
    summary: Dict[str, Any] = {
        'status': 'unavailable',
        't1': None,
        'd1': None,
        'reference_type': None,
        'warning_thresholds': {
            't1_restricted': CCSD_T1_WARNING_THRESHOLD_RESTRICTED,
            't1_open_shell': CCSD_T1_WARNING_THRESHOLD_OPEN_SHELL,
            'd1': CCSD_D1_WARNING_THRESHOLD,
        },
        'promotion_thresholds': {
            't1': CCSD_T1_PROMOTION_THRESHOLD,
            'd1': CCSD_D1_PROMOTION_THRESHOLD,
        },
    }
    if post_hf is None:
        summary['reason'] = 'CCSD amplitudes are unavailable for the selected method'
        return summary
    amplitudes = getattr(post_hf, 't1', None)
    if amplitudes is None:
        summary['reason'] = 'selected post-HF method does not expose CCSD single-excitation amplitudes'
        return summary
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel

        components = list(amplitudes) if isinstance(amplitudes, tuple) else [amplitudes]
        matrices = [np.asarray(component, dtype=float) for component in components]
        matrices = [matrix for matrix in matrices if matrix.ndim == 2]
        if not matrices:
            summary['reason'] = 'CCSD single-excitation amplitudes are not two-dimensional matrices'
            return summary
        electron_count = _mean_field_electron_count(mf, mol)
        if electron_count is None or electron_count <= 0:
            summary['reason'] = 'electron count is unavailable for T1 normalization'
            return summary
        norm_squared = sum(float(np.vdot(matrix, matrix).real) for matrix in matrices)
        t1_value = float(np.sqrt(norm_squared / electron_count))
        singular_values = [
            np.linalg.svd(matrix, compute_uv=False)
            for matrix in matrices
            if matrix.size
        ]
        d1_value = max(
            (float(values[0]) for values in singular_values if values.size),
            default=0.0,
        )
        open_shell = bool(getattr(mol, 'spin', 0)) or len(matrices) > 1
        t1_warning = CCSD_T1_WARNING_THRESHOLD_OPEN_SHELL if open_shell else CCSD_T1_WARNING_THRESHOLD_RESTRICTED
        warning = t1_value >= t1_warning or d1_value >= CCSD_D1_WARNING_THRESHOLD
        promotion = t1_value >= CCSD_T1_PROMOTION_THRESHOLD or d1_value >= CCSD_D1_PROMOTION_THRESHOLD
        summary.update({
            'status': 'available',
            't1': t1_value,
            'd1': d1_value,
            'reference_type': 'unrestricted_or_open_shell' if open_shell else 'restricted',
            'electron_count': electron_count,
            'spin_channels': len(matrices),
            'warning': warning,
            'promote_to_multireference': promotion,
            'level': 'strong' if promotion else 'elevated' if warning else 'normal',
            'interpretation': (
                'CCSD singles amplitudes exceed the multireference-promotion threshold.'
                if promotion else
                'CCSD singles amplitudes are elevated; validate before perturbative triples.'
                if warning else
                'CCSD singles amplitudes are within the single-reference routing range.'
            ),
        })
        return summary
    except Exception as exc:  # pragma: no cover - diagnostic must not invalidate a completed calculation
        summary['reason'] = str(exc)
        return summary


def _molecular_method_recommendation(
    level: str,
    physics_score: float,
    solver_stress_score: float,
    ccsd_t1_d1: Optional[Dict[str, Any]] = None,
    current_method: Optional[str] = None,
) -> Dict[str, Any]:
    ccsd_t1_d1 = ccsd_t1_d1 if isinstance(ccsd_t1_d1, dict) else {}
    normalized_method = str(current_method or '').strip().lower()
    if normalized_method in ('casci', 'casscf'):
        return {
            'preferred': [normalized_method],
            'post_cas': ['sc_nevpt2'],
            'screening': [],
            'risky': [],
            'next_step': (
                'The current multireference treatment completed; validate active-space completeness '
                'and add supported post-CAS dynamic correlation when quantitative accuracy is required.'
            ),
        }
    if ccsd_t1_d1.get('promote_to_multireference'):
        return {
            'preferred': ['casscf', 'casci'],
            'post_cas': ['sc_nevpt2'],
            'screening': ['mp2_active_space_audit'],
            'risky': ['ccsd_t'],
            'next_step': 'CCSD T1/D1 diagnostics exceed the promotion threshold; build or approve an ActiveSpaceAudit candidate.',
        }
    if level == 'strong':
        return {
            'preferred': ['casscf', 'casci'],
            'post_cas': ['sc_nevpt2'],
            'screening': ['mp2_active_space_audit'],
            'risky': ['mp2', 'ccsd_t'],
            'next_step': 'Build or approve an ActiveSpaceAudit candidate before multireference treatment.',
        }
    if level == 'moderate':
        preferred = ['ccsd']
        if physics_score >= solver_stress_score:
            preferred.append('casscf_if_active_space_is_compact')
        if ccsd_t1_d1.get('warning'):
            return {
                'preferred': preferred + ['casscf_if_active_space_is_compact'],
                'post_cas': ['sc_nevpt2_if_casscf'],
                'screening': ['mp2_active_space_audit'],
                'risky': ['ccsd_t_before_t1_d1_recovers'],
                'next_step': 'CCSD T1/D1 is elevated; review the active space before using perturbative triples.',
            }
        return {
            'preferred': preferred,
            'post_cas': ['sc_nevpt2_if_casscf'],
            'screening': ['mp2_active_space_audit'],
            'risky': ['ccsd_t_without_diagnostics'],
            'next_step': 'Run CCSD diagnostics or review the MP2 active-space audit before using perturbative triples.',
        }
    return {
        'preferred': ['ccsd_t', 'ccsd'],
        'post_cas': [],
        'screening': ['mp2'],
        'risky': [],
        'next_step': 'Single-reference methods are acceptable unless later diagnostics become unstable.',
    }


def build_molecular_correlation_risk(
    mf: Any,
    mol: Any,
    *,
    homo: Optional[float] = None,
    lumo: Optional[float] = None,
    gap: Optional[float] = None,
    spin_square: Optional[float] = None,
    fractional_occupations: Optional[List[Dict[str, Any]]] = None,
    frontier_degeneracy: Optional[Dict[str, Any]] = None,
    low_energy_manifold: Optional[Dict[str, Any]] = None,
    post_hf: Any = None,
    scf_stability: Optional[Dict[str, Any]] = None,
    reference_energy: Any = None,
    correlation_energy: Any = None,
    correlated_natural_occupations: Optional[Sequence[Any]] = None,
    current_method: Optional[str] = None,
) -> Dict[str, Any]:
    physics_components: List[Dict[str, Any]] = []
    solver_components: List[Dict[str, Any]] = []
    physics_scores: List[Tuple[float, float]] = []
    solver_scores: List[Tuple[float, float]] = []
    spin_summary = _reference_spin_symmetry_summary(mf, mol, spin_square=spin_square)
    if spin_square is None:
        spin_square = spin_summary.get('spin_square')

    if gap is not None:
        gap_score = 1.0 if gap <= 0 else _clip_unit((0.20 - float(gap)) / 0.18)
        physics_components.append({
            'name': 'homo_lumo_gap',
            'score': round(gap_score, 6),
            'value': {'homo': homo, 'lumo': lumo, 'gap': gap},
            'interpretation': 'Small frontier-orbital gaps indicate near-degeneracy.',
        })
        physics_scores.append((0.30, gap_score))

    if frontier_degeneracy is None:
        frontier_degeneracy = _frontier_orbital_degeneracy_summary(
            mf,
            evaluate_alpha_beta=bool(spin_summary.get('alpha_beta_relevant')),
        )
    if isinstance(frontier_degeneracy, dict) and frontier_degeneracy.get('status') == 'available':
        degeneracy_score = _clip_unit(float(frontier_degeneracy.get('score') or 0.0))
        physics_components.append({
            'name': 'frontier_orbital_degeneracy',
            'score': round(degeneracy_score, 6),
            'value': frontier_degeneracy,
            'interpretation': 'Near-degenerate HOMO/LUMO frontier clusters or alpha/beta frontier pairs indicate multireference risk.',
        })
        physics_scores.append((0.16, degeneracy_score))

    if isinstance(low_energy_manifold, dict) and low_energy_manifold.get('status') == 'available':
        manifold_score = _clip_unit(float(low_energy_manifold.get('risk_score') or 0.0))
        classification = low_energy_manifold.get('classification')
        if classification == 'numerically_degenerate':
            interpretation = (
                'Multiple computed many-body roots are numerically degenerate in the selected particle, spin, and '
                'configured symmetry sector. Ground-state observables may require root resolution or a physically '
                'defined ensemble.'
            )
        elif classification == 'near_degenerate':
            interpretation = (
                'Several many-body roots lie within the low-energy window, providing direct evidence that one '
                'electronic state may not describe the system robustly.'
            )
        else:
            interpretation = (
                'The lowest computed root is isolated within the selected particle, spin, and configured symmetry '
                'sector; uncomputed sectors can still contain degenerate states.'
            )
        physics_components.append({
            'name': 'many_body_low_energy_manifold',
            'score': round(manifold_score, 6),
            'value': low_energy_manifold,
            'interpretation': interpretation,
        })
        physics_scores.append((0.30, manifold_score))

    spin = getattr(mol, 'spin', 0)
    try:
        spin_value = int(spin)
    except (TypeError, ValueError):
        spin_value = 0
    physics_components.append({
        'name': 'open_shell',
        'status': 'informational',
        'score': None,
        'value': {'spin': spin_value},
        'interpretation': 'Open-shell electron counts define the occupation baseline; they are not multireference evidence by themselves.',
    })

    mean_field_occupation_score = None
    if fractional_occupations is not None:
        occupations = [
            _as_float(item.get('occupation'))
            for item in fractional_occupations if isinstance(item, dict)
        ]
        occupations = [value for value in occupations if value is not None]
        occupation_routing = _occupation_routing_summary(occupations, spin_value)
        mean_field_occupation_score = occupation_routing['score']
        physics_components.append({
            'name': 'mean_field_fractional_occupations',
            'status': occupation_routing['status'],
            'score': None if mean_field_occupation_score is None else round(mean_field_occupation_score, 6),
            'value': {
                'count': len(fractional_occupations),
                'max_fractionality': max((_occupation_fractionality(value) for value in occupations), default=0.0),
                'routing': occupation_routing,
            },
            'interpretation': 'Occupation deviations beyond the spin-adapted determinant baseline indicate correlation risk; SOMOs remain active-space evidence.',
        })
        if mean_field_occupation_score is not None:
            physics_scores.append((0.18, mean_field_occupation_score))
    else:
        physics_components.append({
            'name': 'mean_field_fractional_occupations', 'status': 'unavailable',
            'score': None, 'value': None,
            'interpretation': 'Spatial natural occupations are unavailable; this component is not scored.',
        })

    natural_summary = (
        _natural_orbital_summary_from_occupations(
            correlated_natural_occupations,
            source='cas_result.natural_occupations',
        )
        if correlated_natural_occupations is not None
        else _natural_orbital_summary(post_hf, mf=mf)
    )
    natural_metrics = _natural_occupation_metrics(natural_summary, spin=spin_value)
    natural_score = 0.0
    if natural_metrics['routing']['status'] == 'available':
        natural_score = float(natural_metrics['routing']['score'])
        physics_components.append({
            'name': 'correlated_natural_occupations',
            'score': round(natural_score, 6),
            'value': natural_metrics,
            'interpretation': 'Correlated natural occupations deviating from the spin-adapted determinant baseline identify multireference risk.',
        })
        physics_scores.append((0.34, natural_score))
    elif natural_metrics['status'] == 'out_of_bounds':
        solver_components.append({
            'name': 'nonphysical_correlated_density',
            'score': 1.0,
            'value': natural_metrics,
            'interpretation': (
                'Correlated occupations outside [0, 2] indicate breakdown of the current approximate '
                'density; they are solver-stress evidence and are not active-space selection data.'
            ),
        })
        solver_scores.append((0.30, 1.0))

    converged = getattr(mf, 'converged', None)
    if converged is not None:
        convergence_score = 0.0 if bool(converged) else 1.0
        solver_components.append({
            'name': 'scf_convergence',
            'score': convergence_score,
            'value': {'converged': bool(converged)},
            'interpretation': 'SCF convergence failure is solver stress, not a physical phase label by itself.',
        })
        solver_scores.append((0.20, convergence_score))

    if spin_square is not None:
        target_s = abs(float(getattr(mol, 'spin', 0))) / 2.0
        target_s2 = target_s * (target_s + 1.0)
        spin_deviation = abs(float(spin_square) - target_s2)
        contamination_score = _clip_unit(spin_deviation / 1.0)
        solver_components.append({
            'name': 'spin_contamination',
            'score': round(contamination_score, 6),
            'value': {'spin_square': spin_square, 'target_spin_square': target_s2, 'deviation': spin_deviation},
            'interpretation': 'Spin contamination indicates that the reference determinant is strained.',
        })
        solver_scores.append((0.22, contamination_score))

    if spin_summary.get('broken_symmetry'):
        solver_components.append({
            'name': 'unrestricted_spin_symmetry_breaking',
            'score': 1.0,
            'value': spin_summary,
            'interpretation': 'A spin-zero unrestricted reference has broken alpha/beta symmetry; this is a static-correlation warning, not an open-shell electron count by itself.',
        })
        solver_scores.append((0.28, 1.0))

    if isinstance(scf_stability, dict) and scf_stability.get('stable') is not None:
        stable = scf_stability.get('stable')
        stability_score = 1.0 if stable is False else 0.0 if stable is True else 0.5
        solver_components.append({
            'name': 'scf_stability',
            'score': round(stability_score, 6),
            'value': scf_stability,
            'interpretation': 'An unstable SCF solution means the current reference is not a reliable expansion point.',
        })
        solver_scores.append((0.28, stability_score))

    if post_hf is not None and getattr(post_hf, 'converged', None) is not None:
        post_hf_score = 0.0 if bool(getattr(post_hf, 'converged')) else 0.8
        solver_components.append({
            'name': 'post_hf_convergence',
            'score': round(post_hf_score, 6),
            'value': {'converged': bool(getattr(post_hf, 'converged'))},
            'interpretation': 'Post-HF convergence failure flags method stress in the selected reference.',
        })
        solver_scores.append((0.16, post_hf_score))

    ccsd_t1_d1 = _ccsd_t1_d1_diagnostics(post_hf, mf, mol)
    if ccsd_t1_d1.get('status') == 'available':
        t1_value = float(ccsd_t1_d1.get('t1') or 0.0)
        d1_value = float(ccsd_t1_d1.get('d1') or 0.0)
        amplitude_score = max(
            _clip_unit(t1_value / CCSD_T1_PROMOTION_THRESHOLD),
            _clip_unit(d1_value / CCSD_D1_PROMOTION_THRESHOLD),
        )
        solver_components.append({
            'name': 'ccsd_t1_d1',
            'score': round(amplitude_score, 6),
            'value': ccsd_t1_d1,
            'interpretation': 'Large CCSD single-excitation amplitudes indicate that the single-reference expansion is becoming unreliable.',
        })
        solver_scores.append((0.26, amplitude_score))

    t2_evidence = _t2_orbital_importance(post_hf, mf)
    t2_importance = t2_evidence['importance']
    if t2_importance:
        max_t2 = max(float(value) for value in t2_importance.values())
        t2_score = _clip_unit(max_t2 / 0.5)
        solver_components.append({
            'name': 'max_double_excitation_amplitude',
            'score': round(t2_score, 6),
            'value': {'max_abs_t2': max_t2},
            'interpretation': 'Large double-excitation amplitudes indicate stress in a low-order single-reference expansion.',
        })
        solver_scores.append((0.22, t2_score))

    corr = _as_float(correlation_energy)
    ref = _as_float(reference_energy)
    if corr is not None:
        energy_scale = max(abs(ref) if ref is not None else 0.0, 1.0)
        corr_fraction = abs(corr) / energy_scale
        corr_score = _clip_unit(corr_fraction / 0.25)
        solver_components.append({
            'name': 'correlation_energy_stress',
            'score': round(corr_score, 6),
            'value': {
                'correlation_energy': corr,
                'reference_energy': ref,
                'correlation_fraction': corr_fraction,
            },
            'interpretation': 'Large reference-relative correlation energy is a method-stress warning.',
        })
        solver_scores.append((0.12, corr_score))

    physics_score = _mean_score(physics_scores)
    solver_stress_score = _mean_score(solver_scores)
    overall_score = 0.65 * physics_score + 0.35 * solver_stress_score
    physics_level = _risk_level(physics_score)
    solver_stress_level = _risk_level(solver_stress_score)
    scf_unstable = (
        isinstance(scf_stability, dict)
        and scf_stability.get('status') != 'unavailable'
        and scf_stability.get('stable') is False
    )
    if scf_unstable and solver_stress_level == 'weak':
        solver_stress_level = 'moderate'
    level = _risk_level(overall_score)
    level_reason = 'score'
    scf_failed = getattr(mf, 'converged', None) is False
    post_hf_failed = post_hf is not None and getattr(post_hf, 'converged', None) is False
    nonphysical_correlated_density = natural_metrics['status'] == 'out_of_bounds'
    if scf_failed or post_hf_failed:
        level = 'strong'
        solver_stress_level = 'strong'
        level_reason = 'solver_nonconvergence'
    elif nonphysical_correlated_density:
        # Approximate correlated 1RDMs are not guaranteed to be N-representable.
        # Once their occupations leave the physical [0, 2] interval they must
        # not be averaged back into a weak-correlation route or used to select
        # an active space.  Treat this as a hard method-breakdown signal while
        # preserving the independently evaluated physics level.
        level = 'strong'
        solver_stress_level = 'strong'
        level_reason = 'nonphysical_correlated_density'
    elif (
        isinstance(low_energy_manifold, dict)
        and low_energy_manifold.get('classification') == 'numerically_degenerate'
    ):
        level = 'strong'
        physics_level = 'strong'
        level_reason = 'many_body_ground_state_degeneracy'
    elif ccsd_t1_d1.get('promote_to_multireference'):
        level = 'strong'
        level_reason = 'ccsd_t1_d1_promotion'
    elif natural_score >= CORRELATED_NATURAL_OCCUPATION_STRONG_THRESHOLD:
        level = 'strong'
        physics_level = 'strong'
        level_reason = 'correlated_natural_occupations'
    elif natural_score >= CORRELATED_NATURAL_OCCUPATION_MODERATE_THRESHOLD:
        if physics_level == 'weak':
            physics_level = 'moderate'
        if level == 'weak':
            level = 'moderate'
        if level_reason == 'score':
            level_reason = 'correlated_natural_occupations_moderate'
    elif (
        spin_summary.get('broken_symmetry')
        and mean_field_occupation_score is not None
        and mean_field_occupation_score >= 0.5
    ):
        # At dissociation, a nominal singlet UHF solution can lower its energy
        # by localizing alpha and beta electrons on different fragments.  UNO
        # occupations near one make this a direct static-correlation signal.
        level = 'strong'
        physics_level = 'strong'
        level_reason = 'uhf_broken_symmetry_uno'
    routing_level = level
    routing_reason = level_reason
    if (
        physics_level != 'strong'
        and solver_stress_level == 'strong'
        and level_reason in (
            'solver_nonconvergence',
            'nonphysical_correlated_density',
            'score',
        )
    ):
        routing_level = 'moderate'
        routing_reason = 'validate_single_reference_with_ccsd_before_multireference_promotion'
    confidence_count = len(physics_scores) + len(solver_scores)
    confidence = 'high' if confidence_count >= 5 else 'medium' if confidence_count >= 3 else 'low'
    return {
        'kind': 'molecular_correlation_risk',
        't2_orbital_importance_evidence': t2_evidence,
        'physics_score': round(float(physics_score), 6),
        'physics_level': physics_level,
        'solver_stress_score': round(float(solver_stress_score), 6),
        'solver_stress_level': solver_stress_level,
        'overall_score': round(float(overall_score), 6),
        'level': level,
        'level_reason': level_reason,
        'routing_level': routing_level,
        'routing_reason': routing_reason,
        'confidence': confidence,
        'physics_components': physics_components,
        'solver_stress_components': solver_components,
        'ccsd_t1_d1': ccsd_t1_d1,
        'reference_spin_symmetry': spin_summary,
        'correlated_natural_occupation_summary': natural_summary,
        'correlated_natural_occupation_metrics': natural_metrics,
        'method_recommendation': _molecular_method_recommendation(
            routing_level,
            physics_score,
            solver_stress_score,
            ccsd_t1_d1,
            current_method=current_method,
        ),
        'limitations': [
            'MolecularCorrelationRisk is a routing heuristic, not a replacement for benchmark-quality multireference diagnostics.',
            'physics_score tracks near-degeneracy and occupation evidence; solver_stress_score tracks reference and solver fragility.',
        ],
    }


def build_correlation_diagnostics(
    mf: Any,
    mol: Any,
    *,
    post_hf: Any = None,
    scf_stability: Optional[Dict[str, Any]] = None,
    reference_energy: Any = None,
    correlation_energy: Any = None,
    state_energies: Any = None,
    state_sector: Optional[Dict[str, Any]] = None,
    correlated_natural_occupations: Optional[Sequence[Any]] = None,
    current_method: Optional[str] = None,
) -> Dict[str, Any]:
    homo, lumo = _extract_homo_lumo(getattr(mf, 'mo_energy', None), getattr(mf, 'mo_occ', None))
    gap = None if homo is None or lumo is None else lumo - homo
    spin_summary = _reference_spin_symmetry_summary(mf, mol)
    spin_square = spin_summary.get('spin_square')
    frontier_degeneracy = _frontier_orbital_degeneracy_summary(
        mf,
        evaluate_alpha_beta=bool(spin_summary.get('alpha_beta_relevant')),
    )
    warnings = []
    if gap is not None and gap < 0.05:
        warnings.append('small_homo_lumo_gap')
    if isinstance(frontier_degeneracy, dict) and frontier_degeneracy.get('status') == 'available':
        if float(frontier_degeneracy.get('score') or 0.0) >= 0.5:
            warnings.append('frontier_orbital_degeneracy')
        alpha_beta = frontier_degeneracy.get('alpha_beta') if isinstance(frontier_degeneracy.get('alpha_beta'), dict) else {}
        if alpha_beta.get('degenerate_pairs'):
            warnings.append('alpha_beta_frontier_degeneracy')
    if getattr(mol, 'spin', 0):
        warnings.append('open_shell_system')

    if spin_summary.get('spin_contaminated'):
        warnings.append('spin_contamination')
    if spin_summary.get('broken_symmetry'):
        warnings.append('unrestricted_spin_symmetry_breaking')
    if (
        isinstance(scf_stability, dict)
        and scf_stability.get('status') != 'unavailable'
        and scf_stability.get('stable') is False
    ):
        warnings.append('scf_instability')

    occupation_summary = _mean_field_occupation_summary(mf, mol)
    occupations = occupation_summary.get('occupations') if isinstance(occupation_summary.get('occupations'), list) else []
    fractional_occupations = [
        {'index': index, 'occupation': occupation}
        for index, occupation in enumerate(occupations)
        if 1.0e-6 < occupation < 1.999999
    ]
    if occupation_summary.get('status') != 'available':
        fractional_occupations = None
        warnings.append('mean_field_natural_occupations_unavailable')

    state_sector_scope = (
        str(state_sector.get('scope')).strip()
        if isinstance(state_sector, dict) and state_sector.get('scope')
        else 'fixed_particle_spin_and_configured_symmetry_sector'
    )
    low_energy_manifold = summarize_low_energy_manifold(
        state_energies if state_energies is not None else [],
        spectrum_scope=f'targeted_roots_{state_sector_scope}',
        numerical_tolerance=1.0e-8,
        near_degeneracy_tolerance=1.0e-2,
    )
    if state_sector:
        low_energy_manifold['state_sector'] = dict(state_sector)
    low_energy_classification = low_energy_manifold.get('classification')
    if low_energy_classification == 'numerically_degenerate':
        warnings.append('many_body_ground_state_degeneracy')
    elif low_energy_classification == 'near_degenerate':
        warnings.append('many_body_near_degeneracy')

    recommended_next_steps = []
    if low_energy_classification in ('numerically_degenerate', 'near_degenerate'):
        recommended_next_steps.append('review_root_resolved_observables')
    if low_energy_manifold.get('manifold_may_extend_beyond_computed_roots'):
        recommended_next_steps.append('increase_nroots')

    risk = build_molecular_correlation_risk(
        mf,
        mol,
        homo=homo,
        lumo=lumo,
        gap=gap,
        spin_square=spin_square,
        fractional_occupations=fractional_occupations,
        frontier_degeneracy=frontier_degeneracy,
        low_energy_manifold=low_energy_manifold,
        post_hf=post_hf,
        scf_stability=scf_stability,
        reference_energy=reference_energy,
        correlation_energy=correlation_energy,
        correlated_natural_occupations=correlated_natural_occupations,
        current_method=current_method,
    )
    natural_metrics = risk.get('correlated_natural_occupation_metrics') or {}
    if natural_metrics.get('status') == 'out_of_bounds':
        warnings.append('nonphysical_correlated_density')
    elif float((natural_metrics.get('routing') or {}).get('score') or 0.0) >= 0.5:
        warnings.append('correlated_natural_occupations')
    if any(
        component['name'] == 'mean_field_fractional_occupations'
        and component.get('score') is not None and component['score'] > 1.0e-6
        for component in risk['physics_components']
    ):
        warnings.append('fractional_orbital_occupations')
    multireference_completed = str(current_method or '').strip().lower() in ('casci', 'casscf')
    recommendation = (
        risk.get('method_recommendation')
        if isinstance(risk.get('method_recommendation'), dict)
        else {}
    )
    preferred_methods = {
        str(item).strip().lower()
        for item in (recommendation.get('preferred') or [])
    }
    active_space_recommended = bool(preferred_methods.intersection({'casscf', 'casci'}))
    if multireference_completed:
        recommended_next_steps = [
            step
            for step in recommended_next_steps
            if step not in ('review_active_space', 'consider_casci_or_casscf')
        ]
        if risk['level'] in ('moderate', 'strong'):
            recommended_next_steps.append('validate_active_space_completeness')
            recommended_next_steps.append('consider_post_cas_dynamic_correlation')
    else:
        if active_space_recommended and 'review_active_space' not in recommended_next_steps:
            recommended_next_steps.append('review_active_space')
        if active_space_recommended and 'consider_casci_or_casscf' not in recommended_next_steps:
            recommended_next_steps.append('consider_casci_or_casscf')
        if risk.get('routing_level') == 'moderate':
            recommended_next_steps.append('run_ccsd_diagnostics')
    ccsd_t1_d1 = risk.get('ccsd_t1_d1') if isinstance(risk.get('ccsd_t1_d1'), dict) else {}
    if ccsd_t1_d1.get('status') == 'available' and ccsd_t1_d1.get('warning'):
        warnings.append('ccsd_t1_d1_elevated')
    if ccsd_t1_d1.get('promote_to_multireference'):
        warnings.append('ccsd_t1_d1_multireference_promotion')

    return {
        'kind': 'molecular_correlation_diagnostics',
        'homo': homo,
        'lumo': lumo,
        'gap': gap,
        'frontier_orbital_degeneracy': frontier_degeneracy,
        'low_energy_manifold': low_energy_manifold,
        'spin_square': spin_square,
        'reference_spin_symmetry': spin_summary,
        'scf_stability': scf_stability,
        'ccsd_t1_d1': ccsd_t1_d1,
        'warnings': warnings,
        'fractional_occupations': fractional_occupations,
        'mean_field_occupation_summary': occupation_summary,
        'recommended_next_steps': recommended_next_steps,
        'molecular_correlation_risk': risk,
        'physics_score': risk['physics_score'],
        'physics_level': risk['physics_level'],
        'solver_stress_score': risk['solver_stress_score'],
        'solver_stress_level': risk['solver_stress_level'],
        'routing_level': risk['routing_level'],
        'routing_reason': risk['routing_reason'],
        'score': risk['overall_score'],
        'level': risk['level'],
        'level_reason': risk['level_reason'],
        'confidence': risk['confidence'],
        'method_recommendation': risk['method_recommendation'],
        'correlated_natural_occupation_summary': risk.get('correlated_natural_occupation_summary'),
    }


def build_orbital_processing_summary(mf: Any, mol: Any, spec: OrbitalProcessingSpec) -> Dict[str, Any]:
    localization_method = (spec.localization_method or 'none').strip().lower()
    if localization_method == 'pm':
        localization_method = 'pipek_mezey'
    summary: Dict[str, Any] = {
        'enabled': bool(spec.enabled),
        'localization_method': localization_method,
        'localization_scope': spec.localization_scope,
        'use_natural_orbitals': bool(spec.use_natural_orbitals),
        'orbital_ordering': spec.orbital_ordering,
        'orbital_order': list(spec.orbital_order or []),
        'orbital_table': _orbital_table(mf),
        'notes': [],
    }
    if spec.orbital_ordering != 'canonical':
        summary['notes'].append(
            'The requested orbital ordering is applied by the correlated solver and recorded in its provenance.'
        )
    if spec.use_natural_orbitals:
        summary['notes'].append('Natural-orbital occupations currently use the available mean-field occupations unless a correlated density is provided.')

    if spec.enabled and localization_method not in ('none', '') and spec.localization_scope == 'analysis':
        if getattr(mf, 'mo_coeff', None) is None:
            summary['localization_status'] = 'unavailable'
            summary['notes'].append('Mean-field molecular orbitals are unavailable.')
            return summary
        if isinstance(_safe_to_list(getattr(mf, 'mo_coeff', None)), list) and _safe_to_list(getattr(mf, 'mo_coeff', None)) and isinstance(_safe_to_list(getattr(mf, 'mo_coeff', None))[0], list):
            pass
        try:
            from pyscf import lo  # pylint: disable=import-outside-toplevel

            if localization_method == 'boys':
                localized = lo.Boys(mol, mf.mo_coeff).kernel()
            elif localization_method == 'pipek_mezey':
                localized = lo.PM(mol, mf.mo_coeff).kernel()
            else:
                localized = None
            summary['localization_status'] = 'completed' if localized is not None else 'skipped'
            if localized is not None:
                try:
                    summary['localized_orbital_shape'] = list(localized.shape)
                except AttributeError:
                    summary['localized_orbital_shape'] = None
        except Exception as exc:
            summary['localization_status'] = 'failed'
            summary['notes'].append('Localization failed: {0}'.format(exc))
    else:
        summary['localization_status'] = (
            'deferred_to_active_space_solver'
            if localization_method not in ('none', '') and spec.localization_scope == 'active_space'
            else 'not_requested'
        )
    return summary
