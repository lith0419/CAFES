from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional


def _electron_count(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return int(value)


def _expanded_electron_count(value: Any, occupied_orbitals_added: int) -> Any:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError('Spin-resolved nelecas must contain alpha and beta counts.')
        return [
            int(value[0]) + int(occupied_orbitals_added),
            int(value[1]) + int(occupied_orbitals_added),
        ]
    return int(value) + 2 * int(occupied_orbitals_added)


def entanglement_active_space_recommendation(
    dmrg_result: Dict[str, Any],
    *,
    ncore: int,
    ncas: int,
    nelecas: Any,
    nmo: int,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Propose a bounded CAS expansion from in-space saturation evidence.

    Entanglement from an active-space DMRG wavefunction cannot directly measure
    excluded orbitals.  The recommendation therefore requires both appreciable
    in-space entanglement and natural occupations that remain fractional at the
    occupied/virtual edges, and records that limitation explicitly.
    """

    payload = dict(options or {})
    enabled = bool(payload.get('entanglement_active_space_review', True))
    entropy_threshold = float(payload.get('entanglement_entropy_threshold', 0.10))
    mutual_information_threshold = float(
        payload.get('entanglement_mutual_information_threshold', 0.02)
    )
    occupation_tolerance = float(payload.get('entanglement_occupation_tolerance', 0.02))
    expansion_step = max(1, int(payload.get('active_space_expansion_step', 1)))
    max_orbitals = max(int(ncas), int(payload.get('active_space_max_orbitals', 64)))
    diagnostics = dmrg_result.get('entanglement_diagnostics')
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    occupations = [
        float(value) for value in (dmrg_result.get('natural_occupations') or [])
    ]
    entropies = [
        float(value) for value in (diagnostics.get('single_orbital_entropy') or [])
    ]
    max_entropy = max(entropies) if entropies else None
    max_mutual_information = diagnostics.get('max_mutual_information')
    try:
        max_mutual_information = float(max_mutual_information)
    except (TypeError, ValueError):
        max_mutual_information = None
    entangled_count = sum(value >= entropy_threshold for value in entropies)
    occupation_high = max(occupations) if occupations else None
    occupation_low = min(occupations) if occupations else None
    occupied_edge_fractional = bool(
        occupation_high is not None
        and occupation_high < 2.0 - occupation_tolerance
    )
    virtual_edge_fractional = bool(
        occupation_low is not None
        and occupation_low > occupation_tolerance
    )
    entanglement_present = bool(
        (max_entropy is not None and max_entropy >= entropy_threshold)
        or (
            max_mutual_information is not None
            and max_mutual_information >= mutual_information_threshold
        )
    )
    base = {
        'schema': 'pyscf-agent.entanglement-active-space-recommendation.v1',
        'enabled': enabled,
        'status': 'not_recommended',
        'source': 'block2_active_space_dmrg',
        'evidence': {
            'max_single_orbital_entropy': max_entropy,
            'entangled_orbital_count': int(entangled_count),
            'max_mutual_information': max_mutual_information,
            'highest_natural_occupation': occupation_high,
            'lowest_natural_occupation': occupation_low,
            'occupied_edge_fractional': occupied_edge_fractional,
            'virtual_edge_fractional': virtual_edge_fractional,
        },
        'thresholds': {
            'single_orbital_entropy': entropy_threshold,
            'mutual_information': mutual_information_threshold,
            'natural_occupation_edge': occupation_tolerance,
        },
        'current_active_space': {
            'ncas': int(ncas),
            'nelecas': _electron_count(nelecas),
            'ncore': int(ncore),
            'nmo': int(nmo),
        },
        'limitation': (
            'The active-space MPS measures entanglement only among included orbitals. '
            'Expansion is a boundary-saturation heuristic; the proposed external '
            'orbitals must be recomputed and reviewed.'
        ),
    }
    if not enabled:
        base['status'] = 'disabled'
        return base
    if not diagnostics or not occupations:
        base.update({
            'status': 'insufficient_evidence',
            'reason': 'Both entanglement diagnostics and active-space natural occupations are required.',
        })
        return base
    if not entanglement_present or not (occupied_edge_fractional or virtual_edge_fractional):
        base['reason'] = (
            'The active space does not simultaneously show strong in-space entanglement '
            'and fractional occupation at an expandable boundary.'
        )
        return base
    if int(ncas) >= max_orbitals:
        base.update({
            'status': 'limit_reached',
            'reason': 'The configured active-space orbital limit has been reached.',
        })
        return base

    lower_available = max(0, int(ncore))
    upper_available = max(0, int(nmo) - int(ncore) - int(ncas))
    lower_added = min(expansion_step, lower_available) if occupied_edge_fractional else 0
    upper_added = min(expansion_step, upper_available) if virtual_edge_fractional else 0
    room = max_orbitals - int(ncas)
    if lower_added + upper_added > room:
        if lower_added and upper_added and room == 1:
            # Prefer the boundary farther from an integer occupation.
            occupied_deviation = 2.0 - float(occupation_high)
            virtual_deviation = float(occupation_low)
            lower_added, upper_added = (
                (1, 0) if occupied_deviation >= virtual_deviation else (0, 1)
            )
        else:
            upper_added = min(upper_added, max(0, room - lower_added))
    if lower_added + upper_added == 0:
        base.update({
            'status': 'no_external_orbitals',
            'reason': 'No adjacent external orbitals are available within the configured limit.',
        })
        return base

    target_ncore = int(ncore) - lower_added
    target_ncas = int(ncas) + lower_added + upper_added
    target_indices = list(range(target_ncore, target_ncore + target_ncas))
    target_nelecas = _expanded_electron_count(nelecas, lower_added)
    reasons = []
    if lower_added:
        reasons.append(
            'Add {0} occupied-side orbital(s): the largest active natural occupation is {1:.6g}, below {2:.6g}.'.format(
                lower_added,
                occupation_high,
                2.0 - occupation_tolerance,
            )
        )
    if upper_added:
        reasons.append(
            'Add {0} virtual-side orbital(s): the smallest active natural occupation is {1:.6g}, above {2:.6g}.'.format(
                upper_added,
                occupation_low,
                occupation_tolerance,
            )
        )
    base.update({
        'status': 'approval_recommended',
        'reason': ' '.join(reasons),
        'candidate_active_space': {
            'enabled': True,
            'selection_method': 'manual',
            'ncas': target_ncas,
            'nelecas': target_nelecas,
            'orbital_indices': target_indices,
            'orbital_index_basis': 'checkpoint_optimized_mo',
            'approved': False,
            'rationale': 'Entanglement and natural-occupation boundary saturation suggest a larger active space.',
        },
        'expansion': {
            'occupied_orbitals_added': lower_added,
            'virtual_orbitals_added': upper_added,
            'new_ncore': target_ncore,
            'new_ncas': target_ncas,
            'new_nelecas': copy.deepcopy(target_nelecas),
            'new_orbital_indices': target_indices,
            'estimated_dimension_growth_note': (
                'The CI/DMRG state space grows combinatorially; review the attached cost estimate before execution.'
            ),
        },
        'confidence': (
            'high'
            if entangled_count >= max(2, int(math.ceil(ncas / 2.0)))
            and occupied_edge_fractional
            and virtual_edge_fractional
            else 'moderate'
        ),
    })
    candidate = base['candidate_active_space']
    candidate['audit'] = {
        'version': 1,
        'orbital_index_basis': 'checkpoint_optimized_mo',
        'selection_method': 'entanglement_boundary_expansion',
        'selected_candidate_method': 'entanglement_expansion',
        'candidate_active_spaces': [{
            'method': 'entanglement_expansion',
            'status': 'available',
            'orbital_index_basis': 'checkpoint_optimized_mo',
            'orbital_indices': list(target_indices),
            'ncas': target_ncas,
            'estimated_nelecas': copy.deepcopy(target_nelecas),
            'selection_reasons': {'global': list(reasons)},
            'note': base['limitation'],
        }],
        'entanglement_evidence': copy.deepcopy(base['evidence']),
        'selection_parameters': copy.deepcopy(base['thresholds']),
        'ncas_nelecas_consistency': {
            'consistent': True,
            'messages': ['Expanded ncas/nelecas preserve a paired inactive core.'],
            'final_ncas': target_ncas,
            'final_nelecas': copy.deepcopy(target_nelecas),
        },
        'manual_approval': {
            'approved': False,
            'status': 'requires_user_review',
            'recorded_in_request': False,
        },
        'method_recommendation': {
            'recommended_next_step': 'block2_dmrg_casscf',
            'note': 'Reoptimize the expanded active space with a fresh MPS seeded by the saved optimized orbitals.',
        },
        'limitations': [base['limitation']],
    }
    return base


__all__ = ['entanglement_active_space_recommendation']
