from __future__ import annotations

from typing import Any, Dict, Iterable


def summarize_low_energy_manifold(
    state_energies: Iterable[Any],
    *,
    spectrum_scope: str,
    numerical_tolerance: float,
    near_degeneracy_tolerance: float,
) -> Dict[str, Any]:
    """Classify computed low-energy roots without claiming missing sectors."""
    try:
        levels = sorted(float(value) for value in state_energies)
    except (TypeError, ValueError):
        levels = []
    if len(levels) < 2:
        return {
            'status': 'insufficient_roots',
            'spectrum_scope': spectrum_scope,
            'computed_level_count': len(levels),
        }

    numerical_tolerance = max(float(numerical_tolerance), 0.0)
    near_degeneracy_tolerance = max(
        float(near_degeneracy_tolerance),
        numerical_tolerance,
    )
    excitation_energies = [max(0.0, value - levels[0]) for value in levels]
    gap = excitation_energies[1]
    degenerate_count = sum(value <= numerical_tolerance for value in excitation_energies)
    low_energy_count = sum(value <= near_degeneracy_tolerance for value in excitation_energies)
    if degenerate_count > 1:
        classification = 'numerically_degenerate'
        risk_score = 1.0
    elif low_energy_count > 1:
        classification = 'near_degenerate'
        risk_score = max(
            0.0,
            min(1.0, 1.0 - gap / max(near_degeneracy_tolerance, 1.0e-16)),
        )
    else:
        classification = 'isolated_within_computed_sector'
        risk_score = 0.0

    targeted_roots = spectrum_scope.startswith('targeted_roots')
    return {
        'status': 'available',
        'state_energies': levels,
        'excitation_energies': excitation_energies,
        'gap': gap,
        'classification': classification,
        'numerical_degeneracy_tolerance': numerical_tolerance,
        'near_degeneracy_tolerance': near_degeneracy_tolerance,
        'ground_state_count_within_numerical_tolerance': degenerate_count,
        'low_energy_state_count': low_energy_count,
        'computed_level_count': len(levels),
        'spectrum_scope': spectrum_scope,
        'manifold_may_extend_beyond_computed_roots': (
            targeted_roots and low_energy_count == len(levels)
        ),
        'risk_score': risk_score,
    }
