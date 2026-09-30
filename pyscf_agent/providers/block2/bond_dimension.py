from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, Iterable, Tuple

from ...backend.electronic_hamiltonian import ElectronicHamiltonian
from .config import Block2DMRGConfig


BOND_DIMENSION_PLAN_SCHEMA = 'pyscf-agent.block2-bond-dimension-plan.v1'


def _comb_or_zero(n: int, k: int) -> int:
    return math.comb(n, k) if 0 <= k <= n else 0


def target_sector_dimension(
    n_orbitals: int,
    nalpha: int,
    nbeta: int,
    symmetry: str,
) -> int:
    """Return the determinant or spin-adapted dimension of the target sector."""

    determinant_dimension = _comb_or_zero(n_orbitals, nalpha) * _comb_or_zero(
        n_orbitals, nbeta
    )
    if symmetry != 'su2':
        return int(determinant_dimension)
    high_spin_dimension = _comb_or_zero(n_orbitals, nalpha + 1) * _comb_or_zero(
        n_orbitals, nbeta - 1
    )
    return int(max(0, determinant_dimension - high_spin_dimension))


def _allowed_occupations(orbitals: int, electrons: int, total_orbitals: int) -> Iterable[int]:
    remaining = total_orbitals - orbitals
    lower = max(0, electrons - remaining)
    upper = min(orbitals, electrons)
    return range(lower, upper + 1)


def _cut_schmidt_rank_cap(
    n_orbitals: int,
    nalpha: int,
    nbeta: int,
    cut: int,
) -> Tuple[int, int]:
    """Return the fixed-(Nalpha, Nbeta) single-state rank cap at one cut."""

    right_orbitals = n_orbitals - cut
    per_state_rank = 0
    sectors = 0
    for left_alpha in _allowed_occupations(cut, nalpha, n_orbitals):
        right_alpha = nalpha - left_alpha
        alpha_left_dimension = math.comb(cut, left_alpha)
        alpha_right_dimension = math.comb(right_orbitals, right_alpha)
        for left_beta in _allowed_occupations(cut, nbeta, n_orbitals):
            right_beta = nbeta - left_beta
            left_dimension = alpha_left_dimension * math.comb(cut, left_beta)
            right_dimension = alpha_right_dimension * math.comb(
                right_orbitals,
                right_beta,
            )
            per_state_rank += min(left_dimension, right_dimension)
            sectors += 1
    return int(per_state_rank), int(sectors)


def exact_bond_dimension_plan(
    hamiltonian: ElectronicHamiltonian,
    config: Block2DMRGConfig,
) -> Dict[str, Any]:
    """Plan a bounded DMRG schedule from exact fixed-sector Schmidt ranks.

    The spin-resolved Fock-space rank is a conservative exact upper bound for
    both SZ and SU(2) calculations. Entanglement and discarded weight still
    determine how much of this capacity is needed in practice.
    """

    hamiltonian.validate()
    n_orbitals = int(hamiltonian.n_orbitals)
    nalpha, nbeta = (int(value) for value in hamiltonian.n_electrons)
    nroots = int(config.nroots)
    cuts = []
    for cut in range(1, n_orbitals):
        per_state_rank, sectors = _cut_schmidt_rank_cap(
            n_orbitals,
            nalpha,
            nbeta,
            cut,
        )
        cuts.append({
            'cut': cut,
            'left_orbitals': cut,
            'right_orbitals': n_orbitals - cut,
            'per_state_schmidt_rank_cap': per_state_rank,
            'schmidt_rank_cap': per_state_rank,
            'particle_number_sectors': sectors,
        })
    per_state_exact_cap = max(
        (int(item['per_state_schmidt_rank_cap']) for item in cuts),
        default=1,
    )
    determinant_space_dimension = (
        math.comb(n_orbitals, nalpha) * math.comb(n_orbitals, nbeta)
    )
    effective_symmetry = 'su2' if config.symmetry == 'auto' else config.symmetry
    root_sector_dimension = target_sector_dimension(
        n_orbitals,
        nalpha,
        nbeta,
        effective_symmetry,
    )
    exact_cap = min(determinant_space_dimension, per_state_exact_cap * nroots)
    requested_dimensions = [int(value) for value in config.bond_dimensions]
    requested_maximum = int(config.max_bond_dimension)
    enabled = bool(config.bond_dimension_planning)
    effective_dimensions = (
        [min(value, exact_cap) for value in requested_dimensions]
        if enabled
        else list(requested_dimensions)
    )
    effective_maximum = min(requested_maximum, exact_cap) if enabled else requested_maximum
    cap_applied = bool(
        effective_dimensions != requested_dimensions
        or effective_maximum != requested_maximum
    )
    return {
        'schema': BOND_DIMENSION_PLAN_SCHEMA,
        'enabled': enabled,
        'method': 'fixed_particle_number_schmidt_rank',
        'n_orbitals': n_orbitals,
        'n_electrons': [nalpha, nbeta],
        'spin': int(hamiltonian.spin),
        'nroots': nroots,
        'determinant_space_dimension': determinant_space_dimension,
        'target_root_sector_dimension': root_sector_dimension,
        'coarse_fock_space_cap': 2 ** n_orbitals,
        'per_state_exact_bond_dimension_cap': per_state_exact_cap,
        'exact_bond_dimension_cap': exact_cap,
        'cap_kind': 'exact_single_state' if nroots == 1 else 'conservative_multi_root',
        'cut_profile': cuts,
        'requested_bond_dimensions': requested_dimensions,
        'requested_max_bond_dimension': requested_maximum,
        'effective_bond_dimensions': effective_dimensions,
        'effective_max_bond_dimension': effective_maximum,
        'cap_applied': cap_applied,
        'symmetry_interpretation': (
            'Conservative fixed-(Nalpha,Nbeta) upper bound; SU(2) symmetry may '
            'require fewer retained multiplets.'
        ),
        'multi_root_interpretation': (
            'For state-averaged calculations, nroots times the exact single-state '
            'cap is used as a conservative numerical budget, bounded by the '
            'determinant-space dimension.'
        ),
        'stopping_policy': (
            'The exact cap limits capacity; energy change and discarded weight '
            'still decide early convergence.'
        ),
    }


def apply_bond_dimension_plan(
    config: Block2DMRGConfig,
    plan: Dict[str, Any],
) -> Block2DMRGConfig:
    if not plan.get('enabled'):
        return config
    return replace(
        config,
        bond_dimensions=[int(value) for value in plan['effective_bond_dimensions']],
        max_bond_dimension=int(plan['effective_max_bond_dimension']),
    )


__all__ = [
    'BOND_DIMENSION_PLAN_SCHEMA',
    'apply_bond_dimension_plan',
    'exact_bond_dimension_plan',
    'target_sector_dimension',
]
