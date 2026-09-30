from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List

from ...backend.electronic_hamiltonian import ElectronicHamiltonian


BLOCK2_RESULT_CONTRACT_SCHEMA = 'pyscf-agent.block2-result-contract.v1'


def _float_list(values: Iterable[Any]) -> List[float]:
    return [float(value) for value in values]


def block2_state_sector(
    hamiltonian: ElectronicHamiltonian,
    symmetry: str,
) -> Dict[str, Any]:
    symmetry_name = str(symmetry or '').strip().lower()
    return {
        'scope': (
            'fixed_particle_total_spin_sector'
            if symmetry_name == 'su2'
            else 'fixed_particle_spin_projection_sector'
        ),
        'n_orbitals': int(hamiltonian.n_orbitals),
        'n_electrons': [int(value) for value in hamiltonian.n_electrons],
        'total_electrons': int(hamiltonian.total_electrons),
        'spin': int(hamiltonian.spin),
        'symmetry': symmetry_name,
        'orbital_symmetries': [
            int(value) for value in (hamiltonian.orbital_symmetries or ())
        ],
    }


def _block2_quality_checks(
    result: Dict[str, Any],
    hamiltonian: ElectronicHamiltonian,
    config: Any,
) -> List[Dict[str, Any]]:
    convergence = result.get('convergence') if isinstance(result.get('convergence'), dict) else {}
    state_energies = result.get('state_energies') if isinstance(result.get('state_energies'), list) else []
    nalpha, nbeta = (int(value) for value in hamiltonian.n_electrons)
    norb = int(hamiltonian.n_orbitals)
    sector_capacity = min(nalpha, nbeta, norb - nalpha, norb - nbeta)
    sector_consistent = bool(
        nalpha + nbeta == int(hamiltonian.total_electrons)
        and nalpha - nbeta == int(hamiltonian.spin)
    )
    return [
        {
            'id': 'block2_energy_change',
            'category': 'convergence',
            'observed': convergence.get('final_energy_change'),
            'operator': 'less_than_or_equal',
            'limit': float(getattr(config, 'energy_tolerance')),
            'required': True,
            'source': 'convergence.final_energy_change',
        },
        {
            'id': 'block2_discarded_weight',
            'category': 'convergence',
            'observed': convergence.get('final_discarded_weight'),
            'operator': 'less_than_or_equal',
            'limit': float(getattr(config, 'discarded_weight_tolerance')),
            'required': True,
            'source': 'convergence.final_discarded_weight',
        },
        {
            'id': 'block2_total_energy_finite',
            'category': 'constraint',
            'observed': result.get('energy'),
            'operator': 'finite',
            'required': True,
            'source': 'energy',
        },
        {
            'id': 'block2_targeted_roots_complete',
            'category': 'constraint',
            'observed': len(state_energies),
            'operator': 'equal',
            'expected': int(getattr(config, 'nroots', len(state_energies)) or 1),
            'required': True,
            'source': 'state_energies',
        },
        {
            'id': 'block2_state_sector_capacity',
            'category': 'constraint',
            'observed': sector_capacity,
            'operator': 'greater_than_or_equal',
            'limit': 0,
            'required': True,
            'source': 'hamiltonian.n_electrons',
        },
        {
            'id': 'block2_state_sector',
            'category': 'constraint',
            'observed': sector_consistent,
            'operator': 'equal',
            'expected': True,
            'required': True,
            'source': 'hamiltonian.n_electrons+spin',
        },
    ]


def finalize_block2_result_contract(
    result: Dict[str, Any],
    hamiltonian: ElectronicHamiltonian,
    config: Any,
    symmetry: str,
) -> Dict[str, Any]:
    """Attach the common molecular/model DMRG result contract in place."""

    state_energies = _float_list(result.get('state_energies') or [result['energy']])
    excitation_energies = _float_list(
        result.get('excitation_energies')
        or [value - state_energies[0] for value in state_energies]
    )
    requested_roots = int(getattr(config, 'nroots', len(state_energies)) or 1)
    convergence = result.get('convergence') if isinstance(result.get('convergence'), dict) else {}
    adaptive = (
        result.get('adaptive_schedule')
        if isinstance(result.get('adaptive_schedule'), dict)
        else {}
    )
    error_estimate = (
        result.get('energy_error_estimate')
        if isinstance(result.get('energy_error_estimate'), dict)
        else {}
    )
    ground_occupations = result.get('natural_occupations')
    ground_occupations = (
        _float_list(ground_occupations)
        if isinstance(ground_occupations, (list, tuple))
        else None
    )
    root_signatures = []
    for root, energy in enumerate(state_energies):
        root_signatures.append({
            'root': root,
            'energy': float(energy),
            'excitation_energy': float(excitation_energies[root]),
            'natural_occupations': ground_occupations if root == 0 else None,
        })

    result.update({
        'result_contract_schema': BLOCK2_RESULT_CONTRACT_SCHEMA,
        'requested_root_count': requested_roots,
        'computed_root_count': len(state_energies),
        'targeted_roots_complete': len(state_energies) == requested_roots,
        'state_sector': block2_state_sector(hamiltonian, symmetry),
        'state_energies': state_energies,
        'excitation_energies': excitation_energies,
        'root_signatures': root_signatures,
        'final_bond_dimension': adaptive.get('final_bond_dimension'),
        'maximum_bond_dimension': adaptive.get('maximum_bond_dimension'),
        'final_discarded_weight': convergence.get('final_discarded_weight'),
        'final_energy_change': convergence.get('final_energy_change'),
        'sweeps_completed': convergence.get('sweeps_completed'),
        'estimated_absolute_energy_error': error_estimate.get('estimated_absolute_error'),
        'checkpoint_available': bool(
            isinstance(result.get('checkpoint_manifest'), dict)
            and result['checkpoint_manifest'].get('files')
        ),
        'result_scope': {
            'states': 'targeted_low_energy_roots',
            'rdm1': 'ground_state' if getattr(config, 'compute_1rdm', True) else 'not_requested',
            'rdm2': 'ground_state' if getattr(config, 'compute_2rdm', False) else 'not_requested',
            'diagnostics': 'ground_state',
            'complete_spectrum': False,
        },
    })
    result['quality_checks'] = _block2_quality_checks(result, hamiltonian, config)
    return result


def block2_compact_result_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    fields = (
        'result_contract_schema',
        'requested_root_count',
        'computed_root_count',
        'targeted_roots_complete',
        'state_sector',
        'state_energies',
        'excitation_energies',
        'root_signatures',
        'final_bond_dimension',
        'maximum_bond_dimension',
        'final_discarded_weight',
        'final_energy_change',
        'sweeps_completed',
        'estimated_absolute_energy_error',
        'checkpoint_available',
        'result_scope',
        'recovery_recommendation',
        'quality_checks',
    )
    return {
        field: copy.deepcopy(result[field])
        for field in fields
        if result.get(field) is not None
    }


__all__ = [
    'BLOCK2_RESULT_CONTRACT_SCHEMA',
    'block2_compact_result_fields',
    'block2_state_sector',
    'finalize_block2_result_contract',
]
