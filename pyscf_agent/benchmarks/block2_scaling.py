from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .fixtures import hubbard_ring_spec as _hubbard_ring_spec
from .fixtures import block2_unavailable_outcome as _unavailable_outcome


def _linear_hydrogen_hamiltonian(atom_count: int, spacing_angstrom: float, basis: str):
    from pyscf import gto, lo, scf  # pylint: disable=import-outside-toplevel
    from ..providers.block2 import build_molecular_active_space_hamiltonian

    atom = '; '.join(
        'H 0 0 {0:.12g}'.format(index * spacing_angstrom)
        for index in range(int(atom_count))
    )
    mol = gto.M(
        atom=atom,
        basis=basis,
        unit='Angstrom',
        charge=0,
        spin=0,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('Hydrogen-chain RHF reference did not converge.')
    localized_coeff = lo.Boys(mol, mf.mo_coeff).kernel()
    hamiltonian = build_molecular_active_space_hamiltonian(
        mf,
        localized_coeff,
        ncas=int(atom_count),
        nelecas=int(atom_count),
        active_orbital_indices=list(range(int(atom_count))),
        orbital_provenance={
            'input_basis': 'canonical',
            'output_basis': 'boys_localized_active_space',
            'localization_method': 'boys',
            'localization_scope': 'active_space',
            'localization_applied': True,
        },
    )
    return mol, mf, hamiltonian


def _dmrg_scaling_options(
    max_bond_dimension: int,
    *,
    ordering: str = 'fiedler',
) -> Dict[str, Any]:
    dimension = int(max_bond_dimension)
    return {
        'preset': 'screening',
        'bond_dimensions': [max(16, dimension // 2)] * 3 + [dimension] * 13,
        'noises': [1e-4, 1e-5, 1e-6] + [0.0] * 13,
        'davidson_thresholds': [1e-8] * 3 + [1e-9] * 13,
        'sweeps': 16,
        'energy_tolerance': 1e-6,
        'discarded_weight_tolerance': 5e-4,
        'save_mps': False,
        'symmetry': 'su2',
        'orbital_ordering': ordering,
    }


def run_block2_hydrogen_chain_scaling(
    definition: Dict[str, Any],
    work_dir: Optional[Path],
) -> Dict[str, Any]:
    unavailable = _unavailable_outcome()
    if unavailable:
        return unavailable
    from pyscf import fci  # pylint: disable=import-outside-toplevel
    from ..providers.block2 import run_block2_dmrg

    root = work_dir or Path(tempfile.mkdtemp(prefix='pyscf-agent-block2-h-chain-'))
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for atom_count in (6, 8, 10):
        mol, mf, hamiltonian = _linear_hydrogen_hamiltonian(atom_count, 1.8, 'sto-6g')
        reference_started = time.monotonic()
        reference_energy, _vector = fci.FCI(mol, mf.mo_coeff).kernel()
        reference_seconds = time.monotonic() - reference_started
        dmrg_started = time.monotonic()
        dmrg = run_block2_dmrg(
            hamiltonian,
            _dmrg_scaling_options(128),
            scratch_directory=str(root / 'h{0}-m128'.format(atom_count)),
        )
        dmrg_seconds = time.monotonic() - dmrg_started
        rows.append({
            'atom_count': atom_count,
            'basis': 'sto-6g',
            'spacing_angstrom': 1.8,
            'orbital_basis': 'boys_localized_active_space',
            'bond_dimension': 128,
            'fci_energy_Ha': float(reference_energy),
            'block2_energy_Ha': float(dmrg['energy']),
            'energy_abs_error_Ha': abs(float(dmrg['energy']) - float(reference_energy)),
            'block2_converged': bool(dmrg.get('converged')),
            'final_energy_change_Ha': dmrg['convergence'].get('final_energy_change'),
            'final_discarded_weight': dmrg['convergence'].get('final_discarded_weight'),
            'fci_wall_seconds': reference_seconds,
            'block2_wall_seconds': dmrg_seconds,
            'orbital_ordering': dmrg.get('orbital_ordering'),
        })

    _mol, _mf, h20_hamiltonian = _linear_hydrogen_hamiltonian(20, 1.8, 'sto-6g')
    h20_runs = []
    for dimension in (64, 128):
        started = time.monotonic()
        dmrg = run_block2_dmrg(
            h20_hamiltonian,
            _dmrg_scaling_options(dimension),
            scratch_directory=str(root / 'h20-m{0}'.format(dimension)),
        )
        h20_runs.append({
            'bond_dimension': dimension,
            'energy_Ha': float(dmrg['energy']),
            'converged': bool(dmrg.get('converged')),
            'final_energy_change_Ha': dmrg['convergence'].get('final_energy_change'),
            'final_discarded_weight': dmrg['convergence'].get('final_discarded_weight'),
            'wall_seconds': time.monotonic() - started,
            'orbital_ordering': dmrg.get('orbital_ordering'),
        })
    h20_delta = abs(h20_runs[-1]['energy_Ha'] - h20_runs[0]['energy_Ha'])
    tolerances = definition['tolerances']
    max_small_error = max(row['energy_abs_error_Ha'] for row in rows)
    return {
        'metrics': {
            'basis': 'sto-6g',
            'spacing_angstrom': 1.8,
            'orbital_basis': 'boys_localized_active_space',
            'small_chain_accuracy': rows,
            'max_small_chain_energy_abs_error_Ha': max_small_error,
            'h20_convergence': h20_runs,
            'h20_bond_dimension_delta_Ha': h20_delta,
        },
        'checks': [
            {
                'name': 'h6_h8_h10_fci_accuracy',
                'passed': (
                    max_small_error <= tolerances['small_chain_energy_abs_Ha']
                    and all(row['block2_converged'] for row in rows)
                ),
                'value': max_small_error,
                'limit': tolerances['small_chain_energy_abs_Ha'],
            },
            {
                'name': 'h20_variational_bond_dimension_convergence',
                'passed': (
                    h20_runs[-1]['energy_Ha'] <= h20_runs[0]['energy_Ha'] + 1e-8
                    and h20_delta <= tolerances['h20_bond_dimension_delta_Ha']
                    and all(row['converged'] for row in h20_runs)
                ),
                'value': h20_delta,
                'limit': tolerances['h20_bond_dimension_delta_Ha'],
            },
        ],
    }


def run_block2_hubbard_ring_scaling(
    definition: Dict[str, Any],
    work_dir: Optional[Path],
) -> Dict[str, Any]:
    unavailable = _unavailable_outcome()
    if unavailable:
        return unavailable
    from ..backend.model_hamiltonian.solver import run_model_hamiltonian_solver

    root = work_dir or Path(tempfile.mkdtemp(prefix='pyscf-agent-block2-hubbard-scaling-'))
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    fci8 = run_model_hamiltonian_solver(
        _hubbard_ring_spec(8),
        solver_name='fci',
        outputs=['energy'],
    )
    for site_count in (8, 12, 16):
        started = time.monotonic()
        dmrg = run_model_hamiltonian_solver(
            _hubbard_ring_spec(site_count),
            solver_name='block2_dmrg',
            outputs=['energy'],
            solver_options=_dmrg_scaling_options(128),
            scratch_directory=str(root / 'ring-{0}-m128'.format(site_count)),
        )
        rows.append({
            'site_count': site_count,
            'U': 4.0,
            't': -1.0,
            'bond_dimension': 128,
            'energy': float(dmrg['energy']),
            'converged': bool(dmrg.get('converged')),
            'final_energy_change_Ha': dmrg['dmrg_result']['convergence'].get('final_energy_change'),
            'final_discarded_weight': dmrg['dmrg_result']['convergence'].get('final_discarded_weight'),
            'wall_seconds': time.monotonic() - started,
            'orbital_ordering': dmrg['dmrg_result'].get('orbital_ordering'),
        })
    eight_site_error = abs(rows[0]['energy'] - float(fci8['energy']))
    started = time.monotonic()
    low_dimension = run_model_hamiltonian_solver(
        _hubbard_ring_spec(16),
        solver_name='block2_dmrg',
        outputs=['energy'],
        solver_options=_dmrg_scaling_options(64),
        scratch_directory=str(root / 'ring-16-m64'),
    )
    low_row = {
        'site_count': 16,
        'bond_dimension': 64,
        'energy': float(low_dimension['energy']),
        'converged': bool(low_dimension.get('converged')),
        'final_energy_change_Ha': low_dimension['dmrg_result']['convergence'].get('final_energy_change'),
        'final_discarded_weight': low_dimension['dmrg_result']['convergence'].get('final_discarded_weight'),
        'wall_seconds': time.monotonic() - started,
    }
    sixteen_delta = abs(rows[-1]['energy'] - low_row['energy'])
    medium_scale_weights = [
        float(row['final_discarded_weight'])
        for row in rows[1:]
        if row.get('final_discarded_weight') is not None
    ]
    max_medium_scale_weight = max(medium_scale_weights) if medium_scale_weights else None
    tolerances = definition['tolerances']
    return {
        'metrics': {
            'eight_site_fci_energy': fci8['energy'],
            'eight_site_energy_abs_error_Ha': eight_site_error,
            'size_scaling': rows,
            'sixteen_site_low_bond_dimension': low_row,
            'sixteen_site_bond_dimension_delta_Ha': sixteen_delta,
            'medium_scale_max_discarded_weight': max_medium_scale_weight,
        },
        'checks': [
            {
                'name': 'eight_site_fci_accuracy',
                'passed': eight_site_error <= tolerances['eight_site_energy_abs_Ha'],
                'value': eight_site_error,
                'limit': tolerances['eight_site_energy_abs_Ha'],
            },
            {
                'name': 'medium_scale_execution_and_variational_refinement',
                'passed': (
                    rows[-1]['energy'] <= low_row['energy'] + 1e-8
                    and rows[-1]['final_discarded_weight'] < low_row['final_discarded_weight']
                    and max_medium_scale_weight is not None
                    and max_medium_scale_weight <= tolerances['medium_scale_max_discarded_weight']
                    and all(row['converged'] for row in rows)
                    and low_row['converged']
                ),
                'value': {
                    'sixteen_site_energy_lowering_Ha': low_row['energy'] - rows[-1]['energy'],
                    'sixteen_site_discarded_weight_reduction': (
                        low_row['final_discarded_weight']
                        - rows[-1]['final_discarded_weight']
                    ),
                    'medium_scale_max_discarded_weight': max_medium_scale_weight,
                },
                'limit': {
                    'medium_scale_max_discarded_weight': tolerances[
                        'medium_scale_max_discarded_weight'
                    ],
                },
            },
        ],
    }


__all__ = [
    'run_block2_hubbard_ring_scaling',
    'run_block2_hydrogen_chain_scaling',
]
