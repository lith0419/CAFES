"""Compile one Hamiltonian dataset contract into the existing Study layer."""

from __future__ import annotations

import copy
from typing import Any, Iterable, List

from computational_study_agent.datasets.hamiltonian.contracts import (
    HamiltonianDatasetSpec,
    MolecularGeometry,
)
from computational_study_agent.planner import build_study_plan
from computational_study_agent.schema import StudyPlan, StudySpec


_QH9_ELEMENT_SYMBOLS = {
    1: 'H',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F',
}


def build_hamiltonian_dataset_study_plan(
    dataset_spec: Any,
    seed_geometries: Iterable[MolecularGeometry],
) -> StudyPlan:
    """Build one molecular Study case per seed molecule.

    Each case executes a complete NVE trajectory.  Its ten sampled frames stay
    inside the case's typed MD artifacts and are assembled into dataset samples
    only after the Study report is available.
    """

    spec = (
        dataset_spec
        if isinstance(dataset_spec, HamiltonianDatasetSpec)
        else HamiltonianDatasetSpec.from_dict(dataset_spec)
    )
    geometries = list(seed_geometries)
    _validate_seed_geometries(spec, geometries)

    electronic_structure = spec.electronic_structure
    sampling = spec.molecular_dynamics
    runtime = {
        'max_cycle': 50,
        'verbose': 4,
        'scf_algorithm': 'standard',
        **copy.deepcopy(electronic_structure.scf_options),
    }
    molecular_dynamics = {
        'profile': spec.compatibility_profile,
        'ensemble': sampling.ensemble,
        'temperature_kelvin': sampling.temperature_kelvin,
        'time_step_au': sampling.time_step_au,
        'steps': sampling.steps,
        'sample_stride': sampling.sample_stride,
        'sample_offset': sampling.sample_offset,
        'ao_convention': spec.ao_convention,
        'store_velocities': True,
        'store_fock': True,
        'store_overlap': True,
    }
    base_task = {
        'task_type': 'molecular',
        'basis': electronic_structure.basis,
        'unit': spec.coordinate_unit,
        'charge': 0,
        'spin': 0,
        'symmetry': False,
        'method': electronic_structure.method,
        'restricted': electronic_structure.restricted,
        'xc': electronic_structure.xc,
        'job': 'molecular_dynamics',
        'outputs': ['trajectory'],
        'runtime': runtime,
        'molecular_dynamics': molecular_dynamics,
    }
    cases: List[dict] = []
    for case_index, geometry in enumerate(geometries):
        case_md = {
            **molecular_dynamics,
            'velocity_seed': sampling.velocity_seed_for_case(case_index),
        }
        cases.append({
            'label': geometry.molecule_id,
            'variables': {
                'dataset_id': spec.dataset_id,
                'molecule_id': geometry.molecule_id,
                'seed_geometry_id': geometry.geometry_id,
                'expected_sample_count': sampling.sampled_frame_count,
                'velocity_seed': case_md['velocity_seed'],
            },
            'request_updates': {
                'atom': _geometry_atom_text(geometry),
                'charge': geometry.charge,
                'spin': geometry.spin,
                'molecular_dynamics': case_md,
            },
        })

    study_spec = StudySpec(
        name=spec.name,
        objective='build_qh9_compatible_hamiltonian_dataset',
        system_type='molecular',
        base_task=base_task,
        case_design={'mode': 'cases', 'cases': cases},
        observables=['trajectory'],
        comparison={
            'mode': 'hamiltonian_dataset_assembly',
            'dataset_spec': spec.to_dict(),
            'group_key': 'molecule_id',
            'expected_case_count': spec.target_molecule_count,
            'expected_sample_count': spec.target_structure_count,
        },
        resource_policy={
            'review_work_estimates': True,
            'approved': False,
        },
    )
    return build_study_plan(study_spec)


def _validate_seed_geometries(
    spec: HamiltonianDatasetSpec,
    geometries: List[MolecularGeometry],
) -> None:
    if len(geometries) != spec.target_molecule_count:
        raise ValueError(
            'Expected exactly {0} seed geometries, received {1}.'.format(
                spec.target_molecule_count,
                len(geometries),
            )
        )
    if any(not isinstance(geometry, MolecularGeometry) for geometry in geometries):
        raise TypeError('seed_geometries must contain MolecularGeometry objects')
    molecule_ids = [geometry.molecule_id for geometry in geometries]
    if len(set(molecule_ids)) != len(molecule_ids):
        raise ValueError('Seed geometries must contain one unique molecule_id per trajectory.')
    for geometry in geometries:
        if geometry.coordinate_unit.lower() != spec.coordinate_unit.lower():
            raise ValueError(
                'Seed geometry {0} uses {1}; expected {2}.'.format(
                    geometry.molecule_id,
                    geometry.coordinate_unit,
                    spec.coordinate_unit,
                )
            )
        if geometry.charge != 0 or geometry.spin != 0:
            raise ValueError('QH9 seed geometries must be neutral closed-shell molecules.')
        unsupported = sorted(set(geometry.atomic_numbers).difference(_QH9_ELEMENT_SYMBOLS))
        if unsupported:
            raise ValueError(
                'QH9 seed geometry {0} contains unsupported atomic numbers: {1}.'.format(
                    geometry.molecule_id,
                    unsupported,
                )
            )


def _geometry_atom_text(geometry: MolecularGeometry) -> str:
    return '; '.join(
        '{0} {1}'.format(
            _QH9_ELEMENT_SYMBOLS[atomic_number],
            ' '.join(format(float(value), '.16g') for value in position),
        )
        for atomic_number, position in zip(geometry.atomic_numbers, geometry.positions)
    )


__all__ = ['build_hamiltonian_dataset_study_plan']
