"""PySCF provider implementation for one molecular-dynamics Task."""

from __future__ import annotations

import io
import math
from typing import Any, Dict, List

from ..contracts import TaskSpec
from ..registry import default_registry
from .ao_conventions import qh9_ao_metadata, transform_ao_matrix


MOLECULAR_MD_MANIFEST_SCHEMA = 'pyscf-agent.molecular-md-manifest.v1'
MOLECULAR_MD_FRAME_ARRAYS_SCHEMA = 'pyscf-agent.molecular-md-frame-arrays.v1'


class MolecularDynamicsExecutionError(RuntimeError):
    """A failed trajectory with its numerical evidence, not a partial dataset."""

    def __init__(self, message: str, *, raw_scf_output: str, diagnostics: Dict[str, Any]):
        super().__init__(message)
        self.raw_scf_output = raw_scf_output
        self.diagnostics = diagnostics


def run_molecular_dynamics_task(task_spec: TaskSpec) -> Dict[str, Any]:
    """Run one QH9-compatible NVE trajectory and return compact + transient data."""

    import numpy as np  # pylint: disable=import-outside-toplevel
    import pyscf  # pylint: disable=import-outside-toplevel
    from pyscf import dft, gto, md  # pylint: disable=import-outside-toplevel
    from pyscf.data import nist  # pylint: disable=import-outside-toplevel
    from pyscf.md.distributions import MaxwellBoltzmannVelocity  # pylint: disable=import-outside-toplevel

    molecular_dynamics = task_spec.molecular_dynamics
    scf_stdout = io.StringIO()
    mol = gto.Mole()
    mol.stdout = scf_stdout
    mol.build(
        dump_input=False,
        parse_arg=False,
        atom=task_spec.system.atom,
        basis=task_spec.system.basis,
        unit=task_spec.system.unit,
        charge=task_spec.system.charge,
        spin=task_spec.system.spin,
        symmetry=task_spec.system.symmetry,
        verbose=task_spec.runtime.verbose,
    )

    mf = dft.RKS(mol)
    mf.xc = task_spec.method.xc
    mf.stdout = scf_stdout
    mf.max_cycle = task_spec.runtime.max_cycle
    if task_spec.runtime.conv_tol is not None:
        mf.conv_tol = task_spec.runtime.conv_tol
    if task_spec.runtime.conv_tol_grad is not None:
        mf.conv_tol_grad = task_spec.runtime.conv_tol_grad
    if task_spec.runtime.grid_level is not None:
        mf.grids.level = task_spec.runtime.grid_level
    if task_spec.runtime.diis_space is not None:
        mf.diis_space = task_spec.runtime.diis_space
    if task_spec.runtime.scf_algorithm == 'newton':
        mf = mf.newton()
        mf.stdout = scf_stdout

    ao_basis = qh9_ao_metadata(mol, task_spec.system.basis or '')
    source_indices = ao_basis['source_indices']
    phase_signs = ao_basis['phase_signs']
    sampled_indices = frozenset(molecular_dynamics.sampled_frame_indices)
    frames: List[Dict[str, Any]] = []
    sampled_arrays: Dict[str, List[Any]] = {
        'positions_angstrom': [],
        'potential_energies_hartree': [],
        'kinetic_energies_hartree': [],
        'total_energies_hartree': [],
        'temperatures_kelvin': [],
        'times_au': [],
        'frame_indices': [],
        'fock_matrices': [],
        'overlap_matrices': [],
    }
    if molecular_dynamics.store_velocities:
        sampled_arrays['velocities_au'] = []

    integrator = md.NVE(
        mf,
        dt=molecular_dynamics.time_step_au,
        steps=molecular_dynamics.steps,
        verbose=task_spec.runtime.verbose,
    )
    last_scf_iteration: Dict[str, Any] = {}

    def record_scf_iteration(environment: Dict[str, Any]) -> None:
        # The scanner owns a copy of mf. Set its callback directly so every MD
        # geometry records the actual SCF iteration rather than stale mf state.
        last_scf_iteration.clear()
        last_scf_iteration['frame_index'] = int(integrator._step)
        if 'cycle' in environment:
            last_scf_iteration['cycle'] = int(environment['cycle']) + 1
        values = {
            'energy_hartree': environment.get('e_tot'),
            'energy_change_hartree': (
                environment['e_tot'] - environment['last_hf_e']
                if 'e_tot' in environment and 'last_hf_e' in environment else None
            ),
            'orbital_gradient_norm': environment.get('norm_gorb'),
            'density_change_norm': environment.get('norm_ddm'),
        }
        for key, value in values.items():
            last_scf_iteration[key] = (
                float(value) if value is not None and math.isfinite(float(value)) else None
            )

    integrator.scanner.base.callback = record_scf_iteration

    def capture_frame(environment: Dict[str, Any]) -> None:
        frame_index = int(environment['self']._step)  # pylint: disable=protected-access
        if frame_index not in sampled_indices:
            return
        frame = environment['current_frame']
        scanner = environment['scanner']
        current_mf = scanner.base
        if not bool(scanner.converged):
            raise RuntimeError('SCF did not converge for MD frame {0}.'.format(frame_index))

        fock = transform_ao_matrix(current_mf.get_fock(), source_indices, phase_signs)
        overlap = transform_ao_matrix(current_mf.get_ovlp(), source_indices, phase_signs)
        temperature = float(environment['self'].temperature())
        array_index = len(frames)
        positions_angstrom = np.asarray(frame.coord, dtype=float) * float(nist.BOHR)
        frames.append({
            'array_index': array_index,
            'frame_index': frame_index,
            'time_au': float(frame.time),
            'potential_energy_hartree': float(frame.epot),
            'kinetic_energy_hartree': float(frame.ekin),
            'total_energy_hartree': float(frame.etot),
            'temperature_kelvin': temperature,
            'converged': True,
            'positions_angstrom': positions_angstrom.tolist(),
        })
        sampled_arrays['positions_angstrom'].append(positions_angstrom)
        sampled_arrays['potential_energies_hartree'].append(float(frame.epot))
        sampled_arrays['kinetic_energies_hartree'].append(float(frame.ekin))
        sampled_arrays['total_energies_hartree'].append(float(frame.etot))
        sampled_arrays['temperatures_kelvin'].append(temperature)
        sampled_arrays['times_au'].append(float(frame.time))
        sampled_arrays['frame_indices'].append(frame_index)
        sampled_arrays['fock_matrices'].append(np.asarray(fock, dtype=float))
        sampled_arrays['overlap_matrices'].append(np.asarray(overlap, dtype=float))
        if molecular_dynamics.store_velocities:
            sampled_arrays['velocities_au'].append(np.asarray(frame.veloc, dtype=float))

    integrator.callback = capture_frame
    rng = np.random.Generator(np.random.PCG64(molecular_dynamics.velocity_seed))
    initial_velocities = MaxwellBoltzmannVelocity(
        mol,
        T=molecular_dynamics.temperature_kelvin,
        rng=rng,
    )
    try:
        integrator.kernel(
            veloc=initial_velocities,
            steps=molecular_dynamics.steps,
            dump_flags=False,
            verbose=task_spec.runtime.verbose,
        )
    except Exception as exc:
        # PySCF increments _step only after the frame and callback succeed.
        # This counts integrated frames even when sampling omits them.
        frame_index = int(integrator._step)
        diagnostics = {
            'schema': 'pyscf-agent.molecular-md-failure.v1',
            'exception_type': type(exc).__name__,
            'message': str(exc),
            'failed_frame_index': frame_index,
            'time_au': float(integrator.time),
            'completed_frame_count': frame_index,
            'sampled_frame_count': len(frames),
            'steps_requested': molecular_dynamics.steps,
            'velocity_seed': molecular_dynamics.velocity_seed,
            'scf_converged': bool(integrator.scanner.converged),
            'last_scf_iteration': dict(last_scf_iteration),
            'positions_angstrom': integrator.mol.atom_coords(unit='Angstrom').tolist(),
            'runtime': {
                'max_cycle': task_spec.runtime.max_cycle,
                'conv_tol': task_spec.runtime.conv_tol,
                'conv_tol_grad': task_spec.runtime.conv_tol_grad,
                'scf_algorithm': task_spec.runtime.scf_algorithm,
            },
        }
        raise MolecularDynamicsExecutionError(
            str(exc), raw_scf_output=scf_stdout.getvalue(), diagnostics=diagnostics,
        ) from exc

    if len(frames) != len(molecular_dynamics.sampled_frame_indices):
        raise RuntimeError(
            'Expected {0} sampled MD frames, captured {1}.'.format(
                len(molecular_dynamics.sampled_frame_indices),
                len(frames),
            )
        )

    arrays: Dict[str, Any] = {
        key: np.asarray(values)
        for key, values in sampled_arrays.items()
    }
    arrays['atomic_numbers'] = np.asarray(mol.atom_charges(), dtype=np.int64)
    arrays['ao_source_indices'] = np.asarray(source_indices, dtype=np.int64)
    arrays['ao_phase_signs'] = np.asarray(phase_signs, dtype=np.int8)

    array_metadata = {
        key: {
            'shape': list(value.shape),
            'dtype': str(value.dtype),
            'finite': bool(np.isfinite(value).all()),
        }
        for key, value in arrays.items()
    }
    fock_matrices = arrays['fock_matrices']
    overlap_matrices = arrays['overlap_matrices']
    if (
        fock_matrices.ndim != 3
        or overlap_matrices.shape != fock_matrices.shape
        or fock_matrices.shape[1] != fock_matrices.shape[2]
    ):
        raise RuntimeError('Sampled Fock and overlap matrices have inconsistent shapes.')
    if not all(metadata['finite'] for metadata in array_metadata.values()):
        raise RuntimeError('Sampled MD arrays contain NaN or infinite values.')
    fock_max_asymmetry = float(np.max(np.abs(
        fock_matrices - np.swapaxes(fock_matrices, -1, -2)
    )))
    overlap_max_asymmetry = float(np.max(np.abs(
        overlap_matrices - np.swapaxes(overlap_matrices, -1, -2)
    )))
    symmetry_tolerance = 1e-10
    if max(fock_max_asymmetry, overlap_max_asymmetry) > symmetry_tolerance:
        raise RuntimeError(
            'Sampled Fock or overlap matrices exceed the symmetry tolerance.'
        )

    runtime = {
        'max_cycle': task_spec.runtime.max_cycle,
        'conv_tol': task_spec.runtime.conv_tol,
        'conv_tol_grad': task_spec.runtime.conv_tol_grad,
        'grid_level': task_spec.runtime.grid_level,
        'diis_space': task_spec.runtime.diis_space,
        'scf_algorithm': task_spec.runtime.scf_algorithm,
    }
    reference_runtime = default_registry().capability(
        'molecular_dynamics', namespace='molecular.job',
    ).metadata['profiles']['qh9']['runtime']
    reference_deviations = [
        {
            'field': 'runtime.{0}'.format(field),
            'qh9_reference': expected,
            'configured': runtime[field],
            'reason': 'campaign_runtime_tradeoff',
        }
        for field, expected in reference_runtime.items()
        if runtime[field] != expected
    ]
    trajectory = {
        'schema': MOLECULAR_MD_MANIFEST_SCHEMA,
        'ensemble': molecular_dynamics.ensemble,
        'temperature_kelvin': molecular_dynamics.temperature_kelvin,
        'time_step_au': molecular_dynamics.time_step_au,
        'steps_requested': molecular_dynamics.steps,
        'steps_completed': molecular_dynamics.steps,
        'sample_stride': molecular_dynamics.sample_stride,
        'sample_offset': molecular_dynamics.sample_offset,
        'sampled_frame_count': len(frames),
        'velocity_seed': molecular_dynamics.velocity_seed,
        'method': task_spec.method.name,
        'xc': task_spec.method.xc,
        'basis': task_spec.system.basis,
        'reference': 'rks',
        'runtime': runtime,
        'coordinate_unit': 'Angstrom',
        'atomic_numbers': [int(value) for value in mol.atom_charges()],
        'velocity_unit': 'Bohr/atomic_time',
        'energy_unit': 'Hartree',
        'time_unit': 'atomic_time',
        'ao_basis': ao_basis,
        'frames': frames,
        'array_metadata': array_metadata,
        'matrix_validation': {
            'passed': True,
            'symmetry_tolerance': symmetry_tolerance,
            'fock_max_asymmetry': fock_max_asymmetry,
            'overlap_max_asymmetry': overlap_max_asymmetry,
        },
        'versions': {
            'pyscf_runtime': str(pyscf.__version__),
            'qh9_reference_pyscf': '2.2.1',
        },
        'compatibility': {
            'profile': molecular_dynamics.profile,
            'scope': 'method_basis_sampling_and_ao_convention',
            'bitwise_reproduction': False,
            'reference_deviations': reference_deviations,
        },
    }
    last_frame = frames[-1]
    analysis_text = (
        'Completed {0} NVE frames and retained {1} QH9-format-compatible samples; '
        'final sampled potential energy = {2:.12f} Hartree.'
    ).format(
        molecular_dynamics.steps,
        len(frames),
        last_frame['potential_energy_hartree'],
    )
    return {
        'task_type': 'molecular',
        'job': 'molecular_dynamics',
        'method': task_spec.method.name,
        'xc': task_spec.method.xc,
        'reference': 'rks',
        'basis': task_spec.system.basis,
        'nao': int(mol.nao_nr()),
        'nelectron': int(mol.nelectron),
        'converged': True,
        'reference_converged': True,
        'energy': last_frame['potential_energy_hartree'],
        'energy_unit': 'Hartree',
        'trajectory': trajectory,
        'requested_outputs': list(task_spec.analysis.outputs),
        'raw_scf_output': scf_stdout.getvalue(),
        'analysis_text': analysis_text,
        '_transient_molecular_md_arrays': arrays,
    }


__all__ = [
    'MolecularDynamicsExecutionError',
    'MOLECULAR_MD_FRAME_ARRAYS_SCHEMA',
    'MOLECULAR_MD_MANIFEST_SCHEMA',
    'run_molecular_dynamics_task',
]
