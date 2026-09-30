"""Paired RHF/CCSD numerical labels from a single converged calculation."""

from __future__ import annotations

from typing import Any, Dict

from .ao_conventions import qh9_ao_metadata, transform_ao_matrix


SCHEMA = 'pyscf-agent.ccsd-labels.v2'


def validate_ccsd_labels_request(task_spec: Any) -> None:
    if 'ccsd_labels' not in task_spec.analysis.outputs:
        return
    if not (
        task_spec.task_type == 'molecular'
        and task_spec.job.name == 'single_point'
        and task_spec.method.name == 'ccsd'
        and task_spec.method.restricted
        and task_spec.system.spin == 0
        and str(task_spec.system.basis).lower() == 'def2-svp'
        and not task_spec.density_fitting.enabled
    ):
        raise ValueError('ccsd_labels requires molecular RHF/CCSD single_point, def2-svp, spin=0, without density fitting.')


def capture_ccsd_labels(task_spec: Any, mf: Any, solver: Any) -> Dict[str, Any]:
    import numpy as np
    import pyscf

    validate_ccsd_labels_request(task_spec)
    mol = mf.mol
    ao = qh9_ao_metadata(mol, task_spec.system.basis)
    if not mf.converged or not solver.converged:
        raise ValueError('CCSD labels require converged HF and CCSD calculations.')
    if solver.frozen is not None:
        raise ValueError('CCSD labels currently require all-electron CCSD (frozen=None).')
    solver.solve_lambda()
    if not solver.converged_lambda:
        raise ValueError('CCSD labels require converged Lambda equations.')
    density = solver.make_rdm1(ao_repr=True, with_mf=True, with_frozen=True)
    source = {
        'F_HF': mf.get_fock(dm=mf.make_rdm1()),
        'D_CCSD': density,
        'S': mf.get_ovlp(),
    }
    arrays = {}
    residuals = {}
    for key, value in source.items():
        value = np.asarray(value)
        if np.iscomplexobj(value):
            raise ValueError('CCSD labels currently support real AO matrices only.')
        value = np.asarray(transform_ao_matrix(value, ao['source_indices'], ao['phase_signs']), dtype=np.float64)
        if value.shape != (ao['nao'], ao['nao']) or not np.isfinite(value).all():
            raise ValueError('Invalid CCSD label matrix: ' + key)
        residuals[key + '_hermiticity_max_abs'] = float(np.max(np.abs(value - value.T)))
        if residuals[key + '_hermiticity_max_abs'] > 1e-8:
            raise ValueError('Non-Hermitian CCSD label matrix: ' + key)
        arrays[key] = value
    electron_trace = float(np.einsum('ij,ji->', arrays['D_CCSD'], arrays['S']))
    residuals['density_electron_trace'] = electron_trace
    residuals['density_electron_trace_error'] = abs(electron_trace - mol.nelectron)
    if residuals['density_electron_trace_error'] > 1e-6:
        raise ValueError('CCSD density electron trace does not match the molecule.')
    # Keep the exact MO gauge used by the solver. Re-diagonalizing F later
    # cannot recover its phases or its choice within degenerate subspaces.
    nocc, nmo = int(solver.nocc), int(solver.nmo)
    orbital_arrays = {
        't1': (solver.t1, (nocc, nmo - nocc)),
        'C_HF': (solver.mo_coeff, (ao['nao'], nmo)),
        'epsilon_HF': (mf.mo_energy, (nmo,)),
        'mo_occ': (solver.mo_occ, (nmo,)),
    }
    for key, (value, shape) in orbital_arrays.items():
        value = np.asarray(value)
        if np.iscomplexobj(value) or value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Invalid CCSD orbital label: ' + key)
        arrays[key] = np.array(value, dtype=np.float64, copy=True)
    if not np.array_equal(arrays['C_HF'], mf.mo_coeff):
        raise ValueError('CCSD labels require the same HF reference orbitals and MO ordering.')
    if not (np.all(arrays['mo_occ'][:nocc] == 2) and np.all(arrays['mo_occ'][nocc:] == 0)
            and 2 * nocc == mol.nelectron):
        raise ValueError('CCSD labels require occupied-first closed-shell spatial orbitals.')
    arrays['C_HF'] = arrays['C_HF'][ao['source_indices'], :] * np.asarray(ao['phase_signs'])[:, None]
    residuals['mo_orthonormality_max_abs'] = float(np.max(np.abs(
        arrays['C_HF'].T @ arrays['S'] @ arrays['C_HF'] - np.eye(nmo))))
    if residuals['mo_orthonormality_max_abs'] > 1e-8:
        raise ValueError('HF MO coefficients are inconsistent with the saved AO overlap.')
    arrays.update(
        E_HF=np.asarray(mf.e_tot, dtype=np.float64),
        E_CCSD=np.asarray(solver.e_tot, dtype=np.float64),
        atomic_numbers=np.asarray(mol.atom_charges(), dtype=np.int64),
        positions_angstrom=np.asarray(mol.atom_coords(unit='Angstrom'), dtype=np.float64),
        ao_source_indices=np.asarray(ao['source_indices'], dtype=np.int64),
        ao_phase_signs=np.asarray(ao['phase_signs'], dtype=np.int8),
        occupied_mo_indices=np.arange(nocc, dtype=np.int64),
        virtual_mo_indices=np.arange(nocc, nmo, dtype=np.int64),
    )
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('CCSD labels contain non-finite numerical data.')
    metadata = {
        'schema': SCHEMA,
        'method': 'ccsd', 'reference': 'rhf', 'basis_name': task_spec.system.basis,
        'resolved_basis_definition': mol._basis,
        'charge': mol.charge, 'spin': mol.spin, 'electron_count': mol.nelectron,
        'nao': ao['nao'], 'ao_convention': ao['convention'],
        'nmo': nmo, 'nocc': nocc, 'nvir': nmo - nocc,
        't1_convention': 'Restricted spatial-orbital singles amplitudes t1[i,a]; occupied then virtual MO indices; dimensionless',
        'mo_convention': 'Exact RHF reference MO column order and phases used by CCSD; C_HF AO rows use the declared AO convention; no MO rephasing or rotation',
        'orbital_energy_unit': 'Hartree',
        'ao_labels': ao['labels'], 'atom_ao_slices': ao['atom_slices'],
        'ao_source_indices': ao['source_indices'], 'ao_phase_signs': ao['phase_signs'],
        'energy_unit': 'Hartree', 'position_unit': 'Angstrom',
        'energy_includes_nuclear_repulsion': True,
        'density_convention': 'spin-summed unrelaxed CCSD 1-RDM, includes reference; no orbital response',
        'F_HF_definition': 'RHF reference Fock; not a CCSD effective Hamiltonian',
        'reference_converged': True, 'solver_converged': True, 'lambda_converged': True,
        'all_electron': True, 'frozen': None, 'density_fitting': False, 'triples': False,
        'pyscf_version': pyscf.__version__,
        'scf_conv_tol': mf.conv_tol, 'scf_conv_tol_grad': mf.conv_tol_grad,
        'ccsd_conv_tol': solver.conv_tol, 'ccsd_conv_tol_normt': solver.conv_tol_normt,
        'numerical_checks': residuals,
        'tolerances': {'hermiticity_max_abs': 1e-8, 'density_electron_trace_error': 1e-6,
                       'mo_orthonormality_max_abs': 1e-8},
        'array_shapes': {key: list(value.shape) for key, value in arrays.items()},
        'provenance': task_spec.workflow.module_config.get('core.execution', {}).get('label_provenance', {}),
    }
    return {'metadata': metadata, 'arrays': arrays}


def write_ccsd_labels(state: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    import hashlib
    import json
    import numpy as np

    from .artifacts import retry_artifact_filename, serialize_npz_arrays, write_binary_artifact, write_json_artifact
    from ..runtime_identity import load_runtime_identity

    identity = load_runtime_identity()
    metadata = dict(payload['metadata'], run_id=state['run_id'],
                    runtime_release_id=identity.release_id if identity else None)
    archive = serialize_npz_arrays({
        'metadata_json': np.asarray(json.dumps(metadata, sort_keys=True)),
        **payload['arrays'],
    })
    ref = write_binary_artifact(
        state, 'ccsd_label_arrays',
        retry_artifact_filename('result-ccsd-labels', 'npz', state.get('retry_count', 0)),
        archive, mime_type='application/x-npz',
        description='Paired HF Fock, CCSD 1-RDM and singles amplitudes, HF orbitals, overlap, total energies and AO metadata',
    )
    if not ref:
        raise RuntimeError('Required CCSD label archive was not saved.')
    manifest = dict(metadata, data_artifact=ref, artifact_sha256=hashlib.sha256(archive).hexdigest())
    write_json_artifact(
        state, 'ccsd_label_manifest',
        retry_artifact_filename('result-ccsd-labels', 'json', state.get('retry_count', 0)),
        manifest, description='CCSD label definitions, numerical checks and archive checksum',
    )
    return manifest
