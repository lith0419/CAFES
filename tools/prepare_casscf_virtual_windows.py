"""Prepare nested CASSCF rotation spaces without deleting AOs or changing the CAS.

The supplied spin-summed CAS 1-RDM must be expressed in the supplied active MOs.
Only the external virtual block is semicanonicalized. An energy window controls
which external MOs may rotate during CASSCF; it is not an active-space selector.
"""
from __future__ import annotations

import numpy as np

HARTREE_TO_EV = 27.211386245988


def prepare_virtual_windows(mf, mo_coeff, casdm1, ncas, nelecas, frozen_core, windows_ev):
    from pyscf import mcscf

    mo = np.asarray(mo_coeff, dtype=float)
    dm1 = np.asarray(casdm1, dtype=float)
    mc = mcscf.CASCI(mf, ncas, nelecas)
    ncore, nocc = mc.ncore, mc.ncore + ncas
    if mo.ndim != 2 or mo.shape[0] != mf.mol.nao_nr() or mo.shape[1] <= nocc:
        raise ValueError('Expected complete spatial MOs with an external virtual space')
    if not np.isfinite(mo).all() or dm1.shape != (ncas, ncas) or not np.isfinite(dm1).all():
        raise ValueError('Invalid MO coefficients or active-space 1-RDM')
    if not np.allclose(dm1, dm1.T, atol=1e-8, rtol=0):
        raise ValueError('The active-space 1-RDM must be Hermitian')
    if abs(np.trace(dm1) - sum(mc.nelecas)) > 1e-6:
        raise ValueError('The active-space 1-RDM has the wrong electron count')
    core = list(frozen_core)
    if len(set(core)) != len(core) or any(type(i) is not int or not 0 <= i < ncore for i in core):
        raise ValueError('Frozen core indices must be distinct inactive spatial MO indices')
    if any(not np.isfinite(w) or w <= 0 for w in windows_ev):
        raise ValueError('Window widths must be positive finite energies in eV')
    overlap = mf.get_ovlp()
    orth_error = float(np.max(np.abs(mo.T @ overlap @ mo - np.eye(mo.shape[1]))))
    if orth_error > 1e-7:
        raise ValueError('Input orbitals are not orthonormal in the AO metric')

    # CASCI.get_fock uses 2*Ccore*Ccore.T + Cact*casdm1*Cact.T. No SCF or
    # correlated solver is run here. mf.get_jk retains the requested DF backend.
    fock = mc.get_fock(mo_coeff=mo, casdm1=dm1)
    active = mo[:, ncore:nocc]
    external = mo[:, nocc:]
    active_block = active.T @ fock @ active
    active_energies = np.linalg.eigvalsh((active_block + active_block.T) * .5)
    external_block = external.T @ fock @ external
    external_energies, rotation = np.linalg.eigh((external_block + external_block.T) * .5)
    prepared = mo.copy()
    prepared[:, nocc:] = external @ rotation
    upper = float(active_energies[-1])
    cases = []
    for width in [None, *windows_ev]:
        limit = None if width is None else upper + float(width) / HARTREE_TO_EV
        # The tiny tolerance avoids roundoff-dependent splitting at the boundary.
        keep = np.ones(len(external_energies), dtype=bool) if limit is None else external_energies <= limit + 1e-10
        frozen = sorted(core + (np.flatnonzero(~keep) + nocc).tolist())
        cases.append({
            'label': 'full' if width is None else f'{width:g}eV',
            'window_ev': width, 'upper_cutoff_hartree': limit,
            'external_kept': int(keep.sum()), 'external_frozen': int((~keep).sum()),
            'frozen_orbital_indices': frozen,
            'unfrozen_spatial_orbitals': mo.shape[1] - len(frozen),
        })
    # CASCI itself has no rotation mask; use the native CASSCF mask for counts.
    optimizer = mcscf.CASSCF(mf, ncas, nelecas)
    for case in cases:
        case['independent_orbital_rotations'] = int(optimizer.uniq_var_indices(
            mo.shape[1], ncore, ncas, case['frozen_orbital_indices'] or None).sum())
    final_error = float(np.max(np.abs(prepared.T @ overlap @ prepared - np.eye(mo.shape[1]))))
    if final_error > 1e-7:
        raise ValueError('Prepared orbitals lost orthonormality')
    return prepared, {
        'energy_reference': 'max eigenvalue of effective Fock active block at the common initial CAS density',
        'virtual_basis': 'eigenvectors of the external Fock block; occupied and active columns unchanged',
        'selection_fixed_throughout_optimization': True,
        'nmo': mo.shape[1], 'ncore': ncore, 'ncas': ncas,
        'frozen_core_count': len(core), 'unfrozen_inactive_count': ncore - len(core),
        'external_count': mo.shape[1] - nocc,
        'active_electrons': float(np.trace(dm1)),
        'total_electrons': float(2 * ncore + np.trace(dm1)),
        'input_orthonormality_max_error': orth_error,
        'prepared_orthonormality_max_error': final_error,
        'active_fock_eigenvalues_hartree': active_energies.tolist(),
        'active_upper_hartree': upper,
        'external_fock_eigenvalues_hartree': external_energies.tolist(),
        'cases': cases,
    }
