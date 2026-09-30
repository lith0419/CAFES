"""Optimize in the retained MO space and preserve full-space output identities."""
from __future__ import annotations

from typing import Any


def run_casscf_kernel(mc: Any, mo_coeff: Any) -> tuple[Any, dict]:
    """Omit only explicitly frozen external MOs from PySCF's integral space.

    All occupied and active columns remain, so core potentials, active indices
    and the allowed rotations are identical to the full frozen-MO calculation.
    PySCF still owns the integrals and orbital optimization. Checkpoints and
    the returned optimizer contain the full input column layout.
    """
    import numpy as np
    from pyscf import lib

    nmo = mo_coeff.shape[1]
    nocc = mc.ncore + mc.ncas
    frozen = mc.frozen
    frozen_indices = list(range(frozen)) if isinstance(frozen, (int, np.integer)) else list(frozen or [])
    excluded = sorted(i for i in frozen_indices if i >= nocc)
    excluded_set = set(excluded)
    keep = np.array([i for i in range(nmo) if i not in excluded_set], dtype=int)
    summary = {
        'implementation': 'pyscf_rectangular_mo_coeff',
        'full_nmo': int(nmo),
        'working_nmo': int(len(keep)),
        'excluded_frozen_virtual_indices': excluded,
        'working_to_full_indices': keep.tolist(),
        'output_mo_layout': 'full_input_columns',
    }
    if not excluded:
        return mc.kernel(mo_coeff), summary

    original = np.array(mo_coeff, copy=True)
    original_orbsym = getattr(mo_coeff, 'orbsym', None)
    if original_orbsym is None and mc.mol.symmetry:
        from pyscf import symm
        original_orbsym = symm.label_orb_symm(
            mc.mol, mc.mol.irrep_id, mc.mol.symm_orb, original,
            s=mc._scf.get_ovlp(),
        )

    def expand_mo(working):
        restored = original.copy()
        restored[:, keep] = working
        if original_orbsym is not None:
            labels = np.array(original_orbsym, copy=True)
            labels[keep] = np.asarray(getattr(working, 'orbsym', labels[keep]))
            restored = lib.tag_array(restored, orbsym=labels)
        return restored

    def expand_occupations(occupations):
        if occupations is None:
            return None
        restored = np.zeros(nmo)
        restored[keep] = occupations
        return restored

    def full_energies(full_mo, casdm1):
        # A Fock reconstructed from projected MO ERIs has no potential in the
        # excluded subspace. Evaluate the AO potential once for full energies.
        core = full_mo[:, :mc.ncore]
        active = full_mo[:, mc.ncore:nocc]
        density = 2 * core @ core.conj().T + active @ casdm1 @ active.conj().T
        vj, vk = mc.get_jk(mc.mol, density)
        fock = mc.get_hcore() + vj - 0.5 * vk
        return np.einsum('pi,pi->i', full_mo.conj(), fock @ full_mo).real

    working = np.array(original[:, keep], copy=True)
    if original_orbsym is not None:
        working = lib.tag_array(working, orbsym=np.asarray(original_orbsym)[keep])
    extrasym = getattr(mc, 'extrasym', None)
    original_dump = mc.dump_chk
    instance_dump = mc.__dict__.get('dump_chk')
    final_checkpoint_mo = None
    final_checkpoint_energies = None

    def dump_full_checkpoint(envs):
        # Native CASSCF supplies its iteration locals; do not mutate those.
        nonlocal final_checkpoint_mo, final_checkpoint_energies
        snapshot = dict(envs)
        key = 'mo' if 'mo' in snapshot else 'mo_coeff'
        snapshot[key] = expand_mo(envs[key])
        if envs.get('mo_energy') is not None:
            snapshot['mo_energy'] = full_energies(snapshot[key], envs['casdm1'])
            final_checkpoint_mo = envs[key]
            final_checkpoint_energies = snapshot['mo_energy']
        return original_dump(snapshot)

    mc.frozen = [i for i in frozen_indices if i < nocc]
    if extrasym is not None:
        mc.extrasym = np.asarray(extrasym)[keep]
    mc.dump_chk = dump_full_checkpoint
    lib.logger.info(mc, 'CASSCF integral space: %d -> %d MOs; %d frozen virtuals omitted',
                    nmo, len(keep), len(excluded))
    try:
        result = mc.kernel(working)
        optimized = mc.mo_coeff
        restored = expand_mo(optimized)
        if mc.mo_energy is not None:
            if optimized is final_checkpoint_mo:
                energies = final_checkpoint_energies
            else:
                dm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
                energies = full_energies(restored, dm1)
            mc.mo_energy = energies
        mc.mo_coeff = restored
        mc.mo_occ = expand_occupations(mc.mo_occ)
        return (*result[:3], mc.mo_coeff, mc.mo_energy), summary
    finally:
        mc.frozen = frozen
        if extrasym is not None:
            mc.extrasym = extrasym
        if instance_dump is None:
            del mc.dump_chk
        else:
            mc.dump_chk = instance_dump
