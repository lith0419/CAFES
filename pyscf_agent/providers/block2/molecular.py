from __future__ import annotations

import copy
from typing import Any

from ...backend.electronic_hamiltonian import ElectronicHamiltonian


def _active_electron_counts(nelecas: Any, spin: int) -> tuple[int, int]:
    if isinstance(nelecas, (list, tuple)):
        if len(nelecas) != 2:
            raise ValueError('Active-space electron counts must contain nalpha and nbeta.')
        counts = (int(nelecas[0]), int(nelecas[1]))
    else:
        total = int(nelecas)
        if (total + int(spin)) % 2:
            raise ValueError('Active-space electron count and spin have incompatible parity.')
        counts = ((total + int(spin)) // 2, (total - int(spin)) // 2)
    if any(value < 0 for value in counts):
        raise ValueError('Active-space electron counts must be nonnegative.')
    if counts[0] - counts[1] != int(spin):
        raise ValueError('Active-space electron spin must match the molecular spin for a paired inactive core.')
    return counts


def build_active_space_hamiltonian_from_integrals(
    h1e: Any,
    g2e: Any,
    *,
    ncas: int,
    nelecas: Any,
    spin: int,
    core_energy: float = 0.0,
    orbital_symmetries: Any = None,
    source: str = 'molecular_active_space_integrals',
    metadata: Any = None,
) -> ElectronicHamiltonian:
    """Build the portable solver contract used by PySCF CAS optimizers."""

    electron_counts = _active_electron_counts(nelecas, int(spin))
    symmetry_values = orbital_symmetries
    if symmetry_values is None or len(symmetry_values) == 0:
        symmetry_values = [0] * int(ncas)
    symmetries = tuple(int(value) for value in symmetry_values)
    hamiltonian = ElectronicHamiltonian(
        n_orbitals=int(ncas),
        n_electrons=electron_counts,
        spin=int(spin),
        core_energy=float(core_energy),
        h1e=h1e,
        g2e=g2e,
        orbital_symmetries=symmetries,
        source=str(source),
        metadata=copy.deepcopy(metadata or {}),
    )
    hamiltonian.validate()
    return hamiltonian


def build_molecular_active_space_hamiltonian(
    mf: Any,
    mo_coeff: Any,
    *,
    ncas: int,
    nelecas: Any,
    active_orbital_indices: Any = None,
    orbital_provenance: Any = None,
) -> ElectronicHamiltonian:
    """Convert sorted RHF/ROHF orbitals into a block2-ready active Hamiltonian."""
    from pyblock2._pyscf.ao2mo import integrals  # pylint: disable=import-outside-toplevel

    mol = mf.mol
    electron_counts = _active_electron_counts(nelecas, int(mol.spin))
    active_electrons = sum(electron_counts)
    inactive_electrons = int(mol.nelectron) - active_electrons
    if inactive_electrons < 0 or inactive_electrons % 2:
        raise ValueError('DMRG-CASCI requires the inactive molecular electrons to form paired core orbitals.')
    ncore = inactive_electrons // 2
    reference = copy.copy(mf)
    reference.mo_coeff = mo_coeff
    actual_ncas, actual_nelec, spin, ecore, h1e, g2e, orbital_symmetries = integrals.get_rhf_integrals(
        reference,
        ncore=ncore,
        ncas=int(ncas),
        pg_symm=False,
        g2e_symm=8,
    )
    if int(actual_ncas) != int(ncas) or int(actual_nelec) != active_electrons:
        raise ValueError('The generated block2 active-space Hamiltonian does not match the approved CAS size.')
    return ElectronicHamiltonian(
        n_orbitals=int(actual_ncas),
        n_electrons=electron_counts,
        spin=int(spin),
        core_energy=float(ecore),
        h1e=h1e,
        g2e=g2e,
        orbital_symmetries=tuple(int(value) for value in orbital_symmetries),
        source='molecular_active_space',
        metadata={
            'ncore': int(ncore),
            'active_orbital_indices': copy.deepcopy(active_orbital_indices or []),
            'reference': 'rohf' if int(mol.spin) else 'rhf',
            'orbital_processing': copy.deepcopy(orbital_provenance or {}),
        },
    )
