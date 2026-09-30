from __future__ import annotations

from typing import Any, Dict

from ...backend.electronic_hamiltonian import ElectronicHamiltonian


def build_model_electronic_hamiltonian(h1e: Any, g2e: Any, metadata: Dict[str, Any]) -> ElectronicHamiltonian:
    nelec = tuple(int(value) for value in metadata['nelec'])
    return ElectronicHamiltonian(
        n_orbitals=int(metadata['norb']),
        n_electrons=nelec,
        spin=int(nelec[0] - nelec[1]),
        core_energy=0.0,
        h1e=h1e,
        g2e=g2e,
        orbital_symmetries=tuple(0 for _index in range(int(metadata['norb']))),
        source='model_hamiltonian',
        metadata={
            'site_count': int(metadata.get('site_count') or metadata['norb']),
            'bond_count': int(metadata.get('bond_count') or 0),
        },
    )
