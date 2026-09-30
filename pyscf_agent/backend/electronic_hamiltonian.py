from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


ELECTRONIC_HAMILTONIAN_SCHEMA = 'pyscf-agent.electronic-hamiltonian.v1'


@dataclass
class ElectronicHamiltonian:
    """Portable spatial-orbital Hamiltonian passed to correlated solvers."""

    n_orbitals: int
    n_electrons: Tuple[int, int]
    spin: int
    core_energy: float
    h1e: Any
    g2e: Any
    orbital_symmetries: Tuple[int, ...] = ()
    representation: str = 'spatial_orbital'
    source: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema: str = ELECTRONIC_HAMILTONIAN_SCHEMA

    def validate(self) -> None:
        import numpy as np  # pylint: disable=import-outside-toplevel

        norb = int(self.n_orbitals)
        if norb <= 0:
            raise ValueError('Electronic Hamiltonian requires at least one orbital.')
        if len(self.n_electrons) != 2 or any(int(value) < 0 for value in self.n_electrons):
            raise ValueError('Electronic Hamiltonian n_electrons must be (nalpha, nbeta).')
        if sum(int(value) for value in self.n_electrons) > 2 * norb:
            raise ValueError('Electronic Hamiltonian electron count exceeds the orbital capacity.')
        if int(self.spin) != int(self.n_electrons[0]) - int(self.n_electrons[1]):
            raise ValueError('Electronic Hamiltonian spin must equal nalpha - nbeta.')
        unrestricted = self.representation == 'unrestricted_spatial_orbital'
        h1e_values = (
            tuple(self.h1e)
            if unrestricted and isinstance(self.h1e, (list, tuple))
            else (self.h1e,)
        )
        if unrestricted and len(h1e_values) != 2:
            raise ValueError(
                'Unrestricted Electronic Hamiltonian h1e must contain alpha and beta components.'
            )
        for h1e_value in h1e_values:
            h1e = np.asarray(h1e_value)
            if h1e.shape != (norb, norb):
                raise ValueError('Electronic Hamiltonian h1e must have shape (norb, norb).')
            if not np.allclose(h1e, h1e.T.conj(), atol=1e-10):
                raise ValueError('Electronic Hamiltonian h1e must be Hermitian.')
        if isinstance(self.g2e, (list, tuple)) and len(self.g2e) in (3,):
            g2e_values = self.g2e
        else:
            g2e_values = (self.g2e,)
        if unrestricted and len(g2e_values) != 3:
            raise ValueError(
                'Unrestricted Electronic Hamiltonian g2e must contain alpha-alpha, '
                'alpha-beta, and beta-beta components.'
            )
        for g2e in g2e_values:
            shape = np.asarray(g2e).shape
            if shape not in (
                (norb, norb, norb, norb),
                (norb * (norb + 1) // 2, norb * (norb + 1) // 2),
                (norb * (norb + 1) // 2 * (norb * (norb + 1) // 2 + 1) // 2,),
            ):
                raise ValueError('Electronic Hamiltonian g2e has an unsupported shape: {0}.'.format(shape))
        if self.orbital_symmetries and len(self.orbital_symmetries) != norb:
            raise ValueError('Electronic Hamiltonian orbital_symmetries must contain one value per orbital.')

    @property
    def total_electrons(self) -> int:
        return int(self.n_electrons[0]) + int(self.n_electrons[1])

    def summary(self) -> Dict[str, Any]:
        return {
            'schema': self.schema,
            'n_orbitals': int(self.n_orbitals),
            'n_electrons': [int(value) for value in self.n_electrons],
            'spin': int(self.spin),
            'core_energy': float(self.core_energy),
            'orbital_symmetries': [int(value) for value in self.orbital_symmetries],
            'representation': self.representation,
            'source': self.source,
            'metadata': dict(self.metadata),
        }
