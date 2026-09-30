from __future__ import annotations

import copy
import itertools
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


class FragmentLocalCorrelationPotential:
    """A local correlation potential parameterized only on fragment blocks."""

    def __init__(
        self,
        value: Any,
        fragments: Sequence[Mapping[str, Any]],
        *,
        restricted: bool,
        tied: bool,
    ) -> None:
        matrix = np.asarray(value, dtype=float)
        if matrix.ndim != 3 or matrix.shape[0] != 2 or matrix.shape[1] != matrix.shape[2]:
            raise ValueError(
                'Fragment-local correlation potential requires shape (2, norb, norb)'
            )
        blocks = [
            tuple(int(index) for index in fragment['orbital_indices'])
            for fragment in fragments
        ]
        if not blocks or any(not block for block in blocks):
            raise ValueError('Fragment-local correlation potential requires non-empty fragments')
        if tied and len({len(block) for block in blocks}) != 1:
            raise ValueError('Translation-tied fragments must have the same orbital count')
        flattened = [index for block in blocks for index in block]
        if len(flattened) != len(set(flattened)):
            raise ValueError('Fragment-local correlation potential requires disjoint fragments')
        if min(flattened) < 0 or max(flattened) >= matrix.shape[-1]:
            raise ValueError('Fragment orbital index is outside the correlation-potential space')

        self.restricted = bool(restricted)
        self.bogoliubov = False
        self.bogo_res = False
        self.local = True
        self.is_vcor_kpts = False
        self.diag_idx = None
        self._base_value = np.array(matrix, copy=True)
        self._blocks = blocks
        self._tied = bool(tied)
        self._parameter_groups = 1 if tied else len(blocks)
        self._pairs = [
            tuple(itertools.combinations_with_replacement(range(len(blocks[0])), 2))
            if tied
            else tuple(itertools.combinations_with_replacement(range(len(block)), 2))
            for block in (blocks[:1] if tied else blocks)
        ]
        self._gradient = None
        self.param = self._parameters_from_value(matrix)
        self.value = self.evaluate()

    @property
    def blocks(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(self._blocks)

    def _parameters_from_value(self, value: Any) -> np.ndarray:
        matrix = np.asarray(value, dtype=float)
        spin_count = 1 if self.restricted else 2
        parameters: List[float] = []
        for group_index in range(self._parameter_groups):
            group_blocks = self._blocks if self._tied else [self._blocks[group_index]]
            for spin in range(spin_count):
                for local_i, local_j in self._pairs[group_index]:
                    values = [
                        0.5 * (
                            matrix[spin, block[local_i], block[local_j]]
                            + matrix[spin, block[local_j], block[local_i]]
                        )
                        for block in group_blocks
                    ]
                    parameters.append(float(np.mean(values)))
        return np.asarray(parameters, dtype=float)

    def evaluate(self) -> np.ndarray:
        matrix = np.array(self._base_value, copy=True)
        spin_count = 1 if self.restricted else 2
        position = 0
        for group_index in range(self._parameter_groups):
            group_blocks = self._blocks if self._tied else [self._blocks[group_index]]
            for spin in range(spin_count):
                for local_i, local_j in self._pairs[group_index]:
                    value = float(self.param[position])
                    position += 1
                    target_spins = (0, 1) if self.restricted else (spin,)
                    for block in group_blocks:
                        for target_spin in target_spins:
                            matrix[target_spin, block[local_i], block[local_j]] = value
                            matrix[target_spin, block[local_j], block[local_i]] = value
        return matrix

    def gradient(self) -> np.ndarray:
        if self._gradient is not None:
            return self._gradient
        gradients = np.zeros(
            (self.length(),) + self._base_value.shape,
            dtype=float,
        )
        spin_count = 1 if self.restricted else 2
        position = 0
        for group_index in range(self._parameter_groups):
            group_blocks = self._blocks if self._tied else [self._blocks[group_index]]
            for spin in range(spin_count):
                for local_i, local_j in self._pairs[group_index]:
                    target_spins = (0, 1) if self.restricted else (spin,)
                    for block in group_blocks:
                        for target_spin in target_spins:
                            gradients[position, target_spin, block[local_i], block[local_j]] = 1.0
                            gradients[position, target_spin, block[local_j], block[local_i]] = 1.0
                    position += 1
        self._gradient = gradients
        return gradients

    def update(self, parameters: Any) -> None:
        normalized = np.asarray(parameters, dtype=float)
        if normalized.shape != (self.length(),):
            raise ValueError(
                'Fragment-local correlation potential expected {0} parameters'.format(
                    self.length()
                )
            )
        self.param = np.array(normalized, copy=True)
        self.value = self.evaluate()

    def assign(self, value: Any) -> None:
        self.param = self._parameters_from_value(value)
        self.value = self.evaluate()

    def get(self, _cell: int = 0, kspace: bool = True) -> np.ndarray:
        del kspace
        return self.value

    def length(self) -> int:
        spin_count = 1 if self.restricted else 2
        return int(sum(len(pairs) for pairs in self._pairs) * spin_count)

    def is_local(self) -> bool:
        return True

    def islocal(self) -> bool:
        return True


def prepare_density_fit_lattice(lattice: Any) -> Any:
    """Match FitVcorFull's Fock base to the lattice-HF Hamiltonian."""

    if not lattice.use_hcore_as_emb_ham:
        return lattice
    # FitVcorFull always reads getFock(), while lattice HF uses getH1()
    # in this mode. Keep the physical Fock cache for the interacting bath.
    fit_lattice = copy.copy(lattice)
    fit_lattice.fock_lo_k = lattice.getH1(kspace=True)
    fit_lattice.fock_lo_R = lattice.getH1(kspace=False)
    return fit_lattice


def _identity_embedding_basis(global_density: Any, lattice: Any) -> np.ndarray:
    spin_dimension = int(np.asarray(global_density).shape[0])
    cell_count = int(getattr(lattice, 'ncells', getattr(lattice, 'nkpts', 1)))
    basis = np.zeros(
        (spin_dimension, cell_count, lattice.nscsites, lattice.nscsites),
        dtype=float,
    )
    basis[:, 0] = np.eye(lattice.nscsites)
    return basis


def fit_fragment_local_correlation_potential(
    global_density: Any,
    lattice: Any,
    correlation_potential: Any,
    filling: Any,
    fragments: Sequence[Mapping[str, Any]],
    *,
    translation_tied: bool,
    slater: Any,
    beta: float,
    max_iterations: int,
) -> Tuple[Any, float, float, Dict[str, Any]]:
    """Fit only fragment-local correlated-density blocks.

    Translation-equivalent fragments share one parameter block. General finite
    graphs use a block-coordinate sweep with all other fragment potentials held
    fixed while a local density block is fitted.
    """

    density = np.asarray(global_density)
    identity_basis = _identity_embedding_basis(density, lattice)
    fit_lattice = prepare_density_fit_lattice(lattice)
    restricted = bool(getattr(correlation_potential, 'restricted', False))
    current_value = np.asarray(correlation_potential.get(), dtype=float)
    fit_records: List[Dict[str, Any]] = []

    if translation_tied:
        local_potential = FragmentLocalCorrelationPotential(
            current_value,
            fragments,
            restricted=restricted,
            tied=True,
        )
        fitted, error_begin, error_end = slater.FitVcorFull(
            density[:, 0],
            fit_lattice,
            identity_basis,
            local_potential,
            beta,
            filling,
            MaxIter=max_iterations,
            imp_fit=True,
            imp_idx=list(local_potential.blocks[0]),
            method='BFGS',
            ytol=1.0e-7,
            gtol=1.0e-5,
            test_grad=False,
            num_grad=bool(np.isinf(beta)),
        )
        current_value = np.asarray(fitted.get(), dtype=float)
        fit_records.append({
            'fragment_ids': [str(fragment['fragment_id']) for fragment in fragments],
            'error_begin': float(error_begin),
            'error_end': float(error_end),
            'shared_parameters': True,
        })
    else:
        for fragment in fragments:
            local_potential = FragmentLocalCorrelationPotential(
                current_value,
                [fragment],
                restricted=restricted,
                tied=False,
            )
            fitted, error_begin, error_end = slater.FitVcorFull(
                density[:, 0],
                fit_lattice,
                identity_basis,
                local_potential,
                beta,
                filling,
                MaxIter=max_iterations,
                imp_fit=True,
                imp_idx=list(local_potential.blocks[0]),
                method='BFGS',
                ytol=1.0e-7,
                gtol=1.0e-5,
                test_grad=False,
                num_grad=bool(np.isinf(beta)),
            )
            current_value = np.asarray(fitted.get(), dtype=float)
            fit_records.append({
                'fragment_ids': [str(fragment['fragment_id'])],
                'error_begin': float(error_begin),
                'error_end': float(error_end),
                'shared_parameters': False,
            })

    candidate = copy.deepcopy(correlation_potential)
    candidate.assign(current_value)
    begin_errors = [record['error_begin'] for record in fit_records]
    end_errors = [record['error_end'] for record in fit_records]
    fitting_summary = {
        'parameterization': (
            'translation_tied_fragment_blocks'
            if translation_tied
            else 'independent_fragment_blocks'
        ),
        'residual_aggregation': 'maximum_fragment_block_residual',
        'fragment_fits': fit_records,
    }
    return (
        candidate,
        float(max(begin_errors)),
        float(max(end_errors)),
        fitting_summary,
    )
