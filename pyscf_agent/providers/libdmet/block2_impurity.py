from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from ...backend.electronic_hamiltonian import ElectronicHamiltonian
from ..block2.checkpoint import pin_restart_orbital_ordering
from ..block2.driver import run_block2_dmrg


DMET_BLOCK2_IMPURITY_SCHEMA = 'pyscf-agent.dmet-block2-impurity.v1'


def _real_scalar(value: Any, field_name: str) -> float:
    import numpy as np  # pylint: disable=import-outside-toplevel

    scalar = np.asarray(value).reshape(-1)[0]
    if abs(float(np.imag(scalar))) > 1.0e-9:
        raise ValueError('{0} must be real for the current block2 DMET bridge'.format(field_name))
    return float(np.real(scalar))


def _component_arrays(container: Any, key: str) -> list[Any]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    value = container[key]
    array = np.asarray(value)
    if array.ndim == 2 and key == 'cd':
        return [array]
    if array.ndim == 4 and key == 'ccdd':
        return [array]
    return [np.asarray(component) for component in value]


def _block2_hamiltonian(
    hamiltonian: Any,
    *,
    nelec: int,
    spin: int,
    restricted: bool,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ElectronicHamiltonian:
    h1_components = _component_arrays(hamiltonian.H1, 'cd')
    h2_components = _component_arrays(hamiltonian.H2, 'ccdd')
    norb = int(hamiltonian.norb)
    nalpha = (int(nelec) + int(spin)) // 2
    nbeta = (int(nelec) - int(spin)) // 2
    if nalpha + nbeta != int(nelec) or min(nalpha, nbeta) < 0:
        raise ValueError(
            'DMET impurity nelec={0} and libDMET Sz={1} define an invalid block2 sector'.format(
                nelec,
                spin,
            )
        )

    if restricted:
        if len(h1_components) != 1 or len(h2_components) != 1:
            raise ValueError('Restricted DMET block2 impurities require one h1 and one h2 component')
        h1e: Any = h1_components[0]
        g2e: Any = h2_components[0]
        representation = 'spatial_orbital'
    else:
        if len(h1_components) != 2 or len(h2_components) != 3:
            raise ValueError(
                'Unrestricted DMET block2 impurities require alpha/beta h1 and '
                'alpha-alpha/beta-beta/alpha-beta h2 components'
            )
        h1e = (h1_components[0], h1_components[1])
        # libDMET stores unrestricted interactions as aa, bb, ab; pyblock2
        # consumes aa, ab, bb in its SZ quantum-chemistry interface.
        g2e = (h2_components[0], h2_components[2], h2_components[1])
        representation = 'unrestricted_spatial_orbital'

    result = ElectronicHamiltonian(
        n_orbitals=norb,
        n_electrons=(int(nalpha), int(nbeta)),
        spin=int(spin),
        core_energy=_real_scalar(hamiltonian.H0, 'DMET impurity core energy'),
        h1e=h1e,
        g2e=g2e,
        orbital_symmetries=tuple(0 for _ in range(norb)),
        representation=representation,
        source='libdmet_impurity_hamiltonian',
        metadata=dict(metadata or {}),
    )
    result.validate()
    return result


class Block2DmetImpuritySolver:
    """Expose the agent-managed pyblock2 driver through libDMET's solver API."""

    def __init__(
        self,
        *,
        restricted: bool,
        spin: int,
        options: Optional[Mapping[str, Any]] = None,
        scratch_directory: Optional[str] = None,
        solver_id: str = 'impurity-1',
        runner: Callable[..., Dict[str, Any]] = run_block2_dmrg,
    ) -> None:
        self.restricted = bool(restricted)
        self.Sz = int(spin)
        self.options = copy.deepcopy(dict(options or {}))
        self.scratch_directory = Path(
            scratch_directory or (Path.cwd() / 'block2-dmet-scratch')
        ).expanduser().resolve() / str(solver_id)
        self.scratch_directory.mkdir(parents=True, exist_ok=True)
        self._runner = runner
        self.optimized = False
        self.converged = False
        self.E: Optional[float] = None
        self.onepdm: Any = None
        self.twopdm: Any = None
        self._latest_arrays: Dict[str, Any] = {}
        self._latest_result: Dict[str, Any] = {}
        self._restart_manifest: Optional[Dict[str, Any]] = None
        self._fixed_orbital_order: Optional[list[int]] = None
        self._calls: list[Dict[str, Any]] = []

    def _run_options(self) -> Dict[str, Any]:
        options = copy.deepcopy(self.options)
        options.update({
            'nroots': 1,
            'compute_1rdm': True,
            'compute_2rdm': True,
            'rdm_root_scope': 'ground_state',
            'rdm1_root_scope': 'ground_state',
            'rdm2_root_scope': 'ground_state',
            'compute_entanglement': False,
            'compute_mutual_information': False,
            'compute_bipartite_entanglement': False,
            'compute_symmetry_analysis': False,
            'symmetry': 'su2' if self.restricted else 'sz',
            'save_mps': True,
        })
        if self._restart_manifest is not None:
            options.update({
                'restart_manifest': copy.deepcopy(self._restart_manifest),
                'restart_required': True,
                'restart_provenance': {
                    'selection': 'previous_dmet_impurity_iteration',
                    'source_solver_call': len(self._calls),
                },
            })
        if self._fixed_orbital_order is not None:
            options['orbital_ordering'] = 'manual'
            options['orbital_order'] = list(self._fixed_orbital_order)
        return options

    def _prepare_internal_restart(self, result: Mapping[str, Any]) -> None:
        ordering = result.get('orbital_ordering')
        ordering = ordering if isinstance(ordering, Mapping) else {}
        permutation = [int(value) for value in (ordering.get('permutation') or [])]
        if permutation and ordering.get('requested_method') != 'canonical':
            self._fixed_orbital_order = permutation
        manifest = result.get('checkpoint_manifest')
        if not isinstance(manifest, dict) or not manifest.get('files'):
            self._restart_manifest = None
            return
        self._restart_manifest = copy.deepcopy(manifest)
        if self._fixed_orbital_order is not None:
            pin_restart_orbital_ordering(
                self._restart_manifest,
                self._fixed_orbital_order,
            )

    def _set_density_matrices(self, arrays: Mapping[str, Any]) -> None:
        import numpy as np  # pylint: disable=import-outside-toplevel

        if self.restricted:
            rdm1 = arrays.get('rdm1')
            rdm2 = arrays.get('rdm2')
            if rdm1 is None or rdm2 is None:
                raise RuntimeError('Restricted block2 DMET requires ground-state 1- and 2-RDMs')
            # libDMET stores one spin component for restricted 1RDMs and
            # multiplies it by two when evaluating observables.
            self.onepdm = np.asarray(rdm1)[None, ...] * 0.5
            self.twopdm = np.asarray(rdm2)[None, ...]
            return

        required = (
            'rdm1_alpha',
            'rdm1_beta',
            'rdm2_alpha_alpha',
            'rdm2_alpha_beta',
            'rdm2_beta_beta',
        )
        missing = [key for key in required if arrays.get(key) is None]
        if missing:
            raise RuntimeError(
                'Unrestricted block2 DMET is missing spin-resolved RDM components: {0}'.format(
                    ', '.join(missing)
                )
            )
        self.onepdm = np.asarray([
            arrays['rdm1_alpha'],
            arrays['rdm1_beta'],
        ])
        # libDMET's unrestricted energy contraction uses aa, bb, ab order.
        self.twopdm = np.asarray([
            arrays['rdm2_alpha_alpha'],
            arrays['rdm2_beta_beta'],
            arrays['rdm2_alpha_beta'],
        ])

    def run(
        self,
        Ham: Any,
        nelec: Optional[int] = None,
        guess: Any = None,
        calc_rdm2: bool = False,
        **kwargs: Any,
    ) -> tuple[Any, float]:
        del guess, calc_rdm2
        if kwargs.get('Mu') is not None:
            raise ValueError('The block2 DMET bridge requires chemical potential shifts in Ham')
        if nelec is None:
            raise ValueError('block2 DMET impurity execution requires nelec')
        call_index = len(self._calls) + 1
        electronic_hamiltonian = _block2_hamiltonian(
            Ham,
            nelec=int(nelec),
            spin=self.Sz,
            restricted=self.restricted,
            metadata={
                'dmrg_role': 'dmet_impurity_solver',
                'dmrg_call': call_index,
            },
        )
        call_scratch = self.scratch_directory / 'call-{0:04d}'.format(call_index)
        result = self._runner(
            electronic_hamiltonian,
            self._run_options(),
            scratch_directory=str(call_scratch),
            compute_2rdm=True,
        )
        arrays = result.pop('_transient_dmrg_arrays', {})
        self._set_density_matrices(arrays)
        self.E = float(result['energy'])
        self.converged = bool(result.get('converged'))
        if not self.converged:
            convergence = result.get('convergence') or {}
            raise RuntimeError(
                'block2 DMET impurity solver did not converge: '
                'energy_change={0}, discarded_weight={1}'.format(
                    convergence.get('final_energy_change'),
                    convergence.get('final_discarded_weight'),
                )
            )
        self.optimized = True
        self._latest_arrays = dict(arrays)
        self._latest_result = copy.deepcopy(result)
        self._prepare_internal_restart(result)
        self._calls.append({
            'call': call_index,
            'energy': self.E,
            'converged': self.converged,
            'convergence': copy.deepcopy(result.get('convergence') or {}),
            'energy_error_estimate': copy.deepcopy(result.get('energy_error_estimate') or {}),
            'bond_dimension_plan': copy.deepcopy(result.get('bond_dimension_plan') or {}),
            'threading': copy.deepcopy(result.get('threading') or {}),
            'restart_applied': bool((result.get('restart') or {}).get('applied')),
        })
        return self.onepdm, self.E

    def run_dmet_ham(self, Ham: Any, **_kwargs: Any) -> float:
        if self.onepdm is None or self.twopdm is None:
            raise RuntimeError('block2 DMET density matrices are unavailable before solver execution')
        import numpy as np  # pylint: disable=import-outside-toplevel
        from libdmet.solver.scf import restore_Ham  # pylint: disable=import-outside-toplevel

        Ham = restore_Ham(Ham, 1, in_place=True)
        h1 = np.asarray(Ham.H1['cd'])
        h2 = np.asarray(Ham.H2['ccdd'])
        if self.restricted:
            energy_one = np.einsum('pq,qp', h1[0], self.onepdm[0]) * 2.0
            energy_two = np.einsum('pqrs,pqrs', h2[0], self.twopdm[0]) * 0.5
        else:
            energy_one = np.einsum('spq,sqp', h1, self.onepdm)
            energy_two = (
                0.5 * np.einsum('pqrs,pqrs', h2[0], self.twopdm[0])
                + 0.5 * np.einsum('pqrs,pqrs', h2[1], self.twopdm[1])
                + np.einsum('pqrs,pqrs', h2[2], self.twopdm[2])
            )
        return _real_scalar(Ham.H0 + energy_one + energy_two, 'DMET Hamiltonian energy')

    def summary(self) -> Dict[str, Any]:
        latest = self._latest_result
        return {
            'schema': DMET_BLOCK2_IMPURITY_SCHEMA,
            'provider': 'block2',
            'solver': 'block2_dmrg',
            'call_count': len(self._calls),
            'converged': bool(self.converged),
            'energy': self.E,
            'configuration': copy.deepcopy(latest.get('configuration') or {}),
            'convergence': copy.deepcopy(latest.get('convergence') or {}),
            'energy_error_estimate': copy.deepcopy(latest.get('energy_error_estimate') or {}),
            'bond_dimension_plan': copy.deepcopy(latest.get('bond_dimension_plan') or {}),
            'threading': copy.deepcopy(latest.get('threading') or {}),
            'orbital_ordering': copy.deepcopy(latest.get('orbital_ordering') or {}),
            'checkpoint_manifest': copy.deepcopy(latest.get('checkpoint_manifest') or {}),
            'calls': copy.deepcopy(self._calls),
        }

    def cleanup(self) -> None:
        return None


__all__ = [
    'Block2DmetImpuritySolver',
    'DMET_BLOCK2_IMPURITY_SCHEMA',
]
