from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .checkpoint import pin_restart_orbital_ordering, read_checkpoint_manifest
from .config import normalize_block2_options
from .driver import run_block2_dmrg
from .molecular import build_active_space_hamiltonian_from_integrals


@dataclass
class Block2CIState:
    """Opaque CI handle returned to PySCF during a DMRG-CASSCF run."""

    call_index: int
    root_index: int
    energy: float
    rdm1: Any
    rdm2: Any
    dmrg_result: Dict[str, Any]
    arrays: Dict[str, Any]


class Block2FCISolverAdapter:
    """Expose the agent-managed block2 driver through PySCF's FCI protocol."""

    def __init__(
        self,
        mol: Any,
        options: Optional[Dict[str, Any]] = None,
        *,
        scratch_directory: Optional[str] = None,
        runner: Callable[..., Dict[str, Any]] = run_block2_dmrg,
    ) -> None:
        self.mol = mol
        self.stdout = getattr(mol, 'stdout', None)
        self.verbose = int(getattr(mol, 'verbose', 0) or 0)
        self.max_memory = float(getattr(mol, 'max_memory', 0) or 0)
        raw_nroots = (options or {}).get('nroots')
        self.nroots = int(1 if raw_nroots is None else raw_nroots)
        self.spin = int(getattr(mol, 'spin', 0) or 0)
        self.wfnsym = None
        self.converged = False
        self.options = copy.deepcopy(options or {})
        self._final_bond_dimension = normalize_block2_options(self.options).final_bond_dimension
        external_restart_manifest = read_checkpoint_manifest(self.options.pop('restart_manifest', None))
        external_restart_required = bool(self.options.pop('restart_required', True))
        external_restart_provenance = copy.deepcopy(
            self.options.pop('restart_provenance', {}) or {}
        )
        self.scratch_directory = scratch_directory
        self._runner = runner
        self._latest_state: Optional[Block2CIState] = None
        self._latest_states: list[Block2CIState] = []
        self._trace = []
        self._fixed_orbital_order: Optional[list[int]] = None
        self._internal_restart_manifest: Optional[Dict[str, Any]] = (
            copy.deepcopy(external_restart_manifest)
            if isinstance(external_restart_manifest, dict)
            else external_restart_manifest
        )
        if isinstance(self._internal_restart_manifest, dict):
            source_permutation = pin_restart_orbital_ordering(
                self._internal_restart_manifest
            )
            if source_permutation:
                self._fixed_orbital_order = source_permutation
        self._external_restart_pending = external_restart_manifest not in (None, '')
        self._restart_required = external_restart_required
        self._external_restart_provenance = external_restart_provenance
        self._requested_orbital_ordering = str(
            self.options.get('orbital_ordering') or 'canonical'
        ).strip().lower()
        self._requested_orbital_order = [
            int(value) for value in (self.options.get('orbital_order') or [])
        ]

    def dump_flags(self, verbose: Any = None) -> 'Block2FCISolverAdapter':
        del verbose
        return self

    def _iteration_options(self, *, final_analysis: bool = False) -> Dict[str, Any]:
        options = copy.deepcopy(self.options)
        options.pop('final_bond_dimension', None)
        options['compute_1rdm'] = True
        options['compute_2rdm'] = True
        # State-averaged CASSCF needs each root's RDM for orbital optimization.
        # Standalone DMRG analyses default to the ground state only.
        options['rdm_root_scope'] = 'all_states'
        options['rdm1_root_scope'] = 'all_states'
        options['rdm2_root_scope'] = 'all_states'
        options['save_mps'] = True
        if final_analysis and self._final_bond_dimension is not None:
            # A requested final M is an explicit solve, not an adaptive ceiling.
            options.update({
                'bond_dimensions': [self._final_bond_dimension],
                'max_bond_dimension': self._final_bond_dimension,
                'adaptive_schedule': False,
            })
        if not final_analysis:
            options.update({
                'adaptive_schedule': False,
                'compute_entanglement': False,
                'compute_mutual_information': False,
                'compute_bipartite_entanglement': False,
                'compute_symmetry_analysis': False,
            })
        if self._internal_restart_manifest is not None:
            first_external_call = not self._trace and self._external_restart_pending
            options.update({
                'restart_manifest': copy.deepcopy(self._internal_restart_manifest),
                'restart_required': self._restart_required if first_external_call else True,
                'restart_provenance': (
                    copy.deepcopy(self._external_restart_provenance)
                    if first_external_call
                    else {
                        'selection': 'previous_dmrg_casscf_solver_call',
                        'source_solver_call': len(self._trace),
                    }
                ),
            })
        if self._fixed_orbital_order is not None:
            options['orbital_ordering'] = 'manual'
            options['orbital_order'] = list(self._fixed_orbital_order)
        return options

    def _prepare_internal_restart(self, result: Dict[str, Any]) -> None:
        self._external_restart_pending = False
        ordering = result.get('orbital_ordering') if isinstance(result.get('orbital_ordering'), dict) else {}
        permutation = [int(value) for value in (ordering.get('permutation') or [])]
        if permutation and ordering.get('requested_method') != 'canonical':
            self._fixed_orbital_order = permutation
        manifest = result.get('checkpoint_manifest')
        if not isinstance(manifest, dict) or not manifest.get('files'):
            self._internal_restart_manifest = None
            return
        self._internal_restart_manifest = copy.deepcopy(manifest)
        if self._fixed_orbital_order is not None:
            pin_restart_orbital_ordering(
                self._internal_restart_manifest,
                self._fixed_orbital_order,
            )

    def kernel(
        self,
        h1e: Any,
        eri: Any,
        norb: int,
        nelec: Any,
        ci0: Any = None,
        ecore: float = 0.0,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        del ci0
        call_role = str(kwargs.pop('_call_role', 'active_space_energy'))
        final_analysis = bool(kwargs.pop('_final_analysis', False))
        raw_requested_roots = kwargs.get('nroots')
        requested_roots = int(self.nroots if raw_requested_roots is None else raw_requested_roots)
        if requested_roots <= 0:
            raise ValueError('DMRG-CASSCF nroots must be a positive integer.')
        orbsym = kwargs.get('orbsym')
        if orbsym is None:
            orbsym = getattr(self, 'orbsym', None)
        hamiltonian = build_active_space_hamiltonian_from_integrals(
            h1e,
            eri,
            ncas=int(norb),
            nelecas=nelec,
            spin=self.spin,
            core_energy=float(ecore),
            orbital_symmetries=orbsym,
            source='dmrg_casscf_solver_call',
            metadata={
                'solver_call': len(self._trace) + 1,
                'solver_call_role': call_role,
                'orbital_optimization': 'pyscf_casscf',
            },
        )
        iteration_options = self._iteration_options(final_analysis=final_analysis)
        iteration_options['nroots'] = requested_roots
        weights = getattr(self, 'weights', None)
        if weights is not None:
            iteration_options['state_average_weights'] = [float(value) for value in weights]
        result = self._runner(
            hamiltonian,
            iteration_options,
            scratch_directory=self.scratch_directory,
            compute_2rdm=True,
        )
        arrays = result.pop('_transient_dmrg_arrays', {})
        root_energies = list(result.get('state_energies') or [result['energy']])
        root_rdm1 = arrays.get('root_rdm1')
        root_rdm2 = arrays.get('root_rdm2')
        if requested_roots == 1:
            root_rdm1 = [arrays.get('rdm1')]
            root_rdm2 = [arrays.get('rdm2')]
        if (
            len(root_energies) != requested_roots
            or root_rdm1 is None
            or root_rdm2 is None
            or len(root_rdm1) != requested_roots
            or len(root_rdm2) != requested_roots
            or any(value is None for value in root_rdm1)
            or any(value is None for value in root_rdm2)
        ):
            raise RuntimeError('block2 DMRG-CASSCF requires spin-free 1- and 2-RDMs from every requested root.')
        ordering = result.get('orbital_ordering') if isinstance(result.get('orbital_ordering'), dict) else {}
        ordering['casscf_initial_request'] = self._requested_orbital_ordering
        ordering['casscf_initial_requested_order'] = list(self._requested_orbital_order)
        ordering['fixed_across_solver_calls'] = True
        states = [
            Block2CIState(
                call_index=len(self._trace) + 1,
                root_index=root,
                energy=float(root_energies[root]),
                rdm1=root_rdm1[root],
                rdm2=root_rdm2[root],
                dmrg_result=result,
                arrays=arrays,
            )
            for root in range(requested_roots)
        ]
        self._latest_states = states
        self._latest_state = states[0]
        self.converged = bool(result.get('converged'))
        average_weights = (
            iteration_options.get('state_average_weights')
            or [1.0 / requested_roots] * requested_roots
            if requested_roots > 1
            else []
        )
        state_average_energy = (
            sum(
                float(weight) * float(root_energy)
                for weight, root_energy in zip(average_weights, root_energies)
            )
            if requested_roots > 1
            else None
        )
        self._trace.append({
            'solver_call': states[0].call_index,
            'role': call_role,
            'energy': float(root_energies[0]) if requested_roots == 1 else state_average_energy,
            'state_average_energy': state_average_energy if requested_roots > 1 else None,
            'state_energies': [float(value) for value in root_energies],
            'state_average_weights': (
                [float(value) for value in average_weights]
                if requested_roots > 1
                else None
            ),
            'converged': self.converged,
            'bond_dimensions': copy.deepcopy((result.get('configuration') or {}).get('bond_dimensions', [])),
            'sweep_limit': (result.get('configuration') or {}).get('sweeps'),
            'convergence': copy.deepcopy(result.get('convergence') or {}),
            'restart_applied': bool((result.get('restart') or {}).get('applied')),
            'orbital_ordering': copy.deepcopy(ordering),
            'mpo_algorithm': copy.deepcopy(result.get('mpo_algorithm') or {}),
            'timings': copy.deepcopy(result.get('timings') or {}),
        })
        self._prepare_internal_restart(result)
        if requested_roots == 1:
            return states[0].energy, states[0]
        return [state.energy for state in states], states

    def approx_kernel(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        kwargs['_call_role'] = 'orbital_response'
        return Block2FCISolverAdapter.kernel(self, *args, **kwargs)

    def final_analysis(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        kwargs['_call_role'] = 'final_analysis'
        kwargs['_final_analysis'] = True
        return Block2FCISolverAdapter.kernel(self, *args, **kwargs)

    def _state(self, ci: Any) -> Block2CIState:
        if isinstance(ci, Block2CIState):
            return ci
        if self._latest_state is None:
            raise RuntimeError('No block2 CI state is available for reduced-density-matrix evaluation.')
        return self._latest_state

    def make_rdm12(self, ci: Any, norb: int, nelec: Any, **kwargs: Any) -> tuple[Any, Any]:
        del norb, nelec, kwargs
        state = self._state(ci)
        return state.rdm1, state.rdm2

    def make_rdm1(self, ci: Any, norb: int, nelec: Any, **kwargs: Any) -> Any:
        return self.make_rdm12(ci, norb, nelec, **kwargs)[0]

    @property
    def latest_state(self) -> Optional[Block2CIState]:
        return self._latest_state

    @property
    def latest_states(self) -> list[Block2CIState]:
        return list(self._latest_states)

    @property
    def trace(self) -> list[Dict[str, Any]]:
        return copy.deepcopy(self._trace)
