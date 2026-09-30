from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from pyscf_agent.backend.electronic_hamiltonian import ElectronicHamiltonian
from pyscf_agent.backend.execution import _effective_task_spec, _run_pyscf_task, _unconverged_execution_error
from pyscf_agent.backend.module_runtime import get_default_runtime_dispatcher
from pyscf_agent.backend.model_hamiltonian.solver import run_model_hamiltonian_solver
from pyscf_agent.contracts import task_spec_from_dict, task_spec_to_dict
from pyscf_agent.backend.result_artifacts import write_result_artifacts
from pyscf_agent.backend.validation import _validate_strong_correlation_spec
from pyscf_agent.registry import default_registry
from pyscf_agent.result_quality import evaluate_quality_checks
from pyscf_agent.providers.block2 import (
    Block2FCISolverAdapter,
    attach_optimized_orbitals,
    block2_options_for_outputs,
    block2_is_available,
    entanglement_active_space_recommendation,
    exact_bond_dimension_plan,
    load_optimized_orbitals,
    normalized_block2_provider_options,
    normalize_block2_options,
    resolve_block2_runtime_options,
    target_sector_dimension,
)
from pyscf_agent.workflow_gates import compile_task_gates
from pyscf_agent.workflow_modules import compile_task_workflow


def _hubbard_dimer_spec():
    return {
        'schema': 'pyscf-agent.model-hamiltonian.v1',
        'model': 'hubbard',
        'dimension': 1,
        'preset': 'chain',
        'boundary': 'open',
        'nelec': [1, 1],
        'sites': [
            {'id': 0, 'x': 0, 'y': 0, 'epsilon': 0, 'U': 4},
            {'id': 1, 'x': 1, 'y': 0, 'epsilon': 0, 'U': 4},
        ],
        'bonds': [
            {'id': 0, 'source': 0, 'target': 1, 't': -1, 'V': 0, 'effective_t': -1},
        ],
    }


def _small_dmrg_options():
    return {
        'preset': 'screening',
        'bond_dimensions': [16, 32, 32, 32, 32, 32],
        'noises': [1e-5, 1e-6, 0.0, 0.0, 0.0, 0.0],
        'davidson_thresholds': [1e-9] * 6,
        'sweeps': 6,
        'energy_tolerance': 1e-9,
        'discarded_weight_tolerance': 1e-7,
        'save_mps': False,
    }


class Block2ProviderContractTests(unittest.TestCase):
    def test_mpo_algorithm_is_explicit_and_validated(self):
        self.assertEqual(normalize_block2_options().mpo_algorithm, 'conventional')
        registry = default_registry()
        molecular = registry.capability('block2_dmrg', namespace='molecular.active_space_solver')
        model = registry.capability('block2_dmrg', namespace='model_hamiltonian.solver')
        self.assertEqual(molecular.metadata['default_mpo_algorithm'], 'conventional')
        self.assertEqual(model.metadata['solver_options']['mpo_algorithm']['default'], 'conventional')
        self.assertEqual(
            normalized_block2_provider_options({'mpo_algorithm': 'fast_bipartite'})['mpo_algorithm'],
            'fast_bipartite',
        )
        with self.assertRaisesRegex(ValueError, 'mpo_algorithm'):
            normalize_block2_options({'mpo_algorithm': 'unknown'})
        with self.assertRaisesRegex(ValueError, 'integral_cutoff'):
            normalize_block2_options({'integral_cutoff': -1})

    def test_compiled_cas_workflow_preserves_orbital_settings_and_solver_stages(self):
        for available in (True, False):
            for method, order, solver_override in (
                ('fiedler', [], {}),
                ('manual', [1, 0], {}),
                ('canonical', [], {}),
                ('fiedler', [], {'orbital_ordering': 'canonical', 'mpo_algorithm': 'conventional'}),
                ('fiedler', [], {'mpo_algorithm': 'fast_bipartite'}),
            ):
                with self.subTest(available=available, method=method, override=solver_override):
                    payload = task_spec_to_dict(task_spec_from_dict({
                        'task_type': 'molecular',
                        'system': {'atom': 'H 0 0 0; H 0 0 1.5', 'basis': 'sto-3g'},
                        'method': {'name': 'casscf'},
                        'active_space': {'enabled': True, 'ncas': 2, 'nelecas': 2, 'approved': True},
                        'orbital_processing': {'orbital_ordering': method, 'orbital_order': order},
                        'solver': {'name': 'block2_dmrg', 'options': {
                            'bond_dimensions': [600], 'final_bond_dimension': 2000,
                            'sweeps': 30, **solver_override,
                        }},
                        'analysis': {'outputs': ['energy']},
                    }))
                    original = copy.deepcopy(payload)
                    workflow = compile_task_workflow(payload, default_registry().modules_for(scope='task'))
                    with tempfile.TemporaryDirectory() as work_dir, mock.patch(
                        'pyscf_agent.providers.block2.module.block2_availability',
                        return_value={'provider': 'block2_dmrg', 'available': available},
                    ):
                        prepared = get_default_runtime_dispatcher().run_stage({
                            'task_spec': payload, 'workflow_configuration': workflow.to_dict(),
                            'module_execution_trace': [], 'work_dir': work_dir,
                            'run_id': 'cas-prepare', 'artifacts': [], 'logs': [], 'retry_count': 0,
                        }, 'task.prepare')
                    effective = _effective_task_spec(prepared)
                    options = effective.solver.options
                    self.assertEqual(options['orbital_ordering'], solver_override.get('orbital_ordering', method))
                    self.assertEqual(options['orbital_order'], order)
                    self.assertEqual(set(options['bond_dimensions']), {600})
                    self.assertEqual(options['final_bond_dimension'], 2000)
                    self.assertEqual(options['sweeps'], 30)
                    self.assertEqual(options['mpo_algorithm'], solver_override.get('mpo_algorithm', 'conventional'))
                    self.assertEqual(prepared['solver_provider']['available'], available)
                    self.assertEqual(payload, original)

    def test_electronic_hamiltonian_summary_is_stable(self):
        hamiltonian = ElectronicHamiltonian(
            n_orbitals=2,
            n_electrons=(1, 1),
            spin=0,
            core_energy=0.0,
            h1e=np.array([[0.0, -1.0], [-1.0, 0.0]]),
            g2e=np.zeros((2, 2, 2, 2)),
        )
        self.assertEqual(hamiltonian.summary(), hamiltonian.summary())
        self.assertEqual(hamiltonian.summary()['n_electrons'], [1, 1])
        self.assertNotIn('fingerprint', hamiltonian.summary())

    def test_dmrg_options_are_normalized_from_a_preset(self):
        options = normalize_block2_options({
            'preset': 'screening',
            'sweeps': 3,
            'compute_entanglement': True,
            'compute_symmetry_analysis': True,
            'nroots': 2,
        })
        self.assertEqual(options.preset, 'screening')
        self.assertEqual(options.sweeps, 3)
        self.assertTrue(options.compute_1rdm)
        self.assertTrue(options.compute_2rdm)
        self.assertTrue(options.compute_entanglement)
        self.assertTrue(options.compute_symmetry_analysis)
        self.assertEqual(options.nroots, 2)
        self.assertEqual(options.state_average_weights, [0.5, 0.5])
        self.assertEqual(options.rdm_root_scope, 'ground_state')
        self.assertTrue(options.adaptive_schedule)
        self.assertEqual(options.max_adaptive_stages, 2)
        self.assertEqual(options.max_bond_dimension, 800)
        self.assertGreater(options.adaptive_noise, 0.0)
        self.assertTrue(options.estimate_energy_error)
        self.assertTrue(options.bond_dimension_planning)

    def test_state_average_weights_are_validated(self):
        single_root = normalize_block2_options({
            'nroots': 1,
            'state_average_weights': [1.0],
        })
        self.assertEqual(single_root.excited_state_mode, 'state_specific')
        self.assertEqual(single_root.state_average_weights, [])
        options = normalize_block2_options({
            'nroots': 2,
            'state_average_weights': [0.25, 0.75],
        })
        self.assertEqual(options.state_average_weights, [0.25, 0.75])
        with self.assertRaisesRegex(ValueError, 'one value per requested root'):
            normalize_block2_options({'nroots': 2, 'state_average_weights': [1.0]})
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            normalize_block2_options({'nroots': 2, 'state_average_weights': [0.2, 0.2]})

    def test_task_spec_materializes_equal_multi_root_weights(self):
        task_spec = task_spec_from_dict({
            'task_type': 'molecular',
            'method': {'name': 'casscf'},
            'solver': {'name': 'block2_dmrg', 'options': {'nroots': 4}},
            'active_space': {
                'enabled': True,
                'target_method': 'casscf',
                'target_solver': 'block2_dmrg',
                'target_solver_options': {'nroots': 3},
            },
        })

        self.assertEqual(
            task_spec.solver.options['state_average_weights'],
            [0.25, 0.25, 0.25, 0.25],
        )
        self.assertEqual(
            task_spec.active_space.target_solver_options['state_average_weights'],
            [1.0 / 3.0] * 3,
        )

    def test_task_spec_removes_single_root_state_average_weights(self):
        task_spec = task_spec_from_dict({
            'task_type': 'molecular',
            'method': {'name': 'casscf'},
            'solver': {
                'name': 'block2_dmrg',
                'options': {'nroots': 1, 'state_average_weights': [1.0]},
            },
            'active_space': {
                'enabled': True,
                'target_method': 'casscf',
                'target_solver': 'block2_dmrg',
                'target_solver_options': {
                    'nroots': 1,
                    'state_average_weights': [1.0],
                },
            },
        })

        self.assertNotIn('state_average_weights', task_spec.solver.options)
        self.assertNotIn(
            'state_average_weights',
            task_spec.active_space.target_solver_options,
        )

    def test_model_task_keeps_multi_root_weights_internal_to_block2(self):
        task_spec = task_spec_from_dict({
            'task_type': 'model_hamiltonian',
            'solver': {'name': 'block2_dmrg', 'options': {'nroots': 4}},
        })

        self.assertEqual(task_spec.solver.options, {'nroots': 4})

    def test_model_multi_root_workflow_keeps_requested_root_count(self):
        task_spec = task_spec_from_dict({
            'task_type': 'model_hamiltonian',
            'method': {'name': 'hf'},
            'solver': {'name': 'block2_dmrg', 'options': {'nroots': 4}},
            'analysis': {'outputs': ['energy', 'excited_states']},
            'model_hamiltonian': {'spec': _hubbard_dimer_spec()},
        })
        payload = task_spec_to_dict(task_spec)
        workflow = compile_task_workflow(
            payload,
            default_registry().modules_for(scope='task'),
        )
        excited_node = next(
            item for item in workflow.nodes
            if item['module_id'] == 'solver.block2.excited_states'
        )
        self.assertEqual(excited_node['configuration'], {})

        with tempfile.TemporaryDirectory() as work_dir:
            state = {
                'task_spec': payload,
                'workflow_configuration': workflow.to_dict(),
                'module_execution_trace': [],
                'work_dir': work_dir,
                'run_id': 'block2-multi-root-prepare',
                'artifacts': [],
                'logs': [],
                'retry_count': 0,
            }
            prepared = get_default_runtime_dispatcher().run_stage(state, 'task.prepare')

        configuration = prepared['solver_provider']['configuration']
        self.assertEqual(configuration['nroots'], 4)
        self.assertEqual(configuration['state_average_weights'], [0.25] * 4)

    def test_task_spec_preserves_explicit_multi_root_weights_for_validation(self):
        task_spec = task_spec_from_dict({
            'task_type': 'molecular',
            'method': {'name': 'casscf'},
            'solver': {
                'name': 'block2_dmrg',
                'options': {'nroots': 2, 'state_average_weights': [1.0]},
            },
        })

        self.assertEqual(task_spec.solver.options['state_average_weights'], [1.0])
        self.assertIn(
            'invalid_block2_state_average_weights',
            {item['code'] for item in _validate_strong_correlation_spec(task_spec)},
        )

    def test_requested_gap_enables_two_targeted_roots(self):
        self.assertEqual(block2_options_for_outputs({}, ['energy', 'gap'])['nroots'], 2)
        self.assertEqual(block2_options_for_outputs({'nroots': 3}, ['gap'])['nroots'], 3)

    def test_rdm_root_scope_requires_an_explicit_supported_value(self):
        options = normalize_block2_options({'nroots': 2, 'rdm_root_scope': 'all_states'})
        self.assertEqual(options.rdm_root_scope, 'all_states')
        with self.assertRaisesRegex(ValueError, 'ground_state or all_states'):
            normalize_block2_options({'rdm_root_scope': 'excited_only'})

    def test_target_sector_dimension_distinguishes_su2_from_sz(self):
        self.assertEqual(target_sector_dimension(2, 1, 1, 'sz'), 4)
        self.assertEqual(target_sector_dimension(2, 1, 1, 'su2'), 3)
        self.assertEqual(target_sector_dimension(4, 2, 2, 'su2'), 20)

    def test_exact_bond_dimension_plan_caps_small_fixed_particle_spaces(self):
        hamiltonian = ElectronicHamiltonian(
            n_orbitals=4,
            n_electrons=(2, 2),
            spin=0,
            core_energy=0.0,
            h1e=np.zeros((4, 4)),
            g2e=np.zeros((4, 4, 4, 4)),
        )

        plan = exact_bond_dimension_plan(
            hamiltonian,
            normalize_block2_options({'preset': 'balanced'}),
        )

        self.assertEqual(plan['determinant_space_dimension'], 36)
        self.assertEqual(plan['target_root_sector_dimension'], 20)
        self.assertEqual(plan['coarse_fock_space_cap'], 16)
        self.assertEqual(plan['per_state_exact_bond_dimension_cap'], 16)
        self.assertEqual(plan['exact_bond_dimension_cap'], 16)
        self.assertEqual(plan['effective_bond_dimensions'], [16] * 10)
        self.assertEqual(plan['effective_max_bond_dimension'], 16)
        self.assertTrue(plan['cap_applied'])
        self.assertEqual(
            [item['schmidt_rank_cap'] for item in plan['cut_profile']],
            [4, 16, 4],
        )

    def test_bond_dimension_plan_respects_user_budget_below_exact_cap(self):
        hamiltonian = ElectronicHamiltonian(
            n_orbitals=8,
            n_electrons=(4, 4),
            spin=0,
            core_energy=0.0,
            h1e=np.zeros((8, 8)),
            g2e=np.zeros((8, 8, 8, 8)),
        )

        plan = exact_bond_dimension_plan(
            hamiltonian,
            normalize_block2_options({
                'bond_dimensions': [16, 32],
                'max_bond_dimension': 128,
            }),
        )

        self.assertEqual(plan['exact_bond_dimension_cap'], 256)
        self.assertEqual(plan['effective_bond_dimensions'], [16, 32])
        self.assertEqual(plan['effective_max_bond_dimension'], 128)
        self.assertFalse(plan['cap_applied'])

    def test_bond_dimension_planning_can_be_disabled_explicitly(self):
        hamiltonian = ElectronicHamiltonian(
            n_orbitals=2,
            n_electrons=(1, 1),
            spin=0,
            core_energy=0.0,
            h1e=np.zeros((2, 2)),
            g2e=np.zeros((2, 2, 2, 2)),
        )

        plan = exact_bond_dimension_plan(
            hamiltonian,
            normalize_block2_options({
                'bond_dimensions': [100, 200],
                'max_bond_dimension': 800,
                'bond_dimension_planning': False,
            }),
        )

        self.assertFalse(plan['enabled'])
        self.assertEqual(plan['effective_bond_dimensions'], [100, 200])
        self.assertEqual(plan['effective_max_bond_dimension'], 800)
        self.assertFalse(plan['cap_applied'])

    def test_bond_dimension_plan_reserves_capacity_for_multiple_roots(self):
        hamiltonian = ElectronicHamiltonian(
            n_orbitals=4,
            n_electrons=(2, 2),
            spin=0,
            core_energy=0.0,
            h1e=np.zeros((4, 4)),
            g2e=np.zeros((4, 4, 4, 4)),
        )

        plan = exact_bond_dimension_plan(
            hamiltonian,
            normalize_block2_options({'preset': 'balanced', 'nroots': 2}),
        )

        self.assertEqual(plan['nroots'], 2)
        self.assertEqual(plan['per_state_exact_bond_dimension_cap'], 16)
        self.assertEqual(plan['exact_bond_dimension_cap'], 32)
        self.assertEqual(plan['cap_kind'], 'conservative_multi_root')
        self.assertEqual(plan['effective_max_bond_dimension'], 32)

    def test_dmrg_adaptive_schedule_options_are_bounded(self):
        options = normalize_block2_options({
            'preset': 'screening',
            'bond_dimensions': [20, 40],
            'max_adaptive_stages': 1,
            'max_bond_dimension': 80,
            'adaptive_sweeps': 3,
            'bond_dimension_growth_factor': 1.5,
        })
        self.assertEqual(options.max_bond_dimension, 80)
        self.assertEqual(options.adaptive_sweeps, 3)
        self.assertEqual(options.bond_dimension_growth_factor, 1.5)
        with self.assertRaisesRegex(ValueError, 'cannot be smaller'):
            normalize_block2_options({
                'bond_dimensions': [100],
                'max_bond_dimension': 50,
            })

    def test_energy_error_estimate_uses_discarded_weight_extrapolation(self):
        from pyscf_agent.providers.block2.driver import _energy_error_estimate

        estimate = _energy_error_estimate(
            [-1.0, -1.05, -1.08, -1.095],
            [0.1, 0.05, 0.02, 0.005],
            enabled=True,
        )
        self.assertEqual(estimate['status'], 'available')
        self.assertEqual(
            estimate['method'],
            'linear_energy_vs_discarded_weight_extrapolation',
        )
        self.assertGreaterEqual(estimate['estimated_absolute_error'], 0.0)
        self.assertIn('not a rigorous error bound', estimate['interpretation'])

    def test_spin_free_rdm_entropy_fallback_has_physical_limits(self):
        from pyscf_agent.providers.block2.driver import (
            _single_orbital_entropy_from_spin_free_rdms,
        )

        rdm1 = np.diag([1.0, 2.0])
        rdm2 = np.zeros((2, 2, 2, 2))
        rdm2[1, 1, 1, 1] = 2.0

        entropy = _single_orbital_entropy_from_spin_free_rdms(rdm1, rdm2, spin_symmetric=True)

        self.assertAlmostEqual(entropy[0], np.log(2.0), places=12)
        self.assertAlmostEqual(entropy[1], 0.0, places=12)

    def test_adaptive_schedule_increases_bond_dimension_until_converged(self):
        from pyscf_agent.providers.block2.driver import _run_adaptive_schedule

        class FakeDriver:
            def __init__(self):
                self.calls = []
                self.stage = -1

            def dmrg(self, _mpo, _ket, **kwargs):
                self.stage += 1
                self.calls.append(kwargs)
                return (-1.01, -1.02)[self.stage]

            def get_dmrg_results(self):
                if self.stage == 0:
                    return [20, 20], [1e-2, 1e-3], [[-1.0], [-1.01]]
                return [40, 40], [1e-5, 1e-6], [[-1.019999], [-1.02]]

        config = normalize_block2_options({
            'preset': 'screening',
            'bond_dimensions': [20],
            'noises': [0.0],
            'davidson_thresholds': [1e-9],
            'sweeps': 2,
            'energy_tolerance': 1e-5,
            'discarded_weight_tolerance': 1e-4,
            'max_adaptive_stages': 1,
            'max_bond_dimension': 40,
            'adaptive_sweeps': 2,
        })
        driver = FakeDriver()

        energy, dimensions, discarded, energies, schedule = _run_adaptive_schedule(
            driver,
            object(),
            object(),
            config,
        )

        self.assertEqual(energy, -1.02)
        self.assertEqual(len(driver.calls), 2)
        self.assertEqual(driver.calls[1]['bond_dims'], [40, 40])
        self.assertGreater(driver.calls[1]['noises'][0], 0.0)
        self.assertEqual(dimensions.tolist(), [20, 20, 40, 40])
        self.assertEqual(len(discarded), 4)
        self.assertEqual(energies.shape, (4, 1))
        self.assertEqual(schedule['adaptive_stages_used'], 1)
        self.assertTrue(schedule['stages'][-1]['converged_after_stage'])
        self.assertFalse(schedule['budget_exhausted'])

    def test_entanglement_recommendation_expands_both_cas_boundaries(self):
        recommendation = entanglement_active_space_recommendation(
            {
                'natural_occupations': [1.8, 1.2, 0.8, 0.2],
                'entanglement_diagnostics': {
                    'single_orbital_entropy': [0.2, 0.5, 0.5, 0.2],
                    'max_mutual_information': 0.08,
                },
            },
            ncore=2,
            ncas=4,
            nelecas=[2, 2],
            nmo=8,
        )

        candidate = recommendation['candidate_active_space']
        self.assertEqual(recommendation['status'], 'approval_recommended')
        self.assertEqual(candidate['ncas'], 6)
        self.assertEqual(candidate['nelecas'], [3, 3])
        self.assertEqual(candidate['orbital_indices'], [1, 2, 3, 4, 5, 6])
        self.assertFalse(candidate['approved'])
        self.assertEqual(
            candidate['audit']['manual_approval']['status'],
            'requires_user_review',
        )

    def test_entanglement_recommendation_does_not_expand_closed_boundaries(self):
        recommendation = entanglement_active_space_recommendation(
            {
                'natural_occupations': [1.999, 1.5, 0.5, 0.001],
                'entanglement_diagnostics': {
                    'single_orbital_entropy': [0.3, 0.5, 0.5, 0.3],
                    'max_mutual_information': 0.1,
                },
            },
            ncore=1,
            ncas=4,
            nelecas=4,
            nmo=7,
        )

        self.assertEqual(recommendation['status'], 'not_recommended')
        self.assertNotIn('candidate_active_space', recommendation)

    def test_runtime_threads_follow_slurm_allocation(self):
        options, threading = resolve_block2_runtime_options(
            {'preset': 'screening'},
            environ={
                'SLURM_CPUS_PER_TASK': '12',
                'OMP_NUM_THREADS': '8',
            },
            pyscf_thread_reader=lambda: 4,
        )
        self.assertEqual(options['n_threads'], 12)
        self.assertEqual(options['n_mkl_threads'], 1)
        self.assertEqual(threading['n_threads_source'], 'SLURM_CPUS_PER_TASK')
        self.assertFalse(threading['nested_mkl'])

    def test_runtime_memory_reserves_part_of_slurm_allocation(self):
        options, resources = resolve_block2_runtime_options(
            {'preset': 'screening'},
            environ={
                'SLURM_CPUS_PER_TASK': '16',
                'SLURM_MEM_PER_NODE': '32768',
            },
            pyscf_thread_reader=lambda: 4,
        )
        self.assertEqual(options['stack_memory_bytes'], 24 * 1024 ** 3)
        self.assertEqual(resources['effective_stack_memory_bytes'], 24 * 1024 ** 3)
        self.assertEqual(resources['slurm_memory_allocation_bytes'], 32 * 1024 ** 3)
        self.assertEqual(resources['reserved_memory_bytes'], 8 * 1024 ** 3)
        self.assertEqual(resources['stack_memory_source'], 'SLURM_MEM_PER_NODE')

    def test_runtime_memory_supports_slurm_memory_per_cpu(self):
        options, resources = resolve_block2_runtime_options(
            {},
            environ={
                'SLURM_CPUS_PER_TASK': '4',
                'SLURM_MEM_PER_CPU': '2G',
            },
            pyscf_thread_reader=lambda: None,
        )
        self.assertEqual(resources['slurm_memory_allocation_bytes'], 8 * 1024 ** 3)
        self.assertEqual(options['stack_memory_bytes'], 6 * 1024 ** 3)
        self.assertEqual(resources['stack_memory_source'], 'SLURM_MEM_PER_CPU')

    def test_explicit_block2_memory_overrides_slurm_allocation(self):
        options, resources = resolve_block2_runtime_options(
            {'stack_memory_gb': 5},
            environ={'SLURM_MEM_PER_NODE': '32G'},
            pyscf_thread_reader=lambda: 1,
        )
        self.assertNotIn('stack_memory_bytes', options)
        self.assertEqual(resources['effective_stack_memory_bytes'], 5 * 1024 ** 3)
        self.assertEqual(resources['stack_memory_source'], 'solver_options')

    def test_explicit_block2_threads_override_runtime_allocation(self):
        options, threading = resolve_block2_runtime_options(
            {'n_threads': 6, 'n_mkl_threads': 2},
            environ={'SLURM_CPUS_PER_TASK': '12'},
            pyscf_thread_reader=lambda: 4,
        )
        self.assertEqual(options['n_threads'], 6)
        self.assertEqual(options['n_mkl_threads'], 2)
        self.assertEqual(threading['n_threads_source'], 'solver_options')
        self.assertTrue(threading['nested_mkl'])

    def test_runtime_threads_fall_back_to_pyscf_configuration(self):
        options, threading = resolve_block2_runtime_options(
            {},
            environ={'OMP_NUM_THREADS': '9'},
            pyscf_thread_reader=lambda: 5,
        )
        self.assertEqual(options['n_threads'], 5)
        self.assertEqual(threading['n_threads_source'], 'pyscf.lib.num_threads')

    def test_runtime_threads_use_blas_environment_without_pyscf_value(self):
        options, threading = resolve_block2_runtime_options(
            {},
            environ={'OMP_NUM_THREADS': '7'},
            pyscf_thread_reader=lambda: None,
        )
        self.assertEqual(options['n_threads'], 7)
        self.assertEqual(threading['n_threads_source'], 'OMP_NUM_THREADS')

    def test_provider_configuration_preserves_automatic_thread_resolution(self):
        options = normalized_block2_provider_options({'preset': 'screening'})
        self.assertNotIn('n_threads', options)
        self.assertNotIn('n_mkl_threads', options)
        self.assertNotIn('stack_memory_bytes', options)
        explicit = normalized_block2_provider_options({'n_threads': 4})
        self.assertEqual(explicit['n_threads'], 4)
        self.assertNotIn('n_mkl_threads', explicit)
        explicit_memory = normalized_block2_provider_options({'stack_memory_gb': 3})
        self.assertEqual(explicit_memory['stack_memory_bytes'], 3 * 1024 ** 3)

    def test_orbital_ordering_options_are_explicit_and_validated(self):
        options = normalize_block2_options({
            'preset': 'screening',
            'orbital_ordering': 'manual',
            'orbital_order': [1, 0],
        })
        self.assertEqual(options.orbital_ordering, 'manual')
        self.assertEqual(options.orbital_order, [1, 0])
        with self.assertRaisesRegex(ValueError, 'requires orbital_order'):
            normalize_block2_options({'orbital_ordering': 'manual'})

    def test_compiler_selects_provider_module_and_gate(self):
        registry = default_registry()
        spec = {
            'task_type': 'model_hamiltonian',
            'method': {'name': 'hf'},
            'solver': {'name': 'block2_dmrg'},
            'analysis': {'outputs': ['energy', 'strong_correlation_diagnostics']},
        }
        workflow = compile_task_workflow(spec, registry.modules_for(scope='task'))
        gates = compile_task_gates(spec, registry.gates_for(scope='task'))
        self.assertIn('solver.block2.dmrg', workflow.execution_order)
        self.assertIn('solver.block2.bond_dimension_planning', workflow.execution_order)
        self.assertLess(
            workflow.execution_order.index('solver.block2.dmrg'),
            workflow.execution_order.index('solver.block2.bond_dimension_planning'),
        )
        self.assertLess(
            workflow.execution_order.index('solver.block2.bond_dimension_planning'),
            workflow.execution_order.index('core.input_generation'),
        )
        self.assertLess(
            workflow.execution_order.index('solver.block2.dmrg'),
            workflow.execution_order.index('core.execution'),
        )
        self.assertIn('task.block2_availability', gates.evaluation_order)

    def test_compiler_composes_registered_block2_feature_modules(self):
        registry = default_registry()
        spec = {
            'task_type': 'model_hamiltonian',
            'method': {'name': 'hf'},
            'solver': {'name': 'block2_dmrg'},
            'analysis': {'outputs': [
                'energy',
                'entanglement_diagnostics',
                'excited_states',
                'symmetry_analysis',
            ]},
        }
        workflow = compile_task_workflow(spec, registry.modules_for(scope='task'))
        feature_modules = (
            'solver.block2.entanglement_diagnostics',
            'solver.block2.excited_states',
            'solver.block2.symmetry_analysis',
        )
        for module_id in feature_modules:
            self.assertIn(module_id, workflow.execution_order)
            self.assertLess(
                workflow.execution_order.index('solver.block2.dmrg'),
                workflow.execution_order.index(module_id),
            )
            self.assertLess(
                workflow.execution_order.index(module_id),
                workflow.execution_order.index('core.input_generation'),
            )
        continuation = registry.module('solver.block2.mps_continuation')
        self.assertEqual(continuation.default_stage, 'task.prepare')
        self.assertIn('solver.block2.dmrg', continuation.requires_modules)

    def test_resolved_provider_configuration_is_used_by_execution_and_scripts(self):
        state = {
            'task_spec': {
                'task_type': 'model_hamiltonian',
                'solver': {'name': 'dmrg', 'options': {'preset': 'screening'}},
            },
            'solver_provider': {
                'provider': 'block2_dmrg',
                'configuration': {'preset': 'balanced', 'sweeps': 7},
            },
        }
        effective = _effective_task_spec(state)
        self.assertEqual(effective.solver.name, 'block2_dmrg')
        self.assertEqual(effective.solver.options, {'preset': 'balanced', 'sweeps': 7})
        self.assertEqual(state['task_spec']['solver']['name'], 'dmrg')

    def test_dmrg_arrays_are_saved_as_a_separate_artifact(self):
        task_spec = task_spec_from_dict({'task_type': 'molecular', 'method': {'name': 'hf'}})
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = {
                'run_id': 'block2-artifact-test',
                'work_dir': temporary_directory,
                'retry_count': 0,
                'artifacts': [],
            }
            write_result_artifacts(
                state,
                task_spec,
                {
                    'task_type': 'molecular',
                    'energy': -1.0,
                    'converged': True,
                    'raw_scf_output': '',
                    'analysis_text': '',
                    'dmrg_result': {
                        'solver': 'block2_dmrg',
                        'energy': -1.0,
                        'bond_dimension_plan': {
                            'schema': 'pyscf-agent.block2-bond-dimension-plan.v1',
                            'exact_bond_dimension_cap': 4,
                        },
                        'entanglement_active_space_recommendation': {
                            'schema': 'pyscf-agent.entanglement-active-space-recommendation.v1',
                            'status': 'approval_recommended',
                        },
                        'casscf': {
                            'schema': 'pyscf-agent.block2-dmrg-casscf.v1',
                            'orbital_converged': True,
                        },
                        'checkpoint_manifest': {
                            'schema': 'pyscf-agent.block2-mps-manifest.v1',
                            'files': [],
                        },
                    },
                    '_transient_dmrg_arrays': {'rdm1': np.eye(2)},
                },
            )
            kinds = {item['kind'] for item in state['artifacts']}
            self.assertIn('block2_dmrg_result', kinds)
            self.assertIn('block2_bond_dimension_plan', kinds)
            self.assertIn('block2_dmrg_casscf', kinds)
            self.assertIn('block2_dmrg_arrays', kinds)
            self.assertIn('block2_mps_manifest', kinds)
            self.assertIn('entanglement_active_space_recommendation', kinds)
            self.assertTrue((Path(temporary_directory) / 'block2-artifact-test' / 'result-block2-dmrg-arrays.npz').is_file())

    def test_pyscf_adapter_returns_rdms_and_restarts_between_solver_calls(self):
        calls = []
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / 'GS-MPS_INFO'
            checkpoint.write_text('checkpoint', encoding='utf-8')

            def fake_runner(hamiltonian, options, *, scratch_directory, compute_2rdm):
                calls.append({
                    'hamiltonian': hamiltonian,
                    'options': options,
                    'scratch_directory': scratch_directory,
                    'compute_2rdm': compute_2rdm,
                })
                call_index = len(calls)
                return {
                    'energy': -1.0 - 0.01 * call_index,
                    'converged': True,
                    'convergence': {'sweeps_completed': 4},
                    'hamiltonian': hamiltonian.summary(),
                    'orbital_ordering': {
                        'requested_method': 'canonical',
                        'requested_order': [],
                        'permutation': [0, 1],
                    },
                    'restart': {'applied': call_index > 1},
                    'checkpoint_manifest': {
                        'schema': 'pyscf-agent.block2-mps-manifest.v1',
                        'scratch_directory': temporary_directory,
                        'n_orbitals': 2,
                        'n_electrons': [1, 1],
                        'spin': 0,
                        'symmetry': 'su2',
                        'nroots': 1,
                        'mps_tag': 'GS',
                        'orbital_ordering': {
                            'requested_method': 'canonical',
                            'requested_order': [],
                            'permutation': [0, 1],
                        },
                        'files': [{
                            'relative_path': checkpoint.name,
                            'path': str(checkpoint),
                            'size_bytes': checkpoint.stat().st_size,
                        }],
                    },
                    '_transient_dmrg_arrays': {
                        'rdm1': np.eye(2),
                        'rdm2': np.zeros((2, 2, 2, 2)),
                    },
                }

            mol = type('Molecule', (), {
                'spin': 0,
                'verbose': 0,
                'max_memory': 1000,
                'stdout': None,
            })()
            adapter = Block2FCISolverAdapter(
                mol,
                {
                    'preset': 'screening', 'bond_dimensions': [600],
                    'final_bond_dimension': 2000, 'sweeps': 30,
                },
                scratch_directory=temporary_directory,
                runner=fake_runner,
            )
            h1e = np.zeros((2, 2))
            eri = np.zeros(6)
            first_energy, first_state = adapter.kernel(h1e, eri, 2, (1, 1))
            second_energy, second_state = adapter.approx_kernel(h1e, eri, 2, (1, 1), ci0=first_state)
            final_energy, _ = adapter.final_analysis(h1e, eri, 2, (1, 1))

        self.assertAlmostEqual(first_energy, -1.01)
        self.assertAlmostEqual(second_energy, -1.02)
        self.assertAlmostEqual(final_energy, -1.03)
        np.testing.assert_allclose(adapter.make_rdm1(second_state, 2, (1, 1)), np.eye(2))
        self.assertTrue(calls[0]['compute_2rdm'])
        self.assertEqual(calls[0]['options']['rdm_root_scope'], 'all_states')
        self.assertNotIn('restart_manifest', calls[0]['options'])
        self.assertEqual(
            calls[1]['options']['restart_provenance']['selection'],
            'previous_dmrg_casscf_solver_call',
        )
        self.assertEqual(len(adapter.trace), 3)
        self.assertEqual([call['options']['bond_dimensions'] for call in calls], [[600], [600], [2000]])
        self.assertTrue(all(call['options']['sweeps'] == 30 for call in calls))
        self.assertTrue(all(call['options']['adaptive_schedule'] is False for call in calls))
        self.assertEqual(calls[2]['options']['restart_provenance']['source_solver_call'], 2)
        self.assertEqual(calls[2]['options']['max_bond_dimension'], 2000)
        self.assertEqual(adapter.options['bond_dimensions'], [600])
        self.assertIsNone(adapter.trace[0]['state_average_energy'])
        self.assertIsNone(adapter.trace[0]['state_average_weights'])

    def test_final_bond_dimension_survives_option_normalization(self):
        options = normalized_block2_provider_options({'final_bond_dimension': 2000})
        self.assertEqual(options['final_bond_dimension'], 2000)
        self.assertIsNone(normalize_block2_options().final_bond_dimension)
        for value in (0, -1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'final_bond_dimension'):
                normalize_block2_options({'final_bond_dimension': value})

    def test_adapter_without_final_bond_dimension_keeps_adaptive_final_solve(self):
        adapter = Block2FCISolverAdapter(type('Molecule', (), {})(), {
            'bond_dimensions': [600], 'max_bond_dimension': 2000, 'adaptive_schedule': True,
        })
        self.assertFalse(adapter._iteration_options()['adaptive_schedule'])
        final_options = adapter._iteration_options(final_analysis=True)
        self.assertTrue(final_options['adaptive_schedule'])
        self.assertEqual(final_options['bond_dimensions'], [600])

    def test_pyscf_adapter_returns_root_specific_rdms_for_state_average(self):
        def fake_runner(hamiltonian, options, **_kwargs):
            nroots = int(options['nroots'])
            root_rdm1 = np.asarray([np.eye(2) * (root + 1) for root in range(nroots)])
            root_rdm2 = np.zeros((nroots, 2, 2, 2, 2))
            return {
                'energy': -1.0,
                'state_energies': [-1.0, -0.75],
                'converged': True,
                'convergence': {},
                'hamiltonian': hamiltonian.summary(),
                'orbital_ordering': {
                    'requested_method': 'canonical',
                    'requested_order': [],
                    'permutation': [0, 1],
                },
                'restart': {'applied': False},
                'checkpoint_manifest': {
                    'schema': 'pyscf-agent.block2-mps-manifest.v1',
                    'files': [],
                },
                '_transient_dmrg_arrays': {
                    'rdm1': root_rdm1[0],
                    'rdm2': root_rdm2[0],
                    'root_rdm1': root_rdm1,
                    'root_rdm2': root_rdm2,
                },
            }

        molecule = type('Molecule', (), {
            'spin': 0,
            'verbose': 0,
            'max_memory': 1000,
            'stdout': None,
        })()
        adapter = Block2FCISolverAdapter(
            molecule,
            {'nroots': 2, 'state_average_weights': [0.25, 0.75]},
            runner=fake_runner,
        )
        adapter.weights = [0.25, 0.75]
        energies, states = adapter.kernel(
            np.zeros((2, 2)),
            np.zeros(6),
            2,
            (1, 1),
            nroots=2,
        )

        self.assertEqual(energies, [-1.0, -0.75])
        self.assertEqual([state.root_index for state in states], [0, 1])
        np.testing.assert_allclose(adapter.make_rdm1(states[1], 2, (1, 1)), np.eye(2) * 2)
        self.assertAlmostEqual(adapter.trace[0]['state_average_energy'], -0.8125)

    def test_pyscf_adapter_locks_external_checkpoint_orbital_order(self):
        calls = []
        source_manifest = {
            'schema': 'pyscf-agent.block2-mps-manifest.v1',
            'n_orbitals': 2,
            'n_electrons': [1, 1],
            'spin': 0,
            'symmetry': 'su2',
            'nroots': 1,
            'mps_tag': 'GS',
            'orbital_ordering': {
                'requested_method': 'fiedler',
                'requested_order': [],
                'permutation': [1, 0],
            },
            'files': [{'relative_path': 'GS-MPS_INFO'}],
        }

        def fake_runner(hamiltonian, options, **_kwargs):
            calls.append(copy.deepcopy(options))
            return {
                'energy': -1.0,
                'converged': True,
                'convergence': {},
                'hamiltonian': hamiltonian.summary(),
                'orbital_ordering': {
                    'requested_method': 'manual',
                    'requested_order': [1, 0],
                    'permutation': [1, 0],
                },
                'restart': {'applied': True},
                'checkpoint_manifest': copy.deepcopy(source_manifest),
                '_transient_dmrg_arrays': {
                    'rdm1': np.eye(2),
                    'rdm2': np.zeros((2, 2, 2, 2)),
                },
            }

        molecule = type('Molecule', (), {
            'spin': 0,
            'verbose': 0,
            'max_memory': 1000,
            'stdout': None,
        })()
        adapter = Block2FCISolverAdapter(
            molecule,
            {'restart_manifest': source_manifest},
            runner=fake_runner,
        )
        adapter.kernel(np.zeros((2, 2)), np.zeros(6), 2, (1, 1))

        self.assertEqual(calls[0]['orbital_ordering'], 'manual')
        self.assertEqual(calls[0]['orbital_order'], [1, 0])
        self.assertEqual(
            calls[0]['restart_manifest']['orbital_ordering']['requested_method'],
            'manual',
        )
        self.assertFalse(calls[0]['adaptive_schedule'])

    def test_molecular_dmrg_rejects_unsupported_routes_during_validation(self):
        base = {
            'task_type': 'molecular',
            'system': {'spin': 0},
            'method': {'name': 'casci', 'restricted': True},
            'solver': {'name': 'block2_dmrg'},
            'active_space': {
                'enabled': True,
                'ncas': 4,
                'nelecas': 4,
                'orbital_indices': [0, 1, 2, 3],
                'approved': True,
            },
        }
        casscf_payload = dict(base)
        casscf_payload['method'] = {'name': 'casscf', 'restricted': True}
        self.assertFalse(_validate_strong_correlation_spec(task_spec_from_dict(casscf_payload)))
        multiroot_payload = dict(base)
        multiroot_payload.update({
            'method': {'name': 'casscf', 'restricted': True},
            'solver': {
                'name': 'block2_dmrg',
                'options': {'nroots': 2, 'state_average_weights': [0.5, 0.5]},
            },
        })
        self.assertFalse(_validate_strong_correlation_spec(task_spec_from_dict(multiroot_payload)))
        cases = (
            ({'method': {'name': 'casscf', 'restricted': True}, 'solver': {
                'name': 'block2_dmrg',
                'options': {'nroots': 'not-an-integer'},
            }}, 'invalid_block2_dmrg_nroots'),
            ({'method': {'name': 'casscf', 'restricted': True}, 'solver': {
                'name': 'block2_dmrg',
                'options': {'nroots': 2, 'state_average_weights': [1.0]},
            }}, 'invalid_block2_state_average_weights'),
            ({'method': {'name': 'casscf', 'restricted': True}, 'solver': {
                'name': 'block2_dmrg',
                'options': {'nroots': 21},
            }}, 'block2_nroots_exceeds_spin_sector'),
            ({'method': {'name': 'casci', 'restricted': False}}, 'block2_dmrg_unrestricted_not_supported'),
            ({'post_cas': {'sc_nevpt2': {'enabled': True}}}, 'block2_dmrg_nevpt2_not_supported'),
            ({'system': {'spin': 2}, 'active_space': {
                'enabled': True,
                'ncas': 4,
                'nelecas': [2, 2],
                'orbital_indices': [0, 1, 2, 3],
                'approved': True,
            }}, 'block2_dmrg_active_spin_mismatch'),
        )
        for override, expected_code in cases:
            payload = dict(base)
            payload.update(override)
            errors = _validate_strong_correlation_spec(task_spec_from_dict(payload))
            self.assertIn(expected_code, {item['code'] for item in errors})

    def test_orbital_processing_validation_rejects_incompatible_ordering(self):
        payload = {
            'task_type': 'molecular',
            'method': {'name': 'casci', 'restricted': True},
            'solver': {'name': 'fci'},
            'active_space': {'enabled': True, 'ncas': 2, 'nelecas': 2, 'approved': True},
            'orbital_processing': {'orbital_ordering': 'fiedler'},
        }
        codes = {
            item['code']
            for item in _validate_strong_correlation_spec(task_spec_from_dict(payload))
        }
        self.assertIn('orbital_ordering_requires_block2', codes)

    def test_probe_defers_target_block2_orbital_ordering_validation(self):
        task_spec = task_spec_from_dict({
            'task_type': 'molecular',
            'method': {'name': 'hf', 'restricted': False},
            'active_space': {
                'enabled': True,
                'selection_method': 'occupation_window',
                'target_method': 'casscf',
                'target_solver': 'block2_dmrg',
                'approved': False,
            },
            'orbital_processing': {
                'enabled': True,
                'localization_method': 'pipek_mezey',
                'localization_scope': 'active_space',
                'orbital_ordering': 'fiedler',
            },
            'workflow': {'modules': ['molecular.active_space_probe']},
        })

        codes = {
            item['code']
            for item in _validate_strong_correlation_spec(task_spec)
        }
        self.assertNotIn('orbital_ordering_requires_block2', codes)
        self.assertNotIn('active_space_localization_requires_block2_casci', codes)

    def test_unregistered_hf_request_cannot_bypass_block2_orbital_validation(self):
        task_spec = task_spec_from_dict({
            'task_type': 'molecular',
            'method': {'name': 'hf', 'restricted': False},
            'active_space': {
                'enabled': True,
                'selection_method': 'occupation_window',
                'target_method': 'casscf',
                'target_solver': 'block2_dmrg',
                'approved': False,
            },
            'orbital_processing': {
                'enabled': True,
                'localization_method': 'pipek_mezey',
                'localization_scope': 'active_space',
                'orbital_ordering': 'fiedler',
            },
        })

        codes = {
            item['code']
            for item in _validate_strong_correlation_spec(task_spec)
        }
        self.assertIn('orbital_ordering_requires_block2', codes)
        self.assertIn('active_space_localization_requires_block2_casci', codes)

    def test_block2_only_outputs_and_restart_reject_other_solvers(self):
        task_spec = task_spec_from_dict({
            'task_type': 'model_hamiltonian',
            'solver': {
                'name': 'fci',
                'options': {'restart_manifest': '/tmp/prior-mps.json'},
            },
            'analysis': {
                'outputs': ['energy', 'entanglement_diagnostics', 'excited_states', 'symmetry_analysis'],
            },
        })
        errors = _validate_strong_correlation_spec(task_spec)
        codes = {item['code'] for item in errors}
        self.assertIn('block2_output_requires_block2_solver', codes)
        self.assertIn('block2_restart_requires_block2_solver', codes)

    def test_dmrg_casscf_accepts_joint_checkpoint_request_for_runtime_validation(self):
        task_spec = task_spec_from_dict({
            'task_type': 'molecular',
            'system': {'spin': 0},
            'method': {'name': 'casscf', 'restricted': True},
            'solver': {
                'name': 'block2_dmrg',
                'options': {'restart_manifest': '/tmp/prior-mps.json'},
            },
            'active_space': {
                'enabled': True,
                'ncas': 4,
                'nelecas': 4,
                'orbital_indices': [0, 1, 2, 3],
                'approved': True,
            },
        })
        codes = {item['code'] for item in _validate_strong_correlation_spec(task_spec)}
        self.assertNotIn('block2_dmrg_casscf_external_restart_not_supported', codes)

    def test_joint_checkpoint_writes_and_loads_unchanged_orbitals(self):
        from pyscf import gto, scf
        molecule = gto.M(atom='H 0 0 0; H 0 0 .8', basis='sto-3g', verbose=0)
        mean_field = scf.RHF(molecule).run()
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = {
                'schema': 'pyscf-agent.block2-mps-manifest.v1',
                'scratch_directory': temporary_directory,
                'files': [],
            }
            source = mean_field.mo_coeff
            context = attach_optimized_orbitals(
                manifest,
                source,
                ncore=0,
                ncas=2,
                nelecas=2,
                reference='restricted',
                orbital_provenance={'output_basis': 'casscf_optimized_active_space'},
                molecule=molecule,
            )
            loaded, provenance = load_optimized_orbitals(
                manifest,
                mean_field,
                ncas=2,
                nelecas=2,
            )

        self.assertTrue(context['external_restart_supported'])
        self.assertEqual(manifest['checkpoint_components'], ['mps', 'optimized_orbitals'])
        np.testing.assert_allclose(
            loaded.T.dot(mean_field.get_ovlp()).dot(loaded),
            np.eye(2),
            atol=1e-10,
        )
        self.assertTrue(provenance['same_active_space'])
        self.assertLess(provenance['orthonormality_max_abs_error'], 1e-10)

    def test_dmrg_nonconvergence_is_not_reported_as_scf_failure(self):
        task_spec = task_spec_from_dict({
            'task_type': 'model_hamiltonian',
            'solver': {'name': 'block2_dmrg'},
        })
        error = _unconverged_execution_error(
            task_spec,
            {
                'dmrg_result': {
                    'convergence': {
                        'final_energy_change': 1e-4,
                        'final_discarded_weight': 1e-3,
                    },
                },
            },
        )
        self.assertEqual(error['code'], 'solver_unconverged')
        self.assertIn('block2 DMRG', error['message'])
        self.assertEqual(error['details']['final_discarded_weight'], 1e-3)


@unittest.skipUnless(block2_is_available(), 'optional block2 provider is not installed')
class Block2ProviderNumericalTests(unittest.TestCase):
    def test_conventional_mpo_energy_and_rdms_match_fci(self):
        from pyscf import ao2mo, fci, gto, scf
        from pyscf_agent.providers.block2.driver import run_block2_dmrg

        for nsites in (2, 4, 6):
            with self.subTest(nsites=nsites), tempfile.TemporaryDirectory() as scratch:
                mol = gto.M(atom=[('H', (0, 0, i * 1.4)) for i in range(nsites)], basis='sto-3g', verbose=0)
                mf = scf.RHF(mol).run(conv_tol=1e-12)
                self.assertTrue(mf.converged)
                h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
                g2e = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), nsites)
                energy, ci = fci.direct_spin0.kernel(h1e, g2e, nsites, mol.nelec, ecore=mol.energy_nuc(), tol=1e-12)
                dm1, dm2 = fci.direct_spin0.make_rdm12(ci, nsites, mol.nelec)
                hamiltonian = ElectronicHamiltonian(
                    n_orbitals=nsites, n_electrons=mol.nelec, spin=0,
                    core_energy=mol.energy_nuc(), h1e=h1e, g2e=g2e,
                )
                original_h1e, original_g2e = h1e.copy(), g2e.copy()
                physical_scratch = Path(scratch) / 'physical'
                physical_scratch.mkdir()
                shared_scratch = Path(scratch) / 'shared'
                shared_scratch.symlink_to(physical_scratch, target_is_directory=True)
                result = run_block2_dmrg(hamiltonian, {
                    **_small_dmrg_options(), 'mpo_algorithm': 'conventional',
                    'orbital_ordering': 'canonical' if nsites == 6 else 'fiedler',
                    'davidson_thresholds': [1e-13],
                    'save_mps': True,
                }, scratch_directory=str(shared_scratch), compute_2rdm=True)
                manifest = result['checkpoint_manifest']
                self.assertEqual(manifest['scratch_directory'], str(shared_scratch))
                self.assertTrue(manifest['files'])
                for entry in manifest['files']:
                    self.assertTrue(Path(entry['path']).is_relative_to(shared_scratch))
                np.testing.assert_array_equal(h1e, original_h1e)
                np.testing.assert_array_equal(g2e, original_g2e)
                arrays = result['_transient_dmrg_arrays']
                self.assertAlmostEqual(result['energy'], energy, places=9)
                np.testing.assert_allclose(arrays['rdm1'], dm1, atol=1e-6, rtol=0)
                np.testing.assert_allclose(arrays['rdm2'], dm2, atol=1e-6, rtol=0)
                reconstructed = mol.energy_nuc() + np.einsum('ij,ji', h1e, arrays['rdm1']) + 0.5 * np.einsum('ijkl,ijkl', g2e, arrays['rdm2'])
                self.assertAlmostEqual(reconstructed, result['energy'], places=9)
                self.assertEqual(result['mpo_algorithm']['effective'], 'ConventionalNC' if nsites == 2 else 'Conventional')
                self.assertEqual(result['configuration']['integral_cutoff'], 1e-12)

    def test_model_dmrg_matches_fci_energy_and_rdms(self):
        outputs = ['energy', 'density', 'double_occupancy', 'spin_correlation', 'charge_correlation', 'strong_correlation_diagnostics']
        fci_result = run_model_hamiltonian_solver(
            _hubbard_dimer_spec(),
            solver_name='fci',
            outputs=outputs,
        )
        with tempfile.TemporaryDirectory() as scratch:
            with mock.patch.dict('os.environ', {'SLURM_CPUS_PER_TASK': '2'}):
                dmrg_result = run_model_hamiltonian_solver(
                    _hubbard_dimer_spec(),
                    solver_name='block2_dmrg',
                    outputs=outputs,
                    solver_options=_small_dmrg_options(),
                    scratch_directory=scratch,
                )
        self.assertTrue(dmrg_result['converged'])
        quality = evaluate_quality_checks(dmrg_result['quality_checks'])
        self.assertEqual(quality['quality_status'], 'passed')
        self.assertTrue(quality['publication_eligible'])
        self.assertEqual(dmrg_result['dmrg_result']['threading']['effective_n_threads'], 2)
        self.assertEqual(
            dmrg_result['dmrg_result']['threading']['n_threads_source'],
            'SLURM_CPUS_PER_TASK',
        )
        bond_plan = dmrg_result['dmrg_result']['bond_dimension_plan']
        self.assertEqual(bond_plan['exact_bond_dimension_cap'], 4)
        self.assertEqual(bond_plan['effective_bond_dimensions'], [4] * 6)
        self.assertEqual(dmrg_result['dmrg_result']['configuration']['bond_dimensions'], [4] * 6)
        self.assertAlmostEqual(dmrg_result['energy'], fci_result['energy'], places=10)
        self.assertEqual(dmrg_result['natural_occupation_summary']['spin_resolved']['status'], 'available')
        self.assertAlmostEqual(
            dmrg_result['strong_correlation_diagnostics']['physics_score'],
            fci_result['strong_correlation_diagnostics']['physics_score'], places=6,
        )
        for field in ('density', 'double_occupancy', 'spin_correlation', 'charge_correlation'):
            np.testing.assert_allclose(
                dmrg_result[field]['values'],
                fci_result[field]['values'],
                atol=1e-10,
            )

    def test_model_orbital_ordering_preserves_energy_and_manifest_provenance(self):
        energies = {}
        with tempfile.TemporaryDirectory() as temporary_directory:
            for method, order in (
                ('canonical', []),
                ('manual', [1, 0]),
                ('fiedler', []),
            ):
                options = _small_dmrg_options()
                options['save_mps'] = True
                options['orbital_ordering'] = method
                if order:
                    options['orbital_order'] = order
                result = run_model_hamiltonian_solver(
                    _hubbard_dimer_spec(),
                    solver_name='block2_dmrg',
                    outputs=['energy'],
                    solver_options=options,
                    scratch_directory=str(Path(temporary_directory) / method),
                )
                ordering = result['dmrg_result']['orbital_ordering']
                manifest_ordering = result['dmrg_result']['checkpoint_manifest']['orbital_ordering']
                self.assertEqual(ordering, manifest_ordering)
                self.assertEqual(ordering['requested_method'], method)
                energies[method] = result['energy']
        self.assertAlmostEqual(energies['canonical'], energies['manual'], places=9)
        self.assertAlmostEqual(energies['canonical'], energies['fiedler'], places=9)

    def test_molecular_dmrg_casci_matches_fci_casci(self):
        base = {
            'task_type': 'molecular',
            'system': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'unit': 'Angstrom',
                'charge': 0,
                'spin': 0,
            },
            'method': {'name': 'casci', 'restricted': True},
            'active_space': {
                'enabled': True,
                'ncas': 2,
                'nelecas': 2,
                'orbital_indices': [0, 1],
                'approved': True,
            },
            'analysis': {'outputs': ['energy']},
        }
        fci_payload = dict(base)
        fci_payload['solver'] = {'name': 'fci', 'options': {}}
        fci_result = _run_pyscf_task(task_spec_from_dict(fci_payload))
        dmrg_payload = dict(base)
        dmrg_payload['solver'] = {'name': 'block2_dmrg', 'options': _small_dmrg_options()}
        with tempfile.TemporaryDirectory() as scratch:
            dmrg_result = _run_pyscf_task(
                task_spec_from_dict(dmrg_payload),
                execution_directory=Path(scratch),
            )
        self.assertTrue(dmrg_result['converged'])
        self.assertEqual(dmrg_result['cas_result']['solver'], 'block2_dmrg')
        self.assertAlmostEqual(dmrg_result['energy'], fci_result['energy'], places=10)

    def test_molecular_dmrg_casscf_matches_fci_casscf(self):
        base = {
            'task_type': 'molecular',
            'system': {
                'atom': 'H 0 0 0; H 0 0 1.5',
                'basis': '6-31g',
                'unit': 'Angstrom',
                'charge': 0,
                'spin': 0,
            },
            'method': {'name': 'casscf', 'restricted': True},
            'active_space': {
                'enabled': True,
                'ncas': 2,
                'nelecas': 2,
                'orbital_indices': [0, 1],
                'approved': True,
            },
            'analysis': {'outputs': ['energy']},
        }
        fci_payload = dict(base)
        fci_payload['solver'] = {'name': 'fci', 'options': {}}
        fci_result = _run_pyscf_task(task_spec_from_dict(fci_payload))
        dmrg_payload = dict(base)
        dmrg_payload['solver'] = {'name': 'block2_dmrg', 'options': {
            **_small_dmrg_options(), 'mpo_algorithm': 'conventional',
        }}
        dmrg_payload['orbital_processing'] = {
            'enabled': True,
            'orbital_ordering': 'fiedler',
        }
        with tempfile.TemporaryDirectory() as scratch:
            dmrg_result = _run_pyscf_task(
                task_spec_from_dict(dmrg_payload),
                execution_directory=Path(scratch),
            )
        casscf = dmrg_result['cas_result']['dmrg_result']['casscf']
        self.assertTrue(dmrg_result['converged'])
        self.assertEqual(dmrg_result['method'], 'casscf')
        self.assertEqual(dmrg_result['solver'], 'block2_dmrg')
        self.assertEqual(casscf['orbital_optimization_mode'], 'state_specific')
        self.assertNotIn('state_average_energy', casscf)
        self.assertNotIn('state_average_weights', casscf)
        self.assertNotIn('state_average_energy', dmrg_result['cas_result'])
        self.assertNotIn('state_average_weights', dmrg_result['cas_result'])
        self.assertTrue(casscf['orbital_converged'])
        self.assertGreater(casscf['macro_iterations'], 0)
        self.assertGreater(casscf['active_space_solver_calls'], 0)
        self.assertEqual(casscf['macro_iteration_trace'][0]['macro_iteration'], 1)
        ordering = dmrg_result['cas_result']['dmrg_result']['orbital_ordering']
        self.assertEqual(ordering['requested_method'], 'fiedler')
        self.assertEqual(ordering['requested_order'], [])
        self.assertEqual(ordering['internal_continuation_method'], 'manual')
        self.assertAlmostEqual(dmrg_result['energy'], fci_result['energy'], places=8)

    def test_molecular_dmrg_entanglement_stops_at_2rdm(self):
        options = _small_dmrg_options()
        options.update({
            'mpo_algorithm': 'conventional',
            'compute_entanglement': True,
            'compute_mutual_information': True,
            'compute_bipartite_entanglement': False,
        })
        payload = {
            'task_type': 'molecular',
            'system': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'unit': 'Angstrom',
                'charge': 0,
                'spin': 0,
            },
            'method': {'name': 'casscf', 'restricted': True},
            'solver': {'name': 'block2_dmrg', 'options': options},
            'active_space': {
                'enabled': True,
                'ncas': 2,
                'nelecas': 2,
                'orbital_indices': [0, 1],
                'approved': True,
            },
            'analysis': {'outputs': ['energy', 'entanglement_diagnostics']},
        }

        with tempfile.TemporaryDirectory() as scratch:
            result = _run_pyscf_task(
                task_spec_from_dict(payload),
                execution_directory=Path(scratch),
            )

        dmrg_result = result['cas_result']['dmrg_result']
        entanglement = dmrg_result['entanglement_diagnostics']
        self.assertEqual(dmrg_result['configuration']['symmetry'], 'su2')
        self.assertEqual(entanglement['backend'], 'block2_npdm_sz_from_su2')
        self.assertEqual(
            entanglement['npdm_context']['wavefunction_conversion'],
            'su2_to_sz_component',
        )
        self.assertFalse(entanglement['npdm_context']['wavefunction_reoptimized'])
        self.assertEqual(entanglement['npdm_order_limit'], 2)
        self.assertEqual(
            entanglement['availability']['mutual_information'],
            'not_computed_order_limit',
        )
        self.assertEqual(entanglement['errors'], [])
        self.assertNotIn('orbital_mutual_information', entanglement)
        self.assertEqual(entanglement['limitations'][0]['required_npdm_order'], 4)
        self.assertEqual(len(entanglement['single_orbital_entropy']), 2)
        self.assertEqual(dmrg_result['result_scope']['rdm1'], 'ground_state')
        self.assertEqual(dmrg_result['result_scope']['rdm2'], 'ground_state')

    def test_state_averaged_dmrg_casscf_matches_fci_state_average(self):
        from pyscf import fci, gto, mcscf, scf  # pylint: disable=import-outside-toplevel

        mol = gto.M(
            atom='H 0 0 0; H 0 0 1.5',
            basis='6-31g',
            unit='Angstrom',
            spin=0,
            verbose=0,
        )
        mf = scf.RHF(mol).run()
        fci_mc = mcscf.CASSCF(mf, 2, 2)
        fci_mc.fcisolver = fci.direct_spin0.FCI(mol)
        weights = [0.7, 0.3]
        fci_mc = mcscf.state_average_(fci_mc, weights)
        fci_mc.kernel()
        reference_energies = [float(value) for value in fci_mc.e_states]

        options = _small_dmrg_options()
        options.update({'nroots': 2, 'state_average_weights': weights})
        payload = {
            'task_type': 'molecular',
            'system': {
                'atom': 'H 0 0 0; H 0 0 1.5',
                'basis': '6-31g',
                'unit': 'Angstrom',
                'charge': 0,
                'spin': 0,
            },
            'method': {'name': 'casscf', 'restricted': True},
            'solver': {'name': 'block2_dmrg', 'options': options},
            'active_space': {
                'enabled': True,
                'ncas': 2,
                'nelecas': 2,
                'orbital_indices': [0, 1],
                'approved': True,
            },
            'analysis': {'outputs': ['energy', 'excited_states']},
        }
        with tempfile.TemporaryDirectory() as scratch:
            result = _run_pyscf_task(
                task_spec_from_dict(payload),
                execution_directory=Path(scratch),
            )

        dmrg = result['cas_result']['dmrg_result']
        self.assertEqual(dmrg['casscf']['orbital_optimization_mode'], 'state_averaged')
        self.assertEqual(dmrg['casscf']['nroots'], 2)
        self.assertEqual(dmrg['state_average_weights'], weights)
        self.assertEqual(len(dmrg['state_energies']), 2)
        self.assertEqual(dmrg['natural_occupations_scope'], 'root_0')
        self.assertEqual(dmrg['entanglement_diagnostics_scope'], 'root_0')
        for actual, expected in zip(dmrg['state_energies'], reference_energies):
            self.assertAlmostEqual(actual, expected, places=7)
        self.assertAlmostEqual(
            dmrg['state_average_energy'],
            sum(weight * energy for weight, energy in zip(weights, reference_energies)),
            places=7,
        )

    def test_molecular_dmrg_casscf_restarts_from_joint_orbital_mps_checkpoint(self):
        def request(distance, options):
            return {
                'task_type': 'molecular',
                'system': {
                    'atom': 'H 0 0 0; H 0 0 {0}'.format(distance),
                    'basis': '6-31g',
                    'unit': 'Angstrom',
                    'charge': 0,
                    'spin': 0,
                },
                'method': {'name': 'casscf', 'restricted': True},
                'solver': {'name': 'block2_dmrg', 'options': options},
                'active_space': {
                    'enabled': True,
                    'ncas': 2,
                    'nelecas': 2,
                    'orbital_indices': [0, 1],
                    'approved': True,
                },
                'orbital_processing': {
                    'enabled': True,
                    'orbital_ordering': 'fiedler',
                },
                'analysis': {'outputs': ['energy']},
            }

        with tempfile.TemporaryDirectory() as temporary_directory:
            first = _run_pyscf_task(
                task_spec_from_dict(request(1.5, _small_dmrg_options())),
                execution_directory=Path(temporary_directory) / 'first',
            )
            manifest = first['cas_result']['dmrg_result']['checkpoint_manifest']
            second_options = _small_dmrg_options()
            second_options.update({
                'restart_manifest': copy.deepcopy(manifest),
                'restart_required': True,
            })
            second = _run_pyscf_task(
                task_spec_from_dict(request(1.5, second_options)),
                execution_directory=Path(temporary_directory) / 'second',
            )

        dmrg = second['cas_result']['dmrg_result']
        self.assertEqual(
            manifest['checkpoint_components'],
            ['mps', 'optimized_orbitals'],
        )
        self.assertTrue(dmrg['casscf']['joint_checkpoint_restart_applied'])
        self.assertTrue(dmrg['casscf']['joint_checkpoint_restart']['same_active_space'])
        self.assertTrue(dmrg['restart']['applied'])
        self.assertTrue(second['converged'])

    def test_open_shell_rohf_dmrg_casscf_matches_fci_casscf(self):
        base = {
            'task_type': 'molecular',
            'system': {
                'atom': 'H 0 0 0; H 0 0 1.5',
                'basis': '6-31g',
                'unit': 'Angstrom',
                'charge': 0,
                'spin': 2,
            },
            'method': {'name': 'casscf', 'restricted': True},
            'active_space': {
                'enabled': True,
                'ncas': 2,
                'nelecas': [2, 0],
                'orbital_indices': [0, 1],
                'approved': True,
            },
            'analysis': {'outputs': ['energy']},
        }
        fci_payload = dict(base)
        fci_payload['solver'] = {'name': 'fci', 'options': {}}
        fci_result = _run_pyscf_task(task_spec_from_dict(fci_payload))
        dmrg_payload = dict(base)
        dmrg_payload['solver'] = {
            'name': 'block2_dmrg',
            'options': _small_dmrg_options(),
        }
        with tempfile.TemporaryDirectory() as scratch:
            dmrg_result = _run_pyscf_task(
                task_spec_from_dict(dmrg_payload),
                execution_directory=Path(scratch),
            )
        self.assertTrue(dmrg_result['converged'])
        self.assertEqual(dmrg_result['cas_result']['reference'], 'rohf')
        self.assertAlmostEqual(dmrg_result['energy'], fci_result['energy'], places=10)

    def test_molecular_active_orbital_localization_is_executed_and_audited(self):
        payload = {
            'task_type': 'molecular',
            'system': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'unit': 'Angstrom',
                'charge': 0,
                'spin': 0,
            },
            'method': {'name': 'casci', 'restricted': True},
            'solver': {'name': 'block2_dmrg', 'options': _small_dmrg_options()},
            'active_space': {
                'enabled': True,
                'ncas': 2,
                'nelecas': 2,
                'orbital_indices': [0, 1],
                'approved': True,
            },
            'orbital_processing': {
                'enabled': True,
                'localization_method': 'boys',
                'localization_scope': 'active_space',
                'orbital_ordering': 'fiedler',
            },
            'analysis': {'outputs': ['energy']},
        }
        with tempfile.TemporaryDirectory() as scratch:
            result = _run_pyscf_task(
                task_spec_from_dict(payload),
                execution_directory=Path(scratch),
            )
        provenance = result['cas_result']['orbital_provenance']
        self.assertTrue(provenance['localization_applied'])
        self.assertEqual(provenance['output_basis'], 'boys_localized_active_space')
        self.assertLess(provenance['orthonormality_max_abs_error'], 1e-7)
        self.assertEqual(
            result['active_space']['audit']['orbital_ordering']['requested_method'],
            'fiedler',
        )

    def test_dmrg_excited_entanglement_and_symmetry_outputs(self):
        options = _small_dmrg_options()
        options.update({
            'nroots': 2,
            'compute_entanglement': True,
            'compute_mutual_information': True,
            'compute_bipartite_entanglement': True,
            'compute_symmetry_analysis': True,
        })
        with tempfile.TemporaryDirectory() as scratch:
            result = run_model_hamiltonian_solver(
                _hubbard_dimer_spec(),
                solver_name='block2_dmrg',
                outputs=['energy', 'entanglement_diagnostics', 'excited_states', 'symmetry_analysis'],
                solver_options=options,
                scratch_directory=scratch,
            )
        self.assertEqual(len(result['state_energies']), 2)
        self.assertAlmostEqual(result['excitation_energies'][0], 0.0)
        self.assertEqual(result['computed_root_count'], 2)
        self.assertEqual(result['requested_root_count'], 2)
        self.assertTrue(result['targeted_roots_complete'])
        self.assertEqual(result['missing_requested_outputs'], [])
        self.assertNotIn('energy_spectrum_unavailable_reason', result)
        self.assertEqual(result['dmrg_result']['rdm_evaluation_scope'], 'ground_state')
        self.assertEqual(result['dmrg_result']['rdm_evaluation_roots'], [0])
        self.assertEqual(result['dmrg_result']['reported_diagnostics_scope'], 'ground_state')
        self.assertEqual(result['dmrg_result']['reported_diagnostics_roots'], [0])
        self.assertEqual(len(result['entanglement_diagnostics']['single_orbital_entropy']), 2)
        self.assertNotIn('strongest_orbital_pairs', result['entanglement_diagnostics'])
        self.assertEqual(
            result['entanglement_diagnostics']['availability']['mutual_information'],
            'not_computed_order_limit',
        )
        self.assertAlmostEqual(result['symmetry_analysis']['spin_square'], 0.0, places=8)
        self.assertEqual(result['symmetry_analysis']['particle_number'], 2)
        if result['symmetry_analysis']['symmetry_backend'] == 'su2':
            self.assertEqual(result['symmetry_analysis']['target_total_spin'], 0.0)
        else:
            self.assertEqual(result['symmetry_analysis']['target_spin_projection'], 0.0)

    def test_dmrg_can_restart_from_a_registered_mps_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            first_options = _small_dmrg_options()
            first_options['save_mps'] = True
            first = run_model_hamiltonian_solver(
                _hubbard_dimer_spec(),
                solver_name='block2_dmrg',
                outputs=['energy'],
                solver_options=first_options,
                scratch_directory=str(Path(temporary_directory) / 'first'),
            )
            second_options = _small_dmrg_options()
            second_options.update({
                'save_mps': True,
                'restart_manifest': first['dmrg_result']['checkpoint_manifest'],
            })
            second = run_model_hamiltonian_solver(
                _hubbard_dimer_spec(),
                solver_name='block2_dmrg',
                outputs=['energy'],
                solver_options=second_options,
                scratch_directory=str(Path(temporary_directory) / 'second'),
            )
        self.assertTrue(second['dmrg_result']['restart']['applied'])
        self.assertTrue(all(second['dmrg_result']['restart']['checks'].values()))
        self.assertNotIn('same_hamiltonian', second['dmrg_result']['restart'])
        self.assertAlmostEqual(first['energy'], second['energy'], places=10)

    def test_dmrg_rejects_mps_only_restart_when_checkpoint_requires_optimized_orbitals(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            first_options = _small_dmrg_options()
            first_options['save_mps'] = True
            first = run_model_hamiltonian_solver(
                _hubbard_dimer_spec(),
                solver_name='block2_dmrg',
                outputs=['energy'],
                solver_options=first_options,
                scratch_directory=str(Path(temporary_directory) / 'first'),
            )
            manifest = dict(first['dmrg_result']['checkpoint_manifest'])
            manifest['orbital_context'] = {
                'kind': 'casscf_optimized_orbitals',
                'external_restart_supported': False,
            }
            second_options = _small_dmrg_options()
            second_options.update({
                'restart_manifest': manifest,
                'restart_required': True,
            })
            with self.assertRaisesRegex(ValueError, 'optimized orbitals'):
                run_model_hamiltonian_solver(
                    _hubbard_dimer_spec(),
                    solver_name='block2_dmrg',
                    outputs=['energy'],
                    solver_options=second_options,
                    scratch_directory=str(Path(temporary_directory) / 'second'),
                )


if __name__ == '__main__':
    unittest.main()
