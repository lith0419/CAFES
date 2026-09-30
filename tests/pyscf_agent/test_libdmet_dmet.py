from __future__ import annotations

import contextlib
import copy
import gc
import io
import math
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from pyscf_agent.contracts import task_spec_from_dict
from pyscf_agent.backend.validation import _validate_model_solver_spec
from pyscf_agent.backend.model_hamiltonian.solver import (
    generate_model_hamiltonian_input_script,
    run_model_hamiltonian_solver,
)
from pyscf_agent.backend.result_artifacts import write_result_artifacts
from pyscf_agent.embedding.reference_density import build_reference_density_seed
from pyscf_agent.providers.libdmet import (
    DMET_RESULT_SCHEMA,
    libdmet_availability,
    normalize_dmet_options,
    validate_dmet_model_request,
)
from pyscf_agent.providers.libdmet.dmet import (
    _LIBDMET_NATIVE_FIT_BETA,
    _embedding_basis_policy_for_mean_field,
    _embedding_electron_count_from_projected_density,
    _fit_native_lattice_correlation_potential,
    _mean_field_rdm1_change,
    _dmet_quality_checks,
    _native_convergence_contract,
    _native_dmet_converged,
    _prepare_interacting_bath_mean_field,
    _solve_impurity_hamiltonians_with_fitting,
)
from pyscf_agent.providers.libdmet.block2_impurity import Block2DmetImpuritySolver
from pyscf_agent.providers.block2 import block2_availability
from pyscf_agent.providers.libdmet.fragment_fitting import (
    FragmentLocalCorrelationPotential,
    fit_fragment_local_correlation_potential,
)
from pyscf_agent.providers.libdmet.translation import audit_builder_translation
from pyscf_agent.registry import default_registry
from pyscf_agent.registry.platform import artifact_kind_is_registered
from pyscf_agent.result_quality import evaluate_quality_checks
from pyscf_agent.workflow_modules import WorkflowCompilationError, compile_task_workflow


def _ring_spec(site_count: int = 6, onsite_u: float = 4.0):
    return {
        'schema': 'pyscf-agent.model-hamiltonian.v1',
        'model': 'hubbard',
        'representation': 'finite_cluster',
        'dimension': 1,
        'preset': 'ring',
        'boundary': 'periodic',
        'energy_unit': 'a.u.',
        'nelec': [site_count // 2, site_count // 2],
        'sites': [
            {
                'id': index,
                'x': float(index),
                'y': 0.0,
                'epsilon': 0.0,
                'U': onsite_u,
            }
            for index in range(site_count)
        ],
        'bonds': [
            {
                'id': index,
                'source': index,
                'target': (index + 1) % site_count,
                't': -1.0,
                'V': 0.0,
                'effective_t': -1.0,
                'effective_V': 0.0,
                'periodic': index == site_count - 1,
            }
            for index in range(site_count)
        ],
    }


def _square_spec(lx: int = 2, ly: int = 2):
    coordinates = tuple((float(x), float(y)) for y in range(ly) for x in range(lx))
    pairs = []
    for y in range(ly):
        for x in range(lx):
            source = x + lx * y
            pairs.append((source, ((x + 1) % lx) + lx * y))
            pairs.append((source, x + lx * ((y + 1) % ly)))
    site_count = lx * ly
    return {
        'schema': 'pyscf-agent.model-hamiltonian.v1',
        'model': 'hubbard',
        'representation': 'finite_cluster',
        'dimension': 2,
        'preset': 'square',
        'boundary': 'periodic',
        'energy_unit': 'a.u.',
        'nelec': [site_count // 2, site_count // 2],
        'sites': [
            {'id': index, 'x': x, 'y': y, 'epsilon': 0.0, 'U': 4.0}
            for index, (x, y) in enumerate(coordinates)
        ],
        'bonds': [
            {
                'id': index,
                'source': source,
                'target': target,
                't': -1.0,
                'V': 0.0,
                'effective_t': -1.0,
                'effective_V': 0.0,
                'periodic': True,
            }
            for index, (source, target) in enumerate(pairs)
        ],
    }


def _multi_site_cell_spec(basis_size: int, lx: int = 2, ly: int = 1):
    cells = [(x, y) for y in range(ly) for x in range(lx)]
    site_records = []
    site_id = 0
    for basis_index in range(basis_size):
        for x, y in cells:
            site_records.append({
                'id': site_id,
                'x': float(x) + 0.1 * basis_index,
                'y': float(y) + 0.1 * basis_index,
                'cell_index': [x, y],
                'basis_index': basis_index,
                'epsilon': 0.0,
                'U': 4.0,
            })
            site_id += 1
    site_count = len(site_records)
    return {
        'schema': 'pyscf-agent.model-hamiltonian.v1',
        'model': 'hubbard',
        'representation': 'finite_cluster',
        'dimension': 2,
        'preset': 'kagome' if basis_size == 3 else 'honeycomb',
        'boundary': 'open',
        'energy_unit': 'a.u.',
        'nelec': [site_count // 2, site_count - site_count // 2],
        'primitive_cell': {
            'vectors': [[1.0, 0.0], [0.0, 1.0]],
            'repetitions': [lx, ly],
            'basis_size': basis_size,
        },
        'sites': site_records,
        'bonds': [
            {
                'id': index,
                'source': index,
                'target': index + 1,
                't': -1.0,
                'V': 0.0,
                'effective_t': -1.0,
                'effective_V': 0.0,
            }
            for index in range(site_count - 1)
        ],
    }


def _periodic_honeycomb_spec(lx: int = 4, ly: int = 4):
    site_ids = {}
    sites = []
    for y in range(ly):
        for x in range(lx):
            for basis_index, sublattice in enumerate(('a', 'b')):
                site_id = len(sites)
                site_ids[(x, y, basis_index)] = site_id
                sites.append({
                    'id': site_id,
                    'x': float(x) + 0.5 * float(y),
                    'y': 0.75 * float(basis_index) + float(y),
                    'cell_index': [x, y],
                    'basis_index': basis_index,
                    'sublattice': sublattice,
                    'epsilon': 0.0,
                    'U': 4.0,
                })
    bonds = []
    for y in range(ly):
        for x in range(lx):
            source = site_ids[(x, y, 0)]
            for target_cell in ((x, y), ((x - 1) % lx, y), (x, (y - 1) % ly)):
                bonds.append({
                    'id': len(bonds),
                    'source': source,
                    'target': site_ids[(target_cell[0], target_cell[1], 1)],
                    't': -1.0,
                    'V': 0.0,
                    'effective_t': -1.0,
                    'effective_V': 0.0,
                    'periodic': target_cell != (x, y),
                })
    site_count = len(sites)
    return {
        'schema': 'pyscf-agent.model-hamiltonian.v1',
        'model': 'hubbard',
        'representation': 'finite_cluster',
        'dimension': 2,
        'preset': 'honeycomb',
        'boundary': 'periodic',
        'energy_unit': 'a.u.',
        'nelec': [site_count // 2, site_count // 2],
        'primitive_cell': {
            'vectors': [[1.0, 0.0], [0.5, 1.0]],
            'repetitions': [lx, ly],
            'basis_size': 2,
        },
        'sites': sites,
        'bonds': bonds,
    }


class LibDmetContractTests(unittest.TestCase):
    def test_fragment_local_correlation_potential_ties_equivalent_blocks(self):
        value = np.zeros((2, 4, 4), dtype=float)
        value[:, 0, 0] = 1.0
        value[:, 1, 1] = 2.0
        value[:, 2, 2] = 3.0
        value[:, 3, 3] = 4.0
        fragments = [
            {'fragment_id': 'a', 'orbital_indices': [0, 1]},
            {'fragment_id': 'b', 'orbital_indices': [2, 3]},
        ]

        potential = FragmentLocalCorrelationPotential(
            value,
            fragments,
            restricted=False,
            tied=True,
        )
        parameters = np.array(potential.param, copy=True)
        parameters[0] = 7.0
        potential.update(parameters)
        matrix = potential.get()

        self.assertEqual(potential.length(), 6)
        self.assertAlmostEqual(matrix[0, 0, 0], 7.0)
        self.assertAlmostEqual(matrix[0, 2, 2], 7.0)
        self.assertAlmostEqual(matrix[0, 0, 2], 0.0)
        self.assertAlmostEqual(matrix[1, 1, 3], 0.0)

    def test_auto_fragments_preserve_builder_primitive_cells(self):
        errors, configuration = validate_dmet_model_request(_multi_site_cell_spec(3))

        self.assertEqual(errors, [])
        self.assertEqual(configuration['primitive_cell_basis_size'], 3)
        self.assertEqual(configuration['fragment_site_counts'], [3, 3])
        self.assertEqual(
            [fragment['site_ids'] for fragment in configuration['fragments']],
            [[0, 2, 4], [1, 3, 5]],
        )
        self.assertTrue(all(
            fragment['metadata']['selection'] == 'primitive_cell_shape'
            for fragment in configuration['fragments']
        ))

    def test_native_lattice_default_does_not_infer_fragment_size_from_parity(self):
        errors, configuration = validate_dmet_model_request(_ring_spec())

        self.assertEqual(errors, [])
        self.assertEqual(configuration['impurity_shape'], [1])
        self.assertEqual(configuration['fragment_site_counts'], [1])

    def test_cell_shape_groups_complete_multi_site_cells(self):
        errors, configuration = validate_dmet_model_request(
            _multi_site_cell_spec(2, lx=4),
            {'impurity_shape': [2, 1]},
        )

        self.assertEqual(errors, [])
        self.assertEqual(configuration['fragment_site_counts'], [4, 4])
        self.assertEqual(
            [fragment['site_ids'] for fragment in configuration['fragments']],
            [[0, 4, 1, 5], [2, 6, 3, 7]],
        )

    def test_site_count_is_rejected_for_multi_site_primitive_cells(self):
        errors, configuration = validate_dmet_model_request(
            _multi_site_cell_spec(2),
            {'impurity_size': 2},
        )

        self.assertIsNone(configuration)
        self.assertTrue(any('ambiguous for a multi-site primitive cell' in item for item in errors))

    def test_options_and_model_contract_are_normalized(self):
        automatic = normalize_dmet_options({})
        self.assertEqual(automatic['execution_mode'], 'auto')
        self.assertEqual(automatic['fragment_definition'], 'auto')

        options = normalize_dmet_options({
            'impurity_solver': 'exact_diagonalization',
            'impurity_shape': '2',
            'reference': 'unrestricted',
        })
        errors, configuration = validate_dmet_model_request(_ring_spec(), options)

        self.assertEqual(errors, [])
        self.assertEqual(options['impurity_solver'], 'fci')
        self.assertEqual(configuration['impurity_shape'], [2])
        self.assertEqual(configuration['reference'], 'unrestricted')
        self.assertEqual(configuration['libdmet_hopping_parameter'], 1.0)
        self.assertTrue(configuration['interacting_bath'])
        self.assertEqual(configuration['initial_correlation_potential'], 'zero')
        self.assertEqual(configuration['reference_density_guess'], 'pm')
        self.assertEqual(options['max_iterations'], 50)
        self.assertEqual(options['energy_tolerance'], 1.0e-6)
        self.assertEqual(options['density_tolerance'], 1.0e-4)
        self.assertEqual(options['density_fit_tolerance'], 1.0e-4)
        self.assertNotIn('nroots', options)

        noninteracting = normalize_dmet_options({'interacting_bath': False})
        self.assertFalse(noninteracting['interacting_bath'])
        with self.assertRaisesRegex(ValueError, 'supports only zero'):
            normalize_dmet_options({'initial_correlation_potential': 'antiferromagnetic'})
        with self.assertRaisesRegex(ValueError, 'replace correlation_potential_tolerance'):
            normalize_dmet_options({'correlation_potential_tolerance': 1.0e-5})

    def test_unknown_dmet_options_are_rejected_instead_of_using_defaults(self):
        for key in ('density_fit_tolerence', 'conv_tol', 'max_iteratons'):
            with self.subTest(key=key):
                supplied = {key: 1e-8}
                with self.assertRaisesRegex(ValueError, 'Unknown DMET solver option.*' + key):
                    normalize_dmet_options(supplied)
                self.assertEqual(supplied, {key: 1e-8})

    def test_task_validation_rejects_misspelled_dmet_option(self):
        model_spec = _ring_spec()
        task_spec = task_spec_from_dict({
            'task_type': 'model_hamiltonian',
            'solver': {'name': 'dmet', 'options': {'density_fit_tolerence': 1e-8}},
            'model_hamiltonian': {'spec': model_spec},
            'analysis': {'outputs': ['energy']},
        })
        errors = _validate_model_solver_spec(task_spec, model_spec)
        self.assertTrue(any(
            error['code'] == 'invalid_dmet_model_request'
            and 'density_fit_tolerence' in error['message']
            for error in errors
        ), errors)

    def test_registered_dmet_options_preserve_explicit_values_and_round_trip(self):
        supplied = {
            'execution_mode': 'finite_graph', 'fragment_definition': 'site_count',
            'impurity_size': 2, 'density_fit_tolerance': 1e-8,
        }
        original = copy.deepcopy(supplied)
        normalized = normalize_dmet_options(supplied)
        for key, value in supplied.items():
            self.assertEqual(normalized[key], value)
        self.assertEqual(normalize_dmet_options(normalized), normalized)
        self.assertEqual(supplied, original)

    def test_block2_impurity_options_are_normalized_and_owned_by_dmet(self):
        options = normalize_dmet_options({
            'impurity_solver': 'dmrg',
            'reference': 'unrestricted',
            'impurity_solver_options': {
                'preset': 'screening',
                'orbital_ordering': 'fiedler',
            },
        })

        self.assertEqual(options['impurity_solver'], 'block2_dmrg')
        self.assertEqual(options['impurity_solver_options']['preset'], 'screening')
        self.assertEqual(options['impurity_solver_options']['orbital_ordering'], 'fiedler')
        with self.assertRaisesRegex(ValueError, 'manages these block2 impurity options internally'):
            normalize_dmet_options({
                'impurity_solver': 'block2_dmrg',
                'impurity_solver_options': {'nroots': 2},
            })
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            normalize_dmet_options({
                'impurity_solver': 'fci',
                'impurity_solver_options': {'preset': 'screening'},
            })

    def test_ccsd_beta_is_optional_positive_finite_and_validated_at_task_boundary(self):
        options = {'impurity_solver': 'ccsd', 'impurity_solver_options': {'beta': '1000'}}
        normalized = normalize_dmet_options(options)
        self.assertEqual(normalized['impurity_solver_options'], {'beta': 1000.0})
        self.assertEqual(normalize_dmet_options(normalized), normalized)
        self.assertEqual(normalize_dmet_options({'impurity_solver': 'ccsd'})['impurity_solver_options'], {'beta': 1000.0})
        for invalid in (None, True, 0, -1, float('inf'), float('nan'), [], 'bad'):
            with self.subTest(beta=invalid), self.assertRaisesRegex(ValueError, 'beta'):
                normalize_dmet_options({**options, 'impurity_solver_options': {'beta': invalid}})
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            normalize_dmet_options({**options, 'impurity_solver_options': {'betta': 1000}})
        self.assertEqual(
            normalize_dmet_options({**options, 'impurity_solver': 'fci'})['impurity_solver_options'],
            {'beta': 1000.0},
        )
        spec = _ring_spec()
        task = task_spec_from_dict({
            'task_type': 'model_hamiltonian',
            'model_hamiltonian': {'spec': spec},
            'solver': {'name': 'dmet', 'options': options},
        })
        self.assertEqual(_validate_model_solver_spec(task, spec), [])

    def test_ccsd_beta_does_not_change_fci_or_dmrg_policy(self):
        from pyscf_agent.providers.libdmet.dmet import (
            _lattice_scf_beta, _density_fit_beta, _ccsd_smearing_metadata,
        )
        for solver in ('fci', 'block2_dmrg'):
            options = normalize_dmet_options({'impurity_solver': solver})
            self.assertNotIn('beta', options)
            self.assertNotIn('beta', options['impurity_solver_options'])
            self.assertEqual(_lattice_scf_beta(options), math.inf)
            self.assertEqual(_density_fit_beta(options), 1000.0)
            self.assertEqual(_ccsd_smearing_metadata(options), {})
        for requested in ({}, {'beta': 250}):
            options = normalize_dmet_options({'impurity_solver': 'ccsd', 'impurity_solver_options': requested})
            expected = requested.get('beta', 1000.0)
            self.assertEqual(_lattice_scf_beta(options), expected)
            self.assertEqual(_density_fit_beta(options), expected)
            self.assertEqual(normalize_dmet_options(options), options)

    def test_ccsd_smearing_can_be_disabled_without_nonfinite_input(self):
        from pyscf_agent.providers.libdmet.dmet import (
            _lattice_scf_beta, _density_fit_beta, _ccsd_smearing_metadata,
        )
        options = normalize_dmet_options({
            'impurity_solver': 'ccsd',
            'impurity_solver_options': {'smearing': False},
        })
        self.assertEqual(normalize_dmet_options(options), options)
        self.assertEqual(_lattice_scf_beta(options), math.inf)
        self.assertEqual(_density_fit_beta(options), math.inf)
        metadata = _ccsd_smearing_metadata(options)['smearing']
        self.assertEqual(metadata['method'], 'none')
        self.assertEqual(metadata['sigma'], 0.0)
        for key in ('beta', 'lattice_scf_beta', 'density_fit_beta', 'impurity_scf_beta'):
            self.assertIsNone(metadata[key])
        for invalid in (None, [], 'maybe', 0):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, 'smearing'):
                normalize_dmet_options({
                    'impurity_solver': 'ccsd',
                    'impurity_solver_options': {'smearing': invalid},
                })

    def test_task_validation_accepts_block2_as_nested_dmet_impurity_solver(self):
        model_spec = _ring_spec()
        task_spec = task_spec_from_dict({
            'task_type': 'model_hamiltonian',
            'solver': {
                'name': 'dmet',
                'options': {
                    'impurity_solver': 'block2_dmrg',
                    'impurity_solver_options': {'preset': 'screening'},
                },
            },
            'model_hamiltonian': {'spec': model_spec},
            'analysis': {'outputs': ['energy']},
        })

        self.assertEqual(_validate_model_solver_spec(task_spec, model_spec), [])

    def test_block2_impurity_bridge_preserves_libdmet_rdm_conventions(self):
        captured = []
        rdm1_alpha = np.diag([0.8, 0.2])
        rdm1_beta = np.diag([0.3, 0.7])
        rdm2_aa = np.full((2, 2, 2, 2), 0.11)
        rdm2_ab = np.full((2, 2, 2, 2), 0.22)
        rdm2_bb = np.full((2, 2, 2, 2), 0.33)

        def fake_runner(hamiltonian, options, **kwargs):
            captured.append((hamiltonian, options, kwargs))
            return {
                'energy': -1.25,
                'converged': True,
                'convergence': {'converged': True},
                'checkpoint_manifest': {'schema': 'test', 'files': []},
                '_transient_dmrg_arrays': {
                    'rdm1_alpha': rdm1_alpha,
                    'rdm1_beta': rdm1_beta,
                    'rdm2_alpha_alpha': rdm2_aa,
                    'rdm2_alpha_beta': rdm2_ab,
                    'rdm2_beta_beta': rdm2_bb,
                },
            }

        impurity_hamiltonian = SimpleNamespace(
            norb=2,
            H0=0.4,
            H1={'cd': np.asarray([np.eye(2), 2.0 * np.eye(2)])},
            H2={'ccdd': np.asarray([
                np.full((2, 2, 2, 2), 1.0),
                np.full((2, 2, 2, 2), 2.0),
                np.full((2, 2, 2, 2), 3.0),
            ])},
        )
        with tempfile.TemporaryDirectory() as scratch:
            solver = Block2DmetImpuritySolver(
                restricted=False,
                spin=0,
                scratch_directory=scratch,
                runner=fake_runner,
            )
            density, energy = solver.run(impurity_hamiltonian, nelec=2)

        electronic_hamiltonian, run_options, run_kwargs = captured[0]
        self.assertEqual(electronic_hamiltonian.representation, 'unrestricted_spatial_orbital')
        self.assertTrue(np.array_equal(electronic_hamiltonian.g2e[0], impurity_hamiltonian.H2['ccdd'][0]))
        self.assertTrue(np.array_equal(electronic_hamiltonian.g2e[1], impurity_hamiltonian.H2['ccdd'][2]))
        self.assertTrue(np.array_equal(electronic_hamiltonian.g2e[2], impurity_hamiltonian.H2['ccdd'][1]))
        self.assertTrue(run_options['compute_1rdm'])
        self.assertTrue(run_options['compute_2rdm'])
        self.assertEqual(run_options['nroots'], 1)
        self.assertEqual(run_options['symmetry'], 'sz')
        self.assertTrue(run_kwargs['compute_2rdm'])
        self.assertTrue(np.array_equal(density, np.asarray([rdm1_alpha, rdm1_beta])))
        self.assertTrue(np.array_equal(solver.twopdm, np.asarray([rdm2_aa, rdm2_bb, rdm2_ab])))
        self.assertEqual(energy, -1.25)

    def test_block2_impurity_restart_pins_first_fiedler_permutation(self):
        captured = []
        rdm1_alpha = np.diag([0.8, 0.2])
        rdm1_beta = np.diag([0.3, 0.7])
        rdm2 = np.zeros((2, 2, 2, 2))

        def fake_runner(_hamiltonian, options, **_kwargs):
            captured.append(copy.deepcopy(options))
            return {
                'energy': -1.25,
                'converged': True,
                'convergence': {'converged': True},
                'orbital_ordering': {
                    'requested_method': options.get('orbital_ordering'),
                    'requested_order': list(options.get('orbital_order') or []),
                    'permutation': [1, 0],
                },
                'checkpoint_manifest': {
                    'schema': 'test',
                    'files': [{'relative_path': 'GS-MPS_INFO'}],
                    'orbital_ordering': {
                        'requested_method': options.get('orbital_ordering'),
                        'requested_order': list(options.get('orbital_order') or []),
                        'permutation': [1, 0],
                    },
                },
                '_transient_dmrg_arrays': {
                    'rdm1_alpha': rdm1_alpha,
                    'rdm1_beta': rdm1_beta,
                    'rdm2_alpha_alpha': rdm2,
                    'rdm2_alpha_beta': rdm2,
                    'rdm2_beta_beta': rdm2,
                },
            }

        impurity_hamiltonian = SimpleNamespace(
            norb=2,
            H0=0.0,
            H1={'cd': np.asarray([np.eye(2), np.eye(2)])},
            H2={'ccdd': np.asarray([rdm2, rdm2, rdm2])},
        )
        with tempfile.TemporaryDirectory() as scratch:
            solver = Block2DmetImpuritySolver(
                restricted=False,
                spin=0,
                options={'orbital_ordering': 'fiedler'},
                scratch_directory=scratch,
                runner=fake_runner,
            )
            solver.run(impurity_hamiltonian, nelec=2)
            solver.run(impurity_hamiltonian, nelec=2)

        self.assertEqual(captured[0]['orbital_ordering'], 'fiedler')
        self.assertEqual(captured[1]['orbital_ordering'], 'manual')
        self.assertEqual(captured[1]['orbital_order'], [1, 0])
        restart_ordering = captured[1]['restart_manifest']['orbital_ordering']
        self.assertEqual(restart_ordering['requested_method'], 'manual')
        self.assertEqual(restart_ordering['requested_order'], [1, 0])
        self.assertEqual(restart_ordering['permutation'], [1, 0])

    def test_fractional_mean_field_occupations_select_the_eigenvalue_bath(self):
        fractional_kind, fractional = _embedding_basis_policy_for_mean_field(
            {'mo_occ': np.asarray([0.0, 0.5, 1.0])},
            np,
        )
        idempotent_kind, idempotent = _embedding_basis_policy_for_mean_field(
            {'mo_occ': np.asarray([0.0, 1.0])},
            np,
        )

        self.assertEqual(fractional_kind, 'eig')
        self.assertEqual(fractional['fractional_occupation_count'], 1)
        self.assertEqual(fractional['reason'], 'fractional_mean_field_occupations')
        self.assertEqual(idempotent_kind, 'svd')
        self.assertEqual(idempotent['fractional_occupation_count'], 0)

    def test_embedding_electron_count_comes_from_the_projected_density(self):
        projected_density = np.asarray([
            np.diag([0.5, 0.5, 0.5, 0.5]),
            np.diag([0.5, 0.5, 0.5, 0.5]),
        ])

        electron_count, diagnostics = _embedding_electron_count_from_projected_density(
            projected_density,
            4,
            'unrestricted',
            np,
        )

        self.assertEqual(electron_count, 4)
        self.assertEqual(diagnostics['projected_density_traces'], [2.0, 2.0])
        self.assertEqual(
            diagnostics['selection'],
            'projected_mean_field_density_trace',
        )
        self.assertEqual(diagnostics['selected_electron_count'], 4)
        self.assertEqual(diagnostics['rounded_electron_count'], 4)
        fractional_count, fractional_diagnostics = (
            _embedding_electron_count_from_projected_density(
                np.asarray([np.diag([1.0] * 6 + [0.0] * 6)]),
                12,
                'restricted',
                np,
                fractional_occupation_count=6,
                fractional_sector_electron_count=8,
            )
        )
        self.assertEqual(fractional_count, 8)
        self.assertEqual(
            fractional_diagnostics['selection'],
            'valence_sector_for_fractional_mean_field',
        )
        self.assertEqual(fractional_diagnostics['projected_electron_count'], 6.0)
        self.assertEqual(fractional_diagnostics['selected_electron_count'], 8)
        self.assertIsNone(fractional_diagnostics['rounded_electron_count'])
        self.assertEqual(fractional_diagnostics['fractional_sector_electron_count'], 8)
        with self.assertRaisesRegex(ValueError, 'not an integer electron count'):
            _embedding_electron_count_from_projected_density(
                np.asarray([np.diag([0.5, 0.5, 0.5])]),
                3,
                'restricted',
                np,
            )

    def test_native_convergence_uses_energy_and_mean_field_rdm1_only(self):
        configuration = normalize_dmet_options()
        previous = np.array([[[0.5, 0.1], [0.1, 0.5]]])
        current = np.array([[[0.5, 0.10009], [0.10009, 0.5]]])

        density_change = _mean_field_rdm1_change(current, previous, np)

        self.assertAlmostEqual(density_change, 9.0e-5)
        self.assertTrue(_native_dmet_converged(
            9.0e-7,
            density_change,
            9.0e-5,
            configuration,
        ))
        self.assertFalse(_native_dmet_converged(
            1.1e-6,
            density_change,
            9.0e-5,
            configuration,
        ))
        self.assertFalse(_native_dmet_converged(
            9.0e-7,
            1.1e-4,
            9.0e-5,
            configuration,
        ))
        # The density-fit residual is a diagnostic, not a stopping criterion.
        self.assertTrue(_native_dmet_converged(
            9.0e-7,
            density_change,
            3.0e-3,
            configuration,
        ))
        contract = _native_convergence_contract(configuration, energy_measure='finite_graph_total_energy')
        self.assertEqual(contract['criteria'], 'energy_and_mean_field_rdm1')
        self.assertTrue(contract['density_fit_error_is_diagnostic_only'])
        result = {
            'energy': -4.0, 'energy_per_site': -1.0, 'site_count': 4,
            'iteration_history': {
                'convergence_contract': contract,
                'records': [{'energy_change': 1.0e-8, 'mean_field_rdm1_change_max_abs': 1.0e-6,
                             'density_fit_error': 3.0e-3}],
            },
        }
        quality = evaluate_quality_checks(_dmet_quality_checks(result))
        fit_check = next(item for item in quality['checks'] if item['id'] == 'dmet_density_fit_error')
        self.assertEqual(fit_check['status'], 'failed')
        self.assertFalse(fit_check['required'])
        self.assertNotIn('dmet_density_fit_error', quality['required_failed_ids'])
        self.assertAlmostEqual(
            _mean_field_rdm1_change(np.array([[[0.5]]]), None, np),
            0.5,
        )

    def test_flat_chemical_potential_response_raises_a_dmet_domain_error(self):
        provider = SimpleNamespace(
            SolveImpHam_with_fitting=lambda: (_ for _ in ()).throw(
                ValueError(
                    'Cannot calculate a linear regression if all x values are identical'
                )
            )
        )

        with self.assertRaisesRegex(
            RuntimeError,
            'DMET impurity chemical-potential fitting encountered a flat',
        ):
            _solve_impurity_hamiltonians_with_fitting(provider)

    def test_flat_response_matches_installed_scipy_and_preserves_unrelated_errors(self):
        from scipy.stats import linregress

        def fitting():
            regression = linregress([1, 1], [0, 2])
            return (np.zeros((1, 1)), 0.0, None, regression.intercept)

        provider = SimpleNamespace(SolveImpHam_with_fitting=fitting)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            with self.assertRaisesRegex(RuntimeError, 'chemical-potential fitting'):
                _solve_impurity_hamiltonians_with_fitting(provider)

        original = ValueError('unrelated numerical error')
        provider.SolveImpHam_with_fitting = mock.Mock(side_effect=original)
        with self.assertRaises(ValueError) as captured:
            _solve_impurity_hamiltonians_with_fitting(provider)
        self.assertIs(captured.exception, original)

        valid = (np.zeros((1, 1)), 0.0, None, 0.25)
        provider.SolveImpHam_with_fitting = mock.Mock(return_value=valid)
        self.assertIs(_solve_impurity_hamiltonians_with_fitting(provider), valid)

    def test_execution_mode_can_force_finite_graph_or_require_translation(self):
        finite_errors, finite_configuration = validate_dmet_model_request(
            _ring_spec(),
            {
                'execution_mode': 'finite_graph',
                'fragment_definition': 'primitive_cell',
            },
        )

        self.assertEqual(finite_errors, [])
        self.assertEqual(finite_configuration['requested_execution_mode'], 'finite_graph')
        self.assertEqual(finite_configuration['execution_mode'], 'finite_graph')
        self.assertEqual(finite_configuration['fragment_site_counts'], [1] * 6)
        self.assertTrue(finite_configuration['translation_symmetry']['eligible'])

        translated_errors, translated_configuration = validate_dmet_model_request(
            _ring_spec(),
            {
                'execution_mode': 'translational',
                'fragment_definition': 'primitive_cell',
            },
        )
        self.assertEqual(translated_errors, [])
        self.assertEqual(translated_configuration['impurity_shape'], [1])
        self.assertEqual(translated_configuration['fragment_site_counts'], [1])

        perturbed = _ring_spec()
        perturbed['sites'][0]['U'] = 4.25
        invalid_errors, invalid_configuration = validate_dmet_model_request(
            perturbed,
            {'execution_mode': 'translational'},
        )

        self.assertIsNone(invalid_configuration)
        self.assertTrue(any(
            'Translated representative DMET was requested' in item
            and 'uniform_onsite_u' in item
            for item in invalid_errors
        ))

    def test_af_density_seed_is_staggered_without_changing_particle_counts(self):
        errors, configuration = validate_dmet_model_request(
            _ring_spec(),
            {
                'impurity_shape': [2],
                'reference': 'unrestricted',
                'reference_density_guess': 'af',
            },
        )

        self.assertEqual(errors, [])
        density, metadata = build_reference_density_seed(
            _ring_spec(),
            configuration,
            2,
            np=np,
        )

        self.assertEqual(metadata['strategy'], 'af')
        self.assertEqual(metadata['scope'], 'initial_uhf_density_only')
        self.assertAlmostEqual(metadata['applied_bias'], 0.05)
        self.assertTrue(np.allclose(np.diag(density[0]), [0.55, 0.45]))
        self.assertTrue(np.allclose(np.diag(density[1]), [0.45, 0.55]))
        self.assertAlmostEqual(float(np.trace(density[0])), 1.0)
        self.assertAlmostEqual(float(np.trace(density[1])), 1.0)

    def test_fm_density_seed_uses_the_fixed_nonzero_spin_sector(self):
        spec = _ring_spec()
        spec['nelec'] = [4, 2]
        errors, configuration = validate_dmet_model_request(
            spec,
            {
                'impurity_shape': [2],
                'reference': 'unrestricted',
                'reference_density_guess': 'fm',
            },
        )

        self.assertEqual(errors, [])
        density, metadata = build_reference_density_seed(
            spec,
            configuration,
            2,
            np=np,
        )

        self.assertEqual(metadata['strategy'], 'fm')
        self.assertEqual(metadata['bias_kind'], 'fixed_spin_sector')
        self.assertTrue(np.allclose(np.diag(density[0]), [2.0 / 3.0] * 2))
        self.assertTrue(np.allclose(np.diag(density[1]), [1.0 / 3.0] * 2))

    def test_cdw_density_seed_is_a_staggered_charge_bias_in_both_spins(self):
        for reference in ('unrestricted', 'restricted'):
            with self.subTest(reference=reference):
                errors, configuration = validate_dmet_model_request(
                    _ring_spec(),
                    {
                        'impurity_shape': [2],
                        'reference': reference,
                        'reference_density_guess': 'cdw',
                    },
                )

                self.assertEqual(errors, [])
                density, metadata = build_reference_density_seed(
                    _ring_spec(),
                    configuration,
                    2,
                    np=np,
                )

                self.assertEqual(metadata['strategy'], 'cdw')
                self.assertEqual(metadata['bias_kind'], 'staggered_charge_density')
                self.assertAlmostEqual(metadata['applied_bias'], 0.05)
                for spin in (0, 1):
                    self.assertTrue(np.allclose(np.diag(density[spin]), [0.55, 0.45]))
                    self.assertAlmostEqual(float(np.trace(density[spin])), 1.0)

    def test_cdw_reference_density_requires_a_bipartite_graph(self):
        from pyscf_agent.embedding.reference_density import validate_reference_density_guess

        errors = validate_reference_density_guess(
            _ring_spec(3),
            {'reference_density_guess': 'cdw', 'reference': 'restricted'},
        )
        self.assertTrue(any('bipartite' in item for item in errors))

    def test_af_and_fm_reference_density_boundaries_are_validated(self):
        errors, configuration = validate_dmet_model_request(
            _ring_spec(),
            {
                'impurity_shape': [2],
                'reference': 'restricted',
                'reference_density_guess': 'af',
            },
        )
        self.assertIsNone(configuration)
        self.assertTrue(any('requires reference=unrestricted' in item for item in errors))

        errors, configuration = validate_dmet_model_request(
            _ring_spec(),
            {
                'impurity_shape': [2],
                'reference': 'unrestricted',
                'reference_density_guess': 'fm',
            },
        )
        self.assertIsNone(configuration)
        self.assertTrue(any('nonzero target spin sector' in item for item in errors))

    def test_article_scale_square_af_seed_requires_a_two_sublattice_fragment(self):
        spec = _square_spec(4, 4)
        for site in spec['sites']:
            site['sublattice'] = (
                'a'
                if (int(site['x']) + int(site['y'])) % 2 == 0
                else 'b'
            )

        one_site_errors, one_site_configuration = validate_dmet_model_request(
            spec,
            {
                'impurity_shape': [1, 1],
                'reference': 'unrestricted',
                'reference_density_guess': 'af',
            },
        )
        self.assertIsNone(one_site_configuration)
        self.assertIn(
            'AF reference density requires both sublattices in the reference cell',
            one_site_errors,
        )

        errors, configuration = validate_dmet_model_request(
            spec,
            {
                'impurity_shape': [2, 2],
                'reference': 'unrestricted',
                'reference_density_guess': 'af',
            },
        )
        self.assertEqual(errors, [])
        self.assertEqual(configuration['execution_mode'], 'translational')
        self.assertEqual(configuration['translation_backend'], 'native_lattice')
        self.assertEqual(configuration['lattice_shape'], [4, 4])
        self.assertEqual(configuration['impurity_shape'], [2, 2])
        self.assertEqual(configuration['fragments'][0]['site_ids'], [0, 1, 4, 5])

        density, metadata = build_reference_density_seed(
            spec,
            configuration,
            4,
            np=np,
        )
        self.assertEqual(metadata['site_ids'], [0, 1, 4, 5])
        self.assertAlmostEqual(float(np.trace(density[0])), 2.0)
        self.assertAlmostEqual(float(np.trace(density[1])), 2.0)
        self.assertGreater(float(np.max(np.abs(density[0] - density[1]))), 0.0)

    def test_dmet_rejects_nroots_even_when_set_to_one(self):
        with self.assertRaisesRegex(ValueError, 'remove nroots'):
            normalize_dmet_options({'nroots': 1})

    def test_dmet_generated_input_has_no_root_solver_option(self):
        options = normalize_dmet_options({
            'impurity_solver': 'fci',
            'impurity_size': 1,
        })
        script = generate_model_hamiltonian_input_script(
            _ring_spec(),
            solver_name='dmet',
            outputs=['energy'],
            solver_options=options,
        )

        self.assertNotIn("'nroots'", script)

    def test_model_contract_preserves_nonregular_builder_graph(self):
        malformed = _ring_spec()
        malformed['bonds'] = malformed['bonds'][:-1]
        errors, configuration = validate_dmet_model_request(malformed)

        self.assertEqual(errors, [])
        self.assertEqual(configuration['execution_mode'], 'finite_graph')
        self.assertEqual(configuration['fragment_count'], 6)
        self.assertEqual(
            sorted(site_id for fragment in configuration['fragments'] for site_id in fragment['site_ids']),
            list(range(6)),
        )

    def test_local_site_perturbation_disables_translated_representative(self):
        uniform_errors, uniform_configuration = validate_dmet_model_request(_ring_spec())
        self.assertEqual(uniform_errors, [])
        self.assertEqual(uniform_configuration['execution_mode'], 'translational')
        self.assertTrue(uniform_configuration['translation_symmetry']['eligible'])

        perturbed = _ring_spec()
        perturbed['sites'][0]['U'] = 4.25
        errors, configuration = validate_dmet_model_request(perturbed)

        self.assertEqual(errors, [])
        self.assertEqual(configuration['execution_mode'], 'finite_graph')
        self.assertIn(
            'uniform_onsite_u',
            configuration['translation_symmetry']['failed_checks'],
        )
        self.assertEqual(configuration['fragment_count'], 6)

    def test_non_equivalent_multi_site_or_unsupported_lattice_uses_finite_graph(self):
        multi_site = _multi_site_cell_spec(3)
        multi_site['boundary'] = 'periodic'
        errors, configuration = validate_dmet_model_request(multi_site)

        self.assertEqual(errors, [])
        self.assertEqual(configuration['execution_mode'], 'finite_graph')
        self.assertIn(
            'translation_equivalent_graph',
            configuration['translation_symmetry']['failed_checks'],
        )

        triangular = _square_spec(3, 2)
        triangular['preset'] = 'triangular'
        errors, configuration = validate_dmet_model_request(triangular)

        self.assertEqual(errors, [])
        self.assertEqual(configuration['execution_mode'], 'finite_graph')
        self.assertIn(
            'translation_equivalent_graph',
            configuration['translation_symmetry']['failed_checks'],
        )

    def test_builder_honeycomb_uses_one_translated_primitive_cell(self):
        spec = _periodic_honeycomb_spec()
        audit = audit_builder_translation(spec)
        errors, configuration = validate_dmet_model_request(spec)

        self.assertTrue(audit['eligible'])
        self.assertEqual(audit['representative_site_ids'], [0, 1])
        self.assertEqual(audit['basis_bipartition'], [[0], [1]])
        self.assertEqual(errors, [])
        self.assertEqual(configuration['execution_mode'], 'translational')
        self.assertEqual(configuration['translation_backend'], 'builder_primitive_cell')
        self.assertEqual(configuration['impurity_shape'], [1, 1])
        self.assertEqual(configuration['fragment_count'], 1)
        self.assertEqual(configuration['fragment_site_counts'], [2])
        self.assertEqual(configuration['fragments'][0]['site_ids'], [0, 1])

    def test_builder_honeycomb_allows_basis_resolved_onsite_terms(self):
        spec = _periodic_honeycomb_spec()
        for site in spec['sites']:
            if site['basis_index'] == 0:
                site['epsilon'] = 1.0
                site['U'] = 4.0
            else:
                site['epsilon'] = 2.0
                site['U'] = 5.0

        audit = audit_builder_translation(spec)
        errors, configuration = validate_dmet_model_request(spec)

        self.assertTrue(audit['eligible'])
        self.assertEqual(errors, [])
        self.assertTrue(configuration['translation_symmetry']['eligible'])
        self.assertEqual(configuration['execution_mode'], 'translational')
        self.assertEqual(configuration['translation_backend'], 'builder_primitive_cell')
        self.assertEqual(configuration['fragment_count'], 1)
        self.assertEqual(
            [
                (item['basis_index'], item['epsilon'], item['U'])
                for item in configuration['translation_symmetry']['builder_audit']['basis_sites']
            ],
            [(0, 1.0, 4.0), (1, 2.0, 5.0)],
        )

    def test_builder_honeycomb_local_onsite_defect_disables_translation(self):
        spec = _periodic_honeycomb_spec()
        for site in spec['sites']:
            site['epsilon'] = 1.0 if site['basis_index'] == 0 else 2.0
        spec['sites'][-1]['epsilon'] = 2.25

        audit = audit_builder_translation(spec)
        errors, configuration = validate_dmet_model_request(spec)

        self.assertFalse(audit['eligible'])
        self.assertEqual(errors, [])
        self.assertFalse(configuration['translation_symmetry']['eligible'])
        self.assertEqual(configuration['execution_mode'], 'finite_graph')
        self.assertIn(
            'uniform_onsite_energy',
            configuration['translation_symmetry']['failed_checks'],
        )

    def test_native_lattice_allows_a_uniform_onsite_energy_offset(self):
        spec = _ring_spec()
        for site in spec['sites']:
            site['epsilon'] = 1.5

        errors, configuration = validate_dmet_model_request(spec)

        self.assertEqual(errors, [])
        self.assertTrue(configuration['translation_symmetry']['eligible'])
        self.assertEqual(configuration['execution_mode'], 'translational')

    def test_builder_translation_audit_rejects_one_edited_honeycomb_bond(self):
        spec = _periodic_honeycomb_spec()
        spec['bonds'].pop()
        errors, configuration = validate_dmet_model_request(spec)

        self.assertEqual(errors, [])
        self.assertEqual(configuration['execution_mode'], 'finite_graph')
        self.assertIn(
            'translation_equivalent_graph',
            configuration['translation_symmetry']['failed_checks'],
        )
        self.assertTrue(any(
            'Bond environment' in issue
            for issue in configuration['translation_symmetry']['topology_issues']
        ))

    def test_edited_square_connectivity_disables_native_square_rebuild(self):
        edited = _square_spec(3, 2)
        edited['bonds'][0]['source'] = 1
        edited['bonds'][0]['target'] = 4
        errors, configuration = validate_dmet_model_request(edited)

        self.assertEqual(errors, [])
        self.assertEqual(configuration['execution_mode'], 'finite_graph')
        self.assertIn(
            'translation_equivalent_graph',
            configuration['translation_symmetry']['failed_checks'],
        )
        self.assertTrue(any(
            'do not match' in issue
            or 'nearest-neighbor' in issue
            for issue in configuration['translation_symmetry']['topology_issues']
        ))

    def test_model_contract_accepts_builder_physics_and_rejects_adapter_boundaries(self):
        open_boundary = _ring_spec()
        open_boundary['boundary'] = 'open'
        open_errors, _ = validate_dmet_model_request(open_boundary)
        self.assertEqual(open_errors, [])

        extended = _ring_spec()
        extended['bonds'][0]['V'] = 1.0
        extended['bonds'][0]['effective_V'] = 1.0
        extended_errors, extended_configuration = validate_dmet_model_request(extended)
        self.assertEqual(extended_errors, [])
        self.assertEqual(extended_configuration['execution_mode'], 'finite_graph')
        self.assertTrue(extended_configuration['interacting_bath'])

        nonuniform = _ring_spec()
        nonuniform['sites'][0]['U'] = 6.0
        nonuniform['sites'][1]['epsilon'] = 0.25
        nonuniform['bonds'][0]['t'] = -0.5
        nonuniform['bonds'][0]['effective_t'] = -0.5
        nonuniform_errors, nonuniform_configuration = validate_dmet_model_request(nonuniform)
        self.assertEqual(nonuniform_errors, [])
        self.assertEqual(nonuniform_configuration['execution_mode'], 'finite_graph')

        spin_polarized = _ring_spec()
        spin_polarized['nelec'] = [4, 2]
        spin_errors, spin_configuration = validate_dmet_model_request(
            spin_polarized,
            {'impurity_shape': [2]},
        )
        self.assertEqual(spin_errors, [])
        self.assertEqual(spin_configuration['reference'], 'unrestricted')
        self.assertEqual(spin_configuration['spin_filling'], [4.0 / 6.0, 2.0 / 6.0])
        self.assertEqual(spin_configuration['mean_field_filling'], [4.0 / 6.0, 2.0 / 6.0])
        self.assertEqual(spin_configuration['libdmet_sz'], 2)
        self.assertEqual(spin_configuration['physical_sz'], 1.0)

        restricted_spin_errors, _ = validate_dmet_model_request(
            spin_polarized,
            {'impurity_shape': [2], 'reference': 'restricted'},
        )
        self.assertTrue(any('Restricted DMET requires nalpha=nbeta' in item for item in restricted_spin_errors))

        finite_graph_spin = _ring_spec()
        finite_graph_spin['boundary'] = 'open'
        finite_graph_spin['bonds'] = finite_graph_spin['bonds'][:-1]
        finite_graph_spin['nelec'] = [4, 2]
        finite_graph_spin_errors, _ = validate_dmet_model_request(finite_graph_spin)
        self.assertTrue(any('fragment-specific spin sectors' in item for item in finite_graph_spin_errors))

        whole_lattice_errors, _ = validate_dmet_model_request(
            _square_spec(),
            {'impurity_shape': [2, 2]},
        )
        self.assertTrue(any('environment site' in item for item in whole_lattice_errors))

        unrestricted_errors, unrestricted_configuration = validate_dmet_model_request(
            _square_spec(),
            {'impurity_shape': [1, 1], 'reference': 'unrestricted'},
        )
        self.assertEqual(unrestricted_errors, [])
        self.assertEqual(unrestricted_configuration['reference'], 'unrestricted')

    def test_explicit_fragments_use_builder_site_ids(self):
        errors, configuration = validate_dmet_model_request(
            _ring_spec(),
            {
                'fragments': [
                    {'fragment_id': 'left', 'site_ids': [0, 1, 2]},
                    {'fragment_id': 'right', 'site_ids': [3, 4, 5]},
                ],
            },
        )

        self.assertEqual(errors, [])
        self.assertEqual(configuration['execution_mode'], 'finite_graph')
        self.assertEqual(configuration['fragment_site_counts'], [3, 3])
        self.assertEqual(configuration['fragments'][1]['orbital_indices'], [3, 4, 5])

    def test_impurity_size_builds_an_explicit_finite_graph_partition(self):
        errors, configuration = validate_dmet_model_request(
            _ring_spec(),
            {'impurity_size': 2},
        )

        self.assertEqual(errors, [])
        self.assertEqual(configuration['execution_mode'], 'finite_graph')
        self.assertEqual(configuration['fragment_site_counts'], [2, 2, 2])
        self.assertEqual(
            [fragment['site_ids'] for fragment in configuration['fragments']],
            [[0, 1], [2, 3], [4, 5]],
        )

    def test_fragment_selectors_are_mutually_exclusive(self):
        errors, configuration = validate_dmet_model_request(
            _ring_spec(),
            {'impurity_size': 2, 'impurity_shape': [2]},
        )

        self.assertIsNone(configuration)
        self.assertTrue(any('Choose one fragment selector' in item for item in errors))

    def test_dmet_rejects_excited_roots(self):
        errors, configuration = validate_dmet_model_request(
            _ring_spec(),
            {'nroots': 2},
        )

        self.assertIsNone(configuration)
        self.assertTrue(any('ground-state only' in item for item in errors))

    def test_finite_graph_fragments_must_partition_sites(self):
        errors, configuration = validate_dmet_model_request(
            _ring_spec(),
            {'fragments': [{'fragment_id': 'partial', 'site_ids': [0, 1]}]},
        )

        self.assertIsNone(configuration)
        self.assertTrue(any('cover every model site' in item for item in errors))

    def test_registry_links_composite_provider_to_granular_capabilities(self):
        registry = default_registry()

        self.assertEqual(registry.registry_issues(), ())
        self.assertEqual(
            registry.capability('dmet', namespace='model_hamiltonian.solver').status,
            'executable',
        )
        self.assertEqual(
            registry.capability('dmet', namespace='embedding.method').status,
            'executable',
        )
        module = registry.entry('embedding.libdmet.dmet', kind='module')
        self.assertIsNotNone(module)
        self.assertEqual(module.contract.optional_configuration, {})
        implemented = set(module.contract.capability_ids)
        for capability in (
            'embedding.operation.build_dmet_bath',
            'embedding.operation.solve_dmet_impurity',
            'embedding.operation.fit_dmet_correlation_potential',
            'embedding.operation.iterate_dmet_self_consistency',
            'embedding.method.dmet',
            'embedding.impurity_solver.block2_dmrg',
            'model_hamiltonian.solver.dmet',
        ):
            self.assertIn(capability, implemented)
        bindings = [
            entry.contract
            for entry in registry.entries_for(kind='binding')
            if entry.contract.module_id == module.entry_id
        ]
        self.assertEqual(len(bindings), 1)
        self.assertEqual(bindings[0].provider_id, 'provider.libdmet')
        for artifact_kind in (
            'dmet_result',
            'dmet_iteration_history',
            'dmet_numerical_arrays',
            'dmet_output_log',
        ):
            self.assertTrue(artifact_kind_is_registered(artifact_kind, registry))
        dmft = registry.entry('embedding.method.hf_dmft', kind='capability')
        self.assertEqual(
            dmft.contract.metadata['fragment_contract']['inequivalent_fragment_count'],
            1,
        )
        self.assertEqual(
            dmft.contract.metadata['fragment_contract']['impurity_size'],
            'derived from canonical membership',
        )

    def test_dmet_workflow_compiles_provider_before_execution(self):
        registry = default_registry()
        task_spec = {
            'task_type': 'model_hamiltonian',
            'solver': {'name': 'dmet', 'options': {'impurity_shape': [2]}},
            'analysis': {'outputs': ['energy', 'strong_correlation_diagnostics']},
        }
        workflow = compile_task_workflow(task_spec, registry.modules_for(scope='task'))

        self.assertIn('embedding.libdmet.dmet', workflow.execution_order)
        self.assertIn('model.correlation_diagnostics', workflow.execution_order)
        self.assertLess(
            workflow.execution_order.index('embedding.libdmet.dmet'),
            workflow.execution_order.index('core.input_generation'),
        )

    def test_dmet_rejects_duplicate_module_configuration(self):
        registry = default_registry()
        task_spec = {
            'task_type': 'model_hamiltonian',
            'solver': {'name': 'dmet', 'options': {'impurity_solver': 'fci'}},
            'workflow': {
                'module_config': {
                    'embedding.libdmet.dmet': {
                        'impurity_solver': 'block2_dmrg',
                    },
                },
            },
        }

        with self.assertRaises(WorkflowCompilationError) as caught:
            compile_task_workflow(task_spec, registry.modules_for(scope='task'))

        self.assertIn(
            'unsupported_module_configuration',
            [issue.code for issue in caught.exception.issues],
        )

    def test_dmet_result_families_are_written_by_shared_artifact_pipeline(self):
        task_spec = task_spec_from_dict({
            'task_type': 'model_hamiltonian',
            'solver': {'name': 'dmet'},
            'model_hamiltonian': {'spec': _ring_spec()},
        })
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory) / 'dmet-artifact-test'
            provider_log = run_directory / 'solver-dmet' / 'log-libdmet-output.log'
            provider_log.parent.mkdir(parents=True)
            provider_log.write_text('libDMET output\n', encoding='utf-8')
            state = {
                'run_id': 'dmet-artifact-test',
                'work_dir': temporary_directory,
                'retry_count': 0,
                'artifacts': [],
            }
            write_result_artifacts(
                state,
                task_spec,
                {
                    'task_type': 'model_hamiltonian',
                    'energy': -1.0,
                    'converged': True,
                    'raw_scf_output': '',
                    'analysis_text': '',
                    'dmet_result': {
                        'schema': DMET_RESULT_SCHEMA,
                        'energy': -1.0,
                        'iteration_history': {
                            'schema': 'pyscf-agent.dmet-iteration-history.v1',
                            'records': [],
                        },
                    },
                    '_transient_dmet_arrays': {'embedding_basis': np.eye(2)},
                    '_transient_provider_logs': [{
                        'kind': 'dmet_output_log',
                        'path': str(provider_log),
                        'mime_type': 'text/plain; charset=utf-8',
                        'description': 'Complete libDMET provider output',
                    }],
                },
            )
            kinds = {item['kind'] for item in state['artifacts']}
            self.assertIn('dmet_result', kinds)
            self.assertIn('dmet_iteration_history', kinds)
            self.assertIn('dmet_numerical_arrays', kinds)
            self.assertIn('dmet_output_log', kinds)
            provider_log_reference = next(
                item for item in state['artifacts'] if item['kind'] == 'dmet_output_log'
            )
            self.assertGreater(provider_log_reference['size_bytes'], 0)
            self.assertNotIn('sha256', provider_log_reference)
            self.assertTrue(
                (Path(temporary_directory) / 'dmet-artifact-test' / 'result-dmet-arrays.npz').is_file()
            )


@unittest.skipUnless(libdmet_availability()['available'], 'optional libDMET provider is unavailable')
class LibDmetExecutionTests(unittest.TestCase):

    def test_shared_beta_reaches_scf_fitting_and_impurity_in_both_modes(self):
        from libdmet.dmet import Hubbard as dmet
        from libdmet.routine import slater
        from pyscf_agent.providers.libdmet import dmet as adapter

        for mode in ('translational', 'finite_graph'):
            for reference in ('restricted', 'unrestricted'):
                for solver_name, smearing in (('fci', True), ('ccsd', True), ('ccsd', False)):
                    with self.subTest(mode=mode, reference=reference, solver=solver_name), tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        options = {
                            'execution_mode': mode, 'impurity_shape': [2],
                            'reference': reference, 'impurity_solver': solver_name,
                            'max_iterations': 1, 'solver_max_memory_mb': 1000,
                        }
                        # Test both Registry defaults and a non-default value.
                        if reference == 'unrestricted' and solver_name == 'ccsd':
                            options['impurity_solver_options'] = {'beta': 250.0}
                        if not smearing:
                            options.setdefault('impurity_solver_options', {})['smearing'] = False
                        expected = (options.get('impurity_solver_options') or {}).get('beta', 1000.0)
                        if not smearing:
                            expected = math.inf
                        reported_beta = expected if math.isfinite(expected) else None
                        scf_beta = expected if solver_name == 'ccsd' else math.inf
                        native_solver = adapter._solver
                        solvers = []

                        def capture_solver(*args, **kwargs):
                            solver = native_solver(*args, **kwargs)
                            solvers.append(solver)
                            return solver

                        mean_field_name = 'RHartreeFock' if reference == 'restricted' else 'HartreeFock'
                        with mock.patch.object(dmet, mean_field_name, wraps=getattr(dmet, mean_field_name)) as scf_call, mock.patch.object(slater, 'FitVcorFull', wraps=slater.FitVcorFull) as fit_call, mock.patch.object(adapter, '_solver', side_effect=capture_solver):
                            result = run_model_hamiltonian_solver(
                                _ring_spec(), solver_name='dmet', solver_options=options,
                                outputs=['energy'], scratch_directory=scratch,
                            )['dmet_result']
                        self.assertTrue(scf_call.call_args_list)
                        self.assertTrue(fit_call.call_args_list)
                        self.assertTrue(solvers)
                        self.assertTrue(all(call.kwargs['beta'] == scf_beta for call in scf_call.call_args_list))
                        self.assertTrue(all(call.args[4] == expected for call in fit_call.call_args_list))
                        self.assertTrue(all(solver.beta == scf_beta for solver in solvers))
                        self.assertEqual(result['iteration_history']['correlation_potential_fit']['fit_beta'], reported_beta)
                        if solver_name == 'ccsd':
                            self.assertEqual(result['smearing']['lattice_scf_beta'], reported_beta)
                            self.assertEqual(result['smearing']['impurity_scf_beta'], reported_beta)
                            self.assertEqual(result['smearing']['sigma'], 1.0 / expected)
                            calls = [call for detail in result['impurity_solver_details'] for call in detail['scf_calls']]
                            self.assertTrue(calls)
                            self.assertTrue(all(call['beta'] == reported_beta for call in calls))

    def test_native_density_fit_matches_the_lattice_mean_field(self):
        self._assert_density_fit_matches_lattice_mean_field()

    def test_fragment_density_fit_matches_the_lattice_mean_field(self):
        for fragment_mode in ('tied', 'independent'):
            with self.subTest(fragment_mode=fragment_mode):
                self._assert_density_fit_matches_lattice_mean_field(fragment_mode)

    def _assert_density_fit_matches_lattice_mean_field(self, fragment_mode=None):
        from libdmet.dmet import Hubbard as dmet
        from libdmet.routine import slater
        from libdmet.system.hamiltonian import HubbardHamiltonian
        from libdmet.system.lattice import ChainLattice

        native_fit = slater.FitVcorFull
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            for restricted in (True, False):
                for use_hcore in (True, False):
                    with self.subTest(restricted=restricted, use_hcore=use_hcore):
                        # Finite graphs occupy one cell, with three fragments.
                        orbital_count = 6 if fragment_mode else 2
                        lattice = ChainLattice(6, orbital_count)
                        lattice.setHam(
                            HubbardHamiltonian(lattice, 4.0),
                            use_hcore_as_emb_ham=use_hcore,
                        )
                        # A staggered physical field changes the density; a
                        # uniform energy shift would not expose double counting.
                        lattice.fock_lo_k = lattice.hcore_lo_k + np.diag(
                            np.tile([0.7, -0.4], orbital_count // 2)
                        )
                        lattice.fock_lo_R = lattice.k2R(lattice.fock_lo_k)
                        physical_fock_k = lattice.fock_lo_k.copy()
                        physical_fock_r = lattice.fock_lo_R.copy()
                        potential = dmet.VcorLocal(restricted, False, orbital_count)
                        field = np.kron(
                            np.eye(orbital_count // 2), [[0.25, 0.08], [0.08, -0.25]],
                        )
                        potential.assign(np.array([field, field if restricted else -field]))
                        target, _ = dmet.HartreeFock(
                            lattice, potential, 0.5, beta=_LIBDMET_NATIVE_FIT_BETA,
                        )
                        displaced = potential.get().copy() + np.diag(
                            np.tile([0.03, -0.03], orbital_count // 2)
                        )
                        potential.assign(displaced)
                        original_parameters = potential.param.copy()

                        def checked_fit(rho, fit_lattice, basis, value, beta, filling, **kwargs):
                            indices = kwargs.get('imp_idx', list(range(lattice.nimp)))

                            def residual(candidate):
                                density, _ = dmet.HartreeFock(lattice, candidate, filling, beta=beta)
                                delta = density[:, 0] - rho
                                return np.linalg.norm(delta[:, indices][:, :, indices]) / np.sqrt(rho.shape[0])

                            before = residual(value)
                            fitted, error_begin, error_end = native_fit(
                                rho, fit_lattice, basis, value, beta, filling, **kwargs,
                            )
                            self.assertAlmostEqual(error_begin, before, places=10)
                            self.assertAlmostEqual(error_end, residual(fitted), places=10)
                            self.assertLess(error_end, error_begin)
                            return fitted, error_begin, error_end

                        with mock.patch.object(slater, 'FitVcorFull', side_effect=checked_fit) as fit:
                            if fragment_mode:
                                fragments = [
                                    {'fragment_id': str(i // 2), 'orbital_indices': [i, i + 1]}
                                    for i in range(0, orbital_count, 2)
                                ]
                                fit_fragment_local_correlation_potential(
                                    target, lattice, potential, 0.5, fragments,
                                    translation_tied=fragment_mode == 'tied', slater=slater,
                                    beta=_LIBDMET_NATIVE_FIT_BETA, max_iterations=300,
                                )
                            else:
                                _fit_native_lattice_correlation_potential(
                                    target, lattice, potential, 0.5, beta=1000.0, slater=slater, np=np,
                                )
                        self.assertEqual(fit.call_count, 3 if fragment_mode == 'independent' else 1)
                        np.testing.assert_array_equal(potential.param, original_parameters)
                        np.testing.assert_array_equal(lattice.fock_lo_k, physical_fock_k)
                        np.testing.assert_array_equal(lattice.fock_lo_R, physical_fock_r)

    def test_cdw_reference_density_executes_with_zero_auxiliary_vcor(self):
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_model_hamiltonian_solver(
                _ring_spec(6),
                solver_name='dmet',
                outputs=['energy'],
                solver_options={
                    'impurity_shape': [2],
                    'reference': 'unrestricted',
                    'reference_density_guess': 'cdw',
                    'max_iterations': 1,
                    'solver_max_memory_mb': 2000,
                },
                scratch_directory=scratch,
            )

        arrays = result['_transient_dmet_arrays']
        metadata = result['dmet_result']['reference_density_initialization']
        self.assertEqual(metadata['strategy'], 'cdw')
        self.assertEqual(metadata['bias_kind'], 'staggered_charge_density')
        for spin in (0, 1):
            self.assertTrue(np.allclose(np.diag(arrays['reference_density_seed'][spin]), [0.55, 0.45]))
        # The physical baseline is get_veff(seed): a CDW seed shifts the two sublattices.
        baseline = np.diag(arrays['mean_field_baseline_potential'][0])
        self.assertGreater(abs(float(baseline[0] - baseline[1])), 1.0e-6)
        self.assertLess(
            float(np.max(np.abs(arrays['initial_correlation_potential']))),
            1.0e-12,
        )

    def test_interacting_bath_fock_matches_native_local_jk(self):
        from libdmet.routine import pbc_helper, slater
        from libdmet.system.hamiltonian import HubbardHamiltonian
        from libdmet.system.lattice import ChainLattice

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            for restricted in (True, False):
                for length in (2, 6):
                    with self.subTest(restricted=restricted, length=length):
                        lattice = ChainLattice(length, 2)
                        hamiltonian = HubbardHamiltonian(lattice, 4.0)
                        lattice.setHam(hamiltonian, use_hcore_as_emb_ham=True)
                        angles = np.array([0.2, 0.7, 0.7])[:lattice.nkpts]
                        occupied = np.array([np.cos(angles), np.sin(angles)]).T
                        alpha = np.einsum('ki,kj->kij', occupied, occupied)
                        density = np.array([alpha] if restricted else [alpha, np.eye(2) - alpha])
                        original_density = density.copy()
                        total_density = 2.0 * density if restricted else density
                        vj, vk = pbc_helper.get_jk_from_eri_local(hamiltonian.H2, total_density)
                        veff = (vj - 0.5 * vk)[0] if restricted else vj.sum(axis=0)[None] - vk
                        expected_fock = lattice.hcore_lo_k + veff
                        mirror = copy.copy(lattice)

                        _prepare_interacting_bath_mean_field(
                            lattice, hamiltonian, {'rho_k': density},
                            {'reference': 'restricted' if restricted else 'unrestricted'},
                            slater=slater, np=np, mirrors=(mirror,),
                        )

                        np.testing.assert_allclose(lattice.fock_lo_k, expected_fock, atol=1.0e-12)
                        np.testing.assert_allclose(
                            lattice.fock_lo_R, lattice.k2R(expected_fock), atol=1.0e-12,
                        )
                        np.testing.assert_array_equal(density, original_density)
                        np.testing.assert_array_equal(lattice.rdm1_lo_k, total_density)
                        for name in ('fock_lo_k', 'fock_lo_R', 'rdm1_lo_k', 'rdm1_lo_R'):
                            np.testing.assert_array_equal(getattr(mirror, name), getattr(lattice, name))

    @unittest.skipUnless(block2_availability()['available'], 'optional block2 provider is unavailable')
    def test_block2_impurity_solver_matches_fci_and_reuses_mps(self):
        common_options = {
            'impurity_shape': [2],
            'max_iterations': 2,
            'solver_max_memory_mb': 2000,
        }
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            for reference, symmetry in (('restricted', 'su2'), ('unrestricted', 'sz')):
                with self.subTest(reference=reference):
                    fci = run_model_hamiltonian_solver(
                        _ring_spec(4),
                        solver_name='dmet',
                        outputs=['energy'],
                        solver_options={
                            **common_options,
                            'reference': reference,
                            'impurity_solver': 'fci',
                        },
                        scratch_directory=str(Path(scratch) / reference / 'fci'),
                    )
                    dmrg = run_model_hamiltonian_solver(
                        _ring_spec(4),
                        solver_name='dmet',
                        outputs=['energy'],
                        solver_options={
                            **common_options,
                            'reference': reference,
                            'impurity_solver': 'block2_dmrg',
                            'impurity_solver_options': {'preset': 'high_accuracy'},
                        },
                        scratch_directory=str(Path(scratch) / reference / 'block2'),
                    )

                    self.assertLess(abs(dmrg['energy'] - fci['energy']), 1.0e-5)
                    dmet_result = dmrg['dmet_result']
                    self.assertEqual(dmet_result['impurity_solver'], 'block2_dmrg')
                    self.assertEqual(len(dmet_result['impurity_solver_details']), 1)
                    impurity_details = dmet_result['impurity_solver_details'][0]
                    self.assertTrue(impurity_details['converged'])
                    self.assertEqual(impurity_details['configuration']['symmetry'], symmetry)
                    self.assertTrue(impurity_details['configuration']['compute_1rdm'])
                    self.assertTrue(impurity_details['configuration']['compute_2rdm'])
                    self.assertEqual(impurity_details['configuration']['nroots'], 1)
                    self.assertGreaterEqual(impurity_details['call_count'], 2)
                    self.assertTrue(any(
                        call['restart_applied']
                        for call in impurity_details['calls'][1:]
                    ))

    def test_symmetric_honeycomb_finite_graph_matches_translational_dmet(self):
        spec = _periodic_honeycomb_spec(2, 2)
        common_options = {
            'reference': 'unrestricted',
            'max_iterations': 20,
            'energy_tolerance': 1.0e-7,
            'density_tolerance': 1.0e-6,
            # This fixture compares the two execution routes at their shared
            # stationary solution; it is not a strict density-fit benchmark.
            'density_fit_tolerance': 1.0e-1,
            'solver_max_memory_mb': 2000,
        }
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            translated = run_model_hamiltonian_solver(
                spec,
                solver_name='dmet',
                outputs=['energy', 'strong_correlation_diagnostics'],
                solver_options={
                    **common_options,
                    'execution_mode': 'translational',
                },
                scratch_directory=scratch,
            )
            finite_graph = run_model_hamiltonian_solver(
                spec,
                solver_name='dmet',
                outputs=['energy', 'strong_correlation_diagnostics'],
                solver_options={
                    **common_options,
                    'execution_mode': 'finite_graph',
                },
                scratch_directory=scratch,
            )

        translated_dmet = translated['dmet_result']
        finite_dmet = finite_graph['dmet_result']
        self.assertTrue(translated_dmet['converged'])
        self.assertTrue(finite_dmet['converged'])
        self.assertAlmostEqual(
            finite_dmet['energy_per_site'],
            translated_dmet['energy_per_site'],
            places=7,
        )
        self.assertEqual(
            finite_dmet['iteration_history']['correlation_potential_fit']['parameterization'],
            'translation_tied_fragment_blocks',
        )
        self.assertAlmostEqual(
            finite_dmet['iteration_history']['records'][-1]['density_fit_error'],
            translated_dmet['iteration_history']['records'][-1]['density_fit_error'],
            places=10,
        )

    def test_provider_restores_pyscf_logger_globals(self):
        from pyscf.lib import logger as pyscf_logger

        original_flush = pyscf_logger.flush
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            run_model_hamiltonian_solver(
                _ring_spec(),
                solver_name='dmet',
                outputs=['energy'],
                solver_options={
                    'impurity_shape': [2],
                    'reference': 'restricted',
                    'max_iterations': 1,
                    'solver_max_memory_mb': 2000,
                },
                scratch_directory=scratch,
            )

        import libdmet.utils.logger as provider_logger

        self.assertIs(pyscf_logger.flush, original_flush)
        self.assertFalse(getattr(provider_logger.stdout, 'closed', False))

    def test_impurity_diis_switch_preserves_fci_energy_in_both_execution_modes(self):
        from libdmet.solver.scf import SCF

        original_hf = SCF.HF
        for mode in ('finite_graph', 'translational'):
            energies = []
            for enabled in (True, False):
                calls = []

                def observed_hf(instance, *args, **kwargs):
                    calls.append(kwargs.get('do_diis', True))
                    return original_hf(instance, *args, **kwargs)

                with self.subTest(mode=mode, enabled=enabled), tempfile.TemporaryDirectory() as scratch:
                    with mock.patch.object(SCF, 'HF', observed_hf):
                        result = run_model_hamiltonian_solver(
                            _ring_spec(), solver_name='dmet', outputs=['energy'],
                            solver_options={
                                'execution_mode': mode, 'impurity_shape': [2],
                                'reference': 'unrestricted', 'reference_density_guess': 'af',
                                'max_iterations': 1, 'impurity_scf_diis': enabled,
                                'solver_max_memory_mb': 2000,
                            }, scratch_directory=scratch,
                        )
                    self.assertTrue(calls)
                    self.assertTrue(all(value is enabled for value in calls))
                    energies.append(result['energy'])
            self.assertAlmostEqual(energies[0], energies[1], places=7)

    def test_model_solver_emits_structured_dmet_result(self):
        # Full-suite discovery can leave unrelated finalized temporary streams
        # awaiting collection. Flush them before asserting the provider's own
        # stdout/stderr boundary.
        gc.collect()
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            result = run_model_hamiltonian_solver(
                _ring_spec(),
                solver_name='dmet',
                outputs=['energy'],
                solver_options={
                    'impurity_shape': [2],
                    'reference': 'restricted',
                    'max_iterations': 1,
                    'solver_max_memory_mb': 2000,
                },
                scratch_directory=scratch,
            )

            provider_log = Path(scratch) / 'log-libdmet-output.log'
            self.assertTrue(provider_log.is_file())
            self.assertGreater(provider_log.stat().st_size, 0)

        self.assertEqual(result['solver'], 'dmet')
        self.assertEqual(captured_stdout.getvalue(), '')
        self.assertEqual(captured_stderr.getvalue(), '')
        self.assertEqual(result['dmet_result']['provider_log']['provider'], 'libdmet')
        self.assertEqual(result['_transient_provider_logs'][0]['kind'], 'dmet_output_log')
        self.assertEqual(result['dmet_result']['schema'], DMET_RESULT_SCHEMA)
        self.assertIsNone(result['reference_converged'])
        self.assertAlmostEqual(result['energy'], result['energy_per_site'] * 6, places=9)
        self.assertNotIn('reference_energy', result)
        self.assertNotIn('correlation_energy', result)
        self.assertNotIn('reference_energy', result['dmet_result'])
        self.assertNotIn('correlation_energy', result['dmet_result'])
        representative = result['dmet_result']['fragments'][0]
        self.assertAlmostEqual(
            representative['energy'],
            result['energy_per_site'] * representative['size'],
            places=9,
        )
        self.assertEqual(result['dmet_result']['iteration_history']['iteration_count'], 1)
        self.assertEqual(result['quality_checks'], result['dmet_result']['quality_checks'])
        quality_check_ids = {
            item['id'] for item in result['dmet_result']['quality_checks']
        }
        self.assertTrue(all(
            'status' not in item for item in result['dmet_result']['quality_checks']
        ))
        self.assertIn('dmet_energy_change', quality_check_ids)
        self.assertIn('dmet_density_fit_error', quality_check_ids)
        self.assertIn('dmet_projected_sector', quality_check_ids)
        arrays = result['_transient_dmet_arrays']
        self.assertIn('embedding_basis', arrays)
        self.assertIn('impurity_h1_cd', arrays)
        self.assertIn('impurity_h2_ccdd', arrays)

    def test_unrestricted_translated_dmet_preserves_alpha_beta_density(self):
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_model_hamiltonian_solver(
                _ring_spec(6),
                solver_name='dmet',
                outputs=['energy', 'strong_correlation_diagnostics'],
                solver_options={
                    'impurity_shape': [2],
                    'reference': 'unrestricted',
                    'max_iterations': 1,
                    'solver_max_memory_mb': 2000,
                },
                scratch_directory=scratch,
            )

        dmet_result = result['dmet_result']
        self.assertEqual(dmet_result['spin_reference']['mode'], 'unrestricted_fixed_sz')
        self.assertEqual(dmet_result['spin_reference']['initial_density_strategy'], 'pm')
        spin_population = dmet_result['fragments'][0]['spin_population']
        self.assertTrue(spin_population['spin_resolved'])
        local_observables = result['dmet_local_observables']
        self.assertEqual(local_observables['status'], 'available')
        self.assertEqual(len(local_observables['density']), 6)
        self.assertTrue(all(
            abs(value - 1.0) < 1.0e-7
            for value in local_observables['density']
        ))
        self.assertEqual(local_observables['coverage']['rdm2_fragment_count'], 1)
        self.assertGreater(local_observables['coverage']['double_occupancy_fraction'], 0.0)
        self.assertLessEqual(local_observables['coverage']['double_occupancy_fraction'], 1.0)
        self.assertTrue(math.isfinite(local_observables['mean_double_occupancy']))
        self.assertTrue(math.isfinite(local_observables['mean_nearest_neighbor_one_body_coherence']))
        natural_occupations = result['natural_occupation_summary']['occupations']
        self.assertEqual(len(natural_occupations), 6)
        self.assertAlmostEqual(sum(natural_occupations), 6.0, places=7)
        recommendation = result['strong_correlation_diagnostics']['method_recommendation']
        self.assertEqual(recommendation['preferred'], ['dmet'])
        self.assertNotIn('fci', recommendation['risky'])
        self.assertAlmostEqual(
            dmet_result['fragment_electron_count'],
            spin_population['total_electron_count'],
            places=7,
        )
        self.assertAlmostEqual(
            dmet_result['fragment_electron_count'],
            2.0,
            places=7,
        )
        self.assertAlmostEqual(
            dmet_result['iteration_history']['records'][0][
                'fragment_electron_count'
            ],
            2.0,
            places=7,
        )
        self.assertAlmostEqual(
            spin_population['alpha_electron_count'],
            spin_population['beta_electron_count'],
            places=7,
        )
        self.assertLess(spin_population['maximum_absolute_local_magnetization'], 1.0e-8)
        self.assertEqual(
            result['_transient_dmet_arrays']['fragment_density_matrix'].shape[0],
            2,
        )
        self.assertLess(
            float(np.max(np.abs(
                result['_transient_dmet_arrays']['initial_correlation_potential']
            ))),
            1.0e-12,
        )
        self.assertGreater(
            float(np.max(np.abs(
                result['_transient_dmet_arrays']['mean_field_baseline_potential']
            ))),
            1.0e-8,
        )
        self.assertEqual(
            dmet_result['initial_correlation_potential']['mean_field_baseline'],
            'reference_density_seed_hartree_fock',
        )
        self.assertEqual(
            dmet_result['reference_density_initialization']['scope'],
            'initial_uhf_density_only',
        )
        self.assertIn('reference_density_seed', result['_transient_dmet_arrays'])
        self.assertNotIn('reference_density_matrix', result['_transient_dmet_arrays'])

    def test_af_reference_density_executes_with_zero_auxiliary_vcor(self):
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_model_hamiltonian_solver(
                _ring_spec(6),
                solver_name='dmet',
                outputs=['energy'],
                solver_options={
                    'impurity_shape': [2],
                    'reference': 'unrestricted',
                    'reference_density_guess': 'af',
                    'max_iterations': 1,
                    'solver_max_memory_mb': 2000,
                },
                scratch_directory=scratch,
            )

        arrays = result['_transient_dmet_arrays']
        metadata = result['dmet_result']['reference_density_initialization']
        self.assertEqual(metadata['strategy'], 'af')
        self.assertAlmostEqual(metadata['applied_bias'], 0.05)
        self.assertTrue(np.allclose(
            np.diag(arrays['reference_density_seed'][0]),
            [0.55, 0.45],
        ))
        self.assertTrue(np.allclose(
            np.diag(arrays['reference_density_seed'][1]),
            [0.45, 0.55],
        ))
        self.assertLess(
            float(np.max(np.abs(arrays['initial_correlation_potential']))),
            1.0e-12,
        )

    def test_unrestricted_translated_dmet_executes_nonzero_spin_sector(self):
        spec = _ring_spec(6)
        spec['nelec'] = [4, 2]
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_model_hamiltonian_solver(
                spec,
                solver_name='dmet',
                outputs=['energy'],
                solver_options={
                    'impurity_shape': [2],
                    'reference': 'unrestricted',
                    'max_iterations': 1,
                    'solver_max_memory_mb': 2000,
                },
                scratch_directory=scratch,
            )

        dmet_result = result['dmet_result']
        self.assertEqual(dmet_result['spin_reference']['mode'], 'unrestricted_fixed_sz')
        self.assertEqual(dmet_result['spin_reference']['initial_density_strategy'], 'pm')
        self.assertEqual(dmet_result['spin_reference']['libdmet_sz'], 2)
        self.assertEqual(dmet_result['spin_reference']['total_sz'], 1.0)
        self.assertEqual(dmet_result['configuration']['spin_filling'], [4.0 / 6.0, 2.0 / 6.0])
        self.assertEqual(dmet_result['chemical_potential'].__class__, list)
        impurity_sector = dmet_result['fragments'][0]['impurity_spin_sector']
        self.assertEqual(impurity_sector['nelec'], 4)
        self.assertEqual(impurity_sector['libdmet_sz'], 2)
        self.assertEqual((impurity_sector['nalpha'], impurity_sector['nbeta']), (3, 1))
        self.assertTrue(math.isfinite(dmet_result['energy']))

    def test_square_lattice_executes(self):
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_model_hamiltonian_solver(
                _square_spec(4, 4),
                solver_name='dmet',
                outputs=['energy'],
                solver_options={
                    'impurity_shape': [2, 2],
                    'reference': 'restricted',
                    'max_iterations': 1,
                    'solver_max_memory_mb': 2000,
                },
                scratch_directory=scratch,
            )

        self.assertEqual(result['dmet_result']['configuration']['lattice_shape'], [4, 4])
        self.assertEqual(result['dmet_result']['configuration']['impurity_shape'], [2, 2])
        self.assertEqual(result['dmet_result']['configuration']['reference'], 'restricted')
        self.assertTrue(math.isfinite(result['energy_per_site']))

    def test_open_shell_noninteracting_square_uses_fractional_occupation_bath(self):
        spec = _square_spec(4, 4)
        for site in spec['sites']:
            site['U'] = 0.0

        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_model_hamiltonian_solver(
                spec,
                solver_name='dmet',
                outputs=['energy'],
                solver_options={
                    'impurity_shape': [2, 2],
                    'reference': 'restricted',
                    'max_iterations': 10,
                    'energy_tolerance': 1.0e-8,
                    'density_tolerance': 1.0e-6,
                    'density_fit_tolerance': 1.0e-3,
                    'solver_max_memory_mb': 4000,
                },
                scratch_directory=scratch,
            )

        dmet_result = result['dmet_result']
        first_iteration = dmet_result['iteration_history']['records'][0]
        self.assertEqual(first_iteration['bath_basis']['kind'], 'eig')
        self.assertEqual(
            first_iteration['bath_basis']['fractional_occupation_count'],
            6,
        )
        self.assertEqual(first_iteration['embedding_orbital_count'], 12)
        self.assertIn('eig', dmet_result['bath']['basis_kinds_used'])
        self.assertTrue(dmet_result['converged'])
        self.assertAlmostEqual(result['energy_per_site'], -1.5, places=3)
        quality = evaluate_quality_checks(result['quality_checks'])
        self.assertEqual(quality['quality_status'], 'passed')
        self.assertTrue(quality['publication_eligible'])

    def test_atomic_limit_uses_projected_half_filled_impurity_sector(self):
        spec = _square_spec(4, 4)
        for site in spec['sites']:
            site['sublattice'] = (
                'a'
                if (int(site['x']) + int(site['y'])) % 2 == 0
                else 'b'
            )
        for bond in spec['bonds']:
            bond['t'] = 0.0
            bond['effective_t'] = 0.0

        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_model_hamiltonian_solver(
                spec,
                solver_name='dmet',
                outputs=['energy', 'strong_correlation_diagnostics'],
                solver_options={
                    'impurity_shape': [2, 2],
                    'reference': 'unrestricted',
                    'reference_density_guess': 'af',
                    'max_iterations': 5,
                    'energy_tolerance': 1.0e-8,
                    'density_tolerance': 1.0e-6,
                    'density_fit_tolerance': 1.0e-6,
                    'solver_max_memory_mb': 2000,
                },
                scratch_directory=scratch,
            )

        dmet_result = result['dmet_result']
        impurity_sector = dmet_result['fragments'][0]['impurity_spin_sector']
        selection = impurity_sector['electron_count_selection']
        self.assertTrue(dmet_result['converged'])
        self.assertAlmostEqual(result['energy'], 0.0, places=10)
        self.assertEqual(dmet_result['bath']['embedding_orbital_count'], 4)
        self.assertEqual(dmet_result['fragment_electron_count'], 4.0)
        self.assertEqual(impurity_sector['nelec'], 4)
        quality = evaluate_quality_checks(result['quality_checks'])
        self.assertEqual(quality['quality_status'], 'passed')
        self.assertTrue(quality['publication_eligible'])
        self.assertEqual((impurity_sector['nalpha'], impurity_sector['nbeta']), (2, 2))
        self.assertEqual(selection['projected_density_traces'], [2.0, 2.0])
        self.assertEqual(
            selection['selection'],
            'projected_mean_field_density_trace',
        )
        self.assertAlmostEqual(
            result['dmet_local_observables']['mean_double_occupancy'],
            0.0,
            places=10,
        )

    def test_builder_honeycomb_translated_representative_executes(self):
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_model_hamiltonian_solver(
                _periodic_honeycomb_spec(),
                solver_name='dmet',
                outputs=['energy'],
                solver_options={
                    'reference': 'restricted',
                    'max_iterations': 1,
                    'solver_max_memory_mb': 2000,
                },
                scratch_directory=scratch,
            )

        dmet_result = result['dmet_result']
        configuration = dmet_result['configuration']
        self.assertEqual(configuration['execution_mode'], 'translational')
        self.assertEqual(configuration['translation_backend'], 'builder_primitive_cell')
        self.assertEqual(dmet_result['fragment_count'], 1)
        self.assertEqual(dmet_result['fragment_site_count'], 2)
        self.assertEqual(dmet_result['fragments'][0]['site_ids'], [0, 1])
        self.assertTrue(dmet_result['fragments'][0]['translation_equivalent'])
        self.assertTrue(dmet_result['bath']['interacting_bath'])
        self.assertAlmostEqual(
            dmet_result['iteration_history']['records'][0]['mean_field_energy_per_site'] * 2.0,
            -1.17705098312484,
            places=10,
        )
        self.assertTrue(math.isfinite(result['energy_per_site']))

    def test_builder_honeycomb_allows_explicit_noninteracting_bath(self):
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_model_hamiltonian_solver(
                _periodic_honeycomb_spec(),
                solver_name='dmet',
                outputs=['energy'],
                solver_options={
                    'reference': 'restricted',
                    'interacting_bath': False,
                    'max_iterations': 1,
                    'solver_max_memory_mb': 2000,
                },
                scratch_directory=scratch,
            )

        self.assertFalse(result['dmet_result']['bath']['interacting_bath'])
        self.assertFalse(result['dmet_result']['configuration']['interacting_bath'])
        self.assertTrue(math.isfinite(result['energy_per_site']))

    def test_open_nonuniform_builder_graph_executes_with_explicit_fragments(self):
        spec = _ring_spec(4)
        spec['preset'] = 'chain'
        spec['boundary'] = 'open'
        spec['bonds'] = spec['bonds'][:-1]
        spec['sites'][0]['U'] = 3.5
        spec['sites'][1]['epsilon'] = 0.1
        spec['bonds'][1]['t'] = -0.75
        spec['bonds'][1]['effective_t'] = -0.75
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_model_hamiltonian_solver(
                spec,
                solver_name='dmet',
                outputs=['energy'],
                solver_options={
                    'fragments': [
                        {'fragment_id': 'left', 'site_ids': [0, 1]},
                        {'fragment_id': 'right', 'site_ids': [2, 3]},
                    ],
                    'reference': 'unrestricted',
                    'max_iterations': 1,
                    'solver_max_memory_mb': 2000,
                },
                scratch_directory=scratch,
            )

        dmet_result = result['dmet_result']
        self.assertEqual(dmet_result['configuration']['execution_mode'], 'finite_graph')
        self.assertEqual(dmet_result['energy_normalization'], 'finite_graph_fragment_partition')
        self.assertEqual(dmet_result['fragment_count'], 2)
        self.assertEqual([item['site_ids'] for item in dmet_result['fragments']], [[0, 1], [2, 3]])
        self.assertTrue(all(item['spin_population']['spin_resolved'] for item in dmet_result['fragments']))
        self.assertTrue(math.isfinite(dmet_result['energy']))

    def test_periodic_local_perturbation_executes_as_a_finite_graph_partition(self):
        spec = _ring_spec(4)
        spec['sites'][0]['epsilon'] = 0.2
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_model_hamiltonian_solver(
                spec,
                solver_name='dmet',
                outputs=['energy', 'strong_correlation_diagnostics'],
                solver_options={
                    'reference': 'unrestricted',
                    'max_iterations': 1,
                    'solver_max_memory_mb': 2000,
                },
                scratch_directory=scratch,
            )

        dmet_result = result['dmet_result']
        configuration = dmet_result['configuration']
        self.assertEqual(configuration['execution_mode'], 'finite_graph')
        self.assertIn(
            'uniform_onsite_energy',
            configuration['translation_symmetry']['failed_checks'],
        )
        self.assertEqual(dmet_result['fragment_count'], 4)
        self.assertEqual(
            sorted(
                site_id
                for fragment in dmet_result['fragments']
                for site_id in fragment['site_ids']
            ),
            [0, 1, 2, 3],
        )
        self.assertTrue(math.isfinite(dmet_result['energy']))
        local_observables = result['dmet_local_observables']
        self.assertEqual(local_observables['status'], 'available')
        self.assertEqual(
            local_observables['coverage']['double_occupancy_site_count'],
            4,
        )
        self.assertAlmostEqual(
            local_observables['coverage']['double_occupancy_fraction'],
            1.0,
            places=12,
        )
        recommendation = result['strong_correlation_diagnostics']['method_recommendation']
        self.assertEqual(recommendation['preferred'], ['dmet'])

    def test_single_site_fragments_execute_for_a_periodic_triangular_graph(self):
        spec = _square_spec(3, 2)
        spec['preset'] = 'triangular'
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_model_hamiltonian_solver(
                spec,
                solver_name='dmet',
                outputs=['energy'],
                solver_options={
                    'impurity_size': 1,
                    'reference': 'unrestricted',
                    'max_iterations': 2,
                    'solver_max_memory_mb': 2000,
                },
                scratch_directory=scratch,
            )

        dmet_result = result['dmet_result']
        self.assertEqual(dmet_result['fragment_count'], 6)
        self.assertEqual(dmet_result['configuration']['fragment_site_counts'], [1] * 6)
        self.assertAlmostEqual(
            dmet_result['fragment_electron_count'],
            sum(
                fragment['spin_population']['total_electron_count']
                for fragment in dmet_result['fragments']
            ),
            places=7,
        )
        for fragment in dmet_result['fragments']:
            self.assertAlmostEqual(
                fragment['electron_count'],
                fragment['spin_population']['total_electron_count'],
                places=7,
            )
        self.assertTrue(all(
            count <= 2 for count in dmet_result['bath']['embedding_orbital_counts']
        ))
        self.assertTrue(all(
            shapes['two_body_component_shapes']['ccdd'][-4:] == [2, 2, 2, 2]
            for shapes in dmet_result['impurity_hamiltonians']
        ))
        records = dmet_result['iteration_history']['records']
        first_occupancy_error = sum(
            abs(value - 1.0) for value in records[0]['fragment_electron_counts']
        )
        second_occupancy_error = sum(
            abs(value - 1.0) for value in records[1]['fragment_electron_counts']
        )
        self.assertLessEqual(second_occupancy_error, first_occupancy_error + 1.0e-12)
        correlation_potential = result['_transient_dmet_arrays']['correlation_potential']
        off_diagonal = correlation_potential.copy()
        diagonal = np.arange(off_diagonal.shape[-1])
        off_diagonal[:, diagonal, diagonal] = 0.0
        self.assertLess(float(np.max(np.abs(off_diagonal))), 1.0e-12)
        self.assertTrue(math.isfinite(dmet_result['energy']))

    def test_multi_site_builder_cells_execute_as_complete_fragments(self):
        spec = _multi_site_cell_spec(2)
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_model_hamiltonian_solver(
                spec,
                solver_name='dmet',
                outputs=['energy'],
                solver_options={
                    'reference': 'unrestricted',
                    'max_iterations': 1,
                    'solver_max_memory_mb': 2000,
                },
                scratch_directory=scratch,
            )

        dmet_result = result['dmet_result']
        self.assertEqual(dmet_result['configuration']['primitive_cell_basis_size'], 2)
        self.assertEqual(dmet_result['configuration']['fragment_site_counts'], [2, 2])
        self.assertEqual(
            [fragment['site_ids'] for fragment in dmet_result['fragments']],
            [[0, 2], [1, 3]],
        )
        self.assertTrue(math.isfinite(dmet_result['energy']))

    def test_intersite_interaction_uses_interacting_bath_partition(self):
        spec = _ring_spec(4)
        spec['bonds'][0]['V'] = 0.5
        spec['bonds'][0]['effective_V'] = 0.5
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_model_hamiltonian_solver(
                spec,
                solver_name='dmet',
                outputs=['energy'],
                solver_options={
                    'fragments': [
                        {'fragment_id': 'a', 'site_ids': [0, 1]},
                        {'fragment_id': 'b', 'site_ids': [2, 3]},
                    ],
                    'reference': 'unrestricted',
                    'max_iterations': 1,
                    'solver_max_memory_mb': 2000,
                },
                scratch_directory=scratch,
            )

        self.assertTrue(result['dmet_result']['bath']['interacting_bath'])
        self.assertEqual(result['dmet_result']['spin_reference']['mode'], 'unrestricted_fixed_sz')
        self.assertTrue(math.isfinite(result['energy']))

    def test_degenerate_nonzero_spin_ccsd_does_not_return_accepted_result(self):
        # Finite-beta SCF retains the degeneracy of this reference; native
        # integer-occupation CCSD diverges. Never turn it into an accepted Run.
        spec = _ring_spec()
        spec['nelec'] = [4, 2]
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(ValueError, 'infs or NaNs'):
            run_model_hamiltonian_solver(
                spec,
                solver_name='dmet',
                outputs=['energy'],
                solver_options={
                    'impurity_shape': [2],
                    'impurity_solver': 'ccsd',
                    'reference': 'unrestricted',
                    'max_iterations': 1,
                    'solver_max_memory_mb': 2000,
                },
                scratch_directory=scratch,
            )

    def test_ccsd_dmet_diagnostics_keep_smearing_and_fragment_evidence_separate(self):
        for beta, finite_graph in ((None, False), (1000., False), (1000., True)):
            with self.subTest(beta=beta, finite_graph=finite_graph), tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                options = {
                    'impurity_shape': [2], 'impurity_solver': 'ccsd',
                    'reference': 'unrestricted', 'max_iterations': 1,
                    'solver_max_memory_mb': 2000,
                }
                if beta is not None:
                    options['impurity_solver_options'] = {'beta': beta}
                if finite_graph:
                    options.pop('impurity_shape')
                    options['fragments'] = [
                        {'fragment_id': str(index), 'site_ids': [index, index + 1]}
                        for index in (0, 2, 4)
                    ]
                result = run_model_hamiltonian_solver(
                    _ring_spec(), solver_name='dmet', outputs=['energy', 'strong_correlation_diagnostics'],
                    solver_options=options, scratch_directory=scratch,
                )
                diagnostics = result['strong_correlation_diagnostics']
                items = {item['name']: item for item in diagnostics['diagnostics']}
                self.assertIsNone(items['natural_orbital_occupations']['score'])
                self.assertEqual(result['natural_occupation_summary']['scope'], 'assembled_dmet_lattice')
                double = items['double_occupancy_suppression']['value']
                self.assertEqual(double['source'], 'same_fragment_spin_1rdm')
                self.assertEqual(double['scope'], 'covered_fragment_sites')
                self.assertGreater(double['evaluated_site_count'], 0)
                self.assertEqual(double['evaluated_site_count'], 6 if finite_graph else 2)
                self.assertEqual(items['impurity_scf_smearing']['value']['beta'], beta or 1000.0)
                self.assertIsNone(items['impurity_scf_smearing']['score'])

    @unittest.skipUnless(
        os.environ.get('PYSCF_AGENT_RUN_LIBDMET_BENCHMARKS') == '1',
        'set PYSCF_AGENT_RUN_LIBDMET_BENCHMARKS=1 to run the interacting-bath numerical benchmark',
    )
    def test_interacting_bath_one_dimensional_hubbard_benchmark(self):
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_model_hamiltonian_solver(
                _ring_spec(18, onsite_u=4.0),
                solver_name='dmet',
                outputs=['energy'],
                solver_options={
                    'impurity_shape': [2],
                    'reference': 'restricted',
                    'interacting_bath': True,
                    'max_iterations': 50,
                    'solver_tolerance': 1e-11,
                    'solver_max_memory_mb': 4000,
                },
                scratch_directory=scratch,
            )

        self.assertTrue(result['converged'])
        self.assertAlmostEqual(result['energy_per_site'], -0.586516085615, places=4)


if __name__ == '__main__':
    unittest.main()
