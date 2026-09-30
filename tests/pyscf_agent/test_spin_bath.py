import contextlib
import io
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
import numpy as np
from pyscf_agent.providers.libdmet.spin_bath import max_spin_eigen_bath
from pyscf_agent.providers.libdmet import libdmet_availability
from pyscf_agent.backend.model_hamiltonian.solver import run_model_hamiltonian_solver
from pyscf_agent.providers.libdmet.dmet import (
    normalize_dmet_options,
    validate_dmet_model_request,
    _construct_finite_graph_impurity_hamiltonian,
)
from tests.pyscf_agent.test_libdmet_dmet import _ring_spec


def lattice(n=6):
    return SimpleNamespace(
        ncells=1,
        nscsites=n,
        imp_idx=[0, 1],
        val_idx=[0, 1],
        expand=lambda rho: rho[:, 0],
    )


class MaxSpinBathTests(unittest.TestCase):
    def test_unequal_channels_preserve_selected_subspaces_and_add_one_mode(self):
        rng = np.random.default_rng(31)
        q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        rho = np.zeros((2, 1, 6, 6))
        for spin, values in enumerate(([0, 0.2, 0.8, 1], [0, 0, 0.4, 1])):
            rho[spin, 0, 2:, 2:] = (q * np.array(values)) @ q.T
        basis, info = max_spin_eigen_bath(lattice(), rho)
        self.assertEqual(info['original_bath_counts'], [2, 1])
        self.assertEqual(basis.shape, (2, 1, 6, 4))
        for spin, columns in enumerate(([1, 2], [2])):
            b = basis[spin, 0, 2:, 2:]
            np.testing.assert_allclose(
                b @ b.T @ q[:, columns], q[:, columns], atol=1e-12
            )
            np.testing.assert_allclose(
                basis[spin, 0].T @ basis[spin, 0], np.eye(4), atol=1e-12
            )
        self.assertEqual(len(info['added_environment_orbitals'][1]['indices']), 1)
        self.assertEqual(info['added_environment_orbitals'][0]['indices'], [])

    def test_equal_empty_and_complex_channels(self):
        for values, expected in (([0, 0.3, 0.7, 1], 2), ([0, 0, 1, 1], 0)):
            rho = np.array([np.diag([0.5, 0.5] + values)] * 2, dtype=complex)[:, None]
            b, info = max_spin_eigen_bath(lattice(), rho)
            self.assertEqual(b.shape[-1], 2 + expected)
            self.assertEqual(info['original_bath_counts'], [expected] * 2)
            self.assertTrue(
                all(not item['indices'] for item in info['added_environment_orbitals'])
            )

    def test_invalid_density_is_not_masked(self):
        rho = np.array([np.diag([0.5, 0.5, -0.01, 0.3, 0.7, 1])] * 2)[:, None]
        with self.assertRaisesRegex(ValueError, 'physical'):
            max_spin_eigen_bath(lattice(), rho)

    def test_explicit_options_and_validation(self):
        self.assertEqual(
            normalize_dmet_options({})['bath_spin_dimension_policy'], 'native'
        )
        config = normalize_dmet_options({'bath_spin_dimension_policy': 'max'})
        self.assertEqual(config['bath_spin_dimension_policy'], 'max')
        self.assertEqual(normalize_dmet_options(config), config)
        for invalid in ('truncate', '', None, True):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, 'bath_spin_dimension_policy'),
            ):
                normalize_dmet_options({'bath_spin_dimension_policy': invalid})

    def test_max_policy_requires_finite_graph_noninteracting_bath(self):
        for mode, interacting in (
            ('finite_graph', False),
            ('finite_graph', True),
            ('translational', False),
        ):
            errors, config = validate_dmet_model_request(
                _ring_spec(),
                {
                    'bath_spin_dimension_policy': 'max',
                    'execution_mode': mode,
                    'interacting_bath': interacting,
                    'impurity_size': 2,
                },
            )
            if mode == 'finite_graph' and not interacting:
                self.assertEqual(errors, [])
                self.assertEqual(config['bath_spin_dimension_policy'], 'max')
            else:
                self.assertIsNone(config)
                self.assertTrue(
                    any('Max spin bath requires' in error for error in errors)
                )

    def test_restricted_and_complex_nontrivial_density(self):
        rng = np.random.default_rng(19)
        q, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
        rho = np.zeros((1, 6, 6), dtype=complex)
        rho[0, 2:, 2:] = (q * [0, 0.2, 0.8, 1]) @ q.conj().T
        original = rho.copy()
        basis, info = max_spin_eigen_bath(lattice(), rho)
        self.assertEqual(info['original_bath_counts'], [2])
        bath = basis[0, 0, 2:, 2:]
        np.testing.assert_allclose(
            bath @ bath.conj().T, q[:, 1:3] @ q[:, 1:3].conj().T, atol=1e-12
        )
        np.testing.assert_allclose(
            basis[0, 0].conj().T @ basis[0, 0], np.eye(4), atol=1e-12
        )
        np.testing.assert_array_equal(rho, original)

    def test_invalid_lattice_and_density(self):
        rho = np.array([np.diag([0.5, 0.5, 0, 0.3, 0.7, 1])] * 2)[:, None]
        for attribute, value in (('ncells', 2), ('val_idx', [0])):
            lat = lattice()
            setattr(lat, attribute, value)
            with self.assertRaisesRegex(ValueError, 'single-cell all-valence'):
                max_spin_eigen_bath(lat, rho)
        for value, message in (
            (float('nan'), 'Nonfinite'),
            (float('inf'), 'Nonfinite'),
            (1.1, 'physical'),
        ):
            bad = rho.copy()
            bad[0, 0, 2, 2] = value
            with self.assertRaisesRegex(ValueError, message):
                max_spin_eigen_bath(lattice(), bad)
        rho[0, 0, 2, 3] = 0.1
        with self.assertRaisesRegex(ValueError, 'Non-Hermitian'):
            max_spin_eigen_bath(lattice(), rho)

    def test_impurity_construction_dispatch_and_diagnostics(self):
        rho = np.array(
            [np.diag([0.5, 0.5, 0, 0.2, 0.8, 1]), np.diag([0.5, 0.5, 0, 0, 0.4, 1])]
        )[:, None]
        lat = lattice()
        lat.nimp = 2
        for policy, kind in (('max', 'eig'), ('native', 'eig'), ('max', 'svd')):
            with self.subTest(policy=policy, kind=kind):
                slater = mock.Mock()
                slater.embHam.return_value = ('ham', 'energy')
                slater.embBasis.return_value = np.zeros((2, 1, 6, 4))
                diagnostics = []
                with mock.patch(
                    'pyscf_agent.providers.libdmet.dmet._finite_graph_fragment_h2'
                ):
                    _, _, basis = _construct_finite_graph_impurity_hamiltonian(
                        lat,
                        rho,
                        None,
                        {
                            'interacting_bath': False,
                            'contains_intersite_v': False,
                            'bath_spin_dimension_policy': policy,
                        },
                        kind,
                        dmet=mock.Mock(),
                        slater=slater,
                        np=np,
                        bath_diagnostics=diagnostics,
                    )
                if policy == 'max' and kind == 'eig':
                    slater.embBasis.assert_not_called()
                    self.assertEqual(diagnostics[0]['original_bath_counts'], [2, 1])
                    self.assertEqual(basis.shape, (2, 1, 6, 4))
                else:
                    slater.embBasis.assert_called_once()
                    self.assertEqual(diagnostics, [{'policy': 'native', 'kind': kind}])


@unittest.skipUnless(libdmet_availability()['available'], 'libDMET unavailable')
class MaxSpinBathNativeTests(unittest.TestCase):
    def test_finite_graph_execution_records_max_policy(self):
        for reference in ('restricted', 'unrestricted'):
            with (
                self.subTest(reference=reference),
                tempfile.TemporaryDirectory() as scratch,
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                result = run_model_hamiltonian_solver(
                    _ring_spec(4, onsite_u=0),
                    solver_name='dmet',
                    outputs=['energy'],
                    solver_options={
                        'execution_mode': 'finite_graph',
                        'impurity_size': 2,
                        'reference': reference,
                        'interacting_bath': False,
                        'bath_spin_dimension_policy': 'max',
                        'max_iterations': 1,
                        'solver_max_memory_mb': 2000,
                    },
                    scratch_directory=scratch,
                )
                dmet_result = result['dmet_result']
                self.assertEqual(dmet_result['bath']['spin_dimension_policy'], 'max')
                self.assertEqual(dmet_result['bath']['final_basis_kind'], 'eig')
                record = dmet_result['iteration_history']['records'][0]
                diagnostics = record['bath_spin_dimension_diagnostics']
                self.assertEqual(
                    len(diagnostics), dmet_result['configuration']['fragment_count']
                )
                for item in diagnostics:
                    self.assertEqual(item['policy'], 'max')
                    self.assertLess(item['orthogonality_max_abs'], 1e-10)
                self.assertAlmostEqual(result['energy_per_site'], -1.0, places=7)

    def test_equal_dimensions_match_native_bath_projector(self):
        from libdmet.routine import slater

        rho = np.array([np.diag([0.5, 0.5, 0, 0.3, 0.7, 1])] * 2)[:, None]
        lat = lattice()
        native = slater._get_emb_basis_eig(lat, rho, orth=True)
        basis, _ = max_spin_eigen_bath(lat, rho)
        for s in range(2):
            np.testing.assert_allclose(
                basis[s, 0] @ basis[s, 0].T, native[s, 0] @ native[s, 0].T, atol=1e-12
            )
