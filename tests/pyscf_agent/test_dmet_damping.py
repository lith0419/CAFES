"""Outer DMET damping, DIIS ordering, and option validation."""
import unittest
from unittest import mock

import numpy as np

from pyscf_agent.providers.libdmet import normalize_dmet_options
from pyscf_agent.providers.libdmet.dmet import _mix_correlation_potential, _native_dmet_converged


class DmetDampingTests(unittest.TestCase):
    def test_defaults_preserve_existing_update(self):
        options = normalize_dmet_options({})
        self.assertEqual(options['correlation_potential_mixing'], 1.0)
        self.assertTrue(options['diis_enabled'])
        extrapolated = np.array([2., -4.])
        diis = mock.Mock()
        diis.update.return_value = extrapolated
        result, history = _mix_correlation_potential(
            [1., 1.], [3., -2.], options, options['diis_start'], diis, np,
        )
        np.testing.assert_array_equal(result, extrapolated)
        self.assertTrue(history['correlation_potential_diis_applied'])

    def test_option_bounds_and_round_trip(self):
        for mixing in (0.1, 0.2, 1.0):
            options = normalize_dmet_options({'correlation_potential_mixing': mixing, 'diis_enabled': False})
            self.assertEqual(normalize_dmet_options(options), options)
            self.assertFalse(options['diis_enabled'])
        for invalid in (0, -0.1, 1.01, float('nan'), float('inf'), True, 'bad'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_dmet_options({'correlation_potential_mixing': invalid})
        with self.assertRaises(ValueError):
            normalize_dmet_options({'diis_enabled': 'bad'})

    def test_damping_stabilizes_a_two_cycle_without_changing_fixed_point(self):
        # F(x)=2-x has fixed point 1 and a persistent undamped two-cycle.
        for mixing, expected_error in ((1., 1.), (0.2, 0.6**30)):
            options = normalize_dmet_options({'correlation_potential_mixing': mixing, 'diis_enabled': False})
            x = np.zeros(2)
            diis = mock.Mock()
            for iteration in range(30):
                x, _ = _mix_correlation_potential(x, 2-x, options, iteration, diis, np)
            np.testing.assert_allclose(abs(x-1), expected_error, atol=1e-14)
            diis.update.assert_not_called()

    def test_damping_is_applied_after_diis_and_reports_raw_step(self):
        options = normalize_dmet_options({'correlation_potential_mixing': 0.2})
        previous = np.array([1., -1.])
        fitted = np.array([5., 3.])
        diis = mock.Mock()
        diis.update.return_value = np.array([11., 9.])
        result, history = _mix_correlation_potential(previous, fitted, options, 20, diis, np)
        np.testing.assert_allclose(result, [3., 1.])
        np.testing.assert_array_equal(previous, [1., -1.])
        np.testing.assert_array_equal(fitted, [5., 3.])
        np.testing.assert_array_equal(diis.update.call_args.args[0], fitted)
        self.assertAlmostEqual(history['correlation_potential_fit_change_per_parameter'], np.sqrt(8))
        self.assertAlmostEqual(history['correlation_potential_change_per_parameter'],
                               0.2 * history['correlation_potential_proposal_change_per_parameter'])

    def test_before_diis_start_uses_damped_fitted_potential(self):
        options = normalize_dmet_options({'correlation_potential_mixing': 0.2})
        diis = mock.Mock()
        result, history = _mix_correlation_potential([1., -1.], [5., 3.], options, 0, diis, np)
        np.testing.assert_allclose(result, [1.8, -0.2])
        self.assertFalse(history['correlation_potential_diis_applied'])
        diis.update.assert_not_called()

    def test_impurity_diis_remains_independent(self):
        options = normalize_dmet_options({'diis_enabled': False, 'impurity_scf_diis': True})
        self.assertFalse(options['diis_enabled'])
        self.assertTrue(options['impurity_scf_diis'])

    def test_small_mixing_does_not_relax_convergence_thresholds(self):
        options = normalize_dmet_options({'correlation_potential_mixing': 0.01})
        for energy, density in ((1e-2, 0), (0, 1e-2)):
            self.assertFalse(_native_dmet_converged(energy, density, 0, options))
        self.assertTrue(_native_dmet_converged(1e-9, 1e-7, 1e-7, options))
        # The density-fit residual is a diagnostic under the native stopping rule.
        self.assertTrue(_native_dmet_converged(1e-9, 1e-7, 1e-2, options))


if __name__ == '__main__':
    unittest.main()
