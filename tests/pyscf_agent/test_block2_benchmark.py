"""Regression checks for the opt-in native/custom DMRG benchmark."""
import unittest

import numpy as np

from tools.benchmark_block2_mpo import _assess_rdms, _native_sweep_details


class Block2BenchmarkTests(unittest.TestCase):
    def setUp(self):
        # One doubly occupied spatial orbital: E = 2h + (11|11).
        self.h1 = np.array([[-1.0]])
        self.g2 = np.array([[[[0.5]]]])
        self.dm1 = np.array([[2.0]])
        self.dm2 = np.array([[[[2.0]]]])

    def assess(self, energy=-1.51, dm2=None, reference=None):
        return _assess_rdms(self.h1, self.g2, 0.0, (1, 1), energy, self.dm1,
                            self.dm2 if dm2 is None else dm2, reference)

    def test_sweep_minimum_gap_is_not_a_finite_m_rdm_failure(self):
        energies, _, checks = self.assess()
        self.assertAlmostEqual(energies['rdm_energy_hartree'], -1.5)
        self.assertAlmostEqual(energies['rdm_minus_sweep_energy_hartree'], 0.01)
        self.assertTrue(all(checks.values()))
        self.assertNotIn('rdm_energy', checks)

    def test_fci_validation_still_requires_energy_and_rdm_agreement(self):
        reference = {'energy': -1.5, 'rdm1': self.dm1, 'rdm2': self.dm2}
        _, _, checks = self.assess(reference=reference)
        self.assertFalse(checks['energy'])
        self.assertFalse(checks['rdm_energy'])
        _, _, checks = self.assess(energy=-1.5, reference=reference)
        self.assertTrue(all(checks.values()))

    def test_bad_rdm_contraction_is_rejected(self):
        _, _, checks = self.assess(dm2=self.dm2 * 0.5)
        self.assertFalse(checks['rdm2_to_rdm1_contraction'])
        self.assertFalse(checks['rdm2_particle_pairs'])

    def test_nonfinite_solver_energy_is_rejected(self):
        _, _, checks = self.assess(energy=float('nan'))
        self.assertFalse(checks['finite_results'])

    def test_native_parser_handles_first_and_later_sweeps_without_counting_rdm(self):
        output = '''
 --> Site = 0- 1 .. E = -1.00 Error = 0.00e+00
Time elapsed = 1.100 | E = -1.01 | DW = 1.00e-03
Time sweep = 1.100 | 5.00 TFLOP/SWP
 <-- Site = 0- 1 .. E = -1.09 Error = 0.00e+00
Time elapsed = 3.300 | E = -1.11 | DE = -1.00e-01 | DW = 2.00e-03
Time sweep = 2.200 | 5.00 TFLOP/SWP
Build NPDM MPO
Time elapsed = 0.500
Time sweep = 0.500 | 247 GFLOP/SWP
'''
        result = _native_sweep_details(output, 1e-8)
        self.assertEqual(result['sweeps_completed'], 2)
        self.assertEqual(result['sweep_seconds'], [1.1, 2.2])
        self.assertIsNone(result['sweep_observations'][0]['energy_change_hartree'])
        self.assertEqual(result['sweep_observations'][1]['energy_change_hartree'], -0.1)
        self.assertEqual(result['final_site_energy_hartree'], -1.09)
        self.assertFalse(result['energy_converged'])

    def test_one_sweep_does_not_establish_energy_convergence(self):
        result = _native_sweep_details(
            'Time elapsed = 1 | E = -1 | DW = 0\nTime sweep = 1\n', 1e-8)
        self.assertIsNone(result['energy_converged'])

    def test_missing_sweep_evidence_is_reported(self):
        with self.assertRaisesRegex(ValueError, 'no DMRG sweep summaries'):
            _native_sweep_details('Build NPDM MPO\nTime sweep = 1\n', 1e-8)
