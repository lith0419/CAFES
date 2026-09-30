from __future__ import annotations

import contextlib
import io
import math
import tempfile
import unittest
from unittest import mock

import numpy as np

from pyscf_agent.providers.libdmet import libdmet_availability
from pyscf_agent.providers.libdmet.dmet import _solver, normalize_dmet_options


@unittest.skipUnless(libdmet_availability()['available'], 'libDMET is optional')
class CCSDImpuritySCFTests(unittest.TestCase):
    def setUp(self):
        from libdmet.dmet import Hubbard
        from libdmet.system.integral import Integral
        from libdmet.utils import logger

        self.provider = Hubbard
        self.Integral = Integral
        contexts = contextlib.ExitStack()
        self.addCleanup(contexts.close)
        self.temp = contexts.enter_context(tempfile.TemporaryDirectory())
        output = io.StringIO()
        contexts.enter_context(contextlib.redirect_stdout(output))
        contexts.enter_context(contextlib.redirect_stderr(output))
        contexts.enter_context(mock.patch.object(logger, 'stdout', output))

    def make_solver(self, restricted, enabled=True, beta=None):
        configuration = dict(normalize_dmet_options({
            'impurity_solver': 'ccsd', 'impurity_scf_diis': enabled,
            'reference': 'restricted' if restricted else 'unrestricted',
            'solver_max_memory_mb': 1000,
            'impurity_solver_options': {} if beta is None else {'beta': beta},
        }), libdmet_sz=0)
        return _solver(self.provider, configuration, self.temp)

    def hamiltonian(self, restricted):
        h1 = np.array([[0., -1.], [-1., 0.]])
        h2 = np.zeros((2, 2, 2, 2))
        h2[0, 0, 0, 0] = h2[1, 1, 1, 1] = 2.
        return self.Integral(2, restricted, False, 0.,
                             {'cd': np.array([h1] * (1 if restricted else 2))},
                             {'ccdd': np.array([h2] * (1 if restricted else 3))})

    def test_restricted_and_unrestricted_ccsd_match_exact_two_electron_energy(self):
        for restricted in (True, False):
            for enabled in (True, False):
                with self.subTest(restricted=restricted, diis=enabled):
                    solver = self.make_solver(restricted, enabled, beta=1000)
                    guess = np.full((1 if restricted else 2, 2, 2), .5)
                    density, energy = solver.run(self.hamiltonian(restricted), nelec=2,
                                                 dm0=guess, calc_rdm2=True)
                    self.assertAlmostEqual(energy, 1. - math.sqrt(5.), places=7)
                    physical_density = 2 * density[0] if restricted else density.sum(axis=0)
                    self.assertAlmostEqual(np.trace(physical_density), 2., places=9)
                    from pyscf_agent.providers.libdmet.rdm_bridge import impurity_spin_rdm2
                    self.assertTrue(np.all(np.isfinite(impurity_spin_rdm2(solver, 2))))
                    details = solver.summary()
                    self.assertTrue(details['converged'])
                    self.assertEqual(len(details['scf_calls']), 1)
                    self.assertEqual(details['scf_calls'][0]['diis'],
                                     'adiis_cdiis' if enabled else 'disabled')
                    self.assertEqual(details['scf_calls'][0]['beta'], 1000.0)
                    self.assertEqual(solver.scfsolver.mf.sigma, 0.001)
                    self.assertFalse(solver.scfsolver.newton_ah)

    def test_beta_smears_near_degenerate_scf_but_retains_integer_ccsd_sector(self):
        for restricted in (True, False):
            with self.subTest(restricted=restricted):
                solver = self.make_solver(restricted, beta=1000)
                ham = self.hamiltonian(restricted)
                ham.H1['cd'] *= 0.0005
                ham.H2['ccdd'][:] = 0.
                guess = np.full((1 if restricted else 2, 2, 2), .5)
                density, energy = solver.run(ham, nelec=2, dm0=guess, calc_rdm2=True)
                occupations = solver.scfsolver.mf.mo_occ
                self.assertTrue(np.any((occupations > 0.1) & (occupations < 0.9)))
                self.assertEqual(solver.cisolver.nocc, 1 if restricted else (1, 1))
                np.testing.assert_array_equal(solver.cisolver.mo_occ,
                                              [2., 0.] if restricted else [[1., 0.], [1., 0.]])
                physical_density = 2 * density[0] if restricted else density.sum(axis=0)
                self.assertAlmostEqual(np.trace(physical_density), 2., places=9)
                self.assertTrue(np.isfinite(energy))
                self.assertTrue(solver.summary()['converged'])

    def test_exhausted_scf_stops_before_ccsd_even_when_diis_disabled(self):
        for enabled in (True, False):
            with self.subTest(diis=enabled):
                solver = self.make_solver(False, enabled, beta=1000)
                guess = np.array([np.diag([1., 0.]), np.diag([0., 1.])])
                with mock.patch('libdmet.solver.cc.UICCSD') as cc:
                    with self.assertRaisesRegex(RuntimeError, 'CCSD was not started'):
                        solver.run(self.hamiltonian(False), nelec=2, dm0=guess, scf_max_cycle=1)
                    cc.assert_not_called()
                self.assertFalse(solver.summary()['converged'])
                self.assertFalse(solver.scfsolver.history[0]['accepted'])

    def test_each_solve_gets_fresh_diis_history_without_changing_native_classes(self):
        from libdmet.solver import scf
        original = scf.UIHF.DIIS
        solver = self.make_solver(False)
        guess = np.full((2, 2, 2), .5)
        solver.run(self.hamiltonian(False), nelec=2, dm0=guess)
        first_mf = solver.scfsolver.mf
        solver.run(self.hamiltonian(False), nelec=2, dm0=guess)
        self.assertIsNot(first_mf, solver.scfsolver.mf)
        self.assertEqual(len(solver.summary()['scf_calls']), 2)
        self.assertTrue(all(row['beta'] == 1000.0 for row in solver.summary()['scf_calls']))
        self.assertEqual(solver.beta, 1000.0)
        self.assertIs(scf.UIHF.DIIS, original)
        self.assertIs(scf.SCF(newton_ah=False).__class__, scf.SCF)


if __name__ == '__main__':
    unittest.main()
