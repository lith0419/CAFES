"""Compare reduced integral spaces with native full-space frozen CASSCF."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pyscf import gto, scf, mcscf, lib, mrpt

from pyscf_agent.backend.correlation.casscf_space import run_casscf_kernel


class CASSCFSpaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lib.num_threads(1)
        cls.mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='6-31g', verbose=0)
        cls.mf = scf.RHF(cls.mol).density_fit(auxbasis='def2-universal-jkfit').run(conv_tol=1e-12)
        cls.mo = cls.mf.mo_coeff.copy()
        rotation, _ = np.linalg.qr(np.random.default_rng(19).normal(size=(8, 8)))
        cls.mo[:, 3:] = cls.mo[:, 3:] @ rotation

    def optimizer(self, frozen, fitted=True):
        mc = mcscf.CASSCF(self.mf if fitted else self.mf.undo_df(), 2, 2, frozen=frozen)
        mc.conv_tol = 1e-10
        mc.conv_tol_grad = 1e-6
        mc.max_cycle_macro = 100
        return mc

    def test_noncontiguous_virtuals_match_integrals_gradient_and_hessian(self):
        frozen = [0, 4, 8, 10]
        full = self.optimizer(frozen)
        keep = [i for i in range(11) if i not in frozen[1:]]
        small = self.optimizer([0])
        mo_small = self.mo[:, keep]
        small.mo_coeff = mo_small
        eri_full = full.ao2mo(self.mo)
        eri_small = small.ao2mo(mo_small)
        np.testing.assert_allclose(eri_small.ppaa[:], eri_full.ppaa[:][keep][:, keep], atol=1e-12)
        np.testing.assert_allclose(eri_small.papa[:], eri_full.papa[:][keep][:, :, keep], atol=1e-12)
        _, _, ci = full.casci(self.mo, eris=eri_full)
        dm1, dm2 = full.fcisolver.make_rdm12(ci, 2, (1, 1))
        grad, _, hop, diag = full.gen_g_hop(self.mo, 1, dm1, dm2, eri_full)
        grad_small, _, hop_small, diag_small = small.gen_g_hop(mo_small, 1, dm1, dm2, eri_small)
        np.testing.assert_allclose(grad, grad_small, atol=1e-11)
        np.testing.assert_allclose(diag, diag_small, atol=1e-11)
        vector = np.random.default_rng(5).normal(size=grad.shape)
        np.testing.assert_allclose(hop(vector), hop_small(vector), atol=1e-10)

    def test_energy_density_frozen_columns_and_full_checkpoints(self):
        frozen = [0, 4, 8, 10]
        full = self.optimizer(frozen)
        full.kernel(self.mo)
        for fitted in (True, False):
            with self.subTest(fitted=fitted), tempfile.TemporaryDirectory() as directory:
                reference = full if fitted else self.optimizer(frozen, fitted=False).run(self.mo)
                projected = self.optimizer(frozen, fitted=fitted)
                projected.chkfile = str(Path(directory) / 'projected.chk')
                snapshots = []
                def inspect_checkpoint(env):
                    if Path(projected.chkfile).exists():
                        snapshots.append(lib.chkfile.load(projected.chkfile, 'mcscf/mo_coeff').shape)
                projected.callback = inspect_checkpoint
                result, summary = run_casscf_kernel(projected, self.mo)
                self.assertTrue(projected.converged)
                self.assertAlmostEqual(reference.e_tot, projected.e_tot, places=9)
                self.assertEqual(summary['working_to_full_indices'], [0, 1, 2, 3, 5, 6, 7, 9])
                self.assertEqual(summary['working_nmo'], 8)
                self.assertEqual(projected.frozen, frozen)
                self.assertEqual(result[3].shape, (11, 11))
                np.testing.assert_array_equal(projected.mo_coeff[:, frozen], self.mo[:, frozen])
                np.testing.assert_allclose(projected.make_rdm1(), reference.make_rdm1(), atol=2e-6)
                np.testing.assert_allclose(projected.mo_energy, reference.mo_energy, atol=2e-6)
                np.testing.assert_allclose(projected.mo_coeff.T @ self.mf.get_ovlp() @ projected.mo_coeff, np.eye(11), atol=1e-10)
                saved = lib.chkfile.load(projected.chkfile, 'mcscf')
                self.assertTrue(snapshots)
                self.assertTrue(all(shape == (11, 11) for shape in snapshots))
                np.testing.assert_allclose(saved['mo_coeff'], result[3], atol=1e-12)
                np.testing.assert_allclose(saved['mo_energy'], result[4], atol=1e-12)
                self.assertEqual(saved['mo_occ'].shape, (11,))
                resumed = self.optimizer(frozen).update_from_chk(projected.chkfile)
                self.assertEqual(resumed.mo_coeff.shape, (11, 11))

    def test_state_average_and_no_checkpoint(self):
        frozen = [0, 4, 8, 10]
        full = mcscf.state_average_(self.optimizer(frozen), [0.5, 0.5]).run(self.mo)
        projected = mcscf.state_average_(self.optimizer(frozen), [0.5, 0.5])
        projected.chkfile = None
        run_casscf_kernel(projected, self.mo)
        self.assertTrue(projected.converged)
        np.testing.assert_allclose(projected.e_states, full.e_states, atol=1e-8)
        np.testing.assert_allclose(projected.mo_energy, full.mo_energy, atol=2e-6)

    def test_complete_virtual_space_restored_for_nevpt2(self):
        frozen = [0, 8, 9, 10]
        full = self.optimizer(frozen)
        projected = self.optimizer(frozen)
        for mc in (full, projected):
            mc.conv_tol = 1e-13
            mc.conv_tol_grad = 1e-8
            mc.fcisolver.conv_tol = 1e-13
        full.kernel(self.mf.mo_coeff)
        run_casscf_kernel(projected, self.mf.mo_coeff)
        self.assertEqual(projected.mo_coeff.shape[1], 11)
        # Native NEVPT rejects mc.frozen. The optimization constraint is no
        # longer needed when correlating the resulting fixed orbitals.
        full.frozen = projected.frozen = None
        corr_full = mrpt.NEVPT(full).kernel()
        corr_projected = mrpt.NEVPT(projected).kernel()
        # Flat orbital directions permit tiny differences between converged
        # paths; NEVPT is not stationary in the CASSCF orbital parameters.
        self.assertAlmostEqual(corr_full, corr_projected, places=7)

    def test_symmetry_and_no_external_freezing(self):
        mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='6-31g', symmetry=True, verbose=0)
        mf = scf.RHF(mol).density_fit(auxbasis='def2-universal-jkfit').run(conv_tol=1e-12)
        for frozen in ([0], [0, 8, 10]):
            with self.subTest(frozen=frozen):
                full = mcscf.CASSCF(mf, 2, 2, frozen=frozen).run(mf.mo_coeff)
                small = mcscf.CASSCF(mf, 2, 2, frozen=frozen)
                # Initial arrays loaded from artifacts can lack PySCF tags.
                result, summary = run_casscf_kernel(small, np.asarray(mf.mo_coeff))
                self.assertTrue(small.converged)
                self.assertAlmostEqual(full.e_tot, small.e_tot, places=8)
                np.testing.assert_array_equal(result[3].orbsym, full.mo_coeff.orbsym)
                self.assertEqual(summary['working_nmo'], 11 if len(frozen) == 1 else 9)


if __name__ == '__main__':
    unittest.main()
