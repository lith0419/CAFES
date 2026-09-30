"""Scientific invariants of the frozen-external-space benchmark preparation."""
import unittest

import numpy as np
from pyscf import gto, scf, mcscf

from tools.prepare_casscf_virtual_windows import prepare_virtual_windows, HARTREE_TO_EV


class VirtualWindowTests(unittest.TestCase):
    def test_external_rotation_preserves_cas_energy_density_and_frozen_space(self):
        mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='6-31g', verbose=0)
        mf = scf.RHF(mol).density_fit(auxbasis='def2-svp-jkfit').run(conv_tol=1e-12)
        mo = mf.mo_coeff.copy()
        # Noncanonical input verifies that we select external eigenvectors,
        # rather than reusing the arbitrary original external column labels.
        rotation, _ = np.linalg.qr(np.random.default_rng(17).normal(size=(8, 8)))
        mo[:, 3:] = mo[:, 3:] @ rotation
        cas = mcscf.CASCI(mf, 2, 2).density_fit()
        cas.kernel(mo)
        dm1 = cas.fcisolver.make_rdm1(cas.ci, 2, (1, 1))
        before_energy = cas.e_tot
        _, initial = prepare_virtual_windows(mf, mo, dm1, 2, 2, [0], [3, 5])
        offsets = (np.array(initial['external_fock_eigenvalues_hartree']) - initial['active_upper_hartree']) * HARTREE_TO_EV
        widths = [(offsets[i] + offsets[i + 1]) / 2 for i in (1, 5)]
        self.assertTrue(all(w > 0 for w in widths))
        prepared, summary = prepare_virtual_windows(mf, mo, dm1, 2, 2, [0], widths)
        np.testing.assert_array_equal(prepared[:, :3], mo[:, :3])
        np.testing.assert_allclose(prepared.T @ mf.get_ovlp() @ prepared, np.eye(11), atol=1e-10)
        cas.kernel(prepared)
        self.assertAlmostEqual(cas.e_tot, before_energy, places=10)
        self.assertEqual([c['external_kept'] for c in summary['cases']], [8, 2, 6])
        narrow, wide = summary['cases'][1:]
        self.assertTrue(set(wide['frozen_orbital_indices']) < set(narrow['frozen_orbital_indices']))
        _, no_core = prepare_virtual_windows(mf, mo, dm1, 2, 2, [], widths)
        self.assertEqual(no_core['cases'][0]['frozen_orbital_indices'], [])
        optimizer = mcscf.CASSCF(mf, 2, 2, frozen=narrow['frozen_orbital_indices']).density_fit()
        optimizer.kernel(prepared)
        self.assertTrue(optimizer.converged)
        frozen = narrow['frozen_orbital_indices']
        np.testing.assert_allclose(optimizer.mo_coeff[:, frozen], prepared[:, frozen], atol=1e-10)
        with self.assertRaisesRegex(ValueError, 'electron count'):
            prepare_virtual_windows(mf, mo, dm1 / 2, 2, 2, [0], [3])


if __name__ == '__main__':
    unittest.main()
