"""Numerical invariants for occupation-partitioned active-space localization."""
import unittest

import numpy as np
from pyscf import ao2mo, fci, gto, scf

from pyscf_agent.backend.correlation.cas_execution import _active_orbital_provenance
from pyscf_agent.backend.correlation.orbital_localization import localize_orbital_block
from pyscf_agent.contracts import (
    ActiveSpaceSpec, OrbitalProcessingSpec, normalize_orbital_processing,
    task_spec_from_dict, task_spec_to_dict,
)


class OrbitalLocalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom=[('H', (0, 0, i * 1.1)) for i in range(6)], basis='sto-3g', verbose=0)
        cls.mf = scf.RHF(cls.mol).run(conv_tol=1e-12)
        assert cls.mf.converged
        cls.overlap = cls.mf.get_ovlp()

    def test_threshold_contract_and_task_roundtrip(self):
        options = {'localization_method': 'pm', 'localization_scope': 'active_space',
                   'localization_occupation_thresholds': [.1, 1.9], 'orbital_ordering': 'fiedler'}
        spec = task_spec_from_dict({'orbital_processing': options})
        saved = task_spec_to_dict(spec)['orbital_processing']
        self.assertEqual(saved['localization_occupation_thresholds'], [.1, 1.9])
        self.assertEqual(saved['localization_method'], 'pipek_mezey')
        for value in ('1.0', [1., 1.], [1.9, .1], [0.], [2.], [float('nan')]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_orbital_processing({**options, 'localization_occupation_thresholds': value})
        with self.assertRaisesRegex(ValueError, 'requires active_space'):
            normalize_orbital_processing({**options, 'localization_scope': 'analysis'})
        self.assertEqual(normalize_orbital_processing({})['localization_occupation_thresholds'], [])

    def test_rotated_input_recovers_density_subspaces_and_preserves_fci_energy(self):
        coeff = self.mf.mo_coeff
        rotation = np.linalg.qr(np.random.default_rng(42).normal(size=(6, 6)))[0]
        rotated = coeff @ rotation
        density = rotation.T @ np.diag(self.mf.mo_occ) @ rotation
        localized, record = localize_orbital_block(
            self.mol, rotated, method='pipek_mezey', occupation_thresholds=[1.], reference_rdm1=density)
        self.assertEqual([g['count'] for g in record['localization_groups']], [3, 3])
        transform = coeff.T @ self.overlap @ localized
        np.testing.assert_allclose(transform[:3, 3:], 0, atol=1e-10)
        np.testing.assert_allclose(transform[3:, :3], 0, atol=1e-10)
        energies = []
        for basis in (coeff, localized):
            h1 = basis.T @ self.mf.get_hcore() @ basis
            h2 = ao2mo.restore(1, ao2mo.kernel(self.mol, basis), 6)
            energy, _ = fci.direct_spin0.kernel(h1, h2, 6, (3, 3), tol=1e-12)
            energies.append(energy)
        self.assertAlmostEqual(*energies, places=9)
        self.assertLess(record['subspace_max_abs_error'], 1e-10)

    def test_multiple_intervals_preserve_each_density_subspace(self):
        occ = np.array([1.95, 1.8, 1., .2, .1, .02])
        rotation = np.linalg.qr(np.random.default_rng(7).normal(size=(6, 6)))[0]
        localized, record = localize_orbital_block(
            self.mol, self.mf.mo_coeff @ rotation, method='boys',
            occupation_thresholds=[.15, 1.85], reference_rdm1=rotation.T @ np.diag(occ) @ rotation)
        self.assertEqual([g['count'] for g in record['localization_groups']], [2, 3, 1])
        transform = self.mf.mo_coeff.T @ self.overlap @ localized
        bins = np.searchsorted([.15, 1.85], occ, side='left')
        np.testing.assert_allclose(transform[bins[:, None] != bins[None, :]], 0, atol=1e-10)

    def test_empty_intervals_are_kept_without_localizing_empty_matrices(self):
        _, record = localize_orbital_block(
            self.mol, self.mf.mo_coeff, method='pipek_mezey',
            occupation_thresholds=[.5, 1.5], reference_rdm1=np.diag([1.] * 6))
        self.assertEqual([g['count'] for g in record['localization_groups']], [0, 6, 0])
        self.assertEqual(record['localization_groups'][0]['status'], 'empty')

    def test_missing_or_nonphysical_reference_density_is_rejected(self):
        for density in (None, np.eye(5), np.eye(6) * 3, np.triu(np.ones((6, 6)))):
            with self.subTest(density=density), self.assertRaises(ValueError):
                localize_orbital_block(self.mol, self.mf.mo_coeff, method='pipek_mezey',
                                      occupation_thresholds=[1.], reference_rdm1=density)

    def test_cas_localization_preserves_core_external_and_spin_summed_projection(self):
        class Reference:
            def get_ovlp(inner):
                return self.overlap
            def make_rdm1(inner):
                density = self.mf.make_rdm1()
                return np.array([density * .6, density * .4])
        transformed, provenance = _active_orbital_provenance(
            Reference(), self.mol, self.mf.mo_coeff, ncas=4, nelecas=(2, 2),
            active_space=ActiveSpaceSpec(ncas=4, nelecas=(2, 2)),
            orbital_processing=OrbitalProcessingSpec(localization_method='pipek_mezey',
                localization_scope='active_space', localization_occupation_thresholds=[1.]))
        np.testing.assert_array_equal(transformed[:, [0, 5]], self.mf.mo_coeff[:, [0, 5]])
        self.assertEqual([g['count'] for g in provenance['localization_groups']], [2, 2])
        self.assertIn('spin_summed_scf_reference', provenance['localization_density_source'])


if __name__ == '__main__':
    unittest.main()
