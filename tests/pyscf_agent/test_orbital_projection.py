import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pyscf import gto, lib, scf, mcscf

from pyscf_agent.backend.orbital_projection import project_orbital_blocks
from pyscf_agent.backend.correlation.orbital_projection import project_partitioned_orbitals, project_casscf_active_guess
from pyscf_agent.providers.block2.checkpoint import attach_optimized_orbitals, load_optimized_orbitals


class OrbitalProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lib.num_threads(1)
        cls.source = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='6-31g', verbose=0)
        cls.target = gto.M(atom='Li 0 0 0; H 0 0 1.7', basis='6-31g', verbose=0)
        cls.mf = scf.RHF(cls.source).run()

    def test_identity_preserves_every_column_and_frozen_partition(self):
        mo, info = project_partitioned_orbitals(self.source, self.source, self.mf.mo_coeff,
                                               ncore=1, ncas=2, frozen_indices=[0])
        np.testing.assert_allclose(mo, self.mf.mo_coeff, atol=1e-12)
        self.assertEqual([p['dimension'] for p in info['partitions']], [1, 0, 2, 8])

    def test_cross_geometry_uses_cross_overlap_and_keeps_partition_ranks(self):
        mo, info = project_partitioned_orbitals(self.source, self.target, self.mf.mo_coeff,
                                               ncore=1, ncas=2, frozen_indices=[0])
        overlap = self.target.intor_symmetric('int1e_ovlp')
        np.testing.assert_allclose(mo.T @ overlap @ mo, np.eye(11), atol=1e-12)
        raw_error = np.max(np.abs(self.mf.mo_coeff.T @ overlap @ self.mf.mo_coeff - np.eye(11)))
        self.assertGreater(raw_error, 1e-3)
        projected = scf.addons.project_mo_nr2nr(self.source, self.mf.mo_coeff, self.target)
        np.testing.assert_allclose(mo[:, 0], projected[:, 0] / np.sqrt(projected[:, 0] @ overlap @ projected[:, 0]), atol=1e-12)
        self.assertGreater(info['partitions'][2]['minimum_subspace_overlap'], .98)
        self.assertFalse(info['mps_transferred'])

    def test_rank_loss_bad_source_and_frozen_external_fail_closed(self):
        bad = self.mf.mo_coeff.copy(); bad[:, 2] = bad[:, 1]
        with self.assertRaisesRegex(ValueError, 'not orthonormal'):
            project_partitioned_orbitals(self.source, self.target, bad, ncore=1, ncas=2)
        with self.assertRaisesRegex(ValueError, 'frozen inactive'):
            project_partitioned_orbitals(self.source, self.target, self.mf.mo_coeff,
                                         ncore=1, ncas=2, frozen_indices=[10])
        with self.assertRaisesRegex(ValueError, 'loses rank'):
            project_partitioned_orbitals(self.source, self.target, self.mf.mo_coeff,
                                         ncore=1, ncas=2, rank_tolerance=2.)

    def test_checkpoint_distinguishes_same_basis_mps_and_projected_orbital_guess(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = {'scratch_directory': root, 'files': []}
            attach_optimized_orbitals(manifest, self.mf.mo_coeff, ncore=1, ncas=2,
                                      nelecas=2, reference='restricted', molecule=self.source,
                                      frozen_orbital_indices=[0])
            unchanged, info = load_optimized_orbitals(manifest, self.mf, ncas=2, nelecas=2)
            np.testing.assert_array_equal(unchanged, self.mf.mo_coeff)
            self.assertTrue(info['same_geometry'])
            target_mf = scf.RHF(self.target)
            with self.assertRaisesRegex(ValueError, 'same geometry'):
                load_optimized_orbitals(manifest, target_mf, ncas=2, nelecas=2)
            projected, info = load_optimized_orbitals(manifest, target_mf, ncas=2, nelecas=2,
                                                    allow_geometry_projection=True, frozen_orbital_indices=[0])
            self.assertFalse(info['same_geometry'])
            self.assertIsNotNone(info['orbital_projection'])
            with self.assertRaisesRegex(ValueError, 'frozen-core partition'):
                load_optimized_orbitals(manifest, target_mf, ncas=2, nelecas=2,
                                        allow_geometry_projection=True, frozen_orbital_indices=[])
            legacy = copy.deepcopy(manifest); legacy['orbital_context'].pop('molecule')
            with self.assertRaisesRegex(ValueError, 'lacks source geometry'):
                load_optimized_orbitals(legacy, self.mf, ncas=2, nelecas=2)

    def test_checkpoint_preserves_shared_mount_namespace_across_physical_relocation(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            physical = root / 'node-a-mount'
            physical.mkdir()
            shared = root / 'shared'
            shared.symlink_to(physical, target_is_directory=True)
            manifest = {'scratch_directory': str(shared), 'files': []}
            attach_optimized_orbitals(manifest, self.mf.mo_coeff, ncore=1, ncas=2,
                nelecas=2, reference='restricted', molecule=self.source,
                frozen_orbital_indices=[0])
            saved_path = Path(manifest['orbital_context']['orbital_file']['path'])
            self.assertEqual(saved_path.parent, shared)
            relocated = root / 'node-b-mount'
            physical.rename(relocated)
            shared.unlink()
            shared.symlink_to(relocated, target_is_directory=True)
            self.assertTrue(saved_path.is_file())
            restored, _ = load_optimized_orbitals(manifest, self.mf, ncas=2, nelecas=2)
            np.testing.assert_array_equal(restored, self.mf.mo_coeff)


class GeneralOrbitalBlockProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lib.num_threads(1)
        cls.source = gto.M(atom='O 0 0 0; H 0 -.75 .58; H 0 .75 .58', basis='sto-3g', verbose=0)
        cls.target = gto.M(atom='O 0 0 0; H 0 -.78 .60; H 0 .78 .60', basis='sto-3g', verbose=0)
        cls.mf = scf.RHF(cls.source).run(conv_tol=1e-12)
        cls.s_target = cls.target.intor_symmetric('int1e_ovlp')

    def test_target_scf_core_follows_translated_nucleus_and_keeps_full_virtual_space(self):
        target = gto.M(atom='O .4 0 0; H .4 -.75 .58; H .4 .75 .58', basis='sto-3g', verbose=0)
        mf = scf.RHF(target).run(conv_tol=1e-12)
        mc = mcscf.CASSCF(mf, 2, 2)
        native = mcscf.CASSCF(self.mf, 2, 2, frozen=[0]).run()
        mo, info = project_casscf_active_guess(self.source, mc, native.mo_coeff, frozen_indices=[0])
        np.testing.assert_allclose(mo[:, 0], mf.mo_coeff[:, 0], atol=1e-12)
        np.testing.assert_allclose(mo.T @ mf.get_ovlp() @ mo, np.eye(7), atol=1e-12)
        self.assertEqual(info['external_dimension'], 1)
        self.assertFalse(info['external_truncated'])
        projected = scf.addons.project_mo_nr2nr(self.source, self.mf.mo_coeff[:, :1], target)
        projected /= np.sqrt(projected.T @ mf.get_ovlp() @ projected)
        self.assertGreater(np.max(np.abs(projected[:, 0] - mo[:, 0])), .01)
        # Optimization must retain the target core. Active-space/root following
        # is a separate review: a large translation can change the CAS basin.
        mc.frozen = [0]
        mc.kernel(mo)
        self.assertTrue(native.converged and mc.converged)
        np.testing.assert_allclose(mc.mo_coeff[:, 0], mf.mo_coeff[:, 0], atol=1e-12)

    def test_target_scf_policy_requires_orbital_only_restart_and_records_policy(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = {'scratch_directory': root, 'files': []}
            attach_optimized_orbitals(manifest, self.mf.mo_coeff, ncore=4, ncas=2,
                nelecas=2, reference='restricted', molecule=self.source, frozen_orbital_indices=[0])
            mc = mcscf.CASSCF(self.mf, 2, 2)
            with self.assertRaisesRegex(ValueError, 'fresh MPS'):
                load_optimized_orbitals(manifest, self.mf, ncas=2, nelecas=2,
                    continuation_policy='target_scf_core', casscf=mc)
            mo, info = load_optimized_orbitals(manifest, self.mf, ncas=2, nelecas=2,
                continuation_policy='target_scf_core', casscf=mc,
                allow_geometry_projection=True, frozen_orbital_indices=[0])
            self.assertEqual(info['continuation_policy'], 'target_scf_core')
            np.testing.assert_allclose(mo[:, 0], self.mf.mo_coeff[:, 0], atol=1e-12)

    def test_target_scf_policy_rejects_frozen_active_or_external_columns(self):
        mc = mcscf.CASSCF(self.mf, 2, 2)
        for frozen in ([4], [6], [0, 0], [True]):
            with self.subTest(frozen=frozen), self.assertRaisesRegex(ValueError, 'inactive SCF'):
                project_casscf_active_guess(self.source, mc, self.mf.mo_coeff, frozen_indices=frozen)

    def test_noncontiguous_blocks_follow_column_identity_and_preserve_first_subspace(self):
        groups = [('protected', [3, 1]), ('frontier', [0, 6]), ('complement', [2, 4, 5])]
        mo, info = project_orbital_blocks(self.source, self.target, self.mf.mo_coeff, blocks=groups)
        np.testing.assert_allclose(mo.T @ self.s_target @ mo, np.eye(7), atol=1e-12)
        self.assertEqual([p['dimension'] for p in info['partitions']], [2, 2, 3])
        raw = scf.addons.project_mo_nr2nr(self.source, self.mf.mo_coeff[:, [3, 1]], self.target)
        raw_projector = raw @ np.linalg.solve(raw.T @ self.s_target @ raw, raw.T @ self.s_target)
        kept = mo[:, [3, 1]]
        np.testing.assert_allclose(kept @ kept.T @ self.s_target, raw_projector, atol=1e-12)

        order = np.array([6, 2, 4, 0, 5, 3, 1])
        inverse = np.argsort(order)
        reordered_groups = [(name, inverse[columns]) for name, columns in groups]
        reordered, _ = project_orbital_blocks(self.source, self.target,
            self.mf.mo_coeff[:, order], blocks=reordered_groups)
        np.testing.assert_allclose(reordered, mo[:, order], atol=1e-12)

    def test_occupied_only_rectangular_guess_preserves_electrons_and_converges_scf(self):
        occupied = self.mf.mo_coeff[:, :5]
        projected, info = project_orbital_blocks(self.source, self.target, occupied,
                                                blocks=[('occupied', range(5))])
        self.assertEqual(projected.shape, (7, 5))
        self.assertEqual(info['nmo'], 5)
        density = 2 * projected @ projected.T
        self.assertAlmostEqual(np.einsum('ij,ji', density, self.s_target), 10., places=11)
        warm = scf.RHF(self.target).run(dm0=density, conv_tol=1e-12)
        cold = scf.RHF(self.target).run(conv_tol=1e-12)
        self.assertTrue(warm.converged and cold.converged)
        self.assertAlmostEqual(warm.e_tot, cold.e_tot, places=10)

    def test_changed_basis_transfers_supplied_subspace_without_completing_virtuals(self):
        source = gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g', verbose=0)
        target = gto.M(atom='H 0 0 0; H 0 0 .80', basis='6-31g', verbose=0)
        mf = scf.RHF(source).run()
        mo, info = project_orbital_blocks(source, target, mf.mo_coeff,
                                         blocks=[('occupied', [0]), ('virtual', [1])])
        self.assertEqual(mo.shape, (4, 2))
        self.assertEqual((info['source_nao'], info['target_nao']), (2, 4))
        np.testing.assert_allclose(mo.T @ target.intor_symmetric('int1e_ovlp') @ mo,
                                   np.eye(2), atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'same atom ordering and AO basis layout'):
            project_partitioned_orbitals(source, target, mf.mo_coeff, ncore=0, ncas=2)
        target_mf = scf.RHF(target).run()
        with self.assertRaisesRegex(ValueError, 'rank exceeds'):
            project_orbital_blocks(target, source, target_mf.mo_coeff, blocks=[('all', range(4))])

    def test_intrablock_rotation_transfers_the_same_subspace(self):
        groups = [('occupied', range(5)), ('virtual', range(5, 7))]
        mo, _ = project_orbital_blocks(self.source, self.target, self.mf.mo_coeff, blocks=groups)
        rotation = np.eye(7)
        angle = .41
        rotation[3:5, 3:5] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        rotated, _ = project_orbital_blocks(self.source, self.target,
                                            self.mf.mo_coeff @ rotation, blocks=groups)
        np.testing.assert_allclose(rotated, mo @ rotation, atol=1e-12)

    def test_casscf_policy_handles_four_nonempty_blocks_and_empty_partitions(self):
        mo, info = project_partitioned_orbitals(self.source, self.target, self.mf.mo_coeff,
                                               ncore=4, ncas=2, frozen_indices=[0])
        self.assertEqual([p['dimension'] for p in info['partitions']], [1, 3, 2, 1])
        np.testing.assert_allclose(mo.T @ self.s_target @ mo, np.eye(7), atol=1e-12)
        h2 = gto.M(atom='H 0 0 0; H 0 0 .8', basis='sto-3g', verbose=0)
        mf = scf.RHF(h2).run()
        unchanged, info = project_partitioned_orbitals(h2, h2, mf.mo_coeff, ncore=0, ncas=2)
        self.assertEqual([p['dimension'] for p in info['partitions']], [0, 0, 2, 0])
        np.testing.assert_allclose(unchanged, mf.mo_coeff, atol=1e-12)

    def test_invalid_partition_contracts_do_not_silently_reassign_columns(self):
        invalid = [
            [('a', [0, 1]), ('b', [1, 2, 3, 4, 5, 6])],
            [('a', range(6))],
            [('a', [0]), ('a', range(1, 7))],
            [('', range(7))],
            [('a', [False, 1, 2, 3, 4, 5, 6])],
            [('a', [0., 1, 2, 3, 4, 5, 6])],
            [('a', range(8))],
        ]
        for blocks in invalid:
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                project_orbital_blocks(self.source, self.target, self.mf.mo_coeff, blocks=blocks)
        with self.assertRaisesRegex(ValueError, 'finite real'):
            project_orbital_blocks(self.source, self.target, self.mf.mo_coeff.astype(complex),
                                  blocks=[('all', range(7))])


if __name__ == '__main__':
    unittest.main()
