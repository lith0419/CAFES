"""Transport is a wavefunction guess; compare converged target roots independently."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pyscf import gto, lib, scf

from pyscf_agent.backend.correlation.orbital_projection import transport_mps_orbitals
from pyscf_agent.backend.correlation.cas_execution import run_cas_method
from pyscf_agent.backend.execution import _run_pyscf_task
from pyscf_agent.backend.validation import _validate_strong_correlation_spec
from pyscf_agent.contracts import ActiveSpaceSpec, task_spec_from_dict
from pyscf_agent.providers.block2 import block2_is_available, normalize_block2_options
from pyscf_agent.providers.block2.checkpoint import attach_optimized_orbitals, load_optimized_orbitals
from tests.pyscf_agent.test_block2_provider import _small_dmrg_options


class MPSTransportContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lib.num_threads(1)
        cls.source = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='6-31g', verbose=0)
        cls.target = gto.M(atom='Li 0 0 0; H 0 0 1.7', basis='6-31g', verbose=0)
        cls.mf = scf.RHF(cls.source).run()

    def test_existing_fci_route_accepts_omitted_solver_options(self):
        result = run_cas_method(self.mf, self.source, 'casci',
                               ActiveSpaceSpec(ncas=2, nelecas=2), max_cycle=50)
        self.assertTrue(np.isfinite(result['energy']))

    def test_alignment_preserves_labels_and_phases_under_source_gauge_changes(self):
        original = self.mf.mo_coeff.copy()
        target, info = transport_mps_orbitals(self.source, self.target, original, ncore=1, ncas=2)
        np.testing.assert_array_equal(original, self.mf.mo_coeff)
        np.testing.assert_allclose(target.T @ self.target.intor_symmetric('int1e_ovlp') @ target,
                                   np.eye(target.shape[1]), atol=1e-12)
        self.assertEqual([p['partition'] for p in info['partitions']],
                         ['frozen_core', 'active', 'inactive', 'external'])
        self.assertFalse(info['exact_wavefunction_transform'])
        rotation = np.eye(original.shape[1])
        rotation[1:3, 1:3] = [[0., -1.], [1., 0.]]
        changed, _ = transport_mps_orbitals(self.source, self.target, original @ rotation, ncore=1, ncas=2)
        np.testing.assert_allclose(changed, target @ rotation, atol=1e-12)
        cross = gto.intor_cross('int1e_ovlp', self.source, self.target)
        overlap = original[:, 1:3].T @ cross @ target[:, 1:3]
        np.testing.assert_allclose(overlap, overlap.T, atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(overlap).min(), .9)
        with self.assertRaisesRegex(ValueError, 'active-space overlap'):
            transport_mps_orbitals(self.source, self.target, original, ncore=1, ncas=2,
                                   minimum_active_overlap=1.)

    def test_checkpoint_opt_in_and_partition_guards(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = {'scratch_directory': root, 'files': []}
            attach_optimized_orbitals(manifest, self.mf.mo_coeff, ncore=1, ncas=2,
                                      nelecas=2, reference='restricted', molecule=self.source,
                                      frozen_orbital_indices=[0])
            before = copy.deepcopy(manifest)
            target_mf = scf.RHF(self.target)
            kwargs = dict(ncas=2, nelecas=2, frozen_orbital_indices=[0])
            with self.assertRaisesRegex(ValueError, 'same geometry'):
                load_optimized_orbitals(manifest, target_mf, **kwargs)
            _, info = load_optimized_orbitals(manifest, target_mf, **kwargs,
                                             restart_geometry_policy='transport')
            self.assertTrue(info['mps_initial_guess_transport'])
            self.assertEqual(manifest, before)
            for change, error in [({'ncas': 3}, 'same ncas'),
                                  ({'frozen_orbital_indices': []}, 'frozen-core'),
                                  ({'allow_geometry_projection': True}, 'joint checkpoint'),
                                  ({'continuation_policy': 'target_scf_core'}, 'joint checkpoint')]:
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, error):
                    load_optimized_orbitals(manifest, target_mf,
                        **{**kwargs, **change, 'restart_geometry_policy': 'transport'})

    def test_configuration_and_task_admission(self):
        self.assertEqual(normalize_block2_options().restart_geometry_policy, 'same_geometry')
        for threshold in (0, 1.1, float('nan'), True):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                normalize_block2_options({'restart_min_active_overlap': threshold})
        with self.assertRaises(ValueError):
            normalize_block2_options({'restart_geometry_policy': 'ignore'})
        base = {
            'task_type': 'molecular', 'system': {'atom': 'H 0 0 0; H 0 0 1', 'basis': 'sto-3g'},
            'method': {'name': 'casscf', 'restricted': True},
            'active_space': {'ncas': 2, 'nelecas': 2, 'approved': True},
            'solver': {'name': 'block2_dmrg', 'options': {
                'restart_manifest': '/prior.json', 'restart_geometry_policy': 'transport'}},
        }
        self.assertEqual(_validate_strong_correlation_spec(task_spec_from_dict(base)), [])
        for change in ('casci', 'unrestricted', 'optional', 'missing', 'orbital_only'):
            payload = copy.deepcopy(base)
            opts = payload['solver']['options']
            if change == 'casci': payload['method'] = {'name': 'casci', 'restricted': True}
            if change == 'unrestricted': payload['method'] = {'name': 'casscf', 'restricted': False}
            if change == 'optional': opts['restart_required'] = False
            if change == 'missing': opts.pop('restart_manifest')
            if change == 'orbital_only': opts['orbital_restart_manifest'] = opts.pop('restart_manifest')
            with self.subTest(change=change):
                self.assertIn('invalid_mps_transport', [e['code'] for e in
                    _validate_strong_correlation_spec(task_spec_from_dict(payload))])


@unittest.skipUnless(block2_is_available(), 'optional block2 provider is not installed')
class MPSTransportNumericalTests(unittest.TestCase):
    def test_single_and_four_root_cross_geometry_restart_matches_cold_roots(self):
        lib.num_threads(1)
        def request(distance, roots, options):
            options = {**_small_dmrg_options(), 'nroots': roots,
                       'entanglement_active_space_review': False, **options}
            if roots > 1:
                options['state_average_weights'] = [1. / roots] * roots
            return task_spec_from_dict({
                'task_type': 'molecular',
                'system': {'atom': '; '.join('H 0 0 {}'.format(i * distance) for i in range(4)),
                           'basis': 'sto-3g', 'spin': 0, 'unit': 'Angstrom'},
                'method': {'name': 'casscf', 'restricted': True},
                'solver': {'name': 'block2_dmrg', 'options': options},
                'active_space': {'ncas': 4, 'nelecas': 4, 'approved': True},
                'orbital_processing': {'enabled': True, 'orbital_ordering': 'fiedler'},
                'analysis': {'outputs': ['energy']},
            })
        for roots in (1, 4):
            with self.subTest(roots=roots), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                source = _run_pyscf_task(request(1., roots, {}), execution_directory=root / 'source')
                manifest = source['cas_result']['dmrg_result']['checkpoint_manifest']
                manifest_path = root / 'manifest.json'
                manifest_path.write_text(json.dumps(manifest))
                def hashes():
                    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (root / 'source').rglob('*') if p.is_file()}
                before = hashes()
                warm = _run_pyscf_task(request(1.03, roots, {
                    'restart_manifest': str(manifest_path), 'restart_required': True,
                    'restart_geometry_policy': 'transport',
                }), execution_directory=root / 'warm')
                cold = _run_pyscf_task(request(1.03, roots, {}), execution_directory=root / 'cold')
                self.assertEqual(before, hashes())
                self.assertEqual(json.loads(manifest_path.read_text()), manifest)
                dmrg = warm['cas_result']['dmrg_result']
                other = cold['cas_result']['dmrg_result']
                self.assertTrue(source['converged'] and warm['converged'] and cold['converged'])
                self.assertTrue(dmrg['casscf']['mps_initial_guess_transport_applied'])
                self.assertTrue(dmrg['casscf']['solver_call_trace'][0]['restart_applied'])
                self.assertEqual(dmrg['orbital_ordering']['permutation'], manifest['orbital_ordering']['permutation'])
                np.testing.assert_allclose(dmrg['state_energies'], other['state_energies'], atol=1e-7, rtol=0)
                self.assertEqual(len(dmrg['state_energies']), roots)


if __name__ == '__main__':
    unittest.main()
