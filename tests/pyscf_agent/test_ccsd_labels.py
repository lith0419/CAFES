from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pyscf import cc, gto, lib, scf

from pyscf_agent.backend.ccsd_labels import capture_ccsd_labels
from pyscf_agent.backend.workflow import execute_request
from pyscf_agent.backend.execution import _run_pyscf_task
from pyscf_agent.contracts import task_spec_from_dict


def request():
    return {
        'task_type': 'molecular',
        'system': {'atom': 'H 0 0 0; H 0 0 0.9', 'basis': 'def2-svp', 'spin': 0},
        'method': {'name': 'ccsd', 'restricted': True},
        'analysis': {'outputs': ['energy', 'ccsd_labels']},
        'runtime': {'conv_tol': 1e-10, 'conv_tol_grad': 1e-6, 'max_cycle': 100, 'verbose': 0},
        'workflow': {'module_config': {
            'molecular.correlation_diagnostics': {'enabled': False, 'scf_stability': False},
            'core.execution': {'label_provenance': {'sample_id': 'h2-test'}},
        }},
    }


class CCSDLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lib.num_threads(1)

    def test_full_workflow_matches_independent_ccsd_and_never_diagnoses(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch('pyscf_agent.backend.execution.build_correlation_diagnostics', side_effect=AssertionError('diagnostics ran')), \
                patch('pyscf_agent.backend.molecular_modules.build_correlation_diagnostics', side_effect=AssertionError('module diagnostics ran')), \
                patch('pyscf_agent.backend.molecular_modules.build_scf_stability_summary', side_effect=AssertionError('stability ran')):
            report = execute_request(json.dumps(request()), work_dir=directory, run_id='labels')['task_report']
            self.assertEqual(report['execution_status'], 'succeeded', report.get('errors'))
            self.assertNotIn('correlation_diagnostics', report['structured_results'])
            manifest = report['structured_results']['ccsd_labels']
            path = Path(manifest['data_artifact']['path'])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), manifest['artifact_sha256'])
            self.assertTrue(manifest['lambda_converged'])
            self.assertEqual(manifest['schema'], 'pyscf-agent.ccsd-labels.v2')
            self.assertEqual(manifest['provenance']['sample_id'], 'h2-test')
            with np.load(path, allow_pickle=False) as archive:
                labels = {key: archive[key] for key in archive.files}
            mol = gto.M(atom=request()['system']['atom'], basis='def2-svp', verbose=0)
            mf = scf.RHF(mol).set(conv_tol=1e-10, conv_tol_grad=1e-6).run()
            solver = cc.CCSD(mf).run()
            solver.solve_lambda()
            index = labels['ao_source_indices']
            signs = labels['ao_phase_signs']
            def transformed(value):
                return value[np.ix_(index, index)] * signs[:, None] * signs[None, :]
            np.testing.assert_allclose(labels['D_CCSD'], transformed(solver.make_rdm1(ao_repr=True)), atol=1e-9)
            np.testing.assert_allclose(labels['F_HF'], transformed(mf.get_fock()), atol=1e-9)
            np.testing.assert_allclose(labels['S'], transformed(mf.get_ovlp()), atol=1e-12)
            np.testing.assert_allclose(labels['t1'], solver.t1, atol=1e-9)
            np.testing.assert_allclose(labels['C_HF'], mf.mo_coeff[index, :] * signs[:, None], atol=1e-9)
            np.testing.assert_allclose(labels['epsilon_HF'], mf.mo_energy, atol=1e-9)
            np.testing.assert_array_equal(labels['mo_occ'], mf.mo_occ)
            np.testing.assert_array_equal(labels['occupied_mo_indices'], np.arange(solver.nocc))
            np.testing.assert_array_equal(labels['virtual_mo_indices'], np.arange(solver.nocc, solver.nmo))
            self.assertAlmostEqual(float(labels['E_HF']), mf.e_tot, places=10)
            self.assertAlmostEqual(float(labels['E_CCSD']), solver.e_tot, places=10)
            self.assertGreater(np.linalg.norm(labels['D_CCSD'] - transformed(mf.make_rdm1())), 1e-3)
            self.assertAlmostEqual(np.einsum('ij,ji->', labels['D_CCSD'], labels['S']), 2., places=8)

    def test_saved_t1_and_orbitals_preserve_mo_phases_and_ao_mapping(self):
        mol = gto.M(atom='O 0 0 0; H 0 -.75 .59; H 0 .78 .61', basis='def2-svp', verbose=0)
        mf = scf.RHF(mol).set(conv_tol=1e-11).run()
        solver = cc.CCSD(mf).set(conv_tol=1e-10, conv_tol_normt=1e-9).run()
        original = capture_ccsd_labels(task_spec_from_dict(request()), mf, solver)['arrays']
        phases = np.where(np.arange(mf.mo_coeff.shape[1]) % 2, -1., 1.)
        mf.mo_coeff = mf.mo_coeff * phases[None, :]
        flipped = cc.CCSD(mf).set(conv_tol=1e-10, conv_tol_normt=1e-9).run()
        labels = capture_ccsd_labels(task_spec_from_dict(request()), mf, flipped)['arrays']
        nocc = flipped.nocc
        self.assertGreater(np.linalg.norm(original['t1']), 1e-4)
        np.testing.assert_allclose(labels['t1'], original['t1'] * phases[:nocc, None] * phases[None, nocc:], atol=1e-8)
        np.testing.assert_allclose(labels['C_HF'], original['C_HF'] * phases[None, :], atol=1e-12)
        np.testing.assert_allclose(labels['C_HF'].T @ labels['S'] @ labels['C_HF'], np.eye(flipped.nmo), atol=1e-10)
        np.testing.assert_allclose(labels['F_HF'] @ labels['C_HF'],
                                   (labels['S'] @ labels['C_HF']) * labels['epsilon_HF'], atol=1e-5)
        np.testing.assert_allclose(labels['D_CCSD'], original['D_CCSD'], atol=1e-8)
        reconstructed = labels['C_HF'][:, :nocc] @ labels['t1'] @ labels['C_HF'][:, nocc:].T
        reference = original['C_HF'][:, :nocc] @ original['t1'] @ original['C_HF'][:, nocc:].T
        np.testing.assert_allclose(reconstructed, reference, atol=1e-8)

    def test_direct_execution_also_disables_diagnostics(self):
        with patch('pyscf_agent.backend.execution.build_correlation_diagnostics', side_effect=AssertionError('diagnostics ran')), \
                patch('pyscf_agent.backend.execution.build_scf_stability_summary', side_effect=AssertionError('stability ran')):
            result = _run_pyscf_task(task_spec_from_dict(request()))
            self.assertIn('_transient_ccsd_labels', result)
            self.assertNotIn('scf_stability', result)

    def test_lambda_failure_cannot_produce_accepted_labels(self):
        mol = gto.M(atom=request()['system']['atom'], basis='def2-svp', verbose=0)
        mf = SimpleNamespace(mol=mol, converged=True)
        solver = SimpleNamespace(converged=True, frozen=None, solve_lambda=lambda: None, converged_lambda=False)
        with self.assertRaisesRegex(ValueError, 'Lambda'):
            capture_ccsd_labels(task_spec_from_dict(request()), mf, solver)

    def test_incompatible_request_blocked_before_execution(self):
        payload = request()
        payload['method']['name'] = 'mp2'
        with tempfile.TemporaryDirectory() as directory:
            report = execute_request(json.dumps(payload), work_dir=directory)['task_report']
        self.assertEqual(report['execution_status'], 'blocked')
        self.assertTrue(any('ccsd_labels' in value for value in report['validation_errors']))


if __name__ == '__main__':
    unittest.main()
