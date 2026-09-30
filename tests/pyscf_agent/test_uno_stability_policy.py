from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock

import numpy as np
from pyscf import gto, mcscf, scf

from pyscf_agent.application.calculation_service import CalculationApplicationService
from pyscf_agent.backend.correlation.active_space import propose_active_space
from pyscf_agent.backend.correlation.cas_execution import run_cas_method
from pyscf_agent.backend.execution import _run_pyscf_task
from pyscf_agent.contracts import ActiveSpaceSpec, task_spec_from_dict
from pyscf_agent.executors.local import LocalExecutor


class UNOStabilityPolicyTests(unittest.TestCase):
    def test_public_diagnostics_stability_is_opt_in(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as work_dir:
                request = {
                    'system': {'atom': 'H 0 0 0; H 0 0 0.74', 'basis': 'sto-3g'},
                    'method': {'name': 'hf', 'restricted': False},
                    'density_fitting': {'enabled': True},
                    'workflow': {'module_config': {
                        'molecular.correlation_diagnostics': {'scf_stability': True},
                    }} if enabled else {},
                }
                with mock.patch.object(scf.uhf.UHF, 'stability', return_value=(None, None, True, False)) as stability:
                    report = CalculationApplicationService(task_executor=LocalExecutor()).execute_request(
                        json.dumps(request), work_dir=work_dir, run_id='stability-policy',
                        include_llm_feedback=False,
                    )
                self.assertEqual(report['execution_status'], 'succeeded')
                summary = report['structured_results']['scf_stability']
                self.assertEqual(summary['status'], 'completed' if enabled else 'not_requested')
                self.assertEqual(summary['stable'], False if enabled else None)
                self.assertEqual(stability.call_count, int(enabled))
                if enabled:
                    self.assertEqual(stability.call_args.kwargs, {
                        'internal': True, 'external': True, 'return_status': True, 'nroots': 1,
                    })

    def test_inline_runner_also_skips_implicit_stability(self):
        spec = task_spec_from_dict({
            'system': {'atom': 'H 0 0 0; H 0 0 0.74', 'basis': 'sto-3g'},
            'method': {'name': 'hf', 'restricted': False},
        })
        with mock.patch.object(scf.uhf.UHF, 'stability') as stability:
            result = _run_pyscf_task(spec)
        stability.assert_not_called()
        self.assertEqual(result['scf_stability']['status'], 'not_requested')
        risk = result['correlation_diagnostics']['molecular_correlation_risk']
        self.assertNotIn('scf_stability', [item['name'] for item in risk['solver_stress_components']])

    def _uhf_reference(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3', basis='sto-3g', verbose=0)
        mf = scf.UHF(mol)
        values, vectors = np.linalg.eigh(mf.get_ovlp())
        alpha = vectors @ np.diag(values ** -0.5)
        theta, phi = 0.8, 0.6
        rotation = np.eye(4)
        rotation[1:, 1:] = [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta) * np.cos(phi), np.cos(theta) * np.cos(phi), -np.sin(phi)],
            [np.sin(theta) * np.sin(phi), np.cos(theta) * np.sin(phi), np.cos(phi)],
        ]
        mf.mo_coeff = np.asarray([alpha, alpha @ rotation])
        mf.mo_occ = np.asarray([[1., 1., 0., 0.], [1., 1., 0., 0.]])
        mf.mo_energy = np.asarray([[-1., -.2, .4, .8], [-1., -.2, .4, .8]])
        mf.converged = True
        return mf, theta

    def _proposal(self, mf):
        with mock.patch.object(mf, 'get_fock', side_effect=AssertionError('UNO needs no new Fock build')):
            proposal = propose_active_space(mf, ActiveSpaceSpec(
                enabled=True, selection_method='occupation_window', occupation_window=(.01, 1.99),
            ))
        candidate = next(item for item in proposal['audit']['candidate_active_spaces'] if item['method'] == 'uno')
        self.assertEqual(candidate['status'], 'available')
        self.assertEqual(proposal['audit']['selected_candidate_method'], 'uno')
        self.assertEqual((proposal['ncas'], proposal['nelecas']), (2, 2))
        self.assertFalse(proposal['approved'])
        return proposal, candidate

    def test_uno_uses_density_eigenvectors_and_is_beta_permutation_invariant(self):
        mf, theta = self._uhf_reference()
        proposal, candidate = self._proposal(mf)
        np.testing.assert_allclose(candidate['uno_occupations'], [1 + np.cos(theta), 1 - np.cos(theta)], atol=1e-12)
        coeff = np.asarray(proposal['initial_mo_coeff'])
        overlap = mf.get_ovlp()
        np.testing.assert_allclose(coeff.T @ overlap @ coeff, np.eye(4), atol=1e-12)
        active = coeff[:, 1:3]
        density = np.sum(mf.make_rdm1(), axis=0)
        np.testing.assert_allclose(density @ overlap @ active, active * candidate['uno_occupations'], atol=1e-12)
        display = mf.mo_coeff[0][:, proposal['orbital_indices']]
        self.assertGreater(np.linalg.norm(active @ active.T - display @ display.T), .1)

        permutation = [2, 0, 3, 1]
        mf.mo_coeff[1] = mf.mo_coeff[1][:, permutation]
        mf.mo_occ[1] = mf.mo_occ[1][permutation]
        mf.mo_energy[1] = mf.mo_energy[1][permutation]
        permuted, other = self._proposal(mf)
        np.testing.assert_allclose(other['uno_occupations'], candidate['uno_occupations'], atol=1e-12)
        other_active = np.asarray(permuted['initial_mo_coeff'])[:, 1:3]
        np.testing.assert_allclose(other_active @ other_active.T, active @ active.T, atol=1e-12)
        self.assertEqual(permuted['orbital_indices'], proposal['orbital_indices'])

    def test_casci_consumes_uno_matrix_instead_of_canonical_display_indices(self):
        mf, _theta = self._uhf_reference()
        proposal, _candidate = self._proposal(mf)
        expected = mcscf.CASCI(mf, 2, 2).kernel(np.asarray(proposal['initial_mo_coeff']))[0]
        actual = run_cas_method(mf, mf.mol, 'casci', ActiveSpaceSpec(
            enabled=True, selection_method='occupation_window', ncas=2, nelecas=2,
            orbital_indices=proposal['orbital_indices'], initial_mo_coeff=proposal['initial_mo_coeff'],
            approved=True,
        ), max_cycle=30)
        self.assertAlmostEqual(actual['energy'], expected, places=11)
        self.assertEqual(actual['initial_orbital_guess'], 'active_space.initial_mo_coeff')


if __name__ == '__main__':
    unittest.main()
