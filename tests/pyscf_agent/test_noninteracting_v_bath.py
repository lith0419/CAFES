"""Physical limit checks for finite-graph noninteracting U+V embedding."""
import contextlib
import io
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

from pyscf_agent.providers.libdmet import libdmet_availability, validate_dmet_model_request
from pyscf_agent.providers.libdmet import dmet as adapter
from pyscf_agent.backend.model_hamiltonian.solver import run_model_hamiltonian_solver
from tests.pyscf_agent.test_libdmet_dmet import _ring_spec


def model():
    spec = _ring_spec(4, onsite_u=1.0)
    for b in spec['bonds']:
        b['V'] = b['effective_V'] = 0.2
    # Avoid the half-filled four-site ring's one-particle degeneracy.
    for i, site in enumerate(spec['sites']):
        site['epsilon'] = [0.3, -0.2, 0.1, -0.4][i]
    return spec


class NoninteractingVContractTests(unittest.TestCase):
    def test_explicit_false_is_preserved(self):
        errors, config = validate_dmet_model_request(model(), {'interacting_bath': False})
        self.assertEqual(errors, [])
        self.assertFalse(config['interacting_bath'])
        self.assertTrue(config['contains_intersite_v'])
        self.assertEqual(config['execution_mode'], 'finite_graph')


@unittest.skipUnless(libdmet_availability()['available'], 'libDMET unavailable')
class NoninteractingVPhysicsTests(unittest.TestCase):
    def test_omitted_interactions_and_full_physical_energy(self):
        from libdmet.routine import slater
        from libdmet.dmet import Hubbard as dmet
        from pyscf import ao2mo
        with contextlib.redirect_stdout(io.StringIO()):
            lat, ham = adapter._finite_graph_lattice(model(), np)
            h = ham.H1[0]
            _, orbitals = np.linalg.eigh(h)
            dm = orbitals[:, :2] @ orbitals[:, :2].T
            _, beta_orbitals = np.linalg.eigh(h + np.diag([.2,-.1,.3,-.2]))
            beta_dm = beta_orbitals[:,:2] @ beta_orbitals[:,:2].T
            rho = np.array([dm, beta_dm])
            config = {'reference': 'unrestricted', 'interacting_bath': False,
                      'contains_intersite_v': True}
            adapter._prepare_interacting_bath_mean_field(
                lat, ham, {'rho_k': rho[:, None]}, config, slater=slater, np=np)
            baseline = np.array([np.eye(4)*0.7, np.eye(4)*0.6])
            correction = np.array([np.diag([.01,.02,.03,.04])]*2)
            vcor = dmet.VcorLocal(False, False, 4)
            vcor.assign(baseline + correction)
            total = 0.0
            for indices in ([0,1], [2,3]):
                lat.set_val_virt_core(indices, [], [])
                order = indices + [i for i in range(4) if i not in indices]
                coeff = np.eye(4)[:,order]
                basis = np.array([coeff, coeff])[:,None]
                with mock.patch.object(slater, 'embBasis', return_value=basis):
                    emb, _, _ = adapter._construct_finite_graph_impurity_hamiltonian(
                        lat, rho[:,None], vcor, config, 'svd', dmet=dmet,
                        slater=slater, np=np, mean_field_baseline=baseline)
                d = np.array([coeff.T @ x @ coeff for x in rho])
                jk_imp = slater.get_veff(d, emb.H2['ccdd'])
                f = np.array([coeff.T @ x @ coeff for x in lat.fock_lo_k[:,0]])
                u = np.array([coeff.T @ x @ coeff for x in correction])
                u[:,:2,:2] = 0.0
                np.testing.assert_allclose(emb.H1['cd'], f-jk_imp+u, atol=1e-12)
                for spin_eri in emb.H2['ccdd']:
                    eri = ao2mo.restore(1, spin_eri, 4)
                    kept = np.zeros_like(eri); kept[:2,:2,:2,:2] = eri[:2,:2,:2,:2]
                    np.testing.assert_allclose(eri, kept, atol=1e-12)
                    self.assertGreater(abs(eri[0,0,1,1]), 0.1)
                old_jk = lat.JK_core.copy()
                def determinant_expectation(operator, **kwargs):
                    veff = slater.get_veff(d, operator.H2['ccdd'])
                    return np.einsum('sij,sji', operator.H1['cd'] + .5*veff, d) + operator.H0
                total += adapter._noninteracting_v_fragment_energy(
                    lat, basis, vcor, SimpleNamespace(run_dmet_ham=determinant_expectation),
                    {}, slater=slater)
                np.testing.assert_allclose(lat.JK_core, old_jk, atol=1e-12)
            expected = np.einsum('sij,sji', h[None] + .5*slater.get_veff(rho, ham.H2), rho)
            self.assertAlmostEqual(total, expected, places=10)
            np.testing.assert_allclose(vcor.get(), baseline+correction, atol=1e-12)

    def test_full_fragment_recovers_original_hamiltonian(self):
        from libdmet.routine import slater
        from libdmet.dmet import Hubbard as dmet
        from pyscf import ao2mo
        with contextlib.redirect_stdout(io.StringIO()):
            lat, ham = adapter._finite_graph_lattice(model(), np)
            lat.set_val_virt_core([0,1,2,3], [], [])
            _, c = np.linalg.eigh(ham.H1[0])
            rho = np.array([c[:,:2] @ c[:,:2].T]*2)
            config = {'reference': 'unrestricted', 'interacting_bath': False,
                      'contains_intersite_v': True}
            adapter._prepare_interacting_bath_mean_field(
                lat, ham, {'rho_k': rho[:,None]}, config, slater=slater, np=np)
            baseline = np.array([np.eye(4)*.4]*2)
            vcor = dmet.VcorLocal(False, False, 4); vcor.assign(baseline)
            basis = np.array([np.eye(4)]*2)[:,None]
            with mock.patch.object(slater, 'embBasis', return_value=basis):
                emb, _, _ = adapter._construct_finite_graph_impurity_hamiltonian(
                    lat, rho[:,None], vcor, config, 'svd', dmet=dmet,
                    slater=slater, np=np, mean_field_baseline=baseline)
            for h1 in emb.H1['cd']:
                np.testing.assert_allclose(h1, ham.H1[0], atol=1e-12)
            for eri in emb.H2['ccdd']:
                np.testing.assert_allclose(ao2mo.restore(1, eri, 4), ham.H2, atol=1e-12)

    def test_partitioned_noninteracting_v_executes(self):
        for reference, solver in (('restricted', 'fci'), ('unrestricted', 'fci'), ('unrestricted', 'ccsd')):
            with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = run_model_hamiltonian_solver(
                    model(), solver_name='dmet', outputs=['energy'],
                    solver_options={'interacting_bath': False, 'reference': reference, 'impurity_solver': solver,
                                    'fragments': [{'fragment_id': 'a', 'site_ids': [0,1]},
                                                  {'fragment_id': 'b', 'site_ids': [2,3]}],
                                    'max_iterations': 1, 'solver_max_memory_mb': 1000},
                    scratch_directory=scratch)
            self.assertTrue(np.isfinite(result['energy']))
            self.assertFalse(result['dmet_result']['bath']['interacting_bath'])
            self.assertEqual(result['dmet_result']['bath']['noninteracting_v_convention']['energy'],
                             'full_physical_hamiltonian_democratic_partition')
