from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from pyscf import gto, scf
from pyscf.scf import hf

from pyscf_agent.backend.correlation.active_space import propose_active_space
from pyscf_agent.application.calculation_service import CalculationApplicationService
from pyscf_agent.executors import LocalExecutor
from pyscf_agent.backend.one_particle_state import (
    capture_reference_1rdm, projected_initial_density, serialize_one_particle_state,
)
from pyscf_agent.contracts import ActiveSpaceSpec, task_spec_from_dict


class CASReferenceRestartTests(unittest.TestCase):
    def _probe(self):
        spec = task_spec_from_dict({
            'system': {'atom': 'H 0 0 0; H 0 0 0.74; H 0 0 5; H 0 0 8', 'basis': 'sto-3g'},
            'method': {'name': 'hf', 'restricted': False},
            'density_fitting': {'enabled': True},
        })
        mol = gto.M(atom=spec.system.atom, basis=spec.system.basis, verbose=0)
        mf = scf.UHF(mol).density_fit()
        atomic_guess = mf.get_init_guess()
        # Keep inactive, active and external orbitals, as in a molecular CAS.
        alpha = atomic_guess[0].copy()
        beta = atomic_guess[1].copy()
        total = atomic_guess[0] + atomic_guess[1]
        alpha[2, 2], beta[2, 2] = total[2, 2], 0.
        alpha[3, 3], beta[3, 3] = 0., total[3, 3]
        mf.kernel(dm0=(alpha, beta))
        self.assertTrue(mf.converged)
        return spec, mf

    def test_public_cas_workflow_reuses_uhf_density_and_applies_pm_fiedler(self):
        probe, mf = self._probe()
        candidate = propose_active_space(mf, ActiveSpaceSpec(enabled=True, selection_method='occupation_window'))
        self.assertEqual(candidate['audit']['selected_candidate_method'], 'uno')
        native_kernel = hf.kernel
        with tempfile.TemporaryDirectory() as work_dir:
            source = Path(work_dir) / 'probe.npz'
            source.write_bytes(serialize_one_particle_state(capture_reference_1rdm(probe, mf)))
            request = {
                'system': {'atom': probe.system.atom, 'basis': probe.system.basis},
                'method': {'name': 'casscf', 'restricted': True},
                'density_fitting': {'enabled': True, 'apply_to': 'scf_and_casscf'},
                'initial_state': {'mode': 'projected_1rdm', 'source_artifact': {'kind': 'one_particle_state', 'path': str(source)}},
                'active_space': {'enabled': True, 'selection_method': 'manual', 'ncas': 2, 'nelecas': 2,
                                 'orbital_indices': candidate['orbital_indices'], 'initial_mo_coeff': candidate['initial_mo_coeff'], 'approved': True},
            }
            solvers = ['fci'] + (['block2_dmrg'] if importlib.util.find_spec('block2') else [])
            energies = []
            for solver in solvers:
                with self.subTest(solver=solver):
                    request['solver'] = {'name': solver, 'options': {'nroots': 1} if solver == 'fci' else {
                        'nroots': 1, 'preset': 'screening', 'bond_dimensions': [16] * 6,
                        'noises': [1e-5, 1e-6, 0., 0., 0., 0.], 'davidson_thresholds': [1e-10] * 6, 'sweeps': 6,
                        'energy_tolerance': 1e-9, 'discarded_weight_tolerance': 1e-7,
                        'final_bond_dimension': 32, 'bond_dimension_planning': False,
                    }}
                    if solver == 'block2_dmrg':
                        request['orbital_processing'] = {
                            'enabled': True, 'localization_method': 'pipek_mezey',
                            'localization_scope': 'active_space', 'orbital_ordering': 'fiedler',
                            'localization_occupation_thresholds': [1.0],
                        }
                    seen = []
                    def track(reference, *args, **kwargs):
                        seen.append(np.asarray(kwargs.get('dm0')).copy())
                        return native_kernel(reference, *args, **kwargs)
                    with mock.patch.object(hf, 'kernel', track):
                        report = CalculationApplicationService(task_executor=LocalExecutor()).execute_request(
                            json.dumps(request), work_dir=work_dir, run_id=solver,
                            include_llm_feedback=False,
                        )
                    self.assertEqual(report['execution_status'], 'succeeded', report.get('errors'))
                    result = report['structured_results']
                    self.assertEqual(len(seen), 1)
                    np.testing.assert_allclose(seen[0], mf.make_rdm1(), atol=1e-12)
                    self.assertTrue(result['converged'])
                    self.assertTrue(result['cas_spin_adapted'])
                    self.assertEqual(result['reference'], 'uhf')
                    self.assertEqual(result['cas_reference'], 'rhf')
                    self.assertEqual(result['density_fitting']['applied_to'], ['scf', 'casscf'])
                    self.assertEqual(result['initial_state']['source_reference'], 'uhf')
                    self.assertTrue(result['initial_state']['source_scf_converged'])
                    self.assertEqual(result['cas_result']['initial_orbital_guess'],
                                     'pipek_mezey_localized_active_space' if solver == 'block2_dmrg'
                                     else 'active_space.initial_mo_coeff')
                    if solver == 'block2_dmrg':
                        self.assertEqual(result['cas_result']['orbital_provenance']['localization_occupation_thresholds'], [1.0])
                        self.assertEqual(result['cas_result']['dmrg_result']['configuration']['bond_dimensions'], [32])
                        self.assertFalse(result['cas_result']['dmrg_result']['configuration']['adaptive_schedule'])
                        self.assertEqual(result['cas_result']['dmrg_result']['orbital_ordering']['requested_method'], 'fiedler')
                        self.assertTrue(result['cas_result']['dmrg_result']['restart']['applied'])
                        casscf = result['cas_result']['dmrg_result']['casscf']
                        trace = casscf['solver_call_trace']
                        self.assertEqual(trace[-1]['role'], 'final_analysis')
                        self.assertTrue(any(call['role'] == 'orbital_response' for call in trace))
                        for call in trace:
                            self.assertEqual(set(call['bond_dimensions']), {32 if call['role'] == 'final_analysis' else 16})
                            self.assertEqual(call['sweep_limit'], 6)
                            self.assertEqual(call['orbital_ordering']['casscf_initial_request'], 'fiedler')
                        permutation = trace[0]['orbital_ordering']['permutation']
                        self.assertTrue(all(call['orbital_ordering']['permutation'] == permutation for call in trace))
                    energies.append(result['energy'])
            if len(energies) == 2:
                self.assertAlmostEqual(energies[0], energies[1], places=8)

    def test_cas_reference_artifact_records_actual_uhf_spin_policy(self):
        spec, mf = self._probe()
        spec.method.name = 'casscf'
        spec.method.restricted = True
        payload = capture_reference_1rdm(spec, mf)
        self.assertEqual(payload['metadata']['reference'], 'uhf')
        self.assertFalse(payload['metadata']['restricted'])
        self.assertEqual(payload['metadata']['spin_mode'], 'unrestricted')
        with tempfile.TemporaryDirectory() as work_dir:
            source = Path(work_dir) / 'cas-reference.npz'
            source.write_bytes(serialize_one_particle_state(payload))
            spec.initial_state.mode = 'projected_1rdm'
            spec.initial_state.source_artifact = {'kind': 'one_particle_state', 'path': str(source)}
            density, _summary = projected_initial_density(spec, mf.mol)
            np.testing.assert_allclose(density, mf.make_rdm1(), atol=1e-12)
            spec.method.name = 'hf'
            with self.assertRaisesRegex(ValueError, 'restricted/unrestricted'):
                projected_initial_density(spec, mf.mol)


if __name__ == '__main__':
    unittest.main()
