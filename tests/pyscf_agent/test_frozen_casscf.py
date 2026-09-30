"""Frozen MO indices must reach the optimizer and preserve the selected orbitals."""
import json
import tempfile
import unittest
from unittest import mock

import numpy as np
from pyscf import gto, scf, mcscf

from pyscf_agent.application.calculation_service import CalculationApplicationService
from pyscf_agent.backend.correlation.cas_execution import run_cas_method
from pyscf_agent.contracts import (
    ActiveSpaceSpec, DensityFittingSpec, OrbitalProcessingSpec,
    normalize_orbital_processing, task_spec_from_dict, task_spec_to_dict,
)
from pyscf_agent.executors import LocalExecutor
from pyscf_agent.providers.block2 import block2_is_available


class FrozenCASSCFTests(unittest.TestCase):
    def test_contract_rejects_invalid_indices_and_wrong_method(self):
        for indices in ([0, 0], [-1], [True], [1.2], '2'):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                normalize_orbital_processing({'frozen_orbital_indices': indices})
        saved = task_spec_to_dict(task_spec_from_dict({
            'orbital_processing': {'frozen_orbital_indices': [0, 5]},
        }))
        self.assertEqual(saved['orbital_processing']['frozen_orbital_indices'], [0, 5])
        validation = CalculationApplicationService().validate_task_spec({
            'atom': 'H 0 0 0; H 0 0 0.74', 'basis': 'sto-3g', 'method': 'hf',
            'orbital_processing': {'frozen_orbital_indices': [0]},
        })
        self.assertFalse(validation['valid'])
        self.assertTrue(any(e['code'] == 'frozen_orbitals_require_spin_adapted_casscf'
                            for e in validation['errors']))

    def test_continuation_policy_roundtrip_and_validation(self):
        saved = task_spec_to_dict(task_spec_from_dict({
            'orbital_processing': {'continuation_policy': 'target_scf_core'},
        }))
        self.assertEqual(saved['orbital_processing']['continuation_policy'], 'target_scf_core')
        with self.assertRaises(ValueError):
            normalize_orbital_processing({'continuation_policy': 'unknown'})
        validation = CalculationApplicationService().validate_task_spec({
            'atom': 'H 0 0 0; H 0 0 .74', 'basis': 'sto-3g', 'method': 'casscf',
            'active_space': {'ncas': 2, 'nelecas': 2, 'approved': True},
            'orbital_processing': {'continuation_policy': 'target_scf_core'},
        })
        self.assertFalse(validation['valid'])
        self.assertTrue(any(e['code'] == 'invalid_orbital_continuation_policy' for e in validation['errors']))

    def test_fci_preserves_frozen_mos_and_matches_native(self):
        self._check_frozen_mos('fci')

    @unittest.skipUnless(block2_is_available(), 'Optional block2 provider is not installed.')
    def test_block2_preserves_frozen_mos_and_matches_native(self):
        self._check_frozen_mos('block2_dmrg')

    def _check_frozen_mos(self, solver):
        mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).density_fit().run(conv_tol=1e-12)
        mo = mf.mo_coeff.copy()
        frozen = [0, 5]  # Both an occupied and an external MO.
        native = mcscf.CASSCF(mf, 2, 2, frozen=frozen).density_fit()
        native.verbose = 0
        native.kernel(mo)
        self.assertTrue(native.converged)
        with tempfile.TemporaryDirectory() as scratch:
            options = {} if solver == 'fci' else {
                'bond_dimensions': [32], 'final_bond_dimension': 64,
                'noises': [1e-5, 1e-6, 0., 0., 0., 0.], 'davidson_thresholds': [1e-12],
                'sweeps': 6, 'energy_tolerance': 1e-10,
                'adaptive_schedule': False, 'n_threads': 1,
                'entanglement_active_space_review': False,
            }
            result = run_cas_method(
                mf, mol, 'casscf',
                ActiveSpaceSpec(ncas=2, nelecas=2, initial_mo_coeff=mo), 50,
                solver_name=solver, solver_options=options, scratch_directory=scratch,
                orbital_processing=OrbitalProcessingSpec(frozen_orbital_indices=frozen),
                density_fitting=DensityFittingSpec(enabled=True, apply_to='scf_and_casscf'),
            )
            self.assertTrue(result['converged'])
            self.assertEqual(result['orbital_optimization']['frozen_orbital_indices'], frozen)
            self.assertEqual(result['orbital_optimization']['integral_space']['working_nmo'], 5)
            self.assertAlmostEqual(result['energy'], native.e_tot, places=7)
            if solver == 'block2_dmrg':
                from pyscf_agent.providers.block2 import load_optimized_orbitals
                final = result['_transient_dmrg_arrays']['optimized_mo_coeff']
                self.assertEqual(final.shape, mo.shape)
                np.testing.assert_allclose(final[:, frozen], mo[:, frozen], atol=1e-10)
                restored, _ = load_optimized_orbitals(
                    result['dmrg_result']['checkpoint_manifest'], mf, ncas=2, nelecas=(1, 1),
                )
                np.testing.assert_allclose(restored, final, atol=1e-12)

    def test_public_workflow_forwards_frozen_indices(self):
        original = mcscf.CASSCF
        seen = []
        def record(*args, **kwargs):
            optimizer = original(*args, **kwargs)
            seen.append(optimizer)
            return optimizer
        with tempfile.TemporaryDirectory() as root, mock.patch.object(mcscf, 'CASSCF', record):
            report = CalculationApplicationService(task_executor=LocalExecutor()).execute_request(
                json.dumps({
                    'atom': 'Li 0 0 0; H 0 0 1.6', 'basis': 'sto-3g',
                    'method': {'name': 'casscf', 'restricted': True},
                    'active_space': {'enabled': True, 'ncas': 2, 'nelecas': 2, 'approved': True},
                    'orbital_processing': {'frozen_orbital_indices': [0, 5]},
                }), work_dir=root, run_id='frozen', include_llm_feedback=False,
            )
        self.assertEqual(report['execution_status'], 'succeeded', report.get('errors'))
        self.assertEqual(seen[-1].frozen, [0, 5])
        self.assertEqual(seen[-1].mo_coeff.shape, (6, 6))
        self.assertEqual(report['structured_results']['cas_result']['orbital_optimization']['frozen_orbital_indices'], [0, 5])


if __name__ == '__main__':
    unittest.main()
