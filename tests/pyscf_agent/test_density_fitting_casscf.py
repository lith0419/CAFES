from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pyscf_agent.application.calculation_service import CalculationApplicationService
from pyscf_agent.backend.active_space_probe import build_molecular_active_space_probe_request
from pyscf_agent.backend.execution import _run_pyscf_task
from pyscf_agent.contracts import task_spec_from_dict
from pyscf_agent.registry.platform import density_fitting_scopes
from pyscf_agent.request_builder.llm import _llm_response_json_schema


def casscf_request():
    return {
        'task_type': 'molecular',
        'system': {'atom': 'H 0 0 0; H 0 0 1.5', 'basis': 'def2-svp', 'spin': 0},
        'method': {'name': 'casscf', 'restricted': True},
        'density_fitting': {'enabled': True, 'auxbasis': 'def2-svp-jkfit', 'apply_to': 'scf_and_casscf'},
        'active_space': {'enabled': True, 'ncas': 2, 'nelecas': 2, 'orbital_indices': [0, 1], 'approved': True},
        'solver': {'name': 'fci', 'options': {'nroots': 1}},
        'analysis': {'outputs': ['energy']},
    }


class DensityFittingContractTests(unittest.TestCase):
    def test_legacy_casscf_scope_preserves_implicit_fitting(self):
        request = casscf_request()
        request['density_fitting'] = {'enabled': True}
        outcome = CalculationApplicationService().validate_task_spec(request)
        self.assertTrue(outcome['valid'], outcome['errors'])
        self.assertEqual(outcome['task_spec']['density_fitting']['apply_to'], 'scf_and_casscf')

    def test_combined_scope_requires_supported_casscf(self):
        for method, restricted in [('hf', True), ('casci', True), ('casscf', False)]:
            with self.subTest(method=method, restricted=restricted):
                request = casscf_request()
                request['method'] = {'name': method, 'restricted': restricted}
                outcome = CalculationApplicationService().validate_task_spec(request)
                self.assertIn('unsupported_density_fitting_method', [e['code'] for e in outcome['errors']])

    def test_svpd_accepts_matching_jkfit_and_schema_uses_registry(self):
        request = casscf_request()
        request['system']['basis'] = 'def2-svpd'
        outcome = CalculationApplicationService().validate_task_spec(request)
        self.assertTrue(outcome['valid'], outcome['errors'])
        schema = _llm_response_json_schema()['schema']['properties']
        self.assertEqual(schema['density_fitting']['properties']['apply_to']['enum'], density_fitting_scopes())

    def test_reference_probe_uses_scf_scope_without_mutating_target(self):
        request = casscf_request()
        original = copy.deepcopy(request)
        probe = build_molecular_active_space_probe_request(request, target_method='casscf')
        self.assertEqual(probe['request']['density_fitting']['apply_to'], 'scf')
        self.assertEqual(request, original)


@unittest.skipUnless(importlib.util.find_spec('pyscf'), 'PySCF is required')
class DensityFittingNumericalTests(unittest.TestCase):
    def test_public_workflow_resolves_auto_auxbasis_and_summary(self):
        from pyscf_agent.backend.workflow import execute_request
        request = casscf_request()
        request['density_fitting'] = {'enabled': True}
        report = execute_request(json.dumps(request), channel='test', locale='en')['task_report']
        self.assertEqual(report['execution_status'], 'succeeded')
        metadata = report['structured_results']['density_fitting']
        self.assertIsNone(metadata['auxbasis'])
        self.assertEqual(metadata['resolved_auxbasis'], 'def2-svp-jkfit')
        self.assertEqual(metadata['applied_to'], ['scf', 'casscf'])
        self.assertIn('density_fitting=SCF+CASSCF', report['analysis_summary'])

    def test_df_optimizer_uses_df_eris_and_reports_actual_scope(self):
        from pyscf.mcscf import df
        original = df._DFCASSCF.ao2mo
        calls = []

        def track(optimizer, *args, **kwargs):
            eris = original(optimizer, *args, **kwargs)
            calls.append(type(eris).__module__)
            return eris

        with mock.patch.object(df._DFCASSCF, 'ao2mo', track):
            result = _run_pyscf_task(task_spec_from_dict(casscf_request()))
        self.assertTrue(result['converged'])
        self.assertTrue(calls)
        self.assertEqual(set(calls), {'pyscf.mcscf.df'})
        self.assertEqual(result['density_fitting']['applied_to'], ['scf', 'casscf'])
        self.assertEqual(result['density_fitting']['casscf_implementation'], 'DFCASSCF')
        self.assertEqual(result['density_fitting']['resolved_auxbasis'], 'def2-svp-jkfit')

    def test_disabled_fitting_stays_conventional(self):
        request = casscf_request()
        request['density_fitting'] = {'enabled': False}
        result = _run_pyscf_task(task_spec_from_dict(request))
        self.assertTrue(result['converged'])
        self.assertEqual(result['density_fitting']['applied_to'], [])
        self.assertFalse(result['cas_result']['density_fitting']['enabled'])

    @unittest.skipUnless(importlib.util.find_spec('block2'), 'block2 is required')
    def test_ground_state_df_dmrg_casscf_matches_df_fci(self):
        request = casscf_request()
        exact = _run_pyscf_task(task_spec_from_dict(request))
        request['solver'] = {'name': 'block2_dmrg', 'options': {
            'nroots': 1, 'preset': 'screening',
            'bond_dimensions': [16, 32, 32, 32, 32, 32],
            'noises': [1e-5, 1e-6, 0, 0, 0, 0],
            'davidson_thresholds': [1e-9] * 6, 'sweeps': 6,
            'energy_tolerance': 1e-9, 'discarded_weight_tolerance': 1e-7,
            'save_mps': False,
        }}
        with tempfile.TemporaryDirectory() as scratch:
            dmrg = _run_pyscf_task(task_spec_from_dict(request), execution_directory=Path(scratch))
        self.assertTrue(dmrg['converged'])
        self.assertEqual(dmrg['density_fitting']['applied_to'], ['scf', 'casscf'])
        self.assertEqual(dmrg['density_fitting']['casscf_implementation'], 'DFCASSCF')
        self.assertAlmostEqual(dmrg['energy'], exact['energy'], places=8)


if __name__ == '__main__':
    unittest.main()
