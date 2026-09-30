"""Regression tests for scientific input, evidence and recovery boundaries."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from unittest.mock import patch

from pyscf_agent.application.calculation_service import CalculationApplicationService
from pyscf_agent.backend.execution import repair_or_retry
from pyscf_agent.backend.result_artifacts import write_result_artifacts
from pyscf_agent.backend.state import default_state
from pyscf_agent.contracts import TaskSpec, task_spec_from_dict, task_spec_to_dict
from pyscf_agent.providers.block2.config import normalize_block2_options
from pyscf_agent.request_builder.prepare import _merge_structured_request
from computational_study_agent.adaptive.executor import analyze_initial_scan_report, _initial_scan_issues_from_decisions
from computational_study_agent.adaptive.refinement import _nearest_decision
from computational_study_agent.schema import StudySpec
from computational_study_agent.web_api import handle_study_llm_draft_request
from tests.computational_study_agent.support import hubbard_dimer_spec


class ScientificInputBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.service = CalculationApplicationService()
        self.base = {'atom': 'H 0 0 0; H 0 0 0.74', 'basis': 'sto-3g', 'method': 'hf'}

    def test_invalid_explicit_values_never_turn_into_defaults(self):
        for update in (
            {'charge': 'invalid'}, {'spin': True}, {'runtime': {'max_cycle': 'invalid'}},
            {'runtime': {'max_cycle': 3.7}}, {'runtime': {'conv_tol': 'not-a-number'}},
            {'runtime': {'conv_tol': float('nan')}}, {'runtime': {'conv_tol': float('inf')}},
            {'runtime': {'conv_toll': 1e-12}}, {'runtime': 'invalid'}, {'conv_toll': 1e-12},
            {'active_space': {'nelecas': [2, 'bad']}}, {'outputs': ['energy', 7]},
        ):
            with self.subTest(update=update):
                result = self.service.validate_task_spec({**self.base, **update})
                self.assertFalse(result['valid'])
                self.assertTrue(result['errors'])

    def test_missing_defaults_and_lossless_aliases_remain_supported(self):
        result = self.service.validate_task_spec({**self.base, 'charge': '0', 'symmetry': 'false', 'max_cycle': '50'})
        self.assertTrue(result['valid'], result['errors'])
        self.assertEqual(result['task_spec']['runtime']['max_cycle'], 50)
        self.assertFalse(result['task_spec']['system']['symmetry'])

    def test_new_requests_reject_unknown_fields_while_readers_accept_extensions(self):
        payload = task_spec_to_dict(task_spec_from_dict({'system': {'atom': self.base['atom'], 'basis': 'sto-3g'}}))
        payload['runtime']['future_field'] = 1
        self.assertIsInstance(task_spec_from_dict(payload), TaskSpec)
        result = self.service.validate_task_spec(payload)
        self.assertFalse(result['valid'])
        self.assertIn('future_field', result['errors'][0]['message'])

    def test_provider_options_are_validated_before_execution(self):
        cas = {**self.base, 'method': 'casci', 'active_space': {'ncas': 2, 'nelecas': 2, 'approved': True}}
        for options in ({'energy_tolerence': 1e-12}, {'sweeps': 0}, {'nroots': 1.5},
                        {'noises': [float('nan')]}, {'energy_tolerance': float('nan')}):
            with self.subTest(options=options):
                result = self.service.validate_task_spec({**cas, 'solver': {'name': 'block2_dmrg', 'options': options}})
                self.assertFalse(result['valid'])
                self.assertIn('invalid_block2_options', [error['code'] for error in result['errors']])
        self.assertFalse(normalize_block2_options({'adaptive_schedule': 'false'}).adaptive_schedule)

    def test_md_profile_is_explicit(self):
        result = self.service.validate_task_spec({**self.base, 'job': 'molecular_dynamics'})
        self.assertFalse(result['valid'])
        self.assertIn('md_profile_required', [error['code'] for error in result['errors']])
        self.assertEqual(result['task_spec']['system']['basis'], 'sto-3g')

    def test_free_text_does_not_override_structured_method(self):
        for text in ('Use MP2, not CASSCF.', 'Compare MP2 with FCI, keeping MP2.'):
            result = _merge_structured_request(self.base, {'method': 'mp2'}, text)
            self.assertEqual(result['method'], 'mp2')

    def test_malformed_llm_parameters_remain_visible_to_validation(self):
        request = _merge_structured_request(self.base, {'max_cycle': 'bad'}, 'Use 50 SCF cycles.')
        self.assertEqual(request['max_cycle'], 'bad')
        self.assertFalse(self.service.validate_task_spec(request)['valid'])


class PlannerBoundaryTests(unittest.TestCase):
    def draft(self, spec, goal, seed=None):
        with patch('computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
                   return_value={'study_spec': copy.deepcopy(spec), 'wiki_evidence': {}}):
            status, _, body = handle_study_llm_draft_request(json.dumps({'goal': goal, 'study_spec': seed}).encode())
        self.assertEqual(status.value, 200, body)
        return json.loads(body)

    def model_spec(self, solver):
        return {'name': 'model-study', 'objective': 'Compare energies', 'system_type': 'model_hamiltonian',
                'base_model_spec': hubbard_dimer_spec(), 'base_task': {'solver': solver}, 'observables': ['energy']}

    def test_negated_solver_name_does_not_override_draft(self):
        result = self.draft(self.model_spec('fci'), 'Use FCI, not DMET.')
        self.assertEqual(result['study_spec']['base_task']['solver'], 'fci')
        self.assertEqual(result['plan']['cases'][0]['request']['solver'], 'fci')

    def test_impurity_solver_is_not_promoted_to_top_level(self):
        solver = {'name': 'dmet', 'options': {'impurity_solver': 'block2_dmrg', 'execution_mode': 'finite_graph'}}
        result = self.draft(self.model_spec(solver), 'Use DMET with block2_dmrg as the impurity solver.')
        self.assertEqual(result['study_spec']['base_task']['solver'], solver)

    def test_full_revision_preserves_global_and_local_changes(self):
        seed = {'name': 'bond-scan', 'objective': 'Scan bond lengths', 'system_type': 'molecular',
                'base_task': {'atom': 'H 0 0 0; H 0 0 $(bond_factor * 0.74)', 'basis': 'sto-3g', 'method': 'hf'},
                'case_design': {'mode': 'grid', 'variables': {'bond_factor': [1, 1.6, 2]}}, 'observables': ['energy']}
        draft = copy.deepcopy(seed)
        draft['base_task']['basis'] = 'def2-svp'
        draft['case_design']['overrides'] = [{'selector': {'bond_factor': 1.6}, 'request_updates': {'method': 'mp2'}}]
        result = self.draft(draft, 'Change bond_factor=1.6 to MP2 and use def2-svp for the whole scan.', seed)
        self.assertEqual(result['study_spec']['base_task']['basis'], 'def2-svp')
        self.assertEqual([case['request']['method'] for case in result['plan']['cases']], ['hf', 'mp2', 'hf'])

    def test_explicit_unsupported_mode_is_preserved_and_rejected(self):
        spec = self.model_spec('fci')
        spec['study_mode'] = 'adaptive'
        result = self.draft(spec, 'Run an adaptive model study.')
        self.assertEqual(result['study_spec']['study_mode'], 'adaptive')
        self.assertEqual(result['status'], 'needs_input')

    def test_equivalent_geometry_and_range_formats_use_shared_contract(self):
        payload = {'name': 'range', 'objective': 'Scan a bond', 'system_type': 'molecular',
                   'base_task': {'geometry': [['H', [0, 0, 0]], ['H', [0, 0, 0.74]]]},
                   'case_design': {'variables': {'factor': {'start': 0.2, 'stop': 0.6, 'step': 0.2}}}}
        spec = StudySpec.from_dict(payload)
        self.assertEqual(spec.base_task['atom'], 'H 0 0 0; H 0 0 0.74')
        self.assertEqual(spec.case_design['variables']['factor'], [0.2, 0.4, 0.6])


class EvidenceRecoveryBoundaryTests(unittest.TestCase):
    def test_missing_diagnostics_cannot_route_to_weak_correlation(self):
        decisions = analyze_initial_scan_report({'comparison_table': [{'case_id': 'case-1', 'status': 'succeeded'}], 'cases': []})
        self.assertIsNone(decisions[0]['recommended_solver'])
        self.assertTrue(_initial_scan_issues_from_decisions(decisions))
        with self.assertRaisesRegex(ValueError, 'without initial-scan'):
            _nearest_decision({}, [])

    def test_error_text_alone_does_not_trigger_retry(self):
        state = default_state('{}')
        state.update(task_spec=task_spec_to_dict(TaskSpec()), execution_status='failed', max_retries=1,
                     raw_stderr='Unexpected bug while reading convergence metadata')
        self.assertEqual(repair_or_retry(state)['retry_count'], 0)

    def test_same_method_retry_preserves_reference_and_records_changes(self):
        spec = TaskSpec()
        spec.system.spin = 1
        spec.method.restricted = True
        state = default_state('{}')
        state.update(task_spec=task_spec_to_dict(spec), execution_status='unconverged', max_retries=1,
                     structured_results={'reference_converged': False})
        recovered = repair_or_retry(state)
        self.assertTrue(recovered['task_spec']['method']['restricted'])
        self.assertEqual(recovered['task_spec']['runtime']['max_cycle'], 100)
        event = next(item for item in recovered['logs'] if item['event'] == 'workflow.retry_scheduled')
        self.assertEqual(event['details']['previous_task_spec']['runtime']['max_cycle'], 50)

    def test_optional_array_failure_preserves_energy_and_failure_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            state = default_state('{}', work_dir=directory)
            with patch('pyscf_agent.backend.result_artifacts.write_binary_artifact', side_effect=OSError('disk full')):
                result = write_result_artifacts(state, TaskSpec(), {'energy': -1.0, '_transient_dmrg_arrays': {'energies': [0.0]}})
        self.assertEqual(result['energy'], -1.0)
        self.assertEqual(result['artifact_errors'][0]['kind'], 'block2_dmrg_arrays')
        self.assertIn('disk full', result['artifact_errors'][0]['reason'])
        self.assertTrue(any(item['event'] == 'artifact.write_failed' for item in state['logs']))
