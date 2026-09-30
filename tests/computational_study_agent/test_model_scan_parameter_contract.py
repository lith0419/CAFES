from __future__ import annotations

import copy
import json
import unittest
from unittest.mock import Mock, patch

from computational_study_agent.grid.refinement import build_point
from computational_study_agent.llm_planner import PlannerDraftError, build_study_spec_from_goal
from computational_study_agent.planner import build_study_plan
from computational_study_agent.validation import validate_study_spec
from tests.computational_study_agent.support import hubbard_dimer_spec


def scan_spec():
    return {
        'name': 'hubbard-uv-regression', 'objective': 'Scan U and V with fixed CCSD-DMET',
        'system_type': 'model_hamiltonian', 'base_model_spec': hubbard_dimer_spec(),
        'base_task': {'solver': {'name': 'dmet', 'options': {
            'execution_mode': 'finite_graph', 'fragment_definition': 'site_count',
            'impurity_size': 1, 'impurity_solver': 'ccsd',
            'impurity_solver_options': {'beta': 1000}, 'reference': 'unrestricted',
        }}},
        'case_design': {'mode': 'grid', 'variables': {'U': [1, 2, 4, 8], 'V': [.5, 1, 2, 4]},
                        'template': {'operations': [
                            {'op': 'set_global_parameter', 'parameter': 'U', 'value': '$U'},
                            {'op': 'set_global_parameter', 'parameter': 'V', 'value': '$V'},
                        ]}},
        'observables': ['energy'],
        'grid_refinement': {'axes': ['U', 'V'], 'min_spacing': {'U': .25, 'V': .25}},
    }


def misplaced_spec():
    spec = scan_spec()
    spec['case_design']['template'] = {
        'operations': [], 'request_updates': {'parameters': {'U': '$U', 'V': '$V'}},
    }
    return spec


class ModelScanParameterContractTests(unittest.TestCase):
    def test_misplaced_parameters_are_rejected_with_and_without_refinement(self):
        for refine in (False, True):
            for location in ('template', 'cases', 'overrides'):
                with self.subTest(refine=refine, location=location):
                    spec = scan_spec()
                    if not refine:
                        spec.pop('grid_refinement')
                    updates = {'request_updates': {'parameters': {'U': '$U', 'V': '$V'}}}
                    if location == 'template':
                        spec['case_design']['template'] = updates
                    elif location == 'cases':
                        spec['case_design'] = {'mode': 'cases', 'cases': [
                            {'variables': {'U': 1, 'V': .5}, **updates},
                        ]}
                    else:
                        spec['case_design']['overrides'] = [{'selector': {'U': 1}, **updates}]
                    issues = validate_study_spec(spec)
                    errors = [i for i in issues if i.severity == 'error']
                    self.assertEqual(len(errors), 1, errors)
                    self.assertEqual(errors[0].code, 'invalid_model_parameter_updates')
                    self.assertIn('request_updates.parameters', errors[0].message)
                    self.assertIn('set_global_parameter', errors[0].message)
                    with self.assertRaisesRegex(ValueError, r'request_updates\.parameters'):
                        build_study_plan(spec)

    def test_seed_and_inserted_points_change_coefficients_and_keep_solver(self):
        spec = scan_spec()
        before = copy.deepcopy(spec)
        plan = build_study_plan(spec)
        self.assertEqual(len(plan.cases), 16)
        self.assertEqual({(c.variables['U'], c.variables['V']) for c in plan.cases},
                         {(u, v) for u in [1, 2, 4, 8] for v in [.5, 1, 2, 4]})
        midpoint = build_point(plan.grid_refinement_source, {'U': 1.5, 'V': .75})
        for case in [*plan.cases, midpoint]:
            self.assertEqual(case.request['solver'], spec['base_task']['solver'])
            self.assertNotIn('parameters', case.request)
            model = case.request['model_hamiltonian']['spec']
            self.assertTrue(all(site['U'] == case.variables['U'] for site in model['sites']))
            self.assertTrue(all(bond['V'] == case.variables['V'] for bond in model['bonds']))
            self.assertTrue(all(bond['effective_V'] == case.variables['V'] for bond in model['bonds']))
            self.assertTrue(all(bond['t'] == -1 for bond in model['bonds']))
            self.assertEqual(model['nelec'], before['base_model_spec']['nelec'])
        self.assertEqual(spec, before)

    def test_llm_uses_existing_single_repair_for_misplaced_coefficients(self):
        invalid, corrected = misplaced_spec(), scan_spec()
        post = Mock(side_effect=[
            {'choices': [{'message': {'parsed': invalid}}]},
            {'choices': [{'message': {'parsed': corrected}}]},
        ])
        with patch.dict('os.environ', {'PYSCF_AGENT_LLM_BASE_URL': 'https://parameter-repair.test/v1',
                                       'PYSCF_AGENT_LLM_MODEL': 'parameter-repair'}):
            result = build_study_spec_from_goal('Keep the U/V scan.', seed_spec=corrected, http_post=post)
        self.assertEqual(post.call_count, 2)
        repair = json.loads(post.call_args_list[1].args[1]['messages'][-1]['content'])
        self.assertIn('request_updates.parameters', repair['validation_error'])
        self.assertIn('set_global_parameter', repair['validation_error'])
        self.assertEqual(result['case_design'], corrected['case_design'])
        self.assertEqual(result['base_task'], corrected['base_task'])
        self.assertEqual(result['grid_refinement'], corrected['grid_refinement'])
        self.assertEqual(len(build_study_plan(result).cases), 16)

    def test_llm_does_not_accept_a_second_invalid_parameter_draft(self):
        seed = scan_spec()
        before = copy.deepcopy(seed)
        post = Mock(return_value={'choices': [{'message': {'parsed': misplaced_spec()}}]})
        with patch.dict('os.environ', {'PYSCF_AGENT_LLM_BASE_URL': 'https://parameter-reject.test/v1',
                                       'PYSCF_AGENT_LLM_MODEL': 'parameter-reject'}):
            with self.assertRaisesRegex(PlannerDraftError, 'after one contract-repair attempt'):
                build_study_spec_from_goal('Keep the U/V scan.', seed_spec=seed, http_post=post)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(seed, before)
