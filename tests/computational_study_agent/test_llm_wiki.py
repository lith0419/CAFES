from __future__ import annotations

import tempfile
import unittest.mock
from pathlib import Path

from pyscf_agent.backend.parsing import task_spec_from_partial
from pyscf_agent.backend.validation import _validate_strong_correlation_spec

from computational_study_agent.llm_planner import build_study_spec_from_goal
from tests.computational_study_agent.support import (
    StudyAgentTestCase,
    hubbard_dimer_spec,
    json_dumps,
    json_dumps_text,
    json_loads,
)
from computational_study_agent.web_api import handle_study_llm_draft_request
from computational_study_agent.wiki_retriever import build_wiki_evidence_pack, load_wiki_pages, retrieve_wiki_pages


class PlannerLlmAndWikiTests(StudyAgentTestCase):
    def test_gateway_unavailable_response_format_falls_back_and_returns_full_plan(self):
        from pyscf_agent.request_builder.constants import _STRUCTURED_OUTPUT_SUPPORT
        from pyscf_agent.request_builder.llm import LLMHTTPError

        seed = {
            'name': 'gateway-scan', 'objective': 'Scan U and V',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'fci'},
            'sweep': {'U': [2, 4, 8, 16], 'V': [0.5, 1, 1.5, 2]},
            'observables': ['energy'],
        }
        calls = []

        def post(_url, body, _headers, _timeout):
            calls.append(body)
            if 'response_format' in body:
                raise LLMHTTPError(400, json_dumps_text({'error': {
                    'message': 'This response_format type is unavailable now (request_id: test)',
                    'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error',
                }}).decode())
            return {'choices': [{'message': {'content': json_dumps_text(seed).decode()}}]}

        with unittest.mock.patch.dict('os.environ', {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://unavailable-format.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'test', 'PYSCF_AGENT_LLM_STRUCTURED_OUTPUT': 'auto',
        }), unittest.mock.patch.dict(_STRUCTURED_OUTPUT_SUPPORT, {}, clear=True), \
                unittest.mock.patch('computational_study_agent.llm_planner._default_http_post', side_effect=post):
            for _ in range(2):
                status, _, body = handle_study_llm_draft_request(json_dumps_text({
                    'goal': 'scan Hubbard U = [2,4,8,16] and V = [0.5, 1,1.5, 2]',
                    'study_spec': seed,
                }))
                self.assertEqual(status, 200, body)
                self.assertEqual(len(json_loads(body)['plan']['cases']), 16)
        self.assertEqual(len(calls), 3)
        self.assertIn('response_format', calls[0])
        for call in calls[1:]:
            self.assertNotIn('response_format', call)
            self.assertEqual(call['messages'], calls[0]['messages'])

    def test_planner_respects_explicit_structured_output_policy_over_cache(self):
        from pyscf_agent.request_builder.constants import _STRUCTURED_OUTPUT_SUPPORT
        from pyscf_agent.request_builder.llm import LLMHTTPError

        seed = {'name': 'test', 'objective': 'test', 'system_type': 'molecular',
                'base_task': {'atom': 'H 0 0 0; H 0 0 0.74', 'basis': 'sto-3g'},
                'observables': ['energy']}
        for policy, cached in [('false', True), ('true', False)]:
            post = unittest.mock.Mock(return_value={'choices': [{'message': {'parsed': seed}}]})
            if policy == 'true':
                post.side_effect = LLMHTTPError(400, 'unsupported response_format')
            with self.subTest(policy=policy), unittest.mock.patch.dict('os.environ', {
                'PYSCF_AGENT_LLM_BASE_URL': 'https://policy.test/v1',
                'PYSCF_AGENT_LLM_MODEL': 'test', 'PYSCF_AGENT_LLM_STRUCTURED_OUTPUT': policy,
            }), unittest.mock.patch.dict(_STRUCTURED_OUTPUT_SUPPORT,
                                         {'https://policy.test/v1|test': cached}, clear=True):
                if policy == 'true':
                    with self.assertRaises(LLMHTTPError):
                        build_study_spec_from_goal('test', http_post=post)
                else:
                    self.assertEqual(build_study_spec_from_goal('test', http_post=post)['name'], 'test')
                self.assertEqual(post.call_count, 1)
                self.assertEqual('response_format' in post.call_args.args[1], policy == 'true')

    def test_llm_planner_repairs_non_json_responses_without_losing_seed(self):
        seed = {
            'name': 'dmet-scan',
            'objective': 'Scan U and V with DMET',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': {'name': 'dmet', 'options': {
                'execution_mode': 'finite_graph', 'impurity_solver': 'fci',
            }}},
            'sweep': {'U': [1, 2, 3, 4], 'V': [0.1, 0.2, 0.3, 0.4]},
            'observables': ['energy'],
        }
        for content in ['Use finite_graph to preserve V.', '', None, '{"name":', '{"name": invalid}']:
            with self.subTest(content=content):
                calls = []

                def fake_post(_url, body, _headers, _timeout):
                    calls.append(body)
                    if len(calls) == 1:
                        return {'choices': [{'message': {'content': content}}]}
                    original = json_loads(body['messages'][1]['content'].encode())
                    self.assertEqual(original['seed_study_spec'], seed)
                    self.assertEqual(original['goal'], 'Keep all 16 cases and use finite_graph.')
                    self.assertEqual(body['messages'][-2]['content'], content or '')
                    return {'choices': [{'message': {'parsed': seed}}]}

                with unittest.mock.patch.dict('os.environ', {
                    'PYSCF_AGENT_LLM_BASE_URL': 'https://json-repair.test/v1',
                    'PYSCF_AGENT_LLM_MODEL': 'json-repair-model',
                }):
                    result = build_study_spec_from_goal(
                        'Keep all 16 cases and use finite_graph.',
                        seed_spec=seed, http_post=fake_post,
                    )
                self.assertEqual(len(calls), 2)
                self.assertEqual(result['base_model_spec'], seed['base_model_spec'])
                self.assertEqual(result['base_task'], seed['base_task'])
                self.assertEqual(result['sweep'], seed['sweep'])

    def test_llm_planner_stops_after_one_failed_json_or_contract_repair(self):
        incomplete_contract = {
            'schema': 'pyscf-agent.study-spec.v1', 'system_type': 'molecular',
        }
        pairs = [
            ('No JSON.', 'Still no JSON.'),
            (json_dumps(incomplete_contract).decode(), '{"name":'),
            ('No JSON.', json_dumps(incomplete_contract).decode()),
        ]
        for first, second in pairs:
            with self.subTest(first=first, second=second):
                post = unittest.mock.Mock(side_effect=[
                    {'choices': [{'message': {'content': first}}]},
                    {'choices': [{'message': {'content': second}}]},
                ])
                with unittest.mock.patch.dict('os.environ', {
                    'PYSCF_AGENT_LLM_BASE_URL': 'https://json-failure.test/v1',
                    'PYSCF_AGENT_LLM_MODEL': 'json-failure-model',
                }), self.assertRaisesRegex(ValueError, 'after one .*repair attempt'):
                    build_study_spec_from_goal('Scan H2.', http_post=post)
                self.assertEqual(post.call_count, 2)

    def test_llm_draft_does_not_turn_partial_json_into_a_default_single_case(self):
        seed = {
            'name': 'existing-dmet-scan',
            'objective': 'Keep the existing U/V scan',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': {'name': 'dmet', 'options': {
                'execution_mode': 'finite_graph', 'impurity_solver': 'fci',
            }}},
            'sweep': {'U': [1, 2, 3, 4], 'V': [0.1, 0.2, 0.3, 0.4]},
            'observables': ['energy'],
        }
        partials = [
            {},
            {'system_type': 'model_hamiltonian', 'base_task': seed['base_task']},
            {**seed, 'name': None},
            {key: value for key, value in seed.items() if key != 'sweep'},
            {**seed, 'sweep': {}, 'case_design': {'mode': 'cases', 'cases': []}},
        ]
        for partial in partials:
            with self.subTest(partial=partial):
                post = unittest.mock.Mock(side_effect=[
                    {'choices': [{'message': {'parsed': partial}}]},
                    {'choices': [{'message': {'parsed': seed}}]},
                ])
                with unittest.mock.patch.dict('os.environ', {
                    'PYSCF_AGENT_LLM_BASE_URL': 'https://preserve-scan.test/v1',
                    'PYSCF_AGENT_LLM_MODEL': 'preserve-scan-model',
                }), unittest.mock.patch(
                    'computational_study_agent.llm_planner._default_http_post', post,
                ):
                    status, _headers, body = handle_study_llm_draft_request(json_dumps({
                        'goal': 'Keep the scan and use finite_graph.', 'study_spec': seed,
                    }))
                result = json_loads(body)
                self.assertEqual(post.call_count, 2)
                self.assertEqual(status.value, 200, result)
                self.assertEqual(result['study_spec']['name'], seed['name'])
                self.assertEqual(result['study_spec']['sweep'], seed['sweep'])
                self.assertEqual(len(result['plan']['cases']), 16)

        # An intentional, explicitly represented single case remains supported.
        single = {**seed, 'sweep': {}, 'case_design': {
            'mode': 'cases', 'cases': [{'label': 'one selected point', 'variables': {}}],
        }}
        post = unittest.mock.Mock(return_value={'choices': [{'message': {'parsed': single}}]})
        with unittest.mock.patch.dict('os.environ', {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://preserve-scan.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'preserve-scan-model',
        }):
            result = build_study_spec_from_goal(
                'Replace the scan with one explicit point.', seed_spec=seed, http_post=post,
            )
        self.assertEqual(post.call_count, 1)
        self.assertEqual(result['case_design'], single['case_design'])

    def test_failed_llm_revision_does_not_replace_existing_scan_with_local_fallback(self):
        for scan in [
            {'sweep': {'U': [1, 2, 3, 4], 'V': [0.1, 0.2, 0.3, 0.4]}},
            {'case_design': {
                'mode': 'grid', 'variables': {'U': [1, 2, 3, 4]},
                'template': {'operations': [
                    {'op': 'set_site_parameter', 'site': 0, 'parameter': 'U', 'value': '$U'},
                ]},
            }},
        ]:
            with self.subTest(scan=scan), unittest.mock.patch(
                'computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
                side_effect=ValueError('LLM response does not contain a JSON object'),
            ):
                status, _headers, body = handle_study_llm_draft_request(json_dumps({
                    'goal': 'Keep the other points; set U = 4 for this case.',
                    'study_spec': {
                        'name': 'existing-model', 'objective': 'Keep the existing scan',
                        'system_type': 'model_hamiltonian',
                        'base_model_spec': hubbard_dimer_spec(),
                        'base_task': {'solver': 'fci'}, 'observables': ['energy'],
                        **scan,
                    },
                }))
            result = json_loads(body)
            self.assertEqual(status.value, 400)
            self.assertNotIn('study_spec', result)
            self.assertNotIn('plan', result)

    @staticmethod
    def _decorated_four_site_spec():
        spec = hubbard_dimer_spec()
        spec.update({
            'dimension': 2,
            'preset': 'honeycomb',
            'boundary': 'periodic',
            'nelec': [2, 2],
            'sites': [
                {'id': 0, 'x': 0, 'y': 0, 'epsilon': 0, 'U': 4, 'sublattice': 'a', 'basis_index': 0, 'cell_index': [0, 0]},
                {'id': 1, 'x': 1, 'y': 0, 'epsilon': 0, 'U': 4, 'sublattice': 'b', 'basis_index': 1, 'cell_index': [0, 0]},
                {'id': 2, 'x': 2, 'y': 0, 'epsilon': 0, 'U': 4, 'sublattice': 'a', 'basis_index': 0, 'cell_index': [1, 0]},
                {'id': 3, 'x': 3, 'y': 0, 'epsilon': 0, 'U': 4, 'sublattice': 'b', 'basis_index': 1, 'cell_index': [1, 0]},
            ],
            'bonds': [
                {'id': 0, 'source': 0, 'target': 1, 't': -1, 'V': 0},
                {'id': 1, 'source': 1, 'target': 2, 't': -1, 'V': 0},
                {'id': 2, 'source': 2, 'target': 3, 't': -1, 'V': 0},
                {'id': 3, 'source': 3, 'target': 0, 't': -1, 'V': 0, 'periodic': True},
            ],
        })
        return spec

    def test_llm_planner_uses_mock_response(self):
        def fake_post(_url, body, _headers, _timeout):
            content = body['messages'][1]['content']
            self.assertNotIn('capability_registry', content)
            self.assertIn('wiki_evidence', content)
            self.assertIn('Local PySCF Agent Wiki Rules', content)
            return {
                'choices': [{
                    'message': {
                        'content': json_dumps_text({
                            'name': 'mock-study',
                            'objective': 'compare_methods',
                            'system_type': 'molecular',
                            'base_task': {
                                'atom': 'H 0 0 0; H 0 0 0.74',
                                'basis': 'sto-3g',
                                'outputs': ['energy'],
                            },
                            'sweep': {'method': ['hf']},
                            'observables': ['energy'],
                        }).decode('utf-8'),
                    }
                }]
            }

        with unittest.mock.patch.dict('os.environ', {
            'PYSCF_AGENT_LLM_BASE_URL': 'http://localhost:11434/v1',
            'PYSCF_AGENT_LLM_MODEL': 'test-model',
        }):
            study_spec = build_study_spec_from_goal('compare H2 methods', http_post=fake_post)

        self.assertEqual(study_spec['name'], 'mock-study')
        self.assertEqual(study_spec['sweep']['method'], ['hf'])

    def test_planner_retrieves_wiki_from_the_study_without_exporting_registry(self):
        goal = 'Scan the Hubbard model with DMET in finite_graph mode.'
        seed = {
            'name': 'dmet-scan', 'objective': goal,
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': {'name': 'dmet', 'options': {
                'execution_mode': 'finite_graph', 'impurity_solver': 'fci',
            }}},
            'sweep': {'U': [2, 4]}, 'observables': ['energy'],
        }
        captured = {}

        def fake_post(_url, body, _headers, _timeout):
            captured.update(json_loads(body['messages'][1]['content'].encode()))
            return {'choices': [{'message': {'parsed': seed}}]}

        with unittest.mock.patch.dict('os.environ', {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://wiki-only.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'wiki-only-model',
        }), unittest.mock.patch(
            'computational_study_agent.llm_planner.build_wiki_evidence_pack',
            wraps=build_wiki_evidence_pack,
        ) as retrieve, unittest.mock.patch(
            'computational_study_agent.capabilities.StudyCapabilityView.as_dict',
            side_effect=AssertionError('Planner drafting must not export the registry'),
        ):
            result = build_study_spec_from_goal(goal, seed_spec=seed, http_post=fake_post)

        retrieve.assert_called_once()
        query = retrieve.call_args.args[0]
        self.assertIn(goal, query)
        self.assertIn('dmet', query)
        self.assertIn('finite_graph', query)
        self.assertNotIn('base_task', query)
        self.assertNotIn('options', query)
        self.assertNotIn('capability_registry', query)
        self.assertEqual(captured['seed_study_spec'], seed)
        self.assertEqual(captured['model_site_context']['site_count'], 2)
        self.assertIn('[Wiki Page:', captured['wiki_evidence'])
        self.assertIn('Model Hamiltonian Solver Support', captured['wiki_evidence'])
        self.assertNotIn('capability_registry', captured)
        self.assertNotIn('platform_catalog', captured)
        self.assertIn('base_task', captured['study_spec_schema']['required'])
        self.assertEqual(result['sweep'], seed['sweep'])

    def test_wiki_based_draft_still_rejects_an_unregistered_solver(self):
        candidate = {
            'name': 'unsupported-solver', 'objective': 'Scan the model',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'made-up-solver'},
            'sweep': {'U': [2, 4]}, 'observables': ['energy'],
        }
        with unittest.mock.patch.dict('os.environ', {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://wiki-validation.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'wiki-validation-model',
        }), unittest.mock.patch(
            'computational_study_agent.llm_planner._default_http_post',
            return_value={'choices': [{'message': {'parsed': candidate}}]},
        ):
            status, _headers, body = handle_study_llm_draft_request(json_dumps({
                'goal': 'Draft this scan.', 'study_spec': candidate,
            }))
        result = json_loads(body)
        self.assertEqual(status.value, 200, result)
        self.assertEqual(result['status'], 'needs_input')
        self.assertTrue(any(issue['code'] == 'unsupported_model_solver'
                            for issue in result['validation_issues']))
        self.assertNotIn('plan', result)

    def test_llm_planner_repairs_an_incomplete_study_contract_once(self):
        calls = []

        def fake_post(_url, body, _headers, _timeout):
            calls.append(body)
            if len(calls) == 1:
                response_payload = {
                    'schema': 'pyscf-agent.study-spec.v1',
                    'system_type': 'molecular',
                }
            else:
                repair_request = json_loads(
                    body['messages'][-1]['content'].encode('utf-8')
                )
                self.assertIn('missing required field', repair_request['validation_error'])
                response_payload = {
                    'schema': 'pyscf-agent.study-spec.v1',
                    'name': 'repaired-study',
                    'objective': 'scan a molecular coordinate',
                    'system_type': 'molecular',
                    'base_task': {
                        'atom': 'H 0 0 0; H 0 0 0.74',
                        'basis': 'sto-3g',
                        'method': 'hf',
                    },
                    'observables': ['energy'],
                }
            return {
                'choices': [{
                    'message': {
                        'content': json_dumps_text(response_payload).decode('utf-8'),
                    },
                }],
            }

        with unittest.mock.patch.dict('os.environ', {
            'PYSCF_AGENT_LLM_BASE_URL': 'http://localhost:11434/v1',
            'PYSCF_AGENT_LLM_MODEL': 'test-model',
        }):
            study_spec = build_study_spec_from_goal(
                'scan H2',
                http_post=fake_post,
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(study_spec['name'], 'repaired-study')

    def test_llm_planner_contract_repair_keeps_unstructured_fallback(self):
        calls = []

        def fake_post(_url, body, _headers, _timeout):
            calls.append(body)
            if len(calls) == 1:
                raise RuntimeError(
                    'LLM HTTP error 400: unsupported response_format json_schema'
                )
            response_payload = (
                {
                    'schema': 'pyscf-agent.study-spec.v1',
                    'system_type': 'molecular',
                }
                if len(calls) == 2
                else {
                    'schema': 'pyscf-agent.study-spec.v1',
                    'name': 'fallback-repaired-study',
                    'objective': 'scan a molecular coordinate',
                    'system_type': 'molecular',
                    'base_task': {
                        'atom': 'H 0 0 0; H 0 0 0.74',
                        'basis': 'sto-3g',
                        'method': 'hf',
                    },
                    'observables': ['energy'],
                }
            )
            return {
                'choices': [{
                    'message': {
                        'content': json_dumps_text(response_payload).decode('utf-8'),
                    },
                }],
            }

        with unittest.mock.patch.dict('os.environ', {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://fallback-repair.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'fallback-repair-model',
        }):
            study_spec = build_study_spec_from_goal(
                'scan H2',
                http_post=fake_post,
            )

        self.assertEqual(len(calls), 3)
        self.assertIn('response_format', calls[0])
        self.assertNotIn('response_format', calls[1])
        self.assertNotIn('response_format', calls[2])
        initial_context = json_loads(calls[0]['messages'][1]['content'].encode())
        for call in calls:
            context = json_loads(call['messages'][1]['content'].encode())
            self.assertEqual(context, initial_context)
            self.assertNotIn('capability_registry', context)
            self.assertNotIn('platform_catalog', context)
            self.assertIn('name', context['study_spec_schema']['required'])
            self.assertIn('[Wiki Page:', context['wiki_evidence'])
        self.assertEqual(study_spec['name'], 'fallback-repaired-study')

    def test_llm_planner_receives_builder_declared_site_groups(self):
        captured = {}

        def fake_post(_url, body, _headers, _timeout):
            captured.update(json_loads(body['messages'][1]['content'].encode('utf-8')))
            return {
                'choices': [{
                    'message': {
                        'content': json_dumps_text({
                            'name': 'decorated-model-study',
                            'objective': 'scan one site group',
                            'system_type': 'model_hamiltonian',
                            'base_task': {'solver': 'fci'},
                            'base_model_spec': self._decorated_four_site_spec(),
                            'sweep': {'U': [1]},
                            'observables': ['energy'],
                        }).decode('utf-8'),
                    },
                }],
            }

        with unittest.mock.patch.dict('os.environ', {
            'PYSCF_AGENT_LLM_BASE_URL': 'http://localhost:11434/v1',
            'PYSCF_AGENT_LLM_MODEL': 'test-model',
        }):
            build_study_spec_from_goal(
                'scan equivalent sites',
                seed_spec={
                    'system_type': 'model_hamiltonian',
                    'base_model_spec': self._decorated_four_site_spec(),
                    'base_task': {'solver': 'fci'},
                    'observables': ['energy'],
                },
                http_post=fake_post,
            )

        context = captured['model_site_context']
        self.assertEqual(context['builder_declared_groups']['sublattice'], {
            'a': [0, 2],
            'b': [1, 3],
        })
        self.assertEqual(context['builder_declared_groups']['basis_index'], {
            '0': [0, 2],
            '1': [1, 3],
        })

    def test_grouped_site_updates_use_the_shared_native_operation_contract(self):
        with unittest.mock.patch(
            'computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
            return_value={
                'study_spec': {
                    'name': 'honeycomb-sublattice-u-scan',
                    'objective': 'keep A at U=4 and scan B',
                    'system_type': 'model_hamiltonian',
                    'base_task': {'solver': 'fci'},
                    'base_model_spec': self._decorated_four_site_spec(),
                    'case_design': {
                        'mode': 'grid',
                        'variables': {'U_B': [1, 2, 4, 8, 16]},
                        'template': {
                            'operations': [
                                {
                                    'op': 'set_site_parameter',
                                    'sites': [0, 2],
                                    'parameter': 'U',
                                    'value': 4,
                                },
                                {
                                    'op': 'set_site_parameter',
                                    'sites': [1, 3],
                                    'parameter': 'U',
                                    'value': '$U_B',
                                },
                            ],
                        },
                    },
                    'observables': ['energy'],
                },
                'wiki_evidence': {'pages': []},
            },
        ):
            status, _headers, body = handle_study_llm_draft_request(json_dumps({
                'goal': 'keep A at U=4 and scan B',
                'locale': 'en',
            }))

        payload = json_loads(body)
        self.assertEqual(status.value, 200)
        operations = payload['study_spec']['case_design']['template']['operations']
        self.assertEqual([operation['sites'] for operation in operations], [[0, 2], [1, 3]])
        self.assertTrue(all('site' not in operation for operation in operations))
        self.assertEqual(len(payload['plan']['cases']), 5)
        first_sites = payload['plan']['cases'][0]['model_spec']['sites']
        last_sites = payload['plan']['cases'][-1]['model_spec']['sites']
        self.assertEqual([site['U'] for site in first_sites], [4, 1, 4, 1])
        self.assertEqual([site['U'] for site in last_sites], [4, 16, 4, 16])


    def test_invalid_resolved_site_group_preserves_the_draft_for_review(self):
        # A reference may look scalar in the template but resolve to a group.
        # Validate its value in the shared operation engine after substitution.
        draft = {
            'name': 'site-group-scan', 'objective': 'Scan selected sites',
            'system_type': 'model_hamiltonian',
            'base_model_spec': self._decorated_four_site_spec(),
            'base_task': {'solver': 'fci'}, 'observables': ['energy'],
            'case_design': {
                'mode': 'grid', 'variables': {'target': [0, [1, 3]]},
                'template': {'operations': [
                    {'op': 'set_site_parameter', 'site': '$target', 'parameter': 'U', 'value': 8},
                ]},
            },
        }
        with unittest.mock.patch(
            'computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
            return_value={'study_spec': draft, 'wiki_evidence': {'pages': []}},
        ):
            status, _, body = handle_study_llm_draft_request(json_dumps({
                'goal': 'Update selected sites',
            }))
        result = json_loads(body)
        self.assertEqual(status.value, 200)
        self.assertEqual(result['status'], 'needs_input')
        self.assertNotIn('plan', result)
        self.assertNotIn('error', result)
        self.assertEqual(result['study_spec']['case_design'], draft['case_design'])
        self.assertEqual(result['validation_issues'][0]['code'], 'study_plan_build_failed')
        self.assertIn('cases[1].operations[0]: site', result['validation_issues'][0]['message'])
        self.assertIn('use sites', result['validation_issues'][0]['message'])

        operation = draft['case_design']['template']['operations'][0]
        operation['sites'] = operation.pop('site')
        with unittest.mock.patch(
            'computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
            return_value={'study_spec': draft, 'wiki_evidence': {'pages': []}},
        ):
            status, _, body = handle_study_llm_draft_request(json_dumps({'goal': 'Update selected sites'}))
        result = json_loads(body)
        self.assertEqual(status.value, 200, result)
        self.assertFalse(result['validation_issues'])
        cases = result['plan']['cases']
        self.assertEqual(len(cases), 2)
        self.assertEqual([s['U'] for s in cases[1]['model_spec']['sites']], [4, 8, 4, 8])

    def test_study_llm_draft_api_returns_plan_with_mocked_planner(self):
        with unittest.mock.patch(
            'computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
            return_value={
                'study_spec': {
                    'name': 'mock-study',
                    'objective': 'compare_methods',
                    'system_type': 'molecular',
                    'base_task': {
                        'atom': 'H 0 0 0; H 0 0 0.74',
                        'basis': 'sto-3g',
                        'outputs': ['energy'],
                    },
                    'sweep': {'method': ['hf']},
                    'observables': ['energy'],
                },
                'wiki_evidence': {
                    'pages': [{'title': 'Molecular Method Selection Rules', 'slug': 'molecular-method-selection-rules'}],
                },
            },
        ):
            status, _headers, body = handle_study_llm_draft_request(json_dumps({
                'goal': 'compare H2 methods',
                'locale': 'en',
            }))

        payload = json_loads(body)
        self.assertEqual(status.value, 200)
        self.assertEqual(payload['study_spec']['name'], 'mock-study')
        self.assertEqual(len(payload['plan']['cases']), 1)
        self.assertEqual(payload['wiki_evidence']['pages'][0]['title'], 'Molecular Method Selection Rules')

    def test_study_llm_draft_api_expands_numeric_grid_ranges(self):
        with unittest.mock.patch(
            'computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
            return_value={
                'study_spec': {
                    'name': 'n2-factor-scan',
                    'objective': 'scan N2 from 0.2x to 3.0x of its reference bond length',
                    'system_type': 'molecular',
                    'base_task': {
                        'atom': 'N 0 0 0; N 0 0 1.09768',
                        'basis': 'sto-3g',
                        'method': 'mp2',
                    },
                    'case_design': {
                        'mode': 'grid',
                        'variables': {
                            'bond_factor': {'start': 0.2, 'stop': 3.0, 'step': 0.2},
                        },
                        'template': {
                            'request_updates': {
                                'atom': 'N 0 0 0; N 0 0 $(bond_factor * 1.09768)',
                            },
                        },
                    },
                    'observables': ['energy'],
                },
                'wiki_evidence': {'pages': []},
            },
        ):
            status, _headers, body = handle_study_llm_draft_request(json_dumps({
                'goal': 'scan N2 from 0.2x to 3.0x in 0.2x steps',
                'locale': 'en',
            }))

        payload = json_loads(body)
        self.assertEqual(status.value, 200)
        self.assertEqual(payload['study_spec']['case_design']['variables']['bond_factor'], [
            0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0,
        ])
        self.assertEqual(len(payload['plan']['cases']), 15)
        self.assertEqual(payload['plan']['cases'][-1]['request']['atom'], 'N 0 0 0; N 0 0 3.29304')

    def test_regression_planner_molecular_geometry_aliases_expand_n2_scan(self):
        with unittest.mock.patch(
            'computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
            return_value={
                'study_spec': {
                    'name': 'n2-bond-scan',
                    'objective': 'scan N2 from 0.8x to 4.0x of its reference bond length',
                    'system_type': 'molecular',
                    'base_task': {
                        'geometry': 'N 0 0 0; N 0 0 $(bond_factor * 1.09768)',
                        'basis': 'sto-3g',
                        'method': 'hf',
                        'reference': 'rhf',
                        'charge': 0,
                        'spin': 0,
                    },
                    'case_design': {
                        'mode': 'grid',
                        'variables': {
                            'bond_factor': {'start': 0.8, 'stop': 4.0, 'step': 0.1},
                        },
                        'template': {'request_updates': {}},
                    },
                    'observables': ['energy', 'homo_lumo', 'dipole'],
                },
                'wiki_evidence': {'pages': []},
            },
        ):
            status, _headers, body = handle_study_llm_draft_request(json_dumps({
                'goal': 'calculate N2 bond stretching from 0.8x to 4.0x standard bond length, with a step of 0.1x',
                'locale': 'en',
            }))

        payload = json_loads(body)
        self.assertEqual(status.value, 200)
        self.assertEqual(payload['study_spec']['base_task']['atom'], 'N 0 0 0; N 0 0 $(bond_factor * 1.09768)')
        self.assertNotIn('geometry', payload['study_spec']['base_task'])
        self.assertEqual(len(payload['plan']['cases']), 33)
        self.assertEqual(payload['plan']['cases'][0]['request']['atom'], 'N 0 0 0; N 0 0 0.878144')
        self.assertEqual(payload['plan']['cases'][-1]['request']['atom'], 'N 0 0 0; N 0 0 4.39072')

    def test_regression_planner_flattens_structured_n2_atom_rows(self):
        with unittest.mock.patch(
            'computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
            return_value={
                'study_spec': {
                    'name': 'n2-structured-atom-scan',
                    'objective': 'scan N2 with structured atom rows',
                    'system_type': 'molecular',
                    'base_task': {
                        'atom': [['N', [0, 0, 0]], ['N', [0, 0, 1.09768]]],
                        'basis': 'sto-3g',
                        'method': 'mp2',
                    },
                    'case_design': {
                        'mode': 'grid',
                        'variables': {'bond_factor': [0.8, 4.0]},
                        'template': {
                            'request_updates': {
                                'atom': [
                                    ['N', [0, 0, 0]],
                                    ['N', [0, 0, '$(bond_factor * 1.09768)']],
                                ],
                            },
                        },
                    },
                    'observables': ['energy'],
                },
                'wiki_evidence': {'pages': []},
            },
        ):
            status, _headers, body = handle_study_llm_draft_request(json_dumps({
                'goal': 'scan N2 from 0.8x to 4.0x',
                'locale': 'en',
            }))

        payload = json_loads(body)
        self.assertEqual(status.value, 200)
        self.assertEqual(payload['study_spec']['base_task']['atom'], 'N 0 0 0; N 0 0 1.09768')
        self.assertEqual([case['request']['atom'] for case in payload['plan']['cases']], [
            'N 0 0 0; N 0 0 0.878144',
            'N 0 0 0; N 0 0 4.39072',
        ])

    def test_study_llm_draft_api_inherits_seed_model_context(self):
        with unittest.mock.patch(
            'computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
            return_value={
                'study_spec': {
                    'name': 'model-parameter-sweep',
                    'objective': 'sweep U and t',
                    'system_type': 'model_hamiltonian',
                    'base_task': {'solver': 'fci'},
                    'sweep': {
                        'U': [1, 2, 3, 4],
                        't': [-2, -1, 0, 1, 2],
                    },
                    'observables': ['energy'],
                },
                'wiki_evidence': {'pages': []},
            },
        ):
            status, _headers, body = handle_study_llm_draft_request(json_dumps({
                'goal': 'sweep U from 1 to 4, sweep t from -2 to 2, steps are 1',
                'locale': 'en',
                'study_spec': {
                    'name': 'seed-model',
                    'objective': 'current model',
                    'system_type': 'model_hamiltonian',
                    'base_model_spec': hubbard_dimer_spec(),
                    'base_task': {'solver': 'fci'},
                    'observables': ['energy'],
                },
            }))

        payload = json_loads(body)
        self.assertEqual(status.value, 200)
        self.assertNotEqual(payload.get('status'), 'needs_input')
        self.assertIn('base_model_spec', payload['study_spec'])
        self.assertEqual(len(payload['plan']['cases']), 20)
        self.assertEqual(payload['plan']['cases'][0]['variables']['U'], 1)
        self.assertEqual(payload['plan']['cases'][0]['variables']['t'], -2)

    def test_study_llm_draft_api_asks_for_model_solver_when_missing(self):
        with unittest.mock.patch(
            'computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
            return_value={
                'study_spec': {
                    'name': 'model-parameter-sweep',
                    'objective': 'sweep U',
                    'system_type': 'model_hamiltonian',
                    'sweep': {'U': [1, 2]},
                    'observables': ['energy'],
                },
                'wiki_evidence': {'pages': []},
            },
        ):
            status, _headers, body = handle_study_llm_draft_request(json_dumps({
                'goal': 'sweep U = 1, 2',
                'locale': 'en',
                'study_spec': {
                    'name': 'seed-model',
                    'objective': 'current model',
                    'system_type': 'model_hamiltonian',
                    'base_model_spec': hubbard_dimer_spec(),
                    'base_task': {},
                    'observables': ['energy'],
                },
            }))

        payload = json_loads(body)
        self.assertEqual(status.value, 200)
        self.assertEqual(payload['status'], 'needs_input')
        self.assertIn('base_model_spec', payload['study_spec'])
        self.assertEqual(payload['validation_issues'][0]['code'], 'missing_model_solver')
        self.assertIn('fci', payload['message'])
        self.assertIn('mp2', payload['message'])

    def test_study_llm_draft_api_preserves_structured_solver_confirmation(self):
        seed_spec = {
            'name': 'seed-model',
            'objective': 'current model',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {},
            'observables': ['energy'],
        }
        for instruction, include_system_type in (
            ('use FCI', True),
            ('exact diagonalization', False),
        ):
            with self.subTest(instruction=instruction), unittest.mock.patch(
                'computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
                return_value={
                    'study_spec': {
                        'name': 'model-parameter-sweep',
                        'objective': 'sweep U',
                        'base_task': {'solver': 'fci'},
                        **({'system_type': 'model_hamiltonian'} if include_system_type else {}),
                        'sweep': {'U': [4, 8]},
                        'observables': ['energy'],
                    },
                    'wiki_evidence': {'pages': []},
                },
            ):
                status, _headers, body = handle_study_llm_draft_request(json_dumps({
                    'goal': instruction,
                    'locale': 'en',
                    'study_spec': seed_spec,
                }))

            payload = json_loads(body)
            self.assertEqual(status.value, 200)
            self.assertNotEqual(payload.get('status'), 'needs_input')
            self.assertEqual(payload['study_spec']['base_task']['solver'], 'fci')
            self.assertEqual(payload['plan']['cases'][0]['request']['solver'], 'fci')


    def test_study_llm_draft_rejects_operations_without_parameter_identity(self):
        with unittest.mock.patch(
            'computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
            return_value={
                'study_spec': {
                    'name': 'hubbard-u-t-scan',
                    'objective': 'scan U and t',
                    'base_task': {'solver': 'fci'},
                    'system_type': 'model_hamiltonian',
                    'case_design': {
                        'mode': 'grid',
                        'variables': {
                            'U_value': [4, 8, 12, 16, 20],
                            't_value': [1.0, 0.5, 0.2],
                        },
                        'template': {
                            'operations': [
                                {'op': 'set_global_parameter', 'value': '$U_value'},
                                {'op': 'set_global_parameter', 'value': '$t_value'},
                            ],
                        },
                    },
                    'observables': ['energy'],
                },
                'wiki_evidence': {'pages': []},
            },
        ):
            status, _headers, body = handle_study_llm_draft_request(json_dumps({
                'goal': 'Scan Hubbard U from 4 to 20 in steps of 4, t=[1.0, 0.5, 0.2], use exact diagonalization',
                'locale': 'en',
                'study_spec': {
                    'name': 'seed-model',
                    'objective': 'current Hubbard model',
                    'system_type': 'model_hamiltonian',
                    'base_model_spec': hubbard_dimer_spec(),
                    'base_task': {},
                    'observables': ['energy'],
                },
            }))

        payload = json_loads(body)
        self.assertEqual(status.value, 200)
        self.assertNotIn('error', payload)
        self.assertEqual(payload['status'], 'needs_input')
        self.assertTrue(any(issue['code'] == 'missing_operation_parameter' for issue in payload['validation_issues']))
        self.assertNotIn('plan', payload)

    def test_study_llm_draft_api_does_not_use_adaptive_strategy_as_model_solver(self):
        with unittest.mock.patch(
            'computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
            return_value={
                'study_spec': {
                    'name': 'adaptive-model-parameter-sweep',
                    'objective': 'sweep U adaptively',
                    'system_type': 'model_hamiltonian',
                    'sweep': {'U': [0, 4, 8]},
                    'observables': ['energy'],
                },
                'wiki_evidence': {'pages': []},
            },
        ):
            status, _headers, body = handle_study_llm_draft_request(json_dumps({
                'goal': 'sweep U adaptively',
                'locale': 'en',
                'adaptive': {'initial_scan_strategy': 'mp2'},
                'study_spec': {
                    'name': 'seed-model',
                    'objective': 'current model',
                    'system_type': 'model_hamiltonian',
                    'base_model_spec': hubbard_dimer_spec(),
                    'base_task': {},
                    'observables': ['energy'],
                },
            }))

        payload = json_loads(body)
        self.assertEqual(status.value, 200)
        self.assertEqual(payload.get('status'), 'needs_input')
        self.assertEqual(payload['study_spec']['study_mode'], 'static')
        self.assertEqual(payload['validation_issues'][0]['code'], 'missing_model_solver')

    def test_study_llm_draft_api_normalizes_model_adaptive_context_to_static(self):
        with unittest.mock.patch(
            'computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
            return_value={
                'study_spec': {
                    'name': 'adaptive-model-parameter-sweep',
                    'objective': 'sweep U adaptively',
                    'system_type': 'model_hamiltonian',
                    'sweep': {'U': [0, 4]},
                    'observables': ['energy'],
                },
                'wiki_evidence': {'pages': []},
            },
        ):
            status, _headers, body = handle_study_llm_draft_request(json_dumps({
                'goal': 'sweep U adaptively',
                'locale': 'en',
                'study_mode': 'adaptive',
                'adaptive': {},
                'study_spec': {
                    'name': 'seed-model',
                    'objective': 'current model',
                    'system_type': 'model_hamiltonian',
                    'base_model_spec': hubbard_dimer_spec(),
                    'base_task': {},
                    'observables': ['energy'],
                },
            }))

        payload = json_loads(body)
        self.assertEqual(status.value, 200)
        self.assertEqual(payload.get('status'), 'needs_input')
        self.assertEqual(payload['study_spec']['study_mode'], 'static')
        self.assertEqual(payload['validation_issues'][0]['code'], 'missing_model_solver')

    def test_study_llm_draft_api_does_not_treat_model_adaptive_mode_as_solver(self):
        with unittest.mock.patch(
            'computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
            return_value={
                'study_spec': {
                    'name': 'adaptive-model-parameter-sweep',
                    'objective': 'sweep U adaptively',
                    'system_type': 'model_hamiltonian',
                    'sweep': {'U': [0, 4]},
                    'observables': ['energy'],
                },
                'wiki_evidence': {'pages': []},
            },
        ):
            status, _headers, body = handle_study_llm_draft_request(json_dumps({
                'goal': 'sweep U adaptively',
                'locale': 'en',
                'study_mode': 'adaptive',
                'study_spec': {
                    'name': 'seed-model',
                    'objective': 'current model',
                    'system_type': 'model_hamiltonian',
                    'base_model_spec': hubbard_dimer_spec(),
                    'base_task': {},
                    'observables': ['energy'],
                },
            }))

        payload = json_loads(body)
        self.assertEqual(status.value, 200)
        self.assertEqual(payload.get('status'), 'needs_input')
        self.assertEqual(payload['study_spec']['study_mode'], 'static')
        self.assertEqual(payload['validation_issues'][0]['code'], 'missing_model_solver')

    def test_study_llm_draft_api_reports_timeout_without_creating_a_replacement(self):
        with unittest.mock.patch(
            'computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
            side_effect=TimeoutError('The read operation timed out'),
        ):
            status, _headers, body = handle_study_llm_draft_request(json_dumps({
                'goal': 'sweep U = 1, 2',
                'locale': 'en',
                'study_spec': {
                    'name': 'seed-model',
                    'objective': 'current model',
                    'system_type': 'model_hamiltonian',
                    'base_model_spec': hubbard_dimer_spec(),
                    'base_task': {'solver': 'fci'},
                    'observables': ['energy'],
                },
            }))

        payload = json_loads(body)
        self.assertEqual(status.value, 503)
        self.assertIn('timed out', payload['error'])
        self.assertNotIn('study_spec', payload)
        self.assertNotIn('plan', payload)

    def test_study_llm_draft_timeout_is_not_converted_into_a_solver_choice(self):
        with unittest.mock.patch(
            'computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
            side_effect=TimeoutError('The read operation timed out'),
        ):
            status, _headers, body = handle_study_llm_draft_request(json_dumps({
                'goal': 'sweep U = 1, 2',
                'locale': 'en',
                'study_spec': {
                    'name': 'seed-model',
                    'objective': 'current model',
                    'system_type': 'model_hamiltonian',
                    'base_model_spec': hubbard_dimer_spec(),
                    'base_task': {},
                    'observables': ['energy'],
                },
            }))

        payload = json_loads(body)
        self.assertEqual(status.value, 503)
        self.assertIn('timed out', payload['error'])
        self.assertNotIn('study_spec', payload)
        self.assertNotIn('plan', payload)

    def test_study_llm_draft_api_normalizes_total_electron_operations(self):
        with unittest.mock.patch(
            'computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
            return_value={
                'study_spec': {
                    'name': 'mock-electron-scan',
                    'objective': 'scan total electrons',
                    'system_type': 'model_hamiltonian',
                    'base_model_spec': hubbard_dimer_spec(),
                    'base_task': {'solver': 'fci'},
                    'case_design': {
                        'mode': 'grid',
                        'variables': {'N': [1, 2]},
                        'template': {
                            'operations': [
                                {'op': 'change_nelec', 'total_electrons': '$N'},
                            ],
                        },
                    },
                    'observables': ['energy'],
                },
                'wiki_evidence': {'pages': []},
            },
        ):
            status, _headers, body = handle_study_llm_draft_request(json_dumps({
                'goal': 'scan total electrons from 1 to 2',
                'locale': 'en',
            }))

        payload = json_loads(body)
        self.assertEqual(status.value, 200)
        self.assertEqual(payload['plan']['cases'][0]['model_spec']['nelec'], [1, 0])
        self.assertEqual(payload['plan']['cases'][1]['model_spec']['nelec'], [1, 1])

    def test_study_llm_draft_api_returns_needs_input_for_invalid_draft(self):
        with unittest.mock.patch(
            'computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
            return_value={
                'study_spec': {
                    'name': 'bad-defect-draft',
                    'objective': 'scan defect position',
                    'system_type': 'model_hamiltonian',
                    'base_model_spec': hubbard_dimer_spec(),
                    'base_task': {'solver': 'fci'},
                    'case_design': {
                        'mode': 'grid',
                        'variables': {'defect_site': [0, 1]},
                        'template': {
                            'operations': [
                                {'op': 'add_site_defect', 'epsilon_shift': -0.5},
                            ],
                        },
                    },
                    'observables': ['energy'],
                },
                'wiki_evidence': {'pages': []},
            },
        ):
            status, _headers, body = handle_study_llm_draft_request(json_dumps({
                'goal': 'scan defect positions',
                'locale': 'en',
            }))

        payload = json_loads(body)
        self.assertEqual(status.value, 200)
        self.assertEqual(payload['status'], 'needs_input')
        self.assertTrue(any(item['code'] == 'missing_site_selector' for item in payload['validation_issues']))
        self.assertIn('study_spec', payload)
        self.assertNotIn('plan', payload)

    def test_direct_casscf_draft_prepares_active_space_probe_instead_of_fake_review(self):
        with unittest.mock.patch(
            'computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
            return_value={
                'study_spec': {
                    'name': 'h2-casscf-recalculation',
                    'objective': 'recalculate H2 with CASSCF',
                    'system_type': 'molecular',
                    'base_task': {
                        'atom': 'H 0 0 0; H 0 0 0.74',
                        'basis': 'sto-3g',
                        'method': 'casscf',
                        'restricted': True,
                    },
                    'observables': ['energy'],
                },
                'wiki_evidence': {'pages': []},
            },
        ):
            status, _headers, body = handle_study_llm_draft_request(json_dumps({
                'goal': 'recalculate H2 with CASSCF',
                'locale': 'en',
            }))

        payload = json_loads(body)
        self.assertEqual(status.value, 200)
        self.assertEqual(payload['status'], 'active_space_probe')
        self.assertTrue(payload['active_space_probe_plan']['cases'])
        request = payload['active_space_probe_plan']['cases'][0]['request']
        self.assertEqual(request['method'], 'hf')
        probe_config = request['workflow']['module_config']['molecular.active_space_probe']
        self.assertEqual(probe_config['requested_strategy'], 'auto')
        self.assertEqual(probe_config['refinement_method'], 'mp2')
        self.assertFalse(request['active_space']['approved'])
        self.assertIsNone(request['active_space']['ncas'])
        self.assertIsNone(request['active_space']['nelecas'])
        self.assertEqual(request['active_space']['target_method'], 'casscf')
        self.assertFalse(payload['validation_issues'])
        self.assertNotIn('report', payload)

    def test_dmrg_casscf_bond_scan_probes_every_case_before_shared_review(self):
        bond_lengths = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2]
        llm_spec = {
            'name': 'chromium-dimer-dmrg-casscf-scan',
            'objective': 'Scan the chromium dimer with DMRG-CASSCF',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'Cr 0 0 0; Cr $bond_length 0 0',
                'basis': 'def2-tzvp',
                'charge': 0,
                'spin': 0,
                'method': 'casscf',
                'solver': {'name': 'block2_dmrg'},
                'active_space': {
                    'enabled': True,
                    'selection_method': 'manual',
                    'ncas': 12,
                    'nelecas': 12,
                    'approved': False,
                },
            },
            'case_design': {
                'mode': 'grid',
                'variables': {'bond_length': bond_lengths},
                'template': {
                    'request_updates': {
                        'method': 'casscf',
                        'solver': {'name': 'block2_dmrg'},
                    },
                },
            },
            'observables': ['energy'],
        }
        with unittest.mock.patch(
            'computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
            return_value={
                'study_spec': llm_spec,
                'wiki_evidence': {'pages': []},
            },
        ):
            status, _headers, body = handle_study_llm_draft_request(json_dumps({
                'goal': (
                    'Scan chromium dimer from 1.2 Å to 3.2 Å, with a bond length '
                    'of 0.2 Å, use DMRG-CASSCF for all tasks, and use def2-tzvp'
                ),
                'locale': 'en',
            }))

        payload = json_loads(body)
        self.assertEqual(status.value, 200)
        self.assertEqual(payload['status'], 'active_space_probe')
        plan = payload['active_space_probe_plan']
        self.assertEqual(len(plan['cases']), len(bond_lengths))
        self.assertEqual(
            [case['variables']['bond_length'] for case in plan['cases']],
            bond_lengths,
        )
        self.assertTrue(all(case['request']['method'] == 'hf' for case in plan['cases']))
        self.assertTrue(all('solver' not in case['request'] for case in plan['cases']))
        self.assertTrue(all(
            case['request']['active_space']['target_solver'] == 'block2_dmrg'
            for case in plan['cases']
        ))
        self.assertTrue(all(
            case['request']['workflow']['modules'][0] == 'molecular.active_space_probe'
            for case in plan['cases']
        ))
        self.assertTrue(all(
            not _validate_strong_correlation_spec(
                task_spec_from_partial(case['request'])
            )
            for case in plan['cases']
        ))

    def test_incomplete_case_revision_is_rejected_without_rebuilding_seed(self):
        seed_spec = {
            'name': 'n2-bond-scan',
            'objective': 'Scan the N2 bond curve',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'N 0 0 0; N 0 0 1.09768',
                'basis': 'sto-3g',
                'method': 'mp2',
                'restricted': False,
            },
            'case_design': {
                'mode': 'grid',
                'variables': {'bond_factor': [1.0, 1.6, 2.0]},
                'template': {
                    'request_updates': {
                        'atom': 'N 0 0 0; N 0 0 $(bond_factor * 1.09768)',
                    },
                },
            },
            'observables': ['energy', 'homo_lumo'],
        }
        with unittest.mock.patch(
            'computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
            return_value={
                'study_spec': {
                    'name': 'n2-method-revision',
                    'objective': 'change one point to CASSCF',
                    'system_type': 'molecular',
                    'base_task': {'method': 'casscf'},
                    'case_design': {'mode': 'cases', 'cases': []},
                    'observables': ['energy'],
                },
                'wiki_evidence': {'pages': []},
            },
        ):
            status, _headers, body = handle_study_llm_draft_request(json_dumps({
                'goal': 'You: scan N2 bond stretching\nPlanner Agent: plan ready\nYou: change the method of 1.6x to casscf',
                'study_spec': seed_spec,
                'locale': 'en',
            }))

        payload = json_loads(body)
        self.assertEqual(status.value, 200)
        self.assertEqual(payload['status'], 'needs_input')
        self.assertEqual(payload['study_spec']['case_design'], {'mode': 'cases', 'cases': []})
        self.assertNotIn('plan', payload)

    def test_wiki_retriever_finds_electron_count_rules(self):
        hits = retrieve_wiki_pages('scan total electrons from 1 to 15 and normalize nelec')
        titles = [hit.page.title for hit in hits[:4]]

        self.assertIn('Model Hamiltonian Electron Count Contract', titles)
        self.assertTrue(any('Backend Normalization for Electron Counts' == title for title in titles))

    def test_wiki_evidence_pack_is_prompt_ready(self):
        pack = build_wiki_evidence_pack('CASSCF active space density fitting roadmap support')
        titles = [page['title'] for page in pack['pages']]

        self.assertIn('Strong Correlation Roadmap Rules', titles)
        self.assertTrue(all(page.get('content') for page in pack['pages']))

    def test_packaged_wiki_is_used_outside_source_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            with unittest.mock.patch(
                'computational_study_agent.wiki_retriever._repo_root',
                return_value=Path(directory),
            ):
                pages = load_wiki_pages()

        self.assertTrue(pages)
        self.assertIn('Task Specification Contract', {page.title for page in pages})

    def test_runtime_wiki_does_not_advertise_removed_cisd_method(self):
        pages = {page.title: page.body for page in load_wiki_pages()}

        self.assertNotIn('`cisd`:', pages['Molecular Method Selection Rules'])
        self.assertNotIn('`cisd`,', pages['Task Specification Contract'])

    def test_runtime_wiki_keeps_block2_nested_inside_dmet(self):
        pages = {page.title: page.body for page in load_wiki_pages()}
        solver_support = pages['Model Hamiltonian Solver Support']

        self.assertIn('`dmet`: optional self-consistent libDMET', solver_support)
        self.assertIn(
            '`solver.options.impurity_solver` selects `fci`, `ccsd`, or `block2_dmrg`',
            solver_support,
        )
        self.assertIn('keep the top-level solver as `dmet`', solver_support)
