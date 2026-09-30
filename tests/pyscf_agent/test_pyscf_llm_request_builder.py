from __future__ import annotations

import json
import os
import unittest
from unittest import mock

import pyscf_agent.request_builder as llm_request_builder
from pyscf_agent.request_builder import (
    build_model_hamiltonian_operations,
    build_prepared_request,
)


def _fake_chat_response(content):
    return {
        'choices': [
            {
                'message': {
                    'content': json.dumps(content, ensure_ascii=False),
                }
            }
        ]
    }


class LLMRequestBuilderTests(unittest.TestCase):
    def setUp(self):
        llm_request_builder._STRUCTURED_OUTPUT_SUPPORT.clear()

    def test_json_extraction_handles_braces_and_escapes_inside_strings(self):
        from pyscf_agent.request_builder.llm import _extract_json_object

        for note in ['literal }', 'literal {', 'quoted "} with a \\ path', '${U}']:
            with self.subTest(note=note):
                payload = {'notes': note, 'nested': {'value': 4}}
                text = 'Here is the draft:\n```json\n' + json.dumps(payload) + '\n```'
                self.assertEqual(_extract_json_object(text), payload)

    def test_json_extraction_rejects_invalid_outer_object(self):
        from pyscf_agent.request_builder.llm import _extract_json_object

        for text in ['No draft yet.', '', '{"name":', '{bad: {"name": "nested"}}']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                _extract_json_object(text)

    def test_llm_endpoint_accepts_base_or_full_endpoint(self):
        from pyscf_agent.request_builder.llm import _llm_endpoint

        self.assertEqual(
            _llm_endpoint('https://example.test/v1'),
            'https://example.test/v1/chat/completions',
        )
        self.assertEqual(
            _llm_endpoint('https://example.test/v1/chat/completions'),
            'https://example.test/v1/chat/completions',
        )
        self.assertEqual(
            _llm_endpoint('https://example.test/v1/responses'),
            'https://example.test/v1/chat/completions',
        )

    def test_model_operation_builder_uses_registered_site_operation_contract(self):
        captured = {}

        def fake_http_post(_url, body, _headers, _timeout):
            captured.update(body)
            return _fake_chat_response({
                'operations': [{
                    'op': 'set_site_parameter',
                    'site': 1,
                    'parameter': 'epsilon',
                    'value': -0.5,
                }],
                'summary': 'Set site 1 onsite energy.',
                'clarification_questions': [],
            })

        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'test-model',
        }, clear=False):
            result = build_model_hamiltonian_operations(
                'Set site 1 energy to -0.5.',
                {
                    'model': 'hubbard',
                    'sites': [
                        {'id': 0, 'epsilon': 0.0, 'U': 4.0},
                        {'id': 1, 'epsilon': 0.0, 'U': 4.0},
                    ],
                    'bonds': [],
                },
                http_post=fake_http_post,
            )

        self.assertEqual(result['operations'][0]['parameter'], 'epsilon')
        prompt_payload = json.loads(captured['messages'][1]['content'])
        self.assertIn(
            'set_site_parameter',
            prompt_payload['capability_contract']['operation_names'],
        )
        self.assertIn('epsilon', prompt_payload['capability_contract']['site_parameters'])
        self.assertEqual(prompt_payload['model_context']['sites'][1]['id'], 1)

    def test_english_system_prompt_uses_shared_capability_text(self):
        from pyscf_agent.request_builder.llm import _system_prompt

        prompt = _system_prompt('en')

        self.assertIn('supported_outputs', prompt)
        self.assertIn('capability_rules', prompt)
        self.assertNotIn('method currently supports', prompt)
        self.assertNotIn('"hf" | "dft"', prompt)

    def test_llm_payload_and_schema_use_registry_capabilities(self):
        from pyscf_agent.request_builder.llm import _build_llm_payload

        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            payload = _build_llm_payload([], None, structured_output=True, locale='en')

        user_payload = json.loads(payload['messages'][1]['content'])
        schema_properties = payload['response_format']['json_schema']['schema']['properties']

        self.assertIn('casci', user_payload['supported_methods'])
        self.assertIn('molecular_dynamics', user_payload['supported_jobs'])
        self.assertIn('ccsd_t', user_payload['supported_methods'])
        self.assertIn('density_fitting_auxbasis_recommendations', user_payload)
        self.assertIn('capability_rules', user_payload)
        self.assertIn('molecular.active_space_audit', user_payload['supported_workflow_modules'])
        self.assertIn('block2_dmrg', user_payload['supported_active_space_solvers'])
        self.assertIn('block2_dmrg', user_payload['supported_embedding_impurity_solvers'])
        self.assertTrue(any(
            'solver.options.impurity_solver' in rule
            and 'block2_dmrg' in rule
            for rule in user_payload['capability_rules']
        ))
        self.assertEqual(schema_properties['method']['enum'], user_payload['supported_methods'] + [None])
        self.assertEqual(
            schema_properties['solver']['properties']['name']['enum'],
            user_payload['supported_active_space_solvers'],
        )
        self.assertEqual(schema_properties['job']['enum'], user_payload['supported_jobs'] + [None])
        self.assertIn('molecular_dynamics', schema_properties)
        self.assertEqual(
            schema_properties['molecular_dynamics']['properties']['ensemble']['enum'],
            ['nve'],
        )
        self.assertEqual(
            schema_properties['orbital_processing']['properties']['localization_method']['enum'],
            user_payload['supported_orbital_processing_methods'],
        )
        self.assertEqual(
            schema_properties['active_space']['properties']['selection_method']['enum'],
            user_payload['supported_active_space_selection_methods'],
        )
        self.assertIn('post_cas', schema_properties)
        self.assertEqual(
            schema_properties['post_cas']['properties']['sc_nevpt2']['required'],
            ['enabled', 'root', 'density_fit'],
        )
        root_schema = schema_properties['post_cas']['properties']['sc_nevpt2']['properties']['root']
        self.assertEqual(root_schema['minimum'], 0)
        self.assertEqual(root_schema['maximum'], 0)
        self.assertEqual(
            schema_properties['workflow']['properties']['modules']['items']['enum'],
            user_payload['supported_workflow_modules'],
        )

    def test_build_prepared_request_uses_structured_seed_without_messages(self):
        prepared = build_prepared_request([], task_spec={
            'atom': 'H 0 0 0; H 0 0 0.74',
            'basis': 'sto-3g',
            'method': 'hf',
        })

        self.assertEqual(prepared['status'], 'ready')
        self.assertEqual(prepared['source'], 'structured_seed')
        self.assertIn('"basis": "sto-3g"', prepared['request_text'])

    def test_build_prepared_request_materializes_qh9_md_defaults(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': 'Run QH9-compatible MD for this molecule.'}],
                task_spec={'atom': 'H 0 0 0; H 0 0 0.74'},
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'job': 'molecular_dynamics',
                    'molecular_dynamics': {'profile': 'qh9_relaxed_scf'},
                    'request_summary': 'QH9-compatible H2 molecular dynamics',
                    'request': 'Run QH9-compatible MD for this molecule.',
                    'missing_fields': [],
                    'clarification_questions': [],
                }),
            )

        request = prepared['structured_request']
        self.assertEqual(prepared['status'], 'ready')
        self.assertEqual(request['job'], 'molecular_dynamics')
        self.assertEqual(request['method'], 'dft')
        self.assertEqual(request['xc'], 'b3lyp')
        self.assertEqual(request['basis'], 'def2-svp')
        self.assertEqual(request['outputs'], ['trajectory'])
        self.assertEqual(request['conv_tol'], 1e-8)
        self.assertEqual(request['conv_tol_grad'], 3.16e-5)
        self.assertEqual(request['grid_level'], 3)
        self.assertEqual(request['diis_space'], 8)
        self.assertEqual(request['molecular_dynamics']['sample_offset'], 9)
        self.assertEqual(request['molecular_dynamics']['sample_stride'], 10)


    def test_build_prepared_request_preserves_strict_qh9_and_full_sampling(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': 'Use QH9 SCF tolerance 1e-13 and save all 100 frames.'}],
                task_spec={'atom': 'H 0 0 0; H 0 0 0.74'},
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'job': 'molecular_dynamics',
                    'molecular_dynamics': {
                        'profile': 'qh9', 'steps': 100, 'sample_stride': 1, 'sample_offset': 0,
                    },
                    'request_summary': 'Strict QH9 H2 molecular dynamics',
                    'request': 'Use QH9 SCF tolerance 1e-13 and save all 100 frames.',
                    'missing_fields': [], 'clarification_questions': [],
                }),
            )
        self.assertEqual(prepared['status'], 'ready')
        request = prepared['structured_request']
        self.assertEqual(request['conv_tol'], 1e-13)
        self.assertEqual(request['molecular_dynamics']['profile'], 'qh9')
        self.assertEqual(request['molecular_dynamics']['sample_stride'], 1)
        self.assertEqual(request['molecular_dynamics']['sample_offset'], 0)

    def test_build_prepared_request_uses_llm_when_configured(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': '帮我算水分子的 b3lyp/6-31g 单点能'}],
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'atom': 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                    'basis': '6-31g',
                    'method': 'dft',
                    'xc': 'b3lyp',
                    'job': 'single_point',
                    'outputs': ['energy'],
                    'request_summary': '计算水分子的 b3lyp/6-31g 单点能',
                    'request': '计算水分子的单点能',
                    'missing_fields': [],
                    'clarification_questions': [],
                }),
            )

        self.assertEqual(prepared['status'], 'awaiting_approval')
        self.assertEqual(prepared['source'], 'llm')
        self.assertEqual(prepared['structured_request']['method'], 'dft')
        self.assertEqual(prepared['structured_request']['xc'], 'b3lyp')
        self.assertEqual(prepared['structured_request']['request_summary'], '计算水分子的 b3lyp/6-31g 单点能')
        self.assertEqual(prepared['approval']['type'], 'structure')

    def test_cc_basis_does_not_override_method_to_ccsd(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': 'calculate H2 with cc-pvdz'}],
                task_spec={
                    'atom': 'H 0 0 0; H 0 0 0.74',
                    'method': 'hf',
                },
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'atom': 'H 0 0 0; H 0 0 0.74',
                    'basis': 'cc-pvdz',
                    'method': 'hf',
                    'xc': None,
                    'job': 'single_point',
                    'outputs': ['energy'],
                    'request_summary': 'H2 single point with cc-pVDZ',
                    'request': 'calculate H2 with cc-pvdz',
                    'missing_fields': [],
                    'clarification_questions': [],
                }),
            )

        self.assertEqual(prepared['structured_request']['basis'], 'cc-pvdz')
        self.assertEqual(prepared['structured_request']['method'], 'hf')

    def test_structured_ccsd_t_revision_is_preserved(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': 'calculate H2 with CCSD(T)'}],
                task_spec={
                    'atom': 'H 0 0 0; H 0 0 0.74',
                    'basis': 'sto-3g',
                },
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'atom': 'H 0 0 0; H 0 0 0.74',
                    'basis': 'sto-3g',
                    'method': 'ccsd_t',
                    'xc': None,
                    'job': 'single_point',
                    'outputs': ['energy'],
                    'missing_fields': [],
                    'clarification_questions': [],
                }),
            )

        self.assertEqual(prepared['structured_request']['method'], 'ccsd_t')


    def test_dmrg_casscf_routes_through_active_space_probe(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{
                    'role': 'user',
                    'content': 'calculate a hydrogen molecule with standard geometry, using sto-3g and DMRG-CASSCF',
                }],
                task_spec={
                    'atom': 'H 0 0 0; H 0 0 0.74',
                    'basis': 'sto-3g',
                },
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'atom': 'H 0 0 0; H 0 0 0.74',
                    'basis': 'sto-3g',
                    'method': 'casscf',
                    'solver': {'name': 'block2_dmrg'},
                    'xc': None,
                    'job': 'single_point',
                    'outputs': ['energy'],
                    'active_space': {
                        'enabled': True,
                        'selection_method': 'avas',
                        'ncas': 2,
                        'nelecas': 2,
                        'orbital_indices': [0, 1],
                        'approved': False,
                    },
                    'workflow': {
                        'modules': [
                            'solver.block2.dmrg',
                            'solver.block2.bond_dimension_planning',
                        ],
                        'module_config': {
                            'solver.block2.dmrg': {'preset': 'balanced'},
                            'solver.block2.bond_dimension_planning': {},
                        },
                    },
                    'missing_fields': [],
                    'clarification_questions': [],
                }),
            )

        request = prepared['structured_request']
        probe = prepared['probe_request']
        self.assertEqual(prepared['status'], 'ready')
        self.assertEqual(request['method'], 'casscf')
        self.assertEqual(request['solver']['name'], 'block2_dmrg')
        self.assertIn('solver.block2.dmrg', request['workflow']['modules'])
        self.assertEqual(request['active_space']['selection_method'], 'avas')
        self.assertFalse(request['active_space']['approved'])
        self.assertNotIn('target_method', request['active_space'])
        self.assertEqual(probe['method'], 'hf')
        self.assertNotIn('solver', probe)
        self.assertFalse(probe['restricted'])
        self.assertEqual(probe['active_space']['selection_method'], 'occupation_window')
        self.assertEqual(probe['active_space']['target_method'], 'casscf')
        self.assertEqual(probe['active_space']['target_solver'], 'block2_dmrg')
        self.assertEqual(
            probe['workflow']['modules'],
            [
                'molecular.active_space_probe',
                'molecular.active_space_probe_refinement',
            ],
        )
        self.assertEqual(
            probe['workflow']['module_config']['molecular.active_space_probe'],
            {
                'requested_strategy': 'auto',
                'probe_method': 'hf',
                'refinement_method': 'mp2',
                'occupation_window': [0.02, 1.98],
            },
        )
        self.assertEqual(
            probe['workflow']['module_config']['molecular.active_space_probe_refinement'],
            {'initial_method': 'hf', 'refinement_method': 'mp2'},
        )
        self.assertEqual(json.loads(prepared['request_text'])['method'], 'casscf')
        self.assertEqual(json.loads(prepared['execution_request_text'])['method'], 'hf')


    def test_multiroot_dmrg_casscf_options_survive_active_space_probe(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{
                    'role': 'user',
                    'content': 'calculate H2 with 2-root DMRG-CASSCF weights=[0.7,0.3]',
                }],
                task_spec={
                    'atom': 'H 0 0 0; H 0 0 1.5',
                    'basis': '6-31g',
                },
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'atom': 'H 0 0 0; H 0 0 1.5',
                    'basis': '6-31g',
                    'method': 'casscf',
                    'solver': {'name': 'block2_dmrg', 'options': {'nroots': 2, 'state_average_weights': [0.7, 0.3]}},
                    'xc': None,
                    'job': 'single_point',
                    'outputs': ['energy', 'excited_states'],
                    'active_space': None,
                    'missing_fields': [],
                    'clarification_questions': [],
                }),
            )

        request = prepared['structured_request']
        probe = prepared['probe_request']
        self.assertEqual(request['method'], 'casscf')
        self.assertEqual(request['solver']['name'], 'block2_dmrg')
        self.assertEqual(request['solver']['options'], {
            'nroots': 2,
            'state_average_weights': [0.7, 0.3],
        })
        self.assertEqual(probe['method'], 'hf')
        self.assertNotIn('solver', probe)
        self.assertEqual(probe['active_space']['target_solver'], 'block2_dmrg')
        self.assertEqual(probe['active_space']['target_solver_options'], {
            'nroots': 2,
            'state_average_weights': [0.7, 0.3],
        })


    def test_dmrg_casscf_target_survives_standard_structure_approval(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{
                    'role': 'user',
                    'content': 'calculate a hydrogen molecule with standard geometry, using sto-3g and DMRG-CASSCF',
                }],
                locale='en',
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'atom': 'H 0 0 0; H 0 0 0.74',
                    'basis': 'sto-3g',
                    'method': 'casscf',
                    'solver': {'name': 'block2_dmrg'},
                    'xc': None,
                    'job': 'single_point',
                    'outputs': ['energy'],
                    'active_space': None,
                    'missing_fields': [],
                    'clarification_questions': [],
                }),
            )

        self.assertEqual(prepared['status'], 'awaiting_approval')
        self.assertEqual(prepared['approval']['type'], 'structure')
        request = prepared['structured_request']
        probe = prepared['approval']['probe_request']
        self.assertEqual(request['method'], 'casscf')
        self.assertEqual(request['solver']['name'], 'block2_dmrg')
        self.assertEqual(probe['method'], 'hf')
        self.assertNotIn('solver', probe)
        self.assertEqual(probe['active_space']['selection_method'], 'occupation_window')
        self.assertEqual(probe['active_space']['target_method'], 'casscf')
        self.assertEqual(probe['active_space']['target_solver'], 'block2_dmrg')

    def test_structured_casscf_seed_without_active_space_builds_probe(self):
        prepared = build_prepared_request([], task_spec={
            'atom': 'H 0 0 0; H 0 0 0.74',
            'basis': 'sto-3g',
            'method': 'casscf',
        })

        request = prepared['structured_request']
        probe = prepared['probe_request']
        self.assertEqual(request['method'], 'casscf')
        self.assertEqual(request['solver']['name'], 'fci')
        self.assertEqual(request['active_space']['selection_method'], 'manual')
        self.assertEqual(probe['method'], 'hf')
        self.assertNotIn('solver', probe)
        self.assertEqual(probe['active_space']['selection_method'], 'occupation_window')
        self.assertEqual(probe['active_space']['target_method'], 'casscf')
        self.assertEqual(probe['active_space']['target_solver'], 'fci')

    def test_structured_casscf_seed_with_unapproved_contract_opens_active_space_review(self):
        prepared = build_prepared_request([], task_spec={
            'atom': 'H 0 0 0; H 0 0 0.74',
            'basis': 'sto-3g',
            'method': 'casscf',
            'solver': {'name': 'block2_dmrg', 'options': {'preset': 'screening'}},
            'active_space': {
                'enabled': True,
                'selection_method': 'manual',
                'ncas': 2,
                'nelecas': 2,
                'orbital_indices': [0, 1],
                'approved': False,
            },
        })

        self.assertEqual(prepared['status'], 'awaiting_approval')
        self.assertIsNone(prepared['probe_request'])
        approval = prepared['approval']
        self.assertEqual(approval['type'], 'active_space')
        self.assertEqual(approval['source'], 'prepared_active_space_contract')
        self.assertEqual(approval['target_method'], 'casscf')
        self.assertEqual(approval['target_solver'], 'block2_dmrg')
        self.assertEqual(approval['active_space_contract']['ncas'], 2)
        self.assertEqual(approval['active_space_contract']['nelecas'], 2)
        self.assertFalse(approval['active_space_contract']['approved'])

    def test_generated_structure_approval_queues_explicit_active_space_review(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{
                    'role': 'user',
                    'content': 'draft H2 and calculate DMRG-CASSCF CAS(2,2)',
                }],
                locale='en',
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'atom': 'H 0 0 0; H 0 0 0.74',
                    'basis': 'sto-3g',
                    'method': 'casscf',
                    'solver': {'name': 'block2_dmrg'},
                    'job': 'single_point',
                    'outputs': ['energy'],
                    'active_space': {
                        'enabled': True,
                        'selection_method': 'manual',
                        'ncas': 2,
                        'nelecas': 2,
                        'orbital_indices': [0, 1],
                        'approved': False,
                    },
                    'missing_fields': [],
                    'clarification_questions': [],
                }),
            )

        self.assertEqual(prepared['status'], 'awaiting_approval')
        self.assertEqual(prepared['approval']['type'], 'structure')
        self.assertEqual(prepared['approval']['next_approval']['type'], 'active_space')
        self.assertEqual(
            prepared['approval']['next_approval']['active_space_contract']['orbital_indices'],
            [0, 1],
        )
        self.assertIsNone(prepared['probe_request'])

    def test_structured_cas_method_and_solver_survive_probe_preparation(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': 'calculate the configured molecule'}],
                task_spec={
                    'atom': 'N 0 0 0; N 0 0 1.1',
                    'basis': 'sto-3g',
                    'method': 'casscf',
                    'solver': {'name': 'block2_dmrg', 'options': {'nroots': 4}},
                    'active_space': {
                        'enabled': True,
                        'selection_method': 'avas',
                        'avas_targets': ['N 2p'],
                        'approved': False,
                    },
                },
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'atom': 'N 0 0 0; N 0 0 1.1',
                    'basis': 'sto-3g',
                    'method': 'casscf',
                    'solver': {'name': 'block2_dmrg', 'options': {'nroots': 4}},
                    'job': 'single_point',
                    'outputs': ['energy'],
                    'missing_fields': [],
                    'clarification_questions': [],
                }),
            )

        request = prepared['structured_request']
        probe = prepared['probe_request']
        self.assertEqual(request['method'], 'casscf')
        self.assertEqual(request['solver'], {
            'name': 'block2_dmrg',
            'options': {
                'nroots': 4,
                'state_average_weights': [0.25, 0.25, 0.25, 0.25],
            },
        })
        self.assertEqual(request['active_space']['selection_method'], 'avas')
        self.assertNotIn('target_method', request['active_space'])
        self.assertEqual(probe['method'], 'hf')
        self.assertNotIn('solver', probe)
        self.assertEqual(probe['active_space']['selection_method'], 'avas')
        self.assertEqual(probe['active_space']['target_method'], 'casscf')
        self.assertEqual(probe['active_space']['target_solver'], 'block2_dmrg')
        self.assertEqual(probe['active_space']['target_solver_options'], {
            'nroots': 4,
            'state_average_weights': [0.25, 0.25, 0.25, 0.25],
        })


    def test_build_prepared_request_preserves_density_fitting_option(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': 'calculate H2 with density fitting'}],
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'atom': 'H 0 0 0; H 0 0 0.74',
                    'basis': 'sto-3g',
                    'method': 'hf',
                    'density_fitting': {
                        'enabled': True,
                        'auxbasis': None,
                        'apply_to': 'scf',
                    },
                    'outputs': ['energy'],
                    'missing_fields': [],
                    'clarification_questions': [],
                }),
            )

        self.assertEqual(prepared['structured_request']['density_fitting']['enabled'], True)
        self.assertEqual(prepared['structured_request']['density_fitting']['apply_to'], 'scf')

    def test_build_prepared_request_prefers_structured_response_format(self):
        calls = []

        def fake_http_post(url, body, headers, timeout):
            calls.append(body)
            return {
                'choices': [
                    {
                        'message': {
                            'parsed': {
                                'atom': 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                                'basis': '6-31g',
                                'method': 'dft',
                                'xc': 'b3lyp',
                                'job': 'single_point',
                                'outputs': ['energy'],
                                'request_summary': '计算水分子的 b3lyp/6-31g 单点能',
                                'request': '计算水分子的单点能',
                                'missing_fields': [],
                                'clarification_questions': [],
                            }
                        }
                    }
                ]
            }

        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'structured-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': '帮我算水分子的 b3lyp/6-31g 单点能'}],
                http_post=fake_http_post,
            )

        self.assertEqual(prepared['status'], 'awaiting_approval')
        self.assertEqual(calls[0]['response_format']['type'], 'json_schema')
        self.assertTrue(prepared['llm_cache_scope'].endswith(':supported'))

    def test_build_prepared_request_uses_english_prompt_for_english_locale(self):
        calls = []

        def fake_http_post(url, body, headers, timeout):
            calls.append(body)
            return {
                'choices': [
                    {
                        'message': {
                            'parsed': {
                                'atom': None,
                                'basis': None,
                                'method': None,
                                'xc': None,
                                'job': 'single_point',
                                'outputs': None,
                                'request_summary': 'calculate the properties of ethanol',
                                'request': 'calculate the properties of ethanol',
                                'missing_fields': ['atom', 'basis'],
                                'clarification_questions': ['Please provide the atomic coordinates for ethanol, or tell me to generate a standard structure.'],
                            }
                        }
                    }
                ]
            }

        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'structured-model-en',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': 'calculate the properties of ethanol'}],
                locale='en',
                http_post=fake_http_post,
            )

        self.assertEqual(calls[0]['messages'][0]['role'], 'system')
        self.assertIn('Clarification questions must be concise English', calls[0]['messages'][0]['content'])
        self.assertNotIn('Clarification questions must be concise Chinese', calls[0]['messages'][0]['content'])
        self.assertEqual(prepared['status'], 'needs_clarification')

    def test_build_prepared_request_falls_back_when_response_format_is_unsupported(self):
        calls = []

        def fake_http_post(url, body, headers, timeout):
            calls.append(body)
            if len(calls) == 1:
                raise RuntimeError('LLM HTTP error 400: unsupported response_format json_schema')
            return _fake_chat_response({
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'method': 'hf',
                'job': 'single_point',
                'outputs': ['energy'],
                'request_summary': '计算氢分子的 HF 单点能',
                'request': '计算氢分子的单点能',
                'missing_fields': [],
                'clarification_questions': [],
            })

        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'fallback-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': '帮我算氢分子的单点能'}],
                http_post=fake_http_post,
            )

        self.assertEqual(prepared['status'], 'awaiting_approval')
        self.assertIn('response_format', calls[0])
        self.assertNotIn('response_format', calls[1])
        self.assertTrue(prepared['llm_cache_scope'].endswith(':unsupported'))

    def test_build_prepared_request_reuses_cached_unsupported_capability(self):
        calls = []

        def fake_http_post(url, body, headers, timeout):
            calls.append(body)
            if len(calls) == 1:
                raise RuntimeError('LLM HTTP error 400: unsupported response_format json_schema')
            return _fake_chat_response({
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'method': 'hf',
                'job': 'single_point',
                'outputs': ['energy'],
                'request': '计算氢分子的单点能',
                'missing_fields': [],
                'clarification_questions': [],
            })

        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'cached-fallback-model',
        }, clear=False):
            build_prepared_request(
                [{'role': 'user', 'content': '第一次请求'}],
                http_post=fake_http_post,
            )
            build_prepared_request(
                [{'role': 'user', 'content': '第二次请求'}],
                http_post=fake_http_post,
            )

        self.assertEqual(len(calls), 3)
        self.assertIn('response_format', calls[0])
        self.assertNotIn('response_format', calls[1])
        self.assertNotIn('response_format', calls[2])

    def test_build_prepared_request_returns_clarification_questions(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': '帮我算一下这个体系'}],
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'request': '用户想做一次计算',
                    'missing_fields': ['atom', 'basis'],
                    'clarification_questions': ['请提供分子结构。', '请指定基组。'],
                }),
            )

        self.assertEqual(prepared['status'], 'needs_clarification')
        self.assertEqual(prepared['missing_fields'], ['atom', 'basis'])
        self.assertEqual(
            prepared['clarification_questions'],
            [
                'Please provide the molecular structure or atomic coordinates.',
                'Please specify a basis set, for example sto-3g, 6-31g, or cc-pvdz.',
            ],
        )

    def test_build_prepared_request_localizes_empty_input_to_english(self):
        prepared = build_prepared_request([], locale='en')

        self.assertEqual(prepared['status'], 'needs_clarification')
        self.assertEqual(prepared['missing_fields'], ['atom', 'basis'])
        self.assertEqual(
            prepared['clarification_questions'],
            ['Please provide the calculation goal, or fill in basic information such as structure and basis set first.'],
        )
        self.assertEqual(
            prepared['messages'][0]['content'],
            'Please provide the calculation goal, or fill in basic information such as structure and basis set first.',
        )

    def test_build_prepared_request_suggests_structure_draft_for_generation_intent(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': '请你先生成一个水分子的初始结构草案，我确认后再继续计算'}],
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'request': '生成水分子的初始结构草案',
                    'request_summary': '先生成水分子结构草案，再由用户确认',
                    'missing_fields': ['atom'],
                    'clarification_questions': ['请提供分子结构。'],
                }),
            )

        self.assertEqual(prepared['status'], 'needs_clarification')
        self.assertIn('atom', prepared['missing_fields'])
        self.assertNotIn('atom', prepared['structured_request'])
        self.assertTrue(prepared['clarification_questions'])

    def test_build_prepared_request_skips_approval_when_user_already_provided_coordinates(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587，基组 sto-3g，算能量'}],
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'atom': 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                    'basis': 'sto-3g',
                    'method': 'hf',
                    'job': 'single_point',
                    'outputs': ['energy'],
                    'request': '算能量',
                    'missing_fields': [],
                    'clarification_questions': [],
                }),
            )

        self.assertEqual(prepared['status'], 'ready')
        self.assertNotIn('approval', prepared)

    def test_build_prepared_request_applies_safe_defaults_with_authorization(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': '帮我算水分子，缺失参数按默认补全，结构也可以用默认结构'}],
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'request': '计算水分子单点能',
                    'request_summary': '按默认设置计算水分子',
                    'atom': 'O 0 0 0; H 0 0 0.957; H 0.926 0 -0.24',
                    'basis': 'sto-3g',
                    'missing_fields': [],
                    'clarification_questions': [],
                }),
            )

        self.assertEqual(prepared['status'], 'awaiting_approval')
        self.assertEqual(prepared['structured_request']['basis'], 'sto-3g')
        self.assertEqual(prepared['structured_request']['method'], 'hf')
        self.assertIn('O 0 0 0', prepared['structured_request']['atom'])
        self.assertNotIn('atom', prepared['approved_structured_request'])
        self.assertFalse(any(item['field'] == 'basis' for item in prepared['applied_defaults']))
        self.assertEqual(prepared['approval']['source'], 'llm_inferred_structure')
        self.assertEqual(prepared['approval']['type'], 'structure')
        self.assertTrue(prepared['approval']['can_run_immediately'])
        self.assertEqual(prepared['approval']['assumptions'], [])

    def test_build_prepared_request_asks_only_for_high_risk_missing_info_without_authorization(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': '帮我算一个单点能'}],
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'request': '做单点能计算',
                    'missing_fields': ['atom', 'basis'],
                    'clarification_questions': ['请提供分子结构。', '请指定基组。'],
                }),
            )

        self.assertEqual(prepared['status'], 'needs_clarification')
        self.assertEqual(prepared['structured_request']['method'], 'hf')
        self.assertIn('atom', prepared['missing_fields'])
        self.assertIn('basis', prepared['missing_fields'])
        self.assertTrue(any(item['field'] == 'outputs' for item in prepared['applied_defaults']))

    def test_build_prepared_request_uses_missing_fields_without_reclassifying_questions(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': '计算甲烷的性质'}],
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'request': '计算甲烷的性质',
                    'request_summary': '计算甲烷的性质',
                    'missing_fields': ['atom', 'basis', 'unit', 'charge', 'spin', 'symmetry', 'restricted', 'max_cycle', 'conv_tol', 'verbose'],
                    'clarification_questions': [
                        'Please provide atomic coordinates for methane (e.g., in standard XYZ format with units).',
                        'Specify unit for coordinates (default angstrom).',
                        'Charge? (default 0)',
                        'Spin multiplicity? (default 0 for closed-shell)',
                        'Use symmetry? (default false)',
                        'Restricted/unrestricted? (default restricted)',
                        'Max cycles? (default 50)',
                        'Convergence tolerance? (default 1e-8)',
                        'Verbose level? (default 3)',
                    ],
                }),
            )

        self.assertEqual(prepared['status'], 'needs_clarification')
        self.assertEqual(prepared['missing_fields'], ['atom', 'basis'])
        self.assertTrue(any('methane' in question for question in prepared['clarification_questions']))
        self.assertFalse(any('generate everything' in question for question in prepared['clarification_questions']))
        self.assertTrue(any(item['field'] == 'symmetry' for item in prepared['applied_defaults']))
        self.assertTrue(any(item['field'] == 'conv_tol' for item in prepared['applied_defaults']))

    def test_build_prepared_request_does_not_invent_structure_from_authorization_text(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': '全部你来生成'}],
                task_spec={'request_summary': '计算甲烷的性质'},
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'request': '计算甲烷的性质',
                    'request_summary': '计算甲烷的性质',
                    'missing_fields': ['atom', 'basis'],
                    'clarification_questions': ['Please provide atomic coordinates for methane.', 'Please specify the basis set.'],
                }),
            )

        self.assertEqual(prepared['status'], 'needs_clarification')
        self.assertNotIn('atom', prepared['structured_request'])
        self.assertNotIn('basis', prepared['structured_request'])

    def test_build_prepared_request_offers_shorter_detected_structure_guidance(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': '帮我计算乙烷的性质'}],
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'request': '计算乙烷的性质',
                    'request_summary': '计算乙烷的性质',
                    'missing_fields': ['atom', 'basis'],
                    'clarification_questions': ['Please provide atomic coordinates for ethane (e.g., in XYZ format).'],
                }),
            )

        self.assertEqual(prepared['status'], 'needs_clarification')
        self.assertTrue(any('ethane' in question for question in prepared['clarification_questions']))
        self.assertFalse(any('If you want me to draft a structure first' in question for question in prepared['clarification_questions']))

    def test_build_prepared_request_keeps_english_structure_guidance_in_english(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [
                    {'role': 'user', 'content': 'calculate the properties of methane'},
                    {'role': 'user', 'content': 'generate it'},
                ],
                locale='en',
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'request': 'calculate the properties of methane',
                    'request_summary': 'calculate the properties of methane',
                    'atom': 'C 0 0 0; H 0.63 0.63 0.63; H -0.63 -0.63 0.63; H -0.63 0.63 -0.63; H 0.63 -0.63 -0.63',
                    'missing_fields': ['basis'],
                    'clarification_questions': ['Please specify the basis set.'],
                }),
            )

        self.assertEqual(prepared['status'], 'awaiting_approval')
        self.assertEqual(prepared['approval']['source'], 'llm_inferred_structure')
        self.assertEqual(prepared['approval']['assumptions'], [])
        self.assertTrue(all('甲烷' not in item for item in prepared['approval']['assumptions']))
        self.assertEqual(prepared['missing_fields'], ['basis'])
        self.assertTrue(any('basis set' in question.lower() for question in prepared['clarification_questions']))
        self.assertNotIn('基组', prepared['clarification_questions'][0])

    def test_build_prepared_request_does_not_replace_missing_geometry_with_molecule_lookup(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': '全部由你来生成，标准键长'}],
                task_spec={'request_summary': '计算乙烷的性质'},
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'request': '计算乙烷的性质',
                    'request_summary': '计算乙烷的性质',
                    'missing_fields': ['atom', 'basis'],
                    'clarification_questions': ['无法自动生成分子结构。请提供乙烷的分子坐标（XYZ格式）。'],
                }),
            )

        self.assertEqual(prepared['status'], 'needs_clarification')
        self.assertNotIn('atom', prepared['structured_request'])
        self.assertNotIn('basis', prepared['structured_request'])

    def test_build_prepared_request_requires_approval_for_llm_supplied_structure(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': '帮我整理一个氢分子的结构然后直接算单点能'}],
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'atom': 'H 0 0 0; H 0 0 0.74',
                    'basis': 'sto-3g',
                    'method': 'hf',
                    'job': 'single_point',
                    'outputs': ['energy'],
                    'request_summary': '计算氢分子单点能',
                    'request': '计算氢分子单点能',
                    'missing_fields': [],
                    'clarification_questions': [],
                }),
            )

        self.assertEqual(prepared['status'], 'awaiting_approval')
        self.assertEqual(prepared['approval']['source'], 'llm_inferred_structure')
        self.assertEqual(prepared['approval']['field'], 'atom')
        self.assertNotIn('atom', prepared['approved_structured_request'])
        self.assertIn('molecular identity, charge, and spin', prepared['approval']['system_message'])

    def test_build_prepared_request_requires_approval_for_general_modification_suggestions(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': '请帮我建议一个更合适的基组，并在我确认后再继续'}],
                task_spec={
                    'atom': 'H 0 0 0; H 0 0 0.74',
                    'basis': 'sto-3g',
                    'method': 'hf',
                },
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'basis': '6-31g',
                    'request': '建议把基组改成 6-31g',
                    'request_summary': '建议把基组从 sto-3g 改成 6-31g',
                    'missing_fields': [],
                    'clarification_questions': [],
                }),
            )

        self.assertEqual(prepared['status'], 'needs_clarification')
        self.assertEqual(prepared['structured_request']['basis'], '6-31g')
        self.assertTrue(any('Proposed changes' in question for question in prepared['clarification_questions']))
        self.assertTrue(any('Basis set: sto-3g -> 6-31g' in question for question in prepared['clarification_questions']))

    def test_build_prepared_request_asks_user_when_llm_output_is_unsupported(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': '帮我做几何优化并输出轨道图'}],
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'atom': 'H 0 0 0; H 0 0 0.74',
                    'basis': 'sto-3g',
                    'method': 'hf',
                    'job': 'geometry_optimization',
                    'outputs': ['energy', 'orbital_plot'],
                    'request_summary': '对氢分子做几何优化并输出轨道图',
                    'request': '做几何优化并输出轨道图',
                    'missing_fields': [],
                    'clarification_questions': [],
                }),
            )

        self.assertEqual(prepared['status'], 'awaiting_approval')
        self.assertEqual(prepared['approval']['type'], 'structure')
        self.assertIn('job', prepared['missing_fields'])
        self.assertIn('outputs', prepared['missing_fields'])
        self.assertIn('single_point', '\n'.join(prepared['clarification_questions']))
        self.assertIn('orbital_plot', '\n'.join(prepared['clarification_questions']))
        self.assertEqual(prepared['structured_request']['request_summary'], '对氢分子做几何优化并输出轨道图')

    def test_build_prepared_request_preserves_existing_request_summary(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': '基组改成 6-31g'}],
                task_spec={
                    'atom': 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                    'method': 'dft',
                    'xc': 'b3lyp',
                    'request_summary': '计算水分子的 b3lyp 单点能',
                },
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'basis': '6-31g',
                    'request': '基组改成 6-31g',
                    'missing_fields': [],
                    'clarification_questions': [],
                }),
            )

        self.assertEqual(prepared['status'], 'ready')
        self.assertEqual(prepared['structured_request']['basis'], '6-31g')
        self.assertEqual(prepared['structured_request']['request_summary'], '计算水分子的 b3lyp 单点能')

    def test_build_prepared_request_normalizes_hartree_fock_alias(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': '方法改成 Hartree-Fock'}],
                task_spec={
                    'atom': 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                    'basis': '6-31g',
                    'method': 'dft',
                    'xc': 'b3lyp',
                    'outputs': ['energy'],
                },
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'method': 'Hartree-Fock',
                    'xc': None,
                    'request': '方法改成 Hartree-Fock',
                    'missing_fields': [],
                    'clarification_questions': [],
                }),
            )

        self.assertEqual(prepared['status'], 'ready')
        self.assertEqual(prepared['structured_request']['method'], 'hf')
        self.assertNotIn('xc', prepared['structured_request'])

    def test_structured_hartree_fock_revision_overrides_seed(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': 'Hartree-Fock'}],
                task_spec={
                    'atom': 'C 0 0 0',
                    'basis': '6-31g',
                    'method': 'dft',
                    'xc': 'b3lyp',
                    'outputs': ['energy'],
                },
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'method': 'hf',
                    'request': '确认方法',
                    'missing_fields': [],
                    'clarification_questions': [],
                }),
            )

        self.assertEqual(prepared['status'], 'ready')
        self.assertEqual(prepared['structured_request']['method'], 'hf')
        self.assertNotIn('xc', prepared['structured_request'])


    def test_build_prepared_request_maps_uhf_to_hf_with_unrestricted_reference(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': 'Use UHF'}],
                task_spec={
                    'atom': 'H 0 0 0; H 0 0 0.74',
                    'basis': 'sto-3g',
                    'method': 'dft',
                    'xc': 'b3lyp',
                },
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'method': 'hf', 'restricted': False,
                    'request': 'Use UHF',
                    'missing_fields': [],
                    'clarification_questions': [],
                }),
            )

        self.assertEqual(prepared['structured_request']['method'], 'hf')
        self.assertFalse(prepared['structured_request']['restricted'])
        self.assertNotIn('xc', prepared['structured_request'])


    def test_build_prepared_request_maps_rhf_to_hf_with_restricted_reference(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': 'Use RHF'}],
                task_spec={
                    'atom': 'H 0 0 0; H 0 0 0.74',
                    'basis': 'sto-3g',
                    'method': 'dft',
                    'xc': 'b3lyp',
                },
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'method': 'hf', 'restricted': True,
                    'request': 'Use RHF',
                    'missing_fields': [],
                    'clarification_questions': [],
                }),
            )

        self.assertEqual(prepared['structured_request']['method'], 'hf')
        self.assertTrue(prepared['structured_request']['restricted'])
        self.assertNotIn('xc', prepared['structured_request'])


    def test_build_prepared_request_keeps_method_change_when_structure_needs_approval(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': '请生成水分子结构，并把方法改成 Hartree-Fock'}],
                task_spec={
                    'basis': '6-31g',
                    'method': 'dft',
                    'xc': 'b3lyp',
                    'outputs': ['energy'],
                },
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'atom': 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                    'method': 'hartree-fock',
                    'request': '生成水分子结构并改用 Hartree-Fock',
                    'missing_fields': [],
                    'clarification_questions': [],
                }),
            )

        self.assertEqual(prepared['status'], 'awaiting_approval')
        self.assertEqual(prepared['structured_request']['method'], 'hf')
        self.assertNotIn('xc', prepared['structured_request'])
        self.assertEqual(prepared['approved_structured_request']['method'], 'hf')
        self.assertEqual(prepared['approval']['structured_request']['method'], 'hf')

    def test_structured_method_and_basis_survive_structure_approval(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            prepared = build_prepared_request(
                [{'role': 'user', 'content': 'calculate the energy of benzene, using Hartree-fock and cc-pvdz'}],
                task_spec={
                    'basis': 'sto-3g',
                    'method': 'dft',
                    'xc': 'b3lyp',
                    'outputs': ['energy'],
                },
                http_post=lambda url, body, headers, timeout: _fake_chat_response({
                    'atom': 'C 1.397 0 0; C 0.699 1.210 0; C -0.699 1.210 0; C -1.397 0 0; C -0.699 -1.210 0; C 0.699 -1.210 0; H 2.481 0 0; H 1.241 2.149 0; H -1.241 2.149 0; H -2.481 0 0; H -1.241 -2.149 0; H 1.241 -2.149 0',
                    'basis': 'cc-pvdz',
                    'method': 'hf',
                    'request': 'calculate benzene energy',
                    'missing_fields': [],
                    'clarification_questions': [],
                }),
            )

        self.assertEqual(prepared['status'], 'awaiting_approval')
        self.assertEqual(prepared['structured_request']['method'], 'hf')
        self.assertEqual(prepared['structured_request']['basis'], 'cc-pvdz')
        self.assertNotIn('xc', prepared['structured_request'])
        self.assertEqual(prepared['approved_structured_request']['method'], 'hf')
        self.assertEqual(prepared['approved_structured_request']['basis'], 'cc-pvdz')


    def test_build_execution_feedback_returns_assistant_message(self):
        captured_body = {}

        def fake_http_post(url, body, headers, timeout):
            captured_body.update(body)
            return {
                'choices': [
                    {
                        'message': {
                            'content': 'The task failed. Add the missing basis-set information to the next request.',
                        }
                    }
                ]
            }

        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            feedback = llm_request_builder.build_execution_feedback(
                '{"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g"}',
                {
                    'execution_status': 'failed',
                    'analysis_summary': '任务失败：basis missing',
                    'validation_errors': ['basis missing'],
                    'messages': [{'role': 'assistant', 'content': '任务失败：basis missing'}],
                },
                http_post=fake_http_post,
            )

        self.assertEqual(feedback['role'], 'assistant')
        self.assertIn('basis-set', feedback['content'])
        self.assertIn('Active Space approval panel', captured_body['messages'][0]['content'])
        self.assertIn('block2_dmrg', captured_body['messages'][0]['content'])
        feedback_payload = json.loads(captured_body['messages'][1]['content'])
        self.assertIn(
            'block2_dmrg',
            feedback_payload['registered_embedding_impurity_solvers'],
        )

    def test_periodic_basis_feedback_uses_only_backend_verified_replacements(self):
        validation_error = (
            'Periodic basis gth-szv is unavailable for element(s): Zn, Se. '
            'Verified compatible registered basis replacements for all structure elements: '
            'gth-szv-molopt-sr, gth-dzvp-molopt-sr.'
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            feedback = llm_request_builder.build_execution_feedback(
                '{"task_type": "periodic"}',
                {
                    'execution_status': 'blocked',
                    'validation_errors': [validation_error],
                    'errors': [],
                },
            )

        self.assertEqual(feedback['role'], 'assistant')
        self.assertEqual(
            feedback['content'],
            'The periodic basis set `gth-szv` is unavailable for Zn, Se; change `periodic.basis` to '
            '`gth-szv-molopt-sr` or `gth-dzvp-molopt-sr`. '
            'These replacements were verified against the installed PySCF data for every element in the structure.',
        )

    def test_build_execution_feedback_hides_non_actionable_internal_style_text(self):
        calls = []
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            feedback = llm_request_builder.build_execution_feedback(
                '{"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g"}',
                {
                    'execution_status': 'succeeded',
                    'analysis_summary': '任务成功完成',
                    'structured_results': {
                        'solver': 'block2_dmrg',
                        'requested_outputs': ['energy', 'excited_states', 'symmetry_analysis'],
                        'missing_requested_outputs': [],
                        'requested_root_count': 2,
                        'computed_root_count': 2,
                        'targeted_roots_complete': True,
                    },
                },
                http_post=lambda *args: calls.append(args),
            )

        self.assertIsNone(feedback)
        self.assertEqual(calls, [])

    def test_compact_execution_report_uses_the_executed_model_solver(self):
        from pyscf_agent.request_builder.llm import _compact_execution_report

        compact = _compact_execution_report({
            'execution_status': 'succeeded',
            'task_spec': {
                'task_type': 'model_hamiltonian',
                'solver': {'name': 'block2_dmrg', 'options': {'nroots': 2}},
                'model_hamiltonian': {'spec': {'solver': 'fci', 'sites': [{}, {}]}},
            },
            'structured_results': {
                'task_type': 'model_hamiltonian',
                'solver': 'block2_dmrg',
                'requested_outputs': ['energy', 'excited_states'],
                'missing_requested_outputs': [],
            },
        })

        self.assertEqual(compact['task_spec']['solver']['name'], 'block2_dmrg')
        self.assertEqual(compact['task_spec']['solver']['options']['nroots'], 2)
        self.assertEqual(compact['task_spec']['model_hamiltonian']['spec']['solver'], 'block2_dmrg')

    def test_missing_dmrg_symmetry_output_uses_deterministic_feedback(self):
        calls = []
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            feedback = llm_request_builder.build_execution_feedback(
                '{"task_type": "model_hamiltonian"}',
                {
                    'execution_status': 'succeeded',
                    'structured_results': {
                        'solver': 'block2_dmrg',
                        'requested_outputs': ['energy', 'symmetry_analysis'],
                        'missing_requested_outputs': ['symmetry_analysis'],
                    },
                },
                http_post=lambda *args: calls.append(args),
            )

        self.assertIn('`symmetry_analysis`', feedback['content'])
        self.assertIn('output/provider issue', feedback['content'])
        self.assertNotIn('dmrg_spin_square', feedback['content'])
        self.assertEqual(calls, [])

    def test_build_result_analysis_returns_formal_text(self):
        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            analysis = llm_request_builder.build_result_analysis(
                '{"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g"}',
                {
                    'execution_status': 'succeeded',
                    'analysis_summary': '任务成功完成',
                    'structured_results': {'converged': True, 'energy': -1.1},
                },
                http_post=lambda url, body, headers, timeout: {
                    'choices': [
                        {
                            'message': {
                                'content': '一、计算概况\n已完成氢分子单点计算。\n二、关键结果\nSCF 已收敛，总能为 -1.1 Ha。',
                            }
                        }
                    ]
                },
            )

        self.assertIn('一、计算概况', analysis)
        self.assertIn('二、关键结果', analysis)

    def test_build_result_analysis_uses_english_prompt_for_english_locale(self):
        captured_body = {}

        def fake_http_post(url, body, headers, timeout):
            captured_body.update(body)
            return {
                'choices': [
                    {
                        'message': {
                            'content': '1. Calculation Overview\nThe run completed successfully.',
                        }
                    }
                ]
            }

        with mock.patch.dict(os.environ, {
            'PYSCF_AGENT_LLM_BASE_URL': 'https://example.test/v1',
            'PYSCF_AGENT_LLM_MODEL': 'demo-model',
        }, clear=False):
            analysis = llm_request_builder.build_result_analysis(
                '{"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g"}',
                {
                    'execution_status': 'succeeded',
                    'analysis_summary': 'Task completed successfully',
                    'structured_results': {'converged': True, 'energy': -1.1},
                },
                locale='en',
                http_post=fake_http_post,
            )

        self.assertIn('1. Calculation Overview', analysis)
        self.assertEqual(captured_body['messages'][0]['role'], 'system')
        self.assertIn('Reply in English', captured_body['messages'][0]['content'])
        self.assertNotIn('Reply in Chinese', captured_body['messages'][0]['content'])
        user_payload = json.loads(captured_body['messages'][1]['content'])
        self.assertEqual(user_payload['response_language'], 'English')
        self.assertEqual(set(user_payload['registered_output_contracts']), {'energy'})

    def test_compact_execution_report_preserves_orbital_processing_outputs(self):
        from pyscf_agent.request_builder.llm import _compact_execution_report

        compact = _compact_execution_report({
            'execution_status': 'succeeded',
            'analysis_summary': 'Method=HF; orbital_localization=boys(completed)',
            'structured_results': {
                'converged': True,
                'energy': -1.1,
                'orbital_processing': {
                    'enabled': True,
                    'localization_method': 'boys',
                    'localization_status': 'completed',
                    'localized_orbital_shape': [2, 2],
                    'orbital_table': [
                        {'index': 0, 'energy': -0.5, 'occupation': 2.0},
                        {'index': 1, 'energy': 0.6, 'occupation': 0.0},
                    ],
                },
            },
            'artifacts': [
                {
                    'kind': 'orbital_processing',
                    'path': '/tmp/result-orbital-processing.json',
                    'mime_type': 'application/json; charset=utf-8',
                    'description': 'Orbital processing and localization summary',
                },
            ],
        })

        orbital = compact['structured_results']['orbital_processing']
        self.assertEqual(orbital['localization_method'], 'boys')
        self.assertEqual(orbital['localization_status'], 'completed')
        self.assertEqual(orbital['localized_orbital_shape'], [2, 2])
        self.assertEqual(orbital['orbital_table_row_count'], 2)
        self.assertEqual(compact['artifacts'][0]['kind'], 'orbital_processing')
        self.assertTrue(compact['artifacts'][0]['registered_output'])

    def test_compact_execution_report_preserves_observables_from_registry_contracts(self):
        from pyscf_agent.request_builder.llm import _compact_execution_report

        compact = _compact_execution_report({
            'execution_status': 'succeeded',
            'analysis_summary': 'Method=HF; Total energy=-1.1 Ha',
            'structured_results': {
                'task_type': 'molecular',
                'method': 'hf',
                'converged': True,
                'energy': -1.1,
                'reference_energy': -1.1,
                'homo': -0.5,
                'lumo': 0.2,
                'gap': 0.7,
                'dipole': [0.0, 0.0, 0.1],
            },
        })

        results = compact['structured_results']
        self.assertEqual(results['task_type'], 'molecular')
        self.assertEqual(results['energy'], -1.1)
        self.assertEqual(results['reference_energy'], -1.1)
        self.assertEqual(results['homo'], -0.5)
        self.assertEqual(results['lumo'], 0.2)
        self.assertEqual(results['gap'], 0.7)
        self.assertEqual(results['dipole'], [0.0, 0.0, 0.1])

    def test_result_analysis_prompt_leaves_selected_contracts_in_user_payload(self):
        from pyscf_agent.request_builder.llm import _execution_feedback_prompt, _result_analysis_prompt

        prompt = _result_analysis_prompt('en')

        self.assertNotIn('Registered output contracts:', prompt)
        self.assertNotIn('orbital_summary_table', prompt)
        self.assertIn('verified compatible for all structure elements', prompt)
        self.assertIn(
            'requested low-energy FCI or DMRG roots',
            _execution_feedback_prompt('en'),
        )
        self.assertIn(
            'valid nested `solver.options.impurity_solver`',
            _execution_feedback_prompt('en'),
        )
        self.assertIn(
            'requested low-energy FCI or DMRG roots',
            _result_analysis_prompt('en'),
        )
        self.assertNotRegex(prompt, r'[\u3400-\u9fff]')
        self.assertNotRegex(_result_analysis_prompt('zh'), r'[\u3400-\u9fff]')
        self.assertNotRegex(_execution_feedback_prompt('zh'), r'[\u3400-\u9fff]')


if __name__ == '__main__':
    unittest.main()
