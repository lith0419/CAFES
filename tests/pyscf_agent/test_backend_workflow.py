#!/usr/bin/env python

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from pyscf_agent import contracts
from pyscf_agent.backend import execution as backend_execution
from pyscf_agent.backend import parsing as backend_parsing
from pyscf_agent.backend import state as backend_state
from pyscf_agent.backend import workflow as backend_workflow
from pyscf_agent.registry import platform as capability_registry


class DummyWorkflow:
    '''Minimal workflow stub used to verify LangGraph orchestration calls.'''

    def __init__(self):
        self.calls = []

    def invoke(self, state):
        self.calls.append(state)
        final_state = dict(state)
        final_state['execution_status'] = 'stubbed'
        final_state['task_report'] = {
            'channel': state.get('channel'),
            'messages': list(state.get('messages', [])),
        }
        return final_state


class BackendWorkflowTests(unittest.TestCase):
    def setUp(self):
        backend_workflow.get_workflow.cache_clear()

    def tearDown(self):
        backend_workflow.get_workflow.cache_clear()

    def test_run_workflow_invokes_supplied_workflow(self):
        workflow = DummyWorkflow()
        initial_state = backend_state.default_state('atom: H 0 0 0', channel='test')

        result = backend_workflow.run_workflow(initial_state, workflow=workflow)

        self.assertEqual(result['execution_status'], 'stubbed')
        self.assertEqual(len(workflow.calls), 1)
        self.assertEqual(workflow.calls[0]['channel'], 'test')

    def test_capability_registry_is_backend_fact_source(self):
        registry = capability_registry.default_registry()
        payload = capability_registry.public_registry_payload(registry)
        model_status = {
            item['local_id']: item['contract']['status']
            for item in payload['entries']
            if item['namespace'] == 'model_hamiltonian.model'
        }

        self.assertEqual(tuple(
            item.id for item in registry.capabilities(
                namespace='molecular.method', backend_allowed=True
            )
        ), capability_registry.SUPPORTED_METHODS)
        self.assertEqual(tuple(
            item.id for item in registry.capabilities(
                namespace='model_hamiltonian.solver', backend_allowed=True
            )
        ), capability_registry.SUPPORTED_MODEL_SOLVERS)
        self.assertEqual(model_status['hubbard'], 'executable')
        self.assertEqual(model_status['holstein_hubbard'], 'design_only')
        self.assertNotIn('holstein_hubbard', [
            item.id for item in registry.capabilities(
                namespace='model_hamiltonian.model', backend_allowed=True
            )
        ])

    def test_default_state_treats_run_directory_as_parent_work_dir(self):
        state = backend_state.default_state(
            'atom: H 0 0 0',
            channel='test',
            work_dir='/tmp/pyscf-agent-work/builder-run',
            run_id='builder-run',
        )

        self.assertEqual(state['work_dir'], os.path.abspath('/tmp/pyscf-agent-work'))
        self.assertEqual(state['run_id'], 'builder-run')

    def test_execute_request_uses_cached_langgraph_workflow_by_default(self):
        workflow = DummyWorkflow()
        with mock.patch.object(backend_workflow, 'build_workflow', return_value=workflow) as mock_build_workflow:
            first = backend_workflow.execute_request('basis: sto-3g', channel='cli')
            second = backend_workflow.execute_request('basis: 6-31g', channel='web')

        self.assertEqual(mock_build_workflow.call_count, 1)
        self.assertEqual(first['execution_status'], 'stubbed')
        self.assertEqual(second['execution_status'], 'stubbed')
        self.assertEqual(workflow.calls[0]['channel'], 'cli')
        self.assertEqual(workflow.calls[1]['channel'], 'web')

    def test_run_workflow_sequential_keeps_existing_pipeline_behavior(self):
        result = backend_workflow.run_workflow_sequential(backend_state.default_state('basis: sto-3g', channel='test'))

        self.assertEqual(result['execution_status'], 'blocked')
        self.assertIn('Missing molecular geometry', result['raw_stderr'])
        self.assertIn('task_report', result)

    def test_task_report_uses_a_versioned_contract(self):
        report = contracts.TaskReport(
            run_id='single-task',
            execution_status='succeeded',
            task_spec={'task_type': 'molecular'},
        ).to_dict()

        self.assertEqual(report['schema'], contracts.TASK_REPORT_SCHEMA)
        self.assertEqual(report['execution_status'], 'succeeded')

    def test_execute_request_can_return_english_summary_and_questions(self):
        report = backend_workflow.execute_request('{"method": "dft"}', channel='web', locale='en')['task_report']

        self.assertEqual(report['execution_status'], 'blocked')
        self.assertTrue(report['analysis_summary'].startswith('Task was not executed:'))
        self.assertIn('Please provide the molecular structure', report['clarification_questions'][0])
        self.assertEqual(report['messages'][-1]['content'], report['analysis_summary'])

    def test_execute_request_localizes_invalid_atom_type_in_english(self):
        report = backend_workflow.execute_request('{"atom": ["C", 0, 0, 0], "basis": "sto-3g"}', channel='web', locale='en')['task_report']

        self.assertEqual(report['execution_status'], 'blocked')
        self.assertIn('atom coordinates must be specified as strings', report['validation_errors'][0])
        self.assertIn('not as Python list/tuple', report['validation_errors'][0])
        self.assertTrue(report['analysis_summary'].startswith('Task was not executed:'))
        self.assertNotIn('任务未执行', report['analysis_summary'])

    def test_execute_request_accepts_standard_xyz_header(self):
        state = backend_state.default_state('hydrogen xyz header', channel='web', locale='en')
        state['task_spec'] = {
            'atom': '2\nHydrogen\nH 0 0 0\nH 0 0 0.74',
            'basis': 'sto-3g',
            'method': 'hf',
        }
        report = backend_workflow.run_workflow_sequential(state)['task_report']

        self.assertNotEqual(report['execution_status'], 'blocked')
        self.assertFalse(any('Element x y z' in message for message in report.get('validation_errors', [])))
        self.assertFalse(any('atom coordinates must be specified as strings' in message for message in report.get('validation_errors', [])))

    def test_molecular_default_outputs_include_homo_lumo_and_dipole(self):
        request = '{"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g", "method": "hf"}'
        report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']
        results = report['structured_results']

        self.assertEqual(report['execution_status'], 'succeeded')
        self.assertEqual(report['task_spec']['analysis']['outputs'], ['energy', 'homo_lumo', 'dipole'])
        self.assertIsInstance(results['homo'], float)
        self.assertIsInstance(results['lumo'], float)
        self.assertIsInstance(results['gap'], float)
        self.assertIsInstance(results['dipole'], list)
        self.assertEqual(
            [entry['module_id'] for entry in report['module_execution_trace']],
            list(report['workflow_configuration']['execution_order']),
        )
        self.assertIn(
            'workflow-execution-trace',
            [artifact['kind'] for artifact in report['artifacts']],
        )

    def test_molecular_newton_scf_runtime_is_executed_and_reported(self):
        request = {
            'task_type': 'molecular',
            'system': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
            },
            'method': {'name': 'hf', 'restricted': False},
            'analysis': {'outputs': ['energy']},
            'runtime': {
                'max_cycle': 50,
                'scf_algorithm': 'newton',
            },
        }

        report = backend_workflow.execute_request(
            json.dumps(request),
            channel='test',
            locale='en',
        )['task_report']

        self.assertEqual(report['execution_status'], 'succeeded')
        self.assertEqual(report['task_spec']['runtime']['scf_algorithm'], 'newton')
        self.assertEqual(report['structured_results']['scf_algorithm'], 'newton')

    def test_newton_uhf_reference_remains_compatible_with_spin_adapted_casscf(self):
        task_spec = contracts.TaskSpec(
            system=contracts.SystemSpec(
                atom='H 0 0 0; H 0 0 0.74',
                basis='sto-3g',
            ),
            method=contracts.MethodSpec(name='casscf', restricted=True),
            analysis=contracts.AnalysisSpec(outputs=['energy']),
            runtime=contracts.RuntimeSpec(
                max_cycle=50,
                scf_algorithm='newton',
            ),
            solver=contracts.SolverSpec(name='fci'),
            active_space=contracts.ActiveSpaceSpec(
                enabled=True,
                selection_method='manual',
                ncas=2,
                nelecas=2,
                orbital_indices=[0, 1],
                approved=True,
            ),
        )

        results = backend_execution._run_pyscf_task(task_spec)

        self.assertTrue(results['reference_converged'])
        self.assertTrue(results['solver_converged'])
        self.assertTrue(results['converged'])
        self.assertEqual(results['scf_algorithm'], 'newton')
        self.assertEqual(results['initial_reference'], 'uhf')
        self.assertEqual(results['cas_reference'], 'rhf')

    def test_casscf_convergence_is_independent_of_initial_reference_status(self):
        casscf = contracts.TaskSpec(
            method=contracts.MethodSpec(name='casscf', restricted=True),
            solver=contracts.SolverSpec(name='block2_dmrg'),
        )
        casci = contracts.TaskSpec(
            method=contracts.MethodSpec(name='casci', restricted=True),
            solver=contracts.SolverSpec(name='block2_dmrg'),
        )

        self.assertTrue(backend_execution._molecular_task_converged(
            casscf,
            reference_converged=False,
            solver_converged=True,
        ))
        self.assertFalse(backend_execution._molecular_task_converged(
            casci,
            reference_converged=False,
            solver_converged=True,
        ))

    def test_active_space_probe_reference_scf_recovers_independently(self):
        task_spec = contracts.TaskSpec(
            method=contracts.MethodSpec(name='hf', restricted=False),
            runtime=contracts.RuntimeSpec(max_cycle=50),
        )
        state = backend_state.default_state('{}', channel='test')
        state.update({
            'calculation_role': 'active_space_probe',
            'task_spec': contracts.task_spec_to_dict(task_spec),
            'execution_status': 'unconverged',
            'retry_count': 0,
            'max_retries': 1,
            'structured_results': {
                'reference_converged': False,
            },
            'raw_stderr': 'The mean-field reference did not converge.',
            'errors': [{
                'stage': 'execution',
                'code': 'scf_unconverged',
                'message': 'The mean-field reference did not converge.',
            }],
        })

        recovered = backend_execution.repair_or_retry(state)

        self.assertEqual(recovered['execution_status'], 'pending')
        self.assertEqual(recovered['retry_count'], 1)
        self.assertEqual(recovered['task_spec']['runtime']['max_cycle'], 100)
        self.assertEqual(recovered['task_spec']['runtime']['scf_algorithm'], 'standard')

    def test_active_space_approval_carries_recovered_runtime(self):
        task_spec = contracts.TaskSpec(
            method=contracts.MethodSpec(name='hf', restricted=False),
            runtime=contracts.RuntimeSpec(
                max_cycle=100,
                conv_tol=1e-8,
                scf_algorithm='newton',
            ),
            active_space=contracts.ActiveSpaceSpec(
                enabled=True,
                target_method='casscf',
                target_solver='block2_dmrg',
            ),
        )
        state = backend_state.default_state('{}', channel='test')
        state.update({
            'calculation_role': 'active_space_probe',
            'retry_count': 1,
            'structured_results': {
                'reference_converged': False,
                'active_space': {
                    'ncas': 6,
                    'nelecas': 6,
                    'orbital_indices': [1, 2, 3, 4, 5, 6],
                    'target_method': 'casscf',
                    'target_solver': 'block2_dmrg',
                    'approved': False,
                    'audit': {
                        'selection_parameters': {},
                    },
                },
            },
        })

        approval = backend_execution._active_space_approval_contract(state, task_spec)

        self.assertIsNotNone(approval)
        self.assertEqual(approval['runtime_contract'], {
            'max_cycle': 100,
            'conv_tol': 1e-8,
            'verbose': 4,
            'scf_algorithm': 'newton',
        })
        self.assertEqual(approval['probe_reference_status'], {
            'reference_converged': False,
            'retry_count': 1,
        })

    def test_molecular_projected_1rdm_restart_is_saved_and_reused(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_request = {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'method': 'hf',
                'restricted': False,
                'outputs': ['energy'],
            }
            source = backend_workflow.execute_request(
                json.dumps(source_request),
                channel='test',
                locale='en',
                work_dir=tmpdir,
                run_id='one-particle-source',
            )['task_report']
            source_artifact = next(
                artifact for artifact in source['artifacts']
                if artifact.get('kind') == 'one_particle_state'
            )

            restart_request = {
                **source_request,
                'atom': 'H 0 0 0; H 0 0 0.80',
                'initial_state': {
                    'mode': 'projected_1rdm',
                    'source_case_id': 'case-0001',
                    'source_artifact': source_artifact,
                },
            }
            restarted = backend_workflow.execute_request(
                json.dumps(restart_request),
                channel='test',
                locale='en',
                work_dir=tmpdir,
                run_id='one-particle-restart',
            )['task_report']

        self.assertEqual(source['execution_status'], 'succeeded')
        self.assertEqual(source['structured_results']['one_particle_state']['spin_mode'], 'unrestricted')
        self.assertEqual(restarted['execution_status'], 'succeeded')
        self.assertEqual(restarted['task_spec']['initial_state']['mode'], 'projected_1rdm')
        self.assertEqual(restarted['structured_results']['initial_state']['status'], 'applied')
        self.assertEqual(
            restarted['structured_results']['initial_state']['source_case_id'],
            'case-0001',
        )

    def test_molecular_post_hf_methods_support_closed_shell(self):
        for method_name in ('mp2', 'ccsd', 'ccsd_t', 'fci'):
            with self.subTest(method=method_name):
                request = (
                    '{"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g", '
                    '"method": "%s", "outputs": ["energy"]}'
                ) % method_name
                report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

                self.assertEqual(report['execution_status'], 'succeeded')
                self.assertEqual(report['structured_results']['method'], method_name)
                self.assertEqual(report['structured_results']['reference'], 'rhf')
                self.assertIsInstance(report['structured_results']['energy'], float)
                self.assertIsInstance(report['structured_results']['reference_energy'], float)
                self.assertNotIn('correlation_energy', report['structured_results'])
                self.assertNotIn('correlation_energy=', report['analysis_summary'])
                if method_name == 'ccsd_t':
                    self.assertIsInstance(report['structured_results']['triples_correction'], float)

    def test_molecular_fci_supports_requested_low_energy_roots(self):
        request = json.dumps({
            'atom': 'H 0 0 0; H 0 0 0.74',
            'basis': 'sto-3g',
            'method': 'fci',
            'solver': {'name': 'fci', 'options': {'nroots': 2}},
            'outputs': ['energy', 'excited_states'],
        })

        report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

        self.assertEqual(report['execution_status'], 'succeeded')
        results = report['structured_results']
        self.assertEqual(results['requested_root_count'], 2)
        self.assertEqual(results['computed_root_count'], 2)
        self.assertEqual(len(results['state_energies']), 2)
        manifold = results['correlation_diagnostics']['low_energy_manifold']
        self.assertEqual(manifold['classification'], 'isolated_within_computed_sector')
        self.assertEqual(
            manifold['state_sector']['scope'],
            'fixed_particle_spin_projection_sector',
        )

    def test_cisd_is_rejected_outside_the_registered_method_boundary(self):
        request = (
            '{"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g", '
            '"method": "cisd", "outputs": ["energy"]}'
        )

        report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

        self.assertEqual(report['execution_status'], 'blocked')
        self.assertTrue(any('Unsupported method "cisd"' in item for item in report['validation_errors']))
        self.assertEqual(report['errors'][0]['code'], 'unsupported_method')

    def test_post_hf_methods_ignore_blank_xc_field(self):
        request = (
            '{"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g", '
            '"method": "mp2", "xc": "", "outputs": ["energy"]}'
        )
        report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

        self.assertEqual(report['execution_status'], 'succeeded')
        self.assertFalse(any('xc' in message.lower() for message in report.get('validation_errors', [])))
        self.assertEqual(report['structured_results']['method'], 'mp2')

    def test_molecular_density_fitting_is_explicit_workflow_option(self):
        request = (
            '{"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g", '
            '"method": "hf", "density_fitting": {"enabled": true, "auxbasis": null, "apply_to": "scf"}, '
            '"outputs": ["energy"]}'
        )
        report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

        self.assertEqual(report['execution_status'], 'succeeded')
        self.assertEqual(report['structured_results']['density_fitting']['enabled'], True)
        self.assertEqual(report['structured_results']['density_fitting']['apply_to'], 'scf')
        self.assertIn('density_fitting=SCF', report['analysis_summary'])

    def test_generated_script_delegates_to_shared_execution_path(self):
        task_spec = backend_parsing.task_spec_from_partial({
            'atom': 'H 0 0 0; H 0 0 0.74',
            'basis': 'sto-3g',
            'method': 'hf',
            'density_fitting': {'enabled': True, 'auxbasis': 'def2-svp-jkfit', 'apply_to': 'scf'},
        })
        script = backend_execution.generate_input_script(task_spec)

        self.assertIn('from pyscf_agent.backend.execution import _run_pyscf_task', script)
        self.assertIn('task_spec = task_spec_from_dict', script)
        self.assertIn("'density_fitting'", script)
        self.assertIn('def2-svp-jkfit', script)

    def test_density_fitting_auxbasis_must_match_basis(self):
        request = (
            '{"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g", '
            '"method": "hf", "density_fitting": {"enabled": true, "auxbasis": "cc-pvdz-jkfit", "apply_to": "scf"}, '
            '"outputs": ["energy"]}'
        )
        report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

        self.assertEqual(report['execution_status'], 'blocked')
        self.assertTrue(any('not recommended for basis sto-3g' in item for item in report['validation_errors']))

    def test_density_fitting_auxbasis_must_be_registered(self):
        request = (
            '{"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g", '
            '"method": "hf", "density_fitting": {"enabled": true, "auxbasis": "cc-pvdz-ri", "apply_to": "scf"}, '
            '"outputs": ["energy"]}'
        )
        report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

        self.assertEqual(report['execution_status'], 'blocked')
        self.assertTrue(any('Unsupported density fitting auxbasis' in item for item in report['validation_errors']))

    def test_density_fitting_request_text_is_parsed(self):
        parsed = backend_parsing.parse_user_request('calculate H2 HF/STO-3G with density fitting')

        self.assertTrue(parsed['density_fitting'])

    def test_molecular_dynamics_request_text_selects_trajectory_job(self):
        parsed = backend_parsing.parse_user_request('对这个分子做 QH9 分子动力学')

        self.assertEqual(parsed['job'], 'molecular_dynamics')
        self.assertEqual(parsed['outputs'], ['trajectory'])

    def test_cc_basis_name_does_not_imply_coupled_cluster(self):
        parsed = backend_parsing.parse_user_request('calculate H2 with cc-pvdz')

        self.assertEqual(parsed['basis'], 'cc-pvdz')
        self.assertNotEqual(parsed.get('method'), 'ccsd')

    def test_generic_coupled_cluster_text_does_not_imply_ccsd(self):
        parsed = backend_parsing.parse_user_request('calculate H2 with a coupled-cluster method')

        self.assertNotEqual(parsed.get('method'), 'ccsd')

    def test_rhf_and_uhf_text_select_hf_reference_constraints(self):
        rhf = backend_parsing.parse_user_request('calculate H2 with RHF/STO-3G')
        uhf = backend_parsing.parse_user_request('calculate H2 with UHF/STO-3G')

        self.assertEqual(rhf['method'], 'hf')
        self.assertTrue(rhf['restricted'])
        self.assertEqual(uhf['method'], 'hf')
        self.assertFalse(uhf['restricted'])

    def test_common_basis_names_are_parsed_from_request_text(self):
        cases = {
            'calculate H2 with 6-311++g**': '6-311++g**',
            'calculate H2 with cc-pVTZ': 'cc-pvtz',
            'calculate H2 with aug-cc-pVTZ': 'aug-cc-pvtz',
            'calculate H2 with def2-TZVP': 'def2-tzvp',
            'calculate H2 with ANO-RCC': 'ano-rcc',
            'calculate H2 with LANL2DZ': 'lanl2dz',
        }
        for text, expected_basis in cases.items():
            with self.subTest(text=text):
                parsed = backend_parsing.parse_user_request(text)
                self.assertEqual(parsed.get('basis'), expected_basis)

    def test_ccsd_t_text_is_parsed_as_perturbative_triples(self):
        parsed = backend_parsing.parse_user_request('calculate H2 with CCSD(T) sto-3g')

        self.assertEqual(parsed['method'], 'ccsd_t')

    def test_ccsdt_text_keeps_conventional_ccsd_t_alias(self):
        parsed = backend_parsing.parse_user_request('calculate H2 with CCSDT sto-3g')

        self.assertEqual(parsed['method'], 'ccsd_t')

    def test_molecular_post_hf_methods_support_open_shell(self):
        for method_name in ('mp2', 'ccsd', 'fci'):
            with self.subTest(method=method_name):
                request = (
                    '{"atom": "H 0 0 0", "basis": "sto-3g", "spin": 1, '
                    '"method": "%s", "outputs": ["energy"]}'
                ) % method_name
                report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

                self.assertEqual(report['execution_status'], 'succeeded')
                self.assertEqual(report['structured_results']['method'], method_name)
                self.assertEqual(report['structured_results']['reference'], 'uhf')
                self.assertIsInstance(report['structured_results']['energy'], float)
                self.assertIsInstance(report['structured_results']['reference_energy'], float)
                self.assertNotIn('correlation_energy', report['structured_results'])
                self.assertNotIn('correlation_energy=', report['analysis_summary'])

    def test_orbital_processing_is_independent_of_cas_methods(self):
        request = (
            '{"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g", '
            '"method": "hf", "outputs": ["energy"], '
            '"orbital_processing": {"enabled": true, "localization_method": "none", "use_natural_orbitals": true}}'
        )
        report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

        self.assertEqual(report['execution_status'], 'succeeded')
        self.assertEqual(report['structured_results']['method'], 'hf')
        self.assertIn('orbital_processing', report['structured_results'])
        self.assertIn('orbital_table', report['structured_results']['orbital_processing'])

    def test_boys_localization_is_summarized_as_generated_output(self):
        request = (
            '{"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g", '
            '"method": "hf", "outputs": ["energy"], '
            '"orbital_processing": {"enabled": true, "localization_method": "boys", "use_natural_orbitals": false}}'
        )
        report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

        self.assertEqual(report['execution_status'], 'succeeded')
        orbital = report['structured_results']['orbital_processing']
        self.assertEqual(orbital['localization_method'], 'boys')
        self.assertEqual(orbital['localization_status'], 'completed')
        self.assertIn('localized_orbital_shape', orbital)
        self.assertIn('orbital_localization=boys(completed)', report['analysis_summary'])
        self.assertTrue(any(item.get('kind') == 'orbital_processing' for item in report['artifacts']))

    def test_active_space_audit_is_written_as_generated_artifact(self):
        request = (
            '{"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g", '
            '"method": "hf", "outputs": ["energy"], '
            '"active_space": {"enabled": true, "selection_method": "manual", '
            '"ncas": 2, "nelecas": 2, "orbital_indices": [0, 1], "approved": false}}'
        )
        report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

        self.assertEqual(report['execution_status'], 'succeeded')
        self.assertNotIn('solver', report['task_spec'])
        active_space = report['structured_results']['active_space']
        self.assertIn('audit_summary', active_space)
        self.assertIn('audit', active_space)
        self.assertEqual(active_space['audit']['manual_approval']['status'], 'requires_user_review')
        self.assertEqual(active_space['audit']['selected_candidate_method'], 'manual')
        candidate_methods = [item['method'] for item in active_space['audit']['candidate_active_spaces']]
        self.assertEqual(set(candidate_methods), {
            'manual',
            'uno',
            'chemical_valence',
            'evidence_expanded',
            'frontier_fallback',
            'avas',
            'merged',
        })
        reviewable_sizes = [
            item['ncas']
            for item in active_space['audit']['candidate_active_spaces']
            if item.get('status') in ('available', 'mapping_ambiguous')
        ]
        self.assertEqual(reviewable_sizes, sorted(reviewable_sizes))
        diagnostics = report['structured_results']['correlation_diagnostics']
        self.assertIn('molecular_correlation_risk', diagnostics)
        self.assertIn('physics_score', diagnostics['molecular_correlation_risk'])
        self.assertIn('solver_stress_score', diagnostics['molecular_correlation_risk'])
        self.assertIn('scf_stability', report['structured_results'])
        artifact_kinds = {item.get('kind') for item in report['artifacts']}
        self.assertIn('active_space_audit', artifact_kinds)
        self.assertIn('active_space_audit_table', artifact_kinds)
        self.assertIn('correlation_diagnostics', artifact_kinds)
        self.assertIn('scf_stability', artifact_kinds)
        self.assertIn('active_space_audit=requires_user_review', report['analysis_summary'])
        self.assertIn('molecular_correlation_risk=', report['analysis_summary'])
        self.assertNotIn('physics=', report['analysis_summary'])
        self.assertNotIn('solver_stress=', report['analysis_summary'])
        completed_modules = {
            item.get('module_id')
            for item in report['module_runtime_observations']
            if item.get('status') == 'completed'
        }
        self.assertIn('molecular.correlation_diagnostics', completed_modules)
        self.assertIn('molecular.active_space_audit', completed_modules)
        self.assertNotIn('module_context_ref', report)
        self.assertNotIn('_module_results', report)

    def test_active_space_audit_preserves_composite_method_target(self):
        request = (
            '{"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g", '
            '"method": "mp2", "restricted": false, "outputs": ["energy"], '
            '"active_space": {"enabled": true, "selection_method": "occupation_window", '
            '"ncas": null, "nelecas": null, "orbital_indices": [], '
            '"target_method": "casscf", "target_solver": "block2_dmrg", "approved": false}}'
        )

        report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

        self.assertEqual(report['execution_status'], 'succeeded')
        active_space = report['structured_results']['active_space']
        self.assertEqual(active_space['target_method'], 'casscf')
        self.assertEqual(active_space['target_solver'], 'block2_dmrg')
        recommendation = active_space['audit']['method_recommendation']
        self.assertEqual(recommendation['recommended_next_step'], 'casscf')
        self.assertEqual(recommendation['target_solver'], 'block2_dmrg')
        approval = report['approval']
        self.assertEqual(approval['type'], 'active_space')
        self.assertEqual(approval['source'], 'active_space_audit')
        self.assertEqual(approval['status'], 'requires_user_review')
        self.assertEqual(approval['target_method'], 'casscf')
        self.assertEqual(approval['target_solver'], 'block2_dmrg')
        self.assertEqual(
            approval['active_space_contract']['target_solver_options']['preset'],
            'screening',
        )

    def test_explicit_dmrg_cas_probe_falls_back_to_reviewable_frontier_space(self):
        request = (
            '{"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g", '
            '"method": "mp2", "restricted": false, "outputs": ["energy"], '
            '"active_space": {"enabled": true, "selection_method": "occupation_window", '
            '"ncas": null, "nelecas": null, "orbital_indices": [], '
            '"target_method": "casscf", "target_solver": "block2_dmrg", "approved": false}}'
        )

        report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

        self.assertEqual(report['execution_status'], 'succeeded')
        active_space = report['structured_results']['active_space']
        self.assertEqual(active_space['ncas'], 2)
        self.assertEqual(active_space['nelecas'], 2)
        self.assertEqual(active_space['orbital_indices'], [0, 1])
        self.assertEqual(active_space['audit']['selected_candidate_method'], 'chemical_valence')
        self.assertIn('chemical-valence candidate', active_space['rationale'])
        self.assertIsNotNone(active_space['initial_mo_coeff'])
        candidates = {item['method']: item for item in active_space['audit']['candidate_active_spaces']}
        self.assertEqual(candidates['chemical_valence']['targets'], ['H 1s'])
        self.assertEqual(report['approval']['active_space_contract']['ncas'], 2)
        self.assertEqual(report['approval']['active_space_contract']['nelecas'], 2)

    def test_active_space_module_candidate_limit_is_applied(self):
        request = (
            '{"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g", '
            '"method": "hf", "outputs": ["energy"], '
            '"active_space": {"enabled": true, "selection_method": "manual", '
            '"ncas": 2, "nelecas": 2, "orbital_indices": [0, 1], "approved": false}, '
            '"workflow": {"module_config": {"molecular.active_space_audit": '
            '{"candidate_limit": 2, "require_approval": true}}}}'
        )
        report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

        audit = report['structured_results']['active_space']['audit']
        self.assertEqual(len(audit['candidate_active_spaces']), 2)
        self.assertEqual(audit['module_policy']['candidate_limit'], 2)
        self.assertTrue(audit['module_policy']['require_approval'])
        self.assertIn(
            audit['selected_candidate_method'],
            [item['method'] for item in audit['candidate_active_spaces']],
        )

    def test_cas_methods_require_active_space_contract(self):
        request = '{"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g", "method": "casci"}'
        report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

        self.assertEqual(report['execution_status'], 'blocked')
        self.assertTrue(any('active_space.ncas' in message for message in report['validation_errors']))
        self.assertTrue(any('active_space.nelecas' in message for message in report['validation_errors']))

    def test_unapproved_cas_contract_is_preserved_for_web_approval(self):
        request = (
            '{"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g", '
            '"method": "casscf", "outputs": ["energy"], '
            '"active_space": {"enabled": true, "selection_method": "manual", '
            '"ncas": 2, "nelecas": 2, "orbital_indices": [0, 1], "approved": false}}'
        )

        report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

        self.assertEqual(report['execution_status'], 'blocked')
        self.assertEqual(report['task_spec']['method']['name'], 'casscf')
        self.assertEqual(report['task_spec']['active_space']['ncas'], 2)
        self.assertEqual(report['task_spec']['active_space']['nelecas'], 2)
        self.assertFalse(report['task_spec']['active_space']['approved'])
        self.assertIn('active_space_not_approved', {
            item.get('code') for item in report['errors']
        })
        self.assertEqual(report['approval']['type'], 'active_space')
        self.assertEqual(report['approval']['source'], 'blocked_active_space_contract')
        self.assertEqual(report['approval']['active_space_contract']['ncas'], 2)
        self.assertEqual(report['approval']['active_space_contract']['nelecas'], 2)

    def test_casci_supports_minimal_closed_shell_active_space(self):
        request = (
            '{"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g", '
            '"method": "casci", "outputs": ["energy"], '
            '"active_space": {"enabled": true, "selection_method": "manual", '
            '"ncas": 2, "nelecas": 2, "orbital_indices": [0, 1], "approved": true}}'
        )
        report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

        self.assertEqual(report['execution_status'], 'succeeded')
        self.assertEqual(report['structured_results']['method'], 'casci')
        self.assertEqual(report['structured_results']['solver'], 'fci')
        self.assertIn('Solver=FCI', report['analysis_summary'])
        self.assertIn('cas_result', report['structured_results'])
        self.assertEqual(report['structured_results']['cas_result']['ncas'], 2)
        self.assertIsInstance(report['structured_results']['energy'], float)

    def test_open_shell_cas_defaults_to_rohf_spin_adapted_reference(self):
        request = (
            '{"atom": "H 0 0 0", "basis": "sto-3g", "spin": 1, '
            '"method": "casci", "outputs": ["energy"], '
            '"active_space": {"enabled": true, "selection_method": "manual", '
            '"ncas": 1, "nelecas": [1, 0], "orbital_indices": [0], "approved": true}}'
        )
        report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

        self.assertEqual(report['execution_status'], 'succeeded')
        self.assertEqual(report['applied_defaults'][0]['field'], 'method.restricted')
        self.assertTrue(report['applied_defaults'][0]['value'])
        self.assertEqual(report['applied_defaults'][0]['reason'], 'spin_adapted_cas_default')
        self.assertEqual(report['structured_results']['reference'], 'uhf')
        self.assertEqual(report['structured_results']['initial_reference'], 'uhf')
        self.assertEqual(report['structured_results']['cas_reference'], 'rohf')
        cas_result = report['structured_results']['cas_result']
        self.assertEqual(cas_result['reference'], 'rohf')
        self.assertEqual(cas_result['initial_reference'], 'uhf')
        self.assertEqual(cas_result['cas_reference'], 'rohf')
        self.assertTrue(cas_result['spin_adapted'])

    def test_casscf_supports_sc_nevpt2_post_cas_correction(self):
        request = (
            '{"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g", '
            '"method": "casscf", "outputs": ["energy"], '
            '"active_space": {"enabled": true, "selection_method": "manual", '
            '"ncas": 2, "nelecas": 2, "orbital_indices": [0, 1], "approved": true}, '
            '"post_cas": {"sc_nevpt2": {"enabled": true, "root": 0, "density_fit": true}}}'
        )
        report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

        self.assertEqual(report['execution_status'], 'succeeded')
        structured_results = report['structured_results']
        sc_nevpt2 = structured_results['post_cas_results']['sc_nevpt2']
        self.assertEqual(sc_nevpt2['method'], 'sc_nevpt2')
        self.assertIn('correction_energy', sc_nevpt2)
        self.assertIn('total_energy', sc_nevpt2)
        self.assertEqual(structured_results['energy_kind'], 'cas_energy')
        self.assertEqual(structured_results['reference'], 'uhf')
        self.assertEqual(structured_results['initial_reference'], 'uhf')
        self.assertEqual(structured_results['cas_reference'], 'rhf')
        self.assertTrue(structured_results['cas_spin_adapted'])
        self.assertEqual(structured_results['cas_result']['initial_reference'], 'uhf')
        self.assertEqual(structured_results['cas_result']['cas_reference'], 'rhf')
        self.assertEqual(structured_results['final_method'], 'casscf+sc_nevpt2')
        self.assertEqual(structured_results['final_energy'], sc_nevpt2['total_energy'])
        self.assertNotIn('sc_nevpt2', structured_results)
        self.assertNotIn('sc_nevpt2', structured_results['cas_result'])
        artifact_kinds = {artifact['kind'] for artifact in report['artifacts']}
        self.assertIn('sc_nevpt2', artifact_kinds)
        self.assertIn('SC-NEVPT2 correction=', report['analysis_summary'])

    def test_sc_nevpt2_blocks_unrestricted_cas_reference(self):
        request = (
            '{"atom": "H 0 0 0", "basis": "sto-3g", "spin": 1, '
            '"restricted": false, "method": "casci", "outputs": ["energy"], '
            '"active_space": {"enabled": true, "selection_method": "manual", '
            '"ncas": 1, "nelecas": [1, 0], "orbital_indices": [0], "approved": true}, '
            '"post_cas": {"sc_nevpt2": {"enabled": true, "root": 0, "density_fit": true}}}'
        )
        report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

        self.assertEqual(report['execution_status'], 'blocked')
        self.assertTrue(any('unrestricted CAS is not supported' in message for message in report['validation_errors']))

    def test_sc_nevpt2_blocks_excited_state_root_until_multistate_cas_supported(self):
        request = (
            '{"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g", '
            '"method": "casscf", "outputs": ["energy"], '
            '"active_space": {"enabled": true, "selection_method": "manual", '
            '"ncas": 2, "nelecas": 2, "orbital_indices": [0, 1], "approved": true}, '
            '"post_cas": {"sc_nevpt2": {"enabled": true, "root": 2, "density_fit": true}}}'
        )
        report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

        self.assertEqual(report['execution_status'], 'blocked')
        self.assertTrue(any('supports root=0 only' in message for message in report['validation_errors']))

    def test_cas_methods_support_unrestricted_reference_with_shared_active_orbitals(self):
        for method_name in ('casci', 'casscf'):
            with self.subTest(method=method_name):
                request = (
                    '{"atom": "H 0 0 0", "basis": "sto-3g", "spin": 1, '
                    '"restricted": false, "method": "%s", "outputs": ["energy"], '
                    '"active_space": {"enabled": true, "selection_method": "manual", '
                    '"ncas": 1, "nelecas": [1, 0], "orbital_indices": [0], "approved": true}}'
                ) % method_name
                report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

                self.assertEqual(report['execution_status'], 'succeeded')
                self.assertEqual(report['structured_results']['reference'], 'uhf')
                self.assertEqual(report['structured_results']['method'], method_name)
                cas_result = report['structured_results']['cas_result']
                self.assertEqual(cas_result['reference'], 'uhf')
                self.assertEqual(cas_result['ncas'], 1)
                self.assertEqual(cas_result['nelecas'], [1, 0])
                self.assertEqual(cas_result['active_orbital_indices'], [0])
                self.assertEqual(cas_result['active_orbital_index_spin_policy'], 'shared_alpha_beta')
                self.assertIsInstance(report['structured_results']['energy'], float)

    def test_cas_methods_support_unrestricted_spin_resolved_active_orbitals(self):
        for method_name in ('casci', 'casscf'):
            with self.subTest(method=method_name):
                request = (
                    '{"atom": "H 0 0 0", "basis": "sto-3g", "spin": 1, '
                    '"restricted": false, "method": "%s", "outputs": ["energy"], '
                    '"active_space": {"enabled": true, "selection_method": "manual", '
                    '"ncas": 1, "nelecas": [1, 0], '
                    '"orbital_indices": {"alpha": [0], "beta": [0]}, "approved": true}}'
                ) % method_name
                report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

                self.assertEqual(report['execution_status'], 'succeeded')
                cas_result = report['structured_results']['cas_result']
                self.assertEqual(report['structured_results']['reference'], 'uhf')
                self.assertEqual(cas_result['reference'], 'uhf')
                self.assertEqual(cas_result['active_orbital_indices'], {'alpha': [0], 'beta': [0]})
                self.assertEqual(cas_result['active_orbital_index_spin_policy'], 'spin_resolved_alpha_beta')
                self.assertEqual(report['structured_results']['active_space']['orbital_indices'], {'alpha': [0], 'beta': [0]})

    def test_spin_resolved_active_orbitals_require_unrestricted_reference(self):
        request = (
            '{"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g", '
            '"restricted": true, "method": "casscf", "outputs": ["energy"], '
            '"active_space": {"enabled": true, "selection_method": "manual", '
            '"ncas": 1, "nelecas": [1, 0], '
            '"orbital_indices": {"alpha": [0], "beta": [0]}, "approved": true}}'
        )
        report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

        self.assertEqual(report['execution_status'], 'blocked')
        self.assertTrue(any('restricted=false' in message for message in report['validation_errors']))

    def test_spin_resolved_active_orbitals_parse_from_text_form(self):
        request = (
            '{"atom": "H 0 0 0", "basis": "sto-3g", "spin": 1, '
            '"restricted": false, "method": "casci", "outputs": ["energy"], '
            '"active_space": {"enabled": true, "selection_method": "manual", '
            '"ncas": 1, "nelecas": [1, 0], '
            '"orbital_indices": "alpha:0; beta:0", "approved": true}}'
        )
        report = backend_workflow.execute_request(request, channel='test', locale='en')['task_report']

        self.assertEqual(report['execution_status'], 'succeeded')
        self.assertEqual(report['structured_results']['cas_result']['active_orbital_indices'], {'alpha': [0], 'beta': [0]})

    def test_append_message_ignores_blank_content(self):
        state = {'messages': []}

        backend_state.append_message(state, role='assistant', kind='summary', content='   ')

        self.assertEqual(state['messages'], [])


if __name__ == '__main__':
    unittest.main()
