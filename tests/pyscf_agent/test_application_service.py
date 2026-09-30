from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import pyscf_agent.pyscf_agent_cli as pyscf_agent_cli
from pyscf_agent.application import (
    CalculationApplicationService,
    CalculationFeatureUnavailableError,
)
from pyscf_agent.executors import LocalExecutor
from pyscf_agent.lifecycle import task_lifecycle_for_preparation
from pyscf_agent.backend.model_hamiltonian.operations import apply_model_operations
from pyscf_agent.backend.model_hamiltonian.solver import load_model_spec_from_file


class _Backend:
    def __init__(self):
        self.calls = []

    def execute_request(self, request_text, **kwargs):
        self.calls.append((request_text, kwargs))
        return {'task_report': {'execution_status': 'succeeded'}}


class _RequestBuilder:
    def __init__(self):
        self.feedback_calls = []

    @staticmethod
    def build_prepared_request(messages, *, task_spec=None, request=None, locale='zh'):
        return {
            'messages': messages,
            'task_spec': task_spec,
            'request': request,
            'locale': locale,
        }

    def build_execution_feedback(self, request_text, report, locale='zh'):
        self.feedback_calls.append((request_text, report, locale))
        return {'role': 'assistant', 'content': 'execution feedback'}

    @staticmethod
    def build_result_analysis(request_text, report, locale='zh'):
        return '{0}:{1}:{2}'.format(request_text, report['execution_status'], locale)

    @staticmethod
    def build_model_hamiltonian_operations(request, model_spec, **_kwargs):
        return {
            'operations': [{
                'op': 'set_site_parameter',
                'site': 1,
                'parameter': 'epsilon',
                'value': -0.75,
            }],
            'summary': 'Set the onsite energy of site 1.',
            'clarification_questions': [],
        }


class CalculationApplicationServiceTests(unittest.TestCase):
    def test_active_space_probe_auto_starts_from_scf_and_refines_conditionally(self):
        prepared = CalculationApplicationService().prepare_active_space_probe(
            {
                'task_type': 'molecular',
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'method': 'casscf',
                'active_space': {
                    'enabled': True,
                    'selection_method': 'avas',
                    'avas_targets': ['H 1s'],
                },
                'solver': {'name': 'block2_dmrg', 'options': {'nroots': 2}},
            },
            target_solver='block2_dmrg',
            target_solver_options={'nroots': 2},
        )

        probe = prepared['probe_request']
        contract = prepared['probe_contract']
        self.assertEqual(prepared['status'], 'ready')
        self.assertEqual(contract['requested_strategy'], 'auto')
        self.assertEqual(contract['probe_method'], 'hf')
        self.assertEqual(contract['refinement_method'], 'mp2')
        self.assertEqual(contract['selection_method'], 'avas')
        self.assertEqual(probe['method'], 'hf')
        self.assertNotIn('solver', probe)
        self.assertEqual(probe['active_space']['target_method'], 'casscf')
        self.assertEqual(probe['active_space']['target_solver'], 'block2_dmrg')
        self.assertEqual(
            probe['workflow']['module_config']['molecular.active_space_probe']['probe_method'],
            'hf',
        )
        self.assertEqual(
            probe['workflow']['module_config'][
                'molecular.active_space_probe_refinement'
            ]['refinement_method'],
            'mp2',
        )

    def test_task_review_approval_uses_the_authoritative_lifecycle(self):
        lifecycle = task_lifecycle_for_preparation('awaiting_approval', entity_id='cas-task')

        approved = CalculationApplicationService.apply_review_action(
            lifecycle,
            'approve',
            review_type='active_space',
        )

        self.assertEqual(approved['stage'], 'retry_ready')
        self.assertEqual(approved['history'][-1]['event'], 'approval_granted')
        self.assertEqual(approved['history'][-1]['details']['review_type'], 'active_space')

    def test_invalid_lifecycle_is_rejected_before_executor_submission(self):
        backend = _Backend()
        service = CalculationApplicationService(
            task_executor=LocalExecutor(request_runner=backend.execute_request),
        )
        lifecycle = task_lifecycle_for_preparation('awaiting_approval', entity_id='cas-task')

        with self.assertRaises(ValueError):
            service.submit_request_with_lifecycle(
                '{"task_type":"molecular"}',
                lifecycle=lifecycle,
                run_id='must-not-run',
            )

        self.assertEqual(backend.calls, [])

    def test_prepare_execute_and_analyze_share_one_protocol_neutral_boundary(self):
        backend = _Backend()
        builder = _RequestBuilder()
        service = CalculationApplicationService(
            task_executor=LocalExecutor(request_runner=backend.execute_request),
            llm_request_builder=builder,
        )

        prepared = service.prepare_request(
            messages=[{'role': 'user', 'content': 'run H2'}],
            task_spec={'atom': 'H 0 0 0; H 0 0 0.74'},
            locale='en',
        )
        report = service.execute_request(
            '{"task_type":"molecular"}',
            work_dir='runs',
            run_id='application-test',
            locale='en',
        )
        analysis = service.analyze_results('analyze', report, locale='en')

        self.assertEqual(prepared['locale'], 'en')
        self.assertEqual(prepared['lifecycle']['scope'], 'task')
        self.assertEqual(backend.calls[0][1]['run_id'], 'application-test')
        self.assertEqual(report['messages'][0]['content'], 'execution feedback')
        self.assertEqual(report['lifecycle']['stage'], 'succeeded')
        self.assertEqual(len(builder.feedback_calls), 1)
        self.assertEqual(analysis, 'analyze:succeeded:en')

    def test_missing_optional_dependencies_raise_typed_errors(self):
        service = CalculationApplicationService()

        with self.assertRaises(CalculationFeatureUnavailableError):
            service.prepare_request(messages=[])
        with self.assertRaises(CalculationFeatureUnavailableError):
            service.execute_request('{}')
        with self.assertRaises(CalculationFeatureUnavailableError):
            service.submit_request('{}')
        with self.assertRaises(CalculationFeatureUnavailableError):
            service.analyze_results('analyze', {})

    def test_submitted_job_lifecycle_is_available_through_application_service(self):
        backend = _Backend()
        service = CalculationApplicationService(
            task_executor=LocalExecutor(request_runner=backend.execute_request),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            handle = service.submit_request(
                '{"task_type":"molecular"}',
                work_dir=tmpdir,
                run_id='application-job-test',
                locale='en',
            )
            status = service.job_status(handle.to_dict())
            report = service.fetch_job(handle)
            logs = service.job_logs(handle)
            artifacts = service.job_artifacts(handle)
            envelope_path = Path(tmpdir) / 'submitted-job.json'
            envelope_path.write_text(
                json.dumps({'handle': handle.to_dict(), 'status': status.to_dict()}),
                encoding='utf-8',
            )
            loaded_handle = pyscf_agent_cli.load_job_handle(str(envelope_path))

        self.assertEqual(status.state.value, 'completed')
        self.assertEqual(status.task_status, 'succeeded')
        self.assertEqual(report['execution_status'], 'succeeded')
        self.assertEqual(logs, [])
        self.assertEqual(
            [item['kind'] for item in artifacts],
            ['job-state', 'task-report'],
        )
        self.assertEqual(loaded_handle, handle)

    def test_capabilities_and_previews_use_injected_providers(self):
        service = CalculationApplicationService(
            capability_provider=lambda: {'methods': ['hf']},
            model_spec_loader=lambda _path: {
                'model': 'hubbard',
                'sites': [{'id': 0}],
                'bonds': [],
                'nelec': [1, 0],
            },
            periodic_preview_builder=lambda source, fmt, **kwargs: {
                'source': source,
                'format': fmt,
                'options': kwargs,
            },
        )

        self.assertEqual(service.capabilities(), {'methods': ['hf']})
        self.assertEqual(service.preview_model_hamiltonian('model.py')['site_count'], 1)
        periodic = service.preview_periodic_structure('POSCAR', 'poscar', seekpath_symprec=1e-4)
        self.assertEqual(periodic['source'], 'POSCAR')
        self.assertEqual(periodic['options']['seekpath_symprec'], 1e-4)

    def test_model_preview_reuses_dmet_validation_for_fixed_spin_sector(self):
        sites = [
            {'id': index, 'x': float(index), 'y': 0.0, 'epsilon': 0.0, 'U': 4.0}
            for index in range(4)
        ]
        bonds = [
            {'source': index, 'target': (index + 1) % 4, 't': -1.0, 'V': 0.0}
            for index in range(4)
        ]
        service = CalculationApplicationService(
            model_spec_loader=lambda _path: {
                'model': 'hubbard',
                'representation': 'finite_cluster',
                'dimension': 1,
                'preset': 'ring',
                'boundary': 'periodic',
                'sites': sites,
                'bonds': bonds,
                'nelec': [3, 1],
            },
        )

        preview = service.preview_model_hamiltonian(
            'model.py',
            solver={
                'name': 'dmet',
                'options': {
                    'reference': 'unrestricted',
                    'impurity_shape': [2],
                },
            },
        )

        validation = preview['dmet_validation']
        self.assertTrue(validation['valid'])
        self.assertEqual(validation['execution_mode'], 'translational')
        self.assertEqual(validation['effective_reference'], 'unrestricted')
        self.assertEqual(validation['spin_sector']['nalpha'], 3)
        self.assertEqual(validation['spin_sector']['nbeta'], 1)
        self.assertEqual(validation['spin_sector']['libdmet_sz'], 2)
        self.assertEqual(validation['spin_sector']['physical_sz'], 1.0)

    def test_save_model_input_uses_run_artifact_layout(self):
        service = CalculationApplicationService()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = service.save_model_hamiltonian_input(
                'model_spec = {"model": "hubbard"}\n',
                work_dir=tmpdir,
                run_id='builder-service-test',
            )

            path = Path(result['path'])
            self.assertEqual(path.parent, Path(os.path.abspath(tmpdir)) / 'builder-service-test')
            self.assertTrue(path.is_file())
            self.assertEqual(result['run_id'], 'builder-service-test')

    def test_model_message_creates_a_derived_input_with_a_change_audit(self):
        original = {
            'model': 'hubbard',
            'representation': 'finite_cluster',
            'dimension': 1,
            'boundary': 'open',
            'sites': [
                {'id': 0, 'x': 0.0, 'y': 0.0, 'epsilon': 0.0, 'U': 4.0},
                {'id': 1, 'x': 1.0, 'y': 0.0, 'epsilon': 0.0, 'U': 4.0},
            ],
            'bonds': [{'id': 0, 'source': 0, 'target': 1, 't': -1.0, 'V': 0.0}],
            'nelec': [1, 1],
        }
        service = CalculationApplicationService(
            llm_request_builder=_RequestBuilder(),
            model_spec_loader=lambda _path: original,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            prepared = service.prepare_model_hamiltonian_edit(
                {
                    'task_type': 'model_hamiltonian',
                    'model_hamiltonian_input_file': '/source/model.py',
                    'solver': {'name': 'fci'},
                    'outputs': ['energy'],
                },
                'Set site 1 energy to -0.75.',
                locale='en',
                work_dir=tmpdir,
            )
            derived = load_model_spec_from_file(
                prepared['structured_request']['model_hamiltonian_input_file']
            )

        self.assertEqual(prepared['status'], 'ready')
        self.assertEqual(prepared['lifecycle']['stage'], 'prepared')
        self.assertEqual(original['sites'][1]['epsilon'], 0.0)
        self.assertEqual(derived['sites'][1]['epsilon'], -0.75)
        self.assertEqual(prepared['changes'], [{
            'scope': 'site',
            'id': 1,
            'parameter': 'epsilon',
            'before': 0.0,
            'after': -0.75,
        }])

    def test_generic_site_defect_parameter_value_is_executed(self):
        edited = apply_model_operations(
            {
                'sites': [{'id': 0, 'epsilon': 0.25, 'U': 4.0}],
                'bonds': [],
            },
            [{
                'op': 'add_site_defect',
                'site': 0,
                'parameter': 'epsilon',
                'value': -0.5,
            }],
        )

        self.assertEqual(edited['sites'][0]['epsilon'], -0.25)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
