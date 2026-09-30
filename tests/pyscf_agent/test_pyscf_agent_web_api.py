from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import pyscf_agent.pyscf_agent_web_api as web_api
from pyscf_agent.application import ExecutionTargetRegistry
from pyscf_agent.executors import JobHandle, JobState, JobStatus, LocalExecutor
from pyscf_agent.lifecycle import task_lifecycle_for_preparation


class DummyBackend:
    def __init__(self):
        self.calls = []

    def execute_request(self, request_text, *, channel='web', locale='zh', work_dir=None, run_id=None):
        self.calls.append((request_text, channel, locale, work_dir, run_id))
        return {
            'task_report': {
                'execution_status': 'stubbed',
                'request_text': request_text,
                'channel': channel,
                'locale': locale,
                'work_dir': work_dir,
                'run_id': run_id,
            }
        }


class ExplodingBackend:
    def execute_request(self, request_text, *, channel='web', locale='zh', work_dir=None, run_id=None):
        raise RuntimeError('backend exploded')


class ProfileExecutor:
    def __init__(self):
        self.calls = []

    def execute_task(self, request_text, **kwargs):
        self.calls.append((request_text, kwargs))
        return {'execution_status': 'succeeded'}

    def describe(self):
        return {
            'execution_mode': 'scheduler',
            'default_resource_limits': {'memory_mb': 8192},
            'resource_profile_options': [
                {'id': 'standard', 'label': 'Standard'},
                {'id': 'memory_intensive', 'label': 'Memory Intensive', 'memory_mb': 65536},
            ],
            'remote_host': 'must-not-be-public',
        }


class LifecycleExecutor:
    executor_id = 'lifecycle'
    execution_mode = 'test'

    def __init__(self):
        self.submit_calls = []
        self.current_state = JobState.RUNNING
        self.report_available = False
        self.handle = None
        self.report = {
            'execution_status': 'succeeded',
            'run_id': 'lifecycle-run',
            'messages': [],
        }

    def execute_task(self, request_text, **kwargs):
        return deepcopy(self.report)

    def submit_task(self, request_text, **kwargs):
        self.submit_calls.append((request_text, kwargs))
        self.handle = JobHandle(
            job_id=str(kwargs.get('run_id') or 'lifecycle-run'),
            executor_id=self.executor_id,
            run_id=str(kwargs.get('run_id') or 'lifecycle-run'),
            work_dir=str(kwargs.get('work_dir') or '/tmp/lifecycle'),
            submitted_at='2026-08-19T00:00:00+00:00',
        )
        return self.handle

    def status(self, handle):
        normalized = handle if isinstance(handle, JobHandle) else JobHandle.from_dict(handle)
        return JobStatus(
            handle=normalized,
            state=self.current_state,
            updated_at='2026-08-19T00:00:01+00:00',
            report_available=self.report_available,
            message='test lifecycle state',
        )

    def cancel(self, handle):
        self.current_state = JobState.CANCELLED
        self.report_available = False
        return self.status(handle)

    def fetch(self, handle):
        return deepcopy(self.report)

    def logs(self, handle):
        return []

    def artifacts(self, handle):
        return []

    def describe(self):
        return {
            'executor_id': self.executor_id,
            'execution_mode': self.execution_mode,
            'supports_cancel': True,
        }


class DummyLLMBuilder:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self.feedback_calls = []

    def build_prepared_request(self, messages, *, task_spec=None, request=None, locale='zh'):
        self.calls.append({
            'messages': messages,
            'task_spec': task_spec,
            'request': request,
            'locale': locale,
        })
        return self.result

    def build_execution_feedback(self, request_text, execution_report, locale='zh'):
        self.feedback_calls.append({
            'request_text': request_text,
            'execution_report': execution_report,
            'locale': locale,
        })
        return {'role': 'assistant', 'content': '如果你想收敛更稳，可以在提示词里明确要求更大基组或更具体的目标。'}

    def build_result_analysis(self, request_text, execution_report, locale='zh'):
        return '一、计算概况\n已完成结果分析。'


class ExplodingLLMBuilder:
    def build_prepared_request(self, messages, *, task_spec=None, request=None, locale='zh'):
        raise RuntimeError('builder exploded')


class WebApiTests(unittest.TestCase):
    def test_task_review_action_approves_active_space_lifecycle(self):
        lifecycle = task_lifecycle_for_preparation('awaiting_approval', entity_id='cas-task')

        status, _headers, body = web_api.handle_task_review_action_request(
            json.dumps({
                'action': 'approve',
                'review_type': 'active_space',
                'lifecycle': lifecycle,
            }).encode('utf-8')
        )

        payload = json.loads(body.decode('utf-8'))
        self.assertEqual(status.value, 200)
        self.assertEqual(payload['status'], 'approved')
        self.assertEqual(payload['lifecycle']['stage'], 'retry_ready')
        self.assertEqual(payload['lifecycle']['history'][-1]['event'], 'approval_granted')

        executor = LifecycleExecutor()
        registry = ExecutionTargetRegistry({'local': executor}, default_target='local')
        submit_status, _headers, submit_body = web_api.handle_run_submit_request(
            json.dumps({
                'execution_target': 'local',
                'prepared_request': '{"task_type":"molecular"}',
                'lifecycle': payload['lifecycle'],
            }).encode('utf-8'),
            execution_targets=registry,
        )
        submitted = json.loads(submit_body.decode('utf-8'))
        self.assertEqual(submit_status.value, 202)
        self.assertEqual(submitted['lifecycle']['stage'], 'queued')
        self.assertEqual(len(executor.submit_calls), 1)

    def test_run_submit_rejects_pending_review_before_executor_side_effect(self):
        executor = LifecycleExecutor()
        registry = ExecutionTargetRegistry({'local': executor}, default_target='local')
        lifecycle = task_lifecycle_for_preparation('awaiting_approval', entity_id='cas-task')

        status, _headers, body = web_api.handle_run_submit_request(
            json.dumps({
                'execution_target': 'local',
                'prepared_request': '{"task_type":"molecular"}',
                'lifecycle': lifecycle,
            }).encode('utf-8'),
            execution_targets=registry,
        )

        self.assertEqual(status.value, 400)
        self.assertIn("current stage is 'review_required'", json.loads(body.decode('utf-8'))['error'])
        self.assertEqual(executor.submit_calls, [])

    def test_submitted_run_lifecycle_accepts_every_calculation_family(self):
        task_specs = {
            'molecular': {
                'task_type': 'molecular',
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'method': 'hf',
            },
            'periodic': {
                'task_type': 'periodic',
                'method': 'hf',
                'periodic': {
                    'format': 'poscar',
                    'structure_text': 'H\n1.0\n1 0 0\n0 1 0\n0 0 1\nH\n1\nDirect\n0 0 0',
                },
            },
            'model_hamiltonian': {
                'task_type': 'model_hamiltonian',
                'model_hamiltonian_input_file': '/tmp/model-input.py',
                'solver': {'name': 'fci'},
            },
        }

        for task_type, task_spec in task_specs.items():
            with self.subTest(task_type=task_type):
                executor = LifecycleExecutor()
                registry = ExecutionTargetRegistry(
                    {'local': executor},
                    default_target='local',
                )
                prepared_request = json.dumps(task_spec)

                status, _headers, body = web_api.handle_run_submit_request(
                    json.dumps({
                        'execution_target': 'local',
                        'prepared_request': prepared_request,
                        'work_dir': '/tmp/lifecycle-work',
                        'run_id': f'{task_type}-run',
                    }).encode('utf-8'),
                    llm_request_builder=DummyLLMBuilder({}),
                    execution_targets=registry,
                )

                submitted = json.loads(body.decode('utf-8'))
                self.assertEqual(status.value, 202)
                self.assertEqual(submitted['status']['state'], 'queued')
                self.assertEqual(executor.submit_calls[0][0], prepared_request)
                self.assertEqual(
                    json.loads(executor.submit_calls[0][0])['task_type'],
                    task_type,
                )

    def test_submitted_run_lifecycle_supports_status_collection_and_cancel(self):
        executor = LifecycleExecutor()
        builder = DummyLLMBuilder({})
        registry = ExecutionTargetRegistry(
            {'cluster': executor},
            default_target='cluster',
        )
        base_payload = {
            'execution_target': 'cluster',
            'prepared_request': '{"basis": "sto-3g"}',
        }

        status, _headers, body = web_api.handle_run_submit_request(
            json.dumps({
                **base_payload,
                'work_dir': '/tmp/lifecycle-work',
                'run_id': 'lifecycle-run',
                'resource_profile': 'memory',
            }).encode('utf-8'),
            llm_request_builder=builder,
            execution_targets=registry,
        )

        submitted = json.loads(body.decode('utf-8'))
        self.assertEqual(status.value, 202)
        self.assertEqual(submitted['status']['state'], 'queued')
        self.assertEqual(executor.submit_calls[0][1]['resource_profile'], 'memory')
        handle = submitted['handle']

        status, _headers, body = web_api.handle_run_status_request(
            json.dumps({
                'execution_target': 'cluster',
                'handle': handle,
            }).encode('utf-8'),
            execution_targets=registry,
        )
        self.assertEqual(status.value, 200)
        self.assertEqual(json.loads(body.decode('utf-8'))['status']['state'], 'running')

        executor.current_state = JobState.COMPLETED
        executor.report_available = True
        status, _headers, body = web_api.handle_run_collect_request(
            json.dumps({
                **base_payload,
                'handle': handle,
            }).encode('utf-8'),
            llm_request_builder=builder,
            execution_targets=registry,
        )
        collected = json.loads(body.decode('utf-8'))
        self.assertEqual(status.value, 200)
        self.assertEqual(collected['execution_status'], 'succeeded')
        self.assertEqual(collected['messages'][-1]['role'], 'assistant')

        executor.current_state = JobState.RUNNING
        executor.report_available = False
        status, _headers, body = web_api.handle_run_cancel_request(
            json.dumps({
                'execution_target': 'cluster',
                'handle': handle,
            }).encode('utf-8'),
            execution_targets=registry,
        )
        self.assertEqual(status.value, 200)
        self.assertEqual(json.loads(body.decode('utf-8'))['status']['state'], 'cancelled')

    def test_handle_execution_targets_request_returns_public_registry(self):
        registry = ExecutionTargetRegistry(
            {'local': LocalExecutor(request_runner=DummyBackend().execute_request)},
            default_target='local',
            labels={'local': 'Local'},
        )

        status, _headers, body = web_api.handle_execution_targets_request(registry)

        payload = json.loads(body.decode('utf-8'))
        self.assertEqual(status.value, 200)
        self.assertEqual(payload['default_target'], 'local')
        self.assertEqual(payload['targets'][0]['label'], 'Local')

    def test_handle_resource_profiles_request_returns_safe_scheduler_limits(self):
        registry = ExecutionTargetRegistry(
            {'amarel': ProfileExecutor()},
            default_target='amarel',
        )

        status, _headers, body = web_api.handle_resource_profiles_request(
            registry,
            'amarel',
        )

        payload = json.loads(body.decode('utf-8'))
        self.assertEqual(status.value, 200)
        self.assertEqual(payload['execution_target'], 'amarel')
        self.assertEqual(payload['profiles'], [
            {'id': 'auto', 'label': 'Auto', 'memory_mb': 8192},
            {'id': 'standard', 'label': 'Standard'},
            {'id': 'memory_intensive', 'label': 'Memory Intensive', 'memory_mb': 65536},
        ])
        self.assertNotIn('remote_host', payload)

    def test_handle_run_request_forwards_selected_resource_profile(self):
        executor = ProfileExecutor()
        registry = ExecutionTargetRegistry(
            {'amarel': executor},
            default_target='amarel',
        )

        status, _headers, _body = web_api.handle_run_request(
            json.dumps({
                'prepared_request': '{"basis": "sto-3g"}',
                'execution_target': 'amarel',
                'resource_profile': 'memory_intensive',
            }).encode('utf-8'),
            execution_targets=registry,
        )

        self.assertEqual(status.value, 200)
        self.assertEqual(
            executor.calls[0][1]['resource_profile'],
            'memory_intensive',
        )

    def test_handle_run_request_routes_selected_execution_target(self):
        local_backend = DummyBackend()
        remote_backend = DummyBackend()
        registry = ExecutionTargetRegistry(
            {
                'local': LocalExecutor(request_runner=local_backend.execute_request),
                'amarel': LocalExecutor(request_runner=remote_backend.execute_request),
            },
            default_target='local',
        )

        status, _headers, _body = web_api.handle_run_request(
            json.dumps({
                'prepared_request': '{"basis": "sto-3g"}',
                'execution_target': 'amarel',
            }).encode('utf-8'),
            execution_targets=registry,
        )

        self.assertEqual(status.value, 200)
        self.assertEqual(local_backend.calls, [])
        self.assertEqual(len(remote_backend.calls), 1)

    def test_handle_run_request_rejects_unknown_execution_target(self):
        registry = ExecutionTargetRegistry(
            {'local': LocalExecutor(request_runner=DummyBackend().execute_request)},
            default_target='local',
        )

        status, _headers, body = web_api.handle_run_request(
            json.dumps({
                'prepared_request': '{"basis": "sto-3g"}',
                'execution_target': 'missing',
            }).encode('utf-8'),
            execution_targets=registry,
        )

        self.assertEqual(status.value, 400)
        self.assertIn('Available targets: local', json.loads(body.decode('utf-8'))['error'])

    def test_handle_capabilities_request_exposes_statuses(self):
        status, headers, body = web_api.handle_capabilities_request()

        payload = json.loads(body.decode('utf-8'))
        model_capabilities = {
            item['local_id']: item['contract']
            for item in payload['capabilities']['entries']
            if item['namespace'] == 'model_hamiltonian.model'
        }

        self.assertEqual(status.value, 200)
        self.assertEqual(headers['Content-Type'], 'application/json; charset=utf-8')
        self.assertEqual(model_capabilities['hubbard']['status'], 'executable')
        self.assertTrue(model_capabilities['hubbard']['backend_allowed'])
        self.assertEqual(model_capabilities['holstein_hubbard']['status'], 'design_only')
        self.assertFalse(model_capabilities['holstein_hubbard']['backend_allowed'])
        self.assertEqual(payload['capabilities']['schema'], 'pyscf-agent.registry.v3')
        self.assertNotIn('legacy', payload['capabilities'])

    def test_handle_run_request_prefers_prepared_request(self):
        backend = DummyBackend()
        builder = DummyLLMBuilder({})

        status, headers, body = web_api.handle_run_request(
            json.dumps({'prepared_request': '{"basis": "sto-3g"}'}).encode('utf-8'),
            task_executor=LocalExecutor(request_runner=backend.execute_request),
            llm_request_builder=builder,
        )

        payload = json.loads(body.decode('utf-8'))
        self.assertEqual(status.value, 200)
        self.assertEqual(headers['Content-Type'], 'application/json; charset=utf-8')
        self.assertEqual(backend.calls, [('{"basis": "sto-3g"}', 'web', 'en', None, None)])
        self.assertEqual(payload['execution_status'], 'stubbed')
        self.assertEqual(payload['messages'][-1]['role'], 'assistant')
        self.assertEqual(len(builder.feedback_calls), 1)

    def test_build_agent_request_preserves_model_hamiltonian_file_payload(self):
        request_text = web_api.build_agent_request({
            'task_type': 'model_hamiltonian',
            'model_hamiltonian_input_file': '/tmp/pyscf_model_hamiltonian_input.py',
            'solver': 'fci',
            'outputs': ['energy', 'strong_correlation_diagnostics'],
        })

        payload = json.loads(request_text)
        self.assertEqual(payload['task_type'], 'model_hamiltonian')
        self.assertEqual(payload['model_hamiltonian_input_file'], '/tmp/pyscf_model_hamiltonian_input.py')
        self.assertEqual(payload['solver'], 'fci')
        self.assertEqual(payload['outputs'], ['energy', 'strong_correlation_diagnostics'])

    def test_build_agent_request_omits_stale_inline_model_spec_when_file_is_present(self):
        request_text = web_api.build_agent_request({
            'task_type': 'model_hamiltonian',
            'model_hamiltonian_input_file': '/tmp/current_model_input.py',
            'model_hamiltonian': {
                'spec': {
                    'model': 'hubbard',
                    'nelec': [3, 3],
                    'sites': [],
                    'bonds': [],
                }
            },
            'solver': 'fci',
        })

        payload = json.loads(request_text)
        self.assertEqual(payload['model_hamiltonian_input_file'], '/tmp/current_model_input.py')
        self.assertNotIn('model_hamiltonian', payload)

    def test_build_agent_request_preserves_advanced_molecular_fields(self):
        request_text = web_api.build_agent_request({
            'task_spec': {
                'atom': 'H 0 0 0',
                'basis': 'sto-3g',
                'method': 'mp2',
                'charge': 0,
                'spin': 1,
                'restricted': False,
                'outputs': ['energy'],
            },
        })

        payload = json.loads(request_text)
        self.assertEqual(payload['charge'], 0)
        self.assertEqual(payload['spin'], 1)
        self.assertFalse(payload['restricted'])

    def test_build_agent_request_preserves_periodic_payload(self):
        periodic = {
            'structure_format': 'poscar',
            'structure_text': 'periodic structure',
            'basis': 'gth-szv',
            'pseudo': 'gth-pbe',
            'kmesh': [2, 2, 2],
            'dimension': 3,
        }

        request_text = web_api.build_agent_request({
            'task_spec': {
                'task_type': 'periodic',
                'periodic': periodic,
                'method': 'dft',
                'xc': 'pbe',
                'outputs': ['energy', 'band_gap'],
            },
        })

        payload = json.loads(request_text)
        self.assertEqual(payload['task_type'], 'periodic')
        self.assertEqual(payload['periodic'], periodic)
        self.assertEqual(payload['method'], 'dft')
        self.assertEqual(payload['xc'], 'pbe')
        self.assertEqual(payload['outputs'], ['energy', 'band_gap'])

    def test_handle_model_hamiltonian_preview_request_reads_structure_only_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / 'pyscf_model_hamiltonian_input.py'
            input_path.write_text(
                'model_spec = {\n'
                '  "schema": "pyscf-agent.model-hamiltonian.v1",\n'
                '  "model": "hubbard",\n'
                '  "dimension": 1,\n'
                '  "preset": "chain",\n'
                '  "boundary": "open",\n'
                '  "cell": {"origin": {"x": 0, "y": 0}, "a": {"x": 1, "y": 0}, "b": {"x": 0, "y": 1}},\n'
                '  "nelec": [1, 1],\n'
                '  "sites": [{"id": 0, "x": 0, "y": 0, "epsilon": 0, "U": 4}],\n'
                '  "bonds": []\n'
                '}\n',
                encoding='utf-8',
            )

            status, _headers, body = web_api.handle_model_hamiltonian_preview_request(
                json.dumps({'model_hamiltonian_input_file': str(input_path)}).encode('utf-8')
            )

        payload = json.loads(body.decode('utf-8'))
        self.assertEqual(status.value, 200)
        self.assertEqual(payload['preview']['model'], 'hubbard')
        self.assertEqual(payload['preview']['cell']['a']['x'], 1)
        self.assertEqual(payload['preview']['site_count'], 1)
        self.assertEqual(payload['preview']['bond_count'], 0)
        self.assertEqual(payload['preview']['graph_preview']['representation'], 'site-bond graph')
        self.assertEqual(payload['preview']['graph_preview']['nodes'][0]['id'], 0)
        self.assertNotIn('solver', payload['preview'])
        self.assertNotIn('dmet_validation', payload['preview'])

    def test_handle_periodic_structure_preview_request_parses_poscar(self):
        poscar = (
            'Helium cell\n'
            '1.0\n'
            '5 0 0\n'
            '0 5 0\n'
            '0 0 5\n'
            'He\n'
            '1\n'
            'Direct\n'
            '0 0 0\n'
        )

        status, headers, body = web_api.handle_periodic_structure_preview_request(
            json.dumps({
                'structure_format': 'poscar',
                'structure_text': poscar,
                'seekpath_symprec': 1e-5,
                'seekpath_reference_distance': 0.2,
            }).encode('utf-8')
        )

        payload = json.loads(body.decode('utf-8'))
        self.assertEqual(status.value, 200)
        self.assertEqual(headers['Content-Type'], 'application/json; charset=utf-8')
        self.assertEqual(payload['status'], 'ok')
        self.assertEqual(payload['preview']['formula'], 'He')
        self.assertEqual(payload['preview']['atom_count'], 1)
        self.assertEqual(payload['preview']['band_path_preview']['automatic_path'], 'GXMGRX,MR')
        self.assertEqual(
            set(payload['preview']['band_path_preview']['special_points_scaled']),
            {'G', 'M', 'R', 'X'},
        )
        self.assertEqual(payload['preview']['seekpath_preview']['spacegroup_number'], 221)
        self.assertGreater(payload['preview']['seekpath_preview']['explicit_point_count'], 0)

    def test_handle_periodic_structure_preview_request_validates_seekpath_controls(self):
        status, _headers, body = web_api.handle_periodic_structure_preview_request(
            json.dumps({
                'structure_format': 'poscar',
                'structure_text': (
                    'Helium cell\n1.0\n5 0 0\n0 5 0\n0 0 5\n'
                    'He\n1\nDirect\n0 0 0\n'
                ),
                'seekpath_symprec': 1.0,
                'seekpath_reference_distance': 0.001,
            }).encode('utf-8')
        )

        payload = json.loads(body.decode('utf-8'))
        self.assertEqual(status.value, 400)
        self.assertIn('SeeK-path symmetry tolerance', payload['error'])

    def test_handle_periodic_structure_preview_request_reports_parse_error(self):
        status, _headers, body = web_api.handle_periodic_structure_preview_request(
            json.dumps({
                'structure_format': 'poscar',
                'structure_text': 'not a POSCAR',
            }).encode('utf-8')
        )

        payload = json.loads(body.decode('utf-8'))
        self.assertEqual(status.value, 400)
        self.assertIn('Periodic structure preview failed', payload['error'])

    def test_handle_run_request_passes_work_dir_and_run_id(self):
        backend = DummyBackend()

        status, _headers, body = web_api.handle_run_request(
            json.dumps({
                'prepared_request': '{"basis": "sto-3g"}',
                'work_dir': '/tmp/pyscf-agent-work',
                'run_id': 'builder-run',
            }).encode('utf-8'),
            task_executor=LocalExecutor(request_runner=backend.execute_request),
        )

        payload = json.loads(body.decode('utf-8'))
        self.assertEqual(status.value, 200)
        self.assertEqual(backend.calls, [('{"basis": "sto-3g"}', 'web', 'en', '/tmp/pyscf-agent-work', 'builder-run')])
        self.assertEqual(payload['work_dir'], '/tmp/pyscf-agent-work')
        self.assertEqual(payload['run_id'], 'builder-run')

    def test_handle_run_request_infers_model_run_context_from_input_file(self):
        backend = DummyBackend()

        status, _headers, body = web_api.handle_run_request(
            json.dumps({
                'prepared_request': json.dumps({
                    'task_type': 'model_hamiltonian',
                    'model_hamiltonian_input_file': '/tmp/pyscf-agent-work/builder-run/pyscf_model_hamiltonian_input.py',
                    'solver': 'fci',
                }),
            }).encode('utf-8'),
            task_executor=LocalExecutor(request_runner=backend.execute_request),
        )

        payload = json.loads(body.decode('utf-8'))
        self.assertEqual(status.value, 200)
        self.assertEqual(backend.calls[0][3], '/tmp/pyscf-agent-work')
        self.assertEqual(backend.calls[0][4], 'builder-run')
        self.assertEqual(payload['work_dir'], '/tmp/pyscf-agent-work')
        self.assertEqual(payload['run_id'], 'builder-run')

    def test_handle_prepare_request_returns_builder_payload(self):
        builder = DummyLLMBuilder({
            'status': 'needs_clarification',
            'clarification_questions': ['请提供分子结构。'],
            'missing_fields': ['atom'],
            'messages': [{'role': 'assistant', 'content': '请提供分子结构。'}],
        })

        status, headers, body = web_api.handle_prepare_request(
            json.dumps({
                'messages': [{'role': 'user', 'content': '帮我算一下'}],
                'request': '帮我算一下',
                'task_spec': {'basis': 'sto-3g'},
            }).encode('utf-8'),
            llm_request_builder=builder,
        )

        payload = json.loads(body.decode('utf-8'))
        self.assertEqual(status.value, 200)
        self.assertEqual(headers['Content-Type'], 'application/json; charset=utf-8')
        self.assertEqual(payload['status'], 'needs_clarification')
        self.assertEqual(builder.calls[0]['task_spec'], {'basis': 'sto-3g'})

    def test_handle_active_space_probe_request_uses_shared_policy(self):
        status, headers, body = web_api.handle_active_space_probe_request(
            json.dumps({
                'target_solver': 'fci',
                'task_spec': {
                    'task_type': 'molecular',
                    'atom': 'H 0 0 0; H 0 0 0.74',
                    'basis': 'sto-3g',
                    'method': 'casscf',
                },
            }).encode('utf-8')
        )

        payload = json.loads(body.decode('utf-8'))
        self.assertEqual(status.value, 200)
        self.assertEqual(headers['Content-Type'], 'application/json; charset=utf-8')
        self.assertEqual(payload['probe_contract']['requested_strategy'], 'auto')
        self.assertEqual(payload['probe_request']['method'], 'hf')
        self.assertEqual(payload['probe_contract']['refinement_method'], 'mp2')
        self.assertEqual(
            payload['probe_request']['active_space']['target_method'],
            'casscf',
        )

    def test_handle_active_space_probe_request_rejects_non_auto_strategy(self):
        status, _headers, body = web_api.handle_active_space_probe_request(
            json.dumps({
                'strategy': 'fci',
                'task_spec': {'task_type': 'molecular'},
            }).encode('utf-8')
        )

        payload = json.loads(body.decode('utf-8'))
        self.assertEqual(status.value, 400)
        self.assertIn('fixed to auto', payload['error'])

    def test_handle_prepare_request_passes_english_locale(self):
        builder = DummyLLMBuilder({
            'status': 'needs_clarification',
            'clarification_questions': ['Please provide the molecular structure.'],
            'missing_fields': ['atom'],
            'messages': [{'role': 'assistant', 'content': 'Please provide the molecular structure.'}],
        })

        status, headers, body = web_api.handle_prepare_request(
            json.dumps({
                'messages': [{'role': 'user', 'content': 'Help me run this task'}],
                'request': 'Help me run this task',
                'task_spec': {'basis': 'sto-3g'},
                'locale': 'en',
            }).encode('utf-8'),
            llm_request_builder=builder,
        )

        payload = json.loads(body.decode('utf-8'))
        self.assertEqual(status.value, 200)
        self.assertEqual(headers['Content-Type'], 'application/json; charset=utf-8')
        self.assertEqual(payload['clarification_questions'][0], 'Please provide the molecular structure.')
        self.assertEqual(builder.calls[0]['locale'], 'en')

    def test_handle_run_request_passes_english_locale(self):
        backend = DummyBackend()
        builder = DummyLLMBuilder({})

        status, headers, body = web_api.handle_run_request(
            json.dumps({'prepared_request': '{"basis": "sto-3g"}', 'locale': 'en'}).encode('utf-8'),
            task_executor=LocalExecutor(request_runner=backend.execute_request),
            llm_request_builder=builder,
        )

        payload = json.loads(body.decode('utf-8'))
        self.assertEqual(status.value, 200)
        self.assertEqual(headers['Content-Type'], 'application/json; charset=utf-8')
        self.assertEqual(backend.calls, [('{"basis": "sto-3g"}', 'web', 'en', None, None)])
        self.assertEqual(payload['locale'], 'en')
        self.assertEqual(builder.feedback_calls[0]['locale'], 'en')

    def test_handle_prepare_request_returns_json_error_on_builder_exception(self):
        status, headers, body = web_api.handle_prepare_request(
            json.dumps({
                'messages': [{'role': 'user', 'content': '计算丙烷的性质'}],
                'request': '计算丙烷的性质',
                'task_spec': {},
            }).encode('utf-8'),
            llm_request_builder=ExplodingLLMBuilder(),
        )

        payload = json.loads(body.decode('utf-8'))
        self.assertEqual(status.value, 500)
        self.assertEqual(headers['Content-Type'], 'application/json; charset=utf-8')
        self.assertNotIn('builder exploded', payload['error'])
        self.assertIn('Internal server error', payload['error'])

    def test_handle_run_request_returns_json_error_on_backend_exception(self):
        status, headers, body = web_api.handle_run_request(
            json.dumps({'prepared_request': '{"basis": "sto-3g"}'}).encode('utf-8'),
            task_executor=LocalExecutor(request_runner=ExplodingBackend().execute_request),
            llm_request_builder=None,
        )

        payload = json.loads(body.decode('utf-8'))
        self.assertEqual(status.value, 500)
        self.assertEqual(headers['Content-Type'], 'application/json; charset=utf-8')
        self.assertNotIn('backend exploded', payload['error'])
        self.assertIn('Internal server error', payload['error'])

    def test_handle_result_analysis_request_returns_formal_analysis(self):
        builder = DummyLLMBuilder({})

        status, headers, body = web_api.handle_result_analysis_request(
            json.dumps({
                'prepared_request': '{"basis": "sto-3g"}',
                'execution_report': {
                    'execution_status': 'succeeded',
                    'analysis_summary': '任务成功完成',
                },
            }).encode('utf-8'),
            llm_request_builder=builder,
        )

        payload = json.loads(body.decode('utf-8'))
        self.assertEqual(status.value, 200)
        self.assertEqual(headers['Content-Type'], 'application/json; charset=utf-8')
        self.assertIn('一、计算概况', payload['result_analysis'])


if __name__ == '__main__':
    unittest.main()
