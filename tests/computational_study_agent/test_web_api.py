from __future__ import annotations

import copy
import tempfile
import unittest.mock

from computational_study_agent import StudyReport
from computational_study_agent.planner import build_study_plan
from computational_study_agent.application import StudyApplicationService, StudyPlanResult
from computational_study_agent.execution_receipts import StudyExecutionInterrupted
from tests.computational_study_agent.support import (
    StudyAgentTestCase,
    hubbard_dimer_spec,
    json_dumps,
    json_loads,
)
from computational_study_agent.web_api import (
    handle_study_adaptive_plan_request,
    handle_study_adaptive_run_request,
    handle_study_execution_collect_request,
    handle_study_execution_status_request,
    handle_study_plan_request,
    handle_study_result_analysis_request,
    handle_study_review_action_request,
    handle_study_run_request,
)


class PlannerWebApiTests(StudyAgentTestCase):
    def test_study_review_action_api_requires_action_id(self):
        status, _headers, body = handle_study_review_action_request(json_dumps({}))

        self.assertEqual(status.value, 400)
        self.assertIn('action_id', json_loads(body)['error'])

    def test_study_plan_api_returns_expanded_cases(self):
        status, _headers, body = handle_study_plan_request(json_dumps({
            'study_spec': {
                'name': 'h2-methods',
                'objective': 'compare',
                'system_type': 'molecular',
                'base_task': {
                    'atom': 'H 0 0 0; H 0 0 0.74',
                    'basis': 'sto-3g',
                    'outputs': ['energy'],
                },
                'sweep': {'method': ['hf', 'mp2']},
                'observables': ['energy'],
            }
        }))

        payload = json_loads(body)
        self.assertEqual(status.value, 200)
        self.assertEqual(len(payload['plan']['cases']), 2)
        self.assertIn('validation_issues', payload)

    def test_study_plan_api_accepts_hamiltonian_dataset_preset(self):
        seeds = [
            {
                'molecule_id': 'mol-{0:03d}'.format(index),
                'geometry_id': 'seed',
                'atomic_numbers': [1, 1],
                'positions': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
                'coordinate_unit': 'Angstrom',
            }
            for index in range(2)
        ]
        status, _headers, body = handle_study_plan_request(json_dumps({
            'planner_template': 'hamiltonian_dataset',
            'dataset_spec': {
                'dataset_id': 'web-dataset',
                'name': 'Web dataset',
                'target_molecule_count': 2,
            },
            'seed_geometries': seeds,
        }))

        payload = json_loads(body)
        self.assertEqual(status.value, 200)
        self.assertEqual(len(payload['plan']['cases']), 2)
        self.assertEqual(
            payload['plan']['comparison']['mode'],
            'hamiltonian_dataset_assembly',
        )
        self.assertIn(
            'study.hamiltonian_dataset_assembly',
            payload['plan']['workflow_configuration']['execution_order'],
        )

    def test_study_plan_api_rejects_wrong_dataset_seed_count(self):
        status, _headers, body = handle_study_plan_request(json_dumps({
            'planner_template': 'hamiltonian_dataset',
            'dataset_spec': {
                'dataset_id': 'web-dataset',
                'name': 'Web dataset',
                'target_molecule_count': 2,
            },
            'seed_geometries': [{
                'molecule_id': 'mol-001',
                'geometry_id': 'seed',
                'atomic_numbers': [1, 1],
                'positions': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            }],
        }))

        self.assertEqual(status.value, 400)
        self.assertIn('Expected exactly 2 seed geometries', json_loads(body)['error'])

    def test_study_plan_api_uses_selected_execution_resource_profile(self):
        spec = {
            'name': 'remote-plan',
            'objective': 'bind scheduler resources',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'fci'},
            'observables': ['energy'],
        }
        plan = build_study_plan(spec)
        service = unittest.mock.Mock(spec=StudyApplicationService)
        service.prepare_plan.return_value = StudyPlanResult(plan=plan, validation_issues=())
        with unittest.mock.patch(
            'computational_study_agent.web_api.get_study_application_service',
            return_value=service,
        ) as get_service:
            status, _headers, _body = handle_study_plan_request(json_dumps({
                'study_spec': spec,
                'execution_target': 'amarel',
                'resource_profile': 'memory_intensive',
            }))

        self.assertEqual(status.value, 200)
        get_service.assert_called_once_with('amarel')
        self.assertEqual(
            service.prepare_plan.call_args.kwargs['resource_profile'],
            'memory_intensive',
        )

    def test_study_plan_api_returns_validation_issues_for_invalid_spec(self):
        status, _headers, body = handle_study_plan_request(json_dumps({
            'study_spec': {
                'name': 'bad-study',
                'objective': 'bad',
                'system_type': 'model_hamiltonian',
                'base_model_spec': hubbard_dimer_spec(),
                'case_design': {
                    'mode': 'grid',
                    'variables': {'site_id': [0]},
                    'template': {
                        'operations': [{'op': 'bad_op', 'site': '$site_id'}],
                    },
                },
                'observables': ['energy'],
            }
        }))

        payload = json_loads(body)
        self.assertEqual(status.value, 400)
        self.assertIn('validation_issues', payload)
        self.assertTrue(any(item['code'] == 'unsupported_model_operation' for item in payload['validation_issues']))

    def test_study_plan_api_routes_missing_casscf_space_to_shared_probe(self):
        status, _headers, body = handle_study_plan_request(json_dumps({
            'study_spec': {
                'name': 'h2-casscf-scan',
                'objective': 'scan H2 with CASSCF',
                'system_type': 'molecular',
                'base_task': {
                    'atom': 'H 0 0 0; H 0 0 0.74',
                    'basis': 'sto-3g',
                    'method': 'casscf',
                    'restricted': True,
                },
                'case_design': {
                    'mode': 'grid',
                    'variables': {'bond_length': [0.74, 1.20]},
                    'template': {
                        'request_updates': {
                            'atom': 'H 0 0 0; H 0 0 $bond_length',
                        },
                    },
                },
                'observables': ['energy'],
            },
        }))

        payload = json_loads(body)
        self.assertEqual(status.value, 200)
        self.assertEqual(payload['status'], 'active_space_probe')
        self.assertEqual(len(payload['active_space_probe_plan']['cases']), 2)
        self.assertTrue(all(
            case['request']['method'] == 'hf'
            for case in payload['active_space_probe_plan']['cases']
        ))

    def test_study_run_api_executes_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = build_study_plan({
                'name': 'dimer-u',
                'objective': 'sweep',
                'system_type': 'model_hamiltonian',
                'base_model_spec': hubbard_dimer_spec(),
                'base_task': {'solver': 'fci'},
                'sweep': {'U': [2]},
                'observables': ['energy'],
            })
            status, _headers, body = handle_study_run_request(json_dumps({
                'plan': plan.to_dict(),
                'work_dir': tmpdir,
                'locale': 'en',
            }))

        payload = json_loads(body)
        self.assertEqual(status.value, 200)
        self.assertEqual(payload['report']['status'], 'succeeded')
        self.assertEqual(len(payload['report']['comparison_table']), 1)

    def test_study_run_api_forwards_resume_controls(self):
        plan = self._resumable_molecular_plan()
        report = StudyReport(
            study_id=plan.study_id,
            name=plan.name,
            objective=plan.objective,
            system_type='molecular',
            status='completed_with_issues',
            work_dir='runs/test',
            cases=[],
            comparison_table=[],
        )
        service = unittest.mock.Mock(spec=StudyApplicationService)
        service.run_study.return_value = report
        with unittest.mock.patch(
            'computational_study_agent.web_api.get_study_application_service',
            return_value=service,
        ):
            status, _headers, body = handle_study_run_request(json_dumps({
                'plan': plan.to_dict(),
                'resume': False,
                'rerun_case_ids': ['case-0001'],
                'rerun_statuses': ['unconverged'],
                'max_case_attempts': 3,
            }))

        self.assertEqual(status.value, 200)
        self.assertEqual(json_loads(body)['report']['study_id'], plan.study_id)
        self.assertFalse(service.run_study.call_args.kwargs['resume'])
        self.assertEqual(service.run_study.call_args.kwargs['rerun_case_ids'], ['case-0001'])
        self.assertEqual(service.run_study.call_args.kwargs['rerun_statuses'], ['unconverged'])
        self.assertEqual(service.run_study.call_args.kwargs['max_case_attempts'], 3)

    def test_study_run_api_returns_the_complete_study_report_directly(self):
        plan = self._resumable_molecular_plan()
        report = StudyReport(
            study_id=plan.study_id, name=plan.name, objective=plan.objective,
            system_type='molecular', status='succeeded', work_dir='runs/study',
            cases=[], comparison_table=[],
        )
        service = unittest.mock.Mock(spec=StudyApplicationService)
        service.run_study.return_value = report
        with unittest.mock.patch(
            'computational_study_agent.web_api.get_study_application_service', return_value=service,
        ):
            status, _headers, body = handle_study_run_request(json_dumps({
                'plan': {**plan.to_dict(), '_review_kind': 'path_refinement'},
                'study_report': report.to_dict(), 'resume': False,
            }))
        self.assertEqual(status.value, 200)
        self.assertEqual(json_loads(body)['report']['study_id'], plan.study_id)
        self.assertEqual(service.run_study.call_args.kwargs['study_report'], report.to_dict())


    def test_study_run_api_returns_probe_review_from_application_service(self):
        plan = self._resumable_molecular_plan()
        plan_payload = plan.to_dict()
        plan_payload['workflow_provenance'] = {
            'direct_active_space_probe': {
                'schema': 'pyscf-agent/direct-active-space-probe@1',
                'target_cases': [plan_payload['cases'][0]],
                'probe_case_ids': [plan_payload['cases'][0]['case_id']],
            },
        }
        probe_report = StudyReport(
            study_id=plan.study_id,
            name=plan.name,
            objective=plan.objective,
            system_type='molecular',
            status='succeeded',
            work_dir='runs/probe',
            cases=[],
            comparison_table=[],
        )
        review_report = {
            'study_id': plan.study_id,
            'status': 'pending_review',
            'adaptive': {
                'mode': 'direct_casscf_review',
                'workflow': {'stage': 'active_space_review_required'},
            },
        }
        service = unittest.mock.Mock(spec=StudyApplicationService)
        service.run_study.return_value = review_report
        with unittest.mock.patch(
            'computational_study_agent.web_api.get_study_application_service',
            return_value=service,
        ):
            status, _headers, body = handle_study_run_request(json_dumps({
                'plan': plan_payload,
                'resume': False,
            }))

        payload = json_loads(body)
        self.assertEqual(status.value, 200)
        self.assertEqual(payload['report']['status'], 'pending_review')
        self.assertEqual(
            payload['report']['adaptive']['workflow']['stage'],
            'active_space_review_required',
        )
        service.build_active_space_review_from_probe.assert_not_called()

    def test_study_run_api_selects_execution_target(self):
        plan = self._resumable_molecular_plan()
        report = StudyReport(
            study_id=plan.study_id,
            name=plan.name,
            objective=plan.objective,
            system_type='molecular',
            status='succeeded',
            work_dir='runs/test',
            cases=[],
            comparison_table=[],
        )
        service = unittest.mock.Mock(spec=StudyApplicationService)
        service.run_study.return_value = report
        with unittest.mock.patch(
            'computational_study_agent.web_api.get_study_application_service',
            return_value=service,
        ) as get_service:
            status, _headers, _body = handle_study_run_request(json_dumps({
                'plan': plan.to_dict(),
                'execution_target': 'amarel',
                'resource_profile': 'memory_intensive',
            }))

        self.assertEqual(status.value, 200)
        get_service.assert_called_once_with('amarel')
        self.assertEqual(
            service.run_study.call_args.kwargs['resource_profile'],
            'memory_intensive',
        )

    def test_study_run_api_returns_recoverable_execution_after_interruption(self):
        plan = self._resumable_molecular_plan()
        execution = {
            'status': 'results_available',
            'expected': 1,
            'terminal': 1,
            'report_available': 1,
            'can_collect': True,
        }
        service = unittest.mock.Mock(spec=StudyApplicationService)
        service.run_study.side_effect = StudyExecutionInterrupted(
            'connection interrupted',
            {'study_id': plan.study_id},
        )
        service.inspect_execution.return_value = execution
        with unittest.mock.patch(
            'computational_study_agent.web_api.get_study_application_service',
            return_value=service,
        ):
            status, _headers, body = handle_study_run_request(json_dumps({
                'plan': plan.to_dict(),
                'work_dir': '/local/runs',
                'execution_target': 'amarel',
            }))

        payload = json_loads(body)
        self.assertEqual(status.value, 202)
        self.assertEqual(payload['status'], 'execution_interrupted')
        self.assertTrue(payload['execution']['can_collect'])
        service.inspect_execution.assert_called_once_with(
            plan.study_id,
            work_dir='/local/runs',
        )

    def test_execution_status_and_collect_resume_existing_adaptive_study(self):
        execution = {
            'status': 'results_available',
            'expected': 3,
            'terminal': 3,
            'report_available': 3,
            'can_collect': True,
        }
        report = {
            'study_id': 'adaptive-recovery',
            'status': 'succeeded',
            'adaptive': {'mode': 'adaptive_scan'},
        }
        service = unittest.mock.Mock(spec=StudyApplicationService)
        service.inspect_execution.return_value = execution
        service.collect_adaptive_study.return_value = report
        with unittest.mock.patch(
            'computational_study_agent.web_api.get_study_application_service',
            return_value=service,
        ):
            status, _headers, body = handle_study_execution_status_request(json_dumps({
                'study_id': 'adaptive-recovery',
                'work_dir': '/local/runs',
                'execution_target': 'amarel',
            }))
            collect_status, _headers, collect_body = handle_study_execution_collect_request(json_dumps({
                'study_id': 'adaptive-recovery',
                'mode': 'adaptive',
                'work_dir': '/local/runs',
                'execution_target': 'amarel',
                'study_spec': {
                    'name': 'resume',
                    'objective': 'resume adaptive scan',
                    'system_type': 'molecular',
                },
            }))

        self.assertEqual(status.value, 200)
        self.assertTrue(json_loads(body)['execution']['can_collect'])
        self.assertEqual(collect_status.value, 200)
        self.assertEqual(json_loads(collect_body)['report']['study_id'], 'adaptive-recovery')
        self.assertEqual(
            service.collect_adaptive_study.call_args.kwargs['study_id'],
            'adaptive-recovery',
        )

        service.run_adaptive_study.assert_not_called()
        service.run_study.assert_not_called()

    def test_study_adaptive_plan_api_rejects_model_hamiltonian(self):
        study_spec = {
            'name': 'adaptive-api-dimer',
            'objective': 'adaptive api test',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'ccsd'},
            'sweep': {'U': [0, 4, 8]},
            'observables': ['energy'],
        }
        status, _headers, body = handle_study_adaptive_plan_request(json_dumps({
            'study_spec': study_spec,
            'adaptive': {'initial_scan_strategy': 'auto'},
        }))
        plan_payload = json_loads(body)

        self.assertEqual(status.value, 400)
        self.assertTrue(any(
            item['code'] == 'unsupported_study_mode'
            for item in plan_payload['validation_issues']
        ))

    def test_study_adaptive_run_api_rejects_model_hamiltonian(self):
        status, _headers, body = handle_study_adaptive_run_request(json_dumps({
            'study_spec': {
                'name': 'adaptive-api-model-run',
                'objective': 'reject adaptive model execution',
                'system_type': 'model_hamiltonian',
                'base_model_spec': hubbard_dimer_spec(),
                'base_task': {'solver': 'fci'},
                'sweep': {'U': [0, 4]},
                'observables': ['energy'],
            },
            'adaptive': {'initial_scan_strategy': 'auto'},
        }))

        payload = json_loads(body)
        self.assertEqual(status.value, 400)
        self.assertTrue(any(
            item['code'] == 'unsupported_study_mode'
            for item in payload['validation_issues']
        ))

    def test_study_adaptive_api_forwards_resume_study_id(self):
        response_report = {
            'study_id': 'previous-adaptive-study',
            'status': 'succeeded',
            'adaptive': {'mode': 'adaptive_scan'},
        }
        service = unittest.mock.Mock(spec=StudyApplicationService)
        service.run_adaptive_study.return_value = response_report
        with unittest.mock.patch(
            'computational_study_agent.web_api.get_study_application_service',
            return_value=service,
        ):
            status, _headers, body = handle_study_adaptive_run_request(json_dumps({
                'study_spec': {
                    'name': 'resume',
                    'objective': 'resume adaptive scan',
                    'system_type': 'molecular',
                },
                'resume_study_id': 'previous-adaptive-study',
                'requested_study_id': 'new-adaptive-study',
            }))

        self.assertEqual(status.value, 200)
        self.assertEqual(json_loads(body)['report']['study_id'], 'previous-adaptive-study')
        self.assertEqual(service.run_adaptive_study.call_args.kwargs['resume_study_id'], 'previous-adaptive-study')
        self.assertEqual(service.run_adaptive_study.call_args.kwargs['requested_study_id'], 'new-adaptive-study')

    def test_study_result_analysis_api_uses_llm_builder(self):
        class DummyStudyLLM:
            def __init__(self):
                self.calls = []

            def build_result_analysis(self, request_text, execution_report, locale='zh'):
                self.calls.append((request_text, execution_report, locale))
                return '1. Calculation Overview\nThe planner result was analyzed.'

        builder = DummyStudyLLM()
        status, _headers, body = handle_study_result_analysis_request(json_dumps({
            'locale': 'en',
            'report': {
                'name': 'u-sweep',
                'objective': 'sweep U',
                'system_type': 'model_hamiltonian',
                'status': 'succeeded',
                'summary': 'completed',
                'cases': [{'case_id': 'case-0001'}],
                'comparison_table': [{'case_id': 'case-0001', 'energy': '-1.0 a.u.'}],
            },
        }), llm_request_builder=builder)

        payload = json_loads(body)
        self.assertEqual(status.value, 200)
        self.assertIn('Calculation Overview', payload['result_analysis'])
        self.assertEqual(builder.calls[0][2], 'en')
        self.assertIn('planner_study_result', builder.calls[0][0])

    def test_result_analysis_prepares_path_refinement_active_space_review(self):
        class DummyStudyLLM:
            def build_result_analysis(self, _request_text, _execution_report, locale='zh'):
                if locale != 'en':
                    raise AssertionError('Expected English result analysis')
                return 'A method-boundary discontinuity was found.'

        cases = [
            {
                'case_id': 'case-{0:04d}'.format(index),
                'label': 'R={0}'.format(index),
                'request': {
                    'atom': 'N 0 0 0; N 0 0 {0}'.format(0.8 + index * 0.1),
                    'basis': 'sto-3g',
                    'method': 'ccsd' if index <= 3 else 'casscf',
                },
                'variables': {'bond_factor': float(index)},
            }
            for index in range(1, 7)
        ]
        report = {
            'name': 'n2-boundary',
            'objective': 'inspect N2 method boundary',
            'system_type': 'molecular',
            'status': 'succeeded',
            'summary': 'completed',
            'cases': cases,
            'comparison_table': [
                {
                    'case_id': case['case_id'],
                    'label': case['label'],
                    'status': 'succeeded',
                    'bond_factor': case['variables']['bond_factor'],
                    'final_energy': energy,
                    'method': case['request']['method'],
                }
                for case, energy in zip(cases, [-1.00, -1.08, -1.16, -1.00, -1.01, -1.02])
            ],
        }
        refinement_case = copy.deepcopy(cases[1])
        refinement_case['request']['method'] = 'casscf'
        refinement_case['request']['active_space'] = {
            'enabled': True,
            'ncas': 2,
            'nelecas': 2,
            'orbital_indices': [0, 1],
            'approved': False,
        }
        path_refinement = {
            'path_refinement_plan': {
                'study_id': 'path-refinement',
                'system_type': 'molecular',
                'cases': [refinement_case],
                'observables': ['energy'],
            },
            'path_refinement_decisions': [{'case_id': 'case-0002'}],
        }
        build_refinement = unittest.mock.Mock(return_value=path_refinement)
        service = StudyApplicationService(path_refinement_builder=build_refinement)
        with unittest.mock.patch(
            'computational_study_agent.web_api.get_study_application_service',
            return_value=service,
        ):
            status, _headers, body = handle_study_result_analysis_request(json_dumps({
                'locale': 'en',
                'report': report,
                'plan': {
                    'study_id': 'n2-plan',
                    'name': 'n2-boundary',
                    'objective': 'inspect N2 method boundary',
                    'system_type': 'molecular',
                    'cases': cases,
                    'observables': ['energy'],
                },
            }), llm_request_builder=DummyStudyLLM())

        payload = json_loads(body)
        self.assertEqual(status.value, 200)
        self.assertEqual(payload['scan_path_diagnostics']['status'], 'review_required')
        self.assertEqual(payload['path_refinement'], path_refinement)
        self.assertEqual(
            payload['report']['adaptive']['workflow']['stage'],
            'path_refinement_review_required',
        )
        self.assertEqual(
            payload['report']['adaptive']['workflow']['active_plan_kind'],
            'path_refinement',
        )
        self.assertTrue(build_refinement.called)
