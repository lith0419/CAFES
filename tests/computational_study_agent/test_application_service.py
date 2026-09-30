from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pyscf_agent.lifecycle import new_lifecycle, transition_lifecycle

from computational_study_agent.application import (
    StudyApplicationService,
    StudyApplicationValidationError,
    StudyArtifactError,
)
from computational_study_agent.schema import StudyReport
from computational_study_agent.reports import merge_case_records, update_study_report
from tests.computational_study_agent.support import hubbard_dimer_spec


class StudyApplicationServiceTests(unittest.TestCase):
    @staticmethod
    def _valid_spec():
        return {
            'name': 'application-service-dimer',
            'objective': 'verify the application boundary',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'fci'},
            'sweep': {'U': [2, 4]},
            'observables': ['energy'],
        }

    def test_prepare_plan_returns_plan_and_validation_evidence(self):
        result = StudyApplicationService().prepare_plan(self._valid_spec())

        self.assertEqual(result.plan.name, 'application-service-dimer')
        self.assertEqual(len(result.plan.cases), 2)
        self.assertIsInstance(result.validation_issues, tuple)

    def test_prepare_plan_binds_selected_slurm_profile_memory_limit(self):
        class SchedulerExecutor:
            @staticmethod
            def execute_task(_request, **_kwargs):
                return {}

            @staticmethod
            def describe():
                return {
                    'default_resource_limits': {'memory_mb': 8192},
                    'resource_profile_options': [
                        {'id': 'memory_intensive', 'label': 'Memory Intensive', 'memory_mb': 65536},
                    ],
                }

        service = StudyApplicationService(task_executor=SchedulerExecutor())
        result = service.prepare_plan(
            self._valid_spec(),
            resource_profile='memory_intensive',
        )

        self.assertEqual(result.plan.resource_policy['memory_limit_mb'], 65536)
        self.assertEqual(result.plan.resource_policy['resource_profile'], 'memory_intensive')
        self.assertEqual(
            result.plan.cost_estimate['policy']['memory_limit_source'],
            'slurm_profile',
        )

    def test_prepare_plan_does_not_mask_unexpected_builder_errors(self):
        service = StudyApplicationService(plan_builder=mock.Mock(side_effect=TypeError('implementation bug')))
        with self.assertRaisesRegex(TypeError, 'implementation bug'):
            service.prepare_plan(self._valid_spec())

    def test_build_plan_raises_typed_validation_error(self):
        spec = self._valid_spec()
        spec['case_design'] = {
            'mode': 'grid',
            'variables': {'site_id': [0]},
            'template': {'operations': [{'op': 'not_supported'}]},
        }
        spec['sweep'] = {}

        with self.assertRaises(StudyApplicationValidationError) as raised:
            StudyApplicationService().build_plan(spec)

        self.assertEqual(raised.exception.stage, 'study_spec')
        self.assertTrue(any(issue.code == 'unsupported_model_operation' for issue in raised.exception.issues))

    def test_run_study_forwards_execution_controls_to_injected_runner(self):
        runner = mock.Mock()
        task_executor = mock.Mock()
        plan = StudyApplicationService().build_plan(self._valid_spec())
        runner.return_value = StudyReport(
            study_id=plan.study_id,
            name=plan.name,
            objective=plan.objective,
            system_type=plan.system_type,
            status='succeeded',
            work_dir='runs/application-service-test',
            cases=[],
            comparison_table=[],
        )
        service = StudyApplicationService(
            study_runner=runner,
            task_executor=task_executor,
        )

        service.run_study(
            plan,
            resume=False,
            rerun_case_ids=['case-0001'],
            rerun_statuses=['unconverged'],
            max_case_attempts=4,
        )

        self.assertFalse(runner.call_args.kwargs['resume'])
        self.assertEqual(runner.call_args.kwargs['rerun_case_ids'], ['case-0001'])
        self.assertEqual(runner.call_args.kwargs['rerun_statuses'], ['unconverged'])
        self.assertEqual(runner.call_args.kwargs['max_case_attempts'], 4)
        self.assertTrue(runner.call_args.kwargs['batch_independent'])
        self.assertIs(runner.call_args.kwargs['task_executor'], task_executor)

    def test_run_and_collect_install_probe_review_in_saved_report(self):
        with tempfile.TemporaryDirectory() as root:
            plan = StudyApplicationService().build_plan(self._valid_spec())
            plan.workflow_provenance['direct_active_space_probe'] = {'target_cases': []}
            report = StudyReport(study_id=plan.study_id, name=plan.name, objective=plan.objective,
                system_type=plan.system_type, status='succeeded', work_dir=root,
                cases=[], comparison_table=[])
            reviewed = dict(report.to_dict(), status='pending_review', adaptive={
                'workflow': {'stage': 'active_space_review_required'}})
            runner = mock.Mock(return_value=report)
            service = StudyApplicationService(study_runner=runner)
            with mock.patch.object(service, 'build_active_space_review_from_probe', return_value={'report': reviewed}):
                result = service.run_study(plan, study_report={'study_id': 'unrelated-browser-report'})
                self.assertNotIn('study_report', runner.call_args.kwargs)
                self.assertEqual(result.status, 'pending_review')
                with mock.patch('computational_study_agent.application.execution._backend_collect_study', return_value=report):
                    collected = service.collect_study(plan)
            self.assertEqual(collected.workflow, {'stage': 'active_space_review_required'})
            self.assertEqual(json.loads((Path(root) / 'study-report.json').read_text())['status'], 'pending_review')
            completed = StudyReport.from_dict(dict(report.to_dict(), adaptive={
                'direct_active_space_review_report': {'status': 'succeeded'}}))
            self.assertIs(service._finish_probe_review(plan, completed), completed)

    def test_adaptive_plan_rejects_model_hamiltonian_studies(self):
        spec = self._valid_spec()

        with self.assertRaises(StudyApplicationValidationError) as raised:
            StudyApplicationService().build_adaptive_plan(
                spec,
                options={'initial_scan_strategy': 'mp2'},
            )

        self.assertEqual(raised.exception.stage, 'adaptive_study')
        self.assertTrue(any(
            issue.code == 'unsupported_study_mode'
            for issue in raised.exception.issues
        ))

    def test_adaptive_run_rejects_model_before_calling_runner(self):
        runner = mock.Mock(return_value={'status': 'succeeded'})
        task_executor = mock.Mock()
        service = StudyApplicationService(
            adaptive_runner=runner,
            task_executor=task_executor,
        )
        spec = self._valid_spec()

        with self.assertRaises(StudyApplicationValidationError) as raised:
            service.run_adaptive_study(
                spec,
                options={'initial_scan_strategy': 'mp2'},
            )

        self.assertTrue(any(
            issue.code == 'unsupported_study_mode'
            for issue in raised.exception.issues
        ))
        runner.assert_not_called()

    def test_direct_active_space_review_returns_authoritative_study_state(self):
        plan = {
            'study_id': 'casscf-active-space-review',
            'name': 'direct-casscf-review',
            'objective': 'review the proposed active space',
            'system_type': 'molecular',
            'cases': [{
                'case_id': 'case-0001',
                'label': 'H2',
                'request': {
                    'task_type': 'molecular',
                    'method': 'casscf',
                    'active_space': {
                        'enabled': True,
                        'ncas': 2,
                        'nelecas': 2,
                        'orbital_indices': [0, 1],
                        'approved': False,
                    },
                },
            }],
            'observables': ['energy'],
        }
        service = StudyApplicationService(active_space_review_builder=mock.Mock(return_value={
            'active_space_review_plan': plan,
            'active_space_review_decisions': [{'case_id': 'case-0001'}],
        }))

        review = service.build_active_space_review({}, study_state={
            'study_id': 'parent-study',
            'name': 'parent',
            'objective': 'preserve this study',
            'system_type': 'molecular',
            'status': 'succeeded',
            'summary': 'original report',
            'cases': [{'case_id': 'case-0000'}],
            'comparison_table': [{'case_id': 'case-0000', 'status': 'succeeded'}],
        })

        report = review['report']
        self.assertEqual(report['study_id'], 'parent-study')
        self.assertEqual(report['cases'], [{'case_id': 'case-0000'}])
        self.assertEqual(report['adaptive']['mode'], 'direct_casscf_review')
        self.assertEqual(report['adaptive']['workflow']['stage'], 'active_space_review_required')
        self.assertEqual(
            report['adaptive']['workflow']['active_plan_kind'],
            'direct_casscf_review',
        )

    def test_review_execution_is_merged_back_into_parent_study(self):
        service = StudyApplicationService()
        approved_case = {
            'case_id': 'case-0002',
            'label': 'R=2.0',
            'request': {
                'method': 'casscf',
                'active_space': {
                    'enabled': True,
                    'ncas': 2,
                    'nelecas': 2,
                    'orbital_indices': [0, 1],
                    'approved': True,
                },
            },
        }
        plan = {
            'study_id': 'parent-study-direct-review-run',
            'name': 'approved subset',
            'objective': 'run one approved point',
            'system_type': 'molecular',
            'cases': [approved_case],
            'observables': ['energy'],
            '_review_kind': 'direct_casscf_review',
        }
        parent = {
            'study_id': 'parent-study',
            'name': 'parent',
            'objective': 'keep all scan points',
            'system_type': 'molecular',
            'status': 'pending_review',
            'summary': 'review pending',
            'work_dir': 'runs/parent-study',
            'cases': [
                {'case_id': 'case-0001', 'task_report': {'execution_status': 'succeeded'}},
                {'case_id': 'case-0002', 'task_report': {'execution_status': 'blocked'}},
            ],
            'comparison_table': [
                {'case_id': 'case-0001', 'status': 'succeeded', 'final_energy': -1.0},
                {'case_id': 'case-0002', 'status': 'blocked'},
            ],
            'artifacts': [{'kind': 'original'}],
            'adaptive': {
                'mode': 'direct_casscf_review',
                'direct_active_space_review_plan': {
                    **plan,
                    'cases': [{
                        **approved_case,
                        'request': {
                            **approved_case['request'],
                            'active_space': {
                                **approved_case['request']['active_space'],
                                'approved': False,
                            },
                        },
                    }],
                },
            },
        }
        completed = StudyReport(
            study_id=plan['study_id'],
            name='subset',
            objective='approved point',
            system_type='molecular',
            status='succeeded',
            work_dir='runs/subset-run',
            cases=[{'case_id': 'case-0002', 'task_report': {'execution_status': 'succeeded'}}],
            comparison_table=[{
                'case_id': 'case-0002',
                'status': 'succeeded',
                'final_energy': -1.2,
            }],
            artifacts=[{'kind': 'refined'}],
            summary='one approved case succeeded',
        )

        current = completed.to_dict() if isinstance(completed, StudyReport) else copy.deepcopy(completed)
        # Report policy receives a complete task view from the executor.
        current['cases'] = merge_case_records(parent['cases'], current['cases'])
        current['comparison_table'] = merge_case_records(parent['comparison_table'], current['comparison_table'])
        merged = update_study_report(parent, current, review_plan=plan, review_kind=plan['_review_kind'])

        self.assertEqual([item['case_id'] for item in merged['cases']], ['case-0001', 'case-0002'])
        self.assertEqual(merged['cases'][1]['task_report']['execution_status'], 'succeeded')
        self.assertEqual(merged['comparison_table'][0]['final_energy'], -1.0)
        self.assertEqual(merged['comparison_table'][1]['final_energy'], -1.2)
        self.assertEqual(merged['status'], 'succeeded')
        self.assertEqual(
            merged['summary'],
            'Study tasks updated; cases=2; succeeded=2; unresolved=0.',
        )
        self.assertEqual(len(merged['artifacts']), 2)
        self.assertEqual(
            merged['adaptive']['direct_active_space_review_report']['study_id'],
            plan['study_id'],
        )
        self.assertNotEqual(
            merged['adaptive']['workflow']['stage'],
            'active_space_review_required',
        )


    def test_static_review_execution_merges_subset_and_clears_resolved_gate(self):
        service = StudyApplicationService()
        parent = {
            'study_id': 'static-parent',
            'name': 'static parent',
            'objective': 'scan DMET inputs',
            'system_type': 'model_hamiltonian',
            'status': 'completed_with_issues',
            'cases': [
                {'case_id': 'case-0001', 'task_report': {'execution_status': 'succeeded'}},
                {'case_id': 'case-0002', 'task_report': {'execution_status': 'unconverged'}},
            ],
            'comparison_table': [
                {'case_id': 'case-0001', 'status': 'succeeded', 'final_energy': -1.0},
                {'case_id': 'case-0002', 'status': 'unconverged'},
            ],
            'artifacts': [],
            'lifecycle': transition_lifecycle(
                transition_lifecycle(
                    transition_lifecycle(
                        transition_lifecycle(
                            new_lifecycle('study', 'static-parent'),
                            'execution_started',
                        ),
                        'execution_completed',
                    ),
                    'review_requested',
                ),
                'approval_granted',
            ),
        }
        parent['lifecycle'] = transition_lifecycle(parent['lifecycle'], 'refinement_started')
        plan = {
            'study_id': 'static-parent-dmet-iteration-retry',
            '_review_kind': 'static',
            'cases': [{'case_id': 'case-0002', 'request': {'solver': {'name': 'dmet'}}}],
        }
        completed = {
            'study_id': plan['study_id'],
            'status': 'succeeded',
            'cases': [{'case_id': 'case-0002', 'task_report': {'execution_status': 'succeeded'}}],
            'comparison_table': [
                {'case_id': 'case-0002', 'status': 'succeeded', 'final_energy': -1.2},
            ],
            'artifacts': [],
        }

        current = completed.to_dict() if isinstance(completed, StudyReport) else copy.deepcopy(completed)
        # Report policy receives a complete task view from the executor.
        current['cases'] = merge_case_records(parent['cases'], current['cases'])
        current['comparison_table'] = merge_case_records(parent['comparison_table'], current['comparison_table'])
        merged = update_study_report(parent, current, review_plan=plan, review_kind=plan['_review_kind'])

        self.assertEqual(merged['status'], 'succeeded')
        self.assertEqual(len(merged['comparison_table']), 2)
        self.assertEqual(merged['comparison_table'][1]['final_energy'], -1.2)
        self.assertEqual(merged['workflow']['stage'], 'completed')
        self.assertEqual(merged['lifecycle']['stage'], 'completed')
        self.assertIn('retry_plan', merged)
        self.assertNotIn('adaptive', merged)

    def test_read_artifact_enforces_study_work_directory(self):
        service = StudyApplicationService()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_path = root / 'plot.png'
            artifact_path.write_bytes(b'png-data')
            report = {'work_dir': tmpdir}

            artifact = service.read_artifact(report, 'plot.png', allowed_suffixes=('.png',))
            self.assertEqual(artifact.content, b'png-data')
            self.assertEqual(artifact.media_type, 'image/png')

            with self.assertRaises(StudyArtifactError) as raised:
                service.read_artifact(report, '../outside.png')
            self.assertEqual(raised.exception.code, 'outside_work_dir')

    def test_result_analysis_is_protocol_neutral(self):
        class DummyLLM:
            @staticmethod
            def build_result_analysis(request_text, execution_report, locale='en'):
                self_payload = (request_text, execution_report, locale)
                if not all(self_payload):
                    raise AssertionError('Expected complete result-analysis inputs')
                return 'Study-level analysis'

        service = StudyApplicationService(path_analyzer=lambda _report: {'status': 'ok'})
        result = service.analyze_results(
            {
                'name': 'analysis',
                'objective': 'analyze',
                'system_type': 'model_hamiltonian',
                'status': 'succeeded',
                'summary': 'done',
                'cases': [],
                'comparison_table': [],
            },
            llm_request_builder=DummyLLM(),
            locale='en',
        )

        self.assertEqual(result['result_analysis'], 'Study-level analysis')
        self.assertEqual(result['scan_path_diagnostics']['status'], 'ok')

    def test_result_analysis_recomputes_cached_path_state(self):
        analyzer = mock.Mock(return_value={
            'schema': 'pyscf-agent.scan-path-diagnostics.v1',
            'kind': 'scan_path_diagnostics',
            'status': 'completed',
            'anomalies': [],
        })
        service = StudyApplicationService(path_analyzer=analyzer)

        package = service.prepare_result_analysis({
            'name': 'deferred-analysis',
            'objective': 'analyze only on request',
            'system_type': 'molecular',
            'status': 'succeeded',
            'summary': 'independent tasks completed',
            'cases': [],
            'comparison_table': [],
            'scan_path_diagnostics': {
                'schema': 'pyscf-agent.scan-path-diagnostics.v1',
                'kind': 'scan_path_diagnostics',
                'status': 'review_required',
                'anomalies': [{'id': 'stale-anomaly'}],
            },
        })

        analyzer.assert_called_once()
        self.assertEqual(package['scan_path_diagnostics']['status'], 'completed')

    def test_completed_path_refinement_is_not_proposed_again(self):
        previous_plan = {
            'study_id': 'path-refinement',
            'name': 'local-window',
            'objective': 'repair local continuity',
            'system_type': 'molecular',
            'cases': [{
                'case_id': 'case-0002',
                'label': 'R=2.0',
                'variables': {'bond_length': 2.0},
                'request': {
                    'task_type': 'molecular',
                    'method': 'casscf',
                    'active_space': {
                        'enabled': True,
                        'ncas': 6,
                        'nelecas': 6,
                        'orbital_indices': [1, 2, 3, 4, 5, 6],
                        'approved': True,
                    },
                },
            }],
            'observables': ['energy'],
        }
        candidate_plan = json.loads(json.dumps(previous_plan))
        candidate_plan['cases'][0]['request']['active_space']['approved'] = False
        full_plan = json.loads(json.dumps(previous_plan))
        full_plan['study_id'] = 'refined'
        builder = mock.Mock(return_value={
            'path_refinement_plan': candidate_plan,
            'path_refinement_decisions': [{'case_id': 'case-0002'}],
        })
        service = StudyApplicationService(path_refinement_builder=builder)
        report = {
            'system_type': 'molecular',
            'adaptive': {
                'refined_plan': full_plan,
                'path_refinement_plan': previous_plan,
                'path_refinement_report': {
                    'comparison_table': [{'case_id': 'case-0002', 'status': 'succeeded'}],
                },
                'decision_log': [],
            },
        }

        result = service.build_result_path_refinement(
            report,
            previous_plan,
            {'status': 'review_required', 'anomalies': [{'id': 'same-window'}]},
        )

        self.assertIsNone(result)
        self.assertEqual(builder.call_args.args[0].study_id, 'refined')

    def test_path_refinement_merge_invalidates_old_path_diagnostics(self):
        service = StudyApplicationService()
        parent = {
            'study_id': 'parent',
            'name': 'parent',
            'objective': 'scan a path',
            'system_type': 'molecular',
            'status': 'pending_review',
            'cases': [{'case_id': 'case-0001'}],
            'comparison_table': [{'case_id': 'case-0001', 'status': 'succeeded'}],
            'scan_path_diagnostics': {'status': 'review_required'},
            'adaptive': {
                'mode': 'result_path_review',
                'path_diagnostics': {'status': 'review_required'},
            },
        }
        plan = {
            'study_id': 'path-refinement',
            '_review_kind': 'path_refinement',
            'cases': [{'case_id': 'case-0001', 'request': {'method': 'casscf'}}],
        }
        completed = {
            'study_id': plan['study_id'],
            'status': 'succeeded',
            'cases': [{'case_id': 'case-0001'}],
            'comparison_table': [{'case_id': 'case-0001', 'status': 'succeeded'}],
            'artifacts': [],
        }

        current = completed.to_dict() if isinstance(completed, StudyReport) else copy.deepcopy(completed)
        # Report policy receives a complete task view from the executor.
        current['cases'] = merge_case_records(parent['cases'], current['cases'])
        current['comparison_table'] = merge_case_records(parent['comparison_table'], current['comparison_table'])
        merged = update_study_report(parent, current, review_plan=plan, review_kind=plan['_review_kind'])

        self.assertNotIn('scan_path_diagnostics', merged)
        self.assertNotIn('path_diagnostics', merged['adaptive'])
        self.assertEqual(merged['adaptive']['mode'], 'adaptive_scan')

    def test_result_analysis_retries_path_with_projected_1rdm_before_refinement(self):
        class DummyLLM:
            @staticmethod
            def build_result_analysis(request_text, execution_report, locale='en'):
                return 'Continuation-aware analysis'

        cases = []
        report_cases = []
        rows = []
        for index, energy in enumerate((-1.00, -1.08, -1.16, -1.00, -1.01, -1.02), start=1):
            case_id = 'case-{0:04d}'.format(index)
            method = 'ccsd' if index <= 3 else 'casscf'
            request = {
                'task_type': 'molecular',
                'atom': 'H 0 0 0; H 0 0 {0}'.format(0.7 + index / 10),
                'basis': 'sto-3g',
                'charge': 0,
                'spin': 0,
                'method': method,
                'restricted': True,
            }
            cases.append({
                'case_id': case_id,
                'label': 'R={0}'.format(index),
                'request': request,
                'variables': {'bond_factor': float(index)},
            })
            task_report = {
                'execution_status': 'succeeded',
                'task_spec': {
                    'system': {
                        'atom': request['atom'],
                        'basis': request['basis'],
                        'charge': 0,
                        'spin': 0,
                    },
                    'method': {'name': method, 'restricted': True},
                },
            }
            if index in (1, 5):
                task_report['artifacts'] = [{
                    'kind': 'one_particle_state',
                    'path': '/tmp/application-path-anchor-{0}.npz'.format(index),
                }]
            report_cases.append({'case_id': case_id, 'task_report': task_report})
            rows.append({
                'case_id': case_id,
                'status': 'succeeded',
                'bond_factor': float(index),
                'final_energy': '{0:.8f} Ha'.format(energy),
                'method': method,
            })
        plan = {
            'study_id': 'refined',
            'name': 'path-analysis',
            'objective': 'test result continuation',
            'system_type': 'molecular',
            'cases': cases,
            'observables': ['energy'],
        }
        diagnostics = {
            'status': 'review_required',
            'coordinate': 'bond_factor',
            'anomalies': [{
                'id': 'path-anomaly-001',
                'target_case_ids': ['case-0002', 'case-0003'],
                'reason': 'Non-smooth method transition.',
                'transition': {
                    'left_case_id': 'case-0003',
                    'right_case_id': 'case-0004',
                },
            }],
        }
        left_report = StudyReport(
            study_id='path-window-restart-left',
            name='left',
            objective='left restart',
            system_type='molecular',
            status='succeeded',
            work_dir='left',
            cases=[
                {'case_id': 'case-0002', 'task_report': {'execution_status': 'succeeded'}},
                {'case_id': 'case-0003', 'task_report': {'execution_status': 'succeeded'}},
            ],
            comparison_table=[
                dict(rows[1], final_energy='-1.01000000 Ha'),
                dict(rows[2], final_energy='-1.02000000 Ha'),
            ],
        )
        right_report = StudyReport(
            study_id='path-window-restart-right',
            name='right',
            objective='right restart',
            system_type='molecular',
            status='succeeded',
            work_dir='right',
            cases=[
                {'case_id': 'case-0002', 'task_report': {'execution_status': 'succeeded'}},
                {'case_id': 'case-0003', 'task_report': {'execution_status': 'succeeded'}},
            ],
            comparison_table=[
                dict(rows[1], final_energy='-1.01000001 Ha'),
                dict(rows[2], final_energy='-1.02000001 Ha'),
            ],
        )
        runner = mock.Mock(side_effect=[left_report, right_report])
        path_analyzer = mock.Mock(side_effect=[diagnostics, {'status': 'completed', 'anomalies': []}])
        path_refinement_builder = mock.Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            service = StudyApplicationService(
                study_runner=runner,
                path_analyzer=path_analyzer,
                path_refinement_builder=path_refinement_builder,
            )
            review = service.analyze_results(
                {
                    'name': 'path-analysis',
                    'objective': 'test result continuation',
                    'system_type': 'molecular',
                    'status': 'succeeded',
                    'summary': 'complete scan',
                    'work_dir': tmpdir,
                    'cases': report_cases,
                    'comparison_table': rows,
                    'adaptive': {'decision_log': [], 'options': {}},
                },
                plan=plan,
                llm_request_builder=DummyLLM(),
            )
            approval = review['path_restart']['approval']
            result = service.analyze_results(
                review['report'],
                plan=plan,
                llm_request_builder=DummyLLM(),
                path_restart_approval={
                    'decision': 'approve',
                    'approval_token': approval['approval_token'],
                },
            )
            persisted_report = json.loads(
                (Path(tmpdir) / 'adaptive-study-report.json').read_text(encoding='utf-8')
            )
            persisted_validation = json.loads(
                (Path(tmpdir) / 'adaptive-path-window-restart-validation.json').read_text(
                    encoding='utf-8'
                )
            )

        self.assertEqual(review['path_restart']['status'], 'approval_required')
        approval_items = {
            item['case_id']: item
            for item in review['path_restart']['approval']['items']
        }
        self.assertEqual(approval_items['case-0002']['right_source_method'], 'ccsd')
        self.assertEqual(
            approval_items['case-0002']['right_source_artifact']['deferred_case_id'],
            'case-0003',
        )
        self.assertEqual(approval_items['case-0003']['right_source_method'], 'casscf')
        self.assertEqual(runner.call_count, 2)
        self.assertEqual(result['path_restart']['status'], 'validated')
        self.assertTrue(result['report']['comparison_table'][1]['path_restart_applied'])
        self.assertEqual(
            result['report']['comparison_table'][1]['path_restart_validation'],
            'bidirectional_validated',
        )
        self.assertEqual(result['scan_path_diagnostics']['status'], 'completed')
        self.assertEqual(
            persisted_report['adaptive']['path_window_restart_validation']['status'],
            'validated',
        )
        self.assertEqual(persisted_report['adaptive']['workflow']['stage'], 'completed')
        self.assertEqual(persisted_validation['status'], 'validated')
        path_refinement_builder.assert_not_called()

    def test_result_analysis_does_not_repeat_stored_path_restart(self):
        class DummyLLM:
            @staticmethod
            def build_result_analysis(request_text, execution_report, locale='en'):
                return 'Stored continuation analysis'

        path_refinement_builder = mock.Mock(return_value={'path_refinement_plan': {'cases': []}})
        service = StudyApplicationService(
            study_runner=mock.Mock(),
            path_analyzer=lambda _report: {
                'status': 'review_required',
                'coordinate': 'bond_factor',
                'anomalies': [{'target_case_ids': ['case-0002']}],
            },
            path_refinement_builder=path_refinement_builder,
        )
        report = {
            'name': 'stored-restart',
            'objective': 'avoid duplicate execution',
            'system_type': 'molecular',
            'status': 'succeeded',
            'summary': 'done',
            'work_dir': 'runs/stored-restart',
            'cases': [],
            'comparison_table': [],
            'adaptive': {
                'decision_log': [],
                'path_window_restart_decisions': [{'case_id': 'case-0002'}],
                'path_window_restart_validation': {
                    'status': 'review_required',
                    'comparisons': [{'case_id': 'case-0002', 'accepted': False}],
                },
            },
        }
        plan = {
            'study_id': 'refined',
            'name': 'stored-restart',
            'objective': 'avoid duplicate execution',
            'system_type': 'molecular',
            'cases': [],
        }

        result = service.analyze_results(
            report,
            plan=plan,
            llm_request_builder=DummyLLM(),
        )

        self.assertEqual(result['path_restart']['status'], 'already_attempted')
        self.assertEqual(result['path_restart']['validation']['status'], 'review_required')
        path_refinement_builder.assert_called_once()

    def test_cross_method_path_restart_can_be_skipped_without_execution(self):
        runner = mock.Mock()
        restart_builder = mock.Mock(return_value={
            'path_window_restart_plans': {
                'left': {
                    'study_id': 'left',
                    'name': 'left',
                    'objective': 'left restart',
                    'system_type': 'molecular',
                    'cases': [],
                    'observables': ['energy'],
                },
                'right': {
                    'study_id': 'right',
                    'name': 'right',
                    'objective': 'right restart',
                    'system_type': 'molecular',
                    'cases': [],
                    'observables': ['energy'],
                },
            },
            'path_window_restart_decisions': [{
                'case_id': 'case-0002',
                'target_method': 'ccsd',
                'target_reference': 'uhf',
                'window_case_ids': ['case-0002', 'case-0003'],
                'left_source_case_id': 'case-0001',
                'left_source_method': 'ccsd',
                'left_source_reference': 'uhf',
                'right_source_case_id': 'case-0004',
                'right_source_method': 'casscf',
                'right_source_reference': 'uhf',
                'approval_required': True,
                'cross_method_restart': True,
                'initial_state_kind': 'scf_reference_1rdm',
            }],
        })
        service = StudyApplicationService(
            study_runner=runner,
            path_restart_builder=restart_builder,
        )
        plan = {
            'study_id': 'path-plan',
            'name': 'path-plan',
            'objective': 'review continuation',
            'system_type': 'molecular',
            'cases': [{
                'case_id': 'case-0002',
                'label': 'R=2',
                'request': {
                    'atom': 'H 0 0 0; H 0 0 1.0',
                    'basis': 'sto-3g',
                    'method': 'ccsd',
                },
                'variables': {'bond_length': 1.0},
            }],
            'observables': ['energy'],
        }
        diagnostics = {
            'status': 'review_required',
            'coordinate': 'bond_length',
            'anomalies': [{'target_case_ids': ['case-0002']}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            report = {
                'work_dir': tmpdir,
                'system_type': 'molecular',
                'adaptive': {'decision_log': []},
            }
            pending = service.run_result_path_restarts(report, plan, diagnostics)
            skipped = service.run_result_path_restarts(
                pending['report'],
                plan,
                diagnostics,
                approval={
                    'decision': 'skip',
                    'approval_token': pending['approval']['approval_token'],
                },
            )

        self.assertEqual(pending['status'], 'approval_required')
        self.assertEqual(skipped['status'], 'skipped')
        self.assertEqual(
            skipped['report']['adaptive']['path_window_restart_review']['decision'],
            'skip',
        )
        runner.assert_not_called()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
