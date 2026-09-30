import copy
import json
import tempfile
import unittest

from computational_study_agent.executor import _resolve_case_initial_state, run_study
from computational_study_agent.retry import (
    RetryAction,
    compile_retry,
    live_dependencies,
)
from tests.computational_study_agent.test_batch_execution import (
    _plan,
    _RecordingBatchExecutor,
)


def plan():
    result = _plan(3)
    for case in result.cases:
        case.request['solver'] = {
            'name': 'dmet',
            'options': {'execution_mode': 'finite_graph'},
        }
    for index in (1, 2):
        result.cases[index].request['solver']['options']['reference_density_source'] = {
            'source_case_id': result.cases[index - 1].case_id,
        }
    return result


class DensityExecutor(_RecordingBatchExecutor):
    def __init__(self, fail=False):
        super().__init__()
        self.requests = []
        self.fail = fail

    def _report(self, task_id):
        report = super()._report(task_id)
        report.update(
            run_id=task_id,
            artifacts=[
                {
                    'kind': 'dmet_mean_field_state',
                    'path': f'/shared/{task_id}/density.npz',
                    'sha256': 'a' * 64,
                }
            ],
        )
        if self.fail:
            report['execution_status'] = 'unconverged'
            report['structured_results']['converged'] = False
        return report

    def execute_task(self, request, *, run_id=None, **kwargs):
        self.requests.append(json.loads(request))
        return super().execute_task(request, run_id=run_id, **kwargs)


class DmetContinuationTests(unittest.TestCase):
    def test_saved_study_retry_preserves_donor_and_creates_only_one_new_run(self):
        from computational_study_agent.application import StudyApplicationService
        from tests.computational_study_agent.test_retry_collection import (
            Remote,
            plan_for_service,
        )
        from tests.computational_study_agent.test_study_background import (
            inline_launcher,
        )

        class DensityRemote(Remote):
            def _report(self, task_id):
                report = super()._report(task_id)
                report['artifacts'] = [
                    {
                        'kind': 'dmet_mean_field_state',
                        'path': f'/shared/{task_id}/density.npz',
                        'sha256': 'a' * 64,
                    }
                ]
                return report

        executor = DensityRemote(fail_first=False)
        service = StudyApplicationService(
            task_executor=executor, execution_config={'execution_target': 'remote'}
        )
        service._study_launcher = inline_launcher(service)
        original = plan_for_service(3)
        for case in original.cases:
            case.request['solver'] = {
                'name': 'dmet',
                'options': {'execution_mode': 'finite_graph'},
            }
        with tempfile.TemporaryDirectory() as directory:
            before = service.run_study(original, work_dir=directory).to_dict()
            action = dict(
                service.retry_context(original.study_id, work_dir=directory),
                case_ids=['case-0003'],
                case_overrides={
                    'case-0003': {
                        'solver.options.reference_density_source': {
                            'source_case_id': 'case-0001'
                        },
                        'solver.options.correlation_potential_mixing': 0.1,
                        'solver.options.diis_enabled': False,
                    }
                },
            )
            preview = service.prepare_retry(action, work_dir=directory)
            self.assertIsNone(preview['contexts'][0]['dependency_error'])
            service.start_retry(
                original.study_id, preview['action_id'], work_dir=directory
            )
            after = service.load_report(original.study_id, work_dir=directory)
            self.assertEqual(before['cases'][:2], after['cases'][:2])
            self.assertEqual(
                [task.task_id for task in executor.submissions[-1]], ['case-0003']
            )
            submitted = executor.submissions[-1][0]
            self.assertNotEqual(submitted.run_id, 'case-0003')
            request = json.loads(submitted.request)
            source = request['solver']['options']['reference_density_source']
            self.assertEqual(source['source_run_id'], 'case-0001')
            self.assertEqual(source['path'], '/shared/case-0001/density.npz')
            self.assertEqual(source['sha256'], 'a' * 64)

    def test_chain_resolves_donor_runs_and_resume_does_not_resubmit(self):
        executor = DensityExecutor()
        original = plan()
        with tempfile.TemporaryDirectory() as directory:
            report = run_study(original, work_dir=directory, task_executor=executor)
            resumed = run_study(original, work_dir=directory, task_executor=executor)
        self.assertEqual(report.status, 'succeeded')
        self.assertEqual(resumed.execution['counts']['resumed'], 3)
        self.assertEqual(
            [[t.task_id for t in batch] for batch in executor.batch_calls],
            [['case-0001']],
        )
        self.assertEqual(executor.single_calls, ['case-0002', 'case-0003'])
        for index, request in enumerate(executor.requests, 1):
            source = request['solver']['options']['reference_density_source']
            self.assertEqual(source['source_run_id'], f'case-{index:04d}')
            self.assertEqual(source['sha256'], 'a' * 64)
        self.assertNotIn(
            'path',
            original.cases[1].request['solver']['options']['reference_density_source'],
        )

    def test_unconverged_donor_blocks_children_without_submission(self):
        executor = DensityExecutor(fail=True)
        with tempfile.TemporaryDirectory() as directory:
            report = run_study(plan(), work_dir=directory, task_executor=executor)
        self.assertEqual(executor.single_calls, [])
        self.assertEqual(
            [c['task_report']['execution_status'] for c in report.cases],
            ['unconverged', 'blocked', 'blocked'],
        )

    def test_legacy_missing_hash_wrong_run_and_self_reference_block(self):
        case = plan().cases[1]
        donor = DensityExecutor()._report('case-0001')
        for failure in ('legacy', 'hash', 'run', 'unconverged', 'self'):
            target = copy.deepcopy(case)
            report = copy.deepcopy(donor)
            if failure == 'legacy':
                report['artifacts'] = []
            elif failure == 'hash':
                report['artifacts'][0].pop('sha256')
            elif failure == 'run':
                target.request['solver']['options']['reference_density_source'][
                    'source_run_id'
                ] = 'other'
            elif failure == 'unconverged':
                report['structured_results']['converged'] = False
            elif failure == 'self':
                target.request['solver']['options']['reference_density_source'][
                    'source_case_id'
                ] = target.case_id
            with self.subTest(failure=failure):
                _, error = _resolve_case_initial_state(
                    target, {'cases': {'case-0001': {'task_report': report}}}
                )
                self.assertIsNotNone(error)

    def test_retry_override_only_changes_selected_case_and_tracks_dependencies(self):
        original = plan()
        for case in original.cases:
            case.request['solver']['options'].pop('reference_density_source', None)
        action = RetryAction.from_dict(
            {
                'study_id': original.study_id,
                'case_ids': ['case-0003'],
                'base_study_fingerprint': 'revision',
                'case_overrides': {
                    'case-0003': {
                        'solver.options.reference_density_source': {
                            'source_case_id': 'case-0001'
                        },
                        'solver.options.correlation_potential_mixing': 0.1,
                        'solver.options.max_iterations': 400,
                    }
                },
            }
        )
        retried, _, _ = compile_retry(original, action)
        self.assertEqual([case.case_id for case in retried.cases], ['case-0003'])
        self.assertEqual(live_dependencies(retried.cases[0]), {'case-0001'})
        self.assertNotIn(
            'reference_density_source', original.cases[2].request['solver']['options']
        )
        source = retried.cases[0].request['solver']['options'][
            'reference_density_source'
        ]
        source.update(path='/shared/density.npz', sha256='a' * 64)
        self.assertEqual(live_dependencies(retried.cases[0]), set())

    def test_retry_includes_deferred_dependents_and_rejects_unknown_donor(self):
        original = plan()
        action = RetryAction(
            original.study_id, ['case-0001'], {}, 'revision', include_dependents=True
        )
        subset, _, dependents = compile_retry(original, action)
        self.assertEqual(len(subset.cases), 3)
        self.assertEqual(dependents, ['case-0002', 'case-0003'])
        action = RetryAction(
            original.study_id,
            ['case-0002'],
            {
                'case-0002': {
                    'solver.options.reference_density_source': {
                        'source_case_id': 'missing'
                    },
                }
            },
            'revision',
        )
        with self.assertRaisesRegex(ValueError, 'different case'):
            compile_retry(original, action)
