import copy
import json
import tempfile
import unittest

from computational_study_agent.executor import _resolve_case_initial_state, run_study
from computational_study_agent.schema import StudyCase, StudyPlan
from tests.computational_study_agent.test_batch_execution import _RecordingBatchExecutor


def chain_plan():
    cases = []
    for index in range(1, 4):
        request = {
            'task_type': 'molecular', 'atom': 'H 0 0 0; H 0 0 1',
            'basis': 'sto-3g', 'method': 'casci',
            'active_space': {'ncas': 2, 'nelecas': 2, 'approved': True},
            'solver': {'name': 'block2_dmrg', 'options': {}},
        }
        if index > 1:
            request['solver']['options']['orbital_restart_manifest'] = {
                'source_case_id': f'case-{index-1:04d}', 'artifact_kind': 'block2_mps_manifest',
            }
        cases.append(StudyCase(case_id=f'case-{index:04d}', label=f'point {index}', request=request))
    return StudyPlan(study_id='orbital-chain', name='orbital-chain', objective='continue saved orbitals',
                     system_type='molecular', cases=cases, observables=['energy'])


class OrbitalExecutor(_RecordingBatchExecutor):
    def __init__(self, fail_anchor=False):
        super().__init__()
        self.requests = []
        self.fail_anchor = fail_anchor

    def _report(self, task_id):
        report = super()._report(task_id)
        report.update(run_id=task_id, artifacts=[
            {'kind': 'block2_mps_manifest', 'path': f'/shared/{task_id}/manifest.json'},
            {'kind': 'one_particle_state', 'path': f'/shared/{task_id}/density.npz'},
        ])
        if self.fail_anchor and task_id == 'case-0001':
            report['execution_status'] = 'failed'
        return report

    def execute_task(self, request, *, run_id=None, **kwargs):
        self.requests.append(json.loads(request))
        return super().execute_task(request, run_id=run_id, **kwargs)


class OrbitalContinuationTests(unittest.TestCase):
    def test_joint_mps_chain_resolves_parent_and_never_batches_children(self):
        plan = chain_plan()
        for case in plan.cases[1:]:
            options = case.request['solver']['options']
            options['restart_manifest'] = options.pop('orbital_restart_manifest')
            options['restart_geometry_policy'] = 'transport'
        executor = OrbitalExecutor()
        with tempfile.TemporaryDirectory() as directory:
            report = run_study(plan, work_dir=directory, task_executor=executor)
        self.assertEqual(report.status, 'succeeded')
        self.assertEqual(executor.single_calls, ['case-0002', 'case-0003'])
        self.assertEqual([[t.task_id for t in batch] for batch in executor.batch_calls], [['case-0001']])
        for index, request in enumerate(executor.requests, 1):
            options = request['solver']['options']
            self.assertEqual(options['restart_manifest'], f'/shared/case-{index:04d}/manifest.json')
            self.assertEqual(options['restart_provenance']['transfer'], 'orbitals_and_mps')
            self.assertEqual(options['restart_provenance']['source_run_id'], f'case-{index:04d}')

    def test_chain_resolves_each_parent_and_resumes_without_new_runs(self):
        executor = OrbitalExecutor()
        plan = chain_plan()
        with tempfile.TemporaryDirectory() as directory:
            report = run_study(plan, work_dir=directory, task_executor=executor)
            resumed = run_study(plan, work_dir=directory, task_executor=executor)
        self.assertEqual(report.status, 'succeeded')
        self.assertEqual(resumed.execution['counts']['resumed'], 3)
        self.assertEqual([[t.task_id for t in batch] for batch in executor.batch_calls], [['case-0001']])
        self.assertEqual(executor.single_calls, ['case-0002', 'case-0003'])
        for index, request in enumerate(executor.requests, 1):
            options = request['solver']['options']
            parent = f'case-{index:04d}'
            self.assertEqual(options['orbital_restart_manifest'], f'/shared/{parent}/manifest.json')
            self.assertEqual(options['orbital_restart_provenance']['source_run_id'], parent)
            self.assertEqual(options['orbital_restart_provenance']['transfer'], 'orbitals_only')
        self.assertIsInstance(plan.cases[1].request['solver']['options']['orbital_restart_manifest'], dict)

    def test_failed_parent_blocks_descendants_without_submission(self):
        executor = OrbitalExecutor(fail_anchor=True)
        with tempfile.TemporaryDirectory() as directory:
            report = run_study(chain_plan(), work_dir=directory, task_executor=executor)
        self.assertEqual(executor.single_calls, [])
        self.assertEqual([c['task_report']['execution_status'] for c in report.cases],
                         ['failed', 'blocked', 'blocked'])

    def test_resolves_both_density_and_orbitals_and_tracks_new_parent_run(self):
        case = chain_plan().cases[1]
        case.request['initial_state'] = {'mode': 'projected_1rdm', 'source_case_id': 'case-0001'}
        report = OrbitalExecutor()._report('case-0001')
        state = {'cases': {'case-0001': {'task_report': report}}}
        resolved, error = _resolve_case_initial_state(case, state)
        self.assertIsNone(error)
        self.assertEqual(resolved.request['initial_state']['source_artifact']['path'],
                         '/shared/case-0001/density.npz')
        changed = copy.deepcopy(state)
        changed['cases']['case-0001']['task_report']['run_id'] = 'retry-run'
        retried, error = _resolve_case_initial_state(case, changed)
        self.assertIsNone(error)
        self.assertNotEqual(resolved.request, retried.request)

    def test_invalid_reference_and_mps_conflict_are_blocked(self):
        for reference, joint in [
            ({'source_case_id': 'case-0002', 'artifact_kind': 'manifest'}, False),
            ({'source_case_id': 'case-0001'}, False),
            ({'source_case_id': 'case-0001', 'artifact_kind': 'manifest', 'path': '/wrong'}, False),
            ({'source_case_id': 'case-0001', 'artifact_kind': 'manifest'}, True),
        ]:
            with self.subTest(reference=reference, joint=joint):
                case = chain_plan().cases[1]
                options = case.request['solver']['options']
                options['orbital_restart_manifest'] = reference
                if joint:
                    options['restart_manifest'] = '/mps.json'
                self.assertIsNotNone(_resolve_case_initial_state(case, {})[1])

    def test_completed_but_unconverged_source_is_blocked(self):
        report = OrbitalExecutor()._report('case-0001')
        report['structured_results']['converged'] = False
        state = {'cases': {'case-0001': {'task_report': report}}}
        self.assertIn('converged source', _resolve_case_initial_state(chain_plan().cases[1], state)[1])
