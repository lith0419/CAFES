import json
import tempfile
import unittest
from pathlib import Path

from computational_study_agent.executor import collect_study, run_study
from computational_study_agent.planner import build_study_plan
from tests.computational_study_agent.test_grid_refinement import FunctionExecutor, grid_spec


def spec():
    result = grid_spec(require_converged=True, convergence_retry={
        'correlation_potential_mixing': 0.1, 'max_iterations': 400, 'diis_enabled': False,
    })
    result['base_task']['solver'] = {'name': 'dmet', 'options': {
        'impurity_solver': 'fci', 'execution_mode': 'finite_graph',
        'fragment_definition': 'site_count', 'impurity_size': 1,
        'reference': 'unrestricted', 'reference_density_guess': 'af',
        'correlation_potential_mixing': 0.2, 'max_iterations': 200,
        'diis_enabled': False,
    }}
    return result


class DampedExecutor(FunctionExecutor):
    def __init__(self, *, stuck=False, quality_failed=False):
        super().__init__(lambda s: s['sites'][0]['U'])
        self.stuck, self.quality_failed = stuck, quality_failed

    def execute_task(self, request_text, **kwargs):
        result = super().execute_task(request_text, **kwargs)
        request = json.loads(request_text)
        options = request['solver']['options']
        converged = (request['model_hamiltonian']['spec']['sites'][0]['U'] != 4
                     or (options['correlation_potential_mixing'] == 0.1 and not self.stuck))
        result['execution_status'] = 'succeeded' if converged else 'unconverged'
        result['structured_results'].update(converged=converged, solver='dmet', quality_checks=[{
            'id': 'convergence', 'category': 'convergence', 'required': True,
            'operator': 'equal', 'source': 'dmet',
            'observed': converged and not self.quality_failed, 'expected': True,
        }])
        return result


class GridConvergenceBarrierTests(unittest.TestCase):
    def test_retry_keeps_seed_and_history_and_collects_without_resubmitting(self):
        plan, executor = build_study_plan(spec()), DampedExecutor()
        with tempfile.TemporaryDirectory() as root:
            report = run_study(plan, work_dir=root, task_executor=executor,
                               max_case_attempts=1, grid_round_target=0)
            self.assertEqual(report.grid_refinement['status'], 'awaiting_round_barrier')
            self.assertEqual(len(executor.calls), 3)
            self.assertEqual(executor.calls[-1]['solver']['options']['max_iterations'], 400)
            self.assertEqual(executor.calls[-1]['solver']['options']['reference_density_guess'], 'af')
            stored = json.loads((Path(report.work_dir) / 'study-plan.json').read_text())
            self.assertEqual(stored['cases'][1]['request']['solver']['options']['max_iterations'], 200)
            collected = collect_study(plan, work_dir=root, task_executor=executor)
            self.assertEqual(len(executor.calls), 3)
            self.assertEqual(collected.grid_refinement['status'], 'awaiting_round_barrier')
            self.assertTrue(collected.cases[1].get('candidate_runs'))
            run_study(plan, work_dir=root, task_executor=executor, max_case_attempts=1,
                      grid_round_target=0)
            self.assertEqual(len(executor.calls), 3)
            next_round = run_study(plan, work_dir=root, task_executor=executor,
                                  max_case_attempts=1, grid_round_target=1)
            self.assertEqual(len(executor.calls), 4)
            self.assertEqual(next_round.grid_refinement['generation'], 1)
            run_study(plan, work_dir=root, task_executor=executor,
                      max_case_attempts=1, grid_round_target=1)
            self.assertEqual(len(executor.calls), 4)

    def test_failed_fallback_holds_all_refinement(self):
        plan, executor = build_study_plan(spec()), DampedExecutor(stuck=True)
        with tempfile.TemporaryDirectory() as root:
            report = run_study(plan, work_dir=root, task_executor=executor, max_case_attempts=1)
            self.assertEqual(report.grid_refinement['status'], 'awaiting_convergence')
            self.assertEqual(report.grid_refinement['awaiting_convergence'], ['case-0002'])
            self.assertEqual(report.grid_refinement['new_point_count'], 0)
            self.assertEqual(len(executor.calls), 3)
            run_study(plan, work_dir=root, task_executor=executor, max_case_attempts=1)
            self.assertEqual(len(executor.calls), 3)

    def test_quality_failure_blocks_even_with_success_status(self):
        with tempfile.TemporaryDirectory() as root:
            report = run_study(build_study_plan(spec()), work_dir=root,
                               task_executor=DampedExecutor(quality_failed=True), max_case_attempts=1)
            self.assertEqual(report.grid_refinement['status'], 'awaiting_convergence')
            self.assertEqual(report.grid_refinement['new_point_count'], 0)


if __name__ == '__main__':
    unittest.main()
