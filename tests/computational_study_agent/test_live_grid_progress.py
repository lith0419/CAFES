from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from computational_study_agent.application import StudyApplicationService
from computational_study_agent.executor import run_study
from computational_study_agent.grid.refinement import build_point
from computational_study_agent.planner import build_study_plan
from computational_study_agent.study_state import load_checkpoint
from tests.computational_study_agent.test_grid_refinement import grid_spec, FunctionExecutor


class LiveGridProgressTests(unittest.TestCase):
    def test_current_attempts_and_new_points_are_visible_without_collecting(self):
        with tempfile.TemporaryDirectory() as root:
            initial = build_study_plan(grid_spec())
            executor = FunctionExecutor(lambda spec: spec['sites'][0]['U'])
            report = run_study(initial, work_dir=root, task_executor=executor,
                               batch_independent=False)
            directory = Path(report.work_dir)
            plan = json.loads((directory / 'study-plan.json').read_text())
            state = load_checkpoint(directory / 'study-state.json', initial.study_id)
            grid = json.loads((directory / 'grid-refinement-state.json').read_text())
            first = state['cases'][plan['cases'][0]['case_id']]
            old_energy = first['task_report']['structured_results']['final_energy']
            first['execution'] = {'pending': True, 'run_id': 'retry-2', 'attempt_count': 2}
            first['attempt_count'] = 2
            for index, value in enumerate([1, 3, 2.5]):
                case = build_point(plan['grid_refinement_source'], {'U': value})
                case.case_id = 'new-' + str(index)
                plan['cases'].append(case.to_dict())
                grid['additions'].append({'case': case.to_dict(), 'axis': 'U', 'round': 2})
                if index == 0:
                    state['cases'][case.case_id] = {
                        **case.to_dict(), 'attempt_count': 1, 'execution_source': 'executed',
                        'task_report': {'run_id': 'fresh', 'execution_status': 'succeeded',
                            'structured_results': {'task_type': 'model_hamiltonian', 'solver': 'fci',
                                'final_energy': -7.5, 'energy_per_site': -3.75,
                                'converged': True, 'energy_unit': 'a.u.'}}}
                elif index == 1:
                    state['cases'][case.case_id] = {
                        **case.to_dict(), 'attempt_count': 1,
                        'execution': {'pending': True, 'run_id': 'running-1', 'attempt_count': 1}}
            for filename, value in [('study-plan.json', plan), ('study-state.json', state),
                                    ('grid-refinement-state.json', grid)]:
                (directory / filename).write_text(json.dumps(value))
            before = {str(path.relative_to(directory)): path.read_bytes()
                      for path in directory.rglob('*') if path.is_file()}
            service = StudyApplicationService()
            with patch('computational_study_agent.application.background.inspect_invocation',
                       return_value={'running': True, 'error': None, 'interrupted': False}):
                status = service.inspect_execution(initial.study_id, work_dir=root)
                again = service.inspect_execution(initial.study_id, work_dir=root)
            self.assertEqual(status['progress'], again['progress'])
            self.assertEqual(len(executor.calls), 3)
            after = {str(path.relative_to(directory)): path.read_bytes()
                     for path in directory.rglob('*') if path.is_file()}
            self.assertEqual(before, after)  # Polling never changes or collects saved runs.
            rows = {row['case_id']: row for row in status['progress']['comparison_table']}
            self.assertEqual(len(rows), 6)
            retry = rows[first['case_id']]
            self.assertEqual(retry['status'], 'pending')
            self.assertIsNone(retry.get('energy'))
            self.assertIsNone(retry.get('final_energy'))
            self.assertFalse(retry['publication_eligible'])
            self.assertEqual(retry['attempt_count'], 2)
            self.assertEqual(rows['new-0']['status'], 'succeeded')
            self.assertEqual(rows['new-0']['final_energy'], '-7.5 a.u.')
            self.assertEqual(rows['new-0']['grid_origin'], 'refined')
            self.assertEqual(rows['new-0']['grid_axis'], 'U')
            self.assertEqual(rows['new-0']['grid_round'], 2)
            self.assertEqual(rows['new-1']['status'], 'pending')
            self.assertEqual(rows['new-2']['status'], 'not_executed')
            self.assertFalse(rows['new-2']['publication_eligible'])
            self.assertEqual(status['task_status_counts'],
                             {'pending': 2, 'succeeded': 3, 'not_executed': 1})
            self.assertEqual(status['progress']['grid_refinement']['new_point_count'], 4)
            self.assertEqual(status['progress']['grid_refinement']['status'], 'running')
            self.assertFalse(status['can_collect'])
            self.assertEqual(first['task_report']['structured_results']['final_energy'], old_energy)

    def test_prepared_grid_lists_every_waiting_point(self):
        with tempfile.TemporaryDirectory() as root:
            service = StudyApplicationService()
            saved = service.prepare_study(grid_spec(['U', 'V']), work_dir=root)
            status = service.inspect_execution(saved['study_id'], work_dir=root)
            self.assertEqual(status['status'], 'prepared')
            self.assertEqual(status['task_status_counts'], {'not_executed': 4})
            self.assertEqual(len(status['progress']['comparison_table']), 4)
            self.assertTrue(all(row['attempt_count'] == 0
                                for row in status['progress']['comparison_table']))


if __name__ == '__main__':
    unittest.main()
