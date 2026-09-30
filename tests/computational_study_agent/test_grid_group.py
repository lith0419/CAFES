import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from computational_study_agent.application.grid_group import group_step, run_group
from computational_study_agent.executor import run_study
from computational_study_agent.planner import build_study_plan
from tests.computational_study_agent.test_grid_convergence_barrier import DampedExecutor, spec


class SavedService:
    def __init__(self, plans):
        self.plans, self.starts = plans, []

    def load_plan(self, study_id, **kwargs):
        return self.plans[study_id].to_dict()

    def start_study(self, study_id, **kwargs):
        self.starts.append((study_id, kwargs['grid_round_target']))
        return {'started': True}


class GridGroupTests(unittest.TestCase):
    def test_coordinator_persists_group_state_through_artifact_repository(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'group.json'
            group = {'schema': 'pyscf-agent.grid-group.v1', 'execution_config': {},
                     'work_dir': root, 'study_ids': ['example'], 'resource_profile': 'standard'}
            path.write_text(json.dumps(group))
            with patch('computational_study_agent.application.grid_group.create_task_executor'), \
                    patch('computational_study_agent.application.grid_group.group_step',
                          return_value={**group, 'status': 'completed'}):
                run_group(path)
            self.assertEqual(json.loads(path.read_text())['status'], 'completed')

    def test_both_studies_must_reach_barrier_before_granting_next_generation(self):
        with tempfile.TemporaryDirectory() as root:
            plans = {}
            for seed in ('af', 'cdw'):
                source = spec()
                source['base_task']['solver']['options']['reference_density_guess'] = seed
                plan = build_study_plan(source)
                plans[plan.study_id] = plan
                run_study(plan, work_dir=root, task_executor=DampedExecutor(),
                          max_case_attempts=1, grid_round_target=0)
            ids = list(plans)
            group = {'work_dir': root, 'study_ids': ids, 'resource_profile': 'standard'}
            service = SavedService(plans)
            with patch('computational_study_agent.application.grid_group.inspect_invocation',
                       side_effect=[None, {'running': True}]):
                group_step(service, group)
            self.assertEqual(group.get('target_generation', 0), 0)
            self.assertFalse(service.starts)
            with patch('computational_study_agent.application.grid_group.inspect_invocation', return_value=None):
                group_step(service, group)
                self.assertEqual(group['target_generation'], 1)
                self.assertFalse(service.starts)
                group_step(service, group)
            self.assertEqual(service.starts, [(ids[0], 1), (ids[1], 1)])

    def test_unconverged_other_study_holds_group_and_interrupted_worker_needs_attention(self):
        with tempfile.TemporaryDirectory() as root:
            plan = build_study_plan(spec())
            run_study(plan, work_dir=root, task_executor=DampedExecutor(stuck=True),
                      max_case_attempts=1, grid_round_target=0)
            service = SavedService({plan.study_id: plan})
            group = {'work_dir': root, 'study_ids': [plan.study_id], 'resource_profile': 'standard'}
            with patch('computational_study_agent.application.grid_group.inspect_invocation', return_value=None):
                group_step(service, group)
            self.assertEqual(group['status'], 'awaiting_convergence')
            self.assertFalse(service.starts)
            report_path = Path(root) / plan.study_id / 'study-report.json'
            self.assertTrue(json.loads(report_path.read_text())['grid_refinement']['awaiting_convergence'])
            with patch('computational_study_agent.application.grid_group.inspect_invocation',
                       return_value={'running': False, 'interrupted': True}):
                result = group_step(service, copy.deepcopy(group))
            self.assertEqual(result['status'], 'needs_attention')
            self.assertFalse(service.starts)
