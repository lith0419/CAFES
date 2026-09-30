from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from computational_study_agent.executor import run_study, _run_study
from computational_study_agent.grid.execution import run_grid_study, live_grid_progress
from computational_study_agent.planner import build_study_plan
from computational_study_agent.schema import StudyPlan
from tests.computational_study_agent.test_grid_refinement import grid_spec, FunctionExecutor


class LocalGridTests(unittest.TestCase):
    def run_surface(self, spec, function, fail=None):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        executor = FunctionExecutor(lambda s: function(s['sites'][0]['U'], s['bonds'][0]['V']), fail=fail)
        plan = build_study_plan(spec)
        report = run_study(plan, work_dir=root.name, task_executor=executor, batch_independent=False)
        return plan, executor, report, root.name

    def test_localized_curvature_does_not_add_complete_lines(self):
        spec = grid_spec(['U', 'V'], max_new_points=100, max_rounds=3)
        spec['sweep'] = {'U': [0, 1, 2], 'V': [0, 1, 2]}
        def bump(u, v):
            return math.sin(math.pi*u) * math.sin(math.pi*v) if u <= 1 and v <= 1 else 0.
        plan, executor, report, root = self.run_surface(spec, bump)
        points = [case['variables'] for case in report.cases]
        self.assertIn({'U': .25, 'V': .25}, points)
        self.assertNotIn({'U': .25, 'V': 2.}, points)
        self.assertLess(len(points), len({p['U'] for p in points}) * len({p['V'] for p in points}))
        cells = report.grid_refinement['cells']
        self.assertTrue(any(c['depth'] > 0 for c in cells))
        self.assertTrue(all(c['depth'] == 0 for c in cells if c['bounds']['U'][0] >= 1 or c['bounds']['V'][0] >= 1))
        self.assertAlmostEqual(sum((c['bounds']['U'][1]-c['bounds']['U'][0]) *
                                  (c['bounds']['V'][1]-c['bounds']['V'][0]) for c in cells), 4.)
        saved = json.loads((Path(report.work_dir) / 'study-plan.json').read_text())
        status = live_grid_progress(saved, report.work_dir)
        self.assertEqual(len(status['comparison_table']), len(points))
        count = len(executor.calls)
        resumed = run_study(StudyPlan.from_dict(saved), work_dir=root, task_executor=executor, batch_independent=False)
        self.assertEqual(len(executor.calls), count)
        self.assertEqual(resumed.grid_refinement['cells'], cells)

    def test_shared_edges_are_charged_once_and_budget_does_not_split_partial_cell(self):
        spec = grid_spec(['U', 'V'], max_new_points=9, max_rounds=2)
        spec['sweep'] = {'U': [0, 1, 2], 'V': [0, 1]}
        _, executor, report, _ = self.run_surface(spec, lambda u,v: u*u+v*v)
        self.assertEqual(len(executor.calls), 15)  # Six seeds, two centres, seven distinct edges.
        self.assertEqual(report.grid_refinement['new_point_count'], 9)
        self.assertEqual(len(report.grid_refinement['cells']), 8)
        coords = [(c['model_hamiltonian']['spec']['sites'][0]['U'],
                   c['model_hamiltonian']['spec']['bonds'][0]['V']) for c in executor.calls]
        self.assertEqual(len(coords), len(set(coords)))
        self.assertEqual(sum(d['point_count'] for d in report.grid_refinement['decisions']), 9)
        _, executor, report, _ = self.run_surface(grid_spec(['U','V'], max_new_points=4), lambda u,v:u*u+v*v)
        self.assertEqual(len(executor.calls), 5)  # Remaining three cannot cover four edges.
        self.assertEqual(len(report.grid_refinement['cells']), 1)
        self.assertEqual(report.grid_refinement['status'], 'budget_exhausted')

    def test_fixed_groups_are_independent(self):
        spec = grid_spec(['U', 'V'])
        spec['sweep']['nelec'] = [[1, 1], [2, 0]]
        _, executor, report, _ = self.run_surface(spec, lambda u, v: u * v)
        self.assertEqual(len(executor.calls), 10)
        self.assertEqual(len(report.grid_refinement['cells']), 2)
        from computational_study_agent.postprocessing import suggest_plot_specs
        specs = suggest_plot_specs(report)
        self.assertTrue(specs)
        # Duplicate coordinates across fixed groups cannot define one heatmap.
        self.assertTrue(all(spec['tool'] != 'heatmap' for spec in specs))

    def test_local_points_preserve_dmet_beta_and_fragments(self):
        spec = grid_spec(['U', 'V'])
        spec['base_task']['solver'] = {'name': 'dmet', 'options': {
            'impurity_solver': 'ccsd', 'impurity_solver_options': {'beta': 1000},
            'execution_mode': 'finite_graph', 'fragment_definition': 'site_count',
            'impurity_size': 1, 'reference': 'unrestricted'}}
        _, executor, report, _ = self.run_surface(spec, lambda u, v: u * v)
        self.assertEqual(len(executor.calls), 5)
        self.assertTrue(all(c['solver'] == spec['base_task']['solver'] for c in executor.calls))

    def test_missing_results_stop_local_probe_or_split(self):
        for coordinate, expected in [(0, 4), (2, 5)]:
            with self.subTest(coordinate=coordinate):
                _, executor, report, _ = self.run_surface(grid_spec(['U','V']), lambda u,v:u*u+v*v,
                    fail=lambda s: s['sites'][0]['U'] == coordinate)
                self.assertEqual(len(executor.calls), expected)
                self.assertEqual(report.grid_refinement['status'], 'insufficient_evidence')
                self.assertFalse(report.grid_refinement['sampling_satisfied'])
                self.assertEqual(len(report.grid_refinement['cells']), 1)

    def test_local_limits_and_maximum_interval(self):
        for policy, expected in [({'max_rounds':1}, 'max_rounds_reached'),
                                 ({'min_spacing':{'U':3}}, 'min_spacing_reached')]:
            _, _, report, _ = self.run_surface(grid_spec(['U','V'], **policy), lambda u,v:u*u+v*v)
            self.assertEqual(report.grid_refinement['status'], expected)
        _, _, report, _ = self.run_surface(grid_spec(['U','V'], max_interval={'U':2,'V':2}), lambda u,v:1.)
        self.assertEqual(report.grid_refinement['status'], 'satisfied')
        self.assertEqual(len(report.grid_refinement['cells']), 4)

    def test_interrupted_split_recovers_points_and_topology(self):
        plan = build_study_plan(grid_spec(['U','V'], max_rounds=2))
        executor = FunctionExecutor(lambda s:s['sites'][0]['U']**2+s['bonds'][0]['V']**2)
        def interrupted(current, **kwargs):
            if len(current.cases) > 5:
                raise KeyboardInterrupt('Before edge submission')
            return _run_study(current, **kwargs)
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(KeyboardInterrupt):
                run_grid_study(plan, runner=interrupted, work_dir=root, task_executor=executor, batch_independent=False)
            directory = Path(root) / plan.study_id
            state = json.loads((directory / 'grid-refinement-state.json').read_text())
            self.assertEqual(len(state['cells']), 4)
            self.assertEqual(len(executor.calls), 5)
            saved_ids = [c['case']['case_id'] for c in state['additions']]
            report = run_study(plan, work_dir=root, task_executor=executor, batch_independent=False)
            self.assertEqual(len(executor.calls), 13)
            self.assertEqual([c['case_id'] for c in report.cases[4:9]], saved_ids)
            self.assertEqual([c['case_id'] for c in report.cases[4:]], ['grid-%04d' % i for i in range(1,10)])

    def test_legacy_state_keeps_axis_alternation(self):
        plan = build_study_plan(grid_spec(['U','V']))
        executor = FunctionExecutor(lambda s:s['sites'][0]['U']*s['bonds'][0]['V'])
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(KeyboardInterrupt):
                run_grid_study(plan, runner=lambda *a,**kw: (_ for _ in ()).throw(KeyboardInterrupt()), work_dir=root)
            path = Path(root)/plan.study_id/'grid-refinement-state.json'
            state = json.loads(path.read_text());state.pop('strategy');state.pop('cells')
            path.write_text(json.dumps(state))
            report = run_study(plan, work_dir=root, task_executor=executor, batch_independent=False)
            self.assertEqual(report.grid_refinement['strategy'], 'axis_alternating')
            self.assertEqual(len(executor.calls), 9)
            self.assertEqual([d['axis'] for d in report.grid_refinement['decisions']], ['U','V'])

    def test_heatmap_uses_leaf_cells_and_masks_failed_centers(self):
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from computational_study_agent import postprocessing as pp
        _, _, report, _ = self.run_surface(grid_spec(['U','V'],max_rounds=2), lambda u,v:u*u+v*v)
        payload = report.to_dict()
        center_id = payload['grid_refinement']['cells'][0]['center_case_id']
        for row in payload['comparison_table']:
            if row['case_id'] == center_id:
                row['status'] = 'unconverged';row['publication_eligible'] = False
        context = pp.PostprocessContext.from_report(payload)
        spec = pp._normalized_plot_spec(context, {'tool':'heatmap','x':'V','y':'U','color':'energy_per_site'})
        data = pp._plot_data_payload(context, spec)
        self.assertEqual(len(data['cells']), 4)
        self.assertIsNone(data['cells'][0]['color_value'])
        self.assertEqual(data['cells'][0]['value_source'], 'missing_results')
        self.assertTrue(all(c['value_source']=='computed_center' for c in data['cells'][1:]))
        # Exported specs must remain runnable, with topology re-derived from the report.
        self.assertEqual(pp._normalized_plot_spec(context, dict(spec, local_cells=[]))['local_cells'], spec['local_cells'])
        with tempfile.TemporaryDirectory() as output:
            automatic = pp.run_postprocessing(payload, output_dir=output)
        self.assertEqual(automatic['status'], 'succeeded')
        self.assertTrue(all(s['tool'] == 'heatmap' for s in automatic['plot_specs']))
        figures=[];original=plt.subplots
        def capture(*args,**kwargs):
            result=original(*args,**kwargs);figures.append(result);return result
        with tempfile.TemporaryDirectory() as output, patch.object(plt,'subplots',side_effect=capture):
            pp.run_plot_spec(context, {'tool':'heatmap','x':'V','y':'U','color':'energy_per_site'}, output_dir=Path(output))
        ax=figures[0][1]
        self.assertIsInstance(ax.collections[0], PolyCollection)
        self.assertEqual(len(ax.collections),1)
        self.assertIsNone(ax.get_legend())
        self.assertEqual(len(ax.collections[0].get_paths()),4)
        self.assertEqual(list(ax.collections[0].get_array().mask),[True,False,False,False])
        self.assertGreater(len(ax.xaxis.get_minorticklocs()), 0)
        self.assertFalse(any(line.get_visible() for line in ax.get_xgridlines()))


if __name__ == '__main__':
    unittest.main()
