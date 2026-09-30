from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from computational_study_agent.executor import run_study, collect_study
from computational_study_agent.grid.refinement import groups
from computational_study_agent.grid.execution import run_grid_study
from computational_study_agent.planner import build_study_plan
from computational_study_agent.schema import StudyPlan
from computational_study_agent.validation import has_errors, validate_study_spec
from tests.computational_study_agent.support import hubbard_dimer_spec


def grid_spec(axes=None, **policy):
    axes = axes or ['U']
    return {
        'name': 'grid-test', 'system_type': 'model_hamiltonian',
        'base_model_spec': hubbard_dimer_spec(), 'base_task': {'solver': 'fci'},
        'sweep': {axis: [0, 4] for axis in axes},
        'grid_refinement': {'axes': axes, 'max_new_points': 40, 'max_rounds': 4,
                            'metrics': [{'name': 'energy_per_site', 'atol': 0.05, 'rtol': 0.0}], **policy},
    }


class FunctionExecutor:
    def __init__(self, function, fail=None):
        self.function, self.fail, self.calls = function, fail, []

    def describe(self):
        return {'executor_id': 'test-grid', 'execution_mode': 'synchronous'}

    def execute_task(self, request_text, **kwargs):
        request = json.loads(request_text)
        spec = request['model_hamiltonian']['spec']
        self.calls.append(request)
        failed = self.fail and self.fail(spec)
        results = {'task_type': 'model_hamiltonian', 'solver': 'fci', 'converged': not failed,
                   'energy_unit': 'a.u.', 'final_energy': 2 * self.function(spec),
                   'energy_per_site': self.function(spec)}
        return {'execution_status': 'unconverged' if failed else 'succeeded',
                'run_id': kwargs['run_id'], 'work_dir': str(kwargs['work_dir']),
                'structured_results': results, 'artifacts': [], 'errors': []}


class GridRefinementTests(unittest.TestCase):
    def run_grid(self, spec, function, **kwargs):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        plan = build_study_plan(spec)
        executor = FunctionExecutor(function)
        report = run_study(plan, work_dir=directory.name, task_executor=executor,
                           batch_independent=False, **kwargs)
        return plan, executor, report, directory.name

    def test_linear_function_only_needs_midpoint(self):
        plan, executor, report, root = self.run_grid(grid_spec(), lambda s: 7 * s['sites'][0]['U'] + 1)
        self.assertEqual(len(executor.calls), 3)
        self.assertEqual(report.grid_refinement['status'], 'satisfied')
        self.assertEqual(report.grid_refinement['new_point_count'], 1)
        self.assertEqual([r['grid_origin'] for r in report.comparison_table], ['seed', 'seed', 'refined'])
        resumed = run_study(plan, work_dir=root, task_executor=executor, batch_independent=False)
        self.assertEqual(len(executor.calls), 3)
        self.assertEqual(resumed.grid_refinement['status'], 'satisfied')
        self.assertEqual([c['case_id'] for c in resumed.cases],
                         ['case-0001', 'case-0002', 'grid-0001'])
        collected = collect_study(plan, work_dir=root, task_executor=executor)
        self.assertEqual(len(executor.calls), 3)
        self.assertEqual(len(collected.comparison_table), 3)
        self.assertEqual(collected.grid_refinement['status'], 'satisfied')

    def test_curvature_refines_and_preserves_seed_ids(self):
        plan, executor, report, root = self.run_grid(grid_spec(max_rounds=6), lambda s: s['sites'][0]['U'] ** 2)
        self.assertEqual(report.grid_refinement['status'], 'satisfied')
        self.assertEqual(len(executor.calls), 33)
        self.assertEqual([c['case_id'] for c in report.cases[:2]], [c.case_id for c in plan.cases])
        self.assertTrue(any(d['reason'] == 'interpolation_error' for d in report.grid_refinement['decisions']))
        self.assertEqual(len(set(c['case_id'] for c in report.cases)), len(report.cases))
        self.assertEqual([c['case_id'] for c in report.cases],
                         ['case-0001', 'case-0002'] +
                         ['grid-{0:04d}'.format(i) for i in range(1, len(report.cases) - 1)])

    def test_two_dimensions_use_local_cells_within_budget(self):
        _, executor, report, root = self.run_grid(grid_spec(['U', 'V'], max_new_points=100),
            lambda s: s['sites'][0]['U'] ** 2 + s['bonds'][0]['V'] ** 2)
        saved = StudyPlan.from_dict(json.loads((Path(report.work_dir) / 'study-plan.json').read_text()))
        groups(saved.cases, ['U', 'V'], rectangular=False)
        self.assertEqual(report.grid_refinement['strategy'], 'local_cells')
        self.assertTrue(report.grid_refinement['cells'])
        self.assertLessEqual(len(executor.calls), 104)
        self.assertEqual(report.grid_refinement['status'], 'budget_exhausted')

    def test_bilinear_surface_probes_interior(self):
        _, executor, report, _ = self.run_grid(grid_spec(['U', 'V']),
            lambda s: s['sites'][0]['U'] * s['bonds'][0]['V'])
        self.assertEqual(len(executor.calls), 5)
        self.assertEqual(report.grid_refinement['status'], 'satisfied')
        self.assertTrue(any(c['model_hamiltonian']['spec']['sites'][0]['U'] == 2
                            and c['model_hamiltonian']['spec']['bonds'][0]['V'] == 2 for c in executor.calls))

    def test_small_budget_can_probe_one_local_center(self):
        _, executor, report, _ = self.run_grid(grid_spec(['U', 'V'], max_new_points=1), lambda s: 1.)
        self.assertEqual(len(executor.calls), 5)
        self.assertEqual(report.grid_refinement['status'], 'satisfied')
        self.assertEqual(report.grid_refinement['new_point_count'], 1)

    def test_failed_points_do_not_mean_flat_curve(self):
        plan = build_study_plan(grid_spec())
        executor = FunctionExecutor(lambda s: 1., fail=lambda s: s['sites'][0]['U'] == 4)
        with tempfile.TemporaryDirectory() as tmp:
            report = run_study(plan, work_dir=tmp, task_executor=executor, batch_independent=False)
        self.assertEqual(report.grid_refinement['status'], 'insufficient_evidence')
        self.assertEqual(len(executor.calls), 2)

    def test_round_and_spacing_limits_are_not_satisfaction(self):
        for policy, status in (({'max_rounds': 1}, 'max_rounds_reached'),
                               ({'min_spacing': {'U': 3}}, 'min_spacing_reached')):
            with self.subTest(policy=policy):
                _, _, report, _ = self.run_grid(grid_spec(**policy), lambda s: s['sites'][0]['U'] ** 2)
                self.assertEqual(report.grid_refinement['status'], status)
                self.assertFalse(report.grid_refinement['sampling_satisfied'])

    def test_width_constraint_refines_a_linear_curve(self):
        _, executor, report, _ = self.run_grid(grid_spec(max_interval={'U': 0.5}), lambda s: 2.)
        self.assertEqual(len(executor.calls), 9)
        self.assertEqual(report.grid_refinement['status'], 'satisfied')

    def test_fixed_particle_sectors_are_separate_groups(self):
        spec = grid_spec()
        spec['sweep']['nelec'] = [[1, 1], [2, 0]]
        _, executor, report, _ = self.run_grid(spec, lambda s: s['sites'][0]['U'])
        self.assertEqual(len(executor.calls), 6)
        self.assertEqual(report.grid_refinement['new_point_count'], 2)

    def test_dmet_beta_and_fragment_contract_is_preserved(self):
        spec = grid_spec(max_rounds=1)
        spec['base_task']['solver'] = {'name': 'dmet', 'options': {
            'impurity_solver': 'ccsd', 'impurity_solver_options': {'beta': 1000},
            'execution_mode': 'finite_graph', 'fragment_definition': 'site_count',
            'impurity_size': 1, 'reference': 'unrestricted'}}
        _, executor, _, _ = self.run_grid(spec, lambda s: s['sites'][0]['U'])
        self.assertEqual(len(executor.calls), 3)
        self.assertTrue(all(c['solver'] == spec['base_task']['solver'] for c in executor.calls))

    def test_pending_insertion_recovers_without_repeating_seed(self):
        from computational_study_agent.executor import _run_study
        plan = build_study_plan(grid_spec())
        executor = FunctionExecutor(lambda s: 2.)
        calls = []
        def interrupt(plan, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt('Simulated coordinator interruption before submission')
            return _run_study(plan, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(KeyboardInterrupt):
                run_grid_study(plan, runner=interrupt, work_dir=tmp, task_executor=executor, batch_independent=False)
            self.assertEqual(len(executor.calls), 2)
            report = run_study(plan, work_dir=tmp, task_executor=executor, batch_independent=False)
            self.assertEqual(len(executor.calls), 3)
            self.assertEqual(report.grid_refinement['status'], 'satisfied')

    def test_legacy_insertions_keep_identity_and_new_ids_continue_sequence(self):
        from computational_study_agent.executor import _run_study
        plan = build_study_plan(grid_spec(max_rounds=2))
        executor = FunctionExecutor(lambda s: s['sites'][0]['U'] ** 2)
        def interrupt(current, **kwargs):
            if len(current.cases) > 2:
                raise KeyboardInterrupt('Insertion saved before execution')
            return _run_study(current, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(KeyboardInterrupt):
                run_grid_study(plan, runner=interrupt, work_dir=tmp,
                               task_executor=executor, batch_independent=False)
            directory = Path(tmp) / plan.study_id
            path = directory / 'grid-refinement-state.json'
            state = json.loads(path.read_text())
            self.assertEqual(state['additions'][0]['case']['case_id'], 'grid-0001')
            legacy_id = 'grid-legacy-midpoint'
            state['additions'][0]['case']['case_id'] = legacy_id
            state['decisions'][0]['case_ids'] = [legacy_id]
            path.write_text(json.dumps(state))
            # The initial plan can recover the committed insertion transaction.
            report = run_study(plan, work_dir=tmp, task_executor=executor, batch_independent=False)
            expected = ['case-0001', 'case-0002', legacy_id, 'grid-0001', 'grid-0002']
            self.assertEqual([case['case_id'] for case in report.cases], expected)
            self.assertEqual(len(executor.calls), 5)
            resumed = run_study(plan, work_dir=tmp, task_executor=executor, batch_independent=False)
            self.assertEqual([case['case_id'] for case in resumed.cases], expected)
            self.assertEqual(len(executor.calls), 5)

    def test_frozen_model_file_is_not_read_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'model.json'
            spec = grid_spec()
            file.write_text(json.dumps(spec.pop('base_model_spec')))
            spec['base_model_input_file'] = str(file)
            plan = build_study_plan(spec)
            file.unlink()
            executor = FunctionExecutor(lambda s: 1.)
            report = run_study(plan, work_dir=tmp, task_executor=executor, batch_independent=False)
            self.assertEqual(report.grid_refinement['status'], 'satisfied')

    def test_named_grid_template_axis(self):
        from computational_study_agent.web_ui import DEFAULT_STUDY_SPEC
        spec = copy.deepcopy(DEFAULT_STUDY_SPEC)
        spec['grid_refinement'] = {'axes': ['U_value'], 'max_new_points': 5,
                                  'metrics': [{'name': 'energy_per_site'}]}
        _, executor, report, _ = self.run_grid(spec, lambda s: s['sites'][0]['U'])
        self.assertEqual(len(executor.calls), 3)
        self.assertEqual(report.grid_refinement['status'], 'satisfied')

    def test_default_metrics_require_double_occupancy_with_absolute_tolerances(self):
        spec = grid_spec()
        del spec['grid_refinement']['metrics']
        plan, executor, report, _ = self.run_grid(spec, lambda s: 1.)
        self.assertEqual([m['name'] for m in plan.grid_refinement['metrics']],
                         ['energy_per_site', 'mean_double_occupancy'])
        self.assertTrue(all(m['required'] and m['rtol'] == 0 for m in plan.grid_refinement['metrics']))
        self.assertEqual(len(executor.calls), 2)
        self.assertEqual(report.grid_refinement['status'], 'insufficient_evidence')

    def test_malformed_policies_and_discrete_axes_are_rejected(self):
        for policy in ({'axes': ['nelec']}, {'axes': ['U', 'U']}, {'axes': ['U'], 'max_new_points': True},
                       {'axes': ['U'], 'min_spacing': {'U': float('nan')}},
                       {'axes': ['U'], 'metrics': [{'name': 'energy_per_site', 'atol': 0, 'rtol': 0}]},
                       {'axes': ['U'], 'max_rounds': -1}, {'axes': ['U'], 'typo': 1}):
            spec = grid_spec()
            spec['grid_refinement'] = policy
            with self.subTest(policy=policy):
                self.assertTrue(has_errors(validate_study_spec(spec)))
                with self.assertRaises(ValueError):
                    build_study_plan(spec)

    def test_explicit_cases_and_axis_solver_changes_rejected(self):
        spec = grid_spec()
        spec['case_design'] = {'mode': 'cases', 'cases': [{'variables': {'U': 0}}]}
        self.assertTrue(has_errors(validate_study_spec(spec)))
        spec['case_design'] = {'mode': 'grid', 'variables': {'U': [0, 4]}, 'template': {
            'operations': [{'op': 'set_global_parameter', 'parameter': 'U', 'value': '$U'}],
            'request_updates': {'solver': {'name': 'dmet', 'options': {'impurity_solver_options': {'beta': '$U'}}}}}}
        self.assertTrue(has_errors(validate_study_spec(spec)))

    def test_required_missing_metric_blocks_but_optional_missing_metric_does_not(self):
        for required, status in ((True, 'insufficient_evidence'), (False, 'satisfied')):
            spec = grid_spec(metrics=[
                {'name': 'energy_per_site', 'atol': .01, 'rtol': 0, 'required': True},
                {'name': 'mean_double_occupancy', 'atol': .01, 'rtol': 0, 'required': required},
            ])
            _, _, report, _ = self.run_grid(spec, lambda s: 1.)
            self.assertEqual(report.grid_refinement['status'], status)
            self.assertTrue(any(i['missing'] for i in report.grid_refinement['intervals']))

    def test_same_coverage_count_with_different_sites_is_not_comparable(self):
        from computational_study_agent.grid.refinement import metric_sample
        plan = build_study_plan(grid_spec())
        case = plan.cases[0]
        def record(values):
            return {'task_report': {'execution_status': 'succeeded', 'structured_results': {
                'dmet_local_observables': {'double_occupancy': values},
                'strong_correlation_diagnostics': {'diagnostics': [{
                    'name': 'double_occupancy_suppression', 'value': {
                        'mean_double_occupancy': .1, 'coverage': {'site_count': 1},
                        'source': 'fragment', 'scope': 'covered_fragment_sites', 'status': 'available'}}]}}}}
        first = metric_sample(record([.1, None]), {'name':'mean_double_occupancy'}, case, {})
        second = metric_sample(record([None, .1]), {'name':'mean_double_occupancy'}, case, {})
        self.assertNotEqual(first[1], second[1])
        invalid = record([.1, None])
        invalid['task_report']['structured_results']['strong_correlation_diagnostics']['diagnostics'][0]['value']['status'] = 'failed'
        self.assertIsNone(metric_sample(invalid, {'name':'mean_double_occupancy'}, case, {}))

    def test_cost_review_persists_pending_points_and_resumes_same_calculations(self):
        from computational_study_agent.executor import _run_study
        from computational_study_agent.costing import CostApprovalRequired
        plan = build_study_plan(grid_spec())
        executor = FunctionExecutor(lambda s: 1.)
        rounds = []
        def cost_gate(plan, **kwargs):
            rounds.append(1)
            if len(rounds) == 2:
                raise CostApprovalRequired({'can_execute':False, 'reasons':['test cost threshold']})
            return _run_study(plan, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(CostApprovalRequired):
                run_grid_study(plan, runner=cost_gate, work_dir=tmp, task_executor=executor, batch_independent=False)
            saved = json.loads((Path(tmp) / plan.study_id / 'study-report.json').read_text())
            self.assertEqual(saved['grid_refinement']['status'], 'cost_approval_required')
            updated = json.loads((Path(tmp) / plan.study_id / 'study-plan.json').read_text())
            self.assertEqual(len(updated['cases']), 3)
            self.assertEqual(len(executor.calls), 2)
            final = run_study(StudyPlan.from_dict(updated), work_dir=tmp, task_executor=executor, batch_independent=False)
            self.assertEqual(len(executor.calls), 3)
            self.assertEqual(final.grid_refinement['status'], 'satisfied')

    def test_changed_saved_policy_or_calculation_is_rejected(self):
        plan, executor, _, root = self.run_grid(grid_spec(), lambda s: 1.)
        changed = copy.deepcopy(plan)
        changed.grid_refinement['max_new_points'] = 99
        with self.assertRaisesRegex(ValueError, 'contract differs'):
            run_study(changed, work_dir=root, task_executor=executor, batch_independent=False)
        changed = copy.deepcopy(plan)
        changed.cases[0].request['solver'] = 'ccsd'
        with self.assertRaises(ValueError):
            run_study(changed, work_dir=root, task_executor=executor, batch_independent=False)
        self.assertEqual(len(executor.calls), 3)

    def test_plot_export_retains_origin_and_nonuniform_coordinates(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from computational_study_agent import postprocessing as pp
        rows = [{'case_id': str(index), 'x': x, 'y': y, 'energy': x+y,
                 'grid_origin': 'seed' if x in (0,4) else 'refined', 'grid_round': 0}
                for index, (x, y) in enumerate((x,y) for x in (0,1,4) for y in (0,2))]
        context = pp.PostprocessContext.from_report({'comparison_table': rows})
        spec = pp._normalized_plot_spec(context, {'tool':'heatmap','x':'x','y':'y','color':'energy'})
        payload = pp._plot_data_payload(context, spec)
        self.assertIn('grid_origin', payload['columns'])
        self.assertEqual([r['grid_origin'] for r in payload['rows']], [r['grid_origin'] for r in rows])
        self.assertNotIn('grid_round', context.view()['numeric_columns'])
        figures = []
        original = plt.subplots
        def capture(*args, **kwargs):
            value = original(*args, **kwargs)
            figures.append(value)
            return value
        with tempfile.TemporaryDirectory() as tmp, patch.object(plt, 'subplots', side_effect=capture):
            pp.run_plot_spec(context, spec={ 'tool':'heatmap','x':'x','y':'y','color':'energy'}, output_dir=Path(tmp))
        ax = figures[0][1]
        np.testing.assert_allclose(ax.collections[0].get_coordinates()[0,:,0], [-.5,.5,2.5,5.5])
        self.assertEqual(len(ax.collections), 1)  # Heatmap only, no point overlays.
        self.assertIsNone(ax.get_legend())

    def test_disabled_policy_on_expanded_plan_uses_static_runner(self):
        spec = grid_spec()
        spec.pop('grid_refinement')
        plan = build_study_plan(spec)
        plan.grid_refinement = {'enabled': False}
        executor = FunctionExecutor(lambda s: 1.)
        with tempfile.TemporaryDirectory() as tmp:
            report = run_study(plan, work_dir=tmp, task_executor=executor, batch_independent=False)
        self.assertEqual(len(executor.calls), 2)
        self.assertFalse(report.grid_refinement)

    def test_disabled_policy_preserves_static_execution(self):
        spec = grid_spec()
        spec['grid_refinement'] = {'enabled': False}
        _, executor, report, _ = self.run_grid(spec, lambda s: s['sites'][0]['U'] ** 2)
        self.assertEqual(len(executor.calls), 2)
        self.assertFalse(report.grid_refinement)


if __name__ == '__main__':
    unittest.main()
