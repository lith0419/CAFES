from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from computational_study_agent.postprocessing import (
    PostprocessContext,
    run_postprocessing,
    run_plot_spec,
    suggest_plot_specs,
)
from computational_study_agent.web_api import (
    handle_study_artifact_request,
    handle_study_postprocess_request,
    handle_study_postprocess_suggestions_request,
)


def json_dumps(payload):
    return json.dumps(payload, ensure_ascii=False).encode('utf-8')


def json_loads(body):
    return json.loads(body.decode('utf-8'))


def sample_report(work_dir=None):
    return {
        'study_id': 'study-plot-test',
        'name': 'hubbard-u-sweep',
        'objective': 'Sweep U',
        'system_type': 'model_hamiltonian',
        'status': 'succeeded',
        'work_dir': work_dir or tempfile.gettempdir(),
        'cases': [],
        'comparison_table': [
            {'case_id': 'case-0001', 'label': 'U=2', 'status': 'succeeded', 'U': '2 a.u.', 'energy': '-6.2 a.u.', 'solver': 'fci'},
            {'case_id': 'case-0002', 'label': 'U=4', 'status': 'succeeded', 'U': '4 a.u.', 'energy': '-5.1 a.u.', 'solver': 'fci'},
            {'case_id': 'case-0003', 'label': 'U=6', 'status': 'succeeded', 'U': '6 a.u.', 'energy': '-4.3 a.u.', 'solver': 'fci'},
        ],
        'artifacts': [],
        'summary': 'completed',
    }


def report_with_cases_and_rows(cases, rows, work_dir='/tmp/study'):
    report = sample_report(work_dir)
    report['cases'] = cases
    report['comparison_table'] = rows
    return report


class PostprocessingTests(unittest.TestCase):
    def test_context_from_report_preserves_comparison_table(self):
        context = PostprocessContext.from_report(sample_report('/tmp/study'))

        self.assertEqual(context.study_id, 'study-plot-test')
        self.assertEqual(context.system_type, 'model_hamiltonian')
        self.assertEqual(context.work_dir, '/tmp/study')
        self.assertEqual(len(context.comparison_table), 3)

    def test_context_excludes_rows_that_failed_required_quality_checks(self):
        report = sample_report('/tmp/study')
        report['comparison_table'][1]['quality_status'] = 'review_required'
        report['comparison_table'][1]['publication_eligible'] = False

        context = PostprocessContext.from_report(report)

        self.assertEqual(
            [row['case_id'] for row in context.comparison_table],
            ['case-0001', 'case-0003'],
        )
        self.assertEqual(context.metadata['excluded_case_ids'], ['case-0002'])

    def test_postprocessing_skips_when_every_row_is_ineligible(self):
        report = sample_report('/tmp/study')
        for row in report['comparison_table']:
            row['publication_eligible'] = False

        result = run_postprocessing(report)

        self.assertEqual(result['status'], 'skipped')
        self.assertIn('No publication-eligible', result['message'])
        self.assertEqual(
            result['excluded_case_ids'],
            ['case-0001', 'case-0002', 'case-0003'],
        )

    def test_suggest_plot_specs_uses_case_variables_for_axes(self):
        scenarios = [
            {
                'name': 'single sweep variable keeps classic labels',
                'report': report_with_cases_and_rows(
                    [
                        {'case_id': 'case-0001', 'variables': {'U': 2}},
                        {'case_id': 'case-0002', 'variables': {'U': 4}},
                        {'case_id': 'case-0003', 'variables': {'U': 6}},
                    ],
                    sample_report('/tmp/study')['comparison_table'],
                ),
                'expected_axes': {('U', 'energy')},
                'first_spec': {
                    'tool': 'line_plot',
                    'x': 'U',
                    'y': 'energy',
                    'xlabel': 'U',
                    'ylabel': 'Total energy',
                    'style': 'pyscf_agent_classic',
                },
            },
            {
                'name': 'multiple sweep variables each become x axes',
                'report': report_with_cases_and_rows(
                    [
                        {'case_id': 'case-0001', 'variables': {'U': 2, 't': -1.0}},
                        {'case_id': 'case-0002', 'variables': {'U': 4, 't': -1.0}},
                        {'case_id': 'case-0003', 'variables': {'U': 2, 't': -0.5}},
                        {'case_id': 'case-0004', 'variables': {'U': 4, 't': -0.5}},
                    ],
                    [
                        {'case_id': 'case-0001', 'U': '2 a.u.', 't': '-1.0 a.u.', 'energy': '-6.2 a.u.', 'solver': 'fci'},
                        {'case_id': 'case-0002', 'U': '4 a.u.', 't': '-1.0 a.u.', 'energy': '-5.1 a.u.', 'solver': 'fci'},
                        {'case_id': 'case-0003', 'U': '2 a.u.', 't': '-0.5 a.u.', 'energy': '-4.9 a.u.', 'solver': 'fci'},
                        {'case_id': 'case-0004', 'U': '4 a.u.', 't': '-0.5 a.u.', 'energy': '-4.3 a.u.', 'solver': 'fci'},
                    ],
                ),
                'expected_axes': {('U', 'energy'), ('t', 'energy')},
            },
            {
                'name': 'generic variables do not need physics-specific names',
                'report': report_with_cases_and_rows(
                    [
                        {'case_id': 'case-0001', 'variables': {'bond_length': 0.8, 'field_strength': 0.0}},
                        {'case_id': 'case-0002', 'variables': {'bond_length': 1.0, 'field_strength': 0.0}},
                        {'case_id': 'case-0003', 'variables': {'bond_length': 0.8, 'field_strength': 0.2}},
                        {'case_id': 'case-0004', 'variables': {'bond_length': 1.0, 'field_strength': 0.2}},
                    ],
                    [
                        {'case_id': 'case-0001', 'bond_length': 0.8, 'field_strength': 0.0, 'response': 1.2},
                        {'case_id': 'case-0002', 'bond_length': 1.0, 'field_strength': 0.0, 'response': 1.5},
                        {'case_id': 'case-0003', 'bond_length': 0.8, 'field_strength': 0.2, 'response': 1.8},
                        {'case_id': 'case-0004', 'bond_length': 1.0, 'field_strength': 0.2, 'response': 2.1},
                    ],
                ),
                'expected_axes': {('bond_length', 'response'), ('field_strength', 'response')},
                'expected_groups': {'bond_length', 'field_strength'},
            },
        ]

        for scenario in scenarios:
            with self.subTest(scenario['name']):
                specs = suggest_plot_specs(scenario['report'])
                axes = {(spec['x'], spec['y']) for spec in specs}
                self.assertTrue(scenario['expected_axes'].issubset(axes))
                if 'first_spec' in scenario:
                    for key, value in scenario['first_spec'].items():
                        self.assertEqual(specs[0][key], value)
                if 'expected_groups' in scenario:
                    grouped_specs = [spec for spec in specs if spec.get('tool') != 'heatmap']
                    self.assertTrue(all(spec.get('group') in scenario['expected_groups'] for spec in grouped_specs))

    def test_run_postprocessing_skips_when_no_specs_can_be_inferred(self):
        report = sample_report('/tmp/study')
        report['comparison_table'] = [{'case_id': 'case-0001', 'status': 'succeeded'}]

        result = run_postprocessing(report)

        self.assertEqual(result['status'], 'skipped')
        self.assertEqual(result['artifacts'], [])

    def test_suggest_plot_specs_includes_heatmap_for_two_variable_grid(self):
        report = report_with_cases_and_rows(
            [
                {'case_id': 'case-0001', 'variables': {'U': 2, 't': -1.0}},
                {'case_id': 'case-0002', 'variables': {'U': 4, 't': -1.0}},
                {'case_id': 'case-0003', 'variables': {'U': 2, 't': -0.5}},
                {'case_id': 'case-0004', 'variables': {'U': 4, 't': -0.5}},
            ],
            [
                {'case_id': 'case-0001', 'U': '2 a.u.', 't': '-1.0 a.u.', 'energy': '-1.2 a.u.'},
                {'case_id': 'case-0002', 'U': '4 a.u.', 't': '-1.0 a.u.', 'energy': '-0.2 a.u.'},
                {'case_id': 'case-0003', 'U': '2 a.u.', 't': '-0.5 a.u.', 'energy': '0.3 a.u.'},
                {'case_id': 'case-0004', 'U': '4 a.u.', 't': '-0.5 a.u.', 'energy': '1.4 a.u.'},
            ],
        )

        specs = suggest_plot_specs(report)
        heatmap = next(spec for spec in specs if spec['tool'] == 'heatmap')
        self.assertEqual(heatmap['x'], 'U')
        self.assertEqual(heatmap['y'], 't')
        self.assertEqual(heatmap['color'], 'energy')
        self.assertEqual(heatmap['xlabel'], 'U')
        self.assertEqual(heatmap['ylabel'], 't')
        self.assertEqual(heatmap['colorbar_label'], '')
        self.assertEqual(heatmap['title'], 'Total energy')
        self.assertEqual(heatmap['colormap'], 'RdBu_r')
        self.assertEqual(heatmap['center'], 0.0)

    def test_suggest_plot_specs_prefers_final_energy_for_curves(self):
        report = report_with_cases_and_rows(
            [
                {'case_id': 'case-0001', 'variables': {'bond': 0.74}},
                {'case_id': 'case-0002', 'variables': {'bond': 1.20}},
            ],
            [
                {'case_id': 'case-0001', 'bond': 0.74, 'energy': '-1.0 Ha', 'final_energy': '-1.1 Ha'},
                {'case_id': 'case-0002', 'bond': 1.20, 'energy': '-0.8 Ha', 'final_energy': '-0.9 Ha'},
            ],
        )
        report['system_type'] = 'molecular'

        specs = suggest_plot_specs(report)

        self.assertTrue(specs)
        self.assertEqual(specs[0]['x'], 'bond')
        self.assertEqual(specs[0]['y'], 'final_energy')
        self.assertEqual(specs[0]['xlabel'], 'Bond length')
        self.assertEqual(specs[0]['ylabel'], 'Energy')
        self.assertEqual(specs[0]['title'], 'Energy')

    def test_suggestions_do_not_treat_paired_coordinates_as_a_grid(self):
        # Neither the names nor linear correlation determine dimensionality.
        for coordinates in ([(1, -0.05), (2, 0.0), (3, 0.12)],
                            [(1, 1), (2, 8), (3, 27)],
                            [(1, 4), (2, 1), (3, 9)]):
            with self.subTest(coordinates=coordinates):
                rows = [{'case_id': str(i), 'a': a, 'b': b, 'energy': -i - 1.0}
                        for i, (a, b) in enumerate(coordinates)]
                cases = [{'case_id': row['case_id'], 'variables': {'a': row['a'], 'b': row['b']}}
                         for row in rows]
                specs = suggest_plot_specs(report_with_cases_and_rows(cases, rows))
                self.assertTrue(specs)
                self.assertTrue(all(s['tool'] == 'scatter_plot' and s.get('group') is None for s in specs))
                self.assertEqual({s['x'] for s in specs}, {'a', 'b'})

    def test_heatmap_suggestions_require_observed_unique_crossed_samples(self):
        scenarios = [
            ('incomplete grid with a rectangle', [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)], True),
            ('L-shaped sample', [(0, 0), (0, 1), (1, 0)], False),
            ('duplicate coordinate', [(0, 0), (0, 1), (1, 0), (1, 1), (0, 0)], False),
        ]
        for name, coordinates, expect_heatmap in scenarios:
            with self.subTest(name=name):
                rows = [{'case_id': str(i), 'a': a, 'b': b, 'energy': i + 1.0}
                        for i, (a, b) in enumerate(coordinates)]
                cases = [{'case_id': row['case_id'], 'variables': {'a': row['a'], 'b': row['b']}}
                         for row in rows]
                specs = suggest_plot_specs(report_with_cases_and_rows(cases, rows))
                self.assertEqual(any(s['tool'] == 'heatmap' for s in specs), expect_heatmap)

    def test_heatmap_suggestions_ignore_missing_and_ineligible_grid_corners(self):
        for unavailable in ({'energy': None}, {'status': 'failed'}, {'publication_eligible': False}):
            with self.subTest(unavailable=unavailable):
                rows = [{'case_id': str(i), 'a': a, 'b': b, 'energy': i + 1.0}
                        for i, (a, b) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)])]
                cases = [{'case_id': row['case_id'], 'variables': {'a': row['a'], 'b': row['b']}}
                         for row in rows]
                rows[-1].update(unavailable)
                specs = suggest_plot_specs(report_with_cases_and_rows(cases, rows))
                self.assertFalse(any(s['tool'] == 'heatmap' for s in specs))

    def test_heatmap_axes_are_not_limited_to_first_two_variables(self):
        rows = [{'case_id': str(i), 'ordinal': i, 'a': a, 'b': b, 'energy': i + 1.0}
                for i, (a, b) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)])]
        cases = [{'case_id': row['case_id'], 'variables': {k: row[k] for k in ('ordinal', 'a', 'b')}}
                 for row in rows]
        specs = suggest_plot_specs(report_with_cases_and_rows(cases, rows))
        heatmaps = [s for s in specs if s['tool'] == 'heatmap']
        self.assertEqual([(s['x'], s['y']) for s in heatmaps], [('a', 'b')])

    def test_repeated_categorical_series_still_suggest_grouped_lines(self):
        rows = [{'case_id': str(i), 'r': r, 'solver': solver, 'energy': i + 1.0}
                for i, (r, solver) in enumerate([(1, 'fci'), (2, 'fci'), (1, 'dmrg'), (2, 'dmrg')])]
        cases = [{'case_id': row['case_id'], 'variables': {'r': row['r'], 'solver': row['solver']}}
                 for row in rows]
        specs = suggest_plot_specs(report_with_cases_and_rows(cases, rows))
        self.assertEqual(specs[0]['tool'], 'line_plot')
        self.assertEqual(specs[0]['group'], 'solver')

    def test_postprocessing_excludes_internal_execution_metadata(self):
        report = report_with_cases_and_rows(
            [
                {'case_id': 'case-0001', 'variables': {'bond_length': 0.8}},
                {'case_id': 'case-0002', 'variables': {'bond_length': 1.0}},
            ],
            [
                {
                    'case_id': 'case-0001',
                    'status': 'succeeded',
                    'run_dir': 'cases/case-0001',
                    'execution_source': 'executed',
                    'attempt_count': 1,
                    'bond_length': 0.8,
                    'final_energy': '-1.0 Ha',
                    'gap': '0.4 Ha',
                },
                {
                    'case_id': 'case-0002',
                    'status': 'succeeded',
                    'run_dir': 'cases/case-0002',
                    'execution_source': 'resumed',
                    'attempt_count': 2,
                    'bond_length': 1.0,
                    'final_energy': '-1.1 Ha',
                    'gap': '0.2 Ha',
                },
            ],
        )
        report['system_type'] = 'molecular'

        specs = suggest_plot_specs(report)

        self.assertEqual([spec['y'] for spec in specs], ['final_energy', 'gap'])
        for spec in specs:
            self.assertFalse(
                {'attempt_count', 'execution_source', 'run_dir', 'status'}.intersection(
                    {spec.get('x'), spec.get('y'), spec.get('color'), spec.get('group')}
                )
            )
        with self.assertRaisesRegex(ValueError, 'internal execution metadata'):
            run_plot_spec(report, {'x': 'bond_length', 'y': 'attempt_count'})

    def test_suggest_plot_specs_prioritizes_model_correlation_metrics(self):
        rows = [
            {
                'case_id': 'case-0001',
                'U': '2 a.u.',
                'final_energy': '-6.2 a.u.',
                'energy': '-6.2 a.u.',
                'mean_double_occupancy': 0.22,
                'nearest_neighbor_spin_correlation': -0.14,
                'nearest_neighbor_charge_correlation': -0.03,
            },
            {
                'case_id': 'case-0002',
                'U': '4 a.u.',
                'final_energy': '-5.1 a.u.',
                'energy': '-5.1 a.u.',
                'mean_double_occupancy': 0.11,
                'nearest_neighbor_spin_correlation': -0.31,
                'nearest_neighbor_charge_correlation': -0.07,
            },
        ]
        report = report_with_cases_and_rows(
            [
                {'case_id': 'case-0001', 'variables': {'U': 2}},
                {'case_id': 'case-0002', 'variables': {'U': 4}},
            ],
            rows,
        )

        specs = suggest_plot_specs(report)

        self.assertEqual(
            [spec['y'] for spec in specs],
            [
                'final_energy',
                'mean_double_occupancy',
                'nearest_neighbor_spin_correlation',
                'nearest_neighbor_charge_correlation',
            ],
        )
        self.assertEqual(specs[1]['ylabel'], 'Mean double occupancy')
        self.assertEqual(specs[2]['ylabel'], 'Spin correlation')
        self.assertEqual(specs[3]['ylabel'], 'Charge correlation')

    def test_suggest_plot_specs_uses_model_correlation_metrics_for_heatmaps(self):
        rows = [
            {
                'case_id': 'case-0001', 'U': '2 a.u.', 'V': '0 a.u.',
                'final_energy': '-6.2 a.u.', 'mean_double_occupancy': 0.22,
                'nearest_neighbor_spin_correlation': -0.14,
                'nearest_neighbor_charge_correlation': -0.03,
            },
            {
                'case_id': 'case-0002', 'U': '4 a.u.', 'V': '0 a.u.',
                'final_energy': '-5.1 a.u.', 'mean_double_occupancy': 0.11,
                'nearest_neighbor_spin_correlation': -0.31,
                'nearest_neighbor_charge_correlation': -0.07,
            },
            {
                'case_id': 'case-0003', 'U': '2 a.u.', 'V': '1 a.u.',
                'final_energy': '-5.9 a.u.', 'mean_double_occupancy': 0.24,
                'nearest_neighbor_spin_correlation': -0.12,
                'nearest_neighbor_charge_correlation': 0.02,
            },
            {
                'case_id': 'case-0004', 'U': '4 a.u.', 'V': '1 a.u.',
                'final_energy': '-4.7 a.u.', 'mean_double_occupancy': 0.15,
                'nearest_neighbor_spin_correlation': -0.27,
                'nearest_neighbor_charge_correlation': -0.01,
            },
        ]
        report = report_with_cases_and_rows(
            [
                {'case_id': 'case-0001', 'variables': {'U': 2, 'V': 0}},
                {'case_id': 'case-0002', 'variables': {'U': 4, 'V': 0}},
                {'case_id': 'case-0003', 'variables': {'U': 2, 'V': 1}},
                {'case_id': 'case-0004', 'variables': {'U': 4, 'V': 1}},
            ],
            rows,
        )

        specs = suggest_plot_specs(report)

        self.assertEqual([spec['tool'] for spec in specs], ['heatmap'] * 4)
        self.assertEqual(
            [spec['color'] for spec in specs],
            [
                'final_energy',
                'mean_double_occupancy',
                'nearest_neighbor_spin_correlation',
                'nearest_neighbor_charge_correlation',
            ],
        )

    def test_run_plot_spec_reports_missing_matplotlib_cleanly(self):
        context = PostprocessContext.from_report(sample_report('/tmp/study'))
        with unittest.mock.patch('builtins.__import__', side_effect=ImportError('no matplotlib')):
            with self.assertRaisesRegex(RuntimeError, 'matplotlib is required'):
                run_plot_spec(context, {'x': 'U', 'y': 'energy'})

    def test_study_postprocess_api_contracts(self):
        with self.subTest('suggestions endpoint returns specs'):
            status, _headers, body = handle_study_postprocess_suggestions_request(json_dumps({
                'report': sample_report('/tmp/study'),
            }))
            payload = json_loads(body)
            self.assertEqual(status.value, 200)
            self.assertEqual(payload['plot_specs'][0]['x'], 'U')
            self.assertEqual(payload['plot_specs'][0]['y'], 'energy')

        with self.subTest('postprocess endpoint validates spec shape'):
            status, _headers, body = handle_study_postprocess_request(json_dumps({
                'report': sample_report('/tmp/study'),
                'plot_specs': {'x': 'U', 'y': 'energy'},
            }))
            payload = json_loads(body)
            self.assertEqual(status.value, 400)
            self.assertIn('plot_specs must be a list', payload['error'])

        with self.subTest('dataset collection routes through the selected executor'):
            service = unittest.mock.Mock()
            service.run_postprocessing.return_value = {
                'status': 'succeeded',
                'artifacts': [],
                'plot_specs': [],
                'actions': [],
            }
            with unittest.mock.patch(
                'computational_study_agent.web_api.get_study_application_service',
                return_value=service,
            ) as factory:
                status, _headers, body = handle_study_postprocess_request(json_dumps({
                    'report': sample_report('/tmp/study'),
                    'actions': ['collect_hamiltonian_dataset'],
                    'execution_target': {'kind': 'remote', 'profile': 'amarel'},
                }))
            payload = json_loads(body)
            self.assertEqual(status.value, 200)
            self.assertEqual(payload['postprocessing']['status'], 'succeeded')
            factory.assert_called_once_with({'kind': 'remote', 'profile': 'amarel'})
            service.run_postprocessing.assert_called_once_with(
                sample_report('/tmp/study'),
                specs=None,
                actions=['collect_hamiltonian_dataset'],
            )

    def test_study_artifact_api_path_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as outside_dir:
            png_path = Path(tmpdir) / 'postprocessing' / 'plot.png'
            png_path.parent.mkdir(parents=True)
            png_path.write_bytes(b'\x89PNG\r\n\x1a\nfake')
            outside = Path(outside_dir) / 'outside-plot.png'
            outside.write_bytes(b'\x89PNG\r\n\x1a\nfake')
            report = sample_report(tmpdir)

            for name, path_value in (('absolute', str(png_path)), ('relative', 'postprocessing/plot.png')):
                with self.subTest(name):
                    status, headers, body = handle_study_artifact_request(json_dumps({
                        'report': report,
                        'path': path_value,
                    }))
                    self.assertEqual(status.value, 200)
                    self.assertEqual(headers['Content-Type'], 'image/png')
                    self.assertTrue(body.startswith(b'\x89PNG'))

            with self.subTest('outside work dir rejected'):
                status, _headers, body = handle_study_artifact_request(json_dumps({
                    'report': report,
                    'path': str(outside),
                }))
                payload = json_loads(body)
                self.assertEqual(status.value, 403)
                self.assertIn('outside the study work directory', payload['error'])

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'matplotlib is not installed')
    def test_run_plot_spec_writes_prefixed_artifacts_and_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = sample_report(tmpdir)
            context = PostprocessContext.from_report(report)
            result = run_plot_spec(
                context,
                {'tool': 'line_plot', 'x': 'U', 'y': 'energy', 'output': 'energy-vs-u'},
                output_dir=Path(tmpdir) / 'postprocessing',
            )

            paths = [item['path'] for item in result['artifacts']]
            kinds_by_path = {item['path']: item['kind'] for item in result['artifacts']}
            self.assertTrue((Path(tmpdir) / 'postprocessing' / 'plot-energy-vs-u.png').exists())
            self.assertTrue((Path(tmpdir) / 'postprocessing' / 'plot-energy-vs-u.spec.json').exists())
            self.assertTrue((Path(tmpdir) / 'postprocessing' / 'plot-energy-vs-u.data.tsv').exists())
            self.assertTrue((Path(tmpdir) / 'postprocessing' / 'plot-energy-vs-u.data.json').exists())
            self.assertIn('postprocessing/plot-energy-vs-u.png', paths)
            self.assertIn('postprocessing/plot-energy-vs-u.spec.json', paths)
            self.assertIn('postprocessing/plot-energy-vs-u.data.tsv', paths)
            self.assertIn('postprocessing/plot-energy-vs-u.data.json', paths)
            self.assertEqual(kinds_by_path['postprocessing/plot-energy-vs-u.png'], 'postprocess-plot')
            self.assertEqual(kinds_by_path['postprocessing/plot-energy-vs-u.spec.json'], 'postprocess-plot-spec')
            self.assertEqual(kinds_by_path['postprocessing/plot-energy-vs-u.data.tsv'], 'postprocess-plot-data-tsv')
            self.assertEqual(kinds_by_path['postprocessing/plot-energy-vs-u.data.json'], 'postprocess-plot-data-json')
            for artifact in result['artifacts']:
                self.assertTrue({'kind', 'path', 'size_bytes', 'mime_type', 'description'} <= set(artifact))
                self.assertNotIn('sha256', artifact)
            data_payload = json.loads((Path(tmpdir) / 'postprocessing' / 'plot-energy-vs-u.data.json').read_text(encoding='utf-8'))
            self.assertEqual(data_payload['schema'], 'pyscf-agent.postprocess-plot-data.v1')
            self.assertEqual(data_payload['plot_artifact_prefix'], 'plot-')
            self.assertEqual(data_payload['plot_spec']['x'], 'U')
            self.assertEqual(len(data_payload['rows']), 3)
            self.assertIn('U_numeric', data_payload['columns'])
            self.assertIn('energy_numeric', data_payload['columns'])
            tsv_text = (Path(tmpdir) / 'postprocessing' / 'plot-energy-vs-u.data.tsv').read_text(encoding='utf-8')
            self.assertIn('case_id\tlabel\tstatus\tU\tenergy\tU_numeric', tsv_text)
            self.assertIn('case-0001', tsv_text)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'matplotlib is not installed')
    def test_run_heatmap_spec_writes_png_and_spec(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = report_with_cases_and_rows(
                [
                    {'case_id': 'case-0001', 'variables': {'U': 2, 't': -1.0}},
                    {'case_id': 'case-0002', 'variables': {'U': 4, 't': -1.0}},
                    {'case_id': 'case-0003', 'variables': {'U': 2, 't': -0.5}},
                    {'case_id': 'case-0004', 'variables': {'U': 4, 't': -0.5}},
                ],
                [
                    {'case_id': 'case-0001', 'U': '2 a.u.', 't': '-1.0 a.u.', 'energy': '-1.2 a.u.'},
                    {'case_id': 'case-0002', 'U': '4 a.u.', 't': '-1.0 a.u.', 'energy': '-0.2 a.u.'},
                    {'case_id': 'case-0003', 'U': '2 a.u.', 't': '-0.5 a.u.', 'energy': '0.3 a.u.'},
                    {'case_id': 'case-0004', 'U': '4 a.u.', 't': '-0.5 a.u.', 'energy': '1.4 a.u.'},
                ],
                work_dir=tmpdir,
            )
            context = PostprocessContext.from_report(report)
            result = run_plot_spec(
                context,
                {'tool': 'heatmap', 'x': 'U', 'y': 't', 'color': 'energy', 'output': 'energy-heatmap'},
                output_dir=Path(tmpdir) / 'postprocessing',
            )

            self.assertTrue((Path(tmpdir) / 'postprocessing' / 'plot-energy-heatmap.png').exists())
            self.assertTrue((Path(tmpdir) / 'postprocessing' / 'plot-energy-heatmap.spec.json').exists())
            self.assertTrue((Path(tmpdir) / 'postprocessing' / 'plot-energy-heatmap.data.tsv').exists())
            self.assertTrue((Path(tmpdir) / 'postprocessing' / 'plot-energy-heatmap.data.json').exists())
            data_payload = json.loads((Path(tmpdir) / 'postprocessing' / 'plot-energy-heatmap.data.json').read_text(encoding='utf-8'))
            self.assertEqual(len(data_payload['rows']), 4)
            self.assertIn('energy_numeric', data_payload['columns'])
            self.assertEqual(result['spec']['colormap'], 'RdBu_r')
            self.assertEqual(result['spec']['center'], 0.0)


if __name__ == '__main__':
    unittest.main()
