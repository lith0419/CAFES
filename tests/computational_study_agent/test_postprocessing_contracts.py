from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from computational_study_agent import postprocessing as pp
from computational_study_agent.web_api import (
    handle_study_postprocess_request, handle_study_postprocess_suggestions_request,
)
from pyscf_agent.application.calculation_service import CalculationApplicationService
from pyscf_agent.request_builder import llm
from tests.computational_study_agent.test_postprocessing import sample_report


class PostprocessingContractTests(unittest.TestCase):
    def test_eligibility_shared_by_preview_and_execution_including_legacy_rows(self):
        report = sample_report()
        report['comparison_table'][0]['status'] = 'failed'
        report['comparison_table'][1]['publication_eligible'] = False
        context = pp.PostprocessContext.from_report(report)
        self.assertEqual([r['case_id'] for r in context.comparison_table], ['case-0003'])
        status, _, body = handle_study_postprocess_suggestions_request(json.dumps({
            'report': report, 'include_suggestions': False,
        }).encode())
        self.assertEqual(status.value, 200)
        view = json.loads(body)['context']
        self.assertEqual(view['rows'], context.comparison_table)
        self.assertEqual(len(view['excluded_rows']), 2)
        self.assertNotIn('publication_eligible', view['columns'])
        self.assertEqual(set(view['numeric_columns']), {'U', 'energy'})
        # A stale successful comparison row cannot override its failed Task.
        report['cases'] = [{'case_id': 'case-0003', 'task_report': {'execution_status': 'failed'}}]
        self.assertFalse(pp.PostprocessContext.from_report(report).comparison_table)

    def test_units_use_case_metadata_and_preserve_unknown_units(self):
        report = {'system_type': 'molecular', 'cases': [
            {'case_id': 'a', 'request': {'unit': 'Bohr'}},
            {'case_id': 'b', 'request': {'unit': 'Angstrom'},
             'task_report': {'task_spec': {'system': {'unit': 'Bohr'}}}},
        ], 'comparison_table': [
            {'case_id': 'a', 'bond': 1, 'energy': '-1 Ha'},
            {'case_id': 'b', 'bond': 2, 'energy': '-0.9 Ha'},
        ]}
        context = pp.PostprocessContext.from_report(report)
        spec = pp._normalized_plot_spec(context, {'x': 'bond', 'y': 'energy'})
        self.assertEqual(spec['xlabel'], 'Bond length')
        self.assertEqual(pp._plot_data_payload(context, spec)['rows'][0]['bond_unit'], 'Bohr')
        report['cases'] = []
        unknown = pp._normalized_plot_spec(pp.PostprocessContext.from_report(report), {'x': 'bond', 'y': 'energy'})
        self.assertEqual(unknown['xlabel'], 'Bond length')

    def test_mixed_units_rejected_before_artifacts_and_api_returns_input_error(self):
        report = sample_report()
        report['comparison_table'][1]['energy'] = '-27.211386 eV'
        original = copy.deepcopy(report)
        with tempfile.TemporaryDirectory() as tmp, patch.object(pp, '_plot_rows') as render:
            with self.assertRaisesRegex(ValueError, 'mixed or unknown units'):
                pp.run_postprocessing(report, [{'x': 'U', 'y': 'energy'}], output_dir=tmp)
            self.assertEqual(list(Path(tmp).iterdir()), [])
            render.assert_not_called()
        self.assertEqual(report, original)
        status, _, body = handle_study_postprocess_request(json.dumps({
            'report': report, 'plot_specs': [{'x': 'U', 'y': 'energy'}],
        }).encode())
        self.assertEqual(status.value, 400)
        self.assertIn('units', json.loads(body)['error'])

    def test_duplicate_heatmap_coordinates_rejected_for_both_row_orders(self):
        rows = [{'x': x, 'y': y, 'z': x + y} for x in (0, 1) for y in (0, 1)]
        rows.append({'x': '0', 'y': 0, 'z': 999})
        spec = {'tool': 'heatmap', 'x': 'x', 'y': 'y', 'color': 'z'}
        for records in (rows, list(reversed(rows))):
            with tempfile.TemporaryDirectory() as tmp, patch.object(pp, '_plot_rows') as render:
                with self.assertRaisesRegex(ValueError, 'Duplicate heatmap coordinate'):
                    pp.run_plot_spec(pp.PostprocessContext.from_report({'comparison_table': records}), spec, output_dir=Path(tmp))
                self.assertFalse(list(Path(tmp).iterdir()))
                render.assert_not_called()

    def test_invalid_explicit_plot_options_do_not_become_defaults(self):
        context = pp.PostprocessContext.from_report(sample_report())
        base = {'x': 'U', 'y': 'energy'}
        for options in ({'dpi': 0}, {'dpi': 1.5}, {'dpi': True}, {'figure_size': [1]},
                        {'figure_size': [1, float('nan')]}, {'figure_size': [1, -1]},
                        {'group': 'solvre'}, {'style': 'unknown'}, {'sort_by_x': 'maybe'}, {'srot_by_x': False}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                pp._normalized_plot_spec(context, dict(base, **options))
        normalized = pp._normalized_plot_spec(context, dict(base, sort_by_x='false'))
        self.assertFalse(normalized['sort_by_x'])
        self.assertEqual(normalized['dpi'], 600)

    def test_heatmap_rejects_unsupported_grouping_and_nonfinite_center(self):
        context = pp.PostprocessContext.from_report(sample_report())
        for extra in ({'group': 'solver'}, {'center': float('inf')}, {'center': True}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                pp._normalized_plot_spec(context, {'tool': 'heatmap', 'x': 'U', 'y': 'energy', 'color': 'U', **extra})

    def test_numeric_exclusions_are_visible_and_plot_matches_data_export(self):
        report = sample_report()
        report['comparison_table'][1]['energy'] = '1e999'
        report['comparison_table'].append({'case_id': 'bad-string', 'U': '8 a.u.', 'energy': '1.2 eV extra'})
        with tempfile.TemporaryDirectory() as tmp, patch.object(pp, '_plot_rows', return_value={}) as render:
            result = pp.run_postprocessing(report, [{'x': 'U', 'y': 'energy'}], output_dir=tmp)
            plotted = render.call_args.args[0]
            data = json.loads(next(Path(tmp).glob('*.data.json')).read_text())
        self.assertEqual([row['case_id'] for row in plotted], ['case-0001', 'case-0003'])
        self.assertEqual([row['energy'] for row in plotted], [row['energy_numeric'] for row in data['rows']])
        self.assertEqual(result['plots'][0]['row_count'], 2)
        self.assertEqual(len(result['plots'][0]['excluded_rows']), 2)
        self.assertEqual(data['excluded_rows'], result['plots'][0]['excluded_rows'])

    def test_preflight_all_selected_plots_before_rendering_and_no_output_collisions(self):
        report = sample_report()
        for specs in ([{'x': 'U', 'y': 'energy'}, {'x': 'U', 'y': 'typo'}],
                      [{'x': 'U', 'y': 'energy', 'output': 'x'}, {'x': 'U', 'y': 'energy', 'output': 'plot-x'}]):
            with tempfile.TemporaryDirectory() as tmp, patch.object(pp, '_plot_rows') as render:
                with self.assertRaises(ValueError):
                    pp.run_postprocessing(report, specs, output_dir=tmp)
                render.assert_not_called()
                self.assertFalse(list(Path(tmp).iterdir()))

    def test_analysis_sends_only_present_contracts_and_retains_complete_small_table(self):
        rows = [{'U': value, 'energy': -value} for value in range(16)]
        request = {'planner_study_result': {'system_type': 'model_hamiltonian', 'comparison_table': rows}}
        body = None
        def post(_url, payload, _headers, _timeout):
            nonlocal body
            body = payload
            return {'choices': [{'message': {'content': 'Analysis fixture.'}}]}
        with patch.dict(os.environ, {'PYSCF_AGENT_LLM_BASE_URL': 'https://offline.invalid', 'PYSCF_AGENT_LLM_MODEL': 'test'}):
            llm.build_result_analysis(json.dumps(request), {'execution_status': 'succeeded'}, http_post=post)
        user = json.loads(body['messages'][1]['content'])
        self.assertEqual(user['prepared_request'], request)
        self.assertEqual(set(user['registered_output_contracts']), {'energy'})
        self.assertNotIn('Hartree/cell', json.dumps(user['registered_output_contracts']))
        self.assertNotIn('Registered output contracts:', body['messages'][0]['content'])

    def test_actual_orbital_artifacts_keep_their_contract(self):
        context = llm._result_context('{}', {'task_spec': {'task_type': 'molecular'},
            'artifacts': [{'kind': 'orbital_summary_table', 'path': 'orbitals.tsv'}]}, 'en')
        self.assertIn('orbital_processing', context['registered_output_contracts'])
        self.assertNotIn('strong_correlation_diagnostics', context['registered_output_contracts'])

    def test_feedback_content_is_not_filtered_by_language(self):
        with patch.dict(os.environ, {'PYSCF_AGENT_LLM_BASE_URL': 'https://offline.invalid', 'PYSCF_AGENT_LLM_MODEL': 'test'}):
            for text in ('无需调整。', 'No adjustment is needed.'):
                result = llm.build_execution_feedback('Run.', {'execution_status': 'failed'},
                    http_post=lambda *args: {'choices': [{'message': {'content': text}}]})
                self.assertEqual(result['content'], text)

    def test_optional_feedback_failure_is_recorded_without_changing_result(self):
        builder = Mock()
        builder.build_execution_feedback.side_effect = RuntimeError('Provider unavailable')
        service = CalculationApplicationService(llm_request_builder=builder)
        report = {'execution_status': 'succeeded', 'structured_results': {'energy': -1.0}}
        service._append_execution_feedback('Run.', report, locale='en')
        self.assertEqual(report['execution_status'], 'succeeded')
        self.assertEqual(report['structured_results']['energy'], -1.0)
        self.assertEqual(report['llm_feedback'], {'status': 'failed', 'reason': 'Provider unavailable'})

    @unittest.skipUnless(shutil.which('node'), 'Node is required for browser function verification')
    def test_browser_uses_backend_projection_and_ignores_stale_responses(self):
        source = Path(pp.__file__).parent / 'web_assets/planner-postprocessing.js'
        script = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const waiters = [], requests = [], nodes = {};
const ctx = {currentReport: {comparison_table: [{case_id:'failed', status:'failed'}, {case_id:'ok', status:'succeeded'}]},
  currentAnalysisRows: [], currentPlotSpecs: [],
  document: {getElementById: id => nodes[id] || (nodes[id] = {value: '', disabled:false})},
  escapeHtml: String, deepCopy: x => JSON.parse(JSON.stringify(x)), setStatus: () => {},
  errorDetailMessage: (_, msg) => {throw new Error(msg);},
  postJson: async (url, body) => {requests.push(body); return await new Promise(resolve => waiters.push(resolve));},
};
vm.createContext(ctx); vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), ctx);
(async () => {
 const old = ctx.refreshPostprocessContext();
 assert.equal(requests[0].report.comparison_table.length, 2);
 assert.equal(requests[0].include_suggestions, false);
 ctx.currentReport = {comparison_table:[{case_id:'new'}]};
 const latest = ctx.refreshPostprocessContext();
 const projection = {rows:[{case_id:'new'}], columns:['x'], numeric_columns:['x'], case_variable_columns:[], excluded_rows:[]};
 waiters[1]({context:projection}); await latest;
 waiters[0]({context:{...projection, rows:[{case_id:'old'}]}}); await old;
 assert.equal(ctx.postprocessRows()[0].case_id, 'new');
 assert.deepEqual(ctx.numericComparisonColumns(), ['x']);
 assert.equal(typeof ctx.parseNumericCell, 'undefined');
})().catch(error => { console.error(error); process.exit(1); });
'''
        result = subprocess.run(['node', '-e', script, str(source)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
