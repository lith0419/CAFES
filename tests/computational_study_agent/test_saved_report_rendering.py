"""Saved-report rendering must work without example-specific identifiers."""

import shutil
import subprocess
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which('node'), 'node is not installed')
class SavedReportRenderingTests(unittest.TestCase):
    def run_script(self, script):
        assets = Path(__file__).resolve().parents[2] / 'computational_study_agent' / 'web_assets'
        source = (assets / 'planner-rendering.js').read_text()
        subprocess.run(['node'], input=source + '\n' + script,
                       check=True, capture_output=True, text=True)

    def test_missing_receipts_preserve_saved_report_and_live_status_distinction(self):
        self.run_script("""
          const assert = require('node:assert/strict');
          for (const system_type of ['molecular', 'model_hamiltonian', 'periodic']) {
            for (const execution_status of ['succeeded', 'failed']) {
              const report = {study_id: 'arbitrary-study', system_type,
                cases: [{case_id: 'arbitrary-case', task_report: {execution_status}}]};
              const missing = executionStatusSummary({status: 'not_found'}, report);
              assert(missing.includes('saved task reports (execution receipts unavailable)'));
              for (const status of ['running', 'interrupted', 'cancelled', 'results_available']) {
                const text = executionStatusSummary({status}, report);
                assert(text.includes('Execution: ' + status.replace(/_/g, ' ')));
                assert(!text.includes('execution receipts unavailable'));
              }
            }
          }
          for (const report of [null, {}, {cases: []}, {cases: [null, {}]},
              {cases: [{task_report: {}}, {task_report: []}]},
              {example_provenance: {imported: true}}]) {
            assert(executionStatusSummary({status: 'not_found'}, report).includes('Execution: not found'));
          }
          const report = {cases: [{task_report: {execution_status: 'succeeded'}}]};
          assert(executionStatusSummary(null, report).includes('Execution: unknown'));
          const counts = executionStatusSummary({status: 'not_found', task_count: 3,
            task_status_counts: {succeeded: 2, failed: 1}}, report);
          assert(counts.includes('3 planned points: 2 succeeded, 1 failed'));
        """)

    def test_report_restores_all_plot_formats_without_a_directory_convention(self):
        self.run_script("""
          const assert = require('node:assert/strict');
          global.document = {getElementById: () => ({})};
          global.LLM_CONFIGURED = false;
          global.currentReport = null;
          global.currentAnalysisRows = [];
          global.currentAdaptivePreview = null;
          for (const name of ['renderDatasetManifestSummary', 'renderGridRefinementSummary',
              'renderDmetBranchSummary', 'reviewGateForStudyReport']) global[name] = () => '';
          for (const name of ['renderTable', 'renderAdaptiveState', 'renderReviewGate',
              'updateCustomPlotControls', 'updateActiveTaskSessionStatus',
              'snapshotCurrentTaskSession', 'refreshPostprocessContext']) global[name] = () => {};
          global.escapeHtml = value => String(value);
          global.deepCopy = value => JSON.parse(JSON.stringify(value));
          global.postprocessRows = () => [{}];
          let restored = [];
          global.resetPostprocessing = () => { restored = []; };
          global.renderPostprocessArtifacts = artifacts => { restored = artifacts; };
          const plots = [
            {kind: 'postprocess-plot', path: 'arbitrary/energy.png', mime_type: 'image/png'},
            {kind: 'postprocess-plot', path: 'other/spectrum.svg', mime_type: 'image/svg+xml'},
            {kind: 'postprocess-plot', path: 'results/bands.pdf', mime_type: 'application/pdf'},
          ];
          for (const system_type of ['molecular', 'model_hamiltonian', 'periodic']) {
            renderReport({system_type, status: 'succeeded', cases: [], artifacts: [
              null, ...plots, {kind: 'input-image', path: 'input.png', mime_type: 'image/png'},
              {kind: 'postprocess-plot-spec', path: 'spec.json'},
            ]});
            assert.deepEqual(restored, plots);
          }
          for (const artifacts of [undefined, null, {}, []]) {
            renderReport({status: 'succeeded', artifacts});
            assert.deepEqual(restored, []);
          }
        """)
