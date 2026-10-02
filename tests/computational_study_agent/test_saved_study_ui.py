from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which('node'), 'Node is required for browser-state tests')
class SavedStudyUiTests(unittest.TestCase):
    def test_startup_consumes_study_link_and_reload_stays_at_new_task(self):
        assets = Path(__file__).resolve().parents[2] / 'computational_study_agent' / 'web_assets'
        script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const read = name => fs.readFileSync(process.argv[1] + '/' + name, 'utf8');
function start(url, navigationType = 'navigate') {
  const nodes = {}, opens = [];
  let active, ready;
  const context = {
    URL, DEFAULT_WORK_DIR: '/tmp/studies',
    Option: function(text, value) {this.text = text; this.value = value; this.dataset = {};},
    window: {location: {href: url}, addEventListener: () => {},
      performance: {getEntriesByType: () => [{type: navigationType}]}, history: {
      replaceState: (_, __, value) => {context.window.location.href = String(value);},
    }},
    document: {getElementById: id => nodes[id] ||= {value: '', listeners: {}, options: [],
      addEventListener(event, listener) {this.listeners[event] = listener;},
      replaceChildren(...options) {this.options = options;},
      add(option) {this.options.push(option);}}},
    postJson: async route => {
      if (route === '/api/study-examples') {
        return {examples: [{study_id: 'example-one', name: 'Example One'}]};
      }
      assert.equal(route, '/api/study-list');
      return {studies: [{study_id: 'saved-one', name: 'Saved One', status: 'completed',
        work_dir: '/tmp/studies'}]};
    },
    executionWorkDir: () => '/tmp/studies', selectedExecutionTarget: () => 'local',
    setStatus: (_, message, kind) => {assert.notEqual(kind, 'error', message);},
    blankStudySpecForSystem: system_type => ({system_type, base_task: {}}),
    prettyJson: JSON.stringify,
    initializeExecutionTargetSelect: () => new Promise(resolve => {ready = resolve;}),
    activeTaskSession: () => active,
    createTaskSession: () => {active = {}; context.currentReviewGate = null; return active;},
    currentReviewGate: null,
  };
  for (const name of [
    'syncStudySystemFromSpec', 'loadExampleForSelectedSystem', 'draftWithLlm', 'buildPlan',
    'runStudy', 'handleReviewGateAction', 'analyzeStudyResults', 'suggestPostprocessPlots',
    'generateHamiltonianDataset', 'collectHamiltonianDataset', 'addCustomPlotSpec',
    'updateCustomPlotControls', 'handleStudySystemChange', 'handlePlannerTemplateChange',
    'handleStudyModeChange', 'handleDatasetSeedFileChange', 'handleDatasetConfigurationChange',
    'openModelBuilder', 'previewModelHamiltonianStructure', 'handlePlannerModelSolverControlsChange',
    'handleBuilderMessage', 'updateEffectiveWorkDir', 'syncDatasetTemplateControls',
    'syncAdaptiveActiveSpaceControls', 'renderPlanSummary', 'renderPlannerChat',
    'resetModelPreview', 'resetPostprocessing', 'selectEnergyFrames', 'handleGridRefinementChange',
  ]) context[name] = () => {};
  vm.createContext(context);
vm.runInContext(require('node:fs').readFileSync(process.argv[1] + '/../../pyscf_agent/web_assets/shared.js', 'utf8'), context);
  vm.runInContext(read('planner-saved-studies.js'), context);
  context.openSavedStudy = async options => {opens.push(options); context.currentReviewGate = {kind: 'cost'};};
  vm.runInContext(read('planner-init.js'), context);
  return {context, nodes, opens, ready};
}
async function main() {
  const page = start('http://localhost/computational-study/?study=old&work_dir=%2Ftmp%2Fremote&execution_target=remote&theme=dark');
  const clean = page.context.window.location.href;
  assert.equal(new URL(clean).searchParams.has('study'), false);
  assert.equal(new URL(clean).searchParams.has('work_dir'), false);
  assert.equal(new URL(clean).searchParams.has('execution_target'), false);
  assert.equal(new URL(clean).searchParams.get('theme'), 'dark');
  assert.equal(page.opens.length, 0);
  page.ready(); await Promise.resolve();
  await new Promise(resolve => setImmediate(resolve));
  const examples = page.nodes['report-example-select'].options;
  assert.deepEqual(examples.map(option => [option.text, option.value]), [
    ['Select an example', ''], ['Example One', 'example-one'],
  ]);
  await page.context.listSavedStudies();
  const studies = page.nodes['saved-study-list'].options;
  assert.equal(studies.length, 2);
  assert.equal(studies[0].value, '');
  assert.equal(studies[1].value, 'saved-one');
  assert.equal(studies[1].dataset.workDir, '/tmp/studies');
  assert.equal(page.opens.length, 1);
  assert.equal(page.opens[0].studyId, 'old');
  assert.equal(page.opens[0].workDir, '/tmp/remote');
  assert.equal(page.opens[0].executionTarget, 'remote');
  const reload = start(clean, 'reload');
  reload.ready(); await Promise.resolve();
  assert.equal(reload.opens.length, 0);
  assert.equal(reload.context.currentReviewGate, null);
  const oldAddress = start('http://localhost/computational-study/?study=old', 'reload');
  oldAddress.ready(); await Promise.resolve();
  assert.equal(oldAddress.opens.length, 0);
  assert.equal(oldAddress.context.currentReviewGate, null);
  assert.equal(new URL(oldAddress.context.window.location.href).searchParams.has('study'), false);
  const canceled = start('http://localhost/computational-study/?study=old');
  canceled.nodes['new-task'].listeners.click();
  canceled.ready(); await Promise.resolve();
  assert.equal(canceled.opens.length, 0);
  assert.equal(canceled.context.currentReviewGate, null);
}
main().catch(error => {console.error(error); process.exitCode = 1;});
"""
        result = subprocess.run(['node', '-e', script, str(assets)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_late_open_response_does_not_restore_review_over_new_task(self):
        assets = Path(__file__).resolve().parents[2] / 'computational_study_agent' / 'web_assets'
        script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
let active = {}, release, renders = 0, statuses = 0;
const context = {
  currentSavedStudy: null, currentReviewGate: null,
  taskSessions: [],
  activeTaskSession: () => active,
  createTaskSession: () => {active = {};},
  executionWorkDir: () => '/tmp/studies', selectedExecutionTarget: () => 'local',
  postJson: () => new Promise(resolve => {release = resolve;}),
  setStatus: () => {statuses++;},
};
vm.createContext(context);
vm.runInContext(require('node:fs').readFileSync(process.argv[1] + '/../../pyscf_agent/web_assets/shared.js', 'utf8'), context);
vm.runInContext(fs.readFileSync(process.argv[1] + '/planner-saved-studies.js', 'utf8'), context);
context.refreshSavedStudy = async saved => {renders++; context.currentReviewGate = saved.report.pending_review;};
async function main() {
  const opening = context.openSavedStudy({studyId: 'old'});
  active = {}; // New Task or a different selected task.
  release({study_id: 'old', work_dir: '/tmp/studies', report: {pending_review: {}}});
  await opening;
  assert.equal(renders, 0);
  assert.equal(statuses, 0);
  assert.equal(context.currentSavedStudy, null);
  assert.equal(context.currentReviewGate, null);
  const explicit = context.openSavedStudy({studyId: 'selected'});
  release({study_id: 'selected', work_dir: '/tmp/studies', report: {pending_review: {kind: 'cost'}}});
  await explicit;
  assert.equal(renders, 1);
  assert.equal(context.currentSavedStudy.studyId, 'selected');
  assert.equal(context.currentReviewGate.kind, 'cost');
}
main().catch(error => {console.error(error); process.exitCode = 1;});
"""
        result = subprocess.run(['node', '-e', script, str(assets)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_llm_clarification_without_plan_keeps_draft_and_can_be_revised(self):
        import json
        from unittest.mock import patch
        from computational_study_agent.web_api import handle_study_llm_draft_request
        from tests.computational_study_agent.support import hubbard_dimer_spec

        spec = {'name': 'model-sweep', 'objective': 'sweep U', 'system_type': 'model_hamiltonian',
                'base_model_spec': hubbard_dimer_spec(), 'base_task': {},
                'sweep': {'U': [1, 2]}, 'observables': ['energy']}
        invalid = {**spec, 'base_task': {'solver': 'fci'}, 'sweep': {},
                   'case_design': {'mode': 'grid', 'variables': {'defect_site': [0, 1]},
                                   'template': {'operations': [{'op': 'add_site_defect', 'epsilon_shift': -0.5}]}}}
        fixtures = []
        for draft in (spec, invalid, {**spec, 'base_task': {'solver': 'fci'}}):
            with patch('computational_study_agent.web_api.build_study_spec_from_goal_with_evidence',
                       return_value={'study_spec': draft, 'wiki_evidence': {'pages': []}}):
                status, _, body = handle_study_llm_draft_request(json.dumps({'goal': 'sweep U = 1, 2'}).encode())
            self.assertEqual(status.value, 200)
            fixtures.append(json.loads(body))
        for item in fixtures[:2]:
            self.assertEqual(item['status'], 'needs_input')
            self.assertNotIn('plan', item)
        self.assertEqual(len(fixtures[2]['plan']['cases']), 2)
        assets = Path(__file__).resolve().parents[2] / 'computational_study_agent' / 'web_assets'
        script = r"""
const assert = require('node:assert/strict'), fs = require('node:fs'), vm = require('node:vm');
const fixtures = JSON.parse(fs.readFileSync(0, 'utf8'));
const nodes = {}, messages = [], requests = [];
let payload, sessionStatus, tableRows;
const context = {
  document: {getElementById: id => nodes[id] ||= {value: ''}},
  LLM_CONFIGURED: true, currentExecution: null, currentPlan: null, currentReport: null,
  currentAdaptivePreview: null, currentAnalysisRows: [],
  isHamiltonianDatasetPlanner: () => false,
  plannerRequestContext: () => ({studySpec: {}, mode: 'static', adaptive: {}}),
  planningDecomposition: () => '', studyStateForReviewAction: () => ({}),
  syncStudySystemFromSpec: () => {}, syncModelInputFileFromStudySpec: () => {},
  updateTaskSetupPanel: () => {}, renderPlanSummary: () => {}, renderAdaptiveState: () => {},
  summarizePlannerSpec: () => 'Updated model',
  molecularRisk: () => ({}), normalizeSystemType: x => x, currentSystemType: () => 'model_hamiltonian',
  plannerSolverName: x => x, hasCellValue: x => x != null, formatEnergyLikeValue: () => '',
  diagnosticScalarValue: () => null, firstCellValue: () => null,
  molecularRiskComponentValue: () => null, activeSpaceLabel: () => '',
  renderTable: (_, rows) => {tableRows = rows;},
};
vm.createContext(context);
vm.runInContext(require('node:fs').readFileSync(process.argv[1] + '/../../pyscf_agent/web_assets/shared.js', 'utf8'), context);
for (const file of ['planner-core.js', 'planner-rendering.js']) {
  vm.runInContext(fs.readFileSync(process.argv[1] + '/' + file, 'utf8'), context);
}
context.snapshotCurrentTaskSession = () => {};
context.updateActiveTaskSessionStatus = status => {sessionStatus = status;};
context.renderReviewGate = gate => {context.currentReviewGate = gate;};
context.appendPlannerMessage = (role, content) => messages.push({role, content});
context.postJson = async (route, body) => {requests.push({route, body}); return payload;};
async function main() {
  // Both omitted and explicit null plans must reach the useful clarification.
  for (const clarification of [fixtures[0], fixtures[1], {...fixtures[0], plan: null}]) {
    payload = clarification; nodes.goal = {value: 'sweep U'};
    context.currentPlan = {cases: [{case_id: 'old'}]};
    context.currentReviewGate = {kind: 'old'};
    tableRows = [{case_id: 'old'}]; sessionStatus = 'planned'; messages.length = 0;
    await context.draftWithLlm();
    assert.equal(messages.some(m => m.role === 'system'), false, JSON.stringify(messages));
    assert.equal(context.currentPlan, null);
    assert.equal(tableRows.length, 0);
    assert.equal(sessionStatus, 'draft');
    assert.equal(nodes['run-study'].disabled, true);
    assert.equal(nodes['draft-with-llm'].disabled, false);
    assert.equal(context.currentReviewGate, null);
    assert.ok(messages.some(m => m.role === 'assistant' && m.content.includes(clarification.validation_issues[0].message)));
    assert.deepEqual(JSON.parse(nodes['study-spec'].value), clarification.study_spec);
  }
  payload = fixtures[2]; nodes.goal.value = 'use fci';
  await context.draftWithLlm();
  assert.equal(tableRows.length, 2);
  assert.equal(context.currentPlan.study_id, payload.plan.study_id);
  assert.equal(sessionStatus, 'planned');
  assert.equal(nodes['run-study'].disabled, true); // Build Plan still saves before execution.
  assert.ok(messages.at(-1).content.includes('Build Plan'));
  assert.ok(requests.every(x => x.route === '/api/study-llm-draft'));
  // Failed generation is a failed edit, not a new default draft.
  const oldSpec = nodes['study-spec'].value, oldPlan = context.currentPlan;
  const oldRows = tableRows, oldStatus = sessionStatus;
  context.postJson = async () => {
    const error = new Error('Planner returned invalid StudySpec JSON after one contract-repair attempt');
    error.payload = {error: error.message};
    throw error;
  };
  nodes.goal.value = 'Keep the scan and change the solver option';
  await context.draftWithLlm();
  assert.equal(nodes['study-spec'].value, oldSpec);
  assert.equal(context.currentPlan, oldPlan);
  assert.equal(tableRows, oldRows);
  assert.equal(sessionStatus, oldStatus);
  assert.equal(nodes['draft-with-llm'].disabled, false);
  assert.equal(messages.at(-1).role, 'system');
}
main().catch(error => {console.error(error); process.exitCode = 1;});
"""
        result = subprocess.run(['node', '-e', script, str(assets)], input=json.dumps(fixtures),
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_analysis_uses_saved_identity_and_does_not_replace_another_task(self):
        assets = Path(__file__).resolve().parents[2] / 'computational_study_agent' / 'web_assets'
        script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const nodes = {};
let refreshed = 0, calls = [], release;
const identity = {studyId: 'saved'};
const context = {
  currentSavedStudy: identity, currentReport: {study_id: 'saved'}, LLM_CONFIGURED: true,
  document: {getElementById: id => nodes[id] ||= {}},
  savedStudyRequest: () => ({study_id: 'saved', work_dir: '/tmp/studies', execution_target: 'local'}),
  selectedResourceProfile: () => 'small',
  refreshSavedStudy: async () => {refreshed++;},
  adaptiveWorkflow: () => ({}), reviewGateForStudyReport: () => null,
};
vm.createContext(context);
vm.runInContext(require('node:fs').readFileSync(process.argv[1] + '/../../pyscf_agent/web_assets/shared.js', 'utf8'), context);
vm.runInContext(fs.readFileSync(process.argv[1] + '/planner-rendering.js', 'utf8'), context);
context.setStatus = () => {};
context.postJson = async (route, request) => {
  calls.push({route, request});
  return new Promise(resolve => {release = resolve;});
};
async function main() {
  const analyzing = context.analyzeStudyResults();
  assert.equal(calls[0].route, '/api/study-result-analysis');
  assert.equal(calls[0].request.study_id, 'saved');
  assert.equal('report' in calls[0].request, false);
  assert.equal('plan' in calls[0].request, false);
  release({result_analysis: 'Shared results'});
  await analyzing;
  assert.equal(refreshed, 1);
  assert.equal(nodes['study-result-analysis'].textContent, 'Shared results');
  const late = context.analyzeStudyResults();
  context.currentSavedStudy = {studyId: 'another'};
  release({result_analysis: 'Late results'});
  await late;
  assert.equal(refreshed, 1);
  assert.equal(nodes['study-result-analysis'].textContent, 'Shared results');
}
main().catch(error => {console.error(error); process.exitCode = 1;});
"""
        result = subprocess.run(['node', '-e', script, str(assets)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_build_saves_once_and_attaches_to_the_original_draft(self):
        assets = Path(__file__).resolve().parents[2] / 'computational_study_agent' / 'web_assets'
        script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const nodes = {'build-plan': {disabled: false}};
const firstSession = {};
let active = firstSession, mode = 'static', dataset = false, submitted = [], prepared = null;
let release, starts = 0;
const context = {
  currentSavedStudy: null,
  document: {getElementById: id => nodes[id] ||= {}},
  activeTaskSession: () => active, selectedExecutionTarget: () => 'local',
  selectedResourceProfile: () => 'small', executionWorkDir: () => '/tmp/studies',
  isHamiltonianDatasetPlanner: () => dataset,
  plannerRequestContext: () => ({studySpec: {name: 'draft', study_mode: mode}, adaptive: {initial_scan: 'mp2'}}),
  hamiltonianDatasetPlanningPayload: () => ({planner_template: 'hamiltonian_dataset', dataset_spec: {dataset_id: 'small'}}),
  deepCopy: structuredClone, snapshotCurrentTaskSession: () => {}, renderTaskSessionSelector: () => {},
  appendPlannerMessage: () => {},
  refreshSavedStudy: async saved => {prepared = saved;},
  startSavedStudy: async () => {starts++;},
  currentStudyMode: () => mode,
};
vm.createContext(context);
vm.runInContext(require('node:fs').readFileSync(process.argv[1] + '/../../pyscf_agent/web_assets/shared.js', 'utf8'), context);
vm.runInContext(fs.readFileSync(process.argv[1] + '/planner-rendering.js', 'utf8'), context);
context.setStatus = () => {};
context.postJson = async (route, request) => {
  submitted.push({route, request});
  return new Promise(resolve => {release = resolve;});
};
async function main() {
  await context.runStudy();
  assert.equal(starts, 0); // Draft previews cannot be executed.
  for (const type of ['static', 'adaptive', 'dataset']) {
    mode = type === 'adaptive' ? 'adaptive' : 'static';
    dataset = type === 'dataset';
    context.currentSavedStudy = null;
    nodes['build-plan'].disabled = false;
    const before = submitted.length;
    const building = context.buildPlan();
    await context.buildPlan(); // Repeated clicks while saving create only one Study.
    assert.equal(submitted.length, before + 1);
    const call = submitted.at(-1);
    assert.equal(call.route, '/api/study-prepare');
    assert.equal(call.request.work_dir, '/tmp/studies');
    assert.equal(call.request.resource_profile, 'small');
    if (dataset) assert.equal(call.request.dataset_spec.dataset_id, 'small');
    else assert.equal(call.request.study_spec.study_mode, mode);
    release({study_id: type, work_dir: '/tmp/studies', mode, plan: {cases: []}});
    await building;
    assert.equal(context.currentSavedStudy.studyId, type);
    assert.equal(firstSession.currentSavedStudy.studyId, type);
    assert.equal(prepared.study_id, type);
    assert.equal(nodes['build-plan'].disabled, true);
    await context.runStudy();
  }
  assert.equal(starts, 3);
  // A late prepare response belongs to the original draft, not a newly selected one.
  context.currentSavedStudy = null;
  nodes['build-plan'].disabled = false;
  const building = context.buildPlan();
  active = {};
  release({study_id: 'late', work_dir: '/tmp/studies', mode: 'static'});
  await building;
  assert.equal(firstSession.currentSavedStudy.studyId, 'late');
  assert.equal(active.currentSavedStudy, undefined);
  assert.equal(context.currentSavedStudy, null);
}
main().catch(error => {console.error(error); process.exitCode = 1;});
"""
        result = subprocess.run(['node', '-e', script, str(assets)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_retry_preserves_full_report_and_async_plot_context(self):
        assets = Path(__file__).resolve().parents[2] / 'computational_study_agent' / 'web_assets'
        script = r"""
const assert = require('node:assert/strict'), fs = require('node:fs'), vm = require('node:vm');
async function check(outcome, finishesDuringRefresh = false) {
  const nodes = {}, timers = new Map(), calls = [];
  const identity = {studyId: 'scan', workDir: '/tmp/studies', executionTarget: 'local'};
  const rows = [{case_id: 'a', status: 'succeeded', energy: -2},
    {case_id: 'b', status: 'failed'}, {case_id: 'c', status: 'succeeded', energy: -3}];
  const fullPlan = {cases: rows.map(row => ({case_id: row.case_id}))};
  const report = {status: 'completed_with_issues', summary: '3 cases', comparison_table: rows,
    pending_review: {status: 'prepared', message: 'Retry prepared: b', can_run: true,
      plan: {cases: [{case_id: 'b'}]}}};
  const saved = {study_id: 'scan', work_dir: identity.workDir, mode: 'static', plan: fullPlan, report};
  const finalRows = rows.map(row => row.case_id === 'b' ? {...row, status: outcome,
    ...(outcome === 'succeeded' ? {energy: -4} : {})} : row);
  const finalSaved = {...saved, report: {summary: '3 cases after retry', comparison_table: finalRows,
    status: outcome === 'succeeded' ? 'succeeded' : 'completed_with_issues'}};
  let active = !finishesDuringRefresh, contextPromise, releaseContext;
  const context = {
    currentSavedStudy: identity, currentReviewGate: null, currentPlan: null, currentReport: null,
    currentAdaptivePreview: null, currentAnalysisRows: [], currentPlotSpecs: [],
    document: {getElementById: id => nodes[id] ||= {innerHTML: ''}},
    window: {setTimeout: callback => {timers.set(1, callback); return 1;}, clearTimeout: id => timers.delete(id)},
    deepCopy: value => structuredClone(value), escapeHtml: String,
    syncStudyModeControls: () => {}, lockWorkDirectory: () => {}, snapshotCurrentTaskSession: () => {},
    setStatus: () => {}, executionStatusSummary: () => '', renderExecutionRecovery: () => {},
    renderLiveStudyTable: () => {}, renderRunningGridProgress: () => {},
    renderPlan: plan => {context.currentPlan = plan; context.currentReviewGate = null;},
    renderReport: value => {
      context.currentReport = value; nodes['report-summary'] = {innerHTML: value.summary};
      contextPromise = context.refreshPostprocessContext();
    },
    renderReviewGate: gate => {context.currentReviewGate = gate;}, planCostReviewGate: () => null,
    postJson: async (route, body) => {
      calls.push({route, body});
      if (route === '/api/study-postprocess-suggestions') {
        return new Promise(resolve => {releaseContext = () => resolve({context: {
          rows: body.report.comparison_table.filter(row => row.status === 'succeeded'), excluded_rows: [],
        }});});
      }
      if (route === '/api/study-open') return finalSaved;
      assert.equal(route, '/api/study-execution-status');
      return {execution: {task_count: 3, status: active ? 'running' : 'results_available',
        agent: {running: active, finished_at: active ? null : '2026-09-25'},
        study_status: active ? report.status : finalSaved.report.status}};
    },
  };
  vm.createContext(context);
vm.runInContext(require('node:fs').readFileSync(process.argv[1] + '/../../pyscf_agent/web_assets/shared.js', 'utf8'), context);
  for (const name of ['planner-core.js', 'planner-postprocessing.js', 'planner-saved-studies.js'])
    vm.runInContext(fs.readFileSync(process.argv[1] + '/' + name, 'utf8'), context);
  context.updateCustomPlotControls = () => {};
  context.snapshotCurrentTaskSession = () => {};
  context.renderReviewGate = gate => {context.currentReviewGate = gate;};
  await context.refreshSavedStudy(saved);
  assert.equal(context.currentPlan, fullPlan);
  assert.equal(context.currentReport, finishesDuringRefresh ? finalSaved.report : report);
  assert.equal(context.currentReport.comparison_table.length, 3);
  assert(nodes['report-summary'].innerHTML.includes('3 cases'));
  releaseContext(); await contextPromise;
  assert.equal(context.postprocessRows().length, finishesDuringRefresh && outcome === 'succeeded' ? 3 : 2);
  assert.equal(nodes['generate-default-plots'].disabled, false);
  assert.equal(nodes['suggest-plots'].disabled, false);
  if (!finishesDuringRefresh) {
    assert(nodes['report-summary'].innerHTML.includes('Retry prepared: b'));
    assert.equal(timers.size, 1);
    active = false;
    const callback = timers.get(1); timers.delete(1); await callback();
    releaseContext(); await contextPromise;
  }
  assert.equal(context.currentPlan, fullPlan);
  assert.equal(context.currentReport, finalSaved.report);
  assert.equal(context.postprocessRows().length, outcome === 'succeeded' ? 3 : 2);
  assert.equal(nodes['generate-default-plots'].disabled, false);
  assert.equal(timers.size, 0);
  assert.equal(calls.filter(call => call.route === '/api/study-open').length, 1);
  assert(calls.every(call => !call.route.includes('start') && !call.route.includes('collect')));
}
(async () => {
  await check('succeeded'); await check('failed');
  await check('succeeded', true); await check('failed', true);
})().catch(error => {console.error(error); process.exitCode = 1;});
"""
        result = subprocess.run(['node', '-e', script, str(assets)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_deep_link_review_and_static_retry_display(self):
        assets = Path(__file__).resolve().parents[2] / 'computational_study_agent' / 'web_assets'
        script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const nodes = {};
const calls = [];
const identity = {studyId: 'example', workDir: '/tmp/studies', executionTarget: 'local'};
const context = {
  URL, currentSavedStudy: null, currentReviewGate: null, currentPlan: null,
  DEFAULT_WORK_DIR: '/tmp/studies',
  document: {getElementById: id => nodes[id] ||= {}},
  window: {location: {href: 'http://localhost/computational-study/?study=example'},
    performance: {getEntriesByType: () => [{type: 'navigate'}]},
    history: {replaceState: (_, __, url) => context.window.location.href = String(url)}},
  deepCopy: value => structuredClone(value), escapeHtml: String, currentStudyMode: () => 'static',
  syncStudyModeControls: () => {},
  lockWorkDirectory: () => {}, snapshotCurrentTaskSession: () => {}, setStatus: () => {},
  renderPlan: plan => {context.currentPlan = plan; context.currentReviewGate = null;},
  renderReport: report => {context.currentReport = report;},
  renderReviewGate: gate => {context.currentReviewGate = gate;},
  installStudyReviewActionResult: result => {context.currentPlan = result.plan;},
  planCostReviewGate: plan => plan.cost_estimate && !plan.cost_estimate.approved ? {kind: 'cost'} : null,
  executionStatusSummary: result => result.status,
  postJson: async (route, request) => {
    calls.push({route, request});
    return {execution: {status: 'results_available', can_collect: true}};
  },
};
vm.createContext(context);
vm.runInContext(require('node:fs').readFileSync(process.argv[1] + '/../../pyscf_agent/web_assets/shared.js', 'utf8'), context);
vm.runInContext(fs.readFileSync(process.argv[1] + '/planner-saved-studies.js', 'utf8'), context);
vm.runInContext(fs.readFileSync(process.argv[1] + '/planner-adaptive-review.js', 'utf8'), context);
async function main() {
  // An explicit link is captured once; normal refresh starts a new draft.
  const requested = context.takeSavedStudyLocation();
  assert.equal(requested.studyId, 'example');
  assert.equal(requested.workDir, '/tmp/studies');
  assert.equal(new URL(context.window.location.href).searchParams.has('study'), false);
  context.syncSavedStudyControls();
  context.currentSavedStudy = identity;
  context.syncSavedStudyControls();
  assert.equal(new URL(context.window.location.href).searchParams.has('study'), false);
  const plan = {study_id: 'example', cases: [{case_id: 'a'}], cost_estimate: {approved: false}};
  const fullPlan = {...plan, cases: [{case_id: 'a'}, {case_id: 'b'}]};
  const saved = {study_id: 'example', work_dir: identity.workDir, mode: 'static', plan: fullPlan,
    report: {status: 'succeeded', pending_review: {status: 'prepared', can_run: true, plan}}};
  await context.refreshSavedStudy(saved);
  assert.equal(nodes['run-study'].disabled, true); // Explicitly opening a Study still enforces cost approval.
  assert.equal(context.currentReviewGate.kind, 'cost');
  assert.equal(nodes['collect-saved-study'].disabled, true); // Old results are not the pending retry.
  assert.equal(context.currentPlan, fullPlan); // Review gates apply to the subset; display stays complete.
  assert.equal(context.currentReport, saved.report);
  plan.cost_estimate.approved = true;
  await context.refreshSavedStudy(saved);
  assert.equal(nodes['run-study'].disabled, false);
  assert.ok(calls.every(call => call.route === '/api/study-execution-status'));
  assert.ok(calls.every(call => !('plan' in call.request) && call.request.study_id === 'example'));
  context.renderAdaptiveState({adaptive: {retry_plan: plan, workflow: {}}});
  assert.equal(nodes['adaptive-panel'].className, 'adaptive-panel hidden');
  assert.equal(context.takeSavedStudyLocation(), null);
  assert.equal(new URL(context.window.location.href).searchParams.has('study'), false);
  context.window.location.href = 'http://localhost/computational-study/?study=example';
  assert.equal(context.takeSavedStudyLocation().studyId, identity.studyId);
  assert.equal(context.takeSavedStudyLocation(), null);
  // A saved scientific review must wait for an explicit agent start.
  vm.runInContext(fs.readFileSync(process.argv[1] + '/planner-core.js', 'utf8'), context);
  context.refreshSavedStudy = async () => {};
  context.snapshotCurrentTaskSession = () => {};
  context.appendPlannerMessage = () => {};
  await context.applyPlannerReviewAction('approve_cost_estimate', ['a'], {});
  const reviewRequest = calls.at(-1);
  assert.equal(reviewRequest.route, '/api/study-review-action');
  assert.equal(reviewRequest.request.study_id, 'example');
  assert.equal('plan' in reviewRequest.request, false);
  assert.equal('study_state' in reviewRequest.request, false);
  let analysisCalls = 0;
  context.currentReviewGate = {actions: [{id: 'skip_path_restart'}]};
  context.applyPlannerReviewAction = async () => ({analysis_approval: {approved: true}});
  context.analyzeStudyResults = async () => {analysisCalls++;};
  await context.handleReviewGateAction({target: {closest: () => ({getAttribute: () => 'skip_path_restart'})}});
  assert.equal(analysisCalls, 0);
}
main().catch(error => {console.error(error); process.exitCode = 1;});
"""
        result = subprocess.run(['node', '-e', script, str(assets)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
