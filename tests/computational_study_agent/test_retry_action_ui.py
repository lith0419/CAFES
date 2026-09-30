from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node is required for browser state tests')
class RetryActionUiTests(unittest.TestCase):
    def test_preview_confirm_and_late_response_preserve_full_result_identity(self):
        assets = (
            Path(__file__).resolve().parents[2]
            / 'computational_study_agent'
            / 'web_assets'
        )
        script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const nodes = {};
const context = {JSON, console, Set, WeakMap,
  currentPlan: {cases: [{case_id:'case-0001'}, {case_id:'case-0002'}, {case_id:'case-0003'}]},
  currentReport: {comparison_table:[{case_id:'case-0001'}, {case_id:'case-0002'}, {case_id:'case-0003'}]},
  currentSavedStudy: {studyId:'saved'}, currentReviewGate:null,
  document:{getElementById:id => nodes[id] ||= {value:'', hidden:true, innerHTML:'',
    addEventListener(event, handler) {this[event] = handler;}}},
  savedStudyRequest: () => ({study_id:'saved'}),
  refreshSavedStudy: async () => {},
};
vm.createContext(context);
const load = name => vm.runInContext(fs.readFileSync(process.argv[1] + '/' + name, 'utf8'), context);
vm.runInContext(fs.readFileSync(process.argv[1] + '/../../pyscf_agent/web_assets/shared.js', 'utf8'), context);
load('planner-core.js'); load('planner-retry.js'); load('planner-saved-studies.js');
context.savedStudyRequest = () => ({study_id:'saved'});
context.refreshSavedStudy = async () => {};
const originalPlan = context.currentPlan, originalReport = context.currentReport;
const action = {kind:'retry_cases', action_id:'a'.repeat(32),status:'prepared',
  action:{include_dependents:false}, total_case_count:3, execution_case_ids:['case-0003'],
  changes:[{case_id:'case-0003',path:'runtime.max_cycle',before:50,after:100}],
  cost_estimate:{can_execute:true,totals:{peak_memory_mb:1,work_units:10}}};
const calls=[];
context.postJson = async (url, payload) => {
  calls.push({url,payload});
  if (url.endsWith('context')) return {study_id:'saved',base_study_fingerprint:'revision'};
  if (url.endsWith('prepare')) return action;
  if (url.endsWith('start')) return {started:true};
  throw new Error(url);
};
async function main() {
  await context.openRetryEditor(['case-0003']);
  nodes['retry-parameter'].value='runtime.max_cycle'; nodes['retry-value'].value='100';
  await context.prepareSelectedRetry();
  assert.equal(context.currentPlan, originalPlan); assert.equal(context.currentReport, originalReport);
  assert.match(nodes['retry-preview'].innerHTML, /1 of 3 cases/);
  assert.match(nodes['retry-preview'].innerHTML, /case-0003/);
  assert.deepEqual(Array.from(calls[1].payload.retry_action.case_ids), ['case-0003']);
  assert.equal(calls[1].payload.retry_action.case_overrides['case-0003']['runtime.max_cycle'],100);
  await context.startSavedStudy();
  assert.equal(calls.length,2); // Run opens the preview, not the ordinary Study start.
  await context.confirmPendingRetry();
  assert.equal(calls[2].url,'/api/study-retry-start');
  await context.confirmPendingRetry(); assert.equal(calls.length,3);
  assert.equal(context.currentPlan,originalPlan); assert.equal(context.currentReport,originalReport);
  context.installStudyReviewActionResult({kind:'retry_cases', retry_action:action, can_run:true,
    plan:{cases:[{case_id:'case-0003'}]},study_state:{comparison_table:[{case_id:'case-0003'}]}});
  assert.equal(context.currentPlan,originalPlan); assert.equal(context.currentReport,originalReport);
  assert.throws(() => context.installFullStudyPlan({_review_kind:'static',cases:[]}), /subset/);
  let release;
  context.postJson = () => new Promise(resolve => {release=resolve;});
  const opening = context.openRetryEditor(['case-0001']);
  context.currentSavedStudy = {studyId:'different'};
  release({study_id:'saved',base_study_fingerprint:'revision'}); await opening;
  assert.notEqual(nodes['retry-case-label'].textContent,'case-0001');
}
main().catch(error => {console.error(error);process.exit(1);});
"""
        result = subprocess.run(
            ['node', '-e', script, str(assets)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
