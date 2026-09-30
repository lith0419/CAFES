from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node is required for browser state tests')
class DmetContinuationUiTests(unittest.TestCase):
    def test_preview_explicit_start_and_late_response_cannot_change_another_study(self):
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
const calls = [];
let refreshed = 0;
const identity = {studyId:'saved'};
const context = {console, currentSavedStudy:identity,
 currentReport:{dmet_phase_analysis:{policy:{charge_threshold:0.001}}},
 escapeHtml: value => String(value), setStatus: () => {},
 savedStudyRequest:() => ({study_id:context.currentSavedStudy.studyId}),
 refreshSavedStudy:async () => {refreshed++;},
 document:{getElementById:id => nodes[id] ||= {innerHTML:'', addEventListener(event, handler){this[event]=handler;}}},
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1]+'/planner-dmet.js','utf8'),context);
const preview = {action_id:'a'.repeat(32),max_runs:6,cost_estimate:{can_execute:true}};
context.postJson = async (url,body) => {calls.push({url,body}); return url.endsWith('prepare') ? preview : {started:true};};
const button = action => ({dataset:{dmetAction:action},disabled:false});
const click = btn => nodes['report-summary'].click({target:{closest:() => btn}});
async function main(){
 await click(button('prepare'));
 assert.equal(calls.length,1);
 assert.ok(nodes['dmet-continuation-preview'].innerHTML.includes('Up to 6'));
 assert.ok(nodes['dmet-continuation-preview'].innerHTML.includes('Density only'));
 await click(button('start'));
 assert.equal(calls.length,2);
 assert.equal(calls[1].body.action_id,preview.action_id);
 assert.equal(calls[1].body.approve_cost,false);
 assert.equal(refreshed,1);
 await click(button('start'));
 assert.equal(calls.length,2);
 let release;
 context.postJson = async () => new Promise(resolve => {release=resolve;});
 const pending = click(button('prepare'));
 context.currentSavedStudy={studyId:'other'};
 nodes['dmet-continuation-preview'].innerHTML='unchanged';
 release(preview); await pending;
 assert.equal(nodes['dmet-continuation-preview'].innerHTML,'unchanged');
 const summary=context.renderDmetBranchSummary({dmet_phase_analysis:{points:[{
   coordinates:{U:2,V:1},candidates:[{},{}],possible_hysteresis:true,
   winner:{branch:'cdw',energy_per_site:-1,run_id:'new-run',density_source:{source_case_id:'case-1',source_run_id:'source-run'}},
 }]}});
 assert.ok(summary.includes('Possible hysteresis'));
 assert.ok(summary.includes('source-run'));
 assert.ok(summary.includes('new-run'));
}
main().catch(error=>{console.error(error);process.exitCode=1;});
"""
        result = subprocess.run(
            ['node', '-e', script, str(assets)], text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
