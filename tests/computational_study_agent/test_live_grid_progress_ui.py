import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which('node'), 'Node is required for UI tests')
class LiveGridProgressUiTests(unittest.TestCase):
    def test_live_rows_keep_status_quality_and_scroll_without_replacing_report(self):
        script = r'''const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const nodes={'case-table':{scrollLeft:260,scrollTop:80},'report-summary':{}};
const rows=[{case_id:'seed',status:'pending',publication_eligible:false},
  {case_id:'new-done',status:'succeeded',energy:{value:-1,unit:'a.u.'}},
  {case_id:'new-wait',status:'not_executed',publication_eligible:false}];
const context={document:{getElementById:id=>nodes[id]},escapeHtml:String,
  currentReport:{comparison_table:[{case_id:'seed',status:'succeeded',energy:99}]},
  renderTable:(id,data)=>{nodes[id].rows=data;nodes[id].scrollLeft=0;nodes[id].scrollTop=0;}};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1]+'/planner-rendering.js','utf8'),context);
context.renderLiveStudyTable({progress:{comparison_table:rows}});
assert.deepEqual(nodes['case-table'].rows,rows);
assert.equal(nodes['case-table'].scrollLeft,260);
assert.equal(nodes['case-table'].scrollTop,80);
assert.equal(context.currentReport.comparison_table[0].energy,99);
context.renderRunningGridProgress({agent:{running:true},progress:{comparison_table:rows}},
  {cases:rows,grid_refinement:{enabled:true,max_new_points:8}},null);
assert(nodes['report-summary'].innerHTML.includes('refresh automatically every 5 seconds'));
'''
        self.run_node(script)

    def test_polling_is_read_only_stops_on_finish_and_discards_late_responses(self):
        script = r'''const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const timers=new Map(),requests=[],renders=[];let next=0,resolve,refreshes=0;
const identity={studyId:'scan',workDir:'/tmp/studies',executionTarget:'local'};
const running={agent:{running:true},status:'running',task_count:3};
const stopButton={disabled:true,textContent:''};
const context={document:{getElementById:id=>{assert.equal(id,'stop-saved-study');return stopButton;}},window:{setTimeout:(callback,ms)=>{assert.equal(ms,5000);timers.set(++next,callback);return next;},
  clearTimeout:id=>timers.delete(id)},currentSavedStudy:identity,
  currentPlan:{cases:[{},{},{}],grid_refinement:{enabled:true}},currentReport:{},currentExecution:{},
  postJson:(route,body)=>{requests.push({route,body});return new Promise(done=>{resolve=done;});},
  renderLiveStudyTable:value=>renders.push(value),renderRunningGridProgress:()=>{},
  renderExecutionRecovery:()=>{},setStatus:()=>{},executionStatusSummary:()=>''};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1]+'/planner-saved-studies.js','utf8'),context);
context.refreshSavedStudy=async()=>{refreshes++;context.stopSavedStudyPolling();};
async function tick() {const [id,callback]=timers.entries().next().value;timers.delete(id);return callback();}
async function main(){
 context.scheduleSavedStudyPolling(identity,running);assert.equal(timers.size,1);
 const first=tick();assert.equal(timers.size,0); // No overlapping request while one is in flight.
 resolve({execution:running});await first;assert.equal(renders.length,1);assert.equal(timers.size,1);
 const stale=tick();context.stopSavedStudyPolling();resolve({execution:running});await stale;
 assert.equal(renders.length,1);assert.equal(timers.size,0);
 context.scheduleSavedStudyPolling(identity,running);const changed=tick();
 context.currentSavedStudy={studyId:'other'};resolve({execution:running});await changed;
 assert.equal(renders.length,1);assert.equal(timers.size,0);
 context.currentSavedStudy=identity;context.scheduleSavedStudyPolling(identity,running);
 const expanded=tick();resolve({execution:{...running,task_count:5}});await expanded;
 assert.equal(refreshes,1);assert.equal(timers.size,0);
 context.scheduleSavedStudyPolling(identity,running);
 const finished=tick();resolve({execution:{...running,agent:{running:false}}});await finished;
 assert.equal(refreshes,2);assert.equal(timers.size,0);
 // Static scans also need completion refresh after a selected-case retry.
 delete context.currentPlan.grid_refinement;
 context.scheduleSavedStudyPolling(identity,running);assert.equal(timers.size,1);
 const staticRetry=tick();resolve({execution:{...running,agent:{running:false},status:'interrupted'}});await staticRetry;
 assert.equal(refreshes,3);assert.equal(timers.size,0);
 assert(requests.every(request=>request.route==='/api/study-execution-status'));
 assert(requests.every(request=>request.body.study_id==='scan'&&!('plan' in request.body)));
}
main().catch(error=>{console.error(error);process.exitCode=1;});
'''
        self.run_node(script)

    def run_node(self, script):
        result = subprocess.run(['node', '-e', script,
                                 str(ROOT / 'computational_study_agent/web_assets')],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
