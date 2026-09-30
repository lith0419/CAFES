const assert = require('node:assert/strict'), fs = require('node:fs'), vm = require('node:vm');
const nodes={}, calls=[];let resolveStop, refreshes=0;
const context={document:{getElementById:id=>nodes[id]||=( {disabled:false,textContent:''})},window:{clearTimeout:()=>{}},currentSavedStudy:{studyId:'local-study',workDir:'/tmp/local',executionTarget:'local'},setStatus:()=>{},postJson:(url,request)=>{calls.push({url,request});return new Promise(resolve=>resolveStop=resolve);}};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
context.refreshSavedStudy=async()=>{refreshes++;context.syncStudyStopButton({can_stop:false});};
async function main(){
 context.syncStudyStopButton({can_stop:true});assert.equal(nodes['stop-saved-study'].disabled,false);
 const first=context.stopSavedStudy();assert.equal(nodes['stop-saved-study'].disabled,true);assert.equal(nodes['stop-saved-study'].textContent,'Stopping…');
 await context.stopSavedStudy();assert.equal(calls.length,1);assert.equal(calls[0].url,'/api/study-stop');assert.equal(calls[0].request.study_id,'local-study');
 resolveStop({cancelled:true,message:'Stopped'});await first;assert.equal(refreshes,1);assert.equal(nodes['stop-saved-study'].textContent,'Stop Calculation');assert.equal(nodes['stop-saved-study'].disabled,true);
 context.syncStudyStopButton({can_stop:false});assert.equal(nodes['stop-saved-study'].disabled,true);
 console.log('Stop UI: request identity, duplicate-click guard, pending/terminal states passed.');
}
main().catch(e=>{console.error(e);process.exitCode=1;});
