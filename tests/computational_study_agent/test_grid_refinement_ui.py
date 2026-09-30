import shutil
import subprocess
import unittest
from pathlib import Path

from tests.pyscf_agent.test_dmet_beta_web_ui import DOM

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which('node'), 'Node is required for UI contract tests')
class GridRefinementUiTests(unittest.TestCase):
    def test_controls_round_trip_axis_specific_limits_and_metric_tolerances(self):
        script = DOM + (ROOT / 'computational_study_agent/web_assets/planner-spec.js').read_text()
        script += r'''
const policy = {enabled:true, axes:['U_value','V_value'],max_new_points:17,max_rounds:3,
  min_spacing:{U_value:0.01,V_value:0.125},max_interval:{V_value:0.5},
  metrics:[{name:'energy_per_site',atol:0.003,rtol:0,required:true},
           {name:'mean_double_occupancy',atol:0.002,rtol:0.03,required:false}]};
const spec = {system_type:'model_hamiltonian',grid_refinement:policy};
syncGridRefinementControls(spec);
assert.equal(control('grid-max-points').value,'17');
assert(control('grid-axis-1').innerHTML.includes('value="U_value"'));
assert(control('grid-axis-2').innerHTML.includes('value="V_value"'));
assert(control('grid-axis-2').innerHTML.includes('value="">None'));
const expected = {...policy, metrics: policy.metrics.map(metric => ({...metric, rtol:0, required:true}))};
assert.deepEqual(applyGridRefinementControls({system_type:'model_hamiltonian'}).grid_refinement,expected);
assert.equal(policy.metrics[1].required,false);
control('grid-max-points').value='8';
assert.equal(applyGridRefinementControls({system_type:'model_hamiltonian'}).grid_refinement.max_new_points,8);
control('grid-max-points').value='-2';
assert.throws(()=>applyGridRefinementControls({system_type:'model_hamiltonian'}));
syncGridRefinementControls(spec);
control('grid-axis-2').value='U_value';
assert.throws(()=>applyGridRefinementControls({system_type:'model_hamiltonian'}));
syncGridRefinementControls(spec);
assert.equal(applyGridRefinementControls({system_type:'molecular',grid_refinement:policy}).grid_refinement,undefined);
control('grid-refinement-enabled').checked=false;
assert.equal(applyGridRefinementControls({system_type:'model_hamiltonian',grid_refinement:policy}).grid_refinement,undefined);
syncGridRefinementControls({system_type:'model_hamiltonian',sweep:{U:[2,4,8],V:[0.5,1],solver:['fci','ccsd']}});
assert.equal(control('grid-axis-1').value,'U');
assert.equal(control('grid-axis-2').value,'V');
assert(!control('grid-axis-1').innerHTML.includes('value="solver"'));
control('grid-refinement-enabled').checked=true;
control('grid-axis-2').value='';
const defaults=applyGridRefinementControls({system_type:'model_hamiltonian'}).grid_refinement;
assert.deepEqual(defaults.axes,['U']);
assert.deepEqual(defaults.metrics.map(m=>m.name),['energy_per_site','mean_double_occupancy']);
assert(defaults.metrics.every(m=>m.required && m.rtol===0));
control('grid-atol-energy_per_site').value='0';
assert.throws(()=>applyGridRefinementControls({system_type:'model_hamiltonian'}));
control('grid-atol-energy_per_site').value='0.001';
GRID_METRIC_NAMES.forEach(name=>control(`grid-metric-${name}`).checked=false);
assert.throws(()=>applyGridRefinementControls({system_type:'model_hamiltonian'}));
'''
        result = subprocess.run(['node', '-e', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_results_explain_sampling_stop_without_claiming_convergence(self):
        script = DOM + (ROOT / 'computational_study_agent/web_assets/planner-spec.js').read_text()
        script += r'''
function escapeHtml(s) {return String(s).replaceAll('<','&lt;').replaceAll('>','&gt;');}
const html=renderGridRefinementSummary({policy:{max_new_points:8},status:'budget_exhausted',
  seed_point_count:4,new_point_count:6,decisions:[{axis:'U',interval:[0,4],midpoint:2,
  point_count:2,reason:'midpoint_probe'}],intervals:[]});
assert(html.includes('New-point budget reached'));
assert(html.includes('4 initial points + 6 added / 8 allowed'));
assert(!html.includes('Sampling criteria satisfied'));
assert(html.includes('midpoint probe'));
const local=renderGridRefinementSummary({policy:{max_new_points:20},strategy:'local_cells',status:'insufficient_evidence',
  seed_point_count:4,new_point_count:5,decisions:[{bounds:{U:[0,4],V:[0,2]},point_count:4,reason:'interpolation_error'}],
  intervals:[{bounds:{U:[0,2],V:[0,1]},status:'insufficient_evidence'}]});
assert(local.includes('U: 0 → 4; V: 0 → 2'));
assert(local.includes('Some regions lack reliable results'));
assert(local.includes('Unresolved regions'));
assert(!local.includes('undefined'));

'''
        result = subprocess.run(['node', '-e', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_running_grid_uses_live_counts_instead_of_previous_batch(self):
        script = DOM + (ROOT / 'computational_study_agent/web_assets/planner-rendering.js').read_text()
        script += r'''
function escapeHtml(s) {return String(s).replaceAll('<','&lt;').replaceAll('>','&gt;');}
const execution = {status:'running',agent:{running:true},expected:20,terminal:19,report_available:19,
  task_count:28,task_status_counts:{succeeded:16,failed:1,unconverged:2,pending:1,not_executed:8}};
const summary=executionStatusSummary(execution);
assert(summary.includes('28 planned points'));
assert(summary.includes('16 succeeded, 1 failed, 2 unconverged, 1 in progress, 8 awaiting execution'));
assert(!summary.includes('19/20'));
const plan={cases:Array(28).fill({}),grid_refinement:{enabled:true,max_new_points:64}};
const report={status:'completed_with_issues',comparison_table:Array(16).fill({}),
  grid_refinement:{seed_point_count:16,new_point_count:0}};
const original=JSON.stringify(report);
renderRunningGridProgress(execution,plan,report);
const html=control('report-summary').innerHTML;
assert(html.includes('16 initial points + 12 added / 64 allowed'));
assert(html.includes('last saved batch (16 points)'));
assert(!html.includes('completed_with_issues'));
assert.equal(JSON.stringify(report),original);
renderRunningGridProgress({...execution,agent:{running:false}},plan,report);
assert.equal(control('report-summary').innerHTML,html);
assert(executionStatusSummary({status:'running',expected:2,terminal:1}).includes('1/2 terminal'));
'''
        result = subprocess.run(['node', '-e', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
