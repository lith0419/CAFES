from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from computational_study_agent.application import StudyApplicationService
from pyscf_agent.application import CalculationApplicationService
from tests.computational_study_agent.support import hubbard_dimer_spec


ROOT = Path(__file__).resolve().parents[2]
DOM = """
const assert = require('node:assert/strict');
const controls = {};
function control(id) {
  if (!controls[id]) {
    const hidden = new Set();
    controls[id] = {checked: true, disabled: false, validity: {badInput: false},
      _value: '', get value() {return this._value;}, set value(v) {this._value = String(v);},
      options: ['fci', 'ccsd', 'dmet', 'block2_dmrg'].map(value => ({value})),
      classList: {toggle(name, yes) {yes ? hidden.add(name) : hidden.delete(name);},
                  contains(name) {return hidden.has(name);}}};
  }
  return controls[id];
}
const document = {getElementById: control};
function deepCopy(v) {return JSON.parse(JSON.stringify(v));}
function normalizeSystemType(v) {return v;}
function integerInputValue(id, fallback) {return Number(control(id).value) || fallback;}
function numericInputValue(id, fallback) {return Number(control(id).value) || fallback;}
"""


@unittest.skipUnless(shutil.which('node'), 'node is not installed')
class DmetBetaWebUiTests(unittest.TestCase):
    def ui_options(self, planner):
        if planner:
            paths = ['computational_study_agent/web_assets/planner-spec.js']
            prefix, collect, sync = 'planner', 'plannerDmetOptionsFromControls', 'syncPlannerDmetConditionalControls'
            restore = """
syncPlannerModelSolverControls({base_task: {solver: {name: 'dmet', options: saved}}});
assert.equal(control(betaId).value, '1000');
assert.equal(control(bathId).value, 'false');
syncPlannerModelSolverControls({base_task: {solver: 'dmet'}});
assert.equal(control(betaId).value, '');
assert.equal(control(bathId).value, 'true');
"""
        else:
            paths = ['pyscf_agent/web_assets/assistant-structures.js',
                     'pyscf_agent/web_assets/assistant-forms.js']
            prefix, collect, sync = 'model', 'collectDmetSolverOptions', 'syncDmetImpuritySolverControls'
            restore = """
setSelectValue = (id, value) => {control(id).value = value;};
syncTaskFamilyControls = () => {};
syncDmetFragmentControls = () => {};
applyTaskSpecToForm({task_type: 'model_hamiltonian', solver: {name: 'dmet', options: saved}});
assert.equal(control(betaId).value, '1000');
assert.equal(control(bathId).value, 'false');
applyTaskSpecToForm({task_type: 'model_hamiltonian', solver: {name: 'dmet', options: {}}});
assert.equal(control(betaId).value, '');
assert.equal(control(bathId).value, 'true');
"""
        script = DOM + '\n'.join((ROOT / path).read_text() for path in paths)
        script += f"""
const prefix = {json.dumps(prefix)};
const solverId = prefix === 'planner' ? 'planner-model-solver' : 'model-solver';
const betaId = prefix + '-dmet-ccsd-beta';
const bathId = prefix + '-dmet-interacting-bath';
const impurityId = prefix + '-dmet-impurity-solver';
const collect = {collect}, sync = {sync};
control(solverId).value = 'dmet';
control(impurityId).value = 'ccsd';
sync();
assert.equal(control(betaId).disabled, false);
assert.equal(control(betaId + '-control').classList.contains('hidden'), false);
assert.deepEqual(collect().impurity_solver_options, {{}});
control(betaId).value = '1e3';
control(bathId).value = 'true';
assert.equal(collect().interacting_bath, true);
control(bathId).value = 'false';
const saved = collect();
assert.equal(saved.interacting_bath, false);
assert.deepEqual(saved.impurity_solver_options, {{beta: 1000}});
for (const other of ['fci', 'block2_dmrg']) {{
  control(impurityId).value = other; sync();
  assert.equal(control(betaId).disabled, true);
  assert.equal(control(betaId + '-control').classList.contains('hidden'), true);
  assert.equal(Object.hasOwn(collect().impurity_solver_options, 'beta'), false);
}}
control(impurityId).value = 'ccsd'; sync();
assert.equal(collect().impurity_solver_options.beta, 1000);
control(betaId).value = '';
assert.deepEqual(collect().impurity_solver_options, {{}});
control(betaId).validity.badInput = true;
assert.deepEqual(collect().impurity_solver_options, {{beta: null}});
control(betaId).validity.badInput = false;
control(betaId).value = '0';
assert.equal(collect().impurity_solver_options.beta, 0);
control(solverId).value = 'fci'; sync();
assert.equal(control(betaId).disabled, true);
{restore}
console.log(JSON.stringify(saved));
"""
        result = subprocess.run(['node', '-e', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_assistant_beta_round_trip_reaches_task_validation(self):
        options = self.ui_options(planner=False)
        task = {'task_type': 'model_hamiltonian',
                'model_hamiltonian': {'spec': hubbard_dimer_spec()},
                'solver': {'name': 'dmet', 'options': options}}
        service = CalculationApplicationService()
        result = service.validate_task_spec(task)
        self.assertTrue(result['valid'], result['errors'])
        self.assertEqual(result['task_spec']['solver']['options']['impurity_solver_options'], {'beta': 1000.0})
        for beta in (0, None):
            task['solver']['options']['impurity_solver_options']['beta'] = beta
            self.assertFalse(service.validate_task_spec(task)['valid'])

    def test_planner_beta_round_trip_survives_saved_study_preparation(self):
        options = self.ui_options(planner=True)
        spec = {'name': 'beta-ui', 'objective': 'Verify beta input',
                'system_type': 'model_hamiltonian', 'study_mode': 'static',
                'base_model_spec': hubbard_dimer_spec(),
                'base_task': {'solver': {'name': 'dmet', 'options': options}},
                'observables': ['energy']}
        with tempfile.TemporaryDirectory() as directory:
            service = StudyApplicationService()
            prepared = service.prepare_study(spec, work_dir=directory)
            loaded = service.load_plan(prepared['study_id'], work_dir=directory)
            self.assertEqual(loaded['cases'][0]['request']['solver']['options']['impurity_solver_options'],
                             {'beta': 1000})
