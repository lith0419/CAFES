from __future__ import annotations

import copy
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from computational_study_agent.application import StudyApplicationService
from computational_study_agent.costing import estimate_case_cost
from pyscf_agent.providers.libdmet.dmet import validate_dmet_model_request
from pyscf_agent.providers.libdmet.availability import libdmet_availability
from tests.pyscf_agent.test_dmet_beta_web_ui import DOM
from tests.pyscf_agent.test_libdmet_dmet import _periodic_honeycomb_spec


ROOT = Path(__file__).resolve().parents[2]
OPTIONS = {'fragment_definition': 'honeycomb_hexagon'}


class HoneycombHexagonTests(unittest.TestCase):
    def assert_partition(self, spec, expected_count):
        before = copy.deepcopy(spec)
        errors, configuration = validate_dmet_model_request(spec, OPTIONS)
        self.assertEqual(errors, [])
        self.assertEqual(spec, before)
        self.assertEqual(configuration['execution_mode'], 'finite_graph')
        self.assertEqual(configuration['fragment_site_counts'], [6] * expected_count)
        seen = []
        edges = {frozenset((bond['source'], bond['target'])) for bond in spec['bonds']}
        by_id = {site['id']: site for site in spec['sites']}
        lengths = spec['primitive_cell']['repetitions']
        for fragment in configuration['fragments']:
            ring = fragment['site_ids']
            self.assertEqual(len(set(ring)), 6)
            self.assertEqual(sum(by_id[site]['basis_index'] == 0 for site in ring), 3)
            winding = [0, 0]
            for source, target in zip(ring, ring[1:] + ring[:1]):
                self.assertIn(frozenset((source, target)), edges)
                for axis, length in enumerate(lengths):
                    delta = (by_id[target]['cell_index'][axis] - by_id[source]['cell_index'][axis]) % length
                    winding[axis] += delta - length if delta > length / 2 else delta
            self.assertEqual(winding, [0, 0])
            seen.extend(ring)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), set(by_id))
        return configuration

    def test_periodic_sizes_form_complete_nonwinding_rings(self):
        for x, y, count in ((3, 3, 3), (3, 6, 6), (6, 6, 12)):
            with self.subTest(shape=(x, y)):
                self.assert_partition(_periodic_honeycomb_spec(x, y), count)

    def test_partition_does_not_depend_on_input_order_or_contiguous_ids(self):
        spec = _periodic_honeycomb_spec(6, 6)
        mapping = {site['id']: 1000 - site['id'] * 3 for site in spec['sites']}
        for site in spec['sites']:
            site['id'] = mapping[site['id']]
        for bond in spec['bonds']:
            bond['source'], bond['target'] = mapping[bond['source']], mapping[bond['target']]
        original = self.assert_partition(spec, 12)
        spec['sites'].reverse()
        spec['bonds'].reverse()
        reordered = self.assert_partition(spec, 12)
        self.assertEqual(original['fragments'], reordered['fragments'])

    def test_incompatible_size_and_periodic_seam_are_rejected(self):
        for shape, message in (((4, 4), '32 sites'), ((4, 6), 'cannot be tiled'), ((2, 3), 'at least three')):
            with self.subTest(shape=shape):
                errors, config = validate_dmet_model_request(_periodic_honeycomb_spec(*shape), OPTIONS)
                self.assertIsNone(config)
                self.assertTrue(any(message in error for error in errors), errors)

    def test_invalid_graph_and_conflicting_selectors_are_rejected(self):
        for update in ({'execution_mode': 'translational'}, {'impurity_shape': [3, 1]},
                       {'impurity_size': 6}, {'impurity_site_ids': [0, 1, 2, 3, 4, 5]},
                       {'fragments': [{'site_ids': [0, 1]}]}):
            with self.subTest(update=update):
                errors, config = validate_dmet_model_request(_periodic_honeycomb_spec(3, 3), {**OPTIONS, **update})
                self.assertTrue(errors)
                self.assertIsNone(config)
        for change in ('open', 'square', 'missing_bond', 'cell_metadata'):
            spec = _periodic_honeycomb_spec(3, 3)
            if change == 'open':
                spec['boundary'] = 'open'
            elif change == 'square':
                spec['preset'] = 'square'
            elif change == 'missing_bond':
                spec['bonds'].pop()
            else:
                for site in spec['sites']:
                    site.pop('cell_index')
                    site.pop('basis_index')
            with self.subTest(change=change):
                self.assertIsNone(validate_dmet_model_request(spec, OPTIONS)[1])

    def test_nonuniform_interactions_and_all_impurity_solvers_are_preserved(self):
        spec = _periodic_honeycomb_spec(3, 3)
        spec['sites'][0]['U'] = 3.25
        spec['bonds'][0].update(V=0.5, effective_V=0.5)
        for solver in ('fci', 'ccsd', 'block2_dmrg'):
            errors, config = validate_dmet_model_request(spec, {**OPTIONS, 'impurity_solver': solver})
            self.assertEqual(errors, [])
            self.assertTrue(config['contains_intersite_v'])
            self.assertEqual(config['fragment_site_counts'], [6, 6, 6])

    def test_saved_plan_round_trip_and_dmrg_embedding_cost(self):
        spec = _periodic_honeycomb_spec(6, 6)
        options = {**OPTIONS, 'impurity_solver': 'block2_dmrg'}
        study = {'name': 'hexagon', 'objective': 'Six-site honeycomb impurity',
                 'system_type': 'model_hamiltonian', 'study_mode': 'static',
                 'base_model_spec': spec, 'base_task': {'solver': {'name': 'dmet', 'options': options}},
                 'observables': ['energy']}
        with tempfile.TemporaryDirectory() as root:
            service = StudyApplicationService()
            prepared = service.prepare_study(study, work_dir=root)
            loaded = service.load_plan(prepared['study_id'], work_dir=root)
            case = loaded['cases'][0]
            self.assertEqual(case['request']['solver']['options']['fragment_definition'], 'honeycomb_hexagon')
            cost = estimate_case_cost(case, 'model_hamiltonian')
            self.assertEqual(cost['fragment_norb'], 6)
            self.assertEqual(cost['norb'], 12)

    @unittest.skipUnless(shutil.which('node'), 'node is not installed')
    def test_actual_builder_geometry_has_twelve_hexagons(self):
        script = """
const fs = require('fs'), vm = require('vm'), path = require('path');
const context = {console, Math, state: {representation: 'finite_cluster', dimension: 2,
  boundary: 'periodic', bondModulation: 'none', globals: {epsilon: 0, U: 4, V: 0, t: -1, delta: 0.2}}};
vm.createContext(context);
for (const file of ['hamiltonian.js', 'lattice_templates.js']) {
  vm.runInContext(fs.readFileSync(path.join(process.argv[1], file), 'utf8'), context);
}
process.stdout.write(JSON.stringify(context.createHoneycomb(6, 6)));
"""
        result = subprocess.run(['node', '-e', script, str(ROOT / 'model_hamiltonian_ui')],
                                check=True, capture_output=True, text=True, timeout=10)
        raw = json.loads(result.stdout)
        spec = _periodic_honeycomb_spec(6, 6)
        spec['primitive_cell'] = raw['primitiveCell']
        spec['sites'], spec['bonds'] = raw['sites'], raw['bonds']
        for site in spec['sites']:
            site['cell_index'] = site.pop('cellIndex')
            site['basis_index'] = site.pop('basisIndex')
        self.assert_partition(spec, 12)

    @unittest.skipUnless(shutil.which('node'), 'node is not installed')
    def test_both_forms_collect_and_restore_hexagon_without_stale_shape(self):
        for planner in (False, True):
            if planner:
                files = ['computational_study_agent/web_assets/planner-spec.js']
                setup = """
control('planner-model-solver').value = 'dmet';
control('planner-dmet-fragment-definition').value = 'honeycomb_hexagon';
control('planner-dmet-impurity-solver').value = 'fci';
control('planner-dmet-impurity-shape').value = '2,2';
const saved = plannerDmetOptionsFromControls();
syncPlannerModelSolverControls({base_task: {solver: {name: 'dmet', options: saved}}});
assert.equal(control('planner-dmet-fragment-definition').value, 'honeycomb_hexagon');
"""
            else:
                files = ['pyscf_agent/web_assets/assistant-structures.js', 'pyscf_agent/web_assets/assistant-forms.js']
                setup = """
control('model-solver').value = 'dmet';
control('model-dmet-fragment-mode').value = 'honeycomb_hexagon';
control('model-dmet-impurity-solver').value = 'fci';
control('model-dmet-impurity-shape').value = '2,2';
const saved = collectDmetSolverOptions();
setSelectValue = (id, value) => {control(id).value = value;};
syncTaskFamilyControls = () => {};
applyTaskSpecToForm({task_type: 'model_hamiltonian', solver: {name: 'dmet', options: saved}});
syncDmetFragmentControls();
assert.equal(control('model-dmet-fragment-mode').value, 'honeycomb_hexagon');
"""
            script = DOM.replace("_value: '',", "querySelector: () => null, _value: '',")
            script += '\n'.join((ROOT / path).read_text() for path in files) + setup
            script += """
assert.equal(saved.fragment_definition, 'honeycomb_hexagon');
assert.deepEqual(saved.impurity_shape, []);
assert.equal(saved.impurity_size, null);
"""
            with self.subTest(planner=planner):
                result = subprocess.run(['node', '-e', script], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(libdmet_availability()['available'], 'optional libDMET provider is unavailable')
class HoneycombHexagonExecutionTests(unittest.TestCase):
    def check_noninteracting_limit(self, size, solver, reference='restricted'):
        from pyscf_agent.backend.model_hamiltonian.solver import run_model_hamiltonian_solver

        spec = _periodic_honeycomb_spec(size, size)
        # A staggered onsite term opens a gap, avoiding the separate fractional
        # occupation bath approximation at the Dirac-point degeneracy.
        for site in spec['sites']:
            site.update(U=0.0, epsilon=0.2 if site['basis_index'] == 0 else -0.2)
        one_body = np.diag([site['epsilon'] for site in spec['sites']])
        for bond in spec['bonds']:
            i, j = bond['source'], bond['target']
            one_body[i, j] += bond['t']
            one_body[j, i] += bond['t']
        exact_energy = 2 * np.linalg.eigvalsh(one_body)[:len(one_body) // 2].sum()
        with tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_model_hamiltonian_solver(
                spec, solver_name='dmet', outputs=['energy'],
                solver_options={**OPTIONS, 'impurity_solver': solver, 'reference': reference,
                                'max_iterations': 2, 'solver_max_memory_mb': 2000},
                scratch_directory=scratch,
            )
        self.assertTrue(result['converged'])
        self.assertAlmostEqual(result['energy'], exact_energy, places=8)
        config = result['dmet_result']['configuration']
        self.assertEqual(config['fragment_site_counts'], [6] * (size * size // 3))
        self.assertEqual([fragment['site_ids'] for fragment in result['dmet_result']['fragments']],
                         [fragment['site_ids'] for fragment in config['fragments']])

    def test_six_by_six_ccsd_matches_gapped_noninteracting_limit(self):
        for reference in ('restricted', 'unrestricted'):
            with self.subTest(reference=reference):
                self.check_noninteracting_limit(6, 'ccsd', reference)

    def test_three_by_three_fci_matches_gapped_noninteracting_limit(self):
        self.check_noninteracting_limit(3, 'fci')
