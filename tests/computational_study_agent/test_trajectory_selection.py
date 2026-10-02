import copy
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from computational_study_agent.application import StudyApplicationService
from computational_study_agent.datasets.hamiltonian.contracts import (
    HAMILTONIAN_MANIFEST_SCHEMA, HAMILTONIAN_SAMPLE_SCHEMA,
)
from computational_study_agent.trajectory_selection import (
    ATOMIC_TIME_FS, ENERGY_SELECTION_ACTION, build_energy_stratified_selection,
)
from pyscf_agent.registry import default_registry


class TrajectorySelectionTests(unittest.TestCase):
    def fixture(self, root, energies, molecules=('a',)):
        dataset = root / 'dataset'
        dataset.mkdir()
        rows = []
        for molecule in molecules:
            for frame, energy in enumerate(energies):
                geometry_id = 'frame-{0:06d}'.format(frame)
                rows.append({
                    'schema': HAMILTONIAN_SAMPLE_SCHEMA,
                    'sample_id': molecule + ':' + geometry_id,
                    'molecule_id': molecule, 'geometry_id': geometry_id,
                    'split': 'train', 'converged': True,
                    'geometry': {'molecule_id': molecule, 'geometry_id': geometry_id,
                                 'atomic_numbers': [1, 1], 'positions': [[0., 0., 0.], [0., 0., .7 + frame * .01]],
                                 'coordinate_unit': 'Angstrom', 'charge': 0, 'spin': 0},
                    'total_energy_hartree': energy,
                    'electronic_structure': {'method': 'dft', 'xc': 'b3lyp', 'basis': 'def2-svp'},
                    'provenance': {'study_id': 'study', 'case_id': molecule, 'run_id': molecule,
                                   'frame_index': frame, 'array_index': frame},
                    'fock': {'path': '/unavailable/remote/arrays.npz'},
                    'overlap': {'path': '/unavailable/remote/arrays.npz'},
                })
        path = dataset / 'samples.jsonl'
        self.write_rows(path, rows)
        manifest = {'schema': HAMILTONIAN_MANIFEST_SCHEMA,
                    'spec': {'target_molecule_count': len(molecules),
                             'molecular_dynamics': {'time_step_au': 1 / ATOMIC_TIME_FS}},
                    'accepted_structure_count': len(rows), 'rejected_structure_count': 0,
                    'artifacts': {'sample_index': str(path),
                                  'dataset_manifest': str(dataset / 'dataset-manifest.json')}}
        (dataset / 'dataset-manifest.json').write_text(json.dumps(manifest))
        report = {'study_id': 'study', 'system_type': 'molecular', 'work_dir': str(root),
                  'dataset_manifest': manifest, 'cases': [], 'comparison_table': [], 'artifacts': []}
        return path, rows, manifest, report

    @staticmethod
    def write_rows(path, rows):
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def test_quantile_bins_use_population_and_actual_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _, manifest, _ = self.fixture(Path(tmp), [100, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            result = build_energy_stratified_selection(path, manifest, {'count_per_molecule': 3})
        selected = result['selected_frames']
        self.assertEqual([r['potential_energy_hartree'] for r in selected], [1, 5, 9])
        self.assertEqual([r['selection']['bin_size'] for r in selected], [4, 4, 3])
        self.assertEqual([r['frame_index'] for r in selected], [2, 6, 10])
        self.assertEqual(result['status'], 'complete')

    def test_shuffled_ties_remain_deterministic_and_molecules_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, rows, manifest, _ = self.fixture(Path(tmp), [0] * 8, ('a', 'b'))
            options = {'count_per_molecule': 2}
            first = build_energy_stratified_selection(path, manifest, options)
            self.write_rows(path, list(reversed(rows)))
            second = build_energy_stratified_selection(path, manifest, options)
        self.assertEqual(first, second)
        self.assertEqual([r['frame_index'] for r in first['selected_frames']], [1, 5, 1, 5])

    def test_inclusive_window_uses_frame_index_not_array_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, rows, manifest, _ = self.fixture(Path(tmp), list(range(5)))
            for index, row in enumerate(rows):
                row['provenance']['frame_index'] = 199 + index
            self.write_rows(path, rows)
            result = build_energy_stratified_selection(path, manifest, {
                'count_per_molecule': 2, 'time_start_fs': 200, 'time_end_fs': 202})
        self.assertEqual([r['frame_index'] for r in result['selected_frames']], [200, 202])
        self.assertEqual(result['outside_time_window_count'], 2)

    def test_insufficient_candidates_and_failed_source_molecules_are_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _, manifest, _ = self.fixture(Path(tmp), [1, 2, 3])
            manifest['spec']['target_molecule_count'] = 2
            manifest['rejected_structure_count'] = 3
            partial = build_energy_stratified_selection(path, manifest, {'count_per_molecule': 2})
            empty = build_energy_stratified_selection(path, manifest, {'count_per_molecule': 4})
        self.assertEqual(partial['status'], 'partial')
        self.assertEqual(partial['unselected_molecule_count'], 1)
        self.assertEqual(empty['selected_frame_count'], 0)
        self.assertEqual(empty['molecules'][0]['status'], 'insufficient_candidates')

    def test_action_exports_geometry_and_provenance_without_loading_matrices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _, manifest, report = self.fixture(root, list(range(10)), ('a', 'b'))
            original = path.read_bytes()
            original_report = copy.deepcopy(report)
            with patch('numpy.load', side_effect=AssertionError('No trajectory arrays needed')):
                result = StudyApplicationService().run_postprocessing(report, actions=[{
                    'action': ENERGY_SELECTION_ACTION, 'count_per_molecule': 2,
                    'time_start_fs': 2, 'time_end_fs': 9}])
            folder = root / 'postprocessing' / 'energy-frame-selection'
            rows = [json.loads(line) for line in (folder / 'selected-frames.jsonl').read_text().splitlines()]
            saved = json.loads((folder / 'selection-manifest.json').read_text())
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(report, original_report)
            self.assertEqual(result['status'], 'succeeded')
            self.assertEqual(result['actions'][0]['sample_count'], 4)
            self.assertEqual(saved['selected_frame_count'], 4)
            self.assertEqual(len((folder / 'selected-frames.xyz').read_text().splitlines()), 16)
            self.assertEqual(rows[0]['split'], 'train')
            self.assertIn('run_id', rows[0]['provenance'])
            self.assertNotIn('fock', rows[0])
            for artifact in result['artifacts']:
                self.assertGreater(artifact['size_bytes'], 0)

    def test_invalid_options_or_source_do_not_create_selection_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, rows, _, report = self.fixture(root, [1, 2, 3])
            service = StudyApplicationService()
            for options in ({}, {'count_per_molecule': True}, {'count_per_molecule': 1.5},
                            {'count_per_molecule': 0}, {'count_per_molecule': 2, 'typo': 1},
                            {'count_per_molecule': 2, 'time_start_fs': 3, 'time_end_fs': 2}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    service.run_postprocessing(report, actions=[{'action': ENERGY_SELECTION_ACTION, **options}])
            for mutation in ('duplicate', 'nan', 'unconverged'):
                bad = copy.deepcopy(rows)
                if mutation == 'duplicate': bad.append(bad[0])
                if mutation == 'nan': bad[0]['total_energy_hartree'] = float('nan')
                if mutation == 'unconverged': bad[0]['converged'] = False
                self.write_rows(path, bad)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    service.run_postprocessing(report, actions=[{'action': ENERGY_SELECTION_ACTION, 'count_per_molecule': 2}])
            self.assertFalse((root / 'postprocessing').exists())

    def test_registry_exposes_action_and_artifact_contracts(self):
        registry = default_registry()
        action = registry.capability(ENERGY_SELECTION_ACTION, namespace='postprocessing.action')
        self.assertTrue(action.backend_allowed)
        self.assertTrue(action.planner_allowed)
        self.assertTrue(action.metadata['parameters']['count_per_molecule']['required'])
        kinds = {kind for contract in registry.artifact_contracts() for kind in contract.artifact_kinds}
        self.assertTrue(set(action.metadata['artifact_kinds']) <= kinds)

    def test_http_action_preserves_parameters_and_reports_partial_selection(self):
        from computational_study_agent.web_api import handle_study_postprocess_request

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, manifest, report = self.fixture(root, list(range(10)))
            manifest['spec']['target_molecule_count'] = 2
            (root / 'dataset/dataset-manifest.json').write_text(json.dumps(manifest))
            with patch('computational_study_agent.web_api.get_study_application_service',
                       return_value=StudyApplicationService()):
                status, _, body = handle_study_postprocess_request(json.dumps({
                    'report': report, 'actions': [{'action': ENERGY_SELECTION_ACTION,
                    'count_per_molecule': 2, 'time_start_fs': 3, 'time_end_fs': 9}],
                }).encode())
            result = json.loads(body)['postprocessing']
        self.assertEqual(status, 200)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['actions'][0]['parameters']['time_start_fs'], 3)
        self.assertEqual(result['actions'][0]['unselected_molecule_count'], 1)

    @unittest.skipUnless(shutil.which('node'), 'Node is required to verify browser action')
    def test_browser_sends_compact_report_and_selection_parameters(self):
        source = Path(__file__).parents[2] / 'computational_study_agent/web_assets/planner-postprocessing.js'
        script = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const nodes = {}, requests = [], rendered = [], messages = [];
const ctx = {
  currentReport: {study_id:'s', work_dir:'/study', system_type:'molecular',
    dataset_manifest:{accepted_structure_count:1001}, cases:[{large:'Do not send'}]},
  currentExecution:null,
  document:{getElementById:id => nodes[id] || (nodes[id]={value:'',disabled:false,
    reportValidity:() => true, classList:{toggle:() => {}}})},
  deepCopy:x => JSON.parse(JSON.stringify(x)), selectedExecutionTarget:() => 'local',
  setStatus:(...args) => messages.push(args), snapshotCurrentTaskSession:() => {},
  errorDetailMessage:(_, message) => message,
  postJson:async (url, body) => {requests.push({url,body}); return {postprocessing:{
    status:'succeeded', artifacts:[{kind:'selected_trajectory_xyz',path:'/out.xyz'}],
    actions:[{sample_count:20,molecule_count:1,unselected_molecule_count:0}]}};},
};
vm.createContext(ctx); vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),ctx);
ctx.renderPostprocessArtifacts = x => rendered.push(x);
ctx.document.getElementById('energy-selection-count').value='20';
ctx.document.getElementById('energy-selection-start').value='200';
ctx.document.getElementById('energy-selection-end').value='1000';
(async () => {
  await ctx.selectEnergyFrames();
  assert.equal(requests[0].url,'/api/study-postprocess');
  assert.deepEqual(requests[0].body.actions,[{action:'select_energy_stratified_frames',
    count_per_molecule:20,time_start_fs:200,time_end_fs:1000}]);
  assert.equal(requests[0].body.report.cases,undefined);
  assert.equal(ctx.currentReport.frame_selection.sample_count,20);
  assert.equal(rendered.length,1);
  assert.equal(nodes['select-energy-frames'].disabled,false);
  nodes['energy-selection-count'].reportValidity=() => false;
  await ctx.selectEnergyFrames(); assert.equal(requests.length,1);
})().catch(error => {console.error(error);process.exit(1);});
'''
        result = subprocess.run(['node', '-e', script, str(source)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
