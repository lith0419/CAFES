from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import pyscf_agent.web.server as web


class AgentWebServerTests(unittest.TestCase):
    def test_response_writer_ignores_client_disconnect(self):
        class DisconnectedStream:
            def write(self, body):
                raise BrokenPipeError(32, 'Broken pipe')

        handler = object.__new__(web.AgentWebHandler)
        handler.send_response = lambda status: None
        handler.send_header = lambda key, value: None
        handler.end_headers = lambda: None
        handler.wfile = DisconnectedStream()
        handler.close_connection = False

        handler._write_response(
            web.HTTPStatus.OK,
            {'Content-Type': 'application/json'},
            b'{}',
        )

        self.assertTrue(handler.close_connection)

    def test_response_writer_preserves_unexpected_io_errors(self):
        class FailingStream:
            def write(self, body):
                raise OSError(5, 'I/O error')

        handler = object.__new__(web.AgentWebHandler)
        handler.send_response = lambda status: None
        handler.send_header = lambda key, value: None
        handler.end_headers = lambda: None
        handler.wfile = FailingStream()

        with self.assertRaises(OSError):
            handler._write_response(
                web.HTTPStatus.OK,
                {'Content-Type': 'application/json'},
                b'{}',
            )

    def test_web_server_registers_computational_study_routes(self):
        source = Path(web.__file__).read_text(encoding='utf-8')

        self.assertIn('/computational-study/', source)
        self.assertIn('/api/study-plan', source)
        self.assertIn('/api/study-run', source)
        self.assertIn('/api/study-adaptive-plan', source)
        self.assertIn('/api/study-adaptive-run', source)
        self.assertIn('/api/study-execution-status', source)
        self.assertIn('/api/study-execution-collect', source)
        self.assertIn('/api/study-llm-draft', source)
        self.assertIn('/api/study-result-analysis', source)
        self.assertIn('/api/study-postprocess', source)
        self.assertIn('/api/study-postprocess-suggestions', source)
        self.assertIn('/api/study-artifact', source)
        self.assertIn('/api/capabilities', source)
        self.assertIn('/api/execution-targets', source)
        self.assertIn('/api/resource-profiles', source)
        self.assertIn('/api/periodic-structure-preview', source)

    def test_builder_static_assets_are_loaded_from_package_resources(self):
        source = Path(web.__file__).read_text(encoding='utf-8')

        self.assertIn("MODEL_HAMILTONIAN_UI_PACKAGE = 'model_hamiltonian_ui'", source)
        self.assertIn('importlib_resources.files(MODEL_HAMILTONIAN_UI_PACKAGE)', source)
        self.assertNotIn("PROJECT_ROOT / 'model_hamiltonian_ui'", source)

        index_asset = web._read_model_hamiltonian_builder_asset('index.html')
        self.assertIsNotNone(index_asset)
        asset_path, body = index_asset
        self.assertEqual(asset_path, 'index.html')
        self.assertIn(b'id="return-agent-ui"', body)

        example_asset = web._read_model_hamiltonian_builder_asset('examples/hubbard_chain.json')
        self.assertIsNotNone(example_asset)
        self.assertEqual(example_asset[0], 'examples/hubbard_chain.json')
        self.assertIn(b'"model"', example_asset[1])

        self.assertIsNone(web._read_model_hamiltonian_builder_asset('../setup.py'))
        self.assertIsNone(web._read_model_hamiltonian_builder_asset('/index.html'))

    def test_agent_and_planner_assets_are_loaded_from_package_resources(self):
        assistant_asset = web._read_agent_web_asset('assistant-core.js')
        self.assertIsNotNone(assistant_asset)
        self.assertEqual(assistant_asset[0], 'assistant-core.js')
        self.assertIn(b'function escapeHtml', assistant_asset[1])

        execution_target_asset = web._read_agent_web_asset('execution-targets.js')
        self.assertIsNotNone(execution_target_asset)
        self.assertIn(b'initializeExecutionTargetSelect', execution_target_asset[1])

        planner_asset = web._read_study_web_asset('planner-core.js')
        self.assertIsNotNone(planner_asset)
        self.assertEqual(planner_asset[0], 'planner-core.js')
        self.assertIn(b'function escapeHtml', planner_asset[1])

        self.assertIsNone(web._read_agent_web_asset('../setup.py'))
        self.assertIsNone(web._read_study_web_asset('/planner-core.js'))

    def test_packaging_config_includes_builder_static_package(self):
        repo_root = Path(__file__).resolve().parents[2]
        pyproject_text = (repo_root / 'pyproject.toml').read_text(encoding='utf-8')

        self.assertIn('[tool.setuptools.packages.find]', pyproject_text)
        self.assertIn('"model_hamiltonian_ui"', pyproject_text)
        self.assertIn('[tool.setuptools.package-data]', pyproject_text)
        self.assertIn('model_hamiltonian_ui = [', pyproject_text)
        self.assertIn('"pyscf_agent.web_assets" = [', pyproject_text)
        self.assertIn('"computational_study_agent.web_assets" = [', pyproject_text)
        self.assertIn('"ase>=3.22,<3.30"', pyproject_text)
        self.assertIn('"scipy>=1.11,<1.19"', pyproject_text)
        self.assertIn('"*.html"', pyproject_text)
        self.assertIn('"*.css"', pyproject_text)
        self.assertIn('"*.js"', pyproject_text)
        self.assertIn('"examples/*.json"', pyproject_text)

    def test_builder_python_export_is_structure_only(self):
        exporter_text = (Path(__file__).resolve().parents[2] / 'model_hamiltonian_ui' / 'exporters.js').read_text(encoding='utf-8')

        python_export = exporter_text.split('function pythonInputText()', 1)[1]
        self.assertIn('This file intentionally contains only the model Hamiltonian structure', python_export)
        self.assertIn('model_spec = ${JSON.stringify(spec, null, 2)}', python_export)
        self.assertIn('print("energy_unit =", model_spec.get("energy_unit"))', python_export)
        self.assertIn('print("graph_edges =", len(model_spec.get("graph", {}).get("edges", [])))', python_export)
        self.assertNotIn('from pyscf import fci', python_export)
        self.assertNotIn('direct_spin1.kernel', python_export)

    def test_builder_can_return_to_agent_or_study_page(self):
        builder_dir = Path(__file__).resolve().parents[2] / 'model_hamiltonian_ui'
        index_text = (builder_dir / 'index.html').read_text(encoding='utf-8')
        controller_text = (builder_dir / 'ui_controller.js').read_text(encoding='utf-8')

        self.assertIn('id="return-agent-ui"', index_text)
        self.assertIn('id="return-study-ui"', index_text)
        self.assertIn('Back To Planner', index_text)
        self.assertIn("target: params.get('target') || 'agent'", controller_text)
        self.assertIn("target === 'study' ? 'pyscf-computational-study-agent' : 'pyscf-agent-web-ui'", controller_text)
        self.assertIn("target === 'study' ? '/computational-study/' : '/'", controller_text)
        self.assertIn("target: payload.target || builderWorkContext().target || ''", controller_text)
        self.assertIn("returnToTargetWebUi('agent')", controller_text)
        self.assertIn("returnToTargetWebUi('study')", controller_text)
        self.assertIn('function openTargetWindow(target)', controller_text)

    def test_builder_labels_energy_parameters_with_selected_unit(self):
        builder_dir = Path(__file__).resolve().parents[2] / 'model_hamiltonian_ui'
        index_text = (builder_dir / 'index.html').read_text(encoding='utf-8')
        controller_text = (builder_dir / 'ui_controller.js').read_text(encoding='utf-8')
        rendering_text = (builder_dir / 'rendering.js').read_text(encoding='utf-8')
        exporter_text = (builder_dir / 'exporters.js').read_text(encoding='utf-8')

        self.assertIn('class="unit-chip energy-unit-label">a.u.</span>', index_text)
        self.assertIn('function updateEnergyUnitLabels()', controller_text)
        self.assertIn("document.querySelectorAll('.energy-unit-label')", controller_text)
        self.assertIn('formatEnergyValue(effectiveBondT(bond))', rendering_text)
        self.assertIn('h1e (${energyUnitText()})', rendering_text)
        self.assertIn('global U: ${formatNumber(spec.globals.U)} ${unit}', exporter_text)

    def test_builder_2d_cluster_allows_single_cell_and_renders_its_boundary(self):
        builder_dir = Path(__file__).resolve().parents[2] / 'model_hamiltonian_ui'
        index_text = (builder_dir / 'index.html').read_text(encoding='utf-8')
        state_text = (builder_dir / 'state.js').read_text(encoding='utf-8')
        controller_text = (builder_dir / 'ui_controller.js').read_text(encoding='utf-8')
        lattice_text = (builder_dir / 'lattice_templates.js').read_text(encoding='utf-8')
        rendering_text = (builder_dir / 'rendering.js').read_text(encoding='utf-8')
        styles_text = (builder_dir / 'styles.css').read_text(encoding='utf-8')

        self.assertIn('id="size-x" type="number" min="1"', index_text)
        self.assertIn('<option value="2" selected>Two-dimensional</option>', index_text)
        self.assertIn('id="size-x" type="number" min="1" max="64" step="1" value="2"', index_text)
        self.assertIn('id="size-y" type="number" min="1" max="16" step="1" value="2"', index_text)
        self.assertIn('dimension: 2,', state_text)
        self.assertIn("preset: 'square'", state_text)
        self.assertIn('cellSubdivisions: { x: 1, y: 1 }', state_text)
        self.assertIn('const minimumSizeX = !isBloch && !isOneDimensional ? 1 : 2;', controller_text)
        self.assertIn('const minimumSizeY = !isOneDimensional ? 1 : 2;', controller_text)
        self.assertIn('state.dimension === 2 ? 1 : 2', lattice_text)
        self.assertIn('sizeY = clampInt(sizeY, 1, 16);', lattice_text)
        self.assertIn('? { x: sizeX, y: sizeY }', lattice_text)
        self.assertIn('if (state.dimension !== 2 || !state.cell)', rendering_text)
        self.assertIn('function cellGridSegments(', rendering_text)
        self.assertIn("class: 'cell-grid-line'", rendering_text)
        self.assertIn("'axis-line axis-line-subtle'", rendering_text)
        self.assertIn('.axis-line-subtle', styles_text)
        self.assertIn('.cell-boundary.open-cell', styles_text)
        self.assertIn('.cell-grid-line', styles_text)
        self.assertIn('stroke-dasharray: none;', styles_text.split('.axis-line {', 1)[1].split('}', 1)[0])
        self.assertIn('stroke-dasharray: 6 5;', styles_text.split('.cell-grid-line {', 1)[1].split('}', 1)[0])

    @unittest.skipUnless(shutil.which('node'), 'node is not installed')
    def test_builder_cell_grid_segments_subdivide_orthogonal_and_skew_cells(self):
        builder_dir = Path(__file__).resolve().parents[2] / 'model_hamiltonian_ui'
        script = r"""
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const builderDir = process.argv[1];
const context = { console, Math };
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(builderDir, 'rendering.js'), 'utf8'), context);

process.stdout.write(JSON.stringify({
  orthogonal: context.cellGridSegments(
    { x: 0, y: 0 },
    { x: 2, y: 0 },
    { x: 0, y: 2 },
    { x: 2, y: 2 },
  ),
  skew: context.cellGridSegments(
    { x: 0, y: 0 },
    { x: 2, y: 0 },
    { x: 1, y: 2 },
    { x: 2, y: 2 },
  ),
  single: context.cellGridSegments(
    { x: 0, y: 0 },
    { x: 1, y: 0 },
    { x: 0, y: 1 },
    { x: 1, y: 1 },
  ),
}));
"""

        result = subprocess.run(
            [shutil.which('node'), '-e', script, str(builder_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        segments = json.loads(result.stdout)

        self.assertEqual(
            segments['orthogonal'],
            [
                {'start': {'x': 1, 'y': 0}, 'end': {'x': 1, 'y': 2}},
                {'start': {'x': 0, 'y': 1}, 'end': {'x': 2, 'y': 1}},
            ],
        )
        self.assertEqual(
            segments['skew'],
            [
                {'start': {'x': 1, 'y': 0}, 'end': {'x': 2, 'y': 2}},
                {'start': {'x': 0.5, 'y': 1}, 'end': {'x': 2.5, 'y': 1}},
            ],
        )
        self.assertEqual(segments['single'], [])

    def test_builder_export_includes_site_bond_graph_view(self):
        exporter_text = (Path(__file__).resolve().parents[2] / 'model_hamiltonian_ui' / 'exporters.js').read_text(encoding='utf-8')

        self.assertIn('function serializableGraph()', exporter_text)
        self.assertIn("representation: 'site-bond graph'", exporter_text)
        self.assertIn('endpoints: [bond.source, bond.target]', exporter_text)
        self.assertIn('graph: serializableGraph()', exporter_text)

    def test_builder_keeps_bloch_contract_but_hides_the_workflow(self):
        builder_dir = Path(__file__).resolve().parents[2] / 'model_hamiltonian_ui'
        index_text = (builder_dir / 'index.html').read_text(encoding='utf-8')
        exporter_text = (builder_dir / 'exporters.js').read_text(encoding='utf-8')
        controller_text = (builder_dir / 'ui_controller.js').read_text(encoding='utf-8')

        for element_id in (
            'representation-finite',
            'representation-bloch',
            'electrons-per-cell',
            'kmesh-x',
            'kmesh-y',
            'kpoint-scheme',
            'path-points',
            'dos-auto',
            'bond-cell-offset-x',
            'bond-add-hc',
            'spin-multiplicity',
        ):
            self.assertIn('id="{0}"'.format(element_id), index_text)
        self.assertIn('id="representation-control" class="segmented-control" role="group" aria-label="Hamiltonian representation" hidden', index_text)
        self.assertIn('id="bloch-reciprocal-section" class="hidden" hidden', index_text)
        self.assertIn('const BLOCH_UI_ENABLED = false;', (builder_dir / 'state.js').read_text(encoding='utf-8'))
        self.assertIn("state.representation = 'finite_cluster';", controller_text)
        self.assertIn("if (!BLOCH_UI_ENABLED && button.dataset.representation === 'bloch')", controller_text)
        self.assertIn("representation: state.representation", exporter_text)
        self.assertIn("solver: isBlochRepresentation() ? 'tight_binding' : 'fci'", exporter_text)
        self.assertIn('spec.lattice_vectors =', exporter_text)
        self.assertIn('cell_offset:', exporter_text)
        self.assertIn('reciprocal_space', exporter_text)
        self.assertIn("solver: state.representation === 'bloch' ? 'tight_binding' : ''", controller_text)
        self.assertIn("reciprocalSection.hidden = !isBloch;", controller_text)
        self.assertIn("reciprocalSection.querySelectorAll('input, select, button')", controller_text)
        self.assertIn('updateTemplateControls();\n  updateEnergyUnitLabels();', controller_text)
        self.assertIn('state.js?v=20260819a', index_text)
        self.assertIn('ui_controller.js?v=20260819a', index_text)
        self.assertIn('[hidden] {\n  display: none !important;', (builder_dir / 'styles.css').read_text(encoding='utf-8'))
        self.assertIn('.panel section.hidden', (builder_dir / 'styles.css').read_text(encoding='utf-8'))
        self.assertIn('styles.css?v=20260824a', index_text)

    @unittest.skipUnless(shutil.which('node'), 'node is not installed')
    def test_builder_finite_templates_preserve_primitive_cell_membership(self):
        builder_dir = Path(__file__).resolve().parents[2] / 'model_hamiltonian_ui'
        script = r"""
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const builderDir = process.argv[1];
const context = {
  console,
  Math,
  state: {
    representation: 'finite_cluster',
    globals: { epsilon: 0, U: 4, V: 0, t: -1, delta: 0.2 },
    dimension: 2,
    boundary: 'periodic',
    bondModulation: 'none',
  },
};
vm.createContext(context);
for (const file of ['hamiltonian.js', 'lattice_templates.js']) {
  vm.runInContext(fs.readFileSync(path.join(builderDir, file), 'utf8'), context);
}

function summarize(lattice) {
  const cells = new Set(lattice.sites.map((site) => site.cellIndex.join(',')));
  return {
    siteCount: lattice.sites.length,
    cellCount: cells.size,
    basisSize: lattice.primitiveCell.basis_size,
    basisIndices: Array.from(new Set(lattice.sites.map((site) => site.basisIndex))).sort(),
    sublattices: Array.from(new Set(lattice.sites.map((site) => site.sublattice))).sort(),
  };
}

process.stdout.write(JSON.stringify({
  triangular: summarize(context.createTriangular(2, 2)),
  honeycomb: summarize(context.createHoneycomb(2, 2)),
  kagome: summarize(context.createKagome(2, 2)),
}));
"""
        result = subprocess.run(
            [shutil.which('node'), '-e', script, str(builder_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload['triangular']['basisSize'], 1)
        self.assertEqual(payload['honeycomb']['cellCount'], 4)
        self.assertEqual(payload['honeycomb']['basisIndices'], [0, 1])
        self.assertEqual(payload['kagome']['siteCount'], 12)
        self.assertEqual(payload['kagome']['basisIndices'], [0, 1, 2])
        self.assertEqual(payload['kagome']['sublattices'], ['a', 'b', 'c'])

    @unittest.skipUnless(shutil.which('node'), 'node is not installed')
    def test_builder_bloch_templates_use_primitive_cells_and_cell_offsets(self):
        builder_dir = Path(__file__).resolve().parents[2] / 'model_hamiltonian_ui'
        script = r"""
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const builderDir = process.argv[1];
const context = {
  console,
  Math,
  state: {
    representation: 'bloch',
    globals: { epsilon: 0, U: 4, V: 0, t: -1, delta: 0.2 },
    dimension: 1,
    boundary: 'periodic',
    bondModulation: 'none',
  },
};
vm.createContext(context);
for (const file of ['hamiltonian.js', 'lattice_templates.js']) {
  vm.runInContext(fs.readFileSync(path.join(builderDir, file), 'utf8'), context);
}

function summarize(preset, dimension, width = 2) {
  context.state.dimension = dimension;
  const lattice = context.createBlochPrimitive(preset, width);
  return {
    sites: lattice.sites.length,
    bonds: lattice.bonds.length,
    periodic: lattice.bonds.filter((bond) => bond.periodic).length,
    offsetDimensions: Array.from(new Set(lattice.bonds.map((bond) => bond.cellOffset.length))),
    hermitian: lattice.bonds.every((bond) => bond.addHermitianConjugate),
  };
}

const payload = {
  chain: summarize('chain', 1),
  quasi: summarize('quasi_1d', 1, 3),
  square: summarize('square', 2),
  lieb: summarize('lieb', 2),
  honeycomb: summarize('honeycomb', 2),
  triangular: summarize('triangular', 2),
  kagome: summarize('kagome', 2),
  dice: summarize('dice', 2),
  squareOctagon: summarize('square_octagon', 2),
};
context.state.dimension = 1;
context.state.bondModulation = 'ssh';
const ssh = context.createBlochPrimitive('chain', 2);
payload.ssh = {
  sites: ssh.sites.length,
  bonds: ssh.bonds.length,
  kinds: ssh.bonds.map((bond) => bond.kind).sort(),
};
process.stdout.write(JSON.stringify(payload));
"""

        result = subprocess.run(
            [shutil.which('node'), '-e', script, str(builder_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        templates = json.loads(result.stdout)

        self.assertEqual(templates['chain'], {'sites': 1, 'bonds': 1, 'periodic': 1, 'offsetDimensions': [1], 'hermitian': True})
        self.assertEqual(templates['quasi'], {'sites': 3, 'bonds': 5, 'periodic': 3, 'offsetDimensions': [1], 'hermitian': True})
        self.assertEqual(templates['square'], {'sites': 1, 'bonds': 2, 'periodic': 2, 'offsetDimensions': [2], 'hermitian': True})
        self.assertEqual(templates['lieb']['sites'], 3)
        self.assertEqual(templates['lieb']['bonds'], 4)
        self.assertEqual(templates['honeycomb']['sites'], 2)
        self.assertEqual(templates['honeycomb']['bonds'], 3)
        self.assertEqual(templates['triangular']['bonds'], 3)
        self.assertEqual(templates['kagome']['bonds'], 6)
        self.assertEqual(templates['dice']['bonds'], 6)
        self.assertEqual(templates['squareOctagon']['bonds'], 6)
        self.assertEqual(templates['ssh'], {'sites': 2, 'bonds': 2, 'kinds': ['strong', 'weak']})

    @unittest.skipUnless(shutil.which('node'), 'node is not installed')
    def test_builder_small_periodic_lattices_keep_wrap_bonds(self):
        builder_dir = Path(__file__).resolve().parents[2] / 'model_hamiltonian_ui'
        script = r"""
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const builderDir = process.argv[1];
const context = {
  console,
  Math,
  state: {
    globals: { epsilon: 0, U: 4, V: 0, t: -1, delta: 0.2 },
    dimension: 1,
    boundary: 'open',
    bondModulation: 'none',
  },
};
vm.createContext(context);
for (const file of ['hamiltonian.js', 'lattice_templates.js']) {
  vm.runInContext(fs.readFileSync(path.join(builderDir, file), 'utf8'), context);
}

function degreeHistogram(lattice) {
  const degree = new Map(lattice.sites.map((site) => [site.id, 0]));
  lattice.bonds.forEach((bond) => {
    degree.set(bond.source, (degree.get(bond.source) || 0) + 1);
    degree.set(bond.target, (degree.get(bond.target) || 0) + 1);
  });
  const histogram = {};
  Array.from(degree.values()).forEach((value) => {
    histogram[value] = (histogram[value] || 0) + 1;
  });
  return histogram;
}

function summarize(boundary, buildLattice) {
  context.state.boundary = boundary;
  const lattice = buildLattice(boundary);
  return {
    sites: lattice.sites.length,
    bonds: lattice.bonds.length,
    periodic: lattice.bonds.filter((bond) => bond.periodic).length,
    degreeHistogram: degreeHistogram(lattice),
  };
}

process.stdout.write(JSON.stringify({
  chainOpen: summarize('open', () => context.createChain(2, false, false)),
  chainPeriodic: summarize('periodic', () => context.createChain(2, true, false)),
  zigzagOpen: summarize('open', () => context.createChain(2, false, true)),
  zigzagPeriodic: summarize('periodic', () => context.createChain(2, true, true)),
  quasiOpen: summarize('open', () => context.createQuasi1D(2, 2)),
  quasiPeriodic: summarize('periodic', () => context.createQuasi1D(2, 2)),
  squareOpen: summarize('open', () => context.createSquare(2, 2)),
  squarePeriodic: summarize('periodic', () => context.createSquare(2, 2)),
  liebOpen: summarize('open', () => context.createLieb(2, 2)),
  liebPeriodic: summarize('periodic', () => context.createLieb(2, 2)),
  honeycombOpen: summarize('open', () => context.createHoneycomb(2, 2)),
  honeycombPeriodic: summarize('periodic', () => context.createHoneycomb(2, 2)),
  triangularOpen: summarize('open', () => context.createTriangular(2, 2)),
  triangularPeriodic: summarize('periodic', () => context.createTriangular(2, 2)),
  kagomeOpen: summarize('open', () => context.createKagome(2, 2)),
  kagomePeriodic: summarize('periodic', () => context.createKagome(2, 2)),
  diceOpen: summarize('open', () => context.createDice(2, 2)),
  dicePeriodic: summarize('periodic', () => context.createDice(2, 2)),
  squareOctagonOpen: summarize('open', () => context.createSquareOctagon(2, 2)),
  squareOctagonPeriodic: summarize('periodic', () => context.createSquareOctagon(2, 2)),
}));
"""

        result = subprocess.run(
            [shutil.which('node'), '-e', script, str(builder_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        counts = json.loads(result.stdout)

        expected = {
            'chainOpen': {'sites': 2, 'bonds': 1, 'periodic': 0, 'degreeHistogram': {'1': 2}},
            'chainPeriodic': {'sites': 2, 'bonds': 2, 'periodic': 1, 'degreeHistogram': {'2': 2}},
            'zigzagOpen': {'sites': 2, 'bonds': 1, 'periodic': 0, 'degreeHistogram': {'1': 2}},
            'zigzagPeriodic': {'sites': 2, 'bonds': 2, 'periodic': 1, 'degreeHistogram': {'2': 2}},
            'quasiOpen': {'sites': 4, 'bonds': 4, 'periodic': 0, 'degreeHistogram': {'2': 4}},
            'quasiPeriodic': {'sites': 4, 'bonds': 6, 'periodic': 2, 'degreeHistogram': {'3': 4}},
            'squareOpen': {'sites': 4, 'bonds': 4, 'periodic': 0, 'degreeHistogram': {'2': 4}},
            'squarePeriodic': {'sites': 4, 'bonds': 8, 'periodic': 4, 'degreeHistogram': {'4': 4}},
            'liebOpen': {'sites': 12, 'bonds': 12, 'periodic': 0, 'degreeHistogram': {'1': 4, '2': 5, '3': 2, '4': 1}},
            'liebPeriodic': {'sites': 12, 'bonds': 16, 'periodic': 4, 'degreeHistogram': {'2': 8, '4': 4}},
            'honeycombOpen': {'sites': 8, 'bonds': 7, 'periodic': 0, 'degreeHistogram': {'1': 4, '2': 2, '3': 2}},
            'honeycombPeriodic': {'sites': 8, 'bonds': 12, 'periodic': 5, 'degreeHistogram': {'3': 8}},
            'triangularOpen': {'sites': 4, 'bonds': 5, 'periodic': 0, 'degreeHistogram': {'2': 2, '3': 2}},
            'triangularPeriodic': {'sites': 4, 'bonds': 12, 'periodic': 7, 'degreeHistogram': {'6': 4}},
            'kagomeOpen': {'sites': 12, 'bonds': 17, 'periodic': 0, 'degreeHistogram': {'2': 5, '3': 4, '4': 3}},
            'kagomePeriodic': {'sites': 12, 'bonds': 24, 'periodic': 7, 'degreeHistogram': {'4': 12}},
            'diceOpen': {'sites': 16, 'bonds': 17, 'periodic': 0, 'degreeHistogram': {'1': 7, '2': 5, '3': 1, '4': 2, '6': 1}},
            'dicePeriodic': {'sites': 16, 'bonds': 24, 'periodic': 7, 'degreeHistogram': {'2': 12, '6': 4}},
            'squareOctagonOpen': {'sites': 16, 'bonds': 20, 'periodic': 0, 'degreeHistogram': {'2': 8, '3': 8}},
            'squareOctagonPeriodic': {'sites': 16, 'bonds': 24, 'periodic': 4, 'degreeHistogram': {'3': 16}},
        }
        self.assertEqual(counts, expected)

    @unittest.skipUnless(shutil.which('node'), 'node is not installed')
    def test_builder_single_cell_2d_lattices_preserve_periodic_coordination(self):
        builder_dir = Path(__file__).resolve().parents[2] / 'model_hamiltonian_ui'
        script = r"""
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const builderDir = process.argv[1];
const context = {
  console,
  Math,
  state: {
    globals: { epsilon: 0, U: 4, V: 0, t: -1, delta: 0.2 },
    dimension: 2,
    boundary: 'periodic',
    bondModulation: 'none',
  },
};
vm.createContext(context);
for (const file of ['hamiltonian.js', 'lattice_templates.js']) {
  vm.runInContext(fs.readFileSync(path.join(builderDir, file), 'utf8'), context);
}

function summarize(buildLattice, lx = 1, ly = 1) {
  const lattice = buildLattice(lx, ly);
  const degrees = new Map(lattice.sites.map((site) => [site.id, 0]));
  lattice.bonds.forEach((bond) => {
    degrees.set(bond.source, (degrees.get(bond.source) || 0) + 1);
    degrees.set(bond.target, (degrees.get(bond.target) || 0) + 1);
  });
  return {
    sites: lattice.sites.length,
    bonds: lattice.bonds.length,
    periodic: lattice.bonds.filter((bond) => bond.periodic).length,
    degrees: Array.from(degrees.values()).sort((left, right) => left - right),
  };
}

process.stdout.write(JSON.stringify({
  square: summarize(context.createSquare),
  lieb: summarize(context.createLieb),
  honeycomb: summarize(context.createHoneycomb),
  triangular: summarize(context.createTriangular),
  kagome: summarize(context.createKagome),
  dice: summarize(context.createDice),
  squareOctagon: summarize(context.createSquareOctagon),
  squareOneByTwo: summarize(context.createSquare, 1, 2),
  squareTwoByOne: summarize(context.createSquare, 2, 1),
}));
"""

        result = subprocess.run(
            [shutil.which('node'), '-e', script, str(builder_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        summaries = json.loads(result.stdout)

        self.assertEqual(summaries['square'], {'sites': 1, 'bonds': 2, 'periodic': 2, 'degrees': [4]})
        self.assertEqual(summaries['lieb'], {'sites': 3, 'bonds': 4, 'periodic': 2, 'degrees': [2, 2, 4]})
        self.assertEqual(summaries['honeycomb'], {'sites': 2, 'bonds': 3, 'periodic': 2, 'degrees': [3, 3]})
        self.assertEqual(summaries['triangular'], {'sites': 1, 'bonds': 3, 'periodic': 3, 'degrees': [6]})
        self.assertEqual(summaries['kagome'], {'sites': 3, 'bonds': 6, 'periodic': 3, 'degrees': [4, 4, 4]})
        self.assertEqual(summaries['dice'], {'sites': 4, 'bonds': 6, 'periodic': 3, 'degrees': [2, 2, 2, 6]})
        self.assertEqual(summaries['squareOctagon'], {'sites': 4, 'bonds': 6, 'periodic': 2, 'degrees': [3, 3, 3, 3]})
        self.assertEqual(summaries['squareOneByTwo'], {'sites': 2, 'bonds': 4, 'periodic': 3, 'degrees': [4, 4]})
        self.assertEqual(summaries['squareTwoByOne'], {'sites': 2, 'bonds': 4, 'periodic': 3, 'degrees': [4, 4]})

    @unittest.skipUnless(shutil.which('node'), 'node is not installed')
    def test_builder_offset_bond_generation_preserves_ssh_alternation(self):
        builder_dir = Path(__file__).resolve().parents[2] / 'model_hamiltonian_ui'
        script = r"""
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const builderDir = process.argv[1];
const context = {
  console,
  Math,
  state: {
    globals: { epsilon: 0, U: 4, V: 0, t: -1, delta: 0.2 },
    dimension: 1,
    boundary: 'open',
    bondModulation: 'ssh',
  },
};
vm.createContext(context);
for (const file of ['hamiltonian.js', 'lattice_templates.js']) {
  vm.runInContext(fs.readFileSync(path.join(builderDir, file), 'utf8'), context);
}

function kindsFor(lattice) {
  return lattice.bonds.map((bond) => ({
    source: bond.source,
    target: bond.target,
    kind: bond.kind,
    periodic: Boolean(bond.periodic),
  }));
}

const chainOpen = kindsFor(context.createChain(6, false, false));
const chainPeriodic = kindsFor(context.createChain(6, true, false));
context.state.boundary = 'periodic';
const quasiPeriodic = kindsFor(context.createQuasi1D(4, 2));
process.stdout.write(JSON.stringify({ chainOpen, chainPeriodic, quasiPeriodic }));
"""

        result = subprocess.run(
            [shutil.which('node'), '-e', script, str(builder_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        kinds = json.loads(result.stdout)

        self.assertEqual(
            [bond['kind'] for bond in kinds['chainOpen']],
            ['strong', 'weak', 'strong', 'weak', 'strong'],
        )
        self.assertEqual(
            [bond['kind'] for bond in kinds['chainPeriodic']],
            ['strong', 'weak', 'strong', 'weak', 'strong', 'weak'],
        )
        self.assertEqual(
            [bond for bond in kinds['chainPeriodic'] if bond['periodic']],
            [{'source': 0, 'target': 5, 'kind': 'weak', 'periodic': True}],
        )
        self.assertEqual(
            [bond['kind'] for bond in kinds['quasiPeriodic'] if bond['source'] % 2 == bond['target'] % 2],
            ['strong', 'strong', 'weak', 'weak', 'strong', 'strong', 'weak', 'weak'],
        )
        self.assertEqual(
            [bond for bond in kinds['quasiPeriodic'] if bond['periodic']],
            [
                {'source': 0, 'target': 6, 'kind': 'weak', 'periodic': True},
                {'source': 1, 'target': 7, 'kind': 'weak', 'periodic': True},
            ],
        )
        self.assertEqual(
            [bond['kind'] for bond in kinds['quasiPeriodic'] if bond['source'] % 2 != bond['target'] % 2],
            ['strong', 'strong', 'strong', 'strong'],
        )

    def test_save_model_hamiltonian_input_file_writes_expected_locations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            content = 'model_spec = {"model": "hubbard"}\n'

            with self.subTest('explicit output path'):
                output_path = Path(tmpdir) / 'nested' / 'pyscf_model_hamiltonian_input.py'
                result = web.save_model_hamiltonian_input_file(content, output_path)

                self.assertEqual(result['status'], 'ok')
                self.assertEqual(result['path'], str(output_path))
                self.assertEqual(result['run_dir'], str(output_path.parent))
                self.assertEqual(output_path.read_text(encoding='utf-8'), content)

            with self.subTest('work dir and run id'):
                result = web.save_model_hamiltonian_input_file(
                    content,
                    work_dir=tmpdir,
                    run_id='builder-test',
                )

                logical_tmpdir = Path(os.path.abspath(tmpdir))
                expected_path = logical_tmpdir / 'builder-test' / 'input-builder-model-hamiltonian.py'
                self.assertEqual(result['status'], 'ok')
                self.assertEqual(result['run_id'], 'builder-test')
                self.assertEqual(result['work_dir'], str(logical_tmpdir))
                self.assertEqual(result['path'], str(expected_path))
                self.assertEqual(expected_path.read_text(encoding='utf-8'), content)

            with self.subTest('run directory path is not nested twice'):
                logical_tmpdir = Path(os.path.abspath(tmpdir))
                run_dir = logical_tmpdir / 'builder-run'
                result = web.save_model_hamiltonian_input_file(
                    content,
                    work_dir=str(run_dir),
                    run_id='builder-run',
                )

                expected_path = run_dir / 'input-builder-model-hamiltonian.py'
                self.assertEqual(result['work_dir'], str(logical_tmpdir))
                self.assertEqual(result['run_dir'], str(run_dir))
                self.assertEqual(result['path'], str(expected_path))
                self.assertFalse((run_dir / 'builder-run').exists())

    def test_save_model_hamiltonian_input_file_rejects_empty_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'pyscf_model_hamiltonian_input.py'

            with self.assertRaises(ValueError):
                web.save_model_hamiltonian_input_file('', output_path)

            self.assertFalse(output_path.exists())


if __name__ == '__main__':
    unittest.main()
