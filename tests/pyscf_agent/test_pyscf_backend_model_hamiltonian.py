from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pyscf_agent.backend.artifacts import retry_artifact_filename
from pyscf_agent.backend.execution import result_analyst
from pyscf_agent.backend.parsing import intent_parser, spec_builder, spec_validator
from pyscf_agent.backend.state import default_state
from pyscf_agent.backend.workflow import run_workflow_sequential
from pyscf_agent.backend.model_hamiltonian.solver import (
    _model_root_count,
    _natural_occupation_summary_from_dm1,
    generate_model_hamiltonian_input_script,
    normalize_model_spec,
    normalize_solver_name,
    parse_model_spec_from_python_input,
    run_model_hamiltonian_solver,
    validate_model_hamiltonian_spec,
)
from pyscf_agent.backend.model_hamiltonian.bloch import build_bloch_hamiltonian
from pyscf_agent.backend.model_hamiltonian.observables import _compute_strong_correlation_diagnostics


def hubbard_dimer_spec():
    return {
        'schema': 'pyscf-agent.model-hamiltonian.v1',
        'model': 'hubbard',
        'dimension': 1,
        'preset': 'chain',
        'boundary': 'open',
        'nelec': [1, 1],
        'sites': [
            {'id': 0, 'x': 0, 'y': 0, 'epsilon': 0, 'U': 4},
            {'id': 1, 'x': 1, 'y': 0, 'epsilon': 0, 'U': 4},
        ],
        'bonds': [
            {'id': 0, 'source': 0, 'target': 1, 't': -1, 'V': 0, 'effective_t': -1},
        ],
    }


def open_shell_hubbard_trimer_spec():
    return {
        'schema': 'pyscf-agent.model-hamiltonian.v1',
        'model': 'hubbard',
        'dimension': 1,
        'preset': 'chain',
        'boundary': 'open',
        'nelec': [2, 1],
        'sites': [
            {'id': 0, 'x': 0, 'y': 0, 'epsilon': 0, 'U': 2},
            {'id': 1, 'x': 1, 'y': 0, 'epsilon': 0, 'U': 2},
            {'id': 2, 'x': 2, 'y': 0, 'epsilon': 0, 'U': 2},
        ],
        'bonds': [
            {'id': 0, 'source': 0, 'target': 1, 't': -1, 'V': 0, 'effective_t': -1},
            {'id': 1, 'source': 1, 'target': 2, 't': -1, 'V': 0, 'effective_t': -1},
        ],
    }


def hubbard_chain_spec(site_count=6, nalpha=3, nbeta=3, t=-1, u=4):
    return {
        'schema': 'pyscf-agent.model-hamiltonian.v1',
        'model': 'hubbard',
        'dimension': 1,
        'preset': 'chain',
        'boundary': 'open',
        'energy_unit': 'a.u.',
        'nelec': [nalpha, nbeta],
        'sites': [
            {'id': index, 'x': index, 'y': 0, 'epsilon': 0, 'U': u}
            for index in range(site_count)
        ],
        'bonds': [
            {
                'id': index,
                'source': index,
                'target': index + 1,
                't': t,
                'V': 0,
                'effective_t': t,
                'effective_V': 0,
            }
            for index in range(site_count - 1)
        ],
    }


def bloch_chain_spec(*, ssh=False, u=0):
    sites = [
        {'id': 0, 'x': 0, 'y': 0, 'epsilon': 0, 'U': u},
    ]
    bonds = [
        {
            'id': 0,
            'source': 0,
            'target': 0,
            't': -1,
            'V': 0,
            'cell_offset': [1],
            'add_hermitian_conjugate': True,
        },
    ]
    electrons_per_cell = 1
    if ssh:
        sites.append({'id': 1, 'x': 1, 'y': 0, 'epsilon': 0, 'U': u})
        bonds = [
            {
                'id': 0,
                'source': 0,
                'target': 1,
                't': -1.2,
                'V': 0,
                'cell_offset': [0],
                'add_hermitian_conjugate': True,
            },
            {
                'id': 1,
                'source': 0,
                'target': 1,
                't': -0.8,
                'V': 0,
                'cell_offset': [-1],
                'add_hermitian_conjugate': True,
            },
        ]
        electrons_per_cell = 2
    return {
        'schema': 'pyscf-agent.model-hamiltonian.v1',
        'model': 'hubbard',
        'representation': 'bloch',
        'solver': 'tight_binding',
        'dimension': 1,
        'preset': 'chain',
        'boundary': 'periodic',
        'energy_unit': 'a.u.',
        'lattice_vectors': [[2, 0] if ssh else [1, 0]],
        'occupation': {
            'mode': 'filling',
            'electrons_per_cell': electrons_per_cell,
            'spin_degeneracy': 2,
        },
        'reciprocal_space': {
            'kmesh': [100 if ssh else 101],
            'scheme': 'gamma_centered',
            'shift': [0],
            'path': {'mode': 'automatic', 'points_per_segment': 21},
            'dos': {'points': 101, 'sigma': None},
        },
        'sites': sites,
        'bonds': bonds,
    }


class ModelHamiltonianBackendTests(unittest.TestCase):
    def test_generated_script_preserves_literal_input_and_solver_options(self):
        text = '中文 MODEL_SPEC_JSON SOLVER_NAME OUTPUTS SOLVER_OPTIONS RUNTIME_OPTIONS """\ntrailing\\'
        spec = hubbard_dimer_spec()
        spec['label'] = text
        spec['sites'][0]['label'] = text
        options = {'note': text, 'nested': {'value': 'SOLVER_OPTIONS'}}
        script = generate_model_hamiltonian_input_script(spec, solver_name='ccsd', outputs=['energy'], solver_options=options)
        expected = normalize_model_spec(spec)
        expected['solver'] = 'ccsd'

        self.assertEqual(parse_model_spec_from_python_input(script), expected)
        with mock.patch(
            'pyscf_agent.backend.model_hamiltonian.solver.run_model_hamiltonian_solver',
            return_value={'energy': -1.},
        ) as run_solver, contextlib.redirect_stdout(io.StringIO()):
            exec(compile(script, '<generated-model-input>', 'exec'), {})
        self.assertEqual(run_solver.call_args.args[0], expected)
        self.assertEqual(run_solver.call_args.kwargs['solver_name'], 'ccsd')
        self.assertEqual(run_solver.call_args.kwargs['outputs'], ['energy'])
        self.assertEqual(run_solver.call_args.kwargs['solver_options'], options)

    def test_solver_normalization_uses_registry_without_ambiguous_guesses(self):
        self.assertEqual(normalize_solver_name('CCSDT'), 'ccsd_t')
        self.assertEqual(normalize_solver_name('exact diagonalization'), 'fci')
        self.assertEqual(normalize_solver_name('band'), 'band')
        self.assertEqual(normalize_solver_name('bands'), 'bands')
        self.assertEqual(normalize_solver_name('cc'), 'cc')

    def test_dmet_summary_reports_per_site_energy_without_reference_difference(self):
        analyzed = result_analyst({
            'locale': 'en',
            'execution_status': 'succeeded',
            'messages': [],
            'logs': [],
            'structured_results': {
                'task_type': 'model_hamiltonian',
                'solver': 'dmet',
                'model': 'hubbard',
                'energy': -25.6,
                'energy_per_site': -0.8,
                'energy_unit': 'a.u.',
                'reference_energy': -18.0,
                'correlation_energy': -7.6,
            },
        })

        summary = analyzed['analysis_summary']
        self.assertIn('energy=-25.6 a.u.', summary)
        self.assertIn('energy_per_site=-0.8 a.u./site', summary)
        self.assertNotIn('reference_energy', summary)
        self.assertNotIn('correlation_energy', summary)

    def test_model_spin_multiplicity_is_derived_and_validated(self):
        spec = open_shell_hubbard_trimer_spec()
        normalized = normalize_model_spec(spec)
        self.assertEqual(normalized['spin_multiplicity'], 2)
        self.assertEqual(validate_model_hamiltonian_spec(normalized), [])

        normalized['spin_multiplicity'] = 4
        errors = validate_model_hamiltonian_spec(normalized)
        self.assertTrue(any('spin_multiplicity must equal' in item for item in errors))

    def test_bloch_chain_hamiltonian_has_cosine_dispersion(self):
        spec = bloch_chain_spec()

        gamma = build_bloch_hamiltonian(spec, [0.0])
        zone_edge = build_bloch_hamiltonian(spec, [0.5])

        self.assertAlmostEqual(float(gamma[0, 0].real), -2.0)
        self.assertAlmostEqual(float(zone_edge[0, 0].real), 2.0)
        self.assertAlmostEqual(float(gamma[0, 0].imag), 0.0)

    def test_bloch_chain_reports_band_metrics_and_one_body_boundary(self):
        result = run_model_hamiltonian_solver(
            bloch_chain_spec(u=4),
            solver_name='tight_binding',
            outputs=['energy', 'strong_correlation_diagnostics'],
        )

        self.assertEqual(result['representation'], 'bloch')
        self.assertEqual(result['solver'], 'tight_binding')
        self.assertEqual(result['energy_kind'], 'noninteracting_band_energy_per_cell')
        self.assertAlmostEqual(result['energy'], -4.0 / 3.141592653589793, places=3)
        self.assertEqual(result['band_gap'], 0.0)
        self.assertTrue(result['is_metal'])
        self.assertEqual(result['electrons_per_cell'], 1.0)
        self.assertEqual(result['filling'], 1.0)
        self.assertEqual(result['kmesh'], [101])
        self.assertEqual(result['kpoint_count'], 101)
        self.assertEqual(result['interaction_treatment']['status'], 'recorded_not_applied')
        self.assertFalse(result['interaction_treatment']['interaction_terms_applied'])
        self.assertIn('does not evaluate strong-correlation diagnostics', result['strong_correlation_diagnostics_unavailable_reason'])
        self.assertEqual(len(result['bloch_band_structure']['energies']), 41)
        self.assertEqual(len(result['bloch_dos']['energy']), 101)
        self.assertEqual(len(result['bloch_kmesh']['occupations']), 101)

    def test_bloch_ssh_chain_reports_insulating_gap(self):
        result = run_model_hamiltonian_solver(bloch_chain_spec(ssh=True), solver_name='tight_binding')

        self.assertFalse(result['is_metal'])
        self.assertAlmostEqual(result['band_gap'], 0.8, places=6)
        self.assertAlmostEqual(result['direct_gap'], 0.8, places=6)
        self.assertEqual(result['fermi_energy_source'], 'valence_band_max')

    def test_honeycomb_automatic_path_passes_through_dirac_k_point(self):
        spec = bloch_chain_spec()
        spec.update({
            'dimension': 2,
            'preset': 'honeycomb',
            'lattice_vectors': [[3 ** 0.5, 0], [3 ** 0.5 / 2, 1.5]],
            'occupation': {'mode': 'filling', 'electrons_per_cell': 2, 'spin_degeneracy': 2},
            'reciprocal_space': {
                'kmesh': [30, 30],
                'scheme': 'gamma_centered',
                'shift': [0, 0],
                'path': {'mode': 'automatic', 'points_per_segment': 11},
                'dos': {'points': 51, 'sigma': None},
            },
            'sites': [
                {'id': 0, 'x': 0, 'y': 0, 'epsilon': 0, 'U': 0},
                {'id': 1, 'x': 0, 'y': 1, 'epsilon': 0, 'U': 0},
            ],
            'bonds': [
                {'id': 0, 'source': 0, 'target': 1, 't': -1, 'V': 0, 'cell_offset': [0, -1], 'add_hermitian_conjugate': True},
                {'id': 1, 'source': 0, 'target': 1, 't': -1, 'V': 0, 'cell_offset': [0, 0], 'add_hermitian_conjugate': True},
                {'id': 2, 'source': 0, 'target': 1, 't': -1, 'V': 0, 'cell_offset': [1, -1], 'add_hermitian_conjugate': True},
            ],
        })

        k_matrix = build_bloch_hamiltonian(spec, [1.0 / 3.0, -1.0 / 3.0])
        result = run_model_hamiltonian_solver(spec, solver_name='tight_binding')
        k_label = next(item for item in result['bloch_band_structure']['labels'] if item['label'] == 'K')
        k_energies = result['bloch_band_structure']['energies'][k_label['index']]

        self.assertAlmostEqual(abs(k_matrix[0, 1]), 0.0, places=12)
        self.assertAlmostEqual(k_energies[0], 0.0, places=12)
        self.assertAlmostEqual(k_energies[1], 0.0, places=12)

    def test_bloch_validation_requires_integer_cell_offsets_and_tight_binding(self):
        spec = bloch_chain_spec()
        spec['bonds'][0]['cell_offset'] = [0.5]

        errors = validate_model_hamiltonian_spec(spec, solver_name='fci')

        self.assertTrue(any('require the tight_binding solver' in message for message in errors))
        self.assertTrue(any('cell_offset[0] must be an integer' in message for message in errors))

    def test_bloch_validation_rejects_oversized_reciprocal_requests(self):
        oversized_mesh = bloch_chain_spec()
        oversized_mesh['reciprocal_space']['kmesh'] = [50001]
        mesh_errors = validate_model_hamiltonian_spec(oversized_mesh, solver_name='tight_binding')

        oversized_path = bloch_chain_spec()
        oversized_path['reciprocal_space']['path']['points_per_segment'] = 10002
        path_errors = validate_model_hamiltonian_spec(oversized_path, solver_name='tight_binding')

        oversized_vertices = bloch_chain_spec()
        oversized_vertices['reciprocal_space']['path'] = {
            'mode': 'custom',
            'points_per_segment': 2,
            'points': [{'label': 'K{0}'.format(index), 'k': [index / 100.0]} for index in range(65)],
        }
        vertex_errors = validate_model_hamiltonian_spec(oversized_vertices, solver_name='tight_binding')

        oversized_dos = bloch_chain_spec()
        oversized_dos['reciprocal_space']['dos']['points'] = 4002
        dos_errors = validate_model_hamiltonian_spec(oversized_dos, solver_name='tight_binding')

        oversized_work = bloch_chain_spec()
        oversized_work['reciprocal_space']['kmesh'] = [50000]
        oversized_work['reciprocal_space']['dos']['points'] = 4001
        work_errors = validate_model_hamiltonian_spec(oversized_work, solver_name='tight_binding')

        self.assertTrue(any('current limit is 50000' in message for message in mesh_errors))
        self.assertTrue(any('path generates' in message for message in path_errors))
        self.assertTrue(any('current limit is 64' in message for message in vertex_errors))
        self.assertTrue(any('current limit of 4001' in message for message in dos_errors))
        self.assertTrue(any('Reduce kmesh' in message for message in work_errors))

    def test_bloch_workflow_writes_plotting_artifacts(self):
        spec = bloch_chain_spec(ssh=True)
        script = generate_model_hamiltonian_input_script(spec, solver_name='tight_binding')
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / 'input-builder-model-hamiltonian.py'
            input_path.write_text(script, encoding='utf-8')
            request = json.dumps({
                'task_type': 'model_hamiltonian',
                'model_hamiltonian_input_file': str(input_path),
                'outputs': ['energy'],
            })

            report = run_workflow_sequential(
                default_state(request, channel='test', locale='en', work_dir=tmpdir, run_id='bloch-run')
            )['task_report']

            self.assertEqual(report['execution_status'], 'succeeded')
            self.assertEqual(report['structured_results']['representation'], 'bloch')
            artifact_names = {Path(item['path']).name for item in report['artifacts']}
            self.assertIn('result-model-bloch-bands.json', artifact_names)
            self.assertIn('result-model-bloch-bands.tsv', artifact_names)
            self.assertIn('result-model-bloch-dos.json', artifact_names)
            self.assertIn('result-model-bloch-dos.tsv', artifact_names)
            self.assertIn('result-model-bloch-kmesh.json', artifact_names)
            self.assertIn('result-model-bloch-kmesh.tsv', artifact_names)
            self.assertIn('result-model-bloch-hamiltonian-samples.json', artifact_names)
            self.assertIn('representation=bloch', report['analysis_summary'])

    def test_retry_artifact_filename_omits_attempt_for_first_run(self):
        self.assertEqual(retry_artifact_filename('result-structured', 'json', 0), 'result-structured.json')
        self.assertEqual(retry_artifact_filename('result-structured', 'json', 1), 'result-structured-retry-1.json')

    def test_parse_builder_python_input_extracts_model_spec(self):
        spec = hubbard_dimer_spec()
        script = generate_model_hamiltonian_input_script(spec)

        parsed = parse_model_spec_from_python_input(script)

        self.assertEqual(parsed['model'], 'hubbard')
        self.assertEqual(parsed['nelec'], [1, 1])
        self.assertEqual(len(parsed['sites']), 2)

    def test_normalize_model_spec_adds_site_bond_graph_view(self):
        normalized = normalize_model_spec(hubbard_dimer_spec())

        self.assertEqual(normalized['graph']['representation'], 'site-bond graph')
        self.assertEqual(normalized['graph']['nodes'][0]['id'], 0)
        self.assertEqual(normalized['graph']['edges'][0]['endpoints'], [0, 1])

    def test_sequential_workflow_runs_hubbard_model_hamiltonian_from_file(self):
        spec = hubbard_dimer_spec()
        script = generate_model_hamiltonian_input_script(spec)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / 'pyscf_model_hamiltonian_input.py'
            input_path.write_text(script, encoding='utf-8')
            request = json.dumps({
                'task_type': 'model_hamiltonian',
                'model_hamiltonian_input_file': str(input_path),
                'solver': 'fci',
                'outputs': ['energy', 'gap', 'strong_correlation_diagnostics'],
            })

            report = run_workflow_sequential(
                default_state(request, channel='test', locale='en', work_dir=tmpdir, run_id='model-run')
            )['task_report']

            self.assertEqual(report['execution_status'], 'succeeded')
            self.assertEqual(report['run_id'], 'model-run')
            self.assertEqual(report['structured_results']['task_type'], 'model_hamiltonian')
            self.assertEqual(report['structured_results']['model'], 'hubbard')
            self.assertEqual(report['structured_results']['norb'], 2)
            self.assertEqual(report['structured_results']['nelec'], [1, 1])
            self.assertIsInstance(report['structured_results']['energy'], float)
            self.assertEqual(report['structured_results']['energy_unit'], 'a.u.')
            self.assertEqual(report['structured_results']['energy_level_count'], 4)
            self.assertEqual(report['structured_results']['many_body_basis_dimension'], 4)
            self.assertEqual(report['structured_results']['energy_spectrum_method'], 'fci')
            self.assertIn('Model Hamiltonian FCI completed', report['analysis_summary'])
            self.assertIn('FCI_energy_levels=4 saved as artifacts', report['analysis_summary'])
            self.assertTrue(any(item['kind'] == 'model_hamiltonian_spec' for item in report['artifacts']))
            self.assertTrue(any(item['kind'] == 'model_hamiltonian_energy_levels' for item in report['artifacts']))
            self.assertTrue(any(item['kind'] == 'model_hamiltonian_energy_levels_txt' for item in report['artifacts']))
            self.assertTrue(any(item['kind'] == 'strong_correlation_diagnostics' for item in report['artifacts']))
            completed_modules = {
                item.get('module_id')
                for item in report['module_runtime_observations']
                if item.get('status') == 'completed'
            }
            self.assertIn('model.correlation_diagnostics', completed_modules)
            for artifact in report['artifacts']:
                self.assertTrue({'kind', 'path', 'size_bytes', 'mime_type', 'description'} <= set(artifact))
                self.assertNotIn('sha256', artifact)
            self.assertTrue(all('/model-run/' in item['path'] for item in report['artifacts']))
            artifact_names = {Path(item['path']).name for item in report['artifacts']}
            self.assertIn('input-generated-pyscf.py', artifact_names)
            self.assertIn('result-structured.json', artifact_names)
            self.assertIn('artifact-model-hamiltonian-spec.json', artifact_names)
            self.assertIn('result-model-energy-levels.json', artifact_names)
            self.assertIn('result-model-energy-levels.txt', artifact_names)
            self.assertIn('log-stdout.log', artifact_names)
            spectrum_artifact = next(item for item in report['artifacts'] if item['kind'] == 'model_hamiltonian_energy_levels')
            spectrum_payload = json.loads(Path(spectrum_artifact['path']).read_text(encoding='utf-8'))
            self.assertEqual(spectrum_payload['energy_level_count'], 4)
            self.assertEqual(spectrum_payload['schema'], 'pyscf-agent.model-energy-levels.v1')
            self.assertEqual(spectrum_payload['spectrum_method'], 'fci')
            self.assertEqual(spectrum_payload['energy_unit'], 'a.u.')
            self.assertEqual(len(spectrum_payload['levels']), 4)
            table_artifact = next(item for item in report['artifacts'] if item['kind'] == 'model_hamiltonian_energy_levels_txt')
            table_text = Path(table_artifact['path']).read_text(encoding='utf-8')
            self.assertTrue(table_text.startswith('index\tenergy\tunit\n'))
            self.assertIn('\ta.u.\n', table_text)
            diagnostics_artifact = next(
                item for item in report['artifacts']
                if item['kind'] == 'strong_correlation_diagnostics'
            )
            diagnostics_payload = json.loads(Path(diagnostics_artifact['path']).read_text(encoding='utf-8'))
            self.assertEqual(
                diagnostics_payload['schema'],
                'pyscf-agent.strong-correlation-diagnostics.v1',
            )

    def test_model_hamiltonian_input_file_overrides_stale_inline_spec(self):
        stale_spec = hubbard_chain_spec(u=4)
        current_spec = hubbard_chain_spec(u=8)
        script = generate_model_hamiltonian_input_script(current_spec)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / 'pyscf_model_hamiltonian_input.py'
            input_path.write_text(script, encoding='utf-8')
            request = json.dumps({
                'task_type': 'model_hamiltonian',
                'model_hamiltonian_input_file': str(input_path),
                'model_hamiltonian': {'spec': stale_spec},
                'solver': 'fci',
            })

            report = run_workflow_sequential(
                default_state(request, channel='test', locale='en', work_dir=tmpdir, run_id='model-run')
            )['task_report']

        self.assertEqual(report['execution_status'], 'succeeded')
        self.assertAlmostEqual(report['structured_results']['energy'], -1.7680987552612635)
        self.assertEqual(report['structured_results']['onsite_u_count'], 6)
        self.assertEqual(report['task_spec']['model_hamiltonian']['spec']['sites'][0]['U'], 8)

    def test_explicit_model_solver_replaces_builder_solver_metadata(self):
        spec = hubbard_dimer_spec()
        script = generate_model_hamiltonian_input_script(spec, solver_name='fci')
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / 'pyscf_model_hamiltonian_input.py'
            input_path.write_text(script, encoding='utf-8')
            request = json.dumps({
                'task_type': 'model_hamiltonian',
                'model_hamiltonian_input_file': str(input_path),
                'solver': 'block2_dmrg',
                'outputs': ['energy'],
            })
            state = intent_parser(default_state(
                request,
                channel='test',
                locale='en',
                work_dir=tmpdir,
                run_id='solver-provenance',
            ))
            state = spec_builder(state)

        self.assertEqual(state['task_spec']['solver']['name'], 'block2_dmrg')
        self.assertEqual(
            state['task_spec']['model_hamiltonian']['spec']['solver'],
            'block2_dmrg',
        )

    def test_model_hamiltonian_preserves_declared_energy_unit(self):
        spec = hubbard_dimer_spec()
        spec['energy_unit'] = 'eV'

        result = run_model_hamiltonian_solver(spec, solver_name='fci')

        self.assertEqual(result['energy_unit'], 'eV')
        self.assertIn(' energy=', result['analysis_text'])
        self.assertIn(' eV', result['analysis_text'])

    def test_model_hamiltonian_solvers_support_closed_shell(self):
        observable_outputs = [
            'energy',
            'gap',
            'density',
            'double_occupancy',
            'spin_correlation',
            'charge_correlation',
        ]
        for solver_name in ('mp2', 'ccsd', 'ccsd_t', 'fci'):
            with self.subTest(solver=solver_name):
                result = run_model_hamiltonian_solver(
                    hubbard_dimer_spec(),
                    solver_name=solver_name,
                    outputs=observable_outputs,
                )

                self.assertEqual(result['solver'], solver_name)
                self.assertEqual(result['nelec'], [1, 1])
                self.assertIsInstance(result['energy'], float)
                if solver_name == 'fci':
                    self.assertEqual(result['reference'], 'direct_spin1')
                    self.assertEqual(result['correlation_reference'], 'uhf')
                    self.assertFalse(result['restricted_reference'])
                    self.assertIsInstance(result['reference_energy'], float)
                    self.assertIsInstance(result['correlation_energy'], float)
                    self.assertEqual(result['energy_level_count'], 4)
                    self.assertEqual(result['energy_spectrum_method'], 'fci')
                    self.assertEqual(len(result['energy_levels']), 4)
                    self.assertAlmostEqual(result['gap'], result['energy_levels'][1] - result['energy_levels'][0])
                    self.assertEqual(result['density']['kind'], 'site_vector')
                    self.assertEqual(result['density']['operator'], 'n_i_up + n_i_down')
                    self.assertEqual(result['density']['shape'], [2])
                    self.assertEqual(result['density']['site_ids'], [0, 1])
                    self.assertEqual(len(result['density']['values']), 2)
                    self.assertAlmostEqual(sum(result['density']['values']), 2.0)
                    self.assertAlmostEqual(result['density_mean'], 1.0)
                    self.assertEqual(result['double_occupancy']['kind'], 'site_vector')
                    self.assertEqual(result['double_occupancy']['operator'], 'n_i_up n_i_down')
                    self.assertEqual(result['double_occupancy']['shape'], [2])
                    self.assertEqual(result['double_occupancy']['site_ids'], [0, 1])
                    self.assertEqual(len(result['double_occupancy']['values']), 2)
                    self.assertAlmostEqual(sum(result['double_occupancy']['values']), 0.14644660940672624)
                    self.assertAlmostEqual(result['double_occupancy_mean'], 0.07322330470336312)
                    self.assertEqual(result['spin_correlation']['kind'], 'pair_matrix')
                    self.assertEqual(result['spin_correlation']['operator'], 'S_i dot S_j')
                    self.assertEqual(result['spin_correlation']['shape'], [2, 2])
                    self.assertEqual(result['spin_correlation']['site_ids'], [0, 1])
                    spin_matrix = result['spin_correlation']['values']
                    self.assertEqual(len(spin_matrix), 2)
                    self.assertEqual(len(spin_matrix[0]), 2)
                    self.assertAlmostEqual(spin_matrix[0][0], 0.6401650429449551)
                    self.assertAlmostEqual(spin_matrix[1][1], 0.6401650429449552)
                    self.assertAlmostEqual(spin_matrix[0][1], -0.6401650429449551)
                    self.assertAlmostEqual(spin_matrix[1][0], -0.6401650429449551)
                    self.assertAlmostEqual(sum(sum(row) for row in spin_matrix), 0.0)
                    self.assertEqual(result['charge_correlation']['kind'], 'pair_matrix')
                    self.assertEqual(result['charge_correlation']['operator'], '(n_i - <n_i>)(n_j - <n_j>)')
                    self.assertEqual(result['charge_correlation']['shape'], [2, 2])
                    self.assertEqual(result['charge_correlation']['site_ids'], [0, 1])
                    charge_matrix = result['charge_correlation']['values']
                    self.assertEqual(len(charge_matrix), 2)
                    self.assertEqual(len(charge_matrix[0]), 2)
                    self.assertAlmostEqual(charge_matrix[0][0], 0.14644660940672627)
                    self.assertAlmostEqual(charge_matrix[1][1], 0.14644660940672627)
                    self.assertAlmostEqual(charge_matrix[0][1], -0.14644660940672627)
                    self.assertAlmostEqual(charge_matrix[1][0], -0.14644660940672627)
                    self.assertAlmostEqual(sum(sum(row) for row in charge_matrix), 0.0)
                else:
                    self.assertEqual(result['reference'], 'uhf')
                    self.assertFalse(result['restricted_reference'])
                    self.assertIsInstance(result['reference_energy'], float)
                    self.assertIsInstance(result['correlation_energy'], float)
                    if solver_name == 'ccsd_t':
                        self.assertIsInstance(result['triples_correction'], float)
                    self.assertNotIn('energy_levels', result)
                    self.assertIn('does not define a complete excited-state spectrum', result['energy_spectrum_unavailable_reason'])

    def test_model_fci_solver_requests_single_root(self):
        from pyscf import fci  # pylint: disable=import-outside-toplevel

        original_kernel = fci.direct_spin1.kernel
        requested_roots = []

        def wrapped_kernel(*args, **kwargs):
            requested_roots.append(kwargs.get('nroots'))
            return original_kernel(*args, **kwargs)

        with mock.patch('pyscf.fci.direct_spin1.kernel', side_effect=wrapped_kernel):
            run_model_hamiltonian_solver(hubbard_dimer_spec(), solver_name='fci', outputs=['energy'])

        self.assertEqual(requested_roots, [1])

    def test_dmet_rejects_any_explicit_root_contract(self):
        with self.assertRaisesRegex(ValueError, 'supported only by FCI and block2 DMRG'):
            _model_root_count('dmet', ['energy'], {'nroots': 1})
        with self.assertRaisesRegex(ValueError, 'supported only by FCI and block2 DMRG'):
            _model_root_count('dmet', ['energy'], {'nroots': 2})

    def test_model_fci_solver_returns_requested_low_lying_roots(self):
        result = run_model_hamiltonian_solver(
            hubbard_dimer_spec(),
            solver_name='fci',
            outputs=['energy', 'excited_states', 'symmetry_analysis'],
            solver_options={'nroots': 3},
        )

        self.assertEqual(result['requested_root_count'], 3)
        self.assertEqual(result['computed_root_count'], 3)
        self.assertTrue(result['targeted_roots_complete'])
        self.assertEqual(len(result['state_energies']), 3)
        self.assertEqual(len(result['excitation_energies']), 3)
        self.assertAlmostEqual(result['energy'], result['state_energies'][0])
        self.assertAlmostEqual(result['excitation_energies'][0], 0.0)
        self.assertEqual(len(result['symmetry_analysis']['root_spin_square']), 3)
        self.assertEqual(result['missing_requested_outputs'], [])
        self.assertNotIn('energy_levels', result)

    def test_model_fci_excited_state_output_defaults_to_two_roots(self):
        result = run_model_hamiltonian_solver(
            hubbard_dimer_spec(),
            solver_name='fci',
            outputs=['energy', 'excited_states'],
        )

        self.assertEqual(result['requested_root_count'], 2)
        self.assertEqual(len(result['state_energies']), 2)
        self.assertTrue(result['targeted_roots_complete'])

    def test_model_fci_rejects_more_roots_than_the_spin_sector_contains(self):
        with self.assertRaisesRegex(ValueError, 'contains only 4 states'):
            run_model_hamiltonian_solver(
                hubbard_dimer_spec(),
                solver_name='fci',
                outputs=['energy', 'excited_states'],
                solver_options={'nroots': 5},
            )

    def test_model_fci_root_capacity_is_rejected_during_request_validation(self):
        request = json.dumps({
            'task_type': 'model_hamiltonian',
            'model_hamiltonian': {'spec': hubbard_dimer_spec()},
            'solver': {'name': 'fci', 'options': {'nroots': 5}},
            'outputs': ['energy', 'excited_states'],
        })
        state = intent_parser(default_state(request, channel='test', locale='en'))
        state = spec_builder(state)
        state = spec_validator(state)

        self.assertIn('model_nroots_exceeds_spin_sector', {
            item.get('code') for item in state['errors']
        })

    def test_model_hamiltonian_ccsd_t_alias_returns_triples_correction(self):
        result = run_model_hamiltonian_solver(
            hubbard_dimer_spec(),
            solver_name='CCSD(T)',
            outputs=['energy'],
        )

        self.assertEqual(result['solver'], 'ccsd_t')
        self.assertIsInstance(result['triples_correction'], float)

    def test_model_hamiltonian_observables_are_computed_on_request(self):
        default_result = run_model_hamiltonian_solver(hubbard_dimer_spec(), solver_name='fci')
        requested_result = run_model_hamiltonian_solver(
            hubbard_dimer_spec(),
            solver_name='fci',
            outputs=['energy', 'density'],
        )

        self.assertIn('energy', default_result)
        self.assertNotIn('density', default_result)
        self.assertNotIn('spin_correlation', default_result)
        self.assertNotIn('energy_levels', default_result)
        self.assertNotIn('energy_spectrum_unavailable_reason', default_result)
        self.assertIn('density', requested_result)
        self.assertNotIn('spin_correlation', requested_result)
        self.assertNotIn('energy_levels', requested_result)

    def test_dmrg_targeted_roots_are_not_reported_as_an_incomplete_spectrum(self):
        solver_result = {
            'solver': 'block2_dmrg',
            'converged': True,
            'energy': -1.0,
            'dmrg_result': {
                'configuration': {'nroots': 2},
            },
            'state_energies': [-1.0, -0.8],
            'excitation_energies': [0.0, 0.2],
            'symmetry_analysis': {'spin_square': 0.0},
        }
        with mock.patch(
            'pyscf_agent.backend.model_hamiltonian.solver._run_model_post_hf_solver',
            return_value=solver_result,
        ):
            result = run_model_hamiltonian_solver(
                hubbard_dimer_spec(),
                solver_name='block2_dmrg',
                outputs=['energy', 'gap', 'excited_states', 'symmetry_analysis'],
            )

        self.assertEqual(result['requested_root_count'], 2)
        self.assertEqual(result['computed_root_count'], 2)
        self.assertTrue(result['targeted_roots_complete'])
        self.assertAlmostEqual(result['gap'], 0.2)
        self.assertEqual(result['missing_requested_outputs'], [])
        self.assertNotIn('energy_spectrum_unavailable_reason', result)

    def test_dmrg_only_warns_when_a_complete_spectrum_output_is_requested(self):
        solver_result = {
            'solver': 'block2_dmrg',
            'converged': True,
            'energy': -1.0,
            'dmrg_result': {
                'configuration': {'nroots': 1},
            },
            'state_energies': [-1.0],
            'excitation_energies': [0.0],
        }
        with mock.patch(
            'pyscf_agent.backend.model_hamiltonian.solver._run_model_post_hf_solver',
            return_value=solver_result,
        ):
            result = run_model_hamiltonian_solver(
                hubbard_dimer_spec(),
                solver_name='block2_dmrg',
                outputs=['energy', 'energy_levels'],
            )

        self.assertIn('does not define a complete excited-state spectrum', result['energy_spectrum_unavailable_reason'])
        self.assertEqual(result['missing_requested_outputs'], ['energy_levels'])

    def test_model_hamiltonian_strong_correlation_diagnostics_are_requested(self):
        result = run_model_hamiltonian_solver(
            hubbard_dimer_spec(),
            solver_name='fci',
            outputs=['energy', 'strong_correlation_diagnostics'],
        )

        diagnostics = result['strong_correlation_diagnostics']
        self.assertEqual(diagnostics['kind'], 'strong_correlation_diagnostics')
        self.assertIn(diagnostics['level'], ('weak', 'moderate', 'strong'))
        self.assertIsInstance(diagnostics['score'], float)
        self.assertIn('method_recommendation', diagnostics)
        self.assertIn('u_over_t', [item['name'] for item in diagnostics['diagnostics']])
        interpretations = {
            item['name']: item.get('interpretation', '')
            for item in diagnostics['diagnostics']
        }
        self.assertIn('0/1 in each spin channel', interpretations['natural_orbital_occupations'])
        self.assertIn('Connected charge correlations measure correlated density fluctuations', interpretations['nearest_neighbor_charge_correlation'])
        self.assertNotIn('charge-ordering tendency', interpretations['nearest_neighbor_charge_correlation'])
        for key in ('gap', 'density', 'double_occupancy', 'spin_correlation', 'charge_correlation', 'natural_occupations'):
            self.assertTrue(diagnostics['internal_quantities'][key])
        self.assertFalse(diagnostics['internal_quantities']['mean_field_gap'])
        self.assertFalse(diagnostics['internal_quantities']['frontier_orbital_degeneracy'])
        self.assertFalse(diagnostics['internal_quantities']['max_double_excitation_amplitude'])
        self.assertIn('strong_correlation=', result['analysis_text'])
        self.assertNotIn('gap', result)
        self.assertNotIn('energy_levels', result)
        self.assertNotIn('density', result)
        self.assertNotIn('double_occupancy', result)
        self.assertNotIn('spin_correlation', result)
        self.assertNotIn('charge_correlation', result)

    def test_many_body_diagnostic_identifies_degenerate_ground_state_manifold(self):
        diagnostics = _compute_strong_correlation_diagnostics(
            hubbard_dimer_spec(),
            {
                'solver': 'fci',
                'converged': True,
                'spectrum': {'levels': [-2.0, -2.0, -1.8]},
            },
        )

        item = next(
            entry for entry in diagnostics['diagnostics']
            if entry['name'] == 'many_body_gap'
        )
        self.assertEqual(item['value']['classification'], 'numerically_degenerate')
        self.assertEqual(item['value']['ground_state_count_within_numerical_tolerance'], 2)
        self.assertEqual(item['value']['spectrum_scope'], 'complete_fixed_particle_spin_projection_sector')
        self.assertIn('single-root expectation value is not a unique description', item['interpretation'])
        self.assertTrue(diagnostics['internal_quantities']['low_energy_manifold'])

    def test_many_body_diagnostic_warns_when_targeted_root_manifold_may_continue(self):
        diagnostics = _compute_strong_correlation_diagnostics(
            hubbard_dimer_spec(),
            {
                'solver': 'block2_dmrg',
                'converged': True,
                'state_energies': [-2.0, -1.99],
            },
        )

        item = next(
            entry for entry in diagnostics['diagnostics']
            if entry['name'] == 'many_body_gap'
        )
        self.assertEqual(item['value']['classification'], 'near_degenerate')
        self.assertTrue(item['value']['manifold_may_extend_beyond_computed_roots'])
        self.assertIn('increase nroots', item['interpretation'])

    def test_model_solver_nonconvergence_is_strong_solver_stress(self):
        diagnostics = _compute_strong_correlation_diagnostics(
            hubbard_dimer_spec(),
            {'solver': 'ccsd', 'converged': False},
        )

        self.assertEqual(diagnostics['level'], 'strong')
        self.assertEqual(diagnostics['level_reason'], 'solver_nonconvergence')
        self.assertEqual(diagnostics['solver_stress_level'], 'strong')
        self.assertEqual(diagnostics['physics_level'], 'unknown')
        self.assertIsNone(diagnostics['physics_score'])

    def test_model_hamiltonian_mp2_strong_correlation_screening_quantities(self):
        result = run_model_hamiltonian_solver(
            hubbard_dimer_spec(),
            solver_name='mp2',
            outputs=['energy', 'strong_correlation_diagnostics'],
        )

        self.assertEqual(result['solver'], 'mp2')
        self.assertEqual(result['reference'], 'uhf')
        self.assertIsInstance(result['mean_field_gap'], float)
        self.assertIsInstance(result['mean_field_homo'], float)
        self.assertIsInstance(result['mean_field_lumo'], float)
        self.assertIsInstance(result['natural_occupations'], list)
        self.assertIsInstance(result['natural_occupation_fractionality'], float)
        self.assertIsInstance(result['max_double_excitation_amplitude'], float)
        diagnostics = result['strong_correlation_diagnostics']
        diagnostic_names = {item['name'] for item in diagnostics['diagnostics']}
        self.assertIn('mean_field_homo_lumo_gap', diagnostic_names)
        self.assertIn('frontier_orbital_degeneracy', diagnostic_names)
        self.assertIn('natural_orbital_occupations', diagnostic_names)
        self.assertIn('max_double_excitation_amplitude', diagnostic_names)
        self.assertTrue(diagnostics['internal_quantities']['mean_field_gap'])
        self.assertTrue(diagnostics['internal_quantities']['frontier_orbital_degeneracy'])
        self.assertTrue(diagnostics['internal_quantities']['natural_occupations'])
        self.assertTrue(diagnostics['internal_quantities']['max_double_excitation_amplitude'])
        self.assertFalse(diagnostics['internal_quantities']['gap'])
        self.assertFalse(diagnostics['internal_quantities']['density'])
        self.assertEqual(result['mean_field_homo_lumo']['frontier_degeneracy']['status'], 'available')

    def test_natural_occupation_fractionality_uses_average_not_single_maximum(self):
        summary = _natural_occupation_summary_from_dm1([
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])

        self.assertEqual(summary['max_fractionality'], 1.0)
        self.assertAlmostEqual(summary['average_fractionality'], 0.25)

        fci_like_summary = _natural_occupation_summary_from_dm1([
            [1.813012, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.186988],
        ])
        self.assertEqual(fci_like_summary['fractional_orbital_count'], 4)

    def test_periodic_half_filled_hubbard_uses_converged_broken_symmetry_reference(self):
        spec = hubbard_chain_spec(site_count=4, nalpha=2, nbeta=2, t=-1, u=4)
        spec['boundary'] = 'periodic'
        spec['bonds'].append({
            'id': 3,
            'source': 3,
            'target': 0,
            't': -1,
            'V': 0,
            'effective_t': -1,
            'effective_V': 0,
        })

        result = run_model_hamiltonian_solver(
            spec,
            solver_name='fci',
            outputs=['energy', 'strong_correlation_diagnostics'],
        )

        self.assertTrue(result['reference_converged'])
        self.assertLess(result['reference_energy'], 0.0)
        self.assertIn('antiferromagnetic', result['reference_initial_guess'])
        self.assertEqual(result['fractional_natural_orbital_count'], 4)

    def test_model_hamiltonian_solvers_support_open_shell(self):
        for solver_name in ('mp2', 'ccsd', 'fci'):
            with self.subTest(solver=solver_name):
                result = run_model_hamiltonian_solver(open_shell_hubbard_trimer_spec(), solver_name=solver_name, outputs=['energy', 'gap'])

                self.assertEqual(result['solver'], solver_name)
                self.assertEqual(result['nelec'], [2, 1])
                self.assertIsInstance(result['energy'], float)
                if solver_name == 'fci':
                    self.assertEqual(result['reference'], 'direct_spin1')
                    self.assertEqual(result['energy_spectrum_method'], 'fci')
                    self.assertIn('energy_levels', result)
                else:
                    self.assertEqual(result['reference'], 'uhf')
                    self.assertFalse(result['restricted_reference'])
                    self.assertNotIn('energy_levels', result)

    def test_model_hamiltonian_cisd_solver_is_not_supported(self):
        errors = validate_model_hamiltonian_spec(hubbard_dimer_spec(), solver_name='cisd')

        self.assertTrue(any('Unsupported Model Hamiltonian solver: cisd' in message for message in errors))
        with self.assertRaisesRegex(ValueError, 'Unsupported Model Hamiltonian solver: cisd'):
            run_model_hamiltonian_solver(hubbard_dimer_spec(), solver_name='cisd')

    def test_holstein_hubbard_is_blocked_for_now(self):
        spec = hubbard_dimer_spec()
        spec['model'] = 'holstein_hubbard'
        spec['phonons'] = {
            'enabled': True,
            'cutoff': 4,
            'local_modes': [
                {'site': 0, 'omega': 1, 'g': 0.5},
                {'site': 1, 'omega': 1, 'g': 0.5},
            ],
        }

        errors = validate_model_hamiltonian_spec(spec)

        self.assertTrue(any('Holstein-Hubbard' in message for message in errors))
        self.assertTrue(any('design_only' in message for message in errors))
        self.assertTrue(any('phonon Hilbert spaces' in message for message in errors))


if __name__ == '__main__':
    unittest.main()
