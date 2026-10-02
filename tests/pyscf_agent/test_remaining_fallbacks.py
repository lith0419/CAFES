"""Executable regressions for the second scientific-boundary audit."""
from __future__ import annotations

import io
import itertools
import shutil
import subprocess
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from computational_study_agent.adaptive.executor import analyze_initial_scan_report
from computational_study_agent.adaptive.initial_scan import normalize_adaptive_options
from computational_study_agent.adaptive.refinement import _runtime_recovery_request
from computational_study_agent.adaptive.state_tracking import _assign_roots
from computational_study_agent.costing import estimate_case_cost
from computational_study_agent.datasets.hamiltonian.contracts import ElectronicStructureSpec
from pyscf_agent.application.calculation_service import CalculationApplicationService
from pyscf_agent.backend.correlation.molecular import _mean_field_occupation_summary, build_molecular_correlation_risk
from pyscf_agent.backend.correlation.orbital_diagnostics import t2_orbital_importance
from pyscf_agent.backend.execution import _run_pyscf_task
from pyscf_agent.backend.model_hamiltonian.reference_diagnostics import _natural_occupation_summary_from_dm1
from pyscf_agent.backend.model_hamiltonian.solver import _model_hamiltonian_reference, generate_model_hamiltonian_input_script
from pyscf_agent.backend.one_particle_state import capture_reference_1rdm
from pyscf_agent.backend.parsing import task_spec_from_partial
from pyscf_agent.contracts import RuntimeSpec, TaskSpec
from pyscf_agent.providers.block2.config import normalize_block2_options, normalized_block2_provider_options
from pyscf_agent.providers.block2.driver import _single_orbital_entropy_from_spin_free_rdms, _entanglement_summary
from pyscf_agent.providers.fcdmft.contracts import normalize_hf_dmft_options, normalize_periodic_gw_options
from pyscf_agent.request_builder.prepare import _apply_safe_defaults, build_prepared_request
from tests.computational_study_agent.support import hubbard_dimer_spec


class RemainingInputBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.base = {'atom': 'H 0 0 0; H 0 0 0.74', 'basis': 'sto-3g', 'method': 'hf'}

    def test_default_words_and_old_summaries_cannot_authorize_new_values(self):
        for text in ('Do not use default settings.', 'Do not generate the structure.', 'generate everything'):
            with self.subTest(text=text):
                result = _apply_safe_defaults(
                    {'request_summary': 'Generate standard water geometry'}, {},
                    [{'role': 'user', 'content': text}], text,
                    {'request_summary': 'Use default water geometry'}, 'en')
                self.assertNotIn('basis', result['structured_request'])
                self.assertNotIn('atom', result['structured_request'])

    def test_normalize_spin_before_inferring_reference(self):
        result = _apply_safe_defaults({}, {'spin': '0'}, [], None, {}, 'en')
        self.assertTrue(result['structured_request']['restricted'])
        self.assertEqual(result['structured_request']['spin'], 0)

    def test_uninterpreted_revision_is_not_ready(self):
        with patch('pyscf_agent.request_builder.prepare.llm_request_builder_is_configured', return_value=False):
            result = build_prepared_request([{'role': 'user', 'content': 'Change method to MP2.'}], task_spec=self.base)
            seed = build_prepared_request([], task_spec=self.base)
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['structured_request']['method'], 'hf')
        self.assertEqual(result['execution_request_text'], '')
        self.assertEqual(seed['status'], 'ready')

    def test_df_aliases_reach_the_same_task_contract(self):
        for alias, value in {'df': True, 'density_fit': 'true', 'auxbasis': 'weigend',
                             'df_auxbasis': 'weigend', 'density_fitting_auxbasis': 'weigend'}.items():
            with self.subTest(alias=alias):
                spec = task_spec_from_partial({**self.base, alias: value})
                self.assertTrue(spec.density_fitting.enabled)
                if value == 'weigend':
                    self.assertEqual(spec.density_fitting.auxbasis, value)

    def test_manual_orbital_indices_are_not_truncated(self):
        request = {**self.base, 'method': 'casci', 'solver': 'block2_dmrg',
                   'active_space': {'ncas': 2, 'nelecas': 2, 'approved': True},
                   'orbital_processing': {'orbital_ordering': 'manual', 'orbital_order': [0.4, 1.7]}}
        self.assertFalse(CalculationApplicationService().validate_task_spec(request)['valid'])
        request['orbital_processing']['orbital_order'] = ['1', '0']
        self.assertTrue(CalculationApplicationService().validate_task_spec(request)['valid'])

    def test_adaptive_options_reject_invalid_explicit_values(self):
        for options in ({'active_space_solver': 'blcok2'}, {'max_cas_orbitals': 'invalid'},
                        {'max_cas_orbitals': 3.9}, {'max_fci_sites': 0},
                        {'method_policy': {'typo': 'fci'}}, {'method_policy': []},
                        {'block2_dmrg_options': 'invalid'}, {'unknown_option': True},
                        {'model_solver_options': []},
                        {'active_space_orbital_processing': {'localization_method': 'invalid'}},
                        {'active_space_orbital_processing': {'orbital_ordering': 'invalid'}},
                        {'active_space_orbital_processing': {'orbital_order': [0.4, 1.7]}}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                normalize_adaptive_options(options)
        options = normalize_adaptive_options({'max_cas_orbitals': '20',
            'active_space_orbital_processing': {'use_natural_orbitals': 'true'}})
        self.assertEqual(options['max_cas_orbitals'], 20)
        self.assertTrue(options['active_space_orbital_processing']['use_natural_orbitals'])

    def test_adaptive_normalization_is_idempotent_and_keeps_explicit_restart(self):
        request = {'block2_dmrg_options': {'preset': 'screening', 'stack_memory_gb': '2',
            'orbital_restart_manifest': {'kind': 'block2_orbital_state', 'path': '/tmp/orbitals.json'},
            'orbital_restart_provenance': {'selection': 'approved_orbitals'}},
            'active_space_orbital_processing': {}}
        once = normalize_adaptive_options(request)
        self.assertEqual(once, normalize_adaptive_options(once))
        options = once['block2_dmrg_options']
        self.assertEqual(options['stack_memory_bytes'], 2 * 1024**3)
        self.assertNotIn('sweeps', options)
        self.assertEqual(options['orbital_restart_provenance'], request['block2_dmrg_options']['orbital_restart_provenance'])

    def test_provider_values_cannot_disappear_or_flip_boolean_meaning(self):
        for options in ({'orbital_order': 'invalid'}, {'seed': -1}, {'iprint': -1},
                        {'nroots': 1, 'state_average_weights': 'invalid'},
                        {'nroots': 1, 'state_average_weights': [True]}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                normalize_block2_options(options)
        for normalize in (normalized_block2_provider_options, normalize_periodic_gw_options, normalize_hf_dmft_options):
            with self.subTest(normalizer=normalize.__name__), self.assertRaises(ValueError):
                normalize('invalid')
        self.assertFalse(normalize_periodic_gw_options({'gw_finite_size_correction': 'false'})['gw_finite_size_correction'])
        with self.assertRaises(ValueError):
            normalize_periodic_gw_options({'gw_broadening': True})
        self.assertFalse(ElectronicStructureSpec.from_dict({'restricted': 'false'}).restricted)

    def test_recovery_changes_effective_runtime_and_preserves_other_controls(self):
        request = {**self.base, 'max_cycle': 800, 'conv_tol': 1e-11, 'runtime': {'max_cycle': 50}}
        recovered = _runtime_recovery_request(request)
        self.assertNotIn('max_cycle', recovered)
        self.assertNotIn('conv_tol', recovered)
        spec = task_spec_from_partial(recovered)
        self.assertEqual(spec.runtime.max_cycle, 1600)
        self.assertEqual(spec.runtime.conv_tol, 1e-11)
        self.assertEqual(request['max_cycle'], 800)
        with self.assertRaises(ValueError):
            _runtime_recovery_request({**self.base, 'max_cycle': 'invalid'})

    @unittest.skipUnless(shutil.which('node'), 'Node.js is needed for browser control regressions')
    def test_browser_keeps_invalid_numbers_visible(self):
        asset = Path(__file__).resolve().parents[2] / 'computational_study_agent/web_assets/planner-spec.js'
        dataset = asset.with_name('planner-dataset.js')
        script = r'''
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const controls = {};
global.document = {getElementById: id => controls[id] || {value: ''}};
vm.runInThisContext(fs.readFileSync(process.argv[1], 'utf8'));
vm.runInThisContext(fs.readFileSync(process.argv[2], 'utf8'));
for (const value of ['bad', '-1', '3.9']) {
  controls.cycles = {value};
  assert.throws(() => plannerPositiveNumber('cycles', 50, true));
}
controls.cycles = {value: ''};
assert.equal(plannerPositiveNumber('cycles', 50, true), 50);
controls['planner-dmet-impurity-shape'] = {value: '2.8x2bad'};
assert.throws(() => plannerDmetShapeFromInput());
controls['planner-dmet-impurity-shape'].value = '2x2';
assert.deepEqual(plannerDmetShapeFromInput(), [2,2]);
controls['dataset-id'] = {value: 'test'};
controls['dataset-name'] = {value: 'Test'};
for (const value of ['1.5', '2bad', '']) {
  controls['dataset-split-seed'] = {value};
  assert.throws(() => hamiltonianDatasetSpecFromControls(), /Split Seed/);
}
'''
        result = subprocess.run(['node', '-e', script, str(asset), str(dataset)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class RemainingEvidenceBoundaryTests(unittest.TestCase):
    def test_unknown_diagnostics_and_routing_levels_are_blocked(self):
        for diagnostic in ({'level': 'unknown'}, {'level': 'unavailable'}, {'level': 'nonsense'},
                           {'level': 'weak', 'routing_level': 'unknown'}):
            diagnostic = {'kind': 'molecular_correlation_diagnostics', **diagnostic}
            report = {'comparison_table': [{'case_id': 'a', 'status': 'succeeded'}],
                      'cases': [{'case_id': 'a', 'task_report': {'structured_results': {
                          'task_type': 'molecular', 'strong_correlation_diagnostics': diagnostic}}}]}
            with self.subTest(diagnostic=diagnostic):
                decision = analyze_initial_scan_report(report)[0]
                self.assertIsNone(decision['recommended_solver'])
                self.assertEqual(decision['level'], 'initial_scan_blocked')
                self.assertEqual(decision['diagnostics'], diagnostic)

    def test_unavailable_uno_is_not_replaced_by_mo_index_sums(self):
        mf = SimpleNamespace(mo_occ=np.array([[1, 0], [1, 0]]))
        with patch('pyscf_agent.backend.correlation.molecular._spin_summed_uno_occupations',
                   return_value={'status': 'unavailable', 'occupations': [], 'reason': 'overlap unavailable'}):
            summary = _mean_field_occupation_summary(mf, SimpleNamespace(spin=0))
        self.assertEqual(summary['status'], 'unavailable')
        self.assertEqual(summary['occupations'], [])
        risk = build_molecular_correlation_risk(mf, SimpleNamespace(spin=0), fractional_occupations=None)
        component = next(item for item in risk['physics_components'] if item['name'] == 'mean_field_fractional_occupations')
        self.assertEqual(component['status'], 'unavailable')
        self.assertIsNone(component['score'])

    def test_nonphysical_natural_occupations_keep_raw_evidence(self):
        summary = _natural_occupation_summary_from_dm1(np.diag([2.4, -0.4]))
        self.assertEqual(summary['status'], 'out_of_bounds')
        np.testing.assert_allclose(summary['occupations'], [2.4, -0.4])
        self.assertIsNone(summary['max_fractionality'])
        rounded = _natural_occupation_summary_from_dm1(np.diag([2 + 1e-9, -1e-9]))
        self.assertEqual(rounded['status'], 'available')
        self.assertGreater(rounded['occupations'][0], 2)

    def test_root_assignment_has_exact_ambiguity_margin_at_every_size(self):
        for count in (2, 9, 20):
            mapping, score, margin = _assign_roots(np.ones((count, count)))
            self.assertEqual(sorted(mapping), list(range(count)))
            self.assertEqual(score, 1.0)
            self.assertEqual(margin, 0.0)
        scores = [[1.0, 0.9, 0], [0.95, 0, 0], [0, 0.1, 1]]
        mapping, score, margin = _assign_roots(scores)
        totals = sorted((sum(scores[i][p[i]] for i in range(3)) for p in itertools.permutations(range(3))), reverse=True)
        self.assertEqual(mapping, [1, 0, 2])
        self.assertAlmostEqual(score, totals[0] / 3)
        self.assertAlmostEqual(margin, (totals[0] - totals[1]) / 3)

    def test_spin_free_entropy_requires_applicable_spin_evidence(self):
        rdm1, rdm2 = np.ones((1, 1)), np.zeros((1, 1, 1, 1))
        with self.assertRaisesRegex(ValueError, 'singlet'):
            _single_orbital_entropy_from_spin_free_rdms(rdm1, rdm2)
        entropy = _single_orbital_entropy_from_spin_free_rdms(rdm1, rdm2, spin_symmetric=True)
        self.assertAlmostEqual(entropy[0], np.log(2))
        driver = SimpleNamespace(get_orbital_entropies=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('NPDM unavailable')))
        config = normalize_block2_options({'compute_entanglement': True})
        payload, arrays = _entanglement_summary(driver, None, config, rdm1=rdm1, rdm2=rdm2)
        self.assertEqual(payload['availability']['single_orbital_entropy'], 'unavailable')
        self.assertNotIn('single_orbital_entropy', arrays)
        self.assertIn('singlet', payload['errors'][-1]['message'])

    def test_optional_density_and_amplitude_failures_keep_reasons(self):
        mf = SimpleNamespace(make_rdm1=lambda: (_ for _ in ()).throw(RuntimeError('density failed')))
        evidence = capture_reference_1rdm(TaskSpec(), mf)
        self.assertEqual(evidence['status'], 'failed')
        self.assertIn('density failed', evidence['reason'])
        mf = SimpleNamespace(mo_occ=[2, 0])
        evidence = t2_orbital_importance(SimpleNamespace(t2=np.ones(3)), mf)
        self.assertEqual(evidence['status'], 'failed')
        self.assertIn('rank-four', evidence['reason'])
        self.assertEqual(t2_orbital_importance(None, mf)['status'], 'unavailable')

    def test_cas_natural_occupation_failure_preserves_completed_energy(self):
        from pyscf import gto, scf, lib
        from pyscf_agent.backend.correlation.cas_execution import run_cas_method
        from pyscf_agent.contracts import ActiveSpaceSpec

        previous_threads = lib.num_threads()
        try:
            lib.num_threads(1)
            mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
            mf = scf.RHF(mol).run()
            active = ActiveSpaceSpec(enabled=True, ncas=2, nelecas=2, approved=True)
            with patch('pyscf.mcscf.casci.CASCI.cas_natorb', side_effect=RuntimeError('occupation diagonalization failed')):
                result = run_cas_method(mf, mol, 'casci', active, 50)
            self.assertTrue(result['converged'])
            self.assertLess(result['energy'], -1.0)
            self.assertIsNone(result['natural_occupations'])
            self.assertEqual(result['natural_occupation_evidence']['status'], 'failed')
            self.assertIn('diagonalization failed', result['natural_occupation_evidence']['reason'])
            with patch('pyscf.mcscf.casci.CASCI.cas_natorb', return_value=(None, None, None)):
                unavailable = run_cas_method(mf, mol, 'casci', active, 50)
            self.assertEqual(unavailable['natural_occupation_evidence']['status'], 'unavailable')
            self.assertAlmostEqual(unavailable['energy'], result['energy'])
        finally:
            lib.num_threads(previous_threads)

    def test_costing_uses_provider_schedule_and_discloses_scope(self):
        for preset in ('screening', 'balanced', 'high_accuracy'):
            case = {'request': {'method': 'casci', 'solver': {'name': 'block2_dmrg', 'options': {'preset': preset}},
                                'active_space': {'ncas': 4, 'nelecas': 4}}}
            cost = estimate_case_cost(case, 'molecular')
            config = normalize_block2_options({'preset': preset})
            self.assertEqual(cost['sweeps'], config.sweeps)
            self.assertEqual(cost['bond_dimension'], max(config.bond_dimensions))
            self.assertEqual(cost['estimate_scope'], 'initial_dmrg_schedule_only')


class ModelRuntimeBoundaryTests(unittest.TestCase):
    def test_runtime_reaches_dispatch_and_generated_input(self):
        spec = TaskSpec(task_type='model_hamiltonian', runtime=RuntimeSpec(max_cycle=812, conv_tol=1e-12, verbose=0))
        spec.model_hamiltonian.spec = hubbard_dimer_spec()
        with patch('pyscf_agent.backend.execution.run_model_hamiltonian_solver', return_value={}) as run:
            _run_pyscf_task(spec)
        self.assertEqual(run.call_args.kwargs['runtime'].max_cycle, 812)
        script = generate_model_hamiltonian_input_script(spec.model_hamiltonian.spec, runtime=spec.runtime)
        with patch('pyscf_agent.backend.model_hamiltonian.solver.run_model_hamiltonian_solver', return_value={}) as run, redirect_stdout(io.StringIO()):
            exec(script, {})
        self.assertEqual(run.call_args.kwargs['runtime']['max_cycle'], 812)
        self.assertEqual(run.call_args.kwargs['runtime']['conv_tol'], 1e-12)

    def test_model_reference_records_failed_attempts_and_uses_runtime(self):
        objects = []
        class Reference:
            def __init__(self, mol):
                self.converged = True
                self.e_tot = -1.0
                objects.append(self)
            def kernel(self, **kwargs):
                if len(objects) == 1:
                    raise RuntimeError('default guess failed')
        runtime = RuntimeSpec(max_cycle=812, conv_tol=1e-12, conv_tol_grad=1e-6, diis_space=10, verbose=0)
        with patch('pyscf.scf.UHF', Reference):
            mf, _ = _model_hamiltonian_reference(np.eye(2), np.zeros((2, 2, 2, 2)),
                {'norb': 2, 'nelec': [1, 1]}, restricted=False, runtime=runtime)
        self.assertEqual(len(objects), 5)
        self.assertTrue(all(item.max_cycle == 812 and item.conv_tol == 1e-12 and item.diis_space == 10 for item in objects))
        attempts = mf.pyscf_agent_reference_candidates
        self.assertEqual(len(attempts), 5)
        self.assertEqual(attempts[0]['status'], 'failed')
        self.assertIn('default guess failed', attempts[0]['error'])
