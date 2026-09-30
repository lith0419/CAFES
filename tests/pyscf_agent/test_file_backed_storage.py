from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pyscf_agent.artifacts import ArtifactRepository
from pyscf_agent.artifacts.arrays import read_array, write_array
from pyscf_agent.artifacts.task_storage import compact_task_payload
from pyscf_agent.backend.state import default_state
from pyscf_agent.backend.correlation.cas_execution import _initial_mo_coeff
from pyscf_agent.contracts import ActiveSpaceSpec
from computational_study_agent.study_state import checkpoint_payload, load_checkpoint
from computational_study_agent.postprocessing import PostprocessContext


class FileBackedStorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_lutein_sized_matrix_is_stored_once_across_report_copies(self):
        matrix = np.arange(1294 * 1294, dtype=float).reshape(1294, 1294)
        values = matrix.tolist()
        payload = {'task_spec': {'active_space': {'initial_mo_coeff': values}},
                   'approval': {'initial_mo_coeff': values},
                   'messages': [{'content': json.dumps({'active_space': {'initial_mo_coeff': values}})}]}
        report = compact_task_payload(payload, self.root)
        reference = report['task_spec']['active_space']['initial_mo_coeff']
        self.assertLess(len(json.dumps(report)), 10000)
        self.assertEqual(len(list((self.root / 'arrays').glob('*.npy'))), 1)
        self.assertEqual(report['approval']['initial_mo_coeff'], reference)
        self.assertIsInstance(read_array(reference), np.memmap)
        np.testing.assert_array_equal(read_array(reference), matrix)
        self.assertIs(payload['approval']['initial_mo_coeff'], values)

    def test_request_is_compact_before_workflow_history_and_cas_loads_reference(self):
        matrix = np.eye(80)
        state = default_state(json.dumps({'active_space': {'initial_mo_coeff': matrix.tolist()}}),
                              work_dir=str(self.root), run_id='case')
        self.assertLess(len(state['messages'][0]['content']), 1500)
        reference = json.loads(state['user_request'])['active_space']['initial_mo_coeff']
        loaded = _initial_mo_coeff(ActiveSpaceSpec(initial_mo_coeff=reference), False)
        np.testing.assert_array_equal(loaded, matrix)
        np.testing.assert_array_equal(_initial_mo_coeff(ActiveSpaceSpec(initial_mo_coeff=[[1.0]]), False), [[1.]])

    def test_file_reference_cannot_silently_change_shape_or_hide_missing_data(self):
        reference = write_array(np.eye(3), self.root)
        with self.assertRaisesRegex(ValueError, 'shape or dtype'):
            read_array({**reference, 'shape': [2, 2]})
        Path(reference['path']).unlink()
        with self.assertRaises(FileNotFoundError):
            read_array(reference)

    def test_storage_leaves_invalid_guesses_for_the_numerical_validation_boundary(self):
        malformed = [[1.0], [1.0, 2.0]]
        state = default_state(json.dumps({'active_space': {'initial_mo_coeff': malformed}}),
                              work_dir=str(self.root), run_id='invalid')
        self.assertEqual(json.loads(state['user_request'])['active_space']['initial_mo_coeff'], malformed)
        with self.assertRaisesRegex(ValueError, 'numeric two-dimensional'):
            _initial_mo_coeff(ActiveSpaceSpec(initial_mo_coeff=malformed), False)

    def test_repeated_report_saves_keep_the_same_text_preview_and_artifacts(self):
        original = 'solver output\n' * 20000
        report = compact_task_payload({'raw_stdout': original}, self.root)
        self.assertLessEqual(len(report['raw_stdout']), 12000)
        self.assertEqual(Path(report['artifacts'][0]['path']).read_text(), original)
        self.assertEqual(compact_task_payload(report, self.root), report)
        self.assertEqual(len(list((self.root / 'details').glob('*.txt'))), 1)

    def test_checkpoint_status_needs_no_task_reports_or_arrays(self):
        path = self.root / 'study-state.json'
        state = {'schema': 'pyscf-agent.study-state.v1', 'study_id': 'scan', 'cases': {}}
        for i in range(23):
            state['cases'][str(i)] = {'attempt_count': 1,
                'execution': {'pending': False, 'run_id': str(i), 'attempt_count': 1},
                'task_report': {'run_id': str(i), 'execution_status': 'succeeded',
                                'structured_results': {'final_energy': -1. - i}}}
        saved = checkpoint_payload(path, state)
        ArtifactRepository().write_json(path, saved, kind='study-state')
        self.assertLess(path.stat().st_size, 30000)
        self.assertNotIn('task_report', saved['cases']['0'])
        from computational_study_agent.study_state import read_json
        with patch('computational_study_agent.study_state.read_json', wraps=read_json) as reads:
            status = load_checkpoint(path, 'scan', include_reports=False)
            self.assertEqual(reads.call_count, 1)
            self.assertEqual(status['cases']['0']['task_summary']['execution_status'], 'succeeded')
        loaded = load_checkpoint(path, 'scan')
        self.assertEqual(loaded['cases']['3']['task_report']['structured_results']['final_energy'], -4.)
        # A broken report must fail collection, while status remains readable.
        Path(saved['cases']['0']['task_report_ref']['path']).unlink()
        load_checkpoint(path, 'scan', include_reports=False)
        with self.assertRaisesRegex(ValueError, 'Cannot read Study execution record'):
            load_checkpoint(path, 'scan')

    def test_legacy_inline_checkpoint_remains_readable(self):
        path = self.root / 'study-state.json'
        payload = {'schema': 'pyscf-agent.study-state.v1', 'study_id': 'old', 'cases': {
            'a': {'attempt_count': 1, 'task_report': {'execution_status': 'succeeded'}}}}
        path.write_text(json.dumps(payload))
        self.assertEqual(load_checkpoint(path, 'old')['cases']['a']['task_report']['execution_status'], 'succeeded')

    def test_study_resume_and_selected_retry_preserve_file_backed_inputs(self):
        from computational_study_agent.application import StudyApplicationService
        from tests.computational_study_agent.test_retry_collection import Remote, plan_for_service
        plan = plan_for_service()
        for case in plan.cases:
            case.request['active_space'] = {'initial_mo_coeff': np.eye(80).tolist()}
        executor = Remote(fail_first=False)
        service = StudyApplicationService(task_executor=executor)
        first = service.run_study(plan, work_dir=str(self.root))
        first_reference = first.cases[0]['request']['active_space']['initial_mo_coeff']
        self.assertEqual(first_reference['schema'], 'pyscf-agent.numeric-array.v1')
        checkpoint = json.loads((self.root / plan.study_id / 'study-state.json').read_text())
        self.assertTrue(all('task_report' not in case and 'task_report_ref' in case
                            for case in checkpoint['cases'].values()))
        status = service.inspect_execution(plan.study_id, work_dir=str(self.root))
        self.assertEqual(status['task_status_counts'], {'succeeded': 2})
        service.run_study(plan, work_dir=str(self.root))
        self.assertEqual(executor.submit_count, 1)
        collected = service.collect_study(plan, work_dir=str(self.root))
        self.assertEqual(collected.status, 'succeeded')
        retried = service.run_study(plan, work_dir=str(self.root), rerun_case_ids=['case-0001'])
        self.assertEqual(executor.submit_count, 2)
        self.assertEqual([task.task_id for task in executor.submissions[-1]], ['case-0001'])
        self.assertEqual(retried.cases[1]['task_report'], first.cases[1]['task_report'])
        self.assertEqual(retried.cases[0]['request']['active_space']['initial_mo_coeff'], first_reference)

    def test_failed_streaming_write_keeps_the_previous_json(self):
        class Invalid:
            def __str__(self):
                raise ValueError('serialization interrupted')
        repository = ArtifactRepository()
        path = self.root / 'state.json'
        repository.write_json(path, {'version': 1}, kind='study-state')
        before = path.read_bytes()
        with self.assertRaisesRegex(TypeError, 'Unsupported JSON value'):
            repository.write_json(path, {'data': [1, 2, Invalid()]}, kind='study-state')
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.root.glob('.*.tmp')), [])

    def test_plot_context_never_copies_numerical_payload(self):
        class NoCopy:
            def __deepcopy__(self, memo):
                raise AssertionError('Numerical payload must not be copied for a plot')
        context = PostprocessContext.from_report({
            'system_type': 'molecular',
            'comparison_table': [{'case_id': 'a', 'status': 'succeeded', 'energy': -1., 'bond_length': .74}],
            'cases': [{'case_id': 'a', 'variables': {'bond_length': .74}, 'task_report': {
                'execution_status': 'succeeded', 'task_spec': {'system': {'unit': 'Angstrom'}},
                'structured_results': {'matrix': NoCopy()}, 'raw_stdout': NoCopy()}}],
        })
        self.assertEqual(context.cases[0]['task_report']['task_spec']['system']['unit'], 'Angstrom')
        self.assertNotIn('structured_results', context.cases[0]['task_report'])

    def test_remote_numerical_attachment_roundtrip_exceeds_old_text_limit(self):
        from pyscf_agent.executors.ssh_slurm import _request_attachments
        from pyscf_agent.remote.rpc_cli import _materialize_attachments
        reference = write_array(np.eye(900), self.root / 'local')
        self.assertGreater(reference['size_bytes'], 5 * 1024 * 1024)
        request = {'active_space': {'initial_mo_coeff': reference}}
        attachments = _request_attachments(request)
        remote = _materialize_attachments(request, attachments, self.root / 'remote')
        restored = remote['active_space']['initial_mo_coeff']
        self.assertNotEqual(reference['path'], restored['path'])
        np.testing.assert_array_equal(read_array(restored), read_array(reference))

    def test_real_casci_energy_is_identical_with_file_backed_orbitals(self):
        from pyscf import gto, scf, mcscf, lib
        lib.num_threads(1)
        mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        reference = write_array(mf.mo_coeff, self.root)
        initial = _initial_mo_coeff(ActiveSpaceSpec(initial_mo_coeff=reference), False)
        expected = mcscf.CASCI(mf, 2, 2).kernel(mf.mo_coeff)[0]
        actual = mcscf.CASCI(mf, 2, 2).kernel(initial)[0]
        self.assertAlmostEqual(expected, actual, places=12)

    def test_complete_df_casscf_workflow_accepts_file_backed_guess(self):
        from pyscf import gto, scf
        from pyscf_agent.application.calculation_service import CalculationApplicationService
        from pyscf_agent.executors.local import LocalExecutor
        mol = gto.M(atom='H 0 0 0; H 0 0 0.9', basis='6-31g', verbose=0)
        mf = scf.RHF(mol).run()
        reference = write_array(mf.mo_coeff, self.root / 'input')
        service = CalculationApplicationService(task_executor=LocalExecutor())
        reports = []
        for name, guess in [('inline', mf.mo_coeff.tolist()), ('file', reference)]:
            request = {'atom': mol.atom, 'basis': '6-31g', 'method': 'casscf',
                       'active_space': {'enabled': True, 'selection_method': 'manual',
                                        'ncas': 2, 'nelecas': 2, 'approved': True,
                                        'initial_mo_coeff': guess},
                       'density_fitting': {'enabled': True, 'apply_to': 'scf_and_casscf'},
                       'runtime': {'max_cycle': 50, 'verbose': 0}, 'outputs': ['energy']}
            report = service.execute_request(json.dumps(request), work_dir=str(self.root),
                                             run_id=name, include_llm_feedback=False)
            self.assertEqual(report['execution_status'], 'succeeded', report.get('errors'))
            reports.append(report)
        self.assertAlmostEqual(reports[0]['structured_results']['final_energy'],
                               reports[1]['structured_results']['final_energy'], places=12)


if __name__ == '__main__':
    unittest.main()
