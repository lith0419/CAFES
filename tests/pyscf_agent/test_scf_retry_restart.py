from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from pyscf import gto, scf
from pyscf.scf import hf

from pyscf_agent.application.calculation_service import CalculationApplicationService
from pyscf_agent.backend import execution, state
from pyscf_agent.backend.one_particle_state import capture_reference_1rdm
from pyscf_agent.contracts import TaskSpec, task_spec_to_dict
from pyscf_agent.executors.local import LocalExecutor


class SCFRetryRestartTests(unittest.TestCase):
    def test_public_df_scf_retry_preserves_algorithm_and_passes_last_density(self):
        """Both spin policies resume the unfinished SCF, not the atomic guess."""
        native_kernel = hf.kernel
        for restricted in (True, False):
            with self.subTest(restricted=restricted), tempfile.TemporaryDirectory() as tmpdir:
                seen = []

                def observe_kernel(mf, *args, **kwargs):
                    dm0 = kwargs.get('dm0')
                    seen.append(None if dm0 is None else np.asarray(dm0).copy())
                    return native_kernel(mf, *args, **kwargs)

                request = {
                    'system': {
                        'atom': 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                        'basis': 'sto-3g',
                    },
                    'method': {'name': 'hf', 'restricted': restricted},
                    'runtime': {'max_cycle': 3, 'conv_tol': 1e-9},
                    'density_fitting': {'enabled': True},
                    'analysis': {'outputs': ['energy']},
                }
                service = CalculationApplicationService(task_executor=LocalExecutor())
                with mock.patch.object(hf, 'kernel', observe_kernel), mock.patch.object(
                    hf.SCF, 'newton', side_effect=AssertionError('Automatic retry must preserve standard SCF'),
                ):
                    report = service.execute_request(
                        json.dumps(request), work_dir=tmpdir, run_id='scf-retry',
                        include_llm_feedback=False,
                    )

                self.assertEqual(report['execution_status'], 'succeeded')
                self.assertEqual(report['retry_count'], 1)
                self.assertEqual(len(seen), 2)
                self.assertIsNone(seen[0])
                results = report['structured_results']
                self.assertEqual(results['scf_algorithm'], 'standard')
                self.assertTrue(results['reference_converged'])
                initial = results['initial_state']
                self.assertEqual(initial['status'], 'applied')
                self.assertFalse(initial['source_scf_converged'])
                source_path = Path(initial['source_artifact']['path'])
                self.assertEqual(source_path.name, 'result-one-particle-state.npz')
                with np.load(source_path, allow_pickle=False) as archive:
                    metadata = json.loads(archive['metadata_json'].item())
                    self.assertFalse(metadata['scf_converged'])
                    expected = archive['dm_restricted'] if restricted else np.asarray([
                        archive['dm_alpha'], archive['dm_beta'],
                    ])
                    np.testing.assert_allclose(seen[1], expected, atol=1e-10, rtol=1e-10)
                final_ref = results['one_particle_state']['data_artifact']
                self.assertEqual(Path(final_ref['path']).name, 'result-one-particle-state-retry-1.npz')
                # The retry request and its evidence survive serialization.
                serialized = json.loads(json.dumps(report))
                self.assertEqual(
                    serialized['task_spec']['initial_state']['source_artifact']['path'],
                    str(source_path),
                )

    def _unconverged_state(self, spec, reference_artifact=None):
        payload = state.default_state('{}', channel='test')
        payload.update({
            'task_spec': task_spec_to_dict(spec),
            'execution_status': 'unconverged',
            'retry_count': 1,
            'max_retries': 2,
            'structured_results': {
                'reference_converged': False,
                'one_particle_state': {'data_artifact': reference_artifact or {}},
            },
        })
        return payload

    def test_retry_uses_latest_attempt_evidence_and_clears_external_case_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old = Path(tmpdir) / 'old.npz'
            current = Path(tmpdir) / 'current.npz'
            old.touch()
            current.touch()
            spec = TaskSpec()
            spec.initial_state.mode = 'projected_1rdm'
            spec.initial_state.source_case_id = 'previous-study-case'
            spec.initial_state.source_artifact = {'kind': 'one_particle_state', 'path': str(old)}
            current_ref = {'kind': 'one_particle_state', 'path': str(current)}
            payload = self._unconverged_state(spec, current_ref)
            payload['artifacts'] = [current_ref, spec.initial_state.source_artifact]
            recovered = execution.repair_or_retry(json.loads(json.dumps(payload)))
            initial = recovered['task_spec']['initial_state']
            self.assertEqual(initial['source_artifact'], current_ref)
            self.assertIsNone(initial['source_case_id'])

    def test_missing_latest_density_does_not_pick_a_stale_attempt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old = Path(tmpdir) / 'old.npz'
            old.touch()
            payload = self._unconverged_state(
                TaskSpec(), {'kind': 'one_particle_state', 'path': str(Path(tmpdir) / 'missing.npz')},
            )
            payload['artifacts'] = [{'kind': 'one_particle_state', 'path': str(old)}]
            recovered = execution.repair_or_retry(payload)
            self.assertEqual(recovered['task_spec']['runtime']['scf_algorithm'], 'standard')
            self.assertEqual(recovered['task_spec']['initial_state']['mode'], 'none')

    def test_explicit_newton_choice_is_preserved_on_retry(self):
        spec = TaskSpec()
        spec.runtime.scf_algorithm = 'newton'
        recovered = execution.repair_or_retry(self._unconverged_state(spec))
        self.assertEqual(recovered['task_spec']['runtime']['scf_algorithm'], 'newton')
        self.assertEqual(recovered['task_spec']['runtime']['max_cycle'], 2 * spec.runtime.max_cycle)

    def test_retry_preserves_reference_policy_and_compatible_density(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / 'restricted.npz'
            source.touch()
            ref = {'kind': 'one_particle_state', 'path': str(source)}
            spec = TaskSpec()
            spec.system.spin = 1
            spec.method.restricted = True
            spec.initial_state.mode = 'projected_1rdm'
            spec.initial_state.source_artifact = ref
            recovered = execution.repair_or_retry(self._unconverged_state(spec, ref))
            self.assertTrue(recovered['task_spec']['method']['restricted'])
            self.assertEqual(recovered['task_spec']['initial_state']['mode'], 'projected_1rdm')


    def test_invalid_unconverged_density_is_not_saved_as_restart_evidence(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol)
        for density in (np.full((2, 2), np.nan), np.zeros((1, 1))):
            with self.subTest(shape=density.shape), mock.patch.object(mf, 'make_rdm1', return_value=density):
                evidence = capture_reference_1rdm(TaskSpec(), mf)
                self.assertEqual(evidence['status'], 'failed')
                self.assertNotIn('arrays', evidence)
                self.assertIn('incompatible dimensions or nonfinite', evidence['reason'])


if __name__ == '__main__':
    unittest.main()
