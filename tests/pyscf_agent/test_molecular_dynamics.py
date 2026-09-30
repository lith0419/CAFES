from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pyscf import gto

from computational_study_agent import StudyApplicationService
from computational_study_agent.hamiltonian_dataset_contracts import (
    HamiltonianDatasetSpec,
    MolecularDynamicsSamplingSpec,
)
from pyscf_agent.backend.ao_conventions import qh9_ao_metadata, transform_ao_matrix
from pyscf_agent.backend.parsing import spec_validator
from pyscf_agent.backend.state import default_state
from pyscf_agent.backend.workflow import execute_request
from pyscf_agent.contracts import task_spec_from_dict, task_spec_to_dict
from pyscf_agent.registry.platform import (
    SUPPORTED_ANALYSIS,
    SUPPORTED_JOBS,
    artifact_payload_schema,
    default_registry,
)


class MolecularDynamicsContractTests(unittest.TestCase):
    def test_task_spec_round_trip_preserves_qh9_sampling_contract(self):
        task_spec = task_spec_from_dict({
            'job': {'name': 'molecular_dynamics'},
            'molecular_dynamics': {'profile': 'qh9_relaxed_scf'},
            'molecular_dynamics': {
                'profile': 'qh9_relaxed_scf',                'steps': 100,
                'sample_stride': 10,
                'sample_offset': 9,
                'velocity_seed': 17,
            },
            'runtime': {
                'conv_tol': 1e-8,
                'conv_tol_grad': 3.16e-5,
                'grid_level': 3,
                'diis_space': 8,
            },
        })

        restored = task_spec_from_dict(task_spec_to_dict(task_spec))

        self.assertEqual(
            restored.molecular_dynamics.sampled_frame_indices,
            tuple(range(9, 100, 10)),
        )
        self.assertEqual(restored.runtime.grid_level, 3)
        self.assertEqual(restored.runtime.diis_space, 8)

    def test_validator_materializes_qh9_runtime_defaults(self):
        state = default_state('md-validation')
        state['task_spec'] = {
            'task_type': 'molecular',
            'system': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'def2-svp',
                'unit': 'Angstrom',
                'charge': 0,
                'spin': 0,
                'symmetry': False,
            },
            'method': {'name': 'dft', 'restricted': True, 'xc': 'b3lyp'},
            'job': {'name': 'molecular_dynamics'},
            'molecular_dynamics': {'profile': 'qh9_relaxed_scf'},
            'analysis': {'outputs': ['trajectory']},
            'runtime': {'max_cycle': 50, 'verbose': 0, 'scf_algorithm': 'standard'},
        }

        validated = spec_validator(state)

        self.assertEqual(validated['validation_errors'], [])
        self.assertEqual(validated['task_spec']['runtime']['conv_tol'], 1e-8)
        self.assertEqual(validated['task_spec']['runtime']['conv_tol_grad'], 3.16e-5)
        self.assertEqual(validated['task_spec']['runtime']['grid_level'], 3)
        self.assertEqual(validated['task_spec']['runtime']['diis_space'], 8)
        defaulted_fields = {item['field'] for item in validated['applied_defaults']}
        self.assertTrue({
            'runtime.conv_tol',
            'runtime.conv_tol_grad',
            'runtime.grid_level',
            'runtime.diis_space',
        }.issubset(defaulted_fields))

    def test_registry_owns_md_job_observable_and_artifacts(self):
        registry = default_registry()
        module = registry.module('core.execution')

        self.assertIn('molecular_dynamics', SUPPORTED_JOBS)
        self.assertIn('trajectory', SUPPORTED_ANALYSIS)
        self.assertIn('molecular.job.molecular_dynamics', module.capability_ids)
        self.assertEqual(
            artifact_payload_schema('molecular_md_manifest', registry),
            'pyscf-agent.molecular-md-manifest.v1',
        )
        self.assertEqual(
            registry.artifact_contract('molecular_dynamics').binary_artifact_kinds,
            ('molecular_md_frame_arrays',),
        )
        self.assertEqual(registry.registry_issues(), ())

    def test_strict_profile_defaults_and_conflicts(self):
        for profile, tolerance, accepted in (
            ('qh9', None, True),
            ('qh9', 1e-13, True),
            ('qh9', 1e-8, False),
            ('qh9_relaxed_scf', 1e-13, False),
            (None, None, False),
            ('unknown', None, False),
        ):
            with self.subTest(profile=profile, tolerance=tolerance):
                state = default_state('md-profile-validation')
                state['task_spec'] = {
                    'system': {'atom': 'H 0 0 0; H 0 0 0.74', 'basis': 'def2-svp'},
                    'method': {'name': 'dft', 'restricted': True, 'xc': 'b3lyp'},
                    'job': {'name': 'molecular_dynamics'},
                    'molecular_dynamics': {'profile': profile},
                    'runtime': {'conv_tol': tolerance},
                    'analysis': {'outputs': ['trajectory']},
                }
                validated = spec_validator(state)
                self.assertEqual(not validated['validation_errors'], accepted)
                if accepted:
                    self.assertEqual(validated['task_spec']['runtime']['conv_tol'], 1e-13)
                elif tolerance is not None:
                    self.assertEqual(validated['task_spec']['runtime']['conv_tol'], tolerance)

    def test_qh9_ao_transform_is_explicit_and_symmetric(self):
        mol = gto.M(
            atom='H 0 0 0; C 0 0 1.1',
            basis='def2-svp',
            spin=1,
            verbose=0,
        )
        metadata = qh9_ao_metadata(mol, 'def2-svp')
        source = np.arange(mol.nao_nr() ** 2, dtype=float).reshape(
            mol.nao_nr(), mol.nao_nr()
        )
        source = source + source.T

        transformed = transform_ao_matrix(
            source,
            metadata['source_indices'],
            metadata['phase_signs'],
        )

        self.assertEqual(metadata['atom_slices'], ((0, 5), (5, 19)))
        self.assertEqual(metadata['source_indices'][:5], (0, 1, 3, 4, 2))
        self.assertEqual(
            metadata['source_indices'][5:],
            (5, 6, 7, 9, 10, 8, 12, 13, 11, 14, 15, 16, 17, 18),
        )
        self.assertTrue(np.allclose(transformed, transformed.T))
        self.assertTrue(np.array_equal(
            transformed,
            source[np.ix_(metadata['source_indices'], metadata['source_indices'])],
        ))


class MolecularDynamicsExecutionTests(unittest.TestCase):
    def test_failed_scf_preserves_log_and_first_frame_diagnostics(self):
        request = {
            'system': {'atom': 'H 0 0 0; H 0 0 0.74', 'basis': 'def2-svp'},
            'method': {'name': 'dft', 'restricted': True, 'xc': 'b3lyp'},
            'job': {'name': 'molecular_dynamics'},
            'runtime': {'verbose': 4, 'max_cycle': 1},
            'analysis': {'outputs': ['trajectory']},
            'molecular_dynamics': {'profile': 'qh9', 'steps': 2, 'sample_stride': 1, 'sample_offset': 0},
        }
        with tempfile.TemporaryDirectory() as root:
            report = execute_request(json.dumps(request), channel='test', work_dir=root,
                                     run_id='md-failed-scf')['task_report']
            self.assertEqual(report['execution_status'], 'failed')
            self.assertEqual(report['retry_count'], 0)
            artifacts = {a['kind']: a for a in report['artifacts']}
            log = Path(artifacts['raw_scf_output']['path']).read_text()
            self.assertIn('cycle= 1', log)
            self.assertIn('SCF not converged', log)
            self.assertTrue(report['raw_scf_output'])
            failure = json.loads(Path(artifacts['molecular_md_failure']['path']).read_text())
            self.assertEqual(failure['failed_frame_index'], 0)
            self.assertEqual(failure['completed_frame_count'], 0)
            self.assertEqual(failure['last_scf_iteration']['cycle'], 1)
            self.assertIsNotNone(failure['last_scf_iteration']['orbital_gradient_norm'])
            self.assertFalse(failure['scf_converged'])
            self.assertEqual(failure['runtime']['conv_tol'], 1e-13)
            self.assertEqual(report['errors'][-1]['details']['molecular_dynamics'], failure)
            self.assertNotIn('molecular_md_frame_arrays', artifacts)

    def test_failure_frame_counts_unsampled_completed_steps(self):
        from pyscf.md.integrators import VelocityVerlet
        original = VelocityVerlet._compute_accel

        def fail_third_frame(integrator):
            if integrator._step == 2:
                raise RuntimeError('injected force evaluation failure')
            return original(integrator)

        request = {
            'system': {'atom': 'H 0 0 0; H 0 0 0.74', 'basis': 'def2-svp'},
            'method': {'name': 'dft', 'restricted': True, 'xc': 'b3lyp'},
            'job': {'name': 'molecular_dynamics'},
            'runtime': {'verbose': 4},
            'analysis': {'outputs': ['trajectory']},
            'molecular_dynamics': {'profile': 'qh9', 'steps': 4, 'sample_stride': 4, 'sample_offset': 3},
        }
        with tempfile.TemporaryDirectory() as root, patch.object(VelocityVerlet, '_compute_accel', fail_third_frame):
            report = execute_request(json.dumps(request), channel='test', work_dir=root,
                                     run_id='md-failed-later')['task_report']
            self.assertEqual(report['execution_status'], 'failed')
            failure = report['errors'][-1]['details']['molecular_dynamics']
            self.assertEqual(failure['message'], 'injected force evaluation failure')
            self.assertEqual(failure['failed_frame_index'], 2)
            self.assertEqual(failure['completed_frame_count'], 2)
            self.assertEqual(failure['sampled_frame_count'], 0)
            self.assertEqual(failure['time_au'], 100.0)
            self.assertEqual(failure['last_scf_iteration']['frame_index'], 1)
            self.assertTrue(failure['scf_converged'])

    def test_strict_two_frame_run_keeps_profile_and_all_samples(self):
        request = json.dumps({
            'system': {'atom': 'H 0 0 0; H 0 0 0.74', 'basis': 'def2-svp'},
            'method': {'name': 'dft', 'restricted': True, 'xc': 'b3lyp'},
            'job': {'name': 'molecular_dynamics'},
            'runtime': {'verbose': 0},
            'analysis': {'outputs': ['trajectory']},
            'molecular_dynamics': {
                'profile': 'qh9', 'steps': 2, 'sample_stride': 1,
                'sample_offset': 0, 'velocity_seed': 11,
            },
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            report = execute_request(
                request, channel='test', locale='en', work_dir=temp_dir,
                run_id='md-strict-two-frames',
            )['task_report']
            self.assertEqual(report['execution_status'], 'succeeded')
            trajectory = report['structured_results']['trajectory']
            self.assertEqual(trajectory['runtime']['conv_tol'], 1e-13)
            self.assertEqual(trajectory['compatibility']['profile'], 'qh9')
            self.assertEqual(trajectory['compatibility']['reference_deviations'], [])
            self.assertFalse(trajectory['compatibility']['bitwise_reproduction'])
            self.assertEqual(trajectory['sampled_frame_count'], 2)
            arrays_path = next(
                item['path'] for item in report['artifacts']
                if item['kind'] == 'molecular_md_frame_arrays'
            )
            with np.load(arrays_path, allow_pickle=False) as arrays:
                self.assertEqual(arrays['frame_indices'].tolist(), [0, 1])
                self.assertEqual(arrays['times_au'].tolist(), [0.0, 50.0])
                self.assertEqual(arrays['fock_matrices'].shape, (2, 10, 10))

    def test_one_frame_real_pyscf_run_writes_typed_artifacts(self):
        request = json.dumps({
            'task_type': 'molecular',
            'system': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'def2-svp',
                'unit': 'Angstrom',
                'charge': 0,
                'spin': 0,
                'symmetry': False,
            },
            'method': {'name': 'dft', 'restricted': True, 'xc': 'b3lyp'},
            'job': {'name': 'molecular_dynamics'},
            'molecular_dynamics': {'profile': 'qh9_relaxed_scf'},
            'analysis': {'outputs': ['trajectory']},
            'runtime': {'max_cycle': 50, 'verbose': 0, 'scf_algorithm': 'standard'},
            'molecular_dynamics': {
                'profile': 'qh9_relaxed_scf',                'steps': 1,
                'sample_stride': 1,
                'sample_offset': 0,
                'velocity_seed': 11,
            },
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            report = execute_request(
                request,
                channel='test',
                locale='en',
                work_dir=temp_dir,
                run_id='md-one-frame',
            )['task_report']

            self.assertEqual(report['execution_status'], 'succeeded')
            trajectory = report['structured_results']['trajectory']
            self.assertEqual(trajectory['steps_completed'], 1)
            self.assertEqual(trajectory['sampled_frame_count'], 1)
            self.assertEqual(trajectory['runtime']['grid_level'], 3)
            self.assertEqual(trajectory['runtime']['conv_tol'], 1e-8)
            self.assertEqual(trajectory['atomic_numbers'], [1, 1])
            self.assertEqual(len(trajectory['frames'][0]['positions_angstrom']), 2)
            self.assertEqual(trajectory['array_metadata']['fock_matrices']['shape'], [1, 10, 10])
            self.assertTrue(trajectory['array_metadata']['fock_matrices']['finite'])
            self.assertTrue(trajectory['matrix_validation']['passed'])
            self.assertLessEqual(
                trajectory['matrix_validation']['fock_max_asymmetry'],
                trajectory['matrix_validation']['symmetry_tolerance'],
            )
            self.assertEqual(trajectory['compatibility']['profile'], 'qh9_relaxed_scf')
            self.assertEqual(
                trajectory['compatibility']['reference_deviations'][0]['qh9_reference'],
                1e-13,
            )
            self.assertFalse(trajectory['compatibility']['bitwise_reproduction'])
            self.assertNotIn('sha256', json.dumps(trajectory).lower())

            artifacts = {item['kind']: item for item in report['artifacts']}
            self.assertIn('molecular_md_manifest', artifacts)
            self.assertIn('molecular_md_frame_arrays', artifacts)
            arrays_path = Path(artifacts['molecular_md_frame_arrays']['path'])
            self.assertTrue(arrays_path.is_file())
            with np.load(arrays_path, allow_pickle=False) as arrays:
                self.assertEqual(arrays['positions_angstrom'].shape, (1, 2, 3))
                self.assertEqual(arrays['fock_matrices'].shape, (1, 10, 10))
                self.assertEqual(arrays['overlap_matrices'].shape, (1, 10, 10))
                self.assertEqual(arrays['frame_indices'].tolist(), [0])
                self.assertEqual(
                    str(arrays['schema']),
                    'pyscf-agent.molecular-md-frame-arrays.v1',
                )

            dataset_spec = HamiltonianDatasetSpec(
                dataset_id='h2-one-frame',
                name='H2 one-frame smoke dataset',
                target_molecule_count=1,
                geometries_per_molecule=1,
                molecular_dynamics=MolecularDynamicsSamplingSpec(
                    steps=1,
                    sample_stride=1,
                    sample_offset=0,
                ),
            )
            dataset_dir = Path(temp_dir) / 'dataset'
            manifest = StudyApplicationService().finalize_hamiltonian_dataset(
                {
                    'study_id': 'md-study-smoke',
                    'work_dir': temp_dir,
                    'cases': [{
                        'case_id': 'case-0001',
                        'variables': {'molecule_id': 'h2'},
                        'request': json.loads(request),
                        'task_report': report,
                    }],
                },
                dataset_spec,
                output_dir=dataset_dir,
            )

            self.assertEqual(manifest.status, 'complete')
            self.assertEqual(manifest.accepted_structure_count, 1)
            self.assertEqual(manifest.rejected_structure_count, 0)
            sample_payload = json.loads(
                (dataset_dir / 'samples.jsonl').read_text(encoding='utf-8').strip()
            )
            self.assertEqual(sample_payload['molecule_id'], 'h2')
            self.assertEqual(sample_payload['fock']['array_index'], 0)
            self.assertEqual(sample_payload['overlap']['array_index'], 0)
            self.assertNotIn('sha256', json.dumps(manifest.to_dict()).lower())


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
