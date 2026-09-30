from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from computational_study_agent.hamiltonian_dataset_contracts import (
    AOBasisMetadata,
    ElectronicStructureSpec,
    HamiltonianDatasetManifest,
    HamiltonianDatasetSpec,
    HamiltonianSample,
    MatrixArtifactReference,
    MolecularGeometry,
    MolecularDynamicsSamplingSpec,
)
from computational_study_agent import StudyApplicationService
from computational_study_agent.hamiltonian_dataset_finalize import (
    finalize_hamiltonian_dataset,
)
from computational_study_agent.executor import run_study
from computational_study_agent.cli import main as study_cli_main
from computational_study_agent.validation import has_errors, validate_study_plan


def _geometry() -> MolecularGeometry:
    return MolecularGeometry(
        molecule_id='mol-001',
        geometry_id='geo-003',
        atomic_numbers=(1, 1),
        positions=((0.0, 0.0, 0.0), (0.0, 0.0, 0.74)),
        source={'collection': 'curated-structures'},
    )


def _ao_basis() -> AOBasisMetadata:
    return AOBasisMetadata(
        convention='qh9',
        labels=('H1 1s', 'H2 1s'),
        atom_slices=((0, 1), (1, 2)),
        source_indices=(1, 0),
        phase_signs=(1, -1),
    )


def _matrix(key: str, shape=(2, 2)) -> MatrixArtifactReference:
    return MatrixArtifactReference(
        path='samples/mol-001/geo-003.npz',
        array_key=key,
        shape=shape,
        dtype='float64',
    )


class _DatasetTaskExecutor:
    executor_id = 'dataset-test'
    execution_mode = 'test'
    supports_independent_batch = False
    supports_recoverable_batch = False

    @staticmethod
    def _trajectory():
        frames = [
            {
                'array_index': index,
                'frame_index': frame_index,
                'potential_energy_hartree': -1.0 - index / 1000.0,
                'converged': True,
                'positions_angstrom': [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.74 + index / 100.0],
                ],
            }
            for index, frame_index in enumerate(range(9, 100, 10))
        ]
        return {
            'atomic_numbers': [1, 1],
            'frames': frames,
            'ao_basis': _ao_basis().to_dict(),
            'frame_arrays_artifact': {
                'path': '/scratch/remote/result-molecular-md-frames.npz',
            },
            'array_metadata': {
                'fock_matrices': {
                    'shape': [10, 2, 2],
                    'dtype': 'float64',
                    'finite': True,
                },
                'overlap_matrices': {
                    'shape': [10, 2, 2],
                    'dtype': 'float64',
                    'finite': True,
                },
            },
            'matrix_validation': {'passed': True},
        }

    def execute_task(self, _request, *, run_id=None, **_kwargs):
        return {
            'execution_status': 'succeeded',
            'run_id': run_id,
            'structured_results': {'trajectory': self._trajectory()},
            'artifacts': [],
        }

    def describe(self):
        return {
            'executor_id': self.executor_id,
            'execution_mode': self.execution_mode,
        }


class HamiltonianDatasetContractTests(unittest.TestCase):
    def test_default_campaign_is_100_molecules_by_10_geometries(self):
        spec = HamiltonianDatasetSpec(dataset_id='qh9-small', name='QH9 small')

        self.assertEqual(spec.target_molecule_count, 100)
        self.assertEqual(spec.geometries_per_molecule, 10)
        self.assertEqual(spec.target_structure_count, 1000)
        self.assertEqual(spec.split_protocol, 'molecule_random')
        self.assertEqual(
            spec.split_fractions,
            {'train': 0.8, 'validation': 0.1, 'test': 0.1},
        )
        self.assertEqual(spec.electronic_structure.method, 'dft')
        self.assertEqual(spec.electronic_structure.xc, 'b3lyp')
        self.assertEqual(spec.electronic_structure.basis, 'def2-svp')
        self.assertEqual(spec.electronic_structure.scf_options, {
            'grid_level': 3,
            'conv_tol': 1e-8,
            'conv_tol_grad': 3.16e-5,
            'diis_space': 8,
        })
        self.assertEqual(spec.compatibility_profile, 'qh9_relaxed_scf')

    def test_spec_round_trip_preserves_explicit_geometry_protocol(self):
        spec = HamiltonianDatasetSpec(
            dataset_id='geometry-generalization',
            name='Geometry generalization',
            split_protocol='geometry_random',
            compatibility_profile='custom',
            electronic_structure=ElectronicStructureSpec(
                scf_options={'conv_tol': 1e-10},
            ),
        )

        restored = HamiltonianDatasetSpec.from_dict(spec.to_dict())

        self.assertEqual(restored, spec)
        self.assertEqual(restored.target_structure_count, 1000)

    def test_sampling_contract_produces_ten_unambiguous_frame_indices(self):
        sampling = MolecularDynamicsSamplingSpec()

        self.assertEqual(sampling.sampled_frame_indices, tuple(range(9, 100, 10)))
        self.assertEqual(sampling.sampled_frame_count, 10)
        self.assertEqual(sampling.velocity_seed_for_case(3), 3)

        with self.assertRaisesRegex(ValueError, 'exactly geometries_per_molecule'):
            HamiltonianDatasetSpec(
                dataset_id='bad-sampling',
                name='bad sampling',
                geometries_per_molecule=9,
            )

    def test_strict_dataset_profile_reaches_trajectory_tasks(self):
        payload = {
            'dataset_id': 'qh9-strict', 'name': 'Strict QH9 sampling',
            'target_molecule_count': 1, 'geometries_per_molecule': 100,
            'compatibility_profile': 'qh9',
            'molecular_dynamics': {'steps': 100, 'sample_stride': 1, 'sample_offset': 0},
        }
        spec = HamiltonianDatasetSpec.from_dict(payload)
        self.assertEqual(spec.electronic_structure.scf_options['conv_tol'], 1e-13)
        self.assertEqual(HamiltonianDatasetSpec.from_dict(spec.to_dict()), spec)
        plan = StudyApplicationService().build_hamiltonian_dataset_plan(spec, [_geometry()])
        self.assertFalse(has_errors(validate_study_plan(plan)))
        request = plan.cases[0].request
        self.assertEqual(request['runtime']['conv_tol'], 1e-13)
        self.assertEqual(request['molecular_dynamics']['profile'], 'qh9')
        self.assertEqual(request['molecular_dynamics']['sample_stride'], 1)
        self.assertEqual(request['molecular_dynamics']['sample_offset'], 0)
        self.assertEqual(plan.comparison['expected_sample_count'], 100)
        payload['electronic_structure'] = {'scf_options': {
            'conv_tol': 1e-8, 'conv_tol_grad': 3.16e-5, 'grid_level': 3, 'diis_space': 8,
        }}
        with self.assertRaisesRegex(ValueError, 'requires scf_options.conv_tol'):
            HamiltonianDatasetSpec.from_dict(payload)

    def test_dataset_compiler_builds_100_trajectory_cases_and_requires_cost_review(self):
        spec = HamiltonianDatasetSpec(dataset_id='qh9-small', name='QH9 small')
        seeds = [
            MolecularGeometry(
                molecule_id='mol-{0:03d}'.format(index),
                geometry_id='seed',
                atomic_numbers=(1, 1),
                positions=((0.0, 0.0, 0.0), (0.0, 0.0, 0.74)),
            )
            for index in range(100)
        ]

        plan = StudyApplicationService().build_hamiltonian_dataset_plan(spec, seeds)

        self.assertEqual(len(plan.cases), 100)
        self.assertEqual(plan.observables, ['trajectory'])
        self.assertEqual(plan.cases[0].request['job'], 'molecular_dynamics')
        self.assertEqual(plan.cases[0].request['molecular_dynamics']['velocity_seed'], 0)
        self.assertEqual(plan.cases[-1].request['molecular_dynamics']['velocity_seed'], 99)
        self.assertEqual(plan.cases[0].variables['expected_sample_count'], 10)
        self.assertEqual(plan.comparison['expected_sample_count'], 1000)
        self.assertEqual(plan.cost_estimate['cases'][0]['trajectory_steps'], 100)
        self.assertEqual(plan.cost_estimate['cases'][0]['sampled_frame_count'], 10)
        self.assertEqual(plan.cost_estimate['cases'][0]['estimation_model'], 'bomd_dft_gradient_proxy')
        self.assertTrue(plan.cost_estimate['approval_required'])
        self.assertFalse(plan.cost_estimate['can_execute'])
        self.assertFalse(has_errors(validate_study_plan(plan)))
        workflow_order = plan.workflow_configuration['execution_order']
        self.assertIn('study.hamiltonian_dataset_assembly', workflow_order)
        self.assertLess(
            workflow_order.index('study.hamiltonian_dataset_assembly'),
            workflow_order.index('core.study_report'),
        )

    def test_study_execution_automatically_finalizes_dataset_manifest(self):
        spec = HamiltonianDatasetSpec(
            dataset_id='automatic-finalize',
            name='automatic finalize',
            target_molecule_count=1,
        )
        plan = StudyApplicationService().build_hamiltonian_dataset_plan(
            spec,
            [_geometry()],
        )
        plan.resource_policy['approved'] = True
        plan.cost_estimate['approved'] = True
        plan.cost_estimate['policy']['approved'] = True

        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_study(
                plan,
                work_dir=temp_dir,
                resume=False,
                task_executor=_DatasetTaskExecutor(),
            )
            persisted = json.loads(
                (Path(report.work_dir) / 'study-report.json').read_text(encoding='utf-8')
            )

        self.assertEqual(report.dataset_manifest['status'], 'complete')
        self.assertEqual(report.dataset_manifest['accepted_structure_count'], 10)
        self.assertEqual(persisted['dataset_manifest'], report.dataset_manifest)
        self.assertTrue({
            'hamiltonian_dataset_manifest',
            'hamiltonian_sample_index',
        }.issubset({artifact['kind'] for artifact in report.artifacts}))
        self.assertTrue(report.dataset_manifest['artifacts']['rejection_index'].endswith(
            '/dataset/rejections.jsonl'
        ))

    def test_cli_builds_dataset_plan_from_json_seed_file(self):
        spec = HamiltonianDatasetSpec(
            dataset_id='cli-dataset',
            name='CLI dataset',
            target_molecule_count=2,
        )
        seeds = [
            {
                **_geometry().to_dict(),
                'molecule_id': 'mol-{0:03d}'.format(index),
            }
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            spec_path = Path(temp_dir) / 'dataset.json'
            seeds_path = Path(temp_dir) / 'seeds.json'
            spec_path.write_text(json.dumps(spec.to_dict()), encoding='utf-8')
            seeds_path.write_text(json.dumps({'seed_geometries': seeds}), encoding='utf-8')
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = study_cli_main([
                    '--dataset-spec', str(spec_path),
                    '--seed-geometries', str(seeds_path),
                    '--plan-only',
                ])

        plan = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(plan['cases']), 2)
        self.assertEqual(plan['comparison']['mode'], 'hamiltonian_dataset_assembly')

    def test_finalizer_accounts_for_every_frame_of_failed_trajectories(self):
        spec = HamiltonianDatasetSpec(
            dataset_id='failed-trajectories',
            name='failed trajectories',
            target_molecule_count=2,
        )
        report = {
            'study_id': 'failed-study',
            'cases': [
                {
                    'case_id': 'case-{0:04d}'.format(index + 1),
                    'variables': {'molecule_id': 'mol-{0}'.format(index)},
                    'request': {'atom': 'H 0 0 0; H 0 0 0.74'},
                    'task_report': {'execution_status': 'failed'},
                }
                for index in range(2)
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = finalize_hamiltonian_dataset(report, spec, temp_dir)
            rejection_lines = [
                json.loads(line)
                for line in (Path(temp_dir) / 'rejections.jsonl').read_text(encoding='utf-8').splitlines()
            ]

        self.assertEqual(manifest.accepted_structure_count, 0)
        self.assertEqual(manifest.rejected_structure_count, 20)
        self.assertEqual(manifest.accounted_structure_count, 20)
        self.assertEqual(manifest.status, 'complete')
        self.assertEqual(len(rejection_lines), 20)
        self.assertEqual(rejection_lines[0]['reasons'], ['task_status:failed'])

    def test_molecule_geometry_identity_does_not_use_a_hash(self):
        geometry = _geometry()

        self.assertEqual(geometry.sample_id, 'mol-001:geo-003')
        self.assertNotIn('sha256', json.dumps(geometry.to_dict()).lower())

    def test_molecule_geometry_can_be_loaded_from_planner_json(self):
        geometry = MolecularGeometry.from_dict(_geometry().to_dict())

        self.assertEqual(geometry, _geometry())

    def test_finalizer_uses_self_describing_report_without_opening_remote_npz(self):
        spec = HamiltonianDatasetSpec(
            dataset_id='remote-index',
            name='remote index',
            target_molecule_count=1,
        )
        frames = [
            {
                'array_index': index,
                'frame_index': frame_index,
                'potential_energy_hartree': -1.0 - index / 1000.0,
                'converged': True,
                'positions_angstrom': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74 + index / 100.0]],
            }
            for index, frame_index in enumerate(spec.molecular_dynamics.sampled_frame_indices)
        ]
        report = {
            'study_id': 'remote-study',
            'cases': [{
                'case_id': 'case-0001',
                'variables': {'molecule_id': 'mol-remote'},
                'request': {'atom': 'H 0 0 0; H 0 0 0.74'},
                'task_report': {
                    'execution_status': 'succeeded',
                    'run_id': 'remote-run',
                    'structured_results': {'trajectory': {
                        'atomic_numbers': [1, 1],
                        'frames': frames,
                        'ao_basis': _ao_basis().to_dict(),
                        'frame_arrays_artifact': {
                            'path': '/scratch/remote/result-molecular-md-frames.npz',
                        },
                        'array_metadata': {
                            'fock_matrices': {'shape': [10, 2, 2], 'dtype': 'float64', 'finite': True},
                            'overlap_matrices': {'shape': [10, 2, 2], 'dtype': 'float64', 'finite': True},
                        },
                        'matrix_validation': {'passed': True},
                    }},
                },
            }],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = finalize_hamiltonian_dataset(report, spec, temp_dir)
            samples = [
                json.loads(line)
                for line in (Path(temp_dir) / 'samples.jsonl').read_text(encoding='utf-8').splitlines()
            ]

        self.assertEqual(manifest.accepted_structure_count, 10)
        self.assertEqual(samples[0]['fock']['path'], '/scratch/remote/result-molecular-md-frames.npz')
        self.assertEqual(samples[-1]['fock']['array_index'], 9)

    def test_accepted_sample_requires_converged_shape_consistent_matrices(self):
        sample = HamiltonianSample(
            geometry=_geometry(),
            split='train',
            electronic_structure=ElectronicStructureSpec(),
            ao_basis=_ao_basis(),
            fock=_matrix('fock'),
            overlap=_matrix('overlap'),
            converged=True,
            total_energy_hartree=-1.1,
            provenance={'run_id': 'run-001'},
        )

        self.assertEqual(sample.sample_id, 'mol-001:geo-003')
        self.assertEqual(sample.to_dict()['fock']['array_key'], 'fock')
        self.assertEqual(sample.to_dict()['ao_basis']['source_indices'], (1, 0))
        self.assertEqual(sample.to_dict()['ao_basis']['phase_signs'], (1, -1))
        self.assertNotIn('sha256', json.dumps(sample.to_dict()).lower())

        with self.assertRaisesRegex(ValueError, 'must be converged'):
            HamiltonianSample(
                geometry=_geometry(),
                split='train',
                electronic_structure=ElectronicStructureSpec(),
                ao_basis=_ao_basis(),
                fock=_matrix('fock'),
                overlap=_matrix('overlap'),
                converged=False,
                total_energy_hartree=-1.1,
            )
        with self.assertRaisesRegex(ValueError, 'shapes'):
            HamiltonianSample(
                geometry=_geometry(),
                split='train',
                electronic_structure=ElectronicStructureSpec(),
                ao_basis=_ao_basis(),
                fock=_matrix('fock', shape=(3, 3)),
                overlap=_matrix('overlap'),
                converged=True,
                total_energy_hartree=-1.1,
            )

    def test_manifest_counts_splits_without_checksums(self):
        spec = HamiltonianDatasetSpec(dataset_id='qh9-small', name='QH9 small')
        manifest = HamiltonianDatasetManifest(
            spec=spec,
            molecule_count=100,
            accepted_structure_count=990,
            rejected_structure_count=10,
            split_counts={'train': 790, 'validation': 100, 'test': 100},
            artifacts={
                'sample_index': 'dataset/samples.jsonl',
                'matrix_store': 'dataset/matrices.h5',
            },
        )

        payload = manifest.to_dict()

        self.assertEqual(payload['target_structure_count'], 1000)
        self.assertEqual(payload['accepted_structure_count'], 990)
        self.assertEqual(payload['accounted_structure_count'], 1000)
        self.assertEqual(payload['missing_structure_count'], 0)
        self.assertEqual(payload['status'], 'complete')
        self.assertNotIn('sha256', json.dumps(payload).lower())

    def test_contracts_reject_invalid_dimensions_and_split_fractions(self):
        with self.assertRaisesRegex(ValueError, 'sum to 1.0'):
            HamiltonianDatasetSpec(
                dataset_id='bad',
                name='bad',
                split_fractions={'train': 0.8, 'validation': 0.2, 'test': 0.2},
            )
        with self.assertRaisesRegex(ValueError, 'equal length'):
            MolecularGeometry(
                molecule_id='mol',
                geometry_id='geo',
                atomic_numbers=(1, 1),
                positions=((0.0, 0.0, 0.0),),
            )
        with self.assertRaisesRegex(ValueError, 'square'):
            MatrixArtifactReference(
                path='sample.npz',
                array_key='fock',
                shape=(2, 3),
                dtype='float64',
            )
        with self.assertRaisesRegex(ValueError, 'permutation'):
            AOBasisMetadata(
                convention='qh9',
                labels=('H1 1s', 'H2 1s'),
                atom_slices=((0, 1), (1, 2)),
                source_indices=(0, 0),
            )


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
