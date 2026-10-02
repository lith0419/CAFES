from __future__ import annotations

import json
import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from pyscf_agent.artifacts import ArtifactRepository

from computational_study_agent.execution_receipts import _dataset_execution_progress
from computational_study_agent.datasets.hamiltonian.contracts import (
    HAMILTONIAN_MANIFEST_SCHEMA,
    HamiltonianDatasetSpec,
)
from computational_study_agent.datasets.hamiltonian.postprocessing import (
    HAMILTONIAN_DATASET_GENERATION_FILENAME,
    collect_hamiltonian_dataset,
    generate_hamiltonian_dataset,
    materialize_hamiltonian_dataset,
)
from computational_study_agent.postprocessing import run_postprocessing


class HamiltonianDatasetPostprocessingTests(unittest.TestCase):
    @staticmethod
    def _source_report(root: Path, matrix_path: str):
        import numpy as np

        local_matrix_path = root / 'source-arrays.npz'
        np.savez(
            local_matrix_path,
            fock_matrices=np.repeat(np.eye(2)[None, :, :], 10, axis=0),
            overlap_matrices=np.repeat(np.eye(2)[None, :, :], 10, axis=0),
        )
        source_path = matrix_path or str(local_matrix_path)
        dataset_root = root / 'dataset'
        dataset_root.mkdir(parents=True)
        rows = []
        for index in range(10):
            rows.append({
                'schema': 'pyscf-agent.hamiltonian-sample.v1',
                'sample_id': 'mol-001:frame-{0:06d}'.format(index),
                'molecule_id': 'mol-001',
                'geometry_id': 'frame-{0:06d}'.format(index),
                'fock': {
                    'path': source_path,
                    'array_key': 'fock_matrices',
                    'array_index': index,
                    'shape': [2, 2],
                    'dtype': 'float64',
                    'format': 'npz',
                },
                'overlap': {
                    'path': source_path,
                    'array_key': 'overlap_matrices',
                    'array_index': index,
                    'shape': [2, 2],
                    'dtype': 'float64',
                    'format': 'npz',
                },
            })
        (dataset_root / 'samples.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in rows),
            encoding='utf-8',
        )
        (dataset_root / 'rejections.jsonl').write_text('', encoding='utf-8')
        spec = HamiltonianDatasetSpec(
            dataset_id='collection-test',
            name='collection test',
            target_molecule_count=1,
        )
        manifest = {
            'schema': HAMILTONIAN_MANIFEST_SCHEMA,
            'spec': spec.to_dict(),
            'target_structure_count': 10,
            'molecule_count': 1,
            'accepted_structure_count': 10,
            'rejected_structure_count': 0,
            'accounted_structure_count': 10,
            'missing_structure_count': 0,
            'status': 'complete',
            'split_counts': {'train': 10, 'validation': 0, 'test': 0},
            'artifacts': {
                'sample_index': str(dataset_root / 'samples.jsonl'),
                'rejection_index': str(dataset_root / 'rejections.jsonl'),
                'dataset_manifest': str(dataset_root / 'dataset-manifest.json'),
            },
        }
        (dataset_root / 'dataset-manifest.json').write_text(
            json.dumps(manifest),
            encoding='utf-8',
        )
        return {
            'study_id': 'dataset-study',
            'name': 'dataset study',
            'system_type': 'molecular',
            'work_dir': str(root),
            'comparison_table': [],
            'cases': [],
            'artifacts': [],
            'dataset_manifest': manifest,
        }, local_matrix_path

    def test_generate_then_collect_local_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, _source = self._source_report(root, '')
            generated = root / 'postprocessing' / 'hamiltonian-dataset-generated'
            output = root / 'postprocessing' / 'hamiltonian-dataset'

            generation = generate_hamiltonian_dataset(
                report,
                output_dir=generated,
            )
            result = collect_hamiltonian_dataset(report, output_dir=output)
            rows = [
                json.loads(line)
                for line in (output / 'samples.jsonl').read_text(encoding='utf-8').splitlines()
            ]
            manifest = json.loads(
                (output / 'dataset-manifest.json').read_text(encoding='utf-8')
            )

        self.assertEqual(generation['sample_count'], 10)
        self.assertEqual(generation['location'], 'local')
        self.assertEqual(result['sample_count'], 10)
        self.assertEqual(result['trajectory_file_count'], 1)
        self.assertEqual(rows[0]['fock']['path'], 'arrays/trajectory-0001-mol-001.npz')
        self.assertEqual(rows[0]['overlap']['path'], rows[0]['fock']['path'])
        self.assertTrue(manifest['generation']['portable'])
        self.assertEqual(manifest['collection']['status'], 'complete')
        self.assertFalse(manifest['collection']['checksum_validation'])
        self.assertEqual(manifest['artifacts']['sample_index'], 'samples.jsonl')

    @staticmethod
    def _dataset_bytes(root):
        return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}

    def test_failed_regeneration_preserves_previous_dataset_and_success_receipt(self):
        for failure in ('missing_source', 'invalid_shape'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                report, _source = self._source_report(root, '')
                destination = root / 'generated'
                original = generate_hamiltonian_dataset(report, output_dir=destination)
                previous_root = Path(original['dataset_root'])
                previous_files = self._dataset_bytes(previous_root)
                receipt_path = root / 'postprocessing' / HAMILTONIAN_DATASET_GENERATION_FILENAME
                previous_receipt = receipt_path.read_bytes()
                replacement = root / 'replacement.npz'
                shape = 3 if failure == 'invalid_shape' else 2
                np.savez(replacement, fock_matrices=np.full((10, shape, shape), 9.0),
                         overlap_matrices=np.repeat(np.eye(shape)[None], 10, axis=0))
                sample_path = root / 'dataset' / 'samples.jsonl'
                rows = [json.loads(line) for line in sample_path.read_text().splitlines()]
                for row in rows:
                    row['fock']['path'] = str(replacement)
                    row['overlap']['path'] = str(root / 'missing.npz' if failure == 'missing_source' else replacement)
                sample_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

                with self.assertRaises((FileNotFoundError, ValueError)):
                    generate_hamiltonian_dataset(report, output_dir=destination)

                self.assertEqual(self._dataset_bytes(previous_root), previous_files)
                self.assertEqual(receipt_path.read_bytes(), previous_receipt)
                self.assertEqual(list(destination.iterdir()), [previous_root])
                collected = root / 'collected'
                result = collect_hamiltonian_dataset(report, output_dir=collected)
                self.assertEqual(result['status'], 'succeeded')
                for relative, data in previous_files.items():
                    if relative != 'dataset-manifest.json':
                        self.assertEqual((collected / relative).read_bytes(), data)

    def test_successful_regeneration_keeps_previous_arrays_independent_of_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, source = self._source_report(root, '')
            destination = root / 'generated'
            first = generate_hamiltonian_dataset(report, output_dir=destination)
            previous_root = Path(first['dataset_root'])
            previous_files = self._dataset_bytes(previous_root)
            np.savez(source, fock_matrices=np.full((10, 2, 2), 9.0),
                     overlap_matrices=np.repeat(np.eye(2)[None], 10, axis=0))
            second = generate_hamiltonian_dataset(report, output_dir=destination)

            self.assertNotEqual(first['dataset_root'], second['dataset_root'])
            self.assertEqual(self._dataset_bytes(previous_root), previous_files)
            collected = root / 'collected'
            collect_hamiltonian_dataset(report, output_dir=collected)
            with np.load(collected / 'arrays/trajectory-0001-mol-001.npz') as archive:
                self.assertEqual(float(archive['fock_matrices'][0, 0, 0]), 9.0)

    def test_receipt_write_failure_preserves_previous_collectable_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, _source = self._source_report(root, '')
            destination = root / 'generated'
            first = generate_hamiltonian_dataset(report, output_dir=destination)
            previous_files = self._dataset_bytes(Path(first['dataset_root']))
            receipt_path = root / 'postprocessing' / HAMILTONIAN_DATASET_GENERATION_FILENAME
            previous_receipt = receipt_path.read_bytes()
            with mock.patch.object(ArtifactRepository, 'write_json', side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    generate_hamiltonian_dataset(report, output_dir=destination)
            self.assertEqual(receipt_path.read_bytes(), previous_receipt)
            self.assertEqual(self._dataset_bytes(Path(first['dataset_root'])), previous_files)
            self.assertEqual(collect_hamiltonian_dataset(report, output_dir=root / 'collected')['status'], 'succeeded')

    def test_generate_is_remote_and_collect_is_the_only_download_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, fixture = self._source_report(
                root,
                '/scratch/qh9/result-molecular-md-frames.npz',
            )
            server_output = root / 'simulated-server-dataset'
            transfers = {}

            def generate_on_server(request):
                local_request = copy.deepcopy(request)
                for row in local_request['samples']:
                    row['fock']['path'] = str(fixture)
                    row['overlap']['path'] = str(fixture)
                generated = materialize_hamiltonian_dataset(
                    local_request,
                    output_dir=server_output,
                    location='remote_executor',
                )
                generated['dataset_root'] = '/scratch/qh9/generated-dataset'
                generated['manifest_path'] = '/scratch/qh9/generated-dataset/dataset-manifest.json'
                return generated

            def collect_files(paths):
                transfers.update(paths)
                for source, destination in paths.items():
                    Path(destination).parent.mkdir(parents=True, exist_ok=True)
                    relative = Path(source).relative_to('/scratch/qh9/generated-dataset')
                    shutil.copy2(server_output / relative, destination)

            generation_result = run_postprocessing(
                report,
                specs=None,
                actions=['generate_hamiltonian_dataset'],
                dataset_generator=generate_on_server,
            )
            self.assertEqual(transfers, {})
            self.assertFalse(
                (root / 'postprocessing' / 'hamiltonian-dataset' / 'arrays').exists()
            )
            result = run_postprocessing(
                report,
                specs=None,
                actions=['collect_hamiltonian_dataset'],
                artifact_collector=collect_files,
            )

        self.assertEqual(generation_result['actions'][0]['location'], 'remote_executor')
        self.assertEqual(result['status'], 'succeeded')
        self.assertEqual(result['plot_specs'], [])
        self.assertEqual(len(result['actions']), 1)
        self.assertIn(
            '/scratch/qh9/generated-dataset/arrays/trajectory-0001-mol-001.npz',
            transfers,
        )
        self.assertIn(
            'hamiltonian_collected_dataset_manifest',
            {artifact['kind'] for artifact in result['artifacts']},
        )

    def test_collect_requires_a_successful_generation_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, _fixture = self._source_report(root, '')
            with self.assertRaisesRegex(ValueError, 'Generate Dataset'):
                collect_hamiltonian_dataset(
                    report,
                    output_dir=root / 'postprocessing' / 'hamiltonian-dataset',
                )

    def test_dataset_status_reports_trajectory_and_structure_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = HamiltonianDatasetSpec(
                dataset_id='status-test',
                name='status test',
                target_molecule_count=2,
            )
            (root / 'study-plan.json').write_text(json.dumps({
                'comparison': {
                    'mode': 'hamiltonian_dataset_assembly',
                    'dataset_spec': spec.to_dict(),
                },
                'cases': [{}, {}],
            }), encoding='utf-8')
            generation_path = root / 'postprocessing' / HAMILTONIAN_DATASET_GENERATION_FILENAME
            generation_path.parent.mkdir(parents=True)
            generation_path.write_text(json.dumps({
                'schema': 'pyscf-agent.hamiltonian-dataset-generation.v1',
                'status': 'succeeded',
                'location': 'remote_executor',
                'dataset_root': '/scratch/qh9/dataset',
                'trajectory_file_count': 2,
            }), encoding='utf-8')
            progress = _dataset_execution_progress(root, [{
                'summary': {'cases': [
                    {
                        'case_id': 'case-0001',
                        'terminal': True,
                        'task_status': 'succeeded',
                        'state': 'completed',
                    },
                    {
                        'case_id': 'case-0002',
                        'terminal': False,
                        'task_status': None,
                        'state': 'running',
                    },
                ]},
            }])

        self.assertEqual(progress['terminal_trajectories'], 1)
        self.assertEqual(progress['succeeded_trajectories'], 1)
        self.assertEqual(progress['running_trajectories'], 1)
        self.assertEqual(progress['available_structures'], 10)
        self.assertEqual(progress['pending_structures'], 10)
        self.assertTrue(progress['generated'])
        self.assertFalse(progress['collected'])
        self.assertEqual(progress['generation_location'], 'remote_executor')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
