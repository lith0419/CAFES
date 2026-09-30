from __future__ import annotations

import json
import io
import os
import tempfile
import unittest
from pathlib import Path

from pyscf_agent.artifacts import ArtifactRepository
from pyscf_agent.backend.artifacts import (
    register_artifact_reference,
    resolve_work_dir,
    serialize_npz_arrays,
)


class ArtifactRepositoryTests(unittest.TestCase):
    def test_json_artifact_uses_registered_schema_without_content_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'task-report.json'
            reference = ArtifactRepository().write_json(
                path,
                {'status': 'succeeded'},
                kind='task-report',
                description='Task report',
            )

            payload = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(payload['schema'], 'pyscf-agent.task-report.v1')
            self.assertEqual(reference['size_bytes'], path.stat().st_size)
            self.assertNotIn('sha256', reference)

    def test_existing_provider_output_is_registered_without_reading_its_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'log-libdmet-output.log'
            path.write_text('native libDMET output\n', encoding='utf-8')
            reference = ArtifactRepository().register_existing(
                path,
                kind='dmet_output_log',
                mime_type='text/plain; charset=utf-8',
                description='Complete libDMET provider output',
            )

            self.assertIsNotNone(reference)
            self.assertEqual(reference['size_bytes'], path.stat().st_size)
            self.assertNotIn('sha256', reference)

    def test_artifact_reference_preserves_logical_shared_filesystem_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            physical = root / 'physical-scratch'
            physical.mkdir()
            logical = root / 'scratch'
            logical.symlink_to(physical, target_is_directory=True)
            artifact = logical / 'result.json'

            reference = ArtifactRepository().write_json(
                artifact,
                {'status': 'succeeded'},
                kind='task-report',
            )

            expected = os.path.abspath(str(artifact))
            self.assertEqual(reference['path'], expected)
            self.assertNotEqual(reference['path'], str(artifact.resolve()))
            self.assertEqual(str(resolve_work_dir(str(logical))), os.path.abspath(str(logical)))
            self.assertTrue(Path(reference['path']).is_file())

    def test_unknown_artifact_kind_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'not registered'):
                ArtifactRepository().write_text(
                    Path(directory) / 'unknown.txt',
                    'content',
                    kind='unknown-artifact-kind',
                )

    def test_existing_reference_is_deduplicated_by_path(self):
        state = {
            'artifacts': [
                {'kind': 'old', 'path': '/tmp/shared-result.json'},
                {'kind': 'other', 'path': '/tmp/other.json'},
            ],
        }
        reference = {
            'kind': 'current',
            'path': '/tmp/shared-result.json',
            'description': 'Reused provider result',
        }

        registered = register_artifact_reference(state, reference)
        reference['description'] = 'mutated outside state'

        self.assertEqual(len(state['artifacts']), 2)
        self.assertEqual(registered['description'], 'Reused provider result')
        self.assertEqual(state['artifacts'][-1]['kind'], 'current')

    def test_npz_serialization_preserves_schema_and_named_arrays(self):
        import numpy as np

        payload = serialize_npz_arrays(
            {'energy': np.asarray([-1.0, -0.5])},
            schema='pyscf-agent.test-arrays.v1',
        )

        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            self.assertEqual(archive['schema'].item(), 'pyscf-agent.test-arrays.v1')
            np.testing.assert_allclose(archive['energy'], [-1.0, -0.5])


if __name__ == '__main__':
    unittest.main()
