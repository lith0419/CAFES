from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pyscf_agent.runtime_identity import (
    RuntimeIdentity,
    RuntimeIdentityError,
    assert_source_matches_identity,
    build_runtime_identity,
    load_runtime_identity,
    source_fingerprint,
)


class RuntimeIdentityTests(unittest.TestCase):
    def test_archive_identity_without_git_still_detects_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'pyproject.toml').write_text('[project]\nname="demo"\n')
            source = root / 'module.py'
            source.write_text('VALUE = 1\n')
            expected = source_fingerprint(root, files=[Path('module.py'), Path('pyproject.toml')])
            with mock.patch('pyscf_agent.runtime_identity.subprocess.run', side_effect=FileNotFoundError('git')):
                identity = build_runtime_identity(root, environment_id='archive', release_id='test')
                self.assertEqual(identity.source_revision, 'source-snapshot')
                self.assertEqual(identity.source_fingerprint, expected)
                assert_source_matches_identity(identity, root)
                (root / 'runs').mkdir()
                (root / 'runs' / 'output.log').write_text('ignored')
                assert_source_matches_identity(identity, root)
                source.write_text('VALUE = 2\n')
                with self.assertRaisesRegex(RuntimeIdentityError, 'redeploy'):
                    assert_source_matches_identity(identity, root)

    def test_build_load_and_match_source_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'pyproject.toml').write_text('[project]\nname="demo"\n', encoding='utf-8')
            package = root / 'demo'
            package.mkdir()
            (package / '__init__.py').write_text('VALUE = 1\n', encoding='utf-8')
            ignored = root / 'runs'
            ignored.mkdir()
            (ignored / 'output.log').write_text('ignored\n', encoding='utf-8')
            identity = build_runtime_identity(
                root,
                environment_id='worktree-5872',
                release_id='release-001',
            )
            private = root / '.pyscf-agent'
            private.mkdir()
            path = private / 'identity.json'
            path.write_text(json.dumps(identity.to_dict()), encoding='utf-8')

            loaded = load_runtime_identity(path, required=True)
            self.assertEqual(loaded, identity)
            self.assertTrue(identity.source_fingerprint.startswith('blake2b-256:'))
            assert_source_matches_identity(identity, root)

            (ignored / 'output.log').write_text('changed but ignored\n', encoding='utf-8')
            assert_source_matches_identity(identity, root)

    def test_changed_source_requires_redeployment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'pyproject.toml').write_text('[project]\nname="demo"\n', encoding='utf-8')
            source = root / 'module.py'
            source.write_text('VALUE = 1\n', encoding='utf-8')
            identity = build_runtime_identity(
                root,
                environment_id='feature',
                release_id='release-001',
            )
            source.write_text('VALUE = 2\n', encoding='utf-8')

            with self.assertRaisesRegex(RuntimeIdentityError, 'redeploy this worktree'):
                assert_source_matches_identity(identity, root)

    def test_rejects_wrong_identity_schema(self):
        with self.assertRaisesRegex(RuntimeIdentityError, 'Unsupported runtime identity'):
            RuntimeIdentity.from_dict({'schema': 'other'})


if __name__ == '__main__':
    unittest.main()
