from __future__ import annotations

import configparser
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from pyscf_agent.remote.deployment import deploy_worktree_remote
from pyscf_agent.remote.deployment_server import configure_remote_release
from pyscf_agent.runtime_identity import build_runtime_identity


class _Result:
    def __init__(self, stdout=b'', stderr=b'', returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _remote_ini(path: Path) -> None:
    path.write_text('''
[remote:source]
host = amarel.example.edu
port = 22
username = scientist
private_key = /keys/id
known_hosts = /keys/known_hosts
strict_host_key_checking = true
remote_python = /cluster/env/bin/python
remote_slurm_config = /cluster/server-slurm.ini
remote_slurm_profile = amarel
expected_cluster_id = amarel
''', encoding='utf-8')


class RemoteDeploymentTests(unittest.TestCase):
    def test_server_adds_release_profile_without_changing_stable_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / 'release'
            source = release / 'source'
            source.mkdir(parents=True)
            (source / 'pyproject.toml').write_text('[project]\nname="demo"\n', encoding='utf-8')
            server = root / 'server.ini'
            server.write_text('''# keep this comment
[server:amarel]
cluster_id = amarel
work_root = /scratch/agent
project_root = /stable/source
python_executable = /stable/bin/python
shared_filesystem = true

[profile:amarel:standard]
label = Standard
memory = 8G
''', encoding='utf-8')
            identity = build_runtime_identity(
                source,
                environment_id='worktree-5872',
                release_id='release-001',
            )

            result = configure_remote_release(
                server_config=str(server),
                base_profile='amarel',
                server_profile='amarel-worktree-5872',
                environment_id='worktree-5872',
                release_root=str(release),
                base_python='/cluster/env/bin/python',
                identity=identity,
            )

            text = server.read_text(encoding='utf-8')
            self.assertIn('# keep this comment', text)
            parser = configparser.ConfigParser(interpolation=None)
            parser.read_string(text)
            self.assertEqual(parser['server:amarel']['project_root'], '/stable/source')
            branch = parser['server:amarel-worktree-5872']
            self.assertEqual(branch['project_root'], str(source.resolve()))
            self.assertEqual(branch['python_executable'], result['python_wrapper'])
            self.assertEqual(
                parser['profile:amarel-worktree-5872:standard']['memory'],
                '8G',
            )

    def test_deploy_writes_private_worktree_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'pyproject.toml').write_text('[project]\nname="demo"\n', encoding='utf-8')
            package = root / 'demo'
            package.mkdir()
            (package / '__init__.py').write_text('VALUE = 1\n', encoding='utf-8')
            source_config = root / 'source-remote.ini'
            _remote_ini(source_config)
            target_config = root / '.pyscf-agent' / 'remote.ini'
            calls = []

            def runner(args, payload):
                calls.append((list(args), payload))
                if len(calls) == 1:
                    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
                        self.assertIn('demo/__init__.py', archive.getnames())
                    return _Result()
                response = {
                    'schema': 'pyscf-agent.remote-release.v1',
                    'release_id': 'release-001',
                    'release_root': '/cluster/releases/release-001',
                    'python_wrapper': '/cluster/releases/release-001/bin/python',
                    'server_profile': 'amarel-worktree-5872',
                }
                return _Result(stdout=json.dumps(response).encode('utf-8'))

            report = deploy_worktree_remote(
                source_root=str(root),
                environment_id='worktree-5872',
                source_profile='source',
                source_remote_config=str(source_config),
                target_remote_config=str(target_config),
                target_profile='amarel',
                server_profile='amarel-worktree-5872',
                release_id='release-001',
                ssh_runner=runner,
                verify=False,
            )

            parser = configparser.ConfigParser(interpolation=None)
            parser.read(str(target_config), encoding='utf-8')
            profile = parser['remote:amarel']
            self.assertEqual(profile['remote_python'], '/cluster/releases/release-001/bin/python')
            self.assertEqual(profile['remote_slurm_profile'], 'amarel-worktree-5872')
            self.assertEqual(profile['require_runtime_match'], 'true')
            self.assertEqual(profile['deployment_base_python'], '/cluster/env/bin/python')
            self.assertEqual(profile['deployment_base_slurm_profile'], 'amarel')
            self.assertEqual(report['binding']['release_id'], 'release-001')
            self.assertTrue((target_config.parent / 'runtime-identity.json').is_file())
            self.assertTrue((target_config.parent / 'runtime-binding.json').is_file())

    def test_redeploy_resolves_real_python_instead_of_chaining_release_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'pyproject.toml').write_text(
                '[project]\nname="demo"\n',
                encoding='utf-8',
            )
            source_config = root / '.pyscf-agent' / 'remote.ini'
            source_config.parent.mkdir()
            source_config.write_text('''
[remote:amarel]
host = amarel.example.edu
username = scientist
remote_python = /cluster/releases/old/bin/python
remote_slurm_config = /cluster/server-slurm.ini
remote_slurm_profile = amarel-worktree-5872
require_runtime_match = true
runtime_identity_file = runtime-identity.json
local_source_root = ../..
''', encoding='utf-8')
            calls = []

            def runner(args, payload):
                calls.append((list(args), payload))
                command = args[-1]
                if 'print(sys.executable)' in command:
                    return _Result(stdout=b'/cluster/env/bin/python\n')
                if payload:
                    return _Result()
                response = {
                    'schema': 'pyscf-agent.remote-release.v1',
                    'release_id': 'release-002',
                    'release_root': '/cluster/releases/release-002',
                    'python_wrapper': '/cluster/releases/release-002/bin/python',
                    'server_profile': 'amarel-worktree-5872',
                }
                return _Result(stdout=json.dumps(response).encode('utf-8'))

            deploy_worktree_remote(
                source_root=str(root),
                environment_id='worktree-5872',
                source_profile='amarel',
                source_remote_config=str(source_config),
                target_remote_config=str(source_config),
                target_profile='amarel',
                server_profile='amarel-worktree-5872',
                release_id='release-002',
                ssh_runner=runner,
                verify=False,
            )

            configure_command = calls[-1][0][-1]
            self.assertIn('--base-profile amarel', configure_command)
            self.assertIn('--base-python /cluster/env/bin/python', configure_command)
            parser = configparser.ConfigParser(interpolation=None)
            parser.read(str(source_config), encoding='utf-8')
            profile = parser['remote:amarel']
            self.assertEqual(
                profile['deployment_base_python'],
                '/cluster/env/bin/python',
            )
            self.assertEqual(profile['deployment_base_slurm_profile'], 'amarel')


if __name__ == '__main__':
    unittest.main()
