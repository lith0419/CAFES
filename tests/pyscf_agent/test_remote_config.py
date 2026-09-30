from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from pyscf_agent.executors.remote_config import (
    RemoteConfigurationError,
    load_remote_profile,
    resolve_remote_config_path,
)


def _remote_config_text() -> str:
    return '''
[remote:amarel]
host = amarel.example.edu
port = 22
username = scientist
private_key = ~/.ssh/id_ed25519
known_hosts = ~/.ssh/known_hosts
strict_host_key_checking = true
connect_timeout_seconds = 12
remote_python = /cluster/env/bin/python
remote_slurm_config = /cluster/config/server-slurm.ini
remote_slurm_profile = amarel
expected_cluster_id = amarel
runtime_identity_file = runtime-identity.json
require_runtime_match = true
local_source_root = ..
deployment_base_python = /cluster/env/bin/python
deployment_base_slurm_profile = amarel
poll_interval_seconds = 9
wait_timeout_seconds = 0
report_propagation_timeout_seconds = 75
control_persist_seconds = 300
'''


class RemoteConfigurationTests(unittest.TestCase):
    def test_loads_named_remote_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'remote.ini'
            path.write_text(_remote_config_text(), encoding='utf-8')
            profile = load_remote_profile('amarel', str(path))

        self.assertEqual(profile.profile_id, 'amarel')
        self.assertEqual(profile.host, 'amarel.example.edu')
        self.assertEqual(profile.remote_python, '/cluster/env/bin/python')
        self.assertEqual(
            profile.remote_slurm_config,
            '/cluster/config/server-slurm.ini',
        )
        self.assertEqual(profile.remote_slurm_profile, 'amarel')
        self.assertEqual(profile.expected_cluster_id, 'amarel')
        self.assertTrue(profile.require_runtime_match)
        self.assertEqual(
            profile.runtime_identity_file,
            str((path.parent / 'runtime-identity.json').resolve()),
        )
        self.assertEqual(profile.local_source_root, str(path.parent.parent.resolve()))
        self.assertEqual(profile.deployment_base_python, '/cluster/env/bin/python')
        self.assertEqual(profile.deployment_base_slurm_profile, 'amarel')
        self.assertIsNone(profile.wait_timeout_seconds)
        self.assertEqual(profile.report_propagation_timeout_seconds, 75)
        self.assertEqual(profile.control_persist_seconds, 300)

    def test_each_remote_profile_selects_a_named_server_section(self):
        text = _remote_config_text() + '''

[remote:secondary]
host = secondary.example.edu
username = scientist
private_key = ~/.ssh/id_ed25519
remote_python = /secondary/env/bin/python
remote_slurm_config = /secondary/config/server-slurm.ini
remote_slurm_profile = secondary
expected_cluster_id = secondary
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'remote.ini'
            path.write_text(text, encoding='utf-8')
            amarel = load_remote_profile('amarel', str(path))
            secondary = load_remote_profile('secondary', str(path))

        self.assertEqual(amarel.remote_slurm_profile, 'amarel')
        self.assertEqual(secondary.remote_slurm_profile, 'secondary')
        self.assertEqual(secondary.expected_cluster_id, 'secondary')

    def test_rejects_missing_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'remote.ini'
            path.write_text(_remote_config_text(), encoding='utf-8')
            with self.assertRaisesRegex(RemoteConfigurationError, 'remote:other'):
                load_remote_profile('other', str(path))

    def test_resolves_environment_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'remote.ini'
            path.write_text(_remote_config_text(), encoding='utf-8')
            previous = os.environ.get('PYSCF_AGENT_REMOTE_CONFIG')
            os.environ['PYSCF_AGENT_REMOTE_CONFIG'] = str(path)
            try:
                self.assertEqual(resolve_remote_config_path(), path.resolve())
            finally:
                if previous is None:
                    os.environ.pop('PYSCF_AGENT_REMOTE_CONFIG', None)
                else:
                    os.environ['PYSCF_AGENT_REMOTE_CONFIG'] = previous


if __name__ == '__main__':
    unittest.main()
