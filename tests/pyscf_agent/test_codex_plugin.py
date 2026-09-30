from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from tests.pyscf_agent.test_mcp_server import HAS_MCP, HAS_PYSCF
from tests.pyscf_agent import test_mcp_studies as study_tests


PLUGIN = Path(__file__).resolve().parents[2] / 'plugins' / 'pyscf-agent'
spec = importlib.util.spec_from_file_location('codex_plugin_configure', PLUGIN / 'scripts' / 'configure.py')
configure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configure)


class CodexPluginLauncherTests(unittest.TestCase):
    def test_cached_launcher_preserves_quoted_paths_and_protocol_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cached = root / 'installed plugin'
            shutil.copytree(PLUGIN, cached)
            python = root / "python with ' spaces"
            python.write_text(f'#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
            python.chmod(0o755)
            config = root / "local settings' file.sh"
            runs = root / "runs with spaces; $(must-not-execute)"
            arguments = ['--work-dir', str(runs), '--executor', 'local']
            configure.write_configuration(config, arguments, python=str(python))
            manifest = json.loads((cached / '.mcp.json').read_text())['mcpServers']['pyscf-agent']
            result = subprocess.run(
                [str(cached / manifest['command']), '--help'],
                cwd=cached, env={**os.environ, 'PYSCF_AGENT_CODEX_CONFIG': str(config)},
                capture_output=True, text=True, check=True,
            )
            self.assertEqual(json.loads(result.stdout), ['-m', 'pyscf_agent.mcp_server', *arguments, '--help'])
            self.assertEqual(result.stderr, '')
            self.assertFalse((cached / 'runs').exists())

    def test_missing_configuration_fails_without_writing_to_stdout(self):
        with tempfile.TemporaryDirectory() as root:
            result = subprocess.run(
                [str(PLUGIN / 'scripts' / 'start-mcp.sh')], cwd=root,
                env={**os.environ, 'PYSCF_AGENT_CODEX_CONFIG': str(Path(root) / 'missing.sh')},
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, '')
            self.assertIn('not configured', result.stderr)

    def test_configuration_replacement_is_explicit_and_preserves_venv_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / 'venv' / 'bin' / 'python'
            executable.parent.mkdir(parents=True)
            executable.symlink_to(sys.executable)
            config = root / 'config.sh'
            configure.write_configuration(config, ['--executor', 'local'], python=str(executable))
            original = config.read_text()
            self.assertIn(str(executable), original)
            with self.assertRaises(FileExistsError):
                configure.write_configuration(config, ['--executor', 'remote'])
            self.assertEqual(config.read_text(), original)
            configure.write_configuration(config, ['--executor', 'local', '--work-dir', '/tmp/new'], force=True)
            self.assertIn('/tmp/new', config.read_text())

    def test_remote_and_slurm_settings_are_fixed_before_cache_relocation(self):
        args = SimpleNamespace(work_dir='runs/plugin-test', executor='remote',
                               remote_config='config/my remote.ini', remote='amarel',
                               slurm_config='config/my slurm.ini', slurm_profile='small',
                               local_wall_time_seconds=600)
        remote = configure.server_arguments(args)
        self.assertEqual(remote[:4], ['--work-dir', str(Path(args.work_dir).absolute()), '--executor', 'remote'])
        self.assertEqual(remote[4:8], ['--remote-config', str(Path(args.remote_config).resolve()), '--remote', 'amarel'])
        args.executor = 'slurm'
        slurm = configure.server_arguments(args)
        self.assertEqual(slurm[4:8], ['--slurm-config', str(Path(args.slurm_config).resolve()), '--slurm-profile', 'small'])
        self.assertEqual(slurm[-2:], ['--local-wall-time-seconds', '600'])


@unittest.skipUnless(HAS_MCP and HAS_PYSCF and os.environ.get('PYSCF_AGENT_TEST_MCP_NUMERICAL') == '1',
                     'set PYSCF_AGENT_TEST_MCP_NUMERICAL=1 for installed-plugin launcher acceptance')
class CodexPluginNumericalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.cached = self.root / 'installed plugin'
        shutil.copytree(PLUGIN, self.cached)

    def parameters(self, root):
        from mcp import StdioServerParameters
        config = self.root / 'local configuration.sh'
        configure.write_configuration(config, ['--executor', 'local', '--work-dir', root], force=True)
        server = json.loads((self.cached / '.mcp.json').read_text())['mcpServers']['pyscf-agent']
        return StdioServerParameters(
            command=str(self.cached / server['command']), cwd=str(self.cached),
            env={**os.environ, 'PYSCF_AGENT_CODEX_CONFIG': str(config),
                 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
        )

    call = study_tests.MCPStudyNumericalTests.call
    collect_when_ready = study_tests.MCPStudyNumericalTests.collect_when_ready
    test_cached_plugin_study_reconnect_and_subset_retry = (
        study_tests.MCPStudyNumericalTests.test_one_start_finishes_h2_study_after_client_exit_and_subset_retry
    )
