from __future__ import annotations

import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pyscf_agent import configure


class EnvironmentConfigureTests(unittest.TestCase):
    def test_initializes_private_sourceable_llm_environment_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / 'llm.env.template'
            destination = root / 'llm.env'
            template.write_text(
                'export PYSCF_AGENT_LLM_MODEL=""\n',
                encoding='utf-8',
            )

            output, created = configure.initialize_llm_environment(
                destination,
                template_path=template,
            )
            destination.write_text('private-value\n', encoding='utf-8')
            second_output, second_created = configure.initialize_llm_environment(
                destination,
                template_path=template,
            )

            self.assertTrue(created)
            self.assertFalse(second_created)
            self.assertEqual(output, destination.resolve())
            self.assertEqual(second_output, destination.resolve())
            self.assertEqual(destination.read_text(encoding='utf-8'), 'private-value\n')
            self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)

    def test_detects_conda_from_prefix_metadata(self):
        with mock.patch.object(Path, 'is_dir', return_value=True):
            self.assertTrue(configure.is_conda_environment(
                prefix='/env',
                environ={},
            ))

    def test_builds_conda_binary_then_editable_install_commands(self):
        commands = configure.install_commands(
            conda_mode=True,
            conda_command='/opt/conda/bin/conda',
            conda_packages=('pillow', 'spglib', 'h5py'),
        )

        self.assertEqual(commands[0][:2], (
            '/opt/conda/bin/conda',
            'install',
        ))
        self.assertIn('pillow', commands[0])
        self.assertIn('spglib', commands[0])
        self.assertIn('h5py', commands[0])
        self.assertIn('--freeze-installed', commands[0])
        self.assertIn('--strict-channel-priority', commands[0])
        self.assertIn('--override-channels', commands[0])
        self.assertIn('current_repodata.json', commands[0])
        self.assertIn('--no-deps', commands[-1])
        self.assertIn('--no-build-isolation', commands[-1])
        self.assertIn('-e', commands[-1])

    def test_non_conda_install_prefers_binary_dependencies(self):
        commands = configure.install_commands(conda_mode=False)

        self.assertEqual(len(commands), 1)
        self.assertIn('--prefer-binary', commands[0])
        self.assertIn('-e', commands[0])
        self.assertNotIn('--no-deps', commands[0])

    def test_skips_conda_solver_when_binary_dependencies_exist(self):
        commands = configure.install_commands(
            conda_mode=True,
            conda_command='/opt/conda/bin/conda',
            conda_packages=(),
        )

        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0][1:4], ('-m', 'pip', 'install'))

    def test_can_install_missing_conda_packages_one_by_one(self):
        commands = configure.install_commands(
            conda_mode=True,
            conda_command='/opt/conda/bin/conda',
            conda_packages=('pillow', 'spglib', 'h5py'),
            one_by_one=True,
        )

        self.assertEqual(len(commands), 5)
        self.assertEqual(commands[0][-1], 'pillow')
        self.assertEqual(commands[1][-1], 'spglib')
        self.assertEqual(commands[2][-1], 'h5py')

    def test_cli_uses_one_by_one_conda_installs_by_default(self):
        parser = configure.build_parser()

        default_args = parser.parse_args(('install',))
        batch_args = parser.parse_args(('install', '--batch'))

        self.assertTrue(default_args.one_by_one)
        self.assertFalse(batch_args.one_by_one)

    def test_cli_supports_local_and_server_roles(self):
        parser = configure.build_parser()

        local_args = parser.parse_args(('check', '--role', 'local'))
        server_args = parser.parse_args(('install', '--role', 'server', '--dry-run'))

        self.assertEqual(local_args.role, 'local')
        self.assertEqual(server_args.role, 'server')

    def test_cli_initializes_llm_environment_in_current_directory(self):
        args = configure.build_parser().parse_args(('init-llm',))

        self.assertEqual(args.command, 'init-llm')
        self.assertEqual(args.output, 'llm.env')

    def test_cli_initializes_packaged_remote_configuration(self):
        args = configure.build_parser().parse_args((
            'init-config',
            'remote',
            '--output',
            'remote.ini',
        ))

        self.assertEqual(args.command, 'init-config')
        self.assertEqual(args.kind, 'remote')
        self.assertEqual(args.output, 'remote.ini')

    def test_packaged_templates_are_available_without_source_config_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            remote_path, _ = configure.initialize_configuration(
                'remote',
                Path(directory) / 'remote.ini',
            )
            server_path, _ = configure.initialize_configuration(
                'server-slurm',
                Path(directory) / 'server-slurm.ini',
            )

            self.assertIn('[remote:amarel]', remote_path.read_text(encoding='utf-8'))
            self.assertIn('[server:amarel]', server_path.read_text(encoding='utf-8'))

    @mock.patch('pyscf_agent.configure.shutil.which')
    def test_uses_configured_miniconda_instead_of_mamba(self, which):
        which.return_value = '/opt/miniconda/bin/conda'

        executable = configure.conda_executable(environ={
            'CONDA_EXE': '/srv/miniconda/bin/conda',
        })

        self.assertEqual(executable, '/srv/miniconda/bin/conda')
        which.assert_not_called()

    @mock.patch('pyscf_agent.configure.shutil.which')
    def test_discovers_conda_client_only(self, which):
        which.return_value = '/opt/miniconda/bin/conda'

        executable = configure.conda_executable(environ={})

        self.assertEqual(executable, '/opt/miniconda/bin/conda')
        which.assert_called_once_with('conda')


if __name__ == '__main__':
    unittest.main()
