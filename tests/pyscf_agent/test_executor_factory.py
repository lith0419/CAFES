from __future__ import annotations

import unittest
from unittest import mock

from computational_study_agent.cli import build_parser as build_study_parser

from pyscf_agent.executors import LocalExecutor
from pyscf_agent.executors.factory import create_task_executor
from pyscf_agent.pyscf_agent_cli import build_parser as build_calculation_parser
from pyscf_agent.pyscf_agent_web import build_parser as build_web_parser


class ExecutorFactoryTests(unittest.TestCase):
    def test_local_wall_time_selects_supervised_execution(self):
        from pyscf_agent.executors import LocalProcessExecutor
        executor = create_task_executor('local', local_wall_time_seconds=3)
        self.assertIsInstance(executor, LocalProcessExecutor)
        self.assertEqual(executor.describe()['wall_time_seconds'], 3)

    def test_local_is_the_default_calculation_cli_target(self):
        args = build_calculation_parser().parse_args(('request',))

        self.assertEqual(args.executor, 'local')
        self.assertIsInstance(create_task_executor(args.executor), LocalExecutor)

    def test_remote_requires_a_named_profile(self):
        with self.assertRaisesRegex(ValueError, '--remote is required'):
            create_task_executor('remote')

    @mock.patch('pyscf_agent.executors.factory.SshSlurmExecutor.from_config')
    def test_remote_factory_passes_profile_and_config(self, from_config):
        sentinel = object()
        from_config.return_value = sentinel

        result = create_task_executor(
            'remote',
            remote_profile='amarel',
            remote_config='/client/remote.ini',
        )

        self.assertIs(result, sentinel)
        from_config.assert_called_once_with('amarel', '/client/remote.ini')

    def test_web_parser_accepts_remote_execution_target(self):
        args = build_web_parser().parse_args((
            '--executor',
            'remote',
            '--remote',
            'amarel',
        ))

        self.assertEqual(args.executor, 'remote')
        self.assertEqual(args.remote, 'amarel')

    def test_study_parser_accepts_direct_slurm_target(self):
        args = build_study_parser().parse_args((
            'study.json',
            '--executor',
            'slurm',
            '--slurm-config',
            '/cluster/slurm.ini',
            '--slurm-profile',
            'amarel',
        ))

        self.assertEqual(args.executor, 'slurm')
        self.assertEqual(args.slurm_config, '/cluster/slurm.ini')
        self.assertEqual(args.slurm_profile, 'amarel')


if __name__ == '__main__':
    unittest.main()
