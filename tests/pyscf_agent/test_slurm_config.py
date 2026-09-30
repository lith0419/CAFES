from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from pyscf_agent.executors.slurm import SlurmExecutor
from pyscf_agent.executors.slurm_config import (
    SlurmConfigurationError,
    load_slurm_executor_config,
    resolve_slurm_config_path,
)


def _config_text() -> str:
    return '''
[server:amarel]
cluster_id = amarel
work_root = /cluster/work/pyscf-agent
project_root = /cluster/apps/pyscf-agent
python_executable = /cluster/env/bin/python
shared_filesystem = false
setup_commands =
    module load python
    source /cluster/env/bin/activate
account = chemistry
partition = compute
qos =
time = 03:00:00
nodes = 1
ntasks = 1
cpus_per_task = 6
memory = 12G
gres =
constraint = cpu
extra_sbatch_args = --mail-type=FAIL --requeue
task_command_prefix = time srun
job_name_prefix = chemistry-agent
max_array_size = 250
max_parallel = 16
poll_interval_seconds = 9
wait_timeout_seconds = 0

[profile:amarel:molecular.ccsd]
label = Coupled Cluster
time = 10:00:00
cpus_per_task = 12
memory = 48G
'''


class SlurmConfigurationTests(unittest.TestCase):
    def _write(self, directory: str, text: str) -> Path:
        path = Path(directory) / 'slurm.ini'
        path.write_text(text, encoding='utf-8')
        return path

    def test_loads_direct_configuration_and_resource_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, _config_text())
            config = load_slurm_executor_config(str(path))

        self.assertEqual(config.cluster_id, 'amarel')
        self.assertEqual(config.remote.setup_commands, (
            'module load python',
            'source /cluster/env/bin/activate',
        ))
        self.assertEqual(config.max_array_size, 250)
        self.assertEqual(config.max_parallel, 16)
        self.assertIsNone(config.wait_timeout_seconds)
        self.assertIn('--account=chemistry', config.sbatch_args)
        self.assertIn('--mem=12G', config.sbatch_args)
        self.assertIn('--mail-type=FAIL', config.sbatch_args)
        self.assertEqual(
            config.profile_sbatch_args['molecular.ccsd'],
            ('--time=10:00:00', '--cpus-per-task=12', '--mem=48G'),
        )
        self.assertEqual(config.profile_labels['molecular.ccsd'], 'Coupled Cluster')
        self.assertEqual(config.task_command_prefix, ('time', 'srun'))
        self.assertEqual(config.job_name_prefix, 'chemistry-agent')

    def test_rejects_legacy_unnamed_server_configuration(self):
        legacy = '''
[remote]
work_root = /cluster/work
project_root = /cluster/project
python_executable = /cluster/python

[slurm]
partition = compute
'''
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, legacy)
            with self.assertRaisesRegex(
                SlurmConfigurationError,
                r'named \[server:<profile>\]',
            ):
                load_slurm_executor_config(str(path))

    def test_selects_named_servers_and_their_resource_profiles_from_one_file(self):
        text = '''
[server:amarel]
cluster_id = amarel
work_root = /amarel/work
project_root = /amarel/project
python_executable = /amarel/python
partition = amarel-cpu
cpus_per_task = 16
memory = 32G

[profile:amarel:standard]
label = Amarel Standard
memory = 32G

[server:secondary]
cluster_id = secondary
work_root = /secondary/work
project_root = /secondary/project
python_executable = /secondary/python
partition = secondary-cpu
cpus_per_task = 8
memory = 16G

[profile:secondary:standard]
label = Secondary Standard
memory = 16G
'''
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, text)
            amarel = load_slurm_executor_config(str(path), profile_id='amarel')
            secondary = load_slurm_executor_config(str(path), profile_id='secondary')

        self.assertEqual(amarel.cluster_id, 'amarel')
        self.assertIn('--partition=amarel-cpu', amarel.sbatch_args)
        self.assertEqual(amarel.profile_labels['standard'], 'Amarel Standard')
        self.assertEqual(secondary.cluster_id, 'secondary')
        self.assertIn('--partition=secondary-cpu', secondary.sbatch_args)
        self.assertEqual(
            secondary.profile_labels['standard'],
            'Secondary Standard',
        )

    def test_multiple_named_servers_require_an_explicit_profile(self):
        text = '''
[server:first]
work_root = /first/work
project_root = /first/project
python_executable = /first/python

[server:second]
work_root = /second/work
project_root = /second/project
python_executable = /second/python
'''
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, text)
            with self.assertRaisesRegex(
                SlurmConfigurationError,
                '--slurm-profile',
            ):
                load_slurm_executor_config(str(path))

    def test_resolves_environment_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, _config_text())
            previous = os.environ.get('PYSCF_AGENT_SLURM_CONFIG')
            os.environ['PYSCF_AGENT_SLURM_CONFIG'] = str(path)
            try:
                self.assertEqual(resolve_slurm_config_path(), path.resolve())
            finally:
                if previous is None:
                    os.environ.pop('PYSCF_AGENT_SLURM_CONFIG', None)
                else:
                    os.environ['PYSCF_AGENT_SLURM_CONFIG'] = previous

    def test_builds_direct_executor_from_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, _config_text())
            executor = SlurmExecutor.from_config(
                str(path),
                command_runner=lambda args: None,
            )

        description = executor.describe()
        self.assertEqual(description['cluster_id'], 'amarel')
        self.assertEqual(description['max_array_size'], 250)
        self.assertEqual(description['max_parallel'], 16)
        self.assertTrue(description['supports_resource_profiles'])
        self.assertEqual(description['resource_profiles'], ['molecular.ccsd'])
        self.assertEqual(description['resource_profile_options'], [{
            'id': 'molecular.ccsd',
            'label': 'Coupled Cluster',
            'memory_mb': 48 * 1024,
        }])
        self.assertEqual(description['default_resource_limits'], {
            'memory_mb': 12 * 1024,
        })
        self.assertEqual(description['task_command_prefix'], ['time', 'srun'])
        self.assertEqual(description['job_name_prefix'], 'chemistry-agent')
        self.assertEqual(description['configuration_source'], str(path.resolve()))


if __name__ == '__main__':
    unittest.main()
