from __future__ import annotations

import argparse
import os
from typing import Optional

from .base import TaskExecutor
from .local import LocalExecutor
from .local_process import LocalProcessExecutor
from .slurm import SlurmExecutor
from .ssh_slurm import SshSlurmExecutor


EXECUTION_TARGETS = ('local', 'slurm', 'remote')


def add_executor_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--local-wall-time-seconds', type=float,
        default=os.environ.get('PYSCF_AGENT_LOCAL_WALL_TIME_SECONDS'),
        help='Stop each local calculation after this many seconds, including after a Web restart.',
    )
    parser.add_argument(
        '--executor',
        choices=EXECUTION_TARGETS,
        default=os.environ.get('PYSCF_AGENT_EXECUTOR', 'local'),
        help='Run locally, on this host\'s Slurm scheduler, or through remote SSH.',
    )
    parser.add_argument(
        '--slurm-config',
        default=None,
        help='Path to the server-side direct Slurm INI file.',
    )
    parser.add_argument(
        '--slurm-profile',
        default=os.environ.get('PYSCF_AGENT_SLURM_PROFILE'),
        help='Named [server:<profile>] section from server-slurm.ini.',
    )
    parser.add_argument(
        '--remote-config',
        default=None,
        help='Path to the client-side SSH remote profiles INI file.',
    )
    parser.add_argument(
        '--remote',
        default=os.environ.get('PYSCF_AGENT_REMOTE_PROFILE'),
        help='Remote profile name from remote.ini; required for --executor remote.',
    )


def create_task_executor(
    execution_target: str = 'local',
    *,
    slurm_config: Optional[str] = None,
    slurm_profile: Optional[str] = None,
    remote_config: Optional[str] = None,
    remote_profile: Optional[str] = None,
    local_wall_time_seconds: Optional[float] = None,
) -> TaskExecutor:
    target = str(execution_target or 'local').strip().lower()
    if target == 'local':
        return (LocalProcessExecutor(wall_time_seconds=local_wall_time_seconds)
                if local_wall_time_seconds is not None else LocalExecutor())
    if target == 'slurm':
        return SlurmExecutor.from_config(slurm_config, profile_id=slurm_profile)
    if target == 'remote':
        profile = str(remote_profile or '').strip()
        if not profile:
            raise ValueError('--remote is required when --executor remote is selected')
        return SshSlurmExecutor.from_config(profile, remote_config)
    raise ValueError(
        'Unsupported execution target {0}; choose one of {1}'.format(
            execution_target,
            ', '.join(EXECUTION_TARGETS),
        )
    )


def create_task_executor_from_args(
    args: argparse.Namespace,
) -> TaskExecutor:
    return create_task_executor(
        getattr(args, 'executor', 'local'),
        slurm_config=getattr(args, 'slurm_config', None),
        slurm_profile=getattr(args, 'slurm_profile', None),
        remote_config=getattr(args, 'remote_config', None),
        remote_profile=getattr(args, 'remote', None),
        local_wall_time_seconds=getattr(args, 'local_wall_time_seconds', None),
    )


__all__ = [
    'EXECUTION_TARGETS',
    'add_executor_arguments',
    'create_task_executor',
    'create_task_executor_from_args',
]
