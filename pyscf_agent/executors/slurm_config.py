from __future__ import annotations

import configparser
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple


DEFAULT_SLURM_CONFIG_PATH = Path('.pyscf-agent') / 'server-slurm.ini'

_SBATCH_OPTIONS = {
    'account': '--account',
    'partition': '--partition',
    'qos': '--qos',
    'time': '--time',
    'nodes': '--nodes',
    'ntasks': '--ntasks',
    'cpus_per_task': '--cpus-per-task',
    'memory': '--mem',
    'gres': '--gres',
    'constraint': '--constraint',
}
_PLACEHOLDERS = {
    'change_me',
    'server.example.edu',
    'your_username',
    '/path/to/pyscf-agent',
    '/scratch/your_username/pyscf-agent',
}


class SlurmConfigurationError(ValueError):
    """Raised when the user-editable Slurm configuration is incomplete."""


@dataclass(frozen=True)
class SlurmRemoteConfig:
    work_root: str
    project_root: str
    python_executable: str
    shared_filesystem: bool
    setup_commands: Tuple[str, ...]


@dataclass(frozen=True)
class SlurmExecutorConfig:
    source_path: str
    cluster_id: str
    remote: SlurmRemoteConfig
    poll_interval_seconds: float
    wait_timeout_seconds: Optional[float]
    max_array_size: int
    max_parallel: Optional[int]
    sbatch_args: Tuple[str, ...]
    profile_sbatch_args: Mapping[str, Tuple[str, ...]]
    profile_labels: Mapping[str, str]
    task_command_prefix: Tuple[str, ...]
    job_name_prefix: str

    def executor_kwargs(self) -> Dict[str, object]:
        return {
            'poll_interval': self.poll_interval_seconds,
            'wait_timeout': self.wait_timeout_seconds,
            'max_array_size': self.max_array_size,
            'max_parallel': self.max_parallel,
            'sbatch_args': self.sbatch_args,
            'profile_sbatch_args': dict(self.profile_sbatch_args),
            'profile_labels': dict(self.profile_labels),
            'script_preamble': self.remote.setup_commands,
            'python_executable': self.remote.python_executable,
            'task_command_prefix': self.task_command_prefix,
            'job_name_prefix': self.job_name_prefix,
        }


def resolve_slurm_config_path(path: Optional[str] = None) -> Path:
    configured = str(path or os.environ.get('PYSCF_AGENT_SLURM_CONFIG') or '').strip()
    candidate = Path(configured) if configured else DEFAULT_SLURM_CONFIG_PATH
    return candidate.expanduser().resolve()


def _required(parser: configparser.ConfigParser, section: str, option: str) -> str:
    value = parser.get(section, option, fallback='').strip()
    if not value or value.lower() in _PLACEHOLDERS:
        raise SlurmConfigurationError(
            '[{0}] {1} must be set in the Slurm configuration file'.format(
                section,
                option,
            )
        )
    return value


def _positive_int(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    *,
    fallback: int,
    allow_zero: bool = False,
) -> int:
    try:
        value = parser.getint(section, option, fallback=fallback)
    except ValueError as exc:
        raise SlurmConfigurationError(
            '[{0}] {1} must be an integer'.format(section, option)
        ) from exc
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise SlurmConfigurationError(
            '[{0}] {1} must be at least {2}'.format(section, option, minimum)
        )
    return value


def _positive_float(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    *,
    fallback: float,
    allow_zero: bool = False,
) -> float:
    try:
        value = parser.getfloat(section, option, fallback=fallback)
    except ValueError as exc:
        raise SlurmConfigurationError(
            '[{0}] {1} must be numeric'.format(section, option)
        ) from exc
    minimum = 0.0 if allow_zero else 0.000001
    if value < minimum:
        raise SlurmConfigurationError(
            '[{0}] {1} must be {2}zero'.format(
                section,
                option,
                'at least ' if allow_zero else 'greater than ',
            )
        )
    return value


def _boolean(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    *,
    fallback: bool,
) -> bool:
    try:
        return parser.getboolean(section, option, fallback=fallback)
    except ValueError as exc:
        raise SlurmConfigurationError(
            '[{0}] {1} must be true or false'.format(section, option)
        ) from exc


def _command_lines(value: str) -> Tuple[str, ...]:
    return tuple(
        line.strip()
        for line in str(value or '').splitlines()
        if line.strip()
    )


def _shell_words(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
) -> Tuple[str, ...]:
    value = parser.get(section, option, fallback='').strip()
    if not value:
        return ()
    try:
        return tuple(shlex.split(value))
    except ValueError as exc:
        raise SlurmConfigurationError(
            '[{0}] {1} is not valid shell-style text'.format(section, option)
        ) from exc


def _sbatch_args(
    parser: configparser.ConfigParser,
    section: str,
) -> Tuple[str, ...]:
    values = []
    for option, flag in _SBATCH_OPTIONS.items():
        value = parser.get(section, option, fallback='').strip()
        if value:
            values.append('{0}={1}'.format(flag, value))
    extra = parser.get(section, 'extra_sbatch_args', fallback='').strip()
    if extra:
        try:
            values.extend(shlex.split(extra))
        except ValueError as exc:
            raise SlurmConfigurationError(
                '[{0}] extra_sbatch_args is not valid shell-style text'.format(
                    section
                )
            ) from exc
    return tuple(values)


def _profile_args(
    parser: configparser.ConfigParser,
    server_profile: Optional[str] = None,
) -> Dict[str, Tuple[str, ...]]:
    profiles: Dict[str, Tuple[str, ...]] = {}
    selected_prefix = (
        'profile:{0}:'.format(server_profile).lower()
        if server_profile
        else 'profile:'
    )
    for section in parser.sections():
        if not section.lower().startswith(selected_prefix):
            continue
        profile_id = section[len(selected_prefix):].strip()
        if not profile_id:
            raise SlurmConfigurationError(
                'Slurm resource profile sections must include a profile name'
            )
        profiles[profile_id] = _sbatch_args(parser, section)
    return profiles


def _profile_labels(
    parser: configparser.ConfigParser,
    server_profile: Optional[str] = None,
) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    selected_prefix = (
        'profile:{0}:'.format(server_profile).lower()
        if server_profile
        else 'profile:'
    )
    for section in parser.sections():
        if not section.lower().startswith(selected_prefix):
            continue
        profile_id = section[len(selected_prefix):].strip()
        label = parser.get(section, 'label', fallback='').strip()
        labels[profile_id] = label or (
            profile_id.replace('.', ' ').replace('_', ' ').replace('-', ' ').title()
        )
    return labels


def _select_server_section(
    parser: configparser.ConfigParser,
    profile_id: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    named_sections = {
        section.split(':', 1)[1].strip(): section
        for section in parser.sections()
        if section.lower().startswith('server:') and section.split(':', 1)[1].strip()
    }
    if not named_sections:
        return None, None

    requested = str(profile_id or '').strip()
    if not requested:
        if len(named_sections) == 1:
            requested = next(iter(named_sections))
        else:
            raise SlurmConfigurationError(
                'server-slurm.ini contains multiple server profiles; choose one '
                'with --slurm-profile. Available profiles: {0}'.format(
                    ', '.join(sorted(named_sections))
                )
            )
    matched = next(
        (
            name
            for name in named_sections
            if name.lower() == requested.lower()
        ),
        None,
    )
    if matched is None:
        raise SlurmConfigurationError(
            'Slurm server profile {0} was not found. Available profiles: {1}'.format(
                requested,
                ', '.join(sorted(named_sections)),
            )
        )
    return named_sections[matched], matched


def load_slurm_executor_config(
    path: Optional[str] = None,
    *,
    profile_id: Optional[str] = None,
) -> SlurmExecutorConfig:
    source = resolve_slurm_config_path(path)
    if not source.is_file():
        raise SlurmConfigurationError(
            'Slurm configuration file was not found: {0}'.format(source)
        )
    parser = configparser.ConfigParser(interpolation=None)
    try:
        loaded = parser.read(str(source), encoding='utf-8')
    except configparser.Error as exc:
        raise SlurmConfigurationError(
            'Slurm configuration is not valid INI: {0}'.format(source)
        ) from exc
    if not loaded:
        raise SlurmConfigurationError(
            'Slurm configuration could not be read: {0}'.format(source)
        )
    server_section, selected_profile = _select_server_section(parser, profile_id)
    if server_section is None:
        raise SlurmConfigurationError(
            'Slurm configuration must contain named [server:<profile>] sections; '
            'SSH connection settings belong in remote.ini'
        )
    remote_section = server_section
    slurm_section = server_section

    shared_filesystem = _boolean(
        parser,
        remote_section,
        'shared_filesystem',
        fallback=False,
    )
    remote = SlurmRemoteConfig(
        work_root=_required(parser, remote_section, 'work_root'),
        project_root=_required(parser, remote_section, 'project_root'),
        python_executable=_required(parser, remote_section, 'python_executable'),
        shared_filesystem=shared_filesystem,
        setup_commands=_command_lines(
            parser.get(remote_section, 'setup_commands', fallback='')
        ),
    )

    timeout = _positive_float(
        parser,
        slurm_section,
        'wait_timeout_seconds',
        fallback=0.0,
        allow_zero=True,
    )
    max_parallel = _positive_int(
        parser,
        slurm_section,
        'max_parallel',
        fallback=8,
        allow_zero=True,
    )
    return SlurmExecutorConfig(
        source_path=str(source),
        cluster_id=(
            parser.get(server_section, 'cluster_id', fallback='').strip()
            or selected_profile
            or source.stem
        ),
        remote=remote,
        poll_interval_seconds=_positive_float(
            parser,
            slurm_section,
            'poll_interval_seconds',
            fallback=15.0,
        ),
        wait_timeout_seconds=None if timeout == 0 else timeout,
        max_array_size=_positive_int(
            parser,
            slurm_section,
            'max_array_size',
            fallback=1000,
        ),
        max_parallel=None if max_parallel == 0 else max_parallel,
        sbatch_args=_sbatch_args(parser, slurm_section),
        profile_sbatch_args=_profile_args(parser, selected_profile),
        profile_labels=_profile_labels(parser, selected_profile),
        task_command_prefix=_shell_words(
            parser,
            slurm_section,
            'task_command_prefix',
        ),
        job_name_prefix=(
            parser.get(
                slurm_section,
                'job_name_prefix',
                fallback='pyscf-agent',
            ).strip()
            or 'pyscf-agent'
        ),
    )


__all__ = [
    'DEFAULT_SLURM_CONFIG_PATH',
    'SlurmConfigurationError',
    'SlurmExecutorConfig',
    'SlurmRemoteConfig',
    'load_slurm_executor_config',
    'resolve_slurm_config_path',
]
