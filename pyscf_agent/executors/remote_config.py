from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_REMOTE_CONFIG_PATH = Path('.pyscf-agent') / 'remote.ini'

_PLACEHOLDERS = {
    'change_me',
    'server.example.edu',
    'your_username',
    '/path/to/conda/env/bin/python',
    '/path/to/slurm.ini',
    '/path/to/.pyscf-agent/server-slurm.ini',
}


class RemoteConfigurationError(ValueError):
    """Raised when a client-side SSH execution profile is incomplete."""


@dataclass(frozen=True)
class SshRemoteProfile:
    source_path: str
    profile_id: str
    host: str
    port: int
    username: str
    private_key: Optional[str]
    known_hosts: Optional[str]
    strict_host_key_checking: bool
    connect_timeout_seconds: float
    remote_python: str
    remote_slurm_config: str
    poll_interval_seconds: float
    wait_timeout_seconds: Optional[float]
    report_propagation_timeout_seconds: float
    control_persist_seconds: int
    remote_slurm_profile: Optional[str] = None
    expected_cluster_id: Optional[str] = None
    runtime_identity_file: Optional[str] = None
    require_runtime_match: bool = False
    local_source_root: Optional[str] = None
    deployment_base_python: Optional[str] = None
    deployment_base_slurm_profile: Optional[str] = None


def resolve_remote_config_path(path: Optional[str] = None) -> Path:
    configured = str(
        path or os.environ.get('PYSCF_AGENT_REMOTE_CONFIG') or ''
    ).strip()
    candidate = Path(configured) if configured else DEFAULT_REMOTE_CONFIG_PATH
    return candidate.expanduser().resolve()


def _required(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
) -> str:
    value = parser.get(section, option, fallback='').strip()
    if not value or value.lower() in _PLACEHOLDERS:
        raise RemoteConfigurationError(
            '[{0}] {1} must be set in the remote configuration file'.format(
                section,
                option,
            )
        )
    return value


def _optional_local_path(value: str) -> Optional[str]:
    normalized = str(value or '').strip()
    if not normalized:
        return None
    return os.path.abspath(os.path.expandvars(os.path.expanduser(normalized)))


def _optional_config_path(value: str, source: Path) -> Optional[str]:
    normalized = str(value or '').strip()
    if not normalized:
        return None
    expanded = Path(os.path.expandvars(os.path.expanduser(normalized)))
    candidate = expanded if expanded.is_absolute() else source.parent / expanded
    return str(candidate.resolve())


def _positive_int(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    fallback: int,
    *,
    allow_zero: bool = False,
) -> int:
    try:
        value = parser.getint(section, option, fallback=fallback)
    except ValueError as exc:
        raise RemoteConfigurationError(
            '[{0}] {1} must be an integer'.format(section, option)
        ) from exc
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise RemoteConfigurationError(
            '[{0}] {1} must be at least {2}'.format(section, option, minimum)
        )
    return value


def _positive_float(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    fallback: float,
    *,
    allow_zero: bool = False,
) -> float:
    try:
        value = parser.getfloat(section, option, fallback=fallback)
    except ValueError as exc:
        raise RemoteConfigurationError(
            '[{0}] {1} must be numeric'.format(section, option)
        ) from exc
    minimum = 0.0 if allow_zero else 0.000001
    if value < minimum:
        raise RemoteConfigurationError(
            '[{0}] {1} must be {2}zero'.format(
                section,
                option,
                'at least ' if allow_zero else 'greater than ',
            )
        )
    return value


def load_remote_profile(
    profile_id: str,
    path: Optional[str] = None,
) -> SshRemoteProfile:
    source = resolve_remote_config_path(path)
    if not source.is_file():
        raise RemoteConfigurationError(
            'Remote configuration file was not found: {0}'.format(source)
        )
    normalized_profile = str(profile_id or '').strip()
    if not normalized_profile:
        raise RemoteConfigurationError('A remote profile name is required')
    section = 'remote:{0}'.format(normalized_profile)
    parser = configparser.ConfigParser(interpolation=None)
    try:
        loaded = parser.read(str(source), encoding='utf-8')
    except configparser.Error as exc:
        raise RemoteConfigurationError(
            'Remote configuration is not valid INI: {0}'.format(source)
        ) from exc
    if not loaded or not parser.has_section(section):
        raise RemoteConfigurationError(
            'Remote configuration is missing section [{0}]'.format(section)
        )

    strict = parser.getboolean(
        section,
        'strict_host_key_checking',
        fallback=True,
    )
    known_hosts = _optional_local_path(
        parser.get(section, 'known_hosts', fallback='~/.ssh/known_hosts')
    )
    if strict and not known_hosts:
        raise RemoteConfigurationError(
            '[{0}] known_hosts is required when strict host-key checking is enabled'.format(
                section
            )
        )
    timeout = _positive_float(
        parser,
        section,
        'wait_timeout_seconds',
        0.0,
        allow_zero=True,
    )
    return SshRemoteProfile(
        source_path=str(source),
        profile_id=normalized_profile,
        host=_required(parser, section, 'host'),
        port=_positive_int(parser, section, 'port', 22),
        username=_required(parser, section, 'username'),
        private_key=_optional_local_path(
            parser.get(section, 'private_key', fallback='')
        ),
        known_hosts=known_hosts,
        strict_host_key_checking=strict,
        connect_timeout_seconds=_positive_float(
            parser,
            section,
            'connect_timeout_seconds',
            15.0,
        ),
        remote_python=_required(parser, section, 'remote_python'),
        remote_slurm_config=_required(parser, section, 'remote_slurm_config'),
        poll_interval_seconds=_positive_float(
            parser,
            section,
            'poll_interval_seconds',
            15.0,
        ),
        wait_timeout_seconds=None if timeout == 0 else timeout,
        report_propagation_timeout_seconds=_positive_float(
            parser,
            section,
            'report_propagation_timeout_seconds',
            120.0,
        ),
        control_persist_seconds=_positive_int(
            parser,
            section,
            'control_persist_seconds',
            600,
            allow_zero=True,
        ),
        remote_slurm_profile=(
            parser.get(section, 'remote_slurm_profile', fallback='').strip()
            or normalized_profile
        ),
        expected_cluster_id=(
            parser.get(section, 'expected_cluster_id', fallback='').strip()
            or None
        ),
        runtime_identity_file=_optional_config_path(
            parser.get(section, 'runtime_identity_file', fallback=''),
            source,
        ),
        require_runtime_match=parser.getboolean(
            section,
            'require_runtime_match',
            fallback=False,
        ),
        local_source_root=_optional_config_path(
            parser.get(section, 'local_source_root', fallback=''),
            source,
        ),
        deployment_base_python=(
            parser.get(section, 'deployment_base_python', fallback='').strip()
            or None
        ),
        deployment_base_slurm_profile=(
            parser.get(
                section,
                'deployment_base_slurm_profile',
                fallback='',
            ).strip()
            or None
        ),
    )


__all__ = [
    'DEFAULT_REMOTE_CONFIG_PATH',
    'RemoteConfigurationError',
    'SshRemoteProfile',
    'load_remote_profile',
    'resolve_remote_config_path',
]
