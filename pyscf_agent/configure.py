#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.resources
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple, Union


PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parent if (PACKAGE_ROOT.parent / 'pyproject.toml').is_file() else None
PROJECT_ROOT = SOURCE_ROOT or PACKAGE_ROOT
MINIMUM_PYTHON = (3, 10)
TEMPLATE_PACKAGE = 'pyscf_agent.resources'
TEMPLATE_NAMES = {
    'llm': 'llm.env',
    'remote': 'remote.ini',
    'server-slurm': 'server-slurm.ini',
}

CONDA_DEPENDENCIES = (
    ('pillow', 'Pillow'),
    ('spglib', 'spglib'),
    ('h5py', 'h5py'),
    ('pyscf', 'pyscf'),
    ('ase>=3.22', 'ase'),
    ('seekpath>=2.1,<3', 'seekpath'),
    ('matplotlib>=3.7', 'matplotlib'),
)
PIP_PACKAGES = (
    'setuptools>=61',
    'wheel',
    'build>=1.2,<2',
    'langgraph>=0.6.11,<1.0',
)
REQUIRED_IMPORTS = (
    ('PySCF', 'pyscf', 'pyscf'),
    ('h5py', 'h5py', 'h5py'),
    ('spglib', 'spglib', 'spglib'),
    ('Pillow', 'PIL', 'Pillow'),
    ('ASE', 'ase', 'ase'),
    ('SeeK-path', 'seekpath', 'seekpath'),
    ('Matplotlib', 'matplotlib', 'matplotlib'),
    ('LangGraph', 'langgraph', 'langgraph'),
)
SLURM_COMMANDS = ('sbatch', 'srun', 'squeue', 'sacct', 'scancel')
SSH_COMMANDS = ('ssh',)
INSTALL_ROLES = ('local', 'server')


@dataclass(frozen=True)
class EnvironmentCheck:
    status: str
    name: str
    detail: str
    required: bool = True

    @property
    def passed(self) -> bool:
        return self.status in ('ok', 'info')


def is_conda_environment(
    *,
    prefix: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    values = environ if environ is not None else os.environ
    active_prefix = str(values.get('CONDA_PREFIX') or '').strip()
    candidate = Path(prefix or sys.prefix).expanduser()
    return bool(active_prefix or (candidate / 'conda-meta').is_dir())


def conda_executable(
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    values = environ if environ is not None else os.environ
    configured = str(values.get('CONDA_EXE') or '').strip()
    return configured or shutil.which('conda')


def missing_conda_packages() -> Tuple[str, ...]:
    missing = []
    for specification, distribution in CONDA_DEPENDENCIES:
        try:
            importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing.append(specification)
    return tuple(missing)


def _distribution_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return 'unknown version'


def check_environment(
    *,
    check_slurm: bool = False,
    role: str = 'local',
) -> List[EnvironmentCheck]:
    checks: List[EnvironmentCheck] = []
    current_python = sys.version_info[:2]
    checks.append(EnvironmentCheck(
        status='ok' if current_python >= MINIMUM_PYTHON else 'missing',
        name='Python',
        detail='{0}.{1} at {2}'.format(
            current_python[0],
            current_python[1],
            sys.executable,
        ),
    ))
    conda_active = is_conda_environment()
    checks.append(EnvironmentCheck(
        status='info',
        name='Environment',
        detail=(
            'Conda environment at {0}'.format(sys.prefix)
            if conda_active
            else 'Python environment at {0}'.format(sys.prefix)
        ),
        required=False,
    ))
    metadata_path = SOURCE_ROOT / 'pyproject.toml' if SOURCE_ROOT else None
    try:
        installed_metadata = importlib.metadata.metadata('pyscf-agent')
    except importlib.metadata.PackageNotFoundError:
        installed_metadata = None
    metadata_available = bool(metadata_path and metadata_path.is_file()) or installed_metadata is not None
    metadata_detail = (
        str(metadata_path)
        if metadata_path and metadata_path.is_file()
        else 'installed distribution metadata'
    )
    checks.append(EnvironmentCheck(
        status='ok' if metadata_available else 'missing',
        name='Project metadata',
        detail=metadata_detail,
    ))

    for display_name, module_name, distribution in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            checks.append(EnvironmentCheck(
                status='missing',
                name=display_name,
                detail='{0}: {1}'.format(type(exc).__name__, exc),
            ))
        else:
            checks.append(EnvironmentCheck(
                status='ok',
                name=display_name,
                detail=_distribution_version(distribution),
            ))

    try:
        installed_version = importlib.metadata.version('pyscf-agent')
        agent_module = importlib.import_module('pyscf_agent')
    except Exception as exc:
        checks.append(EnvironmentCheck(
            status='missing',
            name='pyscf-agent',
            detail='{0}: {1}'.format(type(exc).__name__, exc),
        ))
    else:
        checks.append(EnvironmentCheck(
            status='ok',
            name='pyscf-agent',
            detail='{0} at {1}'.format(
                installed_version,
                getattr(agent_module, '__file__', 'unknown location'),
            ),
        ))

    normalized_role = str(role or 'local').strip().lower()
    if normalized_role not in INSTALL_ROLES:
        raise ValueError('role must be local or server')
    command_checks = SLURM_COMMANDS if check_slurm or normalized_role == 'server' else SSH_COMMANDS
    command_prefix = 'Slurm' if check_slurm or normalized_role == 'server' else 'Remote client'
    for command in command_checks:
        resolved = shutil.which(command)
        checks.append(EnvironmentCheck(
            status='ok' if resolved else 'missing',
            name='{0} {1}'.format(command_prefix, command),
            detail=resolved or 'command not found',
        ))
    return checks


def install_commands(
    *,
    conda_mode: Optional[bool] = None,
    conda_command: Optional[str] = None,
    conda_packages: Optional[Sequence[str]] = None,
    one_by_one: bool = False,
) -> List[Tuple[str, ...]]:
    use_conda = is_conda_environment() if conda_mode is None else conda_mode
    python = sys.executable
    if not use_conda:
        if SOURCE_ROOT is None:
            return []
        return [(
            python,
            '-m',
            'pip',
            'install',
            '--prefer-binary',
            '-e',
            str(SOURCE_ROOT),
        )]

    executable = conda_command or conda_executable()
    if not executable:
        raise RuntimeError(
            'A Conda environment was detected, but the conda executable was not found.'
        )
    required_conda_packages = tuple(
        missing_conda_packages()
        if conda_packages is None
        else conda_packages
    )
    commands: List[Tuple[str, ...]] = []
    if required_conda_packages:
        metadata_args = (
            ('--repodata-fn', 'current_repodata.json')
            if Path(executable).name == 'conda'
            else ()
        )
        package_groups = (
            [(item,) for item in required_conda_packages]
            if one_by_one
            else [required_conda_packages]
        )
        for package_group in package_groups:
            commands.append((
                executable,
                'install',
                '--yes',
                '--prefix',
                sys.prefix,
                '--override-channels',
                '--channel',
                'conda-forge',
                '--strict-channel-priority',
                '--freeze-installed',
                *metadata_args,
                *package_group,
            ))
    commands.append((
        python,
        '-m',
        'pip',
        'install',
        *PIP_PACKAGES,
    ))
    if SOURCE_ROOT is not None:
        commands.append((
            python,
            '-m',
            'pip',
            'install',
            '-e',
            str(SOURCE_ROOT),
            '--no-deps',
            '--no-build-isolation',
        ))
    return commands


def _packaged_template_text(template_name: str) -> str:
    if template_name not in TEMPLATE_NAMES.values():
        raise ValueError('Unknown packaged configuration template: {0}'.format(template_name))
    return (
        importlib.resources.files(TEMPLATE_PACKAGE)
        .joinpath('templates')
        .joinpath(template_name)
        .read_text(encoding='utf-8')
    )


def initialize_configuration(
    kind: str,
    destination: Union[str, Path],
    *,
    template_path: Optional[Union[str, Path]] = None,
    private: bool = True,
) -> Tuple[Path, bool]:
    """Create one packaged configuration file without overwriting user data."""
    normalized_kind = str(kind or '').strip().lower()
    if normalized_kind not in TEMPLATE_NAMES:
        raise ValueError('Configuration kind must be one of: {0}'.format(
            ', '.join(sorted(TEMPLATE_NAMES))
        ))
    if template_path is None:
        content = _packaged_template_text(TEMPLATE_NAMES[normalized_kind])
    else:
        template = Path(template_path).expanduser().resolve()
        if not template.is_file():
            raise FileNotFoundError(
                'Configuration template was not found: {0}'.format(template)
            )
        content = template.read_text(encoding='utf-8')

    output = Path(destination).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output = output.resolve()
    if output.exists():
        return output, False

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding='utf-8')
    if private:
        output.chmod(0o600)
    return output, True


def initialize_llm_environment(
    destination: Union[str, Path] = 'llm.env',
    *,
    template_path: Optional[Union[str, Path]] = None,
) -> Tuple[Path, bool]:
    """Create a private, sourceable LLM environment file without overwriting it."""
    return initialize_configuration(
        'llm',
        destination,
        template_path=template_path,
        private=True,
    )


def _print_checks(checks: Sequence[EnvironmentCheck]) -> bool:
    width = max((len(item.name) for item in checks), default=0)
    for item in checks:
        print('[{0}] {1:<{2}}  {3}'.format(
            item.status.upper(),
            item.name,
            width,
            item.detail,
        ))
    return all(item.passed for item in checks if item.required)


def _run_install(
    *,
    dry_run: bool,
    check_slurm: bool,
    one_by_one: bool,
    role: str,
) -> int:
    try:
        commands = install_commands(one_by_one=one_by_one)
    except RuntimeError as exc:
        print('Configuration error: {0}'.format(exc), file=sys.stderr)
        return 2
    print('Installation plan for {0}:'.format(sys.executable), flush=True)
    for command in commands:
        print('  {0}'.format(shlex.join(command)), flush=True)
    if dry_run:
        return 0
    for command in commands:
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
        if completed.returncode != 0:
            print(
                'Installation stopped after exit code {0}: {1}'.format(
                    completed.returncode,
                    shlex.join(command),
                ),
                file=sys.stderr,
            )
            return completed.returncode
    importlib.invalidate_caches()
    print('\nEnvironment verification:')
    return 0 if _print_checks(check_environment(
        check_slurm=check_slurm,
        role=role,
    )) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Check and install the pyscf-agent runtime environment.'
    )
    subparsers = parser.add_subparsers(dest='command')
    check = subparsers.add_parser('check', help='Check the current environment.')
    check.add_argument(
        '--role',
        choices=INSTALL_ROLES,
        default='local',
        help='Validate a local client/runtime or a server Slurm runtime.',
    )
    check.add_argument(
        '--slurm',
        action='store_true',
        help='Also require the Slurm client commands.',
    )
    install = subparsers.add_parser(
        'install',
        help='Install dependencies and pyscf-agent into the current environment.',
    )
    install.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without modifying the environment.',
    )
    install.add_argument(
        '--role',
        choices=INSTALL_ROLES,
        default='local',
        help='Install and validate the local or server runtime role.',
    )
    install.add_argument(
        '--slurm',
        action='store_true',
        help='Also verify the Slurm client commands after installation.',
    )
    install_strategy = install.add_mutually_exclusive_group()
    install_strategy.add_argument(
        '--one-by-one',
        dest='one_by_one',
        action='store_true',
        help='Install each missing Conda package separately (the default).',
    )
    install_strategy.add_argument(
        '--batch',
        dest='one_by_one',
        action='store_false',
        help='Resolve all missing Conda packages in one transaction.',
    )
    install.set_defaults(one_by_one=True)
    init_llm = subparsers.add_parser(
        'init-llm',
        help='Create a private ./llm.env file from the distributed template.',
    )
    init_llm.add_argument(
        '--output',
        default='llm.env',
        help='Destination environment file (default: ./llm.env).',
    )
    init_config = subparsers.add_parser(
        'init-config',
        help='Create a private remote or server configuration from package resources.',
    )
    init_config.add_argument(
        'kind',
        choices=('remote', 'server-slurm'),
        help='Configuration template to create.',
    )
    init_config.add_argument(
        '--output',
        help='Destination path (default: ~/.pyscf-agent/<template name>).',
    )
    deploy_remote = subparsers.add_parser(
        'deploy-remote',
        help='Deploy this worktree to an immutable matching remote release.',
    )
    deploy_remote.add_argument('--environment-id', required=True)
    deploy_remote.add_argument('--profile', required=True, help='Connection profile from the source remote.ini.')
    deploy_remote.add_argument('--remote-config', help='Source remote.ini containing SSH connection settings.')
    deploy_remote.add_argument('--target-remote-config', help='Per-worktree remote.ini to create.')
    deploy_remote.add_argument('--target-profile', default='amarel')
    deploy_remote.add_argument('--base-server-profile')
    deploy_remote.add_argument('--server-profile')
    deploy_remote.add_argument('--release-id')
    deploy_remote.add_argument('--source-root', default=str(SOURCE_ROOT or Path.cwd()))
    verify_install = subparsers.add_parser(
        'verify-install',
        help='Build a wheel and validate it from a temporary environment.',
    )
    verify_install.add_argument(
        '--wheelhouse',
        help='Optional offline wheelhouse for a fully isolated dependency install.',
    )
    verify_install.add_argument(
        '--output',
        help='Write the JSON verification report here.',
    )
    verify_remote = subparsers.add_parser(
        'verify-remote',
        help='Validate SSH RPC compatibility and optionally submit a Slurm smoke task.',
    )
    verify_remote.add_argument('--profile', required=True, help='Remote profile name.')
    verify_remote.add_argument('--remote-config', help='Path to remote.ini.')
    verify_remote.add_argument(
        '--submit-smoke',
        action='store_true',
        help='Submit and collect a minimal H2/HF Slurm task.',
    )
    verify_remote.add_argument(
        '--submit-batch-smoke',
        action='store_true',
        help='Submit and collect two independent H2/HF tasks through a Slurm job array.',
    )
    verify_remote.add_argument(
        '--output',
        help='Write the JSON verification report here.',
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or 'check'
    if command == 'install':
        return _run_install(
            dry_run=args.dry_run,
            check_slurm=args.slurm,
            one_by_one=args.one_by_one,
            role=args.role,
        )
    if command == 'init-llm':
        try:
            output, created = initialize_llm_environment(args.output)
        except (OSError, ValueError) as exc:
            print('LLM configuration failed: {0}'.format(exc), file=sys.stderr)
            return 2
        if created:
            print('Created private LLM configuration: {0}'.format(output))
            print('Edit it, then run: source {0}'.format(shlex.quote(str(output))))
        else:
            print('LLM configuration already exists; it was not overwritten: {0}'.format(output))
            print('Load it with: source {0}'.format(shlex.quote(str(output))))
        return 0
    if command == 'init-config':
        destination = args.output or str(
            Path.home() / '.pyscf-agent' / TEMPLATE_NAMES[args.kind]
        )
        try:
            output, created = initialize_configuration(
                args.kind,
                destination,
                private=True,
            )
        except (OSError, ValueError) as exc:
            print('Configuration initialization failed: {0}'.format(exc), file=sys.stderr)
            return 2
        if created:
            print('Created private configuration: {0}'.format(output))
        else:
            print('Configuration already exists; it was not overwritten: {0}'.format(output))
        return 0
    if command == 'deploy-remote':
        from pyscf_agent.remote.deployment import deploy_worktree_remote

        try:
            report = deploy_worktree_remote(
                source_root=args.source_root,
                environment_id=args.environment_id,
                source_profile=args.profile,
                source_remote_config=args.remote_config,
                target_remote_config=args.target_remote_config,
                target_profile=args.target_profile,
                base_server_profile=args.base_server_profile,
                server_profile=args.server_profile,
                release_id=args.release_id,
            )
        except Exception as exc:
            print('Remote deployment failed: {0}'.format(exc), file=sys.stderr)
            return 2
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if command == 'verify-install':
        from pyscf_agent.verification import (
            verify_clean_install,
            verify_installed_package,
            write_verification_report,
        )

        report = (
            verify_clean_install(
                str(SOURCE_ROOT),
                wheelhouse=args.wheelhouse,
            )
            if SOURCE_ROOT is not None
            else verify_installed_package()
        )
        if args.output:
            write_verification_report(report, args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report['status'] == 'passed' else 1
    if command == 'verify-remote':
        from pyscf_agent.verification import verify_remote_cluster, write_verification_report

        report = verify_remote_cluster(
            args.profile,
            config_path=args.remote_config,
            submit_smoke=args.submit_smoke,
            submit_batch_smoke=args.submit_batch_smoke,
        )
        if args.output:
            write_verification_report(report, args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report['status'] == 'passed' else 1
    check_slurm = bool(getattr(args, 'slurm', False))
    role = str(getattr(args, 'role', 'local'))
    return 0 if _print_checks(check_environment(
        check_slurm=check_slurm,
        role=role,
    )) else 1


if __name__ == '__main__':
    raise SystemExit(main())
