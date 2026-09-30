from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import venv
from datetime import datetime, timezone
from importlib import metadata, resources
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .serialization import json_default
from .schema_contracts import (
    INSTALL_VERIFICATION_SCHEMA,
    PUBLIC_CONTRACT_VERSION,
    REMOTE_VERIFICATION_SCHEMA,
    TASK_REPORT_SCHEMA,
    validate_public_payload,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            'Command failed ({0}): {1}\n{2}'.format(
                completed.returncode,
                ' '.join(command),
                (completed.stderr or completed.stdout).strip(),
            )
        )
    return completed


def _venv_python(environment_root: Path) -> Path:
    if os.name == 'nt':
        return environment_root / 'Scripts' / 'python.exe'
    return environment_root / 'bin' / 'python'


def verify_clean_install(
    project_root: str,
    *,
    wheelhouse: Optional[str] = None,
    python: Optional[str] = None,
) -> Dict[str, Any]:
    """Build and validate the distributable wheel outside the source tree.

    Without a wheelhouse, the temporary environment reuses the current
    interpreter's scientific dependencies but installs pyscf-agent itself only
    from the newly built wheel. Supplying a wheelhouse performs a fully isolated,
    offline dependency installation.
    """

    root = Path(project_root).expanduser().resolve()
    if not (root / 'pyproject.toml').is_file():
        raise ValueError('Project root does not contain pyproject.toml: {0}'.format(root))
    source_python = str(python or sys.executable)
    resolved_wheelhouse = (
        Path(wheelhouse).expanduser().resolve() if wheelhouse else None
    )
    if resolved_wheelhouse is not None and not resolved_wheelhouse.is_dir():
        raise ValueError('Wheelhouse directory was not found: {0}'.format(resolved_wheelhouse))

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='pyscf-agent-install-verify-') as directory:
        workspace = Path(directory)
        wheels = workspace / 'wheels'
        wheels.mkdir()
        _run((
            source_python,
            '-m',
            'build',
            '--wheel',
            '--no-isolation',
            '--outdir',
            str(wheels),
            str(root),
        ), cwd=workspace)
        built_wheels = sorted(wheels.glob('pyscf_agent-*.whl'))
        if len(built_wheels) != 1:
            raise RuntimeError('Expected one pyscf-agent wheel, found {0}'.format(len(built_wheels)))

        environment_root = workspace / 'venv'
        venv.EnvBuilder(
            with_pip=True,
            clear=True,
            system_site_packages=resolved_wheelhouse is None,
        ).create(str(environment_root))
        environment_python = _venv_python(environment_root)
        if resolved_wheelhouse is None:
            # A nested venv inherits the base Python's packages, not those in
            # the calling venv. Add only its package directories, never the
            # checkout or its editable-install hooks. The new wheel stays first.
            package_paths = json.loads(_run((
                source_python, '-c',
                'import json, site; print(json.dumps(site.getsitepackages()))',
            ), cwd=workspace).stdout)
            target_packages = Path(_run((
                str(environment_python), '-c',
                'import sysconfig; print(sysconfig.get_path("purelib"))',
            ), cwd=workspace).stdout.strip())
            (target_packages / 'verification-dependencies.pth').write_text(
                ''.join(str(Path(path).resolve()) + '\n' for path in package_paths),
                encoding='utf-8',
            )
        install_command = [
            str(environment_python),
            '-m',
            'pip',
            'install',
        ]
        dependency_mode = (
            'isolated-wheelhouse'
            if resolved_wheelhouse
            else 'wheel-with-system-dependencies'
        )
        if resolved_wheelhouse is not None:
            install_command.extend((
                '--no-index',
                '--find-links',
                str(resolved_wheelhouse),
            ))
        else:
            install_command.extend(('--no-deps', '--ignore-installed'))
        install_command.append(str(built_wheels[0]))
        _run(install_command, cwd=workspace)

        smoke_root = workspace / 'outside-source-tree'
        smoke_root.mkdir()
        smoke_script = '''
import json
import sys
from pathlib import Path
from importlib import metadata, resources
import pyscf_agent
from pyscf_agent.schema_contracts import public_schema_manifest
from pyscf_agent.benchmarks import run_benchmark_suite
from computational_study_agent.wiki_retriever import load_wiki_pages

manifest = public_schema_manifest()
distribution = metadata.distribution("pyscf-agent")
entry_points = sorted(item.name for item in distribution.entry_points)
data = resources.files("pyscf_agent.benchmarks").joinpath(
    "data/n2_contextual_vqe_comparison.tsv"
)
assistant_page = resources.files("pyscf_agent.web_assets").joinpath(
    "assistant_index.html"
)
planner_page = resources.files("computational_study_agent.web_assets").joinpath(
    "planner_index.html"
)
added_web_assets = all(
    resources.files(package).joinpath(name).is_file()
    for package, name in (
        ("pyscf_agent.web_assets", "study-card.html"),
        ("pyscf_agent.web_assets", "task-monitor.html"),
        ("pyscf_agent.web_assets", "task-monitor.js"),
        ("pyscf_agent.web_assets", "task-monitor.css"),
        ("computational_study_agent.web_assets", "planner-saved-studies.js"),
    )
)
wiki_export = resources.files("computational_study_agent.knowledge").joinpath(
    "curated_wiki.json"
)
template_root = resources.files("pyscf_agent.resources").joinpath("templates")
configuration_templates = all(
    template_root.joinpath(name).is_file()
    for name in ("llm.env", "remote.ini", "server-slurm.ini")
)
wiki_pages = load_wiki_pages()
report = run_benchmark_suite(("n2-dissociation-sto3g", "hubbard-dimer-ed"))
print(json.dumps({
    "version": metadata.version("pyscf-agent"),
    "import_from_wheel": Path(pyscf_agent.__file__).resolve().is_relative_to(
        Path(sys.prefix).resolve()
    ) and Path(pyscf_agent.__file__).resolve() == Path(
        distribution.locate_file("pyscf_agent/__init__.py")
    ).resolve(),
    "contract_version": manifest["contract_version"],
    "schema_count": len(manifest["schemas"]),
    "benchmark_data": data.is_file(),
    "web_assets": assistant_page.is_file() and planner_page.is_file() and added_web_assets,
    "wiki_export": wiki_export.is_file(),
    "configuration_templates": configuration_templates,
    "wiki_page_count": len(wiki_pages),
    "entry_points": entry_points,
    "benchmark_statuses": [item["status"] for item in report["results"]],
    "benchmark_failures": [item for item in report["results"] if item["status"] != "passed"],
}))
'''.strip()
        completed = _run(
            (str(environment_python), '-c', smoke_script),
            cwd=smoke_root,
        )
        try:
            smoke = json.loads(completed.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError('Installed-package smoke output was not valid JSON') from exc
        checks = [
            {
                'name': 'wheel_import_outside_source_tree',
                'passed': bool(smoke.get('version')) and smoke.get('import_from_wheel') is True,
                'value': smoke.get('version'),
            },
            {
                'name': 'public_contract_version',
                'passed': smoke.get('contract_version') == PUBLIC_CONTRACT_VERSION,
                'value': smoke.get('contract_version'),
                'expected': PUBLIC_CONTRACT_VERSION,
            },
            {
                'name': 'packaged_benchmark_data',
                'passed': smoke.get('benchmark_data') is True,
                'value': smoke.get('benchmark_data'),
            },
            {
                'name': 'packaged_web_assets',
                'passed': smoke.get('web_assets') is True,
                'value': smoke.get('web_assets'),
            },
            {
                'name': 'installed_entry_points',
                'passed': {
                    'pyscf-agent',
                    'pyscf-agent-benchmark',
                    'pyscf-agent-configure',
                    'pyscf-agent-web',
                }.issubset(set(smoke.get('entry_points') or [])),
                'value': smoke.get('entry_points'),
            },
            {
                'name': 'packaged_configuration_templates',
                'passed': smoke.get('configuration_templates') is True,
                'value': smoke.get('configuration_templates'),
            },
            {
                'name': 'packaged_curated_wiki',
                'passed': (
                    smoke.get('wiki_export') is True
                    and int(smoke.get('wiki_page_count') or 0) > 0
                ),
                'value': {
                    'resource_present': smoke.get('wiki_export'),
                    'page_count': smoke.get('wiki_page_count'),
                },
            },
            {
                'name': 'installed_smoke_benchmarks',
                'passed': all(
                    status == 'passed'
                    for status in (smoke.get('benchmark_statuses') or [])
                ),
                'value': smoke.get('benchmark_statuses'),
                'failures': smoke.get('benchmark_failures'),
            },
        ]
        return {
            'schema': INSTALL_VERIFICATION_SCHEMA,
            'status': 'passed' if all(item['passed'] for item in checks) else 'failed',
            'created_at': _timestamp(),
            'dependency_mode': dependency_mode,
            'checks': checks,
            'wheel': built_wheels[0].name,
            'environment': {
                'python': platform.python_version(),
                'platform': platform.platform(),
            },
            'duration_seconds': round(time.monotonic() - started, 3),
        }


def verify_installed_package() -> Dict[str, Any]:
    """Validate package metadata, resources, entry points, and smoke benchmarks."""
    from computational_study_agent.wiki_retriever import load_wiki_pages
    from pyscf_agent.schema_contracts import public_schema_manifest
    from pyscf_agent.benchmarks import run_benchmark_suite

    started = time.monotonic()
    distribution = metadata.distribution('pyscf-agent')
    entry_points = {item.name for item in distribution.entry_points}
    manifest = public_schema_manifest()
    benchmark_data = resources.files('pyscf_agent.benchmarks').joinpath(
        'data/n2_contextual_vqe_comparison.tsv'
    )
    assistant_page = resources.files('pyscf_agent.web_assets').joinpath(
        'assistant_index.html'
    )
    planner_page = resources.files('computational_study_agent.web_assets').joinpath(
        'planner_index.html'
    )
    template_root = resources.files('pyscf_agent.resources').joinpath('templates')
    templates_present = all(
        template_root.joinpath(name).is_file()
        for name in ('llm.env', 'remote.ini', 'server-slurm.ini')
    )
    wiki_pages = load_wiki_pages()
    benchmark_report = run_benchmark_suite(
        ('n2-dissociation-sto3g', 'hubbard-dimer-ed')
    )
    checks = [
        {
            'name': 'installed_distribution_metadata',
            'passed': bool(distribution.version),
            'value': distribution.version,
        },
        {
            'name': 'public_contract_version',
            'passed': manifest['contract_version'] == PUBLIC_CONTRACT_VERSION,
            'value': manifest['contract_version'],
            'expected': PUBLIC_CONTRACT_VERSION,
        },
        {
            'name': 'packaged_runtime_resources',
            'passed': bool(
                benchmark_data.is_file()
                and assistant_page.is_file()
                and planner_page.is_file()
                and templates_present
                and wiki_pages
            ),
            'value': {
                'benchmark_data': benchmark_data.is_file(),
                'web_assets': assistant_page.is_file() and planner_page.is_file(),
                'configuration_templates': templates_present,
                'wiki_page_count': len(wiki_pages),
            },
        },
        {
            'name': 'installed_entry_points',
            'passed': {
                'pyscf-agent',
                'pyscf-agent-benchmark',
                'pyscf-agent-configure',
                'pyscf-agent-web',
            }.issubset(entry_points),
            'value': sorted(entry_points),
        },
        {
            'name': 'installed_smoke_benchmarks',
            'passed': all(
                item['status'] == 'passed'
                for item in benchmark_report['results']
            ),
            'value': [item['status'] for item in benchmark_report['results']],
        },
    ]
    return {
        'schema': INSTALL_VERIFICATION_SCHEMA,
        'status': 'passed' if all(item['passed'] for item in checks) else 'failed',
        'created_at': _timestamp(),
        'dependency_mode': 'installed-distribution',
        'checks': checks,
        'wheel': None,
        'environment': {
            'python': platform.python_version(),
            'platform': platform.platform(),
        },
        'duration_seconds': round(time.monotonic() - started, 3),
    }


def _remote_smoke_request(bond_length: float = 0.74) -> str:
    return json.dumps({
        'atom': 'H 0 0 0; H 0 0 {0:.12g}'.format(float(bond_length)),
        'basis': 'sto-3g',
        'method': 'hf',
        'restricted': True,
        'job': 'single_point',
        'outputs': ['energy', 'homo_lumo', 'dipole'],
    })


def _redact_remote_error(executor: Any, exc: Exception) -> str:
    error = '{0}: {1}'.format(type(exc).__name__, exc)
    profile = getattr(executor, '_profile', None)
    for value, replacement in (
        (getattr(profile, 'host', None), '<remote-host>'),
        (getattr(profile, 'username', None), '<remote-user>'),
        (getattr(profile, 'private_key', None), '<private-key>'),
    ):
        if value:
            error = error.replace(str(value), replacement)
    return error


def verify_remote_cluster(
    profile_id: str,
    *,
    config_path: Optional[str] = None,
    submit_smoke: bool = False,
    submit_batch_smoke: bool = False,
    executor: Any = None,
) -> Dict[str, Any]:
    """Validate SSH RPC and optional single-task and job-array Slurm cycles."""

    if executor is None:
        from .executors import SshSlurmExecutor

        executor = SshSlurmExecutor.from_config(profile_id, config_path)
    started = time.monotonic()
    try:
        capabilities = executor.remote_capabilities()
    except Exception as exc:
        error = _redact_remote_error(executor, exc)
        return {
            'schema': REMOTE_VERIFICATION_SCHEMA,
            'status': 'failed',
            'created_at': _timestamp(),
            'profile_id': str(profile_id),
            'client_contract_version': PUBLIC_CONTRACT_VERSION,
            'checks': [{
                'name': 'ssh_rpc_capabilities',
                'passed': False,
                'value': None,
                'error': error,
            }],
            'smoke_submitted': False,
            'smoke': None,
            'batch_smoke_submitted': False,
            'batch_smoke': None,
            'duration_seconds': round(time.monotonic() - started, 3),
        }
    if not isinstance(capabilities, dict):
        raise TypeError('Remote capabilities must be a dictionary')
    remote_contract = capabilities.get('public_contract')
    contract_current = (
        isinstance(remote_contract, dict)
        and remote_contract.get('contract_version') == PUBLIC_CONTRACT_VERSION
    )
    checks = [
        {
            'name': 'ssh_rpc_capabilities',
            'passed': bool(capabilities.get('executor_id')),
            'value': capabilities.get('executor_id'),
        },
        {
            'name': 'remote_public_contract',
            'passed': contract_current,
            'value': (
                remote_contract.get('contract_version')
                if isinstance(remote_contract, dict)
                else None
            ),
            'expected': PUBLIC_CONTRACT_VERSION,
        },
    ]
    smoke_summary = None
    if submit_smoke and not contract_current:
        checks.append({
            'name': 'remote_slurm_task_report',
            'passed': False,
            'value': None,
            'error': 'Smoke submission was blocked because the remote public contract is incompatible.',
        })
        smoke_summary = {'submission_blocked': 'remote_public_contract'}
    elif submit_smoke:
        try:
            report = executor.execute_task(
                _remote_smoke_request(),
                channel='verification',
                locale='en',
                run_id='remote-smoke-h2-hf',
            )
            validate_public_payload(report, expected_schema=TASK_REPORT_SCHEMA)
            execution_status = str(report.get('execution_status') or '')
            checks.append({
                'name': 'remote_slurm_task_report',
                'passed': execution_status == 'succeeded',
                'value': execution_status,
                'expected': 'succeeded',
            })
            smoke_summary = {
                'execution_status': execution_status,
                'artifact_count': len(report.get('artifacts') or []),
                'error_count': len(report.get('errors') or []),
            }
        except Exception as exc:
            checks.append({
                'name': 'remote_slurm_task_report',
                'passed': False,
                'value': None,
                'error': _redact_remote_error(executor, exc),
            })
            smoke_summary = {'error': _redact_remote_error(executor, exc)}

    batch_smoke_summary = None
    if submit_batch_smoke and not contract_current:
        checks.append({
            'name': 'remote_slurm_batch_execution',
            'passed': False,
            'value': None,
            'error': 'Batch smoke submission was blocked because the remote public contract is incompatible.',
        })
        batch_smoke_summary = {'submission_blocked': 'remote_public_contract'}
    elif submit_batch_smoke:
        from .executors import BatchTask

        task_ids = ('remote-batch-h2-070', 'remote-batch-h2-090')
        tasks = [
            BatchTask(
                task_id=task_id,
                request=_remote_smoke_request(bond_length),
                channel='verification',
                locale='en',
                run_id=task_id,
            )
            for task_id, bond_length in zip(task_ids, (0.70, 0.90))
        ]
        try:
            batch_result = executor.execute_independent_tasks(tasks)
            returned_ids = tuple(sorted(batch_result.reports))
            expected_ids = tuple(sorted(task_ids))
            report_statuses = {}
            for task_id, report in batch_result.reports.items():
                validate_public_payload(report, expected_schema=TASK_REPORT_SCHEMA)
                report_statuses[task_id] = str(report.get('execution_status') or '')
            checks.extend([
                {
                    'name': 'remote_slurm_batch_task_mapping',
                    'passed': returned_ids == expected_ids,
                    'value': list(returned_ids),
                    'expected': list(expected_ids),
                },
                {
                    'name': 'remote_slurm_batch_task_reports',
                    'passed': (
                        returned_ids == expected_ids
                        and all(status == 'succeeded' for status in report_statuses.values())
                    ),
                    'value': report_statuses,
                    'expected': 'all succeeded',
                },
                {
                    'name': 'remote_slurm_batch_scheduler_evidence',
                    'passed': bool(batch_result.batches),
                    'value': len(batch_result.batches),
                    'expected': 'at least one submitted batch',
                },
                {
                    'name': 'remote_slurm_batch_artifacts',
                    'passed': bool(batch_result.artifacts),
                    'value': len(batch_result.artifacts),
                    'expected': 'scheduler manifest and handle artifacts',
                },
            ])
            batch_smoke_summary = {
                'task_ids': list(returned_ids),
                'execution_statuses': report_statuses,
                'batch_count': len(batch_result.batches),
                'artifact_count': len(batch_result.artifacts),
            }
        except Exception as exc:
            checks.append({
                'name': 'remote_slurm_batch_execution',
                'passed': False,
                'value': None,
                'error': _redact_remote_error(executor, exc),
            })
            batch_smoke_summary = {'error': _redact_remote_error(executor, exc)}
    return {
        'schema': REMOTE_VERIFICATION_SCHEMA,
        'status': 'passed' if all(item['passed'] for item in checks) else 'failed',
        'created_at': _timestamp(),
        'profile_id': str(profile_id),
        'client_contract_version': PUBLIC_CONTRACT_VERSION,
        'checks': checks,
        'smoke_submitted': bool(submit_smoke),
        'smoke': smoke_summary,
        'batch_smoke_submitted': bool(submit_batch_smoke),
        'batch_smoke': batch_smoke_summary,
        'duration_seconds': round(time.monotonic() - started, 3),
    }


def write_verification_report(report: Dict[str, Any], path: str) -> str:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=json_default, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    return str(target)


__all__ = [
    'verify_clean_install',
    'verify_installed_package',
    'verify_remote_cluster',
    'write_verification_report',
]
