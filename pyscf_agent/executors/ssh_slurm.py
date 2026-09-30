from __future__ import annotations

from .job_paths import collectable_status as _collectable_status

from pyscf_agent.executors.commands import (
    BULK_TIMEOUT_SECONDS, CONTROL_TIMEOUT_SECONDS, run_bounded, command_timeout,
)

from pyscf_agent.serialization import json_default

import json
import base64
import math
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .contracts import (
    BatchExecutionResult,
    BatchHandle,
    BatchTask,
    JobHandle,
    JobState,
    JobStatus,
)
from .remote_config import SshRemoteProfile, load_remote_profile
from ..remote.protocol import RemoteProtocolError, response_data
from ..runtime_identity import (
    RuntimeIdentity,
    RuntimeIdentityError,
    assert_source_matches_identity,
    load_runtime_identity,
)


MAX_REMOTE_INPUT_BYTES = 5 * 1024 * 1024
MAX_REMOTE_ARRAY_BYTES = 64 * 1024 * 1024


def _validated_remote_artifact_path(value: Any) -> str:
    normalized = str(value or '').strip()
    path = Path(normalized)
    if (
        not normalized
        or not path.is_absolute()
        or '..' in path.parts
        or any(character in normalized for character in ('\x00', '\n', '\r', '"', '\\'))
    ):
        raise ValueError('Remote artifact paths must be safe absolute paths')
    return normalized





def _structured_request(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    return decoded


def _request_input_paths(value: Any) -> List[Tuple[str, Path]]:
    payload = _structured_request(value)
    if not isinstance(payload, dict):
        return []
    candidates = []
    direct = payload.get('model_hamiltonian_input_file')
    if isinstance(direct, str) and direct.strip():
        candidates.append(direct.strip())
    model = payload.get('model_hamiltonian')
    if isinstance(model, dict):
        nested = model.get('input_file')
        if isinstance(nested, str) and nested.strip():
            candidates.append(nested.strip())
    solver = payload.get('solver') if isinstance(payload.get('solver'), dict) else {}
    options = solver.get('options') or {}
    density_source = options.get('reference_density_source') or {}
    if isinstance(density_source, dict) and density_source.get('path'):
        candidates.append(density_source['path'])
    from ..artifacts.arrays import array_references
    candidates.extend(ref['path'] for ref in array_references(payload))
    paths = []
    seen = set()
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if not path.is_file():
            continue
        resolved = path.resolve()
        if str(resolved) not in seen:
            paths.append((candidate, resolved))
            seen.add(str(resolved))
    return paths


def _request_attachments(value: Any) -> List[Dict[str, Any]]:
    attachments = []
    from ..artifacts.arrays import array_references
    array_paths = {ref['path'] for ref in array_references(_structured_request(value))}
    for source_path, path in _request_input_paths(value):
        limit = MAX_REMOTE_ARRAY_BYTES if source_path in array_paths else MAX_REMOTE_INPUT_BYTES
        if path.stat().st_size > limit:
            raise ValueError(
                'Remote input file exceeds {0} bytes: {1}'.format(
                    limit,
                    path,
                )
            )
        content = path.read_bytes()
        attachments.append({
            'source_path': source_path,
            'filename': path.name,
            'size_bytes': len(content),
            'content_base64': base64.b64encode(content).decode('ascii'),
        })
    return attachments


def _validate_remote_restart_manifest(value: Any) -> None:
    payload = _structured_request(value)
    if not isinstance(payload, dict):
        return
    solver = payload.get('solver') if isinstance(payload.get('solver'), dict) else {}
    options = solver.get('options') if isinstance(solver.get('options'), dict) else {}
    restart_manifest = options.get('restart_manifest')
    if isinstance(restart_manifest, str) and restart_manifest.strip():
        raise ValueError(
            'Remote block2 MPS restart requires an inline checkpoint manifest object '
            'from a prior remote TaskReport. Local or server path strings are not '
            'portable across the SSH boundary.'
        )


def _default_ssh_runner(
    args: Sequence[str],
    input_text: str,
    *, timeout: float = CONTROL_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess:
    return run_bounded(
        list(args),
        timeout=timeout,
        input=input_text,
        capture_output=True,
        check=False,
        text=True,
    )


class SshSlurmExecutor:
    """Submit and monitor server-side Slurm jobs through stateless SSH RPC."""

    executor_id = 'slurm'
    execution_mode = 'remote_scheduler'
    submission_mode = 'ssh_rpc'
    supports_independent_batch = True
    supports_recoverable_batch = True

    def __init__(
        self,
        profile: SshRemoteProfile,
        *,
        command_runner: Callable[[Sequence[str], str], Any] = _default_ssh_runner,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        if not isinstance(profile, SshRemoteProfile):
            raise TypeError('profile must be an SshRemoteProfile')
        if not callable(command_runner):
            raise TypeError('command_runner must be callable')
        if not callable(sleeper):
            raise TypeError('sleeper must be callable')
        self._profile = profile
        self._command_runner = command_runner
        self._sleeper = sleeper

    @classmethod
    def from_config(
        cls,
        profile_id: str,
        path: Optional[str] = None,
        *,
        command_runner: Callable[[Sequence[str], str], Any] = _default_ssh_runner,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> 'SshSlurmExecutor':
        return cls(
            load_remote_profile(profile_id, path),
            command_runner=command_runner,
            sleeper=sleeper,
        )

    def _refresh_profile(self) -> None:
        """Reload an atomically replaced per-worktree binding before SSH use."""

        source = Path(self._profile.source_path).expanduser()
        if not source.is_file():
            return
        self._profile = load_remote_profile(
            self._profile.profile_id,
            str(source),
        )

    def _ssh_command(self, operation: str) -> List[str]:
        self._refresh_profile()
        profile = self._profile
        command = [
            'ssh',
            '-T',
            '-o',
            'BatchMode=yes',
            '-o', 'ServerAliveInterval=15',
            '-o', 'ServerAliveCountMax=3',
            '-o',
            'ConnectTimeout={0}'.format(
                max(1, int(math.ceil(profile.connect_timeout_seconds)))
            ),
            '-o',
            'StrictHostKeyChecking={0}'.format(
                'yes' if profile.strict_host_key_checking else 'no'
            ),
            '-p',
            str(profile.port),
        ]
        if profile.private_key:
            command.extend(('-i', profile.private_key))
        if profile.known_hosts:
            command.extend(('-o', 'UserKnownHostsFile={0}'.format(profile.known_hosts)))
        if profile.control_persist_seconds > 0:
            command.extend((
                '-o',
                'ControlMaster=auto',
                '-o',
                'ControlPersist={0}'.format(profile.control_persist_seconds),
                '-o',
                'ControlPath=~/.ssh/pyscf-agent-%C',
            ))
        command.append('{0}@{1}'.format(profile.username, profile.host))
        remote_command = [
            profile.remote_python,
            '-m',
            'pyscf_agent.remote.rpc_cli',
            '--slurm-config',
            profile.remote_slurm_config,
            '--slurm-profile',
            profile.remote_slurm_profile or profile.profile_id,
            operation,
        ]
        command.append(shlex.join(remote_command))
        return command

    def _sftp_command(self) -> List[str]:
        self._refresh_profile()
        profile = self._profile
        command = [
            'sftp',
            '-b',
            '-',
            '-P',
            str(profile.port),
            '-o',
            'BatchMode=yes',
            '-o', 'ServerAliveInterval=15',
            '-o', 'ServerAliveCountMax=3',
            '-o',
            'ConnectTimeout={0}'.format(
                max(1, int(math.ceil(profile.connect_timeout_seconds)))
            ),
            '-o',
            'StrictHostKeyChecking={0}'.format(
                'yes' if profile.strict_host_key_checking else 'no'
            ),
        ]
        if profile.private_key:
            command.extend(('-i', profile.private_key))
        if profile.known_hosts:
            command.extend(('-o', 'UserKnownHostsFile={0}'.format(profile.known_hosts)))
        if profile.control_persist_seconds > 0:
            command.extend((
                '-o',
                'ControlMaster=auto',
                '-o',
                'ControlPersist={0}'.format(profile.control_persist_seconds),
                '-o',
                'ControlPath=~/.ssh/pyscf-agent-%C',
            ))
        command.append('{0}@{1}'.format(profile.username, profile.host))
        return command

    def _run_transport(self, command, input_text, operation, *, timeout=CONTROL_TIMEOUT_SECONDS):
        try:
            if self._command_runner is _default_ssh_runner:
                return _default_ssh_runner(command, input_text, timeout=timeout)
            return self._command_runner(command, input_text)
        except subprocess.TimeoutExpired as exc:
            raise command_timeout('Remote ' + operation, exc) from exc

    def _rpc(self, operation: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        request_text = json.dumps(payload or {}, ensure_ascii=False, default=json_default, allow_nan=False)
        result = self._run_transport(
            self._ssh_command(operation), request_text, operation,
            timeout=BULK_TIMEOUT_SECONDS if operation == 'generate-hamiltonian-dataset' else CONTROL_TIMEOUT_SECONDS,
        )
        stdout = str(getattr(result, 'stdout', '') or '').strip()
        stderr = str(getattr(result, 'stderr', '') or '').strip()
        return_code = int(getattr(result, 'returncode', 0))
        if not stdout:
            raise RemoteProtocolError(
                'SSH RPC returned no JSON response{0}'.format(
                    ': {0}'.format(stderr) if stderr else ''
                ),
                code='ssh_transport_error',
            )
        response = None
        for line in reversed(stdout.splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and candidate.get('schema'):
                response = candidate
                break
        if response is None:
            raise RemoteProtocolError(
                'SSH RPC returned invalid JSON: {0}'.format(stdout[:300]),
                code='invalid_json',
            )
        if return_code != 0:
            raise RemoteProtocolError(
                'SSH RPC exited with code {0}: {1}'.format(
                    return_code,
                    stderr or 'no error output',
                ),
                code='ssh_transport_error',
            )
        return response_data(response, operation=operation)

    def _local_runtime_identity(self) -> RuntimeIdentity:
        profile = self._profile
        try:
            identity = load_runtime_identity(
                profile.runtime_identity_file,
                required=True,
            )
            assert identity is not None
            if not profile.local_source_root:
                raise RuntimeIdentityError(
                    'local_source_root is required when runtime matching is enabled'
                )
            assert_source_matches_identity(identity, profile.local_source_root)
            return identity
        except RuntimeIdentityError as exc:
            raise RemoteProtocolError(
                str(exc),
                code='local_runtime_identity_invalid',
            ) from exc

    def _verify_runtime_identity(self, capabilities: Dict[str, Any]) -> None:
        if not self._profile.require_runtime_match:
            return
        local = self._local_runtime_identity()
        raw_remote = capabilities.get('runtime_identity')
        try:
            remote = RuntimeIdentity.from_dict(raw_remote)
        except (TypeError, RuntimeIdentityError) as exc:
            raise RemoteProtocolError(
                'Remote runtime identity is missing or invalid for release {0}'.format(
                    local.release_id
                ),
                code='remote_runtime_identity_missing',
            ) from exc
        mismatches = [
            name
            for name in (
                'environment_id',
                'release_id',
                'source_revision',
                'source_fingerprint',
                'public_contract_version',
            )
            if getattr(local, name) != getattr(remote, name)
        ]
        if mismatches:
            raise RemoteProtocolError(
                'Local worktree release {0} does not match remote release {1}; '
                'mismatched field(s): {2}. Redeploy this worktree before submitting.'.format(
                    local.release_id,
                    remote.release_id,
                    ', '.join(mismatches),
                ),
                code='runtime_identity_mismatch',
            )

    @staticmethod
    def _handle_payload(handle: Any) -> Dict[str, Any]:
        if isinstance(handle, JobHandle):
            normalized = handle
        elif isinstance(handle, dict):
            normalized = JobHandle.from_dict(handle)
        else:
            raise TypeError('handle must be a JobHandle or dictionary')
        return {'handle': normalized.to_dict()}

    def submit_task(
        self,
        request: Any,
        *,
        channel: str = 'agent',
        locale: str = 'en',
        work_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        resource_profile: Optional[str] = None,
    ) -> JobHandle:
        self._refresh_profile()
        if self._profile.require_runtime_match:
            self.remote_capabilities()
        _validate_remote_restart_manifest(request)
        payload = {
            'request': request,
            'channel': channel,
            'locale': locale,
            'run_id': run_id,
            'attachments': _request_attachments(request),
        }
        if resource_profile:
            payload['resource_profile'] = resource_profile
        data = self._rpc('submit-task', payload)
        return JobHandle.from_dict(data)

    def status(self, handle: Any) -> JobStatus:
        return JobStatus.from_dict(
            self._rpc('status', self._handle_payload(handle))
        )

    def status_many(self, handles: Sequence[Any]) -> List[JobStatus]:
        jobs = []
        for handle in handles:
            if isinstance(handle, JobHandle):
                jobs.append(handle)
            elif isinstance(handle, dict):
                jobs.append(JobHandle.from_dict(handle))
            else:
                raise TypeError('handles must contain JobHandle objects or dictionaries')
        if not jobs:
            return []
        data = self._rpc('batch-status', {
            'jobs': [job.to_dict() for job in jobs],
        })
        raw_statuses = data.get('statuses') if isinstance(data, dict) else None
        if not isinstance(raw_statuses, list) or len(raw_statuses) != len(jobs):
            raise RemoteProtocolError('Remote batch status response is invalid')
        return [JobStatus.from_dict(item) for item in raw_statuses]

    def cancel(self, handle: Any) -> JobStatus:
        return JobStatus.from_dict(
            self._rpc('cancel', self._handle_payload(handle))
        )

    def fetch(self, handle: Any) -> Dict[str, Any]:
        data = self._rpc('fetch', self._handle_payload(handle))
        if not isinstance(data, dict):
            raise RemoteProtocolError('Remote TaskReport must be a JSON object')
        return data

    def inspect_task(self, handle: Any) -> Dict[str, Any]:
        data = self._rpc('inspect-task', self._handle_payload(handle))
        if not isinstance(data, dict) or data.get('handle') != self._handle_payload(handle)['handle']:
            raise RemoteProtocolError('Remote task view does not match the requested handle')
        return data

    def logs(self, handle: Any) -> List[Dict[str, Any]]:
        data = self._rpc('logs', self._handle_payload(handle))
        return list(data) if isinstance(data, list) else []

    def artifacts(self, handle: Any) -> List[Dict[str, Any]]:
        data = self._rpc('artifacts', self._handle_payload(handle))
        return list(data) if isinstance(data, list) else []

    def collect_files(self, paths: Dict[str, str]) -> List[Dict[str, Any]]:
        """Download executor-owned artifacts in one SFTP batch."""

        if not isinstance(paths, dict) or not paths:
            return []
        transfers = []
        commands = []
        for source, destination in paths.items():
            remote_path = _validated_remote_artifact_path(source)
            local_path = Path(destination).expanduser().resolve()
            if any(
                character in str(local_path)
                for character in ('\x00', '\n', '\r', '"', '\\')
            ):
                raise ValueError('Local artifact destinations must be safe paths')
            local_path.parent.mkdir(parents=True, exist_ok=True)
            commands.append('get "{0}" "{1}"'.format(remote_path, local_path))
            transfers.append({
                'source_path': remote_path,
                'path': str(local_path),
            })
        result = self._run_transport(
            self._sftp_command(), '\n'.join(commands) + '\n', 'SFTP artifact collection',
            timeout=BULK_TIMEOUT_SECONDS,
        )
        return_code = int(getattr(result, 'returncode', 0))
        stderr = str(getattr(result, 'stderr', '') or '').strip()
        if return_code != 0:
            raise RemoteProtocolError(
                'SFTP dataset collection exited with code {0}: {1}'.format(
                    return_code,
                    stderr or 'no error output',
                ),
                code='ssh_transport_error',
            )
        missing = [item['path'] for item in transfers if not Path(item['path']).is_file()]
        if missing:
            raise RemoteProtocolError(
                'SFTP dataset collection did not create: {0}'.format(', '.join(missing)),
                code='remote_artifact_missing',
            )
        return transfers

    def generate_hamiltonian_dataset(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Materialize a complete dataset inside this remote environment."""

        if not isinstance(request, dict):
            raise TypeError('Dataset generation request must be a dictionary')
        self._refresh_profile()
        if self._profile.require_runtime_match:
            self.remote_capabilities()
        data = self._rpc('generate-hamiltonian-dataset', request)
        if not isinstance(data, dict):
            raise RemoteProtocolError('Remote dataset generation response is invalid')
        return data

    def _wait_for_jobs(self, jobs: Sequence[JobHandle]) -> None:
        started = time.monotonic()
        completed_without_report: Dict[str, float] = {}
        while True:
            statuses = self.status_many(jobs)
            if statuses and all(_collectable_status(status) for status in statuses):
                return
            if not jobs:
                return
            now = time.monotonic()
            for status in statuses:
                if status.state == JobState.COMPLETED and not status.report_available:
                    first_seen = completed_without_report.setdefault(
                        status.handle.job_id,
                        now,
                    )
                    if (
                        now - first_seen
                        >= self._profile.report_propagation_timeout_seconds
                    ):
                        raise RemoteProtocolError(
                            'Remote Slurm job completed but its TaskReport did not become visible: {0}'.format(
                                status.handle.job_id
                            )
                        )
                else:
                    completed_without_report.pop(status.handle.job_id, None)
            timeout = self._profile.wait_timeout_seconds
            if timeout is not None and now - started >= timeout:
                raise TimeoutError('Timed out waiting for remote Slurm jobs')
            self._sleeper(self._profile.poll_interval_seconds)

    def execute_task(
        self,
        request: Any,
        *,
        channel: str = 'agent',
        locale: str = 'en',
        work_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        resource_profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        handle = self.submit_task(
            request,
            channel=channel,
            locale=locale,
            run_id=run_id,
            resource_profile=resource_profile,
        )
        self._wait_for_jobs([handle])
        return self.fetch(handle)

    def submit_independent_tasks(
        self,
        tasks: Sequence[BatchTask],
        *,
        resource_profile: Optional[str] = None,
    ) -> List[BatchHandle]:
        normalized = list(tasks)
        if not normalized:
            return []
        self._refresh_profile()
        if self._profile.require_runtime_match:
            self.remote_capabilities()
        if any(not isinstance(task, BatchTask) for task in normalized):
            raise TypeError('tasks must contain BatchTask objects')
        for task in normalized:
            _validate_remote_restart_manifest(task.request)
        payload = {
            'tasks': [task.to_dict() for task in normalized],
            'attachments': {
                task.task_id: _request_attachments(task.request)
                for task in normalized
            },
        }
        if resource_profile:
            payload['resource_profile'] = resource_profile
        data = self._rpc('submit-independent', payload)
        batches = data.get('batches') if isinstance(data, dict) else None
        if not isinstance(batches, list):
            raise RemoteProtocolError('Remote batch submission response is invalid')
        return [BatchHandle.from_dict(item) for item in batches]

    def recover_submission(self, submission_path: str) -> Dict[str, Any]:
        """Read persisted server evidence; this operation never submits work."""
        data = self._rpc('recover-submission', {'submission_path': submission_path})
        if not isinstance(data, dict):
            raise RemoteProtocolError('Remote submission evidence is invalid')
        return data

    def execute_independent_tasks(
        self,
        tasks: Sequence[BatchTask],
        *,
        resource_profile: Optional[str] = None,
    ) -> BatchExecutionResult:
        batches = self.submit_independent_tasks(
            tasks,
            resource_profile=resource_profile,
        )
        if not batches:
            return BatchExecutionResult(reports={}, batches=[])
        self.wait_for_independent_tasks(batches)
        return self.collect_independent_tasks(batches)

    def wait_for_independent_tasks(
        self,
        batches: Sequence[BatchHandle],
    ) -> None:
        normalized = list(batches)
        if any(not isinstance(batch, BatchHandle) for batch in normalized):
            raise TypeError('batches must contain BatchHandle objects')
        self._wait_for_jobs([
            job
            for batch in normalized
            for job in batch.jobs
        ])

    def collect_independent_tasks(
        self,
        batches: Sequence[BatchHandle],
    ) -> BatchExecutionResult:
        normalized = list(batches)
        if any(not isinstance(batch, BatchHandle) for batch in normalized):
            raise TypeError('batches must contain BatchHandle objects')
        data = self._rpc('batch-fetch', {
            'batches': [batch.to_dict() for batch in normalized],
        })
        return BatchExecutionResult.from_dict(data)

    def remote_capabilities(self) -> Dict[str, Any]:
        data = self._rpc('capabilities')
        if not isinstance(data, dict):
            raise RemoteProtocolError('Remote capabilities response is invalid')
        expected_cluster_id = self._profile.expected_cluster_id
        actual_cluster_id = str(data.get('cluster_id') or '').strip()
        if expected_cluster_id and actual_cluster_id != expected_cluster_id:
            raise RemoteProtocolError(
                'Remote profile {0} expected Slurm cluster {1}, but the selected '
                'server configuration reported {2}'.format(
                    self._profile.profile_id,
                    expected_cluster_id,
                    actual_cluster_id or 'no cluster_id',
                )
            )
        self._verify_runtime_identity(data)
        return data

    def describe(self) -> Dict[str, Any]:
        self._refresh_profile()
        profile = self._profile
        local_identity = load_runtime_identity(profile.runtime_identity_file)
        return {
            'executor_id': self.executor_id,
            'execution_mode': self.execution_mode,
            'submission_mode': self.submission_mode,
            'location': 'remote_slurm_cluster',
            'transport': 'ssh',
            'remote_profile': profile.profile_id,
            'remote_slurm_profile': profile.remote_slurm_profile or profile.profile_id,
            'expected_cluster_id': profile.expected_cluster_id,
            'remote_host': profile.host,
            'remote_port': profile.port,
            'supports_queue': True,
            'supports_cancel': True,
            'supports_job_handle': True,
            'supports_independent_batch': True,
            'supports_recoverable_batch': True,
            'supports_file_collection': True,
            'supports_dataset_generation': True,
            'supports_resource_profiles': True,
            'configuration_source': profile.source_path,
            'runtime_match_required': bool(profile.require_runtime_match),
            'runtime_release_id': (
                local_identity.release_id if local_identity is not None else None
            ),
        }


__all__ = ['SshSlurmExecutor']
