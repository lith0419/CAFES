from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from ..executors import TaskExecutor


_TARGET_ID_PATTERN = re.compile(r'^[a-z0-9][a-z0-9._-]*$')
_PUBLIC_EXECUTOR_FIELDS = (
    'execution_mode',
    'submission_mode',
    'location',
    'cluster_id',
    'expected_cluster_id',
    'supports_queue',
    'supports_cancel',
    'supports_job_handle',
    'supports_independent_batch',
    'supports_resource_profiles',
    'runtime_match_required',
    'runtime_release_id',
)


def _profile_label(profile_id: str) -> str:
    return profile_id.replace('.', ' ').replace('_', ' ').replace('-', ' ').title()


def _normalize_target_id(value: Any) -> str:
    target_id = str(value or '').strip().lower()
    if not target_id or not _TARGET_ID_PATTERN.fullmatch(target_id):
        raise ValueError(
            'Execution target ids must start with a letter or number and contain only '
            'lowercase letters, numbers, dots, underscores, or hyphens'
        )
    return target_id


@dataclass(frozen=True)
class ExecutionTarget:
    target_id: str
    label: str
    executor: TaskExecutor

    def public_dict(self) -> Dict[str, Any]:
        metadata = self.executor.describe()
        result = {
            'id': self.target_id,
            'label': self.label,
        }
        for key in _PUBLIC_EXECUTOR_FIELDS:
            if key in metadata:
                result[key] = metadata[key]
        return result


class ExecutionTargetRegistry:
    """Resolve request-level execution targets without changing scientific TaskSpecs."""

    def __init__(
        self,
        executors: Mapping[str, TaskExecutor],
        *,
        default_target: str,
        labels: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not isinstance(executors, Mapping) or not executors:
            raise ValueError('At least one execution target is required')
        normalized_labels = labels or {}
        targets: Dict[str, ExecutionTarget] = {}
        for raw_target_id, executor in executors.items():
            target_id = _normalize_target_id(raw_target_id)
            if executor is None or not callable(getattr(executor, 'execute_task', None)):
                raise TypeError('Execution target {0} must provide a TaskExecutor'.format(target_id))
            raw_label = normalized_labels.get(raw_target_id, normalized_labels.get(target_id, target_id))
            label = str(raw_label or target_id).strip()
            targets[target_id] = ExecutionTarget(target_id, label, executor)

        normalized_default = _normalize_target_id(default_target)
        if normalized_default not in targets:
            raise ValueError('Default execution target {0} is not registered'.format(normalized_default))
        self._targets = targets
        self._default_target = normalized_default

    @property
    def default_target(self) -> str:
        return self._default_target

    def resolve(self, target_id: Optional[str] = None) -> TaskExecutor:
        normalized = self._default_target if not str(target_id or '').strip() else _normalize_target_id(target_id)
        target = self._targets.get(normalized)
        if target is None:
            raise ValueError(
                'Unknown execution target {0}. Available targets: {1}'.format(
                    normalized,
                    ', '.join(self._targets),
                )
            )
        return target.executor

    def services(self) -> Dict[str, TaskExecutor]:
        return {target_id: target.executor for target_id, target in self._targets.items()}

    def public_resource_profiles(self, target_id: Optional[str] = None) -> Dict[str, Any]:
        normalized = self._default_target if not str(target_id or '').strip() else _normalize_target_id(target_id)
        target = self._targets.get(normalized)
        if target is None:
            raise ValueError(
                'Unknown execution target {0}. Available targets: {1}'.format(
                    normalized,
                    ', '.join(self._targets),
                )
            )
        executor = target.executor
        remote_capabilities = getattr(executor, 'remote_capabilities', None)
        metadata = remote_capabilities() if callable(remote_capabilities) else executor.describe()
        raw_options = metadata.get('resource_profile_options') if isinstance(metadata, dict) else None
        if not isinstance(raw_options, list):
            raw_ids = metadata.get('resource_profiles') if isinstance(metadata, dict) else None
            raw_options = [
                {'id': item, 'label': _profile_label(str(item))}
                for item in (raw_ids if isinstance(raw_ids, list) else [])
            ]
        default_limits = metadata.get('default_resource_limits') if isinstance(metadata, dict) else None
        auto_profile = {'id': 'auto', 'label': 'Auto'}
        if isinstance(default_limits, dict):
            try:
                memory_mb = int(default_limits.get('memory_mb'))
            except (TypeError, ValueError):
                memory_mb = 0
            if memory_mb > 0:
                auto_profile['memory_mb'] = memory_mb
        profiles = [auto_profile]
        seen = {'auto'}
        for item in raw_options:
            if isinstance(item, str):
                profile_id = item.strip()
                label = _profile_label(profile_id)
            elif isinstance(item, dict):
                profile_id = str(item.get('id') or '').strip()
                label = str(item.get('label') or profile_id).strip()
            else:
                continue
            if not profile_id or profile_id.lower() == 'auto' or profile_id in seen:
                continue
            seen.add(profile_id)
            profile = {'id': profile_id, 'label': label or profile_id}
            if isinstance(item, dict):
                try:
                    memory_mb = int(item.get('memory_mb'))
                except (TypeError, ValueError):
                    memory_mb = 0
                if memory_mb > 0:
                    profile['memory_mb'] = memory_mb
            profiles.append(profile)
        return {
            'execution_target': normalized,
            'default_profile': 'auto',
            'profiles': profiles,
        }

    def public_dict(self) -> Dict[str, Any]:
        return {
            'default_target': self._default_target,
            'targets': [target.public_dict() for target in self._targets.values()],
        }


__all__ = ['ExecutionTarget', 'ExecutionTargetRegistry']
