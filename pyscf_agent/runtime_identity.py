from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from .schema_contracts import PUBLIC_CONTRACT_VERSION, RUNTIME_IDENTITY_SCHEMA


RUNTIME_IDENTITY_ENV = 'PYSCF_AGENT_RUNTIME_IDENTITY'
SOURCE_FINGERPRINT_ALGORITHM = 'blake2b-256'
_IGNORED_PARTS = frozenset({
    '.git',
    '.pyscf-agent',
    '.venv',
    '__pycache__',
    'build',
    'dist',
    'runs',
})


class RuntimeIdentityError(ValueError):
    """Raised when a local or remote runtime identity is missing or stale."""


def _required_text(value: Any, name: str) -> str:
    normalized = str(value or '').strip()
    if not normalized:
        raise RuntimeIdentityError('{0} must be a non-empty string'.format(name))
    return normalized


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_included(relative_path: Path) -> bool:
    if any(part in _IGNORED_PARTS for part in relative_path.parts):
        return False
    return not any(part.endswith('.egg-info') for part in relative_path.parts)


def _git_source_files(root: Path) -> Optional[Tuple[Path, ...]]:
    try:
        result = subprocess.run(
            ('git', '-C', str(root), 'ls-files', '--cached', '--others', '--exclude-standard', '-z'),
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        # Deployed source archives do not require Git on execution nodes.
        return None
    if result.returncode != 0:
        return None
    files = []
    for raw in result.stdout.split(b'\0'):
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        candidate = root / relative
        if _is_included(relative) and candidate.is_file():
            files.append(relative)
    return tuple(sorted(files, key=lambda item: item.as_posix()))


def source_files(source_root: str | Path) -> Tuple[Path, ...]:
    """Return the deployable source files used by the runtime fingerprint."""

    root = Path(source_root).expanduser().resolve()
    if not (root / 'pyproject.toml').is_file():
        raise RuntimeIdentityError(
            'Source root does not contain pyproject.toml: {0}'.format(root)
        )
    git_files = _git_source_files(root)
    if git_files is not None:
        return git_files
    return tuple(sorted(
        (
            path.relative_to(root)
            for path in root.rglob('*')
            if path.is_file() and _is_included(path.relative_to(root))
        ),
        key=lambda item: item.as_posix(),
    ))


def source_fingerprint(
    source_root: str | Path,
    *,
    files: Optional[Iterable[Path]] = None,
) -> str:
    """Fingerprint the exact source snapshot without involving dataset artifacts."""

    root = Path(source_root).expanduser().resolve()
    digest = hashlib.blake2b(digest_size=32)
    normalized_files = tuple(files) if files is not None else source_files(root)
    for relative in normalized_files:
        relative_path = Path(relative)
        path = root / relative_path
        encoded_name = relative_path.as_posix().encode('utf-8')
        content = path.read_bytes()
        digest.update(len(encoded_name).to_bytes(8, 'big'))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, 'big'))
        digest.update(content)
    return '{0}:{1}'.format(SOURCE_FINGERPRINT_ALGORITHM, digest.hexdigest())


def _git_text(root: Path, *arguments: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ('git', '-C', str(root), *arguments),
            capture_output=True,
            check=False,
            text=True,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


@dataclass(frozen=True)
class RuntimeIdentity:
    environment_id: str
    release_id: str
    source_revision: str
    source_state: str
    source_fingerprint: str
    created_at: str
    public_contract_version: str = PUBLIC_CONTRACT_VERSION

    def __post_init__(self) -> None:
        for name in (
            'environment_id',
            'release_id',
            'source_revision',
            'source_state',
            'source_fingerprint',
            'created_at',
            'public_contract_version',
        ):
            _required_text(getattr(self, name), name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'schema': RUNTIME_IDENTITY_SCHEMA,
            'environment_id': self.environment_id,
            'release_id': self.release_id,
            'source_revision': self.source_revision,
            'source_state': self.source_state,
            'source_fingerprint': self.source_fingerprint,
            'fingerprint_algorithm': SOURCE_FINGERPRINT_ALGORITHM,
            'public_contract_version': self.public_contract_version,
            'created_at': self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'RuntimeIdentity':
        if not isinstance(payload, dict):
            raise TypeError('Runtime identity payload must be a dictionary')
        schema = payload.get('schema')
        if schema not in (None, RUNTIME_IDENTITY_SCHEMA):
            raise RuntimeIdentityError(
                'Unsupported runtime identity schema: {0}'.format(schema)
            )
        return cls(
            environment_id=_required_text(payload.get('environment_id'), 'environment_id'),
            release_id=_required_text(payload.get('release_id'), 'release_id'),
            source_revision=_required_text(payload.get('source_revision'), 'source_revision'),
            source_state=_required_text(payload.get('source_state'), 'source_state'),
            source_fingerprint=_required_text(
                payload.get('source_fingerprint'),
                'source_fingerprint',
            ),
            public_contract_version=_required_text(
                payload.get('public_contract_version') or PUBLIC_CONTRACT_VERSION,
                'public_contract_version',
            ),
            created_at=_required_text(payload.get('created_at'), 'created_at'),
        )


def build_runtime_identity(
    source_root: str | Path,
    *,
    environment_id: str,
    release_id: str,
    files: Optional[Iterable[Path]] = None,
    created_at: Optional[str] = None,
) -> RuntimeIdentity:
    root = Path(source_root).expanduser().resolve()
    revision = _git_text(root, 'rev-parse', 'HEAD') or 'source-snapshot'
    status = _git_text(root, 'status', '--porcelain', '--untracked-files=all')
    source_state = 'dirty' if status else 'clean'
    return RuntimeIdentity(
        environment_id=_required_text(environment_id, 'environment_id'),
        release_id=_required_text(release_id, 'release_id'),
        source_revision=revision,
        source_state=source_state,
        source_fingerprint=source_fingerprint(root, files=files),
        created_at=created_at or _timestamp(),
    )


def load_runtime_identity(
    path: Optional[str | Path] = None,
    *,
    required: bool = False,
) -> Optional[RuntimeIdentity]:
    configured = str(path or os.environ.get(RUNTIME_IDENTITY_ENV) or '').strip()
    if not configured:
        if required:
            raise RuntimeIdentityError('A runtime identity file is required')
        return None
    candidate = Path(configured).expanduser().resolve()
    if not candidate.is_file():
        if required:
            raise RuntimeIdentityError(
                'Runtime identity file was not found: {0}'.format(candidate)
            )
        return None
    try:
        payload = json.loads(candidate.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeIdentityError(
            'Runtime identity file is unreadable: {0}'.format(candidate)
        ) from exc
    return RuntimeIdentity.from_dict(payload)


def assert_source_matches_identity(
    identity: RuntimeIdentity,
    source_root: str | Path,
) -> None:
    current = source_fingerprint(source_root)
    if current != identity.source_fingerprint:
        raise RuntimeIdentityError(
            'Local source no longer matches release {0}; redeploy this worktree '
            'before remote submission'.format(identity.release_id)
        )


__all__ = [
    'RUNTIME_IDENTITY_ENV',
    'RuntimeIdentity',
    'RuntimeIdentityError',
    'assert_source_matches_identity',
    'build_runtime_identity',
    'load_runtime_identity',
    'source_files',
    'source_fingerprint',
]
