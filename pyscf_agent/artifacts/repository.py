from __future__ import annotations

from pyscf_agent.serialization import json_default

import fnmatch
import json
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from ..registry import PlatformRegistry, default_registry
from ..registry.platform import artifact_kind_is_registered


def _logical_absolute_path(path: Any) -> Path:
    """Normalize a path without resolving node-local filesystem symlinks."""

    return Path(os.path.abspath(os.path.expanduser(str(path))))


class ArtifactRepository:
    """Persist registered artifacts and return uniform lightweight references."""

    def __init__(
        self,
        registry: Optional[PlatformRegistry] = None,
        *,
        reference_root: Optional[Path] = None,
    ):
        self._registry_override = registry
        self._reference_root = (
            _logical_absolute_path(reference_root)
            if reference_root
            else None
        )

    @property
    def _registry(self) -> PlatformRegistry:
        return self._registry_override if self._registry_override is not None else default_registry()

    def _contract_for_kind(self, kind: str) -> Any:
        for contract in self._registry.artifact_contracts(status='executable'):
            if kind in contract.artifact_kinds:
                return contract
            if any(fnmatch.fnmatchcase(kind, pattern) for pattern in contract.artifact_kind_patterns):
                return contract
        return None

    def _payload_schema(self, kind: str) -> Optional[str]:
        contract = self._contract_for_kind(kind)
        if contract is None:
            return None
        schema = contract.payload_schemas.get(kind)
        return str(schema) if schema else None

    def _reference_path(self, path: Path) -> str:
        resolved = _logical_absolute_path(path)
        if self._reference_root is None:
            return str(resolved)
        try:
            return str(resolved.relative_to(self._reference_root))
        except ValueError:
            return str(resolved)

    def _validate_kind(self, kind: str) -> None:
        if not artifact_kind_is_registered(kind, self._registry):
            raise ValueError("Artifact kind '{0}' is not registered.".format(kind))

    @staticmethod
    def _write_bytes(path: Path, content: bytes, *, atomic: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not atomic:
            path.write_bytes(content)
            return
        temporary = path.with_name('.{0}.{1}.tmp'.format(path.name, uuid.uuid4().hex))
        try:
            temporary.write_bytes(content)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _reference(
        self,
        path: Path,
        *,
        kind: str,
        mime_type: str,
        description: str,
        size_bytes: int,
    ) -> Dict[str, Any]:
        return {
            'kind': kind,
            'path': self._reference_path(path),
            'size_bytes': int(size_bytes),
            'mime_type': mime_type,
            'description': description,
        }

    def write_bytes(
        self,
        path: Any,
        content: bytes,
        *,
        kind: str,
        mime_type: str = 'application/octet-stream',
        description: str = '',
        atomic: bool = True,
    ) -> Dict[str, Any]:
        self._validate_kind(kind)
        artifact_path = _logical_absolute_path(path)
        encoded = bytes(content)
        self._write_bytes(artifact_path, encoded, atomic=atomic)
        return self._reference(
            artifact_path,
            kind=kind,
            mime_type=mime_type,
            description=description,
            size_bytes=len(encoded),
        )

    def write_text(
        self,
        path: Any,
        content: str,
        *,
        kind: str,
        mime_type: str = 'text/plain; charset=utf-8',
        description: str = '',
        atomic: bool = True,
    ) -> Dict[str, Any]:
        return self.write_bytes(
            path,
            str(content).encode('utf-8'),
            kind=kind,
            mime_type=mime_type,
            description=description,
            atomic=atomic,
        )

    def write_json(
        self,
        path: Any,
        payload: Any,
        *,
        kind: str,
        description: str = '',
        atomic: bool = True,
    ) -> Dict[str, Any]:
        schema = self._payload_schema(kind)
        if schema and isinstance(payload, dict):
            payload = dict(payload)
            payload.setdefault('schema', schema)
        self._validate_kind(kind)
        artifact_path = _logical_absolute_path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        target = (artifact_path.with_name('.{0}.{1}.tmp'.format(artifact_path.name, uuid.uuid4().hex))
                  if atomic else artifact_path)
        try:
            with target.open('w', encoding='utf-8') as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)
            if atomic:
                target.replace(artifact_path)
        finally:
            if atomic:
                target.unlink(missing_ok=True)
        return self._reference(
            artifact_path, kind=kind, mime_type='application/json; charset=utf-8',
            description=description, size_bytes=artifact_path.stat().st_size,
        )

    def register_existing(
        self,
        path: Any,
        *,
        kind: str,
        mime_type: str = 'application/octet-stream',
        description: str = '',
    ) -> Optional[Dict[str, Any]]:
        self._validate_kind(kind)
        artifact_path = _logical_absolute_path(path)
        if not artifact_path.is_file():
            return None
        size_bytes = artifact_path.stat().st_size
        if size_bytes <= 0:
            return None
        return self._reference(
            artifact_path,
            kind=kind,
            mime_type=mime_type,
            description=description,
            size_bytes=size_bytes,
        )

    def with_reference_root(self, root: Optional[Any]) -> 'ArtifactRepository':
        return ArtifactRepository(
            self._registry_override,
            reference_root=_logical_absolute_path(root) if root else None,
        )


@lru_cache(maxsize=1)
def default_artifact_repository() -> ArtifactRepository:
    return ArtifactRepository()
