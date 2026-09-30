"""File-backed numerical values; JSON carries metadata, readers opt in to data."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from .repository import default_artifact_repository
from ..serialization import json_default

ARRAY_SCHEMA = 'pyscf-agent.numeric-array.v1'
INLINE_ORBITAL_LIMIT = 4096


def is_array_reference(value: Any) -> bool:
    return isinstance(value, dict) and value.get('schema') == ARRAY_SCHEMA


def write_array(value: Any, directory: Path) -> dict:
    import numpy as np

    array = np.ascontiguousarray(value)
    if array.dtype.kind not in 'biufc':
        raise ValueError('Numerical artifacts require a numeric array, without Python objects.')
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(memoryview(array).cast('B'))
    path = Path(os.path.abspath(directory)) / (digest.hexdigest() + '.npy')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with temporary.open('wb') as stream:
                np.save(stream, array, allow_pickle=False)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    reference = default_artifact_repository().register_existing(
        path, kind='numerical_array', description='File-backed numerical array',
        mime_type='application/x-npy',
    )
    return {**reference, 'schema': ARRAY_SCHEMA, 'shape': list(array.shape),
            'dtype': str(array.dtype), 'array_sha256': digest.hexdigest()}


def read_array(value: Any, *, mmap_mode: str | None = 'r') -> Any:
    """Read just this array. Legacy inline matrices remain valid inputs."""
    import numpy as np

    if not is_array_reference(value):
        return np.asarray(value)
    array = np.load(value['path'], allow_pickle=False, mmap_mode=mmap_mode)
    if list(array.shape) != value.get('shape') or str(array.dtype) != value.get('dtype'):
        raise ValueError('Numerical artifact shape or dtype differs from its reference.')
    if array.dtype.kind not in 'biufc':
        raise ValueError('Numerical artifacts must not contain Python objects.')
    return array


def externalize_orbital_guesses(value: Any, directory: Path) -> Any:
    """Preserve small inline inputs, move full MO matrices before copying JSON.

    Other numerical results (DMRG RDMs, DMET arrays, AO 1RDM) already have
    provider-owned NPZ artifacts. Do not reinterpret arbitrary model matrices.
    """
    if is_array_reference(value):
        return dict(value)
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == 'initial_mo_coeff' and item is not None and not is_array_reference(item):
                import numpy as np

                try:
                    array = np.asarray(item)
                except (TypeError, ValueError):
                    # Storage does not validate or repair a malformed guess;
                    # the numerical input boundary reports it as before.
                    array = None
                if array is not None and array.ndim == 2 and array.dtype.kind in 'biufc' and array.size > INLINE_ORBITAL_LIMIT:
                    result[key] = write_array(array, directory)
                    continue
            result[key] = externalize_orbital_guesses(item, directory)
        return result
    if isinstance(value, (list, tuple)):
        return [externalize_orbital_guesses(item, directory) for item in value]
    return value


def compact_request(value: Any, directory: Path) -> Any:
    """Accept structured requests and message envelopes without changing form."""
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except (ValueError, TypeError):
            return value
        if isinstance(payload, dict):
            return json.dumps(externalize_orbital_guesses(payload, directory), ensure_ascii=False, default=json_default, allow_nan=False)
        return value
    if isinstance(value, dict):
        result = externalize_orbital_guesses(value, directory)
        if isinstance(result.get('content'), str):
            result['content'] = compact_request(result['content'], directory)
        return result
    return value


def array_references(value: Any):
    if is_array_reference(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from array_references(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from array_references(item)
