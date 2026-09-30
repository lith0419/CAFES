from __future__ import annotations

from pyscf_agent.serialization import json_default

import io
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

from ..backend.artifacts import write_binary_artifact
from .contracts import (
    EMBEDDING_REFERENCE_SCHEMA,
    LOCALIZED_HAMILTONIAN_SCHEMA,
    LOCALIZED_SUBSPACE_SCHEMA,
)


EMBEDDING_HDF5_SCHEMAS = {
    'embedding_reference': EMBEDDING_REFERENCE_SCHEMA,
    'localized_subspace': LOCALIZED_SUBSPACE_SCHEMA,
    'localized_hamiltonian': LOCALIZED_HAMILTONIAN_SCHEMA,
}


def _dependencies() -> Tuple[Any, Any]:
    try:
        import h5py
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError('Embedding HDF5 artifacts require h5py and NumPy') from exc
    return h5py, np


def _dataset_path(value: Any) -> str:
    path = str(value or '').strip().strip('/')
    if not path or any(part in ('', '.', '..') for part in path.split('/')):
        raise ValueError('HDF5 dataset names must be non-empty relative paths')
    return path


def serialize_embedding_hdf5(
    *,
    schema: str,
    datasets: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> bytes:
    """Serialize numerical embedding data without placing arrays in TaskSpec JSON."""

    h5py, np = _dependencies()
    if schema not in set(EMBEDDING_HDF5_SCHEMAS.values()):
        raise ValueError('Unsupported embedding HDF5 schema: {0}'.format(schema))
    if not datasets:
        raise ValueError('At least one embedding dataset is required')
    manifest = []
    buffer = io.BytesIO()
    with h5py.File(buffer, 'w') as handle:
        handle.attrs['schema'] = schema
        handle.attrs['metadata_json'] = json.dumps(dict(metadata), sort_keys=True, default=json_default, allow_nan=False)
        for raw_name in sorted(datasets):
            name = _dataset_path(raw_name)
            array = np.asarray(datasets[raw_name])
            if array.dtype.kind == 'O':
                raise TypeError("Embedding dataset '{0}' has unsupported object dtype".format(name))
            if array.dtype.kind == 'U':
                array = array.astype(h5py.string_dtype(encoding='utf-8'))
            handle.create_dataset(name, data=array)
            manifest.append({
                'name': name,
                'shape': list(array.shape),
                'dtype': str(array.dtype),
            })
        handle.attrs['dataset_manifest_json'] = json.dumps(manifest, sort_keys=True)
    return buffer.getvalue()


def read_embedding_hdf5(source: Union[bytes, bytearray, str, Path]) -> Dict[str, Any]:
    h5py, _np = _dependencies()
    stream: Any = io.BytesIO(bytes(source)) if isinstance(source, (bytes, bytearray)) else str(source)
    datasets: Dict[str, Any] = {}
    with h5py.File(stream, 'r') as handle:
        schema = str(handle.attrs.get('schema') or '')
        if schema not in set(EMBEDDING_HDF5_SCHEMAS.values()):
            raise ValueError('Unsupported embedding HDF5 schema: {0}'.format(schema))

        def collect(name: str, node: Any) -> None:
            if isinstance(node, h5py.Dataset):
                datasets[name] = node[()]

        handle.visititems(collect)
        metadata = json.loads(str(handle.attrs.get('metadata_json') or '{}'))
        manifest = json.loads(str(handle.attrs.get('dataset_manifest_json') or '[]'))
    return {
        'schema': schema,
        'metadata': metadata,
        'dataset_manifest': manifest,
        'datasets': datasets,
    }


def write_embedding_hdf5_artifact(
    state: Dict[str, Any],
    kind: str,
    filename: str,
    *,
    datasets: Mapping[str, Any],
    metadata: Mapping[str, Any],
    description: str = '',
) -> Dict[str, Any]:
    schema = EMBEDDING_HDF5_SCHEMAS.get(str(kind or '').strip())
    if schema is None:
        raise ValueError('Unsupported embedding artifact kind: {0}'.format(kind))
    content = serialize_embedding_hdf5(schema=schema, datasets=datasets, metadata=metadata)
    ref = write_binary_artifact(
        state,
        kind,
        filename,
        content,
        mime_type='application/x-hdf5',
        description=description,
    )
    assert ref is not None
    restored = read_embedding_hdf5(content)
    ref['schema'] = schema
    ref['datasets'] = restored['dataset_manifest']
    return ref
