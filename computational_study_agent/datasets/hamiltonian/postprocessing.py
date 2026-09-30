"""Generate portable Hamiltonian datasets and optionally collect them locally."""

from __future__ import annotations

from pyscf_agent.serialization import json_default

import copy
import json
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from pyscf_agent.artifacts import default_artifact_repository

from computational_study_agent.datasets.hamiltonian.contracts import HAMILTONIAN_MANIFEST_SCHEMA


HAMILTONIAN_DATASET_GENERATE_ACTION = 'generate_hamiltonian_dataset'
HAMILTONIAN_DATASET_COLLECT_ACTION = 'collect_hamiltonian_dataset'
HAMILTONIAN_DATASET_GENERATION_SCHEMA = 'pyscf-agent.hamiltonian-dataset-generation.v1'
HAMILTONIAN_DATASET_GENERATION_REQUEST_SCHEMA = (
    'pyscf-agent.hamiltonian-dataset-generation-request.v1'
)
HAMILTONIAN_DATASET_GENERATION_FILENAME = 'hamiltonian-dataset-generation.json'


def _source_artifact_path(value: Any, root: Path) -> Path:
    candidate = Path(os.path.expanduser(str(value)))
    return candidate if candidate.is_absolute() else root / candidate


def _json_lines(rows: Iterable[Mapping[str, Any]]) -> str:
    return ''.join(
        json.dumps(row, ensure_ascii=False, separators=(',', ':'), default=json_default, allow_nan=False) + '\n'
        for row in rows
    )


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('Dataset JSON artifact must be an object.')
    return payload


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(
                'Dataset index row {0} must be a JSON object.'.format(line_number)
            )
        rows.append(payload)
    return rows


def _safe_slug(value: Any, fallback: str) -> str:
    normalized = re.sub(r'[^A-Za-z0-9._-]+', '-', str(value or '')).strip('-._')
    return (normalized or fallback)[:120]


def _safe_relative_path(value: Any) -> Path:
    path = Path(str(value or '').strip())
    if not str(path) or path.is_absolute() or '..' in path.parts:
        raise ValueError('Generated dataset file paths must be safe relative paths.')
    return path


def _manifest_path(report: Mapping[str, Any], work_root: Path) -> Path:
    manifest = report.get('dataset_manifest')
    artifacts = manifest.get('artifacts') if isinstance(manifest, dict) else None
    raw_path = artifacts.get('dataset_manifest') if isinstance(artifacts, dict) else None
    if raw_path:
        path = _source_artifact_path(raw_path, work_root)
        if path.is_file():
            return path
    fallback = work_root / 'dataset' / 'dataset-manifest.json'
    if fallback.is_file():
        return fallback
    raise ValueError('StudyReport does not reference an available Hamiltonian dataset manifest.')


def _index_path(manifest: Mapping[str, Any], key: str, source_root: Path) -> Path:
    artifacts = manifest.get('artifacts')
    raw_path = artifacts.get(key) if isinstance(artifacts, dict) else None
    if not raw_path:
        raise ValueError('Dataset manifest does not reference {0}.'.format(key))
    path = _source_artifact_path(raw_path, source_root)
    if not path.is_file():
        raise ValueError('Dataset index is unavailable: {0}'.format(path))
    return path


def build_hamiltonian_dataset_generation_request(
    report: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the control payload used by local or remote generation."""

    if not isinstance(report, Mapping):
        raise TypeError('StudyReport must be a mapping.')
    work_dir = str(report.get('work_dir') or '').strip()
    if not work_dir:
        raise ValueError('StudyReport work_dir is required for dataset generation.')
    work_root = Path(os.path.abspath(os.path.expanduser(work_dir)))
    source_manifest_path = _manifest_path(report, work_root)
    source_manifest = _load_json(source_manifest_path)
    if source_manifest.get('schema') != HAMILTONIAN_MANIFEST_SCHEMA:
        raise ValueError('Unsupported Hamiltonian dataset manifest schema.')
    return {
        'schema': HAMILTONIAN_DATASET_GENERATION_REQUEST_SCHEMA,
        'study_id': str(report.get('study_id') or 'study'),
        'source_root': str(work_root),
        'source_manifest': source_manifest,
        'source_manifest_path': str(source_manifest_path),
        'samples': _load_jsonl(_index_path(source_manifest, 'sample_index', work_root)),
        'rejections': _load_jsonl(
            _index_path(source_manifest, 'rejection_index', work_root)
        ),
    }


def _matrix_sources(samples: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, str]]:
    sources: List[str] = []
    labels: Dict[str, str] = {}
    for row in samples:
        molecule_id = str(row.get('molecule_id') or '').strip()
        for field_name in ('fock', 'overlap'):
            reference = row.get(field_name)
            source = str(reference.get('path') or '').strip() if isinstance(reference, dict) else ''
            if not source:
                raise ValueError(
                    'Hamiltonian sample {0} has no {1} artifact path.'.format(
                        row.get('sample_id') or '<unknown>',
                        field_name,
                    )
                )
            if source not in sources:
                sources.append(source)
                labels[source] = molecule_id
    return sources, labels


def _resolved_source(
    source: str,
    *,
    source_root: Path,
    allowed_source_root: Optional[Path],
) -> Path:
    path = _source_artifact_path(source, source_root).resolve()
    if allowed_source_root is not None:
        allowed = allowed_source_root.resolve()
        try:
            path.relative_to(allowed)
        except ValueError as exc:
            raise ValueError(
                'Dataset source is outside the configured server work root: {0}'.format(path)
            ) from exc
    if not path.is_file():
        raise FileNotFoundError('Dataset trajectory array is unavailable: {0}'.format(path))
    return path


def _materialize_sources(
    sources: List[str],
    labels: Mapping[str, str],
    *,
    source_root: Path,
    output_root: Path,
    allowed_source_root: Optional[Path],
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    arrays_root = output_root / 'arrays'
    arrays_root.mkdir(parents=True, exist_ok=True)
    path_map: Dict[str, str] = {}
    generated_files: List[Dict[str, Any]] = []
    for index, source in enumerate(sources, 1):
        label = _safe_slug(labels.get(source), 'trajectory')
        relative = 'arrays/trajectory-{0:04d}-{1}.npz'.format(index, label)
        target = output_root / relative
        source_path = _resolved_source(
            source,
            source_root=source_root,
            allowed_source_root=allowed_source_root,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if source_path != target.resolve():
            if target.exists():
                target.unlink()
            # A published dataset must survive an in-place source rerun.
            shutil.copy2(source_path, target)
            storage = 'copy'
        else:
            storage = 'existing'
        path_map[source] = relative
        generated_files.append({
            'source_path': source,
            'path': relative,
            'molecule_id': labels.get(source),
            'size_bytes': target.stat().st_size,
            'storage': storage,
        })
    return path_map, generated_files


def _validate_arrays(
    samples: List[Dict[str, Any]],
    path_map: Mapping[str, str],
    output_root: Path,
) -> None:
    import numpy as np  # pylint: disable=import-outside-toplevel

    archives: Dict[str, Any] = {}
    try:
        for row in samples:
            for field_name in ('fock', 'overlap'):
                reference = row.get(field_name)
                if not isinstance(reference, dict):
                    raise ValueError('Hamiltonian sample matrix references must be objects.')
                source = str(reference.get('path') or '')
                relative = path_map[source]
                archive = archives.get(relative)
                if archive is None:
                    archive = np.load(output_root / relative, allow_pickle=False)
                    archives[relative] = archive
                array_key = str(reference.get('array_key') or '')
                if array_key not in archive:
                    raise ValueError('{0} is missing array {1}.'.format(relative, array_key))
                values = archive[array_key]
                array_index = reference.get('array_index')
                if array_index is None or int(array_index) < 0 or int(array_index) >= values.shape[0]:
                    raise ValueError('{0}:{1} has an invalid array_index.'.format(relative, array_key))
                expected_shape = tuple(int(value) for value in reference.get('shape') or ())
                if tuple(values[int(array_index)].shape) != expected_shape:
                    raise ValueError('{0}:{1} does not match its indexed matrix shape.'.format(relative, array_key))
    finally:
        for archive in archives.values():
            archive.close()


def materialize_hamiltonian_dataset(
    request: Mapping[str, Any],
    *,
    output_dir: Any,
    allowed_source_root: Optional[Any] = None,
    location: str = 'local',
) -> Dict[str, Any]:
    """Publish a complete portable dataset into a new destination directory."""

    output_root = Path(os.path.abspath(os.path.expanduser(str(output_dir))))
    if output_root.exists():
        raise FileExistsError('Generated dataset destination already exists: {0}'.format(output_root))
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.dataset-staging-', dir=str(output_root.parent)) as staging:
        generation = _materialize_hamiltonian_dataset_files(
            request,
            output_dir=staging,
            allowed_source_root=allowed_source_root,
            location=location,
        )
        # Staging is on the same filesystem; no partial dataset is published.
        Path(staging).rename(output_root)
    generation['dataset_root'] = str(output_root)
    generation['manifest_path'] = str(output_root / 'dataset-manifest.json')
    return generation


def _materialize_hamiltonian_dataset_files(
    request: Mapping[str, Any],
    *,
    output_dir: Any,
    allowed_source_root: Optional[Any] = None,
    location: str = 'local',
) -> Dict[str, Any]:

    if not isinstance(request, Mapping):
        raise TypeError('Dataset generation request must be a mapping.')
    if request.get('schema') != HAMILTONIAN_DATASET_GENERATION_REQUEST_SCHEMA:
        raise ValueError('Unsupported Hamiltonian dataset generation request schema.')
    source_manifest = request.get('source_manifest')
    samples = request.get('samples')
    rejections = request.get('rejections')
    if not isinstance(source_manifest, dict) or source_manifest.get('schema') != HAMILTONIAN_MANIFEST_SCHEMA:
        raise ValueError('Dataset generation request has an invalid source manifest.')
    if not isinstance(samples, list) or any(not isinstance(row, dict) for row in samples):
        raise ValueError('Dataset generation request samples must be a list of objects.')
    if not isinstance(rejections, list) or any(not isinstance(row, dict) for row in rejections):
        raise ValueError('Dataset generation request rejections must be a list of objects.')

    source_root = Path(str(request.get('source_root') or '.')).expanduser().resolve()
    output_root = Path(os.path.abspath(os.path.expanduser(str(output_dir))))
    output_root.mkdir(parents=True, exist_ok=True)
    sources, labels = _matrix_sources(samples)
    path_map, generated_arrays = _materialize_sources(
        sources,
        labels,
        source_root=source_root,
        output_root=output_root,
        allowed_source_root=(
            Path(str(allowed_source_root)).expanduser().resolve()
            if allowed_source_root is not None
            else None
        ),
    )
    _validate_arrays(samples, path_map, output_root)

    rewritten_samples = copy.deepcopy(samples)
    for row in rewritten_samples:
        for field_name in ('fock', 'overlap'):
            reference = row[field_name]
            reference['path'] = path_map[str(reference['path'])]
    (output_root / 'samples.jsonl').write_text(
        _json_lines(rewritten_samples), encoding='utf-8'
    )
    (output_root / 'rejections.jsonl').write_text(
        _json_lines(rejections), encoding='utf-8'
    )

    generated_manifest = copy.deepcopy(source_manifest)
    generated_manifest['artifacts'] = {
        'sample_index': 'samples.jsonl',
        'rejection_index': 'rejections.jsonl',
        'dataset_manifest': 'dataset-manifest.json',
        'matrix_root': 'arrays',
    }
    generated_manifest['generation'] = {
        'status': 'complete',
        'portable': True,
        'location': str(location),
        'source_manifest': str(request.get('source_manifest_path') or ''),
        'trajectory_file_count': len(generated_arrays),
        'files': generated_arrays,
        'integrity_checks': ['file_present', 'npz_array_key', 'array_index', 'matrix_shape'],
        'checksum_validation': False,
    }
    manifest_path = output_root / 'dataset-manifest.json'
    manifest_path.write_text(
        json.dumps(generated_manifest, ensure_ascii=False, indent=2, default=json_default, allow_nan=False),
        encoding='utf-8',
    )
    dataset_files = [
        {
            'path': name,
            'size_bytes': (output_root / name).stat().st_size,
        }
        for name in ('dataset-manifest.json', 'samples.jsonl', 'rejections.jsonl')
    ]
    dataset_files.extend({
        'path': item['path'],
        'size_bytes': item['size_bytes'],
    } for item in generated_arrays)
    spec = source_manifest.get('spec') if isinstance(source_manifest.get('spec'), dict) else {}
    return {
        'schema': HAMILTONIAN_DATASET_GENERATION_SCHEMA,
        'status': 'succeeded',
        'location': str(location),
        'study_id': str(request.get('study_id') or 'study'),
        'dataset_id': str(spec.get('dataset_id') or request.get('study_id') or 'dataset'),
        'dataset_root': str(output_root),
        'manifest_path': str(manifest_path),
        'trajectory_file_count': len(generated_arrays),
        'sample_count': len(rewritten_samples),
        'rejection_count': len(rejections),
        'files': dataset_files,
        'integrity_checks': ['file_present', 'npz_array_key', 'array_index', 'matrix_shape'],
        'checksum_validation': False,
    }


def generate_hamiltonian_dataset(
    report: Mapping[str, Any],
    *,
    output_dir: Any,
    dataset_generator: Optional[Callable[[Mapping[str, Any]], Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate locally or ask the selected executor to generate in place."""

    request = build_hamiltonian_dataset_generation_request(report)
    if callable(dataset_generator):
        raw_generation = dataset_generator(request)
        if not isinstance(raw_generation, Mapping):
            raise ValueError('Dataset generator returned an invalid response.')
        generation = copy.deepcopy(dict(raw_generation))
    else:
        # The successful receipt names one complete version. A failed retry
        # (including receipt persistence) leaves the previous version usable.
        destination = Path(output_dir).expanduser() / ('generation-' + uuid.uuid4().hex)
        generation = materialize_hamiltonian_dataset(
            request,
            output_dir=destination,
            location='local',
        )
    if (
        generation.get('schema') != HAMILTONIAN_DATASET_GENERATION_SCHEMA
        or generation.get('status') != 'succeeded'
    ):
        raise ValueError('Dataset generation did not return a successful generation receipt.')

    work_root = Path(os.path.abspath(os.path.expanduser(str(report.get('work_dir')))))
    receipt_path = work_root / 'postprocessing' / HAMILTONIAN_DATASET_GENERATION_FILENAME
    receipt = copy.deepcopy(generation)
    receipt.update({
        'action': HAMILTONIAN_DATASET_GENERATE_ACTION,
        'generated_at': datetime.now(timezone.utc).isoformat(),
    })
    repository = default_artifact_repository().with_reference_root(work_root)
    receipt_ref = repository.write_json(
        receipt_path,
        receipt,
        kind='hamiltonian_dataset_generation_receipt',
        description='Location and file inventory for a generated Hamiltonian dataset',
    )
    receipt['artifacts'] = [receipt_ref]
    return receipt


def _generation_receipt(report: Mapping[str, Any], work_root: Path) -> Dict[str, Any]:
    path = work_root / 'postprocessing' / HAMILTONIAN_DATASET_GENERATION_FILENAME
    receipt = _load_json(path) if path.is_file() else report.get('dataset_generation')
    if not isinstance(receipt, dict):
        raise ValueError('Generate Dataset must complete before Collect can download it.')
    if (
        receipt.get('schema') != HAMILTONIAN_DATASET_GENERATION_SCHEMA
        or receipt.get('status') != 'succeeded'
    ):
        raise ValueError('Hamiltonian dataset generation receipt is invalid or incomplete.')
    return receipt


def _collect_generated_files(
    receipt: Mapping[str, Any],
    *,
    output_root: Path,
    collector: Optional[Callable[[Mapping[str, str]], Any]],
) -> List[Dict[str, Any]]:
    source_root = Path(str(receipt.get('dataset_root') or '')).expanduser()
    raw_files = receipt.get('files')
    if not str(source_root) or not isinstance(raw_files, list) or not raw_files:
        raise ValueError('Dataset generation receipt has no collectable file inventory.')
    transfers: Dict[str, str] = {}
    collected: List[Dict[str, Any]] = []
    for item in raw_files:
        if not isinstance(item, dict):
            raise ValueError('Dataset generation file inventory is invalid.')
        relative = _safe_relative_path(item.get('path'))
        source = source_root / relative
        target = output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
        else:
            transfers[str(source)] = str(target)
        collected.append({
            'source_path': str(source),
            'path': str(relative),
            'size_bytes': int(item.get('size_bytes') or 0),
        })
    if transfers:
        if not callable(collector):
            raise ValueError(
                'The generated dataset is remote; select the execution target that '
                'generated it before collecting the files.'
            )
        collector(transfers)
    missing = [item['path'] for item in collected if not (output_root / item['path']).is_file()]
    if missing:
        raise FileNotFoundError(
            'Dataset collection did not produce: {0}'.format(', '.join(missing))
        )
    return collected


def collect_hamiltonian_dataset(
    report: Mapping[str, Any],
    *,
    output_dir: Any,
    artifact_collector: Optional[Callable[[Mapping[str, str]], Any]] = None,
) -> Dict[str, Any]:
    """Download an already generated portable dataset into the study directory."""

    if not isinstance(report, Mapping):
        raise TypeError('StudyReport must be a mapping.')
    work_dir = str(report.get('work_dir') or '').strip()
    if not work_dir:
        raise ValueError('StudyReport work_dir is required for dataset collection.')
    work_root = Path(os.path.abspath(os.path.expanduser(work_dir)))
    receipt = _generation_receipt(report, work_root)
    output_root = Path(os.path.abspath(os.path.expanduser(str(output_dir))))
    output_root.mkdir(parents=True, exist_ok=True)
    collected_files = _collect_generated_files(
        receipt,
        output_root=output_root,
        collector=artifact_collector,
    )

    manifest_path = output_root / 'dataset-manifest.json'
    collected_manifest = _load_json(manifest_path)
    if collected_manifest.get('schema') != HAMILTONIAN_MANIFEST_SCHEMA:
        raise ValueError('Collected Hamiltonian dataset manifest schema is invalid.')
    samples = _load_jsonl(output_root / 'samples.jsonl')
    rejections = _load_jsonl(output_root / 'rejections.jsonl')
    relative_sources, _labels = _matrix_sources(samples)
    relative_map = {source: str(_safe_relative_path(source)) for source in relative_sources}
    _validate_arrays(samples, relative_map, output_root)
    collected_manifest['collection'] = {
        'status': 'complete',
        'source_location': receipt.get('location'),
        'source_dataset_root': receipt.get('dataset_root'),
        'trajectory_file_count': int(receipt.get('trajectory_file_count') or 0),
        'file_count': len(collected_files),
        'integrity_checks': ['file_present', 'npz_array_key', 'array_index', 'matrix_shape'],
        'checksum_validation': False,
    }

    repository = default_artifact_repository().with_reference_root(work_root)
    manifest_ref = repository.write_json(
        manifest_path,
        collected_manifest,
        kind='hamiltonian_collected_dataset_manifest',
        description='Locally collected QH9-compatible Hamiltonian dataset manifest',
    )
    sample_ref = repository.register_existing(
        output_root / 'samples.jsonl',
        kind='hamiltonian_collected_sample_index',
        mime_type='application/x-ndjson; charset=utf-8',
        description='Collected Hamiltonian sample index',
    )
    rejection_ref = repository.register_existing(
        output_root / 'rejections.jsonl',
        kind='hamiltonian_collected_rejection_index',
        mime_type='application/x-ndjson; charset=utf-8',
        description='Collected Hamiltonian rejection index',
    )
    array_refs = []
    for source in relative_sources:
        reference = repository.register_existing(
            output_root / relative_map[source],
            kind='hamiltonian_collected_trajectory_arrays',
            mime_type='application/x-npz',
            description='Collected Hamiltonian trajectory arrays',
        )
        if reference is not None and reference not in array_refs:
            array_refs.append(reference)
    artifacts = [manifest_ref]
    artifacts.extend(item for item in (sample_ref, rejection_ref) if item is not None)
    artifacts.extend(array_refs)
    return {
        'action': HAMILTONIAN_DATASET_COLLECT_ACTION,
        'status': 'succeeded',
        'output_dir': str(output_root),
        'source_location': receipt.get('location'),
        'trajectory_file_count': int(receipt.get('trajectory_file_count') or 0),
        'sample_count': len(samples),
        'rejection_count': len(rejections),
        'manifest': collected_manifest,
        'artifacts': artifacts,
    }


__all__ = [
    'HAMILTONIAN_DATASET_COLLECT_ACTION',
    'HAMILTONIAN_DATASET_GENERATE_ACTION',
    'HAMILTONIAN_DATASET_GENERATION_FILENAME',
    'HAMILTONIAN_DATASET_GENERATION_REQUEST_SCHEMA',
    'HAMILTONIAN_DATASET_GENERATION_SCHEMA',
    'build_hamiltonian_dataset_generation_request',
    'collect_hamiltonian_dataset',
    'generate_hamiltonian_dataset',
    'materialize_hamiltonian_dataset',
]
