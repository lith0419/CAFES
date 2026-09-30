from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ..artifacts import default_artifact_repository
from ..identifiers import make_run_id
from ..paths import RUNS_DIR_ENV as RUNS_DIR_ENV, default_runs_root, resolve_work_dir
from ..registry.platform import result_analysis_output_contracts
from ..serialization import sanitize_result_json


PREVIEW_CHARS = 12000
COMPACT_CONTEXT_KEYS = (
    'task_type',
    'representation',
    'model',
    'solver',
    'method',
    'reference',
    'initial_reference',
    'cas_reference',
    'cas_spin_adapted',
    'reference_converged',
    'reference_status',
    'solver_converged',
    'scf_algorithm',
    'restricted_reference',
    'converged',
    'energy_unit',
    'norb',
    'nelec',
    'electrons_per_cell',
    'filling',
    'band_filling_fraction',
    'site_count',
    'bond_count',
    'kmesh',
    'kpoint_count',
    'kpoint_scheme',
    'kpoint_shift',
    'gamma_point',
    'nao',
    'nelectron',
    'basis',
    'pseudo',
    'dimension',
    'nonzero_hopping_count',
    'nonzero_v_count',
    'onsite_u_count',
    'energy_spectrum_method',
    'energy_spectrum_unavailable_reason',
    'cas_ncas',
    'cas_nelecas',
    'active_space_probe',
    'energy_kind',
    'is_metal',
    'partially_filled_band',
    'filled_band_count',
    'valence_band_max',
    'conduction_band_min',
    'interaction_treatment',
    'final_energy',
    'final_method',
    'quality_checks',
    'initial_state',
    'one_particle_state',
)


def compact_result_keys() -> tuple[str, ...]:
    keys = list(COMPACT_CONTEXT_KEYS)
    for result_field, contract in result_analysis_output_contracts().items():
        if contract.get('compact_result') is False:
            continue
        if result_field not in keys:
            keys.append(result_field)
    return tuple(keys)


def ensure_run_id(state: Dict[str, Any]) -> str:
    run_id = state.get('run_id')
    if not isinstance(run_id, str) or not run_id.strip():
        run_id = make_run_id()
        state['run_id'] = run_id
    return run_id


def _runs_root() -> Path:
    return default_runs_root()


def get_run_dir(state: Dict[str, Any]) -> Path:
    run_id = ensure_run_id(state)
    return resolve_work_dir(state.get('work_dir'), run_id) / run_id


def preview_text(value: str, *, max_chars: int = PREVIEW_CHARS) -> str:
    if not isinstance(value, str) or len(value) <= max_chars:
        return value
    return '[truncated: showing last {0} characters]\n{1}'.format(
        max_chars,
        value[-max_chars:],
    )


def retry_artifact_filename(base: str, extension: str, retry_count: Any) -> str:
    """Return a stable artifact filename while preserving numerical retries."""

    try:
        retry_number = int(retry_count)
    except (TypeError, ValueError):
        retry_number = 0
    suffix = '' if retry_number <= 0 else '-retry-{0}'.format(retry_number)
    return '{0}{1}.{2}'.format(base, suffix, extension)


def compact_structured_results(results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(results, dict):
        return {}
    return {
        key: copy.deepcopy(results[key])
        for key in compact_result_keys()
        if key in results
    }


def register_artifact_reference(
    state: Dict[str, Any],
    reference: Mapping[str, Any],
) -> Dict[str, Any]:
    """Keep one current provenance record for each artifact path."""

    ref = copy.deepcopy(dict(reference))
    path = str(ref.get('path') or '')
    if not path:
        raise ValueError('Artifact references require a non-empty path.')
    artifacts = state.setdefault('artifacts', [])
    state['artifacts'] = [
        item for item in artifacts
        if not isinstance(item, dict) or str(item.get('path') or '') != path
    ]
    state['artifacts'].append(ref)
    return ref


def serialize_npz_arrays(
    arrays: Mapping[str, Any],
    *,
    schema: Optional[str] = None,
) -> bytes:
    """Serialize a non-empty numerical mapping as a compressed NPZ payload."""

    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError('NPZ serialization requires at least one named array.')
    import io
    import numpy as np  # pylint: disable=import-outside-toplevel

    payload = dict(arrays)
    if schema:
        payload['schema'] = schema
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **payload)
    return buffer.getvalue()


def register_existing_artifact(
    state: Dict[str, Any],
    kind: str,
    path: Any,
    *,
    mime_type: str = 'application/octet-stream',
    description: str = '',
) -> Optional[Dict[str, Any]]:
    """Register a provider-written file under the shared artifact contract."""
    ref = default_artifact_repository().register_existing(
        path,
        kind=kind,
        mime_type=mime_type,
        description=description,
    )
    if ref is None:
        return None
    return register_artifact_reference(state, ref)


def write_text_artifact(
    state: Dict[str, Any],
    kind: str,
    filename: str,
    content: str,
    *,
    mime_type: str = 'text/plain; charset=utf-8',
    description: str = '',
) -> Optional[Dict[str, Any]]:
    if not isinstance(content, str) or not content:
        return None
    run_dir = get_run_dir(state)
    run_dir.mkdir(parents=True, exist_ok=True)
    ref = default_artifact_repository().write_text(
        run_dir / filename,
        content,
        kind=kind,
        mime_type=mime_type,
        description=description,
    )
    return register_artifact_reference(state, ref)


def write_binary_artifact(
    state: Dict[str, Any],
    kind: str,
    filename: str,
    content: bytes,
    *,
    mime_type: str = 'application/octet-stream',
    description: str = '',
) -> Optional[Dict[str, Any]]:
    """Write a registered non-text artifact with the standard provenance ref."""
    if not isinstance(content, (bytes, bytearray)) or not content:
        return None
    run_dir = get_run_dir(state)
    run_dir.mkdir(parents=True, exist_ok=True)
    ref = default_artifact_repository().write_bytes(
        run_dir / filename,
        bytes(content),
        kind=kind,
        mime_type=mime_type,
        description=description,
    )
    return register_artifact_reference(state, ref)


def write_json_artifact(
    state: Dict[str, Any],
    kind: str,
    filename: str,
    payload: Any,
    *,
    description: str = '',
) -> Dict[str, Any]:
    run_dir = get_run_dir(state)
    run_dir.mkdir(parents=True, exist_ok=True)
    ref = default_artifact_repository().write_json(
        run_dir / filename,
        payload,
        kind=kind,
        description=description,
    )
    return register_artifact_reference(state, ref)


def write_result_json_artifact(
    state: Dict[str, Any],
    kind: str,
    filename: str,
    payload: Any,
    *,
    description: str = '',
) -> Dict[str, Any]:
    """Persist available results without letting one nonfinite diagnostic abort a task."""
    cleaned, warnings = sanitize_result_json(payload)
    reference = write_json_artifact(state, kind, filename, cleaned, description=description)
    recorded = state.setdefault('warnings', [])
    for warning in warnings:
        warning['artifact_path'] = reference['path']
        if warning not in recorded:
            recorded.append(warning)
    return reference
