"""Portable one-particle restart states for molecular continuation workflows.

The public TaskSpec carries only an artifact reference.  AO density matrices
remain binary run artifacts and are projected onto the target AO basis at
execution time.  This keeps restart data reusable across molecular methods
without exposing large numerical arrays to planners or LLM prompts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .artifacts import serialize_npz_arrays
from ..contracts import TaskSpec


ONE_PARTICLE_STATE_SCHEMA = 'pyscf-agent.one-particle-state.v1'
ONE_PARTICLE_STATE_MODE = 'projected_1rdm'


def _atom_symbols(atom: Any) -> list[str]:
    if not isinstance(atom, str):
        return []
    symbols = []
    for entry in atom.replace('\n', ';').split(';'):
        fields = entry.strip().split()
        if fields:
            symbols.append(fields[0])
    return symbols


def _reference_kind(task_spec: TaskSpec) -> str:
    # CAS uses a UHF starting reference even when the CAS optimizer itself
    # is spin-adapted. Restart compatibility concerns that starting SCF.
    if task_spec.method.name in ('casci', 'casscf'):
        return 'uhf'
    return 'rhf' if task_spec.method.restricted else 'uhf'


def _source_metadata(task_spec: TaskSpec, mf: Any) -> Dict[str, Any]:
    return {
        'schema': ONE_PARTICLE_STATE_SCHEMA,
        'representation': 'ao',
        'state_kind': 'scf_reference_1rdm',
        'task_type': 'molecular',
        'source_method': str(task_spec.method.name or ''),
        'reference': _reference_kind(task_spec),
        'restricted': _reference_kind(task_spec) != 'uhf',
        'system': {
            'atom': task_spec.system.atom,
            'atom_symbols': _atom_symbols(task_spec.system.atom),
            'basis': task_spec.system.basis,
            'unit': task_spec.system.unit,
            'charge': task_spec.system.charge,
            'spin': task_spec.system.spin,
            'symmetry': bool(task_spec.system.symmetry),
        },
        'nao': int(getattr(getattr(mf, 'mol', None), 'nao_nr', lambda: 0)() or 0),
        'scf_converged': bool(getattr(mf, 'converged', False)),
    }


def capture_reference_1rdm(task_spec: TaskSpec, mf: Any) -> Optional[Dict[str, Any]]:
    """Capture the last molecular SCF density, including an unconverged guess.

    Convergence is recorded separately: a usable restart density is not evidence
    that the source calculation succeeded.
    """
    if task_spec.task_type != 'molecular':
        return None
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel

        density = mf.make_rdm1()
        metadata = _source_metadata(task_spec, mf)
        arrays: Dict[str, Any]
        spin_resolved_density = (
            isinstance(density, (tuple, list))
            and len(density) == 2
        ) or (
            isinstance(density, np.ndarray)
            and density.ndim == 3
            and density.shape[0] == 2
        )
        if spin_resolved_density:
            arrays = {
                'dm_alpha': np.asarray(density[0], dtype=float),
                'dm_beta': np.asarray(density[1], dtype=float),
            }
            metadata['spin_mode'] = 'unrestricted'
        else:
            arrays = {'dm_restricted': np.asarray(density, dtype=float)}
            metadata['spin_mode'] = 'restricted'
        if any(
            matrix.shape != (metadata['nao'], metadata['nao'])
            or not np.all(np.isfinite(matrix))
            for matrix in arrays.values()
        ):
            raise ValueError('Reference 1RDM has incompatible dimensions or nonfinite entries')
        metadata['array_keys'] = sorted(arrays)
        metadata['array_shapes'] = {key: list(value.shape) for key, value in arrays.items()}
        return {'status': 'available', 'metadata': metadata, 'arrays': arrays}
    except Exception as exc:  # Optional evidence must not invalidate a completed energy.
        return {'status': 'failed', 'quantity': 'reference_1rdm', 'reason': str(exc)}


def one_particle_state_summary(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    metadata = payload.get('metadata')
    if not isinstance(metadata, dict):
        return None
    return {
        'schema': metadata.get('schema') or ONE_PARTICLE_STATE_SCHEMA,
        'representation': metadata.get('representation'),
        'state_kind': metadata.get('state_kind'),
        'source_method': metadata.get('source_method'),
        'reference': metadata.get('reference'),
        'spin_mode': metadata.get('spin_mode'),
        'nao': metadata.get('nao'),
        'scf_converged': metadata.get('scf_converged'),
    }


def serialize_one_particle_state(payload: Dict[str, Any]) -> bytes:
    """Serialize the numerical state to a compressed NumPy archive."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    metadata = payload.get('metadata') if isinstance(payload, dict) else None
    arrays = payload.get('arrays') if isinstance(payload, dict) else None
    if not isinstance(metadata, dict) or not isinstance(arrays, dict) or not arrays:
        raise ValueError('OneParticleState requires metadata and at least one density matrix.')
    return serialize_npz_arrays(
        {
            'metadata_json': np.asarray(
                json.dumps(metadata, ensure_ascii=False, sort_keys=True)
            ),
            **{key: np.asarray(value, dtype=float) for key, value in arrays.items()},
        }
    )


def _load_one_particle_state(path: Any) -> Dict[str, Any]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    candidate = Path(str(path or '')).expanduser()
    if not candidate.is_file():
        raise ValueError('Initial-state artifact does not exist: {0}'.format(candidate))
    try:
        with np.load(candidate, allow_pickle=False) as archive:
            metadata_raw = archive['metadata_json'].item()
            metadata = json.loads(str(metadata_raw))
            arrays = {
                key: np.asarray(archive[key], dtype=float)
                for key in archive.files
                if key != 'metadata_json'
            }
    except Exception as exc:
        raise ValueError('Initial-state artifact is not a valid one-particle-state archive.') from exc
    if not isinstance(metadata, dict) or metadata.get('schema') != ONE_PARTICLE_STATE_SCHEMA:
        raise ValueError('Initial-state artifact has an unsupported one-particle-state schema.')
    if not arrays:
        raise ValueError('Initial-state artifact does not contain a density matrix.')
    return {'metadata': metadata, 'arrays': arrays}


def _compatible_metadata(task_spec: TaskSpec, metadata: Dict[str, Any]) -> None:
    if metadata.get('task_type') != 'molecular' or metadata.get('representation') != 'ao':
        raise ValueError('Projected 1RDM restart requires a molecular AO one-particle-state artifact.')
    source_system = metadata.get('system') if isinstance(metadata.get('system'), dict) else {}
    source_symbols = source_system.get('atom_symbols') or _atom_symbols(source_system.get('atom'))
    target_symbols = _atom_symbols(task_spec.system.atom)
    if source_symbols != target_symbols:
        raise ValueError('Projected 1RDM restart requires the same atom ordering and element sequence.')
    if str(source_system.get('basis') or '').strip().lower() != str(task_spec.system.basis or '').strip().lower():
        raise ValueError('Projected 1RDM restart requires the same molecular basis set.')
    if int(source_system.get('charge') or 0) != int(task_spec.system.charge or 0):
        raise ValueError('Projected 1RDM restart requires the same molecular charge.')
    if int(source_system.get('spin') or 0) != int(task_spec.system.spin or 0):
        raise ValueError('Projected 1RDM restart requires the same molecular spin.')
    if bool(metadata.get('restricted')) != (_reference_kind(task_spec) != 'uhf'):
        raise ValueError('Projected 1RDM restart requires the same restricted/unrestricted reference policy.')


def projected_initial_density(task_spec: TaskSpec, target_mol: Any) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """Load and AO-project an artifact state for a target molecular calculation."""
    initial_state = task_spec.initial_state
    if initial_state.mode == 'none':
        return None, None
    if initial_state.mode != ONE_PARTICLE_STATE_MODE:
        raise ValueError('Unsupported initial_state.mode: {0}'.format(initial_state.mode))
    if task_spec.task_type != 'molecular':
        raise ValueError('Projected 1RDM restart is currently supported for molecular tasks only.')
    artifact = initial_state.source_artifact if isinstance(initial_state.source_artifact, dict) else {}
    if artifact.get('kind') != 'one_particle_state' or not artifact.get('path'):
        raise ValueError('Projected 1RDM restart requires a one_particle_state artifact reference.')
    payload = _load_one_particle_state(artifact.get('path'))
    metadata = payload['metadata']
    _compatible_metadata(task_spec, metadata)

    from pyscf import gto, scf  # pylint: disable=import-outside-toplevel

    source_system = metadata['system']
    source_mol = gto.M(
        atom=source_system.get('atom'),
        basis=source_system.get('basis'),
        unit=source_system.get('unit') or 'Angstrom',
        charge=int(source_system.get('charge') or 0),
        spin=int(source_system.get('spin') or 0),
        symmetry=bool(source_system.get('symmetry', False)),
        verbose=0,
    )
    arrays = payload['arrays']
    if metadata.get('spin_mode') == 'unrestricted':
        if 'dm_alpha' not in arrays or 'dm_beta' not in arrays:
            raise ValueError('Unrestricted one-particle-state artifact is missing alpha or beta density.')
        density = (
            scf.addons.project_dm_nr2nr(source_mol, arrays['dm_alpha'], target_mol),
            scf.addons.project_dm_nr2nr(source_mol, arrays['dm_beta'], target_mol),
        )
    else:
        if 'dm_restricted' not in arrays:
            raise ValueError('Restricted one-particle-state artifact is missing its density matrix.')
        density = scf.addons.project_dm_nr2nr(source_mol, arrays['dm_restricted'], target_mol)
    return density, {
        'mode': ONE_PARTICLE_STATE_MODE,
        'status': 'applied',
        'source_case_id': initial_state.source_case_id,
        'source_artifact': {
            'kind': artifact.get('kind'),
            'path': artifact.get('path'),
        },
        'source_method': metadata.get('source_method'),
        'source_reference': metadata.get('reference'),
        'source_scf_converged': metadata.get('scf_converged'),
        'projection': 'ao_overlap',
    }
