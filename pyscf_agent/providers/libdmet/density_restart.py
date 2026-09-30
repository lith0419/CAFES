"""Finite-graph DMET density warm starts with explicit artifact provenance."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path

STATE_SCHEMA = 'pyscf-agent.dmet-mean-field-state.v1'
STATE_KIND = 'dmet_mean_field_state'


def normalize_density_source(value):
    if value is None:
        return None
    allowed = {'source_case_id', 'source_run_id', 'path', 'sha256'}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError(
            'reference_density_source requires source_case_id or a pinned artifact path'
        )
    result = copy.deepcopy(value)
    if any(not isinstance(item, str) or not item.strip() for item in result.values()):
        raise ValueError('reference_density_source fields must be nonempty strings')
    if result.get('path'):
        if not Path(result['path']).expanduser().is_absolute():
            raise ValueError('reference_density_source.path must be absolute')
        digest = result.get('sha256', '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('reference_density_source.path requires a sha256 checksum')
    elif not result.get('source_case_id') or result.get('sha256'):
        raise ValueError(
            'Deferred reference_density_source requires source_case_id and no checksum'
        )
    return result


def state_metadata(spec, configuration, *, converged):
    """Bind the site basis and numerical policy; allow only U/V to vary."""
    sites = sorted(spec['sites'], key=lambda item: int(item['id']))
    bonds = sorted(
        spec['bonds'],
        key=lambda item: (
            int(item['source']),
            int(item['target']),
            int(item.get('id', 0)),
        ),
    )
    graph = copy.deepcopy(spec)
    if isinstance(graph.get('globals'), dict):
        graph['globals'] = {
            key: value
            for key, value in graph['globals'].items()
            if key not in ('U', 'V')
        }
    graph['sites'] = [{k: v for k, v in site.items() if k != 'U'} for site in sites]
    graph['bonds'] = [
        {k: v for k, v in bond.items() if k not in ('V', 'effective_V')}
        for bond in bonds
    ]
    # Fragment labels do not define the basis, but membership and ordering do.
    compatibility = {
        'graph': graph,
        'fragments': [fragment['site_ids'] for fragment in configuration['fragments']],
        **{
            key: configuration[key]
            for key in (
                'reference',
                'nalpha',
                'nbeta',
                'execution_mode',
                'interacting_bath',
                'impurity_solver',
                'impurity_solver_options',
                'bath_spin_dimension_policy',
            )
        },
    }
    return {
        'schema': STATE_SCHEMA,
        'representation': 'site_spin_1rdm',
        'converged': bool(converged),
        'compatibility': compatibility,
        'site_ids': [int(site['id']) for site in sites],
        'parameters': {
            'onsite_u': [float(site.get('U', 0)) for site in sites],
            'intersite_v': [
                float(bond.get('effective_V', bond.get('V', 0))) for bond in bonds
            ],
        },
        'source_seed': configuration['reference_density_guess'],
    }


def validate_density(density, metadata, *, np):
    rho = np.asarray(density)
    n = len(metadata['site_ids'])
    if rho.shape != (2, n, n) or not np.all(np.isfinite(rho)):
        raise ValueError(
            'DMET source density has an invalid shape or nonfinite entries'
        )
    if np.max(np.abs(rho.imag), initial=0) > 1e-10:
        raise ValueError(
            'DMET density warm starts currently require a real site density'
        )
    rho = rho.real.astype(float)
    if not np.allclose(rho, rho.transpose(0, 2, 1), atol=1e-9, rtol=0):
        raise ValueError('DMET source density is not Hermitian')
    occupations = np.linalg.eigvalsh(rho)
    if np.any(occupations < -1e-7) or np.any(occupations > 1 + 1e-7):
        raise ValueError('DMET source density occupations are outside [0, 1]')
    compatibility = metadata['compatibility']
    counts = [compatibility['nalpha'], compatibility['nbeta']]
    if not np.allclose(np.trace(rho, axis1=1, axis2=2), counts, atol=1e-5, rtol=0):
        raise ValueError(
            'DMET source density does not preserve the requested spin particle counts'
        )
    if compatibility['reference'] == 'restricted' and not np.allclose(
        rho[0], rho[1], atol=1e-9, rtol=0
    ):
        raise ValueError('Restricted DMET requires identical source spin densities')
    return rho.copy()


def serialize_density_state(metadata, density, *, run_id):
    import numpy as np

    metadata = copy.deepcopy(metadata)
    metadata['source_run_id'] = str(run_id)
    rho = validate_density(density, metadata, np=np)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        metadata_json=json.dumps(metadata, sort_keys=True, allow_nan=False),
        mean_field_density_matrix=rho,
    )
    return buffer.getvalue()


def load_density_seed(spec, configuration, *, np):
    source = normalize_density_source(configuration.get('reference_density_source'))
    if not source or not source.get('path'):
        raise ValueError(
            'DMET density source must be resolved to a pinned artifact before execution'
        )
    path = Path(source['path']).expanduser()
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != source['sha256']:
        raise ValueError('DMET source density checksum mismatch')
    try:
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            metadata = json.loads(archive['metadata_json'].item())
            density = archive['mean_field_density_matrix']
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise ValueError(
            'Invalid DMET mean-field state archive; old density seeds cannot be used'
        ) from exc
    if not isinstance(metadata, dict) or metadata.get('schema') != STATE_SCHEMA:
        raise ValueError('Unsupported DMET mean-field state schema')
    if (
        metadata.get('converged') is not True
        or metadata.get('quality_passed') is not True
        or not metadata.get('source_run_id')
    ):
        raise ValueError('DMET warm start requires a converged source Run')
    if (
        source.get('source_run_id')
        and source['source_run_id'] != metadata['source_run_id']
    ):
        raise ValueError('DMET density source Run ID mismatch')
    target = state_metadata(spec, configuration, converged=False)
    if (
        metadata.get('compatibility') != target['compatibility']
        or metadata.get('site_ids') != target['site_ids']
    ):
        raise ValueError(
            'Incompatible DMET density source: require the same site basis, fragments, spin sector and solver policy; only U/V may change'
        )
    rho = validate_density(density, metadata, np=np)
    source_parameters = metadata['parameters']
    changes = {
        key: (np.asarray(target['parameters'][key]) - source_parameters[key]).tolist()
        for key in target['parameters']
    }
    return rho, {
        'strategy': 'neighbor_mean_field_density',
        'source_case_id': source.get('source_case_id'),
        'source_run_id': metadata['source_run_id'],
        'source_artifact': {'kind': STATE_KIND, 'path': str(path), 'sha256': digest},
        'source_seed': metadata.get('source_seed'),
        'source_parameters': source_parameters,
        'target_parameters': target['parameters'],
        'parameter_changes': changes,
        'max_abs_parameter_change': {
            key: float(np.max(np.abs(values), initial=0))
            for key, values in changes.items()
        },
        'site_ids': target['site_ids'],
        'transfer': 'mean_field_density_only',
        'physical_baseline': 'recomputed_with_target_hamiltonian',
    }
