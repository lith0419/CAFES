from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from .fci_observables import (
    _compute_charge_correlation_from_spin_rdms,
    _compute_double_occupancy_from_spin_rdms,
    _compute_spin_correlation_from_spin_rdms,
)


DMET_LOCAL_OBSERVABLES_SCHEMA = 'pyscf-agent.dmet-local-observables.v1'


def _mean(values: List[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def _site_index(spec: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    sites = sorted(
        (site for site in spec.get('sites', []) if isinstance(site, dict) and 'id' in site),
        key=lambda item: int(item['id']),
    )
    return sites, {int(site['id']): position for position, site in enumerate(sites)}


def _global_spin_rdm1(arrays: Mapping[str, Any], restricted: bool, np: Any) -> Optional[Tuple[Any, Any]]:
    density = arrays.get('global_embedding_density_matrix_full')
    if density is None:
        density = arrays.get('global_embedding_density_matrix')
    if density is None:
        return None
    density = np.asarray(density)
    if density.ndim == 4:
        density = density[:, 0]
    if density.ndim != 3 or density.shape[-1] != density.shape[-2]:
        return None
    if restricted:
        return density[0], density[0]
    if density.shape[0] < 2:
        return None
    return density[0], density[1]


def _fragment_array_keys(position: int, arrays: Mapping[str, Any]) -> Tuple[str, str]:
    if position == 0 and 'fragment_density_matrix' in arrays:
        return 'fragment_density_matrix', 'fragment_spin_rdm2'
    prefix = 'fragment_{0}'.format(position + 1)
    return '{0}_density_matrix'.format(prefix), '{0}_spin_rdm2'.format(prefix)


def _averaged_vector(buckets: List[List[float]]) -> List[Optional[float]]:
    return [_mean(values) for values in buckets]


def _averaged_matrix(buckets: List[List[List[float]]]) -> List[List[Optional[float]]]:
    return [[_mean(values) for values in row] for row in buckets]


def _sublattice_summary(
    sites: List[Dict[str, Any]],
    density: List[float],
    magnetization: List[float],
) -> Optional[Dict[str, Any]]:
    groups: Dict[str, Dict[str, List[float]]] = {}
    for position, site in enumerate(sites):
        label = site.get('sublattice')
        if label in (None, ''):
            continue
        group = groups.setdefault(str(label), {'density': [], 'magnetization': []})
        group['density'].append(float(density[position]))
        group['magnetization'].append(float(magnetization[position]))
    if len(groups) < 2:
        return None
    summaries = {
        label: {
            'site_count': len(values['density']),
            'mean_density': _mean(values['density']),
            'mean_magnetization': _mean(values['magnetization']),
        }
        for label, values in sorted(groups.items())
    }
    density_means = [float(item['mean_density']) for item in summaries.values()]
    magnetization_means = [float(item['mean_magnetization']) for item in summaries.values()]
    return {
        'groups': summaries,
        # For two sublattices these are the conventional half-differences.
        'charge_imbalance': 0.5 * (max(density_means) - min(density_means)),
        'staggered_magnetization': 0.5 * (
            max(magnetization_means) - min(magnetization_means)
        ),
    }


def assemble_dmet_local_observables(
    spec: Mapping[str, Any],
    dmet_result: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble model observables from global DMET 1RDM and fragment 2RDMs."""

    import numpy as np  # pylint: disable=import-outside-toplevel

    sites, site_to_index = _site_index(spec)
    site_count = len(sites)
    configuration = dmet_result.get('configuration') or {}
    restricted = str(configuration.get('reference') or '').lower() == 'restricted'
    spin_rdm1 = _global_spin_rdm1(arrays, restricted, np)
    if spin_rdm1 is None or spin_rdm1[0].shape != (site_count, site_count):
        return {
            'schema': DMET_LOCAL_OBSERVABLES_SCHEMA,
            'status': 'unavailable',
            'reason': 'The assembled DMET one-particle density matrix is unavailable or has the wrong shape.',
        }

    dm1a, dm1b = spin_rdm1
    total_rdm1 = np.asarray(dm1a) + np.asarray(dm1b)
    density = np.diag(total_rdm1).real.astype(float).tolist()
    magnetization = np.diag(np.asarray(dm1a) - np.asarray(dm1b)).real.astype(float).tolist()

    double_buckets: List[List[float]] = [[] for _index in range(site_count)]
    double_reference_buckets: List[List[float]] = [[] for _index in range(site_count)]
    spin_buckets = [[[] for _target in range(site_count)] for _source in range(site_count)]
    charge_buckets = [[[] for _target in range(site_count)] for _source in range(site_count)]
    rdm2_fragment_count = 0

    fragments = dmet_result.get('fragments') if isinstance(dmet_result.get('fragments'), list) else []
    for position, fragment in enumerate(fragments):
        if not isinstance(fragment, dict):
            continue
        density_key, rdm2_key = _fragment_array_keys(position, arrays)
        local_density = arrays.get(density_key)
        local_rdm2 = arrays.get(rdm2_key)
        site_ids = [int(value) for value in fragment.get('site_ids', [])]
        if local_density is None or local_rdm2 is None or not site_ids:
            continue
        local_density = np.asarray(local_density)
        local_rdm2 = np.asarray(local_rdm2)
        if local_density.ndim != 3 or local_density.shape[-1] != len(site_ids):
            continue
        if local_rdm2.ndim != 5 or local_rdm2.shape[-1] != len(site_ids):
            continue

        if local_density.shape[0] == 1:
            local_dm1a = local_density[0]
            local_dm1b = local_density[0]
        elif local_density.shape[0] >= 2:
            local_dm1a = local_density[0]
            local_dm1b = local_density[1]
        else:
            continue

        if local_rdm2.shape[0] == 1:
            # A restricted spin-summed 2RDM has <n_i(n_i-1)> = 2<n_ia n_ib>.
            local_double = [
                0.5 * float(local_rdm2[0, index, index, index, index].real)
                for index in range(len(site_ids))
            ]
            local_spin = None
            local_charge = None
        elif local_rdm2.shape[0] >= 3:
            local_double = _compute_double_occupancy_from_spin_rdms(local_rdm2[2])
            local_spin = _compute_spin_correlation_from_spin_rdms(
                local_dm1a,
                local_dm1b,
                local_rdm2[0],
                local_rdm2[2],
                local_rdm2[1],
            )
            local_charge = _compute_charge_correlation_from_spin_rdms(
                local_dm1a,
                local_dm1b,
                local_rdm2[0],
                local_rdm2[2],
                local_rdm2[1],
            )
        else:
            continue

        mapped = [site_to_index.get(site_id) for site_id in site_ids]
        if any(index is None for index in mapped):
            continue
        rdm2_fragment_count += 1
        for local_index, global_index in enumerate(mapped):
            double_buckets[global_index].append(float(local_double[local_index]))
            double_reference_buckets[global_index].append(float(
                local_dm1a[local_index, local_index].real
                * local_dm1b[local_index, local_index].real
            ))
        if local_spin is not None and local_charge is not None:
            for local_source, global_source in enumerate(mapped):
                for local_target, global_target in enumerate(mapped):
                    spin_buckets[global_source][global_target].append(
                        float(local_spin[local_source][local_target])
                    )
                    charge_buckets[global_source][global_target].append(
                        float(local_charge[local_source][local_target])
                    )

    double_occupancy = _averaged_vector(double_buckets)
    spin_correlation = _averaged_matrix(spin_buckets)
    charge_correlation = _averaged_matrix(charge_buckets)

    spin_bond_values = []
    charge_bond_values = []
    coherence_values = []
    bond_term_count = 0
    for bond in spec.get('bonds', []):
        if not isinstance(bond, dict):
            continue
        try:
            source = site_to_index.get(int(bond.get('source', -1)))
            target = site_to_index.get(int(bond.get('target', -1)))
        except (TypeError, ValueError):
            continue
        if source is None or target is None:
            continue
        bond_term_count += 1
        coherence_values.append(float(abs(total_rdm1[source, target])))
        if spin_correlation[source][target] is not None:
            spin_bond_values.append(float(spin_correlation[source][target]))
        if charge_correlation[source][target] is not None:
            charge_bond_values.append(float(charge_correlation[source][target]))

    covered_sites = sum(value is not None for value in double_occupancy)
    sublattice = _sublattice_summary(sites, density, magnetization)
    return {
        'schema': DMET_LOCAL_OBSERVABLES_SCHEMA,
        'status': 'available',
        'site_ids': [int(site['id']) for site in sites],
        'density': density,
        'local_magnetization': magnetization,
        'double_occupancy': double_occupancy,
        'double_occupancy_reference': _averaged_vector(double_reference_buckets),
        'double_occupancy_reference_source': 'same_fragment_spin_1rdm',
        'spin_correlation': spin_correlation,
        'charge_correlation': charge_correlation,
        'mean_double_occupancy': _mean([
            float(value) for value in double_occupancy if value is not None
        ]),
        'nearest_neighbor_spin_correlation': _mean(spin_bond_values),
        'nearest_neighbor_charge_correlation': _mean(charge_bond_values),
        'mean_nearest_neighbor_one_body_coherence': _mean(coherence_values),
        'sublattice_order': sublattice,
        'coverage': {
            'site_count': site_count,
            'double_occupancy_site_count': covered_sites,
            'double_occupancy_fraction': (
                float(covered_sites / site_count) if site_count else 0.0
            ),
            'bond_term_count': bond_term_count,
            'spin_correlation_bond_term_count': len(spin_bond_values),
            'spin_correlation_fraction': (
                float(len(spin_bond_values) / bond_term_count) if bond_term_count else 0.0
            ),
            'charge_correlation_bond_term_count': len(charge_bond_values),
            'charge_correlation_fraction': (
                float(len(charge_bond_values) / bond_term_count) if bond_term_count else 0.0
            ),
            'rdm2_fragment_count': rdm2_fragment_count,
        },
    }


__all__ = ['DMET_LOCAL_OBSERVABLES_SCHEMA', 'assemble_dmet_local_observables']
