from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .active_space_tables import (
    active_space_audit_table as active_space_audit_table,
    active_space_summary_table as active_space_summary_table,
    orbital_summary_table as orbital_summary_table,
)
from .active_space_selection import (
    build_active_space_selection_evidence,
    evaluate_active_space_candidates,
    recommend_active_space_candidate,
)
from .cas_execution import (
    CAS_METHODS as CAS_METHODS,
    _active_orbital_indices_payload,
    configure_fci_solver as configure_fci_solver,
    configure_single_root_fci_solver as configure_single_root_fci_solver,
    run_cas_method as run_cas_method,
    state_target_configuration as state_target_configuration,
)
from .chemical_valence import chemical_valence_targets
from .orbital_diagnostics import (
    _ao_natural_orbitals,
    natural_orbital_summary as _natural_orbital_summary,
    t2_orbital_importance as _t2_orbital_importance,
)
from ...contracts import ActiveSpaceSpec
from ..script_helpers import _extract_homo_lumo
from .orbital_diagnostics import occupation_fractionality as _occupation_fractionality  # noqa: F401 - compatibility export



T2_SUPPLEMENT_THRESHOLD = 0.05
T2_SUPPLEMENT_ENERGY_PADDING = 0.5
NATURAL_OCCUPATION_PHYSICAL_TOLERANCE = 0.02
DEGENERATE_MANIFOLD_TOLERANCE = 0.02


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _canonical_orbital_table(mf: Any) -> List[Dict[str, Any]]:
    """Return one spin-summed row per canonical MO index."""
    mo_energy = getattr(mf, 'mo_energy', None)
    mo_occ = getattr(mf, 'mo_occ', None)
    if hasattr(mo_energy, 'tolist'):
        mo_energy = mo_energy.tolist()
    if hasattr(mo_occ, 'tolist'):
        mo_occ = mo_occ.tolist()

    if isinstance(mo_energy, (list, tuple)) and mo_energy and isinstance(mo_energy[0], (list, tuple)):
        nmo = max(len(spin_energy) for spin_energy in mo_energy)
        rows = []
        for index in range(nmo):
            energies = [
                _as_float(spin_energy[index])
                for spin_energy in mo_energy
                if isinstance(spin_energy, (list, tuple)) and len(spin_energy) > index
            ]
            occupations = []
            spin_occupations = mo_occ if isinstance(mo_occ, (list, tuple)) else []
            for spin_occ in spin_occupations:
                if isinstance(spin_occ, (list, tuple)) and len(spin_occ) > index:
                    occ = _as_float(spin_occ[index])
                    if occ is not None:
                        occupations.append(occ)
            rows.append({
                'index': index,
                'spin': 'spatial',
                'energy': sum(energies) / len(energies) if energies else None,
                'occupation': sum(occupations) if occupations else None,
                'occupation_alpha': occupations[0] if len(occupations) >= 1 else None,
                'occupation_beta': occupations[1] if len(occupations) >= 2 else None,
            })
        return rows

    rows = []
    energies = mo_energy if isinstance(mo_energy, (list, tuple)) else []
    occupations = mo_occ if isinstance(mo_occ, (list, tuple)) else []
    for index, energy in enumerate(energies):
        occ = occupations[index] if len(occupations) > index else None
        rows.append({
            'index': index,
            'spin': None,
            'energy': _as_float(energy),
            'occupation': _as_float(occ),
        })
    return rows


def _natural_occupation_quality(natural_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Check whether a correlated 1-RDM is usable for orbital selection.

    Low-order perturbative densities can become non-N-representable in a
    strongly correlated region. Those occupations still diagnose MP2 stress,
    but they must not be used to expand an executable CAS indiscriminately.
    """
    payload: Dict[str, Any] = {
        'status': 'unavailable',
        'tolerance': NATURAL_OCCUPATION_PHYSICAL_TOLERANCE,
        'out_of_bounds': [],
    }
    if natural_summary.get('status') != 'available':
        return payload
    out_of_bounds = []
    for orbital in natural_summary.get('orbitals') or []:
        occupation = _as_float(orbital.get('occupation'))
        if occupation is None:
            continue
        if occupation < -NATURAL_OCCUPATION_PHYSICAL_TOLERANCE or occupation > 2.0 + NATURAL_OCCUPATION_PHYSICAL_TOLERANCE:
            out_of_bounds.append({
                'natural_orbital_index': orbital.get('natural_orbital_index'),
                'occupation': occupation,
            })
    payload['out_of_bounds'] = out_of_bounds
    payload['status'] = 'out_of_bounds' if out_of_bounds else 'physical'
    return payload


def _screen_t2_supplements(
    t2_importance: Dict[int, float],
    selected_indices: Sequence[int],
    table_by_index: Dict[int, Dict[str, Any]],
    natural_occupation_quality: Dict[str, Any],
) -> Dict[str, Any]:
    """Keep T2 supplements local to occupation-selected frontier orbitals."""
    payload: Dict[str, Any] = {
        'status': 'unavailable' if not t2_importance else 'available',
        'threshold': T2_SUPPLEMENT_THRESHOLD,
        'energy_padding': T2_SUPPLEMENT_ENERGY_PADDING,
        'accepted_indices': [],
        'excluded': [],
    }
    if not t2_importance:
        return payload
    if natural_occupation_quality.get('status') == 'out_of_bounds':
        payload['status'] = 'skipped_unphysical_natural_occupations'
        payload['excluded'] = [
            {
                'index': int(index),
                'importance': importance,
                'reason': 'correlated natural occupations are outside the physical [0, 2] range',
            }
            for index, importance in sorted(t2_importance.items())
            if importance >= T2_SUPPLEMENT_THRESHOLD
        ]
        return payload
    anchor_energies = [
        _as_float(table_by_index.get(int(index), {}).get('energy'))
        for index in selected_indices
    ]
    anchor_energies = [energy for energy in anchor_energies if energy is not None]
    if not anchor_energies:
        payload['status'] = 'skipped_without_frontier_anchor'
        payload['excluded'] = [
            {
                'index': int(index),
                'importance': importance,
                'reason': 'no occupation-selected frontier orbital is available as an energy anchor',
            }
            for index, importance in sorted(t2_importance.items())
            if importance >= T2_SUPPLEMENT_THRESHOLD
        ]
        return payload
    for index, importance in sorted(t2_importance.items()):
        if importance < T2_SUPPLEMENT_THRESHOLD:
            continue
        energy = _as_float(table_by_index.get(int(index), {}).get('energy'))
        if energy is None:
            payload['excluded'].append({
                'index': int(index),
                'importance': importance,
                'reason': 'canonical orbital energy is unavailable',
            })
            continue
        distance = min(abs(energy - anchor) for anchor in anchor_energies)
        if distance <= T2_SUPPLEMENT_ENERGY_PADDING:
            payload['accepted_indices'].append(int(index))
        else:
            payload['excluded'].append({
                'index': int(index),
                'importance': importance,
                'energy_distance_to_frontier': distance,
                'reason': 'outside the T2 frontier-energy locality window',
            })
    return payload


def _ao_atom_fragment_contributions(mf: Any, mol: Any, orbital_index: int, limit: int = 5) -> Dict[str, Any]:
    contribution: Dict[str, Any] = {
        'ao_contributions': [],
        'atom_contributions': [],
        'fragment_contributions': [],
        'fragment_definition': 'atom_as_fragment',
    }
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel

        mo_coeff = getattr(mf, 'mo_coeff', None)
        if isinstance(mo_coeff, (list, tuple)) and mo_coeff:
            mo_coeff = mo_coeff[0]
            contribution['coefficient_source'] = 'alpha'
        matrix = np.asarray(mo_coeff, dtype=float)
        if matrix.ndim != 2 or orbital_index < 0 or orbital_index >= matrix.shape[1]:
            return contribution
        weights = np.abs(matrix[:, orbital_index]) ** 2
        total = float(weights.sum())
        if total > 0:
            weights = weights / total
        labels = []
        if mol is not None and hasattr(mol, 'ao_labels'):
            labels = mol.ao_labels(fmt=False)
        ao_rows = []
        atom_weights: Dict[str, float] = {}
        for ao_index, weight in enumerate(weights):
            atom_index: Any = None
            atom_symbol = ''
            ao_label = str(ao_index)
            if isinstance(labels, (list, tuple)) and len(labels) > ao_index:
                label = labels[ao_index]
                if isinstance(label, (list, tuple)) and len(label) >= 4:
                    atom_index = label[0]
                    atom_symbol = str(label[1])
                    ao_label = ' '.join(str(part) for part in label[2:] if str(part))
                else:
                    ao_label = str(label)
            atom_key = '{0}:{1}'.format(atom_index, atom_symbol) if atom_index is not None else 'unknown'
            atom_weights[atom_key] = atom_weights.get(atom_key, 0.0) + float(weight)
            ao_rows.append({
                'ao_index': int(ao_index),
                'atom': atom_key,
                'label': ao_label,
                'weight': float(weight),
            })
        ao_rows.sort(key=lambda item: item['weight'], reverse=True)
        contribution['ao_contributions'] = ao_rows[:limit]
        atom_rows = [
            {'atom': atom, 'weight': weight}
            for atom, weight in atom_weights.items()
        ]
        atom_rows.sort(key=lambda item: item['weight'], reverse=True)
        contribution['atom_contributions'] = atom_rows[:limit]
        contribution['fragment_contributions'] = [
            {'fragment': 'atom:{0}'.format(item['atom']), 'weight': item['weight']}
            for item in atom_rows[:limit]
        ]
        return contribution
    except Exception:  # pragma: no cover - diagnostic enrichment only
        return contribution


def _nelecas_total(nelecas: Any) -> Optional[int]:
    if nelecas is None:
        return None
    if isinstance(nelecas, tuple):
        return sum(int(item) for item in nelecas)
    if isinstance(nelecas, list):
        return sum(int(item) for item in nelecas)
    try:
        return int(nelecas)
    except (TypeError, ValueError):
        return None


def _target_solver_options(
    active_space: ActiveSpaceSpec,
    *,
    ncas: Optional[int],
    nelecas: Any,
    mol: Any,
) -> Dict[str, Any]:
    options = copy.deepcopy(active_space.target_solver_options or {})
    solver = str(active_space.target_solver or '').strip().lower().replace('-', '_')
    if solver not in ('block2', 'block2_dmrg', 'dmrg') or options.get('preset'):
        return options
    if not ncas:
        options['preset'] = 'balanced'
        return options
    if isinstance(nelecas, (tuple, list)) and len(nelecas) == 2:
        nalpha, nbeta = (int(nelecas[0]), int(nelecas[1]))
    else:
        total_electrons = _nelecas_total(nelecas)
        spin = int(getattr(mol, 'spin', 0) or 0) if mol is not None else 0
        if total_electrons is None or (total_electrons + spin) % 2:
            options['preset'] = 'balanced'
            return options
        nalpha = (int(total_electrons) + spin) // 2
        nbeta = int(total_electrons) - nalpha
    if min(nalpha, nbeta) < 0 or max(nalpha, nbeta) > int(ncas):
        options['preset'] = 'balanced'
        return options
    sector_dimension = math.comb(int(ncas), nalpha) * math.comb(int(ncas), nbeta)
    options['preset'] = 'screening' if sector_dimension <= 4096 else 'balanced'
    return options


def _manual_indices(active_space: ActiveSpaceSpec) -> List[int]:
    if isinstance(active_space.orbital_indices, dict):
        selected: List[int] = []
        for spin_label in ('alpha', 'beta'):
            for index in active_space.orbital_indices.get(spin_label, []):
                if index not in selected:
                    selected.append(index)
        return selected
    if active_space.orbital_indices:
        return list(active_space.orbital_indices)
    if active_space.ncas:
        return list(range(active_space.ncas))
    return []


def _uno_candidate_indices(table: List[Dict[str, Any]], occupation_window: Tuple[float, float]) -> Tuple[List[int], Dict[int, List[str]]]:
    lower, upper = occupation_window
    indices: List[int] = []
    reasons: Dict[int, List[str]] = {}
    for row in table:
        occupation = _as_float(row.get('occupation'))
        if occupation is None or not (lower <= occupation <= upper):
            continue
        index = int(row['index'])
        indices.append(index)
        reasons[index] = [
            'uno_spin_summed_mean_field_occupation_window[{0:g},{1:g}] (n={2:.6g})'.format(
                lower,
                upper,
                occupation,
            )
        ]
    return sorted(indices), reasons


def _estimate_nelecas_from_indices(indices: Sequence[int], table_by_index: Dict[int, Dict[str, Any]]) -> Optional[int]:
    if not indices:
        return None
    return int(round(sum(
        float(table_by_index.get(int(index), {}).get('occupation') or 0.0)
        for index in indices
    )))


def _degenerate_manifold_completion(
    indices: Sequence[int],
    table_by_index: Dict[int, Dict[str, Any]],
    *,
    tolerance: float = DEGENERATE_MANIFOLD_TOLERANCE,
) -> Dict[int, List[str]]:
    """Complete near-degenerate occupied or virtual canonical manifolds."""

    additions: Dict[int, List[str]] = {}
    anchors = [table_by_index.get(int(index), {}) for index in indices]
    for anchor in anchors:
        anchor_index = anchor.get('index')
        anchor_energy = _as_float(anchor.get('energy'))
        anchor_occupation = _as_float(anchor.get('occupation'))
        if anchor_index is None or anchor_energy is None or anchor_occupation is None:
            continue
        anchor_occupied = anchor_occupation > 1.0e-6
        for candidate_index, candidate in table_by_index.items():
            if candidate_index in indices:
                continue
            candidate_energy = _as_float(candidate.get('energy'))
            candidate_occupation = _as_float(candidate.get('occupation'))
            if candidate_energy is None or candidate_occupation is None:
                continue
            if (candidate_occupation > 1.0e-6) != anchor_occupied:
                continue
            if abs(candidate_energy - anchor_energy) <= tolerance:
                additions.setdefault(int(candidate_index), []).append(
                    'degenerate_manifold_partner_of={0} (|dE|={1:.6g})'.format(
                        anchor_index,
                        abs(candidate_energy - anchor_energy),
                    )
                )
    return additions


def _candidate_payload(
    method: str,
    indices: Sequence[int],
    table_by_index: Dict[int, Dict[str, Any]],
    *,
    status: str = 'available',
    reasons: Optional[Dict[int, List[str]]] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    normalized = sorted({int(index) for index in indices})
    if not normalized and status == 'available':
        status = 'empty'
    payload = {
        'method': method,
        'status': status,
        'orbital_index_basis': 'canonical_mo',
        'orbital_indices': normalized,
        'ncas': len(normalized) if normalized else None,
        'estimated_nelecas': _estimate_nelecas_from_indices(normalized, table_by_index),
        'selection_reasons': {
            str(index): list((reasons or {}).get(index, []))
            for index in normalized
        },
    }
    if note:
        payload['note'] = note
    return payload


def _evidence_expanded_candidate(
    chemical_candidate: Dict[str, Any],
    selection_evidence: Dict[str, Any],
    table_by_index: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """Expand a complete chemical baseline with only high-confidence evidence.

    Expansion closes local near-degenerate manifolds and never uses a fixed
    orbital-count cap. The broad threshold union remains a separate candidate.
    """

    if chemical_candidate.get('status') not in ('available', 'mapping_ambiguous'):
        return {
            'method': 'evidence_expanded',
            'role': 'expanded',
            'status': 'unavailable',
            'orbital_index_basis': 'canonical_mo',
            'orbital_indices': [],
            'ncas': None,
            'estimated_nelecas': None,
            'selection_reasons': {},
            'parent_candidate': 'chemical_valence',
            'note': 'A chemically resolved baseline is required before evidence-based expansion.',
        }

    baseline = {int(index) for index in chemical_candidate.get('orbital_indices') or []}
    expanded = set(baseline)
    reasons: Dict[int, List[str]] = {
        int(index): list((chemical_candidate.get('selection_reasons') or {}).get(str(index), []))
        for index in baseline
    }
    for signal in selection_evidence.get('signals') or []:
        index = int(signal['orbital_index'])
        expanded.add(index)
        reasons.setdefault(index, []).append(str(signal.get('reason') or signal.get('source') or 'correlation evidence'))
    for index, completion_reasons in _degenerate_manifold_completion(sorted(expanded), table_by_index).items():
        expanded.add(index)
        reasons.setdefault(index, []).extend(completion_reasons)

    payload = _candidate_payload(
        'evidence_expanded',
        sorted(expanded),
        table_by_index,
        status='available' if expanded != baseline else 'not_needed',
        reasons=reasons,
        note=(
            'Chemical-valence baseline plus complete near-degenerate manifolds required by '
            'high-confidence natural-occupation or T2 evidence.'
        ),
    )
    payload.update({
        'role': 'expanded',
        'parent_candidate': 'chemical_valence',
        'added_orbital_indices': sorted(expanded - baseline),
    })
    return payload


def _resolve_avas_targets(mol: Any, targets: Sequence[Any]) -> List[str]:
    """Expand atom:/fragment: selectors into PySCF AO-label patterns."""
    resolved: List[str] = []
    ao_labels = mol.ao_labels() if mol is not None and hasattr(mol, 'ao_labels') else []
    for raw_target in targets:
        target = str(raw_target or '').strip()
        if not target:
            continue
        prefix, separator, selector = target.partition(':')
        if separator and prefix.strip().lower() in ('atom', 'fragment'):
            try:
                atom_indices = {
                    int(item.strip())
                    for item in selector.replace(';', ',').split(',')
                    if item.strip()
                }
            except ValueError as exc:
                raise ValueError('Invalid AVAS {0} selector: {1}'.format(prefix.strip(), target)) from exc
            matched = []
            for label in ao_labels:
                compact = ' '.join(str(label).split())
                try:
                    atom_index = int(compact.split()[0])
                except (IndexError, ValueError):
                    continue
                if atom_index in atom_indices:
                    matched.append(compact)
            if not matched:
                raise ValueError('AVAS {0} selector did not match any AO labels: {1}'.format(prefix.strip(), target))
            resolved.extend(matched)
        else:
            resolved.append(target)
    return list(dict.fromkeys(resolved))


def _unique_projection_mapping(projection: Any) -> List[Tuple[int, int, float]]:
    """Map projected active orbitals to distinct canonical orbitals."""

    import numpy as np  # pylint: disable=import-outside-toplevel

    weights = np.abs(np.asarray(projection, dtype=float)) ** 2
    if weights.ndim != 2 or not weights.size or weights.shape[0] < weights.shape[1]:
        return []
    from scipy.optimize import linear_sum_assignment  # PySCF requires SciPy.

    canonical_indices, active_indices = linear_sum_assignment(-weights)
    assigned = {
        int(active_index): (int(canonical_index), float(weights[canonical_index, active_index]))
        for canonical_index, active_index in zip(canonical_indices, active_indices)
    }
    return [
        (active_index, assigned[active_index][0], assigned[active_index][1])
        for active_index in range(weights.shape[1])
        if active_index in assigned
    ]


def _compact_chemical_avas_projection(
    avas_mo: Any,
    projection: Any,
    *,
    ncas: int,
    nelecas: Any,
    active_column_start: int,
    target_summary: Optional[Dict[str, Any]],
    spin: int,
) -> Tuple[Any, int, Any, int, Dict[str, Any]]:
    """Remove weak occupied/virtual AVAS split partners above the target AO rank."""

    import numpy as np  # pylint: disable=import-outside-toplevel

    metadata: Dict[str, Any] = {
        'status': 'not_needed',
        'raw_ncas': int(ncas),
        'raw_nelecas': copy.deepcopy(nelecas),
    }
    if not isinstance(target_summary, dict):
        return avas_mo, ncas, nelecas, active_column_start, metadata
    target_ncas = target_summary.get('primary_orbital_count')
    target_nelecas = target_summary.get('primary_electron_count')
    raw_nelecas = _nelecas_total(nelecas)
    try:
        target_ncas = int(target_ncas)
        target_nelecas = int(target_nelecas)
        raw_nelecas = int(raw_nelecas)
        spin = int(spin or 0)
    except (TypeError, ValueError):
        metadata['status'] = 'not_available'
        return avas_mo, ncas, nelecas, active_column_start, metadata
    reduction = int(ncas) - target_ncas
    electron_reduction = raw_nelecas - target_nelecas
    if reduction <= 0:
        return avas_mo, ncas, nelecas, active_column_start, metadata
    if (
        spin != 0
        or electron_reduction < 0
        or electron_reduction % 2
        or (target_nelecas + spin) % 2
        or target_nelecas > 2 * target_ncas
    ):
        metadata['status'] = 'not_applicable'
        metadata['reason'] = 'The target sector cannot be obtained by closed-shell AVAS split-partner removal.'
        return avas_mo, ncas, nelecas, active_column_start, metadata

    occupied_removals = electron_reduction // 2
    virtual_removals = reduction - occupied_removals
    occupied_active = raw_nelecas // 2
    if (
        occupied_removals < 0
        or virtual_removals < 0
        or occupied_removals > occupied_active
        or virtual_removals > int(ncas) - occupied_active
    ):
        metadata['status'] = 'not_applicable'
        metadata['reason'] = 'The requested compact target is incompatible with the AVAS occupied/virtual partition.'
        return avas_mo, ncas, nelecas, active_column_start, metadata

    weights = np.abs(np.asarray(projection, dtype=float)) ** 2
    active_scores = np.max(weights, axis=0) if weights.size else np.zeros(int(ncas))
    occupied_pool = list(range(occupied_active))
    virtual_pool = list(range(occupied_active, int(ncas)))
    removed_occupied = sorted(occupied_pool, key=lambda index: (active_scores[index], index))[:occupied_removals]
    removed_virtual = sorted(virtual_pool, key=lambda index: (active_scores[index], index))[:virtual_removals]
    removed = set(removed_occupied + removed_virtual)
    retained = [index for index in range(int(ncas)) if index not in removed]
    if len(retained) != target_ncas:
        metadata['status'] = 'failed'
        metadata['reason'] = 'AVAS compaction did not produce the requested target dimension.'
        return avas_mo, ncas, nelecas, active_column_start, metadata

    matrix = np.asarray(avas_mo, dtype=float)
    active_columns = [active_column_start + index for index in range(int(ncas))]
    prefix = list(range(active_column_start))
    suffix = list(range(active_column_start + int(ncas), matrix.shape[1]))
    column_order = (
        prefix
        + [active_columns[index] for index in removed_occupied]
        + [active_columns[index] for index in retained]
        + [active_columns[index] for index in removed_virtual]
        + suffix
    )
    compact_mo = matrix[:, column_order]
    compact_start = active_column_start + len(removed_occupied)
    metadata.update({
        'status': 'applied',
        'policy': 'target_ao_rank_with_weak_split_partner_removal',
        'target_ncas': target_ncas,
        'target_nelecas': target_nelecas,
        'removed_occupied_active_indices': removed_occupied,
        'removed_virtual_active_indices': removed_virtual,
        'retained_raw_active_indices': retained,
    })
    return compact_mo, target_ncas, target_nelecas, compact_start, metadata


def _spin_adapted_projection_reference(mf: Any, mol: Any) -> Tuple[Any, Dict[str, Any]]:
    """Return a shared-orbital reference suitable for AVAS projection.

    AVAS expects one spatial MO coefficient matrix.  For a UHF probe, build
    spin-summed UNO orbitals from the AO density and attach them to a converted
    RHF/ROHF object without rerunning the mean-field calculation.
    """
    metadata: Dict[str, Any] = {
        'status': 'unavailable',
        'source': None,
    }
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel

        coefficient = np.asarray(getattr(mf, 'mo_coeff', None), dtype=float)
        if coefficient.ndim == 2:
            metadata.update({'status': 'available', 'source': 'input_spin_adapted_reference'})
            return mf, metadata
        if coefficient.ndim != 3 or coefficient.shape[0] < 2:
            metadata['reason'] = 'mean-field orbitals do not define a spatial or alpha/beta reference'
            return None, metadata
        density_channels = mf.make_rdm1()
        density_array = np.asarray(density_channels, dtype=float)
        if density_array.ndim != 3 or density_array.shape[0] < 2:
            metadata['reason'] = 'UHF AO density does not contain alpha and beta channels'
            return None, metadata
        density = density_array[0] + density_array[1]
        occupations, uno_coeff, overlap = _ao_natural_orbitals(density, mf)

        from pyscf.scf import addons  # pylint: disable=import-outside-toplevel

        projection_mf = addons.convert_to_rhf(mf)
        projection_mf.mo_coeff = uno_coeff
        nalpha, nbeta = getattr(mol, 'nelec', (0, 0))
        doubly_occupied = min(int(nalpha), int(nbeta))
        singly_occupied = abs(int(nalpha) - int(nbeta))
        projection_mf.mo_occ = np.asarray(
            [2.0] * doubly_occupied
            + [1.0] * singly_occupied
            + [0.0] * max(0, uno_coeff.shape[1] - doubly_occupied - singly_occupied)
        )
        # Project the existing canonical eigenvalues; no extra J/K build is
        # needed merely to label the UNO reference used by candidate generation.
        energies = np.asarray(mf.mo_energy, dtype=float)
        weights = [np.abs(channel.T @ overlap @ uno_coeff) ** 2 for channel in coefficient]
        projection_mf.mo_energy = sum(
            np.sum(energy[:, None] * weight, axis=0)
            for energy, weight in zip(energies, weights)
        ) / len(weights)
        projection_mf.converged = False
        metadata.update({
            'status': 'available',
            'source': 'uhf_spin_summed_uno_projection_reference',
            'natural_occupations': [float(value) for value in occupations],
        })
        return projection_mf, metadata
    except Exception as exc:  # pragma: no cover - depends on PySCF reference details
        metadata['reason'] = str(exc)
        return None, metadata


def _uno_candidate(
    mf: Any,
    mol: Any,
    active_space: ActiveSpaceSpec,
    table: List[Dict[str, Any]],
    projection_mf: Any,
    projection_reference: Dict[str, Any],
) -> Dict[str, Any]:
    """Select genuine UHF natural orbitals and retain their executable basis."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    table_by_index = {int(row['index']): row for row in table}
    unrestricted = np.asarray(mf.mo_coeff).ndim == 3
    if not unrestricted:
        indices, reasons = _uno_candidate_indices(table, active_space.occupation_window)
        return _candidate_payload(
            'uno', indices, table_by_index, reasons=reasons,
            note='The restricted reference already has shared spatial occupation eigenvectors.',
        )
    payload = _candidate_payload('uno', [], table_by_index, status='unavailable')
    payload.update({
        'orbital_index_basis': 'uno_orbital_projection',
        'projection_reference': copy.deepcopy(projection_reference),
        'note': (
            'UNO selection diagonalizes the spin-summed UHF AO 1-RDM in the overlap metric. '
            'Canonical indices are display mappings only; initial_mo_coeff preserves the '
            'core-active-virtual orbital basis for execution after approval.'
        ),
    })
    if projection_mf is None:
        return payload
    occupations = projection_reference.get('natural_occupations') or []
    uno_table = [{'index': index, 'occupation': value} for index, value in enumerate(occupations)]
    indices, reasons = _uno_candidate_indices(uno_table, active_space.occupation_window)
    payload['uno_orbital_indices'] = indices
    payload['uno_occupations'] = [occupations[index] for index in indices]
    if not indices:
        payload['status'] = 'empty'
        return payload
    selected = set(indices)
    upper = active_space.occupation_window[1]
    core = [index for index, value in enumerate(occupations) if index not in selected and value > upper]
    virtual = [index for index in range(len(occupations)) if index not in selected and index not in core]
    nelecas = int(mol.nelectron) - 2 * len(core)
    spin = int(mol.spin)
    if (nelecas + spin) % 2 or min(nelecas + spin, nelecas - spin) < 0 or max(nelecas + spin, nelecas - spin) > 2 * len(indices):
        payload.update({'status': 'invalid_electron_sector', 'reason': 'The occupation window does not leave a valid active electron/spin sector.'})
        return payload
    coeff = np.asarray(projection_mf.mo_coeff)
    overlap = mf.get_ovlp()
    # Alpha canonical labels are stable display references even if beta MOs
    # are reordered. Their projection never replaces the actual UNO columns.
    projection = np.asarray(mf.mo_coeff)[0].T @ overlap @ coeff[:, indices]
    mapping = _unique_projection_mapping(projection)
    mapped_reasons = {canonical: reasons[indices[active]] for active, canonical, _weight in mapping}
    ordered_coeff = coeff[:, core + indices + virtual]
    payload.update({
        'status': 'available',
        'orbital_indices': sorted(mapped_reasons),
        'ncas': len(indices),
        'estimated_nelecas': nelecas,
        'active_column_start': len(core),
        'selection_reasons': {str(index): values for index, values in mapped_reasons.items()},
        'canonical_mapping_strategy': 'alpha_global_unique_maximum_overlap_display_only',
        'projected_active_orbitals': [
            {
                'uno_orbital_index': indices[active],
                'occupation': occupations[indices[active]],
                'dominant_canonical_mo': canonical,
                'dominant_canonical_weight': weight,
                **_ao_atom_fragment_contributions(projection_mf, mol, indices[active]),
            }
            for active, canonical, weight in mapping
        ],
        'initial_mo_coeff': ordered_coeff.tolist(),
        'initial_mo_coeff_shape': list(ordered_coeff.shape),
    })
    return payload


def _avas_candidate(
    mf: Any,
    mol: Any,
    active_space: ActiveSpaceSpec,
    table_by_index: Dict[int, Dict[str, Any]],
    *,
    targets_override: Optional[Sequence[Any]] = None,
    method_name: str = 'avas',
    projection_mf: Any = None,
    projection_reference: Optional[Dict[str, Any]] = None,
    compact_target_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run PySCF AVAS and preserve its orbital guess for a later CAS execution."""
    targets = list(targets_override if targets_override is not None else active_space.avas_targets or [])
    payload: Dict[str, Any] = {
        'method': method_name,
        'status': 'not_requested' if not targets else 'unavailable',
        'orbital_index_basis': 'avas_orbital_projection',
        'orbital_indices': [],
        'ncas': None,
        'estimated_nelecas': None,
        'selection_reasons': {},
        'targets': targets,
        'projection_reference': copy.deepcopy(projection_reference or {}),
    }
    if not targets:
        payload['note'] = 'Provide AVAS target AO labels (for example "Fe 3d") or atom:/fragment: selectors to generate this candidate.'
        return payload
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        from pyscf.mcscf import avas  # pylint: disable=import-outside-toplevel

        canonical_coeff = np.asarray(getattr(mf, 'mo_coeff', None), dtype=float)
        canonical_mo = canonical_coeff[0] if canonical_coeff.ndim == 3 else canonical_coeff
        avas_reference = projection_mf if projection_mf is not None else mf
        avas_mo_reference = np.asarray(getattr(avas_reference, 'mo_coeff', None), dtype=float)
        if canonical_mo.ndim != 2 or avas_mo_reference.ndim != 2:
            payload.update({'status': 'unsupported_reference', 'note': 'AVAS needs a shared spatial-orbital projection reference.'})
            return payload
        resolved_targets = _resolve_avas_targets(mol, targets)
        ncas, nelecas, avas_mo = avas.avas(
            avas_reference,
            resolved_targets,
            threshold=float(active_space.avas_threshold),
            minao=str(active_space.avas_minimal_basis or 'minao'),
            with_iao=bool(active_space.avas_with_iao),
            openshell_option=int(active_space.avas_openshell_option),
            ncore=int(active_space.avas_ncore),
        )
        avas_mo = np.asarray(avas_mo, dtype=float)
        ncas = int(ncas)
        nelecas_value = int(nelecas) if not isinstance(nelecas, (list, tuple)) else [int(item) for item in nelecas]
        ncore = int(active_space.avas_ncore)
        overlap = avas_reference.get_ovlp() if hasattr(avas_reference, 'get_ovlp') else mol.intor_symmetric('int1e_ovlp')
        active_electron_count = _nelecas_total(nelecas_value)
        total_electron_count = int(getattr(mol, 'nelectron', 0) or 0)
        active_column_start = (
            max(0, (total_electron_count - active_electron_count) // 2)
            if active_electron_count is not None
            else ncore
        )
        active_columns = avas_mo[:, active_column_start:active_column_start + ncas]
        projection = canonical_mo.T.dot(overlap).dot(active_columns)
        avas_mo, ncas, nelecas_value, active_column_start, projection_compaction = (
            _compact_chemical_avas_projection(
                avas_mo,
                projection,
                ncas=ncas,
                nelecas=nelecas_value,
                active_column_start=active_column_start,
                target_summary=compact_target_summary,
                spin=int(getattr(mol, 'spin', 0) or 0),
            )
        )
        active_columns = avas_mo[:, active_column_start:active_column_start + ncas]
        projection = canonical_mo.T.dot(overlap).dot(active_columns)
        mapped_indices: List[int] = []
        selection_reasons: Dict[int, List[str]] = {}
        projected_orbitals = []
        unique_mapping = _unique_projection_mapping(projection)
        for avas_index, canonical_index, overlap_weight in unique_mapping:
            projected_orbitals.append({
                'avas_active_index': avas_index,
                'dominant_canonical_mo': canonical_index,
                'dominant_canonical_weight': overlap_weight,
            })
            mapped_indices.append(canonical_index)
            selection_reasons.setdefault(canonical_index, []).append(
                '{0} target projection active_orbital={1}, unique_weight={2:.6f}'.format(
                    method_name,
                    avas_index,
                    overlap_weight,
                )
            )
        unique_indices = sorted(set(mapped_indices))
        mapping_status = 'available' if len(unique_indices) == ncas else 'mapping_ambiguous'
        payload.update({
            'status': mapping_status,
            'orbital_indices': unique_indices,
            'ncas': ncas,
            'estimated_nelecas': nelecas_value,
            'selection_reasons': {str(index): reasons for index, reasons in selection_reasons.items()},
            'resolved_targets': resolved_targets,
            'avas_parameters': {
                'threshold': float(active_space.avas_threshold),
                'minimal_basis': str(active_space.avas_minimal_basis or 'minao'),
                'with_iao': bool(active_space.avas_with_iao),
                'openshell_option': int(active_space.avas_openshell_option),
                'ncore': ncore,
                'active_column_start': active_column_start,
            },
            'projection_compaction': projection_compaction,
            'canonical_mapping_strategy': 'global_unique_maximum_overlap',
            'projected_active_orbitals': projected_orbitals,
            'initial_mo_coeff': avas_mo.tolist(),
            'initial_mo_coeff_shape': list(avas_mo.shape),
            'note': 'PySCF AVAS generated this candidate. Its orbital matrix is retained as the CASSCF/CASCI initial guess after approval.',
        })
        return payload
    except Exception as exc:  # pragma: no cover - PySCF/basis behavior depends on user input
        payload.update({'status': 'failed', 'note': 'AVAS projection failed: {0}'.format(exc)})
        return payload


def _chemical_valence_candidate(
    mf: Any,
    mol: Any,
    active_space: ActiveSpaceSpec,
    table_by_index: Dict[int, Dict[str, Any]],
    *,
    projection_mf: Any = None,
    projection_reference: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    target_summary = chemical_valence_targets(mol)
    targets = target_summary.get('primary_targets') or []
    if projection_reference is None:
        projection_mf, projection_reference = _spin_adapted_projection_reference(mf, mol)
    projection_reference = projection_reference or {}
    if target_summary.get('status') != 'available' or projection_mf is None:
        return {
            'method': 'chemical_valence',
            'status': 'unavailable',
            'orbital_index_basis': 'avas_orbital_projection',
            'orbital_indices': [],
            'ncas': None,
            'estimated_nelecas': None,
            'selection_reasons': {},
            'targets': list(targets),
            'target_summary': target_summary,
            'projection_reference': projection_reference,
            'note': projection_reference.get('reason') or 'No chemically resolved valence AO targets are available.',
        }
    candidate = _avas_candidate(
        mf,
        mol,
        active_space,
        table_by_index,
        targets_override=targets,
        method_name='chemical_valence',
        projection_mf=projection_mf,
        projection_reference=projection_reference,
        compact_target_summary=target_summary,
    )
    candidate['target_summary'] = target_summary
    return candidate


def propose_active_space(
    mf: Any,
    active_space: ActiveSpaceSpec,
    post_hf: Any = None,
    mol: Any = None,
    localization_method: Optional[str] = None,
    localization_scope: str = 'analysis',
    orbital_ordering: str = 'canonical',
    orbital_order: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    mol = mol or getattr(mf, 'mol', None)
    table = _canonical_orbital_table(mf)
    table_by_index = {int(row['index']): row for row in table}
    selected: List[int] = []
    selected_set: Set[int] = set()
    selection_reasons: Dict[int, List[str]] = {}
    method = active_space.selection_method or 'manual'
    natural_summary = _natural_orbital_summary(post_hf, mf=mf)
    natural_occupation_quality = _natural_occupation_quality(natural_summary)
    t2_evidence = _t2_orbital_importance(post_hf, mf)
    t2_importance = t2_evidence['importance']
    t2_threshold = T2_SUPPLEMENT_THRESHOLD
    explicit_manual_indices = _manual_indices(active_space) if active_space.orbital_indices else []
    if method == 'manual' and not explicit_manual_indices and active_space.ncas:
        explicit_manual_indices = list(range(int(active_space.ncas)))
    projection_mf, projection_reference = _spin_adapted_projection_reference(mf, mol)
    uno_candidate = _uno_candidate(mf, mol, active_space, table, projection_mf, projection_reference)
    uno_candidate['role'] = 'diagnostic'
    uno_indices = uno_candidate.get('orbital_indices') or []
    uno_reasons = {int(index): values for index, values in (uno_candidate.get('selection_reasons') or {}).items()}
    avas_candidate = _avas_candidate(
        mf,
        mol,
        active_space,
        table_by_index,
        projection_mf=projection_mf,
        projection_reference=projection_reference,
    )
    chemical_valence_candidate = _chemical_valence_candidate(
        mf,
        mol,
        active_space,
        table_by_index,
        projection_mf=projection_mf,
        projection_reference=projection_reference,
    )

    def add_selected(index: Any, reason: str) -> None:
        try:
            normalized = int(index)
        except (TypeError, ValueError):
            return
        if normalized not in table_by_index and table:
            return
        if normalized not in selected_set:
            selected_set.add(normalized)
            selected.append(normalized)
        selection_reasons.setdefault(normalized, [])
        if reason not in selection_reasons[normalized]:
            selection_reasons[normalized].append(reason)

    recovery_candidate_method: Optional[str] = None
    recovery_reason: Optional[str] = None
    t2_supplement = {
        'status': 'not_applicable',
        'threshold': t2_threshold,
        'energy_padding': T2_SUPPLEMENT_ENERGY_PADDING,
        'accepted_indices': [],
        'excluded': [],
    }

    if method == 'manual':
        for index in explicit_manual_indices:
            add_selected(index, 'manual_user_selection' if active_space.orbital_indices else 'manual_first_ncas')
    elif method == 'occupation_window':
        if natural_occupation_quality.get('status') == 'out_of_bounds':
            recovery_reason = (
                'MP2 natural occupations are outside the physical [0, 2] range; '
                'they are retained as solver-stress evidence but are not used to define an executable CAS.'
            )
            if chemical_valence_candidate.get('status') in ('available', 'mapping_ambiguous'):
                recovery_candidate_method = 'chemical_valence'
                for index in chemical_valence_candidate.get('orbital_indices') or []:
                    reasons = (chemical_valence_candidate.get('selection_reasons') or {}).get(str(index), [])
                    for reason in reasons or ['chemical_valence_recovery']:
                        add_selected(index, reason)
        else:
            lower, upper = active_space.occupation_window
            natural_orbitals = natural_summary.get('orbitals') if natural_summary.get('status') == 'available' else []
            if isinstance(natural_orbitals, list) and natural_orbitals:
                for natural_orbital in natural_orbitals:
                    occupation = _as_float(natural_orbital.get('occupation'))
                    canonical_index = natural_orbital.get('dominant_canonical_mo')
                    if occupation is not None and lower <= occupation <= upper:
                        add_selected(
                            canonical_index,
                            'mp2_natural_occupation_window[{0:g},{1:g}] via NO {2} (n={3:.6g})'.format(
                                lower,
                                upper,
                                natural_orbital.get('natural_orbital_index'),
                                occupation,
                            ),
                        )
            else:
                for index in uno_indices:
                    for reason in uno_reasons.get(index, []):
                        add_selected(index, reason)
    elif method == 'energy_window':
        energies = [row.get('energy') for row in table if row.get('energy') is not None]
        if energies:
            homo, lumo = _extract_homo_lumo(getattr(mf, 'mo_energy', None), getattr(mf, 'mo_occ', None))
            center = 0.0
            if homo is not None and lumo is not None:
                center = 0.5 * (homo + lumo)
            window = active_space.energy_window if active_space.energy_window is not None else 0.5
            for row in table:
                if row.get('energy') is not None and abs(float(row['energy']) - center) <= window:
                    add_selected(row['index'], 'energy_window(center={0:.6g}, width={1:.6g})'.format(center, window))
    elif method == 'avas' and avas_candidate.get('status') in ('available', 'mapping_ambiguous'):
        for index in avas_candidate.get('orbital_indices') or []:
            for reason in (avas_candidate.get('selection_reasons') or {}).get(str(index), []):
                add_selected(index, reason)

    targeted_cas_request = (
        active_space.target_method in ('casci', 'casscf')
        or active_space.target_solver == 'block2_dmrg'
    )
    if method == 'occupation_window' and not selected and targeted_cas_request:
        recovery_reason = recovery_reason or (
            'The requested occupation window selected no physical correlated orbitals for the explicit CAS request.'
        )
        if chemical_valence_candidate.get('status') in ('available', 'mapping_ambiguous'):
            recovery_candidate_method = 'chemical_valence'
            for index in chemical_valence_candidate.get('orbital_indices') or []:
                reasons = (chemical_valence_candidate.get('selection_reasons') or {}).get(str(index), [])
                for reason in reasons or ['chemical_valence_recovery']:
                    add_selected(index, reason)

    if method not in ('manual', 'avas'):
        t2_supplement = _screen_t2_supplements(
            t2_importance,
            selected,
            table_by_index,
            natural_occupation_quality,
        )
        for index in t2_supplement['accepted_indices']:
            importance = float(t2_importance[index])
            add_selected(index, 't2_supplement(max|t2|={0:.6g} >= {1:g})'.format(importance, t2_threshold))

    if method not in ('manual', 'avas') and recovery_candidate_method is None and uno_indices:
        for index in uno_indices:
            for reason in uno_reasons.get(index, []):
                add_selected(index, 'merged_uno_candidate: {0}'.format(reason))

    if method not in ('manual', 'avas') and recovery_candidate_method is None and selected:
        for index, reasons in _degenerate_manifold_completion(selected, table_by_index).items():
            for reason in reasons:
                add_selected(index, reason)

    selected.sort()
    merged_indices = list(selected)
    merged_reasons = copy.deepcopy(selection_reasons)
    selection_evidence = build_active_space_selection_evidence(
        natural_summary,
        natural_occupation_quality,
        t2_importance,
        t2_supplement,
    )
    evidence_expanded_candidate = _evidence_expanded_candidate(
        chemical_valence_candidate,
        selection_evidence,
        table_by_index,
    )

    manual_candidate = _candidate_payload(
        'manual',
        explicit_manual_indices,
        table_by_index,
        status='available' if explicit_manual_indices else 'not_provided',
        reasons={index: ['manual_user_selection'] for index in explicit_manual_indices},
        note='Manual candidate reflects user-supplied orbital indices or ncas in manual mode.',
    )
    manual_candidate['role'] = 'manual'
    if active_space.ncas is not None:
        manual_candidate['ncas'] = int(active_space.ncas)
    if active_space.nelecas is not None:
        manual_candidate['estimated_nelecas'] = copy.deepcopy(active_space.nelecas)

    chemical_candidate = copy.deepcopy(chemical_valence_candidate)
    chemical_candidate['role'] = 'baseline'

    frontier_candidate = _candidate_payload(
        'frontier_fallback',
        [],
        table_by_index,
        status='diagnostic_only',
        reasons={},
        note=(
            'Fixed-size frontier fallbacks are not executable candidates. Empty or nonphysical '
            'perturbative selections require a chemically resolved candidate or manual review.'
        ),
    )
    frontier_candidate['role'] = 'diagnostic'

    avas_review_candidate = copy.deepcopy(avas_candidate)
    avas_review_candidate['role'] = 'baseline'

    merged_candidate = _candidate_payload(
        'merged',
        merged_indices,
        table_by_index,
        status='available' if merged_indices else 'empty',
        reasons=merged_reasons,
        note=(
            'Broad diagnostic union of all accepted occupation, UNO, T2, and degeneracy evidence. '
            'Its canonical display union does not reproduce a projected UNO subspace. '
            'It is retained for review and is not preferred over a chemically complete smaller candidate.'
        ),
    )
    merged_candidate['role'] = 'broad'

    candidate_active_spaces = evaluate_active_space_candidates(
        [
            manual_candidate,
            uno_candidate,
            chemical_candidate,
            evidence_expanded_candidate,
            frontier_candidate,
            avas_review_candidate,
            merged_candidate,
        ],
        selection_evidence,
        chemical_baseline_indices=chemical_candidate.get('orbital_indices') or [],
        spin=int(getattr(mol, 'spin', 0) or 0),
        target_solver=active_space.target_solver,
    )
    candidate_active_spaces, candidate_decision = recommend_active_space_candidate(
        candidate_active_spaces,
        selection_method=(
            'uno' if method == 'occupation_window'
            and natural_summary.get('status') != 'available'
            and uno_candidate.get('initial_mo_coeff') is not None
            else method
        ),
        recovery_candidate_method=recovery_candidate_method,
    )
    selected_candidate_method = str(candidate_decision.get('selected_candidate_method') or 'unresolved')
    selected_candidate = next(
        (
            candidate
            for candidate in candidate_active_spaces
            if candidate.get('method') == selected_candidate_method
        ),
        {},
    )
    selected = sorted({int(index) for index in selected_candidate.get('orbital_indices') or []})
    selected_set = set(selected)
    selection_reasons = {
        int(index): list(reasons or [])
        for index, reasons in (selected_candidate.get('selection_reasons') or {}).items()
    }
    selected_projection_candidate = (
        selected_candidate
        if selected_candidate.get('initial_mo_coeff') is not None
        else None
    )
    ncas = selected_candidate.get('ncas') if selected_candidate else None
    estimated_nelecas = None
    natural_occupations_by_index: Dict[int, List[Dict[str, Any]]] = {}
    if natural_summary.get('status') == 'available':
        for natural_orbital in natural_summary.get('orbitals') or []:
            canonical_index = natural_orbital.get('dominant_canonical_mo')
            if canonical_index is None:
                continue
            natural_occupations_by_index.setdefault(int(canonical_index), []).append(natural_orbital)
    if selected:
        estimated_nelecas = _estimate_nelecas_from_indices(selected, table_by_index)
    if isinstance(selected_projection_candidate, dict) and selected_projection_candidate.get('estimated_nelecas') is not None:
        estimated_nelecas = selected_projection_candidate.get('estimated_nelecas')

    nelecas = copy.deepcopy(active_space.nelecas) if selected_candidate_method == 'manual' else None
    if nelecas is None and selected_candidate.get('estimated_nelecas') is not None:
        nelecas = copy.deepcopy(selected_candidate.get('estimated_nelecas'))
    if nelecas is None and estimated_nelecas is not None:
        nelecas = estimated_nelecas

    consistency_messages = []
    consistent = True
    if ncas is not None and selected and int(ncas) != len(selected):
        consistent = False
        consistency_messages.append('ncas={0} but {1} canonical orbital index(es) are selected'.format(ncas, len(selected)))
    total_nelecas = _nelecas_total(nelecas)
    if ncas is not None and total_nelecas is not None:
        if total_nelecas < 0 or total_nelecas > 2 * int(ncas):
            consistent = False
            consistency_messages.append('nelecas={0} is outside the allowed range [0, 2*ncas]'.format(nelecas))
    requested_total = _nelecas_total(active_space.nelecas)
    if selected_projection_candidate is None and requested_total is not None and estimated_nelecas is not None and requested_total != estimated_nelecas:
        consistent = False
        consistency_messages.append(
            'requested nelecas={0} differs from occupation-derived estimate {1}'.format(
                active_space.nelecas,
                estimated_nelecas,
            )
        )
    if not consistency_messages:
        consistency_messages.append('ncas/nelecas checks passed for the selected canonical orbitals')

    orbital_rows = []
    for row in table:
        index = int(row['index'])
        natural_matches = natural_occupations_by_index.get(index, [])
        contributions = _ao_atom_fragment_contributions(mf, mol, index) if index in selected_set else {
            'ao_contributions': [],
            'atom_contributions': [],
            'fragment_contributions': [],
            'fragment_definition': 'atom_as_fragment',
        }
        orbital_rows.append({
            'index': index,
            'index_basis': 'canonical_mo',
            'energy': row.get('energy'),
            'occupation': row.get('occupation'),
            'occupation_alpha': row.get('occupation_alpha'),
            'occupation_beta': row.get('occupation_beta'),
            'natural_orbital_occupations': [
                {
                    'natural_orbital_index': item.get('natural_orbital_index'),
                    'occupation': item.get('occupation'),
                    'dominant_canonical_weight': item.get('dominant_canonical_weight'),
                }
                for item in natural_matches
            ],
            't2_importance': t2_importance.get(index),
            'selected': index in selected_set,
            'selection_reasons': selection_reasons.get(index, []),
            'ao_contributions': contributions.get('ao_contributions', []),
            'atom_contributions': contributions.get('atom_contributions', []),
            'fragment_contributions': contributions.get('fragment_contributions', []),
        })

    evidence = []
    evidence.append('natural_occupations={0}'.format(natural_summary.get('status')))
    evidence.append('natural_occupation_quality={0}'.format(natural_occupation_quality.get('status')))
    evidence.append('t2_importance={0}'.format('available' if t2_importance else 'unavailable'))
    evidence.append('t2_supplement={0}'.format(t2_supplement.get('status')))
    if localization_method:
        evidence.append('localization_method={0}'.format(localization_method))
    evidence.append('localization_scope={0}'.format(localization_scope or 'analysis'))
    evidence.append('orbital_ordering={0}'.format(orbital_ordering or 'canonical'))

    target_solver_options = _target_solver_options(
        active_space,
        ncas=ncas,
        nelecas=nelecas,
        mol=mol,
    )
    selected_evaluation = (
        selected_candidate.get('evaluation')
        if isinstance(selected_candidate.get('evaluation'), dict)
        else {}
    )
    solver_route = str(selected_evaluation.get('solver_route') or 'review_required')
    exact_sector_dimension = selected_evaluation.get('exact_sector_dimension')
    if active_space.target_method in ('casci', 'casscf'):
        recommended_next_step = active_space.target_method
    elif solver_route == 'small_exact_sector':
        recommended_next_step = 'casscf_sc_nevpt2'
    else:
        recommended_next_step = 'block2_dmrg_casscf'
    method_recommendation = {
        'cas_candidate': solver_route in ('small_exact_sector', 'dmrg'),
        'recommended_next_step': recommended_next_step,
        'solver_route': solver_route,
        'exact_sector_dimension': exact_sector_dimension,
        'target_method': active_space.target_method,
        'target_solver': active_space.target_solver,
        'target_solver_options': copy.deepcopy(target_solver_options),
        'note': (
            'DMRG-CASCI and single-state or same-sector state-averaged DMRG-CASSCF are available '
            'through the optional block2 provider for restricted or ROHF references. Solver '
            'routing is based on the active electron sector rather than orbital count alone.'
        ),
    }
    audit_summary = {
        'status': 'approved' if active_space.approved else 'requires_user_review',
        'selected_orbital_count': len(selected),
        'orbital_count': len(table),
        'estimated_nelecas': estimated_nelecas,
        'ncas_nelecas_consistent': consistent,
        'evidence': evidence,
        'recommended_next_step': method_recommendation['recommended_next_step'],
        'candidate_methods': [candidate['method'] for candidate in candidate_active_spaces],
        'selected_candidate_method': selected_candidate_method,
        'selection_confidence': candidate_decision.get('confidence'),
        'recommendation_reason': candidate_decision.get('reason'),
    }
    audit = {
        'version': 1,
        'orbital_index_basis': 'canonical_mo',
        'selection_method': method,
        'selected_candidate_method': selected_candidate_method,
        'selected_orbital_basis': selected_candidate.get('orbital_index_basis'),
        'selection_parameters': {
            'occupation_window': list(active_space.occupation_window),
            'energy_window': active_space.energy_window,
            't2_supplement_threshold': t2_threshold,
            't2_supplement_energy_padding': T2_SUPPLEMENT_ENERGY_PADDING,
            'natural_occupation_physical_tolerance': NATURAL_OCCUPATION_PHYSICAL_TOLERANCE,
            'degenerate_manifold_tolerance': DEGENERATE_MANIFOLD_TOLERANCE,
            'requested_ncas': active_space.ncas,
            'requested_nelecas': active_space.nelecas,
            'requested_orbital_indices': _active_orbital_indices_payload(active_space),
            'avas_targets': list(active_space.avas_targets or []),
            'avas_threshold': active_space.avas_threshold,
            'avas_minimal_basis': active_space.avas_minimal_basis,
            'avas_with_iao': active_space.avas_with_iao,
            'avas_openshell_option': active_space.avas_openshell_option,
            'avas_ncore': active_space.avas_ncore,
        },
        'localization_method': localization_method or 'none',
        'localization_scope': localization_scope or 'analysis',
        'orbital_ordering': {
            'requested_method': orbital_ordering or 'canonical',
            'requested_order': [int(value) for value in (orbital_order or [])],
            'applied_order': 'recorded_after_correlated_solver_execution',
        },
        'natural_orbital_source': natural_summary.get('source'),
        'natural_orbital_status': natural_summary.get('status'),
        'natural_orbitals': natural_summary.get('orbitals', []),
        'natural_occupation_quality': natural_occupation_quality,
        't2_supplement': t2_supplement,
        'selection_evidence': selection_evidence,
        'candidate_decision': candidate_decision,
        'candidate_active_spaces': candidate_active_spaces,
        'selected_orbitals': [row for row in orbital_rows if row.get('selected')],
        'orbitals': orbital_rows,
        'ncas_nelecas_consistency': {
            'consistent': consistent,
            'messages': consistency_messages,
            'estimated_nelecas': estimated_nelecas,
            'final_ncas': ncas,
            'final_nelecas': nelecas,
        },
        'manual_approval': {
            'approved': bool(active_space.approved),
            'status': 'approved' if active_space.approved else 'requires_user_review',
            'recorded_in_request': bool(active_space.approved),
        },
        'method_recommendation': method_recommendation,
    }

    return {
        'enabled': bool(active_space.enabled),
        'selection_method': method,
        'ncas': ncas,
        'nelecas': nelecas,
        'orbital_indices': _active_orbital_indices_payload(active_space) if method == 'manual' and active_space.orbital_indices else selected,
        'initial_mo_coeff': (
            selected_projection_candidate.get('initial_mo_coeff')
            if isinstance(selected_projection_candidate, dict)
            and selected_projection_candidate.get('initial_mo_coeff') is not None
            else None
        ),
        'target_method': active_space.target_method,
        'target_solver': active_space.target_solver,
        'target_solver_options': copy.deepcopy(target_solver_options),
        'avas_targets': list(active_space.avas_targets or []) if method == 'avas' else [],
        'avas_parameters': copy.deepcopy(avas_candidate.get('avas_parameters') or {}) if method == 'avas' else {},
        'approved': bool(active_space.approved),
        'rationale': (
            '{0} {1}'.format(recovery_reason, candidate_decision.get('reason'))
            if recovery_reason
            else candidate_decision.get('reason')
            or 'Candidate active space generated from {0}.'.format(method)
        ),
        't2_orbital_importance_evidence': t2_evidence,
        'audit_summary': audit_summary,
        'audit': audit,
    }
