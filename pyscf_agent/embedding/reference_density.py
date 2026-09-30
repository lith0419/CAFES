from __future__ import annotations

from collections import deque
from math import isfinite
from typing import Any, Dict, List, Mapping, Sequence, Tuple


REFERENCE_DENSITY_GUESSES = ('pm', 'af', 'fm', 'cdw')
DEFAULT_REFERENCE_DENSITY_GUESS = 'pm'
DEFAULT_REFERENCE_DENSITY_BIAS = 0.05


def normalize_reference_density_guess(value: Any) -> str:
    normalized = str(value or DEFAULT_REFERENCE_DENSITY_GUESS).strip().lower().replace('-', '_')
    aliases = {
        'paramagnetic': 'pm',
        'antiferromagnetic': 'af',
        'ferromagnetic': 'fm',
        'charge_density_wave': 'cdw',
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in REFERENCE_DENSITY_GUESSES:
        raise ValueError(
            'reference_density_guess must be pm, af, fm, or cdw'
        )
    return normalized


def normalize_reference_density_bias(value: Any) -> float:
    if value in (None, ''):
        return DEFAULT_REFERENCE_DENSITY_BIAS
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('reference_density_bias must be a finite non-negative number') from exc
    if normalized < 0.0 or not isfinite(normalized):
        raise ValueError('reference_density_bias must be a finite non-negative number')
    return normalized


def _site_ids(spec: Mapping[str, Any]) -> List[int]:
    return sorted(int(site['id']) for site in spec.get('sites') or [])


def _bipartite_colors(spec: Mapping[str, Any]) -> Dict[int, int]:
    site_ids = _site_ids(spec)
    adjacency = {site_id: set() for site_id in site_ids}
    for bond in spec.get('bonds') or []:
        source = int(bond['source'])
        target = int(bond['target'])
        if source == target:
            raise ValueError('AF/CDW reference density requires a bipartite graph without self bonds')
        if source not in adjacency or target not in adjacency:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)

    colors: Dict[int, int] = {}
    for start in site_ids:
        if start in colors:
            continue
        colors[start] = 1
        pending = deque([start])
        while pending:
            source = pending.popleft()
            for target in sorted(adjacency[source]):
                expected = -colors[source]
                if target in colors and colors[target] != expected:
                    raise ValueError('AF/CDW reference density requires a bipartite model graph')
                if target not in colors:
                    colors[target] = expected
                    pending.append(target)
    return colors


def validate_reference_density_guess(
    spec: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> List[str]:
    errors: List[str] = []
    strategy = normalize_reference_density_guess(configuration.get('reference_density_guess'))
    reference = str(configuration.get('reference') or '').strip().lower()
    nalpha = int(configuration.get('nalpha') or 0)
    nbeta = int(configuration.get('nbeta') or 0)

    if strategy in ('af', 'fm') and reference != 'unrestricted':
        errors.append(
            '{0} reference-density initialization requires reference=unrestricted'.format(
                strategy.upper()
            )
        )
    if strategy == 'fm' and nalpha == nbeta:
        errors.append(
            'FM reference-density initialization requires a nonzero target spin sector; '
            'set unequal Nalpha/Nbeta (or a non-singlet spin multiplicity), or choose PM/AF'
        )
    if strategy in ('af', 'cdw'):
        try:
            colors = _bipartite_colors(spec)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            selected = _reference_orbital_site_ids(spec, configuration)
            selected_colors = {colors[site_id] for site_id in selected if site_id in colors}
            if len(selected_colors) < 2:
                errors.append('{0} reference density requires both sublattices in the reference cell'.format(
                    strategy.upper()
                ))
    return errors


def _reference_orbital_site_ids(
    spec: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> List[int]:
    if str(configuration.get('execution_mode') or '') == 'finite_graph':
        return _site_ids(spec)
    fragments = configuration.get('fragments') or []
    if fragments and isinstance(fragments[0], Mapping):
        selected = fragments[0].get('site_ids') or []
        if selected:
            return [int(site_id) for site_id in selected]
    return _site_ids(spec)


def _centered_af_pattern(
    spec: Mapping[str, Any],
    site_ids: Sequence[int],
    np: Any,
) -> Any:
    colors = _bipartite_colors(spec)
    pattern = np.asarray([float(colors[int(site_id)]) for site_id in site_ids], dtype=float)
    pattern = pattern - float(np.mean(pattern))
    scale = float(np.max(np.abs(pattern))) if pattern.size else 0.0
    if scale <= 0.0:
        raise ValueError('AF/CDW reference density requires both sublattices in the reference cell')
    return pattern / scale


def build_reference_density_seed(
    spec: Mapping[str, Any],
    configuration: Mapping[str, Any],
    orbital_count: int,
    *,
    np: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """Build a spin-resolved local 1-RDM seed without changing particle counts."""

    strategy = normalize_reference_density_guess(configuration.get('reference_density_guess'))
    errors = validate_reference_density_guess(spec, configuration)
    if errors:
        raise ValueError('; '.join(errors))

    orbital_site_ids = _reference_orbital_site_ids(spec, configuration)
    if len(orbital_site_ids) != int(orbital_count):
        raise ValueError(
            'Reference-density site count {0} does not match the local orbital count {1}'.format(
                len(orbital_site_ids),
                int(orbital_count),
            )
        )
    fillings = [float(value) for value in configuration['spin_filling']]
    diagonal = np.asarray([
        np.full(int(orbital_count), filling, dtype=float)
        for filling in fillings
    ])
    requested_bias = normalize_reference_density_bias(configuration.get('reference_density_bias'))
    applied_bias = 0.0
    pattern = np.zeros(int(orbital_count), dtype=float)

    if strategy == 'af':
        pattern = _centered_af_pattern(spec, orbital_site_ids, np)
        headroom = min(
            min(fillings),
            min(1.0 - filling for filling in fillings),
        )
        applied_bias = min(requested_bias, max(0.0, float(headroom)))
        diagonal[0] += applied_bias * pattern
        diagonal[1] -= applied_bias * pattern
    elif strategy == 'cdw':
        # Both spins follow the same staggered pattern: a sublattice charge
        # imbalance without magnetization. The centered pattern keeps N fixed.
        pattern = _centered_af_pattern(spec, orbital_site_ids, np)
        headroom = min(
            min(fillings),
            min(1.0 - filling for filling in fillings),
        )
        applied_bias = min(requested_bias, max(0.0, float(headroom)))
        diagonal[0] += applied_bias * pattern
        diagonal[1] += applied_bias * pattern
    elif strategy == 'fm':
        # Fixed Nalpha/Nbeta already determines the uniform magnetization. An
        # additional uniform shift would leave the requested particle sector.
        applied_bias = abs(fillings[0] - fillings[1])

    density = np.asarray([np.diag(spin_diagonal) for spin_diagonal in diagonal])
    metadata = {
        'strategy': strategy,
        'label': {
            'pm': 'Paramagnetic',
            'af': 'Antiferromagnetic',
            'fm': 'Ferromagnetic',
            'cdw': 'Charge density wave',
        }[strategy],
        'reference': str(configuration.get('reference') or ''),
        'requested_bias': requested_bias if strategy in ('af', 'cdw') else None,
        'applied_bias': applied_bias,
        'bias_kind': (
            'staggered_spin_density'
            if strategy == 'af'
            else (
                'staggered_charge_density'
                if strategy == 'cdw'
                else ('fixed_spin_sector' if strategy == 'fm' else 'none')
            )
        ),
        'scope': 'initial_uhf_density_only',
        'site_ids': list(orbital_site_ids),
        'pattern': [float(value) for value in pattern],
        'spin_fillings': fillings,
    }
    return density, metadata


__all__ = [
    'DEFAULT_REFERENCE_DENSITY_BIAS',
    'DEFAULT_REFERENCE_DENSITY_GUESS',
    'REFERENCE_DENSITY_GUESSES',
    'build_reference_density_seed',
    'normalize_reference_density_bias',
    'normalize_reference_density_guess',
    'validate_reference_density_guess',
]
