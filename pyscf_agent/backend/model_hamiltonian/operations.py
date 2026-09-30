from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List

from ...registry import default_registry
from ...input_validation import integer, integer_list


MODEL_OPERATION_NAMES = frozenset(
    item.id
    for item in default_registry().capabilities(
        namespace='model_hamiltonian.operation',
        backend_allowed=True,
    )
)


def _parameters_for_scope(scope: str) -> set:
    return {item.id for item in default_registry().parameters(scope=scope)}


SITE_PARAMETERS = _parameters_for_scope('site')
BOND_PARAMETERS = _parameters_for_scope('bond')
GLOBAL_PARAMETERS = _parameters_for_scope('global')


def normalize_nelec_value(value: Any) -> List[int]:
    """Normalize total-electron or spin-resolved input to ``[nalpha, nbeta]``."""

    if isinstance(value, dict):
        if 'alpha' in value and 'beta' in value:
            return [integer(value['alpha'], 'nelec.alpha'), integer(value['beta'], 'nelec.beta')]
        for key in ('nelec', 'total_electrons', 'total_electron', 'total', 'N', 'n'):
            if key in value:
                return normalize_nelec_value(value[key])
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            return integer_list(value, 'nelec')
        if len(value) == 1:
            return normalize_nelec_value(value[0])
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError('nelec cannot be empty')
    total = integer(value, 'nelec')
    if total < 0:
        raise ValueError('total electron count must be non-negative')
    return [(total + 1) // 2, total // 2]


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _site_lookup(model_spec: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {
        int(site['id']): site
        for site in model_spec.get('sites', [])
        if isinstance(site, dict) and 'id' in site
    }


def _bond_lookup(model_spec: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {
        int(bond['id']): bond
        for bond in model_spec.get('bonds', [])
        if isinstance(bond, dict) and 'id' in bond
    }


def _site_ids_from_selector(model_spec: Dict[str, Any], selector: Dict[str, Any]) -> List[int]:
    kind = str(selector.get('kind') or selector.get('type') or '').strip().lower()
    sites = [site for site in model_spec.get('sites', []) if isinstance(site, dict)]
    if kind in ('all', 'all_sites'):
        return [int(site['id']) for site in sites if 'id' in site]
    if kind in ('site_ids', 'ids'):
        return integer_list(_as_list(selector.get('ids', [])), 'selector.ids')
    if kind == 'boundary_sites':
        if not sites:
            return []
        xs = [float(site.get('x', 0.0)) for site in sites]
        ys = [float(site.get('y', 0.0)) for site in sites]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        tolerance = float(selector.get('tolerance', 1e-8))
        return [
            int(site['id'])
            for site in sites
            if 'id' in site and (
                abs(float(site.get('x', 0.0)) - min_x) <= tolerance
                or abs(float(site.get('x', 0.0)) - max_x) <= tolerance
                or abs(float(site.get('y', 0.0)) - min_y) <= tolerance
                or abs(float(site.get('y', 0.0)) - max_y) <= tolerance
            )
        ]
    if kind == 'nearest_to':
        x_target = float(selector.get('x', 0.0))
        y_target = float(selector.get('y', 0.0))
        count = integer(selector.get('count', 1), 'selector.count')
        ranked = sorted(
            sites,
            key=lambda site: (
                (float(site.get('x', 0.0)) - x_target) ** 2
                + (float(site.get('y', 0.0)) - y_target) ** 2
            ),
        )
        return [int(site['id']) for site in ranked[:count] if 'id' in site]
    if kind == 'within_radius':
        x_target = float(selector.get('x', 0.0))
        y_target = float(selector.get('y', 0.0))
        radius2 = float(selector.get('radius', 0.0)) ** 2
        return [
            int(site['id'])
            for site in sites
            if 'id' in site
            and (
                (float(site.get('x', 0.0)) - x_target) ** 2
                + (float(site.get('y', 0.0)) - y_target) ** 2
            ) <= radius2
        ]
    raise ValueError('Unsupported site selector: {0}'.format(kind or selector))


def resolve_site_ids(model_spec: Dict[str, Any], operation: Dict[str, Any]) -> List[int]:
    if 'site' in operation:
        if isinstance(operation['site'], (list, tuple)):
            raise ValueError('site must be one integer site ID; use sites for a list of site IDs')
        return [integer(operation['site'], 'site')]
    if 'sites' in operation:
        return integer_list(_as_list(operation['sites']), 'sites')
    selector = operation.get('selector')
    if isinstance(selector, dict):
        return _site_ids_from_selector(model_spec, selector)
    raise ValueError('Model operation requires site, sites, or selector')


def _bond_matches_pair(bond: Dict[str, Any], pair: Iterable[Any]) -> bool:
    source, target = pair
    return (
        int(bond.get('source')) == source and int(bond.get('target')) == target
    ) or (
        int(bond.get('source')) == target and int(bond.get('target')) == source
    )


def _bond_ids_from_selector(model_spec: Dict[str, Any], selector: Dict[str, Any]) -> List[int]:
    kind = str(selector.get('kind') or selector.get('type') or '').strip().lower()
    bonds = [bond for bond in model_spec.get('bonds', []) if isinstance(bond, dict)]
    if kind in ('all', 'all_bonds'):
        return [int(bond['id']) for bond in bonds if 'id' in bond]
    if kind in ('bond_ids', 'ids'):
        return integer_list(_as_list(selector.get('ids', [])), 'selector.ids')
    if kind in ('site_pair', 'endpoints', 'between_sites'):
        pair = selector.get('sites', selector.get('endpoints', selector.get('pair')))
        if pair is None and 'source' in selector and 'target' in selector:
            pair = [selector.get('source'), selector.get('target')]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError('Bond site-pair selector requires two site ids')
        pair = integer_list(pair, 'selector.sites')
        matches = [
            int(bond['id'])
            for bond in bonds
            if 'id' in bond and _bond_matches_pair(bond, pair)
        ]
        if not matches:
            raise ValueError('No bond matches site pair: {0}'.format(pair))
        return matches
    raise ValueError('Unsupported bond selector: {0}'.format(kind or selector))


def resolve_bond_ids(model_spec: Dict[str, Any], operation: Dict[str, Any]) -> List[int]:
    if 'source' in operation and 'target' in operation:
        return _bond_ids_from_selector(model_spec, {
            'kind': 'site_pair',
            'sites': [operation.get('source'), operation.get('target')],
        })
    if 'bond' in operation:
        value = operation['bond']
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError('bond must be one integer bond ID or two endpoint site IDs; use bonds for a list of bond IDs')
            return _bond_ids_from_selector(model_spec, {'kind': 'site_pair', 'sites': value})
        return [integer(value, 'bond')]
    if 'bonds' in operation:
        return integer_list(_as_list(operation['bonds']), 'bonds')
    selector = operation.get('selector')
    if isinstance(selector, dict):
        return _bond_ids_from_selector(model_spec, selector)
    raise ValueError('Model operation requires bond, bonds, or selector')


def _operation_value(operation: Dict[str, Any], mode: str) -> float:
    field = {'set': 'value', 'shift': 'shift', 'scale': 'factor'}[mode]
    if field not in operation and 'value' not in operation:
        raise ValueError('{0} operation requires {1}'.format(mode, field))
    return float(operation.get(field, operation.get('value')))


def _set_site_parameter(model_spec: Dict[str, Any], operation: Dict[str, Any], mode: str) -> None:
    parameter = str(operation.get('parameter') or 'epsilon')
    if parameter not in SITE_PARAMETERS:
        raise ValueError('Unsupported site parameter: {0}'.format(parameter))
    value = _operation_value(operation, mode)
    lookup = _site_lookup(model_spec)
    site_ids = resolve_site_ids(model_spec, operation)
    if not site_ids:
        raise ValueError('Site selector did not match any sites')
    for site_id in site_ids:
        if site_id not in lookup:
            raise ValueError('Unknown site id: {0}'.format(site_id))
        current = float(lookup[site_id].get(parameter, 0.0))
        lookup[site_id][parameter] = {
            'set': value,
            'shift': current + value,
            'scale': current * value,
        }[mode]


def _set_bond_parameter(model_spec: Dict[str, Any], operation: Dict[str, Any], mode: str) -> None:
    parameter = str(operation.get('parameter') or 't')
    if parameter not in BOND_PARAMETERS:
        raise ValueError('Unsupported bond parameter: {0}'.format(parameter))
    value = _operation_value(operation, mode)
    lookup = _bond_lookup(model_spec)
    bond_ids = resolve_bond_ids(model_spec, operation)
    if not bond_ids:
        raise ValueError('Bond selector did not match any bonds')
    for bond_id in bond_ids:
        if bond_id not in lookup:
            raise ValueError('Unknown bond id: {0}'.format(bond_id))
        current = float(lookup[bond_id].get(parameter, 0.0))
        lookup[bond_id][parameter] = {
            'set': value,
            'shift': current + value,
            'scale': current * value,
        }[mode]
        if parameter == 't':
            lookup[bond_id]['effective_t'] = lookup[bond_id][parameter]
        elif parameter == 'V':
            lookup[bond_id]['effective_V'] = lookup[bond_id][parameter]


def apply_model_operation(model_spec: Dict[str, Any], operation: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(operation, dict):
        raise ValueError('Model operation must be an object')
    op_name = str(operation.get('op') or operation.get('operation') or '').strip()
    if op_name not in MODEL_OPERATION_NAMES:
        raise ValueError('Unsupported model operation: {0}'.format(op_name or operation))
    if op_name == 'set_global_parameter':
        parameter = str(operation.get('parameter') or '').strip()
        if parameter not in GLOBAL_PARAMETERS:
            raise ValueError('Unsupported global model parameter: {0}'.format(parameter))
        if 'value' not in operation:
            raise ValueError('set_global_parameter requires value')
        value = operation['value']
        model_spec.setdefault('globals', {})[parameter] = value
        target = _set_site_parameter if parameter in SITE_PARAMETERS else _set_bond_parameter
        selector = {'kind': 'all_sites' if parameter in SITE_PARAMETERS else 'all_bonds'}
        target(model_spec, {'selector': selector, 'parameter': parameter, 'value': value}, 'set')
        return model_spec
    if op_name in ('set_site_parameter', 'shift_site_parameter', 'scale_site_parameter'):
        _set_site_parameter(model_spec, operation, op_name.split('_', 1)[0])
        return model_spec
    if op_name == 'add_site_defect':
        if 'parameter' in operation:
            _set_site_parameter(model_spec, operation, 'shift')
        else:
            applied = False
            for parameter, field in (('epsilon', 'epsilon_shift'), ('U', 'U_shift')):
                if field in operation:
                    _set_site_parameter(
                        model_spec,
                        {**operation, 'parameter': parameter, 'shift': operation[field]},
                        'shift',
                    )
                    applied = True
            if not applied:
                raise ValueError('add_site_defect requires parameter/value or epsilon_shift/U_shift')
        return model_spec
    if op_name in ('set_bond_parameter', 'shift_bond_parameter', 'scale_bond_parameter'):
        _set_bond_parameter(model_spec, operation, op_name.split('_', 1)[0])
        return model_spec
    if op_name == 'add_bond_defect':
        if 'parameter' in operation:
            _set_bond_parameter(model_spec, operation, 'shift')
        else:
            applied = False
            for parameter, field in (('t', 't_shift'), ('V', 'V_shift')):
                if field in operation:
                    _set_bond_parameter(
                        model_spec,
                        {**operation, 'parameter': parameter, 'shift': operation[field]},
                        'shift',
                    )
                    applied = True
            if not applied:
                raise ValueError('add_bond_defect requires parameter/value or t_shift/V_shift')
        return model_spec
    if op_name == 'change_nelec':
        value = operation.get(
            'nelec',
            operation.get(
                'total_electrons',
                operation.get('total_electron', operation.get('total', operation.get('value'))),
            ),
        )
        if value is None:
            raise ValueError('change_nelec requires an electron count')
        normalized = normalize_nelec_value(value)
        model_spec['nelec'] = normalized
        model_spec['spin_multiplicity'] = abs(normalized[0] - normalized[1]) + 1
        return model_spec
    if op_name == 'change_boundary':
        boundary = str(operation.get('boundary') or operation.get('value') or '').strip().lower()
        if boundary not in ('open', 'periodic'):
            raise ValueError('change_boundary requires boundary=open or boundary=periodic')
        model_spec['boundary'] = boundary
        return model_spec
    if op_name == 'change_solver':
        solver = str(operation.get('solver') or operation.get('value') or '').strip().lower()
        if not solver:
            raise ValueError('change_solver requires solver')
        model_spec['solver'] = solver
        return model_spec
    raise ValueError('Unsupported model operation: {0}'.format(op_name))


def apply_model_operations(
    model_spec: Dict[str, Any], operations: List[Dict[str, Any]], *, path: str = 'operations',
) -> Dict[str, Any]:
    if not isinstance(operations, list) or not operations:
        raise ValueError('Model operations must be a non-empty list')
    next_spec = copy.deepcopy(model_spec)
    for index, operation in enumerate(operations):
        try:
            apply_model_operation(next_spec, operation)
        except ValueError as exc:
            raise ValueError('{0}[{1}]: {2}'.format(path, index, exc)) from exc
    return next_spec


def describe_model_changes(before: Dict[str, Any], after: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a stable, user-reviewable audit of operation effects."""

    changes: List[Dict[str, Any]] = []
    before_sites, after_sites = _site_lookup(before), _site_lookup(after)
    for site_id in sorted(set(before_sites).intersection(after_sites)):
        for parameter in sorted(SITE_PARAMETERS):
            old = before_sites[site_id].get(parameter, 0.0)
            new = after_sites[site_id].get(parameter, 0.0)
            if old != new:
                changes.append({
                    'scope': 'site', 'id': site_id, 'parameter': parameter,
                    'before': old, 'after': new,
                })
    before_bonds, after_bonds = _bond_lookup(before), _bond_lookup(after)
    for bond_id in sorted(set(before_bonds).intersection(after_bonds)):
        for parameter in sorted(BOND_PARAMETERS):
            old = before_bonds[bond_id].get(parameter, 0.0)
            new = after_bonds[bond_id].get(parameter, 0.0)
            if old != new:
                changes.append({
                    'scope': 'bond', 'id': bond_id, 'parameter': parameter,
                    'before': old, 'after': new,
                })
    for field in ('nelec', 'spin_multiplicity', 'boundary', 'solver'):
        if before.get(field) != after.get(field):
            changes.append({
                'scope': 'model', 'id': None, 'parameter': field,
                'before': copy.deepcopy(before.get(field)),
                'after': copy.deepcopy(after.get(field)),
            })
    return changes
