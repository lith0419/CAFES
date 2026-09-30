from __future__ import annotations

import copy
import math
from decimal import Decimal
from typing import Any, Dict

from pyscf_agent.input_validation import finite_float, reject_unknown_fields


def normalize_system_type(value: Any) -> str:
    normalized = str(value or 'molecular').strip().lower().replace('-', '_')
    aliases = {
        'molecule': 'molecular',
        'molecular_electronic_structure': 'molecular',
        'model': 'model_hamiltonian',
        'model_hamiltonian': 'model_hamiltonian',
    }
    return aliases.get(normalized, normalized)


def normalize_study_mode(value: Any) -> str:
    normalized = str(value or 'static').strip().lower().replace('-', '_')
    aliases = {
        'static_scan': 'static',
        'adaptive_scan': 'adaptive',
    }
    return aliases.get(normalized, normalized)


def supported_study_modes(system_type: Any) -> tuple[str, ...]:
    if normalize_system_type(system_type) == 'model_hamiltonian':
        return ('static',)
    return ('static', 'adaptive')


def solver_contract(value: Any) -> tuple[str, dict[str, Any]]:
    """Return the solver name and options without applying registry aliases."""
    if isinstance(value, dict):
        name = str(value.get('name') or '').strip()
        options = value.get('options')
        return name, copy.deepcopy(options) if isinstance(options, dict) else {}
    return str(value or '').strip(), {}


def solver_payload(name: Any, options: Any = None) -> Any:
    normalized_name = str(name or '').strip()
    normalized_options = copy.deepcopy(options) if isinstance(options, dict) else {}
    if normalized_options:
        return {'name': normalized_name, 'options': normalized_options}
    return normalized_name


__all__ = [
    'normalize_study_mode',
    'normalize_system_type',
    'solver_contract',
    'solver_payload',
    'supported_study_modes',
]


def _has_molecular_geometry(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return bool(value)
    return value is not None


def _coordinate_number_text(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return format(number, '.12g')


def _molecular_atom_text(value: Any) -> str | None:
    """Convert common structured atom rows to the flat molecular request format."""
    if isinstance(value, str):
        return value
    if not isinstance(value, (list, tuple)) or not value:
        return None
    rows = []
    for entry in value:
        symbol = None
        coordinates = None
        if isinstance(entry, str):
            fields = entry.split()
            if len(fields) == 4:
                symbol, coordinates = fields[0], fields[1:]
        elif isinstance(entry, dict):
            symbol = entry.get('element', entry.get('symbol'))
            coordinates = entry.get('coordinates', entry.get('coordinate', entry.get('position')))
            if coordinates is None and all(key in entry for key in ('x', 'y', 'z')):
                coordinates = [entry['x'], entry['y'], entry['z']]
        elif isinstance(entry, (list, tuple)):
            if len(entry) == 2 and isinstance(entry[1], (list, tuple)):
                symbol, coordinates = entry[0], entry[1]
            elif len(entry) == 4:
                symbol, coordinates = entry[0], entry[1:]
        if not isinstance(symbol, str) or not symbol.strip() or not isinstance(coordinates, (list, tuple)):
            return None
        if len(coordinates) != 3:
            return None
        coordinate_text = [_coordinate_number_text(item) for item in coordinates]
        if any(item is None for item in coordinate_text):
            return None
        rows.append('{0} {1}'.format(symbol.strip(), ' '.join(coordinate_text)))
    return '; '.join(rows) if rows else None


def normalize_molecular_geometry_fields(request: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize legacy molecular coordinate aliases to the request ``atom`` field.

    REGRESSION-GUARD(planner-molecular-atom-contract): LLM drafts have
    repeatedly used ``geometry``/``coordinates`` aliases or nested PySCF-style
    atom rows while the execution contract accepts only flat ``atom`` text.
    Keep this compatibility boundary.
    """
    if not isinstance(request, dict):
        return request
    if not _has_molecular_geometry(request.get('atom')):
        for alias in ('geometry', 'coordinates'):
            if _has_molecular_geometry(request.get(alias)):
                request['atom'] = request[alias]
                break
    for alias in ('geometry', 'coordinates'):
        request.pop(alias, None)
    atom_text = _molecular_atom_text(request.get('atom'))
    if atom_text is not None:
        request['atom'] = atom_text
    return request


def normalize_case_variables(variables):
    """Expand explicit numeric ranges without guessing parameter identity."""
    if not isinstance(variables, dict):
        return variables
    normalized = copy.deepcopy(variables)
    for name, values in variables.items():
        if not isinstance(values, dict):
            continue
        reject_unknown_fields(values, ('start', 'stop', 'step'), 'case_design.variables.' + name)
        if set(values) != {'start', 'stop', 'step'}:
            raise ValueError('Numeric ranges require start, stop and step')
        for item in values.values():
            finite_float(item, 'case_design.variables.' + name)
        start, stop, step = (Decimal(str(values[key])) for key in ('start', 'stop', 'step'))
        if not step or (stop - start) / step < 0:
            raise ValueError('Numeric range step must point from start toward stop')
        count = int((stop - start) // step) + 1
        if count > 1000:
            raise ValueError('A numeric range may contain at most 1000 explicit values')
        normalized[name] = [float(start + i * step) for i in range(count)]
    return normalized
