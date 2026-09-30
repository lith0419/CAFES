"""Lossless scalar conversion and field checks at scientific input boundaries."""

from __future__ import annotations

import math
from dataclasses import fields, is_dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Union, get_args, get_origin, get_type_hints


def integer(value: Any, path: str = 'value') -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError('{0} must be an integer'.format(path))
    try:
        number = Decimal(str(value).strip())
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError
        return int(number)
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise ValueError('{0} must be an integer'.format(path)) from exc


def integer_list(value: Any, path: str = 'value') -> list:
    if not isinstance(value, (list, tuple)):
        raise ValueError('{0} must be a list of integers'.format(path))
    return [integer(item, '{0}[{1}]'.format(path, index)) for index, item in enumerate(value)]


def finite_float(value: Any, path: str = 'value') -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError('{0} must be a finite number'.format(path))
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError('{0} must be a finite number'.format(path)) from exc
    if not math.isfinite(number):
        raise ValueError('{0} must be a finite number'.format(path))
    return number


def boolean(value: Any, path: str = 'value') -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ('true', 'yes', 'on', '1'):
            return True
        if normalized in ('false', 'no', 'off', '0'):
            return False
    raise ValueError('{0} must be a boolean'.format(path))


def reject_unknown_fields(value: Mapping[str, Any], allowed, path: str) -> None:
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError('Unknown {0} field(s): {1}'.format(path, ', '.join(sorted(unknown))))


def validate_dataclass_input(value: Any, contract, path: str = 'task_spec', *, strict: bool = True) -> dict:
    """Check owned fields before normalization; dictionaries keep their own contracts.

    Report readers may accept additive fields with strict=False. Execution
    requests use strict=True so a misspelled scientific option cannot disappear.
    """
    if not isinstance(value, dict):
        raise ValueError('{0} must be an object'.format(path))
    value = dict(value)
    declared = {item.name for item in fields(contract)}
    if strict:
        reject_unknown_fields(value, declared | {'schema'}, path)
    hints = get_type_hints(contract)
    for name in declared.intersection(value):
        item, annotation = value[name], hints[name]
        item_path = '{0}.{1}'.format(path, name)
        if get_origin(annotation) is Union:
            candidates = get_args(annotation)
            if item is None and type(None) in candidates:
                continue
            annotation = next((candidate for candidate in candidates if candidate is not type(None)), Any)
        if is_dataclass(annotation):
            value[name] = validate_dataclass_input(item, annotation, item_path, strict=strict)
        elif annotation is int:
            value[name] = integer(item, item_path)
        elif annotation is float:
            value[name] = finite_float(item, item_path)
        elif annotation is bool:
            value[name] = boolean(item, item_path)
        elif get_origin(annotation) is dict and not isinstance(item, dict):
            raise ValueError('{0} must be an object'.format(item_path))
    return value
