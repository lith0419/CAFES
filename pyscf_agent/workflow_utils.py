from __future__ import annotations

from typing import Any, Mapping


def path_value(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for key in str(path or '').split('.'):
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def activation_rule_matches(rule: Any, payload: Mapping[str, Any]) -> bool:
    actual = path_value(payload, rule.path)
    if rule.operator == 'truthy':
        return bool(actual)
    if rule.operator == 'equals':
        return actual == rule.value
    if rule.operator == 'missing_or_equals':
        return actual in (None, '') or actual == rule.value
    if rule.operator == 'in':
        return actual in tuple(rule.value or ())
    if rule.operator == 'contains_any':
        if isinstance(actual, Mapping):
            values = set(actual)
        else:
            values = set(actual if isinstance(actual, (list, tuple, set)) else ())
        return bool(values.intersection(set(rule.value or ())))
    return False


__all__ = [
    'activation_rule_matches',
    'path_value',
]
