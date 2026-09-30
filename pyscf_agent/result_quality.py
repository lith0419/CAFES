from __future__ import annotations

import copy
import math
from typing import Any, Dict, Mapping, Sequence


QUALITY_CHECKS_SCHEMA = 'pyscf-agent.quality-checks.v1'
QUALITY_CHECK_CATEGORIES = ('convergence', 'constraint')
QUALITY_CHECK_OPERATORS = (
    'absolute_less_than',
    'equal',
    'finite',
    'greater_than_or_equal',
    'less_than',
    'less_than_or_equal',
)


def _numeric(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError('a finite numeric value is required')
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError('a finite numeric value is required')
    return numeric


def _required_value(check: Mapping[str, Any], field: str) -> Any:
    if field not in check or check[field] is None:
        raise ValueError('{0} is required and must not be null'.format(field))
    return check[field]


def _compare(check: Mapping[str, Any]) -> bool:
    operator = str(check.get('operator') or '').strip()
    observed = _required_value(check, 'observed')
    if operator == 'finite':
        _numeric(observed)
        return True
    if operator == 'equal':
        expected = _required_value(check, 'expected')
        if isinstance(observed, bool) or isinstance(expected, bool):
            if not (isinstance(observed, bool) and isinstance(expected, bool)):
                raise TypeError('equal requires two booleans when either value is boolean')
        elif isinstance(observed, str) or isinstance(expected, str):
            if not (isinstance(observed, str) and isinstance(expected, str)):
                raise TypeError('equal requires two strings when either value is a string')
        else:
            _numeric(observed)
            _numeric(expected)
        return observed == expected
    observed_number = _numeric(observed)
    limit = _numeric(_required_value(check, 'limit'))
    if operator == 'absolute_less_than':
        return abs(observed_number) < limit
    if operator == 'less_than':
        return observed_number < limit
    if operator == 'less_than_or_equal':
        return observed_number <= limit
    if operator == 'greater_than_or_equal':
        return observed_number >= limit
    raise ValueError("unsupported operator '{0}'".format(operator))


def evaluate_quality_check(check: Any) -> Dict[str, Any]:
    """Evaluate one provider-reported result-quality observation."""

    if not isinstance(check, Mapping):
        return {
            'id': 'invalid_quality_check',
            'category': 'constraint',
            'required': True,
            'status': 'invalid',
            'message': 'Quality-check entries must be mappings.',
        }
    evaluated = copy.deepcopy(dict(check))
    check_id = str(evaluated.get('id') or '').strip()
    category = str(evaluated.get('category') or '').strip()
    operator = str(evaluated.get('operator') or '').strip()
    evaluated['id'] = check_id or 'invalid_quality_check'
    evaluated['category'] = category or 'constraint'
    evaluated['required'] = evaluated.get('required') is not False
    problems = []
    if not check_id:
        problems.append('id is required')
    if category not in QUALITY_CHECK_CATEGORIES:
        problems.append("unsupported category '{0}'".format(category))
    if operator not in QUALITY_CHECK_OPERATORS:
        problems.append("unsupported operator '{0}'".format(operator))
    if not str(evaluated.get('source') or '').strip():
        problems.append('source is required')
    if problems:
        evaluated['status'] = 'invalid'
        evaluated['message'] = '; '.join(problems)
        return evaluated
    try:
        passed = _compare(evaluated)
    except (TypeError, ValueError, OverflowError) as exc:
        evaluated['status'] = 'invalid'
        evaluated['message'] = str(exc)
        return evaluated
    evaluated['status'] = 'passed' if passed else 'failed'
    evaluated['message'] = (
        'Observed value satisfies the reported quality contract.'
        if passed
        else 'Observed value does not satisfy the reported quality contract.'
    )
    return evaluated


def evaluate_quality_checks(checks: Any) -> Dict[str, Any]:
    """Classify quality evidence; absent legacy evidence has unknown eligibility."""

    if checks is None:
        return {
            'schema': QUALITY_CHECKS_SCHEMA,
            'evidence_status': 'legacy_unavailable',
            'quality_status': 'legacy',
            'publication_eligible': None,
            'checks': [],
            'required_failed_ids': [],
            'required_invalid_ids': [],
            'convergence_failed_ids': [],
            'constraint_failed_ids': [],
        }
    if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)) or not checks:
        evaluated = [{
            'id': 'invalid_quality_checks',
            'category': 'constraint',
            'required': True,
            'status': 'invalid',
            'message': 'A declared quality_checks field must be a non-empty list.',
        }]
    else:
        evaluated = [evaluate_quality_check(check) for check in checks]
    required_failed = [
        check['id'] for check in evaluated
        if check.get('required') and check.get('status') == 'failed'
    ]
    required_invalid = [
        check['id'] for check in evaluated
        if check.get('required') and check.get('status') == 'invalid'
    ]
    convergence_failed = [
        check['id'] for check in evaluated
        if check.get('required')
        and check.get('category') == 'convergence'
        and check.get('status') in ('failed', 'invalid')
    ]
    constraint_failed = [
        check['id'] for check in evaluated
        if check.get('required')
        and check.get('category') == 'constraint'
        and check.get('status') in ('failed', 'invalid')
    ]
    if constraint_failed:
        quality_status = 'blocked'
    elif convergence_failed:
        quality_status = 'review_required'
    elif required_invalid:
        quality_status = 'blocked'
    else:
        quality_status = 'passed'
    return {
        'schema': QUALITY_CHECKS_SCHEMA,
        'evidence_status': 'available',
        'quality_status': quality_status,
        'publication_eligible': quality_status == 'passed',
        'checks': evaluated,
        'required_failed_ids': required_failed,
        'required_invalid_ids': required_invalid,
        'convergence_failed_ids': convergence_failed,
        'constraint_failed_ids': constraint_failed,
    }


__all__ = [
    'QUALITY_CHECKS_SCHEMA',
    'QUALITY_CHECK_CATEGORIES',
    'QUALITY_CHECK_OPERATORS',
    'evaluate_quality_check',
    'evaluate_quality_checks',
]
