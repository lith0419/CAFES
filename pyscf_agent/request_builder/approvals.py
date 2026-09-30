from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from ..pyscf_i18n import t


def _solver_payload(value: Any) -> Dict[str, Any]:
    if isinstance(value, str) and value.strip():
        return {'name': value.strip().lower(), 'options': {}}
    if not isinstance(value, dict):
        return {'name': 'fci', 'options': {}}
    name = str(value.get('name') or 'fci').strip().lower()
    options = value.get('options') if isinstance(value.get('options'), dict) else {}
    return {'name': name, 'options': copy.deepcopy(options)}


def build_active_space_approval(
    structured_request: Dict[str, Any],
    *,
    locale: str,
) -> Optional[Dict[str, Any]]:
    """Build the preparation-time approval for an explicit CAS contract."""

    method = str(structured_request.get('method') or '').strip().lower()
    active_space = structured_request.get('active_space')
    if method not in ('casci', 'casscf') or not isinstance(active_space, dict):
        return None
    if active_space.get('approved'):
        return None

    ncas = active_space.get('ncas')
    nelecas = active_space.get('nelecas')
    if ncas is None or nelecas is None:
        return None

    solver = _solver_payload(structured_request.get('solver'))
    contract = {
        'enabled': True,
        'selection_method': str(active_space.get('selection_method') or 'manual').strip().lower(),
        'ncas': copy.deepcopy(ncas),
        'nelecas': copy.deepcopy(nelecas),
        'orbital_indices': copy.deepcopy(active_space.get('orbital_indices') or []),
        'avas_targets': copy.deepcopy(active_space.get('avas_targets') or []),
        'avas_threshold': active_space.get('avas_threshold', 0.2),
        'initial_mo_coeff': copy.deepcopy(active_space.get('initial_mo_coeff')),
        'target_method': method,
        'target_solver': solver['name'],
        'target_solver_options': solver['options'],
        'approved': False,
    }
    safe_request = copy.deepcopy(structured_request)
    safe_request['active_space'] = copy.deepcopy(contract)
    return {
        'type': 'active_space',
        'field': 'active_space',
        'status': 'requires_user_review',
        'source': 'prepared_active_space_contract',
        'reason_code': 'active_space_not_approved',
        'target_method': method,
        'target_solver': solver['name'],
        'system_message': t(
            locale,
            'prepare_active_space_approval_system_message',
            method=method.upper(),
            nelecas=nelecas,
            ncas=ncas,
        ),
        'confirm_label': t(locale, 'prepare_active_space_confirm_label'),
        'cancel_label': t(locale, 'prepare_cancel_label'),
        'structured_request': copy.deepcopy(structured_request),
        'safe_structured_request': safe_request,
        'active_space_contract': contract,
        'remaining_missing_fields': [],
        'remaining_clarification_questions': [],
        'can_run_immediately': True,
    }


__all__ = ['build_active_space_approval']
