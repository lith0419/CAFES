from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from .utils import (
    _compact_request_payload,
    _filter_questions_for_auto_fields,
    _normalize_string,
)
from ..pyscf_i18n import t


def _user_explicitly_provided_atom(messages: List[Dict[str, str]], request_text: Optional[str], candidate_atom: str) -> bool:
    if not candidate_atom:
        return False
    text_parts: List[str] = []
    if request_text:
        text_parts.append(request_text)
    for message in messages:
        if message.get('role') == 'user':
            text_parts.append(message.get('content', ''))
    combined_text = '\n'.join(part for part in text_parts if part)
    return candidate_atom in combined_text


def _build_structure_approval(
    task_spec_seed: Dict[str, Any],
    structured_request: Dict[str, Any],
    llm_payload: Dict[str, Any],
    messages: List[Dict[str, str]],
    request_text: Optional[str],
    missing_fields: List[str],
    clarification_questions: List[str],
    locale: str,
) -> Optional[Dict[str, Any]]:
    candidate_atom = _normalize_string(structured_request.get('atom'))
    if not candidate_atom:
        return None

    seed_atom = _normalize_string(task_spec_seed.get('atom'))
    llm_atom = _normalize_string(llm_payload.get('atom'))
    structure_source = None
    if llm_atom and llm_atom != seed_atom:
        structure_source = 'llm_inferred_structure'

    if structure_source is None:
        return None
    if _user_explicitly_provided_atom(messages, request_text, candidate_atom):
        return None

    feedback = _filter_questions_for_auto_fields(clarification_questions, missing_fields, ['atom'])
    remaining_missing_fields = feedback['missing_fields']
    remaining_questions = feedback['clarification_questions']
    safe_structured_request = copy.deepcopy(structured_request)
    safe_structured_request.pop('atom', None)
    assumptions: List[str] = []
    assumption_summary = None
    return {
        'type': 'structure',
        'field': 'atom',
        'source': structure_source,
        'system_message': t(locale, 'prepare_approval_system_message'),
        'confirm_label': t(locale, 'prepare_confirm_label'),
        'cancel_label': t(locale, 'prepare_cancel_label'),
        'draft_value': candidate_atom,
        'structured_request': copy.deepcopy(structured_request),
        'safe_structured_request': _compact_request_payload(safe_structured_request),
        'remaining_missing_fields': remaining_missing_fields,
        'remaining_clarification_questions': remaining_questions,
        'can_run_immediately': not remaining_missing_fields and not remaining_questions,
        'assumption_summary': assumption_summary,
        'assumptions': assumptions,
    }
