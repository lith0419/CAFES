from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional

from ..backend.model_hamiltonian.operations import MODEL_OPERATION_NAMES
from ..registry import default_registry
from .constants import HttpPost, LLM_BASE_URL_ENV, LLM_MODEL_ENV, _STRUCTURED_OUTPUT_SUPPORT
from .llm import (
    _default_http_post,
    _extract_json_object,
    _extract_response_text,
    _extract_structured_payload,
    _llm_config_key,
    _llm_endpoint,
    _llm_headers,
    _llm_timeout,
    _should_retry_without_structured_output,
    llm_request_builder_is_configured,
)


MODEL_OPERATION_RESPONSE_SCHEMA: Dict[str, Any] = {
    'name': 'pyscf_agent_model_operations',
    'strict': False,
    'schema': {
        'type': 'object',
        'properties': {
            'operations': {
                'type': 'array',
                'items': {'type': 'object'},
            },
            'summary': {'type': 'string'},
            'clarification_questions': {
                'type': 'array',
                'items': {'type': 'string'},
            },
        },
        'required': ['operations', 'summary', 'clarification_questions'],
        'additionalProperties': False,
    },
}


def _model_context(model_spec: Dict[str, Any], *, max_items: int = 256) -> Dict[str, Any]:
    sites = []
    for site in model_spec.get('sites') or []:
        if not isinstance(site, dict) or 'id' not in site:
            continue
        sites.append({
            key: copy.deepcopy(site.get(key))
            for key in ('id', 'epsilon', 'U', 'x', 'y', 'sublattice', 'basis_index', 'cell_index')
            if site.get(key) is not None
        })
    bonds = []
    for bond in model_spec.get('bonds') or []:
        if not isinstance(bond, dict) or 'id' not in bond:
            continue
        bonds.append({
            key: copy.deepcopy(bond.get(key))
            for key in ('id', 'source', 'target', 't', 'V', 'kind', 'periodic', 'cell_offset')
            if bond.get(key) is not None
        })
    return {
        'model': model_spec.get('model'),
        'representation': model_spec.get('representation'),
        'boundary': model_spec.get('boundary'),
        'nelec': copy.deepcopy(model_spec.get('nelec')),
        'spin_multiplicity': model_spec.get('spin_multiplicity'),
        'sites': sites[:max_items],
        'bonds': bonds[:max_items],
        'site_count': len(sites),
        'bond_count': len(bonds),
        'truncated': len(sites) > max_items or len(bonds) > max_items,
    }


def _system_prompt() -> str:
    return '''
You translate one user's requested edit to an existing model Hamiltonian into validated model operations.

Rules:
- Return one JSON object only, with operations, summary, and clarification_questions.
- Use only operation names and parameter scopes supplied in the capability contract.
- "Site energy", "onsite energy", and "onsite potential" mean the site parameter epsilon. Hubbard interaction means U.
- Never infer a site id, bond id, sublattice, or equivalence class that is absent from model_context.
- For one or more explicit site ids, emit explicit site operations. Use selector={"kind":"all_sites"} only when the user clearly says every/all sites.
- For a bond identified by its endpoint sites, use bond=[source, target]; do not confuse endpoint ids with bond ids.
- Preserve every unmentioned model value. Do not generate calculation methods, scans, or result fields.
- If the target, parameter, requested value, or set-versus-shift intent is ambiguous, return no operations and ask one concise clarification question.
- Use set_* for an absolute requested value, shift_* for an increment/decrement, and scale_* for a multiplicative change.
- Canonical site forms are {"op":"set_site_parameter","site":1,"parameter":"epsilon","value":-0.5},
  {"op":"shift_site_parameter","sites":[1,2],"parameter":"U","shift":1.0}, and
  {"op":"scale_site_parameter","selector":{"kind":"all_sites"},"parameter":"epsilon","factor":2.0}.
- Canonical bond forms use bond, bonds, or selector in the same way; a bond endpoint pair is "bond":[source,target].
- change_nelec uses "nelec":[nalpha,nbeta], change_boundary uses "boundary", and change_solver uses "solver".
- Write the summary and questions in English.
'''.strip()


def _payload(
    request: str,
    model_spec: Dict[str, Any],
    messages: Optional[List[Dict[str, Any]]],
    *,
    structured_output: bool,
) -> Dict[str, Any]:
    registry = default_registry()
    operation_names = [
        item.id
        for item in registry.capabilities(
            namespace='model_hamiltonian.operation',
            backend_allowed=True,
        )
    ]
    site_parameters = [item.id for item in registry.parameters(scope='site')]
    bond_parameters = [item.id for item in registry.parameters(scope='bond')]
    body: Dict[str, Any] = {
        'model': os.getenv(LLM_MODEL_ENV),
        'temperature': 0,
        'messages': [
            {'role': 'system', 'content': _system_prompt()},
            {
                'role': 'user',
                'content': json.dumps({
                    'request': request,
                    'recent_conversation': list(messages or [])[-8:],
                    'model_context': _model_context(model_spec),
                    'capability_contract': {
                        'operation_names': operation_names,
                        'site_parameters': site_parameters,
                        'bond_parameters': bond_parameters,
                    },
                }, ensure_ascii=False),
            },
        ],
    }
    if structured_output:
        body['response_format'] = {
            'type': 'json_schema',
            'json_schema': copy.deepcopy(MODEL_OPERATION_RESPONSE_SCHEMA),
        }
    return body


def _parse_response(response: Dict[str, Any]) -> Dict[str, Any]:
    payload = _extract_structured_payload(response)
    if payload is None:
        payload = _extract_json_object(_extract_response_text(response))
    operations = payload.get('operations')
    questions = payload.get('clarification_questions')
    if not isinstance(operations, list) or any(not isinstance(item, dict) for item in operations):
        raise ValueError('Model operation response must contain an operations list')
    if not isinstance(questions, list):
        raise ValueError('Model operation response must contain clarification_questions')
    unsupported = [
        str(item.get('op') or item.get('operation') or '')
        for item in operations
        if str(item.get('op') or item.get('operation') or '') not in MODEL_OPERATION_NAMES
    ]
    if unsupported:
        raise ValueError('Unsupported model operation(s): {0}'.format(', '.join(unsupported)))
    return {
        'operations': copy.deepcopy(operations),
        'summary': str(payload.get('summary') or '').strip(),
        'clarification_questions': [str(item).strip() for item in questions if str(item).strip()],
    }


def build_model_hamiltonian_operations(
    request: str,
    model_spec: Dict[str, Any],
    *,
    messages: Optional[List[Dict[str, Any]]] = None,
    locale: str = 'en',
    http_post: Optional[HttpPost] = None,
) -> Dict[str, Any]:
    del locale  # This contract is intentionally English-only at present.
    if not isinstance(request, str) or not request.strip():
        raise ValueError('A model Hamiltonian edit request is required')
    if not isinstance(model_spec, dict):
        raise TypeError('model_spec must be an object')
    if not llm_request_builder_is_configured():
        raise RuntimeError('LLM request builder is not configured')
    base_url = os.getenv(LLM_BASE_URL_ENV)
    model = os.getenv(LLM_MODEL_ENV)
    if not base_url or not model:
        raise RuntimeError('LLM request builder is not configured')

    post = http_post or _default_http_post
    endpoint = _llm_endpoint(base_url)
    config_key = _llm_config_key(base_url, model)
    structured = config_key is None or _STRUCTURED_OUTPUT_SUPPORT.get(config_key) is not False
    try:
        response = post(
            endpoint,
            _payload(request, model_spec, messages, structured_output=structured),
            _llm_headers(),
            _llm_timeout(),
        )
        if structured and config_key is not None and _extract_structured_payload(response) is not None:
            _STRUCTURED_OUTPUT_SUPPORT[config_key] = True
    except Exception as exc:
        if not structured or not _should_retry_without_structured_output(exc):
            raise
        if config_key is not None:
            _STRUCTURED_OUTPUT_SUPPORT[config_key] = False
        response = post(
            endpoint,
            _payload(request, model_spec, messages, structured_output=False),
            _llm_headers(),
            _llm_timeout(),
        )
    return _parse_response(response)
