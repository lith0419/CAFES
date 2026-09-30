from __future__ import annotations

import copy
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

from ..registry.platform import (
    DEFAULT_ANALYSIS,
    SUPPORTED_ANALYSIS,
    SUPPORTED_AUXBASIS_SETS,
    SUPPORTED_JOBS,
    SUPPORTED_METHODS,
    normalize_molecular_method,
)
from ..registry import default_registry
from ..contracts import MolecularDynamicsSpec, materialize_block2_state_average_weights
from ..backend.active_space_probe import build_molecular_active_space_probe_request
from .approvals import build_active_space_approval
from .constants import HttpPost, PreparedRequest, SUPPORTED_REQUEST_FIELDS, LLM_RESPONSE_JSON_SCHEMA
from ..input_validation import integer, reject_unknown_fields
from .llm import _call_llm, get_llm_cache_scope, llm_request_builder_is_configured
from .modifications import (
    _collect_proposed_changes,
    _has_modification_intent,
    _rewrite_modification_questions,
)
from .structure import (
    _build_structure_approval,
)
from .utils import (
    _append_unique,
    _assistant_message,
    _compact_request_payload,
    _filter_questions_for_auto_fields,
    _json_request_text,
    _normalize_messages,
    _normalize_missing_fields,
    _normalize_questions,
    _normalize_string,
    _normalize_string_list,
    _precision_expectation,
    _record_default,
    _request_context_text,
    _seed_from_task_spec,
)
from ..pyscf_i18n import join_items, normalize_locale, t


def _coerce_supported_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    reject_unknown_fields(payload, LLM_RESPONSE_JSON_SCHEMA['schema']['properties'], 'prepared request')
    normalized = {
        key: copy.deepcopy(payload[key]) for key in SUPPORTED_REQUEST_FIELDS
        if key in payload and payload[key] is not None
    }
    if 'method' in normalized:
        normalized['method'] = _normalize_method_name(normalized['method']) or normalized['method']
    if 'solver' in normalized:
        solver = _normalize_solver_payload(normalized['solver'])
        if solver is not None:
            normalized['solver'] = solver
    return normalized


def _normalize_method_name(value: Any) -> Optional[str]:
    return normalize_molecular_method(value)


def _normalize_solver_payload(value: Any) -> Optional[Dict[str, Any]]:
    payload = copy.deepcopy(value) if isinstance(value, dict) else {'name': value}
    raw_name = _normalize_string(payload.get('name'))
    if raw_name is None:
        return None
    normalized_name = raw_name.lower().replace('-', '_').strip()
    normalized_name = {
        'block2': 'block2_dmrg',
        'dmrg': 'block2_dmrg',
        'block2_dmrg': 'block2_dmrg',
        'full_ci': 'fci',
        'full ci': 'fci',
    }.get(normalized_name, normalized_name)
    if normalized_name not in ('fci', 'block2_dmrg'):
        return None
    normalized = {'name': normalized_name}
    reject_unknown_fields(payload, ('name', 'options'), 'solver')
    if payload.get('options') is not None:
        if not isinstance(payload['options'], dict):
            raise ValueError('solver.options must be an object')
        normalized['options'] = copy.deepcopy(payload['options'])
    return normalized


def _matching_active_space_contract_in_seed(
    task_spec_seed: Dict[str, Any],
    structured_request: Dict[str, Any],
) -> bool:
    active_space = task_spec_seed.get('active_space')
    if not isinstance(active_space, dict):
        return False
    seed_method = _normalize_method_name(task_spec_seed.get('method'))
    request_method = _normalize_method_name(structured_request.get('method'))
    return bool(
        seed_method == request_method
        and active_space.get('ncas') is not None
        and active_space.get('nelecas') is not None
    )


def _prepare_active_space_requests(
    structured_request: Dict[str, Any],
    context_text: str,
    *,
    contract_from_seed: bool = False,
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Keep the requested CAS task separate from its active-space probe."""
    request = copy.deepcopy(structured_request)
    method = str(request.get('method') or '').strip().lower()
    if method not in ('casci', 'casscf'):
        return request, None

    solver = _normalize_solver_payload(request.get('solver')) or {'name': 'fci'}
    if method == 'casscf':
        solver['options'] = materialize_block2_state_average_weights(
            solver.get('name'),
            solver.get('options'),
        )
    active_space = copy.deepcopy(request.get('active_space')) if isinstance(request.get('active_space'), dict) else {}
    explicit_cas_size = bool(re.search(r'\bcas\s*\(\s*\d+\s*[,/]\s*\d+\s*\)', context_text, re.IGNORECASE))
    explicit_contract = contract_from_seed or explicit_cas_size
    if not contract_from_seed:
        active_space['approved'] = False
    selection_method = str(active_space.get('selection_method') or '').strip().lower()
    avas_ready = selection_method != 'avas' or bool(active_space.get('avas_targets'))
    has_active_space_contract = bool(
        active_space.get('ncas') is not None
        and active_space.get('nelecas') is not None
        and avas_ready
    )
    if has_active_space_contract and explicit_contract:
        request['solver'] = solver
        request['active_space'] = active_space
        return request, None

    final_active_space = copy.deepcopy(active_space)
    final_active_space.setdefault('enabled', True)
    final_active_space.setdefault('selection_method', 'manual')
    final_active_space['approved'] = False
    for field in ('target_method', 'target_solver', 'target_solver_options'):
        final_active_space.pop(field, None)
    request['active_space'] = final_active_space
    request['solver'] = solver
    request.pop('xc', None)

    probe_request = build_molecular_active_space_probe_request(
        request,
        strategy='auto',
        target_method=method,
        target_solver=solver['name'],
        target_solver_options=solver.get('options') or {},
        selection_method=selection_method,
    )['request']
    return request, probe_request


def _prepared_request_payload(
    structured_request: Dict[str, Any],
    probe_request: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    execution_request = probe_request or structured_request
    return {
        'request_text': _json_request_text(structured_request),
        'execution_request_text': _json_request_text(execution_request),
        'probe_request': copy.deepcopy(probe_request),
    }


def _apply_safe_defaults(
    task_spec_seed: Dict[str, Any],
    structured_request: Dict[str, Any],
    messages: List[Dict[str, str]],
    request_text: Optional[str],
    llm_payload: Dict[str, Any],
    locale: str,
) -> Dict[str, Any]:
    next_structured_request = copy.deepcopy(structured_request)
    applied_defaults: List[Dict[str, Any]] = []
    generated_fields: List[str] = []
    context_text = _request_context_text(messages, request_text, llm_payload, task_spec_seed)
    raw_method = _normalize_method_name(llm_payload.get('method'))
    raw_job = _normalize_string(llm_payload.get('job'))

    if not next_structured_request.get('job') and not raw_job:
        next_structured_request['job'] = 'single_point'
        _record_default(applied_defaults, 'job', 'single_point', 'default_single_point')

    is_molecular_dynamics = (
        next_structured_request.get('job') or raw_job
    ) == 'molecular_dynamics'
    profile_name = (next_structured_request.get('molecular_dynamics') or {}).get('profile')
    profile = default_registry().capability(
        'molecular_dynamics', namespace='molecular.job',
    ).metadata['profiles'].get(profile_name)
    if is_molecular_dynamics and profile is not None:
        qh9_defaults = {**{key: value for key, value in profile.items() if key != 'runtime'}, **profile['runtime']}
        for field_name, default_value in qh9_defaults.items():
            if next_structured_request.get(field_name) is None:
                next_structured_request[field_name] = default_value
                _record_default(
                    applied_defaults,
                    field_name,
                    default_value,
                    'qh9_molecular_dynamics_default',
                )
        molecular_dynamics = copy.deepcopy(
            next_structured_request.get('molecular_dynamics')
            if isinstance(next_structured_request.get('molecular_dynamics'), dict)
            else {}
        )
        molecular_dynamics_defaults = asdict(MolecularDynamicsSpec(profile=profile_name))
        for field_name, default_value in molecular_dynamics_defaults.items():
            if molecular_dynamics.get(field_name) is None:
                molecular_dynamics[field_name] = default_value
                _record_default(
                    applied_defaults,
                    'molecular_dynamics.{0}'.format(field_name),
                    default_value,
                    'qh9_molecular_dynamics_default',
                )
        next_structured_request['molecular_dynamics'] = molecular_dynamics

    outputs = _normalize_string_list(next_structured_request.get('outputs'))
    if not outputs:
        default_outputs = ['trajectory'] if is_molecular_dynamics else list(DEFAULT_ANALYSIS)
        next_structured_request['outputs'] = default_outputs
        _record_default(
            applied_defaults,
            'outputs',
            default_outputs,
            'molecular_dynamics_output_default' if is_molecular_dynamics else 'default_analysis_outputs',
        )

    if not next_structured_request.get('method') and not raw_method:
        next_structured_request['method'] = 'dft' if next_structured_request.get('xc') else 'hf'
        _record_default(applied_defaults, 'method', next_structured_request['method'], 'default_method')

    if next_structured_request.get('method') == 'dft' and not next_structured_request.get('xc'):
        next_structured_request['xc'] = 'b3lyp'
        _record_default(applied_defaults, 'xc', 'b3lyp', 'default_dft_functional')

    if 'unit' not in next_structured_request:
        next_structured_request['unit'] = 'Angstrom'
        _record_default(applied_defaults, 'unit', 'Angstrom', 'default_unit')

    if 'charge' not in next_structured_request:
        next_structured_request['charge'] = 0
        _record_default(applied_defaults, 'charge', 0, 'default_charge')

    if 'spin' not in next_structured_request:
        next_structured_request['spin'] = 0
        _record_default(applied_defaults, 'spin', 0, 'default_spin')

    for field in ('charge', 'spin'):
        next_structured_request[field] = integer(next_structured_request[field], field)

    if 'restricted' not in next_structured_request:
        if next_structured_request.get('method') in ('casci', 'casscf'):
            restricted = True
        else:
            restricted = next_structured_request.get('spin', 0) == 0
        next_structured_request['restricted'] = restricted
        _record_default(applied_defaults, 'restricted', restricted, 'inferred_from_spin')

    if 'symmetry' not in next_structured_request:
        next_structured_request['symmetry'] = False
        _record_default(applied_defaults, 'symmetry', False, 'default_symmetry')

    if 'max_cycle' not in next_structured_request:
        next_structured_request['max_cycle'] = 50
        _record_default(applied_defaults, 'max_cycle', 50, 'default_max_cycle')

    if 'conv_tol' not in next_structured_request:
        next_structured_request['conv_tol'] = 1e-8
        _record_default(applied_defaults, 'conv_tol', 1e-8, 'default_conv_tol')

    if 'verbose' not in next_structured_request:
        next_structured_request['verbose'] = 4
        _record_default(applied_defaults, 'verbose', 4, 'default_verbose')

    auto_fields = [item['field'] for item in applied_defaults]
    return {
        'structured_request': _compact_request_payload(next_structured_request),
        'applied_defaults': applied_defaults,
        'generated_fields': generated_fields,
        'auto_fields': auto_fields,
        'precision_note': _precision_expectation(next_structured_request, applied_defaults, generated_fields, locale=locale),
        'context_text': context_text,
    }


def _merge_request_summary(task_spec_seed: Dict[str, Any], llm_payload: Dict[str, Any], request_text: Optional[str]) -> Optional[str]:
    llm_summary = _normalize_string(llm_payload.get('request_summary'))
    if llm_summary:
        return llm_summary
    seed_summary = _normalize_string(task_spec_seed.get('request_summary'))
    if seed_summary:
        return seed_summary
    return request_text


def _build_consistency_feedback(task_spec_seed: Dict[str, Any], llm_payload: Dict[str, Any], *, locale: str) -> Dict[str, List[str]]:
    missing_fields = _normalize_missing_fields(llm_payload.get('missing_fields'))
    clarification_questions = _normalize_questions(llm_payload.get('clarification_questions'), locale=locale)

    raw_method = _normalize_method_name(llm_payload.get('method'))
    if raw_method and raw_method not in SUPPORTED_METHODS:
        _append_unique(missing_fields, 'method')
        _append_unique(
            clarification_questions,
            t(locale, 'prepare_supported_method'),
        )

    raw_job = _normalize_string(llm_payload.get('job'))
    if raw_job and raw_job not in SUPPORTED_JOBS:
        _append_unique(missing_fields, 'job')
        _append_unique(
            clarification_questions,
            t(locale, 'prepare_supported_job'),
        )

    raw_outputs = _normalize_string_list(llm_payload.get('outputs'))
    unsupported_outputs = [output for output in raw_outputs if output not in SUPPORTED_ANALYSIS]
    if unsupported_outputs:
        _append_unique(missing_fields, 'outputs')
        _append_unique(
            clarification_questions,
            t(
                locale,
                'prepare_supported_outputs',
                unsupported_outputs=join_items(locale, unsupported_outputs),
                supported_outputs=join_items(locale, SUPPORTED_ANALYSIS),
            ),
        )

    normalized_method = _normalize_method_name(llm_payload.get('method'))
    raw_xc = _normalize_string(llm_payload.get('xc'))
    if normalized_method == 'dft' and not raw_xc and not _normalize_string(task_spec_seed.get('xc')):
        _append_unique(missing_fields, 'xc')
        _append_unique(
            clarification_questions,
            t(locale, 'prepare_missing_xc'),
        )

    density_fitting = llm_payload.get('density_fitting')
    if isinstance(density_fitting, dict):
        raw_auxbasis = _normalize_string(density_fitting.get('auxbasis'))
        if raw_auxbasis and raw_auxbasis not in SUPPORTED_AUXBASIS_SETS:
            _append_unique(missing_fields, 'density_fitting.auxbasis')
            _append_unique(
                clarification_questions,
                'Choose a supported density-fitting auxiliary basis, or leave it blank for PySCF auto-selection.',
            )

    return {
        'missing_fields': missing_fields,
        'clarification_questions': clarification_questions,
    }


def _merge_structured_request(
    task_spec_seed: Dict[str, Any],
    llm_payload: Dict[str, Any],
    request_text: Optional[str],
    messages: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    structured_request = copy.deepcopy(task_spec_seed)
    structured_request.update(_coerce_supported_payload(llm_payload))
    if structured_request.get('method') != 'dft':
        structured_request.pop('xc', None)
    request_summary = _merge_request_summary(task_spec_seed, llm_payload, request_text)
    if request_summary:
        structured_request['request_summary'] = request_summary
    normalized_request = _normalize_string(structured_request.get('request')) or request_text
    if normalized_request:
        structured_request['request'] = normalized_request
    return _compact_request_payload(structured_request)


def build_prepared_request(
    messages: Optional[Sequence[Any]],
    *,
    task_spec: Optional[Dict[str, Any]] = None,
    request: Optional[str] = None,
    locale: str = 'en',
    http_post: Optional[HttpPost] = None,
) -> PreparedRequest:
    locale = normalize_locale(locale)
    llm_cache_scope = get_llm_cache_scope()
    normalized_messages = _normalize_messages(messages)
    normalized_request = _normalize_string(request)
    if normalized_request and (
        not normalized_messages
        or normalized_messages[-1]['role'] != 'user'
        or normalized_messages[-1]['content'] != normalized_request
    ):
        normalized_messages.append({'role': 'user', 'content': normalized_request})

    task_spec_seed = _seed_from_task_spec(task_spec)
    if not normalized_messages:
        if task_spec_seed:
            task_spec_seed, probe_request = _prepare_active_space_requests(
                task_spec_seed,
                '',
                contract_from_seed=True,
            )
            approval = None if probe_request is not None else build_active_space_approval(
                task_spec_seed,
                locale=locale,
            )
            return {
                'status': 'awaiting_approval' if approval is not None else 'ready',
                'source': 'structured_seed',
                'llm_cache_scope': llm_cache_scope,
                **_prepared_request_payload(task_spec_seed, probe_request),
                'structured_request': task_spec_seed,
                'approved_structured_request': (
                    approval['safe_structured_request'] if approval is not None else task_spec_seed
                ),
                'clarification_questions': [],
                'missing_fields': [],
                'approval': approval,
                'messages': (
                    [_assistant_message(approval['system_message'])] if approval is not None else []
                ),
            }
        return {
            'status': 'needs_clarification',
            'source': 'empty_input',
            'llm_cache_scope': llm_cache_scope,
            'request_text': '',
            'structured_request': {},
            'clarification_questions': [t(locale, 'prepare_empty_input')],
            'missing_fields': ['atom', 'basis'],
            'messages': [_assistant_message(t(locale, 'prepare_empty_input'))],
        }

    if not llm_request_builder_is_configured():
        return {
            'status': 'unavailable',
            'source': 'llm_unavailable',
            'llm_cache_scope': llm_cache_scope,
            'request_text': '',
            'structured_request': task_spec_seed,
            'execution_request_text': '',
            'clarification_questions': [],
            'missing_fields': [],
            'messages': normalized_messages + [_assistant_message(t(locale, 'prepare_unavailable'))],
            'error': 'LLM request builder is not configured; the natural-language request was not applied',
        }

    llm_payload = _call_llm(normalized_messages, task_spec_seed, locale=locale, http_post=http_post)
    llm_cache_scope = get_llm_cache_scope()
    structured_request = _merge_structured_request(task_spec_seed, llm_payload, normalized_request, normalized_messages)
    proposed_changes = _collect_proposed_changes(task_spec_seed, structured_request, locale)
    default_resolution = _apply_safe_defaults(task_spec_seed, structured_request, normalized_messages, normalized_request, llm_payload, locale)
    structured_request = default_resolution['structured_request']
    structured_request, probe_request = _prepare_active_space_requests(
        structured_request,
        default_resolution['context_text'],
        contract_from_seed=_matching_active_space_contract_in_seed(
            task_spec_seed,
            structured_request,
        ),
    )
    consistency_feedback = _build_consistency_feedback(task_spec_seed, llm_payload, locale=locale)
    clarification_questions = consistency_feedback['clarification_questions']
    missing_fields = consistency_feedback['missing_fields']
    filtered_feedback = _filter_questions_for_auto_fields(
        clarification_questions,
        missing_fields,
        default_resolution['auto_fields'],
    )
    clarification_questions = filtered_feedback['clarification_questions']
    missing_fields = filtered_feedback['missing_fields']
    clarification_questions = _rewrite_modification_questions(
        clarification_questions,
        proposed_changes,
        modification_intent=_has_modification_intent(normalized_messages, normalized_request, llm_payload),
        locale=locale,
    )
    structure_approval = _build_structure_approval(
        task_spec_seed,
        structured_request,
        llm_payload,
        normalized_messages,
        normalized_request,
        missing_fields,
        clarification_questions,
        locale,
    )
    active_space_approval = None if probe_request is not None else build_active_space_approval(
        structured_request,
        locale=locale,
    )
    if structure_approval is not None and active_space_approval is not None:
        structure_approval['next_approval'] = active_space_approval
    approval = structure_approval or active_space_approval

    safe_structured_request = structured_request
    if approval is not None:
        safe_structured_request = approval['safe_structured_request']

    if approval is not None:
        approval['probe_request'] = copy.deepcopy(probe_request)
        assistant_content = approval['system_message']
        return {
            'status': 'awaiting_approval',
            'source': 'llm',
            'llm_cache_scope': llm_cache_scope,
            **_prepared_request_payload(structured_request, probe_request),
            'structured_request': structured_request,
            'approved_structured_request': safe_structured_request,
            'applied_defaults': default_resolution['applied_defaults'],
            'generated_fields': default_resolution['generated_fields'],
            'precision_note': default_resolution['precision_note'],
            'clarification_questions': approval['remaining_clarification_questions'],
            'missing_fields': approval['remaining_missing_fields'],
            'approval': approval,
            'messages': normalized_messages + [_assistant_message(assistant_content)],
        }

    if clarification_questions or missing_fields:
        assistant_content = '\n'.join(clarification_questions) if clarification_questions else t(locale, 'prepare_needs_more_info')
        return {
            'status': 'needs_clarification',
            'source': 'llm',
            'llm_cache_scope': llm_cache_scope,
            'request_text': '',
            'execution_request_text': '',
            'probe_request': copy.deepcopy(probe_request),
            'structured_request': structured_request,
            'approved_structured_request': safe_structured_request,
            'applied_defaults': default_resolution['applied_defaults'],
            'generated_fields': default_resolution['generated_fields'],
            'precision_note': default_resolution['precision_note'],
            'clarification_questions': clarification_questions,
            'missing_fields': missing_fields,
            'messages': normalized_messages + [_assistant_message(assistant_content)],
        }

    return {
        'status': 'ready',
        'source': 'llm',
        'llm_cache_scope': llm_cache_scope,
        **_prepared_request_payload(structured_request, probe_request),
        'structured_request': structured_request,
        'approved_structured_request': safe_structured_request,
        'applied_defaults': default_resolution['applied_defaults'],
        'generated_fields': default_resolution['generated_fields'],
        'precision_note': default_resolution['precision_note'],
        'clarification_questions': [],
        'missing_fields': [],
        'messages': normalized_messages,
    }
