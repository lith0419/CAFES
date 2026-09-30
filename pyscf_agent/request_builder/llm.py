from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from typing import Any, Dict, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from ..registry.platform import (
    default_registry,
    density_fitting_auxbasis_recommendations,
    density_fitting_scopes,
    artifact_kind_is_registered,
    result_analysis_output_contracts,
)
from .constants import (
    EXECUTION_FEEDBACK_PROMPT,
    HttpPost,
    LLM_API_KEY_ENV,
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
    LLM_RESPONSE_JSON_SCHEMA,
    LLM_TIMEOUT_ENV,
    RESULT_ANALYSIS_PROMPT,
    SYSTEM_PROMPT,
    _STRUCTURED_OUTPUT_SUPPORT,
)
from .utils import _normalize_string
from ..pyscf_i18n import normalize_locale, t


def llm_request_builder_is_configured() -> bool:
    return bool(os.getenv(LLM_BASE_URL_ENV) and os.getenv(LLM_MODEL_ENV))


def _llm_config_key(base_url: Optional[str] = None, model: Optional[str] = None) -> Optional[str]:
    normalized_base_url = _normalize_string(base_url if base_url is not None else os.getenv(LLM_BASE_URL_ENV))
    normalized_model = _normalize_string(model if model is not None else os.getenv(LLM_MODEL_ENV))
    if not normalized_base_url or not normalized_model:
        return None
    return '{0}|{1}'.format(normalized_base_url.rstrip('/'), normalized_model)


class LLMHTTPError(RuntimeError):
    def __init__(self, status: int, detail: str):
        self.status = status
        try:
            payload = json.loads(detail)
        except ValueError:
            payload = {}
        self.error = payload.get('error', {}) if isinstance(payload, dict) else {}
        if not isinstance(self.error, dict):
            self.error = {}
        super().__init__(f'LLM HTTP error {status}: {detail}')


def _structured_output_policy() -> str:
    policy = os.getenv('PYSCF_AGENT_LLM_STRUCTURED_OUTPUT', 'auto').strip().lower()
    if policy not in ('auto', 'true', 'false'):
        raise ValueError('PYSCF_AGENT_LLM_STRUCTURED_OUTPUT must be auto, true or false')
    return policy


def _structured_output_support_status(config_key: Optional[str]) -> str:
    policy = _structured_output_policy()
    if config_key is not None and policy != 'auto':
        return 'supported' if policy == 'true' else 'unsupported'
    if config_key is None:
        return 'disabled'
    support = _STRUCTURED_OUTPUT_SUPPORT.get(config_key)
    if support is True:
        return 'supported'
    if support is False:
        return 'unsupported'
    return 'unknown'


def get_llm_cache_scope() -> str:
    config_key = _llm_config_key()
    status = _structured_output_support_status(config_key)
    if config_key is None:
        return 'llm:disabled'
    digest = hashlib.sha256(config_key.encode('utf-8')).hexdigest()[:16]
    return 'llm:{0}:{1}'.format(digest, status)


def _extract_json_object(text: str) -> Dict[str, Any]:
    start = text.find('{')
    if start < 0:
        raise ValueError('LLM response does not contain a JSON object')
    # Decode one object inside optional prose/fences, respecting quoted braces
    # and escapes. Do not salvage a nested object from a malformed outer draft.
    try:
        payload, _end = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as exc:
        raise ValueError(
            'LLM response contains an invalid or incomplete JSON object: {0} '
            '(line {1}, column {2})'.format(exc.msg, exc.lineno, exc.colno)
        ) from exc
    return payload


def _extract_response_text(response: Dict[str, Any]) -> str:
    choices = response.get('choices')
    if not isinstance(choices, list) or not choices:
        raise ValueError('LLM response is missing choices')
    message = choices[0].get('message') if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise ValueError('LLM response is missing message payload')
    content = message.get('content')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get('text')
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return '\n'.join(parts)
    raise ValueError('LLM response does not include textual content')


def _extract_structured_payload(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    choices = response.get('choices')
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get('message') if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return None

    parsed = message.get('parsed')
    if isinstance(parsed, dict):
        return parsed

    content = message.get('content')
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get('parsed'), dict):
                return item['parsed']
            text = item.get('text')
            if isinstance(text, str):
                try:
                    parsed_text = _extract_json_object(text)
                except ValueError:
                    continue
                return parsed_text
    return None


def _default_http_post(url: str, body: Dict[str, Any], headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
    request = urllib_request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode('utf-8'),
        headers=headers,
        method='POST',
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode('utf-8')
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')
        raise LLMHTTPError(exc.code, detail) from exc
    except urllib_error.URLError as exc:
        raise RuntimeError('LLM request failed: {0}'.format(exc.reason)) from exc
    return json.loads(payload)


def _llm_timeout() -> float:
    raw_timeout = _normalize_string(os.getenv(LLM_TIMEOUT_ENV))
    if raw_timeout is None:
        return 30.0
    try:
        return float(raw_timeout)
    except ValueError:
        return 30.0


def _system_prompt(locale: str) -> str:
    return SYSTEM_PROMPT


def _llm_capability_payload() -> Dict[str, Any]:
    registry = default_registry()
    requestable_outputs = [
        item.id
        for item in registry.observables(
            namespace='molecular.observable',
            requestable=True,
            backend_allowed=True,
        )
    ]
    orbital_processing_methods = ['none'] + [
        item.id
        for item in registry.capabilities(
            namespace='molecular.orbital_processing', backend_allowed=True
        )
        if item.metadata.get('localization_method') is True
    ]
    return {
        'supported_methods': [item.id for item in registry.capabilities(
            namespace='molecular.method', backend_allowed=True
        )],
        'supported_jobs': [item.id for item in registry.capabilities(
            namespace='molecular.job', backend_allowed=True
        )],
        'supported_outputs': requestable_outputs,
        'supported_auxbasis_sets': [item.id for item in registry.option_values(
            'options.molecular.auxbasis', backend_allowed=True
        )],
        'supported_orbital_processing_methods': orbital_processing_methods,
        'supported_active_space_selection_methods': [item.id for item in registry.capabilities(
            namespace='molecular.active_space', backend_allowed=True
        )],
        'supported_active_space_solvers': [item.id for item in registry.capabilities(
            namespace='molecular.active_space_solver', backend_allowed=True
        )],
        'supported_embedding_impurity_solvers': [
            item.id
            for item in registry.capabilities(
                namespace='embedding.impurity_solver', backend_allowed=True
            )
        ],
        'supported_workflow_modules': [
            module.module_id for module in registry.modules_for(scope='task') if not module.always_select
        ],
        'common_xc_functionals': [item.id for item in registry.option_values(
            'options.molecular.xc', backend_allowed=True
        )],
        'density_fitting_auxbasis_recommendations': density_fitting_auxbasis_recommendations(registry),
        'capability_rules': [
            'Use only supported_methods, supported_jobs, supported_outputs, supported_auxbasis_sets, supported_orbital_processing_methods, supported_active_space_selection_methods, supported_active_space_solvers, supported_embedding_impurity_solvers, and supported_workflow_modules from this payload.',
            'For a DMET model task, keep solver.name=dmet and select its nested solver through solver.options.impurity_solver. block2_dmrg is valid there when listed in supported_embedding_impurity_solvers.',
            'For molecular_dynamics, select qh9 (SCF tolerance 1e-13) or qh9_relaxed_scf (1e-8) only for an explicitly requested QH9-like campaign and match the requested accuracy; otherwise preserve the requested method, basis and runtime controls.',
            'Keep density_fitting disabled unless the user explicitly asks for density fitting or task_spec_seed already enables it.',
            'If density fitting is enabled and no auxiliary basis was selected, set auxbasis to null for PySCF automatic selection.',
            'For restricted or ROHF CASSCF with density fitting, use apply_to=scf_and_casscf. Other molecular reference tasks use apply_to=scf.',
            'When choosing an auxiliary basis, use supported_auxbasis_sets and prefer density_fitting_auxbasis_recommendations for the orbital basis.',
            'Methods whose registry limitations require an explicit active-space contract must not be marked ready until ncas, nelecas, and approval are provided.',
        ],
    }


def _llm_response_json_schema() -> Dict[str, Any]:
    schema = copy.deepcopy(LLM_RESPONSE_JSON_SCHEMA)
    capabilities = _llm_capability_payload()
    properties = schema['schema']['properties']
    properties['method']['enum'] = capabilities['supported_methods'] + [None]
    properties['solver']['properties']['name']['enum'] = capabilities['supported_active_space_solvers']
    properties['job']['enum'] = capabilities['supported_jobs'] + [None]
    properties['orbital_processing']['properties']['localization_method']['enum'] = capabilities['supported_orbital_processing_methods']
    properties['density_fitting']['properties']['auxbasis']['enum'] = capabilities['supported_auxbasis_sets'] + [None]
    properties['density_fitting']['properties']['apply_to']['enum'] = density_fitting_scopes()
    properties['active_space']['properties']['selection_method']['enum'] = capabilities['supported_active_space_selection_methods']
    properties['workflow']['properties']['modules']['items']['enum'] = capabilities['supported_workflow_modules']
    return schema


def _build_llm_payload(messages, task_spec_seed, *, structured_output: bool, locale: str):
    capability_payload = _llm_capability_payload()
    payload = {
        'model': os.getenv(LLM_MODEL_ENV),
        'temperature': 0,
        'messages': [
            {
                'role': 'system',
                'content': _system_prompt(locale),
            },
            {
                'role': 'user',
                'content': json.dumps({
                    'messages': messages,
                    'task_spec_seed': task_spec_seed,
                    **capability_payload,
                }, ensure_ascii=False),
            },
        ],
    }
    if structured_output:
        payload['response_format'] = {
            'type': 'json_schema',
            'json_schema': _llm_response_json_schema(),
        }
    return payload


def _llm_headers() -> Dict[str, str]:
    headers = {
        'Content-Type': 'application/json; charset=utf-8',
    }
    api_key = _normalize_string(os.getenv(LLM_API_KEY_ENV))
    if api_key:
        headers['Authorization'] = 'Bearer {0}'.format(api_key)
    return headers


def _llm_endpoint(base_url: str) -> str:
    normalized = str(base_url or '').strip().rstrip('/')
    if normalized.endswith('/chat/completions'):
        return normalized
    if normalized.endswith('/responses'):
        return normalized[:-len('/responses')] + '/chat/completions'
    return normalized + '/chat/completions'


def _execution_feedback_prompt(locale: str) -> str:
    if normalize_locale(locale) == 'en':
        return '''
You are reviewing the result of a PySCF agent run for the user.
Reply in English, concise and actionable.

Rules:
- Only reply when the user should change the next prompt or structured request.
- If the task succeeded and no user action is needed, return an empty response.
- If the task failed, was blocked, unconverged, or missed key information, say what to change in the next prompt.
- If an ActiveSpaceAudit candidate requires approval, tell the user to review the Active Space approval panel instead of asking them to manually set approved=true in chat.
- For an unavailable periodic basis or pseudopotential, recommend only replacements explicitly listed by the backend as verified compatible for all structure elements. Never infer alternatives from basis-set names or size.
- Distinguish requested low-energy FCI or DMRG roots from a complete many-body spectrum. Do not call a calculation incomplete when all requested roots were returned.
- `state_energies` and `excitation_energies` are result fields; the corresponding requestable output is `excited_states`.
- `dmrg_spin_square` is a derived postprocessing metric; the requestable numerical output is `symmetry_analysis`.
- Treat task_spec.solver as the executed solver. Do not report a conflicting solver copied from nested Model Hamiltonian input metadata.
- For a DMET model task, `block2_dmrg` is a valid nested `solver.options.impurity_solver`; keep the top-level solver as `dmet` and do not advise moving the impurity solver to the top level.
- Keep active-space probes separate from the approved CAS calculation. A probe's SCF status is evidence for ActiveSpaceAudit and is not the final CAS task status.
- For CASSCF, `reference_converged=false` describes only the initial mean-field orbital guess when `reference_status.affects_task_status=false`. If CASSCF/DMRG and all targeted roots converged, treat the task as succeeded and mention the reference issue only as a limitation; do not ask for a rerun merely to clear that flag.
- Ground your advice in the execution result provided. Do not invent chemistry facts.
- Keep the answer within 3 short sentences.
'''.strip()
    return EXECUTION_FEEDBACK_PROMPT


def _result_analysis_prompt(locale: str) -> str:
    if normalize_locale(locale) == 'en':
        prompt = '''
You are writing a formal result analysis for a PySCF agent run.
Reply in English, professional, concise, and grounded only in the provided execution result.

Output requirements:
- Use the following section titles exactly once when applicable:
    1. Calculation Overview
    2. Key Results
    3. Interpretation
    4. Reliability and Limitations
    5. Next Steps
- Use complete sentences, not chatty dialogue.
- If the task failed or was blocked, clearly state what prevented a valid chemistry conclusion.
- For an unavailable periodic basis or pseudopotential, mention only replacements explicitly listed by the backend as verified compatible for all structure elements. If none are listed, do not invent one.
- Do not invent chemistry facts not supported by the execution result.
- Respect the relevant output contracts in the user payload; registered result fields and artifact kinds are generated outputs.
- Distinguish requested low-energy FCI or DMRG roots from a complete many-body spectrum, and do not call a completed target-root calculation incomplete.
- Refer to requestable capabilities rather than generated result fields. In particular, use `excited_states` and `symmetry_analysis`, not `state_energies`, `excitation_energies`, or `dmrg_spin_square`.
- Keep active-space probe results separate from the approved CAS calculation. For CASSCF, an unconverged initial mean-field orbital guess is a limitation rather than a failed final result when `reference_status.affects_task_status=false` and the CASSCF/DMRG solver converged.
- Keep the full answer within 8 short paragraphs or bullet-like lines.
'''.strip()
        return prompt
    return RESULT_ANALYSIS_PROMPT


def _result_context(request_text: str, execution_report: Dict[str, Any], locale: str) -> Dict[str, Any]:
    """Project available results once, with only the contracts they actually use."""
    try:
        request = json.loads(request_text)
    except (TypeError, ValueError):
        request = request_text
    compact = _compact_execution_report(execution_report)
    request_mapping = request if isinstance(request, dict) else {}
    study = request_mapping.get('planner_study_result') or {}
    task = execution_report.get('task_spec') or {}
    results = execution_report.get('structured_results') or {}
    system_type = (study.get('system_type') or results.get('task_type') or task.get('task_type')
                   or request_mapping.get('task_type') or '')
    fields, artifact_kinds = set(), set()

    def visit(value):
        if isinstance(value, dict):
            fields.update(value)
            if isinstance(value.get('kind'), str):
                artifact_kinds.add(value['kind'])
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(request)
    visit(execution_report)
    contracts = result_analysis_output_contracts(system_type=system_type)
    selected = {field: contract for field, contract in contracts.items()
                if field in fields or artifact_kinds.intersection(contract['artifact_kinds'])}
    return {
        'prepared_request': request,
        'execution_report': compact,
        'registered_output_contracts': selected,
        'response_language': t(locale, 'response_language'),
    }


def _parse_llm_response(response: Dict[str, Any]) -> Dict[str, Any]:
    structured_payload = _extract_structured_payload(response)
    if structured_payload is not None:
        return structured_payload
    response_text = _extract_response_text(response)
    return _extract_json_object(response_text)


def _should_retry_without_structured_output(exc: Exception) -> bool:
    if _structured_output_policy() != 'auto':
        return False
    if isinstance(exc, LLMHTTPError):
        if exc.status not in (400, 422):
            return False
        if (exc.error.get('param') in ('response_format', 'json_schema')
                and exc.error.get('code') in ('unsupported_parameter', 'unsupported_value', 'unsupported_response_format')):
            return True
        if exc.error.get('code') not in (None, '', 'invalid_request_error'):
            return False
    elif not isinstance(exc, RuntimeError) or not re.match(r'LLM HTTP error (400|422):', str(exc)):
        return False
    # Some gateways use a generic invalid_request_error with param=null for
    # unavailable response formats. Match that capability message narrowly;
    # specific schema-validation codes must never trigger a fallback.
    message = str(exc.error.get('message') or str(exc)).lower() if isinstance(exc, LLMHTTPError) else str(exc).lower()
    if re.search(r'\bresponse_format(?:\s+type)?\s+is\s+(?:currently\s+)?unavailable\b', message):
        return True
    if isinstance(exc, LLMHTTPError) and exc.error.get('code'):
        return False
    # Legacy compatibility for providers without structured error codes.
    return (any(field in message for field in ('response_format', 'json_schema', 'structured output'))
            and any(reason in message for reason in ('unsupported', 'not supported', 'does not support')))


def _call_llm(messages, task_spec_seed, *, locale: str = 'en', http_post: Optional[HttpPost] = None) -> Dict[str, Any]:
    base_url = os.getenv(LLM_BASE_URL_ENV)
    model = os.getenv(LLM_MODEL_ENV)
    if not base_url or not model:
        raise RuntimeError('LLM request builder is not configured')
    config_key = _llm_config_key(base_url, model)

    headers = _llm_headers()
    endpoint = _llm_endpoint(base_url)
    post = http_post or _default_http_post
    prefer_structured_output = _structured_output_support_status(config_key) != 'unsupported'
    if prefer_structured_output:
        try:
            response = post(endpoint, _build_llm_payload(messages, task_spec_seed, structured_output=True, locale=locale), headers, _llm_timeout())
            if config_key is not None and _extract_structured_payload(response) is not None:
                _STRUCTURED_OUTPUT_SUPPORT[config_key] = True
        except Exception as exc:
            if not _should_retry_without_structured_output(exc):
                raise
            if config_key is not None:
                _STRUCTURED_OUTPUT_SUPPORT[config_key] = False
            response = post(endpoint, _build_llm_payload(messages, task_spec_seed, structured_output=False, locale=locale), headers, _llm_timeout())
    else:
        response = post(endpoint, _build_llm_payload(messages, task_spec_seed, structured_output=False, locale=locale), headers, _llm_timeout())
    return _parse_llm_response(response)


def _compact_execution_report(report: Dict[str, Any]) -> Dict[str, Any]:
    registered_output_contracts = result_analysis_output_contracts()
    compact: Dict[str, Any] = {
        'execution_status': report.get('execution_status'),
        'calculation_role': report.get('calculation_role'),
        'analysis_summary': report.get('analysis_summary'),
        'validation_errors': list(report.get('validation_errors', [])),
        'clarification_questions': list(report.get('clarification_questions', [])),
        'errors': copy.deepcopy(report.get('errors', [])),
    }
    task_spec = report.get('task_spec')
    if isinstance(task_spec, dict):
        compact['task_spec'] = copy.deepcopy(task_spec)
    structured_results = report.get('structured_results')
    if isinstance(structured_results, dict):
        compact['structured_results'] = {
            key: structured_results.get(key)
            for key in (
                'task_type',
                'model',
                'solver',
                'method',
                'reference',
                'initial_reference',
                'cas_reference',
                'reference_converged',
                'reference_status',
                'solver_converged',
                'energy_unit',
                'converged',
                'active_space',
                'cas_result',
                'requested_outputs',
                'missing_requested_outputs',
                'requested_root_count',
                'computed_root_count',
                'targeted_roots_complete',
                'energy_spectrum_unavailable_reason',
            )
            if key in structured_results
        }
        for result_field, contract in registered_output_contracts.items():
            if result_field in structured_results:
                compact['structured_results'][result_field] = _compact_registered_result(
                    structured_results[result_field],
                    contract,
                )
        executed_solver = structured_results.get('solver')
        compact_task_spec = compact.get('task_spec')
        if executed_solver and isinstance(compact_task_spec, dict):
            compact_task_spec['solver'] = {
                'name': str(executed_solver),
                'options': copy.deepcopy(
                    compact_task_spec.get('solver', {}).get('options', {})
                    if isinstance(compact_task_spec.get('solver'), dict)
                    else {}
                ),
            }
            model_payload = compact_task_spec.get('model_hamiltonian')
            model_spec = model_payload.get('spec') if isinstance(model_payload, dict) else None
            if isinstance(model_spec, dict):
                model_spec['solver'] = str(executed_solver)
    artifacts = report.get('artifacts')
    if isinstance(artifacts, list):
        compact['artifacts'] = [
            {
                key: artifact.get(key)
                for key in ('kind', 'path', 'mime_type', 'description')
                if key in artifact
            } | {'registered_output': artifact_kind_is_registered(artifact.get('kind'))}
            for artifact in artifacts
            if isinstance(artifact, dict)
        ]
    return compact


def _compact_registered_result(result: Any, contract: Dict[str, Any]) -> Any:
    if not isinstance(result, dict):
        return copy.deepcopy(result)
    compact_result = {
        field_name: copy.deepcopy(result.get(field_name))
        for field_name in contract.get('summary_fields', [])
        if field_name in result
    }
    preview_tables = contract.get('preview_tables', {})
    if isinstance(preview_tables, dict):
        for table_name, table_spec in preview_tables.items():
            table = result.get(table_name)
            if not isinstance(table, list):
                continue
            limit = 8
            if isinstance(table_spec, dict) and table_spec.get('limit') is not None:
                try:
                    limit = max(0, int(table_spec['limit']))
                except (TypeError, ValueError):
                    limit = 8
            compact_result['{0}_preview'.format(table_name)] = copy.deepcopy(table[:limit])
            compact_result['{0}_row_count'.format(table_name)] = len(table)
    return compact_result


_PERIODIC_RESOURCE_UNAVAILABLE_RE = re.compile(
    r'Periodic (?P<kind>basis|pseudopotential) (?P<requested>\S+) is unavailable for element\(s\): '
    r'(?P<elements>[^.]+)\.'
)


def _deterministic_periodic_resource_feedback(
    execution_report: Dict[str, Any],
    locale: str,
) -> Optional[Dict[str, str]]:
    messages = [
        message
        for message in execution_report.get('validation_errors', [])
        if isinstance(message, str)
    ]
    for error in execution_report.get('errors', []):
        if isinstance(error, dict) and isinstance(error.get('message'), str):
            messages.append(error['message'])

    feedback_parts = []
    for message in dict.fromkeys(messages):
        unavailable = _PERIODIC_RESOURCE_UNAVAILABLE_RE.search(message)
        if unavailable is None:
            continue
        kind = unavailable.group('kind')
        requested = unavailable.group('requested')
        elements = unavailable.group('elements').strip()
        replacement_marker = (
            'Verified compatible registered {0} replacements for all structure elements: '.format(kind)
        )
        replacement_start = message.find(replacement_marker)
        replacements = []
        if replacement_start >= 0:
            replacement_text = message[replacement_start + len(replacement_marker):].split('.', 1)[0]
            replacements = [item.strip() for item in replacement_text.split(',') if item.strip()]

        if normalize_locale(locale) == 'en':
            resource_label = 'basis set' if kind == 'basis' else 'pseudopotential'
            field_name = 'periodic.basis' if kind == 'basis' else 'periodic.pseudo'
            if replacements:
                choices = ' or '.join('`{0}`'.format(item) for item in replacements)
                feedback_parts.append(
                    'The periodic {0} `{1}` is unavailable for {2}; change `{3}` to {4}. '
                    'These replacements were verified against the installed PySCF data for every element in the structure.'.format(
                        resource_label, requested, elements, field_name, choices
                    )
                )
            else:
                feedback_parts.append(
                    'The periodic {0} `{1}` is unavailable for {2}, and no registered replacement covers the complete structure. '
                    'Provide a custom PySCF {0} or extend the capability registry.'.format(
                        resource_label, requested, elements
                    )
                )
        else:
            resource_label = '周期基组' if kind == 'basis' else '周期赝势'
            field_name = 'periodic.basis' if kind == 'basis' else 'periodic.pseudo'
            if replacements:
                choices = ' 或 '.join('`{0}`'.format(item) for item in replacements)
                feedback_parts.append(
                    '当前{0} `{1}` 对 {2} 不可用；请将 `{3}` 改为 {4}。'
                    '这些替代项均已用本机 PySCF 数据对结构中的全部元素验证。'.format(
                        resource_label, requested, elements, field_name, choices
                    )
                )
            else:
                feedback_parts.append(
                    '当前{0} `{1}` 对 {2} 不可用，并且已注册选项中没有覆盖完整结构的替代项；'
                    '请提供自定义 PySCF {0}或扩展 capability registry。'.format(
                        resource_label, requested, elements
                    )
                )
    if not feedback_parts:
        return None
    return {
        'role': 'assistant',
        'content': ' '.join(feedback_parts),
    }


def _missing_requested_outputs(execution_report: Dict[str, Any]) -> list:
    structured_results = execution_report.get('structured_results')
    if not isinstance(structured_results, dict):
        return []
    explicit_missing = structured_results.get('missing_requested_outputs')
    if isinstance(explicit_missing, list):
        return [str(item) for item in explicit_missing if str(item).strip()]
    task_spec = execution_report.get('task_spec')
    if not isinstance(task_spec, dict):
        return []
    analysis = task_spec.get('analysis')
    requested = analysis.get('outputs') if isinstance(analysis, dict) else []
    if not isinstance(requested, list):
        return []
    required_fields = {
        'excited_states': ('state_energies', 'excitation_energies'),
        'symmetry_analysis': ('symmetry_analysis',),
        'entanglement_diagnostics': ('entanglement_diagnostics',),
        'gap': ('gap',),
        'energy_levels': ('energy_levels',),
        'energy_level_count': ('energy_level_count',),
        'many_body_basis_dimension': ('many_body_basis_dimension',),
    }
    missing = []
    for output in requested:
        fields = required_fields.get(str(output))
        if fields and any(structured_results.get(field) in (None, {}, []) for field in fields):
            missing.append(str(output))
    return missing


def _successful_execution_needs_feedback(execution_report: Dict[str, Any]) -> bool:
    if str(execution_report.get('execution_status') or '').lower() != 'succeeded':
        return True
    if execution_report.get('validation_errors') or execution_report.get('errors'):
        return True
    return bool(_missing_requested_outputs(execution_report))


def _deterministic_missing_output_feedback(
    execution_report: Dict[str, Any],
) -> Optional[Dict[str, str]]:
    if str(execution_report.get('execution_status') or '').lower() != 'succeeded':
        return None
    missing = _missing_requested_outputs(execution_report)
    if not missing:
        return None
    structured_results = execution_report.get('structured_results') or {}
    solver = str(structured_results.get('solver') or 'selected solver').upper()
    complete_spectrum = {
        'energy_levels',
        'energy_level_count',
        'many_body_basis_dimension',
    }
    if solver == 'BLOCK2_DMRG' and set(missing).issubset(complete_spectrum):
        content = (
            'The block2 DMRG calculation succeeded, but a complete many-body spectrum was requested and is not '
            'available from this workflow. Request `excited_states` for targeted low-energy roots; use FCI only '
            'when the full Hilbert space is small enough.'
        )
    else:
        content = (
            'The {0} calculation succeeded, but the registered output contract did not produce: {1}. '
            'Treat this as an output/provider issue and inspect the task log before rerunning; do not substitute '
            'generated result-field names or switch methods automatically.'
        ).format(solver, ', '.join('`{0}`'.format(item) for item in missing))
    return {'role': 'assistant', 'content': content}


def build_execution_feedback(
    request_text: str,
    execution_report: Dict[str, Any],
    *,
    locale: str = 'en',
    http_post: Optional[HttpPost] = None,
) -> Optional[Dict[str, str]]:
    normalized_request = _normalize_string(request_text)
    if not normalized_request or not isinstance(execution_report, dict):
        return None

    locale = normalize_locale(locale)
    deterministic_feedback = _deterministic_periodic_resource_feedback(execution_report, locale)
    if deterministic_feedback is not None:
        return deterministic_feedback
    deterministic_feedback = _deterministic_missing_output_feedback(execution_report)
    if deterministic_feedback is not None:
        return deterministic_feedback
    if not _successful_execution_needs_feedback(execution_report):
        return None
    if not llm_request_builder_is_configured():
        return None

    base_url = os.getenv(LLM_BASE_URL_ENV)
    model = os.getenv(LLM_MODEL_ENV)
    if not base_url or not model:
        return None

    payload = {
        'model': model,
        'messages': [
            {
                'role': 'system',
                'content': _execution_feedback_prompt(locale),
            },
            {
                'role': 'user',
                'content': json.dumps({
                    **_result_context(normalized_request, execution_report, locale),
                    'registered_embedding_impurity_solvers': _llm_capability_payload().get(
                        'supported_embedding_impurity_solvers', []
                    ),
                }, ensure_ascii=False),
            },
        ],
    }
    response = (http_post or _default_http_post)(
        _llm_endpoint(base_url),
        payload,
        _llm_headers(),
        _llm_timeout(),
    )
    content = _normalize_string(_extract_response_text(response))
    if not content:
        return None
    return {
        'role': 'assistant',
        'content': content,
    }


def build_result_analysis(
    request_text: str,
    execution_report: Dict[str, Any],
    *,
    locale: str = 'en',
    http_post: Optional[HttpPost] = None,
) -> Optional[str]:
    if not llm_request_builder_is_configured():
        return None

    normalized_request = _normalize_string(request_text)
    if not normalized_request or not isinstance(execution_report, dict):
        return None

    base_url = os.getenv(LLM_BASE_URL_ENV)
    model = os.getenv(LLM_MODEL_ENV)
    if not base_url or not model:
        return None

    locale = normalize_locale(locale)
    payload = {
        'model': model,
        'messages': [
            {
                'role': 'system',
                'content': _result_analysis_prompt(locale),
            },
            {
                'role': 'user',
                'content': json.dumps(_result_context(normalized_request, execution_report, locale), ensure_ascii=False),
            },
        ],
    }
    response = (http_post or _default_http_post)(
        _llm_endpoint(base_url),
        payload,
        _llm_headers(),
        _llm_timeout(),
    )
    return _normalize_string(_extract_response_text(response))
