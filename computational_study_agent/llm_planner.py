from __future__ import annotations

import copy
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pyscf_agent.request_builder.constants import (
    HttpPost,
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
    _STRUCTURED_OUTPUT_SUPPORT,
)
from pyscf_agent.request_builder.llm import (
    _default_http_post,
    _extract_json_object,
    _extract_response_text,
    _extract_structured_payload,
    _llm_config_key,
    _llm_endpoint,
    _llm_headers,
    _llm_timeout,
    _should_retry_without_structured_output,
    _structured_output_support_status,
    llm_request_builder_is_configured,
)
from pyscf_agent.backend.model_hamiltonian.solver import (
    load_model_spec_from_file,
    normalize_model_spec,
)
from pyscf_agent.schema_contracts import STUDY_SPEC_SCHEMA, validate_public_payload

from .schema import StudySpec
from .transformations import validate_model_parameter_updates
from .wiki_retriever import build_study_wiki_query, build_wiki_evidence_pack, format_wiki_evidence_for_prompt

LOGGER = logging.getLogger(__name__)


class PlannerDraftError(ValueError):
    """A rejected candidate with provider evidence, not a replacement draft."""

    def __init__(self, message: str, diagnostic: Dict[str, Any]):
        super().__init__(message)
        self.diagnostic = diagnostic


def _response_evidence(response: Dict[str, Any]) -> Dict[str, Any]:
    # Keep the public response channels used by parsing. Headers, credentials
    # and provider-specific reasoning channels are deliberately not recorded.
    choices = response.get('choices') or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get('message') or {}
    return {
        'model': response.get('model'), 'usage': response.get('usage'),
        'finish_reason': choice.get('finish_reason'),
        'message': {key: message[key] for key in ('content', 'parsed', 'refusal')
                    if isinstance(message, dict) and key in message},
    }


def save_planner_diagnostic(error: PlannerDraftError, work_dir: Any = None) -> str:
    """Persist a private failure record independently of scientific Study creation."""
    from pyscf_agent.paths import resolve_work_dir

    directory = resolve_work_dir(work_dir) / '.planner'
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / (uuid.uuid4().hex + '.json')
    record = dict(error.diagnostic, error=str(error), created_at=datetime.now(timezone.utc).isoformat())
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    LOGGER.warning('Planner failure evidence: %s', path)
    return str(path)


STUDY_SPEC_JSON_SCHEMA: Dict[str, Any] = {
    'name': 'pyscf_agent_study_spec',
    'strict': False,
    'schema': {
        'type': 'object',
        'properties': {
            'name': {'type': 'string'},
            'objective': {'type': 'string'},
            'system_type': {'type': 'string', 'enum': ['molecular', 'model_hamiltonian']},
            'base_task': {'type': 'object'},
            'base_model_input_file': {'type': ['string', 'null']},
            'base_model_spec': {'type': 'object'},
            'case_design': {'type': 'object'},
            'sweep': {'type': 'object'},
            'observables': {'type': 'array', 'items': {'type': 'string'}},
            'comparison': {'type': 'object'},
            'resource_policy': {'type': 'object'},
        },
        'required': ['name', 'objective', 'system_type', 'base_task', 'observables'],
        'additionalProperties': True,
    },
}


def _study_system_prompt(locale: str) -> str:
    return '''
You are the computational study planner for PySCF Agent. Convert the user's scientific goal into a constrained StudySpec JSON draft.

Rules:
- Return only one JSON object.
- Return a complete StudySpec, including name, objective, system_type, base_task, and observables. Do not return a partial update or rely on implicit defaults.
- When revising an existing scan, include the complete sweep or case_design. An absent or empty scan is not a single-point revision. If the user explicitly requests replacing a scan with one point, represent that point as an explicit nonempty case_design.cases list.
- Use the retrieved wiki_evidence for scientific planning guidance and documented capability boundaries. Do not treat roadmap items as executable capabilities. The backend validates methods, solvers, parameters, and observables against the runtime registry before a plan can run.
- Follow study_spec_schema for the output format. Wiki guidance does not change this response contract.
- For molecular studies, use system_type "molecular" and put geometry, basis, charge, spin, reference, etc. into base_task. Molecular coordinates must use the field base_task.atom; never use geometry or coordinates.
- For model Hamiltonian studies, use system_type "model_hamiltonian". If the user provides a builder input path, put it in base_model_input_file; do not invent a lattice spec.
- If seed_study_spec already contains base_model_spec or base_model_input_file, preserve that model context unless the user explicitly asks to change models.
- If seed_study_spec already defines a scan and the user asks to change the method, basis, or runtime for one or a few existing points, preserve the complete scan and every unaffected case. For a grid scan, use case_design.overrides, for example {"selector": {"bond_factor": 1.6}, "request_updates": {"method": "casscf", "xc": null}}. Never replace the scan with an empty mode="cases" design or write a point-specific method into base_task, which would change every case.
- Use sweep for simple parameter scans, such as {"method": ["hf", "mp2"]} or {"U": [2, 4, 6]}.
- Hamiltonian coefficients U, V, t and epsilon belong in sweep or case_design.template.operations, never in request_updates.parameters. For example, variables {"U": [1, 2], "V": [0.5, 1]} use operations [{"op": "set_global_parameter", "parameter": "U", "value": "$U"}, {"op": "set_global_parameter", "parameter": "V", "value": "$V"}]. Preserve the seeded model and solver contract.
- Preserve seed_study_spec.grid_refinement. Refined axes may change only continuous Hamiltonian coefficient values, not solver options (including impurity beta), topology or electron counts. Put fixed solver settings in base_task.solver.
- Use case_design for richer studies: mode may be "grid" or "cases"; variables define case variables; template.operations defines controlled model Hamiltonian operations; template.request_updates defines method/solver/request-field changes.
- For mode="grid", every case_design.variables value must be a non-empty JSON array. Expand ranges directly, for example 0.2x to 3.0x in 0.2x steps becomes {"bond_factor": [0.2, 0.4, ..., 3.0]}; do not use a scalar or {"start": ..., "stop": ..., "step": ...} object.
- Use "$bond_length" or "${bond_length}" for direct template substitution. For derived numeric values, use the form "$(scale * reference_value)" only when every name is a defined numeric variable, and never put arithmetic inside "${...}". Expressions may contain numeric variables, numbers, +, -, *, /, and parentheses, including inside a molecular atom-coordinate string.
- For model Hamiltonian case_design.operations, use operations and parameter scopes documented in wiki_evidence; do not invent operation names.
- For defect-position scans, prefer case_design.mode="grid", put site ids in variables, for example {"defect_site": [0, 1, 2]}, and use a template operation like {"op": "add_site_defect", "site": "$defect_site", "parameter": "epsilon", "value": <defect_strength>}; do not invent an unsupported scalar field named defect_position.
- Site operations use site for one integer site ID, sites for a list of integer site IDs, or selector. A template reference must resolve to the corresponding scalar or list; never put a list in site. Bond operations must include bond, bonds, or selector.
- For equivalent-site requests, use only the explicit Builder metadata in model_site_context. Do not infer equivalence from coordinates and do not invent selector kinds.
- For an explicitly identified site group from model_site_context, use one operation with sites containing the group IDs. The shared operation engine applies the update to every selected site without splitting the operation.
- Bond ids and site ids are different namespaces. If the user says "bond 2,4" and means the bond between site 2 and site 4, write {"bond": [2, 4]} or {"selector": {"kind": "site_pair", "sites": [2, 4]}}; write {"bonds": [2, 4]} only when the user explicitly says bond indices/ids 2 and 4.
- If seed/base_model_spec contains graph.edges or bonds.source/target, use it to verify site-pair bonds; do not leave placeholder bond ids in notes.
- For electron-count scans, PySCF expects nelec=[nalpha, nbeta]. If the user gives total electron count N, use [N/2, N/2] for even N and [(N+1)/2, (N-1)/2] for odd N; do not treat scalar total electron count as final nelec.
- If information is incomplete, still create a conservative draft and put user-confirmation notes in comparison.notes.
- Preserve seed_study_spec.resource_policy; deterministic cost-estimate artifacts and approval state control execution, so never set approved to true on your own.
- Prefer small, low-cost plans by default; do not create many expensive cases unless requested.
'''.strip()


def _parse_llm_response(response: Dict[str, Any]) -> Dict[str, Any]:
    structured_payload = _extract_structured_payload(response)
    if structured_payload is not None:
        return structured_payload
    return _extract_json_object(_extract_response_text(response))


def _study_spec_from_llm(payload: Dict[str, Any], seed_spec: Optional[Dict[str, Any]]) -> StudySpec:
    """Validate generated drafts before legacy input defaults can fill omissions."""
    candidate = copy.deepcopy(payload)
    candidate.setdefault('schema', STUDY_SPEC_SCHEMA)
    try:
        validate_public_payload(candidate, expected_schema=STUDY_SPEC_SCHEMA)
    except (TypeError, ValueError) as exc:
        raise ValueError('{0}; received top-level fields: {1}'.format(
            exc, ', '.join(sorted(payload)) or '(empty object)',
        )) from exc
    for field in ('name', 'objective', 'system_type'):
        if not isinstance(candidate[field], str) or not candidate[field].strip():
            raise ValueError('Planner StudySpec.{0} must be a nonempty string'.format(field))
    for field in ('base_task', 'case_design', 'sweep'):
        if field in candidate and not isinstance(candidate[field], dict):
            raise ValueError('Planner StudySpec.{0} must be a JSON object'.format(field))
    if not isinstance(candidate['observables'], list) or not candidate['observables']:
        raise ValueError('Planner StudySpec.observables must be a nonempty array')
    spec = StudySpec.from_dict(candidate)
    if spec.system_type == 'model_hamiltonian':
        validate_model_parameter_updates(spec.case_design)
    if isinstance(seed_spec, dict) and (seed_spec.get('sweep') or seed_spec.get('case_design')):
        design = spec.case_design
        if not (spec.sweep or design.get('variables') or design.get('cases')):
            raise ValueError(
                'A scan revision must include the complete sweep or case_design. '
                'To replace it with a single point, supply an explicit case_design.cases entry.'
            )
    return spec


def _model_site_context(seed_spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Expose Builder-declared site identity without adding new planner selectors."""

    if not isinstance(seed_spec, dict):
        return {}
    system_type = str(seed_spec.get('system_type') or '').strip().lower().replace('-', '_')
    if system_type != 'model_hamiltonian':
        return {}
    try:
        input_file = seed_spec.get('base_model_input_file')
        if isinstance(input_file, str) and input_file.strip():
            model_spec = normalize_model_spec(load_model_spec_from_file(input_file.strip()))
        elif isinstance(seed_spec.get('base_model_spec'), dict):
            model_spec = normalize_model_spec(copy.deepcopy(seed_spec['base_model_spec']))
        else:
            return {}
    except Exception as exc:
        return {
            'available': False,
            'reason': 'Builder model context could not be loaded: {0}'.format(exc),
        }

    site_records = []
    groups: Dict[str, Dict[str, list]] = {
        'sublattice': {},
        'basis_index': {},
    }
    for site in model_spec.get('sites') or []:
        if not isinstance(site, dict) or 'id' not in site:
            continue
        site_id = int(site['id'])
        record = {
            'id': site_id,
            'U': site.get('U'),
            'epsilon': site.get('epsilon'),
        }
        for field in ('sublattice', 'basis_index', 'cell_index'):
            if site.get(field) is not None:
                record[field] = copy.deepcopy(site[field])
        site_records.append(record)
        for field in ('sublattice', 'basis_index'):
            value = site.get(field)
            if value is not None:
                groups[field].setdefault(str(value), []).append(site_id)

    declared_groups = {
        field: values
        for field, values in groups.items()
        if values
    }
    return {
        'available': True,
        'site_count': len(site_records),
        'sites': site_records,
        'builder_declared_groups': declared_groups,
        'rule': 'Use declared groups only as evidence, then emit explicit site operations.',
    }


def _build_payload(
    goal: str,
    *,
    seed_spec: Optional[Dict[str, Any]],
    structured_output: bool,
    locale: str,
    wiki_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    evidence_text = format_wiki_evidence_for_prompt(wiki_evidence or {})
    payload = {
        'model': os.getenv(LLM_MODEL_ENV),
        'temperature': 0,
        'messages': [
            {'role': 'system', 'content': _study_system_prompt(locale)},
            {
                'role': 'user',
                'content': json.dumps({
                    'goal': goal,
                    'seed_study_spec': seed_spec or {},
                    'model_site_context': _model_site_context(seed_spec),
                    'wiki_evidence': evidence_text,
                    'study_spec_schema': STUDY_SPEC_JSON_SCHEMA['schema'],
                }, ensure_ascii=False),
            },
        ],
    }
    if structured_output:
        payload['response_format'] = {
            'type': 'json_schema',
            'json_schema': copy.deepcopy(STUDY_SPEC_JSON_SCHEMA),
        }
    return payload


def _build_contract_repair_payload(
    goal: str,
    *,
    seed_spec: Optional[Dict[str, Any]],
    structured_output: bool,
    locale: str,
    wiki_evidence: Dict[str, Any],
    invalid_payload: Any,
    validation_error: Exception,
) -> Dict[str, Any]:
    """Ask the planner to repair one invalid draft against the same contract."""

    payload = _build_payload(
        goal,
        seed_spec=seed_spec,
        structured_output=structured_output,
        locale=locale,
        wiki_evidence=wiki_evidence,
    )
    payload['messages'].extend([
        {
            'role': 'assistant',
            'content': (
                invalid_payload if isinstance(invalid_payload, str)
                else json.dumps(invalid_payload, ensure_ascii=False)
            ),
        },
        {
            'role': 'user',
            'content': json.dumps({
                'instruction': (
                    'The previous response was not valid StudySpec JSON. Return one '
                    'complete corrected StudySpec JSON object while preserving the '
                    'scientific request, the seed study, and every valid field from '
                    'the prior draft. Return JSON only, without explanatory prose. '
                    'Do not invent missing scientific choices; record clarification '
                    'needs in comparison.notes.'
                ),
                'validation_error': str(validation_error),
            }, ensure_ascii=False),
        },
    ])
    return payload


def build_study_spec_from_goal(
    goal: str,
    *,
    seed_spec: Optional[Dict[str, Any]] = None,
    locale: str = 'en',
    http_post: Optional[HttpPost] = None,
) -> Dict[str, Any]:
    result = build_study_spec_from_goal_with_evidence(
        goal,
        seed_spec=seed_spec,
        locale=locale,
        http_post=http_post,
    )
    return result['study_spec']


def build_study_spec_from_goal_with_evidence(
    goal: str,
    *,
    seed_spec: Optional[Dict[str, Any]] = None,
    locale: str = 'en',
    http_post: Optional[HttpPost] = None,
) -> Dict[str, Any]:
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError('Study goal is required')
    if not llm_request_builder_is_configured():
        raise RuntimeError('LLM request builder is not configured')

    base_url = os.getenv(LLM_BASE_URL_ENV)
    model = os.getenv(LLM_MODEL_ENV)
    if not base_url or not model:
        raise RuntimeError('LLM request builder is not configured')
    config_key = _llm_config_key(base_url, model)
    endpoint = _llm_endpoint(base_url)
    post = http_post or _default_http_post
    headers = _llm_headers()
    evidence_query = build_study_wiki_query(goal, seed_spec)
    wiki_evidence = build_wiki_evidence_pack(evidence_query, top_k=5)
    prefer_structured_output = _structured_output_support_status(config_key) != 'unsupported'
    structured_output_used = prefer_structured_output
    if prefer_structured_output:
        try:
            response = post(endpoint, _build_payload(
                goal,
                seed_spec=seed_spec,
                structured_output=True,
                locale=locale,
                wiki_evidence=wiki_evidence,
            ), headers, _llm_timeout())
            if config_key is not None and _extract_structured_payload(response) is not None:
                _STRUCTURED_OUTPUT_SUPPORT[config_key] = True
        except Exception as exc:
            if not _should_retry_without_structured_output(exc):
                raise
            structured_output_used = False
            if config_key is not None:
                _STRUCTURED_OUTPUT_SUPPORT[config_key] = False
            response = post(endpoint, _build_payload(
                goal,
                seed_spec=seed_spec,
                structured_output=False,
                locale=locale,
                wiki_evidence=wiki_evidence,
            ), headers, _llm_timeout())
    else:
        structured_output_used = False
        response = post(endpoint, _build_payload(
            goal,
            seed_spec=seed_spec,
            structured_output=False,
            locale=locale,
            wiki_evidence=wiki_evidence,
        ), headers, _llm_timeout())
    payload = None
    try:
        payload = _parse_llm_response(response)
        study_spec = _study_spec_from_llm(payload, seed_spec)
    except (TypeError, ValueError) as exc:
        rejected_response = _response_evidence(response)
        LOGGER.warning('Planner draft rejected; requesting one corrected response: %s', exc)
        invalid_payload = payload
        if invalid_payload is None:
            try:
                invalid_payload = _extract_response_text(response)
            except ValueError:
                invalid_payload = ''
        response = post(
            endpoint,
            _build_contract_repair_payload(
                goal,
                seed_spec=seed_spec,
                structured_output=structured_output_used,
                locale=locale,
                wiki_evidence=wiki_evidence,
                invalid_payload=invalid_payload,
                validation_error=exc,
            ),
            headers,
            _llm_timeout(),
        )
        try:
            repaired_payload = _parse_llm_response(response)
            study_spec = _study_spec_from_llm(repaired_payload, seed_spec)
        except (TypeError, ValueError) as repair_exc:
            LOGGER.warning('Planner corrected draft rejected; existing draft retained: %s', repair_exc)
            raise PlannerDraftError(
                'Planner returned invalid StudySpec JSON after one contract-repair attempt: {0}'.format(
                    repair_exc
                ),
                {'goal': goal, 'seed_study_spec': seed_spec,
                 'wiki_pages': [page.get('title') for page in wiki_evidence.get('pages', [])],
                 'attempts': [
                     {'error': str(exc), 'response': rejected_response},
                     {'error': str(repair_exc), 'response': _response_evidence(response)},
                 ]},
            ) from repair_exc
    return {
        'study_spec': study_spec.to_dict(),
        'wiki_evidence': wiki_evidence,
    }
