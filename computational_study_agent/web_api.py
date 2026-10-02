from __future__ import annotations

from pyscf_agent.transport.json import api_exception_response, JsonResponse, json_response as _json_response, json_error as _json_error

import json
import logging
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple

from .adaptive import normalize_adaptive_options
from .application import (
    StudyApplicationService,
    StudyApplicationError,
    StudyApplicationValidationError,
    StudyArtifactError,
    get_study_application_service,
)
from .llm_planner import PlannerDraftError, build_study_spec_from_goal_with_evidence, save_planner_diagnostic
from .normalization import normalize_system_type, solver_contract
from .costing import CostApprovalRequired
from .execution_receipts import StudyExecutionInterrupted, StudyExecutionBusy
from .gates.review_actions import StudyReviewActionError
from computational_study_agent.datasets.hamiltonian.contracts import (
    HamiltonianDatasetSpec,
    MolecularGeometry,
)
from .schema import StudyPlan, StudyReport, StudySpec
from .validation import _issue, issue_dicts



MODEL_SOLVER_RECOMMENDATION = (
    'Please choose a model Hamiltonian solver before I build runnable cases. '
    'Recommendation: use fci for small lattices or benchmark-quality exact diagonalization; '
    'use ccsd or ccsd_t (CCSD(T)) for approximate correlated comparisons; '
    'use dmet for supported larger Hubbard embedding studies; '
    'use mp2 only for a quick weak-correlation initial scan.'
)






LOGGER = logging.getLogger(__name__)


def _handle_error(exc):
    return api_exception_response(exc, LOGGER, client_errors=(ValueError, StudyApplicationError, StudyReviewActionError), conflict_errors=(StudyExecutionBusy,))


def _validation_error_message(prefix: str, issues: Any) -> str:
    issue_list = list(issues or [])
    first_error = next((issue for issue in issue_list if getattr(issue, 'severity', '') == 'error'), None)
    if first_error is None:
        return prefix
    path = getattr(first_error, 'path', '') or getattr(first_error, 'code', '')
    detail = getattr(first_error, 'message', '')
    if path and detail:
        return '{0}: {1}: {2}'.format(prefix, path, detail)
    if detail:
        return '{0}: {1}'.format(prefix, detail)
    return prefix


def _direct_casscf_preparation_response(
    study_spec: Dict[str, Any], validation_issues: List[Any],
    wiki_evidence: Dict[str, Any], adaptive_options: Dict[str, Any],
    study_state: Optional[Dict[str, Any]] = None,
    service: Optional[StudyApplicationService] = None,
) -> Optional[JsonResponse]:
    service = service or get_study_application_service()
    prepared = service.prepare_active_space(study_spec, validation_issues,
        options=adaptive_options, study_state=study_state)
    if prepared is None:
        return None
    return _json_response(HTTPStatus.OK, {
        **prepared, 'wiki_evidence': _public_wiki_evidence(wiki_evidence),
    })


def _public_wiki_evidence(wiki_evidence: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'pages': [
            {
                'title': page.get('title'),
                'slug': page.get('slug'),
                'summary': page.get('summary'),
                'score': page.get('score'),
                'reasons': page.get('reasons'),
            }
            for page in (wiki_evidence.get('pages') or [])
        ],
    }


def _decode_json_request(request_body: bytes) -> Tuple[Optional[Dict[str, Any]], Optional[JsonResponse]]:
    try:
        payload = json.loads(request_body.decode('utf-8') or '{}')
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, _json_error(HTTPStatus.BAD_REQUEST, 'Invalid JSON request: {0}'.format(exc))
    if not isinstance(payload, dict):
        return None, _json_error(HTTPStatus.BAD_REQUEST, 'Request body must be a JSON object')
    return payload, None


def _resource_profile_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    value = str(payload.get('resource_profile') or '').strip()
    return None if not value or value.lower() == 'auto' else value


def _study_spec_from_payload(payload: Dict[str, Any]) -> StudySpec:
    study_spec = payload.get('study_spec')
    if isinstance(study_spec, dict):
        return StudySpec.from_dict(study_spec)
    return StudySpec.from_dict(payload)


def _is_hamiltonian_dataset_request(payload: Dict[str, Any]) -> bool:
    template = str(payload.get('planner_template') or '').strip().lower()
    return template == 'hamiltonian_dataset' or isinstance(payload.get('dataset_spec'), dict)


def _hamiltonian_dataset_inputs_from_payload(payload: Dict[str, Any]):
    dataset_payload = payload.get('dataset_spec')
    seed_payload = payload.get('seed_geometries')
    if not isinstance(dataset_payload, dict):
        raise ValueError('dataset_spec must be a JSON object.')
    if not isinstance(seed_payload, list):
        raise ValueError('seed_geometries must be a JSON array.')
    dataset_spec = HamiltonianDatasetSpec.from_dict(dataset_payload)
    seed_geometries = [
        MolecularGeometry.from_dict(item)
        for item in seed_payload
    ]
    return dataset_spec, seed_geometries


def _hamiltonian_dataset_plan_from_payload(
    payload: Dict[str, Any],
    service: StudyApplicationService,
) -> StudyPlan:
    return service.build_hamiltonian_dataset_plan(
        *_hamiltonian_dataset_inputs_from_payload(payload),
        resource_profile=_resource_profile_from_payload(payload),
    )


def _adaptive_options_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    study_spec = payload.get('study_spec')
    if isinstance(study_spec, dict) and normalize_system_type(study_spec.get('system_type')) == 'model_hamiltonian':
        return {}
    options = payload.get('adaptive')
    if isinstance(options, dict):
        return options if options else normalize_adaptive_options({})
    if isinstance(study_spec, dict) and isinstance(study_spec.get('adaptive'), dict):
        return study_spec['adaptive'] if study_spec['adaptive'] else normalize_adaptive_options({})
    mode = payload.get('study_mode', payload.get('mode', payload.get('studyMode')))
    if isinstance(mode, str) and mode.strip().lower() in ('adaptive', 'adaptive_scan'):
        return normalize_adaptive_options({})
    return {}


def _has_model_context(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    input_file = payload.get('base_model_input_file')
    if isinstance(input_file, str) and input_file.strip():
        return True
    base_model_spec = payload.get('base_model_spec')
    return isinstance(base_model_spec, dict) and bool(base_model_spec)


def _model_solver_from_spec(payload: Dict[str, Any]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    base_task = payload.get('base_task')
    if isinstance(base_task, dict):
        solver, _options = solver_contract(base_task.get('solver'))
        if solver:
            return solver.lower()
    base_model_spec = payload.get('base_model_spec')
    if isinstance(base_model_spec, dict):
        solver, _options = solver_contract(base_model_spec.get('solver'))
        if solver:
            return solver.lower()
    sweep = payload.get('sweep')
    if isinstance(sweep, dict) and isinstance(sweep.get('solver'), list) and sweep.get('solver'):
        return 'sweep'
    case_design = payload.get('case_design')
    if isinstance(case_design, dict):
        template = case_design.get('template')
        if isinstance(template, dict):
            request_updates = template.get('request_updates')
            if isinstance(request_updates, dict):
                solver, _options = solver_contract(request_updates.get('solver'))
                if solver:
                    return solver.lower()
        cases = case_design.get('cases')
        if isinstance(cases, list):
            for item in cases:
                if not isinstance(item, dict):
                    continue
                request_updates = item.get('request_updates')
                if isinstance(request_updates, dict):
                    solver, _options = solver_contract(request_updates.get('solver'))
                    if solver:
                        return solver.lower()
                variables = item.get('variables')
                if isinstance(variables, dict):
                    solver, _options = solver_contract(variables.get('solver'))
                    if solver:
                        return solver.lower()
    return None


def _requires_model_solver_choice(
    study_spec: Dict[str, Any],
    seed_spec: Optional[Dict[str, Any]],
) -> bool:
    if not isinstance(study_spec, dict) or study_spec.get('system_type') != 'model_hamiltonian':
        return False
    if not _has_model_context(study_spec) and not _has_model_context(seed_spec or {}):
        return False
    return _model_solver_from_spec(study_spec) is None


def _model_solver_choice_response(study_spec: Dict[str, Any], wiki_evidence: Dict[str, Any]) -> JsonResponse:
    issue = _issue(
        'error',
        'missing_model_solver',
        MODEL_SOLVER_RECOMMENDATION,
        'base_task.solver',
    )
    comparison = study_spec.setdefault('comparison', {})
    notes = comparison.get('notes')
    if isinstance(notes, list):
        if MODEL_SOLVER_RECOMMENDATION not in notes:
            notes.append(MODEL_SOLVER_RECOMMENDATION)
    elif isinstance(notes, str) and notes.strip():
        comparison['notes'] = [notes, MODEL_SOLVER_RECOMMENDATION]
    else:
        comparison['notes'] = [MODEL_SOLVER_RECOMMENDATION]
    return _json_response(HTTPStatus.OK, {
        'status': 'needs_input',
        'message': MODEL_SOLVER_RECOMMENDATION,
        'study_spec': study_spec,
        'validation_issues': issue_dicts([issue]),
        'wiki_evidence': _public_wiki_evidence(wiki_evidence),
    })


def _inherit_seed_model_context(draft: Dict[str, Any], seed_spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(draft, dict) or not isinstance(seed_spec, dict):
        return draft
    seed_system_type = str(seed_spec.get('system_type') or '').strip().lower().replace('-', '_')
    if seed_system_type != 'model_hamiltonian' or not _has_model_context(seed_spec):
        return draft
    draft_system_type = str(draft.get('system_type') or '').strip().lower().replace('-', '_')
    if draft_system_type and draft_system_type != 'model_hamiltonian':
        return draft

    next_draft = json.loads(json.dumps(draft, ensure_ascii=False))
    next_draft['system_type'] = 'model_hamiltonian'
    if not _has_model_context(next_draft):
        seed_input_file = seed_spec.get('base_model_input_file')
        if isinstance(seed_input_file, str) and seed_input_file.strip():
            next_draft['base_model_input_file'] = seed_input_file.strip()
            next_draft.pop('base_model_spec', None)
        else:
            next_draft['base_model_spec'] = json.loads(json.dumps(seed_spec.get('base_model_spec'), ensure_ascii=False))

    seed_base_task = seed_spec.get('base_task')
    draft_base_task = next_draft.get('base_task')
    if isinstance(seed_base_task, dict):
        if not isinstance(draft_base_task, dict):
            next_draft['base_task'] = json.loads(json.dumps(seed_base_task, ensure_ascii=False))
        elif 'solver' not in draft_base_task:
            seed_solver, _seed_solver_options = solver_contract(seed_base_task.get('solver'))
            if seed_solver:
                draft_base_task['solver'] = json.loads(json.dumps(seed_base_task['solver'], ensure_ascii=False))
    return next_draft


def handle_study_plan_request(request_body: bytes) -> JsonResponse:
    payload, error_response = _decode_json_request(request_body)
    if error_response is not None:
        return error_response
    assert payload is not None
    service: Optional[StudyApplicationService] = None
    study_spec: Optional[StudySpec] = None
    try:
        service = get_study_application_service(payload.get('execution_target'))
        if _is_hamiltonian_dataset_request(payload):
            plan = _hamiltonian_dataset_plan_from_payload(payload, service)
            validation_issues = []
        else:
            study_spec = _study_spec_from_payload(payload)
            result = service.prepare_plan(
                study_spec,
                resource_profile=_resource_profile_from_payload(payload),
            )
            plan = result.plan
            validation_issues = list(result.validation_issues)
    except StudyApplicationValidationError as exc:
        validation_issues = list(exc.issues)
        if study_spec is not None:
            active_space_preparation = _direct_casscf_preparation_response(
                study_spec.to_dict(),
                validation_issues,
                {},
                _adaptive_options_from_payload(payload),
                service=service,
            )
            if active_space_preparation is not None:
                return active_space_preparation
        return _json_response(HTTPStatus.BAD_REQUEST, {
            'error': _validation_error_message(str(exc), validation_issues),
            'validation_issues': issue_dicts(validation_issues),
        })
    except CostApprovalRequired as exc:
        return _json_response(HTTPStatus.CONFLICT, {
            'error': str(exc),
            'cost_review': exc.estimate,
        })
    except Exception as exc:
        return _handle_error(exc)
    return _json_response(HTTPStatus.OK, {
        'plan': plan.to_dict(),
        'validation_issues': issue_dicts(validation_issues),
    })


def handle_study_run_request(request_body: bytes) -> JsonResponse:
    payload, error_response = _decode_json_request(request_body)
    if error_response is not None:
        return error_response
    assert payload is not None
    locale = str(payload.get('locale') or 'en')
    work_dir = payload.get('work_dir')
    postprocess = bool(payload.get('postprocess'))
    postprocess_specs = payload.get('postprocess_specs')
    resume = payload.get('resume', True)
    rerun_case_ids = payload.get('rerun_case_ids')
    rerun_statuses = payload.get('rerun_statuses')
    max_case_attempts = payload.get('max_case_attempts', 2)
    study_report = payload.get('study_report')
    if postprocess_specs is not None and not isinstance(postprocess_specs, list):
        return _json_error(HTTPStatus.BAD_REQUEST, 'postprocess_specs must be a list when provided')
    if not isinstance(resume, bool):
        return _json_error(HTTPStatus.BAD_REQUEST, 'resume must be a boolean when provided')
    if rerun_case_ids is not None and not isinstance(rerun_case_ids, list):
        return _json_error(HTTPStatus.BAD_REQUEST, 'rerun_case_ids must be a list when provided')
    if rerun_statuses is not None and not isinstance(rerun_statuses, list):
        return _json_error(HTTPStatus.BAD_REQUEST, 'rerun_statuses must be a list when provided')
    if not isinstance(max_case_attempts, int) or isinstance(max_case_attempts, bool) or max_case_attempts < 1:
        return _json_error(HTTPStatus.BAD_REQUEST, 'max_case_attempts must be a positive integer when provided')
    if study_report is not None and not isinstance(study_report, dict):
        return _json_error(HTTPStatus.BAD_REQUEST, 'study_report must be an object when provided')
    try:
        service = get_study_application_service(payload.get('execution_target'))
    except ValueError as exc:
        return _json_error(HTTPStatus.BAD_REQUEST, str(exc))
    try:
        plan_payload = payload.get('plan')
        plan_or_spec = plan_payload if isinstance(plan_payload, dict) else _study_spec_from_payload(payload)
        report = service.run_study(
            plan_or_spec,
            work_dir=work_dir,
            locale=locale,
            postprocess=postprocess,
            postprocess_specs=postprocess_specs,
            resume=resume,
            rerun_case_ids=rerun_case_ids,
            rerun_statuses=rerun_statuses,
            max_case_attempts=max_case_attempts,
            resource_profile=_resource_profile_from_payload(payload),
            **({'study_report': study_report} if isinstance(study_report, dict) else {}),
        )
    except StudyExecutionInterrupted as exc:
        interrupted_study_id = str(exc.receipt.get('study_id') or '').strip()
        execution = service.inspect_execution(
            interrupted_study_id,
            work_dir=work_dir,
        )
        return _json_response(HTTPStatus.ACCEPTED, {
            'status': 'execution_interrupted',
            'message': str(exc),
            'execution': execution,
        })
    except StudyApplicationValidationError as exc:
        validation_issues = list(exc.issues)
        return _json_response(HTTPStatus.BAD_REQUEST, {
            'error': _validation_error_message(str(exc), validation_issues),
            'validation_issues': issue_dicts(validation_issues),
        })
    except CostApprovalRequired as exc:
        return _json_response(HTTPStatus.CONFLICT, {
            'error': str(exc),
            'cost_review': exc.estimate,
        })
    except Exception as exc:
        return _handle_error(exc)
    return _json_response(HTTPStatus.OK, {
        'report': report.to_dict() if isinstance(report, StudyReport) else report,
    })


def handle_study_adaptive_plan_request(request_body: bytes) -> JsonResponse:
    payload, error_response = _decode_json_request(request_body)
    if error_response is not None:
        return error_response
    assert payload is not None
    try:
        study_spec = _study_spec_from_payload(payload)
        service = get_study_application_service(payload.get('execution_target'))
        adaptive_payload = service.build_adaptive_plan(
            study_spec,
            options=_adaptive_options_from_payload(payload),
            resource_profile=_resource_profile_from_payload(payload),
        )
    except StudyApplicationValidationError as exc:
        return _json_response(HTTPStatus.BAD_REQUEST, {
            'error': _validation_error_message(str(exc), exc.issues),
            'validation_issues': issue_dicts(exc.issues),
        })
    except Exception as exc:
        return _handle_error(exc)
    return _json_response(HTTPStatus.OK, adaptive_payload)


def handle_example_study_request(request_body: bytes, *, action: str) -> JsonResponse:
    payload, error_response = _decode_json_request(request_body)
    if error_response is not None:
        return error_response
    try:
        service = get_study_application_service('local')
        if action == 'list':
            return _json_response(HTTPStatus.OK, {'examples': service.list_examples()})
        study_id = payload.get('study_id')
        if not isinstance(study_id, str):
            raise ValueError('study_id must be a string')
        return _json_response(HTTPStatus.OK, service.import_example(
            study_id, work_dir=payload.get('work_dir')))
    except Exception as exc:
        return _handle_error(exc)


def handle_saved_study_request(request_body: bytes, *, action: str) -> JsonResponse:
    """HTTP transport for the same saved-Study operations exposed by MCP."""
    payload, error_response = _decode_json_request(request_body)
    if error_response is not None:
        return error_response
    try:
        service = get_study_application_service(payload.get('execution_target'))
        work_dir = payload.get('work_dir')
        if action == 'list':
            return _json_response(HTTPStatus.OK, {'studies': service.list_studies(work_dir=work_dir)})
        if action == 'prepare':
            if _is_hamiltonian_dataset_request(payload):
                prepared = service.prepare_dataset(*_hamiltonian_dataset_inputs_from_payload(payload),
                    work_dir=work_dir, resource_profile=_resource_profile_from_payload(payload))
            else:
                prepared = service.prepare_study(_study_spec_from_payload(payload), work_dir=work_dir,
                    options=_adaptive_options_from_payload(payload),
                    resource_profile=_resource_profile_from_payload(payload))
            # The page uses the same projection for creation and later reopening.
            result = service.open_study(prepared['study_id'], work_dir=work_dir)
            return _json_response(HTTPStatus.CREATED, {**result,
                'preparation_status': prepared.get('status', 'prepared'),
                'message': prepared.get('message'), 'validation_issues': prepared.get('validation_issues', [])})
        study_id = payload.get('study_id')
        if not isinstance(study_id, str) or not study_id.strip():
            raise ValueError('study_id must be a non-empty string')
        if action == 'open':
            result = service.open_study(study_id, work_dir=work_dir)
        elif action == 'retry-context':
            result = service.retry_context(study_id, work_dir=work_dir)
        elif action == 'dmet-branches-analyze':
            result = service.analyze_dmet_branches(study_id, policy=payload.get('policy'), work_dir=work_dir)
        elif action == 'dmet-continuation-prepare':
            result = service.prepare_dmet_continuation(study_id, work_dir=work_dir,
                **{k: payload[k] for k in ('case_ids', 'branches', 'mode', 'policy', 'mixing', 'max_iterations') if k in payload})
        elif action == 'dmet-continuation-start':
            result = service.start_dmet_continuation(study_id, payload.get('action_id'), work_dir=work_dir,
                approve_cost=payload.get('approve_cost') is True, locale=str(payload.get('locale') or 'en'))
        elif action == 'retry-prepare':
            retry_action = payload.get('retry_action')
            if not isinstance(retry_action, dict) or retry_action.get('study_id') != study_id:
                raise ValueError('Retry action must belong to the selected Study')
            result = service.prepare_retry(retry_action, work_dir=work_dir)
        elif action == 'retry-start':
            result = service.start_retry(study_id, payload.get('action_id'), work_dir=work_dir,
                approve_cost=payload.get('approve_cost') is True, locale=str(payload.get('locale') or 'en'))
        elif action == 'stop':
            result = service.stop_study(study_id, work_dir=work_dir)
        elif action == 'start':
            result = service.start_study(study_id, work_dir=work_dir, locale=str(payload.get('locale') or 'en'))
        else:
            raise ValueError('Unknown saved Study action')
    except FileNotFoundError as exc:
        return _json_error(HTTPStatus.NOT_FOUND, str(exc))
    except CostApprovalRequired as exc:
        return _json_response(HTTPStatus.CONFLICT, {'error': str(exc), 'cost_review': exc.estimate})
    except StudyExecutionBusy as exc:
        return _json_error(HTTPStatus.CONFLICT, str(exc))
    except StudyApplicationValidationError as exc:
        return _json_response(HTTPStatus.BAD_REQUEST, {
            'error': _validation_error_message(str(exc), exc.issues),
            'validation_issues': issue_dicts(exc.issues),
        })
    except (ValueError, StudyApplicationError) as exc:
        return _json_error(HTTPStatus.BAD_REQUEST, str(exc))
    except Exception as exc:
        return _handle_error(exc)
    return _json_response(HTTPStatus.ACCEPTED if action == 'start' else HTTPStatus.OK, result)


def handle_study_review_action_request(request_body: bytes) -> JsonResponse:
    payload, error_response = _decode_json_request(request_body)
    if error_response is not None:
        return error_response
    assert payload is not None
    action_id = payload.get('action_id')
    if not isinstance(action_id, str) or not action_id.strip():
        return _json_error(HTTPStatus.BAD_REQUEST, 'action_id must be a non-empty string')
    study_state = payload.get('study_state')
    plan = payload.get('plan')
    case_ids = payload.get('case_ids')
    approval_token = payload.get('approval_token')
    plan_kind = payload.get('plan_kind')
    if study_state is not None and not isinstance(study_state, dict):
        return _json_error(HTTPStatus.BAD_REQUEST, 'study_state must be an object when provided')
    if plan is not None and not isinstance(plan, dict):
        return _json_error(HTTPStatus.BAD_REQUEST, 'plan must be an object when provided')
    if case_ids is not None and not isinstance(case_ids, list):
        return _json_error(HTTPStatus.BAD_REQUEST, 'case_ids must be a list when provided')
    if approval_token is not None and not isinstance(approval_token, str):
        return _json_error(HTTPStatus.BAD_REQUEST, 'approval_token must be a string when provided')
    if plan_kind is not None and not isinstance(plan_kind, str):
        return _json_error(HTTPStatus.BAD_REQUEST, 'plan_kind must be a string when provided')
    try:
        service = get_study_application_service(payload.get('execution_target'))
        review_kwargs = dict(case_ids=case_ids, approval_token=approval_token, plan_kind=plan_kind)
        if payload.get('study_id'):
            result = service.review_study(payload['study_id'], action_id,
                                          work_dir=payload.get('work_dir'), **review_kwargs)
        else:
            result = service.apply_review_action(action_id, study_state=study_state, plan=plan, **review_kwargs)
    except StudyExecutionBusy as exc:
        return _json_error(HTTPStatus.CONFLICT, str(exc))
    except (ValueError, StudyReviewActionError) as exc:
        return _json_error(HTTPStatus.BAD_REQUEST, str(exc))
    except Exception as exc:
        return _handle_error(exc)
    return _json_response(HTTPStatus.OK, result)


def handle_study_adaptive_run_request(request_body: bytes) -> JsonResponse:
    payload, error_response = _decode_json_request(request_body)
    if error_response is not None:
        return error_response
    assert payload is not None
    locale = str(payload.get('locale') or 'en')
    work_dir = payload.get('work_dir')
    resume_study_id = payload.get('resume_study_id')
    requested_study_id = payload.get('requested_study_id')
    if resume_study_id is not None and (not isinstance(resume_study_id, str) or not resume_study_id.strip()):
        return _json_error(HTTPStatus.BAD_REQUEST, 'resume_study_id must be a non-empty string when provided')
    if requested_study_id is not None and (not isinstance(requested_study_id, str) or not requested_study_id.strip()):
        return _json_error(HTTPStatus.BAD_REQUEST, 'requested_study_id must be a non-empty string when provided')
    try:
        service = get_study_application_service(payload.get('execution_target'))
    except ValueError as exc:
        return _json_error(HTTPStatus.BAD_REQUEST, str(exc))
    try:
        study_spec = _study_spec_from_payload(payload)
        report = service.run_adaptive_study(
            study_spec,
            options=_adaptive_options_from_payload(payload),
            work_dir=work_dir,
            locale=locale,
            resume_study_id=resume_study_id,
            requested_study_id=requested_study_id,
            resource_profile=_resource_profile_from_payload(payload),
            lifecycle=payload.get('lifecycle'),
        )
    except StudyExecutionInterrupted as exc:
        adaptive_study_id = str(
            resume_study_id or requested_study_id or ''
        ).strip()
        execution = service.inspect_execution(
            adaptive_study_id,
            work_dir=work_dir,
        ) if adaptive_study_id else {
            'status': 'connection_interrupted',
            'receipt': exc.receipt,
            'can_collect': False,
        }
        return _json_response(HTTPStatus.ACCEPTED, {
            'status': 'execution_interrupted',
            'message': str(exc),
            'execution': execution,
        })
    except StudyApplicationValidationError as exc:
        return _json_response(HTTPStatus.BAD_REQUEST, {
            'error': _validation_error_message(str(exc), exc.issues),
            'validation_issues': issue_dicts(exc.issues),
        })
    except CostApprovalRequired as exc:
        return _json_response(HTTPStatus.CONFLICT, {
            'error': str(exc),
            'cost_review': exc.estimate,
        })
    except Exception as exc:
        return _handle_error(exc)
    return _json_response(HTTPStatus.OK, {'report': report})


def handle_study_execution_status_request(request_body: bytes) -> JsonResponse:
    payload, error_response = _decode_json_request(request_body)
    if error_response is not None:
        return error_response
    assert payload is not None
    study_id = payload.get('study_id')
    if not isinstance(study_id, str) or not study_id.strip():
        return _json_error(HTTPStatus.BAD_REQUEST, 'study_id must be a non-empty string')
    try:
        service = get_study_application_service(payload.get('execution_target'))
        execution = service.inspect_execution(
            study_id,
            work_dir=payload.get('work_dir'),
        )
    except (ValueError, StudyApplicationError) as exc:
        return _json_error(HTTPStatus.BAD_REQUEST, str(exc))
    except Exception as exc:
        return _handle_error(exc)
    return _json_response(HTTPStatus.OK, {'execution': execution})


def handle_study_execution_collect_request(request_body: bytes) -> JsonResponse:
    payload, error_response = _decode_json_request(request_body)
    if error_response is not None:
        return error_response
    assert payload is not None
    study_id = str(payload.get('study_id') or '').strip()
    if not study_id:
        return _json_error(HTTPStatus.BAD_REQUEST, 'study_id must be a non-empty string')
    mode = str(payload.get('mode') or 'static').strip().lower()
    work_dir = payload.get('work_dir')
    locale = str(payload.get('locale') or 'en')
    try:
        service = get_study_application_service(payload.get('execution_target'))
        execution = service.inspect_execution(study_id, work_dir=work_dir)
        if not execution.get('can_collect'):
            return _json_response(HTTPStatus.ACCEPTED, {
                'status': execution.get('status') or 'running',
                'message': (
                    'Submission outcome is unknown. Reconcile the existing job before starting another execution.'
                    if execution.get('status') == 'submission_unknown'
                    else 'Remote results are not all available yet.'
                ),
                'execution': execution,
            })
        if payload.get('saved_study'):
            report = service.collect_saved_study(study_id, work_dir=work_dir, locale=locale)
        elif mode == 'adaptive':
            study_spec = _study_spec_from_payload(payload)
            report = service.collect_adaptive_study(
                study_spec,
                options=_adaptive_options_from_payload(payload),
                work_dir=work_dir,
                locale=locale,
                study_id=study_id,
                resource_profile=_resource_profile_from_payload(payload),
                lifecycle=payload.get('lifecycle'),
            )
        else:
            report = service.collect_study(
                study_id,
                work_dir=work_dir,
                locale=locale,
            ).to_dict()
    except StudyExecutionInterrupted as exc:
        execution = service.inspect_execution(study_id, work_dir=work_dir)
        return _json_response(HTTPStatus.ACCEPTED, {
            'status': 'execution_interrupted',
            'message': str(exc),
            'execution': execution,
        })
    except StudyApplicationValidationError as exc:
        return _json_response(HTTPStatus.BAD_REQUEST, {
            'error': _validation_error_message(str(exc), exc.issues),
            'validation_issues': issue_dicts(exc.issues),
        })
    except Exception as exc:
        return _handle_error(exc)
    return _json_response(HTTPStatus.OK, {
        'status': 'collected',
        'report': report,
    })


def handle_study_postprocess_request(request_body: bytes) -> JsonResponse:
    payload, error_response = _decode_json_request(request_body)
    if error_response is not None:
        return error_response
    assert payload is not None

    report = payload.get('report')
    if not isinstance(report, dict):
        return _json_error(HTTPStatus.BAD_REQUEST, 'report must be an object')
    specs = payload.get('plot_specs')
    if specs is not None and not isinstance(specs, list):
        return _json_error(HTTPStatus.BAD_REQUEST, 'plot_specs must be a list when provided')
    actions = payload.get('actions')
    if actions is not None and not isinstance(actions, list):
        return _json_error(HTTPStatus.BAD_REQUEST, 'actions must be a list when provided')
    try:
        result = get_study_application_service(
            payload.get('execution_target')
        ).run_postprocessing(
            report,
            specs=specs,
            actions=actions,
        )
    except (TypeError, ValueError) as exc:
        return _json_error(HTTPStatus.BAD_REQUEST, 'Study postprocessing failed: {0}'.format(exc))
    except Exception as exc:
        return _handle_error(exc)
    return _json_response(HTTPStatus.OK, {'postprocessing': result})


def handle_study_postprocess_suggestions_request(request_body: bytes) -> JsonResponse:
    payload, error_response = _decode_json_request(request_body)
    if error_response is not None:
        return error_response
    assert payload is not None

    report = payload.get('report')
    if not isinstance(report, dict):
        return _json_error(HTTPStatus.BAD_REQUEST, 'report must be an object')
    if not isinstance(payload.get('include_suggestions', True), bool):
        return _json_error(HTTPStatus.BAD_REQUEST, 'include_suggestions must be a boolean')
    try:
        service = get_study_application_service()
        context = service.postprocessing_context(report)
        specs = service.suggest_postprocessing(report) if payload.get('include_suggestions', True) else []
    except Exception as exc:
        return _handle_error(exc)
    return _json_response(HTTPStatus.OK, {'plot_specs': specs, 'context': context})


def handle_study_artifact_request(request_body: bytes) -> JsonResponse:
    payload, error_response = _decode_json_request(request_body)
    if error_response is not None:
        return error_response
    assert payload is not None

    report = payload.get('report')
    artifact_path = payload.get('path')
    if not isinstance(report, dict):
        return _json_error(HTTPStatus.BAD_REQUEST, 'report must be an object')
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        return _json_error(HTTPStatus.BAD_REQUEST, 'path is required')
    try:
        artifact = get_study_application_service().read_artifact(
            report,
            artifact_path,
            allowed_suffixes=('.png',),
        )
    except StudyArtifactError as exc:
        status_by_code = {
            'outside_work_dir': HTTPStatus.FORBIDDEN,
            'not_found': HTTPStatus.NOT_FOUND,
            'unsupported_type': HTTPStatus.BAD_REQUEST,
            'path_required': HTTPStatus.BAD_REQUEST,
            'work_dir_required': HTTPStatus.BAD_REQUEST,
        }
        return _json_error(status_by_code.get(exc.code, HTTPStatus.BAD_REQUEST), str(exc))
    return HTTPStatus.OK, {'Content-Type': artifact.media_type}, artifact.content


def handle_study_result_analysis_request(request_body: bytes, *, llm_request_builder: Any = None) -> JsonResponse:
    payload, error_response = _decode_json_request(request_body)
    if error_response is not None:
        return error_response
    assert payload is not None

    locale = str(payload.get('locale') or 'en')
    report = payload.get('report')
    if not payload.get('study_id') and not isinstance(report, dict):
        return _json_error(HTTPStatus.BAD_REQUEST, 'report must be an object')
    if llm_request_builder is None or not hasattr(llm_request_builder, 'build_result_analysis'):
        return _json_error(HTTPStatus.BAD_REQUEST, 'LLM result analysis is unavailable')

    try:
        service = get_study_application_service(payload.get('execution_target'))
    except ValueError as exc:
        return _json_error(HTTPStatus.BAD_REQUEST, str(exc))
    try:
        if payload.get('study_id'):
            result = service.analyze_study(
                payload['study_id'], work_dir=payload.get('work_dir'), locale=locale,
                llm_request_builder=llm_request_builder,
                resource_profile=_resource_profile_from_payload(payload),
            )
            return _json_response(HTTPStatus.OK, result)
        result = service.analyze_results(
            report,
            plan=payload.get('plan'),
            llm_request_builder=llm_request_builder,
            locale=locale,
            path_restart_approval=(
                payload.get('path_restart_approval')
                if isinstance(payload.get('path_restart_approval'), dict)
                else None
            ),
            mps_continuation_approval=(
                payload.get('mps_continuation_approval')
                if isinstance(payload.get('mps_continuation_approval'), dict)
                else None
            ),
            resource_profile=_resource_profile_from_payload(payload),
        )
    except FileNotFoundError as exc:
        return _json_error(HTTPStatus.NOT_FOUND, str(exc))
    except StudyExecutionBusy as exc:
        return _json_error(HTTPStatus.CONFLICT, str(exc))
    except (ValueError, StudyApplicationError) as exc:
        return _json_error(HTTPStatus.BAD_REQUEST, str(exc))
    except Exception as exc:
        return _handle_error(exc)
    return _json_response(HTTPStatus.OK, result)


def handle_study_llm_draft_request(request_body: bytes) -> JsonResponse:
    payload, error_response = _decode_json_request(request_body)
    if error_response is not None:
        return error_response
    assert payload is not None
    goal = payload.get('goal')
    seed_spec = payload.get('study_spec') if isinstance(payload.get('study_spec'), dict) else None
    locale = str(payload.get('locale') or 'en')
    validation_issues = []
    try:
        if payload.get('study_id'):
            from .planner_intent import draft_saved_action
            intent = draft_saved_action(get_study_application_service(payload.get('execution_target')),
                payload['study_id'], str(goal or ''), work_dir=payload.get('work_dir'),
                selected_case_ids=payload.get('selected_case_ids'))
            if intent['intent'] in ('retry_cases', 'analyze') or intent.get('needs_input'):
                return _json_response(HTTPStatus.OK, intent)
        llm_result = build_study_spec_from_goal_with_evidence(str(goal or ''), seed_spec=seed_spec, locale=locale)
        adaptive_options = _adaptive_options_from_payload(payload)
        draft_spec = _inherit_seed_model_context(llm_result.get('study_spec') or {}, seed_spec)
        # The planner returns a complete StudySpec. Transport adapters must not
        # reinterpret free text or rebuild it from an older seed.
        study_spec = StudySpec.from_dict(draft_spec).to_dict()
        wiki_evidence = llm_result.get('wiki_evidence') or {}
        if _requires_model_solver_choice(study_spec, seed_spec):
            return _model_solver_choice_response(study_spec, wiki_evidence)
        try:
            plan_result = get_study_application_service().prepare_plan(study_spec)
            plan = plan_result.plan
            validation_issues = list(plan_result.validation_issues)
        except StudyApplicationValidationError as validation_exc:
            validation_issues = list(validation_exc.issues)
            active_space_review = _direct_casscf_preparation_response(
                study_spec,
                validation_issues,
                wiki_evidence,
                adaptive_options,
                payload.get('study_state') if isinstance(payload.get('study_state'), dict) else None,
            )
            if active_space_review is not None:
                return active_space_review
            return _json_response(HTTPStatus.OK, {
                'status': 'needs_input',
                'message': _validation_error_message('Planner draft plan validation failed', validation_issues),
                'study_spec': study_spec,
                'validation_issues': issue_dicts(validation_issues),
                'wiki_evidence': _public_wiki_evidence(wiki_evidence),
            })
    except PlannerDraftError as exc:
        failure = {
            'error': 'Planner draft failed: {0}'.format(exc),
            'validation_issues': issue_dicts(validation_issues),
        }
        try:
            failure['diagnostic_file'] = save_planner_diagnostic(exc, payload.get('work_dir'))
            failure['error'] += '\nDiagnostic: ' + failure['diagnostic_file']
        except OSError:
            failure['error'] += '\nCould not save the Planner diagnostic file.'
        return _json_response(HTTPStatus.BAD_REQUEST, failure)
    except Exception as exc:
        return _handle_error(exc)
    return _json_response(HTTPStatus.OK, {
        'study_spec': study_spec,
        'plan': plan.to_dict(),
        'validation_issues': issue_dicts(validation_issues),
        'wiki_evidence': _public_wiki_evidence(wiki_evidence),
    })
