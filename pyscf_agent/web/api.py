from __future__ import annotations

from pyscf_agent.transport.json import api_exception_response, JsonResponse, json_response as _json_response, json_error as _json_error

import json
import logging
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pyscf_agent.application import CalculationApplicationService, CalculationFeatureUnavailableError, ExecutionTargetRegistry
from pyscf_agent.executors import JobHandle, JobNotReadyError, JobState, JobStatus, LocalExecutor, TaskExecutor
from pyscf_agent.pyscf_i18n import normalize_locale




LOGGER = logging.getLogger(__name__)


def _handle_error(exc):
    return api_exception_response(exc, LOGGER, client_errors=(ValueError,))


def build_agent_request(payload: Dict[str, Any]) -> str:
    prepared_request = payload.get('prepared_request')
    if isinstance(prepared_request, str) and prepared_request.strip():
        return prepared_request.strip()
    request_text = payload.get('request')
    task_spec_payload = payload.get('task_spec')
    task_spec = task_spec_payload if isinstance(task_spec_payload, dict) else {}
    request_note = request_text.strip() if isinstance(request_text, str) else ''
    structured_request = {}
    for key in (
        'task_type',
        'atom',
        'basis',
        'method',
        'xc',
        'job',
        'charge',
        'spin',
        'restricted',
        'model_hamiltonian_input_file',
    ):
        value = task_spec.get(key)
        if value is None and key in payload:
            value = payload.get(key)
        if isinstance(value, str):
            value = value.strip()
        if value is not None and value != '':
            structured_request[key] = value
    has_model_input_file = bool(structured_request.get('model_hamiltonian_input_file'))
    model_hamiltonian = task_spec.get('model_hamiltonian')
    if not isinstance(model_hamiltonian, dict) and isinstance(payload.get('model_hamiltonian'), dict):
        model_hamiltonian = payload['model_hamiltonian']
    if isinstance(model_hamiltonian, dict):
        model_input_file = model_hamiltonian.get('input_file')
        if not has_model_input_file and isinstance(model_input_file, str) and model_input_file.strip():
            structured_request['model_hamiltonian_input_file'] = model_input_file.strip()
            has_model_input_file = True
        if not has_model_input_file:
            structured_request['model_hamiltonian'] = model_hamiltonian
    periodic = task_spec.get('periodic')
    if not isinstance(periodic, dict) and isinstance(payload.get('periodic'), dict):
        periodic = payload['periodic']
    if isinstance(periodic, dict):
        structured_request['periodic'] = periodic
    for nested_key in ('orbital_processing', 'active_space'):
        nested_value = task_spec.get(nested_key)
        if nested_value is None and isinstance(payload.get(nested_key), dict):
            nested_value = payload.get(nested_key)
        if isinstance(nested_value, dict):
            structured_request[nested_key] = nested_value
    solver = task_spec.get('solver')
    if solver is None and 'solver' in payload:
        solver = payload.get('solver')
    if isinstance(solver, (str, dict)):
        structured_request['solver'] = solver
    outputs = task_spec.get('outputs')
    if not isinstance(outputs, list) and isinstance(payload.get('outputs'), list):
        outputs = payload.get('outputs')
    if isinstance(outputs, list):
        normalized_outputs = []
        for item in outputs:
            normalized_item = str(item).strip()
            if normalized_item:
                normalized_outputs.append(normalized_item)
        if normalized_outputs:
            structured_request['outputs'] = normalized_outputs
    if request_note:
        structured_request['request'] = request_note
    if structured_request:
        return json.dumps(structured_request, ensure_ascii=False)
    return request_note


def _model_hamiltonian_input_file_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    candidates = [payload.get('model_hamiltonian_input_file')]
    task_spec = payload.get('task_spec')
    if isinstance(task_spec, dict):
        candidates.append(task_spec.get('model_hamiltonian_input_file'))
        model_hamiltonian = task_spec.get('model_hamiltonian')
        if isinstance(model_hamiltonian, dict):
            candidates.append(model_hamiltonian.get('input_file'))
    prepared_request = payload.get('prepared_request')
    if isinstance(prepared_request, str) and prepared_request.strip():
        try:
            prepared_payload = json.loads(prepared_request)
        except json.JSONDecodeError:
            prepared_payload = None
        if isinstance(prepared_payload, dict):
            candidates.append(prepared_payload.get('model_hamiltonian_input_file'))
            model_hamiltonian = prepared_payload.get('model_hamiltonian')
            if isinstance(model_hamiltonian, dict):
                candidates.append(model_hamiltonian.get('input_file'))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _infer_run_context_from_model_input_file(input_file: str) -> Tuple[Optional[str], Optional[str]]:
    path = Path(input_file).expanduser()
    parent = path.parent
    if not parent.name:
        return None, None
    return str(parent.parent), parent.name




def handle_model_hamiltonian_preview_request(request_body: bytes) -> JsonResponse:
    payload, error_response = decode_json_request(request_body)
    if error_response is not None:
        return error_response
    assert payload is not None

    input_file = payload.get('model_hamiltonian_input_file')
    if not isinstance(input_file, str) or not input_file.strip():
        return _json_error(HTTPStatus.BAD_REQUEST, 'model_hamiltonian_input_file is required')
    try:
        preview = CalculationApplicationService().preview_model_hamiltonian(
            input_file,
            solver=payload.get('solver'),
        )
    except Exception as exc:
        return _handle_error(exc)
    return _json_response(HTTPStatus.OK, {'preview': preview})


def handle_periodic_structure_preview_request(request_body: bytes) -> JsonResponse:
    try:
        payload = json.loads(request_body.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    source_text = payload.get('structure_text')
    structure_format = payload.get('structure_format') or 'poscar'
    try:
        preview = CalculationApplicationService().preview_periodic_structure(
            source_text,
            structure_format,
            seekpath_symprec=payload.get('seekpath_symprec', 1e-5),
            seekpath_reference_distance=payload.get('seekpath_reference_distance'),
        )
    except ValueError as exc:
        return _json_error(HTTPStatus.BAD_REQUEST, 'Periodic structure preview failed: {0}'.format(exc))
    body = json.dumps({'status': 'ok', 'preview': preview}, ensure_ascii=False).encode('utf-8')
    return HTTPStatus.OK, {'Content-Type': 'application/json; charset=utf-8'}, body


def decode_json_request(request_body: bytes) -> Tuple[Optional[Dict[str, Any]], Optional[JsonResponse]]:
    try:
        payload = json.loads(request_body.decode('utf-8') or '{}')
    except json.JSONDecodeError:
        return None, _json_error(HTTPStatus.BAD_REQUEST, 'Invalid JSON body')
    if not isinstance(payload, dict):
        return None, _json_error(HTTPStatus.BAD_REQUEST, 'JSON body must be an object')
    return payload, None


def _resource_profile_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    value = str(payload.get('resource_profile') or '').strip()
    return None if not value or value.lower() == 'auto' else value


def handle_capabilities_request() -> JsonResponse:
    capabilities = CalculationApplicationService().capabilities()
    return _json_response(HTTPStatus.OK, {'capabilities': capabilities})


def handle_execution_targets_request(execution_targets: ExecutionTargetRegistry) -> JsonResponse:
    return _json_response(HTTPStatus.OK, execution_targets.public_dict())


def handle_resource_profiles_request(
    execution_targets: ExecutionTargetRegistry,
    execution_target: Optional[str],
) -> JsonResponse:
    try:
        payload = execution_targets.public_resource_profiles(execution_target)
    except ValueError as exc:
        return _json_error(HTTPStatus.BAD_REQUEST, str(exc))
    except Exception as exc:
        return _handle_error(exc)
    return _json_response(HTTPStatus.OK, payload)


def handle_prepare_request(request_body: bytes, *, llm_request_builder: Any) -> JsonResponse:
    payload, error_response = decode_json_request(request_body)
    if error_response is not None:
        return error_response
    assert payload is not None

    locale = normalize_locale(payload.get('locale'))
    try:
        prepared = CalculationApplicationService(
            llm_request_builder=llm_request_builder,
        ).prepare_request(
            messages=payload.get('messages'),
            task_spec=payload.get('task_spec'),
            request=payload.get('request'),
            locale=locale,
            lifecycle=payload.get('lifecycle'),
        )
    except Exception as exc:
        return _handle_error(exc)
    status_code = HTTPStatus.OK
    if prepared.get('status') == 'unavailable':
        status_code = HTTPStatus.BAD_REQUEST
    return _json_response(status_code, prepared)


def handle_model_hamiltonian_edit_request(
    request_body: bytes,
    *,
    llm_request_builder: Any,
) -> JsonResponse:
    payload, error_response = decode_json_request(request_body)
    if error_response is not None:
        return error_response
    assert payload is not None
    try:
        prepared = CalculationApplicationService(
            llm_request_builder=llm_request_builder,
        ).prepare_model_hamiltonian_edit(
            payload.get('task_spec'),
            payload.get('request'),
            messages=payload.get('messages'),
            locale=normalize_locale(payload.get('locale')),
            lifecycle=payload.get('lifecycle'),
        )
    except CalculationFeatureUnavailableError as exc:
        return _json_error(HTTPStatus.BAD_REQUEST, str(exc))
    except (FileNotFoundError, TypeError, ValueError) as exc:
        return _json_error(HTTPStatus.BAD_REQUEST, str(exc))
    except Exception as exc:
        return _handle_error(exc)
    return _json_response(HTTPStatus.OK, prepared)


def handle_active_space_probe_request(request_body: bytes) -> JsonResponse:
    payload, error_response = decode_json_request(request_body)
    if error_response is not None:
        return error_response
    assert payload is not None
    requested_strategy = str(payload.get('strategy') or 'auto').strip().lower()
    if requested_strategy != 'auto':
        return _json_error(
            HTTPStatus.BAD_REQUEST,
            'Calculation Assistant active-space selection is fixed to auto.',
        )
    try:
        prepared = CalculationApplicationService().prepare_active_space_probe(
            payload.get('task_spec'),
            target_solver=payload.get('target_solver'),
            target_solver_options=(
                payload.get('target_solver_options')
                if isinstance(payload.get('target_solver_options'), dict)
                else None
            ),
        )
    except (TypeError, ValueError) as exc:
        return _json_error(HTTPStatus.BAD_REQUEST, str(exc))
    return _json_response(HTTPStatus.OK, prepared)


def handle_task_review_action_request(request_body: bytes) -> JsonResponse:
    payload, error_response = decode_json_request(request_body)
    if error_response is not None:
        return error_response
    assert payload is not None
    action = str(payload.get('action') or '').strip().lower()
    try:
        lifecycle = CalculationApplicationService.apply_review_action(
            payload.get('lifecycle'),
            action,
            review_type=payload.get('review_type'),
        )
    except (TypeError, ValueError) as exc:
        return _json_error(HTTPStatus.BAD_REQUEST, str(exc))
    return _json_response(HTTPStatus.OK, {'status': 'approved' if action == 'approve' else 'cancelled', 'lifecycle': lifecycle})


def handle_run_request(
    request_body: bytes,
    *,
    llm_request_builder: Any = None,
    task_executor: Optional[TaskExecutor] = None,
    execution_targets: Optional[ExecutionTargetRegistry] = None,
) -> JsonResponse:
    payload, error_response = decode_json_request(request_body)
    if error_response is not None:
        return error_response
    assert payload is not None

    locale = normalize_locale(payload.get('locale'))
    request_text = build_agent_request(payload)
    if not request_text:
        return _json_error(
            HTTPStatus.BAD_REQUEST,
            'At least one of "request" or "task_spec" must be provided with valid content',
        )
    work_dir = payload.get('work_dir')
    run_id = payload.get('run_id')
    model_input_file = _model_hamiltonian_input_file_from_payload(payload)
    if isinstance(model_input_file, str) and model_input_file.strip():
        inferred_work_dir, inferred_run_id = _infer_run_context_from_model_input_file(model_input_file)
        if not (isinstance(work_dir, str) and work_dir.strip()):
            work_dir = inferred_work_dir
        if not (isinstance(run_id, str) and run_id.strip()):
            run_id = inferred_run_id

    try:
        executor = (
            execution_targets.resolve(payload.get('execution_target'))
            if execution_targets is not None
            else task_executor or LocalExecutor()
        )
    except (TypeError, ValueError) as exc:
        return _json_error(HTTPStatus.BAD_REQUEST, str(exc))

    try:
        report = CalculationApplicationService(
            task_executor=executor,
            llm_request_builder=llm_request_builder,
        ).execute_request(
            request_text,
            channel='web',
            locale=locale,
            work_dir=work_dir,
            run_id=run_id,
            resource_profile=_resource_profile_from_payload(payload),
        )
    except Exception as exc:
        return _handle_error(exc)
    return _json_response(HTTPStatus.OK, report)


def _execution_service(
    payload: Dict[str, Any],
    *,
    llm_request_builder: Any = None,
    task_executor: Optional[TaskExecutor] = None,
    execution_targets: Optional[ExecutionTargetRegistry] = None,
) -> CalculationApplicationService:
    executor = (
        execution_targets.resolve(payload.get('execution_target'))
        if execution_targets is not None
        else task_executor or LocalExecutor()
    )
    return CalculationApplicationService(
        task_executor=executor,
        llm_request_builder=llm_request_builder,
    )


def _job_handle_from_payload(payload: Dict[str, Any]) -> JobHandle:
    handle = payload.get('handle')
    if not isinstance(handle, dict):
        raise ValueError('handle must be a JobHandle object')
    return JobHandle.from_dict(handle)


def handle_run_submit_request(
    request_body: bytes,
    *,
    llm_request_builder: Any = None,
    task_executor: Optional[TaskExecutor] = None,
    execution_targets: Optional[ExecutionTargetRegistry] = None,
) -> JsonResponse:
    payload, error_response = decode_json_request(request_body)
    if error_response is not None:
        return error_response
    assert payload is not None
    request_text = build_agent_request(payload)
    if not request_text:
        return _json_error(
            HTTPStatus.BAD_REQUEST,
            'At least one of "request" or "task_spec" must be provided with valid content',
        )
    locale = normalize_locale(payload.get('locale'))
    work_dir = payload.get('work_dir')
    run_id = payload.get('run_id')
    model_input_file = _model_hamiltonian_input_file_from_payload(payload)
    if isinstance(model_input_file, str) and model_input_file.strip():
        inferred_work_dir, inferred_run_id = _infer_run_context_from_model_input_file(model_input_file)
        if not (isinstance(work_dir, str) and work_dir.strip()):
            work_dir = inferred_work_dir
        if not (isinstance(run_id, str) and run_id.strip()):
            run_id = inferred_run_id
    try:
        service = _execution_service(
            payload,
            llm_request_builder=llm_request_builder,
            task_executor=task_executor,
            execution_targets=execution_targets,
        )
        handle, lifecycle = service.submit_request_with_lifecycle(
            request_text,
            lifecycle=payload.get('lifecycle'),
            channel='web',
            locale=locale,
            work_dir=work_dir,
            run_id=run_id,
            resource_profile=_resource_profile_from_payload(payload),
        )
        status = JobStatus(
            handle=handle,
            state=JobState.QUEUED,
            updated_at=handle.submitted_at,
            message='The calculation was accepted by the selected executor.',
        )
    except (TypeError, ValueError) as exc:
        return _json_error(HTTPStatus.BAD_REQUEST, str(exc))
    except Exception as exc:
        return _handle_error(exc)
    return _json_response(HTTPStatus.ACCEPTED, {'handle': handle.to_dict(), 'status': status.to_dict(), 'lifecycle': lifecycle})


def handle_task_view_request(
    request_body: bytes,
    *,
    task_executor: Optional[TaskExecutor] = None,
    execution_targets: Optional[ExecutionTargetRegistry] = None,
    work_dir: Optional[str] = None,
) -> JsonResponse:
    payload, error_response = decode_json_request(request_body)
    if error_response is not None:
        return error_response
    try:
        service = _execution_service(payload, task_executor=task_executor, execution_targets=execution_targets)
        handle = _job_handle_from_payload(payload)
        executor = (execution_targets.resolve(payload.get('execution_target'))
                    if execution_targets is not None else task_executor or LocalExecutor())
        if executor.describe().get('location') != 'remote_slurm_cluster':
            from pyscf_agent.paths import study_search_roots
            directory = (Path(handle.work_dir).expanduser() / handle.run_id).resolve()
            if not any(directory.is_relative_to(root.resolve()) for root in study_search_roots(work_dir)):
                raise ValueError('JobHandle is outside the configured work directory')
        view = service.inspect_task(handle)
    except (TypeError, ValueError) as exc:
        return _json_error(HTTPStatus.BAD_REQUEST, str(exc))
    except Exception as exc:
        return _handle_error(exc)
    return _json_response(HTTPStatus.OK, view, headers={'Cache-Control': 'no-store'})


def handle_run_status_request(
    request_body: bytes,
    *,
    task_executor: Optional[TaskExecutor] = None,
    execution_targets: Optional[ExecutionTargetRegistry] = None,
) -> JsonResponse:
    payload, error_response = decode_json_request(request_body)
    if error_response is not None:
        return error_response
    assert payload is not None
    try:
        service = _execution_service(
            payload,
            task_executor=task_executor,
            execution_targets=execution_targets,
        )
        status = service.job_status(_job_handle_from_payload(payload))
        lifecycle = service.status_lifecycle(payload.get('lifecycle'), status)
    except (TypeError, ValueError) as exc:
        return _json_error(HTTPStatus.BAD_REQUEST, str(exc))
    except Exception as exc:
        return _handle_error(exc)
    return _json_response(HTTPStatus.OK, {'status': status.to_dict(), 'lifecycle': lifecycle})


def handle_run_collect_request(
    request_body: bytes,
    *,
    llm_request_builder: Any = None,
    task_executor: Optional[TaskExecutor] = None,
    execution_targets: Optional[ExecutionTargetRegistry] = None,
) -> JsonResponse:
    payload, error_response = decode_json_request(request_body)
    if error_response is not None:
        return error_response
    assert payload is not None
    request_text = build_agent_request(payload)
    if not request_text:
        return _json_error(HTTPStatus.BAD_REQUEST, 'prepared_request is required to collect a run')
    try:
        service = _execution_service(
            payload,
            llm_request_builder=llm_request_builder,
            task_executor=task_executor,
            execution_targets=execution_targets,
        )
        handle = _job_handle_from_payload(payload)
        status = service.job_status(handle)
        lifecycle = service.status_lifecycle(payload.get('lifecycle'), status)
        if not status.report_available:
            return _json_response(HTTPStatus.ACCEPTED, {'status': status.to_dict(), 'lifecycle': lifecycle})
        report = service.collect_request(
            handle,
            request_text,
            locale=normalize_locale(payload.get('locale')),
            lifecycle=lifecycle,
        )
    except JobNotReadyError as exc:
        return _json_error(HTTPStatus.CONFLICT, str(exc))
    except (TypeError, ValueError) as exc:
        return _json_error(HTTPStatus.BAD_REQUEST, str(exc))
    except Exception as exc:
        return _handle_error(exc)
    return _json_response(HTTPStatus.OK, report)


def handle_run_cancel_request(
    request_body: bytes,
    *,
    task_executor: Optional[TaskExecutor] = None,
    execution_targets: Optional[ExecutionTargetRegistry] = None,
) -> JsonResponse:
    payload, error_response = decode_json_request(request_body)
    if error_response is not None:
        return error_response
    assert payload is not None
    try:
        service = _execution_service(
            payload,
            task_executor=task_executor,
            execution_targets=execution_targets,
        )
        status = service.cancel_job(_job_handle_from_payload(payload))
        lifecycle = service.status_lifecycle(payload.get('lifecycle'), status)
    except (TypeError, ValueError) as exc:
        return _json_error(HTTPStatus.BAD_REQUEST, str(exc))
    except Exception as exc:
        return _handle_error(exc)
    return _json_response(HTTPStatus.OK, {'status': status.to_dict(), 'lifecycle': lifecycle})


def handle_result_analysis_request(request_body: bytes, *, llm_request_builder: Any = None) -> JsonResponse:
    payload, error_response = decode_json_request(request_body)
    if error_response is not None:
        return error_response
    assert payload is not None

    locale = normalize_locale(payload.get('locale'))
    request_text = build_agent_request(payload)
    execution_report = payload.get('execution_report')
    if not request_text:
        return _json_error(
            HTTPStatus.BAD_REQUEST,
            'A prepared_request, request, or task_spec is required to analyze the result',
        )
    if not isinstance(execution_report, dict):
        return _json_error(HTTPStatus.BAD_REQUEST, 'execution_report must be an object')
    try:
        analysis = CalculationApplicationService(
            llm_request_builder=llm_request_builder,
        ).analyze_results(
            request_text,
            execution_report,
            locale=locale,
        )
    except CalculationFeatureUnavailableError as exc:
        return _json_error(HTTPStatus.BAD_REQUEST, str(exc))
    except Exception as exc:
        return _handle_error(exc)
    return _json_response(HTTPStatus.OK, {'result_analysis': analysis})
