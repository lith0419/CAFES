from __future__ import annotations

from pyscf_agent.serialization import canonical_json, sanitize_result_json

import copy
import logging
from typing import Any, Dict, List, Optional

from .script_helpers import _compose_raw_stdout as _compose_raw_stdout

from .artifacts import compact_structured_results, ensure_run_id, resolve_work_dir
from ..contracts import LogEntry, MessageEnvelope, log_entry_to_dict, message_to_dict, utc_timestamp
from ..pyscf_i18n import normalize_locale
from ..lifecycle import new_lifecycle

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def normalize_message(
    user_request: Any,
    *,
    channel: str = 'agent',
    role: str = 'user',
    kind: str = 'request',
) -> Dict[str, Any]:
    if isinstance(user_request, dict):
        metadata = copy.deepcopy(user_request.get('metadata') or {})
        return message_to_dict(MessageEnvelope(
            role=user_request.get('role', role),
            kind=user_request.get('kind', kind),
            content=user_request.get('content', ''),
            channel=user_request.get('channel', channel),
            metadata=metadata,
        ))
    return message_to_dict(MessageEnvelope(
        role=role,
        kind=kind,
        content=str(user_request),
        channel=channel,
    ))


def append_message(
    state: Dict[str, Any],
    *,
    role: str,
    kind: str,
    content: str,
    channel: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        return state
    state.setdefault('messages', []).append(message_to_dict(MessageEnvelope(
        role=role,
        kind=kind,
        content=content,
        channel=channel or state.get('channel', 'agent'),
        metadata=copy.deepcopy(metadata or {}),
    )))
    return state


def append_log(
    state: Dict[str, Any],
    level: str,
    event: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entry = log_entry_to_dict(LogEntry(
        level=level,
        event=event,
        details=copy.deepcopy(details or {}),
    ))
    state.setdefault('logs', []).append(entry)
    log_level = getattr(logging, level.upper(), logging.INFO)
    log_output, warnings = sanitize_result_json(entry)
    if warnings:
        log_output['warnings'] = warnings
    LOGGER.log(log_level, canonical_json(log_output))
    return state


def default_state(user_request: Any, *, channel: str = 'agent', locale: str = 'en', work_dir: Optional[str] = None, run_id: Optional[str] = None) -> Dict[str, Any]:
    from ..artifacts.arrays import compact_request
    from .artifacts import make_run_id

    run_id = str(run_id or '').strip() or make_run_id()
    user_request = compact_request(user_request, resolve_work_dir(work_dir, run_id) / run_id / 'arrays')
    message = normalize_message(user_request, channel=channel)
    normalized_locale = normalize_locale(locale)
    normalized_run_id = run_id.strip() if isinstance(run_id, str) and run_id.strip() else ''
    state = {
        'run_id': normalized_run_id,
        'work_dir': str(resolve_work_dir(work_dir, normalized_run_id)),
        'channel': message['channel'],
        'calculation_role': 'calculation',
        'locale': normalized_locale,
        'user_request': message['content'],
        'task_spec': None,
        'attempts': [],
        'errors': [],
        'warnings': [],
        'applied_defaults': [],
        'validation_errors': [],
        'clarification_questions': [],
        'generated_input': None,
        'execution_status': 'pending',
        'raw_stdout': '',
        'raw_scf_output': '',
        'analysis_text': '',
        'raw_stderr': '',
        'structured_results': None,
        'compact_results': {},
        'analysis_summary': '',
        'workflow_configuration': {},
        'workflow_provenance': {},
        'module_execution_trace': [],
        'module_runtime_observations': [],
        'gate_configuration': {},
        'gate_provenance': {},
        'gate_decisions': [],
        'gate_execution_trace': [],
        'lifecycle': {},
        'retry_count': 0,
        'max_retries': 1,
        'artifacts': [],
        'messages': [message],
        'logs': [],
    }
    ensure_run_id(state)
    state['lifecycle'] = new_lifecycle('task', state['run_id'])
    append_log(state, 'info', 'workflow.initialized', {
        'run_id': state['run_id'],
        'work_dir': state['work_dir'],
        'channel': state['channel'],
        'kind': message['kind'],
        'locale': normalized_locale,
    })
    return state


def _make_error(
    stage: str,
    code: str,
    message: str,
    *,
    details: Optional[Dict[str, Any]] = None,
    exception_type: Optional[str] = None,
) -> Dict[str, Any]:
    error = {
        'stage': stage,
        'code': code,
        'message': message,
        'details': copy.deepcopy(details or {}),
    }
    if exception_type is not None:
        error['exception_type'] = exception_type
    return error


def _append_validation_issue(
    errors: List[Dict[str, Any]],
    validation_errors: List[str],
    questions: List[str],
    code: str,
    message: str,
    *,
    question: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    validation_errors.append(message)
    errors.append(_make_error('validation', code, message, details=details))
    if question:
        questions.append(question)


def _append_errors_to_state(state: Dict[str, Any], new_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not new_errors:
        return state
    state.setdefault('errors', []).extend(copy.deepcopy(new_errors))
    return state


def _append_attempt(
    state: Dict[str, Any],
    *,
    status: str,
    stage: str,
    errors: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    state.setdefault('attempts', []).append({
        'index': len(state.get('attempts', [])) + 1,
        'timestamp': utc_timestamp(),
        'stage': stage,
        'status': status,
        'retry_count': state.get('retry_count', 0),
        'task_spec': copy.deepcopy(state.get('task_spec')),
        'raw_stdout': state.get('raw_stdout', ''),
        'raw_scf_output': state.get('raw_scf_output', ''),
        'analysis_text': state.get('analysis_text', ''),
        'raw_stderr': state.get('raw_stderr', ''),
        'errors': copy.deepcopy(errors or []),
        'structured_results': copy.deepcopy(state.get('compact_results') or compact_structured_results(state.get('structured_results'))),
        'compact_results': copy.deepcopy(state.get('compact_results') or compact_structured_results(state.get('structured_results'))),
        'artifacts': copy.deepcopy(state.get('artifacts', [])),
    })
    return state


def _latest_error_message(state: Dict[str, Any], *, stage: Optional[str] = None) -> Optional[str]:
    errors = state.get('errors') or []
    for error in reversed(errors):
        if stage is None or error.get('stage') == stage:
            message = error.get('message')
            if message:
                return message
    return None
