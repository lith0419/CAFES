from __future__ import annotations

import copy
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


TASK_LIFECYCLE_SCHEMA = 'pyscf-agent.task-lifecycle.v1'
STUDY_LIFECYCLE_SCHEMA = 'pyscf-agent.study-lifecycle.v1'


class LifecycleTransitionError(ValueError):
    """Raised when a workflow lifecycle event is invalid for its current stage."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event(
    target: str,
    sources: Iterable[str],
    *,
    increments_revision: bool = False,
) -> Dict[str, Any]:
    return {
        'target': target,
        'sources': frozenset(str(item) for item in sources),
        'increments_revision': bool(increments_revision),
    }


_TASK_EVENTS = {
    'validation_passed': _event('validated', ('draft', 'blocked')),
    'validation_failed': _event('blocked', ('draft', 'validated', 'prepared')),
    'request_prepared': _event('prepared', ('validated',)),
    'execution_submitted': _event('queued', ('prepared', 'retry_ready')),
    'execution_started': _event('running', ('queued',)),
    'retry_started': _event('running', ('failed', 'unconverged', 'retry_ready')),
    'execution_succeeded': _event('succeeded', ('running',)),
    'execution_failed': _event('failed', ('running',)),
    'execution_unconverged': _event('unconverged', ('running',)),
    'review_requested': _event(
        'review_required',
        ('prepared', 'blocked', 'failed', 'unconverged', 'succeeded'),
    ),
    'approval_granted': _event('retry_ready', ('review_required',)),
    'approval_rejected': _event('cancelled', ('review_required',)),
    'configuration_changed': _event(
        'draft',
        (
            'draft',
            'validated',
            'prepared',
            'blocked',
            'succeeded',
            'failed',
            'unconverged',
            'review_required',
            'retry_ready',
            'cancelled',
        ),
        increments_revision=True,
    ),
    'cancelled': _event(
        'cancelled',
        ('prepared', 'queued', 'running', 'review_required', 'retry_ready'),
    ),
}


_STUDY_EVENTS = {
    'execution_started': _event('executing', ('planned', 'retry_ready')),
    'execution_completed': _event('completed', ('executing', 'refining')),
    'review_requested': _event(
        'review_required',
        ('planned', 'executing', 'refining', 'completed', 'analyzing'),
    ),
    'approval_granted': _event('retry_ready', ('review_required',)),
    'approval_rejected': _event('cancelled', ('review_required',)),
    'refinement_started': _event('refining', ('retry_ready',)),
    'analysis_started': _event('analyzing', ('completed',)),
    'analysis_completed': _event('completed', ('analyzing',)),
    'configuration_changed': _event(
        'planned',
        (
            'planned',
            'completed',
            'review_required',
            'retry_ready',
            'cancelled',
        ),
        increments_revision=True,
    ),
    'cancelled': _event(
        'cancelled',
        ('planned', 'executing', 'review_required', 'retry_ready', 'refining', 'analyzing'),
    ),
}


_TERMINAL_STAGES = {
    'task': frozenset(('succeeded', 'failed', 'unconverged', 'blocked', 'cancelled')),
    'study': frozenset(('completed', 'cancelled')),
}


def _contracts(scope: str) -> Mapping[str, Mapping[str, Any]]:
    normalized = str(scope or '').strip().lower()
    if normalized == 'task':
        return _TASK_EVENTS
    if normalized == 'study':
        return _STUDY_EVENTS
    raise ValueError("Lifecycle scope must be 'task' or 'study'.")


def _schema(scope: str) -> str:
    return TASK_LIFECYCLE_SCHEMA if scope == 'task' else STUDY_LIFECYCLE_SCHEMA


def allowed_lifecycle_events(lifecycle: Mapping[str, Any]) -> Tuple[str, ...]:
    scope = str(lifecycle.get('scope') or '').strip().lower()
    stage = str(lifecycle.get('stage') or '').strip().lower()
    return tuple(
        event_name
        for event_name, contract in _contracts(scope).items()
        if stage in contract['sources']
    )


def _refresh(lifecycle: Dict[str, Any]) -> Dict[str, Any]:
    scope = str(lifecycle['scope'])
    stage = str(lifecycle['stage'])
    lifecycle['terminal'] = stage in _TERMINAL_STAGES[scope]
    lifecycle['allowed_events'] = list(allowed_lifecycle_events(lifecycle))
    return lifecycle


def new_lifecycle(
    scope: str,
    entity_id: Optional[str] = None,
    *,
    at: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_scope = str(scope or '').strip().lower()
    _contracts(normalized_scope)
    created_at = str(at or _timestamp())
    stage = 'draft' if normalized_scope == 'task' else 'planned'
    identifier = str(entity_id or '').strip() or '{0}-{1}'.format(
        normalized_scope,
        uuid.uuid4().hex,
    )
    lifecycle = {
        'schema': _schema(normalized_scope),
        'scope': normalized_scope,
        'entity_id': identifier,
        'revision': 1,
        'stage': stage,
        'terminal': False,
        'current_event': 'created',
        'created_at': created_at,
        'updated_at': created_at,
        'history': [{
            'sequence': 1,
            'event': 'created',
            'from_stage': None,
            'to_stage': stage,
            'revision': 1,
            'at': created_at,
            'details': {},
        }],
    }
    return _refresh(lifecycle)


def ensure_lifecycle(
    lifecycle: Any,
    scope: str,
    *,
    entity_id: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_scope = str(scope or '').strip().lower()
    if not isinstance(lifecycle, dict) or lifecycle.get('scope') != normalized_scope:
        return new_lifecycle(normalized_scope, entity_id)
    value = copy.deepcopy(lifecycle)
    value.setdefault('schema', _schema(normalized_scope))
    value.setdefault('entity_id', str(entity_id or '').strip() or '{0}-{1}'.format(
        normalized_scope,
        uuid.uuid4().hex,
    ))
    value.setdefault('revision', 1)
    value.setdefault('stage', 'draft' if normalized_scope == 'task' else 'planned')
    value.setdefault('history', [])
    value.setdefault('created_at', _timestamp())
    value.setdefault('updated_at', value['created_at'])
    value.setdefault('current_event', 'created')
    return _refresh(value)


def transition_lifecycle(
    lifecycle: Mapping[str, Any],
    event: str,
    *,
    details: Optional[Mapping[str, Any]] = None,
    at: Optional[str] = None,
) -> Dict[str, Any]:
    value = ensure_lifecycle(lifecycle, str(lifecycle.get('scope') or ''))
    event_name = str(event or '').strip().lower()
    contract = _contracts(value['scope']).get(event_name)
    if contract is None:
        raise LifecycleTransitionError(
            "Unknown {0} lifecycle event: {1}".format(value['scope'], event_name)
        )
    source = str(value['stage'])
    target = str(contract['target'])
    if source == target and value.get('current_event') == event_name:
        return value
    if source not in contract['sources']:
        raise LifecycleTransitionError(
            "Cannot apply {0} lifecycle event '{1}' while stage is '{2}'.".format(
                value['scope'],
                event_name,
                source,
            )
        )
    if contract['increments_revision']:
        value['revision'] = int(value.get('revision') or 1) + 1
    event_at = str(at or _timestamp())
    history = list(value.get('history') or [])
    history.append({
        'sequence': len(history) + 1,
        'event': event_name,
        'from_stage': source,
        'to_stage': target,
        'revision': int(value['revision']),
        'at': event_at,
        'details': copy.deepcopy(dict(details or {})),
    })
    value.update({
        'stage': target,
        'current_event': event_name,
        'updated_at': event_at,
        'history': history,
    })
    return _refresh(value)


def prepare_task_for_execution(
    lifecycle: Mapping[str, Any],
    *,
    details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    value = ensure_lifecycle(lifecycle, 'task')
    stage = str(value['stage'])
    if stage == 'draft':
        value = transition_lifecycle(value, 'validation_passed', details=details)
        stage = 'validated'
    if stage == 'validated':
        value = transition_lifecycle(value, 'request_prepared', details=details)
        stage = 'prepared'
    if stage in ('failed', 'unconverged'):
        return transition_lifecycle(value, 'retry_started', details=details)
    if stage in ('prepared', 'retry_ready'):
        value = transition_lifecycle(value, 'execution_submitted', details=details)
        stage = 'queued'
    if stage == 'queued':
        value = transition_lifecycle(value, 'execution_started', details=details)
    return value


def task_lifecycle_for_preparation(
    status: Any,
    *,
    lifecycle: Any = None,
    entity_id: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    value = ensure_lifecycle(lifecycle, 'task', entity_id=entity_id)
    normalized = str(status or '').strip().lower()
    if normalized in ('ready', 'awaiting_approval'):
        if value['stage'] == 'draft':
            value = transition_lifecycle(value, 'validation_passed', details=details)
        if value['stage'] == 'validated':
            value = transition_lifecycle(value, 'request_prepared', details=details)
        if normalized == 'awaiting_approval':
            value = transition_lifecycle(value, 'review_requested', details=details)
        return value
    if normalized in ('needs_clarification', 'unavailable', 'blocked', 'invalid'):
        if value['stage'] in ('draft', 'validated', 'prepared'):
            value = transition_lifecycle(value, 'validation_failed', details=details)
        return value
    return value


def submit_task_lifecycle(
    lifecycle: Any,
    *,
    entity_id: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    value = ensure_lifecycle(lifecycle, 'task', entity_id=entity_id)
    if value['stage'] == 'draft':
        value = transition_lifecycle(value, 'validation_passed', details=details)
    if value['stage'] == 'validated':
        value = transition_lifecycle(value, 'request_prepared', details=details)
    if value['stage'] not in ('prepared', 'retry_ready', 'queued'):
        raise LifecycleTransitionError(
            "Task must be prepared or approved before submission; current stage is '{0}'.".format(
                value['stage']
            )
        )
    if value['stage'] != 'queued':
        value = transition_lifecycle(value, 'execution_submitted', details=details)
    return value


def task_lifecycle_from_job_status(
    lifecycle: Any,
    job_state: Any,
    *,
    task_status: Any = None,
    entity_id: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    value = ensure_lifecycle(lifecycle, 'task', entity_id=entity_id)
    normalized = str(getattr(job_state, 'value', job_state) or '').strip().lower()
    if normalized == 'queued':
        if value['stage'] in ('queued', 'running'):
            return value
        return submit_task_lifecycle(value, entity_id=entity_id, details=details)
    if normalized == 'running':
        if value['stage'] == 'running':
            return value
        value = submit_task_lifecycle(value, entity_id=entity_id, details=details)
        if value['stage'] == 'queued':
            value = transition_lifecycle(value, 'execution_started', details=details)
        return value
    if normalized in ('completed', 'failed'):
        if value['stage'] in ('succeeded', 'failed', 'unconverged', 'cancelled'):
            return value
        if value['stage'] != 'running':
            value = submit_task_lifecycle(value, entity_id=entity_id, details=details)
        if value['stage'] == 'queued':
            value = transition_lifecycle(value, 'execution_started', details=details)
        outcome = str(task_status or normalized).strip().lower()
        if outcome == 'succeeded':
            return transition_lifecycle(value, 'execution_succeeded', details=details)
        if outcome == 'unconverged':
            return transition_lifecycle(value, 'execution_unconverged', details=details)
        if outcome == 'blocked':
            value = transition_lifecycle(value, 'execution_failed', details=details)
            return transition_lifecycle(value, 'review_requested', details=details)
        return transition_lifecycle(value, 'execution_failed', details=details)
    if normalized == 'cancelled':
        if value['stage'] in ('prepared', 'retry_ready'):
            value = transition_lifecycle(value, 'execution_submitted', details=details)
        if value['stage'] in ('queued', 'running'):
            return transition_lifecycle(value, 'cancelled', details=details)
    return value


def lifecycle_requires_review(decisions: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        str(item.get('status') or '').strip().lower() in ('review_required', 'blocked')
        for item in decisions
        if isinstance(item, Mapping)
    )


def begin_study_execution(
    lifecycle: Any,
    *,
    entity_id: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    value = ensure_lifecycle(lifecycle, 'study', entity_id=entity_id)
    if value['stage'] == 'completed':
        value = transition_lifecycle(value, 'configuration_changed', details=details)
    if value['stage'] in ('planned', 'retry_ready'):
        value = transition_lifecycle(value, 'execution_started', details=details)
    if value['stage'] not in ('executing', 'refining'):
        raise LifecycleTransitionError(
            "Study is not ready for execution; current stage is '{0}'.".format(
                value['stage']
            )
        )
    return value


def complete_study_execution(
    lifecycle: Any,
    *,
    entity_id: Optional[str] = None,
    details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    value = ensure_lifecycle(lifecycle, 'study', entity_id=entity_id)
    if value['stage'] in ('executing', 'refining'):
        return transition_lifecycle(value, 'execution_completed', details=details)
    if value['stage'] == 'completed':
        return value
    raise LifecycleTransitionError(
        "Study execution cannot complete while stage is '{0}'.".format(value['stage'])
    )


def synchronize_study_report_lifecycle(
    lifecycle: Any,
    *,
    entity_id: Optional[str] = None,
    decisions: Sequence[Mapping[str, Any]] = (),
    details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    value = ensure_lifecycle(lifecycle, 'study', entity_id=entity_id)
    if value['stage'] == 'planned':
        value = begin_study_execution(value, entity_id=entity_id, details=details)
    if value['stage'] in ('executing', 'refining'):
        value = complete_study_execution(value, entity_id=entity_id, details=details)
    if lifecycle_requires_review(decisions) and value['stage'] != 'review_required':
        value = transition_lifecycle(value, 'review_requested', details=details)
    return value


__all__ = [
    'LifecycleTransitionError',
    'STUDY_LIFECYCLE_SCHEMA',
    'TASK_LIFECYCLE_SCHEMA',
    'allowed_lifecycle_events',
    'begin_study_execution',
    'complete_study_execution',
    'ensure_lifecycle',
    'lifecycle_requires_review',
    'new_lifecycle',
    'prepare_task_for_execution',
    'submit_task_lifecycle',
    'synchronize_study_report_lifecycle',
    'task_lifecycle_for_preparation',
    'task_lifecycle_from_job_status',
    'transition_lifecycle',
]
