from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from typing import Any, Dict, Mapping, Optional


RUNTIME_CONTEXT_SCHEMA = 'pyscf-agent.process-local-runtime-context.v1'
_MAX_CONTEXTS = 128
_LOCK = threading.RLock()
_CONTEXTS: 'OrderedDict[str, Dict[str, Any]]' = OrderedDict()


def store_runtime_context(
    run_id: Any,
    retry_count: Any,
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    """Retain non-serializable solver objects inside the current worker process."""

    if not isinstance(context, Mapping) or not context:
        raise ValueError('runtime context must be a non-empty mapping')
    context_id = 'context-' + uuid.uuid4().hex
    with _LOCK:
        _CONTEXTS[context_id] = dict(context)
        _CONTEXTS.move_to_end(context_id)
        while len(_CONTEXTS) > _MAX_CONTEXTS:
            _CONTEXTS.popitem(last=False)
    return {
        'schema': RUNTIME_CONTEXT_SCHEMA,
        'context_id': context_id,
        'run_id': str(run_id or ''),
        'retry_count': int(retry_count or 0),
        'storage': 'process_local',
    }


def get_runtime_context(reference: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(reference, Mapping):
        return None
    context_id = str(reference.get('context_id') or '').strip()
    if not context_id:
        return None
    with _LOCK:
        context = _CONTEXTS.get(context_id)
        if context is not None:
            _CONTEXTS.move_to_end(context_id)
        return context


def discard_runtime_context(reference: Any) -> None:
    if not isinstance(reference, Mapping):
        return
    context_id = str(reference.get('context_id') or '').strip()
    if not context_id:
        return
    with _LOCK:
        _CONTEXTS.pop(context_id, None)


def clear_runtime_contexts() -> None:
    """Clear process-local state; primarily useful for isolated test workers."""

    with _LOCK:
        _CONTEXTS.clear()


__all__ = [
    'RUNTIME_CONTEXT_SCHEMA',
    'clear_runtime_contexts',
    'discard_runtime_context',
    'get_runtime_context',
    'store_runtime_context',
]
