from __future__ import annotations

from typing import Any, Dict

from ..schema_contracts import REMOTE_RPC_SCHEMA


class RemoteProtocolError(RuntimeError):
    """Raised when the remote endpoint rejects or corrupts an RPC request."""

    def __init__(self, message: str, *, code: str = 'remote_error'):
        super().__init__(message)
        self.code = str(code or 'remote_error')


def success_response(operation: str, data: Any) -> Dict[str, Any]:
    return {
        'schema': REMOTE_RPC_SCHEMA,
        'ok': True,
        'operation': str(operation),
        'data': data,
    }


def error_response(
    operation: str,
    exc: Exception,
    *,
    code: str = 'remote_error',
) -> Dict[str, Any]:
    return {
        'schema': REMOTE_RPC_SCHEMA,
        'ok': False,
        'operation': str(operation),
        'error': {
            'code': str(code or 'remote_error'),
            'type': type(exc).__name__,
            'message': str(exc),
        },
    }


def response_data(payload: Any, *, operation: str) -> Any:
    if not isinstance(payload, dict):
        raise RemoteProtocolError('Remote RPC response must be a JSON object')
    if payload.get('schema') != REMOTE_RPC_SCHEMA:
        raise RemoteProtocolError(
            'Unsupported remote RPC schema: {0}'.format(payload.get('schema')),
            code='unsupported_schema',
        )
    if str(payload.get('operation') or '') != str(operation):
        raise RemoteProtocolError(
            'Remote RPC operation mismatch: expected {0}, received {1}'.format(
                operation,
                payload.get('operation'),
            ),
            code='operation_mismatch',
        )
    if not payload.get('ok'):
        error = payload.get('error') if isinstance(payload.get('error'), dict) else {}
        raise RemoteProtocolError(
            str(error.get('message') or 'Remote execution failed'),
            code=str(error.get('code') or 'remote_error'),
        )
    return payload.get('data')


__all__ = [
    'REMOTE_RPC_SCHEMA',
    'RemoteProtocolError',
    'error_response',
    'response_data',
    'success_response',
]
