"""Uniform JSON response encoding, without application or solver dependencies."""

from __future__ import annotations

from pyscf_agent.serialization import json_default

import json
import logging
from http import HTTPStatus
from typing import Any, Dict, Mapping, Optional, Tuple

JsonResponse = Tuple[HTTPStatus, Dict[str, str], bytes]


def json_response(
    status: HTTPStatus,
    payload: Dict[str, Any],
    *,
    headers: Optional[Mapping[str, str]] = None,
) -> JsonResponse:
    return (
        status,
        {'Content-Type': 'application/json; charset=utf-8', **(headers or {})},
        (
            json.dumps(payload, ensure_ascii=False, indent=2, default=json_default, allow_nan=False).encode(
                'utf-8'
            )
        ),
    )


def json_error(status: HTTPStatus, message: str) -> JsonResponse:
    return json_response(status, {'error': message})


def api_exception_response(
    exc: Exception, logger: logging.Logger, *, client_errors: tuple[type[Exception], ...] = (ValueError,),
    conflict_errors: tuple[type[Exception], ...] = (),
) -> JsonResponse:
    if isinstance(exc, conflict_errors):
        return json_error(HTTPStatus.CONFLICT, str(exc))
    if isinstance(exc, FileNotFoundError):
        return json_error(HTTPStatus.NOT_FOUND, str(exc))
    if isinstance(exc, TimeoutError):
        logger.warning('Control-plane operation timed out', exc_info=True)
        return json_error(HTTPStatus.SERVICE_UNAVAILABLE, 'Operation timed out; status is temporarily unreadable. Try again later.')
    if isinstance(exc, client_errors):
        return json_response(HTTPStatus.BAD_REQUEST, {'error': str(exc), 'validation_issues': []})
    logger.exception('Unexpected API failure')
    return json_error(HTTPStatus.INTERNAL_SERVER_ERROR, 'Internal server error. See the server log for details.')
