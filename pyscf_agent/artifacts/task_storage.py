"""Keep report JSON small while retaining complete numerical and text artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .arrays import array_references, compact_request, externalize_orbital_guesses
from .repository import default_artifact_repository


def compact_task_payload(payload: dict, directory: Path) -> dict:
    result = externalize_orbital_guesses(payload, directory / 'arrays')
    # Legacy callers may still carry a serialized request in message history.
    if 'user_request' in result:
        result['user_request'] = compact_request(result['user_request'], directory / 'arrays')
    for message in result.get('messages') or []:
        if isinstance(message, dict) and isinstance(message.get('content'), str):
            message['content'] = compact_request(message['content'], directory / 'arrays')
    artifacts = {item['path']: item for item in result.get('artifacts') or []
                 if isinstance(item, dict) and item.get('path')}
    for reference in array_references(result):
        artifacts[reference['path']] = reference

    def compact_text(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: compact_text(item) for key, item in value.items()}
        if isinstance(value, list):
            return [compact_text(item) for item in value]
        if isinstance(value, str) and len(value) > 12000:
            name = hashlib.sha256(value.encode('utf-8')).hexdigest() + '.txt'
            reference = default_artifact_repository().write_text(
                directory / 'details' / name, value, kind='task_detail',
                description='Complete task text; report contains a bounded preview',
            )
            artifacts[reference['path']] = reference
            marker = '[Full text: {0}]\n'.format(reference['path'])
            return marker + value[-(12000 - len(marker)):]
        return value

    # Input contracts and result values are never truncated. Only presentation
    # text/history has previews; the original text remains an artifact.
    for key in ('messages', 'logs', 'generated_input', 'raw_stdout', 'raw_scf_output', 'raw_stderr'):
        if key in result:
            result[key] = compact_text(result[key])
    result['artifacts'] = list(artifacts.values())
    return result
