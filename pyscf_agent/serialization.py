"""Lossless JSON extensions and stable fingerprints for persisted contracts."""
from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
import math
from pathlib import PurePath
from typing import Any


def json_default(value: Any) -> Any:
    """Accept explicit portable types; never stringify an unknown object."""
    if isinstance(value, (PurePath, date, datetime)):
        # Preserve the historical representation of these explicitly allowed types.
        return str(value)
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f'Unsupported JSON value: {type(value).__module__}.{type(value).__qualname__}')


def canonical_json(value: Any, *, ensure_ascii: bool = False) -> str:
    return json.dumps(value, ensure_ascii=ensure_ascii, sort_keys=True,
                      separators=(',', ':'), default=json_default, allow_nan=False)


def json_fingerprint(value: Any, *, ensure_ascii: bool = False) -> str:
    return hashlib.sha256(canonical_json(value, ensure_ascii=ensure_ascii).encode('utf-8')).hexdigest()


def sanitize_result_json(value: Any) -> tuple[Any, list[dict[str, str]]]:
    """Copy result output to JSON values, recording nonfinite values as null.

    Paths are RFC 6901 JSON Pointers relative to the supplied payload. This is
    an output policy only: inputs, fingerprints and the repository stay strict.
    Unknown objects still raise TypeError; binary artifacts are never read.
    """
    warnings: list[dict[str, str]] = []

    def visit(item: Any, path: str) -> Any:
        if isinstance(item, float):
            if not math.isfinite(item):
                label = 'NaN' if math.isnan(item) else ('Infinity' if item > 0 else '-Infinity')
                warnings.append({
                    'code': 'nonfinite_result', 'path': path, 'value': label,
                    'message': 'Non-finite result replaced with null in JSON output.',
                })
                return None
            return item
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, dict):
            result = {}
            for key, child in item.items():
                # Match JSON's supported key conversions, without permitting
                # a nonfinite key to become an untraceable string field name.
                key_text = key if isinstance(key, str) else next(iter(
                    json.loads(json.dumps({key: None}, allow_nan=False))))
                token = key_text.replace('~', '~0').replace('/', '~1')
                result[key_text] = visit(child, path + '/' + token)
            return result
        if isinstance(item, (list, tuple)):
            return [visit(child, path + '/' + str(index)) for index, child in enumerate(item)]
        return visit(json_default(item), path)

    return visit(value, ''), warnings
