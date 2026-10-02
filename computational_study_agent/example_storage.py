"""Lossless shared-row templates for portable example data.

Only distributed examples use this representation. Imported Studies retain the
ordinary plan/report schema and fully expanded cases.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


TEMPLATE_SCHEMA = 'cafes.example-row-template.v1'


def content_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(data.encode()).hexdigest()


def _changes(before, after, path=()):
    if type(before) is not type(after):
        yield path, {'set': after}
    elif isinstance(before, dict):
        for key in before:
            if key not in after:
                yield path + (key,), {'remove': True}
        for key, value in after.items():
            if key not in before:
                yield path + (key,), {'set': value}
            else:
                yield from _changes(before[key], value, path + (key,))
    elif isinstance(before, list) and len(before) == len(after):
        for index, (old, new) in enumerate(zip(before, after)):
            yield from _changes(old, new, path + (index,))
    elif before != after or (isinstance(before, float) and json.dumps(before) != json.dumps(after)):
        yield path, {'set': after}


def make_template(document, row_field='cases'):
    """Share one row; group paths that have identical per-row changes."""
    rows = document[row_field]
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError('Template rows must be a nonempty list of objects')
    columns = {}
    for index, row in enumerate(rows):
        for path, operation in _changes(rows[0], row):
            columns.setdefault(path, [None] * len(rows))[index] = operation
    groups = {}
    for path, values in columns.items():
        key = json.dumps(values, sort_keys=True, separators=(',', ':'))
        groups.setdefault(key, {'paths': [], 'values': values})['paths'].append(list(path))
    return {'schema': TEMPLATE_SCHEMA,
            'document': {key: value for key, value in document.items() if key != row_field},
            'row_field': row_field, 'row_count': len(rows), 'row_template': rows[0],
            'changes': list(groups.values()), 'expanded_sha256': content_hash(document)}


def expand_template(payload):
    if payload.get('schema') != TEMPLATE_SCHEMA:
        raise ValueError('Unsupported example template schema')
    result = copy.deepcopy(payload['document'])
    rows = [copy.deepcopy(payload['row_template']) for _ in range(payload['row_count'])]
    for group in payload['changes']:
        if len(group['values']) != len(rows):
            raise ValueError('Template change count differs from row count')
        for index, operation in enumerate(group['values']):
            if operation is None:
                continue
            for path in group['paths']:
                if not path:
                    rows[index] = copy.deepcopy(operation['set'])
                    continue
                parent = rows[index]
                for component in path[:-1]:
                    parent = parent[component]
                if 'set' in operation:
                    parent[path[-1]] = copy.deepcopy(operation['set'])
                elif operation == {'remove': True}:
                    del parent[path[-1]]
                else:
                    raise ValueError('Unsupported template change')
    result[payload['row_field']] = rows
    if content_hash(result) != payload['expanded_sha256']:
        raise ValueError('Expanded example checksum mismatch')
    return result


def read_example_json(path: Path):
    """Read plain JSON or the corresponding shared-template file."""
    source = path if path.is_file() else path.with_suffix('.template.json')
    payload = json.loads(source.read_text())
    return expand_template(payload) if payload.get('schema') == TEMPLATE_SCHEMA else payload


def materialize_study(directory: Path):
    for kind in ('plan', 'report', 'state'):
        target = directory / f'study-{kind}.json'
        template = target.with_suffix('.template.json')
        if template.exists():
            if target.exists():
                raise ValueError(f'Example contains both native and template files: {target.name}')
            payload = read_example_json(template)
            target.write_text(json.dumps(payload, indent=2) + '\n')
            template.unlink()
