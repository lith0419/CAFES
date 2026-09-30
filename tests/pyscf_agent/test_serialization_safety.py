from datetime import datetime, timezone
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pyscf_agent.artifacts import default_artifact_repository
from pyscf_agent.serialization import canonical_json, json_default, json_fingerprint
from pyscf_agent.transport.json import json_response
from http import HTTPStatus


class SerializationSafetyTests(unittest.TestCase):
    def test_production_json_writers_never_silently_stringify_unknown_values(self):
        root = Path(__file__).resolve().parents[2]
        violations = []
        for package in ('pyscf_agent', 'computational_study_agent'):
            for path in (root / package).rglob('*.py'):
                for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr in ('dump', 'dumps')):
                        if any(item.arg == 'default' and isinstance(item.value, ast.Name)
                               and item.value.id == 'str' for item in node.keywords):
                            violations.append(f'{path.relative_to(root)}:{node.lineno}')
        self.assertEqual(violations, [])

    def test_numpy_values_keep_types_and_entire_large_array(self):
        source = {'n': np.int64(5), 'b': np.bool_(True), 'a': np.arange(10000.0)}
        encoded = canonical_json(source)
        result = json.loads(encoded)
        self.assertIs(type(result['n']), int)
        self.assertIs(type(result['b']), bool)
        self.assertEqual(result['a'], list(np.arange(10000.0)))
        self.assertNotIn('...', encoded)
        self.assertEqual(json.loads(json_response(HTTPStatus.OK, source)[2]), result)

    def test_path_datetime_are_explicitly_allowed_but_complex_is_not(self):
        instant = datetime(2026, 9, 25, tzinfo=timezone.utc)
        self.assertEqual(json.loads(canonical_json({'path': Path('/tmp/a'), 'time': instant})),
                         {'path': '/tmp/a', 'time': str(instant)})
        for value in (object(), 1+2j, np.array([1+2j]), {'x'}, b'abc'):
            with self.subTest(value=type(value)), self.assertRaises(TypeError):
                canonical_json(value)
        with self.assertRaises(ValueError):
            canonical_json({'bad': np.array([np.nan])})

    def test_failed_encoding_preserves_old_artifact(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'state.json'
            repo = default_artifact_repository()
            repo.write_json(path, {'version': 1}, kind='study-state')
            before = path.read_bytes()
            with self.assertRaises(TypeError):
                repo.write_json(path, {'values': [1, object()]}, kind='study-state')
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(root).glob('.*.tmp')), [])
            repo.write_json(path, {'n': np.int64(5), 'b': np.bool_(True)}, kind='study-state')
            self.assertEqual(json.loads(path.read_text())['n'], 5)
            self.assertIs(json.loads(path.read_text())['b'], True)

    def test_fingerprint_bytes_remain_compatible_for_saved_json(self):
        from computational_study_agent.grid.refinement import fingerprint as grid_fingerprint, canonical
        value = {'unicode': '能量', 'values': [1, 1.5, True, None]}
        expected = hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(json_fingerprint(value), expected)
        grid_expected = hashlib.sha256(json.dumps(canonical(value), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(grid_fingerprint(value), grid_expected)
        self.assertEqual(grid_fingerprint({'u': 1}), grid_fingerprint({'u': 1.0}))
        self.assertEqual(json_fingerprint(np.array([1, 2])), json_fingerprint([1, 2]))

    def test_imported_consumers_resolve_repository_at_call_time(self):
        from pyscf_agent.backend import artifacts
        default_artifact_repository.cache_clear()
        try:
            with patch('pyscf_agent.artifacts.repository.ArtifactRepository') as factory:
                artifacts.default_artifact_repository().write_json('unused', {})
                factory.assert_called_once_with()
                factory.return_value.write_json.assert_called_once_with('unused', {})
        finally:
            default_artifact_repository.cache_clear()

    def test_unknown_objects_are_not_coerced_by_string_protocol(self):
        class Unknown:
            def __str__(self):
                raise AssertionError('must not stringify')
        with self.assertRaises(TypeError):
            json.dumps(Unknown(), default=json_default)
