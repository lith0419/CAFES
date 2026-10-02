"""Templates preserve native JSON values and independent mutable case data."""

import copy
import unittest

from computational_study_agent.example_storage import content_hash, expand_template, make_template


class ExampleStorageTests(unittest.TestCase):
    def test_roundtrip_preserves_types_missing_values_and_variable_lists(self):
        original = {'schema': 'native-plan', 'cases': [
            {'case_id': 'a', 'left': 1, 'right': 1, 'optional': None, 'removed': 'old',
             'data': {'array': [1, 2], 'nested': {'value': True}}, 'zero': 0.0},
            {'case_id': 'b', 'left': 3.0, 'right': 3.0, 'optional': {'set': None},
             'data': {'array': [2, 4], 'nested': []}, 'added': False, 'zero': -0.0},
            {'case_id': 'c', 'left': None, 'right': None, 'removed': 'old',
             'data': {'array': [8], 'nested': {'value': False}}, 'zero': 0},
        ]}
        encoded = make_template(original)
        restored = expand_template(encoded)
        self.assertEqual(content_hash(original), content_hash(restored))
        self.assertTrue(any(['left'] in group['paths'] and ['right'] in group['paths']
                            for group in encoded['changes']))
        restored['cases'][0]['data']['array'][0] = 100
        self.assertEqual(restored['cases'][1]['data']['array'], [2, 4])
        self.assertEqual(original['cases'][0]['data']['array'], [1, 2])
        self.assertEqual(encoded['row_template']['data']['array'], [1, 2])

    def test_template_integrity_rejects_changed_values(self):
        encoded = make_template({'cases': [{'x': 1}, {'x': 2}]})
        damaged = copy.deepcopy(encoded)
        damaged['changes'][0]['values'][1] = {'set': 9}
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            expand_template(damaged)
        damaged = copy.deepcopy(encoded)
        damaged['changes'][0]['values'].pop()
        with self.assertRaisesRegex(ValueError, 'row count'):
            expand_template(damaged)

    def test_records_and_empty_changes(self):
        document = {'records': [{'data': [3, 4]}, {'data': [3, 4]}]}
        restored = expand_template(make_template(document, 'records'))
        restored['records'][0]['data'].append(5)
        self.assertEqual(restored['records'][1]['data'], [3, 4])
