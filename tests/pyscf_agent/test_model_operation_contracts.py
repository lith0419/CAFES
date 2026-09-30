"""Site groups and scalar IDs keep distinct meanings in the shared edit engine."""
import copy
import unittest

from pyscf_agent.backend.model_hamiltonian.operations import (
    apply_model_operations, normalize_nelec_value,
)


class ModelOperationContractTests(unittest.TestCase):
    def setUp(self):
        self.model = {
            'sites': [{'id': i, 'x': i, 'y': 0, 'U': 4, 'epsilon': 0} for i in range(3)],
            'bonds': [
                {'id': 7, 'source': 0, 'target': 2, 't': -1},
                {'id': 2, 'source': 1, 'target': 2, 't': -1},
            ],
            'nelec': [2, 1],
        }

    def test_site_list_has_an_actionable_error_without_changing_the_model(self):
        original = copy.deepcopy(self.model)
        with self.assertRaisesRegex(ValueError, r'operations\[0\]: site.*use sites'):
            apply_model_operations(self.model, [
                {'op': 'set_site_parameter', 'site': [0, 2], 'parameter': 'U', 'value': 8},
            ])
        self.assertEqual(self.model, original)

    def test_sites_updates_every_selected_site_for_all_site_operation_modes(self):
        for operation, value, expected in [
            ('set_site_parameter', {'value': 8}, 8),
            ('shift_site_parameter', {'shift': 3}, 7),
            ('scale_site_parameter', {'factor': 2}, 8),
            ('add_site_defect', {'value': -1}, 3),
        ]:
            with self.subTest(operation=operation):
                edited = apply_model_operations(self.model, [
                    {'op': operation, 'sites': [0, 2], 'parameter': 'U', **value},
                ])
                self.assertEqual([s['U'] for s in edited['sites']], [expected, 4, expected])
                self.assertEqual([s['U'] for s in self.model['sites']], [4, 4, 4])

    def test_site_ids_are_lossless_integers(self):
        for target in ({'site': 1.5}, {'site': True}, {'sites': [[0, 2]]},
                       {'sites': [0, False]}, {'selector': {'kind': 'ids', 'ids': [[0, 2]]}},
                       {'selector': {'kind': 'nearest_to', 'count': [2]}}):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'must be an integer'):
                apply_model_operations(self.model, [
                    {'op': 'set_site_parameter', 'parameter': 'U', 'value': 8, **target},
                ])
        for value in ('2', 2.0):
            with self.subTest(valid=value):
                edited = apply_model_operations(self.model, [
                    {'op': 'set_site_parameter', 'site': value, 'parameter': 'U', 'value': 8},
                ])
                self.assertEqual([s['U'] for s in edited['sites']], [4, 4, 8])

    def test_bond_endpoint_pairs_and_bond_id_lists_remain_distinct(self):
        for target, expected in [({'bond': [0, 2]}, [-2, -1]), ({'bonds': [2]}, [-1, -2])]:
            with self.subTest(target=target):
                edited = apply_model_operations(self.model, [
                    {'op': 'set_bond_parameter', 'parameter': 't', 'value': -2, **target},
                ])
                self.assertEqual([b['t'] for b in edited['bonds']], expected)

    def test_invalid_bond_targets_have_field_errors(self):
        for target, message in [
            ({'bond': [0, 1, 2]}, 'use bonds'),
            ({'bond': [[0], 2]}, r'selector.sites\[0\] must be an integer'),
            ({'bonds': [[2, 7]]}, r'bonds\[0\] must be an integer'),
            ({'bond': 2.5}, 'bond must be an integer'),
        ]:
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, message):
                apply_model_operations(self.model, [
                    {'op': 'set_bond_parameter', 'parameter': 't', 'value': -2, **target},
                ])

    def test_nested_electron_counts_have_field_errors(self):
        for value, message in [([[2], [1]], r'nelec\[0\]'),
                               ({'alpha': [2], 'beta': 1}, 'nelec.alpha'),
                               ([2, 1.5], r'nelec\[1\]')]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                normalize_nelec_value(value)
        self.assertEqual(normalize_nelec_value(3), [2, 1])
        self.assertEqual(normalize_nelec_value([1, 2]), [1, 2])

    def test_failed_later_operation_does_not_mutate_the_input(self):
        original = copy.deepcopy(self.model)
        with self.assertRaisesRegex(ValueError, r'cases\[1\].operations\[1\]: site'):
            apply_model_operations(self.model, [
                {'op': 'set_site_parameter', 'site': 0, 'parameter': 'U', 'value': 8},
                {'op': 'set_site_parameter', 'site': [1, 2], 'parameter': 'U', 'value': 8},
            ], path='cases[1].operations')
        self.assertEqual(self.model, original)
