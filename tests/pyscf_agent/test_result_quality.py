from __future__ import annotations

import unittest

from pyscf_agent.result_quality import evaluate_quality_checks


def _check(check_id, category, observed, operator, **contract):
    return {
        'id': check_id,
        'category': category,
        'observed': observed,
        'operator': operator,
        'required': True,
        'source': 'test.result',
        **contract,
    }


class ResultQualityTests(unittest.TestCase):
    def test_equal_requires_both_non_null_operands(self):
        for operands in (
            {}, {'observed': True}, {'expected': True},
            {'observed': None, 'expected': None},
            {'observed': None, 'expected': True},
            {'observed': True, 'expected': None},
        ):
            with self.subTest(operands=operands):
                result = evaluate_quality_checks([{
                    'id': 'sector', 'category': 'constraint', 'operator': 'equal',
                    'required': True, 'source': 'test.result', **operands,
                }])
                self.assertEqual(result['quality_status'], 'blocked')
                self.assertEqual(result['required_invalid_ids'], ['sector'])
                self.assertFalse(result['publication_eligible'])

    def test_equal_rejects_incompatible_or_non_scalar_evidence(self):
        for observed, expected in (
            (True, 1), (0, False), (1, '1'), ('true', True),
            ({}, {}), ([], []), ([True], [1]),
            (float('nan'), float('nan')), (float('inf'), float('inf')),
            (1, float('-inf')), (10 ** 1000, 10 ** 1000),
        ):
            with self.subTest(observed=observed, expected=expected):
                result = evaluate_quality_checks([
                    _check('sector', 'constraint', observed, 'equal', expected=expected),
                ])
                self.assertEqual(result['required_invalid_ids'], ['sector'])
                self.assertFalse(result['publication_eligible'])

    def test_equal_accepts_typed_scalar_evidence_without_numeric_rounding(self):
        for observed, expected, passed in (
            (True, True, True), (False, False, True), (False, True, False),
            (0, 0, True), (2, 2.0, True), (2.0, 2, True), (2, 3, False),
            ('su2', 'su2', True), ('sz', 'su2', False),
            (2 ** 53, 2 ** 53 + 1, False),
        ):
            with self.subTest(observed=observed, expected=expected):
                result = evaluate_quality_checks([
                    _check('sector', 'constraint', observed, 'equal', expected=expected),
                ])
                self.assertEqual(result['required_invalid_ids'], [])
                self.assertEqual(result['checks'][0]['status'], 'passed' if passed else 'failed')
                self.assertEqual(result['publication_eligible'], passed)

    def test_numeric_operators_require_finite_numbers_and_explicit_operands(self):
        for operator in (
            'finite', 'absolute_less_than', 'less_than',
            'less_than_or_equal', 'greater_than_or_equal',
        ):
            operands = ['observed'] if operator == 'finite' else ['observed', 'limit']
            for operand in operands:
                for missing, value in (
                    (True, None), (False, None), (False, True), (False, '1'),
                    (False, []), (False, {}), (False, float('nan')),
                    (False, float('inf')), (False, 10 ** 1000),
                ):
                    with self.subTest(operator=operator, operand=operand, missing=missing, value=value):
                        check = _check('number', 'constraint', 0, operator, limit=1)
                        if missing:
                            check.pop(operand)
                        else:
                            check[operand] = value
                        result = evaluate_quality_checks([check])
                        self.assertEqual(result['required_invalid_ids'], ['number'])
                        self.assertFalse(result['publication_eligible'])

    def test_numeric_operator_boundaries_preserve_valid_comparisons(self):
        for operator, observed, limit, passed in (
            ('finite', 0, None, True),
            ('absolute_less_than', -1, 1, False),
            ('absolute_less_than', -0.5, 1, True),
            ('less_than', -2, -1, True), ('less_than', 1, 1, False),
            ('less_than_or_equal', 1, 1, True),
            ('greater_than_or_equal', 0, 0, True),
            ('greater_than_or_equal', -1, 0, False),
        ):
            with self.subTest(operator=operator, observed=observed, limit=limit):
                check = _check('number', 'constraint', observed, operator)
                if operator != 'finite':
                    check['limit'] = limit
                result = evaluate_quality_checks([check])
                self.assertEqual(result['required_invalid_ids'], [])
                self.assertEqual(result['publication_eligible'], passed)

    def test_optional_invalid_evidence_is_recorded_without_blocking(self):
        result = evaluate_quality_checks([
            _check('optional', 'constraint', None, 'equal', required=False),
            _check('energy', 'constraint', -1.0, 'finite'),
        ])
        self.assertEqual(result['checks'][0]['status'], 'invalid')
        self.assertEqual(result['quality_status'], 'passed')
        self.assertTrue(result['publication_eligible'])

    def test_required_convergence_failure_requires_review(self):
        result = evaluate_quality_checks([
            _check(
                'energy_change',
                'convergence',
                2.0e-5,
                'absolute_less_than',
                limit=1.0e-6,
            ),
            _check('energy_finite', 'constraint', -1.0, 'finite'),
        ])

        self.assertEqual(result['quality_status'], 'review_required')
        self.assertFalse(result['publication_eligible'])
        self.assertEqual(result['convergence_failed_ids'], ['energy_change'])

    def test_required_constraint_failure_blocks_result(self):
        result = evaluate_quality_checks([
            _check('sector', 'constraint', False, 'equal', expected=True),
        ])

        self.assertEqual(result['quality_status'], 'blocked')
        self.assertFalse(result['publication_eligible'])
        self.assertEqual(result['constraint_failed_ids'], ['sector'])

    def test_missing_checks_preserve_legacy_behavior(self):
        result = evaluate_quality_checks(None)

        self.assertEqual(result['evidence_status'], 'legacy_unavailable')
        self.assertEqual(result['quality_status'], 'legacy')
        self.assertIsNone(result['publication_eligible'])


    def test_missing_convergence_evidence_requires_review(self):
        result = evaluate_quality_checks([
            _check('missing_value', 'convergence', None, 'less_than', limit=1.0e-6),
        ])

        self.assertEqual(result['quality_status'], 'review_required')
        self.assertEqual(result['required_invalid_ids'], ['missing_value'])

    def test_declared_but_empty_checks_are_not_treated_as_legacy(self):
        result = evaluate_quality_checks([])

        self.assertEqual(result['evidence_status'], 'available')
        self.assertEqual(result['quality_status'], 'blocked')
        self.assertFalse(result['publication_eligible'])


if __name__ == '__main__':
    unittest.main()
