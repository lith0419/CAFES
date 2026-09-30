from __future__ import annotations

import math
import unittest
from pathlib import Path

from computational_study_agent.adaptive.path_diagnostics import analyze_scan_path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARK_PATH = _REPOSITORY_ROOT / 'reports' / 'benchmarks' / 'n2_contextual_vqe_comparison.txt'
_N2_REFERENCE_BOND_LENGTH_ANGSTROM = 1.09768


def _benchmark_rows():
    lines = [
        line.strip()
        for line in _BENCHMARK_PATH.read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.startswith('#')
    ]
    headers = lines[0].split()
    rows = []
    for line in lines[1:]:
        values = line.split()
        row = {
            header: float('nan') if value.lower() == 'nan' else float(value)
            for header, value in zip(headers, values)
        }
        row['bond_factor'] = row['bond_length_A'] / _N2_REFERENCE_BOND_LENGTH_ANGSTROM
        rows.append(row)
    return rows


def _agent_method_for_factor(factor: float) -> str:
    """Executed methods for the archived adaptive N2 curve in the fixture."""
    if factor <= 1.0 + 1e-9:
        return 'ccsd_t'
    if factor <= 1.8 + 1e-9:
        return 'ccsd'
    return 'casscf'


def _n2_adaptive_report():
    cases = []
    comparison_table = []
    for index, row in enumerate(_benchmark_rows(), start=1):
        case_id = 'n2-{0:04d}'.format(index)
        method = _agent_method_for_factor(row['bond_factor'])
        cases.append({
            'case_id': case_id,
            'variables': {'bond_length_A': row['bond_length_A']},
        })
        comparison_table.append({
            'case_id': case_id,
            'label': 'R={0:.6f} A'.format(row['bond_length_A']),
            'status': 'succeeded',
            'bond_length_A': row['bond_length_A'],
            'final_energy': '{0:.12f} Ha'.format(row['agent_final_E_Ha']),
            'method': method,
        })
    return {
        'system_type': 'molecular',
        'cases': cases,
        'comparison_table': comparison_table,
    }


class N2ContextualVqeBenchmarkTests(unittest.TestCase):
    def test_benchmark_table_covers_literature_and_agent_curve(self):
        rows = _benchmark_rows()

        self.assertTrue(_BENCHMARK_PATH.is_file())
        self.assertEqual(len(rows), 37)
        self.assertTrue(any(math.isfinite(row['literature_FCI_E_Ha']) for row in rows))
        self.assertTrue(any(math.isfinite(row['literature_CASSCF_6_6_E_Ha']) for row in rows))
        self.assertTrue(all(math.isfinite(row['agent_final_E_Ha']) for row in rows))

    def test_agent_curve_matches_like_method_literature_references_in_overlap_window(self):
        rows = _benchmark_rows()
        ccsdt_errors_mha = [
            abs(row['agent_final_E_Ha'] - row['literature_CCSDT_E_Ha']) * 1000.0
            for row in rows
            if 0.8 - 1e-9 <= row['bond_factor'] <= 1.0 + 1e-9
            and math.isfinite(row['literature_CCSDT_E_Ha'])
        ]
        ccsd_errors_mha = [
            abs(row['agent_final_E_Ha'] - row['literature_CCSD_E_Ha']) * 1000.0
            for row in rows
            if 1.1 - 1e-9 <= row['bond_factor'] <= 1.8 + 1e-9
            and math.isfinite(row['literature_CCSD_E_Ha'])
        ]

        self.assertEqual(len(ccsdt_errors_mha), 3)
        self.assertEqual(len(ccsd_errors_mha), 8)
        self.assertLess(max(ccsdt_errors_mha), 0.25)
        self.assertLess(max(ccsd_errors_mha), 0.03)

    def test_real_n2_method_boundary_requires_local_casscf_review(self):
        report = _n2_adaptive_report()

        diagnostics = analyze_scan_path(report)

        self.assertEqual(diagnostics['status'], 'review_required')
        self.assertEqual(len(diagnostics['anomalies']), 1)
        anomaly = diagnostics['anomalies'][0]
        transition = anomaly['transition']
        target_rows = {
            row['case_id']: row
            for row in report['comparison_table']
            if row['case_id'] in anomaly['target_case_ids']
        }
        target_coordinates = sorted(row['bond_length_A'] for row in target_rows.values())
        self.assertEqual((transition['left_method'], transition['right_method']), ('ccsd', 'casscf'))
        self.assertEqual(anomaly['recommended_method'], 'casscf')
        self.assertEqual(target_coordinates, [1.866056, 1.975824])
        self.assertGreater(transition['energy_residual'], 0.005)

