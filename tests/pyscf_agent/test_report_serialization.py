"""Result output tolerates missing diagnostics without weakening input storage."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import numpy as np

from pyscf_agent.artifacts import default_artifact_repository
from pyscf_agent.artifacts.arrays import compact_request
from pyscf_agent.backend.execution import task_reporter
from pyscf_agent.backend.state import append_log, default_state
from pyscf_agent.contracts import TaskReport, TaskSpec, task_spec_to_dict
from pyscf_agent.serialization import canonical_json, sanitize_result_json
from pyscf_agent.verification import write_verification_report


class ReportSerializationTests(unittest.TestCase):
    def test_nested_nonfinite_values_have_exact_json_pointer_paths(self):
        original = {'a/b~c': (np.array([[1., np.nan], [np.inf, -np.inf]]),),
                    'count': np.int64(5), 'flag': np.bool_(True)}
        cleaned, warnings = sanitize_result_json(original)
        self.assertEqual(cleaned, {'a/b~c': [[[1., None], [None, None]]],
                                   'count': 5, 'flag': True})
        self.assertEqual([(w['path'], w['value']) for w in warnings], [
            ('/a~1b~0c/0/0/1', 'NaN'), ('/a~1b~0c/0/1/0', 'Infinity'),
            ('/a~1b~0c/0/1/1', '-Infinity')])
        self.assertTrue(np.isnan(original['a/b~c'][0][0, 1]))
        self.assertEqual(sanitize_result_json(cleaned), (cleaned, []))
        self.assertEqual(json.loads(canonical_json(cleaned)), cleaned)
        for value in (object(), np.array([1j])):
            with self.subTest(value=type(value)), self.assertRaises(TypeError):
                sanitize_result_json(value)

    def test_report_preserves_finite_results_existing_warnings_and_status(self):
        with tempfile.TemporaryDirectory() as root:
            state = default_state('report', work_dir=root)
            state.update(task_spec=task_spec_to_dict(TaskSpec()), execution_status='unconverged',
                         warnings=[{'code': 'solver_warning', 'message': 'Keep me'}],
                         structured_results={'energy': -1.5, 'diagnostic': np.float32(np.nan)},
                         attempts=[{'residual': float('inf')}])
            report = task_reporter(state)['task_report']
            self.assertEqual(report['structured_results'], {'energy': -1.5, 'diagnostic': None})
            self.assertEqual(report['execution_status'], 'unconverged')
            self.assertEqual(report['warnings'][0], state['warnings'][0])
            self.assertEqual({w['path'] for w in report['warnings'][1:]},
                             {'/structured_results/diagnostic', '/attempts/0/residual'})
            self.assertTrue(np.isnan(state['structured_results']['diagnostic']))
            self.assertEqual(TaskReport.from_dict(report).to_dict(), report)
            again = task_reporter(report)['task_report']
            self.assertEqual(again['warnings'], report['warnings'])
            path = Path(root) / 'report.json'
            default_artifact_repository().write_json(path, report, kind='task-report')
            self.assertEqual(json.loads(path.read_text()), report)

    def test_inputs_and_unknown_report_types_still_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as root:
            state = default_state('report', work_dir=root)
            state['task_spec'] = task_spec_to_dict(TaskSpec())
            state['task_spec']['runtime']['conv_tol'] = float('nan')
            with self.assertRaises(ValueError):
                task_reporter(state)
            with self.assertRaises(ValueError):
                compact_request('{"threshold": NaN}', Path(root))
            state['task_spec']['runtime']['conv_tol'] = None
            state['structured_results'] = {'unsupported': object()}
            with self.assertRaises(TypeError):
                task_reporter(state)

    def test_numpy_nonfinite_log_does_not_abort_and_keeps_evidence(self):
        state = {}
        with patch('pyscf_agent.backend.state.LOGGER') as logger:
            append_log(state, 'info', 'diagnostic', {'x': np.float32(np.nan)})
        output = json.loads(logger.log.call_args.args[1])
        self.assertIsNone(output['details']['x'])
        self.assertEqual(output['warnings'][0]['path'], '/details/x')
        self.assertTrue(np.isnan(state['logs'][0]['details']['x']))

    def test_late_workflow_evidence_cannot_reintroduce_nonfinite_values(self):
        from pyscf_agent.backend.module_runtime import compiled_task_reporter
        from pyscf_agent.backend.gate_runtime import compiled_gate_before_finalize

        def finalize(state, stage):
            state = task_reporter(state)
            # The dispatcher records evidence after task_reporter returns.
            state['module_execution_trace'] = [{'residual': np.float32(np.inf)}]
            state['module_runtime_observations'] = [{'variance': float('nan')}]
            append_log(state, 'info', 'late_diagnostic', {'x': float('nan')})
            return state

        with tempfile.TemporaryDirectory() as root:
            state = default_state('report', work_dir=root)
            state.update(task_spec=task_spec_to_dict(TaskSpec()),
                         workflow_configuration={'workflow_id': 'test'},
                         gate_configuration={'gate_set_id': 'test'},
                         gate_execution_trace=[{'residual': float('inf')}])
            with patch('pyscf_agent.backend.gate_runtime.run_task_gate_hook',
                       side_effect=lambda state, hook: state):
                state = compiled_gate_before_finalize(state)
            with patch('pyscf_agent.backend.module_runtime.run_compiled_stage', side_effect=finalize):
                report = compiled_task_reporter(state)['task_report']
            canonical_json(report)
            self.assertIsNone(report['module_execution_trace'][0]['residual'])
            self.assertIsNone(report['module_runtime_observations'][0]['variance'])
            self.assertIsNone(report['logs'][-1]['details']['x'])
            self.assertTrue(any(w['path'] == '/entries/0/residual' and
                                w.get('artifact_path', '').endswith('workflow-execution-trace.json')
                                for w in report['warnings']))
            self.assertTrue(any(w['path'] == '/entries/0/residual' and
                                w.get('artifact_path', '').endswith('gate-execution-trace.json')
                                for w in report['warnings']))

    def test_cli_and_verification_use_strict_numpy_aware_json(self):
        from pyscf_agent.cli import main
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'verification.json'
            write_verification_report({'count': np.int64(5)}, str(path))
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                write_verification_report({'diagnostic': float('nan')}, str(path))
            self.assertEqual(path.read_bytes(), before)
        for output_format in ('json', 'pretty'):
            with self.subTest(format=output_format), \
                    patch('pyscf_agent.cli.CalculationApplicationService') as service, \
                    patch('pyscf_agent.cli.load_request', return_value='test'), \
                    patch('sys.stdout', new_callable=io.StringIO) as stdout:
                service.return_value.execute_request.return_value = {'count': np.int64(5)}
                self.assertEqual(main(['--format', output_format]), 0)
                self.assertEqual(json.loads(stdout.getvalue()), {'count': 5})
                service.return_value.execute_request.return_value = {'bad': float('inf')}
                with self.assertRaises(ValueError):
                    main(['--format', output_format])

    def test_child_keeps_dmrg_result_report_and_binary_arrays(self):
        # Exercise the real runner -> early result artifacts -> reporter ->
        # strict persistence boundary in a child, replacing only the solver.
        script = textwrap.dedent('''
            import sys
            from pathlib import Path
            from unittest.mock import patch
            import numpy as np
            from pyscf_agent.backend.execution import runner, task_reporter
            from pyscf_agent.backend.state import default_state
            from pyscf_agent.contracts import TaskSpec, task_spec_to_dict
            from pyscf_agent.artifacts import default_artifact_repository

            state = default_state('synthetic result', work_dir=sys.argv[1], run_id='test')
            state['task_spec'] = task_spec_to_dict(TaskSpec())
            results = {'converged': True, 'energy': -1.5,
                       'dmrg_result': {'energy': -1.5, 'variance': np.float64(np.nan)},
                       '_transient_dmrg_arrays': {'occupations': np.array([1., np.nan])}}
            with patch('pyscf_agent.backend.execution._run_pyscf_task', return_value=results):
                state = runner(state)
            report = task_reporter(state)['task_report']
            default_artifact_repository().write_json(Path(sys.argv[1]) / 'report.json', report,
                                                      kind='task-report')
        ''')
        with tempfile.TemporaryDirectory() as root:
            process = subprocess.run([sys.executable, '-c', script, root],
                                     capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads((Path(root) / 'report.json').read_text())
            self.assertEqual(report['execution_status'], 'succeeded')
            artifacts = {item['kind']: item for item in report['artifacts']}
            result_path = artifacts['block2_dmrg_result']['path']
            dmrg = json.loads(Path(result_path).read_text())
            self.assertEqual(dmrg['energy'], -1.5)
            self.assertIsNone(dmrg['variance'])
            self.assertTrue(any(w.get('artifact_path') == result_path and w['path'] == '/variance'
                                for w in report['warnings']))
            full = json.loads(Path(artifacts['structured_results']['path']).read_text())
            self.assertEqual(full['energy'], -1.5)
            self.assertIsNone(full['dmrg_result']['variance'])
            with np.load(artifacts['block2_dmrg_arrays']['path']) as arrays:
                self.assertEqual(arrays['occupations'][0], 1.)
                self.assertTrue(np.isnan(arrays['occupations'][1]))
