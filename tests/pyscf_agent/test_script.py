from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Callable, Dict, List, Tuple

from pyscf_agent.backend import execution as execution_module
from pyscf_agent.backend import parsing as parsing_module
from pyscf_agent.backend import state as state_module
from pyscf_agent.backend import workflow as workflow_module
from pyscf_agent import contracts


TestResult = Dict[str, Any]
ModuleResult = Dict[str, Any]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _run_test(name: str, fn: Callable[[], Dict[str, Any]]) -> TestResult:
    try:
        details = fn()
        return {
            'name': name,
            'passed': True,
            'details': details,
        }
    except Exception as exc:  # pragma: no cover - test harness failure path
        return {
            'name': name,
            'passed': False,
            'details': {
                'error_type': type(exc).__name__,
                'message': str(exc),
                'traceback': traceback.format_exc(),
            },
        }


def _module_result(module_name: str, tests: List[TestResult]) -> ModuleResult:
    passed = all(test['passed'] for test in tests)
    return {
        'module': module_name,
        'passed': passed,
        'tests': tests,
    }


def test_models_symmetry_coercion() -> Dict[str, Any]:
    spec_false = contracts.task_spec_from_dict({'system': {'symmetry': 'false'}})
    spec_true = contracts.task_spec_from_dict({'system': {'symmetry': 'true'}})
    _assert(spec_false.system.symmetry is False, 'Expected symmetry="false" to coerce to False')
    _assert(spec_true.system.symmetry is True, 'Expected symmetry="true" to coerce to True')
    return {
        'symmetry_false': spec_false.system.symmetry,
        'symmetry_true': spec_true.system.symmetry,
    }


def test_state_defaults() -> Dict[str, Any]:
    state = state_module.default_state('hello')
    _assert(state['attempts'] == [], 'default_state should initialize empty attempts')
    _assert(state['errors'] == [], 'default_state should initialize empty errors')
    _assert(state['applied_defaults'] == [], 'default_state should initialize empty applied_defaults')
    _assert(state['execution_status'] == 'pending', 'default_state should initialize pending status')
    return {
        'execution_status': state['execution_status'],
        'attempts_len': len(state['attempts']),
        'errors_len': len(state['errors']),
    }


def test_parse_user_request_cases() -> Dict[str, Any]:
    cases: List[Tuple[str, Dict[str, Any]]] = [
        (
            'Calculate single point energy and dipole with b3lyp/6-31g* for water',
            {
                'basis': '6-31g*',
                'method': 'dft',
                'xc': 'b3lyp',
                'outputs': ['dipole', 'energy'],
            },
        ),
        (
            'Run a DFT single point with pbe0 and cc-pvdz on CO2',
            {
                'basis': 'cc-pvdz',
                'method': 'dft',
                'xc': 'pbe0',
                'outputs': ['energy'],
            },
        ),
        (
            'Do an hf single point with sto-3g on H2',
            {
                'basis': 'sto-3g',
                'method': 'hf',
                'xc': None,
                'outputs': ['energy'],
            },
        ),
    ]

    observed = []
    for request, expected in cases:
        parsed = parsing_module.parse_user_request(request)
        actual = {
            'basis': parsed.get('basis'),
            'method': parsed.get('method'),
            'xc': parsed.get('xc'),
            'outputs': parsed.get('outputs'),
        }
        _assert(actual == expected, 'Unexpected parse result for request: {0}'.format(request))
        observed.append({'request': request, 'actual': actual})
    return {'cases': observed}


def test_spec_validator_outputs() -> Dict[str, Any]:
    state = state_module.default_state('validator')
    state['task_spec'] = contracts.task_spec_to_dict(parsing_module.task_spec_from_partial({
        'method': 'dft',
        'outputs': ['dipole', 'analysis_text', 'analysis_text'],
    }))
    validated = parsing_module.spec_validator(state)

    error_codes = {error['code'] for error in validated['errors']}
    _assert({'missing_atom', 'missing_basis', 'missing_xc'}.issubset(error_codes), 'Missing required validation errors')
    _assert('unsupported_analysis_outputs' in error_codes, 'Expected unsupported analysis output error')
    _assert(len(validated['clarification_questions']) == len(set(validated['clarification_questions'])), 'Clarification questions should be deduplicated')
    _assert(validated['applied_defaults'], 'Applied defaults should be captured in state')
    return {
        'validation_errors': validated['validation_errors'],
        'error_codes': [error['code'] for error in validated['errors']],
        'clarification_questions': validated['clarification_questions'],
        'applied_defaults': validated['applied_defaults'],
    }


def test_generate_input_script_uses_shared_raw_stdout() -> Dict[str, Any]:
    script = execution_module.generate_input_script(contracts.TaskSpec(
        system=contracts.SystemSpec(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g'),
        method=contracts.MethodSpec(name='hf', restricted=True),
    ))
    _assert('from pyscf_agent.backend.execution import _run_pyscf_task' in script, 'Generated script should import the shared executor')
    _assert('from pyscf_agent.backend.script_helpers import _format_script_output' in script, 'Generated script should import the shared output formatter')
    _assert('def _extract_homo_lumo' not in script, 'Generated script should not inline helper definitions')
    _assert('result = _run_pyscf_task(task_spec)' in script, 'Generated script should delegate execution to the shared executor')
    _assert('print(_format_script_output(result))' in script, 'Generated script should delegate final printing to shared helper')
    return {
        'uses_helper_import': True,
        'inlines_helper_definitions': False,
        'uses_shared_executor': True,
        'uses_shared_output_formatter': True,
        'prints_via_helper': True,
    }


def test_runner_blocked_state() -> Dict[str, Any]:
    state = state_module.default_state('blocked')
    state['task_spec'] = {'method': 'dft'}
    validated = parsing_module.spec_validator(state)
    blocked = execution_module.runner(validated)

    _assert(blocked['execution_status'] == 'blocked', 'Runner should block execution when validation errors exist')
    _assert(len(blocked['attempts']) == 1, 'Blocked runner should record one attempt')
    _assert(blocked['attempts'][0]['stage'] == 'validation', 'Blocked attempt should be tagged as validation stage')
    return {
        'execution_status': blocked['execution_status'],
        'attempts_len': len(blocked['attempts']),
        'raw_stderr': blocked['raw_stderr'],
    }


def test_execute_request_blocked_report() -> Dict[str, Any]:
    report = workflow_module.execute_request('{"method": "dft"}', channel='test')['task_report']

    _assert(report['execution_status'] == 'blocked', 'execute_request should report blocked status for validation failures')
    _assert(report['attempts'], 'Blocked execute_request should include an attempt history entry')
    _assert(report['attempts'][0]['stage'] == 'validation', 'Blocked execute_request should record validation-stage attempt')
    _assert(report['raw_stderr'], 'Blocked execute_request should surface validation errors in raw_stderr')
    _assert(report['analysis_summary'].startswith('Task was not executed:'), 'Blocked execute_request should emit blocked summary text')
    return {
        'execution_status': report['execution_status'],
        'attempts_len': len(report['attempts']),
        'raw_stderr': report['raw_stderr'],
        'analysis_summary': report['analysis_summary'],
    }


def test_blocked_workflow_consistency() -> Dict[str, Any]:
    request = '{"method": "dft"}'

    sequential_state = workflow_module.run_workflow_sequential(state_module.default_state(request, channel='sequential-test'))
    sequential_report = sequential_state['task_report']

    comparison: Dict[str, Any] = {
        'sequential_status': sequential_report['execution_status'],
        'sequential_attempts_len': len(sequential_report['attempts']),
    }

    _assert(sequential_report['execution_status'] == 'blocked', 'Sequential workflow should report blocked status for validation failures')
    _assert(sequential_report['attempts'], 'Sequential workflow should record validation attempt history')
    _assert(sequential_report['attempts'][0]['stage'] == 'validation', 'Sequential workflow should record validation-stage attempt')
    _assert(sequential_report['raw_stderr'], 'Sequential workflow should surface validation errors in raw_stderr')
    _assert(sequential_report['analysis_summary'].startswith('Task was not executed:'), 'Sequential workflow should emit blocked summary text')

    try:
        workflow = workflow_module.build_workflow()
    except ImportError:
        comparison['langgraph_available'] = False
        return comparison

    graph_report = workflow_module.execute_request(request, channel='graph-test', workflow=workflow)['task_report']

    _assert(graph_report['execution_status'] == 'blocked', 'LangGraph workflow should report blocked status for validation failures')
    _assert(graph_report['attempts'], 'LangGraph workflow should record validation attempt history')
    _assert(graph_report['attempts'][0]['stage'] == 'validation', 'LangGraph workflow should record validation-stage attempt')
    _assert(graph_report['raw_stderr'] == sequential_report['raw_stderr'], 'Sequential and LangGraph workflows should agree on raw_stderr for blocked requests')
    _assert(graph_report['analysis_summary'] == sequential_report['analysis_summary'], 'Sequential and LangGraph workflows should agree on blocked analysis_summary')

    comparison.update({
        'langgraph_available': True,
        'graph_status': graph_report['execution_status'],
        'graph_attempts_len': len(graph_report['attempts']),
        'shared_raw_stderr': graph_report['raw_stderr'],
    })
    return comparison


def test_runner_records_execution_exception() -> Dict[str, Any]:
    state = state_module.default_state('failed')
    state['task_spec'] = contracts.task_spec_to_dict(contracts.TaskSpec(
        system=contracts.SystemSpec(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g'),
        method=contracts.MethodSpec(name='hf', restricted=True),
    ))
    state['errors'] = [state_module._make_error('validation', 'preexisting_validation', 'preexisting validation error')]
    state['validation_errors'] = []

    original_run = execution_module._run_pyscf_task
    try:
        def boom(task_spec: Any, **_kwargs: Any) -> Dict[str, Any]:
            raise RuntimeError('boom')

        execution_module._run_pyscf_task = boom
        failed = execution_module.runner(state)
    finally:
        execution_module._run_pyscf_task = original_run

    error_codes = [error['code'] for error in failed['errors']]
    _assert('preexisting_validation' in error_codes, 'Existing errors should be preserved')
    _assert('execution_exception' in error_codes, 'Execution error should be appended after monkeypatch failure')
    _assert(state_module._latest_error_message(failed, stage='execution') == 'PySCF execution failed: boom', 'Latest execution error message should reflect monkeypatched failure')
    return {
        'error_codes': error_codes,
        'latest_execution_error': state_module._latest_error_message(failed, stage='execution'),
        'attempts_len': len(failed['attempts']),
    }


def test_task_report_contents() -> Dict[str, Any]:
    state = state_module.default_state('report')
    state['task_spec'] = contracts.task_spec_to_dict(contracts.TaskSpec(
        system=contracts.SystemSpec(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g'),
        method=contracts.MethodSpec(name='hf', restricted=True),
    ))
    state['attempts'] = [{'index': 1, 'status': 'blocked'}]
    state['applied_defaults'] = [{'field': 'method.restricted', 'value': True, 'reason': 'inferred_from_spin'}]
    state['raw_scf_output'] = 'SCF\n'
    state['analysis_text'] = 'ANALYZE\n'
    state['raw_stdout'] = state_module._compose_raw_stdout(state['raw_scf_output'], state['analysis_text'])
    state['execution_status'] = 'succeeded'
    state['structured_results'] = {'converged': True, 'energy': -1.0}

    reported = execution_module.task_reporter(state)
    task_report = reported['task_report']

    _assert(task_report['attempts'] == state['attempts'], 'TaskReport should include attempts')
    _assert(task_report['applied_defaults'] == state['applied_defaults'], 'TaskReport should include applied_defaults')
    _assert(task_report['raw_scf_output'] == 'SCF\n', 'TaskReport should preserve raw_scf_output')
    _assert(task_report['analysis_text'] == 'ANALYZE\n', 'TaskReport should preserve analysis_text')
    _assert(task_report['raw_stdout'] == 'SCF\n[analysis_text]\nANALYZE\n', 'TaskReport should preserve composed raw_stdout')
    return {
        'attempts_len': len(task_report['attempts']),
        'applied_defaults': task_report['applied_defaults'],
        'raw_stdout': task_report['raw_stdout'],
    }


def main() -> int:
    module_results = [
        _module_result('contracts', [
            _run_test('task_spec_from_dict symmetry coercion', test_models_symmetry_coercion),
        ]),
        _module_result('backend.state', [
            _run_test('default_state initializes tracking fields', test_state_defaults),
            _run_test('state composes raw stdout', lambda: {
                'compose_result': state_module._compose_raw_stdout('SCF\n', 'ANALYZE\n'),
            }),
        ]),
        _module_result('backend.parsing', [
            _run_test('parse_user_request natural language cases', test_parse_user_request_cases),
            _run_test('spec_validator structured validation output', test_spec_validator_outputs),
        ]),
        _module_result('backend.execution', [
            _run_test('generate_input_script uses shared raw_stdout helper', test_generate_input_script_uses_shared_raw_stdout),
            _run_test('runner blocks invalid state', test_runner_blocked_state),
            _run_test('runner records execution exceptions', test_runner_records_execution_exception),
            _run_test('task_reporter preserves state snapshots', test_task_report_contents),
        ]),
        _module_result('backend.workflow', [
            _run_test('execute_request reports blocked validation state', test_execute_request_blocked_report),
            _run_test('blocked workflows stay consistent across orchestrators', test_blocked_workflow_consistency),
        ]),
    ]

    passed_tests = sum(1 for module in module_results for test in module['tests'] if test['passed'])
    failed_tests = sum(1 for module in module_results for test in module['tests'] if not test['passed'])
    all_passed = failed_tests == 0

    report = {
        'modules': module_results,
        'summary': {
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'all_passed': all_passed,
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
