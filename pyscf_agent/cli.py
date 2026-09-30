#!/usr/bin/env python

from __future__ import annotations

from pyscf_agent.serialization import json_default

import argparse
import json
import pathlib
from typing import Any, Dict, List, Optional, Sequence

from .application import CalculationApplicationService
from .executors import (
    JobHandle,
    add_executor_arguments,
    create_task_executor_from_args,
)
from .backend.workflow import example_request
from .registry.platform import result_analysis_output_contracts


def _format_value(value: Any) -> str:
    if value is None:
        return 'None'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=json_default, allow_nan=False)
    return str(value)


def _append_section(lines: List[str], title: str, items: List[str]) -> None:
    if not items:
        return
    lines.append(title)
    lines.extend(items)
    lines.append('')


def format_human_report(
    report: Dict[str, Any],
    *,
    show_generated_input: bool = False,
    show_raw_output: bool = False,
    show_messages: bool = False,
    show_logs: bool = False,
) -> str:
    lines: List[str] = []
    task_spec = report.get('task_spec') or {}
    system = task_spec.get('system') or {}
    method = task_spec.get('method') or {}
    analysis = task_spec.get('analysis') or {}
    runtime = task_spec.get('runtime') or {}
    structured_results = report.get('structured_results') or {}
    status = report.get('execution_status') or 'unknown'

    overview = [
        'Status: {0}'.format(status),
    ]
    analysis_summary = report.get('analysis_summary')
    if analysis_summary:
        overview.append('Summary: {0}'.format(analysis_summary))
    overview.append('Retries: {0} / {1}'.format(report.get('retry_count', 0), report.get('max_retries', 0)))
    overview.append('Attempts: {0}'.format(len(report.get('attempts', []))))
    _append_section(lines, '== Overview ==', overview)

    spec_lines = []
    if system:
        spec_lines.append('System: {0}'.format(system.get('atom') or ''))
        spec_lines.append('Basis: {0}'.format(system.get('basis') or ''))
        spec_lines.append('Unit: {0}'.format(system.get('unit') or ''))
        spec_lines.append('Charge / spin: {0} / {1}'.format(system.get('charge', 0), system.get('spin', 0)))
        spec_lines.append('Symmetry: {0}'.format(_format_value(system.get('symmetry', False))))
    if method:
        method_label = method.get('name', '')
        if method.get('xc'):
            method_label = '{0} / {1}'.format(method_label, method['xc'])
        spec_lines.append('Method: {0}'.format(method_label))
        spec_lines.append('restricted: {0}'.format(_format_value(method.get('restricted'))))
    if analysis:
        spec_lines.append('Analysis outputs: {0}'.format(', '.join(analysis.get('outputs') or [])))
    if runtime:
        spec_lines.append('max_cycle: {0}'.format(runtime.get('max_cycle')))
        spec_lines.append('conv_tol: {0}'.format(_format_value(runtime.get('conv_tol'))))
        spec_lines.append('verbose: {0}'.format(runtime.get('verbose')))
    _append_section(lines, '== Task Parameters ==', spec_lines)

    result_lines = []
    result_keys = ['converged']
    for key in result_analysis_output_contracts().keys():
        if key not in result_keys:
            result_keys.append(key)
    for key in result_keys:
        if key in structured_results:
            result_lines.append('{0}: {1}'.format(key, _format_value(structured_results.get(key))))
    if result_lines:
        _append_section(lines, '== Structured Results ==', result_lines)

    default_lines = [
        '{0} = {1} ({2})'.format(item.get('field'), _format_value(item.get('value')), item.get('reason'))
        for item in report.get('applied_defaults', [])
    ]
    _append_section(lines, '== Applied Defaults ==', default_lines)

    validation_lines = ['- {0}'.format(message) for message in report.get('validation_errors', [])]
    _append_section(lines, '== Validation Errors ==', validation_lines)

    question_lines = ['- {0}'.format(question) for question in report.get('clarification_questions', [])]
    _append_section(lines, '== Required Input ==', question_lines)

    error_lines = []
    for error in report.get('errors', []):
        error_lines.append(
            '- [{0}/{1}] {2}'.format(
                error.get('stage', 'unknown'),
                error.get('code', 'unknown'),
                error.get('message', ''),
            )
        )
    _append_section(lines, '== Structured Errors ==', error_lines)

    attempt_lines = []
    for attempt in report.get('attempts', []):
        parts = [
            '#{0}'.format(attempt.get('index', '?')),
            'status={0}'.format(attempt.get('status', 'unknown')),
            'stage={0}'.format(attempt.get('stage', 'unknown')),
            'retry={0}'.format(attempt.get('retry_count', 0)),
        ]
        if attempt.get('timestamp'):
            parts.append('time={0}'.format(attempt['timestamp']))
        if attempt.get('errors'):
            parts.append(
                'errors={0}'.format(', '.join(
                    item.get('code') or item.get('message', 'unknown')
                    for item in attempt.get('errors', [])
                ))
            )
        attempt_lines.append('- ' + ' | '.join(parts))
    _append_section(lines, '== Attempt History ==', attempt_lines)

    if show_messages:
        message_lines = [
            '- [{0}/{1}] {2}'.format(item.get('role', 'unknown'), item.get('kind', 'message'), item.get('content', ''))
            for item in report.get('messages', [])
        ]
        _append_section(lines, '== Messages ==', message_lines)

    if show_generated_input and report.get('generated_input'):
        _append_section(lines, '== PySCF Input Preview ==', [report['generated_input']])

    if show_raw_output:
        raw_lines = []
        if report.get('raw_stdout'):
            raw_lines.append('[raw_stdout]\n{0}'.format(report['raw_stdout']))
        if report.get('raw_scf_output'):
            raw_lines.append('[raw_scf_output]\n{0}'.format(report['raw_scf_output']))
        if report.get('analysis_text'):
            raw_lines.append('[analysis_text]\n{0}'.format(report['analysis_text']))
        if report.get('raw_stderr'):
            raw_lines.append('[raw_stderr]\n{0}'.format(report['raw_stderr']))
        _append_section(lines, '== Raw Output ==', raw_lines)

    if show_logs:
        log_lines = [json.dumps(item, ensure_ascii=False, default=json_default, allow_nan=False) for item in report.get('logs', [])]
        _append_section(lines, '== Logs ==', log_lines)

    return '\n'.join(lines).strip() + '\n'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run the PySCF agent from the command line')
    parser.add_argument('request', nargs='?', help='User request text or JSON payload')
    parser.add_argument('--request-file', help='Path to a text file containing the request')
    parser.add_argument('--channel', default='cli', help='Structured message channel label')
    parser.add_argument('--format', choices=('human', 'json', 'pretty'), default='human', help='Output format')
    parser.add_argument('--pretty', action='store_true', help='Pretty-print the JSON response')
    parser.add_argument('--show-generated-input', action='store_true', help='Include generated PySCF input in human output')
    parser.add_argument('--show-raw-output', action='store_true', help='Include raw stdout/stderr fields in human output')
    parser.add_argument('--show-messages', action='store_true', help='Include workflow messages in human output')
    parser.add_argument('--show-logs', action='store_true', help='Include workflow logs in human output')
    parser.add_argument(
        '--submit',
        action='store_true',
        help='Submit through the JobHandle lifecycle and print the handle plus current status',
    )
    parser.add_argument(
        '--job-action',
        choices=('status', 'fetch', 'logs', 'artifacts', 'cancel'),
        help='Operate on a previously returned JobHandle',
    )
    parser.add_argument(
        '--job-handle',
        help='JobHandle JSON or a path to a JSON file; required with --job-action',
    )
    parser.add_argument('--work-dir', help='Execution root for a submitted or synchronous task')
    parser.add_argument('--run-id', help='Explicit run/job id')
    parser.add_argument(
        '--check-executor',
        action='store_true',
        help='Validate the selected execution target and print its metadata.',
    )
    add_executor_arguments(parser)
    return parser


def load_request(args: argparse.Namespace) -> str:
    if args.request_file:
        return pathlib.Path(args.request_file).read_text(encoding='utf-8')
    if args.request:
        return args.request
    return example_request()


def load_job_handle(value: str) -> JobHandle:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('JobHandle JSON or file path is required')
    normalized = value.strip()
    if normalized.startswith('{'):
        payload = json.loads(normalized)
    else:
        payload = json.loads(pathlib.Path(normalized).read_text(encoding='utf-8'))
    if isinstance(payload, dict) and isinstance(payload.get('handle'), dict):
        payload = payload['handle']
    return JobHandle.from_dict(payload)


def _print_json(payload: Any, *, pretty: bool) -> None:
    print(json.dumps(
        payload,
        ensure_ascii=False,
        indent=2 if pretty else None,
        default=json_default, allow_nan=False,
    ))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        task_executor = create_task_executor_from_args(args)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    service = CalculationApplicationService(task_executor=task_executor)
    if args.check_executor:
        payload = {'client': task_executor.describe()}
        remote_capabilities = getattr(task_executor, 'remote_capabilities', None)
        try:
            if callable(remote_capabilities):
                payload['server'] = remote_capabilities()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        _print_json(payload, pretty=True)
        return 0
    if args.job_action:
        if not args.job_handle:
            parser.error('--job-handle is required with --job-action')
        handle = load_job_handle(args.job_handle)
        if args.job_action == 'status':
            payload = service.job_status(handle).to_dict()
        elif args.job_action == 'fetch':
            payload = service.fetch_job(handle)
        elif args.job_action == 'logs':
            payload = service.job_logs(handle)
        elif args.job_action == 'artifacts':
            payload = service.job_artifacts(handle)
        else:
            payload = service.cancel_job(handle).to_dict()
        _print_json(payload, pretty=args.pretty or args.format == 'pretty')
        return 0

    request_text = load_request(args)
    if args.submit:
        handle = service.submit_request(
            request_text,
            channel=args.channel,
            work_dir=args.work_dir,
            run_id=args.run_id,
        )
        _print_json(
            {
                'handle': handle.to_dict(),
                'status': service.job_status(handle).to_dict(),
            },
            pretty=args.pretty or args.format == 'pretty',
        )
        return 0

    report = service.execute_request(
        request_text,
        channel=args.channel,
        work_dir=args.work_dir,
        run_id=args.run_id,
        include_llm_feedback=False,
    )
    output_format = 'pretty' if args.pretty else args.format
    if output_format == 'pretty':
        print(json.dumps(report, indent=2, ensure_ascii=False, default=json_default, allow_nan=False))
    elif output_format == 'json':
        print(json.dumps(report, ensure_ascii=False, default=json_default, allow_nan=False))
    else:
        print(format_human_report(
            report,
            show_generated_input=args.show_generated_input,
            show_raw_output=args.show_raw_output,
            show_messages=args.show_messages,
            show_logs=args.show_logs,
        ), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
