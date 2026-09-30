from http import HTTPStatus
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pyscf_agent.executors.commands import CommandTimeoutError, run_bounded
from pyscf_agent.executors.slurm import SlurmExecutor, _default_command_runner
from pyscf_agent.executors.ssh_slurm import _default_ssh_runner
from pyscf_agent.remote.deployment import _default_ssh_runner as deploy_runner
from pyscf_agent.transport.json import api_exception_response


class ErrorBoundaryTests(unittest.TestCase):
    def test_remote_timeout_keeps_receipt_recoverable(self):
        from computational_study_agent.execution_receipts import inspect_receipt
        from tests.pyscf_agent.test_ssh_slurm_executor import _job

        executor = SimpleNamespace(status=Mock(side_effect=CommandTimeoutError('query timed out')))
        receipt = {'executor': {}, 'status': 'running', 'jobs': {'case-0001': _job().to_dict()}}
        result = inspect_receipt(receipt, executor)
        self.assertEqual(result['status'], 'connection_interrupted')
        self.assertEqual(result['status_summary']['terminal'], 0)
        self.assertEqual(result['status_summary']['unreadable'][0]['case_id'], 'case-0001')
        self.assertEqual(receipt['status'], 'running')

    def test_ssh_keepalive_and_sftp_timeout(self):
        from pyscf_agent.executors import SshSlurmExecutor
        from tests.pyscf_agent.test_ssh_slurm_executor import _profile

        executor = SshSlurmExecutor(_profile(), command_runner=Mock(
            side_effect=subprocess.TimeoutExpired(['sftp'], 60)))
        for command in (executor._ssh_command('status'), executor._sftp_command()):
            self.assertIn('ServerAliveInterval=15', command)
            self.assertIn('ServerAliveCountMax=3', command)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(CommandTimeoutError):
                executor.collect_files({'/cluster/work/result.npz': str(Path(directory) / 'result.npz')})

    def test_default_control_commands_are_bounded(self):
        for runner, args, expected in ((_default_command_runner, (['squeue'],), 60),
                                       (_default_ssh_runner, (['ssh'], '{}'), 60),
                                       (deploy_runner, (['ssh'], b'archive'), 300)):
            with self.subTest(runner=runner.__module__), patch('subprocess.run') as run:
                runner(*args)
                self.assertEqual(run.call_args.kwargs['timeout'], expected)
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired(['ssh'], 60)):
            with self.assertRaisesRegex(CommandTimeoutError, 'temporarily unreadable'):
                run_bounded(['ssh'], capture_output=True)

    def test_bulk_remote_operations_have_a_separate_time_budget(self):
        from pyscf_agent.executors import SshSlurmExecutor
        from tests.pyscf_agent.test_ssh_slurm_executor import _profile
        executor = SshSlurmExecutor(_profile())
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired(['ssh'], 1800)) as run:
            with self.assertRaises(CommandTimeoutError):
                executor.generate_hamiltonian_dataset({})
            self.assertEqual(run.call_args.kwargs['timeout'], 1800)
        with tempfile.TemporaryDirectory() as directory:
            with patch('subprocess.run', side_effect=subprocess.TimeoutExpired(['sftp'], 1800)) as run:
                with self.assertRaises(CommandTimeoutError):
                    executor.collect_files({'/cluster/work/data.npz': str(Path(directory) / 'data.npz')})
                self.assertEqual(run.call_args.kwargs['timeout'], 1800)

    def test_injected_scheduler_timeout_does_not_become_job_failure(self):
        executor = object.__new__(SlurmExecutor)
        executor._command_runner = Mock(side_effect=subprocess.TimeoutExpired(['squeue'], 10))
        with self.assertRaises(CommandTimeoutError):
            executor._run_command(['squeue'], operation='squeue')

    def test_api_bugs_log_traceback_and_hide_details(self):
        logger = logging.getLogger('boundary-test')
        with self.assertLogs(logger, level='ERROR') as records:
            try:
                raise KeyError('internal-secret-field')
            except KeyError as exc:
                status, _, body = api_exception_response(exc, logger)
        self.assertEqual(status, HTTPStatus.INTERNAL_SERVER_ERROR)
        self.assertNotIn(b'internal-secret-field', body)
        self.assertIn('internal-secret-field', '\n'.join(records.output))
        self.assertEqual(api_exception_response(ValueError('bad input'), logger)[0], HTTPStatus.BAD_REQUEST)
        with self.assertLogs(logger, level='WARNING'):
            self.assertEqual(api_exception_response(CommandTimeoutError('timeout'), logger)[0], HTTPStatus.SERVICE_UNAVAILABLE)

    def test_both_web_surfaces_return_500_for_internal_key_error(self):
        from computational_study_agent import web_api as study_api
        from pyscf_agent.web import api
        for module in (study_api, api):
            with self.subTest(module=module.__name__), self.assertLogs(module.LOGGER, level='ERROR'):
                status, _, body = module._handle_error(KeyError('implementation detail'))
                self.assertEqual(status, HTTPStatus.INTERNAL_SERVER_ERROR)
                self.assertNotIn('implementation detail', json.loads(body)['error'])

    def test_busy_analysis_is_a_conflict_not_invalid_input(self):
        from computational_study_agent import web_api
        from computational_study_agent.execution_receipts import StudyExecutionBusy
        service = Mock()
        service.analyze_study.side_effect = StudyExecutionBusy('already running')
        with patch.object(web_api, 'get_study_application_service', return_value=service):
            status, _, _ = web_api.handle_study_result_analysis_request(
                b'{"study_id": "test"}', llm_request_builder=Mock())
        self.assertEqual(status, HTTPStatus.CONFLICT)

    def test_solver_setting_failure_is_not_ignored(self):
        from pyscf_agent.backend.correlation.cas_execution import configure_fci_solver
        class ReadOnly:
            @property
            def nroots(self):
                return 1
        with self.assertRaisesRegex(ValueError, 'nroots=2'):
            configure_fci_solver(ReadOnly(), 2)

    def test_structured_output_only_falls_back_on_capability_errors(self):
        from pyscf_agent.request_builder.llm import LLMHTTPError, _should_retry_without_structured_output
        detail = json.dumps({'error': {'param': 'response_format', 'code': 'unsupported_parameter'}})
        with patch.dict(os.environ, {'PYSCF_AGENT_LLM_STRUCTURED_OUTPUT': 'auto'}):
            self.assertTrue(_should_retry_without_structured_output(LLMHTTPError(400, detail)))
            self.assertTrue(_should_retry_without_structured_output(RuntimeError('LLM HTTP error 400: unsupported response_format json_schema')))
            unavailable = json.dumps({'error': {
                'message': 'This response_format type is unavailable now (request_id: test)',
                'type': 'invalid_request_error', 'param': None, 'code': 'invalid_request_error',
            }})
            self.assertTrue(_should_retry_without_structured_output(LLMHTTPError(400, unavailable)))
            for error in (KeyError('response_format'), TimeoutError('structured output timeout'),
                          LLMHTTPError(500, unavailable), LLMHTTPError(429, unavailable),
                          LLMHTTPError(400, json.dumps({'error': {'code': 'invalid_request_error',
                              'message': 'unsupported keyword in response_format schema'}})),
                          LLMHTTPError(400, json.dumps({'error': {'code': 'invalid_request_error',
                              'message': 'The model is unavailable now'}})),
                          LLMHTTPError(500, detail), LLMHTTPError(400, 'response_format schema validation failed'),
                          LLMHTTPError(400, json.dumps({'error': {'code': 'invalid_json_schema',
                              'param': 'response_format', 'message': 'unsupported keyword in response_format'}}))):
                self.assertFalse(_should_retry_without_structured_output(error))
        with patch.dict(os.environ, {'PYSCF_AGENT_LLM_STRUCTURED_OUTPUT': 'false'}):
            self.assertFalse(_should_retry_without_structured_output(LLMHTTPError(400, detail)))
        with patch.dict(os.environ, {'PYSCF_AGENT_LLM_STRUCTURED_OUTPUT': 'true'}):
            self.assertFalse(_should_retry_without_structured_output(LLMHTTPError(400, unavailable)))

    def test_declared_no_structured_output_avoids_capability_probe(self):
        from pyscf_agent.request_builder.llm import _call_llm
        post = Mock(return_value={'choices': [{'message': {'content': '{"task_spec": {}}'}}]})
        with patch.dict(os.environ, {'PYSCF_AGENT_LLM_STRUCTURED_OUTPUT': 'false',
                                    'PYSCF_AGENT_LLM_BASE_URL': 'https://invalid.example/v1',
                                    'PYSCF_AGENT_LLM_MODEL': 'test'}):
            _call_llm([], {}, http_post=post)
        self.assertEqual(post.call_count, 1)
        self.assertNotIn('response_format', post.call_args.args[1])
