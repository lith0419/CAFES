#!/usr/bin/env python

from __future__ import annotations

import argparse
import importlib.resources as importlib_resources
import json
import logging
import mimetypes
import os
import pathlib
import webbrowser
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional, Sequence
from urllib.parse import parse_qs, unquote, urlparse




from pyscf_agent.application import CalculationApplicationService, ExecutionTargetRegistry
from pyscf_agent.executors import LocalProcessExecutor, add_executor_arguments, create_task_executor_from_args
from pyscf_agent import request_builder as llm_request_builder
from pyscf_agent.web.api import (
    handle_capabilities_request,
    handle_active_space_probe_request,
    handle_execution_targets_request,
    handle_model_hamiltonian_edit_request,
    handle_model_hamiltonian_preview_request,
    handle_periodic_structure_preview_request,
    handle_prepare_request,
    handle_resource_profiles_request,
    handle_result_analysis_request,
    handle_task_review_action_request,
    handle_run_cancel_request,
    handle_run_collect_request,
    handle_run_request,
    handle_run_status_request,
    handle_task_view_request,
    handle_run_submit_request,
)
from pyscf_agent.web.ui import build_index_html
from computational_study_agent.web_api import (
    handle_saved_study_request,
    handle_example_study_request,
    handle_study_adaptive_plan_request,
    handle_study_adaptive_run_request,
    handle_study_execution_collect_request,
    handle_study_execution_status_request,
    handle_study_llm_draft_request,
    handle_study_plan_request,
    handle_study_artifact_request,
    handle_study_postprocess_request,
    handle_study_postprocess_suggestions_request,
    handle_study_result_analysis_request,
    handle_study_review_action_request,
    handle_study_run_request,
)
from computational_study_agent.web_ui import build_study_index_html
from computational_study_agent.application import (
    StudyApplicationService,
    set_study_application_services,
)


MODEL_HAMILTONIAN_UI_PACKAGE = 'model_hamiltonian_ui'
AGENT_WEB_ASSETS_PACKAGE = 'pyscf_agent.web_assets'
STUDY_WEB_ASSETS_PACKAGE = 'computational_study_agent.web_assets'


LOGGER = logging.getLogger(__name__)


def _safe_resource_parts(relative_path: str) -> Optional[Sequence[str]]:
    if not relative_path or relative_path == '.':
        return ('index.html',)
    candidate = pathlib.PurePosixPath(relative_path.replace('\\', '/'))
    if candidate.is_absolute():
        return None
    parts = candidate.parts
    if not parts or any(part in ('', '.', '..') for part in parts):
        return None
    return parts


def _read_model_hamiltonian_builder_asset(relative_path: str) -> Optional[tuple[str, bytes]]:
    parts = _safe_resource_parts(relative_path)
    if parts is None:
        return None
    try:
        asset = importlib_resources.files(MODEL_HAMILTONIAN_UI_PACKAGE)
    except ModuleNotFoundError:
        return None
    for part in parts:
        asset = asset.joinpath(part)
    if not asset.is_file():
        return None
    return '/'.join(parts), asset.read_bytes()


def _read_package_web_asset(package_name: str, relative_path: str) -> Optional[tuple[str, bytes]]:
    parts = _safe_resource_parts(relative_path)
    if parts is None:
        return None
    try:
        asset = importlib_resources.files(package_name)
    except ModuleNotFoundError:
        return None
    for part in parts:
        asset = asset.joinpath(part)
    if not asset.is_file():
        return None
    return '/'.join(parts), asset.read_bytes()


def _read_agent_web_asset(relative_path: str) -> Optional[tuple[str, bytes]]:
    return _read_package_web_asset(AGENT_WEB_ASSETS_PACKAGE, relative_path)


def _read_study_web_asset(relative_path: str) -> Optional[tuple[str, bytes]]:
    return _read_package_web_asset(STUDY_WEB_ASSETS_PACKAGE, relative_path)


def save_model_hamiltonian_input_file(
    python_input: str,
    output_path: pathlib.Path = None,
    *,
    work_dir: str = None,
    run_id: str = None,
) -> Dict[str, object]:
    return CalculationApplicationService().save_model_hamiltonian_input(
        python_input,
        output_path,
        work_dir=work_dir,
        run_id=run_id,
    )


class AgentWebHandler(BaseHTTPRequestHandler):
    task_executor = None
    execution_targets = None
    study_work_dir = None
    workbench_id = None

    def _write_response(self, status: HTTPStatus, headers: Dict[str, str], body: bytes) -> None:
        try:
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # Browsers routinely cancel stale asset and API requests during
            # navigation. The response is no longer deliverable, but the HTTP
            # service and any calculation behind it remain healthy.
            self.close_connection = True

    def _serve_model_hamiltonian_builder(self, path: str) -> None:
        if path == '/model-hamiltonian-builder':
            self.send_response(HTTPStatus.FOUND)
            self.send_header('Location', '/model-hamiltonian-builder/')
            self.end_headers()
            return

        route_prefix = '/model-hamiltonian-builder/'
        relative_path = unquote(path[len(route_prefix):]) or 'index.html'
        asset = _read_model_hamiltonian_builder_asset(relative_path)
        if asset is None:
            self._write_response(
                HTTPStatus.NOT_FOUND,
                {'Content-Type': 'text/plain; charset=utf-8'},
                b'Not Found',
            )
            return

        asset_path, body = asset
        content_type = mimetypes.guess_type(asset_path)[0] or 'application/octet-stream'
        if content_type.startswith('text/') or content_type in ('application/javascript', 'application/json'):
            content_type = '{0}; charset=utf-8'.format(content_type)
        self._write_response(HTTPStatus.OK, {'Content-Type': content_type}, body)

    def _serve_package_web_asset(self, path: str, route_prefix: str, reader) -> None:
        relative_path = unquote(path[len(route_prefix):])
        asset = reader(relative_path)
        if asset is None:
            self._write_response(
                HTTPStatus.NOT_FOUND,
                {'Content-Type': 'text/plain; charset=utf-8'},
                b'Not Found',
            )
            return
        asset_path, body = asset
        content_type = mimetypes.guess_type(asset_path)[0] or 'application/octet-stream'
        if content_type.startswith('text/') or content_type in ('application/javascript', 'application/json'):
            content_type = '{0}; charset=utf-8'.format(content_type)
        self._write_response(HTTPStatus.OK, {'Content-Type': content_type}, body)

    def _handle_save_model_hamiltonian_input(self, request_body: bytes) -> None:
        try:
            payload = json.loads(request_body.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        try:
            result = save_model_hamiltonian_input_file(
                payload.get('python_input'),
                work_dir=payload.get('work_dir'),
                run_id=payload.get('run_id'),
            )
            if isinstance(payload.get('target'), str):
                result['target'] = payload['target'].strip()
        except ValueError as exc:
            body = json.dumps({'status': 'error', 'error': str(exc)}, ensure_ascii=False).encode('utf-8')
            self._write_response(HTTPStatus.BAD_REQUEST, {'Content-Type': 'application/json; charset=utf-8'}, body)
            return
        except OSError as exc:
            body = json.dumps({'status': 'error', 'error': str(exc)}, ensure_ascii=False).encode('utf-8')
            self._write_response(HTTPStatus.INTERNAL_SERVER_ERROR, {'Content-Type': 'application/json; charset=utf-8'}, body)
            return

        body = json.dumps(result, ensure_ascii=False).encode('utf-8')
        self._write_response(HTTPStatus.OK, {'Content-Type': 'application/json; charset=utf-8'}, body)

    def do_GET(self) -> None:  # noqa: N802
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        if path == '/api/capabilities':
            status, headers, body = handle_capabilities_request()
            self._write_response(status, headers, body)
            return
        if path == '/api/workbench' and self.workbench_id:
            body = json.dumps({'workbench_id': self.workbench_id, 'pid': os.getpid()}).encode('utf-8')
            self._write_response(HTTPStatus.OK, {'Content-Type': 'application/json'}, body)
            return
        if path == '/api/execution-targets':
            status, headers, body = handle_execution_targets_request(self.execution_targets)
            self._write_response(status, headers, body)
            return
        if path == '/api/resource-profiles':
            query = parse_qs(parsed_url.query)
            execution_target = (query.get('execution_target') or [None])[0]
            status, headers, body = handle_resource_profiles_request(
                self.execution_targets,
                execution_target,
            )
            self._write_response(status, headers, body)
            return
        if path.startswith('/assets/'):
            self._serve_package_web_asset(path, '/assets/', _read_agent_web_asset)
            return
        if path.startswith('/computational-study/assets/'):
            self._serve_package_web_asset(
                path,
                '/computational-study/assets/',
                _read_study_web_asset,
            )
            return
        if path == '/model-hamiltonian-builder' or path.startswith('/model-hamiltonian-builder/'):
            self._serve_model_hamiltonian_builder(path)
            return
        if path == '/computational-study':
            self.send_response(HTTPStatus.FOUND)
            self.send_header('Location', '/computational-study/')
            self.end_headers()
            return
        if path == '/computational-study/':
            body = build_study_index_html(work_dir=self.study_work_dir).encode('utf-8')
            self._write_response(HTTPStatus.OK, {'Content-Type': 'text/html; charset=utf-8'}, body)
            return
        if path in ('/task-monitor', '/task-monitor/'):
            from pyscf_agent.task_monitor import task_monitor_html
            self._write_response(HTTPStatus.OK, {'Content-Type': 'text/html; charset=utf-8'},
                                 task_monitor_html().encode('utf-8'))
            return
        if path != '/':
            self._write_response(
                HTTPStatus.NOT_FOUND,
                {'Content-Type': 'text/plain; charset=utf-8'},
                b'Not Found',
            )
            return
        body = build_index_html(llm_request_builder_module=llm_request_builder).encode('utf-8')
        self._write_response(HTTPStatus.OK, {'Content-Type': 'text/html; charset=utf-8'}, body)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        content_length = int(self.headers.get('Content-Length', '0'))
        request_body = self.rfile.read(content_length)
        if path.startswith('/api/study-') and self.study_work_dir:
            try:
                payload = json.loads(request_body)
                if isinstance(payload, dict) and not payload.get('work_dir'):
                    payload['work_dir'] = self.study_work_dir
                    request_body = json.dumps(payload).encode('utf-8')
            except (ValueError, UnicodeDecodeError):
                pass  # The API handler provides the ordinary invalid-JSON response.
        if path == '/api/save-model-hamiltonian-input':
            self._handle_save_model_hamiltonian_input(request_body)
            return
        routes = {
            '/api/study-examples': partial(handle_example_study_request, action='list'),
            '/api/study-example-import': partial(handle_example_study_request, action='import'),
            '/api/study-list': partial(handle_saved_study_request, action='list'),
            '/api/study-open': partial(handle_saved_study_request, action='open'),
            '/api/study-start': partial(handle_saved_study_request, action='start'),
            '/api/study-stop': partial(handle_saved_study_request, action='stop'),
            '/api/study-retry-context': partial(handle_saved_study_request, action='retry-context'),
            '/api/study-retry-prepare': partial(handle_saved_study_request, action='retry-prepare'),
            '/api/study-retry-start': partial(handle_saved_study_request, action='retry-start'),
            '/api/study-dmet-branches-analyze': partial(handle_saved_study_request, action='dmet-branches-analyze'),
            '/api/study-dmet-continuation-prepare': partial(handle_saved_study_request, action='dmet-continuation-prepare'),
            '/api/study-dmet-continuation-start': partial(handle_saved_study_request, action='dmet-continuation-start'),
            '/api/study-prepare': partial(handle_saved_study_request, action='prepare'),
            '/api/capabilities': lambda _body: handle_capabilities_request(),
            '/api/study-plan': handle_study_plan_request,
            '/api/study-run': handle_study_run_request,
            '/api/study-adaptive-plan': handle_study_adaptive_plan_request,
            '/api/study-adaptive-run': handle_study_adaptive_run_request,
            '/api/study-execution-status': handle_study_execution_status_request,
            '/api/study-execution-collect': handle_study_execution_collect_request,
            '/api/study-llm-draft': handle_study_llm_draft_request,
            '/api/study-result-analysis': partial(handle_study_result_analysis_request, llm_request_builder=llm_request_builder),
            '/api/study-review-action': handle_study_review_action_request,
            '/api/study-postprocess': handle_study_postprocess_request,
            '/api/study-postprocess-suggestions': handle_study_postprocess_suggestions_request,
            '/api/study-artifact': handle_study_artifact_request,
            '/api/model-hamiltonian-preview': handle_model_hamiltonian_preview_request,
            '/api/model-hamiltonian-edit': partial(handle_model_hamiltonian_edit_request, llm_request_builder=llm_request_builder),
            '/api/periodic-structure-preview': handle_periodic_structure_preview_request,
            '/api/prepare': partial(handle_prepare_request, llm_request_builder=llm_request_builder),
            '/api/active-space-probe': handle_active_space_probe_request,
            '/api/analyze-result': partial(handle_result_analysis_request, llm_request_builder=llm_request_builder),
            '/api/task-review-action': handle_task_review_action_request,
            '/api/run-submit': partial(handle_run_submit_request, llm_request_builder=llm_request_builder, task_executor=self.task_executor, execution_targets=self.execution_targets),
            '/api/task-view': partial(handle_task_view_request, task_executor=self.task_executor, execution_targets=self.execution_targets, work_dir=self.study_work_dir),
            '/api/run-status': partial(handle_run_status_request, task_executor=self.task_executor, execution_targets=self.execution_targets),
            '/api/run-collect': partial(handle_run_collect_request, llm_request_builder=llm_request_builder, task_executor=self.task_executor, execution_targets=self.execution_targets),
            '/api/run-cancel': partial(handle_run_cancel_request, task_executor=self.task_executor, execution_targets=self.execution_targets),
            '/api/run': partial(handle_run_request, llm_request_builder=llm_request_builder, task_executor=self.task_executor, execution_targets=self.execution_targets),
        }
        handler = routes.get(path)
        if handler is None:
            self._write_response(HTTPStatus.NOT_FOUND, {'Content-Type': 'text/plain; charset=utf-8'}, b'Not Found')
            return
        status, headers, body = handler(request_body)
        self._write_response(status, headers, body)

    def log_message(self, format: str, *args: object) -> None:
        LOGGER.info(json.dumps({
            'level': 'info',
            'event': 'web.http_access',
            'details': {
                'client': self.address_string(),
                'path': self.path,
                'message': format % args,
            },
        }, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Serve the PySCF agent web frontend')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--work-dir', help='Study output root; use the same directory as the MCP server')
    parser.add_argument('--workbench-state', help=argparse.SUPPRESS)
    parser.add_argument('--workbench-id', help=argparse.SUPPRESS)
    parser.add_argument(
        '--no-open-browser',
        action='store_true',
        help='Do not open the web UI automatically after the server starts',
    )
    add_executor_arguments(parser)
    return parser


def serve(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        configured_executor = create_task_executor_from_args(args)
        local_executor = LocalProcessExecutor(wall_time_seconds=args.local_wall_time_seconds)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    # Interactive local runs use a child process so the Web UI can stop a
    # calculation without terminating the HTTP service itself.
    target_executors = {'local': local_executor}
    target_labels = {'local': 'Local'}
    default_target = 'local'
    if args.executor == 'remote':
        default_target = str(args.remote).strip().lower()
        target_executors[default_target] = configured_executor
        target_labels[default_target] = str(args.remote).strip().title()
    elif args.executor == 'slurm':
        default_target = 'slurm'
        target_executors[default_target] = configured_executor
        target_labels[default_target] = 'Slurm'
    execution_targets = ExecutionTargetRegistry(
        target_executors,
        default_target=default_target,
        labels=target_labels,
    )
    set_study_application_services(
        {
            target_id: StudyApplicationService(task_executor=executor, execution_config={
                'execution_target': 'local' if target_id == 'local' else args.executor,
                'slurm_config': args.slurm_config, 'slurm_profile': args.slurm_profile,
                'remote_config': args.remote_config, 'remote_profile': args.remote,
                'local_wall_time_seconds': args.local_wall_time_seconds,
            })
            for target_id, executor in execution_targets.services().items()
        },
        default_target=execution_targets.default_target,
    )
    task_executor = execution_targets.resolve()
    handler_class = type(
        'ConfiguredAgentWebHandler',
        (AgentWebHandler,),
        {
            'task_executor': task_executor,
            'execution_targets': execution_targets,
            'study_work_dir': str(pathlib.Path(args.work_dir).expanduser().resolve()) if args.work_dir else None,
            'workbench_id': args.workbench_id,
        },
    )
    server = ThreadingHTTPServer((args.host, args.port), handler_class)
    url = 'http://{0}:{1}'.format(args.host, server.server_port)
    if args.workbench_state:
        state_path = pathlib.Path(args.workbench_state)
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'port': server.server_port, 'pid': os.getpid()}), encoding='utf-8')
        temporary.replace(state_path)
    print('Serving on {0}'.format(url))
    if not args.no_open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(serve())
