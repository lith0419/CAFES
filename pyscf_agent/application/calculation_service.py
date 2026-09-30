from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..backend.artifacts import ensure_run_id, get_run_dir
from pyscf_agent.paths import resolve_work_dir
from ..backend.model_hamiltonian.operations import apply_model_operations, describe_model_changes
from ..backend.model_hamiltonian.solver import (
    generate_model_hamiltonian_input_script,
    load_model_spec_from_file,
    normalize_model_spec,
)
from ..backend.model_hamiltonian.spec import validate_model_hamiltonian_spec
from ..backend.periodic.solver import periodic_structure_summary
from ..contracts import TaskSpec, task_spec_from_dict, task_spec_to_dict
from ..backend.active_space_probe import build_molecular_active_space_probe_request
from ..backend.module_runtime import validate_compiled_runtime
from ..backend.gate_runtime import validate_compiled_gates
from ..registry import default_registry
from ..registry.platform import public_registry_payload
from ..executors import JobHandle, JobStatus, TaskExecutor
from ..workflow_modules import compile_task_workflow
from ..workflow_gates import compile_task_gates
from ..lifecycle import (
    ensure_lifecycle,
    lifecycle_requires_review,
    submit_task_lifecycle,
    task_lifecycle_for_preparation,
    task_lifecycle_from_job_status,
    transition_lifecycle,
)


MAX_MODEL_HAMILTONIAN_INPUT_BYTES = 5 * 1024 * 1024


class CalculationApplicationError(Exception):
    """Base error for calculation-assistant use cases."""


class CalculationFeatureUnavailableError(CalculationApplicationError):
    """Raised when an optional LLM or execution dependency is unavailable."""


class CalculationApplicationService:
    """Protocol-neutral application boundary for single calculation tasks."""

    def __init__(
        self,
        *,
        task_executor: Optional[TaskExecutor] = None,
        llm_request_builder: Any = None,
        capability_provider: Callable[[], Dict[str, Any]] = public_registry_payload,
        model_spec_loader: Callable[[str], Dict[str, Any]] = load_model_spec_from_file,
        periodic_preview_builder: Callable[..., Dict[str, Any]] = periodic_structure_summary,
    ):
        self._task_executor = task_executor
        self._llm_request_builder = llm_request_builder
        self._capability_provider = capability_provider
        self._model_spec_loader = model_spec_loader
        self._periodic_preview_builder = periodic_preview_builder

    def capabilities(self) -> Dict[str, Any]:
        return self._capability_provider()

    def validate_task_spec(self, task_spec: Any, *, locale: str = 'en') -> Dict[str, Any]:
        """Normalize and validate structured input without an LLM or calculation.

        Use the same builder, validator and compiler as task execution. Both
        the public nested TaskSpec and existing shorthand requests are accepted.
        Passing this check does not grant any runtime scientific approval.
        """
        from ..backend.parsing import spec_builder, spec_validator
        from ..backend.state import default_state

        if isinstance(task_spec, TaskSpec):
            payload = task_spec_to_dict(task_spec)
        elif isinstance(task_spec, dict):
            payload = copy.deepcopy(task_spec)
            if payload.get('schema') is not None:
                # Check the public envelope before the shorthand builder drops it.
                from ..schema_contracts import TASK_SPEC_SCHEMA, validate_public_payload
                validate_public_payload(payload, expected_schema=TASK_SPEC_SCHEMA)
        else:
            raise TypeError('task_spec must be a TaskSpec object or dictionary')
        state = default_state('', channel='api', locale=locale)
        state['task_spec'] = payload
        state = spec_validator(spec_builder(state))
        errors = copy.deepcopy(state['errors'])
        configuration = None
        if not errors:
            try:
                configuration = self.compile_task_spec(state['task_spec'])
            except (TypeError, ValueError) as exc:
                errors.append({'stage': 'validation', 'code': 'workflow_invalid',
                               'message': str(exc)})
        return {
            'valid': not errors,
            'task_spec': state['task_spec'],
            'errors': errors,
            'applied_defaults': state['applied_defaults'],
            'clarification_questions': state['clarification_questions'],
            'workflow_configuration': configuration,
        }

    @staticmethod
    def apply_review_action(
        lifecycle: Any,
        action: str,
        *,
        review_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply an explicit user decision to one task review gate."""

        normalized_action = str(action or '').strip().lower()
        event = {
            'approve': 'approval_granted',
            'cancel': 'approval_rejected',
        }.get(normalized_action)
        if event is None:
            raise ValueError("Task review action must be 'approve' or 'cancel'.")
        value = ensure_lifecycle(lifecycle, 'task')
        return transition_lifecycle(
            value,
            event,
            details={
                'source': 'calculation_assistant',
                'review_type': str(review_type or 'task').strip().lower(),
            },
        )

    def compile_task_spec(self, task_spec: Any) -> Dict[str, Any]:
        """Compile a TaskSpec without submitting numerical work."""

        if isinstance(task_spec, TaskSpec):
            normalized = task_spec
        elif isinstance(task_spec, dict):
            normalized = task_spec_from_dict(task_spec)
        else:
            raise TypeError('task_spec must be a TaskSpec object or dictionary')
        registry = default_registry()
        task_payload = task_spec_to_dict(normalized)
        configuration = compile_task_workflow(
            task_payload,
            registry.modules_for_template('task.template.default'),
        ).to_dict()
        validate_compiled_runtime(configuration)
        gate_configuration = compile_task_gates(
            task_payload,
            registry.gates_for(scope='task'),
        ).to_dict()
        validate_compiled_gates(gate_configuration)
        configuration['gate_configuration'] = gate_configuration
        return configuration

    def _require_task_executor(self) -> TaskExecutor:
        executor = self._task_executor
        if executor is None:
            raise CalculationFeatureUnavailableError('Task executor is unavailable')
        return executor

    def prepare_request(
        self,
        *,
        messages: Any = None,
        task_spec: Any = None,
        request: Any = None,
        locale: str = 'en',
        lifecycle: Any = None,
    ) -> Dict[str, Any]:
        builder = self._llm_request_builder
        if builder is None or not hasattr(builder, 'build_prepared_request'):
            raise CalculationFeatureUnavailableError('LLM request preparation is unavailable')
        prepared = builder.build_prepared_request(
            messages,
            task_spec=task_spec,
            request=request,
            locale=locale,
        )
        prepared['lifecycle'] = task_lifecycle_for_preparation(
            prepared.get('status'),
            lifecycle=lifecycle,
            details={'source': str(prepared.get('source') or 'request_builder')},
        )
        return prepared

    def prepare_active_space_probe(
        self,
        task_spec: Any,
        *,
        target_solver: Optional[str] = None,
        target_solver_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Prepare the Assistant's automatic active-space probe without an LLM."""

        if not isinstance(task_spec, dict):
            raise TypeError('task_spec must be a dictionary')
        prepared = build_molecular_active_space_probe_request(
            task_spec,
            strategy='auto',
            target_solver=target_solver,
            target_solver_options=target_solver_options,
        )
        return {
            'status': 'ready',
            'probe_request': prepared['request'],
            'probe_contract': prepared['contract'],
        }

    @staticmethod
    def status_lifecycle(lifecycle: Any, status: JobStatus) -> Dict[str, Any]:
        return task_lifecycle_from_job_status(
            lifecycle,
            status.state,
            task_status=status.task_status,
            entity_id=status.handle.job_id,
            details={
                'executor_id': status.handle.executor_id,
                'message': status.message,
            },
        )

    @staticmethod
    def attach_report_lifecycle(
        report: Dict[str, Any],
        lifecycle: Any = None,
    ) -> Dict[str, Any]:
        if not isinstance(report, dict):
            raise TypeError('TaskReport must be a dictionary')
        if lifecycle is None and isinstance(report.get('lifecycle'), dict):
            return report
        value = ensure_lifecycle(
            lifecycle,
            'task',
            entity_id=report.get('run_id'),
        )
        value = task_lifecycle_from_job_status(
            value,
            'completed',
            task_status=report.get('execution_status'),
            entity_id=report.get('run_id'),
            details={'source': 'task_report'},
        )
        if (
            report.get('approval') is not None
            or lifecycle_requires_review(report.get('gate_decisions') or [])
        ) and value['stage'] in ('prepared', 'blocked', 'failed', 'unconverged', 'succeeded'):
            value = transition_lifecycle(
                value,
                'review_requested',
                details={'source': 'task_report'},
            )
        report['lifecycle'] = value
        return report

    def execute_request(
        self,
        request_text: str,
        *,
        channel: str = 'web',
        locale: str = 'en',
        work_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        resource_profile: Optional[str] = None,
        include_llm_feedback: bool = True,
    ) -> Dict[str, Any]:
        if not isinstance(request_text, str) or not request_text.strip():
            raise ValueError('request_text must be a non-empty string')
        executor = self._require_task_executor()
        execute_kwargs = {
            'channel': channel,
            'locale': locale,
            'work_dir': work_dir,
            'run_id': run_id,
        }
        if resource_profile:
            execute_kwargs['resource_profile'] = resource_profile
        report = executor.execute_task(request_text, **execute_kwargs)
        self.attach_report_lifecycle(report)
        if include_llm_feedback:
            self._append_execution_feedback(request_text, report, locale=locale)
        return report

    def submit_request(
        self,
        request_text: str,
        *,
        channel: str = 'api',
        locale: str = 'en',
        work_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        resource_profile: Optional[str] = None,
    ) -> JobHandle:
        """Submit one task without coupling clients to an executor implementation."""

        if not isinstance(request_text, str) or not request_text.strip():
            raise ValueError('request_text must be a non-empty string')
        submit_kwargs = {
            'channel': channel,
            'locale': locale,
            'work_dir': work_dir,
            'run_id': run_id,
        }
        if resource_profile:
            submit_kwargs['resource_profile'] = resource_profile
        return self._require_task_executor().submit_task(
            request_text,
            **submit_kwargs,
        )

    def submit_request_with_lifecycle(
        self,
        request_text: str,
        *,
        lifecycle: Any = None,
        channel: str = 'api',
        locale: str = 'en',
        work_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        resource_profile: Optional[str] = None,
    ) -> tuple[JobHandle, Dict[str, Any]]:
        """Validate lifecycle state before creating an executor-side job."""

        queued_lifecycle = submit_task_lifecycle(
            lifecycle,
            entity_id=run_id,
            details={'run_id': run_id, 'source': 'calculation_application_service'},
        )
        handle = self.submit_request(
            request_text,
            channel=channel,
            locale=locale,
            work_dir=work_dir,
            run_id=run_id,
            resource_profile=resource_profile,
        )
        return handle, queued_lifecycle

    def job_status(self, handle: Any) -> JobStatus:
        return self._require_task_executor().status(handle)

    def inspect_task(self, handle: Any) -> Dict[str, Any]:
        """Read a compact task view without collection, LLM calls or execution."""
        executor = self._require_task_executor()
        inspect = getattr(executor, 'inspect_task', None)
        if callable(inspect):
            return inspect(handle)
        from ..executors.inspection import status_view
        view = status_view(executor.status(handle))
        view['notices'].append('This executor provides status only; solver progress inspection is unavailable.')
        return view

    def cancel_job(self, handle: Any) -> JobStatus:
        return self._require_task_executor().cancel(handle)

    def fetch_job(self, handle: Any) -> Dict[str, Any]:
        return self._require_task_executor().fetch(handle)

    def collect_request(
        self,
        handle: Any,
        request_text: str,
        *,
        locale: str = 'en',
        include_llm_feedback: bool = True,
        lifecycle: Any = None,
    ) -> Dict[str, Any]:
        """Collect one submitted TaskReport and attach optional LLM feedback."""

        report = self.fetch_job(handle)
        self.attach_report_lifecycle(report, lifecycle)
        if include_llm_feedback:
            self._append_execution_feedback(request_text, report, locale=locale)
        return report

    def job_logs(self, handle: Any) -> list[Dict[str, Any]]:
        return self._require_task_executor().logs(handle)

    def job_artifacts(self, handle: Any) -> list[Dict[str, Any]]:
        return self._require_task_executor().artifacts(handle)

    def _append_execution_feedback(
        self,
        request_text: str,
        report: Dict[str, Any],
        *,
        locale: str,
    ) -> None:
        builder = self._llm_request_builder
        if builder is None or not hasattr(builder, 'build_execution_feedback'):
            return
        try:
            feedback = builder.build_execution_feedback(request_text, report, locale=locale)
        except Exception as exc:
            report['llm_feedback'] = {'status': 'failed', 'reason': str(exc)}
            return
        if not isinstance(feedback, dict):
            return
        messages = report.get('messages')
        if not isinstance(messages, list):
            messages = []
            report['messages'] = messages
        messages.append(feedback)

    def analyze_results(
        self,
        request_text: str,
        execution_report: Dict[str, Any],
        *,
        locale: str = 'en',
    ) -> str:
        if not isinstance(request_text, str) or not request_text.strip():
            raise ValueError('request_text must be a non-empty string')
        if not isinstance(execution_report, dict):
            raise TypeError('execution_report must be a dictionary')
        builder = self._llm_request_builder
        if builder is None or not hasattr(builder, 'build_result_analysis'):
            raise CalculationFeatureUnavailableError('LLM result analysis is unavailable')
        analysis = builder.build_result_analysis(request_text, execution_report, locale=locale)
        if not analysis:
            raise CalculationFeatureUnavailableError('LLM result analysis is unavailable')
        return str(analysis)

    def preview_model_hamiltonian(
        self,
        input_file: str,
        *,
        max_items: int = 200,
        solver: Any = None,
    ) -> Dict[str, Any]:
        if not isinstance(input_file, str) or not input_file.strip():
            raise ValueError('model_hamiltonian_input_file is required')
        spec = self._model_spec_loader(input_file)
        preview = self._model_hamiltonian_preview(spec, max_items=max_items)
        solver_name = solver
        solver_options: Dict[str, Any] = {}
        if isinstance(solver, dict):
            solver_name = solver.get('name')
            if isinstance(solver.get('options'), dict):
                solver_options = solver['options']
        if str(solver_name or '').strip().lower() == 'dmet':
            from ..providers.libdmet import validate_dmet_model_request

            normalized = normalize_model_spec(spec)
            errors, configuration = validate_dmet_model_request(normalized, solver_options)
            nelec = normalized.get('nelec')
            spin_sector = None
            if isinstance(nelec, (list, tuple)) and len(nelec) == 2:
                nalpha, nbeta = (int(value) for value in nelec)
                spin_sector = {
                    'nalpha': nalpha,
                    'nbeta': nbeta,
                    'nelec': nalpha + nbeta,
                    'libdmet_sz': nalpha - nbeta,
                    'physical_sz': 0.5 * (nalpha - nbeta),
                    'spin_multiplicity': abs(nalpha - nbeta) + 1,
                }
            preview['dmet_validation'] = {
                'valid': not errors,
                'errors': errors,
                'spin_sector': spin_sector,
                'effective_reference': configuration.get('reference') if configuration else None,
                'execution_mode': configuration.get('execution_mode') if configuration else None,
                'translation_backend': configuration.get('translation_backend') if configuration else None,
                'representative_site_ids': (
                    configuration.get('translation_symmetry', {})
                    .get('builder_audit', {})
                    .get('representative_site_ids')
                    if configuration
                    else None
                ),
                'impurity_shape': configuration.get('impurity_shape') if configuration else None,
                'fragment_site_counts': configuration.get('fragment_site_counts') if configuration else None,
                'primitive_cell_basis_size': configuration.get('primitive_cell_basis_size') if configuration else None,
            }
        return preview

    def prepare_model_hamiltonian_edit(
        self,
        task_spec: Dict[str, Any],
        request: str,
        *,
        messages: Any = None,
        locale: str = 'en',
        lifecycle: Any = None,
        work_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Translate and apply one reviewed natural-language model edit."""

        if not isinstance(task_spec, dict):
            raise TypeError('task_spec must be an object')
        input_file = str(task_spec.get('model_hamiltonian_input_file') or '').strip()
        if not input_file:
            raise ValueError('model_hamiltonian_input_file is required')
        if not isinstance(request, str) or not request.strip():
            raise ValueError('A model Hamiltonian edit request is required')
        builder = self._llm_request_builder
        if builder is None or not hasattr(builder, 'build_model_hamiltonian_operations'):
            raise CalculationFeatureUnavailableError('LLM model editing is unavailable')

        lifecycle_details = {'source': 'llm_model_operations'}
        edit_lifecycle = ensure_lifecycle(lifecycle, 'task')
        if edit_lifecycle['stage'] != 'draft':
            edit_lifecycle = transition_lifecycle(
                edit_lifecycle,
                'configuration_changed',
                details=lifecycle_details,
            )

        def attach_lifecycle(payload: Dict[str, Any]) -> Dict[str, Any]:
            payload['lifecycle'] = task_lifecycle_for_preparation(
                payload.get('status'),
                lifecycle=edit_lifecycle,
                details=lifecycle_details,
            )
            return payload

        original = normalize_model_spec(self._model_spec_loader(input_file))
        proposal = builder.build_model_hamiltonian_operations(
            request,
            original,
            messages=messages if isinstance(messages, list) else None,
            locale=locale,
        )
        questions = proposal.get('clarification_questions') or []
        operations = proposal.get('operations') or []
        if questions or not operations:
            return attach_lifecycle({
                'status': 'needs_clarification',
                'source': 'llm_model_operations',
                'structured_request': copy.deepcopy(task_spec),
                'operations': copy.deepcopy(operations),
                'changes': [],
                'clarification_questions': list(questions) or [
                    'Describe which site or bond to change, the parameter, and its new value.'
                ],
                'messages': [{
                    'role': 'assistant',
                    'content': (list(questions) or [
                        'Describe which site or bond to change, the parameter, and its new value.'
                    ])[0],
                }],
            })

        edited = normalize_model_spec(apply_model_operations(original, operations))
        changes = describe_model_changes(original, edited)
        if not changes:
            raise ValueError('The requested operations did not change the model Hamiltonian')

        next_task_spec = copy.deepcopy(task_spec)
        solver_payload = next_task_spec.get('solver')
        solver_name = solver_payload.get('name') if isinstance(solver_payload, dict) else solver_payload
        solver_options = (
            copy.deepcopy(solver_payload.get('options') or {})
            if isinstance(solver_payload, dict)
            else {}
        )
        if any(str(item.get('op') or item.get('operation') or '') == 'change_solver' for item in operations):
            solver_name = edited.get('solver')
            next_task_spec['solver'] = {
                'name': solver_name,
                **({'options': solver_options} if solver_options else {}),
            }
        generated_input = generate_model_hamiltonian_input_script(
            edited,
            solver_name=solver_name,
            outputs=next_task_spec.get('outputs'),
            solver_options=solver_options,
        )
        validation_errors = validate_model_hamiltonian_spec(edited, solver_name=solver_name)
        if validation_errors:
            raise ValueError('Edited Model Hamiltonian is invalid: {0}'.format(' | '.join(validation_errors)))
        saved = self.save_model_hamiltonian_input(generated_input, work_dir=work_dir)
        next_task_spec['model_hamiltonian_input_file'] = saved['path']
        summary = str(proposal.get('summary') or '').strip()
        if not summary:
            summary = 'Prepared {0} model Hamiltonian change(s) for review.'.format(len(changes))
        audit_text = '; '.join(
            '{0} {1} {2}: {3} -> {4}'.format(
                item['scope'],
                item['id'] if item['id'] is not None else '',
                item['parameter'],
                item['before'],
                item['after'],
            ).replace('  ', ' ')
            for item in changes
        )
        return attach_lifecycle({
            'status': 'ready',
            'source': 'llm_model_operations',
            'structured_request': next_task_spec,
            'request_text': json.dumps(next_task_spec, ensure_ascii=False),
            'execution_request_text': json.dumps(next_task_spec, ensure_ascii=False),
            'operations': copy.deepcopy(operations),
            'changes': changes,
            'proposed_changes': [
                '{0} {1} {2}: {3} -> {4}'.format(
                    item['scope'],
                    item['id'] if item['id'] is not None else '',
                    item['parameter'],
                    item['before'],
                    item['after'],
                ).replace('  ', ' ')
                for item in changes
            ],
            'generated_input': generated_input,
            'source_model_hamiltonian_input_file': input_file,
            'model_hamiltonian_input_file': saved['path'],
            'messages': [{
                'role': 'assistant',
                'content': '{0} Review: {1}.'.format(summary.rstrip('.'), audit_text),
            }],
        })

    @staticmethod
    def _model_hamiltonian_preview(
        spec: Dict[str, Any],
        *,
        max_items: int,
    ) -> Dict[str, Any]:
        normalized = normalize_model_spec(spec)
        sites = normalized.get('sites') if isinstance(normalized.get('sites'), list) else []
        bonds = normalized.get('bonds') if isinstance(normalized.get('bonds'), list) else []
        graph = normalized.get('graph') if isinstance(normalized.get('graph'), dict) else {}
        graph_nodes = graph.get('nodes') if isinstance(graph.get('nodes'), list) else []
        graph_edges = graph.get('edges') if isinstance(graph.get('edges'), list) else []
        phonons = normalized.get('phonons') if isinstance(normalized.get('phonons'), dict) else {}
        return {
            'schema': normalized.get('schema'),
            'model': normalized.get('model'),
            'representation': normalized.get('representation'),
            **({'solver': normalized.get('solver')} if normalized.get('representation') == 'bloch' else {}),
            'dimension': normalized.get('dimension'),
            'preset': normalized.get('preset'),
            'boundary': normalized.get('boundary'),
            'energy_unit': normalized.get('energy_unit'),
            'bond_modulation': normalized.get('bond_modulation'),
            'cell': normalized.get('cell') if isinstance(normalized.get('cell'), dict) else None,
            'primitive_cell': (
                normalized.get('primitive_cell')
                if isinstance(normalized.get('primitive_cell'), dict)
                else None
            ),
            'nelec': normalized.get('nelec'),
            'spin_multiplicity': normalized.get('spin_multiplicity'),
            'lattice_vectors': normalized.get('lattice_vectors'),
            'occupation': normalized.get('occupation') if isinstance(normalized.get('occupation'), dict) else None,
            'reciprocal_space': (
                normalized.get('reciprocal_space')
                if isinstance(normalized.get('reciprocal_space'), dict)
                else None
            ),
            'site_count': len(sites),
            'bond_count': len(bonds),
            'phonons_enabled': bool(phonons.get('enabled')),
            'sites_preview': sites[:max_items],
            'bonds_preview': bonds[:max_items],
            'graph_preview': {
                'representation': graph.get('representation') or 'site-bond graph',
                'nodes': graph_nodes[:max_items],
                'edges': graph_edges[:max_items],
            },
            'truncated': (
                len(sites) > max_items
                or len(bonds) > max_items
                or len(graph_nodes) > max_items
                or len(graph_edges) > max_items
            ),
        }

    def preview_periodic_structure(
        self,
        source_text: Any,
        structure_format: Any,
        *,
        seekpath_symprec: Any = 1e-5,
        seekpath_reference_distance: Any = None,
    ) -> Dict[str, Any]:
        return self._periodic_preview_builder(
            source_text,
            structure_format or 'poscar',
            seekpath_symprec=seekpath_symprec,
            seekpath_reference_distance=seekpath_reference_distance,
        )

    def save_model_hamiltonian_input(
        self,
        python_input: str,
        output_path: Any = None,
        *,
        work_dir: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not isinstance(python_input, str) or not python_input.strip():
            raise ValueError('python_input must be a non-empty string')
        encoded = python_input.encode('utf-8')
        if len(encoded) > MAX_MODEL_HAMILTONIAN_INPUT_BYTES:
            raise ValueError('python_input is too large')
        normalized_run_id = run_id.strip() if isinstance(run_id, str) and run_id.strip() else None
        state = {'work_dir': str(resolve_work_dir(work_dir, normalized_run_id))}
        if normalized_run_id:
            state['run_id'] = normalized_run_id
        normalized_run_id = ensure_run_id(state)
        path = Path(output_path) if output_path is not None else (
            get_run_dir(state) / 'input-builder-model-hamiltonian.py'
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(python_input, encoding='utf-8')
        return {
            'status': 'ok',
            'path': str(path),
            'work_dir': state['work_dir'],
            'run_id': normalized_run_id,
            'run_dir': str(path.parent),
            'bytes_written': len(encoded),
        }
