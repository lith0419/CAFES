from __future__ import annotations

import copy
import itertools
import math
from typing import Any, Dict, Iterable, List

from pyscf_agent.identifiers import make_run_id
from pyscf_agent.backend.model_hamiltonian.solver import load_model_spec_from_file, normalize_model_spec
from pyscf_agent.registry import default_registry as default_platform_capability_registry
from pyscf_agent.workflow_modules import compile_study_workflow
from pyscf_agent.workflow_gates import compile_study_gates

from .capabilities import (
    StudyCapabilityView,
    default_study_capabilities,
    study_capability_snapshot,
)
from .costing import ensure_plan_cost_estimate, normalize_resource_policy
from .gates.runtime import validate_study_gate_runtime
from .module_runtime import validate_study_module_runtime
from .normalization import (
    normalize_molecular_geometry_fields,
    normalize_study_mode,
    normalize_system_type,
    solver_contract,
    solver_payload,
    supported_study_modes,
)
from .schema import StudyCase, StudyPlan, StudySpec
from .transformations import (
    apply_model_operations, normalize_nelec_value, resolve_templates,
    validate_model_parameter_updates,
)


def _as_study_spec(spec: Any) -> StudySpec:
    if isinstance(spec, StudySpec):
        return spec
    return StudySpec.from_dict(spec)


def _model_study_observables(observables: Any, registry: StudyCapabilityView) -> List[str]:
    """Resolve derived metrics and include the common model diagnostic bundle."""
    values = [registry.model_observable_output(value) or value for value in (observables or ['energy'])]
    if 'strong_correlation_diagnostics' not in values:
        values.append('strong_correlation_diagnostics')
    return list(dict.fromkeys(values))


def _cartesian_product(sweep: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    if not sweep:
        yield {}
        return
    keys = list(sweep)
    value_lists = []
    for key in keys:
        values = sweep[key]
        if not isinstance(values, list) or not values:
            raise ValueError('Sweep parameter {0} must provide a non-empty list'.format(key))
        value_lists.append(values)
    for values in itertools.product(*value_lists):
        yield dict(zip(keys, values))


def _case_label(variables: Dict[str, Any]) -> str:
    if not variables:
        return 'base'
    return ', '.join('{0}={1}'.format(key, value) for key, value in variables.items())


def _case_design_variables(case_design: Dict[str, Any]) -> Dict[str, List[Any]]:
    variables = case_design.get('variables', {})
    if not isinstance(variables, dict):
        raise ValueError('case_design.variables must be an object')
    return copy.deepcopy(variables)


def _case_design_template(case_design: Dict[str, Any]) -> Dict[str, Any]:
    template = case_design.get('template', {})
    if template is None:
        template = {}
    if not isinstance(template, dict):
        raise ValueError('case_design.template must be an object')
    return template


def _case_design_overrides(case_design: Dict[str, Any]) -> List[Dict[str, Any]]:
    overrides = case_design.get('overrides', [])
    if overrides is None:
        return []
    if not isinstance(overrides, list):
        raise ValueError('case_design.overrides must be a list')
    normalized = []
    for item in overrides:
        if not isinstance(item, dict):
            raise ValueError('Each case_design override must be an object')
        selector = item.get('selector')
        if not isinstance(selector, dict) or not selector:
            raise ValueError('Each case_design override requires a non-empty selector object')
        normalized.append(copy.deepcopy(item))
    return normalized


def _case_design_selector_matches(selector: Dict[str, Any], variables: Dict[str, Any]) -> bool:
    for name, expected in selector.items():
        if name not in variables:
            return False
        actual = variables[name]
        candidates = expected if isinstance(expected, list) else [expected]
        matched = False
        for candidate in candidates:
            if isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                if math.isclose(float(actual), float(candidate), rel_tol=1e-10, abs_tol=1e-12):
                    matched = True
                    break
            elif actual == candidate:
                matched = True
                break
        if not matched:
            return False
    return True


def _matching_case_design_overrides(case_design: Dict[str, Any], variables: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        override
        for override in _case_design_overrides(case_design)
        if _case_design_selector_matches(override['selector'], variables)
    ]


def _case_design_mode(case_design: Dict[str, Any]) -> str:
    mode = str(case_design.get('mode') or 'grid').strip().lower()
    if mode not in ('grid', 'cases'):
        raise ValueError('Unsupported case_design mode: {0}'.format(mode))
    return mode


def _explicit_case_items(case_design: Dict[str, Any]) -> List[Dict[str, Any]]:
    cases = case_design.get('cases', [])
    if not isinstance(cases, list) or not cases:
        raise ValueError('case_design.cases must be a non-empty list when mode="cases"')
    normalized_cases = []
    for item in cases:
        if not isinstance(item, dict):
            raise ValueError('Each case_design.cases item must be an object')
        normalized_cases.append(copy.deepcopy(item))
    return normalized_cases


def _case_design_case_items(case_design: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    mode = _case_design_mode(case_design)
    if mode == 'grid':
        for variables in _cartesian_product(_case_design_variables(case_design)):
            yield {'variables': variables}
        return
    for item in _explicit_case_items(case_design):
        variables = item.get('variables') if isinstance(item.get('variables'), dict) else {}
        yield {
            'variables': copy.deepcopy(variables),
            'label': item.get('label'),
            'operations': copy.deepcopy(item.get('operations') or []),
            'request_updates': copy.deepcopy(item.get('request_updates') or {}),
        }


def _case_design_operations(case_design: Dict[str, Any], case_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    variables = case_item.get('variables') or {}
    template = _case_design_template(case_design)
    operations = template.get('operations', [])
    if not isinstance(operations, list):
        raise ValueError('case_design.template.operations must be a list')
    case_operations = case_item.get('operations', [])
    if not isinstance(case_operations, list):
        raise ValueError('case_design.cases[].operations must be a list')
    override_operations: List[Dict[str, Any]] = []
    for override in _matching_case_design_overrides(case_design, variables):
        extra_operations = override.get('operations', [])
        if extra_operations is None:
            extra_operations = []
        if not isinstance(extra_operations, list):
            raise ValueError('case_design.overrides[].operations must be a list')
        override_operations.extend(extra_operations)
    return resolve_templates(operations + case_operations + override_operations, variables)


def _case_design_request_updates(case_design: Dict[str, Any], case_item: Dict[str, Any]) -> Dict[str, Any]:
    variables = case_item.get('variables') or {}
    template = _case_design_template(case_design)
    updates = template.get('request_updates', {})
    if updates is None:
        updates = {}
    if not isinstance(updates, dict):
        raise ValueError('case_design.template.request_updates must be an object')
    case_updates = case_item.get('request_updates', {})
    if case_updates is None:
        case_updates = {}
    if not isinstance(case_updates, dict):
        raise ValueError('case_design.cases[].request_updates must be an object')
    resolved_updates = {
        **resolve_templates(updates, variables),
        **resolve_templates(case_updates, variables),
    }
    for override in _matching_case_design_overrides(case_design, variables):
        override_updates = override.get('request_updates', {})
        if override_updates is None:
            override_updates = {}
        if not isinstance(override_updates, dict):
            raise ValueError('case_design.overrides[].request_updates must be an object')
        resolved_updates.update(resolve_templates(override_updates, variables))
    return resolved_updates


def _case_design_label(case_item: Dict[str, Any]) -> str:
    label = case_item.get('label')
    if isinstance(label, str) and label.strip():
        return label.strip()
    return _case_label(case_item.get('variables') or {})


def _validate_molecular_case(request: Dict[str, Any], registry: StudyCapabilityView) -> None:
    method = registry.canonical_molecular_method(request.get('method') or 'hf')
    if not method:
        raise ValueError('Unsupported molecular method in study case: {0}'.format(request.get('method') or 'hf'))
    request['method'] = method
    if method == 'dft' and not request.get('xc'):
        raise ValueError('DFT study cases require an xc functional')


def _molecular_cases(spec: StudySpec, registry: StudyCapabilityView) -> List[StudyCase]:
    base_task = copy.deepcopy(spec.base_task)
    if spec.workflow:
        base_task['workflow'] = copy.deepcopy(spec.workflow)
    normalize_molecular_geometry_fields(base_task)
    base_task['task_type'] = 'molecular'
    cases = []
    case_items = _case_design_case_items(spec.case_design) if spec.case_design else ({'variables': variables} for variables in _cartesian_product(spec.sweep))
    for index, case_item in enumerate(case_items, start=1):
        variables = case_item.get('variables') or {}
        request = resolve_templates(copy.deepcopy(base_task), variables)
        if spec.case_design:
            request_updates = _case_design_request_updates(spec.case_design, case_item)
            normalize_molecular_geometry_fields(request_updates)
            request.update(request_updates)
        else:
            request.update(variables)
        normalize_molecular_geometry_fields(request)
        _validate_molecular_case(request, registry)
        case_id = 'case-{0:04d}'.format(index)
        cases.append(StudyCase(
            case_id=case_id,
            label=_case_design_label(case_item),
            request=request,
            variables=copy.deepcopy(variables),
            operations=[],
        ))
    return cases


def _load_base_model_spec(spec: StudySpec) -> Dict[str, Any]:
    if spec.base_model_input_file:
        return normalize_model_spec(load_model_spec_from_file(spec.base_model_input_file))
    if spec.base_model_spec:
        return normalize_model_spec(copy.deepcopy(spec.base_model_spec))
    raise ValueError('Model Hamiltonian studies require base_model_spec or base_model_input_file')


def _apply_model_override(model_spec: Dict[str, Any], parameter: str, value: Any) -> None:
    if parameter == 'U':
        for site in model_spec.get('sites', []):
            site['U'] = value
        model_spec.setdefault('globals', {})['U'] = value
        return
    if parameter == 'epsilon':
        for site in model_spec.get('sites', []):
            site['epsilon'] = value
        model_spec.setdefault('globals', {})['epsilon'] = value
        return
    if parameter == 't':
        for bond in model_spec.get('bonds', []):
            bond['t'] = value
            bond['effective_t'] = value
        model_spec.setdefault('globals', {})['t'] = value
        return
    if parameter == 'V':
        for bond in model_spec.get('bonds', []):
            bond['V'] = value
            bond['effective_V'] = value
        model_spec.setdefault('globals', {})['V'] = value
        return
    if parameter == 'nelec':
        normalized_nelec = normalize_nelec_value(value)
        model_spec['nelec'] = normalized_nelec
        model_spec['spin_multiplicity'] = abs(normalized_nelec[0] - normalized_nelec[1]) + 1
        return
    if parameter == 'boundary':
        model_spec['boundary'] = str(value)
        return
    raise ValueError('Unsupported model Hamiltonian sweep parameter: {0}'.format(parameter))


def _validate_model_sweep(sweep: Dict[str, List[Any]], registry: StudyCapabilityView) -> None:
    for parameter in sweep:
        if parameter == 'solver':
            continue
        if not registry.supports_model_parameter(parameter):
            raise ValueError('Unsupported model Hamiltonian sweep parameter: {0}'.format(parameter))


def _model_cases(spec: StudySpec, registry: StudyCapabilityView) -> List[StudyCase]:
    base_model_spec = _load_base_model_spec(spec)
    base_model_solver, base_model_solver_options = solver_contract(base_model_spec.get('solver'))
    base_model_solver = base_model_solver.lower()
    default_solver_value = spec.base_task.get('solver') or base_model_solver
    if not default_solver_value and not (isinstance(spec.sweep, dict) and spec.sweep.get('solver')):
        raise ValueError('Model Hamiltonian studies require an explicit solver')
    default_solver_name, default_solver_options = solver_contract(default_solver_value)
    default_solver = default_solver_name.lower()
    if not default_solver_options and default_solver == base_model_solver:
        default_solver_options = base_model_solver_options
    cases = []
    if spec.case_design:
        case_items = _case_design_case_items(spec.case_design)
    else:
        _validate_model_sweep(spec.sweep, registry)
        case_items = ({'variables': variables} for variables in _cartesian_product(spec.sweep))
    for index, case_item in enumerate(case_items, start=1):
        variables = case_item.get('variables') or {}
        model_spec = copy.deepcopy(base_model_spec)
        operations: List[Dict[str, Any]] = []
        request_updates: Dict[str, Any] = {}
        if spec.case_design:
            operations = _case_design_operations(spec.case_design, case_item)
            request_updates = _case_design_request_updates(spec.case_design, case_item)
            model_spec = apply_model_operations(
                model_spec, operations, path='cases[{0}].operations'.format(index - 1),
            )
        else:
            for parameter, value in variables.items():
                if parameter == 'solver':
                    continue
                _apply_model_override(model_spec, parameter, value)
        model_solver_name, model_solver_options = solver_contract(model_spec.get('solver'))
        model_solver = model_solver_name.lower()
        operation_solver = model_solver if model_solver and model_solver != base_model_solver else None
        solver_value = request_updates.get('solver') or variables.get('solver') or operation_solver or default_solver
        solver_name, solver_options = solver_contract(solver_value)
        solver = registry.canonical_model_solver(solver_name)
        if not solver:
            if solver_name:
                raise ValueError('Unsupported model Hamiltonian solver in study case: {0}'.format(solver_name))
            raise ValueError('Model Hamiltonian study case requires an explicit solver')
        if not solver_options:
            if solver == default_solver:
                solver_options = copy.deepcopy(default_solver_options)
            elif solver == model_solver:
                solver_options = copy.deepcopy(model_solver_options)
        request = {
            'task_type': 'model_hamiltonian',
            'model_hamiltonian': {'spec': model_spec},
            'solver': solver_payload(solver, solver_options),
            'analysis': {'outputs': _model_study_observables(spec.observables, registry)},
        }
        if spec.workflow:
            request['workflow'] = copy.deepcopy(spec.workflow)
        request.update({
            key: value
            for key, value in request_updates.items()
            if key != 'solver'
        })
        request['task_type'] = 'model_hamiltonian'
        case_id = 'case-{0:04d}'.format(index)
        cases.append(StudyCase(
            case_id=case_id,
            label=_case_design_label(case_item),
            request=request,
            variables=copy.deepcopy(variables),
            operations=copy.deepcopy(operations),
            model_spec=model_spec,
        ))
    return cases


def build_study_plan(spec: Any, registry: StudyCapabilityView = None) -> StudyPlan:
    study_spec = _as_study_spec(spec)
    capability_registry = registry or default_study_capabilities()
    if study_spec.system_type == 'model_hamiltonian':
        validate_model_parameter_updates(study_spec.case_design)
    from computational_study_agent.grid.refinement import normalize_policy, prepare_source, validate_source
    grid_policy = normalize_policy(study_spec.grid_refinement)
    validate_source(study_spec, grid_policy)
    system_type = normalize_system_type(study_spec.system_type)
    workflow_mode = normalize_study_mode(study_spec.study_mode)
    if workflow_mode not in supported_study_modes(system_type):
        raise ValueError(
            'Model Hamiltonian studies support static scans only; choose an explicit solver for the planned cases.'
        )
    if system_type == 'molecular':
        cases = _molecular_cases(study_spec, capability_registry)
    elif system_type == 'model_hamiltonian':
        cases = _model_cases(study_spec, capability_registry)
    else:
        raise ValueError('Unsupported study system_type: {0}'.format(study_spec.system_type))
    if not cases:
        raise ValueError('Study plan contains no cases')
    resource_policy = normalize_resource_policy(study_spec.resource_policy)
    platform_registry = default_platform_capability_registry()
    workflow_configuration = compile_study_workflow(
        study_spec.to_dict(),
        platform_registry.modules_for_template('study.template.' + workflow_mode),
        study_mode=workflow_mode,
    )
    validate_study_module_runtime(workflow_configuration.to_dict())
    gate_configuration = compile_study_gates(
        study_spec.to_dict(),
        platform_registry.gates_for(scope='study'),
    )
    validate_study_gate_runtime(gate_configuration.to_dict())
    workflow_payload = workflow_configuration.to_dict()
    workflow_payload['gate_configuration'] = gate_configuration.to_dict()
    plan = StudyPlan(
        study_id=make_run_id(),
        name=study_spec.name,
        objective=study_spec.objective,
        system_type=system_type,
        cases=cases,
        observables=(
            _model_study_observables(study_spec.observables, capability_registry)
            if system_type == 'model_hamiltonian'
            else list(study_spec.observables or ['energy'])
        ),
        comparison=copy.deepcopy(study_spec.comparison),
        capability_snapshot=study_capability_snapshot(capability_registry),
        resource_policy=resource_policy,
        grid_refinement=grid_policy,
        grid_refinement_source=prepare_source(study_spec, grid_policy) if grid_policy else {},
        workflow_configuration=workflow_payload,
        workflow_provenance=copy.deepcopy(workflow_configuration.provenance),
        gate_configuration=gate_configuration.to_dict(),
        gate_provenance=copy.deepcopy(gate_configuration.provenance),
    )
    ensure_plan_cost_estimate(plan)
    return plan
