from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List

from pyscf_agent.backend.model_hamiltonian.solver import validate_model_hamiltonian_spec

from .capabilities import (
    StudyCapabilityView,
    default_study_capabilities,
    molecular_request_contract_issues,
)
from .planner import build_study_plan
from .schema import StudyPlan, StudySpec
from .transformations import validate_model_parameter_updates
from .costing import estimate_case_cost
from .normalization import (
    normalize_study_mode,
    normalize_system_type,
    solver_contract,
    supported_study_modes,
)


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str
    path: str = ''

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def _issue(severity: str, code: str, message: str, path: str = '') -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message, path=path)


def issue_dicts(issues: Iterable[ValidationIssue]) -> List[Dict[str, str]]:
    return [issue.to_dict() for issue in issues]


def has_errors(issues: Iterable[ValidationIssue]) -> bool:
    return any(issue.severity == 'error' for issue in issues)


def _validate_sweep(sweep: Dict[str, Any], *, system_type: str, registry: StudyCapabilityView) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not isinstance(sweep, dict):
        return [_issue('error', 'invalid_sweep', 'sweep must be an object', 'sweep')]
    for parameter, values in sweep.items():
        path = 'sweep.{0}'.format(parameter)
        if not isinstance(values, list) or not values:
            issues.append(_issue('error', 'invalid_sweep_values', 'Sweep parameter values must be a non-empty list', path))
        if system_type == 'model_hamiltonian' and parameter != 'solver' and not registry.supports_model_parameter(parameter):
            issues.append(_issue('error', 'unsupported_model_parameter', 'Unsupported model Hamiltonian sweep parameter: {0}'.format(parameter), path))
        if system_type == 'molecular' and parameter == 'method':
            for index, method in enumerate(values if isinstance(values, list) else []):
                if not registry.supports_molecular_method(str(method)):
                    issues.append(_issue('error', 'unsupported_molecular_method', 'Unsupported molecular method: {0}'.format(method), '{0}[{1}]'.format(path, index)))
    return issues


def _validate_case_design(case_design: Dict[str, Any], *, system_type: str, registry: StudyCapabilityView) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not isinstance(case_design, dict):
        return [_issue('error', 'invalid_case_design', 'case_design must be an object', 'case_design')]
    mode = str(case_design.get('mode') or 'grid').strip().lower()
    if mode not in ('grid', 'cases'):
        issues.append(_issue('error', 'unsupported_case_design_mode', 'case_design.mode must be "grid" or "cases"', 'case_design.mode'))
    template = case_design.get('template', {})
    if template is None:
        template = {}
    if not isinstance(template, dict):
        issues.append(_issue('error', 'invalid_case_template', 'case_design.template must be an object', 'case_design.template'))
        template = {}
    operations = template.get('operations', [])
    if operations is None:
        operations = []
    if not isinstance(operations, list):
        issues.append(_issue('error', 'invalid_case_operations', 'case_design.template.operations must be a list', 'case_design.template.operations'))
        operations = []
    request_updates = template.get('request_updates', {})
    if request_updates is None:
        request_updates = {}
    if not isinstance(request_updates, dict):
        issues.append(_issue('error', 'invalid_request_updates', 'case_design.template.request_updates must be an object', 'case_design.template.request_updates'))
    if system_type == 'model_hamiltonian':
        supported_operations = set(registry.model_hamiltonian_operations)
        for index, operation in enumerate(operations):
            path = 'case_design.template.operations[{0}]'.format(index)
            if not isinstance(operation, dict):
                issues.append(_issue('error', 'invalid_case_operation', 'Each operation must be an object', path))
                continue
            issues.extend(_validate_model_operation(operation, supported_operations, path))
    elif operations:
        issues.append(_issue('error', 'operations_require_model_hamiltonian', 'case_design operations are currently supported only for model_hamiltonian studies', 'case_design.template.operations'))
    if mode == 'grid':
        variables = case_design.get('variables', {})
        if not isinstance(variables, dict):
            issues.append(_issue('error', 'invalid_case_variables', 'case_design.variables must be an object', 'case_design.variables'))
        else:
            for name, values in variables.items():
                if not isinstance(values, list) or not values:
                    issues.append(_issue('error', 'invalid_case_variable_values', 'case_design variable values must be a non-empty list', 'case_design.variables.{0}'.format(name)))
    if mode == 'cases':
        cases = case_design.get('cases', [])
        if not isinstance(cases, list) or not cases:
            issues.append(_issue('error', 'invalid_case_list', 'case_design.cases must be a non-empty list', 'case_design.cases'))
        else:
            for index, case_item in enumerate(cases):
                path = 'case_design.cases[{0}]'.format(index)
                if not isinstance(case_item, dict):
                    issues.append(_issue('error', 'invalid_case_item', 'Each case_design case must be an object', path))
                    continue
                case_operations = case_item.get('operations', [])
                if case_operations is None:
                    case_operations = []
                if not isinstance(case_operations, list):
                    issues.append(_issue('error', 'invalid_case_operations', 'case_design.cases[].operations must be a list', path + '.operations'))
                elif system_type == 'model_hamiltonian':
                    supported_operations = set(registry.model_hamiltonian_operations)
                    for operation_index, operation in enumerate(case_operations):
                        operation_path = '{0}.operations[{1}]'.format(path, operation_index)
                        if not isinstance(operation, dict):
                            issues.append(_issue('error', 'invalid_case_operation', 'Each operation must be an object', operation_path))
                            continue
                        issues.extend(_validate_model_operation(operation, supported_operations, operation_path))
                elif case_operations:
                    issues.append(_issue('error', 'operations_require_model_hamiltonian', 'case operations are currently supported only for model_hamiltonian studies', path + '.operations'))
                case_updates = case_item.get('request_updates', {})
                if case_updates is not None and not isinstance(case_updates, dict):
                    issues.append(_issue('error', 'invalid_request_updates', 'case_design.cases[].request_updates must be an object', path + '.request_updates'))
    overrides = case_design.get('overrides', [])
    if overrides is None:
        overrides = []
    if not isinstance(overrides, list):
        issues.append(_issue('error', 'invalid_case_overrides', 'case_design.overrides must be a list', 'case_design.overrides'))
    else:
        for index, override in enumerate(overrides):
            path = 'case_design.overrides[{0}]'.format(index)
            if not isinstance(override, dict):
                issues.append(_issue('error', 'invalid_case_override', 'Each case_design override must be an object', path))
                continue
            selector = override.get('selector')
            if not isinstance(selector, dict) or not selector:
                issues.append(_issue('error', 'invalid_case_override_selector', 'case_design override requires a non-empty selector object', path + '.selector'))
            override_updates = override.get('request_updates', {})
            if override_updates is not None and not isinstance(override_updates, dict):
                issues.append(_issue('error', 'invalid_request_updates', 'case_design.overrides[].request_updates must be an object', path + '.request_updates'))
            override_operations = override.get('operations', [])
            if override_operations is None:
                override_operations = []
            if not isinstance(override_operations, list):
                issues.append(_issue('error', 'invalid_case_operations', 'case_design.overrides[].operations must be a list', path + '.operations'))
            elif system_type == 'model_hamiltonian':
                supported_operations = set(registry.model_hamiltonian_operations)
                for operation_index, operation in enumerate(override_operations):
                    operation_path = '{0}.operations[{1}]'.format(path, operation_index)
                    if not isinstance(operation, dict):
                        issues.append(_issue('error', 'invalid_case_operation', 'Each operation must be an object', operation_path))
                        continue
                    issues.extend(_validate_model_operation(operation, supported_operations, operation_path))
            elif override_operations:
                issues.append(_issue('error', 'operations_require_model_hamiltonian', 'case override operations are currently supported only for model_hamiltonian studies', path + '.operations'))
    return issues


def _validate_model_operation(operation: Dict[str, Any], supported_operations: set, path: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    op_name = str(operation.get('op') or operation.get('operation') or '').strip()
    if op_name not in supported_operations:
        issues.append(_issue('error', 'unsupported_model_operation', 'Unsupported model operation: {0}'.format(op_name or operation), path))
    if op_name in ('set_site_parameter', 'shift_site_parameter', 'scale_site_parameter', 'add_site_defect'):
        if not any(key in operation for key in ('site', 'sites', 'selector')):
            issues.append(_issue('error', 'missing_site_selector', 'Site operation requires site, sites, or selector', path))
    if op_name in ('set_bond_parameter', 'shift_bond_parameter', 'scale_bond_parameter', 'add_bond_defect'):
        has_endpoint_pair = 'source' in operation and 'target' in operation
        if not has_endpoint_pair and not any(key in operation for key in ('bond', 'bonds', 'selector')):
            issues.append(_issue('error', 'missing_bond_selector', 'Bond operation requires bond, bonds, selector, or source/target', path))
    if op_name in ('set_site_parameter', 'shift_site_parameter', 'scale_site_parameter', 'set_bond_parameter', 'shift_bond_parameter', 'scale_bond_parameter'):
        if 'parameter' not in operation:
            issues.append(_issue('error', 'missing_operation_parameter', 'Parameter operation requires parameter', path))
        if not any(key in operation for key in ('value', 'shift', 'factor')):
            issues.append(_issue('error', 'missing_operation_value', 'Parameter operation requires value, shift, or factor', path))
    if op_name == 'set_global_parameter':
        if not str(operation.get('parameter') or '').strip():
            issues.append(_issue('error', 'missing_operation_parameter', 'Global parameter operation requires parameter', path))
        if 'value' not in operation:
            issues.append(_issue('error', 'missing_operation_value', 'Global parameter operation requires value', path))
    if op_name == 'add_site_defect':
        has_explicit_update = 'parameter' in operation and any(key in operation for key in ('value', 'shift', 'factor'))
        has_shorthand_update = any(key in operation for key in ('epsilon_shift', 'U_shift'))
        if not has_explicit_update and not has_shorthand_update:
            issues.append(_issue('error', 'missing_operation_value', 'add_site_defect requires parameter/value or epsilon_shift/U_shift', path))
    if op_name == 'add_bond_defect':
        has_explicit_update = 'parameter' in operation and any(key in operation for key in ('value', 'shift', 'factor'))
        has_shorthand_update = any(key in operation for key in ('t_shift', 'V_shift'))
        if not has_explicit_update and not has_shorthand_update:
            issues.append(_issue('error', 'missing_operation_value', 'add_bond_defect requires parameter/value or t_shift/V_shift', path))
    return issues


def _validate_observables(observables: Any, *, system_type: str, registry: StudyCapabilityView, path: str) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if observables is None:
        return issues
    if not isinstance(observables, list):
        return [_issue('error', 'invalid_observables', 'observables must be a list', path)]
    if not observables:
        issues.append(_issue('error', 'empty_observables', 'observables must contain at least one item', path))
        return issues
    for index, observable in enumerate(observables):
        observable_id = str(observable or '').strip()
        observable_path = '{0}[{1}]'.format(path, index)
        if not observable_id:
            issues.append(_issue('error', 'invalid_observable', 'Observable id must be non-empty', observable_path))
            continue
        if system_type == 'model_hamiltonian':
            if not registry.supports_model_observable(observable_id):
                issues.append(_issue(
                    'error',
                    'unsupported_model_observable',
                    'Unsupported model Hamiltonian observable: {0}'.format(observable_id),
                    observable_path,
                ))
        elif system_type == 'molecular':
            if not registry.supports_molecular_observable(observable_id):
                issues.append(_issue(
                    'error',
                    'unsupported_molecular_observable',
                    'Unsupported molecular observable: {0}'.format(observable_id),
                    observable_path,
                ))
    return issues


def _validate_molecular_request_contract(
    request: Dict[str, Any],
    registry: StudyCapabilityView,
    path: str,
) -> List[ValidationIssue]:
    return [
        _issue('error', code, message, path)
        for code, message in molecular_request_contract_issues(request, registry)
    ]


def validate_study_spec(spec: Any, registry: StudyCapabilityView = None) -> List[ValidationIssue]:
    capability_registry = registry or default_study_capabilities()
    issues: List[ValidationIssue] = []
    try:
        study_spec = spec if isinstance(spec, StudySpec) else StudySpec.from_dict(spec)
    except Exception as exc:
        return [_issue('error', 'invalid_study_spec', 'StudySpec is invalid: {0}'.format(exc), '')]
    if study_spec.system_type == 'model_hamiltonian':
        try:
            validate_model_parameter_updates(study_spec.case_design)
        except ValueError as exc:
            issues.append(_issue('error', 'invalid_model_parameter_updates', str(exc), 'case_design'))
    from computational_study_agent.grid.refinement import normalize_policy, validate_source
    try:
        policy = normalize_policy(study_spec.grid_refinement)
        if not has_errors(issues):
            validate_source(study_spec, policy)
    except (ValueError, TypeError) as exc:
        issues.append(_issue('error', 'invalid_grid_refinement', str(exc), 'grid_refinement'))
    system_type = normalize_system_type(study_spec.system_type)
    study_mode = normalize_study_mode(study_spec.study_mode)
    if system_type not in ('molecular', 'model_hamiltonian'):
        issues.append(_issue('error', 'unsupported_system_type', 'Unsupported study system_type: {0}'.format(study_spec.system_type), 'system_type'))
    if study_mode not in supported_study_modes(system_type):
        issues.append(_issue(
            'error',
            'unsupported_study_mode',
            'Model Hamiltonian studies use static scans with an explicit solver for every case. Correlation diagnostics are calculated outputs, not a separate initial-scan stage.',
            'study_mode',
        ))
    issues.extend(_validate_observables(study_spec.observables, system_type=system_type, registry=capability_registry, path='observables'))
    if study_spec.case_design and study_spec.sweep:
        issues.append(_issue('warning', 'case_design_overrides_sweep', 'case_design is present, so sweep will be ignored by the planner', 'sweep'))
    if study_spec.sweep and not study_spec.case_design:
        issues.extend(_validate_sweep(study_spec.sweep, system_type=system_type, registry=capability_registry))
    if study_spec.case_design:
        issues.extend(_validate_case_design(study_spec.case_design, system_type=system_type, registry=capability_registry))
    if system_type == 'molecular':
        base_task = study_spec.base_task
        if not isinstance(base_task, dict):
            issues.append(_issue('error', 'invalid_base_task', 'base_task must be an object', 'base_task'))
        else:
            method = str(base_task.get('method') or 'hf').lower()
            if not study_spec.sweep and not study_spec.case_design and not capability_registry.supports_molecular_method(method):
                issues.append(_issue('error', 'unsupported_molecular_method', 'Unsupported molecular method: {0}'.format(method), 'base_task.method'))
            if method == 'dft' and not base_task.get('xc'):
                issues.append(_issue('error', 'missing_xc', 'DFT molecular studies require an xc functional', 'base_task.xc'))
    if system_type == 'model_hamiltonian':
        if not study_spec.base_model_spec and not study_spec.base_model_input_file:
            issues.append(_issue('error', 'missing_model_spec', 'Model Hamiltonian studies require base_model_spec or base_model_input_file', 'base_model_spec'))
        base_solver = None
        if isinstance(study_spec.base_task, dict):
            raw_base_solver, _base_solver_options = solver_contract(study_spec.base_task.get('solver'))
            if raw_base_solver:
                base_solver = raw_base_solver.lower()
        if base_solver is None and (
            not study_spec.base_model_input_file
            and isinstance(study_spec.base_model_spec, dict)
        ):
            raw_base_solver, _base_solver_options = solver_contract(study_spec.base_model_spec.get('solver'))
            if raw_base_solver:
                base_solver = raw_base_solver.lower()
        has_solver_sweep = isinstance(study_spec.sweep, dict) and isinstance(study_spec.sweep.get('solver'), list) and bool(study_spec.sweep.get('solver'))
        if base_solver is None and not has_solver_sweep:
            issues.append(_issue(
                'error',
                'missing_model_solver',
                'Model Hamiltonian studies require an explicit solver. Recommended: fci for small exact benchmarks, block2_dmrg or dmet for supported larger correlated models, ccsd/ccsd_t for approximate correlated comparisons, or mp2 when explicitly requested.',
                'base_task.solver',
            ))
        elif base_solver is not None and not capability_registry.supports_model_solver(base_solver):
            issues.append(_issue('error', 'unsupported_model_solver', 'Unsupported model Hamiltonian solver: {0}'.format(base_solver), 'base_task.solver'))
        if study_spec.base_model_spec and not study_spec.base_model_input_file:
            for message in validate_model_hamiltonian_spec(study_spec.base_model_spec, str(base_solver or 'fci')):
                issues.append(_issue('error', 'invalid_base_model_spec', message, 'base_model_spec'))
    return issues


def validate_study_plan(plan: Any, registry: StudyCapabilityView = None) -> List[ValidationIssue]:
    capability_registry = registry or default_study_capabilities()
    issues: List[ValidationIssue] = []
    try:
        study_plan = plan if isinstance(plan, StudyPlan) else StudyPlan.from_dict(plan)
    except Exception as exc:
        return [_issue('error', 'invalid_study_plan', 'StudyPlan is invalid: {0}'.format(exc), '')]
    from computational_study_agent.grid.refinement import validate_plan_contract
    try:
        validate_plan_contract(study_plan)
    except (ValueError, TypeError) as exc:
        issues.append(_issue('error', 'invalid_grid_refinement', str(exc), 'grid_refinement'))
    system_type = normalize_system_type(study_plan.system_type)
    if not study_plan.cases:
        issues.append(_issue('error', 'empty_study_plan', 'StudyPlan must contain at least one case', 'cases'))
    issues.extend(_validate_observables(study_plan.observables, system_type=system_type, registry=capability_registry, path='observables'))
    seen_case_ids = set()
    dmet_issue_cases: Dict[str, List[str]] = {}
    for index, case in enumerate(study_plan.cases):
        path = 'cases[{0}]'.format(index)
        if case.case_id in seen_case_ids:
            issues.append(_issue('error', 'duplicate_case_id', 'Duplicate case_id: {0}'.format(case.case_id), path + '.case_id'))
        seen_case_ids.add(case.case_id)
        if not isinstance(case.request, dict) or not case.request:
            issues.append(_issue('error', 'invalid_case_request', 'Each case must contain a request object', path + '.request'))
            continue
        if system_type == 'molecular':
            raw_method = case.request.get('method') or 'hf'
            method = capability_registry.canonical_molecular_method(raw_method)
            if not method:
                issues.append(_issue('error', 'unsupported_molecular_method', 'Unsupported molecular method: {0}'.format(raw_method), path + '.request.method'))
            if method == 'dft' and not case.request.get('xc'):
                issues.append(_issue('error', 'missing_xc', 'DFT molecular cases require an xc functional', path + '.request.xc'))
            issues.extend(_validate_molecular_request_contract(case.request, capability_registry, path + '.request'))
        elif system_type == 'model_hamiltonian':
            raw_solver = case.request.get('solver') or 'fci'
            raw_solver_name, _solver_options = solver_contract(raw_solver)
            solver = capability_registry.canonical_model_solver(raw_solver_name)
            if not solver:
                issues.append(_issue('error', 'unsupported_model_solver', 'Unsupported model Hamiltonian solver: {0}'.format(raw_solver_name), path + '.request.solver'))
                solver = raw_solver_name.lower()
            model_payload = case.request.get('model_hamiltonian')
            model_spec = model_payload.get('spec') if isinstance(model_payload, dict) else None
            if not isinstance(model_spec, dict):
                issues.append(_issue('error', 'missing_case_model_spec', 'Model Hamiltonian case requests require model_hamiltonian.spec', path + '.request.model_hamiltonian.spec'))
            else:
                for message in validate_model_hamiltonian_spec(model_spec, solver):
                    issues.append(_issue('error', 'invalid_case_model_spec', message, path + '.request.model_hamiltonian.spec'))
                if solver == 'dmet':
                    from pyscf_agent.providers.libdmet import validate_dmet_model_request

                    dmet_errors, _configuration = validate_dmet_model_request(
                        model_spec,
                        _solver_options,
                    )
                    for message in dmet_errors:
                        dmet_issue_cases.setdefault(message, []).append(case.case_id)
                estimate = estimate_case_cost(case, system_type)
                if estimate.get('full_diagonalization') and estimate.get('full_diagonalization_supported') is False:
                    issues.append(_issue(
                        'error',
                        'unsupported_full_fci_diagonalization',
                        'Full FCI diagonalization for diagnostics requires determinant space no larger than {0:,}; this case has {1:,}.'.format(
                            int(estimate.get('full_diagonalization_limit') or 0),
                            int(estimate.get('determinant_count') or 0),
                        ),
                        path + '.request.analysis.outputs',
                    ))
    for message, case_ids in dmet_issue_cases.items():
        affected = (
            'Affects all {0} planned cases.'.format(len(case_ids))
            if len(case_ids) == len(study_plan.cases)
            else 'Affected cases: {0}.'.format(', '.join(case_ids))
        )
        issues.append(_issue(
            'error',
            'invalid_dmet_case_request',
            '{0} {1}'.format(message, affected),
            'cases',
        ))
    return issues


def validate_study_workflow(spec: Any, registry: StudyCapabilityView = None) -> List[ValidationIssue]:
    capability_registry = registry or default_study_capabilities()
    issues = validate_study_spec(spec, capability_registry)
    if has_errors(issues):
        return issues
    try:
        plan = build_study_plan(spec, registry=capability_registry)
    except Exception as exc:
        issues.append(_issue('error', 'study_plan_build_failed', 'Study planning failed: {0}'.format(exc), ''))
        return issues
    issues.extend(validate_study_plan(plan, capability_registry))
    return issues


def assert_valid_study_plan(plan: Any, registry: StudyCapabilityView = None) -> None:
    issues = validate_study_plan(plan, registry)
    if has_errors(issues):
        messages = '; '.join('{0}: {1}'.format(issue.path or issue.code, issue.message) for issue in issues if issue.severity == 'error')
        raise ValueError(messages)
