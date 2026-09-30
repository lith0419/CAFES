from __future__ import annotations

import copy
from typing import Any, Dict, Optional


CAS_METHODS = ('casci', 'casscf')


def normalize_active_space_solver_name(value: Any) -> Optional[str]:
    normalized = str(value or '').strip().lower().replace('-', '_')
    if normalized in ('block2', 'dmrg'):
        normalized = 'block2_dmrg'
    return normalized if normalized in ('fci', 'block2_dmrg') else None


def active_space_solver_contract_from_request(
    request: Dict[str, Any],
) -> tuple[Optional[str], Dict[str, Any]]:
    """Read an explicit CAS solver without replacing scientific intent."""

    if not isinstance(request, dict):
        return None, {}
    method = str(request.get('method') or '').strip().lower()
    solver = request.get('solver')
    solver_options: Dict[str, Any] = {}
    if isinstance(solver, dict):
        solver_name = normalize_active_space_solver_name(solver.get('name'))
        if isinstance(solver.get('options'), dict):
            solver_options = copy.deepcopy(solver['options'])
    else:
        solver_name = normalize_active_space_solver_name(solver)
    if method in CAS_METHODS and solver_name:
        return solver_name, solver_options

    configured = normalize_active_space_solver_name(request.get('active_space_solver'))
    if configured:
        options = request.get('active_space_solver_options')
        return configured, copy.deepcopy(options) if isinstance(options, dict) else {}

    active_space = request.get('active_space')
    active_space = active_space if isinstance(active_space, dict) else {}
    target_solver = normalize_active_space_solver_name(active_space.get('target_solver'))
    target_options = active_space.get('target_solver_options')
    if target_solver:
        return (
            target_solver,
            copy.deepcopy(target_options) if isinstance(target_options, dict) else {},
        )
    return None, {}


def structured_results_from_case_report(case_report: Dict[str, Any]) -> Dict[str, Any]:
    task_report = case_report.get('task_report') if isinstance(case_report, dict) else {}
    structured = task_report.get('structured_results') if isinstance(task_report, dict) else {}
    return structured if isinstance(structured, dict) else {}


def active_space_contract_from_case_report(
    case_report: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Extract the review contract produced by the shared ActiveSpaceAudit."""

    active_space = structured_results_from_case_report(case_report).get('active_space')
    if not isinstance(active_space, dict):
        return None
    if active_space.get('ncas') is None or active_space.get('nelecas') is None:
        return None
    orbital_indices = active_space.get('orbital_indices')
    if not orbital_indices:
        return None
    contract = {
        'enabled': True,
        'selection_method': 'manual',
        'ncas': active_space.get('ncas'),
        'nelecas': copy.deepcopy(active_space.get('nelecas')),
        'orbital_indices': copy.deepcopy(orbital_indices),
        'initial_mo_coeff': copy.deepcopy(active_space.get('initial_mo_coeff')),
        'approved': False,
    }
    if isinstance(active_space.get('audit_summary'), dict):
        contract['audit_summary'] = copy.deepcopy(active_space['audit_summary'])
    if isinstance(active_space.get('audit'), dict):
        audit = active_space['audit']
        selected_method = str(
            audit.get('selected_candidate_method')
            or (active_space.get('audit_summary') or {}).get('selected_candidate_method')
            or ''
        ).strip().lower()
        candidate_rows = audit.get('candidate_active_spaces')
        if selected_method and isinstance(candidate_rows, list):
            candidate_options = []
            for item in candidate_rows:
                if not isinstance(item, dict):
                    continue
                method_name = str(item.get('method') or '').strip().lower()
                if method_name in ('frontier_fallback', 'unresolved'):
                    continue
                if item.get('status') not in ('available', 'mapping_ambiguous'):
                    continue
                option_indices = item.get('orbital_indices')
                if not isinstance(option_indices, list) or not option_indices:
                    continue
                candidate_options.append({
                    key: copy.deepcopy(item.get(key))
                    for key in (
                        'method',
                        'role',
                        'status',
                        'ncas',
                        'estimated_nelecas',
                        'orbital_indices',
                        'initial_mo_coeff',
                        'target_summary',
                        'resolved_targets',
                        'projection_compaction',
                        'orbital_index_basis',
                    )
                    if item.get(key) is not None
                })
            if candidate_options:
                contract['candidate_options'] = candidate_options
            selected_candidate = next(
                (
                    item for item in candidate_rows
                    if isinstance(item, dict)
                    and str(item.get('method') or '').strip().lower() == selected_method
                ),
                None,
            )
            if isinstance(selected_candidate, dict):
                provenance_keys = (
                    'method',
                    'role',
                    'status',
                    'ncas',
                    'estimated_nelecas',
                    'target_summary',
                    'resolved_targets',
                    'projection_compaction',
                    'orbital_index_basis',
                )
                contract['candidate_provenance'] = {
                    key: copy.deepcopy(selected_candidate.get(key))
                    for key in provenance_keys
                    if selected_candidate.get(key) is not None
                }
        method_recommendation = audit.get('method_recommendation')
        if isinstance(method_recommendation, dict):
            contract['method_recommendation'] = copy.deepcopy(method_recommendation)
        consistency = audit.get('ncas_nelecas_consistency')
        if isinstance(consistency, dict):
            contract['audit'] = {
                'ncas_nelecas_consistency': copy.deepcopy(consistency),
            }
        orbital_rows = audit.get('orbitals')
        if isinstance(orbital_rows, list):
            expansion_orbitals = []
            for row in orbital_rows:
                if not isinstance(row, dict) or row.get('index') is None:
                    continue
                expansion_orbitals.append({
                    'index': row.get('index'),
                    'occupation': row.get('occupation'),
                    'occupation_alpha': row.get('occupation_alpha'),
                    'occupation_beta': row.get('occupation_beta'),
                })
            if expansion_orbitals:
                unrestricted_index_proxy = any(
                    row.get('occupation_alpha') is not None
                    or row.get('occupation_beta') is not None
                    for row in expansion_orbitals
                )
                contract['expansion_evidence'] = {
                    'orbital_count': len(expansion_orbitals),
                    'orbitals': expansion_orbitals,
                    'source': 'ActiveSpaceAudit canonical reference occupations',
                    'orbital_mapping': (
                        'unrestricted_index_proxy'
                        if unrestricted_index_proxy
                        else 'spatial_orbitals'
                    ),
                }
    return contract


__all__ = [
    'active_space_contract_from_case_report',
    'active_space_solver_contract_from_request',
    'normalize_active_space_solver_name',
    'structured_results_from_case_report',
]
