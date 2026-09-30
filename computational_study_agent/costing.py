from __future__ import annotations

from pyscf_agent.serialization import json_default

import copy
import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pyscf_agent.registry import default_registry as default_platform_capability_registry
from pyscf_agent.backend.model_hamiltonian.fci_observables import full_diagonalization_resource_estimate


DEFAULT_RESOURCE_POLICY = {
    # These are review gates, not hard numerical accuracy limits.  They make
    # an expensive exact/CAS request explicit before the executor allocates it.
    'determinant_review_threshold': 1_000_000,
    'total_work_review_threshold': 10_000_000,
    'memory_review_mb': 512,
    # Work units are method-dependent scaling proxies. They remain useful in
    # reports, but are not scheduler limits and therefore do not gate by
    # default. A caller may explicitly opt into that policy.
    'review_work_estimates': False,
    'memory_limit_mb': None,
    'memory_limit_source': None,
    'resource_profile': None,
    'approved': False,
}


class CostApprovalRequired(RuntimeError):
    """Raised before an expensive plan is submitted to the task executor."""

    def __init__(self, estimate: Dict[str, Any]):
        self.estimate = copy.deepcopy(estimate)
        prefix = (
            'Plan exceeds the selected execution resource limit'
            if estimate.get('resource_limit_exceeded')
            else 'Plan requires explicit resource approval'
        )
        super().__init__(
            '{0}: {1}'.format(
                prefix,
                estimate.get('review_reason') or 'estimated computational cost exceeds the configured review threshold'
            )
        )


def normalize_resource_policy(payload: Any = None) -> Dict[str, Any]:
    source = payload if isinstance(payload, dict) else {}
    policy = copy.deepcopy(DEFAULT_RESOURCE_POLICY)
    for key in (
        'determinant_review_threshold',
        'total_work_review_threshold',
        'memory_review_mb',
    ):
        try:
            value = int(source.get(key, policy[key]))
        except (TypeError, ValueError):
            value = policy[key]
        policy[key] = max(1, value)
    policy['review_work_estimates'] = bool(source.get('review_work_estimates', False))
    try:
        memory_limit_mb = int(source.get('memory_limit_mb'))
    except (TypeError, ValueError):
        memory_limit_mb = None
    policy['memory_limit_mb'] = memory_limit_mb if memory_limit_mb and memory_limit_mb > 0 else None
    policy['memory_limit_source'] = str(source.get('memory_limit_source') or '').strip() or None
    policy['resource_profile'] = str(source.get('resource_profile') or '').strip() or None
    policy['approved'] = bool(source.get('approved', False))
    return policy


def _comb(norb: Optional[int], nelec: Optional[int]) -> Optional[int]:
    if norb is None or nelec is None or norb < 0 or nelec < 0 or nelec > norb:
        return None
    try:
        return math.comb(int(norb), int(nelec))
    except (TypeError, ValueError):
        return None


def _electron_partition(total_electrons: Any, spin: Any = 0) -> Tuple[Optional[int], Optional[int]]:
    if isinstance(total_electrons, (list, tuple)) and len(total_electrons) >= 2:
        try:
            return int(total_electrons[0]), int(total_electrons[1])
        except (TypeError, ValueError):
            return None, None
    try:
        total = int(total_electrons)
        spin_value = int(spin or 0)
    except (TypeError, ValueError):
        return None, None
    if total < 0 or abs(spin_value) > total or (total + spin_value) % 2:
        return None, None
    return (total + spin_value) // 2, (total - spin_value) // 2


def _atom_electron_count(atom: Any, charge: Any) -> Optional[int]:
    """Use PySCF when possible; return unknown rather than guessing chemistry."""
    try:
        from pyscf import gto  # pylint: disable=import-outside-toplevel

        mol = gto.M(
            atom=atom,
            basis='sto-3g',
            charge=int(charge or 0),
            spin=0,
            verbose=0,
        )
        return int(mol.nelectron)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, KeyError):
        return None


def _molecular_dimensions(request: Dict[str, Any]) -> Dict[str, Any]:
    system = request.get('system') if isinstance(request.get('system'), dict) else request
    atom = system.get('atom') if isinstance(system, dict) else None
    basis = system.get('basis') if isinstance(system, dict) else None
    charge = system.get('charge', request.get('charge', 0)) if isinstance(system, dict) else request.get('charge', 0)
    spin = system.get('spin', request.get('spin', 0)) if isinstance(system, dict) else request.get('spin', 0)
    result: Dict[str, Any] = {
        'norb': None,
        'nalpha': None,
        'nbeta': None,
        'source': 'unavailable',
    }
    if not atom or not basis:
        return result
    try:
        from pyscf import gto  # pylint: disable=import-outside-toplevel

        mol = gto.M(atom=atom, basis=basis, charge=int(charge or 0), spin=int(spin or 0), verbose=0)
        result.update({
            'norb': int(mol.nao_nr()),
            'nalpha': int(mol.nelec[0]),
            'nbeta': int(mol.nelec[1]),
            'source': 'pyscf_basis_count',
        })
        return result
    except Exception:
        total = _atom_electron_count(atom, charge)
        nalpha, nbeta = _electron_partition(total, spin)
        result.update({'nalpha': nalpha, 'nbeta': nbeta, 'source': 'electron_count_only'})
        return result


def _model_dimensions(request: Dict[str, Any]) -> Dict[str, Any]:
    model = request.get('model_hamiltonian') if isinstance(request.get('model_hamiltonian'), dict) else {}
    spec = model.get('spec') if isinstance(model.get('spec'), dict) else request.get('model_spec')
    spec = spec if isinstance(spec, dict) else {}
    sites = spec.get('sites') if isinstance(spec.get('sites'), list) else []
    nalpha, nbeta = _electron_partition(spec.get('nelec'))
    return {
        'norb': len(sites) or None,
        'nalpha': nalpha,
        'nbeta': nbeta,
        'source': 'model_hamiltonian_sites' if sites else 'unavailable',
    }


def _active_space_dimensions(request: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    active = request.get('active_space') if isinstance(request.get('active_space'), dict) else {}
    try:
        ncas = int(active.get('ncas'))
    except (TypeError, ValueError):
        return None, None, None
    nalpha, nbeta = _electron_partition(active.get('nelecas'), request.get('spin', 0))
    return ncas, nalpha, nbeta


def _method_name(request: Dict[str, Any], system_type: str) -> str:
    registry = default_platform_capability_registry()
    if system_type == 'model_hamiltonian':
        raw = request.get('solver')
        if isinstance(raw, dict):
            raw = raw.get('name')
        return registry.canonical_capability_id(
            raw, namespace='model_hamiltonian.solver'
        ) or str(raw or '').strip().lower()
    raw = request.get('method')
    return registry.canonical_capability_id(
        raw, namespace='molecular.method'
    ) or str(raw or '').strip().lower()


def _solver_name(request: Dict[str, Any]) -> str:
    raw = request.get('solver')
    if isinstance(raw, dict):
        raw = raw.get('name')
    normalized = str(raw or '').strip().lower().replace('-', '_')
    if normalized in ('block2', 'dmrg'):
        return 'block2_dmrg'
    return normalized


def _solver_options(request: Dict[str, Any]) -> Dict[str, Any]:
    raw = request.get('solver')
    if isinstance(raw, dict) and isinstance(raw.get('options'), dict):
        return raw['options']
    return {}


def _job_name(request: Dict[str, Any]) -> str:
    raw = request.get('job')
    if isinstance(raw, dict):
        raw = raw.get('name')
    return str(raw or 'single_point').strip().lower().replace('-', '_')


def _molecular_dynamics_cost_factors(request: Dict[str, Any]) -> Tuple[int, int]:
    molecular_dynamics = (
        request.get('molecular_dynamics')
        if isinstance(request.get('molecular_dynamics'), dict)
        else {}
    )
    try:
        steps = max(1, int(molecular_dynamics.get('steps') or 100))
    except (TypeError, ValueError):
        steps = 100
    try:
        stride = max(1, int(molecular_dynamics.get('sample_stride') or 10))
        offset = max(0, int(molecular_dynamics.get('sample_offset', 9)))
    except (TypeError, ValueError):
        stride, offset = 10, 9
    sampled_frames = len(range(offset, steps, stride)) if offset < steps else 0
    return steps, sampled_frames


def _dmet_block2_dimensions(
    request: Dict[str, Any],
    dimensions: Dict[str, Any],
    options: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[int]]:
    """Estimate the largest embedded impurity, including its bath orbitals."""
    model = request.get('model_hamiltonian') if isinstance(request.get('model_hamiltonian'), dict) else {}
    spec = model.get('spec') if isinstance(model.get('spec'), dict) else request.get('model_spec')
    spec = spec if isinstance(spec, dict) else {}
    primitive_cell = spec.get('primitive_cell') if isinstance(spec.get('primitive_cell'), dict) else {}
    try:
        basis_size = max(1, int(primitive_cell.get('basis_size') or 1))
    except (TypeError, ValueError):
        basis_size = 1

    fragment_sizes: List[int] = []
    if options.get('fragment_definition') == 'honeycomb_hexagon':
        fragment_sizes.append(6)
    for fragment in options.get('fragments') or []:
        if not isinstance(fragment, dict):
            continue
        indices = fragment.get('orbital_indices') or fragment.get('site_ids') or []
        if isinstance(indices, (list, tuple)) and indices:
            fragment_sizes.append(len(indices))
    impurity_site_ids = options.get('impurity_site_ids') or []
    if isinstance(impurity_site_ids, (list, tuple)) and impurity_site_ids:
        fragment_sizes.append(len(impurity_site_ids))
    try:
        impurity_size = int(options.get('impurity_size'))
    except (TypeError, ValueError):
        impurity_size = 0
    if impurity_size > 0:
        fragment_sizes.append(impurity_size)
    shape = options.get('impurity_shape') or []
    if isinstance(shape, (list, tuple)) and shape:
        try:
            fragment_sizes.append(basis_size * math.prod(max(1, int(item)) for item in shape))
        except (TypeError, ValueError):
            pass
    if not fragment_sizes:
        fragment_sizes.append(basis_size)

    fragment_norb = max(fragment_sizes)
    total_norb = dimensions.get('norb')
    embedded_norb = 2 * fragment_norb
    if total_norb is not None:
        embedded_norb = min(int(total_norb), embedded_norb)
    embedded_norb = max(1, int(embedded_norb))
    total_electrons = None
    if dimensions.get('nalpha') is not None and dimensions.get('nbeta') is not None:
        total_electrons = int(dimensions['nalpha']) + int(dimensions['nbeta'])
    embedded_electrons = (
        min(embedded_norb, max(0, int(round(total_electrons * embedded_norb / int(total_norb)))))
        if total_electrons is not None and total_norb
        else None
    )
    nalpha, nbeta = _electron_partition(embedded_electrons, 0)
    return {
        'norb': embedded_norb,
        'nalpha': nalpha,
        'nbeta': nbeta,
        'source': 'dmet_fragment_plus_bath_upper_bound',
    }, fragment_norb


def _model_analysis_outputs(request: Dict[str, Any]) -> List[str]:
    analysis = request.get('analysis') if isinstance(request.get('analysis'), dict) else {}
    outputs = analysis.get('outputs') if isinstance(analysis.get('outputs'), list) else []
    return [str(item).strip().lower() for item in outputs if str(item).strip()]


def _requires_full_fci_diagonalization(request: Dict[str, Any], system_type: str, method: str) -> bool:
    if system_type != 'model_hamiltonian' or method != 'fci':
        return False
    outputs = _model_analysis_outputs(request)
    return any(output in outputs for output in (
        'strong_correlation_diagnostics',
        'gap',
        'energy_levels',
        'energy_level_count',
        'many_body_basis_dimension',
    ))


def estimate_case_cost(case: Any, system_type: str) -> Dict[str, Any]:
    """Estimate relative work using only the dimensions available in a plan.

    The estimate is deliberately transparent: it is a planning safeguard, not
    a scheduler or a promise of wall time on a particular machine.
    """
    request = getattr(case, 'request', None) if not isinstance(case, dict) else case.get('request')
    request = request if isinstance(request, dict) else {}
    normalized_system = str(system_type or '').strip().lower()
    method = _method_name(request, normalized_system)
    solver = _solver_name(request)
    job = _job_name(request)
    dimensions = _model_dimensions(request) if normalized_system == 'model_hamiltonian' else _molecular_dimensions(request)
    norb = dimensions.get('norb')
    nalpha = dimensions.get('nalpha')
    nbeta = dimensions.get('nbeta')
    model = 'scaling_proxy'
    determinant_count = None
    work_units = None
    memory_mb = None
    active_norb = None
    full_diagonalization = False
    full_diagonalization_details: Dict[str, Any] = {}
    bond_dimension = None
    sweeps = None
    nroots = None
    impurity_solver = None
    fragment_norb = None
    trajectory_steps = None
    sampled_frame_count = None
    block2_options: Optional[Dict[str, Any]] = None

    if solver == 'block2_dmrg':
        block2_options = _solver_options(request)
    elif solver == 'dmet':
        dmet_options = _solver_options(request)
        raw_impurity_solver = str(dmet_options.get('impurity_solver') or 'fci').strip().lower().replace('-', '_')
        if raw_impurity_solver in ('block2', 'dmrg', 'block2_dmrg'):
            impurity_solver = 'block2_dmrg'
            nested_options = dmet_options.get('impurity_solver_options')
            block2_options = nested_options if isinstance(nested_options, dict) else {}
            dimensions, fragment_norb = _dmet_block2_dimensions(request, dimensions, dmet_options)
            norb = dimensions.get('norb')
            nalpha = dimensions.get('nalpha')
            nbeta = dimensions.get('nbeta')

    if block2_options is not None:
        active_norb, active_alpha, active_beta = _active_space_dimensions(request)
        if active_norb is not None:
            norb, nalpha, nbeta = active_norb, active_alpha, active_beta
        from pyscf_agent.providers.block2.config import normalize_block2_options

        config = normalize_block2_options(block2_options)
        bond_dimension = max(config.bond_dimensions)
        sweeps = config.sweeps
        nroots = config.nroots
        if norb is not None:
            # Transparent tensor-network proxy. It represents contractions and
            # sweep storage, not a wall-time prediction for a particular node.
            work_units = int(nroots * sweeps * ((norb ** 3) * (bond_dimension ** 3) + (norb ** 4) * (bond_dimension ** 2)))
            memory_mb = round(8 * max(1, norb) * (bond_dimension ** 2) * 16 / (1024 ** 2), 2)
            model = (
                'dmet_block2_impurity_sweep_proxy'
                if impurity_solver == 'block2_dmrg'
                else 'block2_dmrg_sweep_proxy'
            )
    elif method in ('fci', 'casci', 'casscf'):
        if method in ('casci', 'casscf'):
            active_norb, active_alpha, active_beta = _active_space_dimensions(request)
            if active_norb is not None:
                norb, nalpha, nbeta = active_norb, active_alpha, active_beta
            model = 'cas_determinant_space'
        else:
            model = 'fci_determinant_space'
        alpha_strings = _comb(norb, nalpha)
        beta_strings = _comb(norb, nbeta)
        if alpha_strings is not None and beta_strings is not None:
            determinant_count = alpha_strings * beta_strings
            macro_factor = 20 if method == 'casscf' else 1
            work_units = determinant_count * macro_factor
            # CI vector plus Davidson/work buffers.  It is intentionally a
            # lower-bound style estimate, not a precise memory reservation.
            memory_mb = round(determinant_count * 8 * (6 if method == 'casscf' else 4) / (1024 ** 2), 2)
            full_diagonalization = _requires_full_fci_diagonalization(request, normalized_system, method)
            if full_diagonalization:
                full_diagonalization_details = full_diagonalization_resource_estimate(determinant_count)
                model = 'fci_full_diagonalization'
                work_units = full_diagonalization_details['work_units']
                memory_mb = full_diagonalization_details['working_memory_mb']
    elif method in ('mp2', 'ccsd', 'ccsd_t'):
        if norb is not None and nalpha is not None and nbeta is not None:
            nocc = max(nalpha, nbeta)
            nvir = max(0, norb - nocc)
            doubles = nocc * nocc * nvir * nvir
            multiplier = {'mp2': 1, 'ccsd': 12, 'ccsd_t': 40}.get(method, 1)
            work_units = doubles * multiplier
            memory_mb = round(doubles * 8 * (2 if method == 'mp2' else 8) / (1024 ** 2), 2)
            model = '{0}_tensor_proxy'.format(method)
    elif method in ('hf', 'dft') and norb is not None:
        # SCF and analytic-gradient work are represented by a transparent
        # quartic AO proxy.  BOMD performs one such electronic-structure and
        # gradient evaluation per integrator frame.
        work_units = max(1, int(norb)) ** 4
        matrix_count = 8
        model = '{0}_scf_proxy'.format(method)
        if job == 'molecular_dynamics':
            trajectory_steps, sampled_frame_count = _molecular_dynamics_cost_factors(request)
            work_units *= trajectory_steps
            matrix_count += 2 * sampled_frame_count
            model = 'bomd_dft_gradient_proxy'
        memory_mb = round(
            max(1, int(norb)) ** 2 * 8 * matrix_count / (1024 ** 2),
            2,
        )

    status = 'estimated' if work_units is not None else 'unavailable'
    if full_diagonalization and not full_diagonalization_details.get('supported'):
        status = 'unsupported'
    return {
        'case_id': getattr(case, 'case_id', None) if not isinstance(case, dict) else case.get('case_id'),
        'label': getattr(case, 'label', None) if not isinstance(case, dict) else case.get('label'),
        'method': method or None,
        'job': job or None,
        'solver': solver or None,
        'impurity_solver': impurity_solver,
        'system_type': normalized_system,
        'estimation_model': model,
        'norb': norb,
        'active_norb': active_norb,
        'fragment_norb': fragment_norb,
        'trajectory_steps': trajectory_steps,
        'sampled_frame_count': sampled_frame_count,
        'nalpha': nalpha,
        'nbeta': nbeta,
        'dimension_source': dimensions.get('source'),
        'determinant_count': determinant_count,
        'work_units': work_units,
        'memory_mb': memory_mb,
        'full_diagonalization': full_diagonalization,
        'full_diagonalization_supported': (
            full_diagonalization_details.get('supported') if full_diagonalization else None
        ),
        'full_diagonalization_limit': (
            full_diagonalization_details.get('max_dimension') if full_diagonalization else None
        ),
        'dense_matrix_memory_mb': (
            full_diagonalization_details.get('matrix_memory_mb') if full_diagonalization else None
        ),
        'estimate_scope': ('initial_dmrg_schedule_only' if block2_options is not None else 'method_proxy'),
        'excluded_stages': (['adaptive_dmrg_extensions', 'final_dmrg_solve', 'casscf_orbital_iterations']
                            if block2_options is not None else []),
        'bond_dimension': bond_dimension,
        'sweeps': sweeps,
        'nroots': nroots,
        'status': status,
    }


def _gate_reasons(
    case_estimates: Iterable[Dict[str, Any]],
    totals: Dict[str, Any],
    policy: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    reasons: List[str] = []
    hard_limit_reasons: List[str] = []
    memory_limit_mb = policy.get('memory_limit_mb')
    for item in case_estimates:
        determinant_count = item.get('determinant_count')
        memory_mb = item.get('memory_mb')
        case_id = item.get('case_id') or 'case'
        if item.get('full_diagonalization') and item.get('full_diagonalization_supported') is False:
            hard_limit_reasons.append(
                '{0}: full FCI diagonalization requires determinant space no larger than {1:,}'.format(
                    case_id,
                    int(item.get('full_diagonalization_limit') or 0),
                )
            )
        if determinant_count is not None and determinant_count >= policy['determinant_review_threshold']:
            reasons.append('{0}: determinant space {1:,} reaches the review threshold'.format(case_id, determinant_count))
        if memory_mb is not None and memory_limit_mb is not None and memory_mb > memory_limit_mb:
            hard_limit_reasons.append(
                '{0}: estimated working memory {1:.1f} MB exceeds the selected Slurm allocation of {2:,} MB'.format(
                    case_id,
                    memory_mb,
                    int(memory_limit_mb),
                )
            )
        elif memory_mb is not None and memory_limit_mb is None and memory_mb >= policy['memory_review_mb']:
            reasons.append('{0}: estimated working memory {1:.1f} MB reaches the review threshold'.format(case_id, memory_mb))
    if (
        policy.get('review_work_estimates')
        and totals['work_units'] is not None
        and totals['work_units'] >= policy['total_work_review_threshold']
    ):
        reasons.append('combined estimated work {0:,} reaches the review threshold'.format(totals['work_units']))
    return reasons, hard_limit_reasons


def estimate_plan_cost(cases: Iterable[Any], system_type: str, resource_policy: Any = None) -> Dict[str, Any]:
    policy = normalize_resource_policy(resource_policy)
    case_estimates = [estimate_case_cost(case, system_type) for case in cases]
    known_work = [item['work_units'] for item in case_estimates if item.get('work_units') is not None]
    known_memory = [item['memory_mb'] for item in case_estimates if item.get('memory_mb') is not None]
    totals = {
        'work_units': sum(known_work) if known_work else None,
        'peak_memory_mb': max(known_memory) if known_memory else None,
        'estimated_case_count': sum(1 for item in case_estimates if item.get('status') == 'estimated'),
        'unavailable_case_count': sum(1 for item in case_estimates if item.get('status') != 'estimated'),
    }
    review_reasons, hard_limit_reasons = _gate_reasons(case_estimates, totals, policy)
    approval_required = bool(review_reasons or hard_limit_reasons)
    resource_limit_exceeded = bool(hard_limit_reasons)
    approved = bool(policy['approved']) and not resource_limit_exceeded
    return {
        'schema': 'pyscf-agent.cost-estimate.v1',
        'version': 1,
        'status': 'blocked' if resource_limit_exceeded else ('review_required' if approval_required and not approved else 'ready'),
        'approval_required': approval_required,
        'approval_allowed': not resource_limit_exceeded,
        'approved': approved,
        'can_execute': not resource_limit_exceeded and (not approval_required or approved),
        'resource_limit_exceeded': resource_limit_exceeded,
        'review_reason': '; '.join(hard_limit_reasons + review_reasons) if approval_required else None,
        'policy': policy,
        'totals': totals,
        'cases': case_estimates,
        'notes': [
            'Estimates are dimension-based planning proxies, not guaranteed wall times.',
            'FCI/CAS estimates use alpha/beta determinant counts; coupled-cluster estimates use tensor scaling proxies.',
            'FCI full-spectrum diagnostics use dense exact diagonalization and include a conservative dense working-set estimate.',
            'Work units are reported as relative scaling proxies and are not an approval gate unless explicitly enabled.',
        ],
    }


def combine_cost_estimates(estimates: Iterable[Dict[str, Any]], resource_policy: Any = None) -> Dict[str, Any]:
    """Aggregate staged adaptive plans without losing their per-stage detail."""
    policy = normalize_resource_policy(resource_policy)
    stage_estimates = [copy.deepcopy(item) for item in estimates if isinstance(item, dict)]
    cases: List[Dict[str, Any]] = []
    for index, estimate in enumerate(stage_estimates, start=1):
        for case in estimate.get('cases') or []:
            if isinstance(case, dict):
                item = copy.deepcopy(case)
                item['stage_index'] = index
                cases.append(item)
    aggregate = estimate_plan_cost([], 'staged', policy)
    aggregate['cases'] = cases
    known_work = [item.get('work_units') for item in cases if item.get('work_units') is not None]
    known_memory = [item.get('memory_mb') for item in cases if item.get('memory_mb') is not None]
    aggregate['totals'] = {
        'work_units': sum(known_work) if known_work else None,
        'peak_memory_mb': max(known_memory) if known_memory else None,
        'estimated_case_count': sum(1 for item in cases if item.get('status') == 'estimated'),
        'unavailable_case_count': sum(1 for item in cases if item.get('status') != 'estimated'),
        'stage_count': len(stage_estimates),
    }
    reasons, hard_limit_reasons = _gate_reasons(cases, aggregate['totals'], policy)
    resource_limit_exceeded = bool(hard_limit_reasons)
    approval_required = bool(reasons or hard_limit_reasons)
    approved = bool(policy['approved']) and not resource_limit_exceeded
    aggregate['approval_required'] = approval_required
    aggregate['approval_allowed'] = not resource_limit_exceeded
    aggregate['approved'] = approved
    aggregate['can_execute'] = not resource_limit_exceeded and (not approval_required or approved)
    aggregate['resource_limit_exceeded'] = resource_limit_exceeded
    aggregate['status'] = 'blocked' if resource_limit_exceeded else ('review_required' if approval_required and not approved else 'ready')
    aggregate['review_reason'] = '; '.join(hard_limit_reasons + reasons) if approval_required else None
    aggregate['stage_estimates'] = stage_estimates
    aggregate['notes'].append('Adaptive totals aggregate every configured initial-scan stage.')
    return aggregate


def _cost_relevant_request(request: Any) -> Dict[str, Any]:
    payload = copy.deepcopy(request) if isinstance(request, dict) else {}
    active_space = payload.get('active_space')
    if isinstance(active_space, dict):
        # Approval changes whether a CAS task may execute, not its dimensions,
        # solver, or estimated resource demand.
        active_space.pop('approved', None)
    return payload


def plan_cost_fingerprint(plan: Any) -> str:
    """Fingerprint only fields that affect the cost and approval decision."""
    cases = []
    for case in getattr(plan, 'cases', []) or []:
        request = getattr(case, 'request', None) if not isinstance(case, dict) else case.get('request')
        case_id = getattr(case, 'case_id', None) if not isinstance(case, dict) else case.get('case_id')
        cases.append({
            'case_id': str(case_id or ''),
            'request': _cost_relevant_request(request),
        })
    resource_policy = normalize_resource_policy(getattr(plan, 'resource_policy', None))
    resource_policy['approved'] = False
    payload = {
        'version': 1,
        'system_type': str(getattr(plan, 'system_type', '') or '').strip().lower(),
        'observables': [str(item).strip().lower() for item in (getattr(plan, 'observables', []) or [])],
        'resource_policy': resource_policy,
        'cases': cases,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=json_default, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def ensure_plan_cost_estimate(plan: Any) -> Dict[str, Any]:
    existing = getattr(plan, 'cost_estimate', None)
    fingerprint = plan_cost_fingerprint(plan)
    existing_is_current = bool(
        isinstance(existing, dict)
        and existing.get('plan_fingerprint') == fingerprint
    )
    existing_approved = bool(
        isinstance(existing, dict)
        and (existing.get('approved') or (existing.get('policy') or {}).get('approved'))
    )
    policy = normalize_resource_policy(getattr(plan, 'resource_policy', None))
    # An explicit policy approval is accepted while first building a plan.  On
    # later runs, approval survives only when the previously reviewed plan has
    # the same cost fingerprint.
    policy['approved'] = bool(policy.get('approved')) if not isinstance(existing, dict) or not existing else (
        existing_is_current and existing_approved
    )
    plan.resource_policy = copy.deepcopy(policy)
    estimate = estimate_plan_cost(
        getattr(plan, 'cases', []),
        getattr(plan, 'system_type', ''),
        policy,
    )
    estimate['plan_fingerprint'] = fingerprint
    if isinstance(existing, dict) and existing and existing_approved and not existing_is_current:
        estimate['approval_invalidated'] = True
        estimate['notes'].append('Previous cost approval was invalidated because the executable plan changed.')
    plan.cost_estimate = estimate
    return estimate


def require_plan_cost_approval(plan: Any) -> Dict[str, Any]:
    estimate = ensure_plan_cost_estimate(plan)
    if not estimate.get('can_execute'):
        raise CostApprovalRequired(estimate)
    return estimate
