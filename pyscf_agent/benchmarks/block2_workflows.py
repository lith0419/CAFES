from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .fixtures import hubbard_ring_spec as _hubbard_ring_spec
from .fixtures import block2_unavailable_outcome as _unavailable_outcome


BLOCK2_WORKFLOW_BENCHMARK = {
    'id': 'block2-adaptive-workflows',
    'domain': 'molecular',
    'tier': 'optional_provider',
    'description': (
        'Block2 joint DMRG-CASSCF continuation, adaptive bond-dimension '
        'escalation, error estimation, and entanglement-driven active-space review.'
    ),
    'runner': 'block2_adaptive_workflows',
    'source': {
        'reference': (
            'Cold-start DMRG-CASSCF and exact diagonalization of the identical '
            'small active Hamiltonians.'
        ),
        'provider_citation': 'Zhai et al., J. Chem. Phys. 159, 234801 (2023).',
        'doi': '10.1063/5.0180424',
    },
    'tolerances': {
        'joint_restart_energy_abs_Ha': 1e-8,
        'adaptive_ring_energy_abs_Ha': 1e-7,
        'adaptive_ring_estimated_error_Ha': 1e-5,
    },
}


def _molecular_options(**updates: Any) -> Dict[str, Any]:
    options = {
        'preset': 'screening',
        'bond_dimensions': [32] * 6,
        'noises': [1e-5, 1e-6, 0.0, 0.0, 0.0, 0.0],
        'davidson_thresholds': [1e-9] * 6,
        'sweeps': 6,
        'energy_tolerance': 1e-8,
        'discarded_weight_tolerance': 1e-7,
        'adaptive_schedule': True,
        'max_adaptive_stages': 1,
        'max_bond_dimension': 64,
        'adaptive_sweeps': 4,
        'adaptive_noise': 1e-5,
        'save_mps': True,
        'compute_entanglement': True,
        'compute_mutual_information': True,
        'entanglement_active_space_review': True,
        'symmetry': 'su2',
    }
    options.update(updates)
    return options


def _h2_request(distance: float, options: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'task_type': 'molecular',
        'system': {
            'atom': 'H 0 0 0; H 0 0 {0:.12g}'.format(float(distance)),
            'basis': '6-31g',
            'unit': 'Angstrom',
            'charge': 0,
            'spin': 0,
        },
        'method': {'name': 'casscf', 'restricted': True},
        'solver': {'name': 'block2_dmrg', 'options': copy.deepcopy(options)},
        'active_space': {
            'enabled': True,
            'ncas': 2,
            'nelecas': 2,
            'orbital_indices': [0, 1],
            'approved': True,
        },
        'analysis': {'outputs': ['energy']},
    }


def _run_molecular_workflow(root: Path) -> Dict[str, Any]:
    from computational_study_agent.adaptive import (
        build_entanglement_active_space_review_plan,
    )
    from computational_study_agent.schema import StudyCase, StudyPlan
    from ..backend.execution import _run_pyscf_task
    from ..contracts import task_spec_from_dict

    source_request = _h2_request(1.5, _molecular_options())
    source = _run_pyscf_task(
        task_spec_from_dict(source_request),
        execution_directory=root / 'h2-source',
    )
    source_dmrg = ((source.get('cas_result') or {}).get('dmrg_result') or {})
    manifest = source_dmrg.get('checkpoint_manifest')

    cold_request = _h2_request(1.5, _molecular_options())
    cold = _run_pyscf_task(
        task_spec_from_dict(cold_request),
        execution_directory=root / 'h2-cold',
    )
    warm_request = _h2_request(
        1.5,
        _molecular_options(
            restart_manifest=copy.deepcopy(manifest),
            restart_required=True,
            restart_provenance={
                'source_case_id': 'case-0001',
                'selection': 'nearest_compatible_dmrg_casscf_checkpoint',
            },
        ),
    )
    warm = _run_pyscf_task(
        task_spec_from_dict(warm_request),
        execution_directory=root / 'h2-warm',
    )
    warm_dmrg = ((warm.get('cas_result') or {}).get('dmrg_result') or {})

    study_plan = StudyPlan(
        study_id='block2-workflow-benchmark',
        name='block2-workflow-benchmark',
        objective='Benchmark entanglement-driven active-space review',
        system_type='molecular',
        cases=[StudyCase(
            case_id='case-0001',
            label='R=1.5 Angstrom',
            request=copy.deepcopy(source_request),
            variables={'bond_length': 1.5},
        )],
        observables=['energy'],
    )
    review = build_entanglement_active_space_review_plan(
        study_plan,
        {
            'cases': [{
                'case_id': 'case-0001',
                'task_report': {'structured_results': copy.deepcopy(source)},
            }],
        },
    )
    return {
        'source': source,
        'source_dmrg': source_dmrg,
        'cold': cold,
        'warm': warm,
        'warm_dmrg': warm_dmrg,
        'review': review,
    }


def _run_adaptive_ring(root: Path) -> Dict[str, Any]:
    from ..backend.model_hamiltonian.solver import run_model_hamiltonian_solver

    spec = _hubbard_ring_spec(8)
    exact = run_model_hamiltonian_solver(
        spec,
        solver_name='fci',
        outputs=['energy'],
    )
    adaptive = run_model_hamiltonian_solver(
        spec,
        solver_name='block2_dmrg',
        outputs=['energy'],
        solver_options={
            'preset': 'screening',
            'bond_dimensions': [16, 16],
            'noises': [1e-4, 0.0],
            'davidson_thresholds': [1e-8, 1e-8],
            'sweeps': 2,
            'energy_tolerance': 1e-6,
            'discarded_weight_tolerance': 1e-5,
            'adaptive_schedule': True,
            'max_adaptive_stages': 3,
            'max_bond_dimension': 128,
            'bond_dimension_growth_factor': 2.0,
            'adaptive_sweeps': 8,
            'adaptive_noise': 1e-4,
            'estimate_energy_error': True,
            'save_mps': False,
            'symmetry': 'su2',
        },
        scratch_directory=str(root / 'hubbard-ring'),
    )
    return {'exact': exact, 'adaptive': adaptive}


def run_block2_adaptive_workflows(
    definition: Dict[str, Any],
    work_dir: Optional[Path],
) -> Dict[str, Any]:
    unavailable = _unavailable_outcome()
    if unavailable:
        return unavailable
    root = work_dir or Path(tempfile.mkdtemp(prefix='pyscf-agent-block2-workflow-'))
    root.mkdir(parents=True, exist_ok=True)
    molecular = _run_molecular_workflow(root)
    ring = _run_adaptive_ring(root)

    source_dmrg = molecular['source_dmrg']
    warm_dmrg = molecular['warm_dmrg']
    manifest = source_dmrg.get('checkpoint_manifest') or {}
    casscf = warm_dmrg.get('casscf') or {}
    review = molecular.get('review') or {}
    review_plan = review.get('entanglement_active_space_plan') or {}
    review_cases = review_plan.get('cases') or []
    review_case = review_cases[0] if review_cases else {}
    review_active_space = review_case.get('request', {}).get('active_space') or {}
    review_options = review_case.get('request', {}).get('solver', {}).get('options') or {}
    warm_cold_error = abs(float(molecular['warm']['energy']) - float(molecular['cold']['energy']))

    adaptive = ring['adaptive']
    adaptive_dmrg = adaptive.get('dmrg_result') or {}
    schedule = adaptive_dmrg.get('adaptive_schedule') or {}
    estimate = adaptive_dmrg.get('energy_error_estimate') or {}
    ring_error = abs(float(adaptive['energy']) - float(ring['exact']['energy']))
    estimated_error = estimate.get('estimated_absolute_error')
    tolerances = definition['tolerances']

    joint_checkpoint_ok = bool(
        set(manifest.get('checkpoint_components') or ()) == {'mps', 'optimized_orbitals'}
        and casscf.get('joint_checkpoint_restart_applied')
        and casscf.get('joint_checkpoint_restart_provenance', {}).get(
            'source_case_id'
        ) == 'case-0001'
    )
    active_space_review_ok = bool(
        review_cases
        and review_active_space.get('approved') is False
        and review_active_space.get('audit', {}).get(
            'ncas_nelecas_consistency', {}
        ).get('consistent') is True
        and review_options.get('orbital_restart_manifest')
        and not review_options.get('restart_manifest')
        and review_plan.get('cost_estimate')
    )
    adaptive_ok = bool(
        adaptive.get('converged')
        and schedule.get('adaptive_stages_used', 0) >= 1
        and schedule.get('final_bond_dimension', 0) > 16
        and estimate.get('status') == 'available'
        and estimated_error is not None
        and float(estimated_error) <= tolerances['adaptive_ring_estimated_error_Ha']
        and ring_error <= tolerances['adaptive_ring_energy_abs_Ha']
    )
    return {
        'metrics': {
            'joint_checkpoint_components': manifest.get('checkpoint_components'),
            'joint_checkpoint_restart_applied': joint_checkpoint_ok,
            'joint_restart_energy_abs_error_Ha': warm_cold_error,
            'active_space_review_case_count': len(review_cases),
            'recommended_active_space': {
                'ncas': review_active_space.get('ncas'),
                'nelecas': review_active_space.get('nelecas'),
                'approved': review_active_space.get('approved'),
            },
            'review_reuses_optimized_orbitals': bool(
                review_options.get('orbital_restart_manifest')
            ),
            'review_reuses_mps': bool(review_options.get('restart_manifest')),
            'adaptive_ring_fci_energy_Ha': ring['exact']['energy'],
            'adaptive_ring_block2_energy_Ha': adaptive['energy'],
            'adaptive_ring_energy_abs_error_Ha': ring_error,
            'adaptive_stages_used': schedule.get('adaptive_stages_used'),
            'adaptive_final_bond_dimension': schedule.get('final_bond_dimension'),
            'adaptive_energy_error_estimate': copy.deepcopy(estimate),
        },
        'checks': [
            {
                'name': 'joint_orbital_mps_continuation',
                'passed': (
                    joint_checkpoint_ok
                    and warm_cold_error <= tolerances['joint_restart_energy_abs_Ha']
                ),
                'value': warm_cold_error,
                'limit': tolerances['joint_restart_energy_abs_Ha'],
            },
            {
                'name': 'entanglement_active_space_review_contract',
                'passed': active_space_review_ok,
                'value': {
                    'review_case_count': len(review_cases),
                    'approved': review_active_space.get('approved'),
                    'orbital_restart_only': bool(
                        review_options.get('orbital_restart_manifest')
                        and not review_options.get('restart_manifest')
                    ),
                },
                'expected': 'one unapproved, costed ActiveSpaceAudit using optimized orbitals only',
            },
            {
                'name': 'adaptive_bond_dimension_ed_accuracy',
                'passed': adaptive_ok,
                'value': {
                    'energy_abs_error_Ha': ring_error,
                    'estimated_absolute_error_Ha': estimated_error,
                    'adaptive_stages_used': schedule.get('adaptive_stages_used'),
                    'final_bond_dimension': schedule.get('final_bond_dimension'),
                },
                'limit': {
                    'energy_abs_error_Ha': tolerances['adaptive_ring_energy_abs_Ha'],
                    'estimated_absolute_error_Ha': tolerances[
                        'adaptive_ring_estimated_error_Ha'
                    ],
                },
            },
        ],
    }


__all__ = [
    'BLOCK2_WORKFLOW_BENCHMARK',
    'run_block2_adaptive_workflows',
]
