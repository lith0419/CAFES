from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set


HARTREE_TO_EV = 27.211386
RUN_ENVIRONMENT_VARIABLE = 'PYSCF_AGENT_RUN_FCDMFT_BENCHMARKS'


FCDMFT_SI_G0W0_BENCHMARK = {
    'id': 'fcdmft-si-g0w0',
    'domain': 'periodic',
    'tier': 'optional_provider',
    'description': (
        'Silicon G0W0@PBE using the periodic fcDMFT example geometry and '
        '4x4x4 k mesh.'
    ),
    'runner': 'fcdmft_si_g0w0',
    'source': {
        'reference': 'fcDMFT examples/Si/si_gw.py',
        'provider_citation': (
            'T. Zhu and G. K.-L. Chan, J. Chem. Theory Comput. 17, '
            '727-741 (2021).'
        ),
        'doi': '10.1021/acs.jctc.0c00704',
        'parameters': {
            'structure': 'two-atom primitive silicon cell, a=5.43 Angstrom',
            'basis': 'gth-dzvp',
            'pseudo': 'gth-pbe',
            'reference_method': 'PBE',
            'kmesh': [4, 4, 4],
            'analytic_continuation': 'pade',
            'broadening_eV': 0.1,
            'frequency_window_eV': [0.0, 18.0],
            'real_frequency_points': 181,
            'imaginary_frequency_points': 100,
            'full_self_energy': True,
            'finite_size_correction': True,
        },
        'execution_policy': (
            'Opt in with {0}=1; the native 4x4x4 calculation is intentionally '
            'excluded from routine test runs.'
        ).format(RUN_ENVIRONMENT_VARIABLE),
    },
    'tolerances': {
        # This is a physical sanity range, not a frozen numerical reference.
        # A server-produced, versioned baseline should replace it only after the
        # native example and public backend have been run with identical builds.
        'quasiparticle_gap_eV_min': 0.5,
        'quasiparticle_gap_eV_max': 2.0,
    },
}


_SI_POSCAR = """Silicon primitive cell from fcDMFT examples/Si/si_gw.py
1.0
0.000000 2.715000 2.715000
2.715000 0.000000 2.715000
2.715000 2.715000 0.000000
Si
2
Direct
0.375000 0.375000 0.375000
0.625000 0.625000 0.625000
"""


_REQUIRED_ARTIFACT_KINDS = {
    'periodic_gw_result',
    'periodic_gw_arrays',
    'periodic_gw_analytic_continuation',
    'periodic_gw_mean_field_potential',
    'periodic_gw_imaginary_self_energy',
}


def _enabled(value: Any) -> bool:
    return str(value or '').strip().lower() in ('1', 'true', 'yes', 'on')


def _artifact_kinds(report: Mapping[str, Any]) -> Set[str]:
    return {
        str(item.get('kind') or '')
        for item in (report.get('artifacts') or [])
        if isinstance(item, Mapping) and item.get('kind')
    }


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _scientific_configuration(report: Mapping[str, Any]) -> Dict[str, Any]:
    task_spec = report.get('task_spec') or {}
    periodic = task_spec.get('periodic') or {}
    method = task_spec.get('method') or {}
    solver = task_spec.get('solver') or {}
    return {
        'task_type': task_spec.get('task_type'),
        'method': method.get('name') if isinstance(method, Mapping) else method,
        'xc': method.get('xc') if isinstance(method, Mapping) else task_spec.get('xc'),
        'restricted': (
            method.get('restricted')
            if isinstance(method, Mapping)
            else task_spec.get('restricted')
        ),
        'basis': periodic.get('basis'),
        'pseudo': periodic.get('pseudo'),
        'kmesh': list(periodic.get('kmesh') or []),
        'kpoint_scheme': periodic.get('kpoint_scheme'),
        'kpoint_shift': list(periodic.get('kpoint_shift') or []),
        'density_fitting_method': periodic.get('density_fitting_method'),
        'precision': periodic.get('precision'),
        'smearing_method': periodic.get('smearing_method'),
        'solver': solver.get('name') if isinstance(solver, Mapping) else solver,
    }


def evaluate_fcdmft_si_g0w0_report(
    report: Mapping[str, Any],
    definition: Mapping[str, Any] = FCDMFT_SI_G0W0_BENCHMARK,
) -> Dict[str, Any]:
    """Evaluate the public TaskReport produced by the expensive Si benchmark."""

    results = report.get('structured_results') or {}
    gw_result = results.get('gw_result') or {}
    gap_hartree = gw_result.get('quasiparticle_gap')
    gap_ev = float(gap_hartree) * HARTREE_TO_EV if _finite_number(gap_hartree) else None
    artifact_kinds = _artifact_kinds(report)
    missing_artifacts = sorted(_REQUIRED_ARTIFACT_KINDS - artifact_kinds)
    tolerances = definition['tolerances']
    configuration = {
        'reference_method': gw_result.get('reference_method'),
        'kpoint_count': gw_result.get('kpoint_count'),
        'analytic_continuation': gw_result.get('analytic_continuation'),
        'full_self_energy': gw_result.get('full_self_energy'),
        'finite_size_correction': gw_result.get('finite_size_correction'),
        'real_frequency_points': gw_result.get('real_frequency_points'),
        'imaginary_frequency_points': gw_result.get('imaginary_frequency_points'),
    }
    expected_configuration = {
        'reference_method': 'dft',
        'kpoint_count': 64,
        'analytic_continuation': 'pade',
        'full_self_energy': True,
        'finite_size_correction': True,
        'real_frequency_points': 181,
        'imaginary_frequency_points': 100,
    }
    scientific_configuration = _scientific_configuration(report)
    expected_scientific_configuration = {
        'task_type': 'periodic',
        'method': 'dft',
        'xc': 'pbe',
        'restricted': True,
        'basis': 'gth-dzvp',
        'pseudo': 'gth-pbe',
        'kmesh': [4, 4, 4],
        'kpoint_scheme': 'gamma_centered',
        'kpoint_shift': [0.0, 0.0, 0.0],
        'density_fitting_method': 'gdf',
        'precision': 1e-12,
        'smearing_method': 'none',
        'solver': 'gw',
    }
    metrics = {
        'execution_status': report.get('execution_status'),
        'reference_converged': bool(results.get('reference_converged')),
        'gw_status': gw_result.get('status'),
        'quasiparticle_homo_Ha': gw_result.get('quasiparticle_homo'),
        'quasiparticle_lumo_Ha': gw_result.get('quasiparticle_lumo'),
        'quasiparticle_gap_Ha': gap_hartree,
        'quasiparticle_gap_eV': gap_ev,
        'fermi_energy_Ha': gw_result.get('fermi_energy'),
        'configuration': configuration,
        'scientific_configuration': scientific_configuration,
        'energy_available': gw_result.get('energy_available'),
        'final_energy': results.get('final_energy'),
        'energy_kind': results.get('energy_kind'),
        'artifact_kinds': sorted(artifact_kinds),
        'missing_artifact_kinds': missing_artifacts,
    }
    checks = [
        {
            'name': 'public_backend_execution',
            'passed': (
                metrics['execution_status'] == 'succeeded'
                and metrics['reference_converged']
                and metrics['gw_status'] == 'completed'
            ),
            'value': {
                'execution_status': metrics['execution_status'],
                'reference_converged': metrics['reference_converged'],
                'gw_status': metrics['gw_status'],
            },
            'expected': 'succeeded, converged PBE reference, completed GW stage',
        },
        {
            'name': 'public_task_scientific_configuration',
            'passed': scientific_configuration == expected_scientific_configuration,
            'value': scientific_configuration,
            'expected': expected_scientific_configuration,
        },
        {
            'name': 'native_example_configuration',
            'passed': configuration == expected_configuration,
            'value': configuration,
            'expected': expected_configuration,
        },
        {
            'name': 'silicon_quasiparticle_gap_sanity',
            'passed': (
                gap_ev is not None
                and tolerances['quasiparticle_gap_eV_min'] <= gap_ev
                <= tolerances['quasiparticle_gap_eV_max']
            ),
            'value': gap_ev,
            'range_eV': [
                tolerances['quasiparticle_gap_eV_min'],
                tolerances['quasiparticle_gap_eV_max'],
            ],
            'classification': 'physical_sanity_range_not_frozen_reference',
        },
        {
            'name': 'gw_total_energy_semantics',
            'passed': (
                metrics['energy_available'] is False
                and metrics['final_energy'] is None
                and metrics['energy_kind'] == 'gw_total_energy_unavailable'
            ),
            'value': {
                'energy_available': metrics['energy_available'],
                'final_energy': metrics['final_energy'],
                'energy_kind': metrics['energy_kind'],
            },
            'expected': 'quasiparticle result without a fabricated GW total energy',
        },
        {
            'name': 'registered_gw_artifacts',
            'passed': not missing_artifacts,
            'value': sorted(artifact_kinds),
            'required': sorted(_REQUIRED_ARTIFACT_KINDS),
        },
    ]
    return {'metrics': metrics, 'checks': checks}


def build_fcdmft_si_g0w0_request() -> Dict[str, Any]:
    """Return the frozen public request matching the native fcDMFT example."""

    return {
        'task_type': 'periodic',
        'periodic': {
            'structure_format': 'poscar',
            'structure_text': _SI_POSCAR,
            'basis': 'gth-dzvp',
            'pseudo': 'gth-pbe',
            'kmesh': [4, 4, 4],
            'kpoint_scheme': 'gamma_centered',
            'kpoint_shift': [0, 0, 0],
            'dimension': 3,
            'precision': 1e-12,
            'density_fitting_method': 'gdf',
            'smearing_method': 'none',
            'band_path_mode': 'auto',
        },
        'method': 'dft',
        'xc': 'pbe',
        'restricted': True,
        'solver': {
            'name': 'gw',
            'options': {
                'gw_analytic_continuation': 'pade',
                'gw_broadening': 0.1 / HARTREE_TO_EV,
                'gw_frequency_window': [0.0, 18.0 / HARTREE_TO_EV],
                'gw_real_frequency_points': 181,
                'gw_imaginary_frequency_points': 100,
                'gw_full_self_energy': True,
                'gw_finite_size_correction': True,
                'gw_quasiparticle_energies': True,
            },
        },
        'job': 'single_point',
        'charge': 0,
        'spin': 0,
        'outputs': ['energy', 'fermi_energy'],
        'max_cycle': 50,
        'conv_tol': 1e-12,
        'verbose': 4,
    }


def run_fcdmft_si_g0w0(
    definition: Mapping[str, Any],
    work_dir: Optional[Path],
) -> Dict[str, Any]:
    if not _enabled(os.getenv(RUN_ENVIRONMENT_VARIABLE)):
        return {
            'metrics': {'opt_in_environment_variable': RUN_ENVIRONMENT_VARIABLE},
            'checks': [],
            'skip_reason': (
                'The native 4x4x4 Si G0W0 benchmark is expensive; set {0}=1 '
                'to run it explicitly.'
            ).format(RUN_ENVIRONMENT_VARIABLE),
        }

    from ..providers.fcdmft import fcdmft_availability

    availability = fcdmft_availability()
    if not availability.get('available'):
        return {
            'metrics': {'provider': availability},
            'checks': [],
            'skip_reason': availability.get('reason') or 'fcDMFT is unavailable',
        }

    from ..executors import LocalExecutor

    def execute(directory: Path) -> Dict[str, Any]:
        return LocalExecutor().execute_task(
            json.dumps(build_fcdmft_si_g0w0_request()),
            channel='benchmark',
            locale='en',
            work_dir=str(directory),
            run_id='fcdmft-si-g0w0',
        )

    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix='pyscf-agent-fcdmft-si-g0w0-') as directory:
            report = execute(Path(directory))
    else:
        report = execute(work_dir)
    return evaluate_fcdmft_si_g0w0_report(report, definition)


__all__ = [
    'FCDMFT_SI_G0W0_BENCHMARK',
    'HARTREE_TO_EV',
    'RUN_ENVIRONMENT_VARIABLE',
    'build_fcdmft_si_g0w0_request',
    'evaluate_fcdmft_si_g0w0_report',
    'run_fcdmft_si_g0w0',
]
