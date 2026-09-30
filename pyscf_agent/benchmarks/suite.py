from __future__ import annotations

import copy
import json
import math
import platform
import tempfile
from datetime import datetime, timezone
from importlib import metadata, resources
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .fixtures import block2_unavailable_outcome as _block2_unavailable_outcome

from ..schema_contracts import BENCHMARK_MANIFEST_SCHEMA, BENCHMARK_RESULT_SCHEMA
from .block2_workflows import (
    BLOCK2_WORKFLOW_BENCHMARK,
    run_block2_adaptive_workflows,
)
from .block2_scaling import (
    run_block2_hubbard_ring_scaling,
    run_block2_hydrogen_chain_scaling,
)
from .fcdmft_si_g0w0 import (
    FCDMFT_SI_G0W0_BENCHMARK,
    run_fcdmft_si_g0w0,
)

_N2_REFERENCE_BOND_LENGTH_ANGSTROM = 1.09768


_BENCHMARKS = (
    {
        'id': 'n2-dissociation-sto3g',
        'domain': 'molecular',
        'tier': 'reference',
        'description': 'N2 STO-3G adaptive dissociation curve against published classical references.',
        'runner': 'n2',
        'source': {
            'citation': (
                'T. Weaving et al., npj Quantum Information 11, 25 (2025)'
            ),
            'doi': '10.1038/s41534-024-00952-4',
            'data': 'https://github.com/TimWeaving/N2-CS-VQE',
        },
        'tolerances': {
            'ccsd_t_max_error_mHa': 0.25,
            'ccsd_max_error_mHa': 0.03,
            'expected_path_anomalies': 1,
        },
    },
    {
        'id': 'hubbard-dimer-ed',
        'domain': 'model_hamiltonian',
        'tier': 'smoke',
        'description': 'Half-filled two-site Hubbard exact diagonalization against the analytic solution.',
        'runner': 'hubbard',
        'source': {
            'reference': 'Analytic two-site Hubbard singlet ground state.',
            'parameters': {'U': [0.0, 2.0, 4.0, 8.0], 't': -1.0},
        },
        'tolerances': {
            'energy_abs_Ha': 1e-10,
            'double_occupancy_abs': 1e-10,
        },
    },
    {
        'id': 'periodic-he-gamma-rhf',
        'domain': 'periodic',
        'tier': 'smoke',
        'description': 'Periodic He Gamma-point RHF with GTH-SZV/GTH-Pade.',
        'runner': 'periodic',
        'source': {
            'reference': 'Frozen PySCF 2.13 periodic HF baseline.',
            'documentation': 'https://pyscf.org/user/pbc/scf.html',
        },
        'tolerances': {
            'energy_abs_Ha_per_cell': 1e-8,
            'fermi_energy_abs_Ha': 1e-8,
        },
    },
    FCDMFT_SI_G0W0_BENCHMARK,
    {
        'id': 'block2-h4-casci',
        'domain': 'molecular',
        'tier': 'optional_provider',
        'description': 'H4/STO-3G CASCI with block2 DMRG against the PySCF FCI active-space solver.',
        'runner': 'block2_h4',
        'source': {
            'reference': 'Same Hamiltonian and orbital space solved by PySCF direct FCI.',
            'provider_citation': 'Zhai et al., J. Chem. Phys. 159, 234801 (2023).',
            'doi': '10.1063/5.0180424',
        },
        'tolerances': {
            'energy_abs_Ha': 1e-8,
        },
    },
    {
        'id': 'block2-h2-casscf',
        'domain': 'molecular',
        'tier': 'optional_provider',
        'description': 'H2/6-31G DMRG-CASSCF orbital optimization against the PySCF FCI CASSCF solver.',
        'runner': 'block2_h2_casscf',
        'source': {
            'reference': 'The same CAS(2,2) orbital optimization performed with the PySCF direct FCI solver.',
            'provider_citation': 'Zhai et al., J. Chem. Phys. 159, 234801 (2023).',
            'doi': '10.1063/5.0180424',
        },
        'tolerances': {
            'energy_abs_Ha': 1e-8,
        },
    },
    {
        'id': 'block2-hubbard-ring',
        'domain': 'model_hamiltonian',
        'tier': 'optional_provider',
        'description': 'Four-site Hubbard ring block2 energy, low roots, observables, symmetry, entanglement, and MPS continuation.',
        'runner': 'block2_hubbard_ring',
        'source': {
            'reference': 'PySCF full diagonalization of the identical finite Hubbard Hamiltonian.',
            'provider_citation': 'Zhai et al., J. Chem. Phys. 159, 234801 (2023).',
            'doi': '10.1063/5.0180424',
        },
        'tolerances': {
            'energy_abs_Ha': 1e-8,
            'excitation_abs_Ha': 1e-7,
            'observable_abs': 1e-7,
            'restart_energy_abs_Ha': 1e-8,
        },
    },
    BLOCK2_WORKFLOW_BENCHMARK,
    {
        'id': 'block2-hydrogen-chain-scaling',
        'domain': 'molecular',
        'tier': 'scaling',
        'description': 'H6/H8/H10 STO-6G DMRG-CASCI accuracy and H20 bond-dimension convergence.',
        'runner': 'block2_hydrogen_chain_scaling',
        'source': {
            'reference': 'PySCF FCI on the identical H6/H8/H10 Hamiltonians; internal variational convergence for H20.',
            'provider_citation': 'Zhai et al., J. Chem. Phys. 159, 234801 (2023).',
            'doi': '10.1063/5.0180424',
        },
        'tolerances': {
            'small_chain_energy_abs_Ha': 1e-7,
            'h20_bond_dimension_delta_Ha': 5e-4,
        },
    },
    {
        'id': 'block2-hubbard-ring-scaling',
        'domain': 'model_hamiltonian',
        'tier': 'scaling',
        'description': 'Eight-site exact accuracy and 12/16-site DMRG convergence for half-filled Hubbard rings.',
        'runner': 'block2_hubbard_ring_scaling',
        'source': {
            'reference': 'Full diagonalization for eight sites; internal variational convergence for larger rings.',
            'provider_citation': 'Zhai et al., J. Chem. Phys. 159, 234801 (2023).',
            'doi': '10.1063/5.0180424',
        },
        'tolerances': {
            'eight_site_energy_abs_Ha': 1e-7,
            'medium_scale_max_discarded_weight': 5e-4,
        },
    },
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _distribution_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def benchmark_manifest() -> Dict[str, Any]:
    return {
        'schema': BENCHMARK_MANIFEST_SCHEMA,
        'benchmarks': [
            {key: copy.deepcopy(value) for key, value in item.items() if key != 'runner'}
            for item in _BENCHMARKS
        ],
    }


def list_benchmarks() -> List[Dict[str, Any]]:
    return benchmark_manifest()['benchmarks']


def load_n2_reference_rows() -> List[Dict[str, float]]:
    data = resources.files('pyscf_agent.benchmarks').joinpath(
        'data/n2_contextual_vqe_comparison.tsv'
    ).read_text(encoding='utf-8')
    lines = [
        line.strip()
        for line in data.splitlines()
        if line.strip() and not line.startswith('#')
    ]
    headers = lines[0].split()
    rows = []
    for line in lines[1:]:
        values = line.split()
        row = {
            header: float('nan') if value.lower() == 'nan' else float(value)
            for header, value in zip(headers, values)
        }
        row['bond_factor'] = row['bond_length_A'] / _N2_REFERENCE_BOND_LENGTH_ANGSTROM
        rows.append(row)
    return rows


def _n2_method(factor: float) -> str:
    if factor <= 1.0 + 1e-9:
        return 'ccsd_t'
    if factor <= 1.8 + 1e-9:
        return 'ccsd'
    return 'casscf'


def _run_n2(definition: Dict[str, Any], _work_dir: Optional[Path]) -> Dict[str, Any]:
    from computational_study_agent.adaptive.path_diagnostics import analyze_scan_path

    rows = load_n2_reference_rows()
    ccsdt_errors = [
        abs(row['agent_final_E_Ha'] - row['literature_CCSDT_E_Ha']) * 1000.0
        for row in rows
        if 0.8 - 1e-9 <= row['bond_factor'] <= 1.0 + 1e-9
        and math.isfinite(row['literature_CCSDT_E_Ha'])
    ]
    ccsd_errors = [
        abs(row['agent_final_E_Ha'] - row['literature_CCSD_E_Ha']) * 1000.0
        for row in rows
        if 1.1 - 1e-9 <= row['bond_factor'] <= 1.8 + 1e-9
        and math.isfinite(row['literature_CCSD_E_Ha'])
    ]
    cases = []
    comparison_table = []
    for index, row in enumerate(rows, start=1):
        case_id = 'n2-{0:04d}'.format(index)
        cases.append({'case_id': case_id, 'variables': {'bond_length_A': row['bond_length_A']}})
        comparison_table.append({
            'case_id': case_id,
            'label': 'R={0:.6f} A'.format(row['bond_length_A']),
            'status': 'succeeded',
            'bond_length_A': row['bond_length_A'],
            'final_energy': '{0:.12f} Ha'.format(row['agent_final_E_Ha']),
            'method': _n2_method(row['bond_factor']),
        })
    diagnostics = analyze_scan_path({
        'system_type': 'molecular',
        'cases': cases,
        'comparison_table': comparison_table,
    })
    metrics = {
        'row_count': len(rows),
        'ccsd_t_overlap_count': len(ccsdt_errors),
        'ccsd_overlap_count': len(ccsd_errors),
        'ccsd_t_max_error_mHa': max(ccsdt_errors),
        'ccsd_max_error_mHa': max(ccsd_errors),
        'path_anomaly_count': len(diagnostics.get('anomalies') or []),
        'path_review_status': diagnostics.get('status'),
    }
    tolerances = definition['tolerances']
    checks = [
        {
            'name': 'ccsd_t_overlap_accuracy',
            'passed': metrics['ccsd_t_max_error_mHa'] < tolerances['ccsd_t_max_error_mHa'],
            'value': metrics['ccsd_t_max_error_mHa'],
            'limit': tolerances['ccsd_t_max_error_mHa'],
        },
        {
            'name': 'ccsd_overlap_accuracy',
            'passed': metrics['ccsd_max_error_mHa'] < tolerances['ccsd_max_error_mHa'],
            'value': metrics['ccsd_max_error_mHa'],
            'limit': tolerances['ccsd_max_error_mHa'],
        },
        {
            'name': 'method_boundary_detection',
            'passed': metrics['path_anomaly_count'] == tolerances['expected_path_anomalies'],
            'value': metrics['path_anomaly_count'],
            'expected': tolerances['expected_path_anomalies'],
        },
    ]
    return {'metrics': metrics, 'checks': checks}


def _hubbard_spec(u_value: float, t_value: float) -> Dict[str, Any]:
    return {
        'schema': 'pyscf-agent.model-hamiltonian.v1',
        'model': 'hubbard',
        'dimension': 1,
        'preset': 'chain',
        'boundary': 'open',
        'energy_unit': 'a.u.',
        'nelec': [1, 1],
        'sites': [
            {'id': 0, 'x': 0, 'y': 0, 'epsilon': 0, 'U': u_value},
            {'id': 1, 'x': 1, 'y': 0, 'epsilon': 0, 'U': u_value},
        ],
        'bonds': [{
            'id': 0,
            'source': 0,
            'target': 1,
            't': t_value,
            'V': 0,
            'effective_t': t_value,
        }],
    }


def _run_hubbard(definition: Dict[str, Any], _work_dir: Optional[Path]) -> Dict[str, Any]:
    from pyscf_agent.backend.model_hamiltonian.solver import run_model_hamiltonian_solver

    t_value = float(definition['source']['parameters']['t'])
    rows = []
    for u_value in definition['source']['parameters']['U']:
        u_value = float(u_value)
        result = run_model_hamiltonian_solver(
            _hubbard_spec(u_value, t_value),
            solver_name='fci',
            outputs=['energy', 'double_occupancy'],
        )
        radical = math.sqrt(u_value * u_value + 16.0 * t_value * t_value)
        reference_energy = 0.5 * (u_value - radical)
        reference_double_occupancy = 0.25 * (1.0 - u_value / radical)
        rows.append({
            'U': u_value,
            't': t_value,
            'energy': result['energy'],
            'reference_energy': reference_energy,
            'energy_abs_error': abs(result['energy'] - reference_energy),
            'double_occupancy_mean': result['double_occupancy_mean'],
            'reference_double_occupancy_mean': reference_double_occupancy,
            'double_occupancy_abs_error': abs(
                result['double_occupancy_mean'] - reference_double_occupancy
            ),
        })
    metrics = {
        'point_count': len(rows),
        'max_energy_abs_error_Ha': max(row['energy_abs_error'] for row in rows),
        'max_double_occupancy_abs_error': max(
            row['double_occupancy_abs_error'] for row in rows
        ),
        'points': rows,
    }
    tolerances = definition['tolerances']
    checks = [
        {
            'name': 'analytic_ground_state_energy',
            'passed': metrics['max_energy_abs_error_Ha'] <= tolerances['energy_abs_Ha'],
            'value': metrics['max_energy_abs_error_Ha'],
            'limit': tolerances['energy_abs_Ha'],
        },
        {
            'name': 'analytic_double_occupancy',
            'passed': (
                metrics['max_double_occupancy_abs_error']
                <= tolerances['double_occupancy_abs']
            ),
            'value': metrics['max_double_occupancy_abs_error'],
            'limit': tolerances['double_occupancy_abs'],
        },
    ]
    return {'metrics': metrics, 'checks': checks}


_HE_POSCAR = """Helium primitive cell
1.0
5.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
He
1
Direct
0.0 0.0 0.0
"""


def _run_periodic(definition: Dict[str, Any], work_dir: Optional[Path]) -> Dict[str, Any]:
    from pyscf_agent.executors import LocalExecutor

    request = {
        'task_type': 'periodic',
        'periodic': {
            'structure_format': 'poscar',
            'structure_text': _HE_POSCAR,
            'basis': 'gth-szv',
            'pseudo': 'gth-pade',
            'kmesh': [1, 1, 1],
            'dimension': 3,
            'band_path_mode': 'auto',
        },
        'method': 'hf',
        'job': 'single_point',
        'charge': 0,
        'spin': 0,
        'outputs': ['energy', 'fermi_energy'],
        'max_cycle': 30,
        'verbose': 0,
    }

    def execute(directory: Path) -> Dict[str, Any]:
        return LocalExecutor().execute_task(
            json.dumps(request),
            channel='benchmark',
            locale='en',
            work_dir=str(directory),
            run_id='periodic-he-gamma-rhf',
        )

    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix='pyscf-agent-benchmark-') as directory:
            report = execute(Path(directory))
    else:
        report = execute(work_dir)
    results = report.get('structured_results') or {}
    reference_energy = -2.864744451428037
    reference_fermi = -0.9309928077853226
    energy = results.get('final_energy')
    fermi = results.get('fermi_energy')
    metrics = {
        'execution_status': report.get('execution_status'),
        'converged': bool(results.get('converged')),
        'energy_Ha_per_cell': energy,
        'reference_energy_Ha_per_cell': reference_energy,
        'energy_abs_error_Ha_per_cell': (
            abs(float(energy) - reference_energy) if energy is not None else None
        ),
        'fermi_energy_Ha': fermi,
        'reference_fermi_energy_Ha': reference_fermi,
        'fermi_energy_abs_error_Ha': (
            abs(float(fermi) - reference_fermi) if fermi is not None else None
        ),
        'artifact_count': len(report.get('artifacts') or []),
    }
    tolerances = definition['tolerances']
    checks = [
        {
            'name': 'periodic_execution',
            'passed': metrics['execution_status'] == 'succeeded' and metrics['converged'],
            'value': metrics['execution_status'],
            'expected': 'succeeded and converged',
        },
        {
            'name': 'periodic_total_energy',
            'passed': (
                metrics['energy_abs_error_Ha_per_cell'] is not None
                and metrics['energy_abs_error_Ha_per_cell']
                <= tolerances['energy_abs_Ha_per_cell']
            ),
            'value': metrics['energy_abs_error_Ha_per_cell'],
            'limit': tolerances['energy_abs_Ha_per_cell'],
        },
        {
            'name': 'periodic_fermi_energy',
            'passed': (
                metrics['fermi_energy_abs_error_Ha'] is not None
                and metrics['fermi_energy_abs_error_Ha']
                <= tolerances['fermi_energy_abs_Ha']
            ),
            'value': metrics['fermi_energy_abs_error_Ha'],
            'limit': tolerances['fermi_energy_abs_Ha'],
        },
    ]
    return {'metrics': metrics, 'checks': checks}


def _block2_options(**updates: Any) -> Dict[str, Any]:
    options = {
        'preset': 'screening',
        'bond_dimensions': [32, 64, 64, 64, 64, 64, 64, 64],
        'noises': [1e-5, 1e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'davidson_thresholds': [1e-10] * 8,
        'sweeps': 8,
        'energy_tolerance': 1e-9,
        'discarded_weight_tolerance': 1e-8,
        'save_mps': True,
        'symmetry': 'su2',
    }
    options.update(updates)
    return options


def _execute_molecular_dmrg(request: Dict[str, Any], directory: Path) -> Dict[str, Any]:
    from ..backend.execution import _run_pyscf_task
    from ..contracts import task_spec_from_dict

    dmrg_request = copy.deepcopy(request)
    dmrg_request['solver'] = {'name': 'block2_dmrg', 'options': _block2_options()}
    return _run_pyscf_task(task_spec_from_dict(dmrg_request), execution_directory=directory)


def _run_block2_h4(definition: Dict[str, Any], work_dir: Optional[Path]) -> Dict[str, Any]:
    unavailable = _block2_unavailable_outcome()
    if unavailable:
        return unavailable
    from ..backend.execution import _run_pyscf_task
    from ..contracts import task_spec_from_dict

    request = {
        'task_type': 'molecular',
        'system': {
            'atom': 'H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2',
            'basis': 'sto-3g',
            'unit': 'Angstrom',
            'charge': 0,
            'spin': 0,
        },
        'method': {'name': 'casci', 'restricted': True},
        'active_space': {
            'enabled': True,
            'ncas': 4,
            'nelecas': 4,
            'orbital_indices': [0, 1, 2, 3],
            'approved': True,
        },
        'analysis': {'outputs': ['energy']},
    }
    fci_request = copy.deepcopy(request)
    fci_request['solver'] = {'name': 'fci', 'options': {}}
    fci_result = _run_pyscf_task(task_spec_from_dict(fci_request))


    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix='pyscf-agent-block2-h4-') as directory:
            dmrg_result = _execute_molecular_dmrg(request, Path(directory))
    else:
        dmrg_result = _execute_molecular_dmrg(request, work_dir)
    error = abs(float(dmrg_result['energy']) - float(fci_result['energy']))
    limit = definition['tolerances']['energy_abs_Ha']
    return {
        'metrics': {
            'geometry': 'linear H4, R=1.4 Angstrom',
            'basis': 'sto-3g',
            'active_space': {'ncas': 4, 'nelecas': 4},
            'fci_energy_Ha': fci_result['energy'],
            'block2_energy_Ha': dmrg_result['energy'],
            'energy_abs_error_Ha': error,
            'block2_converged': bool(dmrg_result.get('converged')),
        },
        'checks': [{
            'name': 'h4_casci_energy',
            'passed': bool(dmrg_result.get('converged')) and error <= limit,
            'value': error,
            'limit': limit,
        }],
    }


def _run_block2_h2_casscf(definition: Dict[str, Any], work_dir: Optional[Path]) -> Dict[str, Any]:
    unavailable = _block2_unavailable_outcome()
    if unavailable:
        return unavailable
    from ..backend.execution import _run_pyscf_task
    from ..contracts import task_spec_from_dict

    request = {
        'task_type': 'molecular',
        'system': {
            'atom': 'H 0 0 0; H 0 0 1.5',
            'basis': '6-31g',
            'unit': 'Angstrom',
            'charge': 0,
            'spin': 0,
        },
        'method': {'name': 'casscf', 'restricted': True},
        'active_space': {
            'enabled': True,
            'ncas': 2,
            'nelecas': 2,
            'orbital_indices': [0, 1],
            'approved': True,
        },
        'analysis': {'outputs': ['energy']},
    }
    fci_request = copy.deepcopy(request)
    fci_request['solver'] = {'name': 'fci', 'options': {}}
    fci_result = _run_pyscf_task(task_spec_from_dict(fci_request))


    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix='pyscf-agent-block2-h2-casscf-') as directory:
            dmrg_result = _execute_molecular_dmrg(request, Path(directory))
    else:
        dmrg_result = _execute_molecular_dmrg(request, work_dir)
    casscf = ((dmrg_result.get('cas_result') or {}).get('dmrg_result') or {}).get('casscf') or {}
    error = abs(float(dmrg_result['energy']) - float(fci_result['energy']))
    limit = definition['tolerances']['energy_abs_Ha']
    return {
        'metrics': {
            'geometry': 'H2, R=1.5 Angstrom',
            'basis': '6-31g',
            'active_space': {'ncas': 2, 'nelecas': 2},
            'fci_casscf_energy_Ha': fci_result['energy'],
            'block2_casscf_energy_Ha': dmrg_result['energy'],
            'energy_abs_error_Ha': error,
            'orbital_converged': bool(casscf.get('orbital_converged')),
            'dmrg_converged': bool(casscf.get('dmrg_converged')),
            'macro_iterations': casscf.get('macro_iterations'),
            'active_space_solver_calls': casscf.get('active_space_solver_calls'),
        },
        'checks': [{
            'name': 'h2_dmrg_casscf_energy',
            'passed': (
                bool(dmrg_result.get('converged'))
                and bool(casscf.get('orbital_converged'))
                and bool(casscf.get('dmrg_converged'))
                and error <= limit
            ),
            'value': error,
            'limit': limit,
        }],
    }


def _hubbard_ring_spec(u_value: float) -> Dict[str, Any]:
    return {
        'schema': 'pyscf-agent.model-hamiltonian.v1',
        'model': 'hubbard',
        'dimension': 1,
        'preset': 'ring',
        'boundary': 'periodic',
        'energy_unit': 'a.u.',
        'nelec': [2, 2],
        'sites': [
            {'id': index, 'x': index, 'y': 0, 'epsilon': 0, 'U': u_value}
            for index in range(4)
        ],
        'bonds': [
            {'id': index, 'source': index, 'target': (index + 1) % 4, 't': -1, 'V': 0, 'effective_t': -1}
            for index in range(4)
        ],
    }


def _max_array_error(left: Any, right: Any) -> float:
    import numpy as np

    return float(np.max(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))))


def _run_block2_hubbard_ring(definition: Dict[str, Any], work_dir: Optional[Path]) -> Dict[str, Any]:
    unavailable = _block2_unavailable_outcome()
    if unavailable:
        return unavailable
    from ..backend.model_hamiltonian.solver import run_model_hamiltonian_solver
    from computational_study_agent.adaptive.state_tracking import analyze_dmrg_state_tracking

    outputs = [
        'energy',
        'gap',
        'double_occupancy',
        'spin_correlation',
        'charge_correlation',
    ]
    fci = run_model_hamiltonian_solver(
        _hubbard_ring_spec(4.0),
        solver_name='fci',
        outputs=outputs,
    )

    def execute_root(root: Path) -> Dict[str, Any]:
        root.mkdir(parents=True, exist_ok=True)
        dmrg = run_model_hamiltonian_solver(
            _hubbard_ring_spec(4.0),
            solver_name='block2_dmrg',
            outputs=outputs + ['entanglement_diagnostics', 'excited_states', 'symmetry_analysis'],
            solver_options=_block2_options(
                nroots=2,
                compute_entanglement=True,
                compute_mutual_information=True,
                compute_bipartite_entanglement=True,
                compute_symmetry_analysis=True,
            ),
            scratch_directory=str(root / 'u4-source'),
        )
        cold = run_model_hamiltonian_solver(
            _hubbard_ring_spec(6.0),
            solver_name='block2_dmrg',
            outputs=['energy', 'excited_states'],
            solver_options=_block2_options(nroots=2),
            scratch_directory=str(root / 'u6-cold'),
        )
        warm = run_model_hamiltonian_solver(
            _hubbard_ring_spec(6.0),
            solver_name='block2_dmrg',
            outputs=['energy', 'excited_states'],
            solver_options=_block2_options(
                nroots=2,
                restart_manifest=dmrg['dmrg_result']['checkpoint_manifest'],
                restart_provenance={'source_case_id': 'U=4'},
            ),
            scratch_directory=str(root / 'u6-warm'),
        )
        return {'dmrg': dmrg, 'cold': cold, 'warm': warm}

    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix='pyscf-agent-block2-ring-') as directory:
            results = execute_root(Path(directory))
    else:
        results = execute_root(work_dir)
    dmrg = results['dmrg']
    cold = results['cold']
    warm = results['warm']
    energy_error = abs(float(dmrg['energy']) - float(fci['energy']))
    excitation_error = abs(float(dmrg['excitation_energies'][1]) - float(fci['gap']))
    observable_errors = {
        field: _max_array_error(dmrg[field]['values'], fci[field]['values'])
        for field in ('double_occupancy', 'spin_correlation', 'charge_correlation')
    }
    restart_error = abs(float(warm['energy']) - float(cold['energy']))
    tolerances = definition['tolerances']
    entanglement = dmrg.get('entanglement_diagnostics') or {}
    symmetry = dmrg.get('symmetry_analysis') or {}
    state_tracking = analyze_dmrg_state_tracking({
        'system_type': 'model_hamiltonian',
        'cases': [
            {
                'case_id': 'U=4',
                'variables': {'U': 4.0},
                'request': {'solver': {'name': 'block2_dmrg'}},
                'task_report': {
                    'execution_status': 'succeeded',
                    'structured_results': {'dmrg_result': dmrg['dmrg_result']},
                },
            },
            {
                'case_id': 'U=6',
                'variables': {'U': 6.0},
                'request': {'solver': {'name': 'block2_dmrg'}},
                'task_report': {
                    'execution_status': 'succeeded',
                    'structured_results': {'dmrg_result': cold['dmrg_result']},
                },
            },
        ],
    })
    contract = dmrg.get('dmrg_result') or {}
    checks = [
        {'name': 'ring_ground_state_energy', 'passed': energy_error <= tolerances['energy_abs_Ha'], 'value': energy_error, 'limit': tolerances['energy_abs_Ha']},
        {'name': 'ring_first_excitation', 'passed': excitation_error <= tolerances['excitation_abs_Ha'], 'value': excitation_error, 'limit': tolerances['excitation_abs_Ha']},
        {'name': 'ring_rdm_observables', 'passed': max(observable_errors.values()) <= tolerances['observable_abs'], 'value': max(observable_errors.values()), 'limit': tolerances['observable_abs']},
        {'name': 'ring_entanglement_and_symmetry', 'passed': bool(entanglement.get('single_orbital_entropy')) and symmetry.get('particle_number') == 4, 'value': {'entropy_count': len(entanglement.get('single_orbital_entropy') or []), 'particle_number': symmetry.get('particle_number')}, 'expected': 'four entropies and four particles'},
        {'name': 'changed_hamiltonian_mps_restart', 'passed': bool(warm['dmrg_result']['restart'].get('applied')) and restart_error <= tolerances['restart_energy_abs_Ha'], 'value': restart_error, 'limit': tolerances['restart_energy_abs_Ha']},
        {
            'name': 'common_result_contract_and_state_tracking',
            'passed': bool(
                contract.get('result_contract_schema') == 'pyscf-agent.block2-result-contract.v1'
                and contract.get('computed_root_count') == 2
                and len(contract.get('root_signatures') or []) == 2
                and state_tracking.get('links')
                and state_tracking['links'][0].get('mapping')
            ),
            'value': {
                'contract_schema': contract.get('result_contract_schema'),
                'computed_root_count': contract.get('computed_root_count'),
                'tracking_status': state_tracking.get('status'),
            },
            'expected': 'two targeted roots represented by the common contract and linked across U=4 to U=6',
        },
    ]
    return {
        'metrics': {
            'U': 4.0,
            't': -1.0,
            'site_count': 4,
            'fci_energy': fci['energy'],
            'block2_energy': dmrg['energy'],
            'energy_abs_error_Ha': energy_error,
            'first_excitation_abs_error_Ha': excitation_error,
            'observable_max_abs_errors': observable_errors,
            'max_single_orbital_entropy': entanglement.get('max_single_orbital_entropy'),
            'max_mutual_information': entanglement.get('max_mutual_information'),
            'spin_square': symmetry.get('spin_square'),
            'restart_source_case': warm['dmrg_result']['restart'].get('source_case_id'),
            'restart_same_hamiltonian': warm['dmrg_result']['restart'].get('same_hamiltonian'),
            'restart_energy_abs_error_Ha': restart_error,
            'result_contract_schema': contract.get('result_contract_schema'),
            'dmrg_state_tracking': state_tracking,
        },
        'checks': checks,
    }


_RUNNERS: Dict[str, Callable[[Dict[str, Any], Optional[Path]], Dict[str, Any]]] = {
    'n2': _run_n2,
    'hubbard': _run_hubbard,
    'periodic': _run_periodic,
    'fcdmft_si_g0w0': run_fcdmft_si_g0w0,
    'block2_h4': _run_block2_h4,
    'block2_h2_casscf': _run_block2_h2_casscf,
    'block2_hubbard_ring': _run_block2_hubbard_ring,
    'block2_adaptive_workflows': run_block2_adaptive_workflows,
    'block2_hydrogen_chain_scaling': run_block2_hydrogen_chain_scaling,
    'block2_hubbard_ring_scaling': run_block2_hubbard_ring_scaling,
}


def run_benchmark(
    benchmark_id: str,
    *,
    work_dir: Optional[Any] = None,
) -> Dict[str, Any]:
    definition = next(
        (item for item in _BENCHMARKS if item['id'] == benchmark_id),
        None,
    )
    if definition is None:
        raise ValueError('Unknown benchmark: {0}'.format(benchmark_id))
    resolved_work_dir = (
        Path(work_dir).expanduser().resolve() / benchmark_id
        if work_dir is not None
        else None
    )
    if resolved_work_dir is not None:
        resolved_work_dir.mkdir(parents=True, exist_ok=True)
    started_at = _timestamp()
    try:
        if resolved_work_dir is None:
            with tempfile.TemporaryDirectory(
                prefix='pyscf-agent-benchmark-{0}-'.format(benchmark_id),
            ) as directory:
                outcome = _RUNNERS[definition['runner']](definition, Path(directory))
        else:
            outcome = _RUNNERS[definition['runner']](definition, resolved_work_dir)
    except Exception as exc:
        outcome = {
            'metrics': {},
            'checks': [],
            'error': '{0}: {1}'.format(type(exc).__name__, exc),
        }
    skipped = bool(outcome.get('skip_reason'))
    passed = not skipped and bool(outcome.get('checks')) and all(
        bool(item.get('passed')) for item in outcome['checks']
    ) and not outcome.get('error')
    return {
        'id': definition['id'],
        'domain': definition['domain'],
        'tier': definition['tier'],
        'status': 'skipped' if skipped else ('passed' if passed else 'failed'),
        'started_at': started_at,
        'completed_at': _timestamp(),
        'source': copy.deepcopy(definition['source']),
        'tolerances': copy.deepcopy(definition['tolerances']),
        **outcome,
    }


def run_benchmark_suite(
    benchmark_ids: Optional[Iterable[str]] = None,
    *,
    work_dir: Optional[Any] = None,
    include_scaling: bool = False,
) -> Dict[str, Any]:
    selected = list(benchmark_ids or [
        item['id']
        for item in _BENCHMARKS
        if include_scaling or item.get('tier') != 'scaling'
    ])
    if not selected:
        raise ValueError('At least one benchmark is required')
    results = [run_benchmark(item, work_dir=work_dir) for item in selected]
    return {
        'schema': BENCHMARK_RESULT_SCHEMA,
        'status': 'passed' if all(item['status'] in ('passed', 'skipped') for item in results) else 'failed',
        'created_at': _timestamp(),
        'environment': {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'pyscf': _distribution_version('pyscf'),
            'pyscf_agent': _distribution_version('pyscf-agent') or 'source-tree',
        },
        'benchmark_count': len(results),
        'passed_count': sum(item['status'] == 'passed' for item in results),
        'skipped_count': sum(item['status'] == 'skipped' for item in results),
        'results': results,
    }


def write_benchmark_report(path: Path, report: Dict[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    return target


__all__ = [
    'BENCHMARK_MANIFEST_SCHEMA',
    'BENCHMARK_RESULT_SCHEMA',
    'benchmark_manifest',
    'list_benchmarks',
    'load_n2_reference_rows',
    'run_benchmark',
    'run_benchmark_suite',
    'write_benchmark_report',
]
