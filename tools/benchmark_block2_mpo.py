"""Opt-in fixed-orbital comparison with the official dmrgscf/block2main interface.

Run with PYTHONPATH=. python tools/benchmark_block2_mpo.py --work-dir runs/mpo-check.
Requires PySCF, block2 and pyscf-dmrgscf in the same Python environment. Each
measurement uses a fresh process and scratch directory; no CASSCF is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
from pathlib import Path
import resource
import statistics
import subprocess
import sys
from time import perf_counter


ARMS = ('fast_bipartite', 'conventional', 'native')
AVAILABLE_ARMS = ARMS + ('agent', 'example')
SCHEDULE = {
    'bond_dimensions': [64] * 12,
    'noises': [1e-5] * 2 + [1e-6] * 2 + [0.0] * 8,
    'davidson_thresholds': [1e-13] * 12,
    'sweeps': 12,
    'energy_tolerance': 1e-11,
    'cutoff': 1e-14,
    'integral_cutoff': 1e-12,
}


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def _peak_rss_mib(who):
    # macOS reports bytes; Linux reports KiB. Values are process peaks, not
    # simultaneous process-tree usage or block2 stack allocation.
    return resource.getrusage(who).ru_maxrss / (1024 ** 2 if sys.platform == 'darwin' else 1024)


def _native_sweep_details(output, energy_tolerance):
    # The first sweep omits DE; subsequent sweeps include it between E and DW.
    # Match the energy summary and its own timing, excluding NPDM sweeps.
    number = r'[-+\d.eE]+'
    rows = re.findall(
        rf'Time elapsed\s*=.*?\| E =\s*({number})\s*'
        rf'(?:\| DE =\s*({number})\s*)?\| DW =\s*({number})\s*\n'
        rf'Time sweep\s*=\s*({number})', output,
    )
    if not rows:
        raise ValueError('Native output contains no DMRG sweep summaries.')
    observations = [
        {'energy_hartree': float(e), 'energy_change_hartree': float(de) if de else None,
         'discarded_weight': float(dw)} for e, de, dw, _ in rows
    ]
    # Native summaries print DE with limited precision. Retain that observation
    # rather than presenting a higher-precision convergence estimate.
    final_change = observations[-1]['energy_change_hartree']
    sites = re.findall(rf'Site\s*=.*?\bE\s*=\s*({number})', output)
    return {
        'sweep_observations': observations,
        'sweep_seconds': [float(seconds) for _, _, _, seconds in rows],
        'sweeps_completed': len(rows),
        'final_site_energy_hartree': float(sites[-1]) if sites else None,
        'energy_converged': abs(final_change) < energy_tolerance if final_change is not None else None,
        'energy_convergence_basis': 'rounded_native_log_energy_change',
    }


def _assess_rdms(h1e, g2e, ecore, nelec, energy, rdm1, rdm2, reference=None):
    import numpy as np

    nelectron = int(sum(nelec))
    reconstructed = float(ecore + np.einsum('ij,ji', h1e, rdm1)
                          + 0.5 * np.einsum('ijkl,ijkl', g2e, rdm2))
    # At finite M, block2's lowest local energy in the final two-site sweep
    # need not equal <H> of the final stored MPS. Keep the gap as a diagnostic;
    # strict equality is appropriate only for the FCI-limit validation below.
    energies = {
        'rdm_energy_hartree': reconstructed,
        'rdm_minus_sweep_energy_hartree': reconstructed - float(energy),
        'reported_energy_semantics': 'lowest_local_energy_in_last_sweep',
    }
    errors = {
        'particle_number': abs(float(np.trace(rdm1)) - nelectron),
        'rdm1_hermiticity': float(np.max(np.abs(rdm1 - rdm1.T.conj()))),
        'rdm2_particle_pairs': abs(float(np.einsum('iijj', rdm2)) - nelectron * (nelectron - 1)),
        'rdm2_to_rdm1_contraction': float(np.max(np.abs(
            np.einsum('ikjj->ki', rdm2) - (nelectron - 1) * rdm1))),
    }
    checks = {name: error < 1e-8 for name, error in errors.items()}
    checks['finite_results'] = bool(np.isfinite(energy) and np.isfinite(reconstructed)
                                   and np.all(np.isfinite(rdm1)) and np.all(np.isfinite(rdm2)))
    if reference is not None:
        errors.update(
            rdm_energy_reconstruction_hartree=abs(reconstructed - float(energy)),
            energy_vs_fci_hartree=abs(float(energy) - float(reference['energy'])),
            rdm1_max_abs_vs_fci=float(np.max(np.abs(rdm1 - reference['rdm1']))),
            rdm2_max_abs_vs_fci=float(np.max(np.abs(rdm2 - reference['rdm2']))),
        )
        checks.update(rdm_energy=errors['rdm_energy_reconstruction_hartree'] < 1e-8,
                      energy=errors['energy_vs_fci_hartree'] < 1e-8,
                      rdm1=errors['rdm1_max_abs_vs_fci'] < 1e-6,
                      rdm2=errors['rdm2_max_abs_vs_fci'] < 1e-6)
    return energies, errors, checks


def _compare_measurements(root, measurements, arms):
    import numpy as np

    comparisons = []
    for repeat in sorted({item['repeat'] for item in measurements}):
        selected = {item['arm']: item for item in measurements if item['repeat'] == repeat}
        first = selected[arms[0]]
        with np.load(root / first['directory'] / 'rdms.npz') as a:
            for arm in arms[1:]:
                other = selected[arm]
                with np.load(root / other['directory'] / 'rdms.npz') as b:
                    comparisons.append({
                        'repeat': repeat, 'arms': [arms[0], arm],
                        'energy_difference_hartree': abs(first['energy_hartree'] - other['energy_hartree']),
                        'rdm_energy_difference_hartree': abs(first['energy_diagnostics']['rdm_energy_hartree']
                                                             - other['energy_diagnostics']['rdm_energy_hartree']),
                        'rdm1_max_abs_difference': float(np.max(np.abs(a['rdm1'] - b['rdm1']))),
                        'rdm2_max_abs_difference': float(np.max(np.abs(a['rdm2'] - b['rdm2']))),
                    })
    return comparisons


def _prepare(root, supplied=None):
    import numpy as np
    from pyscf import ao2mo, fci, gto, scf

    if supplied is not None:
        shutil.copyfile(supplied, root / 'hamiltonian.npz')
        with np.load(supplied) as data:
            norb, nelec = len(data['h1e']), data['nelec'].tolist()
        return {
            'system': 'Supplied fixed Hamiltonian: {0} electrons, {1} orbitals'.format(sum(nelec), norb),
            'orbitals': 'Saved integral basis, fixed; no reordering or localization',
            'input_file': str(supplied.resolve()),
            'hamiltonian_sha256': hashlib.sha256((root / 'hamiltonian.npz').read_bytes()).hexdigest(),
            'fci_reference': 'not_requested_for_supplied_hamiltonian',
        }
    mol = gto.M(atom=[('H', (0, 0, i * 1.4)) for i in range(6)], basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    if not mf.converged:
        raise RuntimeError('Benchmark reference RHF did not converge.')
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    g2e = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), 6)
    energy, ci = fci.direct_spin0.kernel(h1e, g2e, 6, mol.nelec, ecore=mol.energy_nuc(), tol=1e-13)
    dm1, dm2 = fci.direct_spin0.make_rdm12(ci, 6, mol.nelec)
    np.savez(root / 'hamiltonian.npz', h1e=h1e, g2e=g2e, ecore=mol.energy_nuc(), nelec=mol.nelec)
    np.savez(root / 'fci-reference.npz', energy=energy, rdm1=dm1, rdm2=dm2)
    return {
        'system': 'Linear H6, 1.4 Angstrom spacing, STO-3G, CAS(6e,6o)',
        'orbitals': 'One shared canonical RHF basis; fixed, no reordering or localization',
        'fci_energy_hartree': float(energy),
        'hamiltonian_sha256': hashlib.sha256((root / 'hamiltonian.npz').read_bytes()).hexdigest(),
    }


def _worker(root, arm, directory):
    import numpy as np

    settings = json.loads((root / 'settings.json').read_text())
    schedule = settings['schedule']
    threads, memory_bytes = settings['threads'], settings['stack_memory_bytes']
    data = np.load(root / 'hamiltonian.npz')
    h1e, g2e = data['h1e'], data['g2e']
    ecore, nelec, norb = float(data['ecore']), tuple(int(x) for x in data['nelec']), len(h1e)
    if arm == 'native':
        # Configure the official plugin before import. This arm never imports
        # pyscf_agent; the only shared input is the saved Hamiltonian.
        from pyscf import __config__
        executable = Path(sys.executable).parent / 'block2main'
        if not executable.is_file():
            raise FileNotFoundError('Expected block2main beside this Python: ' + str(executable))
        __config__.dmrgscf_BLOCKEXE = str(executable)
        __config__.dmrgscf_BLOCKSCRATCHDIR = str(directory / 'scratch')
        __config__.dmrgscf_BLOCKRUNTIMEDIR = str(directory)
        __config__.dmrgscf_MPIPREFIX = ''
        from pyscf.dmrgscf import dmrgci
        solver = dmrgci.DMRGCI(maxM=max(schedule['bond_dimensions']), tol=schedule['energy_tolerance'],
                              num_thrds=threads, memory=memory_bytes // 1_000_000_000)
        solver.executable = str(executable)
        solver.scratchDirectory = str(directory / 'scratch')
        solver.runtimeDir = str(directory)
        solver.mpiprefix = ''
        solver.verbose = 0
        solver.maxIter = schedule['sweeps']
        rows = []
        for i in range(schedule['sweeps']):
            values = tuple(schedule[key][min(i, len(schedule[key]) - 1)] for key in
                           ('bond_dimensions', 'davidson_thresholds', 'noises'))
            if not rows or values != rows[-1][1:]:
                rows.append((i,) + values)
        solver.scheduleSweeps, solver.scheduleMaxMs, solver.scheduleTols, solver.scheduleNoises = map(list, zip(*rows))
        solver.twodot_to_onedot = 0
        solver.block_extra_keyword = ['noreorder', 'twodot', 'mkl_thrds 1',
                                      'cutoff ' + str(schedule['cutoff']),
                                      'integral_tol ' + str(schedule['integral_cutoff'])]
        if threads == 1:
            solver.block_extra_keyword.append('num_thrds 1')
        started = perf_counter()
        energy, state = solver.kernel(h1e, g2e, norb, nelec, ecore=ecore)
        kernel_seconds = perf_counter() - started
        rdm1, rdm2 = solver.make_rdm12(state, norb, nelec)
        timings = {'kernel_including_rdm_seconds': kernel_seconds, 'solve_and_rdm_seconds': perf_counter() - started}
        details = {'interface': 'pyscf.dmrgscf.DMRGCI + block2main', 'agent_imported': 'pyscf_agent' in sys.modules}
        output = (directory / solver.outputFile).read_text()
        details.update(_native_sweep_details(output, schedule['energy_tolerance']))
    elif arm == 'example':
        # Independent reference following upstream 00-HC/00-dmrg.py. The
        # supplied integral order, resources and schedule are shared with the
        # agent; this arm imports no agent implementation or normalization.
        from pyblock2.driver.core import DMRGDriver, MPOAlgorithmTypes, SymmetryTypes
        started = perf_counter()
        driver = DMRGDriver(scratch=str(directory / 'scratch'), symm_type=SymmetryTypes.SU2,
                            stack_mem=memory_bytes, n_threads=threads, n_mkl_threads=1)
        try:
            driver.bw.b.Random.rand_seed(1234)
            driver.initialize_system(n_sites=norb, n_elec=sum(nelec), spin=0, orb_sym=[0] * norb)
            stage = perf_counter()
            mpo = driver.get_qc_mpo(
                h1e=h1e.copy(), g2e=g2e.copy(), ecore=ecore,
                algo_type=MPOAlgorithmTypes.ConventionalNC if norb == 2 else MPOAlgorithmTypes.Conventional,
                integral_cutoff=schedule['integral_cutoff'], iprint=1,
            )
            timings = {'mpo_construction_seconds': perf_counter() - stage}
            ket = driver.get_random_mps(tag='GS', bond_dim=schedule['bond_dimensions'][0], nroots=1)
            stage = perf_counter()
            energy = driver.dmrg(
                mpo, ket, n_sweeps=schedule['sweeps'], bond_dims=schedule['bond_dimensions'],
                noises=schedule['noises'], thrds=schedule['davidson_thresholds'],
                tol=schedule['energy_tolerance'], cutoff=schedule['cutoff'], iprint=1,
            )
            timings['dmrg_sweeps_seconds'] = perf_counter() - stage
            sweep_energies = np.asarray(driver._dmrg.energies, dtype=float).reshape(-1)
            discarded = np.asarray(driver._dmrg.discarded_weights, dtype=float).reshape(-1)
            delta = abs(float(sweep_energies[-1] - sweep_energies[-2])) if len(sweep_energies) > 1 else None
            details = {
                'interface': 'Independent pyblock2 driver following upstream 00-HC example',
                'agent_imported': 'pyscf_agent' in sys.modules,
                'sweeps_completed': len(sweep_energies),
                'sweep_energies_hartree': sweep_energies.tolist(),
                'discarded_weights': discarded.tolist(),
                'final_energy_change_hartree': delta,
                'energy_converged': delta < schedule['energy_tolerance'] if delta is not None else None,
            }
            stage = perf_counter()
            rdm1 = np.asarray(driver.get_1pdm(ket))
            rdm2 = np.asarray(driver.get_2pdm(ket)).transpose(0, 3, 1, 2)
            timings['rdm_seconds'] = perf_counter() - stage
            timings['solve_and_rdm_seconds'] = perf_counter() - started
        finally:
            driver.finalize()
    else:
        from pyscf_agent.providers.block2.molecular import build_active_space_hamiltonian_from_integrals
        from pyscf_agent.providers.block2.driver import run_block2_dmrg
        hamiltonian = build_active_space_hamiltonian_from_integrals(
            h1e, g2e, ncas=norb, nelecas=nelec, spin=0, core_energy=ecore,
        )
        options = {
            **schedule, 'symmetry': 'su2', 'orbital_ordering': 'canonical',
            'n_threads': threads, 'n_mkl_threads': 1, 'stack_memory_bytes': memory_bytes,
            'adaptive_schedule': False, 'bond_dimension_planning': False,
            'compute_1rdm': True, 'compute_2rdm': True,
            'compute_entanglement': False, 'estimate_energy_error': False,
            'entanglement_active_space_review': False, 'save_mps': False, 'iprint': 1,
        }
        if arm != 'agent':
            options['mpo_algorithm'] = arm
        started = perf_counter()
        result = run_block2_dmrg(hamiltonian, options, scratch_directory=str(directory / 'scratch'))
        elapsed = perf_counter() - started
        arrays = result.pop('_transient_dmrg_arrays')
        energy, rdm1, rdm2 = result['energy'], arrays['rdm1'], arrays['rdm2']
        timings = {**result['timings'], 'solve_and_rdm_seconds': elapsed}
        details = {key: result[key] for key in ('mpo_algorithm', 'converged', 'convergence', 'configuration')}

    reference = None
    if (root / 'fci-reference.npz').exists():
        reference = np.load(root / 'fci-reference.npz')
    energies, errors, checks = _assess_rdms(h1e, g2e, ecore, nelec, energy, rdm1, rdm2, reference)
    np.savez(directory / 'rdms.npz', rdm1=rdm1, rdm2=rdm2)
    _write_json(directory / 'result.json', {
        'arm': arm, 'energy_hartree': float(energy), 'timings': timings,
        'worker_peak_rss_mib': _peak_rss_mib(resource.RUSAGE_SELF),
        'child_peak_rss_mib': _peak_rss_mib(resource.RUSAGE_CHILDREN) if arm == 'native' else None,
        'energy_diagnostics': energies,
        'errors': errors, 'checks': checks, 'passed': all(checks.values()), **details,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True, help='New directory for all inputs, outputs and report.json')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--hamiltonian', type=Path, help='Saved h1e/g2e/ecore/nelec NPZ; skips the small H6/FCI preparation')
    parser.add_argument('--settings', type=Path, help='JSON with schedule, threads and stack_memory_bytes')
    parser.add_argument('--timeout', type=int, default=600, help='Maximum seconds per arm')
    parser.add_argument('--arms', choices=AVAILABLE_ARMS, nargs='+', default=ARMS,
                        help='Paths to compare; agent uses the production default, example calls pyblock2 independently')
    parser.add_argument('--worker', choices=AVAILABLE_ARMS, help=argparse.SUPPRESS)
    parser.add_argument('--measurement-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.work_dir.expanduser().resolve()
    settings_path = root / 'settings.json' if args.worker else args.settings
    settings = json.loads(settings_path.read_text()) if settings_path else {
        'schedule': SCHEDULE, 'threads': 1, 'stack_memory_bytes': 1_000_000_000,
    }
    if settings['stack_memory_bytes'] % 1_000_000_000:
        parser.error('Native dmrgscf requires a whole decimal-GB memory budget')
    # Establish limits before importing any numerical libraries, in every process.
    os.environ['OMP_NUM_THREADS'] = str(settings['threads'])
    for name in ('MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    if args.worker:
        _worker(root, args.worker, args.measurement_dir)
        return 0
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if len(set(args.arms)) != len(args.arms):
        parser.error('--arms must not contain duplicates')
    root.mkdir(parents=True, exist_ok=False)
    _write_json(root / 'settings.json', settings)
    report = {
        **_prepare(root, args.hamiltonian), **settings, 'arms': list(args.arms),
        'python': sys.version, 'platform': platform.platform(),
        'versions': {name: importlib.metadata.version(name) for name in ('pyscf', 'block2', 'pyscf-dmrgscf', 'numpy')},
        'measurements': [],
        'limitations': [
            'Fixed-Hamiltonian benchmark; timing does not include SCF or CASSCF orbital optimization.',
            'Cold independent MPS; native and custom initial wavefunctions are not guaranteed identical.',
            'All arms use SU(2), two-site sweeps, identical integrals/order, M/noise/Davidson schedules and tolerances.',
            'Native kernel time includes executable startup, file I/O, MPO, sweeps and 2-RDM generation; not directly comparable to custom sweep time.',
            'RSS fields are independent process peaks, not summed simultaneous memory; memory allocation is not usage.',
            'For supplied Hamiltonians, self-consistency checks do not establish bond-dimension convergence or equal MPS solutions.',
            'Passed denotes RDM consistency (plus FCI agreement when available), not DMRG convergence.',
            'At finite M, sweep-minimum and final-MPS RDM energies are distinct diagnostics, not a required equality.',
        ],
    }
    for repeat in range(args.repeats):
        # Rotate order to reduce systematic filesystem-cache/order bias.
        offset = repeat % len(args.arms)
        for arm in args.arms[offset:] + args.arms[:offset]:
            directory = root / ('{0}-{1}'.format(repeat + 1, arm))
            directory.mkdir()
            started = perf_counter()
            with (directory / 'worker.log').open('w') as log:
                completed = subprocess.run([
                    sys.executable, str(Path(__file__).resolve()), '--worker', arm,
                    '--work-dir', str(root), '--measurement-dir', str(directory),
                ], stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
            elapsed = perf_counter() - started
            if completed.returncode:
                raise RuntimeError('Benchmark arm failed; inspect ' + str(directory / 'worker.log'))
            measurement = json.loads((directory / 'result.json').read_text())
            measurement.update(repeat=repeat + 1, directory=directory.name, process_wall_seconds=elapsed)
            report['measurements'].append(measurement)
            _write_json(root / 'report.json', report)
            print('{0}: energy={1:.12f}, checks={2}, solve+RDM={3:.3f}s'.format(
                directory.name, measurement['energy_hartree'], measurement['passed'],
                measurement['timings']['solve_and_rdm_seconds']), flush=True)
    report['passed'] = all(item['passed'] for item in report['measurements'])
    report['pairwise_comparisons'] = _compare_measurements(root, report['measurements'], args.arms)
    report['median_solve_and_rdm_seconds'] = {
        arm: statistics.median(item['timings']['solve_and_rdm_seconds'] for item in report['measurements'] if item['arm'] == arm)
        for arm in args.arms
    }
    _write_json(root / 'report.json', report)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
