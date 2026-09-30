"""Time native DF-CASSCF AO2MO for full and explicitly frozen virtual spaces.

This is a kernel benchmark, not a CASSCF convergence or DMRG calculation. The
input TaskSpec, full MO matrix and reviewed case lists are supplied by the caller.
All arms share one in-memory AO DF tensor and the same CPU/memory settings.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import pyscf
from pyscf import gto, lib, mcscf, scf


def benchmark(task_path, mo_path, cases_path, output, *, threads=16, memory_mb=80000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    task = json.loads(Path(task_path).read_text())
    cases = json.loads(Path(cases_path).read_text())['cases']
    mo = np.load(mo_path, allow_pickle=False)
    lib.num_threads(threads)
    mol = gto.M(**task['system'], verbose=5, max_memory=memory_mb)
    mf = scf.RHF(mol).density_fit(auxbasis=task['density_fitting']['auxbasis'])
    mf.mo_coeff = mo
    active = task['active_space']
    mc = mcscf.CASSCF(mf, int(active['ncas']), active['nelecas']).density_fit()
    nocc = mc.ncore + mc.ncas
    started = perf_counter()
    mf.with_df.build()
    preparation = perf_counter() - started
    report = {
        'kind': 'native_df_casscf_integral_kernel_benchmark',
        'pyscf': pyscf.__version__, 'hostname': platform.node(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'threads': threads, 'max_memory_mb': memory_mb,
        'ao_df_preparation_seconds': preparation,
        'ao_df_storage': 'memory' if isinstance(mf.with_df._cderi, np.ndarray) else 'file',
        'nao': mol.nao_nr(), 'full_nmo': mo.shape[1],
        'ncore': mc.ncore, 'ncas': mc.ncas,
        'mo_sha256': hashlib.sha256(Path(mo_path).read_bytes()).hexdigest(),
        'arms': [],
    }
    # Save only retained reference blocks, outside the timed AO2MO operation.
    # This avoids retaining a full 10+ GB ERIS alongside the next arm in RAM.
    reference = output / 'reference-blocks.h5'
    with h5py.File(reference, 'w') as saved:
        started = perf_counter()
        full = mc.ao2mo(mo)
        seconds = perf_counter() - started
        report['arms'].append(dict(label='full', nmo=mo.shape[1], ao2mo_seconds=seconds,
                                   ppaa_bytes=int(np.prod(full.ppaa.shape) * 8),
                                   papa_bytes=int(np.prod(full.papa.shape) * 8)))
        for case in cases:
            excluded = {int(i) for i in case['frozen_orbital_indices'] if int(i) >= nocc}
            if not excluded:
                continue
            keep = [i for i in range(mo.shape[1]) if i not in excluded]
            group = saved.create_group(case['label'])
            group['keep'] = keep
            group['vhf_c'] = full.vhf_c[np.ix_(keep, keep)]
            group['j_pc'] = full.j_pc[keep]
            group['k_pc'] = full.k_pc[keep]
            ppaa = group.create_dataset('ppaa', (len(keep), len(keep), mc.ncas, mc.ncas), 'f8')
            papa = group.create_dataset('papa', (len(keep), mc.ncas, len(keep), mc.ncas), 'f8')
            for row, original in enumerate(keep):
                ppaa[row] = full.ppaa[original][keep]
                papa[row] = full.papa[original][:, keep]
        full.feri.close()
        del full
    gc.collect()
    (output / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
    with h5py.File(reference, 'r') as saved:
        for label in saved:
            group = saved[label]
            keep = group['keep'][:]
            started = perf_counter()
            projected = mc.ao2mo(mo[:, keep])
            seconds = perf_counter() - started
            errors = {name: float(np.max(np.abs(getattr(projected, name) - group[name][:])))
                      for name in ('vhf_c', 'j_pc', 'k_pc')}
            for name in ('ppaa', 'papa'):
                errors[name] = max(float(np.max(np.abs(getattr(projected, name)[i] - group[name][i])))
                                   for i in range(len(keep)))
            arm = dict(label=label, nmo=len(keep), ao2mo_seconds=seconds,
                       speedup=report['arms'][0]['ao2mo_seconds'] / seconds,
                       ppaa_bytes=int(np.prod(projected.ppaa.shape) * 8),
                       papa_bytes=int(np.prod(projected.papa.shape) * 8), max_abs_errors=errors)
            report['arms'].append(arm)
            (output / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
            print('ARM_RESULT', json.dumps(arm), flush=True)
            projected.feri.close()
            del projected
            gc.collect()
            if max(errors.values()) > 1e-9:
                raise ValueError('Projected ERIs differ from the full-space retained blocks: ' + label)
    report['peak_rss_gib_whole_process'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024**3 if platform.system() == 'Darwin' else 1024**2)
    report['scope'] = 'AO2MO including its native core JK call; excludes DF preparation, comparison I/O and orbital optimization'
    (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-spec', required=True)
    parser.add_argument('--mo-coeff', required=True)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--memory-mb', type=int, default=80000)
    args = parser.parse_args()
    benchmark(args.task_spec, args.mo_coeff, args.cases, args.output,
              threads=args.threads, memory_mb=args.memory_mb)
