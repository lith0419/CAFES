from __future__ import annotations

import contextlib
import io
import os
import shutil
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .contracts import (
    GW_DMFT_RESULT_SCHEMA,
    HF_DMFT_RESULT_SCHEMA,
    validate_gw_dmft_request,
    validate_hf_dmft_request,
)


class FCDMFTExecutionError(RuntimeError):
    """Carry provider diagnostics across the workflow-runtime failure boundary."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        provider_log: str,
        scratch_directory: Path,
        exception_type: str,
    ):
        super().__init__(message)
        self.stage = str(stage)
        self.provider_log = str(provider_log)
        self.scratch_directory = Path(scratch_directory)
        self.exception_type = str(exception_type)


@contextlib.contextmanager
def _legacy_fcdmft_numpy_aliases(numpy_module: Any):
    """Temporarily provide aliases used by the current fcDMFT release."""

    aliases = {
        'complex': complex,
        'int': int,
    }
    added = []
    for name, value in aliases.items():
        if name not in numpy_module.__dict__:
            setattr(numpy_module, name, value)
            added.append(name)
    try:
        yield
    finally:
        for name in added:
            delattr(numpy_module, name)


def _positive_environment_int(name: str) -> Optional[int]:
    try:
        value = int(str(os.environ.get(name) or '').strip())
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _slurm_memory_mb() -> Optional[int]:
    per_node = _positive_environment_int('SLURM_MEM_PER_NODE')
    if per_node is not None:
        return per_node
    per_cpu = _positive_environment_int('SLURM_MEM_PER_CPU')
    cpus = _positive_environment_int('SLURM_CPUS_PER_TASK')
    if per_cpu is not None:
        return per_cpu * int(cpus or 1)
    return None


def _runtime_resources(options: Mapping[str, Any], *, label: str = 'fcDMFT') -> Dict[str, Any]:
    allocated_threads = _positive_environment_int('SLURM_CPUS_PER_TASK')
    requested_threads = options.get('n_threads')
    if requested_threads and allocated_threads and int(requested_threads) > allocated_threads:
        raise ValueError(
            '{0} n_threads={1} exceeds the Slurm allocation of {2} CPU(s)'.format(
                label,
                requested_threads, allocated_threads
            )
        )
    threads = requested_threads or allocated_threads
    threads = threads or _positive_environment_int('OMP_NUM_THREADS') or 1
    allocation_mb = _slurm_memory_mb()
    requested_memory = options.get('max_memory_mb')
    if requested_memory and allocation_mb and int(requested_memory) > allocation_mb:
        raise ValueError(
            '{0} max_memory_mb={1} exceeds the Slurm allocation of {2} MB'.format(
                label,
                requested_memory, allocation_mb
            )
        )
    memory_mb = int(requested_memory or (allocation_mb * 0.75 if allocation_mb else 8000))
    return {
        'n_threads': int(threads),
        'n_threads_source': (
            'solver_options' if requested_threads else (
                'SLURM_CPUS_PER_TASK' if allocated_threads else (
                    'OMP_NUM_THREADS' if _positive_environment_int('OMP_NUM_THREADS') else 'default'
                )
            )
        ),
        'max_memory_mb': memory_mb,
        'max_memory_source': (
            'solver_options' if requested_memory else (
                'slurm_allocation_75_percent' if allocation_mb else 'default'
            )
        ),
        'slurm_memory_allocation_mb': allocation_mb,
    }


def _default_dmft_factory() -> Callable[..., Any]:
    from fcdmft.dmft.gwdmft import DMFT

    return DMFT


def _run_dmft(
    task_spec: Any,
    *,
    scratch_directory: Any,
    gw_dmft: bool,
    reference_chemical_potential: Optional[float] = None,
    reference_occupancy: Optional[float] = None,
    dmft_factory: Optional[Callable[..., Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Run the shared fcDMFT loop from approved embedding artifacts."""

    import numpy as np

    artifact_paths = {}
    if gw_dmft:
        options, inputs, artifact_paths = validate_gw_dmft_request(task_spec)
    else:
        options, inputs = validate_hf_dmft_request(task_spec)
    method_label = 'GW+DMFT' if gw_dmft else 'HF+DMFT'
    method_id = 'gw_dmft' if gw_dmft else 'hf_dmft'
    reference_method = 'dft_gw' if gw_dmft else 'hf'
    result_schema = GW_DMFT_RESULT_SCHEMA if gw_dmft else HF_DMFT_RESULT_SCHEMA
    resources = _runtime_resources(options, label=method_label)
    chemical_potential = options['chemical_potential']
    if chemical_potential is None:
        chemical_potential = reference_chemical_potential
    if chemical_potential is None or not np.isfinite(float(chemical_potential)):
        raise ValueError(
            '{0} requires solver.options.chemical_potential or a finite reference Fermi energy'.format(
                method_label
            )
        )
    occupancy = options['target_occupancy']
    if occupancy is None:
        occupancy = float(reference_occupancy or 0)
    if occupancy <= 0:
        occupancy = float(task_spec.embedding.interaction.get('electron_count') or 0)
    if occupancy <= 0:
        occupancy = float(inputs['metadata'].get('electron_count') or 0)
    if occupancy <= 0:
        raise ValueError('{0} requires solver.options.target_occupancy'.format(method_label))

    scratch = Path(scratch_directory).expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    if gw_dmft:
        for key, target_name in (
            ('lattice_ac', 'ac_coeff.h5'),
            ('local_ac', 'imp_ac_coeff.h5'),
            ('local_transform', 'C_mo_lo.h5'),
        ):
            source = artifact_paths[key]
            target = scratch / target_name
            if source.resolve() != target.resolve():
                shutil.copy2(str(source), str(target))
    output = io.StringIO()
    previous = Path.cwd()
    stage = 'provider_import'
    dmft = None
    try:
        with _legacy_fcdmft_numpy_aliases(np):
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                factory = dmft_factory or _default_dmft_factory()
                stage = 'solver_initialization'
                os.chdir(scratch)
                dmft = factory(
                    inputs['hcore_k'],
                    inputs['jk_k'],
                    inputs['density_k'],
                    inputs['eri'],
                    options['nval'],
                    options['ncore'],
                    options['nbath'],
                    options['nb_per_e'],
                    disc_type=options['bath_discretization'],
                    solver_type=options['impurity_solver'],
                )
                dmft.gw_dmft = bool(gw_dmft)
                dmft.max_cycle = options['max_iterations']
                dmft.conv_tol = options['convergence_tolerance']
                dmft.damp = options['damping']
                dmft.gmres_tol = options['gmres_tolerance']
                dmft.max_memory = resources['max_memory_mb']
                dmft.n_threads = resources['n_threads']
                dmft.diag_only = options['diagonal_bath_fit']
                dmft.chkfile = str(scratch / 'dmft-checkpoint.h5')
                stage = 'self_consistency'
                dmft.kernel(
                    mu0=float(chemical_potential),
                    wl=options['bath_window'][0],
                    wh=options['bath_window'][1],
                    occupancy=float(occupancy),
                    delta=options['broadening'],
                    conv_tol=options['convergence_tolerance'],
                    opt_mu=options['optimize_chemical_potential'],
                    dump_chk=True,
                )
    except Exception as exc:
        provider_log = output.getvalue()
        if provider_log and not provider_log.endswith('\n'):
            provider_log += '\n'
        provider_log += (
            'fcDMFT failure stage: {0}\n'
            'exception: {1}: {2}\n'
            '{3}'
        ).format(stage, type(exc).__name__, exc, traceback.format_exc())
        raise FCDMFTExecutionError(
            'fcDMFT {0} failed during {1}: {2}'.format(method_label, stage, exc),
            stage=stage,
            provider_log=provider_log,
            scratch_directory=scratch,
            exception_type=type(exc).__name__,
        ) from exc
    finally:
        os.chdir(previous)

    arrays = {
        name: np.asarray(value)
        for name, value in (
            ('hybridization', getattr(dmft, 'hyb', None)),
            ('self_energy', getattr(dmft, 'sigma', None)),
            ('frequencies', getattr(dmft, 'freqs', None)),
            ('frequency_weights', getattr(dmft, 'wts', None)),
        )
        if value is not None
    }
    result = {
        'schema': result_schema,
        'provider': 'fcdmft',
        'method': method_id,
        'reference_method': reference_method,
        'impurity_solver': options['impurity_solver'],
        'converged': bool(getattr(dmft, 'converged', False)),
        'chemical_potential': float(getattr(dmft, 'mu', chemical_potential)),
        'target_occupancy': float(occupancy),
        'ncore': options['ncore'],
        'nval': options['nval'],
        'nbath': options['nbath'],
        'nb_per_e': options['nb_per_e'],
        'bath_discretization': options['bath_discretization'],
        'bath_window_hartree': list(options['bath_window']),
        'broadening_hartree': options['broadening'],
        'max_iterations': options['max_iterations'],
        'convergence_tolerance': options['convergence_tolerance'],
        'spin_channels': inputs['spin_channels'],
        'kpoint_count': inputs['nkpts'],
        'localized_orbital_count': inputs['nlo'],
        'initial_hybridization_norm': inputs['initial_hybridization_norm'],
        'runtime_resources': resources,
        'checkpoint_available': (scratch / 'dmft-checkpoint.h5').is_file(),
        'array_fields': sorted(arrays),
        'energy_available': False,
        'energy_note': 'fcDMFT {0} does not expose a total-energy estimator in this adapter.'.format(
            method_label
        ),
    }
    return result, arrays, output.getvalue()


def run_hf_dmft(
    task_spec: Any,
    *,
    scratch_directory: Any,
    reference_chemical_potential: Optional[float] = None,
    reference_occupancy: Optional[float] = None,
    dmft_factory: Optional[Callable[..., Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    return _run_dmft(
        task_spec,
        scratch_directory=scratch_directory,
        gw_dmft=False,
        reference_chemical_potential=reference_chemical_potential,
        reference_occupancy=reference_occupancy,
        dmft_factory=dmft_factory,
    )


def run_gw_dmft(
    task_spec: Any,
    *,
    scratch_directory: Any,
    reference_chemical_potential: Optional[float] = None,
    reference_occupancy: Optional[float] = None,
    dmft_factory: Optional[Callable[..., Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    return _run_dmft(
        task_spec,
        scratch_directory=scratch_directory,
        gw_dmft=True,
        reference_chemical_potential=reference_chemical_potential,
        reference_occupancy=reference_occupancy,
        dmft_factory=dmft_factory,
    )


__all__ = ['FCDMFTExecutionError', 'run_gw_dmft', 'run_hf_dmft']
