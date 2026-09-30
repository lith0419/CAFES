from __future__ import annotations

import contextlib
import io
import os
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from ...embedding.artifacts import read_embedding_hdf5
from .adapter import FCDMFTExecutionError, _legacy_fcdmft_numpy_aliases, _runtime_resources
from .contracts import (
    PERIODIC_GW_RESULT_SCHEMA,
    load_hf_dmft_inputs,
    normalize_gw_dmft_options,
    validate_periodic_gw_reference,
)


def _default_periodic_gw_factory() -> Callable[..., Any]:
    from fcdmft.gw.pbc.krgw_gf import KRGWGF

    return KRGWGF


def _default_local_gw_factory() -> Callable[..., Any]:
    from fcdmft.gw.mol.gw_dc import GWGF

    return GWGF


def _contiguous_indices(values: Any, count: int, field: str) -> Tuple[int, ...]:
    indices = tuple(int(value) for value in (values or range(count)))
    if not indices:
        raise ValueError('{0} must select at least one index'.format(field))
    if any(index < 0 or index >= count for index in indices):
        raise ValueError('{0} contains an out-of-range index'.format(field))
    if indices != tuple(range(indices[0], indices[-1] + 1)):
        raise ValueError('{0} must form one contiguous window for fcDMFT'.format(field))
    return indices


def _kpoint_indices(values: Any, count: int) -> Tuple[int, ...]:
    indices = tuple(int(value) for value in (values or range(count)))
    if not indices:
        raise ValueError('gw_kpoint_indices must select at least one k point')
    if len(set(indices)) != len(indices) or any(index < 0 or index >= count for index in indices):
        raise ValueError('gw_kpoint_indices contains duplicate or out-of-range indices')
    return indices


def _closed_shell_occupations(mean_field: Any) -> Any:
    import numpy as np

    occupations = np.asarray(mean_field.mo_occ, dtype=float)
    if occupations.ndim == 1:
        occupations = occupations[np.newaxis, ...]
    if occupations.ndim != 2 or not np.all(
        np.isclose(occupations, 0.0, atol=1e-7)
        | np.isclose(occupations, 2.0, atol=1e-7)
    ):
        raise ValueError(
            'The executable periodic GW contract currently requires integer closed-shell occupations'
        )
    return occupations


def _frontier_summary(energies: Any, occupations: Any) -> Dict[str, Optional[float]]:
    import numpy as np

    energy_array = np.asarray(energies, dtype=float)
    occupied = energy_array[np.asarray(occupations) > 1.0]
    virtual = energy_array[np.asarray(occupations) < 1.0]
    homo = float(np.max(occupied)) if occupied.size else None
    lumo = float(np.min(virtual)) if virtual.size else None
    gap = float(lumo - homo) if homo is not None and lumo is not None else None
    return {'homo': homo, 'lumo': lumo, 'gap': gap}


def run_periodic_gw(
    mean_field: Any,
    task_spec: Any,
    *,
    scratch_directory: Any,
    gw_factory: Optional[Callable[..., Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], str, Dict[str, Path]]:
    """Run the reusable k-resolved fcDMFT G0W0 provider stage."""

    import numpy as np

    options = validate_periodic_gw_reference(
        task_spec,
        for_dmft=str(task_spec.solver.name or '').strip().lower().replace('-', '_') == 'gw_dmft',
    )
    if not bool(getattr(mean_field, 'converged', False)):
        raise ValueError('Periodic GW requires a converged periodic HF or DFT reference')
    if getattr(mean_field, 'with_df', None) is None:
        raise ValueError('Periodic GW requires an attached GDF density-fitting object')
    occupations = _closed_shell_occupations(mean_field)
    mean_field_energies = np.asarray(mean_field.mo_energy, dtype=float)
    if mean_field_energies.ndim == 1:
        mean_field_energies = mean_field_energies[np.newaxis, ...]
    nkpts, nmo = mean_field_energies.shape
    orbitals = _contiguous_indices(options['gw_orbital_indices'], nmo, 'gw_orbital_indices')
    kpoints = _kpoint_indices(options['gw_kpoint_indices'], nkpts)
    resources = _runtime_resources(options, label='periodic GW')
    omega = np.linspace(
        options['gw_frequency_window'][0],
        options['gw_frequency_window'][1],
        options['gw_real_frequency_points'],
    )
    scratch = Path(scratch_directory).expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    output = io.StringIO()
    previous = Path.cwd()
    stage = 'provider_import'
    gw = None
    try:
        with _legacy_fcdmft_numpy_aliases(np):
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                factory = gw_factory or _default_periodic_gw_factory()
                stage = 'solver_initialization'
                os.chdir(scratch)
                gw = factory(mean_field)
                gw.ac = options['gw_analytic_continuation']
                gw.eta = options['gw_broadening']
                gw.fullsigma = options['gw_full_self_energy']
                gw.fc = options['gw_finite_size_correction']
                gw.ev = options['gw_quasiparticle_energies']
                gw.max_memory = resources['max_memory_mb']
                stage = 'lattice_gw'
                gw.kernel(
                    omega=omega,
                    orbs=orbitals,
                    kptlist=kpoints,
                    writefile=1,
                    nw=options['gw_imaginary_frequency_points'],
                )
    except Exception as exc:
        provider_log = output.getvalue()
        if provider_log and not provider_log.endswith('\n'):
            provider_log += '\n'
        provider_log += (
            'fcDMFT failure stage: {0}\nexception: {1}: {2}\n{3}'
        ).format(stage, type(exc).__name__, exc, traceback.format_exc())
        raise FCDMFTExecutionError(
            'fcDMFT periodic GW failed during {0}: {1}'.format(stage, exc),
            stage=stage,
            provider_log=provider_log,
            scratch_directory=scratch,
            exception_type=type(exc).__name__,
        ) from exc
    finally:
        os.chdir(previous)

    provider_files = {
        'lattice_ac': scratch / 'ac_coeff.h5',
        'lattice_vxc': scratch / 'vxc.h5',
        'lattice_sigma_imag': scratch / 'sigma_imag.h5',
    }
    missing = [path.name for path in provider_files.values() if not path.is_file()]
    if missing:
        raise FCDMFTExecutionError(
            'fcDMFT periodic GW did not produce required provider file(s): {0}'.format(
                ', '.join(missing)
            ),
            stage='artifact_collection',
            provider_log=output.getvalue(),
            scratch_directory=scratch,
            exception_type='MissingProviderArtifact',
        )
    try:
        import h5py

        with h5py.File(provider_files['lattice_ac'], 'r') as handle:
            fermi_energy = float(np.asarray(handle['fermi']).reshape(-1)[0])
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise FCDMFTExecutionError(
            'fcDMFT periodic GW analytic-continuation artifact does not contain a finite Fermi energy',
            stage='artifact_collection',
            provider_log=output.getvalue(),
            scratch_directory=scratch,
            exception_type=type(exc).__name__,
        ) from exc
    if not np.isfinite(fermi_energy):
        raise FCDMFTExecutionError(
            'fcDMFT periodic GW analytic-continuation artifact contains a non-finite Fermi energy',
            stage='artifact_collection',
            provider_log=output.getvalue(),
            scratch_directory=scratch,
            exception_type='InvalidProviderArtifact',
        )
    raw_qp_energies = getattr(gw, 'mo_energy', None)
    qp_available = bool(
        options['gw_quasiparticle_energies'] and raw_qp_energies is not None
    )
    qp_energies = np.full(mean_field_energies.shape, np.nan, dtype=float)
    qp_availability = np.zeros(mean_field_energies.shape, dtype=bool)
    if qp_available:
        raw_qp_energies = np.asarray(raw_qp_energies, dtype=float)
        if raw_qp_energies.shape != mean_field_energies.shape:
            raise FCDMFTExecutionError(
                'fcDMFT periodic GW returned quasiparticle energies with an incompatible shape',
                stage='artifact_collection',
                provider_log=output.getvalue(),
                scratch_directory=scratch,
                exception_type='InvalidProviderResult',
            )
        for kpoint in kpoints:
            qp_energies[kpoint, list(orbitals)] = raw_qp_energies[kpoint, list(orbitals)]
            qp_availability[kpoint, list(orbitals)] = True
    qp_frontier = (
        _frontier_summary(qp_energies, occupations)
        if qp_available
        and len(kpoints) == nkpts
        and orbitals[0] == 0
        and orbitals[-1] == nmo - 1
        else {'homo': None, 'lumo': None, 'gap': None}
    )
    result = {
        'schema': PERIODIC_GW_RESULT_SCHEMA,
        'provider': 'fcdmft',
        'method': 'gw',
        'reference_method': str(task_spec.method.name or '').strip().lower(),
        'status': 'completed',
        'analytic_continuation': options['gw_analytic_continuation'],
        'full_self_energy': options['gw_full_self_energy'],
        'finite_size_correction': options['gw_finite_size_correction'],
        'quasiparticle_energies_requested': options['gw_quasiparticle_energies'],
        'quasiparticle_energies_available': qp_available,
        'quasiparticle_homo': qp_frontier['homo'],
        'quasiparticle_lumo': qp_frontier['lumo'],
        'quasiparticle_gap': qp_frontier['gap'],
        'fermi_energy': fermi_energy,
        'orbital_indices': list(orbitals),
        'kpoint_indices': list(kpoints),
        'kpoint_count': nkpts,
        'orbital_count': nmo,
        'real_frequency_window_hartree': list(options['gw_frequency_window']),
        'real_frequency_points': options['gw_real_frequency_points'],
        'imaginary_frequency_points': options['gw_imaginary_frequency_points'],
        'broadening_hartree': options['gw_broadening'],
        'runtime_resources': resources,
        'energy_available': False,
        'energy_note': 'The periodic GW provider exposes quasiparticle energies and self-energy artifacts, not a total energy.',
    }
    arrays = {
        'real_frequency_hartree': omega,
        'mean_field_orbital_energies': mean_field_energies,
        'quasiparticle_orbital_energies': qp_energies,
        'quasiparticle_energy_available': qp_availability,
        'mean_field_occupations': occupations,
    }
    return result, arrays, output.getvalue(), provider_files


def _mo_to_local_transform(mean_field: Any, coefficients_ao_lo: Any) -> Tuple[Any, Any]:
    import numpy as np

    mo_coefficients = np.asarray(mean_field.mo_coeff)
    overlap = np.asarray(mean_field.get_ovlp())
    local_coefficients = np.asarray(coefficients_ao_lo)
    if mo_coefficients.ndim == 2:
        mo_coefficients = mo_coefficients[np.newaxis, ...]
    if overlap.ndim == 2:
        overlap = overlap[np.newaxis, ...]
    if local_coefficients.ndim == 2:
        local_coefficients = local_coefficients[np.newaxis, ...]
    if not (
        mo_coefficients.ndim == overlap.ndim == local_coefficients.ndim == 3
        and mo_coefficients.shape[0] == overlap.shape[0] == local_coefficients.shape[0]
    ):
        raise ValueError('GW double counting requires matching restricted k-point MO, overlap, and LO arrays')
    transformed = np.empty(
        (1, mo_coefficients.shape[0], mo_coefficients.shape[-1], local_coefficients.shape[-1]),
        dtype=np.result_type(mo_coefficients, overlap, local_coefficients, complex),
    )
    for kpoint in range(mo_coefficients.shape[0]):
        transformed[0, kpoint] = (
            mo_coefficients[kpoint].conj().T
            @ overlap[kpoint]
            @ local_coefficients[kpoint]
        )
    return local_coefficients[np.newaxis, ...], transformed


def _reference_fermi_energy(mean_field: Any) -> float:
    occupations = _closed_shell_occupations(mean_field)
    frontier = _frontier_summary(mean_field.mo_energy, occupations)
    if frontier['homo'] is None or frontier['lumo'] is None:
        raise ValueError('GW double counting requires occupied and virtual reference orbitals')
    return 0.5 * (frontier['homo'] + frontier['lumo'])


def run_local_gw_double_counting(
    mean_field: Any,
    task_spec: Any,
    *,
    scratch_directory: Any,
    gw_factory: Optional[Callable[..., Any]] = None,
) -> Tuple[Dict[str, Any], str, Dict[str, Path]]:
    """Build the local GW double-counting artifacts consumed by GW+DMFT."""

    import h5py
    import numpy as np
    from pyscf import gto, scf

    options = normalize_gw_dmft_options(task_spec.solver.options)
    inputs = load_hf_dmft_inputs(task_spec.embedding)
    subspace_path = Path(str(task_spec.embedding.localized_subspace_artifact.get('path') or ''))
    subspace = read_embedding_hdf5(subspace_path)
    coefficients = subspace['datasets'].get('coefficients_ao_lo')
    if coefficients is None:
        raise ValueError('GW double counting requires coefficients_ao_lo in the localized-subspace artifact')
    c_ao_lo, c_mo_lo = _mo_to_local_transform(mean_field, coefficients)
    nlo = int(c_mo_lo.shape[-1])
    nocc = int(getattr(mean_field.cell, 'nelectron', 0)) // 2
    if nocc <= 0 or nocc > nlo:
        raise ValueError('GW double counting requires 0 < occupied orbital count <= localized orbital count')
    resources = _runtime_resources(options, label='GW double counting')
    scratch = Path(scratch_directory).expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    output = io.StringIO()
    previous = Path.cwd()
    stage = 'local_orbital_transform'
    try:
        with _legacy_fcdmft_numpy_aliases(np):
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                from fcdmft.utils import cholesky

                os.chdir(scratch)
                with h5py.File('C_mo_lo.h5', 'w') as handle:
                    handle['C_ao_lo'] = np.asarray(c_ao_lo)
                    handle['C_mo_lo'] = np.asarray(c_mo_lo)
                stage = 'local_eri_factorization'
                factorization = cholesky.cholesky(inputs['eri'][0], tau=1e-8, dimQ=50)
                cderi = factorization.kernel()
                cderi = np.asarray(cderi).reshape(-1, nlo, nlo)
                stage = 'local_gw_double_counting'
                molecule = gto.M(verbose=0)
                molecule.max_memory = resources['max_memory_mb']
                local_reference = scf.RHF(molecule)
                factory = gw_factory or _default_local_gw_factory()
                gw = factory(local_reference)
                gw.nmo = nlo
                gw.nocc = nocc
                gw.eta = options['gw_broadening']
                gw.ac = options['gw_analytic_continuation']
                gw.ef = _reference_fermi_energy(mean_field)
                gw.max_memory = resources['max_memory_mb']
                omega = np.linspace(
                    options['gw_frequency_window'][0],
                    options['gw_frequency_window'][1],
                    options['gw_real_frequency_points'],
                )
                gw.kernel(
                    Lpq=cderi,
                    omega=omega,
                    kmf=mean_field,
                    C_mo_lo=c_mo_lo,
                    nw=options['gw_imaginary_frequency_points'],
                    nt=options['gw_dc_imaginary_time_points'],
                )
    except Exception as exc:
        provider_log = output.getvalue()
        if provider_log and not provider_log.endswith('\n'):
            provider_log += '\n'
        provider_log += (
            'fcDMFT failure stage: {0}\nexception: {1}: {2}\n{3}'
        ).format(stage, type(exc).__name__, exc, traceback.format_exc())
        raise FCDMFTExecutionError(
            'fcDMFT local GW double counting failed during {0}: {1}'.format(stage, exc),
            stage=stage,
            provider_log=provider_log,
            scratch_directory=scratch,
            exception_type=type(exc).__name__,
        ) from exc
    finally:
        os.chdir(previous)

    provider_files = {
        'local_ac': scratch / 'imp_ac_coeff.h5',
        'local_sigma_imag': scratch / 'gw_dc_sigmaI.h5',
        'local_transform': scratch / 'C_mo_lo.h5',
    }
    missing = [path.name for path in provider_files.values() if not path.is_file()]
    if missing:
        raise FCDMFTExecutionError(
            'fcDMFT local GW did not produce required provider file(s): {0}'.format(', '.join(missing)),
            stage='artifact_collection',
            provider_log=output.getvalue(),
            scratch_directory=scratch,
            exception_type='MissingProviderArtifact',
        )
    summary = {
        'provider': 'fcdmft',
        'method': 'gw_double_counting',
        'eri_cholesky_tolerance': 1e-8,
        'status': 'completed',
        'localized_orbital_count': nlo,
        'occupied_orbital_count': nocc,
        'analytic_continuation': options['gw_analytic_continuation'],
        'imaginary_frequency_points': options['gw_imaginary_frequency_points'],
        'imaginary_time_points': options['gw_dc_imaginary_time_points'],
        'runtime_resources': resources,
    }
    return summary, output.getvalue(), provider_files


__all__ = ['run_local_gw_double_counting', 'run_periodic_gw']
