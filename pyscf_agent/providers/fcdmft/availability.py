from __future__ import annotations

import importlib.util
from importlib import metadata
from typing import Any, Dict


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False


def _version() -> str:
    for distribution in ('fcdmft', 'fcDMFT'):
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return ''


def fcdmft_availability() -> Dict[str, Any]:
    """Discover fcDMFT without importing its MPI-enabled solver modules."""

    library_available = _module_available('fcdmft')
    dependencies = {
        'mpi4py': _module_available('mpi4py'),
        'numpy': _module_available('numpy'),
        'scipy': _module_available('scipy'),
        'h5py': _module_available('h5py'),
        'pyscf': _module_available('pyscf'),
    }
    available = library_available and all(dependencies.values())
    missing = [name for name, present in dependencies.items() if not present]
    if not library_available:
        missing.insert(0, 'fcdmft')
    version = _version()
    return {
        'provider': 'provider.fcdmft',
        'available': bool(available),
        'library_available': bool(library_available),
        **{name + '_available': bool(value) for name, value in dependencies.items()},
        'version': version,
        'distribution': 'fcdmft' if version else 'source_or_unknown',
        'supported_stages': [
            'hf_reference',
            'dft_reference',
            'periodic_gw',
            'localization',
            'subspace_audit',
            'operator_transform',
            'interaction_transform',
            'gw_double_counting',
            'dmft_bath',
            'impurity_solution',
        ],
        'reason': '' if available else 'Missing optional dependency: {0}.'.format(', '.join(missing)),
    }


def fcdmft_is_available() -> bool:
    return bool(fcdmft_availability()['available'])


__all__ = ['fcdmft_availability', 'fcdmft_is_available']
