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
    for distribution in ('libdmet', 'libdmet-preview'):
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return ''


def libdmet_availability() -> Dict[str, Any]:
    library_available = _module_available('libdmet')
    # Probing a nested libdmet module imports the package and prints its banner.
    # Keep capability discovery side-effect free; the adapter import validates
    # the concrete transform module when an embedding operation is requested.
    transforms_available = library_available
    hdf5_available = _module_available('h5py')
    numpy_available = _module_available('numpy')
    scipy_available = _module_available('scipy')
    pyscf_available = _module_available('pyscf')
    available = bool(
        library_available
        and transforms_available
        and hdf5_available
        and numpy_available
        and scipy_available
        and pyscf_available
    )
    missing = [
        label
        for label, present in (
            ('libdmet', library_available and transforms_available),
            ('h5py', hdf5_available),
            ('numpy', numpy_available),
            ('scipy', scipy_available),
            ('pyscf', pyscf_available),
        )
        if not present
    ]
    return {
        'provider': 'provider.libdmet',
        'available': available,
        'library_available': library_available,
        'transforms_available': transforms_available,
        'hdf5_available': hdf5_available,
        'numpy_available': numpy_available,
        'scipy_available': scipy_available,
        'pyscf_available': pyscf_available,
        'version': _version(),
        'distribution': 'libdmet' if _version() else 'source_or_unknown',
        'supported_stages': ['embedding_preparation', 'hubbard_dmet'],
        'full_dmet_available': available,
        'reason': '' if available else 'Missing optional dependency: {0}.'.format(', '.join(missing)),
    }


def libdmet_is_available() -> bool:
    return bool(libdmet_availability()['available'])
