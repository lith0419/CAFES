from __future__ import annotations

import importlib.util
from importlib import metadata
from typing import Any, Dict


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return ''


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False


def block2_availability() -> Dict[str, Any]:
    extension_available = _module_available('block2')
    python_api_available = _module_available('pyblock2.driver.core')
    serial_version = _distribution_version('block2')
    mpi_version = _distribution_version('block2-mpi')
    conflict = bool(serial_version and mpi_version)
    available = bool(extension_available and python_api_available and not conflict)
    if conflict:
        reason = 'Both block2 and block2-mpi are installed; install exactly one provider distribution.'
    elif not extension_available or not python_api_available:
        reason = 'The optional block2 Python extension and pyblock2 driver API are required.'
    else:
        reason = ''
    return {
        'provider': 'block2_dmrg',
        'available': available,
        'extension_available': extension_available,
        'python_api_available': python_api_available,
        'distribution': 'block2-mpi' if mpi_version else ('block2' if serial_version else 'source_or_unknown'),
        'version': mpi_version or serial_version,
        'mpi_enabled': bool(mpi_version),
        'distribution_conflict': conflict,
        'reason': reason,
    }


def block2_is_available() -> bool:
    return bool(block2_availability()['available'])
