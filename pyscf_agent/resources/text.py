"""Read packaged text with an explicit source-checkout fallback."""

from __future__ import annotations

from importlib import resources
from pathlib import Path


def read_text_resource(package: str, name: str, *, source_dir: Path) -> str:
    """Read a trusted resource name from its package or source directory.

    Callers specify the directory containing the resource, not the directory
    of an arbitrary importing module. HTTP path validation belongs to the
    serving adapter; this helper is for application-owned asset names.
    """
    try:
        return resources.files(package).joinpath(name).read_text(encoding='utf-8')
    except (FileNotFoundError, ModuleNotFoundError):
        return source_dir.joinpath(name).read_text(encoding='utf-8')
