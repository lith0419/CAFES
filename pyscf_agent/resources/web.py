"""Content-derived asset URLs for both packaged Web surfaces."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache, wraps
from importlib import resources
from pathlib import Path

_ASSETS = re.compile(
    r'(?P<attr>\b(?:src|href)=")(?:'
    + r'(?P<study>/computational-study/assets/)|(?P<task>/assets/))'
    + r'(?P<name>[\w.-]+\.(?:js|css))(?:\?[^"\s]*)?"'
)


@lru_cache(maxsize=256)
def _file_digest(path: Path, signature: tuple) -> str:
    # Stat identity is part of the key so editable installs update without restart.
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


@lru_cache(maxsize=128)
def _package_digest(package: str, name: str) -> str:
    # Non-filesystem packages (e.g. zip imports) are immutable for this process.
    return hashlib.sha256(resources.files(package).joinpath(name).read_bytes()).hexdigest()[:12]


def version_asset_urls(html: str) -> str:
    def replace(match):
        package = (
            'computational_study_agent.web_assets'
            if match['study']
            else 'pyscf_agent.web_assets'
        )
        asset = resources.files(package).joinpath(match['name'])
        if isinstance(asset, Path):
            stat = asset.stat()
            version = _file_digest(asset.resolve(), (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        else:
            version = _package_digest(package, match['name'])
        return f'{match["attr"]}{match["study"] or match["task"]}{match["name"]}?v={version}"'

    return _ASSETS.sub(replace, html)


def versioned_assets(render):
    @wraps(render)
    def wrapped(*args, **kwargs):
        return version_asset_urls(render(*args, **kwargs))

    return wrapped
