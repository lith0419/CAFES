from __future__ import annotations

from pathlib import Path

from .resources.text import read_text_resource


_ASSET_DIR = Path(__file__).resolve().parent / 'web_assets'

# Compatibility exports for callers that still consume the shared preview source.
MOLECULAR_STRUCTURE_PREVIEW_CSS = read_text_resource(
    'pyscf_agent.web_assets', 'molecular-preview.css', source_dir=_ASSET_DIR,
)
MOLECULAR_STRUCTURE_PREVIEW_JS = read_text_resource(
    'pyscf_agent.web_assets', 'molecular-preview.js', source_dir=_ASSET_DIR,
)
