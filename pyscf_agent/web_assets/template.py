from __future__ import annotations

from pathlib import Path

from ..resources.text import read_text_resource


HTML_PAGE = read_text_resource(
    'pyscf_agent.web_assets', 'assistant_index.html',
    source_dir=Path(__file__).resolve().parent,
)
