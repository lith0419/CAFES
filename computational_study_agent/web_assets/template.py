from __future__ import annotations

from pathlib import Path

from pyscf_agent.resources.text import read_text_resource


HTML_PAGE = read_text_resource(
    'computational_study_agent.web_assets', 'planner_index.html',
    source_dir=Path(__file__).resolve().parent,
)
