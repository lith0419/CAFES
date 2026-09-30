"""Postprocessing use cases, composed by StudyApplicationService."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from ..postprocessing import PostprocessContext
from .types import StudyArtifactError, ArtifactContent


def run_postprocessing(
    self,
    report: Dict[str, Any],
    *,
    specs: Optional[List[Dict[str, Any]]] = None,
    actions: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {'specs': specs}
    if actions:
        kwargs['actions'] = actions
        collector = getattr(self._task_executor, 'collect_files', None)
        if callable(collector):
            kwargs['artifact_collector'] = collector
        generator = getattr(
            self._task_executor,
            'generate_hamiltonian_dataset',
            None,
        )
        if callable(generator):
            kwargs['dataset_generator'] = generator
    return self._postprocessor(self._require_report(report), **kwargs)


def suggest_postprocessing(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
    return self._postprocess_suggester(self._require_report(report))


def postprocessing_context(self, report: Dict[str, Any]) -> Dict[str, Any]:
    return PostprocessContext.from_report(self._require_report(report)).view()


def read_artifact(
    self,
    report: Dict[str, Any],
    artifact_path: str,
    *,
    allowed_suffixes: Optional[Sequence[str]] = None,
) -> ArtifactContent:
    report_payload = self._require_report(report)
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        raise StudyArtifactError('path_required', 'Artifact path is required')
    work_dir = report_payload.get('work_dir')
    if not isinstance(work_dir, str) or not work_dir.strip():
        raise StudyArtifactError(
            'work_dir_required', 'StudyReport work_dir is required'
        )

    try:
        root = Path(work_dir).expanduser().resolve()
        requested = Path(artifact_path).expanduser()
        path = (
            requested.resolve()
            if requested.is_absolute()
            else (root / requested).resolve()
        )
        path.relative_to(root)
    except Exception as exc:
        raise StudyArtifactError(
            'outside_work_dir',
            'Artifact path is outside the study work directory',
        ) from exc
    if not path.is_file():
        raise StudyArtifactError('not_found', 'Artifact file does not exist')
    normalized_suffixes = {
        str(suffix).strip().lower()
        for suffix in (allowed_suffixes or ())
        if str(suffix).strip()
    }
    if normalized_suffixes and path.suffix.lower() not in normalized_suffixes:
        raise StudyArtifactError('unsupported_type', 'Artifact type is not supported')
    media_type = mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
    return ArtifactContent(
        path=str(path), media_type=media_type, content=path.read_bytes()
    )
