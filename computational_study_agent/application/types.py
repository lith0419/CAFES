from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple
from ..schema import StudyPlan
from ..validation import ValidationIssue


class StudyApplicationError(Exception):
    """Base class for failures at the study use-case boundary."""


class StudyApplicationValidationError(StudyApplicationError, ValueError):
    def __init__(self, stage: str, issues: Iterable[ValidationIssue]):
        self.stage = str(stage or 'study')
        self.issues = tuple(issues)
        first_error = next(
            (
                issue
                for issue in self.issues
                if getattr(issue, 'severity', '') == 'error'
            ),
            None,
        )
        detail = getattr(first_error, 'message', '') if first_error is not None else ''
        message = '{0} validation failed'.format(self.stage.replace('_', ' ').title())
        if detail:
            message = '{0}: {1}'.format(message, detail)
        super().__init__(message)


class StudyFeatureUnavailableError(StudyApplicationError):
    """Raised when an optional application dependency is unavailable."""


class StudyArtifactError(StudyApplicationError):
    def __init__(self, code: str, message: str):
        self.code = str(code or 'artifact_error')
        super().__init__(message)


@dataclass(frozen=True)
class ArtifactContent:
    path: str
    media_type: str
    content: bytes


@dataclass(frozen=True)
class StudyPlanResult:
    plan: StudyPlan
    validation_issues: Tuple[ValidationIssue, ...]
