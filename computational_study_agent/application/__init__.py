from __future__ import annotations

from .service import (
    ArtifactContent,
    StudyApplicationError,
    StudyApplicationService,
    StudyApplicationValidationError,
    StudyArtifactError,
    StudyFeatureUnavailableError,
    StudyPlanResult,
)


_DEFAULT_STUDY_APPLICATION_SERVICE = StudyApplicationService()
_DEFAULT_STUDY_EXECUTION_TARGET = 'local'
_STUDY_APPLICATION_SERVICES = {
    _DEFAULT_STUDY_EXECUTION_TARGET: _DEFAULT_STUDY_APPLICATION_SERVICE,
}


def get_study_application_service(execution_target: str = None) -> StudyApplicationService:
    """Return the process-wide stateless study application service."""

    target_id = str(execution_target or _DEFAULT_STUDY_EXECUTION_TARGET).strip().lower()
    service = _STUDY_APPLICATION_SERVICES.get(target_id)
    if service is None:
        raise ValueError(
            'Unknown execution target {0}. Available targets: {1}'.format(
                target_id,
                ', '.join(_STUDY_APPLICATION_SERVICES),
            )
        )
    return service


def set_study_application_service(service: StudyApplicationService) -> None:
    """Replace the process-wide service used by the built-in web API."""

    if not isinstance(service, StudyApplicationService):
        raise TypeError('service must be a StudyApplicationService')
    global _DEFAULT_STUDY_APPLICATION_SERVICE
    global _DEFAULT_STUDY_EXECUTION_TARGET
    global _STUDY_APPLICATION_SERVICES
    _DEFAULT_STUDY_APPLICATION_SERVICE = service
    _DEFAULT_STUDY_EXECUTION_TARGET = 'local'
    _STUDY_APPLICATION_SERVICES = {'local': service}


def set_study_application_services(services, *, default_target: str) -> None:
    """Register request-selectable study services keyed by execution target."""

    if not isinstance(services, dict) or not services:
        raise ValueError('services must be a non-empty dictionary')
    normalized = {}
    for raw_target_id, service in services.items():
        target_id = str(raw_target_id or '').strip().lower()
        if not target_id:
            raise ValueError('Study execution target ids must be non-empty')
        if not isinstance(service, StudyApplicationService):
            raise TypeError('Each service must be a StudyApplicationService')
        normalized[target_id] = service
    normalized_default = str(default_target or '').strip().lower()
    if normalized_default not in normalized:
        raise ValueError('Default study execution target is not registered')

    global _DEFAULT_STUDY_APPLICATION_SERVICE
    global _DEFAULT_STUDY_EXECUTION_TARGET
    global _STUDY_APPLICATION_SERVICES
    _DEFAULT_STUDY_EXECUTION_TARGET = normalized_default
    _STUDY_APPLICATION_SERVICES = normalized
    _DEFAULT_STUDY_APPLICATION_SERVICE = normalized[normalized_default]


__all__ = [
    'ArtifactContent',
    'StudyApplicationError',
    'StudyApplicationService',
    'StudyApplicationValidationError',
    'StudyArtifactError',
    'StudyFeatureUnavailableError',
    'StudyPlanResult',
    'get_study_application_service',
    'set_study_application_service',
    'set_study_application_services',
]
