"""Datasets use cases, composed by StudyApplicationService."""

from __future__ import annotations

from typing import Any, Iterable, Optional
from computational_study_agent.datasets.hamiltonian.contracts import (
    HamiltonianDatasetManifest,
    MolecularGeometry,
)
from ..schema import StudyPlan
from ..validation import ValidationIssue


def prepare_dataset(
    self, dataset_spec, seed_geometries, *, work_dir=None, resource_profile=None
):
    plan = self.build_hamiltonian_dataset_plan(
        dataset_spec, seed_geometries, resource_profile=resource_profile
    )
    self._save_prepared_plan(plan, work_dir=work_dir)
    return {
        'study_id': plan.study_id,
        'mode': 'static',
        'plan': plan.to_dict(),
        'validation_issues': [],
    }


def load_dataset(self, study_id, *, work_dir=None):
    manifest = self.load_report(study_id, work_dir=work_dir).get('dataset_manifest')
    if not manifest:
        raise ValueError(
            'No dataset manifest is available; run and collect a dataset Study first'
        )
    return manifest


def build_hamiltonian_dataset_plan(
    self,
    dataset_spec: Any,
    seed_geometries: Iterable[Any],
    *,
    resource_profile: Optional[str] = None,
) -> StudyPlan:
    """Compile one dataset campaign into one trajectory case per molecule."""

    plan = self._hamiltonian_dataset_plan_builder(
        dataset_spec,
        [
            MolecularGeometry.from_dict(item) if isinstance(item, dict) else item
            for item in seed_geometries
        ],
    )
    plan = self._bind_plan_resources(plan, resource_profile)
    issues = list(self._plan_validator(plan))
    from pyscf_agent.application import CalculationApplicationService

    validator = CalculationApplicationService()
    for case in plan.cases:
        validated = validator.validate_task_spec(case.request)
        issues.extend(
            ValidationIssue(
                severity='error',
                code=error['code'],
                message=error['message'],
                path='cases.' + case.case_id,
            )
            for error in validated['errors']
        )
    self._raise_for_issues('hamiltonian_dataset_plan', issues)
    return plan


def finalize_hamiltonian_dataset(
    self,
    study_report: Any,
    dataset_spec: Any,
    *,
    output_dir: Any,
) -> HamiltonianDatasetManifest:
    """Assemble accepted/rejected sample indexes from completed Task artifacts."""

    return self._hamiltonian_dataset_finalizer(
        study_report,
        dataset_spec,
        output_dir,
    )
