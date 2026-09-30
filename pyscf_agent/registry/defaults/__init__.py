from __future__ import annotations

from typing import Any, Callable, Dict, List


from .core import build_core_families
from .molecular import build_molecular_families
from .periodic import build_periodic_families
from .active_space import build_active_space_families
from .model_hamiltonian import build_model_hamiltonian_families
from .postprocessing import build_postprocessing_families
from .study import build_study_families
from .embedding import build_embedding_families
from .artifacts import build_default_artifact_contracts


def build_default_catalog_families(
    item_factory: Callable[..., Any],
    *,
    capability_design_only: str,
    capability_planned: str,
) -> Dict[str, List[Any]]:
    families: Dict[str, List[Any]] = {}
    families.update(build_core_families(
        item_factory,
        capability_design_only=capability_design_only,
        capability_planned=capability_planned,
    ))
    families.update(build_molecular_families(
        item_factory,
        capability_design_only=capability_design_only,
        capability_planned=capability_planned,
    ))
    families.update(build_periodic_families(
        item_factory,
        capability_design_only=capability_design_only,
        capability_planned=capability_planned,
    ))
    families.update(build_active_space_families(
        item_factory,
        capability_design_only=capability_design_only,
        capability_planned=capability_planned,
    ))
    families.update(build_model_hamiltonian_families(
        item_factory,
        capability_design_only=capability_design_only,
        capability_planned=capability_planned,
    ))
    families.update(build_postprocessing_families(
        item_factory,
        capability_design_only=capability_design_only,
        capability_planned=capability_planned,
    ))
    families.update(build_study_families(
        item_factory,
        capability_design_only=capability_design_only,
        capability_planned=capability_planned,
    ))
    families.update(build_embedding_families(
        item_factory,
        capability_design_only=capability_design_only,
        capability_planned=capability_planned,
    ))
    return families


__all__ = ['build_default_artifact_contracts', 'build_default_catalog_families']
