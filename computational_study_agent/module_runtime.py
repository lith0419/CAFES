from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from pyscf_agent.workflow_modules import (
    WorkflowRuntimeError,
    validate_runtime_configuration,
)


@dataclass(frozen=True)
class StudyOrchestrationBinding:
    """Declare which application component owns a compiled study stage."""

    runtime_id: str
    permitted_stages: Tuple[str, ...]
    owner: str
    operation: str


class StudyOrchestrationRegistry:
    """Ownership bindings for stages executed by the study orchestrator."""

    def __init__(self, bindings: Iterable[StudyOrchestrationBinding] = ()):
        self._bindings = {binding.runtime_id: binding for binding in bindings}

    def resolve(self, runtime_id: str) -> Optional[StudyOrchestrationBinding]:
        return self._bindings.get(str(runtime_id or '').strip())

    def ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._bindings))

    def as_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            runtime_id: {
                'runtime_id': binding.runtime_id,
                'permitted_stages': list(binding.permitted_stages),
                'owner': binding.owner,
                'operation': binding.operation,
                'execution_mode': 'orchestrator_owned',
            }
            for runtime_id, binding in sorted(self._bindings.items())
        }


def build_study_module_runtime_registry() -> StudyOrchestrationRegistry:
    entries = (
        ('computational_study_agent.planner.build_study_plan', 'study.prepare', 'build_plan'),
        ('computational_study_agent.executor.run_study', 'study.execute', 'run_study'),
        ('computational_study_agent.adaptive.initial_scan', 'study.probe', 'run_adaptive_study'),
        ('computational_study_agent.adaptive.correlation_routing', 'study.diagnose', 'run_adaptive_study'),
        ('computational_study_agent.adaptive.method_refinement', 'study.configure', 'run_adaptive_study'),
        ('computational_study_agent.adaptive.refined_execution', 'study.execute', 'run_adaptive_study'),
        ('computational_study_agent.adaptive.recovery', 'study.review', 'run_adaptive_study'),
        ('computational_study_agent.adaptive.path_continuity', 'study.review', 'analyze_results'),
        ('computational_study_agent.adaptive.state_tracking', 'study.review', 'analyze_results'),
        ('computational_study_agent.results.integrate', 'study.finalize', 'run_study'),
        ('computational_study_agent.postprocessing.run_postprocessing', 'study.finalize', 'run_postprocessing'),
        (
            'computational_study_agent.hamiltonian_dataset_finalize.finalize_hamiltonian_dataset',
            'study.finalize',
            'run_study',
        ),
        ('computational_study_agent.results.study_report', 'study.finalize', 'run_study'),
    )
    return StudyOrchestrationRegistry([
        StudyOrchestrationBinding(
            runtime_id=runtime_id,
            permitted_stages=(stage,),
            owner='computational_study_agent.application.StudyApplicationService',
            operation=operation,
        )
        for runtime_id, stage, operation in entries
    ])


@functools.lru_cache(maxsize=1)
def get_study_module_runtime_registry() -> StudyOrchestrationRegistry:
    return build_study_module_runtime_registry()


def validate_study_module_runtime(configuration: Mapping[str, Any]) -> None:
    issues = validate_runtime_configuration(configuration, get_study_module_runtime_registry())
    if issues:
        raise WorkflowRuntimeError(issues)


__all__ = [
    'StudyOrchestrationBinding',
    'StudyOrchestrationRegistry',
    'build_study_module_runtime_registry',
    'get_study_module_runtime_registry',
    'validate_study_module_runtime',
]
