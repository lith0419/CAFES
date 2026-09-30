from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple


WORKFLOW_CONFIGURATION_SCHEMA = 'pyscf-agent.workflow-configuration.v1'
WORKFLOW_PROVENANCE_SCHEMA = 'pyscf-agent.workflow-provenance.v1'
STUDY_WORKFLOW_CONFIGURATION_SCHEMA = 'pyscf-agent.study-workflow-configuration.v1'
STUDY_WORKFLOW_PROVENANCE_SCHEMA = 'pyscf-agent.study-workflow-provenance.v1'
WORKFLOW_COMPILER_VERSION = '1.0'

TASK_STAGE_ORDER = (
    'task.prepare',
    'task.execute',
    'task.recover',
    'task.extract',
    'task.diagnose',
    'task.finalize',
)

STUDY_STAGE_ORDER = (
    'study.prepare',
    'study.probe',
    'study.diagnose',
    'study.configure',
    'study.execute',
    'study.review',
    'study.finalize',
)


@dataclass(frozen=True)
class DataPortContract:
    name: str
    schema: str = ''
    merge_policy: str = 'single'
    optional: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModuleCompatibility:
    task_types: Tuple[str, ...] = ()
    methods: Tuple[str, ...] = ()
    jobs: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActivationRule:
    path: str
    operator: str = 'truthy'
    value: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModuleContract:
    module_id: str
    version: str
    permitted_stages: Tuple[str, ...]
    default_stage: str
    required_inputs: Tuple[DataPortContract, ...] = ()
    provided_outputs: Tuple[DataPortContract, ...] = ()
    compatibility: ModuleCompatibility = field(default_factory=ModuleCompatibility)
    optional_configuration: Dict[str, Any] = field(default_factory=dict)
    default_configuration: Dict[str, Any] = field(default_factory=dict)
    capability_ids: Tuple[str, ...] = ()
    requires_modules: Tuple[str, ...] = ()
    conflicts_with: Tuple[str, ...] = ()
    exclusive_group: str = ''
    before: Tuple[str, ...] = ()
    after: Tuple[str, ...] = ()
    activation_rules: Tuple[ActivationRule, ...] = ()
    always_select: bool = False
    description: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedModuleContract(ModuleContract):
    """Module contract joined with its sole provider/runtime binding."""

    provider_id: str = ''
    runtime_id: str = ''


@dataclass(frozen=True)
class CompilationIssue:
    code: str
    message: str
    module_id: str = ''
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WorkflowCompilationError(ValueError):
    def __init__(self, issues: Tuple[CompilationIssue, ...], provenance: Dict[str, Any]):
        self.issues = issues
        self.provenance = provenance
        super().__init__('; '.join(issue.message for issue in issues))


@dataclass(frozen=True)
class WorkflowConfiguration:
    workflow_id: str
    compiler_version: str
    dependency_policy: str
    nodes: Tuple[Dict[str, Any], ...]
    edges: Tuple[Dict[str, Any], ...]
    hooks: Dict[str, Tuple[str, ...]]
    execution_order: Tuple[str, ...]
    expected_outputs: Tuple[str, ...]
    provenance: Dict[str, Any]
    schema: str = WORKFLOW_CONFIGURATION_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StudyWorkflowConfiguration:
    workflow_id: str
    compiler_version: str
    dependency_policy: str
    nodes: Tuple[Dict[str, Any], ...]
    edges: Tuple[Dict[str, Any], ...]
    hooks: Dict[str, Tuple[str, ...]]
    execution_order: Tuple[str, ...]
    expected_outputs: Tuple[str, ...]
    provenance: Dict[str, Any]
    schema: str = STUDY_WORKFLOW_CONFIGURATION_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
