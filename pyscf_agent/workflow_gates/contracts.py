from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

from ..workflow_modules.contracts import ActivationRule, ModuleCompatibility


GATE_CONFIGURATION_SCHEMA = 'pyscf-agent.gate-configuration.v1'
GATE_PROVENANCE_SCHEMA = 'pyscf-agent.gate-provenance.v1'
GATE_DECISION_SCHEMA = 'pyscf-agent.gate-decision.v1'
GATE_EXECUTION_TRACE_SCHEMA = 'pyscf-agent.gate-execution-trace.v1'
GATE_COMPILER_VERSION = '1.0'

GATE_STATUS_PASSED = 'passed'
GATE_STATUS_REVIEW_REQUIRED = 'review_required'
GATE_STATUS_RETRY = 'retry'
GATE_STATUS_BLOCKED = 'blocked'
GATE_STATUS_SKIPPED = 'skipped'
GATE_STATUSES = (
    GATE_STATUS_PASSED,
    GATE_STATUS_REVIEW_REQUIRED,
    GATE_STATUS_RETRY,
    GATE_STATUS_BLOCKED,
    GATE_STATUS_SKIPPED,
)

TASK_GATE_HOOKS = (
    'task.after_compile',
    'task.before_execute',
    'task.after_recovery',
    'task.before_finalize',
)

STUDY_GATE_HOOKS = (
    'study.after_compile',
    'study.before_execute',
    'study.after_execute',
    'study.after_review',
    'study.before_finalize',
)


@dataclass(frozen=True)
class EvidenceContract:
    name: str
    path: str
    schema: str = ''
    optional: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateContract:
    gate_id: str
    version: str
    scope: str
    permitted_hooks: Tuple[str, ...]
    default_hook: str
    evaluator_id: str
    required_evidence: Tuple[EvidenceContract, ...] = ()
    compatibility: ModuleCompatibility = field(default_factory=ModuleCompatibility)
    optional_configuration: Dict[str, Any] = field(default_factory=dict)
    default_configuration: Dict[str, Any] = field(default_factory=dict)
    conflicts_with: Tuple[str, ...] = ()
    before: Tuple[str, ...] = ()
    after: Tuple[str, ...] = ()
    activation_rules: Tuple[ActivationRule, ...] = ()
    always_select: bool = False
    mandatory: bool = False
    priority: int = 100
    description: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateCompilationIssue:
    code: str
    message: str
    gate_id: str = ''
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GateCompilationError(ValueError):
    def __init__(self, issues: Tuple[GateCompilationIssue, ...], provenance: Dict[str, Any]):
        self.issues = issues
        self.provenance = provenance
        super().__init__('; '.join(issue.message for issue in issues))


@dataclass(frozen=True)
class GateConfiguration:
    gate_set_id: str
    scope: str
    compiler_version: str
    nodes: Tuple[Dict[str, Any], ...]
    hooks: Dict[str, Tuple[str, ...]]
    evaluation_order: Tuple[str, ...]
    provenance: Dict[str, Any]
    schema: str = GATE_CONFIGURATION_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateDecision:
    gate_id: str
    scope: str
    hook: str
    status: str
    summary: str
    checks: Tuple[Dict[str, Any], ...] = ()
    recommended_actions: Tuple[Dict[str, Any], ...] = ()
    evidence: Tuple[Dict[str, Any], ...] = ()
    affected_case_ids: Tuple[str, ...] = ()
    details: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    schema: str = GATE_DECISION_SCHEMA

    def __post_init__(self) -> None:
        if self.status not in GATE_STATUSES:
            raise ValueError("Unsupported gate status: '{0}'".format(self.status))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = [
    'EvidenceContract',
    'GATE_COMPILER_VERSION',
    'GATE_CONFIGURATION_SCHEMA',
    'GATE_DECISION_SCHEMA',
    'GATE_EXECUTION_TRACE_SCHEMA',
    'GATE_PROVENANCE_SCHEMA',
    'GATE_STATUSES',
    'GATE_STATUS_BLOCKED',
    'GATE_STATUS_PASSED',
    'GATE_STATUS_RETRY',
    'GATE_STATUS_REVIEW_REQUIRED',
    'GATE_STATUS_SKIPPED',
    'GateCompilationError',
    'GateCompilationIssue',
    'GateConfiguration',
    'GateContract',
    'GateDecision',
    'STUDY_GATE_HOOKS',
    'TASK_GATE_HOOKS',
]
