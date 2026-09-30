from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from pyscf_agent.schema_contracts import (
    ADAPTIVE_STUDY_REPORT_SCHEMA,
    STUDY_PLAN_SCHEMA,
    STUDY_REPORT_SCHEMA,
    STUDY_SPEC_SCHEMA,
    validate_public_payload,
)
from pyscf_agent.lifecycle import ensure_lifecycle
from .normalization import normalize_molecular_geometry_fields, normalize_case_variables, normalize_system_type


@dataclass
class StudySpec:
    name: str
    objective: str
    system_type: str
    base_task: Dict[str, Any] = field(default_factory=dict)
    base_model_input_file: Optional[str] = None
    base_model_spec: Dict[str, Any] = field(default_factory=dict)
    case_design: Dict[str, Any] = field(default_factory=dict)
    sweep: Dict[str, List[Any]] = field(default_factory=dict)
    observables: List[str] = field(default_factory=lambda: ['energy'])
    comparison: Dict[str, Any] = field(default_factory=dict)
    resource_policy: Dict[str, Any] = field(default_factory=dict)
    workflow: Dict[str, Any] = field(default_factory=dict)
    study_mode: str = 'static'
    study_workflow: Dict[str, Any] = field(default_factory=dict)
    quality_gates: Dict[str, Any] = field(default_factory=dict)
    grid_refinement: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.system_type = normalize_system_type(self.system_type)
        self.base_task = copy.deepcopy(self.base_task)
        self.case_design = copy.deepcopy(self.case_design)
        if isinstance(self.case_design, dict):
            if 'variables' in self.case_design:
                self.case_design['variables'] = normalize_case_variables(self.case_design['variables'])
        if self.system_type == 'molecular':
            normalize_molecular_geometry_fields(self.base_task)
            nodes = [self.case_design.get('template', {})] if isinstance(self.case_design, dict) else []
            for key in ('cases', 'overrides'):
                values = self.case_design.get(key, []) if isinstance(self.case_design, dict) else []
                if isinstance(values, list):
                    nodes.extend(values)
            for node in nodes:
                if isinstance(node, dict):
                    normalize_molecular_geometry_fields(node.get('request_updates'))

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'StudySpec':
        if not isinstance(payload, dict):
            raise TypeError('StudySpec payload must be a dictionary')
        if payload.get('schema') is not None:
            validate_public_payload(payload, expected_schema=STUDY_SPEC_SCHEMA)
        return cls(
            name=str(payload.get('name') or 'computational-study'),
            objective=str(payload.get('objective') or 'compare_results'),
            system_type=str(payload.get('system_type') or payload.get('task_type') or 'molecular'),
            base_task=copy.deepcopy(payload.get('base_task') or {}),
            base_model_input_file=payload.get('base_model_input_file'),
            base_model_spec=copy.deepcopy(payload.get('base_model_spec') or {}),
            case_design=copy.deepcopy(payload.get('case_design') or {}),
            sweep=copy.deepcopy(payload.get('sweep') or {}),
            observables=list(payload.get('observables') or ['energy']),
            comparison=copy.deepcopy(payload.get('comparison') or {}),
            resource_policy=copy.deepcopy(payload.get('resource_policy') or {}),
            workflow=copy.deepcopy(payload.get('workflow') or {}),
            study_mode=str(payload.get('study_mode') or 'static'),
            study_workflow=copy.deepcopy(payload.get('study_workflow') or {}),
            quality_gates=copy.deepcopy(payload.get('quality_gates') or {}),
            grid_refinement=copy.deepcopy(payload.get('grid_refinement', {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload['schema'] = STUDY_SPEC_SCHEMA
        return payload


@dataclass
class StudyCase:
    case_id: str
    label: str
    request: Dict[str, Any]
    variables: Dict[str, Any] = field(default_factory=dict)
    operations: List[Dict[str, Any]] = field(default_factory=list)
    model_spec: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'StudyCase':
        if not isinstance(payload, dict):
            raise TypeError('StudyCase payload must be a dictionary')
        return cls(
            case_id=str(payload.get('case_id') or 'case'),
            label=str(payload.get('label') or payload.get('case_id') or 'case'),
            request=copy.deepcopy(payload.get('request') or {}),
            variables=copy.deepcopy(payload.get('variables') or {}),
            operations=copy.deepcopy(payload.get('operations') or []),
            model_spec=copy.deepcopy(payload.get('model_spec') or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StudyPlan:
    study_id: str
    name: str
    objective: str
    system_type: str
    cases: List[StudyCase]
    observables: List[str]
    comparison: Dict[str, Any] = field(default_factory=dict)
    capability_snapshot: Dict[str, Any] = field(default_factory=dict)
    resource_policy: Dict[str, Any] = field(default_factory=dict)
    cost_estimate: Dict[str, Any] = field(default_factory=dict)
    grid_refinement: Dict[str, Any] = field(default_factory=dict)
    grid_refinement_source: Dict[str, Any] = field(default_factory=dict)
    workflow_configuration: Dict[str, Any] = field(default_factory=dict)
    workflow_provenance: Dict[str, Any] = field(default_factory=dict)
    gate_configuration: Dict[str, Any] = field(default_factory=dict)
    gate_provenance: Dict[str, Any] = field(default_factory=dict)
    gate_decisions: List[Dict[str, Any]] = field(default_factory=list)
    lifecycle: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.lifecycle = ensure_lifecycle(
            self.lifecycle,
            'study',
            entity_id=self.study_id,
        )

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'StudyPlan':
        if not isinstance(payload, dict):
            raise TypeError('StudyPlan payload must be a dictionary')
        if payload.get('schema') is not None:
            validate_public_payload(payload, expected_schema=STUDY_PLAN_SCHEMA)
        cases_payload = payload.get('cases') or []
        if not isinstance(cases_payload, list):
            raise TypeError('StudyPlan cases must be a list')
        return cls(
            study_id=str(payload.get('study_id') or 'study'),
            name=str(payload.get('name') or 'computational-study'),
            objective=str(payload.get('objective') or 'compare_results'),
            system_type=str(payload.get('system_type') or 'molecular'),
            cases=[StudyCase.from_dict(item) for item in cases_payload],
            observables=list(payload.get('observables') or ['energy']),
            comparison=copy.deepcopy(payload.get('comparison') or {}),
            capability_snapshot=copy.deepcopy(payload.get('capability_snapshot') or {}),
            resource_policy=copy.deepcopy(payload.get('resource_policy') or {}),
            cost_estimate=copy.deepcopy(payload.get('cost_estimate') or {}),
            grid_refinement=copy.deepcopy(payload.get('grid_refinement', {})),
            grid_refinement_source=copy.deepcopy(payload.get('grid_refinement_source') or {}),
            workflow_configuration=copy.deepcopy(payload.get('workflow_configuration') or {}),
            workflow_provenance=copy.deepcopy(payload.get('workflow_provenance') or {}),
            gate_configuration=copy.deepcopy(payload.get('gate_configuration') or {}),
            gate_provenance=copy.deepcopy(payload.get('gate_provenance') or {}),
            gate_decisions=copy.deepcopy(payload.get('gate_decisions') or []),
            lifecycle=copy.deepcopy(payload.get('lifecycle') or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload['schema'] = STUDY_PLAN_SCHEMA
        payload['cases'] = [case.to_dict() for case in self.cases]
        return payload


@dataclass
class StudyReport:
    study_id: str
    name: str
    objective: str
    system_type: str
    status: str
    work_dir: str
    cases: List[Dict[str, Any]]
    comparison_table: List[Dict[str, Any]]
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ''
    execution: Dict[str, Any] = field(default_factory=dict)
    gate_configuration: Dict[str, Any] = field(default_factory=dict)
    gate_provenance: Dict[str, Any] = field(default_factory=dict)
    gate_decisions: List[Dict[str, Any]] = field(default_factory=list)
    gate_execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    workflow: Dict[str, Any] = field(default_factory=dict)
    lifecycle: Dict[str, Any] = field(default_factory=dict)
    dataset_manifest: Dict[str, Any] = field(default_factory=dict)
    grid_refinement: Dict[str, Any] = field(default_factory=dict)
    adaptive: Optional[Dict[str, Any]] = None
    postprocessing: Optional[Dict[str, Any]] = None
    schema: str = STUDY_REPORT_SCHEMA
    _extra_fields: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.schema not in (STUDY_REPORT_SCHEMA, ADAPTIVE_STUDY_REPORT_SCHEMA):
            raise ValueError('Unsupported StudyReport schema: {0}'.format(self.schema))
        for name in ('adaptive', 'postprocessing'):
            value = getattr(self, name)
            if value is not None and not isinstance(value, dict):
                raise TypeError('StudyReport {0} must be an object or null'.format(name))
        if self.schema == ADAPTIVE_STUDY_REPORT_SCHEMA and self.adaptive is None:
            raise ValueError('Adaptive StudyReport requires an adaptive object')
        self.lifecycle = ensure_lifecycle(
            self.lifecycle,
            'study',
            entity_id=self.study_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        extra_fields = payload.pop('_extra_fields')
        # Additive v1 fields remain at their original top-level keys. Declared
        # fields always reflect the current object, including caller edits.
        return {**extra_fields, **payload}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'StudyReport':
        if not isinstance(payload, dict):
            raise TypeError('StudyReport payload must be a dictionary')
        schema = payload.get('schema') or STUDY_REPORT_SCHEMA
        if schema not in (STUDY_REPORT_SCHEMA, ADAPTIVE_STUDY_REPORT_SCHEMA):
            raise ValueError('Unsupported StudyReport schema: {0}'.format(schema))
        validate_public_payload(
            payload,
            expected_schema=schema,
            allow_missing_schema=True,
        )
        field_names = {name for name, definition in cls.__dataclass_fields__.items() if definition.init}
        report = cls(**{
            key: copy.deepcopy(value)
            for key, value in payload.items()
            if key in field_names and key != 'schema'
        }, schema=schema)
        report._extra_fields = {
            key: copy.deepcopy(value)
            for key, value in payload.items()
            if key not in field_names
        }
        return report
