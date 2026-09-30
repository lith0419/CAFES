"""Stable task specification and report contracts.

The calculation backend consumes these contracts; workflow implementation
details must not be imported into this module.
"""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field
from .timestamps import utc_timestamp as utc_timestamp
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from .input_validation import boolean, finite_float, integer, integer_list, validate_dataclass_input
from .schema_contracts import TASK_REPORT_SCHEMA, TASK_SPEC_SCHEMA, validate_public_payload
from .registry.platform import (
    COMMON_XC_FUNCTIONALS as COMMON_XC_FUNCTIONALS,
    DEFAULT_ANALYSIS,
    DEFAULT_PERIODIC_BASIS_SET,
    SUPPORTED_ANALYSIS as SUPPORTED_ANALYSIS,
    SUPPORTED_AUXBASIS_SETS as SUPPORTED_AUXBASIS_SETS,
    SUPPORTED_JOBS as SUPPORTED_JOBS,
    SUPPORTED_METHODS as SUPPORTED_METHODS,
    SUPPORTED_MODEL_SOLVERS as SUPPORTED_MODEL_SOLVERS,
    SUPPORTED_PERIODIC_ANALYSIS as SUPPORTED_PERIODIC_ANALYSIS,
    SUPPORTED_PERIODIC_BAND_PATH_MODES as SUPPORTED_PERIODIC_BAND_PATH_MODES,
    SUPPORTED_PERIODIC_BASIS_SETS as SUPPORTED_PERIODIC_BASIS_SETS,
    SUPPORTED_PERIODIC_DENSITY_FITTING_METHODS as SUPPORTED_PERIODIC_DENSITY_FITTING_METHODS,
    SUPPORTED_PERIODIC_EXXDIV_OPTIONS as SUPPORTED_PERIODIC_EXXDIV_OPTIONS,
    SUPPORTED_PERIODIC_JOBS as SUPPORTED_PERIODIC_JOBS,
    SUPPORTED_PERIODIC_KPOINT_SCHEMES as SUPPORTED_PERIODIC_KPOINT_SCHEMES,
    SUPPORTED_PERIODIC_METHODS as SUPPORTED_PERIODIC_METHODS,
    SUPPORTED_PERIODIC_PSEUDOPOTENTIALS as SUPPORTED_PERIODIC_PSEUDOPOTENTIALS,
    SUPPORTED_PERIODIC_SMEARING_METHODS as SUPPORTED_PERIODIC_SMEARING_METHODS,
    SUPPORTED_PERIODIC_XC_FUNCTIONALS as SUPPORTED_PERIODIC_XC_FUNCTIONALS,
    SUPPORTED_TASK_TYPES as SUPPORTED_TASK_TYPES,
)
REQUEST_HINT_KEYS = (
    'request',
    'message',
    'prompt',
)
BASIS_SET_PATTERN = re.compile(
    r'(?<![a-z0-9])('
    r'minao'
    r'|sto-\d+g'
    r'|[346]-\d{1,3}(?:\+\+?)?g(?:\([a-z,]+\)|\*{1,2})?'
    r'|(?:aug-)?cc-p(?:v|cv|wcv)(?:d|t|q|5|6)z'
    r'|def2-(?:svp|svpd|svpp|svppd|m?tzvp|m?tzvpp|tzvpd|tzvppd|qzvp|qzvpp|qzvpd|qzvppd)'
    r'|ano(?:-?rcc)?'
    r'|lanl(?:2dz|2tz|08)'
    r')(?![a-z0-9])'
)
SAFE_TOKEN_PATTERN = re.compile(r'[a-z0-9+*(),/_-]+')


@dataclass
class MessageEnvelope:
    role: str
    kind: str
    content: str
    channel: str = 'agent'
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_timestamp)


@dataclass
class LogEntry:
    level: str
    event: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_timestamp)


@dataclass
class SystemSpec:
    atom: Optional[str] = None
    basis: Optional[str] = None
    unit: str = 'Angstrom'
    charge: int = 0
    spin: int = 0
    symmetry: bool = False


@dataclass
class MethodSpec:
    name: str = 'hf'
    restricted: Optional[bool] = None
    xc: Optional[str] = None


@dataclass
class JobSpec:
    name: str = 'single_point'


@dataclass
class AnalysisSpec:
    outputs: List[str] = field(default_factory=lambda: list(DEFAULT_ANALYSIS))


@dataclass
class RuntimeSpec:
    max_cycle: int = 50
    conv_tol: Optional[float] = None
    conv_tol_grad: Optional[float] = None
    grid_level: Optional[int] = None
    diis_space: Optional[int] = None
    verbose: int = 4
    scf_algorithm: str = 'standard'


@dataclass
class MolecularDynamicsSpec:
    """Born-Oppenheimer molecular-dynamics settings for one trajectory.

    ``sample_offset`` is a zero-based PySCF frame index.  With the QH9-like
    defaults below, frames 9, 19, ..., 99 are retained from a 100-frame NVE
    trajectory, producing ten dataset structures without treating frames as
    independent calculation tasks.
    """

    profile: Optional[str] = None
    ensemble: str = 'nve'
    temperature_kelvin: float = 300.0
    time_step_au: float = 50.0
    steps: int = 100
    sample_stride: int = 10
    sample_offset: int = 9
    velocity_seed: int = 0
    ao_convention: str = 'qh9'
    store_velocities: bool = True
    store_fock: bool = True
    store_overlap: bool = True

    @property
    def sampled_frame_indices(self) -> Tuple[int, ...]:
        if self.sample_stride <= 0 or self.sample_offset < 0:
            return ()
        return tuple(range(self.sample_offset, self.steps, self.sample_stride))


@dataclass
class InitialStateSpec:
    """Reference to a reusable numerical initial state.

    Numerical arrays remain in run artifacts.  A TaskSpec only records the
    artifact provenance and the deterministic application mode.
    """

    mode: str = 'none'
    source_case_id: Optional[str] = None
    source_artifact: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelHamiltonianSpec:
    spec: Dict[str, Any] = field(default_factory=dict)
    input_file: Optional[str] = None


@dataclass
class PeriodicSystemSpec:
    structure_format: str = 'poscar'
    structure_text: Optional[str] = None
    basis: str = DEFAULT_PERIODIC_BASIS_SET
    pseudo: str = 'gth-pbe'
    kmesh: Tuple[int, int, int] = (1, 1, 1)
    kpoint_scheme: str = 'gamma_centered'
    kpoint_shift: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dimension: int = 3
    precision: float = 1e-8
    ke_cutoff: Optional[float] = None
    fft_mesh: Optional[Tuple[int, int, int]] = None
    density_fitting_method: str = 'fft'
    density_fitting_auxbasis: Optional[str] = None
    exxdiv: str = 'ewald'
    smearing_method: str = 'none'
    smearing_sigma: Optional[float] = None
    smearing_fix_spin: bool = False
    band_path_mode: str = 'auto'
    band_path: Optional[str] = None
    band_path_special_points: Dict[str, Tuple[float, ...]] = field(default_factory=dict)
    band_path_npoints: int = 80
    band_path_reference_distance: float = 0.025
    band_path_symprec: float = 1e-5


@dataclass
class SolverSpec:
    name: str = 'fci'
    options: Dict[str, Any] = field(default_factory=dict)


def materialize_block2_state_average_weights(
    solver_name: Any,
    options: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Make block2's documented equal-weight multi-root default explicit."""

    normalized = copy.deepcopy(options) if isinstance(options, dict) else {}
    canonical_solver = str(solver_name or '').strip().lower().replace('-', '_')
    if canonical_solver not in ('block2', 'block2_dmrg', 'dmrg'):
        return normalized
    raw_nroots = normalized.get('nroots')
    if isinstance(raw_nroots, bool):
        return normalized
    try:
        nroots = int(raw_nroots)
    except (TypeError, ValueError):
        return normalized
    if str(raw_nroots).strip() not in (str(nroots), '{0}.0'.format(nroots)):
        return normalized
    if nroots <= 1:
        weights = normalized.get('state_average_weights')
        if isinstance(weights, (list, tuple)) and len(weights) == 1 and finite_float(weights[0], 'state_average_weights') == 1.0:
            normalized.pop('state_average_weights')
        return normalized
    if 'state_average_weights' in normalized and normalized['state_average_weights'] is not None:
        return normalized
    normalized['state_average_weights'] = [1.0 / nroots] * nroots
    return normalized


@dataclass
class OrbitalProcessingSpec:
    enabled: bool = False
    localization_method: str = 'none'
    localization_scope: str = 'analysis'
    localization_occupation_thresholds: List[float] = field(default_factory=list)
    use_natural_orbitals: bool = False
    orbital_ordering: str = 'canonical'
    orbital_order: List[int] = field(default_factory=list)
    frozen_orbital_indices: List[int] = field(default_factory=list)
    continuation_policy: str = 'project_all'


def normalize_orbital_processing(value: Any, *, default_scope: str = 'analysis') -> Dict[str, Any]:
    payload = validate_dataclass_input(value, OrbitalProcessingSpec, 'orbital_processing')
    defaults = asdict(OrbitalProcessingSpec(localization_scope=default_scope))
    defaults.update(payload)
    payload = defaults
    for field in ('localization_method', 'localization_scope', 'orbital_ordering', 'continuation_policy'):
        payload[field] = str(payload[field]).strip().lower().replace('-', '_')
    if payload['localization_method'] == 'pm':
        payload['localization_method'] = 'pipek_mezey'
    for field, allowed in (
        ('localization_method', ('none', 'boys', 'pipek_mezey')),
        ('localization_scope', ('analysis', 'active_space')),
        ('orbital_ordering', ('canonical', 'fiedler', 'manual')),
        ('continuation_policy', ('project_all', 'target_scf_core')),
    ):
        if payload[field] not in allowed:
            raise ValueError('Unsupported orbital_processing.{0}: {1}'.format(field, payload[field]))
    thresholds = payload['localization_occupation_thresholds']
    if not isinstance(thresholds, (list, tuple)):
        raise ValueError('orbital_processing.localization_occupation_thresholds must be a list')
    thresholds = [finite_float(x, 'orbital_processing.localization_occupation_thresholds') for x in thresholds]
    if any(not 0 < x < 2 for x in thresholds) or any(a >= b for a, b in zip(thresholds, thresholds[1:])):
        raise ValueError('localization_occupation_thresholds must be strictly increasing values between 0 and 2')
    if thresholds and (payload['localization_scope'] != 'active_space' or payload['localization_method'] == 'none'):
        raise ValueError('localization_occupation_thresholds requires active_space localization')
    payload['localization_occupation_thresholds'] = thresholds
    order = integer_list(payload['orbital_order'], 'orbital_processing.orbital_order')
    if order and (payload['orbital_ordering'] != 'manual' or sorted(order) != list(range(len(order)))):
        raise ValueError('orbital_order must be a complete zero-based permutation with manual ordering')
    payload['orbital_order'] = order
    frozen = integer_list(payload['frozen_orbital_indices'], 'orbital_processing.frozen_orbital_indices')
    if any(index < 0 for index in frozen) or len(set(frozen)) != len(frozen):
        raise ValueError('frozen_orbital_indices must contain unique nonnegative MO indices')
    payload['frozen_orbital_indices'] = frozen
    if 'enabled' not in value:
        payload['enabled'] = (payload['localization_method'] != 'none'
                              or payload['orbital_ordering'] != 'canonical'
                              or payload['use_natural_orbitals'] or bool(frozen))
    return payload


@dataclass
class DensityFittingSpec:
    """SCF fitting, or SCF plus restricted CASSCF integral fitting.

    Validation upgrades legacy SCF requests on restricted CASSCF to the
    combined scope, preserving PySCF's previously implicit inheritance.
    """
    enabled: bool = False
    auxbasis: Optional[str] = None
    apply_to: str = 'scf'


@dataclass
class EmbeddingPreparationSpec:
    """Reviewable input boundary shared by DMET and DMFT preparation."""

    enabled: bool = False
    provider: str = 'libdmet'
    localization_method: str = 'manual'
    minimal_basis: str = 'minao'
    include_pao: bool = True
    correlated_orbital_indices: List[int] = field(default_factory=list)
    correlated_atom_indices: List[int] = field(default_factory=list)
    fragments: List[Dict[str, Any]] = field(default_factory=list)
    interaction: Dict[str, Any] = field(default_factory=dict)
    approved: bool = False
    audit_artifact: Dict[str, Any] = field(default_factory=dict)
    reference_artifact: Dict[str, Any] = field(default_factory=dict)
    localized_subspace_artifact: Dict[str, Any] = field(default_factory=dict)
    localized_hamiltonian_artifact: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActiveSpaceSpec:
    enabled: bool = False
    selection_method: str = 'manual'
    ncas: Optional[int] = None
    nelecas: Optional[Any] = None
    orbital_indices: Any = field(default_factory=list)
    occupation_window: Tuple[float, float] = (0.02, 1.98)
    energy_window: Optional[float] = None
    avas_targets: List[str] = field(default_factory=list)
    avas_threshold: float = 0.2
    avas_minimal_basis: str = 'minao'
    avas_with_iao: bool = False
    avas_openshell_option: int = 2
    avas_ncore: int = 0
    # Small inline matrix or pyscf-agent.numeric-array.v1 reference; loaded by CAS only.
    initial_mo_coeff: Any = None
    target_method: Optional[str] = None
    target_solver: Optional[str] = None
    target_solver_options: Dict[str, Any] = field(default_factory=dict)
    approved: bool = False


@dataclass
class SCNEVPT2Spec:
    enabled: bool = False
    root: int = 0
    density_fit: bool = True


@dataclass
class PostCASSpec:
    sc_nevpt2: SCNEVPT2Spec = field(default_factory=SCNEVPT2Spec)


@dataclass
class WorkflowModuleSpec:
    modules: List[str] = field(default_factory=list)
    module_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dependency_policy: str = 'auto'


@dataclass
class QualityGateSpec:
    enabled: bool = True
    gates: List[str] = field(default_factory=list)
    gate_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class TaskSpec:
    task_type: str = 'molecular'
    system: SystemSpec = field(default_factory=SystemSpec)
    method: MethodSpec = field(default_factory=MethodSpec)
    job: JobSpec = field(default_factory=JobSpec)
    analysis: AnalysisSpec = field(default_factory=AnalysisSpec)
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)
    molecular_dynamics: MolecularDynamicsSpec = field(default_factory=MolecularDynamicsSpec)
    initial_state: InitialStateSpec = field(default_factory=InitialStateSpec)
    model_hamiltonian: ModelHamiltonianSpec = field(default_factory=ModelHamiltonianSpec)
    periodic: PeriodicSystemSpec = field(default_factory=PeriodicSystemSpec)
    solver: SolverSpec = field(default_factory=SolverSpec)
    orbital_processing: OrbitalProcessingSpec = field(default_factory=OrbitalProcessingSpec)
    density_fitting: DensityFittingSpec = field(default_factory=DensityFittingSpec)
    embedding: EmbeddingPreparationSpec = field(default_factory=EmbeddingPreparationSpec)
    active_space: ActiveSpaceSpec = field(default_factory=ActiveSpaceSpec)
    post_cas: PostCASSpec = field(default_factory=PostCASSpec)
    workflow: WorkflowModuleSpec = field(default_factory=WorkflowModuleSpec)
    quality_gates: QualityGateSpec = field(default_factory=QualityGateSpec)


@dataclass
class TaskReport:
    """Structured result contract emitted by one TaskSpec execution."""

    run_id: Optional[str] = None
    work_dir: str = ''
    channel: str = ''
    calculation_role: str = 'calculation'
    task_spec: Dict[str, Any] = field(default_factory=dict)
    generated_input: Optional[str] = None
    execution_status: str = 'pending'
    structured_results: Optional[Dict[str, Any]] = None
    compact_results: Dict[str, Any] = field(default_factory=dict)
    analysis_summary: str = ''
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    applied_defaults: List[Dict[str, Any]] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    clarification_questions: List[str] = field(default_factory=list)
    approval: Optional[Dict[str, Any]] = None
    raw_stdout: str = ''
    raw_scf_output: str = ''
    analysis_text: str = ''
    raw_stderr: str = ''
    retry_count: int = 0
    max_retries: int = 0
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    workflow_configuration: Dict[str, Any] = field(default_factory=dict)
    workflow_provenance: Dict[str, Any] = field(default_factory=dict)
    module_execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    module_runtime_observations: List[Dict[str, Any]] = field(default_factory=list)
    gate_configuration: Dict[str, Any] = field(default_factory=dict)
    gate_provenance: Dict[str, Any] = field(default_factory=dict)
    gate_decisions: List[Dict[str, Any]] = field(default_factory=list)
    gate_execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    lifecycle: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload['schema'] = TASK_REPORT_SCHEMA
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'TaskReport':
        validate_public_payload(
            payload,
            expected_schema=TASK_REPORT_SCHEMA,
            allow_missing_schema=True,
        )
        field_names = cls.__dataclass_fields__
        return cls(**{
            key: copy.deepcopy(value)
            for key, value in payload.items()
            if key in field_names
        })


class WorkflowState(TypedDict, total=False):
    warnings: List[Dict[str, Any]]
    run_id: str
    work_dir: str
    channel: str
    calculation_role: str
    locale: str
    user_request: str
    task_spec: Dict[str, Any]
    attempts: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    applied_defaults: List[Dict[str, Any]]
    validation_errors: List[str]
    clarification_questions: List[str]
    generated_input: Optional[str]
    execution_status: str
    raw_stdout: str
    raw_scf_output: str
    analysis_text: str
    raw_stderr: str
    structured_results: Optional[Dict[str, Any]]
    compact_results: Dict[str, Any]
    analysis_summary: str
    retry_count: int
    max_retries: int
    artifacts: List[Dict[str, Any]]
    messages: List[Dict[str, Any]]
    logs: List[Dict[str, Any]]
    workflow_configuration: Dict[str, Any]
    workflow_provenance: Dict[str, Any]
    module_execution_trace: List[Dict[str, Any]]
    module_runtime_observations: List[Dict[str, Any]]
    gate_configuration: Dict[str, Any]
    gate_provenance: Dict[str, Any]
    gate_decisions: List[Dict[str, Any]]
    gate_execution_trace: List[Dict[str, Any]]
    lifecycle: Dict[str, Any]
    module_context_ref: Dict[str, Any]
    _module_results: Dict[str, Any]
    task_report: Dict[str, Any]


def message_to_dict(message: MessageEnvelope) -> Dict[str, Any]:
    return asdict(message)


def log_entry_to_dict(entry: LogEntry) -> Dict[str, Any]:
    return asdict(entry)


def _filter_known_fields(data: Optional[Dict[str, Any]], allowed_fields: Tuple[str, ...]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    return {
        key: value for key, value in data.items()
        if key in allowed_fields
    }


def _normalize_outputs_value(value: Any) -> List[str]:
    if value is None:
        return list(DEFAULT_ANALYSIS)
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ValueError('analysis.outputs must be a string or a list of strings')


def _normalize_optional_bool(value: Any) -> Optional[bool]:
    return None if value is None else boolean(value)


def _coerce_int(value: Any, default: int) -> int:
    return default if value is None else integer(value)


def _coerce_optional_float(value: Any) -> Optional[float]:
    return None if value is None else finite_float(value)


def _coerce_bool(value: Any, default: bool) -> bool:
    return default if value is None else boolean(value)


def _coerce_optional_int(value: Any) -> Optional[int]:
    return None if value is None or value == '' else integer(value)


def _coerce_int_list(value: Any) -> List[int]:
    if value is None or value == '':
        return []
    if isinstance(value, str):
        raw_items = [item for item in re.split(r'[,\s]+', value.strip()) if item]
    elif isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raise ValueError('orbital/site indices must be a list of integers')
    return [integer(item) for item in raw_items]


def _coerce_orbital_indices(value: Any) -> Any:
    if isinstance(value, dict):
        normalized = {}
        alpha = _coerce_int_list(value.get('alpha', value.get('a')))
        beta = _coerce_int_list(value.get('beta', value.get('b')))
        if alpha:
            normalized['alpha'] = alpha
        if beta:
            normalized['beta'] = beta
        return normalized if normalized else []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if re.search(r'\b(?:alpha|beta|a|b)\s*[:=]', text, flags=re.IGNORECASE):
            normalized = {}
            for spin, aliases in (('alpha', ('alpha', 'a')), ('beta', ('beta', 'b'))):
                for alias in aliases:
                    match = re.search(
                        r'(?:^|[;|])\s*{0}\s*[:=]\s*([^;|]+)'.format(alias),
                        text,
                        flags=re.IGNORECASE,
                    )
                    if match:
                        indices = _coerce_int_list(match.group(1))
                        if indices:
                            normalized[spin] = indices
                        break
            return normalized if normalized else []
    return _coerce_int_list(value)


def _coerce_nelecas(value: Any) -> Optional[Any]:
    if value is None or value == '':
        return None
    if isinstance(value, str):
        value = [item for item in re.split(r'[,\s]+', value.strip()) if item]
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return integer(value[0], 'active_space.nelecas')
        if len(value) == 2:
            return tuple(integer(item, 'active_space.nelecas') for item in value)
        raise ValueError('active_space.nelecas requires an integer or an alpha/beta pair')
    return integer(value, 'active_space.nelecas')


def _coerce_float_window(value: Any, default: Tuple[float, float]) -> Tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, str):
        value = [item for item in re.split(r'[,\s]+', value.strip()) if item]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('occupation_window requires two finite numbers')
    lower, upper = (finite_float(item, 'occupation_window') for item in value)
    if lower > upper:
        raise ValueError('occupation_window lower bound must not exceed upper bound')
    return lower, upper


def _coerce_float(value: Any, default: float) -> float:
    return default if value is None else finite_float(value)


def _coerce_positive_int_tuple(value: Any, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if value is None:
        return default
    if isinstance(value, str):
        value = [item for item in re.split(r'[,xX\s]+', value.strip()) if item]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError('mesh requires three integers')
    return tuple(integer(item, 'mesh') for item in value)


def _coerce_optional_positive_int_tuple(value: Any) -> Optional[Tuple[int, int, int]]:
    return None if value is None or value == '' else _coerce_positive_int_tuple(value, (0, 0, 0))


def _coerce_float_tuple(value: Any, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if value is None:
        return default
    if isinstance(value, str):
        value = [item for item in re.split(r'[,xX\s]+', value.strip()) if item]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError('coordinate tuple requires three finite numbers')
    return tuple(finite_float(item) for item in value)


def _coerce_band_path_special_points(value: Any) -> Dict[str, Tuple[float, ...]]:
    if not isinstance(value, dict):
        return {}
    normalized: Dict[str, Tuple[float, ...]] = {}
    for raw_label, raw_coordinates in value.items():
        label = str(raw_label).strip()
        if not label:
            continue
        if not isinstance(raw_coordinates, (list, tuple)):
            normalized[label] = ()
            continue
        try:
            normalized[label] = tuple(float(item) for item in raw_coordinates)
        except (TypeError, ValueError):
            normalized[label] = ()
    return normalized


def task_spec_to_dict(task_spec: TaskSpec) -> Dict[str, Any]:
    payload = asdict(task_spec)
    method_name = str(task_spec.method.name or '').strip().lower()
    solver_applies = (
        task_spec.task_type == 'model_hamiltonian'
        or (task_spec.task_type == 'molecular' and method_name in ('casci', 'casscf'))
        or (
            task_spec.task_type == 'periodic'
            and str(task_spec.solver.name or '').strip().lower().replace('-', '_')
            in ('gw', 'hf_dmft', 'gw_dmft')
        )
    )
    if not solver_applies:
        payload.pop('solver', None)
    payload['schema'] = TASK_SPEC_SCHEMA
    return payload


def task_spec_from_dict(data: Optional[Dict[str, Any]], *, strict: bool = False) -> TaskSpec:
    if data is None:
        return TaskSpec()
    data = validate_dataclass_input(data, TaskSpec, strict=strict)
    if data.get('schema') is not None:
        validate_public_payload(data, expected_schema=TASK_SPEC_SCHEMA)

    system_data = _filter_known_fields(data.get('system'), ('atom', 'basis', 'unit', 'charge', 'spin', 'symmetry'))
    method_data = _filter_known_fields(data.get('method'), ('name', 'restricted', 'xc'))
    job_data = _filter_known_fields(data.get('job'), ('name',))
    analysis_data = _filter_known_fields(data.get('analysis'), ('outputs',))
    runtime_data = _filter_known_fields(
        data.get('runtime'),
        (
            'max_cycle',
            'conv_tol',
            'conv_tol_grad',
            'grid_level',
            'diis_space',
            'verbose',
            'scf_algorithm',
        ),
    )
    molecular_dynamics_data = _filter_known_fields(data.get('molecular_dynamics'), (
        'profile',
        'ensemble',
        'temperature_kelvin',
        'time_step_au',
        'steps',
        'sample_stride',
        'sample_offset',
        'velocity_seed',
        'ao_convention',
        'store_velocities',
        'store_fock',
        'store_overlap',
    ))
    initial_state_data = _filter_known_fields(data.get('initial_state'), ('mode', 'source_case_id', 'source_artifact'))
    model_hamiltonian_data = _filter_known_fields(data.get('model_hamiltonian'), ('spec', 'input_file'))
    periodic_data = _filter_known_fields(data.get('periodic'), (
        'structure_format',
        'structure_text',
        'basis',
        'pseudo',
        'kmesh',
        'kpoint_scheme',
        'kpoint_shift',
        'dimension',
        'precision',
        'ke_cutoff',
        'fft_mesh',
        'density_fitting_method',
        'density_fitting_auxbasis',
        'exxdiv',
        'smearing_method',
        'smearing_sigma',
        'smearing_fix_spin',
        'band_path_mode',
        'band_path',
        'band_path_special_points',
        'band_path_npoints',
        'band_path_reference_distance',
        'band_path_symprec',
    ))
    solver_data = _filter_known_fields(data.get('solver'), ('name', 'options'))
    orbital_processing_data = _filter_known_fields(data.get('orbital_processing'), (
        'enabled',
        'localization_method',
        'localization_scope',
        'localization_occupation_thresholds',
        'use_natural_orbitals',
        'orbital_ordering',
        'orbital_order',
        'frozen_orbital_indices',
        'continuation_policy',
    ))
    density_fitting_data = _filter_known_fields(data.get('density_fitting'), ('enabled', 'auxbasis', 'apply_to'))
    embedding_data = _filter_known_fields(data.get('embedding'), (
        'enabled',
        'provider',
        'localization_method',
        'minimal_basis',
        'include_pao',
        'correlated_orbital_indices',
        'correlated_atom_indices',
        'fragments',
        'interaction',
        'approved',
        'audit_artifact',
        'reference_artifact',
        'localized_subspace_artifact',
        'localized_hamiltonian_artifact',
    ))
    active_space_data = _filter_known_fields(data.get('active_space'), (
        'enabled',
        'selection_method',
        'ncas',
        'nelecas',
        'orbital_indices',
        'occupation_window',
        'energy_window',
        'avas_targets',
        'avas_threshold',
        'avas_minimal_basis',
        'avas_with_iao',
        'avas_openshell_option',
        'avas_ncore',
        'initial_mo_coeff',
        'target_method',
        'target_solver',
        'target_solver_options',
        'approved',
    ))
    post_cas_data = _filter_known_fields(data.get('post_cas'), ('sc_nevpt2',))
    workflow_data = _filter_known_fields(data.get('workflow'), ('modules', 'module_config', 'dependency_policy'))
    quality_gate_data = _filter_known_fields(data.get('quality_gates'), ('enabled', 'gates', 'gate_config'))

    system_data['charge'] = _coerce_int(system_data.get('charge'), 0)
    system_data['spin'] = _coerce_int(system_data.get('spin'), 0)
    if 'symmetry' in system_data:
        system_data['symmetry'] = _coerce_bool(system_data['symmetry'], False)

    if 'restricted' in method_data:
        method_data['restricted'] = _normalize_optional_bool(method_data.get('restricted'))
    if 'xc' in method_data:
        xc_value = method_data.get('xc')
        method_data['xc'] = str(xc_value).strip() if xc_value is not None and str(xc_value).strip() else None

    analysis_data['outputs'] = _normalize_outputs_value(analysis_data.get('outputs'))

    runtime_data['max_cycle'] = _coerce_int(runtime_data.get('max_cycle'), 50)
    runtime_data['verbose'] = _coerce_int(runtime_data.get('verbose'), 4)
    runtime_data['conv_tol'] = _coerce_optional_float(runtime_data.get('conv_tol'))
    runtime_data['conv_tol_grad'] = _coerce_optional_float(runtime_data.get('conv_tol_grad'))
    runtime_data['grid_level'] = _coerce_optional_int(runtime_data.get('grid_level'))
    runtime_data['diis_space'] = _coerce_optional_int(runtime_data.get('diis_space'))
    runtime_data['scf_algorithm'] = str(
        runtime_data.get('scf_algorithm') or 'standard'
    ).strip().lower().replace('-', '_')

    molecular_dynamics_data['ensemble'] = str(
        molecular_dynamics_data.get('ensemble') or 'nve'
    ).strip().lower()
    molecular_dynamics_data['temperature_kelvin'] = _coerce_float(
        molecular_dynamics_data.get('temperature_kelvin'), 300.0
    )
    molecular_dynamics_data['time_step_au'] = _coerce_float(
        molecular_dynamics_data.get('time_step_au'), 50.0
    )
    molecular_dynamics_data['steps'] = _coerce_int(
        molecular_dynamics_data.get('steps'), 100
    )
    molecular_dynamics_data['sample_stride'] = _coerce_int(
        molecular_dynamics_data.get('sample_stride'), 10
    )
    molecular_dynamics_data['sample_offset'] = _coerce_int(
        molecular_dynamics_data.get('sample_offset'), 9
    )
    molecular_dynamics_data['velocity_seed'] = _coerce_int(
        molecular_dynamics_data.get('velocity_seed'), 0
    )
    molecular_dynamics_data['ao_convention'] = str(
        molecular_dynamics_data.get('ao_convention') or 'qh9'
    ).strip().lower()
    for field_name in ('store_velocities', 'store_fock', 'store_overlap'):
        molecular_dynamics_data[field_name] = _coerce_bool(
            molecular_dynamics_data.get(field_name), True
        )

    initial_state_data['mode'] = str(initial_state_data.get('mode') or 'none').strip().lower()
    source_case_id = initial_state_data.get('source_case_id')
    initial_state_data['source_case_id'] = str(source_case_id).strip() if source_case_id is not None and str(source_case_id).strip() else None
    source_artifact = initial_state_data.get('source_artifact')
    initial_state_data['source_artifact'] = copy.deepcopy(source_artifact) if isinstance(source_artifact, dict) else {}

    if not isinstance(model_hamiltonian_data.get('spec'), dict):
        model_hamiltonian_data['spec'] = {}
    if model_hamiltonian_data.get('input_file') is not None:
        model_hamiltonian_data['input_file'] = str(model_hamiltonian_data['input_file'])

    periodic_data['structure_format'] = str(periodic_data.get('structure_format') or 'poscar').strip().lower()
    structure_text = periodic_data.get('structure_text')
    periodic_data['structure_text'] = str(structure_text) if structure_text is not None else None
    periodic_data['basis'] = str(periodic_data.get('basis') or DEFAULT_PERIODIC_BASIS_SET).strip().lower()
    periodic_data['pseudo'] = str(periodic_data.get('pseudo') or 'gth-pbe').strip().lower()
    periodic_data['kmesh'] = _coerce_positive_int_tuple(
        periodic_data.get('kmesh'),
        (0, 0, 0) if 'kmesh' in periodic_data else (1, 1, 1),
    )
    periodic_data['kpoint_scheme'] = str(
        periodic_data.get('kpoint_scheme') or 'gamma_centered'
    ).strip().lower()
    periodic_data['kpoint_shift'] = _coerce_float_tuple(
        periodic_data.get('kpoint_shift'),
        (1.0, 1.0, 1.0) if 'kpoint_shift' in periodic_data else (0.0, 0.0, 0.0),
    )
    periodic_data['dimension'] = _coerce_int(periodic_data.get('dimension'), 3)
    periodic_data['precision'] = _coerce_float(
        periodic_data.get('precision'),
        -1.0 if 'precision' in periodic_data else 1e-8,
    )
    raw_ke_cutoff = periodic_data.get('ke_cutoff')
    periodic_data['ke_cutoff'] = _coerce_optional_float(raw_ke_cutoff)
    if raw_ke_cutoff not in (None, '') and periodic_data['ke_cutoff'] is None:
        periodic_data['ke_cutoff'] = -1.0
    periodic_data['fft_mesh'] = _coerce_optional_positive_int_tuple(periodic_data.get('fft_mesh'))
    periodic_data['density_fitting_method'] = str(
        periodic_data.get('density_fitting_method') or 'fft'
    ).strip().lower()
    density_fitting_auxbasis = periodic_data.get('density_fitting_auxbasis')
    periodic_data['density_fitting_auxbasis'] = (
        str(density_fitting_auxbasis).strip()
        if density_fitting_auxbasis is not None and str(density_fitting_auxbasis).strip()
        else None
    )
    periodic_data['exxdiv'] = str(periodic_data.get('exxdiv') or 'ewald').strip().lower()
    periodic_data['smearing_method'] = str(
        periodic_data.get('smearing_method') or 'none'
    ).strip().lower()
    periodic_data['smearing_sigma'] = _coerce_optional_float(periodic_data.get('smearing_sigma'))
    periodic_data['smearing_fix_spin'] = _coerce_bool(
        periodic_data.get('smearing_fix_spin'),
        False,
    )
    periodic_data['band_path_mode'] = str(
        periodic_data.get('band_path_mode') or 'auto'
    ).strip().lower()
    raw_band_path = periodic_data.get('band_path')
    periodic_data['band_path'] = (
        str(raw_band_path).strip()
        if raw_band_path is not None and str(raw_band_path).strip()
        else None
    )
    periodic_data['band_path_special_points'] = _coerce_band_path_special_points(
        periodic_data.get('band_path_special_points')
    )
    periodic_data['band_path_npoints'] = _coerce_int(
        periodic_data.get('band_path_npoints'),
        0 if 'band_path_npoints' in periodic_data else 80,
    )
    periodic_data['band_path_reference_distance'] = _coerce_float(
        periodic_data.get('band_path_reference_distance'),
        -1.0 if 'band_path_reference_distance' in periodic_data else 0.025,
    )
    periodic_data['band_path_symprec'] = _coerce_float(
        periodic_data.get('band_path_symprec'),
        -1.0 if 'band_path_symprec' in periodic_data else 1e-5,
    )

    solver_options = solver_data.get('options')
    solver_data['options'] = (
        copy.deepcopy(solver_options)
        if isinstance(solver_options, dict)
        else {}
    )
    if (
        str(data.get('task_type') or 'molecular').strip().lower() == 'molecular'
        and str(method_data.get('name') or '').strip().lower() == 'casscf'
    ):
        solver_data['options'] = materialize_block2_state_average_weights(
            solver_data.get('name'),
            solver_data['options'],
        )

    orbital_processing_data = normalize_orbital_processing(orbital_processing_data)

    density_fitting_data['enabled'] = _coerce_bool(density_fitting_data.get('enabled'), False)
    auxbasis = density_fitting_data.get('auxbasis')
    density_fitting_data['auxbasis'] = str(auxbasis).strip() if auxbasis is not None and str(auxbasis).strip() else None
    density_fitting_data['apply_to'] = str(density_fitting_data.get('apply_to') or 'scf').strip().lower()

    embedding_data['enabled'] = _coerce_bool(embedding_data.get('enabled'), False)
    embedding_data['provider'] = str(embedding_data.get('provider') or 'libdmet').strip().lower()
    embedding_data['localization_method'] = str(
        embedding_data.get('localization_method') or 'manual'
    ).strip().lower()
    embedding_data['minimal_basis'] = str(
        embedding_data.get('minimal_basis') or 'minao'
    ).strip() or 'minao'
    embedding_data['include_pao'] = _coerce_bool(embedding_data.get('include_pao'), True)
    embedding_data['correlated_orbital_indices'] = list(dict.fromkeys(_coerce_int_list(
        embedding_data.get('correlated_orbital_indices')
    )))
    embedding_data['correlated_atom_indices'] = list(dict.fromkeys(_coerce_int_list(
        embedding_data.get('correlated_atom_indices')
    )))
    raw_fragments = embedding_data.get('fragments')
    embedding_data['fragments'] = (
        [copy.deepcopy(item) for item in raw_fragments if isinstance(item, dict)]
        if isinstance(raw_fragments, (list, tuple))
        else []
    )
    for mapping_field in (
        'interaction',
        'audit_artifact',
        'reference_artifact',
        'localized_subspace_artifact',
        'localized_hamiltonian_artifact',
    ):
        raw_mapping = embedding_data.get(mapping_field)
        embedding_data[mapping_field] = copy.deepcopy(raw_mapping) if isinstance(raw_mapping, dict) else {}
    embedding_data['approved'] = _coerce_bool(embedding_data.get('approved'), False)

    active_space_data['enabled'] = _coerce_bool(active_space_data.get('enabled'), False)
    active_space_data['selection_method'] = str(active_space_data.get('selection_method') or 'manual').strip().lower()
    active_space_data['ncas'] = _coerce_optional_int(active_space_data.get('ncas'))
    active_space_data['nelecas'] = _coerce_nelecas(active_space_data.get('nelecas'))
    active_space_data['orbital_indices'] = _coerce_orbital_indices(active_space_data.get('orbital_indices'))
    active_space_data['occupation_window'] = _coerce_float_window(active_space_data.get('occupation_window'), (0.02, 1.98))
    active_space_data['energy_window'] = _coerce_optional_float(active_space_data.get('energy_window'))
    raw_avas_targets = active_space_data.get('avas_targets')
    if isinstance(raw_avas_targets, str):
        raw_avas_targets = [item.strip() for item in raw_avas_targets.split(',') if item.strip()]
    active_space_data['avas_targets'] = [str(item).strip() for item in raw_avas_targets] if isinstance(raw_avas_targets, (list, tuple)) else []
    active_space_data['avas_threshold'] = _coerce_float(active_space_data.get('avas_threshold'), 0.2)
    active_space_data['avas_minimal_basis'] = str(active_space_data.get('avas_minimal_basis') or 'minao').strip() or 'minao'
    active_space_data['avas_with_iao'] = _coerce_bool(active_space_data.get('avas_with_iao'), False)
    active_space_data['avas_openshell_option'] = _coerce_int(active_space_data.get('avas_openshell_option'), 2)
    active_space_data['avas_ncore'] = _coerce_int(active_space_data.get('avas_ncore'), 0)
    active_space_data['initial_mo_coeff'] = copy.deepcopy(active_space_data.get('initial_mo_coeff'))
    target_method = active_space_data.get('target_method')
    active_space_data['target_method'] = (
        str(target_method).strip().lower()
        if target_method is not None and str(target_method).strip()
        else None
    )
    target_solver = active_space_data.get('target_solver')
    active_space_data['target_solver'] = (
        str(target_solver).strip().lower().replace('-', '_')
        if target_solver is not None and str(target_solver).strip()
        else None
    )
    target_solver_options = active_space_data.get('target_solver_options')
    active_space_data['target_solver_options'] = (
        copy.deepcopy(target_solver_options)
        if isinstance(target_solver_options, dict)
        else {}
    )
    if str(active_space_data.get('target_method') or '').strip().lower() == 'casscf':
        active_space_data['target_solver_options'] = materialize_block2_state_average_weights(
            active_space_data.get('target_solver'),
            active_space_data['target_solver_options'],
        )
    active_space_data['approved'] = _coerce_bool(active_space_data.get('approved'), False)

    sc_nevpt2_data = _filter_known_fields(post_cas_data.get('sc_nevpt2'), ('enabled', 'root', 'density_fit'))
    sc_nevpt2_data['enabled'] = _coerce_bool(sc_nevpt2_data.get('enabled'), False)
    sc_nevpt2_data['root'] = _coerce_int(sc_nevpt2_data.get('root'), 0)
    sc_nevpt2_data['density_fit'] = _coerce_bool(sc_nevpt2_data.get('density_fit'), True)

    raw_modules = workflow_data.get('modules')
    if isinstance(raw_modules, str):
        raw_modules = [item.strip() for item in raw_modules.split(',') if item.strip()]
    workflow_data['modules'] = list(dict.fromkeys(
        str(item).strip() for item in (raw_modules or []) if str(item).strip()
    )) if isinstance(raw_modules, (list, tuple)) else []
    raw_module_config = workflow_data.get('module_config')
    workflow_data['module_config'] = copy.deepcopy(raw_module_config) if isinstance(raw_module_config, dict) else {}
    workflow_data['dependency_policy'] = str(
        workflow_data.get('dependency_policy') or 'auto'
    ).strip().lower()

    quality_gate_data['enabled'] = _coerce_bool(quality_gate_data.get('enabled'), True)
    raw_gates = quality_gate_data.get('gates')
    if isinstance(raw_gates, str):
        raw_gates = [item.strip() for item in raw_gates.split(',') if item.strip()]
    quality_gate_data['gates'] = list(dict.fromkeys(
        str(item).strip() for item in (raw_gates or []) if str(item).strip()
    )) if isinstance(raw_gates, (list, tuple)) else []
    raw_gate_config = quality_gate_data.get('gate_config')
    quality_gate_data['gate_config'] = copy.deepcopy(raw_gate_config) if isinstance(raw_gate_config, dict) else {}

    task_type = data.get('task_type', 'molecular')
    if not isinstance(task_type, str):
        task_type = 'molecular'

    return TaskSpec(
        task_type=task_type,
        system=SystemSpec(**system_data),
        method=MethodSpec(**method_data),
        job=JobSpec(**job_data),
        analysis=AnalysisSpec(**analysis_data),
        runtime=RuntimeSpec(**runtime_data),
        molecular_dynamics=MolecularDynamicsSpec(**molecular_dynamics_data),
        initial_state=InitialStateSpec(**initial_state_data),
        model_hamiltonian=ModelHamiltonianSpec(**model_hamiltonian_data),
        periodic=PeriodicSystemSpec(**periodic_data),
        solver=SolverSpec(**solver_data),
        orbital_processing=OrbitalProcessingSpec(**orbital_processing_data),
        density_fitting=DensityFittingSpec(**density_fitting_data),
        embedding=EmbeddingPreparationSpec(**embedding_data),
        active_space=ActiveSpaceSpec(**active_space_data),
        post_cas=PostCASSpec(sc_nevpt2=SCNEVPT2Spec(**sc_nevpt2_data)),
        workflow=WorkflowModuleSpec(**workflow_data),
        quality_gates=QualityGateSpec(**quality_gate_data),
    )


__all__ = [
    'TASK_REPORT_SCHEMA',
    'TASK_SPEC_SCHEMA',
    'TaskReport',
    'TaskSpec',
    'task_spec_from_dict',
    'task_spec_to_dict',
]
