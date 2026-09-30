from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ...input_validation import finite_float, integer, reject_unknown_fields, validate_dataclass_input



_BLAS_THREAD_ENVIRONMENT_VARIABLES = (
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
)

_MEBIBYTE = 1024 ** 2
_GIBIBYTE = 1024 ** 3
_BLOCK2_SLURM_MEMORY_FRACTION = 0.75

# Public block2 diagnostics stop at the ground-state 2-particle RDM.  In
# particular, orbital mutual information needs local two-orbital density
# matrices whose NPDM construction reaches fourth order.
MAX_NPDM_ORDER = 2
MUTUAL_INFORMATION_REQUIRED_NPDM_ORDER = 4


def _positive_int_list(value: Any, field: str) -> List[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError('block2 {0} must be a nonempty list.'.format(field))
    normalized = [integer(item, 'block2.' + field) for item in value]
    if any(item <= 0 for item in normalized):
        raise ValueError('block2 {0} values must be positive.'.format(field))
    return normalized


def _nonnegative_float_list(value: Any, field: str) -> List[float]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError('block2 {0} must be a nonempty list.'.format(field))
    normalized = [finite_float(item, 'block2.' + field) for item in value]
    if any(item < 0 for item in normalized):
        raise ValueError('block2 {0} values must be nonnegative.'.format(field))
    return normalized


@dataclass(frozen=True)
class Block2DMRGConfig:
    preset: str
    bond_dimensions: List[int]
    final_bond_dimension: Optional[int]
    noises: List[float]
    davidson_thresholds: List[float]
    sweeps: int
    energy_tolerance: float
    discarded_weight_tolerance: float
    adaptive_schedule: bool
    max_adaptive_stages: int
    max_bond_dimension: int
    bond_dimension_growth_factor: float
    adaptive_sweeps: int
    adaptive_noise: float
    bond_dimension_planning: bool
    estimate_energy_error: bool
    entanglement_active_space_review: bool
    entanglement_entropy_threshold: float
    entanglement_mutual_information_threshold: float
    entanglement_occupation_tolerance: float
    active_space_expansion_step: int
    active_space_max_orbitals: int
    cutoff: float
    n_threads: int
    n_mkl_threads: int
    stack_memory_bytes: int
    iprint: int
    seed: int
    save_mps: bool
    compute_1rdm: bool
    compute_2rdm: bool
    compute_entanglement: bool
    compute_mutual_information: bool
    compute_bipartite_entanglement: bool
    compute_symmetry_analysis: bool
    nroots: int
    excited_state_mode: str
    state_average_weights: List[float]
    rdm_root_scope: str
    rdm1_root_scope: str
    rdm2_root_scope: str
    orbital_restart_manifest: Any
    orbital_restart_provenance: Dict[str, Any]
    restart_manifest: Any
    restart_provenance: Dict[str, Any]
    restart_tag: str
    restart_required: bool
    restart_geometry_policy: str
    restart_min_active_overlap: float
    symmetry: str
    mpo_algorithm: str
    integral_cutoff: Optional[float]
    orbital_ordering: str
    orbital_order: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_block2_options(options: Any = None, *, compute_2rdm: bool = False) -> Block2DMRGConfig:
    if options is not None and not isinstance(options, dict):
        raise ValueError('block2 solver.options must be an object')
    payload = dict(options or {})
    reject_unknown_fields(payload, {item.name for item in fields(Block2DMRGConfig)} | {
        'stack_memory_gb',
    }, 'block2 solver.options')
    payload = validate_dataclass_input(payload, Block2DMRGConfig, 'block2 solver.options', strict=False)
    if 'stack_memory_gb' in payload:
        payload['stack_memory_gb'] = finite_float(payload['stack_memory_gb'], 'block2.stack_memory_gb')
    preset = str(payload.get('preset') or 'balanced').strip().lower()
    from ...registry import default_registry
    presets = default_registry().capability('block2_dmrg', namespace='molecular.active_space_solver').metadata['presets']
    if preset not in presets:
        raise ValueError('Unsupported block2 preset: {0}.'.format(preset))
    defaults = presets[preset]
    bond_dimensions = _positive_int_list(
        payload.get('bond_dimensions', defaults['bond_dimensions']),
        'bond_dimensions',
    )
    final_bond_dimension = payload.get('final_bond_dimension')
    if final_bond_dimension is not None:
        final_bond_dimension = int(final_bond_dimension)
        if final_bond_dimension <= 0:
            raise ValueError('block2 final_bond_dimension must be positive.')
    sweeps = int(payload.get('sweeps', len(bond_dimensions)))
    if sweeps <= 0:
        raise ValueError('block2 sweeps must be positive.')
    noises = _nonnegative_float_list(payload.get('noises', defaults['noises']), 'noises')
    thresholds = _nonnegative_float_list(
        payload.get('davidson_thresholds', defaults['davidson_thresholds']),
        'davidson_thresholds',
    )
    energy_tolerance = float(payload.get('energy_tolerance', 1e-8))
    discarded_weight_tolerance = float(payload.get('discarded_weight_tolerance', 1e-6))
    adaptive_schedule = bool(payload.get('adaptive_schedule', True))
    max_adaptive_stages = int(payload.get('max_adaptive_stages', 2))
    if max_adaptive_stages < 0:
        raise ValueError('block2 max_adaptive_stages must be nonnegative.')
    bond_dimension_growth_factor = float(payload.get('bond_dimension_growth_factor', 2.0))
    if bond_dimension_growth_factor <= 1.0:
        raise ValueError('block2 bond_dimension_growth_factor must be greater than one.')
    max_bond_dimension = int(payload.get(
        'max_bond_dimension',
        max(bond_dimensions) * (2 ** max_adaptive_stages),
    ))
    if max_bond_dimension < max(bond_dimensions):
        raise ValueError('block2 max_bond_dimension cannot be smaller than the initial schedule.')
    adaptive_sweeps = int(payload.get('adaptive_sweeps', max(4, min(8, sweeps))))
    if adaptive_sweeps <= 0:
        raise ValueError('block2 adaptive_sweeps must be positive.')
    adaptive_noise = float(payload.get('adaptive_noise', max(max(noises), 1.0e-5)))
    if adaptive_noise < 0.0:
        raise ValueError('block2 adaptive_noise must be nonnegative.')
    entanglement_entropy_threshold = float(payload.get('entanglement_entropy_threshold', 0.10))
    entanglement_mutual_information_threshold = float(
        payload.get('entanglement_mutual_information_threshold', 0.02)
    )
    entanglement_occupation_tolerance = float(
        payload.get('entanglement_occupation_tolerance', 0.02)
    )
    if (
        entanglement_entropy_threshold < 0.0
        or entanglement_mutual_information_threshold < 0.0
        or not 0.0 <= entanglement_occupation_tolerance < 1.0
    ):
        raise ValueError('block2 entanglement active-space thresholds are outside their valid ranges.')
    active_space_expansion_step = int(payload.get('active_space_expansion_step', 1))
    active_space_max_orbitals = int(payload.get('active_space_max_orbitals', 64))
    if active_space_expansion_step <= 0 or active_space_max_orbitals <= 0:
        raise ValueError('block2 active-space expansion controls must be positive.')
    cutoff = float(payload.get('cutoff', 1e-20))
    if energy_tolerance <= 0 or discarded_weight_tolerance < 0 or cutoff < 0:
        raise ValueError('block2 convergence tolerances must be positive or zero where appropriate.')
    n_threads = int(payload.get('n_threads', 1))
    n_mkl_threads = int(payload.get('n_mkl_threads', 1))
    if n_threads <= 0 or n_mkl_threads <= 0 or n_threads % n_mkl_threads:
        raise ValueError('block2 n_mkl_threads must be a positive factor of n_threads.')
    stack_memory_bytes = int(payload.get('stack_memory_bytes', int(float(payload.get('stack_memory_gb', 1.0)) * 1024 ** 3)))
    if stack_memory_bytes <= 0:
        raise ValueError('block2 stack memory must be positive.')
    symmetry = str(payload.get('symmetry') or 'auto').strip().lower()
    if symmetry not in ('auto', 'su2', 'sz'):
        raise ValueError('block2 symmetry must be auto, su2, or sz.')
    mpo_algorithm = str(payload.get('mpo_algorithm') or 'conventional').strip().lower()
    if mpo_algorithm not in ('fast_bipartite', 'conventional'):
        raise ValueError('block2 mpo_algorithm must be fast_bipartite or conventional.')
    integral_cutoff = payload.get('integral_cutoff')
    if integral_cutoff is not None:
        integral_cutoff = finite_float(integral_cutoff, 'block2.integral_cutoff')
        if integral_cutoff < 0:
            raise ValueError('block2 integral_cutoff must be nonnegative.')
    orbital_ordering = str(payload.get('orbital_ordering') or 'canonical').strip().lower()
    if orbital_ordering not in ('canonical', 'fiedler', 'manual'):
        raise ValueError('block2 orbital_ordering must be canonical, fiedler, or manual.')
    raw_orbital_order = payload.get('orbital_order')
    if raw_orbital_order is not None and not isinstance(raw_orbital_order, (list, tuple)):
        raise ValueError('block2 orbital_order must be a list.')
    orbital_order = (
        [integer(value, 'block2.orbital_order') for value in raw_orbital_order]
        if isinstance(raw_orbital_order, (list, tuple))
        else []
    )
    if orbital_ordering == 'manual':
        if not orbital_order:
            raise ValueError('block2 manual orbital ordering requires orbital_order.')
        if len(set(orbital_order)) != len(orbital_order) or any(value < 0 for value in orbital_order):
            raise ValueError('block2 orbital_order must contain unique nonnegative indices.')
    elif orbital_order:
        raise ValueError('block2 orbital_order is only valid for manual orbital ordering.')
    raw_nroots = payload.get('nroots')
    nroots = int(1 if raw_nroots is None else raw_nroots)
    if nroots <= 0:
        raise ValueError('block2 nroots must be positive.')
    excited_state_mode = (
        str(payload.get('excited_state_mode') or 'state_averaged').strip().lower()
        if nroots > 1
        else 'state_specific'
    )
    if nroots > 1 and excited_state_mode != 'state_averaged':
        raise ValueError('The first block2 excited-state integration supports state_averaged mode only.')
    raw_state_average_weights = payload.get('state_average_weights')
    if nroots == 1:
        if raw_state_average_weights is not None and raw_state_average_weights != []:
            if (not isinstance(raw_state_average_weights, (list, tuple))
                    or len(raw_state_average_weights) != 1
                    or finite_float(raw_state_average_weights[0], 'block2.state_average_weights') != 1.0):
                raise ValueError('Single-root block2 state_average_weights must be empty or [1.0].')
        state_average_weights = []
    elif raw_state_average_weights is None:
        state_average_weights = [1.0 / nroots] * nroots
    elif isinstance(raw_state_average_weights, (list, tuple)):
        state_average_weights = [finite_float(value, 'block2.state_average_weights') for value in raw_state_average_weights]
    else:
        raise ValueError('block2 state_average_weights must be a list of root weights.')
    if nroots > 1 and len(state_average_weights) != nroots:
        raise ValueError('block2 state_average_weights must contain one value per requested root.')
    if nroots > 1 and any(value < 0.0 for value in state_average_weights):
        raise ValueError('block2 state_average_weights must be nonnegative.')
    if nroots > 1 and abs(sum(state_average_weights) - 1.0) > 1.0e-8:
        raise ValueError('block2 state_average_weights must sum to one.')
    rdm_root_scope = str(payload.get('rdm_root_scope') or 'ground_state').strip().lower()
    if rdm_root_scope not in ('ground_state', 'all_states'):
        raise ValueError('block2 rdm_root_scope must be ground_state or all_states.')
    rdm1_root_scope = str(payload.get('rdm1_root_scope') or rdm_root_scope).strip().lower()
    rdm2_root_scope = str(payload.get('rdm2_root_scope') or rdm_root_scope).strip().lower()
    if rdm1_root_scope not in ('ground_state', 'all_states'):
        raise ValueError('block2 rdm1_root_scope must be ground_state or all_states.')
    if rdm2_root_scope not in ('ground_state', 'all_states'):
        raise ValueError('block2 rdm2_root_scope must be ground_state or all_states.')
    restart_manifest = copy_restart_manifest(payload.get('restart_manifest'))
    restart_geometry_policy = str(payload.get('restart_geometry_policy') or 'same_geometry')
    if restart_geometry_policy not in ('same_geometry', 'transport'):
        raise ValueError('block2 restart_geometry_policy must be same_geometry or transport.')
    restart_min_active_overlap = finite_float(
        payload.get('restart_min_active_overlap', 0.9), 'block2.restart_min_active_overlap')
    if not 0 < restart_min_active_overlap <= 1:
        raise ValueError('block2 restart_min_active_overlap must be in (0, 1].')
    restart_tag = str(payload.get('restart_tag') or 'GS').strip()
    if not restart_tag:
        raise ValueError('block2 restart_tag must be nonempty.')
    compute_entanglement = bool(payload.get('compute_entanglement', False))
    compute_mutual_information = bool(payload.get('compute_mutual_information', False))
    compute_2rdm_effective = bool(
        payload.get('compute_2rdm', compute_2rdm)
        or compute_entanglement
        or compute_mutual_information
    )
    for field in ('seed', 'iprint'):
        if payload.get(field, 0) < 0:
            raise ValueError('block2 {0} must be nonnegative.'.format(field))
    return Block2DMRGConfig(
        preset=preset,
        bond_dimensions=bond_dimensions,
        final_bond_dimension=final_bond_dimension,
        noises=noises,
        davidson_thresholds=thresholds,
        sweeps=sweeps,
        energy_tolerance=energy_tolerance,
        discarded_weight_tolerance=discarded_weight_tolerance,
        adaptive_schedule=adaptive_schedule,
        max_adaptive_stages=max_adaptive_stages,
        max_bond_dimension=max_bond_dimension,
        bond_dimension_growth_factor=bond_dimension_growth_factor,
        adaptive_sweeps=adaptive_sweeps,
        adaptive_noise=adaptive_noise,
        bond_dimension_planning=bool(payload.get('bond_dimension_planning', True)),
        estimate_energy_error=bool(payload.get('estimate_energy_error', True)),
        entanglement_active_space_review=bool(payload.get('entanglement_active_space_review', True)),
        entanglement_entropy_threshold=entanglement_entropy_threshold,
        entanglement_mutual_information_threshold=entanglement_mutual_information_threshold,
        entanglement_occupation_tolerance=entanglement_occupation_tolerance,
        active_space_expansion_step=active_space_expansion_step,
        active_space_max_orbitals=active_space_max_orbitals,
        cutoff=cutoff,
        n_threads=n_threads,
        n_mkl_threads=n_mkl_threads,
        stack_memory_bytes=stack_memory_bytes,
        iprint=int(payload.get('iprint', 0)),
        seed=int(payload.get('seed', 1234)),
        save_mps=bool(payload.get('save_mps', True)),
        compute_1rdm=bool(payload.get('compute_1rdm', True)),
        compute_2rdm=compute_2rdm_effective,
        compute_entanglement=compute_entanglement,
        compute_mutual_information=compute_mutual_information,
        compute_bipartite_entanglement=bool(payload.get('compute_bipartite_entanglement', False)),
        compute_symmetry_analysis=bool(payload.get('compute_symmetry_analysis', False)),
        nroots=nroots,
        excited_state_mode=excited_state_mode,
        state_average_weights=state_average_weights,
        rdm_root_scope=rdm_root_scope,
        rdm1_root_scope=rdm1_root_scope,
        rdm2_root_scope=rdm2_root_scope,
        orbital_restart_manifest=payload.get('orbital_restart_manifest'),
        orbital_restart_provenance=dict(payload.get('orbital_restart_provenance') or {}),
        restart_manifest=restart_manifest,
        restart_provenance=dict(payload.get('restart_provenance') or {}),
        restart_tag=restart_tag,
        restart_required=bool(payload.get('restart_required', True)),
        restart_geometry_policy=restart_geometry_policy,
        restart_min_active_overlap=restart_min_active_overlap,
        symmetry=symmetry,
        mpo_algorithm=mpo_algorithm,
        integral_cutoff=integral_cutoff,
        orbital_ordering=orbital_ordering,
        orbital_order=orbital_order,
    )


def normalized_block2_provider_options(options: Any = None, *, include_defaults: bool = True) -> Dict[str, Any]:
    """Normalize portable options without resolving execution-host threads."""

    if options is not None and not isinstance(options, dict):
        raise ValueError('block2 options must be an object.')
    payload = dict(options or {})
    normalized = normalize_block2_options(payload).to_dict()
    if not include_defaults:
        keys = set(payload)
        if 'stack_memory_gb' in keys:
            keys.remove('stack_memory_gb')
            keys.add('stack_memory_bytes')
        return {key: normalized[key] for key in keys}
    for field in ('n_threads', 'n_mkl_threads'):
        if field not in payload:
            normalized.pop(field, None)
    if 'stack_memory_bytes' not in payload and 'stack_memory_gb' not in payload:
        normalized.pop('stack_memory_bytes', None)
    return normalized


def _positive_thread_value(value: Any) -> Optional[int]:
    text = str(value or '').strip()
    if not text:
        return None
    # OpenMP permits comma-separated thread counts for nested parallel regions.
    text = text.split(',', 1)[0].strip()
    try:
        normalized = int(text)
    except (TypeError, ValueError):
        return None
    return normalized if normalized > 0 else None


def _pyscf_thread_count() -> Optional[int]:
    try:
        from pyscf import lib  # pylint: disable=import-outside-toplevel

        return _positive_thread_value(lib.num_threads())
    except (ImportError, RuntimeError, TypeError, ValueError):
        return None


def _slurm_memory_bytes(value: Any) -> Optional[int]:
    """Parse a Slurm memory value, whose unit is MiB when no suffix is present."""

    text = str(value or '').strip().upper()
    if not text:
        return None
    multipliers = {
        'K': _MEBIBYTE / 1024.0,
        'M': _MEBIBYTE,
        'G': _GIBIBYTE,
        'T': 1024 * _GIBIBYTE,
    }
    suffix = text[-1]
    multiplier = _MEBIBYTE
    if suffix in multipliers:
        multiplier = multipliers[suffix]
        text = text[:-1].strip()
    try:
        normalized = float(text)
    except (TypeError, ValueError):
        return None
    if normalized <= 0.0:
        return None
    return int(normalized * multiplier)


def _slurm_allocation_memory(
    environment: Mapping[str, str],
    *,
    effective_threads: int,
) -> Tuple[Optional[int], Optional[str]]:
    per_node = _slurm_memory_bytes(environment.get('SLURM_MEM_PER_NODE'))
    if per_node is not None:
        return per_node, 'SLURM_MEM_PER_NODE'
    per_cpu = _slurm_memory_bytes(environment.get('SLURM_MEM_PER_CPU'))
    if per_cpu is None:
        return None, None
    allocated_cpus = _positive_thread_value(environment.get('SLURM_CPUS_PER_TASK'))
    return per_cpu * int(allocated_cpus or effective_threads), 'SLURM_MEM_PER_CPU'


def resolve_block2_runtime_options(
    options: Any = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
    pyscf_thread_reader: Optional[Callable[[], Optional[int]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Map the execution host's CPU and memory allocation into block2 options."""

    if options is not None and not isinstance(options, dict):
        raise ValueError('block2 options must be an object.')
    payload = dict(options or {})
    environment = os.environ if environ is None else environ
    explicit_threads = _positive_thread_value(payload.get('n_threads'))
    source = 'solver_options' if explicit_threads is not None else None
    source_value = explicit_threads

    if explicit_threads is None:
        candidate = _positive_thread_value(environment.get('SLURM_CPUS_PER_TASK'))
        if candidate is not None:
            source = 'SLURM_CPUS_PER_TASK'
            source_value = candidate

    if explicit_threads is None and source_value is None:
        reader = pyscf_thread_reader or _pyscf_thread_count
        candidate = _positive_thread_value(reader())
        if candidate is not None:
            source = 'pyscf.lib.num_threads'
            source_value = candidate

    if explicit_threads is None and source_value is None:
        for variable in _BLAS_THREAD_ENVIRONMENT_VARIABLES:
            candidate = _positive_thread_value(environment.get(variable))
            if candidate is not None:
                source = variable
                source_value = candidate
                break

    effective_threads = int(source_value or 1)
    explicit_mkl_threads = _positive_thread_value(payload.get('n_mkl_threads'))
    effective_mkl_threads = int(explicit_mkl_threads or 1)
    payload['n_threads'] = effective_threads
    payload['n_mkl_threads'] = effective_mkl_threads
    explicit_stack_memory = (
        'stack_memory_bytes' in payload or 'stack_memory_gb' in payload
    )
    allocation_memory_bytes, allocation_memory_source = _slurm_allocation_memory(
        environment,
        effective_threads=effective_threads,
    )
    if not explicit_stack_memory and allocation_memory_bytes is not None:
        payload['stack_memory_bytes'] = max(
            1,
            int(allocation_memory_bytes * _BLOCK2_SLURM_MEMORY_FRACTION),
        )
    # Reuse the standard validator after runtime defaults have been resolved.
    normalized = normalize_block2_options(payload)
    stack_memory_source = (
        'solver_options'
        if explicit_stack_memory
        else (allocation_memory_source or 'default')
    )
    return payload, {
        'requested_n_threads': explicit_threads,
        'requested_n_mkl_threads': explicit_mkl_threads,
        'effective_n_threads': effective_threads,
        'effective_n_mkl_threads': effective_mkl_threads,
        'n_threads_source': source or 'default',
        'nested_mkl': effective_mkl_threads > 1,
        'requested_stack_memory_bytes': (
            normalized.stack_memory_bytes if explicit_stack_memory else None
        ),
        'effective_stack_memory_bytes': normalized.stack_memory_bytes,
        'stack_memory_source': stack_memory_source,
        'slurm_memory_allocation_bytes': allocation_memory_bytes,
        'stack_memory_fraction': (
            _BLOCK2_SLURM_MEMORY_FRACTION
            if allocation_memory_source and not explicit_stack_memory
            else None
        ),
        'reserved_memory_bytes': (
            allocation_memory_bytes - normalized.stack_memory_bytes
            if allocation_memory_bytes is not None
            and not explicit_stack_memory
            else None
        ),
    }


def copy_restart_manifest(value: Any) -> Any:
    if value in (None, ''):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return dict(value)
    raise ValueError('block2 restart_manifest must be a manifest object or JSON file path.')


def block2_options_for_orbital_processing(options: Any, orbital_processing: Any) -> Dict[str, Any]:
    """Apply task orbital settings before provider defaults are materialized."""
    payload = dict(options or {})
    if orbital_processing is not None:
        payload.setdefault('orbital_ordering', str(orbital_processing.orbital_ordering or 'canonical'))
        if orbital_processing.orbital_order:
            payload.setdefault('orbital_order', list(orbital_processing.orbital_order))
    return payload


def block2_options_for_outputs(options: Any, outputs: Any) -> Dict[str, Any]:
    """Translate registered requested outputs into provider computation flags."""

    if options is not None and not isinstance(options, dict):
        raise ValueError('block2 options must be an object.')
    payload = dict(options or {})
    requested = {str(item).strip().lower() for item in (outputs or ())}
    if 'entanglement_diagnostics' in requested:
        payload.setdefault('compute_entanglement', True)
        payload.setdefault('compute_mutual_information', False)
        payload.setdefault('compute_bipartite_entanglement', True)
    if 'symmetry_analysis' in requested:
        payload.setdefault('compute_symmetry_analysis', True)
    if requested.intersection({'excited_states', 'gap'}):
        payload.setdefault('nroots', 2)
        # State energies are root-resolved; RDM-based diagnostics remain
        # explicitly ground-state quantities.
        payload.setdefault('rdm1_root_scope', 'ground_state')
        payload.setdefault('rdm2_root_scope', 'ground_state')
    return payload
