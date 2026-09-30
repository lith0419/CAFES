from __future__ import annotations

import copy
from typing import Any, Dict, List

from ..input_validation import integer_list, reject_unknown_fields
from ..lifecycle import ensure_lifecycle, transition_lifecycle

from ..contracts import (
    DEFAULT_ANALYSIS,
    DEFAULT_PERIODIC_BASIS_SET,
    SUPPORTED_ANALYSIS,
    SUPPORTED_JOBS,
    SUPPORTED_METHODS,
    SUPPORTED_MODEL_SOLVERS,
    SUPPORTED_TASK_TYPES,
    TaskSpec,
    _coerce_band_path_special_points,
    _coerce_bool,
    _coerce_float,
    _coerce_int,
    _coerce_nelecas,
    _coerce_orbital_indices,
    _coerce_optional_float,
    _coerce_optional_int,
    _coerce_positive_int_tuple,
    _coerce_float_window,
    _coerce_float_tuple,
    _coerce_optional_positive_int_tuple,
    _normalize_optional_bool,
    _normalize_outputs_value,
    task_spec_from_dict,
    task_spec_to_dict,
    normalize_orbital_processing,
)
from .model_hamiltonian.solver import (
    load_model_spec_from_file,
    normalize_solver_name,
    normalize_model_spec,
    validate_model_hamiltonian_spec,
)
from .periodic.solver import validate_periodic_task
from .state import (
    _append_errors_to_state,
    _append_validation_issue,
    append_log,
    append_message,
)
from .request_parser import (
    _extract_basis_from_text as _extract_basis_from_text,
    _extract_first_match as _extract_first_match,
    _extract_json_object as _extract_json_object,
    _extract_key_value_block as _extract_key_value_block,
    _extract_xc_from_text as _extract_xc_from_text,
    _normalize_method_name,
    parse_user_request,
)
from .validation import (
    MOLECULAR_MD_PROFILES,
    _normalize_atom_text,
    _validate_atom_format,
    _validate_basis_format,
    _validate_embedding_spec,
    _validate_initial_state_spec,
    _validate_model_solver_spec,
    _validate_molecular_dynamics_spec,
    _validate_runtime,
    _validate_strong_correlation_spec,
    _validate_xc_format,
)
from ..pyscf_i18n import join_items, t


def intent_parser(state: Dict[str, Any]) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    locale = state.get('locale', 'en')
    parsed = parse_user_request(state['user_request'])
    if state.get('task_spec') is None:
        state['task_spec'] = parsed
    else:
        merged = copy.deepcopy(state['task_spec'])
        merged.update(parsed)
        state['task_spec'] = merged
    append_log(state, 'info', 'workflow.intent_parsed', {
        'keys': sorted(state['task_spec'].keys()),
    })
    append_message(state, role='system', kind='intent', content=t(locale, 'intent_parsed'), metadata={
        'task_spec': copy.deepcopy(state['task_spec']),
    })
    return state


_DENSITY_ENABLED_ALIASES = ('density_fit', 'df')
_DENSITY_AUXBASIS_ALIASES = ('auxbasis', 'df_auxbasis', 'density_fitting_auxbasis')

_TASK_INPUT_FIELDS = _DENSITY_ENABLED_ALIASES + _DENSITY_AUXBASIS_ALIASES + (
    'active_orbitals',
    'active_space',
    'active_space_approved',
    'active_space_method',
    'analysis',
    'atom',
    'avas_minimal_basis',
    'avas_targets',
    'avas_threshold',
    'basis',
    'charge',
    'conv_tol',
    'conv_tol_grad',
    'density_fitting',
    'diis_space',
    'dimension',
    'embedding',
    'energy_window',
    'grid_level',
    'initial_state',
    'job',
    'kmesh',
    'kpoint_scheme',
    'kpoint_shift',
    'localization_method',
    'max_cycle',
    'message',
    'method',
    'model_hamiltonian',
    'model_hamiltonian_input_file',
    'molecular_dynamics',
    'ncas',
    'nelecas',
    'occupation_window',
    'orbital_indices',
    'orbital_processing',
    'outputs',
    'periodic',
    'periodic_band_path',
    'periodic_band_path_mode',
    'periodic_band_path_npoints',
    'periodic_band_path_reference_distance',
    'periodic_band_path_special_points',
    'periodic_band_path_symprec',
    'periodic_basis',
    'periodic_density_fitting_auxbasis',
    'periodic_density_fitting_method',
    'periodic_exxdiv',
    'periodic_fft_mesh',
    'periodic_ke_cutoff',
    'periodic_precision',
    'periodic_pseudo',
    'periodic_smearing_fix_spin',
    'periodic_smearing_method',
    'periodic_smearing_sigma',
    'periodic_structure_format',
    'periodic_structure_text',
    'post_cas',
    'prompt',
    'quality_gates',
    'request',
    'request_summary',
    'restricted',
    'runtime',
    'sc_nevpt2',
    'sc_nevpt2_density_fit',
    'sc_nevpt2_root',
    'scf_algorithm',
    'schema',
    'selection_method',
    'solver',
    'spin',
    'symmetry',
    'system',
    'task_type',
    'unit',
    'use_natural_orbitals',
    'verbose',
    'workflow',
    'xc',
)


def task_spec_from_partial(raw_spec: Dict[str, Any]) -> TaskSpec:
    if not isinstance(raw_spec, dict):
        raise ValueError('task_spec must be an object')
    reject_unknown_fields(raw_spec, _TASK_INPUT_FIELDS, 'task_spec')
    explicit_task_type = 'task_type' in raw_spec
    nested_keys = (
        'system',
        'method',
        'job',
        'analysis',
        'runtime',
        'molecular_dynamics',
        'initial_state',
        'model_hamiltonian',
        'periodic',
        'solver',
        'orbital_processing',
        'density_fitting',
        'embedding',
        'active_space',
        'post_cas',
        'workflow',
        'quality_gates',
    )
    for key in nested_keys:
        if key in raw_spec and not isinstance(raw_spec[key], dict) and key not in ('method', 'job', 'solver', 'orbital_processing', 'active_space', 'density_fitting'):
            raise ValueError('task_spec.{0} must be an object'.format(key))
    if any(isinstance(raw_spec.get(key), dict) for key in nested_keys):
        task_spec = task_spec_from_dict({
            key: raw_spec[key] for key in nested_keys if isinstance(raw_spec.get(key), dict)
        }, strict=True)
    else:
        task_spec = TaskSpec()

    if 'task_type' in raw_spec:
        task_spec.task_type = str(raw_spec['task_type']).strip() or task_spec.task_type
    elif isinstance(raw_spec.get('periodic'), dict):
        task_spec.task_type = 'periodic'

    if 'atom' in raw_spec:
        task_spec.system.atom = _normalize_atom_text(raw_spec['atom'])
    if 'basis' in raw_spec:
        task_spec.system.basis = raw_spec['basis']
    if 'unit' in raw_spec:
        task_spec.system.unit = raw_spec['unit']
    if 'charge' in raw_spec:
        task_spec.system.charge = _coerce_int(raw_spec['charge'], task_spec.system.charge)
    if 'spin' in raw_spec:
        task_spec.system.spin = _coerce_int(raw_spec['spin'], task_spec.system.spin)
    if 'symmetry' in raw_spec:
        task_spec.system.symmetry = _coerce_bool(raw_spec['symmetry'], task_spec.system.symmetry)

    if 'method' in raw_spec:
        method_payload = raw_spec['method']
        if not isinstance(method_payload, dict):
            task_spec.method.name = _normalize_method_name(method_payload)
    if 'xc' in raw_spec:
        xc_value = raw_spec['xc']
        task_spec.method.xc = str(xc_value).strip() if xc_value is not None and str(xc_value).strip() else None
    if 'restricted' in raw_spec:
        task_spec.method.restricted = _normalize_optional_bool(raw_spec['restricted'])
    if 'job' in raw_spec:
        job_payload = raw_spec['job']
        if not isinstance(job_payload, dict):
            task_spec.job.name = job_payload
    if 'outputs' in raw_spec:
        task_spec.analysis.outputs = _normalize_outputs_value(raw_spec['outputs'])

    if 'max_cycle' in raw_spec:
        task_spec.runtime.max_cycle = _coerce_int(raw_spec['max_cycle'], task_spec.runtime.max_cycle)
    if 'conv_tol' in raw_spec:
        task_spec.runtime.conv_tol = _coerce_optional_float(raw_spec['conv_tol'])
    if 'conv_tol_grad' in raw_spec:
        task_spec.runtime.conv_tol_grad = _coerce_optional_float(raw_spec['conv_tol_grad'])
    if 'grid_level' in raw_spec:
        task_spec.runtime.grid_level = _coerce_optional_int(raw_spec['grid_level'])
    if 'diis_space' in raw_spec:
        task_spec.runtime.diis_space = _coerce_optional_int(raw_spec['diis_space'])
    if 'verbose' in raw_spec:
        task_spec.runtime.verbose = _coerce_int(raw_spec['verbose'], task_spec.runtime.verbose)
    if 'scf_algorithm' in raw_spec:
        task_spec.runtime.scf_algorithm = str(
            raw_spec['scf_algorithm'] or 'standard'
        ).strip().lower().replace('-', '_')

    molecular_dynamics_payload = raw_spec.get('molecular_dynamics')
    if isinstance(molecular_dynamics_payload, dict):
        molecular_dynamics = task_spec.molecular_dynamics
        if molecular_dynamics_payload.get('ensemble') is not None:
            molecular_dynamics.ensemble = str(
                molecular_dynamics_payload.get('ensemble') or 'nve'
            ).strip().lower()
        for field_name in ('temperature_kelvin', 'time_step_au'):
            if molecular_dynamics_payload.get(field_name) is not None:
                setattr(
                    molecular_dynamics,
                    field_name,
                    _coerce_float(
                        molecular_dynamics_payload[field_name],
                        getattr(molecular_dynamics, field_name),
                    ),
                )
        for field_name in ('steps', 'sample_stride', 'sample_offset', 'velocity_seed'):
            if molecular_dynamics_payload.get(field_name) is not None:
                setattr(
                    molecular_dynamics,
                    field_name,
                    _coerce_int(
                        molecular_dynamics_payload[field_name],
                        getattr(molecular_dynamics, field_name),
                    ),
                )
        if molecular_dynamics_payload.get('ao_convention') is not None:
            molecular_dynamics.ao_convention = str(
                molecular_dynamics_payload.get('ao_convention') or 'qh9'
            ).strip().lower()
        for field_name in ('store_velocities', 'store_fock', 'store_overlap'):
            if field_name in molecular_dynamics_payload:
                setattr(
                    molecular_dynamics,
                    field_name,
                    _coerce_bool(
                        molecular_dynamics_payload[field_name],
                        getattr(molecular_dynamics, field_name),
                    ),
                )

    if 'initial_state' in raw_spec:
        initial_state_payload = raw_spec['initial_state']
        if isinstance(initial_state_payload, dict):
            mode = initial_state_payload.get('mode')
            if mode is not None:
                task_spec.initial_state.mode = str(mode or 'none').strip().lower()
            source_case_id = initial_state_payload.get('source_case_id')
            task_spec.initial_state.source_case_id = (
                str(source_case_id).strip()
                if source_case_id is not None and str(source_case_id).strip()
                else None
            )
            source_artifact = initial_state_payload.get('source_artifact')
            task_spec.initial_state.source_artifact = (
                copy.deepcopy(source_artifact)
                if isinstance(source_artifact, dict)
                else {}
            )

    if 'solver' in raw_spec:
        if isinstance(raw_spec['solver'], dict):
            if raw_spec['solver'].get('name'):
                task_spec.solver.name = normalize_solver_name(raw_spec['solver']['name'])
            if isinstance(raw_spec['solver'].get('options'), dict):
                task_spec.solver.options = copy.deepcopy(raw_spec['solver']['options'])
        elif isinstance(raw_spec['solver'], str):
            task_spec.solver.name = normalize_solver_name(raw_spec['solver'])

    if 'orbital_processing' in raw_spec:
        orbital_payload = raw_spec['orbital_processing']
        if isinstance(orbital_payload, dict):
            task_spec.orbital_processing.enabled = _coerce_bool(orbital_payload.get('enabled'), task_spec.orbital_processing.enabled)
            if orbital_payload.get('localization_method') is not None:
                task_spec.orbital_processing.localization_method = str(orbital_payload.get('localization_method') or 'none').strip().lower()
            if orbital_payload.get('localization_scope') is not None:
                task_spec.orbital_processing.localization_scope = str(orbital_payload.get('localization_scope') or 'analysis').strip().lower()
            if 'localization_occupation_thresholds' in orbital_payload:
                task_spec.orbital_processing.localization_occupation_thresholds = normalize_orbital_processing({
                    **vars(task_spec.orbital_processing),
                    'localization_occupation_thresholds': orbital_payload['localization_occupation_thresholds'],
                })['localization_occupation_thresholds']
            task_spec.orbital_processing.use_natural_orbitals = _coerce_bool(
                orbital_payload.get('use_natural_orbitals'),
                task_spec.orbital_processing.use_natural_orbitals,
            )
            if orbital_payload.get('orbital_ordering') is not None:
                task_spec.orbital_processing.orbital_ordering = str(orbital_payload.get('orbital_ordering') or 'canonical').strip().lower()
            if isinstance(orbital_payload.get('orbital_order'), (list, tuple)):
                task_spec.orbital_processing.orbital_order = integer_list(orbital_payload['orbital_order'], 'orbital_processing.orbital_order')
            if 'frozen_orbital_indices' in orbital_payload:
                task_spec.orbital_processing.frozen_orbital_indices = normalize_orbital_processing({
                    **vars(task_spec.orbital_processing),
                    'frozen_orbital_indices': orbital_payload['frozen_orbital_indices'],
                })['frozen_orbital_indices']
            if 'continuation_policy' in orbital_payload:
                task_spec.orbital_processing.continuation_policy = normalize_orbital_processing({
                    **vars(task_spec.orbital_processing),
                    'continuation_policy': orbital_payload['continuation_policy'],
                })['continuation_policy']
        else:
            task_spec.orbital_processing.enabled = _coerce_bool(orbital_payload, task_spec.orbital_processing.enabled)
    if 'localization_method' in raw_spec:
        task_spec.orbital_processing.localization_method = str(raw_spec['localization_method'] or 'none').strip().lower()
        task_spec.orbital_processing.enabled = task_spec.orbital_processing.localization_method not in ('', 'none')
    if 'use_natural_orbitals' in raw_spec:
        task_spec.orbital_processing.use_natural_orbitals = _coerce_bool(raw_spec['use_natural_orbitals'], task_spec.orbital_processing.use_natural_orbitals)

    if 'density_fitting' in raw_spec:
        density_payload = raw_spec['density_fitting']
        if isinstance(density_payload, dict):
            task_spec.density_fitting.enabled = _coerce_bool(density_payload.get('enabled'), task_spec.density_fitting.enabled)
            if density_payload.get('auxbasis') is not None:
                auxbasis = str(density_payload.get('auxbasis') or '').strip()
                task_spec.density_fitting.auxbasis = auxbasis or None
            if density_payload.get('apply_to') is not None:
                task_spec.density_fitting.apply_to = str(density_payload.get('apply_to') or 'scf').strip().lower()
        else:
            task_spec.density_fitting.enabled = _coerce_bool(density_payload, task_spec.density_fitting.enabled)
    for density_alias in _DENSITY_ENABLED_ALIASES:
        if density_alias in raw_spec:
            task_spec.density_fitting.enabled = _coerce_bool(raw_spec[density_alias], task_spec.density_fitting.enabled)
    for auxbasis_key in _DENSITY_AUXBASIS_ALIASES:
        if auxbasis_key in raw_spec:
            auxbasis = str(raw_spec[auxbasis_key] or '').strip()
            task_spec.density_fitting.auxbasis = auxbasis or None
            task_spec.density_fitting.enabled = True

    if 'active_space' in raw_spec:
        active_payload = raw_spec['active_space']
        if isinstance(active_payload, dict):
            task_spec.active_space.enabled = _coerce_bool(active_payload.get('enabled'), task_spec.active_space.enabled)
            if active_payload.get('selection_method') is not None:
                task_spec.active_space.selection_method = str(active_payload.get('selection_method') or 'manual').strip().lower()
            if active_payload.get('ncas') is not None:
                task_spec.active_space.ncas = _coerce_optional_int(active_payload.get('ncas'))
            if active_payload.get('nelecas') is not None:
                task_spec.active_space.nelecas = _coerce_nelecas(active_payload.get('nelecas'))
            if active_payload.get('orbital_indices') is not None:
                task_spec.active_space.orbital_indices = _coerce_orbital_indices(active_payload.get('orbital_indices'))
            if active_payload.get('occupation_window') is not None:
                task_spec.active_space.occupation_window = _coerce_float_window(active_payload.get('occupation_window'), task_spec.active_space.occupation_window)
            if active_payload.get('energy_window') is not None:
                task_spec.active_space.energy_window = _coerce_optional_float(active_payload.get('energy_window'))
            if active_payload.get('avas_targets') is not None:
                raw_targets = active_payload.get('avas_targets')
                if isinstance(raw_targets, str):
                    raw_targets = [item.strip() for item in raw_targets.split(',') if item.strip()]
                task_spec.active_space.avas_targets = [str(item).strip() for item in raw_targets] if isinstance(raw_targets, (list, tuple)) else []
            if active_payload.get('avas_threshold') is not None:
                threshold = _coerce_optional_float(active_payload.get('avas_threshold'))
                if threshold is not None:
                    task_spec.active_space.avas_threshold = threshold
            if active_payload.get('avas_minimal_basis') is not None:
                task_spec.active_space.avas_minimal_basis = str(active_payload.get('avas_minimal_basis') or 'minao').strip() or 'minao'
            if active_payload.get('avas_with_iao') is not None:
                task_spec.active_space.avas_with_iao = _coerce_bool(active_payload.get('avas_with_iao'), False)
            if active_payload.get('avas_openshell_option') is not None:
                task_spec.active_space.avas_openshell_option = _coerce_int(active_payload.get('avas_openshell_option'), 2)
            if active_payload.get('avas_ncore') is not None:
                task_spec.active_space.avas_ncore = _coerce_int(active_payload.get('avas_ncore'), 0)
            if active_payload.get('initial_mo_coeff') is not None:
                task_spec.active_space.initial_mo_coeff = copy.deepcopy(active_payload.get('initial_mo_coeff'))
            if active_payload.get('target_method') is not None:
                task_spec.active_space.target_method = _normalize_method_name(active_payload.get('target_method'))
            if active_payload.get('target_solver') is not None:
                task_spec.active_space.target_solver = normalize_solver_name(active_payload.get('target_solver'))
            if isinstance(active_payload.get('target_solver_options'), dict):
                task_spec.active_space.target_solver_options = copy.deepcopy(
                    active_payload.get('target_solver_options')
                )
            task_spec.active_space.approved = _coerce_bool(active_payload.get('approved'), task_spec.active_space.approved)
        else:
            task_spec.active_space.enabled = _coerce_bool(active_payload, task_spec.active_space.enabled)
    if 'active_space_method' in raw_spec:
        task_spec.active_space.selection_method = str(raw_spec['active_space_method'] or 'manual').strip().lower()
        task_spec.active_space.enabled = True
    if 'selection_method' in raw_spec:
        task_spec.active_space.selection_method = str(raw_spec['selection_method'] or 'manual').strip().lower()
        task_spec.active_space.enabled = True
    if 'ncas' in raw_spec:
        task_spec.active_space.ncas = _coerce_optional_int(raw_spec['ncas'])
        task_spec.active_space.enabled = True
    if 'nelecas' in raw_spec:
        task_spec.active_space.nelecas = _coerce_nelecas(raw_spec['nelecas'])
        task_spec.active_space.enabled = True
    if 'active_orbitals' in raw_spec:
        task_spec.active_space.orbital_indices = _coerce_orbital_indices(raw_spec['active_orbitals'])
        task_spec.active_space.enabled = True
    if 'orbital_indices' in raw_spec:
        task_spec.active_space.orbital_indices = _coerce_orbital_indices(raw_spec['orbital_indices'])
        task_spec.active_space.enabled = True
    if 'occupation_window' in raw_spec:
        task_spec.active_space.occupation_window = _coerce_float_window(raw_spec['occupation_window'], task_spec.active_space.occupation_window)
        task_spec.active_space.enabled = True
    if 'energy_window' in raw_spec:
        task_spec.active_space.energy_window = _coerce_optional_float(raw_spec['energy_window'])
        task_spec.active_space.enabled = True
    if 'avas_targets' in raw_spec:
        raw_targets = raw_spec['avas_targets']
        if isinstance(raw_targets, str):
            raw_targets = [item.strip() for item in raw_targets.split(',') if item.strip()]
        task_spec.active_space.avas_targets = [str(item).strip() for item in raw_targets] if isinstance(raw_targets, (list, tuple)) else []
        task_spec.active_space.enabled = True
    if 'avas_threshold' in raw_spec:
        threshold = _coerce_optional_float(raw_spec['avas_threshold'])
        if threshold is not None:
            task_spec.active_space.avas_threshold = threshold
        task_spec.active_space.enabled = True
    if 'avas_minimal_basis' in raw_spec:
        task_spec.active_space.avas_minimal_basis = str(raw_spec['avas_minimal_basis'] or 'minao').strip() or 'minao'
        task_spec.active_space.enabled = True
    if 'active_space_approved' in raw_spec:
        task_spec.active_space.approved = _coerce_bool(raw_spec['active_space_approved'], task_spec.active_space.approved)

    if 'post_cas' in raw_spec:
        post_cas_payload = raw_spec['post_cas']
        if isinstance(post_cas_payload, dict):
            sc_nevpt2_payload = post_cas_payload.get('sc_nevpt2')
            if isinstance(sc_nevpt2_payload, dict):
                task_spec.post_cas.sc_nevpt2.enabled = _coerce_bool(
                    sc_nevpt2_payload.get('enabled'),
                    task_spec.post_cas.sc_nevpt2.enabled,
                )
                if sc_nevpt2_payload.get('root') is not None:
                    task_spec.post_cas.sc_nevpt2.root = _coerce_int(sc_nevpt2_payload.get('root'), task_spec.post_cas.sc_nevpt2.root)
                if sc_nevpt2_payload.get('density_fit') is not None:
                    task_spec.post_cas.sc_nevpt2.density_fit = _coerce_bool(
                        sc_nevpt2_payload.get('density_fit'),
                        task_spec.post_cas.sc_nevpt2.density_fit,
                    )
            elif sc_nevpt2_payload is not None:
                task_spec.post_cas.sc_nevpt2.enabled = _coerce_bool(sc_nevpt2_payload, task_spec.post_cas.sc_nevpt2.enabled)
    if 'sc_nevpt2' in raw_spec:
        task_spec.post_cas.sc_nevpt2.enabled = _coerce_bool(raw_spec['sc_nevpt2'], task_spec.post_cas.sc_nevpt2.enabled)
    if 'sc_nevpt2_root' in raw_spec:
        task_spec.post_cas.sc_nevpt2.root = _coerce_int(raw_spec['sc_nevpt2_root'], task_spec.post_cas.sc_nevpt2.root)
    if 'sc_nevpt2_density_fit' in raw_spec:
        task_spec.post_cas.sc_nevpt2.density_fit = _coerce_bool(raw_spec['sc_nevpt2_density_fit'], task_spec.post_cas.sc_nevpt2.density_fit)

    if 'model_hamiltonian' in raw_spec:
        model_payload = raw_spec['model_hamiltonian']
        if isinstance(model_payload, dict):
            wrapped_spec = model_payload.get('spec')
            input_file = model_payload.get('input_file')
            is_wrapped_contract = 'spec' in model_payload or 'input_file' in model_payload
            has_wrapped_input = bool(input_file) or (
                isinstance(wrapped_spec, dict) and bool(wrapped_spec)
            )
            if is_wrapped_contract and has_wrapped_input:
                task_spec.model_hamiltonian.spec = copy.deepcopy(wrapped_spec or {})
                if input_file:
                    task_spec.model_hamiltonian.input_file = str(input_file)
            elif not is_wrapped_contract and model_payload:
                task_spec.model_hamiltonian.spec = copy.deepcopy(model_payload)
                has_wrapped_input = True
            if has_wrapped_input and not explicit_task_type:
                task_spec.task_type = 'model_hamiltonian'
    if 'model_hamiltonian_input_file' in raw_spec:
        task_spec.model_hamiltonian.input_file = str(raw_spec['model_hamiltonian_input_file']).strip()
        if not explicit_task_type:
            task_spec.task_type = 'model_hamiltonian'

    if 'periodic_structure_format' in raw_spec:
        task_spec.periodic.structure_format = str(raw_spec['periodic_structure_format'] or 'poscar').strip().lower()
        if not explicit_task_type:
            task_spec.task_type = 'periodic'
    if 'periodic_structure_text' in raw_spec:
        task_spec.periodic.structure_text = str(raw_spec['periodic_structure_text'] or '')
        if not explicit_task_type:
            task_spec.task_type = 'periodic'
    if 'periodic_basis' in raw_spec:
        task_spec.periodic.basis = str(raw_spec['periodic_basis'] or DEFAULT_PERIODIC_BASIS_SET).strip().lower()
        if not explicit_task_type:
            task_spec.task_type = 'periodic'
    if 'periodic_pseudo' in raw_spec:
        task_spec.periodic.pseudo = str(raw_spec['periodic_pseudo'] or 'gth-pbe').strip().lower()
        if not explicit_task_type:
            task_spec.task_type = 'periodic'
    if 'kmesh' in raw_spec:
        task_spec.periodic.kmesh = _coerce_positive_int_tuple(raw_spec['kmesh'], (0, 0, 0))
    if 'kpoint_scheme' in raw_spec:
        task_spec.periodic.kpoint_scheme = str(raw_spec['kpoint_scheme'] or 'gamma_centered').strip().lower()
    if 'kpoint_shift' in raw_spec:
        task_spec.periodic.kpoint_shift = _coerce_float_tuple(
            raw_spec['kpoint_shift'],
            (1.0, 1.0, 1.0),
        )
    if 'dimension' in raw_spec:
        task_spec.periodic.dimension = _coerce_int(raw_spec['dimension'], task_spec.periodic.dimension)
    if 'periodic_precision' in raw_spec:
        precision = _coerce_optional_float(raw_spec['periodic_precision'])
        task_spec.periodic.precision = precision if precision is not None else -1.0
    if 'periodic_ke_cutoff' in raw_spec:
        raw_ke_cutoff = raw_spec['periodic_ke_cutoff']
        task_spec.periodic.ke_cutoff = _coerce_optional_float(raw_ke_cutoff)
        if raw_ke_cutoff not in (None, '') and task_spec.periodic.ke_cutoff is None:
            task_spec.periodic.ke_cutoff = -1.0
    if 'periodic_fft_mesh' in raw_spec:
        task_spec.periodic.fft_mesh = _coerce_optional_positive_int_tuple(raw_spec['periodic_fft_mesh'])
    if 'periodic_density_fitting_method' in raw_spec:
        task_spec.periodic.density_fitting_method = str(
            raw_spec['periodic_density_fitting_method'] or 'fft'
        ).strip().lower()
    if 'periodic_density_fitting_auxbasis' in raw_spec:
        auxbasis = raw_spec['periodic_density_fitting_auxbasis']
        task_spec.periodic.density_fitting_auxbasis = (
            str(auxbasis).strip() if auxbasis is not None and str(auxbasis).strip() else None
        )
    if 'periodic_exxdiv' in raw_spec:
        task_spec.periodic.exxdiv = str(raw_spec['periodic_exxdiv'] or 'ewald').strip().lower()
    if 'periodic_smearing_method' in raw_spec:
        task_spec.periodic.smearing_method = str(
            raw_spec['periodic_smearing_method'] or 'none'
        ).strip().lower()
    if 'periodic_smearing_sigma' in raw_spec:
        task_spec.periodic.smearing_sigma = _coerce_optional_float(raw_spec['periodic_smearing_sigma'])
    if 'periodic_smearing_fix_spin' in raw_spec:
        task_spec.periodic.smearing_fix_spin = _coerce_bool(
            raw_spec['periodic_smearing_fix_spin'],
            task_spec.periodic.smearing_fix_spin,
        )
    if 'periodic_band_path_mode' in raw_spec:
        task_spec.periodic.band_path_mode = str(
            raw_spec['periodic_band_path_mode'] or 'auto'
        ).strip().lower()
    if 'periodic_band_path' in raw_spec:
        raw_band_path = raw_spec['periodic_band_path']
        task_spec.periodic.band_path = (
            str(raw_band_path).strip()
            if raw_band_path is not None and str(raw_band_path).strip()
            else None
        )
    if 'periodic_band_path_special_points' in raw_spec:
        task_spec.periodic.band_path_special_points = _coerce_band_path_special_points(
            raw_spec['periodic_band_path_special_points']
        )
    if 'periodic_band_path_npoints' in raw_spec:
        task_spec.periodic.band_path_npoints = _coerce_int(
            raw_spec['periodic_band_path_npoints'],
            0,
        )
    if 'periodic_band_path_reference_distance' in raw_spec:
        reference_distance = _coerce_optional_float(raw_spec['periodic_band_path_reference_distance'])
        task_spec.periodic.band_path_reference_distance = (
            reference_distance if reference_distance is not None else -1.0
        )
    if 'periodic_band_path_symprec' in raw_spec:
        symprec = _coerce_optional_float(raw_spec['periodic_band_path_symprec'])
        task_spec.periodic.band_path_symprec = symprec if symprec is not None else -1.0

    explicit_outputs = 'outputs' in raw_spec or (
        isinstance(raw_spec.get('analysis'), dict) and 'outputs' in raw_spec['analysis']
    )
    if task_spec.task_type == 'periodic' and not explicit_outputs:
        task_spec.analysis.outputs = ['energy', 'band_gap', 'fermi_energy']

    task_spec.method.name = _normalize_method_name(task_spec.method.name)
    task_spec.solver.name = normalize_solver_name(task_spec.solver.name)
    if task_spec.task_type == 'model_hamiltonian' and 'solver' not in raw_spec and task_spec.method.name in SUPPORTED_MODEL_SOLVERS:
        task_spec.solver.name = task_spec.method.name

    return task_spec


def spec_builder(state: Dict[str, Any]) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    raw_spec = state.get('task_spec') or {}
    explicit_model_solver = 'solver' in raw_spec or (
        'method' in raw_spec
        and _normalize_method_name(
            raw_spec.get('method', {}).get('name')
            if isinstance(raw_spec.get('method'), dict)
            else raw_spec.get('method')
        ) in SUPPORTED_MODEL_SOLVERS
    )
    try:
        task_spec = task_spec_from_partial(raw_spec)
    except (TypeError, ValueError) as exc:
        error = {'stage': 'validation', 'code': 'invalid_task_input', 'message': str(exc)}
        state.setdefault('errors', []).append(error)
        state['validation_errors'] = [str(exc)]
        state['lifecycle'] = transition_lifecycle(
            ensure_lifecycle(state.get('lifecycle'), 'task', entity_id=state.get('run_id')),
            'validation_failed', details={'error_count': 1},
        )
        return state
    task_spec.system.atom = _normalize_atom_text(task_spec.system.atom)
    if task_spec.task_type == 'model_hamiltonian' and task_spec.model_hamiltonian.input_file:
        # Builder output is the canonical source for mutable model Hamiltonians.
        # Inline specs may be stale UI state from an earlier preparation.
        try:
            task_spec.model_hamiltonian.spec = normalize_model_spec(
                load_model_spec_from_file(task_spec.model_hamiltonian.input_file)
            )
        except Exception as exc:
            task_spec.model_hamiltonian.spec = {
                '_load_error': str(exc),
            }
    if (
        task_spec.task_type == 'model_hamiltonian'
        and not explicit_model_solver
        and isinstance(task_spec.model_hamiltonian.spec, dict)
        and task_spec.model_hamiltonian.spec.get('solver')
    ):
        task_spec.solver.name = normalize_solver_name(task_spec.model_hamiltonian.spec['solver'])
    if (
        task_spec.task_type == 'model_hamiltonian'
        and isinstance(task_spec.model_hamiltonian.spec, dict)
        and '_load_error' not in task_spec.model_hamiltonian.spec
    ):
        task_spec.model_hamiltonian.spec['solver'] = normalize_solver_name(task_spec.solver.name)
    state['task_spec'] = task_spec_to_dict(task_spec)
    append_log(state, 'info', 'workflow.spec_built', {
        'task_type': task_spec.task_type,
        'has_atom': bool(task_spec.system.atom),
        'basis': task_spec.system.basis,
        'charge': task_spec.system.charge,
        'spin': task_spec.system.spin,
        'symmetry': task_spec.system.symmetry,
        'method': task_spec.method.name,
        'restricted': task_spec.method.restricted,
        'xc': task_spec.method.xc,
        'job': task_spec.job.name,
        'outputs': list(task_spec.analysis.outputs),
        'max_cycle': task_spec.runtime.max_cycle,
        'conv_tol': task_spec.runtime.conv_tol,
        'verbose': task_spec.runtime.verbose,
        'scf_algorithm': task_spec.runtime.scf_algorithm,
        'solver': task_spec.solver.name,
        'has_model_hamiltonian': bool(task_spec.model_hamiltonian.spec),
        'model_hamiltonian_input_file': task_spec.model_hamiltonian.input_file,
        'periodic_structure_format': task_spec.periodic.structure_format,
        'periodic_basis': task_spec.periodic.basis,
        'periodic_pseudo': task_spec.periodic.pseudo,
        'periodic_kmesh': list(task_spec.periodic.kmesh),
        'periodic_kpoint_scheme': task_spec.periodic.kpoint_scheme,
        'periodic_kpoint_shift': list(task_spec.periodic.kpoint_shift),
        'periodic_precision': task_spec.periodic.precision,
        'periodic_ke_cutoff': task_spec.periodic.ke_cutoff,
        'periodic_fft_mesh': list(task_spec.periodic.fft_mesh) if task_spec.periodic.fft_mesh else None,
        'periodic_density_fitting_method': task_spec.periodic.density_fitting_method,
        'periodic_exxdiv': task_spec.periodic.exxdiv,
        'periodic_smearing_method': task_spec.periodic.smearing_method,
        'periodic_smearing_sigma': task_spec.periodic.smearing_sigma,
        'periodic_band_path_mode': task_spec.periodic.band_path_mode,
        'periodic_band_path': task_spec.periodic.band_path,
        'periodic_band_path_special_points': task_spec.periodic.band_path_special_points,
        'periodic_band_path_npoints': task_spec.periodic.band_path_npoints,
        'periodic_band_path_reference_distance': task_spec.periodic.band_path_reference_distance,
        'periodic_band_path_symprec': task_spec.periodic.band_path_symprec,
        'orbital_processing': task_spec.orbital_processing.enabled,
        'localization_method': task_spec.orbital_processing.localization_method,
        'active_space': task_spec.active_space.enabled,
        'ncas': task_spec.active_space.ncas,
        'nelecas': task_spec.active_space.nelecas,
    })
    return state


def spec_validator(state: Dict[str, Any]) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    if any(error.get('code') == 'invalid_task_input' for error in state.get('errors', [])):
        return state
    locale = state.get('locale', 'en')
    task_spec = task_spec_from_dict(state['task_spec'])
    errors: List[Dict[str, Any]] = []
    validation_errors: List[str] = []
    questions: List[str] = []
    applied_defaults: List[Dict[str, Any]] = []

    for embedding_error in _validate_embedding_spec(task_spec):
        errors.append(embedding_error)
        validation_errors.append(embedding_error['message'])

    if task_spec.task_type not in SUPPORTED_TASK_TYPES:
        _append_validation_issue(
            errors,
            validation_errors,
            questions,
            'unsupported_task_type',
            'Unsupported task type: {0}'.format(task_spec.task_type),
            details={'task_type': task_spec.task_type},
        )

    if task_spec.task_type == 'model_hamiltonian':
        task_spec.solver.name = normalize_solver_name(task_spec.solver.name)
        model_spec = task_spec.model_hamiltonian.spec or {}
        if model_spec.get('_load_error'):
            _append_validation_issue(
                errors,
                validation_errors,
                questions,
                'model_hamiltonian_load_failed',
                'Failed to load Model Hamiltonian input: {0}'.format(model_spec['_load_error']),
                details={'input_file': task_spec.model_hamiltonian.input_file},
            )
        elif not model_spec:
            _append_validation_issue(
                errors,
                validation_errors,
                questions,
                'missing_model_hamiltonian',
                'Missing Model Hamiltonian spec or input file.',
                question='Please provide a Model Hamiltonian JSON spec or a builder-generated PySCF input file.',
            )
        if task_spec.solver.name not in SUPPORTED_MODEL_SOLVERS:
            _append_validation_issue(
                errors,
                validation_errors,
                questions,
                'unsupported_model_solver',
                'Unsupported Model Hamiltonian solver: {0}'.format(task_spec.solver.name),
                details={'solver': task_spec.solver.name},
            )
        if model_spec and not model_spec.get('_load_error'):
            for message in validate_model_hamiltonian_spec(model_spec, solver_name=task_spec.solver.name):
                _append_validation_issue(
                    errors,
                    validation_errors,
                    questions,
                    'invalid_model_hamiltonian',
                    message,
                )
            for option_error in _validate_model_solver_spec(task_spec, model_spec):
                errors.append(option_error)
                validation_errors.append(option_error['message'])
            task_spec.model_hamiltonian.spec = normalize_model_spec(model_spec)
        state['task_spec'] = task_spec_to_dict(task_spec)
        state['applied_defaults'] = copy.deepcopy(applied_defaults)
        state['errors'] = [
            error for error in state.get('errors', [])
            if error.get('stage') != 'validation'
        ]
        _append_errors_to_state(state, errors)
        state['validation_errors'] = validation_errors
        state['clarification_questions'] = [
            question for index, question in enumerate(questions)
            if question not in questions[:index]
        ]
        append_log(state, 'info', 'workflow.spec_validated', {
            'valid': not validation_errors,
            'error_count': len(validation_errors),
            'question_count': len(state['clarification_questions']),
            'applied_defaults': applied_defaults,
            'task_type': task_spec.task_type,
        })
        return state

    if task_spec.task_type == 'periodic':
        if task_spec.method.restricted is None:
            task_spec.method.restricted = task_spec.system.spin == 0
            applied_defaults.append({
                'field': 'method.restricted',
                'value': task_spec.method.restricted,
                'reason': 'inferred_from_spin',
            })
        if task_spec.method.name == 'dft' and task_spec.method.xc:
            task_spec.method.xc = str(task_spec.method.xc).strip().lower()
        for message in validate_periodic_task(task_spec):
            _append_validation_issue(
                errors,
                validation_errors,
                questions,
                'invalid_periodic_task',
                message,
            )
        for runtime_error in _validate_runtime(task_spec, locale):
            errors.append(runtime_error)
            validation_errors.append(runtime_error['message'])
        state['task_spec'] = task_spec_to_dict(task_spec)
        state['applied_defaults'] = copy.deepcopy(applied_defaults)
        state['errors'] = [
            error for error in state.get('errors', [])
            if error.get('stage') != 'validation'
        ]
        _append_errors_to_state(state, errors)
        state['validation_errors'] = validation_errors
        state['clarification_questions'] = []
        append_log(state, 'info', 'workflow.spec_validated', {
            'valid': not validation_errors,
            'error_count': len(validation_errors),
            'question_count': 0,
            'applied_defaults': applied_defaults,
            'task_type': task_spec.task_type,
        })
        return state

    if not task_spec.system.atom:
        _append_validation_issue(
            errors,
            validation_errors,
            questions,
            'missing_atom',
            t(locale, 'validation_missing_atom'),
            question=t(locale, 'question_missing_atom'),
        )
    if not task_spec.system.basis:
        _append_validation_issue(
            errors,
            validation_errors,
            questions,
            'missing_basis',
            t(locale, 'validation_missing_basis'),
            question=t(locale, 'question_missing_basis'),
        )

    atom_error = _validate_atom_format(task_spec.system.atom, locale)
    if atom_error is not None:
        errors.append(atom_error)
        validation_errors.append(atom_error['message'])
        questions.append(t(locale, 'question_atom_format'))

    basis_error = _validate_basis_format(task_spec.system.basis, locale)
    if basis_error is not None:
        errors.append(basis_error)
        validation_errors.append(basis_error['message'])
        questions.append(t(locale, 'question_basis_format'))

    if task_spec.method.name not in SUPPORTED_METHODS:
        _append_validation_issue(
            errors,
            validation_errors,
            questions,
            'unsupported_method',
            t(locale, 'validation_unsupported_method', method=task_spec.method.name, supported_methods=', '.join(SUPPORTED_METHODS)),
            question=t(locale, 'question_unsupported_method'),
            details={'method': task_spec.method.name},
        )
    if task_spec.method.name == 'dft' and not task_spec.method.xc:
        _append_validation_issue(
            errors,
            validation_errors,
            questions,
            'missing_xc',
            t(locale, 'validation_missing_xc'),
            question=t(locale, 'question_missing_xc'),
        )

    xc_error = _validate_xc_format(task_spec.method.xc, locale)
    if xc_error is not None:
        errors.append(xc_error)
        validation_errors.append(xc_error['message'])
        questions.append(t(locale, 'question_xc_format'))

    if task_spec.job.name not in SUPPORTED_JOBS:
        _append_validation_issue(
            errors,
            validation_errors,
            questions,
            'unsupported_job',
            t(locale, 'validation_unsupported_job', job=task_spec.job.name, supported_jobs=', '.join(SUPPORTED_JOBS)),
            details={'job': task_spec.job.name},
        )

    if task_spec.method.restricted is None:
        if task_spec.method.name in ('casci', 'casscf'):
            task_spec.method.restricted = True
            default_reason = 'spin_adapted_cas_default'
        else:
            task_spec.method.restricted = (task_spec.system.spin == 0)
            default_reason = 'inferred_from_spin'
        applied_defaults.append({
            'field': 'method.restricted',
            'value': task_spec.method.restricted,
            'reason': default_reason,
        })

    md_profile = MOLECULAR_MD_PROFILES.get(task_spec.molecular_dynamics.profile)
    if task_spec.job.name == 'molecular_dynamics' and md_profile is not None:
        for field_name, default_value in md_profile['runtime'].items():
            if getattr(task_spec.runtime, field_name) is None:
                setattr(task_spec.runtime, field_name, default_value)
                applied_defaults.append({
                    'field': 'runtime.{0}'.format(field_name),
                    'value': default_value,
                    'reason': 'qh9_molecular_dynamics_default',
                })
        if task_spec.analysis.outputs == list(DEFAULT_ANALYSIS):
            task_spec.analysis.outputs = ['trajectory']
            applied_defaults.append({
                'field': 'analysis.outputs',
                'value': ['trajectory'],
                'reason': 'molecular_dynamics_output_default',
            })

    normalized_outputs = []
    for output in task_spec.analysis.outputs:
        if output not in normalized_outputs:
            normalized_outputs.append(output)

    if not normalized_outputs:
        normalized_outputs = list(DEFAULT_ANALYSIS)
        applied_defaults.append({
            'field': 'analysis.outputs',
            'value': list(normalized_outputs),
            'reason': 'default_outputs_applied',
        })

    unsupported_outputs = [
        output for output in normalized_outputs
        if output not in SUPPORTED_ANALYSIS
    ]
    if unsupported_outputs:
        _append_validation_issue(
            errors,
            validation_errors,
            questions,
            'unsupported_analysis_outputs',
            t(locale, 'validation_unsupported_outputs', unsupported_outputs=', '.join(unsupported_outputs)),
            question=t(locale, 'question_supported_outputs', outputs=join_items(locale, SUPPORTED_ANALYSIS)),
            details={'unsupported_outputs': list(unsupported_outputs)},
        )

    task_spec.analysis.outputs = [
        output for output in normalized_outputs
        if output in SUPPORTED_ANALYSIS
    ]

    runtime_errors = _validate_runtime(task_spec, locale)
    for runtime_error in runtime_errors:
        errors.append(runtime_error)
        validation_errors.append(runtime_error['message'])
        if runtime_error.get('code') == 'restricted_open_shell':
            questions.append(t(locale, 'question_restricted_open_shell'))

    from .ccsd_labels import validate_ccsd_labels_request
    try:
        validate_ccsd_labels_request(task_spec)
    except ValueError as exc:
        _append_validation_issue(errors, validation_errors, questions,
                                 'invalid_ccsd_labels_request', str(exc))

    molecular_dynamics_errors = _validate_molecular_dynamics_spec(task_spec)
    for molecular_dynamics_error in molecular_dynamics_errors:
        errors.append(molecular_dynamics_error)
        validation_errors.append(molecular_dynamics_error['message'])

    initial_state_errors = _validate_initial_state_spec(task_spec)
    for initial_state_error in initial_state_errors:
        errors.append(initial_state_error)
        validation_errors.append(initial_state_error['message'])

    strong_correlation_errors = _validate_strong_correlation_spec(task_spec)
    for strong_correlation_error in strong_correlation_errors:
        errors.append(strong_correlation_error)
        validation_errors.append(strong_correlation_error['message'])
        if strong_correlation_error.get('code') in ('missing_active_space_ncas', 'missing_active_space_nelecas', 'active_space_not_approved'):
            questions.append('Please provide an approved active space with ncas and nelecas before running CASCI/CASSCF.')

    state['task_spec'] = task_spec_to_dict(task_spec)
    deduplicated_questions = []
    for question in questions:
        if question not in deduplicated_questions:
            deduplicated_questions.append(question)

    state['applied_defaults'] = copy.deepcopy(applied_defaults)
    state['errors'] = [
        error for error in state.get('errors', [])
        if error.get('stage') != 'validation'
    ]
    _append_errors_to_state(state, errors)
    state['validation_errors'] = validation_errors
    state['clarification_questions'] = deduplicated_questions
    lifecycle = ensure_lifecycle(
        state.get('lifecycle'),
        'task',
        entity_id=state.get('run_id'),
    )
    state['lifecycle'] = transition_lifecycle(
        lifecycle,
        'validation_failed' if validation_errors else 'validation_passed',
        details={
            'error_count': len(validation_errors),
            'question_count': len(deduplicated_questions),
        },
    )

    append_log(state, 'info', 'workflow.spec_validated', {
        'valid': not validation_errors,
        'error_count': len(validation_errors),
        'question_count': len(deduplicated_questions),
        'applied_defaults': applied_defaults,
    })
    return state
