from __future__ import annotations

import copy
import math
import re
from typing import Any, Dict, List, Optional

from ..registry import default_registry
from ..registry.platform import DENSITY_FITTING_AUXBASIS_RECOMMENDATIONS, density_fitting_scopes
from ..providers.block2.bond_dimension import target_sector_dimension
from ..contracts import SAFE_TOKEN_PATTERN, SUPPORTED_AUXBASIS_SETS, TaskSpec, normalize_orbital_processing
from .active_space_probe import ACTIVE_SPACE_PROBE_MODULE_ID
from .state import _make_error
from ..pyscf_i18n import t


MOLECULAR_MD_PROFILES = default_registry().capability(
    'molecular_dynamics', namespace='molecular.job',
).metadata['profiles']


def _normalize_atom_text(atom: Any) -> Any:
    if not isinstance(atom, str):
        return atom

    stripped = atom.strip()
    if not stripped:
        return stripped

    lines = [line.strip() for line in stripped.replace('\r\n', '\n').replace('\r', '\n').split('\n') if line.strip()]
    if len(lines) >= 3 and re.fullmatch(r'\d+', lines[0]):
        atom_count = int(lines[0])
        coordinate_lines = lines[2:]
        if atom_count > 0 and len(coordinate_lines) >= atom_count:
            candidate_lines = coordinate_lines[:atom_count]
            if all(len(entry.split()) >= 4 for entry in candidate_lines):
                return '\n'.join(candidate_lines)
    return stripped


def _validate_atom_format(atom: Any, locale: str) -> Optional[Dict[str, Any]]:
    if atom is None:
        return None
    if isinstance(atom, (list, tuple)):
        return _make_error(
            'validation',
            'invalid_atom_format',
            t(locale, 'validation_invalid_atom_type'),
            details={'received_type': type(atom).__name__},
        )
    if not isinstance(atom, str) or not atom.strip():
        return _make_error(
            'validation',
            'invalid_atom_format',
            t(locale, 'validation_invalid_atom_string'),
        )

    entries = [entry.strip() for entry in re.split(r';|\n', atom) if entry.strip()]
    if not entries:
        return _make_error(
            'validation',
            'invalid_atom_format',
            t(locale, 'validation_invalid_atom_empty'),
        )

    for entry in entries:
        parts = entry.split()
        if len(parts) != 4:
            return _make_error(
                'validation',
                'invalid_atom_format',
                t(locale, 'validation_invalid_atom_entry'),
                details={'entry': entry},
            )
        try:
            float(parts[1])
            float(parts[2])
            float(parts[3])
        except ValueError:
            return _make_error(
                'validation',
                'invalid_atom_format',
                t(locale, 'validation_invalid_atom_numeric'),
                details={'entry': entry},
            )
    return None


def _validate_basis_format(basis: Optional[str], locale: str) -> Optional[Dict[str, Any]]:
    if basis is None:
        return None
    if not isinstance(basis, str) or not basis.strip():
        return _make_error(
            'validation',
            'invalid_basis_format',
            t(locale, 'validation_invalid_basis_format'),
        )
    normalized_basis = basis.strip().lower()
    if not SAFE_TOKEN_PATTERN.fullmatch(normalized_basis):
        return _make_error(
            'validation',
            'invalid_basis_format',
            t(locale, 'validation_invalid_basis_chars'),
            details={'basis': basis},
        )
    return None


def _validate_xc_format(xc: Optional[str], locale: str) -> Optional[Dict[str, Any]]:
    if xc is None:
        return None
    if not isinstance(xc, str) or not xc.strip():
        return _make_error(
            'validation',
            'invalid_xc_format',
            t(locale, 'validation_invalid_xc_format'),
        )
    normalized_xc = xc.strip().lower()
    if not SAFE_TOKEN_PATTERN.fullmatch(normalized_xc):
        return _make_error(
            'validation',
            'invalid_xc_format',
            t(locale, 'validation_invalid_xc_chars'),
            details={'xc': xc},
        )
    return None


def _canonical_density_fitting_auxbasis(value: Any) -> Optional[str]:
    if value is None:
        return None
    auxbasis = str(value).strip().lower()
    if auxbasis in ('', '__off__', 'auto', 'default', 'none', 'null'):
        return None
    aliases = {
        'def2-universal-jk-fit': 'def2-universal-jkfit',
        'def2-universal-j-fit': 'def2-universal-jfit',
        'def2svp-jkfit': 'def2-svp-jkfit',
        'def2tzvp-jkfit': 'def2-tzvp-jkfit',
        'def2qzvp-jkfit': 'def2-qzvp-jkfit',
        'ccpvdz-jkfit': 'cc-pvdz-jkfit',
        'ccpvtz-jkfit': 'cc-pvtz-jkfit',
        'augccpvdz-jkfit': 'aug-cc-pvdz-jkfit',
        'heavyaug-cc-pvdz-jkfit': 'heavy-aug-cc-pvdz-jkfit',
    }
    return aliases.get(auxbasis, auxbasis)


def _basis_auxbasis_recommendations(basis: Any) -> List[str]:
    normalized_basis = str(basis or '').strip().lower()
    recommendations = DENSITY_FITTING_AUXBASIS_RECOMMENDATIONS
    if normalized_basis in recommendations:
        return list(recommendations[normalized_basis])
    if normalized_basis.startswith('def2-'):
        return ['def2-universal-jkfit', 'weigend+etb']
    if normalized_basis.startswith('cc-'):
        return ['def2-universal-jkfit', 'weigend+etb']
    return ['def2-universal-jkfit', 'weigend+etb']


def _validate_density_fitting_spec(task_spec: TaskSpec) -> List[Dict[str, Any]]:
    errors = []
    if not task_spec.density_fitting.enabled:
        task_spec.density_fitting.auxbasis = None
        return errors

    apply_to = task_spec.density_fitting.apply_to or 'scf'
    if (apply_to == 'scf' and task_spec.method.name == 'casscf'
            and task_spec.method.restricted):
        apply_to = 'scf_and_casscf'
    task_spec.density_fitting.apply_to = apply_to
    if apply_to not in density_fitting_scopes():
        errors.append(_make_error(
            'validation',
            'unsupported_density_fitting_scope',
            'Unsupported density fitting scope.',
            details={'apply_to': apply_to},
        ))
    elif apply_to == 'scf_and_casscf' and (
            task_spec.method.name != 'casscf' or not task_spec.method.restricted):
        errors.append(_make_error(
            'validation',
            'unsupported_density_fitting_method',
            'SCF+CASSCF density fitting requires restricted or ROHF CASSCF.',
            details={'method': task_spec.method.name, 'restricted': task_spec.method.restricted},
        ))

    auxbasis = _canonical_density_fitting_auxbasis(task_spec.density_fitting.auxbasis)
    task_spec.density_fitting.auxbasis = auxbasis
    if auxbasis is None:
        return errors

    if not SAFE_TOKEN_PATTERN.fullmatch(auxbasis):
        errors.append(_make_error(
            'validation',
            'invalid_density_fitting_auxbasis',
            'Density fitting auxbasis contains unsupported characters.',
            details={'auxbasis': task_spec.density_fitting.auxbasis},
        ))
        return errors

    supported_auxbasis = set(SUPPORTED_AUXBASIS_SETS)
    if auxbasis not in supported_auxbasis:
        errors.append(_make_error(
            'validation',
            'unsupported_density_fitting_auxbasis',
            'Unsupported density fitting auxbasis: {0}'.format(auxbasis),
            details={'auxbasis': auxbasis, 'supported_auxbasis': list(SUPPORTED_AUXBASIS_SETS)},
        ))
        return errors

    recommended = _basis_auxbasis_recommendations(task_spec.system.basis)
    if auxbasis not in recommended:
        errors.append(_make_error(
            'validation',
            'inconsistent_density_fitting_auxbasis',
            'Density fitting auxbasis {0} is not recommended for basis {1}.'.format(auxbasis, task_spec.system.basis),
            details={
                'basis': task_spec.system.basis,
                'auxbasis': auxbasis,
                'recommended_auxbasis': recommended,
            },
        ))
    return errors


def _validate_embedding_spec(task_spec: TaskSpec) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    embedding = task_spec.embedding
    if not embedding.enabled:
        return errors
    registry = default_registry()
    if not registry.capability_is_allowed(
        embedding.provider, namespace='embedding.backend'
    ):
        errors.append(_make_error(
            'validation',
            'unsupported_embedding_provider',
            'Unsupported embedding provider: {0}'.format(embedding.provider),
            details={'provider': embedding.provider},
        ))
    if not registry.capability_is_allowed(
        embedding.localization_method, namespace='embedding.localization'
    ):
        errors.append(_make_error(
            'validation',
            'unsupported_embedding_localization',
            'Unsupported embedding localization method: {0}'.format(embedding.localization_method),
            details={'localization_method': embedding.localization_method},
        ))
    for field_name, values in (
        ('correlated_orbital_indices', embedding.correlated_orbital_indices),
        ('correlated_atom_indices', embedding.correlated_atom_indices),
    ):
        if any(index < 0 for index in values):
            errors.append(_make_error(
                'validation',
                'invalid_embedding_indices',
                'Embedding {0} must be non-negative.'.format(field_name),
                details={'field': field_name, 'values': list(values)},
            ))
    if not embedding.approved:
        return errors

    required_refs = (
        ('audit_artifact', embedding.audit_artifact, 'correlated_subspace_audit'),
        ('reference_artifact', embedding.reference_artifact, 'embedding_reference'),
        ('localized_subspace_artifact', embedding.localized_subspace_artifact, 'localized_subspace'),
        ('localized_hamiltonian_artifact', embedding.localized_hamiltonian_artifact, 'localized_hamiltonian'),
    )
    for field_name, ref, expected_kind in required_refs:
        kind = str(ref.get('kind') or '') if isinstance(ref, dict) else ''
        path = str(ref.get('path') or '').strip() if isinstance(ref, dict) else ''
        if kind != expected_kind or not path:
            errors.append(_make_error(
                'validation',
                'invalid_embedding_approval_evidence',
                'Approved embedding requests require a registered {0} artifact.'.format(expected_kind),
                details={'field': field_name, 'expected_kind': expected_kind},
            ))
    return errors


def _validate_active_orbital_indices(task_spec: TaskSpec) -> List[Dict[str, Any]]:
    errors = []
    orbital_indices = task_spec.active_space.orbital_indices
    initial_mo_coeff = task_spec.active_space.initial_mo_coeff
    if not orbital_indices:
        return errors

    if isinstance(orbital_indices, dict):
        unexpected_keys = sorted(str(key) for key in orbital_indices if key not in ('alpha', 'beta'))
        if unexpected_keys:
            errors.append(_make_error(
                'validation',
                'invalid_spin_resolved_active_orbital_keys',
                'Spin-resolved active orbital indices may only contain alpha and beta keys.',
                details={'orbital_indices': copy.deepcopy(orbital_indices), 'unexpected_keys': unexpected_keys},
            ))
        if task_spec.method.restricted is not False:
            errors.append(_make_error(
                'validation',
                'spin_resolved_active_orbitals_require_unrestricted',
                'Spin-resolved active orbital indices require restricted=false.',
                details={'orbital_indices': copy.deepcopy(orbital_indices)},
            ))
        for spin_label in ('alpha', 'beta'):
            indices = orbital_indices.get(spin_label)
            if not indices:
                errors.append(_make_error(
                    'validation',
                    'missing_spin_resolved_active_orbitals',
                    'Spin-resolved active orbital indices require both alpha and beta lists.',
                    details={'missing_spin': spin_label, 'orbital_indices': copy.deepcopy(orbital_indices)},
                ))
                continue
            if any(index < 0 for index in indices):
                errors.append(_make_error(
                    'validation',
                    'invalid_active_orbital_index',
                    'Active orbital indices must be zero-based nonnegative integers.',
                    details={'spin': spin_label, 'orbital_indices': list(indices)},
                ))
            if initial_mo_coeff is None and task_spec.active_space.ncas and len(indices) != task_spec.active_space.ncas:
                errors.append(_make_error(
                    'validation',
                    'active_orbital_count_mismatch',
                    'Each spin channel active orbital count must match active_space.ncas.',
                    details={
                        'spin': spin_label,
                        'ncas': task_spec.active_space.ncas,
                        'orbital_indices': list(indices),
                    },
                ))
        return errors

    if any(index < 0 for index in orbital_indices):
        errors.append(_make_error(
            'validation',
            'invalid_active_orbital_index',
            'Active orbital indices must be zero-based nonnegative integers.',
            details={'orbital_indices': list(orbital_indices)},
        ))
    if initial_mo_coeff is None and task_spec.active_space.ncas and len(orbital_indices) != task_spec.active_space.ncas:
        errors.append(_make_error(
            'validation',
            'active_orbital_count_mismatch',
            'The number of active orbital indices must match active_space.ncas.',
            details={
                'ncas': task_spec.active_space.ncas,
                'orbital_indices': list(orbital_indices),
            },
        ))
    return errors


def _validate_runtime(task_spec: TaskSpec, locale: str) -> List[Dict[str, Any]]:
    errors = []
    if task_spec.system.spin < 0:
        errors.append(_make_error(
            'validation',
            'invalid_spin',
            t(locale, 'validation_invalid_spin'),
            details={'spin': task_spec.system.spin},
        ))
    if task_spec.runtime.max_cycle <= 0:
        errors.append(_make_error(
            'validation',
            'invalid_max_cycle',
            t(locale, 'validation_invalid_max_cycle'),
            details={'max_cycle': task_spec.runtime.max_cycle},
        ))
    if task_spec.runtime.conv_tol is not None and task_spec.runtime.conv_tol <= 0:
        errors.append(_make_error(
            'validation',
            'invalid_conv_tol',
            t(locale, 'validation_invalid_conv_tol'),
            details={'conv_tol': task_spec.runtime.conv_tol},
        ))
    if task_spec.runtime.conv_tol_grad is not None and task_spec.runtime.conv_tol_grad <= 0:
        errors.append(_make_error(
            'validation',
            'invalid_conv_tol_grad',
            'runtime.conv_tol_grad must be positive when provided.',
            details={'conv_tol_grad': task_spec.runtime.conv_tol_grad},
        ))
    if task_spec.runtime.grid_level is not None and task_spec.runtime.grid_level < 0:
        errors.append(_make_error(
            'validation',
            'invalid_grid_level',
            'runtime.grid_level must be non-negative when provided.',
            details={'grid_level': task_spec.runtime.grid_level},
        ))
    if task_spec.runtime.diis_space is not None and task_spec.runtime.diis_space <= 0:
        errors.append(_make_error(
            'validation',
            'invalid_diis_space',
            'runtime.diis_space must be positive when provided.',
            details={'diis_space': task_spec.runtime.diis_space},
        ))
    if task_spec.runtime.verbose < 0:
        errors.append(_make_error(
            'validation',
            'invalid_verbose',
            t(locale, 'validation_invalid_verbose'),
            details={'verbose': task_spec.runtime.verbose},
        ))
    if task_spec.runtime.scf_algorithm not in ('standard', 'newton'):
        errors.append(_make_error(
            'validation',
            'invalid_scf_algorithm',
            'runtime.scf_algorithm must be standard or newton.',
            details={'scf_algorithm': task_spec.runtime.scf_algorithm},
        ))
    if task_spec.method.restricted and task_spec.system.spin != 0 and task_spec.method.name not in ('casci', 'casscf'):
        errors.append(_make_error(
            'validation',
            'restricted_open_shell',
            t(locale, 'validation_restricted_open_shell'),
            details={
                'restricted': task_spec.method.restricted,
                'spin': task_spec.system.spin,
            },
        ))
    return errors


def _validate_molecular_dynamics_spec(task_spec: TaskSpec) -> List[Dict[str, Any]]:
    if task_spec.job.name != 'molecular_dynamics':
        return []

    errors: List[Dict[str, Any]] = []
    molecular_dynamics = task_spec.molecular_dynamics
    profile = MOLECULAR_MD_PROFILES.get(molecular_dynamics.profile)
    if profile is None:
        errors.append(_make_error(
            'validation', 'md_profile_required',
            'Select molecular_dynamics.profile explicitly from {0}; generic molecular dynamics is not yet supported.'.format(
                ', '.join(MOLECULAR_MD_PROFILES),
            ),
        ))
    checks = (
        (molecular_dynamics.temperature_kelvin > 0, 'invalid_md_temperature', 'molecular_dynamics.temperature_kelvin must be positive.'),
        (molecular_dynamics.time_step_au > 0, 'invalid_md_time_step', 'molecular_dynamics.time_step_au must be positive.'),
        (molecular_dynamics.steps > 0, 'invalid_md_steps', 'molecular_dynamics.steps must be positive.'),
        (molecular_dynamics.sample_stride > 0, 'invalid_md_sample_stride', 'molecular_dynamics.sample_stride must be positive.'),
        (molecular_dynamics.sample_offset >= 0, 'invalid_md_sample_offset', 'molecular_dynamics.sample_offset must be non-negative.'),
        (molecular_dynamics.sample_offset < molecular_dynamics.steps, 'invalid_md_sample_offset', 'molecular_dynamics.sample_offset must be smaller than molecular_dynamics.steps.'),
    )
    for valid, code, message in checks:
        if not valid:
            errors.append(_make_error('validation', code, message))
    if molecular_dynamics.ensemble != 'nve':
        errors.append(_make_error(
            'validation',
            'unsupported_md_ensemble',
            'The current molecular-dynamics provider supports ensemble="nve" only.',
            details={'ensemble': molecular_dynamics.ensemble, 'supported_ensembles': ['nve']},
        ))
    if molecular_dynamics.ao_convention != 'qh9':
        errors.append(_make_error(
            'validation',
            'unsupported_md_ao_convention',
            'The current Hamiltonian trajectory provider supports ao_convention="qh9" only.',
            details={'ao_convention': molecular_dynamics.ao_convention},
        ))
    if task_spec.method.name != 'dft' or not task_spec.method.restricted:
        errors.append(_make_error(
            'validation',
            'unsupported_md_method',
            'QH9-compatible molecular dynamics currently requires restricted DFT.',
            details={
                'method': task_spec.method.name,
                'restricted': task_spec.method.restricted,
            },
        ))
    if str(task_spec.method.xc or '').strip().lower() != 'b3lyp':
        errors.append(_make_error(
            'validation',
            'unsupported_md_xc',
            'QH9-compatible molecular dynamics currently requires xc="b3lyp".',
            details={'xc': task_spec.method.xc},
        ))
    if str(task_spec.system.basis or '').strip().lower() != 'def2-svp':
        errors.append(_make_error(
            'validation',
            'unsupported_md_basis',
            'QH9-compatible molecular dynamics currently requires basis="def2-svp".',
            details={'basis': task_spec.system.basis},
        ))
    if not molecular_dynamics.store_fock or not molecular_dynamics.store_overlap:
        errors.append(_make_error(
            'validation',
            'missing_md_hamiltonian_outputs',
            'QH9-compatible molecular dynamics requires both Fock and overlap matrices.',
        ))
    if task_spec.system.charge != 0 or task_spec.system.spin != 0:
        errors.append(_make_error(
            'validation',
            'unsupported_md_charge_or_spin',
            'The QH9 trajectory slice currently supports neutral closed-shell molecules only.',
            details={'charge': task_spec.system.charge, 'spin': task_spec.system.spin},
        ))
    if task_spec.system.symmetry:
        errors.append(_make_error(
            'validation',
            'unsupported_md_symmetry',
            'QH9-compatible molecular dynamics requires system.symmetry=false.',
        ))
    if task_spec.density_fitting.enabled:
        errors.append(_make_error(
            'validation',
            'unsupported_md_density_fitting',
            'QH9-compatible molecular dynamics does not enable density fitting.',
        ))
    if task_spec.runtime.scf_algorithm != 'standard':
        errors.append(_make_error(
            'validation',
            'unsupported_md_scf_algorithm',
            'QH9-compatible molecular dynamics requires runtime.scf_algorithm="standard".',
        ))
    if task_spec.initial_state.mode != 'none':
        errors.append(_make_error(
            'validation',
            'unsupported_md_initial_state',
            'Molecular-dynamics trajectories do not currently accept initial_state restarts.',
        ))
    if task_spec.orbital_processing.enabled or task_spec.active_space.enabled:
        errors.append(_make_error(
            'validation',
            'unsupported_md_workflow_options',
            'QH9-compatible molecular dynamics does not run orbital-processing or active-space modules.',
        ))
    atom_symbols = {
        entry.split()[0].strip().capitalize()
        for entry in re.split(r';|\n', str(task_spec.system.atom or ''))
        if entry.strip() and entry.split()
    }
    unsupported_symbols = sorted(atom_symbols.difference(('H', 'C', 'N', 'O', 'F')))
    if unsupported_symbols:
        errors.append(_make_error(
            'validation',
            'unsupported_md_elements',
            'QH9-compatible molecular dynamics supports H, C, N, O, and F only.',
            details={'unsupported_elements': unsupported_symbols},
        ))
    for field_name, expected in (profile['runtime'] if profile else {}).items():
        observed = getattr(task_spec.runtime, field_name)
        if observed is None or not math.isclose(
            float(observed),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-18,
        ):
            errors.append(_make_error(
                'validation',
                'incompatible_qh9_runtime',
                'The {0} molecular-dynamics profile requires runtime.{1}={2}.'.format(
                    molecular_dynamics.profile,
                    field_name,
                    expected,
                ),
                details={
                    'field': 'runtime.{0}'.format(field_name),
                    'expected': expected,
                    'received': observed,
                },
            ))
    return errors


def _validate_initial_state_spec(task_spec: TaskSpec) -> List[Dict[str, Any]]:
    """Validate only the portable reference; numerical compatibility is runtime-specific."""
    initial_state = task_spec.initial_state
    mode = str(initial_state.mode or 'none').strip().lower()
    initial_state.mode = mode
    if mode == 'none':
        initial_state.source_case_id = None
        initial_state.source_artifact = {}
        return []
    if mode != 'projected_1rdm':
        return [_make_error(
            'validation',
            'unsupported_initial_state_mode',
            'Unsupported initial_state.mode: {0}.'.format(mode),
            details={'mode': mode, 'supported_modes': ['none', 'projected_1rdm']},
        )]
    if task_spec.task_type != 'molecular':
        return [_make_error(
            'validation',
            'initial_state_task_type_mismatch',
            'Projected 1RDM restart is currently supported for molecular tasks only.',
            details={'task_type': task_spec.task_type},
        )]
    artifact = initial_state.source_artifact if isinstance(initial_state.source_artifact, dict) else {}
    if artifact.get('kind') != 'one_particle_state' or not str(artifact.get('path') or '').strip():
        return [_make_error(
            'validation',
            'invalid_initial_state_artifact',
            'Projected 1RDM restart requires a one_particle_state artifact reference with a path.',
        )]
    return []


def _validate_model_solver_spec(task_spec: TaskSpec, model_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    errors = []
    solver_name = str(task_spec.solver.name or '').strip().lower().replace('-', '_')
    options = task_spec.solver.options if isinstance(task_spec.solver.options, dict) else {}
    if options.get('restart_geometry_policy') == 'transport':
        errors.append(_make_error('validation', 'invalid_mps_transport',
                                  'Geometry MPS transport requires molecular CASSCF.'))
    if solver_name == 'block2_dmrg':
        from ..providers.block2.config import normalize_block2_options
        try:
            normalize_block2_options(options)
        except (TypeError, ValueError) as exc:
            errors.append(_make_error('validation', 'invalid_block2_options', str(exc)))
    if solver_name == 'dmet':
        from ..providers.libdmet import validate_dmet_model_request  # pylint: disable=import-outside-toplevel

        request_errors, _configuration = validate_dmet_model_request(model_spec, options)
        errors.extend(
            _make_error(
                'validation',
                'invalid_dmet_model_request',
                message,
                details={'solver': 'dmet'},
            )
            for message in request_errors
        )
        if 'excited_states' in task_spec.analysis.outputs or 'nroots' in options:
            errors.append(_make_error(
                'validation',
                'dmet_roots_not_supported',
                'The current DMET result contract is ground-state only; remove nroots and excited_states.',
                details={'solver': 'dmet'},
            ))
        return errors
    requests_roots = 'excited_states' in task_spec.analysis.outputs or 'nroots' in options
    if requests_roots and solver_name not in ('fci', 'block2_dmrg'):
        errors.append(_make_error(
            'validation',
            'model_roots_require_eigensolver',
            'Low-lying roots are supported only by FCI and block2 DMRG model solvers.',
            details={'solver': solver_name, 'supported_solvers': ['fci', 'block2_dmrg']},
        ))
        return errors
    raw_nroots = options.get('nroots')
    if raw_nroots is None:
        return errors
    try:
        nroots = int(raw_nroots)
        integer_text = str(raw_nroots).strip() in (str(nroots), '{0}.0'.format(nroots))
    except (TypeError, ValueError):
        nroots = 0
        integer_text = False
    if isinstance(raw_nroots, bool) or nroots <= 0 or not integer_text:
        errors.append(_make_error(
            'validation',
            'invalid_model_nroots',
            'solver.options.nroots must be a positive integer.',
            details={'nroots': raw_nroots},
        ))
        return errors
    if solver_name == 'fci':
        sites = model_spec.get('sites') if isinstance(model_spec, dict) else None
        nelec = model_spec.get('nelec') if isinstance(model_spec, dict) else None
        if isinstance(sites, list) and isinstance(nelec, (list, tuple)) and len(nelec) == 2:
            norb = len(sites)
            nalpha, nbeta = (int(value) for value in nelec)
            if 0 <= nalpha <= norb and 0 <= nbeta <= norb:
                root_capacity = math.comb(norb, nalpha) * math.comb(norb, nbeta)
                if nroots > root_capacity:
                    errors.append(_make_error(
                        'validation',
                        'model_nroots_exceeds_spin_sector',
                        'FCI nroots exceeds the number of states in the requested particle/spin sector.',
                        details={
                            'nroots': nroots,
                            'spin_sector_capacity': root_capacity,
                            'norb': norb,
                            'nelec': [nalpha, nbeta],
                        },
                    ))
    return errors


def _validate_strong_correlation_spec(task_spec: TaskSpec) -> List[Dict[str, Any]]:
    errors = []
    solver = str(task_spec.solver.name or '').lower().replace('-', '_')
    if solver in ('block2', 'dmrg', 'block2_dmrg'):
        from ..providers.block2.config import normalize_block2_options
        try:
            normalize_block2_options(task_spec.solver.options)
        except (TypeError, ValueError) as exc:
            errors.append(_make_error('validation', 'invalid_block2_options', str(exc)))
    active_space_solver = str(task_spec.solver.name or 'fci').strip().lower().replace('-', '_')
    if active_space_solver in ('dmrg', 'block2'):
        active_space_solver = 'block2_dmrg'
    if task_spec.solver.options.get('restart_geometry_policy') == 'transport':
        options = task_spec.solver.options
        if (task_spec.task_type != 'molecular' or task_spec.method.name != 'casscf'
                or not task_spec.method.restricted or active_space_solver != 'block2_dmrg'
                or not options.get('restart_manifest') or options.get('orbital_restart_manifest')
                or not options.get('restart_required', True)
                or task_spec.orbital_processing.continuation_policy != 'project_all'):
            errors.append(_make_error(
                'validation', 'invalid_mps_transport',
                'MPS transport requires restricted molecular block2 CASSCF, a required joint '
                'restart_manifest, and project_all orbital continuation.',
            ))
    requested_block2_outputs = sorted(set(task_spec.analysis.outputs).intersection({
        'entanglement_diagnostics',
        'symmetry_analysis',
    }))
    if requested_block2_outputs and active_space_solver != 'block2_dmrg':
        errors.append(_make_error(
            'validation',
            'block2_output_requires_block2_solver',
            'The requested tensor-network outputs require solver=block2_dmrg.',
            details={'outputs': requested_block2_outputs, 'solver': active_space_solver},
        ))
    requests_state_roots = (
        'excited_states' in task_spec.analysis.outputs
        or 'nroots' in task_spec.solver.options
    )
    state_capable_method = task_spec.method.name in ('fci', 'casci', 'casscf')
    if requests_state_roots and not state_capable_method:
        errors.append(_make_error(
            'validation',
            'molecular_roots_require_fci_solver',
            'Molecular low-energy roots require FCI, CASCI, or CASSCF with an FCI or block2 active-space solver.',
            details={'method': task_spec.method.name},
        ))

    requested_nroots = 2 if 'excited_states' in task_spec.analysis.outputs else 1
    raw_nroots = task_spec.solver.options.get('nroots', requested_nroots)
    nroots_valid = True
    try:
        requested_nroots = int(raw_nroots)
        integer_text = str(raw_nroots).strip() in (
            str(requested_nroots),
            '{0}.0'.format(requested_nroots),
        )
    except (TypeError, ValueError):
        requested_nroots = 0
        integer_text = False
    if (
        requests_state_roots
        and (isinstance(raw_nroots, bool) or requested_nroots <= 0 or not integer_text)
    ):
        nroots_valid = False
        errors.append(_make_error(
            'validation',
            (
                'invalid_block2_dmrg_nroots'
                if active_space_solver == 'block2_dmrg'
                else 'invalid_molecular_nroots'
            ),
            'solver.options.nroots must be a positive integer.',
            details={'nroots': raw_nroots},
        ))

    if requests_state_roots and nroots_valid and task_spec.method.name in ('casci', 'casscf'):
        ncas_value = task_spec.active_space.ncas
        nelecas_value = task_spec.active_space.nelecas
        if ncas_value and nelecas_value is not None:
            if isinstance(nelecas_value, tuple) and len(nelecas_value) == 2:
                nalpha, nbeta = (int(value) for value in nelecas_value)
            elif isinstance(nelecas_value, int):
                nalpha = (int(nelecas_value) + int(task_spec.system.spin)) // 2
                nbeta = int(nelecas_value) - nalpha
            else:
                nalpha = nbeta = -1
            ncas = int(ncas_value)
            if 0 <= nbeta <= nalpha <= ncas:
                root_capacity = (
                    target_sector_dimension(ncas, nalpha, nbeta, 'su2')
                    if active_space_solver == 'block2_dmrg'
                    else math.comb(ncas, nalpha) * math.comb(ncas, nbeta)
                )
                if requested_nroots > root_capacity:
                    errors.append(_make_error(
                        'validation',
                        (
                            'block2_nroots_exceeds_spin_sector'
                            if active_space_solver == 'block2_dmrg'
                            else 'molecular_nroots_exceeds_spin_sector'
                        ),
                        'nroots exceeds the number of states in the requested active-space particle/spin sector.',
                        details={
                            'nroots': requested_nroots,
                            'spin_sector_capacity': root_capacity,
                            'ncas': ncas,
                            'nelecas': [nalpha, nbeta],
                            'solver': active_space_solver,
                        },
                    ))
    raw_weights = task_spec.solver.options.get('state_average_weights')
    if raw_weights is not None:
        if nroots_valid and requested_nroots <= 1:
            errors.append(_make_error(
                'validation',
                'state_average_requires_multiple_roots',
                'state_average_weights are only valid when nroots is greater than one.',
                details={
                    'nroots': requested_nroots,
                    'state_average_weights': raw_weights,
                },
            ))
            raw_weights = None
    if raw_weights is not None:
        try:
            weights = [float(value) for value in raw_weights]
        except (TypeError, ValueError):
            weights = []
        if (
            not nroots_valid
            or len(weights) != requested_nroots
            or any(value < 0.0 for value in weights)
            or abs(sum(weights) - 1.0) > 1.0e-8
        ):
            errors.append(_make_error(
                'validation',
                (
                    'invalid_block2_state_average_weights'
                    if active_space_solver == 'block2_dmrg'
                    else 'invalid_state_average_weights'
                ),
                'state_average_weights must contain one nonnegative value per root and sum to one.',
                details={
                    'nroots': requested_nroots,
                    'state_average_weights': raw_weights,
                },
            ))
    if task_spec.solver.options.get('restart_manifest') and active_space_solver != 'block2_dmrg':
        errors.append(_make_error(
            'validation',
            'block2_restart_requires_block2_solver',
            'An MPS restart manifest can only be used with solver=block2_dmrg.',
        ))
    localization_method = (task_spec.orbital_processing.localization_method or 'none').strip().lower()
    if task_spec.orbital_processing.continuation_policy != 'project_all':
        if (task_spec.orbital_processing.continuation_policy != 'target_scf_core'
                or task_spec.method.name != 'casscf' or not task_spec.method.restricted
                or active_space_solver != 'block2_dmrg'
                or not task_spec.solver.options.get('orbital_restart_manifest')
                or task_spec.solver.options.get('restart_manifest')):
            errors.append(_make_error(
                'validation', 'invalid_orbital_continuation_policy',
                'target_scf_core currently requires spin-adapted block2 CASSCF with an orbital-only restart and fresh MPS.',
            ))
    if task_spec.orbital_processing.frozen_orbital_indices:
        if task_spec.method.name != 'casscf' or not task_spec.method.restricted:
            errors.append(_make_error(
                'validation', 'frozen_orbitals_require_spin_adapted_casscf',
                'Frozen orbital optimization requires restricted/ROHF CASSCF.',
            ))
        try:
            normalize_orbital_processing(vars(task_spec.orbital_processing))
        except ValueError as exc:
            errors.append(_make_error('validation', 'invalid_frozen_orbitals', str(exc)))
    supported_localization_methods = ('none', 'boys', 'pipek_mezey', 'pm')
    if localization_method not in supported_localization_methods:
        errors.append(_make_error(
            'validation',
            'unsupported_localization_method',
            'Unsupported localization method: {0}'.format(localization_method),
            details={'supported_methods': list(supported_localization_methods)},
        ))
    localization_scope = (task_spec.orbital_processing.localization_scope or 'analysis').strip().lower()
    if task_spec.orbital_processing.localization_occupation_thresholds:
        try:
            normalize_orbital_processing(vars(task_spec.orbital_processing))
        except ValueError as exc:
            errors.append(_make_error('validation', 'invalid_localization_occupation_thresholds', str(exc)))
    if localization_scope not in ('analysis', 'active_space'):
        errors.append(_make_error(
            'validation',
            'unsupported_localization_scope',
            'Orbital localization scope must be analysis or active_space.',
            details={'supported_scopes': ['analysis', 'active_space']},
        ))
    orbital_ordering = (task_spec.orbital_processing.orbital_ordering or 'canonical').strip().lower()
    active_space_probe = ACTIVE_SPACE_PROBE_MODULE_ID in set(
        task_spec.workflow.modules or []
    )
    if orbital_ordering not in ('canonical', 'fiedler', 'manual'):
        errors.append(_make_error(
            'validation',
            'unsupported_orbital_ordering',
            'Orbital ordering must be canonical, fiedler, or manual.',
            details={'supported_orderings': ['canonical', 'fiedler', 'manual']},
        ))
    orbital_order = list(task_spec.orbital_processing.orbital_order or [])
    if orbital_ordering == 'manual' and not active_space_probe:
        if not orbital_order:
            errors.append(_make_error(
                'validation',
                'manual_orbital_order_required',
                'Manual orbital ordering requires a permutation in orbital_order.',
            ))
        elif len(set(orbital_order)) != len(orbital_order) or any(value < 0 for value in orbital_order):
            errors.append(_make_error(
                'validation',
                'invalid_manual_orbital_order',
                'Manual orbital_order must contain unique nonnegative indices.',
            ))
        elif task_spec.active_space.ncas is not None and sorted(orbital_order) != list(range(int(task_spec.active_space.ncas))):
            errors.append(_make_error(
                'validation',
                'manual_orbital_order_size_mismatch',
                'Manual orbital_order must be a complete zero-based permutation of the active orbitals.',
                details={'ncas': int(task_spec.active_space.ncas)},
            ))
    elif orbital_ordering != 'manual' and orbital_order:
        errors.append(_make_error(
            'validation',
            'unused_manual_orbital_order',
            'orbital_order is only valid when orbital_ordering is manual.',
        ))
    block2_solver = str(task_spec.solver.name or '').strip().lower().replace('-', '_') in ('block2_dmrg', 'block2', 'dmrg')
    if orbital_ordering != 'canonical' and not block2_solver and not active_space_probe:
        errors.append(_make_error(
            'validation',
            'orbital_ordering_requires_block2',
            'Fiedler and manual orbital ordering currently require the block2 DMRG solver.',
        ))
    if localization_scope == 'active_space' and localization_method not in ('none', ''):
        if (task_spec.method.name not in ('casci', 'casscf') or not block2_solver) and not active_space_probe:
            errors.append(_make_error(
                'validation',
                'active_space_localization_requires_block2_casci',
                'Active-space localization currently requires a block2 DMRG-CASCI/CASSCF task.',
            ))

    if task_spec.density_fitting.enabled:
        errors.extend(_validate_density_fitting_spec(task_spec))

    if task_spec.method.name in ('casci', 'casscf'):
        task_spec.active_space.enabled = True
        if active_space_solver not in ('fci', 'block2_dmrg'):
            errors.append(_make_error(
                'validation',
                'unsupported_active_space_solver',
                'Unsupported active-space solver: {0}.'.format(task_spec.solver.name),
                details={'supported_solvers': ['fci', 'block2_dmrg']},
            ))
        if active_space_solver == 'block2_dmrg':
            if task_spec.method.restricted is False:
                errors.append(_make_error(
                    'validation',
                    'block2_dmrg_unrestricted_not_supported',
                    'block2 DMRG-CASCI/CASSCF currently requires a restricted or ROHF spin-adapted reference.',
                ))
            if task_spec.post_cas.sc_nevpt2.enabled:
                errors.append(_make_error(
                    'validation',
                    'block2_dmrg_nevpt2_not_supported',
                    'SC-NEVPT2 is not available for the block2 DMRG active-space provider yet.',
                ))
            nelecas = task_spec.active_space.nelecas
            if isinstance(nelecas, tuple) and len(nelecas) == 2:
                active_spin = int(nelecas[0]) - int(nelecas[1])
                if active_spin != int(task_spec.system.spin):
                    errors.append(_make_error(
                        'validation',
                        'block2_dmrg_active_spin_mismatch',
                        'block2 DMRG-CASCI/CASSCF requires active-space spin to match system.spin because inactive core orbitals are paired.',
                        details={'active_spin': active_spin, 'system_spin': task_spec.system.spin},
                    ))
        if not task_spec.active_space.ncas or task_spec.active_space.ncas <= 0:
            errors.append(_make_error(
                'validation',
                'missing_active_space_ncas',
                'CASCI/CASSCF require active_space.ncas.',
            ))
        if task_spec.active_space.nelecas is None:
            errors.append(_make_error(
                'validation',
                'missing_active_space_nelecas',
                'CASCI/CASSCF require active_space.nelecas.',
            ))
        if not task_spec.active_space.approved:
            errors.append(_make_error(
                'validation',
                'active_space_not_approved',
                'CASCI/CASSCF require active_space.approved=true after user review.',
            ))
    elif active_space_solver == 'block2_dmrg' and task_spec.task_type == 'molecular':
        errors.append(_make_error(
            'validation',
            'block2_dmrg_requires_cas_method',
            'Molecular block2 DMRG requires method=casci or casscf with an approved active space.',
            details={'method': task_spec.method.name},
        ))
    if task_spec.active_space.enabled:
        if task_spec.active_space.ncas is not None and task_spec.active_space.ncas <= 0:
            errors.append(_make_error(
                'validation',
                'invalid_active_space_ncas',
                'active_space.ncas must be a positive integer.',
                details={'ncas': task_spec.active_space.ncas},
            ))
        if task_spec.active_space.nelecas is not None:
            nelecas = task_spec.active_space.nelecas
            if isinstance(nelecas, tuple):
                invalid = len(nelecas) != 2 or any(item < 0 for item in nelecas)
            else:
                invalid = not isinstance(nelecas, int) or nelecas < 0
            if invalid:
                errors.append(_make_error(
                    'validation',
                    'invalid_active_space_nelecas',
                    'active_space.nelecas must be a nonnegative integer or a two-item electron tuple.',
                    details={'nelecas': nelecas},
                ))
        errors.extend(_validate_active_orbital_indices(task_spec))
        if task_spec.active_space.selection_method not in ('manual', 'occupation_window', 'energy_window', 'avas'):
            errors.append(_make_error(
                'validation',
                'unsupported_active_space_selection_method',
                'Unsupported active-space selection method: {0}'.format(task_spec.active_space.selection_method),
                details={'supported_methods': ['manual', 'occupation_window', 'energy_window', 'avas']},
            ))
        if task_spec.active_space.selection_method == 'avas':
            targets = task_spec.active_space.avas_targets
            if not isinstance(targets, list) or not [item for item in targets if str(item).strip()]:
                errors.append(_make_error(
                    'validation',
                    'missing_avas_targets',
                    'AVAS active-space selection requires active_space.avas_targets (AO labels or atom:/fragment: selectors).',
                ))
            threshold = task_spec.active_space.avas_threshold
            if threshold <= 0.0 or threshold > 1.0:
                errors.append(_make_error(
                    'validation',
                    'invalid_avas_threshold',
                    'active_space.avas_threshold must be in the interval (0, 1].',
                    details={'avas_threshold': threshold},
                ))
            if task_spec.active_space.avas_ncore < 0:
                errors.append(_make_error(
                    'validation',
                    'invalid_avas_ncore',
                    'active_space.avas_ncore must be a nonnegative integer.',
                    details={'avas_ncore': task_spec.active_space.avas_ncore},
                ))
    if task_spec.post_cas.sc_nevpt2.enabled:
        if task_spec.method.name not in ('casci', 'casscf'):
            errors.append(_make_error(
                'validation',
                'sc_nevpt2_requires_cas',
                'SC-NEVPT2 requires method=casci or method=casscf.',
                details={'method': task_spec.method.name},
            ))
        if task_spec.method.restricted is False:
            errors.append(_make_error(
                'validation',
                'sc_nevpt2_unrestricted_not_supported',
                'SC-NEVPT2 currently requires a spin-adapted restricted/ROHF-style CAS reference; unrestricted CAS is not supported for this post-CAS correction.',
                details={'restricted': task_spec.method.restricted},
            ))
        if task_spec.post_cas.sc_nevpt2.root < 0:
            errors.append(_make_error(
                'validation',
                'invalid_sc_nevpt2_root',
                'SC-NEVPT2 root must be a zero-based nonnegative integer.',
                details={'root': task_spec.post_cas.sc_nevpt2.root},
            ))
        elif task_spec.post_cas.sc_nevpt2.root > 0:
            errors.append(_make_error(
                'validation',
                'unsupported_sc_nevpt2_excited_root',
                'SC-NEVPT2 excited-state roots require a multi-state or state-averaged CAS workflow; the current backend supports root=0 only.',
                details={'root': task_spec.post_cas.sc_nevpt2.root, 'supported_roots': [0]},
            ))
    return errors
