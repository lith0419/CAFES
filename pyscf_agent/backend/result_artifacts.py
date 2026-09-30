from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Dict, List

from .artifacts import (
    compact_structured_results,
    get_run_dir,
    preview_text,
    register_existing_artifact,
    retry_artifact_filename,
    serialize_npz_arrays,
    write_binary_artifact,
    write_result_json_artifact,
    write_json_artifact,
    write_text_artifact,
)
from .state import append_log
from .script_helpers import _compose_raw_stdout
from .correlation.strong import (
    active_space_audit_table,
    active_space_summary_table,
    orbital_summary_table,
)
from .periodic.solver import normalized_periodic_poscar
from ..contracts import task_spec_to_dict
from ..artifacts.arrays import externalize_orbital_guesses, array_references
from .artifacts import register_artifact_reference
from .one_particle_state import one_particle_state_summary, serialize_one_particle_state


def _record_artifact_failure(state, results, kind, exc):
    evidence = {'kind': kind, 'status': 'unavailable', 'reason': '{0}: {1}'.format(type(exc).__name__, exc)}
    results.setdefault('artifact_errors', []).append(evidence)
    append_log(state, 'warning', 'artifact.write_failed', evidence)


def _model_energy_levels_table(levels: List[Any], energy_unit: str) -> str:
    lines = ['index\tenergy\tunit']
    for index, value in enumerate(levels):
        lines.append('{0}\t{1:.16g}\t{2}'.format(index, float(value), energy_unit))
    return '\n'.join(lines) + '\n'


def _bloch_bands_table(payload: Dict[str, Any]) -> str:
    dimension = max((len(point) for point in payload.get('reduced_kpoints', []) if isinstance(point, list)), default=0)
    headers = ['path_index', 'distance'] + ['k{0}'.format(index + 1) for index in range(dimension)] + ['band', 'energy', 'unit']
    lines = ['\t'.join(headers)]
    energy_unit = str(payload.get('energy_unit') or 'a.u.')
    for point_index, (kpoint, distance, energies) in enumerate(zip(
        payload.get('reduced_kpoints', []),
        payload.get('distance', []),
        payload.get('energies', []),
    )):
        for band_index, energy in enumerate(energies):
            row = [point_index, distance] + list(kpoint) + [band_index, energy, energy_unit]
            lines.append('\t'.join('{0:.16g}'.format(float(value)) if isinstance(value, (int, float)) else str(value) for value in row))
    return '\n'.join(lines) + '\n'


def _bloch_dos_table(payload: Dict[str, Any]) -> str:
    lines = ['energy\tdensity\tenergy_unit\tdensity_unit']
    energy_unit = str(payload.get('energy_unit') or 'a.u.')
    density_unit = str(payload.get('density_unit') or 'states per cell per energy unit')
    for energy, density in zip(payload.get('energy', []), payload.get('density', [])):
        lines.append('{0:.16g}\t{1:.16g}\t{2}\t{3}'.format(float(energy), float(density), energy_unit, density_unit))
    return '\n'.join(lines) + '\n'


def _bloch_kmesh_table(payload: Dict[str, Any]) -> str:
    dimension = max((len(point) for point in payload.get('reduced_kpoints', []) if isinstance(point, list)), default=0)
    headers = ['kpoint_index'] + ['k{0}'.format(index + 1) for index in range(dimension)] + ['band', 'energy', 'occupation', 'unit']
    lines = ['\t'.join(headers)]
    energy_unit = str(payload.get('energy_unit') or 'a.u.')
    for point_index, (kpoint, energies, occupations) in enumerate(zip(
        payload.get('reduced_kpoints', []),
        payload.get('energies', []),
        payload.get('occupations', []),
    )):
        for band_index, (energy, occupation) in enumerate(zip(energies, occupations)):
            row = [point_index] + list(kpoint) + [band_index, energy, occupation, energy_unit]
            lines.append('\t'.join('{0:.16g}'.format(float(value)) if isinstance(value, (int, float)) else str(value) for value in row))
    return '\n'.join(lines) + '\n'


def _periodic_band_structure_table(payload: Dict[str, Any]) -> str:
    headers = [
        'path_index',
        'distance_1_per_angstrom',
        'special_label',
        'scaled_kx',
        'scaled_ky',
        'scaled_kz',
        'spin',
        'band',
        'energy_hartree',
        'energy_minus_fermi_ev',
    ]
    lines = ['\t'.join(headers)]
    distances = payload.get('path_distances_1_per_angstrom', [])
    kpoints = payload.get('kpoints_scaled', [])
    labels_by_position: Dict[float, List[str]] = {}
    for position, label in zip(
        payload.get('special_point_positions_1_per_angstrom', []),
        payload.get('special_point_labels', []),
    ):
        labels_by_position.setdefault(round(float(position), 12), []).append(str(label))
    for channel in payload.get('channels', []):
        spin = str(channel.get('spin') or 'restricted')
        energies_hartree = channel.get('energies_hartree') or []
        energies_ev = channel.get('energies_relative_to_fermi_ev')
        for point_index, (distance, kpoint, point_energies) in enumerate(zip(distances, kpoints, energies_hartree)):
            relative_point = energies_ev[point_index] if isinstance(energies_ev, list) and point_index < len(energies_ev) else []
            special_label = '|'.join(labels_by_position.get(round(float(distance), 12), []))
            for band_index, energy in enumerate(point_energies):
                relative_energy = relative_point[band_index] if band_index < len(relative_point) else ''
                row = [
                    point_index,
                    distance,
                    special_label,
                    *list(kpoint),
                    spin,
                    band_index,
                    energy,
                    relative_energy,
                ]
                lines.append('\t'.join(
                    '{0:.16g}'.format(float(value)) if isinstance(value, (int, float)) else str(value)
                    for value in row
                ))
    return '\n'.join(lines) + '\n'


def _write_molecular_module_artifacts(state: Dict[str, Any], full_results: Dict[str, Any]) -> None:
    diagnostics = full_results.get('correlation_diagnostics')
    if isinstance(diagnostics, dict) and diagnostics:
        write_result_json_artifact(
            state,
            'correlation_diagnostics',
            retry_artifact_filename('result-correlation-diagnostics', 'json', state.get('retry_count', 0)),
            diagnostics,
            description='Strong-correlation diagnostic summary',
        )
    scf_stability = full_results.get('scf_stability')
    if isinstance(scf_stability, dict) and scf_stability:
        write_result_json_artifact(
            state,
            'scf_stability',
            retry_artifact_filename('result-scf-stability', 'json', state.get('retry_count', 0)),
            scf_stability,
            description='SCF stability analysis summary',
        )
    orbital_summary = full_results.get('orbital_processing')
    if isinstance(orbital_summary, dict) and orbital_summary:
        write_result_json_artifact(
            state,
            'orbital_processing',
            retry_artifact_filename('result-orbital-processing', 'json', state.get('retry_count', 0)),
            orbital_summary,
            description='Orbital processing and localization summary',
        )
        write_text_artifact(
            state,
            'orbital_summary_table',
            retry_artifact_filename('result-orbital-summary', 'txt', state.get('retry_count', 0)),
            orbital_summary_table(orbital_summary),
            description='Tab-separated molecular orbital table',
        )
    active_space = full_results.get('active_space')
    if isinstance(active_space, dict) and active_space:
        write_result_json_artifact(
            state,
            'active_space',
            retry_artifact_filename('result-active-space', 'json', state.get('retry_count', 0)),
            active_space,
            description='Active-space proposal or executed active-space contract',
        )
        write_text_artifact(
            state,
            'active_space_summary_table',
            retry_artifact_filename('result-active-space', 'txt', state.get('retry_count', 0)),
            active_space_summary_table(active_space),
            description='Tab-separated active-space summary',
        )
        audit = active_space.get('audit')
        if isinstance(audit, dict) and audit:
            write_result_json_artifact(
                state,
                'active_space_audit',
                retry_artifact_filename('result-active-space-audit', 'json', state.get('retry_count', 0)),
                audit,
                description='Detailed ActiveSpaceAudit with orbital evidence and approval state',
            )
            write_text_artifact(
                state,
                'active_space_audit_table',
                retry_artifact_filename('result-active-space-audit', 'txt', state.get('retry_count', 0)),
                active_space_audit_table(active_space),
                description='Tab-separated ActiveSpaceAudit orbital evidence table',
            )


def write_result_artifacts(state: Dict[str, Any], task_spec: Any, results: Dict[str, Any]) -> Dict[str, Any]:
    ccsd_labels = results.get('_transient_ccsd_labels')
    dmrg_arrays = results.get('_transient_dmrg_arrays')
    dmet_arrays = results.get('_transient_dmet_arrays')
    molecular_md_arrays = results.get('_transient_molecular_md_arrays')
    provider_logs = results.get('_transient_provider_logs')
    full_results = externalize_orbital_guesses({
        key: value for key, value in results.items()
        if key not in (
            '_transient_ccsd_labels',
            '_transient_dmrg_arrays',
            '_transient_dmet_arrays',
            '_transient_molecular_md_arrays',
            '_transient_provider_logs',
        )
    }, get_run_dir(state) / 'arrays')
    for reference in array_references(full_results):
        register_artifact_reference(state, reference)
    if ccsd_labels is not None:
        from .ccsd_labels import write_ccsd_labels
        full_results['ccsd_labels'] = write_ccsd_labels(state, ccsd_labels)
    one_particle_state = full_results.pop('_transient_one_particle_state', None)
    one_particle_state_summary_payload = one_particle_state_summary(one_particle_state)
    if one_particle_state_summary_payload:
        full_results['one_particle_state'] = one_particle_state_summary_payload
    raw_scf_output = full_results.get('raw_scf_output', '')
    analysis_text = full_results.get('analysis_text', '')
    raw_stdout = full_results.get('raw_stdout', _compose_raw_stdout(
        raw_scf_output,
        analysis_text,
    ))
    trajectory = full_results.get('trajectory')
    if isinstance(molecular_md_arrays, dict) and molecular_md_arrays:
        arrays_ref = write_binary_artifact(
            state,
            'molecular_md_frame_arrays',
            retry_artifact_filename('result-molecular-md-frames', 'npz', state.get('retry_count', 0)),
            serialize_npz_arrays(
                molecular_md_arrays,
                schema='pyscf-agent.molecular-md-frame-arrays.v1',
            ),
            mime_type='application/x-npz',
            description='Sampled MD coordinates, velocities, energies, Fock matrices, and overlap matrices',
        )
        if isinstance(trajectory, dict) and arrays_ref:
            trajectory['frame_arrays_artifact'] = copy.deepcopy(arrays_ref)
            write_result_json_artifact(
                state,
                'molecular_md_manifest',
                retry_artifact_filename('result-molecular-md-manifest', 'json', state.get('retry_count', 0)),
                trajectory,
                description='Molecular-dynamics protocol, AO convention, sampled frame index, and array reference',
            )
    dmrg_result = full_results.get('dmrg_result')
    if not isinstance(dmrg_result, dict):
        cas_result_payload = full_results.get('cas_result')
        dmrg_result = cas_result_payload.get('dmrg_result') if isinstance(cas_result_payload, dict) else None
    if isinstance(dmrg_result, dict) and dmrg_result:
        write_result_json_artifact(
            state,
            'block2_dmrg_result',
            retry_artifact_filename('result-block2-dmrg', 'json', state.get('retry_count', 0)),
            dmrg_result,
            description='block2 DMRG energy, sweep convergence, configuration, and provenance',
        )
        bond_dimension_plan = dmrg_result.get('bond_dimension_plan')
        if isinstance(bond_dimension_plan, dict) and bond_dimension_plan:
            write_result_json_artifact(
                state,
                'block2_bond_dimension_plan',
                retry_artifact_filename('result-block2-bond-dimension-plan', 'json', state.get('retry_count', 0)),
                bond_dimension_plan,
                description='Exact-sector bond-dimension limits and effective block2 schedule',
            )
        checkpoint_manifest = dmrg_result.get('checkpoint_manifest')
        entanglement_active_space = dmrg_result.get('entanglement_active_space_recommendation')
        if isinstance(entanglement_active_space, dict) and entanglement_active_space:
            write_result_json_artifact(
                state,
                'entanglement_active_space_recommendation',
                retry_artifact_filename('result-entanglement-active-space-recommendation', 'json', state.get('retry_count', 0)),
                entanglement_active_space,
                description='Entanglement and natural-occupation evidence for a reviewable active-space expansion',
            )
        dmrg_casscf = dmrg_result.get('casscf')
        if isinstance(dmrg_casscf, dict) and dmrg_casscf:
            write_result_json_artifact(
                state,
                'block2_dmrg_casscf',
                retry_artifact_filename('result-block2-dmrg-casscf', 'json', state.get('retry_count', 0)),
                dmrg_casscf,
                description='block2 DMRG-CASSCF orbital and active-space solver convergence trace',
            )
        if isinstance(checkpoint_manifest, dict) and checkpoint_manifest:
            write_result_json_artifact(
                state,
                'block2_mps_manifest',
                retry_artifact_filename('artifact-block2-mps-manifest', 'json', state.get('retry_count', 0)),
                checkpoint_manifest,
                description='block2 MPS checkpoint file manifest and Hamiltonian fingerprint',
            )
    if isinstance(dmrg_arrays, dict) and dmrg_arrays:
        try:
            write_binary_artifact(
                state,
                'block2_dmrg_arrays',
                retry_artifact_filename('result-block2-dmrg-arrays', 'npz', state.get('retry_count', 0)),
                serialize_npz_arrays(dmrg_arrays),
                mime_type='application/x-npz',
                description='block2 sweep history and reduced-density-matrix arrays',
            )
        except Exception as exc:  # Preserve completed energy and report unavailable evidence.
            _record_artifact_failure(state, full_results, 'block2_dmrg_arrays', exc)
    dmet_result = full_results.get('dmet_result')
    if isinstance(dmet_result, dict) and dmet_result:
        metadata = dmet_result.get('mean_field_state')
        if isinstance(metadata, dict) and isinstance(dmet_arrays, dict):
            from ..providers.libdmet.density_restart import serialize_density_state
            try:
                content = serialize_density_state(
                    metadata, dmet_arrays['mean_field_density_matrix'], run_id=state['run_id'],
                )
                reference = write_binary_artifact(
                    state, 'dmet_mean_field_state',
                    retry_artifact_filename('result-dmet-mean-field-state', 'npz', state.get('retry_count', 0)),
                    content, mime_type='application/x-npz',
                    description='Final full-site spin-resolved mean-field 1RDM for DMET warm starts',
                )
                reference['sha256'] = hashlib.sha256(content).hexdigest()
                register_artifact_reference(state, reference)
                metadata['source_run_id'] = state['run_id']
                metadata['data_artifact'] = reference
            except Exception as exc:
                _record_artifact_failure(state, full_results, 'dmet_mean_field_state', exc)
        write_result_json_artifact(
            state,
            'dmet_result',
            retry_artifact_filename('result-dmet', 'json', state.get('retry_count', 0)),
            dmet_result,
            description='libDMET self-consistency result, fragment definition, and convergence summary',
        )
        iteration_history = dmet_result.get('iteration_history')
        if isinstance(iteration_history, dict) and iteration_history:
            write_result_json_artifact(
                state,
                'dmet_iteration_history',
                retry_artifact_filename('result-dmet-iterations', 'json', state.get('retry_count', 0)),
                iteration_history,
                description='Per-iteration DMET energy, density fit, chemical potential, and correlation-potential changes',
            )
    if isinstance(dmet_arrays, dict) and dmet_arrays:
        try:
            write_binary_artifact(
                state,
                'dmet_numerical_arrays',
                retry_artifact_filename('result-dmet-arrays', 'npz', state.get('retry_count', 0)),
                serialize_npz_arrays(dmet_arrays),
                mime_type='application/x-npz',
                description='DMET embedding basis, density matrices, correlation potential, and impurity Hamiltonian arrays',
            )
        except Exception as exc:  # Preserve completed energy and report unavailable evidence.
            _record_artifact_failure(state, full_results, 'dmet_numerical_arrays', exc)
    if isinstance(provider_logs, list):
        run_directory = get_run_dir(state).resolve()
        for provider_log in provider_logs:
            if not isinstance(provider_log, dict):
                continue
            log_path = Path(str(provider_log.get('path') or '')).expanduser().resolve()
            try:
                log_path.relative_to(run_directory)
            except ValueError:
                continue
            register_existing_artifact(
                state,
                str(provider_log.get('kind') or 'provider_output_log'),
                log_path,
                mime_type=str(provider_log.get('mime_type') or 'text/plain; charset=utf-8'),
                description=str(provider_log.get('description') or 'Provider output log'),
            )
    if task_spec.task_type == 'model_hamiltonian':
        model_spec_artifact = copy.deepcopy(task_spec.model_hamiltonian.spec)
        if isinstance(model_spec_artifact, dict):
            model_spec_artifact['solver'] = task_spec.solver.name
        write_json_artifact(
            state,
            'model_hamiltonian_spec',
            retry_artifact_filename('artifact-model-hamiltonian-spec', 'json', state.get('retry_count', 0)),
            model_spec_artifact,
            description='Normalized Model Hamiltonian input spec with the executed solver',
        )
        strong_correlation_diagnostics = full_results.get('strong_correlation_diagnostics')
        if isinstance(strong_correlation_diagnostics, dict) and strong_correlation_diagnostics:
            diagnostics_payload = copy.deepcopy(strong_correlation_diagnostics)
            diagnostics_payload.setdefault('schema', 'pyscf-agent.strong-correlation-diagnostics.v1')
            write_result_json_artifact(
                state,
                'strong_correlation_diagnostics',
                retry_artifact_filename('result-strong-correlation-diagnostics', 'json', state.get('retry_count', 0)),
                diagnostics_payload,
                description='Model-Hamiltonian strong-correlation diagnostics and method recommendation',
            )
        energy_levels = full_results.get('energy_levels')
        if isinstance(energy_levels, list) and energy_levels:
            energy_unit = str(full_results.get('energy_unit') or 'a.u.')
            spectrum_payload = {
                'model': full_results.get('model'),
                'solver': full_results.get('solver'),
                'norb': full_results.get('norb'),
                'nelec': full_results.get('nelec'),
                'basis_dimension': full_results.get('many_body_basis_dimension'),
                'spectrum_method': full_results.get('energy_spectrum_method') or full_results.get('solver'),
                'energy_level_count': len(energy_levels),
                'energy_unit': energy_unit,
                'levels': energy_levels,
            }
            write_result_json_artifact(
                state,
                'model_hamiltonian_energy_levels',
                retry_artifact_filename('result-model-energy-levels', 'json', state.get('retry_count', 0)),
                spectrum_payload,
                description='Energy levels returned by the current Model Hamiltonian solver',
            )
            write_text_artifact(
                state,
                'model_hamiltonian_energy_levels_txt',
                retry_artifact_filename('result-model-energy-levels', 'txt', state.get('retry_count', 0)),
                _model_energy_levels_table(energy_levels, energy_unit),
                description='Tab-separated energy levels returned by the current Model Hamiltonian solver',
            )
        bloch_bands = full_results.get('bloch_band_structure')
        if isinstance(bloch_bands, dict) and bloch_bands:
            write_result_json_artifact(
                state,
                'model_bloch_bands',
                retry_artifact_filename('result-model-bloch-bands', 'json', state.get('retry_count', 0)),
                bloch_bands,
                description='Bloch band energies along the selected reciprocal-space path',
            )
            write_text_artifact(
                state,
                'model_bloch_bands_tsv',
                retry_artifact_filename('result-model-bloch-bands', 'tsv', state.get('retry_count', 0)),
                _bloch_bands_table(bloch_bands),
                description='Tab-separated Bloch band plotting data',
            )
        bloch_dos = full_results.get('bloch_dos')
        if isinstance(bloch_dos, dict) and bloch_dos:
            write_result_json_artifact(
                state,
                'model_bloch_dos',
                retry_artifact_filename('result-model-bloch-dos', 'json', state.get('retry_count', 0)),
                bloch_dos,
                description='Gaussian-broadened Bloch density of states',
            )
            write_text_artifact(
                state,
                'model_bloch_dos_tsv',
                retry_artifact_filename('result-model-bloch-dos', 'tsv', state.get('retry_count', 0)),
                _bloch_dos_table(bloch_dos),
                description='Tab-separated Bloch density-of-states plotting data',
            )
        bloch_kmesh = full_results.get('bloch_kmesh')
        if isinstance(bloch_kmesh, dict) and bloch_kmesh:
            write_result_json_artifact(
                state,
                'model_bloch_kmesh',
                retry_artifact_filename('result-model-bloch-kmesh', 'json', state.get('retry_count', 0)),
                bloch_kmesh,
                description='Bloch k-mesh eigenvalues and zero-temperature occupations',
            )
            write_text_artifact(
                state,
                'model_bloch_kmesh_tsv',
                retry_artifact_filename('result-model-bloch-kmesh', 'tsv', state.get('retry_count', 0)),
                _bloch_kmesh_table(bloch_kmesh),
                description='Tab-separated Bloch k-mesh eigenvalues and occupations',
            )
        bloch_samples = full_results.get('bloch_hamiltonian_samples')
        if isinstance(bloch_samples, list) and bloch_samples:
            write_result_json_artifact(
                state,
                'model_bloch_hamiltonian_samples',
                retry_artifact_filename('result-model-bloch-hamiltonian-samples', 'json', state.get('retry_count', 0)),
                bloch_samples,
                description='Complex h(k) matrices at labeled high-symmetry path vertices',
            )
    elif task_spec.task_type == 'periodic':
        write_json_artifact(
            state,
            'periodic_task_input',
            retry_artifact_filename('input-periodic-task', 'json', state.get('retry_count', 0)),
            task_spec_to_dict(task_spec),
            description='Normalized structured periodic calculation input',
        )
        structure_format = str(task_spec.periodic.structure_format or 'poscar').strip().lower()
        structure_extension = 'cif' if structure_format == 'cif' else 'vasp'
        write_text_artifact(
            state,
            'periodic_structure_source',
            retry_artifact_filename('input-periodic-structure', structure_extension, state.get('retry_count', 0)),
            task_spec.periodic.structure_text or '',
            description='Original periodic structure input',
        )
        periodic_structure = full_results.get('periodic_structure')
        if isinstance(periodic_structure, dict) and periodic_structure:
            write_result_json_artifact(
                state,
                'periodic_cell',
                retry_artifact_filename('artifact-periodic-cell', 'json', state.get('retry_count', 0)),
                periodic_structure,
                description='Parsed periodic cell and atomic coordinates',
            )
            write_text_artifact(
                state,
                'periodic_structure_normalized',
                retry_artifact_filename('artifact-periodic-structure-normalized', 'vasp', state.get('retry_count', 0)),
                normalized_periodic_poscar(periodic_structure),
                description='Effective normalized P1 POSCAR used by PySCF',
            )
        periodic_input_structure = full_results.get('periodic_input_structure')
        if isinstance(periodic_input_structure, dict) and periodic_input_structure:
            write_result_json_artifact(
                state,
                'periodic_input_cell',
                retry_artifact_filename('artifact-periodic-input-cell', 'json', state.get('retry_count', 0)),
                periodic_input_structure,
                description='Parsed user-input cell before SeeK-path standardization',
            )
            write_text_artifact(
                state,
                'periodic_input_structure_normalized',
                retry_artifact_filename(
                    'artifact-periodic-input-structure-normalized',
                    'vasp',
                    state.get('retry_count', 0),
                ),
                normalized_periodic_poscar(periodic_input_structure),
                description='Normalized user-input P1 POSCAR before SeeK-path standardization',
            )
        seekpath_standardization = full_results.get('seekpath_standardization')
        if isinstance(seekpath_standardization, dict) and seekpath_standardization:
            write_result_json_artifact(
                state,
                'periodic_seekpath_standardization',
                retry_artifact_filename('artifact-periodic-seekpath-standardization', 'json', state.get('retry_count', 0)),
                seekpath_standardization,
                description='SeeK-path primitive-cell, symmetry, path, and transformation audit',
            )
        periodic_numerics = full_results.get('periodic_numerics')
        if isinstance(periodic_numerics, dict) and periodic_numerics:
            write_result_json_artifact(
                state,
                'periodic_numerics',
                retry_artifact_filename('artifact-periodic-numerics', 'json', state.get('retry_count', 0)),
                periodic_numerics,
                description='Effective periodic numerical settings, k-points, and occupations contract',
            )
        periodic_orbitals = full_results.get('periodic_orbitals')
        if isinstance(periodic_orbitals, dict) and periodic_orbitals:
            write_result_json_artifact(
                state,
                'periodic_orbitals',
                retry_artifact_filename('result-periodic-orbitals', 'json', state.get('retry_count', 0)),
                periodic_orbitals,
                description='Periodic mean-field orbital energies and occupations by spin and k-point',
            )
        periodic_band_structure = full_results.get('periodic_band_structure')
        if isinstance(periodic_band_structure, dict) and periodic_band_structure:
            write_result_json_artifact(
                state,
                'periodic_band_structure',
                retry_artifact_filename('result-periodic-band-structure', 'json', state.get('retry_count', 0)),
                periodic_band_structure,
                description='Periodic mean-field bands along the selected high-symmetry path',
            )
            write_text_artifact(
                state,
                'periodic_band_structure_tsv',
                retry_artifact_filename('result-periodic-band-structure', 'tsv', state.get('retry_count', 0)),
                _periodic_band_structure_table(periodic_band_structure),
                description='Tab-separated periodic band-structure plotting data',
            )
        write_result_json_artifact(
            state,
            'periodic_scf_summary',
            retry_artifact_filename('result-periodic-scf', 'json', state.get('retry_count', 0)),
            {
                'method': full_results.get('method'),
                'xc': full_results.get('xc'),
                'reference': full_results.get('reference'),
                'converged': full_results.get('converged'),
                'energy': full_results.get('energy'),
                'energy_unit': full_results.get('energy_unit'),
                'kmesh': full_results.get('kmesh'),
                'kpoint_count': full_results.get('kpoint_count'),
                'kpoint_scheme': full_results.get('kpoint_scheme'),
                'kpoint_shift': full_results.get('kpoint_shift'),
                'band_gap': full_results.get('band_gap'),
                'direct_gap': full_results.get('direct_gap'),
                'gap_type': full_results.get('gap_type'),
                'is_metal': full_results.get('is_metal'),
                'valence_band_max': full_results.get('valence_band_max'),
                'conduction_band_min': full_results.get('conduction_band_min'),
                'fermi_energy': full_results.get('fermi_energy'),
                'fermi_energy_by_spin': full_results.get('fermi_energy_by_spin'),
                'fermi_energy_source': full_results.get('fermi_energy_source'),
                'band_edges': full_results.get('band_edges'),
                'band_path_mode': full_results.get('band_path_mode'),
                'band_path': full_results.get('band_path'),
                'band_path_point_count': full_results.get('band_path_point_count'),
                'band_path_labels': full_results.get('band_path_labels'),
                'band_structure_status': full_results.get('band_structure_status'),
                'cell_role': periodic_structure.get('cell_role') if isinstance(periodic_structure, dict) else None,
                'seekpath_standardization': full_results.get('seekpath_standardization'),
                'density_fitting': full_results.get('density_fitting'),
                'smearing': full_results.get('smearing'),
            },
            description='Periodic mean-field calculation summary',
        )
    else:
        if one_particle_state_summary_payload:
            try:
                archive = serialize_one_particle_state(one_particle_state)
                state_ref = write_binary_artifact(
                    state,
                    'one_particle_state',
                    retry_artifact_filename('result-one-particle-state', 'npz', state.get('retry_count', 0)),
                    archive,
                    mime_type='application/x-npz',
                    description='Reusable AO reference 1RDM for compatible molecular continuation calculations',
                )
                if state_ref:
                    full_results['one_particle_state']['data_artifact'] = state_ref
                    write_result_json_artifact(
                        state,
                        'one_particle_state_metadata',
                        retry_artifact_filename('result-one-particle-state', 'json', state.get('retry_count', 0)),
                        {
                            **one_particle_state_summary_payload,
                            'data_artifact': state_ref,
                        },
                        description='One-particle-state provenance and binary artifact reference',
                    )
            except Exception as exc:
                full_results.pop('one_particle_state', None)
                _record_artifact_failure(state, full_results, 'one_particle_state', exc)
        _write_molecular_module_artifacts(state, full_results)
        cas_result = full_results.get('cas_result')
        if isinstance(cas_result, dict) and cas_result:
            write_result_json_artifact(
                state,
                'cas_result',
                retry_artifact_filename('result-cas', 'json', state.get('retry_count', 0)),
                cas_result,
                description='CASCI/CASSCF structured result',
            )
        post_cas_results = full_results.get('post_cas_results') if isinstance(full_results.get('post_cas_results'), dict) else {}
        sc_nevpt2 = post_cas_results.get('sc_nevpt2') if isinstance(post_cas_results, dict) else None
        if isinstance(sc_nevpt2, dict) and sc_nevpt2:
            write_result_json_artifact(
                state,
                'sc_nevpt2',
                retry_artifact_filename('result-sc-nevpt2', 'json', state.get('retry_count', 0)),
                sc_nevpt2,
                description='SC-NEVPT2 post-CAS correction result',
            )
    write_text_artifact(
        state,
        'raw_scf_output',
        retry_artifact_filename('log-pyscf-output', 'log', state.get('retry_count', 0)),
        raw_scf_output,
        description='Full PySCF SCF output log',
    )
    write_text_artifact(
        state,
        'raw_stdout',
        retry_artifact_filename('log-stdout', 'log', state.get('retry_count', 0)),
        raw_stdout,
        description='Composed stdout preview source',
    )
    write_result_json_artifact(
        state,
        'structured_results',
        retry_artifact_filename('result-structured', 'json', state.get('retry_count', 0)),
        full_results,
        description='Full structured PySCF result payload',
    )
    compact_results = compact_structured_results(full_results)
    state['structured_results'] = compact_results
    state['compact_results'] = compact_results
    state['raw_scf_output'] = preview_text(raw_scf_output)
    state['analysis_text'] = preview_text(analysis_text)
    state['raw_stdout'] = preview_text(raw_stdout)
    return full_results


def refresh_molecular_module_artifacts(
    state: Dict[str, Any],
    task_spec: Any,
    full_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist numerical module additions without rewriting solver logs or restart data."""

    if task_spec.task_type != 'molecular':
        raise ValueError('molecular module artifacts require a molecular TaskSpec')
    payload = externalize_orbital_guesses(full_results, get_run_dir(state) / 'arrays')
    for reference in array_references(payload):
        register_artifact_reference(state, reference)
    _write_molecular_module_artifacts(state, payload)
    write_result_json_artifact(
        state,
        'structured_results',
        retry_artifact_filename('result-structured', 'json', state.get('retry_count', 0)),
        payload,
        description='Full structured PySCF result payload',
    )
    compact_results = compact_structured_results(payload)
    state['structured_results'] = compact_results
    state['compact_results'] = compact_results
    return payload


def _refresh_structured_module_results(
    state: Dict[str, Any],
    full_results: Dict[str, Any],
) -> Dict[str, Any]:
    payload = externalize_orbital_guesses(full_results, get_run_dir(state) / 'arrays')
    for reference in array_references(payload):
        register_artifact_reference(state, reference)
    write_result_json_artifact(
        state,
        'structured_results',
        retry_artifact_filename('result-structured', 'json', state.get('retry_count', 0)),
        payload,
        description='Full structured PySCF result payload',
    )
    compact_results = compact_structured_results(payload)
    state['structured_results'] = compact_results
    state['compact_results'] = compact_results
    state['analysis_text'] = preview_text(payload.get('analysis_text', ''))
    return payload


def refresh_model_module_artifacts(
    state: Dict[str, Any],
    full_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist model diagnostics computed after the numerical solver returns."""

    diagnostics = full_results.get('strong_correlation_diagnostics')
    if isinstance(diagnostics, dict) and diagnostics:
        payload = copy.deepcopy(diagnostics)
        payload.setdefault('schema', 'pyscf-agent.strong-correlation-diagnostics.v1')
        write_result_json_artifact(
            state,
            'strong_correlation_diagnostics',
            retry_artifact_filename('result-strong-correlation-diagnostics', 'json', state.get('retry_count', 0)),
            payload,
            description='Model-Hamiltonian strong-correlation diagnostics and method recommendation',
        )
    return _refresh_structured_module_results(state, full_results)


def refresh_periodic_module_artifacts(
    state: Dict[str, Any],
    full_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist periodic band-path results computed after SCF completion."""

    band_structure = full_results.get('periodic_band_structure')
    if isinstance(band_structure, dict) and band_structure:
        write_result_json_artifact(
            state,
            'periodic_band_structure',
            retry_artifact_filename('result-periodic-band-structure', 'json', state.get('retry_count', 0)),
            band_structure,
            description='Periodic mean-field bands along the selected high-symmetry path',
        )
        write_text_artifact(
            state,
            'periodic_band_structure_tsv',
            retry_artifact_filename('result-periodic-band-structure', 'tsv', state.get('retry_count', 0)),
            _periodic_band_structure_table(band_structure),
            description='Tab-separated periodic band-structure plotting data',
        )
    return _refresh_structured_module_results(state, full_results)
