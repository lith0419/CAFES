"""Deterministic energy-stratified selection from accepted trajectory indexes."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping

from pyscf_agent.artifacts import default_artifact_repository
from pyscf_agent.input_validation import finite_float, integer, reject_unknown_fields

from computational_study_agent.datasets.hamiltonian.contracts import (
    HAMILTONIAN_MANIFEST_SCHEMA, HAMILTONIAN_SAMPLE_SCHEMA, MolecularGeometry,
)
from computational_study_agent.datasets.hamiltonian.postprocessing import _index_path, _load_json, _manifest_path


ENERGY_SELECTION_ACTION = 'select_energy_stratified_frames'
ENERGY_SELECTION_SCHEMA = 'pyscf-agent.trajectory-frame-selection.v1'
SELECTED_FRAME_SCHEMA = 'pyscf-agent.selected-trajectory-frame.v1'
# Atomic unit of time in femtoseconds; the dataset protocol stores dt in a.u.
ATOMIC_TIME_FS = 0.024188843265857


def selection_options(options: Mapping[str, Any]) -> Dict[str, Any]:
    reject_unknown_fields(options, {'count_per_molecule', 'time_start_fs', 'time_end_fs'},
                          'energy selection')
    count = integer(options.get('count_per_molecule'), 'count_per_molecule')
    start = finite_float(options.get('time_start_fs', 0.0), 'time_start_fs')
    end = options.get('time_end_fs')
    if end is not None:
        end = finite_float(end, 'time_end_fs')
    if count < 1:
        raise ValueError('count_per_molecule must be positive')
    if start < 0 or (end is not None and end < start):
        raise ValueError('Require 0 <= time_start_fs <= time_end_fs')
    return {'count_per_molecule': count, 'time_start_fs': start, 'time_end_fs': end}


def build_energy_stratified_selection(
    sample_index: Path, source_manifest: Mapping[str, Any], options: Mapping[str, Any],
) -> Dict[str, Any]:
    """Read JSONL metadata only. Each populated molecule gets exactly K frames.

    Equal-population energy bins differ in size by at most one (larger bins
    first). Take the lower median rank in each bin; break energy ties by frame
    index, then sample ID. No coordinates, energies or source records change.
    """
    options = selection_options(options)
    if source_manifest.get('schema') != HAMILTONIAN_MANIFEST_SCHEMA:
        raise ValueError('Unsupported Hamiltonian dataset manifest schema')
    dt = finite_float(source_manifest['spec']['molecular_dynamics']['time_step_au'],
                      'dataset time_step_au')
    if dt <= 0:
        raise ValueError('dataset time_step_au must be positive')
    groups = defaultdict(list)
    seen = set()
    source_count = outside_count = 0
    # Stream the index; retain only the small geometry/energy/provenance fields.
    # Fock, overlap and AO metadata are neither retained nor dereferenced.
    with Path(sample_index).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('schema') != HAMILTONIAN_SAMPLE_SCHEMA or row.get('converged') is not True:
                raise ValueError('Energy selection requires accepted, converged Hamiltonian samples')
            sample_id = row['sample_id']
            if sample_id in seen:
                raise ValueError('Duplicate sample_id: {0}'.format(sample_id))
            seen.add(sample_id)
            source_count += 1
            molecule_id = row['molecule_id']
            candidates = groups[molecule_id]
            provenance = row['provenance']
            frame_index = integer(provenance['frame_index'], 'frame_index')
            if frame_index < 0:
                raise ValueError('frame_index must be non-negative')
            # Older indexes omit time. Their fixed-step MD protocol explicitly
            # defines t = frame_index * dt, not array_index * dt.
            time_au = finite_float(provenance.get('time_au', frame_index * dt), 'time_au')
            time_fs = time_au * ATOMIC_TIME_FS
            if (time_fs < options['time_start_fs'] - 1e-8 or
                    (options['time_end_fs'] is not None and time_fs > options['time_end_fs'] + 1e-8)):
                outside_count += 1
                continue
            energy = finite_float(row['total_energy_hartree'], 'sample electronic energy')
            candidates.append({
                'schema': SELECTED_FRAME_SCHEMA,
                'sample_id': sample_id, 'molecule_id': molecule_id,
                'geometry_id': row['geometry_id'], 'geometry': row['geometry'],
                'split': row['split'], 'electronic_structure': row['electronic_structure'],
                'provenance': provenance, 'frame_index': frame_index, 'time_fs': time_fs,
                # HamiltonianSample.total_energy_hartree is the electronic BO
                # total energy (MD potential), not potential + nuclear kinetic.
                'potential_energy_hartree': energy,
            })
    count = options['count_per_molecule']
    selected, summaries = [], []
    for molecule_id, candidates in sorted(groups.items()):
        summary = {'molecule_id': molecule_id, 'candidate_count': len(candidates),
                   'selected_count': 0, 'status': 'insufficient_candidates'}
        if len(candidates) >= count:
            candidates.sort(key=lambda row: (row['potential_energy_hartree'],
                                             row['frame_index'], row['sample_id']))
            quotient, remainder = divmod(len(candidates), count)
            offset = 0
            for bin_index in range(count):
                size = quotient + (bin_index < remainder)
                rank = offset + (size - 1) // 2
                row = candidates[rank]
                geometry = MolecularGeometry.from_dict(row['geometry'])
                if geometry.sample_id != row['sample_id'] or geometry.molecule_id != molecule_id:
                    raise ValueError('Selected sample identity does not match its geometry')
                if geometry.coordinate_unit != 'Angstrom':
                    raise ValueError('Selected geometry must use Angstrom coordinates')
                row['selection'] = {
                    'energy_bin': bin_index + 1, 'bin_size': size,
                    'energy_rank_0based': rank,
                    'bin_energy_range_hartree': [candidates[offset]['potential_energy_hartree'],
                                               candidates[offset + size - 1]['potential_energy_hartree']],
                }
                selected.append(row)
                offset += size
            summary.update(status='selected', selected_count=count)
        summaries.append(summary)
    expected_molecules = integer(source_manifest['spec']['target_molecule_count'],
                                 'target_molecule_count')
    selected_molecules = sum(row['status'] == 'selected' for row in summaries)
    status = ('complete' if selected_molecules == expected_molecules
              else 'partial' if selected else 'empty')
    return {
        'schema': ENERGY_SELECTION_SCHEMA, 'method': ENERGY_SELECTION_ACTION,
        'status': status, 'parameters': options,
        'algorithm': 'energy ascending; equal-population bins, larger bins first; lower median per bin',
        'tie_break': ['frame_index', 'sample_id'],
        'energy_source': 'HamiltonianSample.total_energy_hartree (MD potential energy)',
        'time_source': 'provenance.time_au or frame_index * dataset time_step_au',
        'expected_molecule_count': expected_molecules,
        'source_accepted_frame_count': source_count,
        'source_rejected_frame_count': source_manifest.get('rejected_structure_count'),
        'outside_time_window_count': outside_count,
        'molecule_count': selected_molecules, 'selected_frame_count': len(selected),
        'unselected_molecule_count': expected_molecules - selected_molecules,
        'molecules': summaries, 'selected_frames': selected,
    }


def _xyz_text(rows):
    from pyscf.data.elements import ELEMENTS

    lines = []
    for row in rows:
        geometry = row['geometry']
        lines.extend([str(len(geometry['atomic_numbers'])), json.dumps({
            'sample_id': row['sample_id'], 'time_fs': row['time_fs'],
            'potential_energy_hartree': row['potential_energy_hartree'],
            'charge': geometry.get('charge', 0), 'spin': geometry.get('spin', 0),
        }, ensure_ascii=True)])
        for number, position in zip(geometry['atomic_numbers'], geometry['positions']):
            lines.append('{0} {1}'.format(ELEMENTS[number], ' '.join(format(v, '.17g') for v in position)))
    return '\n'.join(lines) + ('\n' if lines else '')


def select_energy_stratified_frames(
    report: Mapping[str, Any], *, options: Mapping[str, Any], output_dir: Path,
) -> Dict[str, Any]:
    """Registered action: export geometries without executing any new calculation."""
    options = selection_options(options)
    work_root = Path(report['work_dir']).expanduser().absolute()
    manifest_path = _manifest_path(report, work_root)
    manifest = _load_json(manifest_path)
    sample_index = _index_path(manifest, 'sample_index', manifest_path.parent)
    selection = build_energy_stratified_selection(sample_index, manifest, options)
    selection.update(study_id=report['study_id'], source_manifest=str(manifest_path),
                     source_sample_index=str(sample_index))
    return write_energy_stratified_selection(selection, output_dir=output_dir)


def write_energy_stratified_selection(selection: Mapping[str, Any], *, output_dir: Path) -> Dict[str, Any]:
    """Write a computed selection; remote readers may return just this small payload."""
    selection = dict(selection)
    options = selection['parameters']
    rows = selection.pop('selected_frames')
    xyz = _xyz_text(rows)
    repository = default_artifact_repository()
    output_dir = Path(output_dir)
    artifacts = [repository.write_text(
        output_dir / 'selected-frames.jsonl',
        ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in rows),
        kind='selected_trajectory_frames', mime_type='application/x-ndjson',
        description='Selected geometries, source identities, energies and energy-bin ranks',
    ), repository.write_text(
        output_dir / 'selected-frames.xyz', xyz, kind='selected_trajectory_xyz',
        description='Selected trajectory geometries in Angstrom, original atom order',
    )]
    selection['artifacts'] = artifacts.copy()
    artifacts.append(repository.write_json(
        output_dir / 'selection-manifest.json', selection, kind='trajectory_frame_selection',
        description='Energy-stratified selection parameters and per-molecule counts',
    ))
    return {'action': ENERGY_SELECTION_ACTION,
            'status': 'succeeded' if selection['status'] == 'complete' else 'partial' if rows else 'skipped',
            'selection_status': selection['status'], 'parameters': options,
            'molecule_count': selection['molecule_count'], 'sample_count': len(rows),
            'unselected_molecule_count': selection['unselected_molecule_count'], 'artifacts': artifacts}
