"""Finalize sampled MD artifacts into a QH9-compatible dataset index."""

from __future__ import annotations

from pyscf_agent.serialization import json_default

import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pyscf_agent.artifacts import default_artifact_repository

from computational_study_agent.datasets.hamiltonian.contracts import (
    AOBasisMetadata,
    DATASET_SPLITS,
    HamiltonianDatasetManifest,
    HamiltonianDatasetSpec,
    HamiltonianSample,
    HamiltonianSampleRejection,
    MatrixArtifactReference,
    MolecularGeometry,
)


_MD_ARRAY_SCHEMA = 'pyscf-agent.molecular-md-frame-arrays.v1'


def finalize_hamiltonian_dataset(
    study_report: Any,
    dataset_spec: Any,
    output_dir: Any,
) -> HamiltonianDatasetManifest:
    """Build sample/rejection JSONL indexes without copying matrix arrays."""

    spec = (
        dataset_spec
        if isinstance(dataset_spec, HamiltonianDatasetSpec)
        else HamiltonianDatasetSpec.from_dict(dataset_spec)
    )
    report = _report_payload(study_report)
    case_records = report.get('cases')
    if not isinstance(case_records, list):
        raise ValueError('Study report cases must be a list.')
    if len(case_records) != spec.target_molecule_count:
        raise ValueError(
            'Study report contains {0} cases; dataset spec requires {1}.'.format(
                len(case_records),
                spec.target_molecule_count,
            )
        )

    split_assignments = _split_assignments(case_records, spec)
    samples: List[HamiltonianSample] = []
    rejections: List[HamiltonianSampleRejection] = []
    for case_record in case_records:
        molecule_id = _case_molecule_id(case_record)
        task_report = case_record.get('task_report') if isinstance(case_record, dict) else None
        task_report = task_report if isinstance(task_report, dict) else {}
        status = str(task_report.get('execution_status') or 'missing').strip().lower()
        if status != 'succeeded':
            rejections.extend(_trajectory_rejections(
                molecule_id,
                spec,
                reasons=('task_status:{0}'.format(status),),
                provenance={'case_id': case_record.get('case_id')},
            ))
            continue
        try:
            case_samples, case_rejections = _samples_from_case(
                case_record,
                spec,
                split_assignments,
                report,
            )
        except Exception as exc:
            rejections.extend(_trajectory_rejections(
                molecule_id,
                spec,
                reasons=('trajectory_artifact_invalid', str(exc)),
                provenance={'case_id': case_record.get('case_id')},
            ))
            continue
        samples.extend(case_samples)
        rejections.extend(case_rejections)

    if len(samples) + len(rejections) != spec.target_structure_count:
        raise ValueError(
            'Finalization accounted for {0} structures; expected {1}.'.format(
                len(samples) + len(rejections),
                spec.target_structure_count,
            )
        )

    output_path = Path(os.path.abspath(os.path.expanduser(str(output_dir))))
    output_path.mkdir(parents=True, exist_ok=True)
    sample_index_ref = default_artifact_repository().write_text(
        output_path / 'samples.jsonl',
        _json_lines(sample.to_dict() for sample in samples),
        kind='hamiltonian_sample_index',
        mime_type='application/x-ndjson; charset=utf-8',
        description='Accepted QH9-compatible Hamiltonian sample index',
    )
    rejection_index_ref = default_artifact_repository().write_text(
        output_path / 'rejections.jsonl',
        _json_lines(rejection.to_dict() for rejection in rejections),
        kind='hamiltonian_rejection_index',
        mime_type='application/x-ndjson; charset=utf-8',
        description='Rejected or missing Hamiltonian sample index',
    )
    split_counts = {
        split: sum(sample.split == split for sample in samples)
        for split in DATASET_SPLITS
    }
    manifest_path = output_path / 'dataset-manifest.json'
    manifest = HamiltonianDatasetManifest(
        spec=spec,
        molecule_count=len({_case_molecule_id(case) for case in case_records}),
        accepted_structure_count=len(samples),
        rejected_structure_count=len(rejections),
        split_counts=split_counts,
        artifacts={
            'sample_index': sample_index_ref['path'],
            'rejection_index': rejection_index_ref['path'],
            'dataset_manifest': str(manifest_path),
        },
    )
    default_artifact_repository().write_json(
        manifest_path,
        manifest.to_dict(),
        kind='hamiltonian_dataset_manifest',
        description='QH9-compatible dataset completion, split, and artifact manifest',
    )
    return manifest


def _samples_from_case(
    case_record: Mapping[str, Any],
    spec: HamiltonianDatasetSpec,
    split_assignments: Mapping[str, str],
    study_report: Mapping[str, Any],
) -> Tuple[List[HamiltonianSample], List[HamiltonianSampleRejection]]:
    embedded = _samples_from_embedded_metadata(
        case_record,
        spec,
        split_assignments,
        study_report,
    )
    if embedded is not None:
        return embedded

    import numpy as np  # pylint: disable=import-outside-toplevel

    molecule_id = _case_molecule_id(case_record)
    task_report = case_record['task_report']
    structured = task_report.get('structured_results')
    trajectory = structured.get('trajectory') if isinstance(structured, dict) else None
    if not isinstance(trajectory, dict):
        raise ValueError('Task report does not contain trajectory metadata.')
    arrays_ref = trajectory.get('frame_arrays_artifact')
    if not isinstance(arrays_ref, dict) or not arrays_ref.get('path'):
        raise ValueError('Trajectory does not reference a frame array artifact.')
    arrays_path = _resolve_artifact_path(arrays_ref['path'], task_report, study_report)
    if not arrays_path.is_file():
        raise ValueError('Frame array artifact is missing: {0}'.format(arrays_path))

    with np.load(arrays_path, allow_pickle=False) as archive:
        if str(archive['schema']) != _MD_ARRAY_SCHEMA:
            raise ValueError('Unsupported MD frame array schema.')
        required = (
            'atomic_numbers',
            'positions_angstrom',
            'potential_energies_hartree',
            'frame_indices',
            'fock_matrices',
            'overlap_matrices',
        )
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError('Frame array artifact is missing keys: {0}'.format(', '.join(missing)))
        atomic_numbers = tuple(int(value) for value in archive['atomic_numbers'].tolist())
        positions = np.asarray(archive['positions_angstrom'])
        energies = np.asarray(archive['potential_energies_hartree'])
        frame_indices = np.asarray(archive['frame_indices'])
        fock = np.asarray(archive['fock_matrices'])
        overlap = np.asarray(archive['overlap_matrices'])
        expected_count = spec.geometries_per_molecule
        for name, values in (
            ('positions_angstrom', positions),
            ('potential_energies_hartree', energies),
            ('frame_indices', frame_indices),
            ('fock_matrices', fock),
            ('overlap_matrices', overlap),
        ):
            if values.shape[0] != expected_count:
                raise ValueError(
                    '{0} contains {1} samples; expected {2}.'.format(
                        name,
                        values.shape[0],
                        expected_count,
                    )
                )
        if positions.shape[1:] != (len(atomic_numbers), 3):
            raise ValueError('Coordinate array shape does not match atomic_numbers.')
        if fock.ndim != 3 or overlap.shape != fock.shape or fock.shape[1] != fock.shape[2]:
            raise ValueError('Fock and overlap arrays must contain equally shaped square matrices.')
        ao_basis = _ao_basis_from_manifest(trajectory.get('ao_basis'), fock.shape[1])
        frame_metadata = trajectory.get('frames')
        if not isinstance(frame_metadata, list) or len(frame_metadata) != expected_count:
            raise ValueError('Trajectory frame metadata count is inconsistent.')

        samples: List[HamiltonianSample] = []
        rejections: List[HamiltonianSampleRejection] = []
        for array_index in range(expected_count):
            frame_index = int(frame_indices[array_index])
            geometry_id = 'frame-{0:06d}'.format(frame_index)
            frame = frame_metadata[array_index]
            converged = bool(frame.get('converged')) if isinstance(frame, dict) else False
            provenance = {
                'study_id': study_report.get('study_id'),
                'case_id': case_record.get('case_id'),
                'run_id': task_report.get('run_id'),
                'frame_index': frame_index,
                'array_index': array_index,
            }
            if not converged:
                rejections.append(HamiltonianSampleRejection(
                    molecule_id=molecule_id,
                    geometry_id=geometry_id,
                    reasons=('scf_unconverged',),
                    provenance=provenance,
                ))
                continue
            geometry = MolecularGeometry(
                molecule_id=molecule_id,
                geometry_id=geometry_id,
                atomic_numbers=atomic_numbers,
                positions=tuple(
                    tuple(float(value) for value in row)
                    for row in positions[array_index].tolist()
                ),
                coordinate_unit='Angstrom',
                source=provenance,
            )
            matrix_shape = (int(fock.shape[1]), int(fock.shape[2]))
            split = split_assignments.get(
                '{0}:{1}'.format(molecule_id, geometry_id),
                split_assignments.get(molecule_id, ''),
            )
            samples.append(HamiltonianSample(
                geometry=geometry,
                split=split,
                electronic_structure=spec.electronic_structure,
                ao_basis=ao_basis,
                fock=MatrixArtifactReference(
                    path=str(arrays_path),
                    array_key='fock_matrices',
                    array_index=array_index,
                    shape=matrix_shape,
                    dtype=str(fock.dtype),
                ),
                overlap=MatrixArtifactReference(
                    path=str(arrays_path),
                    array_key='overlap_matrices',
                    array_index=array_index,
                    shape=matrix_shape,
                    dtype=str(overlap.dtype),
                ),
                converged=converged,
                total_energy_hartree=float(energies[array_index]),
                provenance=provenance,
            ))
        return samples, rejections


def _samples_from_embedded_metadata(
    case_record: Mapping[str, Any],
    spec: HamiltonianDatasetSpec,
    split_assignments: Mapping[str, str],
    study_report: Mapping[str, Any],
) -> Optional[Tuple[List[HamiltonianSample], List[HamiltonianSampleRejection]]]:
    """Build indexes from a self-describing TaskReport without opening remote NPZ files."""

    molecule_id = _case_molecule_id(case_record)
    task_report = case_record['task_report']
    structured = task_report.get('structured_results')
    trajectory = structured.get('trajectory') if isinstance(structured, dict) else None
    if not isinstance(trajectory, dict):
        return None
    atomic_numbers_value = trajectory.get('atomic_numbers')
    frame_metadata = trajectory.get('frames')
    array_metadata = trajectory.get('array_metadata')
    if not (
        isinstance(atomic_numbers_value, list)
        and isinstance(frame_metadata, list)
        and isinstance(array_metadata, dict)
    ):
        return None
    arrays_ref = trajectory.get('frame_arrays_artifact')
    if not isinstance(arrays_ref, dict) or not arrays_ref.get('path'):
        raise ValueError('Trajectory does not reference a frame array artifact.')
    matrix_validation = trajectory.get('matrix_validation')
    if not isinstance(matrix_validation, dict) or matrix_validation.get('passed') is not True:
        raise ValueError('Trajectory matrices do not have a passing producer-side validation record.')

    expected_count = spec.geometries_per_molecule
    if len(frame_metadata) != expected_count:
        raise ValueError('Trajectory frame metadata count is inconsistent.')
    atomic_numbers = tuple(int(value) for value in atomic_numbers_value)
    fock_metadata = array_metadata.get('fock_matrices')
    overlap_metadata = array_metadata.get('overlap_matrices')
    if not isinstance(fock_metadata, dict) or not isinstance(overlap_metadata, dict):
        raise ValueError('Trajectory array metadata does not describe Fock and overlap matrices.')
    fock_shape = tuple(int(value) for value in fock_metadata.get('shape') or ())
    overlap_shape = tuple(int(value) for value in overlap_metadata.get('shape') or ())
    if (
        len(fock_shape) != 3
        or fock_shape != overlap_shape
        or fock_shape[0] != expected_count
        or fock_shape[1] != fock_shape[2]
    ):
        raise ValueError('Trajectory matrix metadata contains inconsistent shapes.')
    if fock_metadata.get('finite') is not True or overlap_metadata.get('finite') is not True:
        raise ValueError('Trajectory matrix metadata reports NaN or infinite values.')

    ao_basis = _ao_basis_from_manifest(trajectory.get('ao_basis'), fock_shape[1])
    arrays_path = _resolve_artifact_path(arrays_ref['path'], task_report, study_report)
    matrix_shape = (fock_shape[1], fock_shape[2])
    samples: List[HamiltonianSample] = []
    rejections: List[HamiltonianSampleRejection] = []
    for array_index, frame in enumerate(frame_metadata):
        if not isinstance(frame, dict):
            raise ValueError('Trajectory frame metadata entries must be objects.')
        frame_index = int(frame.get('frame_index'))
        geometry_id = 'frame-{0:06d}'.format(frame_index)
        provenance = {
            'study_id': study_report.get('study_id'),
            'case_id': case_record.get('case_id'),
            'run_id': task_report.get('run_id'),
            'frame_index': frame_index,
            'array_index': array_index,
        }
        if frame.get('converged') is not True:
            rejections.append(HamiltonianSampleRejection(
                molecule_id=molecule_id,
                geometry_id=geometry_id,
                reasons=('scf_unconverged',),
                provenance=provenance,
            ))
            continue
        positions_value = frame.get('positions_angstrom')
        if not isinstance(positions_value, list) or len(positions_value) != len(atomic_numbers):
            raise ValueError('Trajectory frame positions do not match atomic_numbers.')
        geometry = MolecularGeometry(
            molecule_id=molecule_id,
            geometry_id=geometry_id,
            atomic_numbers=atomic_numbers,
            positions=tuple(
                tuple(float(value) for value in position)
                for position in positions_value
            ),
            coordinate_unit='Angstrom',
            source=provenance,
        )
        split = split_assignments.get(
            '{0}:{1}'.format(molecule_id, geometry_id),
            split_assignments.get(molecule_id, ''),
        )
        samples.append(HamiltonianSample(
            geometry=geometry,
            split=split,
            electronic_structure=spec.electronic_structure,
            ao_basis=ao_basis,
            fock=MatrixArtifactReference(
                path=str(arrays_path),
                array_key='fock_matrices',
                array_index=array_index,
                shape=matrix_shape,
                dtype=str(fock_metadata.get('dtype') or ''),
            ),
            overlap=MatrixArtifactReference(
                path=str(arrays_path),
                array_key='overlap_matrices',
                array_index=array_index,
                shape=matrix_shape,
                dtype=str(overlap_metadata.get('dtype') or ''),
            ),
            converged=True,
            total_energy_hartree=float(frame.get('potential_energy_hartree')),
            provenance=provenance,
        ))
    return samples, rejections


def _ao_basis_from_manifest(value: Any, expected_nao: int) -> AOBasisMetadata:
    if not isinstance(value, dict):
        raise ValueError('Trajectory manifest does not contain AO metadata.')
    metadata = AOBasisMetadata(
        convention=value.get('convention', ''),
        source_convention=value.get('source_convention', 'pyscf'),
        labels=tuple(value.get('labels') or ()),
        atom_slices=tuple(tuple(int(item) for item in pair) for pair in value.get('atom_slices') or ()),
        source_indices=tuple(int(item) for item in value.get('source_indices') or ()),
        phase_signs=tuple(int(item) for item in value.get('phase_signs') or ()),
    )
    if metadata.nao != expected_nao:
        raise ValueError('AO metadata size does not match matrix dimensions.')
    return metadata


def _split_assignments(
    case_records: Sequence[Mapping[str, Any]],
    spec: HamiltonianDatasetSpec,
) -> Dict[str, str]:
    molecule_ids = [_case_molecule_id(case) for case in case_records]
    if len(set(molecule_ids)) != len(molecule_ids):
        raise ValueError('Study report contains duplicate molecule_id values.')
    if spec.split_protocol == 'geometry_random':
        identities = [
            '{0}:frame-{1:06d}'.format(molecule_id, frame_index)
            for molecule_id in molecule_ids
            for frame_index in spec.molecular_dynamics.sampled_frame_indices
        ]
        random.Random(spec.split_seed).shuffle(identities)
        counts = _partition_counts(len(identities), spec.split_fractions)
        assignments: Dict[str, str] = {}
        cursor = 0
        for split in DATASET_SPLITS:
            for identity in identities[cursor:cursor + counts[split]]:
                assignments[identity] = split
            cursor += counts[split]
        return assignments

    counts = _partition_counts(len(molecule_ids), spec.split_fractions)
    if spec.split_protocol == 'molecule_size_ood':
        ordered = sorted(
            molecule_ids,
            key=lambda molecule_id: (_case_atom_count(case_records, molecule_id), molecule_id),
        )
    else:
        ordered = list(molecule_ids)
        random.Random(spec.split_seed).shuffle(ordered)
    assignments: Dict[str, str] = {}
    cursor = 0
    for split in DATASET_SPLITS:
        for molecule_id in ordered[cursor:cursor + counts[split]]:
            assignments[molecule_id] = split
        cursor += counts[split]
    return assignments


def _partition_counts(total: int, fractions: Mapping[str, float]) -> Dict[str, int]:
    raw = {split: total * float(fractions[split]) for split in DATASET_SPLITS}
    counts = {split: int(math.floor(raw[split])) for split in DATASET_SPLITS}
    remainder = total - sum(counts.values())
    order = sorted(
        DATASET_SPLITS,
        key=lambda split: (-(raw[split] - counts[split]), DATASET_SPLITS.index(split)),
    )
    for split in order[:remainder]:
        counts[split] += 1
    return counts


def _trajectory_rejections(
    molecule_id: str,
    spec: HamiltonianDatasetSpec,
    *,
    reasons: Tuple[str, ...],
    provenance: Dict[str, Any],
) -> List[HamiltonianSampleRejection]:
    return [
        HamiltonianSampleRejection(
            molecule_id=molecule_id,
            geometry_id='frame-{0:06d}'.format(frame_index),
            reasons=reasons,
            provenance={**copy.deepcopy(provenance), 'frame_index': frame_index},
        )
        for frame_index in spec.molecular_dynamics.sampled_frame_indices
    ]


def _case_molecule_id(case_record: Mapping[str, Any]) -> str:
    variables = case_record.get('variables') if isinstance(case_record, Mapping) else None
    molecule_id = variables.get('molecule_id') if isinstance(variables, dict) else None
    normalized = str(molecule_id or '').strip()
    if not normalized:
        raise ValueError('Each dataset Study case requires variables.molecule_id.')
    return normalized


def _case_atom_count(case_records: Sequence[Mapping[str, Any]], molecule_id: str) -> int:
    for case in case_records:
        if _case_molecule_id(case) != molecule_id:
            continue
        request = case.get('request') if isinstance(case, Mapping) else None
        atom = request.get('atom') if isinstance(request, dict) else None
        return len([item for item in str(atom or '').replace('\n', ';').split(';') if item.strip()])
    return 0


def _resolve_artifact_path(
    path: Any,
    task_report: Mapping[str, Any],
    study_report: Mapping[str, Any],
) -> Path:
    candidate = Path(str(path)).expanduser()
    if candidate.is_absolute():
        return Path(os.path.abspath(str(candidate)))
    work_dir = task_report.get('work_dir') or study_report.get('work_dir') or '.'
    return Path(os.path.abspath(str(Path(str(work_dir)).expanduser() / candidate)))


def _report_payload(value: Any) -> Dict[str, Any]:
    if hasattr(value, 'to_dict') and callable(value.to_dict):
        value = value.to_dict()
    if not isinstance(value, dict):
        raise TypeError('study_report must be a StudyReport or dictionary')
    return copy.deepcopy(value)


def _json_lines(values: Iterable[Mapping[str, Any]]) -> str:
    return ''.join(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=json_default, allow_nan=False) + '\n'
        for value in values
    )


__all__ = ['finalize_hamiltonian_dataset']
