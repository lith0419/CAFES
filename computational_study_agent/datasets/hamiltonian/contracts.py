"""Public contracts for QH9-compatible molecular Hamiltonian datasets.

The contracts deliberately separate scientific metadata from array storage.
They describe a campaign, one accepted sample, one rejected sample, and the
ports that later PySCF and storage implementations must satisfy.  This module
does not execute calculations or write dataset files.
"""

from __future__ import annotations

import copy
import math

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Tuple, runtime_checkable

from pyscf_agent.input_validation import boolean
from pyscf_agent.registry import default_registry


HAMILTONIAN_DATASET_SPEC_SCHEMA = 'pyscf-agent.hamiltonian-dataset-spec.v1'
HAMILTONIAN_SAMPLE_SCHEMA = 'pyscf-agent.hamiltonian-sample.v1'
HAMILTONIAN_REJECTION_SCHEMA = 'pyscf-agent.hamiltonian-sample-rejection.v1'
HAMILTONIAN_MANIFEST_SCHEMA = 'pyscf-agent.hamiltonian-dataset-manifest.v1'

DATASET_SPLITS = ('train', 'validation', 'test')
SPLIT_PROTOCOLS = ('molecule_random', 'geometry_random', 'molecule_size_ood')
_MD_PROFILES = default_registry().capability(
    'molecular_dynamics', namespace='molecular.job',
).metadata['profiles']
QH9_REFERENCE_SCF_OPTIONS = copy.deepcopy(_MD_PROFILES['qh9']['runtime'])
QH9_RELAXED_SCF_OPTIONS = copy.deepcopy(_MD_PROFILES['qh9_relaxed_scf']['runtime'])
# Backward-compatible name for the executable campaign defaults.  Exact QH9
# provenance remains available separately through QH9_REFERENCE_SCF_OPTIONS.
QH9_SCF_OPTIONS = QH9_RELAXED_SCF_OPTIONS


def _required_text(value: Any, field_name: str) -> str:
    normalized = str(value or '').strip()
    if not normalized:
        raise ValueError('{0} must be non-empty'.format(field_name))
    return normalized


def _positive_integer(value: Any, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError('{0} must be a positive integer'.format(field_name))
    return value


def _non_negative_integer(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError('{0} must be a non-negative integer'.format(field_name))
    return value


def _normalized_fractions(values: Mapping[str, Any]) -> Dict[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError('split_fractions must be a mapping')
    if set(values) != set(DATASET_SPLITS):
        raise ValueError(
            'split_fractions must define exactly: {0}'.format(
                ', '.join(DATASET_SPLITS)
            )
        )
    normalized = {name: float(values[name]) for name in DATASET_SPLITS}
    if any(not math.isfinite(value) or value < 0.0 for value in normalized.values()):
        raise ValueError('split_fractions must be finite and non-negative')
    if not math.isclose(sum(normalized.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError('split_fractions must sum to 1.0')
    return normalized


@dataclass(frozen=True)
class ElectronicStructureSpec:
    """Homogeneous electronic-structure settings for one dataset."""

    method: str = 'dft'
    xc: str = 'b3lyp'
    basis: str = 'def2-svp'
    restricted: bool = True
    scf_options: Dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(QH9_SCF_OPTIONS)
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, 'restricted', boolean(self.restricted, 'restricted'))
        object.__setattr__(self, 'method', _required_text(self.method, 'method').lower())
        object.__setattr__(self, 'basis', _required_text(self.basis, 'basis'))
        object.__setattr__(self, 'xc', str(self.xc or '').strip().lower())
        if self.method == 'dft' and not self.xc:
            raise ValueError('xc must be non-empty for DFT datasets')
        if not isinstance(self.scf_options, dict):
            raise TypeError('scf_options must be a dictionary')

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> 'ElectronicStructureSpec':
        return cls(
            method=payload.get('method', 'dft'),
            xc=payload.get('xc', 'b3lyp'),
            basis=payload.get('basis', 'def2-svp'),
            restricted=payload.get('restricted', True),
            scf_options=copy.deepcopy(
                payload['scf_options']
                if 'scf_options' in payload
                else QH9_SCF_OPTIONS
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MolecularDynamicsSamplingSpec:
    """Trajectory protocol used to produce several geometries per molecule."""

    ensemble: str = 'nve'
    temperature_kelvin: float = 300.0
    time_step_au: float = 50.0
    steps: int = 100
    sample_stride: int = 10
    sample_offset: int = 9
    base_velocity_seed: int = 0

    def __post_init__(self) -> None:
        ensemble = _required_text(self.ensemble, 'ensemble').lower()
        if ensemble != 'nve':
            raise ValueError('The QH9 sampling profile currently supports NVE only')
        object.__setattr__(self, 'ensemble', ensemble)
        if not math.isfinite(float(self.temperature_kelvin)) or self.temperature_kelvin <= 0:
            raise ValueError('temperature_kelvin must be finite and positive')
        if not math.isfinite(float(self.time_step_au)) or self.time_step_au <= 0:
            raise ValueError('time_step_au must be finite and positive')
        _positive_integer(self.steps, 'steps')
        _positive_integer(self.sample_stride, 'sample_stride')
        _non_negative_integer(self.sample_offset, 'sample_offset')
        _non_negative_integer(self.base_velocity_seed, 'base_velocity_seed')
        if self.sample_offset >= self.steps:
            raise ValueError('sample_offset must be smaller than steps')

    @property
    def sampled_frame_indices(self) -> Tuple[int, ...]:
        return tuple(range(self.sample_offset, self.steps, self.sample_stride))

    @property
    def sampled_frame_count(self) -> int:
        return len(self.sampled_frame_indices)

    def velocity_seed_for_case(self, case_index: int) -> int:
        _non_negative_integer(case_index, 'case_index')
        return self.base_velocity_seed + case_index

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> 'MolecularDynamicsSamplingSpec':
        return cls(
            ensemble=payload.get('ensemble', 'nve'),
            temperature_kelvin=float(payload.get('temperature_kelvin', 300.0)),
            time_step_au=float(payload.get('time_step_au', 50.0)),
            steps=int(payload.get('steps', 100)),
            sample_stride=int(payload.get('sample_stride', 10)),
            sample_offset=int(payload.get('sample_offset', 9)),
            base_velocity_seed=int(payload.get('base_velocity_seed', 0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload['sampled_frame_indices'] = list(self.sampled_frame_indices)
        payload['sampled_frame_count'] = self.sampled_frame_count
        return payload


@dataclass(frozen=True)
class HamiltonianDatasetSpec:
    """Design-time contract for an approximately 100 x 10 campaign.

    ``molecule_random`` is the default because all geometries belonging to one
    molecule must remain in one split.  ``geometry_random`` remains available
    for the QH9 geometry-generalization protocol and must be selected explicitly.
    """

    dataset_id: str
    name: str
    target_molecule_count: int = 100
    geometries_per_molecule: int = 10
    electronic_structure: ElectronicStructureSpec = field(
        default_factory=ElectronicStructureSpec
    )
    molecular_dynamics: MolecularDynamicsSamplingSpec = field(
        default_factory=MolecularDynamicsSamplingSpec
    )
    compatibility_profile: str = 'qh9_relaxed_scf'
    ao_convention: str = 'qh9'
    coordinate_unit: str = 'Angstrom'
    split_protocol: str = 'molecule_random'
    split_fractions: Dict[str, float] = field(
        default_factory=lambda: {
            'train': 0.8,
            'validation': 0.1,
            'test': 0.1,
        }
    )
    split_seed: int = 0
    require_converged: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, 'dataset_id', _required_text(self.dataset_id, 'dataset_id'))
        object.__setattr__(self, 'name', _required_text(self.name, 'name'))
        _positive_integer(self.target_molecule_count, 'target_molecule_count')
        _positive_integer(self.geometries_per_molecule, 'geometries_per_molecule')
        if not isinstance(self.electronic_structure, ElectronicStructureSpec):
            raise TypeError('electronic_structure must be an ElectronicStructureSpec')
        if not isinstance(self.molecular_dynamics, MolecularDynamicsSamplingSpec):
            raise TypeError('molecular_dynamics must be a MolecularDynamicsSamplingSpec')
        if self.molecular_dynamics.sampled_frame_count != self.geometries_per_molecule:
            raise ValueError(
                'molecular_dynamics must sample exactly geometries_per_molecule frames'
            )
        profile = _required_text(self.compatibility_profile, 'compatibility_profile').lower()
        convention = _required_text(self.ao_convention, 'ao_convention').lower()
        protocol = _required_text(self.split_protocol, 'split_protocol').lower()
        if protocol not in SPLIT_PROTOCOLS:
            raise ValueError(
                'split_protocol must be one of: {0}'.format(', '.join(SPLIT_PROTOCOLS))
            )
        object.__setattr__(self, 'compatibility_profile', profile)
        object.__setattr__(self, 'ao_convention', convention)
        object.__setattr__(self, 'coordinate_unit', _required_text(self.coordinate_unit, 'coordinate_unit'))
        object.__setattr__(self, 'split_protocol', protocol)
        object.__setattr__(self, 'split_fractions', _normalized_fractions(self.split_fractions))
        _non_negative_integer(self.split_seed, 'split_seed')
        if profile in ('qh9', 'qh9_relaxed_scf'):
            electronic_structure = self.electronic_structure
            if (
                electronic_structure.method != 'dft'
                or electronic_structure.xc != 'b3lyp'
                or electronic_structure.basis.lower() != 'def2-svp'
                or electronic_structure.restricted is not True
            ):
                raise ValueError(
                    'The {0} compatibility profile requires restricted B3LYP/def2-SVP'.format(profile)
                )
            if convention != 'qh9':
                raise ValueError('The {0} compatibility profile requires ao_convention="qh9"'.format(profile))
            expected_scf_options = (
                QH9_REFERENCE_SCF_OPTIONS
                if profile == 'qh9'
                else QH9_RELAXED_SCF_OPTIONS
            )
            for field_name, expected in expected_scf_options.items():
                observed = electronic_structure.scf_options.get(field_name)
                if observed is None or not math.isclose(
                    float(observed),
                    float(expected),
                    rel_tol=0.0,
                    abs_tol=1e-18,
                ):
                    raise ValueError(
                        'The {0} compatibility profile requires scf_options.{1}={2}'.format(
                            profile,
                            field_name,
                            expected,
                        )
                    )

    @property
    def target_structure_count(self) -> int:
        return self.target_molecule_count * self.geometries_per_molecule

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> 'HamiltonianDatasetSpec':
        if payload.get('schema') not in (None, HAMILTONIAN_DATASET_SPEC_SCHEMA):
            raise ValueError('Unsupported HamiltonianDatasetSpec schema')
        profile = str(payload.get('compatibility_profile', 'qh9_relaxed_scf')).strip().lower()
        electronic_structure = copy.deepcopy(payload.get('electronic_structure') or {})
        if 'scf_options' not in electronic_structure and profile in _MD_PROFILES:
            electronic_structure['scf_options'] = copy.deepcopy(_MD_PROFILES[profile]['runtime'])
        return cls(
            dataset_id=payload.get('dataset_id', ''),
            name=payload.get('name', ''),
            target_molecule_count=int(payload.get('target_molecule_count', 100)),
            geometries_per_molecule=int(payload.get('geometries_per_molecule', 10)),
            electronic_structure=ElectronicStructureSpec.from_dict(electronic_structure),
            molecular_dynamics=MolecularDynamicsSamplingSpec.from_dict(
                payload.get('molecular_dynamics') or {}
            ),
            compatibility_profile=profile,
            ao_convention=payload.get('ao_convention', 'qh9'),
            coordinate_unit=payload.get('coordinate_unit', 'Angstrom'),
            split_protocol=payload.get('split_protocol', 'molecule_random'),
            split_fractions=copy.deepcopy(payload.get('split_fractions') or {
                'train': 0.8,
                'validation': 0.1,
                'test': 0.1,
            }),
            split_seed=int(payload.get('split_seed', 0)),
            require_converged=bool(payload.get('require_converged', True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload['electronic_structure'] = self.electronic_structure.to_dict()
        payload['molecular_dynamics'] = self.molecular_dynamics.to_dict()
        payload['schema'] = HAMILTONIAN_DATASET_SPEC_SCHEMA
        payload['target_structure_count'] = self.target_structure_count
        return payload


@dataclass(frozen=True)
class MolecularGeometry:
    """One geometry identified within a molecule, without content hashing."""

    molecule_id: str
    geometry_id: str
    atomic_numbers: Tuple[int, ...]
    positions: Tuple[Tuple[float, float, float], ...]
    coordinate_unit: str = 'Angstrom'
    charge: int = 0
    spin: int = 0
    source: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'molecule_id', _required_text(self.molecule_id, 'molecule_id'))
        object.__setattr__(self, 'geometry_id', _required_text(self.geometry_id, 'geometry_id'))
        object.__setattr__(self, 'coordinate_unit', _required_text(self.coordinate_unit, 'coordinate_unit'))
        if not self.atomic_numbers:
            raise ValueError('atomic_numbers must be non-empty')
        if len(self.atomic_numbers) != len(self.positions):
            raise ValueError('atomic_numbers and positions must have equal length')
        if any(type(value) is not int or value <= 0 for value in self.atomic_numbers):
            raise ValueError('atomic_numbers must contain positive integers')
        for position in self.positions:
            if len(position) != 3:
                raise ValueError('each position must contain exactly three coordinates')
            if any(not math.isfinite(float(value)) for value in position):
                raise ValueError('positions must contain only finite coordinates')
        if not isinstance(self.source, dict):
            raise TypeError('source must be a dictionary')

    @property
    def sample_id(self) -> str:
        return '{0}:{1}'.format(self.molecule_id, self.geometry_id)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> 'MolecularGeometry':
        if not isinstance(payload, Mapping):
            raise TypeError('MolecularGeometry payload must be a mapping')
        return cls(
            molecule_id=payload.get('molecule_id', ''),
            geometry_id=payload.get('geometry_id', 'seed'),
            atomic_numbers=tuple(int(value) for value in payload.get('atomic_numbers') or ()),
            positions=tuple(
                tuple(float(value) for value in position)
                for position in payload.get('positions') or ()
            ),
            coordinate_unit=payload.get('coordinate_unit', 'Angstrom'),
            charge=int(payload.get('charge', 0)),
            spin=int(payload.get('spin', 0)),
            source=copy.deepcopy(payload.get('source') or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload['sample_id'] = self.sample_id
        return payload


@dataclass(frozen=True)
class AOBasisMetadata:
    """Auditable mapping from the PySCF AO basis into the target convention.

    Target AO ``i`` is obtained from source AO ``source_indices[i]`` and
    multiplied by ``phase_signs[i]``.  Empty mappings mean identity ordering
    and phases and are normalized into explicit tuples during construction.
    """

    convention: str
    labels: Tuple[str, ...]
    atom_slices: Tuple[Tuple[int, int], ...]
    source_convention: str = 'pyscf'
    source_indices: Tuple[int, ...] = ()
    phase_signs: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, 'convention', _required_text(self.convention, 'convention').lower())
        object.__setattr__(self, 'source_convention', _required_text(self.source_convention, 'source_convention').lower())
        if not self.labels:
            raise ValueError('labels must be non-empty')
        if not self.atom_slices:
            raise ValueError('atom_slices must be non-empty')
        cursor = 0
        for start, stop in self.atom_slices:
            if start != cursor or stop <= start:
                raise ValueError('atom_slices must be positive, contiguous AO ranges')
            cursor = stop
        if cursor != len(self.labels):
            raise ValueError('atom_slices must cover every AO label exactly once')
        source_indices = self.source_indices or tuple(range(len(self.labels)))
        phase_signs = self.phase_signs or (1,) * len(self.labels)
        if len(source_indices) != len(self.labels) or set(source_indices) != set(range(len(self.labels))):
            raise ValueError('source_indices must be a permutation of all AO indices')
        if len(phase_signs) != len(self.labels) or any(sign not in (-1, 1) for sign in phase_signs):
            raise ValueError('phase_signs must contain one -1 or +1 value per AO')
        object.__setattr__(self, 'source_indices', tuple(source_indices))
        object.__setattr__(self, 'phase_signs', tuple(phase_signs))

    @property
    def nao(self) -> int:
        return len(self.labels)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload['nao'] = self.nao
        return payload


@dataclass(frozen=True)
class MatrixArtifactReference:
    """Location and array metadata for one persisted matrix; no checksum field."""

    path: str
    array_key: str
    shape: Tuple[int, int]
    dtype: str
    format: str = 'npz'
    array_index: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'path', _required_text(self.path, 'path'))
        object.__setattr__(self, 'array_key', _required_text(self.array_key, 'array_key'))
        object.__setattr__(self, 'dtype', _required_text(self.dtype, 'dtype'))
        object.__setattr__(self, 'format', _required_text(self.format, 'format').lower())
        if len(self.shape) != 2 or any(type(value) is not int or value <= 0 for value in self.shape):
            raise ValueError('shape must contain two positive dimensions')
        if self.shape[0] != self.shape[1]:
            raise ValueError('Hamiltonian matrix artifacts must be square')
        if self.array_index is not None:
            _non_negative_integer(self.array_index, 'array_index')

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HamiltonianSample:
    """Accepted sample whose matrices have passed the dataset quality gate."""

    geometry: MolecularGeometry
    split: str
    electronic_structure: ElectronicStructureSpec
    ao_basis: AOBasisMetadata
    fock: MatrixArtifactReference
    overlap: MatrixArtifactReference
    converged: bool
    total_energy_hartree: float
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        split = str(self.split or '').strip().lower()
        if split not in DATASET_SPLITS:
            raise ValueError('split must be one of: {0}'.format(', '.join(DATASET_SPLITS)))
        object.__setattr__(self, 'split', split)
        if self.converged is not True:
            raise ValueError('accepted Hamiltonian samples must be converged')
        if not math.isfinite(float(self.total_energy_hartree)):
            raise ValueError('total_energy_hartree must be finite')
        if len(self.ao_basis.atom_slices) != len(self.geometry.atomic_numbers):
            raise ValueError('ao_basis must define one atom slice per atom')
        expected_shape = (self.ao_basis.nao, self.ao_basis.nao)
        if self.fock.shape != expected_shape or self.overlap.shape != expected_shape:
            raise ValueError('Fock and overlap shapes must match ao_basis.nao')
        if not isinstance(self.provenance, dict):
            raise TypeError('provenance must be a dictionary')

    @property
    def sample_id(self) -> str:
        return self.geometry.sample_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            'schema': HAMILTONIAN_SAMPLE_SCHEMA,
            'sample_id': self.sample_id,
            'molecule_id': self.geometry.molecule_id,
            'geometry_id': self.geometry.geometry_id,
            'split': self.split,
            'geometry': self.geometry.to_dict(),
            'electronic_structure': self.electronic_structure.to_dict(),
            'ao_basis': self.ao_basis.to_dict(),
            'fock': self.fock.to_dict(),
            'overlap': self.overlap.to_dict(),
            'converged': self.converged,
            'total_energy_hartree': float(self.total_energy_hartree),
            'provenance': copy.deepcopy(self.provenance),
        }


@dataclass(frozen=True)
class HamiltonianSampleRejection:
    """Explicit record for a structure excluded by execution or quality gates."""

    molecule_id: str
    geometry_id: str
    reasons: Tuple[str, ...]
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'molecule_id', _required_text(self.molecule_id, 'molecule_id'))
        object.__setattr__(self, 'geometry_id', _required_text(self.geometry_id, 'geometry_id'))
        normalized = tuple(dict.fromkeys(str(reason).strip() for reason in self.reasons if str(reason).strip()))
        if not normalized:
            raise ValueError('reasons must contain at least one non-empty reason')
        object.__setattr__(self, 'reasons', normalized)
        if not isinstance(self.provenance, dict):
            raise TypeError('provenance must be a dictionary')

    @property
    def sample_id(self) -> str:
        return '{0}:{1}'.format(self.molecule_id, self.geometry_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'schema': HAMILTONIAN_REJECTION_SCHEMA,
            'sample_id': self.sample_id,
            'molecule_id': self.molecule_id,
            'geometry_id': self.geometry_id,
            'reasons': list(self.reasons),
            'provenance': copy.deepcopy(self.provenance),
        }


@dataclass(frozen=True)
class HamiltonianDatasetManifest:
    """Completion summary and artifact locations for one dataset build."""

    spec: HamiltonianDatasetSpec
    molecule_count: int
    accepted_structure_count: int
    rejected_structure_count: int
    split_counts: Dict[str, int]
    artifacts: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name, value in (
            ('molecule_count', self.molecule_count),
            ('accepted_structure_count', self.accepted_structure_count),
            ('rejected_structure_count', self.rejected_structure_count),
        ):
            _non_negative_integer(value, field_name)
        if set(self.split_counts) != set(DATASET_SPLITS):
            raise ValueError('split_counts must define train, validation, and test')
        if any(type(value) is not int or value < 0 for value in self.split_counts.values()):
            raise ValueError('split_counts must contain non-negative integers')
        if sum(self.split_counts.values()) != self.accepted_structure_count:
            raise ValueError('split_counts must sum to accepted_structure_count')
        if not isinstance(self.artifacts, dict):
            raise TypeError('artifacts must be a dictionary')

    @property
    def accounted_structure_count(self) -> int:
        return self.accepted_structure_count + self.rejected_structure_count

    @property
    def missing_structure_count(self) -> int:
        return max(0, self.spec.target_structure_count - self.accounted_structure_count)

    @property
    def status(self) -> str:
        return 'complete' if self.missing_structure_count == 0 else 'partial'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'schema': HAMILTONIAN_MANIFEST_SCHEMA,
            'spec': self.spec.to_dict(),
            'target_structure_count': self.spec.target_structure_count,
            'molecule_count': self.molecule_count,
            'accepted_structure_count': self.accepted_structure_count,
            'rejected_structure_count': self.rejected_structure_count,
            'accounted_structure_count': self.accounted_structure_count,
            'missing_structure_count': self.missing_structure_count,
            'status': self.status,
            'split_counts': copy.deepcopy(self.split_counts),
            'artifacts': copy.deepcopy(self.artifacts),
        }


@dataclass(frozen=True)
class HamiltonianMatrices:
    """In-memory result exchanged between an extractor and a sample store."""

    fock: Any
    overlap: Any
    ao_basis: AOBasisMetadata
    converged: bool
    total_energy_hartree: float
    provenance: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class HamiltonianLabelExtractor(Protocol):
    """Port implemented by the PySCF molecular backend."""

    def extract(
        self,
        *,
        geometry: MolecularGeometry,
        electronic_structure: ElectronicStructureSpec,
        mean_field: Any,
    ) -> HamiltonianMatrices:
        """Extract final AO-basis matrices from a completed mean-field object."""


@runtime_checkable
class HamiltonianDatasetStore(Protocol):
    """Port implemented by NPZ/HDF5 or another dataset storage backend."""

    def write_sample(
        self,
        *,
        geometry: MolecularGeometry,
        split: str,
        electronic_structure: ElectronicStructureSpec,
        matrices: HamiltonianMatrices,
    ) -> HamiltonianSample:
        """Persist one accepted sample and return its artifact references."""

    def write_rejection(self, rejection: HamiltonianSampleRejection) -> None:
        """Persist one explicit rejection record."""

    def finalize(
        self,
        *,
        spec: HamiltonianDatasetSpec,
        samples: Iterable[HamiltonianSample],
        rejections: Iterable[HamiltonianSampleRejection],
    ) -> HamiltonianDatasetManifest:
        """Finalize indexes and return the dataset manifest."""


__all__ = [
    'AOBasisMetadata',
    'DATASET_SPLITS',
    'ElectronicStructureSpec',
    'HAMILTONIAN_DATASET_SPEC_SCHEMA',
    'HAMILTONIAN_MANIFEST_SCHEMA',
    'HAMILTONIAN_REJECTION_SCHEMA',
    'HAMILTONIAN_SAMPLE_SCHEMA',
    'HamiltonianDatasetManifest',
    'HamiltonianDatasetSpec',
    'HamiltonianDatasetStore',
    'HamiltonianLabelExtractor',
    'HamiltonianMatrices',
    'HamiltonianSample',
    'HamiltonianSampleRejection',
    'MatrixArtifactReference',
    'MolecularDynamicsSamplingSpec',
    'MolecularGeometry',
    'QH9_SCF_OPTIONS',
    'QH9_REFERENCE_SCF_OPTIONS',
    'QH9_RELAXED_SCF_OPTIONS',
    'SPLIT_PROTOCOLS',
]
