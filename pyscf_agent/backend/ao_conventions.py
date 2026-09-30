"""Provider-owned AO convention transforms for molecular dataset artifacts."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


_QH9_SUPPORTED_ATOMIC_NUMBERS = frozenset((1, 6, 7, 8, 9))
_QH9_LOCAL_SOURCE_INDICES = {
    # def2-SVP H: 1s, 2s, 2px, 2py, 2pz
    1: (0, 1, 3, 4, 2),
    # def2-SVP C/N/O/F: 1s, 2s, 3s, two p shells, one d shell.
    6: (0, 1, 2, 4, 5, 3, 7, 8, 6, 9, 10, 11, 12, 13),
    7: (0, 1, 2, 4, 5, 3, 7, 8, 6, 9, 10, 11, 12, 13),
    8: (0, 1, 2, 4, 5, 3, 7, 8, 6, 9, 10, 11, 12, 13),
    9: (0, 1, 2, 4, 5, 3, 7, 8, 6, 9, 10, 11, 12, 13),
}


def qh9_ao_metadata(mol: Any, basis: str) -> Dict[str, Any]:
    """Return the explicit PySCF-to-QH9 AO permutation for one molecule.

    The mapping follows QH9's published def2-SVP convention: spherical p
    functions are reordered from PySCF ``x,y,z`` to ``y,z,x`` while s and d
    functions retain their source order.  All phase factors are +1.
    """

    if str(basis or '').strip().lower() != 'def2-svp':
        raise ValueError('The QH9 AO convention requires the def2-SVP basis.')

    atomic_numbers = tuple(int(value) for value in mol.atom_charges())
    unsupported = sorted(set(atomic_numbers).difference(_QH9_SUPPORTED_ATOMIC_NUMBERS))
    if unsupported:
        raise ValueError(
            'The QH9 AO convention supports atomic numbers 1, 6, 7, 8, and 9; '
            'received unsupported values {0}.'.format(unsupported)
        )

    source_labels = tuple(mol.ao_labels(fmt=False))
    atom_ao_slices = mol.aoslice_by_atom()
    source_indices: List[int] = []
    atom_slices: List[Tuple[int, int]] = []
    cursor = 0
    for atom_index, atomic_number in enumerate(atomic_numbers):
        source_start = int(atom_ao_slices[atom_index][2])
        source_stop = int(atom_ao_slices[atom_index][3])
        local_permutation = _QH9_LOCAL_SOURCE_INDICES[atomic_number]
        source_count = source_stop - source_start
        if source_count != len(local_permutation):
            raise ValueError(
                'Unexpected def2-SVP AO count for atom {0}: expected {1}, received {2}.'.format(
                    atom_index,
                    len(local_permutation),
                    source_count,
                )
            )
        source_indices.extend(source_start + local for local in local_permutation)
        atom_slices.append((cursor, cursor + source_count))
        cursor += source_count

    labels = tuple(_format_ao_label(source_labels[index]) for index in source_indices)
    return {
        'convention': 'qh9',
        'source_convention': 'pyscf',
        'labels': labels,
        'atom_slices': tuple(atom_slices),
        'source_indices': tuple(source_indices),
        'phase_signs': (1,) * len(source_indices),
        'nao': len(source_indices),
    }


def transform_ao_matrix(
    matrix: Any,
    source_indices: Sequence[int],
    phase_signs: Sequence[int],
) -> Any:
    """Transform one square AO matrix into an explicit target convention."""

    import numpy as np  # pylint: disable=import-outside-toplevel

    values = np.asarray(matrix)
    indices = np.asarray(tuple(source_indices), dtype=int)
    signs = np.asarray(tuple(phase_signs), dtype=values.dtype)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError('AO matrices must be square two-dimensional arrays.')
    if len(indices) != values.shape[0] or set(indices.tolist()) != set(range(values.shape[0])):
        raise ValueError('source_indices must permute every AO matrix index exactly once.')
    if len(signs) != values.shape[0] or not np.all(np.isin(signs, (-1, 1))):
        raise ValueError('phase_signs must contain one -1 or +1 value per AO.')
    transformed = values[np.ix_(indices, indices)]
    return transformed * signs[:, None] * signs[None, :]


def _format_ao_label(label: Sequence[Any]) -> str:
    atom_index, symbol, shell, component = label
    suffix = str(component or '')
    return '{0}:{1}:{2}{3}'.format(int(atom_index), symbol, shell, suffix)


__all__ = ['qh9_ao_metadata', 'transform_ao_matrix']
