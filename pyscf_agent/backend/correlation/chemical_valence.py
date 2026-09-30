from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


_SHELL_PATTERN = re.compile(r'^(\d+)([spdf])', re.IGNORECASE)
_SHELL_CAPACITY = {'s': 1, 'p': 3, 'd': 5, 'f': 7}
_SHELL_FIRST_PRINCIPAL_QUANTUM_NUMBER = {'s': 1, 'p': 2, 'd': 3, 'f': 4}


def _period(atomic_number: int) -> int:
    for period, upper_bound in enumerate((2, 10, 18, 36, 54, 86, 118), start=1):
        if atomic_number <= upper_bound:
            return period
    return 7


def _element_charge(symbol: str) -> Optional[int]:
    try:
        from pyscf.data import elements  # pylint: disable=import-outside-toplevel

        return int(elements.charge(symbol))
    except Exception:
        return None


def _element_configuration(atomic_number: int) -> Optional[List[int]]:
    try:
        from pyscf.data import elements  # pylint: disable=import-outside-toplevel

        return [int(value) for value in elements.CONFIGURATION[atomic_number]]
    except Exception:
        return None


def _shell_electron_count(atomic_number: int, shell: str) -> Optional[int]:
    match = _SHELL_PATTERN.match(str(shell or '').strip())
    configuration = _element_configuration(atomic_number)
    if match is None or configuration is None:
        return None
    principal = int(match.group(1))
    angular = match.group(2).lower()
    angular_index = 'spdf'.index(angular)
    capacity = 2 * _SHELL_CAPACITY[angular]
    inner_shell_count = max(
        0,
        principal - _SHELL_FIRST_PRINCIPAL_QUANTUM_NUMBER[angular],
    )
    electrons = configuration[angular_index] - capacity * inner_shell_count
    return max(0, min(capacity, int(electrons)))


def _atom_counts(mol: Any, available: Dict[str, List[str]]) -> Dict[str, int]:
    symbols: List[str] = []
    try:
        symbols = [str(mol.atom_symbol(index)).strip() for index in range(int(mol.natm))]
    except Exception:
        labels = mol.ao_labels(fmt=False) if mol is not None and hasattr(mol, 'ao_labels') else []
        atom_symbols: Dict[int, str] = {}
        for label in labels:
            if not isinstance(label, (tuple, list)) or len(label) < 2:
                continue
            try:
                atom_index = int(label[0])
            except (TypeError, ValueError):
                continue
            atom_symbols[atom_index] = str(label[1]).strip()
        symbols = [atom_symbols[index] for index in sorted(atom_symbols)]
    counts = Counter(symbol for symbol in symbols if symbol)
    return {
        symbol: int(counts.get(symbol) or 1)
        for symbol in available
    }


def _primary_valence_shells(atomic_number: int) -> Tuple[List[str], List[str]]:
    """Return primary and extended valence shells for a neutral element.

    The primary manifold follows the orbitals most commonly responsible for
    bond breaking and local moments.  The extended manifold is retained as
    provenance for a later user-approved expansion.
    """

    period = _period(atomic_number)
    if 57 <= atomic_number <= 71 or 89 <= atomic_number <= 103:
        primary = [f'{period - 2}f', f'{period - 1}d', f'{period}s']
        return primary, primary
    if 21 <= atomic_number <= 30 or 39 <= atomic_number <= 48 or 72 <= atomic_number <= 80 or 104 <= atomic_number <= 112:
        primary = [f'{period - 1}d', f'{period}s']
        return primary, primary
    if atomic_number in (1, 2) or atomic_number in (3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88):
        primary = [f'{period}s']
        return primary, primary
    primary = [f'{period}p']
    return primary, [f'{period}s', f'{period}p']


def _available_shells(mol: Any) -> Dict[str, List[str]]:
    shells: Dict[str, List[str]] = {}
    labels = mol.ao_labels(fmt=False) if mol is not None and hasattr(mol, 'ao_labels') else []
    for label in labels:
        if not isinstance(label, (tuple, list)) or len(label) < 3:
            continue
        symbol = str(label[1]).strip()
        match = _SHELL_PATTERN.match(str(label[2]).strip())
        if not symbol or not match:
            continue
        shell = ''.join(match.groups()).lower()
        shells.setdefault(symbol, [])
        if shell not in shells[symbol]:
            shells[symbol].append(shell)
    return shells


def chemical_valence_targets(mol: Any) -> Dict[str, Any]:
    """Build element-resolved AVAS targets without molecule-specific rules."""

    available = _available_shells(mol)
    primary_targets: List[str] = []
    extended_targets: List[str] = []
    elements: Dict[str, Dict[str, Any]] = {}
    atom_counts = _atom_counts(mol, available)
    primary_orbital_count = 0
    primary_electron_count = 0
    electron_count_available = True
    for symbol, shells in available.items():
        atomic_number = _element_charge(symbol)
        if atomic_number is None:
            primary_shells = shells[-1:]
            extended_shells = list(shells)
        else:
            requested_primary, requested_extended = _primary_valence_shells(atomic_number)
            primary_shells = [shell for shell in requested_primary if shell in shells]
            extended_shells = [shell for shell in requested_extended if shell in shells]
        if not primary_shells and shells:
            primary_shells = shells[-1:]
        if not extended_shells:
            extended_shells = list(primary_shells)
        for shell in primary_shells:
            primary_targets.append(f'{symbol} {shell}')
        for shell in extended_shells:
            extended_targets.append(f'{symbol} {shell}')
        atom_count = int(atom_counts.get(symbol) or 1)
        element_orbital_count = sum(
            _SHELL_CAPACITY.get(shell[-1].lower(), 0)
            for shell in primary_shells
        )
        shell_electron_counts = {
            shell: _shell_electron_count(atomic_number, shell)
            if atomic_number is not None
            else None
            for shell in primary_shells
        }
        if any(value is None for value in shell_electron_counts.values()):
            electron_count_available = False
            element_electron_count = None
        else:
            element_electron_count = sum(int(value) for value in shell_electron_counts.values())
            primary_electron_count += atom_count * element_electron_count
        primary_orbital_count += atom_count * element_orbital_count
        elements[symbol] = {
            'atomic_number': atomic_number,
            'atom_count': atom_count,
            'available_shells': list(shells),
            'primary_shells': primary_shells,
            'extended_shells': extended_shells,
            'primary_orbital_count_per_atom': element_orbital_count,
            'primary_electron_count_per_atom': element_electron_count,
            'primary_shell_electron_counts': shell_electron_counts,
        }
    primary_targets = list(dict.fromkeys(primary_targets))
    extended_targets = list(dict.fromkeys(extended_targets))
    if electron_count_available:
        try:
            primary_electron_count -= int(getattr(mol, 'charge', 0) or 0)
        except (TypeError, ValueError):
            pass
        primary_electron_count = max(0, min(2 * primary_orbital_count, primary_electron_count))
    return {
        'status': 'available' if primary_targets else 'unavailable',
        'strategy': 'element_valence_manifold',
        'primary_targets': primary_targets,
        'extended_targets': extended_targets,
        'primary_orbital_count': primary_orbital_count or None,
        'primary_electron_count': primary_electron_count if electron_count_available else None,
        'elements': elements,
        'note': (
            'Primary targets preserve complete element valence manifolds; the extended targets are '
            'recorded for user-approved expansion when the primary space is insufficient.'
        ),
    }


__all__ = ['chemical_valence_targets']
