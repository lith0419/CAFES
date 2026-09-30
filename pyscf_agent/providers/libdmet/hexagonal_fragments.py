"""Disjoint elementary honeycomb rings, preserving Builder site identities."""
from __future__ import annotations

from typing import Any, Mapping

from .translation import resolve_primitive_cell_layout


def honeycomb_hexagonal_fragments(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Find contractible six-cycles and an exact cover of the periodic graph.

    Cell displacement distinguishes a plaquette from a six-step loop winding
    around a small periodic lattice. Neither site numbering nor drawing order
    defines the fragment. Parameters on the original sites/bonds are untouched.
    """
    if spec.get('preset') != 'honeycomb' or int(spec.get('dimension', 0)) != 2:
        raise ValueError('honeycomb_hexagon requires a two-dimensional honeycomb model')
    if spec.get('boundary') != 'periodic':
        raise ValueError('honeycomb_hexagon currently requires periodic boundary conditions')
    sites = {int(site['id']): site for site in spec.get('sites', [])}
    if len(sites) % 6:
        raise ValueError(
            f'honeycomb_hexagon cannot cover {len(sites)} sites with disjoint six-site rings. '
            'Use a compatible periodic lattice, such as 3x3 or 6x6 primitive cells '
            '(18 or 72 sites); a 4x4 honeycomb has 32 sites.'
        )
    layout, issues = resolve_primitive_cell_layout(list(sites.values()))
    if issues or layout is None or layout['dimension'] != 2 or layout['basis_size'] != 2:
        raise ValueError('honeycomb_hexagon requires complete two-site Builder primitive-cell metadata')
    lengths = layout['lattice_shape']
    if any(length < 3 for length in lengths):
        raise ValueError('honeycomb_hexagon requires at least three primitive cells in each direction')
    adjacency: dict[int, set[int]] = {site_id: set() for site_id in sites}
    for bond in spec.get('bonds', []):
        source, target = int(bond['source']), int(bond['target'])
        if source not in sites or target not in sites or source == target:
            raise ValueError('honeycomb_hexagon requires valid nearest-neighbor honeycomb bonds')
        adjacency[source].add(target)
        adjacency[target].add(source)
    if any(len(neighbors) != 3 for neighbors in adjacency.values()):
        raise ValueError('honeycomb_hexagon requires the periodic nearest-neighbor honeycomb graph (three neighbors per site)')
    if any(sites[source]['basis_index'] == sites[target]['basis_index']
           for source, neighbors in adjacency.items() for target in neighbors):
        raise ValueError('honeycomb_hexagon bonds must join opposite honeycomb sublattices')

    def is_plaquette(ring):
        members = set(ring)
        if any(len(adjacency[site_id] & members) != 2 for site_id in ring):
            return False
        winding = [0, 0]
        for source, target in zip(ring, ring[1:] + ring[:1]):
            for axis, length in enumerate(lengths):
                delta = (sites[target]['cell_index'][axis] - sites[source]['cell_index'][axis]) % length
                if delta > length / 2:
                    delta -= length
                # Builder nearest-neighbor honeycomb edges span at most one cell.
                if abs(delta) > 1:
                    return False
                winding[axis] += delta
        return winding == [0, 0]

    rings = set()

    def walk(path):
        if len(path) == 6:
            if path[0] in adjacency[path[-1]] and path[1] < path[-1] and is_plaquette(path):
                rings.add(tuple(path))
            return
        for neighbor in sorted(adjacency[path[-1]]):
            if neighbor > path[0] and neighbor not in path:
                walk(path + [neighbor])

    for site_id in sorted(sites):
        walk([site_id])
    candidates = sorted(rings)
    positions = {site_id: index for index, site_id in enumerate(sorted(sites))}
    masks = [sum(1 << positions[site_id] for site_id in ring) for ring in candidates]
    by_site = [[index for index, ring in enumerate(candidates) if site_id in ring] for site_id in sorted(sites)]
    # A regular honeycomb torus has three elementary faces incident on each site.
    if any(len(choices) != 3 for choices in by_site):
        raise ValueError('honeycomb_hexagon could not identify three elementary, non-winding hexagons at each site')
    full_mask = (1 << len(sites)) - 1
    pending = [(0, [])]
    visited = set()
    while pending:
        covered, chosen = pending.pop()
        if covered == full_mask:
            return [
                {'fragment_id': f'hexagon-{position + 1}', 'label': f'Hexagon {position + 1}',
                 'site_ids': list(candidates[index]), 'orbital_indices': [],
                 'metadata': {'selection': 'honeycomb_hexagon', 'ring_size': 6}}
                for position, index in enumerate(sorted(chosen))
            ]
        if covered in visited:
            continue
        visited.add(covered)
        if len(visited) > 10000:
            raise ValueError('honeycomb_hexagon partition search exceeded its limit; use a regular 3x3 or 6x6 honeycomb')
        choices = min(
            ([index for index in by_site[position] if not masks[index] & covered]
             for position in range(len(sites)) if not covered & (1 << position)),
            key=len,
        )
        pending.extend((covered | masks[index], chosen + [index]) for index in reversed(choices))
    raise ValueError(
        'This honeycomb cannot be tiled by disjoint elementary six-site rings; '
        'use compatible periodic dimensions such as 3x3, 3x6, or 6x6 primitive cells.'
    )
