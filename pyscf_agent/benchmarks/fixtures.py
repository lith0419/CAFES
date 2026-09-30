from __future__ import annotations

from typing import Any, Dict, Optional


def block2_unavailable_outcome() -> Optional[Dict[str, Any]]:
    from ..providers.block2 import block2_availability

    availability = block2_availability()
    if availability.get('available'):
        return None
    return {
        'metrics': {'provider': availability},
        'checks': [],
        'skip_reason': availability.get('reason') or 'block2 is unavailable',
    }


def hubbard_ring_spec(site_count: int, u_value: float = 4.0) -> Dict[str, Any]:
    count = int(site_count)
    return {
        'schema': 'pyscf-agent.model-hamiltonian.v1',
        'model': 'hubbard',
        'dimension': 1,
        'preset': 'ring',
        'boundary': 'periodic',
        'energy_unit': 'a.u.',
        'nelec': [count // 2, count // 2],
        'sites': [
            {'id': index, 'x': index, 'y': 0, 'epsilon': 0, 'U': u_value}
            for index in range(count)
        ],
        'bonds': [
            {
                'id': index,
                'source': index,
                'target': (index + 1) % count,
                't': -1,
                'V': 0,
                'effective_t': -1,
            }
            for index in range(count)
        ],
    }
