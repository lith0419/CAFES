"""Read observed sweep metrics; do not infer scientific convergence from a log."""

from __future__ import annotations

import math
import re


_NUMBER = r'[-+]?\d+(?:\.\d*)?(?:[eEdD][-+]?\d+)?'
_HEADER = re.compile(
    r'Sweep\s*=\s*(\d+)\s*\|.*?Bond dimension\s*=\s*(\d+)'
    r'\s*\|\s*Noise\s*=\s*(' + _NUMBER + r')'
    r'\s*\|\s*Dav threshold\s*=\s*(' + _NUMBER + r')'
)
_RESULT = re.compile(
    r'Time elapsed\s*=\s*' + _NUMBER + r'\s*\|\s*E\s*=\s*(' + _NUMBER + r')'
    r'\s*\|\s*DE\s*=\s*(' + _NUMBER + r')\s*\|\s*DW\s*=\s*(' + _NUMBER + r')'
)


def parse_sweep_progress(text):
    """Only attach a completed-sweep result to its own preceding header.

    A new sweep zero starts another solve/stage. A bounded tail can begin
    halfway through a sweep, whose result must not acquire an invented index.
    """
    current = completed = None
    for line in text.splitlines():
        header = _HEADER.search(line)
        if header:
            index, bond, noise, threshold = header.groups()
            if int(index) == 0:
                completed = None
            current = {'sweep_index': int(index), 'bond_dimension': int(bond),
                       'noise': float(noise.replace('D', 'e').replace('d', 'e')),
                       'davidson_threshold': float(threshold.replace('D', 'e').replace('d', 'e'))}
        result = _RESULT.search(line)
        if result and current is not None:
            values = [float(v.replace('D', 'e').replace('d', 'e')) for v in result.groups()]
            if all(math.isfinite(v) for v in values):
                completed = {**current, 'energy': values[0], 'energy_change': abs(values[1]),
                             'discarded_weight': values[2]}
    if current is None:
        return None
    return {'provider': 'block2', 'observed_sweep': current, 'last_completed_sweep': completed,
            'note': 'Last logged solver sweep, not a final task energy or convergence assessment. Sweep indices start at zero and reset between solves/stages.'}
