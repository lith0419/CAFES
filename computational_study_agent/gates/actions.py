from __future__ import annotations

from typing import Any, Dict, List, Optional


def gate_action(
    action_id: str,
    label: str,
    *,
    secondary: bool = False,
    case_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    action: Dict[str, Any] = {
        'id': action_id,
        'label': label,
        'secondary': secondary,
    }
    if case_ids is not None:
        action['case_ids'] = list(case_ids)
    return action


__all__ = ['gate_action']
