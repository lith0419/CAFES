from __future__ import annotations

from typing import Any, Dict, List


def case_ids_from_plan(plan: Dict[str, Any]) -> List[str]:
    cases = plan.get('cases') if isinstance(plan, dict) else []
    if not isinstance(cases, list):
        return []
    return [
        str(item.get('case_id'))
        for item in cases
        if isinstance(item, dict) and item.get('case_id') is not None
    ]
