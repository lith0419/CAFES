"""Explicit, saved-Study retry contracts and deterministic scope compilation."""

from __future__ import annotations

from pyscf_agent.serialization import json_fingerprint

import copy
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from .schema import StudyPlan
from .study_state import read_json, load_checkpoint

RETRY_ACTION_SCHEMA = 'pyscf-agent.retry-action.v1'
# A retry can change the numerical method/settings, never the Hamiltonian,
# geometry, scan design, identity, or shared resource policy.
REQUEST_FIELDS = {'runtime', 'solver', 'method', 'xc', 'active_space', 'post_cas'}


def saved_revision(directory):
    return json_fingerprint(
        {
            'plan': read_json(directory / 'study-plan.json'),
            'state': load_checkpoint(directory / 'study-state.json', directory.name)
            if (directory / 'study-state.json').exists()
            else None,
        }
    )


def case_field_fingerprints(plan):
    def fields(value, prefix):
        if isinstance(value, dict) and value:
            return {
                path: digest
                for key, item in value.items()
                for path, digest in fields(item, prefix + '.' + key).items()
            }
        return {prefix: json_fingerprint(value)}

    return {case.case_id: fields(case.to_dict(), 'case') for case in plan.cases}


@dataclass(frozen=True)
class RetryAction:
    study_id: str
    case_ids: List[str]
    case_overrides: Dict[str, Dict[str, Any]]
    base_study_fingerprint: str
    reason: str = ''
    include_dependents: bool = False
    schema: str = RETRY_ACTION_SCHEMA

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict):
            raise ValueError('RetryAction must be an object')
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                'RetryAction does not accept fields: ' + ', '.join(sorted(unknown))
            )
        try:
            action = cls(**copy.deepcopy(payload))
            json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError('Invalid or incomplete RetryAction: ' + str(exc)) from exc
        if action.schema != RETRY_ACTION_SCHEMA:
            raise ValueError('Unsupported RetryAction schema')
        if not isinstance(action.study_id, str) or not action.study_id:
            raise ValueError('RetryAction requires study_id')
        ids = action.case_ids
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(i, str) or not i.strip() for i in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError(
                'RetryAction requires nonempty, unique case_ids; an empty retry never means run all'
            )
        if (
            not isinstance(action.base_study_fingerprint, str)
            or not action.base_study_fingerprint
        ):
            raise ValueError(
                'RetryAction requires base_study_fingerprint from the saved Study'
            )
        if not isinstance(action.reason, str) or not isinstance(
            action.include_dependents, bool
        ):
            raise ValueError('Invalid retry reason or include_dependents')
        if not isinstance(action.case_overrides, dict) or set(
            action.case_overrides
        ) - set(ids):
            raise ValueError('case_overrides may only modify declared case_ids')
        for patches in action.case_overrides.values():
            if not isinstance(patches, dict):
                raise ValueError(
                    'Each case override must be a mapping of request paths to values'
                )
            for path in patches:
                if (
                    not isinstance(path, str)
                    or path.split('.')[0] not in REQUEST_FIELDS
                    or any(not part or part.startswith('_') for part in path.split('.'))
                ):
                    raise ValueError('Retry cannot change field: ' + str(path))
                if any(
                    other != path and other.startswith(path + '.') for other in patches
                ):
                    raise ValueError('Overlapping retry override paths: ' + path)
        return action

    def to_dict(self):
        return asdict(self)


def live_dependencies(case):
    """Only deferred references follow a new parent Run; fixed artifact paths do not."""
    request = case.request
    initial = request.get('initial_state') or {}
    artifact = initial.get('source_artifact') or {}
    sources = set()
    if initial.get('mode') == 'projected_1rdm' and not artifact.get('path'):
        source = initial.get('source_case_id') or artifact.get('deferred_case_id')
        if source:
            sources.add(source)
    solver = request.get('solver')
    options = solver.get('options') or {} if isinstance(solver, dict) else {}
    density_source = options.get('reference_density_source')
    if isinstance(density_source, dict) and not density_source.get('path') and density_source.get('source_case_id'):
        sources.add(density_source['source_case_id'])
    for key in ('restart_manifest', 'orbital_restart_manifest'):
        reference = options.get(key)
        if isinstance(reference, dict) and reference.get('source_case_id'):
            sources.add(reference['source_case_id'])
    return sources


def compile_retry(plan, action):
    """Derive a subset from the authoritative plan, never from a caller's plan."""
    selected = set(action.case_ids)
    known = {case.case_id for case in plan.cases}
    if selected - known:
        raise ValueError(
            'Unknown retry case ids: ' + ', '.join(sorted(selected - known))
        )
    descendants = set(selected)
    while True:
        expanded = descendants | {
            c.case_id for c in plan.cases if live_dependencies(c) & descendants
        }
        if expanded == descendants:
            break
        descendants = expanded
    scope = descendants if action.include_dependents else selected
    subset = copy.deepcopy(plan)
    subset.cases = [c for c in subset.cases if c.case_id in scope]
    changes = []
    for case in subset.cases:
        for path, value in action.case_overrides.get(case.case_id, {}).items():
            node = case.request
            parts = path.split('.')
            for part in parts[:-1]:
                if part == 'solver' and isinstance(node.get(part), str):
                    node[part] = {'name': node[part], 'options': {}}
                if node.get(part) is None:
                    node[part] = {}
                if not isinstance(node[part], dict):
                    raise ValueError('Cannot apply retry override: ' + path)
                node = node[part]
            changes.append(
                {
                    'case_id': case.case_id,
                    'path': path,
                    'before': copy.deepcopy(node.get(parts[-1])),
                    'after': copy.deepcopy(value),
                }
            )
            node[parts[-1]] = copy.deepcopy(value)
        solver = case.request.get('solver')
        name = solver.get('name') if isinstance(solver, dict) else solver
        options = (solver.get('options') or {}) if isinstance(solver, dict) else {}
        if options.get('reference_density_source') is not None:
            from pyscf_agent.providers.libdmet.density_restart import normalize_density_source
            if name != 'dmet':
                raise ValueError('reference_density_source requires a DMET case')
            source = normalize_density_source(options['reference_density_source'])
            if not source.get('path') and (source['source_case_id'] not in known or source['source_case_id'] == case.case_id):
                raise ValueError('DMET density source must name a different case in this Study')
        for key in (
            'max_iterations',
            'energy_tolerance',
            'density_tolerance',
            'impurity_scf_diis',
        ):
            if key not in options:
                continue
            value = options[key]
            if name != 'dmet':
                raise ValueError('solver.options.' + key + ' requires a DMET case')
            if key == 'impurity_scf_diis':
                valid = isinstance(value, bool)
            elif key == 'max_iterations':
                valid = (
                    isinstance(value, int) and not isinstance(value, bool) and value > 0
                )
            else:
                valid = (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and value > 0
                )
            if not valid:
                raise ValueError('Invalid retry parameter solver.options.' + key)
    # Retrying a point must never resume grid expansion.
    subset.grid_refinement = {}
    subset.cost_estimate = {}
    subset.resource_policy = dict(subset.resource_policy, approved=False)
    return (
        subset,
        changes,
        [c.case_id for c in plan.cases if c.case_id in descendants - selected],
    )


def preview_contexts(plan, state):
    from .executor import _case_execution_context

    ids = {c.case_id for c in plan.cases}
    return [
        {
            'case_id': c.case_id,
            'dependency_error': context['dependency_error'],
            'will_submit': context['should_execute']
            and not context['dependency_error'],
        }
        for c in plan.cases
        for context in [
            _case_execution_context(
                c,
                state,
                resume=True,
                selected_case_ids=ids,
                selected_statuses=set(),
                max_case_attempts=2,
            )
        ]
    ]


def validate_execution_guard(directory, plan, guard, case_ids):
    """Called under the executor lock before any plan/checkpoint mutation."""
    action = RetryAction.from_dict(guard['action'])
    if saved_revision(directory) != action.base_study_fingerprint:
        current = case_field_fingerprints(
            StudyPlan.from_dict(read_json(directory / 'study-plan.json'))
        )
        before = guard.get('base_case_fields') or {}
        changed = (
            [
                case_id + ':' + path
                for case_id in sorted(set(before) | set(current))
                for path in sorted(
                    set(before.get(case_id, {})) | set(current.get(case_id, {}))
                )
                if before.get(case_id, {}).get(path)
                != current.get(case_id, {}).get(path)
            ]
            if before
            else []
        )
        detail = (
            '; changed fields: ' + ', '.join(changed[:12])
            if changed
            else '; saved results or Study metadata changed'
        )
        raise ValueError(
            'Study changed since retry preview' + detail + '; refresh and preview again'
        )
    expected, _, _ = compile_retry(
        StudyPlan.from_dict(read_json(directory / 'study-plan.json')), action
    )
    actual = [c.case_id for c in plan.cases]
    if (
        actual != guard['execution_case_ids']
        or actual != [c.case_id for c in expected.cases]
        or set(case_ids or []) != set(actual)
    ):
        raise ValueError(
            'Retry execution exceeds or differs from the previewed case scope'
        )
    for left, right in zip(plan.cases, expected.cases):
        if left.to_dict() != right.to_dict():
            raise ValueError('Retry request differs from preview for ' + left.case_id)
