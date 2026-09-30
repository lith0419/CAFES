"""Resolve explicit DMET density donors from saved Study results."""

from pyscf_agent.providers.libdmet.density_restart import (
    normalize_density_source,
    STATE_KIND,
)


def density_source(case):
    solver = case.request.get('solver') or {}
    return (
        (solver.get('options') or {}).get('reference_density_source')
        if isinstance(solver, dict)
        else None
    )


def resolve_dmet_density(case, study_state):
    """Mutate an executor-owned case copy, never the authoritative Study plan."""
    raw = density_source(case)
    if raw is None:
        return None
    solver = case.request['solver']
    if solver.get('name') != 'dmet':
        return 'reference_density_source requires a DMET case.'
    try:
        source = normalize_density_source(raw)
    except ValueError as exc:
        return str(exc)
    if source.get('path'):
        return None
    source_id = source['source_case_id']
    if source_id == case.case_id:
        return 'DMET neighbor warm start requires a different source_case_id.'
    checkpoint = (study_state.get('cases') or {}).get(source_id) or {}
    report = checkpoint.get('task_report') or {}
    if source.get('source_run_id'):
        from .dmet_branches import candidate_records
        report = next((candidate['task_report'] for candidate in candidate_records(checkpoint)
                       if candidate['task_report']['run_id'] == source['source_run_id']), {})
    results = report.get('structured_results') or {}
    if (
        report.get('execution_status') != 'succeeded'
        or results.get('converged') is not True
    ):
        return (
            'DMET density continuation requires a succeeded, converged source case: '
            + source_id
        )
    run_id = report.get('run_id')
    if not run_id or (
        source.get('source_run_id') and source['source_run_id'] != run_id
    ):
        return 'DMET density source Run ID is unavailable or differs from the requested Run.'
    artifact = next(
        (
            item
            for item in report.get('artifacts', [])
            if isinstance(item, dict) and item.get('kind') == STATE_KIND
        ),
        None,
    )
    if not artifact or not artifact.get('path') or not artifact.get('sha256'):
        return 'Source Run has no verified final DMET mean-field state; an old initial seed cannot substitute for it.'
    resolved = {
        'source_case_id': source_id,
        'source_run_id': run_id,
        'path': artifact['path'],
        'sha256': artifact['sha256'],
    }
    try:
        resolved = normalize_density_source(resolved)
    except ValueError as exc:
        return str(exc)
    solver['options']['reference_density_source'] = resolved
    return None
