"""Reviewable DMET recovery proposals derived from the failing solver stage."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def dmet_scf_recovery(
    options: Mapping[str, Any], *, exception_type: str, traceback_text: str,
) -> Optional[Dict[str, Any]]:
    """Recognize impurity DIIS failures, including reports saved before this feature.

    A LinAlgError alone is insufficient: lattice diagonalization, density fitting
    and the correlated solver require different recovery actions.
    """
    frames = str(traceback_text).replace('\\', '/')
    if exception_type != 'LinAlgError' or not all(marker in frames for marker in (
        'libdmet/solver/scf.py', 'pyscf/lib/diis.py', 'in extrapolate',
    )):
        return None
    from .dmet import normalize_dmet_options

    configuration = normalize_dmet_options(options)
    impurity_solver = configuration['impurity_solver']
    solver_file = {'fci': 'fci.py', 'ccsd': 'cc.py'}.get(impurity_solver)
    if not solver_file or 'libdmet/solver/' + solver_file not in frames:
        return None
    if not configuration['impurity_scf_diis']:
        return None
    return {
        'schema': 'pyscf-agent.dmet-recovery-recommendation.v1',
        'status': 'review_required',
        'failure_class': 'impurity_scf_diis_failure',
        'automatic_retry_safe': False,
        'recommended_action': 'retry_dmet_without_impurity_diis',
        'summary': (
            'The preliminary impurity SCF failed during DIIS extrapolation. '
            'Retry this case with solver.options.impurity_scf_diis=false '
            '(previously true). This avoids the failing extrapolation; the model, '
            'impurity solver and convergence tolerances stay unchanged. '
            'SCF may take more iterations, and the new DMET result must still pass '
            'its convergence checks.'
        ),
        'solver_options_patch': {'impurity_scf_diis': False},
    }


def dmet_task_recovery(
    task_report: Mapping[str, Any], request: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if task_report.get('execution_status') != 'failed':
        return None
    task = request if request is not None else task_report.get('task_spec') or {}
    solver = task.get('solver')
    name = solver.get('name') if isinstance(solver, Mapping) else solver
    if str(name or '').strip().lower() != 'dmet':
        return None
    options = (solver.get('options') or {}) if isinstance(solver, Mapping) else {}
    for error in reversed(task_report.get('errors') or []):
        if not isinstance(error, Mapping) or error.get('stage') != 'execution':
            continue
        details = error.get('details') or {}
        return dmet_scf_recovery(
            options, exception_type=error.get('exception_type') or '',
            traceback_text=details.get('traceback') or '',
        )
    return None
