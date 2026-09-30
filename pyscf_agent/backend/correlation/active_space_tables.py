from __future__ import annotations

from typing import Any, Dict


def orbital_summary_table(orbital_summary: Dict[str, Any]) -> str:
    rows = orbital_summary.get('orbital_table')
    if not isinstance(rows, list):
        rows = []
    lines = ['index\tspin\tenergy\toccupation']
    for row in rows:
        lines.append('{0}\t{1}\t{2}\t{3}'.format(
            row.get('index'),
            row.get('spin') or '',
            '' if row.get('energy') is None else '{0:.12g}'.format(float(row['energy'])),
            '' if row.get('occupation') is None else '{0:.12g}'.format(float(row['occupation'])),
        ))
    return '\n'.join(lines) + '\n'


def active_space_summary_table(active_space: Dict[str, Any]) -> str:
    audit_summary = active_space.get('audit_summary') if isinstance(active_space.get('audit_summary'), dict) else {}
    lines = [
        'field\tvalue',
        'selection_method\t{0}'.format(active_space.get('selection_method')),
        'ncas\t{0}'.format(active_space.get('ncas')),
        'nelecas\t{0}'.format(active_space.get('nelecas')),
        'orbital_indices\t{0}'.format(active_space.get('orbital_indices')),
        'approved\t{0}'.format(active_space.get('approved')),
    ]
    if audit_summary:
        lines.extend([
            'audit_status\t{0}'.format(audit_summary.get('status')),
            'selected_orbital_count\t{0}'.format(audit_summary.get('selected_orbital_count')),
            'estimated_nelecas\t{0}'.format(audit_summary.get('estimated_nelecas')),
            'ncas_nelecas_consistent\t{0}'.format(audit_summary.get('ncas_nelecas_consistent')),
            'recommended_next_step\t{0}'.format(audit_summary.get('recommended_next_step')),
            'candidate_methods\t{0}'.format(','.join(str(item) for item in audit_summary.get('candidate_methods', []))),
            'selected_candidate_method\t{0}'.format(audit_summary.get('selected_candidate_method')),
            'selection_confidence\t{0}'.format(audit_summary.get('selection_confidence')),
            'recommendation_reason\t{0}'.format(audit_summary.get('recommendation_reason')),
        ])
    return '\n'.join(lines) + '\n'


def active_space_audit_table(active_space: Dict[str, Any]) -> str:
    audit = active_space.get('audit')
    if not isinstance(audit, dict):
        return ''
    lines = [
        'field\tvalue',
        'orbital_index_basis\t{0}'.format(audit.get('orbital_index_basis')),
        'selection_method\t{0}'.format(audit.get('selection_method')),
        'selected_candidate_method\t{0}'.format(audit.get('selected_candidate_method')),
        'localization_method\t{0}'.format(audit.get('localization_method')),
        'natural_orbital_status\t{0}'.format(audit.get('natural_orbital_status')),
    ]
    candidates = audit.get('candidate_active_spaces')
    if isinstance(candidates, list):
        lines.append('candidate_methods\t{0}'.format(','.join(
            str(candidate.get('method'))
            for candidate in candidates
            if isinstance(candidate, dict)
        )))
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            evaluation = candidate.get('evaluation') if isinstance(candidate.get('evaluation'), dict) else {}
            coverage = evaluation.get('evidence_coverage') if isinstance(evaluation.get('evidence_coverage'), dict) else {}
            lines.append('candidate.{0}\trole={1}; status={2}; CAS({3},{4}); coverage={5:.6g}; recommended={6}'.format(
                candidate.get('method'),
                candidate.get('role') or '',
                candidate.get('status'),
                candidate.get('estimated_nelecas'),
                candidate.get('ncas'),
                float(coverage.get('coverage') or 0.0),
                bool(candidate.get('recommended')),
            ))
    decision = audit.get('candidate_decision')
    if isinstance(decision, dict):
        lines.extend([
            'candidate_decision_confidence\t{0}'.format(decision.get('confidence')),
            'candidate_decision_reason\t{0}'.format(decision.get('reason')),
        ])
    consistency = audit.get('ncas_nelecas_consistency')
    if isinstance(consistency, dict):
        lines.extend([
            'final_ncas\t{0}'.format(consistency.get('final_ncas')),
            'final_nelecas\t{0}'.format(consistency.get('final_nelecas')),
            'estimated_nelecas\t{0}'.format(consistency.get('estimated_nelecas')),
            'consistent\t{0}'.format(consistency.get('consistent')),
            'messages\t{0}'.format('; '.join(str(item) for item in consistency.get('messages', []))),
        ])
    approval = audit.get('manual_approval')
    if isinstance(approval, dict):
        lines.extend([
            'manual_approval_status\t{0}'.format(approval.get('status')),
            'manual_approval_recorded\t{0}'.format(approval.get('recorded_in_request')),
        ])
    lines.append('')
    lines.append('index\tselected\tenergy\toccupation\tnatural_occupations\tt2_importance\tselection_reasons\ttop_atom')
    for row in audit.get('orbitals') or []:
        if not isinstance(row, dict):
            continue
        natural_occupations = ','.join(
            '{0}:{1:.8g}'.format(
                item.get('natural_orbital_index'),
                float(item.get('occupation')),
            )
            for item in row.get('natural_orbital_occupations', [])
            if isinstance(item, dict) and item.get('occupation') is not None
        )
        atom_contributions = row.get('atom_contributions') or []
        top_atom = ''
        if atom_contributions and isinstance(atom_contributions[0], dict):
            top_atom = '{0}:{1:.4g}'.format(
                atom_contributions[0].get('atom'),
                float(atom_contributions[0].get('weight', 0.0)),
            )
        lines.append('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}'.format(
            row.get('index'),
            row.get('selected'),
            '' if row.get('energy') is None else '{0:.12g}'.format(float(row['energy'])),
            '' if row.get('occupation') is None else '{0:.12g}'.format(float(row['occupation'])),
            natural_occupations,
            '' if row.get('t2_importance') is None else '{0:.8g}'.format(float(row['t2_importance'])),
            '; '.join(str(item) for item in row.get('selection_reasons', [])),
            top_atom,
        ))
    return '\n'.join(lines) + '\n'
