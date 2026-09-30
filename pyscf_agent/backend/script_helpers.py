from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence, Tuple


def _compose_raw_stdout(raw_scf_output: str, analysis_text: str = '') -> str:
    sections = []
    if raw_scf_output:
        sections.append(raw_scf_output)
    if analysis_text:
        if sections and not sections[-1].endswith('\n'):
            sections[-1] += '\n'
        sections.append('[analysis_text]\n{0}'.format(analysis_text))
    return ''.join(sections)


def _extract_homo_lumo(mo_energy: Any, mo_occ: Any) -> Tuple[Optional[float], Optional[float]]:
    try:
        occ_ndim = mo_occ.ndim
    except AttributeError:
        occ_ndim = None

    if occ_ndim == 2:
        alpha_occ = mo_occ[0]
        alpha_energy = mo_energy[0]
        occupied = [float(energy) for energy, occ in zip(alpha_energy, alpha_occ) if occ > 0]
        virtual = [float(energy) for energy, occ in zip(alpha_energy, alpha_occ) if occ == 0]
    else:
        occupied = [float(energy) for energy, occ in zip(mo_energy, mo_occ) if occ > 0]
        virtual = [float(energy) for energy, occ in zip(mo_energy, mo_occ) if occ == 0]

    homo = occupied[-1] if occupied else None
    lumo = virtual[0] if virtual else None
    return homo, lumo


def _safe_to_list(value: Any) -> Any:
    if hasattr(value, 'tolist'):
        return _safe_to_list(value.tolist())
    if isinstance(value, dict):
        return {key: _safe_to_list(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_to_list(item) for item in value]
    return value


def _collect_result_payload(
    mf: Any,
    mol: Any,
    energy: float,
    outputs: Sequence[str],
    raw_scf_output: str,
    analysis_text: str = '',
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        'converged': bool(mf.converged),
        'energy': float(energy),
        'raw_scf_output': raw_scf_output,
    }

    if 'homo_lumo' in outputs:
        mo_occ = mf.mo_occ
        mo_energy = mf.mo_energy
        homo, lumo = _extract_homo_lumo(mo_energy, mo_occ)
        result['homo'] = homo
        result['lumo'] = lumo
        result['gap'] = None if homo is None or lumo is None else lumo - homo
        result['mo_energy'] = _safe_to_list(mo_energy)
        result['mo_occ'] = _safe_to_list(mo_occ)

    if 'dipole' in outputs:
        dm = mf.make_rdm1()
        result['dipole'] = _safe_to_list(mf.dip_moment(mol, dm))

    if analysis_text:
        result['analysis_text'] = analysis_text

    result['raw_stdout'] = _compose_raw_stdout(raw_scf_output, analysis_text)
    return result


def _format_script_output(result: Dict[str, Any]) -> str:
    raw_output = result.get('raw_scf_output', '') or ''
    structured_result = dict(result)
    structured_result.pop('raw_stdout', None)
    structured_result.pop('raw_scf_output', None)
    structured_result.pop('analysis_text', None)

    sections = []
    if raw_output:
        sections.append(raw_output if raw_output.endswith('\n') else raw_output + '\n')
        sections.append('\n=== Structured Result ===\n')
    sections.append(json.dumps(structured_result, indent=2))
    return ''.join(sections)


__all__ = ['_collect_result_payload', '_compose_raw_stdout', '_extract_homo_lumo', '_format_script_output', '_safe_to_list']