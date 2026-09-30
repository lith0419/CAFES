from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from ..registry.platform import normalize_molecular_method
from ..contracts import (
    BASIS_SET_PATTERN,
    COMMON_XC_FUNCTIONALS,
    REQUEST_HINT_KEYS,
    SUPPORTED_MODEL_SOLVERS,
)


def _extract_first_match(pattern: re.Pattern[str], text: str) -> Optional[str]:
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip().lower()


def _extract_xc_from_text(text: str) -> Optional[str]:
    explicit_xc = _extract_first_match(
        re.compile(r'\bxc\s*(?:=|:)?\s*([a-z0-9+*(),/_-]+)\b'),
        text,
    )
    if explicit_xc:
        return explicit_xc
    for xc_name in COMMON_XC_FUNCTIONALS:
        if re.search(r'(?<![a-z0-9]){0}(?![a-z0-9])'.format(re.escape(xc_name)), text):
            return xc_name
    return None


def _normalize_method_name(value: Any) -> str:
    return str(normalize_molecular_method(value, default='hf'))


def _detect_post_hf_method(text: str) -> Optional[str]:
    if re.search(r'\bcasscf\b|\bcomplete[- ]active[- ]space[- ]scf\b', text):
        return 'casscf'
    if re.search(r'\bcasci\b|\bcomplete[- ]active[- ]space[- ]ci\b', text):
        return 'casci'
    if re.search(r'\b(?:full[- ]?ci|fullci|fci)\b', text):
        return 'fci'
    if re.search(r'\bccsd\s*\(\s*t\s*\)(?![a-z0-9])|\bccsd[- ]?t\b|\bccsdt\b|\bccsd triples\b', text):
        return 'ccsd_t'
    if re.search(r'\bccsd\b', text):
        return 'ccsd'
    if re.search(r'\bmp2\b', text):
        return 'mp2'
    return None


def _extract_basis_from_text(text: str) -> Optional[str]:
    explicit_basis = _extract_first_match(
        re.compile(r'\bbasis\s*(?:=|:)?\s*([a-z0-9+*(),_-]+)(?![a-z0-9])'),
        text,
    )
    if explicit_basis:
        return explicit_basis
    match = BASIS_SET_PATTERN.search(text)
    if not match:
        return None
    return match.group(1).lower()


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    start = text.find('{')
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:idx + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _extract_key_value_block(text: str) -> Dict[str, Any]:
    parsed = {}
    for line in text.splitlines():
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        key = key.strip().lower()
        value = value.strip()
        value = value.split('#', 1)[0].strip()
        if not value:
            continue
        if key == 'periodic_band_path_special_points':
            try:
                decoded_points = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded_points, dict):
                parsed[key] = decoded_points
        elif key in (
            'atom',
            'basis',
            'method',
            'job',
            'xc',
            'unit',
            'task_type',
            'solver',
            'model_hamiltonian_input_file',
            'periodic_structure_format',
            'periodic_structure_text',
            'periodic_basis',
            'periodic_pseudo',
            'kmesh',
            'kpoint_scheme',
            'kpoint_shift',
            'periodic_precision',
            'periodic_ke_cutoff',
            'periodic_fft_mesh',
            'periodic_density_fitting_method',
            'periodic_density_fitting_auxbasis',
            'periodic_exxdiv',
            'periodic_smearing_method',
            'periodic_smearing_sigma',
            'periodic_smearing_fix_spin',
            'periodic_band_path_mode',
            'periodic_band_path',
            'localization_method',
            'auxbasis',
            'df_auxbasis',
            'density_fitting_auxbasis',
            'active_space_method',
            'selection_method',
            'nelecas',
            'active_orbitals',
            'orbital_indices',
            'occupation_window',
            'avas_targets',
            'avas_minimal_basis',
            'molecular_dynamics',
        ):
            parsed[key] = value
        elif key in (
            'charge', 'spin', 'max_cycle', 'grid_level', 'diis_space',
            'verbose', 'ncas', 'periodic_band_path_npoints',
        ):
            try:
                parsed[key] = int(value)
            except ValueError:
                continue
        elif key in ('symmetry', 'orbital_processing', 'use_natural_orbitals', 'density_fitting', 'density_fit', 'df', 'active_space', 'active_space_approved'):
            parsed[key] = value.lower() in ('1', 'true', 'yes', 'on')
        elif key in (
            'conv_tol',
            'conv_tol_grad',
            'energy_window',
            'avas_threshold',
            'periodic_band_path_reference_distance',
            'periodic_band_path_symprec',
        ):
            try:
                parsed[key] = float(value)
            except ValueError:
                continue
        elif key == 'outputs':
            parsed[key] = [item.strip() for item in value.split(',') if item.strip()]
    return parsed


def parse_user_request(user_request: str) -> Dict[str, Any]:
    parsed = _extract_json_object(user_request)
    request_hints = []
    parsed_from_json = parsed is not None

    if parsed is None:
        parsed = _extract_key_value_block(user_request)

    if parsed_from_json:
        for key in REQUEST_HINT_KEYS:
            value = parsed.get(key)
            if isinstance(value, str):
                normalized_value = value.strip()
                if normalized_value and normalized_value not in request_hints:
                    request_hints.append(normalized_value)
        lower_source = request_hints
    else:
        lower_source = [user_request]
    lower = '\n'.join(lower_source).lower()

    if 'task_type' not in parsed:
        if 'model_hamiltonian' in parsed or 'model_hamiltonian_input_file' in parsed:
            parsed['task_type'] = 'model_hamiltonian'
        elif 'periodic' in parsed or 'periodic_structure_text' in parsed:
            parsed['task_type'] = 'periodic'
        elif re.search(r'\b(?:model hamiltonian|hubbard model|hubbard-model)\b', lower):
            parsed['task_type'] = 'model_hamiltonian'
        elif re.search(r'\b(?:periodic|crystal|solid-state|solid state|poscar|cif|k-point|k point)\b', lower):
            parsed['task_type'] = 'periodic'

    if 'xc' not in parsed:
        detected_xc = _extract_xc_from_text(lower)
        if detected_xc:
            parsed['xc'] = detected_xc

    if 'basis' not in parsed:
        detected_basis = _extract_basis_from_text(lower)
        if detected_basis:
            parsed['basis'] = detected_basis

    detected_post_hf = _detect_post_hf_method(lower)
    if 'method' not in parsed:
        if detected_post_hf:
            parsed['method'] = detected_post_hf
        elif re.search(r'\b(?:hf|rhf|uhf|hartree-fock|hartree fock)\b', lower):
            parsed['method'] = 'hf'
        elif parsed.get('xc') or re.search(r'\bdft\b', lower):
            parsed['method'] = 'dft'
    elif isinstance(parsed.get('method'), str):
        explicit_method = parsed['method'].strip().lower()
        if explicit_method in ('rhf', 'uhf'):
            parsed['method'] = 'hf'
            parsed['restricted'] = explicit_method == 'rhf'
        else:
            parsed['method'] = _normalize_method_name(parsed['method'])

    if parsed.get('method') == 'hf':
        if re.search(r'\buhf\b', lower):
            parsed['restricted'] = False
        elif re.search(r'\brhf\b', lower):
            parsed['restricted'] = True

    if parsed.get('task_type') == 'model_hamiltonian' and 'solver' not in parsed:
        method_as_solver = _normalize_method_name(parsed.get('method'))
        if method_as_solver in SUPPORTED_MODEL_SOLVERS:
            parsed['solver'] = method_as_solver

    if 'localization_method' not in parsed:
        if re.search(r'\bboys\b', lower):
            parsed['localization_method'] = 'boys'
            parsed['orbital_processing'] = True
        elif re.search(r'\b(?:pipek[- ]?mezey|pm)\b', lower):
            parsed['localization_method'] = 'pipek_mezey'
            parsed['orbital_processing'] = True

    if not any(key in parsed for key in ('density_fitting', 'density_fit', 'df')):
        if re.search(r'\bdensity[- ]?fitting\b|\bdf\b', lower):
            parsed['density_fitting'] = True

    if 'active_space' not in parsed and re.search(r'\b(?:active space|active-space|casci|casscf)\b', lower):
        parsed['active_space'] = True

    if 'job' not in parsed:
        if re.search(
            r'\b(?:molecular[- ]?dynamics|bomd|md|nve)\b|分子动力学',
            lower,
        ):
            parsed['job'] = 'molecular_dynamics'
        else:
            parsed['job'] = 'single_point'

    if 'outputs' not in parsed:
        outputs = []
        if parsed.get('job') == 'molecular_dynamics':
            outputs.append('trajectory')
        if 'homo' in lower or 'lumo' in lower:
            outputs.append('homo_lumo')
        if 'dipole' in lower:
            outputs.append('dipole')
        if 'energy' in lower or 'single point' in lower or 'single-point' in lower:
            outputs.append('energy')
        if outputs:
            parsed['outputs'] = sorted(set(outputs))

    return parsed
