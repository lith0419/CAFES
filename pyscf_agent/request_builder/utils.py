from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional

from .constants import SUPPORTED_REQUEST_FIELDS
from ..pyscf_i18n import normalize_locale


def _normalize_string(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_string_list(value: Any) -> List[str]:
    if not isinstance(value, (list, tuple)):
        return []
    normalized: List[str] = []
    for item in value:
        normalized_item = _normalize_string(item)
        if normalized_item and normalized_item not in normalized:
            normalized.append(normalized_item)
    return normalized


def _normalize_message(item: Any) -> Optional[Dict[str, str]]:
    if not isinstance(item, dict):
        return None
    role = _normalize_string(item.get('role'))
    content = _normalize_string(item.get('content'))
    if role not in ('system', 'user', 'assistant') or not content:
        return None
    return {
        'role': role,
        'content': content,
    }


def _normalize_messages(messages: Any) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in messages or []:
        message = _normalize_message(item)
        if message is not None:
            normalized.append(message)
    return normalized


def _seed_from_task_spec(task_spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(task_spec, dict):
        return {}
    seed: Dict[str, Any] = {}
    for key in SUPPORTED_REQUEST_FIELDS:
        value = task_spec.get(key)
        if key == 'outputs':
            outputs = _normalize_string_list(value)
            if outputs:
                seed[key] = outputs
            continue
        normalized_value = _normalize_string(value) if isinstance(value, str) else value
        if normalized_value is not None and normalized_value != '':
            seed[key] = normalized_value
    return seed


def _compact_request_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in SUPPORTED_REQUEST_FIELDS:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, list) and not value:
            continue
        compact[key] = copy.deepcopy(value)
    return compact


def _json_request_text(payload: Dict[str, Any]) -> str:
    return json.dumps(_compact_request_payload(payload), ensure_ascii=False)


def _append_unique(items: List[str], value: Optional[str]) -> None:
    normalized = _normalize_string(value)
    if normalized and normalized not in items:
        items.append(normalized)


def _localize_question(question: str, locale: str = 'en') -> str:
    normalized = _normalize_string(question)
    if not normalized:
        return ''

    lower_question = normalized.lower()
    normalized_locale = normalize_locale(locale)
    localized_patterns = {
        'zh': (
            ('please provide atomic coordinates', '请提供分子结构或原子坐标。'),
            ('please provide the atomic coordinates', '请提供分子结构或原子坐标。'),
            ('specify unit for coordinates', '请确认坐标单位；默认使用 Angstrom。'),
            ('charge?', '请确认电荷；默认使用 0。'),
            ('spin multiplicity?', '请确认自旋；默认使用 0。'),
            ('default 0 for closed-shell', '请确认自旋；默认使用 0。'),
            ('use symmetry?', '请确认是否使用对称性；默认关闭。'),
            ('should symmetry be used?', '请确认是否使用对称性；默认关闭。'),
            ('restricted/unrestricted?', '请确认是否采用限制性计算；默认按 restricted 处理。'),
            ('max cycles?', '请确认最大迭代步数；默认使用 50。'),
            ('convergence tolerance?', '请确认收敛阈值；默认使用 1e-8。'),
            ('what convergence tolerance should be used?', '请确认收敛阈值；默认使用 1e-8。'),
            ('verbose level?', '请确认输出级别；默认使用 4。'),
            ('please specify the basis set', '请指定基组，例如 sto-3g、6-31g 或 cc-pvdz。'),
            ('please provide the basis set', '请指定基组，例如 sto-3g、6-31g 或 cc-pvdz。'),
        ),
        'en': (
            ('请提供分子结构', 'Please provide the molecular structure or atomic coordinates.'),
            ('请提供结构', 'Please provide the molecular structure or atomic coordinates.'),
            ('请给出结构', 'Please provide the molecular structure or atomic coordinates.'),
            ('请提供坐标', 'Please provide the molecular structure or atomic coordinates.'),
            ('请给出坐标', 'Please provide the molecular structure or atomic coordinates.'),
            ('请提供原子坐标', 'Please provide the molecular structure or atomic coordinates.'),
            ('请确认坐标单位', 'Please confirm the coordinate unit; Angstrom is the default.'),
            ('请确认电荷', 'Please confirm the charge; 0 is the default.'),
            ('请确认自旋', 'Please confirm the spin; 0 is the default.'),
            ('请确认是否使用对称性', 'Please confirm whether symmetry should be used; it is off by default.'),
            ('请确认是否采用限制性计算', 'Please confirm whether to use a restricted treatment; restricted is the default.'),
            ('请确认最大迭代步数', 'Please confirm the maximum number of iterations; 50 is the default.'),
            ('请确认收敛阈值', 'Please confirm the convergence tolerance; 1e-8 is the default.'),
            ('请确认输出级别', 'Please confirm the verbosity level; 4 is the default.'),
            ('请指定基组', 'Please specify a basis set, for example sto-3g, 6-31g, or cc-pvdz.'),
        ),
    }.get(normalized_locale, ())
    for pattern, replacement in localized_patterns:
        if pattern in lower_question:
            return replacement
    return normalized


def _normalize_questions(questions: Any, locale: str = 'en') -> List[str]:
    normalized_questions: List[str] = []
    for question in _normalize_string_list(questions):
        _append_unique(normalized_questions, _localize_question(question, locale=locale))
    return normalized_questions


def _normalize_missing_fields(fields: Any) -> List[str]:
    normalized_fields: List[str] = []
    for field in _normalize_string_list(fields):
        _append_unique(normalized_fields, field)
    return normalized_fields


def _request_context_text(
    messages: List[Dict[str, str]],
    request_text: Optional[str],
    llm_payload: Dict[str, Any],
    task_spec_seed: Optional[Dict[str, Any]] = None,
) -> str:
    text_parts: List[str] = []
    if request_text:
        text_parts.append(request_text)
    for message in messages:
        if message.get('role') == 'user':
            text_parts.append(message.get('content', ''))
    if isinstance(task_spec_seed, dict):
        for key in ('request', 'request_summary'):
            value = _normalize_string(task_spec_seed.get(key))
            if value:
                text_parts.append(value)
    for key in ('request', 'request_summary'):
        value = _normalize_string(llm_payload.get(key))
        if value:
            text_parts.append(value)
    return '\n'.join(part for part in text_parts if part)


def _record_default(applied_defaults: List[Dict[str, Any]], field: str, value: Any, reason: str) -> None:
    applied_defaults.append({
        'field': field,
        'value': copy.deepcopy(value),
        'reason': reason,
    })


def _filter_questions_for_auto_fields(
    clarification_questions: List[str],
    missing_fields: List[str],
    auto_fields: List[str],
) -> Dict[str, List[str]]:
    remaining = [field for field in missing_fields if field not in auto_fields]
    return {
        'missing_fields': remaining,
        'clarification_questions': (
            [] if missing_fields and not remaining else clarification_questions
        ),
    }


def _precision_expectation(structured_request: Dict[str, Any], applied_defaults: List[Dict[str, Any]], generated_fields: List[str], locale: str = 'en') -> str:
    normalized_locale = normalize_locale(locale)
    if 'atom' in generated_fields:
        return '结构为自动生成草案，只适合初步探索和提示词迭代，不适合直接作为高精度定量结论依据。' if normalized_locale == 'zh' else 'The structure is an auto-generated draft. It is suitable for early exploration and prompt iteration, but not for high-precision quantitative conclusions.'
    if any(item.get('field') == 'basis' and item.get('value') == 'sto-3g' for item in applied_defaults):
        return '当前结果基于默认的 sto-3g 基组，更适合快速定性筛查；如果需要更可靠的定量结果，建议改用更大的基组并复算。' if normalized_locale == 'zh' else 'The current result uses the default sto-3g basis set, which is better suited to fast qualitative screening. For more reliable quantitative results, use a larger basis set and rerun the calculation.'
    if structured_request.get('method') == 'hf':
        return '当前结果更适合快速基线判断；若需要更高精度，建议结合更高质量基组或 DFT 泛函进一步验证。' if normalized_locale == 'zh' else 'The current result is better suited to a fast baseline check. If you need higher accuracy, validate it with a higher-quality basis set or a DFT functional.'
    return '当前结果适合常规探索性计算；精度仍受结构、方法和基组选择影响。' if normalized_locale == 'zh' else 'The current result is suitable for routine exploratory calculations, but accuracy still depends on the structure, method, and basis set.'


def _assistant_message(content: str) -> Dict[str, str]:
    return {
        'role': 'assistant',
        'content': content,
    }
