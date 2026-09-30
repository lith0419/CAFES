from __future__ import annotations

from typing import Any, Dict, List, Optional

from .constants import SUPPORTED_REQUEST_FIELDS
from .utils import (
    _append_unique,
    _normalize_string,
    _normalize_string_list,
)
from ..pyscf_i18n import join_items, normalize_locale, t


def _has_modification_intent(messages: List[Dict[str, str]], request_text: Optional[str], llm_payload: Dict[str, Any]) -> bool:
    text_parts: List[str] = []
    if request_text:
        text_parts.append(request_text)
    for message in messages:
        if message.get('role') == 'user':
            text_parts.append(message.get('content', ''))
    for key in ('request', 'request_summary'):
        value = _normalize_string(llm_payload.get(key))
        if value:
            text_parts.append(value)

    if not text_parts:
        return False

    lower_text = '\n'.join(text_parts).lower()
    suggestive_english_patterns = (
        'suggest',
        'recommend',
        'optimize',
        'improve',
        'replace',
        'generate',
    )
    if any(pattern in lower_text for pattern in suggestive_english_patterns):
        return True

    suggestive_chinese_patterns = (
        '建议',
        '推荐',
        '优化',
        '生成',
        '草案',
    )
    if any(pattern in lower_text for pattern in suggestive_chinese_patterns):
        return True

    approval_markers = ('确认后', '批准后', '同意后', '让我确认', '等我确认', 'approve', 'confirm first')
    modification_markers = ('修改', '改成', '换成', '调整', '替换', 'modify', 'change', 'adjust')
    return any(marker in lower_text for marker in approval_markers) and any(marker in lower_text for marker in modification_markers)


def _values_differ(seed_value: Any, candidate_value: Any) -> bool:
    if isinstance(seed_value, list) or isinstance(candidate_value, list):
        return _normalize_string_list(seed_value) != _normalize_string_list(candidate_value)
    return seed_value != candidate_value


def _describe_change(field: str, seed_value: Any, candidate_value: Any, locale: str) -> str:
    def _format_value(value: Any) -> str:
        if isinstance(value, list):
            normalized_list = _normalize_string_list(value)
            if normalize_locale(locale) == 'en':
                return ', '.join(normalized_list) if normalized_list else 'empty'
            return '、'.join(normalized_list) if normalized_list else '空'
        if isinstance(value, bool):
            return 'true' if value else 'false'
        if value is None or value == '':
            return 'not set' if normalize_locale(locale) == 'en' else '未设置'
        return str(value)

    labels = {
        'zh': {
            'atom': '结构',
            'basis': '基组',
            'method': '方法',
            'xc': '泛函',
            'job': '任务类型',
            'outputs': '分析输出',
            'charge': '电荷',
            'spin': '自旋',
            'symmetry': '对称性',
            'restricted': '限制性',
            'unit': '单位',
            'max_cycle': '最大循环数',
            'conv_tol': '收敛阈值',
            'verbose': '输出级别',
        },
        'en': {
            'atom': 'Structure',
            'basis': 'Basis set',
            'method': 'Method',
            'xc': 'Functional',
            'job': 'Job type',
            'outputs': 'Analysis outputs',
            'charge': 'Charge',
            'spin': 'Spin',
            'symmetry': 'Symmetry',
            'restricted': 'Restricted',
            'unit': 'Unit',
            'max_cycle': 'Max cycles',
            'conv_tol': 'Convergence tolerance',
            'verbose': 'Verbosity',
        },
    }
    label = labels.get(normalize_locale(locale), labels['en']).get(field, field)
    return '{0}: {1} -> {2}'.format(label, _format_value(seed_value), _format_value(candidate_value))


def _collect_proposed_changes(task_spec_seed: Dict[str, Any], structured_request: Dict[str, Any], locale: str) -> List[str]:
    proposed_changes: List[str] = []
    for field in SUPPORTED_REQUEST_FIELDS:
        if field in ('request', 'request_summary'):
            continue
        seed_value = task_spec_seed.get(field)
        candidate_value = structured_request.get(field)
        if candidate_value is None:
            continue
        if _values_differ(seed_value, candidate_value):
            proposed_changes.append(_describe_change(field, seed_value, candidate_value, locale))
    return proposed_changes


def _rewrite_modification_questions(
    clarification_questions: List[str],
    proposed_changes: List[str],
    *,
    modification_intent: bool,
    locale: str,
) -> List[str]:
    if not modification_intent or not proposed_changes:
        return clarification_questions

    rewritten = clarification_questions[:]
    _append_unique(
        rewritten,
        t(locale, 'prepare_modification_review', changes=join_items(locale, proposed_changes)),
    )
    return rewritten
