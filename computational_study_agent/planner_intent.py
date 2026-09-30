"""Conversation routing: models suggest settings, saved data determines scope."""

from __future__ import annotations

import json
import os
import re

from .llm_planner import (
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
    _default_http_post,
    _llm_endpoint,
    _llm_headers,
    _llm_timeout,
    _parse_llm_response,
)

INTENTS = {'new_study', 'modify_study', 'retry_cases', 'analyze'}


def classify_goal(goal, *, http_post=None):
    # This common command is fully deterministic and works without an LLM.
    match = re.fullmatch(
        r'\s*(?:retry|rerun|重试|重跑)\s+(case-[\w-]+)'
        r'(?:\s+(?:with\s+)?(max_iterations|max_cycle|impurity_scf_diis)'
        r'\s*(?:=|to|改成|设为)?\s*(\d+|true|false))?\s*[.!。]?\s*',
        goal,
        re.I,
    )
    if match:
        name, value = match.group(2), match.group(3)
        changes = {}
        if name:
            name = name.lower()
            path = (
                'runtime.max_cycle' if name == 'max_cycle' else 'solver.options.' + name
            )
            changes[path] = json.loads(value.lower())
        return {'intent': 'retry_cases', 'overrides': changes}
    base_url, model = os.getenv(LLM_BASE_URL_ENV), os.getenv(LLM_MODEL_ENV)
    if not base_url or not model:
        if re.search(r'retry|rerun|重试|重跑', goal, re.I):
            return {
                'intent': 'retry_cases',
                'needs_input': True,
                'message': 'Select a case and set its retry parameters in the retry form.',
            }
        raise ValueError('LLM is not configured for conversation intent classification')
    payload = {
        'model': model,
        'temperature': 0,
        'messages': [
            {
                'role': 'system',
                'content': (
                    'Classify the goal for an existing saved computational Study. Return JSON only with '
                    'intent (new_study, modify_study, retry_cases, analyze), overrides (flat request-path '
                    'to JSON value mapping), and needs_input (boolean). Retry/rerun of existing points '
                    'is retry_cases, never modify_study. Never output case_ids, StudySpec, base_task, '
                    'sweep, or a plan. Code determines the affected cases. For retry suggest only '
                    'explicitly requested settings, using runtime.max_cycle for molecular SCF or '
                    'solver.options.max_iterations for DMET outer iterations; use other solver.options '
                    'paths only when unambiguous. Missing parameter values require needs_input=true.'
                ),
            },
            {'role': 'user', 'content': goal},
        ],
    }
    result = _parse_llm_response(
        (http_post or _default_http_post)(
            _llm_endpoint(base_url), payload, _llm_headers(), _llm_timeout()
        )
    )
    if (
        not isinstance(result, dict)
        or result.get('intent') not in INTENTS
        or set(result) - {'intent', 'overrides', 'needs_input', 'message'}
    ):
        raise ValueError('Invalid planner intent response; no Study was changed')
    if (
        re.search(r'\bretry\b|\brerun\b|重试|重跑', goal, re.I)
        and result['intent'] != 'retry_cases'
    ):
        raise ValueError('Retry intent cannot be converted into a complete Study draft')
    return result


def resolve_retry_cases(goal, report, selected_case_ids=None):
    explicit = list(dict.fromkeys(re.findall(r'\bcase-[\w-]+\b', goal)))
    selected = selected_case_ids or []
    known = {c['case_id'] for c in report.get('cases') or []}
    if explicit:
        ids = explicit
        if selected and set(ids) != set(selected):
            raise ValueError(
                'Text and selected rows name different cases; choose one retry scope'
            )
    elif selected:
        ids = selected
    else:
        failed = bool(re.search(r'\bfailed\b|失败', goal, re.I))
        unconverged = bool(re.search(r'\bunconverged\b|未收敛|不收敛', goal, re.I))
        ids = []
        if failed or unconverged:
            for case in report.get('cases') or []:
                task = case.get('task_report') or {}
                status = task.get('execution_status')
                if (failed and status == 'failed') or (
                    unconverged
                    and (
                        status == 'unconverged'
                        or (task.get('structured_results') or {}).get('converged')
                        is False
                    )
                ):
                    ids.append(case['case_id'])
    if not ids or set(ids) - known:
        raise ValueError(
            'Select existing case IDs or an explicit failed/unconverged group; empty retry is not a full scan'
        )
    return ids


def draft_saved_action(
    service, study_id, goal, *, work_dir=None, selected_case_ids=None
):
    intent = classify_goal(goal)
    if intent['intent'] != 'retry_cases' or intent.get('needs_input'):
        return intent
    context = service.retry_context(study_id, work_dir=work_dir)
    report = service.load_report(study_id, work_dir=work_dir)
    ids = resolve_retry_cases(goal, report, selected_case_ids)
    action = dict(
        context,
        case_ids=ids,
        case_overrides={i: intent.get('overrides') or {} for i in ids},
        reason=goal,
    )
    return {
        'intent': 'retry_cases',
        'retry_action': service.prepare_retry(action, work_dir=work_dir),
    }
