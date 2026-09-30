from __future__ import annotations

import ast
import copy
import math
import re
from typing import Any, Dict, List

from pyscf_agent.backend.model_hamiltonian.operations import (
    apply_model_operations as apply_model_operations,
    normalize_nelec_value as normalize_nelec_value,
)


TEMPLATE_TOKEN_RE = re.compile(r'\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<bare>[A-Za-z_][A-Za-z0-9_]*))')
TEMPLATE_EXACT_RE = re.compile(r'^\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<bare>[A-Za-z_][A-Za-z0-9_]*))$')
BRACED_CASE_EXPRESSION_RE = re.compile(r'\$\{(?P<expression>[^{}]+)\}')
CASE_VARIABLE_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
CASE_EXPRESSION_PREFIX = '$('


def validate_model_parameter_updates(case_design: Dict[str, Any]) -> None:
    """Reject invented request fields before a parameter scan can be compiled."""
    if not isinstance(case_design, dict):
        return  # Ordinary case-design validation owns malformed containers.
    containers = [('case_design.template', case_design.get('template'))]
    for field in ('cases', 'overrides'):
        entries = case_design.get(field)
        if isinstance(entries, list):
            containers.extend(('case_design.{0}[{1}]'.format(field, i), item)
                              for i, item in enumerate(entries))
    for path, item in containers:
        updates = item.get('request_updates') if isinstance(item, dict) else None
        if isinstance(updates, dict) and 'parameters' in updates:
            raise ValueError(
                path + '.request_updates.parameters is not a supported Hamiltonian update. '
                'For uniform U/V/t/epsilon scans, use sweep or operations such as '
                '{"op": "set_global_parameter", "parameter": "U", "value": "$U"}. '
                'Use site/bond operations for selected sites/bonds; keep solver settings fixed '
                'when grid refinement is enabled.'
            )


def _case_expression_end(value: str, start: int) -> int:
    """Return the closing parenthesis for a $(...) case expression."""
    depth = 0
    for index in range(start + 1, len(value)):
        character = value[index]
        if character == '(':
            depth += 1
        elif character == ')':
            depth -= 1
            if depth == 0:
                return index
    raise ValueError('Unclosed case expression: {0}'.format(value[start:]))


def _numeric_case_variable(name: str, variables: Dict[str, Any]) -> float:
    if name not in variables:
        raise ValueError('Unknown case variable in expression: {0}'.format(name))
    value = variables[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            'Case expression variable must be numeric: {0}'.format(name)
        )
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(
            'Case expression variable must be finite: {0}'.format(name)
        )
    return numeric_value


def _evaluate_case_expression(expression: str, variables: Dict[str, Any]) -> float:
    """Evaluate a deliberately small arithmetic language for case templates."""
    try:
        parsed = ast.parse(expression, mode='eval')
    except SyntaxError as exc:
        raise ValueError('Invalid case expression: {0}'.format(expression)) from exc

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(
                    'Case expression constants must be numeric: {0}'.format(expression)
                )
            numeric_value = float(node.value)
            if not math.isfinite(numeric_value):
                raise ValueError(
                    'Case expression constants must be finite: {0}'.format(expression)
                )
            return numeric_value
        if isinstance(node, ast.Name):
            return _numeric_case_variable(node.id, variables)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = evaluate(node.operand)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = evaluate(node.left)
            right = evaluate(node.right)
            try:
                if isinstance(node.op, ast.Add):
                    result = left + right
                elif isinstance(node.op, ast.Sub):
                    result = left - right
                elif isinstance(node.op, ast.Mult):
                    result = left * right
                else:
                    result = left / right
            except ZeroDivisionError as exc:
                raise ValueError(
                    'Case expression divides by zero: {0}'.format(expression)
                ) from exc
            if not math.isfinite(result):
                raise ValueError(
                    'Case expression must evaluate to a finite value: {0}'.format(expression)
                )
            return result
        raise ValueError(
            'Unsupported case expression; use numeric variables with +, -, *, /, and parentheses: {0}'.format(expression)
        )

    return evaluate(parsed.body)


def _replace_case_expressions(value: str, variables: Dict[str, Any]) -> Any:
    first_expression = value.find(CASE_EXPRESSION_PREFIX)
    if first_expression < 0:
        return value
    first_end = _case_expression_end(value, first_expression)
    if first_expression == 0 and first_end == len(value) - 1:
        return _evaluate_case_expression(value[2:first_end], variables)

    resolved_parts: List[str] = []
    cursor = 0
    while True:
        start = value.find(CASE_EXPRESSION_PREFIX, cursor)
        if start < 0:
            resolved_parts.append(value[cursor:])
            break
        resolved_parts.append(value[cursor:start])
        end = _case_expression_end(value, start)
        resolved_parts.append(str(_evaluate_case_expression(value[start + 2:end], variables)))
        cursor = end + 1
    return ''.join(resolved_parts)


def _replace_braced_case_expressions(value: str, variables: Dict[str, Any]) -> Any:
    """Accept legacy ``${expression}`` syntax while retaining ``${variable}``."""
    matches = list(BRACED_CASE_EXPRESSION_RE.finditer(value))
    expression_matches = [
        match for match in matches
        if not CASE_VARIABLE_NAME_RE.fullmatch(match.group('expression').strip())
    ]
    if not expression_matches:
        return value
    if len(expression_matches) == 1:
        match = expression_matches[0]
        if match.start() == 0 and match.end() == len(value):
            return _evaluate_case_expression(match.group('expression').strip(), variables)

    def replace_expression(match: re.Match) -> str:
        expression = match.group('expression').strip()
        if CASE_VARIABLE_NAME_RE.fullmatch(expression):
            return match.group(0)
        return str(_evaluate_case_expression(expression, variables))

    return BRACED_CASE_EXPRESSION_RE.sub(replace_expression, value)


def _resolve_template(value: Any, variables: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        # REGRESSION-GUARD(case-template-braced-expression): planner/LLM drafts
        # sometimes emit ${bond_factor * scale}; normalize it to the same safe
        # arithmetic path used by the documented $(...) form.
        value = _replace_braced_case_expressions(value, variables)
        if not isinstance(value, str):
            return value
        value = _replace_case_expressions(value, variables)
        if not isinstance(value, str):
            return value
        exact_match = TEMPLATE_EXACT_RE.match(value)
        if exact_match:
            key = exact_match.group('braced') or exact_match.group('bare')
            if key not in variables:
                raise ValueError('Unknown case variable reference: {0}'.format(value))
            return variables[key]
        if '$' not in value:
            return value

        def replace_token(match: re.Match) -> str:
            key = match.group('braced') or match.group('bare')
            if key not in variables:
                raise ValueError('Unknown case variable reference: ${0}'.format(key))
            return str(variables[key])

        resolved = TEMPLATE_TOKEN_RE.sub(replace_token, value)
        if '$' in resolved:
            key = resolved[resolved.index('$'):]
            raise ValueError('Unknown case variable reference: {0}'.format(key))
        return resolved
    if isinstance(value, list):
        return [_resolve_template(item, variables) for item in value]
    if isinstance(value, dict):
        return {
            key: _resolve_template(item, variables)
            for key, item in value.items()
        }
    return value


def resolve_templates(payload: Any, variables: Dict[str, Any]) -> Any:
    return _resolve_template(copy.deepcopy(payload), variables)
