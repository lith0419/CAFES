from __future__ import annotations

import functools
import json
from typing import Any, Dict

from .gate_runtime import (
    compiled_gate_after_compile,
    compiled_gate_after_recovery,
    compiled_gate_before_execute,
    compiled_gate_before_finalize,
)
from .module_compilation import module_compiler
from .module_runtime import (
    compiled_input_generator,
    compiled_repair_or_retry,
    compiled_result_analyst,
    compiled_result_extractor,
    compiled_runner,
    compiled_task_reporter,
)
from ..contracts import WorkflowState
from .parsing import intent_parser, spec_builder, spec_validator
from .state import default_state


def run_workflow_sequential(initial_state: Dict[str, Any]) -> Dict[str, Any]:
    state = intent_parser(initial_state)
    state = spec_builder(state)
    state = spec_validator(state)

    if state['validation_errors']:
        state = compiled_runner(state)
    else:
        state = module_compiler(state)
        state = compiled_gate_after_compile(state)
        if state['validation_errors']:
            state = compiled_runner(state)
        else:
            state = compiled_input_generator(state)
            state = compiled_gate_before_execute(state)
            state = compiled_runner(state)

        while not state['validation_errors']:
            next_state = compiled_repair_or_retry(state)
            if next_state['execution_status'] != 'pending':
                state = next_state
                break
            state = compiled_input_generator(next_state)
            state = compiled_gate_before_execute(state)
            state = compiled_runner(state)

    state = compiled_gate_after_recovery(state)
    state = compiled_result_extractor(state)
    state = compiled_result_analyst(state)
    state = compiled_gate_before_finalize(state)
    state = compiled_task_reporter(state)
    return state


@functools.lru_cache(maxsize=1)
def get_workflow():
    return build_workflow()


def run_workflow(initial_state: Dict[str, Any], *, workflow: Any = None) -> Dict[str, Any]:
    workflow_instance = workflow or get_workflow()
    return workflow_instance.invoke(initial_state)


def execute_request(
    user_request: Any,
    *,
    channel: str = 'agent',
    locale: str = 'en',
    workflow: Any = None,
    work_dir: str = None,
    run_id: str = None,
) -> Dict[str, Any]:
    return run_workflow(
        default_state(user_request, channel=channel, locale=locale, work_dir=work_dir, run_id=run_id),
        workflow=workflow,
    )


def build_workflow():
    try:
        from langgraph.graph import END, StateGraph  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError('langgraph is required to build the workflow graph') from exc

    graph = StateGraph(WorkflowState)
    graph.add_node('intent_parser', intent_parser)
    graph.add_node('spec_builder', spec_builder)
    graph.add_node('spec_validator', spec_validator)
    graph.add_node('module_compiler', module_compiler)
    graph.add_node('compilation_gate', compiled_gate_after_compile)
    graph.add_node('input_generator', compiled_input_generator)
    graph.add_node('feasibility_gate', compiled_gate_before_execute)
    graph.add_node('runner', compiled_runner)
    graph.add_node('repair_or_retry', compiled_repair_or_retry)
    graph.add_node('result_quality_gate', compiled_gate_after_recovery)
    graph.add_node('result_extractor', compiled_result_extractor)
    graph.add_node('result_analyst', compiled_result_analyst)
    graph.add_node('finalization_gate', compiled_gate_before_finalize)
    graph.add_node('task_reporter', compiled_task_reporter)

    graph.set_entry_point('intent_parser')
    graph.add_edge('intent_parser', 'spec_builder')
    graph.add_edge('spec_builder', 'spec_validator')
    graph.add_conditional_edges(
        'spec_validator',
        lambda state: 'runner' if state.get('validation_errors') else 'module_compiler',
        {
            'module_compiler': 'module_compiler',
            'runner': 'runner',
        },
    )
    graph.add_edge('module_compiler', 'compilation_gate')
    graph.add_conditional_edges(
        'compilation_gate',
        lambda state: 'runner' if state.get('validation_errors') else 'input_generator',
        {
            'input_generator': 'input_generator',
            'runner': 'runner',
        },
    )
    graph.add_edge('input_generator', 'feasibility_gate')
    graph.add_edge('feasibility_gate', 'runner')
    graph.add_conditional_edges(
        'runner',
        lambda state: (
            'repair_or_retry'
            if state.get('workflow_configuration') and not state.get('validation_errors')
            else 'result_extractor'
        ),
        {
            'repair_or_retry': 'repair_or_retry',
            'result_extractor': 'result_extractor',
        },
    )
    graph.add_conditional_edges(
        'repair_or_retry',
        lambda state: 'input_generator' if state.get('execution_status') == 'pending' else 'result_quality_gate',
        {
            'input_generator': 'input_generator',
            'result_quality_gate': 'result_quality_gate',
        },
    )
    graph.add_edge('result_quality_gate', 'result_extractor')
    graph.add_edge('result_extractor', 'result_analyst')
    graph.add_edge('result_analyst', 'finalization_gate')
    graph.add_edge('finalization_gate', 'task_reporter')
    graph.add_edge('task_reporter', END)
    return graph.compile()


def save_workflow_mermaid(path: str = 'workflow_graph.mmd') -> str:
    workflow = build_workflow()
    mermaid = workflow.get_graph().draw_mermaid()
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(mermaid)
    return path


def example_request() -> str:
    return json.dumps({
        'atom': 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
        'basis': '6-31g',
        'method': 'dft',
        'xc': 'b3lyp',
        'job': 'single_point',
        'outputs': ['energy', 'homo_lumo', 'dipole'],
    })


def main() -> None:
    try:
        output_path = save_workflow_mermaid()
    except ImportError as exc:
        print(f'Unable to export Mermaid graph: {exc}')
    else:
        print(f'Mermaid graph saved to: {output_path}')

    try:
        final_state = execute_request(example_request())
    except ImportError as exc:
        print(f'Unable to execute example request: {exc}')
        return
    except Exception as exc:  # pragma: no cover - demo entrypoint safeguard
        print(f'Example request failed: {exc}')
        return

    print(json.dumps(final_state['task_report'], indent=2, ensure_ascii=False))
