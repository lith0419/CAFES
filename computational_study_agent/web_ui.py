from __future__ import annotations

from pyscf_agent.resources.web import versioned_assets

import json
import html

from pyscf_agent.paths import resolve_work_dir
from pyscf_agent.registry import default_registry
import pyscf_agent.request_builder as llm_request_builder
from computational_study_agent.web_assets.template import HTML_PAGE


DEFAULT_STUDY_SPEC = {
    'name': 'hubbard-u-sweep',
    'objective': 'Sweep the Hubbard U parameter and compare energies',
    'system_type': 'model_hamiltonian',
    'base_model_spec': {
        'schema': 'pyscf-agent.model-hamiltonian.v1',
        'model': 'hubbard',
        'dimension': 1,
        'preset': 'chain',
        'boundary': 'open',
        'energy_unit': 'a.u.',
        'nelec': [1, 1],
        'sites': [
            {'id': 0, 'x': 0, 'y': 0, 'epsilon': 0, 'U': 4},
            {'id': 1, 'x': 1, 'y': 0, 'epsilon': 0, 'U': 4},
        ],
        'bonds': [
            {'id': 0, 'source': 0, 'target': 1, 't': -1, 'V': 0, 'effective_t': -1, 'effective_V': 0},
        ],
    },
    'base_task': {'solver': 'fci'},
    'case_design': {
        'mode': 'grid',
        'variables': {'U_value': [2, 4]},
        'template': {
            'operations': [
                {
                    'op': 'set_global_parameter',
                    'parameter': 'U',
                    'value': '$U_value',
                },
            ],
        },
    },
    'observables': ['energy', 'strong_correlation_diagnostics'],
    'comparison': {'x_axis': 'U_value', 'y_axis': 'energy'},
}

DEFAULT_MOLECULAR_STUDY_SPEC = {
    'name': 'h2-bond-adaptive',
    'objective': 'Scan H2 bond stretching and adaptively route weak to strong correlation regions',
    'system_type': 'molecular',
    'base_task': {
        'atom': 'H 0 0 0; H 0 0 0.74',
        'basis': 'sto-3g',
        'method': 'mp2',
        'restricted': False,
    },
    'case_design': {
        'mode': 'cases',
        'cases': [
            {
                'label': 'R=0.74',
                'variables': {'bond': 0.74},
                'request_updates': {'atom': 'H 0 0 0; H 0 0 0.74'},
            },
            {
                'label': 'R=1.20',
                'variables': {'bond': 1.20},
                'request_updates': {'atom': 'H 0 0 0; H 0 0 1.20'},
            },
            {
                'label': 'R=1.80',
                'variables': {'bond': 1.80},
                'request_updates': {'atom': 'H 0 0 0; H 0 0 1.80'},
            },
            {
                'label': 'R=2.50',
                'variables': {'bond': 2.50},
                'request_updates': {'atom': 'H 0 0 0; H 0 0 2.50'},
            },
        ],
    },
    'observables': ['energy', 'homo_lumo', 'dipole'],
    'comparison': {'x_axis': 'bond', 'y_axis': 'energy'},
}

STUDY_SYSTEM_LABELS = {
    'molecular': 'Molecular',
    'model_hamiltonian': 'Model Hamiltonian',
}
POSTPROCESSING_TOOL_LABELS = {
    'line_plot': 'Line',
    'scatter_plot': 'Scatter',
    'bar_plot': 'Bar',
}
POSTPROCESSING_TOOL_OPTIONS = tuple(
    (
        capability.id,
        POSTPROCESSING_TOOL_LABELS.get(capability.id, capability.label),
    )
    for capability in default_registry().capabilities(
        namespace='postprocessing.tool', backend_allowed=True
    )
)
STUDY_SYSTEM_OPTIONS = tuple(
    (capability.id, STUDY_SYSTEM_LABELS.get(capability.id, capability.label))
    for capability in default_registry().capabilities(
        namespace='task_type', planner_allowed=True
    )
    if capability.id in ('molecular', 'model_hamiltonian')
)
STUDY_MODE_OPTIONS = tuple(
    (capability.id, capability.label)
    for capability in default_registry().capabilities(
        namespace='study.mode', backend_allowed=True
    )
)
ADAPTIVE_INITIAL_SCAN_STRATEGY_OPTIONS = tuple(
    (capability.id, capability.label)
    for capability in default_registry().capabilities(
        namespace='study.adaptive_initial_scan',
        backend_allowed=True,
    )
)
ADAPTIVE_INITIAL_SCAN_STRATEGIES = {
    capability.id: {
        'aliases': list(capability.metadata.get('aliases') or []),
        'execution_method': str(capability.metadata.get('execution_method') or ''),
        'refinement_method': str(capability.metadata.get('refinement_method') or ''),
        'refinement_triggers': list(capability.metadata.get('refinement_triggers') or []),
        'default': bool(capability.metadata.get('default')),
    }
    for capability in default_registry().capabilities(
        namespace='study.adaptive_initial_scan',
        backend_allowed=True,
    )
}
MODEL_SOLVER_OPTIONS = tuple(
    (capability.id, capability.label)
    for capability in default_registry().capabilities(
        namespace='model_hamiltonian.solver',
        planner_allowed=True,
        backend_allowed=True,
    )
)
_DMET_CAPABILITY = default_registry().capability(
    'dmet', namespace='model_hamiltonian.solver'
)
_DMET_SOLVER_OPTIONS = (
    (_DMET_CAPABILITY.metadata.get('solver_options') or {})
    if _DMET_CAPABILITY is not None
    else {}
)
DMET_REFERENCE_DENSITY_OPTIONS = tuple(
    (value, {'pm': 'Paramagnetic', 'af': 'Antiferromagnetic', 'fm': 'Ferromagnetic', 'cdw': 'Charge density wave'}.get(value, value.upper()))
    for value in (
        (_DMET_SOLVER_OPTIONS.get('reference_density_guess') or {}).get('enum')
        or ['pm', 'af', 'fm', 'cdw']
    )
)
ACTIVE_SPACE_SOLVER_OPTIONS = tuple(
    [('auto', 'Auto')] + [
        (capability.id, 'block2 DMRG' if capability.id == 'block2_dmrg' else capability.label)
        for capability in default_registry().capabilities(
            namespace='molecular.active_space_solver', backend_allowed=True
        )
    ]
)
ACTIVE_SPACE_LOCALIZATION_OPTIONS = tuple(
    [('none', 'Canonical / no localization')] + [
        (capability.id, capability.label)
        for capability in default_registry().capabilities(
            namespace='molecular.orbital_processing', backend_allowed=True
        )
        if capability.id in ('boys', 'pipek_mezey')
    ]
)
_BLOCK2_CAPABILITY = default_registry().capability(
    'block2_dmrg', namespace='molecular.active_space_solver'
)
ACTIVE_SPACE_ORDERING_OPTIONS = tuple(
    (ordering, ordering.capitalize())
    for ordering in (
        (_BLOCK2_CAPABILITY.metadata.get('orbital_ordering') or ['canonical'])
        if _BLOCK2_CAPABILITY is not None
        else ['canonical']
    )
)


def _build_options(options) -> str:
    rendered = []
    for option in options:
        value, label = option if isinstance(option, tuple) else (option, option)
        rendered.append(
            '<option value="{value}">{label}</option>'.format(
                value=html.escape(str(value)),
                label=html.escape(str(label)),
            )
        )
    return '\n'.join(rendered)


@versioned_assets
def build_study_index_html(*, work_dir=None) -> str:
    return (
        HTML_PAGE
        .replace('__DEFAULT_STUDY_SPEC__', json.dumps(DEFAULT_STUDY_SPEC, ensure_ascii=False))
        .replace('__DEFAULT_MOLECULAR_STUDY_SPEC__', json.dumps(DEFAULT_MOLECULAR_STUDY_SPEC, ensure_ascii=False))
        .replace('__DEFAULT_WORK_DIR__', json.dumps(str(resolve_work_dir(work_dir)), ensure_ascii=False).replace('<', '\\u003c'))
        .replace('__LLM_CONFIGURED__', 'true' if llm_request_builder.llm_request_builder_is_configured() else 'false')
        .replace('__STUDY_SYSTEM_OPTIONS__', _build_options(STUDY_SYSTEM_OPTIONS))
        .replace('__STUDY_MODE_OPTIONS__', _build_options(STUDY_MODE_OPTIONS))
        .replace('__ADAPTIVE_INITIAL_SCAN_STRATEGY_OPTIONS__', _build_options(ADAPTIVE_INITIAL_SCAN_STRATEGY_OPTIONS))
        .replace('__ADAPTIVE_INITIAL_SCAN_STRATEGIES__', json.dumps(ADAPTIVE_INITIAL_SCAN_STRATEGIES, ensure_ascii=False))
        .replace('__MODEL_SOLVER_OPTIONS__', _build_options(MODEL_SOLVER_OPTIONS))
        .replace('__DMET_REFERENCE_DENSITY_OPTIONS__', _build_options(DMET_REFERENCE_DENSITY_OPTIONS))
        .replace('__ACTIVE_SPACE_SOLVER_OPTIONS__', _build_options(ACTIVE_SPACE_SOLVER_OPTIONS))
        .replace('__ACTIVE_SPACE_LOCALIZATION_OPTIONS__', _build_options(ACTIVE_SPACE_LOCALIZATION_OPTIONS))
        .replace('__ACTIVE_SPACE_ORDERING_OPTIONS__', _build_options(ACTIVE_SPACE_ORDERING_OPTIONS))
        .replace('__POSTPROCESSING_TOOL_OPTIONS__', _build_options(POSTPROCESSING_TOOL_OPTIONS))
    )
