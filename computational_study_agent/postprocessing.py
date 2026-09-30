from __future__ import annotations

from pyscf_agent.serialization import json_default

import copy
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from pyscf_agent.artifacts import default_artifact_repository
from pyscf_agent.registry import default_registry
from pyscf_agent.input_validation import boolean, finite_float, integer, reject_unknown_fields

from computational_study_agent.datasets.hamiltonian.postprocessing import (
    HAMILTONIAN_DATASET_COLLECT_ACTION,
    HAMILTONIAN_DATASET_GENERATE_ACTION,
    collect_hamiltonian_dataset,
    generate_hamiltonian_dataset,
)
from .schema import StudyReport
from .trajectory_selection import ENERGY_SELECTION_ACTION, select_energy_stratified_frames


NUMERIC_WITH_UNIT_PATTERN = re.compile(
    r'^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*([A-Za-z][A-Za-z0-9./()_-]*)?\s*$'
)
DEFAULT_FIGURE_STYLE = 'pyscf_agent_classic'
DEFAULT_FIGURE_DPI = 600
DEFAULT_FIGURE_SIZE = (16.0 / 2.54, 12.8 / 2.54)  # Matplotlib uses inches.
DEFAULT_HEATMAP_SIZE = (16.0 / 2.54, 16.0 / 2.54)
DEFAULT_FIGURE_FONT = 'Arial'
PLOT_ARTIFACT_PREFIX = 'plot-'
POSTPROCESSING_INTERNAL_COLUMNS = frozenset({
    'case_id',
    'label',
    'status',
    'run_dir',
    'execution_source',
    'attempt_count',
    'quality_status',
    'publication_eligible',
    'grid_origin', 'grid_round', 'grid_axis',
})
SUPPORTED_PLOT_TOOLS = tuple(
    item.id
    for item in default_registry().capabilities(
        namespace='postprocessing.tool', backend_allowed=True
    )
)
SUPPORTED_POSTPROCESSING_ACTIONS = tuple(
    item.id
    for item in default_registry().capabilities(
        namespace='postprocessing.action', backend_allowed=True
    )
)
MOLECULAR_BOND_LENGTH_COLUMNS = {'bond', 'bond_length', 'bond_distance'}
MODEL_POSTPROCESSING_METRICS = tuple(
    default_registry().observables(
        namespace='postprocessing.metric',
        system_type='model_hamiltonian',
        backend_allowed=True,
    )
)
MODEL_POSTPROCESSING_METRIC_PRIORITY = {
    (item.comparison_field or item.id): int(item.metadata.get('plot_priority', 100))
    for item in MODEL_POSTPROCESSING_METRICS
}


@dataclass
class PostprocessContext:
    study_id: str
    name: str
    objective: str
    system_type: str
    work_dir: str
    comparison_table: List[Dict[str, Any]] = field(default_factory=list)
    cases: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_report(cls, report: Any) -> 'PostprocessContext':
        payload = vars(report) if isinstance(report, StudyReport) else report
        if not isinstance(payload, dict):
            raise TypeError('PostprocessContext requires a StudyReport or report dictionary')
        comparison_table = payload.get('comparison_table')
        cases = payload.get('cases')
        artifacts = payload.get('artifacts')
        rows = comparison_table if isinstance(comparison_table, list) else []
        case_records = {str(case.get('case_id')): case for case in (cases or []) if isinstance(case, dict)}
        eligible_rows, eligible_indexes, excluded_rows = [], [], []
        for index, row in enumerate(rows):
            reason = None
            if not isinstance(row, dict):
                reason = 'row_is_not_an_object'
                row = {}
            case = case_records.get(str(row.get('case_id')), {})
            task = case.get('task_report') or {}
            statuses = [row.get('status'), task.get('execution_status')]
            if any(str(status).strip().lower() not in ('succeeded', 'completed', 'converged')
                   for status in statuses if status is not None and str(status).strip()):
                reason = 'execution_not_successful'
            eligibility = row.get('publication_eligible')
            if eligibility is False:
                reason = 'publication_ineligible'
            elif eligibility is not None and eligibility is not True:
                reason = 'invalid_publication_eligibility'
            if reason:
                excluded_rows.append({'row_index': index, 'case_id': row.get('case_id'), 'reason': reason})
            else:
                eligible_rows.append(row)
                eligible_indexes.append(index)
        return cls(
            study_id=str(payload.get('study_id') or 'study'),
            name=str(payload.get('name') or 'computational-study'),
            objective=str(payload.get('objective') or ''),
            system_type=str(payload.get('system_type') or ''),
            work_dir=str(payload.get('work_dir') or ''),
            comparison_table=copy.deepcopy(eligible_rows),
            cases=[_postprocess_case(case) for case in (cases or []) if isinstance(case, dict)],
            artifacts=copy.deepcopy(artifacts if isinstance(artifacts, list) else []),
            metadata={
                'status': payload.get('status'),
                'summary': payload.get('summary'),
                'grid_refinement': copy.deepcopy(payload.get('grid_refinement') or {}),
                'excluded_rows': excluded_rows,
                'row_indexes': eligible_indexes,
                'excluded_case_ids': [
                    str(row.get('case_id'))
                    for row in excluded_rows
                    if row.get('case_id') is not None
                ],
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def view(self) -> Dict[str, Any]:
        """The same eligible data and column choices used by every client."""
        columns = [key for key in _ordered_columns(self.comparison_table)
                   if key not in POSTPROCESSING_INTERNAL_COLUMNS]
        return {
            'rows': copy.deepcopy(self.comparison_table),
            'columns': columns,
            'numeric_columns': [key for key in columns if _is_numeric_column(self.comparison_table, key)],
            'case_variable_columns': _case_variable_columns(self, columns),
            'excluded_rows': copy.deepcopy(self.metadata.get('excluded_rows') or []),
        }


def _postprocess_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """Plots need variables and coordinate units, not wavefunctions or logs."""
    task = case.get('task_report') or {}
    result = {key: copy.deepcopy(case[key]) for key in ('case_id', 'label', 'variables') if key in case}
    result['task_report'] = {'execution_status': task.get('execution_status')}
    for source, target, key in ((case, result, 'request'), (task, result['task_report'], 'task_spec')):
        request = source.get(key)
        if isinstance(request, dict):
            system = request.get('system') or {}
            unit = system.get('unit') or request.get('unit')
            if unit:
                target[key] = {'system': {'unit': unit}}
    return result


def _safe_slug(value: Any, default: str = 'plot') -> str:
    slug = ''.join(character if character.isalnum() or character in ('-', '_') else '-' for character in str(value or ''))
    slug = '-'.join(part for part in slug.strip('-_').split('-') if part)
    return slug or default


def _artifact_ref(
    path: Path,
    *,
    kind: str,
    mime_type: str,
    description: str = '',
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    repository = default_artifact_repository().with_reference_root(root)
    reference = repository.register_existing(
        path,
        kind=kind,
        mime_type=mime_type,
        description=description,
    )
    if reference is None:
        raise FileNotFoundError('Postprocessing artifact is unavailable: {0}'.format(path))
    return reference


def _write_json(
    path: Path,
    payload: Any,
    *,
    kind: str,
    description: str = '',
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    return default_artifact_repository().with_reference_root(root).write_json(
        path,
        payload,
        kind=kind,
        description=description,
    )


def _write_text(
    path: Path,
    content: str,
    *,
    kind: str,
    description: str = '',
    root: Optional[Path] = None,
    mime_type: str = 'text/plain; charset=utf-8',
) -> Dict[str, Any]:
    return default_artifact_repository().with_reference_root(root).write_text(
        path,
        content,
        kind=kind,
        mime_type=mime_type,
        description=description,
    )


def _parse_numeric_value(value: Any) -> Tuple[Optional[float], Optional[str]]:
    if isinstance(value, bool) or value is None:
        return None, None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric, None
        return None, None
    if isinstance(value, str):
        match = NUMERIC_WITH_UNIT_PATTERN.match(value)
        if not match:
            return None, None
        try:
            numeric = float(match.group(1))
        except (TypeError, ValueError):
            return None, None
        if not math.isfinite(numeric):
            return None, None
        return numeric, match.group(2)
    return None, None


def _normalized_column_key(column: Any) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(column or '').strip().lower()).strip('_')


def _row_unit(context: PostprocessContext, row: Dict[str, Any], column: str) -> Optional[str]:
    _value, unit = _parse_numeric_value(row.get(column))
    if unit:
        return unit
    if context.system_type == 'molecular' and _normalized_column_key(column) in MOLECULAR_BOND_LENGTH_COLUMNS:
        case = next((case for case in context.cases if isinstance(case, dict)
                     and row.get('case_id') is not None and case.get('case_id') == row['case_id']), {})
        task = case.get('task_report') or {}
        # A stored TaskSpec carries its actual coordinate unit. A legacy bare
        # comparison column does not provide enough evidence to infer one.
        for request in (task.get('task_spec'), case.get('request')):
            if isinstance(request, dict):
                system = request.get('system') or {}
                unit = system.get('unit') or request.get('unit')
                if unit:
                    return unit
    return None


def _column_unit(context: PostprocessContext, column: str) -> Optional[str]:
    units = {_row_unit(context, row, column) for row in context.comparison_table
             if _parse_numeric_value(row.get(column))[0] is not None}
    if len(units) > 1:
        raise ValueError('Column {0} has mixed or unknown units ({1}); provide a consistent unit before plotting.'.format(
            column, ', '.join(sorted(str(unit) if unit else 'unspecified' for unit in units))))
    return next(iter(units), None)


def _is_numeric_column(rows: Sequence[Dict[str, Any]], column: str) -> bool:
    return any(_parse_numeric_value(row.get(column))[0] is not None for row in rows)


def _ordered_columns(rows: Sequence[Dict[str, Any]]) -> List[str]:
    columns: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for column in row:
            if column not in columns:
                columns.append(column)
    return columns


def _string_key(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=json_default, allow_nan=False)
    return str(value)


def _distinct_values(rows: Sequence[Dict[str, Any]], column: str) -> List[str]:
    values: List[str] = []
    for row in rows:
        if not isinstance(row, dict) or column not in row:
            continue
        value = row.get(column)
        if value is None or value == '':
            continue
        key = _string_key(value)
        if key not in values:
            values.append(key)
    return values


def _varies(rows: Sequence[Dict[str, Any]], column: str) -> bool:
    return len(_distinct_values(rows, column)) > 1


def _case_variable_columns(context: PostprocessContext, columns: Sequence[str]) -> List[str]:
    variable_columns: List[str] = []
    for case in context.cases:
        if not isinstance(case, dict):
            continue
        variables = case.get('variables')
        if not isinstance(variables, dict):
            continue
        for column in variables:
            if column in columns and column not in variable_columns:
                variable_columns.append(column)
    return variable_columns


def _axis_label(column: str, unit: Optional[str] = None) -> str:
    labels = {
        'bond': 'Bond length',
        'bond_length': 'Bond length',
        'bond_distance': 'Bond length',
        'energy': 'Total energy',
        'final_energy': 'Energy',
        'energy_per_site': 'Energy per site',
        'energy_over_abs_t': 'Energy / |t|',
        'energy_per_site_over_abs_t': 'Energy per site / |t|',
        'gap': 'Energy gap',
        'reference_energy': 'Reference energy',
        'mean_field_gap': 'Mean-field gap',
        'natural_occupation_fractionality': 'Natural occupation fractionality',
        'fractional_natural_orbital_count': 'Fractional natural orbital count',
        'max_double_excitation_amplitude': 'Max double-excitation amplitude',
        'mean_double_occupancy': 'Mean double occupancy',
        'nearest_neighbor_spin_correlation': 'Spin correlation',
        'nearest_neighbor_charge_correlation': 'Charge correlation',
        'sublattice_charge_imbalance': 'Sublattice charge imbalance',
        'staggered_magnetization': 'Staggered magnetization',
        'mean_nearest_neighbor_one_body_coherence': 'One-body coherence',
        'dmrg_state_count': 'DMRG state count',
        'first_excitation_energy': 'First excitation energy',
        'max_single_orbital_entropy': 'Max single-orbital entropy',
        'mean_single_orbital_entropy': 'Mean single-orbital entropy',
        'max_mutual_information': 'Max orbital mutual information',
        'max_bipartite_entanglement': 'Max bipartite entanglement',
        'dmrg_spin_square': 'DMRG <S^2>',
        'dmrg_inferred_total_spin': 'DMRG inferred total spin',
        'dmrg_spin_square_deviation': 'DMRG spin-square deviation',
        'dmrg_final_discarded_weight': 'DMRG final discarded weight',
        'dmrg_final_energy_change': 'DMRG final energy change',
        'dmrg_estimated_energy_error': 'Estimated DMRG energy error',
        'dmrg_extrapolated_energy': 'Extrapolated DMRG energy',
        'dmrg_adaptive_stages': 'Adaptive DMRG stages',
        'dmrg_final_bond_dimension': 'Final DMRG bond dimension',
        'density': 'Density',
        'density_mean': 'Mean density',
        'energy_level_count': 'Energy level count',
        'many_body_basis_dimension': 'Many-body basis dimension',
        'gap': 'Gap',
        'U': 'U',
        'V': 'V',
        't': 't',
        'epsilon': 'epsilon',
    }
    column_text = str(column)
    label = labels.get(column_text, labels.get(_normalized_column_key(column), column_text.replace('_', ' ')))
    # Units remain in the spec and data exports, not in display titles.
    return label


def _plot_artifact_stem(output: Any) -> str:
    slug = _safe_slug(output, default='plot')
    return slug if slug.startswith(PLOT_ARTIFACT_PREFIX) else '{0}{1}'.format(PLOT_ARTIFACT_PREFIX, slug)


def _normalized_plot_spec(context: PostprocessContext, spec: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise TypeError('plot spec must be a dictionary')
    reject_unknown_fields(spec, {
        'schema', 'tool', 'kind', 'x', 'y', 'color', 'z', 'group', 'title', 'xlabel', 'ylabel',
        'colorbar_label', 'colormap', 'center', 'output', 'style', 'sort_by_x', 'figure_size', 'dpi', 'units',
        'local_cells',  # Exported derived metadata; always rebuilt from the report below.
    }, 'plot_spec')
    for key in ('tool', 'kind', 'x', 'y', 'color', 'z', 'group', 'title', 'xlabel', 'ylabel',
                'colorbar_label', 'colormap', 'output', 'style'):
        if key in spec and spec[key] is not None and not isinstance(spec[key], str):
            raise ValueError('plot_spec.{0} must be a string'.format(key))
    tool = str(spec.get('tool') or spec.get('kind') or 'line_plot').strip().lower()
    if tool not in SUPPORTED_PLOT_TOOLS:
        raise ValueError('Unsupported postprocessing plot tool: {0}'.format(tool))
    x_column = str(spec.get('x') or '').strip()
    y_column = str(spec.get('y') or '').strip()
    color_column = str(spec.get('color') or spec.get('z') or '').strip()
    group_column = str(spec.get('group') or '').strip()
    if not x_column or not y_column:
        raise ValueError('plot spec requires x and y columns')
    internal_columns = {
        column
        for column in (x_column, y_column, color_column, group_column)
        if column in POSTPROCESSING_INTERNAL_COLUMNS
    }
    if internal_columns:
        raise ValueError(
            'Plot specs cannot use internal execution metadata: {0}'.format(
                ', '.join(sorted(internal_columns))
            )
        )
    if not context.comparison_table:
        raise ValueError('comparison_table is empty')
    columns = set()
    for row in context.comparison_table:
        if isinstance(row, dict):
            columns.update(row.keys())
    if x_column not in columns:
        raise ValueError('Unknown x column: {0}'.format(x_column))
    if y_column not in columns:
        raise ValueError('Unknown y column: {0}'.format(y_column))
    if group_column and group_column not in columns:
        raise ValueError('Unknown group column: {0}'.format(group_column))
    if tool == 'heatmap':
        if not color_column:
            raise ValueError('heatmap plot spec requires a color column')
        if color_column not in columns:
            raise ValueError('Unknown color column: {0}'.format(color_column))
        if group_column:
            raise ValueError('Heatmap requires one series; select a single group before plotting.')

    y_unit = _column_unit(context, y_column)
    x_unit = _column_unit(context, x_column)
    color_unit = _column_unit(context, color_column) if tool == 'heatmap' else None
    sort_by_x = boolean(spec.get('sort_by_x', True), 'plot_spec.sort_by_x')
    dpi = integer(spec.get('dpi', DEFAULT_FIGURE_DPI), 'plot_spec.dpi')
    if dpi <= 0:
        raise ValueError('plot_spec.dpi must be positive')
    default_size = DEFAULT_HEATMAP_SIZE if tool == 'heatmap' else DEFAULT_FIGURE_SIZE
    figure_size = spec.get('figure_size', default_size)
    if not isinstance(figure_size, (list, tuple)) or len(figure_size) != 2:
        raise ValueError('plot_spec.figure_size must contain width and height')
    figure_size = [finite_float(value, 'plot_spec.figure_size') for value in figure_size]
    if any(value <= 0 for value in figure_size):
        raise ValueError('plot_spec.figure_size values must be positive')
    style = spec.get('style', DEFAULT_FIGURE_STYLE)
    if style != DEFAULT_FIGURE_STYLE:
        raise ValueError('Unsupported plot style: {0}'.format(style))
    output_default = '{0}-heatmap-{1}-{2}'.format(color_column, x_column, y_column) if tool == 'heatmap' else '{0}-vs-{1}'.format(y_column, x_column)
    output = _safe_slug(spec.get('output') or output_default)
    plotted_quantity = color_column if tool == 'heatmap' else y_column
    plotted_unit = color_unit if tool == 'heatmap' else None
    normalized = {
        'tool': tool,
        'x': x_column,
        'y': y_column,
        'color': color_column if tool == 'heatmap' else spec.get('color'),
        'group': group_column or None,
        'title': str(spec.get('title') or _axis_label(plotted_quantity, plotted_unit)),
        'xlabel': str(spec.get('xlabel') or _axis_label(x_column, x_unit)),
        'ylabel': str(spec.get('ylabel') or _axis_label(y_column, y_unit)),
        'colorbar_label': str(spec.get('colorbar_label') or ''),
        'colormap': str(spec.get('colormap') or 'RdBu_r') if tool == 'heatmap' else spec.get('colormap'),
        'center': finite_float(spec.get('center', 0.0), 'plot_spec.center') if tool == 'heatmap' else spec.get('center'),
        'output': output,
        'style': style,
        'sort_by_x': sort_by_x,
        'figure_size': figure_size,
        'dpi': dpi,
        'units': {column: unit for column, unit in ((x_column, x_unit), (y_column, y_unit),
                  (color_column, color_unit)) if column},
    }
    for key in ('title', 'xlabel', 'ylabel', 'colorbar_label'):
        label = re.sub(r'^nearest[- ]neighbor\s+', '', normalized[key], flags=re.IGNORECASE)
        if label != normalized[key]:
            label = label[:1].upper() + label[1:]
        for unit in (x_unit, y_unit, color_unit):
            if unit:
                label = label.replace(' ({0})'.format(unit), '').replace(' [{0}]'.format(unit), '')
        normalized[key] = label
    grid = context.metadata.get('grid_refinement') or {}
    if (tool == 'heatmap' and grid.get('strategy') == 'local_cells'
            and set((grid.get('policy') or {}).get('axes', [])) == {x_column, y_column}):
        cells = grid.get('cells') or []
        if len({json.dumps(cell['group'], sort_keys=True) for cell in cells}) > 1:
            raise ValueError('Heatmap requires one fixed-parameter group; select a single group before plotting.')
        normalized['local_cells'] = copy.deepcopy(cells)
    return normalized


def _plot_data_columns(spec: Dict[str, Any]) -> List[str]:
    columns = ['row_index', 'case_id', 'label', 'status', 'grid_origin', 'grid_round', 'grid_axis']
    for column in (spec.get('x'), spec.get('y'), spec.get('color') if spec.get('tool') == 'heatmap' else None, spec.get('group')):
        if column and column not in columns:
            columns.append(str(column))
    for suffix_column in (spec.get('x'), spec.get('y'), spec.get('color') if spec.get('tool') == 'heatmap' else None):
        if not suffix_column:
            continue
        column = str(suffix_column)
        for suffix in ('numeric', 'unit'):
            derived = '{0}_{1}'.format(column, suffix)
            if derived not in columns:
                columns.append(derived)
    return columns


def _local_heatmap_cells(rows, spec):
    by_id = {row.get('case_id'): row for row in rows}
    def value(case_id):
        return _parse_numeric_value(by_id.get(case_id, {}).get(spec['color']))[0] if case_id else None
    result = []
    for cell in spec.get('local_cells') or []:
        corners = [value(case_id) for case_id in cell['corner_case_ids']]
        center_id = cell.get('center_case_id')
        color = None
        source = 'missing_results'
        if len(corners) == 4 and all(v is not None for v in corners):
            if center_id:
                color = value(center_id)
                source = 'computed_center' if color is not None else 'missing_results'
            else:
                color = sum(corners) / 4
                source = 'bilinear_corner_mean'
        result.append({**copy.deepcopy(cell), 'color_value': color, 'value_source': source})
    return result


def _plot_data_payload(context: PostprocessContext, spec: Dict[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    excluded = copy.deepcopy(context.metadata.get('excluded_rows') or [])
    coordinates = set()
    required_numeric_columns = [spec['x'], spec['y']]
    if spec.get('tool') == 'heatmap':
        required_numeric_columns.append(spec['color'])

    for row_index, row in enumerate(context.comparison_table):
        row_indexes = context.metadata.get('row_indexes')
        source_index = row_indexes[row_index] if row_indexes is not None else row_index
        if not isinstance(row, dict):
            continue
        parsed: Dict[str, Tuple[Optional[float], Optional[str]]] = {
            column: _parse_numeric_value(row.get(column))
            for column in required_numeric_columns
        }
        invalid_columns = [column for column, (value, _unit) in parsed.items() if value is None]
        if spec.get('group') and row.get(spec['group']) in (None, ''):
            invalid_columns.append(spec['group'])
        if invalid_columns:
            excluded.append({'case_id': row.get('case_id'), 'row_index': source_index,
                             'reason': 'missing_or_invalid_plot_value', 'columns': invalid_columns})
            continue
        if spec['tool'] == 'heatmap':
            coordinate = (parsed[spec['x']][0], parsed[spec['y']][0])
            if coordinate in coordinates:
                raise ValueError('Duplicate heatmap coordinate {0}; select one series or aggregate explicitly before plotting.'.format(coordinate))
            coordinates.add(coordinate)
        payload_row: Dict[str, Any] = {
            'row_index': source_index,
            'case_id': row.get('case_id'),
            'label': row.get('label'),
            'status': row.get('status'),
        }
        for name in ('grid_origin', 'grid_round', 'grid_axis'):
            if name in row:
                payload_row[name] = row[name]
        for column in (spec.get('x'), spec.get('y'), spec.get('color') if spec.get('tool') == 'heatmap' else None, spec.get('group')):
            if column:
                payload_row[str(column)] = row.get(str(column))
        for column, (numeric, unit) in parsed.items():
            payload_row['{0}_numeric'.format(column)] = numeric
            payload_row['{0}_unit'.format(column)] = _row_unit(context, row, column)
        rows.append(payload_row)

    if not rows:
        raise ValueError('No numeric rows are available for the selected plot columns')

    columns = [
        column for column in _plot_data_columns(spec)
        if any(row.get(column) not in (None, '') for row in rows)
    ]
    return {
        **({'cells': _local_heatmap_cells(rows, spec)} if 'local_cells' in spec else {}),
        'kind': 'postprocess_plot_data',
        'study_id': context.study_id,
        'plot_output': spec['output'],
        'plot_artifact_prefix': PLOT_ARTIFACT_PREFIX,
        'plot_spec': copy.deepcopy(spec),
        'columns': columns,
        'rows': rows,
        'row_count': len(rows),
        'excluded_rows': excluded,
    }


def _tsv_cell(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=json_default, allow_nan=False)
    else:
        text = str(value)
    return text.replace('\t', ' ').replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')


def _plot_data_tsv(payload: Dict[str, Any]) -> str:
    columns = payload.get('columns') if isinstance(payload.get('columns'), list) else []
    rows = payload.get('rows') if isinstance(payload.get('rows'), list) else []
    if not columns:
        return ''
    lines = ['\t'.join(str(column) for column in columns)]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append('\t'.join(_tsv_cell(row.get(column)) for column in columns))
    return '\n'.join(lines) + '\n'


def _has_sampled_grid(rows: Sequence[Dict[str, Any]], x: str, y: str, color: str) -> bool:
    """Require observed crossed sampling, not merely two changing columns.

    A rectangle is conservative evidence for a grid (possibly incomplete).
    Paired coordinates and irregular point clouds are not automatically grids.
    Repeated coordinates need an explicit slice or aggregation before a heatmap.
    """
    by_x: Dict[float, set] = {}
    for row in rows:
        values = [_parse_numeric_value(row.get(column))[0] for column in (x, y, color)]
        if any(value is None for value in values):
            continue
        x_value, y_value, _ = values
        ys = by_x.setdefault(x_value, set())
        if y_value in ys:
            return False
        ys.add(y_value)
    # Finding the same y pair at two x values establishes an observed rectangle.
    seen_pairs = set()
    for ys in by_x.values():
        ordered = sorted(ys)
        for i, first in enumerate(ordered):
            for second in ordered[i + 1:]:
                pair = (first, second)
                if pair in seen_pairs:
                    return True
                seen_pairs.add(pair)
    return False


def _supports_plot_groups(rows: Sequence[Dict[str, Any]], x: str, y: str, group: str) -> bool:
    """Only propose grouping when every displayed group has a real x sequence."""
    groups: Dict[str, set] = {}
    for row in rows:
        x_value, _ = _parse_numeric_value(row.get(x))
        y_value, _ = _parse_numeric_value(row.get(y))
        group_value = row.get(group)
        if x_value is None or y_value is None:
            continue
        if group_value in (None, ''):
            return False
        groups.setdefault(_string_key(group_value), set()).add(x_value)
    return len(groups) >= 2 and all(len(xs) >= 2 for xs in groups.values())


def suggest_plot_specs(report: Any, *, max_specs: int = 4) -> List[Dict[str, Any]]:
    context = PostprocessContext.from_report(report)
    rows = context.comparison_table
    if not rows or max_specs <= 0:
        return []
    columns = [
        column
        for column in _ordered_columns(rows)
        if column not in POSTPROCESSING_INTERNAL_COLUMNS
    ]
    numeric_columns = [column for column in columns if _is_numeric_column(rows, column)]
    if len(numeric_columns) < 2:
        return []

    case_variable_columns = _case_variable_columns(context, columns)
    varying_case_variables = [
        column
        for column in case_variable_columns
        if column in numeric_columns and _varies(rows, column)
    ]
    x_columns = varying_case_variables[:]
    if not x_columns:
        fallback_x = next((column for column in numeric_columns if _varies(rows, column)), numeric_columns[0])
        x_columns = [fallback_x]

    y_columns = [
        column
        for column in numeric_columns
        if column not in case_variable_columns
    ]
    if 'final_energy' in y_columns:
        y_columns = [column for column in y_columns if column != 'energy']
    preferred_y_order = {
        'final_energy': 0,
        'energy': 1,
        **MODEL_POSTPROCESSING_METRIC_PRIORITY,
        'gap': 11,
        'first_excitation_energy': 12,
        'max_single_orbital_entropy': 20,
        'max_mutual_information': 21,
        'max_bipartite_entanglement': 22,
        'dmrg_final_discarded_weight': 30,
        'dmrg_estimated_energy_error': 31,
        'dmrg_extrapolated_energy': 32,
        'dmrg_final_bond_dimension': 33,
        'dmrg_adaptive_stages': 34,
    }
    y_columns.sort(key=lambda column: (preferred_y_order.get(column, 100), columns.index(column)))
    if not y_columns:
        y_columns = [column for column in numeric_columns if column not in case_variable_columns and column not in x_columns]
    if not x_columns:
        x_columns = [column for column in numeric_columns if column not in y_columns]
    if not x_columns or not y_columns:
        return []

    specs = []
    seen = set()
    for index, x_column in enumerate(varying_case_variables):
        for y_axis_column in varying_case_variables[index + 1:]:
            for color_column in y_columns:
                if not _has_sampled_grid(rows, x_column, y_axis_column, color_column):
                    continue
                specs.append(_normalized_plot_spec(context, {
                    'tool': 'heatmap',
                    'x': x_column,
                    'y': y_axis_column,
                    'color': color_column,
                    'output': '{0}-heatmap-{1}-{2}'.format(color_column, x_column, y_axis_column),
                }))
                if len(specs) >= max_specs:
                    return specs

    if specs and len(varying_case_variables) == 2 and context.metadata.get('grid_refinement'):
        return specs

    for x_column in x_columns:
        for y_column in y_columns:
            if x_column == y_column:
                continue
            key = (x_column, y_column)
            if key in seen:
                continue
            seen.add(key)
            group_column = next((column for column in case_variable_columns
                                 if column != x_column
                                 and _supports_plot_groups(rows, x_column, y_column, column)), None)
            # Without a declared single coordinate or repeated series, the
            # ordering of a multi-coordinate design is ambiguous. Do not invent
            # a path by connecting unrelated points in sorted-x order.
            tool = 'scatter_plot' if len(varying_case_variables) > 1 and group_column is None else 'line_plot'
            specs.append(_normalized_plot_spec(context, {
                'tool': tool,
                'x': x_column,
                'y': y_column,
                'group': group_column,
                'output': '{0}-vs-{1}'.format(y_column, x_column),
            }))
            if len(specs) >= max_specs:
                return specs
    return specs


def _load_matplotlib():
    if not os.environ.get('XDG_CACHE_HOME'):
        xdg_cache_dir = Path(tempfile.gettempdir()) / 'pyscf-agent-cache'
        xdg_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ['XDG_CACHE_HOME'] = str(xdg_cache_dir)
    if not os.environ.get('MPLCONFIGDIR'):
        mpl_config_dir = Path(tempfile.gettempdir()) / 'pyscf-agent-matplotlib'
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ['MPLCONFIGDIR'] = str(mpl_config_dir)
    try:
        import matplotlib  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError('matplotlib is required for study postprocessing plots') from exc
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    from matplotlib.ticker import AutoMinorLocator  # pylint: disable=import-outside-toplevel
    return plt, AutoMinorLocator


def _apply_classic_style(ax: Any, *, xlabel: str, ylabel: str, title: str = '') -> None:
    _plt, AutoMinorLocator = _load_matplotlib()
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_linewidth(4.0)
        spine.set_color('black')
    ax.tick_params(
        axis='both',
        which='major',
        direction='in',
        length=8,
        width=4.0,
        colors='black',
        labelsize=22,
        top=True,
        right=True,
    )
    ax.tick_params(
        axis='both',
        which='minor',
        direction='in',
        length=5,
        width=4.0,
        colors='black',
        top=True,
        right=True,
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(xlabel, fontsize=32, fontweight='bold', labelpad=2)
    ax.set_ylabel(ylabel, fontsize=32, fontweight='bold', labelpad=2)
    if title:
        ax.figure.suptitle(title, fontsize=32, fontweight='bold')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    ax.grid(False)


def _apply_colorbar_style(colorbar: Any) -> None:
    colorbar.ax.tick_params(
        which='major',
        direction='in',
        length=8,
        width=4.0,
        colors='black',
        labelsize=22,
    )
    colorbar.outline.set_linewidth(4.0)
    colorbar.outline.set_edgecolor('black')
    colorbar.ax.yaxis.label.set_fontsize(32)
    colorbar.ax.yaxis.label.set_fontweight('bold')
    for label in colorbar.ax.get_yticklabels():
        label.set_fontweight('bold')


def _save_styled_figure(fig: Any, spec: Dict[str, Any], output_path: Path) -> None:
    from matplotlib.text import Text  # pylint: disable=import-outside-toplevel

    for text in fig.findobj(Text):
        text.set_fontfamily(DEFAULT_FIGURE_FONT)
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            axis.get_offset_text().set_fontsize(22)
            axis.get_offset_text().set_fontweight('bold')

    # Wrap long titles without shrinking the requested font or enlarging the canvas.
    title = fig._suptitle
    if title is not None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        available_width = fig.bbox.width - 0.3 * fig.dpi
        lines = []
        for paragraph in title.get_text().split('\n'):
            line = ''
            for word in paragraph.split():
                candidate = (line + ' ' + word).strip()
                width, _, _ = renderer.get_text_width_height_descent(
                    candidate, title.get_fontproperties(), ismath=False,
                )
                if line and width > available_width:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
        title.set_text('\n'.join(lines))
    fig.tight_layout()
    if title is not None:
        # Place the title just above the axes decorations, avoiding endpoint labels.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axes_top = max(ax.get_tightbbox(renderer).y1 for ax in fig.axes)
        title.set_verticalalignment('bottom')
        title.set_y((axes_top + 3.0 * fig.dpi / 72.0) / fig.bbox.height)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Tight bounding boxes change the physical size; keep the full fixed canvas.
    fig.savefig(output_path, dpi=int(spec.get('dpi') or DEFAULT_FIGURE_DPI), facecolor='white')


def _plot_grid_markers(ax, rows, x_column, y_column):
    plotted = False
    for origin, label, marker, color in (
        ('seed', 'Initial points', 's', 'black'),
        ('refined', 'Added points', 'o', '#c25416'),
    ):
        points = []
        for row in rows:
            if row.get('grid_origin') != origin:
                continue
            x, _ = _parse_numeric_value(row.get(x_column))
            y, _ = _parse_numeric_value(row.get(y_column))
            if x is not None and y is not None:
                points.append((x, y))
        if points:
            ax.scatter([p[0] for p in points], [p[1] for p in points], marker=marker,
                       s=38, facecolors='white', edgecolors=color, linewidths=1.3,
                       label=label, zorder=5)
            plotted = True
    return plotted


def _plot_heatmap(
    rows: Sequence[Dict[str, Any]],
    spec: Dict[str, Any],
    output_path: Path,
    *,
    artifact_root: Optional[Path] = None,
) -> Dict[str, Any]:
    plt, _AutoMinorLocator = _load_matplotlib()
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        from matplotlib.colors import Normalize, TwoSlopeNorm, ListedColormap  # pylint: disable=import-outside-toplevel
        from matplotlib.ticker import MaxNLocator  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError('numpy and matplotlib are required for study postprocessing heatmaps') from exc

    points: List[Tuple[float, float, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        x_value, _x_unit = _parse_numeric_value(row.get(spec['x']))
        y_value, _y_unit = _parse_numeric_value(row.get(spec['y']))
        color_value, _color_unit = _parse_numeric_value(row.get(spec['color']))
        if x_value is None or y_value is None or color_value is None:
            continue
        points.append((x_value, y_value, color_value))

    if not points:
        raise ValueError('No numeric rows are available for heatmap color {0} over {1}/{2}'.format(spec['color'], spec['x'], spec['y']))

    x_values = sorted({point[0] for point in points})
    y_values = sorted({point[1] for point in points})
    if len(x_values) < 2 or len(y_values) < 2:
        raise ValueError('Heatmap requires at least two distinct x and y values')

    x_index = {value: index for index, value in enumerate(x_values)}
    y_index = {value: index for index, value in enumerate(y_values)}
    grid = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    for x_value, y_value, color_value in points:
        grid[y_index[y_value], x_index[x_value]] = color_value

    finite_values = grid[np.isfinite(grid)]
    if finite_values.size == 0:
        raise ValueError('Heatmap grid has no finite color values')
    cells = _local_heatmap_cells(rows, spec) if 'local_cells' in spec else None
    if cells is not None:
        cell_values = [cell['color_value'] for cell in cells if cell['color_value'] is not None]
        if cell_values:
            finite_values = np.asarray(cell_values)
    center = float(spec.get('center', 0.0))
    lower, upper = float(np.min(finite_values)), float(np.max(finite_values))
    base_cmap = plt.get_cmap(spec.get('colormap') or 'RdBu_r')
    constant_center = lower == upper == center
    if constant_center:
        # A neutral constant field has no signed range; label only its value.
        norm = Normalize(vmin=center, vmax=center + 1.0)
        cmap = ListedColormap([base_cmap(0.5)])
    elif lower >= center:
        norm = Normalize(vmin=center, vmax=upper)
        cmap = ListedColormap(base_cmap(np.linspace(0.5, 1.0, 256)))
    elif upper <= center:
        norm = Normalize(vmin=lower, vmax=center)
        cmap = ListedColormap(base_cmap(np.linspace(0.0, 0.5, 256)))
    else:
        max_abs = max(center - lower, upper - center)
        norm = TwoSlopeNorm(vmin=center - max_abs, vcenter=center, vmax=center + max_abs)
        cmap = base_cmap
    cmap = cmap.with_extremes(bad='#e5e7eb')

    figure_size = spec.get('figure_size') or DEFAULT_HEATMAP_SIZE
    fig, ax = plt.subplots(figsize=(float(figure_size[0]), float(figure_size[1])), facecolor='white')
    def cell_edges(values):
        # Adaptive grids are nonuniform: preserve physical parameter distances.
        values = np.asarray(values, dtype=float)
        mids = (values[1:] + values[:-1]) / 2
        return np.r_[values[0] - (mids[0] - values[0]), mids,
                     values[-1] + (values[-1] - mids[-1])]

    if 'local_cells' in spec:
        from matplotlib.collections import PolyCollection
        vertices, colors = [], []
        for cell in cells:
            left, right = cell['bounds'][spec['x']]
            bottom, top = cell['bounds'][spec['y']]
            vertices.append([(left, bottom), (right, bottom), (right, top), (left, top)])
            colors.append(cell['color_value'] if cell['color_value'] is not None else np.nan)
        mesh = PolyCollection(vertices, array=np.ma.masked_invalid(colors), cmap=cmap,
                              norm=norm, edgecolors='black', linewidths=0.7)
        ax.add_collection(mesh)
        ax.autoscale_view()
        ax.margins(0)
    else:
        mesh = ax.pcolormesh(
            cell_edges(x_values), cell_edges(y_values), grid,
            shading='flat', cmap=cmap, norm=norm, edgecolors='black', linewidths=0.7,
        )
    # Anchor major ticks to the data endpoints, using readable interior steps.
    def endpoint_ticks(values):
        lower, upper = values[0], values[-1]
        offsets = MaxNLocator(nbins=7, steps=[1, 2, 5, 10]).tick_values(0, upper - lower)
        ticks = lower + offsets
        interior = ticks[(ticks > lower) & (ticks < upper)]
        interior = interior[~np.isclose(interior, upper, rtol=0, atol=(upper - lower) * 1e-10)]
        return np.r_[lower, interior, upper]

    ax.set_xticks(endpoint_ticks(x_values))
    ax.set_yticks(endpoint_ticks(y_values))
    ax.set_xlim(x_values[0], x_values[-1])
    ax.set_ylim(y_values[0], y_values[-1])
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.035)
    if constant_center:
        colorbar.set_ticks([center])
    if spec.get('colorbar_label'):
        colorbar.set_label(spec['colorbar_label'])
    _apply_colorbar_style(colorbar)
    _apply_classic_style(ax, xlabel=spec['xlabel'], ylabel=spec['ylabel'], title=spec.get('title') or '')
    _save_styled_figure(fig, spec, output_path)
    plt.close(fig)
    return _artifact_ref(
        output_path,
        kind='postprocess-plot',
        mime_type='image/png',
        description='heatmap: {0} over {1}/{2}'.format(spec['color'], spec['x'], spec['y']),
        root=artifact_root,
    )


def _plot_rows(
    rows: Sequence[Dict[str, Any]],
    spec: Dict[str, Any],
    output_path: Path,
    *,
    artifact_root: Optional[Path] = None,
) -> Dict[str, Any]:
    plt, _AutoMinorLocator = _load_matplotlib()
    if spec['tool'] == 'heatmap':
        return _plot_heatmap(rows, spec, output_path, artifact_root=artifact_root)

    figure_size = spec.get('figure_size') or DEFAULT_FIGURE_SIZE
    fig, ax = plt.subplots(figsize=(float(figure_size[0]), float(figure_size[1])), facecolor='white')
    tool = spec['tool']
    group_column = spec.get('group')
    grouped: Dict[str, List[Tuple[float, float]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        x_value, _x_unit = _parse_numeric_value(row.get(spec['x']))
        y_value, _y_unit = _parse_numeric_value(row.get(spec['y']))
        if x_value is None or y_value is None:
            continue
        group = str(row.get(group_column)) if group_column and row.get(group_column) not in (None, '') else ''
        grouped.setdefault(group, []).append((x_value, y_value))

    if not grouped:
        raise ValueError('No numeric rows are available for {0} vs {1}'.format(spec['y'], spec['x']))

    marker_cycle = ['D', 'o', 's', '^', 'v', 'P', 'X']
    color_cycle = ['black', '#444444', '#777777', '#111111', '#999999']
    for index, (group, values) in enumerate(grouped.items()):
        if spec.get('sort_by_x', True):
            values = sorted(values, key=lambda item: item[0])
        x_values = [item[0] for item in values]
        y_values = [item[1] for item in values]
        marker = marker_cycle[index % len(marker_cycle)]
        color = color_cycle[index % len(color_cycle)]
        label = group or None
        if tool == 'scatter_plot':
            ax.scatter(
                x_values,
                y_values,
                marker=marker,
                s=80,
                c=color,
                edgecolors='black',
                linewidths=1.4,
                label=label,
                zorder=3,
            )
        elif tool == 'bar_plot':
            ax.bar(
                x_values,
                y_values,
                width=0.72,
                color=color,
                edgecolor='black',
                linewidth=2.0,
                label=label,
                zorder=3,
            )
        else:
            ax.plot(
                x_values,
                y_values,
                color=color,
                linewidth=4.0,
                marker=None if any(row.get('grid_origin') for row in rows) else marker,
                markersize=8.5,
                markerfacecolor=color,
                markeredgecolor='black',
                markeredgewidth=1.4,
                label=label,
                zorder=3,
            )
    grid_markers = _plot_grid_markers(ax, rows, spec['x'], spec['y'])
    if grid_markers or (len(grouped) > 1 and group_column):
        legend = ax.legend(frameon=False, fontsize=13)
        for text in legend.get_texts():
            text.set_fontweight('bold')
    _apply_classic_style(ax, xlabel=spec['xlabel'], ylabel=spec['ylabel'], title=spec.get('title') or '')
    _save_styled_figure(fig, spec, output_path)
    plt.close(fig)
    return _artifact_ref(
        output_path,
        kind='postprocess-plot',
        mime_type='image/png',
        description='{0}: {1} vs {2}'.format(tool, spec['y'], spec['x']),
        root=artifact_root,
    )


def run_plot_spec(context: PostprocessContext, spec: Dict[str, Any], *, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    normalized_spec = _normalized_plot_spec(context, spec)
    data_payload = _plot_data_payload(context, normalized_spec)
    return _render_plot(context, normalized_spec, data_payload, output_dir=output_dir)


def _render_plot(context, normalized_spec, data_payload, *, output_dir=None):
    if output_dir is None:
        if not context.work_dir:
            raise ValueError('PostprocessContext.work_dir is required when output_dir is not provided')
        output_dir = Path(context.work_dir) / 'postprocessing'
    output_dir = Path(output_dir)
    artifact_root = Path(context.work_dir) if context.work_dir else None
    artifact_stem = _plot_artifact_stem(normalized_spec['output'])
    plot_path = output_dir / '{0}.png'.format(artifact_stem)
    spec_path = output_dir / '{0}.spec.json'.format(artifact_stem)
    data_tsv_path = output_dir / '{0}.data.tsv'.format(artifact_stem)
    data_json_path = output_dir / '{0}.data.json'.format(artifact_stem)
    numeric_columns = [normalized_spec['x'], normalized_spec['y']]
    if normalized_spec['tool'] == 'heatmap':
        numeric_columns.append(normalized_spec['color'])
    plot_rows = [dict(row, **{key: row[key + '_numeric'] for key in numeric_columns})
                 for row in data_payload['rows']]
    plot_artifact = _plot_rows(
        plot_rows,
        normalized_spec,
        plot_path,
        artifact_root=artifact_root,
    )
    spec_artifact = _write_json(
        spec_path,
        normalized_spec,
        kind='postprocess-plot-spec',
        description='Postprocessing plot specification',
        root=artifact_root,
    )
    data_tsv_artifact = _write_text(
        data_tsv_path,
        _plot_data_tsv(data_payload),
        kind='postprocess-plot-data-tsv',
        description='Postprocessing plot data table',
        root=artifact_root,
        mime_type='text/tab-separated-values; charset=utf-8',
    )
    data_json_artifact = _write_json(
        data_json_path,
        data_payload,
        kind='postprocess-plot-data-json',
        description='Postprocessing plot data payload',
        root=artifact_root,
    )
    return {
        'spec': normalized_spec,
        'artifacts': [plot_artifact, spec_artifact, data_tsv_artifact, data_json_artifact],
        'row_count': data_payload['row_count'],
        'excluded_rows': data_payload['excluded_rows'],
    }


def run_postprocessing(
    report: Any,
    specs: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    output_dir: Optional[str] = None,
    actions: Optional[Sequence[Any]] = None,
    artifact_collector: Optional[Callable[[Mapping[str, str]], Any]] = None,
    dataset_generator: Optional[
        Callable[[Mapping[str, Any]], Mapping[str, Any]]
    ] = None,
) -> Dict[str, Any]:
    context = PostprocessContext.from_report(report)
    if specs is not None and not isinstance(specs, (list, tuple)):
        raise ValueError('plot specs must be a list')
    if actions is not None and not isinstance(actions, (list, tuple)):
        raise ValueError('postprocessing actions must be a list')
    excluded_case_ids = list(context.metadata.get('excluded_case_ids') or [])
    selected_actions = list(actions or [])
    if not context.comparison_table and excluded_case_ids and not selected_actions:
        return {
            'status': 'skipped',
            'message': 'No publication-eligible study rows are available for postprocessing.',
            'artifacts': [],
            'plot_specs': [],
            'actions': [],
            'excluded_case_ids': excluded_case_ids,
            'excluded_rows': context.metadata['excluded_rows'],
        }
    selected_specs = list(
        specs
        if specs is not None and context.comparison_table
        else (
            []
            if selected_actions or not context.comparison_table
            else suggest_plot_specs(report)
        )
    )
    if not selected_specs and not selected_actions:
        return {
            'status': 'skipped',
            'message': 'No postprocessing plot specs or actions were provided or inferred.',
            'artifacts': [],
            'plot_specs': [],
            'actions': [],
            'excluded_case_ids': excluded_case_ids,
            'excluded_rows': context.metadata['excluded_rows'],
        }
    target_dir = Path(output_dir) if output_dir else None
    artifacts: List[Dict[str, Any]] = []
    normalized_specs: List[Dict[str, Any]] = []
    action_results: List[Dict[str, Any]] = []
    prepared_plots = []
    for spec in selected_specs:
        normalized = _normalized_plot_spec(context, spec)
        prepared_plots.append((normalized, _plot_data_payload(context, normalized)))
    outputs = [_plot_artifact_stem(spec['output']) for spec, _data in prepared_plots]
    if len(outputs) != len(set(outputs)):
        raise ValueError('Plot outputs must have distinct names')
    plot_results = []
    for spec, data in prepared_plots:
        result = _render_plot(context, spec, data, output_dir=target_dir)
        artifacts.extend(result['artifacts'])
        normalized_specs.append(result['spec'])
        plot_results.append({key: result[key] for key in ('spec', 'row_count', 'excluded_rows')})
    for action in selected_actions:
        action_id = str(
            action.get('action') or action.get('tool') or ''
            if isinstance(action, dict)
            else action
        ).strip().lower()
        if action_id not in SUPPORTED_POSTPROCESSING_ACTIONS:
            raise ValueError('Unsupported postprocessing action: {0}'.format(action_id or '<empty>'))
        if not context.work_dir:
            raise ValueError('StudyReport work_dir is required for dataset postprocessing.')
        report_payload = report.to_dict() if isinstance(report, StudyReport) else report
        if action_id == ENERGY_SELECTION_ACTION:
            options = {key: value for key, value in action.items() if key not in ('action', 'tool')} if isinstance(action, dict) else {}
            result = select_energy_stratified_frames(
                report_payload, options=options,
                output_dir=(target_dir or Path(context.work_dir) / 'postprocessing') / 'energy-frame-selection',
            )
        elif action_id == HAMILTONIAN_DATASET_GENERATE_ACTION:
            action_output_dir = (
                Path(target_dir) / 'hamiltonian-dataset-generated'
                if target_dir is not None
                else Path(context.work_dir) / 'postprocessing' / 'hamiltonian-dataset-generated'
            )
            result = generate_hamiltonian_dataset(
                report_payload,
                output_dir=action_output_dir,
                dataset_generator=dataset_generator,
            )
        elif action_id == HAMILTONIAN_DATASET_COLLECT_ACTION:
            action_output_dir = (
                Path(target_dir) / 'hamiltonian-dataset'
                if target_dir is not None
                else Path(context.work_dir) / 'postprocessing' / 'hamiltonian-dataset'
            )
            result = collect_hamiltonian_dataset(
                report_payload,
                output_dir=action_output_dir,
                artifact_collector=artifact_collector,
            )
        else:  # pragma: no cover - future registered adapters
            raise ValueError('Postprocessing action has no runtime adapter: {0}'.format(action_id))
        artifacts.extend(result.get('artifacts') or [])
        action_results.append(result)
    status = 'succeeded'
    if any(result.get('status') in ('partial', 'skipped') for result in action_results):
        status = 'partial' if plot_results or any(result.get('status') != 'skipped' for result in action_results) else 'skipped'
    return {
        'status': status,
        'artifacts': artifacts,
        'plot_specs': normalized_specs,
        'actions': action_results,
        'excluded_case_ids': excluded_case_ids,
        'excluded_rows': context.metadata['excluded_rows'],
        'plots': plot_results,
    }


__all__ = [
    'PostprocessContext',
    'SUPPORTED_PLOT_TOOLS',
    'SUPPORTED_POSTPROCESSING_ACTIONS',
    'run_plot_spec',
    'run_postprocessing',
    'suggest_plot_specs',
]
