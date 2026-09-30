from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List

from ...registry import default_registry
from ...registry.platform import SUPPORTED_MODEL_HAMILTONIANS, SUPPORTED_MODEL_SOLVERS


MODEL_HAMILTONIAN_SCHEMA = 'pyscf-agent.model-hamiltonian.v1'
MODEL_HAMILTONIAN_REPRESENTATIONS = ('finite_cluster', 'bloch')
MAX_BLOCH_KPOINTS = 50_000
MAX_BLOCH_PATH_VERTICES = 64
MAX_BLOCH_PATH_POINTS = 20_000
MAX_BLOCH_DOS_POINTS = 4_001
MAX_BLOCH_DOS_WORK_ITEMS = 200_000_000


def model_spec_graph_view(spec: Dict[str, Any]) -> Dict[str, Any]:
    sites = spec.get('sites') if isinstance(spec.get('sites'), list) else []
    bonds = spec.get('bonds') if isinstance(spec.get('bonds'), list) else []
    nodes = []
    for site in sites:
        if not isinstance(site, dict) or 'id' not in site:
            continue
        nodes.append({
            'id': site.get('id'),
            'x': site.get('x'),
            'y': site.get('y'),
            'sublattice': site.get('sublattice'),
            'cell_index': copy.deepcopy(site.get('cell_index')),
            'basis_index': site.get('basis_index'),
        })
    edges = []
    for bond in bonds:
        if not isinstance(bond, dict) or 'id' not in bond:
            continue
        source = bond.get('source')
        target = bond.get('target')
        edges.append({
            'id': bond.get('id'),
            'source': source,
            'target': target,
            'endpoints': [source, target],
            'periodic': bool(bond.get('periodic')),
            'offset': bond.get('offset') if isinstance(bond.get('offset'), dict) else {'x': 0, 'y': 0},
            'cell_offset': bond.get('cell_offset') if isinstance(bond.get('cell_offset'), list) else None,
            'add_hermitian_conjugate': bool(bond.get('add_hermitian_conjugate', True)),
            'kind': bond.get('kind') or 'custom',
        })
    return {
        'representation': 'site-bond graph',
        'node_id_field': 'id',
        'edge_id_field': 'id',
        'edge_endpoint_fields': ['source', 'target'],
        'nodes': nodes,
        'edges': edges,
    }


def ensure_model_spec_graph(spec: Dict[str, Any]) -> Dict[str, Any]:
    graph = spec.get('graph')
    if not isinstance(graph, dict) or not isinstance(graph.get('edges'), list) or not isinstance(graph.get('nodes'), list):
        spec['graph'] = model_spec_graph_view(spec)
    return spec


def _extract_balanced_object(text: str, start: int) -> str:
    depth = 0
    in_string = ''
    escape = False
    object_start = -1
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == '\\':
                escape = True
            elif char == in_string:
                in_string = ''
            continue
        if char in ('"', "'"):
            in_string = char
            continue
        if char == '{':
            if depth == 0:
                object_start = index
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and object_start >= 0:
                return text[object_start:index + 1]
    raise ValueError('Unable to find a complete model_spec object')


def parse_model_spec_from_python_input(text: str) -> Dict[str, Any]:
    marker = 'model_spec'
    try:
        module = ast.parse(text)
    except SyntaxError:
        module = None
    if module is not None:
        for node in module.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == marker for target in node.targets):
                continue
            value = node.value
            if isinstance(value, ast.Call) and value.args:
                function_name = ''
                if isinstance(value.func, ast.Attribute):
                    function_name = value.func.attr
                elif isinstance(value.func, ast.Name):
                    function_name = value.func.id
                first_arg = value.args[0]
                if function_name == 'loads' and isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                    spec = json.loads(first_arg.value)
                    if not isinstance(spec, dict):
                        raise ValueError('model_spec must be a JSON/Python object')
                    return spec
            try:
                spec = ast.literal_eval(value)
            except (TypeError, ValueError):
                break
            if not isinstance(spec, dict):
                raise ValueError('model_spec must be a JSON/Python object')
            return spec

    marker_index = text.find(marker)
    if marker_index < 0:
        raise ValueError('The input file does not contain a model_spec assignment')
    equals_index = text.find('=', marker_index + len(marker))
    if equals_index < 0:
        raise ValueError('The model_spec assignment is incomplete')
    object_text = _extract_balanced_object(text, equals_index + 1)
    try:
        spec = json.loads(object_text)
    except json.JSONDecodeError:
        spec = ast.literal_eval(object_text)
    if not isinstance(spec, dict):
        raise ValueError('model_spec must be a JSON/Python object')
    return spec


def load_model_spec_from_file(path_value: str) -> Dict[str, Any]:
    path = Path(path_value).expanduser()
    if not path.exists():
        raise FileNotFoundError('Model Hamiltonian input file does not exist: {0}'.format(path))
    if not path.is_file():
        raise ValueError('Model Hamiltonian input path is not a file: {0}'.format(path))
    text = path.read_text(encoding='utf-8')
    if path.suffix.lower() == '.json':
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError('Model Hamiltonian JSON file must contain an object')
        return payload
    return parse_model_spec_from_python_input(text)


def normalize_model_spec(raw_spec: Dict[str, Any]) -> Dict[str, Any]:
    spec = copy.deepcopy(raw_spec)
    spec.setdefault('schema', MODEL_HAMILTONIAN_SCHEMA)
    spec.setdefault('model', 'hubbard')
    representation = normalize_model_representation(spec.get('representation'))
    spec['representation'] = representation
    spec.setdefault('solver', 'tight_binding' if representation == 'bloch' else 'fci')
    spec.setdefault('energy_unit', 'a.u.')
    if 'nelec' in spec and isinstance(spec['nelec'], tuple):
        spec['nelec'] = list(spec['nelec'])
    for site in spec.get('sites') or []:
        if isinstance(site, dict) and isinstance(site.get('cell_index'), tuple):
            site['cell_index'] = list(site['cell_index'])
    if representation == 'finite_cluster':
        nelec = spec.get('nelec')
        if isinstance(nelec, (list, tuple)) and len(nelec) == 2:
            try:
                derived_multiplicity = abs(int(nelec[0]) - int(nelec[1])) + 1
            except (TypeError, ValueError):
                derived_multiplicity = None
            if derived_multiplicity is not None:
                spec.setdefault('spin_multiplicity', derived_multiplicity)
    if representation == 'bloch':
        dimension = _dimension_value(spec.get('dimension'))
        spec['boundary'] = 'periodic'
        occupation = spec.get('occupation') if isinstance(spec.get('occupation'), dict) else {}
        occupation.setdefault('mode', 'filling')
        occupation.setdefault('electrons_per_cell', float(len(spec.get('sites') or [])))
        occupation.setdefault('spin_degeneracy', 2)
        spec['occupation'] = occupation

        reciprocal_space = (
            spec.get('reciprocal_space')
            if isinstance(spec.get('reciprocal_space'), dict)
            else {}
        )
        reciprocal_space.setdefault('kmesh', [100] if dimension == 1 else [30, 30])
        reciprocal_space.setdefault('scheme', 'gamma_centered')
        reciprocal_space.setdefault('shift', [0.0] * dimension)
        path = reciprocal_space.get('path') if isinstance(reciprocal_space.get('path'), dict) else {}
        path.setdefault('mode', 'automatic')
        path.setdefault('points_per_segment', 80)
        reciprocal_space['path'] = path
        dos = reciprocal_space.get('dos') if isinstance(reciprocal_space.get('dos'), dict) else {}
        dos.setdefault('points', 401)
        dos.setdefault('sigma', None)
        reciprocal_space['dos'] = dos
        spec['reciprocal_space'] = reciprocal_space
    return ensure_model_spec_graph(spec)


def normalize_model_representation(value: Any) -> str:
    name = str(value or 'finite_cluster').strip().lower().replace('-', '_').replace(' ', '_')
    aliases = {
        'finite': 'finite_cluster',
        'cluster': 'finite_cluster',
        'real_space': 'finite_cluster',
        'periodic_bloch': 'bloch',
        'bloch_lattice': 'bloch',
        'k_space': 'bloch',
        'reciprocal_space': 'bloch',
    }
    return aliases.get(name, name)


def _dimension_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _as_float(value: Any, field: str, errors: List[str], default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        errors.append('{0} must be numeric'.format(field))
        return default
    if not math.isfinite(numeric):
        errors.append('{0} must be finite'.format(field))
        return default
    return numeric


def _as_int(value: Any, field: str, errors: List[str], default: int = 0) -> int:
    if isinstance(value, bool):
        errors.append('{0} must be an integer'.format(field))
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        errors.append('{0} must be an integer'.format(field))
        return default


def _is_integer_value(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric.is_integer()


def _validate_bloch_spec(spec: Dict[str, Any], sites: List[Any], errors: List[str]) -> None:
    dimension = _dimension_value(spec.get('dimension'))
    if dimension not in (1, 2):
        errors.append('Bloch Model Hamiltonians currently support dimension 1 or 2')
        return

    lattice_vectors = spec.get('lattice_vectors')
    if not isinstance(lattice_vectors, list) or len(lattice_vectors) != dimension:
        errors.append('lattice_vectors must contain one vector per periodic dimension')
        lattice_vectors = []
    parsed_vectors = []
    for index, vector in enumerate(lattice_vectors):
        if not isinstance(vector, (list, tuple)) or len(vector) < dimension or len(vector) > 3:
            errors.append('lattice_vectors[{0}] must contain {1}-3 numeric components'.format(index, dimension))
            continue
        parsed = [
            _as_float(value, 'lattice_vectors[{0}][{1}]'.format(index, component), errors)
            for component, value in enumerate(vector)
        ]
        parsed_vectors.append(parsed)
    if len(parsed_vectors) == dimension:
        if dimension == 1:
            if math.sqrt(sum(value * value for value in parsed_vectors[0])) <= 1e-12:
                errors.append('lattice_vectors must define a nonzero primitive vector')
        else:
            a = parsed_vectors[0]
            b = parsed_vectors[1]
            if len(a) == 2 and len(b) == 2:
                area = a[0] * b[1] - a[1] * b[0]
            else:
                a3 = a + [0.0] * (3 - len(a))
                b3 = b + [0.0] * (3 - len(b))
                cross = [
                    a3[1] * b3[2] - a3[2] * b3[1],
                    a3[2] * b3[0] - a3[0] * b3[2],
                    a3[0] * b3[1] - a3[1] * b3[0],
                ]
                area = math.sqrt(sum(value * value for value in cross))
            if abs(area) <= 1e-12:
                errors.append('lattice_vectors must be linearly independent')

    for index, site in enumerate(sites):
        if not isinstance(site, dict):
            continue
        _as_float(site.get('x', 0.0), 'site[{0}].x'.format(index), errors)
        _as_float(site.get('y', 0.0), 'site[{0}].y'.format(index), errors)

    occupation = spec.get('occupation')
    if not isinstance(occupation, dict):
        errors.append('Bloch Model Hamiltonian spec requires an occupation object')
    else:
        mode = str(occupation.get('mode') or '').strip().lower()
        if mode != 'filling':
            errors.append('occupation.mode must be filling')
        electrons = _as_float(occupation.get('electrons_per_cell'), 'occupation.electrons_per_cell', errors, default=-1.0)
        capacity = 2.0 * len(sites)
        if electrons < 0 or electrons > capacity + 1e-12:
            errors.append('occupation.electrons_per_cell must be between 0 and {0:g}'.format(capacity))
        spin_degeneracy = occupation.get('spin_degeneracy', 2)
        if not _is_integer_value(spin_degeneracy) or int(float(spin_degeneracy)) != 2:
            errors.append('occupation.spin_degeneracy must be 2 for the current tight-binding solver')

    reciprocal = spec.get('reciprocal_space')
    if not isinstance(reciprocal, dict):
        errors.append('Bloch Model Hamiltonian spec requires reciprocal_space settings')
        return
    kmesh = reciprocal.get('kmesh')
    kpoint_count = None
    if not isinstance(kmesh, (list, tuple)) or len(kmesh) != dimension:
        errors.append('reciprocal_space.kmesh must contain one positive integer per periodic dimension')
    else:
        valid_kmesh = True
        for index, value in enumerate(kmesh):
            if not _is_integer_value(value) or int(float(value)) <= 0:
                errors.append('reciprocal_space.kmesh[{0}] must be a positive integer'.format(index))
                valid_kmesh = False
        if valid_kmesh:
            kpoint_count = math.prod(int(float(value)) for value in kmesh)
            if kpoint_count > MAX_BLOCH_KPOINTS:
                errors.append(
                    'reciprocal_space.kmesh contains {0} k-points; the current limit is {1}'.format(
                        kpoint_count,
                        MAX_BLOCH_KPOINTS,
                    )
                )
    scheme = str(reciprocal.get('scheme') or '').strip().lower().replace('-', '_')
    if scheme not in ('gamma_centered', 'monkhorst_pack'):
        errors.append('reciprocal_space.scheme must be gamma_centered or monkhorst_pack')
    shift = reciprocal.get('shift')
    if not isinstance(shift, (list, tuple)) or len(shift) != dimension:
        errors.append('reciprocal_space.shift must contain one numeric value per periodic dimension')
    else:
        for index, value in enumerate(shift):
            _as_float(value, 'reciprocal_space.shift[{0}]'.format(index), errors)

    path = reciprocal.get('path')
    path_vertex_count = 4 if dimension == 2 else 3
    points_per_segment_value = None
    if not isinstance(path, dict):
        errors.append('reciprocal_space.path must be an object')
    else:
        mode = str(path.get('mode') or '').strip().lower()
        if mode not in ('automatic', 'custom'):
            errors.append('reciprocal_space.path.mode must be automatic or custom')
        points_per_segment = path.get('points_per_segment', 80)
        if not _is_integer_value(points_per_segment) or int(float(points_per_segment)) < 2:
            errors.append('reciprocal_space.path.points_per_segment must be an integer of at least 2')
        else:
            points_per_segment_value = int(float(points_per_segment))
        if mode == 'custom':
            points = path.get('points')
            if not isinstance(points, list) or len(points) < 2:
                errors.append('reciprocal_space.path.points must contain at least two labeled k-points')
            else:
                path_vertex_count = len(points)
                if path_vertex_count > MAX_BLOCH_PATH_VERTICES:
                    errors.append(
                        'reciprocal_space.path.points contains {0} vertices; the current limit is {1}'.format(
                            path_vertex_count,
                            MAX_BLOCH_PATH_VERTICES,
                        )
                    )
                for index, point in enumerate(points):
                    coordinates = point.get('k') if isinstance(point, dict) else None
                    if not isinstance(coordinates, (list, tuple)) or len(coordinates) != dimension:
                        errors.append('reciprocal_space.path.points[{0}].k must match the periodic dimension'.format(index))
                    else:
                        for component, value in enumerate(coordinates):
                            _as_float(value, 'reciprocal_space.path.points[{0}].k[{1}]'.format(index, component), errors)
        if points_per_segment_value is not None and path_vertex_count >= 2:
            path_point_count = 1 + (path_vertex_count - 1) * (points_per_segment_value - 1)
            if path_point_count > MAX_BLOCH_PATH_POINTS:
                errors.append(
                    'reciprocal_space.path generates {0} k-points; the current limit is {1}'.format(
                        path_point_count,
                        MAX_BLOCH_PATH_POINTS,
                    )
                )

    dos = reciprocal.get('dos')
    dos_point_count = None
    if not isinstance(dos, dict):
        errors.append('reciprocal_space.dos must be an object')
    else:
        points = dos.get('points', 401)
        if not _is_integer_value(points) or int(float(points)) < 2:
            errors.append('reciprocal_space.dos.points must be an integer of at least 2')
        else:
            dos_point_count = int(float(points))
            if dos_point_count > MAX_BLOCH_DOS_POINTS:
                errors.append(
                    'reciprocal_space.dos.points={0} exceeds the current limit of {1}'.format(
                        dos_point_count,
                        MAX_BLOCH_DOS_POINTS,
                    )
                )
        sigma = dos.get('sigma')
        if sigma is not None and _as_float(sigma, 'reciprocal_space.dos.sigma', errors, default=-1.0) <= 0:
            errors.append('reciprocal_space.dos.sigma must be positive or null for automatic broadening')
    if kpoint_count is not None and dos_point_count is not None and sites:
        dos_work_items = kpoint_count * len(sites) * dos_point_count
        if dos_work_items > MAX_BLOCH_DOS_WORK_ITEMS:
            errors.append(
                'Bloch DOS request requires {0} k-point/orbital/grid evaluations; the current limit is {1}. '
                'Reduce kmesh or reciprocal_space.dos.points.'.format(
                    dos_work_items,
                    MAX_BLOCH_DOS_WORK_ITEMS,
                )
            )


def validate_model_hamiltonian_spec(spec: Dict[str, Any], solver_name: Any = None) -> List[str]:
    errors: List[str] = []
    if not isinstance(spec, dict):
        return ['Model Hamiltonian spec must be an object']
    representation = normalize_model_representation(spec.get('representation'))
    solver_name = normalize_solver_name(
        solver_name or spec.get('solver') or ('tight_binding' if representation == 'bloch' else 'fci')
    )

    if spec.get('schema') not in (None, MODEL_HAMILTONIAN_SCHEMA):
        errors.append('Unsupported Model Hamiltonian schema: {0}'.format(spec.get('schema')))

    model_name = str(spec.get('model', 'hubbard')).strip().lower()
    registry = default_registry()
    model_capability = registry.capability(
        model_name, namespace='model_hamiltonian.model'
    )
    if model_capability is not None and not model_capability.backend_allowed:
        detail = '; '.join(model_capability.limitations) or 'not executable by the backend'
        errors.append('{0} model Hamiltonians are {1}: {2}'.format(model_capability.label, model_capability.status, detail))
    elif model_name not in SUPPORTED_MODEL_HAMILTONIANS:
        errors.append('Unsupported Model Hamiltonian model: {0}'.format(model_name))

    solver_capability = registry.capability(
        solver_name, namespace='model_hamiltonian.solver'
    )
    if solver_capability is not None and not solver_capability.backend_allowed:
        detail = '; '.join(solver_capability.limitations) or 'not executable by the backend'
        errors.append('{0} model Hamiltonian solver is {1}: {2}'.format(solver_capability.label, solver_capability.status, detail))
    elif solver_name not in SUPPORTED_MODEL_SOLVERS:
        errors.append('Unsupported Model Hamiltonian solver: {0}'.format(solver_name))

    if representation not in MODEL_HAMILTONIAN_REPRESENTATIONS:
        errors.append('Unsupported Model Hamiltonian representation: {0}'.format(representation))
    elif representation == 'bloch' and solver_name != 'tight_binding':
        errors.append('Bloch Model Hamiltonians currently require the tight_binding solver')
    elif representation == 'finite_cluster' and solver_name == 'tight_binding':
        errors.append('The tight_binding solver requires representation=bloch')

    sites = spec.get('sites')
    if not isinstance(sites, list) or not sites:
        errors.append('Model Hamiltonian spec must contain at least one site')
        sites = []
    site_ids = []
    for index, site in enumerate(sites):
        if not isinstance(site, dict):
            errors.append('site[{0}] must be an object'.format(index))
            continue
        site_id = _as_int(site.get('id'), 'site[{0}].id'.format(index), errors, default=index)
        site_ids.append(site_id)
        _as_float(site.get('epsilon', 0.0), 'site[{0}].epsilon'.format(index), errors)
        _as_float(site.get('U', 0.0), 'site[{0}].U'.format(index), errors)
    if len(site_ids) != len(set(site_ids)):
        errors.append('Site ids must be unique')
    site_id_set = set(site_ids)

    if representation == 'finite_cluster':
        nelec = spec.get('nelec')
        if not isinstance(nelec, (list, tuple)) or len(nelec) != 2:
            errors.append('nelec must be a two-item list: [nalpha, nbeta]')
        else:
            nalpha = _as_int(nelec[0], 'nelec[0]', errors)
            nbeta = _as_int(nelec[1], 'nelec[1]', errors)
            if nalpha < 0 or nbeta < 0:
                errors.append('nelec values must be non-negative')
            if sites and nalpha + nbeta > 2 * len(sites):
                errors.append('nelec exceeds the spin-orbital capacity of the model')
            multiplicity = _as_int(
                spec.get('spin_multiplicity', abs(nalpha - nbeta) + 1),
                'spin_multiplicity',
                errors,
            )
            if multiplicity < 1:
                errors.append('spin_multiplicity must be a positive integer')
            elif multiplicity != abs(nalpha - nbeta) + 1:
                errors.append(
                    'spin_multiplicity must equal |nalpha-nbeta|+1 for the selected highest-weight sector'
                )
    elif representation == 'bloch':
        _validate_bloch_spec(spec, sites, errors)

    bonds = spec.get('bonds', [])
    if not isinstance(bonds, list):
        errors.append('bonds must be a list')
        bonds = []
    for index, bond in enumerate(bonds):
        if not isinstance(bond, dict):
            errors.append('bond[{0}] must be an object'.format(index))
            continue
        source = _as_int(bond.get('source'), 'bond[{0}].source'.format(index), errors)
        target = _as_int(bond.get('target'), 'bond[{0}].target'.format(index), errors)
        if source not in site_id_set:
            errors.append('bond[{0}].source references an unknown site'.format(index))
        if target not in site_id_set:
            errors.append('bond[{0}].target references an unknown site'.format(index))
        _as_float(bond.get('effective_t', bond.get('t', 0.0)), 'bond[{0}].t'.format(index), errors)
        _as_float(bond.get('effective_V', bond.get('V', 0.0)), 'bond[{0}].V'.format(index), errors)
        if representation == 'bloch':
            cell_offset = bond.get('cell_offset')
            dimension = _dimension_value(spec.get('dimension'))
            if not isinstance(cell_offset, (list, tuple)) or len(cell_offset) != dimension:
                errors.append('bond[{0}].cell_offset must contain one integer per periodic dimension'.format(index))
            else:
                for component, value in enumerate(cell_offset):
                    if not _is_integer_value(value):
                        errors.append('bond[{0}].cell_offset[{1}] must be an integer'.format(index, component))
                if source == target and all(int(float(value)) == 0 for value in cell_offset if _is_integer_value(value)) and all(_is_integer_value(value) for value in cell_offset):
                    errors.append('bond[{0}] cannot be an on-site zero-cell-offset bond; use site epsilon instead'.format(index))
            if 'add_hermitian_conjugate' in bond and not isinstance(bond.get('add_hermitian_conjugate'), bool):
                errors.append('bond[{0}].add_hermitian_conjugate must be boolean'.format(index))

    return errors


def normalize_solver_name(value: Any) -> str:
    raw = str(value or 'fci').strip()
    canonical = default_registry().canonical_capability_id(
        raw, namespace='model_hamiltonian.solver'
    )
    if canonical is not None:
        return canonical
    return raw.lower().replace('-', '_').replace(' ', '_')
