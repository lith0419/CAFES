function formatNumber(value) {
  if (!Number.isFinite(value)) {
    return 'nan';
  }
  if (Math.abs(value) < 1e-12) {
    return '0';
  }
  return Number(value.toFixed(4)).toString();
}

function energyUnitText() {
  return state.energyUnit || 'a.u.';
}

function formatEnergyValue(value) {
  return `${formatNumber(value)} ${energyUnitText()}`;
}

function distanceFunctionText(baseSymbol, functionName, parameterSymbol = 'a') {
  if (functionName === 'constant') {
    return `${baseSymbol}(r) = ${baseSymbol}_0`;
  }
  if (functionName === 'power') {
    return `${baseSymbol}(r) = ${baseSymbol}_0 * (r0/r)^${parameterSymbol}`;
  }
  if (functionName === 'linear') {
    return `${baseSymbol}(r) = ${baseSymbol}_0 * [1 - ${parameterSymbol}(r-r0)]`;
  }
  if (functionName === 'inverse') {
    return `${baseSymbol}(r) = ${baseSymbol}_0 * r0/r`;
  }
  return `${baseSymbol}(r) = ${baseSymbol}_0 * exp[-${parameterSymbol}(r-r0)]`;
}

function h1Matrix() {
  const orderedSites = state.sites.slice().sort((a, b) => a.id - b.id);
  const siteIndexById = new Map(orderedSites.map((site, index) => [site.id, index]));
  const n = orderedSites.length;
  const matrix = Array.from({ length: n }, () => Array.from({ length: n }, () => 0));
  orderedSites.forEach((site, index) => {
    matrix[index][index] = site.epsilon;
  });
  state.bonds.forEach((bond) => {
    const i = siteIndexById.get(bond.source);
    const j = siteIndexById.get(bond.target);
    if (i === undefined || j === undefined) {
      return;
    }
    const t = effectiveBondT(bond);
    if (isBlochRepresentation()) {
      matrix[i][j] += t;
      if (bond.addHermitianConjugate !== false) {
        matrix[j][i] += t;
      }
    } else {
      matrix[i][j] = t;
      matrix[j][i] = t;
    }
  });
  return matrix;
}

function interactionTerms() {
  return {
    onsite_u: state.sites.map((site) => ({ site: site.id, U: site.U })),
    nearest_neighbor_v: state.bonds
      .filter((bond) => Math.abs(effectiveBondV(bond)) > 1e-12)
      .map((bond) => ({ source: bond.source, target: bond.target, V: effectiveBondV(bond) })),
  };
}

function phononTerms() {
  if (!isHolsteinHubbardModel()) {
    return {
      enabled: false,
      cutoff: 0,
      local_modes: [],
    };
  }
  return {
    enabled: true,
    cutoff: state.globals.phononCutoff,
    local_modes: state.sites.map((site) => ({
      site: site.id,
      omega: site.phononOmega,
      g: site.electronPhononG,
    })),
  };
}

function serializableGlobals() {
  const globals = {
    t: state.globals.t,
    delta: state.globals.delta,
    epsilon: state.globals.epsilon,
    U: state.globals.U,
    V: state.globals.V,
  };
  if (isHolsteinHubbardModel()) {
    globals.phononOmega = state.globals.phononOmega;
    globals.electronPhononG = state.globals.electronPhononG;
    globals.phononCutoff = state.globals.phononCutoff;
  }
  return globals;
}

function serializableSite(site) {
  const item = {
    id: site.id,
    x: site.x,
    y: site.y,
    epsilon: site.epsilon,
    U: site.U,
  };
  if (site.sublattice) {
    item.sublattice = site.sublattice;
  }
  if (Array.isArray(site.cellIndex)) {
    item.cell_index = site.cellIndex.slice();
  }
  if (Number.isInteger(site.basisIndex)) {
    item.basis_index = site.basisIndex;
  }
  if (isHolsteinHubbardModel()) {
    item.phononOmega = site.phononOmega;
    item.electronPhononG = site.electronPhononG;
  }
  return item;
}

function serializableGraph() {
  const nodes = state.sites.map((site) => ({
    id: site.id,
    x: site.x,
    y: site.y,
    sublattice: site.sublattice || null,
    cell_index: Array.isArray(site.cellIndex) ? site.cellIndex.slice() : null,
    basis_index: Number.isInteger(site.basisIndex) ? site.basisIndex : null,
  }));
  const edges = state.bonds.map((bond) => ({
    id: bond.id,
    source: bond.source,
    target: bond.target,
    endpoints: [bond.source, bond.target],
    periodic: !!bond.periodic,
    offset: bond.offset || { x: 0, y: 0 },
    cell_offset: isBlochRepresentation() ? (bond.cellOffset || []).slice() : null,
    add_hermitian_conjugate: isBlochRepresentation() ? bond.addHermitianConjugate !== false : null,
    kind: bond.kind || 'custom',
  }));
  return {
    representation: 'site-bond graph',
    node_id_field: 'id',
    edge_id_field: 'id',
    edge_endpoint_fields: ['source', 'target'],
    nodes,
    edges,
  };
}

function serializableSpec() {
  const spec = {
    schema: 'pyscf-agent.model-hamiltonian.v1',
    model: state.model,
    representation: state.representation,
    solver: isBlochRepresentation() ? 'tight_binding' : 'fci',
    dimension: state.dimension,
    preset: state.preset,
    bond_modulation: state.bondModulation,
    boundary: state.boundary,
    energy_unit: state.energyUnit,
    geometry_coupled_hopping: state.geometryCoupled,
    geometry_coupled_interaction: state.geometryCoupledV,
    geometry: { ...state.geometry },
    globals: serializableGlobals(),
    phonons: phononTerms(),
    cell: state.cell ? JSON.parse(JSON.stringify(state.cell)) : null,
    primitive_cell: state.primitiveCell ? JSON.parse(JSON.stringify(state.primitiveCell)) : null,
    sites: state.sites.map((site) => serializableSite(site)),
    bonds: state.bonds.map((bond) => ({
      id: bond.id,
      source: bond.source,
      target: bond.target,
      t: bond.t,
      V: bond.V,
      kind: bond.kind || 'custom',
      periodic: !!bond.periodic,
      offset: bond.offset || { x: 0, y: 0 },
      effective_t: effectiveBondT(bond),
      effective_V: effectiveBondV(bond),
      ...(isBlochRepresentation() ? {
        cell_offset: (bond.cellOffset || Array.from({ length: state.dimension }, () => 0)).slice(),
        add_hermitian_conjugate: bond.addHermitianConjugate !== false,
      } : {}),
    })),
    graph: serializableGraph(),
  };
  if (isBlochRepresentation()) {
    spec.lattice_vectors = state.latticeVectors.map((vector) => [vector.x, vector.y]);
    spec.occupation = {
      mode: 'filling',
      electrons_per_cell: state.occupation.electronsPerCell,
      spin_degeneracy: 2,
    };
    spec.reciprocal_space = {
      kmesh: state.reciprocal.kmesh.slice(0, state.dimension),
      scheme: state.reciprocal.scheme,
      shift: state.reciprocal.shift.slice(0, state.dimension),
      path: {
        mode: 'automatic',
        points_per_segment: state.reciprocal.pointsPerSegment,
      },
      dos: {
        points: Math.min(state.reciprocal.dosPoints, blochDosPointLimit()),
        sigma: state.reciprocal.dosSigma > 0 ? state.reciprocal.dosSigma : null,
      },
    };
  } else {
    spec.nelec = state.nelec.slice();
    spec.spin_multiplicity = Math.abs(state.nelec[0] - state.nelec[1]) + 1;
  }
  return spec;
}

function specJson() {
  return JSON.stringify(serializableSpec(), null, 2);
}

function matrixPreview(matrix) {
  return matrix.map((row) => row.map((value) => formatNumber(value).padStart(8, ' ')).join(' ')).join('\n');
}

function summaryText() {
  const spec = serializableSpec();
  const interactions = interactionTerms();
  const phonons = spec.phonons;
  const unit = spec.energy_unit || energyUnitText();
  const lines = [
    `model: ${spec.model}`,
    `representation: ${spec.representation}`,
    `solver: ${spec.solver}`,
    `dimension: ${spec.dimension}D`,
    `preset: ${spec.preset}`,
    `bond modulation: ${spec.bond_modulation}`,
    `boundary: ${spec.boundary}`,
    `energy unit: ${spec.energy_unit}`,
    `norb: ${spec.sites.length}`,
    `sites: ${spec.sites.length}`,
    `bonds: ${spec.bonds.length}`,
    `global epsilon: ${formatNumber(spec.globals.epsilon)} ${unit}`,
    `global t: ${formatNumber(spec.globals.t)} ${unit}`,
    `global U: ${formatNumber(spec.globals.U)} ${unit}`,
    `global V: ${formatNumber(spec.globals.V)} ${unit}`,
    `onsite U terms: ${interactions.onsite_u.length}`,
    `nonzero V terms: ${interactions.nearest_neighbor_v.length}`,
    `local phonon modes: ${phonons.enabled ? phonons.local_modes.length : 0}`,
    '',
    'Hamiltonian:',
    'H = sum_i epsilon_i n_i',
    '  + sum_<ij>,sigma t_ij c^dag_i,sigma c_j,sigma + h.c.',
    '  + sum_i U_i n_i,alpha n_i,beta',
    '  + sum_<ij> V_ij n_i n_j',
  ];
  if (isBlochRepresentation()) {
    lines.splice(7, 0,
      `electrons per cell: ${formatNumber(spec.occupation.electrons_per_cell)}`,
      `filling per orbital: ${formatNumber(spec.occupation.electrons_per_cell / spec.sites.length)}`,
      `k-mesh: ${spec.reciprocal_space.kmesh.join(' x ')}`,
      `k-point scheme: ${spec.reciprocal_space.scheme}`,
      `path sampling: automatic, ${spec.reciprocal_space.path.points_per_segment} points/segment`,
    );
    lines.push('', 'Bloch solver boundary:', 'h(k) includes epsilon and hopping terms; U and V are recorded but not applied by tight_binding.');
  } else {
    lines.splice(7, 0, `nelec: (${spec.nelec[0]}, ${spec.nelec[1]})`);
  }
  if (phonons.enabled) {
    lines.push(`global omega: ${formatNumber(spec.globals.phononOmega)} ${unit}`);
    lines.push(`global g: ${formatNumber(spec.globals.electronPhononG)} ${unit}`);
    lines.push('  + sum_i omega_i b^dag_i b_i');
    lines.push('  + sum_i g_i n_i (b^dag_i + b_i)');
    lines.push(`phonon cutoff: ${phonons.cutoff}`);
  }
  if (spec.geometry_coupled_hopping) {
    lines.push('', `geometry-coupled hopping: ${distanceFunctionText('t', spec.geometry.hoppingFunction, 'a_t')}, a_t=${spec.geometry.hoppingParameter}, r0=${spec.geometry.r0}`);
  }
  if (spec.geometry_coupled_interaction) {
    lines.push(`geometry-coupled interaction: ${distanceFunctionText('V', spec.geometry.interactionFunction, 'b')}, b=${spec.geometry.interactionParameter}, r0=${spec.geometry.r0}`);
  }
  return lines.join('\n');
}

function pythonInputText() {
  const spec = serializableSpec();
  return `from __future__ import annotations

# Generated by model_hamiltonian_ui.
# This file intentionally contains only the model Hamiltonian structure.
# Select MP2/CCSD/CCSD(T)/FCI later in the PySCF Agent Web UI for finite clusters.
# Bloch lattice inputs select the one-body tight_binding solver automatically.

model_spec = ${JSON.stringify(spec, null, 2)}

if __name__ == "__main__":
    sites = sorted(model_spec["sites"], key=lambda item: item["id"])
    bonds = model_spec.get("bonds", [])
    print("model =", model_spec.get("model"))
    print("representation =", model_spec.get("representation"))
    print("solver =", model_spec.get("solver"))
    print("dimension =", model_spec.get("dimension"))
    print("boundary =", model_spec.get("boundary"))
    print("energy_unit =", model_spec.get("energy_unit"))
    print("nelec =", tuple(model_spec.get("nelec", [])))
    print("electrons_per_cell =", model_spec.get("occupation", {}).get("electrons_per_cell"))
    print("kmesh =", model_spec.get("reciprocal_space", {}).get("kmesh"))
    print("sites =", len(sites))
    print("bonds =", len(bonds))
    print("graph_edges =", len(model_spec.get("graph", {}).get("edges", [])))
`;
}

function downloadText(filename, text, mimeType) {
  const blob = new Blob([text], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
