function bondVector(bond) {
  const source = siteById(bond.source);
  const target = siteById(bond.target);
  if (!source || !target) {
    return { dx: 0, dy: 0, length: 0 };
  }
  const offset = bond.offset || { x: 0, y: 0 };
  const dx = target.x + offset.x - source.x;
  const dy = target.y + offset.y - source.y;
  return { dx, dy, length: Math.hypot(dx, dy) };
}

function bondDistance(bond) {
  return bondVector(bond).length;
}

function baseHoppingForKind(kind) {
  const t = state.globals.t;
  const delta = state.globals.delta;
  if (!isSshBondModulationEnabled()) {
    return t;
  }
  if (kind === 'strong') {
    return t * (1 + delta);
  }
  if (kind === 'weak') {
    return t * (1 - delta);
  }
  return t;
}

function isSshBondModulationEnabled() {
  return state.dimension === 1 && state.bondModulation === 'ssh';
}

function isHolsteinHubbardModel() {
  return state.model === 'holstein_hubbard';
}

function isBlochRepresentation() {
  return state.representation === 'bloch';
}

function physicalOffsetFromCellOffset(cellOffset) {
  const offset = { x: 0, y: 0 };
  (state.latticeVectors || []).forEach((vector, index) => {
    const scale = Number(cellOffset && cellOffset[index]) || 0;
    offset.x += scale * (Number(vector.x) || 0);
    offset.y += scale * (Number(vector.y) || 0);
  });
  return offset;
}

function refreshBondPhysicalOffset(bond) {
  if (!isBlochRepresentation()) {
    return;
  }
  bond.offset = physicalOffsetFromCellOffset(bond.cellOffset || []);
  bond.periodic = (bond.cellOffset || []).some((value) => Number(value) !== 0);
}

function applyDefaultSiteParameters(site) {
  if (!Number.isFinite(site.epsilon)) {
    site.epsilon = state.globals.epsilon;
  }
  if (!Number.isFinite(site.U)) {
    site.U = state.globals.U;
  }
  if (!Number.isFinite(site.phononOmega)) {
    site.phononOmega = state.globals.phononOmega;
  }
  if (!Number.isFinite(site.electronPhononG)) {
    site.electronPhononG = state.globals.electronPhononG;
  }
  return site;
}

function effectiveBondT(bond) {
  const source = siteById(bond.source);
  const target = siteById(bond.target);
  if (!source || !target) {
    return bond.t;
  }
  const base = bond.kind === 'custom' ? bond.t : baseHoppingForKind(bond.kind);
  if (!state.geometryCoupled) {
    return base;
  }
  const length = bondDistance(bond);
  const scaled = applyDistanceFunction(
    base,
    length,
    state.geometry.r0,
    state.geometry.hoppingParameter,
    state.geometry.hoppingFunction,
  );
  return Number(scaled.toFixed(10));
}

function effectiveBondV(bond) {
  const source = siteById(bond.source);
  const target = siteById(bond.target);
  if (!state.geometryCoupledV || !source || !target) {
    return bond.V;
  }
  const length = Math.max(bondDistance(bond), 1e-8);
  const scaled = applyDistanceFunction(
    bond.V,
    length,
    state.geometry.r0,
    state.geometry.interactionParameter,
    state.geometry.interactionFunction,
  );
  return Number(scaled.toFixed(10));
}

function applyDistanceFunction(baseValue, distanceValue, referenceLength, parameter, functionName) {
  const r = Math.max(distanceValue, 1e-8);
  const r0 = Math.max(referenceLength, 1e-8);
  const a = Number.isFinite(parameter) ? parameter : 1;
  if (functionName === 'constant') {
    return baseValue;
  }
  if (functionName === 'power') {
    return baseValue * Math.pow(r0 / r, a);
  }
  if (functionName === 'linear') {
    return baseValue * (1 - a * (r - r0));
  }
  if (functionName === 'inverse') {
    return baseValue * (r0 / r);
  }
  return baseValue * Math.exp(-a * (r - r0));
}

function addSite(x, y) {
  clearPrimitiveCellMembership();
  const id = nextSiteId();
  state.sites.push({
    id,
    x,
    y,
    epsilon: state.globals.epsilon,
    U: state.globals.U,
    phononOmega: state.globals.phononOmega,
    electronPhononG: state.globals.electronPhononG,
  });
  selectItem('site', id);
  render();
}

function addBond(source, target, kind = 'custom') {
  if (source === target) {
    return;
  }
  const existing = state.bonds.find((bond) => (
    (bond.source === source && bond.target === target)
    || (bond.source === target && bond.target === source)
  ));
  if (existing) {
    selectItem('bond', existing.id);
    return;
  }
  const id = nextBondId();
  state.bonds.push({
    id,
    source,
    target,
    t: baseHoppingForKind(kind),
    V: state.globals.V,
    kind,
    periodic: false,
    offset: { x: 0, y: 0 },
    ...(isBlochRepresentation() ? {
      cellOffset: Array.from({ length: state.dimension }, () => 0),
      addHermitianConjugate: true,
    } : {}),
  });
  selectItem('bond', id);
}

function pushUniqueBond(bonds, source, target, indexHint = bonds.length, options = {}) {
  const periodic = Boolean(options.periodic);
  const offset = options.offset ? { x: options.offset.x, y: options.offset.y } : { x: 0, y: 0 };
  const zeroOffset = Math.abs(offset.x) <= 1e-10 && Math.abs(offset.y) <= 1e-10;
  if (source === target && (!periodic || zeroOffset)) {
    return;
  }
  function sameOffset(bond) {
    const existingOffset = bond.offset || { x: 0, y: 0 };
    return Math.abs(existingOffset.x - offset.x) <= 1e-10
      && Math.abs(existingOffset.y - offset.y) <= 1e-10;
  }
  function oppositeOffset(bond) {
    const existingOffset = bond.offset || { x: 0, y: 0 };
    return Math.abs(existingOffset.x + offset.x) <= 1e-10
      && Math.abs(existingOffset.y + offset.y) <= 1e-10;
  }
  const exists = bonds.some((bond) => (
    bond.periodic === periodic
    && (!periodic || sameOffset(bond) || (source === target && oppositeOffset(bond)))
    && (
      (bond.source === source && bond.target === target)
      || (bond.source === target && bond.target === source)
    )
  ));
  if (exists) {
    return;
  }
  const kind = isSshBondModulationEnabled() ? (indexHint % 2 === 0 ? 'strong' : 'weak') : 'custom';
  bonds.push({
    id: bonds.length,
    source,
    target,
    t: baseHoppingForKind(kind),
    V: state.globals.V,
    kind,
    periodic,
    offset,
  });
}

function removeSelected() {
  if (!state.selected) {
    return;
  }
  if (state.selected.type === 'site') {
    clearPrimitiveCellMembership();
    const siteId = state.selected.id;
    state.sites = state.sites.filter((site) => site.id !== siteId);
    state.bonds = state.bonds.filter((bond) => bond.source !== siteId && bond.target !== siteId);
  } else if (state.selected.type === 'bond') {
    state.bonds = state.bonds.filter((bond) => bond.id !== state.selected.id);
  }
  state.selected = null;
  render();
}

function applyGlobalsToAll() {
  updateGlobalsFromControls();
  state.sites.forEach((site) => {
    site.epsilon = state.globals.epsilon;
    site.U = state.globals.U;
    site.phononOmega = state.globals.phononOmega;
    site.electronPhononG = state.globals.electronPhononG;
  });
  state.bonds.forEach((bond, index) => {
    bond.V = state.globals.V;
    if (isSshBondModulationEnabled() && bond.kind !== 'custom') {
      bond.t = baseHoppingForKind(bond.kind);
    } else if (isSshBondModulationEnabled() && bond.kind === 'custom') {
      bond.kind = index % 2 === 0 ? 'strong' : 'weak';
      bond.t = baseHoppingForKind(bond.kind);
    } else {
      bond.kind = 'custom';
      bond.t = state.globals.t;
    }
  });
  render();
}
