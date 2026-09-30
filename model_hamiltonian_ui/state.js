const canvas = document.getElementById('lattice-canvas');
const axisLayer = document.getElementById('axis-layer');
const ghostLayer = document.getElementById('ghost-layer');
const cellBoundaryLayer = document.getElementById('cell-boundary-layer');
const bondLayer = document.getElementById('bond-layer');
const siteLayer = document.getElementById('site-layer');
const labelLayer = document.getElementById('label-layer');

// Keep the one-body Bloch workflow available to developers without exposing
// it as a supported user workflow until interacting-method routing is ready.
const BLOCH_UI_ENABLED = false;

const MAX_BLOCH_KMESH_AXIS_1D = 1001;
const MAX_BLOCH_KMESH_AXIS_2D = 223;
const MAX_BLOCH_DOS_POINTS = 4001;
const MAX_BLOCH_DOS_WORK_ITEMS = 200000000;

const state = {
  model: 'hubbard',
  representation: 'finite_cluster',
  dimension: 2,
  template1d: 'chain',
  template2d: 'square',
  preset: 'square',
  bondModulation: 'none',
  boundary: 'open',
  energyUnit: 'a.u.',
  nelec: [4, 4],
  spinMultiplicity: 1,
  occupation: {
    electronsPerCell: 1,
  },
  reciprocal: {
    kmesh: [100, 30],
    scheme: 'gamma_centered',
    shift: [0, 0],
    pointsPerSegment: 80,
    dosSigma: 0,
    dosPoints: 401,
  },
  geometryCoupled: false,
  geometryCoupledV: false,
  geometry: {
    r0: 1,
    hoppingFunction: 'exponential',
    hoppingParameter: 1.5,
    interactionFunction: 'inverse',
    interactionParameter: 1,
  },
  globals: {
    t: -1,
    delta: 0.2,
    epsilon: 0,
    U: 4,
    V: 0,
    phononOmega: 1,
    electronPhononG: 0.5,
    phononCutoff: 4,
  },
  sites: [],
  bonds: [],
  selected: null,
  tool: 'select',
  pendingBondSource: null,
  activeTab: 'summary',
  view: {
    zoom: 1,
    panX: 0,
    panY: 0,
    mode: 'edit',
  },
  cell: null,
  primitiveCell: null,
  cellSubdivisions: { x: 1, y: 1 },
  latticeVectors: [],
};

let dragState = null;
let panState = null;

function blochDosPointLimit() {
  const kmesh = state.reciprocal.kmesh || [1, 1];
  const kpointCount = Math.max(1, kmesh[0] || 1) * (state.dimension === 1 ? 1 : Math.max(1, kmesh[1] || 1));
  const orbitalCount = Math.max(1, state.sites.length);
  return Math.max(2, Math.min(
    MAX_BLOCH_DOS_POINTS,
    Math.floor(MAX_BLOCH_DOS_WORK_ITEMS / (kpointCount * orbitalCount)),
  ));
}

function numberValue(id, fallback) {
  const value = Number(document.getElementById(id).value);
  return Number.isFinite(value) ? value : fallback;
}

function setNumberValue(id, value) {
  document.getElementById(id).value = Number.isFinite(value) ? String(Number(value.toFixed(6))) : '';
}

function clampInt(value, min, max) {
  const rounded = Math.round(value);
  return Math.max(min, Math.min(max, rounded));
}

function nextSiteId() {
  return state.sites.reduce((highest, site) => Math.max(highest, site.id), -1) + 1;
}

function nextBondId() {
  return state.bonds.reduce((highest, bond) => Math.max(highest, bond.id), -1) + 1;
}

function siteById(id) {
  return state.sites.find((site) => site.id === id) || null;
}

function bondById(id) {
  return state.bonds.find((bond) => bond.id === id) || null;
}

function clearPrimitiveCellMembership() {
  state.primitiveCell = null;
  state.sites.forEach((site) => {
    delete site.cellIndex;
    delete site.basisIndex;
  });
}
