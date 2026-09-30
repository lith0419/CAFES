const HAMILTONIAN_DATASET_TEMPLATE = 'hamiltonian_dataset';

function isHamiltonianDatasetPlanner() {
  const select = document.getElementById('planner-template');
  return Boolean(select && select.value === HAMILTONIAN_DATASET_TEMPLATE);
}

function datasetTargetMoleculeCount() {
  return Number.parseInt(document.getElementById('dataset-molecule-count').value, 10) || 100;
}

function renderDatasetSeedStatus() {
  const target = document.getElementById('dataset-seed-status');
  if (!target) return;
  const expected = datasetTargetMoleculeCount();
  const observed = currentDatasetSeedGeometries.length;
  const filename = currentDatasetSeedFileName ? ` from ${currentDatasetSeedFileName}` : '';
  target.className = 'hint';
  if (!observed) {
    target.textContent = `Import exactly ${expected} neutral closed-shell seed molecules as JSON or JSONL.`;
    return;
  }
  if (observed !== expected) {
    target.className = 'hint dataset-seed-error';
    target.textContent = `Loaded ${observed}${filename}; exactly ${expected} seed molecules are required.`;
    return;
  }
  target.className = 'hint dataset-seed-ready';
  target.textContent = `Loaded ${observed} seed molecules${filename}. Ready to build 100 MD cases and 1,000 sampled structures.`;
}

function parseDatasetSeedGeometries(text) {
  let payload;
  try {
    payload = JSON.parse(text);
  } catch (_error) {
    payload = text.split(/\r?\n/).filter((line) => line.trim()).map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (lineError) {
        throw new Error(`Invalid JSONL on line ${index + 1}: ${lineError.message}`);
      }
    });
  }
  if (payload && typeof payload === 'object' && !Array.isArray(payload)) {
    payload = payload.seed_geometries;
  }
  if (!Array.isArray(payload)) {
    throw new Error('Seed geometry file must contain a JSON array, JSONL records, or {"seed_geometries": [...]}.');
  }
  if (payload.some((item) => !item || typeof item !== 'object' || Array.isArray(item))) {
    throw new Error('Every seed geometry must be a JSON object.');
  }
  return deepCopy(payload);
}

function resetDatasetPlan(message) {
  installFullStudyPlan(null);
  currentReport = null;
  currentAdaptivePreview = null;
  currentAnalysisRows = [];
  document.getElementById('run-study').disabled = true;
  renderPlanSummary(null);
  document.getElementById('case-table').className = 'empty';
  document.getElementById('case-table').innerHTML = 'Build a plan to see cases.';
  renderReviewGate(null);
  document.getElementById('report-summary').className = 'empty';
  document.getElementById('report-summary').innerHTML = 'No run yet.';
  setStatus('run-status', message || 'Dataset configuration changed. Build the plan again.');
  updateActiveTaskSessionStatus('draft');
}

async function handleDatasetSeedFileChange(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) {
    currentDatasetSeedGeometries = [];
    currentDatasetSeedFileName = '';
    renderDatasetSeedStatus();
    resetDatasetPlan('Seed geometries cleared.');
    snapshotCurrentTaskSession();
    return;
  }
  try {
    currentDatasetSeedGeometries = parseDatasetSeedGeometries(await file.text());
    currentDatasetSeedFileName = file.name;
    renderDatasetSeedStatus();
    resetDatasetPlan(`Loaded ${currentDatasetSeedGeometries.length} seed geometries. Build the dataset plan when ready.`);
  } catch (error) {
    currentDatasetSeedGeometries = [];
    currentDatasetSeedFileName = file.name;
    renderDatasetSeedStatus();
    setStatus('run-status', error.message, 'error');
  }
  snapshotCurrentTaskSession();
}

function hamiltonianDatasetSpecFromControls() {
  const datasetId = document.getElementById('dataset-id').value.trim();
  const name = document.getElementById('dataset-name').value.trim();
  const rawSplitSeed = document.getElementById('dataset-split-seed').value.trim();
  const splitSeed = rawSplitSeed ? Number(rawSplitSeed) : NaN;
  if (!datasetId) throw new Error('Dataset ID is required.');
  if (!name) throw new Error('Dataset Name is required.');
  if (!Number.isSafeInteger(splitSeed) || splitSeed < 0) {
    throw new Error('Split Seed must be a non-negative integer.');
  }
  return {
    dataset_id: datasetId,
    name,
    target_molecule_count: 100,
    geometries_per_molecule: 10,
    electronic_structure: {
      method: 'dft',
      xc: 'b3lyp',
      basis: 'def2-svp',
      restricted: true,
      scf_options: {
        grid_level: 3,
        conv_tol: 1e-8,
        conv_tol_grad: 3.16e-5,
        diis_space: 8,
      },
    },
    molecular_dynamics: {
      ensemble: 'nve',
      temperature_kelvin: 300.0,
      time_step_au: 50.0,
      steps: 100,
      sample_stride: 10,
      sample_offset: 9,
      base_velocity_seed: 0,
    },
    compatibility_profile: 'qh9_relaxed_scf',
    ao_convention: 'qh9',
    coordinate_unit: 'Angstrom',
    split_protocol: document.getElementById('dataset-split-protocol').value,
    split_fractions: {train: 0.8, validation: 0.1, test: 0.1},
    split_seed: splitSeed,
    require_converged: true,
  };
}

function hamiltonianDatasetPlanningPayload() {
  const expected = datasetTargetMoleculeCount();
  if (currentDatasetSeedGeometries.length !== expected) {
    throw new Error(`Import exactly ${expected} seed geometries before building the dataset plan.`);
  }
  return {
    planner_template: HAMILTONIAN_DATASET_TEMPLATE,
    dataset_spec: hamiltonianDatasetSpecFromControls(),
    seed_geometries: deepCopy(currentDatasetSeedGeometries),
  };
}

function syncDatasetTemplateControls() {
  const enabled = isHamiltonianDatasetPlanner();
  const systemSelect = document.getElementById('study-system');
  if (enabled) {
    systemSelect.value = 'molecular';
    document.getElementById('study-mode').value = 'static';
    document.getElementById('dataset-molecule-count').value = 100;
    document.getElementById('dataset-frame-count').value = 10;
  }
  systemSelect.disabled = enabled;
  syncAdaptiveActiveSpaceControls();
  updateTaskSetupPanel();
  renderDatasetSeedStatus();
}

function handlePlannerTemplateChange() {
  syncDatasetTemplateControls();
  resetDatasetPlan(isHamiltonianDatasetPlanner()
    ? 'Hamiltonian Dataset selected. Import 100 seed geometries, then build the plan.'
    : 'Standard Study selected. Build a plan after configuring the study.');
  updatePlannerComposerForSystem(document.getElementById('study-system').value);
  snapshotCurrentTaskSession();
}

function handleDatasetConfigurationChange() {
  if (!isHamiltonianDatasetPlanner()) return;
  renderDatasetSeedStatus();
  resetDatasetPlan('Dataset configuration changed. Build the plan again.');
  snapshotCurrentTaskSession();
}

function renderDatasetManifestSummary(manifest) {
  if (!manifest || typeof manifest !== 'object' || !manifest.schema) return '';
  const splits = manifest.split_counts && typeof manifest.split_counts === 'object'
    ? Object.entries(manifest.split_counts).map(([name, count]) => `${name}=${count}`).join(', ')
    : 'n/a';
  const values = [
    ['Dataset', manifest.status || 'unknown'],
    ['Accepted', manifest.accepted_structure_count],
    ['Rejected', manifest.rejected_structure_count],
    ['Splits', splits],
  ];
  return `<div class="dataset-manifest-summary">${values.map(([label, value]) => (
    `<div class="metric"><strong>${escapeHtml(value ?? 'n/a')}</strong><span>${escapeHtml(label)}</span></div>`
  )).join('')}</div>`;
}
