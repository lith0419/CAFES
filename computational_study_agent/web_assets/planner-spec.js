function summarizePlannerSpec(spec) {
  if (!spec || typeof spec !== 'object') {
    return 'No planning state is available yet.';
  }
  const lines = [];
  if (spec.objective) {
    lines.push(`Objective: ${spec.objective}`);
  }
  if (spec.system_type) {
    lines.push(`System: ${spec.system_type}`);
  }
  if (spec.base_model_input_file) {
    lines.push(`Model input: ${spec.base_model_input_file}`);
  }
  if (spec.base_task && typeof spec.base_task === 'object') {
    const taskBits = Object.entries(spec.base_task).map(([key, value]) => {
      const rendered = key === 'solver' ? plannerSolverName(value) : setupValue(value, '');
      return `${key}=${rendered}`;
    });
    if (taskBits.length) {
      lines.push(`Base task: ${taskBits.join('; ')}`);
    }
  }
  const variableKeys = spec.case_design && spec.case_design.variables ? Object.keys(spec.case_design.variables) : [];
  const sweepKeys = spec.sweep && typeof spec.sweep === 'object' ? Object.keys(spec.sweep) : [];
  if (variableKeys.length) {
    lines.push(`Case variables: ${variableKeys.join(', ')}`);
  } else if (sweepKeys.length) {
    lines.push(`Sweep variables: ${sweepKeys.join(', ')}`);
  }
  if (Array.isArray(spec.observables) && spec.observables.length) {
    lines.push(`Observables: ${spec.observables.join(', ')}`);
  }
  return lines.length ? lines.join('\n') : 'A draft planning state has been prepared.';
}

function setupValue(value, fallback = 'n/a') {
  if (value === null || value === undefined || value === '') {
    return fallback;
  }
  if (typeof value === 'boolean') {
    return value ? 'yes' : 'no';
  }
  if (Array.isArray(value)) {
    return value.length ? value.map((item) => setupValue(item, '')).filter(Boolean).join(', ') : fallback;
  }
  if (typeof value === 'object') {
    const keys = Object.keys(value);
    return keys.length ? JSON.stringify(value) : fallback;
  }
  return String(value);
}

function plannerSolverContract(value) {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return {
      name: String(value.name || '').trim().toLowerCase().replace(/[-\s]+/g, '_'),
      options: value.options && typeof value.options === 'object' && !Array.isArray(value.options)
        ? deepCopy(value.options)
        : {},
    };
  }
  return {
    name: String(value || '').trim().toLowerCase().replace(/[-\s]+/g, '_'),
    options: {},
  };
}

function plannerSolverName(value) {
  return plannerSolverContract(value).name;
}

function plannerDmetDefaults() {
  return {
    execution_mode: '',
    fragment_definition: '',
    impurity_solver: 'fci',
    impurity_solver_options: {},
    impurity_shape: [],
    impurity_size: null,
    reference: 'unrestricted',
    reference_density_guess: 'pm',
    interacting_bath: true,
    max_iterations: 50,
    energy_tolerance: 1e-6,
    density_tolerance: 1e-4,
  };
}

function plannerPositiveNumber(id, fallback, integer = false) {
  const raw = document.getElementById(id).value.trim();
  if (!raw) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0 || (integer && !Number.isSafeInteger(value))) {
    throw new Error(`${id}: enter a positive ${integer ? 'integer' : 'number'}.`);
  }
  return value;
}

function plannerDmetShapeFromInput() {
  const raw = document.getElementById('planner-dmet-impurity-shape').value.trim();
  if (!raw) return [];
  const values = raw.split(/[,xX\s]+/).filter(Boolean).map((value) => Number(value));
  if (!values.length || !values.every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new Error('Impurity shape must contain positive integers.');
  }
  return values;
}

function plannerDmetOptionsFromControls() {
  const fragmentDefinition = document.getElementById('planner-dmet-fragment-definition').value;
  const impuritySolver = document.getElementById('planner-dmet-impurity-solver').value;
  const executionMode = document.getElementById('planner-dmet-execution-mode').value;
  const options = {
    impurity_solver: impuritySolver,
    impurity_solver_options: impuritySolver === 'block2_dmrg'
      ? {
        preset: document.getElementById('planner-dmet-block2-preset').value,
        orbital_ordering: document.getElementById('planner-dmet-block2-ordering').value,
      }
      : {},
    impurity_shape: fragmentDefinition === 'cell_shape' ? plannerDmetShapeFromInput() : [],
    impurity_size: fragmentDefinition === 'site_count'
      ? plannerPositiveNumber('planner-dmet-impurity-size', 1, true)
      : null,
    reference: document.getElementById('planner-dmet-reference').value,
    reference_density_guess: document.getElementById('planner-dmet-reference-density').value,
    interacting_bath: document.getElementById('planner-dmet-interacting-bath').value !== 'false',
    max_iterations: plannerPositiveNumber('planner-dmet-max-iterations', 50, true),
    energy_tolerance: plannerPositiveNumber('planner-dmet-energy-tolerance', 1e-6),
    density_tolerance: plannerPositiveNumber('planner-dmet-density-tolerance', 1e-4),
  };
  const betaInput = document.getElementById('planner-dmet-ccsd-beta');
  if (impuritySolver === 'ccsd' && (betaInput.value.trim() || betaInput.validity.badInput)) {
    options.impurity_solver_options.beta = betaInput.validity.badInput ? null : Number(betaInput.value);
  }
  if (executionMode) options.execution_mode = executionMode;
  if (fragmentDefinition) options.fragment_definition = fragmentDefinition;
  return options;
}

function syncPlannerDmetConditionalControls() {
  const isDmet = document.getElementById('planner-model-solver').value === 'dmet';
  const fragmentDefinition = document.getElementById('planner-dmet-fragment-definition').value;
  const impuritySolver = document.getElementById('planner-dmet-impurity-solver').value;
  document.getElementById('planner-dmet-controls').classList.toggle('hidden', !isDmet);
  document.getElementById('planner-dmet-ccsd-beta-control').classList.toggle('hidden', !isDmet || impuritySolver !== 'ccsd');
  document.getElementById('planner-dmet-ccsd-beta').disabled = !isDmet || impuritySolver !== 'ccsd';
  document.getElementById('planner-dmet-impurity-shape-control').classList.toggle('hidden', !isDmet || fragmentDefinition !== 'cell_shape');
  document.getElementById('planner-dmet-impurity-size-control').classList.toggle('hidden', !isDmet || fragmentDefinition !== 'site_count');
  document.getElementById('planner-dmet-block2-preset-control').classList.toggle('hidden', !isDmet || impuritySolver !== 'block2_dmrg');
  document.getElementById('planner-dmet-block2-ordering-control').classList.toggle('hidden', !isDmet || impuritySolver !== 'block2_dmrg');
}

function syncPlannerModelSolverControls(spec) {
  const baseTask = spec && spec.base_task && typeof spec.base_task === 'object' ? spec.base_task : {};
  const contract = plannerSolverContract(baseTask.solver || 'fci');
  const solverSelect = document.getElementById('planner-model-solver');
  if ([...solverSelect.options].some((option) => option.value === contract.name)) {
    solverSelect.value = contract.name;
  } else {
    solverSelect.value = 'fci';
  }
  const options = {...plannerDmetDefaults(), ...(contract.name === 'dmet' ? contract.options : {})};
  document.getElementById('planner-dmet-execution-mode').value = options.execution_mode;
  document.getElementById('planner-dmet-fragment-definition').value = options.fragment_definition;
  document.getElementById('planner-dmet-impurity-shape').value = Array.isArray(options.impurity_shape) ? options.impurity_shape.join(',') : '';
  document.getElementById('planner-dmet-impurity-size').value = options.impurity_size || 1;
  document.getElementById('planner-dmet-impurity-solver').value = options.impurity_solver;
  const impurityOptions = options.impurity_solver_options && typeof options.impurity_solver_options === 'object'
    ? options.impurity_solver_options
    : {};
  document.getElementById('planner-dmet-ccsd-beta').value = impurityOptions.beta ?? '';
  document.getElementById('planner-dmet-block2-preset').value = impurityOptions.preset || 'balanced';
  document.getElementById('planner-dmet-block2-ordering').value = impurityOptions.orbital_ordering || 'canonical';
  document.getElementById('planner-dmet-reference').value = options.reference;
  document.getElementById('planner-dmet-reference-density').value = options.reference_density_guess;
  document.getElementById('planner-dmet-interacting-bath').value = String(options.interacting_bath !== false);
  document.getElementById('planner-dmet-max-iterations').value = options.max_iterations;
  document.getElementById('planner-dmet-energy-tolerance').value = options.energy_tolerance;
  document.getElementById('planner-dmet-density-tolerance').value = options.density_tolerance;
  syncPlannerDmetConditionalControls();
}

function applyPlannerModelSolverToSpec(spec) {
  if (!spec || normalizeSystemType(spec.system_type) !== 'model_hamiltonian') return spec;
  if (!spec.base_task || typeof spec.base_task !== 'object' || Array.isArray(spec.base_task)) {
    spec.base_task = {};
  }
  const solver = document.getElementById('planner-model-solver').value || 'fci';
  spec.base_task.solver = solver === 'dmet'
    ? {name: 'dmet', options: plannerDmetOptionsFromControls()}
    : solver;
  return spec;
}

function handlePlannerModelSolverControlsChange() {
  syncPlannerDmetConditionalControls();
  let spec;
  try {
    spec = parseStudySpec();
  } catch (_error) {
    spec = blankStudySpecForSystem('model_hamiltonian');
  }
  spec.system_type = 'model_hamiltonian';
  applyPlannerModelSolverToSpec(spec);
  document.getElementById('study-spec').value = prettyJson(spec);
  handleStudyModeChange();
}

function caseCountFromStudySpec(spec) {
  if (!spec || typeof spec !== 'object') {
    return 0;
  }
  const caseDesign = spec.case_design && typeof spec.case_design === 'object' ? spec.case_design : {};
  if (Array.isArray(caseDesign.cases) && caseDesign.cases.length) {
    return caseDesign.cases.length;
  }
  const variables = caseDesign.variables && typeof caseDesign.variables === 'object'
    ? caseDesign.variables
    : (spec.sweep && typeof spec.sweep === 'object' ? spec.sweep : {});
  const keys = Object.keys(variables || {});
  if (!keys.length) {
    return 1;
  }
  return keys.reduce((total, key) => {
    const values = variables[key];
    return total * (Array.isArray(values) && values.length ? values.length : 1);
  }, 1);
}

function variableKeysFromStudySpec(spec) {
  if (!spec || typeof spec !== 'object') {
    return [];
  }
  const caseVariables = spec.case_design && spec.case_design.variables && typeof spec.case_design.variables === 'object'
    ? Object.keys(spec.case_design.variables)
    : [];
  if (caseVariables.length) {
    return caseVariables;
  }
  return spec.sweep && typeof spec.sweep === 'object' ? Object.keys(spec.sweep) : [];
}

function renderTaskSetupItem(label, value) {
  return `<div class="task-setup-item"><span>${escapeHtml(label)}</span><strong>${escapeHtml(setupValue(value))}</strong></div>`;
}

function firstPreviewCaseVariables(spec) {
  const caseDesign = spec && spec.case_design && typeof spec.case_design === 'object'
    ? spec.case_design
    : {};
  if (Array.isArray(caseDesign.cases) && caseDesign.cases.length) {
    const firstCase = caseDesign.cases[0];
    return firstCase && typeof firstCase.variables === 'object' ? firstCase.variables : {};
  }
  const variables = caseDesign.variables && typeof caseDesign.variables === 'object'
    ? caseDesign.variables
    : (spec && spec.sweep && typeof spec.sweep === 'object' ? spec.sweep : {});
  return Object.fromEntries(Object.entries(variables).map(([name, values]) => [
    name,
    Array.isArray(values) ? values[0] : values,
  ]).filter(([, value]) => value !== undefined));
}

function resolvePreviewTemplateValue(value, variables) {
  if (typeof value !== 'string') {
    return { value, resolved: true };
  }
  let resolved = true;
  const rendered = value.replace(/\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))/g, (token, braced, bare) => {
    const name = braced || bare;
    if (!Object.prototype.hasOwnProperty.call(variables, name)) {
      resolved = false;
      return token;
    }
    return String(variables[name]);
  });
  // Arithmetic templates are expanded by the backend when it builds a case.
  // Do not partially render their coordinates in the client-side preview.
  if (/\$[({]/.test(rendered)) {
    resolved = false;
  }
  return { value: rendered, resolved };
}

function molecularPreviewTask(spec, task) {
  const plannedCase = currentPlan && Array.isArray(currentPlan.cases)
    ? currentPlan.cases.find((item) => item && item.request && typeof item.request === 'object'
      && (currentPlan.system_type === 'molecular' || item.request.task_type === 'molecular'))
    : null;
  if (plannedCase && typeof plannedCase.request.atom === 'string' && !plannedCase.request.atom.includes('$')) {
    return {
      atom: plannedCase.request.atom,
      unit: plannedCase.request.unit || task.unit || 'Angstrom',
      resolved: true,
    };
  }

  const caseDesign = spec && spec.case_design && typeof spec.case_design === 'object'
    ? spec.case_design
    : {};
  const firstCase = Array.isArray(caseDesign.cases) ? caseDesign.cases[0] : null;
  const template = caseDesign.template && typeof caseDesign.template === 'object'
    ? caseDesign.template
    : {};
  const templateUpdates = template.request_updates && typeof template.request_updates === 'object'
    ? template.request_updates
    : {};
  const caseUpdates = firstCase && firstCase.request_updates && typeof firstCase.request_updates === 'object'
    ? firstCase.request_updates
    : {};
  const atomTemplate = Object.prototype.hasOwnProperty.call(caseUpdates, 'atom')
    ? caseUpdates.atom
    : (Object.prototype.hasOwnProperty.call(templateUpdates, 'atom') ? templateUpdates.atom : task.atom);
  const preview = resolvePreviewTemplateValue(atomTemplate, firstPreviewCaseVariables(spec));
  return {
    atom: preview.value,
    unit: task.unit || 'Angstrom',
    resolved: preview.resolved,
  };
}

function renderMolecularTaskSetup(spec) {
  const task = spec && spec.base_task && typeof spec.base_task === 'object' ? spec.base_task : {};
  const preview = molecularPreviewTask(spec, task);
  const adaptive = currentStudyMode() === 'adaptive';
  const initialScanSelect = document.getElementById('adaptive-initial-scan-strategy');
  const initialScanLabel = initialScanSelect && initialScanSelect.selectedOptions.length
    ? initialScanSelect.selectedOptions[0].textContent
    : 'Auto';
  const reference = task.reference || (task.restricted === false ? 'unrestricted' : (task.restricted === true ? 'restricted' : 'default'));
  const chargeSpin = [
    `charge=${setupValue(task.charge, '0')}`,
    `spin=${setupValue(task.spin, '0')}`,
  ].join('; ');
  const items = [
    ['Objective', spec && spec.objective],
    ['Basis', task.basis],
    [adaptive ? 'Initial scan' : 'Method', adaptive ? initialScanLabel : task.method],
    ['Reference', reference],
    ['Charge / Spin', chargeSpin],
    ['Cases', caseCountFromStudySpec(spec)],
    ['Variables', variableKeysFromStudySpec(spec).join(', ') || 'n/a'],
    ['Observables', Array.isArray(spec && spec.observables) ? spec.observables.join(', ') : 'n/a'],
  ];
  document.getElementById('molecular-task-summary').innerHTML = items
    .map(([label, value]) => renderTaskSetupItem(label, value))
    .join('');
  document.getElementById('molecular-task-structure').textContent = setupValue(preview.atom, 'No molecular structure specified.');
  if (!preview.resolved) {
    document.getElementById('molecular-task-structure-preview').innerHTML = '<div class="molecule-preview-empty">Build Plan to resolve the scan coordinates for this preview.</div>';
    return;
  }
  renderMolecularStructurePreview('molecular-task-structure-preview', preview.atom, {
    unit: preview.unit,
    emptyText: 'No molecular structure specified.',
    parseErrorText: 'Could not parse molecular coordinates.',
    atomLabel: 'atoms',
    bondLabel: 'inferred bonds',
    unitLabel: 'unit',
    bondLengthLabel: 'bond length',
  });
}

function updateTaskSetupPanel(spec = null) {
  const payload = spec || currentStudySpecSafe();
  const datasetPlanner = isHamiltonianDatasetPlanner();
  const systemType = normalizeSystemType(payload.system_type || document.getElementById('study-system').value);
  const isMolecular = systemType === 'molecular';
  document.getElementById('task-setup-title').textContent = datasetPlanner ? 'Dataset Setup' : 'Task Setup';
  document.getElementById('task-setup-hint').textContent = datasetPlanner
    ? 'One MD trajectory per seed molecule; ten Hamiltonian frames per trajectory.'
    : (isMolecular
    ? 'Molecular Electronic Structure'
    : 'Model Hamiltonian');
  document.getElementById('hamiltonian-dataset-setup').classList.toggle('hidden', !datasetPlanner);
  document.getElementById('model-hamiltonian-setup').classList.toggle('hidden', datasetPlanner || isMolecular);
  document.getElementById('molecular-task-setup').classList.toggle('hidden', datasetPlanner || !isMolecular);
  if (datasetPlanner) {
    renderDatasetSeedStatus();
  } else if (isMolecular) {
    renderMolecularTaskSetup(payload);
  } else {
    syncPlannerModelSolverControls(payload);
    syncGridRefinementControls(payload);
  }
}

function parseIntegerRange(text) {
  const rangeMatch = text.match(/(?:from|between)\s+(-?\d+)\s+(?:to|and)\s+(-?\d+)/i)
    || text.match(/(-?\d+)\s*(?:\.\.|-|to)\s*(-?\d+)/i);
  if (!rangeMatch) {
    return null;
  }
  const start = Number.parseInt(rangeMatch[1], 10);
  const end = Number.parseInt(rangeMatch[2], 10);
  if (!Number.isFinite(start) || !Number.isFinite(end)) {
    return null;
  }
  const step = start <= end ? 1 : -1;
  const values = [];
  for (let value = start; step > 0 ? value <= end : value >= end; value += step) {
    values.push(value);
  }
  return values;
}

function electronPair(total) {
  return [Math.floor((total + 1) / 2), Math.floor(total / 2)];
}

function planningDecomposition(message, spec) {
  const lower = message.toLowerCase();
  const steps = [];
  steps.push('Interpret the request against the current input model and hidden planning state.');
  if (lower.includes('electron') || lower.includes('nelec')) {
    const values = parseIntegerRange(message);
    if (values && values.length) {
      const preview = values.slice(0, 6).map((value) => `${value}->(${electronPair(value).join(',')})`).join(', ');
      steps.push(`Decompose total electron scan into spin-resolved nelec values: ${preview}${values.length > 6 ? ', ...' : ''}.`);
      steps.push('Use alpha=beta for even total electron counts and alpha=beta+1 for odd counts.');
    } else {
      steps.push('Treat electron-count changes as PySCF nelec=(nalpha, nbeta) updates, not as a scalar field.');
    }
  }
  if (lower.includes('defect') || lower.includes('site')) {
    steps.push('Map local modifications to supported site operations such as set_site_parameter or add_site_defect.');
  }
  if (lower.includes('method') || lower.includes('solver') || lower.includes('mp2') || lower.includes('fci') || lower.includes('ccsd') || lower.includes('ci')) {
    steps.push('Map method choices to the supported solver list before building cases.');
  }
  if (spec && spec.system_type) {
    steps.push(`Keep the current system type fixed unless the conversation asks to change it: ${spec.system_type}.`);
  }
  steps.push('Validate the generated workflow before showing it as ready for approval.');
  return steps.join('\n');
}

function parseStudySpec() {
  try {
    return JSON.parse(document.getElementById('study-spec').value || '{}');
  } catch (error) {
    throw new Error(`Planner spec JSON is invalid: ${error.message}`);
  }
}

function effectiveWorkDir() {
  return document.getElementById('work-dir').value.trim() || DEFAULT_WORK_DIR;
}

function executionWorkDir() {
  const session = activeTaskSession();
  if (session && session.workDirLocked && session.workDirRoot) {
    return session.workDirRoot;
  }
  return effectiveWorkDir();
}

function lockWorkDirectory(displayPath, rootPath = executionWorkDir()) {
  const input = document.getElementById('work-dir');
  const resolvedDisplayPath = String(displayPath || '').trim();
  if (resolvedDisplayPath) {
    input.value = resolvedDisplayPath;
  }
  input.disabled = true;
  const session = activeTaskSession();
  if (session) {
    session.workDir = input.value;
    session.workDirRoot = String(rootPath || '').trim() || DEFAULT_WORK_DIR;
    session.workDirLocked = true;
  }
  updateEffectiveWorkDir();
}

function updateEffectiveWorkDir() {
  const rawValue = document.getElementById('work-dir').value.trim();
  document.getElementById('effective-work-dir').textContent = rawValue ? `Using: ${effectiveWorkDir()}` : '';
}

function makeRunId() {
  return AgentUI.makeRunId();
}

function ensureCurrentRunId() {
  if (!currentRunId) {
    currentRunId = makeRunId();
  }
  return currentRunId;
}

function setStudySpec(nextSpec) {
  document.getElementById('study-spec').value = prettyJson(nextSpec);
  syncStudySystemFromSpec(nextSpec);
  installFullStudyPlan(null);
  currentAdaptivePreview = null;
  currentAnalysisRows = [];
  document.getElementById('run-study').disabled = true;
  setStatus('run-status', 'Plan changed. Build before running.');
  renderAdaptiveState(null);
  updateActiveTaskSessionStatus('draft');
  snapshotCurrentTaskSession();
}

function defaultStudySpecForSystem(systemType) {
  return normalizeSystemType(systemType) === 'molecular'
    ? deepCopy(DEFAULT_MOLECULAR_STUDY_SPEC)
    : deepCopy(DEFAULT_STUDY_SPEC);
}

function blankStudySpecForSystem(systemType) {
  if (normalizeSystemType(systemType) === 'molecular') {
    return {
      name: 'molecular-study',
      objective: '',
      system_type: 'molecular',
      base_task: {},
      case_design: {'mode': 'cases', 'cases': []},
      observables: ['energy', 'homo_lumo', 'dipole'],
      comparison: {},
    };
  }
  return {
    name: 'model-hamiltonian-study',
    objective: '',
    system_type: 'model_hamiltonian',
    base_model_spec: {},
    base_task: {solver: 'fci'},
    case_design: {'mode': 'cases', 'cases': []},
    observables: ['energy', 'strong_correlation_diagnostics'],
    comparison: {},
  };
}

function updatePlannerComposerForSystem(systemType) {
  const goal = document.getElementById('goal');
  if (!goal) return;
  goal.placeholder = isHamiltonianDatasetPlanner()
    ? 'After submission, ask the Planner to check status or collect results.'
    : (normalizeSystemType(systemType) === 'molecular'
    ? 'Example: Scan a diatomic bond from 0.8x to 3.0x its reference length.'
    : 'Example: Scan Hubbard U = 2, 4, 6, 8 and compare FCI energies.');
}

function selectedStudySystem() {
  return normalizeSystemType(document.getElementById('study-system').value || currentSystemType());
}

function syncStudySystemFromSpec(spec = null) {
  const payload = spec || currentStudySpecSafe();
  const systemType = normalizeSystemType(payload.system_type || 'model_hamiltonian');
  document.getElementById('study-system').value = systemType === 'molecular' ? 'molecular' : 'model_hamiltonian';
  if (systemType === 'model_hamiltonian') {
    document.getElementById('study-mode').value = 'static';
  } else if (payload.study_mode === 'static' || payload.study_mode === 'adaptive') {
    document.getElementById('study-mode').value = payload.study_mode;
  }
  syncStudyModeControls();
  updateTaskSetupPanel(payload);
  updatePlannerComposerForSystem(systemType);
}

function loadExampleForSelectedSystem() {
  const systemType = selectedStudySystem();
  const exampleSpec = defaultStudySpecForSystem(systemType);
  setStudySpec(exampleSpec);
  if (systemType === 'molecular') {
    document.getElementById('adaptive-initial-scan-strategy').value = 'auto';
    document.getElementById('model-hamiltonian-input-file').value = '';
    resetModelPreview('Molecular studies do not use the model builder input.');
    document.getElementById('study-mode').value = 'adaptive';
  } else {
    document.getElementById('study-mode').value = 'static';
    document.getElementById('model-hamiltonian-input-file').value = '';
    resetModelPreview();
  }
  clearPlannerTaskViews({ preserveMessages: true });
  syncStudyModeControls();
  updateTaskSetupPanel(exampleSpec);
  plannerMessages = [
    {
      role: 'system',
      content: systemType === 'molecular'
        ? 'Example loaded: molecular adaptive H2 bond scan.'
        : 'Example loaded: sweep Hubbard U and compare FCI energies.',
    },
    {
      role: 'assistant',
      content: summarizePlannerSpec(exampleSpec),
    },
  ];
  renderPlannerChat();
  setStatus('draft-status', 'Example loaded.', 'ok');
  updateActiveTaskSessionStatus('draft');
  snapshotCurrentTaskSession();
}

function handleStudySystemChange() {
  const systemType = selectedStudySystem();
  const blankSpec = blankStudySpecForSystem(systemType);
  setStudySpec(blankSpec);
  clearPlannerTaskViews();
  document.getElementById('goal').value = '';
  updatePlannerComposerForSystem(systemType);
  document.getElementById('study-mode').value = systemType === 'molecular' ? 'adaptive' : 'static';
  document.getElementById('adaptive-initial-scan-strategy').value = 'auto';
  syncAdaptiveActiveSpaceControls();
  updateTaskSetupPanel(blankSpec);
  setStatus('run-status', systemType === 'molecular'
    ? 'New molecular task draft created.'
    : 'New model-Hamiltonian task draft created.');
  updateActiveTaskSessionStatus('draft');
  snapshotCurrentTaskSession();
}

function defaultModelHamiltonianStudySpec() {
  return {
    name: 'model-hamiltonian-study',
    objective: 'Study a model Hamiltonian generated by the builder',
    system_type: 'model_hamiltonian',
    base_task: {solver: 'fci'},
    sweep: {},
    observables: ['energy', 'strong_correlation_diagnostics'],
    comparison: {},
  };
}

function setModelInputFile(inputFile) {
  document.getElementById('model-hamiltonian-input-file').value = inputFile || '';
  if (!inputFile) {
    return;
  }
  let spec = {};
  try {
    spec = parseStudySpec();
  } catch (_error) {
    spec = defaultModelHamiltonianStudySpec();
  }
  if (!spec || typeof spec !== 'object' || Array.isArray(spec)) {
    spec = defaultModelHamiltonianStudySpec();
  }
  spec.system_type = 'model_hamiltonian';
  spec.base_model_input_file = inputFile;
  delete spec.base_model_spec;
  if (!spec.base_task || typeof spec.base_task !== 'object') {
    spec.base_task = {};
  }
  setStudySpec(spec);
}

function syncModelInputFileFromStudySpec() {
  try {
    const spec = parseStudySpec();
    const inputFile = typeof spec.base_model_input_file === 'string' ? spec.base_model_input_file.trim() : '';
    document.getElementById('model-hamiltonian-input-file').value = inputFile;
    return inputFile;
  } catch (_error) {
    return document.getElementById('model-hamiltonian-input-file').value.trim();
  }
}

function applyPlannerModeContext(spec, mode = currentStudyMode(), adaptive = null) {
  if (!spec || typeof spec !== 'object' || Array.isArray(spec)) {
    return spec;
  }
  const systemType = normalizeSystemType(spec.system_type);
  const resolvedMode = systemType === 'model_hamiltonian' ? 'static' : mode;
  spec.study_mode = resolvedMode;
  if (resolvedMode !== 'adaptive') {
    delete spec.adaptive;
    return spec;
  }
  const options = adaptive || adaptiveOptions();
  spec.adaptive = options;
  return spec;
}

function prepareStudySpecForRequest(mode = currentStudyMode(), adaptive = null) {
  const spec = parseStudySpec();
  spec.system_type = selectedStudySystem();
  applyPlannerModelSolverToSpec(spec);
  applyGridRefinementControls(spec);
  const linkedInputFile = document.getElementById('model-hamiltonian-input-file').value.trim() || syncModelInputFileFromStudySpec();
  if (normalizeSystemType(spec.system_type) !== 'model_hamiltonian' || !linkedInputFile) {
    return applyPlannerModeContext(spec, mode, adaptive);
  }
  const hasModelInput = typeof spec.base_model_input_file === 'string' && spec.base_model_input_file.trim();
  if (hasModelInput && spec.base_model_input_file.trim() === linkedInputFile) {
    return applyPlannerModeContext(spec, mode, adaptive);
  }
  spec.system_type = 'model_hamiltonian';
  spec.base_model_input_file = linkedInputFile;
  delete spec.base_model_spec;
  if (!spec.base_task || typeof spec.base_task !== 'object' || Array.isArray(spec.base_task)) {
    spec.base_task = {};
  }
  return applyPlannerModeContext(spec, mode, adaptive);
}

function plannerRequestContext() {
  const mode = currentStudyMode();
  const adaptive = mode === 'adaptive' ? adaptiveOptions() : null;
  return {
    mode,
    adaptive,
    studySpec: prepareStudySpecForRequest(mode, adaptive),
  };
}

function openModelBuilder() {
  window.name = 'pyscf-computational-study-agent';
  const params = new URLSearchParams({
    work_dir: effectiveWorkDir(),
    run_id: ensureCurrentRunId(),
    target: 'study',
  });
  window.open(`/model-hamiltonian-builder/?${params.toString()}`, 'pyscf-agent-model-builder');
}

function resetModelPreview(message = 'No model input yet.') {
  document.getElementById('model-hamiltonian-structure-preview').innerHTML = `<div class="model-structure-empty">${escapeHtml(message)}</div>`;
}

function renderModelHamiltonianStructurePreview(preview) {
  const target = document.getElementById('model-hamiltonian-structure-preview');
  if (!preview || typeof preview !== 'object') {
    resetModelPreview();
    return;
  }
  const sites = Array.isArray(preview.sites_preview) ? preview.sites_preview : [];
  const bonds = Array.isArray(preview.bonds_preview) ? preview.bonds_preview : [];
  if (!sites.length) {
    resetModelPreview('The input file does not contain drawable sites.');
    return;
  }
  const numericSites = sites.map((site) => ({
    id: site.id,
    x: Number.isFinite(Number(site.x)) ? Number(site.x) : 0,
    y: Number.isFinite(Number(site.y)) ? Number(site.y) : 0,
  }));
  const siteMap = new Map(numericSites.map((site) => [String(site.id), site]));
  const plotPoints = numericSites.map((site) => ({ x: site.x, y: site.y }));
  const cell = preview.cell && typeof preview.cell === 'object' ? preview.cell : null;
  if (preview.boundary === 'periodic' && cell && cell.a && cell.b) {
    const origin = cell.origin || { x: 0, y: 0 };
    const a = cell.a;
    const b = cell.b;
    plotPoints.push(
      { x: Number(origin.x) || 0, y: Number(origin.y) || 0 },
      { x: (Number(origin.x) || 0) + (Number(a.x) || 0), y: (Number(origin.y) || 0) + (Number(a.y) || 0) },
      { x: (Number(origin.x) || 0) + (Number(a.x) || 0) + (Number(b.x) || 0), y: (Number(origin.y) || 0) + (Number(a.y) || 0) + (Number(b.y) || 0) },
      { x: (Number(origin.x) || 0) + (Number(b.x) || 0), y: (Number(origin.y) || 0) + (Number(b.y) || 0) },
    );
  }
  bonds.filter((bond) => bond.periodic).forEach((bond) => {
    const source = siteMap.get(String(bond.source));
    const targetSite = siteMap.get(String(bond.target));
    const offset = bond.offset || { x: 0, y: 0 };
    const dx = Number(offset.x) || 0;
    const dy = Number(offset.y) || 0;
    if (source) {
      plotPoints.push({ x: source.x - dx, y: source.y - dy });
    }
    if (targetSite) {
      plotPoints.push({ x: targetSite.x + dx, y: targetSite.y + dy });
    }
  });
  const minX = Math.min(...plotPoints.map((point) => point.x));
  const maxX = Math.max(...plotPoints.map((point) => point.x));
  const minY = Math.min(...plotPoints.map((point) => point.y));
  const maxY = Math.max(...plotPoints.map((point) => point.y));
  const spanX = Math.max(maxX - minX, 1);
  const spanY = Math.max(maxY - minY, 1);
  const width = 720;
  const height = 420;
  const padding = 46;
  const scale = Math.min((width - padding * 2) / spanX, (height - padding * 2) / spanY);
  const xOffset = (width - spanX * scale) / 2;
  const yOffset = (height - spanY * scale) / 2;
  const project = (point) => ({
    x: xOffset + ((Number(point.x) || 0) - minX) * scale,
    y: height - (yOffset + ((Number(point.y) || 0) - minY) * scale),
  });
  const cellMarkup = (() => {
    if (preview.boundary !== 'periodic' || !cell || !cell.a || !cell.b) {
      return '';
    }
    const origin = cell.origin || { x: 0, y: 0 };
    const a = cell.a;
    const b = cell.b;
    const corners = [
      origin,
      { x: (Number(origin.x) || 0) + (Number(a.x) || 0), y: (Number(origin.y) || 0) + (Number(a.y) || 0) },
      { x: (Number(origin.x) || 0) + (Number(a.x) || 0) + (Number(b.x) || 0), y: (Number(origin.y) || 0) + (Number(a.y) || 0) + (Number(b.y) || 0) },
      { x: (Number(origin.x) || 0) + (Number(b.x) || 0), y: (Number(origin.y) || 0) + (Number(b.y) || 0) },
    ].map((point) => project(point));
    return `<polygon class="cell-boundary" points="${corners.map((point) => `${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(' ')}" />`;
  })();
  const edgeMarkup = bonds.map((bond) => {
    const source = siteMap.get(String(bond.source));
    const targetSite = siteMap.get(String(bond.target));
    if (!source || !targetSite) {
      return '';
    }
    const p1 = project(source);
    const p2 = project(targetSite);
    if (bond.periodic) {
      const offset = bond.offset || { x: 0, y: 0 };
      const targetImage = { x: targetSite.x + (Number(offset.x) || 0), y: targetSite.y + (Number(offset.y) || 0) };
      const dx = targetImage.x - source.x;
      const dy = targetImage.y - source.y;
      const length = Math.max(Math.hypot(dx, dy), 1e-8);
      const unit = { x: dx / length, y: dy / length };
      const halfLength = Math.min(38, Math.max(20, length * scale * 0.42));
      const sourceEnd = { x: p1.x + unit.x * halfLength, y: p1.y - unit.y * halfLength };
      const targetEnd = { x: p2.x - unit.x * halfLength, y: p2.y + unit.y * halfLength };
      return [
        `<line x1="${p1.x.toFixed(2)}" y1="${p1.y.toFixed(2)}" x2="${sourceEnd.x.toFixed(2)}" y2="${sourceEnd.y.toFixed(2)}" stroke="#475569" stroke-width="3.2" stroke-linecap="round" />`,
        `<line x1="${p2.x.toFixed(2)}" y1="${p2.y.toFixed(2)}" x2="${targetEnd.x.toFixed(2)}" y2="${targetEnd.y.toFixed(2)}" stroke="#475569" stroke-width="3.2" stroke-linecap="round" />`,
      ].join('');
    }
    return `<line x1="${p1.x.toFixed(2)}" y1="${p1.y.toFixed(2)}" x2="${p2.x.toFixed(2)}" y2="${p2.y.toFixed(2)}" stroke="#475569" stroke-width="3.2" stroke-linecap="round" />`;
  }).join('');
  const ghostMarkup = bonds.filter((bond) => bond.periodic).map((bond) => {
    const source = siteMap.get(String(bond.source));
    const targetSite = siteMap.get(String(bond.target));
    if (!source || !targetSite) {
      return '';
    }
    const offset = bond.offset || { x: 0, y: 0 };
    const dx = Number(offset.x) || 0;
    const dy = Number(offset.y) || 0;
    const sourceMirror = project({ x: source.x - dx, y: source.y - dy });
    const targetMirror = project({ x: targetSite.x + dx, y: targetSite.y + dy });
    const sourcePoint = project(source);
    const targetPoint = project(targetSite);
    const ghostSite = (point, label) => [
      `<circle cx="${point.x.toFixed(2)}" cy="${point.y.toFixed(2)}" r="12" class="ghost-site" />`,
      `<text x="${point.x.toFixed(2)}" y="${(point.y + 4).toFixed(2)}" class="ghost-site-label">${escapeHtml(label)}</text>`,
    ].join('');
    return [
      `<line x1="${sourcePoint.x.toFixed(2)}" y1="${sourcePoint.y.toFixed(2)}" x2="${targetMirror.x.toFixed(2)}" y2="${targetMirror.y.toFixed(2)}" class="ghost-bond" />`,
      `<line x1="${sourceMirror.x.toFixed(2)}" y1="${sourceMirror.y.toFixed(2)}" x2="${targetPoint.x.toFixed(2)}" y2="${targetPoint.y.toFixed(2)}" class="ghost-bond" />`,
      ghostSite(sourceMirror, source.id),
      ghostSite(targetMirror, targetSite.id),
    ].join('');
  }).join('');
  const siteMarkup = numericSites.map((site) => {
    const p = project(site);
    return [
      `<circle cx="${p.x.toFixed(2)}" cy="${p.y.toFixed(2)}" r="17" fill="#ffffff" stroke="#334155" stroke-width="3" />`,
      `<text x="${p.x.toFixed(2)}" y="${(p.y + 5).toFixed(2)}" text-anchor="middle" font-size="14" font-weight="700" fill="#0f172a">${escapeHtml(site.id)}</text>`,
    ].join('');
  }).join('');
  const meta = [
    `model: ${preview.model || ''}`,
    `representation: ${preview.representation || 'finite_cluster'}`,
    `solver: ${preview.solver || ''}`,
    `dimension: ${preview.dimension || ''}D`,
    `preset: ${preview.preset || ''}`,
    `boundary: ${preview.boundary || ''}`,
    `nelec: ${Array.isArray(preview.nelec) ? `(${preview.nelec.join(', ')})` : ''}`,
    `electrons/cell: ${preview.occupation && preview.occupation.electrons_per_cell !== undefined ? preview.occupation.electrons_per_cell : ''}`,
    `k-mesh: ${preview.reciprocal_space && Array.isArray(preview.reciprocal_space.kmesh) ? preview.reciprocal_space.kmesh.join(' x ') : ''}`,
    `sites: ${preview.site_count ?? 0}`,
    `bonds: ${preview.bond_count ?? 0}`,
  ].filter((item) => !item.endsWith(': '));
  target.innerHTML = [
    `<div class="model-structure-meta">${meta.map((item) => `<span class="model-structure-pill">${escapeHtml(item)}</span>`).join('')}</div>`,
    `<svg class="model-structure-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Model structure preview">`,
    '<g>',
    cellMarkup,
    ghostMarkup,
    edgeMarkup,
    siteMarkup,
    '</g>',
    '</svg>',
    preview.truncated ? '<div class="model-structure-truncated">Graph preview truncated to the first sites and bonds.</div>' : '',
  ].join('');
}

async function previewModelHamiltonianStructure() {
  const inputFile = document.getElementById('model-hamiltonian-input-file').value.trim() || syncModelInputFileFromStudySpec();
  if (!inputFile) {
    resetModelPreview('Save a builder input first.');
    return;
  }
  resetModelPreview('Reading model...');
  try {
    const payload = await postJson('/api/model-hamiltonian-preview', {
      model_hamiltonian_input_file: inputFile,
    });
    renderModelHamiltonianStructurePreview(payload.preview);
  } catch (error) {
    resetModelPreview(`Preview failed: ${error.message}`);
  }
}

function handleBuilderMessage(event) {
  if (event.origin !== window.location.origin || !event.data || event.data.type !== 'pyscf-agent:model-hamiltonian-input-saved') {
    return;
  }
  if (event.data.target && event.data.target !== 'study') {
    return;
  }
  if (typeof event.data.path === 'string' && event.data.path.trim()) {
    setModelInputFile(event.data.path.trim());
  }
  if (typeof event.data.work_dir === 'string' && event.data.work_dir.trim()) {
    const workDir = document.getElementById('work-dir');
    if (!workDir.disabled) {
      workDir.value = event.data.work_dir.trim();
    }
  }
  if (typeof event.data.run_id === 'string' && event.data.run_id.trim()) {
    currentRunId = event.data.run_id.trim();
  }
  updateEffectiveWorkDir();
  appendPlannerMessage('system', 'Builder input linked to the planner.');
  previewModelHamiltonianStructure();
}

function renderPlanSummary(plan) {
  const items = [
    { label: 'Cases', value: plan && plan.cases ? plan.cases.length : 0, className: '' },
    { label: 'Objective', value: plan ? plan.objective : '-', className: 'wide' },
  ];
  document.getElementById('plan-summary').innerHTML = items.map((item) => (
    `<div class="metric ${escapeHtml(item.className)}"><strong>${escapeHtml(item.value)}</strong><span>${escapeHtml(item.label)}</span></div>`
  )).join('');
}

function costEstimateRequiresApproval(estimate) {
  return Boolean(estimate && estimate.approval_required && !estimate.approved);
}

function reviewGateCostDetails(estimate, caseIds = []) {
  if (!estimate || !estimate.approval_required) {
    return null;
  }
  const selectedIds = new Set((Array.isArray(caseIds) ? caseIds : []).map((value) => String(value)));
  const allCases = Array.isArray(estimate.cases) ? estimate.cases : [];
  const cases = selectedIds.size
    ? allCases.filter((item) => item && selectedIds.has(String(item.case_id)))
    : allCases;
  const details = [];
  cases.forEach((item) => {
    if (!item) return;
    const pieces = [];
    if (item.determinant_count !== null && typeof item.determinant_count !== 'undefined') {
      pieces.push(`determinants=${Number(item.determinant_count).toLocaleString()}`);
    }
    if (item.memory_mb !== null && typeof item.memory_mb !== 'undefined') {
      pieces.push(`working memory~${Number(item.memory_mb).toFixed(1)} MB`);
    }
    if (pieces.length) {
      details.push(`${item.case_id || 'case'}: ${pieces.join(', ')}`);
    }
  });
  const totals = estimate.totals || {};
  if (!details.length && totals.work_units !== null && typeof totals.work_units !== 'undefined') {
    details.push(`combined work=${Number(totals.work_units).toLocaleString()}`);
  }
  if (!details.length && totals.peak_memory_mb !== null && typeof totals.peak_memory_mb !== 'undefined') {
    details.push(`peak memory~${Number(totals.peak_memory_mb).toFixed(1)} MB`);
  }
  return {
    summary: details.join(' | ') || 'Dimension-based cost estimate is available for this calculation.',
    reason: estimate.review_reason || '',
  };
}

function planCostReviewGate(plan, estimate = null, planKind = 'static') {
  const cost = estimate || (plan && plan.cost_estimate);
  if (!costEstimateRequiresApproval(cost)) {
    return null;
  }
  const totals = cost.totals || {};
  const items = (Array.isArray(cost.cases) ? cost.cases : [])
    .filter((item) => item && (item.determinant_count !== null && typeof item.determinant_count !== 'undefined' || item.memory_mb !== null && typeof item.memory_mb !== 'undefined'))
    .slice(0, 8)
    .map((item) => {
      const detail = [];
      if (item.determinant_count !== null && typeof item.determinant_count !== 'undefined') {
        detail.push(`determinants=${Number(item.determinant_count).toLocaleString()}`);
      }
      if (item.memory_mb !== null && typeof item.memory_mb !== 'undefined') {
        detail.push(`working memory~${Number(item.memory_mb).toFixed(1)} MB`);
      }
      return {
        title: `${item.case_id || 'case'} · ${(item.method || 'method').toUpperCase()}`,
        detail: detail.join(', '),
        status: 'cost review',
      };
    });
  const totalDetail = [];
  if (totals.work_units !== null && typeof totals.work_units !== 'undefined') {
    totalDetail.push(`combined work=${Number(totals.work_units).toLocaleString()}`);
  }
  if (totals.peak_memory_mb !== null && typeof totals.peak_memory_mb !== 'undefined') {
    totalDetail.push(`peak memory~${Number(totals.peak_memory_mb).toFixed(1)} MB`);
  }
  const limitExceeded = Boolean(cost.resource_limit_exceeded || cost.approval_allowed === false);
  return {
    kind: 'cost_approval',
    plan_kind: planKind,
    severity: limitExceeded ? 'error' : 'warning',
    kicker: 'Resource Review',
    title: limitExceeded ? 'Select a larger memory profile' : 'Approve estimated computational cost',
    status: limitExceeded ? 'blocked' : 'approval required',
    message: cost.review_reason || 'This plan exceeds the configured resource-review threshold.',
    hint: limitExceeded
      ? `The selected Slurm profile cannot hold the estimated working set. ${totalDetail.join(', ') || 'Choose another resource profile'}, then rebuild the plan.`
      : `The estimate is a dimension-based planning proxy. ${totalDetail.join(', ') || 'Review the case estimates'} before submitting the calculation.`,
    items,
    actions: limitExceeded
      ? [{ id: 'acknowledge', label: 'Close', secondary: true }]
      : [
        { id: 'approve_cost_estimate', label: 'Approve Estimated Cost', secondary: false },
        { id: 'acknowledge', label: 'Acknowledge', secondary: true },
      ],
  };
}

const GRID_METRIC_NAMES = ['energy_per_site', 'mean_double_occupancy',
  'nearest_neighbor_spin_correlation', 'nearest_neighbor_charge_correlation',
  'staggered_magnetization', 'sublattice_charge_imbalance'];

function syncGridRefinementControls(spec) {
  const policy = spec.grid_refinement || {};
  const enabled = policy.enabled !== false && Array.isArray(policy.axes) && policy.axes.length > 0;
  document.getElementById('grid-refinement-enabled').checked = enabled;
  document.getElementById('grid-refinement-controls').classList.toggle('hidden', !enabled);
  const variables = (spec.case_design || {}).variables || spec.sweep || {};
  const candidates = Object.keys(variables).filter(name => !['nelec', 'solver', 'boundary'].includes(name)
    && Array.isArray(variables[name]) && variables[name].length >= 2
    && variables[name].every(value => typeof value === 'number' && Number.isFinite(value)));
  const axes = policy.axes || (candidates.length ? candidates.slice(0, 2) : ['U']);
  const choices = [...new Set([...candidates, ...axes,
    ...(candidates.length ? [] : ['U', 'V', 't', 'epsilon'])])]
    .filter(name => /^[A-Za-z_][A-Za-z0-9_]*$/.test(name));
  [1, 2].forEach(index => {
    const axis = axes[index - 1] || '';
    const select = document.getElementById(`grid-axis-${index}`);
    select.innerHTML = (index === 2 ? '<option value="">None</option>' : '')
      + choices.map(name => `<option value="${name}">${name}</option>`).join('');
    select.value = axis;
    document.getElementById(`grid-spacing-${index}`).value = (policy.min_spacing || {})[axis] ?? 0.0001;
    document.getElementById(`grid-width-${index}`).value = (policy.max_interval || {})[axis] ?? '';
  });
  document.getElementById('grid-max-points').value = policy.max_new_points ?? 64;
  document.getElementById('grid-max-rounds').value = policy.max_rounds ?? 4;
  const metrics = policy.metrics || [{name: 'energy_per_site'}, {name: 'mean_double_occupancy'}];
  GRID_METRIC_NAMES.forEach(name => {
    const metric = metrics.find(item => item.name === name);
    document.getElementById(`grid-metric-${name}`).checked = Boolean(metric);
    document.getElementById(`grid-atol-${name}`).value = metric?.atol ?? 0.001;
  });
}

function gridControlNumber(id, {integer = false, zero = false} = {}) {
  const control = document.getElementById(id);
  const value = Number(control.value);
  if (control.value.trim() === '' || !Number.isFinite(value) || (zero ? value < 0 : value <= 0)
      || (integer && !Number.isInteger(value))) throw new Error('Enter a valid value for ' + id.replace(/^grid-/, '').replaceAll('-', ' '));
  return value;
}

function applyGridRefinementControls(spec) {
  if (normalizeSystemType(spec.system_type) !== 'model_hamiltonian'
      || !document.getElementById('grid-refinement-enabled').checked) {
    delete spec.grid_refinement;
    return spec;
  }
  const entries = [1, 2].map(index => ({index, axis: document.getElementById(`grid-axis-${index}`).value.trim()})).filter(item => item.axis);
  if (!entries.length || new Set(entries.map(item => item.axis)).size !== entries.length) throw new Error('Choose one or two distinct parameters to refine.');
  const policy = {enabled: true, axes: entries.map(item => item.axis),
    max_new_points: gridControlNumber('grid-max-points', {integer: true}),
    max_rounds: gridControlNumber('grid-max-rounds', {integer: true}), min_spacing: {}, max_interval: {}, metrics: []};
  entries.forEach(({axis, index}) => {
    policy.min_spacing[axis] = gridControlNumber(`grid-spacing-${index}`);
    if (document.getElementById(`grid-width-${index}`).value.trim()) policy.max_interval[axis] = gridControlNumber(`grid-width-${index}`);
  });
  GRID_METRIC_NAMES.filter(name => document.getElementById(`grid-metric-${name}`).checked).forEach(name => {
    const metric = {name, atol: gridControlNumber(`grid-atol-${name}`), rtol: 0, required: true};
    policy.metrics.push(metric);
  });
  if (!policy.metrics.length) throw new Error('Select at least one quantity for grid refinement.');
  spec.grid_refinement = policy;
  return spec;
}

function handleGridRefinementChange() {
  document.getElementById('grid-refinement-controls').classList.toggle('hidden', !document.getElementById('grid-refinement-enabled').checked);
  try {
    const spec = applyGridRefinementControls(parseStudySpec());
    document.getElementById('study-spec').value = prettyJson(spec);
    handleStudyModeChange();
  } catch (error) {
    setStatus('draft-status', error.message, 'error');
  }
}

function renderGridRefinementSummary(grid) {
  if (!grid || !grid.policy) return '';
  const labels = {satisfied: 'Sampling criteria satisfied', budget_exhausted: 'New-point budget reached',
    max_rounds_reached: 'Refinement level limit reached', min_spacing_reached: 'Minimum spacing reached',
    insufficient_evidence: 'Some regions lack reliable results', awaiting_results: 'Waiting for calculations',
    needs_resume: 'Results changed; resume to reassess the grid', running: 'Refining the grid',
    cost_approval_required: 'New points require resource review'};
  const region = item => item.bounds
    ? Object.entries(item.bounds).map(([axis, bounds]) => `${axis}: ${bounds.join(' → ')}`).join('; ')
    : `${item.axis}: ${(item.interval || []).join(' → ')}`;
  const decisions = (grid.decisions || []).map(item => `<li>${escapeHtml(region(item))}${item.midpoint == null ? '' : `; added ${escapeHtml(item.midpoint)}`} · ${item.point_count} points · ${escapeHtml(item.reason.replaceAll('_', ' '))}</li>`).join('');
  const unresolved = (grid.intervals || []).filter(item => item.status !== 'satisfied').map(item => `<li>${escapeHtml(region(item))} · ${escapeHtml(item.status.replaceAll('_', ' '))}</li>`).join('');
  return `<div class="analysis-box"><strong>Parameter grid: ${escapeHtml(labels[grid.status] || grid.status)}</strong>
    <p>${grid.seed_point_count} initial points + ${grid.new_point_count} added / ${grid.policy.max_new_points} allowed.</p>
    <details><summary>Why points were added</summary><ul>${decisions || '<li>No points added.</li>'}</ul></details>
    ${unresolved ? `<details><summary>Unresolved regions</summary><ul>${unresolved}</ul></details>` : ''}</div>`;
}
