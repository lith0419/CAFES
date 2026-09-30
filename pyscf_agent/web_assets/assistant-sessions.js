function collectFormSnapshot() {
  return {
    taskFamily: currentTaskFamily(),
    workDir: document.getElementById('work-dir').value,
    workDirLocked: document.getElementById('work-dir').disabled,
    modelHamiltonianInputFile: document.getElementById('model-hamiltonian-input-file').value,
    modelSolver: document.getElementById('model-solver').value,
    modelNroots: document.getElementById('model-nroots').value,
    modelDmetExecutionMode: document.getElementById('model-dmet-execution-mode').value,
    modelDmetFragmentMode: document.getElementById('model-dmet-fragment-mode').value,
    modelDmetImpurityShape: document.getElementById('model-dmet-impurity-shape').value,
    modelDmetImpuritySize: document.getElementById('model-dmet-impurity-size').value,
    modelDmetImpuritySolver: document.getElementById('model-dmet-impurity-solver').value,
    modelDmetCcsdBeta: document.getElementById('model-dmet-ccsd-beta').value,
    modelDmetBlock2Preset: document.getElementById('model-dmet-block2-preset').value,
    modelDmetBlock2Ordering: document.getElementById('model-dmet-block2-ordering').value,
    modelDmetReference: document.getElementById('model-dmet-reference').value,
    modelDmetReferenceDensity: document.getElementById('model-dmet-reference-density').value,
    modelDmetInteractingBath: document.getElementById('model-dmet-interacting-bath').value !== 'false',
    modelDmetMaxIterations: document.getElementById('model-dmet-max-iterations').value,
    modelDmetEnergyTolerance: document.getElementById('model-dmet-energy-tolerance').value,
    modelDmetDensityTolerance: document.getElementById('model-dmet-density-tolerance').value,
    periodicFormat: document.getElementById('periodic-format').value,
    periodicStructureText: document.getElementById('periodic-structure-text').value,
    periodicBasis: document.getElementById('periodic-basis').value,
    periodicPseudo: document.getElementById('periodic-pseudo').value,
    periodicMethod: document.getElementById('periodic-method').value,
    periodicXc: document.getElementById('periodic-xc').value,
    periodicCorrelationTreatment: document.getElementById('periodic-correlation-treatment').value,
    periodicGwBroadening: document.getElementById('periodic-gw-broadening').value,
    periodicGwFrequencyMaximum: document.getElementById('periodic-gw-frequency-maximum').value,
    periodicGwRealPoints: document.getElementById('periodic-gw-real-points').value,
    periodicGwImaginaryPoints: document.getElementById('periodic-gw-imaginary-points').value,
    periodicGwFiniteSizeCorrection: document.getElementById('periodic-gw-finite-size-correction').checked,
    periodicDmftLocalization: document.getElementById('periodic-dmft-localization').value,
    periodicDmftImpuritySolver: document.getElementById('periodic-dmft-impurity-solver').value,
    periodicDmftCorrelatedOrbitals: document.getElementById('periodic-dmft-correlated-orbitals').value,
    periodicDmftMinimalBasis: document.getElementById('periodic-dmft-minimal-basis').value,
    periodicDmftNbath: document.getElementById('periodic-dmft-nbath').value,
    periodicDmftMaxIterations: document.getElementById('periodic-dmft-max-iterations').value,
    periodicDmftConvergence: document.getElementById('periodic-dmft-convergence').value,
    periodicKmeshX: document.getElementById('periodic-kmesh-x').value,
    periodicKmeshY: document.getElementById('periodic-kmesh-y').value,
    periodicKmeshZ: document.getElementById('periodic-kmesh-z').value,
    periodicKpointScheme: document.getElementById('periodic-kpoint-scheme').value,
    periodicBandPathMode: document.getElementById('periodic-band-path-mode').value,
    periodicBandPath: document.getElementById('periodic-band-path').value,
    periodicBandSpecialPoints: document.getElementById('periodic-band-special-points').value,
    periodicBandPathNpoints: document.getElementById('periodic-band-path-npoints').value,
    periodicBandPathReferenceDistance: document.getElementById('periodic-band-path-reference-distance').value,
    periodicBandPathSymprec: document.getElementById('periodic-band-path-symprec').value,
    periodicKpointShiftX: document.getElementById('periodic-kpoint-shift-x').value,
    periodicKpointShiftY: document.getElementById('periodic-kpoint-shift-y').value,
    periodicKpointShiftZ: document.getElementById('periodic-kpoint-shift-z').value,
    periodicPrecision: document.getElementById('periodic-precision').value,
    periodicKeCutoff: document.getElementById('periodic-ke-cutoff').value,
    periodicFftMeshX: document.getElementById('periodic-fft-mesh-x').value,
    periodicFftMeshY: document.getElementById('periodic-fft-mesh-y').value,
    periodicFftMeshZ: document.getElementById('periodic-fft-mesh-z').value,
    periodicDensityFitting: document.getElementById('periodic-density-fitting').value,
    periodicDfAuxbasis: document.getElementById('periodic-df-auxbasis').value,
    periodicExxdiv: document.getElementById('periodic-exxdiv').value,
    periodicSmearing: document.getElementById('periodic-smearing').value,
    periodicSmearingSigma: document.getElementById('periodic-smearing-sigma').value,
    periodicSmearingFixSpin: document.getElementById('periodic-smearing-fix-spin').checked,
    periodicCharge: document.getElementById('periodic-charge').value,
    periodicSpin: document.getElementById('periodic-spin').value,
    periodicRestricted: document.getElementById('periodic-restricted').value,
    atom: document.getElementById('atom').value,
    basis: document.getElementById('basis').value,
    method: document.getElementById('method').value,
    xc: document.getElementById('xc').value,
    job: document.getElementById('job').value,
    charge: document.getElementById('charge').value,
    spin: document.getElementById('spin').value,
    restricted: document.getElementById('restricted').value,
    orbitalProcessingEnabled: document.getElementById('orbital-processing-enabled').checked,
    densityFittingEnabled: document.getElementById('density-fitting-auxbasis').value !== '__off__',
    densityFittingAuxbasis: document.getElementById('density-fitting-auxbasis').value,
    localizationMethod: document.getElementById('localization-method').value,
    localizationScope: document.getElementById('localization-scope').value,
    orbitalOrdering: document.getElementById('orbital-ordering').value,
    manualOrbitalOrder: document.getElementById('manual-orbital-order').value,
    stateTargetNroots: document.getElementById('state-target-nroots').value,
    stateAverageWeights: document.getElementById('state-average-weights').value,
    useNaturalOrbitals: document.getElementById('use-natural-orbitals').checked,
    activeSpaceEnabled: document.getElementById('active-space-enabled').checked,
    activeSpaceMethod: document.getElementById('active-space-method').value,
    activeSpaceSolver: document.getElementById('active-space-solver').value,
    avasTargets: document.getElementById('avas-targets').value,
    avasThreshold: document.getElementById('avas-threshold').value,
    ncas: document.getElementById('ncas').value,
    nelecas: document.getElementById('nelecas').value,
    activeOrbitals: document.getElementById('active-orbitals').value,
    activeSpaceApproved: document.getElementById('active-space-approved').checked,
    scNevpt2Enabled: document.getElementById('sc-nevpt2-enabled').checked,
    scNevpt2Root: document.getElementById('sc-nevpt2-root').value,
    scNevpt2DensityFit: document.getElementById('sc-nevpt2-density-fit').checked,
    outputs: selectedOutputValues(),
    request: document.getElementById('request').value,
  };
}

function blankFormSnapshot(taskFamily, inheritedWorkDir) {
  return {
    taskFamily: normalizeTaskFamily(taskFamily),
    workDir: inheritedWorkDir || '',
    workDirLocked: false,
    modelHamiltonianInputFile: '',
    modelSolver: 'fci',
    modelNroots: '1',
    modelDmetExecutionMode: '',
    modelDmetFragmentMode: '',
    modelDmetImpurityShape: '',
    modelDmetImpuritySize: '',
    modelDmetImpuritySolver: 'fci',
    modelDmetCcsdBeta: '',
    modelDmetBlock2Preset: 'balanced',
    modelDmetBlock2Ordering: 'canonical',
    modelDmetReference: 'unrestricted',
    modelDmetReferenceDensity: 'pm',
    modelDmetInteractingBath: true,
    modelDmetMaxIterations: '50',
    modelDmetEnergyTolerance: '1e-6',
    modelDmetDensityTolerance: '1e-4',
    periodicFormat: 'poscar',
    periodicStructureText: '',
    periodicBasis: DEFAULT_PERIODIC_BASIS,
    periodicPseudo: 'gth-pbe',
    periodicMethod: 'dft',
    periodicXc: 'pbe',
    periodicCorrelationTreatment: 'none',
    periodicGwBroadening: '0.1',
    periodicGwFrequencyMaximum: '18',
    periodicGwRealPoints: '181',
    periodicGwImaginaryPoints: '100',
    periodicGwFiniteSizeCorrection: true,
    periodicDmftLocalization: 'iao',
    periodicDmftImpuritySolver: 'cc',
    periodicDmftCorrelatedOrbitals: '',
    periodicDmftMinimalBasis: 'minao',
    periodicDmftNbath: '4',
    periodicDmftMaxIterations: '10',
    periodicDmftConvergence: '1e-3',
    periodicKmeshX: '1',
    periodicKmeshY: '1',
    periodicKmeshZ: '1',
    periodicKpointScheme: 'gamma_centered',
    periodicBandPathMode: 'auto',
    periodicBandPath: '',
    periodicBandSpecialPoints: '',
    periodicBandPathNpoints: '80',
    periodicBandPathReferenceDistance: '0.025',
    periodicBandPathSymprec: '1e-5',
    periodicKpointShiftX: '0',
    periodicKpointShiftY: '0',
    periodicKpointShiftZ: '0',
    periodicPrecision: '1e-8',
    periodicKeCutoff: '',
    periodicFftMeshX: '',
    periodicFftMeshY: '',
    periodicFftMeshZ: '',
    periodicDensityFitting: 'fft',
    periodicDfAuxbasis: '',
    periodicExxdiv: 'ewald',
    periodicSmearing: 'none',
    periodicSmearingSigma: '0.01',
    periodicSmearingFixSpin: false,
    periodicCharge: '0',
    periodicSpin: '0',
    periodicRestricted: 'auto',
    atom: '',
    basis: 'sto-3g',
    method: 'dft',
    xc: 'b3lyp',
    job: 'single_point',
    charge: '0',
    spin: '0',
    restricted: 'auto',
    orbitalProcessingEnabled: false,
    densityFittingEnabled: false,
    densityFittingAuxbasis: '__off__',
    localizationMethod: 'none',
    localizationScope: 'analysis',
    orbitalOrdering: 'canonical',
    manualOrbitalOrder: '',
    useNaturalOrbitals: false,
    activeSpaceEnabled: false,
    activeSpaceMethod: 'manual',
    avasTargets: '',
    avasThreshold: '0.2',
    ncas: '',
    nelecas: '',
    activeOrbitals: '',
    activeSpaceApproved: false,
    stateTargetNroots: '1',
    stateAverageWeights: '',
    scNevpt2Enabled: false,
    scNevpt2Root: '0',
    scNevpt2DensityFit: true,
    outputs: ['energy'],
    request: '',
  };
}

function applyFormSnapshot(snapshot) {
  const form = snapshot || blankFormSnapshot('molecular', '');
  document.getElementById('task-family').value = normalizeTaskFamily(form.taskFamily);
  document.getElementById('work-dir').value = form.workDir || '';
  setWorkDirLocked(Boolean(form.workDirLocked));
  document.getElementById('model-hamiltonian-input-file').value = form.modelHamiltonianInputFile || '';
  setSelectValue('model-solver', form.modelSolver || 'fci');
  document.getElementById('model-nroots').value = form.modelNroots || '1';
  setSelectValue(
    'model-dmet-execution-mode',
    ['translational', 'finite_graph'].includes(form.modelDmetExecutionMode)
      ? form.modelDmetExecutionMode
      : ''
  );
  setSelectValue(
    'model-dmet-fragment-mode',
    ['primitive_cell', 'cell_shape', 'site_count'].includes(form.modelDmetFragmentMode)
      ? form.modelDmetFragmentMode
      : ''
  );
  document.getElementById('model-dmet-impurity-shape').value = form.modelDmetImpurityShape || '';
  document.getElementById('model-dmet-impurity-size').value = form.modelDmetImpuritySize || '';
  setSelectValue('model-dmet-impurity-solver', form.modelDmetImpuritySolver || 'fci');
  document.getElementById('model-dmet-ccsd-beta').value = form.modelDmetCcsdBeta ?? '';
  setSelectValue('model-dmet-block2-preset', form.modelDmetBlock2Preset || 'balanced');
  setSelectValue('model-dmet-block2-ordering', form.modelDmetBlock2Ordering || 'canonical');
  setSelectValue('model-dmet-reference', form.modelDmetReference || 'unrestricted');
  setSelectValue('model-dmet-reference-density', form.modelDmetReferenceDensity || 'pm');
  document.getElementById('model-dmet-interacting-bath').value = String(form.modelDmetInteractingBath !== false);
  document.getElementById('model-dmet-max-iterations').value = form.modelDmetMaxIterations || '50';
  document.getElementById('model-dmet-energy-tolerance').value = form.modelDmetEnergyTolerance || '1e-6';
  document.getElementById('model-dmet-density-tolerance').value = form.modelDmetDensityTolerance || '1e-4';
  syncDmetFragmentControls();
  syncDmetImpuritySolverControls();
  setSelectValue('periodic-format', form.periodicFormat || 'poscar');
  document.getElementById('periodic-structure-text').value = form.periodicStructureText || '';
  setSelectValue('periodic-basis', form.periodicBasis || DEFAULT_PERIODIC_BASIS);
  setSelectValue('periodic-pseudo', form.periodicPseudo || 'gth-pbe');
  setSelectValue('periodic-method', form.periodicMethod || 'dft');
  setSelectValue('periodic-xc', form.periodicXc || 'pbe');
  setSelectValue('periodic-correlation-treatment', form.periodicCorrelationTreatment || 'none');
  document.getElementById('periodic-gw-broadening').value = form.periodicGwBroadening || '0.1';
  document.getElementById('periodic-gw-frequency-maximum').value = form.periodicGwFrequencyMaximum || '18';
  document.getElementById('periodic-gw-real-points').value = form.periodicGwRealPoints || '181';
  document.getElementById('periodic-gw-imaginary-points').value = form.periodicGwImaginaryPoints || '100';
  document.getElementById('periodic-gw-finite-size-correction').checked = form.periodicGwFiniteSizeCorrection !== false;
  setSelectValue('periodic-dmft-localization', form.periodicDmftLocalization || 'iao');
  setSelectValue('periodic-dmft-impurity-solver', form.periodicDmftImpuritySolver || 'cc');
  document.getElementById('periodic-dmft-correlated-orbitals').value = form.periodicDmftCorrelatedOrbitals || '';
  document.getElementById('periodic-dmft-minimal-basis').value = form.periodicDmftMinimalBasis || 'minao';
  document.getElementById('periodic-dmft-nbath').value = form.periodicDmftNbath || '4';
  document.getElementById('periodic-dmft-max-iterations').value = form.periodicDmftMaxIterations || '10';
  document.getElementById('periodic-dmft-convergence').value = form.periodicDmftConvergence || '1e-3';
  document.getElementById('periodic-kmesh-x').value = form.periodicKmeshX || '1';
  document.getElementById('periodic-kmesh-y').value = form.periodicKmeshY || '1';
  document.getElementById('periodic-kmesh-z').value = form.periodicKmeshZ || '1';
  setSelectValue('periodic-kpoint-scheme', form.periodicKpointScheme || 'gamma_centered');
  setSelectValue('periodic-band-path-mode', form.periodicBandPathMode || 'auto');
  document.getElementById('periodic-band-path').value = form.periodicBandPath || '';
  document.getElementById('periodic-band-special-points').value = form.periodicBandSpecialPoints || '';
  document.getElementById('periodic-band-path-npoints').value = form.periodicBandPathNpoints || '80';
  document.getElementById('periodic-band-path-reference-distance').value = form.periodicBandPathReferenceDistance || '0.025';
  document.getElementById('periodic-band-path-symprec').value = form.periodicBandPathSymprec || '1e-5';
  document.getElementById('periodic-kpoint-shift-x').value = form.periodicKpointShiftX || '0';
  document.getElementById('periodic-kpoint-shift-y').value = form.periodicKpointShiftY || '0';
  document.getElementById('periodic-kpoint-shift-z').value = form.periodicKpointShiftZ || '0';
  document.getElementById('periodic-precision').value = form.periodicPrecision || '1e-8';
  document.getElementById('periodic-ke-cutoff').value = form.periodicKeCutoff || '';
  document.getElementById('periodic-fft-mesh-x').value = form.periodicFftMeshX || '';
  document.getElementById('periodic-fft-mesh-y').value = form.periodicFftMeshY || '';
  document.getElementById('periodic-fft-mesh-z').value = form.periodicFftMeshZ || '';
  setSelectValue('periodic-density-fitting', form.periodicDensityFitting || 'fft');
  document.getElementById('periodic-df-auxbasis').value = form.periodicDfAuxbasis || '';
  setSelectValue('periodic-exxdiv', form.periodicExxdiv || 'ewald');
  setSelectValue('periodic-smearing', form.periodicSmearing || 'none');
  document.getElementById('periodic-smearing-sigma').value = form.periodicSmearingSigma || '0.01';
  document.getElementById('periodic-smearing-fix-spin').checked = Boolean(form.periodicSmearingFixSpin);
  document.getElementById('periodic-charge').value = form.periodicCharge || '0';
  document.getElementById('periodic-spin').value = form.periodicSpin || '0';
  document.getElementById('periodic-restricted').value = form.periodicRestricted || 'auto';
  document.getElementById('atom').value = form.atom || '';
  setSelectValue('basis', form.basis || 'sto-3g');
  orderedAuxbasisOptionsForBasis(document.getElementById('basis').value);
  setSelectValue('method', form.method || 'dft');
  setSelectValue('xc', form.xc || 'b3lyp');
  setSelectValue('job', form.job || 'single_point');
  document.getElementById('charge').value = form.charge || '0';
  document.getElementById('spin').value = form.spin || '0';
  document.getElementById('restricted').value = form.restricted || 'auto';
  document.getElementById('orbital-processing-enabled').checked = Boolean(form.orbitalProcessingEnabled);
  document.getElementById('density-fitting-auxbasis').value = densityFittingSelectValue(form.densityFittingEnabled, form.densityFittingAuxbasis);
  setSelectValue('localization-method', form.localizationMethod || 'none');
  setSelectValue('localization-scope', form.localizationScope || 'analysis');
  setSelectValue('orbital-ordering', form.orbitalOrdering || 'canonical');
  document.getElementById('manual-orbital-order').value = form.manualOrbitalOrder || '';
  document.getElementById('state-target-nroots').value = form.stateTargetNroots || form.dmrgNroots || '1';
  document.getElementById('state-average-weights').value = form.stateAverageWeights || form.dmrgStateAverageWeights || '';
  document.getElementById('use-natural-orbitals').checked = Boolean(form.useNaturalOrbitals);
  document.getElementById('active-space-enabled').checked = Boolean(form.activeSpaceEnabled);
  setSelectValue('active-space-method', form.activeSpaceMethod || 'manual');
  setSelectValue('active-space-solver', form.activeSpaceSolver || 'fci');
  document.getElementById('avas-targets').value = form.avasTargets || '';
  document.getElementById('avas-threshold').value = form.avasThreshold || '0.2';
  document.getElementById('ncas').value = form.ncas || '';
  document.getElementById('nelecas').value = form.nelecas || '';
  document.getElementById('active-orbitals').value = form.activeOrbitals || '';
  document.getElementById('active-space-approved').checked = Boolean(form.activeSpaceApproved);
  syncActiveSpaceMethodControls();
  document.getElementById('sc-nevpt2-enabled').checked = Boolean(form.scNevpt2Enabled);
  document.getElementById('sc-nevpt2-root').value = form.scNevpt2Root || '0';
  document.getElementById('sc-nevpt2-density-fit').checked = typeof form.scNevpt2DensityFit === 'undefined' ? true : Boolean(form.scNevpt2DensityFit);
  const outputs = Array.isArray(form.outputs) && form.outputs.length ? form.outputs : ['energy'];
  document.querySelectorAll('input[name="outputs"]').forEach((item) => {
    item.checked = outputs.includes(item.value);
  });
  document.getElementById('request').value = form.request || '';
  syncTaskFamilyControls();
  syncMethodControls();
  syncOrbitalProcessingControls();
  syncPeriodicMethodControls();
  syncPeriodicCorrelationControls();
  syncPeriodicNumericalControls();
  syncPeriodicBandPathControls();
  syncActiveSpaceMethodControls();
  updateActiveSpaceSummary();
  syncPostCasControls();
  updateMolecularStructurePreview();
}

function makeTaskSessionId() {
  return AgentUI.makeSessionId('task');
}

function activeTaskSession() {
  return AgentUI.findSession(taskSessions, activeTaskSessionId);
}

function taskFamilyLabel(taskFamily) {
  if (taskFamily === 'model_hamiltonian') return t('taskSessionModelHamiltonian');
  if (taskFamily === 'periodic') return t('taskSessionPeriodic');
  return t('taskSessionMolecular');
}

function taskStatusLabel(status) {
  if (status === 'ready') return t('taskStatusReady');
  if (status === 'changed') return t('taskStatusChanged');
  if (status === 'running') return t('taskStatusRunning');
  if (status === 'cancelled') return t('taskStatusCancelled');
  if (status === 'completed') return t('taskStatusCompleted');
  if (status === 'failed') return t('taskStatusFailed');
  return t('taskStatusDraft');
}

function deriveTaskSessionLabel(session) {
  const form = session.formData || {};
  const family = form.taskFamily || session.taskFamily || 'molecular';
  const base = taskFamilyLabel(family);
  if (family === 'model_hamiltonian') {
    const solver = form.modelSolver ? String(form.modelSolver).toUpperCase() : 'FCI';
    return `${base} / ${solver}`;
  }
  if (family === 'periodic') {
    const method = form.periodicMethod ? String(form.periodicMethod).toUpperCase() : '';
    const basis = form.periodicBasis || '';
    const detail = [method, basis].filter(Boolean).join(' / ');
    return detail ? `${base} / ${detail}` : base;
  }
  const method = form.method ? String(form.method).toUpperCase() : '';
  const basis = form.basis || '';
  const detail = [method, basis].filter(Boolean).join(' / ');
  return detail ? `${base} / ${detail}` : base;
}

function renderTaskSessionSelector() {
  const select = document.getElementById('task-session-select');
  if (!select) {
    return;
  }
  select.innerHTML = AgentUI.sessionOptions(
    taskSessions, activeTaskSessionId, deriveTaskSessionLabel, taskStatusLabel);
  const selectedOption = select.options[select.selectedIndex];
  const fullLabel = selectedOption ? selectedOption.textContent : '';
  select.title = fullLabel;
  const detail = document.getElementById('task-session-effective');
  if (detail) {
    detail.textContent = fullLabel;
  }
}

function updateActiveTaskSessionStatus(status) {
  const session = activeTaskSession();
  if (!session) {
    return;
  }
  session.status = status || 'draft';
  session.updatedAt = Date.now();
  renderTaskSessionSelector();
}

function snapshotCurrentTaskSession(overrides = {}) {
  const session = activeTaskSession();
  if (!session || isRestoringTaskSession) {
    return;
  }
  const formData = collectFormSnapshot();
  if (overrides.taskFamily) {
    formData.taskFamily = overrides.taskFamily;
  }
  session.taskFamily = formData.taskFamily;
  session.formData = formData;
  session.conversationHistory = conversationHistory.slice();
  session.currentRunId = currentRunId;
  session.pendingStructuredRequest = { ...(pendingStructuredRequest || {}) };
  session.pendingApproval = pendingApproval ? JSON.parse(JSON.stringify(pendingApproval)) : null;
  session.pendingExecutionRequest = pendingExecutionRequest;
  session.lastPrepareFingerprint = lastPrepareFingerprint;
  session.lastPreparedPayload = lastPreparedPayload ? JSON.parse(JSON.stringify(lastPreparedPayload)) : null;
  session.lastPreparationMeta = lastPreparationMeta ? JSON.parse(JSON.stringify(lastPreparationMeta)) : null;
  session.lastExecutionPayload = lastExecutionPayload ? JSON.parse(JSON.stringify(lastExecutionPayload)) : null;
  session.lastExecutedRequestText = lastExecutedRequestText;
  session.activeRunHandle = activeRunHandle ? JSON.parse(JSON.stringify(activeRunHandle)) : null;
  session.activeRunTarget = activeRunTarget;
  session.activeRunPreparedRequestText = activeRunPreparedRequestText;
  session.activeRunLastState = activeRunLastState;
  session.lifecycle = currentTaskLifecycle ? JSON.parse(JSON.stringify(currentTaskLifecycle)) : null;
  session.lastModelHamiltonianPreview = lastModelHamiltonianPreview ? JSON.parse(JSON.stringify(lastModelHamiltonianPreview)) : null;
  session.lastPeriodicStructurePreview = lastPeriodicStructurePreview ? JSON.parse(JSON.stringify(lastPeriodicStructurePreview)) : null;
  const preservesExplicitStatus = ['running', 'cancelled', 'completed', 'failed', 'changed'].includes(session.status);
  if (!preservesExplicitStatus) {
    if (pendingExecutionRequest) {
      session.status = 'ready';
    } else if (lastExecutionPayload) {
      session.status = lastExecutionPayload.execution_status === 'succeeded' ? 'completed' : 'failed';
    } else {
      session.status = session.status || 'draft';
    }
  }
  session.updatedAt = Date.now();
  renderTaskSessionSelector();
}

function restoreTaskSession(session) {
  if (!session) {
    return;
  }
  isRestoringTaskSession = true;
  try {
    conversationHistory = (session.conversationHistory || []).slice();
    currentRunId = session.currentRunId || '';
    pendingStructuredRequest = { ...(session.pendingStructuredRequest || {}) };
    pendingApproval = session.pendingApproval ? JSON.parse(JSON.stringify(session.pendingApproval)) : null;
    pendingExecutionRequest = session.pendingExecutionRequest || '';
    lastPrepareFingerprint = session.lastPrepareFingerprint || '';
    lastPreparedPayload = session.lastPreparedPayload ? JSON.parse(JSON.stringify(session.lastPreparedPayload)) : null;
    lastPreparationMeta = session.lastPreparationMeta ? JSON.parse(JSON.stringify(session.lastPreparationMeta)) : null;
    lastExecutionPayload = session.lastExecutionPayload ? JSON.parse(JSON.stringify(session.lastExecutionPayload)) : null;
    lastExecutedRequestText = session.lastExecutedRequestText || '';
    activeRunHandle = session.activeRunHandle ? JSON.parse(JSON.stringify(session.activeRunHandle)) : null;
    activeRunTarget = session.activeRunTarget || '';
    activeRunPreparedRequestText = session.activeRunPreparedRequestText || '';
    activeRunLastState = session.activeRunLastState || '';
    currentTaskLifecycle = session.lifecycle ? JSON.parse(JSON.stringify(session.lifecycle)) : null;
    lastModelHamiltonianPreview = session.lastModelHamiltonianPreview ? JSON.parse(JSON.stringify(session.lastModelHamiltonianPreview)) : null;
    lastPeriodicStructurePreview = session.lastPeriodicStructurePreview ? JSON.parse(JSON.stringify(session.lastPeriodicStructurePreview)) : null;
    applyFormSnapshot(session.formData || blankFormSnapshot(session.taskFamily || 'molecular', ''));
    clearPanels({ preserveState: true });
    const restoredPendingExecutionRequest = pendingExecutionRequest;
    renderConversation(conversationHistory, lastExecutionPayload ? lastExecutionPayload.messages || [] : []);
    renderModelHamiltonianStructurePreview(lastModelHamiltonianPreview);
    renderPeriodicStructurePreview(lastPeriodicStructurePreview);
    if (lastExecutionPayload) {
      renderSummary(lastExecutionPayload);
    } else if (lastPreparedPayload) {
      renderPreparationState(lastPreparedPayload);
    }
    if (activeRunHandle && activeRunLastState) {
      renderStatus(activeRunLastState);
    }
    pendingExecutionRequest = restoredPendingExecutionRequest;
    if (pendingApproval) {
      renderApprovalPanel(pendingApproval);
    } else {
      clearApprovalPanel();
    }
    if (pendingExecutionRequest) {
      renderComputeConfirmPanel(t('computeReady'), pendingExecutionRequest);
    } else if (!pendingApproval) {
      clearComputeConfirmPanel();
    }
    setAnalyzeState(false);
    updateStatusButtonState();
  } finally {
    isRestoringTaskSession = false;
  }
  renderTaskSessionSelector();
}

function createTaskSession(taskFamily = 'molecular', options = {}) {
  if (options.snapshotExisting !== false) {
    snapshotCurrentTaskSession(options.previousTaskFamily ? { taskFamily: options.previousTaskFamily } : {});
  }
  const inheritedWorkDir = options.inheritWorkDir === false ? '' : builderWorkDirValue();
  const formData = options.formData || blankFormSnapshot(taskFamily, inheritedWorkDir);
  const session = {
    id: makeTaskSessionId(),
    taskFamily: formData.taskFamily || taskFamily,
    status: 'draft',
    createdAt: Date.now(),
    updatedAt: Date.now(),
    formData,
    conversationHistory: [],
    currentRunId: '',
    pendingStructuredRequest: {},
    pendingApproval: null,
    pendingExecutionRequest: '',
    lastPrepareFingerprint: '',
    lastPreparedPayload: null,
    lastPreparationMeta: null,
    lastExecutionPayload: null,
    lastExecutedRequestText: '',
    activeRunHandle: null,
    activeRunTarget: '',
    activeRunPreparedRequestText: '',
    activeRunLastState: '',
    lifecycle: null,
    lastModelHamiltonianPreview: null,
    lastPeriodicStructurePreview: null,
  };
  taskSessions.push(session);
  activeTaskSessionId = session.id;
  restoreTaskSession(session);
  if (!options.silent) {
    conversationHistory = appendAssistantMessages(conversationHistory, [
      { role: 'system', content: `${t('taskStarted')}: ${taskFamilyLabel(session.taskFamily)}` },
    ]);
    renderConversation(conversationHistory);
    snapshotCurrentTaskSession();
  }
  renderTaskSessionSelector();
  return session;
}

function switchTaskSession(sessionId) {
  if (sessionId === activeTaskSessionId) {
    return;
  }
  snapshotCurrentTaskSession();
  const session = taskSessions.find((item) => item.id === sessionId);
  if (!session) {
    return;
  }
  activeTaskSessionId = session.id;
  restoreTaskSession(session);
}

function currentTaskHasMeaningfulState() {
  return Boolean(
    conversationHistory.length ||
    lastExecutionPayload ||
    activeRunHandle ||
    lastPreparedPayload ||
    pendingExecutionRequest ||
    pendingApproval ||
    document.getElementById('request').value.trim() ||
    document.getElementById('model-hamiltonian-input-file').value.trim() ||
    document.getElementById('periodic-structure-text').value.trim()
  );
}

function makeRunId() {
  return AgentUI.makeRunId();
}

function workDirValue() {
  return document.getElementById('work-dir').value.trim();
}

function effectiveWorkDir() {
  return workDirValue() || DEFAULT_WORK_DIR;
}

function builderWorkDirValue() {
  const value = effectiveWorkDir();
  const runId = currentRunId || '';
  if (runId && value.endsWith(`/${runId}`)) {
    return value.slice(0, -runId.length - 1);
  }
  return value;
}

function directoryFromPath(pathValue) {
  if (typeof pathValue !== 'string' || !pathValue.trim()) {
    return '';
  }
  const normalized = pathValue.trim().replaceAll('\\', '/');
  const index = normalized.lastIndexOf('/');
  return index > 0 ? normalized.slice(0, index) : '';
}

function updateEffectiveWorkDirLabel() {
  const label = document.getElementById('work-dir-effective');
  if (!label) {
    return;
  }
  const prefix = document.getElementById('work-dir').disabled ? t('workDirLockedHint') : t('effectiveWorkDirLabel');
  label.textContent = `${prefix}: ${effectiveWorkDir()}`;
}

function setWorkDirLocked(isLocked) {
  const input = document.getElementById('work-dir');
  input.disabled = Boolean(isLocked);
  input.classList.toggle('locked-input', Boolean(isLocked));
  updateEffectiveWorkDirLabel();
}

function ensureCurrentRunId() {
  if (!currentRunId) {
    currentRunId = makeRunId();
  }
  return currentRunId;
}

function beginTaskExecutionRun() {
  const workDir = builderWorkDirValue();
  currentRunId = makeRunId();
  const runDirectory = `${workDir.replace(/\/$/, '')}/${currentRunId}`;
  document.getElementById('work-dir').value = runDirectory;
  setWorkDirLocked(true);
  return { runId: currentRunId, workDir };
}
