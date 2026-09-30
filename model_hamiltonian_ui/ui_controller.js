function spinMultiplicityFromElectronCounts() {
  const nalpha = clampInt(numberValue('nalpha', 0), 0, 999);
  const nbeta = clampInt(numberValue('nbeta', 0), 0, 999);
  return Math.abs(nalpha - nbeta) + 1;
}

function syncMultiplicityFromElectronCounts() {
  const input = document.getElementById('spin-multiplicity');
  const validation = document.getElementById('spin-multiplicity-validation');
  const multiplicity = spinMultiplicityFromElectronCounts();
  input.value = String(multiplicity);
  input.setCustomValidity('');
  input.removeAttribute('aria-invalid');
  validation.textContent = '';
  return multiplicity;
}

function syncElectronCountsFromMultiplicity() {
  const input = document.getElementById('spin-multiplicity');
  const validation = document.getElementById('spin-multiplicity-validation');
  const nalpha = clampInt(numberValue('nalpha', 0), 0, 999);
  const nbeta = clampInt(numberValue('nbeta', 0), 0, 999);
  const electronCount = nalpha + nbeta;
  const multiplicity = clampInt(numberValue('spin-multiplicity', 1), 1, 1999);
  const spinDifference = multiplicity - 1;
  let message = '';
  if (spinDifference > electronCount) {
    message = 'Multiplicity exceeds the available electron count.';
  } else if ((electronCount + spinDifference) % 2 !== 0) {
    message = 'Electron count and multiplicity must have compatible parity.';
  }
  input.setCustomValidity(message);
  validation.textContent = message;
  if (message) {
    input.setAttribute('aria-invalid', 'true');
    return false;
  }
  input.removeAttribute('aria-invalid');
  document.getElementById('nalpha').value = String((electronCount + spinDifference) / 2);
  document.getElementById('nbeta').value = String((electronCount - spinDifference) / 2);
  return true;
}

function updateGlobalsFromControls() {
  state.model = document.getElementById('model-type').value;
  state.dimension = clampInt(numberValue('dimension', 1), 1, 2);
  state.template1d = document.getElementById('template-1d').value;
  state.template2d = document.getElementById('template-2d').value;
  state.preset = state.dimension === 1 ? state.template1d : state.template2d;
  state.bondModulation = state.dimension === 1 ? document.getElementById('bond-modulation').value : 'none';
  state.boundary = document.getElementById('boundary').value;
  state.energyUnit = document.getElementById('energy-unit').value;
  state.globals.t = numberValue('global-t', -1);
  state.globals.delta = numberValue('ssh-delta', 0.2);
  state.globals.epsilon = numberValue('global-epsilon', 0);
  state.globals.U = numberValue('global-u', 4);
  state.globals.V = numberValue('global-v', 0);
  state.globals.phononOmega = numberValue('phonon-omega', 1);
  state.globals.electronPhononG = numberValue('electron-phonon-g', 0.5);
  state.globals.phononCutoff = clampInt(numberValue('phonon-cutoff', 4), 1, 64);
  state.nelec = [
    clampInt(numberValue('nalpha', 0), 0, 999),
    clampInt(numberValue('nbeta', 0), 0, 999),
  ];
  state.spinMultiplicity = Math.abs(state.nelec[0] - state.nelec[1]) + 1;
  state.occupation.electronsPerCell = Math.max(0, numberValue('electrons-per-cell', state.occupation.electronsPerCell));
  const kmeshAxisLimit = state.dimension === 1 ? MAX_BLOCH_KMESH_AXIS_1D : MAX_BLOCH_KMESH_AXIS_2D;
  state.reciprocal.kmesh = [
    clampInt(numberValue('kmesh-x', 100), 1, kmeshAxisLimit),
    clampInt(numberValue('kmesh-y', 30), 1, kmeshAxisLimit),
  ];
  state.reciprocal.scheme = document.getElementById('kpoint-scheme').value;
  state.reciprocal.shift = [numberValue('kshift-x', 0), numberValue('kshift-y', 0)];
  state.reciprocal.pointsPerSegment = clampInt(numberValue('path-points', 80), 2, 1001);
  state.reciprocal.dosSigma = document.getElementById('dos-auto').checked
    ? 0
    : Math.max(numberValue('dos-sigma', 0.05), 1e-6);
  state.reciprocal.dosPoints = clampInt(numberValue('dos-points', 401), 2, blochDosPointLimit());
  state.geometryCoupled = document.getElementById('geometry-coupled').checked;
  state.geometryCoupledV = document.getElementById('geometry-coupled-v').checked;
  state.geometry.r0 = numberValue('geo-r0', 1);
  state.geometry.hoppingFunction = document.getElementById('hopping-function').value;
  state.geometry.hoppingParameter = numberValue('hopping-parameter', 1.5);
  state.geometry.interactionFunction = document.getElementById('interaction-function').value;
  state.geometry.interactionParameter = numberValue('interaction-parameter', 1);
}

function updateControlsFromState() {
  document.getElementById('model-type').value = state.model;
  document.getElementById('dimension').value = String(state.dimension);
  document.getElementById('template-1d').value = state.template1d;
  document.getElementById('template-2d').value = state.template2d;
  document.getElementById('bond-modulation').value = state.bondModulation;
  document.getElementById('boundary').value = state.boundary;
  document.getElementById('energy-unit').value = state.energyUnit;
  document.getElementById('global-t').value = state.globals.t;
  document.getElementById('ssh-delta').value = state.globals.delta;
  document.getElementById('global-epsilon').value = state.globals.epsilon;
  document.getElementById('global-u').value = state.globals.U;
  document.getElementById('global-v').value = state.globals.V;
  document.getElementById('phonon-omega').value = state.globals.phononOmega;
  document.getElementById('electron-phonon-g').value = state.globals.electronPhononG;
  document.getElementById('phonon-cutoff').value = state.globals.phononCutoff;
  document.getElementById('nalpha').value = state.nelec[0];
  document.getElementById('nbeta').value = state.nelec[1];
  document.getElementById('spin-multiplicity').value = Math.abs(state.nelec[0] - state.nelec[1]) + 1;
  document.getElementById('electrons-per-cell').value = state.occupation.electronsPerCell;
  document.getElementById('kmesh-x').value = state.reciprocal.kmesh[0];
  document.getElementById('kmesh-y').value = state.reciprocal.kmesh[1];
  document.getElementById('kpoint-scheme').value = state.reciprocal.scheme;
  document.getElementById('kshift-x').value = state.reciprocal.shift[0];
  document.getElementById('kshift-y').value = state.reciprocal.shift[1];
  document.getElementById('path-points').value = state.reciprocal.pointsPerSegment;
  document.getElementById('dos-auto').checked = !(state.reciprocal.dosSigma > 0);
  document.getElementById('dos-sigma').value = state.reciprocal.dosSigma > 0 ? state.reciprocal.dosSigma : 0.05;
  document.getElementById('dos-points').value = state.reciprocal.dosPoints;
  document.getElementById('geometry-coupled').checked = state.geometryCoupled;
  document.getElementById('geometry-coupled-v').checked = state.geometryCoupledV;
  document.getElementById('geo-r0').value = state.geometry.r0;
  document.getElementById('hopping-function').value = state.geometry.hoppingFunction;
  document.getElementById('hopping-parameter').value = state.geometry.hoppingParameter;
  document.getElementById('interaction-function').value = state.geometry.interactionFunction;
  document.getElementById('interaction-parameter').value = state.geometry.interactionParameter;
  updateTemplateControls();
  updateEnergyUnitLabels();
}

function updateEnergyUnitLabels() {
  const unit = state.energyUnit || document.getElementById('energy-unit').value || 'a.u.';
  document.querySelectorAll('.energy-unit-label').forEach((item) => {
    item.textContent = unit;
  });
}

function updateTemplateControls() {
  if (!BLOCH_UI_ENABLED && state.representation !== 'finite_cluster') {
    state.representation = 'finite_cluster';
  }
  const representationControl = document.getElementById('representation-control');
  representationControl.hidden = !BLOCH_UI_ENABLED;
  representationControl.setAttribute('aria-hidden', String(!BLOCH_UI_ENABLED));
  const isOneDimensional = document.getElementById('dimension').value === '1';
  const isBloch = BLOCH_UI_ENABLED && isBlochRepresentation();
  const sizeXInput = document.getElementById('size-x');
  const sizeYInput = document.getElementById('size-y');
  const needsWidth = !isOneDimensional || document.getElementById('template-1d').value === 'quasi_1d';
  const showBlochWidth = isBloch && isOneDimensional && document.getElementById('template-1d').value === 'quasi_1d';
  const sizeYNeeded = isBloch ? showBlochWidth : needsWidth;
  const minimumSizeX = !isBloch && !isOneDimensional ? 1 : 2;
  const minimumSizeY = !isOneDimensional ? 1 : 2;
  const bondModulationInput = document.getElementById('bond-modulation');
  const bondModulationLabel = document.getElementById('bond-modulation-label');
  const sshDeltaInput = document.getElementById('ssh-delta');
  const sshDeltaLabel = document.getElementById('ssh-delta-label');
  const phononParameters = document.getElementById('phonon-parameters');
  const isSshBondModulation = isOneDimensional && bondModulationInput.value === 'ssh';
  const ringOption = Array.from(document.getElementById('template-1d').options).find((option) => option.value === 'ring');
  if (ringOption) {
    ringOption.disabled = isBloch;
  }
  document.querySelectorAll('[data-representation]').forEach((button) => {
    button.disabled = !BLOCH_UI_ENABLED && button.dataset.representation === 'bloch';
    button.classList.toggle('active', button.dataset.representation === state.representation);
  });
  document.getElementById('template-1d-row').classList.toggle('hidden', !isOneDimensional);
  document.getElementById('template-2d-row').classList.toggle('hidden', isOneDimensional);
  document.getElementById('boundary-row').classList.toggle('hidden', isBloch);
  document.getElementById('boundary').disabled = isBloch;
  if (isBloch) {
    document.getElementById('boundary').value = 'periodic';
    state.boundary = 'periodic';
  }
  document.getElementById('size-x-control').classList.toggle('hidden', isBloch);
  document.getElementById('size-y-control').classList.toggle('hidden', isBloch && !showBlochWidth);
  document.getElementById('size-controls').classList.toggle('hidden', isBloch && !showBlochWidth);
  document.getElementById('finite-electron-controls').classList.toggle('hidden', isBloch);
  const reciprocalSection = document.getElementById('bloch-reciprocal-section');
  reciprocalSection.classList.toggle('hidden', !isBloch);
  // A finite cluster may have periodic boundary conditions, but it is not a
  // Bloch problem with a Brillouin-zone mesh. Keep the UI and its inputs
  // unavailable outside the explicit Bloch representation.
  reciprocalSection.hidden = !isBloch;
  reciprocalSection.setAttribute('aria-hidden', String(!isBloch));
  reciprocalSection.querySelectorAll('input, select, button').forEach((control) => {
    control.disabled = !isBloch;
  });
  document.getElementById('kmesh-y-row').classList.toggle('hidden', isOneDimensional);
  const kmeshAxisLimit = isOneDimensional ? MAX_BLOCH_KMESH_AXIS_1D : MAX_BLOCH_KMESH_AXIS_2D;
  document.getElementById('kmesh-x').max = String(kmeshAxisLimit);
  document.getElementById('kmesh-y').max = String(kmeshAxisLimit);
  document.getElementById('kshift-y-row').classList.toggle('hidden', isOneDimensional);
  document.getElementById('bond-cell-offset-y-row').classList.toggle('hidden', isOneDimensional);
  sizeXInput.min = String(minimumSizeX);
  if (!isBloch && clampInt(numberValue('size-x', minimumSizeX), 1, 64) < minimumSizeX) {
    sizeXInput.value = String(minimumSizeX);
  }
  sizeYInput.disabled = !sizeYNeeded;
  sizeYInput.min = sizeYNeeded ? String(minimumSizeY) : '1';
  if (sizeYNeeded && clampInt(numberValue('size-y', minimumSizeY), 1, 16) < minimumSizeY) {
    sizeYInput.value = String(minimumSizeY);
  }
  document.getElementById('size-x-label').textContent = isOneDimensional ? 'Sites / unit cells' : 'Lx';
  document.getElementById('size-y-label').textContent = isOneDimensional ? 'Width' : 'Ly';
  bondModulationInput.disabled = !isOneDimensional;
  bondModulationLabel.classList.toggle('hidden', !isOneDimensional);
  sshDeltaLabel.classList.toggle('hidden', !isOneDimensional);
  sshDeltaInput.disabled = !isSshBondModulation;
  sshDeltaLabel.classList.toggle('disabled-control', !isSshBondModulation);
  if (!isOneDimensional) {
    bondModulationInput.value = 'none';
    state.bondModulation = 'none';
  }
  phononParameters.classList.toggle('hidden', !isHolsteinHubbardModel());
  const dosAutomatic = document.getElementById('dos-auto').checked;
  const dosPointLimit = blochDosPointLimit();
  document.getElementById('dos-points').max = String(dosPointLimit);
  if (state.reciprocal.dosPoints > dosPointLimit) {
    state.reciprocal.dosPoints = dosPointLimit;
    document.getElementById('dos-points').value = String(dosPointLimit);
  }
  document.getElementById('dos-sigma').disabled = dosAutomatic;
  document.getElementById('dos-sigma-row').classList.toggle('disabled-control', dosAutomatic);
}

function handleSitePointerDown(event, siteId) {
  event.stopPropagation();
  if (state.view.mode === 'pan') {
    handleCanvasPointerDown(event);
    return;
  }
  if (state.tool === 'add_bond') {
    if (state.pendingBondSource === null) {
      state.pendingBondSource = siteId;
      state.selected = { type: 'site', id: siteId };
      render();
      return;
    }
    addBond(state.pendingBondSource, siteId, 'custom');
    state.pendingBondSource = null;
    setTool('select');
    render();
    return;
  }

  selectItem('site', siteId);
  const site = siteById(siteId);
  const start = screenToModel(event.clientX, event.clientY);
  dragState = {
    siteId,
    startPointer: start,
    startX: site.x,
    startY: site.y,
  };
  canvas.setPointerCapture(event.pointerId);
}

function handleCanvasPointerDown(event) {
  if (state.view.mode === 'pan') {
    const point = canvas.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const svgPoint = point.matrixTransform(canvas.getScreenCTM().inverse());
    panState = {
      startX: svgPoint.x,
      startY: svgPoint.y,
      panX: state.view.panX,
      panY: state.view.panY,
    };
    canvas.setPointerCapture(event.pointerId);
    return;
  }
  if (event.target !== canvas && !event.target.classList.contains('canvas-bg')) {
    return;
  }
  const point = screenToModel(event.clientX, event.clientY);
  if (state.tool === 'add_site') {
    addSite(point.x, point.y);
    setTool('select');
    return;
  }
  state.selected = null;
  state.pendingBondSource = null;
  render();
}

function handleCanvasPointerMove(event) {
  if (panState) {
    const point = canvas.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const svgPoint = point.matrixTransform(canvas.getScreenCTM().inverse());
    state.view.panX = panState.panX + (svgPoint.x - panState.startX);
    state.view.panY = panState.panY + (svgPoint.y - panState.startY);
    render();
    return;
  }
  if (!dragState) {
    return;
  }
  const site = siteById(dragState.siteId);
  if (!site) {
    return;
  }
  const point = screenToModel(event.clientX, event.clientY);
  site.x = dragState.startX + (point.x - dragState.startPointer.x);
  site.y = dragState.startY + (point.y - dragState.startPointer.y);
  render();
}

function handleCanvasPointerUp(event) {
  if (panState) {
    panState = null;
    try {
      canvas.releasePointerCapture(event.pointerId);
    } catch (error) {
      // Pointer may already be released by the browser.
    }
    return;
  }
  if (dragState) {
    dragState = null;
    try {
      canvas.releasePointerCapture(event.pointerId);
    } catch (error) {
      // Pointer may already be released by the browser.
    }
  }
}

function updateSelectedSiteFromInspector() {
  if (!state.selected || state.selected.type !== 'site') {
    return;
  }
  const site = siteById(state.selected.id);
  if (!site) {
    return;
  }
  site.x = numberValue('site-x', site.x);
  site.y = numberValue('site-y', site.y);
  site.epsilon = numberValue('site-epsilon', site.epsilon);
  site.U = numberValue('site-u', site.U);
  site.phononOmega = numberValue('site-phonon-omega', site.phononOmega);
  site.electronPhononG = numberValue('site-electron-phonon-g', site.electronPhononG);
  renderGhostLattice();
  renderCellBoundary();
  renderBonds();
  renderSites();
  updateCanvasText();
  updatePreview();
}

function updateSelectedBondFromInspector() {
  if (!state.selected || state.selected.type !== 'bond') {
    return;
  }
  const bond = bondById(state.selected.id);
  if (!bond) {
    return;
  }
  bond.t = numberValue('bond-t', bond.t);
  bond.V = numberValue('bond-v', bond.V);
  bond.kind = 'custom';
  if (isBlochRepresentation()) {
    bond.cellOffset = [
      Math.round(numberValue('bond-cell-offset-x', 0)),
      ...(state.dimension === 2 ? [Math.round(numberValue('bond-cell-offset-y', 0))] : []),
    ];
    bond.addHermitianConjugate = document.getElementById('bond-add-hc').checked;
    refreshBondPhysicalOffset(bond);
  }
  renderBonds();
  updateCanvasText();
  updatePreview();
}

function setTab(tab) {
  state.activeTab = tab;
  document.querySelectorAll('.tab-button').forEach((button) => button.classList.remove('active'));
  document.getElementById(`tab-${tab}`).classList.add('active');
  updatePreview();
}

function builderWorkContext() {
  const params = new URLSearchParams(window.location.search);
  return {
    work_dir: params.get('work_dir') || '',
    run_id: params.get('run_id') || '',
    target: params.get('target') || 'agent',
  };
}

function targetWindowName(target = builderWorkContext().target) {
  return target === 'study' ? 'pyscf-computational-study-agent' : 'pyscf-agent-web-ui';
}

function targetUrl(target = builderWorkContext().target) {
  return target === 'study' ? '/computational-study/' : '/';
}

function openTargetWindow(target) {
  const targetWindow = window.open('', targetWindowName(target));
  if (!targetWindow) {
    return null;
  }
  const desiredUrl = new URL(targetUrl(target), window.location.origin);
  try {
    const currentUrl = targetWindow.location.href === 'about:blank'
      ? null
      : new URL(targetWindow.location.href);
    if (!currentUrl || currentUrl.origin !== desiredUrl.origin || currentUrl.pathname !== desiredUrl.pathname) {
      targetWindow.location.href = desiredUrl.href;
    }
  } catch (_error) {
    targetWindow.location.href = desiredUrl.href;
  }
  return targetWindow;
}

async function savePythonInputFile() {
  const button = document.getElementById('save-python');
  const status = document.getElementById('save-status');
  if (!syncElectronCountsFromMultiplicity()) {
    document.getElementById('spin-multiplicity').reportValidity();
    return;
  }
  updateGlobalsFromControls();
  button.disabled = true;
  status.className = 'save-status';
  status.textContent = 'Saving...';
  try {
    const response = await fetch('/api/save-model-hamiltonian-input', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ python_input: pythonInputText(), ...builderWorkContext() }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || payload.status !== 'ok') {
      throw new Error(payload.error || `Save failed with HTTP ${response.status}`);
    }
    status.className = 'save-status ok';
    status.textContent = `Saved: ${payload.path}`;
    if (window.opener && !window.opener.closed) {
      window.opener.postMessage({
        type: 'pyscf-agent:model-hamiltonian-input-saved',
        target: payload.target || builderWorkContext().target || '',
        path: payload.path || '',
        work_dir: payload.work_dir || '',
        run_id: payload.run_id || '',
        run_dir: payload.run_dir || '',
        representation: state.representation,
        solver: state.representation === 'bloch' ? 'tight_binding' : '',
        model: state.model,
      }, window.location.origin);
    }
  } catch (error) {
    status.className = 'save-status error';
    status.textContent = error.message || 'Save failed';
  } finally {
    button.disabled = false;
  }
}

function returnToTargetWebUi(target) {
  const currentTarget = builderWorkContext().target;
  let targetWindow = null;
  if (target === currentTarget && window.opener && !window.opener.closed) {
    targetWindow = window.opener;
  } else {
    targetWindow = openTargetWindow(target);
  }
  if (targetWindow) {
    targetWindow.focus();
    window.close();
  }
}

function setupEventListeners() {
  document.getElementById('build-lattice').addEventListener('click', buildLattice);
  document.getElementById('apply-globals').addEventListener('click', applyGlobalsToAll);
  document.getElementById('reset-view').addEventListener('click', () => {
    buildLattice();
  });
  document.getElementById('center-layout').addEventListener('click', () => {
    centerLayout();
    render();
  });
  document.getElementById('zoom-in').addEventListener('click', () => setZoom(state.view.zoom * 1.2));
  document.getElementById('zoom-out').addEventListener('click', () => setZoom(state.view.zoom / 1.2));
  document.getElementById('zoom-reset').addEventListener('click', resetView);
  document.getElementById('tool-pan').addEventListener('click', () => setPanMode(state.view.mode !== 'pan'));
  document.getElementById('delete-selected').addEventListener('click', removeSelected);
  document.getElementById('download-json').addEventListener('click', () => {
    if (!syncElectronCountsFromMultiplicity()) {
      document.getElementById('spin-multiplicity').reportValidity();
      return;
    }
    updateGlobalsFromControls();
    downloadText('model_hamiltonian_spec.json', specJson(), 'application/json;charset=utf-8');
  });
  document.getElementById('save-python').addEventListener('click', savePythonInputFile);
  document.getElementById('return-agent-ui').addEventListener('click', () => returnToTargetWebUi('agent'));
  document.getElementById('return-study-ui').addEventListener('click', () => returnToTargetWebUi('study'));

  document.querySelectorAll('[data-representation]').forEach((button) => {
    button.addEventListener('click', () => {
      if (!BLOCH_UI_ENABLED && button.dataset.representation === 'bloch') {
        return;
      }
      state.representation = button.dataset.representation;
      if (state.representation === 'bloch' && state.template1d === 'ring') {
        state.template1d = 'chain';
      }
      updateTemplateControls();
      buildLattice();
    });
  });

  document.getElementById('tool-select').addEventListener('click', () => setTool('select'));
  document.getElementById('tool-add-site').addEventListener('click', () => setTool('add_site'));
  document.getElementById('tool-add-bond').addEventListener('click', () => setTool('add_bond'));

  document.getElementById('tab-summary').addEventListener('click', () => setTab('summary'));
  document.getElementById('tab-h1').addEventListener('click', () => setTab('h1'));
  document.getElementById('tab-json').addEventListener('click', () => setTab('json'));
  document.getElementById('tab-python').addEventListener('click', () => setTab('python'));

  ['dimension', 'template-1d', 'template-2d', 'boundary', 'size-x', 'size-y'].forEach((id) => {
    document.getElementById(id).addEventListener('input', () => {
      updateGlobalsFromControls();
      updateTemplateControls();
      buildLattice();
    });
  });

  document.getElementById('bond-modulation').addEventListener('input', () => {
    updateGlobalsFromControls();
    updateTemplateControls();
    buildLattice();
  });

  ['model-type', 'energy-unit', 'global-t', 'ssh-delta', 'global-epsilon', 'global-u', 'global-v', 'phonon-omega', 'electron-phonon-g', 'phonon-cutoff', 'nalpha', 'nbeta', 'spin-multiplicity', 'geometry-coupled', 'geometry-coupled-v', 'hopping-function', 'interaction-function', 'geo-r0', 'hopping-parameter', 'interaction-parameter'].forEach((id) => {
    document.getElementById(id).addEventListener('input', () => {
      if (id === 'spin-multiplicity') {
        if (!syncElectronCountsFromMultiplicity()) return;
      } else if (id === 'nalpha' || id === 'nbeta') {
        syncMultiplicityFromElectronCounts();
      }
      updateGlobalsFromControls();
      updateTemplateControls();
      updateEnergyUnitLabels();
      updateInspector();
      updatePreview();
      updateCanvasText();
      renderBonds();
    });
  });

  ['electrons-per-cell', 'kmesh-x', 'kmesh-y', 'kpoint-scheme', 'kshift-x', 'kshift-y', 'path-points', 'dos-auto', 'dos-sigma', 'dos-points'].forEach((id) => {
    document.getElementById(id).addEventListener('input', () => {
      updateGlobalsFromControls();
      updateTemplateControls();
      updatePreview();
      updateCanvasText();
    });
  });

  ['site-x', 'site-y', 'site-epsilon', 'site-u', 'site-phonon-omega', 'site-electron-phonon-g'].forEach((id) => {
    document.getElementById(id).addEventListener('input', updateSelectedSiteFromInspector);
  });
  ['bond-t', 'bond-v', 'bond-cell-offset-x', 'bond-cell-offset-y', 'bond-add-hc'].forEach((id) => {
    document.getElementById(id).addEventListener('input', updateSelectedBondFromInspector);
  });

  canvas.addEventListener('pointerdown', handleCanvasPointerDown);
  canvas.addEventListener('pointermove', handleCanvasPointerMove);
  canvas.addEventListener('pointerup', handleCanvasPointerUp);
  canvas.addEventListener('pointercancel', handleCanvasPointerUp);
}

function initializeApp() {
  setupEventListeners();
  updateTemplateControls();
  updateEnergyUnitLabels();
  buildLattice();
}
