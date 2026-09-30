function densityFittingSelectValue(enabled, auxbasis) {
  if (enabled === false || auxbasis === '__off__') {
    return '__off__';
  }
  if (enabled === true) {
    return typeof auxbasis === 'string' ? auxbasis : '';
  }
  return typeof auxbasis === 'string' && auxbasis ? auxbasis : '__off__';
}

function orderedAuxbasisOptionsForBasis(basisValue) {
  const select = document.getElementById('density-fitting-auxbasis');
  const basisKey = typeof basisValue === 'string' ? basisValue.trim().toLowerCase() : '';
  const recommendations = AUXBASIS_RECOMMENDATIONS[basisKey] || ['def2-universal-jkfit', 'weigend+etb'];
  const visibleOrder = ['__off__', '', ...recommendations];
  const seen = new Set();
  const currentValue = select.value;
  const options = Array.from(select.options);
  const ranked = [];
  visibleOrder.forEach((value) => {
    const match = options.find((option) => option.value === value);
    if (match && !seen.has(match.value)) {
      ranked.push(match);
      seen.add(match.value);
    }
  });
  options.forEach((option) => {
    if (!seen.has(option.value)) {
      ranked.push(option);
      seen.add(option.value);
    }
  });
  select.replaceChildren(...ranked);
  if (Array.from(select.options).some((option) => option.value === currentValue)) {
    select.value = currentValue;
  } else {
    select.value = '__off__';
  }
}

function collectFormData() {
  if (currentTaskFamily() === 'model_hamiltonian') {
    return collectModelHamiltonianFormData();
  }
  if (currentTaskFamily() === 'periodic') {
    return collectPeriodicFormData();
  }
  const method = document.getElementById('method').value;
  const ncasText = document.getElementById('ncas').value.trim();
  const nelecasText = document.getElementById('nelecas').value.trim();
  const activeOrbitalsText = document.getElementById('active-orbitals').value.trim();
  const localizationMethod = document.getElementById('localization-method').value;
  const localizationScope = document.getElementById('localization-scope').value;
  const orbitalOrdering = document.getElementById('orbital-ordering').value;
  const orbitalOrder = document.getElementById('manual-orbital-order').value
    .split(/[\s,]+/)
    .map((item) => item.trim())
    .filter((item) => /^\d+$/.test(item))
    .map((item) => Number.parseInt(item, 10));
  const densityFittingAuxbasis = document.getElementById('density-fitting-auxbasis').value;
  const densityFittingEnabled = densityFittingAuxbasis !== '__off__';
  const activeSpaceEnabled = document.getElementById('active-space-enabled').checked || method === 'casci' || method === 'casscf';
  const activeSpaceSolver = document.getElementById('active-space-solver').value;
  const targetsCas = ['casci', 'casscf'].includes(method);
  const supportsStateTargets = method === 'fci' || targetsCas;
  const stateTargetNroots = Math.max(1, integerInputValue('state-target-nroots', 1));
  const stateAverageWeights = document.getElementById('state-average-weights').value
    .split(',')
    .map((item) => Number.parseFloat(item.trim()))
    .filter((item) => Number.isFinite(item));
  const effectiveStateAverageWeights = stateAverageWeights.length
    ? stateAverageWeights
    : (method === 'casscf' && stateTargetNroots > 1
      ? Array(stateTargetNroots).fill(1 / stateTargetNroots)
      : []);
  const outputs = selectedOutputValues();
  if (supportsStateTargets && stateTargetNroots > 1 && !outputs.includes('excited_states')) {
    outputs.push('excited_states');
  }
  const stateTargetOptions = supportsStateTargets
    ? {
      nroots: stateTargetNroots,
      ...(method === 'casscf' && stateTargetNroots > 1 && effectiveStateAverageWeights.length
        ? { state_average_weights: effectiveStateAverageWeights }
        : {}),
    }
    : {};
  const payload = {
    task_type: 'molecular',
    atom: document.getElementById('atom').value.trim(),
    basis: document.getElementById('basis').value,
    method,
    xc: method === 'dft' ? document.getElementById('xc').value : null,
    job: document.getElementById('job').value,
    charge: integerInputValue('charge', 0),
    spin: integerInputValue('spin', 0),
    outputs,
    orbital_processing: {
      enabled: localizationMethod !== 'none' || orbitalOrdering !== 'canonical',
      localization_method: localizationMethod,
      localization_scope: localizationScope,
      use_natural_orbitals: document.getElementById('use-natural-orbitals').checked,
      orbital_ordering: orbitalOrdering,
      orbital_order: orbitalOrdering === 'manual' ? orbitalOrder : [],
    },
    density_fitting: {
      enabled: densityFittingEnabled,
      auxbasis: densityFittingEnabled && densityFittingAuxbasis ? densityFittingAuxbasis : null,
      apply_to: 'scf',
    },
    active_space: {
      enabled: activeSpaceEnabled,
      selection_method: document.getElementById('active-space-method').value,
      ncas: ncasText ? Number.parseInt(ncasText, 10) : null,
      nelecas: nelecasText,
      orbital_indices: activeOrbitalsText,
      avas_targets: document.getElementById('avas-targets').value.trim(),
      avas_threshold: Number.parseFloat(document.getElementById('avas-threshold').value) || 0.2,
      target_method: targetsCas ? method : null,
      target_solver: targetsCas ? activeSpaceSolver : null,
      target_solver_options: targetsCas ? stateTargetOptions : {},
      approved: document.getElementById('active-space-approved').checked,
    },
    post_cas: {
      sc_nevpt2: {
        enabled: document.getElementById('sc-nevpt2-enabled').checked,
        root: 0,
        density_fit: document.getElementById('sc-nevpt2-density-fit').checked,
      },
    },
  };
  const restricted = document.getElementById('restricted').value;
  if (restricted !== 'auto') {
    payload.restricted = restricted === 'true';
  }
  if (targetsCas) {
    payload.solver = {
      name: activeSpaceSolver,
      options: stateTargetOptions,
    };
  } else if (method === 'fci') {
    payload.solver = { name: 'fci', options: stateTargetOptions };
  }
  return payload;
}

function setSelectValue(selectId, value) {
  if (typeof value !== 'string') {
    return;
  }
  const select = document.getElementById(selectId);
  const normalizedValue = value.trim();
  if (!normalizedValue) {
    if (Array.from(select.options).some((option) => option.value === '')) {
      select.value = '';
    }
    return;
  }
  const exactOption = Array.from(select.options).find((option) => option.value === normalizedValue);
  if (exactOption) {
    select.value = exactOption.value;
    return;
  }
  const caseInsensitiveOption = Array.from(select.options).find((option) => option.value.toLowerCase() === normalizedValue.toLowerCase());
  if (caseInsensitiveOption) {
    select.value = caseInsensitiveOption.value;
    return;
  }
  const option = document.createElement('option');
  option.value = normalizedValue;
  option.textContent = normalizedValue;
  select.appendChild(option);
  select.value = normalizedValue;
}

function applyTaskSpecToForm(taskSpec) {
  if (!taskSpec || typeof taskSpec !== 'object') {
    return;
  }
  if (taskSpec.task_type === 'model_hamiltonian') {
    document.getElementById('task-family').value = 'model_hamiltonian';
    document.getElementById('model-nroots').value = '1';
    setSelectValue('model-dmet-execution-mode', '');
    setSelectValue('model-dmet-fragment-mode', '');
    document.getElementById('model-dmet-impurity-shape').value = '';
    document.getElementById('model-dmet-impurity-size').value = '';
    setSelectValue('model-dmet-impurity-solver', 'fci');
    document.getElementById('model-dmet-ccsd-beta').value = '';
    setSelectValue('model-dmet-block2-preset', 'balanced');
    setSelectValue('model-dmet-block2-ordering', 'canonical');
    setSelectValue('model-dmet-reference', 'unrestricted');
    setSelectValue('model-dmet-reference-density', 'pm');
    document.getElementById('model-dmet-interacting-bath').value = String(true);
    document.getElementById('model-dmet-max-iterations').value = '50';
    document.getElementById('model-dmet-energy-tolerance').value = '1e-6';
    document.getElementById('model-dmet-density-tolerance').value = '1e-4';
    if (typeof taskSpec.model_hamiltonian_input_file === 'string') {
      document.getElementById('model-hamiltonian-input-file').value = taskSpec.model_hamiltonian_input_file;
    }
    if (typeof taskSpec.solver === 'string') {
      setSelectValue('model-solver', taskSpec.solver);
    } else if (taskSpec.solver && typeof taskSpec.solver.name === 'string') {
      setSelectValue('model-solver', taskSpec.solver.name);
      const solverOptions = taskSpec.solver.options && typeof taskSpec.solver.options === 'object'
        ? taskSpec.solver.options
        : {};
      document.getElementById('model-nroots').value = String(solverOptions.nroots || 1);
      if (taskSpec.solver.name === 'dmet') {
        const impurityShape = Array.isArray(solverOptions.impurity_shape)
          ? solverOptions.impurity_shape.join(',')
          : '';
        document.getElementById('model-dmet-impurity-shape').value = impurityShape;
        document.getElementById('model-dmet-impurity-size').value = solverOptions.impurity_size || '';
        setSelectValue(
          'model-dmet-execution-mode',
          ['translational', 'finite_graph'].includes(solverOptions.execution_mode)
            ? solverOptions.execution_mode
            : ''
        );
        const fragmentDefinition = ['primitive_cell', 'cell_shape', 'site_count', 'honeycomb_hexagon'].includes(
          solverOptions.fragment_definition
        )
          ? solverOptions.fragment_definition
          : (impurityShape ? 'cell_shape' : (solverOptions.impurity_size ? 'site_count' : ''));
        setSelectValue('model-dmet-fragment-mode', fragmentDefinition);
        setSelectValue('model-dmet-impurity-solver', solverOptions.impurity_solver || 'fci');
        const impuritySolverOptions = solverOptions.impurity_solver_options
          && typeof solverOptions.impurity_solver_options === 'object'
          ? solverOptions.impurity_solver_options
          : {};
        document.getElementById('model-dmet-ccsd-beta').value = impuritySolverOptions.beta ?? '';
        setSelectValue('model-dmet-block2-preset', impuritySolverOptions.preset || 'balanced');
        setSelectValue(
          'model-dmet-block2-ordering',
          impuritySolverOptions.orbital_ordering || 'canonical'
        );
        setSelectValue(
          'model-dmet-reference',
          solverOptions.reference === 'restricted' ? 'restricted' : 'unrestricted'
        );
        setSelectValue('model-dmet-reference-density', solverOptions.reference_density_guess || 'pm');
        document.getElementById('model-dmet-interacting-bath').value = String(solverOptions.interacting_bath !== false);
        document.getElementById('model-dmet-max-iterations').value = String(solverOptions.max_iterations || 50);
        document.getElementById('model-dmet-energy-tolerance').value = String(solverOptions.energy_tolerance || 1e-6);
        document.getElementById('model-dmet-density-tolerance').value = String(
          solverOptions.density_tolerance || 1e-4
        );
      }
    }
    syncTaskFamilyControls();
    syncDmetImpuritySolverControls();
    syncDmetFragmentControls();
    return;
  }
  if (taskSpec.task_type === 'periodic') {
    const periodic = taskSpec.periodic && typeof taskSpec.periodic === 'object' ? taskSpec.periodic : {};
    const system = taskSpec.system && typeof taskSpec.system === 'object' ? taskSpec.system : {};
    const method = taskSpec.method && typeof taskSpec.method === 'object' ? taskSpec.method : {};
    const solver = taskSpec.solver && typeof taskSpec.solver === 'object' ? taskSpec.solver : {};
    const embedding = taskSpec.embedding && typeof taskSpec.embedding === 'object' ? taskSpec.embedding : {};
    const analysis = taskSpec.analysis && typeof taskSpec.analysis === 'object' ? taskSpec.analysis : {};
    document.getElementById('task-family').value = 'periodic';
    if (typeof periodic.structure_format === 'string') {
      setSelectValue('periodic-format', periodic.structure_format);
    }
    if (typeof periodic.structure_text === 'string') {
      document.getElementById('periodic-structure-text').value = periodic.structure_text;
    }
    if (typeof periodic.basis === 'string') {
      setSelectValue('periodic-basis', periodic.basis);
    }
    if (typeof periodic.pseudo === 'string') {
      setSelectValue('periodic-pseudo', periodic.pseudo);
    }
    const methodName = typeof taskSpec.method === 'string' ? taskSpec.method : method.name;
    if (typeof methodName === 'string') {
      setSelectValue('periodic-method', methodName);
    }
    const solverName = typeof taskSpec.solver === 'string' ? taskSpec.solver : solver.name;
    const normalizedSolver = String(solverName || '').toLowerCase().replace('-', '_');
    const correlationTreatment = ['gw', 'hf_dmft', 'gw_dmft'].includes(normalizedSolver)
      ? normalizedSolver
      : 'none';
    const usesGw = correlationTreatment === 'gw' || correlationTreatment === 'gw_dmft';
    const usesDmft = correlationTreatment === 'hf_dmft' || correlationTreatment === 'gw_dmft';
    setSelectValue('periodic-correlation-treatment', correlationTreatment);
    const solverOptions = solver.options && typeof solver.options === 'object' ? solver.options : {};
    if (usesDmft) {
      setSelectValue('periodic-dmft-localization', embedding.localization_method || 'iao');
      setSelectValue('periodic-dmft-impurity-solver', solverOptions.impurity_solver || 'cc');
      document.getElementById('periodic-dmft-correlated-orbitals').value = Array.isArray(
        embedding.correlated_orbital_indices
      ) ? embedding.correlated_orbital_indices.join(',') : '';
      document.getElementById('periodic-dmft-minimal-basis').value = embedding.minimal_basis || 'minao';
      document.getElementById('periodic-dmft-nbath').value = solverOptions.nbath || 4;
      document.getElementById('periodic-dmft-max-iterations').value = solverOptions.max_iterations || 10;
      document.getElementById('periodic-dmft-convergence').value = solverOptions.convergence_tolerance || 1e-3;
    }
    if (usesGw) {
      document.getElementById('periodic-gw-broadening').value = (solverOptions.gw_broadening || (0.1 / 27.211386)) * 27.211386;
      document.getElementById('periodic-gw-frequency-maximum').value = Array.isArray(solverOptions.gw_frequency_window)
        ? Number(solverOptions.gw_frequency_window[1] || 0) * 27.211386
        : 18;
      document.getElementById('periodic-gw-real-points').value = solverOptions.gw_real_frequency_points || 181;
      document.getElementById('periodic-gw-imaginary-points').value = solverOptions.gw_imaginary_frequency_points || 100;
      document.getElementById('periodic-gw-finite-size-correction').checked = solverOptions.gw_finite_size_correction !== false;
    }
    const xc = typeof taskSpec.xc === 'string' ? taskSpec.xc : method.xc;
    if (typeof xc === 'string') {
      setSelectValue('periodic-xc', xc);
    }
    if (Array.isArray(periodic.kmesh) && periodic.kmesh.length === 3) {
      document.getElementById('periodic-kmesh-x').value = periodic.kmesh[0];
      document.getElementById('periodic-kmesh-y').value = periodic.kmesh[1];
      document.getElementById('periodic-kmesh-z').value = periodic.kmesh[2];
    }
    if (typeof periodic.kpoint_scheme === 'string') {
      setSelectValue('periodic-kpoint-scheme', periodic.kpoint_scheme);
    }
    if (typeof periodic.band_path_mode === 'string') {
      setSelectValue('periodic-band-path-mode', periodic.band_path_mode);
    } else {
      setSelectValue('periodic-band-path-mode', 'auto');
    }
    document.getElementById('periodic-band-path').value = periodic.band_path || '';
    document.getElementById('periodic-band-special-points').value = formatPeriodicBandSpecialPoints(
      periodic.band_path_special_points
    );
    document.getElementById('periodic-band-path-npoints').value = periodic.band_path_npoints ?? '80';
    document.getElementById('periodic-band-path-reference-distance').value = periodic.band_path_reference_distance ?? '0.025';
    document.getElementById('periodic-band-path-symprec').value = periodic.band_path_symprec ?? '1e-5';
    if (Array.isArray(periodic.kpoint_shift) && periodic.kpoint_shift.length === 3) {
      document.getElementById('periodic-kpoint-shift-x').value = periodic.kpoint_shift[0];
      document.getElementById('periodic-kpoint-shift-y').value = periodic.kpoint_shift[1];
      document.getElementById('periodic-kpoint-shift-z').value = periodic.kpoint_shift[2];
    }
    if (typeof periodic.precision !== 'undefined' && periodic.precision !== null) {
      document.getElementById('periodic-precision').value = periodic.precision;
    }
    document.getElementById('periodic-ke-cutoff').value = periodic.ke_cutoff ?? '';
    if (Array.isArray(periodic.fft_mesh) && periodic.fft_mesh.length === 3) {
      document.getElementById('periodic-fft-mesh-x').value = periodic.fft_mesh[0];
      document.getElementById('periodic-fft-mesh-y').value = periodic.fft_mesh[1];
      document.getElementById('periodic-fft-mesh-z').value = periodic.fft_mesh[2];
    } else {
      document.getElementById('periodic-fft-mesh-x').value = '';
      document.getElementById('periodic-fft-mesh-y').value = '';
      document.getElementById('periodic-fft-mesh-z').value = '';
    }
    if (typeof periodic.density_fitting_method === 'string') {
      setSelectValue('periodic-density-fitting', periodic.density_fitting_method);
    }
    document.getElementById('periodic-df-auxbasis').value = periodic.density_fitting_auxbasis || '';
    if (typeof periodic.exxdiv === 'string') {
      setSelectValue('periodic-exxdiv', periodic.exxdiv);
    }
    if (typeof periodic.smearing_method === 'string') {
      setSelectValue('periodic-smearing', periodic.smearing_method);
    }
    document.getElementById('periodic-smearing-sigma').value = periodic.smearing_sigma ?? '0.01';
    document.getElementById('periodic-smearing-fix-spin').checked = Boolean(periodic.smearing_fix_spin);
    const charge = typeof taskSpec.charge !== 'undefined' ? taskSpec.charge : system.charge;
    const spin = typeof taskSpec.spin !== 'undefined' ? taskSpec.spin : system.spin;
    if (typeof charge !== 'undefined') {
      document.getElementById('periodic-charge').value = Number.parseInt(charge, 10);
    }
    if (typeof spin !== 'undefined') {
      document.getElementById('periodic-spin').value = Number.parseInt(spin, 10);
    }
    const restricted = typeof taskSpec.restricted === 'boolean' ? taskSpec.restricted : method.restricted;
    document.getElementById('periodic-restricted').value = typeof restricted === 'boolean'
      ? (restricted ? 'true' : 'false')
      : 'auto';
    const outputs = Array.isArray(taskSpec.outputs)
      ? taskSpec.outputs
      : (Array.isArray(analysis.outputs) ? analysis.outputs : DEFAULT_PERIODIC_OUTPUTS);
    document.querySelectorAll('#periodic-output-group input[name="outputs"]').forEach((item) => {
      item.checked = outputs.map((value) => String(value).toLowerCase()).includes(item.value.toLowerCase());
    });
    resetPeriodicStructurePreview();
    syncTaskFamilyControls();
    syncPeriodicMethodControls();
    syncPeriodicCorrelationControls();
    syncPeriodicNumericalControls();
    syncPeriodicBandPathControls();
    return;
  }
  if (taskSpec.task_type === 'molecular') {
    document.getElementById('task-family').value = 'molecular';
  }
  if (typeof taskSpec.atom === 'string') {
    document.getElementById('atom').value = taskSpec.atom;
  }
  if (typeof taskSpec.basis === 'string') {
    setSelectValue('basis', taskSpec.basis);
    orderedAuxbasisOptionsForBasis(document.getElementById('basis').value);
  }
  if (typeof taskSpec.method === 'string') {
    setSelectValue('method', taskSpec.method);
  }
  if (typeof taskSpec.xc === 'string') {
    setSelectValue('xc', taskSpec.xc);
  }
  if (typeof taskSpec.job === 'string') {
    setSelectValue('job', taskSpec.job);
  }
  if (typeof taskSpec.charge !== 'undefined') {
    document.getElementById('charge').value = Number.parseInt(taskSpec.charge, 10);
  }
  if (typeof taskSpec.spin !== 'undefined') {
    document.getElementById('spin').value = Number.parseInt(taskSpec.spin, 10);
  }
  if (typeof taskSpec.restricted === 'boolean') {
    document.getElementById('restricted').value = taskSpec.restricted ? 'true' : 'false';
  } else {
    document.getElementById('restricted').value = 'auto';
  }
  if (taskSpec.orbital_processing && typeof taskSpec.orbital_processing === 'object') {
    document.getElementById('orbital-processing-enabled').checked = Boolean(taskSpec.orbital_processing.enabled);
    if (typeof taskSpec.orbital_processing.localization_method === 'string') {
      setSelectValue('localization-method', taskSpec.orbital_processing.localization_method);
    }
    if (typeof taskSpec.orbital_processing.localization_scope === 'string') {
      setSelectValue('localization-scope', taskSpec.orbital_processing.localization_scope);
    }
    if (typeof taskSpec.orbital_processing.orbital_ordering === 'string') {
      setSelectValue('orbital-ordering', taskSpec.orbital_processing.orbital_ordering);
    }
    document.getElementById('manual-orbital-order').value = Array.isArray(taskSpec.orbital_processing.orbital_order)
      ? taskSpec.orbital_processing.orbital_order.join(',')
      : '';
    document.getElementById('use-natural-orbitals').checked = Boolean(taskSpec.orbital_processing.use_natural_orbitals);
  }
  if (taskSpec.density_fitting && typeof taskSpec.density_fitting === 'object') {
    orderedAuxbasisOptionsForBasis(document.getElementById('basis').value);
    const auxbasis = typeof taskSpec.density_fitting.auxbasis === 'string' ? taskSpec.density_fitting.auxbasis : '';
    const densityFittingValue = densityFittingSelectValue(taskSpec.density_fitting.enabled, auxbasis);
    document.getElementById('density-fitting-auxbasis').value = Array.from(document.getElementById('density-fitting-auxbasis').options).some((option) => option.value === densityFittingValue) ? densityFittingValue : '__off__';
  }
  if (taskSpec.active_space && typeof taskSpec.active_space === 'object') {
    document.getElementById('active-space-enabled').checked = Boolean(taskSpec.active_space.enabled);
    if (typeof taskSpec.active_space.selection_method === 'string') {
      setSelectValue('active-space-method', taskSpec.active_space.selection_method);
    }
    if (typeof taskSpec.active_space.ncas !== 'undefined' && taskSpec.active_space.ncas !== null) {
      document.getElementById('ncas').value = String(taskSpec.active_space.ncas);
    }
    if (typeof taskSpec.active_space.nelecas !== 'undefined' && taskSpec.active_space.nelecas !== null) {
      const nelecasValue = Array.isArray(taskSpec.active_space.nelecas)
        ? taskSpec.active_space.nelecas.join(',')
        : String(taskSpec.active_space.nelecas);
      document.getElementById('nelecas').value = nelecasValue;
    }
    if (taskSpec.active_space.orbital_indices) {
      document.getElementById('active-orbitals').value = activeOrbitalIndicesToText(taskSpec.active_space.orbital_indices);
    }
    if (taskSpec.active_space.avas_targets) {
      const targets = Array.isArray(taskSpec.active_space.avas_targets)
        ? taskSpec.active_space.avas_targets.join(', ')
        : String(taskSpec.active_space.avas_targets);
      document.getElementById('avas-targets').value = targets;
    }
    if (taskSpec.active_space.avas_threshold !== null && typeof taskSpec.active_space.avas_threshold !== 'undefined') {
      document.getElementById('avas-threshold').value = String(taskSpec.active_space.avas_threshold);
    }
    document.getElementById('active-space-approved').checked = Boolean(taskSpec.active_space.approved);
    syncActiveSpaceMethodControls();
  }
  if (taskSpec.solver && typeof taskSpec.solver === 'object' && typeof taskSpec.solver.name === 'string') {
    setSelectValue('active-space-solver', taskSpec.solver.name);
    const solverOptions = taskSpec.solver.options && typeof taskSpec.solver.options === 'object'
      ? taskSpec.solver.options
      : {};
    document.getElementById('state-target-nroots').value = solverOptions.nroots || 1;
    document.getElementById('state-average-weights').value = Array.isArray(solverOptions.state_average_weights)
      ? solverOptions.state_average_weights.join(',')
      : '';
  }
  if (taskSpec.post_cas && typeof taskSpec.post_cas === 'object') {
    const scNevpt2 = taskSpec.post_cas.sc_nevpt2 && typeof taskSpec.post_cas.sc_nevpt2 === 'object'
      ? taskSpec.post_cas.sc_nevpt2
      : {};
    document.getElementById('sc-nevpt2-enabled').checked = Boolean(scNevpt2.enabled);
    document.getElementById('sc-nevpt2-root').value = 0;
    document.getElementById('sc-nevpt2-density-fit').checked = typeof scNevpt2.density_fit === 'undefined' ? true : Boolean(scNevpt2.density_fit);
  }
  const outputs = Array.isArray(taskSpec.outputs) ? taskSpec.outputs.map((item) => String(item).toLowerCase()) : [];
  document.querySelectorAll('input[name="outputs"]').forEach((item) => {
    item.checked = outputs.includes(item.value.toLowerCase());
  });
  syncTaskFamilyControls();
  syncMethodControls();
  syncOrbitalProcessingControls();
  updateActiveSpaceSummary();
  syncPostCasControls();
  updateMolecularStructurePreview();
}

function updateRequestPlaceholder(payload) {
  const requestInput = document.getElementById('request');
  if (payload && payload.clarification_questions && payload.clarification_questions.length) {
    requestInput.placeholder = payload.clarification_questions.join(' ');
    return;
  }
  requestInput.placeholder = t('requestPlaceholder');
}

function syncMethodControls() {
  const method = document.getElementById('method').value;
  const xc = document.getElementById('xc');
  xc.disabled = method !== 'dft';
  const isCasMethod = method === 'casci' || method === 'casscf';
  if (isCasMethod) {
    document.getElementById('active-space-enabled').checked = true;
  }
  syncActiveSpaceSolverControls();
  syncPostCasControls();
}
