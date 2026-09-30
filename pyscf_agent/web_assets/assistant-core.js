function escapeHtml(value) {
  return AgentUI.escapeHtml(value);
}


function setPre(id, value) {
  document.getElementById(id).textContent = value || '';
}

function t(key) {
  const bundle = TRANSLATIONS.en || {};
  return bundle[key] || key;
}

function normalizedActiveSpaceSolverName(value) {
  const name = String(value || '').trim().toLowerCase().replaceAll('-', '_');
  if (['block2', 'dmrg', 'block2_dmrg'].includes(name)) {
    return 'block2_dmrg';
  }
  return name || null;
}

function activeSpaceSolverLabel(value) {
  const solver = normalizedActiveSpaceSolverName(value);
  if (solver === 'block2_dmrg') {
    return 'block2 DMRG';
  }
  if (solver === 'fci') {
    return langText('PySCF Full CI', 'PySCF Full CI');
  }
  return solver ? solver.toUpperCase() : 'n/a';
}

function resultsSolverName(results) {
  if (!results || typeof results !== 'object') {
    return null;
  }
  const casResult = results.cas_result && typeof results.cas_result === 'object'
    ? results.cas_result
    : {};
  const dmrgResult = casResult.dmrg_result && typeof casResult.dmrg_result === 'object'
    ? casResult.dmrg_result
    : {};
  return normalizedActiveSpaceSolverName(results.solver || casResult.solver || dmrgResult.solver);
}

function updateOutputLabels() {
  const labels = (TRANSLATIONS.en || {}).outputLabels || {};
  document.querySelectorAll('input[name="outputs"]').forEach((input) => {
    const label = input.parentElement;
    if (!label) {
      return;
    }
    const textNode = label.lastChild;
    if (textNode) {
      textNode.textContent = labels[input.value] || input.value;
    }
  });
}

function updateRegisteredOptionLabels() {
  document.querySelectorAll('option[data-label-key]').forEach((option) => {
    option.textContent = t(option.dataset.labelKey);
  });
}

function updateStaticText() {
  document.documentElement.lang = t('htmlLang');
  document.title = t('pageTitle');
  document.getElementById('lang-toggle').textContent = t('toggleLabel');
  document.getElementById('study-agent-link').textContent = t('openStudyAgent');
  document.getElementById('hero-title').textContent = t('heroTitle');
  document.getElementById('hero-description').textContent = t('heroDescription');
  document.getElementById('task-sessions-title').textContent = t('taskSessionsTitle');
  document.getElementById('new-task').textContent = t('newTask');
  document.getElementById('config-title').textContent = t('configTitle');
  document.getElementById('label-task-family').textContent = t('labelTaskFamily');
  document.getElementById('label-work-dir').textContent = t('labelWorkDir');
  document.getElementById('label-execution-target').textContent = t('labelExecutionTarget');
  document.getElementById('label-resource-profile').textContent = t('labelResourceProfile');
  document.getElementById('work-dir').placeholder = t('workDirPlaceholder');
  updateEffectiveWorkDirLabel();
  document.getElementById('open-model-builder').textContent = t('openModelHamiltonianBuilder');
  document.getElementById('label-model-solver').textContent = t('labelModelSolver');
  document.getElementById('preview-model-hamiltonian').textContent = t('previewModelHamiltonian');
  document.getElementById('model-hamiltonian-preview-title').textContent = t('modelHamiltonianPreviewTitle');
  renderModelHamiltonianStructurePreview(lastModelHamiltonianPreview);
  document.getElementById('periodic-structure-title').textContent = t('periodicStructureTitle');
  document.getElementById('label-periodic-format').textContent = t('labelPeriodicFormat');
  document.getElementById('label-periodic-file').textContent = t('labelPeriodicFile');
  document.getElementById('label-periodic-structure').textContent = t('labelPeriodicStructure');
  document.getElementById('periodic-structure-text').placeholder = t('periodicStructurePlaceholder');
  document.getElementById('preview-periodic-structure').textContent = t('previewPeriodicStructure');
  document.getElementById('periodic-calculation-settings-title').textContent = t('sectionCalculationSettings');
  document.getElementById('label-periodic-basis').textContent = t('labelPeriodicBasis');
  document.getElementById('label-periodic-pseudo').textContent = t('labelPeriodicPseudo');
  document.getElementById('label-periodic-method').textContent = t('labelPeriodicMethod');
  document.getElementById('label-periodic-xc').textContent = t('labelPeriodicXc');
  document.getElementById('label-periodic-kmesh').textContent = t('labelPeriodicKmesh');
  document.getElementById('label-periodic-kpoint-scheme').textContent = t('labelPeriodicKpointScheme');
  document.getElementById('label-periodic-band-path-mode').textContent = t('labelPeriodicBandPathMode');
  document.getElementById('label-periodic-band-path').textContent = t('labelPeriodicBandPath');
  document.getElementById('periodic-band-path').placeholder = t('periodicBandPathPlaceholder');
  document.getElementById('label-periodic-band-special-points').textContent = t('labelPeriodicBandSpecialPoints');
  document.getElementById('periodic-band-special-points').placeholder = t('periodicBandSpecialPointsPlaceholder');
  document.getElementById('label-periodic-band-path-npoints').textContent = t('labelPeriodicBandPathNpoints');
  document.getElementById('label-periodic-band-path-reference-distance').textContent = t('labelPeriodicBandPathReferenceDistance');
  document.getElementById('label-periodic-band-path-symprec').textContent = t('labelPeriodicBandPathSymprec');
  document.getElementById('periodic-numerics-title').textContent = t('periodicNumericsTitle');
  document.getElementById('label-periodic-kpoint-shift').textContent = t('labelPeriodicKpointShift');
  document.getElementById('label-periodic-precision').textContent = t('labelPeriodicPrecision');
  document.getElementById('label-periodic-ke-cutoff').textContent = t('labelPeriodicKeCutoff');
  document.getElementById('label-periodic-fft-mesh').textContent = t('labelPeriodicFftMesh');
  document.getElementById('label-periodic-density-fitting').textContent = t('labelPeriodicDensityFitting');
  document.getElementById('label-periodic-df-auxbasis').textContent = t('labelPeriodicDfAuxbasis');
  document.getElementById('periodic-df-auxbasis').placeholder = t('periodicDfAuxbasisPlaceholder');
  document.getElementById('label-periodic-exxdiv').textContent = t('labelPeriodicExxdiv');
  document.getElementById('label-periodic-smearing').textContent = t('labelPeriodicSmearing');
  document.getElementById('label-periodic-smearing-sigma').textContent = t('labelPeriodicSmearingSigma');
  document.getElementById('label-periodic-smearing-fix-spin').textContent = t('labelPeriodicSmearingFixSpin');
  document.getElementById('label-periodic-charge').textContent = t('labelCharge');
  document.getElementById('label-periodic-spin').textContent = t('labelPeriodicSpin');
  document.getElementById('label-periodic-restricted').textContent = t('labelRestricted');
  document.querySelector('#periodic-restricted option[value="auto"]').textContent = t('restrictedAuto');
  document.querySelector('#periodic-restricted option[value="true"]').textContent = t('restrictedTrue');
  document.querySelector('#periodic-restricted option[value="false"]').textContent = t('restrictedFalse');
  renderPeriodicStructurePreview(lastPeriodicStructurePreview);
  document.getElementById('structure-section-title').textContent = t('sectionStructure');
  document.getElementById('label-atom').textContent = t('labelAtom');
  document.getElementById('calculation-settings-title').textContent = t('sectionCalculationSettings');
  document.getElementById('label-basis').textContent = t('labelBasis');
  document.getElementById('label-method').textContent = t('labelMethod');
  document.getElementById('label-xc').textContent = t('labelXc');
  document.getElementById('label-job').textContent = t('labelJob');
  document.getElementById('spin-parameters-title').textContent = t('sectionSpinParameters');
  document.getElementById('label-charge').textContent = t('labelCharge');
  document.getElementById('label-spin').textContent = t('labelSpin');
  document.getElementById('label-restricted').textContent = t('labelRestricted');
  document.querySelector('#restricted option[value="auto"]').textContent = t('restrictedAuto');
  document.querySelector('#restricted option[value="true"]').textContent = t('restrictedTrue');
  document.querySelector('#restricted option[value="false"]').textContent = t('restrictedFalse');
  document.getElementById('active-space-section-title').textContent = t('sectionActiveSpace');
  document.getElementById('label-density-fitting-auxbasis').textContent = t('labelDensityFittingAuxbasis');
  document.querySelector('#density-fitting-auxbasis option[value="__off__"]').textContent = t('densityFittingOff');
  document.querySelector('#density-fitting-auxbasis option[value=""]').textContent = t('densityFittingAuxbasisAuto');
  orderedAuxbasisOptionsForBasis(document.getElementById('basis').value);
  document.getElementById('label-localization-method').textContent = t('labelLocalizationMethod');
  document.getElementById('label-localization-scope').textContent = t('labelLocalizationScope');
  document.querySelector('#localization-scope option[value="analysis"]').textContent = t('localizationScopeAnalysis');
  document.querySelector('#localization-scope option[value="active_space"]').textContent = t('localizationScopeActiveSpace');
  document.getElementById('label-active-space').textContent = t('labelActiveSpace');
  document.getElementById('label-active-space-method').textContent = t('labelActiveSpaceMethod');
  document.getElementById('label-active-space-solver').textContent = t('labelActiveSpaceSolver');
  document.getElementById('label-avas-targets').textContent = t('labelAvasTargets');
  document.getElementById('label-avas-threshold').textContent = t('labelAvasThreshold');
  document.getElementById('avas-targets').placeholder = t('avasTargetsPlaceholder');
  document.getElementById('label-ncas').textContent = t('labelNcas');
  document.getElementById('label-nelecas').textContent = t('labelNelecas');
  document.getElementById('label-active-orbitals').textContent = t('labelActiveOrbitals');
  document.getElementById('active-orbitals').placeholder = t('activeOrbitalsPlaceholder');
  document.getElementById('nelecas').placeholder = t('nelecasPlaceholder');
  document.getElementById('active-space-contract-title').textContent = t('activeSpaceManualEditor');
  document.getElementById('orbital-basis-section-title').textContent = t('orbitalBasisSection');
  document.getElementById('label-orbital-ordering').textContent = t('labelOrbitalOrdering');
  document.querySelector('#orbital-ordering option[value="canonical"]').textContent = t('orbitalOrderingCanonical');
  document.querySelector('#orbital-ordering option[value="fiedler"]').textContent = t('orbitalOrderingFiedler');
  document.querySelector('#orbital-ordering option[value="manual"]').textContent = t('orbitalOrderingManual');
  document.getElementById('label-manual-orbital-order').textContent = t('labelManualOrbitalOrder');
  document.getElementById('manual-orbital-order').placeholder = t('manualOrbitalOrderPlaceholder');
  document.getElementById('state-target-section-title').textContent = t('stateTargetSection');
  document.getElementById('label-state-target-nroots').textContent = t('labelStateTargetNroots');
  document.getElementById('label-state-average-weights').textContent = t('labelStateAverageWeights');
  document.getElementById('state-average-weights').placeholder = t('stateAverageWeightsPlaceholder');
  document.getElementById('post-cas-section-title').textContent = t('postCasSection');
  document.getElementById('label-sc-nevpt2').textContent = t('labelScNevpt2');
  document.getElementById('label-sc-nevpt2-root').textContent = t('labelScNevpt2Root');
  document.getElementById('label-sc-nevpt2-density-fit').textContent = t('labelScNevpt2DensityFit');
  document.getElementById('screen-active-space').textContent = t('activeSpaceScreenButton');
  document.getElementById('label-request').textContent = t('labelRequest');
  document.getElementById('atom').placeholder = t('atomPlaceholder');
  document.getElementById('approval-assumptions-title').textContent = t('approvalAssumptionsTitle');
  document.getElementById('approval-atom-title').textContent = t('approvalAtomTitle');
  document.getElementById('approval-atom').placeholder = t('approvalAtomPlaceholder');
  document.getElementById('approve-structure').textContent = t('confirmAndContinue');
  document.getElementById('cancel-structure').textContent = t('cancel');
  document.getElementById('confirm-compute').textContent = t('confirmCompute');
  document.getElementById('refresh-status').textContent = t('refreshStatus');
  document.getElementById('clear').textContent = t('clear');
  document.getElementById('conversation-title').textContent = t('conversationTitle');
  document.getElementById('summary-title').textContent = t('summaryTitle');
  document.getElementById('summary-description').textContent = t('summaryDescription');
  document.getElementById('stat-status-label').textContent = t('statStatusLabel');
  document.getElementById('stat-retry-label').textContent = t('statRetryLabel');
  document.getElementById('stat-attempts-label').textContent = t('statAttemptsLabel');
  document.getElementById('structured-results-summary').textContent = t('structuredResultsSummary');
  document.getElementById('strong-correlation-diagnostics-title').textContent = t('strongCorrelationDiagnosticsTitle');
  document.getElementById('active-space-audit-title').textContent = t('activeSpaceAuditTitle');
  document.getElementById('llm-result-analysis-title').textContent = t('llmResultAnalysisTitle');
  document.getElementById('input-preview-summary').textContent = t('inputPreviewSummary');
  document.getElementById('raw-output-summary').textContent = t('rawOutputSummary');
  document.getElementById('defaults-and-retries-title').textContent = t('defaultsAndRetriesTitle');
  document.getElementById('diagnostics-title').textContent = t('diagnosticsTitle');
  document.getElementById('attempts-title').textContent = t('attemptsTitle');
  if (!lastExecutionPayload || !lastExecutionPayload.result_analysis) {
    document.getElementById('llm-result-analysis').textContent = llmResultAnalysisAvailable() ? t('llmResultAnalysisEmpty') : t('llmResultAnalysisUnavailable');
  }
  updateRegisteredOptionLabels();
  updateOutputLabels();
  syncActiveSpaceMethodControls();
  renderTaskSessionSelector();
  updateRequestPlaceholder(null);
  updateActiveSpaceSummary();
  updateMolecularStructurePreview();
  setRunState(document.getElementById('run').disabled);
  if (pendingExecutionRequest) {
    renderComputeConfirmPanel(t('computeReady'), pendingExecutionRequest);
  } else {
    clearComputeConfirmPanel();
  }
  if (pendingApproval) {
    renderApprovalPanel(pendingApproval);
  }
  setAnalyzeState(false);
}

function setLanguage(nextLanguage) {
  currentLanguage = 'en';
  try {
    window.localStorage.removeItem('pyscf-agent-ui-language');
  } catch (error) {
  }
  updateStaticText();
  renderConversation(conversationHistory, lastExecutionPayload ? lastExecutionPayload.messages || [] : []);
}

function formatRequestSource(source) {
  if (source === 'llm') {
    return t('sourceLlm');
  }
  if (source === 'structured_seed') {
    return t('sourceStructuredSeed');
  }
  if (source === 'fallback') {
    return t('sourceFallback');
  }
  if (source === 'llm_unavailable') {
    return t('sourceLlmUnavailable');
  }
  if (source === 'empty_input') {
    return t('sourceEmptyInput');
  }
  return source || 'unknown';
}

function formatLlmCacheScope(scope) {
  if (typeof scope !== 'string' || !scope) {
    return t('llmScopeUnknown');
  }
  if (scope.endsWith(':supported')) {
    return t('llmScopeSupported');
  }
  if (scope.endsWith(':unsupported')) {
    return t('llmScopeUnsupported');
  }
  if (scope === 'llm:disabled') {
    return t('llmScopeDisabled');
  }
  return t('llmScopePending');
}

function formatDefaultEntry(item) {
  if (!item || typeof item !== 'object') {
    return '';
  }
  const value = Array.isArray(item.value) ? item.value.join(', ') : String(item.value);
  return `${item.field} = ${value} (${item.reason})`;
}

function renderList(id, items, emptyText) {
  const node = document.getElementById(id);
  if (!items || !items.length) {
    node.innerHTML = `<li class="muted">${escapeHtml(emptyText)}</li>`;
    return;
  }
  node.innerHTML = items.map((item) => `<li>${escapeHtml(item)}</li>`).join('');
}

function renderTable(id, headers, rows, emptyText, className = '') {
  const node = document.getElementById(id);
  if (!rows || !rows.length) {
    node.innerHTML = '<div class="muted">' + escapeHtml(emptyText) + '</div>';
    return;
  }
  const tableClass = className ? 'result-table ' + className : 'result-table';
  node.innerHTML = '<div class="result-table-wrap">' + AgentUI.tableMarkup(
    headers || [], rows.map(row => Array.isArray(row) ? row : []),
    {className: tableClass, rowHeaders: true}) + '</div>';
}

function renderTableWithNotes(id, headers, rows, notes, emptyText, className = '') {
  const node = document.getElementById(id);
  const noteItems = Array.isArray(notes) ? notes.filter((item) => typeof item === 'string' && item.trim()) : [];
  const tableHtml = (!rows || !rows.length)
    ? '<div class="muted">' + escapeHtml(emptyText) + '</div>'
    : (() => {
        const tableClass = className ? 'result-table ' + className : 'result-table';
        const headerHtml = headers && headers.length
          ? '<thead><tr>' + headers.map((header) => '<th>' + escapeHtml(header) + '</th>').join('') + '</tr></thead>'
          : '';
        const bodyHtml = rows.map((row) => {
          const cells = Array.isArray(row) ? row : [];
          return '<tr>' + cells.map((cell, index) => {
            const tag = index === 0 ? 'th' : 'td';
            return '<' + tag + '>' + escapeHtml(cell) + '</' + tag + '>';
          }).join('') + '</tr>';
        }).join('');
        return '<div class="result-table-wrap"><table class="' + tableClass + '">' + headerHtml + '<tbody>' + bodyHtml + '</tbody></table></div>';
      })();
  const notesHtml = noteItems.length
    ? '<ul class="result-notes">' + noteItems.map((item) => '<li>' + escapeHtml(item) + '</li>').join('') + '</ul>'
    : '';
  node.innerHTML = tableHtml + notesHtml;
}

function renderTableWithSections(id, headers, rows, sections, emptyText, className = '') {
  const node = document.getElementById(id);
  const tableHtml = (!rows || !rows.length)
    ? '<div class="muted">' + escapeHtml(emptyText) + '</div>'
    : (() => {
        const tableClass = className ? 'result-table ' + className : 'result-table';
        const headerHtml = headers && headers.length
          ? '<thead><tr>' + headers.map((header) => '<th>' + escapeHtml(header) + '</th>').join('') + '</tr></thead>'
          : '';
        const bodyHtml = rows.map((row) => {
          const cells = Array.isArray(row) ? row : [];
          return '<tr>' + cells.map((cell, index) => {
            const tag = index === 0 ? 'th' : 'td';
            return '<' + tag + '>' + escapeHtml(cell) + '</' + tag + '>';
          }).join('') + '</tr>';
        }).join('');
        return '<div class="result-table-wrap"><table class="' + tableClass + '">' + headerHtml + '<tbody>' + bodyHtml + '</tbody></table></div>';
      })();
  const sectionItems = Array.isArray(sections) ? sections : [];
  const sectionsHtml = sectionItems
    .map((section) => {
      const items = Array.isArray(section.items)
        ? section.items.filter((item) => typeof item === 'string' && item.trim())
        : [];
      if (!items.length) {
        return '';
      }
      return '<section class="result-note-section"><h5>' + escapeHtml(section.title || '') + '</h5><ul class="result-notes">'
        + items.map((item) => '<li>' + escapeHtml(item) + '</li>').join('')
        + '</ul></section>';
    })
    .filter(Boolean)
    .join('');
  node.innerHTML = tableHtml + (sectionsHtml ? '<div class="result-note-sections">' + sectionsHtml + '</div>' : '');
}

function pathRelativeToWorkDir(pathValue, workDir) {
  if (typeof pathValue !== 'string' || !pathValue.trim()) {
    return '';
  }
  const normalizedPath = pathValue.trim().replaceAll('\\', '/');
  if (typeof workDir !== 'string' || !workDir.trim()) {
    return normalizedPath;
  }
  const normalizedBase = workDir.trim().replaceAll('\\', '/').replace(/\/+$/, '');
  if (!normalizedBase) {
    return normalizedPath;
  }
  if (normalizedPath === normalizedBase) {
    return '.';
  }
  if (normalizedPath.startsWith(`${normalizedBase}/`)) {
    return normalizedPath.slice(normalizedBase.length + 1);
  }
  return normalizedPath;
}

function formatTupleValue(value) {
  if (Array.isArray(value)) {
    return `(${value.join(', ')})`;
  }
  return String(value);
}

function isSpinResolvedOrbitalIndices(value) {
  return value && typeof value === 'object' && !Array.isArray(value)
    && (Array.isArray(value.alpha) || Array.isArray(value.beta));
}

function activeOrbitalIndicesToText(value) {
  if (Array.isArray(value)) {
    return value.join(',');
  }
  if (isSpinResolvedOrbitalIndices(value)) {
    const alpha = Array.isArray(value.alpha) ? value.alpha.join(',') : '';
    const beta = Array.isArray(value.beta) ? value.beta.join(',') : '';
    return `alpha:${alpha}; beta:${beta}`;
  }
  return '';
}

function compactNumber(value, digits = 4) {
  if (value === null || typeof value === 'undefined' || value === '') {
    return '';
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return String(value);
  }
  return numeric.toFixed(digits).replace(/\.?0+$/, '');
}

function normalizeOrbitalIndexList(value) {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => Number.parseInt(item, 10))
    .filter((item) => Number.isInteger(item) && item >= 0);
}

function normalizeApprovalOrbitalIndices(value) {
  if (Array.isArray(value)) {
    return normalizeOrbitalIndexList(value);
  }
  if (isSpinResolvedOrbitalIndices(value)) {
    return {
      alpha: normalizeOrbitalIndexList(value.alpha || []),
      beta: normalizeOrbitalIndexList(value.beta || []),
    };
  }
  return [];
}

function orbitalIndicesMatchNcas(value, ncas) {
  if (isSpinResolvedOrbitalIndices(value)) {
    return Array.isArray(value.alpha) && Array.isArray(value.beta)
      && value.alpha.length === ncas
      && value.beta.length === ncas;
  }
  return Array.isArray(value) && value.length === ncas;
}

function naturalOccupationSummary(orbital) {
  const occupations = orbital && Array.isArray(orbital.natural_orbital_occupations)
    ? orbital.natural_orbital_occupations
    : [];
  return occupations
    .filter((item) => item && item.occupation !== null && typeof item.occupation !== 'undefined')
    .slice(0, 3)
    .map((item) => `${item.natural_orbital_index}:${compactNumber(item.occupation, 3)}`)
    .join(', ');
}

function activeSpaceCandidateSize(candidate) {
  if (!candidate || candidate.ncas === null || typeof candidate.ncas === 'undefined') {
    return Number.POSITIVE_INFINITY;
  }
  const value = Number(candidate.ncas);
  return Number.isFinite(value) && value >= 0 ? value : Number.POSITIVE_INFINITY;
}

function sortedActiveSpaceCandidates(candidates) {
  return [...(Array.isArray(candidates) ? candidates : [])].sort((left, right) => {
    const leftSize = activeSpaceCandidateSize(left);
    const rightSize = activeSpaceCandidateSize(right);
    if (leftSize !== rightSize) {
      return leftSize < rightSize ? -1 : 1;
    }
    return String((left && left.method) || '').localeCompare(String((right && right.method) || ''));
  });
}

function activeSpaceCandidateLabel(candidate) {
  if (!candidate || !candidate.method) {
    return 'n/a';
  }
  return `${candidate.method}: CAS(${candidate.estimated_nelecas ?? '?'}e, ${candidate.ncas ?? '?'}o)`;
}

function activeSpaceApprovalReview(activeSpace, contract) {
  const audit = activeSpace && activeSpace.audit ? activeSpace.audit : {};
  const consistency = audit.ncas_nelecas_consistency || {};
  const recommendation = audit.method_recommendation || {};
  const decision = audit.candidate_decision || {};
  const candidates = Array.isArray(audit.candidate_active_spaces) ? audit.candidate_active_spaces : [];
  const selectedCandidate = candidates.find((item) => item && item.recommended)
    || candidates.find((item) => item && item.method === audit.selected_candidate_method)
    || {};
  const expansionCandidate = candidates.find(
    (item) => item && item.method === decision.expansion_candidate_method,
  ) || {};
  const candidateEvaluation = selectedCandidate.evaluation || {};
  const evidenceCoverage = candidateEvaluation.evidence_coverage || {};
  const selectedOrbitals = Array.isArray(audit.selected_orbitals) ? audit.selected_orbitals : [];
  const ordering = audit.orbital_ordering && typeof audit.orbital_ordering === 'object'
    ? audit.orbital_ordering
    : {};
  const localizationMethod = audit.localization_method || 'none';
  const localizationScope = audit.localization_scope || 'analysis';
  const requestedOrder = Array.isArray(ordering.requested_order) ? ordering.requested_order.join(', ') : '';
  const solverOptions = contract.target_solver_options && typeof contract.target_solver_options === 'object'
    ? contract.target_solver_options
    : (recommendation.target_solver_options || {});
  const targetNroots = Number(solverOptions.nroots || 1);
  const explicitContractIsConsistent = Boolean(
    contract
    && contract.ncas !== null
    && typeof contract.ncas !== 'undefined'
    && contract.nelecas !== null
    && typeof contract.nelecas !== 'undefined'
  );
  return {
    rows: [
      [langText('CAS', 'CAS'), `CAS(${contract.nelecas ?? consistency.final_nelecas ?? activeSpace.nelecas ?? '?'}e, ${contract.ncas ?? consistency.final_ncas ?? activeSpace.ncas ?? '?'}o)`],
      [langText('Orbital indices', '轨道编号'), activeOrbitalIndicesToText(contract.orbital_indices) || selectedOrbitals.map((item) => item.index).join(', ') || 'n/a'],
      [langText('Selection method', '选择方式'), audit.selection_method || activeSpace.selection_method || 'n/a'],
      ['Recommended candidate', selectedCandidate.method || audit.selected_candidate_method || 'n/a'],
      ['Next expansion candidate', expansionCandidate.method ? activeSpaceCandidateLabel(expansionCandidate) : null],
      ['Candidate role', selectedCandidate.role || 'n/a'],
      ['Evidence coverage', Number.isFinite(Number(evidenceCoverage.coverage)) ? `${(Number(evidenceCoverage.coverage) * 100).toFixed(1)}%` : 'n/a'],
      ['Selection confidence', decision.confidence || 'n/a'],
      [langText('Index basis', '编号基准'), audit.orbital_index_basis || 'canonical_mo'],
      [langText('Localization', '局域化'), `${localizationMethod} (${localizationScope})`],
      [langText('Orbital ordering', '轨道排序'), `${ordering.requested_method || 'canonical'}${requestedOrder ? `: ${requestedOrder}` : ''}`],
      [langText('Consistency', '一致性'), consistency.consistent || explicitContractIsConsistent ? langText('passed', '通过') : langText('needs review', '需检查')],
      [langText('Next step', '下一步'), recommendation.recommended_next_step || contract.target_method || 'casscf'],
      [langText('Target solver', '目标求解器'), activeSpaceSolverLabel(contract.target_solver || recommendation.target_solver)],
      [langText('Number of roots', '根数'), solverOptions.nroots],
      [langText('State-average weights', '态平均权重'), targetNroots > 1 && Array.isArray(solverOptions.state_average_weights) ? solverOptions.state_average_weights.join(', ') : null],
    ],
    notes: [
      ...(decision.reason ? [decision.reason] : []),
      ...(decision.expansion_reason ? [decision.expansion_reason] : []),
      ...(Array.isArray(consistency.messages) ? consistency.messages.filter((item) => typeof item === 'string' && item.trim()) : []),
      ...(() => {
        const alternatives = sortedActiveSpaceCandidates(candidates)
          .filter((item) => item
            && item.method !== selectedCandidate.method
            && item.method !== expansionCandidate.method
            && ['available', 'mapping_ambiguous'].includes(item.status))
          .map(activeSpaceCandidateLabel);
        return alternatives.length ? [`Alternative candidates retained: ${alternatives.join('; ')}`] : [];
      })(),
    ],
    orbitals: selectedOrbitals.map((orbital) => ({
      index: orbital.index,
      energy: compactNumber(orbital.energy, 5),
      occupation: compactNumber(orbital.occupation, 3),
      natural: naturalOccupationSummary(orbital),
      t2: compactNumber(orbital.t2_importance, 3),
      reason: Array.isArray(orbital.selection_reasons) ? orbital.selection_reasons.join('; ') : '',
    })),
  };
}

function renderActiveSpaceApprovalReview(review) {
  const node = document.getElementById('approval-active-space-review');
  if (!review) {
    node.classList.add('hidden');
    node.innerHTML = '';
    return;
  }
  const summaryRows = (review.rows || [])
    .filter((row) => row && row[1] !== null && typeof row[1] !== 'undefined' && String(row[1]).trim())
    .map((row) => `
      <div class="active-space-review-item">
        <span class="active-space-review-label">${escapeHtml(row[0])}</span>
        <div class="active-space-review-value">${escapeHtml(row[1])}</div>
      </div>
    `)
    .join('');
  const orbitalColumns = Array.isArray(review.orbital_columns) && review.orbital_columns.length
    ? review.orbital_columns
    : [
      ['index', langText('Orbital', '轨道')],
      ['energy', langText('Energy', '能量')],
      ['occupation', langText('Occ', '占据')],
      ['natural', 'NOON'],
      ['t2', 'max|t2|'],
      ['reason', langText('Reason', '理由')],
    ];
  const orbitalRows = (review.orbitals || []).map((orbital) => `
    <tr>${orbitalColumns.map(([key]) => `<td>${escapeHtml(orbital[key] ?? '')}</td>`).join('')}</tr>
  `).join('');
  const orbitalsHtml = orbitalRows ? `
    <div>
      <h4>${escapeHtml(review.orbitals_title || t('approvalActiveSpaceOrbitalsTitle'))}</h4>
      <div class="active-space-orbital-table-wrap">
        <table class="active-space-orbital-table">
          <thead>
            <tr>${orbitalColumns.map(([, label]) => `<th>${escapeHtml(label)}</th>`).join('')}</tr>
          </thead>
          <tbody>${orbitalRows}</tbody>
        </table>
      </div>
    </div>
  ` : '';
  const notesHtml = (review.notes || []).length
    ? `<ul class="active-space-review-notes">${review.notes.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>`
    : '';
  node.innerHTML = `
    <div>
      <h4>${escapeHtml(review.title || t('approvalActiveSpaceReviewTitle'))}</h4>
      <div class="active-space-review-grid">${summaryRows}</div>
    </div>
    ${notesHtml}
    ${orbitalsHtml}
  `;
  node.classList.remove('hidden');
}

function buildCorrelatedSubspaceApprovalFromResult(payload) {
  const approval = payload && payload.approval && payload.approval.type === 'correlated_subspace'
    ? payload.approval
    : null;
  if (!approval || !approval.task_spec_patch || !approval.correlated_subspace_review) {
    return null;
  }
  const proposal = approval.correlated_subspace_review;
  const audit = proposal.audit && typeof proposal.audit === 'object' ? proposal.audit : {};
  const electronCount = audit.electron_count && typeof audit.electron_count === 'object'
    ? audit.electron_count
    : {};
  const interaction = audit.interaction && typeof audit.interaction === 'object'
    ? audit.interaction
    : {};
  const orbitalRecords = Array.isArray(audit.orbital_records) ? audit.orbital_records : [];
  return {
    type: 'correlated_subspace',
    field: 'embedding',
    source: approval.source || 'correlated_subspace_audit',
    system_message: 'Periodic HF generated a candidate localized correlated subspace for fcDMFT. Review the numerical boundary before execution.',
    draft_value: JSON.stringify(approval.task_spec_patch, null, 2),
    task_spec_patch: approval.task_spec_patch,
    correlated_subspace_review: {
      title: 'Correlated Subspace Review',
      rows: [
        ['Provider', proposal.provider || audit.provider || 'fcdmft'],
        ['Localization', proposal.localization_method || audit.localization_method || 'n/a'],
        ['Localized orbitals', proposal.localized_orbital_count ?? audit.consistency?.selected_orbital_count],
        ['Orbital indices', Array.isArray(audit.orbital_indices) ? audit.orbital_indices.join(', ') : 'n/a'],
        ['Cell electrons', electronCount.periodic_cell],
        ['Subspace electrons', compactNumber(electronCount.correlated_subspace, 5)],
        ['Interaction tensor', interaction.source || interaction.representation || 'n/a'],
        ['Estimated ERI memory', Number.isFinite(Number(proposal.estimated_eri_memory_mb))
          ? `${compactNumber(proposal.estimated_eri_memory_mb, 2)} MB`
          : 'n/a'],
      ],
      notes: Array.isArray(audit.selection_reasons) ? audit.selection_reasons : [],
      orbitals_title: 'Localized Orbitals',
      orbital_columns: [
        ['index', 'Orbital'],
        ['label', 'Label'],
        ['occupation', 'Occupation'],
        ['source', 'Source index'],
        ['fragments', 'Fragments'],
      ],
      orbitals: orbitalRecords.map((orbital) => ({
        index: orbital.orbital_index,
        label: orbital.label || '',
        occupation: compactNumber(orbital.occupation, 5),
        source: orbital.contributions && orbital.contributions.source_localized_orbital_index,
        fragments: Array.isArray(orbital.fragment_ids) ? orbital.fragment_ids.join(', ') : '',
      })),
    },
    can_run_immediately: true,
  };
}

function buildTaskApprovalFromResult(payload) {
  return buildCorrelatedSubspaceApprovalFromResult(payload)
    || buildActiveSpaceApprovalFromResult(payload);
}

function activeSpaceContractRequiresUnrestricted(contract) {
  return Boolean(contract && isSpinResolvedOrbitalIndices(contract.orbital_indices));
}

function casRestrictedValueForActiveSpace(contract) {
  if (activeSpaceContractRequiresUnrestricted(contract)) {
    return false;
  }
  return true;
}

function activeSpaceFormSummary() {
  const enabled = document.getElementById('active-space-enabled').checked;
  const ncas = document.getElementById('ncas').value.trim();
  const nelecas = document.getElementById('nelecas').value.trim();
  const orbitals = document.getElementById('active-orbitals').value.trim();
  const approved = document.getElementById('active-space-approved').checked;
  if (!enabled && !ncas && !nelecas && !orbitals) {
    return t('activeSpaceSummaryEmpty');
  }
  const parts = [];
  if (ncas || nelecas) {
    parts.push(`CAS(${nelecas || '?'}e, ${ncas || '?'}o)`);
  }
  parts.push(approved ? langText('approved', '已确认') : langText('review required', '需确认'));
  return parts.join(' | ');
}

function updateActiveSpaceSummary() {
  const node = document.getElementById('active-space-summary');
  if (node) {
    node.textContent = activeSpaceFormSummary();
  }
}

function syncActiveSpaceMethodControls() {
  const isAvas = document.getElementById('active-space-method').value === 'avas';
  document.getElementById('avas-controls').classList.toggle('hidden', !isAvas);
}

function syncActiveSpaceSolverControls() {
  const select = document.getElementById('active-space-solver');
  const method = document.getElementById('method').value;
  const hint = document.getElementById('active-space-solver-hint');
  const dependency = hint && hint.dataset.optionalDependency ? hint.dataset.optionalDependency : 'block2';
  const dependencyHint = t('block2SolverDependencyHint').replaceAll('{dependency}', dependency);
  const methodHint = t('block2SolverRequiresCas');
  const block2Option = Array.from(select.options).find((option) => option.value === 'block2_dmrg');
  if (block2Option) {
    block2Option.disabled = method !== 'casci' && method !== 'casscf';
    block2Option.title = method !== 'casci' && method !== 'casscf'
      ? `${methodHint} ${dependencyHint}`
      : dependencyHint;
  }
  if (select.value === 'block2_dmrg' && method !== 'casci' && method !== 'casscf') {
    select.value = 'fci';
  }
  select.disabled = method !== 'casci' && method !== 'casscf';
  if (hint) {
    hint.textContent = method !== 'casci' && method !== 'casscf'
      ? `${methodHint} ${dependencyHint}`
      : dependencyHint;
  }
  select.title = hint ? hint.textContent : '';
  const supportsStateTargets = method === 'fci' || method === 'casci' || method === 'casscf';
  document.getElementById('state-target-controls').classList.toggle('hidden', !supportsStateTargets);
  document.getElementById('state-target-nroots').disabled = !supportsStateTargets;
  document.getElementById('state-average-weights').disabled = method !== 'casscf';
  syncOrbitalProcessingControls();
}

function syncOrbitalProcessingControls() {
  const method = document.getElementById('method').value;
  const activeSpaceSolver = document.getElementById('active-space-solver').value;
  const localizationMethod = document.getElementById('localization-method').value;
  const scope = document.getElementById('localization-scope');
  const ordering = document.getElementById('orbital-ordering');
  const manualControl = document.getElementById('manual-orbital-order-control');
  const manualInput = document.getElementById('manual-orbital-order');
  const block2Cas = (method === 'casci' || method === 'casscf') && activeSpaceSolver === 'block2_dmrg';
  const activeSpaceOption = Array.from(scope.options).find((option) => option.value === 'active_space');
  if (activeSpaceOption) {
    activeSpaceOption.disabled = !block2Cas;
  }
  if (!block2Cas && scope.value === 'active_space') {
    scope.value = 'analysis';
  }
  scope.disabled = localizationMethod === 'none';
  ordering.disabled = !block2Cas;
  if (!block2Cas) {
    ordering.value = 'canonical';
  }
  const manualEnabled = block2Cas && ordering.value === 'manual';
  manualControl.classList.toggle('hidden', !manualEnabled);
  manualInput.disabled = !manualEnabled;
  scope.title = block2Cas
    ? ''
    : 'Active-space localization requires block2 DMRG-CASCI/CASSCF.';
  ordering.title = block2Cas
    ? ''
    : 'Fiedler and manual ordering require block2 DMRG-CASCI/CASSCF.';
}

function syncPostCasControls() {
  const checkbox = document.getElementById('sc-nevpt2-enabled');
  const method = document.getElementById('method').value;
  const activeSpaceSolver = document.getElementById('active-space-solver').value;
  const unrestricted = document.getElementById('restricted').value === 'false';
  const supportedReference = (method === 'casci' || method === 'casscf') && !unrestricted && activeSpaceSolver !== 'block2_dmrg';
  if (!supportedReference) {
    checkbox.checked = false;
  }
  checkbox.disabled = !supportedReference;
  checkbox.title = unrestricted && (method === 'casci' || method === 'casscf')
    ? 'SC-NEVPT2 requires a spin-adapted RHF/ROHF CAS reference; unrestricted CAS is not supported.'
    : (activeSpaceSolver === 'block2_dmrg'
      ? 'SC-NEVPT2 is not available for the block2 DMRG active-space provider yet.'
      : (!supportedReference ? 'Select CASCI or CASSCF to enable SC-NEVPT2.' : ''));
  const enabled = checkbox.checked && supportedReference;
  const controls = document.getElementById('post-cas-controls');
  if (controls) {
    controls.classList.toggle('disabled', !enabled);
  }
  document.getElementById('sc-nevpt2-root').disabled = !enabled;
  document.getElementById('sc-nevpt2-density-fit').disabled = !enabled;
}

function isModelHamiltonianCompletionSummary(text) {
  return typeof text === 'string'
    && text.trim().startsWith('Model Hamiltonian ')
    && text.includes(' completed');
}

function appendConversation(role, label, content) {
  if (typeof content !== 'string' || !content.trim()) return;
  const wrapper = document.createElement('div');
  wrapper.className = `message ${role}`;
  wrapper.innerHTML = `<div class="meta">${escapeHtml(label)}</div><div>${escapeHtml(content)}</div>`;
  const conversation = document.getElementById('conversation');
  conversation.appendChild(wrapper);
  conversation.scrollTop = conversation.scrollHeight;
}

function systemMessageHtml(content) {
  return `<div class="meta">${escapeHtml(t('systemLabel'))}</div><div>${escapeHtml(content || '')}</div>`;
}

function normalizeConversation(messages) {
  if (!Array.isArray(messages)) {
    return [];
  }
  return messages.filter((item) => item && typeof item.role === 'string' && typeof item.content === 'string' && item.content.trim());
}

function renderConversation(messages, backendMessages = []) {
  const conversation = document.getElementById('conversation');
  conversation.innerHTML = '';
  normalizeConversation(messages).forEach((item) => {
    const label = labelForRole(item.role);
    appendConversation(item.role === 'assistant' ? 'assistant' : item.role === 'user' ? 'user' : 'system', label, item.content);
  });
  normalizeConversation(backendMessages)
    .filter((item) => item.role !== 'user')
    .filter((item) => !isParsedUserRequestMessage(item))
    .forEach((item) => {
      const label = labelForRole(item.role);
      appendConversation(item.role === 'assistant' ? 'assistant' : 'system', label, item.content);
    });
}

function resetPrepareCache() {
  lastPrepareFingerprint = '';
  lastPreparedPayload = null;
  lastPreparationMeta = null;
}

function langText(enText) {
  return enText;
}

function updateMolecularStructurePreview() {
  renderMolecularStructurePreview('molecular-structure-preview', document.getElementById('atom').value, {
    unit: 'Angstrom',
    emptyText: langText('No molecular structure specified.', '尚未指定分子结构。'),
    parseErrorText: langText('Could not parse molecular coordinates.', '无法解析分子坐标。'),
    atomLabel: langText('atoms', '原子'),
    bondLabel: langText('inferred bonds', '推断化学键'),
    unitLabel: langText('unit', '单位'),
    bondLengthLabel: langText('bond length', '键长'),
  });
}

function labelForRole(role) {
  return role === 'assistant' ? t('assistantLabel') : role === 'user' ? t('userLabel') : t('systemLabel');
}

function isParsedUserRequestMessage(item) {
  return Boolean(item && item.role === 'system' && (item.kind === 'intent' || item.content === '已解析用户请求' || item.content === 'Parsed user request'));
}

function buildPrepareMessages(request) {
  if (!request) {
    return [];
  }
  const latestUserMessage = { role: 'user', content: request };
  if (!pendingStructuredRequest || !Object.keys(pendingStructuredRequest).length) {
    return [latestUserMessage];
  }
  const lastAssistantMessage = [...conversationHistory].reverse().find((item) => item.role === 'assistant');
  if (!lastAssistantMessage) {
    return [latestUserMessage];
  }
  return [
    { role: 'assistant', content: lastAssistantMessage.content },
    latestUserMessage,
  ];
}

function clearApprovalPanel() {
  pendingApproval = null;
  const panel = document.getElementById('approval-panel');
  panel.classList.remove('active');
  document.getElementById('approval-system-message').innerHTML = '';
  document.getElementById('approval-assumptions-title').textContent = t('approvalAssumptionsTitle');
  document.getElementById('approval-assumptions').style.display = 'none';
  document.getElementById('approval-assumptions-list').innerHTML = '';
  renderActiveSpaceApprovalReview(null);
  const draftDetails = document.getElementById('approval-draft-details');
  draftDetails.open = true;
  document.getElementById('approval-draft-summary').classList.add('hidden');
  document.getElementById('approval-draft-summary').textContent = '';
  document.getElementById('approval-atom-title').textContent = t('approvalAtomTitle');
  document.getElementById('approval-atom').value = '';
  document.getElementById('approval-atom').placeholder = t('approvalAtomPlaceholder');
  document.getElementById('approval-atom').spellcheck = true;
  document.getElementById('approval-hint').textContent = '';
}

function clearComputeConfirmPanel() {
  pendingExecutionRequest = '';
  const panel = document.getElementById('compute-confirm-panel');
  panel.classList.remove('ready');
  document.getElementById('confirm-compute').disabled = true;
}

function renderComputeConfirmPanel(message, preparedRequestText) {
  pendingExecutionRequest = preparedRequestText || '';
  const panel = document.getElementById('compute-confirm-panel');
  panel.classList.toggle('ready', !!pendingExecutionRequest);
  document.getElementById('confirm-compute').disabled = !pendingExecutionRequest;
}

function buildActiveSpaceApprovalFromResult(payload) {
  const explicitApproval = payload && payload.approval && payload.approval.type === 'active_space'
    ? payload.approval
    : null;
  let activeSpace = payload
    && payload.structured_results
    && payload.structured_results.active_space;
  let source = explicitApproval && explicitApproval.source
    ? explicitApproval.source
    : 'active_space_audit';
  let targetMethod = explicitApproval && explicitApproval.target_method
    ? explicitApproval.target_method
    : 'casscf';
  let targetSolver = explicitApproval ? explicitApproval.target_solver : null;
  const runtimeContract = explicitApproval && explicitApproval.runtime_contract
    && typeof explicitApproval.runtime_contract === 'object'
    ? explicitApproval.runtime_contract
    : null;
  const probeReferenceStatus = explicitApproval && explicitApproval.probe_reference_status
    && typeof explicitApproval.probe_reference_status === 'object'
    ? explicitApproval.probe_reference_status
    : null;
  let explicitContract = explicitApproval && explicitApproval.active_space_contract
    ? explicitApproval.active_space_contract
    : null;
  if (explicitContract && (!activeSpace || !activeSpace.audit)) {
    activeSpace = { ...explicitContract };
  }
  if (!explicitApproval && (!activeSpace || !activeSpace.audit || activeSpace.approved)) {
    const taskSpec = payload && payload.task_spec && typeof payload.task_spec === 'object'
      ? payload.task_spec
      : null;
    const methodValue = taskSpec && taskSpec.method;
    const methodName = typeof methodValue === 'string'
      ? methodValue
      : (methodValue && methodValue.name);
    const errors = Array.isArray(payload && payload.errors) ? payload.errors : [];
    const validationErrors = Array.isArray(payload && payload.validation_errors)
      ? payload.validation_errors
      : [];
    const approvalBlocked = errors.some((item) => item && item.code === 'active_space_not_approved')
      || validationErrors.some((item) => String(item).includes('active_space.approved=true'));
    const requestedActiveSpace = taskSpec && taskSpec.active_space;
    const hasCasContract = requestedActiveSpace
      && requestedActiveSpace.ncas !== null
      && typeof requestedActiveSpace.ncas !== 'undefined'
      && requestedActiveSpace.nelecas !== null
      && typeof requestedActiveSpace.nelecas !== 'undefined';
    if (!approvalBlocked || !['casci', 'casscf'].includes(String(methodName || '').toLowerCase())
      || !hasCasContract || requestedActiveSpace.approved) {
      return null;
    }
    targetMethod = String(methodName).toLowerCase();
    const solverValue = taskSpec && taskSpec.solver;
    targetSolver = typeof solverValue === 'string'
      ? solverValue
      : (solverValue && solverValue.name);
    source = 'blocked_active_space_contract';
    activeSpace = {
      ...requestedActiveSpace,
      target_solver_options: solverValue && typeof solverValue === 'object' && solverValue.options
        ? solverValue.options
        : {},
      audit: {
        selection_method: requestedActiveSpace.selection_method || 'manual',
        ncas_nelecas_consistency: {
          consistent: true,
          final_ncas: requestedActiveSpace.ncas,
          final_nelecas: requestedActiveSpace.nelecas,
          messages: [],
        },
        method_recommendation: { recommended_next_step: targetMethod },
      },
    };
  }
  const contract = explicitContract || activeSpaceAuditContract(activeSpace);
  if (!contract || contract.ncas === null || typeof contract.ncas === 'undefined' || contract.nelecas === null || typeof contract.nelecas === 'undefined') {
    return null;
  }
  const activeSpaceDraft = {
    enabled: true,
    selection_method: contract.selection_method || 'manual',
    ncas: contract.ncas,
    nelecas: contract.nelecas,
    orbital_indices: contract.orbital_indices || [],
    avas_targets: contract.avas_targets || [],
    avas_threshold: contract.avas_threshold || 0.2,
    initial_mo_coeff: contract.initial_mo_coeff || null,
    target_method: contract.target_method || targetMethod,
    target_solver: contract.target_solver || targetSolver,
    target_solver_options: contract.target_solver_options || {},
    approved: true,
  };
  targetMethod = activeSpaceDraft.target_method || targetMethod;
  targetSolver = activeSpaceDraft.target_solver || targetSolver;
  const casLabel = `CAS(${activeSpaceDraft.nelecas}e, ${activeSpaceDraft.ncas}o)`;
  const assumptions = [];
  if (probeReferenceStatus && probeReferenceStatus.reference_converged === false) {
    const algorithm = runtimeContract && runtimeContract.scf_algorithm === 'newton'
      ? 'Newton SCF'
      : 'the recovered SCF settings';
    assumptions.push(`The probe reference remained unconverged. The approved calculation will retain ${algorithm} and max_cycle=${runtimeContract && runtimeContract.max_cycle ? runtimeContract.max_cycle : 'the recovered value'}.`);
  }
  return {
    type: 'active_space',
    field: 'active_space',
    source,
    target_method: targetMethod,
    target_solver: targetSolver,
    system_message: `${source === 'active_space_audit'
      ? t('approvalActiveSpaceSystemMessage')
      : t('approvalActiveSpaceDirectSystemMessage')} ${casLabel}.`,
    draft_value: JSON.stringify(activeSpaceDraft, null, 2),
    active_space_contract: activeSpaceDraft,
    active_space_review: activeSpaceApprovalReview(activeSpace, activeSpaceDraft),
    runtime_contract: runtimeContract,
    probe_reference_status: probeReferenceStatus,
    assumptions,
    can_run_immediately: true,
  };
}

function renderApprovalPanel(approval) {
  const isStructureApproval = approval && approval.type === 'structure';
  const isActiveSpaceApproval = approval && approval.type === 'active_space';
  const isCorrelatedSubspaceApproval = approval && approval.type === 'correlated_subspace';
  const isNumericalBoundaryApproval = isActiveSpaceApproval || isCorrelatedSubspaceApproval;
  if (!isStructureApproval && !isNumericalBoundaryApproval) {
    clearApprovalPanel();
    return;
  }
  clearComputeConfirmPanel();
  pendingApproval = approval;
  const panel = document.getElementById('approval-panel');
  panel.classList.add('active');
  document.getElementById('approval-system-message').innerHTML = systemMessageHtml(approval.system_message || '');
  const assumptionsPanel = document.getElementById('approval-assumptions');
  document.getElementById('approval-assumptions-title').textContent = isActiveSpaceApproval
    ? t('approvalActiveSpaceEvidenceTitle')
    : (isCorrelatedSubspaceApproval ? 'Selection Evidence' : t('approvalAssumptionsTitle'));
  const assumptionItems = [];
  if (typeof approval.assumption_summary === 'string' && approval.assumption_summary) {
    assumptionItems.push(approval.assumption_summary);
  }
  if (Array.isArray(approval.assumptions)) {
    approval.assumptions.forEach((item) => assumptionItems.push(item));
  }
  if (assumptionItems.length) {
    assumptionsPanel.style.display = 'block';
    renderList('approval-assumptions-list', assumptionItems, '');
  } else {
    assumptionsPanel.style.display = 'none';
    document.getElementById('approval-assumptions-list').innerHTML = '';
  }
  renderActiveSpaceApprovalReview(
    isActiveSpaceApproval
      ? approval.active_space_review
      : (isCorrelatedSubspaceApproval ? approval.correlated_subspace_review : null)
  );
  const draftDetails = document.getElementById('approval-draft-details');
  const draftSummary = document.getElementById('approval-draft-summary');
  draftDetails.open = !isNumericalBoundaryApproval;
  draftSummary.classList.toggle('hidden', !isNumericalBoundaryApproval);
  draftSummary.textContent = isActiveSpaceApproval
    ? t('approvalActiveSpaceJsonEditor')
    : (isCorrelatedSubspaceApproval ? 'Advanced TaskSpec Patch' : '');
  const draftInput = document.getElementById('approval-atom');
  document.getElementById('approval-atom-title').textContent = isActiveSpaceApproval
    ? t('approvalActiveSpaceTitle')
    : (isCorrelatedSubspaceApproval ? 'Correlated Subspace Patch' : t('approvalAtomTitle'));
  draftInput.placeholder = isActiveSpaceApproval
    ? t('approvalActiveSpacePlaceholder')
    : (isCorrelatedSubspaceApproval ? 'Review or edit the embedding and solver patch' : t('approvalAtomPlaceholder'));
  draftInput.spellcheck = !isNumericalBoundaryApproval;
  draftInput.value = approval.draft_value || '';
  document.getElementById('approve-structure').textContent = isActiveSpaceApproval
    ? t('activeSpaceApprove')
    : (isCorrelatedSubspaceApproval ? 'Approve Correlated Subspace' : t('confirmAndContinue'));
  document.getElementById('cancel-structure').textContent = t('cancel');
  const hintParts = [];
  if (approval.source === 'active_space_audit') {
    hintParts.push(t('approvalActiveSpaceSource'));
  } else if (approval.source === 'prepared_active_space_contract') {
    hintParts.push(t('approvalActiveSpacePreparedSource'));
  } else if (approval.source === 'blocked_active_space_contract') {
    hintParts.push(t('approvalActiveSpaceDirectSource'));
  } else if (approval.source === 'correlated_subspace_audit') {
    hintParts.push('Approval records this localized orbital boundary and prepares the fcDMFT task.');
  } else if (approval.source === 'generated_default_structure') {
    hintParts.push(t('approvalGeneratedSource'));
  } else if (approval.source === 'llm_inferred_structure') {
    hintParts.push(t('approvalLlmSource'));
  }
  if (Array.isArray(approval.remaining_missing_fields) && approval.remaining_missing_fields.length) {
    hintParts.push(`${t('approvalNeedsInputAfterConfirm')}: ${approval.remaining_missing_fields.join(', ')}`);
  } else if (isNumericalBoundaryApproval && approval.can_run_immediately) {
    hintParts.push(t('approvalActiveSpaceRunsAfterConfirm'));
  } else if (approval.can_run_immediately) {
    hintParts.push(t('approvalRunsImmediately'));
  }
  document.getElementById('approval-hint').textContent = hintParts.join(' ');
  snapshotCurrentTaskSession();
}

function fingerprintPrepareInput(taskSpec, messages) {
  return JSON.stringify({
    llm_cache_scope: currentPrepareCacheScope || 'llm:unknown',
    task_spec: taskSpec || {},
    messages: messages || [],
  });
}

function appendAssistantMessages(history, messages) {
  const nextHistory = history.slice();
  const existingAssistantContents = new Set(
    nextHistory
      .filter((item) => (item.role === 'assistant' || item.role === 'system') && typeof item.content === 'string' && item.content.trim())
      .map((item) => `${item.role}:${item.content}`)
  );
  normalizeConversation(messages)
    .filter((item) => item.role === 'assistant' || item.role === 'system')
    .forEach((item) => {
      const messageKey = `${item.role}:${item.content}`;
      if (existingAssistantContents.has(messageKey)) {
        return;
      }
      const lastMessage = nextHistory[nextHistory.length - 1];
      if (lastMessage && lastMessage.role === item.role && lastMessage.content === item.content) {
        return;
      }
      nextHistory.push({ role: item.role, content: item.content });
      existingAssistantContents.add(messageKey);
    });
  return nextHistory;
}

function mergeTaskSpec(baseSpec, overrideSpec) {
  return {
    ...(baseSpec || {}),
    ...(overrideSpec || {}),
  };
}

function normalizeTaskFamily(value) {
  if (value === 'model_hamiltonian') return 'model_hamiltonian';
  if (value === 'periodic') return 'periodic';
  return 'molecular';
}

function currentTaskFamily() {
  return normalizeTaskFamily(document.getElementById('task-family').value);
}

function selectedOutputValues() {
  if (currentTaskFamily() === 'molecular') {
    return DEFAULT_MOLECULAR_OUTPUTS.slice();
  }
  if (currentTaskFamily() === 'periodic') {
    const selected = Array.from(document.querySelectorAll('#periodic-output-group input[name="outputs"]:checked'))
      .map((item) => item.value);
    return Array.from(new Set([...DEFAULT_PERIODIC_OUTPUTS, ...selected]));
  }
  const groupId = 'model-output-group';
  return Array.from(document.querySelectorAll(`#${groupId} input[name="outputs"]:checked`)).map((item) => item.value);
}
