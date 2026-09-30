// Transitional accessors keep saved sessions and existing integrations readable.
// Full Study data and a pending execution action have separate ownership.
var plannerState = {
  pendingAction: null,
  get fullPlan() { return currentPlan; },
  set fullPlan(value) { installFullStudyPlan(value); },
  get savedStudy() { return currentSavedStudy; },
  get report() { return currentReport; },
};

function installFullStudyPlan(plan) {
  if (plan && plan._review_kind) throw new Error('A retry subset cannot replace the full Study plan');
  currentPlan = plan;
  if (!plan) plannerState.pendingAction = null;
}

function prettyJson(value) {
  return JSON.stringify(value, null, 2);
}

function setStatus(id, message, kind = '') {
  const target = document.getElementById(id);
  target.textContent = message || '';
  target.className = kind ? `status ${kind}` : 'status';
}

function escapeHtml(value) {
  return AgentUI.escapeHtml(value);
}


function roleLabel(role) {
  if (role === 'user') {
    return 'You';
  }
  if (role === 'trace') {
    return 'Planner Progress';
  }
  if (role === 'system') {
    return 'System';
  }
  return 'Planner Agent';
}

function renderPlannerChat() {
  const target = document.getElementById('planner-chat');
  const visibleMessages = plannerMessages.filter((message) => message.role !== 'trace');
  if (!visibleMessages.length) {
    target.innerHTML = '';
    return;
  }
  target.innerHTML = visibleMessages.map((message) => (
    `<div class="planner-message ${escapeHtml(message.role || 'assistant')}"><span class="role">${escapeHtml(roleLabel(message.role))}</span><span>${escapeHtml(message.content || '')}</span></div>`
  )).join('');
  target.scrollTop = target.scrollHeight;
}

function appendPlannerMessage(role, content) {
  plannerMessages.push({ role, content });
  renderPlannerChat();
  snapshotCurrentTaskSession();
}

function deepCopy(value) {
  if (value === null || value === undefined) {
    return value;
  }
  return JSON.parse(JSON.stringify(value));
}

function currentStudyLifecycle() {
  const candidates = [
    currentReport && currentReport.lifecycle,
    currentAdaptivePreview && currentAdaptivePreview.lifecycle,
    currentAdaptivePreview && currentAdaptivePreview.adaptive
      && currentAdaptivePreview.adaptive.workflow
      && currentAdaptivePreview.adaptive.workflow.lifecycle,
    currentPlan && currentPlan.lifecycle,
  ];
  const lifecycle = candidates.find((item) => item && typeof item === 'object');
  return lifecycle ? deepCopy(lifecycle) : null;
}

function makeTaskSessionId() {
  return AgentUI.makeSessionId('planner-task');
}

function activeTaskSession() {
  return AgentUI.findSession(taskSessions, activeTaskSessionId);
}

function taskStatusLabel(status) {
  if (status === 'planned') return 'planned';
  if (status === 'running') return 'running';
  if (status === 'completed') return 'completed';
  if (status === 'failed') return 'failed';
  if (status === 'review') return 'review';
  return 'draft';
}

function deriveTaskSessionLabel(session) {
  if (session && session.plannerTemplate === HAMILTONIAN_DATASET_TEMPLATE) {
    return session.datasetName || session.datasetId || 'Hamiltonian Dataset';
  }
  const spec = session && session.studySpec ? session.studySpec : {};
  const name = spec.name || spec.objective || 'Planner Study';
  const variableKeys = spec.case_design && spec.case_design.variables
    ? Object.keys(spec.case_design.variables)
    : (spec.sweep && typeof spec.sweep === 'object' ? Object.keys(spec.sweep) : []);
  const suffix = variableKeys.length ? ` / ${variableKeys.join(', ')}` : '';
  return `${name}${suffix}`;
}

function renderTaskSessionSelector() {
  const select = document.getElementById('task-session-select');
  if (!select) {
    return;
  }
  select.innerHTML = AgentUI.sessionOptions(
    taskSessions, activeTaskSessionId, deriveTaskSessionLabel, taskStatusLabel);
}

function renderReviewGate(gate) {
  currentReviewGate = gate && typeof gate === 'object' ? deepCopy(gate) : null;
  const target = document.getElementById('review-gate');
  if (!target) {
    return;
  }
  if (!currentReviewGate) {
    target.className = 'review-gate';
    target.innerHTML = '';
    return;
  }
  document.getElementById('run-study').disabled = true;
  const severity = currentReviewGate.severity === 'error' ? 'error' : 'warning';
  const status = currentReviewGate.status || (severity === 'error' ? 'blocked' : 'requires review');
  const items = Array.isArray(currentReviewGate.items) ? currentReviewGate.items : [];
  const actions = Array.isArray(currentReviewGate.actions) && currentReviewGate.actions.length
    ? currentReviewGate.actions
    : [
        { id: 'focus_table', label: 'Review Table', secondary: false },
        { id: 'acknowledge', label: 'Acknowledge', secondary: true },
      ];
  const costDetails = reviewGateCostDetails(
    currentReviewGate.cost_estimate,
    currentReviewGate.case_ids,
  );
  target.className = `review-gate active ${severity}`;
  target.innerHTML = [
    '<div class="review-gate-header">',
    '<div>',
    `<span class="review-gate-kicker">${escapeHtml(currentReviewGate.kicker || 'Review Required')}</span>`,
    `<strong class="review-gate-title">${escapeHtml(currentReviewGate.title || 'Manual review required')}</strong>`,
    '</div>',
    `<span class="review-gate-status">${escapeHtml(status)}</span>`,
    '</div>',
    currentReviewGate.message ? `<p class="review-gate-message">${escapeHtml(currentReviewGate.message)}</p>` : '',
    items.length ? `<div class="review-gate-list">${items.map((item) => (
      `<div class="review-gate-item"><div><strong>${escapeHtml(item.title || 'case')}</strong><span>${escapeHtml(item.detail || '')}</span></div><span class="adaptive-status ${adaptiveStatusClass(item.status || status)}">${escapeHtml(item.status || status)}</span></div>`
    )).join('')}</div>` : '',
    costDetails ? `<div class="review-gate-cost"><strong>Estimated Cost</strong><span>${escapeHtml(costDetails.summary)}</span>${costDetails.reason ? `<p>${escapeHtml(costDetails.reason)}</p>` : ''}</div>` : '',
    currentReviewGate.hint ? `<p class="review-gate-hint">${escapeHtml(currentReviewGate.hint)}</p>` : '',
    `<div class="review-gate-actions">${actions.map((action) => (
      `<button type="button" class="${action.secondary ? 'secondary' : ''}" data-review-action="${escapeHtml(action.id || '')}">${escapeHtml(action.label || action.id || 'Action')}</button>`
    )).join('')}</div>`,
  ].join('');
}

function focusPlanReviewForGate(gate) {
  if (gate && gate.kind === 'active_space_approval') {
    const payload = currentAdaptivePreview || currentReport || {};
    const adaptive = payload.adaptive && typeof payload.adaptive === 'object' ? payload.adaptive : {};
    const approvalPlan = adaptivePlanByKind(adaptive, gate.plan_kind || 'refined');
    if (approvalPlan) {
      renderPlanSummary(approvalPlan);
      currentAnalysisRows = renderPlanCaseTable(
        approvalPlan,
        currentAdaptivePreview && currentAdaptivePreview.adaptive ? adaptiveDecisionEntries(currentAdaptivePreview) : [],
        currentReport && Array.isArray(currentReport.comparison_table) ? currentReport.comparison_table : [],
        currentReport && Array.isArray(currentReport.cases) ? currentReport.cases : [],
      );
    }
  }
  document.getElementById('case-table').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function studyStateForReviewAction() {
  const state = currentReport || currentAdaptivePreview;
  if (state && typeof state === 'object') {
    return deepCopy(state);
  }
  return currentPlan && currentPlan.lifecycle
    ? { lifecycle: deepCopy(currentPlan.lifecycle) }
    : {};
}

function installStudyReviewActionResult(result) {
  const subset = result && (result.kind === 'retry_cases' ||
    (result.workflow && result.workflow.scope === 'subset') ||
    (result.plan && result.plan._review_kind));
  if (subset) {
    plannerState.pendingAction = deepCopy(result.retry_action || result);
    if (typeof renderRetryPreview === 'function') renderRetryPreview();
    document.getElementById('run-study').disabled = !result.can_run;
    return;
  }
  const state = result && result.study_state && typeof result.study_state === 'object'
    ? deepCopy(result.study_state)
    : null;
  if (state) {
    if (currentReport) {
      currentReport = deepCopy(state);
    }
    if (currentAdaptivePreview || !currentReport) {
      currentAdaptivePreview = deepCopy(state);
    }
  }
  if (result && result.plan && typeof result.plan === 'object') {
    installFullStudyPlan(deepCopy(result.plan));
  }
  if (result && result.study_spec_patch && typeof result.study_spec_patch === 'object') {
    try {
      const spec = parseStudySpec();
      Object.entries(result.study_spec_patch).forEach(([key, value]) => {
        spec[key] = deepCopy(value);
      });
      document.getElementById('study-spec').value = prettyJson(spec);
      updateTaskSetupPanel(spec);
    } catch (_error) {
      // The approved backend plan remains authoritative even if the editable
      // text view cannot be refreshed.
    }
  }
  if (currentPlan) {
    renderPlanSummary(currentPlan);
    const source = currentAdaptivePreview || currentReport || {};
    currentAnalysisRows = renderPlanCaseTable(
      currentPlan,
      source.adaptive ? adaptiveDecisionEntries(source) : [],
      currentReport && Array.isArray(currentReport.comparison_table) ? currentReport.comparison_table : [],
      currentReport && Array.isArray(currentReport.cases) ? currentReport.cases : [],
    );
  }
  const source = currentAdaptivePreview || currentReport;
  renderAdaptiveState(source);
  renderReviewGate(result && result.status === 'review_required'
    ? reviewGateForStudyReport(source)
    : null);
  document.getElementById('run-study').disabled = !(result && result.can_run);
}

async function applyPlannerReviewAction(action, caseIds, gate) {
  const reviewGate = gate || currentReviewGate || {};
  const approvalToken = reviewGate.approval_token || null;
  setStatus('run-status', 'Applying review action...');
  if (!currentSavedStudy) throw new Error('Build Plan to save the Study before reviewing it.');
  const result = await postJson('/api/study-review-action', {
    ...savedStudyRequest(),
    action_id: action,
    case_ids: Array.isArray(caseIds) ? caseIds : [],
    approval_token: approvalToken,
    plan_kind: reviewGate.plan_kind || reviewGate.active_plan_kind || null,
  });
  await refreshSavedStudy();
  appendPlannerMessage('system', result.message || 'Review action applied.');
  setStatus(
    'run-status',
    result.message || 'Review action applied.',
    result.status === 'review_required' || result.status === 'prepared' ? 'ok' : '',
  );
  snapshotCurrentTaskSession();
  return result;
}

async function handleReviewGateAction(event) {
  const button = event.target.closest('[data-review-action]');
  if (!button || !currentReviewGate) {
    return;
  }
  const action = button.getAttribute('data-review-action');
  const actionConfig = (Array.isArray(currentReviewGate.actions) ? currentReviewGate.actions : [])
    .find((item) => item && item.id === action);
  const reviewCaseIds = actionConfig && Array.isArray(actionConfig.case_ids)
    ? actionConfig.case_ids
    : (Array.isArray(currentReviewGate.case_ids) ? currentReviewGate.case_ids : []);
  if (action === 'focus_table') {
    focusPlanReviewForGate(currentReviewGate);
    const guidance = reviewPlanTableGuidance(currentReviewGate, currentReport || currentAdaptivePreview);
    appendPlannerMessage('assistant', guidance);
    setStatus('run-status', 'Plan Review focused. Suggested recovery options were added to Planner Conversation.', 'ok');
    snapshotCurrentTaskSession();
    return;
  }
  const backendReviewActions = new Set([
    'increase_recovery_max_cycle',
    'increase_dmet_iterations',
    'retry_dmet_without_impurity_diis',
    'try_fci_recovery',
    'expand_active_space',
    'promote_casscf_recovery',
    'approve_active_space',
    'cancel_active_space_review',
    'approve_cost_estimate',
    'approve_path_restart',
    'skip_path_restart',
    'cancel_path_restart_review',
    'approve_mps_continuation',
    'skip_mps_continuation',
    'cancel_mps_continuation_review',
  ]);
  if (backendReviewActions.has(action)) {
    const gate = deepCopy(currentReviewGate);
    try {
      await applyPlannerReviewAction(action, reviewCaseIds, gate);
    } catch (error) {
      const message = errorDetailMessage(error.payload || {}, error.message);
      appendPlannerMessage('system', message);
      setStatus('run-status', message, 'error');
    }
    return;
  }
  if (action === 'show_case_guidance') {
    appendPlannerMessage('assistant', reviewPlanTableGuidance(currentReviewGate, currentReport || currentAdaptivePreview));
    setStatus('run-status', 'Case-specific recovery guidance was added to Planner Conversation.', 'ok');
    snapshotCurrentTaskSession();
    return;
  }
  if (action === 'check_execution_status') {
    inspectCurrentExecution();
    return;
  }
  if (action === 'collect_execution_results') {
    collectCurrentExecution();
    return;
  }
  if (action === 'cancel_execution_review') {
    renderReviewGate(null);
    appendPlannerMessage('system', 'Remote execution review closed. The submitted jobs and execution receipt were kept.');
    setStatus('run-status', 'Execution remains recoverable from Planner Conversation.', 'ok');
    snapshotCurrentTaskSession();
    return;
  }
  if (action === 'acknowledge') {
    const title = currentReviewGate.title || 'Manual review required';
    renderReviewGate(null);
    appendPlannerMessage('system', `${title} acknowledged.`);
    setStatus('run-status', 'Review acknowledged. Adjust settings or rerun when ready.', 'ok');
    snapshotCurrentTaskSession();
  }
}

function clearPlannerTaskViews({ preserveMessages = false } = {}) {
  plannerState.pendingAction = null;
  if (typeof closeRetryEditor === 'function') closeRetryEditor();
  currentSavedStudy = null;
  installFullStudyPlan(null);
  currentReport = null;
  currentAdaptivePreview = null;
  currentExecution = null;
  currentAnalysisRows = [];
  currentPlotSpecs = [];
  currentPostprocessArtifacts = [];
  artifactPreviewUrls.forEach((url) => URL.revokeObjectURL(url));
  artifactPreviewUrls = [];
  document.getElementById('run-study').disabled = true;
  document.getElementById('analyze-study-results').disabled = true;
  renderPlanSummary(null);
  document.getElementById('case-table').className = 'empty';
  document.getElementById('case-table').innerHTML = 'Build a plan to see cases.';
  renderReviewGate(null);
  renderAdaptiveState(null);
  document.getElementById('report-summary').className = 'empty';
  document.getElementById('report-summary').innerHTML = 'No run yet.';
  document.getElementById('study-result-analysis').className = 'empty analysis-box';
  document.getElementById('study-result-analysis').textContent = 'Run first.';
  setStatus('draft-status', '');
  setStatus('run-status', '');
  setStatus('analysis-status', '');
  resetPostprocessing();
  if (!preserveMessages) {
    plannerMessages = [];
    renderPlannerChat();
  }
}

function snapshotCurrentTaskSession() {
  const session = activeTaskSession();
  if (!session || isRestoringTaskSession) {
    return;
  }
  let studySpec = {};
  try {
    studySpec = parseStudySpec();
  } catch (_error) {
    studySpec = {};
  }
  session.studySpec = deepCopy(studySpec);
  session.workDir = document.getElementById('work-dir').value;
  session.workDirLocked = Boolean(document.getElementById('work-dir').disabled);
  session.workDirRoot = session.workDirLocked
    ? (session.workDirRoot || DEFAULT_WORK_DIR)
    : effectiveWorkDir();
  session.studySystem = document.getElementById('study-system').value;
  session.studyMode = document.getElementById('study-mode').value;
  session.plannerTemplate = document.getElementById('planner-template').value;
  session.datasetId = document.getElementById('dataset-id').value;
  session.datasetName = document.getElementById('dataset-name').value;
  session.datasetSplitProtocol = document.getElementById('dataset-split-protocol').value;
  session.datasetSplitSeed = document.getElementById('dataset-split-seed').value;
  session.datasetSeedGeometries = deepCopy(currentDatasetSeedGeometries) || [];
  session.datasetSeedFileName = currentDatasetSeedFileName;
  session.adaptiveInitialScanStrategy = document.getElementById('adaptive-initial-scan-strategy').value;
  session.adaptiveActiveSpaceSolver = document.getElementById('adaptive-active-space-solver').value;
  session.adaptiveActiveSpaceLocalization = document.getElementById('adaptive-active-space-localization').value;
  session.adaptiveOrbitalOrdering = document.getElementById('adaptive-orbital-ordering').value;
  session.adaptiveManualOrbitalOrder = document.getElementById('adaptive-manual-orbital-order').value;
  session.modelHamiltonianInputFile = document.getElementById('model-hamiltonian-input-file').value;
  session.currentRunId = currentRunId;
  session.currentPlan = deepCopy(currentPlan);
  session.currentReport = deepCopy(currentReport);
  session.currentAdaptivePreview = deepCopy(currentAdaptivePreview);
  session.currentExecution = deepCopy(currentExecution);
  session.currentSavedStudy = deepCopy(currentSavedStudy);
  session.currentAnalysisRows = deepCopy(currentAnalysisRows) || [];
  session.currentPlotSpecs = deepCopy(currentPlotSpecs) || [];
  session.currentPostprocessArtifacts = deepCopy(currentPostprocessArtifacts) || [];
  session.plannerMessages = deepCopy(plannerMessages) || [];
  const lifecycle = currentStudyLifecycle();
  const lifecycleStage = lifecycle && lifecycle.stage ? String(lifecycle.stage) : '';
  if (currentExecution) {
    session.status = 'running';
  } else if ((currentReport && currentReport.pending_review) || lifecycleStage === 'review_required' || lifecycleStage === 'retry_ready') {
    session.status = 'review';
  } else if (['executing', 'refining', 'analyzing'].includes(lifecycleStage) || currentExecution) {
    session.status = 'running';
  } else if (lifecycleStage === 'completed') {
    session.status = currentReport && currentReport.status === 'succeeded' ? 'completed' : 'failed';
  } else if (currentReport) {
    session.status = currentReport.status === 'succeeded' ? 'completed' : 'failed';
  } else if (currentPlan) {
    session.status = 'planned';
  } else {
    session.status = session.status || 'draft';
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
    artifactPreviewUrls.forEach((url) => URL.revokeObjectURL(url));
    artifactPreviewUrls = [];
    document.getElementById('work-dir').value = session.workDir || '';
    document.getElementById('work-dir').disabled = Boolean(session.workDirLocked);
    document.getElementById('planner-template').value = session.plannerTemplate || 'standard';
    document.getElementById('dataset-id').value = session.datasetId || 'qh9-small-1000';
    document.getElementById('dataset-name').value = session.datasetName || 'QH9-compatible 1000-structure dataset';
    document.getElementById('dataset-split-protocol').value = session.datasetSplitProtocol || 'molecule_random';
    document.getElementById('dataset-split-seed').value = session.datasetSplitSeed || '0';
    document.getElementById('dataset-seed-file').value = '';
    currentDatasetSeedGeometries = deepCopy(session.datasetSeedGeometries) || [];
    currentDatasetSeedFileName = session.datasetSeedFileName || '';
    document.getElementById('study-system').value = normalizeSystemType(session.studySystem || (session.studySpec && session.studySpec.system_type) || 'model_hamiltonian');
    document.getElementById('study-mode').value = session.studyMode || 'static';
    document.getElementById('adaptive-initial-scan-strategy').value = normalizedInitialScanStrategy(session.adaptiveInitialScanStrategy);
    document.getElementById('adaptive-active-space-solver').value = session.adaptiveActiveSpaceSolver || 'auto';
    document.getElementById('adaptive-active-space-localization').value = session.adaptiveActiveSpaceLocalization || 'none';
    document.getElementById('adaptive-orbital-ordering').value = session.adaptiveOrbitalOrdering || 'canonical';
    document.getElementById('adaptive-manual-orbital-order').value = session.adaptiveManualOrbitalOrder || '';
    syncStudyModeControls();
    syncAdaptiveActiveSpaceControls();
    syncDatasetTemplateControls();
    document.getElementById('study-spec').value = prettyJson(session.studySpec || DEFAULT_STUDY_SPEC);
    document.getElementById('model-hamiltonian-input-file').value = session.modelHamiltonianInputFile || '';
    updatePlannerComposerForSystem(session.studySystem || (session.studySpec && session.studySpec.system_type));
    plannerMessages = deepCopy(session.plannerMessages) || [];
    updateEffectiveWorkDir();
    resetModelPreview();
    if (session.modelHamiltonianInputFile) {
      previewModelHamiltonianStructure();
    }
    clearPlannerTaskViews({ preserveMessages: true });
    updateTaskSetupPanel(session.studySpec || DEFAULT_STUDY_SPEC);
    currentRunId = session.currentRunId || '';
    installFullStudyPlan(session.currentPlan && !session.currentPlan._review_kind ? deepCopy(session.currentPlan) : null);
    currentReport = deepCopy(session.currentReport) || null;
    currentAdaptivePreview = deepCopy(session.currentAdaptivePreview) || null;
    currentExecution = deepCopy(session.currentExecution) || null;
    currentSavedStudy = deepCopy(session.currentSavedStudy) || null;
    const pendingRetry = currentReport && currentReport.pending_review;
    plannerState.pendingAction = pendingRetry && pendingRetry.kind === 'retry_cases'
      ? deepCopy(pendingRetry.retry_action) : null;
    if (typeof renderRetryPreview === 'function') renderRetryPreview();
    currentAnalysisRows = deepCopy(session.currentAnalysisRows) || [];
    const restoredPlotSpecs = deepCopy(session.currentPlotSpecs) || [];
    const restoredPostprocessArtifacts = deepCopy(session.currentPostprocessArtifacts) || [];
    currentPlotSpecs = restoredPlotSpecs;
    currentPostprocessArtifacts = restoredPostprocessArtifacts;
    if (currentPlan) {
      renderPlan(currentPlan);
      document.getElementById('run-study').disabled = true;
    }
    if (currentReport) {
      renderReport(currentReport);
    } else {
      renderAdaptiveState(currentAdaptivePreview);
      if (currentExecution) {
        renderExecutionRecovery(currentExecution.execution || currentExecution);
      }
    }
    if (restoredPlotSpecs.length) {
      renderPlotSpecs(restoredPlotSpecs);
    }
    if (restoredPostprocessArtifacts.length) {
      renderPostprocessArtifacts(restoredPostprocessArtifacts);
    }
    renderPlannerChat();
  } finally {
    isRestoringTaskSession = false;
  }
  renderTaskSessionSelector();
  if (typeof syncSavedStudyControls === 'function') syncSavedStudyControls();
}

function createTaskSession(options = {}) {
  if (options.snapshotExisting !== false) {
    snapshotCurrentTaskSession();
  }
  const previousSession = activeTaskSession();
  const inheritedWorkDir = options.inheritWorkDir === false
    ? ''
    : ((previousSession && previousSession.workDirRoot) || document.getElementById('work-dir').value);
  const systemType = normalizeSystemType(options.systemType || document.getElementById('study-system').value);
  const studySpec = deepCopy(options.studySpec || blankStudySpecForSystem(systemType));
  const session = {
    id: makeTaskSessionId(),
    status: 'draft',
    createdAt: Date.now(),
    updatedAt: Date.now(),
    studySpec,
    plannerTemplate: options.plannerTemplate || document.getElementById('planner-template').value || 'standard',
    studySystem: systemType,
    studyMode: options.studyMode || document.getElementById('study-mode').value || (systemType === 'molecular' ? 'adaptive' : 'static'),
    adaptiveInitialScanStrategy: options.adaptiveInitialScanStrategy || 'auto',
    adaptiveActiveSpaceSolver: options.adaptiveActiveSpaceSolver || document.getElementById('adaptive-active-space-solver').value || 'auto',
    adaptiveActiveSpaceLocalization: options.adaptiveActiveSpaceLocalization || document.getElementById('adaptive-active-space-localization').value || 'none',
    adaptiveOrbitalOrdering: options.adaptiveOrbitalOrdering || document.getElementById('adaptive-orbital-ordering').value || 'canonical',
    adaptiveManualOrbitalOrder: options.adaptiveManualOrbitalOrder || document.getElementById('adaptive-manual-orbital-order').value || '',
    workDir: inheritedWorkDir || '',
    workDirRoot: inheritedWorkDir || '',
    workDirLocked: false,
    modelHamiltonianInputFile: '',
    datasetId: options.datasetId || 'qh9-small-1000',
    datasetName: options.datasetName || 'QH9-compatible 1000-structure dataset',
    datasetSplitProtocol: options.datasetSplitProtocol || 'molecule_random',
    datasetSplitSeed: String(options.datasetSplitSeed || '0'),
    datasetSeedGeometries: deepCopy(options.datasetSeedGeometries) || [],
    datasetSeedFileName: options.datasetSeedFileName || '',
    currentRunId: '',
    currentPlan: null,
    currentReport: null,
    currentAdaptivePreview: null,
    currentExecution: null,
    currentAnalysisRows: [],
    currentPlotSpecs: [],
    currentPostprocessArtifacts: [],
    plannerMessages: [],
  };
  taskSessions.push(session);
  activeTaskSessionId = session.id;
  restoreTaskSession(session);
  snapshotCurrentTaskSession();
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
  if (currentSavedStudy) refreshSavedStudy();
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

function errorDetailMessage(data, fallback) {
  const pieces = [fallback || data.error || 'Request failed'];
  const issues = Array.isArray(data.validation_issues) ? data.validation_issues : [];
  if (issues.length) {
    const issueLines = issues
      .filter((item) => item && item.severity === 'error')
      .slice(0, 6)
      .map((item) => `${item.path || item.code}: ${item.message || item.code}`);
    if (issueLines.length) {
      pieces.push(`Validation details:\n${issueLines.join('\n')}`);
    }
  }
  return pieces.join('\n');
}

function plannerClarificationMessage(payload) {
  const issues = Array.isArray(payload.validation_issues) ? payload.validation_issues : [];
  const issueLines = issues
    .filter((item) => item && item.severity === 'error')
    .slice(0, 6)
    .map((item) => `- ${item.path || item.code}: ${item.message || item.code}`);
  if (!issueLines.length) {
    return 'I drafted a plan, but need a bit more detail before approval.';
  }
  return [
    'I need a few fixes before approval:',
    issueLines.join('\n'),
    'Send the missing details and I will revise it.',
  ].join('\n');
}
