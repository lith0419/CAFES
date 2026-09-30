const CALCULATION_POLL_INTERVAL_MS = 1000;

function calculationDelay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function updateCurrentTaskLifecycle(payload) {
  if (payload && payload.lifecycle && typeof payload.lifecycle === 'object') {
    currentTaskLifecycle = JSON.parse(JSON.stringify(payload.lifecycle));
  }
  return currentTaskLifecycle;
}

function markTaskConfigurationChanged(changedFields = ['task_spec']) {
  if (!currentTaskLifecycle || typeof currentTaskLifecycle !== 'object') return;
  const source = String(currentTaskLifecycle.stage || 'draft');
  if (['queued', 'running'].includes(source)) return;
  const history = Array.isArray(currentTaskLifecycle.history)
    ? currentTaskLifecycle.history.slice()
    : [];
  const revision = Number.parseInt(currentTaskLifecycle.revision, 10) || 1;
  const nextRevision = revision + 1;
  const at = new Date().toISOString();
  history.push({
    sequence: history.length + 1,
    event: 'configuration_changed',
    from_stage: source,
    to_stage: 'draft',
    revision: nextRevision,
    at,
    details: { changed_fields: changedFields.slice() },
  });
  currentTaskLifecycle = {
    ...currentTaskLifecycle,
    revision: nextRevision,
    stage: 'draft',
    terminal: false,
    current_event: 'configuration_changed',
    updated_at: at,
    history,
  };
}

async function postCalculationLifecycle(path, body) {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  let payload = {};
  try {
    payload = await response.json();
  } catch (error) {
    payload = {};
  }
  if (!response.ok) {
    throw new Error(payload.error || `${path} failed with HTTP ${response.status}`);
  }
  return { response, payload };
}

async function applyPendingApprovalLifecycle(action, reviewType) {
  const { payload } = await postCalculationLifecycle('/api/task-review-action', {
    action,
    review_type: reviewType,
    lifecycle: currentTaskLifecycle,
  });
  updateCurrentTaskLifecycle(payload);
  return currentTaskLifecycle;
}

function updateStatusButtonState() {
  const button = document.getElementById('refresh-status');
  if (!button) return;
  button.disabled = !activeRunHandle || runStatusRefreshInProgress;
  button.textContent = runStatusRefreshInProgress ? t('refreshingStatus') : t('refreshStatus');
}

function clearActiveRunContext() {
  activeRunHandle = null;
  activeRunTarget = '';
  activeRunPreparedRequestText = '';
  activeRunLastState = '';
  updateStatusButtonState();
}

function jobStateLabel(state) {
  const labels = {
    queued: t('statusQueued'),
    running: t('statusRunning'),
    completed: t('statusCompleted'),
    failed: t('statusFailed'),
    cancelled: t('statusCancelled'),
  };
  return labels[state] || state || t('statusNotRun');
}

function recordRunStatus(status) {
  const state = String((status && status.state) || activeRunLastState || 'unknown');
  activeRunLastState = state;
  renderStatus(state);
  if (['queued', 'running'].includes(state)) {
    updateActiveTaskSessionStatus('running');
  } else if (['completed', 'failed', 'cancelled'].includes(state)) {
    updateActiveTaskSessionStatus(state);
  }
  snapshotCurrentTaskSession();
  updateStatusButtonState();
  return state;
}

function appendRunStatusMessage(status) {
  const state = String((status && status.state) || activeRunLastState || 'unknown');
  const message = String((status && status.message) || '').trim();
  const content = t('runStatusUpdated')
    .replace('{state}', jobStateLabel(state))
    .replace('{message}', message);
  conversationHistory = appendAssistantMessages(conversationHistory, [{ role: 'system', content }]);
  renderConversation(conversationHistory);
}

function calculationControlElements() {
  const elements = Array.from(document.querySelectorAll(
    '.task-setup-card input, .task-setup-card select, .task-setup-card textarea, .task-setup-card button'
  ));
  ['run', 'confirm-compute', 'task-session-select', 'new-task', 'execution-target', 'resource-profile']
    .forEach((id) => {
      const element = document.getElementById(id);
      if (element && !elements.includes(element)) elements.push(element);
    });
  return elements;
}

function setCalculationControls(isCalculating, isStopping = false) {
  const clearButton = document.getElementById('clear');
  if (isCalculating) {
    if (!calculationControlsSnapshot) {
      calculationControlsSnapshot = calculationControlElements().map((element) => ({
        element,
        disabled: element.disabled,
      }));
    }
    calculationControlsSnapshot.forEach(({ element }) => { element.disabled = true; });
    document.getElementById('run').textContent = t('run');
    document.getElementById('confirm-compute').textContent = t('confirmCompute');
    document.getElementById('analyze-result').disabled = true;
    document.getElementById('analyze-result').textContent = t('analyzeResult');
    clearButton.disabled = false;
    clearButton.textContent = isStopping ? t('stopping') : t('stop');
    clearButton.classList.add('stop-action');
    renderStatus('calculating');
    updateStatusButtonState();
    return;
  }
  if (calculationControlsSnapshot) {
    calculationControlsSnapshot.forEach(({ element, disabled }) => { element.disabled = disabled; });
    calculationControlsSnapshot = null;
  }
  document.getElementById('run').disabled = false;
  document.getElementById('run').textContent = t('run');
  document.getElementById('confirm-compute').disabled = !pendingExecutionRequest;
  document.getElementById('confirm-compute').textContent = t('confirmCompute');
  clearButton.disabled = false;
  clearButton.textContent = t('clear');
  clearButton.classList.remove('stop-action');
  setAnalyzeState(false);
  updateStatusButtonState();
}

function updateRunDirectoryFromHandle(handle) {
  if (!handle || typeof handle !== 'object') return;
  const workDir = String(handle.work_dir || '').replace(/\/$/, '');
  const runId = String(handle.run_id || '');
  if (!workDir || !runId) return;
  document.getElementById('work-dir').value = workDir.endsWith(`/${runId}`)
    ? workDir
    : `${workDir}/${runId}`;
  setWorkDirLocked(true);
}

function markActiveRunCancelled() {
  if (activeRunCancelledNotified) return;
  activeRunCancelledNotified = true;
  activeRunLastState = 'cancelled';
  conversationHistory = appendAssistantMessages(conversationHistory, [
    { role: 'system', content: t('runCancelled') },
  ]);
  renderConversation(conversationHistory);
  renderStatus('cancelled');
  updateActiveTaskSessionStatus('cancelled');
  snapshotCurrentTaskSession();
}

async function requestActiveRunCancellation() {
  if (!activeRunHandle) return false;
  if (activeRunCancelPromise) return activeRunCancelPromise;
  const handle = activeRunHandle;
  const target = activeRunTarget;
  activeRunCancelPromise = (async () => {
    const { payload } = await postCalculationLifecycle('/api/run-cancel', {
      handle,
      execution_target: target,
      lifecycle: currentTaskLifecycle,
    });
    updateCurrentTaskLifecycle(payload);
    const state = payload.status && payload.status.state;
    if (state) activeRunLastState = state;
    if (state === 'cancelled') {
      markActiveRunCancelled();
      return true;
    }
    activeRunStopRequested = false;
    setCalculationControls(true, false);
    return false;
  })();
  try {
    return await activeRunCancelPromise;
  } catch (error) {
    activeRunStopRequested = false;
    setCalculationControls(true, false);
    conversationHistory = appendAssistantMessages(conversationHistory, [{
      role: 'system',
      content: `${t('runCancellationFailed')}: ${error instanceof Error ? error.message : String(error)}`,
    }]);
    renderConversation(conversationHistory);
    return false;
  } finally {
    activeRunCancelPromise = null;
  }
}

async function stopActiveCalculation() {
  if (!calculationInProgress) return false;
  activeRunStopRequested = true;
  setCalculationControls(true, true);
  if (!activeRunHandle) return true;
  return requestActiveRunCancellation();
}

async function waitForSubmittedRun(handle, executionTarget) {
  while (true) {
    if (activeRunStopRequested) await requestActiveRunCancellation();
    const { payload } = await postCalculationLifecycle('/api/run-status', {
      handle,
      execution_target: executionTarget,
      lifecycle: currentTaskLifecycle,
    });
    updateCurrentTaskLifecycle(payload);
    const status = payload.status || {};
    recordRunStatus(status);
    if (status.state === 'cancelled') {
      markActiveRunCancelled();
      return null;
    }
    if (status.state === 'failed') {
      throw new Error(status.message || t('runFailed'));
    }
    if (status.state === 'completed' && status.report_available) return status;
    await calculationDelay(CALCULATION_POLL_INTERVAL_MS);
  }
}

async function collectSubmittedRun(handle, preparedRequestText, executionTarget) {
  while (true) {
    const { response, payload } = await postCalculationLifecycle('/api/run-collect', {
      handle,
      prepared_request: preparedRequestText,
      locale: currentLanguage,
      execution_target: executionTarget,
      lifecycle: currentTaskLifecycle,
    });
    updateCurrentTaskLifecycle(payload);
    if (response.status !== 202) return payload;
    await calculationDelay(CALCULATION_POLL_INTERVAL_MS);
  }
}

function collectedPayloadMatchesHandle(handle) {
  return Boolean(
    handle &&
    lastExecutionPayload &&
    lastExecutionPayload.run_id &&
    lastExecutionPayload.run_id === handle.run_id
  );
}

function applyCollectedRunPayload(payload, preparedRequestText, expectedRunId = '') {
  if (payload.run_id && expectedRunId && payload.run_id !== expectedRunId) {
    throw new Error(`Run report mismatch: expected ${expectedRunId}, received ${payload.run_id}.`);
  }
  clearApprovalPanel();
  clearComputeConfirmPanel();
  const completionMessage = payload.execution_status === 'succeeded' ? t('runCompleted') : t('runEnded');
  conversationHistory = appendAssistantMessages(conversationHistory, [{ role: 'system', content: completionMessage }]);
  conversationHistory = appendAssistantMessages(conversationHistory, payload.messages || []);
  renderConversation(conversationHistory);
  lastExecutionPayload = payload;
  updateCurrentTaskLifecycle(payload);
  lastExecutedRequestText = preparedRequestText || '';
  document.getElementById('input-preview').textContent = payload.generated_input || '';
  renderSummary(payload);
  renderApprovalPanel(buildTaskApprovalFromResult(payload));
  pendingStructuredRequest = {};
  resetPrepareCache();
  updateRequestPlaceholder(null);
  document.getElementById('request').value = '';
  const finalState = payload.execution_status === 'succeeded' ? 'completed' : 'failed';
  activeRunLastState = finalState;
  renderStatus(finalState);
  updateActiveTaskSessionStatus(finalState);
  snapshotCurrentTaskSession();
}

async function refreshActiveRunStatus() {
  if (!activeRunHandle) {
    conversationHistory = appendAssistantMessages(conversationHistory, [
      { role: 'system', content: t('runStatusUnavailable') },
    ]);
    renderConversation(conversationHistory);
    updateStatusButtonState();
    return false;
  }
  if (runStatusRefreshInProgress) return false;
  runStatusRefreshInProgress = true;
  updateStatusButtonState();
  const handle = activeRunHandle;
  const executionTarget = activeRunTarget;
  try {
    const { payload } = await postCalculationLifecycle('/api/run-status', {
      handle,
      execution_target: executionTarget,
      lifecycle: currentTaskLifecycle,
    });
    updateCurrentTaskLifecycle(payload);
    const status = payload.status || {};
    const state = recordRunStatus(status);
    appendRunStatusMessage(status);
    const shouldCollect = (
      !calculationInProgress &&
      status.report_available &&
      ['completed', 'failed'].includes(state) &&
      activeRunPreparedRequestText &&
      !collectedPayloadMatchesHandle(handle)
    );
    if (shouldCollect) {
      const report = await collectSubmittedRun(handle, activeRunPreparedRequestText, executionTarget);
      applyCollectedRunPayload(report, activeRunPreparedRequestText, handle.run_id || '');
    }
    return true;
  } catch (error) {
    conversationHistory = appendAssistantMessages(conversationHistory, [{
      role: 'system',
      content: `${t('runStatusRefreshFailed')}: ${error instanceof Error ? error.message : String(error)}`,
    }]);
    renderConversation(conversationHistory);
    snapshotCurrentTaskSession();
    return false;
  } finally {
    runStatusRefreshInProgress = false;
    snapshotCurrentTaskSession();
    updateStatusButtonState();
  }
}

async function executePreparedRequest(preparedRequestText) {
  const { runId, workDir } = beginTaskExecutionRun();
  calculationInProgress = true;
  clearActiveRunContext();
  activeRunTarget = selectedExecutionTarget();
  activeRunPreparedRequestText = preparedRequestText || '';
  activeRunStopRequested = false;
  activeRunCancelPromise = null;
  activeRunCancelledNotified = false;
  setCalculationControls(true);
  try {
    const { payload: submission } = await postCalculationLifecycle('/api/run-submit', {
      prepared_request: preparedRequestText,
      locale: currentLanguage,
      work_dir: workDir,
      run_id: runId,
      execution_target: activeRunTarget,
      resource_profile: selectedResourceProfile(),
      lifecycle: currentTaskLifecycle,
    });
    updateCurrentTaskLifecycle(submission);
    activeRunHandle = submission.handle;
    activeRunLastState = String((submission.status && submission.status.state) || 'queued');
    updateRunDirectoryFromHandle(activeRunHandle);
    conversationHistory = appendAssistantMessages(conversationHistory, [{ role: 'system', content: t('runStarted') }]);
    renderConversation(conversationHistory);
    updateActiveTaskSessionStatus('running');
    snapshotCurrentTaskSession();
    updateStatusButtonState();
    if (activeRunStopRequested && await requestActiveRunCancellation()) return false;
    const terminalStatus = await waitForSubmittedRun(activeRunHandle, activeRunTarget);
    if (!terminalStatus) return false;
    const payload = await collectSubmittedRun(activeRunHandle, preparedRequestText, activeRunTarget);
    applyCollectedRunPayload(payload, preparedRequestText, runId);
    return true;
  } catch (error) {
    if (!activeRunCancelledNotified) {
      const submittedRunCanBeQueried = Boolean(activeRunHandle && activeRunLastState !== 'failed');
      conversationHistory = appendAssistantMessages(conversationHistory, [{
        role: 'system',
        content: submittedRunCanBeQueried
          ? `${t('runConnectionInterrupted')} (${error instanceof Error ? error.message : String(error)})`
          : `${t('runFailed')}: ${error instanceof Error ? error.message : String(error)}`,
      }]);
      renderConversation(conversationHistory);
      if (submittedRunCanBeQueried) {
        updateActiveTaskSessionStatus(['completed', 'cancelled'].includes(activeRunLastState) ? activeRunLastState : 'running');
        renderStatus(activeRunLastState || 'running');
      } else {
        updateActiveTaskSessionStatus('failed');
        renderStatus('failed');
      }
      snapshotCurrentTaskSession();
    }
    return false;
  } finally {
    calculationInProgress = false;
    activeRunStopRequested = false;
    activeRunCancelPromise = null;
    setCalculationControls(false);
    snapshotCurrentTaskSession();
  }
}

async function analyzeCurrentResult() {
  if (!llmResultAnalysisAvailable()) {
    conversationHistory = appendAssistantMessages(conversationHistory, [{ role: 'system', content: t('analyzeNotConfigured') }]);
    renderConversation(conversationHistory);
    return;
  }
  if (!lastExecutionPayload || !lastExecutedRequestText) {
    conversationHistory = appendAssistantMessages(conversationHistory, [{ role: 'system', content: t('analyzeNoResult') }]);
    renderConversation(conversationHistory);
    return;
  }
  setAnalyzeState(true);
  conversationHistory = appendAssistantMessages(conversationHistory, [{ role: 'system', content: t('analyzeGenerating') }]);
  renderConversation(conversationHistory);
  try {
    const response = await fetch('/api/analyze-result', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prepared_request: lastExecutedRequestText,
        execution_report: lastExecutionPayload,
        locale: currentLanguage,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      conversationHistory = appendAssistantMessages(conversationHistory, [{ role: 'system', content: payload.error || t('analyzeFailed') }]);
      renderConversation(conversationHistory);
      return;
    }
    lastExecutionPayload = { ...lastExecutionPayload, result_analysis: payload.result_analysis || '' };
    renderSummary(lastExecutionPayload);
    conversationHistory = appendAssistantMessages(conversationHistory, [
      { role: 'system', content: t('analyzeDone') },
      { role: 'assistant', content: payload.result_analysis || t('analyzeEmptyResult') },
    ]);
    renderConversation(conversationHistory);
    snapshotCurrentTaskSession();
  } catch (error) {
    conversationHistory = appendAssistantMessages(conversationHistory, [{ role: 'system', content: `${t('analyzeFailed')}: ${error instanceof Error ? error.message : String(error)}` }]);
    renderConversation(conversationHistory);
    snapshotCurrentTaskSession();
  } finally {
    setAnalyzeState(false);
  }
}

async function approvePendingStructure() {
  if (!pendingApproval || pendingApproval.type !== 'structure') {
    return;
  }
  const approval = pendingApproval;
  const approvedAtom = document.getElementById('approval-atom').value.trim();
  if (!approvedAtom) {
    renderConversation(conversationHistory, [{ role: 'system', content: t('approvalEmptyDraft') }]);
    return;
  }
  const approvedRequest = { ...(approval.structured_request || {}), atom: approvedAtom };
  const approvedProbeRequest = approval.probe_request && typeof approval.probe_request === 'object'
    ? { ...approval.probe_request, atom: approvedAtom }
    : null;
  const hasNextApproval = approval.next_approval && approval.next_approval.type === 'active_space';
  if (!hasNextApproval) {
    await applyPendingApprovalLifecycle('approve', 'structure');
  }
  pendingStructuredRequest = approvedRequest;
  applyTaskSpecToForm(approvedRequest);
  clearApprovalPanel();
  conversationHistory = appendAssistantMessages(conversationHistory, [{ role: 'system', content: t('approvalConfirmed') }]);
  renderConversation(conversationHistory);
  if (hasNextApproval) {
    const nextApproval = buildActiveSpaceApprovalFromResult({
      approval: approval.next_approval,
      structured_request: approvedRequest,
    });
    if (nextApproval) {
      renderApprovalPanel(nextApproval);
      snapshotCurrentTaskSession();
      return;
    }
  }
  if (!approval.can_run_immediately) {
    renderPreparationState({
      status: 'needs_clarification',
      source: lastPreparationMeta && lastPreparationMeta.source ? lastPreparationMeta.source : 'llm',
      llm_cache_scope: lastPreparationMeta && lastPreparationMeta.llm_cache_scope ? lastPreparationMeta.llm_cache_scope : currentPrepareCacheScope,
      structured_request: approvedRequest,
      applied_defaults: lastPreparationMeta && Array.isArray(lastPreparationMeta.applied_defaults) ? lastPreparationMeta.applied_defaults : [],
      generated_fields: lastPreparationMeta && Array.isArray(lastPreparationMeta.generated_fields) ? lastPreparationMeta.generated_fields : [],
      precision_note: lastPreparationMeta && typeof lastPreparationMeta.precision_note === 'string' ? lastPreparationMeta.precision_note : '',
      clarification_questions: Array.isArray(approval.remaining_clarification_questions) ? approval.remaining_clarification_questions : [],
      missing_fields: Array.isArray(approval.remaining_missing_fields) ? approval.remaining_missing_fields : [],
    });
    snapshotCurrentTaskSession();
    return;
  }
  renderComputeConfirmPanel(
    t('computeReady'),
    JSON.stringify(approvedProbeRequest || approvedRequest)
  );
  snapshotCurrentTaskSession();
}

function parseActiveSpaceApprovalDraft() {
  const rawText = document.getElementById('approval-atom').value.trim();
  if (!rawText) {
    return null;
  }
  let parsed = null;
  try {
    parsed = JSON.parse(rawText);
  } catch (error) {
    return null;
  }
  const activeSpace = parsed && parsed.active_space && typeof parsed.active_space === 'object'
    ? parsed.active_space
    : parsed;
  if (!activeSpace || typeof activeSpace !== 'object') {
    return null;
  }
  const ncas = Number.parseInt(activeSpace.ncas, 10);
  const orbitalIndices = normalizeApprovalOrbitalIndices(activeSpace.orbital_indices);
  const nelecas = activeSpace.nelecas;
  const hasInitialOrbitals = Array.isArray(activeSpace.initial_mo_coeff);
  const isAvas = activeSpace.selection_method === 'avas' && hasInitialOrbitals;
  const hasNelecas = Array.isArray(nelecas)
    ? nelecas.length > 0
    : nelecas !== null && typeof nelecas !== 'undefined' && String(nelecas).trim() !== '';
  const usesDefaultOrbitalWindow = Array.isArray(orbitalIndices) && orbitalIndices.length === 0;
  if (!Number.isInteger(ncas) || ncas <= 0 || !hasNelecas
    || (!hasInitialOrbitals && !usesDefaultOrbitalWindow && !orbitalIndicesMatchNcas(orbitalIndices, ncas))) {
    return null;
  }
  return {
    enabled: true,
    selection_method: isAvas ? 'avas' : 'manual',
    ncas,
    nelecas,
    orbital_indices: orbitalIndices,
    avas_targets: Array.isArray(activeSpace.avas_targets) ? activeSpace.avas_targets : [],
    avas_threshold: Number.parseFloat(activeSpace.avas_threshold) || 0.2,
    initial_mo_coeff: hasInitialOrbitals ? activeSpace.initial_mo_coeff : null,
    target_method: typeof activeSpace.target_method === 'string' ? activeSpace.target_method : null,
    target_solver: typeof activeSpace.target_solver === 'string' ? activeSpace.target_solver : null,
    target_solver_options: activeSpace.target_solver_options && typeof activeSpace.target_solver_options === 'object'
      ? activeSpace.target_solver_options
      : {},
    approved: true,
  };
}

async function approvePendingActiveSpace() {
  if (!pendingApproval || pendingApproval.type !== 'active_space') {
    return;
  }
  const activeSpaceContract = parseActiveSpaceApprovalDraft();
  if (!activeSpaceContract) {
    renderConversation(conversationHistory, [{ role: 'system', content: t('approvalInvalidActiveSpace') }]);
    return;
  }
  await applyPendingApprovalLifecycle('approve', 'active_space');
  const selectedSolver = pendingApproval.target_solver
    || activeSpaceContract.target_solver
    || document.getElementById('active-space-solver').value;
  const targetMethod = pendingApproval.target_method || 'casscf';
  setSelectValue('active-space-solver', selectedSolver);
  const targetSolverOptions = activeSpaceContract.target_solver_options || {};
  document.getElementById('state-target-nroots').value = targetSolverOptions.nroots || 1;
  document.getElementById('state-average-weights').value = Array.isArray(targetSolverOptions.state_average_weights)
    ? targetSolverOptions.state_average_weights.join(',')
    : '';
  writeActiveSpaceContractToForm(activeSpaceContract, true);
  setSelectValue('method', targetMethod);
  syncMethodControls();
  updateActiveSpaceSummary();
  const taskSpec = collectFormData();
  taskSpec.method = targetMethod;
  taskSpec.xc = null;
  taskSpec.outputs = molecularOutputsWith(taskSpec.outputs);
  taskSpec.active_space = activeSpaceContract;
  if (pendingApproval.runtime_contract && typeof pendingApproval.runtime_contract === 'object') {
    taskSpec.runtime = { ...pendingApproval.runtime_contract };
  }
  taskSpec.solver = {
    ...(taskSpec.solver || {}),
    name: selectedSolver,
    options: {
      ...((taskSpec.solver && taskSpec.solver.options) || {}),
      ...(activeSpaceContract.target_solver_options || {}),
    },
  };
  taskSpec.restricted = selectedSolver === 'block2_dmrg'
    ? true
    : (activeSpaceContract.selection_method === 'avas' ? true : casRestrictedValueForActiveSpace(activeSpaceContract));
  document.getElementById('restricted').value = taskSpec.restricted ? 'true' : 'false';
  pendingStructuredRequest = taskSpec;
  clearApprovalPanel();
  renderComputeConfirmPanel(t('computeReady'), JSON.stringify(taskSpec));
  conversationHistory = appendAssistantMessages(conversationHistory, [{ role: 'system', content: t('approvalActiveSpaceConfirmed') }]);
  renderConversation(conversationHistory);
  snapshotCurrentTaskSession();
}

function parseCorrelatedSubspaceApprovalDraft() {
  try {
    const patch = JSON.parse(document.getElementById('approval-atom').value || '{}');
    if (!patch || typeof patch !== 'object' || !patch.embedding || !patch.solver) {
      return null;
    }
    const orbitalIndices = patch.embedding.correlated_orbital_indices;
    if (!Array.isArray(orbitalIndices) || !orbitalIndices.length
      || orbitalIndices.some((value) => !Number.isInteger(value) || value < 0)) {
      return null;
    }
    patch.embedding.approved = true;
    patch.embedding.enabled = true;
    patch.embedding.provider = 'fcdmft';
    const solverName = String(patch.solver.name || '').toLowerCase().replace('-', '_');
    if (!['hf_dmft', 'gw_dmft'].includes(solverName)) {
      return null;
    }
    patch.solver.name = solverName;
    return patch;
  } catch (error) {
    return null;
  }
}

async function approvePendingCorrelatedSubspace() {
  if (!pendingApproval || pendingApproval.type !== 'correlated_subspace') {
    return;
  }
  const patch = parseCorrelatedSubspaceApprovalDraft();
  if (!patch) {
    conversationHistory = appendAssistantMessages(conversationHistory, [{
      role: 'system',
      content: 'The correlated-subspace patch is invalid. Keep a non-empty list of orbital indices and the HF+DMFT or GW+DMFT solver contract.',
    }]);
    renderConversation(conversationHistory);
    return;
  }
  await applyPendingApprovalLifecycle('approve', 'correlated_subspace');
  const baseTaskSpec = lastExecutionPayload && lastExecutionPayload.task_spec
    && typeof lastExecutionPayload.task_spec === 'object'
    ? JSON.parse(JSON.stringify(lastExecutionPayload.task_spec))
    : collectPeriodicFormData();
  const approvedSolver = String(patch.solver.name || '').toLowerCase().replace('-', '_');
  baseTaskSpec.method = approvedSolver === 'gw_dmft' ? 'dft' : 'hf';
  baseTaskSpec.xc = approvedSolver === 'gw_dmft'
    ? (baseTaskSpec.xc || 'pbe')
    : null;
  baseTaskSpec.embedding = {
    ...((baseTaskSpec.embedding && typeof baseTaskSpec.embedding === 'object') ? baseTaskSpec.embedding : {}),
    ...patch.embedding,
    approved: true,
  };
  baseTaskSpec.solver = {
    ...((baseTaskSpec.solver && typeof baseTaskSpec.solver === 'object') ? baseTaskSpec.solver : {}),
    ...patch.solver,
    options: {
      ...((baseTaskSpec.solver && baseTaskSpec.solver.options) || {}),
      ...((patch.solver && patch.solver.options) || {}),
    },
  };
  pendingStructuredRequest = baseTaskSpec;
  applyTaskSpecToForm(baseTaskSpec);
  clearApprovalPanel();
  renderComputeConfirmPanel('The approved correlated subspace is ready for fcDMFT execution.', JSON.stringify(baseTaskSpec));
  conversationHistory = appendAssistantMessages(conversationHistory, [{
    role: 'system',
    content: 'The correlated subspace has been approved. Start Calculation will run fcDMFT without rebuilding the HF proposal.',
  }]);
  renderConversation(conversationHistory);
  snapshotCurrentTaskSession();
}

async function approvePendingApproval() {
  if (!pendingApproval) {
    return;
  }
  try {
    if (pendingApproval.type === 'active_space') {
      await approvePendingActiveSpace();
    } else if (pendingApproval.type === 'correlated_subspace') {
      await approvePendingCorrelatedSubspace();
    } else {
      await approvePendingStructure();
    }
  } catch (error) {
    conversationHistory = appendAssistantMessages(conversationHistory, [{
      role: 'system',
      content: `${t('approvalFailed')}: ${error instanceof Error ? error.message : String(error)}`,
    }]);
    renderConversation(conversationHistory);
    snapshotCurrentTaskSession();
  }
}

function cancelPendingStructure() {
  if (!pendingApproval) {
    return;
  }
  pendingStructuredRequest = pendingApproval.safe_structured_request || pendingStructuredRequest;
  applyTaskSpecToForm(pendingStructuredRequest);
  clearApprovalPanel();
  clearComputeConfirmPanel();
  conversationHistory = appendAssistantMessages(conversationHistory, [{ role: 'system', content: t('approvalCanceled') }]);
  renderConversation(conversationHistory);
  updateRequestPlaceholder({
    clarification_questions: [t('continueEditingPrompt')],
  });
  snapshotCurrentTaskSession();
}

function cancelPendingActiveSpace() {
  if (!pendingApproval || pendingApproval.type !== 'active_space') {
    return;
  }
  clearApprovalPanel();
  clearComputeConfirmPanel();
  conversationHistory = appendAssistantMessages(conversationHistory, [{ role: 'system', content: t('approvalActiveSpaceCanceled') }]);
  renderConversation(conversationHistory);
  snapshotCurrentTaskSession();
}

function cancelPendingApproval() {
  if (!pendingApproval) {
    return;
  }
  if (pendingApproval.type === 'active_space') {
    cancelPendingActiveSpace();
  } else {
    cancelPendingStructure();
  }
}

async function confirmCompute() {
  if (!pendingExecutionRequest) {
    return;
  }
  await executePreparedRequest(pendingExecutionRequest);
}

function molecularOutputsWith(values) {
  const merged = new Set(Array.isArray(values) ? values : []);
  DEFAULT_MOLECULAR_OUTPUTS.forEach((item) => merged.add(item));
  return Array.from(merged);
}

async function screenActiveSpace() {
  if (currentTaskFamily() !== 'molecular') {
    return;
  }
  const taskSpec = collectFormData();
  const targetSolver = document.getElementById('active-space-solver').value || 'fci';
  const targetSolverOptions = taskSpec.solver && taskSpec.solver.options
    ? taskSpec.solver.options
    : {};
  document.getElementById('active-space-enabled').checked = true;
  document.getElementById('active-space-approved').checked = false;
  updateActiveSpaceSummary();
  clearComputeConfirmPanel();
  conversationHistory = appendAssistantMessages(conversationHistory, [
    { role: 'system', content: t('activeSpaceScreeningStarted') },
  ]);
  renderConversation(conversationHistory);
  try {
    const response = await fetch('/api/active-space-probe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        task_spec: taskSpec,
        target_solver: targetSolver,
        target_solver_options: targetSolverOptions,
      }),
    });
    const prepared = await response.json();
    if (!response.ok || !prepared.probe_request) {
      conversationHistory = appendAssistantMessages(conversationHistory, [
        { role: 'system', content: prepared.error || t('prepareFailed') },
      ]);
      renderConversation(conversationHistory);
      return;
    }
    setSelectValue(
      'active-space-method',
      prepared.probe_contract && prepared.probe_contract.selection_method
        ? prepared.probe_contract.selection_method
        : 'occupation_window',
    );
    await executePreparedRequest(JSON.stringify(prepared.probe_request));
  } catch (error) {
    conversationHistory = appendAssistantMessages(conversationHistory, [
      { role: 'system', content: `${t('prepareFailed')} ${error}` },
    ]);
    renderConversation(conversationHistory);
  }
}

function invalidatePendingExecution() {
  if (!pendingExecutionRequest) {
    return;
  }
  pendingExecutionRequest = '';
  markTaskConfigurationChanged();
  const panel = document.getElementById('compute-confirm-panel');
  panel.classList.remove('ready');
  document.getElementById('confirm-compute').disabled = true;
  const changedPayload = {
    status: 'changed',
    structured_request: collectFormData(),
    missing_fields: [],
    clarification_questions: [],
    lifecycle: currentTaskLifecycle,
  };
  lastPreparedPayload = changedPayload;
  conversationHistory = appendAssistantMessages(conversationHistory, [
    { role: 'system', content: t('configurationChangedMessage') },
  ]);
  renderConversation(conversationHistory);
  if (!lastExecutionPayload) {
    renderPreparationState(changedPayload);
    updateActiveTaskSessionStatus('changed');
  }
  snapshotCurrentTaskSession();
}

function clearMolecularStructureValidation() {
  const atom = document.getElementById('atom');
  const validation = document.getElementById('atom-validation');
  atom.classList.remove('input-invalid');
  atom.removeAttribute('aria-invalid');
  validation.textContent = '';
}

function validateMolecularStructureBeforePrepare(request) {
  const atom = document.getElementById('atom');
  if (atom.value.trim() || String(request || '').trim()) {
    clearMolecularStructureValidation();
    return '';
  }
  const message = t('molecularStructureRequired');
  atom.classList.add('input-invalid');
  atom.setAttribute('aria-invalid', 'true');
  document.getElementById('atom-validation').textContent = message;
  atom.focus();
  atom.scrollIntoView({ behavior: 'smooth', block: 'center' });
  return message;
}

async function prepareModelHamiltonianEdit(taskSpec, request) {
  let nextConversation = conversationHistory.slice();
  nextConversation.push({ role: 'user', content: request });
  nextConversation = appendAssistantMessages(nextConversation, [{ role: 'system', content: t('runPreparing') }]);
  renderConversation(nextConversation);
  setRunState(true);
  try {
    const response = await fetch('/api/model-hamiltonian-edit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        request,
        task_spec: taskSpec,
        messages: buildPrepareMessages(request),
        locale: currentLanguage,
        lifecycle: currentTaskLifecycle,
      }),
    });
    const prepared = await response.json();
    if (!response.ok) {
      conversationHistory = appendAssistantMessages(nextConversation, [{
        role: 'system',
        content: prepared.error || t('prepareFailed'),
      }]);
      renderConversation(conversationHistory);
      renderPreparationState({
        status: 'needs_clarification',
        structured_request: taskSpec,
        missing_fields: [],
        clarification_questions: [prepared.error || t('prepareFailed')],
      });
      updateActiveTaskSessionStatus('draft');
      snapshotCurrentTaskSession();
      return;
    }

    lastPreparedPayload = prepared;
    updateCurrentTaskLifecycle(prepared);
    conversationHistory = appendAssistantMessages(nextConversation, prepared.messages || []);
    pendingStructuredRequest = prepared.structured_request || taskSpec;
    applyTaskSpecToForm(pendingStructuredRequest);
    await previewModelHamiltonianStructure();
    renderConversation(conversationHistory);
    renderPreparationState(prepared);
    document.getElementById('request').value = '';
    if (prepared.status !== 'ready') {
      updateRequestPlaceholder(prepared);
      updateActiveTaskSessionStatus('draft');
      snapshotCurrentTaskSession();
      return;
    }

    const dmetFormError = await validateDmetModelControls();
    if (dmetFormError) {
      conversationHistory = appendAssistantMessages(conversationHistory, [{ role: 'system', content: dmetFormError }]);
      renderConversation(conversationHistory);
      renderPreparationState({
        ...prepared,
        status: 'needs_clarification',
        missing_fields: ['solver.options'],
        clarification_questions: [dmetFormError],
      });
      updateActiveTaskSessionStatus('draft');
      snapshotCurrentTaskSession();
      return;
    }

    lastPreparationMeta = {
      source: prepared.source || 'llm_model_operations',
      llm_cache_scope: currentPrepareCacheScope || '',
      request_summary: request,
      applied_defaults: [],
      generated_fields: ['model_hamiltonian_input_file'],
      precision_note: '',
    };
    renderComputeConfirmPanel(
      t('computeReady'),
      prepared.execution_request_text || prepared.request_text || JSON.stringify(pendingStructuredRequest)
    );
    updateActiveTaskSessionStatus('ready');
    snapshotCurrentTaskSession();
  } catch (error) {
    conversationHistory = appendAssistantMessages(nextConversation, [{
      role: 'system',
      content: `${t('prepareFailed')}: ${error instanceof Error ? error.message : String(error)}`,
    }]);
    renderConversation(conversationHistory);
    updateActiveTaskSessionStatus('failed');
    snapshotCurrentTaskSession();
  } finally {
    setRunState(false);
  }
}

async function runAgent() {
  const request = document.getElementById('request').value.trim();
  if (currentTaskFamily() === 'molecular') {
    const molecularStructureError = validateMolecularStructureBeforePrepare(request);
    if (molecularStructureError) {
      clearComputeConfirmPanel();
      conversationHistory = appendAssistantMessages(conversationHistory, [
        { role: 'system', content: molecularStructureError },
      ]);
      renderConversation(conversationHistory);
      const invalidPayload = {
        status: 'needs_clarification',
        structured_request: collectFormData(),
        missing_fields: ['atom'],
        clarification_questions: [molecularStructureError],
      };
      lastPreparedPayload = invalidPayload;
      renderPreparationState(invalidPayload);
      updateActiveTaskSessionStatus('draft');
      snapshotCurrentTaskSession();
      return;
    }
  }
  if (currentTaskFamily() === 'periodic') {
    const periodicFormError = validatePeriodicFormControls();
    if (periodicFormError) {
      document.getElementById('periodic-numerics-section').open = true;
      conversationHistory = appendAssistantMessages(conversationHistory, [{ role: 'system', content: periodicFormError }]);
      renderConversation(conversationHistory);
      snapshotCurrentTaskSession();
      return;
    }
  }
  const formTaskSpec = collectFormData();
  const task_spec = formTaskSpec.task_type === 'model_hamiltonian' || formTaskSpec.task_type === 'periodic'
    ? formTaskSpec
    : mergeTaskSpec(pendingStructuredRequest, formTaskSpec);
  clearComputeConfirmPanel();
  if (task_spec.task_type === 'model_hamiltonian') {
    if (!task_spec.model_hamiltonian_input_file) {
      conversationHistory = appendAssistantMessages(conversationHistory, [{ role: 'system', content: t('modelHamiltonianMissingFile') }]);
      renderConversation(conversationHistory);
      renderPreparationState({
        status: 'needs_clarification',
        structured_request: task_spec,
        missing_fields: ['model_hamiltonian_input_file'],
        clarification_questions: [t('modelHamiltonianMissingFile')],
      });
      snapshotCurrentTaskSession();
      return;
    }
    if (request) {
      await prepareModelHamiltonianEdit(task_spec, request);
      return;
    }
    const dmetFormError = await validateDmetModelControls();
    if (dmetFormError) {
      conversationHistory = appendAssistantMessages(conversationHistory, [{ role: 'system', content: dmetFormError }]);
      renderConversation(conversationHistory);
      renderPreparationState({
        status: 'needs_clarification',
        structured_request: task_spec,
        missing_fields: ['solver.options'],
        clarification_questions: [dmetFormError],
      });
      updateActiveTaskSessionStatus('draft');
      snapshotCurrentTaskSession();
      return;
    }
    const preparedRequestText = JSON.stringify(task_spec);
    lastPreparationMeta = {
      source: 'structured_seed',
      llm_cache_scope: currentPrepareCacheScope || '',
      request_summary: t('taskFamilyModelHamiltonian'),
      applied_defaults: [],
      generated_fields: [],
      precision_note: '',
    };
    pendingStructuredRequest = task_spec;
    const preparedPayload = {
      status: 'ready',
      source: 'structured_seed',
      structured_request: task_spec,
      messages: [{ role: 'system', content: t('modelHamiltonianPrepared') }],
      lifecycle: currentTaskLifecycle,
    };
    lastPreparedPayload = preparedPayload;
    renderPreparationState(preparedPayload);
    conversationHistory = appendAssistantMessages(conversationHistory, [{ role: 'system', content: t('modelHamiltonianPrepared') }]);
    renderConversation(conversationHistory);
    renderComputeConfirmPanel(t('computeReady'), preparedRequestText);
    document.getElementById('request').value = '';
    updateActiveTaskSessionStatus('ready');
    snapshotCurrentTaskSession();
    return;
  }
  if (task_spec.task_type === 'periodic') {
    let nextConversation = conversationHistory.slice();
    if (request) {
      nextConversation.push({ role: 'user', content: request });
    }
    const structureText = task_spec.periodic && typeof task_spec.periodic.structure_text === 'string'
      ? task_spec.periodic.structure_text.trim()
      : '';
    if (!structureText) {
      conversationHistory = appendAssistantMessages(nextConversation, [{ role: 'system', content: t('periodicMissingStructure') }]);
      renderConversation(conversationHistory);
      renderPreparationState({
        status: 'needs_clarification',
        structured_request: task_spec,
        missing_fields: ['periodic.structure_text'],
        clarification_questions: [t('periodicMissingStructure')],
      });
      updateActiveTaskSessionStatus('draft');
      snapshotCurrentTaskSession();
      return;
    }
    const structureIsValid = await previewPeriodicStructure();
    if (!structureIsValid) {
      conversationHistory = appendAssistantMessages(nextConversation, [{ role: 'system', content: t('periodicPreviewFailed') }]);
      renderConversation(conversationHistory);
      renderPreparationState({
        status: 'needs_clarification',
        structured_request: task_spec,
        missing_fields: [],
        clarification_questions: [t('periodicPreviewFailed')],
      });
      updateActiveTaskSessionStatus('draft');
      snapshotCurrentTaskSession();
      return;
    }
    const preparedRequestText = JSON.stringify(task_spec);
    lastPreparationMeta = {
      source: 'structured_seed',
      llm_cache_scope: currentPrepareCacheScope || '',
      request_summary: t('taskFamilyPeriodic'),
      applied_defaults: [],
      generated_fields: [],
      precision_note: '',
    };
    pendingStructuredRequest = task_spec;
    const preparedPayload = {
      status: 'ready',
      source: 'structured_seed',
      structured_request: task_spec,
      messages: [{ role: 'system', content: t('periodicPrepared') }],
      lifecycle: currentTaskLifecycle,
    };
    lastPreparedPayload = preparedPayload;
    renderPreparationState(preparedPayload);
    conversationHistory = appendAssistantMessages(nextConversation, [{ role: 'system', content: t('periodicPrepared') }]);
    renderConversation(conversationHistory);
    renderComputeConfirmPanel(t('computeReady'), preparedRequestText);
    document.getElementById('request').value = '';
    updateActiveTaskSessionStatus('ready');
    snapshotCurrentTaskSession();
    return;
  }
  let nextConversation = conversationHistory.slice();
  if (request) {
    nextConversation.push({ role: 'user', content: request });
  }
  nextConversation = appendAssistantMessages(nextConversation, [{ role: 'system', content: t('runPreparing') }]);
  renderConversation(nextConversation);
  setRunState(true);
  try {
    const prepareMessages = buildPrepareMessages(request);
    const prepareFingerprint = fingerprintPrepareInput(task_spec, prepareMessages);
    let prepared = null;
    if (prepareFingerprint === lastPrepareFingerprint && lastPreparedPayload) {
      prepared = lastPreparedPayload;
    } else {
      const prepareResponse = await fetch('/api/prepare', {
         method: 'POST',
         headers: { 'Content-Type': 'application/json' },
         body: JSON.stringify({
           request,
           task_spec,
           messages: prepareMessages,
           locale: currentLanguage,
           lifecycle: currentTaskLifecycle,
         })
      });
      prepared = await prepareResponse.json();
      if (!prepareResponse.ok) {
        conversationHistory = appendAssistantMessages(nextConversation, [{ role: 'system', content: prepared.error || t('prepareFailed') }]);
        renderConversation(conversationHistory);
        updateActiveTaskSessionStatus('failed');
        snapshotCurrentTaskSession();
        return;
      }
      if (typeof prepared.llm_cache_scope === 'string' && prepared.llm_cache_scope) {
        currentPrepareCacheScope = prepared.llm_cache_scope;
      }
      lastPrepareFingerprint = prepareFingerprint;
      lastPreparedPayload = prepared;
    }
    updateCurrentTaskLifecycle(prepared);
    conversationHistory = appendAssistantMessages(nextConversation, prepared.messages || []);
    lastPreparationMeta = {
      source: prepared.source || '',
      llm_cache_scope: prepared.llm_cache_scope || '',
      request_summary: prepared.structured_request && prepared.structured_request.request_summary ? prepared.structured_request.request_summary : '',
      applied_defaults: Array.isArray(prepared.applied_defaults) ? prepared.applied_defaults : [],
      generated_fields: Array.isArray(prepared.generated_fields) ? prepared.generated_fields : [],
      precision_note: typeof prepared.precision_note === 'string' ? prepared.precision_note : '',
    };
    pendingStructuredRequest = prepared.approved_structured_request || prepared.structured_request || {};
    applyTaskSpecToForm(pendingStructuredRequest);
    updateRequestPlaceholder(prepared);
    renderConversation(conversationHistory);
    if (prepared.approval) {
      renderPreparationState(prepared);
      document.getElementById('request').value = '';
      updateActiveTaskSessionStatus('draft');
      snapshotCurrentTaskSession();
      return;
    }
    if (prepared.status !== 'ready') {
      renderPreparationState(prepared);
      document.getElementById('request').value = '';
      updateActiveTaskSessionStatus('draft');
      snapshotCurrentTaskSession();
      return;
    }
    renderPreparationState(prepared);
    renderComputeConfirmPanel(
      t('computeReady'),
      prepared.execution_request_text || prepared.request_text
    );
    document.getElementById('request').value = '';
    updateActiveTaskSessionStatus('ready');
    snapshotCurrentTaskSession();
  } catch (error) {
    conversationHistory = appendAssistantMessages(nextConversation, [{ role: 'system', content: `${t('requestFailed')}: ${error instanceof Error ? error.message : String(error)}` }]);
    renderConversation(conversationHistory);
    updateActiveTaskSessionStatus('failed');
    snapshotCurrentTaskSession();
  } finally {
    setRunState(false);
    snapshotCurrentTaskSession();
  }
}
