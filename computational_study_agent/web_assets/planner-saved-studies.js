// This view reads the agent's saved artifacts. It owns no execution workflow.
let savedStudyPollTimer = null;
let savedStudyRefreshVersion = 0;
let savedStudyStopping = false;

function syncStudyStopButton(execution) {
  const button = document.getElementById('stop-saved-study');
  button.disabled = savedStudyStopping || !currentSavedStudy || !execution || !execution.can_stop;
  button.textContent = savedStudyStopping ? 'Stopping…' : 'Stop Calculation';
}


function stopSavedStudyPolling() {
  if (savedStudyPollTimer !== null) window.clearTimeout(savedStudyPollTimer);
  savedStudyPollTimer = null;
  savedStudyRefreshVersion += 1;
}

function scheduleSavedStudyPolling(identity, execution) {
  if (savedStudyPollTimer !== null) window.clearTimeout(savedStudyPollTimer);
  savedStudyPollTimer = null;
  if (currentSavedStudy !== identity || !(execution.agent && execution.agent.running)
      || !currentPlan) return;
  const version = savedStudyRefreshVersion;
  savedStudyPollTimer = window.setTimeout(async () => {
    savedStudyPollTimer = null;
    if (currentSavedStudy !== identity || version !== savedStudyRefreshVersion) return;
    try {
      const {execution: latest} = await postJson('/api/study-execution-status', savedStudyRequest());
      if (currentSavedStudy !== identity || version !== savedStudyRefreshVersion) return;
      if (!(latest.agent && latest.agent.running)
          || Number(latest.task_count) !== (currentPlan.cases || []).length) {
        await refreshSavedStudy();
        return;
      }
      syncStudyStopButton(latest);
      renderLiveStudyTable(latest);
      renderRunningGridProgress(latest, currentPlan, currentReport);
      if (currentExecution) renderExecutionRecovery(latest);
      setStatus('saved-study-status', `${identity.studyId} · running · ${executionStatusSummary(latest)}`);
      scheduleSavedStudyPolling(identity, latest);
    } catch (error) {
      if (currentSavedStudy !== identity || version !== savedStudyRefreshVersion) return;
      setStatus('saved-study-status', `${error.message} Live updates will retry automatically.`, 'error');
      scheduleSavedStudyPolling(identity, execution);
    }
  }, 5000);
}

function savedStudyRequest() {
  return currentSavedStudy ? {
    study_id: currentSavedStudy.studyId,
    work_dir: currentSavedStudy.workDir,
    execution_target: currentSavedStudy.executionTarget,
    saved_study: true,
  } : null;
}

function syncSavedStudyControls() {
  const saved = Boolean(currentSavedStudy);
  syncStudyStopButton(null);
  // Saved inputs belong to the agent; use New Task to draft a different Study.
  ['build-plan', 'load-example', 'draft-with-llm', 'study-system', 'study-mode',
    'planner-template', 'execution-target', 'resource-profile', 'study-inputs',
    'adaptive-initial-scan-strategy', 'adaptive-active-space-solver',
    'adaptive-active-space-localization', 'adaptive-orbital-ordering',
    'adaptive-manual-orbital-order', 'top-open-model-builder'].forEach((id) => {
    document.getElementById(id).disabled = saved;
  });
  document.getElementById('refresh-saved-study').disabled = !saved;
  document.getElementById('collect-saved-study').disabled = true;
  document.getElementById('saved-study-id').value = saved ? currentSavedStudy.studyId : '';
  if (!saved) {
    stopSavedStudyPolling();
    setStatus('saved-study-status', 'Draft. Select Build Plan to save a Study.');
  }
}

function takeSavedStudyLocation() {
  const url = new URL(window.location.href);
  const requested = url.searchParams.get('study') ? {
    studyId: url.searchParams.get('study'),
    workDir: url.searchParams.get('work_dir') || DEFAULT_WORK_DIR,
    executionTarget: url.searchParams.get('execution_target') || undefined,
  } : null;
  // A Study link opens once. The workspace URL stays clean so a page reload
  // starts a draft instead of reopening a previous Study and its review gates.
  ['study', 'work_dir', 'execution_target'].forEach((key) => url.searchParams.delete(key));
  window.history.replaceState(null, '', url);
  const navigation = window.performance.getEntriesByType('navigation')[0];
  return navigation && navigation.type === 'reload' ? null : requested;
}

async function listSavedStudies() {
  try {
    const payload = await postJson('/api/study-list', {
      work_dir: executionWorkDir(), execution_target: selectedExecutionTarget(),
    });
    const select = document.getElementById('saved-study-list');
    select.replaceChildren(new Option('Select a Study', ''));
    payload.studies.forEach((study) => {
      const duplicate = payload.studies.filter(item => item.study_id === study.study_id).length > 1;
      const location = duplicate ? ` · ${study.work_dir}` : '';
      const option = new Option(`${study.study_id} · ${study.name || ''} · ${study.status}${location}`, study.study_id);
      option.dataset.workDir = study.work_dir || executionWorkDir();
      select.add(option);
    });
    setStatus('saved-study-status', `${payload.studies.length} saved Studies. Select one to open.`);
  } catch (error) {
    setStatus('saved-study-status', error.message, 'error');
  }
}

async function loadReportExamples() {
  const select = document.getElementById('report-example-select');
  try {
    const payload = await postJson('/api/study-examples', {});
    select.replaceChildren(new Option('Select an example', ''));
    (payload.examples || []).forEach(item => select.add(new Option(item.name, item.study_id)));
    if (!(payload.examples || []).length) {
      setStatus('report-example-status', 'Report examples are available in the source checkout.');
    }
  } catch (error) {
    select.replaceChildren(new Option('Examples unavailable', ''));
    setStatus('report-example-status', error.message, 'error');
  }
}

async function openReportExample() {
  const studyId = document.getElementById('report-example-select').value;
  if (!studyId) return;
  const button = document.getElementById('open-report-example');
  button.disabled = true;
  setStatus('report-example-status', 'Opening example…');
  try {
    await executionTargetsReady;
    const imported = await postJson('/api/study-example-import', {
      study_id: studyId, work_dir: executionWorkDir(),
    });
    const opened = await openSavedStudy({studyId: imported.study_id, workDir: imported.work_dir,
      executionTarget: imported.execution_target});
    if (!opened) throw new Error('The example could not be opened. See the Saved Study status.');
    setStatus('report-example-status', 'Example loaded. Saved results and figures are ready to inspect.');
  } catch (error) {
    setStatus('report-example-status', error.message, 'error');
  } finally {
    button.disabled = !document.getElementById('report-example-select').value;
  }
}

async function openSavedStudy(options = {}) {
  const studyId = options.studyId || document.getElementById('saved-study-id').value.trim();
  if (!studyId) {
    setStatus('saved-study-status', 'Enter a Study ID first.', 'error');
    return;
  }
  const request = {
    study_id: studyId, work_dir: options.workDir || executionWorkDir(),
    execution_target: options.executionTarget || selectedExecutionTarget(),
  };
  const openingSession = activeTaskSession();
  try {
    const saved = await postJson('/api/study-open', request);
    if (activeTaskSession() !== openingSession) return;
    const plan = saved.plan || saved.initial_scan_plan || {};
    const existing = taskSessions.find((session) => session.currentSavedStudy
      && session.currentSavedStudy.studyId === studyId
      && session.currentSavedStudy.workDir === saved.work_dir
      && session.currentSavedStudy.executionTarget === request.execution_target);
    if (existing) {
      switchTaskSession(existing.id);
    } else {
      // Static plans already contain expanded cases. This is a display projection,
      // never an input to saved start/review/collect operations.
      const cases = plan.cases || [];
      const displaySpec = saved.study_spec || (plan.grid_refinement_source && Object.keys(plan.grid_refinement_source).length
        ? {...deepCopy(plan.grid_refinement_source), grid_refinement: deepCopy(plan.grid_refinement)} : null) || {
        name: plan.name, system_type: plan.system_type, objective: plan.objective,
        observables: plan.observables, base_task: cases.length ? cases[0].request : {},
        case_design: { mode: 'cases', cases: cases.map((item) => ({
          label: item.label, variables: item.variables, request_updates: item.request,
        })) },
      };
      createTaskSession({ studySpec: displaySpec,
        plannerTemplate: plan.comparison && plan.comparison.mode === 'hamiltonian_dataset_assembly'
          ? 'hamiltonian_dataset' : 'standard',
        systemType: plan.system_type, studyMode: saved.mode });
    }
    currentSavedStudy = { studyId, workDir: saved.work_dir, executionTarget: request.execution_target };
    return await refreshSavedStudy(saved);
  } catch (error) {
    if (activeTaskSession() === openingSession) {
      setStatus('saved-study-status', error.message, 'error');
    }
  }
}

async function refreshSavedStudy(loaded = null, reconcileCompletedReview = true) {
  stopSavedStudyPolling();
  if (!currentSavedStudy) return;
  const version = savedStudyRefreshVersion;
  const identity = currentSavedStudy;
  const request = savedStudyRequest();
  try {
    const saved = loaded || await postJson('/api/study-open', request);
    if (currentSavedStudy !== identity || version !== savedStudyRefreshVersion) return;
    currentExecution = null;
    currentReport = null;
    currentAdaptivePreview = null;
    const report = saved.report;
    const pending = report && report.pending_review;
    if (typeof plannerState !== 'undefined') {
      plannerState.pendingAction = pending && pending.kind === 'retry_cases' ? pending.retry_action : null;
      if (typeof renderRetryPreview === 'function') renderRetryPreview();
    }
    const plan = saved.plan || saved.initial_scan_plan;
    if (plan && plan.grid_refinement && plan.grid_refinement.enabled && plan.grid_refinement_source) {
      document.getElementById('study-spec').value = prettyJson({
        ...deepCopy(plan.grid_refinement_source), grid_refinement: deepCopy(plan.grid_refinement),
      });
    }
    lockWorkDirectory(`${saved.work_dir}/${saved.study_id}`, saved.work_dir);
    document.getElementById('study-mode').value = saved.mode;
    document.getElementById('execution-target').value = identity.executionTarget;
    syncStudyModeControls();
    syncSavedStudyControls();
    if (saved.mode === 'adaptive' && !report) currentAdaptivePreview = saved;
    renderPlan(plan);
    const dataset = plan && plan.comparison && plan.comparison.dataset_spec;
    if (dataset) {
      const fields = { 'dataset-id': dataset.dataset_id, 'dataset-name': dataset.name,
        'dataset-molecule-count': dataset.target_molecule_count,
        'dataset-frame-count': dataset.geometries_per_molecule,
        'dataset-split-protocol': dataset.split_protocol, 'dataset-split-seed': dataset.split_seed };
      Object.entries(fields).forEach(([id, value]) => { document.getElementById(id).value = value; });
      document.getElementById('dataset-seed-status').textContent = `${plan.cases.length} seed molecules are saved in this Study plan.`;
    }
    if (report) renderReport(report);
    // An approved subset is a saved review, not a replacement Study.
    if (pending) {
      // Keep the complete plan and the report object installed by renderReport.
      // Replacing either with the retry subset hides the other cases and
      // invalidates the in-flight postprocessing context for the full report.
      if (pending.status !== 'review_required') renderReviewGate(null);
      if (pending.message) {
        document.getElementById('report-summary').innerHTML += `<p>${escapeHtml(pending.message)}</p>`;
      }
      if (pending.can_run && pending.plan && pending.plan.cases) {
        const costGate = planCostReviewGate(pending.plan, null,
          (pending.workflow && pending.workflow.active_plan_kind) || 'static');
        if (costGate) renderReviewGate(costGate);
      }
    } else if (!report || (report.grid_refinement || {}).status === 'cost_approval_required') {
      renderReviewGate(planCostReviewGate(plan, plan && plan.cost_estimate,
        saved.mode === 'adaptive' ? 'initial' : 'static'));
    }
    const gridStatus = (report && report.grid_refinement || {}).status;
    const canResumeGrid = ['needs_resume', 'running', 'awaiting_results', 'insufficient_evidence', 'cost_approval_required'].includes(gridStatus);
    document.getElementById('run-study').disabled = Boolean(currentReviewGate) || Boolean(report && !(pending && pending.can_run) && !canResumeGrid);
    snapshotCurrentTaskSession();
    const { execution } = await postJson('/api/study-execution-status', request);
    if (currentSavedStudy !== identity || version !== savedStudyRefreshVersion) return;
    document.getElementById('collect-saved-study').disabled = !execution.can_collect || Boolean(pending);
    syncStudyStopButton(execution);
    const running = execution.agent && execution.agent.running;
    // A retry can finish between reading the report and reading its status.
    // Re-read once when that status confirms the saved review was consumed.
    if (reconcileCompletedReview && pending && !running && execution.study_status
        && !execution.pending_review && execution.agent && execution.agent.finished_at) {
      return await refreshSavedStudy(null, false);
    }
    if (running || (!pending && ['running', 'interrupted', 'connection_interrupted', 'submission_unknown'].includes(execution.status))) {
      currentExecution = { studyId: identity.studyId, workDir: identity.workDir,
        executionTarget: identity.executionTarget, mode: saved.mode, savedStudy: true };
      renderExecutionRecovery(execution);
    }
    if (!pending) renderLiveStudyTable(execution);
    if (running) renderRunningGridProgress(execution, plan, report);
    const displayStatus = execution.status === 'cancelled' ? 'cancelled' : running ? 'running' : pending ? pending.status : report ? report.status : 'prepared';
    setStatus('saved-study-status', `${saved.study_id} · ${displayStatus} · ${executionStatusSummary(execution, report)}`);
    snapshotCurrentTaskSession();
    scheduleSavedStudyPolling(identity, execution);
    return true;
  } catch (error) {
    if (currentSavedStudy === identity && version === savedStudyRefreshVersion) {
      document.getElementById('run-study').disabled = true;
      setStatus('saved-study-status', `${error.message} Use Refresh Status to reconnect.`, 'error');
    }
  }
}

async function startSavedStudy() {
  if (typeof plannerState !== 'undefined' && plannerState.pendingAction && plannerState.pendingAction.kind === 'retry_cases') {
    renderRetryPreview();
    setStatus('run-status', 'Review the retry scope and select Confirm Retry.');
    return;
  }
  if (!currentSavedStudy || currentReviewGate) return;
  document.getElementById('run-study').disabled = true;
  try {
    const result = await postJson('/api/study-start', savedStudyRequest());
    setStatus('run-status', result.started ? 'Agent started. You can close this page and reopen the Study from Saved Studies.' : 'Agent is already running.', 'ok');
    await refreshSavedStudy();
  } catch (error) {
    renderReturnedCostReview(error, currentStudyMode() === 'adaptive' ? 'initial' : 'static');
    setStatus('run-status', error.message, 'error');
  }
}

async function collectSavedStudy() {
  if (!currentSavedStudy) return;
  document.getElementById('collect-saved-study').disabled = true;
  try {
    const result = await postJson('/api/study-execution-collect', savedStudyRequest());
    setStatus('run-status', result.message || 'Saved Study results collected.', 'ok');
    await refreshSavedStudy();
  } catch (error) {
    setStatus('saved-study-status', error.message, 'error');
  }
}

async function stopSavedStudy() {
  if (!currentSavedStudy || savedStudyStopping) return;
  const identity = currentSavedStudy;
  const request = savedStudyRequest();
  savedStudyStopping = true;
  stopSavedStudyPolling();
  syncStudyStopButton(null);
  document.getElementById('run-study').disabled = true;
  setStatus('run-status', 'Stopping calculation…');
  try {
    const result = await postJson('/api/study-stop', request);
    if (currentSavedStudy !== identity) return;
    if (!result.cancelled) throw new Error(result.message);
    setStatus('run-status', result.message, 'ok');
  } catch (error) {
    if (currentSavedStudy === identity) setStatus('run-status', error.message, 'error');
  } finally {
    savedStudyStopping = false;
    if (currentSavedStudy === identity) await refreshSavedStudy();
  }
}
