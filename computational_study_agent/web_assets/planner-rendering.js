function renderPlanCaseTable(plan, decisions = [], results = [], caseReports = []) {
  const decisionByCaseId = {};
  (Array.isArray(decisions) ? decisions : []).forEach((item) => {
    if (item && item.case_id) {
      decisionByCaseId[item.case_id] = item;
    }
  });
  const resultByCaseId = {};
  (Array.isArray(results) ? results : []).forEach((item) => {
    if (item && item.case_id) {
      resultByCaseId[item.case_id] = item;
    }
  });
  const structuredByCaseId = {};
  (Array.isArray(caseReports) ? caseReports : []).forEach((item) => {
    if (item && item.case_id && item.task_report && item.task_report.structured_results) {
      structuredByCaseId[item.case_id] = item.task_report.structured_results;
    }
  });
  const hasAdaptiveDetails = (Array.isArray(decisions) && decisions.length > 0) || (Array.isArray(results) && results.length > 0) || (Array.isArray(caseReports) && caseReports.length > 0);
  // Clarification responses intentionally have no plan yet.
  const cases = plan && Array.isArray(plan.cases) ? plan.cases : [];
  const analysisRows = cases.map((item) => {
    const decision = decisionByCaseId[item.case_id] || {};
    const result = resultByCaseId[item.case_id] || {};
    const structured = structuredByCaseId[item.case_id] || {};
    const diagnostics = decision.diagnostics || structured.strong_correlation_diagnostics || structured.correlation_diagnostics || {};
    const risk = molecularRisk(diagnostics);
    const systemType = normalizeSystemType(plan.system_type || structured.task_type || currentSystemType());
    const parameterSummary = diagnostics.parameter_summary || {};
    const energyUnit = structured.energy_unit || '';
    const finalSolver = plannerSolverName(
      result.solver || decision.recovery_solver || decision.recommended_solver || (item.request && item.request.solver)
    );
    const finalMethod = result.method || decision.recovery_method || decision.recommended_method || decision.recommended_solver || (item.request && item.request.method);
    const gap = hasCellValue(result.gap)
      ? result.gap
      : formatEnergyLikeValue(
          hasCellValue(structured.gap)
            ? structured.gap
            : (hasCellValue(diagnostics.gap) ? diagnostics.gap : diagnosticScalarValue(diagnostics, 'many_body_gap', 'gap')),
          energyUnit,
        );
    const finalEnergy = firstCellValue(
      result.final_energy,
      result.energy,
      formatEnergyLikeValue(structured.final_energy, energyUnit),
      formatEnergyLikeValue(structured.energy, energyUnit)
    );
    const meanFieldGap = hasCellValue(result.mean_field_gap)
      ? result.mean_field_gap
      : formatEnergyLikeValue(
          hasCellValue(structured.mean_field_gap) ? structured.mean_field_gap : diagnosticScalarValue(diagnostics, 'mean_field_homo_lumo_gap', 'gap'),
          energyUnit,
        );
    const naturalOccupationFractionality = hasCellValue(result.natural_occupation_fractionality)
      ? result.natural_occupation_fractionality
      : (
          hasCellValue(structured.natural_occupation_fractionality)
            ? structured.natural_occupation_fractionality
            : firstCellValue(
                diagnosticScalarValue(diagnostics, 'natural_orbital_occupations', 'average_fractionality'),
                diagnosticScalarValue(diagnostics, 'natural_orbital_occupations', 'max_fractionality'),
                molecularRiskComponentValue(diagnostics, 'correlated_natural_occupations', 'average_frontier_fractionality'),
                molecularRiskComponentValue(diagnostics, 'correlated_natural_occupations', 'max_fractionality')
              )
        );
    const maxDoubleExcitationAmplitude = hasCellValue(result.max_double_excitation_amplitude)
      ? result.max_double_excitation_amplitude
      : (
          hasCellValue(structured.max_double_excitation_amplitude)
            ? structured.max_double_excitation_amplitude
            : firstCellValue(
                diagnosticScalarValue(diagnostics, 'max_double_excitation_amplitude', 'max_abs_t2'),
                molecularRiskComponentValue(diagnostics, 'max_double_excitation_amplitude', 'max_abs_t2')
              )
        );
    const activeSpace = activeSpaceLabel(decision.active_space_contract || structured.active_space);
    const row = {
      case_id: item.case_id,
      label: item.label,
      ...item.variables,
      ...(systemType === 'model_hamiltonian'
        ? { solver: finalSolver }
        : { method: finalMethod }),
    };
    if (hasAdaptiveDetails) {
      row.correlation = reviewCorrelationLevel(decision, result, structured);
      if (systemType === 'model_hamiltonian') {
        row.filling = hasCellValue(parameterSummary.filling) ? parameterSummary.filling : modelFillingFromCase(item);
        row.solver_stress = diagnostics.solver_stress_level || '';
      }
      row.gap = hasCellValue(gap) ? gap : '';
      row.mean_field_gap = hasCellValue(meanFieldGap) ? meanFieldGap : '';
      row.natural_occupation_fractionality = hasCellValue(naturalOccupationFractionality) ? naturalOccupationFractionality : '';
      row.max_double_excitation_amplitude = hasCellValue(maxDoubleExcitationAmplitude) ? maxDoubleExcitationAmplitude : '';
      if (systemType === 'molecular') {
        row.active_space = activeSpace;
      }
      row.final_energy = hasCellValue(finalEnergy) ? finalEnergy : '';
      row.status = result.status || decision.status || '';
    }
    return row;
  });
  const sparseDiagnosticColumns = new Set([
    'solver_stress',
    'gap',
    'mean_field_gap',
    'natural_occupation_fractionality',
    'max_double_excitation_amplitude',
    'active_space',
  ]);
  const displayRows = analysisRows.map((row) => {
    const displayRow = {};
    Object.entries(row).forEach(([key, value]) => {
      if (sparseDiagnosticColumns.has(key)) {
        return;
      }
      if (key === 'status' && hasAdaptiveDetails) {
        displayRow.diagnostics = diagnosticSummaryText(row);
      }
      displayRow[key] = value;
    });
    return displayRow;
  });
  renderTable('case-table', displayRows, { className: 'case-table' });
  return analysisRows;
}

function renderPlan(plan) {
  if (plan && plan._review_kind) throw new Error('A retry subset cannot replace the full Study plan');
  installFullStudyPlan(plan);
  currentAnalysisRows = [];
  updateTaskSetupPanel();
  renderReviewGate(null);
  renderPlanSummary(plan);
  renderPlanCaseTable(plan);
  renderAdaptiveState(currentAdaptivePreview);
  updateActiveTaskSessionStatus(plan ? 'planned' : 'draft');
  snapshotCurrentTaskSession();
}

function renderReport(report) {
  if (!report || typeof report !== 'object') {
    currentReport = null;
    currentAnalysisRows = [];
    document.getElementById('report-summary').className = 'empty';
    document.getElementById('report-summary').innerHTML = 'No run yet.';
    document.getElementById('analyze-study-results').disabled = true;
    document.getElementById('study-result-analysis').className = 'empty analysis-box';
    document.getElementById('study-result-analysis').textContent = 'Run first.';
    resetPostprocessing();
    renderAdaptiveState(null);
    renderReviewGate(null);
    snapshotCurrentTaskSession();
    return;
  }
  currentReport = report;
  if (typeof report.work_dir === 'string' && report.work_dir.trim()) {
    lockWorkDirectory(report.work_dir, executionWorkDir());
  }
  currentAnalysisRows = [];
  document.getElementById('report-summary').className = '';
  document.getElementById('report-summary').innerHTML = [
    `<p>${escapeHtml(report.summary || '')}</p>`,
    renderDatasetManifestSummary(report.dataset_manifest),
    renderGridRefinementSummary(report.grid_refinement),
    renderDmetBranchSummary(report),
  ].join('');
  if (report.adaptive && (currentStudyMode() === 'adaptive' || report.adaptive.mode || report.adaptive.initial_scan_plan)) {
    currentAdaptivePreview = report;
    const workflow = adaptiveWorkflow(report);
    const activePlan = adaptivePlanByKind(
      report.adaptive,
      workflow && workflow.active_plan_kind ? workflow.active_plan_kind : 'refined',
    ) || report.adaptive.refined_plan;
    if (activePlan && !currentSavedStudy) {
      installFullStudyPlan(activePlan);
      renderPlanSummary(activePlan);
      currentAnalysisRows = renderPlanCaseTable(activePlan, adaptiveDecisionEntries(report), report.comparison_table || [], report.cases || []);
    } else if (currentSavedStudy && currentPlan) {
      currentAnalysisRows = renderPlanCaseTable(currentPlan, adaptiveDecisionEntries(report), report.comparison_table || [], report.cases || []);
    }
    renderAdaptiveState(report);
    renderReviewGate(reviewGateForStudyReport(report));
  } else {
    currentAdaptivePreview = null;
    currentAnalysisRows = Array.isArray(report.comparison_table) ? deepCopy(report.comparison_table) : [];
    renderTable('case-table', report.comparison_table || [], { className: 'case-table' });
    renderAdaptiveState(null);
    renderReviewGate(reviewGateForStudyReport(report));
  }
  document.getElementById('analyze-study-results').disabled = !LLM_CONFIGURED;
  document.getElementById('study-result-analysis').className = 'empty analysis-box';
  document.getElementById('study-result-analysis').textContent = LLM_CONFIGURED
    ? 'LLM analysis has not been requested yet.'
    : 'LLM is not configured for result analysis.';
  const hasPostprocessRows = postprocessRows().length > 0;
  resetPostprocessing(
    hasPostprocessRows
      ? 'Suggest plots or generate defaults.'
      : postprocessingUnavailableMessage(),
  );
  document.getElementById('suggest-plots').disabled = !hasPostprocessRows;
  document.getElementById('generate-default-plots').disabled = !hasPostprocessRows;
  updateCustomPlotControls();
  updateActiveTaskSessionStatus(report && report.status === 'succeeded' ? 'completed' : 'failed');
  snapshotCurrentTaskSession();
  refreshPostprocessContext();
  const phasePlots = (report.artifacts || []).filter(a => a.kind === 'postprocess-plot' && /\/dmet-phases\//.test(a.path || ''));
  if (phasePlots.length) renderPostprocessArtifacts(phasePlots);
}

function handleStudyModeChange() {
  syncAdaptiveActiveSpaceControls();
  updateTaskSetupPanel();
  installFullStudyPlan(null);
  currentReport = null;
  currentAdaptivePreview = null;
  currentAnalysisRows = [];
  document.getElementById('run-study').disabled = true;
  document.getElementById('case-table').className = 'empty';
  document.getElementById('case-table').innerHTML = 'Build a plan to see cases.';
  renderReviewGate(null);
  document.getElementById('report-summary').className = 'empty';
  document.getElementById('report-summary').innerHTML = 'No run yet.';
  renderPlanSummary(null);
  renderAdaptiveState(null);
  setStatus('run-status', currentStudyMode() === 'adaptive' ? 'Adaptive scan mode selected. Build an initial-scan plan.' : 'Static scan mode selected. Build a plan.');
  updateActiveTaskSessionStatus('draft');
  snapshotCurrentTaskSession();
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.error || `HTTP ${response.status}`);
    error.payload = data;
    throw error;
  }
  return data;
}

function renderReturnedCostReview(error, planKind = 'static') {
  const costReview = error && error.payload && error.payload.cost_review;
  if (!costReview || typeof costReview !== 'object' || !currentPlan) {
    return false;
  }
  currentPlan.cost_estimate = deepCopy(costReview);
  const gate = planCostReviewGate(currentPlan, costReview, planKind);
  if (!gate) {
    return false;
  }
  renderReviewGate(gate);
  document.getElementById('run-study').disabled = true;
  return true;
}

function validationSummary(issues) {
  if (!Array.isArray(issues) || !issues.length) {
    return '';
  }
  const errors = issues.filter((item) => item && item.severity === 'error').length;
  const warnings = issues.filter((item) => item && item.severity === 'warning').length;
  const first = issues.find((item) => item && item.message);
  const pieces = [];
  if (errors) {
    pieces.push(`${errors} validation error${errors === 1 ? '' : 's'}`);
  }
  if (warnings) {
    pieces.push(`${warnings} validation warning${warnings === 1 ? '' : 's'}`);
  }
  if (first && first.message) {
    pieces.push(first.message);
  }
  return pieces.join(': ');
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

function wikiEvidenceSummary(evidence) {
  const pages = evidence && Array.isArray(evidence.pages) ? evidence.pages : [];
  if (!pages.length) {
    return 'Retrieved Wiki Rules: none.';
  }
  return `Retrieved Wiki Rules:\n${pages.slice(0, 6).map((page) => `- ${page.title || page.slug || 'Untitled rule'}`).join('\n')}`;
}

function executionStatusSummary(execution) {
  const payload = execution && typeof execution === 'object' ? execution : {};
  const expected = Number(payload.expected || payload.task_count || 0);
  const terminal = Number(payload.terminal || 0);
  const available = Number(payload.report_available || 0);
  const state = String(payload.status || 'unknown').replace(/_/g, ' ');
  const counts = payload.task_status_counts;
  const planned = Number(payload.task_count || 0);
  const labels = {pending: 'in progress', not_executed: 'awaiting execution'};
  const breakdown = counts && typeof counts === 'object' ? Object.entries(counts)
    .filter(([, count]) => Number(count) > 0)
    .map(([status, count]) => `${Number(count)} ${labels[status] || status.replace(/_/g, ' ')}`) : [];
  const pieces = [planned && breakdown.length
    ? `Execution: ${state}; ${planned} planned points: ${breakdown.join(', ')}.`
    : `Execution: ${state}; ${terminal}/${expected} terminal; ${available} report${available === 1 ? '' : 's'} available.`,
  ];
  const dataset = payload.dataset && typeof payload.dataset === 'object' ? payload.dataset : null;
  if (dataset) {
    const expectedTrajectories = Number(dataset.expected_trajectories || 0);
    const terminalTrajectories = Number(dataset.terminal_trajectories || 0);
    const succeededTrajectories = Number(dataset.succeeded_trajectories || 0);
    const failedTrajectories = Number(dataset.failed_trajectories || 0);
    const expectedStructures = Number(dataset.expected_structures || 0);
    const availableStructures = Number(dataset.available_structures || 0);
    pieces.push(
      `Dataset: ${terminalTrajectories}/${expectedTrajectories} trajectories terminal `
      + `(${succeededTrajectories} succeeded, ${failedTrajectories} failed); `
      + `${availableStructures}/${expectedStructures} structures available.`,
    );
    if (dataset.finalized) {
      pieces.push(
        `Finalized: ${Number(dataset.accepted_structures || 0)} accepted, `
        + `${Number(dataset.rejected_structures || 0)} rejected.`,
      );
    }
    if (dataset.generated) {
      const location = dataset.generation_location === 'remote_executor' ? 'server' : 'local machine';
      pieces.push(
        `Portable dataset generated on the ${location} `
        + `(${Number(dataset.generated_trajectory_files || 0)} trajectory files).`,
      );
    }
    if (dataset.collected) {
      pieces.push(`Dataset collected locally (${Number(dataset.collected_trajectory_files || 0)} trajectory files).`);
    }
  }
  return pieces.join(' ');
}

function renderLiveStudyTable(execution) {
  const progress = execution && execution.progress;
  if (!progress || !Array.isArray(progress.comparison_table)) return;
  const table = document.getElementById('case-table');
  const scrollLeft = table.scrollLeft, scrollTop = table.scrollTop;
  renderTable('case-table', progress.comparison_table, {className: 'case-table'});
  table.scrollLeft = scrollLeft;
  table.scrollTop = scrollTop;
}

function renderRunningGridProgress(execution, plan, report) {
  if (!(execution.agent && execution.agent.running) || !(plan && plan.grid_refinement && plan.grid_refinement.enabled)) return;
  const planned = Math.max(Number(execution.task_count || 0), (plan.cases || []).length);
  const savedGrid = execution.progress && execution.progress.grid_refinement || report && report.grid_refinement;
  const seedCount = savedGrid && savedGrid.seed_point_count;
  const gridCounts = Number.isInteger(seedCount) && seedCount <= planned
    ? `${seedCount} initial points + ${planned - seedCount} added / ${plan.grid_refinement.max_new_points} allowed.`
    : `${planned} planned points.`;
  const liveRows = execution.progress && execution.progress.comparison_table;
  const rows = report && report.comparison_table || [];
  document.getElementById('report-summary').className = '';
  document.getElementById('report-summary').innerHTML =
    `<p>${escapeHtml(executionStatusSummary(execution))}</p>`
    + `<div class="analysis-box"><strong>Parameter grid: Refining the grid</strong><p>${escapeHtml(gridCounts)}</p></div>`
    + (Array.isArray(liveRows)
      ? '<p>All planned points are shown in the table, including points awaiting execution. Status and completed results refresh automatically every 5 seconds.</p>'
      : `<p>The results table shows the last saved batch (${rows.length} points). Use Refresh Status to reconnect to live results.</p>`);
}

function renderExecutionRecovery(execution) {
  if (!currentExecution) return;
  currentExecution.execution = deepCopy(execution || {});
  if (typeof syncHamiltonianDatasetActionControls === 'function') {
    syncHamiltonianDatasetActionControls();
  }
  const canCollect = Boolean(execution && execution.can_collect);
  renderReviewGate({
    kind: 'execution_recovery',
    severity: execution && execution.status === 'connection_interrupted' ? 'warning' : 'warning',
    kicker: 'Study Execution',
    title: canCollect ? 'Results are ready to collect' : 'Study execution is saved',
    status: canCollect ? 'RESULTS AVAILABLE' : String((execution && execution.status) || 'CHECK STATUS').replace(/_/g, ' ').toUpperCase(),
    message: executionStatusSummary(execution),
    hint: 'Checking or collecting uses the persisted execution receipt. Completed tasks will not be submitted again.',
    actions: [
      { id: 'check_execution_status', label: 'Check Status', secondary: false },
      ...(canCollect ? [{ id: 'collect_execution_results', label: 'Collect Results', secondary: false }] : []),
      { id: 'cancel_execution_review', label: 'Close', secondary: true },
    ],
  });
  document.getElementById('run-study').disabled = true;
}

async function inspectCurrentExecution() {
  return refreshSavedStudy();
}

async function collectCurrentExecution() {
  return collectSavedStudy();
}

function executionConversationIntent(message) {
  if (!currentExecution) return '';
  const text = String(message || '').trim().toLowerCase();
  if (/(collect|fetch|resume|continue|取回|收集|整合|继续)/.test(text)) return 'collect';
  if (/(check|status|progress|inspect|检查|状态|进度|完成了|结束了)/.test(text)) return 'inspect';
  return '';
}

async function draftWithLlm() {
  const goal = document.getElementById('goal').value.trim();
  if (!goal) {
    setStatus('draft-status', 'Send a planner message first.', 'error');
    return;
  }
  appendPlannerMessage('user', goal);
  document.getElementById('goal').value = '';
  const executionIntent = executionConversationIntent(goal);
  if (executionIntent) {
    setStatus('draft-status', executionIntent === 'collect' ? 'Collecting the recoverable execution...' : 'Checking the recoverable execution...');
    if (executionIntent === 'collect') {
      await collectCurrentExecution();
    } else {
      await inspectCurrentExecution();
    }
    setStatus('draft-status', 'Planner execution check completed.', 'ok');
    return;
  }
  if (isHamiltonianDatasetPlanner()) {
    const message = 'The Hamiltonian Dataset template is deterministic. Import 100 seed geometries and select Build Plan; no LLM draft is required.';
    appendPlannerMessage('assistant', message);
    setStatus('draft-status', message, 'ok');
    return;
  }
  if (!LLM_CONFIGURED && !currentSavedStudy) {
    const message = 'LLM is not configured. Set PYSCF_AGENT_LLM_BASE_URL and PYSCF_AGENT_LLM_MODEL.';
    appendPlannerMessage('system', message);
    setStatus('draft-status', message, 'error');
    return;
  }
  const requestContext = plannerRequestContext();
  const draftIdentity = typeof currentSavedStudy !== 'undefined' ? currentSavedStudy : null;
  const currentSpec = requestContext.studySpec;
  appendPlannerMessage('trace', planningDecomposition(goal, currentSpec));
  setStatus('draft-status', 'Planner is drafting an updated plan...');
  document.getElementById('draft-with-llm').disabled = true;
  try {
    // The structured StudySpec is the authoritative context. Feeding old
    // assistant errors back as part of the goal makes a one-point revision
    // look like a new incomplete study to the LLM.
    const payload = await postJson('/api/study-llm-draft', {
      ...(typeof currentSavedStudy !== 'undefined' && currentSavedStudy ? savedStudyRequest() : {}),
      goal,
      study_spec: currentSpec,
      study_mode: requestContext.mode,
      adaptive: requestContext.adaptive,
      study_state: studyStateForReviewAction(),
      locale: 'en',
    });
    if ((typeof currentSavedStudy !== 'undefined' ? currentSavedStudy : null) !== draftIdentity) return;
    if (payload.intent === 'retry_cases' || payload.needs_input) {
      if (payload.retry_action) {
        plannerState.pendingAction = payload.retry_action;
        renderRetryPreview();
        await refreshSavedStudy();
      }
      appendPlannerMessage('assistant', payload.message || (payload.retry_action
        ? `Retry preview ready: ${payload.retry_action.execution_case_ids.join(', ')}. Review and confirm below.`
        : 'Select the case rows to retry and set the parameters in the retry form.'));
      setStatus('draft-status', 'Full Study results retained.', 'ok');
      return;
    }
    if (payload.intent === 'analyze') {
      await analyzeStudyResults();
      return;
    }
    document.getElementById('study-spec').value = prettyJson(payload.study_spec);
    syncStudySystemFromSpec(payload.study_spec);
    syncModelInputFileFromStudySpec();
    currentReport = null;
    currentAdaptivePreview = null;
    renderPlan(payload.plan || null);
    renderReviewGate(null);
    document.getElementById('run-study').disabled = true;
    appendPlannerMessage('trace', wikiEvidenceSummary(payload.wiki_evidence));
    if (payload.status === 'needs_input') {
      appendPlannerMessage('assistant', plannerClarificationMessage(payload));
      setStatus('draft-status', validationSummary(payload.validation_issues) || payload.message || 'Planner needs more information.', 'error');
      return;
    }
    const summary = summarizePlannerSpec(payload.study_spec);
    appendPlannerMessage('assistant', `${payload.message || 'Draft updated.'} Select Build Plan to save the Study for review and execution.\n\n${summary}`);
    setStatus('draft-status', validationSummary(payload.validation_issues) || 'Draft ready. Build Plan to save it.', 'ok');
  } catch (error) {
    const message = errorDetailMessage(error.payload || {}, error.message);
    appendPlannerMessage('system', message);
    setStatus('draft-status', message, 'error');
  } finally {
    document.getElementById('draft-with-llm').disabled = false;
  }
}

async function buildPlan() {
  const button = document.getElementById('build-plan');
  if (currentSavedStudy || button.disabled) return;
  button.disabled = true;
  const session = activeTaskSession();
  const executionTarget = selectedExecutionTarget();
  setStatus('run-status', 'Preparing and saving the Study...');
  try {
    const datasetPlanner = isHamiltonianDatasetPlanner();
    const requestContext = datasetPlanner ? null : plannerRequestContext();
    const inputs = datasetPlanner ? hamiltonianDatasetPlanningPayload() : {
      study_spec: requestContext.studySpec, adaptive: requestContext.adaptive,
    };
    const saved = await postJson('/api/study-prepare', {
      ...inputs, work_dir: executionWorkDir(), execution_target: executionTarget,
      resource_profile: selectedResourceProfile(),
    });
    const identity = { studyId: saved.study_id, workDir: saved.work_dir, executionTarget };
    if (session) session.currentSavedStudy = deepCopy(identity);
    if (session !== activeTaskSession()) {
      if (session) session.status = 'planned';
      renderTaskSessionSelector();
      return;
    }
    currentSavedStudy = identity;
    await refreshSavedStudy(saved);
    const message = saved.message || 'Study saved. Review the plan before selecting Run Plan.';
    appendPlannerMessage('assistant', `${saved.study_id}: ${message}`);
    setStatus('draft-status', '');
    setStatus('run-status', validationSummary(saved.validation_issues) || message, 'ok');
    snapshotCurrentTaskSession();
  } catch (error) {
    const message = errorDetailMessage(error.payload || {}, error.message);
    appendPlannerMessage('system', message);
    setStatus('run-status', message, 'error');
  } finally {
    button.disabled = Boolean(currentSavedStudy);
  }
}

async function runStudy() {
  if (!currentSavedStudy) {
    setStatus('run-status', 'Build Plan to save the Study before running.', 'error');
    return;
  }
  return startSavedStudy();
}

function pathRestartArtifactLabel(artifact) {
  const payload = artifact && typeof artifact === 'object' ? artifact : {};
  const deferredCaseId = String(payload.deferred_case_id || '').trim();
  if (deferredCaseId) {
    return `generated by ${deferredCaseId} earlier in this branch`;
  }
  const path = String(payload.path || '').trim();
  const filename = path ? path.split(/[\\/]/).pop() : '';
  const checksum = String(payload.sha256 || '').trim();
  const checksumLabel = checksum ? `sha256=${checksum.slice(0, 12)}` : '';
  return [
    filename || String(payload.kind || 'one_particle_state'),
    checksumLabel,
  ].filter(Boolean).join(', ');
}

function pathRestartApprovalBranches(item) {
  if (Array.isArray(item.branches) && item.branches.length) {
    return item.branches;
  }
  return ['left', 'right'].map((side) => ({
    direction: side === 'left' ? 'left_to_right' : 'right_to_left',
    target_case_id: item.case_id,
    target_method: item.target_method,
    target_reference: item.target_reference,
    source_case_id: item[`${side}_source_case_id`],
    source_coordinate: item[`${side}_source_coordinate`],
    source_method: item[`${side}_source_method`],
    source_reference: item[`${side}_source_reference`],
    initial_state_kind: item.initial_state_kind,
    source_artifact: item[`${side}_source_artifact`] || {},
  }));
}

function pathRestartBranchDetail(branch) {
  const leftAnchored = branch.direction === 'left_to_right';
  const branchLabel = leftAnchored ? 'Left-anchored branch' : 'Right-anchored branch';
  const method = String(branch.source_method || '').toUpperCase() || 'unknown method';
  const reference = String(branch.source_reference || '').toUpperCase() || 'unknown reference';
  const coordinate = branch.source_coordinate === undefined || branch.source_coordinate === null
    ? ''
    : `coordinate=${branch.source_coordinate}`;
  const artifact = pathRestartArtifactLabel(branch.source_artifact);
  const sourceMode = String(branch.source_mode || '').trim();
  const sourceDescription = sourceMode === 'endpoint_baseline'
    ? 'endpoint baseline'
    : (sourceMode === 'branch_continuation' ? 'adjacent branch result' : 'external anchor');
  return [
    `${branchLabel} uses AO 1RDM from ${branch.source_case_id || 'unknown source'}`,
    `${method} (${reference})`,
    sourceDescription,
    coordinate,
    artifact ? `artifact=${artifact}` : '',
    `initializes ${branch.target_case_id || 'the target case'}`,
  ].filter(Boolean).join(' | ');
}

function pathRestartApprovalGate(pathRestart) {
  const approval = pathRestart && pathRestart.approval;
  if (!approval || approval.status !== 'approval_required' || !approval.approval_token) {
    return null;
  }
  const items = Array.isArray(approval.items) ? approval.items : [];
  const endpointValidation = items.some(
    (item) => item.approval_reason === 'endpoint_bidirectional_validation',
  );
  return {
    kind: 'path_restart_approval',
    severity: 'warning',
    kicker: 'Continuation Review',
    title: endpointValidation
      ? 'Approve endpoint 1RDM validation'
      : 'Approve bidirectional 1RDM restart',
    status: 'approval required',
    message: endpointValidation
      ? 'The endpoint has only one-sided scan history. The local window can be recomputed from left to right and right to left without changing its target methods.'
      : 'The flagged local window can be retried without changing its target methods by propagating compatible densities from both directions.',
    hint: 'The transferred state is the converged SCF-reference AO 1RDM, not a correlated wavefunction. Each interior task uses the 1RDM produced by the preceding task in that branch, and both directions must agree before any result is replaced.',
    approval_token: approval.approval_token,
    case_ids: Array.isArray(approval.case_ids) ? approval.case_ids : [],
    items: items.map((item) => {
      const variables = Object.entries(item.variables || {})
        .map(([key, value]) => `${key}=${value}`)
        .join(', ');
      const windowIds = Array.isArray(item.window_case_ids) ? item.window_case_ids.join(', ') : '';
      const branches = pathRestartApprovalBranches(item);
      return {
        title: `${item.case_id || 'case'}${item.label ? ` · ${item.label}` : ''}`,
        detail: [
          `Target restart: ${item.case_id || 'case'} | ${String(item.target_method || '').toUpperCase()} (${String(item.target_reference || '').toUpperCase()})${variables ? ` | ${variables}` : ''}`,
          ...branches.map((branch) => pathRestartBranchDetail(branch)),
          windowIds ? `Validation window: ${windowIds}. Both branches must agree before any result is replaced.` : '',
        ].filter(Boolean).join('\n'),
        status: 'approval required',
      };
    }),
    actions: [
      { id: 'approve_path_restart', label: 'Approve 1RDM Restart', secondary: false },
      { id: 'skip_path_restart', label: 'Skip to Method Review', secondary: true },
      { id: 'cancel_path_restart_review', label: 'Cancel', secondary: true },
    ],
  };
}

function mpsContinuationApprovalGate(payload) {
  const approval = payload && payload.approval;
  if (!approval || approval.status !== 'approval_required' || !approval.approval_token) {
    return null;
  }
  return {
    kind: 'mps_continuation_approval',
    severity: 'warning',
    kicker: 'MPS Continuation Review',
    title: `Approve ${Array.isArray(approval.case_ids) ? approval.case_ids.length : 0} same-method MPS retry case(s)`,
    status: 'approval required',
    message: 'These unresolved block2 cases can be retried without changing method by reusing the nearest compatible succeeded MPS checkpoint.',
    hint: 'Compatibility covers active-orbital count, electron sector, spin, symmetry, and root count. The source Hamiltonian may differ; block2 records whether it is identical when the retry starts.',
    approval_token: approval.approval_token,
    case_ids: Array.isArray(approval.case_ids) ? approval.case_ids : [],
    items: (Array.isArray(approval.items) ? approval.items : []).map((item) => {
      const checks = item.compatibility && item.compatibility.checks ? item.compatibility.checks : {};
      const passed = Object.entries(checks)
        .filter(([_key, value]) => Boolean(value))
        .map(([key]) => key.replaceAll('_', ' '))
        .join(', ');
      const artifact = item.source_artifact || {};
      return {
        title: `${item.case_id || 'case'}${item.label ? ` · ${item.label}` : ''}`,
        detail: [
          `Source MPS: ${item.source_case_id || 'unknown'}${item.source_label ? ` · ${item.source_label}` : ''}`,
          item.normalized_distance === null || item.normalized_distance === undefined
            ? ''
            : `normalized parameter distance=${Number(item.normalized_distance).toPrecision(4)}`,
          passed ? `compatible: ${passed}` : '',
          artifact.path ? `manifest=${artifact.path}` : `checkpoint=${item.source_scratch_directory || 'executor-local storage'}`,
          'Target method and solver settings are unchanged.',
        ].filter(Boolean).join('\n'),
        status: 'approval required',
      };
    }),
    actions: [
      { id: 'approve_mps_continuation', label: 'Approve MPS Retry', secondary: false },
      { id: 'skip_mps_continuation', label: 'Skip MPS Retry', secondary: true },
      { id: 'cancel_mps_continuation_review', label: 'Cancel', secondary: true },
    ],
  };
}

async function analyzeStudyResults() {
  if (!currentSavedStudy || !currentReport) {
    setStatus('analysis-status', 'Run a plan first.', 'error');
    return null;
  }
  if (!LLM_CONFIGURED) {
    setStatus('analysis-status', 'LLM is not configured.', 'error');
    return null;
  }
  setStatus('analysis-status', 'Analyzing...');
  document.getElementById('analyze-study-results').disabled = true;
  const identity = currentSavedStudy;
  try {
    const requestPayload = {
      ...savedStudyRequest(),
      locale: 'en',
      resource_profile: selectedResourceProfile(),
    };
    const payload = await postJson('/api/study-result-analysis', requestPayload);
    if (currentSavedStudy !== identity) return payload;
    await refreshSavedStudy();
    if (currentSavedStudy !== identity) return payload;
    const target = document.getElementById('study-result-analysis');
    target.className = 'analysis-box';
    const pathDiagnostics = payload.scan_path_diagnostics || {};
    const pathSummary = pathDiagnostics.status === 'review_required' && pathDiagnostics.summary
      ? `Scan-path review: ${pathDiagnostics.summary}`
      : '';
    const pathRestart = payload.path_restart || {};
    const restartSummary = ['validated', 'review_required', 'approval_required', 'skipped', 'unavailable'].includes(pathRestart.status) && pathRestart.summary
      ? `Continuation restart: ${pathRestart.summary}`
      : '';
    const mpsContinuation = payload.mps_continuation || {};
    const mpsSummary = ['validated', 'review_required', 'approval_required', 'skipped', 'unavailable'].includes(mpsContinuation.status) && mpsContinuation.summary
      ? `MPS continuation: ${mpsContinuation.summary}`
      : '';
    target.textContent = [mpsSummary, restartSummary, pathSummary, payload.result_analysis || 'No analysis was returned.']
      .filter(Boolean)
      .join('\n\n');
    if (restartSummary) {
      appendPlannerMessage('assistant', restartSummary);
    }
    if (mpsSummary) {
      appendPlannerMessage('assistant', mpsSummary);
    }
    if (pathSummary) {
      appendPlannerMessage('assistant', pathSummary);
    }
    const workflow = adaptiveWorkflow(currentReport);
    const workflowStage = workflow && workflow.stage ? workflow.stage : '';
    const mpsReviewActivated = workflowStage === 'mps_continuation_review_required';
    const restartReviewActivated = workflowStage === 'continuation_review_required';
    const entanglementReviewActivated = (
      workflowStage === 'active_space_review_required'
      && workflow.active_plan_kind === 'entanglement_active_space'
    );
    const reviewActivated = (
      ['active_space_review_required', 'path_refinement_review_required'].includes(workflowStage)
      && workflow.active_plan_kind === 'path_refinement'
    );
    if (mpsReviewActivated) {
      appendPlannerMessage('system', 'A compatible neighboring block2 MPS is available for the unresolved case(s). Review the source checkpoint and sector checks before authorizing same-method retries.');
      setStatus('analysis-status', 'MPS continuation requires approval.', 'ok');
    } else if (restartReviewActivated) {
      appendPlannerMessage('system', 'A bidirectional SCF-reference 1RDM continuation is available for the flagged local window. Review the seed artifacts and adjacent-task propagation chain before authorizing the retry.');
      setStatus('analysis-status', 'Continuation restart requires approval.', 'ok');
    } else if (entanglementReviewActivated) {
      appendPlannerMessage('system', 'Block2 entanglement and natural-occupation boundary evidence recommend a larger ActiveSpaceAudit. Review the proposed CAS expansion; an approved retry reuses optimized orbitals and initializes a fresh MPS.');
      setStatus('analysis-status', 'Entanglement-driven active-space expansion requires approval.', 'ok');
    } else if (reviewActivated) {
      appendPlannerMessage('system', 'Prepared a shared-size CASSCF ActiveSpaceAudit for the non-smooth transition. Review the proposed active spaces, approve or edit them, then Run Plan will recompute the full affected overlap window.');
      setStatus('analysis-status', 'Scan-path review requires active-space approval.', 'ok');
    } else {
      const gate = reviewGateForStudyReport(currentReport);
      if (gate) {
        renderReviewGate(gate);
      }
      setStatus('analysis-status', 'Analysis ready.', 'ok');
    }
    return payload;
  } catch (error) {
    const message = errorDetailMessage(error.payload || {}, error.message);
    if (currentSavedStudy === identity) setStatus('analysis-status', message, 'error');
    return null;
  } finally {
    if (currentSavedStudy === identity) document.getElementById('analyze-study-results').disabled = false;
  }
}
