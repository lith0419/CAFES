function currentStudyMode() {
  if (isHamiltonianDatasetPlanner()) {
    return 'static';
  }
  if (normalizeSystemType(document.getElementById('study-system').value) === 'model_hamiltonian') {
    return 'static';
  }
  return document.getElementById('study-mode').value || 'static';
}

function normalizeSystemType(value) {
  const normalized = String(value || 'molecular').trim().toLowerCase().replace(/[-\s]+/g, '_');
  if (normalized === 'model' || normalized === 'model_hamiltonian') {
    return 'model_hamiltonian';
  }
  if (normalized === 'molecule' || normalized === 'molecular_electronic_structure') {
    return 'molecular';
  }
  return normalized || 'molecular';
}

function currentStudySpecSafe() {
  try {
    const spec = parseStudySpec();
    return spec && typeof spec === 'object' && !Array.isArray(spec) ? spec : {};
  } catch (_error) {
    return {};
  }
}

function currentSystemType() {
  if (currentPlan && currentPlan.system_type) {
    return normalizeSystemType(currentPlan.system_type);
  }
  if (currentReport && currentReport.system_type) {
    return normalizeSystemType(currentReport.system_type);
  }
  return normalizeSystemType(currentStudySpecSafe().system_type);
}

function normalizedInitialScanStrategy(value) {
  const raw = String(value || '').trim().toLowerCase();
  const strategy = raw.replace(/[-\s]+/g, '_');
  const entries = Object.entries(ADAPTIVE_INITIAL_SCAN_STRATEGIES || {});
  const matched = entries.find(([id, config]) => {
    const aliases = Array.isArray(config.aliases) ? config.aliases : [];
    return [id, ...aliases].some((candidate) => {
      const normalized = String(candidate || '').trim().toLowerCase().replace(/[-\s]+/g, '_');
      return normalized === strategy || String(candidate || '').trim().toLowerCase() === raw;
    });
  });
  if (matched) {
    return matched[0];
  }
  if (raw) {
    return strategy;
  }
  const defaultEntry = entries.find(([_id, config]) => Boolean(config.default));
  return defaultEntry ? defaultEntry[0] : (entries[0] ? entries[0][0] : 'auto');
}

function adaptiveOptions() {
  const initialScanStrategy = normalizedInitialScanStrategy(document.getElementById('adaptive-initial-scan-strategy').value);
  const activeSpaceSolverElement = document.getElementById('adaptive-active-space-solver');
  const activeSpaceSolver = activeSpaceSolverElement ? activeSpaceSolverElement.value : 'auto';
  const options = {
    initial_scan_strategy: initialScanStrategy,
    active_space_solver: activeSpaceSolver,
  };
  if (activeSpaceSolver === 'block2_dmrg') {
    const localizationMethod = document.getElementById('adaptive-active-space-localization').value || 'none';
    const orbitalOrdering = document.getElementById('adaptive-orbital-ordering').value || 'canonical';
    const orbitalOrder = document.getElementById('adaptive-manual-orbital-order').value
      .split(/[\s,]+/)
      .map((item) => item.trim())
      .filter((item) => /^\d+$/.test(item))
      .map((item) => Number.parseInt(item, 10));
    options.active_space_orbital_processing = {
      enabled: localizationMethod !== 'none' || orbitalOrdering !== 'canonical',
      localization_method: localizationMethod,
      localization_scope: 'active_space',
      orbital_ordering: orbitalOrdering,
      orbital_order: orbitalOrdering === 'manual' ? orbitalOrder : [],
    };
  }
  return options;
}

function syncStudyModeControls() {
  const modelStudy = normalizeSystemType(document.getElementById('study-system').value) === 'model_hamiltonian';
  const datasetStudy = isHamiltonianDatasetPlanner();
  const modeSelect = document.getElementById('study-mode');
  const initialScanControl = document.getElementById('adaptive-initial-scan-control');
  const initialScanSelect = document.getElementById('adaptive-initial-scan-strategy');
  if (modelStudy || datasetStudy) {
    modeSelect.value = 'static';
  }
  modeSelect.disabled = modelStudy || datasetStudy;
  const initialScanVisible = !modelStudy && !datasetStudy && modeSelect.value === 'adaptive';
  initialScanControl.classList.toggle('hidden', !initialScanVisible);
  initialScanSelect.disabled = !initialScanVisible;
}

function syncAdaptiveActiveSpaceControls() {
  syncStudyModeControls();
  const solver = document.getElementById('adaptive-active-space-solver').value;
  const solverControl = document.getElementById('adaptive-active-space-solver-control');
  const container = document.getElementById('adaptive-block2-orbital-controls');
  const ordering = document.getElementById('adaptive-orbital-ordering');
  const manualControl = document.getElementById('adaptive-manual-orbital-order-control');
  const manualInput = document.getElementById('adaptive-manual-orbital-order');
  const adaptiveMolecular = currentStudyMode() === 'adaptive'
    && normalizeSystemType(document.getElementById('study-system').value) === 'molecular';
  solverControl.classList.toggle('hidden', !adaptiveMolecular);
  const visible = adaptiveMolecular && solver === 'block2_dmrg';
  container.classList.toggle('hidden', !visible);
  const manualVisible = visible && ordering.value === 'manual';
  manualControl.classList.toggle('hidden', !manualVisible);
  manualInput.disabled = !manualVisible;
}

function planCaseCount(plan) {
  return plan && Array.isArray(plan.cases) ? plan.cases.length : 0;
}

function planMethodList(plan) {
  const methods = new Set();
  if (plan && Array.isArray(plan.cases)) {
    plan.cases.forEach((item) => {
      const request = item && item.request ? item.request : {};
      const method = plannerSolverName(request.solver) || request.method || '';
      if (method) {
        methods.add(String(method).toUpperCase());
      }
    });
  }
  return Array.from(methods).sort();
}

function planSolverList(plan) {
  return planMethodList(plan);
}

function decisionSolverList(decisions) {
  const methods = new Set();
  (Array.isArray(decisions) ? decisions : []).forEach((item) => {
    const method = item && (item.recommended_method || item.recommended_solver);
    if (method) {
      methods.add(String(method).toUpperCase());
    }
  });
  return Array.from(methods).sort();
}

function reviewCorrelationLevel(decision = {}, result = {}, structured = {}) {
  const decisionDiagnostics = decision.diagnostics && typeof decision.diagnostics === 'object' ? decision.diagnostics : {};
  const structuredDiagnostics = structured.correlation_diagnostics && typeof structured.correlation_diagnostics === 'object'
    ? structured.correlation_diagnostics
    : (structured.strong_correlation_diagnostics && typeof structured.strong_correlation_diagnostics === 'object'
      ? structured.strong_correlation_diagnostics
      : {});
  for (const diagnostics of [decisionDiagnostics, structuredDiagnostics]) {
    if (diagnostics.kind === 'strong_correlation_diagnostics'
        && ['weak', 'moderate', 'strong', 'unknown'].includes(diagnostics.physics_level)) {
      return diagnostics.physics_level;
    }
  }
  const candidates = [
    decision.diagnostic_level,
    decisionDiagnostics.molecular_correlation_risk && decisionDiagnostics.molecular_correlation_risk.level,
    decisionDiagnostics.level,
    structuredDiagnostics.molecular_correlation_risk && structuredDiagnostics.molecular_correlation_risk.level,
    structuredDiagnostics.level,
    result.correlation,
    decision.physics_level,
    decision.level,
  ];
  for (const candidate of candidates) {
    const level = String(candidate || '').trim().toLowerCase();
    if (['weak', 'moderate', 'strong'].includes(level)) {
      return level;
    }
  }
  return '';
}

function adaptiveStatusClass(status) {
  return String(status || 'pending')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'pending';
}

function adaptivePhaseCard(title, detail, extra = '', status = '') {
  const extraItems = Array.isArray(extra) ? extra : [extra];
  const extraMarkup = extraItems
    .filter((item) => item !== null && item !== undefined && String(item).trim())
    .map((item) => `<span>${escapeHtml(item)}</span>`)
    .join('');
  const statusText = status ? String(status) : '';
  const statusMarkup = statusText
    ? `<span class="adaptive-status ${adaptiveStatusClass(statusText)}">${escapeHtml(statusText)}</span>`
    : '';
  return `<div class="adaptive-decision"><div class="adaptive-decision-main"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(detail)}</span>${extraMarkup}</div>${statusMarkup}</div>`;
}

function adaptiveInitialScanIssues(payload) {
  const adaptive = payload && (payload.adaptive || payload);
  return adaptive && Array.isArray(adaptive.initial_scan_issues) ? adaptive.initial_scan_issues : [];
}

function adaptivePayloadSystemType(payload) {
  const adaptive = payload && (payload.adaptive || payload);
  const refinedPlan = adaptive && (adaptive.refined_plan || payload.refined_plan);
  const initialScanPlan = adaptive && (adaptive.initial_scan_plan || payload.initial_scan_plan);
  return normalizeSystemType(
    (refinedPlan && refinedPlan.system_type)
      || (initialScanPlan && initialScanPlan.system_type)
      || (payload && payload.system_type)
      || currentSystemType()
  );
}

function adaptiveMethodLabel(systemType) {
  return normalizeSystemType(systemType) === 'molecular' ? 'methods' : 'solvers';
}

function adaptiveDecisionEntries(report) {
  const adaptive = report && report.adaptive ? report.adaptive : {};
  const entries = [];
  (Array.isArray(adaptive.decision_log) ? adaptive.decision_log : []).forEach((item) => {
    if (item) {
      entries.push(item);
    }
  });
  (Array.isArray(adaptive.recovery_decisions) ? adaptive.recovery_decisions : []).forEach((item) => {
    if (item) {
      entries.push(item);
    }
  });
  (Array.isArray(adaptive.path_refinement_decisions) ? adaptive.path_refinement_decisions : []).forEach((item) => {
    if (item) {
      entries.push(item);
    }
  });
  (Array.isArray(adaptive.direct_active_space_review_decisions) ? adaptive.direct_active_space_review_decisions : []).forEach((item) => {
    if (item) {
      entries.push(item);
    }
  });
  return entries;
}

function summarizeAdaptiveCounts(label, values, formatter = (value) => value) {
  const counts = {};
  (Array.isArray(values) ? values : []).forEach((value) => {
    if (value === null || value === undefined || String(value).trim() === '') {
      return;
    }
    const key = String(value).trim();
    counts[key] = (counts[key] || 0) + 1;
  });
  const entries = Object.keys(counts).sort().map((key) => `${formatter(key)}=${counts[key]}`);
  return entries.length ? `${label}: ${entries.join(', ')}` : '';
}

function initialScanIssueMessage(payload) {
  const issues = adaptiveInitialScanIssues(payload);
  if (!issues.length) {
    return '';
  }
  const lines = [
    `Adaptive scan stopped after the initial scan because ${issues.length} planned case${issues.length === 1 ? '' : 's'} did not run successfully. No refined method recommendation was made from blocked initial-scan data.`,
  ];
  issues.forEach((issue) => {
    const variables = issue.variables && typeof issue.variables === 'object'
      ? Object.entries(issue.variables).map(([key, value]) => `${key}=${Array.isArray(value) ? value.join(',') : value}`).join(', ')
      : '';
    const reasons = Array.isArray(issue.reasons) && issue.reasons.length
      ? issue.reasons.slice(0, 3).join(' | ')
      : 'No validation detail was reported.';
    lines.push(`${issue.case_id || issue.label || 'case'}${variables ? ` (${variables})` : ''}: status=${issue.initial_scan_status || 'unknown'}; ${reasons}`);
  });
  return lines.join('\n');
}

function adaptiveNextStepMessage(payload) {
  const adaptive = payload && (payload.adaptive || payload);
  if (!adaptive) {
    return '';
  }
  const systemType = adaptivePayloadSystemType(payload);
  const studyLabel = payload && payload.adaptive ? 'Adaptive scan' : 'Study';
  const methodLabel = adaptiveMethodLabel(systemType);
  const issues = adaptiveInitialScanIssues(payload);
  if (issues.length) {
    return `${initialScanIssueMessage(payload)}\n\nNext: fix the listed input validation issues, rebuild the initial-scan plan, then run the adaptive scan again.`;
  }
  const approvalRows = systemType === 'molecular' ? activeSpaceApprovalRows(payload) : [];
  const requiresActiveSpaceApproval = approvalRows.length > 0;
  const rows = adaptiveRefinedRows(payload);
  const unresolvedRows = rows.filter((row) => row && row.status && row.status !== 'succeeded');
  const recoveryCount = Array.isArray(adaptive.recovery_decisions) ? adaptive.recovery_decisions.length : 0;
  if (systemType === 'molecular' && requiresActiveSpaceApproval) {
    const pathRefinementRows = approvalRows.filter((row) => row.plan_kind === 'path_refinement');
    if (pathRefinementRows.length) {
      return `Adaptive molecular scan found a non-smooth method transition across ${pathRefinementRows.length} local case${pathRefinementRows.length === 1 ? '' : 's'}. Next: review the path-continuity ActiveSpaceAudit, approve or edit the shared CAS, then rerun the affected overlap window.`;
    }
    return `Adaptive molecular scan identified ${approvalRows.length} CASSCF/CASCI active-space candidate${approvalRows.length === 1 ? '' : 's'} awaiting approval. Next: review the ActiveSpaceAudit/CAS columns, approve or edit the active space, then run the approved refined/recovery calculation.`;
  }
  if (unresolvedRows.length) {
    const recoveryByCaseId = adaptiveRecoveryDecisionMap(payload);
    const suggestions = unresolvedRows.slice(0, 3).map((row) => {
      const decision = recoveryByCaseId[row.case_id] || {};
      const suggestion = adaptiveUnresolvedSuggestion(row, decision, reviewActionsFromWorkflow(adaptiveWorkflow(payload)));
      const action = row.recovery_action || decision.recovery_action || '';
      const maxCycle = row.recovery_max_cycle || (decision.recovery_runtime && decision.recovery_runtime.max_cycle);
      const actionLabel = action ? `; recovery=${action}${maxCycle ? `, max_cycle=${maxCycle}` : ''}` : '';
      return `${row.case_id || row.label || 'case'}: status=${row.status || 'unknown'}${actionLabel}; next=${suggestion}`;
    });
    const recoveryText = recoveryCount
      ? ` Automatic recovery was already attempted for ${recoveryCount} case${recoveryCount === 1 ? '' : 's'}.`
      : '';
    return [
      `${studyLabel} completed with ${unresolvedRows.length} unresolved case${unresolvedRows.length === 1 ? '' : 's'}.${recoveryText}`,
      'Next actions:',
      ...suggestions,
      `After applying the case-specific fix, rerun the affected ${methodLabel} before trusting trends or plots.`,
    ].join('\n');
  }
  if (recoveryCount) {
    return `Adaptive scan completed and recovered ${recoveryCount} unconverged refined case${recoveryCount === 1 ? '' : 's'} with fallback ${methodLabel}. The case calculations remain independent. Select Analyze Results when you are ready to compare them and check scan-path continuity.`;
  }
  return `Adaptive scan completed with independent case results. Review the refined table and ${methodLabel} choices; select Analyze Results only when you are ready to compare cases and check scan-path continuity.`;
}

function adaptiveRefinedRows(payload) {
  const adaptive = payload && (payload.adaptive || payload);
  const refinedReport = adaptive && adaptive.refined_report ? adaptive.refined_report : {};
  if (Array.isArray(payload && payload.comparison_table) && payload.comparison_table.length) {
    return payload.comparison_table;
  }
  return Array.isArray(refinedReport.comparison_table) ? refinedReport.comparison_table : [];
}

function adaptiveUnresolvedRows(payload) {
  return adaptiveRefinedRows(payload).filter((row) => row && row.status && row.status !== 'succeeded');
}

function adaptiveRecoveryDecisionMap(report) {
  const entries = adaptiveDecisionEntries(report).slice().reverse();
  const byCaseId = {};
  entries.forEach((entry) => {
    if (entry && entry.case_id && !byCaseId[entry.case_id]) {
      byCaseId[entry.case_id] = entry;
    }
  });
  return byCaseId;
}

function adaptiveUnresolvedSuggestion(row, decision = {}, actions = []) {
  if (!row) {
    return '';
  }
  const proposal = actions.find((item) => item.description
    && (!Array.isArray(item.case_ids) || item.case_ids.includes(row.case_id)));
  if (proposal) {
    return proposal.description;
  }
  const action = row.recovery_action || decision.recovery_action || '';
  const maxCycle = row.recovery_max_cycle || (decision.recovery_runtime && decision.recovery_runtime.max_cycle);
  const method = String(row.method || row.solver || decision.recovery_method || decision.recovery_solver || '').toLowerCase();
  const statusText = String(row.status || '').toLowerCase();
  if (statusText === 'failed') {
    return row.dmrg_recovery_summary || 'Execution stopped with an error. Inspect the task error and provider log before choosing recovery; increasing iterations does not address an execution exception.';
  }
  if (statusText === 'unconverged' && method === 'dmet') {
    return 'Review the DMET energy, density-change, and density-fit residuals. Increase solver.options.max_iterations for more outer iterations; this does not change impurity SCF or FCI settings or guarantee convergence.';
  }
  const nextStep = row.next_step || decision.next_step || '';
  const staleApprovalStep = (
    nextStep
    && statusText === 'unconverged'
    && ['casscf', 'casci'].includes(method)
    && nextStep.toLowerCase().includes('approve')
  );
  if (nextStep && !staleApprovalStep) {
    return nextStep;
  }
  if (statusText === 'unconverged') {
    if (['casscf', 'casci'].includes(method)) {
      return 'The approved CASSCF/CASCI recovery did not finish; expand the active space, review CAS orbitals/electron count, increase CAS max_cycle, or use FCI as a tiny-basis benchmark when affordable.';
    }
    if (action === 'increase_max_cycle') {
      return `The max_cycle recovery${maxCycle ? ` (${maxCycle})` : ''} still did not converge. Inspect the convergence history and scientific diagnostics before changing the method.`;
    }
    return 'The calculation did not meet its convergence criteria. Inspect the reported residuals and solver settings before choosing a retry.';
  }
  if (String(row.status || '').toLowerCase() === 'blocked') {
    return 'The recovery request was blocked; inspect validation details, active-space approval, and method-specific required fields before rerunning.';
  }
  return 'Inspect this case, adjust runtime or method settings, and rerun the affected calculation.';
}

function reviewPlanTableGuidance(gate, report) {
  if (!gate || !report) {
    return 'Plan Review is focused. Inspect case status, method, final energy, diagnostics, and artifacts before rerunning.';
  }
  const lines = [];
  if (gate.kind === 'initial_scan_blocked') {
    const items = Array.isArray(gate.items) ? gate.items : [];
    lines.push(`Initial scan focus: ${items.length || (gate.case_ids || []).length} blocked case${(items.length || (gate.case_ids || []).length) === 1 ? '' : 's'} need input or method fixes before refinement.`);
    items.slice(0, 8).forEach((item) => {
      lines.push(`${item.title || 'case'}: ${item.detail || 'No diagnostic reason was reported.'}`);
    });
    lines.push('Recommended path: fix the listed input fields or initial-scan method settings, rebuild the adaptive initial-scan plan, then rerun.');
    return lines.join('\n');
  }
  if (gate.kind === 'active_space_approval') {
    const rows = activeSpaceApprovalRows(report);
    lines.push(`Plan Review focus: ${rows.length} CASSCF/CASCI active-space candidate${rows.length === 1 ? '' : 's'} need approval.`);
    rows.slice(0, 8).forEach((row) => {
      lines.push(`${row.case_id || row.title}: ${row.detail || 'Review ActiveSpaceAudit'}; option=Approve Active Space, or edit CAS indices/electron count before rerun.`);
    });
    lines.push('Recommended path: approve/edit the ActiveSpaceAudit, then Run Plan to execute the approved refined/recovery calculation without repeating the initial scan.');
    return lines.join('\n');
  }
  if (gate.kind === 'adaptive_unresolved_refined_cases') {
    const rows = filterRowsForCaseIds(adaptiveUnresolvedRows(report), gate.case_ids || []);
    const recoveryByCaseId = adaptiveRecoveryDecisionMap(report);
    const approvalRows = activeSpaceApprovalRows(report);
    const cases = Array.isArray(report.cases) ? report.cases
      : (((report.adaptive || {}).refined_report || {}).cases || []);
    lines.push(`Plan Review focus: ${rows.length} unresolved case${rows.length === 1 ? '' : 's'}.`);
    rows.slice(0, 8).forEach((row) => {
      const decision = recoveryByCaseId[row.case_id] || {};
      const action = row.recovery_action || decision.recovery_action || 'none reported';
      const method = row.method || row.solver || decision.recovery_method || decision.recovery_solver || '';
      const maxCycle = row.recovery_max_cycle || (decision.recovery_runtime && decision.recovery_runtime.max_cycle) || '';
      const task = (cases.find((item) => item.case_id === row.case_id) || {}).task_report || {};
      const errors = [...(task.errors || []), ...(task.validation_errors || [])];
      const reason = errors.map((error) => error.message).filter(Boolean).join(' | ')
        || row.recovery_reason || decision.recovery_reason || 'Inspect the task report for convergence details.';
      const next = adaptiveUnresolvedSuggestion(row, decision, gate.actions || []);
      const options = (gate.actions || [])
        .filter((item) => !['show_case_guidance', 'acknowledge'].includes(item.id)
          && (!Array.isArray(item.case_ids) || item.case_ids.includes(row.case_id)))
        .map((item) => item.label || item.id);
      if (!options.length) {
        options.push('inspect the task error and convergence evidence');
      }
      lines.push(`${row.case_id || row.label || 'case'}: status=${row.status || 'unknown'}${method ? `, method=${String(method).toUpperCase()}` : ''}, recovery=${action}${maxCycle ? `, max_cycle=${maxCycle}` : ''}. Reason: ${reason}. Options: ${options.join('; ')}. Next: ${next}`);
    });
    if (approvalRows.length) {
      lines.push(`Detected ${approvalRows.length} approvable ActiveSpaceAudit candidate${approvalRows.length === 1 ? '' : 's'}; use Approve Active Space, then Run Plan.`);
    }
    lines.push('After applying one option, rerun only the affected refined/recovery plan when possible; do not trust plots until these cases succeed or are deliberately excluded.');
    return lines.join('\n');
  }
  return gate.hint || 'Plan Review is focused. Inspect the table and rerun after applying the suggested fix.';
}
