function adaptiveWorkflow(payload) {
  if (!payload) {
    return null;
  }
  const adaptive = payload.adaptive && typeof payload.adaptive === 'object' ? payload.adaptive : null;
  if (adaptive && adaptive.workflow && typeof adaptive.workflow === 'object') {
    return adaptive.workflow;
  }
  if (payload.workflow && typeof payload.workflow === 'object') {
    return payload.workflow;
  }
  return null;
}

function adaptivePlanByKind(adaptive, planKind) {
  if (!adaptive) {
    return null;
  }
  if (planKind === 'initial') return adaptive.initial_scan_plan || null;
  if (planKind === 'recovery') return adaptive.recovery_plan || null;
  if (planKind === 'path_refinement') return adaptive.path_refinement_plan || null;
  if (planKind === 'entanglement_active_space') return adaptive.entanglement_active_space_plan || null;
  if (planKind === 'direct_casscf_review') return adaptive.direct_active_space_review_plan || null;
  return adaptive.refined_plan || null;
}

function reviewActionsFromWorkflow(workflow) {
  const supported = new Set([
    'approve_cost_estimate',
    'approve_active_space',
    'expand_active_space',
    'cancel_active_space_review',
    'increase_recovery_max_cycle',
    'increase_dmet_iterations',
    'retry_dmet_without_impurity_diis',
    'try_fci_recovery',
    'promote_casscf_recovery',
    'approve_path_restart',
    'skip_path_restart',
    'cancel_path_restart_review',
    'show_case_guidance',
    'acknowledge',
  ]);
  const actions = workflow && Array.isArray(workflow.allowed_actions)
    ? workflow.allowed_actions
    : [];
  const normalized = actions
    .filter((action) => action && supported.has(action.id))
    .map((action) => ({
      id: action.id,
      label: action.label || action.id,
      description: action.description || '',
      secondary: Boolean(action.secondary),
      case_ids: Array.isArray(action.case_ids) ? action.case_ids : null,
    }));
  return normalized;
}

function workflowCaseIds(workflow) {
  return workflow && Array.isArray(workflow.pending_case_ids)
    ? workflow.pending_case_ids.map((value) => String(value)).filter(Boolean)
    : [];
}

function firstCaseId(caseIds) {
  const ids = (Array.isArray(caseIds) ? caseIds : []).map((value) => String(value)).filter(Boolean);
  return ids.length ? ids[0] : '';
}

function workflowCurrentCaseIds(workflow, fallbackCaseIds = []) {
  const fallback = (Array.isArray(fallbackCaseIds) ? fallbackCaseIds : []).map((value) => String(value)).filter(Boolean);
  const fallbackSet = new Set(fallback);
  if (workflow && Array.isArray(workflow.current_case_ids) && workflow.current_case_ids.length) {
    const current = workflow.current_case_ids.map((value) => String(value)).filter((value) => value && fallbackSet.has(value)).slice(0, 1);
    if (current.length) {
      return current;
    }
  }
  if (workflow && workflow.current_case_id && fallbackSet.has(String(workflow.current_case_id))) {
    return [String(workflow.current_case_id)];
  }
  const current = firstCaseId(fallback);
  return current ? [current] : [];
}

function workflowQueueSuffix(workflow) {
  if (!workflow || workflow.review_mode !== 'case_queue') {
    return '';
  }
  const total = Number.parseInt(workflow.queue_total, 10);
  const position = Number.parseInt(workflow.queue_position, 10);
  if (!Number.isFinite(total) || total <= 1) {
    return '';
  }
  return ` (${Number.isFinite(position) && position > 0 ? position : 1} of ${total})`;
}

function filterRowsForCaseIds(rows, caseIds) {
  const ids = new Set((Array.isArray(caseIds) ? caseIds : []).map((value) => String(value)).filter(Boolean));
  if (!ids.size) {
    return Array.isArray(rows) ? rows : [];
  }
  return (Array.isArray(rows) ? rows : []).filter((row) => row && ids.has(String(row.case_id || '')));
}

function reviewGateForStudyReport(report) {
  if (!report || typeof report !== 'object') {
    return null;
  }
  const adaptive = report.adaptive && typeof report.adaptive === 'object'
    ? report.adaptive
    : {};
  const workflow = adaptiveWorkflow(report);
  if (workflow && workflow.stage === 'cost_review_required') {
    const planKind = workflow.active_plan_kind || 'refined';
    const plan = adaptivePlanByKind(adaptive, planKind);
    const gate = planCostReviewGate(plan, workflow.cost_estimate || (plan && plan.cost_estimate), planKind);
    if (gate) {
      gate.actions = reviewActionsFromWorkflow(workflow);
      return gate;
    }
  }
  if (workflow && workflow.stage === 'mps_continuation_review_required') {
    return mpsContinuationApprovalGate({
      approval: adaptive.mps_continuation_approval,
    });
  }
  const pathRestartValidation = adaptive.path_window_restart_validation;
  const pathRestartApproval = adaptive.path_window_restart_approval;
  if (
    pathRestartValidation
    && pathRestartValidation.status === 'approval_required'
    && pathRestartApproval
  ) {
    return pathRestartApprovalGate({ approval: pathRestartApproval });
  }
  const allApprovalRows = activeSpaceApprovalRows(report);
  if (allApprovalRows.length) {
    const workflowApprovalQueue = workflow
      && ['active_space_review_required', 'path_refinement_review_required'].includes(workflow.stage)
      && workflow.review_mode === 'case_queue';
    const workflowPlanKind = workflow && ['active_space_review_required', 'path_refinement_review_required'].includes(workflow.stage)
      ? workflow.active_plan_kind
      : '';
    const selectedPlanKind = workflowPlanKind || ((allApprovalRows.find((row) => row.plan_kind === 'direct_casscf_review')
      || allApprovalRows.find((row) => row.plan_kind === 'entanglement_active_space')
      || allApprovalRows.find((row) => row.plan_kind === 'recovery')
      || allApprovalRows.find((row) => row.plan_kind === 'refined')
      || allApprovalRows.find((row) => row.plan_kind === 'path_refinement')
      || {}).plan_kind || 'refined');
    const approvalRows = allApprovalRows.filter((row) => row.plan_kind === selectedPlanKind);
    const workflowCaseIdsForApproval = workflowApprovalQueue ? workflowCurrentCaseIds(workflow, approvalRows.map((row) => row.case_id)) : [];
    const displayApprovalRows = workflowApprovalQueue
      ? filterRowsForCaseIds(approvalRows, workflowCaseIdsForApproval)
      : approvalRows;
    const directApprovalCount = displayApprovalRows.filter((row) => row.plan_kind === 'direct_casscf_review').length;
    const recoveryApprovalCount = displayApprovalRows.filter((row) => row.plan_kind === 'recovery').length;
    const pathRefinementApprovalCount = displayApprovalRows.filter((row) => row.plan_kind === 'path_refinement').length;
    const entanglementApprovalCount = displayApprovalRows.filter((row) => row.plan_kind === 'entanglement_active_space').length;
    const directCasscfReview = adaptive.mode === 'direct_casscf_review';
    const planKind = selectedPlanKind;
    const approvalPlan = adaptivePlanByKind(adaptive, planKind);
    const costEstimate = workflow && ['active_space_review_required', 'path_refinement_review_required'].includes(workflow.stage)
      ? workflow.cost_estimate
      : (approvalPlan && approvalPlan.cost_estimate);
    return {
      kind: 'active_space_approval',
      severity: 'warning',
      kicker: 'Active Space Review',
      title: pathRefinementApprovalCount
        ? `Review path-continuity ActiveSpaceAudit${workflowQueueSuffix(workflow)}`
        : entanglementApprovalCount
        ? `Review entanglement-driven ActiveSpaceAudit${workflowQueueSuffix(workflow)}`
        : recoveryApprovalCount
        ? `Approve recovery ActiveSpaceAudit${workflowQueueSuffix(workflow)}`
        : directCasscfReview
        ? `Review CASSCF ActiveSpaceAudit${workflowQueueSuffix(workflow)}`
        : `Approve ${approvalRows.length} ActiveSpaceAudit candidate${approvalRows.length === 1 ? '' : 's'}`,
      status: 'approval required',
      message: pathRefinementApprovalCount
        ? 'A sharp change at a method boundary was found in the final energy curve. Recompute the full local overlap window with a shared-size CASSCF active space.'
        : entanglementApprovalCount
        ? 'Block2 entanglement and natural-occupation evidence indicate that the current active space may be saturated. Review the proposed adjacent-orbital expansion before rerunning.'
        : recoveryApprovalCount
        ? 'Automatic recovery promoted an unresolved refined case to CASSCF/CASCI. Review this case-specific ActiveSpaceAudit before running the recovery calculation.'
        : directCasscfReview
        ? 'The requested CASSCF/CASCI calculation needs an explicit review of the proposed active space before it can run.'
        : 'CASSCF/CASCI refined cases require explicit approval of the proposed active space before execution.',
      hint: pathRefinementApprovalCount
        ? 'Check the local ActiveSpaceAudits and the reported transition mismatch. Approve or edit the shared CAS size, then Run Plan recomputes the overlap window and merges it back into the full curve.'
        : entanglementApprovalCount
        ? 'Check the proposed ncas, nelecas, optimized-orbital indices, evidence, and cost. Approval reuses the saved optimized orbitals, starts a fresh MPS, and merges the result into the full study.'
        : 'Use the existing ActiveSpaceAudit approval flow: check CAS size, electron count, orbital indices, and recovery reason; approve or edit, then Run Plan executes the approved refined/recovery calculation without rerunning the initial scan.',
      items: displayApprovalRows,
      plan_kind: planKind,
      cost_estimate: costEstimateRequiresApproval(costEstimate) ? costEstimate : null,
      case_ids: workflowApprovalQueue ? workflowCaseIdsForApproval : approvalRows.map((row) => row.case_id),
      actions: reviewActionsFromWorkflow(workflow),
    };
  }
  if (workflow && workflow.stage === 'initial_scan_blocked') {
    const issueItems = Array.isArray(workflow.items) ? workflow.items : [];
    return {
      kind: 'initial_scan_blocked',
      severity: 'error',
      kicker: 'Review Required',
      title: `Adaptive initial scan blocked${workflowQueueSuffix(workflow)}`,
      status: 'blocked',
      message: workflow.message || 'Initial scan did not produce valid diagnostics for every case.',
      hint: 'Fix the listed input fields or method settings, rebuild the initial-scan plan, then rerun adaptive scan.',
      items: issueItems.slice(0, 8).map((item) => ({
        title: `${item.case_id || 'case'}${item.label ? ` · ${item.label}` : ''}`,
        detail: Array.isArray(item.reasons) && item.reasons.length ? item.reasons.join(' | ') : 'No diagnostic reason was reported.',
        status: item.initial_scan_status || 'blocked',
      })),
      case_ids: workflowCaseIds(workflow),
      actions: reviewActionsFromWorkflow(workflow),
    };
  }
  const unresolvedRows = adaptiveUnresolvedRows(report);
  if (!unresolvedRows.length) {
    return null;
  }
  const workflowUnresolvedQueue = workflow && workflow.stage === 'recovery_review_required' && workflow.review_mode === 'case_queue';
  const workflowCaseIdsForUnresolved = workflowUnresolvedQueue ? workflowCurrentCaseIds(workflow, unresolvedRows.map((row) => row.case_id)) : [];
  const displayUnresolvedRows = workflowUnresolvedQueue
    ? filterRowsForCaseIds(unresolvedRows, workflowCaseIdsForUnresolved)
    : unresolvedRows;
  const systemType = adaptivePayloadSystemType(report);
  const isAdaptiveStudy = Boolean(report.adaptive);
  const methodLabel = adaptiveMethodLabel(systemType);
  const recoveryByCaseId = adaptiveRecoveryDecisionMap(report);
  const itemRows = displayUnresolvedRows.slice(0, 8).map((row) => {
    const decision = recoveryByCaseId[row.case_id] || {};
    const method = row.method || row.solver || decision.recovery_method || decision.recovery_solver || '';
    const variables = Object.entries(row)
      .filter(([key, value]) => !['case_id', 'label', 'status', 'run_dir', 'method', 'solver', 'energy', 'final_energy', 'diagnostics', 'recovery_reason', 'next_step'].includes(key) && value !== null && value !== undefined && String(value).trim() !== '')
      .slice(0, 4)
      .map(([key, value]) => `${key}=${value}`)
      .join(', ');
    const recoveryAction = row.recovery_action || decision.recovery_action || '';
    const recoveryMaxCycle = row.recovery_max_cycle || (decision.recovery_runtime && decision.recovery_runtime.max_cycle);
    const recoveryReason = row.recovery_reason || decision.recovery_reason || '';
    const suggestion = adaptiveUnresolvedSuggestion(row, decision, reviewActionsFromWorkflow(workflow));
    const detailParts = [
      method ? `${methodLabel.slice(0, -1)}=${method}` : '',
      variables,
      recoveryAction ? `recovery=${recoveryAction}${recoveryMaxCycle ? `, max_cycle=${recoveryMaxCycle}` : ''}` : '',
      recoveryReason,
      suggestion ? `next=${suggestion}` : '',
      row.diagnostics ? `diagnostics=${row.diagnostics}` : '',
    ].filter(Boolean);
    return {
      title: `${row.case_id || 'case'}${row.label ? ` · ${row.label}` : ''}`,
      detail: detailParts.join(' | ') || 'No additional case detail was reported.',
      status: row.status,
    };
  });
  if (!workflowUnresolvedQueue && unresolvedRows.length > itemRows.length) {
    itemRows.push({
      title: `${unresolvedRows.length - itemRows.length} additional unresolved case(s)`,
      detail: 'Open the Plan Review table for the full list.',
      status: 'requires review',
    });
  }
  return {
    kind: 'adaptive_unresolved_refined_cases',
    severity: 'error',
    kicker: 'Review Required',
    title: workflowUnresolvedQueue
      ? `${isAdaptiveStudy ? 'Adaptive scan' : 'Study'} unresolved case${workflowQueueSuffix(workflow)}`
      : `${isAdaptiveStudy ? 'Adaptive scan' : 'Study'} has ${unresolvedRows.length} unresolved case${unresolvedRows.length === 1 ? '' : 's'}`,
    status: 'requires review',
    message: isAdaptiveStudy
      ? `These cases did not finish successfully after automatic recovery. Review the recovery action and final ${methodLabel} before trusting trends or plots.`
      : 'These cases failed or did not converge. Review each cause before selecting a recovery action.',
    hint: workflowUnresolvedQueue
      ? 'Use one suggested action for this case, run the affected recovery calculation, then return to the queue for the next unresolved case.'
      : 'Use the suggested next step for each case: expand or approve/edit active space, promote to a strong-correlation recovery method, or fix blocked request fields before rerunning.',
    items: itemRows,
    case_ids: workflowUnresolvedQueue ? workflowCaseIdsForUnresolved : unresolvedRows.map((row) => row.case_id).filter(Boolean),
    actions: reviewActionsFromWorkflow(workflow),
  };
}

function renderAdaptiveState(payload) {
  const panel = document.getElementById('adaptive-panel');
  const target = document.getElementById('adaptive-summary');
  const mode = currentStudyMode();
  const adaptive = payload && (payload.adaptive || payload);
  if (mode !== 'adaptive' && !(adaptive && (adaptive.mode || adaptive.initial_scan_plan))) {
    panel.className = 'adaptive-panel hidden';
    target.className = 'empty';
    target.innerHTML = 'Build or run an adaptive scan to see initial-scan decisions.';
    return;
  }
  panel.className = 'adaptive-panel';
  if (!payload) {
    target.className = 'empty';
    target.innerHTML = 'Build or run an adaptive scan to see initial-scan decisions.';
    return;
  }
  if (adaptive.mode === 'result_path_review') {
    const pathPlan = adaptive.path_refinement_plan || {};
    const pathCases = planCaseCount(pathPlan);
    const pendingApprovals = activeSpaceApprovalRows(payload)
      .filter((row) => row && row.plan_kind === 'path_refinement');
    target.className = 'adaptive-decision-list';
    target.innerHTML = [
      adaptivePhaseCard('Completed scan', 'A result-analysis continuity check found a non-smooth method boundary.', '', pendingApprovals.length ? 'review required' : 'updated'),
      adaptivePhaseCard(
        'Path-continuity refinement',
        pendingApprovals.length
          ? `${pathCases} local CASSCF case${pathCases === 1 ? '' : 's'} await ActiveSpaceAudit approval.`
          : `${pathCases} local CASSCF case${pathCases === 1 ? '' : 's'} were recomputed and merged into the original curve.`,
        pendingApprovals.length
          ? 'Approve or edit the proposed shared CAS, then Run Plan will replace the affected overlap window in the original curve.'
          : 'Review the updated curve and rerun result analysis before trusting the repaired segment.',
        pendingApprovals.length ? 'approval required' : 'complete',
      ),
    ].join('');
    return;
  }
  if (adaptive.mode === 'direct_casscf_review') {
    const reviewPlan = adaptive.direct_active_space_review_plan || {};
    const reviewCases = planCaseCount(reviewPlan);
    const pendingApprovals = activeSpaceApprovalRows(payload)
      .filter((row) => row && row.plan_kind === 'direct_casscf_review');
    target.className = 'adaptive-decision-list';
    target.innerHTML = [
      adaptivePhaseCard(
        'CASSCF request',
        `${reviewCases} case${reviewCases === 1 ? '' : 's'} prepared for multireference treatment.`,
        'The proposed active space comes from the shared ActiveSpaceAudit and uses one CAS definition across the study.',
        pendingApprovals.length ? 'review required' : 'ready',
      ),
      adaptivePhaseCard(
        'Active-space approval',
        pendingApprovals.length
          ? 'Review the CAS size, electron count, orbital indices, and cost estimate.'
          : 'The approved CASSCF/CASCI calculation is ready to run.',
        pendingApprovals.length
          ? 'Approve Active Space to make the calculation executable.'
          : 'Run Plan executes the approved calculation.',
        pendingApprovals.length ? 'approval required' : 'approved',
      ),
    ].join('');
    return;
  }
  const initialScanPlan = adaptive.initial_scan_plan || payload.initial_scan_plan || {};
  const refinedPlan = adaptive.refined_plan || payload.refined_plan || {};
  const refinedDecisions = adaptive.decision_log || payload.decision_log || [];
  const initialScanDecisions = adaptive.initial_scan_decisions || payload.initial_scan_decisions || [];
  const decisions = refinedDecisions.length ? refinedDecisions : initialScanDecisions;
  const initialScanIssues = adaptiveInitialScanIssues(payload);
  const systemType = adaptivePayloadSystemType(payload);
  const methodLabel = adaptiveMethodLabel(systemType);
  const initialScanMethods = planMethodList(initialScanPlan);
  const initialScanMethodLabel = initialScanMethods.length ? initialScanMethods.join(', ') : 'initial';
  const initialScanDetail = systemType === 'molecular'
    ? `${initialScanMethodLabel} MolecularCorrelationRisk + ActiveSpaceAudit`
    : `${initialScanMethodLabel} strong-correlation diagnostics`;
  const initialScanCases = planCaseCount(initialScanPlan);
  const refinedCases = planCaseCount(refinedPlan);
  const refinedMethods = planMethodList(refinedPlan);
  const recommendedMethods = decisionSolverList(decisions);
  if (initialScanIssues.length) {
    target.className = 'adaptive-decision-list';
    target.innerHTML = [
      adaptivePhaseCard('Step 1: Initial scan', `${initialScanIssues.length} planned case${initialScanIssues.length === 1 ? '' : 's'} did not produce valid initial-scan diagnostics.`, initialScanDetail, 'blocked'),
      adaptivePhaseCard('Step 2: Refined calculation', `No refined ${methodLabel.slice(0, -1)} recommendation was made from blocked initial-scan data.`, '', 'not run'),
      adaptivePhaseCard('Initial scan issues', 'Case-level validation details were posted to Planner Conversation.', 'Fix the input issues, rebuild the initial-scan plan, then run adaptive scan again.', 'blocked'),
    ].join('');
    return;
  }
  if (!decisions.length) {
    target.className = 'adaptive-decision-list';
    target.innerHTML = [
      adaptivePhaseCard('Step 1: Initial scan', initialScanCases ? `${initialScanCases} planned case(s); ${initialScanDetail}.` : 'Build the initial-scan plan.', '', initialScanCases ? 'ready' : 'pending'),
      adaptivePhaseCard('Step 2: Refined calculation', 'Pending until initial-scan diagnostics are computed.', `Run the adaptive scan to classify correlation strength and choose refined ${methodLabel}.`, 'pending'),
    ].join('');
    return;
  }
  target.className = 'adaptive-decision-list';
  const recoveryCount = Array.isArray(adaptive.recovery_decisions) ? adaptive.recovery_decisions.length : 0;
  const refinedResultsAvailable = Boolean(adaptive.refined_report || (Array.isArray(payload.comparison_table) && payload.comparison_table.length));
  const refinedRows = adaptiveRefinedRows(payload);
  const recoveryByCaseId = adaptiveRecoveryDecisionMap(payload);
  const finalLevels = refinedRows.map((row) => {
    if (!row || !row.case_id) {
      return '';
    }
    return reviewCorrelationLevel(recoveryByCaseId[row.case_id] || {}, row, {});
  });
  const studyActiveSpace = adaptive.study_active_space_policy && typeof adaptive.study_active_space_policy === 'object'
    ? adaptive.study_active_space_policy
    : null;
  const summaryLines = [
    summarizeAdaptiveCounts('Correlation regions', finalLevels.some((level) => level) ? finalLevels : decisions.map((decision) => decision && decision.level)),
    summarizeAdaptiveCounts(systemType === 'molecular' ? 'Refined methods' : 'Refined solvers', decisions.map((decision) => decision && (decision.recommended_method || decision.recommended_solver)), (value) => String(value).toUpperCase()),
    studyActiveSpace && studyActiveSpace.ncas
      ? `Study active space: ${activeSpaceLabel(studyActiveSpace)}; each case keeps its own orbital mapping`
      : '',
    summarizeAdaptiveCounts('Final status', refinedRows.map((row) => row && row.status)),
  ].filter(Boolean);
  target.innerHTML = [
    adaptivePhaseCard('Step 1: Initial scan', `${initialScanCases || initialScanDecisions.length} planned case(s); ${initialScanDetail}.`, '', 'complete'),
    adaptivePhaseCard('Step 2: Refined calculation', `${refinedCases || refinedDecisions.length || decisions.length} refined case(s); ${methodLabel}=${(refinedMethods.length ? refinedMethods : recommendedMethods).join(', ') || 'selected by diagnostics'}`, recoveryCount ? `${recoveryCount} fallback recovery case(s) applied.` : '', recoveryCount ? 'recovered' : (refinedResultsAvailable ? 'complete' : 'ready')),
    summaryLines.length ? adaptivePhaseCard('Selection summary', summaryLines[0], summaryLines.slice(1), 'summary') : '',
  ].join('');
}

function diagnosticItemValue(diagnostics, name) {
  const items = diagnostics && Array.isArray(diagnostics.diagnostics) ? diagnostics.diagnostics : [];
  const item = items.find((entry) => entry && entry.name === name);
  return item ? item.value : null;
}

function diagnosticScalarValue(diagnostics, name, key) {
  const value = diagnosticItemValue(diagnostics, name);
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value[key];
  }
  return value;
}

function molecularRisk(diagnostics) {
  if (!diagnostics || typeof diagnostics !== 'object') {
    return {};
  }
  return diagnostics.molecular_correlation_risk && typeof diagnostics.molecular_correlation_risk === 'object'
    ? diagnostics.molecular_correlation_risk
    : diagnostics;
}

function molecularRiskComponentValue(diagnostics, name, key = null) {
  const risk = molecularRisk(diagnostics);
  const components = []
    .concat(Array.isArray(risk.physics_components) ? risk.physics_components : [])
    .concat(Array.isArray(risk.solver_stress_components) ? risk.solver_stress_components : []);
  const component = components.find((item) => item && item.name === name);
  if (!component) {
    return '';
  }
  if (!key) {
    return component.value !== undefined ? component.value : component.score;
  }
  const value = component.value;
  return value && typeof value === 'object' && !Array.isArray(value) ? value[key] : '';
}

function activeSpaceLabel(contract) {
  if (!contract || typeof contract !== 'object') {
    return '';
  }
  const ncas = contract.ncas;
  const nelecas = contract.nelecas;
  if (ncas === null || ncas === undefined || nelecas === null || nelecas === undefined) {
    return '';
  }
  const electronLabel = Array.isArray(nelecas) ? nelecas.join(',') : nelecas;
  return `CAS(${electronLabel}, ${ncas})`;
}

function activeSpaceIndexLabel(indices) {
  if (!indices) {
    return '';
  }
  if (Array.isArray(indices)) {
    return indices.join(', ');
  }
  if (typeof indices === 'object') {
    return Object.entries(indices)
      .map(([spin, values]) => `${spin}:${Array.isArray(values) ? values.join(',') : values}`)
      .join('; ');
  }
  return String(indices);
}

function activeSpaceRequestSolverName(request) {
  const solver = request && request.solver;
  if (solver && typeof solver === 'object') {
    return String(solver.name || '').trim().toLowerCase();
  }
  return String(solver || '').trim().toLowerCase();
}

function activeSpaceOrbitalProcessingLabels(request) {
  const processing = request && request.orbital_processing;
  if (!processing || typeof processing !== 'object') {
    return [];
  }
  const labels = [];
  const localization = String(processing.localization_method || 'none').trim().toLowerCase();
  const scope = String(processing.localization_scope || 'analysis').trim().toLowerCase();
  if (localization && localization !== 'none') {
    labels.push(`localization=${localization}${scope === 'active_space' ? ' (active orbitals)' : ''}`);
  }
  const ordering = String(processing.orbital_ordering || 'canonical').trim().toLowerCase();
  if (ordering) {
    labels.push(`ordering=${ordering}`);
  }
  if (ordering === 'manual' && Array.isArray(processing.orbital_order) && processing.orbital_order.length) {
    labels.push(`permutation=${processing.orbital_order.join(',')}`);
  }
  return labels;
}

function activeSpaceApprovalRows(report, requestedPlanKind = '') {
  const adaptive = report && report.adaptive ? report.adaptive : null;
  if (!adaptive) {
    return [];
  }
  const workflow = adaptive.workflow && typeof adaptive.workflow === 'object' ? adaptive.workflow : {};
  const modePlanKind = {
    direct_casscf_review: 'direct_casscf_review',
    entanglement_active_space_review: 'entanglement_active_space',
    result_path_review: 'path_refinement',
  }[String(adaptive.mode || '')] || '';
  const workflowPlanKind = ['active_space_review_required', 'path_refinement_review_required'].includes(workflow.stage)
    ? String(workflow.active_plan_kind || '')
    : '';
  const selectedPlanKind = String(requestedPlanKind || workflowPlanKind || modePlanKind || '');
  let candidatePlans = [
    { plan: adaptive.direct_active_space_review_plan || null, kind: 'direct_casscf_review' },
    { plan: adaptive.entanglement_active_space_plan || null, kind: 'entanglement_active_space' },
    { plan: adaptive.recovery_plan || null, kind: 'recovery' },
    { plan: adaptive.refined_plan || null, kind: 'refined' },
    { plan: adaptive.path_refinement_plan || null, kind: 'path_refinement' },
  ];
  if (selectedPlanKind) {
    candidatePlans = candidatePlans.filter(({ kind }) => kind === selectedPlanKind);
  }
  const decisions = adaptiveDecisionEntries(report);
  const decisionByCaseId = {};
  decisions.forEach((decision) => {
    if (decision && decision.case_id) {
      decisionByCaseId[decision.case_id] = decision;
    }
  });
  const resultByCaseId = {};
  adaptiveRefinedRows(report).forEach((row) => {
    if (row && row.case_id) {
      resultByCaseId[row.case_id] = row;
    }
  });
  const seen = new Set();
  const rows = [];
  candidatePlans.forEach(({ plan, kind }) => {
    (plan && Array.isArray(plan.cases) ? plan.cases : []).forEach((item) => {
      if (!item || seen.has(String(item.case_id))) {
        return;
      }
      const request = item && item.request ? item.request : {};
      const decision = decisionByCaseId[item.case_id] || {};
      const solver = activeSpaceRequestSolverName(request);
      const method = String(request.method || decision.recommended_method || solver || decision.recommended_solver || '').toLowerCase();
      const activeSpace = request.active_space || decision.active_space_contract || {};
      const tags = Array.isArray(decision.tags) ? decision.tags : [];
      const needsCasApproval = ['casci', 'casscf'].includes(method) && activeSpace && activeSpace.enabled !== false && !activeSpace.approved;
      const taggedForApproval = tags.includes('requires_active_space_approval') && activeSpace && !activeSpace.approved;
      if (!needsCasApproval && !taggedForApproval) {
        return;
      }
      seen.add(String(item.case_id));
      const result = resultByCaseId[item.case_id] || {};
      const activeLabel = activeSpaceLabel(activeSpace) || 'CAS candidate';
      const orbitalLabel = activeSpaceIndexLabel(activeSpace.orbital_indices);
      const diagnosticCaseId = decision['source_initial' + '_scan_case_id'];
      const recoveryReason = decision.recovery_reason || decision.reason || '';
      const maxCycle = decision.recovery_runtime && decision.recovery_runtime.max_cycle;
      const detailParts = [
        method ? `method=${method.toUpperCase()}` : '',
        solver ? `solver=${solver === 'block2_dmrg' ? 'block2 DMRG' : solver.toUpperCase()}` : '',
        activeLabel,
        orbitalLabel ? `orbitals=${orbitalLabel}` : '',
        ...activeSpaceOrbitalProcessingLabels(request),
        kind === 'recovery' ? 'recovery candidate' : '',
        kind === 'path_refinement' ? 'path-continuity candidate' : '',
        kind === 'entanglement_active_space' ? 'entanglement expansion candidate' : '',
        kind === 'direct_casscf_review' ? 'direct CASSCF request' : '',
        maxCycle ? `max_cycle=${maxCycle}` : '',
        recoveryReason,
        diagnosticCaseId ? `diagnostic case=${diagnosticCaseId}` : '',
      ].filter(Boolean);
      rows.push({
        case_id: item.case_id,
        plan_kind: kind,
        title: `${item.case_id || 'case'}${item.label ? ` · ${item.label}` : ''}`,
        detail: detailParts.join(' | '),
        status: result.status === 'blocked' ? 'approval required' : (result.status || 'approval required'),
      });
    });
  });
  return rows;
}

function formatEnergyLikeValue(value, unit) {
  if (value === null || value === undefined || String(value).trim() === '') {
    return '';
  }
  if (typeof value === 'number' && Number.isFinite(value) && unit) {
    return `${value} ${unit}`;
  }
  return value;
}

function hasCellValue(value) {
  return value !== null && value !== undefined && String(value).trim() !== '';
}

function firstCellValue(...values) {
  return values.find((value) => hasCellValue(value));
}

function modelFillingFromCase(item) {
  const modelSpec = item && item.model_spec ? item.model_spec : {};
  const sites = Array.isArray(modelSpec.sites) ? modelSpec.sites : [];
  const nelec = Array.isArray(modelSpec.nelec) ? modelSpec.nelec : [];
  if (!sites.length || nelec.length !== 2) {
    return '';
  }
  const totalElectrons = Number(nelec[0]) + Number(nelec[1]);
  return Number.isFinite(totalElectrons) ? totalElectrons / sites.length : '';
}

function diagnosticSummaryText(row) {
  const parts = [];
  if (hasCellValue(row.solver_stress)) {
    parts.push(`solver risk=${row.solver_stress}`);
  }
  if (hasCellValue(row.gap)) {
    parts.push(`${hasCellValue(row.method) ? 'HOMO-LUMO gap' : 'many-body gap'}=${row.gap}`);
  }
  if (hasCellValue(row.mean_field_gap)) {
    parts.push(`MF gap=${row.mean_field_gap}`);
  }
  if (hasCellValue(row.natural_occupation_fractionality)) {
    parts.push(`NO fractionality=${row.natural_occupation_fractionality}`);
  }
  if (hasCellValue(row.max_double_excitation_amplitude)) {
    parts.push(`max |T2|=${row.max_double_excitation_amplitude}`);
  }
  if (hasCellValue(row.active_space)) {
    parts.push(`active space=${row.active_space}`);
  }
  return parts.join('; ');
}
