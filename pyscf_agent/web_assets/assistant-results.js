function renderStatus(status) {
  const value = status || 'unknown';
  const labels = {
    unknown: t('statusNotRun'),
    prepared: t('statusPrepared'),
    needs_input: t('statusNeedsInput'),
    needs_review: t('statusNeedsReview'),
    changed: t('statusChanged'),
    calculating: t('statusCalculating'),
    queued: t('statusQueued'),
    running: t('statusRunning'),
    completed: t('statusCompleted'),
    failed: t('statusFailed'),
    cancelled: t('statusCancelled'),
  };
  const displayValue = labels[value] || value;
  const pill = document.getElementById('status-pill');
  pill.className = `status-pill ${value}`;
  pill.textContent = displayValue;
  document.getElementById('stat-status').textContent = displayValue;
}

function renderStats(payload) {
  document.getElementById('stat-retry').textContent = `${payload.retry_count || 0} / ${payload.max_retries || 0}`;
  document.getElementById('stat-attempts').textContent = String((payload.attempts || []).length);
}

function renderStructuredResults(results) {
  if (!results) {
    setPre('structured-results', '');
    return;
  }
  const compactResults = JSON.parse(JSON.stringify(results));
  delete compactResults.raw_stdout;
  delete compactResults.raw_scf_output;
  delete compactResults.analysis_text;
  if (compactResults.correlation_diagnostics && typeof compactResults.correlation_diagnostics === 'object') {
    delete compactResults.correlation_diagnostics.physics_score;
    delete compactResults.correlation_diagnostics.solver_stress_score;
    const risk = compactResults.correlation_diagnostics.molecular_correlation_risk;
    if (risk && typeof risk === 'object') {
      delete risk.physics_score;
      delete risk.solver_stress_score;
    }
  }
  setPre('structured-results', JSON.stringify(compactResults, null, 2));
}

function formatDiagnosticValue(value) {
  if (value === null || typeof value === 'undefined') {
    return '';
  }
  if (typeof value === 'number') {
    return Number.isFinite(value) ? String(Number(value.toFixed(6))) : String(value);
  }
  if (typeof value === 'object') {
    return JSON.stringify(value);
  }
  return String(value);
}

function prettifyDiagnosticKey(key) {
  const labelMap = {
    gap: langText('Gap', '能隙'),
    gap_over_t: 'Gap / |t|',
    homo: 'HOMO',
    lumo: 'LUMO',
    min_spin_channel_gap: langText('Min spin-channel gap', '最小自旋通道能隙'),
    max_fractionality: langText('Max fractionality', '最大分数占据度'),
    average_fractionality: langText('Average fractionality', '平均分数占据度'),
    fractional_orbital_count: langText('Fractional orbital count', '分数占据轨道数'),
    reference_converged: langText('Reference converged', '参考态收敛'),
    reference_initial_guess: langText('Reference initial guess', '参考态初猜'),
    occupations: langText('Natural occupations', '自然轨道占据数'),
    frontier_occupations: langText('Frontier occupations', '前线自然占据数'),
    max_abs_t2: langText('Max |T2|', '最大 |T2|'),
    rms_t2: 'RMS |T2|',
    large_amplitude_count_0_10: langText('|T2| >= 0.10 count', '|T2| >= 0.10 数量'),
    large_amplitude_count_0_20: langText('|T2| >= 0.20 count', '|T2| >= 0.20 数量'),
    mean_double_occupancy: langText('Mean double occupancy', '平均双占据'),
    reference_uncorrelated_estimate: langText('Uncorrelated estimate', '非关联参考估计'),
    suppression: langText('Suppression', '抑制程度'),
    mean: langText('Mean', '平均值'),
    minimum: langText('Minimum', '最小值'),
    maximum: langText('Maximum', '最大值'),
    mean_abs: langText('Mean absolute value', '平均绝对值'),
  };
  if (Object.prototype.hasOwnProperty.call(labelMap, key)) {
    return labelMap[key];
  }
  return String(key || '')
    .replace(/_/g, ' ')
    .replace(/t/g, '|t|')
    .replace(/u/g, 'U')
    .replace(/v/g, 'V');
}

function appendDiagnosticRows(rows, label, value) {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const entries = Object.entries(value).filter(([, entryValue]) => entryValue !== null && typeof entryValue !== 'undefined');
    if (!entries.length) {
      rows.push([label, 'n/a']);
      return;
    }
    entries.forEach(([key, entryValue]) => {
      rows.push([`${label}: ${prettifyDiagnosticKey(key)}`, formatDiagnosticValue(entryValue) || 'n/a']);
    });
    return;
  }
  rows.push([label, formatDiagnosticValue(value) || 'n/a']);
}

function hasDiagnosticValue(item) {
  if (!item || item.value === null || typeof item.value === 'undefined') {
    return false;
  }
  if (typeof item.value === 'object' && !Array.isArray(item.value)) {
    return Object.values(item.value).some((value) => value !== null && typeof value !== 'undefined');
  }
  return true;
}

function shouldShowDiagnosticTableRow(item) {
  const hiddenWhenUnavailable = new Set(['mean_field_homo_lumo_gap', 'frontier_orbital_degeneracy', 'max_double_excitation_amplitude']);
  return !(hiddenWhenUnavailable.has(item && item.name) && !hasDiagnosticValue(item));
}

function shouldShowDiagnosticAnalysisNote(item) {
  const hiddenAnalysisItems = new Set(['reference_correlation_energy', 'solver_convergence']);
  if (!item || hiddenAnalysisItems.has(item.name)) {
    return false;
  }
  return shouldShowDiagnosticTableRow(item);
}

function sanitizeDiagnosticSummary(text) {
  if (typeof text !== 'string') {
    return '';
  }
  return text
    .replace(/\s*\(score=[^)]*\)\.?/i, '.')
    .replace(/\s*,\s*score=[^,)]+/i, '')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

function renderStrongCorrelationDiagnostics(results) {
  const panel = document.getElementById('strong-correlation-diagnostics-panel');
  const dataAnalysisNotes = [];
  const levelNotes = [];
  const methodNotes = [];
  const diagnostics = results && results.strong_correlation_diagnostics;
  if (!diagnostics) {
    panel.classList.add('hidden');
    renderTableWithNotes('strong-correlation-diagnostics', [], [], [], t('strongCorrelationDiagnosticsEmpty'), 'diagnostic-table');
    return;
  }
  panel.classList.remove('hidden');
  const parameterSummary = diagnostics.parameter_summary || {};
  const recommendation = diagnostics.method_recommendation || {};
  const labelMap = {
    u_over_t: 'U/t',
    mean_field_homo_lumo_gap: langText('Mean-field HOMO-LUMO gap', '平均场 HOMO-LUMO 能隙'),
    frontier_orbital_degeneracy: langText('Frontier orbital degeneracy', '前线轨道简并度'),
    natural_orbital_occupations: langText('Natural occupations', '自然轨道占据数'),
    max_double_excitation_amplitude: langText('Max double-excitation amplitude', '最大双激发振幅'),
    solver_convergence: langText('Solver convergence', '求解器收敛'),
    many_body_gap: langText('Many-body gap', '多体能隙'),
    double_occupancy_suppression: langText('Double occupancy', '双占据'),
    nearest_neighbor_spin_correlation: langText('Spin correlation', '自旋关联'),
    nearest_neighbor_charge_correlation: langText('Charge correlation', '电荷关联'),
  };
  const rows = [];
  if (parameterSummary.u_over_t !== null && typeof parameterSummary.u_over_t !== 'undefined') {
    rows.push(['U/t', Number(parameterSummary.u_over_t).toFixed(3)]);
  }
  if (parameterSummary.filling !== null && typeof parameterSummary.filling !== 'undefined') {
    rows.push([langText('Filling', '填充度'), Number(parameterSummary.filling).toFixed(3)]);
  }
  if (parameterSummary.mean_abs_t !== null && typeof parameterSummary.mean_abs_t !== 'undefined') {
    rows.push([langText('Mean |t|', '平均 |t|'), Number(parameterSummary.mean_abs_t).toFixed(3)]);
  }
  if (parameterSummary.mean_abs_u !== null && typeof parameterSummary.mean_abs_u !== 'undefined') {
    rows.push([langText('Mean |U|', '平均 |U|'), Number(parameterSummary.mean_abs_u).toFixed(3)]);
  }
  if (parameterSummary.mean_abs_v !== null && typeof parameterSummary.mean_abs_v !== 'undefined') {
    rows.push([langText('Mean |V|', '平均 |V|'), Number(parameterSummary.mean_abs_v).toFixed(3)]);
  }
  (diagnostics.diagnostics || []).forEach((item) => {
    if (item.name === 'u_over_t') {
      return;
    }
    if (!shouldShowDiagnosticTableRow(item)) {
      return;
    }
    const label = labelMap[item.name] || item.name || 'diagnostic';
    const categoryLabel = {
      physics: langText('Correlation evidence', '关联证据'),
      solver_stress: langText('Solver reliability', '求解器可靠性'),
      context: langText('State and parameter context', '态与参数特征'),
    }[item.category];
    appendDiagnosticRows(rows, categoryLabel ? `${categoryLabel} · ${label}` : label, item.value);
  });
  if (diagnostics.summary) {
    const summaryText = sanitizeDiagnosticSummary(diagnostics.summary);
    if (summaryText) {
      levelNotes.push(summaryText);
    }
  }
  levelNotes.push(`${langText('Correlation evidence', '电子关联证据')}: ${diagnostics.physics_level || 'unknown'} | ${langText('Solver stress', '求解器风险')}: ${diagnostics.solver_stress_level || 'unknown'}`);
  levelNotes.push(`${langText('Overall warning', '总体警示')}: ${diagnostics.level || 'unknown'} | confidence: ${diagnostics.confidence || 'n/a'}`);
  if (recommendation.preferred && recommendation.preferred.length) {
    methodNotes.push(`${langText('Preferred methods', '推荐方法')}: ${recommendation.preferred.join(', ')}`);
  }
  if (recommendation.usable_for_trends && recommendation.usable_for_trends.length) {
    methodNotes.push(`${langText('Trend methods', '趋势方法')}: ${recommendation.usable_for_trends.join(', ')}`);
  }
  if (recommendation.risky && recommendation.risky.length) {
    methodNotes.push(`${langText('Risky methods', '风险较高的方法')}: ${recommendation.risky.join(', ')}`);
  }
  if (Array.isArray(diagnostics.limitations)) {
    diagnostics.limitations.forEach((item) => {
      if (typeof item === 'string' && item.trim()) {
        levelNotes.push(item);
      }
    });
  }
  if (Array.isArray(diagnostics.diagnostics)) {
    diagnostics.diagnostics.forEach((item) => {
      if (shouldShowDiagnosticAnalysisNote(item) && typeof item.interpretation === 'string' && item.interpretation.trim()) {
        const label = labelMap[item.name] || item.name || 'diagnostic';
        dataAnalysisNotes.push(`${label}: ${item.interpretation}`);
      }
    });
  }
  renderTableWithSections('strong-correlation-diagnostics', ['Item', 'Value'], rows, [
    { title: t('diagnosticDataAnalysisTitle'), items: dataAnalysisNotes },
    { title: t('diagnosticLevelTitle'), items: levelNotes },
    { title: t('diagnosticMethodTitle'), items: methodNotes },
  ], t('strongCorrelationDiagnosticsEmpty'), 'diagnostic-table');
}

function renderActiveSpaceAudit(results) {
  const panel = document.getElementById('active-space-audit-panel');
  const activeSpace = results && results.active_space;
  const audit = activeSpace && activeSpace.audit;
  if (!audit) {
    panel.classList.add('hidden');
    renderTableWithNotes('active-space-audit', [], [], [], t('activeSpaceAuditEmpty'), 'compact');
    return;
  }
  panel.classList.remove('hidden');
  const consistency = audit.ncas_nelecas_consistency || {};
  const approval = audit.manual_approval || {};
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
  const notes = [];
  if (Array.isArray(consistency.messages)) {
    consistency.messages.forEach((item) => {
      if (typeof item === 'string' && item.trim()) {
        notes.push(item);
      }
    });
  }
  if (recommendation.recommended_next_step) {
    notes.push(`${langText('Next step', '下一步')}: ${recommendation.recommended_next_step}`);
  }
  if (decision.reason) {
    notes.push(decision.reason);
  }
  if (decision.expansion_reason) {
    notes.push(decision.expansion_reason);
  }
  const alternatives = sortedActiveSpaceCandidates(candidates)
    .filter((item) => item
      && item.method !== selectedCandidate.method
      && item.method !== expansionCandidate.method
      && ['available', 'mapping_ambiguous'].includes(item.status))
    .map(activeSpaceCandidateLabel);
  if (alternatives.length) {
    notes.push(`Alternatives retained in ActiveSpaceAudit: ${alternatives.join('; ')}`);
  }
  const rows = [
    ['CAS', `(${consistency.final_nelecas ?? activeSpace.nelecas ?? 'n/a'}, ${consistency.final_ncas ?? activeSpace.ncas ?? 'n/a'})`],
    [langText('Orbital index basis', '轨道编号基准'), audit.orbital_index_basis || 'canonical_mo'],
    [langText('Selection method', '选择方法'), audit.selection_method || activeSpace.selection_method || 'n/a'],
    ['Recommended candidate', selectedCandidate.method || audit.selected_candidate_method || 'n/a'],
    ['Next expansion candidate', expansionCandidate.method ? activeSpaceCandidateLabel(expansionCandidate) : null],
    ['Candidate role', selectedCandidate.role || 'n/a'],
    ['Evidence coverage', Number.isFinite(Number(evidenceCoverage.coverage)) ? `${(Number(evidenceCoverage.coverage) * 100).toFixed(1)}%` : 'n/a'],
    ['Selection confidence', decision.confidence || 'n/a'],
    [langText('Approval', '人工确认'), approval.status || (activeSpace.approved ? 'approved' : 'requires_user_review')],
    [langText('Consistency', '一致性'), consistency.consistent ? langText('passed', '通过') : langText('needs review', '需检查')],
  ];
  (audit.selected_orbitals || []).forEach((orbital) => {
    const natural = Array.isArray(orbital.natural_orbital_occupations)
      ? orbital.natural_orbital_occupations
          .filter((item) => item && item.occupation !== null && typeof item.occupation !== 'undefined')
          .map((item) => `NO ${item.natural_orbital_index}:${Number(item.occupation).toFixed(4)}`)
          .join(', ')
      : '';
    const reason = Array.isArray(orbital.selection_reasons) ? orbital.selection_reasons.join('; ') : '';
    const parts = [
      orbital.energy !== null && typeof orbital.energy !== 'undefined' ? `e=${Number(orbital.energy).toFixed(6)}` : '',
      orbital.occupation !== null && typeof orbital.occupation !== 'undefined' ? `occ=${Number(orbital.occupation).toFixed(4)}` : '',
      natural ? `NOON=${natural}` : '',
      orbital.t2_importance !== null && typeof orbital.t2_importance !== 'undefined' ? `max|t2|=${Number(orbital.t2_importance).toFixed(4)}` : '',
      reason,
    ].filter(Boolean);
    rows.push([`${langText('Orbital', '轨道')} ${orbital.index}`, parts.join(' | ')]);
  });
  renderTableWithNotes('active-space-audit', [langText('Item', '项目'), langText('Value', '数值')], rows, notes, t('activeSpaceAuditEmpty'), 'compact');
}

function block2DmrgResult(results) {
  if (!results || typeof results !== 'object') {
    return null;
  }
  if (results.dmrg_result && typeof results.dmrg_result === 'object') {
    return results.dmrg_result;
  }
  const casResult = results.cas_result;
  return casResult && casResult.dmrg_result && typeof casResult.dmrg_result === 'object'
    ? casResult.dmrg_result
    : null;
}

function compactOrbitalPermutation(values) {
  if (!Array.isArray(values) || !values.length) {
    return '';
  }
  const displayed = values.slice(0, 16).join(', ');
  return values.length > 16 ? `${displayed}, ... (${values.length})` : displayed;
}

function isIdentityPermutation(values) {
  return Array.isArray(values) && values.length > 0 && values.every((value, index) => Number(value) === index);
}

function block2OrbitalBasisLabel(value) {
  const labels = {
    canonical: 'Canonical orbitals',
    casscf_optimized_active_space: 'CASSCF-optimized active orbitals',
    localized_active_space: 'Localized active orbitals',
  };
  return labels[value] || String(value || '').replace(/_/g, ' ');
}

function block2AvailabilityLabel(value) {
  const labels = {
    available: 'available from block2 NPDM',
    available_fallback: 'available from spin-free 1/2-RDM fallback',
    unavailable: 'unavailable',
    not_computed_order_limit: 'not computed (requires NPDM above the 2-RDM limit)',
  };
  return labels[value] || value;
}

function formatBlock2Energy(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return 'n/a';
  }
  if (numeric !== 0 && Math.abs(numeric) < 1e-5) {
    return numeric.toExponential(3);
  }
  return numeric.toFixed(6);
}

function setBlock2SectionVisibility(sectionId, visible) {
  const section = document.getElementById(sectionId);
  section.classList.toggle('hidden', !visible);
}

function renderBlock2Empty(element, message) {
  const empty = document.createElement('p');
  empty.className = 'block2-empty';
  empty.textContent = message;
  element.appendChild(empty);
}

function renderBlock2Bars(elementId, values, labelPrefix, emptyMessage = 'Not requested') {
  const element = document.getElementById(elementId);
  element.innerHTML = '';
  if (!Array.isArray(values) || !values.length) {
    renderBlock2Empty(element, emptyMessage);
    return;
  }
  const finiteValues = values.map((value) => Number(value)).filter((value) => Number.isFinite(value));
  const maximum = Math.max(...finiteValues.map((value) => Math.abs(value)), Number.EPSILON);
  values.forEach((rawValue, index) => {
    const value = Number(rawValue);
    if (!Number.isFinite(value)) {
      return;
    }
    const row = document.createElement('div');
    row.className = 'block2-bar-row';
    const label = document.createElement('span');
    label.textContent = `${labelPrefix}${index}`;
    const track = document.createElement('span');
    track.className = 'block2-bar-track';
    const fill = document.createElement('span');
    fill.className = 'block2-bar-fill';
    fill.style.width = `${Math.max(1, Math.min(100, 100 * Math.abs(value) / maximum))}%`;
    track.appendChild(fill);
    const formatted = document.createElement('span');
    formatted.textContent = Number(value.toPrecision(4)).toString();
    row.append(label, track, formatted);
    element.appendChild(row);
  });
}

function renderBlock2MutualInformation(entanglement) {
  const element = document.getElementById('block2-mutual-information');
  element.innerHTML = '';
  const matrix = entanglement.orbital_mutual_information;
  if (!Array.isArray(matrix) || !matrix.length || !matrix.every((row) => Array.isArray(row))) {
    if (entanglement.orbital_mutual_information_artifact) {
      renderBlock2Empty(element, 'Stored in the block2 array artifact');
    } else if ((entanglement.availability || {}).mutual_information === 'not_computed_order_limit') {
      renderBlock2Empty(element, 'Not computed because orbital mutual information requires NPDM terms above the 2-RDM limit');
    } else if ((entanglement.availability || {}).mutual_information === 'unavailable') {
      renderBlock2Empty(element, 'Unavailable because the block2 NPDM interface failed; see Structured Results for details');
    } else {
      renderBlock2Empty(element, 'Not requested');
    }
    return;
  }
  const values = matrix.flat().map((value) => Number(value)).filter((value) => Number.isFinite(value));
  const maximum = Math.max(...values.map((value) => Math.abs(value)), Number.EPSILON);
  element.style.gridTemplateColumns = `repeat(${matrix[0].length}, minmax(0, 1fr))`;
  matrix.forEach((row, rowIndex) => {
    row.forEach((rawValue, columnIndex) => {
      const value = Number(rawValue);
      const fraction = Number.isFinite(value) ? Math.min(1, Math.abs(value) / maximum) : 0;
      const cell = document.createElement('span');
      cell.className = 'block2-matrix-cell';
      cell.style.background = `rgb(${Math.round(241 - 226 * fraction)}, ${Math.round(245 - 127 * fraction)}, ${Math.round(249 - 139 * fraction)})`;
      cell.title = `I(${rowIndex}, ${columnIndex}) = ${Number.isFinite(value) ? Number(value.toPrecision(5)) : 'n/a'}`;
      element.appendChild(cell);
    });
  });
}

function renderBlock2Analysis(results) {
  const panel = document.getElementById('block2-analysis-panel');
  const dmrg = block2DmrgResult(results);
  if (!dmrg) {
    panel.classList.add('hidden');
    return;
  }
  panel.classList.remove('hidden');
  const convergence = dmrg.convergence || {};
  const symmetry = dmrg.symmetry_analysis || {};
  const restart = dmrg.restart || {};
  const entanglement = dmrg.entanglement_diagnostics || {};
  const adaptiveSchedule = dmrg.adaptive_schedule || {};
  const bondPlan = dmrg.bond_dimension_plan || {};
  const errorEstimate = dmrg.energy_error_estimate || {};
  const runtimeResources = dmrg.threading || {};
  const casResult = results && results.cas_result && typeof results.cas_result === 'object'
    ? results.cas_result
    : {};
  const provenance = casResult.orbital_provenance && typeof casResult.orbital_provenance === 'object'
    ? casResult.orbital_provenance
    : {};
  const ordering = dmrg.orbital_ordering && typeof dmrg.orbital_ordering === 'object'
    ? dmrg.orbital_ordering
    : {};
  const casscf = dmrg.casscf && typeof dmrg.casscf === 'object'
    ? dmrg.casscf
    : {};
  const entanglementConfiguration = dmrg.configuration || {};
  const entanglementAvailability = entanglement.availability || {};
  const localizationApplied = provenance.localization_applied === true;
  const permutation = ordering.permutation;
  const requestedRoots = Number(
    casscf.nroots
    || dmrg.requested_root_count
    || entanglementConfiguration.nroots
    || (Array.isArray(dmrg.state_energies) ? dmrg.state_energies.length : 1),
  );
  const isStateAveraged = requestedRoots > 1;
  const rows = [
    ['Converged', dmrg.converged ? 'yes' : 'no'],
    ['Orbital optimization', casscf.orbital_converged === true
      ? 'converged'
      : (casscf.orbital_converged === false ? 'not converged' : null)],
    ['CASSCF macro iterations', casscf.macro_iterations],
    ['DMRG solver calls', casscf.active_space_solver_calls],
    ['State optimization', casscf.orbital_optimization_mode],
    ['Number of roots', casscf.nroots],
    ['State-average weights', isStateAveraged && Array.isArray(dmrg.state_average_weights) ? dmrg.state_average_weights.join(', ') : null],
    ['State-average energy', isStateAveraged ? dmrg.state_average_energy : null],
    ['Reported diagnostics', dmrg.reported_diagnostics_scope === 'ground_state'
      ? 'Ground state (root 0)'
      : dmrg.reported_diagnostics_scope],
    ['Execution threads', runtimeResources.effective_n_threads],
    ['block2 stack memory', Number.isFinite(Number(runtimeResources.effective_stack_memory_bytes))
      ? `${Number((Number(runtimeResources.effective_stack_memory_bytes) / (1024 ** 3)).toPrecision(4))} GiB (${runtimeResources.stack_memory_source || 'default'})`
      : null],
    ['Sweeps', convergence.sweeps_completed],
    ['Final discarded weight', convergence.final_discarded_weight],
    ['Final energy change', convergence.final_energy_change],
    ['Adaptive stages', adaptiveSchedule.adaptive_stages_used],
    ['Final bond dimension', adaptiveSchedule.final_bond_dimension],
    ['Requested bond dimensions', Array.isArray(bondPlan.requested_bond_dimensions) ? bondPlan.requested_bond_dimensions.join(', ') : null],
    ['Planned bond-dimension cap', bondPlan.exact_bond_dimension_cap],
    ['Effective bond dimensions', Array.isArray(bondPlan.effective_bond_dimensions) ? bondPlan.effective_bond_dimensions.join(', ') : null],
    ['Bond-dimension cap applied', bondPlan.cap_applied === true
      ? 'yes'
      : (bondPlan.cap_applied === false ? 'no' : null)],
    ['Adaptive budget', adaptiveSchedule.budget_exhausted === true
      ? 'exhausted'
      : (adaptiveSchedule.budget_exhausted === false ? 'available' : null)],
    ['Estimated energy error', errorEstimate.estimated_absolute_error],
    ['Error estimate method', errorEstimate.method],
    ['Extrapolated energy', errorEstimate.extrapolated_energy],
    ['Symmetry', symmetry.symmetry_backend],
    ['<S^2>', symmetry.spin_square],
    ['Inferred total spin', symmetry.inferred_total_spin],
    ['Entanglement source', entanglement.backend === 'spin_free_rdm_fallback'
      ? 'spin-free 1/2-RDM fallback'
      : (entanglement.backend === 'block2_npdm_sz_from_su2'
        ? 'block2 NPDM (SU2 wavefunction analyzed in fixed-Sz form)'
        : (entanglement.backend === 'block2_npdm' ? 'block2 NPDM' : null))],
    ['Single-orbital entropy', block2AvailabilityLabel(entanglementAvailability.single_orbital_entropy)],
    ['Orbital mutual information', block2AvailabilityLabel(entanglementAvailability.mutual_information)],
    ['MPS restart', restart.requested ? (restart.applied ? 'applied' : 'not applied') : 'not requested'],
    ['Source case', restart.source_case_id],
    ['Source checkpoint', restart.source_scratch_directory],
    ['Initial active orbitals', provenance.input_basis ? block2OrbitalBasisLabel(provenance.input_basis) : null],
    ['Final active orbitals', provenance.output_basis ? block2OrbitalBasisLabel(provenance.output_basis) : null],
    ['Localization', localizationApplied && provenance.localization_method
      ? `${provenance.localization_method} (${provenance.localization_scope || 'analysis'}; ${provenance.localization_applied ? 'applied' : 'not applied'})`
      : null],
    ['Localized active columns', Array.isArray(provenance.active_column_range) ? provenance.active_column_range.join(' - ') : null],
    ['Orthonormality error', provenance.orthonormality_max_abs_error],
    ['DMRG orbital ordering', ordering.requested_method || provenance.orbital_ordering],
    ['Executed permutation', isIdentityPermutation(permutation) ? null : compactOrbitalPermutation(permutation)],
  ].filter((row) => row[1] !== null && typeof row[1] !== 'undefined');
  renderTableWithNotes(
    'block2-analysis-summary',
    ['Item', 'Value'],
    rows,
    [],
    'No block2 summary is available',
    'compact'
  );
  const stateEnergies = Array.isArray(dmrg.state_energies) ? dmrg.state_energies : [];
  const excitationEnergies = Array.isArray(dmrg.excitation_energies) ? dmrg.excitation_energies : [];
  const showStateEnergies = requestedRoots > 1 || stateEnergies.length > 1;
  setBlock2SectionVisibility('block2-state-section', showStateEnergies);
  if (showStateEnergies) {
    renderTableWithNotes(
      'block2-state-energies',
      ['Root', 'Total energy (a.u.)', 'Excitation energy (a.u.)'],
      stateEnergies.map((energy, index) => [
        index === 0 ? '0 (ground)' : String(index),
        formatBlock2Energy(energy),
        formatBlock2Energy(excitationEnergies[index]),
      ]),
      [],
      'No targeted state energies are available',
      'block2-state-table'
    );
  } else {
    document.getElementById('block2-state-energies').innerHTML = '';
  }
  const entropyValues = entanglement.single_orbital_entropy;
  const bipartiteValues = entanglement.bipartite_entanglement;
  const mutualInformation = entanglement.orbital_mutual_information;
  const hasEntropy = Array.isArray(entropyValues) && entropyValues.length > 0;
  const hasMutualInformation = (
    Array.isArray(mutualInformation) && mutualInformation.length > 0
  ) || Boolean(entanglement.orbital_mutual_information_artifact);
  const hasBipartite = Array.isArray(bipartiteValues) && bipartiteValues.length > 0;
  const entropyRequested = Boolean(entanglementConfiguration.compute_entanglement)
    || Boolean(entanglementConfiguration.compute_mutual_information);
  const mutualInformationRequested = Boolean(entanglementConfiguration.compute_mutual_information);
  const bipartiteRequested = Boolean(entanglementConfiguration.compute_bipartite_entanglement);
  const showEntropy = hasEntropy;
  const showMutualInformation = hasMutualInformation;
  const showBipartite = hasBipartite;
  setBlock2SectionVisibility('block2-entropy-section', showEntropy);
  setBlock2SectionVisibility('block2-mutual-section', showMutualInformation);
  setBlock2SectionVisibility('block2-bipartite-section', showBipartite);
  const entanglementGrid = document.getElementById('block2-entanglement-grid');
  const entanglementNote = document.getElementById('block2-entanglement-note');
  const hasEntanglementSection = showEntropy || showMutualInformation || showBipartite;
  const visibleEntanglementSections = [showEntropy, showMutualInformation, showBipartite]
    .filter(Boolean).length;
  entanglementGrid.classList.toggle('hidden', !hasEntanglementSection);
  entanglementGrid.classList.toggle('single-column', visibleEntanglementSections === 1);
  const unavailableDiagnostics = [];
  if (entropyRequested && !hasEntropy) {
    unavailableDiagnostics.push('single-orbital entropy');
  }
  if (mutualInformationRequested && !hasMutualInformation) {
    unavailableDiagnostics.push('orbital mutual information');
  }
  if (bipartiteRequested && !hasBipartite) {
    unavailableDiagnostics.push('bipartite entanglement');
  }
  const anyEntanglementRequested = entropyRequested
    || mutualInformationRequested
    || bipartiteRequested;
  entanglementNote.textContent = unavailableDiagnostics.length
    ? `Unavailable requested diagnostics: ${unavailableDiagnostics.join(', ')}. See Structured Results for provider details.`
    : (!hasEntanglementSection && !anyEntanglementRequested
      ? 'Entanglement diagnostics were not requested for this run.'
      : '');
  entanglementNote.classList.toggle('hidden', !entanglementNote.textContent);
  const entropyEmptyMessage = entanglementAvailability.single_orbital_entropy === 'unavailable'
    ? 'Unavailable; see Structured Results for the block2 and fallback errors'
    : 'Requested but unavailable; see Structured Results for details';
  const bipartiteEmptyMessage = entanglementConfiguration.compute_bipartite_entanglement
    ? 'Requested but unavailable; see Structured Results for details'
    : '';
  renderBlock2Bars('block2-orbital-entropy', entropyValues, 's', entropyEmptyMessage);
  renderBlock2Bars('block2-bipartite-entanglement', bipartiteValues, 'b', bipartiteEmptyMessage);
  renderBlock2MutualInformation(entanglement);
}

function renderTargetedStateEnergies(results) {
  const panel = document.getElementById('state-energies-panel');
  const element = document.getElementById('state-energies');
  const stateEnergies = results && Array.isArray(results.state_energies)
    ? results.state_energies
    : [];
  const requestedOutputs = results && Array.isArray(results.requested_outputs)
    ? results.requested_outputs
    : [];
  const requestedRootCount = Number(results && results.requested_root_count);
  const shouldShow = !block2DmrgResult(results)
    && stateEnergies.length > 0
    && (requestedRootCount > 1 || requestedOutputs.includes('excited_states'));
  panel.classList.toggle('hidden', !shouldShow);
  if (!shouldShow) {
    element.innerHTML = '';
    return;
  }
  const excitationEnergies = Array.isArray(results.excitation_energies)
    ? results.excitation_energies
    : [];
  const unit = results.energy_unit || 'a.u.';
  const diagnostics = results.correlation_diagnostics || {};
  const manifold = diagnostics.low_energy_manifold || {};
  const notes = [];
  const classificationLabels = {
    numerically_degenerate: 'The computed ground-state manifold is numerically degenerate.',
    near_degenerate: 'The computed low-energy roots are near-degenerate.',
    isolated_within_computed_sector: 'The lowest root is isolated within the computed sector.',
  };
  if (classificationLabels[manifold.classification]) {
    notes.push(classificationLabels[manifold.classification]);
  }
  if (manifold.spectrum_scope) {
    notes.push(`Scope: ${String(manifold.spectrum_scope).replace(/_/g, ' ')}.`);
  }
  if (manifold.manifold_may_extend_beyond_computed_roots) {
    notes.push('The low-energy manifold reaches the requested root limit; increase nroots before concluding its size.');
  }
  renderTableWithNotes(
    'state-energies',
    ['Root', `Total energy (${unit})`, `Excitation energy (${unit})`],
    stateEnergies.map((energy, index) => [
      index === 0 ? '0 (ground)' : String(index),
      Number(energy).toPrecision(10),
      excitationEnergies[index] === null || typeof excitationEnergies[index] === 'undefined'
        ? 'n/a'
        : Number(excitationEnergies[index]).toPrecision(8),
    ]),
    notes,
    'No targeted state energies are available',
    'compact'
  );
}

function activeSpaceAuditContract(activeSpace) {
  const audit = activeSpace && activeSpace.audit;
  const consistency = audit && audit.ncas_nelecas_consistency ? audit.ncas_nelecas_consistency : {};
  const selectedOrbitals = audit && Array.isArray(audit.selected_orbitals) ? audit.selected_orbitals : [];
  const fallbackNcas = selectedOrbitals.length || null;
  return {
    ncas: consistency.final_ncas ?? activeSpace.ncas ?? fallbackNcas,
    nelecas: consistency.final_nelecas ?? activeSpace.nelecas ?? consistency.estimated_nelecas ?? null,
    selection_method: activeSpace.selection_method === 'avas' && activeSpace.initial_mo_coeff ? 'avas' : 'manual',
    orbital_indices: (Array.isArray(activeSpace.orbital_indices) && activeSpace.orbital_indices.length) || isSpinResolvedOrbitalIndices(activeSpace.orbital_indices)
      ? activeSpace.orbital_indices
      : selectedOrbitals.map((item) => item.index).filter((item) => item !== null && typeof item !== 'undefined'),
    avas_targets: Array.isArray(activeSpace.avas_targets) ? activeSpace.avas_targets : [],
    avas_threshold: activeSpace.avas_parameters && activeSpace.avas_parameters.threshold ? activeSpace.avas_parameters.threshold : 0.2,
    initial_mo_coeff: activeSpace.initial_mo_coeff || null,
    target_method: activeSpace.target_method || (audit.method_recommendation && audit.method_recommendation.target_method) || null,
    target_solver: activeSpace.target_solver || (audit.method_recommendation && audit.method_recommendation.target_solver) || null,
    target_solver_options: activeSpace.target_solver_options
      || (audit.method_recommendation && audit.method_recommendation.target_solver_options)
      || {},
  };
}

function writeActiveSpaceContractToForm(contract, approved) {
  document.getElementById('active-space-enabled').checked = true;
  setSelectValue('active-space-method', contract.selection_method || 'manual');
  document.getElementById('ncas').value = contract.ncas !== null && typeof contract.ncas !== 'undefined' ? String(contract.ncas) : '';
  document.getElementById('nelecas').value = Array.isArray(contract.nelecas) ? contract.nelecas.join(',') : (contract.nelecas !== null && typeof contract.nelecas !== 'undefined' ? String(contract.nelecas) : '');
  document.getElementById('active-orbitals').value = activeOrbitalIndicesToText(contract.orbital_indices);
  document.getElementById('avas-targets').value = Array.isArray(contract.avas_targets) ? contract.avas_targets.join(', ') : '';
  document.getElementById('avas-threshold').value = contract.avas_threshold ? String(contract.avas_threshold) : '0.2';
  document.getElementById('active-space-approved').checked = Boolean(approved);
  document.getElementById('active-space-section').open = true;
  syncActiveSpaceMethodControls();
  updateActiveSpaceSummary();
  invalidatePendingExecution();
  snapshotCurrentTaskSession();
}

function renderDefaultsAndRetries(payload) {
  const items = [];
  (payload.applied_defaults || []).forEach((item) => {
    items.push(`${item.field} = ${JSON.stringify(item.value)} (${item.reason})`);
  });
  items.push(`retry_count = ${payload.retry_count || 0} / ${payload.max_retries || 0}`);
  renderList('defaults-and-retries', items, t('defaultsAndRetriesEmpty'));
}

function renderDiagnostics(payload) {
  const items = [];
  (payload.validation_errors || []).forEach((message) => {
    items.push(`${langText('Validation error', '校验错误')}: ${message}`);
  });
  (payload.clarification_questions || []).forEach((question) => {
    items.push(`${langText('Needs input', '待补充')}: ${question}`);
  });
  (payload.errors || []).forEach((error) => {
    const code = error.code ? `[${error.code}] ` : '';
    const stage = error.stage ? ` (${error.stage})` : '';
    items.push(`${langText('Structured error', '结构化错误')}${stage}: ${code}${error.message || ''}`);
  });
  renderList('diagnostics', items, t('diagnosticsEmpty'));
}

function renderAttempts(payload) {
  const items = (payload.attempts || []).map((attempt) => {
    const parts = [
      `#${attempt.index || '?'}`,
      `status=${attempt.status || 'unknown'}`,
      `stage=${attempt.stage || 'unknown'}`,
      `retry=${attempt.retry_count || 0}`,
    ];
    if (attempt.timestamp) {
      parts.push(`time=${attempt.timestamp}`);
    }
    if (attempt.errors && attempt.errors.length) {
      parts.push(`errors=${attempt.errors.map((item) => item.code || item.message || 'unknown').join(', ')}`);
    }
    return parts.join(' | ');
  });
  renderList('attempts', items, t('attemptsEmpty'));
}

function renderRawOutputs(payload) {
  const rawStdout = payload.raw_stdout || '';
  const rawScfOutput = payload.raw_scf_output || '';
  const analysisText = payload.analysis_text || '';
  const rawStdoutIsDuplicate = rawStdout && rawStdout === rawScfOutput && !analysisText;

  setPre('raw-scf-output', rawScfOutput || t('noPyscfOut'));
  setPre('raw-stdout', rawStdoutIsDuplicate ? t('sameAsPyscfOut') : (rawStdout || t('noAgentOut')));
  const blocks = [];
  if (payload.raw_stderr) {
    blocks.push(`[raw_stderr]\n${payload.raw_stderr}`);
  }
  if (analysisText) {
    blocks.push(`[analysis_text]\n${analysisText}`);
  }
  setPre('stderr-and-analysis', blocks.length ? blocks.join('\n\n') : t('noStderrAnalysis'));
}

function clearPanels(options = {}) {
  document.getElementById('conversation').innerHTML = '';
  document.getElementById('summary').innerHTML = '';
  document.getElementById('input-preview').textContent = '';
  document.getElementById('structured-results').textContent = '';
  document.getElementById('strong-correlation-diagnostics-panel').classList.add('hidden');
  document.getElementById('strong-correlation-diagnostics').innerHTML = '';
  document.getElementById('active-space-audit-panel').classList.add('hidden');
  document.getElementById('active-space-audit').innerHTML = '';
  document.getElementById('state-energies-panel').classList.add('hidden');
  document.getElementById('state-energies').innerHTML = '';
  document.getElementById('block2-analysis-panel').classList.add('hidden');
  document.getElementById('block2-analysis-summary').innerHTML = '';
  document.getElementById('block2-state-energies').innerHTML = '';
  document.getElementById('block2-orbital-entropy').innerHTML = '';
  document.getElementById('block2-mutual-information').innerHTML = '';
  document.getElementById('block2-bipartite-entanglement').innerHTML = '';
  document.getElementById('llm-result-analysis').textContent = llmResultAnalysisAvailable() ? t('llmResultAnalysisEmpty') : t('llmResultAnalysisUnavailable');
  document.getElementById('defaults-and-retries').innerHTML = '';
  document.getElementById('diagnostics').innerHTML = '';
  document.getElementById('attempts').innerHTML = '';
  document.getElementById('raw-stdout').textContent = '';
  document.getElementById('raw-scf-output').textContent = '';
  document.getElementById('stderr-and-analysis').textContent = '';
  if (!options.preserveState) {
    pendingStructuredRequest = {};
    lastExecutionPayload = null;
    lastExecutedRequestText = '';
    clearApprovalPanel();
    clearComputeConfirmPanel();
    resetPrepareCache();
  }
  updateRequestPlaceholder(null);
  renderStatus('unknown');
  renderStats({ retry_count: 0, max_retries: 0, attempts: [] });
  setAnalyzeState(false);
}

function setRunState(isRunning) {
  const button = document.getElementById('run');
  button.disabled = isRunning;
  button.textContent = isRunning ? t('runLoading') : t('run');
}

function llmResultAnalysisAvailable() {
  return currentPrepareCacheScope !== 'llm:disabled';
}

function canAnalyzeCurrentResult() {
  return llmResultAnalysisAvailable() && !!lastExecutionPayload && !!lastExecutedRequestText;
}

function setAnalyzeState(isRunning) {
  const button = document.getElementById('analyze-result');
  if (isRunning) {
    button.disabled = true;
    button.textContent = t('analyzeLoading');
    return;
  }
  if (!llmResultAnalysisAvailable()) {
    button.disabled = true;
    button.textContent = t('analyzeUnavailable');
    return;
  }
  button.disabled = !canAnalyzeCurrentResult();
  button.textContent = t('analyzeResult');
}

function renderSummary(payload) {
  const rows = [];
  const notes = [];
  const appendRow = (label, value) => {
    if (value === null || typeof value === 'undefined') {
      return;
    }
    const text = Array.isArray(value) ? value.join(', ') : String(value);
    if (!text.trim()) {
      return;
    }
    rows.push([label, text]);
  };
  const appendPreparationMetaRows = () => {
    if (lastPreparationMeta && lastPreparationMeta.applied_defaults && lastPreparationMeta.applied_defaults.length) {
      appendRow(t('summaryDefaultsLabel'), lastPreparationMeta.applied_defaults.map((item) => formatDefaultEntry(item)).join('; '));
    }
    if (lastPreparationMeta && lastPreparationMeta.generated_fields && lastPreparationMeta.generated_fields.length) {
      appendRow(t('summaryGeneratedLabel'), lastPreparationMeta.generated_fields.join(', '));
    }
    if (lastPreparationMeta && lastPreparationMeta.precision_note) {
      appendRow(t('summaryPrecisionLabel'), lastPreparationMeta.precision_note);
    }
  };
  renderStatus(payload.execution_status || 'unknown');
  renderStats(payload);
  const taskSpec = payload.task_spec || {};
  const system = taskSpec.system || {};
  const method = taskSpec.method || {};
  const analysis = taskSpec.analysis || {};
  const taskType = taskSpec.task_type || (payload.structured_results && payload.structured_results.task_type) || 'molecular';
  const modelHamiltonian = taskSpec.model_hamiltonian || {};
  const periodic = taskSpec.periodic || {};
  const solver = taskSpec.solver || {};
  const calculationRole = payload.calculation_role || 'calculation';
  const isActiveSpaceProbe = calculationRole === 'active_space_probe';
  if (taskType !== 'model_hamiltonian' && lastPreparationMeta && lastPreparationMeta.request_summary) {
    appendRow(t('summaryRequestLabel'), lastPreparationMeta.request_summary);
  }
  if (taskType !== 'model_hamiltonian') {
    appendPreparationMetaRows();
  }
  if (payload.analysis_summary && !isModelHamiltonianCompletionSummary(payload.analysis_summary)) {
    notes.push(payload.analysis_summary);
  }
  if (taskType === 'model_hamiltonian') {
    const results = payload.structured_results || {};
    appendRow(langText('Task type', '任务类型'), langText('Model Hamiltonian', '模型哈密顿量'));
    appendRow(langText('Model', '模型'), results.model || modelHamiltonian.model);
    appendRow(langText('Representation', '表示'), results.representation);
    if (modelHamiltonian.input_file) {
      const baseWorkDir = payload.work_dir || effectiveWorkDir();
      appendRow(langText('Input file', '输入文件'), pathRelativeToWorkDir(modelHamiltonian.input_file, baseWorkDir));
    }
    if (solver.name) {
      appendRow(langText('Solver', '求解器'), String(solver.name).toUpperCase());
    }
    if (results.norb !== null && typeof results.norb !== 'undefined') {
      appendRow('norb', results.norb);
    }
    if (results.nelec !== null && typeof results.nelec !== 'undefined') {
      appendRow('nelec', formatTupleValue(results.nelec));
    }
    appendRow(langText('Electrons per cell', '每原胞电子数'), results.electrons_per_cell);
    appendRow(langText('Filling per orbital', '每轨道填充度'), results.filling);
    appendRow('k-mesh', Array.isArray(results.kmesh) ? results.kmesh.join(' x ') : results.kmesh);
    appendRow(langText('k-points', 'k 点数'), results.kpoint_count);
    if (results.band_gap !== null && typeof results.band_gap !== 'undefined') {
      appendRow(langText('Band gap', '能带带隙'), `${results.band_gap} ${results.energy_unit || 'a.u.'}`);
    }
    if (results.direct_gap !== null && typeof results.direct_gap !== 'undefined') {
      appendRow(langText('Direct gap', '直接带隙'), `${results.direct_gap} ${results.energy_unit || 'a.u.'}`);
    }
    if (results.fermi_energy !== null && typeof results.fermi_energy !== 'undefined') {
      appendRow(langText('Fermi energy', '费米能'), `${results.fermi_energy} ${results.energy_unit || 'a.u.'}`);
    }
    if (results.bandwidth !== null && typeof results.bandwidth !== 'undefined') {
      appendRow(langText('Bandwidth', '带宽'), `${results.bandwidth} ${results.energy_unit || 'a.u.'}`);
    }
    if (results.is_metal !== null && typeof results.is_metal !== 'undefined') {
      appendRow(langText('Metallic', '金属性'), results.is_metal ? langText('yes', '是') : langText('no', '否'));
    }
    const interactionTreatment = results.interaction_treatment;
    if (interactionTreatment && interactionTreatment.status === 'recorded_not_applied') {
      appendRow(langText('Interaction treatment', '相互作用处理'), langText('U/V recorded, not applied by tight binding', 'U/V 已记录，但未进入紧束缚求解'));
    }
    if (results.site_count !== null && typeof results.site_count !== 'undefined') {
      appendRow(langText('Sites', '格点数'), results.site_count);
    }
    if (results.bond_count !== null && typeof results.bond_count !== 'undefined') {
      appendRow(langText('Bonds', '键数'), results.bond_count);
    }
    appendPreparationMetaRows();
  }
  if (taskType === 'periodic') {
    const results = payload.structured_results || {};
    const structure = results.periodic_structure && typeof results.periodic_structure === 'object'
      ? results.periodic_structure
      : {};
    const standardization = results.periodic_standardization && typeof results.periodic_standardization === 'object'
      ? results.periodic_standardization
      : {};
    const numerics = results.periodic_numerics && typeof results.periodic_numerics === 'object'
      ? results.periodic_numerics
      : {};
    const kpoints = numerics.kpoints && typeof numerics.kpoints === 'object' ? numerics.kpoints : {};
    const densityFitting = results.density_fitting && typeof results.density_fitting === 'object'
      ? results.density_fitting
      : (numerics.density_fitting || {});
    const smearing = results.smearing && typeof results.smearing === 'object'
      ? results.smearing
      : (numerics.smearing || {});
    appendRow(langText('Task type', '任务类型'), langText('Periodic electronic structure', '周期电子结构'));
    appendRow(langText('Formula', '化学式'), structure.formula);
    appendRow(langText('Atoms per cell', '每晶胞原子数'), structure.atom_count);
    if (structure.cell_role === 'seekpath_standardized_primitive') {
      appendRow(
        langText('Calculation cell', '计算晶胞'),
        langText('SeeK-path standardized primitive cell', 'SeeK-path 标准原胞')
      );
      appendRow(
        langText('Input / primitive atoms', '输入/标准原胞原子数'),
        `${standardization.input_atom_count ?? ''} / ${standardization.primitive_atom_count ?? structure.atom_count ?? ''}`
      );
      if (standardization.volume_original_wrt_primitive !== null
          && typeof standardization.volume_original_wrt_primitive !== 'undefined') {
        appendRow(
          langText('Input / primitive volume', '输入/标准原胞体积比'),
          Number(standardization.volume_original_wrt_primitive).toFixed(4)
        );
      }
      appendRow(
        langText('SeeK-path convention', 'SeeK-path 约定'),
        `${standardization.recipe || 'hpkot'} / symprec=${standardization.symprec_angstrom ?? ''} Angstrom`
      );
    }
    if (structure.source_site_count !== null && typeof structure.source_site_count !== 'undefined') {
      appendRow(langText('Source sites', '输入独立位点数'), structure.source_site_count);
    }
    if (structure.space_group_symbol || structure.space_group_number) {
      appendRow(
        structure.cell_role === 'seekpath_standardized_primitive'
          ? langText('Effective space group', '实际空间群')
          : langText('Source space group', '输入空间群'),
        `${structure.space_group_symbol || ''}${structure.space_group_number ? ` (#${structure.space_group_number})` : ''}`.trim()
      );
    }
    appendRow(
      langText('Normalized symmetry', '规范化对称性'),
      structure.symmetry_expanded ? langText('Expanded to P1', '已展开为 P1') : 'P1'
    );
    appendRow(langText('Structure format', '结构格式'), String(periodic.structure_format || structure.source_format || '').toUpperCase());
    appendRow(
      langText('Lattice lengths', '晶格长度'),
      structure.lattice_lengths_angstrom ? `${formatPeriodicNumberList(structure.lattice_lengths_angstrom)} Angstrom` : null
    );
    appendRow(
      langText('Cell volume', '晶胞体积'),
      typeof structure.cell_volume_angstrom3 !== 'undefined'
        ? `${Number(structure.cell_volume_angstrom3).toFixed(4)} Angstrom^3`
        : null
    );
    appendRow(langText('Basis set', '基组'), results.basis || periodic.basis);
    appendRow(langText('Pseudopotential', '赝势'), results.pseudo || periodic.pseudo);
    if (method.name || results.method) {
      const methodName = String(method.name || results.method).toUpperCase();
      const methodXc = method.xc || results.xc;
      appendRow(langText('Method', '方法'), methodXc ? `${methodName} / ${methodXc}` : methodName);
    }
    const kmesh = results.kmesh || periodic.kmesh;
    appendRow('k-mesh', Array.isArray(kmesh) ? kmesh.join(' x ') : kmesh);
    appendRow(
      langText('k-point scheme / shift', 'k 点方案 / 位移'),
      `${results.kpoint_scheme || kpoints.scheme || periodic.kpoint_scheme || ''} / ${formatPeriodicNumberList(results.kpoint_shift || kpoints.shift_mesh_steps || periodic.kpoint_shift, 3)}`
    );
    appendRow(langText('k-points', 'k 点数'), results.kpoint_count);
    if (typeof numerics.precision !== 'undefined') {
      appendRow(langText('Integral precision', '积分精度'), numerics.precision);
    }
    if (Array.isArray(numerics.fft_mesh)) {
      appendRow(
        langText('Effective FFT mesh', '实际 FFT 网格'),
        `${numerics.fft_mesh.join(' x ')} (${numerics.fft_mesh_source || 'automatic'})`
      );
    }
    if (densityFitting.method) {
      appendRow(
        langText('Density fitting', '密度拟合'),
        `${String(densityFitting.method).toUpperCase()}${densityFitting.auxbasis ? ` / ${densityFitting.auxbasis}` : ''}`
      );
    }
    if (numerics.exxdiv) {
      appendRow(langText('Exact-exchange divergence', '精确交换发散处理'), numerics.exxdiv);
    }
    if (smearing.method && smearing.method !== 'none') {
      appendRow(
        langText('Occupation smearing', '占据展宽'),
        `${smearing.method} / sigma=${smearing.sigma} Ha${smearing.fix_spin ? langText(' / fixed spin', ' / 固定自旋') : ''}`
      );
      if (smearing.free_energy !== null && typeof smearing.free_energy !== 'undefined') {
        appendRow(langText('Smearing free energy', '展宽自由能'), `${smearing.free_energy} Ha/cell`);
      }
      if (smearing.zero_temperature_energy !== null && typeof smearing.zero_temperature_energy !== 'undefined') {
        appendRow(langText('Zero-temperature extrapolated energy', '零温外推能量'), `${smearing.zero_temperature_energy} Ha/cell`);
      }
    }
    if (analysis.outputs && analysis.outputs.length) {
      appendRow(langText('Analysis outputs', '分析输出'), analysis.outputs.join(', '));
    }
    appendRow(langText('Charge / Spin', '电荷 / 自旋'), `${system.charge ?? 0} / ${system.spin ?? 0}`);
    if (results.reference) {
      appendRow(langText('Reference', 'Reference'), String(results.reference).toUpperCase());
    } else if (typeof method.restricted === 'boolean') {
      appendRow(langText('Reference', 'Reference'), method.restricted ? langText('Restricted', '限制性') : langText('Unrestricted', '非限制性'));
    }
    if (results.band_structure_status) {
      appendRow(
        langText('Band structure', '能带结构'),
        results.band_structure_status === 'completed'
          ? `${results.band_path_mode || 'auto'} / ${results.band_path || ''} / ${results.band_path_point_count || 0} ${langText('points', '个点')}`
          : results.band_structure_status
      );
    }
  }
  if (taskType === 'molecular' && system.atom) {
    appendRow(langText('System', '体系'), system.atom);
  }
  if (taskType === 'molecular' && system.basis) {
    appendRow(langText('Basis set', '基组'), system.basis);
  }
  if (taskType === 'molecular' && isActiveSpaceProbe) {
    const activeSpace = taskSpec.active_space || {};
    appendRow(langText('Calculation role', '计算角色'), langText('Active-space probe', '活性空间探测'));
    if (activeSpace.target_method) {
      appendRow(langText('Target method', '目标方法'), String(activeSpace.target_method).toUpperCase());
    }
    if (activeSpace.target_solver) {
      appendRow(langText('Target solver', '目标求解器'), activeSpaceSolverLabel(activeSpace.target_solver));
    }
  }
  if (taskType === 'molecular' && method.name) {
    let methodLabel = method.name.toUpperCase();
    if (method.name === 'dft' && method.xc) {
      methodLabel += ` / ${method.xc}`;
    }
    appendRow(
      isActiveSpaceProbe ? langText('Probe method', '探测方法') : langText('Method', '方法'),
      methodLabel
    );
  }
  if (taskType === 'molecular' && (resultsSolverName(payload.structured_results) || solver.name)
      && (method.name === 'casci' || method.name === 'casscf')) {
    appendRow(
      langText('Solver', '求解器'),
      activeSpaceSolverLabel(resultsSolverName(payload.structured_results) || solver.name)
    );
  }
  if (taskType === 'molecular' && analysis.outputs && analysis.outputs.length) {
    appendRow(langText('Analysis outputs', '分析输出'), analysis.outputs.join(', '));
  }
  if (taskType === 'molecular' && payload.structured_results
      && payload.structured_results.scf_algorithm
      && payload.structured_results.scf_algorithm !== 'standard') {
    appendRow('SCF algorithm', String(payload.structured_results.scf_algorithm).toUpperCase());
  }
  if (taskType === 'molecular' && (typeof system.charge !== 'undefined' || typeof system.spin !== 'undefined')) {
    appendRow(langText('Charge / Spin', '电荷 / 自旋'), `${system.charge ?? 0} / ${system.spin ?? 0}`);
  }
  if (taskType === 'molecular' && (method.name === 'casci' || method.name === 'casscf')) {
    const molecularResults = payload.structured_results || {};
    const initialReference = molecularResults.initial_reference || molecularResults.reference || 'uhf';
    const casReference = molecularResults.cas_reference
      || (molecularResults.cas_result && molecularResults.cas_result.reference)
      || (method.restricted ? ((system.spin ?? 0) === 0 ? 'rhf' : 'rohf') : 'uhf');
    appendRow('Initial reference', String(initialReference).toUpperCase());
    appendRow(
      'CAS reference',
      `${String(casReference).toUpperCase()}${molecularResults.cas_spin_adapted === false ? '' : ' (spin-adapted)'}`
    );
  } else if (taskType === 'molecular' && typeof method.restricted === 'boolean') {
    appendRow(langText('Reference', 'Reference'), method.restricted ? langText('Restricted', '限制性') : langText('Unrestricted', '非限制性'));
  }
  if (payload.structured_results
      && payload.structured_results.energy !== undefined
      && payload.structured_results.energy !== null) {
    const energyUnit = payload.structured_results.energy_unit || (taskType === 'model_hamiltonian' ? 'a.u.' : 'Ha');
    const energyLabel = taskType === 'periodic'
      ? langText('Energy per cell', '每晶胞总能')
      : payload.structured_results.energy_kind === 'noninteracting_band_energy_per_cell'
      ? langText('Band energy per cell', '每原胞能带能量')
      : payload.structured_results.energy_kind === 'cas_energy'
      ? langText('CAS energy', 'CAS 能量')
      : langText('Total energy', '总能');
    appendRow(energyLabel, `${payload.structured_results.energy} ${energyUnit}`);
  }
  if (taskType === 'model_hamiltonian' && payload.structured_results) {
    if (payload.structured_results.energy_over_abs_t !== null
        && typeof payload.structured_results.energy_over_abs_t !== 'undefined') {
      appendRow('Energy / mean |t|', Number(payload.structured_results.energy_over_abs_t).toFixed(6));
    }
    if (payload.structured_results.energy_per_site_over_abs_t !== null
        && typeof payload.structured_results.energy_per_site_over_abs_t !== 'undefined') {
      appendRow(
        'Energy per site / mean |t|',
        Number(payload.structured_results.energy_per_site_over_abs_t).toFixed(6)
      );
    }
    const dmet = payload.structured_results.dmet_result;
    if (dmet && typeof dmet === 'object') {
      const dmetConfiguration = dmet.configuration && typeof dmet.configuration === 'object'
        ? dmet.configuration
        : {};
      if (dmet.energy_per_site !== null && typeof dmet.energy_per_site !== 'undefined') {
        appendRow('DMET energy per site', `${dmet.energy_per_site} ${payload.structured_results.energy_unit || 'a.u.'}`);
      }
      const localObservables = (
        payload.structured_results.dmet_local_observables
        && typeof payload.structured_results.dmet_local_observables === 'object'
      )
        ? payload.structured_results.dmet_local_observables
        : ((dmet.local_observables && typeof dmet.local_observables === 'object')
          ? dmet.local_observables
          : {});
      if (localObservables.status === 'available') {
        const coverage = localObservables.coverage && typeof localObservables.coverage === 'object'
          ? localObservables.coverage
          : {};
        const coveredValue = (value, count, total) => {
          const numeric = Number(value);
          if (!Number.isFinite(numeric)) return null;
          const covered = Number(count);
          const available = Number(total);
          const suffix = Number.isFinite(covered) && Number.isFinite(available) && available > 0
            ? ` (${covered}/${available} covered)`
            : '';
          return `${numeric.toFixed(6)}${suffix}`;
        };
        const density = Array.isArray(localObservables.density)
          ? localObservables.density.map(Number).filter(Number.isFinite)
          : [];
        if (density.length) {
          appendRow(
            'Local density range',
            `${Math.min(...density).toFixed(6)} to ${Math.max(...density).toFixed(6)}`
          );
        }
        const magnetization = Array.isArray(localObservables.local_magnetization)
          ? localObservables.local_magnetization.map((value) => Math.abs(Number(value))).filter(Number.isFinite)
          : [];
        if (magnetization.length) {
          appendRow('Maximum |local magnetization|', Math.max(...magnetization).toFixed(6));
        }
        appendRow(
          'Mean double occupancy',
          coveredValue(
            localObservables.mean_double_occupancy,
            coverage.double_occupancy_site_count,
            coverage.site_count
          )
        );
        appendRow(
          'Nearest-neighbor spin correlation',
          coveredValue(
            localObservables.nearest_neighbor_spin_correlation,
            coverage.spin_correlation_bond_term_count,
            coverage.bond_term_count
          )
        );
        appendRow(
          'Nearest-neighbor charge correlation',
          coveredValue(
            localObservables.nearest_neighbor_charge_correlation,
            coverage.charge_correlation_bond_term_count,
            coverage.bond_term_count
          )
        );
        appendRow(
          'Nearest-neighbor one-body coherence',
          coveredValue(
            localObservables.mean_nearest_neighbor_one_body_coherence,
            coverage.bond_term_count,
            coverage.bond_term_count
          )
        );
        const sublattice = localObservables.sublattice_order
          && typeof localObservables.sublattice_order === 'object'
          ? localObservables.sublattice_order
          : {};
        appendRow(
          'Sublattice charge imbalance',
          Number.isFinite(Number(sublattice.charge_imbalance))
            ? Number(sublattice.charge_imbalance).toFixed(6)
            : null
        );
        appendRow(
          'Staggered magnetization',
          Number.isFinite(Number(sublattice.staggered_magnetization))
            ? Number(sublattice.staggered_magnetization).toFixed(6)
            : null
        );
      }
      appendRow('DMET execution mode', dmetConfiguration.execution_mode === 'finite_graph'
        ? 'Finite graph fragment partition'
        : (dmetConfiguration.execution_mode === 'translational' ? 'Translated fragment' : null));
      appendRow('DMET translation source', dmetConfiguration.translation_backend === 'builder_primitive_cell'
        ? 'Audited Builder primitive cell'
        : (dmetConfiguration.translation_backend === 'native_lattice' ? 'Native lattice template' : null));
      appendRow('DMET partition rationale', dmetConfiguration.execution_mode_reason);
      const dmetFragments = Array.isArray(dmet.fragments) ? dmet.fragments : [];
      const fragmentSummary = dmetFragments.length
        ? dmetFragments.map((fragment, index) => {
          const label = fragment.label || fragment.fragment_id || `Fragment ${index + 1}`;
          const sites = Array.isArray(fragment.site_ids) ? fragment.site_ids.join(', ') : '';
          return sites ? `${label} [sites ${sites}]` : label;
        }).join('; ')
        : (Array.isArray(dmetConfiguration.impurity_shape)
          ? dmetConfiguration.impurity_shape.join(' x ')
          : null);
      appendRow('DMET fragments', fragmentSummary);
      appendRow(
        'DMET impurity solver',
        dmet.impurity_solver === 'block2_dmrg'
          ? 'block2 DMRG'
          : (dmet.impurity_solver ? String(dmet.impurity_solver).toUpperCase() : null)
      );
      if (dmet.smearing && dmet.smearing.beta != null) {
        appendRow('CCSD / DMET β (SCF / density fit)', dmet.smearing.beta);
        appendRow('Fermi smearing width (model-energy units)', dmet.smearing.sigma);
      }
      const impuritySolverDetails = Array.isArray(dmet.impurity_solver_details)
        ? dmet.impurity_solver_details
        : [];
      if (dmet.impurity_solver === 'ccsd') {
        const betas = [...new Set(impuritySolverDetails
          .flatMap((details) => details.scf_calls || [])
          .filter((call) => Object.hasOwn(call, 'beta'))
          .map((call) => call.beta === null ? '∞ (zero-temperature)' : String(call.beta)))];
        appendRow('CCSD impurity SCF β', betas.join(', ') || null);
      }
      if (dmet.impurity_solver === 'block2_dmrg' && impuritySolverDetails.length) {
        const impurityCalls = impuritySolverDetails.reduce(
          (total, details) => total + Number(details.call_count || 0),
          0
        );
        const impurityConverged = impuritySolverDetails.every(
          (details) => details.converged === true
        );
        const discardedWeights = impuritySolverDetails
          .map((details) => Number((details.convergence || {}).final_discarded_weight))
          .filter((value) => Number.isFinite(value));
        const bondDimensions = impuritySolverDetails
          .map((details) => {
            const convergence = details.convergence || {};
            const schedule = convergence.adaptive_schedule || {};
            const plan = details.bond_dimension_plan || {};
            return Number(
              schedule.final_bond_dimension
              || plan.effective_max_bond_dimension
              || 0
            );
          })
          .filter((value) => Number.isFinite(value) && value > 0);
        const continuedCalls = impuritySolverDetails.reduce(
          (total, details) => total + (Array.isArray(details.calls)
            ? details.calls.filter((call) => call.restart_applied === true).length
            : 0),
          0
        );
        appendRow('DMRG impurity convergence', impurityConverged ? 'Converged' : 'Not converged');
        appendRow('DMRG impurity solves', impurityCalls || null);
        appendRow(
          'DMRG final discarded weight',
          discardedWeights.length ? Math.max(...discardedWeights) : null
        );
        appendRow(
          'DMRG maximum bond dimension',
          bondDimensions.length ? Math.max(...bondDimensions) : null
        );
        appendRow('DMRG MPS continuations', continuedCalls);
      }
      const dmetBath = dmet.bath && typeof dmet.bath === 'object' ? dmet.bath : {};
      appendRow('DMET bath interactions', dmetBath.interacting_bath === true
        ? 'Interacting'
        : (dmetBath.interacting_bath === false ? 'Noninteracting approximation' : null));
      const initialVcor = dmet.initial_correlation_potential
        && typeof dmet.initial_correlation_potential === 'object'
        ? dmet.initial_correlation_potential
        : {};
      appendRow('DMET initial correlation potential', initialVcor.strategy === 'zero'
        ? 'Zero auxiliary correction'
        : initialVcor.strategy);
      const referenceDensityInitialization = dmet.reference_density_initialization
        && typeof dmet.reference_density_initialization === 'object'
        ? dmet.reference_density_initialization
        : {};
      appendRow(
        'DMET initial spin density',
        referenceDensityInitialization.label || (
          referenceDensityInitialization.strategy
            ? String(referenceDensityInitialization.strategy).toUpperCase()
            : null
          )
      );
      appendRow('DMET spin-density bias', referenceDensityInitialization.bias_kind === 'staggered_spin_density'
        ? referenceDensityInitialization.applied_bias
        : null);
      appendRow('DMET reference', dmetConfiguration.reference === 'unrestricted'
        ? 'Unrestricted'
        : dmetConfiguration.reference);
      const dmetSpinReference = dmet.spin_reference && typeof dmet.spin_reference === 'object'
        ? dmet.spin_reference
        : {};
      const dmetLibdmetSz = Number(dmetSpinReference.libdmet_sz);
      const dmetPhysicalSz = Number(dmetSpinReference.total_sz);
      const dmetNalpha = Number(dmetSpinReference.nalpha);
      const dmetNbeta = Number(dmetSpinReference.nbeta);
      if (Number.isFinite(dmetLibdmetSz) && Number.isFinite(dmetPhysicalSz)) {
        const particleSector = Number.isFinite(dmetNalpha) && Number.isFinite(dmetNbeta)
          ? `Nalpha=${dmetNalpha}, Nbeta=${dmetNbeta}; `
          : '';
        appendRow('DMET spin sector', `${particleSector}libDMET Sz=${dmetLibdmetSz}; physical Sz=${dmetPhysicalSz}`);
      }
      const impuritySectors = dmetFragments.map((fragment, index) => {
        const sector = fragment.impurity_spin_sector && typeof fragment.impurity_spin_sector === 'object'
          ? fragment.impurity_spin_sector
          : null;
        if (!sector) return null;
        const label = fragment.label || fragment.fragment_id || `Fragment ${index + 1}`;
        return `${label}: Nalpha=${sector.nalpha}, Nbeta=${sector.nbeta}, libDMET Sz=${sector.libdmet_sz}`;
      }).filter(Boolean).join('; ');
      if (impuritySectors) {
        appendRow('DMET impurity spin sectors', impuritySectors);
      }
      if (dmetConfiguration.reference === 'unrestricted' && dmetFragments.length) {
        const spinSummary = dmetFragments.map((fragment, index) => {
          const label = fragment.label || fragment.fragment_id || `Fragment ${index + 1}`;
          const spin = fragment.spin_population && typeof fragment.spin_population === 'object'
            ? fragment.spin_population
            : {};
          const alpha = Number(spin.alpha_electron_count);
          const beta = Number(spin.beta_electron_count);
          const maximum = Number(spin.maximum_absolute_local_magnetization);
          if (![alpha, beta, maximum].every(Number.isFinite)) return null;
          return `${label}: Nalpha=${alpha.toFixed(4)}, Nbeta=${beta.toFixed(4)}, max |nalpha-nbeta|=${maximum.toFixed(4)}`;
        }).filter(Boolean).join('; ');
        appendRow('DMET fragment spin populations', spinSummary || null);
      }
      appendRow('DMET converged', dmet.converged === true ? 'yes' : (dmet.converged === false ? 'no' : null));
      if (dmet.iteration_history && typeof dmet.iteration_history === 'object') {
        appendRow('DMET iterations', dmet.iteration_history.iteration_count);
        const convergence = dmet.iteration_history.convergence_contract;
        if (convergence && typeof convergence === 'object') {
          const energyTolerance = Number(convergence.energy_tolerance);
          const densityTolerance = Number(convergence.density_tolerance);
          if (Number.isFinite(energyTolerance) && Number.isFinite(densityTolerance)) {
            appendRow(
              'DMET convergence criteria',
              `|dE| < ${energyTolerance.toExponential(1)}; max |d1RDM| < ${densityTolerance.toExponential(1)}`
            );
          }
        }
        const records = Array.isArray(dmet.iteration_history.records)
          ? dmet.iteration_history.records
          : [];
        const finalRecord = records.length ? records[records.length - 1] : null;
        if (finalRecord) {
          const energyChange = Number(finalRecord.energy_change);
          const densityChange = Number(finalRecord.mean_field_rdm1_change_max_abs);
          if (Number.isFinite(energyChange) && Number.isFinite(densityChange)) {
            appendRow(
              'Final DMET changes',
              `dE=${energyChange.toExponential(3)}; max |d1RDM|=${densityChange.toExponential(3)}`
            );
          }
        }
      }
    }
  }
  if (taskType === 'periodic' && payload.structured_results) {
    const gw = payload.structured_results.gw_result;
    if (gw && typeof gw === 'object') {
      appendRow('GW status', gw.status || 'completed');
      appendRow('GW reference', gw.reference_method ? String(gw.reference_method).toUpperCase() : null);
      appendRow(
        'Reference SCF converged',
        payload.structured_results.reference_converged === true
          ? 'yes'
          : (payload.structured_results.reference_converged === false ? 'no' : null)
      );
      if (gw.quasiparticle_gap !== null && typeof gw.quasiparticle_gap !== 'undefined') {
        appendRow('GW quasiparticle gap', `${gw.quasiparticle_gap} Ha`);
      }
      appendRow('GW k points', gw.kpoint_count);
      appendRow('GW orbitals', gw.orbital_count);
      appendRow(
        'GW total energy',
        gw.energy_available === false ? 'not provided by the current fcDMFT adapter' : null
      );
    }
    const dmft = payload.structured_results.dmft_result;
    if (dmft && typeof dmft === 'object') {
      const isGwDmft = String(dmft.method || '').toLowerCase().replace('-', '_') === 'gw_dmft';
      const referenceLabel = isGwDmft ? 'DFT reference converged' : 'HF reference converged';
      appendRow(referenceLabel, payload.structured_results.reference_converged === true ? 'yes' : 'no');
      if (payload.structured_results.reference_energy !== null
          && typeof payload.structured_results.reference_energy !== 'undefined') {
        appendRow(`${isGwDmft ? 'DFT' : 'HF'} mean-field reference energy`, `${payload.structured_results.reference_energy} Ha/cell`);
      }
      appendRow(`${isGwDmft ? 'GW+DMFT' : 'HF+DMFT'} converged`, dmft.converged === true ? 'yes' : 'no');
      appendRow('DMFT impurity solver', dmft.impurity_solver ? String(dmft.impurity_solver).toUpperCase() : null);
      appendRow('Correlated orbitals', dmft.localized_orbital_count);
      if (dmft.ncore !== null && typeof dmft.ncore !== 'undefined'
          && dmft.nval !== null && typeof dmft.nval !== 'undefined') {
        appendRow('Correlated window', `[${dmft.ncore}, ${dmft.nval})`);
      }
      appendRow(
        'DMFT bath',
        dmft.nbath !== null && typeof dmft.nbath !== 'undefined'
          ? `${dmft.nbath} orbitals / ${dmft.bath_discretization || 'unspecified'}`
          : null
      );
      appendRow('DMFT maximum iterations', dmft.max_iterations);
      appendRow('DMFT hybridization tolerance', dmft.convergence_tolerance);
      if (dmft.chemical_potential !== null && typeof dmft.chemical_potential !== 'undefined') {
        appendRow('DMFT chemical potential', `${dmft.chemical_potential} Ha`);
      }
      appendRow('Initial hybridization norm', dmft.initial_hybridization_norm);
      const resources = dmft.runtime_resources && typeof dmft.runtime_resources === 'object'
        ? dmft.runtime_resources
        : {};
      if (resources.n_threads || resources.max_memory_mb) {
        appendRow(
          'DMFT resources',
          `${resources.n_threads || '?'} threads / ${resources.max_memory_mb || '?'} MB`
        );
      }
      appendRow(
        'DMFT total energy',
        dmft.energy_available === false ? 'not provided by the current fcDMFT adapter' : null
      );
    }
  }
  if (payload.structured_results && payload.structured_results.post_cas_results && payload.structured_results.post_cas_results.sc_nevpt2) {
    const scNevpt2 = payload.structured_results.post_cas_results.sc_nevpt2;
    if (scNevpt2.correction_energy !== null && typeof scNevpt2.correction_energy !== 'undefined') {
      appendRow('SC-NEVPT2 correction', `${scNevpt2.correction_energy} Ha`);
    }
  }
  if (payload.structured_results && payload.structured_results.final_method) {
    appendRow(langText('Final method', '最终方法'), payload.structured_results.final_method);
  }
  if (payload.structured_results && payload.structured_results.final_energy !== null && typeof payload.structured_results.final_energy !== 'undefined') {
    const finalEnergyUnit = payload.structured_results.energy_unit || 'Ha';
    appendRow(langText('Final energy', '最终能量'), `${payload.structured_results.final_energy} ${finalEnergyUnit}`);
  }
  if (payload.structured_results && payload.structured_results.converged !== undefined) {
    const hasDmftResult = taskType === 'periodic'
      && payload.structured_results.dmft_result
      && typeof payload.structured_results.dmft_result === 'object';
    const hasGwResult = taskType === 'periodic'
      && payload.structured_results.gw_result
      && typeof payload.structured_results.gw_result === 'object';
    const isMolecularCas = taskType === 'molecular'
      && (method.name === 'casci' || method.name === 'casscf');
    const convergenceLabel = taskType === 'model_hamiltonian'
      ? langText('Solver converged', '求解器收敛')
      : (isMolecularCas
        ? `${String(method.name).toUpperCase()} ${langText('converged', '收敛')}`
        : ((hasDmftResult || hasGwResult) ? null : `SCF ${langText('converged', '收敛')}`));
    if (convergenceLabel) {
      appendRow(convergenceLabel, payload.structured_results.converged ? langText('yes', '是') : langText('no', '否'));
    }
    if (isMolecularCas && payload.structured_results.reference_converged !== undefined) {
      const warningOnly = payload.structured_results.reference_status
        && payload.structured_results.reference_status.affects_task_status === false
        && payload.structured_results.reference_converged === false;
      appendRow(
        langText('Initial reference converged', '初始参考态收敛'),
        `${payload.structured_results.reference_converged ? langText('yes', '是') : langText('no', '否')}${warningOnly ? langText(' (warning only)', '（仅警告）') : ''}`
      );
    }
  }
  if (taskType === 'model_hamiltonian' && payload.structured_results && payload.structured_results.reference_converged !== undefined && payload.structured_results.reference_converged !== null) {
    appendRow(langText('Reference SCF converged', '参考 SCF 收敛'), payload.structured_results.reference_converged ? langText('yes', '是') : langText('no', '否'));
  }
  const displayedGap = payload.structured_results
    ? (taskType === 'periodic' ? payload.structured_results.band_gap : payload.structured_results.gap)
    : null;
  if (displayedGap !== undefined && displayedGap !== null) {
    const gapLabel = taskType === 'periodic'
      ? langText('Mean-field band gap', '平均场带隙')
      : langText('Gap', '能隙');
    appendRow(gapLabel, `${displayedGap} Ha`);
  }
  if (taskType === 'periodic' && payload.structured_results) {
    const results = payload.structured_results;
    if (results.direct_gap !== undefined && results.direct_gap !== null) {
      appendRow(langText('Direct band gap', '直接带隙'), `${results.direct_gap} Ha`);
    }
    if (results.gap_type) {
      appendRow(langText('Gap character', '带隙类型'), results.gap_type);
    }
    if (typeof results.is_metal === 'boolean') {
      appendRow(langText('Electronic character', '电子结构类型'), results.is_metal
        ? langText('metallic / fractionally occupied', '金属或存在分数占据')
        : langText('insulating / semiconducting', '绝缘体或半导体'));
    }
    if (results.valence_band_max !== undefined && results.valence_band_max !== null) {
      appendRow(langText('Valence-band maximum', '价带顶'), `${results.valence_band_max} Ha`);
    }
    if (results.conduction_band_min !== undefined && results.conduction_band_min !== null) {
      appendRow(langText('Conduction-band minimum', '导带底'), `${results.conduction_band_min} Ha`);
    }
  }
  if (taskType === 'periodic' && payload.structured_results && payload.structured_results.fermi_energy !== undefined && payload.structured_results.fermi_energy !== null) {
    appendRow(langText('Fermi level', '费米能级'), `${payload.structured_results.fermi_energy} Ha`);
    const fermiBySpin = payload.structured_results.fermi_energy_by_spin;
    if (fermiBySpin && typeof fermiBySpin === 'object' && Object.keys(fermiBySpin).length > 1) {
      appendRow(
        langText('Fermi level by spin', '分自旋费米能级'),
        Object.entries(fermiBySpin).map(([spin, value]) => `${spin}=${value} Ha`).join(', ')
      );
    }
    appendRow(
      langText('Fermi convention', '费米能级约定'),
      payload.structured_results.fermi_energy_source === 'pyscf_get_fermi_smearing_chemical_potential'
        ? langText('PySCF get_fermi; smearing chemical potential', 'PySCF get_fermi；展宽占据下的化学势')
        : payload.structured_results.fermi_energy_source === 'pyscf_get_fermi_vbm_convention'
          ? langText('PySCF get_fermi; VBM for integer occupations', 'PySCF get_fermi；整数占据时取价带顶')
          : langText('VBM fallback', '价带顶回退值')
    );
  }
  if (payload.structured_results && payload.structured_results.dipole) {
    appendRow(langText('Dipole', '偶极矩'), payload.structured_results.dipole.join(', '));
  }
  if (payload.result_analysis) {
    notes.push(payload.result_analysis);
  }
  renderTableWithNotes('summary', [langText('Item', '项目'), langText('Value', '数值')], rows, notes, t('summaryEmpty'), 'compact');
  renderStrongCorrelationDiagnostics(payload.structured_results);
  renderActiveSpaceAudit(payload.structured_results);
  renderTargetedStateEnergies(payload.structured_results);
  renderBlock2Analysis(payload.structured_results);
  renderStructuredResults(payload.structured_results);
  setPre(
    'llm-result-analysis',
    payload.result_analysis || (llmResultAnalysisAvailable() ? t('llmResultAnalysisEmpty') : t('llmResultAnalysisUnavailable'))
  );
  renderDefaultsAndRetries(payload);
  renderDiagnostics(payload);
  renderAttempts(payload);
  renderRawOutputs(payload);
  setAnalyzeState(false);
}

function renderPreparationState(payload) {
  const hasApproval = Boolean(payload.approval);
  const preparationStatus = hasApproval
    ? 'needs_review'
    : (payload.status === 'ready'
      ? 'prepared'
      : (payload.status === 'changed' ? 'changed' : 'needs_input'));
  const preparationNote = preparationStatus === 'prepared'
    ? t('preparationReadyNote')
    : (preparationStatus === 'changed'
      ? t('preparationChangedNote')
      : (preparationStatus === 'needs_review' ? t('preparationReviewNote') : t('preparationNeedsInputNote')));
  renderStatus(preparationStatus);
  clearComputeConfirmPanel();
  renderStats({ retry_count: 0, max_retries: 0, attempts: [] });
  const summaryRows = [];
  const appendRow = (label, value) => {
    if (value === null || typeof value === 'undefined') {
      return;
    }
    const text = Array.isArray(value) ? value.join(', ') : String(value);
    if (!text.trim()) {
      return;
    }
    summaryRows.push([label, text]);
  };
  if (payload.structured_request && payload.structured_request.request_summary) {
    appendRow(t('summaryRequestLabel'), payload.structured_request.request_summary);
  }
  if (payload.applied_defaults && payload.applied_defaults.length) {
    appendRow(t('summaryDefaultsLabel'), payload.applied_defaults.map((item) => formatDefaultEntry(item)).join('; '));
  }
  if (payload.generated_fields && payload.generated_fields.length) {
    appendRow(t('summaryGeneratedLabel'), payload.generated_fields.join(', '));
  }
  if (payload.precision_note) {
    appendRow(t('summaryPrecisionLabel'), payload.precision_note);
  }
  if (payload.approval && payload.approval.type === 'structure') {
    appendRow(t('summaryApprovalPending'), langText('Pending', '待确认'));
  }
  if (payload.clarification_questions && payload.clarification_questions.length) {
    appendRow(t('summaryNeedsInput'), langText('Pending', '待补充'));
  }
  renderTableWithNotes('summary', [t('summaryItemLabel'), t('summaryValueLabel')], summaryRows, [preparationNote], t('notExecutedYet'), 'compact');
  setPre('structured-results', JSON.stringify(payload.structured_request || {}, null, 2));
  setPre('input-preview', '');
  renderList('defaults-and-retries', [], t('noDefaultsOrRetries'));
  renderList('diagnostics', [
    ...(payload.missing_fields || []).map((field) => `${t('diagnosticMissingField')}: ${field}`),
    ...(payload.clarification_questions || []),
  ], t('noDiagnostics'));
  renderList('attempts', [], t('notExecutedYet'));
  setPre('raw-scf-output', '');
  setPre('raw-stdout', '');
  setPre('stderr-and-analysis', '');
  setPre('llm-result-analysis', llmResultAnalysisAvailable() ? t('llmResultAnalysisEmpty') : t('llmResultAnalysisUnavailable'));
  if (preparationStatus === 'changed') {
    clearApprovalPanel();
  } else {
    renderApprovalPanel(
      payload.approval && payload.approval.type === 'structure'
        ? payload.approval
        : buildTaskApprovalFromResult(payload)
    );
  }
  setAnalyzeState(false);
}
