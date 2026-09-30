// Scientific selection and execution live in the Study service.
let dmetContinuationPreview = null;

function renderDmetBranchSummary(report) {
  const analysis = report.dmet_phase_analysis;
  if (!analysis) return '';
  const points = analysis.points || [];
  const policy = analysis.policy || {};
  const rows = points.map(point => {
    const winner = point.winner;
    const origin = winner && winner.density_source
      ? `From ${winner.density_source.source_case_id || 'saved density'} / ${winner.density_source.source_run_id || ''}`
      : `Independent ${winner && winner.seed || ''} seed`;
    return `<tr><td>${escapeHtml(point.coordinates.U)}</td><td>${escapeHtml(point.coordinates.V)}</td>`
      + `<td>${escapeHtml(winner ? winner.branch : 'No qualified result')}</td>`
      + `<td>${escapeHtml(winner ? winner.energy_per_site : '—')}</td>`
      + `<td>${escapeHtml(winner ? winner.run_id : '—')}</td><td>${escapeHtml(winner ? origin : '—')}</td>`
      + `<td>${point.candidates.length}</td><td>${point.possible_hysteresis ? 'Possible hysteresis' : '—'}</td></tr>`;
  });
  const crossings = (analysis.crossings || []).map(item =>
    `U=${item.U}: V in [${item.V_bracket.join(', ')}] (${item.status === 'lower_competing_state' ? 'a third state has lower energy' : 'candidate energy crossing'})`);
  return '<section><h3>DMET competing states</h3>'
    + `<p>Lowest qualified energy across saved runs. Charge / magnetic thresholds: ${escapeHtml(policy.charge_threshold)} / ${escapeHtml(policy.magnetic_threshold)} (sublattice half-differences).</p>`
    + '<p>Possible hysteresis marks distinct converged states. Failed calculations do not establish branch endpoints.</p>'
    + (rows.length ? '<div class="table-wrap"><table><thead><tr><th>U</th><th>V</th><th>State</th><th>Energy/site</th><th>Selected run</th><th>Origin</th><th>Saved runs</th><th>Coexistence</th></tr></thead><tbody>' + rows.join('') + '</tbody></table></div>' : '<p>No comparable saved DMET results yet.</p>')
    + `<p>${escapeHtml(crossings.join('; ') || 'No bracketed energy crossing yet.')}</p>`
    + ((analysis.unclassified_identity_runs || []).length ? `<p>${analysis.unclassified_identity_runs.length} runs have unavailable or incompatible model information.</p>` : '')
    + (currentSavedStudy ? '<button type="button" data-dmet-action="analyze">Update state analysis and plots</button> <button type="button" data-dmet-action="prepare">Preview two-way density continuation</button><div id="dmet-continuation-preview"></div>' : '')
    + '</section>';
}

document.getElementById('report-summary').addEventListener('click', async event => {
  const button = event.target.closest('[data-dmet-action]');
  const identity = currentSavedStudy;
  if (!button || !identity) return;
  button.disabled = true;
  const action = button.dataset.dmetAction;
  try {
    if (action === 'analyze') {
      await postJson('/api/study-dmet-branches-analyze', {...savedStudyRequest(), policy: (currentReport.dmet_phase_analysis || {}).policy});
      if (currentSavedStudy === identity) await refreshSavedStudy();
    } else if (action === 'prepare') {
      const preview = await postJson('/api/study-dmet-continuation-prepare', {...savedStudyRequest(), policy: (currentReport.dmet_phase_analysis || {}).policy});
      if (currentSavedStudy !== identity) return;
      dmetContinuationPreview = {identity, action: preview};
      const cost = preview.cost_estimate || {};
      const unavailable = (cost.totals || {}).unavailable_case_count || 0;
      const blocked = cost.approval_allowed === false || cost.resource_limit_exceeded;
      document.getElementById('dmet-continuation-preview').innerHTML =
        `<p>Up to ${preview.max_runs} new runs: AFM toward larger V, then CDW toward smaller V, at fixed U. Each step selects the nearest compatible converged state on that branch. Steps without a donor are skipped.</p>`
        + '<p>Density only; initial auxiliary potential is zero. Mixing 0.2, DIIS off, spin bath policy max. All earlier runs remain available.</p>'
        + `<p>${escapeHtml(cost.summary || cost.status || '')}</p>`
        + (unavailable ? `<p>Resource estimates are unavailable for ${unavailable} planned runs. The run count is bounded; runtime and memory are not predicted.</p>` : '')
        + `<button type="button" data-dmet-action="start"${blocked ? ' disabled' : ''}>${cost.can_execute ? 'Start continuation' : 'Approve cost and start continuation'}</button>`;
    } else if (action === 'start') {
      const preview = dmetContinuationPreview;
      if (!preview || preview.identity !== identity) return;
      dmetContinuationPreview = null;
      const response = await postJson('/api/study-dmet-continuation-start', {
        ...savedStudyRequest(), action_id: preview.action.action_id,
        approve_cost: !(preview.action.cost_estimate || {}).can_execute,
      });
      if (currentSavedStudy !== identity) return;
      setStatus('run-status', response.started ? 'Density continuation started.' : 'This continuation was already submitted.', 'ok');
      await refreshSavedStudy();
    }
  } catch (error) {
    if (currentSavedStudy === identity) setStatus('run-status', error.message, 'error');
  } finally {
    if (action !== 'start') button.disabled = false;
  }
});
