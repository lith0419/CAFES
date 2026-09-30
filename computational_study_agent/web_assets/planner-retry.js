// A retry is a saved action; it never owns the full plan or result table.
let retrySelection = null;
let retryPreviewVersion = 0;

function decorateCaseRetryTable(target, rows) {
  if (!plannerState.savedStudy || !plannerState.report) return;
  const heading = target.querySelector('thead tr');
  if (heading) heading.insertAdjacentHTML('beforeend', '<th>Retry</th>');
  target.querySelectorAll('tbody tr').forEach((row, index) => {
    const id = rows[index] && rows[index].case_id;
    row.insertAdjacentHTML('beforeend', id
      ? `<td><button type="button" class="secondary" data-retry-case="${escapeHtml(id)}">Retry this point</button></td>`
      : '<td></td>');
  });
}

function closeRetryEditor() {
  retrySelection = null;
  retryPreviewVersion += 1;
  const editor = document.getElementById('retry-editor');
  if (editor) editor.hidden = true;
  const preview = document.getElementById('retry-preview');
  if (preview && !plannerState.pendingAction) preview.hidden = true;
}

async function openRetryEditor(caseIds) {
  const identity = plannerState.savedStudy;
  if (!identity) return;
  const version = ++retryPreviewVersion;
  try {
    const context = await postJson('/api/study-retry-context', savedStudyRequest());
    if (identity !== plannerState.savedStudy || version !== retryPreviewVersion) return;
    retrySelection = {identity, context, caseIds: [...caseIds]};
    document.getElementById('retry-case-label').textContent = caseIds.join(', ');
    document.getElementById('retry-editor').hidden = false;
    document.getElementById('retry-dependents').checked = false;
    document.getElementById('retry-parameter').value = '';
    document.getElementById('retry-value').value = '';
  } catch (error) {
    setStatus('run-status', error.message, 'error');
  }
}

async function prepareSelectedRetry() {
  const selection = retrySelection;
  if (!selection || selection.identity !== plannerState.savedStudy) return;
  const version = ++retryPreviewVersion;
  const button = document.getElementById('preview-retry');
  button.disabled = true;
  try {
    const path = document.getElementById('retry-parameter').value;
    const patch = path ? {[path]: JSON.parse(document.getElementById('retry-value').value)} : {};
    const result = await postJson('/api/study-retry-prepare', {
      ...savedStudyRequest(), retry_action: {
        ...selection.context, case_ids: selection.caseIds,
        case_overrides: Object.fromEntries(selection.caseIds.map(id => [id, patch])),
        include_dependents: document.getElementById('retry-dependents').checked,
        reason: 'Retry selected result rows',
      },
    });
    if (selection.identity !== plannerState.savedStudy || version !== retryPreviewVersion) return;
    plannerState.pendingAction = result;
    renderRetryPreview();
    document.getElementById('retry-editor').hidden = true;
    setStatus('run-status', 'Retry preview ready. Review the scope before confirming.', 'ok');
  } catch (error) {
    setStatus('run-status', error.message, 'error');
  } finally {
    button.disabled = false;
  }
}

function renderRetryPreview() {
  const target = document.getElementById('retry-preview');
  if (!target) return;
  const action = plannerState.pendingAction;
  target.hidden = !action || action.kind !== 'retry_cases';
  document.getElementById('run-study').textContent = target.hidden ? 'Run Plan' : 'Review Retry';
  if (target.hidden) return;
  const ids = action.execution_case_ids || [];
  const changes = (action.changes || []).map(change =>
    `${change.case_id}: ${change.path}: ${JSON.stringify(change.before)} → ${JSON.stringify(change.after)}`);
  const contexts = (action.contexts || []).filter(item => item.dependency_error || item.awaits_selected_parent);
  const estimate = action.cost_estimate || {};
  const needsCostApproval = !estimate.can_execute;
  const costBlocked = estimate.approval_allowed === false || estimate.resource_limit_exceeded;
  const totals = estimate.totals || {};
  target.innerHTML = [
    '<h3>Confirm Retry</h3>',
    `<p>Will retry ${ids.length} of ${action.total_case_count} cases: <strong>${escapeHtml(ids.join(', '))}</strong></p>`,
    `<p>${escapeHtml(changes.join('\n') || 'Current settings will be retained.')}</p>`,
    action.dependent_case_ids && action.dependent_case_ids.length
      ? `<p>Dependent points: ${escapeHtml(action.dependent_case_ids.join(', '))}. ${action.action.include_dependents ? 'Included above.' : 'Their saved results will be retained.'}</p>` : '',
    contexts.length ? `<p>${escapeHtml(contexts.map(c => `${c.case_id}: ${c.awaits_selected_parent ? 'waits for the selected parent; blocked if it fails' : c.dependency_error}`).join('; '))}</p>` : '',
    `<p>Cost review: ${escapeHtml(estimate.summary || estimate.status || 'See estimated resources')}${needsCostApproval ? ' — confirmation also approves the estimated cost for these cases.' : ''}</p>`,
    `<p>Estimated peak memory: ${escapeHtml(totals.peak_memory_mb ?? 'unknown')} MB; relative work: ${escapeHtml(totals.work_units ?? 'unknown')}. ${costBlocked ? 'The resource limit prevents execution.' : ''}</p>`,
    `<button type="button" id="confirm-retry"${action.status !== 'prepared' || costBlocked ? ' disabled' : ''}>${needsCostApproval ? 'Approve Cost and Retry' : 'Confirm Retry'}</button>`,
  ].join('');
  document.getElementById('confirm-retry').addEventListener('click', confirmPendingRetry);
}

async function confirmPendingRetry() {
  const action = plannerState.pendingAction;
  const identity = plannerState.savedStudy;
  if (!action || action.status !== 'prepared' || !identity) return;
  document.getElementById('confirm-retry').disabled = true;
  try {
    const response = await postJson('/api/study-retry-start', {
      ...savedStudyRequest(), action_id: action.action_id,
      approve_cost: !(action.cost_estimate || {}).can_execute,
    });
    if (identity !== plannerState.savedStudy) return;
    plannerState.pendingAction = null;
    closeRetryEditor();
    setStatus('run-status', response.started ? 'Selected retry started.' : (response.message || 'Retry already submitted.'), 'ok');
    await refreshSavedStudy();
  } catch (error) {
    if (identity !== plannerState.savedStudy) return;
    // A failed or uncertain start needs a fresh server preview, never a full Run fallback.
    action.status = 'needs_refresh';
    renderRetryPreview();
    setStatus('run-status', `${error.message} Refresh the Study and prepare a new retry if needed.`, 'error');
  }
}

document.getElementById('case-table').addEventListener('click', event => {
  const button = event.target.closest('[data-retry-case]');
  if (button) openRetryEditor([button.getAttribute('data-retry-case')]);
});
document.getElementById('preview-retry').addEventListener('click', prepareSelectedRetry);
document.getElementById('close-retry').addEventListener('click', closeRetryEditor);
