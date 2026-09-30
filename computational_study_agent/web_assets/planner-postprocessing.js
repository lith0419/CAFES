function renderTable(targetId, rows, options = {}) {
  const target = document.getElementById(targetId);
  if (!rows || !rows.length) {
    target.className = 'empty';
    target.innerHTML = 'No rows to display.';
    return;
  }
  const columns = [];
  rows.forEach((row) => {
    Object.keys(row).forEach((key) => {
      if (!columns.includes(key)) {
        columns.push(key);
      }
    });
  });
  target.className = `table-wrap ${options.className || ''}`.trim();
  target.innerHTML = AgentUI.tableMarkup(
    columns.map(postprocessColumnLabel), rows.map(row => columns.map(column => row[column] ?? '')));
  if (targetId === 'case-table' && typeof decorateCaseRetryTable === 'function') decorateCaseRetryTable(target, rows);
}

const POSTPROCESS_CONTEXTS = new WeakMap();

function postprocessContext() {
  return (currentReport && POSTPROCESS_CONTEXTS.get(currentReport)) || {};
}

async function refreshPostprocessContext() {
  if (!currentReport) return;
  const report = currentReport;
  const pending = {};
  POSTPROCESS_CONTEXTS.set(report, pending);
  updateCustomPlotControls();
  document.getElementById('suggest-plots').disabled = true;
  document.getElementById('generate-default-plots').disabled = true;
  try {
    const payload = await postJson('/api/study-postprocess-suggestions', {
      report: postprocessReportPayload(), include_suggestions: false,
    });
    if (currentReport !== report || POSTPROCESS_CONTEXTS.get(report) !== pending) return;
    POSTPROCESS_CONTEXTS.set(report, payload.context);
    const hasRows = postprocessRows().length > 0;
    document.getElementById('suggest-plots').disabled = !hasRows;
    document.getElementById('generate-default-plots').disabled = !hasRows;
    updateCustomPlotControls();
    if (!currentPlotSpecs.length) {
      document.getElementById('plot-spec-table').textContent = hasRows ? 'Suggest plots or generate defaults.' : postprocessingUnavailableMessage();
    }
    const excluded = payload.context.excluded_rows || [];
    setStatus('postprocess-status', excluded.length ? `${excluded.length} rows excluded from plotting. ${excluded.map(row => `${row.case_id || 'Row'}: ${row.reason.replace(/_/g, ' ')}`).join('; ')}` : '');
  } catch (error) {
    if (currentReport !== report || POSTPROCESS_CONTEXTS.get(report) !== pending) return;
    setStatus('postprocess-status', errorDetailMessage(error.payload || {}, error.message), 'error');
  }
}

function analysisRows() {
  const reportRows = currentReport && Array.isArray(currentReport.comparison_table)
    ? currentReport.comparison_table
    : [];
  const derivedRows = Array.isArray(currentAnalysisRows) ? currentAnalysisRows : [];
  if (!reportRows.length) {
    return derivedRows;
  }
  if (!derivedRows.length) {
    return reportRows;
  }
  const derivedByCaseId = new Map(
    derivedRows
      .filter((row) => row && row.case_id)
      .map((row) => [row.case_id, row]),
  );
  const mergedCaseIds = new Set();
  const rows = reportRows.map((row) => {
    const caseId = row && row.case_id;
    const derived = caseId ? derivedByCaseId.get(caseId) : null;
    if (caseId) {
      mergedCaseIds.add(caseId);
    }
    return derived ? { ...derived, ...row } : row;
  });
  derivedRows.forEach((row) => {
    if (row && (!row.case_id || !mergedCaseIds.has(row.case_id))) {
      rows.push(row);
    }
  });
  return rows;
}

function postprocessRows() {
  return postprocessContext().rows || [];
}

function postprocessingFailureReason() {
  const cases = currentReport && Array.isArray(currentReport.cases) ? currentReport.cases : [];
  for (const item of cases) {
    const taskReport = item && item.task_report;
    const messages = taskReport && Array.isArray(taskReport.messages) ? taskReport.messages : [];
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const message = messages[index];
      if (message && message.role === 'assistant' && typeof message.content === 'string' && message.content.trim()) {
        return message.content.trim();
      }
    }
  }
  return '';
}

function postprocessingUnavailableMessage() {
  const rows = analysisRows();
  if (!rows.length) {
    return 'No calculation outputs are available for postprocessing.';
  }
  const counts = {};
  rows.forEach((row) => {
    const status = String((row && row.status) || 'unavailable').trim().toLowerCase() || 'unavailable';
    counts[status] = (counts[status] || 0) + 1;
  });
  const countText = Object.entries(counts)
    .map(([status, count]) => `${count} ${status.replace(/_/g, ' ')}`)
    .join(', ');
  const reason = postprocessingFailureReason();
  return `No successful calculation outputs are available for postprocessing (${countText}).${reason ? ` ${reason}` : ''}`;
}

function postprocessReportPayload() {
  const payload = deepCopy(currentReport || {});
  payload.comparison_table = deepCopy(analysisRows());
  return payload;
}

function comparisonColumns() {
  return postprocessContext().columns || [];
}

function numericComparisonColumns() {
  return postprocessContext().numeric_columns || [];
}

function distinctColumnValues(column) {
  const rows = postprocessRows();
  const values = [];
  rows.forEach((row) => {
    if (!row || typeof row !== 'object' || !(column in row)) {
      return;
    }
    const value = row[column];
    if (value === null || value === undefined || value === '') {
      return;
    }
    const key = typeof value === 'object' ? JSON.stringify(value) : String(value);
    if (!values.includes(key)) {
      values.push(key);
    }
  });
  return values;
}

function columnVaries(column) {
  return distinctColumnValues(column).length > 1;
}

function caseVariableColumns() {
  return postprocessContext().case_variable_columns || [];
}

const POSTPROCESS_COLUMN_LABELS = {
  final_energy: 'Energy',
  mean_double_occupancy: 'Mean double occupancy',
  nearest_neighbor_spin_correlation: 'Nearest-neighbor spin correlation',
  nearest_neighbor_charge_correlation: 'Nearest-neighbor charge correlation',
};

function postprocessColumnLabel(value) {
  const gridLabels = {grid_origin: 'Point source', grid_round: 'Refinement round', grid_axis: 'Refined parameter', energy_per_site: 'Energy per site'};
  return gridLabels[value] || POSTPROCESS_COLUMN_LABELS[value] || value || 'None';
}

function setSelectOptions(selectId, values, { includeNone = false, preferred = '' } = {}) {
  const select = document.getElementById(selectId);
  const items = includeNone ? [''].concat(values) : values.slice();
  select.innerHTML = items.map((value) => (
    `<option value="${escapeHtml(value)}">${escapeHtml(postprocessColumnLabel(value))}</option>`
  )).join('');
  if (preferred && items.includes(preferred)) {
    select.value = preferred;
  } else if (items.length) {
    select.value = items[0];
  }
  select.disabled = !values.length;
}

function updateCustomPlotControls() {
  const numericColumns = numericComparisonColumns();
  const allColumns = comparisonColumns();
  const variableColumns = caseVariableColumns();
  const tool = document.getElementById('custom-plot-tool').value || 'line_plot';
  const isHeatmap = tool === 'heatmap';
  const preferredX = variableColumns.find((column) => numericColumns.includes(column) && columnVaries(column))
    || numericColumns.find((column) => columnVaries(column))
    || numericColumns[0]
    || '';
  const preferredY = (isHeatmap
    ? variableColumns.find((column) => column !== preferredX && numericColumns.includes(column) && columnVaries(column))
    : '')
    || numericColumns.find((column) => column !== preferredX && !variableColumns.includes(column))
    || numericColumns.find((column) => column !== preferredX)
    || '';
  const preferredColor = numericColumns.find((column) => column !== preferredX && column !== preferredY && !variableColumns.includes(column))
    || numericColumns.find((column) => column !== preferredX && column !== preferredY)
    || '';
  const preferredGroup = variableColumns.find((column) => column !== preferredX && allColumns.includes(column) && columnVaries(column)) || '';
  setSelectOptions('custom-plot-x', numericColumns, { preferred: preferredX });
  setSelectOptions('custom-plot-y', numericColumns, { preferred: preferredY });
  setSelectOptions('custom-plot-color', numericColumns, { preferred: preferredColor });
  setSelectOptions('custom-plot-group', allColumns, { includeNone: true, preferred: preferredGroup });
  document.getElementById('custom-plot-color').disabled = !isHeatmap || !numericColumns.length;
  document.getElementById('custom-plot-group').disabled = isHeatmap || !allColumns.length;
  const enabled = Boolean(currentReport && numericColumns.length >= (isHeatmap ? 3 : 2));
  ['custom-plot-tool', 'custom-plot-x', 'custom-plot-y', 'custom-plot-output', 'add-custom-plot'].forEach((id) => {
    document.getElementById(id).disabled = !enabled;
  });
  if (!enabled) {
    document.getElementById('custom-plot-color').disabled = true;
    document.getElementById('custom-plot-group').disabled = true;
  }
}

function resetPostprocessing(message = 'No plot specs yet.') {
  const hasSuccessfulRows = postprocessRows().length > 0;
  currentPlotSpecs = [];
  currentPostprocessArtifacts = [];
  document.getElementById('plot-spec-table').className = 'empty';
  document.getElementById('plot-spec-table').innerHTML = escapeHtml(message);
  document.getElementById('postprocess-artifacts').className = 'empty';
  document.getElementById('postprocess-artifacts').innerHTML = 'No postprocessing artifacts yet.';
  document.getElementById('generate-selected-plots').disabled = true;
  document.getElementById('suggest-plots').disabled = !currentReport || !hasSuccessfulRows;
  document.getElementById('generate-default-plots').disabled = !currentReport || !hasSuccessfulRows;
  syncHamiltonianDatasetActionControls();
  updateCustomPlotControls();
  setStatus('postprocess-status', '');
}

function hamiltonianDatasetReady() {
  const datasetManifest = currentReport && currentReport.dataset_manifest;
  return Boolean(
    datasetManifest
    && typeof datasetManifest === 'object'
    && Number(datasetManifest.accepted_structure_count || 0) > 0
  );
}

function hamiltonianDatasetGenerated() {
  const generation = currentReport && currentReport.dataset_generation;
  if (generation && generation.status === 'succeeded') return true;
  const datasetProgress = currentExecution
    && currentExecution.execution
    && currentExecution.execution.dataset;
  return Boolean(datasetProgress && datasetProgress.generated);
}

function syncHamiltonianDatasetActionControls() {
  const generateButton = document.getElementById('generate-hamiltonian-dataset');
  const collectButton = document.getElementById('collect-hamiltonian-dataset');
  const ready = hamiltonianDatasetReady();
  const generated = ready && hamiltonianDatasetGenerated();
  generateButton.classList.toggle('hidden', !ready);
  generateButton.disabled = !ready || generated;
  collectButton.classList.toggle('hidden', !generated);
  collectButton.disabled = !generated;
  document.getElementById('energy-selection-controls').classList.toggle('hidden', !ready);
  document.getElementById('select-energy-frames').disabled = !ready;
}

function renderPlotSpecs(specs) {
  currentPlotSpecs = Array.isArray(specs) ? specs : [];
  const target = document.getElementById('plot-spec-table');
  if (!currentPlotSpecs.length) {
    target.className = 'empty';
    target.innerHTML = 'No plot suggestions.';
    document.getElementById('generate-selected-plots').disabled = true;
    return;
  }
  const rows = currentPlotSpecs.map((spec, index) => ({
    select: `<input type="checkbox" class="plot-spec-select" data-index="${index}" checked>`,
    tool: escapeHtml(spec.tool || 'line_plot'),
    x: escapeHtml(spec.x || ''),
    y: escapeHtml(postprocessColumnLabel(spec.y || '')),
    color: escapeHtml(postprocessColumnLabel(spec.color || '')),
    group: escapeHtml(postprocessColumnLabel(spec.group || '')),
    output: escapeHtml(spec.output || ''),
  }));
  target.className = 'table-wrap plot-spec-table';
  target.innerHTML = [
    '<table>',
    '<thead><tr><th></th><th>tool</th><th>x</th><th>y</th><th>color</th><th>group</th><th>output</th></tr></thead><tbody>',
    rows.map((row) => (
      `<tr><td>${row.select}</td><td>${row.tool}</td><td>${row.x}</td><td>${row.y}</td><td>${row.color}</td><td>${row.group}</td><td>${row.output}</td></tr>`
    )).join(''),
    '</tbody></table>',
  ].join('');
  document.getElementById('generate-selected-plots').disabled = false;
}

function addCustomPlotSpec() {
  if (!currentReport) {
    setStatus('postprocess-status', 'Run a plan first.', 'error');
    return;
  }
  const xColumn = document.getElementById('custom-plot-x').value;
  const yColumn = document.getElementById('custom-plot-y').value;
  if (!xColumn || !yColumn) {
    setStatus('postprocess-status', 'Choose x and y.', 'error');
    return;
  }
  if (xColumn === yColumn) {
    setStatus('postprocess-status', 'Choose different x and y.', 'error');
    return;
  }
  const tool = document.getElementById('custom-plot-tool').value || 'line_plot';
  const colorColumn = document.getElementById('custom-plot-color').value;
  if (tool === 'heatmap' && (!colorColumn || colorColumn === xColumn || colorColumn === yColumn)) {
    setStatus('postprocess-status', 'Choose a distinct color field.', 'error');
    return;
  }
  const output = document.getElementById('custom-plot-output').value.trim()
    || (tool === 'heatmap' ? `${colorColumn}-heatmap-${xColumn}-${yColumn}` : `${yColumn}-vs-${xColumn}`);
  const nextSpec = {
    tool,
    x: xColumn,
    y: yColumn,
    color: tool === 'heatmap' ? colorColumn : null,
    group: tool === 'heatmap' ? null : document.getElementById('custom-plot-group').value || null,
    output,
  };
  renderPlotSpecs(currentPlotSpecs.concat([nextSpec]));
  setStatus('postprocess-status', 'Plot spec added.', 'ok');
}

function selectedPlotSpecs() {
  const selected = [];
  document.querySelectorAll('.plot-spec-select').forEach((item) => {
    if (!item.checked) {
      return;
    }
    const index = Number.parseInt(item.dataset.index || '-1', 10);
    if (Number.isInteger(index) && currentPlotSpecs[index]) {
      selected.push(currentPlotSpecs[index]);
    }
  });
  return selected;
}

function renderPostprocessArtifacts(artifacts) {
  artifactPreviewUrls.forEach((url) => URL.revokeObjectURL(url));
  artifactPreviewUrls = [];
  currentPostprocessArtifacts = (Array.isArray(artifacts) ? artifacts : [])
    .filter((artifact) => artifact && artifact.kind !== 'postprocess-plot-spec');
  const target = document.getElementById('postprocess-artifacts');
  if (!currentPostprocessArtifacts.length) {
    target.className = 'empty';
    target.innerHTML = 'No postprocessing artifacts.';
    return;
  }
  target.className = 'artifact-list';
  target.innerHTML = currentPostprocessArtifacts.map((artifact, index) => (
    `<div class="artifact-item" data-artifact-index="${index}"><span class="artifact-kind">${escapeHtml(artifact.kind || 'artifact')}</span><div class="artifact-path">${escapeHtml(artifact.path || '')}</div><div class="artifact-preview-slot"></div></div>`
  )).join('');
  loadArtifactPreviews();
}

function isPreviewablePlotArtifact(artifact) {
  return artifact
    && artifact.kind === 'postprocess-plot'
    && artifact.mime_type === 'image/png'
    && typeof artifact.path === 'string'
    && artifact.path.toLowerCase().endsWith('.png');
}

async function loadArtifactPreviews() {
  if (!currentReport) {
    return;
  }
  const items = Array.from(document.querySelectorAll('#postprocess-artifacts .artifact-item'));
  await Promise.all(items.map(async (item) => {
    const index = Number.parseInt(item.dataset.artifactIndex || '-1', 10);
    const artifact = currentPostprocessArtifacts[index];
    const slot = item.querySelector('.artifact-preview-slot');
    if (!slot || !isPreviewablePlotArtifact(artifact)) {
      return;
    }
    slot.innerHTML = '<div class="hint">Loading...</div>';
    try {
      const response = await fetch('/api/study-artifact', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          report: currentReport,
          path: artifact.path,
        }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || `HTTP ${response.status}`);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      artifactPreviewUrls.push(url);
      slot.innerHTML = `<div class="artifact-preview"><img src="${url}" alt="${escapeHtml(artifact.description || artifact.path || 'Plot preview')}"></div>`;
    } catch (error) {
      slot.innerHTML = `<div class="artifact-preview-error">${escapeHtml(error.message || 'Preview failed')}</div>`;
    }
  }));
}

async function suggestPostprocessPlots() {
  if (!currentReport) {
    setStatus('postprocess-status', 'Run a plan first.', 'error');
    return;
  }
  setStatus('postprocess-status', 'Suggesting...');
  document.getElementById('suggest-plots').disabled = true;
  try {
    const payload = await postJson('/api/study-postprocess-suggestions', {
      report: postprocessReportPayload(),
    });
    renderPlotSpecs(payload.plot_specs || []);
    setStatus('postprocess-status', (payload.plot_specs || []).length ? 'Suggestions ready.' : 'No suggestions.', 'ok');
  } catch (error) {
    const message = errorDetailMessage(error.payload || {}, error.message);
    setStatus('postprocess-status', message, 'error');
  } finally {
    document.getElementById('suggest-plots').disabled = false;
  }
}

async function runPostprocessing(specs, { inferDefault = false } = {}) {
  if (!currentReport) {
    setStatus('postprocess-status', 'Run a plan first.', 'error');
    return;
  }
  const selectedSpecs = Array.isArray(specs) ? specs : selectedPlotSpecs();
  if (!inferDefault && !selectedSpecs.length) {
    setStatus('postprocess-status', 'Select a plot spec.', 'error');
    return;
  }
  setStatus('postprocess-status', 'Generating plots...');
  document.getElementById('generate-selected-plots').disabled = true;
  document.getElementById('generate-default-plots').disabled = true;
  try {
    const requestPayload = {
      report: postprocessReportPayload(),
    };
    if (!inferDefault) {
      requestPayload.plot_specs = selectedSpecs;
    }
    const payload = await postJson('/api/study-postprocess', requestPayload);
    const result = payload.postprocessing || {};
    renderPostprocessArtifacts(result.artifacts || []);
    if (Array.isArray(result.plot_specs)) {
      renderPlotSpecs(result.plot_specs);
    }
    if (!Array.isArray(currentReport.artifacts)) {
      currentReport.artifacts = [];
    }
    currentReport.artifacts = currentReport.artifacts.concat(result.artifacts || []);
    const exclusions = (result.plots || []).flatMap(plot => (plot.excluded_rows || []).map(row => `${plot.spec.output}: ${row.case_id || 'Row'} — ${row.reason.replace(/_/g, ' ')}`));
    setStatus('postprocess-status', `Postprocessing ${result.status || 'completed'}.${exclusions.length ? ` Excluded: ${exclusions.join('; ')}` : ''}`, result.status === 'succeeded' ? 'ok' : '');
    snapshotCurrentTaskSession();
  } catch (error) {
    const message = errorDetailMessage(error.payload || {}, error.message);
    setStatus('postprocess-status', message, 'error');
  } finally {
    document.getElementById('generate-selected-plots').disabled = !currentPlotSpecs.length;
    document.getElementById('generate-default-plots').disabled = !postprocessRows().length;
  }
}

async function selectEnergyFrames() {
  if (!hamiltonianDatasetReady()) return;
  for (const id of ['energy-selection-count', 'energy-selection-start', 'energy-selection-end']) {
    if (!document.getElementById(id).reportValidity()) return;
  }
  const report = currentReport;
  const button = document.getElementById('select-energy-frames');
  const end = document.getElementById('energy-selection-end').value.trim();
  button.disabled = true;
  setStatus('postprocess-status', 'Selecting geometries by potential-energy quantiles...');
  try {
    const payload = await postJson('/api/study-postprocess', {
      report: {
        study_id: report.study_id, work_dir: report.work_dir,
        system_type: report.system_type, dataset_manifest: deepCopy(report.dataset_manifest),
      },
      actions: [{
        action: 'select_energy_stratified_frames',
        count_per_molecule: Number(document.getElementById('energy-selection-count').value),
        time_start_fs: Number(document.getElementById('energy-selection-start').value),
        time_end_fs: end === '' ? null : Number(end),
      }],
      execution_target: selectedExecutionTarget(),
    });
    if (currentReport !== report) return;
    const result = payload.postprocessing;
    const selection = result.actions[0];
    renderPostprocessArtifacts(result.artifacts);
    report.artifacts = (report.artifacts || []).concat(result.artifacts);
    report.frame_selection = deepCopy(selection);
    setStatus('postprocess-status', `Selected ${selection.sample_count} geometries from ${selection.molecule_count} molecules. ${selection.unselected_molecule_count} molecules have no complete selection.`, result.status === 'succeeded' ? 'ok' : '');
    snapshotCurrentTaskSession();
  } catch (error) {
    if (currentReport === report) setStatus('postprocess-status', errorDetailMessage(error.payload || {}, error.message), 'error');
  } finally {
    syncHamiltonianDatasetActionControls();
  }
}

async function generateHamiltonianDataset() {
  if (!currentReport || !currentReport.dataset_manifest) {
    setStatus('postprocess-status', 'Run and finalize a Hamiltonian dataset study first.', 'error');
    return;
  }
  const button = document.getElementById('generate-hamiltonian-dataset');
  button.disabled = true;
  setStatus(
    'postprocess-status',
    'Generating and validating the complete dataset on the selected execution target; trajectory arrays will not be downloaded...',
  );
  try {
    const payload = await postJson('/api/study-postprocess', {
      report: deepCopy(currentReport),
      actions: ['generate_hamiltonian_dataset'],
      execution_target: selectedExecutionTarget(),
    });
    const result = payload.postprocessing || {};
    const action = Array.isArray(result.actions) ? result.actions[0] : null;
    renderPostprocessArtifacts(result.artifacts || []);
    if (!Array.isArray(currentReport.artifacts)) {
      currentReport.artifacts = [];
    }
    currentReport.artifacts = currentReport.artifacts.concat(result.artifacts || []);
    if (action) {
      currentReport.dataset_generation = deepCopy(action);
    }
    const location = action && action.location === 'remote_executor' ? 'server' : 'local machine';
    setStatus(
      'postprocess-status',
      action
        ? `Dataset generated on the ${location}: ${Number(action.sample_count || 0)} samples in ${Number(action.trajectory_file_count || 0)} trajectory files. Collect is optional.`
        : `Dataset generation ${result.status || 'completed'}.`,
      result.status === 'succeeded' ? 'ok' : '',
    );
    snapshotCurrentTaskSession();
  } catch (error) {
    const message = errorDetailMessage(error.payload || {}, error.message);
    setStatus('postprocess-status', message, 'error');
  } finally {
    syncHamiltonianDatasetActionControls();
  }
}

async function collectHamiltonianDataset() {
  if (!currentReport || !currentReport.dataset_manifest || !hamiltonianDatasetGenerated()) {
    setStatus('postprocess-status', 'Generate the dataset on its execution target before collecting it.', 'error');
    return;
  }
  const button = document.getElementById('collect-hamiltonian-dataset');
  button.disabled = true;
  setStatus('postprocess-status', 'Downloading the generated dataset to this study directory...');
  try {
    const payload = await postJson('/api/study-postprocess', {
      report: deepCopy(currentReport),
      actions: ['collect_hamiltonian_dataset'],
      execution_target: selectedExecutionTarget(),
    });
    const result = payload.postprocessing || {};
    const action = Array.isArray(result.actions) ? result.actions[0] : null;
    renderPostprocessArtifacts(result.artifacts || []);
    if (!Array.isArray(currentReport.artifacts)) {
      currentReport.artifacts = [];
    }
    currentReport.artifacts = currentReport.artifacts.concat(result.artifacts || []);
    if (action) {
      currentReport.dataset_collection = deepCopy(action);
    }
    setStatus(
      'postprocess-status',
      action
        ? `Dataset collected locally: ${Number(action.sample_count || 0)} samples in ${Number(action.trajectory_file_count || 0)} trajectory files.`
        : `Dataset collection ${result.status || 'completed'}.`,
      result.status === 'succeeded' ? 'ok' : '',
    );
    snapshotCurrentTaskSession();
  } catch (error) {
    const message = errorDetailMessage(error.payload || {}, error.message);
    setStatus('postprocess-status', message, 'error');
  } finally {
    syncHamiltonianDatasetActionControls();
  }
}
