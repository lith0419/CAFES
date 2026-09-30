const INITIAL_STUDY_SPEC = blankStudySpecForSystem('molecular');
const savedStudyLocation = takeSavedStudyLocation();
const executionTargetsReady = initializeExecutionTargetSelect();
document.getElementById('study-spec').value = prettyJson(INITIAL_STUDY_SPEC);
syncStudySystemFromSpec(INITIAL_STUDY_SPEC);
document.getElementById('study-mode').value = 'adaptive';
document.getElementById('adaptive-initial-scan-strategy').value = 'auto';
document.getElementById('load-example').addEventListener('click', loadExampleForSelectedSystem);
document.getElementById('draft-with-llm').addEventListener('click', draftWithLlm);
document.getElementById('build-plan').addEventListener('click', buildPlan);
document.getElementById('run-study').addEventListener('click', runStudy);
document.getElementById('stop-saved-study').addEventListener('click', stopSavedStudy);
document.getElementById('review-gate').addEventListener('click', handleReviewGateAction);
document.getElementById('analyze-study-results').addEventListener('click', analyzeStudyResults);
document.getElementById('suggest-plots').addEventListener('click', suggestPostprocessPlots);
document.getElementById('generate-default-plots').addEventListener('click', () => runPostprocessing([], { inferDefault: true }));
document.getElementById('generate-selected-plots').addEventListener('click', () => runPostprocessing());
document.getElementById('generate-hamiltonian-dataset').addEventListener('click', generateHamiltonianDataset);
document.getElementById('collect-hamiltonian-dataset').addEventListener('click', collectHamiltonianDataset);
document.getElementById('select-energy-frames').addEventListener('click', selectEnergyFrames);
document.getElementById('add-custom-plot').addEventListener('click', addCustomPlotSpec);
document.getElementById('custom-plot-tool').addEventListener('input', updateCustomPlotControls);
document.getElementById('task-session-select').addEventListener('change', (event) => switchTaskSession(event.target.value));
document.getElementById('new-task').addEventListener('click', () => {
  createTaskSession({ inheritWorkDir: true });
});
document.getElementById('study-system').addEventListener('change', handleStudySystemChange);
document.getElementById('planner-template').addEventListener('change', handlePlannerTemplateChange);
document.getElementById('study-mode').addEventListener('change', handleStudyModeChange);
document.getElementById('planner-grid-refinement').addEventListener('change', handleGridRefinementChange);
document.getElementById('dataset-seed-file').addEventListener('change', handleDatasetSeedFileChange);
[
  'dataset-id',
  'dataset-name',
  'dataset-split-protocol',
  'dataset-split-seed',
].forEach((id) => {
  document.getElementById(id).addEventListener('input', handleDatasetConfigurationChange);
});
document.getElementById('adaptive-initial-scan-strategy').addEventListener('change', handleStudyModeChange);
document.getElementById('adaptive-active-space-solver').addEventListener('change', () => {
  syncAdaptiveActiveSpaceControls();
  handleStudyModeChange();
});
document.getElementById('adaptive-active-space-localization').addEventListener('change', handleStudyModeChange);
document.getElementById('adaptive-orbital-ordering').addEventListener('change', () => {
  syncAdaptiveActiveSpaceControls();
  handleStudyModeChange();
});
document.getElementById('adaptive-manual-orbital-order').addEventListener('input', handleStudyModeChange);
document.getElementById('top-open-model-builder').addEventListener('click', openModelBuilder);
document.getElementById('open-model-builder').addEventListener('click', openModelBuilder);
document.getElementById('preview-model-hamiltonian').addEventListener('click', previewModelHamiltonianStructure);
[
  'planner-model-solver',
  'planner-dmet-execution-mode',
  'planner-dmet-fragment-definition',
  'planner-dmet-impurity-shape',
  'planner-dmet-impurity-size',
  'planner-dmet-impurity-solver',
  'planner-dmet-ccsd-beta',
  'planner-dmet-block2-preset',
  'planner-dmet-block2-ordering',
  'planner-dmet-reference',
  'planner-dmet-reference-density',
  'planner-dmet-interacting-bath',
  'planner-dmet-max-iterations',
  'planner-dmet-energy-tolerance',
  'planner-dmet-density-tolerance',
].forEach((id) => {
  document.getElementById(id).addEventListener('change', handlePlannerModelSolverControlsChange);
});
document.getElementById('work-dir').addEventListener('input', () => {
  updateEffectiveWorkDir();
  snapshotCurrentTaskSession();
});
document.getElementById('study-spec').addEventListener('input', () => {
  installFullStudyPlan(null);
  currentAdaptivePreview = null;
  currentAnalysisRows = [];
  syncStudySystemFromSpec();
  setStatus('run-status', 'Plan changed. Build before running.');
  renderAdaptiveState(null);
  updateActiveTaskSessionStatus('draft');
  snapshotCurrentTaskSession();
});
window.addEventListener('message', handleBuilderMessage);
updateEffectiveWorkDir();
syncDatasetTemplateControls();
syncAdaptiveActiveSpaceControls();
renderPlanSummary(null);
document.getElementById('run-study').disabled = true;
document.getElementById('analyze-study-results').disabled = true;
document.getElementById('suggest-plots').disabled = true;
document.getElementById('generate-default-plots').disabled = true;
document.getElementById('generate-selected-plots').disabled = true;
document.getElementById('generate-hamiltonian-dataset').disabled = true;
document.getElementById('collect-hamiltonian-dataset').disabled = true;
renderPlannerChat();
resetModelPreview();
resetPostprocessing();
updateCustomPlotControls();
const initialTaskSession = createTaskSession({
  studySpec: INITIAL_STUDY_SPEC,
  snapshotExisting: false,
  silent: true,
  inheritWorkDir: true,
});
document.getElementById('open-saved-study').addEventListener('click', () => openSavedStudy());
document.getElementById('refresh-saved-studies').addEventListener('click', listSavedStudies);
document.getElementById('refresh-saved-study').addEventListener('click', () => refreshSavedStudy());
document.getElementById('collect-saved-study').addEventListener('click', collectSavedStudy);
document.getElementById('saved-study-list').addEventListener('change', (event) => {
  if (event.target.value) openSavedStudy({ studyId: event.target.value,
    workDir: event.target.selectedOptions[0].dataset.workDir });
});
executionTargetsReady.then(() => {
  if (savedStudyLocation && activeTaskSession() === initialTaskSession) {
    return openSavedStudy(savedStudyLocation);
  }
});
