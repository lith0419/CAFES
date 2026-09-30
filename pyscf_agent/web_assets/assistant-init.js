initializeExecutionTargetSelect();
const languageToggle = document.getElementById('lang-toggle');
languageToggle.hidden = true;
languageToggle.disabled = true;
languageToggle.setAttribute('aria-hidden', 'true');
currentLanguage = 'en';
try {
  window.localStorage.removeItem('pyscf-agent-ui-language');
} catch (error) {
}
window.addEventListener('message', handleBuilderMessage);
document.getElementById('task-family').addEventListener('change', () => {
  if (isRestoringTaskSession) {
    syncTaskFamilyControls();
    return;
  }
  const nextTaskFamily = currentTaskFamily();
  const previousSession = activeTaskSession();
  const previousTaskFamily = previousSession && previousSession.formData ? previousSession.formData.taskFamily : 'molecular';
  if (currentTaskHasMeaningfulState()) {
    createTaskSession(nextTaskFamily, { previousTaskFamily });
    return;
  }
  syncTaskFamilyControls();
  invalidatePendingExecution();
  snapshotCurrentTaskSession();
});
document.getElementById('task-session-select').addEventListener('change', (event) => {
  switchTaskSession(event.target.value);
});
document.getElementById('new-task').addEventListener('click', () => {
  createTaskSession(currentTaskFamily(), { inheritWorkDir: true });
});
document.getElementById('open-model-builder').addEventListener('click', openModelHamiltonianBuilder);
document.getElementById('preview-model-hamiltonian').addEventListener('click', previewModelHamiltonianStructure);
document.getElementById('periodic-file').addEventListener('change', handlePeriodicFile);
document.getElementById('preview-periodic-structure').addEventListener('click', previewPeriodicStructure);
document.getElementById('work-dir').addEventListener('input', () => {
  currentRunId = '';
  document.getElementById('model-hamiltonian-input-file').value = '';
  setWorkDirLocked(false);
  syncTaskFamilyControls();
  invalidatePendingExecution();
  resetModelHamiltonianStructurePreview();
  snapshotCurrentTaskSession();
});
document.getElementById('work-dir').addEventListener('change', () => {
  currentRunId = '';
  document.getElementById('model-hamiltonian-input-file').value = '';
  setWorkDirLocked(false);
  syncTaskFamilyControls();
  invalidatePendingExecution();
  resetModelHamiltonianStructurePreview();
  snapshotCurrentTaskSession();
});
document.getElementById('method').addEventListener('change', syncMethodControls);
document.getElementById('model-solver').addEventListener('change', syncModelSolverControls);
document.getElementById('periodic-method').addEventListener('change', syncPeriodicMethodControls);
document.getElementById('periodic-correlation-treatment').addEventListener('change', () => {
  syncPeriodicCorrelationControls();
  invalidatePendingExecution();
  snapshotCurrentTaskSession();
});
document.getElementById('periodic-restricted').addEventListener('change', syncPeriodicCorrelationControls);
document.getElementById('periodic-band-path-mode').addEventListener('change', () => {
  syncPeriodicBandPathControls(true);
});
document.getElementById('periodic-density-fitting').addEventListener('change', syncPeriodicNumericalControls);
document.getElementById('periodic-smearing').addEventListener('change', syncPeriodicNumericalControls);
document.getElementById('model-dmet-fragment-mode').addEventListener('change', () => {
  syncDmetFragmentControls();
  invalidateDmetPreviewValidation();
});
document.getElementById('model-dmet-execution-mode').addEventListener('change', () => {
  syncDmetFragmentControls();
  invalidateDmetPreviewValidation();
});
document.getElementById('model-dmet-impurity-solver').addEventListener('change', () => {
  syncDmetImpuritySolverControls();
  invalidateDmetPreviewValidation();
});
[
  'model-solver',
  'model-dmet-execution-mode',
  'model-dmet-fragment-mode',
  'model-dmet-impurity-shape',
  'model-dmet-impurity-size',
  'model-dmet-impurity-solver',
  'model-dmet-ccsd-beta',
  'model-dmet-block2-preset',
  'model-dmet-block2-ordering',
  'model-dmet-reference',
  'model-dmet-reference-density',
  'model-dmet-interacting-bath',
].forEach((id) => {
  document.getElementById(id).addEventListener('input', invalidateDmetPreviewValidation);
  document.getElementById(id).addEventListener('change', invalidateDmetPreviewValidation);
});
document.getElementById('basis').addEventListener('change', () => {
  orderedAuxbasisOptionsForBasis(document.getElementById('basis').value);
});
[
  'atom',
  'basis',
  'method',
  'xc',
  'job',
  'charge',
  'spin',
  'restricted',
  'request',
  'model-hamiltonian-input-file',
  'model-solver',
  'model-nroots',
  'model-dmet-execution-mode',
  'model-dmet-fragment-mode',
  'model-dmet-impurity-shape',
  'model-dmet-impurity-size',
  'model-dmet-impurity-solver',
  'model-dmet-ccsd-beta',
  'model-dmet-block2-preset',
  'model-dmet-block2-ordering',
  'model-dmet-reference',
  'model-dmet-reference-density',
  'model-dmet-interacting-bath',
  'model-dmet-max-iterations',
  'model-dmet-energy-tolerance',
  'model-dmet-density-tolerance',
  'periodic-format',
  'periodic-structure-text',
  'periodic-basis',
  'periodic-pseudo',
  'periodic-method',
  'periodic-xc',
  'periodic-dmft-localization',
  'periodic-dmft-impurity-solver',
  'periodic-dmft-correlated-orbitals',
  'periodic-dmft-minimal-basis',
  'periodic-dmft-nbath',
  'periodic-dmft-max-iterations',
  'periodic-dmft-convergence',
  'periodic-kmesh-x',
  'periodic-kmesh-y',
  'periodic-kmesh-z',
  'periodic-kpoint-scheme',
  'periodic-band-path-mode',
  'periodic-band-path',
  'periodic-band-special-points',
  'periodic-band-path-npoints',
  'periodic-band-path-reference-distance',
  'periodic-band-path-symprec',
  'periodic-kpoint-shift-x',
  'periodic-kpoint-shift-y',
  'periodic-kpoint-shift-z',
  'periodic-precision',
  'periodic-ke-cutoff',
  'periodic-fft-mesh-x',
  'periodic-fft-mesh-y',
  'periodic-fft-mesh-z',
  'periodic-density-fitting',
  'periodic-df-auxbasis',
  'periodic-exxdiv',
  'periodic-smearing',
  'periodic-smearing-sigma',
  'periodic-smearing-fix-spin',
  'periodic-charge',
  'periodic-spin',
  'periodic-restricted',
  'density-fitting-auxbasis',
  'orbital-processing-enabled',
  'localization-method',
  'localization-scope',
  'orbital-ordering',
  'manual-orbital-order',
  'state-target-nroots',
  'state-average-weights',
  'use-natural-orbitals',
  'active-space-enabled',
  'active-space-method',
  'avas-targets',
  'avas-threshold',
  'ncas',
  'nelecas',
  'active-orbitals',
  'active-space-approved',
  'sc-nevpt2-enabled',
  'sc-nevpt2-root',
  'sc-nevpt2-density-fit',
].forEach((id) => {
  document.getElementById(id).addEventListener('input', () => {
    invalidatePendingExecution();
    updateActiveSpaceSummary();
    snapshotCurrentTaskSession();
  });
  document.getElementById(id).addEventListener('change', () => {
    invalidatePendingExecution();
    updateActiveSpaceSummary();
    snapshotCurrentTaskSession();
  });
});
document.getElementById('atom').addEventListener('input', () => {
  clearMolecularStructureValidation();
  updateMolecularStructurePreview();
});
document.getElementById('atom').addEventListener('change', () => {
  clearMolecularStructureValidation();
  updateMolecularStructurePreview();
});
document.getElementById('periodic-structure-text').addEventListener('input', () => resetPeriodicStructurePreview());
document.getElementById('periodic-format').addEventListener('change', () => resetPeriodicStructurePreview());
document.getElementById('sc-nevpt2-enabled').addEventListener('change', syncPostCasControls);
document.getElementById('restricted').addEventListener('change', syncPostCasControls);
document.getElementById('active-space-method').addEventListener('change', syncActiveSpaceMethodControls);
document.getElementById('localization-method').addEventListener('change', syncOrbitalProcessingControls);
document.getElementById('localization-scope').addEventListener('change', syncOrbitalProcessingControls);
document.getElementById('orbital-ordering').addEventListener('change', syncOrbitalProcessingControls);
document.getElementById('active-space-solver').addEventListener('change', () => {
  syncOrbitalProcessingControls();
  syncPostCasControls();
  invalidatePendingExecution();
  snapshotCurrentTaskSession();
});
document.querySelectorAll('input[name="outputs"]').forEach((item) => {
  item.addEventListener('change', () => {
    invalidatePendingExecution();
    snapshotCurrentTaskSession();
  });
});
document.getElementById('run').addEventListener('click', runAgent);
document.getElementById('screen-active-space').addEventListener('click', screenActiveSpace);
document.getElementById('analyze-result').addEventListener('click', analyzeCurrentResult);
document.getElementById('approve-structure').addEventListener('click', approvePendingApproval);
document.getElementById('cancel-structure').addEventListener('click', cancelPendingApproval);
document.getElementById('confirm-compute').addEventListener('click', confirmCompute);
document.getElementById('refresh-status').addEventListener('click', refreshActiveRunStatus);
document.getElementById('clear').addEventListener('click', async () => {
  if (calculationInProgress) {
    await stopActiveCalculation();
    return;
  }
  const reusableWorkDir = builderWorkDirValue();
  conversationHistory = [];
  pendingStructuredRequest = {};
  currentTaskLifecycle = null;
  currentRunId = '';
  clearActiveRunContext();
  document.getElementById('work-dir').value = reusableWorkDir === DEFAULT_WORK_DIR ? '' : reusableWorkDir;
  document.getElementById('model-hamiltonian-input-file').value = '';
  lastModelHamiltonianPreview = null;
  setWorkDirLocked(false);
  clearApprovalPanel();
  clearComputeConfirmPanel();
  resetPrepareCache();
  syncTaskFamilyControls();
  clearPanels();
  resetModelHamiltonianStructurePreview();
  resetPeriodicStructurePreview();
  updateActiveTaskSessionStatus('draft');
  snapshotCurrentTaskSession();
});
syncTaskFamilyControls();
syncActiveSpaceMethodControls();
syncActiveSpaceSolverControls();
syncOrbitalProcessingControls();
syncMethodControls();
syncPeriodicMethodControls();
syncPeriodicCorrelationControls();
syncPeriodicNumericalControls();
syncPeriodicBandPathControls();
updateStaticText();
orderedAuxbasisOptionsForBasis(document.getElementById('basis').value);
clearPanels();
createTaskSession(currentTaskFamily(), {
  formData: collectFormSnapshot(),
  inheritWorkDir: true,
  snapshotExisting: false,
  silent: true,
});
updateActiveSpaceSummary();
syncPostCasControls();
