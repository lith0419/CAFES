from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import unittest

from tests.computational_study_agent.support import StudyAgentTestCase, planner_ui_source
from computational_study_agent.web_ui import build_study_index_html


class PlannerWebUiTests(StudyAgentTestCase):
    @unittest.skipUnless(shutil.which('node'), 'node is not installed')
    def test_recovery_guidance_uses_report_errors_and_backend_actions_for_static_studies(self):
        assets = Path(__file__).resolve().parents[2] / 'computational_study_agent' / 'web_assets'
        source = '\n'.join((assets / name).read_text() for name in (
            'planner-core.js', 'planner-adaptive-review.js', 'planner-adaptive-decisions.js',
        ))
        script = source + """
          const assert = require('node:assert/strict');
          const failed = {case_id: 'case-0003', status: 'failed', solver: 'dmet',
            next_step: 'Increase max_cycle and promote to CASSCF'};
          const unconverged = {case_id: 'case-0012', status: 'unconverged', solver: 'dmet'};
          const report = {
            comparison_table: [failed, unconverged],
            cases: [{case_id: 'case-0003', task_report: {errors: [
              {message: 'PySCF execution failed: Internal Error.'},
            ]}}],
          };
          const gate = {kind: 'adaptive_unresolved_refined_cases', case_ids: ['case-0003'],
            actions: [{id: 'show_case_guidance', label: 'Show Failure Details'},
              {id: 'acknowledge', label: 'Acknowledge'}]};
          const message = reviewPlanTableGuidance(gate, report);
          assert(message.includes('Internal Error.'));
          assert(message.includes('provider log'));
          assert(!message.includes('CASSCF'));
          assert(!message.includes('Increase max_cycle'));
          assert(!message.includes('case-0012'));
          gate.actions.unshift({id: 'retry_dmet_without_impurity_diis', case_ids: ['case-0003'],
            label: 'Prepare Retry without Impurity SCF DIIS',
            description: 'Retry this case with solver.options.impurity_scf_diis=false (previously true).'});
          const normalized = reviewActionsFromWorkflow({allowed_actions: gate.actions});
          assert.equal(normalized[0].id, 'retry_dmet_without_impurity_diis');
          assert(normalized[0].description.includes('previously true'));
          const retryMessage = reviewPlanTableGuidance({...gate, actions: normalized}, report);
          assert(retryMessage.includes('Options: Prepare Retry without Impurity SCF DIIS'));
          assert(retryMessage.includes('impurity_scf_diis=false'));
          assert(!retryMessage.includes('increasing iterations does not address'));
          gate.case_ids = ['case-0012'];
          gate.actions = [{id: 'increase_dmet_iterations',
            label: 'Increase DMET Outer Iterations (max_iterations)', case_ids: ['case-0012']}];
          const dmetMessage = reviewPlanTableGuidance(gate, report);
          assert(dmetMessage.includes('Options: Increase DMET Outer Iterations'));
          assert(dmetMessage.includes('solver.options.max_iterations'));
          assert(!dmetMessage.includes('CASSCF'));
          assert.equal(adaptiveUnresolvedSuggestion({status: 'failed', solver: 'block2_dmrg',
            dmrg_recovery_summary: 'Select a larger memory profile.'}), 'Select a larger memory profile.');
        """
        subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True)

    def test_web_ui_contains_study_controls(self):
        html = planner_ui_source(build_study_index_html())

        required_fragments = [
            'Computational Study Planner',
            'Design multi-task studies',
            '<span>Work Directory</span>',
            'Leave empty to use runs/ under the current directory',
            'id="execution-target"',
            'id="resource-profile"',
            'id="planner-template"',
            'Hamiltonian Dataset',
            'id="hamiltonian-dataset-setup"',
            'id="dataset-seed-file"',
            'B3LYP / def2-SVP',
            'SCF 1e-8 · grid 3 · DIIS 8',
            'function hamiltonianDatasetPlanningPayload()',
            'planner_template: HAMILTONIAN_DATASET_TEMPLATE',
            'seed_geometries: deepCopy(currentDatasetSeedGeometries)',
            'function renderDatasetManifestSummary(manifest)',
            '/assets/execution-targets.js',
            'execution_target: selectedExecutionTarget()',
            'resource_profile: selectedResourceProfile()',
            'function executionWorkDir()',
            'function lockWorkDirectory(displayPath, rootPath = executionWorkDir())',
            '.inline-field input:disabled',
            'textarea, input, select, option { font-weight: 400; }',
            'Task Setup',
            'id="model-hamiltonian-setup"',
            'id="planner-model-solver" aria-label="Model Hamiltonian method"',
            'DMET Configuration',
            '<option value="dmet">libDMET DMET</option>',
            'id="planner-dmet-controls"',
            'id="planner-dmet-impurity-solver"',
            'id="planner-dmet-fragment-definition"',
            '<option value="">Automatic</option>',
            'function plannerDmetOptionsFromControls()',
            'if (executionMode) options.execution_mode = executionMode;',
            'if (fragmentDefinition) options.fragment_definition = fragmentDefinition;',
            'function applyPlannerModelSolverToSpec(spec)',
            'id="study-mode-control"',
            'id="adaptive-initial-scan-control"',
            'function syncStudyModeControls()',
            "modeSelect.value = 'static';",
            'id="molecular-task-setup"',
            'id="molecular-task-summary"',
            'id="molecular-task-structure-preview" class="molecule-preview"',
            'function parseMolecularAtomString(atomText)',
            'function renderMolecularStructurePreview(targetOrId, atomText, options = {})',
            "renderMolecularStructurePreview('molecular-task-structure-preview'",
            'function renderMolecularTaskSetup(spec)',
            'function firstPreviewCaseVariables(spec)',
            'function resolvePreviewTemplateValue(value, variables)',
            'function molecularPreviewTask(spec, task)',
            'Build Plan to resolve the scan coordinates for this preview.',
            'function updateTaskSetupPanel(spec = null)',
            'Planner Conversation',
            'Plan Review',
            'Run Plan',
            'id="review-gate" class="review-gate"',
            'function renderReviewGate(gate)',
            'function focusPlanReviewForGate(gate)',
            'function handleReviewGateAction(event)',
            'function reviewGateForStudyReport(report)',
            'function adaptiveDecisionEntries(report)',
            'function adaptiveUnresolvedRows(payload)',
            'function adaptiveRecoveryDecisionMap(report)',
            'function adaptiveUnresolvedSuggestion(row, decision = {}, actions = [])',
            'function reviewPlanTableGuidance(gate, report)',
            'function adaptiveWorkflow(payload)',
            'function reviewActionsFromWorkflow(workflow)',
            'function workflowCaseIds(workflow)',
            'function workflowCurrentCaseIds(workflow, fallbackCaseIds = [])',
            'function workflowQueueSuffix(workflow)',
            'function filterRowsForCaseIds(rows, caseIds)',
            "function activeSpaceApprovalRows(report, requestedPlanKind = '')",
            'function studyStateForReviewAction()',
            'function installStudyReviewActionResult(result)',
            'async function applyPlannerReviewAction(action, caseIds, gate)',
            "await postJson('/api/study-review-action'",
            'study_state: studyStateForReviewAction()',
            'function renderReturnedCostReview(error, planKind = \'static\')',
            'function reviewGateCostDetails(estimate, caseIds = [])',
            'Estimated Cost',
            "'approve_cost_estimate'",
            "'expand_active_space'",
            "'approve_active_space'",
            "'increase_dmet_iterations'",
            'Approve 1RDM Restart',
            'function pathRestartArtifactLabel(artifact)',
            'function pathRestartApprovalBranches(item)',
            'Left-anchored branch',
            'Right-anchored branch',
            'generated by ${deferredCaseId} earlier in this branch',
            'Approve endpoint 1RDM validation',
            'Both branches must agree before any result is replaced.',
            'Both branches must agree before any result is replaced.',
            'data-review-action=',
            'approve_active_space',
            'cancel_active_space_review',
            'Approve Active Space',
            'Cancel',
            'Review Required',
            'Active Space Review',
            'Automatic recovery promoted an unresolved refined case to CASSCF/CASCI',
            'Use the existing ActiveSpaceAudit approval flow',
            'approved refined/recovery calculation',
            "isAdaptiveStudy ? 'Adaptive scan' : 'Study'",
            'These cases did not finish successfully after automatic recovery',
            'recovery=',
            'max_cycle=',
            'Plan Review focused. Suggested recovery options were added to Planner Conversation.',
            'Options:',
            'Study System',
            'id="study-system"',
            '<option value="molecular">Molecular</option>',
            'Study Mode',
            'Static Scan',
            'Adaptive Scan',
            'Initial Scan',
            '<option value="auto">Auto</option>',
            '<option value="mp2">MP2</option>',
            '<option value="fci">FCI</option>',
            'id="adaptive-panel"',
            'adaptive-status',
            'function adaptiveOptions()',
            'function renderAdaptiveState(payload)',
            'function adaptiveStatusClass(status)',
            'function adaptiveNextStepMessage(payload)',
            'function summarizeAdaptiveCounts(label',
            'function normalizeSystemType(value)',
            'function currentSystemType()',
            'const DEFAULT_MOLECULAR_STUDY_SPEC',
            'h2-bond-adaptive',
            'function defaultStudySpecForSystem(systemType)',
            'function blankStudySpecForSystem(systemType)',
            'function updatePlannerComposerForSystem(systemType)',
            'function loadExampleForSelectedSystem()',
            'function handleStudySystemChange()',
            "const INITIAL_STUDY_SPEC = blankStudySpecForSystem('molecular');",
            'goal.placeholder = isHamiltonianDatasetPlanner()',
            'Example: Scan a diatomic bond from 0.8x to 3.0x its reference length.',
            "target.innerHTML = '';",
            "document.getElementById('study-system').addEventListener('change'",
            'function planMethodList(plan)',
            'function molecularRisk(diagnostics)',
            'function molecularRiskComponentValue(diagnostics',
            'function activeSpaceLabel(contract)',
            'function reviewCorrelationLevel(decision = {}, result = {}, structured = {})',
            'function firstCellValue(',
            'let currentAnalysisRows = []',
            'function analysisRows()',
            'function postprocessRows()',
            'function postprocessingUnavailableMessage()',
            'function postprocessReportPayload()',
            'const reportRows = currentReport && Array.isArray(currentReport.comparison_table)',
            'const derivedByCaseId = new Map(',
            'function currentStudyMode()',
            'id="adaptive-active-space-solver"',
            'id="adaptive-active-space-solver-control" class="hidden"',
            '<option value="block2_dmrg">block2 DMRG</option>',
            'id="adaptive-block2-orbital-controls"',
            'id="adaptive-active-space-localization"',
            '<option value="boys">Boys localization</option>',
            'id="adaptive-orbital-ordering"',
            '<option value="fiedler">Fiedler</option>',
            'id="adaptive-manual-orbital-order"',
            'active_space_orbital_processing',
            'function syncAdaptiveActiveSpaceControls()',
            'function activeSpaceOrbitalProcessingLabels(request)',
            'solver=${solver === \'block2_dmrg\' ? \'block2 DMRG\'',
            'localization=${localization}',
            'ordering=${ordering}',
            'MolecularCorrelationRisk + ActiveSpaceAudit',
            'initialScanMethods',
            'Step 1: Initial scan',
            'Step 2: Refined calculation',
            'Selection summary',
            'Refined methods',
            'not run',
            'recovered',
            'Adaptive scan completed with independent case results.',
            'function initialScanIssueMessage(payload)',
            'function renderPlanCaseTable(plan',
            'function diagnosticItemValue(diagnostics',
            'function formatEnergyLikeValue(value',
            'function diagnosticSummaryText(row)',
            'function modelFillingFromCase(item)',
            'resultByCaseId',
            'structuredByCaseId',
            'filling',
            'gap',
            'mean_field_gap',
            'average_fractionality',
            'natural_occupation_fractionality',
            'max_double_excitation_amplitude',
            'active_space',
            'final_energy',
            'displayRow.diagnostics',
            'report: postprocessReportPayload()',
            '/api/study-execution-status',
            '/api/study-execution-collect',
            'let currentExecution = null;',
            'function renderExecutionRecovery(execution)',
            'function inspectCurrentExecution()',
            'function collectCurrentExecution()',
            'function executionConversationIntent(message)',
            'Checking or collecting uses the persisted execution receipt.',
            'Completed tasks will not be submitted again.',
            'Send To Planner',
            'id="planner-chat"',
            'id="task-session-select"',
            'id="new-task"',
            'Tasks',
            'New Task',
            'class="hidden-state" aria-hidden="true"',
            'function snapshotCurrentTaskSession()',
            'function restoreTaskSession(session)',
            'function createTaskSession(options = {})',
            'function switchTaskSession(sessionId)',
            'function deriveTaskSessionLabel(session)',
            'function planningDecomposition(message, spec)',
            'Decompose total electron scan into spin-resolved nelec values',
            'alpha=beta for even total electron counts',
            "document.getElementById('run-study').disabled = true;",
            'font-size: 0.92rem',
            'function renderPlanSummary(plan)',
            "if (!report || typeof report !== 'object')",
            "renderTable('case-table', displayRows, { className: 'case-table' });",
            'return analysisRows;',
            'id="draft-with-llm"',
            'id="top-open-model-builder"',
            'id="open-model-builder"',
            'id="preview-model-hamiltonian"',
            'id="model-hamiltonian-structure-preview"',
            'function prepareStudySpecForRequest(',
            'function plannerRequestContext()',
            'function applyPlannerModeContext(',
            'function openModelBuilder()',
            "target: 'study'",
            "window.name = 'pyscf-computational-study-agent';",
            "pyscf-agent:model-hamiltonian-input-saved",
            "window.addEventListener('message', handleBuilderMessage);",
            "document.getElementById('task-session-select').addEventListener('change'",
            "document.getElementById('new-task').addEventListener('click'",
            '/api/model-hamiltonian-preview',
            'function renderModelHamiltonianStructurePreview(preview)',
            'base_model_input_file',
            'function validationSummary(issues)',
            'function errorDetailMessage(data, fallback)',
            'function plannerClarificationMessage(payload)',
            'function wikiEvidenceSummary(evidence)',
            'Retrieved Wiki Rules',
            "const visibleMessages = plannerMessages.filter((message) => message.role !== 'trace');",
            'validation_issues',
            "payload.status === 'needs_input'",
            '/api/study-llm-draft',
            'study_mode: requestContext.mode',
            '/api/study-result-analysis',
            'await refreshSavedStudy();',
            'direct_active_space_review_plan',
            'path_refinement',
            'entanglement_active_space_plan',
            "postJson('/api/study-result-analysis', requestPayload)",
            'id="analyze-study-results"',
            'id="study-result-analysis"',
            'Postprocessing',
            '/api/study-postprocess',
            '/api/study-postprocess-suggestions',
            '/api/study-artifact',
            'id="suggest-plots"',
            'id="generate-default-plots"',
            'id="generate-selected-plots"',
            'id="generate-hamiltonian-dataset"',
            'id="collect-hamiltonian-dataset"',
            'id="plot-spec-table"',
            'id="postprocess-artifacts"',
            'id="custom-plot-x"',
            'id="custom-plot-y"',
            'id="custom-plot-color"',
            'id="custom-plot-group"',
            'id="add-custom-plot"',
            '<option value="heatmap">Heatmap</option>',
            'function updateCustomPlotControls()',
            'function postprocessContext()',
            'No successful calculation outputs are available for postprocessing',
            "[adaptive ? 'Initial scan' : 'Method', adaptive ? initialScanLabel : task.method]",
            "adaptiveInitialScanStrategy: options.adaptiveInitialScanStrategy || 'auto'",
            'function addCustomPlotSpec()',
            'function caseVariableColumns()',
            'function suggestPostprocessPlots()',
            'function runPostprocessing(specs',
            'function renderPostprocessArtifacts(artifacts)',
            'function generateHamiltonianDataset()',
            "actions: ['generate_hamiltonian_dataset']",
            'function collectHamiltonianDataset()',
            "actions: ['collect_hamiltonian_dataset']",
            "artifact.kind !== 'postprocess-plot-spec'",
            'function loadArtifactPreviews()',
            'function isPreviewablePlotArtifact(artifact)',
        ]
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, html)

        forbidden_fragments = [
            '<h2 class="section-title">Work Directory</h2>',
            'Directory path',
            'Human Approval',
            'Planner Spec</h2>',
            'Planning Goal',
            "['U', 'V', 't', 'epsilon', 'nelec']",
            'id="artifact-list"',
            "strong_large: 'dmet'",
            "base_task: { solver: 'fci' }",
            "spec.base_task.solver = 'fci'",
            'initial_scan_case',
            'decisions.slice(0, 12)',
            'initialScanIssues.slice(0, 12)',
            'issues.slice(0, 6)',
            'score=${decision.score}',
            'Adaptive Decisions recovery notes',
            'refined_energy',
            'row.initial_scan_solver',
            'initial_scan_energy',
            'row.physics_score',
            'row.solver_stress_score',
            'physics=${row.physics_score}',
            'solver stress=${row.solver_stress_score}',
            'id="comparison-table"',
            'Adaptive results are merged into the Plan Review table.',
            'Static Sweep',
            'function normalizeSuspiciousActiveElectrons(activeSpace)',
            "document.getElementById('run-study').disabled = awaitingCostReview;",
            'loadExampleForSelectedSystem();',
            'New planner task created.',
            'approvedAdaptiveRefinedPlanReady',
            'function activeSpacePlanForCaseIds(caseIds)',
            'function approveActiveSpaceForCases(caseIds)',
            'function acceptVisibleActiveSpaceCostEstimate(plan)',
            'function activeSpaceExpansionEvidence(activeSpace)',
            'function orbitalElectronIncrement(row, spinResolved)',
            'function expandActiveSpaceContract(activeSpace)',
            'function prepareExpandedActiveSpace(caseIds)',
            'let approvedPathRestartApproval = null;',
        ]
        for fragment in forbidden_fragments:
            with self.subTest(forbidden=fragment):
                self.assertNotIn(fragment, html)

        self.assertLess(html.index('Task Setup'), html.index('Planner Conversation'))
        self.assertLess(html.index('Plan Review'), html.index('id="case-table"'))
        self.assertGreater(html.index('Plan Review'), html.index('<div class="stack">\n        <div class="panel stack">\n          <div>\n            <h2 class="section-title">Plan Review</h2>'))

    @unittest.skipUnless(shutil.which('node'), 'node is not installed')
    def test_molecular_task_preview_resolves_scan_coordinates(self):
        html = planner_ui_source(build_study_index_html())
        preview_source = html[
            html.index('function firstPreviewCaseVariables(spec)'):
            html.index('function renderMolecularTaskSetup(spec)')
        ]
        script = """
          let currentPlan = null;
          %s
          const spec = {
            case_design: { mode: 'grid', variables: { bond_length: [0.8, 1.2] } },
            base_task: { atom: 'N 0 0 0; N 0 0 ${bond_length}', unit: 'Angstrom' },
          };
          const draftPreview = molecularPreviewTask(spec, spec.base_task);
          if (!draftPreview.resolved || draftPreview.atom !== 'N 0 0 0; N 0 0 0.8') {
            throw new Error('Draft preview did not resolve the first scan coordinate.');
          }
          currentPlan = {
            system_type: 'molecular',
            cases: [{ request: { task_type: 'molecular', atom: 'N 0 0 0; N 0 0 1.2', unit: 'Angstrom' } }],
          };
          const plannedPreview = molecularPreviewTask(spec, spec.base_task);
          if (plannedPreview.atom !== 'N 0 0 0; N 0 0 1.2') {
            throw new Error('Preview did not prefer the resolved planned case.');
          }
        """ % preview_source
        subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True)

    def test_build_and_run_use_saved_study_endpoints(self):
        html = planner_ui_source(build_study_index_html())
        self.assertIn("postJson('/api/study-prepare'", html)
        self.assertIn("postJson('/api/study-start'", html)
        for legacy in ('/api/study-run', '/api/study-adaptive-run', '/api/study-plan', '/api/study-adaptive-plan'):
            self.assertNotIn(legacy, html)
        self.assertNotIn('function shouldRunApprovedAdaptiveRefinedPlan()', html)
        self.assertNotIn('function adaptiveResumeStudyId()', html)

    @unittest.skipUnless(shutil.which('node'), 'node is not installed')
    def test_frontend_consumes_backend_adaptive_workflow_without_rebuilding_it(self):
        html = planner_ui_source(build_study_index_html())
        self.assertNotIn('function refreshAdaptiveWorkflow(', html)
        self.assertNotIn('function mergeReportPayloadByCase(', html)
        self.assertNotIn('function adaptiveReportWithApprovedRefinedResults(', html)
        workflow_source = html[
            html.index('function adaptiveWorkflow(payload)'):
            html.index('function adaptivePlanByKind(adaptive, planKind)')
        ]
        script = """
          %s
          const workflow = {
            schema: 'pyscf-agent.adaptive-workflow.v1',
            stage: 'recovery_review_required',
            allowed_actions: [{ id: 'increase_recovery_max_cycle' }],
          };
          const payload = { adaptive: { workflow } };
          const result = adaptiveWorkflow(payload);
          if (result !== workflow || result.stage !== 'recovery_review_required') {
            throw new Error('The frontend did not consume the backend workflow object directly.');
          }
          const staticPayload = { workflow };
          if (adaptiveWorkflow(staticPayload) !== workflow) {
            throw new Error('The frontend did not consume the static study workflow object directly.');
          }
        """ % workflow_source
        subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True)

    @unittest.skipUnless(shutil.which('node'), 'node is not installed')
    def test_planner_correlation_level_is_evidence_driven(self):
        html = planner_ui_source(build_study_index_html())
        correlation_source = html[
            html.index('function reviewCorrelationLevel(decision = {}, result = {}, structured = {})'):
            html.index('function adaptiveStatusClass(status)')
        ]
        script = """
          function deepCopy(value) { return JSON.parse(JSON.stringify(value)); }
          %s
          const levels = {
            failedSolverStress: reviewCorrelationLevel(
              { level: 'strong', diagnostic_level: 'moderate', solver_stress_level: 'strong', tags: ['review_active_space'] },
              { method: 'casscf', status: 'failed' },
              {}
            ),
            recoveredSolverStress: reviewCorrelationLevel(
              { level: 'strong', diagnostic_level: 'moderate', physics_level: 'weak', solver_stress_level: 'strong' },
              { method: 'casscf', status: 'succeeded' },
              {}
            ),
            blockedApproval: reviewCorrelationLevel(
              { level: 'moderate', tags: ['requires_active_space_approval'] },
              { method: 'casscf', status: 'blocked' },
              {}
            ),
            successfulCasscf: reviewCorrelationLevel(
              { level: 'moderate' },
              { method: 'casscf', status: 'succeeded' },
              {}
            ),
            weakFci: reviewCorrelationLevel({ level: 'weak' }, { method: 'fci', status: 'succeeded' }, {}),
            modelSolverStress: reviewCorrelationLevel({}, {}, {
              strong_correlation_diagnostics: {
                kind: 'strong_correlation_diagnostics', level: 'strong', physics_level: 'weak', solver_stress_level: 'strong',
              },
            }),
            modelMissingEvidence: reviewCorrelationLevel({}, {}, {
              strong_correlation_diagnostics: {
                kind: 'strong_correlation_diagnostics', level: 'strong', physics_level: 'unknown', solver_stress_level: 'strong',
              },
            }),
          };
          process.stdout.write(JSON.stringify({ levels }));
        """ % correlation_source

        completed = subprocess.run(
            [shutil.which('node'), '-e', script],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload['levels']['failedSolverStress'], 'moderate')
        self.assertEqual(payload['levels']['recoveredSolverStress'], 'moderate')
        self.assertEqual(payload['levels']['blockedApproval'], 'moderate')
        self.assertEqual(payload['levels']['successfulCasscf'], 'moderate')
        self.assertEqual(payload['levels']['weakFci'], 'weak')
        self.assertEqual(payload['levels']['modelSolverStress'], 'weak')
        self.assertEqual(payload['levels']['modelMissingEvidence'], 'unknown')
