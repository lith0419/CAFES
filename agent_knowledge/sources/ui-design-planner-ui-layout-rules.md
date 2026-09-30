# Planner UI Layout Rules

## Current Layout

The planner uses a responsive two-column workspace.

- The left column holds work directory/system/mode controls, **Task Setup**, and
  **Planner Conversation** with its message composer.
- The right column holds **Plan Review**, adaptive decision/review gates,
  **Postprocessing**, and the resulting `StudyReport` analysis.
- Model studies show builder and model-preview controls in Task Setup; molecular
  studies show the assistant-style task summary and molecular structure preview.

## Rules

- Task setup should show the current scientific input, not raw planner JSON.
- Conversation is the place for intent and system/assistant messages; message
  input follows the conversation history.
- Build and Run remain explicit Plan Review commands.
- Adaptive failure/recovery choices use the same reusable approval-card pattern
  as active-space review, with obvious status and case-local actions.
- Status must appear at the right edge of decision rows and explain what action
  is available next.
- Keep only one collapsed level in complex controls. Avoid nested expandable
  panels that hide scientific input or approval context.
- Task sessions must snapshot their own StudySpec, model input, plan, report,
  review gate, postprocessing specs, and artifacts when users switch tasks.
- Keep the HTML file as the semantic page shell. Planner styling belongs in
  `planner.css`; browser behavior belongs in responsibility-scoped
  `planner-*.js` assets loaded in their declared order.
- Reuse `/assets/molecular-preview.css` and
  `/assets/molecular-preview.js` for molecular previews instead of duplicating
  their implementation in Planner assets.

## Must Not

- Do not show a model-Hamiltonian input requirement for molecular studies.
- Do not make the user locate a failure in a table before offering the relevant
  review action.
- Do not let left/right columns overlap at intermediate viewport widths.
- Do not duplicate case-level detail in both adaptive decisions and the review
  table; the decision panel should summarize state and offer actions.
