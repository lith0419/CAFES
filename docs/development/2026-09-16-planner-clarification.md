# Planner Clarification Without A Plan

Inspection date: 2026-09-16. Follows the saved Study planning migration.

## Problem And Correction

`draftWithLlm` passes `payload.plan || null` to `renderPlan`. The planning API
intentionally omits `plan` when it returns `status=needs_input`, for example
when a model solver is missing or a case operation fails validation.
`renderPlanCaseTable` still read `plan.cases` unconditionally. This raised
`null is not an object (evaluating 'plan.cases')` in WebKit, hiding the agent's
useful clarification and leaving the preceding table visible.

The table now accepts an absent plan as an empty case list. `renderPlan` keeps
such a session in `draft` state. The existing clarification flow then displays
the actual missing inputs. A later valid response renders its cases normally;
Build Plan still saves the Study before any execution is allowed.

The three most recent local directories inspected contained only generated
model input files, without Study plans, invocation records or TaskReports.
Those files do not show a submitted or failed numerical calculation. The exact
missing field in the user's response was not persisted; it cannot be recovered
from the generic JavaScript error alone.

## Verification

A regression feeds actual planning-handler responses into the frontend draft
flow with a stubbed LLM: missing model solver, invalid site operation, explicit
null plan, and a subsequent valid two-case plan. Before the fix it reproduces
`Cannot read properties of null (reading 'cases')`. After the fix it verifies
the clarification text, cleared previous rows, editable draft, enabled planner
button and disabled execution button, followed by successful plan revision.

All **56 tests** across saved Study UI, LLM/Wiki planning, Web API and WebUI
passed. The running local workbench's served JavaScript matches the corrected
file byte-for-byte. Refresh the page to load the updated script; no server
restart or numerical rerun is required for this correction.
