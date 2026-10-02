# Planner Draft Contract And Failure Semantics

Inspection date: 2026-09-17. Follows the
[clarification rendering correction](2026-09-16-planner-clarification.md).

## Problem

JSON extraction happened outside the existing one-attempt contract repair.
Plain text, empty content and truncated JSON therefore failed immediately.
The shared extractor also counted braces inside quoted strings as JSON syntax.

A second path could silently change the study: legacy `StudySpec.from_dict`
defaults turned incomplete, unversioned model output into a default name and
an empty scan. Separately, the Web API's local regular-expression fallback
constructed a new simple sweep after an LLM failure, discarding the seed's
case design and other fields. A failed revision could become a successful
replacement plan.

## Contract

The LLM returns a complete candidate StudySpec. Generated output must satisfy
the existing versioned public contract before legacy constructor defaults are
applied, even when the model omits the schema marker. Names, objectives and
system types must be nonempty strings; task/scan definitions must be objects,
and observables must be a nonempty array.

A revision of an existing scan must explicitly describe its sweep or cases.
Omission is not permission to collapse a scan to a default single point. An
intentional single-point replacement uses an explicit nonempty cases list.
This rule does not infer user intent from Chinese or English phrases or copy
missing scan points into a generated plan after validation.

Parsing and contract validation share one bounded correction request, carrying
the same user goal, seed and retrieved Wiki evidence. Both the first and corrected
response pass the same contract. Failure returns an error; the Web client keeps
its existing spec, plan and case rows. A complete candidate requiring a
scientific clarification still uses the existing `needs_input` response and
remains editable. DMET/model validation and execution approval are unchanged.

The local simple-sweep fallback and its numeric parser were removed. There is
one generation path, not a second planner that guesses a replacement after a
failed request. Constructor compatibility for handwritten/legacy inputs stays
outside the stricter LLM boundary.

The shared JSON extractor now uses Python's JSON decoder to read one object
within optional surrounding prose or Markdown. It respects string escapes
and rejects malformed outer objects instead of salvaging a nested fragment.
Contract rejection reasons are logged at warning level, without logging API
credentials or full model replies.

## Wiki Retrieval And Planner Input

A follow-up audit found that Study drafting exported the entire runtime
registry twice: into the local Wiki search query and into the LLM message.
The current capability snapshot serialized to 352,512 characters. Its growth
since the early implementation was inherited by every drafting request,
including correction requests. This is an input-boundary defect; it is not
evidence that context length caused the reported malformed response.

Drafting now queries the compiled Wiki using only the current goal and seed
StudySpec. The LLM receives selected Wiki excerpts, the goal, seed, Builder
site context, and the StudySpec schema. The prompt no longer refers to an
injected capability registry. Structured-output fallback and contract repair
reuse the same input contract, including the output schema in the message.
Unused registry parameters and imports were removed from the drafting layer.

The runtime registry remains the backend authority for plan validation. Its
exports used by plan artifacts and other consumers are unchanged. No new
capability catalog, scientific fallback planner, or retry layer was introduced.

## Verification And Deployment

The focused 141-test suite passed under both Python 3.12.14 and Python 3.9.6,
the interpreter version used by the user's port-8001 service. It covers:

- Plain text, empty/null content, malformed and truncated JSON; successful
  correction and termination after one unsuccessful correction.
- Unversioned/partial drafts, null names, omitted/empty scan definitions and
  preservation of a 16-case U/V scan through the real planning handler.
- An explicitly requested single-point replacement.
- Structured-output transport fallback, without restoring the removed
  scientific-plan fallback.
- Frontend retention of the previous spec, plan and table after an error.
- Existing request-builder, scientific validation and Web API behavior.
- Wiki retrieval from the goal and seed without exporting the registry;
  preservation of the same Wiki context and output schema through transport
  fallback and correction.
- Backend rejection of an unregistered solver returned by the LLM.

Reconstructing the same 32-site Builder / 16-case DMET example locally reduced
the combined message contents from 377,102 to 23,441 characters. The retrieval
query is now 400 characters. These are character counts for a synthetic audit
request, not provider token usage or a captured failing request.

LLM transport is stubbed in these regressions; no real LLM or numerical job was
submitted. The exact offending provider reply was not persisted and cannot be
reconstructed from the reported error. The user's service was identified as
the manually launched Python 3.9 process on port 8001 with the Amarel executor;
its stdout/stderr point to a terminal, not a log file. Terminal inspection was
unavailable through the computer-use tool. The three saved Studies returned by
that service were completed September-16 acceptance Studies; the reported
16-case draft was not among them.

Python changes require restarting that service in its original configured
shell and then refreshing the page. Refreshing alone does not reload the
backend. The existing browser draft is not a server-side log or saved Study;
this change does not claim to restore an already overwritten draft.
