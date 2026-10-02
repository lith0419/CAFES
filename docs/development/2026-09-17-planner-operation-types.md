# Planner operation target types

The shared model-operation engine previously called `int()` directly on site
IDs. A generated `site: [0, 1]` therefore passed the StudySpec presence checks
and failed during case expansion with an uninformative Python TypeError. The
Web handler discarded the candidate in that exception path. This reproduces
the reported message, although the original failed live LLM reply was not
available to establish its exact field.

Site and bond targets, selector IDs/counts, and normalized electron counts now
use the existing lossless integer helpers. `site` remains one site ID and
`sites` retains its native group behavior. Endpoint pairs in `bond` retain
their meaning, distinct from bond-ID lists in `bonds`. Invalid lists are not
flattened or reduced to one element.

The operation engine annotates input errors with an operation index; Study
planning supplies the case index. `StudyApplicationService.prepare_plan`
exposes build-time ValueErrors through its existing validation result, so the
Web adapter returns `needs_input` with the complete candidate and its error.
It does not return a partial plan. Unexpected TypeErrors still propagate;
there is no catch-all replacement plan or extra LLM retry.

The Planner prompt and curated Wiki now describe native grouped-site updates
and no longer claim that a group must be compiled into separate operations.
Validation happens after template substitution, including when a reference
resolves to a list only in a later case.

Verification:

- 102 focused tests passed on Python 3.12 and on the WebUI-compatible Python
  3.9.6 environment, with no failures or skips.
- The existing 32-site, 48-bond Honeycomb model was replayed through the Web
  handler with explicit fixture replies. An invalid `site` list preserved the
  draft and returned its case/operation location. Correcting it to `sites`
  generated all four U/V cases; every site U and bond V/effective_V was checked.
  The equivalent simple `sweep` also generated four cases.
- Wiki curation exported 43 pages. Lint reported zero errors and three existing
  citation warnings in unchanged roadmap/benchmark pages.
- `git diff --check` passed.

Evidence is in `reports/verification/planner-operation-types-2026-09-17/`.
The Honeycomb replay did not call an external LLM or run DMET. Existing
WebUI/MCP processes must be restarted to load the changed Python source.
