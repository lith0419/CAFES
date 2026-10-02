# Saved Study Workbench And MCP Interoperation

Inspection date: 2026-09-16. Baseline: `c620f3b`; development branch:
`codex/codex-plugin`. This follows the [plugin packaging increment](2026-09-16-codex-plugin.md).

## Implementation

`StudyApplicationService.open_study` reads the existing preparation and optional
report. `list_studies` discovers these artifacts under the output root, with
unreadable entries identified individually. Neither method submits work,
contacts a scheduler or introduces an index.

The Web adapter adds list/open/start routes. Review requests with a Study ID
delegate to `review_study`; collection with `saved_study` delegates to
`collect_saved_study`. These are the same methods and invocation lock used by
MCP. Browser copies of plans and reports are omitted from saved review requests
and are not used by the saved service branch. Saved scientific-analysis
approvals wait for an explicit agent start rather than invoking the legacy
browser analysis runner.

The Web bootstrap accepts `--work-dir` and passes executor connection options
to its Study services, including remote/Slurm background execution. The new
frontend module opens by ID, retains an addressable URL, restores pending
reviews and reuses existing plan, result and review rendering. Initial draft
rendering preserves incoming URL parameters while connection setup completes.

Static retry metadata previously triggered an adaptive display even though the
underlying Study was static. Rendering now requires adaptive mode or actual
adaptive/review-stage content, preserving the static result table. Pending
cost approvals and selected retry plans remain visible after refresh.

## Verification

See machine-readable evidence (author archive: `reports/verification/study-workbench-2026-09-16.json`).

- **91 focused tests passed, no skips**, including the optional real numerical
  MCP/installed-plugin tests, in 36.096 seconds. Coverage includes static and
  adaptive saved workflows, selective retry, cost approval, separate MCP/Web
  service instances, HTTP route integration and existing Web behavior.
- After the final browser review-request adjustment, **6 UI checks passed**.
  The behavior test covers deep-link preservation, persisted cost gates,
  omission of browser plan/report copies, static retry rendering, and avoiding
  an automatic legacy analysis invocation after saved review.
- Real installed-plugin MCP prepared a two-case H2 Study. In Codex's browser,
  the workbench opened it, approved its resource estimate, survived refresh,
  started both calculations and collected them. MCP then prepared a retry of
  the first case; the page reopened that decision, approved its estimate,
  executed the retry and collected the final report.
- Final Study: `20260916-144523-9cf9e952`, **2 Tasks / 3 Runs**, attempt counts
  `[2, 1]`. Energies: `-1.1167593073964255` Ha and `-1.126755317196932` Ha.
  Repeated collection retained the same counts. The original installed-plugin
  acceptance Study also opened successfully without new execution.

The numerical runtime was Python 3.12.14, PySCF 2.13.1 and MCP 2.2.0, using
one OMP/OpenBLAS/MKL thread. No Amarel jobs were submitted. HTTP integration
uses a fixture executor for remote/adaptive behavior; it is not a live cluster
acceptance claim.

## Remaining Work

At this increment's completion, browser draft planning still used the original
endpoints. The subsequent [saved planning migration](2026-09-16-saved-study-planning.md)
moved Build Plan onto saved preparation and removed the frontend Run dispatcher.
Plugin-driven workbench startup and links in tool responses can follow without
adding orchestration. Inline MCP Apps rendering remains a separate host
compatibility experiment. See the [workbench guide](../guides/study-workbench.md).
