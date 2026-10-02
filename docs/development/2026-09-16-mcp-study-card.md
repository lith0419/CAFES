# Read-Only MCP Apps Study Card — 2026-09-16

## Scope

Add a small inline-capable view over the existing Study application service.
The full WebUI, scientific computation, adaptive decisions and Task/Run management
remain in their existing layers. This increment adds no calculation or review buttons.

## Implementation

- `pyscf_agent/mcp_server/study_view.py` projects `open_study` and
  `inspect_execution` into a bounded summary. `show_study` links to a versioned
  HTML resource; app-only `refresh_study_view` returns the same data without
  requesting another card.
- `pyscf_agent/web_assets/study-card.html` is a self-contained MCP Apps component.
  It initializes the standard host bridge, renders tool results, supports theme
  and size updates, refreshes on demand, and opens the existing workbench through
  the host. It needs no JavaScript build step or external assets.
- The card preserves separate execution and scientific report statuses. Case
  counts and scheduler jobs have distinct labels. Saved results explicitly may
  precede a running retry. Unknown convergence stays unknown; explicit false
  stays false. The 20-row limit does not truncate total case counts.
- Energy units come from result metadata, with molecular/periodic defaults;
  model-Hamiltonian values are not silently relabeled Hartree.
- Card display/refresh do not start the workbench. The explicit workbench button
  uses `open_workbench`, then `ui/open-link`; unsupported/denied navigation retains
  a visible link. Non-UI MCP clients retain structured/text responses.
- The plugin skill selects quick card summaries or full browser interaction.
  Plugin version `0.1.0+codex.20260916202319` was installed in the personal
  marketplace; all six installed plugin files match the source package.

## Verification

- **105 focused tests passed, no skips**, including MCP lifecycle, saved WebUI,
  workbench launch/reuse, and small real H2 static/adaptive/retry calculations.
- After the final energy-unit display refinement, **four affected tests passed**.
  They cover wire metadata and resource MIME/CSP, text-only fallback, no changes
  to saved Study files, no execution/workbench startup during viewing, bounded
  result projection, false/missing convergence, and model energy units.
- The JavaScript host fixture covers initialization, refresh, navigation denial,
  absent host capabilities, timeouts, teardown, duplicate clicks, foreign messages
  and rendering untrusted labels as text.
- Installed stdio discovery returned **16 tools**, including one app-only refresh
  tool. Resource read, card display, reconnect/refresh and existing workbench reuse
  passed. Study `20260916-155300-67cdc98e` retained **4 Tasks / 4 Runs**; all **103
  saved files** were byte-for-byte unchanged during the final acceptance.
- The local test host loaded the actual MCP UI resource in a sandboxed browser
  iframe. Four saved H2 energies/convergence flags rendered correctly; refreshing
  updated the timestamp and opening the workbench showed the same saved Study.
  Pointer automation could not inspect fractional iframe coordinates; keyboard
  activation verified both buttons. The temporary test host was stopped afterward.
- Plugin/skill validators and whitespace checks passed.

Evidence: machine-readable acceptance (author archive: `reports/verification/mcp-study-card-2026-09-16.json`).
Usage and reproducible preview: [Study card guide](../guides/mcp-study-card.md).

## Remaining Host Acceptance

**Actual Codex conversation-inline rendering remains unverified.** This development
task does not have the PySCF Agent tools in its loaded inventory. Reinstalling
updates the plugin for a new task; it does not establish that this running task
can call or render the new component. The browser test host is deliberately
labeled as a protocol preview and is not evidence of Codex inline support.

The next acceptance is to load the updated plugin in a new Codex task, call
`show_study` for the saved H2 Study, and verify the native inline card and both
buttons. If the host does not support it, retain the structured summary and
browser workbench fallback. No remote, publication, or large-solver acceptance
was performed in this increment.
