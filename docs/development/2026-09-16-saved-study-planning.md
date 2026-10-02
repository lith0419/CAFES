# Saved Browser Planning And Frontend Orchestration Removal

Inspection date: 2026-09-16. Baseline: `c620f3b`; development branch:
`codex/codex-plugin`. This follows the
[saved workbench increment](2026-09-16-study-workbench.md).

## Implementation

Browser Build Plan now calls `/api/study-prepare`. The adapter delegates static
and adaptive requests to `StudyApplicationService.prepare_study`, and dataset
requests to `prepare_dataset`. The response uses `open_study`'s saved projection,
so creation and reopening feed the same view. No new scientific runner, task
index or orchestration state was added.

Build Plan saves a stable Study ID before execution. The browser retains that
ID in its URL and task session, locks saved inputs, and uses the existing saved
review/start/collect operations. Repeated clicks during preparation create one
Study. If the user changes tasks before the request finishes, the response stays
with the original draft. New Task creates independent inputs. Dataset fields
and adaptive controls are restored from saved metadata.

The frontend static/adaptive/retry/resume dispatcher and its helper predicates
were removed. Run Plan calls the agent by ID; the agent chooses the existing
execution or continuation path from saved state. Old plan/run HTTP endpoints
remain for compatibility with external clients, with their original behavior;
the current browser does not call them.

LLM-generated plans remain editable previews until Build Plan saves them. The
result-analysis action also now reads by ID through `analyze_study`, persists
scientific diagnostics and review state under the existing invocation lock,
and optionally adds the WebUI's LLM interpretation. Its language-generation
helper is shared with the legacy analysis entry point. MCP analysis continues
to work without an LLM. Review decisions do not automatically execute an old
browser analysis branch.

## Verification

See machine-readable evidence (author archive: `reports/verification/saved-study-planning-2026-09-16.json`).

**98 focused tests passed, no skips, in 41.504 seconds**, including optional
real numerical MCP and installed-plugin tests. JavaScript syntax and whitespace
checks also passed. No package or dependency changes were needed.

HTTP/MCP tests use separate application service instances sharing saved files.
They cover browser preparation of static molecular, model-Hamiltonian,
adaptive and dataset plans; active-space probe routing; invalid inputs; saved
analysis; review; execution; and collection. Node behavior tests cover duplicate
Build clicks, delayed responses after task switches, ID-only analysis/review,
deep links and saved cost gates. Existing legacy API tests remain enabled.

A real browser acceptance created a four-point H2/UMP2/STO-3G plan, refreshed
the page, then started and collected it. Before execution, the installed Codex
plugin's actual MCP transport reported four unexecuted Tasks and no Run state
files. After browser execution, the same transport read and recollected the
completed Study without increasing attempts or creating Runs:

- Study ID: `20260916-155300-67cdc98e`.
- Final status: `succeeded`; **4 Tasks / 4 Runs**, attempts `[1, 1, 1, 1]`.
- H–H distances: 0.74, 1.20, 1.80 and 2.50 Å.
- Energies: −1.1298973809859583, −1.033662879778054,
  −0.8978820219177821 and −0.853542316806196 Ha.

The existing example retains the name `h2-bond-adaptive`; this acceptance
explicitly selected **Static Scan**. These energies verify transport and
execution continuity, not the accuracy of stretched-bond MP2. Runtime:
Python 3.12.14, PySCF 2.13.1, MCP 2.2.0; one OMP/OpenBLAS/MKL thread.

## Remaining Work

At this increment's completion, workbench startup was manual. The subsequent
[launcher increment](2026-09-16-workbench-launcher.md) provides a plugin tool
using the same configured environment and output root. Inline MCP Apps rendering is still a separate host compatibility
experiment. Neither change needs a second scientific workflow.

No Amarel jobs or full dataset campaign were run. Adaptive/model/dataset HTTP
execution coverage uses fixtures; dataset preparation does not demonstrate a
completed molecular dynamics campaign. LLM analysis transport uses a stubbed
language response while exercising the real saved-analysis service.
