# Saved Review, Analysis And Dataset MCP Workflows

Inspection date: 2026-09-07. Follow-up to the
[thin agent interface](2026-09-07-mcp-agent-interface.md).

## Problem And Result

MCP review previously returned a complete plan which the client had to retain
and send back. Review decisions now live in StudyReport.pending_review: selected
action, plan, case IDs, workflow, run readiness and existing continuation
approval. The application reads that record on start by Study ID. Current Task
results remain unchanged until execution; no task database or queue is added.

Start holds the existing invocation lock while reading the decision and handing
it to the worker. The worker consumes it after its application call returns;
launch/transport interruptions retain it for recovery. Collection preserves the
saved decision without executing it. Existing Python explicit-plan calls remain
available, while the MCP prepared_plan argument has been removed.

The old analyze_results is split into deterministic diagnose_results and an
optional LLM interpretation wrapper. Both reuse existing path/state diagnostics,
MPS/path continuation and active-space review builders. MCP analyze_study saves
scientific evidence by Study ID without an LLM or new calculations. Optional
postprocessing uses the existing plot/data generator. Reviewed MPS/path
continuation starts in the background through those same application services.

prepare_dataset converts JSON geometry contracts and calls the existing MD
Study planner. Task validation now checks runtime-supported MD settings before
saving the plan; a custom contract cannot imply an unsupported executable
method. Ordinary Study execution assembles accepted/rejected sample indexes and
the manifest. MCP exposes the saved manifest as a resource. The executable smoke
uses the existing B3LYP/def2-SVP settings with two MD frames, not a full QH9
benchmark trajectory.

The stdio surface now has thirteen tools. No scientific execution loop was
added to MCP. Registry, wiki and report authority remain unchanged.

The real retry campaign exposed a stale cost-approval bug: changing the selected
case's runtime left the old estimate attached, so approval was immediately
invalidated on start. Review actions now refresh the existing cost estimator
for their current plan, and approval checks resource limits against that fresh
estimate. The fingerprint binding remains enforced. A regression covers both
new reviews and previously saved stale estimates.

## Verification

The focused tests cover saved review across client restart and failed launch,
reviewed subset preservation, continuation dispatch, no-LLM diagnostics and
postprocessing, direct-CAS approval, and real two-frame H2 dataset generation.
The source and installed MCP checks use Python 3.12 with the optional SDK.

`tools/verify_mcp_workflows.py` runs an isolated ten-Task acceptance campaign:
a deliberately low-cycle DFT case and a converged reference, selected-case
retry, analysis/plots, adaptive H2, an MD dataset and probe-to-CASSCF approval.
Each tool call uses a newly started MCP server. Remote mode also terminates
only this campaign's local coordinator after two Slurm handles are persisted,
then collects the original jobs through a new client. The numerical failure is
real nonconvergence; no synthetic TaskReport is substituted.

The initial local harness stopped at the expected cost gate because raising an
error inside the SDK client context wrapped it in ExceptionGroup. The harness
now handles the returned tool error after closing that context. No calculation
was submitted in that initial attempt; its evidence is retained separately.

The harness also needed to account for the legitimate new cost review after
retry changes and to count `(work_dir, run_id)` pairs: adaptive stage Run IDs
are scoped to their directories. These harness failures and the actual stale
approval failure are retained under `mcp-workflows-2026-09-07-harness-*` and
`mcp-workflows-2026-09-07-stale-cost-approval.*`; none is presented as a passing
acceptance run.

- Python 3.12.14 focused suite: **189 tests passed**, including real stdio,
  background process, analysis, dataset and review-action tests.
- Local production-MCP campaign: **10 Task runs, all final checks passed**.
  The initial DFT case is deliberately unconverged; after review both cases
  succeed and their attempt counts are `[2, 1]`.
- Wheel inventory: installed package is loaded from Python 3.12 site-packages
  outside the source checkout; **13 tools** are exposed. The wheel checksum and
  package path are in `reports/verification/mcp-workflows-2026-09-07-wheel.json`.
- Installed-wheel campaign: the same **10 Task runs passed** from a working
  directory outside the checkout. This includes saved review across fresh MCP
  clients, optional plots and two accepted MD frames.
- Python 3.9.6 source suite: **1,015 tests, 993 passed and 22 skipped**. This
  existing scientific environment is source-regression evidence, not a
  supported-version installation check. The Python 3.12 optional-provider
  full-suite limitations from earlier records remain unchanged.
- Wiki lint: **0 errors, 3 existing warnings**, with 43 curated pages.

The exact logs and campaign JSON are under
`reports/verification/mcp-workflows-2026-09-07-*`.

The Amarel release is
`mcp-workflows-20260907-20260907-050936z-064be8e7b5`, in the isolated
`mcp-workflows-20260907` environment. Preflight and post-deployment capability
checks passed with source fingerprint
`blake2b-256:064be8e7b5226ddcdae10f9cd8e9efdaf3ce0633751c6b57831fa5f292030881`.
The private local config is
`.pyscf-agent/mcp-workflows-20260907/remote.ini`; its source snapshot is kept in
the sibling `source` directory so later verification logs do not invalidate
runtime matching. The 247 runtime files match the current checkout. Subsequent
documentation and evidence updates belong to the live checkout.

The Amarel campaign passed **17 checks across 10 Task runs**. Two initial
Slurm handles were saved before the harness terminated its own coordinator.
A new MCP client detected the interruption and collected those exact runs.
The low-cycle case was unconverged and its reference succeeded; approving the
selected retry produced attempts `[2, 1]` with the reference report unchanged.
Analysis and plots created no additional Task runs. One adaptive start
completed four runs, the MD run produced two accepted frames and zero rejected
frames, and probe-to-approved-CASSCF execution completed in the same Study.
All ten receipts are collected. Slurm array task IDs and the matching runtime
release are archived in
`reports/verification/mcp-workflows-2026-09-07-amarel-jobs.json`.

The compact aggregate record is
`reports/verification/mcp-workflows-2026-09-07-summary.json`. This campaign
verifies the MCP execution and recovery paths with tiny molecules; it does not
claim full QH9 numerical reproduction or a new cluster-scale MPS benchmark.

## Boundaries

The remote campaign uses an isolated frozen source release and the existing
scientific dependency environment. New remote submissions require the matching
private config/source snapshot. Numeric arrays remain ordinary executor-owned
artifacts. MCP still does not expose generic remote submission reconciliation,
HTTP transport or protocol Tasks. Direct Python/Web callers should not overlap
a background writer to the same Study.

No GitHub commit or publication is part of this increment.
