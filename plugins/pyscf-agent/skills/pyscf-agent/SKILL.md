---
name: pyscf-agent
description: Prepare, execute, inspect and analyze quantum chemistry or model-Hamiltonian calculations with the PySCF Agent MCP tools. Use for existing Study status, active-space review, local or Slurm calculations, selected-case retries and checkpoint continuation through this agent.
---

# PySCF Agent

Use the installed PySCF Agent MCP server. The agent owns scientific execution,
static/adaptive orchestration, Task/Run identity, recovery and report merging.
Use the user's language for discussion and explanations.

## Establish context

1. Call `get_capabilities` to identify the configured executor and relevant
   Registry namespaces. Registered support does not prove an optional solver
   is installed on the execution host.
2. Read relevant schema/Wiki resources when constructing an unfamiliar input.
   Capability limits and scientific options come from those resources, not
   guessed parameters or copied solver implementations.
3. Preserve a returned Study ID or JobHandle in the conversation. For an
   existing calculation, use that identity rather than preparing it again.
   This version does not list all saved Studies; if its ID is absent, obtain
   it from the user or an existing saved report.

## Study workflow

- Use `prepare_study` for a new Study. It saves a static or adaptive preparation
  and returns its ID, plan and cost information. Preparing again creates a new
  Study. Multi-case work and workflows needing review/recovery belong here.
- Review the prepared scientific choices and any required decisions with the
  user. Honor authorization already given; apply existing scientific review
  gates without adding a second approval system.
- Call `submit_study(study_id)` once to start the complete agent workflow.
  The agent schedules its Tasks and advances adaptive stages itself.
- Use `get_study_status` for progress. While `agent.running` is true, inspect
  later; collect when `can_collect` permits it. Status checks do not advance
  calculations. Avoid rapid repeated polling of long cluster jobs.
- Call `collect_study` for the saved scientific results. Default to the compact
  result and returned resource URIs. Request full reports only when needed.
- Use `analyze_study` for the agent's diagnostics and follow-up proposals;
  `postprocess=true` also generates existing plot/data artifacts.

For review, use only the displayed action IDs, case IDs and approval token.
`review_study` saves a decision without starting work. After an authorized
decision returns `can_run=true`, use `submit_study` on the same Study ID.

## Retry, continuation and reconnect

Keep the parent Study and stable Tasks; a selected retry creates new Runs.
Use the existing Study submission/review options for selected cases. Adaptive
subset retries and checkpoint continuation follow the agent's saved review
actions. Leave MPS compatibility and restart validation to the agent.

Reconnect using the same execution target, output root and Study ID. Inspect
before starting anything after an uncertain submission. Existing submitted
remote jobs are managed by Slurm; later stages require the agent coordinator
to run. A closed client is different from a stopped or sleeping execution host.

For a standalone Task, use `validate_task`, then `submit_task` with a unique
Run ID, and retain its exact JobHandle for status, collection or cancellation.
Task submission is not idempotent: after an ambiguous transport failure,
inspect existing executor evidence before resubmitting. Study-wide cancellation
is not a tool in this version; do not describe a child Task cancellation as
cancelling the complete Study.

## Interpret results

Distinguish scheduler completion, scientific convergence, and result-quality
evidence. Explain observed failures and missing evidence directly.

- For DMRG-CASSCF, distinguish orbital-optimization M from a later fixed-orbital
  M, and outer CASSCF convergence from inner DMRG sweeps/discarded weights.
- For DMET, outer convergence does not certify every intermediate impurity SCF.
- Keep numerical arrays/checkpoints as artifacts. Read concise evidence before
  loading large reports; do not promise generic binary downloads through MCP.
- Dataset preparation uses `prepare_dataset`, followed by the same Study flow.

## Interface and setup

For an existing standalone Task, call `show_task(handle)` with its exact saved
JobHandle. Compatible hosts render a task monitor with automatic refresh;
other hosts return the same structured data. For a browser view, call
`open_task_monitor(handle)` and open the returned URL in Codex. It uses the
configured executor, submits no work, and reconnects to the same Run on reload.
`refresh_task_view` is app-only. Visible active views refresh every 15 seconds,
pause when hidden, and stop at terminal status. These updates do not invoke an
LLM. Keep execution status separate from scientific success. Solver metrics
are last observed log evidence, not final task energies or convergence claims.
SSH inspection requires the corresponding updated remote agent runtime.

For a quick visual summary of an existing Study, call `show_study(study_id)`.
Hosts supporting MCP Apps can render a read-only card with saved case results,
status refresh and an Open full workbench button. The same tool returns a
structured summary on other hosts. If no card appears, use the summary and
offer/open the workbench; do not claim inline rendering was verified. The card
does not collect results, approve reviews, or submit calculations. Saved energies
may precede a running retry. `refresh_study_view` is the card's app-only data tool.

For full Study interaction, call `open_workbench(study_id)`.
It starts or reuses the existing local WebUI with the
same Python environment, output root and executor as this MCP server. Open the
returned `url` in Codex's browser using the available browser/open-panel tool,
or provide it as a Markdown link. Omit the ID for the workbench home and saved
Study list. Do not create a new Study merely to open an existing one.

Once the workbench is running, Study tool responses also include
`workbench_url`. If a saved URL no longer responds, call `open_workbench` again.
The Web service survives MCP disconnects and binds only to local loopback. It
does not run calculations when opened. On a remote MCP host, its localhost URL
belongs to that host; do not claim it is reachable from the desktop without an
existing forwarding setup. The full workbench opens in a browser; the smaller
Study card is the optional inline component. Neither creates a custom sidebar.

For missing runtime configuration, follow the plugin's README and
`scripts/configure.py` using an environment with `pyscf-agent[mcp]`. No additional
LLM API key is needed for this structured MCP workflow.
