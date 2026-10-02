# Independent Task Monitor — 2026-09-20

## Problem and behavior

Standalone calculations already had persistent JobHandles and reports, but
the workbench and MCP Apps card could only display saved Studies. Checking a
single long-running solver repeatedly required a conversation turn and manual
log inspection. The new monitor opens an existing Run directly and refreshes
its status and available solver metrics without an LLM call.

## Design

- `CalculationApplicationService.inspect_task` is the shared application entry.
  An optional executor inspection port reads existing scheduler state and saved
  evidence. No new task store, coordinator, scientific policy or recovery path.
- Local, local-process and Slurm use `executors/inspection.py`. SSH forwards one
  `inspect-task` RPC to the same Slurm implementation. Third-party executors
  without this optional port explicitly report status-only support.
- The block2 provider owns its sweep-text parser. Active and last-completed
  sweep identities stay separate; a new sweep zero resets stage metrics. Partial
  log lines and results without a preceding header do not invent progress.
- `show_task` and app-only `refresh_task_view` expose that view. The optional
  `open_task_monitor` launcher reuses the existing WebUI configuration.
- `/task-monitor/` and the MCP Apps resource share one HTML/CSS/JavaScript
  implementation, selecting HTTP or MCP transport. The URL fragment carries the
  exact JobHandle and execution target; refresh cannot change Run identity.
- Visible active views poll at 15 seconds; terminal states stop polling. Errors
  preserve the last displayed data and back off up to 60 seconds. Manual pause,
  hidden-page suspension, request serialization, timeouts and teardown are
  transport behavior only, without scientific fallback or automatic resubmission.

## Verification

The focused suite ran 77 tests: 72 passed and five opt-in numerical tests were
skipped. Coverage includes task evidence, MCP discovery/resources, SSH/RPC
routing, existing Study/workbench behavior and the browser/MCP refresh lifecycle.
After the final display refinement, the Task monitor and SSH inspection tests
were rerun successfully. Initial sandbox-only loopback/process restrictions were
resolved by running the integration checks with those local operations enabled.

The installed personal plugin was refreshed and checked through its actual
stdio launcher. It discovers 19 tools, including all three new Task monitor
tools. The existing honeycomb DMRG Run was inspected and opened without any
new calculation; job state, report and solver log hashes were unchanged.

The real page was opened in the Codex in-app browser. It showed completed
execution with failed science, 15m 47s elapsed, M=2000, completed sweep index 7,
|delta E|=2.05e-5 and discarded weight=6.44344e-4. It correctly displayed no
final task energy and stopped automatic refresh on this terminal Run.

Machine-readable evidence: `reports/verification/task-monitor-2026-09-20.json`.

## Remaining boundaries

This is observed log progress, not a solver-wide progress percentage or proof
of convergence. Other solvers retain generic status/report/log display until
they have structured parsers. Study cards keep their existing manual refresh.
The remote RPC route has automated coverage but was not deployed or accepted on
Amarel in this change. Actual native Codex inline card rendering is unverified;
the browser monitor is the verified delivery path. A new Codex task is needed
to discover updated plugin tools and instructions.
