# Task Monitor

Use `show_task(handle)` for an existing standalone Task, or any Study child Run
whose exact JobHandle is available. It returns a compact status projection and
an MCP Apps resource for hosts that support inline components. Use
`open_task_monitor(handle)` for the same monitor in the local browser workbench.
Opening or refreshing never submits, retries, collects, analyzes or cancels work.

The browser URL retains the exact handle and execution target in its fragment.
Reloading reconnects to that Run; it creates no Study, Run, or alternate task store.
The execution target must be the one that owns the original handle.

## What Is Displayed

- Execution state and scientific task status separately. A completed worker can
  have a failed or unconverged TaskReport.
- Elapsed time, report availability, saved task energy when reported, and failure
  messages. No final energy is invented from an intermediate solver sweep.
- Observed block2 sweep index, M, energy change and discarded weight when a
  readable text log contains them. The displayed completed-sweep metrics remain
  attached to their own sweep header, even while the next sweep is in progress.
- Recent bounded text-log excerpts and their modification times.

The view refreshes every 15 seconds while active and visible. It stops on a
terminal executor status, can be paused, and backs off to at most 60 seconds
between checks after errors. Manual refresh remains available after completion.
Closing the view or hiding it stops automatic polling; closing it never stops a
calculation. Browser refresh is an HTTP request; embedded refresh is an app-only
MCP call. Neither invokes an LLM nor feeds every log update into the conversation.

## Evidence And Limits

`CalculationApplicationService.inspect_task` delegates to the configured executor.
Local, local-process and direct Slurm executors read their own saved evidence;
SSH-to-Slurm uses one `inspect-task` RPC, with the same projection on the remote
host. The remote runtime must include this operation. An older remote runtime
returns an explicit error; this change does not deploy software to a cluster.
Third-party executors without the optional inspection port return status only
with a notice that solver details are unavailable.

Inspection reads at most eight recent log tails of 64 KiB each, returning at most
30 lines / 6,000 characters per log. It searches the Run root and two directory
levels for text logs, not binary checkpoints or external symlink targets. Saved
reports up to 16 MiB are projected to compact fields; larger reports need the
existing full-report collection interface. No raw arrays, full reports or full
logs are sent on each refresh.

Solver fields are **last observed evidence**, not a reliable indication of the
current phase after that log entry. A provider may buffer its output, stop
printing, or proceed into RDM evaluation. Sweep indices start at zero and reset
between calls/stages. Missing fields remain absent; the monitor does not derive
numerical convergence, progress percentages, or time-to-completion estimates.
Currently only block2's native sweep output has a structured parser. Other
methods still show execution status, saved results and available log excerpts.

The resource is `ui://pyscf-agent/task-v1.html` with MCP Apps metadata. The browser
and MCP resource use the same HTML/CSS/JavaScript. Actual Codex inline rendering
remains host-dependent and is not established by a browser preview. Start a new
Codex task after updating the plugin to discover `show_task`,
`refresh_task_view` (app-only) and `open_task_monitor`.
