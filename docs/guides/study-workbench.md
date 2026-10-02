# Saved Study Workbench

The existing WebUI can create a saved Study or open one prepared through MCP
or the Python agent. Both paths display and continue the same saved plan,
report and review/execution workflow. The workbench reads the same output
directory as MCP. No separate task database or scientific workflow runs in
the browser.

## Start And Open

With the Codex plugin, ask to open a Study in the workbench. The
`open_workbench(study_id)` MCP tool starts or reuses the existing WebUI and
returns its `url`; Codex can open it in the browser. Omit the ID for the home
page and saved Study list. Once the service is running, Study tool responses
also include `workbench_url`. Opening does not execute or collect calculations.

The launcher uses the MCP process's Python environment, output root and
executor configuration. It binds an available `127.0.0.1` port, survives MCP
disconnects, and reuses the same service after reconnect. If a link stops
responding, call `open_workbench` again. State and logs are under
`<work-dir>/.workbench/`; the tool also returns the process ID and log path.
Stop that Web process before updating an editable agent's Python source, then
open again to load the updated runtime. Stopping the Web service does not
cancel independent calculations.

For manual startup from the repository, use the same Python environment,
output root and executor options as MCP. For a local setup:

```sh
.venv/bin/python -m pyscf_agent.web.server \
  --no-open-browser --port 8765 --work-dir ./runs
```

Open `http://127.0.0.1:8765/computational-study/` in Codex's browser. Paste the
Study ID returned by `prepare_study` into **Study ID**, then choose **Open Study**.
**List Studies** discovers saved Studies in the selected work directory. It
reads local metadata; it does not poll schedulers or collect remote results.

A direct link uses `/computational-study/?study=STUDY_ID`. It opens that Study
once, then clears the Study parameters from the workspace address. Refreshing
the page starts a new draft without reopening previous review panels. Use
**Open Study**, **List Studies**, or the original link returned by MCP to return
to saved work. Opening or switching Studies does not rewrite the workspace URL.
For a remote profile, start the
Web server with the same `--executor remote --remote PROFILE --remote-config
CONFIG` options as MCP and select that execution target before opening. Opening
a link does not configure a new remote connection.

## Continue A Saved Study

- **Refresh Status** rereads saved state and checks the configured executor.
- Review buttons save a decision through `StudyApplicationService.review_study`.
  Reopening that Study retains the approved plan or selected retry cases.
- **Run Plan** calls `start_study` using the Study ID. The agent runs in the
  background, independent of the browser connection. Scientific and resource
  gates are enforced by the existing agent.
- **Collect Results** calls `collect_saved_study`. It collects existing work;
  it does not submit missing tasks. A pending review is displayed before further
  collection of the preceding calculation.
- **Analyze Results**, when an LLM is configured, calls `analyze_study` with the
  Study ID. The agent saves scientific diagnostics and review state before
  adding language interpretation. MCP can read the same saved report. Analysis
  does not itself authorize follow-up calculations.

Selected retries keep the original Study and Task identities. Only retried
Tasks gain another Run. The report table shows the current results and attempt
counts. A static retry does not become an adaptive scan merely because its
review metadata is stored in the report's `adaptive` extension.

Saved-plan start and review requests carry an ID and action parameters, not a
browser copy of the full plan/report. The agent's persisted inputs are authoritative.
Use **New Task** to draft different inputs.

## Create A Study In The Browser

Choose **New Task**, set the calculation inputs and execution profile, then
select **Build Plan**. This saves a Study and puts its ID in the page URL.
Planning does not execute calculations or create Runs. Review any scientific
or resource gate, then select **Run Plan** to start the agent. Refreshing the
page restores the saved Study, including approved review decisions.

Static and adaptive studies call the agent's `prepare_study`; the Hamiltonian
Dataset template calls `prepare_dataset`. A CASSCF request that needs active
space selection uses the agent's existing probe/review workflow. The browser
does not implement separate adaptive, retry or continuation scheduling.

LLM-generated plans are draft previews. Select **Build Plan** to save the
chosen inputs before reviewing or running them. Once saved, input controls are
locked; **New Task** creates a separate draft without changing the saved Study.

## Current Boundary

The workbench now uses saved preparation and execution for new and reopened
Studies. The old plan/run HTTP endpoints remain for existing clients, but the
browser no longer calls them. Do not use those legacy endpoints concurrently
to write a Study being run by a background agent. Saved start/review/collect
and analysis use the same invocation lock as MCP.

The full UI is served locally and displayed in Codex's browser. A separate,
read-only [MCP Apps Study card](mcp-study-card.md) can link to this workbench;
it does not embed the whole WebUI or create a custom Codex sidebar. MCP discovery and scientific
status checks do not start a Web service; `open_workbench` requests startup.
When MCP runs on a different host, its localhost URL belongs to that host and
requires existing port forwarding to reach it from the desktop. Live remote
UI acceptance and large numerical calculations are outside this increment's
verification scope.

See [MCP setup](mcp.md) and the [Python API guide](python-api.md)
for launcher configuration and saved Study operations.
