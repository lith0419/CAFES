# Plugin Workbench Startup And Study Links

Inspection date: 2026-09-16. Baseline: `c620f3b`; development branch:
`codex/codex-plugin`. Follows the
[saved planning migration](2026-09-16-saved-study-planning.md).

## Implementation

The stdio CLI now registers `open_workbench(study_id=None)` alongside the
thirteen scientific tools. The tool validates a supplied Study ID against the
existing application service and starts or reuses the existing WebUI. Omitting
the ID opens the workbench home. It returns a browser URL, process ID, output
root, execution target and log path. It does not prepare or execute a Study.

`pyscf_agent.workbench.LocalWorkbench` launches `pyscf_agent_web` using the same
virtual-environment Python, absolute output root and executor arguments as MCP.
The child binds `127.0.0.1` on an available port and remains independent of the
MCP connection. A configuration-specific file lock serializes concurrent starts.
Small files under `<work-dir>/.workbench/` contain the port/PID and Web logs;
Study data stays in its original artifacts. A loopback identity check prevents
reusing a stale port now occupied by another process. Configuration identity
includes the runtime path, arguments and remote/Slurm configuration contents.

`/api/workbench` identifies launcher-managed Web processes. The regular Web
entry point remains usable manually. Study tool responses add `workbench_url`
only while the matching service responds; discovering tools or inspecting a
Study never starts a Web server. The thirteen-tool embedded server remains
available when no launcher is supplied to `create_server`.

The plugin skill directs Codex to call the tool and open its URL with available
browser/panel tools. The installed personal-marketplace plugin was updated to
`0.1.0+codex.20260916184711` using the cachebuster/reinstall flow. Existing
`codex-mcp.sh` configuration remains valid. No credentials, numerical libraries
or additional LLM settings were changed.

## Verification

See machine-readable evidence (author archive: `reports/verification/workbench-launcher-2026-09-16.json`).

- The focused regression exercised **101 test methods** with numerical MCP
  acceptance enabled. One old stdio test still expected thirteen tools; after
  updating it to fourteen, the affected test and three launcher tests all
  passed. The other 100 regression methods passed in the initial run.
- Launcher acceptance copies the plugin to a separate cache-like directory,
  configures paths containing spaces/quotes, prepares a Study, and checks that
  tool discovery and planning create no Web process. Invalid Study IDs do not
  start one. Two concurrent opens share a single process. The WebUI survives
  MCP exit, a fresh connection reuses it, and a stopped service is restarted.
  No calculation Run is created.
- Remote/Slurm argument tests verify absolute configuration paths and profile
  separation without contacting a cluster. Launch failures return the log path
  and do not retry automatically.
- The actual reinstalled Codex transport discovered fourteen tools. It opened
  the previously completed Study `20260916-155300-67cdc98e`, disconnected and
  reconnected, reused the same Web process, and returned the same link in Study
  status. Codex's browser displayed all four successful H2 results. The four
  Run state files remained byte-for-byte unchanged; attempts stayed `[1,1,1,1]`.
- Plugin/skill validators and whitespace checks passed. Source and installed
  plugin files were compared after reinstall.

The environment remains Python 3.12.14, PySCF 2.13.1 and MCP 2.2.0. Tests use
one OMP/OpenBLAS/MKL thread. No Amarel jobs were submitted.

## Boundaries

The returned URL belongs to the machine hosting MCP. A remote MCP host needs
existing port forwarding to make that loopback service reachable from the
desktop. No remote UI acceptance is claimed here. A changed editable Python
source needs the Web service restarted; static files load from the installed
package as before. The URL/port may change after restart; call `open_workbench`
again instead of retaining a dead link.

This implementation provides a browser workbench, not an inline MCP Apps
component or custom Codex sidebar. New Codex tasks pick up the updated skill
and tool list; an already-active task's inventory is not replaced. Actual
model selection of the new tool is a new-task pickup check, separate from the
installed-transport and browser acceptance above.
