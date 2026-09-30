# CAFES for Codex

This local desktop plugin bundles one workflow skill, scientific stdio MCP
tools, browser launchers, and read-only Study/Task cards with app-only refresh
endpoints. Calculations, scientific review, static/adaptive Studies,
retries and analysis remain in the agent's application services.

The conversational interface and Study-linked WebUI share the same agent.
The full workbench opens in Codex's browser. `show_study(study_id)` also exposes
a read-only [MCP Apps card](../../docs/guides/mcp-study-card.md) for compatible
hosts. Actual Codex inline rendering still needs host acceptance in a new task;
the card protocol and browser test host have been verified separately.

For an independent calculation, use `show_task(handle)` or
`open_task_monitor(handle)` with its saved JobHandle. The same Task monitor
refreshes automatically while active, showing execution state, scientific
outcome, failure messages and available solver progress. See the
[Task monitor guide](../../docs/guides/task-monitor.md) for evidence boundaries
and remote-runtime requirements.

## Configure the runtime

Use Python 3.10 or newer with the agent and its MCP extra installed. From the
repository root, for example:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[mcp]'
.venv/bin/python plugins/pyscf-agent/scripts/configure.py \
  --executor local --work-dir "$HOME/pyscf-agent-runs"
```

The configuring interpreter becomes the runtime interpreter, including its
virtual environment. The script validates the existing executor configuration,
then writes a small local launch command to
`~/.config/pyscf-agent/codex-mcp.sh` (or `$XDG_CONFIG_HOME/pyscf-agent/`). It does
not submit a calculation, contact SSH, or install numerical dependencies.
An existing configuration requires `--force` to replace it.

For Amarel, use your existing client profile:

```sh
.venv/bin/python plugins/pyscf-agent/scripts/configure.py \
  --executor remote --remote amarel \
  --remote-config /absolute/path/to/remote.ini \
  --work-dir "$HOME/pyscf-agent-runs" --force
```

Direct Slurm uses the same `--executor slurm`, `--slurm-config` and
`--slurm-profile` flags as the existing MCP command. Work directories and INI
paths are made absolute during configuration. Remote numerical files retain
the existing executor's remote-workspace rules.

The launch command is trusted local shell configuration. Keep it on your own
machine; it contains machine-specific paths. The plugin cache contains only
portable files. `PYSCF_AGENT_CODEX_CONFIG` can select another configuration;
when using `--config-file`, set that environment variable for the MCP process
as well. Changing the configuration takes effect on the next MCP connection;
use the original target and work directory when recovering an existing Study.

No additional LLM API key is required. PySCF, block2 and other scientific
providers are managed in the calculation environment, separately from plugin
installation. A local editable installation follows this checkout; use an
installed wheel when you need a fixed source version.

## Install in Codex

The plugin uses the supported Codex compatibility layout:

```text
pyscf-agent/
  .codex-plugin/plugin.json
  .mcp.json
  scripts/configure.py
  scripts/start-mcp.sh
  skills/pyscf-agent/SKILL.md
```

Add this folder to a local personal marketplace using Codex's plugin-creator,
then install `pyscf-agent@<marketplace-name>`. Keep this repository's plugin
directory as the development source. The plugin can be copied into the
personal marketplace; numerical code and outputs remain outside that copy.
Codex installs a cached copy, so source edits require reinstalling the plugin.
Start a new Codex task after installation to load its skill and MCP tools.

Try: “Use CAFES to prepare a two-case H2 basis-comparison Study.”
The agent returns a Study ID. Use that ID for status, collection, analysis,
review and an authorized retry. A missing configuration produces an actionable
startup error instead of starting in a plugin-cache output directory.

## Open the workbench

Ask: “Open this Study in the PySCF workbench.” Codex calls
`open_workbench(study_id)` and opens the returned URL. Omit the Study ID to open
the home page, list saved Studies, or build a new one. Opening submits no jobs.

The tool starts the existing Web server on an available loopback port using
the configured runtime, output root and executor. It reuses a running service
for the same configuration, including after MCP reconnects. Once started,
Study responses include `workbench_url`. Call `open_workbench` again to recover
after a stopped server or host restart. No fixed port or second configuration
file is needed; the existing `codex-mcp.sh` configuration remains compatible.

Launcher state and logs live under `<work-dir>/.workbench/`, outside the plugin
cache. This is Web-process metadata, not another Study store. The response
includes the process ID and log path for troubleshooting. Stop that Web process
when no longer needed, or before changing an editable agent's Python source;
the next call starts the updated runtime. Stopping the Web service does not
cancel the agent's independent calculation processes. When MCP itself runs on
another machine, the loopback link needs that host's existing port forwarding.

Official references: [plugin packaging](https://developers.openai.com/plugins/build/plugins)
and [Codex MCP](https://learn.chatgpt.com/docs/extend/mcp).
See the project's [MCP guide](../../docs/guides/mcp.md) for the complete tools,
resources, review semantics and execution boundaries.

## Verification

With the MCP extra installed, from the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYSCF_AGENT_TEST_MCP_NUMERICAL=1 \
python -B -m unittest tests.pyscf_agent.test_codex_plugin -v
```

The acceptance test copies the plugin into a separate installation directory,
starts its configured stdio launcher, runs a two-case H2 Study, reconnects,
reviews and reruns one case, and checks stable Task identities and three total
Runs. This checks the plugin launch path; actual Codex discovery/installation
is verified separately.
