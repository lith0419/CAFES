# Codex Plugin: Packaging And Study Acceptance

Inspection date: 2026-09-16. Baseline: `c620f3b`; development branch:
`codex/codex-plugin`. Agent version remains `0.2.0`; plugin version is `0.1.0`.

## Implemented Boundary

The [plugin package](../../plugins/pyscf-agent/README.md) contains the supported
Codex compatibility manifest, a bundled stdio MCP configuration, one workflow
skill, and runtime configuration/launch helpers. It reuses the existing
thirteen tools and application services. No additional scientific execution,
adaptive orchestration, retry state or task database is introduced.

The configure helper uses the existing MCP argument parser and executor
configuration validation. It records the selected virtual-environment Python,
absolute output root and applicable remote/Slurm INI paths in trusted local
shell configuration outside the plugin cache. It does not contact SSH or
submit work. The launcher replaces its process with the existing MCP server,
preserving protocol-only stdout and process lifecycle behavior.

The skill directs Codex to Registry/Wiki resources, saved Study IDs, existing
review actions and agent-owned execution. It distinguishes scheduler state,
scientific convergence and result quality. Current limitations remain explicit:
there is no all-Study listing, Study-wide cancellation tool, generic binary
transfer, Study-linked dashboard or inline component in this increment.

## Local Installation

The development source remains in this repository. A personal-marketplace
source link points to it; Codex installs its own cached copy. The local plugin
is registered as `pyscf-agent@personal` and was verified installed and enabled
through the Codex CLI. Source and installed cache match for all six plugin
files, including the executable launcher.

The configured local runtime is a separate Python 3.12 environment with the
agent installed editable and its MCP extra. It uses the existing repository
`runs/` directory. Local configuration, environment, plugin cache and numerical
artifacts are excluded from version control. This environment does not install
optional block2 or libDMET. Remote profiles can be selected through the same
configure helper, but this acceptance submits no Amarel work.

New Codex tasks load the installed skill and tools. Installing a plugin does
not replace the already-active conversation's tool list. Actual model tool
selection in a new task remains a user-facing follow-up check; protocol and
execution acceptance are recorded below.

## Verification

See the compact verification evidence (author archive: `reports/verification/codex-plugin-2026-09-16.json`).

- Plugin manifest and skill validators pass; shell syntax and `pip check` pass.
- Plugin/MCP regression: **30 tests passed, no skips**, in 43.034 seconds,
  including opt-in numerical acceptance. The copied-plugin test executes two
  H2 cases, reconnects, reviews and retries one case, and retains two Tasks
  with three Runs and attempt counts `[2, 1]`.
- Codex exposes the installed plugin's stdio transport with its cached working
  directory. A real MCP client using that exact command discovers 13 tools
  and 43 Wiki pages, validates H2, starts a two-case Study, closes the
  connection, reconnects and collects both successful calculations.
- Installed-plugin energies are `-1.1167593073964255` Ha (STO-3G) and
  `-1.126755317196932` Ha (6-31G). Repeated collection leaves exactly two Runs.
- The numerical environment uses Python 3.12.14, PySCF 2.13.1 and MCP 2.2.0,
  with OMP/OpenBLAS/MKL thread counts set to one during acceptance.

The focused command is:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYSCF_AGENT_TEST_MCP_NUMERICAL=1 \
python -B -m unittest tests.pyscf_agent.test_codex_plugin \
  tests.pyscf_agent.test_mcp_server tests.pyscf_agent.test_mcp_studies -v
```

## Follow-Up Increment

The [saved Study workbench increment](2026-09-16-study-workbench.md) implements
discovery and opening by ID, with saved start/review/collect actions delegated
to the same application methods used by MCP.
Inline MCP Apps UI is a later capability test, not a prerequisite for opening
the existing workbench in Codex's browser.
