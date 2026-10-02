# Architecture maintenance

Runtime serialization, timeout, Study ownership and provider-error policies are
documented in [Runtime reliability boundaries](runtime-reliability.md).

## Local checks

The same checks run locally and in GitHub Actions:

```sh
python -m pip install -e '.[dev]'
ruff check .
mypy
python -m unittest discover -s tests -t . -p 'test*.py'
python -m pyscf_agent.configure verify-install --output /tmp/pyscf-agent-wheel-check.json
```

Ruff checks correctness, unused imports (F401), and misplaced imports (E402),
without imposing a repository-wide format change. Mypy strictly checks the five explicitly listed infrastructure
modules in `pyproject.toml`; other modules are not claimed as type-checked.
Architecture tests scan imports inside functions as well as at module scope.
Node must be installed to run browser-state tests. Process-control and HTTP tests
need a normal local environment that permits process inspection and loopback
listeners. Optional scientific providers keep their existing skip conditions.

## Run directories

New calculations default to `$XDG_DATA_HOME/pyscf-agent/runs`, or
`~/.local/share/pyscf-agent/runs` when `XDG_DATA_HOME` is unset. Precedence is:

1. An explicit work directory supplied by the caller.
2. `PYSCF_AGENT_RUNS_DIR`.
3. The user data default.

When browsing the ordinary default, saved-Study discovery also checks `./runs`
in the current checkout. Entries carry their actual work directory; opening,
collecting or retrying an old Study continues there. An explicit custom work
directory or environment override remains a confined search root. Duplicate IDs
in separate roots retain separate locations in the saved list; an ID-only open
prefers the new default. Pass `--work-dir /old/root` for an unambiguous old root.

Nothing migrates existing runs, remote release caches or private configuration
automatically. Use the existing explicit remote work-directory configuration on
clusters. Resource/path lookup itself does not create directories.

## Supported entry points

Use these entry points:

| Area | Module or command |
| --- | --- |
| Calculation CLI | `pyscf_agent.cli` / `pyscf-agent` |
| Web server | `pyscf_agent.web.server` / `pyscf-agent-web` |
| Web API and UI | `pyscf_agent.web.api`, `pyscf_agent.web.ui` |
| Hamiltonian datasets | `computational_study_agent.datasets.hamiltonian.{contracts,planner,finalize,postprocessing}` |
| Source configuration | `python -m pyscf_agent.configure` |
| Installed configuration | `pyscf-agent-configure` |

## Dataset tools and assets

Reusable QM9 preparation tools live in `tools/datasets/qm9`; see the
[dataset tools guide](../../tools/datasets/README.md) for commands and dependencies.
Keep source datasets, temporary caches and calculation outputs in configured
locations outside the checkout.

Asset hashes use bounded caching. Filesystem identity, size, modification time,
and change time invalidate the cache for editable installations. Non-filesystem
package resources are treated as immutable during the server process lifetime.
