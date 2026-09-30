# CAFES

CAFES is a reviewable workflow platform for molecular, periodic, and
model-Hamiltonian calculations built around PySCF. It turns scientific requests
into validated task or study specifications, executes numerical work through
trusted backends, and preserves structured results and reproducible artifacts.

CAFES is distributed as the Python package `pyscf-agent`. Installation commands,
Python imports, plugin IDs, configuration paths, and data schemas retain their
existing names.

## Interfaces

One Web server provides three connected interfaces:

| Path | Interface | Purpose |
| --- | --- | --- |
| `/` | Calculation Assistant | Prepare, approve, run, and analyze one calculation. |
| `/model-hamiltonian-builder/` | Model Hamiltonian Builder | Define finite lattices, sites, bonds, and interactions. |
| `/computational-study/` | Study Planner | Run static or adaptive multi-case calculations. |

The same application services are available through the CLI and local, direct
Slurm, or SSH-to-Slurm executors. An optional [MCP interface](docs/guides/mcp.md)
exposes task validation, submission, status, collection and cancellation, plus
static/adaptive Study planning, background agent execution, collection and
review, scientific analysis and dataset preparation. One Study start runs the
existing agent workflow through completion or required review. Reports and wiki
pages are available as resources.

A local [Codex plugin](plugins/pyscf-agent/README.md) packages these MCP tools
with one workflow skill. Its configured Python environment and calculation
directory stay outside the installed plugin cache. The [saved Study workbench](docs/guides/study-workbench.md)
creates or opens the same Study ID in Codex's browser and shares saved review,
background execution, collection and scientific analysis with MCP. Browser
Build Plan saves the inputs before any execution.
The plugin's `open_workbench` tool starts or reuses the local WebUI and returns
a direct Study link using that same configured environment and output root.
`show_study` provides a compact [MCP Apps Study card](docs/guides/mcp-study-card.md)
with status refresh and a workbench button. It also returns structured data on
hosts without inline rendering support; Codex inline host acceptance is pending.
Independent Runs have a [Task monitor](docs/guides/task-monitor.md) with automatic
status refresh and observed solver progress, shared by the browser and MCP.

## Core Capabilities

- **Molecules:** HF/DFT, MP2, CCSD, CCSD(T), FCI, CASCI/CASSCF,
  ground-state SC-NEVPT2, genuine UNO active-space review, explicit DF-CASSCF,
  and optional block2 DMRG with Conventional MPO, initial whole/split PM/Boys
  localization, Fiedler ordering, and explicit frozen CASSCF orbitals.
- **Periodic systems:** POSCAR/CIF input, gamma-point or k-mesh HF/DFT, band
  paths, occupations, structured cell artifacts, and an optional approval-
  gated HF+DMFT route with automatic IAO/IAO+PAO subspace preparation.
- **Model Hamiltonians:** Hubbard-style finite clusters, exact diagonalization,
  MP2/coupled cluster, optional block2 DMRG, and optional libDMET embedding.
- **Studies:** full-grid scans, correlation diagnostics, method routing,
  approval gates, recovery, anomaly review, plots, and raw plot data.
- **Molecular datasets:** QH9-format-compatible B3LYP/def2-SVP NVE sampling,
  Fock/overlap artifacts, sample and rejection indexes, and explicit dataset
  generation on the execution target followed by optional local collection.
- **Execution:** local processes, direct Slurm, stateless SSH-to-Slurm, job
  arrays for independent cases, structured `TaskReport` results, and optional
  source-matched remote releases. Web local processes retain cancellation after
  restart and support an optional hard wall-time limit.

See the [documentation index](docs/README.md) and the maintained
[architecture](docs/architecture.md) for supported workflows and ownership.

## Install

Python 3.10 or newer is required. From a source checkout, the tested Conda
setup is:

```bash
export CONDA_SOLVER=classic
conda env create --file config/environments/conda.yml
conda activate pyscf-agent
python configure.py install
pyscf-agent-configure check
```

Virtual environments, optional providers, and offline server bundles are
covered in the [installation guide](docs/guides/installation.md).

## Run

Start the Web application:

```bash
pyscf-agent-web --port 8000
```

Inspect the command-line interfaces:

```bash
pyscf-agent --help
pyscf-computational-study --help
pyscf-agent-benchmark --help
```

Run a structured study:

```bash
pyscf-computational-study \
  computational_study_agent/examples/model_u_sweep.json \
  --postprocess
```

Local execution is the default. See the
[remote execution guide](docs/guides/remote-execution.md) for Slurm and remote
Web/CLI profiles.

## Reproducibility

Calculations write normalized inputs, generated PySCF scripts, logs, structured
reports, numerical arrays, and plotting data under
`${XDG_DATA_HOME:-~/.local/share}/pyscf-agent/runs/` by default.
`--work-dir` and `PYSCF_AGENT_RUNS_DIR` override this location. Saved Study
discovery also checks an existing checkout's `./runs/` when using the default;
old calculations stay in place. Private
configuration and run artifacts are excluded from releases and source snapshots.
Large orbital guesses use lossless array files, and new Study checkpoints keep
TaskReport references and compact status summaries. Status reads do not load
numerical arrays; existing inline records remain readable. See the [architecture guide](docs/architecture.md) for storage ownership.

## Documentation

- [Documentation index](docs/README.md)
- [Architecture and ownership](docs/architecture.md)
- [Installation](docs/guides/installation.md)
- [Remote and Slurm execution](docs/guides/remote-execution.md)
- [Optional LLM configuration](docs/guides/llm-configuration.md)
- [Python API boundaries](docs/guides/python-api.md)
- [Verification and distribution](docs/guides/verification-and-distribution.md)
- [Configuration layout](config/README.md)
- [Distribution layout](distribution/README.md)
- [LLM Wiki maintenance](agent_knowledge/README.md)

## Verify

```bash
python3 -B -m unittest discover -s tests -t . -p 'test*.py'
```

Scientific benchmarks and clean-wheel checks are described in the
[verification guide](docs/guides/verification-and-distribution.md).

## Author

Tenghui Li, Rutgers University, tenghui.li@rutgers.edu
