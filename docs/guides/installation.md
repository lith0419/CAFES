# Installation

CAFES requires Python 3.10 or newer. PySCF and the ordinary runtime
dependencies are declared in `pyproject.toml`.

## Python Wheel

Install a built wheel, or install directly from a source checkout:

```bash
python -m pip install dist/pyscf_agent-0.2.0-py3-none-any.whl
# Source checkout alternative:
python -m pip install .
pyscf-agent-configure check
pyscf-agent-configure verify-install
```

The installed `pyscf-agent-configure` command and its configuration templates
are package resources; these commands do not require a Git checkout.

## Conda Environment

Create the tested Python 3.10 environment without inheriting user channels:

```bash
export CONDA_SOLVER=classic
conda env create --file config/environments/conda.yml
conda activate pyscf-agent
python -m pyscf_agent.configure install
pyscf-agent-configure check
```

The final command installs the project in editable mode and skips compiled
dependencies already present in Conda. Use a role to validate host-specific
commands:

```bash
pyscf-agent-configure check --role local
pyscf-agent-configure check --role server
pyscf-agent-configure check --slurm
```

Preview source-checkout installation commands with
`python -m pyscf_agent.configure install --dry-run`.
Missing compiled Conda dependencies are installed one at a time by default;
`--batch` requests one combined solve.

After a successful Linux installation, an exact platform lock avoids future
dependency solving:

```bash
conda list --explicit > conda-linux-64.lock
conda create --yes --name pyscf-agent --file conda-linux-64.lock
```

## Virtual Environment

For a manual non-Conda installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

## Online Locked Installation

For a Linux x86_64 host that can reach PyPI, use the lock matching the active
Python minor version. The lock is already a complete dependency closure, so
`--no-deps` disables further dependency solving:

```bash
python3.12 -m venv ~/.venvs/pyscf-agent
source ~/.venvs/pyscf-agent/bin/activate
python -m pip install --no-deps \
  -r distribution/offline/requirements-linux-x86_64-py312.lock
python -m pip install --no-deps dist/pyscf_agent-0.2.0-py3-none-any.whl
python -m pip check
```

Use `py310`, `py311`, `py312`, `py313`, or `py314` to match the interpreter.
The Python 3.14 lock requires glibc 2.28 or newer; the earlier locks require
glibc 2.17 or newer.

Optional providers are installed separately:

```bash
python -m pip install -e '.[block2]'
python -m pip install -e '.[embedding-runtime]'
python -m pip install -e '.[dmft-runtime]'
python -m pip install 'libdmet @ git+https://github.com/gkclab/libdmet_preview.git@main'
```

The `embedding-runtime` and `dmft-runtime` extras install bridge dependencies
only. They do not install libDMET or fcDMFT; each provider must be installed
separately on every execution host.

fcDMFT currently has no standard package metadata in the supported source
checkout. Install its bridge dependencies with `.[dmft-runtime]`, then add the
directory that contains the `fcdmft/` Python package to `PYTHONPATH` on every
execution host:

```bash
export PYTHONPATH=/path/to/fcdmft/source-root:${PYTHONPATH:-}
python -c "from pyscf_agent.providers.fcdmft import fcdmft_availability; print(fcdmft_availability())"
```

The provider performs side-effect-free discovery during planning and imports
the MPI-enabled fcDMFT solver only inside the numerical execution stage.

## Locked Linux Bundles

The committed Linux locks cover CPython 3.10-3.14. Versions 3.10-3.13 require
glibc 2.17 or newer; Python 3.14 requires glibc 2.28 or newer. Build one target
bundle on an internet-connected macOS or Linux machine:

```bash
python -m pip install 'build>=1.2,<2'
./distribution/offline/package.sh --python-version 3.10
```

The archive under `dist/` contains a source snapshot, the project wheel, and
binary dependencies compatible with the target recorded in its filename and
manifest. Transfer and install it without contacting PyPI or invoking a Conda
solve:

```bash
tar -xzf pyscf-agent-*-linux-x86_64-py310-*.tar.gz
cd pyscf-agent-*-linux-x86_64-py310-*
./install_offline.sh --prefix "$HOME/.venvs/pyscf-agent" --role server
source "$HOME/.venvs/pyscf-agent/bin/activate"
hash -r
pyscf-agent-configure check --role server
```

To use an active Conda environment with the same Python minor version:

```bash
./install_offline.sh --current --check-slurm
```

The installer prints the absolute directory containing the installed commands.
If an already-running shell still reports `command not found`, refresh its
command cache with `hash -r`. Diagnose the active interpreter and bypass
`PATH` lookup with:

```bash
command -v python
python -c 'import sys, sysconfig; print(sys.executable); print(sysconfig.get_path("scripts"))'
"$(python -c 'import sysconfig; print(sysconfig.get_path("scripts"))')/pyscf-agent-configure" check
```

The installer first validates every bundled wheel against
`WHEELHOUSE_SHA256SUMS.txt`, then validates the platform, uses only bundled
wheels with dependency resolution disabled, runs `pip check`, verifies runtime
imports, and optionally checks Slurm commands.

Regenerate all released locks intentionally with:

```bash
python -m pip install 'uv>=0.8,<1'
./distribution/offline/generate-locks.sh
```

The generator accepts `--python-version 3.15` as a readiness check. As of
2026-08-24, h5py does not publish a CPython 3.15 Linux wheel, so no installable
3.15 lock or bundle is released yet.
