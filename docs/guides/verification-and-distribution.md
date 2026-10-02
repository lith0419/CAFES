# Verification And Distribution

## Test Suite

```bash
python3 -B -m unittest discover -s tests -t . -p 'test*.py'
python -c "import pyscf_agent, computational_study_agent; print('ok')"
python -m pyscf_agent.configure verify-install --output reports/verification/clean-wheel.json
pyscf-agent-configure verify-install --output reports/verification/installed-package.json
```

From a source checkout, `python -m pyscf_agent.configure verify-install` builds a wheel,
installs it in a temporary environment, and imports it outside the source tree.
From an installed wheel, `pyscf-agent-configure verify-install` validates that
installation in place. Both modes check schemas, CLI entry points, Web assets,
packaged templates, the curated Wiki, benchmark data, and lightweight
scientific smoke calculations. Use `--wheelhouse PATH` with the source mode for
isolated offline dependency installation.

Source verification needs the build tools (`python -m pip install 'build>=1.2,<2'
wheel`) in the selected environment. Without `--wheelhouse`, it reuses that
interpreter's scientific dependency directories, including when launched from
a virtual environment, while installing and importing a fresh agent wheel in
the temporary environment. This does not certify a target-specific offline
wheelhouse. Failed scientific smoke results are retained in the report.

## Scientific Benchmarks

```bash
pyscf-agent-benchmark \
  --work-dir /tmp/pyscf-agent-benchmarks \
  --output reports/benchmarks/formal-benchmark.json --pretty
```

Optional medium-scale block2 campaigns are enabled explicitly:

```bash
pyscf-agent-benchmark --include-scaling \
  --work-dir /tmp/pyscf-agent-benchmarks \
  --output reports/benchmarks/block2-scaling.json --pretty
```

The native silicon `G0W0@PBE` benchmark follows the fcDMFT `examples/Si/si_gw.py`
geometry and `4x4x4` k mesh. It is excluded from routine runs because it is a
large periodic calculation and must be enabled explicitly:

```bash
PYSCF_AGENT_RUN_FCDMFT_BENCHMARKS=1 \
  pyscf-agent-benchmark fcdmft-si-g0w0 \
  --work-dir /scratch/$USER/pyscf-agent-benchmarks \
  --output reports/benchmarks/fcdmft-si-g0w0.json --pretty
```

This benchmark strictly checks the public-backend execution contract, the
native example settings, GW total-energy semantics, and registered self-energy
artifacts. Until a server result is archived, its silicon gap check is labeled
as a physical sanity range rather than a frozen numerical-accuracy reference.

Reports retain references, units, tolerances, environment information, and
metrics. Optional-provider cases are recorded as skipped when unavailable.

## Python Release

Build the standard wheel and sdist declared by `pyproject.toml`:

```bash
python -m pip install 'build>=1.2,<2'
python -m build
```

These artifacts are the formal Python release and can be installed without a
repository checkout.

## Source Snapshot

Create a timestamped research/development snapshot of the current working tree:

```bash
./distribution/package.sh
```

Create a version-specific Linux x86_64 offline bundle:

```bash
python -m pip install 'build>=1.2,<2'
./distribution/offline/package.sh --python-version 3.10
```

The snapshot is not a Python sdist. The offline bundle uses the committed,
fully pinned lock for the requested Python minor version and verifies wheel
hashes before installation. Archives and checksums are written to `dist/`.
Runs, private configuration, Git metadata, environments, caches, and existing
build outputs are excluded.

Regenerate the Python 3.10-3.14 lock matrix from its shared direct inputs:

```bash
python -m pip install 'uv>=0.8,<1'
./distribution/offline/generate-locks.sh
```

The resolver accepts only target-compatible binary wheels. Python 3.15 is not
part of the released matrix until h5py and the remaining scientific stack
publish compatible Linux wheels.

## Cleanup

```bash
./clean.sh
```

This removes caches and build intermediates, then verifies the selected scope.
`--check` audits without deleting. Additional scopes require explicit flags:

- `--all`: local environments and Node dependencies.
- `--dist`: prepared distribution packages.
- `--wiki`: generated wiki pages and curation state.
- `--scratch`: `block2-scratch/` solver files.
- `--runs`: calculation logs and artifacts; refuses the entire cleanup if
  a Git `.bundle` backup is present under `runs/`.

Default cleanup preserves all of those directories.
Before creating a release, use:

```bash
./clean.sh --yes --release
```

Release mode also requires a clean Git source tree. It reports tracked or
untracked source changes but never discards them. Private `.pyscf-agent/`
configuration and `llm.env` are always preserved.
