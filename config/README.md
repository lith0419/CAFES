# Configuration

This directory contains developer/environment inputs, not runtime templates or
private configuration.

- `environments/conda.yml` defines the tested Conda environment.

Runtime templates are distributed once under `pyscf_agent/resources/templates`
and created with `pyscf-agent-configure init-llm` or
`pyscf-agent-configure init-config`. Local files such as `llm.env` and
`.pyscf-agent/*.ini` are ignored by Git and excluded from distributions.
