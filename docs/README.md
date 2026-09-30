# CAFES Documentation

This directory is the single entry point for human-readable project
documentation.

## Current Documentation

- [Architecture and ownership](architecture.md)
- [Architecture maintenance and migration](guides/architecture-maintenance.md)
- [PySCF example baselines](pyscf_example_baselines.md)

## User Guides

- [Installation](guides/installation.md)
- [Remote and Slurm execution](guides/remote-execution.md)
- [Optional LLM configuration](guides/llm-configuration.md)
- [Python API boundaries](guides/python-api.md)
- [MCP interface](guides/mcp.md)
- [Codex plugin setup](../plugins/pyscf-agent/README.md)
- [Saved Study workbench](guides/study-workbench.md)
- [Selected-case retry actions and previews](guides/selected-case-retries.md)
- [MCP Apps Study card](guides/mcp-study-card.md)
- [Task monitor and automatic progress refresh](guides/task-monitor.md)
- [Verification and distribution](guides/verification-and-distribution.md)

## Directory Boundaries

- `docs/` contains documentation intended for people to read and maintain.
- Benchmark and verification commands write evidence to user-selected output directories.
- `agent_knowledge/` contains source and compiled knowledge consumed by the
  LLM-assisted runtime.
- Component-level `README.md` files remain beside the component they describe.

Architecture figures used in papers or presentations should be maintained with
their source document rather than copied into this repository as stale exports.
