# CAFES Documentation

This directory is the single entry point for human-readable project
documentation.

Guides and development logs use `your_username`, `your_node`, and
`cluster.example.edu` as placeholders for private account, node, and host names.
Replace them with your own authorized settings before using the commands.

## Current Documentation

- [Lutein and Honeycomb report examples](../examples/README.md)

- [Architecture and ownership](architecture.md)
- [Architecture maintenance and migration](guides/architecture-maintenance.md)
- [Architecture review and migration priorities](architecture-review.md)
- [Platform status](status/strong-correlation-agent-platform.md)
- September 21 development and verification report (author archive: `reports/development-status-2026-09-21.md`)
- September 21 lutein results and virtual-space accuracy (author archive: `reports/lutein-overnight-results-2026-09-21.md`)
- File-backed task storage verification (author archive: `reports/file-backed-task-storage-2026-09-21.md`)
- Earlier lutein workflow status (September 14) (author archive: `reports/lutein-workflow-status-2026-09-14.md`)
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
- [Repository storage and archived dataset locations](guides/repository-storage.md)

## Development History

- [2026-09-24: CCSD impurity SCF beta and Hubbard comparison](development/2026-09-24-ccsd-impurity-beta.md)
- [2026-09-23: Energy-stratified trajectory frame selection](development/2026-09-23-energy-stratified-frame-selection.md)
- [2026-09-21: File-backed numerical inputs and compact Study state](development/2026-09-21-file-backed-task-storage.md)
- [2026-09-20: Frozen CASSCF orbitals](development/2026-09-20-frozen-casscf-orbitals.md)
- [2026-09-20: Split active-space localization](development/2026-09-20-split-active-localization.md)
- [2026-09-20: Independent Task monitor and automatic refresh](development/2026-09-20-task-monitor.md)
- [2026-09-20: CCSD impurity ADIIS/CDIIS and convergence handoff](development/2026-09-20-ccsd-impurity-diis.md)
- [2026-09-19: block2 MPO selection and native-interface comparison](development/2026-09-19-block2-mpo-comparison.md)
- [2026-09-18: Wiki implementation and scientific evidence alignment](development/2026-09-18-wiki-code-alignment.md)
- [2026-09-17: Remaining fallback cleanup and runtime/evidence consistency](development/2026-09-17-remaining-fallback-cleanup.md)
- [2026-09-17: Remaining fallbacks and specific policies audit](development/2026-09-17-remaining-fallback-audit.md)
- [2026-09-17: Saved Study recovery actions](development/2026-09-17-study-recovery-actions.md)
- [2026-09-17: Postprocessing contracts and analysis context](development/2026-09-17-postprocessing-contract-cleanup.md)
- [2026-09-17: WebUI context and postprocessing audit](development/2026-09-17-webui-context-postprocessing-audit.md)
- [2026-09-17: Planner operation types](development/2026-09-17-planner-operation-types.md)
- [2026-09-17: Scientific input and recovery boundaries](development/2026-09-17-input-and-recovery-boundaries.md)
- [2026-09-17: Planner draft contract and failure semantics](development/2026-09-17-planner-draft-contract.md)
- [2026-09-17: Wiki query relevance and passage selection](development/2026-09-17-wiki-retrieval.md)
- [2026-09-16: Planner clarification without a plan](development/2026-09-16-planner-clarification.md)
- [2026-09-16: Read-only MCP Apps Study card](development/2026-09-16-mcp-study-card.md)
- [2026-09-16: Plugin workbench startup and Study links](development/2026-09-16-workbench-launcher.md)
- [2026-09-16: Saved browser planning and frontend orchestration removal](development/2026-09-16-saved-study-planning.md)
- [2026-09-16: Saved Study workbench and MCP interoperation](development/2026-09-16-study-workbench.md)
- [2026-09-16: Codex plugin packaging and Study acceptance](development/2026-09-16-codex-plugin.md)
- [2026-09-14: DMET mean-field and interacting-bath consistency](development/2026-09-14-dmet-mean-field-consistency.md)
- [2026-09-14: DMET density-fit audit and shared solver correction](development/2026-09-14-dmet-solver-fitting-audit.md)
- [2026-09-14: Source consolidation and verification](development/2026-09-14-source-consolidation.md)
- [2026-09-13: CAS workflow ordering and small-space acceptance](development/2026-09-13-cas-workflow-acceptance.md)
- [2026-09-10: Optional SCF stability and genuine UNO candidates](development/2026-09-10-uno-stability-policy.md)
- [2026-09-10: SCF retry from the last reference density](development/2026-09-10-scf-retry-restart.md)
- [2026-09-09: Explicit DF-CASSCF and lutein acceptance](development/2026-09-09-density-fitting-casscf.md)
- [2026-09-06: Single-task MCP interface](development/2026-09-06-mcp-single-task-interface.md)
- [2026-09-06: Study MCP submission and collection](development/2026-09-06-mcp-study-interface.md)
- [2026-09-07: Thin MCP agent interface](development/2026-09-07-mcp-agent-interface.md)
- [2026-09-07: Saved review, analysis and dataset MCP workflows](development/2026-09-07-mcp-study-workflows.md)
- [2026-09-06: Study, Task, and Run organization](development/2026-09-06-study-task-run-organization.md)
- [2026-09-06: Study task-management simplification](development/2026-09-06-study-task-management-simplification.md)
- [2026-09-06: Study child-task references and current-result merging](development/2026-09-06-study-child-task-references.md)
- [2026-09-06: DMET option and quality-evidence validation](development/2026-09-06-dmet-option-and-quality-validation.md)
- [2026-09-06: Local process control and retry acceptance](development/2026-09-06-local-process-and-retry-acceptance.md)
- [2026-09-04: Study retry, collection, and migration status](development/2026-09-04-study-retry-and-collection.md)
- [2026-08-11: block2 and continuation workflows](development/2026-08-11-block2-continuation.md)
- [2026-08-12: DMRG-CASSCF](development/2026-08-12-dmrg-casscf.md)
- [2026-08-16: block2 low-energy workflows](development/2026-08-16-block2-low-energy-workflows.md)
- [2026-08-16: runtime, block2, and remote execution](development/2026-08-16-runtime-block2-remote.md)
- [2026-08-23: fcDMFT HF+DMFT provider boundary](development/2026-08-23-fcdmft-hf-dmft.md)
- [2026-08-23: fcDMFT periodic GW and GW+DMFT](development/2026-08-23-fcdmft-gw-dmft.md)
- [2026-08-23: shared molecular active-space probe](development/2026-08-23-shared-active-space-probe.md)
- [2026-08-25: conditional active-space probe and study policy](development/2026-08-25-study-active-space-policy.md)
- [2026-09-03: QH9 MD datasets and runtime identity](development/2026-09-03-qh9-md-dataset-and-runtime-identity.md)

## Directory Boundaries

- `docs/` contains documentation intended for people to read and maintain.
- `reports/` contains generated benchmark and verification evidence.
- `agent_knowledge/` contains source and compiled knowledge consumed by the
  LLM-assisted runtime.
- Component-level `README.md` files remain beside the component they describe.

Architecture figures used in papers or presentations should be maintained with
their source document rather than copied into this repository as stale exports.
