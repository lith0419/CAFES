# PySCF Roadmap Properties and Extensions

## Scope

This document separates registered property/trajectory workflows from additional
PySCF user-guide topics that remain future integrations. The Registry and task
validator determine executable scope.

## Source Coverage

- User guide chunks: `tddft-tdhf`, `adc`, `gw`, `agf2`, `solvent`, `qmmm`,
  `molecular-dynamics`, `electron-phonon`, `geomopt`, `gpu4pyscf`,
  `forge-extensions`, `extensions`, `pprpa`.
- Source wiki pages: `Excited-State Rules`, `Gradients Hessian Optimization`,
  `Solvent QM/MM and Tools`.

## Roadmap Topics

| Topic | Current agent status | Notes |
| --- | --- | --- |
| TDDFT/TDHF | `roadmap` | Requires excited-state task spec and result schema. |
| Molecular ADC/GW/AGF2/ppRPA | `roadmap` | Periodic fcDMFT G0W0 is a separate implemented single-task route. |
| Standalone gradients, Hessians, geometry optimization | `roadmap` | Forces used internally by the registered MD job do not imply a public gradient or optimizer task. |
| Solvent and QM/MM | `roadmap` | Requires environment/model fields and validation. |
| QH9 molecular dynamics | `implemented`, restricted scope | `job=molecular_dynamics`, explicit `qh9` (SCF 1e-13) or `qh9_relaxed_scf` (1e-8), restricted B3LYP/def2-SVP NVE for neutral closed-shell H/C/N/O/F molecules; typed trajectory/Fock/overlap artifacts. |
| General molecular dynamics and mid-trajectory restart | `roadmap` | Other methods, thermostats, ensembles, and trajectory restart need their own registered contracts. |
| Electron-phonon coupling | `roadmap` | Requires specialized workflow and output contracts. |
| GPU4PySCF | `reference_only` | Depends on external package availability and hardware. |
| PySCF forge/extensions | `reference_only` unless detected and registered | Must not be assumed installed. |

## Planner Rules

- The planner may use these topics to explain future directions or propose an
  implementation roadmap.
- The planner must not create executable cases for roadmap topics until they are
  represented in the capability registry and validation layer.
- External packages and hardware-dependent features must be treated as optional
  dependencies.
- Long-running workflows such as dynamics or geometry optimization require
  explicit cost/step controls and user approval.
- A QH9 trajectory is one Task, not one Task per sampled frame. The implemented
  SCF profiles and short smoke checks do not establish reference-QH9
  equivalence or long-trajectory energy conservation. See
  [[Molecular Hamiltonian Dataset Workflow]].

## Implementation Requirements

Each future feature needs:

- Task-spec fields for method/job-specific inputs.
- Backend execution and structured result extraction.
- Artifact contracts for trajectories, excited-state tables, spectra,
  geometries, or environment definitions.
- Validation and regression tests.
- UI or planner summaries that distinguish approximations, costs, and failure
  modes.

## Related Pages

- [[PySCF Reference Capability Boundary]]
- [[Strong Correlation Roadmap Rules]]
- [[Run Directory Artifact Storage]]
