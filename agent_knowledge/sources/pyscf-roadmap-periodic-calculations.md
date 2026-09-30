# PySCF Roadmap Periodic Calculations

## Scope

This document records the implemented periodic MVP and separates it from the
remaining periodic roadmap. The runtime capability registry and backend
validation remain the execution authority.

## Source Coverage

- Source wiki pages: `Cell Input Contract`, `Periodic Calculation Rules`, `PBC
  Density Fitting and Post-HF`.
- User guide chunks: `crystal-cell`, `pbc-scf-dft`, `pbc-density-fitting`,
  `pbc-dft-settings`, `pbc-mix-mol`.

## PySCF Reference Rules

- Use `pyscf.pbc.gto.Cell` or `pyscf.pbc.gto.M(...)` for periodic systems.
- Gamma-point calculations use objects such as `scf.RHF(cell)` or
  `dft.RKS(cell)`.
- K-point calculations require explicit reciprocal-space k points and
  k-point-aware classes such as `scf.KRHF(cell, kpts=kpts)` or
  `dft.KRKS(cell, kpts=kpts)`.
- Periodic workflows must keep gamma-point and k-point object/result shapes
  separate.
- PBC density fitting must use `pyscf.pbc.df` implementations rather than the
  molecular density-fitting contract.
- PySCF `get_fermi()` provides the mean-field Fermi level. With integer
  occupations it follows the occupied-frontier/VBM convention; with smearing
  it is the occupation chemical potential.

## Agent Status

| Feature | Current status |
| --- | --- |
| Molecular calculations | `implemented` |
| Model-Hamiltonian periodic representation | one-body Bloch `tight_binding` is `implemented`; interacting `U/V` solvers are not |
| PBC `Cell` input from POSCAR/CIF | `implemented` for 3D cells; ASE expands CIF symmetry |
| PBC SCF/DFT execution | `implemented` for HF/DFT |
| PBC k-point calculations | `implemented` for gamma-centered/Monkhorst-Pack meshes and shifts, up to 512 points |
| PBC numerical controls | `implemented` for precision, auxiliary cutoff or FFT mesh, and exact-exchange divergence |
| PBC density fitting | `implemented` for FFTDF/GDF/MDF/AFTDF; optional auxbasis for GDF/MDF |
| PBC occupation smearing | `implemented` for Fermi-Dirac/Gaussian smearing and optional fixed spin populations |
| PBC high-symmetry bands | `implemented` for ASE automatic, SeeK-path standardized primitive-cell, custom standard-point, and explicit reduced-coordinate paths |
| Structured PBC I/O | `implemented` for task, input/effective normalized cells, SeeK-path transformation audit, numerics, orbital spectrum, SCF, and band-path artifacts |
| Periodic G0W0 | `implemented` as an optional restricted fcDMFT single-task provider with registered self-energy artifacts and no fabricated total energy |
| Periodic HF+DMFT/GW+DMFT | `implemented` as optional approval-gated fcDMFT single-task workflows; physical baselines and Planner routing remain roadmap |
| Periodic MP2/CC | `roadmap` |

## Implemented Contract

- Do not convert a periodic user request into molecular `atom` input.
- Use `task_type: periodic` with nested structure, basis/pseudopotential,
  k-point, numerical-grid, density-fitting, exchange-divergence, and smearing
  fields.
- Accept POSCAR and CIF input. ASE expands CIF asymmetric units and symmetry
  operations to explicit P1 atoms. Reject partial/mixed site occupancies because
  PySCF requires an ordered atom list.
- Require VASP 5-style element symbols in POSCAR input.
- Require three-dimensional cells and a positive three-integer regular k mesh.
- Gamma `(1,1,1)` selects RHF/UHF or RKS/UKS. Other meshes select
  KRHF/KUHF or KRKS/KUKS.
- DFT requires a registered XC functional. HF must not carry an XC functional.
- All 11 pseudopotential names and all 21 named GTH basis aliases shipped by
  PySCF are registered. `GTH-LDA` aliases `GTH-PADE`; `GTH-PADE`, `GTH-LDA`,
  `GTH-PBE`, and `GTH-HF-REV` cover H through Rn. Always validate basis and
  pseudopotential availability for every species because other families have
  narrower elemental coverage.
- Persist normalized task JSON, original structure, parsed cell,
  symmetry-expanded P1 POSCAR, effective numerics/k-points, full orbital
  energies/occupations, periodic SCF summary, optional band-path JSON/TSV, and
  PySCF output log.
- Report both fundamental and direct mean-field gaps, VBM/CBM records, metallic
  classification, and the PySCF Fermi convention. These are not quasiparticle
  observables.
- Keep regular SCF sampling k points separate from high-symmetry analysis-path
  k points. The automatic band route follows `pbc/09-band_ase.py` through
  `pyscf.pbc.tools.pyscf_ase.bandpath`, `Cell.get_abs_kpts`, and
  `mean_field.get_bands`; it stores absolute Hartree energies and energies
  relative to `mean_field.get_fermi()` in eV.
- Custom paths use compact ASE labels such as `GXWKGLUWLK,UX`. Validate labels
  against the special points identified for the uploaded cell. Explicit mode
  requires a finite three-component reduced reciprocal coordinate for every
  path label. Preserve path mode, requested path, and coordinate provenance in
  the structured result.
- SeeK-path mode uses the HPKOT recipe to standardize the uploaded structure,
  constructs the explicit path from `reference_distance`, and runs both SCF
  and band evaluation on the resulting primitive cell. The regular SCF
  `kmesh` therefore belongs to the effective primitive cell, not the uploaded
  supercell. Report total energy per standardized primitive cell.
- Preserve the uploaded structure, effective primitive structure, space group,
  transformation matrices, input/primitive volume ratio, tolerances, warnings,
  and path segments as separate audit artifacts. If standardization changes
  the cell while the input-cell charge/spin is nonzero or the reference is
  unrestricted, block execution and require an explicit primitive-cell input
  or a cell-preserving path mode. This protects broken-symmetry magnetic cells.
- When smearing is active, retain entropy, free energy, and zero-temperature
  extrapolated energy separately; do not relabel free energy as total energy.
- Model-Hamiltonian periodic boundary conditions are not the same as PySCF PBC
  `Cell` workflows.

## UI And Planner Boundary

- The Calculation Assistant exposes periodic as a third task family alongside
  molecular and model-Hamiltonian tasks.
- The periodic UI reads or pastes POSCAR/CIF; it does not provide a 3D crystal
  builder.
- High-symmetry mean-field bands are an optional output. The UI controls the
  total path sampling count for ASE/custom modes and the reciprocal-space point
  spacing plus symmetry tolerance for SeeK-path. It offers automatic,
  standardized primitive-cell, validated standard-label, and explicit-coordinate
  modes; none overloads the SCF k mesh.
- Structure preview is a parsed cell/composition/symmetry audit, not a 3D
  renderer.
- Periodic static/adaptive scans are not yet accepted by the Computational
  Study Planner.
- Periodic registry entries are therefore backend-enabled but
  `planner_allowed=false` until that Planner contract is implemented.

## Remaining Roadmap

- Add DOS analysis and plotting artifacts while retaining the implemented
  separation between SCF k meshes and high-symmetry analysis paths.
- Add memory/cost estimates and convergence studies for k meshes, FFT meshes,
  auxiliary cutoffs, and basis sets.
- Add forces, stress, geometry/cell optimization, and dimensionality-specific
  electrostatic controls.
- Add periodic MP2/CC only after dedicated result contracts, cost limits, and
  regression tests exist. Calibrate the implemented fcDMFT G0W0/HF+DMFT/
  GW+DMFT routes against archived physical baselines before recommendation or
  Planner use.
- Extend the Planner with periodic structure-preserving parameter scans and
  periodic-specific postprocessing.

## Implementation References

- PySCF 2.13.0 source snapshot, with paths below relative to its `examples/`
  directory.
- Cell input and coordinate conventions: `pbc/00-input_cell.py`.
- Gamma and regular-k SCF/DFT: `pbc/10-gamma_point_scf.py` and
  `pbc/20-k_points_scf.py`.
- Smearing thermodynamics: `pbc/23-smearing.py`.
- PBC Gaussian density fitting: `pbc/35-gaussian_density_fit.py`.
- Implemented band-path separation and Fermi referencing:
  `pbc/09-band_ase.py`.
- SeeK-path HPKOT primitive-cell standardization and explicit paths are an
  agent extension with a dedicated transformation audit.
- Future periodic MP2/CC contracts: `pbc/12-gamma_point_post_hf.py` and
  `pbc/22-k_points_mp2.py`. The implemented optional G0W0 adapter follows the
  native fcDMFT silicon example rather than exposing the PySCF GW API directly.
- Maintained agent-side mapping: `docs/pyscf_example_baselines.md`.
- PySCF periodic SCF and smearing: <https://pyscf.org/user/pbc/scf.html>
- PySCF periodic density fitting: <https://pyscf.org/user/pbc/df.html>
- ASE CIF symmetry/occupancy parser: <https://wiki.fysik.dtu.dk/ase/_modules/ase/io/cif.html>

## Related Pages

- [[PySCF Reference Core API Rules]]
- [[PySCF Reference Capability Boundary]]
- [[Strong Correlation Roadmap Rules]]
