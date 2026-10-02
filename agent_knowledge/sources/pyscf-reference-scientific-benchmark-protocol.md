# Scientific Benchmark Protocol

## Purpose

The formal benchmark suite separates software regressions from scientific
accuracy claims. It is invoked with `pyscf-agent-benchmark` and writes a
versioned `pyscf-agent.benchmark-result.v1` JSON report containing environment,
source, tolerance, metric, and pass/fail provenance.

## Current Benchmarks

### N2 Dissociation

`n2-dissociation-sto3g` uses the maintained 37-point adaptive N2 dissociation
dataset. It compares overlap regions with the published CCSD and CCSD(T)
curves from T. Weaving et al., *npj Quantum Information* 11, 25 (2025), DOI
`10.1038/s41534-024-00952-4`, and verifies that scan-path diagnostics identify
the archived method-boundary discontinuity.

This is a reference-data regression: it verifies accuracy of the archived
calculation and behavior of the study-level continuity detector. It does not
rerun the full N2 curve during every unit-test pass, and production routing
thresholds must never be keyed to N2 or its bond length.

### Hubbard Exact Diagonalization

`hubbard-dimer-ed` directly solves the half-filled two-site Hubbard model for
`U = 0, 2, 4, 8` and `t = -1`. FCI/exact-diagonalization energy and mean double
occupancy are compared with the analytic singlet solution at `1e-10`
tolerance. This checks model construction, interaction signs, electron count,
solver output, and a many-body observable.

### Periodic PySCF

`periodic-he-gamma-rhf` directly executes a cubic periodic He cell with
GTH-SZV/GTH-Pade and a `1x1x1` Gamma mesh. Total energy per cell and the
Fermi/VBM energy are compared with a frozen PySCF 2.13 baseline at `1e-8 Ha`.
The calculation runs through the public backend and also verifies convergence
and artifact production.

### Optional block2 DMRG

`block2-h4-casci` solves linear H4/STO-3G CAS(4,4) with fixed-orbital block2
DMRG-CASCI and compares the energy with the PySCF direct-FCI solver for the same
active Hamiltonian.

`block2-h2-casscf` solves stretched H2/6-31G CAS(2,2) with PySCF orbital
macroiterations and block2 as the active-space solver. It compares converged
orbital-optimized energy with the same CASSCF calculation using PySCF direct
FCI, and verifies both orbital and DMRG convergence.

`block2-hubbard-ring` solves a half-filled four-site Hubbard ring with block2
and full diagonalization. It checks the ground energy, first excitation, RDM-
derived double occupancy/spin/charge observables, entanglement and symmetry
outputs, and a warm MPS restart from `U=4` to `U=6` against a cold `U=6` run.
The restart deliberately permits changed Hamiltonian integrals while retaining
the same orbital, electron, spin, symmetry, root, and orbital-ordering sector.
The maintained contract also requests two roots at `U=4` and `U=6`, verifies the
shared `pyscf-agent.block2-result-contract.v1` fields, and requires non-empty
cross-case root mapping from `pyscf-agent.dmrg-state-tracking.v1`.

`block2-adaptive-workflows` checks three connected contracts. H2/6-31G
DMRG-CASSCF restores optimized orbitals and a compatible MPS across geometries
and must agree with a cold target run. An eight-site Hubbard ring starts from a
deliberately small bond dimension, escalates within a fixed budget, reports an
energy-error estimate, and is compared with exact diagonalization. Finally,
entanglement/occupation evidence must create an unapproved, costed
ActiveSpaceAudit that restores optimized orbitals but initializes a fresh MPS
for the enlarged active space.

The maintained 2026-08-12 report passed: the H2 warm/cold difference was
`3.418e-10 Ha` and the adaptive ring error was `1.296e-9 Ha` after increasing
the bond dimension from 16 to 128.
All block2 benchmarks are reported as skipped, not failed, when the optional
provider is unavailable.

### Fixed-orbital MPO And Official-interface Comparison

`tools/benchmark_block2_mpo.py` is an opt-in local developer benchmark requiring
PySCF, block2 and pyscf-dmrgscf in the same environment. It prepares one H6/STO-3G
CAS(6e,6o) Hamiltonian and an independent FCI reference. FastBipartite,
Conventional, and official `pyscf.dmrgscf.DMRGCI + block2main` solve those same
fixed integrals in fresh processes and scratch directories, with matching
orbital order, spin, M/noise/Davidson schedules, two-site sweeps, integral
threshold, thread count, and 1 GB (1,000,000,000-byte) stack budgets.

Three rotated repetitions compare energies, full 1-/2-RDMs, particle number,
and energy reconstructed from RDMs. Saved inputs, FCIDUMP, native configuration,
outputs and JSON measurements support inspection. Timing and process peak RSS
are recorded; native kernel time includes executable startup, I/O and RDM
generation. Initial MPS implementations need not match. These small-system
checks do not establish a CAS(20,20) speedup, an optimized orbital minimum, or
adequate bond dimension for lutein. This benchmark is separate from the default
benchmark suite and must be requested explicitly.

### Lutein Orbital-Optimization And Frozen-Space Comparison

The September 20–21 CAS(20e,20o)/def2-SVPD campaign separates three questions:
inner-solver convergence, CASSCF orbital convergence, and the error introduced
by a frozen external space. With PM + Fiedler and Conventional MPO, five
42-core-frozen tasks each converged in three macroiterations with all twelve
inner calls accepted. Orbital optimization used M=600. A separate earlier
fixed-orbital M=2000 result is not M=2000 orbital-optimization evidence.

For the same geometry, +3/+5 eV external virtual windows changed energies by
17.435/15.444 mHa and maximum sorted NOONs by 0.0393/0.0318 relative to the full
external space. Both fail the predeclared 0.1 mHa and 0.01 targets. Compressed
versus original MO workspaces at the same frozen constraints agree within
8e-10 Ha; their observed runtime reductions were 7.4–12.6% in single runs.
The full AO DF tensor keeps peak RSS near 42 GiB in all these cases. Neither
small discarded weights nor reduced integral timings establish chemical
accuracy, active-space adequacy, or a general full-workflow speedup.

See `reports/lutein-overnight-results-2026-09-21.md` for settings, identities
and numerical evidence. The 23-geometry scan and full M=2000 orbital optimization
remain unperformed; cross-geometry orbital projection is separate work.

### Opt-in block2 scaling campaigns

`block2-hydrogen-chain-scaling` checks Boys-localized, Fiedler-ordered
H6/H8/H10 STO-6G DMRG-CASCI against PySCF FCI and compares bond dimensions 64
and 128 for H20. `block2-hubbard-ring-scaling` checks an eight-site ring
against exact diagonalization and then exercises 12/16-site half-filled rings,
including a 16-site bond-dimension refinement and discarded-weight limits.

These campaigns have `tier=scaling`, are excluded from the default suite, and
run only when selected by ID or with `--include-scaling`. The maintained
2026-08-11 report passed both campaigns: the largest H6/H8/H10 FCI error was
`2.014e-9 Ha`, the H20 bond-dimension energy change was `2.627e-9 Ha`, the
eight-site Hubbard error was `1.182e-10 Ha`, and the largest 12/16-site
discarded weight was `1.203e-4`.

## Rules

- Fast smoke benchmarks may run in ordinary regression tests.
- Expensive accuracy campaigns should be scheduled separately but use the same
  report schema.
- Every frozen number must record units, method, basis/pseudopotential, geometry
  or lattice, software baseline, and tolerance.
- A benchmark failure must report the failed metric; it must not tune a
  production diagnostic automatically.
- Multi-root DMRG benchmarks must compare energies with FCI where feasible and
  separately verify requested/computed-root counts, sector metadata, result
  scope, state-tracking evidence, and warm/cold continuation consistency.
- New DMRG, DMET, DMFT, GW, and periodic correlated methods require a numerical
  benchmark before registry promotion.

## Diagnostic Validation: Not Yet An Established Result

The archived benchmarks above establish their reported interface, numerical,
and continuation checks. They do not calibrate MolecularCorrelationRisk,
validate its confidence labels, or show that automated method/active-space
selection outperforms simpler choices across molecular families.

A proposed diagnostic validation campaign should record raw diagnostics,
reference/density provenance, convergence, method decisions, independent
same-Hamiltonian reference errors, and total cost including diagnostic work.
Use fixed-method, single-indicator, current-policy, and evidence-quality-aware
comparisons. Vary geometry/interaction separately from numerical settings, and
keep rule-development systems separate from final evaluation systems. Judge
specified energy/property errors and scan continuity, not just job completion.

Small-basis N2/H4/H6 comparisons are proposed starting points, not completed
acceptance evidence, Registry capabilities, or production routing rules. An
FCI reference is exact only within its stated Hamiltonian/basis and sector; a
small-basis result does not establish complete-basis or experimental accuracy.
Larger-system workflow demonstrations likewise need separate evidence for
orbital, inner-solver, and active-space adequacy. See
[[Correlation Diagnostic Interpretation]].
