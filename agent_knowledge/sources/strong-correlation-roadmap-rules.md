# Strong Correlation Rules And Roadmap

## Scope

These rules distinguish the currently executable small-system workflow from
future solver integrations. Finite-cluster heuristics are decision aids, not
phase-boundary claims.

## Executable Molecular Flow

- Calculation Assistant active-space selection and adaptive molecular
  **initial scans** share one Registry-backed probe contract. `Auto` first runs
  an HF/SCF probe and accepts a complete, consistent chemical-valence or AVAS
  candidate. A registered refinement module adds MP2 evidence when that
  proposal is unresolved or ambiguous, or SCF diagnostics show instability,
  frontier degeneracy, or a routing-boundary score. Explicit `MP2` forces the perturbative
  probe, while explicit `FCI` is a small-system exact-probe option subject to
  resource review.
- Probe-only reference, output, and post-CAS settings must not replace the
  requested CASCI/CASSCF method, target solver, or target solver options.
- `MolecularCorrelationRisk` combines physical signals (gap, frontier
  degeneracy, open-shell behavior, natural occupations) and solver-stress
  signals (T2, CCSD T1/D1 amplitudes, instability, non-convergence) internally.
- It preserves independent `physics_level` and `solver_stress_level` fields.
  Failed or unconverged SCF/post-HF execution forces the overall diagnostic level
  to `strong` with `level_reason=solver_nonconvergence`, without rewriting the
  physical evidence level.
- CCSD/CCSD(T) T1/D1 diagnostics use the normalized singles norm and largest
  spin-channel singles singular value. T1 >= 0.05 or D1 >= 0.10 promotes the
  route to an approved multireference treatment; smaller elevated values warn
  against applying perturbative triples without further review.
- Weak regions route toward CCSD(T); moderate regions route toward CCSD; strong
  regions or failed single-reference refinements create CASSCF/CASCI candidates.
- Keep `diagnostic_level`, `physics_level`, `solver_stress_level`, and
  `routing_level` distinct. A strong diagnostic caused only by nonconvergence
  or a nonphysical approximate density routes to CCSD first. Correlated natural
  occupations become moderate at fractionality 0.50 and strong at 0.80.
- Failed CCSD(T) calculations retry with CCSD before CAS promotion. Failure
  remains strong solver-stress evidence without replacing the independent
  physics level.
- The planner must describe the evidence and recommended next action without
  exposing internal score values as physical observables.
- Scores and thresholds are current routing heuristics, not learned weights or
  calibrated error estimates. `confidence` counts available evidence components;
  it is not a probability of correctness. See [[Correlation Diagnostic Interpretation]].
- Every proposed active space needs explicit approval before CAS runs. Candidates
  already assembled in one executable plan are reviewed together; unrelated
  recovery plans remain separate.
- Ground-state SC-NEVPT2 may follow approved spin-adapted CASCI/CASSCF only.

## ActiveSpaceAudit Contract

The audit is a first-class artifact. It records canonical orbital indices,
orbital energies, occupations, AO/atom and atom-as-fragment contributions,
localization metadata, selection reasons, `ncas`/`nelecas` consistency,
estimated active electrons, candidate methods, and manual approval state.

Current candidates are `manual`, UNO-style occupation-window,
`chemical_valence`, `evidence_expanded`, AVAS AO/fragment projection, and a
broad merged-evidence record. AVAS resolves AO labels or `atom:`/`fragment:`
selectors, records the projected active orbitals, and retains a PySCF orbital
matrix for the approved restricted/ROHF CAS initial guess. The default
occupation window is `0.02 < n < 1.98`.

Automatic selection follows `smallest_chemically_complete_first`. Among valid
chemically resolved candidates, the smallest complete active space is proposed
for the initial approval and calculation. Strong natural-occupation or T2
evidence outside that space is retained as a machine-readable next-expansion
candidate rather than silently replacing the smaller baseline. Reviewable
candidates are ordered by increasing `ncas`; expansion occurs only after the
baseline result remains unreliable or the user explicitly approves a larger
space. This rule depends on candidate validity, chemical completeness, and
size, not on an element, molecule, or fixed CAS dimension. Explicit manual and
AVAS requests retain precedence.

For related molecular cases, the Planner resolves one study-level chemical
target and CAS dimension from the case audits before asking for approval. Each
case retains its own canonical-orbital indices and initial orbital matrix.
Missing or conflicting mappings are sent back to the shared probe rather than
filled with a fixed frontier space or copied from another geometry. This policy
is deterministic, evidence-backed, and independent of any named molecule.

Every plan also contains a dimension-based resource estimate. FCI/CAS use
alpha/beta determinant dimensions and MP2/CCSD use tensor-scaling proxies.
Crossing a configured determinant, memory, or total-work threshold creates an
explicit approval gate before execution.

Each Study persists `study-state.json` with current TaskReports, run/receipt
associations, attempt counts, and per-case contract fingerprints. Run reuses
compatible successful cases and first recovers pending executions. A reviewed
retry preserves Study and case IDs while creating a new run for selected cases;
it does not create a child Study. Status and Collect do not submit retries.
Run and Collect rebuild the complete StudyReport, preserving unselected results
and adaptive/postprocessing evidence. See [[Study Orchestration And Report Authority]].

Treat active spaces up to 14 orbitals as conventional FCI-based CAS candidates.
With `active_space_solver=auto`, larger approved active spaces or audits that
explicitly recommend DMRG route to optional block2 DMRG-CASSCF after the same
ActiveSpaceAudit and cost review. The estimate records bond dimension, sweeps,
roots, and a DMRG work proxy. Single-state and same-spin/symmetry-sector state-
averaged DMRG-CASSCF are executable; different-spin averaging and AFQMC remain
design recommendations.

For approved block2 CASCI/CASSCF, the task may localize only the initial active
block with Boys or Pipek-Mezey, then use canonical, Fiedler, or a complete
manual orbital permutation. The audit and CAS result record the transformation;
the DMRG result and checkpoint manifest record the requested and executed
ordering. CASCI leaves the orbitals fixed. CASSCF uses PySCF macroiterations and
passes block2 1/2-RDMs back to the orbital optimizer while reusing an internal
MPS between solver calls.

A compatible cross-task DMRG-CASSCF checkpoint restores optimized orbitals and
the MPS together. Compatibility includes atom/AO layout, basis, active-space
electron/orbital counts, spin, symmetry, roots, and orbital ordering. If an
entanglement-driven review expands the active space, reuse only the optimized
orbitals and initialize a fresh MPS after explicit ActiveSpaceAudit approval.
The recommendation must preserve its limitation: in-space entanglement does
not directly measure excluded orbitals.

DMRG may increase bond dimension and add sweeps only within configured limits.
Reintroduce bounded noise during an escalation stage. Convergence requires both
energy change and discarded weight. A compatible checkpoint supports bounded
same-method escalation under the Registry recovery policy (currently `M=1024`);
this is an automatic-recovery budget, not a maximum supported or scientifically
sufficient M. Explicitly configured calculations may use larger M. Reaching
the automatic limit without satisfying discarded weight requires review. A
heuristic energy-error estimate is not a rigorous bound and small systems must
still be checked against FCI/ED.

Fixed-orbital block2 tasks may persist an MPS checkpoint manifest. After the
user explicitly requests result analysis, the workflow first tracks targeted
roots across ordered successful cases. Per-root 1RDM similarity is preferred;
natural occupations are the fallback and energy proximity is secondary. Root
reordering or an ambiguous assignment is recorded in
`pyscf-agent.dmrg-state-tracking.v1`, and ambiguous cases are not continuation
anchors.

An unresolved block2 case may then propose a same-method retry from the nearest
compatible state-trusted successful MPS. Up to four compatible candidates are
retained for provenance and fallback. Compatibility includes orbital,
electron, spin, symmetry, root, and orbital-ordering sectors; Hamiltonian
integrals may differ across a parameter scan. Approval is mandatory, source and
target must use the same executor identity/remote profile, and source case,
checkpoint, distance, compatibility, Hamiltonian fingerprints, and state-
tracking evidence are preserved.

## Executable Model-Hamiltonian Flow

- Model-Hamiltonian studies use a static scan over the full requested grid.
  Every case uses the solver explicitly selected in the study task contract;
  there is no model initial-scan strategy or ActiveSpaceAudit stage.
- The report preserves each original `case_id`, variables, solver, and
  structured diagnostics. Result analysis may flag unreliable or anomalous
  cases, but it does not silently replace their solver through a molecular
  active-space workflow.
- Model-Hamiltonian studies request `strong_correlation_diagnostics` by
  default. This supplies correlation routing and the registered scalar
  postprocessing metrics without a separate pre-run selection step.
- Model MP2/CCSD/CCSD(T) use unrestricted mean-field references by default.
- Routing can use `U/t`, filling, mean-field gap, frontier degeneracy, natural
  occupations, T2 amplitudes, solver convergence, and FCI many-body observables.
- Full CI may add many-body gap, double occupancy, spin correlation, and
  connected charge correlation; interpret all of them as finite-size evidence.
  When strong-correlation diagnostics are present, postprocessing exposes the
  site-averaged double occupancy and signed nearest-neighbor spin and connected
  charge correlations as scalar trend metrics while retaining the raw vectors
  and matrices in the diagnostic payload.
- Model FCI and block2 DMRG accept an explicit `nroots`; the resulting
  low-energy manifold is classified as isolated, near-degenerate, or
  numerically degenerate within the computed sector. If every requested root
  lies in the near-degenerate window, request more roots before calling the
  manifold complete.
- Explicit complete-spectrum FCI diagnostics use dense diagonalization and are
  rejected above determinant dimension 5,000 after cost review. Targeted roots
  do not imply complete diagonalization.
- Reference-relative correlation energy is internal diagnostic bookkeeping and
  is not exposed as a result or plotting field. Do not expose a generic
  `screening_energy` as a physical result.
- Builder-defined single-band Hubbard finite clusters may use the optional
  libDMET ground-state solver with translated or explicit finite-graph
  fragments, FCI, CCSD, or block2 DMRG impurities, and restricted or
  broken-symmetry UHF references. Finite-graph execution requires `Sz=0`;
  translated representative fragments may use a validated nonzero spin sector.
  Outer-loop energy, density change, and density-fit residual determine DMET
  convergence; correlation-potential change is diagnostic only. These are
  solver-stress evidence, not a thermodynamic phase diagnostic.

## Adaptive Recovery Rule

Unconverged or failed refinement is **strong solver-stress evidence** and raises
the overall adaptive correlation-risk label to `strong`, while the independent
`physics_level` remains unchanged. It is not proof of a single physical cause.
The `blocked` state means validation or approval is incomplete and must not be
used as correlation evidence. The planner should propose a bounded next action:
increase iterations, promote a single-reference molecular case to CASSCF with
an audit, expand a failed CAS active space, try guarded FCI when eligible, or
explain a blocked input. Present different failed cases one at a time and merge
subset results back into the full report.

The model diagnostic artifact likewise reports independent physics and
solver-stress levels. A solver result with `converged=false` has overall
`level=strong` and `level_reason=solver_nonconvergence`; a validation/approval
`blocked` status never enters that physical diagnostic calculation.

For block2, nonconvergence first creates a solver-specific recommendation:
increase bond dimension when discarded weight is too large, extend sweeps while
energy is still changing, or review resources after the automatic limit. Slurm
out-of-memory and timeout states request a larger memory or wall-time profile;
node failure/preemption may be retried with the same profile. These terminal
scheduler outcomes become per-case failed `TaskReport` records so other cases
remain usable.

## Roadmap Only

- Unrestricted AVAS selection, chemically named fragment definitions, and rich
  orbital visualization.
- Different-spin/unrestricted DMRG-CASSCF, transition-RDM/direct-wavefunction
  root tracking, ab initio DMET including DMRG impurities, and AFQMC solver adapters.
- State-specific orbital optimization and excited-state SC-NEVPT2.
- Periodic MP2/CC, calibrated periodic correlated diagnostics, periodic Planner
  scans, and interacting model twist scans. Registered fcDMFT periodic
  G0W0/HF+DMFT/GW+DMFT single-task routes already exist within their provider
  boundaries; numerical validation must still be reported separately.
- Structure-factor and finite-size-scaling postprocessing.

## Must Not

- Do not claim a thermodynamic phase from a finite cluster diagnostic.
- Do not run CASCI/CASSCF without explicit `ncas`, `nelecas`, and approval.
- Do not present unrestricted AVAS or an external-solver recommendation as an
  executable calculation.
- Do not attach SC-NEVPT2 to an active-space probe or unrestricted CAS.

## Related Pages

- [[Adaptive Scan Workflow State Machine]]
- [[PySCF Roadmap Multireference Active Space]]
- [[Model Hamiltonian Solver Support]]
