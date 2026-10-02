# Small Active-Space CASSCF Workflow Acceptance

Inspection date: 2026-09-13.
Outcome updated: 2026-09-14.

## Problem And Result

The full task workflow normalized block2 options before applying TaskSpec
orbital processing. Normalization inserted `orbital_ordering=canonical`, so the
later fallback could not apply a requested Fiedler order. Direct CAS tests
bypassed this preparation stage and missed the bug. The completed lutein task
61400178_0 applied PM but retained canonical ordering throughout its solver calls.

A shared option resolver now applies task orbital settings before provider
defaults. Explicit solver options retain precedence over task orbital settings;
workflow module overrides retain precedence over solver options. The numerical
CAS entry point uses the same resolver. No new workflow or MCP implementation
is introduced.

Provider configuration is now retained on hosts without block2, preventing
feature preparation from replacing a requested root count with defaults. The
existing availability gate remains responsible for blocking unavailable solvers.

The existing solver-call trace adds the effective bond-dimension schedule and
sweep limit returned by the driver. Together with role, ordering, MPS continuation
and convergence fields, these distinguish orbital optimization from final
fixed-orbital refinement. Scientific convergence criteria are unchanged.

## Verification

The focused local suite passed 70 tests. The public CalculationApplicationService
test executes a broken-symmetry H4 DF-UHF reference, diagnoses its UNO active
space, reuses the reference density, applies PM/Fiedler in CAS(2,2), optimizes
orbitals and performs a distinct final solve. It agrees with DF-FCI within
1e-8 Hartree, verifies the real per-call M=16/32 settings, and checks that the
initial Fiedler permutation remains fixed through MPS continuation. Configuration
tests separately cover M=600/2000, manual/canonical/Fiedler ordering, explicit
solver precedence and hosts with/without block2.

Restoring the previous preparation behavior through a test-only patch makes the
public workflow regression fail because it observes canonical instead of Fiedler.
This verifies that the test detects the original bug, rather than just exercising
the lower-level numerical entry point.

The same 70-test suite also passed on Amarel, job `61529316`, using Python
3.10.20, PySCF 2.13.1 and block2 0.5.3. This is a completed small-molecule
workflow acceptance, including the public service, rather than a convergence
claim for lutein.

## Lutein Acceptance Case

The supplied paper, DOI 10.1039/d6cp02191c, Section 3.1 (PDF page 6), lists
CAS(8,14), CAS(10,16), CAS(12,18), CAS(14,20) and CAS(20,20). The smallest
DMRG-CASSCF space is CAS(8,14). The FOMO-CISD(6,9) on page 5 belongs to
semiempirical geometry generation and is a different method.

The acceptance request reuses geometry 12, def2-SVPD and the previously
converged DF-UHF density. Starting from the diagnostic UNO coefficient matrix
used for CAS(20,20), freeze its six highest-occupation active orbitals, leaving
four occupied-like and ten virtual-like orbitals. This gives 152 inactive
doubly occupied orbitals plus 8 active electrons in 14 orbitals, preserving
312 total electrons. Use the full UNO coefficients with their matching active
block; canonical display indices are not substituted for that transformation.

This is a ground-state workflow acceptance at the paper's smallest CAS size.
The precise correspondence with the paper's individual orbitals has not been
verified against its supplementary orbital table. It is not a reproduction of
the paper's SA4 spectra. PM is applied to the initial active block, Fiedler is
then fixed across solver calls, orbital M is 600 and final fixed-orbital M is
2000, with at most 30 sweeps per call and 200 SCF/CASSCF cycles. The smaller CAS
does not reduce the AO basis or eliminate DF integral and orbital-gradient costs.

Use a fresh MPS after changing the CAS; the existing CAS(20,20) MPS is incompatible.
No automatic increase above M=2000 is requested. Inspect effective ordering,
per-call M, final energy change, discarded weight, orbital convergence and
checkpoint integrity in the completed artifacts. Scheduler completion alone
is insufficient. Per-run deployment, selection, submission and runtime evidence
are kept in `.pyscf-agent/lutein-cas8-workflow-20260913/`.

## Final Disposition: Cancelled

Job `61529443_0` was cancelled at the user's request on September 14 at
09:25:09 EDT after 22:20:26. The latest inspection was still in orbital
macroiteration 13; orbital convergence and the final fixed-orbital solve had
not been reached. All 52 completed DMRG calls used M=600 and the same actual
Fiedler permutation. The requested final M=2000 was capped to 1941 by the
small-space planner, but that final phase never ran.

The full 1294-AO orbital response, particularly DF JK evaluations, dominated
the observed runtime. The smaller CAS did not provide a cheap lutein acceptance
case. Numerical/performance follow-up is deferred. See the
current campaign report (author archive: `reports/lutein-workflow-status-2026-09-14.md`)
for completed-run comparisons and the separate convergence criteria.
