# Explicit MPO Selection And Native DMRG Comparison

## Scope

The custom block2 provider previously delegated MPO selection to the driver,
whose default is FastBipartite. The first improvement batch adds Conventional
as an explicit choice and verifies fixed-orbital energy/RDM equivalence against
both FCI and the official PySCF dmrgscf + block2main interface.

This batch does not change CASSCF convergence thresholds, orbital localization,
the `approx_kernel` budget, or the orbital optimizer. Split occupied/virtual
localization remains subsequent work. The Amarel lutein comparison is now
recorded in the follow-up report linked below. FastBipartite was retained as
the default for the initial comparison. On September 20, after the user's
review of those results, Conventional became the default; explicit
FastBipartite selections are preserved. See the
official-example follow-up (author archive: `reports/block2-example-comparison-2026-09-20.md`).

## Configuration And Results

Add these fields to the existing `solver.options`:

```json
{
  "mpo_algorithm": "conventional",
  "integral_cutoff": 1e-12
}
```

The supported algorithms are `fast_bipartite` and `conventional`. The latter
selects `ConventionalNC` for two orbitals, following block2's two-site guidance,
and mixed NC/CN `Conventional` otherwise. The same algorithm is used for
orbital-optimization calls, the final fixed-orbital solve, and the auxiliary
MPO built when converting an SU(2) state for entropy analysis.

An omitted/null integral cutoff remains unresolved through preparation and
option normalization. Execution chooses `1e-12` for Conventional (the
block2main default) or the previous `1e-20` for FastBipartite. This avoids
materializing one algorithm's default before workflow-module overrides select
another. Explicit nonnegative values are preserved. Workflow-module
configuration may omit the field or provide a number.

`integral_cutoff` screens integrals; `cutoff` remains the DMRG density-matrix
truncation setting. Results record the effective integral cutoff in
`configuration`, requested/effective MPO algorithm in `mpo_algorithm`, and
seconds spent on MPO construction, DMRG sweeps, and RDM extraction in `timings`.
CASSCF solver-call traces retain algorithm and timing information for every
call. Total provider time also includes setup and diagnostics and excludes
final driver teardown, so the three stage times need not sum to it.

## Problems Found During Validation

H2 and H4 initially passed, but canonical H6 with Conventional and the driver's
`1e-20` integral cutoff failed on the first sweep with
`IndexError: unordered_map::at: key not found`. Disabling cached or delayed
contractions did not resolve it. Screening at the official `1e-12` threshold
resolved the reproduction without changing the basis, M, or sweep schedule.
This establishes the tested remedy in block2 0.5.3; it is not a general proof
about the underlying C++ failure mechanism. No exception-based algorithm
replacement was introduced.

The upstream MPO builder also screens/symmetrizes its input arrays in place.
The provider now copies them before calling it, preserving PySCF's Hamiltonian
for subsequent orbital-response calculations and independent RDM checks.

## Numerical Comparison

`tools/benchmark_block2_mpo.py` constructs one canonical-RHF H6/STO-3G
Hamiltonian, at 1.4 Angstrom spacing, for CAS(6e,6o). All three paths use that
saved Hamiltonian, the same orbital order and singlet sector, two-site sweeps,
M=64, at most 12 sweeps, noise 1e-5/1e-6/0 from sweeps 0/2/4, Davidson residual
threshold 1e-13, sweep-energy tolerance 1e-11 Ha, integral cutoff 1e-12, and
DMRG truncation cutoff 1e-14. Adaptive schedules and bond-dimension capping are
disabled for this comparison.

Each arm runs in a fresh process/scratch directory, with three rotated
repetitions, one thread and 1,000,000,000 bytes of stack allocation. The native
interface interprets `memory=1` as decimal GB; the custom driver therefore
receives the same byte count. Its worker imports no agent modules. Cold
initial MPS implementations are not guaranteed identical.

Environment: PySCF 2.13.1, block2 0.5.3, pyscf-dmrgscf 0.1.0; local macOS arm64.

| Path | Total energy (Ha) | Max energy error vs FCI (Ha) | Max 1-RDM error | Max 2-RDM error | Median solve + RDM time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Custom FastBipartite | -3.044600244087 | 3.6e-15 | 3.6e-8 | 6.8e-8 | 0.149 s |
| Custom Conventional | -3.044600244087 | 9.8e-15 | 3.6e-8 | 6.8e-8 | 0.145 s |
| Official dmrgscf + block2main | -3.044600244087 | 5.6e-14 | 5.4e-8 | 1.7e-7 | 0.380 s |

All nine solves pass energy, full 1-/2-RDM, particle-number, and RDM-contracted
energy checks. The maximum energy reconstruction error is 1.3e-14 Ha.
Custom-path median MPO times are 1.90 ms / 1.20 ms and sweep times
33.5 ms / 29.6 ms for FastBipartite / Conventional. The native time includes
subprocess startup, file I/O, MPO construction, sweeps, and 2-RDM generation.
These short timings do not establish a meaningful scaling speedup or show
that the custom interface is intrinsically faster than the official interface.

Peak worker RSS was about 145 MiB for each custom path. The native Python
worker peaked at 88 MiB and its child-process peak at 152 MiB; these peaks are
not simultaneous process-tree totals. Stack allocation is not observed usage.

The focused suite passed 130 tests, including real H2/H4/H6 energy and RDM
checks, two-orbital Conventional CASSCF against FCI-CASSCF, Fiedler ordering,
SU(2)-to-SZ entropy analysis, and existing provider/workflow/recovery contracts.
The Wiki sources, editorial guidance and 44-page runtime export were updated;
20 curation/retrieval tests pass. Wiki lint reports zero errors and the three
existing page-level inferred-paragraph citation warnings. No LLM recompilation
was needed.

Machine-readable evidence:
`block2-mpo-native-2026-09-19.json` (author archive: `reports/verification/block2-mpo-native-2026-09-19.json`).
Full local inputs, native FCIDUMP/configuration, stdout, RDM arrays and per-arm
results are in `runs/block2-mpo-native-20260919-final/`.

## Reproduction And Amarel Follow-Up

Use a Python environment containing PySCF, block2 and pyscf-dmrgscf, with
`block2main` beside its Python executable:

```sh
PYTHONPATH=. python tools/benchmark_block2_mpo.py --work-dir runs/new-mpo-comparison
```

The work directory must be new. The benchmark is explicitly opt-in and is not
part of the default suite. For a larger fixed Hamiltonian, use `--hamiltonian`
with an NPZ containing `h1e`, full `g2e`, `ecore` and the two-component `nelec`,
plus `--settings` for the schedule, thread count and stack byte budget. This
mode skips FCI reference construction. `--timeout` sets the per-arm time limit.

The subsequent Amarel campaign used the frozen lutein CAS(20e,20o) Hamiltonian,
M=600, two repetitions and all three paths on `your_node`. Conventional reduced
custom solver-process peak RSS by about 55%, while median solve-plus-RDM time
decreased by only 2.3%. All paths ran 30 sweeps without energy convergence.
See the complete report (author archive: `reports/lutein-mpo-efficiency-2026-09-19.md`).

That campaign also exposed an invalid benchmark assumption: at finite M the
lowest local energy in the final two-site sweep need not equal the RDM energy
of the final stored MPS. The benchmark now reports both energies and their gap,
checks RDM trace/Hermiticity/contraction consistency, and keeps strict energy
and RDM comparisons in the FCI-reference mode. Passing these checks does not
mean the finite-M calculation converged. Native sweep parsing handles optional
`DE` and excludes NPDM timings; seven regression tests plus H6 FCI-limit and
low-M comparisons exercise the corrected behavior. Original remote outputs
retain their old failed assertions; the follow-up report explains them.

Upstream references:
[Conventional driver example](https://github.com/hczhai/block2-example-data/blob/master/00-HC/00-dmrg.py),
[official dmrgscf interface](https://github.com/pyscf/dmrgscf/blob/master/pyscf/dmrgscf/dmrgci.py).
