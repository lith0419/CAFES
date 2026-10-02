# QM9 500 — DFT/AIMD submission plan

Storage update (2026-10-01): inputs are now archived under
`/home/your_username/datasets/` on Amarel (`your_username` is a placeholder for
the archive owner's account). Historical `datasets/...` paths below
refer to that archive root; they no longer exist in this checkout. See the
[storage and restore guide](../repository-storage.md). The dated calculation
status below is a historical record.

Latest check (2026-09-23 13:16 UTC): original-pilot40 array `61772962` has
finished with **39 successful trajectories / 39,039 accepted frames** and one
DIIS linear-algebra exception (qm9_000160, tetrahydrofuran, frame 698). All three
representative trajectories also succeeded. Their separate dataset summary
still has obsolete 2 ps metadata and must be corrected; the raw 828-frame
results are valid. The main-batch aggregate report has grown to ~1.24 GB,
causing full-refresh timeout. See
`runs/qm9-aimd-main-20260922/oneps-check-20260923.md` for diagnoses. No retry,
scientific parameter change or automatic tracking was started. Submission
settings and historical milestones follow.

Current submission (2026-09-23 UTC): the user selected the **original pilot40**
and a **1 fs timestep / 1 ps duration**. All 40 Tasks are submitted as Study
`20260923-040228-28e4e7aa`, Slurm array `61772962`, with **1001 frames per
molecule (40,040 total)**, including t=0 and t=1 ps. Each Task requests 16 CPUs,
16 GiB and 24 hours on main/general. The batch allows 17 concurrent Tasks;
the three existing representative Tasks retain their reserved slots, keeping
combined concurrency at most 20. See
`runs/qm9-aimd-main-20260922/pilot40-1ps-report.md` for the complete molecule/Job
mapping. No automatic tracking is enabled.

The original three representative trajectories retain their **1.209442 fs**
timestep and 828 frames (1.000209 ps). Their history is as follows: the three
previous 2 ps Runs (array `61771589`) were cancelled; their records are retained.
Replacement array `61772271` completed under the same Study
`20260923-031401-7eed0bd5` and the same three Tasks (70/128/226 AOs), with new
attempt-2 Run IDs. Each uses the original geometry and velocity seed, **828
frames (1.000209 ps)**, 16 CPUs, 16 GiB and a 24-hour limit on main/general.
SCF conv_tol=1e-13 and max_cycle=200 remain unchanged. This is a fresh run
from t=0; the backend cannot shorten a live integrator or resume partial MD.
See `runs/qm9-aimd-main-20260922/duration1ps-report.md` for current records.

The selected list is `datasets/qm9_ccsdt_seeds_500/provenance/pilot40_selection.json`,
not the later 40 failed-task retries. Original list order, geometries and the
full-campaign velocity-seed mapping are preserved. CF4 occurs in both the new
batch and the representative Study, but at different timesteps; these are
distinct calculations and the existing trajectory remains available for
comparison. The user explicitly requested manual status checks only.

Retain one original initial conformation per molecule. The 5000-frame pilot is
technically successful, but the broader geometry/energy coverage for later
CCSD(T) labeling is not yet established. See
`reports/qm9_sampling_timescales_20260922/REPORT.md` for source-frequency periods,
finite-window trajectory evidence and the earlier 2/5-ps duration candidates.
The selected duration is now 1 ps. The remaining 450 molecules and any CCSD(T)
labeling remain unsubmitted.

Latest status (2026-09-22 20:00 EDT): all 50 pilot molecules are complete,
yielding 5000 accepted frames and no failures or missing frames. All 43 selected
retries succeeded with max_cycle=200 and conv_tol=1e-13. The remaining 40-case
batch reached at most 172 SCF cycles per frame. Saved scalar trajectory checks
passed; energy drift and finite-step oscillations are documented in
`runs/qm9-aimd-main-20260922/remaining40-maxcycle200-report.md` and
`pilot50-energy-audit.json`. This establishes workflow success at the requested
QH9 sampling settings, not timestep convergence for precision dynamics.
The remaining 450 molecules have not been submitted. Earlier milestones follow.

Status: two full 100-frame preflight trajectories (CH4 and NH3) were submitted
on 2026-09-21 EDT (2026-09-22 UTC) as Slurm array `61746183` on `main`.
The initial concurrency limit of one was raised to **20** at the user's request.
Both preflight trajectories passed artifact and convergence checks (200 accepted
frames). The remaining 48 pilot molecules were submitted as array `61746270`,
with 20 concurrent tasks, 4 CPUs, 16 GiB and 12 hours per task. Early pilot results
include SCF convergence failures; full-campaign expansion is deferred until
these are diagnosed. The remaining 450 molecules are not submitted. Records are
in `runs/qm9-aimd-main-20260922/`, including `pilot-report.md`.
Update at 2026-09-22 18:51 UTC: the three representative retries all succeeded
with max_cycle=200 and unchanged conv_tol=1e-13. The remaining 40 failed pilot
Tasks were submitted as array `61761182`, with 20 concurrent tasks and unchanged
4-CPU / 16-GiB / 12-hour resources. Successful results are retained. See
`runs/qm9-aimd-main-20260922/remaining40-maxcycle200-report.md`; the 450-molecule
expansion remains unsubmitted pending pilot acceptance.
This is the current submission direction. CCSD(T) labeling is deferred; the
earlier CCSD(T) guide remains a separate future protocol.

## Inputs and scientific settings

Use the existing 500 seeds in `datasets/qm9_ccsdt_seeds_500/seed_geometries.json`
without changing their geometry, order, IDs or checksums. The historical input
directory name does not determine the method of the new calculation.

`dataset-spec.json` records the original 100-frame pilot settings below. The
current original-pilot40 batch has its own saved spec under the remote
`pilot40-1ps-dt1fs/dataset-spec.json`: 1001 frames, `time_step_au=41.34137333518211`,
max_cycle=200, verbosity=5, and otherwise the same electronic/dynamics settings.
The 1 fs timestep differs from the published QH9 sampling schedule.

| Setting | Value |
|---|---|
| Electronic structure | Restricted B3LYP/def2-SVP, neutral singlet |
| SCF energy tolerance | **1e-13 Ha**, explicit `qh9` profile |
| SCF orbital-gradient tolerance | 3.16e-5 |
| SCF grid / DIIS / max_cycle | 3 / 8 / 50 initially; 200 for approved failed-task retries |
| Symmetry / density fitting | Disabled / disabled |
| Dynamics | NVE, 300 K Maxwell–Boltzmann initial velocities |
| Time step | 50 a.u. (approximately 1.20944 fs) |
| Frames / save stride / offset | 100 / 1 / 0 |
| Saved frame indices | 0–99, including the starting geometry |
| Planned trajectories / retained geometries | 500 / 50,000 |

With the current PySCF integrator, frame zero is at t=0; the last frame is at
4,950 a.u., approximately 119.735 fs. No extra equilibration stage is configured.
NVE does not thermostat the trajectory. The starting DFT geometry is already
included; an independent preliminary single point is additional work if run.
All frames require DFT energy and force evaluation. Saving every frame adds
matrix extraction and I/O but does not multiply the MD force evaluations.

The strict SCF profile does not change legacy sampling defaults. Full retention
is explicitly selected in this campaign file. The existing relaxed profile
remains available for previously prepared work. Reject explicit settings that
conflict with the selected profile rather than silently relaxing convergence.

The settings follow the published QH9-dynamic-300k schedule and SCF thresholds,
not an exact reproduction of its molecule set or trajectories. Record the actual
PySCF version, chosen profile, random seeds and numerical evidence. Do not claim
bitwise reproduction. See the [QH9 paper](https://arxiv.org/html/2306.09549v4)
and [official repository](https://github.com/divelab/AIRS/tree/main/OpenDFT/QHBench/QH9).

## Execution sequence on Amarel

1. Deploy the verified source version into an independent campaign environment
   and select an SSH/Slurm executor with partition `main`, account `general`,
   explicit walltime, and no fixed node. The main partition's observed default
   walltime is only two minutes; maximum walltime is 72 hours. Keep the lutein
   private-partition configuration separate.
2. Freeze the molecule-to-velocity-seed map before splitting the campaign into
   batches. The current planner assigns base seed + case index; preserve the
   full-campaign mapping when preparing pilot/subset batches and retries.
3. Use small local/remote smoke calculations to verify integration, then run
   the existing 50-molecule pilot with the complete 100-frame protocol. Confirm
   SCF convergence, energy drift, matrix integrity, timing, memory and scratch.
   Smoke runs with different frame counts are not full pilot trajectories.
4. Expand to the remaining 450 molecules only after pilot acceptance. Use
   measured DFT-gradient costs to set resources and concurrency; the earlier
   CCSD(T) resource table is not a DFT/AIMD timing estimate. The current requested
   concurrency is 20. The executor applies caps per Slurm array; divide the
   available slots between overlapping batches (currently 17 + 3), or run them
   sequentially, to maintain a campaign-wide maximum of 20 simultaneous tasks.
5. Finalize accepted/rejected samples through the existing Study dataset flow.
   Keep all frames from one molecule in one data split. Save coordinate,
   velocity, energy, time, temperature, Fock/overlap and AO metadata artifacts.

Mid-trajectory continuation is not currently supported by the agent contract.
An interrupted trajectory requires a new Run; preserve the original attempt and
seed. A passed short smoke test does not establish 100-frame energy conservation
or campaign success.

Integration failures now preserve `log-pyscf-output.log` and
`result-molecular-md-failure.json`, including the failed frame, geometry and last
observed SCF residuals. Selected pilot retries test `max_cycle=200` while retaining
`conv_tol=1e-13`, the complete 100-frame trajectory, and the original velocity
seed. Only the selected failed Tasks receive new Runs under the original Study.
