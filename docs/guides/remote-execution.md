# Remote And Slurm Execution

The default execution target is local. Scheduler and SSH configuration remains
outside scientific `TaskSpec` data.

## Direct Slurm

On a cluster login node, create the private server configuration from the
template:

```bash
pyscf-agent-configure init-config server-slurm
pyscf-agent "run H2 with HF/STO-3G" \
  --executor slurm \
  --slurm-config ~/.pyscf-agent/server-slurm.ini \
  --slurm-profile amarel
```

The same target options are accepted by `pyscf-computational-study`.

The packaged Amarel template uses partition `p_cs2114_1` and pins new jobs to
`halk0121` through `extra_sbatch_args = --nodelist=halk0121`. The executor passes
this option to `sbatch`, with the same effect as `#SBATCH --nodelist=halk0121`.
Every resource profile inherits the server setting. Change the partition and
node when configuring another cluster. Existing installations use their private
server configuration; editing the template only changes newly generated configs.

## SSH To Slurm

On a local client, create `remote.ini` from the packaged template and fill in
the SSH host, username, private-key path, and remote Python:

```bash
pyscf-agent-configure init-config remote
pyscf-agent \
  --executor remote \
  --remote amarel \
  --remote-config ~/.pyscf-agent/remote.ini \
  --check-executor
```

Run the CLI, Planner, or Web UI through the same remote profile:

```bash
pyscf-agent "run H2 with HF/STO-3G" --executor remote --remote amarel \
  --remote-config ~/.pyscf-agent/remote.ini
pyscf-computational-study study.json --executor remote --remote amarel \
  --remote-config ~/.pyscf-agent/remote.ini
pyscf-agent-web --executor remote --remote amarel \
  --remote-config ~/.pyscf-agent/remote.ini --port 8000
```

The SSH route is stateless. Each request invokes `pyscf-agent-rpc`, submits or
queries Slurm, returns versioned JSON, and exits. No daemon or public server
port is required. Independent study cases use job arrays; refinement tasks are
submitted individually.

Each `[remote:<name>]` profile selects a `[server:<name>]` section in the
server-owned Slurm configuration. Resource choices are named
`[profile:<server>:<name>]` sections. Private keys remain in `~/.ssh` and are
never included in RPC payloads.

## Submitted Jobs

Submission writes a reusable handle and an execution receipt before polling:

```bash
pyscf-agent "run H2 with HF/STO-3G" --submit --work-dir runs --pretty > submitted-job.json
pyscf-agent --job-action status --job-handle submitted-job.json --pretty
pyscf-agent --job-action fetch --job-handle submitted-job.json --pretty
```

Repeat the remote target options for status, fetch, logs, artifacts, or cancel
on a remote handle. Scheduler status and the scientific status in the fetched
`TaskReport` remain separate. Slurm `COMPLETED` without a visible TaskReport
means collection is still pending; it must not create a synthetic failure.
Only failed or cancelled scheduler jobs can yield a scheduler failure report.

Study status and collection are separate from Run. Collect never starts a new
calculation, including a failed subset or the next adaptive stage. A later Run
can retry the selected cases with fresh run IDs and retained old artifacts.

The study coordinator writes submission intent before contacting the executor.
If submission acknowledgement is lost, status is `submission_unknown`; check
and reconcile that existing job before retrying. Repeating Run cannot safely
infer that no job was submitted. Corrupt checkpoints or receipts are explicit
errors. Compatible old receipts are adopted when their plan and target match.

The updated SSH endpoint stores `submission.json` beside each server submission's
`cases/` directory before returning its acknowledgement. Inspect the target's
`remote-submissions/` directory to identify the existing submission; then use
the original saved StudyPlan and local work root:

```bash
pyscf-computational-study --saved-plan runs/study-id/study-plan.json --work-dir runs \
  --executor remote --remote amarel \
  --reconcile-submission /scratch/user/agent/remote-submissions/remote-.../submission.json
pyscf-computational-study --saved-plan runs/study-id/study-plan.json --work-dir runs \
  --executor remote --remote amarel --collect-only
```

Pass `--execution-receipt` when reconciling a nested retry/sequential receipt.
The reconciliation entry point checks server evidence against local intent;
it cannot infer a missing or partially acknowledged server submission. No
automatic resubmission or new submission-ID service is introduced.

## Local Process Limits

Web local jobs persist the supervisor identity, calculation identity, cancellation
request, and status in the existing run directory. Restarting the Web server does
not remove control of these jobs. Configure an optional per-task limit with
`--local-wall-time-seconds 600` or `PYSCF_AGENT_LOCAL_WALL_TIME_SECONDS=600`.
The same flag selects supervised execution for local CLI/Study runs. The default
in-process `LocalExecutor` remains synchronous when no limit is configured.
This mechanism requires POSIX process groups and permission to inspect `ps`.
It controls descendants that remain in the calculation's group; processes that
deliberately create a new session and machine reboots are outside this boundary.

## Source-Matched Remote Releases

From a source checkout, deploy the current source and bind this worktree to its
own immutable server release:

```bash
python configure.py deploy-remote \
  --environment-id research-md \
  --profile amarel \
  --remote-config ~/.pyscf-agent/remote.ini \
  --target-remote-config .pyscf-agent/remote.ini \
  --target-profile amarel
```

The source connection profile and its server Python/Slurm environment must
already exist. This command uploads source, prepares a release wrapper and
server profile, and writes private runtime identity/binding files. It does not
install a new scientific dependency environment. The generated profile enables
`require_runtime_match`: changed deployable source or mismatched remote identity
blocks submission until a new matching deployment is created. Source includes
the packaged Wiki, so regenerate it before deploying a documentation update.
Execution nodes do not need Git: archived source snapshots use filesystem
fingerprinting when Git is unavailable. Changed source still fails identity
validation; this does not disable the version match.

## Verification

Check protocol compatibility without submitting, then optionally run single
and batch smoke calculations:

```bash
pyscf-agent-configure verify-remote --profile amarel \
  --remote-config ~/.pyscf-agent/remote.ini
pyscf-agent-configure verify-remote --profile amarel \
  --remote-config ~/.pyscf-agent/remote.ini --submit-smoke
pyscf-agent-configure verify-remote --profile amarel \
  --remote-config ~/.pyscf-agent/remote.ini --submit-batch-smoke
```

Planner messages such as `check remote status` and `collect remote results`
operate on the persisted receipt and do not resubmit completed cases.
