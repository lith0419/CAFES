# Installation And Remote Verification

## Clean Installation

`python configure.py verify-install` builds a wheel from the current checkout,
installs that wheel into a temporary environment, changes to a directory outside
the source tree, and verifies:

- package import and distribution metadata;
- the public schema manifest;
- CLI entry points;
- Calculation Assistant and Planner Web assets;
- the packaged curated LLM Wiki;
- packaged N2 reference data;
- lightweight N2/Hubbard benchmark execution.

By default the temporary environment reuses the current interpreter's compiled
scientific dependencies while installing `pyscf-agent` only from the wheel.
Supplying `--wheelhouse` creates a fully isolated offline installation using
only that dependency bundle.

## Remote Cluster Contract

The remote route is stateless. `SshSlurmExecutor` invokes the installed
`pyscf-agent-rpc` command over SSH, the server submits or inspects Slurm work,
returns a versioned JSON response, and exits. No daemon or public service port
is required.

Client connection details live in `.pyscf-agent/remote.ini`. Each
`[remote:<name>]` profile selects a server-side Slurm section through
`remote_slurm_profile`. All clusters may share one private
`.pyscf-agent/server-slurm.ini`, with these namespaces:

```text
[server:<server-name>]
[profile:<server-name>:<resource-profile>]
```

The server section owns cluster identity, paths, scheduler defaults, and
resource limits. Resource profiles override CPU, memory, queue, or wall time
for that server only. The client and Web UI pass profile ids, not raw Slurm
arguments. `expected_cluster_id` must match the selected server section before
the remote result is trusted.

`python configure.py verify-remote --profile NAME` performs a read-only RPC
capability check and compares the server's public contract version with the
client. Adding `--submit-smoke` submits a minimal H2/HF calculation, polls it to
a terminal scheduler state, fetches the result, validates the returned
`TaskReport`, and requires scientific status `succeeded`.

Adding `--submit-batch-smoke` submits two independent H2/HF tasks as one Slurm
batch. Verification requires exact task-id mapping, two valid and successful
`TaskReport` payloads, scheduler-batch evidence, and batch manifest/handle
artifacts. A public-contract mismatch blocks both single and batch smoke jobs.

After Slurm reports `COMPLETED`, direct and SSH executors allow a bounded
propagation interval for the report to become visible on the shared filesystem.
They do not classify a successful scheduler state as collectable until the
scientific report exists; failed and cancelled terminal states remain
collectable for error inspection.

## Source-Matched Remote Releases

`configure.py deploy-remote` creates an immutable source release plus a private
worktree-specific `remote.ini`, runtime identity, and binding. The deployment
reuses the configured server Python/dependencies and derives a server profile
from the chosen base profile; it does not provision a new scientific environment.

With `require_runtime_match=true`, `SshSlurmExecutor` checks the local source
snapshot and compares the remote environment/release ids, source revision,
source fingerprint, and public contract. Changes to deployable source, including the
packaged Wiki, require a new matching deployment before further submissions.
Do not disable the identity requirement to hide a stale deployment. Ordinary
profiles without this option still use their configured RPC/cluster checks.

Deployment source fingerprints are separate from scientific artifact
references. Dataset generation on an SSH target returns a lightweight receipt;
only explicit dataset collection transfers the generated arrays to the client.

Regression tests with controlled transports verify these boundaries. A passing
identity test is not evidence that a new live server deployment or Slurm
campaign completed. Retain the date and source version of archived installation
and remote verification reports rather than relabeling them as current runs.

## Failure Semantics

- DNS, SSH, authentication, RPC, schema, Slurm, and scientific failures are
  distinct checks in a structured verification report.
- A successful SSH connection does not prove Slurm execution works.
- A completed Slurm job does not prove the PySCF task succeeded.
- A terminal failed/cancelled scheduler state without `task-report.json` is
  converted into a synthetic failed per-case `TaskReport` with scheduler and
  resource evidence. It must not discard successful sibling cases or be
  misreported as an SSH interruption.
- For block2 jobs, OOM and timeout reports include a registered recovery action
  for a larger memory or wall-time profile. Node failure, preemption, or
  revocation may be marked safe for a same-profile retry; unknown failures
  require log inspection.
- A server with a different public contract version must be upgraded before a
  smoke submission is trusted.
- A server RPC that does not recognize the client's options, including
  `--slurm-profile`, is an out-of-date deployment and must be synchronized
  before retrying. Removing the option on the client would hide the version
  mismatch and can select the wrong cluster configuration.
- Verification reports redact configured host, user, and private-key values
  from transport errors.

The client and server must install the same package revision. Private SSH and
Slurm configuration stays outside the repository and outside task payloads.
Changing a server resource profile normally requires synchronizing only the
private `server-slurm.ini`; changing Python code or RPC options requires
synchronizing the package checkout as well.
