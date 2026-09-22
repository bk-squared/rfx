Measurement tooling for the multi-node probe (solver = the pinned green commit of the mirror, i.e. main plus this tooling)
======================================

No VESSL job is submitted by generating these files. The worktree was found at
the mirror's `green` commit, so GPU jobs check out **that commit** (main tip plus this tooling), copy only the two probe
Python scripts from the tooling commit, and record both commits. The `rfx` tree
must equal `<green>:rfx`. The lead must first incorporate these tooling files
into a commit and push it to the bare mirror as `green`; the agent made no commit.
`scripts/vessl_multinode_oracle.yaml` is ignored by the existing gitignore and
needs explicit `git add -f` when the lead stages the tooling.

The installed CLI is VESSL **0.1.188**. `vessl run create --help` failed during
import because authentication tried to resolve `api.vessl.ai`. Source inspection
of `vessl/cli/run.py:153-177` exposes `--file`, `--watch`, organization and project.
`vessl/run.py:354-394` checks image/resources and forwards YAML to a server; it is
not a complete local schema validator. `openapi_client/models/v1_run_spec.py`
has no distributed/worker-count member. Neither the [Run YAML reference](https://docs.vessl.ai/reference/yaml/run-yaml)
nor its [cheat sheet](https://docs.vessl.ai/reference/yaml/cheatsheet) lists workers.
No supported multi-worker **Run YAML** was found; server acceptance of unknown
keys was not tested. `resources.node_names` selects candidate nodes, not replicas.

The installed `vessl/cli/experiment.py:502-518,522-630` exposes `--worker-count`,
`--framework-type`, `--cluster`, `--resource`, `--image-url`, `--hyperparameter`,
`--upload-local-file`, `--no-use-vesslignore`, `--working-dir`, `--output-dir` and
`--command`. `vessl/experiment.py:270-300` maps workers to
`distributed_experiment_create_api(... worker_replicas=..., worker_resource_spec_id=...)`.
The [distributed Experiment reference](https://docs.vessl.ai/guides/experiments/distributed)
documents the same CLI and provider environment. The earlier evidence root's
`criterion-4-green.txt`, `vessl/experiment-9/env-rank-{0,1}.txt`, and
`vessl/experiment-10/status.txt` record two workers, one 4090 each, 14 CPU/48Gi,
RANK/WORLD_SIZE/MASTER_PORT, and `<base>-experiment-tcp-0` DNS (September 2, 2026).
Those records used CLI 0.1.199, not the locally installed 0.1.188.

Experiment CLI 0.1.188 has **no NFS mount option**. Generated commands upload a
self-contained copy of the lead's bare mirror plus the bootstrap scripts.
They do not invent an Experiment `--mount` or a Run `num_workers` key. Workers
write `/root/workspace/claude-workspace/rfx/runs/multinode-<stamp>/<job>/` and
trap-copy to provider-backed shared `/output/multinode-<stamp>/<job>/`. The former
path may be pod-local in an Experiment; the generated collection commands download
the latter into the lead's mounted runs directory. Oracle Run YAMLs mount NFS
directly. Each worker has a summary receipt; `summary.json` reports workers seen
and whether all expected workers are present. Collection recomputes that summary.

After the lead has pushed the tooling commit to mirror `green`, run on the Mac
(replace the two path prefixes if submitting from a pod):

```sh
PY=/Users/byungkwankim/Documents/rfx/.venv/bin/python
MN_SHARED=/Users/byungkwankim/mnt/remilab-fs/personal-workspaces/claude-workspace/rfx
MN_MIRROR=$MN_SHARED/mirrors/multinode-probe.git
MN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
MN_SHA=$(git -C "$MN_MIRROR" rev-parse 'green^{commit}')
MN_SPEC=/tmp/rfx-mn/launch-$MN_STAMP
"$PY" scripts/diagnostics/prepare_multinode_launch.py --tooling-sha "$MN_SHA" \
  --mirror "$MN_MIRROR" --artifact-root "$MN_SHARED/runs" --stamp "$MN_STAMP" --output "$MN_SPEC"
sh "$MN_SPEC/prepare-upload.sh"
```

Before submission, the lead follows `~/.claude/rules/vessl-jobs.md`'s existing-run
inspection, backup and retention rules. The preparation step creates eight shell
files; inspect them and the three oracle YAMLs. Submit with:

```sh
sh "$MN_SPEC/submit-oracles.sh"
sh "$MN_SPEC/submit-two-100.sh" &
sh "$MN_SPEC/submit-two-200.sh" &
sh "$MN_SPEC/submit-two-400.sh" &
wait
```

Oracle submissions use `scripts/vessl_submit.sh` to record run IDs. Experiment
submissions parse the CLI's `Created '<number>'` output into
`<job>/experiment_number.txt` on the submitter; workers never guess an ID.
All workers receive RFX_STAMP, RFX_EXPECTED_SHA (tooling), RFX_KIND, RFX_NX,
RFX_SOURCE_ARCHIVE, RFX_BOOTSTRAP, RFX_RUNS_ROOT, PYTHONUNBUFFERED,
HDF5_USE_FILE_LOCKING, LANG, XLA_PYTHON_CLIENT_PREALLOCATE and JAX_PLATFORMS.
The worker adds MPLBACKEND, MPLCONFIGDIR and PYTHONPATH. Two-worker jobs require
provider RANK=0/1, WORLD_SIZE=2, MASTER_PORT and HOSTNAME; their absence aborts.
The coordinator is derived from HOSTNAME and RANK; MASTER_ADDR is recorded for
comparison. No torchrun process is added. Each worker starts exactly one probe
process with `--local-device-id 0`. Its initialization precedes all device use.

After both workers in each Experiment are terminal:

```sh
sh "$MN_SPEC/collect-two-100.sh"
sh "$MN_SPEC/collect-two-200.sh"
sh "$MN_SPEC/collect-two-400.sh"
"$PY" scripts/diagnostics/compare_multinode_probe.py \
  --oracle "$MN_SHARED/runs/multinode-$MN_STAMP" \
  --distributed "$MN_SHARED/runs/multinode-$MN_STAMP" \
  | tee "$MN_SHARED/runs/multinode-$MN_STAMP/comparison.json"
```

The three comparisons are 200/400/800 x 116 x 116 on one GPU against slabs
100/200/400 on two workers, 200 steps, one warm-up and three timed public calls.
The box length is `(array_shape - 1) * 1 mm` because the grid includes an endpoint;
the realized shape is asserted, so no x padding is needed. Source and three
probes are Ez, source at x=nx/2 mm, probes at nx/4, nx/2, 3nx/4 mm, y/z at center.
The source is a field-amplitude GaussianPulse(f0=7.5 GHz, bandwidth=0.8).

`run_seconds` includes setup, compilation, scan, native gather and blocking.
`scan_seconds` observes the actual scan-call boundary and blocks carry and trace
before native gathering; it includes compilation if that invocation compiles.
The observer uses Python line tracing of only the outer runner, adds overhead,
and does not replace any solver function. There is no process_allgather or
sharding repair. Native result addressability and trace origin are recorded.
Exit 2 means gather/trace materialization failed after the scan; exit 1 means
another error; exit 0 means all calls and trace materializations completed.
If gathering fails, remaining repetitions still run, scan timings remain, and a
native scan-output trace is saved only if it can already be materialized.

Comparison prints both scan and full-call two/one ratios, per-step times,
per-probe max absolute difference and oracle-peak-normalized difference, and
per-rank allocator peak/limit. It requires all ranks/repetitions for a timing
ratio and uses median(max rank wall-clock at each repeat). Missing values stay
null. A zero oracle peak has a null relative error and an explicit flag.
Peak memory is the allocator high-water of that process, not a per-repeat delta.
Pod hostname alone does not establish two nodes: compare recorded DMI host UUIDs
and provider placement; GPU UUIDs and unavailable host UUIDs are recorded too.
