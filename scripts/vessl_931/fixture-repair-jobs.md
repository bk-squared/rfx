# Fixture repair evidence jobs

Live launch results and resource correction are recorded in
`docs/design_notes/931_migration/T10-fixture-live-qualification.md`.
The five ordinary lanes use verified CPU-only `base-pod` (8 CPU / 32 GiB).
The memory-heavy refinement uses its own `cpu-32-mem-64` job (32 CPU / 64 GiB).
VESSL silently discarded both the original top-level cpu/memory fields and
custom requests on this cluster; use the named presets. Every job records and
checks its numeric cgroup CPU/memory limits before cloning or simulation.
The submitter copies the YAML to a plain directory, avoiding the CLI's worktree
.git-file error; the pod also trusts the exact linked Git administrative path.

Each job requires the full final commit SHA, and the source checkout is fixed
to `/root/workspace/byungkwan-workspace/research/rfx-931-fastlane-reds`. The job
creates a unique temporary clone of that commit in its future pod. It asserts
the commit, clean initial tree, CPU backend and local `rfx` import, with
`PYTHONPATH=$PWD`. It runs commands serially, with one pytest process and at
most four workers. Each test has a 7200-second thread timeout, each command a
21600-second wall timeout, and the complete evidence runner an 86400-second
wall timeout, except the refinement capture: its command cap is 86400 seconds
(24 hours) and its job cap is 129600 seconds (36 hours). No golden write is
requested.

| Job | Evidence |
|---|---|
| `fixture-repair-v173a.yaml` | Named capture and independent pytest comparison with the committed composition baseline |
| `fixture-repair-msl.yaml` | Base 12-period capture; independent original historical-base E2E test and fast qualification contract tests |
| `fixture-repair-msl-long.yaml` | Long 24-period capture and fast qualification contract tests only |
| `fixture-repair-msl-refine.yaml` | Refinement 2 at 12 periods and fast qualification contract tests only |
| `fixture-repair-farfield.yaml` | Repaired fixture, current owner, 600 and 1200 steps; complete in-plane nonuniform test module |
| `fixture-repair-short.yaml` | Repaired coarse and fine current-owner traces; geometry contracts and complete passivity-guard test module |

The MSL cases are independent jobs: long and refinement do not repeat the base
FDTD test. Static mesh/timestep accounting estimates the refinement at 15.77
times the base field-update work. This is a work estimate, not a measured
wall-time ratio: compilation, memory traffic and CPU admission can change
runtime. The 24-hour refinement cap is a budget, not evidence that it will
finish; neither completion within that cap nor admission of these 8-CPU/32-GiB
jobs has been verified. Check active jobs and the completed resource probes before launching all six jobs.

MSL qualification failures and unchanged historical gate failures are recorded
with their true return codes; remaining evidence commands still run. The job
exits nonzero if any command fails. Re-pinning requires a separate review of
qualified evidence and is not an action these jobs perform.

Artifacts live under the unique directory
`/root/workspace/byungkwan-workspace/research/rfx-931-fastlane-reds/output/931-fixture-repair-vessl/issue931-fixture-repair-<lane>-<UTC>-<sha12>-<pid>`.
Evidence and latest pointers stay in the authorized worktree; only the future
pod's isolated temporary clone is outside it.
Each lane has its own `.latest` pointer. The compatible `msl.yaml` file uses
artifact label `msl-base` so its submitter prefix cannot match `msl-long` or
`msl-refine` directories. `commit.txt`, `summary.json`, per-command
logs and return codes, `junit.xml`, the full tee'd `run.log`, and an EXIT-trap
`job-exit.json` make aggregation mechanical. Run names are the exact artifact
directory basename, with command suffixes where the harness accepts a name.
The submitter writes `run_id.txt` using `scripts/vessl_submit.sh`; no pod-provided
run ID is assumed or fabricated. The submitter accepts this artifact root as
its optional third argument (the historical default remains for other callers).
Its temporary timestamp marker precedes run creation, so a fast job cannot
create its evidence directory before the accepted time window begins.

After the final commit, render review copies locally:

```sh
RFX_REPAIR_SHA=$(git rev-parse HEAD)
printf '%s\n' "$RFX_REPAIR_SHA" | grep -Eq '^[0-9a-f]{40}$'
mkdir -p output/931-fixture-repair/jobs
for lane in v173a msl msl-long msl-refine farfield short; do
  sed "s/SET_FULL_COMMIT_SHA_BEFORE_SUBMIT/$RFX_REPAIR_SHA/" \
    "scripts/vessl_931/fixture-repair-$lane.yaml" \
    > "output/931-fixture-repair/jobs/fixture-repair-$lane.yaml"
done
```

Before submission, the submitter must follow the existing VESSL rule's active-run
check and prior-run log backup/cleanup, preserving cited reference runs. Review
the rendered SHA and exact pytest selections. Submit independent jobs together:

```sh
for lane in v173a msl msl-long msl-refine farfield short; do
  case "$lane" in msl) artifact_lane=msl-base ;; *) artifact_lane=$lane ;; esac
  scripts/vessl_submit.sh \
    "output/931-fixture-repair/jobs/fixture-repair-$lane.yaml" \
    "issue931-fixture-repair-$artifact_lane" \
    /root/workspace/byungkwan-workspace/research/rfx-931-fastlane-reds/output/931-fixture-repair-vessl &
done
wait
```

The older `repin-v173a-lock-evidence.yaml` now passes the harness's required
`--output` and `--run-name` arguments. It remains a legacy entrypoint; use the
six reviewed fixture-repair YAMLs for this campaign.
