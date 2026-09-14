# cv26 round 3 — the lane that produced these artifacts

Every file in this directory comes from lane **`20260913b`**, one sha for the whole
lane: **`a4a0fea1380d2a4600fb499f58185f06a8dacd44`**. Each artifact carries that sha in
its own `commit` field, and `rfx.json` refuses to merge shards that disagree on it
(`scripts/crossval/merge_cv26_arm_shards.py`, `_MUST_AGREE`).

Staged checkout on NFS: `/root/workspace/claude-workspace/rfx/checkouts/cv26-round3-r2`
(`.staged_commit` = the sha above). Shared results directory the shards wrote into:
`/root/workspace/claude-workspace/rfx/runs/cv26r3res-20260913b`.

## Why a whole lane was re-run rather than the failed part

An earlier lane, `20260913a`, ran at the sha before `a4a0fea1`. All seven of its Meep
shards died with `ModuleNotFoundError: No module named 'jax'` before a single Meep step:
the comparator read the auxiliary absorber's constants by importing
`rfx.sources.tfsf_2d`, and the Meep leg imports that comparator from a pymeep conda
environment that has neither rfx nor JAX. `a4a0fea1` fixed that and also re-pointed the
cv04 envelope at the producer's adoption record, which the gated arms read. Both change
what the rfx arms measure, so the eight rfx shards that had succeeded in `20260913a`
were re-run here rather than mixed with the new ones. Nothing from `20260913a` is
committed.

## Regenerate

    python3 scripts/vessl_cv26_shards.py --lane <label> \
        --checkout <NFS checkout> --out-dir <dir>
    # then, per shard yaml, in submission order:
    scripts/vessl_submit.sh <dir>/cv26r3_<shard>.yaml cv26r3-<shard>-

The emitter prints the order. It is a real dependency, not a convention: the `base-*`
and `fals-meep-k2pi` shards read the Meep JSONs from the shared results directory, and
an rfx arm whose Meep JSON is absent takes the `[SKIP]` path and exits 2, which fails a
baseline shard's `rc == 0` check. `base-te00tm00` needs only `meep_te_00.json`.

Merge the baseline arms into `rfx.json` (a concatenation with checks, never a
recomputation):

    python scripts/crossval/merge_cv26_arm_shards.py \
        --in validation/crossval/_26_oblique_results \
        --run-ids validation/crossval/_26_oblique_results/lane_20260913b_run_ids.json

## Run ids

VESSL does not export a run id into the pod on this cluster, so the submitter records
it (`scripts/vessl_submit.sh`). The seven baseline shard ids are also inside `rfx.json`
under `shards[].run_id`; `lane_20260913b_run_ids.json` is the map the merge reads.

| shard | run id | status | artifact it wrote |
|---|---|---|---|
| `meep-te00` | 369367260792 | completed | `meep_te_00.json` |
| `meep-te30` | 369367260885 | completed | `meep_te_30.json` |
| `meep-te45` | 369367260886 | completed | `meep_te_45.json` |
| `meep-te60` | 369367260887 | completed | `meep_te_60.json` |
| `meep-tm45` | 369367260888 | completed | `meep_tm_45.json` |
| `meep-tm60` | 369367260889 | completed | `meep_tm_60.json` |
| `meep-k2pi` | 369367260890 | completed | `meep_te_45__falsifier_k_2pi.json` |
| `base-te00tm00` | 369367260891 | completed | `rfx__shard_te00tm00.json` (te_00, tm_00) |
| `base-graze` | 369367260892 | completed | `rfx__shard_graze.json` (graze_vac, graze_pec, graze_te) |
| `base-te30` | 369367260903 | completed | `rfx__shard_te30.json` |
| `base-te45` | 369367260904 | completed | `rfx__shard_te45.json` |
| `base-te60` | 369367260905 | completed | `rfx__shard_te60.json` |
| `base-tm45` | 369367260906 | completed | `rfx__shard_tm45.json` |
| `base-tm60` | 369367260907 | completed | `rfx__shard_tm60.json` |
| `fals-te60-angle` | 369367260893 | completed | `rfx__falsifier_te_60_angle_m5.json` |
| `fals-te45-swap` | 369367260894 | completed | `rfx__falsifier_te_45_swap_tm.json` |
| `fals-tm60-swap` | 369367260895 | completed | `rfx__falsifier_tm_60_swap_te.json` |
| `fals-te45-eps` | 369367260896 | completed | `rfx__falsifier_te_45_eps_x1p2.json` |
| `fals-graze` | 369367260897 | completed | `rfx__falsifier_graze_pec_{depth,sigma}_half.json` |
| `fals-meep-k2pi` | 369367260908 | completed | `rfx__falsifier_meep_te_45_k_2pi.json` |
| `ladder-depth` | 369367260898 | completed | `rfx__graze_pec_d{8,16,32}.json` |
| `ladder-dx` | 369367260899 | completed | `rfx__{te_30,te_45,te_60,tm_45,tm_60}_dx1.json` |

`completed` is VESSL's word for exit 0, and each shard's exit code is its own
pre-declaration: a baseline shard must exit 0, a pre-declared falsifier MUST exit 1, an
evidence rung may exit 0, 1 or 2. The shard wraps that rule around the case's exit code
and exits 0 only when the rule held, so 22/22 `completed` means 22/22 behaved as
declared — not that every underlying case exited 0. The table above is committed
evidence of the ids; the logs are backed up outside this public repo under
`docs/vessl-logs/` (gitignored).

## Preflight

There is none to quote. `validation/crossval/26_oblique_slab_fresnel.py` drives the Yee
core directly (`init_state` / `update_e` / `update_h` / `init_cpml` / `init_tfsf_2d`)
rather than through `Simulation.run()`, so `sim.preflight()` never runs on this case.
The rig checks it does run are in-script assertions on the cell bookkeeping (`cfg.x_lo`
/ `cfg.x_hi`, the slab and probe clearances, `dt` against the declared value) plus the
per-arm settling witness `G3_tail`, which every arm carries in its `tail` block.
