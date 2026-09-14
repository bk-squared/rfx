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

## Lane 20260914a — the settling re-run of pre-declaration section 19 (append-only)

Two arms only, at sha **`708d7eef960d510b252679cc7195221a60a94f3b`**, staged checkout
`/root/workspace/claude-workspace/rfx/checkouts/cv26-rerun-20260914a`, shared results
`/root/workspace/claude-workspace/rfx/runs/cv26r3res-20260914a`. Nothing from lane
20260913b is touched: these are `--tag` diagnostic rungs, so they write
`rfx__<tag>.json`, and `--settling-bar` REQUIRES `--tag` precisely so a tightened-bar
run can never land in `rfx.json`.

| shard | run id | status | artifact |
|---|---|---|---|
| `rerun-te00-s60` | 369367260941 | completed | `rfx__te_00_settle60.json` |
| `rerun-tm00-s60` | 369367260942 | completed | `rfx__tm_00_settle60.json` |

Emit and submit:

    python3 scripts/vessl_cv26_shards.py --rerun --lane <label> \
        --checkout <NFS checkout> --out-dir <dir>
    scripts/vessl_submit.sh <dir>/cv26r3_rerun-te00-s60.yaml cv26r3-rerun-te00-s60-

These two arms carry a DIFFERENT sha from the rest of the directory, and that is the
point: they were run after the `--settling-bar` knob existed. They are not merged into
`rfx.json` and never can be — `merge_cv26_arm_shards.py` reads only `rfx__shard_*.json`
and refuses shards whose `commit` disagrees.

## Why `meep_te_00.json`'s run id sat outside lane 20260913b's block (superseded by lane 20260914b below)

`meep_te_00.json` was written by run **369367260792** at 2026-09-13T19:12:43Z, while the
other six Meep legs are 369367260885-890 from 2026-09-14T06:10Z. All seven ran at the
SAME sha `a4a0fea1`: te00 was submitted first, on its own, as the check that the Meep
import fix worked before the remaining six were committed to. Its output was kept rather
than re-run at no benefit. The sha is what the lane pins, and it agrees.


## Lane 20260914b — the Meep legs re-run so `commit` is native (append-only)

The seven Meep records committed here are now from lane **`20260914b`** at sha
**`99ca7553aed272f5699ee39539b06fa03e806033`**, staged checkout
`/root/workspace/claude-workspace/rfx/checkouts/cv26-meepcommit-20260914b`, shared
results `/root/workspace/claude-workspace/rfx/runs/cv26r3res-20260914b`. This also
retires the out-of-block run id explained in the section above: all seven now come from
one lane and one sha.

**Why they were re-run rather than annotated.** The Meep leg did not record the sha it
ran at, so these were the only artifacts of this case carrying no code state, and
`merge_cv26_arm_shards.py`'s `_MUST_AGREE` guard had nothing to check them against. The
leg records `commit` now. The committed records were NOT hand-edited to insert the field:
an artifact that says it was produced by a run it was not is worse than the gap it
closes, so the legs were re-run and the real records replaced the old ones.

| shard | run id | status | artifact |
|---|---|---|---|
| `meep-te00` | 369367260956 | completed | `meep_te_00.json` |
| `meep-te30` | 369367260957 | completed | `meep_te_30.json` |
| `meep-te45` | 369367260958 | completed | `meep_te_45.json` |
| `meep-te60` | 369367260959 | completed | `meep_te_60.json` |
| `meep-tm45` | 369367260960 | completed | `meep_tm_45.json` |
| `meep-tm60` | 369367260961 | completed | `meep_tm_60.json` |
| `meep-k2pi` | 369367260962 | completed | `meep_te_45__falsifier_k_2pi.json` |

**Why the rfx side did not need re-running.** The only code change on the leg is the
`commit` field; the Meep physics is untouched and Meep is deterministic. Every measured
field of all seven records is **bit-identical** to what lane 20260913b wrote — `R_meep`,
`T_meep`, `freqs_hz`, `acceptance`, `accepted`, `k_point`, `ky_declared_rad_m`,
`fcen_meep`, `fwidth_meep`, `dt_meep_s`, `resolution`, `courant`, `nfreq`, `a_m`,
`geometry` — with only `commit` added and the two wall-clock timings `run.t_ref_s` /
`run.t_slab_s` moved. That was checked field by field BEFORE the swap, because the E4
block stored in `rfx.json` is an evaluation of these files: had any measured field moved,
the stored E4 would no longer describe the file beside it and the swap would have been
wrong. `rfx.json` and every `rfx__shard_*.json` are byte-identical to what they were.

**The guard that now exists.** `26_oblique_slab_fresnel.py` records `meep.leg_commit` and
`meep.leg_commit_matches_run` per arm and prints a warning when they disagree. Records
written before the leg recorded a commit read as `None` and are reported as such, never
as agreement. Demonstrated on real records at this sha: matching gives
`leg_commit_matches_run = true` with no warning, and the same run against a copy whose
`commit` was changed prints

    E4 WARNING: the Meep leg records commit 000000000000 but this run is 99ca7553aed2
    -- the reference is from a different code state

It is a report, not a gate: the numbers are what they are, and a reader who sees the
warning knows to check the provenance rather than having the comparison silently refused.

## Three things a reader of this directory should know (append-only, 2026-09-14)

**The E4 commit guard fires on this directory today, and that is expected.** `rfx.json`'s
E4 blocks were evaluated at sha `a4a0fea1` against the lane-20260913b Meep legs; the legs
on disk now carry `99ca7553`, because they were re-run so their sha would be native. Any
run that re-evaluates E4 here will therefore print

    E4 WARNING: the Meep leg records commit 99ca7553aed2 but this run is <...>
    -- the reference is from a different code state

That warning is correct and the situation is benign, for a reason that was CHECKED rather
than assumed: every measured field of all seven re-run records is bit-identical to what
lane 20260913b wrote, so the stored E4 is still an exact evaluation of the files beside
it. The guard cannot know that, which is why it warns rather than staying silent. It is a
report, not a gate.

**`staged_commit` now carries the evidence that it was checked.** `rfx.json` records
`staged_commit_checked_across`, the list of shard files whose `commit` was compared
against it by `merge_cv26_arm_shards.py`'s `_MUST_AGREE`. Without that list the key is
indistinguishable from a copy of the first shard's commit.

**The two `settle60` rungs have `commit` but no `staged_commit`, and that is correct.**
`commit` is the sha the run itself was staged at, written by the case. `staged_commit` is
a MERGE-level property — the one sha every shard of a merged document agreed on — so it
exists only in `rfx.json`. A single-arm tagged rung has no shards to agree, and inventing
the key there would assert a check that never happened.

## Derived lattice-witness numbers live in `lattice_witness_replay.json`

The case script writes `lattice.W_witness_*`, `GL1_*` and `GL2_*` per arm, but that
plumbing landed AFTER the lanes that produced these records, so none of them carries those
keys. `lattice_witness_replay.json` recomputes the derived witness from the committed
per-arm records — no FDTD, the same arithmetic `evaluate_e2` does — and names every input
(which record, that record's own commit, the arm's own `src_tau_s`, the standard and
section). It is labelled `"kind": "replay"` so a recomputation cannot be read as a
measurement. Regenerate with

    python validation/crossval/comparators/emit_cv26_lattice_witness_replay.py

It reports `tm_60` FAILING GL2_R. See close note §10.5.

Since 2026-09-14 (#1015) it also carries `domain_R` / `domain_T` per arm: the bins on
which the standard's window is a valid bound at all, and how the GL1 breaches split
across that line (standard §13). Nothing numeric moved when those keys were added —
every value the file already carried is bit-identical, and `tm_60` still fails GL2_R.
