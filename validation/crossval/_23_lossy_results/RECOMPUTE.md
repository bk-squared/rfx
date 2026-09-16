# post-#931 recompute — cv23

VESSL run id **369367259202**, submitted from branch `feat/931-crossval-E`
(worktree `/root/workspace/byungkwan-workspace/research/rfx-931-XE-crossvalE`).

Run block, command, preset, expected runtime, the pre-recorded prediction for
this case, and which artifact keys are expected to move:
`scripts/vessl_issue931_post_XE/RECOMPUTE.md` (single source; do not
duplicate the table here).

Submit with:

    cd /tmp   # NOT inside the worktree: the VESSL CLI reads .git/HEAD as a
              # directory and a linked worktree's .git is a file
    vessl run create -f <repo>/scripts/vessl_issue931_post_XE/post-cv23.yaml

Artifacts land in
`/root/workspace/claude-workspace/rfx/runs/issue931-post-cv23-<ts>/`.
Nothing is written back into this branch; the ingest phase commits.

An earlier submission of this same case is SUPERSEDED (see the single source above for its id and why); do not ingest its artifacts.

## also pending: the #888 auxiliary absorber

The records here were produced with the 20-cell TF/SF auxiliary absorber. #888
replaced it with a 200-cell absorber derived from a reflection target
(`rfx/sources/tfsf.py::AUX_N_CPML_1D`), so the injected incident field changed
after they were written. Their `aux_echo` blocks declare the layout they were
written under and are replayed against it
(`tests/crossval/test_aux_echo_record_invariant.py`); the shipped absorber's
arrival is 19-20 steps EARLIER at every rung and every committed record stays
admissible under it (ratios 0.520-0.599 -> 0.525-0.604, limit 1.0). Fold this
into the post-#931 recompute above rather than running a second job, and then
take the case out of `_ABSORBER_RECOMPUTE_PENDING` in that test.

This case's `W_BIN` is derived from cv04's envelope, which the derived absorber
also invalidates — loose direction only, magnitude unestablished, re-derivation
is an explicit adoption edit under #928. Read
`validation/crossval/_04_fresnel_results/RECOMPUTE.md` before re-pinning any
window here.


## the falsifier artifacts' `gates` dicts are old-window verdicts (#928 item 2)

`rfx__falsifier_tand0p1_sigma_x1p5.json`, `rfx__falsifier_tand1_sigma_x1p5.json`
and `rfx__falsifier_tand3_sigma_x1p5.json` record gates as `true` that the live
script would no longer write: E2 windows are now derived per arm from the arm's
own lattice and record, not from cv04's per-bin max `|R+T-1|`, and under them
those three fail `G1_R`/`G1_A`, `G1_T`/`G1_A` and `G1_T`/`G1_A`/`G2_T`
respectively. Each record's verdict (exit 1) is unchanged, and the gate tests
here replay them against the windows they were judged by, saying so. Every
before/after verdict is tabled in
`docs/design_notes/slab_family_per_arm_lattice_window_predeclaration.md`
section 6.1.

## r4 — the shipped-absorber record, appended and unadopted (2026-09-16, #928 item 3)

**Run: 2026-09-16. Adopted: no.** `r4/rfx.json` is this case re-run end to end on
the absorber the repository ships, carrying the `GL_witness` gate the committed
record predates. It sits beside the committed `rfx.json`, which is unchanged;
nothing reads it, `_ABSORBER_RECOMPUTE_PENDING` still names cv23, and the gate
tests still replay the committed record. What closes those is an adoption
change, which is a separate reviewed step by construction.

### What it is

    python validation/crossval/23_lossy_slab_fresnel.py \
        --out-dir <tmp> --meep-dir validation/crossval/_23_lossy_results --no-plots

**local CPU** (jax 0.6.2, CpuDevice), three arms, 17.6 s of FDTD
(2.6 / 2.5 / 12.5 s), exit 0, every gate PASS on R, T and A. The Meep leg is the
committed one: this run recomputed the rfx side only and read the same
`meep_*.json` files. `r4/commit.txt` and `r4/run.log` are the source commit and
the full stdout.

Two things the committed record cannot carry:

* **`GL_witness`.** The committed `rfx.json` was written 2026-09-02, before the
  lattice-witness gate was wired into `main()` (#970), so its `gates` dict has
  no `GL_witness` key at all. Nothing checks for it, so nothing is red — it is
  simply absent. In r4 it is present and TRUE on all three arms, with
  `|rfx - lattice|` in the gated mean of R at 2.62e-05 (tand0p1), 2.42e-05
  (tand1) and 4.96e-06 (tand3).
* **The #888 absorber.** The committed record was produced with the 20-cell TF/SF
  auxiliary absorber; r4 ran on the shipped 200-cell one derived from a
  reflection target. That is the recompute the section above says this case owes.

### How it compares to the committed record

Every observable reproduces. Band means move at 3e-5 to 9e-4 relative; the
per-bin maxima, which are a max over 229 bins of a float32 field, move at 0.4 %
to 1.5 %:

| arm | `mean_dR_gated` | `max_dR_gated` | `mean_dA_gated` |
|---|---|---|---|
| tand0p1 | 0.00391879 vs 0.00391892 (3.3e-05) | 0.0139942 vs 0.0140481 (3.8e-03) | 0.00194401 vs 0.00195146 (3.8e-03) |
| tand1 | 0.00509565 vs 0.00509398 (3.3e-04) | 0.00784294 vs 0.00773809 (1.4e-02) | 0.00333944 vs 0.00333630 (9.4e-04) |
| tand3 | 0.00311345 vs 0.00311520 (5.6e-04) | 0.00462759 vs 0.00469931 (1.5e-02) | 0.00308354 vs 0.00308533 (5.8e-04) |

(r4 vs committed, relative difference in brackets.) Record lengths are identical
(1067 / 1158 / 2362 steps — the record recipe is deterministic) and so are the
grids. What is left is the field arithmetic: the committed record came from a
VESSL GPU job (run 369367259202, commit `796aa1e6`) and this one from a CPU,
both float32. Two independent local runs of r4 reproduced each other exactly.

### Why it is not adopted here

Three reasons, none of them a reason to withhold the evidence:

1. The committed record is claims-bearing and was produced on this case's own
   VESSL lane, with the run id, the commit and the log tracked. Swapping in a
   local CPU record changes the measurement lane and belongs in a change that
   says so.
2. `scripts/vessl_issue931_post_XE/RECOMPUTE.md` is the single source for this
   case's post-#931 recompute, including a pre-recorded prediction of which
   artifact keys move. r4 was not run under that procedure, so it does not close
   it.
3. The windows r4 was judged by are the per-arm ones (#928 item 2, design note
   `docs/design_notes/slab_family_per_arm_lattice_window_predeclaration.md`).
   Until that change is merged, a record carrying them describes gates that are
   not yet the repository's.

### What adopting it would take

Promote `r4/rfx.json` to `rfx.json` (keeping r4 as the provenance copy), take
`"cv23"` off `_ABSORBER_RECOMPUTE_PENDING` in
`tests/crossval/test_aux_echo_record_invariant.py`, and re-read whatever the
gate tests pin by value. The one thing that cannot be decided from these numbers
is (1).

**Not a file move on its own.** r4's `gates` dict carries NEW-window verdicts --
it was judged by the per-arm windows, and it is the first cv23 record that will
be -- so the gate-test call sites that compare a committed `gates` dict against
a replay have to take the per-arm windows IN THE SAME COMMIT as the promoted
record, or that comparison reds. Those call sites are explicit about which
window they use (`_replay_e2` for an artifact's own recorded verdicts,
`_live_e2` for the live gate, `tests/crossval/test_cv23_lossy_slab_gates.py`);
adopting r4 makes the per-arm windows the reference for this record and retires
the old-window replay for it. Doing the file move without the call sites leaves
a red suite; doing the call sites without the record leaves the committed
old-window verdicts unexplained. The section above, on the falsifier artifacts,
is the same fact one directory over.
