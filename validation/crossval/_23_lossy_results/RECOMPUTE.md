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
