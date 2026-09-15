# post-#931 recompute — cv24

VESSL run id **369367259203**, submitted from branch `feat/931-crossval-E`
(worktree `/root/workspace/byungkwan-workspace/research/rfx-931-XE-crossvalE`).

Run block, command, preset, expected runtime, the pre-recorded prediction for
this case, and which artifact keys are expected to move:
`scripts/vessl_issue931_post_XE/RECOMPUTE.md` (single source; do not
duplicate the table here).

Submit with:

    cd /tmp   # NOT inside the worktree: the VESSL CLI reads .git/HEAD as a
              # directory and a linked worktree's .git is a file
    vessl run create -f <repo>/scripts/vessl_issue931_post_XE/post-cv24.yaml

Artifacts land in
`/root/workspace/claude-workspace/rfx/runs/issue931-post-cv24-<ts>/`.
Nothing is written back into this branch; the ingest phase commits.

An earlier submission of this same case is SUPERSEDED (see the single source above for its id and why); do not ingest its artifacts.
