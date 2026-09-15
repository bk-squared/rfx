# T1 (tests/unit/autodiff, geometry, boundaries) — recompute record

Branch `feat/931-t1-autodiff-geometry-boundaries`, worktree
`/root/workspace/byungkwan-workspace/research/rfx-931-T1-autodiff-geometry-boundaries`.

PI rule, 2026-09-07: the shared pod runs no test lane and no solve longer than
about a minute. Everything below ran on VESSL, one run per case, reading this
worktree from NFS as it stands (commit first — the job does not clone).

The run yamls live in `scripts/` in the worktree but are NOT committed: the
repo's `.gitignore:31` is `**/vessl*.yaml`. Their contents are reproduced at
the end of this file so the ingest phase can recreate them.

## Runs

| run id | what | command | runtime |
|---|---|---|---|
| 369367259135 | the group's three directories at the pre-migration commit 299b0f9f (the failure list this migration worked from) | `vessl run create -f scripts/vessl_931_t1_unit_tests.yaml` | 164 s wall, 12 workers |
| 369367259162 | same lane at 250399d8 (post-migration) | same | ~3 min |
| 369367259185 | same lane at 9ed75fbb, after merging the core's outer-jit fix and the shared `tests/_realized_geometry.py` | same | 195 s |
| 369367259163 | per-bin \|S11\| of the cv11-style PEC short — FAILED to start | — | — |
| 369367259190 | the same probe, re-submitted | `vessl run create -f scripts/vessl_931_t1_pec_short_s11.yaml` | 10 min (4 waveguide S-matrix solves) |
| 369367259195 | FINAL lane at 496b5bd1 | `vessl run create -f scripts/vessl_931_t1_unit_tests.yaml` | 191 s — **5 failed, 784 passed, 9 skipped**; all 5 are the junction fixture below |

Outputs land in
`/root/workspace/claude-workspace/rfx/runs/issue931-post-t1-*` and
`.../issue931-post-t1-pecshort-*`, with a `.latest` pointer beside them.
Two operational notes, both cost a re-submit. `vessl run create` must be
issued from a directory that is NOT a git worktree
— the CLI reads `.git/HEAD` as a directory and a linked worktree has a `.git`
FILE, which raises `NotADirectoryError`. Run it from `/tmp` with an absolute
path to the yaml. And a `run:` block that contains a heredoc is rewritten by
VESSL's own wrapper into something the shell cannot parse (run 369367259163
died at `syntax error: unexpected end of file (expecting ")")` in
`/opt/vessl/scripts/*.sh`, though `sh -n` on the block passes locally); ship
an inline script as `echo <base64> | base64 -d > file` instead.

## Fixtures NOT regenerated, and why

* `tests/fixtures/msl_replay_accumulators.npz`, `msl_replay_golden_f64.npy`,
  `msl_s_matrix_golden.npy` — the MSL trace migrated to a SHEET with the same
  physical corners (§4 rule 1), which lands on node plane 4 (320 um), the
  plane the pre-#931 rule realized. `test_replay_float64_equivalence` is green
  against the committed goldens (it was red at 2.385e-01 against a 1e-5 gate
  while the trace was a volume). Nothing to recapture.
* `tests/unit/autodiff/test_waveguide_sparam_ad.py`'s three sha256-pinned
  golden S-matrices — they are set by the DOMAIN-FACE PEC convention, which
  §1.8 fences out of the contract. Green, unchanged, no recapture.

## Left for the ingest phase

1. CLOSED. `test_real_interior_pec_under_outer_jit_matches_eager` was red on
   this branch until the core landed 49fdf579 ("the zero-cell refusal decides
   emptiness on the host, never through a jnp mask"), merged in here. The
   write-up that reported it,
   `T1-rasterize_grid-outer-jit-tracer-regression.md`, is kept as the record
   of the defect and its reproduction.
2. The five `test_fidelity_topology_findings` reds — three owners, listed in
   that file's module docstring and in `T1-fidelity-sheet-overlap.md` /
   `T1-preflight-junction-plane.md`.
3. CLOSED, measured. `test_pec_short_s11_baseline_unchanged_with_binary_path`
   read min \|S11\| 0.9892 against a 0.99 gate. Run **369367259190**
   (`pec_short_s11.json`) settles it: the realized walls are on the drawn
   faces (planes 28 and 29, 0.084 and 0.087 m), and

   | lane | num_periods | \|S11\| min | \|S11\| max |
   |---|---|---|---|
   | binary | 40 | 0.9892 | 1.0057 |
   | binary | 80 | 0.9974 | 1.0024 |
   | conformal | 40 | 0.9942 | 1.0278 |
   | conformal | 80 | 0.5701 | 2.7691 |

   Three bins above 1 for a passive reflector at 40 periods: the record's
   own noise is +-0.6% there, and the short moved one cell toward the left
   port when it stopped being a zero-thickness wall, so the 40-period window
   no longer contains the settled round trip. The binary test now runs at 80
   periods with the 0.99 gate UNCHANGED, plus a new upper assertion
   (max \|S11\| <= 1.01) so a contaminated record cannot be read as physics
   again.

   OBSERVATION, not fixed and not caused by #931 as far as this run can say:
   the CONFORMAL face-PEC lane diverges at 80 periods on the same geometry
   (\|S11\| from 0.57 to 2.77). Its own test is left at 40 periods, where it
   passes. That lane is §1.8-fenced; someone should look at it.

## Run yamls (gitignored; reproduce from here)

Both are copies of `scripts/vessl_fast_lane_pytest.yaml` with the checkout
retargeted to this worktree.

`scripts/vessl_931_t1_unit_tests.yaml` differs from the fast lane only in:

    name: rfx-931-post-t1-unit-tests
    env.RFX_CHECKOUT: /root/workspace/byungkwan-workspace/research/rfx-931-T1-autodiff-geometry-boundaries
    env.RFX_LABEL: t1
    pytest target: tests/unit/autodiff tests/unit/geometry tests/unit/boundaries
    OUT: /root/workspace/claude-workspace/rfx/runs/issue931-post-t1-<ts>-<sha>

`scripts/vessl_931_t1_pec_short_s11.yaml` uses the same header and runs, in
place of pytest, a probe that for `conformal in (False, True)` and
`num_periods in (40, 80)` builds
`tests.unit.geometry.test_subpixel_pec._pec_short_sim`, records the realized
x wall planes through `realized_pec_edge_masks` / `realized_wall_planes`, calls
`compute_waveguide_s_matrix(num_periods=..., normalize=False)` and writes each
row as JSON to `pec_short_s11.json`.
