# Group X-E post-#931 recompute — what was submitted, and what each run decides

Branch `feat/931-crossval-E`, worktree
`/root/workspace/byungkwan-workspace/research/rfx-931-XE-crossvalE`.
Every run is one case, submitted with `vessl run create -f <this dir>/<file>`
(from a directory that is NOT a git worktree — the VESSL CLI reads `.git/HEAD`
as a directory and a linked worktree's `.git` is a file, so submitting from
inside the worktree fails with `NotADirectoryError`).

Each yaml is a copy of that case's own pre-change baseline
(`scratchpad/vessl_baseline/<case>.yaml`) with the checkout retargeted to this
worktree and the run renamed `rfx-931-post-<case>`. Same image, same preset,
same invocation, same harvest, so the post artifact is comparable to the
baseline artifact line for line. The job copies the worktree to `/root/work`
and runs there, so **nothing is written back into the branch**; every produced
file is harvested under
`/root/workspace/claude-workspace/rfx/runs/issue931-post-<case>-<ts>/produced/`
for the ingest phase to commit. No fixture is committed on this branch.

| case | run id | preset | timeout | expected | what it decides |
|---|---|---|---|---|---|
| cv20-producer | **369367259200** | gpu-rtx4090 (JAX_PLATFORMS=cpu) | 3 h | ~26 min (baseline elapsed_s 1543.7) | the rfx-side MSL fixture, re-solved under the volume rule |
| cv22 | **369367259201** | gpu-rtx4090 | 3 h | ~1 h | dielectric control: must be BIT-IDENTICAL |
| cv23 | **369367259202** | gpu-rtx4090 | 3 h | ~1 h | dielectric control: must be BIT-IDENTICAL |
| cv24 | **369367259203** | gpu-rtx4090 | 2 h | ~1 h | boundary-PEC control: must be BIT-IDENTICAL |
| thru-feedpost | **369367259204** | gpu-rtx4090 | 4 h | ~1-2 h | the trace-as-sheet research lane, `--verify` arm |
| tmtt-msl-stub | **369367259205** | gpu-rtx4090 | 6 h | ~2-4 h | sheet feed line + volume stub, AD arm + cross-solver gate |
| tmtt-beam-steer | **369367259206** | **gpu-a6000-1** | 8 h | ~4-6 h | reflector as a declared sheet, SMOKE=0 |

### Superseded first submission — ignore, do not delete

An earlier set of the same seven runs was submitted at 10:30 UTC and is
SUPERSEDED: 369367259171 (cv20-producer), 369367259173 (cv22), 369367259176
(cv23), 369367259180 (cv24), 369367259183 (thru-feedpost), 369367259187
(tmtt-msl-stub), 369367259188 (tmtt-beam-steer). They were submitted before
this branch was rebased onto four new commits on `feat/931-lattice-ownership`
(including a rasterizer fix and the shared `tests/_realized_geometry.py`
helper this group's gate now delegates to), and the jobs copy the worktree at
container start, so which tree they actually read is not knowable from the
outside. Provenance that cannot be pinned is not provenance. The run ids in
the table above are the definitive set; the superseded ones are left running
rather than deleted (VESSL runs are not deleted here), and their artifacts
must not be ingested.

`tmtt-beam-steer` is the one case that is NOT on a 4090: the full-resolution
forward plus `value_and_grad` OOMs on 24 GB (recorded in the baseline set;
`tmtt-beam-steer-a6000.yaml` is the baseline this copy is modelled on).

## Predictions, recorded BEFORE the runs land

Pass or fail, these are reported verbatim against what comes back.

**cv20-producer.** Measured at build time (no solve) on this branch, the rfx
trace realizes wall planes `[5, 6]` at z = 250 and 300 um, `t_metal = 50 um` =
one cell, `trace_realization_kind = "volume"`, and realized trace y bounds
900 -> 1500 um (width 600 um = drawn). The new fixture `meta` must carry
exactly those. `h_sub_realized_m` stays 0.0003 and `n_z_sub_realized` stays 6
(dielectric; the contract does not move it). The S11/S21/Z0/beta arrays MUST
move — the realized trace gained its wall at 250 um and a node row of width —
so an unchanged array is a failure of this migration, not a pass.

**cv22 / cv23 / cv24.** Bit-identical to the committed artifacts. These three
contain no conductor body (cv22/cv23 are dielectric slabs, cv24's walls are
the domain boundary), and each now asserts that at build time. If any of them
moves, the change leaked outside the conductor path and THAT is the finding —
not a window to widen.

**thru-feedpost.** Every pre-declared window in
`docs/design_notes/thru_feedpost_twoseg_predeclaration.md` was measured on the
pre-#931 realization and is NOT touched by this branch. The realized trace
plane is unchanged (z = H, the node the old single wall landed on) but the
realized WIDTH gains a cell (the old node sampler dropped the hi row; a sheet
footprint is closed), so Zc moves. Expect `F_I3_ZC = (44.0, 53.0)` ohm and
`ZC_CENTER = 48.25` to need re-derivation. Report the measured Zc against the
old window verbatim; do not widen it.

**tmtt-msl-stub.** The notch position moves: the trace is now a sheet at
254 um with realized width 635 um (the pre-#931 rule realized a single wall at
254 um with a narrower footprint). The paper's -45.9 dB at L ~ 7.0 mm and the
`L_TARGET_AN` comparison are both re-measured. The cross-solver gate compares
the imperative volume stub against the AD volume stub and must still agree —
that identity is also checked without a solve by
`assert_soft_pec_equals_hard`, which passed on this branch (3909 hard edges,
2353 of them the declared sheet, zero disagreement).

**tmtt-beam-steer.** The reflector's realized WALL PLANE is unchanged (index
32, exactly `plate_z`), so the dipole is still lambda/4 above the metal. The
plate's realized APERTURE changes from 31 cells to 30 = 149.896 mm, which is
the declared 1.5 lambda exactly — the old `pec_mask_override` cell write was
one cell too wide. Expect the 5.9 dBi bare-plate reference and the 9.5 / 9.45
dBi steered numbers to move slightly; a large move is a finding, not a
tolerance.

## What is NOT submitted here, and why

**cv20 Stage B (openEMS).** It must be re-run, but not by this group's runs:
the rfx fixture has to land first, because Stage B builds its openEMS trace
from the fixture's realized bounds. Those bounds move one cell in y
(950/1550 -> 900/1500 um), for two reasons at once — the realization changed,
and the committed values came from `sim.fidelity_report()`, whose conductor
rows are computed with the shape's NODE sampler while the solver realizes PEC
volumes from cell CENTRES (measured; see `_realized_trace_geometry` in the
producer). `_stage_b_layout` now REFUSES a pre-#931 fixture by name, so cv20
cannot silently run on the old board. Sequence: (1) this cv20-producer run,
(2) ingest the fixture, (3) re-run cv20 Stage B on the openEMS lane, (4)
supersede — never edit — the two committed
`_20_msl_phase_referee_logs/*_result.json` history artifacts with a run-3.

**cv21.** Fenced by design note section 1.8 (sigma-stamped conductors are not
PEC realization). Nothing in it changed, so nothing is re-run.

**convergence_floor / w4r / issue683 / issue764 / issue770.** Their geometry
changed (node-plane redraw, or a realization that gained the far face), so
their committed numbers are stale, but each has its own pre-declared window
set that must be re-derived BEFORE the gates are read again — which is a
decision, not a recompute. They are listed in the group report, not submitted.

## Known red on this branch, one cause, cleared by the cv20 fixture ingest

`tests/crossval/test_msl_phase_referee_header.py` — **14 failed, 49 passed**
(measured on this branch, 2026-09-07). Every one of the 14 fails inside
`_stage_b_layout`, on the same `KeyError`: the committed fixture
`tests/fixtures/msl_phase_referee/msl_thru_rfx_dx50.json` carries no
`meta['trace_wall_planes_realized']`, i.e. it predates the contract. That
refusal is the intended behaviour — cv20 must not run Stage B against a board
solved under the old realization — and it clears when run 369367259171's
fixture is ingested. Anything still red after that ingest is a real pin on a
pre-contract number and must be re-derived, not relaxed.
