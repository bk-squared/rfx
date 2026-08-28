# Issue #763 — `_make_dz_profile` must hand `smooth_grading` the substrate block as `preserve_regions`

**Status:** pre-declaration (this commit precedes any fix or fix-validation measurement).
**Class:** declared-vs-realized, mesh edition (family: #740, #745, #752).

## Defect (measured on main @ b29f9de, reproduced deterministically, zero FDTD)

`_make_dz_profile` (`rfx/auto_config.py:659`) returns
`smooth_grading(cells, max_ratio=1.3)` with no `preserve_regions`, although the
kwarg exists and `tests/test_smooth_grading_preserve.py` pins exactly this
failure mode. On the declared demo fixture (h_sub = 254 um, eps_r = 3.38,
W = 6*h_sub low-Z MSL, dx = W/8 = 190.5 um, phys_z = h_sub + 1.5 mm margin
= 1.754 mm), production code realizes:

- nz = 24, sum(dz) = 2.3800 mm (declared 1.754 mm)
- dz_min = 21.167 um
- 254 um substrate-top interface mid-cell at fraction 0.3462
  (inside preflight's own [0.10, 0.40] mixed-cell danger zone)

These match the issue #763 / adversarial-reviewer numbers exactly and were
re-reproduced in this worktree before this note was committed.

## Fix design (declared before implementation)

1. `_make_dz_profile` records the realized index range of each dielectric
   feature block after `apply_thirds_rule` (the thirds rule preserves each
   side's sum, so every declared interface remains a cell edge post-thirds),
   and passes those ranges to `smooth_grading(..., preserve_regions=...)`.
2. `preserve_regions` alone is NOT sufficient for falsifier (c): transition
   cells are still inserted in the free (air) runs and inflate the column.
   Therefore each free run is renormalized back to its declared physical
   length after smoothing: first remove duplicated plateau (coarse) cells
   while at least two identical max-size neighbours remain and the run stays
   >= its declared length, then uniformly rescale the run's cells by
   f = L_declared / L_run (f <= 1 by construction; f == 1 -> no-op).
   Renormalization touches ONLY free-run cells, never a protected block, so
   every declared interface coordinate and the total column become exact
   while the block cells stay bit-identical.
3. Behavior with no z-features (uniform early-return path) is unchanged.
4. Sibling check: `_make_dz_profile` is the ONLY production caller of
   `smooth_grading` in `rfx/` (verified by grep; `mesh_planner.plan_mesh`
   wraps `auto_configure` and builds no profile of its own; there is no
   production dx/dy profile builder — tests/scripts that call
   `smooth_grading` directly own their fixtures). No sibling fix needed;
   this claim is falsified if any `rfx/` module besides `auto_config`
   invokes `smooth_grading`.

Known pre-existing limit, explicitly OUT of scope: an inter-feature gap or
top air region <= dx/2 is dropped entirely by `_make_dz_profile` (subsequent
coordinates shift by that gap). Unchanged by this fix.

## Fix-validation falsifiers (pre-declared, profile level, no FDTD)

Exactly as issue #763 declares, with numeric tolerances held from this commit
onward (never to be widened after measurement):

- **(a) Interface snap:** on the demo fixture, some realized cell edge
  (cumulative sum of dz including 0) lies within **1e-12 m** of the declared
  z = 254 um substrate-top coordinate. (Main currently fails: interface is
  mid-cell at fraction 0.3462.)
- **(b) Protected block bit-identical:** the post-thirds substrate-block cell
  widths appear verbatim (numpy bitwise equality, `np.array_equal`) as a
  contiguous run in the returned profile — pre/post smoothing identical to
  the last bit.
- **(c) Column length:** on the demo fixture,
  |sum(dz) - 1.754e-3 m| <= **1e-12 m**. (Main currently realizes 2.3800 mm,
  0.626 mm over.)

Generic fixture (same tolerances): a two-layer stack with an interior air gap
(layer1 0.2-0.5 mm, layer2 1.1-1.35 mm, phys_z 3.0 mm, dx 0.3 mm) must
realize ALL four declared interface coordinates on cell edges within 1e-12 m
and total column within 1e-12 m of 3.0 mm.

Revert-proofing: the regression test first documents that the pre-fix
production values (fraction 0.3462, sum 2.3800 mm) were reproduced before the
fix was applied (recorded above); the test itself asserts the (a)-(c)
invariants, which main @ b29f9de fails.

## Impact sweep rule (pre-declared)

This changes every auto-configured NU z-mesh (cell count, dz values, dt via
dz_min, memory estimates). For every locked test value that moves, the NEW
value is accepted ONLY if the realized mesh now matches the declaration
(interface on edge, column length exact) and the moved value follows from
that by arithmetic/physics; re-pin WITH provenance in the commit message.
Any moved value that cannot be justified physically => STOP and report.

## Optional FDTD A/B fix-validation (pre-declared BEFORE running; one attempt)

If run (CPU, demo fixture, both arms ~45-90 s each): the graded arm's
**median |Z0 error| must drop below the 5% committed envelope** (pre-fix
measured +6.58%; uniform-z control 1.14%). One attempt, no re-rolls.
This validates the FIX only. The efficiency demo's F5/F6 verdicts
(z-savings 1.54x < 2.0, z-cost 1.78x >= 1.0, thirds-rule dt penalty) are
UNAFFECTED by this fix and that demo stays STOPPED.
