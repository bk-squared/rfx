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

---

## Results (appended AFTER the measurements; no threshold above was changed)

Profile-level falsifiers, measured post-fix (tests/test_auto_dz_profile_preserve.py):

- (a) HELD: demo-fixture substrate-top edge distance = 0.0 m (<= 1e-12 m).
- (b) HELD: post-thirds block [63.5, 63.5, 63.5, 42.333, 21.167] um passes
  through bit-identically (np.array_equal); dz_min unchanged at 21.167 um,
  so the demo dt is unchanged.
- (c) HELD: sum(dz) = 1.754 mm, |err| = 4.3e-19 m (<= 1e-12 m); nz 24 -> 19.
- Generic two-layer fixture: all four interfaces within 2.2e-19 m of the
  declaration, total exact.
- Revert-proof: with the fix stashed, the falsifier tests fail on the
  pre-fix builder (fraction 0.3462 / 2.3800 mm reproduced).

Impact sweep: NU battery (nonuniform/auto_config/mesh_planner, not
gpu/slow) 253 passed; auto-profile consumers + preflight NU tests 328
passed; MSL battery subset 100 passed + 1 pre-existing xfail. Exactly ONE
locked value moved: test_make_dz_profile_applies_thirds_rule's global
max-ratio <= 1.3, re-pinned to free-run-only with provenance (commit
6973459) — the in-block 2/3 -> 1/3 thirds splits are the declaration's
own construction and smoothing them away was the defect.

FDTD A/B (one attempt, run after this note's threshold was committed;
docs/design_notes/issue763_fix_ab_results.json, script byte-identical to
agent/graded-z-lowz-demo @ 11c65c0):

- Graded arm B (production profile, fixed): median |Z0 err| = 3.33 %
  < 5 % envelope -> the pre-declared A/B falsifier HELD (pre-fix measured
  +6.58 %, F3 fired). Uniform control arm A: 1.27 %. F1/F2/F4 held on
  both arms; settling precondition OK.
- Realized graded column 1.754 mm (= declared), nz_B = 19,
  dz_min = 21.167 um (unchanged, so arm-B dt is unchanged).
- As pre-declared: F5 (z-cell ratio 1.47 < 2.0) and F6 (z-cost ratio
  1.86 >= 1.0) STILL FIRE — the fix does not rescue the efficiency
  claim; the graded-z efficiency demo stays STOPPED.

Observed side finding (NOT addressed here, same declared-vs-realized
family as #745, preflight edition): the MSL-port preflight substrate-cell
and mixed-cell-danger-zone checks evaluate h_sub against the UNIFORM dx
(190.5 um -> "1 substrate cell", "h_sub/dx = 1.333") on both arms, even
though arm A's realized z-mesh is exactly aligned (dz = h_sub/4) and arm
B's realized interface is now on a cell edge. The realized-profile-aware
version of these checks is follow-up material.
