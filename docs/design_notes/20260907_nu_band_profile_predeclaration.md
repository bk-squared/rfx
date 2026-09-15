# Axis-agnostic band profile builder (`make_band_profile`) — pre-declaration

**Status:** pre-declaration. This commit precedes any fix, any builder code
and any fix-validation measurement. Every numeric window below is frozen
from this commit onward; results are appended under "Results", never
edited into the windows.
**Class:** declared-vs-realized, mesh builder edition (family #325, #763;
siblings #740, #745, #752, #811).
**Tree:** worktree `rfx-nu-band`, branch `feat/nu-band-profile`, based on
`origin/main d990e18c` (`d990e18ce0870ac7a893a86627e7232d1cf92c7b`).
**Import provenance for every number here:** `rfx.__file__` printed as
`/Users/byungkwankim/Documents/rfx-nu-band/rfx/__init__.py` (PYTHONPATH
pinned to this tree; the editable install points at the primary checkout).
Date: 2026-09-07 (KST).

## 1. Defects, re-reproduced on this tree (zero FDTD)

The lead measured all three on `main d990e18c`. Each was re-run here with
the tree pinned; numbers match unless a discrepancy is stated.

### Defect 1 — `rfx/auto_config.py::_make_dz_profile` on a 5-layer PCB stack

Fixture: layers from z = 0.5 mm upward: core 0.8, prepreg 0.1, core 0.8,
prepreg 0.1, core 0.8 mm, all eps_r 4.3; `domain_z` 4.0 mm; `dx` 0.2 mm
(`_make_dz_profile(feats, 4.0e-3, 0.2e-3)`).

Measured here: nz = 45, sum = 4.000000 mm (exact), dz_min = 8.333 um,
max adjacent ratio **8.000** at index 30 (8.333 um next to 66.667 um), 25
adjacent ratios > 1.4, every declared interface on a node (max error
8.7e-19 m). Realized cells (um):

```
[136.0, 104.6, 90.7, 69.7, 53.6, 45.3 | 66.7, 133.3, 200, 200, 133.3, 66.7 |
 8.33, 16.67, 25, 25, 16.67, 8.33 | 53.3, 106.7, 160, 160, 160, 106.7, 53.3 |
 8.33, 16.67, 25, 25, 16.67, 8.33 | 66.7, 133.3, 200, 200, 133.3, 66.7 |
 51.2, 66.5, 86.5, 102.3, 133.0, 153.5, 153.5, 153.5]
```

Matches the lead (nz 45, 8.333 um, ratio 8.000 at idx 30, 25 ratios > 1.4).
Cause, confirmed by reading the code: each feature block is realized
uniform (`n = max(4, ceil(thickness/dx))`), `apply_thirds_rule` then splits
the cells on BOTH sides of every boundary index into 2/3 + 1/3, and
`_smooth_preserving_blocks` smooths only the free (air) runs — a seam
between two adjacent protected blocks (core | prepreg: 200/3 = 66.7 um next
to 25/3 = 8.33 um) can never receive a ramp.

Two facts the reading adds to the lead's statement:

- The thirds rule fires at **every** feature boundary index, including a
  dielectric | dielectric seam (core | prepreg) where its stated rationale
  ("1/3 inside the conductor, 2/3 outside") has no conductor. This is where
  the 8.33 um cells — the dz_min of the whole column — come from. Ownership
  question, #931, out of scope here (section 5).
- The middle core (1.4 to 2.2 mm) realizes **5 cells of 160 um** while the
  two identical 0.8 mm cores realize 4 cells of 200 um: `2.2e-3 - 1.4e-3`
  evaluates to `8.000000000000002e-4`, so `ceil(span/dx)` returns 5. A
  floating-point ceiling on a physically exact quotient. Design refinement
  R3 below.

### Defect 2 — `rfx/nonuniform.py::make_z_profile` has no descending ramp

Fixture: `make_z_profile([1.0e-3, 1.2e-3, 2.5e-3, 2.7e-3], 4.0e-3,
50e-6, 200e-6, 1.4)`.

Measured here: nz = 32, sum = 4.000000 mm, all five feature coordinates on
a node (max error 4.3e-19 m), but three adjacent ratios above the cap:
**4.527** at index 6 (226.36 -> 50 um), 3.764 at index 19 (188.18 -> 50 um),
1.600 at index 22 (80 -> 50 um); last cell 188.18 um, not the 50 um
boundary cell the docstring ("fine -> coarse -> fine") implies. Cells (um):

```
[50, 70, 98, 137.2, 192.1, 226.4, 226.4 | 50 x4 | 50, 70, 98, 137.2, 192.1,
 188.2 x4 | 50, 70, 80 | 50, 70, 98, 137.2, 192.1, 188.2 x4]
```

Matches the lead (ratio 4.53, 226.4 -> 50 um, last cell 188 um). Cause:
the per-segment loop ramps up from the left edge only and fills the rest
with coarse cells; nothing ramps back down before the next feature. Only
caller: `docs/public/guide/nonuniform-mesh.mdx` (lines 97-113), which then
tells the reader to run `smooth_grading` on the result — Defect 3.

### Defect 3 — public `smooth_grading` shifts every downstream interface

Fixture: `cells = [0.2e-3]*5 + [0.05e-3]*4 + [0.2e-3]*5` (a 0.2 mm slab
at z in [1.0, 1.2] mm on four 50 um cells), `smooth_grading(cells,
max_ratio=1.3)`, then `Simulation(domain=(3e-3, 3e-3, 0.0), dx=0.2e-3,
dz_profile=sm, cpml_layers=4, freq_max=10e9, boundary="cpml")` with the
slab as a `Box((1e-3, 1e-3, 1.0e-3), (2e-3, 2e-3, 1.2e-3))` in fr4.

Measured here: 14 -> 24 cells, column 2.2000 -> 3.2749 mm, slab-top edge
missed by 46.2 um (nearest realized edge 1.1538 mm); preflight
`graded_box_rasterization`: "Box material 'fr4' rasterizes to **2 z cells
(implied 4.0)** over z-span [1mm, 1.2mm)". The same signature appears at
max_ratio 1.4 (column 3.0671 mm, edge missed by 44.9 um).

**Discrepancy with the lead's numbers:** the lead quotes "1 z cells
(implied 3.0)". A sweep over slab position (2-5 coarse cells below), fine
count (3, 4) and ratio (1.3, 1.4) on this tree never produced that pair; the
3-cell variants stay silent (2 nodes >= ceil(0.5 x 3)), the 4-cell variants
report "2 (implied 4.0)". Same defect class and mechanism (transition cells
inserted, downstream interfaces shifted, #325 advisory reproduced); the
lead's exact fixture was not recovered from the numbers given. Recorded, not
resolved.

## 2. The design (the lead's decision, with refinements where the code read contradicts it)

New public builder in `rfx/nonuniform.py`, exported from `rfx/__init__.py`:

```
make_band_profile(edges, cell_sizes, *, max_ratio=1.4, protected=None,
                  boundary_cell=None, min_cells=1) -> np.ndarray  (float64 cells)
```

- `edges`: sorted physical coordinates of every interface that must land
  on a node plane, including 0 and the domain end. `cell_sizes`: target
  cell size per segment (`len(edges) - 1`). `protected`: per-segment bool
  (default False). A protected segment is realized UNIFORM
  (`n = max(min_cells, ceil(span/target))` equal cells, no ramp cells
  inside); a free segment may host ramp cells.
- Invariants the output MUST satisfy for ANY valid input:
  - **I1** every edge is a cumulative node coordinate, |err| <= 1e-12 m;
  - **I2** every adjacent ratio <= max_ratio (+1e-9), INCLUDING seams
    between two adjacent protected segments;
  - **I3** sum == edges[-1] - edges[0] to 1e-12 m;
  - **I4** with `boundary_cell` given, `cells[0] == cells[-1] ==
    boundary_cell` exactly (the x/y CPML contract,
    `rfx/nonuniform.py:355-370`: `dx_profile[0] == dx_profile[-1] == dx`;
    dy ends must equal each other), achieved by ramping inside the end
    segments;
  - **I5** axis-agnostic: the same function serves `dx_profile`,
    `dy_profile` and `dz_profile`.
- Ramps are geometric with ratio <= max_ratio, placed INSIDE the coarser
  free segment on whichever side needs it (ascending and descending), and
  the free segment is renormalized to its exact declared length. Two
  adjacent PROTECTED segments with a seam ratio > cap: the coarser
  protected segment's uniform cell size is reduced (more cells) until the
  seam ratio <= cap — interfaces stay exact, the ratio law holds, the cost
  is cells, not accuracy. Iterate to a fixed point (a reduction can create
  a new violation at its other seam).
- Rewire `make_z_profile` on top of it (same signature; the docstring's
  "fine -> coarse -> fine" becomes true) and rewire `_make_dz_profile`'s
  smoothing step so the auto z mesh satisfies I1-I3 (thirds rule kept
  as-is, before the smoothing step, exactly like today). The #763 preserve
  tests keep passing bit-identically for the demo fixture; if the generic
  fixture's realized cells change, justify by the invariants and re-pin
  with provenance.
- Preflight/docs: replace the "up to 3 fine bands" wording in
  `docs/guides/support_matrix.md` and
  `docs/public/guide/nonuniform-mesh.mdx` with the transition law
  (per-transition reflection set by ratio r, local cells/lambda and band
  width; band count only sums) plus the new unwitnessed-range statement
  (bands narrower than the measured width, in-plane still uncovered). Keep
  every existing number; add none that this lane did not measure.
  CHANGELOG entry under Unreleased.
- Tests: `tests/unit/nonuniform/test_band_profile_builder.py` — the PCB
  fixture, the `make_z_profile` fixture, seeded random-stack fuzz, a dx/dy
  round trip, and the revert-proof test (pins the NEW numbers; the OLD ones,
  8.000 / 4.527, quoted in its docstring).

### Refinements (each one is a place where the code read contradicts the design as written)

- **R1 — I2 and the thirds rule cannot both hold inside a block.**
  `apply_thirds_rule` splits a boundary cell d into [2/3 d, 1/3 d]; the
  ratio between those two sub-cells is exactly **2.0**, and between the
  full cell and the 2/3 sub-cell exactly **1.5**, by construction. #763
  already accepted this (its air-run-only ratio re-pin, commit 6973459).
  So, for the auto z path (`_make_dz_profile`), I2 is evaluated on every
  adjacent pair EXCEPT the pairs internal to a thirds split, and the
  protected-protected seam rule compares the ACTUAL seam cells, i.e. the
  1/3 sub-cells (PCB: 66.7 vs 8.33 um today). The public builder itself
  never sees a thirds split, so I2 holds globally there. Consequence for
  the PCB fixture, derived by hand from the rule: prepreg 4 x 25 um
  (thirds 8.333 um) forces `d_core/3 <= 1.4 x 8.333 um`, i.e.
  d_core <= 35.0 um -> **23 cells of 34.783 um per core**, seam ratio
  11.594/8.333 = **1.391**; reference realization nz = **105** (today 45),
  dz_min **8.333 um unchanged**. Cells, not accuracy — and 2.3x the cells.
  That cost is the #931 question, not a reason to widen the cap.
- **R2 — renormalization must never rescale UP and must not touch a pinned
  boundary cell.** `_smooth_preserving_blocks` states "f <= 1 by
  construction since smoothing only inserts". With ramps placed by the new
  builder that no longer holds: a geometric ramp shorter than its segment
  with no room for a full plateau cell leaves f > 1, and an up-rescale
  raises the seam cell above cap x neighbour (checked by hand on the PCB
  top air run: 8-cell ramp 558.3 um in a 900 um run, remainder 341.7 um;
  a single 341.7 um plateau cell gives ratio 1.997). Rule: plateau cells
  are `rem / ceil(rem / target)` (never above target; the ramp's top cell
  is >= target/cap, so the plateau seam stays <= cap); when the ramps alone
  exceed the segment, uniform DOWN-scale f < 1 of the un-pinned cells (the
  #763 method: internal ratios unchanged, seam ratios only improve);
  `cells[0]`/`cells[-1]` pinned by `boundary_cell` are excluded from any
  rescale.
- **R3 — cell count from a float quotient.** `n = max(min_cells,
  ceil(span/target - 1e-9))` (relative tolerance on the quotient), so
  0.8 mm / 0.2 mm realizes 4 cells for every 0.8 mm core, not 4-5-4 as
  today (Defect 1 measurement).
- **R4 — `boundary_cell` preconditions.** An end segment carrying a pinned
  boundary cell must be free (a protected end segment whose uniform cell is
  not `boundary_cell` is a contradiction -> `ValueError`), and long enough
  to hold the pinned cell plus its ramp (`ValueError` otherwise, message
  naming the segment and the minimum span). The fuzz family only generates
  feasible inputs (end segments free, span >= 4 x boundary_cell, with
  boundary_cell equal to the end segment's target).
- **R5 — thin free segments are refinement sources too.** A free segment
  too thin to host any ramp (say 20 um between two 3 mm protected layers
  at 150 um) realizes one 20 um cell; I2 then requires the neighbours to
  come down: a protected neighbour is refined by the seam rule, a free one
  ramps. The fixed-point iteration therefore runs over ALL seams, not only
  protected-protected ones. The fuzz family (thicknesses 20 um-3 mm,
  targets 0.05-2x the layer) exercises this; the resulting cell counts are
  REPORTED (max nz over the family), not gated.
- **R6 — the in-plane cap is 1.3, not 1.4.** `_PreflightMixin._INPLANE_RATIO_CAP
  == 1.3` and `test_inplane_grading_lock_stays_at_1_3` lock it (WP6R.8: no
  in-plane witness exists). The builder's default 1.4 is fine for z; the
  F5 dx/dy round trip must call it with `max_ratio=1.3`, and the auto z
  path keeps its current 1.3 (the #763 air-run pin `<= 1.301`). Neither
  is a cap move.
- **R7 — absorber runway (open decision for the lead).** `boundary_cell`
  pins ONE cell per end; preflight's `nu_grading_reaches_absorber`
  (`rfx/api/_preflight.py`) wants `cpml_layers` uniform interior cells
  against each absorbing face, ratio deviation <= 1e-6. Under R2 the
  plateau next to a pinned cell is `rem/ceil(rem/target)`, generally NOT
  equal to `boundary_cell` (F5 fixture: 12 mm end segment, ramp 0.65 +
  0.845 mm, remainder 10.505 mm -> ten cells of 0.9505 mm beside the 1.0 mm
  pin, ratio 1.052), so that advisory WILL fire on a pinned profile unless
  the builder also holds a runway of `cpml_layers` cells at exactly
  `boundary_cell` and absorbs the remainder in the ramp (re-solving the
  ramp ratio rho <= cap; feasible here with a 10-cell runway and a 3-cell
  ramp at rho = 1.23). Not required by I1-I5; F5 therefore REPORTS this
  advisory and the realized end-run cells, and gates on the invariants
  only. If the lead wants the runway inside the builder, it is a
  `boundary_cells: int` argument added BEFORE F5 is measured, with F5's
  report line promoted to a gate in the same commit — not after.
- **R8 — `_make_dz_profile` dispatch.** `min_cells_per_feature=4` maps to
  `min_cells=4` on protected segments; free (air) segments keep `min_cells=1`
  (today's `max(1, round(gap/dx))`). The pre-existing "gap or top air
  <= dx/2 is dropped" limit (#763 note) is untouched by this lane.

## 3. Falsifiers (frozen; tolerances never widened after measurement)

Fixtures named here: (a) PCB 5-layer (Defect 1); (b) `make_z_profile`
(Defect 2); (c) #763 demo (h_sub 254 um, dx 190.5 um, column 1.754 mm)
and generic two-layer (layers 0.2-0.5 and 1.1-1.35 mm, column 3.0 mm,
dx 0.3 mm); (d) seeded fuzz, `np.random.default_rng(20260907)`, >= 200
stacks: 1-8 layers, thicknesses 20 um-3 mm, per-segment targets 0.05-2x
the segment, protected mix, boundary_cell on/off (R4 feasibility).

- **F1 (I1, interface snap):** every declared edge within **1e-12 m** of a
  cumulative node, on (a), (b), (c), (d), and on the auto-z output of
  `_make_dz_profile` for (a) and (c). Today: (a) and (b) already hold
  (8.7e-19, 4.3e-19 m); Defect 3's public path misses by 46.2 um.
- **F2 (I2, ratio law):** every adjacent ratio <= max_ratio + **1e-9** on
  (a) via the builder, (b), (d); on the auto-z (a) and (c) outputs, every
  pair outside a thirds split (R1) <= 1.3 + 1e-9, and every
  protected-protected seam (actual seam cells) <= 1.3 + 1e-9. Today: (a)
  8.000, (b) 4.527 — these two numbers are the revert-proof.
- **F3 (I3, column):** |sum - (edges[-1] - edges[0])| <= **1e-12 m** on all
  fixtures.
- **F4 (I4, boundary pin):** with `boundary_cell` set, `cells[0] ==
  cells[-1] == boundary_cell` bit-exactly (`==`, no tolerance) on (d) and
  on the F5 profiles.
- **(c) locks, verbatim from `test_auto_dz_profile_preserve.py`:** demo
  substrate-top edge <= 1e-12 m; post-thirds block
  `[63.5, 63.5, 63.5, 42.333, 21.167] um` bit-identical (`np.array_equal`);
  dz_min 21.167 um; column 1.754 mm to 1e-12 m; air-run ratio <= 1.301;
  generic fixture: four interfaces on nodes, both blocks bit-identical. No
  protected-protected seam exists in either, so the seam rule never fires
  there; the air runs MAY re-realize (geometric ramp from the seam instead
  of smooth_grading insertion + plateau drop). If they do, the old and new
  air cells are printed side by side in the results and the change is
  accepted only because the four locks above still hold — nz is not a lock.
- **F5 (I5, axis round trip):** `dx_profile = dy_profile =
  make_band_profile([0, 12e-3, 15e-3, 27e-3], [1e-3, 0.5e-3, 1e-3],
  max_ratio=1.3, boundary_cell=1e-3)`, `Simulation(freq_max=10e9,
  domain=(sum, sum, 10e-3), dx=1e-3, dx_profile=..., dy_profile=...,
  boundary="cpml", cpml_layers=8)` constructs with **zero** warnings
  containing "adjacent cell ratio", `preflight()` does not emit
  `nu_grading_ratio_beyond_validated_cap`, I4 holds bit-exactly on both
  profiles, and the realized interior extents from the grid's own node
  coordinates (`make_nonuniform_grid(...)` x/y cumulative sums, CPML pad
  excluded) equal 27 mm to **1e-12 m** on both axes. Same profile as
  `dz_profile`: clean as well. `nu_grading_reaches_absorber` and the
  realized end-run cells are REPORTED, not gated (R7).
- **F6 (locked-value audit rule, as in the #763 note):** every committed
  value that moves is listed with old -> new and accepted ONLY if the
  realized mesh now satisfies I1-I3 (and I2 under R1) and the moved value
  follows from that by arithmetic; re-pinned WITH provenance in the commit
  message. A moved value that cannot be justified => STOP and report.
  Batteries (declared):
  ```
  pytest tests/unit/nonuniform/test_band_profile_builder.py tests/unit/nonuniform/test_auto_dz_profile_preserve.py \
         tests/unit/nonuniform/test_smooth_grading_preserve.py tests/unit/grid/test_auto_config.py -q
  pytest tests/unit/nonuniform tests/unit/grid tests/unit/preflight -k "nonuniform or nu or profile or auto_config or mesh_planner" -q -o addopts="" -m "not gpu"
  pytest tests/unit/nonuniform/test_multiband_nu_envelope.py -q -o addopts="" -m "not gpu and not slow_physics"
  pytest tests/contracts -q -o addopts="" -m "not gpu"
  ruff check rfx/ tests/ --select E,F,W --ignore E501,F401,E741,E731,E701,E702,E402
  ```
  Bucket (a) candidates read on this tree: `tests/unit/grid/test_auto_config.py`
  (thirds tests 223-264 use `apply_thirds_rule`/`_make_dz_profile` directly;
  `test_make_dz_profile_applies_thirds_rule` pins free-run ratio <= 1.3),
  mesh_planner / auto_configure consumers (nz, dz_min, dt, memory
  estimates), `tests/_example_fidelity_lib.py` (explicit `dz_profile`
  vectors from `fixtures.py`, builder not called — expected no move),
  preflight NU tests (explicit profiles — expected no move).

### F7 — chain model on the OLD vs NEW PCB profile (no FDTD)

Instrument: `validation/research/multiband_nu/chain_model.py::scattering`
on the profile embedded in W2-count runways — **140 lead cells of the
profile's own first cell, 150 tail cells of its last cell** (the chain
solve needs uniform ends equal to the profile ends; the W2 1 mm runway
cells cannot be used without adding a 0.136 -> 1 mm transition that is not
the builder's). F0 = 10 GHz, transverse dy = dx = 0.2 mm (the fixture's
own dx), b = B_Y = 30 mm (TE10, fc 5 GHz), dt from
`make_nonuniform_grid` on the embedded profile: **2.7471e-14 s** (set by the
8.333 um cell, common to OLD and NEW). "Step" = each adjacent pair with
ratio != 1; its single-step value = `scattering([d_k]*140 + [d_{k+1}]*150)`.

Reference numbers (hand realization of the design per R1-R3; the
implementer's builder must reproduce the per-layer cell counts
23 / 4 / 23 / 4 / 23 (core / prepreg / core / prepreg / core, before the
thirds splits) or write down why):

| profile | nz | total \|R\| | total \|R\|² | Σ\|R_step\| | (Σ\|R_step\|)² | Σ\|R_step\|² | max non-thirds step |
|---|---|---|---|---|---|---|---|
| OLD (this tree) | 45 | 9.756e-6 | 9.518e-11 | 5.065e-4 | 2.566e-7 | 1.481e-8 | 1.556e-5 (136.0 -> 104.6 um, r 1.300) |
| NEW (reference) | 105 | 5.747e-5 | 3.303e-9 | 1.267e-4 | 1.606e-8 | 1.958e-9 | 2.955e-5 (122.2 -> 171.1 um, r 1.400) |

OLD's ratio-8 seams reflect **9.0e-6** each (indices 11 and 30): at 8-67 um
cells a 10 GHz wave sees lambda/450 and worse, so the defect is invisible
to a reflection number at this frequency. It is a ratio-law / dz_min defect,
and **F2 is the gate for it, not F7.** F7 is a consistency witness on the
NEW profile:

- **F7a (frozen):** `total |R(P)|^2 <= 1.5 x (Σ_steps |R_step|)^2` on the
  builder's actual output P. Reference window **2.409e-8**, reference total
  **3.303e-9**. This is the lead's rule with the sum taken in AMPLITUDE
  (coherent), not power. Reason, from the chain model itself: the W2
  r = 1.4 two-step ramp (two steps of 1.1832) reflects **1.9536e-3**, while
  one 1.1832 step reflects **8.307e-4** — steps closer than a wavelength add
  in amplitude (2 x 8.307e-4 = 1.661e-3 is the right order; the incoherent
  sqrt(2) x 8.307e-4 = 1.175e-3 is not). On the reference NEW profile the
  literal power rule, 1.5 x Σ|R_step|² = **2.937e-9**, is BELOW the total
  3.303e-9 and would fire on a profile that satisfies every invariant.
  Recorded here so the window is not read as a loosening: the amplitude
  bound is the triangle inequality with 50 % headroom.
- **F7b (frozen):** max over non-thirds steps of |R_step| <=
  |R_single(r = 1.4, d = max(P))| x (1 + 1e-9), where the bound is the
  chain-model single step at the cap ending on P's coarsest cell.
  Reference: **2.9545e-5 vs 2.9545e-5** (equality: the reference's largest
  step is exactly a cap step onto its coarsest cell, 171.1 um). Thirds
  pairs reported separately (reference max **1.385e-6**).
- Reported, not gated: OLD vs NEW totals as above; the OLD 9.8e-6 must not
  be quoted as "OLD reflects less" without the lambda/450 caveat.

### F8 — FDTD witness, narrow fine band (CPU, one attempt)

Fixture, all lengths from `fixtures.py` constants: coarse cell
`DC = DZ_FINE x 1.4^2 = 1.96 mm` (15.30 cells per free-space wavelength at
10 GHz; fine band 30.0). Profile A(n_b), incident from the coarse side:

```
[1.96 mm] x 140  |  1.4 mm  |  [1.0 mm] x n_b  |  1.4 mm  |  [1.96 mm] x 150
```

Both ramps sit exactly at the cap (1.96 -> 1.4 -> 1.0 and back). Built
with the builder under test: `make_band_profile(edges=[0, 275.8e-3,
275.8e-3 + n_b*1e-3, 275.8e-3 + n_b*1e-3 + 295.4e-3], cell_sizes=[1.96e-3,
1e-3, 1.96e-3], protected=[False, True, False], max_ratio=1.4)`; the test
asserts the builder's output equals the vector above to 1e-12 m per cell
(if it does not, the chain model is re-run on the builder's actual output,
that prediction is the gate, and the deviation is reported). B run
(2-run differencing reference): `[1.96 mm] x 400 + [1.0 mm] x 4` — the four
trailing fine cells pin dt to A's (verified: dt A = dt B =
**2.402765e-12 s**, both 0.99 CFL with dxy 1.5 mm) and sit beyond every
gate (their return reaches the probe at 5.055 ns). Source plane K_SRC = 85,
probe K_PRB = 100 (coarse cells, 166.6 / 196.0 mm), TE10 soft Ex source,
Gaussian-modulated sine F0 = 10 GHz, sigma_t = 64 ps, t0 = 5 sigma_t; PEC
box a = 4.5 mm, b = 30 mm, `cpml_layers = 0` (`harness.build_pec_fixture`);
**n_steps = 1200** (2.883 ns). Gates from geometry, same construction as
`w2_arm`: reflection arrival t_r = 1.047 ns (+4 sigma = 1.303 ns), last
band-internal return 1.099 ns at n_b = 4 (1.192 ns at 16), source-wall
echo t_s = 2.347 ns, far-wall t_f = 3.393 ns, gate_end = 2.091 ns
(870 steps); incident gate closes at 0.947 ns. `R_meas = |DFT_F0(A - B,
[0, gate_end])| / |DFT_F0(B, [0, t_inc_end])|` — an AMPLITUDE ratio, as in
W2.

Chain-model predictions (exact discrete solve on the vector above, dt
2.402765e-12 s, dy 1.5 mm, b 30 mm), frozen:

| n_b (cells) | band (mm) | \|R\|_model | \|R\|²_model | dB | window on \|R\|_meas |
|---|---|---|---|---|---|
| 2 | 2 | 7.4916e-3 | 5.612e-5 | -42.5 | [5.9633e-3, 9.0199e-3] |
| **4 (gate)** | 4 | **1.0141e-2** | **1.0283e-4** | -39.9 | **[8.0826e-3, 1.2199e-2]** |
| 8 | 8 | 1.1296e-2 | 1.276e-4 | -38.9 | [9.0066e-3, 1.3585e-2] |
| 16 | 16 | 1.2101e-3 | 1.464e-6 | -58.3 | [9.3810e-4, 1.4822e-3] |

Window: `|R_meas - R_model| <= 0.20 x R_model + 3e-5` (FS2_FLOOR = 3e-5 is
the W2 amplitude floor — W2's `R_meas` is an amplitude ratio — so the
window is applied in amplitude; in power that is about +/-(40 % + 6e-5 x R)).
|R|² is reported alongside. **The gate is the n_b = 4 row;** the four rows
together are the validity-domain deliverable. The law they trace, read off
the chain model before any FDTD: a single 1.96 -> 1.4 -> 1.0 ramp reflects
**5.7907e-3** (-44.7 dB, either direction); a band of width L between two
such ramps reflects about `2 x 5.79e-3 x |sin(k_g L_eff)|` — a Fabry-Perot
sum of the two opposite-sign ramp reflections. 8 cells is about lambda_g/4
(lambda_g = 34.64 mm at 10 GHz, b = 30 mm) and sits near the 2 x 5.79e-3 =
1.16e-2 maximum; 16 cells is about lambda_g/2 and lands in the null. So
"narrow" is not "worse": the narrow-band reflection is bounded by twice the
single-transition value and oscillates with band width. Rows n_b = 32
(1.5111e-3) and 64 (6.5575e-3) may be run as extra law points; reported,
not gated. All of this is one resolution (fine 30, coarse 15.3
cells/lambda) — quote the law, not the dB.

Run commands (declared) and output:

```
cd /Users/byungkwankim/Documents/rfx-nu-band && PYTHONPATH=/Users/byungkwankim/Documents/rfx-nu-band \
  /Users/byungkwankim/Documents/rfx/.venv/bin/python -m validation.research.multiband_nu.w6_band_builder \
  --widths 2,4,8,16 --out validation/research/multiband_nu/results/w6_band_builder.json
```

`w6_band_builder.py` (committed BEFORE it is run) records `rfx.__file__`,
the builder's profile per width, the chain-model prediction, the gates in
ns, `R_meas`, `R_meas**2`, and the F8 verdict per row. One attempt, no
re-rolls; a fired gate is reported as fired.

## 4. Impact-sweep rule

This changes every auto-configured NU z mesh whose stack has a
protected-protected seam or an air run (cell count, dz values, dt via
dz_min, memory estimates), `make_z_profile`'s output for every input with a
descending edge, and adds two public names. For every locked test value
that moves, the NEW value is accepted only if the realized mesh now
satisfies I1-I3 (I2 under R1) and the moved value follows from that by
arithmetic; re-pin WITH provenance in the commit message. Any moved value
that cannot be justified physically => STOP and report. Values expected to
move: nz / dz_min-derived estimates on auto_configure fixtures with
adjacent dielectric layers (none of the (c) locks). Values expected NOT to
move: the (c) demo block and dz_min, every explicit-profile fixture in
`tests/_example_fidelity_lib.py`, `fixtures.py`, and the preflight NU
tests, the 1.4 z cap, the 1.3 in-plane cap, `apply_thirds_rule` and its
three tests.

## 5. Hand-off to #931 (geometry -> lattice ownership)

The thirds rule is an OWNERSHIP assumption, not a mesh-builder detail:
`_make_dz_profile` applies it at every feature boundary index — dielectric
| air AND dielectric | dielectric (core | prepreg, measured above) — and
never at a conductor, although its docstring reasons about a conductor
side. Its measured consequence on the PCB fixture: a 0.1 mm prepreg on
four 25 um cells becomes 8.333 / 16.667 / 25 / 25 / 16.667 / 8.333 um, so
**dz_min = 8.333 um** sets dt for the whole column, and under I2 the
neighbouring cores are forced to 23 cells each (nz 45 -> 105). This lane
keeps `apply_thirds_rule` and its locked tests unchanged and pays the cell
cost; whether a dielectric | dielectric seam should be split at all, and
whether dz_min should ever be a thirds sub-cell of the thinnest layer, is
#931's question. Rule from the 2026-08-23 ledger, restated so the answer
does not drift: thin copper sheets register on a NODE PLANE, never as a
17 um cell — a conductor boundary needs the node, not a split cell.

---

## Results (appended after measurement; no window above changed)

Measured 2026-09-07 (KST) on this tree, `rfx.__file__` =
`/Users/byungkwankim/Documents/rfx-nu-band/rfx/__init__.py`. Commits:
`a61d0a89` (builder, both rewires, tests, W6 instrument), `d79ad615`
(engine exact-endpoint fix, before any FDTD ran), `d51879a7` (W6
provenance-loader fix, before any FDTD ran), then this commit (results,
docs, CHANGELOG). Raw W6 output:
`validation/research/multiband_nu/results/w6_band_builder.json`.

### Where the implementation deviates from section 2, and why

- **R2 replaced by an exact-fit plateau solve.** "Plateau cells
  `rem / ceil(rem / target)`, else uniform down-scale" has a corner: the
  plateau cell can be arbitrarily small next to a ramp top (rem -> 0), and
  a down-scale shrinks the seam-adjacent ramp cell, which breaks the seam
  against a neighbour that did not scale. Implemented instead: geometric
  ramps (per-step ratio spread evenly, <= cap) from each anchor to a
  common plateau value u <= target, the ramp's LAST cell being u itself,
  and u bisected so that ramps + an integer number of plateau cells sum to
  the span exactly (`_band_realize_free`). The ramp/plateau junction is
  then the ramp's own ratio for any plateau count; nothing is rescaled;
  a pinned cell never moves. R2's "never above target" and "pinned cells
  excluded" both hold.
- **R1 reference count: 23 cells per core at cap 1.4, 25 at cap 1.3.** R1
  derived 23 with 1.4, but F2 (frozen) and R6 pin the auto-z path at 1.3
  (the #763 air-run lock, `test_make_dz_profile_applies_thirds_rule`).
  At 1.3: d_core/3 <= 1.3 x 8.333 um -> d_core <= 32.5 um ->
  ceil(0.8 / 0.0325) = **25 cells of 32.000 um**, seam 10.667 / 8.333 =
  **1.280**, nz **115** (not 105). The public builder at 1.4 on the same
  stack (no thirds) realizes exactly the reference 23 / 4 / 23 / 4 / 23
  (34.783 um cores, seam 1.3913, nz 89) — the reference table's per-layer
  counts are the cap-1.4 numbers.
- **R4 fuzz feasibility rule.** "span >= 4 x boundary_cell" is not
  sufficient: a pinned end segment whose neighbour is much finer must
  DESCEND from the pin at ratio cap, which needs up to bc / (cap - 1) of
  span (5 bc at cap 1.2), and 4 bc only covers that for cap >= 1.34. The
  family uses `bc <= span x min(1/4, (cap - 1) / (1.1 cap))` with caps
  drawn from {1.2, 1.3, 1.4}. Under the rule as written the first fuzz
  pass raised `ValueError` on 610 of 2000 stacks (all pinned, and including
  the engine defect (1) below); under the cap-aware rule and the fixed
  engine, 0 of 3000.
- **Three engine defects found by the fuzz / the F8 declaration and fixed
  BEFORE any FDTD ran:** (1) at a piece endpoint the ramp switches between
  "empty" and "one cell equal to u" (same cells, different accounting), so
  G jumps by an integer there and evaluating G at the endpoint
  misclassified the piece — a single segment with both ends pinned and
  span 4 bc raised `ValueError` (fuzz trial 10); fixed by evaluating just
  inside each piece and snapping a root that lands on an endpoint.
  (2) that inset moves G by n x 1e-10, past a fixed 1e-8 tolerance when
  n > 100: the F8 fixture's 139-cell plateau was rounded up to 140 (142
  cells at 1.946 mm instead of 141) — caught by this note's own "builder
  output equals the declared vector" assertion, fixed by an exact-fit check
  at the target on its own accounting plus a tolerance proportional to G
  (commit `d79ad615`). (3) two adjacent free runs flip each other's seam
  cells by one ulp (1e-18 m) in a two-cycle, so exact-equality convergence
  never settled (11 of 3000 stacks); convergence is judged at 1e-11
  relative. Every pinned fixture number was unchanged by (2) and (3).
- **W6 instrument:** the d990e18c source is loaded with `git show` into a
  module that must be registered in `sys.modules` (its `@dataclass` looks
  itself up); the first launch crashed there before F7's first number.
- **Exports:** `rfx.make_band_profile` only; `make_z_profile` stays on
  `rfx.nonuniform` (the guide imports it from there). `rfx.__all__`
  211 -> 212 under the 213 curation ceiling
  (`tests/contracts/test_forward_docstring_contract.py`). Exporting both
  would have tripped that gate (213), and the design requires one name.
- `_smooth_preserving_blocks` (#763) is removed; `_make_dz_profile` calls
  the engine through `_band_smooth_post_thirds`. Its locks still hold
  (below).

### F1-F4 and (c) — profile level, no FDTD

| Fixture | I1 max interface error | I2 | I3 column error | nz | notes |
|---|---|---|---|---|---|
| (a) PCB via builder, cap 1.4, min_cells 4 | 8.7e-19 m | max 1.397368 (air ramp), 0 pairs > 1.4; core\|prepreg seam 1.3913 | 0 | 89 | 23/4/23/4/23; OLD (auto-z) 8.000 |
| (a) PCB via `_make_dz_profile`, cap 1.3 | 6.1e-18 m | max non-thirds 1.288500, 0 pairs > 1.3; every block seam 1.280000 on the 1/3 sub-cells; thirds pairs 2.0 / 1.5 by construction (20 pairs) | 0 | 115 (OLD 45) | dz_min 8.333 um unchanged; cores 25 x 32 um + splits; bottom air 10 cells 117.2 -> 13.56 um, top air 12 cells 13.74 -> 173.4 um |
| (b) `make_z_profile` Defect-2 fixture | 1.7e-18 m | max 1.352395, 0 pairs > 1.4 (OLD 4.527) | 0 | 42 (OLD 32) | first and last cell 50 um exactly; max cell 167.3 um |
| (c) #763 demo | 0 | air run 1.2718 (lock <= 1.301) | 0 | 18 | block `[63.5, 63.5, 63.5, 42.333, 21.167]` um bit-identical, dz_min 21.167 um; air run re-realized: 26.92, 34.24, 43.54, 55.37, 70.42, 89.56, 113.9, 144.9, 184.2 x5 um |
| (c) generic two-layer | 2.2e-19 m | max outside thirds <= 1.3 | 4.3e-19 m | 39 | both blocks bit-identical, four interfaces on nodes |
| (d) fuzz, rng 20260907, 3000 stacks (400 in the test) | all <= 1e-12 (0 failures) | 0 pairs over cap, worst excess 0 | all <= 1e-12 | max 1772, median 48 | 1481 pinned stacks, I4 bit-exact on all; protected blocks uniform with >= min_cells |

### F5 — axis round trip

`make_band_profile([0, 12, 15, 27] mm, [1, 0.5, 1] mm, max_ratio=1.3,
boundary_cell=1 mm)`: 32 cells, max ratio 1.243140, both ends 1e-3 exactly.
`Simulation(freq_max=10e9, domain=(27, 27, 10) mm, dx=1e-3, dx_profile=p,
dy_profile=p, boundary="cpml", cpml_layers=8)`: **zero** "adjacent cell
ratio" warnings; `preflight()` emits no
`nu_grading_ratio_beyond_validated_cap`; interior extents from
`make_nonuniform_grid` node sums 27 mm to <= 1e-12 m on x and y, interior
end cells 1e-3 exactly. Same profile as `dz_profile`: clean as well.
**Reported, not gated (R7):** `nu_grading_reaches_absorber` fires on all
four in-plane faces — end run `[1.0, 0.9606 x 10, 0.7727, 0.6216, 0.5]` mm,
ratio deviation 0.0394 (lo faces) / 0.041 (hi faces) inside the 8-cell
runway. The pinned cell is one cell; a `boundary_cells` runway argument
remains the lead's decision (R7), not added here.

### F6 — batteries and moved values

Batteries (final tree): `tests/unit/nonuniform tests/unit/grid/test_auto_config.py`
(-m "not gpu and not slow") 266 passed before the engine fix, re-run after
it — see the hand-off report; the four-file lock battery 67 passed;
`tests -k "nonuniform or auto_config or mesh_planner or dz_profile or
grading"` 381 passed / 2 skipped; `tests/unit/preflight -k "nu or
nonuniform or graded or profile"` 11 passed; `tests/contracts` 1008 passed
/ 1 failed before the export was trimmed to one name (the `__all__`
ceiling), passing after; ruff clean. **Locked values moved: none.** The
(c) locks, `apply_thirds_rule` and its three tests, the 1.4 z cap, the 1.3
in-plane cap, every explicit-profile fixture (`_example_fidelity_lib.py`,
`fixtures.py`, preflight NU tests) are untouched. Realized cells that
changed without a lock: the demo air run (above), `make_z_profile` outputs
with a descending edge, and auto-z stacks with adjacent dielectric blocks
(cell count, dt unchanged where dz_min is a thirds sub-cell).

### F7 — chain model, OLD vs NEW auto-z PCB profile (no FDTD)

dt = 2.7471385804e-14 s for both (the 8.333 um cell); F0 10 GHz, dy 0.2 mm,
b 30 mm, 140 / 150 runway cells of the profile's own end cells.

| profile | nz | total \|R\| | total \|R\|² | Σ\|R_step\| | (Σ\|R_step\|)² | Σ\|R_step\|² | max non-thirds step | max thirds pair |
|---|---|---|---|---|---|---|---|---|
| OLD (d990e18c, from `git show`) | 45 | 9.7561e-6 | 9.518e-11 | 5.0653e-4 | 2.566e-7 | 1.481e-8 | 1.5558e-5 (136.0 -> 104.6 um, r 1.300) | 4.5788e-5 (133.3 -> 200 um, r 1.5) |
| NEW (this tree, cap 1.3) | 115 | 5.6060e-5 | 3.1427e-9 | 1.0596e-4 | 1.1228e-8 | 1.1542e-9 | 2.4626e-5 (134.5 -> 173.4 um, r 1.288) | 1.1719e-6 |

The OLD row reproduces the reference table to the digits quoted. The NEW
row is the cap-1.3 realization (nz 115), not the cap-1.4 reference
(nz 105), per the R1 deviation above.

- **F7a: not fired.** 3.1427e-9 <= 1.5 x (1.0596e-4)² = **1.6841e-8**.
  (The literal power-sum rule would again have fired on a compliant
  profile: 1.5 x 1.1542e-9 = 1.7314e-9 < 3.1427e-9 — recorded as the
  reference predicted.)
- **F7b: not fired.** 2.4626e-5 <= |R_single(r 1.4, d 173.36 um)| =
  **3.0330e-5**. The largest NEW step is an r = 1.288 step onto the coarsest
  cell; the largest OLD step outside a thirds pair was 1.5558e-5, but its
  thirds pairs reached 4.5788e-5 (r = 1.5, 133 -> 200 um).
- Reported: OLD total 9.8e-6 vs NEW 5.6e-5 — OLD's ratio-8 seams sit at
  8-67 um cells (lambda/450 and finer at 10 GHz) and reflect 9.0e-6 each,
  so the OLD number is smaller for the reason section 3 gave, not because
  the OLD mesh is better; NEW's total is set by its air ramps
  (117 -> 173 um cells). Transmission |T| 1.0000104 (OLD) / 1.0000336 (NEW),
  raw amplitude, not flux-normalized.

### F8 — FDTD witness, narrow fine band (CPU, one attempt)

`w6_band_builder.py` run once (commit `d51879a7` + fixed engine
`d79ad615`), widths 2, 4, 8, 16 plus the extra law points 32, 64;
wallclock 3.7 s. dt(A) = dt(B) = 2.402764937e-12 s; n_steps 1200; gates
t_r 1.047 ns, t_s 2.347 ns, gate_end 2.091 ns (870 steps), incident gate
0.947 ns; t_f 3.378 / 3.393 / 3.424 / 3.486 / 3.609 / 3.856 ns and last
band-internal return 1.084 / 1.099 / 1.130 / 1.192 / 1.315 / 1.562 ns for
the six widths — all inside the gate. **The builder produced the declared
vector on every width** (max per-cell deviation 5.6e-17 m), so the
chain-model predictions are the frozen ones.

| n_b | nz | \|R\|_meas | \|R\|²_meas | dB | \|R\|_model | deviation | window half-width | inside |
|---|---|---|---|---|---|---|---|---|
| 2 | 294 | 7.4364e-3 | 5.530e-5 | -42.6 | 7.4916e-3 | 5.51e-5 | 1.528e-3 | yes |
| **4 (gate)** | 296 | **1.0063e-2** | 1.013e-4 | -39.9 | 1.0141e-2 | 7.80e-5 | 2.058e-3 | **yes** |
| 8 | 300 | 1.1164e-2 | 1.246e-4 | -39.0 | 1.1296e-2 | 1.32e-4 | 2.289e-3 | yes |
| 16 | 308 | 1.2559e-3 | 1.577e-6 | -58.0 | 1.2101e-3 | 4.58e-5 | 2.720e-4 | yes |
| 32 (extra) | 324 | 1.4248e-3 | 2.030e-6 | -56.9 | 1.5111e-3 | 8.63e-5 | 3.322e-4 | yes |
| 64 (extra) | 356 | 6.4596e-3 | 4.173e-5 | -43.8 | 6.5575e-3 | 9.79e-5 | 1.342e-3 | yes |

Every deviation is within 0.2 % to 1.2 % of the model in the peak rows and
3.8 % / 5.7 % in the two null rows (16, 32 cells), all far inside the
20 % + 3e-5 windows. The law read off the chain model before the run —
narrow-band reflection bounded by about twice the single-ramp 5.79e-3 and
oscillating with band width (near-maximum at 8 cells, null at 16) — is what
the FDTD returns. Validity domain now witnessed: fine bands of 2 to 64 cells
at fine 30 / coarse 15.3 cells per free-space wavelength, ratio 1.4, PEC
closed, one frequency. Not witnessed: bands under 2 cells, other
resolutions, in-plane grading, an absorber present.

### Impact sweep and #931

No locked value moved (F6). The #931 question stands as written in
section 5, with the measured cost now on record: the prepreg's thirds
sub-cell (8.333 um) sets dt for the whole PCB column and, under the ratio
law, forces 25-cell cores — nz 45 -> 115 — for a dielectric | dielectric
seam that has no conductor to justify the split. The cost is larger on
ordinary stacks than the PCB fixture suggests (reviewer notes below): a
42.5 um dielectric between a 2.5 mm and a 2.4 mm one at dx 0.255 mm goes
nz 40 -> 391 (x9.8) at an unchanged dz_min of 3.543 um, because the thin
layer's thirds sub-cell pulls both thick neighbours down to <= 1.3 x 3 x
3.543 um cells.

### Reviewer notes (second pass, 2026-09-07; no window above changed)

Fifteen findings on `e65542c8`. Each was reproduced on this tree before
any code moved (command and OLD number per item); dispositions and NEW
numbers below. Commit `af54b501` (code, tests, CLASSIFICATION); the docs
commit that carries this section follows it. **No frozen window in
section 3 was found wrong** — every F1-F8 result above stands as
measured, and none of the fixes moved a pinned number (four-file lock
battery 67 passed; `test_band_profile_builder.py` + the example-fidelity
and forward-docstring contracts 219 passed). `stopped = false`.

**Blocking**

- *Pin invisible when an end segment's span equals `boundary_cell`.*
  Reproduced: `make_band_profile([0, 1, 5] mm, [1, 0.1] mm, boundary_cell
  = 1 mm)` -> `[1.0, 0.0993, ...]`, ratio 10.066 at index 0, 16 cells, no
  error; both-ends form `[0, 1, 5, 6] mm` ratio 10.0 at both ends. Cause
  as the reviewer read it: `_hi_cell(0)` / `_lo_cell(n-1)` returned the
  pin from the outside only, so the neighbour got no anchor and the seam
  audit skipped the pin seam. Fixed: an empty end segment's seam cell is
  the pin on both sides; the neighbour ramps from it as kind `"pin"`.
  NEW on the two repros: the first RAISES `cannot hold its ramps` — and
  that is the right answer: `boundary_cell` pins BOTH ends, so segment 1
  (4 mm) holds the 1 mm hi pin and must descend 1 -> 0.1 mm twice
  (2 x 2.25 mm at 1.4) in 3 mm of free span, which no compliant profile
  can do; `[0, 1, 9] mm` (8 mm free) realizes 40 cells, max ratio
  1.3905, pins exact. The protected-middle form raises `no room for a
  ramp` (ratio 10.000 > 1.4), as the docstring promised.
- *`w6_band_builder.py` had no CLASSIFICATION entry.* Reproduced (the
  contract test failed on the full battery). Added as `no_simulation`
  (zero `Simulation()` calls, the W2 functional path); the first-pass
  "tests/contracts 1008 passed / 1 failed" was measured before the W6
  script existed on the tree, so that report line was wrong for the
  final tree — recorded here.

**Major**

- *Unbounded cell count on a protected sliver.* Reproduced at bounded
  scale: 1e-9 m sliver -> 1,428,573 cells in 0.23 s. NEW: two edges
  closer than 1e-12 m (the I1 node tolerance) raise `ValueError`; a
  profile past `_MAX_PROFILE_CELLS = 1,000,000` raises naming the sliver
  (the 1e-9 case raises at 1,428,573). A 10 um layer between 1 mm blocks
  still realizes (145 cells). A free 2e-12 m sliver realizes a 2e-12 m
  cell with I1-I4 intact — the caller's input, documented.
- *`max_ratio` close to 1.* Reproduced: 1.0001 on a 40-cell profile
  8.61 s, 1.0002 2.19 s; NaN passed `cap <= 1.0`, inf was accepted. NEW:
  ramp sums are closed-form geometric series (O(1) per evaluation
  whatever the step count) with the step count held fixed per piece;
  1.0001 -> 0.03 s, a 1 um pin descending into 10 mm of 100 um cells
  (8110 cells) 0.42 s, `make_z_profile` at 1.0001 0.21 s. Floor
  `_RATIO_FLOOR = 1.0001` (a 2x transition already needs 6932 cells
  there) and finiteness are enforced; 1.00009 / 1.0 / NaN / inf raise.
- *Exact-fit knife edge on long columns.* Reproduced: F8-shaped stack,
  n = 17000 / 20000 (33 / 39 m columns) raised `cannot hold its ramps`;
  n = 100000 at cap 1.3 returned 200018 cells (plateau 1.96 mm) on this
  tree — the reviewer's 338,025 was not reproduced in that form, the
  n >= 17000 false raise was. Cause confirmed: the band cell carries
  5.5e-13 relative noise and the 1e-12 step slack flipped 2 -> 3 steps.
  NEW: slack `_STEP_TOL = 1e-10` (excess ratio at most cap ln(cap) 1e-10
  / m = 4.7e-11 at 1.4, inside the 1e-9 I2 tolerance); pieces are closed
  intervals with per-piece step counts, so a root at a threshold is
  found without inset or snap. n = 10000 / 17000 / 20000 / 24000 return
  2n + 16 cells matching the declared vector to 1e-11 relative; the
  19.6 um variant at n = 16000 likewise.
- *`_make_dz_profile` assembly (pre-existing on main).* Reproduced on
  the OLD source (`git show d990e18c`): gap-50-um stack column 3.9500 mm
  (nz 30), overlap stack 4.2000 mm (nz 32), dx 10 mm on a 4 mm domain
  0.8000 mm (nz 4). NEW: 4.0000 mm on all three (nz 32 / 54 / 19),
  interfaces on nodes to <= 4.3e-19 m, every non-thirds ratio <= 1.3.
  This is a deviation from section 2 ("thirds rule kept as-is, before
  the smoothing step, exactly like today"): the thirds rule IS
  unchanged, but the block/air assembly that feeds it now partitions the
  column at every distinct boundary (features are bounding boxes of any
  non-PEC shape and overlap legitimately — a sphere inside a substrate),
  realizes every gap as an air run, and raises for a feature outside
  `[0, domain_z]`. Reason: the rewired docstring promises an exact column
  and exact interfaces, and the engine handles thin free runs (R5), so
  the dx/2 drop had no remaining purpose. For disjoint stacks with gaps
  wider than dx/2 the pre-thirds cells and boundary indices are
  identical to before (the (c) locks and the PCB fixture are
  unchanged). Consequence worth knowing: a genuine narrow gap is now
  meshed — `[(0.5, 1.5), (1.51, 2.51)] mm` at dx 0.2 mm (a 10 um air gap
  between two dielectrics) OLD nz 30, dz_min 45.333 um, column short by
  10 um -> NEW nz 81, dz_min 10.000 um (the gap cell), column exact.
- *Guide claim "keep the end segments wide".* Reproduced: plateau
  beside the 1 mm pin 0.9606 / 0.9774 / 0.9853 / 0.9913 / 0.9957 mm for
  end segments of 12 / 20 / 30 / 50 / 100 mm (ratio 1.3), deviation
  3.9e-2 down to 4.3e-3, never the advisory's 1e-6. The sentence is
  replaced by these numbers and "close the face"; R7 stands.
- *dt cost of the protected-seam refinement.* Measured here (rng 1,
  3000 stacks, 2-6 segments, spans 20 um-3 mm, targets 0.05-2x the span,
  70 % protected, caps {1.2, 1.3, 1.4}; "declared minimum" = the minimum
  over each segment's initial realization): **410 of 3000** stacks
  realize a cell below their declared minimum, worst factor **0.308**
  (two single-cell protected blocks 369.8 / 455.7 um at cap 1.2). The
  reviewer's 1103 / 0.3008 came from a different draw; the direction
  and the worst case agree. Hand case `[0, 0.1, 0.1355] mm, [25, 100]
  um, both protected, 1.4`: 5 x 20 um + 2 x 17.75 um, minimum 0.71x the
  declared 25 um. Docstring, CHANGELOG and this note now say "cells and,
  when the refined block held the coarsest declared cell, a smaller
  minimum cell (dt)". Auto-z path, 400 PCB-like contiguous stacks (rng
  3) OLD vs NEW dz_min: lower in 17, higher in 241, equal in 142; every
  one of the 17 is the gap fix above, not seam refinement — on those
  stacks main had dropped the sub-dx/2 air runs, so its column was short
  and its block never received a thirds split (factors 0.333 x16, 0.302
  x1). nz ratio NEW/OLD median 1.70, max 26.27 on that family.

**Minor**

- *`make_z_profile` features outside the domain / near-duplicates.*
  Reproduced (sum 5 mm for domain 4 mm; a 5.4e-20 m cell). NEW: outside
  -> `ValueError`; planes closer than 1e-12 m merge (the 5.4e-20 case
  realizes 34 cells, minimum 50 um).
- *I1 not scale-invariant.* Reproduced: `[0, 10 m]` at 100 um -> 100,000
  cells, running-sum end node off by 9.97e-12 m (np.sum: 0). The
  docstring now states the running-sum figure and its range (about 1 m
  / 1e4 cells for 1e-12 m); the long-column test asserts I1 relative to
  the column (1e-12 x column) and each cell to 1e-11 relative.
- *`make_z_profile` raised on `grading <= 1`.* Reproduced (main:
  30 uniform 0.1 mm cells). Restored: `grading <= 1` means no grading
  (uniform `dx_fine`; the seams between the uniform segments are held at
  the default 1.4). Declared in the CHANGELOG.
- *F6 side-by-side not printed for the (c) fixtures.* Printed here (um,
  OLD = d990e18c `_make_dz_profile`, NEW = this tree):
  - generic two-layer (0.2-0.5 / 1.1-1.35 mm, column 3.0 mm, dx 0.3 mm):
    OLD nz 49, dz_min 10.711 -> NEW nz 39, dz_min 20.833.
    OLD: 53.06, 40.82, 31.40, 26.53, 20.41, 15.70, 12.08, [25.00, 50.00,
    75.00, 75.00, 50.00, 25.00], 12.93, 16.80, 21.84, 28.40, 36.92, 39.77,
    51.70, 67.21, 79.54, 61.19, 47.07, 39.77, 30.59, 23.53, 18.10, 13.92,
    10.71, [20.83, 41.67, 62.50, 62.50, 41.67, 20.83], 24.06, 31.28,
    40.66, 52.86, 68.72, 81.43, 105.86, 137.62, 162.87, 211.73, 244.30 x3.
    NEW: 70.86, 54.61, 42.09, 32.44, [25.00, 50.00, 75.00, 75.00, 50.00,
    25.00], 32.19, 41.45, 53.37, 68.71, 88.47, 88.47, 69.52, 54.63, 42.93,
    33.74, 26.51, [20.83, 41.67, 62.50, 62.50, 41.67, 20.83], 26.93,
    34.81, 45.00, 58.18, 75.21, 97.22, 125.68, 162.46, 210.02, 271.49 x3.
    Both blocks (bracketed) bit-identical; the OLD 10.71 um and 12.08 um
    cells were smooth_grading-plus-rescale artefacts of the air run.
  - #763 demo: OLD nz 19: [63.50, 63.50, 63.50, 42.33, 21.17], 25.10,
    32.63, 42.41, 55.14, 57.00, 74.11, 96.34, 114.01, 148.21, 171.01 x5
    -> NEW nz 18: [same block], 26.92, 34.24, 43.54, 55.37, 70.42, 89.56,
    113.91, 144.86, 184.23 x5. dz_min 21.167 um both.
  - `[(0.5, 1.5), (1.51, 2.51)] mm`, dx 0.2 mm: see the assembly item
    above (OLD 30 / 45.333 um with the 10 um gap deleted -> NEW 81 /
    10.000 um with the gap meshed).
- *R1 / #931 cost on ordinary stacks.* Reproduced: `[(0.41851, 2.96435),
  (2.96435, 3.00687), (3.00687, 5.40470)] mm`, domain 5.62563 mm, dx
  0.254968 mm: OLD nz 40 -> NEW nz 391 (x9.8), dz_min 3.543 um both (the
  42.5 um layer's thirds sub-cell). Added to the #931 hand-off: the
  thirds split at a dielectric | dielectric seam sets this cost, and
  under the ratio law it propagates into both thick neighbours.
- *W6 row: L_eff dropped, 5.79e-3 labelled as measured.* Measured on the
  chain model (`scattering` on `a_profile_expected(n_b)`, n_b = 0..80,
  dt 2.402765e-12 s, dy 1.5 mm, b 30 mm): single ramp 5.7907e-3;
  discrete k_g(1.0 mm) = 0.18163 /mm (lambda_g 34.593 mm); with L alone
  the rows read 4.12e-3 / 7.69e-3 / 1.15e-2 / 2.70e-3 / 5.26e-3 /
  9.37e-3 against 7.49e-3 / 1.014e-2 / 1.130e-2 / 1.21e-3 / 1.51e-3 /
  6.56e-3 (off up to 3.5x in the null rows); with L_eff = L + **1.87 mm**
  every row is within 0.5 %. Chain nulls at 15 / 33 / 50 / 67 cells,
  peaks at 7 / 24 / 41 / 59 / 76, maximum 1.1581e-2 = 2 x 5.7907e-3 x
  0.9999; n_b = 1 (unwitnessed) 5.77e-3, n_b = 0 3.87e-3. The matrix
  row now carries L_eff and labels the single-ramp value as modelled.
  The reviewer's own FDTD check of the single ramp (5.768e-3, -0.4 %)
  is theirs, not this lane's, and is not quoted in the row.

Constants introduced (`rfx/nonuniform.py`): `_RATIO_FLOOR = 1.0001`,
`_MAX_PROFILE_CELLS = 1_000_000`, `_STEP_TOL = 1e-10`, `_EDGE_TOL =
1e-12`. None is a measurement gate; each is an input-validity bound
stated in the `make_band_profile` docstring.

### Re-execution on 5bf9d16b (measurer pass, 2026-09-07; no window above changed)

Why a second execution: the F7/F8 numbers above were produced on the
engine of `d79ad615`; the review pass `af54b501` then rewrote the ramp
sums (closed form), the pin visibility and the auto-z assembly, and the
first W6 JSON recorded `rfx.__file__` and wallclock but no git sha, so it
could not say which engine made it. This is a re-execution of the
declared command on the final engine, not a re-roll: nothing had fired,
and both attempts are kept (first attempt: the tables above and the
`e65542c8` JSON in git history; this one: the JSON now on the tree).

Provenance: `5bf9d16b` (the script gained `git_sha`, `git_dirty`,
`started_utc`; committed before the run), argv `--widths 2,4,8,16,32,64
--out validation/research/multiband_nu/results/w6_band_builder.json`,
`git_dirty` false, started 2026-09-07T04:46:53Z, wallclock 4.68 s (the
fixture is 3 x 20 x 296-356 cells, 1200 steps, seven runs).

**F7 (chain model, OLD vs NEW auto-z PCB profile).** dt 2.7471385804e-14 s
both sides. The OLD row is bit-identical to the first attempt (every
scalar, every cell, every step). The NEW profile differs from the first
attempt by one ulp in its cells (largest 173.36333674322023 ->
173.36333674322032 um, 4.9e-16 relative; nz 115, dz_min 8.333 um, 0
non-thirds pairs over 1.3 — unchanged), which moves the chain-model totals
at the 1e-8 level: `R_total` 5.6060172506e-5 -> 5.6060173316e-5 (1.4e-8),
`sum|R_step|` 1.0596039505e-4 -> 1.0596039240e-4 (2.5e-8),
`R_single(1.4, d_max)` 3.0330238597e-5 -> 3.0330235115e-5 (1.1e-7), one
`R_step` of a pair that differs by ulps 3.7e-13 -> 1.9e-13 (index 45; the
step count stays 46). Reason: the PCB chain solve holds 8.333 um cells
against a 10 GHz wavenumber, so the `inv_e inv_h` terms (about 1.4e10 /m^2)
sit six orders above `S0^2 - Sy^2` (about 4e4 /m^2) and the direct solve
carries about 1e-8 relative noise on R; this is the solver's conditioning,
not a physics change. Every digit quoted in the F7 table above is
reproduced.

| rule | measured (F7 recompute, 2026-09-11) | window (frozen) | verdict |
|---|---|---|---|
| F7a `|R|^2 <= 1.5 (sum|R_step|)^2` on NEW | 3.1427428e-9 | <= 1.6841469e-8 | **HELD** |
| F7b max non-thirds step <= R_single(1.4, d_max) (1 + 1e-9) on NEW | 2.4625978e-5 | <= 3.0330239e-5 | **HELD** |
| reported: OLD total / NEW total | 9.7561e-6 / 5.6060e-5 | — | lambda/450 caveat as above |

**F7 recomputed, 2026-09-11 (per-step reflections in closed form).** The
paragraph above blames the solver's conditioning for a 1e-8 wobble on
R_total. It is right about the mechanism and three orders low on the
size, and the first CI run of the replay test found it: on GitHub's
ubuntu runners every per-step R_step disagreed with this macOS-written
JSON by a median 2.9e-5 and up to 3.4e-4 relative, and the 46-step sum by
4.3e-6, past the replay's declared 1e-6. A single cell-size step has
exactly one non-uniform recurrence row, so its reflection is a 1e-7
residue that a dense float64 solve produces by cancelling 1e10-scale
terms — the last bits belong to the LAPACK build, not to the model.

That junction has a closed form, derived in the docstring of
`chain_model.py::step_reflection`: |R| = gam (d1^2 - d0^2)/4 divided by
(sqrt(1 - gam d0^2/4) + sqrt(1 - gam d1^2/4))^2, an admittance mismatch
with no subtraction of nearly equal numbers. Against a 60-decimal-digit
Gaussian elimination on the same rows the dense solve builds, it agrees
to 5e-51 relative: it is the exact value of the solved system, and the
macOS and linux float64 solves straddle it. The F7 block of the results
JSON was recomputed on it (F8 carried unchanged, its FDTD provenance kept
in a new `f8_provenance` key). Cells, dt, nz, the step count, max_ratio
and the over-1.4 pair count are bit-identical; no rule changed verdict;
every five-figure value in the F7 table earlier in this section is
reproduced. What moved is the eighth figure of the two rule rows above,
and the R_total / T_total pair by the linux-vs-macOS solve difference
alone (4.5e-7 OLD, 4.0e-8 NEW, 2.8e-12 T).

Open, for the lead: R_total and T_total still come from the dense solve,
and the OLD row's 4.5e-7 cross-platform spread uses 45 % of the replay's
1e-6 budget — flipping one cell of the OLD profile by a single ulp moves
R_total by up to 3.2e-7, so that budget is about one ulp-flip wide. A
stable form is plausible: each junction's exact reflection is the
admittance mismatch above, so a Riccati cascade over the uniform runs is
the candidate. It is not a drop-in — a throwaway prototype of that cascade
missed both rows by about 20 %, so the node/cell bookkeeping between
adjacent interfaces has to be derived, not guessed. Left out of a CI fix
on purpose.

**F8 (FDTD, narrow fine band).** dt(A) = dt(B) = 2.402764937e-12 s, gates
t_r 1.047 / gate_end 2.091 ns (870 steps) as declared; the builder emitted
the declared vector on every width (`builder_matches_declared_vector` true,
six of six). Every row is identical to the first attempt to <= 3e-10
relative (the largest difference is in a `deviation` field, i.e. below
1e-11 relative on `R_meas` itself).

| n_b (cells) | band (mm) | \|R\|_meas | \|R\|_model | window (frozen) | verdict |
|---|---|---|---|---|---|
| 2 | 2 | 7.4364e-3 | 7.4916e-3 | [5.9633e-3, 9.0199e-3] | **HELD** |
| **4 (gate)** | 4 | **1.0063e-2** | 1.0141e-2 | [8.0826e-3, 1.2199e-2] | **HELD** |
| 8 | 8 | 1.1164e-2 | 1.1296e-2 | [9.0066e-3, 1.3585e-2] | **HELD** |
| 16 | 16 | 1.2559e-3 | 1.2101e-3 | [9.3810e-4, 1.4822e-3] | **HELD** |
| 32 (extra, reported) | 32 | 1.4248e-3 | 1.5111e-3 | [1.1789e-3, 1.8434e-3] | inside |
| 64 (extra, reported) | 64 | 6.4596e-3 | 6.5575e-3 | [5.2160e-3, 7.8990e-3] | inside |

Validity domain, as the table reads: fine bands of 2, 4, 8 and 16 cells
(and the reported 32 and 64) are inside the domain at this resolution
pair (fine 30 / coarse 15.3 cells per free-space wavelength, ratio 1.4,
PEC closed, 10 GHz); no width fired. Still unwitnessed: bands under 2
cells, other resolutions, in-plane grading, an absorber present.

**Regression test added:** `tests/unit/nonuniform/test_band_builder_chain_model.py`
(15 tests, about 3 s, no FDTD) replays F7 from the committed JSON — the NEW
profile regenerated by `_make_dz_profile` cell for cell (1e-12 m), the OLD
profile from the JSON (no `git show` in a unit test), the chain-model
totals of both, the F7a/F7b rules recomputed, the F2 revert-proof numbers
(OLD max ratio 8.000, 25 pairs over 1.4) pinned — and the F8 model side
(builder profile per row cell for cell, `R_model` to 1e-9, the frozen
window arithmetic, the committed `R_meas` inside it). The transition law
is pinned on builder profiles A(n_b), n_b = 1..80: single ramp 5.7907e-3
and k_g(1.0 mm) 0.18163 /mm to 1e-4; every width bounded by twice the
single ramp; peaks at 7 / 24 / 41 / 59 / 76, nulls at 15 / 33 / 50 / 67,
maximum 2 x R_single x 0.9999; the Fabry-Perot form with L_eff = L +
1.87 mm within 1e-2 on the six W6 rows. Tolerances were written in the
file before its first run. Two facts from writing it, recorded rather
than rounded away: (i) the F7 replay tolerance is 1e-6 relative, not
1e-9, for the conditioning reason above (the JSON diff between the two
engines showed 1.4e-8 on `R_total` from a 5e-16 cell change); (ii) the
reviewer note's "with L_eff every row is within 0.5 %" measures 0.066 /
0.029 / 0.016 / 0.53 / 0.42 / 0.079 % for n_b = 2 / 4 / 8 / 16 / 32 / 64
on this tree — the n_b = 16 null row is 0.53 %, so the support-matrix
sentence now says "0.03-0.53 %". The thirds split's two pairs carry
ratios 1.5 and 2.0 (1/3, 2/3, 1 of the block cell), pinned as such.
