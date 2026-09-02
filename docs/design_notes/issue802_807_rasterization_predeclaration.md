# Pre-declaration — exact node coordinates for rasterization (#802, #807)

Date: 2026-09-01. Written BEFORE any fix run. Implementer session, isolated
worktree branched from `main` @ 92018513.

## R1 memory citations

- `docs/agent-memory/rfx-known-issues.md` line 24: "#802 realized cells depend
  on JAX_ENABLE_X64 — f32 node rounding at node-aligned faces". This change
  implements the issue's own proposed direction (exact host coordinates,
  traced path preserved). Consistent.
- Memory "Generalize, don't debug per example": the fix removes the duplicated
  node-coordinate constructions (csg `_grid_coords`, rasterize_grid uniform
  builder, NU cumsum cast) and replaces them with ONE exact builder, rather
  than re-pinning fixtures one by one. Consistent.
- Memory "A physics gate can bind an ARTIFACT" + "Root cause required before
  changing test gates": every committed value that moves is enumerated below
  (F3) with provenance; no tolerance is loosened.
- Memory "A uniform-valued dz_profile tests plumbing, not metrics": #807 is
  that entry in reverse — the plumbing itself differs between lanes. The lane
  contract test grades each axis separately so an axis swap cannot pass.
- `.claude/rules/rfx-feature-discovery.md` (two hand-copies drift): the second
  uniform builder in `rasterize_grid.py` already disagrees with csg's at ~1/3
  of nodes at dx=100 µm — this fix deletes the duplication instead of syncing it.

## The defect (measured in recon, this session)

Node coordinates are built three ways, and under the default precision
(x64=0) all three round differently at the last ulp:

1. `rfx/geometry/csg.py::_grid_coords` — `(jnp.arange(n)-pad)*dx` in JAX
   default dtype → f32(f32(i)·f32(dx)) (double rounding). Production uniform
   lane (`api/_compile.py`, `Shape.mask`).
2. `rfx/geometry/rasterize_grid.py::coords_from_uniform_grid` — f64 product
   cast once to f32. Viewer/tests.
3. `rfx/geometry/rasterize_grid.py::coords_from_nonuniform_grid` — f64 cumsum
   of the grid's f32-STORED cell sizes, force-cast f32. NU lane.

At node-aligned faces the f32 coordinate lands ~1e-10 m off the exact node
value, so the documented half-open `[lo, hi)` Box convention flips per face,
per axis, per rounding accident (#802), and a one-cell sheet's thin-branch
argmin (an exact half-cell tie) resolves to different planes in different
lanes (#807). x64=1 obeys the convention exactly on the uniform lane but the
NU lane stays f32 — "just enable x64" widens the lane split.

## The fix (design, adversarially reviewed before this session)

- ONE exact builder `_uniform_axis_nodes(n, pad, dx)` → host numpy float64
  `(arange - pad) * dx`, flag-independent, bit-identical to the x64=1
  realization the issue verified as convention-exact. `csg._grid_coords`,
  `coords_from_uniform_grid`, and the uniform-valued branch of
  `_axis_node_positions` all call it.
- `coords_from_nonuniform_grid` concrete path returns host float64 (drops the
  f32 cast). The exact f64 cell-size profile is carried on new trailing
  optional `NonUniformGrid` fields populated by `make_nonuniform_grid`
  (concrete path only); solver arithmetic keeps reading the existing f32
  fields, so field updates are untouched — including under x64=1, which
  cv09/cv01/cv02/cv03 pin. Hand-built grids without the fields fall back to
  the f64-widened f32 profile (today's values).
- `Box._axis_mask`, `Cylinder`/`Sphere`/`PolylineWire.mask_on_coords`:
  concrete coordinates are compared on the host in float64 (`jnp.asarray`
  would silently downcast under x64=0 — the defect itself); traced
  coordinates (mesh-as-design-variable) keep the existing jnp path unchanged.
  MeshShape already raises on tracers and already uses host f64.
- `coords_from_fine_grid` (subgrid, cv12/13 experimental fence) follows the
  same policy: concrete producers emit host f64. Noted, not separately gated —
  the fine region is outside the validated set.

## Falsifiers (declared before any fix run)

Interpreter: `python` from the worktree with `PYTHONPATH=<worktree>`;
every evidence JSON records `rfx.__file__`. Scratchpad =
`/tmp/claude-0/-root-workspace-byungkwan-workspace-research-rfx/84ed5b7d-c5f2-49fd-b30c-d2477926a026/scratchpad`.

**F1 — convention exactness, flag invariance (#802 repro).**
Fixture verbatim from the issue: `Box((0,0,2.5e-3),(12.5e-3,3.4e-3,2.8e-3))`,
domain (12.5, 3.4, 3.9) mm, dx=100 µm, cpml_layers=8.
Command: `JAX_ENABLE_X64=0 python <scratchpad>/repro_802.py` and the same
with `=1`.
PASS iff BOTH flags give: cells=12750, occupied interior nodes x 0..124,
y 0..33, z 25..27, and per-face: lo-face node occupied, hi-face node not
occupied, on every axis. Anything else = fix falsified.
Baseline (pre-fix) expectation: x64=0 gives 13230 cells, x 0..125, y 0..34
(hi-face nodes included against the convention).

**F2 — lane equality (#807).**
(a) The #807 fixture: dom (3,3,2) mm, dx=100 µm, cpml_layers=4,
`Box((0.5,0.5,0.4),(2.5,2.5,0.5))` mm (one cell thick in z).
(b) A 3-axis sweep: the same one-cell sheet with its normal along x, y, z,
each with a uniform-valued profile on EVERY axis of the NU grid.
Command: `JAX_ENABLE_X64=0 python <scratchpad>/f2_lane_equality.py` and
`=1`.
PASS iff for every case and BOTH flags: `shape.mask(uniform_grid)` equals
`shape.mask_on_coords(*coords_from_nonuniform_grid(nu_grid))` bitwise on the
shared index space, the sheet occupies exactly ONE plane, the SAME plane in
both lanes, and the interior node-coordinate arrays are bitwise equal
(the stronger by-construction claim).
Baseline (pre-fix) expectation: x64=0 lanes agree by structural luck
(plane [9]); x64=1 lanes split ([8] uniform vs [9] NU).

**F3 — no off-list movement.**
Every committed value that moves must be ON the WILL-MOVE list below,
each moved as an enumerated, separately-committed re-pin with provenance
text (old = f32 rounding artifact of #802, new = documented convention).
Any moved committed value NOT on the list = STOP and report, not a re-pin.

**F4 — traced paths unbroken.**
Command: `pytest tests/test_nonuniform_forward_grad.py
tests/test_nonuniform_gradient.py -q` plus a new traced-profile contract
test (jax.grad through a mask-dependent scalar w.r.t. dz_profile).
PASS iff all pass.

**F5 — off-lattice geometry bit-identical pre/post.**
The fix must change ONLY the broken node-aligned/knife-edge cases. Fixture
(one grid, dom (3,3,2) mm, dx=100 µm, cpml=4, x64=0): a midpoint-recipe
volume Box, a one-cell sheet registered ON a node plane (mid = node), an
off-lattice Cylinder, Sphere, and PolylineWire (radii/centres away from
node-coincident distances).
Command pre-fix: `JAX_ENABLE_X64=0 python <scratchpad>/f5_offlattice.py
capture` (writes masks npz + JSON). Post-fix: `... f5_offlattice.py compare`.
PASS iff every mask is `array_equal` to its pre-fix capture.

## F3 WILL-MOVE enumeration (superset allowed to move; each its own re-pin commit)

CI-visible (red on this PR without a re-pin):
1. `tests/data/example_fidelity_snapshot.json` — realized_um / n_cells /
   findings rows for variants with nonzero node-aligned faces (cv06b msl
   notch, ports_and_sparams_101, cv11 pec_short, slab_rt_flux_monitor
   with_slab, artifact_report_demo, cv15 patch, thru_feedpost_* /
   twoseg / offdiag-uniform, convergence_floor fixtures, multiband w4/w4r) +
   face_residual/message drift. One whole-file enumerated re-capture commit
   via `scripts/capture_example_fidelity_snapshot.py`, key-by-key diff quoted.
   (PR #734 owns the separate domain-row defect — NOT folded in.)
2. `tests/test_waveguide_geometry_hygiene.py` —
   `test_production_node_coords_differ_from_an_f64_construction` INVERTS
   (production nodes become the f64 construction); `_NOMINAL_EXCESS`
   re-measured (expected uniform 1, but measured, not predicted); docstrings.
3. `tests/test_rasterized_slice_viewer.py` two-plane wall test — body plane
   9 → 8, per-plane cells 361 → 400, wall plane 10 → 9.
4. `tests/fixtures/golden_msl_sheet_thread_{s,freqs}_13de212.npy` +
   `tests/test_msl_sheet_threading.py` byte-identity gate — trace boxes are
   node-aligned at dx=80 µm.
5. `tests/crossval/test_crossval_cv15_wall_planes.py` + cv15 committed wall-plane
   expectations (live fast-CI build on node-aligned z faces + one-cell ground).
6. `tests/fixtures/msl_z0_length_invariance/platform_datums.json` — new datum
   entries per the ledger's append-only protocol (datums recorded at x64=0).
7. `tests/locks/test_two_plane_pec_slab.py` off-state golden + pinned plane k=3
   (exact half-cell tie — verify, may be stable).
8. `tests/test_coax_msl_transition.py` knife-edge cell count narrative
   (assertion doesn't pin the count; verify it survives).
9. `tests/crossval/test_msl_phase_referee_header.py` realized-substrate pins (live).
10. `tests/test_sheet_node_permittivity.py` sheet plane / eps values (verify).
11. `tests/test_lumped_twoport_vi_validation_battery.py` (verify n_live).
12. `tests/locks/test_refplane_port_waves.py` physics legs (verify; plane indices
    themselves are python arithmetic, safe).
13. `tests/test_preflight_campaign_statics.py` docstring-recorded node sets /
    cell counts (verify).
14. `tests/test_conductor_mask_accessor.py` (shares the thru fixture; verify).
15. Box/csg docstrings (f32 double-rounding narrative → historical) and
    `validation/crossval/11_waveguide_port_wr90.py` PRECISION REQUIREMENT
    paragraph.
16. `CHANGELOG.md` — user-visible realized-geometry change for default-
    precision runs at node-aligned faces; #589 f64 replicate unblocked.

Slow / opt-in / regen lanes (won't red on this PR; enumerated so the next
regeneration is not a silent re-pin):
17. `tests/data/v173a_pre_t7_phase2_baseline.json` (slow_physics lane; its
    own "chore(baseline): rebump" protocol is the vehicle).
18. `tests/crossval/test_patch_canonical_farfield_e4.py` slow FDTD gates (cv05 patch
    is node-aligned at dx=2 mm).
19. Frozen-replay evidence fixtures that become stale at next regeneration:
    cv05/cv06b/cv07 committed results, Mie RCS fixtures (Sphere node-radius
    knife edges), broad-E5 envelope fixtures (cv11-class PEC shorts), Palace/
    openEMS referee fixtures.

Cross-references only — NOT fixed here: #806 (preflight half-cell-margin
advisory), #722/#729/#752 (declared-vs-realized catalogues; part of #722's
"half-open rounds up" findings re-scope to f32 error — note on the issue,
don't rewrite history), #589 (f64 replicate unblocks when this lands),
#720/#767 (per #807's text), PR #734 (owns the snapshot domain-row defect).

## R2 / R3 posture

R2 attempt count entering implementation: 0 (recon + design probes all closed
with predeclared expectations). This is a one-attempt implementation of a
reviewed design; if F1/F2/F5 do not pass as declared, that is a STOP —
re-read memory and report, not iterate.

## Baseline (pre-fix) measurements — recorded before the first fix edit

Run on the unmodified worktree (main @ 92018513), interpreter importing the
worktree (`rfx.__file__` recorded in the JSON evidence), jax 0.6.2.

**F1 baseline (RED, as declared).**
```
x64=False  coords dtype=float32  cells=13230
  x occupied 0..125 (hi-face node 125 coord=f32(0.012499999) < hi -> INCLUDED, against convention)
  y occupied 0..34  (hi-face node 34  coord=f32(0.0033999998) < hi -> INCLUDED, against convention)
  z occupied 25..27 (both faces per convention; z-lo compares f32-vs-f32 and stays included)
x64=True   coords dtype=float64  cells=12750
  x 0..124, y 0..33, z 25..27 — every face exactly per the documented convention
```
(The issue's per-face table row "z lo node 25 EXCLUDED" does not reproduce in
production — the comparison runs f32-vs-f32 — consistent with the issue's own
headline 13230 = 126·35·3. Regression tests compare masks, not that row.)

**F2 baseline (as declared: agree by luck at x64=0, split at x64=1).**
```
x64=0: all 3 cases planes U=[9] NU=[9], 361 cells, masks_equal=True, coords bitwise equal
x64=1: all 3 cases planes U=[8] NU=[9], cells 400 vs 361, masks_equal=False, coords differ
```
Evidence: scratchpad f2_result_x640.json / f2_result_x641.json.

**F5 baseline captured (x64=0).** f5_baseline.npz cells:
box_midpoint_volume=1444, box_sheet_on_node=361, cylinder_offlattice=1071,
sphere_offlattice=1506, wire_offlattice=699.

## Post-fix falsifier results

Measured immediately after the implementation commit, same commands, same
interpreter (`rfx.__file__` = worktree, jax 0.6.2).

**F1 — PASS.** Both flags:
```
coords dtype=float64  cells=12750
x occupied 0..124: lo node 0 coord=0.0 IN; hi node 125 coord=0.0125 OUT
y occupied 0..33:  lo node 0 coord=0.0 IN; hi node 34 coord=0.0034000000000000002 OUT
z occupied 25..27: lo node 25 coord=0.0025 IN; hi node 28 coord=0.0028 OUT
```
Identical output at x64=0 and x64=1; every face per the documented
convention.

**F2 — PASS.** Both flags, all three sheet normals:
planes U=[8] NU=[8], 400 cells per plane (20x20 nodes for the half-open
[0.5, 2.5) mm transverse span), masks bitwise equal, node-coordinate
arrays bitwise equal. Evidence: scratchpad f2_result_x640.json /
f2_result_x641.json. Note the plane moved DOWN one ([9] -> [8]) and the
transverse span gained its convention-owed node (19 -> 20): both are the
f32 artifact leaving, as pre-enumerated (WILL-MOVE items 3, 5).

**F4 — PASS.** `tests/test_nonuniform_forward_grad.py` +
`tests/test_nonuniform_gradient.py`: 15 passed. New
`tests/test_rasterization_coordinate_exactness.py` (includes the traced
jit/grad guard and the shape-class census): 57 passed.

**F5 — PASS.** All five off-lattice shapes bit-identical to the pre-fix
capture: box_midpoint_volume 1444, box_sheet_on_node 361,
cylinder_offlattice 1071, sphere_offlattice 1506, wire_offlattice 699
cells, `array_equal` True each. Evidence: scratchpad
f5_compare_result.json.

**F3 — tracked through the re-pin commits that follow; any moved committed
value not on the list above is a STOP.**

## F3 investigation — the thin-branch tie sub-defect (declared before its fix)

The broad sweep surfaced five failing surfaces beyond the enumerated
re-pins. Root cause, measured per fixture (scratchpad tie_probe.py): a
face-registered ONE-CELL box is an exact half-cell tie in the thin branch,
and with exact f64 coordinates the argmin is decided by the last ulp of
``(lo+hi)*0.5`` — it flipped UP on cv15's sheets and the fidelity-report
fixture ([2.0, 2.5] mm at 500 µm) while flipping DOWN on the viewer
fixture. Same declaration idiom, different realized side, decided by
invisible ulps — the exact class this fix exists to kill, one level down.

Three committed surfaces independently encode the convention-consistent
answer (cv15's stack contract, ``test_exact_faces_report_zero_residual``,
the ground-own-cell-vacuum fixture): a face-registered one-cell box
realizes on its LO node — the node its own half-open ``[lo, hi)`` volume
window keeps.

**Declared rule (implemented next):** in the thin branch, when the volume
window ``[lo, hi)`` selects EXACTLY ONE node, that node IS the realized
plane (it is the documented convention's own answer, and it is ulp-robust
in exactly the way the volume branch is); only a window with zero or
several nodes falls back to nearest-node argmin. Falsifier: the two
fidelity-report tests and cv15's ground wall go green WITHOUT touching
their assertions; F1/F2/F5 and the 57-test contract suite stay green;
the already-committed F2 values (plane [8], 400 cells) are unchanged
because [0.4, 0.5) mm selects node 8 uniquely.

Tie-rule verification (post-implementation, same session): the two
fidelity-report tests, cv15's wall-planes test, the two-plane slab pins,
the sheet-node permittivity suite and the 57-test contract suite all
green with no assertion touched (134 passed in one run); F1/F2/F5
re-verified bitwise-identical results; a traced one-cell sheet still
jits and differentiates (scratchpad traced_thin_smoke.py). The
authoritative waveguide-port battery: 9 passed.

**Knife-edge residual (measured, not fixable by any tie rule):** cv15's
``z_sub_hi = AIR_BELOW + H_SUB`` lands one f64 ulp ABOVE the node
``14*DX`` (0.0111125 vs 0.011112499999999999) — a sum-vs-product route
mismatch at a corner INTENDED on-lattice (the script's own comments say
so). Under f32 both routes collapsed to one value; exact comparison
surfaces it. The remedy is the declaration route, not the comparator:
spell the intended-on-lattice corner in lattice arithmetic
(``(10 + N_SUB) * DX``) — the same real number, the bit-exact spelling.
That is a registration fix consistent with the script's stated intent,
not a geometry change; the committed cv15 results assume walls at the
declared planes and keep their validity.

## F3 closure — final verification sweeps (post all re-pins)

- Broad targeted sweep `pytest tests -k "raster or geometry or mask or
  fidelity or realized"`: 520 passed, 1 skipped (one apparent failure was
  a race with this session's own snapshot re-capture rewriting
  `tests/data/example_fidelity_snapshot.json` mid-run; the full
  `tests/contracts/test_example_fidelity_contract.py` re-run against the committed
  state: 154 passed).
- WILL-MOVE candidate suites (msl sheet threading golden, two-plane
  slab, refplane port waves, preflight campaign statics, lumped two-port
  battery, msl phase referee header, conductor-mask accessor, msl port
  preflight, mesh import, non-box thin conductor, Sheen LPF and MSL
  notch referee gates, example fidelity contract): 374 passed.
- Waveguide port validation battery (authoritative port gates): 9 passed.
- New contract suite: 57 passed. NU AD tests: 15 passed. Ruff over
  `rfx/ tests/` + touched validation files: clean.
- Every committed value that moved is on the pre-enumerated WILL-MOVE
  list and carried its own re-pin commit; the surfaces that moved
  OFF-list in the first pass (fidelity-report tie fixtures, cv15 wall
  planes, msl z0 anchor, coax junction chain, stage1 bit-identity) were
  investigated per the F3 protocol, root-caused to the thin-branch tie
  sub-defect and the f64 route-mismatch class, fixed by rule (not by
  fixture), and are green or re-pinned with provenance above. No
  tolerance was loosened anywhere.

## Residual ledger — what this change does NOT close (next lanes)

- **cv11 quoted realized numbers** are an as-solved historical record
  (its slab now realizes x-nodes 95..105 under both flags; the quoted
  numbers were measured on the pre-fix 95..104 realization at x64=0).
  The file's precision paragraph says so; the next regeneration
  re-measures, never splices eras.
- **MSL z0 anchor artifact**: refreshing it requires RE-SOLVING the
  frozen bias-floor sweep on the exact-coordinate realization (the
  aligned class's trace width moved by one cell in three of four rows) —
  a re-measurement lane, pre-declared here so the next regeneration is
  not a silent re-pin.
- **Slow / opt-in lanes not run in this session** (enumerated, may move
  and then need their own root-caused re-pins with this note as
  provenance): `tests/crossval/test_patch_canonical_farfield_e4.py` slow FDTD
  gates (cv05 patch is node-aligned at dx=2 mm),
  `tests/test_v173a_physics_equivalence_slow.py` (its own rebump
  protocol), GPU/VESSL envelope regeneration lanes (broad-E5 fixtures
  with cv11-class PEC shorts).
- **Frozen-replay evidence fixtures** (cv05/cv06b/cv07 results, Mie RCS
  sphere fixtures, Palace/openEMS referee fixtures): CI-green now,
  become stale at their next regeneration — regenerate against the
  exact-coordinate realization and attribute movements to this change.
- **#722**: part of its "half-open rounds up" catalogue was f32
  coordinate error, not the convention — note on the issue when closing
  #802, do not rewrite its history.
- **#806** (preflight half-cell-margin advisory): untouched here,
  cross-referenced only; its premise should be re-checked now that
  preflight and production share exact node values.
- **#589**: the f64 replicate is unblocked (realized geometry no longer
  depends on the flag; the coax junction byte-identity chain that caught
  #802 is x64-invariant again).
- **rfx-known-issues ledger**: gitignored and local to the primary
  checkout — this worktree cannot update it; the merging session must
  move #802/#807 to resolved there with this note as the artifact.
