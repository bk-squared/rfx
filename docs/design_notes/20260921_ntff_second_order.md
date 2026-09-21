# The Huygens-box far field, made second order in the cell size

2026-09-21. Implementation note for the NTFF change on `fix/ntff-second-order`.

## What was wrong

A closed box is drawn round the radiator. On each of its six faces the two
tangential E and the two tangential H components are accumulated into a
running DFT during the run, and afterwards the radiation integral of
J = n x H and M = -n x E over that box gives the far field. Three things in
that path were first order or simply misregistered.

**The four components of a face do not sit on top of each other.** On the Yee
lattice, relative to node (i, j, k): ex at (i+1/2, j, k), ey at (i, j+1/2, k),
ez at (i, j, k+1/2), hx at (i, j+1/2, k+1/2), hy at (i+1/2, j, k+1/2), hz at
(i+1/2, j+1/2, k). For a face at node plane i, the two E components straddle
the face cell along different in-plane edges and the two H components sit
**half a cell off the face along its normal** — they are stored at the centres
of the cells on either side of the face, not on the face. The integral read all four at
the same array index and treated them as if they all sat at the cell's
lower-corner node.

**The quadrature was a left-endpoint rectangle rule.** Each face was summed
over the half-open index range [lo, hi) with the sample at that lower-corner
node and weight dx*dy — no half weights at the box edges, no midpoint. First
order in the cell size.

**E was stamped a full step early.** All three runners call
`accumulate_ntff` after the E update, so the state holds E at (n+1)*dt and H
at (n+1/2)*dt. The accumulator stamped E at n*dt and H at (n+1/2)*dt, which
puts E half a step BEHIND H instead of half a step ahead — a full step of
relative register error, not half.

A fourth, latent: the scalar area element returned dx*dy for every face,
which is the wrong pair for a y face (spanned by x and z) on any grid whose
y cells differ from its z cells. Not reachable from `Simulation` today.

T. Martin, 'An improved near- to far-zone transformation for the
finite-difference time-domain method', IEEE Trans. Antennas Propag.
46(9):1263-1271, 1998 (doi:10.1109/8.719968) treats the same stagger and
reports that integrating E and H on two displaced surfaces is more accurate
than averaging one of them onto a single surface; rfx keeps one surface,
because the box, its accumulators and every consumer are built around it, and
moves all four components to the face-cell centre instead - second order
either way.

## The design

1. **Collocate at the face-cell centre while accumulating.** For an x face at
   node plane i and face cell (j, k) whose centre is (x_i, y_{j+1/2},
   z_{k+1/2}): ey is averaged over k and k+1, ez over j and j+1, hy over
   j and j+1 AND interpolated across the face between planes i-1 and i, hz
   over k and k+1 AND the same normal interpolation. The same pattern on the
   y and z faces. Accumulator shapes are unchanged.

   The in-plane averages are between two NODES, so they are exact midpoints
   on any mesh. The normal interpolation is between two CELL CENTRES, so it
   carries linear weights from the two adjacent cell widths: the weight on
   the LOWER-INDEX sample is d[idx] / (d[idx-1] + d[idx]), which reduces to
   1/2 on a uniform axis. The
   weights are read off the grid at box-construction time and reach the scan
   as Python floats, so nothing inside `lax.scan` touches a grid attribute.

2. **Integrate with the midpoint rule.** `compute_far_field` and
   `compute_far_field_jax` place every sample at the centre of its face cell:
   the two in-plane coordinates are cell-edge midpoints, the normal
   coordinate stays on the face's node plane. dS remains the product of the
   two in-plane cell widths, now with the correct pair per face.

3. **Time stamps.** E at (n+1)*dt, H at (n+1/2)*dt — each field stamped with
   its own sample time, which makes the half-step register exact.

4. **Old dumps stay readable.** `NTFFBox` carries `face_centre`, defaulting to
   False (the pre-existing layout); `box.collocation` is a read-only property
   giving the readable name. It is a bool and not the string because every
   field of that NamedTuple is a JAX pytree LEAF, and a str leaf makes the
   whole box an invalid `jit`/`vmap`/`tree_map` argument — which it had never
   been before this field existed. `accumulate_ntff` and
   `compute_far_field(_jax)` both honour it, so accumulators saved by an
   earlier run are read with the geometry they were accumulated with. Every
   path that builds a box for a new run sets it: `make_ntff_box`,
   `NTFFBox.from_grid`, and the two runners that build a box inline
   (`rfx/runners/nonuniform.py`, `rfx/runners/subgridded.py`).

   The legacy setting reproduces the old READ geometry only. Accumulation
   always stamps E at (n+1)*dt now, whatever the collocation — the time
   register was a defect, not a layout choice, and a saved accumulator has
   its stamps baked in already.

   For reference, the flux monitor in the same scan body stamps E at n*dt and
   H at (n-1/2)*dt: the same E-H register as the NTFF accumulator, one step
   earlier in absolute time, which is a common phase that cancels in every
   magnitude.

5. Kahan summation, the `ntff_accum_dtype` policy (#646), the oblique-Bloch
   complex-field path (#404) and the direction chunking (#727) are untouched.

**A box must now sit at least one cell inside the array bounds on every
axis**, because the half-cell averages read one index further out than the
face. For a box drawn on the domain corners that further sample is
the innermost absorber cell. rfx grades the CPML conductivity to exactly zero
there (`rfx/boundaries/cpml.py`: `rho = 1 - arange(n)/(n-1)`, `sigma =
sigma_max * rho**order`), so that cell is updated as free space and the sample
is an ordinary one; no clearance from the absorber is needed beyond what the
equivalence principle already asks. Nothing upstream guaranteed that. Two configurations that ran before
now raise: a 2-D run (`mode="2d_tmz"`, one cell in z) and a box spanning the
whole domain with `cpml_layers=0`. The refusal happens in
`with_face_centre_collocation`, where the grid is still in hand, so it can
name the offending axis in metres as well as in cells and give the range that
axis can carry a face in; `accumulate_ntff` keeps an index-only backstop for
boxes built by hand, and the subgridded runner checks before its scan. A
1-cell axis is told the transform needs a 3-D box rather than to move its
faces. Verbatim, for the flush-box case:

```
NTFF box has no room for the face-centre half-cell averages. Moving a face
sample to the centre of its cell reads one index further out than the face
itself, so every face must sit at least one cell inside the array bounds
(1 <= lo < hi <= n-1).
  x: faces at index 0 and 20 of 21 cells (0 m and 0.06 m); this axis can
     carry a face anywhere in [0.003 m, 0.06 m]
  ...
Remedy: move each face at least one cell further inside the domain. The
corners come from Simulation.add_ntff_box(corner_lo=..., corner_hi=...)
(rfx.farfield.make_ntff_box).
```

## Measured convergence

Closed-form fields of three electric and one magnetic point dipole inside a
0.7-wavelength box at 3 GHz, written into the six field arrays at the true
Yee positions and true times, stepped through the production
`accumulate_ntff` and transformed by the production `compute_far_field`.
Error is the complex relative L2 over 25 x 16 directions, both far-field
components. Uniform mesh:

| surface sampling | lambda/10 | lambda/20 | lambda/40 | ratios |
|---|---|---|---|---|
| face-centre (now) | 1.479e-02 | 3.691e-03 | 9.225e-04 | 4.01, 4.00 |
| node rule (before) | 3.174e-01 | 1.599e-01 | 8.011e-02 | 1.99, 2.00 |

Same box and dipoles on a mesh whose cells stretch smoothly by 3x across one
axis (per-cell ratio 1.10 at the coarse mesh, 1.03 at the fine one), read
through the `dx_arr`/`dy_arr`/`dz` path. Every axis is graded in turn,
because the accumulator's slicing, the edge arrays and the area element are
written out per axis and a defect can hide on one of them:

| graded axis | lambda/10 | lambda/20 | lambda/40 | ratios |
|---|---|---|---|---|
| x, face-centre (now) | 1.577e-02 | 4.234e-03 | 1.128e-03 | 3.72, 3.75 |
| y, face-centre (now) | 1.859e-02 | 5.237e-03 | 1.442e-03 | 3.55, 3.63 |
| z, face-centre (now) | 1.663e-02 | 4.789e-03 | 1.339e-03 | 3.47, 3.58 |
| z, node rule (before) | 3.205e-01 | 1.621e-01 | 8.165e-02 | 1.98, 1.99 |

The graded family is not a pure halving — the grading ratio changes with the
cell count — so its ratios sit below 4 while the uniform ones do not.

The numpy and JAX transforms are checked against each other on a uniform box
and on a box graded on all three axes: relative L2 between them 4e-07 to
7e-07, which is float32 in the JAX path, not geometry.

At lambda/20 the midpoint rule's own quadrature error on the radiation phase,
(k*d)^2/24, is 4.1e-3, so 3.7e-3 is the floor of what this rule can reach on
that mesh.

## Mutation evidence

Every row re-introduces one defect with every helper call left in place, then
runs the gates. Unmutated on this branch: oracle 32 passed, battery fast
8 passed, battery slow_physics 5 passed, NU fixture slow_physics 2 passed.

| # | mutation | verdict | caught by |
|---|---|---|---|
| M1 | integrate at the corner node again (`centre=False` in the numpy position call) | RED | oracle error bound + convergence rate, all three meshes; 2.256e-01 / 1.134e-01 / 5.673e-02, ratios 1.99, 2.00 |
| M2 | no normal interpolation for H (weight 0 on the lower-index cell) | RED | same; 1.935e-01 / 9.737e-02 / 4.873e-02, ratios 1.99, 2.00 |
| M3 | E stamped at n*dt | RED | oracle (3.404e-01 / 3.427e-01 / 3.433e-01, ratios 0.99, 1.00); battery `test_ntff_absolute_power_calibration_vs_flux_box` (ratio 0.4922200), `test_dipole_directivity_regression_lock`, the ladder, the NU lane (ratio 0.492220329, D err 0.010132 dB), the 27 mm box |
| M4 | no in-plane half-cell average for E | RED | oracle; 2.977e-02 / 1.324e-02 / 6.366e-03, ratios 2.25, 2.08 |
| M5 | y-face area element back to dx*dy | RED | `test_scalar_face_area_elements_span_their_own_face` |
| M6 | face-cell layout recorded as a str again | RED | `test_box_is_a_valid_jax_pytree`, both refusal tests, the subgridded wiring test |
| M7 | JAX transform back on the corner node (numpy left alone) | RED | both `test_jax_transform_matches_numpy_on_a_*_box` |
| M8 | NU runner loses its `with_face_centre_collocation` call | RED | `test_nonuniform_runner_hands_back_a_face_centre_box`; battery slow `test_nonuniform_lane_power_calibration_and_slot`. Battery FAST lane stays green — the NU gate is slow_physics |
| M9 | subgridded runner loses `face_centre=True` | RED | `test_subgridded_runner_hands_back_a_face_centre_box` |
| M10 | weights applied to the wrong side (consumer swap) | RED | `test_accumulator_lands_on_the_face_cell_centre_exactly` |
| M11 | weight read from the wrong cell (producer swap) | RED | the same, plus `test_normal_weight_does_not_wrap_at_index_zero` and `test_normal_interpolation_lands_on_the_face_plane` |
| M12 | x and y cell widths crossed in the weight builder | RED | `test_accumulator_lands_on_the_face_cell_centre_exactly` |
| M13 | NTFF accumulate moved above the E update, uniform runner | RED | battery fast ratio + directivity; slow ladder and the 27 mm box |
| M14 | NTFF accumulate moved above the E update, NU runner | RED | battery slow `test_nonuniform_lane_power_calibration_and_slot`: ratio 0.506218892 against 0.5 +/- 0.003 and D err 0.010687 dB against 0.005. Fast lane green, as for M8 |
| M15 | flush-box refusal removed | RED | all three refusal tests |
| M16 | NU far-field fixture back to 200 steps | RED | both `test_nu_ntff_dipole_directivity` cases, on the settling witness |

M1-M4, M10-M12 are the ones the far-field oracle alone could not separate:
M10-M12 are all second order either way, and only the accumulate-level
exactness check sees them.

Not vacuous: perturbing the oracle's reference far field by a constant factor
and re-running the unmutated code turns the gates red at

| injected error | lambda/10 | lambda/20 | lambda/40 |
|---|---|---|---|
| 1.0 % | RED | RED | RED |
| 0.3 % | green | RED | RED |
| 0.1 % | green | green | RED |

## What moved in the existing gates

Same test files, same fixtures, run against `origin/main` and against this
branch. Nothing changed colour: 113 passed on both sides across
`tests/unit/farfield`, `tests/oracle/test_farfield.py`,
`tests/oracle/test_oblique_rcs_specular.py`,
`tests/unit/grid/test_x64_scan_carry_dtypes.py` and the four autodiff
far-field files, plus 8 passed on
`tests/locks/test_ntff_directivity_validation_battery.py`.

| fixture | quantity | before | after | expectation | closer? |
|---|---|---|---|---|---|
| locks NTFF battery, base rung | z-dipole D | 1.800012 dBi | 1.761430 dBi | 1.760913 dBi (analytic, 10log10(1.5)) | yes: 0.0391 -> 0.0005 dB |
| locks NTFF battery, base rung | P_ntff / P_flux | 0.494778 | 0.500321 | 0.5 (derived, convention audit) | yes: 1.04 % -> 0.064 % |
| locks NTFF battery, base rung | P_ntff | 2.165282e-32 W | 2.189538e-32 W | — | +1.12 % |
| locks NTFF battery, base rung | P_flux (independent witness) | 4.376268e-32 W | 4.376268e-32 W | — | unchanged |
| locks NTFF battery, x-dipole rung | D | 1.799945 dBi | 1.761379 dBi | 1.760913 dBi (analytic) | yes: 0.0390 -> 0.0005 dB |
| oracle test_farfield | short-dipole D | 1.836 dBi | 1.770 dBi | 1.76 dBi (analytic) | yes: 0.076 -> 0.010 dB |
| oracle test_farfield | half-wave dipole D | 2.380 dBi | 2.156 dBi | 2.15 dBi (analytic) | yes: 0.230 -> 0.006 dB |
| test_farfield_nonuniform, uniform-z via NU (settled, 600 steps) | D | 1.7062 dBi | 1.7001 dBi | 1.7609 dBi (analytic) | no: 0.055 -> 0.061 dB |
| test_farfield_nonuniform, graded-z (settled, 600 steps) | D | 1.7074 dBi | 1.7002 dBi | 1.7609 dBi (analytic) | no: 0.054 -> 0.061 dB |

The assertions in `tests/oracle/test_farfield.py` and
`test_farfield_nonuniform.py` are analytic expectations with tolerance bands
(0.75 dB, 0.5 dB, 0.3 dB). The locks battery is different: `_LADDER_RUNGS`
carried per-rung error caps and power-ratio centres measured with the old
rule. They stayed green, but at those caps a full revert of this change would
have stayed green too. This PR re-pins them to what the corrected transform
measures and rewrites the battery's caveat 2, which attributed part of the
ratio offset to the E time label removed here. The root cause of the move is
the one this note describes; the caps only tighten.

| rung | old cap | old measured | new measured | new cap |
|---|---|---|---|---|
| dx 3.00 mm, 400 steps | 0.15 dB | 0.0391 dB | 0.00052 dB | 0.005 dB |
| dx 1.50 mm, 800 steps | 0.08 dB | 0.0147 dB | 0.00012 dB | 0.002 dB |
| dx 0.75 mm, 1600 steps | 0.04 dB | 0.0064 dB | 0.00006 dB | 0.001 dB |

| ratio centre | old | new |
|---|---|---|
| dx 3.00 mm | 0.4948 | 0.5003207 |
| dx 1.50 mm | 0.5128 | 0.5150090 |
| dx 0.75 mm | 0.5215 | 0.5225705 |

Band +/-0.02 per rung, unchanged. The two fast base-rung gates move with
them: the directivity cap 0.25 -> 0.005 dB and the derived-0.5 power-ratio
tolerance 0.05 -> 0.003.

The two non-uniform rows say nothing about the transform. That test stopped
at 200 steps, when the field at the probe had not begun to decay (end/peak
0.94 and 1.00; the run's own settling witness fails at -1.5 dB and -0.2 dB
against the -40 dB rule), so both columns of those rows are from a cut
transient. With the record settled — 600 steps, unchanged to 3000, tail/peak
1.3e-4 — the fixture gives 1.7062 / 1.7074 dBi with the old rule and
1.7001 / 1.7002 dBi with this one, against 1.7609 dBi. Both rules sit
0.05-0.06 dB low, and that bulk is common to both. The 0.006-0.007 dB between
them is the surface rule: the first-order rule reads high on every fixture
measured (+0.039 / +0.015 / +0.006 dB on the battery ladder), and here that
bias happens to point toward the analytic value. Which rule is closer to this
fixture's true directivity is not known, because the fixture's own 0.06 dB
offset is unexplained. Its cause was
not investigated: it is thirty times inside the 2 dB bar, on a
0.3-wavelength domain whose box faces realize three cells from the source
(and seven from the absorber, which rfx pads outside the declared domain).
The test now runs 600 steps and asserts the settling witness.

The non-uniform lane, which the uniform gates never reached:

| lane | realized grid | D | \|err\| vs 1.760913 dBi | P_ntff / P_flux | settling |
|---|---|---|---|---|---|
| non-uniform runner, 20 z cells | 33 x 33 x 33 | 1.761428040 dBi | 0.000515 dB | 0.500321038 | -78.40 dB |
| uniform runner, same fixture | 33 x 33 x 33 | 1.761429829 dBi | 0.000517 dB | 0.500320703 | -- |

On a z profile that realizes the same grid as the uniform fixture (20 cells;
both lanes 33 x 33 x 33) the non-uniform lane reproduces the uniform one:
NTFF-to-flux power ratio 0.5003210 against 0.5003207, directivity 1.761428040
dBi against 1.7614298 dBi. The far-field chain is the same on both lanes,
accumulation slot included, and the battery now asserts it. The fixture also
asserts the realized grid shape, because the power ratio follows where the
absorber is (the battery's caveat 2 records the same sensitivity to the
absorber's thickness); a z profile with a different cell count is a different
structure, not a different lane.

`tests/crossval/` and `validation/crossval/` were not run (another session
owns them this week).

## The subgridded lane

Experimental (3D SBP-SAT falsified, PR #90). It gets a wiring assertion only
— that its inline box comes back face-centre — and no accuracy claim is made
for its far field here.

## Where the numbers live

- Oracle, the accumulate-level exactness check, the numpy/JAX parity check,
  the pytree check, the runner wiring checks and the refusal messages:
  `tests/unit/farfield/test_ntff_second_order_oracle.py`
- The re-pinned rungs, the NU lane slot witness and the box-size witness:
  `tests/locks/test_ntff_directivity_validation_battery.py`
- The settled non-uniform fixture:
  `tests/unit/farfield/test_farfield_nonuniform.py`
- The transform: `rfx/farfield.py`
  (`accumulate_ntff`, `compute_far_field`, `compute_far_field_jax`,
  `with_face_centre_collocation`, `_normal_weight`, `_scalar_face_dS`,
  `_raise_face_centre_margin`)

## Verdict against the v2 accuracy bar

Against the v2 accuracy bar (power and magnitude within 2 dB of the reference
on a converged mesh) the old rule was already inside on every fixture above;
its largest error here is 0.23 dB, on the half-wave dipole. This change does
not rescue a failing result. It changes the order of the error: the old one
halved per halving of the mesh and grew to a sixth of a dB as the box drawn
round the radiator was enlarged, the new one falls by four and stays eighty
times smaller over the same sweep — at box half-width 9 / 18 / 27 mm the
directivity error was 0.0212 / 0.0677 / 0.1660 dB and is now 0.00013 /
0.00050 / 0.00204 dB
(`test_directivity_error_stays_small_however_large_the_huygens_box`). A user gets a
far field that is good to a few thousandths of a dB however large they drew
the box (up to the 27 mm tested), and a radiated power that agrees with an
independent flux box to 0.06 % instead of 1 %.
