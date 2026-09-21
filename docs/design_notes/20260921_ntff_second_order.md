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
**half a cell off the face along its normal** — they are stored at the centre
of the cell outside the face, not on the face. The integral read all four at
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
face. Nothing upstream guaranteed that, so `accumulate_ntff` refuses a box
that does not, with the offending axes named.

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

Same box and dipoles on a mesh whose z cells stretch smoothly by 3x across
the axis (per-cell ratio 1.10 at the coarse mesh, 1.03 at the fine one), read
through the `dx_arr`/`dy_arr`/`dz` path:

| surface sampling | lambda/10 | lambda/20 | lambda/40 | ratios |
|---|---|---|---|---|
| face-centre (now) | 1.663e-02 | 4.789e-03 | 1.339e-03 | 3.47, 3.58 |
| node rule (before) | 3.205e-01 | 1.621e-01 | 8.165e-02 | 1.98, 1.99 |

The graded family is not a pure halving — the grading ratio changes with the
cell count — so its ratios sit below 4 while the uniform ones do not.

At lambda/20 the midpoint rule's own quadrature error on the radiation phase,
(k*d)^2/24, is 4.1e-3, so 3.7e-3 is the floor of what this rule can reach on
that mesh.

## Mutation evidence

Each row re-introduces one defect in `rfx/farfield.py` with every helper call
left in place, then runs
`tests/unit/farfield/test_ntff_second_order_oracle.py`. Unmutated: 16 passed.

| mutation | result | oracle error, lambda/10 / /20 / /40 | ratios |
|---|---|---|---|
| integrate at the corner node again (`centre=False` in the position call) | RED | 2.256e-01 / 1.134e-01 / 5.673e-02 | 1.99, 2.00 |
| no normal interpolation for H (weight 0 on the inner cell) | RED | 1.935e-01 / 9.737e-02 / 4.873e-02 | 1.99, 2.00 |
| E stamped at n*dt | RED | 3.404e-01 / 3.427e-01 / 3.433e-01 | 0.99, 1.00 |
| no in-plane half-cell average for E | RED | 2.977e-02 / 1.324e-02 / 6.366e-03 | 2.25, 2.08 |
| y-face area back to dx*dy | RED | (area invariant test) | — |

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
| test_farfield_nonuniform, uniform-z via NU | D | 1.7218 dBi | 1.7075 dBi | 1.7609 dBi (analytic) | no: 0.039 -> 0.053 dB |
| test_farfield_nonuniform, graded-z | D | 1.7167 dBi | 1.7135 dBi | 1.7609 dBi (analytic) | no: 0.044 -> 0.047 dB |

Every one of these assertions is an analytic expectation with a tolerance
band (0.25 dB, 0.75 dB, 0.5 dB, 0.3 dB respectively); none is a recorded
snapshot, and no gate was re-pinned. The two non-uniform rows sit on a
fixture whose NTFF box is 5 mm from the source while lambda/4 is 25 mm —
preflight emits six near-field advisories on it.

LEADER FILLS — what this says about the transform.

`tests/crossval/` and `validation/crossval/` were not run (another session
owns them this week).

## Where the numbers live

- Oracle, mutations and the always-on gate:
  `tests/unit/farfield/test_ntff_second_order_oracle.py`
- The transform: `rfx/farfield.py`
  (`accumulate_ntff`, `compute_far_field`, `compute_far_field_jax`,
  `with_face_centre_collocation`, `_scalar_face_dS`,
  `_require_face_centre_margin`)

LEADER FILLS — verdict against the v2 accuracy bar.
