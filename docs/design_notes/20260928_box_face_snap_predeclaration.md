# Pre-declaration: a Box face drawn on a node plane realizes on that plane (issue 1138)

**Status: PRE-DECLARATION. Committed before the new test, the fixture census and any suite run
of this change.** The only numbers below come from build-only measurements of the node
builders on `main` dce96837 (no solve, no fixture), made to size the tolerance. They are
reproducible with the scratch scripts named in the PR body.

## The physics

A dielectric laminate drawn 787 µm thick on a mesh of h/6 realized 7 cells (918 µm) and on
h/3 realized 4 cells (1049 µm). The patch sheet on its top face was buried inside the
dielectric and the patch TM010 offset moved by about 1.8 percentage points. The top face had
been computed as `z0 + h`, which landed at 36.00000000000001 cells instead of 36. The volume
test is an exact half-open `lo <= node < hi`, so node 36 passed `< hi` and the laminate took
one more cell. A `lo` face one ulp above its node drops that node and the volume loses a cell
instead.

Measured on `main` (Box volume branch, one axis, build only): a face spelled `k*dx`
realizes exactly N cells in 1890 of 1890 cases. With the cell set to `h/N`, a face spelled `h`
realizes N + 1 in 25 of 441 cases and a face spelled `z0 + h` realizes N + 1 in 236 of 2205.
On a graded axis (node line from the cumulative sum of the cell sizes, face `z0 + h`) 947 of
2205 cases are wrong, 564 by +1 and 383 by −1. (N = 2..64, seven laminate thicknesses from
0.127 to 3.175 mm, six cell sizes, five offsets.)

## The rule

In the half-open test a face that lies within a tolerance `tol` of a sample point is ON that
sample point. `lo` then keeps the sample and `hi` drops it, which is what the convention
already does for a face bitwise equal to the sample:

    in  =  (x >= lo - tol)  &  (x < hi - tol)

Applied at both places the Box half-open test is written:

- `Box.mask_on_coords` (`rfx/geometry/csg.py`, `_axis_mask`), the node sampler every
  dielectric Box and every consumer of `Box.mask` reads. The thin-sheet tie rule below it
  keeps reading this window (`n_vol == 1`), so a one-cell box whose faces carry rounding dust
  still lands on its `lo` node.
- `_box_axis_volume` (`rfx/geometry/rasterize_grid.py`), the cell-centre sampler for PEC
  volumes, and the zero-cell refusal `_volume_is_empty`, which repeats the same comparison and
  must agree with it. There the sample points are cell centres, so a face within `tol` of a
  centre is on that centre. A face on a node is half a cell from every centre and is not
  affected.

Nothing else moves: the thin-sheet test, the nearest-node rule, the sheet footprint and the
curved-shape samplers are unchanged.

## The tolerance

    tol = max(1e-9 * d_local,  64 * eps(dtype of x) * max|x|)

`d_local` is the local cell at the Box's midpoint on that axis, the width `_axis_mask` already
uses for its thin-sheet test (on the centre line, the centre spacing there).

- **The first term** is the repo's existing "on the lattice" tolerance
  (`rasterize_grid._REL_TOL`, `_grid_metric.NODE_TIE_REL`), which the sheet footprint already
  applies (`_box_axis_closed`). A volume and a sheet then agree on what "on the node" means.
  It covers the concrete float64 node lines. Measured largest distance between a face
  intended on a node and that node, in local cells: uniform-valued axes up to 10 000 cells,
  1.8e-12; graded axes (grading up to 16:1) of 100 / 300 / 1000 cells, 5.1e-12 / 3.3e-11 /
  2.7e-10. **Out of envelope:** on graded axes of 3000 cells with 8:1 grading the float64
  cumulative sum reaches 1.6e-9 of the fine cell, and 10 000 graded cells reach 4e-8. There
  the flip can survive; the sheet footprint has the same limit today.
- **The second term** covers the traced node line (mesh as a design variable), which is
  float32 (`coords_from_nonuniform_grid` casts it). Its rounding grows with the axis extent:
  measured at most 5.4 · eps32 · max|x| over 16 000 on-node faces on 400 random graded
  profiles of 20 to 3000 cells, which is up to 7e-3 of a cell on the longest ones and far
  above 1e-9 of a cell. 64 gives a factor of about 12. For float64 the second term is below
  the first on any axis shorter than about 7e4 cells, so it does not change the concrete
  behaviour.
- **What the tolerance can move.** Only a face within `tol` of a sample: 1e-9 of a cell on
  a concrete grid, and on a traced grid about 7.6e-6 · (axis extent in cells) of a cell (7.6e-4
  at 100 cells). No drawing intended off a node sits that close to one. The traced lane then
  snaps faces in a band the concrete lane does not (between 1e-9 of a cell and its own
  `tol`); inside that band the traced node line cannot place a face anyway.

## Expected outcome (the new always-on test, no solve)

1. On a uniform axis, a Box whose faces are spelled `k*dx` and `(k+N)*dx`, or `0` and `h`
   with `dx = h/N`, or `z0` and `z0 + h` with `z0 = k*dx`, realizes **exactly N node planes**
   for N = 2..64, several `dx`, `h` and `k`, on x, y and z.
2. The same on a graded axis (concrete float64 node line) whose laminate cells are `h/N` and
   whose neighbours are coarser, faces spelled `z0 + h`.
3. The same on a **traced** float32 node line built the way the design-variable path builds
   it, under `jax.jit`, on a uniform-valued and a graded profile.
4. Through `Simulation`: the 787 µm laminate at h/3 and h/6 realizes 3 and 6 cells in the
   assembled permittivity (the reported board's drawing).
5. A PEC volume whose faces are spelled on cell centres through `a + b` realizes exactly N
   centre cells.
6. A one-cell Box whose faces carry rounding dust lands on its `lo`-face node.
7. **Unchanged elsewhere:** faces at 1e-7, 1e-3 and 0.5 of a cell from a node (concrete) and
   at 1e-2 of a cell (traced) realize exactly what the unsnapped rule gives.

## Falsifiers

- (a) Remove the snap (the raw `(x >= lo) & (x < hi)` at both sites): the test must red,
  and the PR lists which (N, dx or h, route, axis) cases flip.
- (b) Keep every helper call as it is and restore the defect inside the helper (tolerance
  set to zero): the test must red. The test's expected count is the integer N the drawing
  states, never a value computed by the helper under test.

## Blast radius

Every committed test, lock, fixture and example whose realized geometry changes under the fix
is listed in the PR: file, plane, physical quantity and how much it moves. Procedure: an audit
plugin (not committed) wraps the two helpers during the full test suite on VESSL and logs every
concrete call where the snapped window differs from the unsnapped one, with the test id and
the drawing. A lock that moves is re-derived only with this fix as the written cause and with
the old and new realized geometry. **Anything that moves for another reason is reported, not
re-pinned.** No expectation is declared for how many move (it is unknown).
