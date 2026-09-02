"""A node-thin conductor's cell must carry the material its LIVE edge sits in.

A conductor thinner than a cell is realized node-thin: it occupies one cell
layer, ``apply_pec_mask`` zeroes that cell's two in-plane E edges and leaves
the sheet-NORMAL edge alone (it carries the surface charge). Nothing writes
``eps_r`` at such a cell — a PEC entry writes only ``pec_mask`` — so wherever
the surrounding dielectric boxes abut the metal faces instead of spanning its
thickness (what a real stackup or a CAD export gives: the metal layer is a
slot no dielectric fills) that cell keeps the default vacuum. The surviving
normal edge then integrates vacuum in series across the cavity the sheet
bounds.

Measured before the fix on a 31.43 um mesh, 17 um copper between eps_r 3.52
below and 3.38 above: sheet-node ``eps_r`` 1.000, and a 14-cell cavity
``sum(d/eps_r)`` of 149.726 um against the physical stack's 127.59 um. The
whole error is that one cell, ``31.43*(1 - 1/3.38) = 22.13 um``.

The fix samples ``eps_r``/``sigma`` for exactly those cells at ``node + d/2``
along the live axis, which is where rfx's own staggering puts the normal E
component. It changes no geometry and no mesh. Crucially it does NOT invent
dielectric: an OUTER conductor with air above resamples to air.
"""
import warnings

import numpy as np
import pytest
import jax.numpy as jnp

from rfx.api import Simulation
from rfx.boundaries.pec import tangential_edge_masks
from rfx.boundaries.spec import BoundarySpec
from rfx.geometry.csg import Box, _grid_coords
from rfx.geometry.rasterize_grid import (
    GridCoords,
    coords_from_nonuniform_grid,
    rasterize_geometry,
)
from rfx.runners.nonuniform import assemble_materials_nu, build_nonuniform_grid

T_CU = 17e-6          # sub-cell copper thickness
D_STACK = 31.43e-6    # stack cell
N_STACK, N_PAD = 30, 6
WIDE = (-1.0, 1.0)
FACES = {"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"}

DZ_UNIFORM_STACK = np.concatenate([
    np.full(N_PAD, 100e-6), np.full(N_STACK, D_STACK), np.full(N_PAD, 100e-6)])
# One profile carrying BOTH a 20 um and a 100 um stack cell, so a global or a
# neighbour's half-step cannot pass the graded test below.
DZ_GRADED = np.concatenate([
    np.full(N_PAD, 100e-6),
    np.array([100e-6, 100e-6, 20e-6, 100e-6, 100e-6, 100e-6, 100e-6, 100e-6]),
    np.full(N_PAD, 100e-6)])

K_LOWER = N_PAD + 2            # sheet node, lower plate
K_UPPER = N_PAD + 16           # sheet node, upper plate (14 cells above)


def _nu(dz):
    """(sim, grid, node_z) for a PEC-walled column on the given z profile."""
    dom = (400e-6, 400e-6, float(dz.sum()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(40e9, dom, dx=100e-6, dz_profile=dz, boundary="pec")
        grid = build_nonuniform_grid(40e9, dom, 100e-6, 0, dz, pec_faces=FACES)
    return sim, grid, np.asarray(coords_from_nonuniform_grid(grid).z)


def _slab(sim, lo, hi, material):
    sim.add(Box((WIDE[0], WIDE[0], lo), (WIDE[1], WIDE[1], hi)),
            material=material)


def _assemble(sim, grid, sheet_specs=None):
    mats, _, _, pec = assemble_materials_nu(sim, grid, sheet_specs=sheet_specs)
    return (np.asarray(mats.eps_r), np.asarray(mats.sigma),
            None if pec is None else np.asarray(pec))


def _stacked_pair(sim, z, eps_below=3.52, eps_above=3.38):
    """Two 17 um plates 14 cells apart, laminate ABUTTING every copper face.

    Nothing is drawn inside either 17 um copper slot — that is the stackup the
    defect needs, and drawing dielectric there is the retracted edit.
    """
    sim.add_material("cu", eps_r=1.0, sigma=5.8e7)
    sim.add_material("lo_d", eps_r=eps_below)
    sim.add_material("hi_d", eps_r=eps_above)
    za, zb = float(z[K_LOWER]), float(z[K_UPPER])
    mid = 0.5 * (za + zb)
    _slab(sim, float(z[N_PAD]), za - T_CU / 2, "lo_d")
    _slab(sim, za + T_CU / 2, mid, "hi_d")
    _slab(sim, mid, zb - T_CU / 2, "lo_d")
    _slab(sim, zb + T_CU / 2, float(z[N_PAD + N_STACK]), "hi_d")
    _slab(sim, za - T_CU / 2, za + T_CU / 2, "cu")
    _slab(sim, zb - T_CU / 2, zb + T_CU / 2, "cu")


def _series(eps, dz, k0, k1, i=2, j=2):
    """sum(d/eps_r) across cells [k0, k1) — the series-capacitance measure."""
    return float(sum(float(dz[k]) / float(eps[i, j, k]) for k in range(k0, k1)))


# --------------------------------------------------------------------------
# the defect itself
# --------------------------------------------------------------------------

def test_buried_sheet_node_carries_the_dielectric_at_its_live_edge():
    sim, grid, z = _nu(DZ_UNIFORM_STACK)
    _stacked_pair(sim, z)
    eps, _, pec = _assemble(sim, grid)
    for k in (K_LOWER, K_UPPER):
        assert bool(pec[2, 2, k]), f"node {k} should be a PEC sheet cell"
        assert float(eps[2, 2, k]) == pytest.approx(3.38, abs=1e-3), (
            f"node {k}: sheet cell kept eps_r {float(eps[2, 2, k]):.3f}; the "
            "live (sheet-normal) E edge sits half a cell ABOVE the node, "
            "inside the 3.38 laminate")
    # the whole plane, not just one column
    assert float(eps[:, :, K_LOWER].mean()) == pytest.approx(3.38, abs=1e-3)


def test_buried_cavity_series_sum_matches_the_physical_stack():
    """sum(d/eps) is the measure that sets the stacked-pair coupling."""
    sim, grid, z = _nu(DZ_UNIFORM_STACK)
    _stacked_pair(sim, z)
    eps, _, _ = _assemble(sim, grid)
    got = _series(eps, DZ_UNIFORM_STACK, K_LOWER, K_UPPER)
    # 7 cells of 3.38 + 7 of 3.52 — no vacuum cell anywhere in the cavity
    want = 7 * D_STACK / 3.38 + 7 * D_STACK / 3.52
    assert got == pytest.approx(want, rel=1e-4), (
        f"cavity sum(d/eps) = {got*1e6:.3f} um, want {want*1e6:.3f} um")
    # the pre-fix value, spelled out so a regression is recognizable
    vacuum_penalty = float(DZ_UNIFORM_STACK[K_LOWER]) * (1.0 - 1.0 / 3.38)
    assert vacuum_penalty == pytest.approx(22.13e-6, abs=0.02e-6)
    assert got + vacuum_penalty == pytest.approx(149.73e-6, abs=0.05e-6)
    assert not np.any(eps[2, 2, K_LOWER:K_UPPER] < 1.001), (
        "no cell of the cavity may read vacuum")


def test_sheet_node_sigma_follows_its_eps():
    """sigma feeds the same live edge, so it must move with eps_r."""
    sim, grid, z = _nu(DZ_UNIFORM_STACK)
    sim.add_material("cu", eps_r=1.0, sigma=5.8e7)
    sim.add_material("lossy", eps_r=3.38, sigma=0.05)
    zt = float(z[K_LOWER])
    _slab(sim, zt + T_CU / 2, float(z[N_PAD + N_STACK]), "lossy")
    _slab(sim, zt - T_CU / 2, zt + T_CU / 2, "cu")
    eps, sigma, _ = _assemble(sim, grid)
    assert float(eps[2, 2, K_LOWER]) == pytest.approx(3.38, abs=1e-3)
    assert float(sigma[2, 2, K_LOWER]) == pytest.approx(0.05, rel=1e-5)


def _offset_subcell_fill_case():
    """A 17 um fill 31.5..48.5 um above a sheet in a 100 um cell.

    The live edge is at +50 um — just ABOVE the fill, and offset far enough
    that the ``Box`` thin-branch snap is unambiguous rather than a tie.
    Shared with the traced-mesh test below, which needs the same geometry to
    show the window being skipped.
    """
    sim, grid, z = _nu(DZ_GRADED)
    k = N_PAD + 4                          # the 100 um cell
    node = float(z[k])
    sim.add_material("cu", eps_r=1.0, sigma=5.8e7)
    sim.add_material("fill", eps_r=3.52)
    sim.add_material("core", eps_r=3.38)
    _slab(sim, float(z[N_PAD]), node - T_CU / 2, "core")
    _slab(sim, node + T_CU / 2, float(z[N_PAD + 8]), "core")
    _slab(sim, node + 31.5e-6, node + 48.5e-6, "fill")   # sub-cell, offset
    _slab(sim, node - T_CU / 2, node + T_CU / 2, "cu")
    return sim, grid, k


def test_subcell_dielectric_fill_does_not_follow_the_shifted_sample():
    """A dielectric thinner than a cell must not chase the sample point.

    ``Box``'s thin branch gives a sub-cell shape the ONE node nearest its
    midpoint so it does not vaporize. Re-run against half-cell-shifted nodes,
    that branch re-snaps the shape onto whichever shifted node is now nearest
    — so a fill the live edge is NOT inside can still claim the cell.

    This is not hypothetical. On a real board two identical 17 um buried-level
    dielectric fills, both registered at their mid-plane with the live edge
    ~7 um above the fill's top face, disagreed: one re-snapped onto the shifted
    node and one did not, giving eps_r 3.520 at one buried level and 3.380 at
    the other from the same geometry and the same mesh. The two candidate
    shifted nodes are equidistant from a mid-plane-registered fill by
    construction, so float32 rounding was deciding a material value.

    The fixture below is the general form, offset so the snap is unambiguous
    rather than a tie: a 17 um fill sitting 31.5..48.5 um above the sheet in a
    100 um cell, with the live edge at +50 um — just above it.
    """
    sim, grid, k = _offset_subcell_fill_case()
    eps, _, pec = _assemble(sim, grid)
    assert bool(pec[2, 2, k])
    assert float(eps[2, 2, k]) == pytest.approx(3.38, abs=1e-3)


def test_subcell_dielectric_fill_IS_read_when_the_live_edge_is_inside_it():
    """The other direction: the window test admits it when the point is in it."""
    sim, grid, z = _nu(DZ_GRADED)
    k = N_PAD + 4                       # the 100 um cell
    node = float(z[k])
    sim.add_material("cu", eps_r=1.0, sigma=5.8e7)
    sim.add_material("fill", eps_r=3.52)
    sim.add_material("core", eps_r=3.38)
    _slab(sim, float(z[N_PAD]), node - T_CU / 2, "core")
    _slab(sim, node + 60e-6, float(z[N_PAD + 8]), "core")
    _slab(sim, node - T_CU / 2, node + 60e-6, "fill")   # 68.5 um < 100 um cell
    _slab(sim, node - T_CU / 2, node + T_CU / 2, "cu")
    eps, _, _ = _assemble(sim, grid)
    # live edge at node + 50 um, inside the sub-cell fill
    assert float(eps[2, 2, k]) == pytest.approx(3.52, abs=1e-3)


# --------------------------------------------------------------------------
# it must NOT invent dielectric — the retracted-fill trap
# --------------------------------------------------------------------------

def test_outer_sheet_with_air_above_stays_vacuum():
    """The outermost plate of a board radiates into air.

    Filling that cell with laminate is what this campaign had to retract: it
    puts dielectric where the designer's stackup has air. The resample reads
    what is actually at the live edge, so here it reads air.
    """
    sim, grid, z = _nu(DZ_UNIFORM_STACK)
    sim.add_material("cu", eps_r=1.0, sigma=5.8e7)
    sim.add_material("d", eps_r=3.38)
    zt = float(z[K_UPPER])
    _slab(sim, float(z[N_PAD]), zt - T_CU / 2, "d")     # laminate BELOW only
    _slab(sim, zt - T_CU / 2, zt + T_CU / 2, "cu")
    eps, _, pec = _assemble(sim, grid)
    assert bool(pec[2, 2, K_UPPER])
    assert float(eps[2, 2, K_UPPER]) == pytest.approx(1.0, abs=1e-6)


def test_outer_sheet_with_dielectric_above_takes_it():
    """Sign check: the live edge is at +d/2, never at -d/2."""
    sim, grid, z = _nu(DZ_UNIFORM_STACK)
    sim.add_material("cu", eps_r=1.0, sigma=5.8e7)
    sim.add_material("d", eps_r=3.38)
    zg = float(z[K_LOWER])
    _slab(sim, zg + T_CU / 2, float(z[N_PAD + N_STACK]), "d")   # ABOVE only
    _slab(sim, zg - T_CU / 2, zg + T_CU / 2, "cu")
    eps, _, _ = _assemble(sim, grid)
    assert float(eps[2, 2, K_LOWER]) == pytest.approx(3.38, abs=1e-3)


def test_sheet_in_vacuum_stays_vacuum():
    sim, grid, z = _nu(DZ_UNIFORM_STACK)
    sim.add_material("cu", eps_r=1.0, sigma=5.8e7)
    zt = float(z[K_LOWER])
    _slab(sim, zt - T_CU / 2, zt + T_CU / 2, "cu")
    eps, sigma, _ = _assemble(sim, grid)
    assert np.allclose(eps, 1.0)
    assert np.allclose(sigma, 0.0)


# --------------------------------------------------------------------------
# the half-step must be the LOCAL primal cell
# --------------------------------------------------------------------------

@pytest.mark.parametrize("k_sheet, expect", [(N_PAD + 2, 1.0),    # 20 um cell
                                             (N_PAD + 4, 3.38)])  # 100 um cell
def test_graded_profile_uses_the_local_half_cell(k_sheet, expect):
    """Same 20 um clearance above each plate, two very different cells.

    node + dz[k]/2 lands below the dielectric in the 20 um cell and above it
    in the 100 um one. A constant half-step, a neighbour's, or the profile
    min/max gets one of the two wrong.
    """
    sim, grid, z = _nu(DZ_GRADED)
    node = float(z[k_sheet])
    sim.add_material("cu", eps_r=1.0, sigma=5.8e7)
    sim.add_material("lo_d", eps_r=3.52)
    sim.add_material("hi_d", eps_r=3.38)
    _slab(sim, float(z[N_PAD]), node - T_CU / 2, "lo_d")
    _slab(sim, node + 20e-6, float(z[N_PAD + 8]), "hi_d")
    _slab(sim, node - T_CU / 2, node + T_CU / 2, "cu")
    eps, _, pec = _assemble(sim, grid)
    assert bool(pec[2, 2, k_sheet])
    assert float(eps[2, 2, k_sheet]) == pytest.approx(expect, abs=1e-3)


# a profile whose PAD cells are 137 um, so no z half-step anywhere in it
# equals the 50 um in-plane one — an axis-index slip cannot pass by luck
DZ_ODD_PAD = np.concatenate([
    np.full(N_PAD, 137e-6), np.full(N_STACK, D_STACK), np.full(N_PAD, 137e-6)])


@pytest.mark.parametrize("gap_um, expect", [(40.0, 3.38), (60.0, 1.0)])
def test_vertical_sheet_resamples_along_its_own_axis(gap_um, expect):
    """A sub-cell sheet whose normal is Y, on a grid whose dy differs from dz.

    Every other fixture here is a z-sheet, so nothing else catches an
    axis-index slip (using a z half-step for a y-normal sheet, or shifting the
    wrong coordinate). In-plane cells are 100 um, so the live edge sits at
    +50 um: a dielectric starting 40 um out is read, one starting 60 um out is
    not. The pair brackets the half-step, so any other axis' half-step — the
    31.43 um stack cells or the 137 um pads — gets one of the two wrong.
    """
    sim, grid, z = _nu(DZ_ODD_PAD)
    coords = coords_from_nonuniform_grid(grid)
    cy = float(np.asarray(coords.y)[2])
    sim.add_material("cu", eps_r=1.0, sigma=5.8e7)
    sim.add_material("d", eps_r=3.38)
    sim.add(Box((-1.0, cy + gap_um * 1e-6, float(z[N_PAD])),
                (1.0, 1.0, float(z[N_PAD + N_STACK]))), material="d")
    sim.add(Box((-1.0, cy - T_CU / 2, float(z[N_PAD])),
                (1.0, cy + T_CU / 2, float(z[N_PAD + N_STACK]))), material="cu")
    eps, _, pec = _assemble(sim, grid)
    k = K_LOWER + 3
    assert bool(pec[2, 2, k])
    assert float(eps[2, 2, k]) == pytest.approx(expect, abs=1e-3)


# --------------------------------------------------------------------------
# the classification must use the RUN's periodic flags
# --------------------------------------------------------------------------

# A z-periodic unit cell, 6 x 6 x 11 nodes at 100 um. The film is 4x4 and
# inset from the x/y faces, which is the configuration the seam behaviour was
# measured on: two 4x4 films on the z seam of that mask give per-axis live
# counts [0, 0, 32] under periodic=(F, F, F) and [0, 0, 0] under (F, F, T).
SEAM_DX = 100e-6
SEAM_NZ = 10                       # z cells -> 11 nodes
SEAM_LXY = 5 * SEAM_DX             # 6 nodes per in-plane axis
SEAM_LZ = SEAM_NZ * SEAM_DX
SEAM_CLEAR = 20e-6                 # laminate clearance above each film node


def _seam_case(lane, k_lo, k_hi):
    """A z-periodic column carrying one film on node ``k_lo`` and one on ``k_hi``.

    With ``k_lo, k_hi = 0, -1`` those two are the SAME film: metal sitting on
    the boundary of a z-periodic unit cell has to be entered at both faces,
    and through the seam the pair is one body two cells thick. With
    ``1, -2`` it is two genuinely separate node-thin sheets. The geometry is
    otherwise identical, so the two cases differ only in the classification.

    The laminate is drawn clear of BOTH film nodes and reaches to within
    ``SEAM_CLEAR`` (20 um) of them, so the node sample reads vacuum and a
    ``+d/2`` (50 um) resample reads 3.38 — the discriminator.

    Returns ``(sim, coords, eps_r, pec_mask, k_lo, k_hi)`` with the two
    node indices normalized to non-negative.
    """
    dom = (SEAM_LXY, SEAM_LXY, SEAM_LZ)
    spec = BoundarySpec(x="pec", y="pec", z="periodic")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if lane == "nu":
            dz = np.full(SEAM_NZ, SEAM_DX)
            sim = Simulation(40e9, dom, dx=SEAM_DX, dz_profile=dz,
                             boundary=spec)
            grid = build_nonuniform_grid(
                40e9, dom, SEAM_DX, 0, dz,
                pec_faces={"x_lo", "x_hi", "y_lo", "y_hi"})
            coords = coords_from_nonuniform_grid(grid)
        else:
            sim = Simulation(40e9, dom, dx=SEAM_DX, boundary=spec)
            grid = sim._build_grid()
            _cx, _cy, _cz = _grid_coords(grid)
            coords = GridCoords(x=_cx, y=_cy, z=_cz, shape=grid.shape)
    x = np.asarray(coords.x).ravel()
    y = np.asarray(coords.y).ravel()
    z = np.asarray(coords.z).ravel()
    k_lo, k_hi = k_lo % z.size, k_hi % z.size
    sim.add_material("cu", eps_r=1.0, sigma=5.8e7)
    sim.add_material("core", eps_r=3.38)
    z_lo, z_hi = float(z[k_lo]), float(z[k_hi])
    sim.add(Box((-1.0, -1.0, z_lo + SEAM_CLEAR), (1.0, 1.0, z_hi - SEAM_CLEAR)),
            material="core")
    fx = (float(x[1]) - 1e-9, float(x[4]) + 1e-9)     # 4x4, inset from the
    fy = (float(y[1]) - 1e-9, float(y[4]) + 1e-9)     # x/y faces
    for zc in (z_lo, z_hi):
        sim.add(Box((fx[0], fy[0], zc - T_CU / 2), (fx[1], fy[1], zc + T_CU / 2)),
                material="cu")
    if lane == "nu":
        mats, _, _, pec = assemble_materials_nu(sim, grid)
    else:
        mats, _, _, pec, _, _, _ = sim._assemble_materials(grid)
    return sim, coords, np.asarray(mats.eps_r), np.asarray(pec), k_lo, k_hi


def _run_periodic_flags(sim):
    """The run's flags, spelled WITHOUT ``periodic_flags_from_axes``.

    Deliberately an independent expression of the same fact: a test that
    asked the production helper could not tell a mis-threaded flag from a
    correctly threaded one.
    """
    return tuple(ax in (sim._periodic_axes or "") for ax in "xyz")


@pytest.mark.parametrize("lane", ["uniform", "nu"])
@pytest.mark.parametrize("k_lo, k_hi, expect", [
    (0, -1, 1.0),      # on the seam: one 2-cell body, no live normal edge
    (1, -2, 3.38),     # one cell inward: two node-thin sheets
])
def test_periodic_seam_film_is_classified_with_the_run_flags(lane, k_lo, k_hi,
                                                             expect):
    """A seam-straddling body is not two node-thin sheets.

    ``tangential_edge_masks`` keeps the wrap on a periodic axis (#689), so
    cell 0 and cell n-1 are neighbours there and a film entered on both faces
    of a z-periodic unit cell reads as ONE body two cells thick: its z
    component is tangential, ``apply_pec_mask`` zeroes it, and it has no live
    normal edge to serve. Move the same pair one cell inward and they are two
    separate sheets, each with a live normal edge, each resampled.

    Both call sites therefore have to hand the resample the RUN's flags. That
    threading was load-bearing but ungated: with the flags forced to
    ``(False, False, False)`` the seam pair reads as two sheets and each
    steals the laminate half a cell away, and the 22 tests that existed
    before this one all stayed green (the verifier's mutation M11,
    "22 passed (SURVIVES)"). Re-run verbatim against this file, with
    ``periodic_flags_from_axes`` edited to ``return (False, False, False)``::

        4 failed, 26 passed, 3 warnings in 19.25s
        FAILED test_periodic_seam_film_is_classified_with_the_run_flags[0--1-1.0-uniform]
        FAILED test_periodic_seam_film_is_classified_with_the_run_flags[0--1-1.0-nu]
        FAILED test_resampled_cells_stay_inside_the_operators_live_set[uniform]
        FAILED test_resampled_cells_stay_inside_the_operators_live_set[nu]
        E AssertionError: uniform lane, film nodes (0, 10) on a z-periodic run: eps_r = 3.380, want 1.000
        E AssertionError: nu lane, film nodes (0, 10) on a z-periodic run: eps_r = 3.380, want 1.000
        E AssertionError: uniform lane: 16 cell(s) resampled where the operator left nothing live, first at (1, 1, 0) (eps_r 3.380)
        E AssertionError: nu lane: 16 cell(s) resampled where the operator left nothing live, first at (1, 1, 0) (eps_r 3.380)

    Clean tree, same command: ``30 passed, 3 warnings in 18.88s``.

    The ``(1, -2)`` cases stayed green under the same mutation, which is what
    says the fixture discriminates the classification and not merely the
    presence of the resample.
    """
    sim, _coords, eps, pec, k_lo, k_hi = _seam_case(lane, k_lo, k_hi)
    assert bool(pec[2, 2, k_lo]), "the lower film must be a PEC cell"
    assert bool(pec[2, 2, k_hi]), "the upper film must be a PEC cell"
    assert float(eps[2, 2, k_lo]) == pytest.approx(expect, abs=1e-3), (
        f"{lane} lane, film nodes ({k_lo}, {k_hi}) on a z-periodic run: "
        f"eps_r = {float(eps[2, 2, k_lo]):.3f}, want {expect:.3f}")


@pytest.mark.parametrize("lane", ["uniform", "nu"])
def test_resampled_cells_stay_inside_the_operators_live_set(lane):
    """The invariant behind the fixture above, stated directly.

    A cell may only be resampled where the operator actually left a component
    live, because that is the only place the stored ``eps_r`` still feeds a
    field update. Comparing the assembled statics against the plain
    node-sampled ones names the cells the resample touched; the live set is
    recomputed here from the run's own flags (see ``_run_periodic_flags``).
    On the seam fixture the live set is empty, so the resample must be a
    no-op — under mutation M11 it is not.
    """
    sim, coords, eps, pec, _k_lo, _k_hi = _seam_case(lane, 0, -1)
    node_mats, _, _, _, _, _ = rasterize_geometry(
        sim._geometry, sim._resolve_material, coords,
        pec_sigma_threshold=sim._PEC_SIGMA_THRESHOLD)
    changed = eps != np.asarray(node_mats.eps_r)
    tang = tangential_edge_masks(jnp.asarray(pec), _run_periodic_flags(sim))
    live = np.zeros_like(changed)
    for t in tang:
        live |= np.asarray(jnp.asarray(pec) & ~t)
    assert not live.any(), (
        "the seam pair is one 2-cell body; no component of it is live")
    stray = [tuple(int(v) for v in row) for row in np.argwhere(changed & ~live)]
    assert not stray, (
        f"{lane} lane: {len(stray)} cell(s) resampled where the operator left "
        f"nothing live, first at {stray[0]} (eps_r {float(eps[stray[0]]):.3f})")


# --------------------------------------------------------------------------
# bodies that are NOT node-thin sheets keep the node sample
# --------------------------------------------------------------------------

def test_subcell_wire_thin_on_two_axes_is_left_alone():
    """A sub-cell wire keeps Ex AND Ey live, at two different half-cell offsets.

    One isotropic eps per cell cannot serve both, so such a cell keeps its node
    sample. The dielectric here starts a QUARTER of a cell to the +x side of
    the wire column, so the node sample reads vacuum and an x-resample would
    read 3.38 — the discriminator this test exists for.
    """
    sim, grid, z = _nu(DZ_UNIFORM_STACK)
    coords = coords_from_nonuniform_grid(grid)
    cx = float(np.asarray(coords.x)[2])
    cy = float(np.asarray(coords.y)[2])
    dx = 100e-6
    sim.add_material("cu", eps_r=1.0, sigma=5.8e7)
    sim.add_material("d", eps_r=3.38)
    # dielectric a quarter cell to the +x side AND a quarter cell to +y, so a
    # resample along EITHER in-plane axis reads 3.38 whichever fires last
    sim.add(Box((cx + 0.25 * dx, -1.0, float(z[N_PAD])),
                (1.0, 1.0, float(z[N_PAD + N_STACK]))), material="d")
    sim.add(Box((-1.0, cy + 0.25 * dx, float(z[N_PAD])),
                (1.0, 1.0, float(z[N_PAD + N_STACK]))), material="d")
    sim.add(Box((cx - 5e-6, cy - 5e-6, float(z[K_LOWER])),
                (cx + 5e-6, cy + 5e-6, float(z[K_UPPER]))), material="cu")
    eps, _, pec = _assemble(sim, grid)
    k = K_LOWER + 3
    assert bool(pec[2, 2, k])
    assert float(eps[2, 2, k]) == pytest.approx(1.0, abs=1e-6)
    # the dielectric IS there half a cell away, so the fixture is live
    assert float(eps[3, 2, k]) == pytest.approx(3.38, abs=1e-3)
    assert float(eps[2, 3, k]) == pytest.approx(3.38, abs=1e-3)


def test_subcell_wire_does_not_pull_material_from_half_a_cell_away():
    """Same guard along z: a wire is not a z-sheet."""
    sim, grid, z = _nu(DZ_UNIFORM_STACK)
    coords = coords_from_nonuniform_grid(grid)
    cx = float(np.asarray(coords.x)[2])
    cy = float(np.asarray(coords.y)[2])
    sim.add_material("cu", eps_r=1.0, sigma=5.8e7)
    sim.add_material("d", eps_r=3.38)
    _slab(sim, float(z[K_LOWER]) + 0.4 * D_STACK,
          float(z[N_PAD + N_STACK]), "d")
    sim.add(Box((cx - 5e-6, cy - 5e-6, float(z[K_LOWER])),
                (cx + 5e-6, cy + 5e-6, float(z[K_UPPER]))), material="cu")
    eps, _, _ = _assemble(sim, grid)
    assert float(eps[2, 2, K_LOWER]) == pytest.approx(1.0, abs=1e-6)


def test_volumetric_pec_block_is_left_alone():
    sim, grid, z = _nu(DZ_UNIFORM_STACK)
    sim.add_material("cu", eps_r=1.0, sigma=5.8e7)
    sim.add_material("d", eps_r=3.38)
    _slab(sim, float(z[K_LOWER]) + 0.4 * D_STACK,
          float(z[N_PAD + N_STACK]), "d")
    _slab(sim, float(z[K_LOWER]), float(z[K_LOWER + 3]), "cu")
    eps, _, pec = _assemble(sim, grid)
    assert bool(pec[2, 2, K_LOWER])
    assert float(eps[2, 2, K_LOWER]) == pytest.approx(1.0, abs=1e-6), (
        "a >=2-cell PEC volume has no live edge to serve; its cells keep the "
        "node sample")


# --------------------------------------------------------------------------
# every node-thin conductor path, not just geometry PEC
# --------------------------------------------------------------------------

def _sheet_over_laminate(sim, z, k, **tc_kwargs):
    sim.add_material("lo_d", eps_r=3.52)
    sim.add_material("hi_d", eps_r=3.38)
    zt = float(z[k])
    _slab(sim, float(z[N_PAD]), zt - T_CU / 2, "lo_d")
    _slab(sim, zt + T_CU / 2, float(z[N_PAD + N_STACK]), "hi_d")
    sim.add_thin_conductor(
        Box((WIDE[0], WIDE[0], zt - T_CU / 2), (WIDE[1], WIDE[1], zt + T_CU / 2)),
        **tc_kwargs)


def test_thin_conductor_pec_sheet_node_takes_the_dielectric():
    sim, grid, z = _nu(DZ_UNIFORM_STACK)
    _sheet_over_laminate(sim, z, K_LOWER, sigma_bulk=5.8e7, thickness=T_CU)
    eps, _, pec = _assemble(sim, grid)
    assert bool(pec[2, 2, K_LOWER])
    assert float(eps[2, 2, K_LOWER]) == pytest.approx(3.38, abs=1e-3)


def test_surface_impedance_sheet_node_takes_the_dielectric():
    """The f0 sheet is not in pec_mask; it declares its own normal axis."""
    sim, grid, z = _nu(DZ_UNIFORM_STACK)
    _sheet_over_laminate(sim, z, K_LOWER, sigma_bulk=5.8e7, thickness=T_CU,
                         surface_impedance_f0=28e9)
    specs = []
    eps, _, pec = _assemble(sim, grid, sheet_specs=specs)
    assert len(specs) == 1 and specs[0].normal_axis == 2
    assert pec is None
    assert float(eps[2, 2, K_LOWER]) == pytest.approx(3.38, abs=1e-3)


def test_legacy_dc_fold_still_writes_its_own_eps():
    """The DC fold is VOLUMETRIC — it stamps ``conductor.eps_r`` on its cell.

    That path is not a node-thin surface and this fix must not silently
    change it; whether its default of 1.0 should keep erasing the surrounding
    dielectric is a separate decision with its own root cause.
    """
    sim, grid, z = _nu(DZ_UNIFORM_STACK)
    sim.add_material("d", eps_r=3.38)
    zt = float(z[K_LOWER])
    _slab(sim, float(z[N_PAD]), float(z[N_PAD + N_STACK]), "d")
    sim.add_thin_conductor(
        Box((WIDE[0], WIDE[0], zt - T_CU / 2), (WIDE[1], WIDE[1], zt + T_CU / 2)),
        sigma_bulk=1e3, thickness=T_CU)
    eps, sigma, _ = _assemble(sim, grid)
    assert float(eps[2, 2, K_LOWER]) == pytest.approx(1.0, abs=1e-6)
    assert float(sigma[2, 2, K_LOWER]) > 0.0


# --------------------------------------------------------------------------
# the two lanes must not diverge (this rule has been hand-ported before)
# --------------------------------------------------------------------------

def test_uniform_lane_agrees_with_the_nonuniform_lane():
    dx = 100e-6
    lz = 4000e-6
    sim = Simulation(40e9, (400e-6, 400e-6, lz), dx=dx, boundary="pec")
    sim.add_material("cu", eps_r=1.0, sigma=5.8e7)
    sim.add_material("lo_d", eps_r=3.52)
    sim.add_material("hi_d", eps_r=3.38)
    zs = 20 * dx
    _slab(sim, 0.0, zs - T_CU / 2, "lo_d")
    _slab(sim, zs + T_CU / 2, lz, "hi_d")
    _slab(sim, zs - T_CU / 2, zs + T_CU / 2, "cu")
    grid = sim._build_grid()
    mats, _, _, pec, _, _, _ = sim._assemble_materials(grid)
    k = 20 + grid.axis_pads[2]
    assert bool(np.asarray(pec)[2, 2, k])
    assert float(np.asarray(mats.eps_r)[2, 2, k]) == pytest.approx(3.38, abs=1e-3)


def test_uniform_lane_outer_sheet_with_air_above_stays_vacuum():
    dx = 100e-6
    lz = 4000e-6
    sim = Simulation(40e9, (400e-6, 400e-6, lz), dx=dx, boundary="pec")
    sim.add_material("cu", eps_r=1.0, sigma=5.8e7)
    sim.add_material("d", eps_r=3.38)
    zs = 20 * dx
    _slab(sim, 0.0, zs - T_CU / 2, "d")
    _slab(sim, zs - T_CU / 2, zs + T_CU / 2, "cu")
    grid = sim._build_grid()
    mats, _, _, _, _, _, _ = sim._assemble_materials(grid)
    k = 20 + grid.axis_pads[2]
    assert float(np.asarray(mats.eps_r)[2, 2, k]) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# documented scope limits
# --------------------------------------------------------------------------

def test_dispersive_neighbour_moves_eps_inf_but_not_the_pole():
    """Scope limit, pinned so it cannot change silently.

    A sheet node whose live edge lands in a Debye material takes that
    material's ``eps_r`` (its eps_inf) but NOT its pole: relocating a resonant
    pole mask is the change #627b measured turning a stable run divergent, so
    it needs its own stability argument.
    """
    from rfx.materials.debye import DebyePole
    sim, grid, z = _nu(DZ_UNIFORM_STACK)
    sim.add_material("cu", eps_r=1.0, sigma=5.8e7)
    sim.add_material("dby", eps_r=3.38,
                     debye_poles=[DebyePole(delta_eps=2.0, tau=1e-11)])
    zt = float(z[K_LOWER])
    _slab(sim, zt + T_CU / 2, float(z[N_PAD + N_STACK]), "dby")
    _slab(sim, zt - T_CU / 2, zt + T_CU / 2, "cu")
    mats, debye_spec, _, _ = assemble_materials_nu(sim, grid)
    eps = np.asarray(mats.eps_r)
    assert float(eps[2, 2, K_LOWER]) == pytest.approx(3.38, abs=1e-3)
    assert debye_spec is not None
    _poles, masks = debye_spec
    assert not bool(np.asarray(masks)[..., 2, 2, K_LOWER].any()), (
        "the pole mask must stay at its node sampling")


def test_tracer_path_takes_every_axis_and_agrees_with_the_eager_one():
    """Under jit the "does this axis have a sheet?" predicate is a tracer.

    The eager lane skips an axis with no sheet to avoid an extra rasterization
    pass; the traced lane cannot, so it takes all three. Both must produce the
    same arrays — the skip is an optimization, never a behaviour switch.
    """
    import jax
    from rfx.geometry.rasterize_grid import resample_sheet_node_materials

    sim, grid, z = _nu(DZ_UNIFORM_STACK)
    _stacked_pair(sim, z)
    coords = coords_from_nonuniform_grid(grid)
    eps0 = jnp.ones(coords.shape, dtype=jnp.float32)
    sig0 = jnp.zeros(coords.shape, dtype=jnp.float32)
    mats, _, _, pec = assemble_materials_nu(sim, grid)
    half = (jnp.asarray(grid.dx_arr) * 0.5, jnp.asarray(grid.dy_arr) * 0.5,
            jnp.asarray(grid.dz) * 0.5)

    def go(cond_mask):
        return resample_sheet_node_materials(
            sim._geometry, sim._resolve_material, coords, eps0, sig0,
            half_steps=half, conductor_cell_mask=cond_mask,
            pec_sigma_threshold=sim._PEC_SIGMA_THRESHOLD)

    eager_eps, eager_sigma = go(pec)
    traced_eps, traced_sigma = jax.jit(go)(pec)
    assert np.allclose(np.asarray(eager_eps), np.asarray(traced_eps))
    assert np.allclose(np.asarray(eager_sigma), np.asarray(traced_sigma))
    assert float(np.asarray(traced_eps)[2, 2, K_LOWER]) == pytest.approx(
        3.38, abs=1e-3)


def test_f0_sheet_coefficients_are_in_the_resistive_limit():
    """Scope correction: an f0 sheet's TANGENTIAL edges are live too.

    The resample is described as serving "the one component a node-thin sheet
    leaves live". For a hard-PEC sheet that is literal — ``apply_pec_mask``
    zeroes the other two. For a ``surface_impedance_f0`` sheet it is NOT:
    ``apply_sheet_impedance_e`` resistively updates the tangential edges, and
    ``sheet_update_coeffs`` reads the SAME ``materials.eps_r``. So on that
    path the resample moves the tangential coefficients as well, and the
    premise as first written was false.

    It moves no number at any real metal, and this test pins the reason
    rather than the claim. ``x2 = sigma_tot*dt/(eps0*eps_r)`` is thousands
    for a metal sheet, and both coefficients saturate there: ``A -> 0`` and
    ``B -> 1/sigma_tot``, which is the resistive-sheet limit ``E_tan =
    Rs*Js`` and carries no eps at all. The eps-independence is that limit,
    not a general property — the second half of this test shows the
    coefficients separating cleanly at a sheet three orders of magnitude
    more resistive than copper foil.
    """
    from rfx.core.yee import MaterialArrays
    from rfx.materials.thin_conductor import sheet_update_coeffs

    c0, mu0, eps0 = 299792458.0, 4e-7 * np.pi, 8.8541878128e-12
    dt = 0.99 * D_STACK / (c0 * np.sqrt(3.0))          # 5.992e-14 s
    sigma_cu, f0 = 5.8e7, 28e9
    skin = np.sqrt(1.0 / (np.pi * f0 * mu0 * sigma_cu))
    sigma_sheet = 1.0 / ((1.0 / (sigma_cu * skin)) * D_STACK)
    assert sigma_sheet == pytest.approx(7.288e5, rel=1e-3)

    def coeffs(eps_r, ss):
        mats = MaterialArrays(eps_r=jnp.asarray(eps_r), sigma=jnp.asarray(0.0),
                              mu_r=jnp.asarray(1.0))
        a, b = sheet_update_coeffs(jnp.asarray(ss), mats, dt)
        return float(a), float(b)

    a1, b1 = coeffs(1.0, sigma_sheet)       # what the sheet node used to read
    a2, b2 = coeffs(3.38, sigma_sheet)      # what the resample gives it
    assert (a1, b1) == (a2, b2), (
        f"copper at {f0/1e9:.0f} GHz: eps_r 1.0 -> (A={a1:.6e}, B={b1:.6e}) "
        f"but eps_r 3.38 -> (A={a2:.6e}, B={b2:.6e})")
    assert a1 == 0.0
    assert b1 == pytest.approx(1.0 / sigma_sheet, rel=1e-6)
    assert b1 == pytest.approx(1.372111e-06, rel=1e-4)
    for eps_r in (1.0, 3.38):
        assert sigma_sheet * dt / (eps0 * eps_r) > 1e3, "not in the limit"

    # the converse, so the limit is what carries the claim
    a_lo, b_lo = coeffs(1.0, 1e3)
    a_hi, b_hi = coeffs(3.38, 1e3)
    assert a_lo != a_hi and b_lo != b_hi, (
        "at sigma_sheet = 1e3 S/m the coefficients must still see eps_r")


def test_traced_mesh_skips_the_subcell_window():
    """Documented divergence: the sub-cell window needs a CONCRETE mesh.

    ``_subcell_box_axis_window`` has to decide "is this Box thinner than its
    local cell", which needs real node coordinates and a real half-step, so
    it returns ``None`` as soon as either is a tracer — the differentiable
    mesh lane (a traced ``dz_profile``, or any jit taking the half-steps as
    arguments rather than closing over them). The resample then falls back to
    the plain shifted mask, i.e. to exactly the ``Box`` thin-branch re-snap
    that window exists to prevent.

    On the fixture above that is a visible difference: eager ``eps_r`` 3.38,
    traced 3.52 — for the combination (sub-cell dielectric Box + node-thin
    conductor + traced mesh) the traced primal is a slightly different model
    from the eager one, and nothing reports it at run time. Narrow, but
    silent, which is why it is written down here rather than left to be
    found as "the gradient is for a different model".

    This test pins the fallback, not its physics. If it ever reds because
    the traced path now AGREES, the divergence was fixed — update
    ``_subcell_box_axis_window``'s docstring rather than the assertion.
    """
    import jax
    from rfx.geometry.rasterize_grid import (
        _subcell_box_axis_window, resample_sheet_node_materials)

    sim, grid, k = _offset_subcell_fill_case()
    coords = coords_from_nonuniform_grid(grid)
    _mats, _, _, pec = assemble_materials_nu(sim, grid)
    eps0 = jnp.ones(coords.shape, dtype=jnp.float32)
    sig0 = jnp.zeros(coords.shape, dtype=jnp.float32)
    half = (jnp.asarray(grid.dx_arr) * 0.5, jnp.asarray(grid.dy_arr) * 0.5,
            jnp.asarray(grid.dz) * 0.5)

    def go(half_steps):
        return resample_sheet_node_materials(
            sim._geometry, sim._resolve_material, coords, eps0, sig0,
            half_steps=half_steps, conductor_cell_mask=pec,
            pec_sigma_threshold=sim._PEC_SIGMA_THRESHOLD)

    eager = float(np.asarray(go(half)[0])[2, 2, k])
    traced = float(np.asarray(jax.jit(go)(half)[0])[2, 2, k])
    assert eager == pytest.approx(3.38, abs=1e-3)
    assert traced != pytest.approx(eager, abs=1e-3), (
        "the traced mesh lane no longer diverges here — see the docstring")

    # the mechanism itself, so a future reader is not left inferring it
    fill = [e.shape for e in sim._geometry if e.material_name == "fill"][0]
    assert _subcell_box_axis_window(fill, 2, coords.z, half[2]) is not None
    seen = []

    def _probe(half_steps):
        seen.append(
            _subcell_box_axis_window(fill, 2, coords.z, half_steps[2]))
        return jnp.zeros(())

    jax.jit(_probe)(half)
    assert seen == [None], f"window under trace: {seen}"
