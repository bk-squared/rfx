"""Surface-impedance (Leontovich) sheets on NON-Box shapes (issue #674).

#669 scoped ``add_thin_conductor(surface_impedance_f0=...)`` to ``Box`` and said
so loudly. That was a SCOPING decision, not physics: the fold
``sigma_eff = 1/(Rs0*d_norm)`` is per occupied cell, and thin-conductor
rasterisation has been coords-based on both lanes since #369. The sheets that
dominate conductor loss on a printed board are patterned — ground planes with
clearance holes, meandered arms, CAD outlines — so the restriction excluded
exactly the geometry the feature exists for.

What this module pins:

O674-1  a Box and an INDEPENDENTLY implemented mask shape covering the same
        cells fold BIT-IDENTICALLY (SHA-256 over the assembled sigma), on the
        uniform lane and on a graded NU lane with the sheet ON a grading step
        (where the dual-spacing normalization is load-bearing);
O674-2  a patterned sheet (plate MINUS a clearance hole) folds on the occupied
        cells only — asserted per cell, with the hole proven non-empty so the
        gate cannot pass vacuously;
O674-3  the two failure modes the Box guard was standing in for still fail
        LOUD on both lanes: a body with HEIGHT (>1 rasterized layer along its
        normal) and a sheet that rasterizes to ZERO cells (the #369
        silently-vaporized-metal class, which a sub-cell mesh slab reaches by
        simply missing every node plane);
O674-4  the ``thin_conductor_graded_node`` advisory follows a non-Box sheet —
        including a patterned one whose bounding-box centre lands in its hole,
        which is where a single-point probe would silently read "no sheet";
O674-5  the design-IR contract: a registered non-Box primitive round-trips, an
        unregistered shape class is refused loudly by the EXISTING shape codec;
O674-6  the real CAD path: an imported ``MeshShape`` slab folds bit-identically
        to the Box it stands for, and a four-bar frame (a plane with a
        clearance opening, as CAD delivers one) leaves its opening alone;
O674-7  the new occupancy guard reads CONCRETE masks only, so the
        differentiable-mesh path still differentiates;
O674-8  the batched (``vmap_sweep``) material build folds a non-Box sheet onto
        its slices exactly as the serial build does.

The #671 transition-node FDTD oracle re-run with a non-Box sheet lives in
``test_alpha_invariance_transfers_to_a_nonbox_sheet`` (slow_physics).
"""

import hashlib
import warnings

import numpy as np
import pytest
import jax.numpy as jnp

from rfx import Box, Simulation
from rfx.geometry.csg import Cylinder, Sphere
from rfx.materials.thin_conductor import ThinConductor, leontovich_rs
from rfx.runners.nonuniform import assemble_materials_nu, build_nonuniform_grid

from tests.test_thin_conductor_nu_dual_spacing import INVARIANCE_CASES

F0 = 10e9
SIGMA_BULK = 1e4
THICKNESS = 35e-6


# ---------------------------------------------------------------------------
# Test shapes: a planar sheet with an optional rectangular hole, implemented
# from the Shape protocol (mask_on_coords + bounding_box) WITHOUT deriving
# from Box — the point of O674-1 is that two independent occupancy
# implementations reach the same folded sigma, not that Box calls itself.
# ---------------------------------------------------------------------------

class PlanarSheet:
    """Flat sheet on ``axis = coord``, footprint ``[lo, hi)``, optional hole.

    Conventions match the primitives deliberately: half-open ``[lo, hi)`` on
    the in-plane axes (:class:`rfx.geometry.csg.Box`'s volume rule) and the
    single nearest node on the normal axis (Box's thin-sheet rule), so an
    equivalent Box and this shape must rasterize to the same cells.
    """

    def __init__(self, axis, coord, plane_lo, plane_hi, hole=None):
        self.axis = int(axis)
        self.coord = float(coord)
        self.plane_lo = tuple(float(v) for v in plane_lo)   # (a0, a1) lows
        self.plane_hi = tuple(float(v) for v in plane_hi)
        self.hole = None if hole is None else (
            tuple(float(v) for v in hole[0]), tuple(float(v) for v in hole[1]))

    @property
    def _plane_axes(self):
        return tuple(a for a in range(3) if a != self.axis)

    def bounding_box(self):
        lo, hi = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        lo[self.axis] = hi[self.axis] = self.coord
        for i, a in enumerate(self._plane_axes):
            lo[a], hi[a] = self.plane_lo[i], self.plane_hi[i]
        return tuple(lo), tuple(hi)

    def mask_on_coords(self, x, y, z):
        # Comparisons run in the COORDINATE dtype, exactly as the primitives
        # do, so a node sitting on a knife edge lands on the same side for
        # both: what this module tests is the FOLD, not float32 boundary
        # conventions (which the Box docstring covers at length).
        coords = [jnp.asarray(c).ravel() for c in (x, y, z)]
        per_axis = []
        for a in range(3):
            c = coords[a]
            if a == self.axis:
                m = jnp.zeros(c.shape, dtype=bool).at[
                    jnp.argmin(jnp.abs(c - self.coord))].set(True)
            else:
                i = self._plane_axes.index(a)
                m = (c >= self.plane_lo[i]) & (c < self.plane_hi[i])
            per_axis.append(m)
        out = (per_axis[0][:, None, None] & per_axis[1][None, :, None]
               & per_axis[2][None, None, :])
        if self.hole is not None:
            holes = []
            for a in range(3):
                if a == self.axis:
                    holes.append(jnp.ones(coords[a].shape, dtype=bool))
                else:
                    i = self._plane_axes.index(a)
                    holes.append((coords[a] >= self.hole[0][i])
                                 & (coords[a] < self.hole[1][i]))
            out = out & ~(holes[0][:, None, None] & holes[1][None, :, None]
                          & holes[2][None, None, :])
        return out

    def mask(self, grid):
        from rfx.geometry.csg import _grid_coords
        return self.mask_on_coords(*_grid_coords(grid))


class MaskOnlyShape:
    """A shape that can rasterize but cannot say where its normal is."""

    def __init__(self, inner):
        self._inner = inner

    def mask_on_coords(self, x, y, z):
        return self._inner.mask_on_coords(x, y, z)

    def mask(self, grid):
        return self._inner.mask(grid)


class BoundsOnlyShape:
    """A shape with a bounding box and no way to rasterize itself."""

    def __init__(self, lo, hi):
        self._lo, self._hi = tuple(lo), tuple(hi)

    def bounding_box(self):
        return self._lo, self._hi


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

def _sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


# uniform fixture: 20x20x3 mm at dx = 1 mm, sheet on the z = 1 mm node plane
U_DX = 1e-3
U_DOMAIN = (0.02, 0.02, 0.003)
U_Z = 1e-3
U_FOOT = ((5e-3, 5e-3), (15e-3, 15e-3))          # [lo, hi) in x and y
U_HOLE = ((8e-3, 8e-3), (12e-3, 12e-3))

# NU fixture: dz = [0.5 mm]x8 + [1.5 mm]x8, sheet ON the 4.0 mm step node
NU_DX = 0.5e-3
NU_DZ = [0.5e-3] * 8 + [1.5e-3] * 8
NU_L = 24 * NU_DX
NU_Z = 4.0e-3
NU_FOOT = ((6 * NU_DX, 6 * NU_DX), (18 * NU_DX, 18 * NU_DX))
NU_HOLE = ((10 * NU_DX, 10 * NU_DX), (14 * NU_DX, 14 * NU_DX))


def _uniform_sigma(shape, **kw):
    sim = Simulation(freq_max=10e9, domain=U_DOMAIN, dx=U_DX)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.add_thin_conductor(shape, sigma_bulk=SIGMA_BULK,
                               thickness=THICKNESS, **kw)
    grid = sim._build_grid()
    mats, _, _, pec_mask, *_ = sim._assemble_materials(grid)
    return np.asarray(mats.sigma), pec_mask, grid, sim


def _nu_grid(sim):
    return build_nonuniform_grid(
        sim._freq_max, sim._domain, sim._dx, sim._cpml_layers,
        sim._dz_profile, dx_profile=sim._dx_profile,
        dy_profile=sim._dy_profile,
        pec_faces=sim._boundary_spec.pec_faces(),
        pmc_faces=sim._boundary_spec.pmc_faces(),
        cpml_axes="xyz")


def _nu_sigma(shape, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=10e9, domain=(NU_L, NU_L, 0),
                         dx=NU_DX, dz_profile=NU_DZ, boundary="cpml",
                         cpml_layers=6)
        sim.add_thin_conductor(shape, sigma_bulk=SIGMA_BULK,
                               thickness=THICKNESS, **kw)
        grid = _nu_grid(sim)
        mats, _, _, pec_mask = assemble_materials_nu(sim, grid)
    return np.asarray(mats.sigma), pec_mask, grid, sim


def _box_sheet(z, foot):
    (x0, y0), (x1, y1) = foot
    return Box((x0, y0, z), (x1, y1, z))


def _planar_sheet(z, foot, hole=None):
    (x0, y0), (x1, y1) = foot
    return PlanarSheet(2, z, (x0, y0), (x1, y1), hole=hole)


# ---------------------------------------------------------------------------
# O674-1: Box vs an equivalent mask shape — bit-identical fold
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lane", ["uniform", "nonuniform"])
def test_box_and_equivalent_mask_shape_fold_bit_identically(lane):
    """Same occupancy, different shape class -> identical sigma digest.

    The NU leg puts the sheet on the 0.5/1.5 mm grading step, so the fold's
    dual-spacing normalization (#671) is what is being reproduced, not a
    uniform-mesh coincidence.
    """
    if lane == "uniform":
        z, foot, cell, run = U_Z, U_FOOT, U_DX, _uniform_sigma
    else:
        z, foot, cell, run = NU_Z, NU_FOOT, NU_DX, _nu_sigma

    kw = dict(surface_impedance_f0=F0)
    sig_box, pec_box, grid, _ = run(_box_sheet(z, foot), **kw)
    sig_msk, pec_msk, _, _ = run(_planar_sheet(z, foot), **kw)

    # the two occupancies must agree cell-for-cell first: if they do not, the
    # digest below would be reporting a geometry difference, not a fold one
    n_box = int((sig_box > 0).sum())
    assert n_box > 0, "fixture folded no cells"
    np.testing.assert_array_equal(sig_box > 0, sig_msk > 0)
    assert _sha(sig_box) == _sha(sig_msk), (
        f"{lane}: non-Box sheet folded a different sigma "
        f"(max {sig_box.max():.6g} vs {sig_msk.max():.6g})")

    # neither contributes PEC bits, and the realized sheet is the requested one
    for pec in (pec_box, pec_msk):
        assert pec is None or int(np.asarray(pec).sum()) == 0
    rs0 = float(leontovich_rs(F0, SIGMA_BULK))
    if lane == "uniform":
        d_norm = np.full(1, grid.dx)
        ks = [0]
    else:
        from rfx.nonuniform import e_node_dual_spacings
        ks = sorted({int(k) for k in np.argwhere(sig_msk > 0)[:, 2]})
        assert len(ks) == 1, ks
        d_norm = np.asarray(e_node_dual_spacings(grid.dz))[ks]
        primal = np.asarray(grid.dz)[ks[0]]
        assert abs(primal / d_norm[0] - 1.5) < 1e-3, (primal, d_norm)
    prod = float(sig_msk.max()) * rs0 * float(d_norm[0])
    assert abs(prod - 1.0) < 1e-5, f"{lane}: sigma_eff*Rs0*d_norm = {prod}"

    # negative control: a digest-equality gate that cannot fail is not a gate.
    # One cell of footprint shift must move it.
    (fx, fy), hi_corner = foot
    shifted = _planar_sheet(z, ((fx + cell, fy), hi_corner))
    sig_shift, _, _, _ = run(shifted, **kw)
    assert _sha(sig_shift) != _sha(sig_box), (
        f"{lane}: shifting the sheet one cell did not change the digest")
    assert int((sig_shift > 0).sum()) < n_box


def test_dc_fold_also_accepts_a_mask_shape_on_the_uniform_lane():
    """The legacy DC fold was never Box-only on the uniform lane (it reads
    ``shape.mask``); pin that #674 did not change it. The NU DC path keeps its
    documented warn-and-skip for non-Box shapes."""
    sig_box, _, _, _ = _uniform_sigma(_box_sheet(U_Z, U_FOOT))
    sig_msk, _, _, _ = _uniform_sigma(_planar_sheet(U_Z, U_FOOT))
    assert _sha(sig_box) == _sha(sig_msk)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sim = Simulation(freq_max=10e9, domain=(NU_L, NU_L, 0), dx=NU_DX,
                         dz_profile=NU_DZ, boundary="cpml", cpml_layers=6)
        sim.add_thin_conductor(_planar_sheet(NU_Z, NU_FOOT),
                               sigma_bulk=SIGMA_BULK, thickness=THICKNESS)
        mats_nu, *_ = assemble_materials_nu(sim, _nu_grid(sim))
    assert any("non-Box shape is not yet supported" in str(w.message)
               for w in rec), [str(w.message) for w in rec]
    assert float(np.asarray(mats_nu.sigma).max()) == 0.0


# ---------------------------------------------------------------------------
# O674-2: patterned sheet — the hole stays untouched
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lane", ["uniform", "nonuniform"])
def test_patterned_sheet_folds_only_occupied_cells(lane):
    if lane == "uniform":
        z, foot, hole, run = U_Z, U_FOOT, U_HOLE, _uniform_sigma
    else:
        z, foot, hole, run = NU_Z, NU_FOOT, NU_HOLE, _nu_sigma

    solid = _planar_sheet(z, foot)
    holed = _planar_sheet(z, foot, hole=hole)
    sig_solid, _, grid, _ = run(solid, surface_impedance_f0=F0)
    sig_holed, _, _, _ = run(holed, surface_impedance_f0=F0)

    if lane == "uniform":
        m_solid = np.asarray(solid.mask(grid))
        m_holed = np.asarray(holed.mask(grid))
    else:
        from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid
        coords = coords_from_nonuniform_grid(grid)
        m_solid = np.asarray(solid.mask_on_coords(
            coords.x, coords.y, coords.z))
        m_holed = np.asarray(holed.mask_on_coords(
            coords.x, coords.y, coords.z))

    n_hole = int((m_solid & ~m_holed).sum())
    assert n_hole > 0, "clearance hole rasterized to nothing — gate is blind"
    assert int(m_holed.sum()) > 0

    # per cell: folded exactly on the occupied set, and NOT in the hole
    np.testing.assert_array_equal(sig_holed > 0, m_holed)
    assert float(sig_holed[m_solid & ~m_holed].max(initial=0.0)) == 0.0, (
        "clearance-hole cells carry sheet conductivity")
    # the cells that remain carry exactly the same sigma as in the solid sheet
    np.testing.assert_array_equal(sig_holed[m_holed], sig_solid[m_holed])
    # ... and the background is untouched
    assert float(sig_holed[~m_holed].max(initial=0.0)) == 0.0


# ---------------------------------------------------------------------------
# O674-3: what still fails loud
# ---------------------------------------------------------------------------

def test_body_with_height_refused_on_both_lanes():
    """A 3-D shape is not a sheet: it rasterizes to more than one layer along
    its normal, and folding it per cell would multiply the sheet conductance
    by the layer count. Refused at build time on both lanes."""
    ball = Sphere((10e-3, 10e-3, 1.5e-3), 1.2e-3)
    with pytest.raises(ValueError, match="cell layers along its normal"):
        _uniform_sigma(ball, surface_impedance_f0=F0)
    ball_nu = Sphere((6e-3, 6e-3, 4.0e-3), 1.2e-3)
    with pytest.raises(ValueError, match="cell layers along its normal"):
        _nu_sigma(ball_nu, surface_impedance_f0=F0)


def test_sheet_that_rasterizes_to_nothing_is_refused_not_vaporized():
    """The #369 class, reachable by a non-Box sheet a Box could not reach: a
    footprint that falls entirely between node planes folds zero cells. It
    must raise, never silently vanish."""
    # footprint strictly inside one cell in x: [5.2, 5.8) mm on a 1 mm grid
    ghost = PlanarSheet(2, U_Z, (5.2e-3, 5e-3), (5.8e-3, 15e-3))
    with pytest.raises(ValueError, match="ZERO cells"):
        _uniform_sigma(ghost, surface_impedance_f0=F0)


def test_shape_without_a_mask_or_bounds_is_refused_at_add_time():
    sim = Simulation(freq_max=10e9, domain=U_DOMAIN, dx=U_DX)
    with pytest.raises(ValueError, match="mask_on_coords"):
        sim.add_thin_conductor(
            BoundsOnlyShape((0, 0, U_Z), (1e-3, 1e-3, U_Z)),
            surface_impedance_f0=F0)
    with pytest.raises(ValueError, match="bounding box"):
        sim.add_thin_conductor(
            MaskOnlyShape(_planar_sheet(U_Z, U_FOOT)),
            surface_impedance_f0=F0)


def test_nu_defensive_refusal_for_a_bounds_less_sheet():
    """Built outside ``add_thin_conductor`` (which is where the add-time check
    lives), a bounds-less f0 sheet must still fail loud on the NU lane rather
    than inherit the DC path's warn-and-skip."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=10e9, domain=(NU_L, NU_L, 0), dx=NU_DX,
                         dz_profile=NU_DZ, boundary="cpml", cpml_layers=6)
        sim.add_thin_conductor(_box_sheet(NU_Z, NU_FOOT),
                               sigma_bulk=SIGMA_BULK, thickness=THICKNESS)
        grid = _nu_grid(sim)
    sim._thin_conductors[0] = ThinConductor(
        shape=MaskOnlyShape(_planar_sheet(NU_Z, NU_FOOT)),
        sigma_bulk=SIGMA_BULK, thickness=THICKNESS, surface_impedance_f0=F0)
    with pytest.raises(ValueError, match="refusing to skip"):
        assemble_materials_nu(sim, grid)


# ---------------------------------------------------------------------------
# O674-4: the graded-node advisory follows a non-Box sheet
# ---------------------------------------------------------------------------

def _preflight_text(shape, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=10e9, domain=(NU_L, NU_L, 0), dx=NU_DX,
                         dz_profile=NU_DZ, boundary="cpml", cpml_layers=6)
        sim.add_thin_conductor(shape, sigma_bulk=SIGMA_BULK,
                               thickness=THICKNESS, **kw)
        return " ".join(sim.preflight())


def test_graded_node_advisory_follows_a_nonbox_sheet():
    """Fires for a plain non-Box sheet on the step, fires for a PATTERNED one
    whose bounding-box centre falls in its clearance hole (a single-point
    probe reads "no sheet" there), and stays quiet on a matched node."""
    on_step = _preflight_text(_planar_sheet(NU_Z, NU_FOOT),
                              surface_impedance_f0=F0)
    assert "adjacent cells differ" in on_step, on_step
    assert "DUAL spacing" in on_step, on_step
    assert "500µm below" in on_step and "1.5mm above" in on_step, on_step

    # bounding-box centre is inside the hole by construction
    centre_hole = _planar_sheet(NU_Z, NU_FOOT, hole=NU_HOLE)
    lo, hi = centre_hole.bounding_box()
    mid = [0.5 * (lo[a] + hi[a]) for a in range(3)]
    assert not bool(np.asarray(centre_hole.mask_on_coords(
        np.array([mid[0]]), np.array([mid[1]]), np.array([mid[2]]))).any())
    patterned = _preflight_text(centre_hole, surface_impedance_f0=F0)
    assert "adjacent cells differ" in patterned, patterned

    # quiet: same sheets on a locally uniform node (z = 8 mm, deep in the
    # 1.5 mm region)
    for shape in (_planar_sheet(8.0e-3, NU_FOOT),
                  _planar_sheet(8.0e-3, NU_FOOT, hole=NU_HOLE)):
        quiet = _preflight_text(shape, surface_impedance_f0=F0)
        assert "adjacent cells differ" not in quiet, quiet


# ---------------------------------------------------------------------------
# O674-5: design-IR contract
# ---------------------------------------------------------------------------

def test_design_ir_records_a_registered_nonbox_sheet_and_refuses_the_rest():
    """The EXISTING shape codec decides: a registered primitive round-trips
    with its f0 field, an unregistered shape class is refused loudly (never
    degraded to a bounding box)."""
    from rfx.interop import design_to_dict, simulation_from_design
    from rfx.interop._errors import UnsupportedDesignFeature

    # height well under one cell so the disc is a SHEET: only the node plane
    # at its centre is contained (|h| <= height/2 = 50 um vs dx = 1 mm)
    disc = Cylinder(center=(10e-3, 10e-3, U_Z), radius=4e-3, height=1e-4,
                    axis="z")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=10e9, domain=U_DOMAIN, dx=U_DX)
        sim.add_thin_conductor(disc, sigma_bulk=SIGMA_BULK,
                               thickness=THICKNESS, surface_impedance_f0=F0)
        doc = design_to_dict(sim)
        back = simulation_from_design(doc)
    tc = back._thin_conductors[0]
    assert tc.shape == disc
    assert float(tc.surface_impedance_f0) == F0
    sig_a, _, _, _ = _uniform_sigma(disc, surface_impedance_f0=F0)
    sig_b = np.asarray(
        back._assemble_materials(back._build_grid())[0].sigma)
    assert _sha(sig_a) == _sha(sig_b)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim2 = Simulation(freq_max=10e9, domain=U_DOMAIN, dx=U_DX)
        sim2.add_thin_conductor(_planar_sheet(U_Z, U_FOOT),
                                sigma_bulk=SIGMA_BULK, thickness=THICKNESS,
                                surface_impedance_f0=F0)
    with pytest.raises(UnsupportedDesignFeature, match="PlanarSheet"):
        design_to_dict(sim2)


# ---------------------------------------------------------------------------
# The real CAD path: MeshShape (issue #358) carrying a surface-impedance sheet
# ---------------------------------------------------------------------------

def _mesh_slab(x0, x1, y0, y1, z0, z1, extra=()):
    """Watertight slab (or union of disjoint slabs) as a MeshShape."""
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("rtree")          # trimesh.contains needs it
    from rfx.geometry.mesh_import import MeshShape

    def _box(a0, a1, b0, b1, c0, c1):
        m = trimesh.creation.box(
            extents=(a1 - a0, b1 - b0, c1 - c0))
        m.apply_translation(((a0 + a1) / 2, (b0 + b1) / 2, (c0 + c1) / 2))
        return m

    parts = [_box(x0, x1, y0, y1, z0, z1)]
    parts += [_box(*p) for p in extra]
    mesh = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    return MeshShape(mesh)


def test_mesh_shape_sheet_folds_bit_identically_to_its_box():
    """An imported CAD slab and the Box it stands for fold the same sigma.

    Bounds are chosen OFF the node planes (4.6 .. 14.6 mm on a 1 mm grid) so
    the mesh's closed containment test and Box's half-open ``[lo, hi)`` rule
    select the same nodes 5..14 mm — the comparison is of the FOLD, not of two
    boundary conventions.
    """
    slab = _mesh_slab(4.6e-3, 14.6e-3, 4.6e-3, 14.6e-3,
                      U_Z - 1e-4, U_Z + 1e-4)
    sig_mesh, pec_mesh, grid, _ = _uniform_sigma(slab,
                                                 surface_impedance_f0=F0)
    sig_box, _, _, _ = _uniform_sigma(_box_sheet(U_Z, U_FOOT),
                                      surface_impedance_f0=F0)
    assert int((sig_mesh > 0).sum()) == 100, int((sig_mesh > 0).sum())
    np.testing.assert_array_equal(sig_mesh > 0, sig_box > 0)
    assert _sha(sig_mesh) == _sha(sig_box)
    assert pec_mesh is None or int(np.asarray(pec_mesh).sum()) == 0


def test_mesh_shape_patterned_sheet_leaves_its_clearance_hole_alone():
    """A ground plane with a clearance hole, as it arrives from CAD: four
    disjoint bars around an opening. The fold touches the metal only."""
    x0, x1, y0, y1 = 4.6e-3, 14.6e-3, 4.6e-3, 14.6e-3
    h0, h1 = 7.6e-3, 11.6e-3            # clearance opening
    z0, z1 = U_Z - 1e-4, U_Z + 1e-4
    frame = _mesh_slab(
        x0, x1, y0, h0, z0, z1,
        extra=((x0, x1, h1, y1, z0, z1),
               (x0, h0, h0, h1, z0, z1),
               (h1, x1, h0, h1, z0, z1)))
    sig, _, grid, _ = _uniform_sigma(frame, surface_impedance_f0=F0)
    mask = np.asarray(frame.mask(grid))

    solid = _mesh_slab(x0, x1, y0, y1, z0, z1)
    hole = np.asarray(solid.mask(grid)) & ~mask
    assert int(hole.sum()) == 16, int(hole.sum())     # 4x4 opening cells
    np.testing.assert_array_equal(sig > 0, mask)
    assert float(sig[hole].max(initial=0.0)) == 0.0

    rs0 = float(leontovich_rs(F0, SIGMA_BULK))
    prod = np.asarray(sig)[mask] * rs0 * grid.dx
    np.testing.assert_allclose(prod, 1.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# The #671 transition-node FDTD oracle, re-run with a NON-Box sheet
# ---------------------------------------------------------------------------

def _guide_planar_sheet(zs):
    """The oracle guide's plate, as a non-Box mask shape of equal footprint."""
    from tests.test_leontovich_alpha_oracle import DOMAIN as _D
    return PlanarSheet(2, zs, (0.0, 0.0), (_D[0], _D[1]))


@pytest.mark.slow_physics
@pytest.mark.parametrize("case,control,dual_over_primal",
                         [pytest.param(*c, id=c[0]) for c in INVARIANCE_CASES])
def test_alpha_invariance_transfers_to_a_nonbox_sheet(case, control,
                                                      dual_over_primal):
    """#671's oracle with the plates expressed as non-Box mask shapes.

    The gate is the same one: attenuation on a mesh where the sheet sits ON a
    grading step, over the locally-uniform control that shares that mesh, must
    be 1 within [0.98, 1.02] — a mesh-independent sheet has no other option.
    Because the two shapes rasterize the same cells, the folded sigma is
    bit-identical and so is alpha; both are asserted, so a divergence names
    itself (occupancy vs fold) instead of arriving as a drifted ratio.
    """
    from tests.test_thin_conductor_nu_dual_spacing import (
        RATIO_GATE, _run_nu_guide)
    got = _run_nu_guide(case, sheet_shape=_guide_planar_sheet, tag="planar")
    ref = _run_nu_guide(control, sheet_shape=_guide_planar_sheet,
                        tag="planar")

    hi_case = got["sheets"][-1]
    assert abs(hi_case[2] / hi_case[1] - dual_over_primal) < 1e-3, hi_case
    for _k, prim, dual, _prod in ref["sheets"]:
        assert abs(dual / prim - 1.0) < 1e-6, (prim, dual)
    for _k, _p, _d, prod in got["sheets"] + ref["sheets"]:
        assert abs(prod - 1.0) < 1e-5, prod
    for out in (got, ref):
        assert out["resid"] < 0.02, out["resid"]
        assert out["settle_db"] < -40.0, out["settle_db"]
        assert not any("PreflightError" in w for w in out["warnings"])

    ratio = got["alpha"] / ref["alpha"]
    lo, hi = RATIO_GATE
    assert lo <= ratio <= hi, (
        f"{case} (non-Box sheet): alpha {got['alpha']:.5f} vs control "
        f"{control} {ref['alpha']:.5f} -> ratio {ratio:.4f} outside "
        f"[{lo}, {hi}]")

    # ... and it is the SAME number the Box sheet gives on the same mesh
    for name, out in ((case, got), (control, ref)):
        box = _run_nu_guide(name)
        assert out["alpha"] == box["alpha"], (
            f"{name}: non-Box sheet alpha {out['alpha']:.9f} != Box "
            f"{box['alpha']:.9f} on identical occupancy")


def test_occupancy_guard_does_not_break_the_traced_mesh_path():
    """The #674 guard reads CONCRETE occupancy; on the differentiable-mesh
    path the mask is a tracer and the guard must step aside rather than raise
    a ConcretizationTypeError. Closed form: sigma_eff = 1/(Rs0*d_norm) scales
    as 1/scale, so d(sum sigma)/d(scale) = -sum/scale.
    """
    import jax

    base = jnp.asarray(NU_DZ)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=10e9, domain=(NU_L, NU_L, 0), dx=NU_DX,
                         dz_profile=base, boundary="cpml", cpml_layers=6)
        sim.add_thin_conductor(_box_sheet(NU_Z, NU_FOOT),
                               sigma_bulk=SIGMA_BULK, thickness=THICKNESS,
                               surface_impedance_f0=F0)

    def loss(scale):
        sim._dz_profile = base * scale
        return jnp.sum(assemble_materials_nu(sim, _nu_grid(sim))[0].sigma)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = float(loss(1.0))
        grad = float(jax.grad(loss)(1.0))
    sim._dz_profile = base
    assert value > 0.0
    assert abs(grad + value) / value < 1e-5, (grad, -value)


def test_vmap_batched_build_folds_a_nonbox_sheet_identically():
    """The batched (``vmap_sweep``) material build re-applies the same shared
    fold, so a non-Box sheet must land on the batched slices exactly as it
    lands on the serial ones (#669's O7, re-run off the Box)."""
    from rfx.vmap_sweep import _build_batched_materials

    def make(eps_val):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = Simulation(freq_max=10e9, domain=U_DOMAIN, dx=U_DX)
            sim.add_material("substrate", eps_r=eps_val)
            sim.add(Box((0.0, 0.0, 0.0), (0.02, 0.02, 1e-3)),
                    material="substrate")
            sim.add_thin_conductor(_planar_sheet(U_Z, U_FOOT, hole=U_HOLE),
                                   sigma_bulk=SIGMA_BULK,
                                   thickness=THICKNESS,
                                   surface_impedance_f0=F0)
        return sim

    eps_values = np.array([2.0, 6.0])
    sim = make(4.0)
    grid = sim._build_grid()
    base, *_ = sim._assemble_materials(grid)
    batched = _build_batched_materials(
        sim, grid, base, "substrate.eps_r", jnp.asarray(eps_values))
    assert batched.sigma.shape[0] == 2
    for idx, eps_val in enumerate(eps_values):
        serial, *_ = make(float(eps_val))._assemble_materials(grid)
        assert np.array_equal(np.asarray(batched.sigma[idx]),
                              np.asarray(serial.sigma))
        assert float(np.asarray(serial.sigma).max()) > 0.0
