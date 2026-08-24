"""CAD mesh import — MeshShape occupancy vs analytic primitives (issue #358).

Skips cleanly when the optional 'cad' extra (trimesh) is absent (module-level importorskip).
Acceptance: an imported STL/mesh rasterises to the same Yee occupancy as the equivalent CSG
primitive to within one cell (the shared cell-centre staircase), including a rotated body;
watertightness is enforced; and a MeshShape plugs into Simulation.add like any other Shape.
"""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh", reason="optional 'cad' extra (trimesh) not installed")

from rfx.geometry.mesh_import import MeshShape
from rfx.geometry.csg import Sphere


def _coords(lo, hi, dx):
    n = int(round((hi - lo) / dx)) + 1
    return np.linspace(lo, hi, n)


def _within_one_cell(xor_mask, surf_dist, dx):
    """Every disagreeing cell must lie within one cell of the surface (|dist| < dx)."""
    bad = np.argwhere(xor_mask)
    if bad.size == 0:
        return True
    return bool(np.all(surf_dist[xor_mask] < dx * 1.0000001))


def test_mesh_sphere_matches_primitive_within_one_cell():
    """An icosphere mesh and the analytic Sphere agree on occupancy except within one
    cell of the surface — proof the import rasterises identically to a CSG primitive."""
    R, dx = 0.03, 0.004
    mesh = MeshShape(trimesh.creation.icosphere(subdivisions=4, radius=R))
    prim = Sphere(center=(0.0, 0.0, 0.0), radius=R)

    x = _coords(-0.05, 0.05, dx)
    m_mesh = np.asarray(mesh.mask_on_coords(x, x, x))
    m_prim = np.asarray(prim.mask_on_coords(x, x, x))

    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    surf_dist = np.abs(np.sqrt(X ** 2 + Y ** 2 + Z ** 2) - R)
    xor = m_mesh ^ m_prim
    frac = xor.sum() / m_prim.sum()
    assert frac < 0.10, f"mesh vs primitive disagree on {frac:.1%} of interior cells (>10%)"
    assert _within_one_cell(xor, surf_dist, dx), (
        "mesh/primitive disagreement is NOT confined to the one-cell surface shell")


def test_mesh_rotated_box_matches_analytic_within_one_cell():
    """A 45°-rotated box mesh matches an independent analytic rotated-box occupancy within
    one cell — the containment test handles orientation (not just axis-aligned)."""
    ex, ey, ez = 0.04, 0.02, 0.03
    dx = 0.003
    box = trimesh.creation.box(extents=(ex, ey, ez))
    theta = np.pi / 4
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                   [np.sin(theta), np.cos(theta), 0, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]])
    box.apply_transform(Rz)
    mesh = MeshShape(box)

    x = _coords(-0.05, 0.05, dx)
    m_mesh = np.asarray(mesh.mask_on_coords(x, x, x))

    # independent analytic mask: rotate points back into the box frame, half-extent test
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    xr = np.cos(theta) * X + np.sin(theta) * Y
    yr = -np.sin(theta) * X + np.cos(theta) * Y
    m_ana = (np.abs(xr) <= ex / 2) & (np.abs(yr) <= ey / 2) & (np.abs(Z) <= ez / 2)
    # surface distance in the box frame (min distance to any face plane, signed→abs)
    dfx = np.abs(np.abs(xr) - ex / 2)
    dfy = np.abs(np.abs(yr) - ey / 2)
    dfz = np.abs(np.abs(Z) - ez / 2)
    surf_dist = np.minimum(np.minimum(dfx, dfy), dfz)

    xor = m_mesh ^ m_ana
    frac = xor.sum() / m_ana.sum()
    assert frac < 0.12, f"rotated-box mesh vs analytic disagree on {frac:.1%} of cells"
    assert _within_one_cell(xor, surf_dist, dx * np.sqrt(2)), (
        "rotated-box disagreement not confined to the ~one-cell surface shell")


def test_mesh_requires_watertight():
    """A non-watertight mesh (open surface) is rejected at construction — a leaky mesh
    gives an undefined inside/outside test and a silently-wrong occupancy mask."""
    # a single triangle: has open edges, not watertight
    leaky = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])
    assert not leaky.is_watertight
    with pytest.raises(ValueError, match="watertight"):
        MeshShape(leaky)


def test_mesh_from_file_roundtrip_and_scale(tmp_path):
    """from_file loads an STL, applies the REQUIRED explicit scale (STL is unitless), and
    positions via translate; bounding_box and a known interior point come out right."""
    R_mm = 30.0  # sphere drawn in millimetres
    sphere_mm = trimesh.creation.icosphere(subdivisions=3, radius=R_mm)
    stl = tmp_path / "sphere_mm.stl"
    sphere_mm.export(stl)

    # load with mm→m scale and an offset
    shape = MeshShape.from_file(str(stl), scale=1e-3, translate=(0.1, 0.0, 0.0))
    (lox, loy, loz), (hix, hiy, hiz) = shape.bounding_box()
    assert lox == pytest.approx(0.1 - 0.03, abs=2e-3) and hix == pytest.approx(0.1 + 0.03, abs=2e-3)
    # centre point is inside; a point well outside is not
    inside = np.asarray(shape.mask_on_coords(np.array([0.1]), np.array([0.0]), np.array([0.0])))
    outside = np.asarray(shape.mask_on_coords(np.array([0.2]), np.array([0.0]), np.array([0.0])))
    assert bool(inside[0, 0, 0]) and not bool(outside[0, 0, 0])

    with pytest.raises(ValueError, match="scale"):
        MeshShape.from_file(str(stl), scale=0.0)


def test_mesh_plugs_into_simulation():
    """A MeshShape composes through Simulation.add(...) like any CSG shape and rasterises
    onto the grid (end-to-end integration, not just the mask helper)."""
    from rfx.api import Simulation
    from rfx.grid import Grid

    sim = Simulation(freq_max=10e9, domain=(0.06, 0.06, 0.06), dx=0.004,
                     boundary="cpml", cpml_layers=6, mode="3d")
    sim.add(MeshShape(trimesh.creation.icosphere(subdivisions=3, radius=0.012)), material="pec")
    grid = Grid(freq_max=10e9, domain=(0.06, 0.06, 0.06), dx=0.004, cpml_layers=6)
    m = np.asarray(sim._geometry[-1].shape.mask(grid))
    assert m.shape == grid.shape and 0 < m.sum() < m.size, "mesh did not rasterise a partial volume"


def test_mesh_preflight_underresolved_advisory():
    """The preflight advisory fires when the mesh's THINNEST dimension is below ~2 cells (a thin
    plate/wall — the #330 class), and stays SILENT on a well-resolved part regardless of how finely
    it is tessellated (the proxy is bbox extent, not triangle-edge, so smooth CAD doesn't cry wolf).
    preflight() collects messages into the returned PreflightReport."""
    from rfx.api import Simulation

    sim = Simulation(freq_max=5e9, domain=(0.06, 0.06, 0.06), dx=0.006,
                     boundary="cpml", cpml_layers=6, mode="3d")
    sim.add_source((0.03, 0.03, 0.03), component="ez")
    # a thin plate: 0.8 mm thick << 2·dx (12 mm) — its thickness is lost by rasterisation
    plate = trimesh.creation.box(extents=(0.03, 0.02, 0.0008))
    plate.apply_translation([0.03, 0.03, 0.03])
    sim.add(MeshShape(plate), material="pec")
    report = sim.preflight()
    assert any("thinnest dimension" in str(m) and "below 2 cells" in str(m) for m in report), (
        f"under-resolved (thin-plate) mesh advisory did not fire; report={[str(m) for m in report]}")

    # control: a finely-tessellated but WELL-RESOLVED sphere must NOT trip the advisory — its
    # triangle edges are ~sub-mm, but its thinnest dimension (diameter 80 mm) spans many cells.
    sim2 = Simulation(freq_max=5e9, domain=(0.12, 0.12, 0.12), dx=0.002,
                      boundary="cpml", cpml_layers=6, mode="3d")
    sim2.add_source((0.06, 0.06, 0.06), component="ez")
    ball = trimesh.creation.icosphere(subdivisions=4, radius=0.04)
    ball.apply_translation([0.06, 0.06, 0.06])
    sim2.add(MeshShape(ball), material="pec")
    report2 = sim2.preflight()
    assert not any("thinnest dimension" in str(m) for m in report2), (
        "advisory false-fired on a finely-tessellated but well-resolved sphere (tessellation cry-wolf)")


def test_mesh_from_step_file():
    """STEP import via the cascadio backend (issue #358 Stage 2): the committed 30×20×2 mm box
    loads watertight (tessellation seams welded), already in metres (scale=1.0), and its occupancy
    matches an analytic axis-aligned box within one cell."""
    pytest.importorskip("cascadio", reason="STEP import needs the optional cascadio backend")
    import os

    step = os.path.join(os.path.dirname(__file__), "fixtures", "mesh", "box_30x20x2mm.step")
    shape = MeshShape.from_file(step, scale=1.0)  # cascadio → metres already
    (lox, loy, loz), (hix, hiy, hiz) = shape.bounding_box()
    assert (hix - lox, hiy - loy, hiz - loz) == pytest.approx((0.03, 0.02, 0.002), abs=1e-4)

    dx = 0.0008
    x = _coords(-0.004, 0.034, dx)
    y = _coords(-0.004, 0.024, dx)
    z = _coords(-0.003, 0.005, dx)
    m = np.asarray(shape.mask_on_coords(x, y, z))
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    ana = ((X >= lox) & (X <= hix) & (Y >= loy) & (Y <= hiy) & (Z >= loz) & (Z <= hiz))
    dfx = np.minimum(np.abs(X - lox), np.abs(X - hix))
    dfy = np.minimum(np.abs(Y - loy), np.abs(Y - hiy))
    dfz = np.minimum(np.abs(Z - loz), np.abs(Z - hiz))
    surf_dist = np.minimum(np.minimum(dfx, dfy), dfz)
    xor = m ^ ana
    assert xor.sum() / ana.sum() < 0.15, "STEP box occupancy disagrees with analytic by >15%"
    assert _within_one_cell(xor, surf_dist, dx), "STEP box disagreement not confined to one cell"


def test_mesh_caches_occupancy():
    """The expensive point-in-mesh containment is cached by coordinate content: re-rasterising the
    same grid returns the SAME mask object (the S-param paths rebuild materials repeatedly)."""
    shape = MeshShape(trimesh.creation.icosphere(subdivisions=3, radius=0.01))
    x = np.linspace(-0.02, 0.02, 20)
    m1 = shape.mask_on_coords(x, x, x)
    m2 = shape.mask_on_coords(x, x, x)
    assert m1 is m2, "repeat rasterisation of the same grid should return the cached mask"
    m3 = shape.mask_on_coords(x * 0.5, x, x)   # different grid → cache miss, distinct entry
    assert m3 is not m1 and len(shape._mask_cache) == 2


def test_mesh_rejects_traced_coordinates():
    """MeshShape rasterises host-side (trimesh.contains) so it can't be traced/jitted — a traced
    coordinate must raise a clear MeshShape error, not a cryptic JAX array-conversion failure."""
    import jax
    import jax.numpy as jnp

    shape = MeshShape(trimesh.creation.icosphere(subdivisions=2, radius=0.01))
    coords = jnp.linspace(-0.02, 0.02, 8)

    def rasterize(c):
        return shape.mask_on_coords(c, c, c).sum()

    with pytest.raises(NotImplementedError, match="cannot be traced/jitted"):
        jax.jit(rasterize)(coords)


# --------------------------------------------------------------------------- #
# #687 — deterministic rasterization of surface-registered sample points.
#
# The boundary-fitted meshers register conductor sheets at mid-plane NODES, so
# rasterization sample points land exactly on mesh faces. trimesh's
# ``contains`` decides such points along an OS-entropy-random ray direction
# (measured 2026-08-24: a sample 1 ulp off a sheet's lateral face flipped
# inside/outside across 40 fresh calls, states 0,1,1,0,1,0,...; a point
# 0.2 um strictly INSIDE a face returned True,False,False,... over 8 calls).
# MeshShape therefore casts its own fixed-direction parity rays from a
# tie-break-nudged origin (mesh_import.py module block).


def _box_volume_expectation(x, y, z, lo, hi, tie_break=False):
    """Analytic Box VOLUME-branch occupancy: half-open [lo, hi) per axis
    (csg.py convention), computed in float64 on the exact sample coords.

    ``tie_break=True`` states the documented #687 contract for samples INSIDE
    the sub-nanometre tie band of a face: they resolve as the half-open test
    of the NUDGED coordinate, i.e. ``lo <= x + eps_ax < hi`` with the fixed
    per-axis nudge component — identical to the plain rule everywhere farther
    than ~1e-12 m from a face, lo-inclusive/hi-exclusive ON it."""
    from rfx.geometry.mesh_import import _TIE_BREAK_EPS_M, _D1
    e = _TIE_BREAK_EPS_M * _D1 if tie_break else np.zeros(3)
    mx = (x + e[0] >= lo[0]) & (x + e[0] < hi[0])
    my = (y + e[1] >= lo[1]) & (y + e[1] < hi[1])
    mz = (z + e[2] >= lo[2]) & (z + e[2] < hi[2])
    return mx[:, None, None] & my[None, :, None] & mz[None, None, :]


def test_surface_on_nodes_matches_box_convention_both_sides():
    """A cube whose every face lands EXACTLY on grid nodes must rasterize to
    the Box half-open [lo, hi) convention — on-surface lo-face nodes inside,
    on-surface hi-face nodes outside, and the node one step off each face
    correct on BOTH sides.

    Mutation-falsified both directions (measured 2026-08-24, this worktree;
    output verbatim):
    - defect direction — ``_contains_deterministic`` replaced by the original
      ``self._mesh.contains``: this test FAILS
      ``AssertionError: cells=343 expected 512 wrong=169`` (every on-surface
      node resolved outside), and the sheet test FAILS
      ``AssertionError: build 3 differs from build 0`` (the RNG coin flip
      itself);
    - overcorrection direction — tie-break direction negated (``-_D1``):
      this test FAILS ``AssertionError: cells=512 expected 512 wrong=338``
      ([lo, hi) became (lo, hi] on every axis: right count, planes shifted).
      The sheet test alone does NOT catch this one — its tie-band
      expectation imports ``_D1`` and mutates with the implementation —
      which is why THIS test states its expectation without the module
      constants;
    - nudge removed (``_TIE_BREAK_EPS_M = 0.0``): FAILS at
      ``AssertionError: 1-ulp-below-lo-face sample must tie-break INSIDE``
      (the exact-on-node planes happened to resolve correctly under this
      fixture's arithmetic, so the shifted-plane assertion below is what
      pins the nudge), and the sheet test FAILS ``sheet occupancy != f64
      half-open tie-break expectation (wrong=1)``.
    """
    dx = 50e-6
    nodes = np.arange(13) * dx                       # exact f64 multiples
    lo, hi = (3 * dx,) * 3, (11 * dx,) * 3
    cube = trimesh.creation.box(bounds=[lo, hi])
    m = np.asarray(MeshShape(cube).mask_on_coords(nodes, nodes, nodes))
    expect = _box_volume_expectation(nodes, nodes, nodes, lo, hi)
    assert np.array_equal(m, expect), (
        f"cells={int(m.sum())} expected {int(expect.sum())} "
        f"wrong={int((m ^ expect).sum())}")
    # both sides of the lo x-face, spelled out
    assert m[3, 6, 6] and not m[2, 6, 6], "lo face: on-surface in, one-out out"
    # both sides of the hi x-face
    assert m[10, 6, 6] and not m[11, 6, 6], "hi face: one-in in, on-surface out"

    # A sample plane 1 ulp BELOW the lo face is a tie-class point whose f64
    # answer is 'outside' but whose intended registration is the face; the
    # documented tie-break (positive nudge ~1e-12 m, orders above ulp noise)
    # must pull it INSIDE deterministically — this is the assertion that pins
    # the nudge itself (without it the outcome follows per-point roundoff).
    x_shift = nodes.copy()
    x_shift[3] = np.nextafter(nodes[3], -np.inf)     # 1 ulp below the face
    m2 = np.asarray(MeshShape(trimesh.creation.box(bounds=[lo, hi]))
                    .mask_on_coords(x_shift, nodes, nodes))
    assert m2[3, 6, 6], "1-ulp-below-lo-face sample must tie-break INSIDE"


def test_mid_plane_registered_sheet_deterministic_and_correct_both_sides():
    """The issue-#687 class: a 17 um sheet registered at its mid-plane node,
    lateral faces on cumsum-derived (float-noisy) nodes. Repeated fresh
    builds must be bit-identical (the issue's falsifier: 'build three times,
    compare with np.array_equal — today they differ'), and the mid-plane
    row must match the f64 analytic expectation on both sides of each
    lateral face.

    Pre-fix measurement on THIS fixture (2026-08-24): per-build inside-count
    alternated 17/18 across 40 fresh ``mesh.contains`` builds — states of the
    flipping corner sample: 0,1,1,0,1,0,1,1,1,1,0,1,... (OS-entropy RNG).
    Post-fix: 40/40 builds identical.
    """
    dx = 50e-6
    dz = np.full(15, 31.43e-6)
    znodes = np.concatenate([[0.0], np.cumsum(dz)])
    zmid = float(znodes[7])                          # mid-plane-registered
    xnodes = np.cumsum(np.full(12, dx)) - dx
    rng = np.random.default_rng(0)                   # jitter is the fixture

    def _jit(v):
        return v + rng.integers(-4, 5, size=np.shape(v)) * np.spacing(v)

    xs, ys = _jit(xnodes), _jit(xnodes)
    lo = (float(xnodes[3]), float(xnodes[3]), zmid - 8.5e-6)
    hi = (float(xnodes[8]), float(xnodes[8]), zmid + 8.5e-6)

    masks = []
    for _ in range(5):                               # fresh MeshShape: no cache
        sheet = trimesh.creation.box(bounds=[lo, hi])
        masks.append(np.asarray(MeshShape(sheet).mask_on_coords(xs, ys, znodes)))
    for k, mk in enumerate(masks[1:], 1):
        assert np.array_equal(masks[0], mk), f"build {k} differs from build 0"

    m = masks[0]
    expect = _box_volume_expectation(xs, ys, znodes, lo, hi, tie_break=True)
    assert np.array_equal(m, expect), (
        f"sheet occupancy != f64 half-open tie-break expectation "
        f"(wrong={int((m ^ expect).sum())})")
    # both sides of the lateral surface at the registered mid-plane, spelled
    # out: strictly-inside node occupied, strictly-outside node empty.
    assert m[4, 5, 7] and m[7, 5, 7], "interior mid-plane cells must be PEC"
    assert not m[2, 5, 7] and not m[9, 5, 7], "exterior mid-plane cells empty"
