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
