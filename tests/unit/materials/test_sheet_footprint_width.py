"""A resistive strip is solved at its drawn width, by the DC and the f0 model alike.

Structure: a line with PEC walls in x and PMC walls in y carries a TEM wave
(Ex) along z. A z-normal strip of sheet resistance R_s = eta0 fills a fraction
``f`` of the line's width in y, its two drawn edges on grid nodes. The cross
section is lambda/10 wide at most, so the strip is a quasi-static shunt and

    |S21| = 1 / (1 + f/2)

at every frequency. |S21| therefore reads the electrical width that was solved.

Before the end-row rule the f0 (Leontovich) sheet loaded the Ex row lying ON
each drawn edge at full weight, one dual cell each, and was solved one cell
wider than drawn: (f + 1/N) in place of f, 0.6897 against 0.7143 at 8 of 10
cells. The DC fold was already at the drawn width, because the four-cell
average gives its two end rows half weight. ``END_ROW_WEIGHT`` (0.5) on the
f0 sheet's end rows gives both models the drawn width.

Set-up after the leader's measurement of 2026-10-07 (rfx-archive
``rfx/records/20261007-s1-g7-edge-order-g4-lossy-footprint/scripts``): cells of
lambda_min/100, a sheet of point sources, |S21| from a DFT plane divided by the
same plane of the empty line. The record is 2500 steps; at 5000 steps |S21|
moved by at most 1e-5 on both paths and both models (measured 2026-10-07).
"""
from functools import lru_cache

import numpy as np
import pytest

import rfx.materials.thin_conductor as thin_conductor
from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries.spec import BoundarySpec

C0 = 299792458.0
ETA0 = 376.730313668
MU0 = 4e-7 * np.pi
F0 = 9e9
FREQS = np.linspace(8e9, 10e9, 5)
DX = C0 / 10e9 / 100
NX, NZ = 4, 100
N_STEPS = 2500
THICKNESS = 35e-6
TOL = 1e-3


def _build(arm, width, graded, *, wall_to_wall=False, sources=True,
           y_boundary="pmc", span=None, shape=None):
    """``arm``: 'ref' (empty line), 'dc' or 'f0'. ``width`` cells of 1.25*width.

    ``span`` overrides the strip's two y faces (in cells); ``shape`` builds the
    strip from its corners instead of a Box.
    """
    ny = width * 5 // 4
    dom = (NX * DX, ny * DX, NZ * DX)
    kw = {}
    if graded:
        # Graded along the line only; the cells across the strip are uniform.
        half = [1.25 * DX] * 8 + [DX] * 40
        kw["dz_profile"] = np.array(half + half[::-1])
    sim = Simulation(freq_max=10e9, domain=dom, dx=DX,
                     boundary=BoundarySpec(x="pec", y=y_boundary, z="cpml"),
                     cpml_layers=12, snap="declared", **kw)
    if arm != "ref":
        lo, hi = ((0.0, dom[1]) if wall_to_wall
                  else ((ny - width) // 2 * DX, (ny + width) // 2 * DX))
        if span is not None:
            lo, hi = span[0] * DX, span[1] * DX
        sigma = (np.pi * F0 * MU0 / ETA0**2 if arm == "f0"
                 else 1.0 / (ETA0 * THICKNESS))
        corners = ((0.0, lo, dom[2] / 2), (dom[0], hi, dom[2] / 2))
        sim.add_thin_conductor(
            Box(*corners) if shape is None else shape(*corners),
            sigma_bulk=sigma, thickness=THICKNESS,
            surface_impedance_f0=F0 if arm == "f0" else None)
    if sources:
        wave = GaussianPulse(f0=F0, bandwidth=0.8)
        for i in range(NX):
            for j in range(ny):
                sim.add_source((i * DX, j * DX, dom[2] / 8), component="ex",
                               waveform=wave, amplitude_kind="field")
        sim.add_dft_plane_probe(axis="z", coordinate=2 * dom[2] / 3,
                                component="ex", freqs=FREQS, name="trans")
    return sim


def _plane(arm, width, graded, **kw):
    sim = _build(arm, width, graded, **kw)
    assert sim._uses_nonuniform_mesh == graded
    result = sim.run(n_steps=N_STEPS, compute_s_params=False, skip_preflight=True)
    return np.asarray(result.dft_planes["trans"].accumulator)


@lru_cache(maxsize=None)
def _reference(width, graded, y_boundary="pmc"):
    return _plane("ref", width, graded, y_boundary=y_boundary).mean(axis=(1, 2))


def _s21(arm, width, graded, y_boundary="pmc", **kw):
    return np.abs(_plane(arm, width, graded, y_boundary=y_boundary, **kw).mean(axis=(1, 2))
                  / _reference(width, graded, y_boundary))


@pytest.mark.parametrize("width", [
    8,
    pytest.param(16, marks=pytest.mark.slow),
    pytest.param(24, marks=pytest.mark.slow),
])
@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_resistive_strip_is_solved_at_its_drawn_width(width, graded):
    expected = 1.0 / (1.0 + 0.8 / 2.0)
    dc = _s21("dc", width, graded)
    f0 = _s21("f0", width, graded)
    assert np.max(np.abs(dc - expected)) < TOL, (dc, expected)
    assert np.max(np.abs(f0 - expected)) < TOL, (f0, expected)
    assert np.max(np.abs(dc - f0)) < TOL, (dc, f0)


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_full_weight_end_rows_solve_the_f0_strip_one_cell_wider(monkeypatch, graded):
    """The weight is what sets the width: at 1.0 the strip is 9 of 10 cells."""
    monkeypatch.setattr(thin_conductor, "END_ROW_WEIGHT", 1.0)
    f0 = _s21("f0", 8, graded)
    assert np.max(np.abs(f0 - 1.0 / (1.0 + 0.9 / 2.0))) < TOL, f0


def test_wall_to_wall_f0_sheet_does_not_read_the_end_row_weight(monkeypatch):
    """A sheet that runs into both walls has no free edge and no end row."""
    half = _plane("f0", 8, False, wall_to_wall=True)
    monkeypatch.setattr(thin_conductor, "END_ROW_WEIGHT", 1.0)
    full = _plane("f0", 8, False, wall_to_wall=True)
    np.testing.assert_allclose(half, full, rtol=1e-6, atol=0.0)
    s21 = np.abs(half.mean(axis=(1, 2)) / _reference(8, False))
    assert np.max(np.abs(s21 - 1.0 / 1.5)) < TOL, s21


def _sheet_ctx(sim, graded=False):
    from rfx.materials.thin_conductor import build_sheet_impedance_ctx
    from rfx.model.conductors import realized_conductors
    grid = sim._build_nonuniform_grid() if graded else sim._build_grid()
    root = realized_conductors(sim, grid, nonuniform=graded, mode="audit")
    return build_sheet_impedance_ctx(root.sheet_impedance,
                                     pec_edge_masks=root.pec_edges,
                                     periodic=root.periodic)


def _rows(mask, axis):
    other = tuple(a for a in range(3) if a != axis)
    return [] if mask is None else np.flatnonzero(np.asarray(mask).any(axis=other)).tolist()


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_end_rows_are_the_edges_on_a_drawn_free_edge(graded):
    """Ex on the two y rows of the drawn edges; no Ey edge; the edge SET is unchanged."""
    ctx = _sheet_ctx(_build("f0", 8, graded, sources=False), graded)
    assert _rows(ctx.mask_ex, 1) == list(range(1, 10))
    assert _rows(ctx.mask_ey, 1) == list(range(1, 9))
    assert _rows(ctx.end_ex, 1) == [1, 9]
    assert ctx.end_ey is None and ctx.end_ez is None
    assert not np.any(np.asarray(ctx.end_ex) & ~np.asarray(ctx.mask_ex))


def _f0_sim(*boxes):
    sim = Simulation(freq_max=10e9, domain=(10 * DX, 10 * DX, 10 * DX), dx=DX,
                     boundary=BoundarySpec(x="pec", y="pmc", z="pec"), snap="declared")
    for lo, hi in boxes:
        sim.add_thin_conductor(Box(tuple(v * DX for v in lo), tuple(v * DX for v in hi)),
                               sigma_bulk=1e4, thickness=THICKNESS, surface_impedance_f0=F0)
    return sim


def test_end_row_marking_cases():
    from rfx.geometry.csg import Cylinder
    # A patch free on all four sides: Ex on its y edges, Ey on its x edges.
    ctx = _sheet_ctx(_f0_sim(((2, 3, 5), (7, 8, 5))))
    assert _rows(ctx.end_ex, 1) == [3, 8] and _rows(ctx.end_ey, 0) == [2, 7]
    # Drawn edges between nodes: the last row is inside the metal, full weight.
    ctx = _sheet_ctx(_f0_sim(((0, 2.5, 5), (10, 7.5, 5))))
    assert ctx.end_ex is None and ctx.end_ey is None
    # Two sheets meeting on node row 5: the seam row is an end row of both,
    # so it carries half of each and the pair is one continuous strip.
    ctx = _sheet_ctx(_f0_sim(((0, 2, 5), (10, 5, 5)), ((0, 5, 5), (10, 8, 5))))
    assert _rows(ctx.end_ex, 1) == [2, 5, 8]
    # One sheet's edge inside another sheet: that row stays at full weight.
    ctx = _sheet_ctx(_f0_sim(((0, 2, 5), (10, 8, 5)), ((0, 5, 5), (10, 8, 5))))
    assert _rows(ctx.end_ex, 1) == [2, 8]
    # Another normal, by rotation: an x-normal strip free in z carries Ey end rows.
    ctx = _sheet_ctx(_f0_sim(((5, 0, 2), (5, 10, 8))))
    assert _rows(ctx.end_ey, 2) == [2, 8] and ctx.end_ez is None
    # A curved outline has no drawn face along a node row: weight 1 everywhere.
    sim = _f0_sim()
    sim.add_thin_conductor(Cylinder((5 * DX, 5 * DX, 5 * DX), 3 * DX, 0.0, axis="z"),
                           sigma_bulk=1e4, thickness=THICKNESS, surface_impedance_f0=F0)
    ctx = _sheet_ctx(sim)
    assert ctx.end_ex is None and ctx.end_ey is None


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_f0_record_reports_the_solved_extent(monkeypatch, graded):
    """Node footprint, not cells: the record was one cell too wide on every in-plane axis."""
    def axes():
        entity = _build("f0", 8, graded, sources=False).realized_geometry().entities[0]
        return {a.axis: a for a in entity.axes}

    rec = axes()
    assert rec["y"].node_range == (1, 9) and rec["y"].cell_range is None
    assert rec["y"].extent_m == pytest.approx(8 * DX, rel=1e-9)
    assert rec["y"].bounds_m == pytest.approx((1 * DX, 9 * DX), rel=1e-9)
    assert rec["x"].extent_m == pytest.approx(NX * DX, rel=1e-9)      # wall to wall
    assert rec["z"].extent_m == 0.0                                    # one node plane
    # The record follows what is solved: full-weight end rows are a cell wider.
    monkeypatch.setattr(thin_conductor, "END_ROW_WEIGHT", 1.0)
    assert axes()["y"].extent_m == pytest.approx(9 * DX, rel=1e-9)


def _planar(lo, hi):
    """The Box's rectangle as a non-Box shape: its own sampler, the same drawn faces."""
    from tests.unit.materials.test_sheet_impedance import PlanarSheet
    return PlanarSheet(2, lo[2], (lo[0], lo[1]), (hi[0], hi[1]))


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_a_rectangle_is_marked_by_its_drawn_faces_whatever_shape_declares_it(graded):
    """The rule reads the drawn faces, not the shape's type."""
    box = _sheet_ctx(_build("f0", 8, graded, sources=False), graded)
    other = _sheet_ctx(_build("f0", 8, graded, sources=False, shape=_planar), graded)
    assert _rows(other.end_ex, 1) == [1, 9]
    for name in ("mask_ex", "mask_ey", "mask_ez", "end_ex", "sigma_sheet"):
        np.testing.assert_array_equal(np.asarray(getattr(other, name)),
                                      np.asarray(getattr(box, name)), err_msg=name)
    assert other.end_ey is None and other.end_ez is None


@pytest.mark.parametrize("span", [(0, 5), (5, 10), (2, 7)])
def test_a_periodic_axis_has_no_wall(span):
    """A drawn face on the seam node is a free edge like any other."""
    sim = _build("f0", 8, False, sources=False, y_boundary="periodic", span=span)
    ctx = _sheet_ctx(sim)
    assert _rows(ctx.end_ex, 1) == sorted({span[0] % 10, span[1] % 10})
    axis = {a.axis: a for a in sim.realized_geometry().entities[0].axes}["y"]
    assert axis.node_range == span
    assert axis.bounds_m == pytest.approx((span[0] * DX, span[1] * DX), rel=1e-9, abs=1e-15)
    assert axis.extent_m == pytest.approx(5 * DX, rel=1e-9)


def test_a_sheet_spanning_the_whole_period_has_no_end_row():
    sim = _build("f0", 8, False, sources=False, y_boundary="periodic", span=(0, 10))
    ctx = _sheet_ctx(sim)
    assert _rows(ctx.mask_ex, 1) == list(range(10))
    assert ctx.end_ex is None and ctx.end_ey is None
    axis = {a.axis: a for a in sim.realized_geometry().entities[0].axes}["y"]
    assert axis.extent_m == pytest.approx(10 * DX, rel=1e-9)


def test_strips_on_either_side_of_a_periodic_seam_are_solved_the_same_width():
    """Half of the period, drawn 0..5 or 5..10: |S21| = 1/(1 + 0.5/2) for both."""
    low = _s21("f0", 8, False, "periodic", span=(0, 5))
    high = _s21("f0", 8, False, "periodic", span=(5, 10))
    assert np.max(np.abs(low - 0.8)) < TOL, low
    assert np.max(np.abs(high - 0.8)) < TOL, high
    assert np.max(np.abs(low - high)) < 1e-4, (low, high)


def test_traced_and_eager_graded_meshes_solve_the_same_strip():
    """A mesh that is a JAX tracer marks the same end rows as its concrete twin.

    The objective is summed over the record, so the cross-trace bar is 1e-4 of
    it; one cell of strip width moves it by 7 %, half a cell by 3.7 %.
    """
    import jax
    import jax.numpy as jnp

    from rfx.core.jax_utils import is_tracer
    from rfx.runners.nonuniform import run_nonuniform_path

    ny, nz = 10, 60
    dt = 0.9 / (C0 * np.sqrt(3.0 / (0.9 * DX) ** 2))

    def loss(scale):
        dy = ((jnp.full((ny,), DX) * scale).astype(jnp.float32) if is_tracer(scale)
              else np.full((ny,), DX) * float(scale))
        half = [1.25 * DX] * 4 + [DX] * 25
        sim = Simulation(freq_max=10e9, domain=(NX * DX, ny * DX, nz * DX), dx=DX,
                         dy_profile=dy, dz_profile=np.array(half + half[::-1]),
                         boundary=BoundarySpec(x="pec", y="pmc", z="cpml"),
                         cpml_layers=8, dt=float(dt), dt_min_cell=float(0.9 * DX))
        z = nz * DX / 2
        sim.add_thin_conductor(Box((0.0, 1 * DX, z), (NX * DX, 9 * DX, z)),
                               sigma_bulk=np.pi * F0 * MU0 / ETA0**2,
                               thickness=THICKNESS, surface_impedance_f0=F0)
        for j in range(ny):
            sim.add_source((DX, j * DX, nz * DX / 8), component="ex",
                           waveform=GaussianPulse(f0=F0, bandwidth=0.8),
                           amplitude_kind="field")
        sim.add_probe((DX, 5 * DX, 2 * nz * DX / 3), "ex")
        result = run_nonuniform_path(sim, n_steps=1500)
        return jnp.sum(jnp.asarray(result.time_series)[:, 0] ** 2)

    eager = float(loss(1.0))
    traced = float(jax.value_and_grad(loss)(jnp.float32(1.0))[0])
    assert eager > 0
    assert abs(traced - eager) <= 1e-4 * eager, (traced, eager)
