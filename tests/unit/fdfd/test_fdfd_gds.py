"""rfx.fdfd.gds: GDS round trip, grid-exact meshing, exact-area rasterisation,
and the parametric spiral generators against their analytic lengths.

Everything here is host geometry (no JAX), so no x64 scope is needed.
Reference values are analytic: polygon areas by the shoelace formula on the
input vertices, metal length as strip area / width (exact for a mitred
strip), conductor volume as area * layer thickness. Polygon validity is
cross-checked against shapely (an independent implementation).
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import sys
import time

import numpy as np
import pytest

import rfx.fdfd.gds as gds_module
from rfx.fdfd.gds import (
    Layer, LayerStack, Spiral, fill_fractions, grade_lines, load_stack, mesh_lines,
    octagonal_spiral, offset_polyline, polygon_area, polygon_is_simple, rasterise, read_gds,
    rect_from_bounds, rect_spiral, sg13g2_stack, write_gds, z_lines,
)

gdstk = pytest.importorskip("gdstk")
shapely = pytest.importorskip("shapely")

UM = 1e-6
RECT = rect_from_bounds(10 * UM, -5 * UM, 40 * UM, 7 * UM)            # 30 x 12 um
OCT = np.array([[np.cos(a), np.sin(a)] for a in np.pi / 8 + np.arange(8) * np.pi / 4]) * 50 * UM
KEY_M, KEY_U = (134, 0), (126, 0)


def _stack() -> LayerStack:
    """Two-metal toy stack: oxide 0..10 um, metal M at 5..7 um (GDS 134),
    underpass U at 2..3 um (GDS 126), vacuum above."""
    return LayerStack((
        Layer("oxide", "dielectric", 0.0, 10 * UM, eps_r=4.1),
        Layer("U", "conductor", 2 * UM, 1 * UM, gds=KEY_U, sigma=3e7),
        Layer("M", "conductor", 5 * UM, 2 * UM, gds=KEY_M, sigma=3e7),
    ))


def _max_ratio(lines: np.ndarray) -> float:
    d = np.diff(lines)
    if len(d) < 2:
        return 1.0
    return float(max((d[1:] / d[:-1]).max(), (d[:-1] / d[1:]).max()))


def _holes(poly: np.ndarray, d: float) -> int:
    """Number of interior rings of the polygon buffered by d."""
    g = shapely.buffer(shapely.Polygon(poly), d)
    parts = g.geoms if hasattr(g, "geoms") else [g]
    return sum(len(p.interiors) for p in parts)


# ---------------------------------------------------------------- T1

def test_gds_round_trip_layers_and_areas(tmp_path):
    """gdstk write -> read_gds: layer keys, polygon count, areas to 1e-12
    (the file quantises to 1 nm; the vertices used are on that grid, and
    the octagon vertices are rounded to it before comparing)."""
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    top = lib.new_cell("TOP")
    oct_um = np.round(OCT / UM, 3)                # on the 1 nm database grid
    top.add(gdstk.Polygon(RECT / UM, layer=134, datatype=0))
    top.add(gdstk.Polygon(oct_um, layer=126, datatype=2))
    # a referenced sub-cell must be flattened
    sub = lib.new_cell("SUB")
    sub.add(gdstk.rectangle((0, 0), (2, 3), layer=8, datatype=0))
    top.add(gdstk.Reference(sub, (100, 100)))
    path = tmp_path / "t1.gds"
    lib.write_gds(str(path))

    polys = read_gds(path)
    assert set(polys) == {(134, 0), (126, 2), (8, 0)}
    assert len(polys[(134, 0)]) == 1 and len(polys[(126, 2)]) == 1 and len(polys[(8, 0)]) == 1
    a_rect, a_oct, a_sub = (abs(polygon_area(polys[k][0])) for k in ((134, 0), (126, 2), (8, 0)))
    assert abs(a_rect / (30 * 12 * UM * UM) - 1) < 1e-12
    assert abs(a_oct / abs(polygon_area(oct_um * UM)) - 1) < 1e-12
    assert abs(a_sub / (6 * UM * UM) - 1) < 1e-12
    # the reference was translated to (100, 100) um
    assert np.allclose(polys[(8, 0)][0].min(axis=0), [100 * UM, 100 * UM], rtol=0, atol=1e-15)

    # write_gds is the inverse (same 1 nm grid)
    write_gds(tmp_path / "t1b.gds", polys)
    back = read_gds(tmp_path / "t1b.gds")
    for k in polys:
        assert abs(abs(polygon_area(back[k][0])) / abs(polygon_area(polys[k][0])) - 1) < 1e-12


def test_gds_import_is_lazy(monkeypatch):
    """With gdstk made unimportable the module still loads, its namespace
    holds no gdstk binding, and only read_gds raises a pointed ImportError.
    The source is executed under a throwaway module name so the real
    rfx.fdfd.gds (and the classes this file imported) are untouched."""
    monkeypatch.setitem(sys.modules, "gdstk", None)   # 'import gdstk' -> ImportError
    spec = importlib.util.spec_from_file_location("_gds_lazy_probe", gds_module.__file__)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, probe)   # dataclasses look the module up by name
    spec.loader.exec_module(probe)                    # must not raise
    assert "gdstk" not in vars(probe)
    with pytest.raises(ImportError, match="gdstk"):
        probe.read_gds("nonexistent.gds")
    assert probe.rect_spiral(2, 100 * UM, 10 * UM, 5 * UM).length > 0   # rest of the module works


# ---------------------------------------------------------------- T2

def test_mesh_lines_rectangle_edges_exact_and_graded():
    x, y = mesh_lines([RECT], (0.0, -20 * UM, 60 * UM, 20 * UM), base_dx=7 * UM)
    for v in (10 * UM, 40 * UM):
        assert v in x                      # exact float membership, not allclose
    for v in (-5 * UM, 7 * UM):
        assert v in y
    assert x[0] == 0.0 and x[-1] == 60 * UM and y[0] == -20 * UM and y[-1] == 20 * UM
    assert np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)
    assert np.diff(x).max() <= 7 * UM * (1 + 1e-12) and np.diff(y).max() <= 7 * UM * (1 + 1e-12)
    # neighbour ratio: the peel refinement caps it at 0.95 * 1.5 = 1.425
    # (measured 1.4250000000000669 on this and the spiral geometries)
    assert _max_ratio(x) <= 1.5 and _max_ratio(y) <= 1.5


def test_grade_lines_ratio_bound_on_adversarial_gaps():
    """A 1 nm gap beside a 1 mm one, a 0.3 pm gap that is kept (far above
    merge_tol * span = 1e-15 m) and graded up from, and a 1e-16 m duplicate
    that IS merged: ratio holds and the mandatory lines survive exactly."""
    dup = 0.3e-3 + 1e-16
    assert dup != 0.3e-3                     # distinct floats, below the merge tolerance
    mand = [0.0, 0.3e-3, dup, 0.3e-3 + 1e-9, 0.3e-3 + 1e-9 + 3e-13, 1e-3]
    lines = grade_lines(mand, base_dx=50e-6, ratio=1.5)
    assert _max_ratio(lines) <= 1.5
    assert 0.3e-3 in lines and 0.3e-3 + 1e-9 in lines and 0.3e-3 + 1e-9 + 3e-13 in lines
    assert np.sum(np.abs(lines - 0.3e-3) < 1e-14) == 1    # the duplicate was merged
    assert np.all(np.diff(lines) > 0)
    assert np.diff(lines).max() <= 50e-6 * (1 + 1e-12)
    # the docstring's example of halves below the small neighbour
    assert np.allclose(np.diff(grade_lines([0.0, 1.0, 2.5], 10.0)), [1.0, 0.75, 0.75], rtol=0, atol=1e-15)


def test_grade_lines_random_sweep_terminates_within_budget():
    """40 random mandatory sets (gaps 1 nm..1 mm, base_dx 1..100 um): ratio
    <= 1.5, mandatory lines exact, cells <= base_dx, and well inside the
    split budget (measured <= 12 % over 300 sets; this asserts termination
    without a RuntimeError). Whole sweep measured 0.1 s."""
    rng = np.random.default_rng(1)
    t0 = time.perf_counter()
    worst = 1.0
    for _ in range(40):
        k = int(rng.integers(2, 11))
        gaps = 10 ** rng.uniform(-9, -3, size=k - 1)
        m = np.concatenate([[0.0], np.cumsum(gaps)])
        base = 10 ** rng.uniform(-6, -4)
        lines = grade_lines(m, base)
        worst = max(worst, _max_ratio(lines))
        assert all(v in lines for v in m)
        assert np.diff(lines).max() <= base * (1 + 1e-9)
        assert np.all(np.diff(lines) > 0)
    assert worst <= 1.5
    assert time.perf_counter() - t0 < 5.0


def test_z_lines_interfaces_exact_and_metal_cells():
    st = _stack()
    z = z_lines(st, base_dz=3 * UM, metal_cells=2)
    for lay in st.layers:                # interfaces exact (float membership)
        assert lay.z0 in z and lay.z1 in z
    for lay in st.layers:
        if lay.is_metal:
            assert np.sum((z > lay.z0) & (z < lay.z1)) >= 1  # >= 2 cells across it
    assert np.diff(z).max() <= 3 * UM * (1 + 1e-12)


# ---------------------------------------------------------------- T3

def test_rasterise_rectangle_on_snapped_grid_is_exact():
    st = _stack()
    x, y = mesh_lines([RECT], (0.0, -20 * UM, 60 * UM, 20 * UM), base_dx=7 * UM)
    z = z_lines(st, base_dz=1 * UM)
    r = rasterise({KEY_M: [RECT]}, st, x, y, z, freq=2.4e9)
    assert r.eps_r.shape == (len(x) - 1, len(y) - 1, len(z) - 1)
    assert set(np.unique(r.fill).tolist()) == {0.0, 1.0}
    assert np.array_equal(r.conductor_mask, r.fill == 1.0)
    vol = np.einsum("i,j,k,ijk->", np.diff(x), np.diff(y), np.diff(z), r.fill)
    assert abs(vol / (30 * 12 * 2 * UM ** 3) - 1) < 1e-12
    # metal only inside the M slab, sigma there, background eps elsewhere
    zc = 0.5 * (z[1:] + z[:-1])
    assert not r.conductor_mask[:, :, (zc < 5 * UM) | (zc > 7 * UM)].any()
    assert np.all(r.sigma[r.conductor_mask] == 3e7) and np.all(r.sigma[~r.conductor_mask] == 0)
    assert np.all(r.eps_r[:, :, zc < 10 * UM] == 4.1) and np.all(r.eps_r[:, :, zc > 10 * UM] == 1.0)


# ---------------------------------------------------------------- T4

def test_rasterise_octagon_total_fill_area_exact():
    """Oblique edges: the summed fill area equals the polygon area (exact
    clipping; measured 1.1e-15 relative) -- gate 1e-9."""
    x, y = mesh_lines([OCT], (-80 * UM, -80 * UM, 80 * UM, 80 * UM), base_dx=6 * UM)
    f = fill_fractions([OCT], x, y)
    assert f.min() >= 0.0 and f.max() <= 1.0
    area = float(np.einsum("i,j,ij->", np.diff(x), np.diff(y), f))
    assert abs(area / abs(polygon_area(OCT)) - 1) < 1e-9
    assert ((f > 0) & (f < 1)).any()   # genuinely partial cells exist
    # full 3-D path with a dielectric drawn layer: area-weighted eps
    st = LayerStack((Layer("bg", "dielectric", 0.0, 2 * UM, eps_r=2.0),
                     Layer("d", "dielectric", 0.0, 2 * UM, gds=(1, 0), eps_r=6.0 - 0.5j)))
    r = rasterise({(1, 0): [OCT]}, st, x, y, np.array([0.0, 2 * UM]))
    eps_expected = 2.0 * (1 - f) + (6.0 - 0.5j) * f
    assert np.max(np.abs(r.eps_r[:, :, 0] - eps_expected)) < 1e-12
    assert not r.conductor_mask.any()


def test_rasterise_substrate_loss_folded_into_eps():
    st = sg13g2_stack()
    z = np.array([-4 * UM, -3 * UM])           # inside the epi (eps 11.9, 5 S/m)
    r = rasterise({}, st, np.array([0.0, UM]), np.array([0.0, UM]), z, freq=2.4e9)
    omega = 2 * np.pi * 2.4e9
    assert abs(r.eps_r[0, 0, 0] - (11.9 - 1j * 5.0 / (omega * 8.8541878128e-12))) < 1e-9


# ---------------------------------------------------------------- T5

PAPER = dict(n_turns=3, r_out=218 * UM, width=30 * UM, spacing=14 * UM, lead=40 * UM)


def _check_spiral(sp: Spiral, n_sides: int, prm: dict = PAPER):
    n_turns, w, s = prm["n_turns"], prm["width"], prm["spacing"]
    assert [len(v) for v in (sp.polygons[(134, 0)], sp.polygons[(126, 0)], sp.polygons[(133, 0)])] == [1, 1, 1]
    strip = sp.polygons[(134, 0)][0]
    under = sp.polygons[(126, 0)][0]
    via = sp.polygons[(133, 0)][0]
    for p in (strip, under, via):
        g = shapely.Polygon(p)
        assert g.is_valid and g.is_simple           # no self-intersection (shapely)
        assert polygon_is_simple(p)                 # ... and by the module's own test
        assert polygon_area(p) > 0                  # counter-clockwise
    # analytic centreline: lead + n_sides * n_turns sides + inner extension
    n_lead = 1 if prm["lead"] > 0 else 0
    assert len(sp.segment_lengths) == n_sides * n_turns + 1 + n_lead
    assert abs(sp.segment_lengths.sum() / sp.length - 1) < 1e-15
    ext = w + 0.5 * w * np.tan(np.pi / n_sides)
    assert abs(sp.segment_lengths[-1] / ext - 1) < 1e-14
    assert sp.segment_lengths[n_lead:-1].min() >= w * (1 - 1e-12)     # every side >= width
    # mitred strip area == width * centreline length (exact; measured 7.8e-16 rect, 3.3e-16 octagon)
    assert abs(polygon_area(strip) / (w * sp.length) - 1) < 0.05
    assert abs(polygon_area(strip) / (w * sp.length) - 1) < 1e-12
    assert abs(polygon_area(under) / (w * sp.underpass_length) - 1) < 1e-12
    assert abs(polygon_area(via) / w ** 2 - 1) < 1e-12
    # the via sits inside both the main-layer strip and the underpass
    main = shapely.Polygon(strip)
    v = shapely.Polygon(via)
    assert main.intersection(v).area / v.area > 1 - 1e-12
    assert shapely.Polygon(under).intersection(v).area / v.area > 1 - 1e-12
    # clearance between turns (and from the inner extension to the previous
    # turn) is exactly `spacing`: buffering the strip by just under
    # spacing/2 leaves the channel open (no hole), just over closes it.
    # Holds at 1e-9 relative on all 572 accepted spirals of the sweep below.
    assert _holes(strip, 0.5 * s * (1 - 1e-9)) == 0
    assert _holes(strip, 0.5 * s * (1 + 1e-9)) >= 1
    # both ports on the same y line (the lead end), lead at the outer edge
    assert abs(sp.ports[0][1] - sp.ports[1][1]) < 1e-15
    assert abs(sp.ports[0][0] - (prm["r_out"] - w / 2)) < 1e-15
    # outer extent = r_out on the right-hand vertical side
    assert abs(strip[:, 0].max() - prm["r_out"]) < 1e-15


def test_rect_spiral_paper_parameters():
    sp = rect_spiral(**PAPER)
    _check_spiral(sp, 4)
    # square spiral side lengths: 2 a0 for the first three sides, then -pitch
    # every two, then the 1.5 w inner extension
    a0, p = PAPER["r_out"] - PAPER["width"] / 2, PAPER["width"] + PAPER["spacing"]
    expect = [PAPER["lead"]] + [2 * a0 - p * (max(k - 1, 0) // 2) for k in range(12)] + [1.5 * PAPER["width"]]
    assert np.allclose(sp.segment_lengths, expect, rtol=1e-14, atol=0)
    # every strip edge is axis-aligned -> grid-exact under mesh_lines
    strip = sp.polygons[(134, 0)][0]
    q = np.roll(strip, -1, axis=0)
    assert np.all((strip[:, 0] == q[:, 0]) | (strip[:, 1] == q[:, 1]))
    x, y = mesh_lines(sp.polygons, (-300 * UM, -300 * UM, 300 * UM, 300 * UM), base_dx=20 * UM)
    f = fill_fractions([strip], x, y)
    assert set(np.unique(f).tolist()) == {0.0, 1.0}
    area = float(np.einsum("i,j,ij->", np.diff(x), np.diff(y), f))
    assert abs(area / polygon_area(strip) - 1) < 1e-12
    assert _max_ratio(x) <= 1.5 and _max_ratio(y) <= 1.5


def test_rect_spiral_default_lead_zero():
    """lead=0 (the documented default) puts the outer port at the outer corner."""
    sp = rect_spiral(3, 218 * UM, 30 * UM, 14 * UM)
    a0 = 218 * UM - 15 * UM
    assert sp.ports[0] == (a0, -a0)
    assert len(sp.segment_lengths) == 12 + 1
    assert shapely.Polygon(sp.polygons[(134, 0)][0]).is_valid


def test_octagonal_spiral_paper_parameters():
    sp = octagonal_spiral(**PAPER)
    _check_spiral(sp, 8)
    # regular-octagon side = 2 a tan(pi/8); first 7 sides at apothem a0
    a0 = PAPER["r_out"] - PAPER["width"] / 2
    side = 2 * a0 * np.tan(np.pi / 8)
    assert np.allclose(sp.segment_lengths[1:8], side, rtol=1e-13, atol=0)
    # last 45-degree side absorbs the pitch step: 2 c tan(pi/8) - pitch / sin(pi/4)
    c = a0 - 2 * (PAPER["width"] + PAPER["spacing"])
    assert abs(sp.segment_lengths[-2] / (2 * c * np.tan(np.pi / 8) - 44 * UM / np.sin(np.pi / 4)) - 1) < 1e-13
    # straight sides are grid-exact: every strip edge that is axis-aligned to
    # 1e-15 m is EXACTLY axis-aligned (measured 30 of 54 edges) and sits on
    # a grid line by exact float membership
    strip = sp.polygons[(134, 0)][0]
    q = np.roll(strip, -1, axis=0)
    exact = (strip[:, 0] == q[:, 0]) | (strip[:, 1] == q[:, 1])
    near = (np.abs(strip[:, 0] - q[:, 0]) < 1e-15) | (np.abs(strip[:, 1] - q[:, 1]) < 1e-15)
    assert np.array_equal(exact, near) and exact.sum() == 30 and len(strip) == 54
    x, y = mesh_lines(sp.polygons, (-300 * UM, -300 * UM, 300 * UM, 300 * UM), base_dx=10 * UM)
    for k in np.flatnonzero(exact):
        assert (strip[k, 0] == q[k, 0] and strip[k, 0] in x) or (strip[k, 1] == q[k, 1] and strip[k, 1] in y)
    # rasterised metal area on the graded grid reproduces the strip area
    # (exact clipping; measured -3.4e-15)
    f = fill_fractions([strip], x, y)
    area = float(np.einsum("i,j,ij->", np.diff(x), np.diff(y), f))
    assert abs(area / polygon_area(strip) - 1) < 1e-9
    assert _max_ratio(x) <= 1.5 and _max_ratio(y) <= 1.5
    # grading cost: measured 121 x 159 lines for 60 base cells per axis
    assert len(x) <= 3 * 60 and len(y) <= 3 * 60


def test_spiral_rejects_overfull_geometry():
    with pytest.raises(ValueError):
        rect_spiral(n_turns=6, r_out=100 * UM, width=30 * UM, spacing=14 * UM)
    # octagons whose last 45-degree side would collapse (these used to be
    # emitted as self-intersecting strips): last side 0.828 c - 1.414 pitch
    for n, r in ((4, 150), (3, 120)):          # last side -1.0 um
        with pytest.raises(ValueError, match="last side"):
            octagonal_spiral(n, r * UM, 20 * UM, 10 * UM, lead=50 * UM)
    with pytest.raises(ValueError, match="last side"):    # 7.3 um, shorter than the 20 um strip
        octagonal_spiral(4, 160 * UM, 20 * UM, 10 * UM, lead=50 * UM)
    # analytic threshold for (4 turns, w 20, s 10): last side >= w at r_out >= 175.36 um
    with pytest.raises(ValueError, match="last side"):
        octagonal_spiral(4, 175 * UM, 20 * UM, 10 * UM, lead=50 * UM)
    sp = octagonal_spiral(4, 176 * UM, 20 * UM, 10 * UM, lead=50 * UM)
    assert sp.segment_lengths[-2] >= 20 * UM
    assert shapely.Polygon(sp.polygons[(134, 0)][0]).is_simple
    # a 3-turn octagon with lead=0 has its inner end below the port line
    with pytest.raises(ValueError, match="lead"):
        octagonal_spiral(**{**PAPER, "lead": 0.0})


def test_spiral_parameter_sweep_accepted_means_valid():
    """Every (rect, octagon) x turns x r_out x width x spacing combination is
    either refused with ValueError or yields a simple strip (shapely and
    polygon_is_simple agree), exact area, via containment and clearance ==
    spacing. Measured: 572 accepted / 228 refused in 0.5 s."""
    accepted = refused = 0
    t0 = time.perf_counter()
    for n_sides, gen in ((4, rect_spiral), (8, octagonal_spiral)):
        for n, r, w, s in itertools.product([1, 2, 3, 4, 5], [60, 100, 150, 218, 300],
                                            [5, 10, 20, 30], [2, 5, 10, 14]):
            prm = dict(n_turns=n, r_out=r * UM, width=w * UM, spacing=s * UM, lead=r * UM)
            try:
                sp = gen(**prm)
            except ValueError:
                refused += 1
                continue
            accepted += 1
            _check_spiral(sp, n_sides, prm)
    assert accepted == 572 and refused == 228
    assert time.perf_counter() - t0 < 20.0


def test_polygon_is_simple_agrees_with_shapely():
    """The module's own simplicity test against shapely on a square, a
    bow-tie, and a strip folded onto itself by a too-short middle segment
    (the failure mode the spiral guard prevents)."""
    square = rect_from_bounds(0, 0, 1, 1)
    bowtie = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=float)
    folded = offset_polyline(np.array([[0, 0], [10, 0], [10.5, 0.5], [0, 1]], dtype=float), 2.0)
    fine = offset_polyline(np.array([[0, 0], [10, 0], [14, 4], [0, 8]], dtype=float), 2.0)
    for p, expect in ((square, True), (bowtie, False), (folded, False), (fine, True)):
        assert polygon_is_simple(p) is expect
        assert shapely.Polygon(p).is_simple is expect


# ---------------------------------------------------------------- stack I/O

def test_stack_json_round_trip_and_sg13g2_shape(tmp_path):
    st = sg13g2_stack()
    assert "approx" in st.name
    tm2 = st["TopMetal2"]
    assert tm2.gds == (134, 0) and abs(tm2.thickness - 3 * UM) < 1e-15 and tm2.sigma == 3.05e7
    assert abs(st["TopMetal1"].thickness - 2 * UM) < 1e-15
    assert st["substrate"].eps_r == 11.9 and st["substrate"].sigma == 2.0 and st["epi"].sigma == 5.0
    assert st["oxide"].eps_r == 4.1 and st["passivation"].eps_r == 6.6
    assert len([lay for lay in st.layers if lay.kind == "conductor"]) == 7
    # metals do not overlap each other and sit inside the oxide
    metals = sorted((lay.z0, lay.z1) for lay in st.layers if lay.is_metal)
    assert all(b0 >= a1 - 1e-18 for (_, a1), (b0, _) in zip(metals[:-1], metals[1:]))
    assert metals[0][0] >= 0 and metals[-1][1] <= st["oxide"].z1 + 1e-18
    # JSON round trip through load_stack (complex eps as [re, im])
    doc = st.to_dict()
    doc["layers"][3]["eps_r"] = [4.1, -0.01]
    path = tmp_path / "stack.json"
    path.write_text(json.dumps(doc))
    back = load_stack(path)
    assert [lay.name for lay in back.layers] == [lay.name for lay in st.layers]
    assert back.layers[3].eps_r == 4.1 - 0.01j
    assert all(abs(a.z0 - b.z0) < 1e-18 and abs(a.thickness - b.thickness) < 1e-18
               for a, b in zip(back.layers, st.layers))
    # micron-unit dict
    um_doc = {"unit": 1e-6, "layers": [{"name": "m", "kind": "conductor", "z0": 1, "thickness": 2,
                                        "gds": [1, 0], "sigma": 1e7}]}
    assert abs(load_stack(um_doc)["m"].z1 - 3 * UM) < 1e-18
    with pytest.raises(ValueError):
        Layer("bad", "metal", 0.0, 1.0)
