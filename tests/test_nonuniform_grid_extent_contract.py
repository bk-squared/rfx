"""The NU grid must realize the extent it was asked for (#562).

`NonUniformGrid` allocated one E-node per profile cell, but the non-uniform
Yee stencil zeroes the last cell's H term (`inv_d_h[N-1] = 0`), so an
N-cell profile realized only N-1 usable gaps: the wall-to-wall extent came
out `sum(d) - d[-1]`. Independently, `coords_from_nonuniform_grid` placed
sample *i* at the cell CENTRE while the E-nodes the stencil differences sit
at cell EDGES (`inv_d_e[i] = 2/(d[i-1]+d[i])` is the dual spacing of an
edge-centred node), and while `coords_from_uniform_grid` returns edges.

Measured consequence before the fix (a=45, b=39 cells at dx=1 mm, PEC box,
TM110 by harminv): the uniform builder read -0.007 % against the closed
form, the NU builder +2.469 % — and -0.008 % against the closed form
evaluated at the extent it had actually built. Both solvers were accurate;
only one built the requested cavity.

These are grid-construction checks: no FDTD, no fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx.api import Simulation
from rfx.geometry.rasterize_grid import (coords_from_nonuniform_grid,
                                         coords_from_uniform_grid)

A = 22.86e-3
DX = 0.254e-3
NB = 8


def _sim(**kw):
    return Simulation(freq_max=13e9, domain=(A, A, NB * DX),
                      boundary="pec", dx=DX, **kw)


def _spans(sim, nonuniform):
    grid = (sim._build_nonuniform_grid() if nonuniform else sim._build_grid())
    coords = (coords_from_nonuniform_grid(grid) if nonuniform
              else coords_from_uniform_grid(grid))
    out = []
    for axis in range(3):
        v = np.asarray(coords[axis], dtype=float)
        out.append((len(v), float(v[0]), float(v[-1] - v[0])))
    return grid.shape, out


REQUESTED = (A, A, NB * DX)

# rfx grids are float32 by design, so coordinates carry ~1e-7 relative
# representation error (2.7e-9 m on a 22.86 mm span). The tolerances below are
# set just above that floor and are still 1e-5 of a cell — four orders tighter
# than the one-cell defect this file pins, and tighter than any physical
# registration question.
_ABS_TOL = 1e-8


@pytest.mark.parametrize("profiles", [
    pytest.param({}, id="uniform-builder"),
    pytest.param({"dx_profile": np.full(90, DX)}, id="dx-only"),
    pytest.param({"dy_profile": np.full(90, DX)}, id="dy-only"),
    pytest.param({"dz_profile": np.full(NB, DX)}, id="dz-only"),
    pytest.param({"dx_profile": np.full(90, DX),
                  "dy_profile": np.full(90, DX),
                  "dz_profile": np.full(NB, DX)}, id="all-three"),
])
def test_realized_extent_equals_requested_domain(profiles):
    """Wall-to-wall extent == requested domain, on every axis, either builder.

    A PEC face is enforced at the outermost E-node, so the span between the
    first and last node IS the electrical cavity/guide dimension. It has to
    be what the caller asked for: at WR-90's 0.254 mm this defect cost one
    cell = 1.11 % of the guide width = +37 MHz of TE101 centre frequency.
    """
    shape, axes = _spans(_sim(**profiles), nonuniform=bool(profiles))
    for name, requested, (n, first, span) in zip("xyz", REQUESTED, axes):
        assert first == pytest.approx(0.0, abs=1e-12), (
            f"{name}: first node at {first * 1e3:.4f} mm, not on the domain "
            f"face — geometry and the PEC wall disagree by that offset")
        assert span == pytest.approx(requested, rel=0, abs=_ABS_TOL), (
            f"{name}: realized {span * 1e3:.4f} mm vs requested "
            f"{requested * 1e3:.4f} mm "
            f"(deficit {(requested - span) / DX:+.2f} cells)")
        assert n == round(requested / DX) + 1, (
            f"{name}: {n} nodes for {round(requested / DX)} cells — a cell "
            f"count of N needs N+1 bounding nodes")


def test_uniform_profile_through_the_nu_builder_reduces_to_the_uniform_grid():
    """A uniform mesh expressed as profiles must BE the uniform mesh.

    This is the reduction property that makes NU-vs-uniform comparisons
    meaningful, and the one that failed silently: the shapes differed by
    one sample per axis and every coordinate by half a cell, so the two
    builders described different structures for identical input. A
    committed accuracy test read +2.4 % on both a graded and an ungraded
    profile and attributed the shared bias to "the standard coarse-Yee
    PEC-cavity extent-convention" — both legs had gone through this
    builder.
    """
    su = _sim()
    sn = _sim(dx_profile=np.full(90, DX), dy_profile=np.full(90, DX),
              dz_profile=np.full(NB, DX))
    shape_u, axes_u = _spans(su, nonuniform=False)
    shape_n, axes_n = _spans(sn, nonuniform=True)
    assert shape_n == shape_u, (shape_n, shape_u)

    cu = coords_from_uniform_grid(su._build_grid())
    cn = coords_from_nonuniform_grid(sn._build_nonuniform_grid())
    for axis, name in enumerate("xyz"):
        vu = np.asarray(cu[axis], dtype=float)
        vn = np.asarray(cn[axis], dtype=float)
        assert vn.shape == vu.shape, (name, vn.shape, vu.shape)
        worst = float(np.max(np.abs(vn - vu)))
        assert worst < _ABS_TOL, (
            f"{name}: NU coordinates differ from the uniform ones by up to "
            f"{worst * 1e3:.4f} mm ({worst / DX:.2f} cells) on an identical "
            f"mesh")


def test_graded_profile_realizes_its_own_sum():
    """The contract has to hold for a genuinely graded profile too, where
    the dropped cell was the (largest) boundary cell rather than a
    representative one."""
    fine, coarse = DX / 2, DX
    prof = np.array([coarse] * 10 + [fine] * 20 + [coarse] * 10)
    total = float(np.sum(prof))
    sim = Simulation(freq_max=13e9, domain=(total, A, NB * DX),
                     boundary="pec", dx=coarse, dx_profile=prof)
    coords = coords_from_nonuniform_grid(sim._build_nonuniform_grid())
    x = np.asarray(coords[0], dtype=float)
    assert float(x[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(x[-1] - x[0]) == pytest.approx(total, abs=_ABS_TOL)
    assert len(x) == len(prof) + 1
    # every interior node sits on a cumulative cell edge of the profile
    edges = np.concatenate([[0.0], np.cumsum(prof)])
    assert np.max(np.abs(x - edges)) < _ABS_TOL


def test_extent_and_pad_symmetry_hold_with_cpml_pads():
    """The extent contract must hold with absorber pads too (#562 review F6).

    The other cases are all PEC-bounded (pad = 0), so this one covers the
    open-domain half of the convention. Of the assertions below only the
    INTERIOR SPAN is tied to #562; the pad-symmetry ones are invariants that
    held before it as well — measured across the merge for the same request
    (60 cells, 8 CPML layers):

        pre-#562:  nx 76, interior span 59.000 cells, pad depths 8.000/8.000
        post-#562: nx 77, interior span 60.000 cells, pad depths 8.000/8.000

    I claimed in the #564 thread that the hi face had carried one physical
    cell fewer and that this test witnessed the correction. **It did not, and
    the underlying claim does not reproduce**: the absorber was symmetric on
    both sides of the merge. Pinning the invariants is still worth doing —
    they are cheap and they are load-bearing for CPML behaviour — but they are
    labelled here as invariants, not as evidence of a change.
    """
    prof = np.full(60, DX)
    sim = Simulation(freq_max=13e9, domain=(60 * DX, A, NB * DX),
                     dx=DX, cpml_layers=8, dx_profile=prof)
    grid = sim._build_nonuniform_grid()
    coords = coords_from_nonuniform_grid(grid)
    x = np.asarray(coords[0], dtype=float)

    # interior span: first interior node at 0, last interior node on the far face
    assert float(x[grid.pad_x_lo]) == pytest.approx(0.0, abs=1e-12)
    interior_last = grid.nx - grid.pad_x_hi - 1
    assert float(x[interior_last] - x[grid.pad_x_lo]) == pytest.approx(
        60 * DX, abs=_ABS_TOL), (
        f"interior span {float(x[interior_last] - x[grid.pad_x_lo]) * 1e3:.4f} mm "
        f"vs requested {60 * DX * 1e3:.4f} mm")

    # pad symmetry: the absorber occupies the same physical depth on both faces
    lo_depth = float(x[grid.pad_x_lo] - x[0])
    hi_depth = float(x[grid.nx - 1] - x[interior_last])
    assert lo_depth == pytest.approx(hi_depth, abs=_ABS_TOL), (
        f"absorber depth lo {lo_depth * 1e3:.4f} mm vs hi {hi_depth * 1e3:.4f} mm "
        "— the CPML ramps are not mirror images")
    assert lo_depth == pytest.approx(8 * DX, abs=_ABS_TOL)


def test_cpml_sigma_ramps_are_mirror_images_across_the_two_faces():
    """The two faces' sigma ramps are mirror images — an INVARIANT, not a
    witness (#568 item 2, resolved by measurement).

    #568 asked for this assertion as the direct witness for a hi-face
    realignment. It is not one: run against the pre-#562 commit it PASSES
    (measured, `bb61494^`), because both faces build sigma from the same
    n_layers and the same boundary cell size, so the arrays are mirror images
    by construction regardless of how many usable cells sit behind them. The
    same measurement shows the premise did not hold either — pad depths were
    8.000/8.000 cells before and after — so there is no realignment to
    witness, and the "unclaimed improvement" recorded in the #564 review, and
    repeated by me in that PR's body, is withdrawn.

    Keeping the assertion anyway: it is cheap, it is true, and an asymmetric
    ramp would be a real defect if a future change made the two faces derive
    sigma independently (the NU z-faces already do use independent cell
    sizes). The non-vacuity check below matters for exactly that reason — a
    constant ramp would satisfy any mirror test.
    """
    from rfx.boundaries.cpml import init_cpml

    prof = np.full(60, DX)
    sim = Simulation(freq_max=13e9, domain=(60 * DX, A, NB * DX),
                     dx=DX, cpml_layers=8, dx_profile=prof)
    grid = sim._build_nonuniform_grid()
    params, _state = init_cpml(grid)

    lo = np.asarray(params.x_lo.sigma, dtype=float)
    hi = np.asarray(params.x_hi.sigma, dtype=float)
    # non-vacuity: a constant ramp would satisfy any mirror test
    assert lo.size == 8, lo.size
    assert float(lo.max()) > 10.0 * max(float(lo.min()), 1e-30), (
        "sigma ramp is nearly constant — the mirror comparison below would be "
        "vacuous", lo)
    assert np.allclose(lo, hi[::-1], rtol=1e-6, atol=0.0), (
        "x-lo and x-hi CPML sigma ramps are not mirror images", lo, hi[::-1])
    # the boundary cell sizes the two faces derive sigma_max from must match too
    assert float(params.dx_x_lo) == pytest.approx(float(params.dx_x_hi), abs=1e-15)
