"""cv18 and cv19 read ONE realized-geometry helper, and it agrees with the comparator.

WHY THIS FILE EXISTS (#931 lattice ownership contract, inventory crossval-D).
Before the contract the two merged WR-90 iris cases ran OPPOSITE iris-thickness
conventions and the suite stayed green: case 18 fed its oracle ``t_c*dx`` while
case 19 drew ``round(t/dx) + 1`` and fed ``(t_c - 1)*dx``, and case 19's own
fixture recorded the disagreement as an unresolved half-cell ambiguity and
gated around it. Nothing failed, because each case asserted its own rule
against its own drawing.

Under the contract there is one realization rule
(:func:`rfx.boundaries.pec.realized_pec_edge_masks`) and one reader for these
two cases (``validation/crossval/_wr90_iris_realized.py``). These tests are the
cheap check that both cases actually use it and get the same answer for the
same drawn box, and that the independent FDFD comparator's Dirichlet-block
convention maps onto it as the identity.

Build-time only: no time stepping anywhere in this file.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CROSSVAL = os.path.join(_REPO, "validation", "crossval")
A_WR90 = 22.86e-3


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cv18():
    return _load("_cv18_shared", os.path.join(_CROSSVAL, "18_wr90_iris_modematch.py"))


@pytest.fixture(scope="module")
def cv19():
    return _load("_cv19_shared",
                 os.path.join(_CROSSVAL, "19_wr90_iris_filter_aghanim.py"))


@pytest.fixture(scope="module")
def realized():
    sys.path.insert(0, _CROSSVAL)
    return _load("_wr90_iris_realized_test",
                 os.path.join(_CROSSVAL, "_wr90_iris_realized.py"))


def _stub_s_matrix(monkeypatch, cv18, n_freq):
    """Let cv18's run_point build and assert without solving."""
    from rfx.api import Simulation

    class _Res:
        s_params = np.zeros((2, 2, n_freq), dtype=complex)

    monkeypatch.setattr(Simulation, "compute_waveguide_s_matrix",
                        lambda self, **kw: _Res(), raising=True)


@pytest.mark.parametrize("t_cells", [1, 2, 4])
def test_both_cases_realize_the_same_thickness_for_the_same_drawn_box(
        monkeypatch, cv18, cv19, realized, t_cells):
    """One drawn box, two cases, one realized thickness.

    This is the test the pre-#931 tree could not have had: the two cases did
    not share a definition of "how thick is this iris", so there was nothing
    to compare. Both now read wall planes from the same function, so the
    comparison is well posed and the answer is the drawn count.
    """
    _stub_s_matrix(monkeypatch, cv18, len(cv18.FREQS))
    row = cv18.run_point(cv18.D_WORST, cv18.COARSE_CELLS, t_cells=t_cells)
    assert row["realized_thickness_cells"] == t_cells
    lo, hi = row["iris_wall_nodes"]
    assert hi - lo == t_cells, (lo, hi)

    # the same drawn thickness, built by case 19's builder at its coarse rung
    geo = cv19.rasterized_geometry(cv19.COARSE_CELLS, allow_asymmetric=True)
    geo = dict(geo)
    geo["t_cells"] = t_cells
    geo["thicknesses"] = np.full(5, t_cells * geo["dx"])
    sim, _, _ = cv19.build(geo)
    _, _, x_runs = cv19.raster_assert(sim, geo)
    assert [hi - lo for lo, hi in x_runs] == [t_cells] * 5, x_runs


def test_one_cell_pec_volume_stands_two_walls_in_both_builders(
        monkeypatch, cv18, cv19, realized):
    """The contract's one-cell claim, at the level of the built geometry.

    Design note 20260906 section 1.2: a 1-cell PEC Box is a filled slab with
    two faces, at every thickness, on every axis, with no flag. Case 18's
    ``one_cell_volume_witness`` checks the same thing against physics; this is
    the free build-time version, and it is the one that fails loudly if the
    realization regresses.
    """
    _stub_s_matrix(monkeypatch, cv18, len(cv18.FREQS))
    row = cv18.run_point(cv18.D_WORST, cv18.COARSE_CELLS, t_cells=1)
    lo, hi = row["iris_wall_nodes"]
    assert hi == lo + 1, ("a one-cell PEC volume did not stand two wall "
                          "planes", lo, hi)


def test_realized_reader_refuses_a_non_contiguous_aperture(cv18, realized):
    """The parasitic wall-slot fence (case 18 setup defect 1), on the new reader."""
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box

    cells = cv18.COARSE_CELLS
    dx = A_WR90 / cells
    sim = Simulation(freq_max=13e9, domain=(40 * dx, A_WR90, cv18.B_WR90),
                     dx=dx,
                     boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                                           y=Boundary(lo="pec", hi="pec"),
                                           z=Boundary(lo="pec", hi="pec")),
                     cpml_layers=4)
    # three fins on one plane: two openings, not one
    for y_lo, y_hi in ((0.0, 8 * dx), (12 * dx, 16 * dx), (24 * dx, A_WR90)):
        sim.add(Box((20 * dx, y_lo, -1.0), (22 * dx, y_hi, 1.0)), material="pec")
    grid, edges = realized.realized_edge_masks(sim)
    k = realized.grid_plane(grid, 0, 20)
    with pytest.raises(AssertionError, match="parasitic wall-slot"):
        realized.aperture_walls(edges, 1,
                                (slice(k, k + 1), slice(None), slice(None)))


def test_fdfd_comparator_convention_maps_onto_the_realized_edge_set(cv19):
    """The comparator's Dirichlet block and rfx's realized walls are the same geometry.

    The mapping used to live only in prose and in the caller's arithmetic, and
    the comparator has been bitten by this class once already (an earlier
    revision realized every aperture two cells wide; Richardson cancelled the
    bias, so the extrapolated numbers were right for the wrong per-level
    geometry). Under the contract the mapping is the identity, which is the
    moment to pin it.
    """
    sys.path.insert(0, os.path.join(_CROSSVAL, "comparators"))
    import fdfd_hplane

    geo = cv19.rasterized_geometry(cv19.GATED_CELLS, allow_asymmetric=False)
    e = cv19.measured_electrical_geometry(geo)
    d_e = [int(v) for v in e["electrical_aperture_cells"]]
    cav_e = [int(v) for v in e["electrical_cavity_cells"]]
    th_e = int(e["electrical_thickness_cells"])

    _, _, ctx = fdfd_hplane._assemble(A_WR90, 11.0e9, cv19.GATED_CELLS, 1,
                                      d_e, cav_e, th_e, 45)
    metal = np.asarray(ctx["metal"])
    ix = np.asarray(ctx["ix"])

    # longitudinal: five blocks, each bounded by two Dirichlet planes th_e
    # apart, with cav_e clear planes between consecutive blocks
    occupied = np.flatnonzero(metal.any(axis=0))
    runs = np.split(occupied, np.flatnonzero(np.diff(occupied) != 1) + 1)
    assert len(runs) == 5, runs
    assert [int(r[-1] - r[0]) for r in runs] == [th_e] * 5
    assert [int(runs[i + 1][0] - runs[i][-1]) for i in range(4)] == cav_e

    # transverse: the two bounding Dirichlet planes of each iris are d_e apart,
    # the same number rfx realizes between its two innermost y wall planes
    for run, d_c, (y_lo, y_hi) in zip(runs, d_e, e["aperture_wall_nodes"]):
        col = metal[:, int(run[0])]
        open_nodes = ix[~col]
        assert int(open_nodes[-1] - open_nodes[0]) == d_c - 2, (
            "fdfd open span is not d_c - 2 interior nodes", d_c)
        assert int(open_nodes[0] - 1) == int(ix[col][ix[col] < open_nodes[0]][-1])
        assert int(y_hi - y_lo) == d_c, (y_lo, y_hi, d_c)
