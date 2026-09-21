"""The single WR-90 inductive iris case reads ONE realized-geometry helper.

WHY THIS FILE EXISTS (#931 lattice ownership contract, inventory crossval-D).
Before the contract the two merged WR-90 iris cases ran OPPOSITE iris-thickness
conventions and the suite stayed green, because each case asserted its own rule
against its own drawing.

Under the contract there is one realization rule
(:func:`rfx.boundaries.pec.realized_pec_edge_masks`) and one reader
(``validation/crossval/_wr90_iris_realized.py``). These tests are the cheap
check that the case actually uses it and that the reader still refuses a
non-contiguous aperture.

2026-09-21: the five-iris band-pass filter case was removed. The two arms that
needed its builder -- the same-drawn-box thickness comparison between the two
cases, and the FDFD comparator's Dirichlet-block mapping -- went with it. What
is left reads the single-iris case only.

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
def test_the_case_realizes_the_thickness_of_the_drawn_box(
        monkeypatch, cv18, realized, t_cells):
    """One drawn box, one realized thickness, read from the shared helper.

    This is the test the pre-#931 tree could not have had: the two WR-90 iris
    cases did not share a definition of "how thick is this iris". The case now
    reads its wall planes from the shared function, so the answer is the drawn
    count.
    """
    _stub_s_matrix(monkeypatch, cv18, len(cv18.FREQS))
    row = cv18.run_point(cv18.D_WORST, cv18.COARSE_CELLS, t_cells=t_cells)
    assert row["realized_thickness_cells"] == t_cells
    lo, hi = row["iris_wall_nodes"]
    assert hi - lo == t_cells, (lo, hi)


def test_one_cell_pec_volume_stands_two_walls_in_the_builder(
        monkeypatch, cv18, realized):
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
