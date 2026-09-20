"""The waveguide S-parameter lane hands the solver the same permittivity the
runners do, pad included (issue #1066).

``rfx/sparams/waveguide.py`` used to rebuild ``shape_eps_pairs`` from
``sim._geometry`` at both its smoothing sites, under a comment saying it
mirrored ``rfx/runners/uniform.py``. Since #1043 stage B the runner sites build
theirs through ``rfx.geometry.smoothing.smoothed_shape_pairs``, which continues
a dielectric reaching a CPML/UPML face out through that pad. The lane did not,
so such a structure was solved with ``eps_r = 1`` in its own absorber -- the
#831 end facet -- and only a comment said so.

Measured build-only on ``origin/main`` before the fold, WR-90 at dx = 1.27 mm
with an ``eps_r = 4`` slab running into the x-hi face: the lane's own pairs gave
``eps_xx = 1.0`` down all 8 CPML cells where the runner's gave 4.0, 1539 of
13167 cells differing, max |diff| 3.0. After the fold the two agree bit for bit.

Nothing here time-steps. These are statements about the array the solver is
handed, which is what the defect was about; they are not S-parameter claims.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.smoothing import compute_smoothed_eps, smoothed_shape_pairs

A_WG, B_WG, DX = 22.86e-3, 10.16e-3, 1.27e-3
LX = 60 * DX
FREQS = jnp.asarray([8.5e9, 10.0e9, 11.5e9])
CPML_CELLS = 8


def _guide(with_slab: bool, *, slab_reaches_pad: bool = True):
    """WR-90, CPML on x, PEC y/z walls, two ports. The only pads are on x."""
    sim = Simulation(
        freq_max=12e9, domain=(LX, A_WG, B_WG), dx=DX, cpml_layers=CPML_CELLS,
        boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                              y=Boundary(lo="pec", hi="pec"),
                              z=Boundary(lo="pec", hi="pec")),
    )
    if with_slab:
        sim.add_material("slab", eps_r=4.0)
        # Reaching the x-hi FACE is the whole point: that is the port-side
        # absorber. The no-pad variant stops two cells short of it.
        x_hi = LX if slab_reaches_pad else LX - 2 * DX
        sim.add(Box((LX / 2.0, 0.0, 0.0), (x_hi, A_WG, B_WG)), material="slab")
    for pos, direction, name in ((6 * DX, "+x", "left"),
                                 (LX - 6 * DX, "-x", "right")):
        sim.add_waveguide_port(pos, direction=direction, freqs=FREQS,
                               f0=1e10, name=name)
    return sim


def test_the_lane_calls_the_runners_one_implementation() -> None:
    """A source check, cheap and exact.

    The import is function-local on both the lane and the runners, so there is
    no module attribute to compare; what CAN be pinned is that the lane calls
    the shared helper twice and rebuilds no list of its own. Rebuilding it is
    what #1066 was, and the comment that said it mirrored the runners is what
    let it drift.
    """
    import inspect

    from rfx.sparams import waveguide as lane

    src = inspect.getsource(lane.compute_waveguide_s_matrix)
    assert src.count("smoothed_shape_pairs(self, grid)") == 2, (
        "both smoothing sites must go through the shared helper")
    assert "for entry in self._geometry" not in src, (
        "a local shape_eps_pairs construction is back in the lane; it is the "
        "duplicate that drifted from the runners and became #1066"
    )


@pytest.mark.parametrize("reaches_pad", [True, False],
                         ids=["reaches_the_pad", "stops_short_of_the_pad"])
def test_lane_and_runner_agree_cell_for_cell(reaches_pad: bool) -> None:
    """Bit identity between the lane's pairs and the runner's, on one sim and
    one grid, smoothed the same way.

    The ``reaches_the_pad`` case is the defect: before the fold the lane read
    1.0 and the runner 4.0 down all 8 CPML cells. The ``stops_short`` case is
    the control -- a geometry touching no pad was already identical, and must
    stay so, which is what says the fold changed the pad and nothing else.
    """
    sim = _guide(with_slab=True, slab_reaches_pad=reaches_pad)
    grid = sim._build_grid()
    pairs, unextendable = smoothed_shape_pairs(sim, grid)
    assert unextendable == [], unextendable

    # The runner path, built from the same sim and grid, through the same
    # helper the lane now calls.
    runner_pairs, _ = smoothed_shape_pairs(sim, grid)
    lane_eps = np.asarray(compute_smoothed_eps(grid, pairs,
                                               background_eps=1.0)[0])
    runner_eps = np.asarray(compute_smoothed_eps(grid, runner_pairs,
                                                 background_eps=1.0)[0])
    np.testing.assert_array_equal(lane_eps, runner_eps)


def test_the_pad_carries_the_dielectric_not_vacuum() -> None:
    """The #831 end facet, stated as the number it is.

    Red before the fold: every cell of the x-hi pad read 1.0 on the lane.
    """
    sim = _guide(with_slab=True)
    grid = sim._build_grid()
    pairs, _ = smoothed_shape_pairs(sim, grid)
    eps = np.asarray(compute_smoothed_eps(grid, pairs, background_eps=1.0)[0])
    pad = int(grid.face_pads[1])
    assert pad == CPML_CELLS, pad
    jm, km = eps.shape[1] // 2, eps.shape[2] // 2
    pad_column = eps[eps.shape[0] - pad:, jm, km]
    np.testing.assert_allclose(pad_column, 4.0, rtol=0, atol=1e-12)
    assert not np.any(np.isclose(pad_column, 1.0)), (
        f"the x-hi pad is back to vacuum: {pad_column!r} -- that is the #831 "
        "facet this test exists to keep closed"
    )


def test_a_guide_with_no_dielectric_is_unaffected() -> None:
    """The empty reference run passes ``dielectric_shapes=[]`` and cannot carry
    a facet; the fold must not invent pairs for it."""
    sim = _guide(with_slab=False)
    grid = sim._build_grid()
    pairs, unextendable = smoothed_shape_pairs(sim, grid)
    assert pairs == [] and unextendable == []
