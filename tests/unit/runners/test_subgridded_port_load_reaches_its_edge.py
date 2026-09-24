"""A port's load on the SUBGRIDDED lane reaches its edge whole (#1210).

The subgridded runner stamps a port's equivalent conductivity into the fine
grid's own material arrays (``rfx/runners/subgridded.py``), and that lane has
no port S-parameter fixture of its own. Stamped bare, the #1210 edge average
divides the conductance among the four cells the port edge touches: a 50 ohm
termination presents 200 ohm, and a passive wire port reads S11 = +1/3 where it
must read -1/3. The same defect was live and measured on the graded-mesh lane.

What is asserted is the REALIZED edge conductance -- what the E update
multiplies the port's own component by at the port cell -- against the
conductivity the runner declared. That is the quantity the S-parameter
normalization assumes, and it is one step removed from S11 rather than a
restatement of the stamping code: the mutation arm stamps bare, exactly as the
runner did before, and reads a quarter.

Not an S11 gate: the subgrid S-parameter path needs a driven two-port fixture
inside a refinement region, which does not exist yet. The closed-form S11 for
this class is covered on the uniform lane by
tests/unit/boundaries/test_magnetic_wall_faces_not_shorted.py.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.core.yee import (MaterialArrays, cell_component_e_materials,
                          init_materials)
from rfx.sources.sources import stamp_lumped_sigma

DX_F = 0.25e-3      # a subgridded fine cell, ratio 4 on a 1 mm coarse mesh
SHAPE = (12, 12, 12)
Z0 = 50.0


def _edge_stamp(materials, cell, value):
    """The runner's stamp for a port on Ez (#1210, per component #1236)."""
    return stamp_lumped_sigma(materials, cell, value, "ez")


def _bare_stamp(materials, cell, value):
    """What ``rfx/runners/subgridded.py`` did before #1210."""
    i, j, k = cell
    return materials._replace(sigma=materials.sigma.at[i, j, k].add(value))


@pytest.mark.parametrize("stamp,expect",
                         [(_edge_stamp, 1.0), (_bare_stamp, 0.25)],
                         ids=["edge-owned", "mutation-bare-stamp"])
def test_the_fine_grid_port_conductance_is_realized_whole(stamp, expect):
    cell = (6, 6, 6)
    # The runner's own spelling for a single-cell lumped port on a cubic
    # fine mesh: sigma_port = 1 / (Z0 * dx_f).
    sigma_port = 1.0 / (Z0 * DX_F)

    mats = init_materials(SHAPE)
    mats = stamp(mats, cell, sigma_port)

    # The dielectric background is uniform, so everything the edge average
    # does to sigma here is done to the stamp.
    _, sigma_edge = cell_component_e_materials(mats, cell, "ez")
    realized = float(sigma_edge)

    assert realized == pytest.approx(expect * sigma_port, rel=1e-6), (
        f"the port cell declared sigma = {sigma_port:.4g} S/m and the E update "
        f"multiplies Ez there by {realized:.4g} S/m "
        f"({realized / sigma_port:.3f}x). A quarter is the bare stamp: the "
        f"load spread over the four cells the Ez edge touches, so a "
        f"{Z0:.0f} ohm termination presents {Z0 / (realized / sigma_port):.0f} "
        f"ohm.")


def test_the_stamp_does_not_leak_onto_neighbouring_fine_edges():
    """A lumped load belongs to one edge; averaging would put it on twelve."""
    cell = (6, 6, 6)
    sigma_port = 1.0 / (Z0 * DX_F)
    mats = stamp_lumped_sigma(init_materials(SHAPE), cell, sigma_port, "ez")

    for neighbour in [(6, 7, 6), (6, 6, 7), (7, 6, 6), (6, 5, 6)]:
        _, s = cell_component_e_materials(mats, neighbour, "ez")
        assert float(s) == pytest.approx(0.0, abs=1e-9), (
            f"the port's conductance reached {neighbour}, an edge it does "
            f"not span")


def test_the_wire_port_branch_stamps_every_live_cell():
    """The multi-cell branch: n_cells stamps, each realized whole."""
    sigma_per_cell = 3.0 / (Z0 * DX_F)
    cells = [(6, 6, 5), (6, 6, 6), (6, 6, 7)]
    mats = init_materials(SHAPE)
    for c in cells:
        mats = stamp_lumped_sigma(mats, c, sigma_per_cell, "ez")

    assert isinstance(mats, MaterialArrays)
    assert mats.sigma_lumped is not None
    # The record is per E component (#1236): every stamp on the Ez record.
    sx, sy, sz = mats.sigma_lumped
    assert sx is None and sy is None
    assert np.count_nonzero(np.asarray(sz)) == len(cells)
    for c in cells:
        _, s = cell_component_e_materials(mats, c, "ez")
        assert float(s) == pytest.approx(sigma_per_cell, rel=1e-6)
    # Series conductance of the three-cell wire: each cell carries 3/(Z0*d),
    # three in series gives 1/(Z0*d) over the whole span.
    total = 1.0 / sum(1.0 / sigma_per_cell for _ in cells)
    assert total == pytest.approx(1.0 / (Z0 * DX_F), rel=1e-6)
    assert float(jnp.sum(mats.sigma)) == pytest.approx(
        len(cells) * sigma_per_cell, rel=1e-6)
