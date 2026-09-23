"""The design box fence counts a port's EDGES, not its end nodes (#1179).

A wire port declared from node ``n0`` to node ``n1`` on its component's axis
drives the E edges ``n0 .. n1 - 1``: node ``n1`` is the far terminal, the
place the current arrives, and the plane through it carries no port cell.
That is the one spelling in :func:`rfx.sources.sources.wire_port_edge_span`,
which both runners rasterize with; the step-level fence
(``rfx.simulation._resolve_design_box``) reads the port's ``live_cells`` on
the uniform lane and, on the graded lane, the mid cell plus the excited edges
through its ``sources`` entry.

The declaration-level fence in ``_resolve_design_box_override`` built the
port's index range from its two declared NODES and refused any design box
meeting ``[n0, n1]`` inclusive. So a design box on the plane of the far
terminal — a patch fed from below by a via, the box being the patch metal —
was refused although no port cell is in it.

What is pinned here:

* a one-cell-thick design box on the ez port's END-node plane is ACCEPTED and
  runs;
* a box on the port's LAST edge cell (one plane lower) is refused;
* a box on a mid edge cell is refused;
* the same two rows for an ``ex`` port with an x extent — the ``- 1`` is on
  the extent axis, whichever axis that is;
* a SUB-CELL extent (both ends snapping to one node) drives the one edge on
  the side the extent occupies: the box holding that edge is refused, the box
  on the node's own plane is accepted. ``hi - 1`` cannot tell those apart;
  reading :func:`wire_port_edge_span` can.

Mutation: putting the node-inclusive comparison back reds the first test and
leaves the refusals green.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation

F0 = 10e9
DX = 1e-3
CPML = 5
DOMAIN = (20e-3, 20e-3, 16e-3)

# Board: a ground sheet on interior node plane 4, an ez wire port from that
# plane up three cells to the "patch" plane at z = 7 mm.
GROUND_PLANE = 4
PORT_XY = (10e-3, 10e-3)
PORT_Z0 = 4e-3
PORT_EXTENT = 3e-3
PATCH_Z = PORT_Z0 + PORT_EXTENT          # the far terminal's plane
LAST_EDGE_Z = PATCH_Z - DX               # the port's last driven edge
MID_EDGE_Z = PORT_Z0 + DX                # an edge inside the port

# In-plane footprint of the design box, four cells each way around the feed.
BOX_XY_LO = (PORT_XY[0] - 2 * DX, PORT_XY[1] - 2 * DX)
BOX_XY_HI = (PORT_XY[0] + 2 * DX, PORT_XY[1] + 2 * DX)

N_STEPS = 40


def _sim():
    sim = Simulation(freq_max=2 * F0, domain=DOMAIN, dx=DX,
                     boundary="cpml", cpml_layers=CPML)
    sim.add_pinned_sheet(plane_index=GROUND_PLANE,
                         i_range=(4, 16), j_range=(4, 16), normal_axis=2)
    sim.add_port(position=(PORT_XY[0], PORT_XY[1], PORT_Z0), component="ez",
                 extent=PORT_EXTENT, impedance=50.0, direction="-x",
                 waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    return sim


def _box(z_plane):
    """One-cell-thick design box on the z plane through ``z_plane``."""
    return ((BOX_XY_LO[0], BOX_XY_LO[1], z_plane),
            (BOX_XY_HI[0], BOX_XY_HI[1], z_plane))


def _sheet_sigma(sim, box):
    """``(eps_r, (sigma_x, sigma_y, sigma_z))`` for the box's realized cells.

    The sheet carries 1e5 S/m along its two in-plane edges and nothing
    through its thickness — the per-component conductivity this fence sits
    in front of.
    """
    grid = sim._build_grid()
    lo = grid.position_to_index(box[0])
    hi = grid.position_to_index(box[1])
    shape = tuple(int(hi[d]) - int(lo[d]) + 1 for d in range(3))
    eps = jnp.ones(shape, jnp.float32)
    in_plane = jnp.full(shape, 1.0e5, jnp.float32)
    through = jnp.zeros(shape, jnp.float32)
    return eps, (in_plane, in_plane, through)


def _forward(sim, box):
    eps, sigma = _sheet_sigma(sim, box)
    return sim.forward(design_box=box, design_eps_override=eps,
                       design_sigma_override=sigma,
                       n_steps=N_STEPS, checkpoint=False,
                       skip_preflight=True)


def test_port_node_indices_are_the_ones_this_file_assumes():
    """The fixture's planes are the port's end node / last edge / mid edge.

    Without this the three tests below could all be measuring the same cell
    on some future grid and still pass.
    """
    sim = _sim()
    grid = sim._build_grid()
    k0 = int(sim._design_box_index_of(grid, (0.0, 0.0, PORT_Z0))[2])
    k1 = int(sim._design_box_index_of(grid, (0.0, 0.0, PATCH_Z))[2])
    assert k1 - k0 == 3, (
        f"the port was declared over 3 cells but rasterizes to {k1 - k0}")
    assert int(sim._design_box_index_of(
        grid, (0.0, 0.0, LAST_EDGE_Z))[2]) == k1 - 1
    assert int(sim._design_box_index_of(
        grid, (0.0, 0.0, MID_EDGE_Z))[2]) == k0 + 1


def test_a_box_on_the_ports_end_node_plane_is_accepted():
    """The patch plane is the port's far terminal and holds no port cell."""
    sim = _sim()
    result = _forward(sim, _box(PATCH_Z))
    ts = np.asarray(result.time_series)
    assert np.all(np.isfinite(ts)), "the accepted run did not stay finite"


def test_a_box_on_the_ports_last_edge_cell_is_refused():
    sim = _sim()
    with pytest.raises(ValueError, match="holds port cells"):
        _forward(sim, _box(LAST_EDGE_Z))


def test_a_box_on_a_mid_edge_cell_of_the_port_is_refused():
    sim = _sim()
    with pytest.raises(ValueError, match="holds port cells"):
        _forward(sim, _box(MID_EDGE_Z))


# ---- the same rule on another axis, and on a sub-cell extent ----------------

PORT_X0 = 4e-3
PORT_YZ = (10e-3, 7e-3)


def _sim_x_port():
    """An ``ex`` wire port from x = 4 mm three cells along +x."""
    sim = Simulation(freq_max=2 * F0, domain=DOMAIN, dx=DX,
                     boundary="cpml", cpml_layers=CPML)
    sim.add_pinned_sheet(plane_index=GROUND_PLANE,
                         i_range=(4, 16), j_range=(4, 16), normal_axis=2)
    sim.add_port(position=(PORT_X0, PORT_YZ[0], PORT_YZ[1]), component="ex",
                 extent=PORT_EXTENT, impedance=50.0, direction="+y",
                 waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    return sim


def _box_x(x_plane):
    """One-cell-thick design box on the x plane through ``x_plane``."""
    return ((x_plane, PORT_YZ[0] - 2 * DX, PORT_YZ[1] - 2 * DX),
            (x_plane, PORT_YZ[0] + 2 * DX, PORT_YZ[1] + 2 * DX))


def test_x_extent_port_end_node_plane_is_accepted_and_last_edge_refused():
    sim = _sim_x_port()
    result = _forward(sim, _box_x(PORT_X0 + PORT_EXTENT))
    assert np.all(np.isfinite(np.asarray(result.time_series)))
    sim = _sim_x_port()
    with pytest.raises(ValueError, match="holds port cells"):
        _forward(sim, _box_x(PORT_X0 + PORT_EXTENT - DX))


SUBCELL_Z = 8e-3
SUBCELL_EXTENT = -0.4e-3        # both ends snap to the node at 8 mm


def _sim_subcell_port():
    sim = Simulation(freq_max=2 * F0, domain=DOMAIN, dx=DX,
                     boundary="cpml", cpml_layers=CPML)
    sim.add_pinned_sheet(plane_index=GROUND_PLANE,
                         i_range=(4, 16), j_range=(4, 16), normal_axis=2)
    sim.add_port(position=(PORT_XY[0], PORT_XY[1], SUBCELL_Z), component="ez",
                 extent=SUBCELL_EXTENT, impedance=50.0, direction="-x",
                 waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    return sim


def test_subcell_extent_refuses_the_edge_it_occupies_not_the_node_plane():
    """Both ends snap to node 8 mm; the extent lies below it, so the port
    drives the edge from 7 mm to 8 mm (z cell 7 mm) and no other."""
    from rfx.sources.sources import wire_port_edge_span
    sim = _sim_subcell_port()
    grid = sim._build_grid()
    k = int(sim._design_box_index_of(grid, (0.0, 0.0, SUBCELL_Z))[2])
    k_end = int(sim._design_box_index_of(
        grid, (0.0, 0.0, SUBCELL_Z + SUBCELL_EXTENT))[2])
    assert k_end == k, "the fixture's extent must snap both ends to one node"
    lo, hi = wire_port_edge_span(grid, 2, k, k, SUBCELL_Z,
                                 SUBCELL_Z + SUBCELL_EXTENT)
    assert (lo, hi) == (k - 1, k - 1)
    with pytest.raises(ValueError, match="holds port cells"):
        _forward(sim, _box(SUBCELL_Z - DX))
    sim = _sim_subcell_port()
    result = _forward(sim, _box(SUBCELL_Z))
    assert np.all(np.isfinite(np.asarray(result.time_series)))
