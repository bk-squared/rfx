"""The three coax lanes stamp exactly their declared axial cell range, without FDTD."""
from __future__ import annotations

import numpy as np
import pytest

from rfx.api import Simulation
from rfx.sources import coaxial_port
from tests._coax_msl_instrument_fixture import (
    GROUND,
    build_instrument_junction,
    instrument_kwargs,
)


class _Stamped(Exception):
    """Stop before time stepping after inspecting the real stamp."""


@pytest.mark.parametrize("lane,rung", [
    (lane, rung) for lane in ("reflection", "two_port") for rung in (4, 6, 9)
] + [("transition", 4)])
def test_coax_line_fills_its_declared_range(monkeypatch, lane, rung):
    import rfx.simulation

    stamp = coaxial_port.stamp_coaxial_line

    def must_not_run(*args, **kwargs):
        pytest.fail("the extent contract must stop before time stepping")

    monkeypatch.setattr(rfx.simulation, "run", must_not_run)

    def inspect_stamp(grid, materials, **kw):
        stamped, _, cells = stamp(grid, materials, **kw)
        expected = np.arange(kw["z_lo_index"], kw["z_hi_index"] + 1)
        # Read the arrays that the caller receives, not the cylinder bounds.
        # Both pin and shell must span the range independently; checking their
        # union alone could hide a missing end on one of the conductors.
        x = (np.arange(grid.shape[0]) - grid.pad_x_lo) * float(grid.dx)
        y = (np.arange(grid.shape[1]) - grid.pad_y_lo) * float(grid.dx)
        radius = np.hypot(x[:, None] - kw["center_xy"][0],
                          y[None, :] - kw["center_xy"][1])
        pin = cells & (radius <= kw["pin_radius"])[:, :, None]
        shell = cells & (radius > kw["outer_radius"])[:, :, None]
        fill = np.asarray(stamped.eps_r) != np.asarray(materials.eps_r)
        # Independent physical layout oracle: do not derive these planes from
        # the stamper's bounds (a caller can pass a self-consistent wrong range).
        dz = float(grid.dx)
        if lane == "reflection":
            end_nodes = (4 * dz, None)  # DUT reference plane, default offset.
        elif lane == "two_port":
            feed_bottom = 3 * dz
            feed_top = (grid.shape[2] - grid.pad_z_hi - 3 - grid.pad_z_lo) * dz
            # The declared line extends one cell below the bottom feed and
            # through the cell one above the top feed (two nodes above it).
            end_nodes = (feed_bottom - dz, feed_top + 2 * dz)
        else:
            end_nodes = (None, GROUND)  # Junction node, below the substrate.
        for name, mask in (("pin", pin), ("shell", shell), ("fill", fill)):
            indices = np.flatnonzero(mask.any(axis=(0, 1)))
            np.testing.assert_array_equal(indices, expected, err_msg=f"{lane}: {name}")
            realized_nodes = ((indices[0] - grid.pad_z_lo) * dz,
                              (indices[-1] + 1 - grid.pad_z_lo) * dz)
            for realized, plane in zip(realized_nodes, end_nodes):
                if plane is not None:
                    assert realized == pytest.approx(plane, abs=dz * 1e-6), (
                        f"{lane}: {name} end node {realized} != physical plane {plane}"
                    )
            # No holes in any occupied axial column, including the endpoints.
            footprint = mask.any(axis=2)
            np.testing.assert_array_equal(mask[:, :, expected],
                                          np.repeat(footprint[:, :, None],
                                                    len(expected), axis=2))
        raise _Stamped

    monkeypatch.setattr(coaxial_port, "stamp_coaxial_line", inspect_stamp)
    if lane == "transition":
        sim = build_instrument_junction()

        def call():
            return sim.compute_coax_msl_transition(**instrument_kwargs(1))
    else:
        dx = (coaxial_port.SMA_OUTER_RADIUS - coaxial_port.SMA_PIN_RADIUS) / rung
        sim = Simulation(freq_max=40e9, domain=(0.008, 0.008, 0.020),
                         boundary="cpml", dx=dx)
        sim.add_coaxial_port((0.004, 0.004, 0.010), face="top", pin_length=5e-3)
        if lane == "reflection":
            def call():
                return sim.compute_coaxial_line_reflection(
                    termination="open", n_steps=1, freqs=np.array([8e9]), probe_count=3)
        else:
            def call():
                return sim.compute_coaxial_two_port(
                    n_steps=1, freqs=np.array([8e9]), probe_count=3,
                    probe_start_cells=4, probe_spacing_cells=2)
    with pytest.raises(_Stamped):
        call()
