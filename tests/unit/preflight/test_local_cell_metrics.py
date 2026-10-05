"""G17: preflight verdicts at cells half the boundary width; no FDTD."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.preflight import msl
from rfx.preflight._common import _axis_pad_thickness_m

U = 2.0**-12


def _profile():
    # Span 20 U, with the entire [4 U, 16 U] band at half the boundary cell.
    return np.array([U] * 4 + [U / 2] * 24 + [U] * 4)


@pytest.mark.parametrize("direction", ["+x", "-x", "+y", "-y"])
@pytest.mark.parametrize("tolerance", ["propagation", "width"])
def test_msl_reflector_verdict_uses_feed_cells(direction, tolerance):
    axis = direction[-1]
    prop = "xy".index(axis)
    width = 1 - prop
    refined = axis if tolerance == "propagation" else "xy"[width]
    sim = Simulation(freq_max=20e9, domain=(20 * U, 20 * U, 8 * U),
                     dx=U, boundary="cpml", cpml_layers=2,
                     **{f"d{refined}_profile": _profile()})
    position = (8 * U, 8 * U, 0.0)
    sim.add_msl_port(position=position, width=2 * U, height=2 * U,
                     direction=direction, name="sense", mode="uniform", eps_r_sub=2.0,
                     n_probe_offset=3, n_probe_spacing=2, n_probes=3)
    lo, hi = [7 * U, 7 * U, 0.0], [9 * U, 9 * U, 2 * U]
    if tolerance == "propagation":
        # A same-width conductor starts 0.75 coarse cells downstream.
        lo[prop], hi[prop] = ((8.75 * U, 12 * U) if direction[0] == "+"
                              else (4 * U, 7.25 * U))
    else:
        # It crosses the feed, but differs in width by 0.75 coarse cells.
        lo[prop], hi[prop] = 7 * U, 9 * U
        lo[width], hi[width] = 6.625 * U, 9.375 * U
    sim.add(Box(tuple(lo), tuple(hi)), material="pec")
    grid = sim._build_realized_grid()
    assert grid.boundary_cell(refined, "lo") == U
    assert grid.cells(refined)[grid.index_of(refined, 8 * U)] == U / 2
    record = msl.msl_probe_clearance_for_port(sim, sim._msl_ports[0], grid)
    # With a boundary-sized tolerance this is excluded as the port's own
    # trace, producing 'satisfied' and no reflector. The local cell counts it.
    assert record.status == "insufficient"
    assert record.reflector is not None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        msl.preflight_msl_probe_clearance(sim, warnings)
    assert any(getattr(w.message, "code", None) == "msl_port_geometry"
               and record.reflector in str(w.message) for w in caught)


@pytest.mark.parametrize("axis", list("xyz"))
def test_waveguide_reference_plane_verdict_uses_local_cells(axis):
    sim = Simulation(freq_max=100e9, domain=(20 * U,) * 3, dx=U,
                     boundary="cpml", cpml_layers=2, **{f"d{axis}_profile": _profile()})
    freqs = np.array([80e9])
    for sign, source, reference in (("+", 8, 8), ("-", 12, 12.375)):
        sim.add_waveguide_port(source * U, direction=sign + axis,
                              reference_plane=reference * U, freqs=freqs,
                              ref_offset=1, probe_offset=2)
    grid = sim._build_realized_grid()
    for coordinate in (8 * U, 12.375 * U):
        assert grid.cells(axis)[grid.index_of(axis, coordinate)] == U / 2
    assert grid.boundary_cell(axis, "lo") == U
    # Public preflight builds the actual NU port configs and calls the audit.
    report = sim.preflight_sparameters(calculator="waveguide", include_general=False, normalize=True)
    hits = [issue for issue in report.issues
            if issue.code == "port_index_mirror_asymmetry"
            and "reference plane (" in str(issue)]
    assert len(hits) == 1
    assert f"tolerance {U / 4 * 1e3:.3g} mm" in str(hits[0])
    # The 0.375 U residual is above half a local cell (0.25 U), but below
    # half the old boundary cell (0.5 U), so restoring the scalar is silent.


@pytest.mark.parametrize("axis", list("xyz"))
@pytest.mark.parametrize("side", ["lo", "hi"])
def test_graded_absorber_depth_counts_pad_cells_not_the_fine_port_cell(axis, side):
    sim = Simulation(freq_max=20e9, domain=(20 * U,) * 3, dx=U,
                     boundary="cpml", cpml_layers=2,
                     **{f"d{axis}_profile": _profile()})
    grid = sim._build_realized_grid()
    assert grid.cells(axis)[grid.index_of(axis, 8 * U)] == U / 2
    assert _axis_pad_thickness_m(grid, "xyz".index(axis), side) == 2 * U
