"""G17: local rasterization tolerances on graded meshes; no FDTD."""
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


def _msl_screen(profile, axis, feed, direction, box):
    sim = Simulation(freq_max=20e9, domain=(20 * U, 20 * U, 8 * U),
                     dx=U, boundary="cpml", cpml_layers=2,
                     **{f"d{axis}_profile": profile})
    sim.add_msl_port(position=(feed * U, 8 * U, 0.0),
                     width=2 * U, height=2 * U, direction=direction,
                     name="sense", mode="uniform", eps_r_sub=2.0,
                     n_probe_offset=3, n_probe_spacing=2, n_probes=3)
    sim.add(Box(*(tuple(v * U for v in corner) for corner in box)), material="pec")
    return sim, sim._build_realized_grid()


def _screen_record(sim, grid):
    return msl.msl_probe_clearance_for_port(sim, sim._msl_ports[0], grid)


def test_msl_grading_step_mirror_verdicts():
    # Reviewer B: each port faces a fine cell at a coarse/fine step.
    # The profiles and conductors mirror about x=10U; the port is off it.
    profile = np.array([U] * 12 + [U / 2] * 8 + [U] * 4)
    cases = ((profile, 12, "+x", ((12.75, 7, 0), (16, 9, 2)), (U, U / 2)),
             (profile[::-1], 8, "-x", ((4, 7, 0), (7.25, 9, 2)), (U / 2, U)))
    statuses = []
    for cells, feed, direction, box, adjacent in cases:
        sim, grid = _msl_screen(cells, "x", feed, direction, box)
        index = grid.index_of("x", feed * U)
        np.testing.assert_array_equal(grid.cells("x")[index - 1:index + 1], adjacent)
        record = _screen_record(sim, grid)
        statuses.append(record.status)
    # A +side-only lookup reports insufficient for +x and satisfied for -x.
    assert statuses == ["insufficient", "insufficient"]


@pytest.mark.parametrize("bands", ["both_edges_fine", "one_edge_fine", "centre_fine"])
def test_msl_width_verdict_follows_strip_edges(bands):
    if bands == "both_edges_fine":
        widths = [1] * 6 + [.5] + [.25] * 4 + [.5] * 2 + [.25] * 4 + [.5] + [1] * 10
        expected, edge_sizes, centre_size = "insufficient", (U / 4, U / 4), U / 2
    elif bands == "one_edge_fine":
        widths = [1] * 6 + [.5] + [.25] * 4 + [.5] * 5 + [1] * 10
        expected, edge_sizes, centre_size = "insufficient", (U / 4, U / 2), U / 2
    else:
        widths = [1] * 6 + [.5] * 3 + [.25] * 4 + [.5] * 3 + [1] * 10
        expected, edge_sizes, centre_size = "satisfied", (U / 2, U / 2), U / 4
    # The measured trace spans y=7U..9U. The candidate's width differs by
    # 0.375U, between a fine edge's 0.25U and a coarse edge's 0.5U.
    sim, grid = _msl_screen(np.array(widths) * U, "y", 8, "+x",
                           ((7, 6.8125, 0), (9, 9.1875, 2)))
    for coordinate, size in zip((7, 9, 8), (*edge_sizes, centre_size)):
        index = grid.index_of("y", coordinate * U)
        np.testing.assert_array_equal(grid.cells("y")[index - 1:index + 1], [size, size])
    record = _screen_record(sim, grid)
    assert record.status == expected
    assert (record.reflector is not None) == (expected == "insufficient")


@pytest.mark.parametrize("axis", list("xyz"))
@pytest.mark.parametrize("fine_plane", ["plus", "minus"])
def test_waveguide_different_reference_cells_choose_min(axis, fine_plane):
    widths = ([U] * 4 + [U / 2] * 16 + [U] * 8)
    if fine_plane == "minus":
        widths = widths[::-1]
    sim = Simulation(freq_max=100e9, domain=(20 * U,) * 3, dx=U,
                     boundary="cpml", cpml_layers=2,
                     **{f"d{axis}_profile": np.array(widths)})
    for sign, source, reference in (("+", 6, 6), ("-", 14, 14.375)):
        sim.add_waveguide_port(source * U, direction=sign + axis,
                              reference_plane=reference * U, freqs=np.array([80e9]),
                              ref_offset=1, probe_offset=2)
    grid = sim._build_realized_grid()
    expected_sizes = (U / 2, U) if fine_plane == "plus" else (U, U / 2)
    for coordinate, size in zip((6, 14.375), expected_sizes):
        index = grid.index_of(axis, coordinate * U)
        np.testing.assert_array_equal(grid.cells(axis)[index - 1:index + 1], [size, size])
    report = sim.preflight_sparameters(calculator="waveguide", include_general=False,
                                      normalize=True)
    assert not report.by_code("waveguide_setup_audit_skipped")
    hits = [issue for issue in report.by_code("port_index_mirror_asymmetry")
            if "reference plane (" in str(issue)]
    # Residual 0.375U exceeds min/2=0.25U but not max/2=0.5U. Neither
    # always choosing the plus plane nor always choosing the minus can pass.
    assert len(hits) == 1
    assert f"tolerance {U / 4 * 1e3:.3g} mm" in str(hits[0])


@pytest.mark.parametrize("direction", ["+x", "-x"])
def test_msl_coarse_local_cell_intentionally_removes_finding(direction):
    profile = np.array([U] * 4 + [2 * U] * 6 + [U] * 4)
    feed, box = ((8, ((9.5, 7, 0), (12, 9, 2))) if direction == "+x"
                 else (12, ((8, 7, 0), (10.5, 9, 2))))
    sim, grid = _msl_screen(profile, "x", feed, direction, box)
    index = grid.index_of("x", feed * U)
    np.testing.assert_array_equal(grid.cells("x")[index - 1:index + 1], [2 * U, 2 * U])
    assert grid.boundary_cell("x", "lo") == U
    record = _screen_record(sim, grid)
    # Intended: the 1.5U gap lies within the local 2U rasterization ambiguity.
    # The old U boundary cell counted this conductor; it has no metric role
    # at this feed. No boundary-cell cap/floor is applied to the local scale.
    assert record.status == "satisfied"
    assert record.reflector is None


def test_msl_traced_width_axis_retains_verdict():
    import jax
    import jax.numpy as jnp
    from rfx.core.jax_utils import is_tracer

    records = []

    def assess(profile):
        sim, grid = _msl_screen(profile, "y", 8, "+x", ((12, 7, 0), (14, 9, 2)))
        assert is_tracer(grid.cells("y"))
        records.append(_screen_record(sim, grid))
        return jnp.sum(profile)

    jax.grad(assess)(jnp.full(20, U))
    assert len(records) == 1
    assert records[0].status == "insufficient"
    assert records[0].reflector is not None
    assert records[0].note is None


def test_local_cell_uses_only_existing_adjacent_cells():
    from rfx.nonuniform import make_nonuniform_grid
    from rfx.preflight._common import local_cell

    # Asymmetric z faces: the first node must not wrap to the last cell.
    grid = make_nonuniform_grid((4 * U, 4 * U), np.array([4 * U, U, 2 * U]),
                                U, cpml_layers=0)
    for coordinate, expected in ((0, 4 * U), (4, U), (5, U), (7, 2 * U)):
        assert local_cell(grid, "z", coordinate * U) == expected
    # The final entry provides a node, not an adjacent physical cell.
    # Alter only that sentinel, leaving every physical node unchanged.
    grid.dz_f64[-1] = U / 8
    assert local_cell(grid, "z", 7 * U) == 2 * U


@pytest.mark.parametrize("dx", [U, 1e-3, 0.0007])
def test_local_cell_uniform_float_bits(dx):
    from rfx.preflight._common import local_cell

    grid = Simulation(freq_max=20e9, domain=(20 * dx,) * 3, dx=dx,
                      boundary="cpml", cpml_layers=2)._build_grid()
    for axis in "xyz":
        for index in (0, 8, len(grid.cells(axis)) - 1):
            cell = local_cell(grid, axis, grid.node_of(axis, index))
            assert cell.hex() == float(grid.dx).hex()
