"""Finite-window geometry and physical-cell exclusion, without an RF solve."""
from types import SimpleNamespace
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.grid import Grid
from rfx.probes.flux_region import resolve_flux_axis, resolve_flux_region
from rfx.runners.uniform import build_flux_monitor_cfgs


def _sim(**kwargs):
    return Simulation(freq_max=1e9, domain=(.020, .020, .020), dx=.001,
                      cpml_layers=8, boundary="cpml", **kwargs)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_oversized_uniform_region_excludes_cpml_and_bounding_node(axis):
    sim = _sim()
    sim.add_flux_monitor(axis=axis, coordinate=.010, size=(.060, .060), name="oversized")
    grid = sim._build_grid()
    assert grid.shape == (37, 37, 37)
    with pytest.warns(UserWarning, match="CLAMPED"):
        monitor, = build_flux_monitor_cfgs(sim, grid, 1)
    assert (monitor.lo1, monitor.hi1, monitor.lo2, monitor.hi2) == (8, 28, 8, 28)
    region = resolve_flux_region(grid, sim._flux_monitors[0], sim._domain, warn=False)
    assert region["realized_bounds_m"] == [[0., .020], [0., .020]]
    assert region["clamped_ends"] == [[True, True], [True, True]]
    # A constant physical density plus a deliberately large artificial value
    # in the absorber exposes accidental integration of even one pad cell.
    density = np.full((37, 37), 1000.)
    density[8:28, 8:28] = 2.
    selected = density[monitor.lo1:monitor.hi1, monitor.lo2:monitor.hi2]
    assert selected.sum() * grid.dx**2 == pytest.approx(2 * .020**2)


@pytest.mark.parametrize("center", [(-.001, .010), (.021, .010)])
def test_partial_uniform_clamp_is_warned_even_below_half_a_cell(center):
    sim = _sim()
    # Tiny endpoint overflows used to be labelled indistinguishable from
    # snapping on the NU lane. The physical request identifies the clamp.
    c = (.0009 if center[0] < 0 else .0191, center[1])
    sim.add_flux_monitor(axis="x", coordinate=.010, size=(.002, .004), center=c)
    with pytest.warns(UserWarning, match="CLAMPED"):
        monitor, = build_flux_monitor_cfgs(sim, sim._build_grid(), 1)
    assert monitor.lo1 >= 8 and monitor.hi1 <= 28


@pytest.mark.parametrize("scale", [1e-6, 1., 1e6])
@pytest.mark.parametrize("side", ["low", "high"])
def test_boundary_roundoff_guard_scales_with_coordinate_arithmetic(scale, side):
    edges = np.linspace(0., .020 * scale, 21)
    if side == "high":
        lo, hi = .015 * scale, .020 * scale
        center, size = (lo + hi) / 2, hi - lo
    else:
        # The endpoint cancellation is near zero, but its uncertainty comes
        # from the nonzero centre and extent operands.
        size = .010 * scale
        center = np.nextafter(size / 2, -np.inf)
    record = resolve_flux_axis(edges, 8, center, size)
    assert not record["clamped_low"] and not record["clamped_high"]
    shift = (-1 if side == "low" else 1) * 1e-14 * scale
    outside = resolve_flux_axis(edges, 8, center + shift, size)
    assert outside[f"clamped_{side}"]
    assert outside["cell_slice"] == record["cell_slice"]


@pytest.mark.parametrize("center", [(-.010, .010), (.030, .010)])
def test_wholly_outside_window_is_rejected_instead_of_wrapping(center):
    sim = _sim()
    sim.add_flux_monitor(axis="x", coordinate=.010, size=(.002, .004), center=center)
    with pytest.raises(ValueError, match="no interior cell"):
        build_flux_monitor_cfgs(sim, sim._build_grid(), 1)


@pytest.mark.parametrize("key,value", [
    ("size", (.001,)), ("size", (.001, .002, .003)),
    ("size", ((.001, .002),)), ("size", (0., .002)),
    ("size", (-.001, .002)), ("size", (np.inf, .002)),
    ("center", (.001,)), ("center", (.001, .002, .003)),
    ("center", (np.nan, .001)), ("center", (1j, .001)),
])
def test_bad_tangential_inputs_are_rejected_before_registration(key, value):
    sim = _sim()
    kwargs = dict(size=(.004, .004), center=(.010, .010))
    kwargs[key] = value
    with pytest.raises(ValueError, match=key):
        sim.add_flux_monitor(axis="x", coordinate=.010, **kwargs)
    assert not sim._flux_monitors


def test_direct_entry_cannot_bypass_two_coordinate_validation():
    sim = _sim()
    entry = SimpleNamespace(axis="x", coordinate=.010, size=(.004, .004),
                            center=(.010, .010, .010), name="external", freqs=jnp.array([1e9]))
    with pytest.raises(ValueError, match="exactly two"):
        build_flux_monitor_cfgs(sim, sim._build_grid(), 1, entries=[entry])


def test_interior_rounding_and_legacy_full_plane_are_preserved():
    sim = _sim()
    sim.add_flux_monitor(axis="x", coordinate=.010, size=(.004, .008), center=(.010, .010))
    sim.add_flux_monitor(axis="x", coordinate=.010)
    with warnings.catch_warnings(record=True) as caught:
        finite, full = build_flux_monitor_cfgs(sim, sim._build_grid(), 1)
    assert not [w for w in caught if "flux monitor" in str(w.message).lower()]
    assert (finite.lo1, finite.hi1, finite.lo2, finite.hi2) == (16, 20, 14, 22)
    assert (full.lo1, full.hi1, full.lo2, full.hi2) == (0, 37, 0, 37)


def test_two_dimensional_collapsed_axis_is_one_integration_cell():
    sim = Simulation(freq_max=1e9, domain=(.020, .020, .001), dx=.001,
                     cpml_layers=8, mode="2d_tmz", boundary="cpml")
    sim.add_flux_monitor(axis="x", coordinate=.010, size=(.010, .001), center=(.010, .0005))
    monitor, = build_flux_monitor_cfgs(sim, sim._build_grid(), 1)
    assert (monitor.lo2, monitor.hi2) == (0, 1)


def test_finite_flux_window_keeps_nonzero_field_gradient():
    from rfx.probes.probes import flux_spectrum

    sim = _sim()
    sim.add_flux_monitor(axis="x", coordinate=.010, freqs=jnp.array([1e9]), size=(.060, .060))
    with pytest.warns(UserWarning, match="CLAMPED"):
        monitor, = build_flux_monitor_cfgs(sim, sim._build_grid(), 1)

    def power(amplitude):
        shape = monitor.e1_dft.shape
        e = jnp.ones(shape, dtype=monitor.e1_dft.dtype) * amplitude
        h = jnp.ones(shape, dtype=monitor.h2_dft.dtype)
        current = monitor._replace(e1_dft=e, h2_dft=h)
        return jnp.sum(flux_spectrum(current))

    derivative = jax.jit(jax.grad(power))(jnp.float32(2.))
    assert np.isfinite(derivative) and derivative != 0
    # flux_spectrum's existing DFT convention has no time-average 1/2.
    np.testing.assert_allclose(derivative, .020**2, rtol=2e-6)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_graded_region_uses_all_actual_interior_cells_on_each_plane(axis):
    profile = np.array([.001] * 4 + [.00075] * 8 + [.00125] * 8 + [.001] * 4)
    length = float(profile.sum())
    sim = Simulation(freq_max=1e9, domain=(length,) * 3, dx=.001,
                     cpml_layers=8, boundary="cpml", dx_profile=profile,
                     dy_profile=profile, dz_profile=profile)
    sim.add_flux_monitor(axis=axis, coordinate=length / 2,
                         size=(3 * length, 3 * length), name="graded")
    grid = sim._build_nonuniform_grid()
    with pytest.warns(UserWarning, match="CLAMPED"):
        region = resolve_flux_region(grid, sim._flux_monitors[0], sim._domain)
    assert region["cell_slices"] == [[8, 8 + len(profile)], [8, 8 + len(profile)]]
    np.testing.assert_allclose(region["realized_bounds_m"], [[0., length], [0., length]],
                               rtol=0, atol=1e-14)
    assert region["tangential_axes"] == [a for a in "xyz" if a != axis]
