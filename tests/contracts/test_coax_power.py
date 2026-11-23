"""V1 unit-voltage closure and V2 accepted power against independent line flux."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.core.yee import init_materials
from rfx.grid import Grid
from rfx.measurement.coax_records import (
    coaxial_line_plane_voltage, coaxial_line_plane_voltage_jnp,
)
from rfx.measurement.ports import coax_owner
from rfx.model.coax_line import build_coax_line, coax_port_layout
from rfx.model.coax_stamp import _coaxial_tem_edge_profile
from rfx.sources.sources import GaussianPulse


@pytest.mark.parametrize('dx', [.00030, .00025, .0001875, .000125])
def test_unit_voltage_tem(dx):
    grid = Grid(10e9, (.012, .012, .003), dx=dx, cpml_layers=4)
    center = (.006, .006)
    a, b = .000635, .002055
    k = grid.pad_z_lo + 4
    line = build_coax_line(grid, init_materials(grid.shape), center_xy=center,
        pin_radius=a, outer_radius=b, z_lo_index=k-2, z_hi_index=k+2,
        ends=(), n_steps=1, field_scale=1.)
    port = SimpleNamespace(position=(*center, grid.node_of(2, k)), pin_radius=a)
    ex, ey = _coaxial_tem_edge_profile(grid, port, k, line.shell_inner_radius,
                                      line.pec_cells)
    kwargs = dict(center_xy=center, pin_radius=a, outer_radius=b,
                  pec_cell_mask=line.pec_cells, plane_index=k)
    numpy_voltage = coaxial_line_plane_voltage(grid, ex, ey, **kwargs)
    jax_voltage = coaxial_line_plane_voltage_jnp(grid, jnp.asarray(ex), **kwargs)
    owner = coax_owner(grid, port_id=0, plane_indices=(k,), center_xy=center,
        pin_radius=a, outer_radius=b, frequencies=(8e9,), pec_cell_mask=line.pec_cells)
    node_voltage = sum(ex[n.i, n.j] * n.weight for n in owner.channels[0].nodes)
    u = (np.arange(grid.nx) - grid.pad_x_lo) * grid.dx - center[0]
    selected = (u >= a) & (u <= b)
    j = grid.position_to_index((*center, 0.))[1]
    old = np.trapezoid(ex[selected, j], u[selected])
    print(f'V1 dx={dx:.9g}: old_R={old:.12g}, numpy={numpy_voltage.real:.12g}, '
          f'jax={float(jax_voltage):.12g}, owner={node_voltage:.12g}')
    np.testing.assert_allclose([numpy_voltage, jax_voltage, node_voltage], 1.,
                               rtol=0, atol=1e-6)


def test_instrument_fixture_unit_voltage_ratio():
    from tests._coax_msl_instrument_fixture import build_instrument_junction
    sim = build_instrument_junction()
    grid = sim._build_grid()
    port = sim._coaxial_ports[0]
    k = grid.pad_z_lo + 10
    line = build_coax_line(grid, init_materials(grid.shape), center_xy=port.position[:2],
        pin_radius=port.pin_radius, outer_radius=port.outer_radius,
        z_lo_index=k-2, z_hi_index=k+2, ends=(), n_steps=1, field_scale=1.)
    ex, ey = _coaxial_tem_edge_profile(grid, port, k, line.shell_inner_radius, line.pec_cells)
    from rfx.measurement.coax_records import coaxial_tem_reference_plane_vi_from_cartesian_plane
    u = (np.arange(grid.nx) - grid.pad_x_lo) * grid.dx
    v = (np.arange(grid.ny) - grid.pad_y_lo) * grid.dx
    old = coaxial_tem_reference_plane_vi_from_cartesian_plane(u, v, ex, ey, ex, ey,
        center_u_m=port.position[0], center_v_m=port.position[1],
        inner_radius=port.pin_radius, outer_radius=port.outer_radius).vi.voltage
    new = coaxial_line_plane_voltage(grid, ex, ey, center_xy=port.position[:2],
        pin_radius=port.pin_radius, outer_radius=port.outer_radius,
        pec_cell_mask=line.pec_cells, plane_index=k)
    print(f'coax-MSL instrument dx={grid.dx:.12g}: old_R={old}, new={new}')
    np.testing.assert_allclose(new, 1., rtol=0, atol=1e-6)


READ_BINS = np.array([7.5e9, 8e9, 8.5e9])


def power_scene():
    """A 7.5 mm monopole fed through a finite ground plate by a bottom coax."""
    sim = Simulation(freq_max=12e9, domain=(.018, .018, .036), dx=.0003,
                     boundary='cpml', cpml_layers=6, snap='declared')
    sim.add(Box((.0015, .0015, .0225), (.0165, .0165, .0225)), material='pec')
    sim.add_coaxial_port((.009, .009, .0225), face='bottom', pin_length=.0075,
                         waveform=GaussianPulse(f0=8e9, bandwidth=.8))
    grid = sim._build_grid()
    port = sim._coaxial_ports[0]
    pads = tuple(getattr(grid, f'pad_{a}_{s}') for a in 'xyz' for s in ('lo', 'hi'))
    layout = coax_port_layout(grid, port, absorber_pads=pads)
    plane = layout.plane_indices[len(layout.plane_indices)//2]
    sim.add_flux_monitor(axis='z', coordinate=grid.node_of(2, plane),
        freqs=READ_BINS, center=port.position[:2], size=(2*port.outer_radius,)*2,
        name='line_power')
    return sim


def check_power(ratio):
    np.testing.assert_allclose(ratio, 1., rtol=0, atol=.05)


@pytest.mark.slow_physics
def test_accepted_power_equals_line_flux():
    from rfx.measurement.probes import flux_spectrum
    result = power_scene().run(n_steps=4500, s_param_freqs=READ_BINS, skip_preflight=True)
    data = result.coax_port_metadata['coax_port:0:mode:0']
    fraction = 1 - np.abs(np.asarray(data['native_gamma']))**2
    accepted = np.abs(np.asarray(data['forward_voltage']))**2 * fraction / (2*data['z_tem_ohm'])
    flux = np.asarray(flux_spectrum(result.flux_monitors['line_power'], exact_f64=True))/2
    residual = np.asarray(data['recurrence_residual'])
    ratio = accepted/flux
    for frequency, value, remainder, share in zip(READ_BINS, ratio, residual, fraction):
        print(f'V2 f={frequency:.9g}: accepted/(flux/2)={value:.12g}, '
              f'residual={remainder:.12g}, accepted_fraction={share:.12g}')
    assert np.all(fraction >= .3)
    assert np.all(np.isfinite(residual))
    assert np.any(residual < .01)
    check_power(ratio[residual < .01])
