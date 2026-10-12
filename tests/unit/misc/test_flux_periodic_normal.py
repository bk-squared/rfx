"""A periodic seam samples the half-cell H planes on both sides."""
import jax.numpy as jnp
import numpy as np

from rfx import Simulation
from rfx.core.yee import init_state
from rfx.measurement.accumulators import flux, flux_metadata
from rfx.measurement.plan import build_measurement_plan
from rfx.probes.probes import update_flux_monitor
from rfx.runners.uniform import build_flux_monitor_cfgs


def test_periodic_normal_plane_uses_the_last_h_plane():
    sim = Simulation(20e9, (.004, .003, .002), dx=.001, cpml_layers=0,
                     boundary={'x': 'periodic', 'y': 'pec', 'z': 'pec'})
    sim.add_flux_monitor(axis='x', coordinate=0., freqs=np.array([0.]))
    grid = sim._build_realized_grid()
    mon, = build_flux_monitor_cfgs(sim, grid, 1)
    state = init_state(grid.shape)
    h = jnp.zeros(grid.shape).at[0].set(2.).at[-1].set(6.)
    state = state._replace(hy=h, hz=3*h, step=jnp.asarray(1))
    sampled = update_flux_monitor(mon, state, 1.)
    np.testing.assert_array_equal(sampled.h1_dft, 4.)
    np.testing.assert_array_equal(sampled.h2_dft, 12.)
    accs = [(mon.e1_dft, mon.e2_dft, mon.h1_dft, mon.h2_dft)]
    streamed, = flux(state, accs, flux_metadata([mon], window=True), 1., 0)
    np.testing.assert_array_equal(streamed[2], sampled.h1_dft)
    np.testing.assert_array_equal(streamed[3], sampled.h2_dft)
    owner, = build_measurement_plan(sim, grid, n_steps=1).owners
    for channel in owner.channels[2:]:
        assert {n.i for n in channel.nodes} == {0, grid.shape[0]-1}
        assert {n.weight for n in channel.nodes} == {.5}
