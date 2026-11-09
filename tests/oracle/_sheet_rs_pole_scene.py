"""Two resistive x walls in a vacuum TE101 box, with a fixed physical feed."""
import time

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Box, GaussianPulse, Simulation
from rfx import ringdown as rd
from rfx.model.conductors import realized_conductors


C0 = 299792458.0
MU0 = 4 * np.pi * 1e-7
F0 = 8e9
RS0 = 0.2334
SIGMA = 1.44e-5
RECORD_S = 12e-9
BINS = np.linspace(7e9, 9e9, 21)


def scene(h=1e-3, Rs=RS0):
    sim = Simulation(freq_max=15e9, domain=(.030 + 2*h, .012, .024),
                     dx=h, boundary="pec", precision="float64", snap="declared")
    sigma_bulk = np.pi * F0 * MU0 / Rs**2
    for x in (h, h + .030):
        sim.add_thin_conductor(Box((x, 0., 0.), (x, .012, .024)),
                               sigma_bulk=sigma_bulk, thickness=35e-6,
                               surface_impedance_f0=F0)
    sim.add_port(position=(h + .006, .003, .006), component="ey",
                 extent=.003, impedance=10e3,
                 waveform=GaussianPulse(f0=F0, bandwidth=.5))
    spec = rd.RingdownSpec(
        freq_max=15e9,
        identification_probes=(((h + .021, .003, .015), "ey"),
                               ((h + .009, .009, .021), "ey")))
    return sim, sim._build_grid(), spec


def measure(h=1e-3, record_s=RECORD_S):
    """One forward and one pullback; keep selection failures separate from slopes."""
    start = time.monotonic()
    sim, grid, _ = scene(h)
    built = realized_conductors(sim, grid)
    near, far = built.sheet_specs
    # Read the concrete production sheet planes, never the traced realization row.
    a = grid.node_of("x", far.plane) - grid.node_of("x", near.plane)
    b = grid.node_of("y", grid.shape[1] - 1) - grid.node_of("y", 0)
    d = grid.node_of("z", grid.shape[2] - 1) - grid.node_of("z", 0)
    target = C0 / 2 * np.sqrt(a**-2 + d**-2)
    n_steps = round(record_s / grid.dt)

    def pole(Rs):
        # The user's Rs forms sigma_bulk inside the differentiated scene build.
        traced_sim, traced_grid, spec = scene(h, Rs)
        result = traced_sim.forward(
            sigma_override=jnp.full(traced_grid.shape, SIGMA), n_steps=n_steps,
            port_s11_freqs=BINS, ringdown=spec, skip_preflight=True).ringdown
        return result.pole(target), result

    values, pullback, result = jax.vjp(pole, jnp.float64(RS0), has_aux=True)
    jax.block_until_ready(values)
    reason = rd.pole_selection_reason(result.poles, result.pole_mask,
                                      target, result._pole_window_s())
    assert reason is None, f"cannot judge: pole selection: {reason}"
    f, inv_q = map(float, values)
    assert np.isfinite(inv_q), f"cannot judge: inv_q={inv_q}"
    s, = pullback((0., 1.))
    s = float(s)
    omega = 2 * np.pi * f
    wavelength = C0 / f
    s_cf = wavelength**2 / (omega * MU0 * a**3)
    return dict(h=h, a=a, b=b, d=d, n_steps=n_steps, dt=grid.dt,
                f=f, inv_q=inv_q, s=s, s_cf=s_cf, r=s/s_cf - 1,
                wall_s=time.monotonic() - start, nodes=int(np.prod(grid.shape)))
