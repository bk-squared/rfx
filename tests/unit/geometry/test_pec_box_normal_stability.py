"""A normal PEC capacitor must remain passive with the unchanged CPML order."""

import jax.numpy as jnp
import numpy as np

from rfx import Box, Simulation
from rfx.core.yee import EPS_0, MU_0
from rfx.geometry import smoothing
from rfx.simulation import SnapshotSpec


def test_off_grid_block_cpml_energy(monkeypatch):
    """After an Ez pulse leaves an off-grid PEC block, stored energy cannot grow.

    Measure the Yee leapfrog energy E(n)^2/xi + Z0^2 H(n-1/2)H(n+1/2),
    including every padded cell. The H cross-time product removes the
    staggered-time oscillation in the naive E^2 + H^2 diagnostic.
    """
    h = 1 / 1024
    sim = Simulation(freq_max=20e9, domain=(12*h,)*3, dx=h,
                     boundary='cpml', cpml_layers=4, precision='float32')
    sim.add(Box((6.3*h, 4.3*h, 4.3*h), (8.7*h, 7.7*h, 7.7*h)), material='pec')
    grid = sim.freeze_mesh()
    source_end = 40
    end_time = source_end * grid.dt
    def pulse(t):
        u = t / end_time
        return jnp.where((u > 0) & (u < 1), jnp.sin(2*jnp.pi*u)*jnp.sin(jnp.pi*u)**2, 0.)
    sim.add_source((3*h, 6*h, 6*h), 'ez', waveform=pulse, amplitude_kind='field')
    # The last emitted wave can traverse the complete padded diagonal.
    departure = end_time + np.linalg.norm(np.asarray(grid.shape)*h) / 299792458.
    steps = int(np.ceil(4*departure/grid.dt)) + 1
    captured = []
    original = smoothing.compute_inv_eps_tensor_diag
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(tuple(np.asarray(a, dtype=float) for a in result))
        return result
    monkeypatch.setattr(smoothing, 'compute_inv_eps_tensor_diag', capture)
    result = sim.run(n_steps=steps, subpixel_smoothing='kottke_pec', skip_preflight=True,
                     snapshot=SnapshotSpec(interval=1, components=('ex', 'ey', 'ez', 'hx', 'hy', 'hz')))
    assert len(captured) == 1
    energy = np.zeros(steps - 1)
    for axis, component in enumerate(('ex', 'ey', 'ez')):
        field = np.asarray(result.snapshots[component][:-1], dtype=float)
        xi = captured[0][axis]
        electric = np.divide(field**2, xi, out=np.zeros_like(field), where=xi > 0)
        energy += np.sum(electric, axis=(1, 2, 3))
    for component in ('hx', 'hy', 'hz'):
        field = np.asarray(result.snapshots[component], dtype=float)
        energy += (MU_0/EPS_0)*np.sum(field[:-1]*field[1:], axis=(1, 2, 3))
    energy *= .5*EPS_0*h**3
    after = energy[source_end - 1:]
    assert np.all(np.isfinite(energy)) and after[0] > 0
    growth = np.max(after - np.minimum.accumulate(after)) / after[0]
    print(f'B6: steps={steps}, duration/departure={steps*grid.dt/departure:.6f}, '
          f'energy_at_source_end={after[0]:.9g}, max_regrowth={growth:.9g}')
    assert growth <= 1e-6
