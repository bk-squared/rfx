"""A boundary-cell deformation reaches CPML through public forward()."""

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Simulation


H = 1 / 1024
DT = 1e-12


def _pulse(t):
    return jnp.exp(-((t / DT - 10) / 3) ** 2)


def _probe_energy(scale):
    # Concrete x profiles require their end cells to equal scalar dx.
    # Under AD keep dx nominal and deform both traced end cells together.
    # Explicit y/z profiles and pinned dt hold all other metrics fixed.
    traced = isinstance(scale, jax.core.Tracer)
    if traced:
        profile = jnp.full(8, H).at[0].set(H * scale).at[-1].set(H * scale)
    else:
        profile = np.full(8, H)
        profile[[0, -1]] = H * scale
    sim = Simulation(
        freq_max=30e9, domain=(8 * H, 6 * H, 6 * H),
        dx=H if traced else float(H * scale), dx_profile=profile,
        dy_profile=np.full(6, H), dz_profile=np.full(6, H),
        boundary="cpml", cpml_layers=3, dt=DT, dt_min_cell=.9 * H,
    )
    sim.add_source((3 * H, 3 * H, 3 * H), "ez", waveform=_pulse,
                   amplitude_kind="field")
    sim.add_probe((2 * H, 3 * H, 3 * H), "ez")
    result = sim.forward(n_steps=60, skip_preflight=True, checkpoint=False)
    return jnp.sum(result.time_series ** 2)


def test_cpml_boundary_cell_gradient_matches_concrete_difference():
    value, ad = jax.value_and_grad(_probe_energy)(jnp.float32(1))
    delta = .005
    fd = (float(_probe_energy(1 + delta))
          - float(_probe_energy(1 - delta))) / (2 * delta)
    assert np.isfinite(float(value)) and float(value) > 0
    assert abs(fd) > 1e-3
    np.testing.assert_allclose(float(ad), fd, rtol=5e-3, atol=1e-6)
