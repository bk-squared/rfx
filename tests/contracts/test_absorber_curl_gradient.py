"""Judge 4: float32 conductivity sensitivity through the lossy absorber."""
import json

import jax
import jax.numpy as jnp
import numpy as np

from tests.contracts._absorber_curl_scene import build


def test_sigma_gradient_matches_central_difference(record_property):
    sim = build('plain', 'x_lo')
    grid = sim._build_grid()
    materials, *_ = sim._assemble_materials(grid)
    mask = jnp.asarray(materials.sigma != 0, dtype=jnp.float32)
    sigma = jnp.max(materials.sigma).astype(jnp.float32)

    def objective(value):
        out = sim.forward(sigma_override=mask * value, n_steps=800,
                          checkpoint=False, skip_preflight=True)
        return jnp.sum(out.time_series ** 2)

    value, gradient = jax.value_and_grad(objective)(sigma)
    step = jnp.float32(1e-2) * sigma
    fd = (objective(sigma + step) - objective(sigma - step)) / (2 * step)
    relative = float(jnp.abs(gradient - fd) / jnp.maximum(jnp.abs(gradient), jnp.abs(fd)))
    values = dict(objective=float(value), sigma=float(sigma), gradient=float(gradient),
                  finite_difference=float(fd), relative_difference=relative)
    record_property('judge4', json.dumps(values))
    print(json.dumps(values))
    assert np.isfinite(gradient), f'nonfinite conductivity gradient: {values}'
    assert relative <= 1e-2, f'conductivity gradient/FD relative error: {values}'
