"""The uniform step receives the four-incident-cell design-box coefficients."""

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import MaterialArrays
from rfx.grid import Grid
import rfx.simulation as simulation


@pytest.mark.parametrize("design_sigma", [False, True], ids=["background-sigma", "design-sigma"])
def test_design_box_step_coefficients(monkeypatch, design_sigma):
    faces = {f"{axis}_{side}" for axis in "xyz" for side in ("lo", "hi")}
    grid = Grid(freq_max=10e9, domain=(.014, .013, .012), dx=.001,
                cpml_layers=0, pec_faces=faces)
    rng = np.random.default_rng(1179)
    background_eps = rng.uniform(1., 4., grid.shape).astype(np.float32)
    background_sigma = rng.uniform(0., .05, grid.shape).astype(np.float32)
    bounds = (4, 9, 3, 8, 5, 9)
    box = (slice(4, 9), slice(3, 8), slice(5, 9))
    eps = rng.uniform(1., 6., (5, 5, 4)).astype(np.float32)
    sigma = rng.uniform(0., .2, (5, 5, 4)).astype(np.float32) if design_sigma else None
    materials = MaterialArrays(jnp.asarray(background_eps), jnp.asarray(background_sigma),
                               jnp.ones(grid.shape, dtype=jnp.float32))
    seen = []

    class ContextCaptured(Exception):
        pass

    def capture(ctx, *args, **kwargs):
        seen.append(ctx)
        raise ContextCaptured

    monkeypatch.setattr(simulation, "make_core_step", capture)
    # Stop after run() has built the exact context handed to its step.
    with pytest.raises(ContextCaptured):
        simulation.run(
            grid, materials, n_steps=1, boundary="pec", field_dtype=jnp.float32,
            design_box=simulation.DesignBoxSpec(bounds, jnp.asarray(eps),
                                               None if sigma is None else jnp.asarray(sigma)))
    assert len(seen) == 1
    ctx = seen[0]
    assert ctx.use_design_box
    assert ctx.pec_faces_frozen == frozenset(faces)
    assert not ctx.use_cpml and not ctx.use_upml
    assert ctx.dt == grid.dt
    coefficients = ctx.design_box
    assert coefficients.bounds == (4, 10, 3, 9, 5, 10)

    # Independent float64 reference on the complete cell material arrays.
    m_eps = background_eps.astype(np.float64)
    m_eps[box] = eps
    m_sig = background_sigma.astype(np.float64)
    if sigma is not None:
        m_sig[box] = sigma
    indices = np.meshgrid(np.arange(4, 10), np.arange(3, 9), np.arange(5, 10),
                          indexing="ij")
    eps0 = 8.8541878128e-12
    for c, transverse in enumerate(((1, 2), (0, 2), (0, 1))):
        e = np.zeros((6, 6, 5), dtype=np.float64)
        s = np.zeros_like(e)
        for offset1, offset2 in ((0, 0), (1, 0), (0, 1), (1, 1)):
            neighbour = [index.copy() for index in indices]
            neighbour[transverse[0]] -= offset1
            neighbour[transverse[1]] -= offset2
            e += m_eps[tuple(neighbour)] / 4.
            s += m_sig[tuple(neighbour)] / 4.
        loss = s * ctx.dt / (2. * eps0 * e)
        ca = (1. - loss) / (1. + loss)
        cb = (ctx.dt / (eps0 * e)) / (1. + loss)
        for name, actual, expected in (("Ca", coefficients.ca[c], ca),
                                       ("Cb", coefficients.cb[c], cb)):
            actual = np.asarray(actual, dtype=np.float64)
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0.,
                                       err_msg=f"component {'xyz'[c]} {name}")
