"""The gradient's temporary memory is the phase factor exp(j k r̂·r′)
formed for one face in one direction chunk, plus currents and output work.

Synthetic surface records isolate this physical term from the FDTD tape.
The bounds below come from array shapes, independently of transform helpers.
"""

import jax
import jax.numpy as jnp
import numpy as np

from rfx.farfield import NTFFBox, NTFFData
from rfx.grid import Grid
from rfx.measurement.farfield import compute_far_field_jax


def _fixture(side):
    rng = np.random.default_rng(1482)
    shape = (3, side, side, 4)
    faces = tuple(jnp.asarray((rng.normal(size=shape) + 1j * rng.normal(size=shape))
                             .astype(np.complex64)) for _ in range(6))
    grid = Grid(freq_max=40e9, domain=((side + 2) * 0.0005,) * 3,
                dx=0.0005, cpml_layers=0)
    box = NTFFBox(1, side + 1, 1, side + 1, 1, side + 1,
                  jnp.asarray([20e9, 30e9, 40e9], dtype=jnp.float32))
    return faces, box, grid


def test_farfield_grad_memory_bound():
    with jax.default_device(jax.devices("cpu")[0]):
        faces, box, grid = _fixture(16)
        theta = jnp.linspace(0, jnp.pi, 48, dtype=jnp.float32)
        phi = jnp.linspace(0, 2 * jnp.pi, 72, endpoint=False, dtype=jnp.float32)
        nf, n1, n2, _ = faces[0].shape
        nd = theta.size * phi.size
        nc = n1 * n2
        cb = faces[0].dtype.itemsize
        rb = theta.dtype.itemsize
        # Six components of J/M cotangents per face, four input-component
        # cotangents per face; N/L and their cotangents (2 * 6 components),
        # two E outputs and their cotangents (2 * 2 components), direction
        # unit vectors (3 * 3 real components), angles and frequencies.
        small = (len(faces) * nf * nc * (6 + 4) * cb
                 + nf * nd * (2 * 6 + 2 * 2) * cb
                 + nd * 9 * rb + (theta.size + phi.size + nf) * rb)
        one_face = nf * nd * nc * cb
        temp = {}
        for chunks in (1, 16):
            # The public budget's established rule counts 16 bytes per phase
            # entry; do not import the implementation's sizing helpers.
            budget = (theta.size // chunks) * nf * phi.size * nc * 16

            def objective(values):
                ff = compute_far_field_jax(NTFFData(*values), box, grid, theta, phi,
                                           max_phase_bytes=budget)
                return jnp.sum(jnp.abs(ff.E_theta)**2 + jnp.abs(ff.E_phi)**2)

            compiled = jax.jit(jax.grad(objective)).lower(faces).compile()
            analysis = compiled.memory_analysis()
            assert analysis is not None
            temp[chunks] = analysis.temp_size_in_bytes
        print(f"temp={temp}, one_face={one_face}, analytic_small={small}")
        assert temp[16] <= 0.25 * temp[1], temp
        assert temp[1] <= 1.25 * (one_face + small), (temp, one_face, small)


def test_farfield_transform_compositions():
    with jax.default_device(jax.devices("cpu")[0]):
        faces, box, grid = _fixture(3)
        # Seven rows with a three-row chunk also exercises padding and its
        # zero cotangent. Frequencies and both angles are traced under jit.
        theta = jnp.linspace(0.1, 3.0, 7, dtype=jnp.float32)
        phi = jnp.linspace(0.2, 6.0, 5, dtype=jnp.float32)
        budget = 3 * 3 * phi.size * 9 * 16

        def fields(scale, frequencies, th, ph):
            data = NTFFData(*(scale * face for face in faces))
            ff = compute_far_field_jax(data, box._replace(freqs=frequencies), grid,
                                       th, ph, max_phase_bytes=budget)
            return jnp.stack((ff.E_theta, ff.E_phi))

        def scaled(scale):
            return fields(scale, box.freqs, theta, phi)

        reference = scaled(jnp.float32(1))
        compiled = jax.jit(fields)(jnp.float32(1), box.freqs, theta, phi)
        np.testing.assert_allclose(compiled, reference, rtol=2e-5, atol=1e-7)
        batched = jax.vmap(scaled)(jnp.asarray([0.5, 2.0], dtype=jnp.float32))
        np.testing.assert_allclose(batched, reference[None] * jnp.asarray([0.5, 2.0])[:, None, None, None, None],
                                   rtol=2e-5, atol=1e-7)
        _, tangent = jax.jvp(scaled, (jnp.float32(1),), (jnp.float32(1),))
        np.testing.assert_allclose(tangent, reference, rtol=2e-5, atol=1e-7)

        def objective(scale):
            return jnp.sum(jnp.abs(scaled(scale))**2)

        expected = 2 * objective(jnp.float32(1))
        for fn in (objective, jax.checkpoint(objective)):
            gradient = jax.grad(fn)(jnp.float32(1))
            np.testing.assert_allclose(gradient, expected, rtol=2e-5, atol=1e-7)
