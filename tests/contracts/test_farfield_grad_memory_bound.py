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
from rfx.measurement.farfield import _stacked_face_inputs, compute_far_field_jax


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



# A box of 8 x 12 x 16 cells: the x faces have 12 x 16 cells, the y faces
# 8 x 16 and the z faces 8 x 12, three distinct face shapes.
_BOX_CELLS = (8, 12, 16)


def _box_fixture():
    rng = np.random.default_rng(1483)
    nx, ny, nz = _BOX_CELLS
    spans = {"x": (ny, nz), "y": (nx, nz), "z": (nx, ny)}
    faces = []
    for axis in "xxyyzz":
        shape = (3, *spans[axis], 4)
        faces.append(jnp.asarray((rng.normal(size=shape) + 1j * rng.normal(size=shape))
                                 .astype(np.complex64)))
    grid = Grid(freq_max=40e9, domain=tuple((n + 2) * 0.0005 for n in _BOX_CELLS),
                dx=0.0005, cpml_layers=0)
    box = NTFFBox(1, nx + 1, 1, ny + 1, 1, nz + 1,
                  jnp.asarray([20e9, 30e9, 40e9], dtype=jnp.float32))
    return tuple(faces), box, grid


def test_farfield_grad_memory_bound_non_cubic_box():
    """With faces of unequal size, the face scan pads every face to the
    largest one; the temp memory is then the largest face's factor."""
    with jax.default_device(jax.devices("cpu")[0]):
        faces, box, grid = _box_fixture()
        theta = jnp.linspace(0, jnp.pi, 48, dtype=jnp.float32)
        phi = jnp.linspace(0, 2 * jnp.pi, 72, endpoint=False, dtype=jnp.float32)
        nf = faces[0].shape[0]
        nd = theta.size * phi.size
        cells = [f.shape[1] * f.shape[2] for f in faces]
        nc_max = max(cells)
        n_faces = len(faces)
        cb = faces[0].dtype.itemsize
        rb = theta.dtype.itemsize
        # Four input-component cotangents per face at its own size; the
        # stacked padded scan inputs at the largest size: J and M (six
        # components) and their cotangents (six), positions (three real)
        # and weights (one real); N/L and their cotangents (2 * 6
        # components), two E outputs and their cotangents (2 * 2), direction
        # unit vectors (3 * 3 real), angles and frequencies.
        small = (sum(nf * nc * 4 * cb for nc in cells)
                 + n_faces * nf * nc_max * (6 + 6) * cb
                 + n_faces * nc_max * (3 + 1) * rb
                 + nf * nd * (2 * 6 + 2 * 2) * cb
                 + nd * 9 * rb + (theta.size + phi.size + nf) * rb)
        largest_face = nf * nd * nc_max * cb
        budget = theta.size * nf * phi.size * nc_max * 16   # c = 1

        def objective(values):
            ff = compute_far_field_jax(NTFFData(*values), box, grid, theta, phi,
                                       max_phase_bytes=budget)
            return jnp.sum(jnp.abs(ff.E_theta)**2 + jnp.abs(ff.E_phi)**2)

        analysis = jax.jit(jax.grad(objective)).lower(faces).compile().memory_analysis()
        assert analysis is not None
        temp = analysis.temp_size_in_bytes
        print(f"temp={temp}, largest_face={largest_face}, analytic_small={small}")
        assert temp <= 1.25 * (largest_face + small), (temp, largest_face, small)


def test_stacked_face_inputs_are_finite_and_padded_with_zero():
    """A padded cell must add exactly zero: zero weight, zero currents and a
    finite position (0.0), so no NaN or Inf enters the contraction."""
    with jax.default_device(jax.devices("cpu")[0]):
        faces, box, grid = _box_fixture()
        cells = [f.shape[1] * f.shape[2] for f in faces]
        nc_max = max(cells)
        nf = faces[0].shape[0]
        J, M, pos, w = _stacked_face_inputs(NTFFData(*faces), box, grid, jnp.float32)
        assert J.shape == (len(faces), nf, nc_max, 3)
        assert M.shape == (len(faces), nf, nc_max, 3)
        assert pos.shape == (len(faces), nc_max, 3)
        assert w.shape == (len(faces), nc_max)
        for name, array in (("positions", pos), ("weights", w),
                            ("J", J), ("M", M)):
            assert bool(jnp.all(jnp.isfinite(array))), f"stacked {name} not finite"
        for index, nc in enumerate(cells):
            assert bool(jnp.all(w[index, nc:] == 0)), f"face {index}: padded weight"
            assert bool(jnp.all(J[index, :, nc:] == 0)), f"face {index}: padded J"
            assert bool(jnp.all(M[index, :, nc:] == 0)), f"face {index}: padded M"
            assert bool(jnp.all(pos[index, nc:] == 0)), f"face {index}: padded position"
            assert bool(jnp.all(w[index, :nc] > 0)), f"face {index}: real cell weight"


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
