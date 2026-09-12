"""Electric-substep work and material AD of the existing MSL launch.

These are prescribed-field checks of the actual load/source/E update, not
propagating-mode, full-run, or RF-accuracy certificates. The launch shape and
resistance are fixed while the electric material coefficient is varied.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import EPS_0, MaterialArrays, init_state, update_e
from rfx.grid import Grid
from rfx.sources.msl_port import (
    MSLPort,
    make_msl_port_sources,
    setup_msl_port,
)
try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64


@pytest.mark.parametrize("direction", ["+x", "-x", "+y", "-y"])
@pytest.mark.parametrize("shaped", [False, True], ids=["uniform", "laplace_shape"])
@pytest.mark.parametrize("x64", [False, True], ids=["float32", "float64"])
def test_source_load_work_and_material_gradient(direction, shaped, x64):
    with enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        grid = Grid(freq_max=1e8, domain=(6., 6., 6.), dx=1.,
                    cpml_layers=0, cpml_axes="")
        along_x = direction.endswith("x")
        ip, iw = (0, 1) if along_x else (1, 0)
        cells = [(3, w, z) if along_x else (w, 3, z)
                 for w in (2, 3) for z in (2, 3)]
        indices = tuple(np.asarray(cells).T)
        profile_e = np.array([.2, .1, .5, .5]) if shaped else np.full(4, .5)
        profile = dict(
            ez_profile=profile_e.reshape(2, 2), cell_indices=cells,
            j_grid_lo=2, k_grid_lo=2, n_z_sub=2,
            prop_idx=ip, width_idx=iw, normal_idx=2,
            prop_axis=direction[-1], width_axis="y" if along_x else "x",
            normal_axis="z",
        ) if shaped else None
        resistance, u = 50., .8
        port = MSLPort(feed_x=3., y_lo=2., y_hi=3., z_lo=2., z_hi=4.,
                       direction=direction, impedance=resistance,
                       excitation=lambda t: jnp.asarray(u, dtype=dtype))
        base = MaterialArrays(
            eps_r=jnp.full(grid.shape, 2.25, dtype=dtype),
            sigma=jnp.zeros(grid.shape, dtype=dtype),
            mu_r=jnp.ones(grid.shape, dtype=dtype),
        )
        loaded = setup_msl_port(grid, port, base, mode_profile=profile)
        # Independent normalization uses the physical volumes, not a source
        # helper or the loaded array. Unequal profile entries refute sigma~e^2.
        volume = grid.dx**3
        norm = volume * np.sum(profile_e**2)
        expected_sigma = 1 / (resistance * norm)
        np.testing.assert_allclose(np.asarray(loaded.sigma)[indices], expected_sigma,
                                   rtol=8 * np.finfo(dtype).eps)
        assert np.count_nonzero(loaded.sigma) == len(cells)

        old = init_state(grid.shape, field_dtype=dtype)
        # A centre-voltage-preserving perturbation orthogonal to the shaped
        # Ez profile, plus transverse E, exercises the extra positive loss.
        fringe = np.array([.1, -.2, 0., 0.]) if shaped else np.array([.1, -.1, 0., 0.])
        values = np.stack((np.full(4, .3), np.full(4, -.2), profile_e + fringe))
        for name, value in zip(("ex", "ey", "ez"), values):
            old = old._replace(**{name: getattr(old, name).at[indices].set(value)})

        def advance(alpha):
            materials = loaded._replace(eps_r=base.eps_r * alpha)
            # H=0: the electric substep has no curl work, and no PEC/PML is
            # present. Source coefficients must retain the material tape.
            new = update_e(old, materials, grid.dt, grid.dx)
            sources = make_msl_port_sources(grid, port, materials, 1,
                                            mode_profile=profile)
            assert len(sources) == len(cells)
            assert {(s.i, s.j, s.k) for s in sources} == set(cells)
            for source in sources:
                assert source.component == "ez"
                new = new._replace(ez=new.ez.at[source.i, source.j, source.k].add(
                    source.waveform[0]))
            return jnp.stack([getattr(new, name)[indices] for name in ("ex", "ey", "ez")])

        new = np.asarray(advance(jnp.asarray(1., dtype=dtype)))
        old_values = np.stack([np.asarray(getattr(old, name))[indices]
                               for name in ("ex", "ey", "ez")]).astype(float)
        # Midpoint work identity is derived from the field equation, not the
        # source builder's Cb. Include all three electric components in loss.
        mid = (old_values + new) / 2
        storage_rate = volume * np.sum(2.25 * EPS_0 * (new**2 - old_values**2)) / (2 * grid.dt)
        loss = volume * expected_sigma * np.sum(mid**2)
        source_work = u * volume * np.dot(profile_e, mid[2])
        scale = max(abs(storage_rate), loss, abs(source_work))
        assert abs(storage_rate + loss - source_work) < 64 * np.finfo(dtype).eps * scale
        voltage = volume * np.dot(profile_e, mid[2]) / norm
        assert loss > voltage**2 / resistance

        # Independent scalar midpoint equation; no JAX/source/update helpers.
        # Double precision FD avoids using an f32 loss as its own referee.
        force = np.zeros_like(old_values)
        force[2] = profile_e * u

        def reference(alpha):
            epsilon = 2.25 * alpha * EPS_0
            return ((epsilon - expected_sigma * grid.dt / 2) * old_values
                    + grid.dt * force) / (epsilon + expected_sigma * grid.dt / 2)

        np.testing.assert_allclose(new, reference(1.), rtol=64 * np.finfo(dtype).eps,
                                   atol=64 * np.finfo(dtype).eps)

        def objective(alpha):
            return jnp.mean(advance(alpha)**2)

        alpha = jnp.asarray(1., dtype=dtype)
        value, gradient = jax.jit(jax.value_and_grad(objective))(alpha)
        assert value.dtype == dtype and gradient.dtype == dtype
        np.testing.assert_allclose(value, np.mean(reference(1.)**2),
                                   rtol=64 * np.finfo(dtype).eps)
        h = 1e-4
        g_fd = (np.mean(reference(1. + h)**2) - np.mean(reference(1. - h)**2)) / (2 * h)
        assert np.isfinite(gradient) and abs(gradient) > 1e-4
        np.testing.assert_allclose(gradient, g_fd, rtol=2e-5 if not x64 else 1e-7)
