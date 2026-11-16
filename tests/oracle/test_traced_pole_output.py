"""Closed-form conductivity and permittivity sums in a synthetic closed box."""
import time

import jax
import jax.numpy as jnp
import numpy as np

from rfx import ringdown as rd
from rfx.core.yee import EPS_0
from rfx.grid import C0
from rfx.measurement.ringdown_resolution import completion_poles
from tests._x64_compat import enable_x64
from tests.oracle._traced_pole_scene import scene, SIGMA, RECORD_S, BINS


def test_closed_box_pole_derivative_sums(monkeypatch):
    with enable_x64():
        sim, grid, spec = scene()
        n = round(RECORD_S / grid.dt)
        a, _, d = [(size - 1) * grid.dx for size in grid.shape]
        target = C0 / 2 * np.sqrt(a ** -2 + d ** -2)
        eps, sigma = jnp.ones(grid.shape), jnp.full(grid.shape, SIGMA)
        models = []
        original = rd.identify

        def capture(*args, **kw):
            model = original(*args, **kw)
            if args[2] == 0:  # completion callback; exclude lazy report identification
                models.append(model)
            return model

        monkeypatch.setattr(rd, 'identify', capture)

        def both(e, s):
            result = sim.forward(eps_override=e, sigma_override=s, n_steps=n,
                                 port_s11_freqs=BINS, ringdown=spec, skip_preflight=True).ringdown
            return result.pole(target), result

        start = time.monotonic()
        values, pullback, result = jax.vjp(both, eps, sigma, has_aux=True)
        g_eps, _ = pullback((1., 0.))
        _, g_sigma = pullback((0., 1.))
        jax.block_until_ready((g_eps, g_sigma))
        print(f'oracle one forward two backward wall_s={time.monotonic() - start:.6f}')
        for p, mask, model in zip((result.poles, result.poles_long),
                                  (result.pole_mask, result.pole_mask_long), models[:2]):
            expected, complete = completion_poles(model, n, spec.unit_tol)
            np.testing.assert_array_equal(np.asarray(p)[np.asarray(mask) > 0], expected)
            np.testing.assert_array_equal(np.asarray(mask)[:len(expected)], np.where(complete, 1, 2))
            assert rd.pole_selection_reason(p, mask, target, result._pole_window_s()) is None
        delta = 1 / result._pole_window_s()
        reference = [p for p in result.report.poles
                     if p.f_hz > 0 and p.decay_per_s > 0 and abs(p.f_hz - target) <= delta]
        assert len(reference) == 1
        f = reference[0].f_hz
        omega = 2 * np.pi * f
        theta = omega * grid.dt / 2
        expected_sigma = 1 / (omega * EPS_0)
        expected_eps = -(f / 2) * np.tan(theta) / theta
        r_sigma = float(jnp.sum(g_sigma)) / expected_sigma - 1
        r_eps = float(jnp.sum(g_eps)) / expected_eps - 1
        print(f'f={values[0]}, inv_q={values[1]}, r_sigma={r_sigma}, r_eps={r_eps}')
        assert abs(r_sigma) <= 1e-2, f'B1: abs(r_sigma)={abs(r_sigma)} exceeds 0.01'
        assert abs(r_eps) <= 1e-2, f'B2: abs(r_eps)={abs(r_eps)} exceeds 0.01'
